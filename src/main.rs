use tokio::time::Duration;
use std::process;
use windows::{
    core::*, Win32::{System::Com::*, UI::{Accessibility::*, WindowsAndMessaging::*}, Foundation::HWND},
};
use clap::Parser;
use log::{error, info, warn};
use anyhow::{Result, Context, anyhow};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Interval of seconds for capturing captions
    #[arg(short = 'i', long, default_value_t = 1.0)]
    capture_interval: f64,   

    /// Interval of seconds for checking if Live Captions is running
    #[arg(short = 'c', long, default_value_t = 10)]
    check_interval: u64,
}

struct Engine {
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
    previous_text: String,
    displayed_text: String, // 用于跟踪显示在终端的文本
    cached_element: Option<IUIAutomationElement>, // 缓存主窗口元素
    cached_text_element: Option<IUIAutomationElement>, // 缓存文本元素
    last_window_handle: HWND, // 存储上次的窗口句柄
    consecutive_empty_captures: usize, // 连续空捕获的次数
    adaptive_interval: f64, // 自适应捕获间隔
    min_interval: f64, // 最小捕获间隔
    max_interval: f64, // 最大捕获间隔
    max_text_length: usize, // 存储文本的最大长度
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { CoUninitialize(); }
        info!("COM resources released");
    }
}

impl Engine {
    fn new() -> Result<Self> {
        unsafe { 
            // 修复错误：正确处理HRESULT
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
            if hr.is_err() {
                return Err(anyhow!("Failed to initialize Windows COM: {:?}", hr));
            }
        }

        let automation: IUIAutomation = unsafe { 
            CoCreateInstance(&CUIAutomation, None, CLSCTX_ALL)
                .map_err(|e| anyhow!("Failed to initialize Windows Accessibility API: {:?}", e))?
        };
        
        let condition = unsafe { 
            automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from("CaptionsTextBlock"))
                .map_err(|e| anyhow!("Failed to create property condition: {:?}", e))?
        };
        
        // 默认配置值
        let default_min_interval = 0.5; // 最小捕获间隔为0.5秒
        let default_max_interval = 3.0; // 最大捕获间隔为3秒
        let default_max_text_length = 10000; // 最大文本长度为10000字符
        
        Ok(Self {
            automation,
            condition,
            previous_text: String::new(),
            displayed_text: String::new(),
            cached_element: None,
            cached_text_element: None,
            last_window_handle: HWND(0),
            consecutive_empty_captures: 0,
            adaptive_interval: default_min_interval,
            min_interval: default_min_interval,
            max_interval: default_max_interval,
            max_text_length: default_max_text_length,
        })
    }

    // 优化的文本差异检测算法
    fn extract_new_text(&self, current: &str) -> String {
        if self.previous_text.is_empty() {
            return current.to_string();
        }
        
        // 使用更高效的算法查找共同前缀和新增内容
        let mut i = 0;
        let prev_bytes = self.previous_text.as_bytes();
        let curr_bytes = current.as_bytes();
        
        // 查找共同前缀长度
        while i < prev_bytes.len() && i < curr_bytes.len() && prev_bytes[i] == curr_bytes[i] {
            i += 1;
        }
        
        // 如果有新增内容，返回
        if i < curr_bytes.len() {
            let new_content = &current[i..];
            // 确保从完整单词或句子开始（避免截断单词）
            if i > 0 && !new_content.starts_with(char::is_whitespace) {
                // 找到第一个空格后的位置
                if let Some(pos) = new_content.find(char::is_whitespace) {
                    if pos + 1 < new_content.len() {
                        return new_content[pos+1..].to_string();
                    }
                }
            }
            return new_content.to_string();
        }
        
        // 如果当前文本完全不同于之前的文本，返回整个当前文本
        if i == 0 && current != self.previous_text {
            return current.to_string();
        }
        
        String::new()
    }

    async fn get_livecaptions(&mut self) -> Result<Option<String>> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        if window.0 == 0 {
            return Err(anyhow!("Live Captions window not found"));
        }
        
        // 检查窗口是否发生变化，只有需要时才重新获取元素
        let window_changed = window.0 != self.last_window_handle.0;
        if window_changed {
            info!("Window handle changed, refreshing UI elements");
            self.cached_element = None;
            self.cached_text_element = None;
            self.last_window_handle = window;
        }
        
        // 如果缓存无效，重新获取元素
        if self.cached_element.is_none() {
            self.cached_element = Some(unsafe { self.automation.ElementFromHandle(window) }
                .map_err(|e| anyhow!("Failed to get element from window handle: {:?}", e))?);
        }
        
        // 使用缓存的元素获取文本元素
        if self.cached_text_element.is_none() {
            if let Some(element) = &self.cached_element {
                self.cached_text_element = Some(unsafe { element.FindFirst(TreeScope_Descendants, &self.condition) }
                    .map_err(|e| anyhow!("Failed to find captions text element: {:?}", e))?);
            }
        }
        
        // 使用已缓存的文本元素获取文本
        let current_text = if let Some(text_element) = &self.cached_text_element {
            unsafe { text_element.CurrentName() }
                .map_err(|e| {
                    // 如果获取文本失败，可能是元素已失效，清除缓存以便下次重新获取
                    self.cached_element = None;
                    self.cached_text_element = None;
                    anyhow!("Failed to get text from element: {:?}", e)
                })?
                .to_string()
        } else {
            return Err(anyhow!("Text element not found"));
        };
        
        // 如果文本为空或与上次相同，返回None
        if current_text.is_empty() || current_text == self.previous_text {
            return Ok(None);
        }
        
        // 使用优化的文本差异检测算法提取新增内容
        let new_text = self.extract_new_text(&current_text);
        
        // 更新上一次的文本
        self.previous_text = current_text;
        
        if !new_text.is_empty() {
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
    
    // 添加重试逻辑
    async fn get_livecaptions_with_retry(&mut self, max_retries: usize) -> Result<Option<String>> {
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < max_retries {
            match self.get_livecaptions().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    warn!("Attempt {}/{} failed: {}", attempts, max_retries, e);
                    last_error = Some(e);
                    if attempts < max_retries {
                        // 指数退避
                        let backoff_ms = 100 * 2u64.pow(attempts as u32);
                        info!("Retrying in {} ms", backoff_ms);
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| anyhow!("Unknown error after retries")))
    }

    // 限制文本长度，防止内存无限增长
    fn limit_text_length(&mut self) {
        if self.displayed_text.len() > self.max_text_length {
            // 保留后半部分文本，丢弃前半部分
            let start_pos = self.displayed_text.len() - self.max_text_length / 2;
            // 找到一个合适的断句点（如空格或标点）
            if let Some(pos) = self.displayed_text[start_pos..].find(|c: char| c.is_whitespace() || c == '.' || c == ',' || c == '!' || c == '?') {
                self.displayed_text = self.displayed_text[(start_pos + pos + 1)..].to_string();
                info!("Text truncated to {} characters", self.displayed_text.len());
            } else {
                // 如果找不到合适的断句点，直接截断
                self.displayed_text = self.displayed_text[start_pos..].to_string();
                info!("Text truncated to {} characters", self.displayed_text.len());
            }
        }
    }

    async fn graceful_shutdown(&mut self) -> Result<()> {
        info!("Performing graceful shutdown");
        // 使用带重试的方法获取最终字幕
        match self.get_livecaptions_with_retry(3).await {
            Ok(Some(text)) => {
                // 将最后的文本追加到显示文本中
                self.displayed_text.push_str(&text);
                // 限制文本长度
                self.limit_text_length();
                info!("Final captions captured: {}", text);
                println!("\n> Final caption: {}", self.displayed_text);
            }
            Ok(None) => {
                info!("No new captions at shutdown");
                if !self.displayed_text.is_empty() {
                    println!("\n> Final caption: {}", self.displayed_text);
                }
            }
            Err(err) => {
                warn!("Could not capture final captions: {}", err);
                if !self.displayed_text.is_empty() {
                    println!("\n> Final caption: {}", self.displayed_text);
                }
            }
        }
        
        info!("Shutdown complete");
        Ok(())
    }
}

fn is_livecaptions_running() -> bool {
    unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None).0 != 0 }
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    info!("get-livecaptions starting");

    if !is_livecaptions_running() {
        error!("Live Captions is not running. Program exiting.");
        process::exit(1);
    }

    let mut engine = Engine::new()
        .context("Failed to initialize engine")?;

    let mut windows_timer = tokio::time::interval(Duration::from_secs(args.check_interval));
    let mut capture_timer = tokio::time::interval(Duration::from_secs_f64(args.capture_interval));

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    println!("Live captions monitoring started:");
    println!("  - Capture interval: {} seconds", args.capture_interval);
    println!("  - Check interval: {} seconds", args.check_interval);
    println!("Press Ctrl+C to exit");
    println!("-----------------------------------");
    
    loop {
        tokio::select! {
            _ = windows_timer.tick() => {
                info!("Checking if Live Captions is running");
                if !is_livecaptions_running() {
                    error!("Live Captions is no longer running. Program exiting.");
                    if let Err(e) = engine.graceful_shutdown().await {
                        error!("Error during shutdown: {}", e);
                    }
                    process::exit(1);
                }
            },
            _ = capture_timer.tick() => {
                info!("Capturing live captions");
                // 使用带重试的方法获取字幕
                match engine.get_livecaptions_with_retry(2).await {
                    Ok(Some(text)) => {
                        // 将新文本追加到已显示的文本中
                        engine.displayed_text.push_str(&text);
                        
                        // 限制文本长度
                        engine.limit_text_length();
                        
                        // 清除当前行并显示完整的累积文本
                        print!("\r\x1B[K> {}", engine.displayed_text);
                        io::stdout().flush().ok(); // 确保立即输出
                        
                        // 有新内容时重置连续空捕获计数并降低间隔
                        engine.consecutive_empty_captures = 0;
                        engine.adaptive_interval = engine.min_interval;
                        // 更新定时器
                        capture_timer = tokio::time::interval(Duration::from_secs_f64(engine.adaptive_interval));
                    },
                    Ok(None) => {
                        info!("No new captions available");
                        // 连续没有新内容时逐渐增加间隔以减少资源使用
                        engine.consecutive_empty_captures += 1;
                        if engine.consecutive_empty_captures > 5 {
                            engine.adaptive_interval = (engine.adaptive_interval * 1.2).min(engine.max_interval);
                            info!("Adjusting capture interval to {} seconds", engine.adaptive_interval);
                            // 更新定时器
                            capture_timer = tokio::time::interval(Duration::from_secs_f64(engine.adaptive_interval));
                        }
                    },
                    Err(e) => {
                        warn!("Failed to capture captions after retries: {}", e);
                    }
                }
            },
            _ = &mut ctrl_c => {
                println!("\nReceived shutdown signal");
                if let Err(e) = engine.graceful_shutdown().await {
                    error!("Error during shutdown: {}", e);
                    process::exit(1);
                }
                info!("Program terminated successfully");
                return Ok(());
            }
        };
    }
}
