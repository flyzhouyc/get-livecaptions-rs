use tokio::time::Duration;
use std::process;
use windows::{
    core::*, Win32::{System::Com::*, UI::{Accessibility::*, WindowsAndMessaging::*}},
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
        
        Ok(Self {
            automation,
            condition,
            previous_text: String::new(),
            displayed_text: String::new(),
        })
    }

    async fn get_livecaptions(&mut self) -> Result<Option<String>> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        if window.0 == 0 {
            return Err(anyhow!("Live Captions window not found"));
        }
        
        let element = unsafe { self.automation.ElementFromHandle(window) }
            .map_err(|e| anyhow!("Failed to get element from window handle: {:?}", e))?;
            
        let text_element = unsafe { element.FindFirst(TreeScope_Descendants, &self.condition) }
            .map_err(|e| anyhow!("Failed to find captions text element: {:?}", e))?;
            
        let current_text = unsafe { text_element.CurrentName() }
            .map_err(|e| anyhow!("Failed to get text from element: {:?}", e))?
            .to_string();
        
        // 如果文本为空或与上次相同，返回None
        if current_text.is_empty() || current_text == self.previous_text {
            return Ok(None);
        }
        
        // 提取新增的文本行
        let new_text = if self.previous_text.is_empty() {
            current_text.clone()
        } else {
            // 寻找新增的内容
            if current_text.starts_with(&self.previous_text) {
                current_text[self.previous_text.len()..].trim_start().to_string()
            } else {
                // 如果旧文本不是新文本的前缀，可能是文本完全更新了
                current_text.clone()
            }
        };
        
        // 更新上一次的文本
        self.previous_text = current_text;
        
        if !new_text.is_empty() {
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }

    async fn graceful_shutdown(&mut self) -> Result<()> {
        info!("Performing graceful shutdown");
        match self.get_livecaptions().await {
            Ok(Some(text)) => {
                // 将最后的文本追加到显示文本中
                self.displayed_text.push_str(&text);
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
                match engine.get_livecaptions().await {
                    Ok(Some(text)) => {
                        // 将新文本追加到已显示的文本中
                        engine.displayed_text.push_str(&text);
                        
                        // 清除当前行并显示完整的累积文本
                        print!("\r\x1B[K> {}", engine.displayed_text);
                        io::stdout().flush().ok(); // 确保立即输出
                    },
                    Ok(None) => {
                        info!("No new captions available");
                    },
                    Err(e) => {
                        warn!("Failed to capture captions: {}", e);
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
