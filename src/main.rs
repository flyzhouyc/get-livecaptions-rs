use tokio::time::Duration;
use std::process;
use std::path::PathBuf;
use windows::{
    core::*, Win32::{System::Com::*, UI::{Accessibility::*, WindowsAndMessaging::*}, Foundation::HWND},
};
use clap::Parser;
use log::{error, info, warn};
use anyhow::{Result, Context, anyhow};
use std::io::{self, Write};
//use std::borrow::Cow;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::fs;
use serde::{Deserialize, Serialize};

// 翻译API类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TranslationApiType {
    DeepL,    // DeepL API
    Generic,  // 通用API
    Demo,     // 演示模式
    OpenAI,   // OpenAI 及兼容接口
}

// 翻译客户端结构体
struct TranslationClient {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    api_type: TranslationApiType,
    // OpenAI 特定字段
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
}

impl TranslationClient {
    fn new(
        api_key: String, 
        target_language: String, 
        api_type: TranslationApiType,
        openai_api_url: Option<String>,
        openai_model: Option<String>,
        openai_system_prompt: Option<String>,
    ) -> Self {
        Self {
            api_key,
            target_language,
            client: reqwest::Client::new(),
            api_type,
            openai_api_url,
            openai_model,
            openai_system_prompt,
        }
    }
    
    async fn translate(&self, text: &str) -> Result<String> {
        match self.api_type {
            TranslationApiType::Demo => {
                // 仅用于演示的简单"翻译"
                Ok(format!("翻译: {}", text))
            },
            
            TranslationApiType::DeepL => {
                // DeepL API 调用 (免费版)
                let url = "https://api-free.deepl.com/v2/translate";
                
                let response = self.client.post(url)
                    .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
                    .form(&[
                        ("text", text),
                        ("target_lang", &self.target_language),
                        ("source_lang", "EN"), // 假设源语言为英语，也可以设置为"auto"
                    ])
                    .send()
                    .await
                    .context("Failed to send translation request to DeepL")?;
                    
                if !response.status().is_success() {
                    return Err(anyhow!("DeepL API error: {}", response.status()));
                }
                
                #[derive(Deserialize)]
                struct DeepLResponse {
                    translations: Vec<Translation>,
                }
                
                #[derive(Deserialize)]
                struct Translation {
                    text: String,
                }
                
                let result: DeepLResponse = response.json()
                    .await
                    .context("Failed to parse DeepL response")?;
                    
                if let Some(translation) = result.translations.first() {
                    Ok(translation.text.clone())
                } else {
                    Err(anyhow!("Empty translation result from DeepL"))
                }
            },
            
            TranslationApiType::Generic => {
                // 通用API调用
                let url = "https://translation-api.example.com/translate";
                
                let response = self.client.post(url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .json(&serde_json::json!({
                        "text": text,
                        "source_language": "auto",
                        "target_language": self.target_language
                    }))
                    .send()
                    .await
                    .context("Failed to send translation request")?;
                    
                if !response.status().is_success() {
                    return Err(anyhow!("Translation API error: {}", response.status()));
                }
                
                let result: serde_json::Value = response.json()
                    .await
                    .context("Failed to parse translation response")?;
                    
                // 根据API响应格式提取翻译结果
                result.get("translated_text")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .ok_or_else(|| anyhow!("Invalid translation response format"))
            },
            
            TranslationApiType::OpenAI => {
                // OpenAI 兼容 API 调用
                // 获取 API URL，如果未指定则使用默认的 OpenAI URL
                let url = self.openai_api_url.as_deref()
                    .unwrap_or("https://api.openai.com/v1/chat/completions");
                
                // 获取模型名称，如果未指定则使用默认模型
                let model = self.openai_model.as_deref()
                    .unwrap_or("gpt-3.5-turbo");
                
                // 获取系统提示词，如果未指定则使用默认提示词
                let system_prompt = self.openai_system_prompt.as_deref()
                    .unwrap_or("You are a translator. Translate the following text to the target language. Only respond with the translation, no explanations.");
                
                // 构建请求体
                let request_body = serde_json::json!({
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": format!("Translate the following text to {}: {}", self.target_language, text)
                        }
                    ],
                    "temperature": 0.3
                });
                
                // 发送请求
                let response = self.client.post(url)
                    .header("Authorization", format!("Bearer {}", self.api_key))
                    .header("Content-Type", "application/json")
                    .json(&request_body)
                    .send()
                    .await
                    .context("Failed to send translation request to OpenAI compatible API")?;
                    
                if !response.status().is_success() {
                    return Err(anyhow!("OpenAI API error: {}", response.status()));
                }
                
                // 解析响应
                #[derive(Deserialize)]
                struct OpenAIResponse {
                    choices: Vec<Choice>,
                }
                
                #[derive(Deserialize)]
                struct Choice {
                    message: Message,
                }
                
                #[derive(Deserialize)]
                struct Message {
                    content: String,
                }
                
                let result: OpenAIResponse = response.json()
                    .await
                    .context("Failed to parse OpenAI response")?;
                    
                if let Some(choice) = result.choices.first() {
                    Ok(choice.message.content.clone())
                } else {
                    Err(anyhow!("Empty translation result from OpenAI"))
                }
            }
        }
    }
}

/// 配置文件结构
#[derive(Debug, Serialize, Deserialize)]
struct Config {
    // 基本设置
    capture_interval: f64,  // 捕获间隔（秒）
    check_interval: u64,    // 检查 Live Captions 是否运行的间隔（秒）
    
    // 高级设置
    min_interval: f64,      // 最小捕获间隔（秒）
    max_interval: f64,      // 最大捕获间隔（秒）
    max_text_length: usize, // 存储文本的最大长度
    
    // 输出设置
    output_file: Option<String>, // 输出文件路径（可选）
    
    // 翻译设置
    enable_translation: bool,    // 是否启用翻译
    translation_api_key: Option<String>, // 翻译API密钥（可选）
    translation_api_type: Option<String>, // 翻译API类型（"deepl", "generic", "demo", "openai"）
    target_language: Option<String>,    // 目标语言（可选）
    
    // OpenAI 相关配置
    openai_api_url: Option<String>,     // API 端点 URL
    openai_model: Option<String>,       // 模型名称
    openai_system_prompt: Option<String>, // 系统提示词
}

impl Default for Config {
    fn default() -> Self {
        Self {
            capture_interval: 1.0,
            check_interval: 10,
            min_interval: 0.5,
            max_interval: 3.0,
            max_text_length: 10000,
            output_file: None,
            enable_translation: false,
            translation_api_key: None,
            translation_api_type: Some("deepl".to_string()), // 默认使用DeepL API
            target_language: None,
            openai_api_url: None,
            openai_model: None,
            openai_system_prompt: None,
        }
    }
}

impl Config {
    // 从文件加载配置
    fn load(path: &PathBuf) -> Result<Self> {
        if path.exists() {
            let content = fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file: {:?}", path))?;
            serde_json::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {:?}", path))
        } else {
            // 如果配置文件不存在，创建默认配置
            let config = Config::default();
            let content = serde_json::to_string_pretty(&config)
                .context("Failed to serialize default config")?;
            fs::write(path, content)
                .with_context(|| format!("Failed to write default config to {:?}", path))?;
            info!("Created default config at {:?}", path);
            Ok(config)
        }
    }
    
    // 保存配置到文件
    fn save(&self, path: &PathBuf) -> Result<()> {
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize config")?;
        fs::write(path, content)
            .with_context(|| format!("Failed to write config to {:?}", path))?;
        info!("Saved config to {:?}", path);
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Interval of seconds for capturing captions
    #[arg(short = 'i', long)]
    capture_interval: Option<f64>,   

    /// Interval of seconds for checking if Live Captions is running
    #[arg(short = 'c', long)]
    check_interval: Option<u64>,
    
    /// Path to config file
    #[arg(short = 'f', long)]
    config: Option<PathBuf>,
    
    /// Path to output file
    #[arg(short = 'o', long)]
    output_file: Option<String>,
}

// 用于缓存的键类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CacheKey {
    WindowElement(isize), // 窗口句柄的值作为键
    TextElement(isize),   // 窗口句柄的值作为键
}

struct Engine {
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
    previous_text: String,
    displayed_text: String, // 用于跟踪显示在终端的文本
    element_cache: LruCache<CacheKey, IUIAutomationElement>, // 使用LRU缓存存储UI元素
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
        
        // 创建一个容量为10的LRU缓存
        let cache_capacity = NonZeroUsize::new(10).unwrap();
        
        Ok(Self {
            automation,
            condition,
            previous_text: String::new(),
            displayed_text: String::new(),
            element_cache: LruCache::new(cache_capacity),
            last_window_handle: HWND(0),
            consecutive_empty_captures: 0,
            adaptive_interval: default_min_interval,
            min_interval: default_min_interval,
            max_interval: default_max_interval,
            max_text_length: default_max_text_length,
        })
    }

    // 使用 Myers 差异算法的文本差异检测
    fn extract_new_text<'a>(&self, current: &'a str) -> std::borrow::Cow<'a, str> {
        use std::borrow::Cow;
        
        if self.previous_text.is_empty() {
            return Cow::Borrowed(current);
        }
        
        // 对于非常长的文本，使用滑动窗口方法减少比较范围
        if current.len() > 1000 && self.previous_text.len() > 1000 {
            // 假设新内容主要添加在末尾，从后半部分开始比较
            let window_start = self.previous_text.len().saturating_sub(500);
            let prev_window = &self.previous_text[window_start..];
            
            // 如果当前文本包含前一个窗口的内容，找出差异部分
            if let Some(pos) = current.find(prev_window) {
                if pos + prev_window.len() < current.len() {
                    // 返回窗口后的新内容
                    return Cow::Borrowed(&current[(pos + prev_window.len())..]);
                }
            }
        }
        
        // 实现简化版的 Myers 差异算法来查找最长公共子序列
        // 这里使用前缀匹配作为高效近似
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
                        return Cow::Borrowed(&new_content[pos+1..]);
                    }
                }
            }
            return Cow::Borrowed(new_content);
        }
        
        // 如果当前文本完全不同于之前的文本，返回整个当前文本
        if i == 0 && current != self.previous_text {
            return Cow::Borrowed(current);
        }
        
        Cow::Borrowed("")
    }

    async fn get_livecaptions(&mut self) -> Result<Option<String>> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        if window.0 == 0 {
            return Err(anyhow!("Live Captions window not found"));
        }
        
        // 检查窗口是否发生变化，只有需要时才清理缓存
        let window_handle_value = window.0;
        let window_changed = window_handle_value != self.last_window_handle.0;
        if window_changed {
            info!("Window handle changed from {:?} to {:?}, refreshing UI elements", 
                  self.last_window_handle.0, window_handle_value);
            // 清除所有缓存的元素
            self.element_cache.clear();
            self.last_window_handle = window;
        }
        
        // 获取或缓存窗口元素
        let window_key = CacheKey::WindowElement(window_handle_value);
        let window_element = if let Some(element) = self.element_cache.get(&window_key) {
            element.clone()
        } else {
            let element = unsafe { self.automation.ElementFromHandle(window) }
                .map_err(|e| anyhow!("Failed to get element from window handle: {:?}", e))?;
            self.element_cache.put(window_key, element.clone());
            element
        };
        
        // 获取或缓存文本元素
        let text_key = CacheKey::TextElement(window_handle_value);
        let text_element = if let Some(element) = self.element_cache.get(&text_key) {
            element.clone()
        } else {
            let element = unsafe { window_element.FindFirst(TreeScope_Descendants, &self.condition) }
                .map_err(|e| anyhow!("Failed to find captions text element: {:?}", e))?;
            self.element_cache.put(text_key, element.clone());
            element
        };
        
        // 从文本元素获取字幕内容
        let current_text = unsafe { text_element.CurrentName() }
            .map_err(|e| {
                // 如果获取文本失败，可能是元素已失效，清除缓存以便下次重新获取
                self.element_cache.pop(&window_key);
                self.element_cache.pop(&text_key);
                anyhow!("Failed to get text from element: {:?}", e)
            })?
            .to_string();
        
        // 如果文本为空或与上次相同，返回None
        if current_text.is_empty() || current_text == self.previous_text {
            return Ok(None);
        }
        
        // 使用优化的文本差异检测算法提取新增内容
        let new_text = self.extract_new_text(&current_text);
        
        // 更新上一次的文本（克隆以避免借用冲突）
        self.previous_text = current_text.clone();
        
        if !new_text.is_empty() {
            Ok(Some(new_text.into_owned()))
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
                // 使用更兼容的方式显示最终字幕
                println!("\n");
                print!("> {}", self.displayed_text);
                io::stdout().flush().ok();
            }
            Ok(None) => {
                info!("No new captions at shutdown");
                if !self.displayed_text.is_empty() {
                    // 使用更兼容的方式显示最终字幕
                    println!("\n");
                    print!("> {}", self.displayed_text);
                    io::stdout().flush().ok();
                }
            }
            Err(err) => {
                warn!("Could not capture final captions: {}", err);
                if !self.displayed_text.is_empty() {
                    // 使用更兼容的方式显示最终字幕
                    println!("\n");
                    print!("> {}", self.displayed_text);
                    io::stdout().flush().ok();
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
    
    // 解析命令行参数
    let args = Args::parse();
    info!("get-livecaptions starting");
    
    // 配置路径
    let config_path = args.config.unwrap_or_else(|| {
        let mut path = dirs::config_dir().unwrap_or_else(|| PathBuf::from("."));
        path.push("get-livecaptions");
        fs::create_dir_all(&path).unwrap_or_else(|_| {
            warn!("Failed to create config directory, using current directory");
            path = PathBuf::from(".");
        });
        path.push("config.json");
        path
    });
    
    // 加载或创建配置
    let mut config = Config::load(&config_path)
        .context("Failed to load config")?;
    
    // 使用命令行参数覆盖配置（如果提供）
    if let Some(interval) = args.capture_interval {
        config.capture_interval = interval;
    }
    if let Some(check) = args.check_interval {
        config.check_interval = check;
    }
    if let Some(output) = args.output_file {
        config.output_file = Some(output);
    }
    
    // 保存更新后的配置
    config.save(&config_path)?;
    
    // 检查 Live Captions 是否运行
    if !is_livecaptions_running() {
        error!("Live Captions is not running. Program exiting.");
        process::exit(1);
    }
    
    // 创建输出文件（如果配置了）
    let mut output_file = if let Some(path) = &config.output_file {
        let file = fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Failed to open output file: {}", path))?;
        info!("Writing output to file: {}", path);
        Some(file)
    } else {
        None
    };
    
    // 初始化引擎
    let mut engine = Engine::new()
        .context("Failed to initialize engine")?;
    
    // 从配置设置引擎参数
    engine.min_interval = config.min_interval;
    engine.max_interval = config.max_interval;
    engine.adaptive_interval = config.min_interval;
    engine.max_text_length = config.max_text_length;
    
    // 初始化定时器
    let mut windows_timer = tokio::time::interval(Duration::from_secs(config.check_interval));
    let mut capture_timer = tokio::time::interval(Duration::from_secs_f64(config.capture_interval));
    
    // 设置 Ctrl+C 处理
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    
    println!("Live captions monitoring started:");
    println!("  - Capture interval: {} seconds", config.capture_interval);
    println!("  - Check interval: {} seconds", config.check_interval);
    if config.enable_translation {
        println!("  - Translation enabled (target: {})", 
                 config.target_language.as_deref().unwrap_or("auto"));
    }
    if output_file.is_some() {
        println!("  - Writing to file: {}", config.output_file.as_deref().unwrap());
    }
    println!("Press Ctrl+C to exit");
    println!("-----------------------------------");
    
    // 创建翻译客户端（如果启用）
    let translation_client = if config.enable_translation {
        if let Some(api_key) = &config.translation_api_key {
            // 确定API类型
            let api_type = match config.translation_api_type.as_deref() {
                Some("deepl") => TranslationApiType::DeepL,
                Some("generic") => TranslationApiType::Generic,
                Some("demo") => TranslationApiType::Demo,
                Some("openai") => TranslationApiType::OpenAI,
                _ => TranslationApiType::DeepL, // 默认使用DeepL
            };
            
            // 确定目标语言
            let target_lang = config.target_language.clone().unwrap_or_else(|| {
                match api_type {
                    TranslationApiType::DeepL => "ZH".to_string(), // DeepL 使用大写语言代码
                    TranslationApiType::OpenAI => "Chinese".to_string(), // OpenAI 使用自然语言描述
                    _ => "zh-CN".to_string(),
                }
            });
            
            info!("Initializing translation service with {:?} API", api_type);
            Some(TranslationClient::new(
                api_key.clone(),
                target_lang,
                api_type,
                config.openai_api_url.clone(),
                config.openai_model.clone(),
                config.openai_system_prompt.clone()
            ))
        } else {
            warn!("Translation enabled but no API key provided");
            None
        }
    } else {
        None
    };
    
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
                        // 处理翻译（如果启用）
                        let display_text = if let Some(client) = &translation_client {
                            match client.translate(&text).await {
                                Ok(translated) => {
                                    info!("Translated: {} -> {}", text, translated);
                                    format!("{} [{}]", text, translated)
                                }
                                Err(e) => {
                                    warn!("Translation failed: {}", e);
                                    text.clone()
                                }
                            }
                        } else {
                            text.clone()
                        };
                        
                        // 将新文本追加到已显示的文本中
                        engine.displayed_text.push_str(&display_text);
                        
                        // 限制文本长度
                        engine.limit_text_length();
                        
                        // 清除当前行并显示完整的累积文本 - 使用更兼容的方式
                        print!("\r");  // 回车到行首
                        // 用足够多的空格覆盖旧内容
                        for _ in 0..120 {  // 假设终端宽度不超过120个字符
                            print!(" ");
                        }
                        print!("\r> {}", engine.displayed_text);
                        io::stdout().flush().ok(); // 确保立即输出
                        
                        // 如果配置了输出文件，写入文件
                        if let Some(file) = &mut output_file {
                            if let Err(e) = writeln!(file, "{}", display_text) {
                                warn!("Failed to write to output file: {}", e);
                            }
                        }
                        
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
