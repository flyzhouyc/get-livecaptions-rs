use tokio::time::{Duration, Instant};
use std::path::PathBuf;
use std::sync::Arc;
use windows::{
    core::*, Win32::System::Com::*, Win32::UI::{Accessibility::*, WindowsAndMessaging::*}, 
    Win32::Foundation::{HWND, HANDLE, CloseHandle},
    Win32::System::Pipes::*,
    Win32::Storage::FileSystem::{CreateFileW, FILE_SHARE_MODE, FILE_SHARE_NONE, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, WriteFile, ReadFile},
    Win32::Security::*,
};
use windows::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
use clap::Parser;
use log::{debug, error, info, warn};
use thiserror::Error;
use anyhow::{Result, Context};
use std::io::{self, Write};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::fs;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use tokio::sync::{mpsc, oneshot};
use tokio::task;
use std::process::{Command, Stdio};
use std::ffi::OsString;
use std::os::windows::process::CommandExt;

// (现有代码保持不变)

/// 命令行参数结构
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// 捕获字幕的时间间隔（秒）
    #[arg(short = 'i', long)]
    capture_interval: Option<f64>,   

    /// 检查Live Captions是否运行的时间间隔（秒）
    #[arg(short = 'c', long)]
    check_interval: Option<u64>,
    
    /// 配置文件路径
    #[arg(short = 'f', long)]
    config: Option<PathBuf>,
    
    /// 输出文件路径
    #[arg(short = 'o', long)]
    output_file: Option<String>,
    
    /// 是否启用翻译
    #[arg(short = 't', long)]
    enable_translation: Option<bool>,
    
    /// 目标语言
    #[arg(short = 'l', long)]
    target_language: Option<String>,
    
    /// 标记这是一个翻译窗口进程
    #[arg(long, hide = true)]
    translation_window: bool,
    
    /// 用于进程间通信的命名管道名称
    #[arg(long, hide = true)]
    pipe_name: Option<String>,
}

/// 进程间通信的消息类型
#[derive(Debug, Serialize, Deserialize)]
enum IpcMessage {
    /// 发送给翻译窗口的配置消息
    Config(Config),
    /// 发送给翻译窗口的文本消息
    Text(String),
    /// 关闭翻译窗口的消息
    Shutdown,
}

/// Windows命名管道封装
struct NamedPipe {
    handle: HANDLE,
}

impl NamedPipe {
    /// 创建一个新的命名管道服务器
    fn create_server(pipe_name: &str) -> Result<Self, AppError> {
        let pipe_path = format!(r"\\.\pipe\{}", pipe_name);
        
        let handle = unsafe {
            CreateNamedPipeW(
                &HSTRING::from(pipe_path),
                windows::Win32::Storage::FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0x00000003), // PIPE_ACCESS_DUPLEX
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                1, // 最大实例数
                4096, // 输出缓冲区大小
                4096, // 输入缓冲区大小
                0, // 默认超时
                None, // 默认安全属性
            )
        };
        
        if handle.is_invalid() {
            return Err(AppError::Io(io::Error::last_os_error()));
        }
        
        Ok(Self { handle })
    }
    
    /// 连接到现有的命名管道
    fn connect_client(pipe_name: &str) -> Result<Self, AppError> {
        let pipe_path = format!(r"\\.\pipe\{}", pipe_name);
        
        let handle = unsafe {
            CreateFileW(
                &HSTRING::from(pipe_path),
                (GENERIC_READ | GENERIC_WRITE).0,
                FILE_SHARE_NONE,
                None,
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL,
                HANDLE(0),
            )?
        };
        
        Ok(Self { handle })
    }
    
    /// 等待客户端连接到管道
    fn wait_for_connection(&self) -> Result<(), AppError> {
        let result = unsafe { ConnectNamedPipe(self.handle, None) };
        if result.is_err() {
            let error = io::Error::last_os_error();
            // ERROR_PIPE_CONNECTED表示客户端已经连接
            if error.raw_os_error() != Some(535) {
                return Err(AppError::Io(error));
            }
        }
        Ok(())
    }
    
    /// 向管道写入消息
    fn write_message(&self, message: &IpcMessage) -> Result<(), AppError> {
        let data = serde_json::to_vec(message)
            .map_err(|e| AppError::Io(io::Error::new(io::ErrorKind::InvalidData, e)))?;
        
        let mut bytes_written = 0;
        let success = unsafe {
            WriteFile(
                self.handle,
                Some(&data),
                Some(&mut bytes_written),
                None,
            )
        };
        
        if success.is_err() || bytes_written as usize != data.len() {
            return Err(AppError::Io(io::Error::last_os_error()));
        }
        
        Ok(())
    }
    
    /// 从管道读取消息
    fn read_message(&self) -> Result<IpcMessage, AppError> {
        let mut buffer = vec![0u8; 4096];
        let mut bytes_read = 0;
        
        let success = unsafe {
            ReadFile(
                self.handle,
                Some(&mut buffer),
                Some(&mut bytes_read),
                None,
            )
        };
        
        if success.is_err() {
            return Err(AppError::Io(io::Error::last_os_error()));
        }
        
        buffer.truncate(bytes_read as usize);
        
        let message = serde_json::from_slice(&buffer)
            .map_err(|e| AppError::Io(io::Error::new(io::ErrorKind::InvalidData, e)))?;
        
        Ok(message)
    }
}

impl Drop for NamedPipe {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

/// 字幕处理引擎
struct Engine {
    config: Config,
    displayed_text: String, // 在终端显示的文本
    caption_handle: CaptionHandle,
    translation_service: Option<Arc<dyn TranslationService>>,
    consecutive_empty_captures: usize, // 连续空白捕获的计数
    adaptive_interval: f64, // 自适应捕获间隔
    output_file: Option<fs::File>, // 输出文件句柄
    translation_process: Option<std::process::Child>, // 翻译窗口进程
    translation_pipe: Option<NamedPipe>, // 用于与翻译窗口进行IPC的命名管道
}

impl Engine {
    /// 创建并初始化新的引擎实例
    async fn new(config: Config) -> Result<Self, AppError> {
        debug!("使用配置初始化引擎: {:?}", config);
        
        // 创建字幕句柄
        let caption_handle = CaptionHandle::new()?;
        
        // 只在主窗口中创建翻译服务（用于文件输出）
        let translation_service = if config.enable_translation {
            if let Some(api_key) = &config.translation_api_key {
                // 确保目标语言已设置
                let target_lang = config.target_language.clone().unwrap_or_else(|| {
                    match config.translation_api_type {
                        TranslationApiType::DeepL => "ZH".to_string(), 
                        TranslationApiType::OpenAI => "Chinese".to_string(),
                        _ => "zh-CN".to_string(),
                    }
                });
                
                info!("初始化翻译服务，使用 {:?} API", config.translation_api_type);
                Some(create_translation_service(
                    api_key.clone(),
                    target_lang,
                    config.translation_api_type,
                    config.openai_api_url.clone(),
                    config.openai_model.clone(),
                    config.openai_system_prompt.clone()
                ))
            } else if config.translation_api_type == TranslationApiType::Demo {
                // Demo模式不需要API密钥
                let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
                Some(create_translation_service(
                    "".to_string(),
                    target_lang,
                    TranslationApiType::Demo,
                    None,
                    None,
                    None
                ))
            } else {
                warn!("翻译已启用但未提供API密钥");
                None
            }
        } else {
            None
        };
        
        // 创建输出文件（如果配置了）
        let output_file = if let Some(path) = &config.output_file {
            let file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(path)
                .map_err(|e| AppError::Io(e))?;
            info!("输出写入到文件: {}", path);
            Some(file)
        } else {
            None
        };
        
        // 如果启用了翻译，创建翻译窗口
        let (translation_process, translation_pipe) = if config.enable_translation {
            // 生成唯一的管道名称
            let pipe_name = format!("get-livecaptions-pipe-{}", std::process::id());
            info!("创建命名管道: {}", pipe_name);
            
            // 创建命名管道服务器
            let pipe = NamedPipe::create_server(&pipe_name)?;
            
            // 获取当前可执行文件路径
            let exe_path = std::env::current_exe()
                .map_err(|e| AppError::Io(e))?;
            
            // 启动翻译窗口进程
            info!("启动翻译窗口进程");
            let process = Command::new(exe_path)
                .arg("--translation-window")
                .arg("--pipe-name")
                .arg(&pipe_name)
                .stdin(Stdio::null())
                // 调试时可保留标准输出和错误输出
                //.stdout(Stdio::null())
                //.stderr(Stdio::null())
                .creation_flags(0x00000010) // CREATE_NEW_CONSOLE
                .spawn()
                .map_err(|e| AppError::Io(e))?;
            
            // 等待客户端连接
            info!("等待翻译窗口连接");
            pipe.wait_for_connection()?;
            info!("翻译窗口已连接");
            
            // 发送配置到翻译窗口
            pipe.write_message(&IpcMessage::Config(config.clone()))?;
            
            (Some(process), Some(pipe))
        } else {
            (None, None)
        };
        
        Ok(Self {
            displayed_text: String::new(),
            caption_handle,
            translation_service,
            consecutive_empty_captures: 0,
            adaptive_interval: config.min_interval,
            output_file,
            config,
            translation_process,
            translation_pipe,
        })
    }
    
    // (其他方法保持不变，但run方法和graceful_shutdown需要修改)
    
    /// 引擎的主循环
    async fn run(&mut self) -> Result<(), AppError> {
        info!("启动引擎主循环");
        
        // 初始化检查计时器
        let mut check_timer = tokio::time::interval(Duration::from_secs(self.config.check_interval));
        let mut capture_timer = tokio::time::interval(Duration::from_secs_f64(self.config.capture_interval));
        
        // 设置Ctrl+C处理器
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        
        println!("实时字幕监控已启动:");
        println!("  - 捕获间隔: {} 秒", self.config.capture_interval);
        println!("  - 检查间隔: {} 秒", self.config.check_interval);
        if self.config.enable_translation {
            println!("  - 翻译已启用: {}", self.translation_service.as_ref().map_or("否", |s| s.get_name()));
            println!("  - 目标语言: {}", self.translation_service.as_ref().map_or("无", |s| s.get_target_language()));
            println!("  - 翻译将在单独窗口中显示");
        }
        if self.output_file.is_some() {
            println!("  - 写入到文件: {}", self.config.output_file.as_deref().unwrap());
        }
        println!("按Ctrl+C退出");
        println!("-----------------------------------");
        
        // 主事件循环
        loop {
            tokio::select! {
                _ = check_timer.tick() => {
                    info!("检查字幕源是否可用");
                    match self.caption_handle.is_available().await {
                        Ok(available) => {
                            if !available {
                                error!("字幕源不再可用。程序退出。");
                                self.graceful_shutdown().await?;
                                return Err(AppError::UiAutomation("字幕源不可用".to_string()));
                            }
                        },
                        Err(e) => {
                            error!("检查字幕源可用性失败: {}", e);
                            self.graceful_shutdown().await?;
                            return Err(e);
                        }
                    }
                },
                _ = capture_timer.tick() => {
                    info!("捕获实时字幕");
                    match self.caption_handle.get_captions().await {
                        Ok(Some(text)) => {
                            debug!("捕获到新文本: {}", text);
                            
                            // 如果启用了翻译且有管道，将文本发送到翻译窗口
                            if let Some(pipe) = &self.translation_pipe {
                                if let Err(e) = pipe.write_message(&IpcMessage::Text(text.clone())) {
                                    warn!("发送文本到翻译窗口失败: {}", e);
                                }
                            }
                            
                            // 附加文本到显示文本（仅原始文本，不包含翻译）
                            self.displayed_text.push_str(&text);
                            
                            // 限制文本长度
                            self.limit_text_length();
                            
                            // 显示文本
                            Self::display_text(&self.displayed_text)?;
                            
                            // 写入到输出文件（如果配置了）
                            if let Some(file) = &mut self.output_file {
                                // 如果需要同时写入原文和译文到文件，可以在这里处理
                                let processed_text = if let Some(service) = &self.translation_service {
                                    match service.translate(&text).await {
                                        Ok(translated) => {
                                            format!("{} [{}]", text, translated)
                                        },
                                        Err(e) => {
                                            warn!("翻译失败: {}", e);
                                            text
                                        }
                                    }
                                } else {
                                    text
                                };
                                
                                if let Err(e) = writeln!(file, "{}", processed_text) {
                                    warn!("写入到输出文件失败: {}", e);
                                }
                            }
                            
                            // 重置连续空白计数和自适应间隔
                            self.consecutive_empty_captures = 0;
                            self.adaptive_interval = self.config.min_interval;
                            capture_timer = tokio::time::interval(Duration::from_secs_f64(self.adaptive_interval));
                        },
                        Ok(None) => {
                            info!("没有新字幕可用");
                            // 在连续空白捕获上逐渐增加间隔
                            self.consecutive_empty_captures += 1;
                            if self.consecutive_empty_captures > 5 {
                                self.adaptive_interval = (self.adaptive_interval * 1.2).min(self.config.max_interval);
                                info!("调整捕获间隔到 {} 秒", self.adaptive_interval);
                                capture_timer = tokio::time::interval(Duration::from_secs_f64(self.adaptive_interval));
                            }
                        },
                        Err(e) => {
                            warn!("捕获字幕失败: {}", e);
                        }
                    }
                },
                _ = &mut ctrl_c => {
                    println!("\n收到关闭信号");
                    self.graceful_shutdown().await?;
                    info!("程序成功终止");
                    return Ok(());
                }
            };
        }
    }
    
    /// 进行优雅关闭
    async fn graceful_shutdown(&mut self) -> Result<(), AppError> {
        info!("执行优雅关闭");
        
        // 尝试获取最终字幕
        match self.caption_handle.get_captions().await {
            Ok(Some(text)) => {
                // 如果翻译窗口处于活动状态，发送最终文本
                if let Some(pipe) = &self.translation_pipe {
                    if let Err(e) = pipe.write_message(&IpcMessage::Text(text.clone())) {
                        warn!("发送最终文本到翻译窗口失败: {}", e);
                    }
                }
                
                // 附加到显示文本
                self.displayed_text.push_str(&text);
                
                // 限制文本长度
                self.limit_text_length();
                
                info!("捕获到最终字幕: {}", text);
            },
            Ok(None) => {
                info!("关闭时没有新字幕");
            },
            Err(err) => {
                warn!("无法捕获最终字幕: {}", err);
            }
        }
        
        // 显示最终文本
        if !self.displayed_text.is_empty() {
            println!("\n");
            print!("> {}", self.displayed_text);
            io::stdout().flush()?;
        }
        
        // 如果翻译窗口存在，关闭它
        if let Some(pipe) = &self.translation_pipe {
            info!("关闭翻译窗口");
            if let Err(e) = pipe.write_message(&IpcMessage::Shutdown) {
                warn!("发送关闭消息到翻译窗口失败: {}", e);
            }
            
            // 给翻译窗口一些时间关闭
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        // 关闭字幕句柄
        if let Err(e) = self.caption_handle.shutdown().await {
            warn!("关闭字幕actor时出错: {}", e);
        }
        
        info!("关闭完成");
        Ok(())
    }
    
    // (其他现有方法保持不变)
}

/// 运行翻译窗口进程
/// 
/// 此函数处理翻译窗口进程，该进程从主进程接收文本，
/// 翻译它，并在单独的窗口中显示。
async fn run_translation_window(pipe_name: String) -> Result<(), AppError> {
    info!("启动翻译窗口，管道: {}", pipe_name);
    
    // 连接到命名管道
    let pipe = NamedPipe::connect_client(&pipe_name)?;
    info!("已连接到管道: {}", pipe_name);
    
    // 等待配置消息
    let config = match pipe.read_message()? {
        IpcMessage::Config(config) => config,
        _ => return Err(AppError::Config("第一条消息应为Config消息".to_string())),
    };
    
    info!("收到配置");
    
    // 创建翻译服务
    let translation_service = if let Some(api_key) = &config.translation_api_key {
        // 确保目标语言已设置
        let target_lang = config.target_language.clone().unwrap_or_else(|| {
            match config.translation_api_type {
                TranslationApiType::DeepL => "ZH".to_string(),
                TranslationApiType::OpenAI => "Chinese".to_string(),
                _ => "zh-CN".to_string(),
            }
        });
        
        info!("初始化翻译服务，使用 {:?} API", config.translation_api_type);
        Some(create_translation_service(
            api_key.clone(),
            target_lang,
            config.translation_api_type,
            config.openai_api_url.clone(),
            config.openai_model.clone(),
            config.openai_system_prompt.clone()
        ))
    } else if config.translation_api_type == TranslationApiType::Demo {
        // Demo模式不需要API密钥
        let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
        Some(create_translation_service(
            "".to_string(),
            target_lang,
            TranslationApiType::Demo,
            None,
            None,
            None
        ))
    } else {
        warn!("翻译已启用但未提供API密钥");
        None
    };
    
    if translation_service.is_none() {
        return Err(AppError::Translation("创建翻译服务失败".to_string()));
    }
    
    let translation_service = translation_service.unwrap();
    
    // 设置Ctrl+C处理器
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    
    // 清屏并显示窗口标题
    print!("\x1B[2J\x1B[1;1H");
    println!("翻译窗口");
    println!("  - 翻译服务: {}", translation_service.get_name());
    println!("  - 目标语言: {}", translation_service.get_target_language());
    println!("按Ctrl+C退出");
    println!("----------------------------------------");
    io::stdout().flush()?;
    
    // 用于积累文本进行更好的翻译的缓冲区
    let mut text_buffer = String::new();
    let mut displayed_text = String::new();
    
    // 主循环
    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("\n收到关闭信号");
                break;
            },
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // 检查新消息
                match pipe.read_message() {
                    Ok(IpcMessage::Text(text)) => {
                        // 添加文本到缓冲区
                        text_buffer.push_str(&text);
                        
                        // 检查是否有足够的文本进行翻译
                        if !text_buffer.is_empty() {
                            match translation_service.translate(&text_buffer).await {
                                Ok(translated) => {
                                    // 显示翻译文本
                                    displayed_text.push_str(&translated);
                                    displayed_text.push_str("\n");
                                    
                                    // 限制文本长度
                                    if displayed_text.len() > config.max_text_length {
                                        let lines: Vec<&str> = displayed_text.lines().collect();
                                        if lines.len() > 20 {
                                            displayed_text = lines[lines.len() - 20..].join("\n");
                                            displayed_text.push_str("\n");
                                        }
                                    }
                                    
                                    // 显示翻译文本（使用专用于翻译窗口的显示方法）
                                    display_translation_window(&displayed_text)?;
                                    
                                    // 清除缓冲区
                                    text_buffer.clear();
                                },
                                Err(e) => {
                                    warn!("翻译失败: {}", e);
                                }
                            }
                        }
                    },
                    Ok(IpcMessage::Shutdown) => {
                        info!("收到关闭消息");
                        break;
                    },
                    Ok(_) => {
                        // 忽略其他消息类型
                    },
                    Err(e) => {
                        // 检查错误是否由于管道关闭
                        if let AppError::Io(io_err) = &e {
                            if io_err.kind() == io::ErrorKind::BrokenPipe {
                                info!("管道已关闭，正在关闭");
                                break;
                            }
                        }
                        warn!("从管道读取时出错: {}", e);
                    }
                }
            }
        }
    }
    
    println!("\n翻译窗口正在关闭");
    Ok(())
}

/// 专用于翻译窗口的显示方法
fn display_translation_window(text: &str) -> Result<(), AppError> {
    // 清屏并移动到左上角
    print!("\x1B[2J\x1B[1;1H");
    println!("翻译窗口");
    println!("----------------------------------------");
    println!("{}", text);
    io::stdout().flush()?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // 初始化日志记录器
    env_logger::init();
    
    // 解析命令行参数
    let args = Args::parse();
    
    // 检查这是否是翻译窗口进程
    if args.translation_window {
        if let Some(pipe_name) = args.pipe_name {
        if let Some(pipe_name) = args.pipe_name {
            // 作为翻译窗口运行
            return run_translation_window(pipe_name).await.map_err(|e| anyhow::anyhow!(e));
        } else {
            return Err(anyhow::anyhow!("翻译窗口需要管道名称"));
        }
    }
    
    // 创建引擎
    let mut engine = create_engine().await.map_err(|e| anyhow::anyhow!(e))?;
    
    // 运行引擎
    engine.run().await.map_err(|e| anyhow::anyhow!(e))?;
    
    Ok(())
}