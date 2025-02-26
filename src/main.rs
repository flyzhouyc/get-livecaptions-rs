use tokio::time::Duration;
use std::process;
use windows::{
    core::*, Win32::{System::Com::*, UI::{Accessibility::*, WindowsAndMessaging::*}},
};
use clap::Parser;
use log::{error, info, warn};
use anyhow::{Result, Context};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Interval of seconds for capturing captions
    #[arg(short, long, default_value_t = 1, value_parser = clap::value_parser!(u64).range(1..10))]
    capture_interval: u64,

    /// Interval of seconds for checking if Live Captions is running
    #[arg(short = 'c', long, default_value_t = 10)]
    check_interval: u64,
}

struct Engine {
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
    prebuffer: String,
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
            CoInitializeEx(None, COINIT_MULTITHREADED)
                .context("Failed to initialize Windows COM")?;
        }

        let automation: IUIAutomation = unsafe { 
            CoCreateInstance(&CUIAutomation, None, CLSCTX_ALL)
                .context("Failed to initialize Windows Accessibility API")?
        };
        
        let condition = unsafe { 
            automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from("CaptionsTextBlock"))
                .context("Failed to create property condition")?
        };
        
        Ok(Self {
            automation,
            condition,
            prebuffer: Default::default(),
        })
    }

    async fn get_livecaptions(&self) -> Result<String> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        if window.0 == 0 {
            return Err(anyhow::anyhow!("Live Captions window not found"));
        }
        
        let element = unsafe { self.automation.ElementFromHandle(window) }
            .context("Failed to get element from window handle")?;
            
        let text_element = unsafe { element.FindFirst(TreeScope_Descendants, &self.condition) }
            .context("Failed to find captions text element")?;
            
        let text = unsafe { text_element.CurrentName() }
            .context("Failed to get text from element")?;
            
        Ok(text.to_string())
    }

    async fn graceful_shutdown(&mut self) -> Result<()> {
        info!("Performing graceful shutdown");
        match self.get_livecaptions().await {
            Ok(text) => {
                info!("Final captions captured: {}", text);
                // Any final processing of the text could go here
            }
            Err(err) => {
                warn!("Could not capture final captions: {}", err);
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
    let mut capture_timer = tokio::time::interval(Duration::from_secs(args.capture_interval));

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    println!("Live captions monitoring started:");
    println!("  - Capture interval: {} seconds", args.capture_interval);
    println!("  - Check interval: {} seconds", args.check_interval);
    println!("Press Ctrl+C to exit");
    
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
                    Ok(text) => {
                        if !text.is_empty() {
                            println!("Captions: {}", text);
                        } else {
                            info!("No captions text available");
                        }
                    },
                    Err(e) => {
                        warn!("Failed to capture captions: {}", e);
                    }
                }
            },
            _ = &mut ctrl_c => {
                println!("Received shutdown signal");
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
