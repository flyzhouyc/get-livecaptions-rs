use tokio::time::Duration;
use std::process;
use chrono::prelude::*;
use windows::{
    core::*, Win32::{System::Com::*, UI::{Accessibility::*, WindowsAndMessaging::*}},
};
use clap::Parser;
use log::{error, info};
use anyhow::Result;
use reqwest;
use serde_json;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// interval of minutes for one cycle
    #[arg(short, long, default_value_t = 3, value_parser = clap::value_parser!(u8).range(1..6))]
    interval: u8,
}

struct Engine {
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
    prebuffer: String,
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { CoUninitialize(); }
    }
}

impl Engine {
    fn new() -> Self {
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok().expect("Failed to initialize Windows COM."); }

        let automation: IUIAutomation = unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_ALL).expect("Failed to initialize Windows Accessibility API.") };
        let condition = unsafe { automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from("CaptionsTextBlock")).unwrap() };
        Self {
            automation,
            condition,
            prebuffer: Default::default(),
        }
    }

    async fn get_livecaptions(&self) -> Result<String> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        let element = unsafe { self.automation.ElementFromHandle(window) }?;
        let text = unsafe { element.FindFirst(TreeScope_Descendants, &self.condition) }?;
        let text = unsafe { text.CurrentName() }?;
        println!("Live Captions: {}", text);
        Ok(text.to_string())
    }

    // Commenting out the save_current_captions function
    /*
    fn save_current_captions(&mut self, current: &str, include_last_line: bool) -> Result<()> {
        use std::fs::OpenOptions;
        use std::io::prelude::*;
        let last_line = if !include_last_line { 1 } else { 0 };

        let mut lines: Vec<&str> = current.lines().collect();
        let mut first_new_line = None;

        for (i, line) in lines.iter().enumerate() {
            if !self.prebuffer.contains(line) {
                first_new_line = Some(i);
                break;
            }
        }

        if let Some(start) = first_new_line {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.sfilename)?;

            let local: DateTime<Local> = Local::now();
            write!(file, "{}\n", local.format("[%Y-%m-%d][%H:%M:%S]"))?;
            for line in lines.drain(start..lines.len() - last_line) {
                self.prebuffer.push_str(line);
                self.prebuffer.push('\n');

                file.write_all(line.as_bytes())?;
                file.write(b"\n")?;
            }
        }
        Ok(())
    }
    */

    async fn graceful_shutdown(&mut self) -> Result<()> {
        let text = self.get_livecaptions().await?;
        // Commenting out the call to save_current_captions
        // self.save_current_captions(&text, true)?;
        Ok(())
    }
}

fn is_livecaptions_running() -> bool {
    unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None).0 != 0 }
}

#[tokio::main]
async fn main() {
    env_logger::init();
    let args = Args::parse();
    info!("get-livecaptions running.");

    if !is_livecaptions_running() {
        error!("livecaptions is not running. program exiting.");
        return;
    }

    let mut engine = Engine::new();

    let mut windows_timer = tokio::time::interval(Duration::from_secs(10));
    let mut writefile_timer = tokio::time::interval(Duration::from_secs(1)); // Change to 1-second interval

    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    println!("get-livecaptions is running now, and save content into '{}', every {} min. ctrl-c for exit.", args.file, args.interval);
    loop {
        tokio::select! {
            _ = windows_timer.tick() => {
                log::info!("running checking, every 10s.");
                if !is_livecaptions_running() {
                    println!("livecaptions is not running. program exiting.");
                    let _ = engine.graceful_shutdown().await;
                    process::exit(0);
                }
            },
            _ = writefile_timer.tick() => {
                log::info!("save content into file, every {} min.", args.interval);
                let text = engine.get_livecaptions().await;
                if let Ok(text) = text {
                    println!("Translated Text: {}", text); // Print the translated text to the command line
                    // Commenting out the call to save_current_captions
                    // engine.save_current_captions(&text, false).expect("save file failed.");
                }
            },
            _ = &mut ctrl_c => {
                let _ = engine.graceful_shutdown().await;
                process::exit(0);
            }
        };
    }
}
