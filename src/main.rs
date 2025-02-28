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

/// Main module documentation
/// 
/// This application captures text from accessibility interfaces (primarily Windows Live Captions),
/// processes it, optionally translates it using various translation services, and outputs it
/// to the terminal and/or a file.
/// 
/// # Features
/// 
/// - Captures text from Windows Live Captions using UI Automation
/// - Processes the captured text to extract only new content
/// - Optionally translates captions using various services (DeepL, OpenAI, etc.)
/// - Displays processed text in the terminal and/or writes to a file
/// - Configurable via JSON configuration file and command-line arguments
/// - Adaptive capture intervals based on content availability

/// Define custom error types for better error handling
#[derive(Debug, Error)]
enum AppError {
    /// Errors related to UI automation operations
    #[error("UI automation error: {0}")]
    UiAutomation(String),
    
    /// Errors related to translation operations
    #[error("Translation error: {0}")]
    Translation(String),
    
    /// Errors related to configuration operations
    #[error("Configuration error: {0}")]
    Config(String),
    
    /// I/O errors
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// HTTP request errors
    #[error("HTTP request error: {0}")]
    Request(#[from] reqwest::Error),

    /// COM errors
    #[error("COM error: {0}")]
    Com(String),
    
    /// Task errors
    #[error("Task error: {0}")]
    Task(String),
    
    /// Validation errors
    #[error("Validation error: {0}")]
    Validation(String),

    /// Anyhow errors
    #[error("Error: {0}")]
    Anyhow(#[from] anyhow::Error),
}

// Implement From for windows errors
impl From<windows::core::Error> for AppError {
    fn from(err: windows::core::Error) -> Self {
        AppError::Com(format!("{:?}", err))
    }
}

/// Windows Live Captions handler that operates on a dedicated thread
struct WindowsLiveCaptions {
    previous_text: String,
    element_cache: LruCache<CacheKey, IUIAutomationElement>,
    last_window_handle: HWND,
    max_retries: usize,
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
}

impl Drop for WindowsLiveCaptions {
    fn drop(&mut self) {
        unsafe { CoUninitialize(); }
        info!("COM resources released");
    }
}

impl WindowsLiveCaptions {
    /// Creates a new WindowsLiveCaptions instance
    /// 
    /// # Errors
    /// 
    /// Returns an error if COM initialization or UI Automation setup fails
    fn new() -> Result<Self, AppError> {
        unsafe { 
            let hr = CoInitializeEx(None, COINIT_APARTMENTTHREADED); // Changed to APARTMENTTHREADED
            if hr.is_err() {
                return Err(AppError::Com(format!("Failed to initialize Windows COM: {:?}", hr)));
            }
        }

        let automation: IUIAutomation = unsafe { 
            CoCreateInstance(&CUIAutomation, None, CLSCTX_ALL)?
        };
        
        let condition = unsafe { 
            automation.CreatePropertyCondition(UIA_AutomationIdPropertyId, &VARIANT::from("CaptionsTextBlock"))?
        };
        
        // Create a cache with a reasonable capacity
        let cache_capacity = NonZeroUsize::new(10).unwrap();
        
        Ok(Self {
            automation,
            condition,
            previous_text: String::new(),
            element_cache: LruCache::new(cache_capacity),
            last_window_handle: HWND(0),
            max_retries: 3,
        })
    }

    /// Extracts new text from the current text using diff detection
    /// 
    /// This function uses the similar library to compute differences between
    /// the previous text and the current text, and extracts only the inserted portions.
    /// 
    /// # Arguments
    /// 
    /// * `current` - The current text to compare with the previous text
    /// 
    /// # Returns
    /// 
    /// A String containing only the newly added text
    fn extract_new_text(&self, current: &str) -> String {
        if self.previous_text.is_empty() {
            return current.to_string();
        }
        
        // Use similar library to compute differences
        let diff = TextDiff::from_chars(self.previous_text.as_str(), current);
        
        // Extract only the inserted parts
        let mut new_text = String::new();
        for change in diff.iter_all_changes() {
            if change.tag() == ChangeTag::Insert {
                if let Some(ch) = change.value().chars().next() {
                    new_text.push(ch);
                }
            }
        }
        
        // If diff detection found no new content but texts differ, return entire text
        if new_text.is_empty() && current != self.previous_text {
            return current.to_string();
        }
        
        new_text
    }

    /// Checks if a UI Automation element is still valid
    fn is_element_valid(element: &IUIAutomationElement) -> bool {
        // Try to get an attribute to check if the element is still valid
        unsafe { 
            element.CurrentProcessId().is_ok()
        }
    }

    /// Gets the latest captions from Windows Live Captions
    /// 
    /// # Returns
    /// 
    /// * `Ok(Some(String))` - New captions were found
    /// * `Ok(None)` - No new captions were found
    /// * `Err(AppError)` - An error occurred
    fn get_livecaptions(&mut self) -> Result<Option<String>, AppError> {
        let window = unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None) };
        if window.0 == 0 {
            return Err(AppError::UiAutomation("Live Captions window not found".to_string()));
        }
        
        // Check if window handle changed, clear cache if needed
        let window_handle_value = window.0;
        let window_changed = window_handle_value != self.last_window_handle.0;
        if window_changed {
            debug!("Window handle changed from {:?} to {:?}, refreshing UI elements", 
                  self.last_window_handle.0, window_handle_value);
            self.element_cache.clear();
            self.last_window_handle = window;
        }
        
        // Get or cache window element
        let window_key = CacheKey::WindowElement(window_handle_value);
        let window_element = if let Some(element) = self.element_cache.get(&window_key) {
            if Self::is_element_valid(element) {
                element.clone()
            } else {
                // Element invalid, remove from cache and get new one
                self.element_cache.pop(&window_key);
                let element = unsafe { self.automation.ElementFromHandle(window) }?;
                self.element_cache.put(window_key, element.clone());
                element
            }
        } else {
            let element = unsafe { self.automation.ElementFromHandle(window) }?;
            self.element_cache.put(window_key, element.clone());
            element
        };
        
        // Get or cache text element
        let text_key = CacheKey::TextElement(window_handle_value);
        let text_element = if let Some(element) = self.element_cache.get(&text_key) {
            if Self::is_element_valid(element) {
                element.clone()
            } else {
                // Element invalid, remove from cache and get new one
                self.element_cache.pop(&text_key);
                let element = unsafe { window_element.FindFirst(TreeScope_Descendants, &self.condition) }?;
                self.element_cache.put(text_key, element.clone());
                element
            }
        } else {
            let element = unsafe { window_element.FindFirst(TreeScope_Descendants, &self.condition) }?;
            self.element_cache.put(text_key, element.clone());
            element
        };
        
        // Get caption text from element
        let current_text = unsafe { 
            match text_element.CurrentName() {
                Ok(name) => name.to_string(),
                Err(e) => {
                    // If getting text fails, element may be invalid; clear cache
                    self.element_cache.pop(&window_key);
                    self.element_cache.pop(&text_key);
                    return Err(AppError::UiAutomation(format!("Failed to get text from element: {:?}", e)));
                }
            }
        };
        
        // If text is empty or unchanged, return None
        if current_text.is_empty() || current_text == self.previous_text {
            return Ok(None);
        }
        
        // Extract new content using diff detection
        let new_text = self.extract_new_text(&current_text);
        
        // Update previous text
        self.previous_text = current_text;
        
        if !new_text.is_empty() {
            Ok(Some(new_text))
        } else {
            Ok(None)
        }
    }
    
    /// Gets captions with automatic retry on failure
    /// 
    /// # Arguments
    /// 
    /// * `max_retries` - Maximum number of retry attempts
    /// 
    /// # Returns
    /// 
    /// * `Ok(Some(String))` - New captions were found
    /// * `Ok(None)` - No new captions were found
    /// * `Err(AppError)` - All retry attempts failed
    fn get_livecaptions_with_retry(&mut self, max_retries: usize) -> Result<Option<String>, AppError> {
        let mut attempts = 0;
        let mut last_error = None;
        
        while attempts < max_retries {
            match self.get_livecaptions() {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    warn!("Attempt {}/{} failed: {}", attempts, max_retries, e);
                    last_error = Some(e);
                    if attempts < max_retries {
                        // Exponential backoff - note we're not using async sleep here
                        let backoff_ms = 100 * 2u64.pow(attempts as u32);
                        debug!("Retrying in {} ms", backoff_ms);
                        std::thread::sleep(Duration::from_millis(backoff_ms));
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| AppError::UiAutomation("Unknown error after retries".to_string())))
    }
    
    /// Checks if Live Captions is running
    fn is_available(&self) -> bool {
        unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None).0 != 0 }
    }
}

// Message types for caption worker thread
enum CaptionCommand {
    GetCaption(oneshot::Sender<Result<Option<String>, AppError>>),
    CheckAvailability(oneshot::Sender<bool>),
    Shutdown(oneshot::Sender<()>),
}

/// Caption Actor that runs on its own thread to interact with Windows COM objects
struct CaptionActor {
    captions: WindowsLiveCaptions,
    receiver: mpsc::Receiver<CaptionCommand>,
}

impl CaptionActor {
    /// Main loop for the caption actor
    fn run(&mut self) {
        while let Some(cmd) = self.receiver.blocking_recv() {
            match cmd {
                CaptionCommand::GetCaption(sender) => {
                    let result = self.captions.get_livecaptions_with_retry(3);
                    let _ = sender.send(result);
                },
                CaptionCommand::CheckAvailability(sender) => {
                    let available = self.captions.is_available();
                    let _ = sender.send(available);
                },
                CaptionCommand::Shutdown(sender) => {
                    debug!("Caption actor shutting down");
                    let _ = sender.send(());
                    break;
                }
            }
        }
        debug!("Caption actor terminated");
    }
}

/// Thread-safe handle to communicate with the caption actor
#[derive(Clone)]
struct CaptionHandle {
    sender: mpsc::Sender<CaptionCommand>,
}

impl CaptionHandle {
    /// Creates a new caption actor on a dedicated thread
    /// 
    /// # Returns
    /// 
    /// * `Ok(CaptionHandle)` - The handle to communicate with the actor
    /// * `Err(AppError)` - Failed to create the caption actor
    fn new() -> Result<Self, AppError> {
        let (sender, receiver) = mpsc::channel(32);
        
        // Start the actor on a dedicated thread
        std::thread::Builder::new()
            .name("caption-actor".into())
            .spawn(move || {
                match WindowsLiveCaptions::new() {
                    Ok(captions) => {
                        let mut actor = CaptionActor {
                            captions,
                            receiver,
                        };
                        actor.run();
                    },
                    Err(e) => {
                        error!("Failed to initialize WindowsLiveCaptions: {}", e);
                    }
                }
            })
            .map_err(|e| AppError::Task(format!("Failed to spawn caption actor thread: {}", e)))?;
        
        Ok(Self { sender })
    }
    
    /// Gets captions asynchronously
    /// 
    /// # Returns
    /// 
    /// * `Ok(Some(String))` - New captions were found
    /// * `Ok(None)` - No new captions were found
    /// * `Err(AppError)` - An error occurred
    async fn get_captions(&self) -> Result<Option<String>, AppError> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send(CaptionCommand::GetCaption(sender))
            .await
            .map_err(|_| AppError::Task("Caption actor is no longer running".to_string()))?;
        
        receiver.await
            .map_err(|_| AppError::Task("Caption actor didn't respond".to_string()))?
    }
    
    /// Checks if Live Captions is available
    /// 
    /// # Returns
    /// 
    /// * `true` - Live Captions is available
    /// * `false` - Live Captions is not available
    async fn is_available(&self) -> Result<bool, AppError> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send(CaptionCommand::CheckAvailability(sender))
            .await
            .map_err(|_| AppError::Task("Caption actor is no longer running".to_string()))?;
        
        receiver.await
            .map_err(|_| AppError::Task("Caption actor didn't respond".to_string()))
    }
    
    /// Shuts down the caption actor
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The actor was shut down successfully
    /// * `Err(AppError)` - Failed to shut down the actor
    async fn shutdown(&self) -> Result<(), AppError> {
        let (sender, receiver) = oneshot::channel();
        if let Err(_) = self.sender.send(CaptionCommand::Shutdown(sender)).await {
            // Actor is already gone, that's fine
            return Ok(());
        }
        
        receiver.await
            .map_err(|_| AppError::Task("Caption actor didn't respond to shutdown command".to_string()))
    }
}

/// Translation API types supported by the application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum TranslationApiType {
    /// DeepL translation API
    DeepL,
    /// Generic translation API
    Generic,
    /// Demo mode (no actual translation)
    Demo,
    /// OpenAI and compatible APIs
    OpenAI,
}

impl Default for TranslationApiType {
    fn default() -> Self {
        TranslationApiType::DeepL
    }
}

/// Rate limiter to prevent API throttling
struct RateLimiter {
    last_request: Option<Instant>,
    min_interval: Duration,
}

impl RateLimiter {
    /// Creates a new rate limiter with the specified minimum interval
    /// 
    /// # Arguments
    /// 
    /// * `min_interval_ms` - Minimum interval between requests in milliseconds
    fn new(min_interval_ms: u64) -> Self {
        Self {
            last_request: None,
            min_interval: Duration::from_millis(min_interval_ms),
        }
    }
    
    /// Waits if necessary to respect the minimum interval
    async fn wait(&mut self) {
        if let Some(last) = self.last_request {
            let elapsed = last.elapsed();
            if elapsed < self.min_interval {
                tokio::time::sleep(self.min_interval - elapsed).await;
            }
        }
        self.last_request = Some(Instant::now());
    }
}

/// Trait defining a translation service
#[async_trait::async_trait]
trait TranslationService: Send + Sync {
    /// Translates text to the target language
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to translate
    /// 
    /// # Returns
    /// 
    /// * `Ok(String)` - The translated text
    /// * `Err(AppError)` - Translation failed
    async fn translate(&self, text: &str) -> Result<String, AppError>;
    
    /// Gets the name of the translation service
    fn get_name(&self) -> &str;
    
    /// Gets the target language code
    fn get_target_language(&self) -> &str;
}

/// Demo Translation Service implementation
struct DemoTranslation {
    target_language: String,
}

#[async_trait::async_trait]
impl TranslationService for DemoTranslation {
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        Ok(format!("[{} Translation]: {}", self.target_language, text))
    }
    
    fn get_name(&self) -> &str {
        "Demo"
    }
    
    fn get_target_language(&self) -> &str {
        &self.target_language
    }
}

/// DeepL Translation Service implementation
struct DeepLTranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    rate_limiter: tokio::sync::Mutex<RateLimiter>,
}

#[async_trait::async_trait]
impl TranslationService for DeepLTranslation {
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // DeepL API call (free version)
        let url = "https://api-free.deepl.com/v2/translate";
        
        let response = self.client.post(url)
            .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
            .form(&[
                ("text", text),
                ("target_lang", &self.target_language),
                ("source_lang", "EN"), // Assume source language is English, could be "auto"
            ])
            .send()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to send translation request to DeepL: {}", e)))?;
                
        if !response.status().is_success() {
            return Err(AppError::Translation(format!("DeepL API error: {}", response.status())));
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
            .map_err(|e| AppError::Translation(format!("Failed to parse DeepL response: {}", e)))?;
                
        if let Some(translation) = result.translations.first() {
            Ok(translation.text.clone())
        } else {
            Err(AppError::Translation("Empty translation result from DeepL".to_string()))
        }
    }
    
    fn get_name(&self) -> &str {
        "DeepL"
    }
    
    fn get_target_language(&self) -> &str {
        &self.target_language
    }
}

/// Generic Translation Service implementation
struct GenericTranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    rate_limiter: tokio::sync::Mutex<RateLimiter>,
}

#[async_trait::async_trait]
impl TranslationService for GenericTranslation {
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // Generic API call
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
            .map_err(|e| AppError::Translation(format!("Failed to send translation request: {}", e)))?;
                
        if !response.status().is_success() {
            return Err(AppError::Translation(format!("Translation API error: {}", response.status())));
        }
        
        let result: serde_json::Value = response.json()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to parse translation response: {}", e)))?;
                
        // Extract translation result based on API response format
        result.get("translated_text")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| AppError::Translation("Invalid translation response format".to_string()))
    }
    
    fn get_name(&self) -> &str {
        "Generic"
    }
    
    fn get_target_language(&self) -> &str {
        &self.target_language
    }
}

/// OpenAI Translation Service implementation
struct OpenAITranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    api_url: String,
    model: String,
    system_prompt: String,
    rate_limiter: tokio::sync::Mutex<RateLimiter>,
}

#[async_trait::async_trait]
impl TranslationService for OpenAITranslation {
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // Construct request body
        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": format!("Translate the following text to {}: {}", self.target_language, text)
                }
            ],
            "temperature": 0.3
        });
        
        // Send request
        let response = self.client.post(&self.api_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to send translation request to OpenAI compatible API: {}", e)))?;
                
        if !response.status().is_success() {
            return Err(AppError::Translation(format!("OpenAI API error: {}", response.status())));
        }
        
        // Parse response
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
            .map_err(|e| AppError::Translation(format!("Failed to parse OpenAI response: {}", e)))?;
                
        if let Some(choice) = result.choices.first() {
            Ok(choice.message.content.clone())
        } else {
            Err(AppError::Translation("Empty translation result from OpenAI".to_string()))
        }
    }
    
    fn get_name(&self) -> &str {
        "OpenAI"
    }
    
    fn get_target_language(&self) -> &str {
        &self.target_language
    }
}

/// Factory function to create the appropriate translation service
fn create_translation_service(
    api_key: String,
    target_language: String,
    api_type: TranslationApiType,
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
) -> Arc<dyn TranslationService> {
    match api_type {
        TranslationApiType::Demo => {
            Arc::new(DemoTranslation {
                target_language,
            })
        },
        TranslationApiType::DeepL => {
            Arc::new(DeepLTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(500)), // 500ms between requests
            })
        },
        TranslationApiType::Generic => {
            Arc::new(GenericTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(500)),
            })
        },
        TranslationApiType::OpenAI => {
            Arc::new(OpenAITranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                api_url: openai_api_url.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string()),
                model: openai_model.unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
                system_prompt: openai_system_prompt.unwrap_or_else(|| 
                    "You are a translator. Translate the following text to the target language. Only respond with the translation, no explanations.".to_string()
                ),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(1000)), // OpenAI might need more time between requests
            })
        },
    }
}

/// Keys for UI element caching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CacheKey {
    /// Window element cache key
    WindowElement(isize), // Window handle value as key
    /// Text element cache key
    TextElement(isize),   // Window handle value as key
}

/// Configuration file structure
#[derive(Debug, Serialize, Deserialize, Clone)]
struct Config {
    // Basic settings
    #[serde(default = "default_capture_interval")]
    capture_interval: f64,  // Capture interval (seconds)
    
    #[serde(default = "default_check_interval")]
    check_interval: u64,    // Interval to check if Live Captions is running (seconds)
    
    // Advanced settings
    #[serde(default = "default_min_interval")]
    min_interval: f64,      // Minimum capture interval (seconds)
    
    #[serde(default = "default_max_interval")]
    max_interval: f64,      // Maximum capture interval (seconds)
    
    #[serde(default = "default_max_text_length")]
    max_text_length: usize, // Maximum length of stored text
    
    // Output settings
    output_file: Option<String>, // Output file path (optional)
    
    // Translation settings
    #[serde(default)]
    enable_translation: bool,    // Whether to enable translation
    translation_api_key: Option<String>, // Translation API key (optional)
    #[serde(default)]
    translation_api_type: TranslationApiType, // Translation API type
    target_language: Option<String>,    // Target language (optional)
    
    // OpenAI related configuration
    openai_api_url: Option<String>,     // API endpoint URL
    openai_model: Option<String>,       // Model name
    openai_system_prompt: Option<String>, // System prompt
}

// Default value functions for serde
fn default_capture_interval() -> f64 { 1.0 }
fn default_check_interval() -> u64 { 10 }
fn default_min_interval() -> f64 { 0.5 }
fn default_max_interval() -> f64 { 3.0 }
fn default_max_text_length() -> usize { 10000 }

impl Config {
    /// Loads configuration from a file, creating a default if it doesn't exist
    /// 
    /// # Arguments
    /// 
    /// * `path` - Path to the configuration file
    /// 
    /// # Returns
    /// 
    /// * `Ok(Config)` - The loaded configuration
    /// * `Err(AppError)` - An error occurred while loading the configuration
    fn load(path: &PathBuf) -> Result<Self, AppError> {
        if path.exists() {
            let content = fs::read_to_string(path)
                .map_err(|e| AppError::Config(format!("Failed to read config file: {:?}: {}", path, e)))?;
            let config: Self = serde_json::from_str(&content)
                .map_err(|e| AppError::Config(format!("Failed to parse config file: {:?}: {}", path, e)))?;
            
            // Validate configuration after loading
            config.validate()?;
            
            Ok(config)
        } else {
            // If the configuration file doesn't exist, create a default configuration
            let config = Config::default();
            let content = serde_json::to_string_pretty(&config)
                .map_err(|e| AppError::Config(format!("Failed to serialize default config: {}", e)))?;
            fs::write(path, content)
                .map_err(|e| AppError::Config(format!("Failed to write default config to {:?}: {}", path, e)))?;
            info!("Created default config at {:?}", path);
            Ok(config)
        }
    }
    
    /// Saves configuration to a file
    /// 
    /// # Arguments
    /// 
    /// * `path` - Path to save the configuration file to
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The configuration was saved successfully
    /// * `Err(AppError)` - An error occurred while saving the configuration
    fn save(&self, path: &PathBuf) -> Result<(), AppError> {
        // Validate before saving
        self.validate()?;
        
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| AppError::Config(format!("Failed to serialize config: {}", e)))?;
        fs::write(path, content)
            .map_err(|e| AppError::Config(format!("Failed to write config to {:?}: {}", path, e)))?;
        info!("Saved config to {:?}", path);
        Ok(())
    }
    
    /// Validates the configuration values
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The configuration is valid
    /// * `Err(AppError)` - The configuration is invalid
    fn validate(&self) -> Result<(), AppError> {
        // Validate capture intervals
        if self.capture_interval <= 0.0 {
            return Err(AppError::Validation("Capture interval must be positive".to_string()));
        }
        
        if self.min_interval <= 0.0 {
            return Err(AppError::Validation("Minimum interval must be positive".to_string()));
        }
        
        if self.max_interval <= 0.0 {
            return Err(AppError::Validation("Maximum interval must be positive".to_string()));
        }
        
        if self.min_interval > self.max_interval {
            return Err(AppError::Validation("Minimum interval cannot be greater than maximum interval".to_string()));
        }
        
        // Validate check interval
        if self.check_interval == 0 {
            return Err(AppError::Validation("Check interval cannot be zero".to_string()));
        }
        
        // Validate text length
        if self.max_text_length == 0 {
            return Err(AppError::Validation("Maximum text length cannot be zero".to_string()));
        }
        
        // Validate translation settings
        if self.enable_translation {
            if self.translation_api_type != TranslationApiType::Demo && self.translation_api_key.is_none() {
                return Err(AppError::Validation("Translation API key is required when translation is enabled".to_string()));
            }
            
            if self.target_language.is_none() {
                return Err(AppError::Validation("Target language is required when translation is enabled".to_string()));
            }
            
            // Validate OpenAI specific settings
            if self.translation_api_type == TranslationApiType::OpenAI {
                if self.openai_model.is_none() {
                    warn!("No OpenAI model specified, will use default");
                }
            }
        }
        
        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            capture_interval: default_capture_interval(),
            check_interval: default_check_interval(),
            min_interval: default_min_interval(),
            max_interval: default_max_interval(),
            max_text_length: default_max_text_length(),
            output_file: None,
            enable_translation: false,
            translation_api_key: None,
            translation_api_type: TranslationApiType::default(),
            target_language: None,
            openai_api_url: None,
            openai_model: None,
            openai_system_prompt: None,
        }
    }
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

/// Command line arguments structure
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
    
    /// Whether to enable translation
    #[arg(short = 't', long)]
    enable_translation: Option<bool>,
    
    /// Target language for translation
    #[arg(short = 'l', long)]
    target_language: Option<String>,
    
    /// Flag to indicate this is a translation window process
    #[arg(long, hide = true)]
    translation_window: bool,
    
    /// Named pipe name for inter-process communication
    #[arg(long, hide = true)]
    pipe_name: Option<String>,
}

/// Main engine for the caption process
struct Engine {
    config: Config,
    displayed_text: String, // Text displayed in the terminal
    caption_handle: CaptionHandle,
    translation_service: Option<Arc<dyn TranslationService>>,
    consecutive_empty_captures: usize, // Count of consecutive empty captures
    adaptive_interval: f64, // Adaptive capture interval
    output_file: Option<fs::File>, // Output file handle
    translation_process: Option<std::process::Child>, // Translation window process
    translation_pipe: Option<NamedPipe>, // Named pipe for IPC with translation window
    text_buffer: String, // 文本缓冲区，用于累积文本直到形成完整句子
}

impl Engine {
    /// Creates and initializes a new engine instance
    /// 
    /// # Arguments
    /// 
    /// * `config` - The application configuration
    /// 
    /// # Returns
    /// 
    /// * `Ok(Engine)` - The initialized engine
    /// * `Err(AppError)` - An error occurred during initialization
    async fn new(config: Config) -> Result<Self, AppError> {
        debug!("Initializing engine with config: {:?}", config);
        
        // Create caption handle
        let caption_handle = CaptionHandle::new()?;
        
        // Create translation service if enabled
        let translation_service = if config.enable_translation {
            if let Some(api_key) = &config.translation_api_key {
                // Ensure target language is set
                let target_lang = config.target_language.clone().unwrap_or_else(|| {
                    match config.translation_api_type {
                        TranslationApiType::DeepL => "ZH".to_string(), // DeepL uses uppercase language codes
                        TranslationApiType::OpenAI => "Chinese".to_string(), // OpenAI uses natural language descriptions
                        _ => "zh-CN".to_string(),
                    }
                });
                
                info!("Initializing translation service with {:?} API", config.translation_api_type);
                Some(create_translation_service(
                    api_key.clone(),
                    target_lang,
                    config.translation_api_type,
                    config.openai_api_url.clone(),
                    config.openai_model.clone(),
                    config.openai_system_prompt.clone()
                ))
            } else if config.translation_api_type == TranslationApiType::Demo {
                // Demo mode doesn't need an API key
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
                warn!("Translation enabled but no API key provided");
                None
            }
        } else {
            None
        };
        
        // Create output file if configured
        let output_file = if let Some(path) = &config.output_file {
            let file = fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(path)
                .map_err(|e| AppError::Io(e))?;
            info!("Writing output to file: {}", path);
            Some(file)
        } else {
            None
        };
        
        // Create translation window if translation is enabled
        let (translation_process, translation_pipe) = if config.enable_translation {
            // Generate a unique pipe name
            let pipe_name = format!("get-livecaptions-pipe-{}", std::process::id());
            info!("Creating named pipe: {}", pipe_name);
            
            // Create named pipe server
            let pipe = NamedPipe::create_server(&pipe_name)?;
            
            // Get current executable path
            let exe_path = std::env::current_exe()
                .map_err(|e| AppError::Io(e))?;
            
            // Launch translation window process
            info!("Launching translation window process");
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
            
            // Wait for client to connect
            info!("Waiting for translation window to connect");
            pipe.wait_for_connection()?;
            info!("Translation window connected");
            
            // Send configuration to translation window
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
            text_buffer: String::new(), // 初始化文本缓冲区
        })
    }
    
    /// 检查文本是否应该被发送进行翻译
    /// 
    /// 满足以下条件之一时返回true：
    /// 1. 文本包含句子结束符（句号、问号、感叹号）
    /// 2. 文本长度超过指定阈值
    /// 3. 文本中包含段落分隔符（如逗号）且已累积足够长度
    fn should_send_for_translation(&self) -> bool {
        // 检查是否包含句子结束符
        let has_sentence_end = self.text_buffer.contains(|c| c == '.' || c == '?' || c == '!');
        
        // 检查长度是否达到阈值
        let exceeds_length_threshold = self.text_buffer.len() > 100;
        
        // 检查是否包含逗号且长度适中
        let has_comma_with_length = self.text_buffer.contains(',') && self.text_buffer.len() > 50;
        
        has_sentence_end || exceeds_length_threshold || has_comma_with_length
    }
    
    /// Main loop for the engine
    /// 
    /// This function handles the main event loop, capturing, processing,
    /// and displaying captions.
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The engine ran successfully until shutdown
    /// * `Err(AppError)` - An error occurred during the run
    async fn run(&mut self) -> Result<(), AppError> {
        info!("Starting engine main loop");
        
        // Initialize checking timer
        let mut check_timer = tokio::time::interval(Duration::from_secs(self.config.check_interval));
        let mut capture_timer = tokio::time::interval(Duration::from_secs_f64(self.config.capture_interval));
        
        // Set up Ctrl+C handler
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        
        println!("Live captions monitoring started:");
        println!("  - Capture interval: {} seconds", self.config.capture_interval);
        println!("  - Check interval: {} seconds", self.config.check_interval);
        if self.config.enable_translation {
            println!("  - Translation enabled: {}", self.translation_service.as_ref().map_or("No", |s| s.get_name()));
            println!("  - Target language: {}", self.translation_service.as_ref().map_or("None", |s| s.get_target_language()));
            if self.translation_pipe.is_some() {
                println!("  - Translation will be displayed in a separate window");
            }
        }
        if self.output_file.is_some() {
            println!("  - Writing to file: {}", self.config.output_file.as_deref().unwrap());
        }
        println!("Press Ctrl+C to exit");
        println!("-----------------------------------");
        
        // Main event loop
        loop {
            tokio::select! {
                _ = check_timer.tick() => {
                    info!("Checking if caption source is available");
                    match self.caption_handle.is_available().await {
                        Ok(available) => {
                            if !available {
                                error!("Caption source is no longer available. Program exiting.");
                                self.graceful_shutdown().await?;
                                return Err(AppError::UiAutomation("Caption source not available".to_string()));
                            }
                        },
                        Err(e) => {
                            error!("Failed to check caption source availability: {}", e);
                            self.graceful_shutdown().await?;
                            return Err(e);
                        }
                    }
                },
                _ = capture_timer.tick() => {
                    info!("Capturing live captions");
                    match self.caption_handle.get_captions().await {
                        Ok(Some(text)) => {
                            debug!("Captured new text: {}", text);
                            
                            // 将新捕获的文本添加到缓冲区
                            self.text_buffer.push_str(&text);
                            
                            // 附加文本到显示文本（仅原始文本，不包含翻译）
                            self.displayed_text.push_str(&text);
                            
                            // 检查是否有完整句子可以发送进行翻译
                            if self.should_send_for_translation() && !self.text_buffer.is_empty() {
                                // 如果翻译窗口存在，发送完整句子进行翻译
                                if let Some(pipe) = &self.translation_pipe {
                                    debug!("Sending text for translation: {}", self.text_buffer);
                                    if let Err(e) = pipe.write_message(&IpcMessage::Text(self.text_buffer.clone())) {
                                        warn!("Failed to send text to translation window: {}", e);
                                    }
                                    // 翻译后清空缓冲区
                                    self.text_buffer.clear();
                                }
                            }
                            
                            // 限制文本长度
                            self.limit_text_length();
                            
                            // 显示文本
                            Self::display_text(&self.displayed_text)?;
                            
                            // 写入到输出文件（如果配置了）
                            if let Some(file) = &mut self.output_file {
                                // 如果需要同时写入原文和译文到文件，可以在这里处理
                                if let Err(e) = writeln!(file, "{}", text) {
                                    warn!("Failed to write to output file: {}", e);
                                }
                            }
                            
                            // 重置连续空白计数和自适应间隔
                            self.consecutive_empty_captures = 0;
                            self.adaptive_interval = self.config.min_interval;
                            capture_timer = tokio::time::interval(Duration::from_secs_f64(self.adaptive_interval));
                        },
                        Ok(None) => {
                            info!("No new captions available");
                            
                            // 检查是否应该发送当前缓冲区内容
                            // 长时间没有新内容时也发送，以避免文本在缓冲区中滞留
                            if !self.text_buffer.is_empty() && self.consecutive_empty_captures > 2 {
                                if let Some(pipe) = &self.translation_pipe {
                                    debug!("Sending accumulated text for translation due to inactivity: {}", self.text_buffer);
                                    if let Err(e) = pipe.write_message(&IpcMessage::Text(self.text_buffer.clone())) {
                                        warn!("Failed to send text to translation window: {}", e);
                                    }
                                    self.text_buffer.clear();
                                }
                            }
                            
                            // 逐渐增加间隔时间
                            self.consecutive_empty_captures += 1;
                            if self.consecutive_empty_captures > 5 {
                                self.adaptive_interval = (self.adaptive_interval * 1.2).min(self.config.max_interval);
                                info!("Adjusting capture interval to {} seconds", self.adaptive_interval);
                                capture_timer = tokio::time::interval(Duration::from_secs_f64(self.adaptive_interval));
                            }
                        },
                        Err(e) => {
                            warn!("Failed to capture captions: {}", e);
                        }
                    }
                },
                _ = &mut ctrl_c => {
                    println!("\nReceived shutdown signal");
                    self.graceful_shutdown().await?;
                    info!("Program terminated successfully");
                    return Ok(());
                }
            };
        }
    }
    
    /// Limits text length to prevent excessive memory use
    /// 
    /// This method ensures that the displayed text stays within the configured
    /// maximum length by removing older sentences when necessary.
    fn limit_text_length(&mut self) {
        if self.displayed_text.len() > self.config.max_text_length {
            // Split text into sentences
            let sentences: Vec<&str> = self.displayed_text
                .split(|c| c == '.' || c == '!' || c == '?' || c == '\n')
                .filter(|s| !s.trim().is_empty())
                .collect();
            
            // Keep a fixed number of most recent sentences
            let max_sentences = 20; // Keep the last 20 sentences
            if sentences.len() > max_sentences {
                let start_idx = sentences.len() - max_sentences;
                // Recombine sentences, adding separators
                let mut new_text = String::new();
                for (i, sentence) in sentences[start_idx..].iter().enumerate() {
                    if i > 0 {
                        // Add appropriate separator
                        new_text.push_str(". ");
                    }
                    new_text.push_str(sentence.trim());
                }
                
                self.displayed_text = new_text;
                info!("Text truncated to {} sentences", max_sentences);
            }
        }
    }
    
    /// Displays text in the terminal
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to display
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The text was displayed successfully
    /// * `Err(AppError)` - An error occurred while displaying the text
    fn display_text(text: &str) -> Result<(), AppError> {
        // Clear line and display new text
        print!("\r");  // Return to start of line
        // Cover old content with spaces
        for _ in 0..120 {  // Assume terminal width is up to 120 characters
            print!(" ");
        }
        print!("\r> {}", text);
        io::stdout().flush()?;
        Ok(())
    }
    
    /// Performs a graceful shutdown
    /// 
    /// This method ensures all resources are properly released
    /// and captures any final captions before exiting.
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - Shutdown completed successfully
    /// * `Err(AppError)` - An error occurred during shutdown
    async fn graceful_shutdown(&mut self) -> Result<(), AppError> {
        info!("Performing graceful shutdown");
        
        // 发送最后的缓冲区内容
        if !self.text_buffer.is_empty() {
            if let Some(pipe) = &self.translation_pipe {
                info!("Sending final buffered text for translation: {}", self.text_buffer);
                if let Err(e) = pipe.write_message(&IpcMessage::Text(self.text_buffer.clone())) {
                    warn!("Failed to send final buffered text to translation window: {}", e);
                }
                self.text_buffer.clear();
            }
        }
        
        // Try to get final captions
        match self.caption_handle.get_captions().await {
            Ok(Some(text)) => {
                // If translation window is active, send the final text
                if let Some(pipe) = &self.translation_pipe {
                    if let Err(e) = pipe.write_message(&IpcMessage::Text(text.clone())) {
                        warn!("Failed to send final text to translation window: {}", e);
                    }
                }
                
                // Append to displayed text
                self.displayed_text.push_str(&text);
                
                // Limit text length
                self.limit_text_length();
                
                info!("Final captions captured: {}", text);
            },
            Ok(None) => {
                info!("No new captions at shutdown");
            },
            Err(err) => {
                warn!("Could not capture final captions: {}", err);
            }
        }
        
        // Display final text
        if !self.displayed_text.is_empty() {
            println!("\n");
            print!("> {}", self.displayed_text);
            io::stdout().flush()?;
        }
        
        // Shut down the translation window if it exists
        if let Some(pipe) = &self.translation_pipe {
            info!("Shutting down translation window");
            if let Err(e) = pipe.write_message(&IpcMessage::Shutdown) {
                warn!("Failed to send shutdown message to translation window: {}", e);
            }
            
            // Give the translation window some time to shut down
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        
        // Shut down the caption handle
        if let Err(e) = self.caption_handle.shutdown().await {
            warn!("Error shutting down caption actor: {}", e);
        }
        
        info!("Shutdown complete");
        Ok(())
    }
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
    
    // 用于显示的文本
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
                        // 接收到完整句子，直接翻译
                        debug!("收到文本进行翻译: {}", text);
                        if !text.is_empty() {
                            match translation_service.translate(&text).await {
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
                                    
                                    // 显示翻译文本
                                    display_translation_window(&displayed_text)?;
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

/// Creates and configures an engine based on command line arguments and config file
async fn create_engine() -> Result<Engine, AppError> {
    // Parse command line arguments
    let args = Args::parse();
    info!("get-livecaptions starting");
    
    // Determine config path
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
    
    // Load or create configuration
    let mut config = Config::load(&config_path)?;
    
    // Override config with command line arguments
    if let Some(interval) = args.capture_interval {
        config.capture_interval = interval;
    }
    if let Some(check) = args.check_interval {
        config.check_interval = check;
    }
    if let Some(output) = args.output_file {
        config.output_file = Some(output);
    }
    if let Some(enable) = args.enable_translation {
        config.enable_translation = enable;
    }
    if let Some(lang) = args.target_language {
        config.target_language = Some(lang);
    }
    
    // Validate the configuration
    config.validate()?;
    
    // Save updated configuration
    config.save(&config_path)?;
    
    // Create and initialize the engine
    Engine::new(config).await
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logger
    env_logger::init();
    
    // Parse command line arguments
    let args = Args::parse();
    
    // Check if this is a translation window process
    if args.translation_window {
        if let Some(pipe_name) = args.pipe_name {
            // Run as translation window
            return run_translation_window(pipe_name).await.map_err(|e| anyhow::anyhow!(e));
        } else {
            return Err(anyhow::anyhow!("Translation window requires pipe name"));
        }
    }
    
    // Create engine
    let mut engine = create_engine().await?;
    
    // Run the engine
    engine.run().await?;
    
    Ok(())
}

/// Tests module
#[cfg(test)]
mod tests {
    use super::*;
    
    /// Tests for text difference detection
    #[test]
    fn test_extract_new_text() {
        // Create a caption source instance for testing
        let captions = WindowsLiveCaptions {
            automation: unsafe { std::mem::zeroed() }, // Not used in the test
            condition: unsafe { std::mem::zeroed() },  // Not used in the test
            previous_text: "Hello world".to_string(),
            element_cache: LruCache::new(NonZeroUsize::new(1).unwrap()),
            last_window_handle: HWND(0),
            max_retries: 1,
        };
        
        // Simple append test
        let result = captions.extract_new_text("Hello world, how are you?");
        assert_eq!(result, ", how are you?");
    }
    
    /// Tests for configuration validation
    #[test]
    fn test_config_validation() {
        // Test valid configuration
        let config = Config {
            capture_interval: 1.0,
            check_interval: 10,
            min_interval: 0.5,
            max_interval: 3.0,
            max_text_length: 10000,
            output_file: None,
            enable_translation: false,
            translation_api_key: None,
            translation_api_type: TranslationApiType::DeepL,
            target_language: None,
            openai_api_url: None,
            openai_model: None,
            openai_system_prompt: None,
        };
        assert!(config.validate().is_ok());
        
        // Test invalid intervals
        let mut invalid_config = config.clone();
        invalid_config.min_interval = 5.0;
        invalid_config.max_interval = 3.0;
        assert!(invalid_config.validate().is_err());
    }
}