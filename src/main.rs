use tokio::time::{Duration, Instant};
use std::process;
use std::path::PathBuf;
use std::sync::Arc;
use windows::{
    core::*, Win32::System::Com::*, Win32::UI::{Accessibility::*, WindowsAndMessaging::*}, Win32::Foundation::HWND,
};
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
use async_trait::async_trait;
use tokio::sync::{mpsc, Mutex};
use tokio::task;
use std::collections::HashMap;

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
/// 
/// # Components
/// 
/// - Error handling: Custom error types for detailed error reporting
/// - Translation services: Multiple translation service implementations
/// - UI automation: Windows UI Automation interfacing with caching
/// - Configuration: File-based JSON configuration with command-line overrides
/// - Concurrency: Asynchronous design with worker tasks

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
}

// Implement From for windows errors
impl From<windows::core::Error> for AppError {
    fn from(err: windows::core::Error) -> Self {
        AppError::Com(format!("{:?}", err))
    }
}

/// Trait that defines a source of captions
/// 
/// This abstraction allows for different implementations of caption sources,
/// not just Windows Live Captions.
#[async_trait]
trait CaptionSource: Send + Sync {
    /// Initialize the caption source
    async fn initialize(&mut self) -> Result<(), AppError>;
    
    /// Check if the caption source is available
    async fn is_available(&self) -> bool;
    
    /// Get captions from the source
    /// 
    /// Returns None if no new captions are available
    async fn get_captions(&mut self) -> Result<Option<String>, AppError>;
    
    /// Shutdown the caption source
    async fn shutdown(&mut self) -> Result<(), AppError>;
}

/// Windows Live Captions implementation of the CaptionSource trait
struct WindowsLiveCaptions {
    automation: IUIAutomation,
    condition: IUIAutomationCondition,
    previous_text: String,
    element_cache: LruCache<CacheKey, IUIAutomationElement>,
    last_window_handle: HWND,
    max_retries: usize,
}

impl Drop for WindowsLiveCaptions {
    fn drop(&mut self) {
        unsafe { CoUninitialize(); }
        info!("COM resources released");
    }
}

#[async_trait]
impl CaptionSource for WindowsLiveCaptions {
    async fn initialize(&mut self) -> Result<(), AppError> {
        debug!("Initializing Windows Live Captions source");
        Ok(())
    }
    
    async fn is_available(&self) -> bool {
        unsafe { FindWindowW(w!("LiveCaptionsDesktopWindow"), None).0 != 0 }
    }
    
    async fn get_captions(&mut self) -> Result<Option<String>, AppError> {
        self.get_livecaptions_with_retry(self.max_retries).await
    }
    
    async fn shutdown(&mut self) -> Result<(), AppError> {
        debug!("Shutting down Windows Live Captions source");
        Ok(())
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
            let hr = CoInitializeEx(None, COINIT_MULTITHREADED);
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
    fn extract_new_text<'a>(&self, current: &'a str) -> String {
        if self.previous_text.is_empty() {
            return current.to_string();
        }
        
        // Use similar library to compute differences
        let diff = TextDiff::from_chars(&self.previous_text, current);
        
        // Extract only the inserted parts
        let mut new_text = String::new();
        for change in diff.iter_all_changes() {
            if change.tag() == ChangeTag::Insert {
                new_text.push(change.value().chars().next().unwrap_or(' '));
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
    async fn get_livecaptions(&mut self) -> Result<Option<String>, AppError> {
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
    async fn get_livecaptions_with_retry(&mut self, max_retries: usize) -> Result<Option<String>, AppError> {
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
                        // Exponential backoff
                        let backoff_ms = 100 * 2u64.pow(attempts as u32);
                        debug!("Retrying in {} ms", backoff_ms);
                        tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap_or_else(|| AppError::UiAutomation("Unknown error after retries".to_string())))
    }
}

/// Define a trait for translation services
#[async_trait]
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

/// DeepL Translation Service implementation
struct DeepLTranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    rate_limiter: Mutex<RateLimiter>,
}

/// Generic Translation Service implementation
struct GenericTranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    rate_limiter: Mutex<RateLimiter>,
}

/// OpenAI Translation Service implementation
struct OpenAITranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    api_url: String,
    model: String,
    system_prompt: String,
    rate_limiter: Mutex<RateLimiter>,
}

/// Demo Translation Service implementation
struct DemoTranslation {
    target_language: String,
}

#[async_trait]
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

#[async_trait]
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

#[async_trait]
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

#[async_trait]
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
                rate_limiter: Mutex::new(RateLimiter::new(500)), // 500ms between requests
            })
        },
        TranslationApiType::Generic => {
            Arc::new(GenericTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: Mutex::new(RateLimiter::new(500)),
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
                rate_limiter: Mutex::new(RateLimiter::new(1000)), // OpenAI might need more time between requests
            })
        },
    }
}

/// Configuration file structure
#[derive(Debug, Serialize, Deserialize)]
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
    
    // Concurrency settings
    #[serde(default = "default_worker_threads")]
    worker_threads: usize,       // Number of worker threads for processing
    
    // UI automation settings
    #[serde(default = "default_caption_source")]
    caption_source: String,      // Caption source type
    
    // Custom window settings (for non-Windows Live Captions sources)
    custom_window_class: Option<String>,  // Custom window class name
    custom_window_name: Option<String>,   // Custom window name
    custom_element_id: Option<String>,    // Custom element ID
}

// Default value functions for serde
fn default_capture_interval() -> f64 { 1.0 }
fn default_check_interval() -> u64 { 10 }
fn default_min_interval() -> f64 { 0.5 }
fn default_max_interval() -> f64 { 3.0 }
fn default_max_text_length() -> usize { 10000 }
fn default_worker_threads() -> usize { 2 }
fn default_caption_source() -> String { "windows_live_captions".to_string() }

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
        
        // Validate worker threads
        if self.worker_threads == 0 {
            return Err(AppError::Validation("Worker threads cannot be zero".to_string()));
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
        
        // Validate caption source
        if self.caption_source.is_empty() {
            return Err(AppError::Validation("Caption source cannot be empty".to_string()));
        }
        
        // Validate custom window settings if needed
        if self.caption_source == "custom" {
            if self.custom_window_class.is_none() && self.custom_window_name.is_none() {
                return Err(AppError::Validation("Custom caption source requires either window class or window name".to_string()));
            }
            
            if self.custom_element_id.is_none() {
                return Err(AppError::Validation("Custom caption source requires element ID".to_string()));
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
            worker_threads: default_worker_threads(),
            caption_source: default_caption_source(),
            custom_window_class: None,
            custom_window_name: None,
            custom_element_id: None,
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
    
    /// Number of worker threads
    #[arg(short = 'w', long)]
    worker_threads: Option<usize>,
    
    /// Caption source (windows_live_captions, custom)
    #[arg(short = 's', long)]
    caption_source: Option<String>,
}

/// Keys for UI element caching
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum CacheKey {
    /// Window element cache key
    WindowElement(isize), // Window handle value as key
    /// Text element cache key
    TextElement(isize),   // Window handle value as key
}

/// Main engine for the caption process
struct Engine {
    config: Config,
    displayed_text: String, // Text displayed in the terminal
    caption_source: Arc<Mutex<dyn CaptionSource + Send + Sync>>,
    translation_service: Option<Arc<dyn TranslationService>>,
    consecutive_empty_captures: usize, // Count of consecutive empty captures
    adaptive_interval: f64, // Adaptive capture interval
    output_file: Option<fs::File>, // Output file handle
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
        
        // Create caption source based on configuration
        let caption_source: Arc<Mutex<dyn CaptionSource + Send + Sync>> = match config.caption_source.as_str() {
            "windows_live_captions" => {
                let source = WindowsLiveCaptions::new()?;
                Arc::new(Mutex::new(source))
            },
            "custom" => {
                return Err(AppError::Config("Custom caption source not yet implemented".to_string()));
            },
            _ => {
                return Err(AppError::Config(format!("Unknown caption source: {}", config.caption_source)));
            }
        };
        
        // Initialize the caption source
        caption_source.lock().await.initialize().await?;
        
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
                .with_context(|| format!("Failed to open output file: {}", path))?;
            info!("Writing output to file: {}", path);
            Some(file)
        } else {
            None
        };
        
        Ok(Self {
            displayed_text: String::new(),
            caption_source,
            translation_service,
            consecutive_empty_captures: 0,
            adaptive_interval: config.min_interval,
            output_file,
            config,
        })
    }
    
    /// Main loop for the engine
    /// 
    /// This function handles the main event loop and coordinates
    /// the capture, processing, and output tasks.
    /// 
    /// # Returns
    /// 
    /// * `Ok(())` - The engine ran successfully until shutdown
    /// * `Err(AppError)` - An error occurred during the run
    async fn run(&mut self) -> Result<(), AppError> {
        info!("Starting engine main loop");
        
        // Create channels for communication between tasks
        let (caption_tx, mut caption_rx) = mpsc::channel::<String>(32);
        let (processed_tx, mut processed_rx) = mpsc::channel::<String>(32);
        
        // Clone references for the worker tasks
        let caption_source = self.caption_source.clone();
        let translation_service = self.translation_service.clone();
        let config = self.config.clone();
        
        // Spawn caption capture task
        let capture_handle = task::spawn(async move {
            let mut adaptive_interval = config.min_interval;
            let mut consecutive_empty = 0;
            
            let mut capture_timer = tokio::time::interval(Duration::from_secs_f64(config.capture_interval));
            
            loop {
                // Wait for the next tick
                capture_timer.tick().await;
                
                // Get captions
                let mut source = caption_source.lock().await;
                match source.get_captions().await {
                    Ok(Some(text)) => {
                        debug!("Captured new text: {}", text);
                        if caption_tx.send(text).await.is_err() {
                            warn!("Failed to send captured text to processor");
                            break;
                        }
                        
                        // Reset consecutive empty counter and adaptive interval
                        consecutive_empty = 0;
                        adaptive_interval = config.min_interval;
                        capture_timer = tokio::time::interval(Duration::from_secs_f64(adaptive_interval));
                    },
                    Ok(None) => {
                        debug!("No new captions available");
                        // Gradually increase interval on consecutive empty captures
                        consecutive_empty += 1;
                        if consecutive_empty > 5 {
                            adaptive_interval = (adaptive_interval * 1.2).min(config.max_interval);
                            debug!("Adjusting capture interval to {} seconds", adaptive_interval);
                            capture_timer = tokio::time::interval(Duration::from_secs_f64(adaptive_interval));
                        }
                    },
                    Err(e) => {
                        warn!("Failed to capture captions: {}", e);
                        // Don't break on errors, continue trying
                    }
                }
                
                // Check if source is still available
                if !source.is_available().await {
                    error!("Caption source is no longer available");
                    break;
                }
            }
        });
        
        // Spawn processor tasks
        let mut processor_handles = Vec::new();
        for id in 0..self.config.worker_threads {
            let processor_tx = processed_tx.clone();
            let mut processor_rx = caption_rx.clone();
            let translation = translation_service.clone();
            
            let handle = task::spawn(async move {
                debug!("Starting processor task {}", id);
                while let Some(text) = processor_rx.recv().await {
                    // Process text (translate if enabled)
                    let processed = match &translation {
                        Some(service) => {
                            match service.translate(&text).await {
                                Ok(translated) => {
                                    debug!("Translated text: {} -> {}", text, translated);
                                    format!("{} [{}]", text, translated)
                                },
                                Err(e) => {
                                    warn!("Translation failed: {}", e);
                                    text
                                }
                            }
                        },
                        None => text,
                    };
                    
                    // Send processed text
                    if processor_tx.send(processed).await.is_err() {
                        warn!("Failed to send processed text to output");
                        break;
                    }
                }
                debug!("Processor task {} exiting", id);
            });
            
            processor_handles.push(handle);
        }
        
        // Initialize checking timer
        let mut check_timer = tokio::time::interval(Duration::from_secs(self.config.check_interval));
        
        // Set up Ctrl+C handler
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        
        println!("Live captions monitoring started:");
        println!("  - Capture interval: {} seconds", self.config.capture_interval);
        println!("  - Check interval: {} seconds", self.config.check_interval);
        if self.config.enable_translation {
            println!("  - Translation enabled: {}", self.translation_service.as_ref().map_or("No", |s| s.get_name()));
            println!("  - Target language: {}", self.translation_service.as_ref().map_or("None", |s| s.get_target_language()));
        }
        if self.output_file.is_some() {
            println!("  - Writing to file: {}", self.config.output_file.as_deref().unwrap());
        }
        println!("  - Worker threads: {}", self.config.worker_threads);
        println!("Press Ctrl+C to exit");
        println!("-----------------------------------");
        
        // Main event loop
        loop {
            tokio::select! {
                _ = check_timer.tick() => {
                    info!("Checking if caption source is available");
                    let source = self.caption_source.lock().await;
                    if !source.is_available().await {
                        error!("Caption source is no longer available. Program exiting.");
                        drop(source); // Release the lock before shutdown
                        self.graceful_shutdown().await?;
                        return Err(AppError::UiAutomation("Caption source not available".to_string()));
                    }
                },
                Some(processed_text) = processed_rx.recv() => {
                    // Append processed text to displayed text
                    self.displayed_text.push_str(&processed_text);
                    
                    // Limit text length
                    self.limit_text_length();
                    
                    // Display text
                    Self::display_text(&self.displayed_text)?;
                    
                    // Write to output file if configured
                    if let Some(file) = &mut self.output_file {
                        if let Err(e) = writeln!(file, "{}", processed_text) {
                            warn!("Failed to write to output file: {}", e);
                        }
                    }
                },
                _ = &mut ctrl_c => {
                    println!("\nReceived shutdown signal");
                    self.graceful_shutdown().await?;
                    
                    // Cancel tasks
                    capture_handle.abort();
                    for handle in processor_handles {
                        handle.abort();
                    }
                    
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
        
        // Try to get final captions
        let mut source = self.caption_source.lock().await;
        match source.get_captions().await {
            Ok(Some(text)) => {
                // Process the final caption if possible
                let final_text = if let Some(service) = &self.translation_service {
                    match service.translate(&text).await {
                        Ok(translated) => format!("{} [{}]", text, translated),
                        Err(_) => text,
                    }
                } else {
                    text
                };
                
                // Append to displayed text
                self.displayed_text.push_str(&final_text);
                
                // Limit text length
                self.limit_text_length();
                
                info!("Final captions captured: {}", final_text);
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
        
        // Shut down the caption source
        source.shutdown().await?;
        
        info!("Shutdown complete");
        Ok(())
    }
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
    let mut config = Config::load(&config_path)
        .context("Failed to load config")?;
    
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
    if let Some(threads) = args.worker_threads {
        config.worker_threads = threads;
    }
    if let Some(source) = args.caption_source {
        config.caption_source = source;
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
    #[tokio::test]
    async fn test_extract_new_text() {
        // Create a caption source instance
        let mut captions = WindowsLiveCaptions::new().unwrap();
        
        // Simple append test
        captions.previous_text = "Hello world".to_string();
        let result = captions.extract_new_text("Hello world, how are you?");
        assert_eq!(result, ", how are you?");
        
        // Empty previous text test
        captions.previous_text = "".to_string();
        let result = captions.extract_new_text("New text");
        assert_eq!(result, "New text");
        
        // No changes test
        captions.previous_text = "Unchanged text".to_string();
        let result = captions.extract_new_text("Unchanged text");
        assert_eq!(result, "");
        
        // Complex change test
        captions.previous_text = "This is a test with some text".to_string();
        let result = captions.extract_new_text("This is a new test with additional text");
        assert_eq!(result, "new additional");
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
            worker_threads: 2,
            caption_source: "windows_live_captions".to_string(),
            custom_window_class: None,
            custom_window_name: None,
            custom_element_id: None,
        };
        assert!(config.validate().is_ok());
        
        // Test invalid intervals
        let mut invalid_config = config.clone();
        invalid_config.min_interval = 5.0;
        invalid_config.max_interval = 3.0;
        assert!(invalid_config.validate().is_err());
        
        // Test invalid worker threads
        let mut invalid_config = config.clone();
        invalid_config.worker_threads = 0;
        assert!(invalid_config.validate().is_err());
        
        // Test missing translation API key
        let mut invalid_config = config.clone();
        invalid_config.enable_translation = true;
        invalid_config.translation_api_type = TranslationApiType::DeepL;
        invalid_config.translation_api_key = None;
        assert!(invalid_config.validate().is_err());
        
        // Test missing target language
        let mut invalid_config = config.clone();
        invalid_config.enable_translation = true;
        invalid_config.translation_api_type = TranslationApiType::Demo;
        invalid_config.target_language = None;
        assert!(invalid_config.validate().is_err());
        
        // Test custom caption source validation
        let mut invalid_config = config.clone();
        invalid_config.caption_source = "custom".to_string();
        invalid_config.custom_window_class = None;
        invalid_config.custom_window_name = None;
        assert!(invalid_config.validate().is_err());
    }
    
    /// Tests for demo translation service
    #[tokio::test]
    async fn test_demo_translation() {
        let service = DemoTranslation {
            target_language: "Spanish".to_string(),
        };
        
        let result = service.translate("Hello world").await.unwrap();
        assert_eq!(result, "[Spanish Translation]: Hello world");
        assert_eq!(service.get_name(), "Demo");
        assert_eq!(service.get_target_language(), "Spanish");
    }
}