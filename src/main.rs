use tokio::time::{Duration, Instant};
use std::path::PathBuf;
use std::sync::Arc;
use windows::{
    core::*, Win32::System::Com::*, Win32::UI::{Accessibility::*, WindowsAndMessaging::*}, 
    Win32::Foundation::{HWND, HANDLE, CloseHandle},
    Win32::System::Pipes::*,
    Win32::Storage::FileSystem::{CreateFileW, FILE_SHARE_NONE, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, WriteFile, ReadFile},
    Win32::Security::*,
};
use windows::Win32::Foundation::{GENERIC_READ, GENERIC_WRITE};
use clap::Parser;
use log::{debug, error, info, warn};
use thiserror::Error;
use anyhow::Result;
use std::io::{self, Write};
use lru::LruCache;
use std::num::NonZeroUsize;
use std::fs;
use serde::{Deserialize, Serialize};
use similar::{ChangeTag, TextDiff};
use tokio::sync::{mpsc, oneshot};
use std::process::{Command, Stdio};
use std::os::windows::process::CommandExt;
use std::collections::{HashMap, BTreeMap, HashSet};

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
    // Activity tracking for ultra-responsive mode
    last_activity: Instant,
    active_mode: bool,
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
            last_activity: Instant::now(),
            active_mode: false,
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
        
        // If we have new content, update activity state
        if !new_text.is_empty() {
            self.last_activity = Instant::now();
            self.active_mode = true;
        }
        
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
    
    /// Checks if we're in active caption mode
    fn is_active(&self, timeout_sec: f64) -> bool {
        if !self.active_mode {
            return false;
        }
        
        // Check if we've seen activity within the timeout period
        self.last_activity.elapsed().as_secs_f64() < timeout_sec
    }
}

// Message types for caption worker thread
enum CaptionCommand {
    GetCaption(oneshot::Sender<Result<Option<String>, AppError>>),
    CheckAvailability(oneshot::Sender<bool>),
    IsActive(oneshot::Sender<bool>, f64),
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
                CaptionCommand::IsActive(sender, timeout) => {
                    let active = self.captions.is_active(timeout);
                    let _ = sender.send(active);
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
    
    /// Checks if captions are active (recently seen)
    /// 
    /// # Arguments
    /// 
    /// * `timeout_sec` - How long after last caption to consider still active
    /// 
    /// # Returns
    /// 
    /// * `Ok(bool)` - Whether captions are active
    /// * `Err(AppError)` - An error occurred
    async fn is_active(&self, timeout_sec: f64) -> Result<bool, AppError> {
        let (sender, receiver) = oneshot::channel();
        self.sender.send(CaptionCommand::IsActive(sender, timeout_sec))
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

/// Translation cache for storing previous translations
struct TranslationCache {
    // Using an LRU cache with default capacity of 1000 entries
    cache: LruCache<String, String>,
    hits: usize,
    misses: usize,
}

impl TranslationCache {
    /// Creates a new translation cache with the specified capacity
    fn new(capacity: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(capacity).unwrap_or(NonZeroUsize::new(1000).unwrap())),
            hits: 0,
            misses: 0,
        }
    }

    /// Gets a translation from the cache
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to look up
    /// 
    /// # Returns
    /// 
    /// * `Some(String)` - The translation was found in the cache
    /// * `None` - The translation was not found in the cache
    fn get(&mut self, text: &str) -> Option<String> {
        if let Some(translation) = self.cache.get(text) {
            self.hits += 1;
            Some(translation.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    /// Puts a translation in the cache
    /// 
    /// # Arguments
    /// 
    /// * `text` - The original text
    /// * `translation` - The translated text
    fn put(&mut self, text: String, translation: String) {
        self.cache.put(text, translation);
    }

    /// Adds multiple translations to the cache at once
    /// 
    /// # Arguments
    /// 
    /// * `texts` - The original texts
    /// * `translations` - The translated texts
    fn put_batch(&mut self, texts: Vec<String>, translations: Vec<String>) {
        for (text, translation) in texts.into_iter().zip(translations.into_iter()) {
            self.cache.put(text, translation);
        }
    }

    /// Gets the hit rate of the cache
    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Gets stats for the cache
    fn stats(&self) -> (usize, usize, usize, f64) {
        (self.cache.len(), self.hits, self.misses, self.hit_rate())
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
    
    /// Batch translates multiple texts
    /// 
    /// # Arguments
    /// 
    /// * `texts` - The texts to translate
    /// 
    /// # Returns
    /// 
    /// * `Ok(Vec<String>)` - The translated texts
    /// * `Err(AppError)` - Translation failed
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        // Default implementation translates each text individually
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(self.translate(text).await?);
        }
        Ok(results)
    }
    
    /// Gets the name of the translation service
    fn get_name(&self) -> &str;
    
    /// Gets the target language code
    fn get_target_language(&self) -> &str;
}

/// Batch processor for translation requests
struct BatchTranslationProcessor {
    service: Arc<dyn TranslationService>,
    cache: Arc<tokio::sync::Mutex<TranslationCache>>,
    queue: Arc<tokio::sync::Mutex<HashMap<String, Vec<oneshot::Sender<Result<String, AppError>>>>>>,
    batch_size: usize,
    max_batch_delay: Duration,
    stats: Arc<tokio::sync::Mutex<BatchStats>>,
}

/// Statistics for batch processing
#[derive(Debug, Default)]
struct BatchStats {
    requests: usize,
    batches: usize,
    cache_hits: usize,
    batch_sizes: Vec<usize>,
}

impl BatchTranslationProcessor {
    /// Creates a new batch processor
    fn new(
        service: Arc<dyn TranslationService>,
        cache_size: usize,
        batch_size: usize,
        max_batch_delay_ms: u64,
    ) -> Self {
        let processor = Self {
            service,
            cache: Arc::new(tokio::sync::Mutex::new(TranslationCache::new(cache_size))),
            queue: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            batch_size,
            max_batch_delay: Duration::from_millis(max_batch_delay_ms),
            stats: Arc::new(tokio::sync::Mutex::new(BatchStats::default())),
        };
        
        // Start the batch processing loop
        processor.start_processing_loop();
        
        processor
    }
    
    /// Starts the background processing loop
    fn start_processing_loop(&self) {
        let service = self.service.clone();
        let queue = self.queue.clone();
        let cache = self.cache.clone();
        let batch_size = self.batch_size;
        let max_batch_delay = self.max_batch_delay;
        let stats = self.stats.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(max_batch_delay);
            
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        // Process any pending translations
                        let should_process = {
                            let queue_guard = queue.lock().await;
                            !queue_guard.is_empty()
                        };
                        
                        if should_process {
                            if let Err(e) = Self::process_batch(&service, &queue, &cache, &stats).await {
                                warn!("Failed to process translation batch: {}", e);
                            }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(10)) => {
                        // Check if we have enough translations to process
                        let should_process = {
                            let queue_guard = queue.lock().await;
                            let queue_size: usize = queue_guard.values().map(|v| v.len()).sum();
                            queue_size >= batch_size
                        };
                        
                        if should_process {
                            if let Err(e) = Self::process_batch(&service, &queue, &cache, &stats).await {
                                warn!("Failed to process translation batch: {}", e);
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Processes a batch of translation requests
    async fn process_batch(
        service: &Arc<dyn TranslationService>,
        queue: &Arc<tokio::sync::Mutex<HashMap<String, Vec<oneshot::Sender<Result<String, AppError>>>>>>,
        cache: &Arc<tokio::sync::Mutex<TranslationCache>>,
        stats: &Arc<tokio::sync::Mutex<BatchStats>>,
    ) -> Result<(), AppError> {
        // Take the current queue
        let mut current_queue = {
            let mut queue_guard = queue.lock().await;
            std::mem::take(&mut *queue_guard)
        };
        
        if current_queue.is_empty() {
            return Ok(());
        }
        
        // Collect unique texts to translate
        let mut texts_to_translate = Vec::new();
        let mut text_to_senders = HashMap::new();
        
        // Check cache first
        {
            let mut cache_guard = cache.lock().await;
            
            for (text, mut senders) in current_queue.drain() {
                if let Some(cached_translation) = cache_guard.get(&text) {
                    // Cache hit, respond immediately
                    for sender in senders {
                        let _ = sender.send(Ok(cached_translation.clone()));
                    }
                    
                    // Update stats
                    let mut stats_guard = stats.lock().await;
                    stats_guard.cache_hits += senders.len();
                } else if !senders.is_empty() {
                    // Cache miss, add to translation batch
                    if !text_to_senders.contains_key(&text) {
                        texts_to_translate.push(text.clone());
                    }
                    text_to_senders.insert(text, senders);
                }
            }
        }
        
        // Translate batch
        if !texts_to_translate.is_empty() {
            {
                // Update stats
                let mut stats_guard = stats.lock().await;
                stats_guard.batches += 1;
                stats_guard.batch_sizes.push(texts_to_translate.len());
            }
            
            debug!("Processing batch of {} translations", texts_to_translate.len());
            
            let translations = service.translate_batch(&texts_to_translate).await?;
            
            // Cache results and respond to senders
            let mut cache_guard = cache.lock().await;
            
            for (i, text) in texts_to_translate.iter().enumerate() {
                if i < translations.len() {
                    let translation = translations[i].clone();
                    
                    // Cache the result
                    cache_guard.put(text.clone(), translation.clone());
                    
                    // Respond to all senders for this text
                    if let Some(senders) = text_to_senders.remove(text) {
                        for sender in senders {
                            let _ = sender.send(Ok(translation.clone()));
                        }
                    }
                } else {
                    // This shouldn't happen, but handle it anyway
                    if let Some(senders) = text_to_senders.remove(text) {
                        for sender in senders {
                            let _ = sender.send(Err(AppError::Translation("Batch translation returned fewer results than expected".to_string())));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Translates text with caching and batching
    /// 
    /// # Arguments
    /// 
    /// * `text` - The text to translate
    /// 
    /// # Returns
    /// 
    /// * `Ok(String)` - The translated text
    /// * `Err(AppError)` - Translation failed
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Update stats
        {
            let mut stats_guard = self.stats.lock().await;
            stats_guard.requests += 1;
        }
        
        // Check cache first
        {
            let mut cache_guard = self.cache.lock().await;
            if let Some(cached) = cache_guard.get(text) {
                // Update stats
                let mut stats_guard = self.stats.lock().await;
                stats_guard.cache_hits += 1;
                
                return Ok(cached);
            }
        }
        
        // Create a channel for the result
        let (sender, receiver) = oneshot::channel();
        
        // Add to queue
        {
            let mut queue_guard = self.queue.lock().await;
            queue_guard.entry(text.to_string())
                .or_insert_with(Vec::new)
                .push(sender);
        }
        
        // Wait for result
        receiver.await.map_err(|_| AppError::Translation("Translation task cancelled".to_string()))?
    }
    
    /// Gets statistics about the batch processor
    async fn get_stats(&self) -> (usize, usize, usize, Vec<usize>, f64) {
        let stats_guard = self.stats.lock().await;
        let cache_guard = self.cache.lock().await;
        
        let (_, _, _, hit_rate) = cache_guard.stats();
        
        (
            stats_guard.requests,
            stats_guard.batches,
            stats_guard.cache_hits,
            stats_guard.batch_sizes.clone(),
            hit_rate,
        )
    }
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
    
    // Override the default batch implementation with a more efficient one
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        let mut results = Vec::with_capacity(texts.len());
        for text in texts {
            results.push(format!("[{} Translation]: {}", self.target_language, text));
        }
        Ok(results)
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
    
    // Implement batch translation for DeepL
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // DeepL API supports batch translation natively
        let url = "https://api-free.deepl.com/v2/translate";
        
        // Build form data with multiple text entries
        let mut form = Vec::new();
        for text in texts {
            form.push(("text", text.as_str()));
        }
        form.push(("target_lang", self.target_language.as_str()));
        form.push(("source_lang", "EN"));
        
        let response = self.client.post(url)
            .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
            .form(&form)
            .send()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to send batch translation request to DeepL: {}", e)))?;
                
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
            .map_err(|e| AppError::Translation(format!("Failed to parse DeepL batch response: {}", e)))?;
        
        if result.translations.len() != texts.len() {
            return Err(AppError::Translation(format!("DeepL returned {} translations for {} texts", 
                                                    result.translations.len(), texts.len())));
        }
        
        // Extract translations
        let translations = result.translations.into_iter()
            .map(|t| t.text)
            .collect();
        
        Ok(translations)
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
    
    // Implement batch translation if the API supports it
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // Generic batch API call
        let url = "https://translation-api.example.com/translate/batch";
        
        let response = self.client.post(url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "texts": texts,
                "source_language": "auto",
                "target_language": self.target_language
            }))
            .send()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to send batch translation request: {}", e)))?;
                
        if !response.status().is_success() {
            return Err(AppError::Translation(format!("Translation API error: {}", response.status())));
        }
        
        let result: serde_json::Value = response.json()
            .await
            .map_err(|e| AppError::Translation(format!("Failed to parse batch translation response: {}", e)))?;
        
        // Extract translations array from the response
        let translations = result.get("translations")
            .and_then(|v| v.as_array())
            .ok_or_else(|| AppError::Translation("Invalid batch translation response format".to_string()))?;
        
        if translations.len() != texts.len() {
            return Err(AppError::Translation(format!("Received {} translations for {} texts", 
                                                   translations.len(), texts.len())));
        }
        
        // Extract translated texts
        let mut results = Vec::with_capacity(translations.len());
        for translation in translations {
            let translated_text = translation.get("translated_text")
                .and_then(|v| v.as_str())
                .ok_or_else(|| AppError::Translation("Invalid translation entry in response".to_string()))?;
            
            results.push(translated_text.to_string());
        }
        
        Ok(results)
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
    
    // Implement batch translation for OpenAI
    // Since OpenAI doesn't have a native batch endpoint, we'll use a delimiter approach
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // Use a special delimiter unlikely to appear in the text
        const DELIMITER: &str = "|||SPLIT|||";
        
        // Join texts with delimiter
        let combined_text = texts.join(&format!(" {} ", DELIMITER));
        
        // Construct request body
        let request_body = serde_json::json!({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": format!("{} 
                    You will be given multiple texts separated by the delimiter '{}'. 
                    Translate each text to {} and keep them separated by the same delimiter.", 
                    self.system_prompt, DELIMITER, self.target_language)
                },
                {
                    "role": "user",
                    "content": combined_text
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
            .map_err(|e| AppError::Translation(format!("Failed to send batch translation request to OpenAI compatible API: {}", e)))?;
                
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
            // Split response by delimiter
            let translated_texts: Vec<String> = choice.message.content
                .split(DELIMITER)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            
            // Verify we got the expected number of translations
            if translated_texts.len() != texts.len() {
                return Err(AppError::Translation(format!(
                    "Received {} translations for {} texts", translated_texts.len(), texts.len()
                )));
            }
            
            Ok(translated_texts)
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

/// Factory function to create a batch translation processor
fn create_batch_translation_processor(
    api_key: String,
    target_language: String,
    api_type: TranslationApiType,
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
    cache_size: usize,
    batch_size: usize,
    batch_delay_ms: u64,
) -> BatchTranslationProcessor {
    // Create the underlying translation service
    let service = match api_type {
        TranslationApiType::Demo => {
            Arc::new(DemoTranslation {
                target_language,
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::DeepL => {
            Arc::new(DeepLTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(500)), // 500ms between requests
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::Generic => {
            Arc::new(GenericTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(500)),
            }) as Arc<dyn TranslationService>
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
            }) as Arc<dyn TranslationService>
        },
    };
    
    // Create the batch processor
    BatchTranslationProcessor::new(service, cache_size, batch_size, batch_delay_ms)
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
    
    // Ultra-responsive mode settings (new additions)
    #[serde(default = "default_ultra_responsive")]
    ultra_responsive: bool,       // Whether to use ultra-responsive mode
    
    #[serde(default = "default_active_interval_ms")]
    active_interval_ms: u64,      // Polling interval when captions are active (milliseconds)
    
    #[serde(default = "default_idle_interval_ms")]
    idle_interval_ms: u64,        // Polling interval when captions are idle (milliseconds)
    
    #[serde(default = "default_active_timeout_sec")]
    active_timeout_sec: f64,      // How long to stay in active mode (seconds)
    
    // Translation batch and cache settings (new additions)
    #[serde(default = "default_translation_cache_size")]
    translation_cache_size: usize, // Maximum number of entries in the translation cache
    
    #[serde(default = "default_translation_batch_size")]
    translation_batch_size: usize, // Maximum number of texts to translate in a batch
    
    #[serde(default = "default_translation_batch_delay_ms")]
    translation_batch_delay_ms: u64, // Maximum delay before processing a batch (milliseconds)
}

// Default value functions for serde
fn default_capture_interval() -> f64 { 1.0 }
fn default_check_interval() -> u64 { 10 }
fn default_min_interval() -> f64 { 0.5 }
fn default_max_interval() -> f64 { 3.0 }
fn default_max_text_length() -> usize { 10000 }
fn default_ultra_responsive() -> bool { true }
fn default_active_interval_ms() -> u64 { 50 }  // 50ms polling when active (very responsive)
fn default_idle_interval_ms() -> u64 { 500 }   // 500ms polling when idle
fn default_active_timeout_sec() -> f64 { 5.0 } // 5 seconds after last caption
fn default_translation_cache_size() -> usize { 1000 } // Cache 1000 translations
fn default_translation_batch_size() -> usize { 5 }   // Translate 5 texts at a time
fn default_translation_batch_delay_ms() -> u64 { 500 } // Wait 500ms max for batch accumulation

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
            
            // Validate batch and cache settings
            if self.translation_cache_size == 0 {
                return Err(AppError::Validation("Translation cache size cannot be zero".to_string()));
            }
            
            if self.translation_batch_size == 0 {
                return Err(AppError::Validation("Translation batch size cannot be zero".to_string()));
            }
            
            if self.translation_batch_delay_ms == 0 {
                return Err(AppError::Validation("Translation batch delay cannot be zero".to_string()));
            }
        }
        
        // Validate ultra-responsive mode settings
        if self.ultra_responsive {
            if self.active_interval_ms == 0 {
                return Err(AppError::Validation("Active interval cannot be zero".to_string()));
            }
            
            if self.idle_interval_ms == 0 {
                return Err(AppError::Validation("Idle interval cannot be zero".to_string()));
            }
            
            if self.active_timeout_sec <= 0.0 {
                return Err(AppError::Validation("Active timeout must be positive".to_string()));
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
            ultra_responsive: default_ultra_responsive(),
            active_interval_ms: default_active_interval_ms(),
            idle_interval_ms: default_idle_interval_ms(),
            active_timeout_sec: default_active_timeout_sec(),
            translation_cache_size: default_translation_cache_size(),
            translation_batch_size: default_translation_batch_size(),
            translation_batch_delay_ms: default_translation_batch_delay_ms(),
        }
    }
}

/// Sentence tracker for maintaining sentence-level synchronization with batching support
struct SentenceTracker {
    next_id: u64,
    pending_sentences: HashMap<u64, PendingSentence>,
    buffered_text: String,
    // New fields to support batching
    batch_size: usize,
    batch_timeout_ms: u64,
    last_batch_time: u64,
}

struct PendingSentence {
    original: String,
    timestamp: u64,
    sent_for_translation: bool,
}

impl SentenceTracker {
    fn new(batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            next_id: 1,
            pending_sentences: HashMap::new(),
            buffered_text: String::new(),
            batch_size,
            batch_timeout_ms,
            last_batch_time: Self::current_time_ms(),
        }
    }
    
    /// Add new text to the buffer and extract complete sentences
    fn add_text(&mut self, text: &str) -> Vec<(u64, String)> {
        self.buffered_text.push_str(text);
        
        // Extract complete sentences
        let mut complete_sentences = Vec::new();
        let mut sentence_start = 0;
        let chars: Vec<char> = self.buffered_text.chars().collect();
        
        for i in 0..chars.len() {
            if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' || 
               (chars[i] == '\n' && i > 0 && chars[i-1] != '.') {
                // Found a sentence ending
                if i > sentence_start {
                    let sentence = self.buffered_text[sentence_start..=i].to_string();
                    let id = self.next_id;
                    self.next_id += 1;
                    
                    // Store and prepare for sending
                    self.pending_sentences.insert(id, PendingSentence {
                        original: sentence.clone(),
                        timestamp: Self::current_time_ms(),
                        sent_for_translation: false,
                    });
                    
                    complete_sentences.push((id, sentence));
                    sentence_start = i + 1;
                }
            }
        }
        
        // Update the buffer to contain only the incomplete sentence
        if sentence_start > 0 && sentence_start < self.buffered_text.len() {
            self.buffered_text = self.buffered_text[sentence_start..].to_string();
        } else if sentence_start > 0 {
            self.buffered_text.clear();
        }
        
        complete_sentences
    }
    
    /// Check if the current buffer should be sent even if incomplete
    fn should_send_incomplete(&self) -> bool {
        // Send incomplete sentences if they're long enough or have been buffering too long
        self.buffered_text.len() > 100 || 
        (!self.pending_sentences.is_empty() && 
         Self::current_time_ms() - self.pending_sentences.values().map(|s| s.timestamp).max().unwrap_or(0) > 5000)
    }
    
    /// Get the current incomplete sentence for sending
    fn get_incomplete_sentence(&mut self) -> Option<(u64, String)> {
        if self.buffered_text.is_empty() {
            return None;
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        // Store and prepare for sending
        self.pending_sentences.insert(id, PendingSentence {
            original: self.buffered_text.clone(),
            timestamp: Self::current_time_ms(),
            sent_for_translation: false,
        });
        
        let result = (id, self.buffered_text.clone());
        self.buffered_text.clear();
        Some(result)
    }
    
    /// Check if we have enough sentences to form a batch
    fn should_send_batch(&self) -> bool {
        let unsent_count = self.pending_sentences.values()
            .filter(|s| !s.sent_for_translation)
            .count();
        
        // Send if we have enough sentences or enough time has passed
        unsent_count >= self.batch_size || 
            (unsent_count > 0 && Self::current_time_ms() - self.last_batch_time >= self.batch_timeout_ms)
    }
    
    /// Get all pending sentences that haven't been sent yet
    fn get_pending_batch(&self) -> Vec<(u64, String)> {
        self.pending_sentences.iter()
            .filter(|(_, s)| !s.sent_for_translation)
            .map(|(id, s)| (*id, s.original.clone()))
            .collect()
    }
    
    /// Mark a sentence as sent for translation
    fn mark_sent_for_translation(&mut self, id: u64) {
        if let Some(sentence) = self.pending_sentences.get_mut(&id) {
            sentence.sent_for_translation = true;
        }
    }
    
    /// Mark multiple sentences as sent for translation
    fn mark_batch_sent_for_translation(&mut self, ids: &[u64]) {
        for id in ids {
            self.mark_sent_for_translation(*id);
        }
        self.last_batch_time = Self::current_time_ms();
    }
    
    /// Get current timestamp in milliseconds
    fn current_time_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// IPC messages for communication between processes
#[derive(Debug, Serialize, Deserialize)]
enum IpcMessage {
    /// Configuration message
    Config(Config),
    
    /// Enhanced text message with metadata for better synchronization
    Text {
        id: u64,             // Unique identifier for this text segment
        content: String,     // The text content to translate
        timestamp: u64,      // Timestamp when the text was captured
        is_complete: bool,   // Whether this is a complete sentence
    },
    
    /// Shutdown message
    Shutdown,
}

/// Windows named pipe wrapper
struct NamedPipe {
    handle: HANDLE,
}

impl NamedPipe {
    /// Creates a new named pipe server
    fn create_server(pipe_name: &str) -> Result<Self, AppError> {
        let pipe_path = format!(r"\\.\pipe\{}", pipe_name);
        
        let handle = unsafe {
            CreateNamedPipeW(
                &HSTRING::from(pipe_path),
                windows::Win32::Storage::FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0x00000003), // PIPE_ACCESS_DUPLEX
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                1, // Max instances
                4096, // Output buffer size
                4096, // Input buffer size
                0, // Default timeout
                None, // Default security attributes
            )
        };
        
        if handle.is_invalid() {
            return Err(AppError::Io(io::Error::last_os_error()));
        }
        
        Ok(Self { handle })
    }
    
    /// Connects to an existing named pipe
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
    
    /// Waits for a client to connect to the pipe
    fn wait_for_connection(&self) -> Result<(), AppError> {
        let result = unsafe { ConnectNamedPipe(self.handle, None) };
        if result.is_err() {
            let error = io::Error::last_os_error();
            // ERROR_PIPE_CONNECTED means client is already connected
            if error.raw_os_error() != Some(535) {
                return Err(AppError::Io(error));
            }
        }
        Ok(())
    }
    
    /// Writes a message to the pipe
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
    
    /// Reads a message from the pipe
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
    
    /// Whether to use ultra-responsive mode
    #[arg(short = 'u', long)]
    ultra_responsive: Option<bool>,
    
    /// Flag to indicate this is a translation window process
    #[arg(long, hide = true)]
    translation_window: bool,
    
    /// Named pipe name for inter-process communication
    #[arg(long, hide = true)]
    pipe_name: Option<String>,
}

/// Translation window for displaying translations
struct TranslationWindow {
    sentences: BTreeMap<u64, TranslatedSentence>,
    displayed_text: String,
    pending_translations: HashMap<u64, PendingTranslation>,
}

struct TranslatedSentence {
    id: u64,
    original: String,
    translation: String,
    timestamp: u64,
    is_complete: bool,
}

struct PendingTranslation {
    content: String,
    timestamp: u64,
    is_complete: bool,
}

impl TranslationWindow {
    fn new() -> Self {
        Self {
            sentences: BTreeMap::new(),
            displayed_text: String::new(),
            pending_translations: HashMap::new(),
        }
    }
    
    /// Process a new text message from the main process
    async fn process_text_message(
        &mut self, 
        id: u64, 
        content: String, 
        timestamp: u64,
        is_complete: bool,
        translation_processor: &BatchTranslationProcessor
    ) -> Result<(), AppError> {
        // First check if this is a duplicate (might happen with batching)
        if self.sentences.contains_key(&id) {
            return Ok(());
        }
        
        // Add to pending translations
        self.pending_translations.insert(id, PendingTranslation {
            content: content.clone(),
            timestamp,
            is_complete,
        });
        
        // Translate the text with batching and caching
        match translation_processor.translate(&content).await {
            Ok(translated) => {
                // Store the translated sentence
                self.sentences.insert(id, TranslatedSentence {
                    id,
                    original: content,
                    translation: translated,
                    timestamp,
                    is_complete,
                });
                
                // Remove from pending
                self.pending_translations.remove(&id);
                
                // Update the displayed text
                self.update_display();
                Ok(())
            },
            Err(e) => Err(e),
        }
    }
    
    /// Update the displayed text with the latest translations
    fn update_display(&mut self) {
        // Clear the current display
        self.displayed_text.clear();
        
        // Add each sentence with proper formatting
        for (_, sentence) in &self.sentences {
            // Add original text with formatting
            self.displayed_text.push_str(&format!("[Original] {}\n", sentence.original));
            
            // Add translation with formatting
            self.displayed_text.push_str(&format!("[Translation] {}\n\n", sentence.translation));
        }
        
        // Keep only the most recent sentences if the text gets too long
        self.maintain_display_length();
        
        // Display the updated text
        display_translation_window(&self.displayed_text).unwrap_or_else(|e| {
            warn!("Failed to update translation display: {}", e);
        });
    }
    
    /// Ensure the display doesn't get too long
    fn maintain_display_length(&mut self) {
        const MAX_SENTENCES: usize = 10;
        
        if self.sentences.len() > MAX_SENTENCES {
            // Remove oldest sentences
            let keys_to_remove: Vec<u64> = self.sentences.keys()
                .take(self.sentences.len() - MAX_SENTENCES)
                .cloned()
                .collect();
                
            for key in keys_to_remove {
                self.sentences.remove(&key);
            }
            
            // Rebuild the display
            self.displayed_text = String::new();
            for (_, sentence) in &self.sentences {
                self.displayed_text.push_str(&format!("[Original] {}\n", sentence.original));
                self.displayed_text.push_str(&format!("[Translation] {}\n\n", sentence.translation));
            }
        }
    }
    
    /// Checks if there are pending translations
    fn has_pending_translations(&self) -> bool {
        !self.pending_translations.is_empty()
    }
    
    /// Gets the IDs of pending translations
    fn get_pending_translation_ids(&self) -> Vec<u64> {
        self.pending_translations.keys().cloned().collect()
    }
}

/// Main engine for the caption process
struct Engine {
    config: Config,
    displayed_text: String, // Text displayed in the terminal
    caption_handle: CaptionHandle,
    translation_processor: Option<BatchTranslationProcessor>,
    consecutive_empty_captures: usize, // Count of consecutive empty captures
    adaptive_interval: f64, // Adaptive capture interval
    output_file: Option<fs::File>, // Output file handle
    translation_process: Option<std::process::Child>, // Translation window process
    translation_pipe: Option<NamedPipe>, // Named pipe for IPC with translation window
    sentence_tracker: SentenceTracker, // Sentence tracker for translation
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
        
        let min_interval = config.min_interval;
        let batch_size = config.translation_batch_size;
        let batch_delay_ms = config.translation_batch_delay_ms;
        
        // Create translation processor if enabled
        let translation_processor = if config.enable_translation {
            if let Some(api_key) = &config.translation_api_key {
                // Ensure target language is set
                let target_lang = config.target_language.clone().unwrap_or_else(|| {
                    match config.translation_api_type {
                        TranslationApiType::DeepL => "ZH".to_string(), // DeepL uses uppercase language codes
                        TranslationApiType::OpenAI => "Chinese".to_string(), // OpenAI uses natural language descriptions
                        _ => "zh-CN".to_string(),
                    }
                });
                
                info!("Initializing translation service with {:?} API, caching and batching", config.translation_api_type);
                Some(create_batch_translation_processor(
                    api_key.clone(),
                    target_lang,
                    config.translation_api_type,
                    config.openai_api_url.clone(),
                    config.openai_model.clone(),
                    config.openai_system_prompt.clone(),
                    config.translation_cache_size,
                    config.translation_batch_size,
                    config.translation_batch_delay_ms
                ))
            } else if config.translation_api_type == TranslationApiType::Demo {
                // Demo mode doesn't need an API key
                let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
                Some(create_batch_translation_processor(
                    "".to_string(),
                    target_lang,
                    TranslationApiType::Demo,
                    None,
                    None,
                    None,
                    config.translation_cache_size,
                    config.translation_batch_size,
                    config.translation_batch_delay_ms
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
                // For debugging, keep stdout and stderr
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
            translation_processor,
            consecutive_empty_captures: 0,
            adaptive_interval: min_interval,  // Use stored value instead of config.min_interval
            output_file,
            config: config.clone(),  // Clone config before moving it
            translation_process,
            translation_pipe,
            sentence_tracker: SentenceTracker::new(
                batch_size,         // Use stored value instead of config.translation_batch_size
                batch_delay_ms      // Use stored value instead of config.translation_batch_delay_ms
            ),
        })
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
        
        // Set up Ctrl+C handler
        let ctrl_c = tokio::signal::ctrl_c();
        tokio::pin!(ctrl_c);
        
        // Initialize caption timer with appropriate interval
        let mut current_interval_ms = if self.config.ultra_responsive {
            self.config.active_interval_ms
        } else {
            (self.config.capture_interval * 1000.0) as u64
        };
        
        let mut caption_timer = tokio::time::interval(Duration::from_millis(current_interval_ms));
        
        // Print startup info
        println!("Live captions monitoring started:");
        if self.config.ultra_responsive {
            println!("  - Using ultra-responsive mode");
            println!("    - Active polling: {}ms", self.config.active_interval_ms);
            println!("    - Idle polling: {}ms", self.config.idle_interval_ms);
            println!("    - Active timeout: {}s", self.config.active_timeout_sec);
        } else {
            println!("  - Using standard polling with {} second interval", self.config.capture_interval);
        }
        println!("  - Check interval: {} seconds", self.config.check_interval);
        if self.config.enable_translation {
            println!("  - Translation enabled with batch processing and caching");
            if let Some(processor) = &self.translation_processor {
                let service = processor.service.as_ref();
                println!("    - Translation service: {}", service.get_name());
                println!("    - Target language: {}", service.get_target_language());
                println!("    - Cache size: {} entries", self.config.translation_cache_size);
                println!("    - Batch size: {} texts", self.config.translation_batch_size);
                println!("    - Batch delay: {}ms", self.config.translation_batch_delay_ms);
            }
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
                    
                    // Log translation statistics if enabled
                    if let Some(processor) = &self.translation_processor {
                        let (requests, batches, cache_hits, batch_sizes, hit_rate) = processor.get_stats().await;
                        info!("Translation stats: {} requests, {} batches, {} cache hits, {:.1}% hit rate", 
                              requests, batches, cache_hits, hit_rate * 100.0);
                    }
                },
                _ = caption_timer.tick() => {
                    // Process captions
                    match self.caption_handle.get_captions().await {
                        Ok(Some(text)) => {
                            debug!("Captured new text: {}", text);
                            
                            // Add to displayed text (original text only)
                            self.displayed_text.push_str(&text);
                            self.limit_text_length();
                            Self::display_text(&self.displayed_text)?;
                            
                            // Process for translation if translation is enabled
                            if self.translation_pipe.is_some() && !text.is_empty() {
                                // Extract complete sentences for translation
                                let complete_sentences = self.sentence_tracker.add_text(&text);
                                
                                // Check if we should send a batch
                                if self.sentence_tracker.should_send_batch() {
                                    let batch = self.sentence_tracker.get_pending_batch();
                                    if !batch.is_empty() {
                                        let batch_ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
                                        
                                        debug!("Sending batch of {} sentences for translation", batch.len());
                                        
                                        // Send batch to translation window
                                        if let Some(pipe) = &self.translation_pipe {
                                            for (id, content) in batch {
                                                let message = IpcMessage::Text {
                                                    id,
                                                    content,
                                                    timestamp: SentenceTracker::current_time_ms(),
                                                    is_complete: true,
                                                };
                                                
                                                if let Err(e) = pipe.write_message(&message) {
                                                    warn!("Failed to send batch text to translation window: {}", e);
                                                }
                                            }
                                            
                                            // Mark batch as sent
                                            self.sentence_tracker.mark_batch_sent_for_translation(&batch_ids);
                                        }
                                    }
                                }
                                
                                // Check if we should send incomplete sentences
                                if self.sentence_tracker.should_send_incomplete() {
                                    if let Some((id, text)) = self.sentence_tracker.get_incomplete_sentence() {
                                        if let Some(pipe) = &self.translation_pipe {
                                            debug!("Sending incomplete sentence for translation [{}]: {}", id, text);
                                            let message = IpcMessage::Text {
                                                id,
                                                content: text,
                                                timestamp: SentenceTracker::current_time_ms(),
                                                is_complete: false,
                                            };
                                            
                                            if let Err(e) = pipe.write_message(&message) {
                                                warn!("Failed to send text to translation window: {}", e);
                                            } else {
                                                self.sentence_tracker.mark_sent_for_translation(id);
                                            }
                                        }
                                    }
                                }
                            }
                            
                            // Write to output file (if configured)
                            if let Some(file) = &mut self.output_file {
                                if let Err(e) = writeln!(file, "{}", text) {
                                    warn!("Failed to write to output file: {}", e);
                                }
                            }
                            
                            // In ultra-responsive mode, switch to fast polling on activity
                            if self.config.ultra_responsive && current_interval_ms != self.config.active_interval_ms {
                                current_interval_ms = self.config.active_interval_ms;
                                caption_timer = tokio::time::interval(Duration::from_millis(current_interval_ms));
                                debug!("Switched to active polling interval: {}ms", current_interval_ms);
                            }
                            
                            // Reset consecutive empty captures counter
                            self.consecutive_empty_captures = 0;
                        },
                        Ok(None) => {
                            debug!("No new captions available");
                            
                            // In ultra-responsive mode, check if we should switch to idle polling
                            if self.config.ultra_responsive {
                                match self.caption_handle.is_active(self.config.active_timeout_sec).await {
                                    Ok(active) => {
                                        let target_interval_ms = if active {
                                            self.config.active_interval_ms
                                        } else {
                                            self.config.idle_interval_ms
                                        };
                                        
                                        // Only update if interval significantly different
                                        if current_interval_ms != target_interval_ms {
                                            current_interval_ms = target_interval_ms;
                                            caption_timer = tokio::time::interval(Duration::from_millis(current_interval_ms));
                                            debug!("Adjusted polling interval to {}ms", current_interval_ms);
                                        }
                                    },
                                    Err(e) => warn!("Failed to check caption activity: {}", e)
                                }
                            }
                            
                            // Check if we should send any buffered text due to inactivity
                            if self.translation_pipe.is_some() && self.consecutive_empty_captures > 2 {
                                // Check if we have any pending sentences to send as a batch
                                if self.sentence_tracker.should_send_batch() {
                                    let batch = self.sentence_tracker.get_pending_batch();
                                    if !batch.is_empty() {
                                        let batch_ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
                                        
                                        debug!("Sending batch of {} sentences for translation due to inactivity", batch.len());
                                        
                                        // Send batch to translation window
                                        if let Some(pipe) = &self.translation_pipe {
                                            for (id, content) in batch {
                                                let message = IpcMessage::Text {
                                                    id,
                                                    content,
                                                    timestamp: SentenceTracker::current_time_ms(),
                                                    is_complete: true,
                                                };
                                                
                                                if let Err(e) = pipe.write_message(&message) {
                                                    warn!("Failed to send batch text to translation window: {}", e);
                                                }
                                            }
                                            
                                            // Mark batch as sent
                                            self.sentence_tracker.mark_batch_sent_for_translation(&batch_ids);
                                        }
                                    }
                                }
                                
                                // Also check for incomplete sentences
                                if let Some((id, text)) = self.sentence_tracker.get_incomplete_sentence() {
                                    if let Some(pipe) = &self.translation_pipe {
                                        debug!("Sending buffered text for translation due to inactivity [{}]: {}", id, text);
                                        let message = IpcMessage::Text {
                                            id,
                                            content: text,
                                            timestamp: SentenceTracker::current_time_ms(),
                                            is_complete: false,
                                        };
                                        
                                        if let Err(e) = pipe.write_message(&message) {
                                            warn!("Failed to send text to translation window: {}", e);
                                        } else {
                                            self.sentence_tracker.mark_sent_for_translation(id);
                                        }
                                    }
                                }
                            }
                            
                            // Track consecutive empty captures
                            self.consecutive_empty_captures += 1;
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
        
        // Process any remaining pending sentences as a batch
        if self.translation_pipe.is_some() {
            let batch = self.sentence_tracker.get_pending_batch();
            if !batch.is_empty() {
                let batch_ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
                
                info!("Sending final batch of {} sentences for translation", batch.len());
                
                // Send batch to translation window
                if let Some(pipe) = &self.translation_pipe {
                    for (id, content) in batch {
                        let message = IpcMessage::Text {
                            id,
                            content,
                            timestamp: SentenceTracker::current_time_ms(),
                            is_complete: true,
                        };
                        
                        if let Err(e) = pipe.write_message(&message) {
                            warn!("Failed to send final batch text to translation window: {}", e);
                        }
                    }
                    
                    // Mark batch as sent
                    self.sentence_tracker.mark_batch_sent_for_translation(&batch_ids);
                }
            }
        }
        
        // Also send any incomplete sentences
        if let Some((id, text)) = self.sentence_tracker.get_incomplete_sentence() {
            if let Some(pipe) = &self.translation_pipe {
                info!("Sending final buffered text for translation [{}]: {}", id, text);
                let message = IpcMessage::Text {
                    id,
                    content: text,
                    timestamp: SentenceTracker::current_time_ms(),
                    is_complete: false,
                };
                
                if let Err(e) = pipe.write_message(&message) {
                    warn!("Failed to send final buffered text to translation window: {}", e);
                } else {
                    self.sentence_tracker.mark_sent_for_translation(id);
                }
            }
        }
        
        // Try to get final captions
        match self.caption_handle.get_captions().await {
            Ok(Some(text)) => {
                // If translation window is active, send the final text
                if let Some(pipe) = &self.translation_pipe {
                    let id = self.sentence_tracker.next_id;
                    let message = IpcMessage::Text {
                        id,
                        content: text.clone(),
                        timestamp: SentenceTracker::current_time_ms(),
                        is_complete: true,
                    };
                    if let Err(e) = pipe.write_message(&message) {
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

/// Displays translation in a dedicated window
fn display_translation_window(text: &str) -> Result<(), AppError> {
    // Clear screen and move to top-left corner
    print!("\x1B[2J\x1B[1;1H");
    println!("Translation Window (with Batch Processing and Caching)");
    println!("----------------------------------------");
    println!("{}", text);
    io::stdout().flush()?;
    Ok(())
}

/// Runs the translation window process
/// 
/// This function handles the translation window process, which receives text from
/// the main process, translates it, and displays it in a separate window.
async fn run_translation_window(pipe_name: String) -> Result<(), AppError> {
    info!("Starting translation window, pipe: {}", pipe_name);
    
    // Connect to the named pipe
    let pipe = NamedPipe::connect_client(&pipe_name)?;
    info!("Connected to pipe: {}", pipe_name);
    
    // Wait for configuration message
    let config = match pipe.read_message()? {
        IpcMessage::Config(config) => config,
        _ => return Err(AppError::Config("First message should be Config message".to_string())),
    };
    
    info!("Received configuration");
    
    // Create translation service with batching and caching
    let translation_processor = if let Some(api_key) = &config.translation_api_key {
        // Ensure target language is set
        let target_lang = config.target_language.clone().unwrap_or_else(|| {
            match config.translation_api_type {
                TranslationApiType::DeepL => "ZH".to_string(),
                TranslationApiType::OpenAI => "Chinese".to_string(),
                _ => "zh-CN".to_string(),
            }
        });
        
        info!("Initializing translation service with {:?} API, caching and batching", config.translation_api_type);
        Some(create_batch_translation_processor(
            api_key.clone(),
            target_lang,
            config.translation_api_type,
            config.openai_api_url.clone(),
            config.openai_model.clone(),
            config.openai_system_prompt.clone(),
            config.translation_cache_size,
            config.translation_batch_size,
            config.translation_batch_delay_ms
        ))
    } else if config.translation_api_type == TranslationApiType::Demo {
        // Demo mode doesn't need an API key
        let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
        Some(create_batch_translation_processor(
            "".to_string(),
            target_lang,
            TranslationApiType::Demo,
            None,
            None,
            None,
            config.translation_cache_size,
            config.translation_batch_size,
            config.translation_batch_delay_ms
        ))
    } else {
        warn!("Translation enabled but no API key provided");
        None
    };
    
    if translation_processor.is_none() {
        return Err(AppError::Translation("Failed to create translation processor".to_string()));
    }
    
    let translation_processor = translation_processor.unwrap();
    
    // Set up Ctrl+C handler
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);
    
    // Initialize translation window
    let mut translation_window = TranslationWindow::new();
    
    // Clear screen and display window title
    print!("\x1B[2J\x1B[1;1H");
    println!("Translation Window (with Batch Processing and Caching)");
    println!("  - Translation service: {}", translation_processor.service.get_name());
    println!("  - Target language: {}", translation_processor.service.get_target_language());
    println!("  - Cache size: {} entries", config.translation_cache_size);
    println!("  - Batch size: {} texts", config.translation_batch_size);
    println!("  - Batch delay: {}ms", config.translation_batch_delay_ms);
    println!("Press Ctrl+C to exit");
    println!("----------------------------------------");
    io::stdout().flush()?;
    
    // Create timer for periodic stats logging
    let mut stats_timer = tokio::time::interval(Duration::from_secs(30));
    
    // Main loop
    loop {
        tokio::select! {
            _ = &mut ctrl_c => {
                println!("\nReceived shutdown signal");
                break;
            },
            _ = stats_timer.tick() => {
                // Log translation statistics
                let (requests, batches, cache_hits, batch_sizes, hit_rate) = translation_processor.get_stats().await;
                if requests > 0 {
                    info!("Translation stats: {} requests, {} batches, {} cache hits, {:.1}% hit rate",
                          requests, batches, cache_hits, hit_rate * 100.0);
                }
            },
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Check for new messages
                match pipe.read_message() {
                    Ok(IpcMessage::Text { id, content, timestamp, is_complete }) => {
                        debug!("Received text for translation [{}]: {}", id, content);
                        if !content.is_empty() {
                            if let Err(e) = translation_window.process_text_message(
                                id, content, timestamp, is_complete, &translation_processor
                            ).await {
                                warn!("Translation failed: {}", e);
                            }
                        }
                    },
                    Ok(IpcMessage::Shutdown) => {
                        info!("Received shutdown message");
                        
                        // Display final translation stats
                        let (requests, batches, cache_hits, _, hit_rate) = translation_processor.get_stats().await;
                        info!("Final translation stats: {} requests, {} batches, {} cache hits, {:.1}% hit rate",
                              requests, batches, cache_hits, hit_rate * 100.0);
                        
                        break;
                    },
                    Ok(_) => {
                        // Ignore other message types
                    },
                    Err(e) => {
                        // Check if error is due to pipe being closed
                        if let AppError::Io(io_err) = &e {
                            if io_err.kind() == io::ErrorKind::BrokenPipe {
                                info!("Pipe closed, shutting down");
                                break;
                            }
                        }
                        warn!("Error reading from pipe: {}", e);
                    }
                }
            }
        }
    }
    
    println!("\nTranslation window closing");
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
    if let Some(ultra) = args.ultra_responsive {
        config.ultra_responsive = ultra;
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
            last_activity: Instant::now(),
            active_mode: false,
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
            ultra_responsive: true,
            active_interval_ms: 50,
            idle_interval_ms: 500,
            active_timeout_sec: 5.0,
            translation_cache_size: 1000,
            translation_batch_size: 5,
            translation_batch_delay_ms: 500,
        };
        assert!(config.validate().is_ok());
        
        // Test invalid intervals
        let mut invalid_config = config.clone();
        invalid_config.min_interval = 5.0;
        invalid_config.max_interval = 3.0;
        assert!(invalid_config.validate().is_err());
    }
    
    /// Tests for sentence tracker with batching
    #[test]
    fn test_sentence_tracker() {
        let mut tracker = SentenceTracker::new(3, 500);
        
        // Test adding complete sentences
        let sentences = tracker.add_text("Hello. This is a test.");
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].1, "Hello.");
        assert_eq!(sentences[1].1, " This is a test.");
        
        // Test incomplete sentence
        let sentences = tracker.add_text("This is incomplete");
        assert_eq!(sentences.len(), 0);
        assert_eq!(tracker.buffered_text, "This is incomplete");
        
        // Test completing a sentence
        let sentences = tracker.add_text(" and now complete.");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].1, "This is incomplete and now complete.");
        assert_eq!(tracker.buffered_text, "");
        
        // Test batch formation
        assert!(tracker.should_send_batch()); // We have 3 sentences now
        let batch = tracker.get_pending_batch();
        assert_eq!(batch.len(), 3);
        
        // Test marking batch as sent
        let ids: Vec<u64> = batch.iter().map(|(id, _)| *id).collect();
        tracker.mark_batch_sent_for_translation(&ids);
        
        // Verify batch was marked as sent
        assert!(!tracker.should_send_batch());
        assert_eq!(tracker.get_pending_batch().len(), 0);
    }
    
    /// Tests for translation cache
    #[test]
    fn test_translation_cache() {
        let mut cache = TranslationCache::new(5);
        
        // Test cache miss
        assert_eq!(cache.get("Hello"), None);
        
        // Test cache put and hit
        cache.put("Hello".to_string(), "Hola".to_string());
        assert_eq!(cache.get("Hello"), Some("Hola".to_string()));
        
        // Test cache stats
        let (size, hits, misses, hit_rate) = cache.stats();
        assert_eq!(size, 1);
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert_eq!(hit_rate, 0.5);
        
        // Test LRU eviction
        cache.put("One".to_string(), "Uno".to_string());
        cache.put("Two".to_string(), "Dos".to_string());
        cache.put("Three".to_string(), "Tres".to_string());
        cache.put("Four".to_string(), "Cuatro".to_string());
        cache.put("Five".to_string(), "Cinco".to_string());
        
        // Cache is full, this should evict "Hello"
        cache.put("Six".to_string(), "Seis".to_string());
        
        // "Hello" should be evicted
        assert_eq!(cache.get("Hello"), None);
        
        // Newer entries should still be present
        assert_eq!(cache.get("One"), Some("Uno".to_string()));
        assert_eq!(cache.get("Six"), Some("Seis".to_string()));
    }
}