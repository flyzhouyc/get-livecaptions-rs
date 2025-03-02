use tokio::time::{Duration, Instant};
use std::path::PathBuf;
use std::sync::Arc;
use windows::{
    core::*, Win32::System::Com::*, Win32::UI::{Accessibility::*, WindowsAndMessaging::*}, 
    Win32::Foundation::{HWND, HANDLE, CloseHandle},
    Win32::System::Pipes::*,
    Win32::Storage::FileSystem::{CreateFileW, FILE_SHARE_NONE, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, WriteFile, ReadFile},
    //Win32::Security::*,
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
use std::collections::{HashMap, BTreeMap};
use std::collections::VecDeque;
use chrono::{Local, TimeZone};

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
/// Optimized batch processor for translation with improved timing and reliability
struct OptimizedBatchTranslationProcessor {
    service: Arc<dyn TranslationService>,
    cache: Arc<tokio::sync::Mutex<TranslationCache>>,
    queue: Arc<tokio::sync::Mutex<HashMap<String, BatchQueueEntry>>>,
    batch_size: usize,              // Reduced for better responsiveness
    max_batch_delay: Duration,      // Reduced for better responsiveness
    min_batch_delay: Duration,      // Minimum delay before processing even if batch size not reached
    priority_threshold: usize,      // Threshold for priority processing
    stats: Arc<tokio::sync::Mutex<EnhancedBatchStats>>,
    shutdown_signal: Arc<AtomicBool>,
}

/// Enhanced entry for the batch queue with priority and timing
struct BatchQueueEntry {
    senders: Vec<oneshot::Sender<Result<String, AppError>>>,
    priority: usize,                // Higher number = higher priority
    first_enqueued: Instant,        // When the first request for this text was enqueued
    last_enqueued: Instant,         // When the most recent request was enqueued
}

/// Enhanced statistics for batch processing
#[derive(Debug, Default)]
struct EnhancedBatchStats {
    requests: usize,                // Total number of requests
    batches: usize,                 // Total number of batches processed
    cache_hits: usize,              // Number of cache hits
    priority_batches: usize,        // Number of priority-triggered batches
    timeout_batches: usize,         // Number of timeout-triggered batches
    batch_sizes: Vec<usize>,        // Size of each batch
    processing_times: Vec<u64>,     // Processing time for each batch in ms
    average_latency: f64,           // Moving average of request-to-completion time
}

impl OptimizedBatchTranslationProcessor {
    /// Creates a new optimized batch processor with reduced batch size and timeouts
    fn new(
        service: Arc<dyn TranslationService>,
        cache_size: usize,
        batch_size: usize,          // Recommended: 3 (reduced from 5)
        max_batch_delay_ms: u64,    // Recommended: 150-200ms (reduced from 500)
        min_batch_delay_ms: u64,    // Recommended: 50ms (new parameter)
    ) -> Self {
        let processor = Self {
            service,
            cache: Arc::new(tokio::sync::Mutex::new(TranslationCache::new(cache_size))),
            queue: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            batch_size,
            max_batch_delay: Duration::from_millis(max_batch_delay_ms),
            min_batch_delay: Duration::from_millis(min_batch_delay_ms),
            priority_threshold: 3,  // Process immediately when 3+ high-priority items are queued
            stats: Arc::new(tokio::sync::Mutex::new(EnhancedBatchStats::default())),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };
        
        // Start the batch processing loop
        processor.start_processing_loop();
        
        processor
    }
    
    /// Starts the background processing loop with improved timing logic
    fn start_processing_loop(&self) {
        let service = self.service.clone();
        let queue = self.queue.clone();
        let cache = self.cache.clone();
        let batch_size = self.batch_size;
        let max_batch_delay = self.max_batch_delay;
        let min_batch_delay = self.min_batch_delay;
        let priority_threshold = self.priority_threshold;
        let stats = self.stats.clone();
        let shutdown_signal = self.shutdown_signal.clone();
        
        tokio::spawn(async move {
            let mut last_process_time = Instant::now();
            let mut check_interval = tokio::time::interval(Duration::from_millis(10));
            
            loop {
                // Check shutdown signal
                if shutdown_signal.load(std::sync::atomic::Ordering::Relaxed) {
                    debug!("Translation batch processor shutting down");
                    break;
                }
                
                // Wait for the next interval tick, but also use tokio::select to handle cancellation
                tokio::select! {
                    _ = check_interval.tick() => {
                        // Time to check the batch
                    }
                    _ = tokio::time::sleep(Duration::from_secs(3600)) => {
                        // Fallback for safety, should never be reached
                        continue;
                    }
                }
                
                // Check if we should process a batch
                let time_since_last = last_process_time.elapsed();
                let (should_process, reason) = {
                    let queue_guard = queue.lock().await;
                    
                    if queue_guard.is_empty() {
                        (false, "empty") 
                    } else {
                        // Count high priority items (recent additions)
                        let high_priority_count = queue_guard.values()
                            .filter(|entry| entry.priority >= priority_threshold)
                            .count();
                            
                        // Count total items
                        let queue_size: usize = queue_guard.values()
                            .map(|v| v.senders.len())
                            .sum();
                            
                        // Check if any entry has been waiting too long
                        let oldest_entry = queue_guard.values()
                            .map(|entry| entry.first_enqueued.elapsed())
                            .max()
                            .unwrap_or_else(|| Duration::from_secs(0));
                            
                        if high_priority_count >= priority_threshold {
                            (true, "priority") 
                        } else if queue_size >= batch_size {
                            (true, "batch_size")
                        } else if !queue_guard.is_empty() && time_since_last >= max_batch_delay {
                            // Maximum delay reached, process whatever we have
                            (true, "max_delay")
                        } else if !queue_guard.is_empty() && oldest_entry >= min_batch_delay {
                            // Minimum delay reached with some entries, process
                            (true, "min_delay")
                        } else {
                            (false, "waiting")
                        }
                    }
                };
                
                if should_process {
                    debug!("Processing translation batch. Trigger: {}", reason);
                    last_process_time = Instant::now();
                    
                    if let Err(e) = Self::process_batch(&service, &queue, &cache, &stats, reason).await {
                        warn!("Failed to process translation batch: {}", e);
                    }
                }
            }
        });
    }
    
    /// Processes a batch of translation requests with improved prioritization
    async fn process_batch(
        service: &Arc<dyn TranslationService>,
        queue: &Arc<tokio::sync::Mutex<HashMap<String, BatchQueueEntry>>>,
        cache: &Arc<tokio::sync::Mutex<TranslationCache>>,
        stats: &Arc<tokio::sync::Mutex<EnhancedBatchStats>>,
        trigger_reason: &str,
    ) -> Result<(), AppError> {
        // Start timing the batch processing
        let batch_start = Instant::now();
        
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
        let mut total_senders = 0;
        
        // Check cache first
        {
            let mut cache_guard = cache.lock().await;
            
            for (text, entry) in current_queue.drain() {
                let sender_count = entry.senders.len();
                total_senders += sender_count;
                
                if let Some(cached_translation) = cache_guard.get(&text) {
                    // Cache hit, respond immediately
                    for sender in entry.senders {
                        let _ = sender.send(Ok(cached_translation.clone()));
                    }
                    
                    // Update stats
                    let mut stats_guard = stats.lock().await;
                    stats_guard.cache_hits += sender_count;
                } else if !entry.senders.is_empty() {
                    // Cache miss, add to translation batch
                    if !text_to_senders.contains_key(&text) {
                        texts_to_translate.push(text.clone());
                    }
                    text_to_senders.insert(text, entry.senders);
                }
            }
        }
        
        // Update batch stats based on trigger reason
        {
            let mut stats_guard = stats.lock().await;
            match trigger_reason {
                "priority" => stats_guard.priority_batches += 1,
                "max_delay" | "min_delay" => stats_guard.timeout_batches += 1,
                _ => {}
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
            
            debug!("Processing batch of {} texts (from {} total requests). Trigger: {}", 
                   texts_to_translate.len(), total_senders, trigger_reason);
            
            // Call the translation service
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
                            let _ = sender.send(Err(AppError::Translation(
                                "Batch translation returned fewer results than expected".to_string())));
                        }
                    }
                }
            }
            
            // Record processing time
            let processing_time = batch_start.elapsed().as_millis() as u64;
            let mut stats_guard = stats.lock().await;
            stats_guard.processing_times.push(processing_time);
            
            // Update average latency
            if stats_guard.processing_times.len() > 0 {
                let sum: u64 = stats_guard.processing_times.iter().sum();
                stats_guard.average_latency = sum as f64 / stats_guard.processing_times.len() as f64;
                
                // Keep only the last 100 processing times
                if stats_guard.processing_times.len() > 100 {
                    stats_guard.processing_times.remove(0);
                }
            }
            
            debug!("Batch processed in {}ms (avg latency: {:.1}ms)", 
                   processing_time, stats_guard.average_latency);
        }
        
        Ok(())
    }
    
    /// Translates text with caching and adaptive batching
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Update stats
        {
            let mut stats_guard = self.stats.lock().await;
            stats_guard.requests += 1;
        }
        
        // Check cache first (fast path)
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
        
        // Determine priority based on text length (shorter texts get higher priority)
        let priority = if text.len() < 50 { 4 }    // Very short: highest priority
                       else if text.len() < 100 { 3 }  // Short: high priority
                       else if text.len() < 200 { 2 }  // Medium: normal priority
                       else { 1 };                  // Long: low priority
        
        // Add to queue with priority
        {
            let mut queue_guard = self.queue.lock().await;
            let entry = queue_guard.entry(text.to_string())
                .or_insert_with(|| BatchQueueEntry {
                    senders: Vec::new(),
                    priority,
                    first_enqueued: Instant::now(),
                    last_enqueued: Instant::now(),
                });
            
            entry.senders.push(sender);
            entry.last_enqueued = Instant::now();
            entry.priority = entry.priority.max(priority); // Use highest priority if multiple requests
        }
        
        // Wait for result with timeout handling
        match tokio::time::timeout(Duration::from_secs(10), receiver).await {
            Ok(result) => result.map_err(|_| AppError::Translation("Translation task cancelled".to_string())),
            Err(_) => Err(AppError::Translation("Translation timed out after 10 seconds".to_string())),
        }
    }
    
    /// Signals the processor to shut down
    fn shutdown(&self) {
        self.shutdown_signal.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    
    /// Gets enhanced statistics about the batch processor
    async fn get_detailed_stats(&self) -> EnhancedBatchStats {
        self.stats.lock().await.clone()
    }
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
            
            for (text, senders) in current_queue.drain() {
                if let Some(cached_translation) = cache_guard.get(&text) {
                    // Cache hit, respond immediately
                    let sender_count = senders.len();
                    for sender in senders {
                        let _ = sender.send(Ok(cached_translation.clone()));
                    }
                    
                    // Update stats
                    let mut stats_guard = stats.lock().await;
                    stats_guard.cache_hits += sender_count;
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
#[async_trait::async_trait]
impl TranslationService for DeepLTranslation {
    async fn translate(&self, text: &str) -> Result<String, AppError> {
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // DeepL API call with improved error handling and retry logic
        let url = "https://api-free.deepl.com/v2/translate";
        
        // Try up to 3 times with exponential backoff
        let mut retry_count = 0;
        let max_retries = 3;
        let mut last_error = None;
        
        while retry_count < max_retries {
            match self.client.post(url)
                .timeout(Duration::from_secs(5))  // Add timeout to prevent hanging requests
                .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
                .form(&[
                    ("text", text),
                    ("target_lang", &self.target_language),
                    ("source_lang", "EN"),
                ])
                .send()
                .await
            {
                Ok(response) => {
                    // Check for rate limiting or server errors
                    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        retry_count += 1;
                        let backoff = Duration::from_millis(300 * 2u64.pow(retry_count as u32));
                        warn!("DeepL rate limit reached, backing off for {}ms", backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if response.status().is_server_error() {
                        retry_count += 1;
                        let backoff = Duration::from_millis(200 * 2u64.pow(retry_count as u32));
                        warn!("DeepL server error {}, backing off for {}ms", 
                              response.status(), backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if !response.status().is_success() {
                        return Err(AppError::Translation(
                            format!("DeepL API error: {} - {}", 
                                    response.status(), 
                                    response.text().await.unwrap_or_default())
                        ));
                    }
                    
                    // Parse successful response
                    #[derive(Deserialize)]
                    struct DeepLResponse {
                        translations: Vec<Translation>,
                    }
                    
                    #[derive(Deserialize)]
                    struct Translation {
                        text: String,
                    }
                    
                    match response.json::<DeepLResponse>().await {
                        Ok(result) => {
                            if let Some(translation) = result.translations.first() {
                                return Ok(translation.text.clone());
                            } else {
                                return Err(AppError::Translation("Empty translation result from DeepL".to_string()));
                            }
                        },
                        Err(e) => {
                            retry_count += 1;
                            last_error = Some(AppError::Translation(
                                format!("Failed to parse DeepL response: {}", e)));
                            
                            if retry_count < max_retries {
                                let backoff = Duration::from_millis(100 * 2u64.pow(retry_count as u32));
                                tokio::time::sleep(backoff).await;
                            }
                        }
                    }
                },
                Err(e) => {
                    retry_count += 1;
                    last_error = Some(AppError::Translation(
                        format!("Failed to send translation request to DeepL: {}", e)));
                    
                    if retry_count < max_retries {
                        let backoff = Duration::from_millis(100 * 2u64.pow(retry_count as u32));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        
        // If we got here, all retries failed
        Err(last_error.unwrap_or_else(|| 
            AppError::Translation("Unknown error after retries".to_string())))
    }
    
    // Enhanced batch translation implementation for DeepL
    async fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        
        // Split large batches into smaller ones (DeepL has limits)
        const MAX_BATCH_SIZE: usize = 10; // DeepL recommends no more than ~10 texts per batch
        if texts.len() > MAX_BATCH_SIZE {
            debug!("Splitting large batch of {} texts into smaller batches", texts.len());
            
            let mut all_results = Vec::with_capacity(texts.len());
            for chunk in texts.chunks(MAX_BATCH_SIZE) {
                let chunk_results = self.translate_batch_chunk(chunk).await?;
                all_results.extend(chunk_results);
            }
            return Ok(all_results);
        }
        
        // For smaller batches, use the optimized implementation
        self.translate_batch_chunk(texts).await
    }
    
    // Helper methods remain the same...
}

impl DeepLTranslation {
    /// Process a single batch chunk with retries and improved error handling
    async fn translate_batch_chunk(&self, texts: &[String]) -> Result<Vec<String>, AppError> {
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
        
        // Try up to 3 times with exponential backoff
        let mut retry_count = 0;
        let max_retries = 3;
        let mut last_error = None;
        
        while retry_count < max_retries {
            match self.client.post(url)
                .timeout(Duration::from_secs(10))  // Longer timeout for batches
                .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
                .form(&form)
                .send()
                .await
            {
                Ok(response) => {
                    // Handle response status
                    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        retry_count += 1;
                        let backoff = Duration::from_millis(500 * 2u64.pow(retry_count as u32));
                        warn!("DeepL rate limit reached during batch, backing off for {}ms", backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if response.status().is_server_error() {
                        retry_count += 1;
                        let backoff = Duration::from_millis(300 * 2u64.pow(retry_count as u32));
                        warn!("DeepL server error {} during batch, backing off for {}ms", 
                              response.status(), backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if !response.status().is_success() {
                        return Err(AppError::Translation(
                            format!("DeepL API batch error: {} - {}", 
                                    response.status(), 
                                    response.text().await.unwrap_or_default())
                        ));
                    }
                    
                    // Parse successful response
                    #[derive(Deserialize)]
                    struct DeepLResponse {
                        translations: Vec<Translation>,
                    }
                    
                    #[derive(Deserialize)]
                    struct Translation {
                        text: String,
                    }
                    
                    match response.json::<DeepLResponse>().await {
                        Ok(result) => {
                            if result.translations.len() != texts.len() {
                                return Err(AppError::Translation(
                                    format!("DeepL returned {} translations for {} texts", 
                                            result.translations.len(), texts.len())
                                ));
                            }
                            
                            // Extract translations
                            let translations = result.translations.into_iter()
                                .map(|t| t.text)
                                .collect();
                            
                            return Ok(translations);
                        },
                        Err(e) => {
                            retry_count += 1;
                            last_error = Some(AppError::Translation(
                                format!("Failed to parse DeepL batch response: {}", e)));
                            
                            if retry_count < max_retries {
                                let backoff = Duration::from_millis(200 * 2u64.pow(retry_count as u32));
                                tokio::time::sleep(backoff).await;
                            }
                        }
                    }
                },
                Err(e) => {
                    retry_count += 1;
                    last_error = Some(AppError::Translation(
                        format!("Failed to send batch translation request to DeepL: {}", e)));
                    
                    if retry_count < max_retries {
                        let backoff = Duration::from_millis(200 * 2u64.pow(retry_count as u32));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        
        // If we got here, all retries failed
        Err(last_error.unwrap_or_else(|| 
            AppError::Translation("Unknown error after batch retries".to_string())))
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

/// Enhanced configuration with new options for improved functionality
#[derive(Debug, Serialize, Deserialize, Clone)]
struct EnhancedConfig {
    // Original basic settings
    #[serde(default = "default_capture_interval")]
    capture_interval: f64,
    
    #[serde(default = "default_check_interval")]
    check_interval: u64,
    
    // Original advanced settings
    #[serde(default = "default_min_interval")]
    min_interval: f64,
    
    #[serde(default = "default_max_interval")]
    max_interval: f64,
    
    #[serde(default = "default_max_text_length")]
    max_text_length: usize,
    
    // Original output settings
    output_file: Option<String>,
    
    // Original translation settings
    #[serde(default)]
    enable_translation: bool,
    translation_api_key: Option<String>,
    #[serde(default)]
    translation_api_type: TranslationApiType,
    target_language: Option<String>,
    
    // Original OpenAI configuration
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
    
    // Original ultra-responsive mode settings
    #[serde(default = "default_ultra_responsive")]
    ultra_responsive: bool,
    
    #[serde(default = "default_active_interval_ms")]
    active_interval_ms: u64,
    
    #[serde(default = "default_idle_interval_ms")]
    idle_interval_ms: u64,
    
    #[serde(default = "default_active_timeout_sec")]
    active_timeout_sec: f64,
    
    // Original translation batch and cache settings
    #[serde(default = "default_translation_cache_size")]
    translation_cache_size: usize,
    
    #[serde(default = "default_translation_batch_size")]
    translation_batch_size: usize,
    
    #[serde(default = "default_translation_batch_delay_ms")]
    translation_batch_delay_ms: u64,
    
    // New settings for enhanced features
    
    // Sentence processing settings
    #[serde(default = "default_min_sentence_length")]
    min_sentence_length: usize,
    
    #[serde(default = "default_max_pause_without_boundary")]
    max_pause_without_boundary: usize,
    
    #[serde(default = "default_context_history_size")]
    context_history_size: usize,
    
    // New UI settings
    #[serde(default = "default_use_enhanced_ui")]
    use_enhanced_ui: bool,
    
    #[serde(default = "default_initial_display_mode")]
    initial_display_mode: DisplayModeConfig,
    
    #[serde(default = "default_enable_colors")]
    enable_colors: bool,
    
    // Advanced translation settings
    #[serde(default = "default_min_batch_delay_ms")]
    min_batch_delay_ms: u64,
    
    #[serde(default = "default_translation_priority_threshold")]
    translation_priority_threshold: usize,
    
    #[serde(default = "default_translation_retry_count")]
    translation_retry_count: usize,
    
    #[serde(default = "default_translation_timeout_ms")]
    translation_timeout_ms: u64,
    
    // Translation quality settings
    #[serde(default = "default_use_context_for_translation")]
    use_context_for_translation: bool,
    
    #[serde(default = "default_context_prompt_template")]
    context_prompt_template: String,
    
    // Additional debugging settings
    #[serde(default)]
    debug_mode: bool,
    
    #[serde(default)]
    show_timings: bool,
    
    #[serde(default)]
    log_translations: bool,
}

// Display mode configuration
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DisplayModeConfig {
    SideBySide,
    Interleaved,
    OriginalOnly,
    TranslationOnly,
}

// Default functions for new configuration options
fn default_min_sentence_length() -> usize { 15 }
fn default_max_pause_without_boundary() -> usize { 30 }
fn default_context_history_size() -> usize { 5 }
fn default_use_enhanced_ui() -> bool { true }
fn default_initial_display_mode() -> DisplayModeConfig { DisplayModeConfig::SideBySide }
fn default_enable_colors() -> bool { true }
fn default_min_batch_delay_ms() -> u64 { 50 }
fn default_translation_priority_threshold() -> usize { 3 }
fn default_translation_retry_count() -> usize { 5 }
fn default_translation_timeout_ms() -> u64 { 10000 }
fn default_use_context_for_translation() -> bool { true }
fn default_context_prompt_template() -> String { 
    "CONTEXT: {context} TEXT TO TRANSLATE: {text}".to_string() 
}

// Convert from config to application display mode
impl From<DisplayModeConfig> for DisplayMode {
    fn from(mode: DisplayModeConfig) -> Self {
        match mode {
            DisplayModeConfig::SideBySide => DisplayMode::SideBySide,
            DisplayModeConfig::Interleaved => DisplayMode::Interleaved,
            DisplayModeConfig::OriginalOnly => DisplayMode::OriginalOnly,
            DisplayModeConfig::TranslationOnly => DisplayMode::TranslationOnly,
        }
    }
}

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

impl EnhancedConfig {
    /// Loads enhanced configuration from a file, creating a default if it doesn't exist
    fn load(path: &PathBuf) -> Result<Self, AppError> {
        if path.exists() {
            let content = fs::read_to_string(path)
                .map_err(|e| AppError::Config(format!("Failed to read config file: {:?}: {}", path, e)))?;
            
            // Try to parse as enhanced config first
            match serde_json::from_str::<EnhancedConfig>(&content) {
                Ok(config) => {
                    // Validate configuration after loading
                    config.validate()?;
                    Ok(config)
                },
                Err(e) => {
                    // Try to parse as legacy config and convert
                    match serde_json::from_str::<Config>(&content) {
                        Ok(legacy_config) => {
                            info!("Converting legacy config to enhanced config format");
                            let enhanced = Self::from_legacy(legacy_config);
                            enhanced.save(path)?;
                            Ok(enhanced)
                        },
                        Err(_) => {
                            // Both parsing attempts failed
                            return Err(AppError::Config(format!("Failed to parse config file: {:?}: {}", path, e)));
                        }
                    }
                }
            }
        } else {
            // If the configuration file doesn't exist, create a default configuration
            let config = EnhancedConfig::default();
            let content = serde_json::to_string_pretty(&config)
                .map_err(|e| AppError::Config(format!("Failed to serialize default config: {}", e)))?;
            fs::write(path, content)
                .map_err(|e| AppError::Config(format!("Failed to write default config to {:?}: {}", path, e)))?;
            info!("Created default enhanced config at {:?}", path);
            Ok(config)
        }
    }
    
    /// Converts a legacy config to the enhanced format
    fn from_legacy(legacy: Config) -> Self {
        Self {
            // Copy over original settings
            capture_interval: legacy.capture_interval,
            check_interval: legacy.check_interval,
            min_interval: legacy.min_interval,
            max_interval: legacy.max_interval,
            max_text_length: legacy.max_text_length,
            output_file: legacy.output_file,
            enable_translation: legacy.enable_translation,
            translation_api_key: legacy.translation_api_key,
            translation_api_type: legacy.translation_api_type,
            target_language: legacy.target_language,
            openai_api_url: legacy.openai_api_url,
            openai_model: legacy.openai_model,
            openai_system_prompt: legacy.openai_system_prompt,
            ultra_responsive: legacy.ultra_responsive,
            active_interval_ms: legacy.active_interval_ms,
            idle_interval_ms: legacy.idle_interval_ms,
            active_timeout_sec: legacy.active_timeout_sec,
            translation_cache_size: legacy.translation_cache_size,
            translation_batch_size: legacy.translation_batch_size.min(3), // Cap at 3 for better responsiveness
            translation_batch_delay_ms: legacy.translation_batch_delay_ms.min(200), // Cap at 200ms
            
            // Set new settings to defaults
            min_sentence_length: default_min_sentence_length(),
            max_pause_without_boundary: default_max_pause_without_boundary(),
            context_history_size: default_context_history_size(),
            use_enhanced_ui: default_use_enhanced_ui(),
            initial_display_mode: default_initial_display_mode(),
            enable_colors: default_enable_colors(),
            min_batch_delay_ms: default_min_batch_delay_ms(),
            translation_priority_threshold: default_translation_priority_threshold(),
            translation_retry_count: default_translation_retry_count(),
            translation_timeout_ms: default_translation_timeout_ms(),
            use_context_for_translation: default_use_context_for_translation(),
            context_prompt_template: default_context_prompt_template(),
            debug_mode: false,
            show_timings: false,
            log_translations: false,
        }
    }
    
    /// Saves configuration to a file
    fn save(&self, path: &PathBuf) -> Result<(), AppError> {
        // Validate before saving
        self.validate()?;
        
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| AppError::Config(format!("Failed to serialize config: {}", e)))?;
        fs::write(path, content)
            .map_err(|e| AppError::Config(format!("Failed to write config to {:?}: {}", path, e)))?;
        info!("Saved enhanced config to {:?}", path);
        Ok(())
    }
    
    /// Validates the configuration values
    fn validate(&self) -> Result<(), AppError> {
        // Validate original settings
        self.validate_basic_settings()?;
        
        // Validate new settings
        if self.min_sentence_length == 0 {
            return Err(AppError::Validation("Minimum sentence length cannot be zero".to_string()));
        }
        
        if self.max_pause_without_boundary == 0 {
            return Err(AppError::Validation("Maximum pause without boundary cannot be zero".to_string()));
        }
        
        if self.context_history_size == 0 {
            return Err(AppError::Validation("Context history size cannot be zero".to_string()));
        }
        
        if self.min_batch_delay_ms == 0 {
            return Err(AppError::Validation("Minimum batch delay cannot be zero".to_string()));
        }
        
        if self.translation_retry_count == 0 {
            return Err(AppError::Validation("Translation retry count cannot be zero".to_string()));
        }
        
        if self.translation_timeout_ms == 0 {
            return Err(AppError::Validation("Translation timeout cannot be zero".to_string()));
        }
        
        if self.use_context_for_translation && self.context_prompt_template.is_empty() {
            return Err(AppError::Validation("Context prompt template cannot be empty when context is enabled".to_string()));
        }
        
        if self.use_context_for_translation && 
           !(self.context_prompt_template.contains("{context}") && self.context_prompt_template.contains("{text}")) {
            return Err(AppError::Validation("Context prompt template must contain {context} and {text} placeholders".to_string()));
        }
        
        Ok(())
    }
    
    /// Validates the basic configuration settings
    fn validate_basic_settings(&self) -> Result<(), AppError> {
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
            
            // Validate translation batch settings
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

impl Default for EnhancedConfig {
    fn default() -> Self {
        Self {
            // Original settings with defaults
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
            translation_batch_size: default_translation_batch_size().min(3), // Cap at 3 for better responsiveness
            translation_batch_delay_ms: default_translation_batch_delay_ms().min(200), // Cap at 200ms
            
            // New settings with defaults
            min_sentence_length: default_min_sentence_length(),
            max_pause_without_boundary: default_max_pause_without_boundary(),
            context_history_size: default_context_history_size(),
            use_enhanced_ui: default_use_enhanced_ui(),
            initial_display_mode: default_initial_display_mode(),
            enable_colors: default_enable_colors(),
            min_batch_delay_ms: default_min_batch_delay_ms(),
            translation_priority_threshold: default_translation_priority_threshold(),
            translation_retry_count: default_translation_retry_count(),
            translation_timeout_ms: default_translation_timeout_ms(),
            use_context_for_translation: default_use_context_for_translation(),
            context_prompt_template: default_context_prompt_template(),
            debug_mode: false,
            show_timings: false,
            log_translations: false,
        }
    }
}

/// Enhanced sentence tracker with improved NLP-inspired boundary detection
struct SentenceTracker {
    next_id: u64,
    pending_sentences: HashMap<u64, PendingSentence>,
    buffered_text: String,
    // Context tracking for improved sentence detection
    context_history: VecDeque<String>,
    // Batch and timing configuration
    batch_size: usize,
    batch_timeout_ms: u64,
    last_batch_time: u64,
    last_text_time: u64,
    // Enhanced boundary detection configuration
    min_sentence_length: usize,
    max_pause_without_boundary: usize,
}


struct PendingSentence {
    original: String,
    timestamp: u64,
    sent_for_translation: bool,
    context: Option<String>,        // Added context field to maintain surrounding context
    acknowledgment_received: bool,  // Added for IPC acknowledgment tracking
    retry_count: usize,             // Track retry attempts
}

/// Sentence tracker for maintaining sentence-level synchronization with batching support

impl SentenceTracker {
    fn new(batch_size: usize, batch_timeout_ms: u64) -> Self {
        Self {
            next_id: 1,
            pending_sentences: HashMap::new(),
            buffered_text: String::new(),
            context_history: VecDeque::with_capacity(5), // Keep last 5 sentences for context
            batch_size,
            batch_timeout_ms,
            last_batch_time: Self::current_time_ms(),
            last_text_time: Self::current_time_ms(),
            min_sentence_length: 15,  // Minimum length to consider a complete thought
            max_pause_without_boundary: 30, // Max characters to wait before considering a pause a boundary
        }
    }
    
    /// Add new text to the buffer and extract complete sentences with improved boundary detection
    fn add_text(&mut self, text: &str) -> Vec<(u64, String, String)> {  // Returns (id, text, context)
        if !text.is_empty() {
            self.last_text_time = Self::current_time_ms();
        }
        
        self.buffered_text.push_str(text);
        
        // Extract complete sentences with enhanced NLP-inspired detection
        let mut complete_sentences = Vec::new();
        
        // Process the buffer to find sentence boundaries
        let mut sentence_start = 0;
        let mut last_boundary_candidates = Vec::new();
        let chars: Vec<char> = self.buffered_text.chars().collect();
        
        // Enhanced boundary detection state
        let mut in_quote = false;
        let mut quote_char = ' ';
        let mut acronym_state = false;
        let mut consecutive_caps = 0;
        
        for i in 0..chars.len() {
            // Track quotes for improved boundary detection (avoid splitting inside quotes)
            if chars[i] == '"' || chars[i] == '\'' || chars[i] == '"' || chars[i] == '"' || chars[i] == ''' || chars[i] == ''' {
                if !in_quote {
                    in_quote = true;
                    quote_char = chars[i];
                } else if (chars[i] == quote_char) || 
                          (quote_char == '"' && chars[i] == '"') || 
                          (quote_char == ''' && chars[i] == ''') {
                    in_quote = false;
                }
            }
            
            // Track potential acronyms (e.g., U.S.A.) to avoid treating periods as sentence boundaries
            if chars[i].is_uppercase() {
                consecutive_caps += 1;
                if i > 0 && chars[i-1] == '.' {
                    acronym_state = true;
                }
            } else if chars[i] == '.' && consecutive_caps > 0 {
                // Period after capital letter might be part of an acronym
                acronym_state = true;
            } else {
                consecutive_caps = 0;
                if chars[i] != '.' {
                    acronym_state = false;
                }
            }
            
            // Primary sentence boundary detection
            let is_sentence_end = 
                // Traditional end markers, but not in acronyms
                ((chars[i] == '.' || chars[i] == '!' || chars[i] == '?') && !acronym_state && 
                 // Look ahead to confirm it's really the end (space or new paragraph)
                 (i + 1 >= chars.len() || chars[i+1] == ' ' || chars[i+1] == '\n')) ||
                // Line breaks as potential boundaries, especially multiple breaks
                (chars[i] == '\n' && (i + 1 >= chars.len() || chars[i+1] == '\n')) ||
                // Em-dash and other punctuation that might indicate a sentence break
                (chars[i] == '—' && i + 1 < chars.len() && chars[i+1] == ' ');
            
            // Secondary boundary candidates (for soft breaks)
            let is_boundary_candidate = 
                // Various pause indicators
                (chars[i] == ',' || chars[i] == ';' || chars[i] == ':') && !in_quote && i + 1 < chars.len() && chars[i+1] == ' ';
            
            if is_boundary_candidate {
                last_boundary_candidates.push(i);
                // Keep only the 3 most recent candidates
                if last_boundary_candidates.len() > 3 {
                    last_boundary_candidates.remove(0);
                }
            }
            
            if is_sentence_end || 
               // Handle soft breaks based on length constraints
               (i - sentence_start > self.max_pause_without_boundary && !last_boundary_candidates.is_empty() && !in_quote) ||
               // Long text segment without any punctuation needs to be broken
               (i - sentence_start > 80 && chars[i] == ' ' && !in_quote) {
                
                // Determine the best boundary position
                let sentence_end = if is_sentence_end {
                    i
                } else if !last_boundary_candidates.is_empty() {
                    // Use the most recent boundary candidate
                    *last_boundary_candidates.last().unwrap()
                } else {
                    // Fallback to the current position if we have to break
                    i
                };
                
                // Process the sentence if there's content
                if sentence_end >= sentence_start && sentence_end - sentence_start >= self.min_sentence_length {
                    let sentence = self.buffered_text[sentence_start..=sentence_end].to_string();
                    
                    // Skip empty or whitespace-only sentences
                    if !sentence.trim().is_empty() {
                        let id = self.next_id;
                        self.next_id += 1;
                        
                        // Create context from recent history
                        let context = self.build_context();
                        
                        // Store and prepare for translation
                        self.pending_sentences.insert(id, PendingSentence {
                            original: sentence.clone(),
                            timestamp: Self::current_time_ms(),
                            sent_for_translation: false,
                            context: Some(context.clone()),
                            acknowledgment_received: false,
                            retry_count: 0,
                        });
                        
                        // Add this sentence to the history for future context
                        self.update_context_history(&sentence);
                        
                        complete_sentences.push((id, sentence, context));
                    }
                    
                    // Move the start position for the next sentence
                    sentence_start = sentence_end + 1;
                    
                    // Reset boundary candidates after creating a sentence
                    last_boundary_candidates.clear();
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
    /// Build context from the recent sentence history
    fn build_context(&self) -> String {
        self.context_history.iter().take(3).cloned().collect::<Vec<_>>().join(" ")
    }
    
    /// Update context history with a new sentence
    fn update_context_history(&mut self, sentence: &str) {
        if self.context_history.len() >= 5 {
            self.context_history.pop_front();
        }
        self.context_history.push_back(sentence.to_string());
    }
    
    /// Check if the current buffer should be sent even if incomplete, with improved logic
    fn should_send_incomplete(&self) -> bool {
        // Use more sophisticated logic to decide when to send incomplete sentences
        let buffer_len = self.buffered_text.len();
        
        (buffer_len >= self.min_sentence_length && Self::current_time_ms() - self.last_text_time > 500) ||  // Short pause
        (buffer_len > 0 && Self::current_time_ms() - self.last_text_time > 1000) ||  // Longer pause, any content
        (!self.pending_sentences.is_empty() && buffer_len > 0 && 
         Self::current_time_ms() - self.last_batch_time > 1500)  // Been a while since last batch, and we have buffer
    }
    
    /// Get the current incomplete sentence for sending, with context
    fn get_incomplete_sentence(&mut self) -> Option<(u64, String, String)> {
        if self.buffered_text.is_empty() {
            return None;
        }
        
        let id = self.next_id;
        self.next_id += 1;
        
        // Create context from recent history
        let context = self.build_context();
        
        // Store and prepare for sending
        self.pending_sentences.insert(id, PendingSentence {
            original: self.buffered_text.clone(),
            timestamp: Self::current_time_ms(),
            sent_for_translation: false,
            context: Some(context.clone()),
            acknowledgment_received: false,
            retry_count: 0,
        });
        
        // Update history for future context
        self.update_context_history(&self.buffered_text);
        
        let result = (id, self.buffered_text.clone(), context);
        self.buffered_text.clear();
        Some(result)
    }
    /// Mark a sentence as having received acknowledgment
    fn mark_acknowledgment_received(&mut self, id: u64) -> bool {
        if let Some(sentence) = self.pending_sentences.get_mut(&id) {
            sentence.acknowledgment_received = true;
            return true;
        }
        false
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

/// Enhanced IPC messages with acknowledgment and improved robustness
#[derive(Debug, Serialize, Deserialize, Clone)]
enum IpcMessage {
    /// Configuration message
    Config(Config),
    
    /// Enhanced text message with metadata for better synchronization
    Text {
        id: u64,                // Unique identifier for this text segment
        content: String,        // The text content to translate
        timestamp: u64,         // Timestamp when the text was captured
        is_complete: bool,      // Whether this is a complete sentence
        context: String,        // Context for improved translation
        sequence_number: u64,   // Sequence number for ordering
    },
    
    /// New acknowledgment message for reliability
    Acknowledgment {
        id: u64,                // The id being acknowledged
        status: AckStatus,      // Status of the acknowledged message
        timestamp: u64,         // Timestamp of acknowledgment
    },
    
    /// Enhanced status message for synchronization
    Status {
        pending_count: usize,        // Number of pending translations
        completed_count: usize,      // Number of completed translations
        translation_active: bool,    // Whether translation is actively processing
        timestamp: u64,              // Status timestamp
    },
    
    /// Shutdown message
    Shutdown,
}
/// Acknowledgment status for IPC reliability
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
enum AckStatus {
    Received,         // Message was received
    Processing,       // Message is being processed
    Completed,        // Translation completed successfully
    Failed(String),   // Translation failed with reason
}
/// Enhanced named pipe implementation with message queue and reliability
struct EnhancedNamedPipe {
    handle: HANDLE,
    message_queue: Arc<Mutex<VecDeque<QueuedMessage>>>,
    ack_status: Arc<Mutex<HashMap<u64, AckStatus>>>,
    last_sequence: Arc<AtomicU64>,
}
struct QueuedMessage {
    message: IpcMessage,
    first_attempt: u64,  // Timestamp of first attempt
    last_attempt: u64,   // Timestamp of last attempt
    attempts: usize,     // Number of attempts
    max_attempts: usize, // Maximum number of attempts
    priority: u8,        // Priority (higher = more important)
}

impl EnhancedNamedPipe {
    /// Creates a new named pipe server with enhanced reliability
    fn create_server(pipe_name: &str) -> Result<Self, AppError> {
        let pipe_path = format!(r"\\.\pipe\{}", pipe_name);
        
        let handle = unsafe {
            CreateNamedPipeW(
                &HSTRING::from(pipe_path),
                windows::Win32::Storage::FileSystem::FILE_FLAGS_AND_ATTRIBUTES(0x00000003), // PIPE_ACCESS_DUPLEX
                PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT,
                1, // Max instances
                8192, // Increased output buffer size
                8192, // Increased input buffer size
                0, // Default timeout
                None, // Default security attributes
            )
        };
        
        if handle.is_invalid() {
            return Err(AppError::Io(io::Error::last_os_error()));
        }
        
        Ok(Self { 
            handle,
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            ack_status: Arc::new(Mutex::new(HashMap::new())),
            last_sequence: Arc::new(AtomicU64::new(0)),
        })
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
        
        Ok(Self { 
            handle,
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            ack_status: Arc::new(Mutex::new(HashMap::new())),
            last_sequence: Arc::new(AtomicU64::new(0)),
        })
    }
    
    /// Starts the background message processing thread
    fn start_message_processor(&self, pipe_clone: Arc<Self>) {
        std::thread::spawn(move || {
            let mut retry_interval = Duration::from_millis(50);
            
            loop {
                // Process queued messages with priority and timing logic
                let maybe_message = {
                    let mut queue = pipe_clone.message_queue.lock().unwrap();
                    
                    if queue.is_empty() {
                        None
                    } else {
                        // Find the highest priority message that's due for retry
                        let current_time = Self::current_time_ms();
                        let mut best_index = None;
                        let mut best_priority = 0;
                        
                        for (i, msg) in queue.iter().enumerate() {
                            // Calculate if this message is due for retry (exponential backoff)
                            let backoff = 50 * (1 << msg.attempts.min(6));  // Cap at 2^6 = 64
                            let due_time = msg.last_attempt + backoff;
                            
                            if current_time >= due_time && msg.priority >= best_priority {
                                best_index = Some(i);
                                best_priority = msg.priority;
                            }
                        }
                        
                        // Remove and return the best message if found
                        if let Some(idx) = best_index {
                            Some(queue.remove(idx).unwrap())
                        } else {
                            None
                        }
                    }
                };
                
                if let Some(mut queued_msg) = maybe_message {
                    // Attempt to send the message
                    match pipe_clone.write_message_internal(&queued_msg.message) {
                        Ok(()) => {
                            // Successfully sent
                            if let IpcMessage::Text { id, .. } = &queued_msg.message {
                                let mut ack_map = pipe_clone.ack_status.lock().unwrap();
                                ack_map.insert(*id, AckStatus::Received);
                            }
                        },
                        Err(e) => {
                            // Failed to send, requeue with increased attempts
                            queued_msg.attempts += 1;
                            queued_msg.last_attempt = Self::current_time_ms();
                            
                            // Only requeue if under max attempts
                            if queued_msg.attempts < queued_msg.max_attempts {
                                let mut queue = pipe_clone.message_queue.lock().unwrap();
                                queue.push_back(queued_msg);
                            } else {
                                // Log final failure
                                error!("Failed to send message after {} attempts: {:?}", 
                                      queued_msg.max_attempts, e);
                                
                                // Update status map for text messages
                                if let IpcMessage::Text { id, .. } = &queued_msg.message {
                                    let mut ack_map = pipe_clone.ack_status.lock().unwrap();
                                    ack_map.insert(*id, AckStatus::Failed(format!("Send failed: {}", e)));
                                }
                            }
                        }
                    }
                }
                
                // Sleep to avoid spinning too fast when queue is empty
                std::thread::sleep(retry_interval);
                
                // Adaptive sleep - shorter when queue has items, longer when empty
                {
                    let queue_size = pipe_clone.message_queue.lock().unwrap().len();
                    retry_interval = if queue_size > 0 {
                        Duration::from_millis(20)  // Process quickly when busy
                    } else {
                        Duration::from_millis(100) // Sleep longer when idle
                    };
                }
            }
        });
    }
    
    /// Queues a message with reliability guarantees
    fn queue_message(&self, message: IpcMessage, priority: u8, max_attempts: usize) {
        let queued_message = QueuedMessage {
            message,
            first_attempt: Self::current_time_ms(),
            last_attempt: Self::current_time_ms(),
            attempts: 0,
            max_attempts,
            priority,
        };
        
        let mut queue = self.message_queue.lock().unwrap();
        queue.push_back(queued_message);
    }
    
    /// Assigns a sequence number to a message
    fn get_next_sequence(&self) -> u64 {
        self.last_sequence.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
    
    /// Waits for acknowledgment of a specific message
    async fn wait_for_acknowledgment(&self, id: u64, timeout_ms: u64) -> Result<AckStatus, AppError> {
        let start = Self::current_time_ms();
        
        while Self::current_time_ms() - start < timeout_ms {
            {
                let ack_map = self.ack_status.lock().unwrap();
                if let Some(status) = ack_map.get(&id) {
                    return Ok(status.clone());
                }
            }
            
            // Wait a bit before checking again
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        
        Err(AppError::Task(format!("Acknowledgment timeout for message {}", id)))
    }
    
    /// Internal direct write method
    fn write_message_internal(&self, message: &IpcMessage) -> Result<(), AppError> {
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
    
    /// Sends a text message with reliability and acknowledgment
    async fn send_text_with_ack(
        &self, 
        id: u64, 
        content: String, 
        context: String,
        is_complete: bool,
        timeout_ms: u64
    ) -> Result<AckStatus, AppError> {
        // Create the text message with sequence number
        let seq = self.get_next_sequence();
        let message = IpcMessage::Text {
            id,
            content,
            timestamp: Self::current_time_ms(),
            is_complete,
            context,
            sequence_number: seq,
        };
        
        // Queue with high priority and retries
        self.queue_message(message, 2, 10);
        
        // Wait for acknowledgment with timeout
        self.wait_for_acknowledgment(id, timeout_ms).await
    }
    
    /// Current timestamp in milliseconds
    fn current_time_ms() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
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

/// Updated command line arguments to support enhanced features
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct EnhancedArgs {
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
    
    /// Whether to use enhanced UI
    #[arg(short = 'e', long)]
    enhanced_ui: Option<bool>,
    
    /// Initial display mode (side_by_side, interleaved, original_only, translation_only)
    #[arg(long)]
    display_mode: Option<String>,
    
    /// Whether to enable debug mode
    #[arg(long)]
    debug: Option<bool>,
    
    /// Whether to use context for translation
    #[arg(long)]
    use_context: Option<bool>,
    
    /// Translation batch size (1-10)
    #[arg(long)]
    batch_size: Option<usize>,
    
    /// Translation batch delay in milliseconds (50-1000)
    #[arg(long)]
    batch_delay: Option<u64>,
}

/// Enhanced engine creation with the new configuration format
async fn create_enhanced_engine() -> Result<Engine, AppError> {
    // Parse command line arguments
    let args = EnhancedArgs::parse();
    info!("get-livecaptions starting with enhanced features");
    
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
    
    // Load or create enhanced configuration
    let mut config = EnhancedConfig::load(&config_path)?;
    
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
    
    // Enhanced feature overrides
    if let Some(enhanced_ui) = args.enhanced_ui {
        config.use_enhanced_ui = enhanced_ui;
    }
    if let Some(mode) = args.display_mode {
        config.initial_display_mode = match mode.to_lowercase().as_str() {
            "side_by_side" => DisplayModeConfig::SideBySide,
            "interleaved" => DisplayModeConfig::Interleaved,
            "original_only" => DisplayModeConfig::OriginalOnly,
            "translation_only" => DisplayModeConfig::TranslationOnly,
            _ => {
                warn!("Unknown display mode: {}, using default", mode);
                config.initial_display_mode
            }
        };
    }
    if let Some(debug) = args.debug {
        config.debug_mode = debug;
    }
    if let Some(use_context) = args.use_context {
        config.use_context_for_translation = use_context;
    }
    if let Some(batch_size) = args.batch_size {
        config.translation_batch_size = batch_size.clamp(1, 10); // Reasonable limits
    }
    if let Some(batch_delay) = args.batch_delay {
        config.translation_batch_delay_ms = batch_delay.clamp(50, 1000); // Reasonable limits
    }
    
    // Validate the configuration
    config.validate()?;
    
    // Save updated configuration
    config.save(&config_path)?;
    
    // Configure logging based on debug mode
    if config.debug_mode {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();
        debug!("Debug mode enabled");
    }
    
    // Create sentence tracker with enhanced settings
    let sentence_tracker = SentenceTracker::new_with_settings(
        config.translation_batch_size,
        config.translation_batch_delay_ms,
        config.min_sentence_length,
        config.max_pause_without_boundary,
        config.context_history_size
    );
    
    // Create optimized translation processor if enabled
    let translation_processor = if config.enable_translation {
        if let Some(api_key) = &config.translation_api_key {
            // Ensure target language is set
            let target_lang = config.target_language.clone().unwrap_or_else(|| {
                match config.translation_api_type {
                    TranslationApiType::DeepL => "ZH".to_string(),
                    TranslationApiType::OpenAI => "Chinese".to_string(),
                    _ => "zh-CN".to_string(),
                }
            });
            
            info!("Initializing optimized translation service with {:?} API", config.translation_api_type);
            Some(OptimizedBatchTranslationProcessor::new(
                create_enhanced_translation_service(
                    api_key.clone(),
                    target_lang,
                    config.translation_api_type,
                    config.openai_api_url.clone(),
                    config.openai_model.clone(),
                    config.openai_system_prompt.clone(),
                    config.use_context_for_translation,
                    config.context_prompt_template.clone()
                ),
                config.translation_cache_size,
                config.translation_batch_size,
                config.translation_batch_delay_ms,
                config.min_batch_delay_ms
            ))
        } else if config.translation_api_type == TranslationApiType::Demo {
            // Demo mode doesn't need an API key
            let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
            Some(OptimizedBatchTranslationProcessor::new(
                Arc::new(DemoTranslation {
                    target_language: target_lang,
                }),
                config.translation_cache_size,
                config.translation_batch_size,
                config.translation_batch_delay_ms,
                config.min_batch_delay_ms
            ))
        } else {
            warn!("Translation enabled but no API key provided");
            None
        }
    } else {
        None
    };
    
    // Create and initialize the engine with enhanced configuration
    Engine::new_with_enhanced_config(
        config,
        translation_processor,
        sentence_tracker
    ).await
}

/// Create an enhanced translation service with context support
fn create_enhanced_translation_service(
    api_key: String,
    target_language: String,
    api_type: TranslationApiType,
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
    use_context: bool,
    context_prompt_template: String,
) -> Arc<dyn TranslationService> {
    match api_type {
        TranslationApiType::DeepL => {
            Arc::new(EnhancedDeepLTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(300)),
                use_context,
                context_prompt_template,
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::Generic => {
            // Similarly enhanced implementation for Generic
            Arc::new(EnhancedGenericTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(300)),
                use_context,
                context_prompt_template,
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::OpenAI => {
            Arc::new(EnhancedOpenAITranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                api_url: openai_api_url.unwrap_or_else(|| "https://api.openai.com/v1/chat/completions".to_string()),
                model: openai_model.unwrap_or_else(|| "gpt-3.5-turbo".to_string()),
                system_prompt: openai_system_prompt.unwrap_or_else(|| 
                    "You are a translator. Translate the following text to the target language. Only respond with the translation, no explanations.".to_string()
                ),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(600)),
                use_context,
                context_prompt_template,
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::Demo => {
            Arc::new(EnhancedDemoTranslation {
                target_language,
                use_context,
                context_prompt_template,
            }) as Arc<dyn TranslationService>
        },
    }
}

/// Run the enhanced display thread with termion UI
async fn run_enhanced_display_thread(
    config: EnhancedConfig,
    mut receiver: mpsc::Receiver<DisplayCommand>
) -> Result<(), AppError> {
    info!("Enhanced translation display thread started");
    
    // Create translation window with initial display mode from config
    let mut window = EnhancedTranslationWindow::new_with_mode(DisplayMode::from(config.initial_display_mode));
    
    // Initialize terminal UI if enabled
    let mut terminal = if config.use_enhanced_ui {
        match terminal::TerminalUI::new() {
            Ok(mut term) => {
                // Initialize with alternate screen
                if let Err(e) = term.init(true) {
                    warn!("Failed to initialize terminal UI: {}, falling back to basic UI", e);
                    None
                } else {
                    Some(term)
                }
            },
            Err(e) => {
                warn!("Failed to create terminal UI: {}, falling back to basic UI", e);
                None
            }
        }
    } else {
        None
    };
    
    // For basic UI mode, clear screen initially
    if terminal.is_none() {
        print!("\x1B[2J\x1B[1;1H");
        window.update_display();
    } else if let Some(term) = &terminal {
        window.draw_with_termion(term)?;
    }
    
    let mut check_input_interval = tokio::time::interval(Duration::from_millis(50));
    let mut last_update = Instant::now();
    
    loop {
        tokio::select! {
            maybe_command = receiver.recv() => {
                match maybe_command {
                    Some(DisplayCommand::UpdateContent(update)) => {
                        // Update translation content
                        process_translation_update(&mut window, update);
                        
                        // Update display if enough time has passed to avoid flickering
                        if last_update.elapsed() >= Duration::from_millis(50) {
                            if let Some(term) = &terminal {
                                window.draw_with_termion(term)?;
                            } else {
                                window.update_display();
                            }
                            last_update = Instant::now();
                        }
                    },
                    Some(DisplayCommand::ChangeMode(mode)) => {
                        window.display_mode = mode;
                        if let Some(term) = &terminal {
                            window.draw_with_termion(term)?;
                        } else {
                            window.update_display();
                        }
                    },
                    Some(DisplayCommand::Scroll(direction)) => {
                        apply_scroll(&mut window, direction);
                        if let Some(term) = &terminal {
                            window.draw_with_termion(term)?;
                        } else {
                            window.update_display();
                        }
                    },
                    Some(DisplayCommand::Shutdown) => {
                        info!("Display thread received shutdown command");
                        break;
                    },
                    None => {
                        info!("Display control channel closed, shutting down");
                        break;
                    }
                }
            },
            _ = check_input_interval.tick() => {
                // Check for keyboard input
                let quit = if let Some(term) = &terminal {
                    window.process_termion_input(term)?
                } else {
                    match window.process_input() {
                        Ok(quit) => quit,
                        Err(e) => {
                            warn!("Error processing input: {}", e);
                            false
                        }
                    }
                };
                
                if quit {
                    info!("User requested quit from translation window");
                    break;
                }
                
                // Update window periodically to animate waiting indicators
                if last_update.elapsed() >= Duration::from_millis(100) {
                    if let Some(term) = &terminal {
                        window.draw_with_termion(term)?;
                    } else {
                        window.update_display();
                    }
                    last_update = Instant::now();
                }
            }
        }
    }
    
    // Clean up terminal UI if used
    if let Some(mut term) = terminal.take() {
        let _ = term.cleanup();
    }
    
    info!("Translation display thread terminated");
    Ok(())
}

/// Process a translation update command for the window
fn process_translation_update(window: &mut EnhancedTranslationWindow, update: TranslationUpdate) {
    match update.status {
        TranslationStatus::Pending => {
            // Add as pending
            window.pending_translations.insert(update.id, PendingTranslation {
                content: update.original.clone(),
                timestamp: update.timestamp,
                is_complete: update.is_complete,
                sequence_number: update.sequence_number,
                context: update.context,
                request_sent: SentenceTracker::current_time_ms(),
            });
            window.status_indicators.insert(update.id, TranslationStatus::Pending);
        },
        TranslationStatus::InProgress => {
            // Update status to in-progress
            window.status_indicators.insert(update.id, TranslationStatus::InProgress);
        },
        TranslationStatus::Completed => {
            // Add completed translation
            if let Some(translation) = update.translation {
                window.sentences.insert(update.id, TranslatedSentence {
                    id: update.id,
                    original: update.original,
                    translation,
                    timestamp: update.timestamp,
                    is_complete: update.is_complete,
                    sequence_number: update.sequence_number,
                    processing_time_ms: 0, // Could calculate if needed
                    context: update.context,
                });
                
                // Remove from pending and update status
                window.pending_translations.remove(&update.id);
                window.status_indicators.insert(update.id, TranslationStatus::Completed);
            }
        },
        TranslationStatus::Failed => {
            // Mark as failed
            window.status_indicators.insert(update.id, TranslationStatus::Failed);
        },
    }
}

/// Apply scroll command to the window
fn apply_scroll(window: &mut EnhancedTranslationWindow, direction: ScrollDirection) {
    match direction {
        ScrollDirection::Up => window.scroll_position = window.scroll_position.saturating_sub(1),
        ScrollDirection::Down => window.scroll_position += 1,
        ScrollDirection::PageUp => window.scroll_position = window.scroll_position.saturating_sub(10),
        ScrollDirection::PageDown => window.scroll_position += 10,
    }
}

/// Enhanced translation service implementations with context support
struct EnhancedDeepLTranslation {
    api_key: String,
    target_language: String,
    client: reqwest::Client,
    rate_limiter: tokio::sync::Mutex<RateLimiter>,
    use_context: bool,
    context_prompt_template: String,
}

#[async_trait::async_trait]
impl TranslationService for EnhancedDeepLTranslation {
    async fn translate(&self, text_input: &str) -> Result<String, AppError> {
        // Rate limiting
        self.rate_limiter.lock().await.wait().await;
        
        // Process text input based on context settings
        let (text_to_translate, has_context) = self.process_input_text(text_input);
        
        // DeepL API call with improved error handling and retry logic
        let url = "https://api-free.deepl.com/v2/translate";
        
        // Try up to 3 times with exponential backoff
        let mut retry_count = 0;
        let max_retries = 3;
        let mut last_error = None;
        
        while retry_count < max_retries {
            match self.client.post(url)
                .timeout(Duration::from_secs(5))
                .header("Authorization", format!("DeepL-Auth-Key {}", self.api_key))
                .form(&[
                    ("text", &text_to_translate),
                    ("target_lang", &self.target_language),
                    ("source_lang", "EN"),
                    // Use formality setting for better translations
                    ("formality", "default"),
                ])
                .send()
                .await
            {
                Ok(response) => {
                    // Handle response status
                    if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
                        retry_count += 1;
                        let backoff = Duration::from_millis(300 * 2u64.pow(retry_count as u32));
                        warn!("DeepL rate limit reached, backing off for {}ms", backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if response.status().is_server_error() {
                        retry_count += 1;
                        let backoff = Duration::from_millis(200 * 2u64.pow(retry_count as u32));
                        warn!("DeepL server error {}, backing off for {}ms", 
                              response.status(), backoff.as_millis());
                        tokio::time::sleep(backoff).await;
                        continue;
                    } else if !response.status().is_success() {
                        return Err(AppError::Translation(
                            format!("DeepL API error: {} - {}", 
                                    response.status(), 
                                    response.text().await.unwrap_or_default())
                        ));
                    }
                    
                    // Parse successful response
                    #[derive(Deserialize)]
                    struct DeepLResponse {
                        translations: Vec<Translation>,
                    }
                    
                    #[derive(Deserialize)]
                    struct Translation {
                        text: String,
                    }
                    
                    match response.json::<DeepLResponse>().await {
                        Ok(result) => {
                            if let Some(translation) = result.translations.first() {
                                // Process the result if context was included
                                let final_result = if has_context {
                                    self.extract_translation_from_context(&translation.text)
                                } else {
                                    translation.text.clone()
                                };
                                
                                return Ok(final_result);
                            } else {
                                return Err(AppError::Translation("Empty translation result from DeepL".to_string()));
                            }
                        },
                        Err(e) => {
                            retry_count += 1;
                            last_error = Some(AppError::Translation(
                                format!("Failed to parse DeepL response: {}", e)));
                            
                            if retry_count < max_retries {
                                let backoff = Duration::from_millis(100 * 2u64.pow(retry_count as u32));
                                tokio::time::sleep(backoff).await;
                            }
                        }
                    }
                },
                Err(e) => {
                    retry_count += 1;
                    last_error = Some(AppError::Translation(
                        format!("Failed to send translation request to DeepL: {}", e)));
                    
                    if retry_count < max_retries {
                        let backoff = Duration::from_millis(100 * 2u64.pow(retry_count as u32));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        }
        
        // If we got here, all retries failed
        Err(last_error.unwrap_or_else(|| 
            AppError::Translation("Unknown error after retries".to_string())))
    }
    
    // Implementation for batch_translate and other methods
    // ...
    
    fn get_name(&self) -> &str {
        "EnhancedDeepL"
    }
    
    fn get_target_language(&self) -> &str {
        &self.target_language
    }
}

impl EnhancedDeepLTranslation {
    /// Process input text, handling context if included
    fn process_input_text(&self, text: &str) -> (String, bool) {
        // Check if this is already in our context format from upstream
        if text.contains("CONTEXT:") && text.contains("TEXT TO TRANSLATE:") {
            // Already formatted, pass through
            return (text.to_string(), true);
        }
        
        // Check if we should use context
        if !self.use_context {
            return (text.to_string(), false);
        }
        
        // Check if context is included in structured format (from SentenceTracker)
        if let Some(context_start) = text.find("CONTEXT:") {
            if let Some(text_start) = text.find("TEXT TO TRANSLATE:") {
                // Parse context and text
                let context = text[context_start + 8..text_start].trim();
                let translate_text = text[text_start + 18..].trim();
                
                // Use template format
                let formatted = self.context_prompt_template
                    .replace("{context}", context)
                    .replace("{text}", translate_text);
                
                return (formatted, true);
            }
        }
        
        // No context detected, just return text
        (text.to_string(), false)
    }
    
    /// Extract translation from context-aware response
    fn extract_translation_from_context(&self, result: &str) -> String {
        // For DeepL, we might need to post-process the translation to remove any context markers
        // This depends on how DeepL handles the context in the prompt
        
        // Attempt to find markers
        if let Some(idx) = result.rfind("TEXT TO TRANSLATE:") {
            // Extract only the translated part
            return result[idx + "TEXT TO TRANSLATE:".len()..].trim().to_string();
        }
        
        // If no markers found, return as is
        result.to_string()
    }
}

/// Enhanced Demo translation implementation
struct EnhancedDemoTranslation {
    target_language: String,
    use_context: bool,
    context_prompt_template: String,
}

#[async_trait::async_trait]
impl TranslationService for EnhancedDemoTranslation {
    async fn translate(&self, text_input: &str) -> Result<String, AppError> {
        // Process text input based on context settings
        let (text_to_translate, has_context) = self.process_input_text(text_input);
        
        // In demo mode, just prepend the target language
        let result = format!("[{} Translation]: {}", self.target_language, text_to_translate);
        
        // Process the result if context was included
        let final_result = if has_context {
            self.extract_translation_from_context(&result)
        } else {
            result
        };
        
        Ok(final_result)
    }
    
    // Similar implementations for batch_translate, get_name, etc.
    // ...
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
struct EnhancedTranslationWindow {
    sentences: BTreeMap<u64, TranslatedSentence>,
    pending_translations: HashMap<u64, PendingTranslation>,
    status_indicators: HashMap<u64, TranslationStatus>,
    display_mode: DisplayMode,
    terminal_width: usize,
    terminal_height: usize,
    scroll_position: usize,
    colors_enabled: bool,
}

/// Enhanced translated sentence structure
struct TranslatedSentence {
    id: u64,
    original: String,
    translation: String,
    timestamp: u64,
    is_complete: bool,
    sequence_number: u64,
    processing_time_ms: u64,   // Time taken to translate
    context: String,           // Context used for translation
}

/// Enhanced pending translation tracking
struct PendingTranslation {
    content: String,
    timestamp: u64,
    is_complete: bool,
    sequence_number: u64,
    context: String,
    request_sent: u64,         // When the request was sent
}

/// Translation status for visual indicators
#[derive(Debug, Clone, Copy, PartialEq)]
enum TranslationStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// Display modes for the UI
enum DisplayMode {
    SideBySide,
    Interleaved,
    OriginalOnly,
    TranslationOnly,
}
impl EnhancedTranslationWindow {
    fn new() -> Self {
        // Try to get terminal size
        let (width, height) = match term_size::dimensions() {
            Some((w, h)) => (w, h),
            None => (80, 24),  // Default fallback
        };
        
        // Check if colors are supported
        let colors_enabled = atty::is(atty::Stream::Stdout);
        
        Self {
            sentences: BTreeMap::new(),
            pending_translations: HashMap::new(),
            status_indicators: HashMap::new(),
            display_mode: DisplayMode::SideBySide,
            terminal_width: width,
            terminal_height: height,
            scroll_position: 0,
            colors_enabled,
        }
    }
    
    /// Process a new text message with improved context handling
    async fn process_text_message(
        &mut self, 
        id: u64, 
        content: String, 
        timestamp: u64,
        is_complete: bool,
        context: String,
        sequence_number: u64,
        translation_processor: &BatchTranslationProcessor
    ) -> Result<(), AppError> {
        // First check if this is a duplicate
        if self.sentences.contains_key(&id) {
            return Ok(());
        }
        
        // Add to pending translations with enhanced tracking
        self.pending_translations.insert(id, PendingTranslation {
            content: content.clone(),
            timestamp,
            is_complete,
            sequence_number,
            context: context.clone(),
            request_sent: SentenceTracker::current_time_ms(),
        });
        
        // Update status indicator
        self.status_indicators.insert(id, TranslationStatus::Pending);
        
        // Update the display to show pending status
        self.update_display();
        
        // Start translation with context
        let translation_start = SentenceTracker::current_time_ms();
        self.status_indicators.insert(id, TranslationStatus::InProgress);
        self.update_display();
        
        // Combine context with content for better translation quality
        let translation_text = if !context.is_empty() {
            format!("CONTEXT: {} TEXT TO TRANSLATE: {}", context, content)
        } else {
            content.clone()
        };
        
        // Translate the text with improved handling and retries
        match translation_processor.translate(&translation_text).await {
            Ok(mut translated) => {
                let translation_time = SentenceTracker::current_time_ms() - translation_start;
                
                // Remove any "CONTEXT:" prompt that might be in the response
                if translated.contains("TEXT TO TRANSLATE:") {
                    if let Some(idx) = translated.find("TEXT TO TRANSLATE:") {
                        translated = translated[(idx + "TEXT TO TRANSLATE:".len())..].trim().to_string();
                    }
                }
                
                // Store the translated sentence with enhanced metadata
                self.sentences.insert(id, TranslatedSentence {
                    id,
                    original: content,
                    translation: translated,
                    timestamp,
                    is_complete,
                    sequence_number,
                    processing_time_ms: translation_time,
                    context,
                });
                
                // Remove from pending and update status
                self.pending_translations.remove(&id);
                self.status_indicators.insert(id, TranslationStatus::Completed);
                
                // Update the displayed text
                self.update_display();
                Ok(())
            },
            Err(e) => {
                self.status_indicators.insert(id, TranslationStatus::Failed);
                self.update_display();
                Err(e)
            }
        }
    }
    
    /// Update the displayed text with advanced formatting and visual indicators
    fn update_display(&mut self) {
        // Clear screen for redraw
        print!("\x1B[2J\x1B[1;1H");
        
        // Print header with status
        self.print_header();
        
        // Display based on current mode
        match self.display_mode {
            DisplayMode::SideBySide => self.display_side_by_side(),
            DisplayMode::Interleaved => self.display_interleaved(),
            DisplayMode::OriginalOnly => self.display_original_only(),
            DisplayMode::TranslationOnly => self.display_translation_only(),
        }
        
        // Display help footer
        self.print_footer();
        
        // Flush output
        io::stdout().flush().unwrap_or_else(|e| {
            warn!("Failed to flush display: {}", e);
        });
    }
    
    /// Print header with stats
    fn print_header(&self) {
        let completed = self.sentences.len();
        let pending = self.pending_translations.len();
        let total = completed + pending;
        
        if self.colors_enabled {
            println!("\x1B[1;36mEnhanced Translation Window\x1B[0m");
            println!("\x1B[1;33m────────────────────────────────────────────────\x1B[0m");
            println!("\x1B[1;32mStatus:\x1B[0m Completed: \x1B[1;32m{}\x1B[0m Pending: \x1B[1;33m{}\x1B[0m Total: \x1B[1;34m{}\x1B[0m", 
                     completed, pending, total);
        } else {
            println!("Enhanced Translation Window");
            println!("────────────────────────────────────────────────");
            println!("Status: Completed: {} Pending: {} Total: {}", 
                     completed, pending, total);
        }
        
        // Display current mode
        let mode_str = match self.display_mode {
            DisplayMode::SideBySide => "Side by Side",
            DisplayMode::Interleaved => "Interleaved",
            DisplayMode::OriginalOnly => "Original Only",
            DisplayMode::TranslationOnly => "Translation Only",
        };
        println!("Display Mode: {}", mode_str);
        println!();
    }
    
    /// Print help footer
    fn print_footer(&self) {
        if self.colors_enabled {
            println!("\n\x1B[1;33m────────────────────────────────────────────────\x1B[0m");
            println!("\x1B[1;36mControls:\x1B[0m [1] Side by Side [2] Interleaved [3] Original [4] Translation [↑/↓] Scroll");
        } else {
            println!("\n────────────────────────────────────────────────");
            println!("Controls: [1] Side by Side [2] Interleaved [3] Original [4] Translation [↑/↓] Scroll");
        }
    }
    
    /// Display translations side by side with original text
    fn display_side_by_side(&self) {
        // Calculate column widths
        let col_width = (self.terminal_width / 2).saturating_sub(2);
        
        // Print column headers
        if self.colors_enabled {
            println!("\x1B[1;37m{:<width$} | {:<width$}\x1B[0m", 
                     "Original Text", "Translation", width=col_width);
            println!("\x1B[1;33m{}\x1B[0m", "─".repeat(self.terminal_width));
        } else {
            println!("{:<width$} | {:<width$}", 
                     "Original Text", "Translation", width=col_width);
            println!("{}", "─".repeat(self.terminal_width));
        }
        
        // Create a combined view of all sentences and pending translations
        let mut all_entries: Vec<(u64, Option<&TranslatedSentence>, Option<&PendingTranslation>)> = Vec::new();
        
        // Add completed translations
        for (id, sentence) in &self.sentences {
            all_entries.push((*id, Some(sentence), None));
        }
        
        // Add pending translations
        for (id, pending) in &self.pending_translations {
            // Only add if not already in completed
            if !self.sentences.contains_key(id) {
                all_entries.push((*id, None, Some(pending)));
            }
        }
        
        // Sort by sequence number for proper ordering
        all_entries.sort_by(|a, b| {
            let a_seq = match (a.1, a.2) {
                (Some(s), _) => s.sequence_number,
                (_, Some(p)) => p.sequence_number,
                _ => 0,
            };
            
            let b_seq = match (b.1, b.2) {
                (Some(s), _) => s.sequence_number,
                (_, Some(p)) => p.sequence_number,
                _ => 0,
            };
            
            a_seq.cmp(&b_seq)
        });
        
        // Apply scrolling
        let display_lines = self.terminal_height.saturating_sub(10); // Reserve lines for header/footer
        let scroll_max = all_entries.len().saturating_sub(display_lines).max(0);
        let scroll_pos = self.scroll_position.min(scroll_max);
        let entries_to_display = all_entries.iter().skip(scroll_pos).take(display_lines);
        
        // Display entries
        for (id, maybe_sentence, maybe_pending) in entries_to_display {
            match (maybe_sentence, maybe_pending) {
                (Some(sentence), _) => {
                    // Completed translation
                    self.display_completed_row(sentence, col_width);
                },
                (_, Some(pending)) => {
                    // Pending translation
                    self.display_pending_row(*id, pending, col_width);
                },
                _ => continue, // Shouldn't happen
            }
        }
        
        // Show scroll indicators if needed
        if scroll_pos > 0 || scroll_pos < scroll_max {
            println!();
            if self.colors_enabled {
                println!("\x1B[1;36mScroll: {}/{}\x1B[0m", scroll_pos + 1, scroll_max + 1);
            } else {
                println!("Scroll: {}/{}", scroll_pos + 1, scroll_max + 1);
            }
        }
    }
    
    /// Display a completed translation row
    fn display_completed_row(&self, sentence: &TranslatedSentence, col_width: usize) {
        // Format timestamp
        let time = match Local.timestamp_millis_opt(sentence.timestamp as i64) {
            chrono::LocalResult::Single(dt) => dt,
            _ => Local::now()
        }.format("%H:%M:%S").to_string();
        
        // Format texts with word wrapping
        let original_wrapped = textwrap::wrap(&sentence.original, col_width.saturating_sub(12));
        let translation_wrapped = textwrap::wrap(&sentence.translation, col_width.saturating_sub(10));
        
        // Determine row count based on the longer text
        let row_count = original_wrapped.len().max(translation_wrapped.len());
        
        for i in 0..row_count {
            let original_line = if i < original_wrapped.len() {
                original_wrapped[i].to_string()
            } else {
                String::new()
            };
            
            let translation_line = if i < translation_wrapped.len() {
                translation_wrapped[i].to_string()
            } else {
                String::new()
            };
            
            // Add timestamp and sequence indicator only on first line
            if i == 0 {
                if self.colors_enabled {
                    println!("\x1B[1;32m[{}] \x1B[1;37m{:<width$} \x1B[1;32m|\x1B[0m \x1B[1;37m{:<width$}\x1B[0m",
                             time, original_line, translation_line, width=col_width);
                } else {
                    println!("[{}] {:<width$} | {:<width$}",
                             time, original_line, translation_line, width=col_width);
                }
            } else {
                // Continuation lines
                if self.colors_enabled {
                    println!("       \x1B[1;37m{:<width$} \x1B[1;32m|\x1B[0m \x1B[1;37m{:<width$}\x1B[0m",
                             original_line, translation_line, width=col_width);
                } else {
                    println!("       {:<width$} | {:<width$}",
                             original_line, translation_line, width=col_width);
                }
            }
        }
        
        // Add separator between entries
        println!();
    }
    
    /// Display a pending translation row
    fn display_pending_row(&self, id: u64, pending: &PendingTranslation, col_width: usize) {
        // Format timestamp
        let time = match Local.timestamp_millis_opt(pending.timestamp as i64) {
            chrono::LocalResult::Single(dt) => dt,
            _ => Local::now()
        }.format("%H:%M:%S").to_string();
        
        // Format original text with word wrapping
        let original_wrapped = textwrap::wrap(&pending.content, col_width.saturating_sub(12));
        
        // Get status indicator
        let status = self.status_indicators.get(&id).cloned().unwrap_or(TranslationStatus::Pending);
        let status_str = match status {
            TranslationStatus::Pending => "[Pending...]",
            TranslationStatus::InProgress => "[Translating...]",
            TranslationStatus::Failed => "[Failed]",
            _ => "[Waiting...]",
        };
        
        // Calculate elapsed time for visual feedback
        let elapsed_ms = SentenceTracker::current_time_ms() - pending.request_sent;
        let dots = ".".repeat(((elapsed_ms / 300) % 4) as usize);
        let wait_indicator = format!("{}{}", status_str, dots);
        
        for (i, line) in original_wrapped.iter().enumerate() {
            if i == 0 {
                // First line with timestamp and status
                if self.colors_enabled {
                    let status_color = match status {
                        TranslationStatus::Pending => "\x1B[1;33m",      // Yellow
                        TranslationStatus::InProgress => "\x1B[1;36m",   // Cyan
                        TranslationStatus::Failed => "\x1B[1;31m",       // Red
                        _ => "\x1B[1;37m",                              // White
                    };
                    println!("\x1B[1;33m[{}] \x1B[1;37m{:<width$} \x1B[1;32m|\x1B[0m {}{:<width$}\x1B[0m",
                             time, line, status_color, wait_indicator, width=col_width);
                } else {
                    println!("[{}] {:<width$} | {:<width$}",
                             time, line, wait_indicator, width=col_width);
                }
            } else {
                // Continuation lines
                if self.colors_enabled {
                    println!("       \x1B[1;37m{:<width$} \x1B[1;32m|\x1B[0m",
                             line, width=col_width);
                } else {
                    println!("       {:<width$} |", line, width=col_width);
                }
            }
        }
        
        // Add separator between entries
        println!();
    }
    
    /// Display translations in interleaved format
    fn display_interleaved(&self) {
        // Similar implementation to side_by_side but with different formatting
        // ...
    }
    
    /// Display original text only
    fn display_original_only(&self) {
        // Implementation for original-only view
        // ...
    }
    
    /// Display translations only
    fn display_translation_only(&self) {
        // Implementation for translation-only view
        // ...
    }
    
    /// Process keyboard input to handle UI controls
    fn process_input(&mut self) -> Result<bool, AppError> {
        // Non-blocking input check
        if let Ok(true) = poll_stdin(Duration::from_millis(10)) {
            if let Some(Ok(key)) = read_key() {
                match key {
                    Key::Char('1') => {
                        self.display_mode = DisplayMode::SideBySide;
                        self.update_display();
                    },
                    Key::Char('2') => {
                        self.display_mode = DisplayMode::Interleaved;
                        self.update_display();
                    },
                    Key::Char('3') => {
                        self.display_mode = DisplayMode::OriginalOnly;
                        self.update_display();
                    },
                    Key::Char('4') => {
                        self.display_mode = DisplayMode::TranslationOnly;
                        self.update_display();
                    },
                    Key::Up => {
                        if self.scroll_position > 0 {
                            self.scroll_position -= 1;
                            self.update_display();
                        }
                    },
                    Key::Down => {
                        self.scroll_position += 1;
                        self.update_display();
                    },
                    Key::Char('q') => {
                        return Ok(true); // Signal quit
                    },
                    _ => {}
                }
            }
        }
        
        Ok(false) // Continue running
    }
    
    /// Send acknowledgment for a received message
    fn send_acknowledgment(&self, pipe: &EnhancedNamedPipe, id: u64, status: AckStatus) -> Result<(), AppError> {
        let ack_message = IpcMessage::Acknowledgment {
            id,
            status,
            timestamp: SentenceTracker::current_time_ms(),
        };
        
        // Queue with highest priority
        pipe.queue_message(ack_message, 3, 5);
        Ok(())
    }
}

// Terminal input handling utilities 
fn poll_stdin(timeout: Duration) -> io::Result<bool> {
    // Platform-specific polling for terminal input
    // ...
    Ok(false) // Stub implementation
}

enum Key {
    Char(char),
    Up,
    Down,
    // Other keys...
}

fn read_key() -> Option<io::Result<Key>> {
    // Platform-specific key reading
    // ...
    None // Stub implementation
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
        
        // Add each sentence with improved formatting and timestamps
        for (_, sentence) in &self.sentences {
            // Format timestamp for display//
            let time = match Local.timestamp_millis_opt(sentence.timestamp as i64) {
                chrono::LocalResult::Single(dt) => dt,
                _ => Local::now()
            }
            .format("%H:%M:%S").to_string();
            
            // Add original text with formatting
            self.displayed_text.push_str(&format!("[{}] [Original] {}\n", time, sentence.original));
            
            // Add translation with formatting
            self.displayed_text.push_str(&format!("[{}] [Translation] {}\n\n", time, sentence.translation));
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
        const MAX_SENTENCES: usize = 50; // Increased from 10 to 25
        
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
                // Format timestamp for display
                let time = match Local.timestamp_millis_opt(sentence.timestamp as i64) {
                    chrono::LocalResult::Single(dt) => dt,
                    _ => Local::now()
                }
                .format("%H:%M:%S").to_string();
                
                self.displayed_text.push_str(&format!("[{}] [Original] {}\n", time, sentence.original));
                self.displayed_text.push_str(&format!("[{}] [Translation] {}\n\n", time, sentence.translation));
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

/// Architectural changes to use a single process with multi-threading
/// instead of separate processes for translation window
struct Engine {
    config: Config,
    displayed_text: String,
    caption_handle: CaptionHandle,
    translation_processor: Option<OptimizedBatchTranslationProcessor>,
    consecutive_empty_captures: usize,
    adaptive_interval: f64,
    output_file: Option<fs::File>,
    // New unified approach - no separate process
    translation_window: Option<EnhancedTranslationWindow>,
    sentence_tracker: SentenceTracker,
    // Synchronization and threading
    display_thread_handle: Option<JoinHandle<()>>,
    display_control_sender: Option<Sender<DisplayCommand>>,
    // Message handling tracking
    sequence_counter: AtomicU64,
}

/// Commands for the display thread
enum DisplayCommand {
    UpdateContent(TranslationUpdate),
    ChangeMode(DisplayMode),
    Scroll(ScrollDirection),
    Shutdown,
}

/// Translation update data
struct TranslationUpdate {
    id: u64,
    original: String,
    translation: Option<String>,
    timestamp: u64,
    is_complete: bool,
    sequence_number: u64,
    status: TranslationStatus,
    context: String,
}

/// Scroll direction
enum ScrollDirection {
    Up,
    Down,
    PageUp,
    PageDown,
}


impl Engine {
    /// Creates and initializes a new engine instance with unified architecture
    async fn new(config: Config) -> Result<Self, AppError> {
        debug!("Initializing engine with unified architecture");
        
        // Create caption handle
        let caption_handle = CaptionHandle::new()?;
        
        // Smaller batch sizes for better responsiveness
        let batch_size = config.translation_batch_size.min(3); // Cap at 3 for responsiveness
        let batch_delay_ms = config.translation_batch_delay_ms.min(200); // Cap at 200ms
        let min_batch_delay_ms = 50; // New parameter for minimum delay
        
        // Create optimized translation processor if enabled
        let translation_processor = if config.enable_translation {
            if let Some(api_key) = &config.translation_api_key {
                // Ensure target language is set
                let target_lang = config.target_language.clone().unwrap_or_else(|| {
                    match config.translation_api_type {
                        TranslationApiType::DeepL => "ZH".to_string(),
                        TranslationApiType::OpenAI => "Chinese".to_string(),
                        _ => "zh-CN".to_string(),
                    }
                });
                
                info!("Initializing optimized translation service with {:?} API", config.translation_api_type);
                Some(OptimizedBatchTranslationProcessor::new(
                    create_translation_service(
                        api_key.clone(),
                        target_lang,
                        config.translation_api_type,
                        config.openai_api_url.clone(),
                        config.openai_model.clone(),
                        config.openai_system_prompt.clone()
                    ),
                    config.translation_cache_size,
                    batch_size,
                    batch_delay_ms,
                    min_batch_delay_ms
                ))
            } else if config.translation_api_type == TranslationApiType::Demo {
                // Demo mode doesn't need an API key
                let target_lang = config.target_language.clone().unwrap_or_else(|| "zh-CN".to_string());
                Some(OptimizedBatchTranslationProcessor::new(
                    Arc::new(DemoTranslation {
                        target_language: target_lang,
                    }),
                    config.translation_cache_size,
                    batch_size,
                    batch_delay_ms,
                    min_batch_delay_ms
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
        
        // Create translation window and display thread if translation is enabled
        let (translation_window, display_thread_handle, display_control_sender) = 
            if config.enable_translation {
                // Initialize the translation window
                let window = EnhancedTranslationWindow::new();
                
                // Create a channel for display thread communication
                let (sender, receiver) = mpsc::channel(100);
                
                // Spawn the display thread
                let window_clone = window.clone();
                let handle = tokio::spawn(async move {
                    run_display_thread(window_clone, receiver).await;
                });
                
                (Some(window), Some(handle), Some(sender))
            } else {
                (None, None, None)
            };
        
        Ok(Self {
            displayed_text: String::new(),
            caption_handle,
            translation_processor,
            consecutive_empty_captures: 0,
            adaptive_interval: config.min_interval,
            output_file,
            config: config.clone(),
            translation_window,
            sentence_tracker: SentenceTracker::new(
                batch_size,
                batch_delay_ms
            ),
            display_thread_handle,
            display_control_sender,
            sequence_counter: AtomicU64::new(0),
        })
    }
    
    /// Get next sequence number for ordered processing
    fn next_sequence(&self) -> u64 {
        self.sequence_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
    
    /// Unified main loop for the engine with improved synchronization
    async fn run(&mut self) -> Result<(), AppError> {
        info!("Starting unified engine main loop");
        
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
        
        // Initialize more frequent status update timer
        let mut status_timer = tokio::time::interval(Duration::from_millis(100));
        
        // Print startup info
        println!("Live captions monitoring started with unified architecture:");
        println!("  - Using optimized sentence processing and enhanced synchronization");
        if self.config.ultra_responsive {
            println!("  - Using ultra-responsive mode");
            println!("    - Active polling: {}ms", self.config.active_interval_ms);
            println!("    - Idle polling: {}ms", self.config.idle_interval_ms);
        } else {
            println!("  - Using standard polling with {} second interval", self.config.capture_interval);
        }
        
        if self.config.enable_translation {
            println!("  - Translation enabled with optimized batching and improved synchronization");
            println!("    - Batch size: {} texts", self.config.translation_batch_size.min(3));
            println!("    - Max batch delay: {}ms", self.config.translation_batch_delay_ms.min(200));
            println!("    - Min batch delay: 50ms");
        }
        
        println!("Press Ctrl+C to exit");
        println!("-----------------------------------");
        
        // Main event loop
        loop {
            tokio::select! {
                _ = check_timer.tick() => {
                    // Periodic availability check
                    self.check_availability().await?;
                },
                _ = caption_timer.tick() => {
                    // Process captions with improved synchronization
                    self.process_captions().await?;
                    
                    // Adjust polling interval based on activity
                    self.adjust_polling_interval(&mut caption_timer, &mut current_interval_ms).await;
                },
                _ = status_timer.tick() => {
                    // Update status more frequently to improve responsiveness
                    if let Some(sender) = &self.display_control_sender {
                        let pending_count = self.sentence_tracker.pending_sentences.values()
                            .filter(|s| !s.acknowledgment_received)
                            .count();
                            
                        if pending_count > 0 {
                            // Only send updates when there are pending translations
                            let _ = sender.send(DisplayCommand::UpdateContent(
                                TranslationUpdate {
                                    id: 0, // Special ID for status updates
                                    original: String::new(),
                                    translation: None,
                                    timestamp: SentenceTracker::current_time_ms(),
                                    is_complete: true,
                                    sequence_number: self.next_sequence(),
                                    status: TranslationStatus::Pending,
                                    context: String::new(),
                                }
                            )).await;
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
    
    /// Check if caption source is available
    async fn check_availability(&self) -> Result<(), AppError> {
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
            let stats = processor.get_detailed_stats().await;
            info!("Translation stats: {} requests, {} batches, {} cache hits, {:.1}% hit rate, {:.1}ms avg latency", 
                  stats.requests, stats.batches, stats.cache_hits, 
                  (stats.cache_hits as f64 / stats.requests.max(1) as f64) * 100.0,
                  stats.average_latency);
        }
        
        Ok(())
    }
    
    /// Process captions with improved synchronization and error handling
    async fn process_captions(&mut self) -> Result<(), AppError> {
        match self.caption_handle.get_captions().await {
            Ok(Some(text)) => {
                debug!("Captured new text: {}", text);
                
                // Add to displayed text (original text only)
                self.displayed_text.push_str(&text);
                self.limit_text_length();
                Self::display_text(&self.displayed_text)?;
                
                // Process for translation with improved sentence handling
                self.process_for_translation(&text).await?;
                
                // Write to output file (if configured)
                if let Some(file) = &mut self.output_file {
                    if let Err(e) = writeln!(file, "{}", text) {
                        warn!("Failed to write to output file: {}", e);
                    }
                }
                
                // Reset consecutive empty captures counter
                self.consecutive_empty_captures = 0;
            },
            Ok(None) => {
                // No new captions, check if we should send buffered content
                self.check_pending_sentences().await?;
                self.consecutive_empty_captures += 1;
            },
            Err(e) => {
                warn!("Failed to capture captions: {}", e);
            }
        }
        
        Ok(())
    }
    
    /// Adjust polling interval based on caption activity
    async fn adjust_polling_interval(&self, caption_timer: &mut Interval, current_interval_ms: &mut u64) {
        if self.config.ultra_responsive {
            match self.caption_handle.is_active(self.config.active_timeout_sec).await {
                Ok(active) => {
                    let target_interval_ms = if active {
                        self.config.active_interval_ms
                    } else {
                        self.config.idle_interval_ms
                    };
                    
                    // Only update if interval significantly different
                    if *current_interval_ms != target_interval_ms {
                        *current_interval_ms = target_interval_ms;
                        *caption_timer = tokio::time::interval(Duration::from_millis(*current_interval_ms));
                        debug!("Adjusted polling interval to {}ms", current_interval_ms);
                    }
                },
                Err(e) => warn!("Failed to check caption activity: {}", e)
            }
        }
    }
    
    /// Process text for translation with the enhanced sentence tracker
    async fn process_for_translation(&mut self, text: &str) -> Result<(), AppError> {
        if self.translation_processor.is_none() || self.display_control_sender.is_none() {
            return Ok(());
        }
        
        // Extract complete sentences with context for better translation
        let complete_sentences = self.sentence_tracker.add_text(text);
        
        // Process each sentence
        for (id, sentence, context) in complete_sentences {
            if let Some(sender) = &self.display_control_sender {
                // Get sequence number for ordered display
                let seq = self.next_sequence();
                
                // Send to display thread with "Pending" status
                let _ = sender.send(DisplayCommand::UpdateContent(
                    TranslationUpdate {
                        id,
                        original: sentence.clone(),
                        translation: None,
                        timestamp: SentenceTracker::current_time_ms(),
                        is_complete: true,
                        sequence_number: seq,
                        status: TranslationStatus::Pending,
                        context: context.clone(),
                    }
                )).await;
                
                // Start the translation process asynchronously
                self.translate_sentence(id, sentence, context, seq).await?;
            }
        }
        
        // Check if we should also send incomplete sentences
        if self.sentence_tracker.should_send_incomplete() {
            if let Some((id, text, context)) = self.sentence_tracker.get_incomplete_sentence() {
                if let Some(sender) = &self.display_control_sender {
                    // Get sequence number
                    let seq = self.next_sequence();
                    
                    // Send to display thread
                    let _ = sender.send(DisplayCommand::UpdateContent(
                        TranslationUpdate {
                            id,
                            original: text.clone(),
                            translation: None,
                            timestamp: SentenceTracker::current_time_ms(),
                            is_complete: false,
                            sequence_number: seq,
                            status: TranslationStatus::Pending,
                            context: context.clone(),
                        }
                    )).await;
                    
                    // Start translation
                    self.translate_sentence(id, text, context, seq).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Translate a single sentence and update the display
    async fn translate_sentence(&self, id: u64, text: String, context: String, seq: u64) -> Result<(), AppError> {
        if let (Some(processor), Some(sender)) = (&self.translation_processor, &self.display_control_sender) {
            // Update status to "InProgress"
            let _ = sender.send(DisplayCommand::UpdateContent(
                TranslationUpdate {
                    id,
                    original: text.clone(),
                    translation: None,
                    timestamp: SentenceTracker::current_time_ms(),
                    is_complete: true,
                    sequence_number: seq,
                    status: TranslationStatus::InProgress,
                    context: context.clone(),
                }
            )).await;
            
            // Perform the translation (fire and forget)
            let processor_clone = processor.clone();
            let sender_clone = sender.clone();
            let text_clone = text.clone();
            let context_clone = context.clone();
            
            tokio::spawn(async move {
                let translation_result = processor_clone.translate(&text_clone).await;
                
                match translation_result {
                    Ok(translation) => {
                        // Send completed translation to display thread
                        let _ = sender_clone.send(DisplayCommand::UpdateContent(
                            TranslationUpdate {
                                id,
                                original: text_clone,
                                translation: Some(translation),
                                timestamp: SentenceTracker::current_time_ms(),
                                is_complete: true,
                                sequence_number: seq,
                                status: TranslationStatus::Completed,
                                context: context_clone,
                            }
                        )).await;
                    },
                    Err(e) => {
                        // Send failure status
                        warn!("Translation failed for sentence {}: {}", id, e);
                        let _ = sender_clone.send(DisplayCommand::UpdateContent(
                            TranslationUpdate {
                                id,
                                original: text_clone,
                                translation: None,
                                timestamp: SentenceTracker::current_time_ms(),
                                is_complete: true,
                                sequence_number: seq,
                                status: TranslationStatus::Failed,
                                context: context_clone,
                            }
                        )).await;
                    }
                }
            });
        }
        
        Ok(())
    }
    
    /// Check if there are pending sentences that should be sent
    async fn check_pending_sentences(&mut self) -> Result<(), AppError> {
        if self.translation_processor.is_none() || self.display_control_sender.is_none() {
            return Ok(());
        }
        
        // Check for inactivity triggers
        if self.consecutive_empty_captures > 2 {
            // Check if we have any unsent sentences in the tracker
            let batch = self.sentence_tracker.get_pending_batch();
            if !batch.is_empty() {
                debug!("Processing {} pending sentences due to inactivity", batch.len());
                
                // Process each pending sentence
                for (id, sentence, context) in batch {
                    if let Some(sender) = &self.display_control_sender {
                        // Get sequence number
                        let seq = self.next_sequence();
                        
                        // Send to display thread
                        let _ = sender.send(DisplayCommand::UpdateContent(
                            TranslationUpdate {
                                id,
                                original: sentence.clone(),
                                translation: None,
                                timestamp: SentenceTracker::current_time_ms(),
                                is_complete: true,
                                sequence_number: seq,
                                status: TranslationStatus::Pending,
                                context: context.clone(),
                            }
                        )).await;
                        
                        // Start translation
                        self.translate_sentence(id, sentence, context, seq).await?;
                    }
                }
            }
            
            // Also check for incomplete sentences that should be sent
            if self.sentence_tracker.should_send_incomplete() {
                if let Some((id, text, context)) = self.sentence_tracker.get_incomplete_sentence() {
                    debug!("Processing incomplete sentence due to inactivity: {}", text);
                    
                    if let Some(sender) = &self.display_control_sender {
                        // Get sequence number
                        let seq = self.next_sequence();
                        
                        // Send to display thread
                        let _ = sender.send(DisplayCommand::UpdateContent(
                            TranslationUpdate {
                                id,
                                original: text.clone(),
                                translation: None,
                                timestamp: SentenceTracker::current_time_ms(),
                                is_complete: false,
                                sequence_number: seq,
                                status: TranslationStatus::Pending,
                                context: context.clone(),
                            }
                        )).await;
                        
                        // Start translation
                        self.translate_sentence(id, text, context, seq).await?;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Graceful shutdown with improved cleanup
    async fn graceful_shutdown(&mut self) -> Result<(), AppError> {
        info!("Performing graceful shutdown");
        
        // Process any remaining sentences
        self.check_pending_sentences().await?;
        
        // Try to get final captions
        match self.caption_handle.get_captions().await {
            Ok(Some(text)) => {
                // Process the final text
                self.process_for_translation(&text).await?;
                
                // Update displayed text
                self.displayed_text.push_str(&text);
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
        
        // Shutdown the display thread if it exists
        if let Some(sender) = &self.display_control_sender {
            info!("Shutting down display thread");
            let _ = sender.send(DisplayCommand::Shutdown).await;
            
            // Wait for the thread to finish (with timeout)
            if let Some(handle) = self.display_thread_handle.take() {
                match tokio::time::timeout(Duration::from_secs(3), handle).await {
                    Ok(_) => info!("Display thread shut down successfully"),
                    Err(_) => warn!("Display thread shutdown timed out"),
                }
            }
        }
        
        // Shut down translation processor if it exists
        if let Some(processor) = &self.translation_processor {
            info!("Shutting down translation processor");
            processor.shutdown();
        }
        
        // Shut down the caption handle
        if let Err(e) = self.caption_handle.shutdown().await {
            warn!("Error shutting down caption actor: {}", e);
        }
        
        info!("Shutdown complete");
        Ok(())
    }
    
    // Other utility methods remain mostly the same...
}

/// Run the display thread that manages the translation window
async fn run_display_thread(mut window: EnhancedTranslationWindow, mut receiver: mpsc::Receiver<DisplayCommand>) {
    info!("Translation display thread started");
    
    let mut last_update = Instant::now();
    let mut check_input_interval = tokio::time::interval(Duration::from_millis(50));
    
    // Clear screen initially
    print!("\x1B[2J\x1B[1;1H");
    window.update_display();
    
    loop {
        tokio::select! {
            maybe_command = receiver.recv() => {
                match maybe_command {
                    Some(DisplayCommand::UpdateContent(update)) => {
                        // Update translation content
                        match update.status {
                            TranslationStatus::Pending => {
                                // Add as pending
                                window.pending_translations.insert(update.id, PendingTranslation {
                                    content: update.original.clone(),
                                    timestamp: update.timestamp,
                                    is_complete: update.is_complete,
                                    sequence_number: update.sequence_number,
                                    context: update.context,
                                    request_sent: SentenceTracker::current_time_ms(),
                                });
                                window.status_indicators.insert(update.id, TranslationStatus::Pending);
                            },
                            TranslationStatus::InProgress => {
                                // Update status to in-progress
                                window.status_indicators.insert(update.id, TranslationStatus::InProgress);
                            },
                            TranslationStatus::Completed => {
                                // Add completed translation
                                if let Some(translation) = update.translation {
                                    window.sentences.insert(update.id, TranslatedSentence {
                                        id: update.id,
                                        original: update.original,
                                        translation,
                                        timestamp: update.timestamp,
                                        is_complete: update.is_complete,
                                        sequence_number: update.sequence_number,
                                        processing_time_ms: 0, // Could calculate if needed
                                        context: update.context,
                                    });
                                    
                                    // Remove from pending and update status
                                    window.pending_translations.remove(&update.id);
                                    window.status_indicators.insert(update.id, TranslationStatus::Completed);
                                }
                            },
                            TranslationStatus::Failed => {
                                // Mark as failed
                                window.status_indicators.insert(update.id, TranslationStatus::Failed);
                            },
                        }
                        
                        // Throttle display updates to avoid flickering
                        if last_update.elapsed() >= Duration::from_millis(50) {
                            window.update_display();
                            last_update = Instant::now();
                        }
                    },
                    Some(DisplayCommand::ChangeMode(mode)) => {
                        window.display_mode = mode;
                        window.update_display();
                    },
                    Some(DisplayCommand::Scroll(direction)) => {
                        match direction {
                            ScrollDirection::Up => window.scroll_position = window.scroll_position.saturating_sub(1),
                            ScrollDirection::Down => window.scroll_position += 1,
                            ScrollDirection::PageUp => window.scroll_position = window.scroll_position.saturating_sub(10),
                            ScrollDirection::PageDown => window.scroll_position += 10,
                        }
                        window.update_display();
                    },
                    Some(DisplayCommand::Shutdown) => {
                        info!("Display thread received shutdown command");
                        break;
                    },
                    None => {
                        info!("Display control channel closed, shutting down");
                        break;
                    }
                }
            },
            _ = check_input_interval.tick() => {
                // Check for keyboard input
                match window.process_input() {
                    Ok(true) => {
                        // User pressed quit key
                        info!("User requested quit from translation window");
                        break;
                    },
                    Ok(false) => {
                        // Normal operation, continue
                    },
                    Err(e) => {
                        warn!("Error processing input: {}", e);
                    }
                }
                
                // Update window periodically to show animation effects
                if last_update.elapsed() >= Duration::from_millis(100) {
                    window.update_display();
                    last_update = Instant::now();
                }
            }
        }
    }
    
    info!("Translation display thread terminated");
}

/// Helper to create a translation service based on type
fn create_translation_service(
    api_key: String,
    target_language: String,
    api_type: TranslationApiType,
    openai_api_url: Option<String>,
    openai_model: Option<String>,
    openai_system_prompt: Option<String>,
) -> Arc<dyn TranslationService> {
    match api_type {
        TranslationApiType::DeepL => {
            Arc::new(DeepLTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(300)), // Reduced from 500ms
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::Generic => {
            Arc::new(GenericTranslation {
                api_key,
                target_language,
                client: reqwest::Client::new(),
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(300)),
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
                rate_limiter: tokio::sync::Mutex::new(RateLimiter::new(600)),
            }) as Arc<dyn TranslationService>
        },
        TranslationApiType::Demo => {
            Arc::new(DemoTranslation {
                target_language,
            }) as Arc<dyn TranslationService>
        },
    }
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
                let (requests, batches, cache_hits, _batch_sizes, hit_rate) = translation_processor.get_stats().await;
                if requests > 0 {
                    info!("Translation stats: {} requests, {} batches, {} cache hits, {:.1}% hit rate",
                          requests, batches, cache_hits, hit_rate * 100.0);
                }
            },
            _ = tokio::time::sleep(Duration::from_millis(100)) => {
                // Check for new messages with improved error handling
                match pipe.read_message() {
                    Ok(IpcMessage::Text { id, content, timestamp, is_complete }) => {
                        debug!("Received text for translation [{}]: {}", id, content);
                        if !content.is_empty() {
                            // Process text with retry on translation failure
                            let mut retry_count = 0;
                            const MAX_RETRIES: usize = 3;
                            
                            while retry_count < MAX_RETRIES {
                                match translation_window.process_text_message(
                                    id, content.clone(), timestamp, is_complete, &translation_processor
                                ).await {
                                    Ok(_) => break,
                                    Err(e) => {
                                        retry_count += 1;
                                        if retry_count >= MAX_RETRIES {
                                            warn!("Translation failed after {} retries: {}", MAX_RETRIES, e);
                                        } else {
                                            debug!("Translation retry {}/{}: {}", retry_count, MAX_RETRIES, e);
                                            tokio::time::sleep(Duration::from_millis(100 * retry_count as u64)).await;
                                        }
                                    }
                                }
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
                        
                        // Short pause before trying again to avoid CPU spinning
                        tokio::time::sleep(Duration::from_millis(100)).await;
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
    // Initialize default logger
    env_logger::init();
    
    // Parse command line arguments
    let args = EnhancedArgs::parse();
    
    // Check if we should enable debug logging early
    if args.debug.unwrap_or(false) {
        env_logger::Builder::new()
            .filter_level(log::LevelFilter::Debug)
            .init();
    }
    
    // Create engine with enhanced features
    let mut engine = create_enhanced_engine().await?;
    
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
            active_interval_ms: 30,
            idle_interval_ms: 500,
            active_timeout_sec: 5.0,
            translation_cache_size: 2000,
            translation_batch_size: 5,
            translation_batch_delay_ms: 200,
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