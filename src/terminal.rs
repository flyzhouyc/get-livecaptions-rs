/// Terminal UI utilities for improved input handling and display
mod terminal {
    use std::io::{self, Write};
    use std::time::Duration;
    use termion::event::Key;
    use termion::input::TermRead;
    use termion::raw::IntoRawMode;
    use termion::screen::AlternateScreen;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;

    /// Terminal UI manager for advanced input and display
    pub struct TerminalUI {
        width: u16,
        height: u16,
        raw_mode: bool,
        alternate_screen: bool,
        running: Arc<AtomicBool>,
        input_thread: Option<thread::JoinHandle<()>>,
        key_receiver: Option<crossbeam_channel::Receiver<KeyEvent>>,
    }

    /// Key event with metadata
    pub struct KeyEvent {
        pub key: Key,
        pub timestamp: std::time::Instant,
    }

    impl TerminalUI {
        /// Create a new terminal UI manager
        pub fn new() -> io::Result<Self> {
            let (width, height) = termion::terminal_size()?;
            
            Ok(Self {
                width,
                height,
                raw_mode: false,
                alternate_screen: false,
                running: Arc::new(AtomicBool::new(false)),
                input_thread: None,
                key_receiver: None,
            })
        }
        
        /// Initialize terminal for interactive use
        pub fn init(&mut self, use_alternate_screen: bool) -> io::Result<()> {
            // Enter raw mode
            let _ = io::stdout().into_raw_mode()?;
            self.raw_mode = true;
            
            // Optionally switch to alternate screen
            if use_alternate_screen {
                let _ = write!(io::stdout(), "{}", termion::screen::ToAlternateScreen)?;
                self.alternate_screen = true;
            }
            
            // Hide cursor
            let _ = write!(io::stdout(), "{}", termion::cursor::Hide)?;
            
            // Start input thread
            self.start_input_thread()?;
            
            io::stdout().flush()?;
            Ok(())
        }
        
        /// Start background thread for input handling
        fn start_input_thread(&mut self) -> io::Result<()> {
            let (sender, receiver) = crossbeam_channel::unbounded();
            let running = self.running.clone();
            running.store(true, Ordering::SeqCst);
            
            // Create thread for non-blocking input handling
            let handle = thread::spawn(move || {
                let stdin = io::stdin();
                let mut keys = stdin.keys();
                
                while running.load(Ordering::SeqCst) {
                    // Try to read a key with timeout
                    if let Some(Ok(key)) = keys.next() {
                        let _ = sender.send(KeyEvent {
                            key,
                            timestamp: std::time::Instant::now(),
                        });
                        
                        // Exit thread if Ctrl+C is pressed
                        if let Key::Ctrl('c') = key {
                            break;
                        }
                    }
                    
                    // Small sleep to avoid CPU spin
                    thread::sleep(Duration::from_millis(10));
                }
            });
            
            self.input_thread = Some(handle);
            self.key_receiver = Some(receiver);
            
            Ok(())
        }
        
        /// Poll for key press with timeout
        pub fn poll_key(&self, timeout: Duration) -> Option<KeyEvent> {
            if let Some(receiver) = &self.key_receiver {
                match receiver.recv_timeout(timeout) {
                    Ok(event) => Some(event),
                    Err(_) => None,
                }
            } else {
                None
            }
        }
        
        /// Check if terminal size has changed and update dimensions
        pub fn check_resize(&mut self) -> io::Result<bool> {
            let (new_width, new_height) = termion::terminal_size()?;
            
            if new_width != self.width || new_height != self.height {
                self.width = new_width;
                self.height = new_height;
                return Ok(true);
            }
            
            Ok(false)
        }
        
        /// Get current terminal dimensions
        pub fn dimensions(&self) -> (u16, u16) {
            (self.width, self.height)
        }
        
        /// Clean up terminal state on exit
        pub fn cleanup(&mut self) -> io::Result<()> {
            // Stop input thread
            if self.running.load(Ordering::SeqCst) {
                self.running.store(false, Ordering::SeqCst);
                
                if let Some(handle) = self.input_thread.take() {
                    let _ = handle.join();
                }
            }
            
            // Show cursor
            let _ = write!(io::stdout(), "{}", termion::cursor::Show)?;
            
            // Exit alternate screen if used
            if self.alternate_screen {
                let _ = write!(io::stdout(), "{}", termion::screen::ToMainScreen)?;
                self.alternate_screen = false;
            }
            
            // Exit raw mode
            if self.raw_mode {
                // Termion automatically exits raw mode when dropped
                self.raw_mode = false;
            }
            
            io::stdout().flush()?;
            Ok(())
        }
    }

    impl Drop for TerminalUI {
        fn drop(&mut self) {
            let _ = self.cleanup();
        }
    }
    
    /// Draw text with optional coloring
    pub fn draw_text(
        x: u16, 
        y: u16, 
        text: &str, 
        fg_color: Option<Color>, 
        bg_color: Option<Color>,
        bold: bool
    ) -> io::Result<()> {
        // Position cursor
        write!(io::stdout(), "{}", termion::cursor::Goto(x, y))?;
        
        // Apply styling
        if let Some(fg) = fg_color {
            match fg {
                Color::Black => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Black))?,
                Color::Red => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Red))?,
                Color::Green => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Green))?,
                Color::Yellow => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Yellow))?,
                Color::Blue => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Blue))?,
                Color::Magenta => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Magenta))?,
                Color::Cyan => write!(io::stdout(), "{}", termion::color::Fg(termion::color::Cyan))?,
                Color::White => write!(io::stdout(), "{}", termion::color::Fg(termion::color::White))?,
                Color::LightBlack => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightBlack))?,
                Color::LightRed => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightRed))?,
                Color::LightGreen => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightGreen))?,
                Color::LightYellow => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightYellow))?,
                Color::LightBlue => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightBlue))?,
                Color::LightMagenta => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightMagenta))?,
                Color::LightCyan => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightCyan))?,
                Color::LightWhite => write!(io::stdout(), "{}", termion::color::Fg(termion::color::LightWhite))?,
            }
        }
        
        if let Some(bg) = bg_color {
            match bg {
                Color::Black => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Black))?,
                Color::Red => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Red))?,
                Color::Green => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Green))?,
                Color::Yellow => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Yellow))?,
                Color::Blue => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Blue))?,
                Color::Magenta => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Magenta))?,
                Color::Cyan => write!(io::stdout(), "{}", termion::color::Bg(termion::color::Cyan))?,
                Color::White => write!(io::stdout(), "{}", termion::color::Bg(termion::color::White))?,
                Color::LightBlack => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightBlack))?,
                Color::LightRed => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightRed))?,
                Color::LightGreen => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightGreen))?,
                Color::LightYellow => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightYellow))?,
                Color::LightBlue => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightBlue))?,
                Color::LightMagenta => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightMagenta))?,
                Color::LightCyan => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightCyan))?,
                Color::LightWhite => write!(io::stdout(), "{}", termion::color::Bg(termion::color::LightWhite))?,
            }
        }
        
        if bold {
            write!(io::stdout(), "{}", termion::style::Bold)?;
        }
        
        // Draw text
        write!(io::stdout(), "{}", text)?;
        
        // Reset styling
        write!(io::stdout(), "{}", termion::style::Reset)?;
        
        Ok(())
    }
    
    /// Draw a box with optional title
    pub fn draw_box(
        x: u16, y: u16, 
        width: u16, height: u16, 
        title: Option<&str>,
        fg_color: Option<Color>, 
        bg_color: Option<Color>
    ) -> io::Result<()> {
        // Draw top border
        draw_text(x, y, &format!("╭{}╮", "─".repeat((width - 2) as usize)), fg_color, bg_color, false)?;
        
        // Add title if provided
        if let Some(title_text) = title {
            let title_x = x + 2;
            let title_with_space = format!(" {} ", title_text);
            draw_text(title_x, y, &title_with_space, fg_color, bg_color, true)?;
        }
        
        // Draw sides
        for i in 1..height-1 {
            draw_text(x, y + i, "│", fg_color, bg_color, false)?;
            draw_text(x + width - 1, y + i, "│", fg_color, bg_color, false)?;
        }
        
        // Draw bottom border
        draw_text(x, y + height - 1, &format!("╰{}╯", "─".repeat((width - 2) as usize)), fg_color, bg_color, false)?;
        
        Ok(())
    }
    
    /// Draw a progress bar
    pub fn draw_progress_bar(
        x: u16, y: u16, 
        width: u16, 
        progress: f64, // 0.0 to 1.0
        fg_color: Option<Color>, 
        bg_color: Option<Color>
    ) -> io::Result<()> {
        let fill_chars = (progress.clamp(0.0, 1.0) * (width as f64)) as u16;
        let empty_chars = width - fill_chars;
        
        // Draw filled portion
        if fill_chars > 0 {
            draw_text(x, y, &"█".repeat(fill_chars as usize), fg_color, bg_color, false)?;
        }
        
        // Draw empty portion
        if empty_chars > 0 {
            draw_text(x + fill_chars, y, &"░".repeat(empty_chars as usize), None, bg_color, false)?;
        }
        
        Ok(())
    }
    
    /// Clear a specific region of the screen
    pub fn clear_region(x: u16, y: u16, width: u16, height: u16) -> io::Result<()> {
        let empty_line = " ".repeat(width as usize);
        for i in 0..height {
            draw_text(x, y + i, &empty_line, None, None, false)?;
        }
        Ok(())
    }
    
    /// Color definitions
    #[derive(Copy, Clone, Debug)]
    pub enum Color {
        Black,
        Red,
        Green,
        Yellow,
        Blue,
        Magenta,
        Cyan,
        White,
        LightBlack,
        LightRed,
        LightGreen,
        LightYellow,
        LightBlue,
        LightMagenta,
        LightCyan,
        LightWhite,
    }
}

/// Enhanced UI for the EnhancedTranslationWindow with termion support
impl EnhancedTranslationWindow {
    /// Draw the window using termion for better UI
    fn draw_with_termion(&self, terminal: &terminal::TerminalUI) -> io::Result<()> {
        let (term_width, term_height) = terminal.dimensions();
        
        // Clear screen
        write!(io::stdout(), "{}", termion::clear::All)?;
        
        // Draw main border
        terminal::draw_box(
            1, 1, 
            term_width, term_height, 
            Some("Enhanced Translation Window"),
            Some(terminal::Color::LightCyan),
            None
        )?;
        
        // Draw status area
        self.draw_status_area(terminal, term_width)?;
        
        // Draw content based on display mode
        match self.display_mode {
            DisplayMode::SideBySide => self.draw_side_by_side(terminal, term_width, term_height)?,
            DisplayMode::Interleaved => self.draw_interleaved(terminal, term_width, term_height)?,
            DisplayMode::OriginalOnly => self.draw_original_only(terminal, term_width, term_height)?,
            DisplayMode::TranslationOnly => self.draw_translation_only(terminal, term_width, term_height)?,
        }
        
        // Draw help footer
        self.draw_help_footer(terminal, term_width, term_height)?;
        
        io::stdout().flush()?;
        Ok(())
    }
    
    /// Draw the status area
    fn draw_status_area(&self, terminal: &terminal::TerminalUI, term_width: u16) -> io::Result<()> {
        // Draw stats box
        terminal::draw_box(
            2, 2, 
            term_width - 2, 4, 
            Some("Status"),
            Some(terminal::Color::LightYellow),
            None
        )?;
        
        // Calculate statistics
        let completed = self.sentences.len();
        let pending = self.pending_translations.len();
        let total = completed + pending;
        
        // Draw statistics
        let status_text = format!(
            "Completed: {} | Pending: {} | Total: {} | Mode: {}", 
            completed, pending, total,
            match self.display_mode {
                DisplayMode::SideBySide => "Side by Side",
                DisplayMode::Interleaved => "Interleaved",
                DisplayMode::OriginalOnly => "Original Only",
                DisplayMode::TranslationOnly => "Translation Only",
            }
        );
        
        terminal::draw_text(
            4, 3, 
            &status_text,
            Some(terminal::Color::LightGreen),
            None,
            true
        )?;
        
        // Draw a progress bar for translation progress
        let progress = if total > 0 {
            completed as f64 / total as f64
        } else {
            1.0
        };
        
        let progress_text = format!("Progress: {:.1}%", progress * 100.0);
        terminal::draw_text(
            4, 4, 
            &progress_text,
            Some(terminal::Color::White),
            None,
            false
        )?;
        
        terminal::draw_progress_bar(
            15, 4, 
            term_width - 20, 
            progress,
            Some(terminal::Color::LightGreen),
            None
        )?;
        
        Ok(())
    }
    
    /// Draw side-by-side display
    fn draw_side_by_side(&self, terminal: &terminal::TerminalUI, term_width: u16, term_height: u16) -> io::Result<()> {
        // Draw content box
        terminal::draw_box(
            2, 6, 
            term_width - 2, term_height - 9, 
            Some("Original Text | Translation"),
            Some(terminal::Color::LightBlue),
            None
        )?;
        
        // Calculate column width
        let col_width = (term_width - 8) / 2;
        
        // Draw header
        terminal::draw_text(
            4, 7, 
            "Original Text",
            Some(terminal::Color::LightYellow),
            None,
            true
        )?;
        
        terminal::draw_text(
            4 + col_width + 2, 7, 
            "Translation",
            Some(terminal::Color::LightYellow),
            None,
            true
        )?;
        
        terminal::draw_text(
            4, 8, 
            &"─".repeat(col_width as usize),
            Some(terminal::Color::LightYellow),
            None,
            false
        )?;
        
        terminal::draw_text(
            4 + col_width + 2, 8, 
            &"─".repeat(col_width as usize),
            Some(terminal::Color::LightYellow),
            None,
            false
        )?;
        
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
        
        // Available lines for content
        let available_lines = term_height.saturating_sub(12) as usize;
        
        // Calculate maximum scroll position
        let max_scroll = all_entries.len().saturating_sub(available_lines);
        let clamped_scroll = self.scroll_position.min(max_scroll);
        
        // Display entries with scrolling
        let entries_to_display = all_entries.iter().skip(clamped_scroll).take(available_lines);
        
        for (i, (id, maybe_sentence, maybe_pending)) in entries_to_display.enumerate() {
            let y_pos = 9 + i as u16;
            
            match (maybe_sentence, maybe_pending) {
                (Some(sentence), _) => {
                    // Format timestamp
                    let time = match Local.timestamp_millis_opt(sentence.timestamp as i64) {
                        chrono::LocalResult::Single(dt) => dt,
                        _ => Local::now()
                    }.format("%H:%M:%S").to_string();
                    
                    // Show sequence number and timestamp
                    terminal::draw_text(
                        4, y_pos, 
                        &format!("[{}] ", time),
                        Some(terminal::Color::LightBlack),
                        None,
                        false
                    )?;
                    
                    // Draw original text (truncated if needed)
                    let original_display = if sentence.original.len() > col_width as usize - 10 {
                        format!("{}...", &sentence.original[0..col_width as usize - 13])
                    } else {
                        sentence.original.clone()
                    };
                    
                    terminal::draw_text(
                        4 + 10, y_pos, 
                        &original_display,
                        Some(terminal::Color::White),
                        None,
                        false
                    )?;
                    
                    // Draw translation (truncated if needed)
                    let translation_display = if sentence.translation.len() > col_width as usize {
                        format!("{}...", &sentence.translation[0..col_width as usize - 3])
                    } else {
                        sentence.translation.clone()
                    };
                    
                    terminal::draw_text(
                        4 + col_width + 2, y_pos, 
                        &translation_display,
                        Some(terminal::Color::LightGreen),
                        None,
                        false
                    )?;
                },
                (_, Some(pending)) => {
                    // Format timestamp
                    let time = match Local.timestamp_millis_opt(pending.timestamp as i64) {
                        chrono::LocalResult::Single(dt) => dt,
                        _ => Local::now()
                    }.format("%H:%M:%S").to_string();
                    
                    // Show sequence number and timestamp
                    terminal::draw_text(
                        4, y_pos, 
                        &format!("[{}] ", time),
                        Some(terminal::Color::LightBlack),
                        None,
                        false
                    )?;
                    
                    // Draw original text
                    let original_display = if pending.content.len() > col_width as usize - 10 {
                        format!("{}...", &pending.content[0..col_width as usize - 13])
                    } else {
                        pending.content.clone()
                    };
                    
                    terminal::draw_text(
                        4 + 10, y_pos, 
                        &original_display,
                        Some(terminal::Color::White),
                        None,
                        false
                    )?;
                    
                    // Get status indicator
                    let status = self.status_indicators.get(id).cloned().unwrap_or(TranslationStatus::Pending);
                    
                    // Calculate elapsed time for animation
                    let elapsed_ms = SentenceTracker::current_time_ms() - pending.request_sent;
                    let animation_char = match (elapsed_ms / 250) % 4 {
                        0 => "|",
                        1 => "/",
                        2 => "-",
                        _ => "\\",
                    };
                    
                    // Status text with animation
                    let status_text = match status {
                        TranslationStatus::Pending => format!("[Pending {}]", animation_char),
                        TranslationStatus::InProgress => format!("[Translating {}]", animation_char),
                        TranslationStatus::Failed => "[Failed]",
                        _ => "[Waiting]",
                    };
                    
                    // Color based on status
                    let status_color = match status {
                        TranslationStatus::Pending => terminal::Color::LightYellow,
                        TranslationStatus::InProgress => terminal::Color::LightCyan,
                        TranslationStatus::Failed => terminal::Color::LightRed,
                        _ => terminal::Color::White,
                    };
                    
                    terminal::draw_text(
                        4 + col_width + 2, y_pos, 
                        &status_text,
                        Some(status_color),
                        None,
                        false
                    )?;
                },
                _ => continue, // Shouldn't happen
            }
        }
        
        // Show scroll indicators
        if clamped_scroll > 0 || clamped_scroll < max_scroll {
            let indicator = format!("Scroll: {}/{}", clamped_scroll + 1, max_scroll + 1);
            let x_pos = term_width - indicator.len() as u16 - 3;
            
            terminal::draw_text(
                x_pos, term_height - 3, 
                &indicator,
                Some(terminal::Color::LightCyan),
                None,
                false
            )?;
        }
        
        Ok(())
    }
    
    /// Draw interleaved display
    fn draw_interleaved(&self, terminal: &terminal::TerminalUI, term_width: u16, term_height: u16) -> io::Result<()> {
        // Implementation for interleaved view
        // Similar to side_by_side but with different layout
        // ...
        
        Ok(())
    }
    
    /// Draw original text only
    fn draw_original_only(&self, terminal: &terminal::TerminalUI, term_width: u16, term_height: u16) -> io::Result<()> {
        // Implementation for original-only view
        // ...
        
        Ok(())
    }
    
    /// Draw translation only
    fn draw_translation_only(&self, terminal: &terminal::TerminalUI, term_width: u16, term_height: u16) -> io::Result<()> {
        // Implementation for translation-only view
        // ...
        
        Ok(())
    }
    
    /// Draw help footer
    fn draw_help_footer(&self, terminal: &terminal::TerminalUI, term_width: u16, term_height: u16) -> io::Result<()> {
        // Draw footer box
        terminal::draw_box(
            2, term_height - 3, 
            term_width - 2, 3, 
            Some("Controls"),
            Some(terminal::Color::LightMagenta),
            None
        )?;
        
        // Draw control hints
        let control_text = "[1] Side by Side [2] Interleaved [3] Original [4] Translation [↑/↓] Scroll [q] Quit";
        let x_pos = (term_width - control_text.len() as u16) / 2;
        
        terminal::draw_text(
            x_pos, term_height - 2, 
            control_text,
            Some(terminal::Color::White),
            None,
            false
        )?;
        
        Ok(())
    }
    
    /// Process keyboard input using termion events
    fn process_termion_input(&mut self, terminal: &terminal::TerminalUI) -> io::Result<bool> {
        if let Some(event) = terminal.poll_key(Duration::from_millis(10)) {
            match event.key {
                Key::Char('1') => {
                    self.display_mode = DisplayMode::SideBySide;
                    self.draw_with_termion(terminal)?;
                },
                Key::Char('2') => {
                    self.display_mode = DisplayMode::Interleaved;
                    self.draw_with_termion(terminal)?;
                },
                Key::Char('3') => {
                    self.display_mode = DisplayMode::OriginalOnly;
                    self.draw_with_termion(terminal)?;
                },
                Key::Char('4') => {
                    self.display_mode = DisplayMode::TranslationOnly;
                    self.draw_with_termion(terminal)?;
                },
                Key::Up => {
                    if self.scroll_position > 0 {
                        self.scroll_position -= 1;
                        self.draw_with_termion(terminal)?;
                    }
                },
                Key::Down => {
                    self.scroll_position += 1;
                    self.draw_with_termion(terminal)?;
                },
                Key::PageUp => {
                    let page_size = 10;
                    self.scroll_position = self.scroll_position.saturating_sub(page_size);
                    self.draw_with_termion(terminal)?;
                },
                Key::PageDown => {
                    let page_size = 10;
                    self.scroll_position += page_size;
                    self.draw_with_termion(terminal)?;
                },
                Key::Char('q') | Key::Char('Q') | Key::Ctrl('c') => {
                    return Ok(true); // Signal to exit
                },
                _ => {}
            }
        }
        
        // Check for terminal resize
        if terminal.check_resize()? {
            self.draw_with_termion(terminal)?;
        }
        
        Ok(false) // Continue running
    }
}