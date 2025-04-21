// src/ui.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{JoinHandle},
    time::{Duration, Instant},
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use egui::{
    Align, Color32, ImageData, Layout, Pos2, Rect, Response, Sense, Stroke, Spinner,
    TextureHandle, TextureOptions, Vec2, widgets,
};
use log::{error, info, warn};
use nokhwa::utils::{CameraIndex, Resolution};
use cpal::{traits::StreamTrait, Stream}; // Need Stream to hold onto it

use crate::{
    camera::{self, CameraThreadMsg},
    segmentation::{self, SegmentationThreadMsg, UserInteractionSegMsg, MAX_TRACKS, TRACK_COLORS},
    music::{self, AudioProcessor}, // Updated import
    live_audio, // Added import
};

const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

// Enum to represent the status of the live audio capture
#[derive(Debug, Clone, PartialEq)]
enum LiveAudioStatus {
    Initializing,
    Running(u32, u16), // Sample Rate, Channels
    Error(String),
    Disabled, // Or Stopped
}

pub struct WebcamAppUI {
    // --- Core ---
    texture: Option<TextureHandle>,
    seg_to_ui_rx: Receiver<SegmentationThreadMsg>,
    user_interaction_tx: Sender<UserInteractionSegMsg>,

    // --- Threads & Signals ---
    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,
    audio_capture_thread_handle: Option<Stream>, // cpal stream handle
    audio_capture_stop_signal: Arc<AtomicBool>,
    audio_processor_thread_handle: Option<JoinHandle<()>>,
    audio_processor_stop_signal: Arc<AtomicBool>,

    // --- State ---
    camera_error: Option<String>, // Specific camera errors
    seg_error: Option<String>,    // Specific segmentation/processing errors
    live_audio_status: LiveAudioStatus, // Status of audio capture
    camera_resolution: Option<Resolution>,
    texture_size: Option<Vec2>,
    current_selection_slot: usize,

    // --- Performance ---
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI with FastSAM (Multi-Object + Live Audio)");
        let camera_index = CameraIndex::Index(0); // Or allow selection

        // Configure Model Options for FastSAM
        let device_str = "mps"; // Or "cpu", "cuda", etc.
        let dtype_str = "fp16"; // Or "fp32"
        let model_options = match usls::Options::fastsam_s()
            .with_model_device(device_str.try_into().expect("Bad device string"))
            .with_model_dtype(dtype_str.try_into().unwrap_or(usls::DType::Fp32))
            .with_model_file("models/FastSAM-s.onnx") // Ensure this path is correct!
            .with_nc(1) // Number of classes (1 for generic "object")
            .with_class_names(&["object"])
            .with_class_confs(&[0.35]) // Confidence threshold
            .with_iou(0.45) // IoU threshold for NMS
            .with_find_contours(true) // Find contours for masks
            .commit()
         {
            Ok(opts) => opts,
            Err(e) => {
                // Use panic here because the app can't run without model options
                panic!("FATAL: Model options failed to commit: {}. Check model path and dependencies.", e);
            }
         };


        // --- Channels ---
        let (cam_to_seg_tx, cam_to_seg_rx) = unbounded::<CameraThreadMsg>();
        let (seg_to_ui_tx, seg_to_ui_rx) = bounded::<SegmentationThreadMsg>(1); // Bounded to prevent UI lag
        let (user_interaction_tx, user_interaction_rx) = unbounded::<UserInteractionSegMsg>();
        // Audio Channels
        let (raw_samples_tx, raw_samples_rx) = bounded::<Vec<f32>>(10); // Bounded channel for audio samples
        let (intensities_tx, intensities_rx) = bounded::<Vec<f32>>(5);  // Bounded channel for FFT results


        // --- Stop Signals ---
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_capture_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_processor_stop_signal = Arc::new(AtomicBool::new(false));

        // Clones for threads
        let cam_stop_signal_clone = cam_stop_signal.clone();
        let cam_ctx_clone = cc.egui_ctx.clone();
        let seg_stop_signal_clone = seg_stop_signal.clone();
        let seg_ctx_clone = cc.egui_ctx.clone();
        let audio_capture_stop_signal_clone = audio_capture_stop_signal.clone();
        let audio_processor_stop_signal_clone = audio_processor_stop_signal.clone();


        // --- Start Camera Thread ---
        let cam_thread_handle = Some(camera::start_camera_thread(
            camera_index, cam_to_seg_tx, cam_stop_signal_clone, cam_ctx_clone,
        ));

        // --- Start Audio Capture ---
        let mut initial_audio_status = LiveAudioStatus::Initializing;
        let audio_capture_thread_handle = match live_audio::start_audio_capture(
            raw_samples_tx, // Sender for raw audio data
            audio_capture_stop_signal_clone,
        ) {
            Ok((stream, rate, channels)) => {
                initial_audio_status = LiveAudioStatus::Running(rate, channels);
                Some(stream) // Keep the stream object alive
            }
            Err(e) => {
                let err_msg = format!("Failed to start audio capture: {}", e);
                error!("{}", err_msg);
                initial_audio_status = LiveAudioStatus::Error(err_msg);
                None
            }
        };

        // --- Start Audio Processor Thread ---
        let audio_processor_thread_handle = match initial_audio_status {
            LiveAudioStatus::Running(rate, channels) => {
                let mut audio_processor = music::AudioProcessor::new(
                    raw_samples_rx, // Receiver for raw audio
                    intensities_tx, // Sender for FFT results
                    rate,
                    channels,
                    MAX_TRACKS, // Number of bands needed
                );
                 // Spawn the processor's run loop
                 Some(std::thread::spawn(move || {
                    audio_processor.run(audio_processor_stop_signal_clone);
                }))
            }
             _ => {
                warn!("Audio capture failed, not starting audio processor thread.");
                None // Don't start if capture failed
            }
        };


        // --- Start Segmentation Thread ---
        let seg_thread_handle = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx,
            cam_to_seg_rx,
            user_interaction_rx,
            intensities_rx, // Pass intensity receiver
            seg_stop_signal_clone,
            seg_ctx_clone,
            model_options,
        ));

        Self {
            texture: None,
            user_interaction_tx,
            seg_to_ui_rx,
            cam_thread_handle,
            cam_stop_signal,
            seg_thread_handle,
            seg_stop_signal,
            audio_capture_thread_handle,
            audio_capture_stop_signal,
            audio_processor_thread_handle,
            audio_processor_stop_signal,
            camera_error: None,
            seg_error: None,
            live_audio_status: initial_audio_status,
            camera_resolution: None,
            texture_size: None,
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
            current_selection_slot: 0,
        }
    }

    fn update_fps_counter(&mut self) {
        self.frames_since_last_update += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_update_time);

        if elapsed >= FPS_UPDATE_INTERVAL {
            let elapsed_secs = elapsed.as_secs_f32();
            self.last_calculated_fps = if elapsed_secs > 0.0 {
                self.frames_since_last_update as f32 / elapsed_secs
            } else { f32::INFINITY };
            self.frames_since_last_update = 0;
            self.last_fps_update_time = now;
            // log::debug!("UI FPS: {:.1}", self.last_calculated_fps); // Optional: log FPS
        }
    }

    fn ui_pos_to_image_coords(
        &self,
        ui_pos: Pos2,
        image_rect: Rect,
        displayed_image_size: Vec2,
    ) -> Option<(f32, f32)> {
         if !image_rect.contains(ui_pos) { return None; }
         if let Some(original_res) = self.camera_resolution {
             let original_width = original_res.width() as f32;
             let original_height = original_res.height() as f32;
             if displayed_image_size.x <= 0.0 || displayed_image_size.y <= 0.0 {
                 warn!("Displayed image size has zero dimension: {:?}", displayed_image_size);
                 return None;
             }
             // Coordinates relative to the top-left of the displayed image rect
             let relative_pos = ui_pos - image_rect.min;
             // Normalize coordinates based on the *displayed* size in the UI
             let norm_x = (relative_pos.x / displayed_image_size.x).clamp(0.0, 1.0);
             let norm_y = (relative_pos.y / displayed_image_size.y).clamp(0.0, 1.0);
             // Scale normalized coordinates to the original image dimensions
             let image_x = norm_x * original_width;
             let image_y = norm_y * original_height;

             // Ensure the calculated coordinates are within the original image bounds
             if image_x >= 0.0 && image_x < original_width && image_y >= 0.0 && image_y < original_height {
                  Some((image_x, image_y))
             } else {
                  // This should ideally not happen if normalization/scaling is correct, but clamp defensively
                  warn!("Calculated image coords slightly out of bounds: ({}, {}), clamping.", image_x, image_y);
                  Some((image_x.clamp(0.0, original_width - 1.0), image_y.clamp(0.0, original_height - 1.0)))
             }
         } else { None } // Original resolution unknown
    }
}

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();

        // --- 1. Process incoming messages FIRST (mutates state) ---
        let mut received_frame_this_update = false;
        loop {
            match self.seg_to_ui_rx.try_recv() {
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(processed_frame_arc) => {
                        received_frame_this_update = true;
                        let size = processed_frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);

                        // Check and update camera resolution if needed
                        if self.camera_resolution.is_none() ||
                           self.camera_resolution.map_or(false, |res| res.width() != size[0] as u32 || res.height() != size[1] as u32)
                        {
                            let new_res = Resolution::new(size[0] as u32, size[1] as u32);
                            if self.camera_resolution.is_some() {
                                warn!("Frame resolution ({:?}) differs from stored camera resolution ({:?}). Updating and clearing selections.", size, self.camera_resolution.unwrap());
                                // Resolution changed, likely invalidates selections
                                let _ = self.user_interaction_tx.send(UserInteractionSegMsg::ClearSelection);
                            } else {
                                info!("Received first frame, setting camera resolution to {}x{}", size[0], size[1]);
                            }
                            self.camera_resolution = Some(new_res);
                            self.texture_size = Some(frame_size_vec); // Update texture size as well
                        }

                        // Update egui texture
                        match self.texture {
                            Some(ref mut texture) => {
                                texture.set(ImageData::Color(processed_frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                let new_texture = ctx.load_texture(
                                    "webcam_stream", // Texture name
                                    ImageData::Color(processed_frame_arc),
                                    TextureOptions::LINEAR, // Or NEAREST
                                );
                                self.texture = Some(new_texture);
                            }
                        }
                        // Clear segmentation error if we successfully received a frame
                        self.seg_error = None;
                    }
                    SegmentationThreadMsg::Error(err) => {
                         // Avoid logging the same error repeatedly
                         if self.seg_error.as_ref() != Some(&err) {
                             error!("Segmentation Thread Error: {}", err);
                             self.seg_error = Some(err);
                         }
                    }
                    // SegmentationThreadMsg::AudioLoaded(_) => { /* Removed */ }
                },
                Err(TryRecvError::Empty) => break, // No more messages this cycle
                Err(TryRecvError::Disconnected) => {
                    let err_msg = "Segmentation thread disconnected unexpectedly.".to_string();
                    error!("{}", err_msg);
                    if self.seg_error.is_none() { self.seg_error = Some(err_msg); } // Show error in UI
                    // Attempt to join the thread to clean up resources
                    if let Some(handle) = self.seg_thread_handle.take() {
                         warn!("Attempting to join disconnected segmentation thread...");
                         if let Err(e) = handle.join() { self.seg_error = Some(format!("Segmentation thread panicked: {:?}", e)); }
                    }
                    break; // Stop processing messages from this disconnected channel
                 }
            }
        }

        // --- 2. Prepare state for UI drawing ---
        // let mut trigger_file_dialog = false; // Removed
        let mut clear_all_clicked = false;
        let mut interaction_error_this_frame: Option<String> = None;
        let mut new_selection_slot = self.current_selection_slot; // Local copy for radio buttons

        // --- 3. Define and draw UI ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
             egui::menu::bar(ui, |ui| {
                 let is_web = cfg!(target_arch = "wasm32");
                 if !is_web {
                    ui.menu_button("File", |ui| {
                         if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                         }
                    });
                    ui.add_space(16.0);
                 }
                 widgets::global_dark_light_mode_buttons(ui); // Use the new helper
             });
         });

        egui::SidePanel::left("control_panel")
            .resizable(false)
            .default_width(180.0) // Slightly wider for audio status
            .show(ctx, |ui| {
                ui.heading("Track Control").on_hover_text("Assign audio bands to detected objects"); ui.separator();
                ui.label("Select target slot:");
                ui.vertical(|ui| {
                    for i in 0..MAX_TRACKS {
                        ui.horizontal(|ui| {
                            let color = TRACK_COLORS[i];
                            let egui_color = Color32::from_rgb(color[0], color[1], color[2]);
                            let is_selected = new_selection_slot == i; // Use local var

                            // Use radio_value for simpler state management
                            if ui.radio_value(&mut new_selection_slot, i, "").clicked() {
                                // State is updated automatically by radio_value
                                info!("Selected slot {}", i);
                            };

                            // Draw colored rectangle next to radio button
                            let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                            let stroke = if is_selected { Stroke::new(2.0, ui.visuals().widgets.active.fg_stroke.color) } else { Stroke::NONE };
                            ui.painter().rect_filled(rect, 3.0, egui_color);
                            ui.painter().rect_stroke(rect, 3.0, stroke,egui::StrokeKind::Inside);

                            let label_text = match i {
                                0 => "Track 1 (Bass)",
                                1 => "Track 2 (Mid)",
                                2 => "Track 3 (High)",
                                _ => "Track ?",
                            };
                             ui.label(label_text).on_hover_text(format!("Click object in view to assign to {}", label_text));
                        });
                    }
                });
                ui.separator();
                ui.label("Left-Click Image: Assign object to selected track.");
                ui.label("Right-Click Image: Remove track at click location.");
                ui.separator();
                if ui.button("Clear All Selections").clicked() { clear_all_clicked = true; }
                ui.separator();

                ui.heading("Audio Status");
                ui.separator();
                match &self.live_audio_status {
                    LiveAudioStatus::Initializing => { ui.horizontal(|ui| { ui.spinner(); ui.label("Initializing..."); }); }
                    LiveAudioStatus::Running(rate, channels) => { ui.label(format!("Capturing: {} Hz, {} ch", rate, channels)).on_hover_text("Capturing live audio output"); }
                    LiveAudioStatus::Error(err) => { ui.colored_label(Color32::RED, "Error").on_hover_text(err); }
                    LiveAudioStatus::Disabled => { ui.colored_label(Color32::GRAY, "Disabled"); }
                }
                ui.separator();


                 ui.heading("Info"); ui.separator();
                 ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                 if let Some(res) = &self.camera_resolution { ui.label(format!("Cam Res: {}x{}", res.width(), res.height())); }
                 else if self.camera_error.is_none() && self.seg_error.is_none() && self.texture.is_none() { ui.label("Cam Res: Initializing..."); }
                 else if self.camera_error.is_some() || self.seg_error.is_some(){ ui.label("Cam Res: Error"); }
                 else { ui.label("Cam Res: Waiting..."); } // Waiting for first frame
            });

        // Apply deferred radio button mutation AFTER drawing the panel
        self.current_selection_slot = new_selection_slot;

        egui::CentralPanel::default().show(ctx, |ui| {
            // Display errors prominently if they exist
            if let Some(err) = &self.camera_error { ui.colored_label(egui::Color32::YELLOW, format!("Camera Status: {}", err)); }
            if let Some(err) = &self.seg_error { ui.colored_label(egui::Color32::RED, format!("Processing Status: {}", err)); }
            if let LiveAudioStatus::Error(err) = &self.live_audio_status { ui.colored_label(egui::Color32::RED, format!("Audio Status: {}", err)); }


            match &self.texture {
                Some(texture) => {
                    if let Some(tex_size) = self.texture_size {
                        // Calculate display size maintaining aspect ratio
                        let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
                        let available_width = ui.available_width();
                        let available_height = ui.available_height();
                        let mut image_width = available_width;
                        let mut image_height = available_width / aspect_ratio;
                        if image_height > available_height {
                            image_height = available_height;
                            image_width = available_height * aspect_ratio;
                        }
                        let displayed_size = Vec2::new(image_width, image_height);

                        // Center the image
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            let image_widget = egui::Image::new(texture) // Use texture ID and calculated size
                                .max_height(image_height)
                                .max_width(image_width)
                                .sense(Sense::click()); // Sense clicks for interaction
                            let response: Response = ui.add(image_widget);

                            // --- Click Handlers: Send messages, store potential errors locally ---
                            let tx = self.user_interaction_tx.clone(); // Clone sender for click handlers
                            // Use interact_pointer_pos for position relative to screen/window
                            // Then use ui_pos_to_image_coords to convert if it's within the image rect
                            let interact_pos = response.interact_pointer_pos();
                            let maybe_coords = interact_pos.and_then(|pos| {
                                self.ui_pos_to_image_coords(pos, response.rect, displayed_size)
                            });
                            let slot_to_use = self.current_selection_slot; // Read current slot

                            if response.clicked_by(egui::PointerButton::Primary) {
                                if let Some((img_x, img_y)) = maybe_coords {
                                    info!("LClick on Image -> Assign Slot {} @ ({:.1}, {:.1})", slot_to_use, img_x, img_y);
                                    if let Err(e) = tx.send(UserInteractionSegMsg::SelectObject { slot: slot_to_use, x: img_x, y: img_y }) {
                                        let err_msg = format!("Comm Err (Select): {}", e);
                                        error!("{}", err_msg);
                                        interaction_error_this_frame = Some(err_msg); // Store error locally
                                    }
                                } else if interact_pos.is_some() {
                                    warn!("LClick outside image bounds (or resolution unknown).");
                                }
                            } else if response.clicked_by(egui::PointerButton::Secondary) {
                                if let Some((img_x, img_y)) = maybe_coords {
                                     info!("RClick on Image -> Remove @ ({:.1}, {:.1})", img_x, img_y);
                                     if let Err(e) = tx.send(UserInteractionSegMsg::RemoveObjectAt { x: img_x, y: img_y }) {
                                         let err_msg = format!("Comm Err (Remove): {}", e);
                                         error!("{}", err_msg);
                                         interaction_error_this_frame = Some(err_msg); // Store error locally
                                     }
                                } else if interact_pos.is_some() {
                                     warn!("RClick outside image bounds (or resolution unknown).");
                                }
                            }
                            // --- End Click Handlers ---

                        }); // End ui.with_layout for centering
                    } else {
                        // Should not happen if texture exists, but handle defensively
                        ui.label("Texture exists but size unknown.");
                    }
                }
                 None if self.camera_error.is_none() && self.seg_error.is_none() && self.live_audio_status != LiveAudioStatus::Disabled => {
                     // No texture yet, no errors reported -> Show loading spinner
                     ui.centered_and_justified(|ui| {
                        ui.add(Spinner::new()); // Use egui's built-in spinner
                        ui.label("Initializing stream...");
                    });
                 }
                 None => { // No texture, and likely an error occurred
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(ui.visuals().error_fg_color, "Stream unavailable");
                        ui.label("Check error messages in side panel or console.");
                    });
                 }
            } // End match &self.texture
        }); // End CentralPanel

        // --- 4. Process deferred actions AFTER drawing UI ---
        if clear_all_clicked {
            info!("Sending deferred ClearSelection message.");
            if let Err(e) = self.user_interaction_tx.send(UserInteractionSegMsg::ClearSelection) {
                let err_msg = format!("Comm Err (Clear): {}", e);
                error!("{}", err_msg);
                interaction_error_this_frame = Some(err_msg); // Store error locally
            }
        }

        // Removed file dialog trigger

        // --- 5. Apply interaction error state mutation LAST ---
        if let Some(err) = interaction_error_this_frame {
            // Only update if no more critical error is already present from message loop
            // Prioritize segmentation errors over interaction errors for display
            if self.seg_error.is_none() {
                self.seg_error = Some(err);
            }
        }

        // Request repaint if no frame was received but we might need UI updates (e.g., status change)
        if !received_frame_this_update {
             ctx.request_repaint_after(Duration::from_millis(100)); // Request repaint reasonably soon
        }

    } // --- End of update method ---

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping all threads...");

        // 1. Signal all threads to stop
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);
        self.audio_capture_stop_signal.store(true, Ordering::Relaxed);
        self.audio_processor_stop_signal.store(true, Ordering::Relaxed);
        info!("Stop signals sent.");

        // 2. Stop the audio stream explicitly (important for cpal)
        if let Some(stream) = self.audio_capture_thread_handle.take() {
            info!("Stopping audio stream...");
            if let Err(e) = stream.pause() { // Pause instead of drop? Or just let drop handle it? Pause seems safer.
                error!("Error pausing audio stream: {}", e);
            } else {
                info!("Audio stream paused.");
            }
            drop(stream); // Ensure it's dropped
             info!("Audio stream dropped.");
        } else {
            info!("Audio stream handle already taken or never existed.");
        }


        // 3. Join threads (with timeouts?)
        if let Some(handle) = self.cam_thread_handle.take() {
             info!("Joining camera thread...");
             if let Err(e) = handle.join() { error!("Camera thread join error: {:?}", e); } else { info!("Camera thread joined."); }
        }
        if let Some(handle) = self.audio_processor_thread_handle.take() {
            info!("Joining audio processor thread...");
            if let Err(e) = handle.join() { error!("Audio processor thread join error: {:?}", e); } else { info!("Audio processor thread joined."); }
        }
         if let Some(handle) = self.seg_thread_handle.take() {
            info!("Joining segmentation thread...");
            if let Err(e) = handle.join() { error!("Segmentation thread join error: {:?}", e); } else { info!("Segmentation thread joined."); }
        }

        info!("All threads stopped and joined.");
    }
}

// Helper for centering content, useful for loading/error states
trait CenteredJustified {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui));
}

impl CenteredJustified for egui::Ui {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui)) {
        self.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.add_space(ui.available_height() / 3.0); // Adjust vertical spacing
            add_contents(ui);
        });
    }
}