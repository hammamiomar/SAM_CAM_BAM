// src/ui.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, // Keep Arc
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

// --- Use crossbeam_channel ---
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
// --- End crossbeam_channel ---

use egui::{Align, ColorImage, ImageData, Layout, TextureHandle, TextureOptions, Vec2};
use log::{debug, error, info};
use nokhwa::utils::{CameraIndex, Resolution}; // Only need these utils

// Import functions and types from our other modules
// Need camera module for start_camera_thread and its message type (for channel)
use crate::camera::{self, CameraThreadMsg};
// Need segmentation module for start_segmentation_thread and its message type
use crate::segmentation::{self, SegmentationThreadMsg};
// Need usls Options for configuration
use usls::Options;

// --- Constants specific to UI ---
const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

pub struct WebcamAppUI {
    texture: Option<TextureHandle>,

    // --- Thread Handles and Signals ---
    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,

    // --- Channel Receiver (Receives from Segmentation Thread) ---
    seg_thread_rx: Receiver<SegmentationThreadMsg>, // Using crossbeam Receiver

    // --- State Fields ---
    camera_error: Option<String>,
    seg_error: Option<String>,
    camera_resolution: Option<Resolution>,
    texture_size: Option<Vec2>,

    // --- FPS Fields ---
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebcamAppUI {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI");
        let camera_index = CameraIndex::Index(0); // Or specific index

        // --- Configure Model Options ---
        // Make sure the path is correct for your system!
        let model_options = usls::Options::fastsam_s()
            .with_model_device("mps".try_into().expect("Failed to parse MPS device string"))
            .with_model_dtype("fp16".try_into().unwrap_or(usls::DType::Fp32))
            .with_model_file("/Users/omarhammami/Projects/RC/CV/SAM_CAM_BAM/models/FastSam-S.onnx") // <-- YOUR PATH HERE
            .with_nc(1) // FastSAM usually has 1 class ("object")
            .with_class_names(&["object"]) // Optional name for the class
            .with_class_confs(&[0.25]) // Confidence threshold
            .with_iou(0.45) // NMS IoU threshold
             // No commit() here, let the thread handle potential download/load
            ;

        // --- Channels using crossbeam-channel ---
        // Camera -> Segmentation: Unbounded to prevent camera thread blocking.
        // Segmentation thread will manage latency by dropping old frames.
        let (cam_to_seg_tx, cam_to_seg_rx) = unbounded::<CameraThreadMsg>();
        // Segmentation -> UI: Bounded(1) to ensure UI gets latest, and seg thread can drop if UI lags.
        let (seg_to_ui_tx, seg_to_ui_rx) = bounded::<SegmentationThreadMsg>(1);

        // --- Stop Signals ---
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));

        // --- Clones for Threads ---
        let cam_stop_signal_clone = cam_stop_signal.clone();
        let cam_ctx_clone = cc.egui_ctx.clone();
        let seg_stop_signal_clone = seg_stop_signal.clone();
        let seg_ctx_clone = cc.egui_ctx.clone();

        // --- Start Camera Thread ---
        let cam_thread_handle = Some(camera::start_camera_thread(
            camera_index.clone(),
            cam_to_seg_tx, // Pass crossbeam sender
            cam_stop_signal_clone,
            cam_ctx_clone, // For error reporting from camera thread
        ));

        // --- Start Segmentation Thread ---
        let seg_thread_handle = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx,      // Pass crossbeam sender (to UI)
            cam_to_seg_rx,     // Pass crossbeam receiver (from Cam)
            seg_stop_signal_clone,
            seg_ctx_clone,     // For repaint requests from seg thread
            model_options,     // Pass the configured model options
        ));

        // --- Initialize UI State ---
        Self {
            texture: None,
            cam_thread_handle,
            cam_stop_signal,
            seg_thread_handle,
            seg_stop_signal,
            seg_thread_rx: seg_to_ui_rx, // Store the receiver from the seg thread
            camera_error: None,
            seg_error: None,
            camera_resolution: None,
            texture_size: None,
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
        }
    }

    /// Calculates and updates the FPS counter.
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
        }
    }
}


// --- Implement eframe::App for our UI struct ---
impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();
        // Clear errors each frame, let new messages overwrite them if they occur
        self.camera_error = None;
        self.seg_error = None;

        // --- Process messages from SEGMENTATION thread ---
        loop {
            match self.seg_thread_rx.try_recv() { // Use crossbeam try_recv
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(processed_frame_arc) => {
                        // We received a final, displayable ColorImage
                        let size = processed_frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);

                        // Update resolution if needed (might get from camera thread eventually)
                        if self.camera_resolution.is_none() {
                            self.camera_resolution =
                                Some(Resolution::new(size[0] as u32, size[1] as u32));
                        }

                        // Update texture logic remains the same
                        match self.texture {
                            Some(ref mut texture) => {
                                if self.texture_size.map_or(true, |s| s != frame_size_vec) {
                                    debug!("Texture size changed to: {:?}", frame_size_vec);
                                    self.texture_size = Some(frame_size_vec);
                                }
                                texture.set(ImageData::Color(processed_frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                info!("Creating texture with size: {:?}", size);
                                let new_texture = ctx.load_texture(
                                    "webcam_stream",
                                    ImageData::Color(processed_frame_arc),
                                    TextureOptions::LINEAR,
                                );
                                self.texture_size = Some(frame_size_vec);
                                self.texture = Some(new_texture);
                            }
                        }
                    }
                    SegmentationThreadMsg::Error(err) => {
                        // Distinguish errors reported BY the segmentation thread
                        self.seg_error = Some(err);
                    }
                },
                Err(TryRecvError::Empty) => break, // No more messages this frame
                Err(TryRecvError::Disconnected) => {
                    // Handle segmentation thread unexpected stop
                    self.seg_error = Some("Segmentation thread disconnected unexpectedly.".to_string());
                    error!("Segmentation thread disconnected!");
                    if let Some(handle) = self.seg_thread_handle.take() {
                         if let Err(e) = handle.join() {
                            error!("Segmentation thread panicked: {:?}", e);
                            self.seg_error = Some(format!("Segmentation thread panicked: {:?}", e));
                        }
                    }
                    // Consider stopping camera if seg thread dies
                    // self.cam_stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        // --- Define the UI ---
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
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::TopBottomPanel::bottom("bottom_panel").resizable(false).show(ctx, |ui| {
             ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                 ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                 ui.add_space(10.0);
                 if let Some(res) = self.camera_resolution {
                     ui.label(format!("Cam Res: {}x{}", res.width(), res.height()));
                 } else if self.camera_error.is_none() && self.seg_error.is_none() {
                     ui.label("Cam Res: ...");
                 } else {
                     ui.label("Cam Res: Error");
                 }
             });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SAM_CAM_BAM Stream");
            ui.separator();

            // Display errors (can distinguish source now)
            if let Some(err) = &self.camera_error { ui.colored_label(egui::Color32::YELLOW, format!("Camera Status: {}", err)); }
            if let Some(err) = &self.seg_error { ui.colored_label(egui::Color32::RED, format!("Segmentation Status: {}", err)); }

            // Display texture or loading/error state
            match &self.texture {
                Some(texture) => {
                     if let Some(tex_size) = self.texture_size {
                        let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
                        let available_width = ui.available_width(); let available_height = ui.available_height();
                        let mut image_width = available_width; let mut image_height = available_width / aspect_ratio;
                        if image_height > available_height { image_height = available_height; image_width = available_height * aspect_ratio; }
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                             ui.add( egui::Image::new(texture).max_width(image_width).max_height(image_height).maintain_aspect_ratio(true).rounding(5.0), );
                        });
                    } else { ui.label("Texture exists but size unknown."); }
                }
                None if self.camera_error.is_none() && self.seg_error.is_none() => {
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0); ui.spinner(); ui.label("Initializing stream...");
                    });
                }
                None => {} // Error message(s) shown above
            }
        });
    }

    /// Called when the application is about to close.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping threads...");
        // --- Signal BOTH threads to stop ---
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);

        // --- Join camera thread ---
        if let Some(handle) = self.cam_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining camera thread: {:?}", e); }
             else { info!("Camera thread joined successfully."); }
        }
        // --- Join segmentation thread ---
        if let Some(handle) = self.seg_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining segmentation thread: {:?}", e); }
             else { info!("Segmentation thread joined successfully."); }
        }
    }
}