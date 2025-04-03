// src/ui.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, TryRecvError}, // Use mpsc Receiver
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use egui::{Align, ColorImage, ImageData, Layout, TextureHandle, TextureOptions, Vec2};
use log::{debug, error, info};
use nokhwa::utils::{CameraIndex, Resolution};

// Use messages from both camera (indirectly via seg) and segmentation
use crate::{
    camera::{self}, // Keep camera module for start function
    segmentation::{self, SegmentationThreadMsg}, // Use segmentation module
};

const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

pub struct WebcamAppUI {
    texture: Option<TextureHandle>,

    // --- Thread Handles and Signals ---
    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,

    // --- Channel Receiver ---
    // We only receive from the *segmentation* thread now
    seg_thread_rx: Receiver<SegmentationThreadMsg>,

    // --- State Fields ---
    camera_error: Option<String>, // For camera init/capture errors relayed
    seg_error: Option<String>,    // For segmentation errors
    camera_resolution: Option<Resolution>, // Can still be useful to know
    texture_size: Option<Vec2>,

    // --- FPS Fields ---
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI");
        let camera_index = CameraIndex::Index(0);

        // --- Channels ---
        // Camera -> Segmentation
        let (cam_to_seg_tx, cam_to_seg_rx) = mpsc::channel();
        // Segmentation -> UI
        let (seg_to_ui_tx, seg_to_ui_rx) = mpsc::channel();

        // --- Stop Signals ---
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));

        // --- Start Camera Thread ---
        let cam_stop_signal_clone = cam_stop_signal.clone();
        let cam_ctx_clone = cc.egui_ctx.clone(); // Clone for camera thread error reporting
        let cam_thread_handle = Some(camera::start_camera_thread(
            camera_index.clone(),
            cam_to_seg_tx, // Send to segmentation thread
            cam_stop_signal_clone,
            cam_ctx_clone,
        ));

        let model_options = usls::Options::fastsam_s() // Start with the preset for FastSAM-Small
        .with_model_device("mps".try_into().expect("Failed to parse MPS device string")) // Specify MPS backend
        // Optional: Specify data type (fp16 often faster on Apple Silicon if supported by model/ORT)
        .with_model_dtype("fp16".try_into().unwrap_or(usls::DType::Fp32)) // Use fp16 if possible, else default
        // Optional: Adjust thresholds (defaults might exist in fastsam_s constructor)
        .with_nc(1) // Specify 1 class for FastSAM (usually just "object")
        // --- End of added line ---
        .with_class_confs(&[0.25]) // Confidence threshold for the single "object" class
        .with_iou(0.45) // NMS threshold
        // Optional: provide a name if you want it displayed
        .with_class_names(&["object"])
        ; // Keep uncommitted

        // --- Start Segmentation Thread ---
        let seg_stop_signal_clone = seg_stop_signal.clone();
        let seg_ctx_clone = cc.egui_ctx.clone(); // Clone for segmentation thread repaint requests
        let seg_thread_handle = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx,      // Send final images to UI
            cam_to_seg_rx,     // Receive frames from camera
            seg_stop_signal_clone,
            seg_ctx_clone,
            model_options
        ));


        // --- Initialize UI State ---
        Self {
            texture: None,
            cam_thread_handle, // Store handles and signals
            cam_stop_signal,
            seg_thread_handle,
            seg_stop_signal,
            seg_thread_rx: seg_to_ui_rx, // Receive from segmentation thread
            camera_error: None,          // Clear errors initially
            seg_error: None,
            camera_resolution: None,
            texture_size: None,
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
        }
    }

    // update_fps_counter remains the same
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

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();
        // Clear errors each frame, let new messages overwrite them
        self.camera_error = None;
        self.seg_error = None;

        // --- Process messages from SEGMENTATION thread ---
        loop {
            match self.seg_thread_rx.try_recv() { // Receive from seg thread
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(processed_frame_arc) => {
                        // We received a final, displayable ColorImage
                        let size = processed_frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);

                        // Still useful to store resolution info if available
                        if self.camera_resolution.is_none() {
                             // Note: This assumes seg output res == camera res
                             self.camera_resolution = Some(Resolution::new(size[0] as u32, size[1] as u32));
                        }

                        // Update texture logic remains the same
                        match self.texture {
                            Some(ref mut texture) => {
                                if self.texture_size.map_or(true, |s| s != frame_size_vec) {
                                    debug!("Texture size changed to: {:?}", frame_size_vec);
                                    self.texture_size = Some(frame_size_vec);
                                }
                                // Use the processed frame
                                texture.set(ImageData::Color(processed_frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                info!("Creating texture with size: {:?}", size);
                                // Use the processed frame
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
                        // Distinguish segmentation errors
                        self.seg_error = Some(err);
                    }
                },
                Err(TryRecvError::Empty) => break, // No more messages
                Err(TryRecvError::Disconnected) => {
                    // Handle segmentation thread disconnection
                    self.seg_error = Some("Segmentation thread disconnected unexpectedly.".to_string());
                    error!("Segmentation thread disconnected!");
                    if let Some(handle) = self.seg_thread_handle.take() {
                         if let Err(e) = handle.join() {
                            error!("Segmentation thread panicked: {:?}", e);
                            self.seg_error = Some(format!("Segmentation thread panicked: {:?}", e));
                        }
                    }
                    // Should we also stop the camera thread if seg thread dies? Maybe.
                    // self.cam_stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        // --- Define the UI ---
        // (Largely the same, but display both camera and seg errors if needed)
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| { /* ... menu ... */
             egui::menu::bar(ui, |ui| {
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web { ui.menu_button("File", |ui| { if ui.button("Quit").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); } }); ui.add_space(16.0); }
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::TopBottomPanel::bottom("bottom_panel").resizable(false).show(ctx, |ui| { /* ... FPS / Res ... */
             ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                 ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                 ui.add_space(10.0);
                 if let Some(res) = self.camera_resolution { ui.label(format!("Cam Res: {}x{}", res.width(), res.height())); }
                 else if self.camera_error.is_none() { ui.label("Cam Res: ..."); }
             });
        });

        egui::CentralPanel::default().show(ctx, |ui| { /* ... heading ... */
            ui.heading("SAM_CAM_BAM Stream");
            ui.separator();

            // Display errors from both sources
            if let Some(err) = &self.camera_error { ui.colored_label(egui::Color32::YELLOW, format!("Camera Status: {}", err)); }
            if let Some(err) = &self.seg_error { ui.colored_label(egui::Color32::RED, format!("Segmentation Status: {}", err)); }

            // Display texture or loading/error state
            match &self.texture {
                Some(texture) => { /* ... display image ... */
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
                None if self.camera_error.is_none() && self.seg_error.is_none() => { /* ... show spinner ... */
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0); ui.spinner(); ui.label("Initializing stream...");
                    });
                }
                None => {} // Error shown above
            }
        });
    }

    // --- UPDATED on_exit ---
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping threads...");
        // Signal BOTH threads to stop
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);

        // Join camera thread
        if let Some(handle) = self.cam_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining camera thread: {:?}", e); }
             else { info!("Camera thread joined successfully."); }
        }
        // Join segmentation thread
        if let Some(handle) = self.seg_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining segmentation thread: {:?}", e); }
             else { info!("Segmentation thread joined successfully."); }
        }
    }
}