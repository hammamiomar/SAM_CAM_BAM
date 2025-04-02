// src/ui.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError}, // Only Receiver needed here
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use egui::{Align, ColorImage, ImageData, Layout, TextureHandle, TextureOptions, Vec2};
use log::{debug, error, info};
use nokhwa::utils::{CameraIndex, Resolution}; // Only need these utils

// Import functions and types from our other modules
use crate::{
    camera::{self, CameraThreadMsg}, 
    segmentation::{self,SegmentationThreadMsg}}; // Use camera module
// Use processing module

// --- Constants specific to UI ---
const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

// --- Renamed App Struct ---
// Changed name to avoid confusion with the old app.rs if it still exists
pub struct WebcamAppUI {
    texture: Option<TextureHandle>,
    // Receiver for messages from the camera thread
    // Join handle for the camera thread
    camera_thread_handle: Option<JoinHandle<()>>,
    // Shared signal to stop the camera thread
    cam_stop_signal: Arc<AtomicBool>,
    // State fields
    camera_error: Option<String>,
    camera_resolution: Option<Resolution>,

    seg_thread_rx: Receiver<SegmentationThreadMsg>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,
    seg_error: Option<String>,
    
    texture_size: Option<Vec2>,
    // FPS fields
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebcamAppUI {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI");
        let camera_index = CameraIndex::Index(0); // Define index here or get from config

        // Create the channel for communication
        let (camera_thread_tx, camera_thread_rx) = mpsc::channel();

        // Create the atomic bool for stopping the thread
        let cam_stop_signal = Arc::new(AtomicBool::new(false));

        // Clone necessary items for the camera thread
        let cam_stop_signal_clone = cam_stop_signal.clone();
        let cam_egui_ctx_clone = cc.egui_ctx.clone();
        let cam_thread_index = camera_index.clone();

        // --- Start the camera thread using the function from camera.rs ---
        let camera_thread_handle = Some(camera::start_camera_thread(
            cam_thread_index,
            camera_thread_tx,
            cam_stop_signal_clone,
            cam_egui_ctx_clone,
        ));

        // -------- Do the segmentation thread

        let (seg_thread_tx, seg_thread_rx) = mpsc::channel();
        let seg_stop_signal = Arc::new(AtomicBool::new(false));

        let seg_stop_signal_clone = seg_stop_signal.clone();
        let seg_egui_ctx_clone = cc.egui_ctx.clone();

        let seg_thread_handle = Some(segmentation::start_segmentation_thread(
            seg_thread_tx,
            camera_thread_rx,
            seg_stop_signal_clone,
            seg_egui_ctx_clone,
        ));


        // Initialize the UI state
        Self {
            texture: None, // Initialize texture as None // Store the receiver end of the channel
            camera_thread_handle, // Store the join handle
            cam_stop_signal: cam_stop_signal, // Store the stop signal Arc
            camera_error: None,
            camera_resolution: None,
            seg_stop_signal:seg_stop_signal,
            seg_thread_handle:seg_thread_handle,
            seg_error:None,
            seg_thread_rx:seg_thread_rx,
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
            } else {
                f32::INFINITY
            };
            self.frames_since_last_update = 0;
            self.last_fps_update_time = now;
        }
    }
}


// --- Implement eframe::App for our UI struct ---
impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();
        self.camera_error = None; // Clear previous error

        // --- Process messages from camera thread ---
        loop {
            match self.seg_thread_rx.try_recv() {
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(frame_arc) => {
                        // --- Pass frame to processing module ---
//TODO HERE I PROCESS WHEN I GET THE CAMERA FRAME
                        //let processed_frame_arc = segmentation::apply_visual_processing(raw_frame_arc);
                        
                        // Use the *processed* frame to update the texture
                        let size = frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);

                        if self.camera_resolution.is_none() {
                            self.camera_resolution =
                                Some(Resolution::new(size[0] as u32, size[1] as u32));
                        }

                        match self.texture {
                            Some(ref mut texture) => {
                                if self.texture_size.map_or(true, |s| s != frame_size_vec) {
                                    debug!("Texture size changed to: {:?}", frame_size_vec);
                                    self.texture_size = Some(frame_size_vec);
                                }
                                texture.set(ImageData::Color(frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                info!("Creating texture with size: {:?}", size);
                                let new_texture = ctx.load_texture(
                                    "webcam_stream",
                                    ImageData::Color(frame_arc),
                                    TextureOptions::LINEAR,
                                );
                                self.texture_size = Some(frame_size_vec);
                                self.texture = Some(new_texture);
                            }
                        }
                    }
                    SegmentationThreadMsg::Error(err) => {
                        self.seg_error = Some(err);
                    }
                    // Handle other message types like ResolutionInfo if added
                },
                Err(TryRecvError::Empty) => break, // No more messages
                Err(TryRecvError::Disconnected) => {
                    self.camera_error =
                        Some("Camera thread disconnected unexpectedly.".to_string());
                    error!("Camera thread disconnected!");
                    if let Some(handle) = self.camera_thread_handle.take() {
                         if let Err(e) = handle.join() {
                            error!("Camera thread panicked: {:?}", e);
                            self.camera_error = Some(format!("Camera thread panicked: {:?}", e));
                        }
                    }
                    break;
                }
            }
        }

        // --- Define the UI ---
        // (UI definition remains largely the same as the previous version)
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

        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(false)
            .show(ctx, |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                     ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                     ui.add_space(10.0);
                     if let Some(res) = self.camera_resolution {
                         ui.label(format!("Cam Res: {}x{}", res.width(), res.height()));
                     } else if self.camera_error.is_none() {
                         ui.label("Cam Res: ...");
                     }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SAM_CAM_BAM Stream");
            ui.separator();

            if let Some(err) = &self.camera_error {
                ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
            }

            match &self.texture {
                Some(texture) => {
                     if let Some(tex_size) = self.texture_size {
                        let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
                        let available_width = ui.available_width();
                        let available_height = ui.available_height();
                        let mut image_width = available_width;
                        let mut image_height = available_width / aspect_ratio;
                        if image_height > available_height {
                            image_height = available_height;
                            image_width = available_height * aspect_ratio;
                        }
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                             ui.add(
                                egui::Image::new(texture)
                                    .max_width(image_width)
                                    .max_height(image_height)
                                    .maintain_aspect_ratio(true)
                                    .rounding(5.0),
                             );
                        });
                    } else { ui.label("Texture exists but size unknown."); }
                }
                None if self.camera_error.is_none() => {
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0);
                        ui.spinner();
                        ui.label("Initializing camera stream...");
                    });
                }
                None => {} // Error shown above
            }
        });
    }

    /// Called when the application is about to close.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping camera thread...");
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.camera_thread_handle.take() {
             if let Err(e) = handle.join() {
                error!("Error joining camera thread: {:?}", e);
            } else {
                info!("Camera thread joined successfully.");
            }
        }
    }
}