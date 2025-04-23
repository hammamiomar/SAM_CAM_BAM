// src/ui.rs
use cpal::{traits::StreamTrait, Stream};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use crate::segmentation::{SegmentationOutputMsg, UserInteractionSegMsg, VisualObjectData, MAX_TRACKS};
use crate::{
    camera::{self, CameraToSegMsg, CameraToUiMsg}, // Import new message types
    live_audio,
    music::{self},
    segmentation::{self},
    visuals,
};
use egui::{
    widgets, Color32, ImageData, TextureHandle, TextureOptions, Pos2, Rect, // Removed unused Align, Layout, Painter, Vec2
};
use image::{RgbImage}; // Need RgbImage for receiving direct frames
use log::{error, info, warn};
use nokhwa::utils::{CameraIndex, Resolution};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);
#[derive(Debug, Clone, PartialEq)]
enum LiveAudioStatus {
    Initializing,
    Running(u32, u16),
    Error(String),
    Disabled,
}

pub struct WebcamAppUI {
    // Texture for the background camera feed
    bg_texture: Option<TextureHandle>,
    // Receive segmentation results (potentially lagging)
    seg_to_ui_rx: Receiver<SegmentationOutputMsg>,
    // Receive latest camera frames directly for low-latency display
    cam_to_ui_rx: Receiver<CameraToUiMsg>, // New receiver
    _user_interaction_tx: Sender<UserInteractionSegMsg>,

    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,
    audio_capture_thread_handle: Option<Stream>,
    audio_capture_stop_signal: Arc<AtomicBool>,
    audio_processor_thread_handle: Option<JoinHandle<()>>,
    audio_processor_stop_signal_clone: Arc<AtomicBool>,

    camera_error: Option<String>,
    seg_error: Option<String>,
    live_audio_status: LiveAudioStatus,
    camera_resolution: Option<Resolution>,
    // Store the latest *processed* visual objects
    last_visual_objects: Vec<VisualObjectData>,
    last_segmentation_processing_time: Duration,

    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
    ui_frame_count: u64,
    rng: SmallRng,
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI (epaint Viz - Decoupled)");
        let camera_index = CameraIndex::Index(0);
        let device_str = "mps";
        let dtype_str = "fp16";
        let model_options = match usls::Options::fastsam_s()
            .with_model_device(device_str.try_into().expect("Bad device"))
            .with_model_dtype(dtype_str.try_into().unwrap_or(usls::DType::Fp32))
            .with_find_contours(true)
            .commit()
        {
            Ok(o) => o,
            Err(e) => {
                panic!("FATAL: Failed to setup model options: {}", e);
            }
        };

        // --- Channels ---
        let (cam_to_seg_tx, cam_to_seg_rx) = unbounded::<CameraToSegMsg>(); // Renamed type
        let (cam_to_ui_tx, cam_to_ui_rx) = bounded::<CameraToUiMsg>(2); // New channel, small buffer
        let (seg_to_ui_tx, seg_to_ui_rx) = bounded::<SegmentationOutputMsg>(1);
        let (_user_interaction_tx, user_interaction_rx) = unbounded();
        let (raw_samples_tx, raw_samples_rx) = bounded(10);
        let (intensities_tx, intensities_rx) = bounded(5);

        // --- Stop Signals ---
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_capture_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_processor_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_proc_stop_for_thread = audio_processor_stop_signal.clone();
        let audio_proc_stop_for_ui = audio_processor_stop_signal.clone();

        // --- Start Threads ---
        let cam_thread = Some(camera::start_camera_thread(
            camera_index,
            cam_to_seg_tx, // Sender for segmentation thread
            cam_to_ui_tx,  // Sender for UI thread
            cam_stop_signal.clone(),
            cc.egui_ctx.clone(),
        ));

        let initial_audio_status;
        let audio_capture_thread =
            match live_audio::start_audio_capture(raw_samples_tx, audio_capture_stop_signal.clone()) {
                Ok((s, r, c)) => {
                    initial_audio_status = LiveAudioStatus::Running(r, c);
                    Some(s)
                }
                Err(e) => {
                    let m = format!("Audio capture failed: {}", e);
                    error!("{}", m);
                    initial_audio_status = LiveAudioStatus::Error(m);
                    None
                }
            };

        let audio_processor_thread = match initial_audio_status {
            LiveAudioStatus::Running(r, c) => {
                let mut p =
                    music::AudioProcessor::new(raw_samples_rx, intensities_tx, r, c, MAX_TRACKS);
                Some(std::thread::spawn(move || p.run(audio_proc_stop_for_thread)))
            }
            _ => {
                warn!("No audio processor thread started due to audio init status.");
                None
            }
        };

        let seg_thread = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx,
            cam_to_seg_rx, // Use renamed type
            user_interaction_rx,
            intensities_rx,
            seg_stop_signal.clone(),
            cc.egui_ctx.clone(),
            model_options,
        ));

        Self {
            bg_texture: None,
            seg_to_ui_rx,
            cam_to_ui_rx, // Store the new receiver
            _user_interaction_tx,
            cam_thread_handle: cam_thread,
            cam_stop_signal,
            seg_thread_handle: seg_thread,
            seg_stop_signal,
            audio_capture_thread_handle: audio_capture_thread,
            audio_capture_stop_signal,
            audio_processor_thread_handle: audio_processor_thread,
            audio_processor_stop_signal_clone: audio_proc_stop_for_ui,
            camera_error: None,
            seg_error: None,
            live_audio_status: initial_audio_status,
            camera_resolution: None,
            last_visual_objects: Vec::new(),
            last_segmentation_processing_time: Duration::default(),
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
            ui_frame_count: 0,
            rng: SmallRng::from_rng(&mut rand::thread_rng()),
        }
    }

    fn update_fps_counter(&mut self) {
        // ... (unchanged) ...
        self.frames_since_last_update += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_update_time);
        if elapsed >= FPS_UPDATE_INTERVAL {
            self.last_calculated_fps =
                self.frames_since_last_update as f32 / elapsed.as_secs_f32().max(0.001);
            self.frames_since_last_update = 0;
            self.last_fps_update_time = now;
        }
    }

    // Helper to convert RgbImage to ColorImage
    fn rgb_to_color_image(rgb: Arc<RgbImage>) -> egui::ColorImage {
        let size = [rgb.width() as _, rgb.height() as _];
        let pixels = rgb.as_flat_samples();
        egui::ColorImage::from_rgb(size, pixels.as_slice())
    }
}

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();
        self.ui_frame_count += 1;

        // --- 1. Receive Latest Camera Frame for Display ---
        let mut latest_cam_frame: Option<Arc<RgbImage>> = None;
        loop {
            match self.cam_to_ui_rx.try_recv() {
                Ok(CameraToUiMsg::Frame(f)) => {
                    latest_cam_frame = Some(f); // Keep only the latest
                }
                 Ok(CameraToUiMsg::Error(e)) => {
                    if self.camera_error.as_ref() != Some(&e) {
                        self.camera_error = Some(e);
                    }
                    latest_cam_frame = None; // Clear frame on error
                 }
                Err(TryRecvError::Empty) => {
                    break; // No more frames this cycle
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Camera thread disconnected (UI receiver).");
                    self.camera_error = Some("Camera disconnected.".to_string());
                    if let Some(h) = self.cam_thread_handle.take() { h.join().ok(); }
                    break;
                }
            }
        }

        // Update background texture if a new frame was received
        if let Some(rgb_frame) = latest_cam_frame {
            let res = Resolution::new(rgb_frame.width(), rgb_frame.height());
            if self.camera_resolution.map_or(true, |r| r != res) {
                 self.camera_resolution = Some(res);
                 info!("Detected camera resolution: {}x{}", res.width(), res.height());
            }
             let color_image = Self::rgb_to_color_image(rgb_frame); // Convert here
             let texture_options = TextureOptions::LINEAR;
            match self.bg_texture {
                Some(ref mut handle) => {
                    handle.set(ImageData::Color(Arc::new(color_image)), texture_options);
                }
                None => {
                    self.bg_texture = Some(ctx.load_texture(
                        "cam_feed",
                        ImageData::Color(Arc::new(color_image)),
                        texture_options,
                    ));
                }
            }
            self.camera_error = None; // Clear camera error on successful frame
        }


        // --- 2. Receive Segmentation Results (Might be older than displayed frame) ---
        loop {
            match self.seg_to_ui_rx.try_recv() {
                Ok(msg) => {
                    match msg {
                        SegmentationOutputMsg::FrameData {
                             // We might not strictly *need* the frame here anymore
                             // if bg_texture is updated above, but keeping it
                             // doesn't hurt much and simplifies logic slightly.
                            original_frame: _frame_data_color_image, // Can ignore for display
                            objects,
                            processing_time,
                        } => {
                            // Just store the latest object data and processing time
                            self.last_visual_objects = objects;
                            self.last_segmentation_processing_time = processing_time;
                            self.seg_error = None; // Clear seg error on success
                        }
                        SegmentationOutputMsg::Error(e) => {
                             warn!("Received error from segmentation thread: {}", e);
                            if self.seg_error.as_ref() != Some(&e) {
                                self.seg_error = Some(e);
                            }
                            // Maybe clear last_visual_objects on error? Optional.
                            // self.last_visual_objects.clear();
                        }
                    }
                }
                Err(TryRecvError::Empty) => {
                    break; // No more seg results this cycle
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Segmentation thread disconnected (UI receiver).");
                     let m = "Segmentation thread disconnected.".to_string();
                    if self.seg_error.is_none() { self.seg_error = Some(m); }
                    if let Some(h) = self.seg_thread_handle.take() { h.join().ok(); }
                    self.last_visual_objects.clear(); // Clear visuals on disconnect
                    break;
                }
            }
        }

        // --- UI Layout (Mostly Unchanged) ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
             egui::menu::bar(ui, |ui| {
                if !cfg!(target_arch = "wasm32") {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
                widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::SidePanel::left("control_panel")
            .resizable(true)
            .default_width(200.0)
            .show(ctx, |ui| {
                 // ... (Audio Status, Performance, Status Panel code remains the same) ...
                 // Status Panel Example:
                ui.heading("Status").on_hover_text("Camera and processing status");
                ui.separator();
                match &self.camera_resolution {
                    Some(r) => {ui.label(format!("Cam Res: {}x{}", r.width(), r.height()));},
                    None if self.camera_error.is_none() && self.seg_error.is_none() && self.bg_texture.is_none() =>
                        {ui.label("Cam Res: Initializing...");},
                    None => {ui.label("Cam Res: Unknown/Error");},
                }
                if let Some(err) = &self.camera_error { /* Display camera error */ ui.small(err); }
                if let Some(err) = &self.seg_error { /* Display seg error */ ui.small(err); }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // --- Drawing Area ---
            if let Some(texture) = &self.bg_texture {
                let available_rect = ui.available_rect_before_wrap();
                let tex_size = texture.size_vec2(); // Use texture's reported size
                let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };

                let mut draw_size = available_rect.size();
                 // Calculate draw size based on aspect ratio
                if draw_size.x / aspect_ratio > available_rect.height() {
                    draw_size.x = available_rect.height() * aspect_ratio;
                    draw_size.y = available_rect.height();
                } else {
                    draw_size.y = available_rect.width() / aspect_ratio;
                    draw_size.x = available_rect.width();
                }

                let image_rect = Rect::from_center_size(available_rect.center(), draw_size);

                // 1. Draw Background Image (Latest camera frame)
                 ui.painter().image(
                    texture.id(),
                    image_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE
                );

                // 2. Draw Visualizations (Latest processed objects) on top
                if let Some(cam_res) = self.camera_resolution { // Use camera_resolution for mapping
                    if cam_res.width() > 0 && cam_res.height() > 0 {
                         let painter = ui.painter_at(image_rect);
                        // Use the *stored* last_visual_objects
                        for object_data in &self.last_visual_objects {
                            if !object_data.contours.is_empty() {
                                visuals::draw_object_visuals_epaint(
                                    &painter,
                                    image_rect, // Target drawing area on screen
                                    cam_res.width() as f32, // Original frame width for mapping
                                    cam_res.height() as f32, // Original frame height for mapping
                                    object_data,
                                    self.ui_frame_count,
                                    &mut self.rng,
                                );
                            }
                        }
                    }
                }
            } else if self.camera_error.is_none() && self.seg_error.is_none() {
                // Loading state
                ui.centered_and_justified(|ui| {
                    ui.spinner();
                    ui.label("Initializing stream...");
                });
            } else {
                 // Error state
                ui.centered_and_justified(|ui| {
                    ui.colored_label(ui.visuals().error_fg_color, "Stream unavailable");
                    ui.label("Check status panel for errors.");
                });
            }
        });

        // Always repaint to keep checking for new frames/data
        ctx.request_repaint();
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // ... (on_exit remains the same, using self.audio_processor_stop_signal_clone) ...
        info!("Exit requested. Stopping threads...");
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);
        self.audio_capture_stop_signal.store(true, Ordering::Relaxed);
        self.audio_processor_stop_signal_clone
            .store(true, Ordering::Relaxed);
        info!("Stop signals sent.");

        if let Some(stream) = self.audio_capture_thread_handle.take() {
            info!("Pausing audio stream...");
            if let Err(e) = stream.pause() { error!("Error pausing audio stream: {}", e); }
            drop(stream);
            info!("Audio stream dropped.");
        } else { info!("Audio stream was already stopped or not initialized."); }

        let join_thread = |name: &str, handle: Option<JoinHandle<()>>| {
            if let Some(h) = handle {
                info!("Joining {} thread...", name);
                if let Err(e) = h.join() { error!("Error joining {} thread: {:?}", name, e); }
                else { info!("{} thread joined successfully.", name); }
            } else { info!("{} thread was already stopped or not initialized.", name); }
        };

        join_thread("Camera", self.cam_thread_handle.take());
        join_thread("Audio Processor", self.audio_processor_thread_handle.take());
        join_thread("Segmentation", self.seg_thread_handle.take());
        info!("All threads stopped and joined.");
    }
}

// Helper trait for centering (Unchanged)
trait CenteredJustified {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui));
}
impl CenteredJustified for egui::Ui {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui)) {
        use egui::{Align, Layout};
        self.with_layout(Layout::top_down(Align::Center), |ui| {
            ui.allocate_space(ui.available_size() / 3.0);
            add_contents(ui);
        });
    }
}