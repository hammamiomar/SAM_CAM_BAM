use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
    // --- Add time types ---
    time::{Duration, Instant},
};

use egui::{ColorImage, ImageData, TextureHandle, TextureOptions, Vec2, Align, Layout}; // Added Align, Layout
use log::{debug, error, info, warn};
use nokhwa::{
    pixel_format::{RgbFormat, YuyvFormat},
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera, NokhwaError,
};

// --- Define desired settings ---
const REQUESTED_WIDTH: u32 = 1280;
const REQUESTED_HEIGHT: u32 = 720;
const REQUESTED_FPS: u32 = 30;
// --- How often to update the FPS counter (e.g., every 500ms) ---
const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);


// --- Message from Camera Thread to UI Thread ---
enum CameraThreadMsg {
    Frame(Arc<ColorImage>),
    Error(String),
}

// --- eframe App Struct ---
pub struct WebCamApp {
    texture: Option<TextureHandle>,
    camera_thread_rx: Receiver<CameraThreadMsg>,
    camera_thread_handle: Option<JoinHandle<()>>,
    stop_signal: Arc<AtomicBool>,
    camera_error: Option<String>,
    camera_resolution: Option<Resolution>,
    texture_size: Option<Vec2>,

    // --- Fields for FPS calculation ---
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebCamApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let camera_index = CameraIndex::Index(0);
        let texture = None;
        let (camera_thread_tx, camera_thread_rx) = mpsc::channel();
        let stop_signal = Arc::new(AtomicBool::new(false));
        let stop_signal_clone = stop_signal.clone();
        let egui_ctx = cc.egui_ctx.clone();
        let thread_index = camera_index.clone();

        let camera_thread_handle = Some(thread::spawn(move || {
            camera_capture_thread(
                thread_index,
                camera_thread_tx,
                stop_signal_clone,
                egui_ctx,
            );
        }));

        Self {
            texture,
            camera_thread_rx,
            camera_thread_handle,
            stop_signal,
            camera_error: None,
            camera_resolution: None,
            texture_size: None,
            // --- Initialize FPS fields ---
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
                f32::INFINITY // Avoid division by zero if interval is tiny
            };
            self.frames_since_last_update = 0;
            self.last_fps_update_time = now;
        }
    }
}

// --- Camera Capture Thread Logic (remains the same as before) ---
fn camera_capture_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
) {
    info!("Camera capture thread started. Requesting YUYV format.");
    let requested_resolution = Resolution::new(REQUESTED_WIDTH, REQUESTED_HEIGHT);
    let requested_cam_format = CameraFormat::new(
        requested_resolution,
        FrameFormat::YUYV,
        REQUESTED_FPS,
    );
    let requested_format = RequestedFormat::new::<YuyvFormat>(RequestedFormatType::Closest(
        requested_cam_format,
    ));
    info!("Requested camera format: {:?}", requested_format);

    let camera_result = Camera::new(index.clone(), requested_format.clone())
        .or_else(|err| {
            warn!(
                "Default backend failed: {}. Trying AVFoundation explicitly...",
                err
            );
            Camera::with_backend(
                index,
                requested_format,
                ApiBackend::AVFoundation,
            )
        });

    let mut camera = match camera_result {
        Ok(cam) => {
            info!("Camera initialized successfully.");
            cam
        }
        Err(err) => {
            let error_msg =
                format!("Failed to open camera with AVFoundation backend: {}", err);
            error!("{}", error_msg);
            let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
            ctx.request_repaint();
            return;
        }
    };

    let camera_format = camera.camera_format();
    let actual_resolution = camera_format.resolution();
    info!("Actual camera format received: {:?}", camera_format);

    if let Err(err) = camera.open_stream() {
        let error_msg = format!("Failed to open camera stream: {}", err);
        error!("{}", error_msg);
        let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
        ctx.request_repaint();
        return;
    }
    info!("Camera stream opened successfully.");

    while !stop_signal.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => {
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded_rgb_image) => {
                        let resolution = [
                            decoded_rgb_image.width() as usize,
                            decoded_rgb_image.height() as usize,
                        ];
                        let buffer: Vec<u8> = decoded_rgb_image.into_raw();
                        let color_image = ColorImage::from_rgb(resolution, &buffer);

                        if msg_sender
                            .send(CameraThreadMsg::Frame(Arc::new(color_image)))
                            .is_ok()
                        {
                            ctx.request_repaint();
                        } else {
                            info!("UI thread receiver disconnected. Stopping camera thread.");
                            break;
                        }
                    }
                    Err(err) => {
                        warn!("Failed to decode frame to RGB: {}", err);
                        thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
            Err(err) => {
                match err {
                    NokhwaError::ReadFrameError(msg) if msg.contains("Timeout") => {
                        warn!("Camera frame read timeout.");
                        thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => {
                        let error_msg = format!("Failed to capture frame: {}", err);
                        error!("{}", error_msg);
                        if msg_sender.send(CameraThreadMsg::Error(error_msg)).is_err() {
                            info!("UI thread receiver disconnected after capture error. Stopping camera thread.");
                            break;
                        }
                        ctx.request_repaint();
                        thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }
        }
    }
    info!("Camera capture thread stopping signal received.");
    if let Err(e) = camera.stop_stream() {
        error!("Failed to stop camera stream cleanly: {}", e);
    }
    info!("Camera capture thread finished.");
}


// --- eframe App Implementation ---
impl eframe::App for WebCamApp {
    /// Called each time the UI needs repainting
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // --- Update FPS counter at the start of the frame ---
        self.update_fps_counter();

        self.camera_error = None; // Clear previous error each frame

        // --- Process messages from camera thread (remains the same) ---
        loop {
            match self.camera_thread_rx.try_recv() {
                 Ok(msg) => match msg {
                    CameraThreadMsg::Frame(frame_arc) => {
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
                    CameraThreadMsg::Error(err) => {
                        self.camera_error = Some(err);
                    }
                },
                Err(TryRecvError::Empty) => {
                    break;
                }
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

        // --- Top Panel (Menu Bar) ---
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

        // --- Bottom Panel (FPS Counter) ---
        egui::TopBottomPanel::bottom("bottom_panel")
            .resizable(false) // Optional: prevent resizing
            .show(ctx, |ui| {
                // Align FPS counter to the right
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                     ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                     // Add a little space
                     ui.add_space(10.0);
                     // Add resolution info here as well if desired
                     if let Some(res) = self.camera_resolution {
                         ui.label(format!("Cam Res: {}x{}", res.width(), res.height()));
                     } else if self.camera_error.is_none() {
                         ui.label("Cam Res: ...");
                     }
                });
            });

        // --- Central Panel (Webcam Image) ---
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("SAM_CAM_BAM Stream");
            ui.separator();

            if let Some(err) = &self.camera_error {
                ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
            }

            match &self.texture {
                Some(texture) => {
                    if let Some(tex_size) = self.texture_size {
                        let aspect_ratio = if tex_size.y > 0.0 {
                            tex_size.x / tex_size.y
                        } else {
                            1.0
                        };
                        // Fill available space, calculating height from width and aspect ratio
                        let available_width = ui.available_width();
                        let available_height = ui.available_height(); // Use remaining height
                        let mut image_width = available_width;
                        let mut image_height = available_width / aspect_ratio;

                        // If calculated height exceeds available, scale based on height instead
                        if image_height > available_height {
                            image_height = available_height;
                            image_width = available_height * aspect_ratio;
                        }


                        // Center the image (optional)
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                             ui.add(
                                egui::Image::new(texture)
                                    .max_width(image_width)
                                    .max_height(image_height)
                                    .maintain_aspect_ratio(true)
                                    .rounding(5.0),
                             );
                        });

                    } else {
                        ui.label("Texture exists but size unknown.");
                    }
                }
                None if self.camera_error.is_none() => {
                    // Center the loading indicator
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0); // Add some space above
                        ui.spinner();
                        ui.label("Initializing camera stream...");
                    });
                }
                None => {} // Error message shown above
            }

            // Removed resolution label from here, moved to bottom panel
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