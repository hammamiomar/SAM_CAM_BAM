use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, Sender, TryRecvError},
        Arc,
    },
    thread::{self, JoinHandle},
};

use egui::{ColorImage, ImageData, TextureHandle, TextureOptions, Vec2};
use log::{debug, error, info, warn};
use nokhwa::{
    // --- FIX 1: Correct import name ---
    pixel_format::{RgbFormat, YuyvFormat}, // Use YuyvFormat
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution, // Need Resolution struct
    },
    Camera, NokhwaError,
};

// --- Define desired settings ---
const REQUESTED_WIDTH: u32 = 1280;
const REQUESTED_HEIGHT: u32 = 720;
const REQUESTED_FPS: u32 = 30;

// --- Message from Camera Thread to UI Thread ---
enum CameraThreadMsg {
    Frame(Arc<ColorImage>), // Send decoded RGB image
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
        }
    }
}

// --- Camera Capture Thread Logic ---
fn camera_capture_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
) {
    info!("Camera capture thread started. Requesting YUYV format.");

    // --- Define the Requested Format ---
    // --- FIX 2: Create Resolution struct first ---
    let requested_resolution = Resolution::new(REQUESTED_WIDTH, REQUESTED_HEIGHT);
    let requested_cam_format = CameraFormat::new(
        requested_resolution, // Pass Resolution struct
        FrameFormat::YUYV,
        REQUESTED_FPS,
    );
    // --- Use YuyvFormat pixel type ---
    let requested_format = RequestedFormat::new::<YuyvFormat>(RequestedFormatType::Closest(
        requested_cam_format,
    ));
    info!("Requested camera format: {:?}", requested_format);

    // --- Initialize Camera *inside* the thread ---
    // Try default backend first
    let camera_result = Camera::new(index.clone(), requested_format.clone())
        .or_else(|err| {
            warn!(
                "Default backend failed: {}. Trying AVFoundation explicitly...",
                err
            );
            // --- FIX 3: Use Camera::with_backend ---
            Camera::with_backend(
                index, // Use original index here
                requested_format, // Pass the requested format
                ApiBackend::AVFoundation, // Explicitly specify backend
            )
        });

    // --- Handle Camera Initialization Result ---
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

    // --- Get Actual Camera Format ---
    let camera_format = camera.camera_format();
    let actual_resolution = camera_format.resolution();
    info!("Actual camera format received: {:?}", camera_format);

    // --- Open Camera Stream ---
    if let Err(err) = camera.open_stream() {
        let error_msg = format!("Failed to open camera stream: {}", err);
        error!("{}", error_msg);
        let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
        ctx.request_repaint();
        return;
    }
    info!("Camera stream opened successfully.");

    // --- Frame Capture Loop ---
    while !stop_signal.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => {
                // Frame buffer here is likely YUYV
                match frame.decode_image::<RgbFormat>() { // Decode to RGB
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

    // --- Cleanup ---
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
        self.camera_error = None;

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
                        let available_width = ui.available_width();
                        let image_height = available_width / aspect_ratio;

                        ui.add(
                            egui::Image::new(texture)
                                .max_width(available_width)
                                .max_height(image_height)
                                .maintain_aspect_ratio(true)
                                .rounding(5.0),
                        );
                    } else {
                        ui.label("Texture exists but size unknown.");
                    }
                }
                None if self.camera_error.is_none() => {
                    ui.spinner();
                    ui.label("Initializing camera stream...");
                }
                None => {} // Error message shown above
            }

            ui.separator();

            if let Some(res) = self.camera_resolution {
                ui.label(format!(
                    "Detected Camera Resolution: {}x{}",
                    res.width(),
                    res.height()
                ));
            } else if self.camera_error.is_none() {
                ui.label("Camera Resolution: Waiting...");
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