// src/camera.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::TrySendError as StdTrySendError, // Keep std error for reference if needed
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration, // Keep Duration
};

// --- Use crossbeam_channel ---
use crossbeam_channel::{Sender, SendError}; // Only need Sender and SendError here
// --- End crossbeam_channel ---

// Need RgbImage for the message type payload
use image::RgbImage;
use log::{error, info, warn};
use nokhwa::{
    pixel_format::{RgbFormat, YuyvFormat},
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera, NokhwaError,
};

// --- Constants specific to camera configuration ---
// --- Using Lowered Resolution ---
const REQUESTED_WIDTH: u32 = 640;
const REQUESTED_HEIGHT: u32 = 480;
const REQUESTED_FPS: u32 = 30;

// --- Message from Camera Thread to Segmentation Thread ---
#[derive(Debug)]
pub enum CameraThreadMsg {
    Frame(Arc<RgbImage>), // Send RgbImage
    Error(String),
}

// --- Public function signature uses crossbeam Sender ---
pub fn start_camera_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>, // crossbeam Sender
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context, // Keep context for potential error reporting repaints
) -> JoinHandle<()> {
    info!("Spawning camera capture thread.");
    thread::spawn(move || {
        camera_capture_loop(index, msg_sender, stop_signal, ctx);
    })
}

// --- Internal loop logic uses crossbeam Sender ---
fn camera_capture_loop(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>, // crossbeam Sender
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context, // For error repaints
) {
    info!("Camera capture loop started. Requesting YUYV format.");
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

    // --- Initialize Camera (remains the same) ---
    let camera_result = Camera::new(index.clone(), requested_format.clone())
        .or_else(|err| {
            warn!("Default backend failed: {}. Trying AVFoundation explicitly...", err);
            Camera::with_backend(index, requested_format, ApiBackend::AVFoundation)
        });

    let mut camera = match camera_result {
        Ok(cam) => { info!("Camera initialized successfully."); cam }
        Err(err) => {
            let error_msg = format!("Failed to open camera: {}", err);
            error!("{}", error_msg);
            // Send error and request repaint *from this thread* for initialization errors
            let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
            ctx.request_repaint();
            return;
        }
    };

    let camera_format = camera.camera_format();
    info!("Actual camera format received: {:?}", camera_format);
    if let Err(err) = camera.open_stream() {
        let error_msg = format!("Failed to open stream: {}", err);
        error!("{}", error_msg);
        let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
        ctx.request_repaint(); // Repaint needed for stream open error
        return;
    }
    info!("Camera stream opened successfully.");

    // --- Frame Capture Loop ---
    while !stop_signal.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => {
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded_rgb_image) => {
                        let frame_arc = Arc::new(decoded_rgb_image);
                        // --- Use crossbeam blocking send (since channel is unbounded) ---
                        // This will only fail if the receiver is disconnected.
                        if let Err(SendError(_)) = msg_sender.send(CameraThreadMsg::Frame(frame_arc)) {
                            info!("Segmentation thread receiver disconnected. Stopping camera loop.");
                            break; // Exit loop if receiver is gone
                        }
                        // No need to request repaint on success, seg thread does that
                    }
                    Err(err) => {
                        warn!("Failed to decode frame to RGB: {}", err);
                        // Consider sending error?
                        // let _ = msg_sender.send(CameraThreadMsg::Error(format!("Decode Error: {}", err)));
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
                        // Send critical capture errors
                        if let Err(SendError(_)) = msg_sender.send(CameraThreadMsg::Error(error_msg)) {
                            info!("Segmentation thread receiver disconnected after capture error.");
                            break;
                        }
                        // Maybe request repaint for critical capture errors?
                        // ctx.request_repaint();
                        thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }
        }
    }
    // --- Cleanup ---
    info!("Camera capture loop stopping signal received.");
    if let Err(e) = camera.stop_stream() { error!("Failed to stop camera stream cleanly: {}", e); }
    info!("Camera capture loop finished.");
}