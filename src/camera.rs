// src/camera.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender,
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};
// *** CHANGED: Removed egui::ColorImage import ***
use image::RgbImage; // Send RgbImage now
use log::{error, info, warn};
use nokhwa::{
    pixel_format::{RgbFormat, YuyvFormat},
    utils::{
        ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
        Resolution,
    },
    Camera, NokhwaError,
};

const REQUESTED_WIDTH: u32 = 640;
const REQUESTED_HEIGHT: u32 = 480;
const REQUESTED_FPS: u32 = 30;

// --- Message from Camera Thread to Segmentation Thread ---
#[derive(Debug)]
pub enum CameraThreadMsg {
    Frame(Arc<RgbImage>), // *** CHANGED: Send RgbImage ***
    Error(String),
}

pub fn start_camera_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context, // Context still needed for repaint requests from *this* thread if errors occur
) -> JoinHandle<()> {
    info!("Spawning camera capture thread.");
    thread::spawn(move || {
        camera_capture_loop(index, msg_sender, stop_signal, ctx);
    })
}

fn camera_capture_loop(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context, // Keep ctx for error repaints
) {
    // --- Camera Initialization (remains the same) ---
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
    let camera_result = Camera::new(index.clone(), requested_format.clone())
        .or_else(|err| {
            warn!(
                "Default backend failed: {}. Trying AVFoundation explicitly...",
                err
            );
            Camera::with_backend(index, requested_format, ApiBackend::AVFoundation)
        });
    let mut camera = match camera_result {
        Ok(cam) => { info!("Camera initialized successfully."); cam }
        Err(err) => { /* ... error handling sends msg, ctx.request_repaint(), returns ... */
            let error_msg = format!("Failed to open camera: {}", err);
            error!("{}", error_msg);
            let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
            ctx.request_repaint(); // Repaint needed to show camera init error
            return;
        }
    };
    let camera_format = camera.camera_format();
    info!("Actual camera format received: {:?}", camera_format);
    if let Err(err) = camera.open_stream() { /* ... error handling sends msg, ctx.request_repaint(), returns ... */
        let error_msg = format!("Failed to open stream: {}", err);
        error!("{}", error_msg);
        let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
        ctx.request_repaint(); // Repaint needed to show stream open error
        return;
    }
    info!("Camera stream opened successfully.");

    // --- Frame Capture Loop ---
    while !stop_signal.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => {
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded_rgb_image) => {
                        // decoded_rgb_image is already an RgbImage
                        if msg_sender
                            .send(CameraThreadMsg::Frame(Arc::new(decoded_rgb_image))) // Send Arc<RgbImage>
                            .is_ok()
                        {
                            // No repaint request needed here, segmentation thread will do it
                        } else {
                            info!("Segmentation thread receiver disconnected. Stopping camera loop.");
                            break;
                        }
                    }
                    Err(err) => { /* ... error handling ... */
                        warn!("Failed to decode frame to RGB: {}", err);
                        thread::sleep(std::time::Duration::from_millis(50));
                    }
                }
            }
            Err(err) => { /* ... error handling, send msg, maybe ctx.request_repaint() on critical error ... */
                match err {
                    NokhwaError::ReadFrameError(msg) if msg.contains("Timeout") => {
                        warn!("Camera frame read timeout.");
                        thread::sleep(std::time::Duration::from_millis(100));
                    }
                    _ => {
                        let error_msg = format!("Failed to capture frame: {}", err);
                        error!("{}", error_msg);
                        // Send error, segmentation thread might relay it or UI can handle CameraThreadMsg::Error
                        if msg_sender.send(CameraThreadMsg::Error(error_msg)).is_err() {
                            info!("Segmentation thread receiver disconnected after capture error.");
                            break;
                        }
                        // Maybe request repaint if capture totally fails?
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