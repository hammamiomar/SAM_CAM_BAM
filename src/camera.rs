// src/camera.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
use crossbeam_channel::{SendError, Sender};
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

// --- Constants ---
const REQUESTED_WIDTH: u32 = 640;
const REQUESTED_HEIGHT: u32 = 480;
const REQUESTED_FPS: u32 = 30;

#[derive(Debug)]
pub enum CameraThreadMsg {
    Frame(Arc<RgbImage>),
    Error(String),
}

pub fn start_camera_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>, 
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
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
    ctx: egui::Context,
) {
    info!("Camera capture loop started. Requesting YUYV format.");
    let requested_resolution = Resolution::new(REQUESTED_WIDTH, REQUESTED_HEIGHT);
    let requested_cam_format =
        CameraFormat::new(requested_resolution, FrameFormat::YUYV, REQUESTED_FPS);
    let requested_format =
        RequestedFormat::new::<YuyvFormat>(RequestedFormatType::Closest(requested_cam_format));
    info!("Requested camera format: {:?}", requested_format);

    // --- Initialize Camera ---
    let camera_result = Camera::new(index.clone(), requested_format).or_else(|err| {
        warn!(
            "Default backend failed: {}. Trying AVFoundation explicitly...",
            err
        );
        Camera::with_backend(index, requested_format, ApiBackend::AVFoundation)
    });

    let mut camera = match camera_result {
        Ok(cam) => {
            info!("Camera initialized successfully.");
            cam
        }
        Err(err) => {
            let error_msg = format!("Failed to open camera: {}", err);
            error!("{}", error_msg);
            let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
            ctx.request_repaint();
            return;
        }
    };

    let camera_format = camera.camera_format();
    info!("Actual camera format received: {:?}", camera_format);
    info!("Camera Description{:?}", camera.info().description());
    if let Err(err) = camera.open_stream() {
        let error_msg = format!("Failed to open stream: {}", err);
        error!("{}", error_msg);
        let _ = msg_sender.send(CameraThreadMsg::Error(error_msg));
        ctx.request_repaint();
        return;
    }
    info!("Camera stream opened successfully.");

    // --- Frame Capture Loop ---
    while !stop_signal.load(Ordering::Relaxed) {
        match camera.frame() {
            Ok(frame) => match frame.decode_image::<RgbFormat>() {
                Ok(decoded_rgb_image) => {
                    let frame_arc = Arc::new(decoded_rgb_image);
                    if let Err(SendError(_)) = msg_sender.send(CameraThreadMsg::Frame(frame_arc)) {
                        info!("Segmentation thread receiver disconnected. Stopping camera loop.");
                        break;
                    }
                }
                Err(err) => {
                    warn!("Failed to decode frame to RGB: {}", err);
                    thread::sleep(std::time::Duration::from_millis(50));
                }
            },
            Err(err) => match err {
                NokhwaError::ReadFrameError(msg) if msg.contains("Timeout") => {
                    warn!("Camera frame read timeout.");
                    thread::sleep(std::time::Duration::from_millis(100));
                }
                _ => {
                    let error_msg = format!("Failed to capture frame: {}", err);
                    error!("{}", error_msg);
                    if let Err(SendError(_)) = msg_sender.send(CameraThreadMsg::Error(error_msg)) {
                        info!("Segmentation thread receiver disconnected after capture error.");
                        break;
                    }
                    thread::sleep(std::time::Duration::from_secs(1));
                }
            },
        }
    }
    // --- Cleanup ---
    info!("Camera capture loop stopping signal received.");
    if let Err(e) = camera.stop_stream() {
        error!("Failed to stop camera stream cleanly: {}", e);
    }
    info!("Camera capture loop finished.");
}
