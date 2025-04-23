// src/camera.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};
// Use TrySendError for frame dropping
use crossbeam_channel::{Sender, TrySendError}; // Removed SendError, added TrySendError
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
pub enum CameraToSegMsg { // Renamed for clarity
    Frame(Arc<RgbImage>),
    Error(String),
}

#[derive(Debug)]
pub enum CameraToUiMsg { // New message type for direct UI feed
    Frame(Arc<RgbImage>),
    Error(String),
}


pub fn start_camera_thread(
    index: CameraIndex,
    // Two senders now
    seg_sender: Sender<CameraToSegMsg>,
    ui_sender: Sender<CameraToUiMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
) -> JoinHandle<()> {
    info!("Spawning camera capture thread.");
    thread::spawn(move || {
        camera_capture_loop(index, seg_sender, ui_sender, stop_signal, ctx);
    })
}

fn camera_capture_loop(
    index: CameraIndex,
    // Two senders now
    seg_sender: Sender<CameraToSegMsg>,
    ui_sender: Sender<CameraToUiMsg>,
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
            // Send error to both consumers
            let _ = seg_sender.try_send(CameraToSegMsg::Error(error_msg.clone()));
            let _ = ui_sender.try_send(CameraToUiMsg::Error(error_msg));
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
         // Send error to both consumers
        let _ = seg_sender.try_send(CameraToSegMsg::Error(error_msg.clone()));
        let _ = ui_sender.try_send(CameraToUiMsg::Error(error_msg));
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

                    // --- Send to Segmentation Thread (non-blocking) ---
                    match seg_sender.try_send(CameraToSegMsg::Frame(frame_arc.clone())) {
                        Ok(_) => {} // Sent successfully
                        Err(TrySendError::Full(_)) => {
                             warn!("Segmentation channel full. Dropping frame.");
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            info!("Segmentation thread receiver disconnected. Stopping camera loop.");
                            break; // Stop if seg thread is gone
                        }
                    }

                    // --- Send to UI Thread (non-blocking) ---
                     match ui_sender.try_send(CameraToUiMsg::Frame(frame_arc)) { // Send the original arc here
                        Ok(_) => {
                             ctx.request_repaint(); // Request UI repaint on new frame
                        }
                        Err(TrySendError::Full(_)) => {
                            // Don't warn excessively here, UI might skip frames intentionally
                            // warn!("UI channel full. Dropping frame for UI.");
                        }
                        Err(TrySendError::Disconnected(_)) => {
                            info!("UI thread receiver disconnected. Stopping camera loop.");
                            break; // Stop if UI is gone
                        }
                    }
                }
                Err(err) => {
                    warn!("Failed to decode frame to RGB: {}", err);
                    // Maybe send decode error?
                    // let _ = seg_sender.try_send(CameraToSegMsg::Error(format!("Decode Error: {}", err)));
                    // let _ = ui_sender.try_send(CameraToUiMsg::Error(format!("Decode Error: {}", err)));
                    thread::sleep(std::time::Duration::from_millis(50));
                }
            },
            Err(err) => {
                // Handle specific errors like timeout
                if let NokhwaError::ReadFrameError(msg) = &err {
                     if msg.contains("Timeout") {
                         warn!("Camera frame read timeout.");
                         thread::sleep(std::time::Duration::from_millis(100));
                         continue; // Don't send error for timeout, just retry
                     }
                }
                // Send other errors
                 let error_msg = format!("Failed to capture frame: {}", err);
                 error!("{}", error_msg);
                 let _ = seg_sender.try_send(CameraToSegMsg::Error(error_msg.clone()));
                 let _ = ui_sender.try_send(CameraToUiMsg::Error(error_msg));
                 // Consider breaking or pausing on persistent errors
                 thread::sleep(std::time::Duration::from_secs(1));
            }
        }
    }
    // --- Cleanup ---
    info!("Camera capture loop stopping signal received.");
    if let Err(e) = camera.stop_stream() {
        error!("Failed to stop camera stream cleanly: {}", e);
    }
    info!("Camera capture loop finished.");
}
