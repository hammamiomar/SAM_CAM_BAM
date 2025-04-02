// src/camera.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::Sender, // Only Sender needed here
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use egui::ColorImage; use image::{ImageBuffer, Rgb, RgbImage};
// Needed for CameraThreadMsg
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
const REQUESTED_WIDTH: u32 = 1280;
const REQUESTED_HEIGHT: u32 = 720;
const REQUESTED_FPS: u32 = 30;

// --- Message from Camera Thread to UI Thread ---
// Make this public so ui.rs can use it
#[derive(Debug)] // Added Debug for easier printing if needed
pub enum CameraThreadMsg {
    Frame(Arc<RgbImage>), // Send decoded RGB image
    Error(String),
    // Maybe add resolution info here too if desired:
    // ResolutionInfo(Resolution),
}

// --- Public function to start the camera thread ---
pub fn start_camera_thread(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>, // UI gives us the sender
    stop_signal: Arc<AtomicBool>,      // UI gives us the stop signal
    ctx: egui::Context,                // UI gives us the context clone
) -> JoinHandle<()> {
    info!("Spawning camera capture thread.");
    thread::spawn(move || {
        camera_capture_loop(index, msg_sender, stop_signal, ctx);
    })
}

// --- Internal capture loop logic ---
// Renamed from camera_capture_thread to avoid confusion with the public start function
fn camera_capture_loop(
    index: CameraIndex,
    msg_sender: Sender<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
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
    // Could optionally send this back:
    // let _ = msg_sender.send(CameraThreadMsg::ResolutionInfo(camera_format.resolution()));
    info!("Actual camera format received: {:?}", camera_format);

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
                match frame.decode_image::<RgbFormat>() {
                    Ok(decoded_rgb_image) => {
                        //let resolution = [
                        //    decoded_rgb_image.width() as usize,
                        //    decoded_rgb_image.height() as usize,
                        //];
                        //let buffer: Vec<u8> = decoded_rgb_image.into_raw();
                        //TODO SEND IT AS IMAGEBUFFER AND NOT EPAINT COLORIMAGE
                        //let color_image = ColorImage::from_rgb(resolution, &buffer);

                        // Send pre-processed frame
                        let rgb_image:RgbImage = decoded_rgb_image;
                        if msg_sender
                            .send(CameraThreadMsg::Frame(Arc::new(rgb_image)))
                            .is_ok()
                        {
                            ctx.request_repaint();
                        } else {
                            info!("UI thread receiver disconnected. Stopping camera loop.");
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
                            info!("UI thread receiver disconnected after capture error. Stopping camera loop.");
                            break;
                        }
                        ctx.request_repaint();
                        thread::sleep(std::time::Duration::from_secs(1));
                    }
                }
            }
        }
    }
    info!("Camera capture loop stopping signal received.");
    if let Err(e) = camera.stop_stream() {
        error!("Failed to stop camera stream cleanly: {}", e);
    }
    info!("Camera capture loop finished.");
}