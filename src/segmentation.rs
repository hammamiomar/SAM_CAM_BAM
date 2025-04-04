// src/segmentation.rs
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, // Keep Arc
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// --- Use crossbeam_channel ---
use crossbeam_channel::{Receiver, Sender, TryRecvError, TrySendError};
// --- End crossbeam_channel ---

use egui::ColorImage;
use image::{DynamicImage, Pixel, Rgb, RgbImage};
use imageproc::drawing; // Import drawing functions
use imageproc::rect::Rect; // Import Rect for drawing
use log::{debug, error, info, warn};

// --- Corrected usls imports ---
use usls::{
    models::YOLO,
    Options, Y, Ys, // Use Y and Ys for output
};

// Message type RECEIVING from camera thread
use crate::camera::CameraThreadMsg; // Make sure this path is correct

// Message type SENDING to UI thread
#[derive(Debug)]
pub enum SegmentationThreadMsg {
    Frame(Arc<ColorImage>), // Send image ready for egui
    Error(String),
}

// --- Updated function signature with crossbeam types ---
pub fn start_segmentation_thread(
    ui_sender: Sender<SegmentationThreadMsg>,   // crossbeam Sender
    camera_receiver: Receiver<CameraThreadMsg>, // crossbeam Receiver
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options, // Receive configured options
) -> JoinHandle<()> {
    info!("Spawning segmentation thread");
    thread::spawn(move || {
        segmentation_loop(ui_sender, camera_receiver, stop_signal, ctx, model_options);
    })
}

// --- Updated function signature with crossbeam types ---
fn segmentation_loop(
    ui_sender: Sender<SegmentationThreadMsg>,   // crossbeam Sender
    camera_receiver: Receiver<CameraThreadMsg>, // crossbeam Receiver
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) {
    info!("Segmentation loop started.");

    // --- Initialize Model ---
    let model_result = YOLO::new(model_options.clone()); // Use passed options
    let mut model = match model_result {
        Ok(m) => { info!("Segmentation model loaded successfully."); m },
        Err(e) => {
            let error_msg = format!("Failed to load model: {}", e);
            error!("{}", error_msg); let _ = ui_sender.send(SegmentationThreadMsg::Error(error_msg)); ctx.request_repaint(); return;
        }
    };

    // --- Get class names locally if needed for drawing labels later ---
    // let class_names = model.names().to_vec();

    let mut segmentation_time = Duration::from_secs(0);

    // --- Main Segmentation Loop ---
    while !stop_signal.load(Ordering::Relaxed) {

        // --- Receive LATEST Frame from Camera Thread ---
        let mut latest_frame_arc: Option<Arc<RgbImage>> = None;
        loop {
            match camera_receiver.try_recv() { // Use crossbeam try_recv
                Ok(CameraThreadMsg::Frame(frame)) => {
                    latest_frame_arc = Some(frame); // Overwrite with the latest frame
                }
                Ok(CameraThreadMsg::Error(err)) => {
                    warn!("Received error from camera thread: {}", err);
                    // Relay important errors?
                    let _ = ui_sender.send(SegmentationThreadMsg::Error(format!("Camera Error: {}", err)));
                    ctx.request_repaint();
                }
                Err(TryRecvError::Empty) => {
                    // No more frames waiting in the channel right now
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Camera thread disconnected. Stopping segmentation loop.");
                    // Send error to UI
                    let _ = ui_sender.send(SegmentationThreadMsg::Error("Camera thread disconnected.".to_string()));
                    ctx.request_repaint();
                    stop_signal.store(true, Ordering::Relaxed); // Signal self to stop
                    break; // Exit the inner receive loop
                }
            }
        }

        // Check if stop signal was set during receive loop
        if stop_signal.load(Ordering::Relaxed) {
            break; // Exit the main while loop
        }

        // --- Process the latest frame if one was received ---
        if let Some(frame_arc) = latest_frame_arc {
            let loop_start_time = Instant::now();

            // --- 1. Clone RgbImage for drawing target ---
            let mut display_image = (*frame_arc).clone();
            let (orig_width, orig_height) = display_image.dimensions();

            // --- 2. Convert to DynamicImage for model input ---
            let dynamic_image_input = DynamicImage::ImageRgb8(display_image.clone());

            // --- 3. Run Inference + Postprocessing (within model.forward) ---
            let seg_start_time = Instant::now();
            let results = model.forward(&[dynamic_image_input]);
            segmentation_time = seg_start_time.elapsed();

            // --- 4. Draw results onto display_image ---
            match results {
                Ok(ys) => {
                    if let Some(y) = ys.get(0) { // Assuming batch size 1
                        // Draw bounding boxes if they exist
                        if let Some(bboxes) = y.bboxes() {
                            let bbox_color = Rgb([0u8, 255u8, 0u8]); // Green
                            for bbox in bboxes {
                                let x = bbox.xmin().round() as i32;
                                let y = bbox.ymin().round() as i32;
                                let width = bbox.width().round() as u32;
                                let height = bbox.height().round() as u32;

                                // Create Rect for drawing
                                let rect = Rect::at(x, y).of_size(width, height);
                                // Draw hollow rectangle on the mutable display_image
                                drawing::draw_hollow_rect_mut(&mut display_image, rect, bbox_color);
                            }
                        }
                        // Can add Polygon/Mask drawing here later if needed
                    } else {
                        warn!("Model output Ys was empty.");
                    }
                }
                Err(e) => {
                    warn!("Model forward pass failed: {}", e);
                    let _ = ui_sender.send(SegmentationThreadMsg::Error(format!("Segmentation failed: {}", e)));
                    ctx.request_repaint();
                }
            };
            // display_image now contains the original image + drawn boxes

            // --- 5. Convert final image to egui::ColorImage ---
            let final_color_image = {
                let size = [display_image.width() as usize, display_image.height() as usize];
                let pixels = display_image.into_raw();
                ColorImage::from_rgb(size, &pixels)
            };

            // --- 6. Try Send to UI Thread (using crossbeam try_send) ---
            match ui_sender.try_send(SegmentationThreadMsg::Frame(Arc::new(final_color_image))) {
                Ok(_) => {
                    ctx.request_repaint(); // Signal UI only if send succeeds
                }
                Err(TrySendError::Full(_)) => {
                    warn!("UI channel full. Dropping segmented frame."); // Frame dropped
                }
                Err(TrySendError::Disconnected(_)) => {
                    info!("UI receiver disconnected. Stopping segmentation loop.");
                    break; // Exit main while loop
                }
            }
            debug!("Segmentation processed frame in {:?}, model time: {:?}", loop_start_time.elapsed(), segmentation_time);

        } else {
            // No new frame was available from the camera thread
            thread::sleep(Duration::from_millis(5)); // Yield CPU
        }
    } // End main while loop

    info!("Segmentation loop finishing (stop signal received or channel disconnected).");
}