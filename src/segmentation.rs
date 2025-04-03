// src/segmentation.rs
use std::{
    sync::{atomic::{AtomicBool, Ordering}, mpsc::{Receiver, RecvTimeoutError, Sender}, Arc},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use egui::ColorImage;
use image::{DynamicImage, Rgb, RgbImage}; // Keep RgbImage for drawing
use imageproc::drawing; // <--- Import drawing functions
use imageproc::rect::Rect; // <--- Import Rect for drawing
use log::{debug, error, info, warn};

// --- Corrected usls imports ---
use usls::{
    models::YOLO,
    Options, Y, Ys, // Use Y and Ys for output
    // Mask, Polygon, Bbox are part of Y struct now
};

use crate::camera::CameraThreadMsg;

#[derive(Debug)]
pub enum SegmentationThreadMsg {
    Frame(Arc<ColorImage>),
    Error(String),
}

pub fn start_segmentation_thread(
    ui_sender: Sender<SegmentationThreadMsg>,
    camera_receiver: Receiver<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) -> JoinHandle<()> {
    info!("Spawning segmentation thread");
    thread::spawn(move || {
        segmentation_loop(ui_sender, camera_receiver, stop_signal, ctx, model_options);
    })
}

fn segmentation_loop(
    ui_sender: Sender<SegmentationThreadMsg>,
    camera_receiver: Receiver<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) {
    info!("Segmentation loop started.");

    // --- Initialize Model ---
    let model_result = YOLO::new(model_options.clone()); // Use passed options
    let mut model = match model_result {
        Ok(m) => { info!("Segmentation model loaded successfully."); m },
        Err(e) => { /* ... error handling ... */
            let error_msg = format!("Failed to load model: {}", e);
            error!("{}", error_msg); let _ = ui_sender.send(SegmentationThreadMsg::Error(error_msg)); ctx.request_repaint(); return;
        }
    };

    // --- Store needed parameters locally ---
    let class_names = model.names().to_vec(); // Get names from the initialized model

    let mut segmentation_time = Duration::from_secs(0);

    while !stop_signal.load(Ordering::Relaxed) {
        match camera_receiver.recv_timeout(Duration::from_millis(100)) {
            Ok(camera_msg) => {
                match camera_msg {
                    CameraThreadMsg::Frame(frame_arc) => {
                        let loop_start_time = Instant::now();
                        let mut display_image = (*frame_arc).clone(); // Clone for drawing
                        let dynamic_image_input = DynamicImage::ImageRgb8(display_image.clone());

                        // --- Run Inference + Postprocessing ---
                        let seg_start_time = Instant::now();
                        let results = model.forward(&[dynamic_image_input]);
                        segmentation_time = seg_start_time.elapsed();

                        // --- Process Results ---
                        match results {
                            Ok(ys) => {
                                if let Some(y) = ys.get(0) {
                                    // --- Draw Bounding Boxes ---
                                    if let Some(bboxes) = y.bboxes() {
                                        let bbox_color = Rgb([0u8, 255u8, 0u8]); // Green for boxes
                                        for bbox in bboxes {
                                            // Create a Rect for imageproc
                                            // Ensure coordinates are within bounds and cast safely
                                            let x = bbox.xmin().round() as i32;
                                            let y = bbox.ymin().round() as i32;
                                            let width = bbox.width().round() as u32;
                                            let height = bbox.height().round() as u32;

                                            // Prevent drawing outside image bounds (important for hollow rect)
                                            let rect = Rect::at(x, y).of_size(width, height);

                                            drawing::draw_hollow_rect_mut(&mut display_image, rect, bbox_color);

                                            // --- Optional: Draw Label ---
                                            // let label = bbox.label(true, true, 2); // name: conf
                                            // if !label.is_empty() {
                                            //    // Need font loading for drawing text (more complex)
                                            //    // drawing::draw_text_mut(&mut display_image, Rgb([0u8, 255u8, 0u8]), x, y - 10, scale, &font, &label);
                                            // }
                                        }
                                    }
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

                        // --- Convert final image and Send ---
                        let final_color_image = {
                            let size = [display_image.width() as usize, display_image.height() as usize];
                            let pixels = display_image.into_raw();
                            ColorImage::from_rgb(size, &pixels)
                        };
                        if ui_sender.send(SegmentationThreadMsg::Frame(Arc::new(final_color_image))).is_ok() {
                            ctx.request_repaint();
                        } else {
                            info!("UI receiver disconnected."); break;
                        }
                        debug!("Segmentation processed frame in {:?}, model time: {:?}", loop_start_time.elapsed(), segmentation_time);
                    },
                    CameraThreadMsg::Error(err) => { /* ... relay error ... */
                         warn!("Received error from camera thread: {}", err);
                         let _ = ui_sender.send(SegmentationThreadMsg::Error(format!("Camera Error: {}", err)));
                         ctx.request_repaint();
                    }
                }
            }
            Err(RecvTimeoutError::Timeout) => { continue; }
            Err(RecvTimeoutError::Disconnected) => { /* ... handle disconnect ... */
                 error!("Camera disconnected."); let _ = ui_sender.send(SegmentationThreadMsg::Error("Camera disconnected.".to_string())); ctx.request_repaint(); break;
            }
        }
    }
    info!("Segmentation loop finishing.");
}