// src/segmentation.rs
use crossbeam_channel::{
    Receiver as CrossbeamReceiver, Sender as CrossbeamSender, TryRecvError, TrySendError,
};
use egui::ColorImage;
use image::{DynamicImage, RgbImage};
use imageproc::{rect::Rect};
use log::{debug, error, info, warn};
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;
use std::{
    collections::{HashMap, HashSet}, 
    f32::consts::PI,                 
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use usls::{models::YOLO, Bbox, Nms, Options};

use crate::camera::CameraThreadMsg;
use crate::visuals;
#[derive(Debug, Clone)]
pub enum UserInteractionSegMsg {}

#[derive(Debug)]
pub enum SegmentationThreadMsg {
    Frame(Arc<ColorImage>),
    Error(String),
}

// --- Constants ---
pub const MAX_TRACKS: usize = 3; // Still represents Bass, Mid, High bands
const IOU_THRESHOLD: f32 = 0.3; // Threshold for matching track

// --- TrackedObject Struct --- (Persistent Assignment Version) ---
#[derive(Debug, Clone)]
struct TrackedObject {
    bbox: Bbox,           // Bbox from the *last known* frame it was seen in
    band_index: usize,    // 0, 1, or 2 (Bass, Mid, High) - Persists for the object's lifetime
    animation_phase: f32, // For visual effects
}

pub fn start_segmentation_thread(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    _user_interaction_receiver: CrossbeamReceiver<UserInteractionSegMsg>,
    intensity_receiver: CrossbeamReceiver<Vec<f32>>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) -> JoinHandle<()> {
    info!("Spawning segmentation thread (Persistent Random Assignment - Individual Viz)");
    thread::spawn(move || {
        segmentation_loop(
            ui_sender,
            camera_receiver,
            intensity_receiver,
            stop_signal,
            ctx,
            model_options,
        );
    })
}

fn segmentation_loop(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    intensity_receiver: CrossbeamReceiver<Vec<f32>>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) {
    info!("Segmentation loop started (Persistent Random Assignment - Individual Viz).");

    let mut model = match YOLO::new(model_options) {
        Ok(m) => m,
        Err(e) => {
            let emsg = format!("Model load failed: {}", e);
            error!("{}", emsg);
            let _ = ui_sender.send(SegmentationThreadMsg::Error(emsg));
            ctx.request_repaint();
            return;
        }
    };

    let mut tracked_objects: Vec<TrackedObject> = Vec::new();
    let mut processing_time = Duration::from_secs(0);
    let mut current_band_intensities = vec![0.0f32; MAX_TRACKS];
    let mut rng = SmallRng::from_rng(&mut rand::thread_rng()); 
    let mut frame_count: u64 = 0;

    while !stop_signal.load(Ordering::Relaxed) {
        frame_count += 1;
        // Interactions Removed
        // Receive Frame
        let mut latest_frame_arc: Option<Arc<RgbImage>> = None;
        loop {
            match camera_receiver.try_recv() {
                Ok(CameraThreadMsg::Frame(f)) => {
                    latest_frame_arc = Some(f);
                }
                Ok(CameraThreadMsg::Error(e)) => {
                    warn!("Cam Err: {}", e);
                }
                Err(TryRecvError::Empty) => {
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Cam disconnected.");
                    stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }
        // Receive Intensities
        loop {
            match intensity_receiver.try_recv() {
                Ok(i) => {
                    if i.len() >= MAX_TRACKS {
                        current_band_intensities.copy_from_slice(&i[0..MAX_TRACKS]);
                    } else {
                        current_band_intensities.fill(0.0);
                        current_band_intensities[0..i.len()].copy_from_slice(&i);
                    }
                }
                Err(TryRecvError::Empty) => {
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Audio proc disconnected.");
                    stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        if stop_signal.load(Ordering::Relaxed) {
            break;
        }

        if let Some(frame_arc) = latest_frame_arc {
            let loop_start_time = Instant::now();
            let _original_image = (*frame_arc).clone();
            let mut display_image = (*frame_arc).clone();
            let (_frame_w, _frame_h) = display_image.dimensions(); // Use _ if not needed
            let dynamic_image_input = DynamicImage::ImageRgb8((*frame_arc).clone());

            let proc_start = Instant::now();
            let results = model.forward(&[dynamic_image_input]);
            processing_time = proc_start.elapsed();

            // Store mapping from CURRENT detection index to relevant info for drawing
            // Value: (band_idx, animation_phase)
            let mut current_detection_info: HashMap<usize, (usize, f32)> = HashMap::new();
            let mut next_tracked_objects: Vec<TrackedObject> = Vec::new();

            match results {
                Ok(ys) => {
                    if let Some(y) = ys.first() {
                        let current_bboxes = y.bboxes().unwrap_or_default();
                        let current_masks = y.masks().unwrap_or_default();

                        // Selection Logic Removed

                        // Match Existing Tracks
                        let mut matched_current_indices: HashSet<usize> = HashSet::new();
                        for tracked_obj in tracked_objects.iter() {
                            let mut best_match_for_this_track: Option<(usize, f32)> = None;
                            for (det_idx, current_bbox) in current_bboxes.iter().enumerate() {
                                if matched_current_indices.contains(&det_idx) {
                                    continue;
                                }
                                let iou = tracked_obj.bbox.iou(current_bbox);
                                if iou > IOU_THRESHOLD {
                                    let is_better = match best_match_for_this_track {
                                        Some((_, cur_iou)) => iou > cur_iou,
                                        None => true,
                                    };
                                    if is_better {
                                        best_match_for_this_track = Some((det_idx, iou));
                                    }
                                }
                            }
                            if let Some((matched_det_idx, _iou)) = best_match_for_this_track {
                                let updated_obj = TrackedObject {
                                    bbox: current_bboxes[matched_det_idx].clone(),
                                    band_index: tracked_obj.band_index,
                                    animation_phase: tracked_obj.animation_phase
                                        + 0.05
                                        + current_band_intensities[tracked_obj.band_index] * 0.1,
                                };
                                // Store info needed for drawing THIS frame
                                current_detection_info.insert(
                                    matched_det_idx,
                                    (updated_obj.band_index, updated_obj.animation_phase),
                                );
                                next_tracked_objects.push(updated_obj); // Add to list for NEXT frame
                                matched_current_indices.insert(matched_det_idx);
                            }
                        }

                        // Assign New Detections
                        for det_idx in 0..current_masks.len() {
                            if !matched_current_indices.contains(&det_idx) {
                                let assigned_band = rng.gen_range(0..MAX_TRACKS);
                                let new_obj = TrackedObject {
                                    bbox: current_bboxes[det_idx].clone(),
                                    band_index: assigned_band,
                                    animation_phase: rng.gen::<f32>() * 2.0 * PI,
                                };
                                // Store info needed for drawing THIS frame
                                current_detection_info
                                    .insert(det_idx, (new_obj.band_index, new_obj.animation_phase));
                                next_tracked_objects.push(new_obj); // Add to list for NEXT frame
                            }
                        }

                        // Update tracked objects state for the *NEXT* frame
                        tracked_objects = next_tracked_objects;

                        // Combined Canvas Logic Removed

                        // Drawing Logic (Visualize ALL *currently detected* and assigned objects)
                        for (det_idx, (band_idx, anim_phase)) in &current_detection_info {
                            // Get required data (mask, bbox, intensity) using det_idx
                            if let (Some(mask_to_draw), Some(bbox_to_draw)) =
                                (current_masks.get(*det_idx), current_bboxes.get(*det_idx))
                            {
                                let intensity = current_band_intensities[*band_idx];
                                let mask_image = mask_to_draw.mask();
                                let bbox_rect = Rect::at(
                                    bbox_to_draw.xmin() as i32,
                                    bbox_to_draw.ymin() as i32,
                                )
                                .of_size(
                                    bbox_to_draw.width().max(1.0) as u32,
                                    bbox_to_draw.height().max(1.0) as u32,
                                );

                                // Call the visuals drawing function FOR THIS OBJECT
                                visuals::draw_visuals(
                                    &mut display_image,
                                    mask_image,
                                    bbox_rect,
                                    *band_idx,
                                    intensity,
                                    frame_count,
                                    *anim_phase,
                                    &mut rng,
                                );
                            }
                        } // End drawing loop
                    } // End if let Some(y)
                } // End Ok(ys)
                Err(e) => {
                    warn!("FastSAM model forward pass failed: {}", e);
                }
            } // End match results

            // --- Send Final Image to UI ---
            let final_color_image = {
                let size = [
                    display_image.width() as usize,
                    display_image.height() as usize,
                ];
                let pixels = display_image.into_raw();
                ColorImage::from_rgb(size, &pixels)
            };
            match ui_sender.try_send(SegmentationThreadMsg::Frame(Arc::new(final_color_image))) {
                Ok(_) => {
                    ctx.request_repaint();
                }
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => {
                    info!("UI disconnected.");
                    stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
            debug!(
                "Seg loop: {:.2?}, Model: {:.2?}, Tracked: {}, AudioInt: [{:.2}, {:.2}, {:.2}]",
                loop_start_time.elapsed(),
                processing_time,
                tracked_objects.len(),
                current_band_intensities.get(0).cloned().unwrap_or(0.0),
                current_band_intensities.get(1).cloned().unwrap_or(0.0),
                current_band_intensities.get(2).cloned().unwrap_or(0.0)
            );
        } else {
            thread::sleep(Duration::from_millis(5));
        }
    } // End while !stop_signal

    info!("Segmentation loop finishing.");
} 
