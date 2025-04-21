// src/segmentation.rs
use fnv::FnvHashSet;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use crossbeam_channel::{
    Receiver as CrossbeamReceiver, Sender as CrossbeamSender, TryRecvError, // Correct import
    TrySendError,
};
use egui::ColorImage;
use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb, RgbImage, Luma}; // Keep Luma
use imageproc::{drawing, rect::Rect};
use log::{debug, error, info, warn};
// --- Add Rng ---
use rand::{Rng, SeedableRng}; // Add SeedableRng if needed later
use rand::rngs::SmallRng; // Use a faster RNG

use usls::{
    models::YOLO,
    Options, Bbox, Nms,
};

use crate::{camera::CameraThreadMsg, visuals};


// --- Message types (Unchanged) ---
#[derive(Debug, Clone)]
pub enum UserInteractionSegMsg {
    SelectObject { slot: usize, x: f32, y: f32 },
    RemoveObjectAt { x: f32, y: f32 },
    ClearSelection,
}
#[derive(Debug)]
pub enum SegmentationThreadMsg {
    Frame(Arc<ColorImage>),
    Error(String),
}

// --- Constants (Unchanged) ---
pub const MAX_TRACKS: usize = 3;
const IOU_THRESHOLD: f32 = 0.3;
const SELECTION_IOU_THRESHOLD: f32 = 0.6;
pub const TRACK_COLORS: [Rgb<u8>; MAX_TRACKS] = [
    Rgb([255u8, 0u8, 0u8]),
    Rgb([0u8, 255u8, 0u8]),
    Rgb([0u8, 0u8, 255u8]),
];
// const WHITE: Rgb<u8> = Rgb([255u8, 255u8, 255u8]); // Defined in visuals
// const BLACK: Rgb<u8> = Rgb([0u8, 0u8, 0u8]); // Defined in visuals

// --- TrackedObject Struct (Unchanged) ---
#[derive(Debug, Clone)]
struct TrackedObject {
    bbox: Bbox,
    animation_phase: f32,
}

// --- start_segmentation_thread (Unchanged) ---
pub fn start_segmentation_thread(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    user_interaction_receiver: CrossbeamReceiver<UserInteractionSegMsg>,
    intensity_receiver: CrossbeamReceiver<Vec<f32>>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) -> JoinHandle<()> {
    info!("Spawning segmentation thread (FastSAM Multi-Object + Advanced Viz)");
    thread::spawn(move || {
        segmentation_loop(
            ui_sender, camera_receiver, user_interaction_receiver, intensity_receiver,
            stop_signal, ctx, model_options,
        );
    })
 }


fn segmentation_loop(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    user_interaction_receiver: CrossbeamReceiver<UserInteractionSegMsg>,
    intensity_receiver: CrossbeamReceiver<Vec<f32>>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) {
    info!("Segmentation loop started (FastSAM Multi-Object + Advanced Viz).");

    let mut model = match YOLO::new(model_options) {
        Ok(m) => m,
        Err(e) => { /* Error handling */
            let error_msg = format!("Failed to load FastSAM model: {}", e); error!("{}", error_msg);
            let _ = ui_sender.send(SegmentationThreadMsg::Error(error_msg)); ctx.request_repaint(); return;
        }
    };

    // --- State ---
    let mut tracked_objects: Vec<Option<TrackedObject>> = vec![None; MAX_TRACKS];
    let mut pending_selection: Option<(usize, f32, f32)> = None;
    let mut processing_time = Duration::from_secs(0);
    let mut current_band_intensities = vec![0.0f32; MAX_TRACKS];
    // --- Use a faster, seedable RNG if performance becomes an issue, otherwise thread_rng is fine ---
    let mut rng = SmallRng::from_rng(&mut rand::thread_rng()); // Faster RNG
    // Initialize frame counter to track the number of processed frames
    let mut frame_count: u64 = 0;

    while !stop_signal.load(Ordering::Relaxed) {
        frame_count += 1;
        // --- 1. Receive User Interactions ---
        loop {
             match user_interaction_receiver.try_recv() {
                 Ok(UserInteractionSegMsg::SelectObject { slot, x, y }) => { pending_selection = Some((slot, x, y)); }
                 Ok(UserInteractionSegMsg::RemoveObjectAt { x, y }) => {
                     let mut found_slot = None;
                     for (slot_idx, tracked_opt) in tracked_objects.iter().enumerate() {
                         if let Some(tracked) = tracked_opt { if x >= tracked.bbox.xmin() && x <= tracked.bbox.xmax() && y >= tracked.bbox.ymin() && y <= tracked.bbox.ymax() { found_slot = Some(slot_idx); break; } }
                     }
                     if let Some(slot_idx) = found_slot { tracked_objects[slot_idx] = None; }
                 }
                 Ok(UserInteractionSegMsg::ClearSelection) => { tracked_objects.fill(None); pending_selection = None; }
                 Err(TryRecvError::Empty) => break,
                 Err(TryRecvError::Disconnected) => { warn!("User interaction disconnected."); break; }
             }
        }

        // --- 2. Receive LATEST Frame ---
        let mut latest_frame_arc: Option<Arc<RgbImage>> = None;
        loop {
            match camera_receiver.try_recv() {
                Ok(CameraThreadMsg::Frame(frame)) => { latest_frame_arc = Some(frame); }
                Ok(CameraThreadMsg::Error(err)) => { warn!("Camera Error: {}", err); }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => { error!("Camera disconnected."); stop_signal.store(true, Ordering::Relaxed); break; }
            }
        }

        // --- 3. Receive LATEST Audio Intensities ---
         loop {
             match intensity_receiver.try_recv() {
                 Ok(intensities) => {
                     if intensities.len() >= MAX_TRACKS { current_band_intensities.copy_from_slice(&intensities[0..MAX_TRACKS]); }
                     else { warn!("Received {} intensities < {}.", intensities.len(), MAX_TRACKS); current_band_intensities.fill(0.0); current_band_intensities[0..intensities.len()].copy_from_slice(&intensities); }
                 }
                 Err(TryRecvError::Empty) => break,
                 Err(TryRecvError::Disconnected) => { error!("Audio processor disconnected."); stop_signal.store(true, Ordering::Relaxed); break; }
             }
         }

        if stop_signal.load(Ordering::Relaxed) { break; }

        if let Some(frame_arc) = latest_frame_arc {
            let loop_start_time = Instant::now();
            let mut display_image = (*frame_arc).clone();
            let (_frame_w, _frame_h) = display_image.dimensions();
            let dynamic_image_input = DynamicImage::ImageRgb8((*frame_arc).clone());

            let proc_start = Instant::now();
            let results = model.forward(&[dynamic_image_input]);
            processing_time = proc_start.elapsed();

            let mut current_frame_draw_info: Vec<(usize, usize)> = Vec::new();

            match results {
                Ok(ys) => {
                    if let Some(y) = ys.first() {
                        let current_bboxes = y.bboxes().unwrap_or_default();
                        let current_masks = y.masks().unwrap_or_default();

                        // --- 4a. Attempt Selection ---
                         if let Some((slot, select_x, select_y)) = pending_selection.take() {
                             let mut candidates: Vec<(usize, Bbox)> = Vec::new();
                             for (i, bbox) in current_bboxes.iter().enumerate() {
                                 if select_x >= bbox.xmin() && select_x <= bbox.xmax() && select_y >= bbox.ymin() && select_y <= bbox.ymax() {
                                     let overlaps_other_track = tracked_objects.iter().enumerate().any( |(other_slot, tracked_opt)| other_slot != slot && tracked_opt.as_ref().map_or(false, |t| bbox.iou(&t.bbox) > SELECTION_IOU_THRESHOLD) );
                                     if !overlaps_other_track { candidates.push((i, bbox.clone())); }
                                 }
                             }
                             if !candidates.is_empty() {
                                 candidates.sort_by(|(_, bbox_a), (_, bbox_b)| (bbox_a.width() * bbox_a.height()).partial_cmp(&(bbox_b.width() * bbox_b.height())).unwrap_or(std::cmp::Ordering::Equal));
                                 let (best_candidate_index, best_candidate_bbox) = candidates.remove(0);
                                 tracked_objects[slot] = Some(TrackedObject { bbox: best_candidate_bbox, animation_phase: rng.gen::<f32>() * 2.0 * std::f32::consts::PI });
                                 current_frame_draw_info.push((slot, best_candidate_index));
                             } else { tracked_objects[slot] = None; }
                         }

                        // --- 4b. Track Existing Objects ---
                        let mut next_tracked_objects: Vec<Option<TrackedObject>> = vec![None; MAX_TRACKS];
                        let mut matched_current_indices: FnvHashSet<usize> = FnvHashSet::default();
                        for &(_, detection_index) in &current_frame_draw_info { matched_current_indices.insert(detection_index); }
                        for slot_idx in 0..MAX_TRACKS {
                             if current_frame_draw_info.iter().any(|(s, _)| *s == slot_idx) { next_tracked_objects[slot_idx] = tracked_objects[slot_idx].clone(); continue; }
                             if let Some(tracked_obj) = &tracked_objects[slot_idx] {
                                 let mut best_match: Option<(usize, f32, Bbox)> = None;
                                 for (i, current_bbox) in current_bboxes.iter().enumerate() {
                                     if matched_current_indices.contains(&i) { continue; }
                                     let iou = tracked_obj.bbox.iou(current_bbox);
                                     if iou > IOU_THRESHOLD {
                                         let is_better = match best_match { Some((_, current_best_iou, _)) => iou > current_best_iou, None => true, };
                                         if is_better { best_match = Some((i, iou, current_bbox.clone())); }
                                     }
                                 }
                                 if let Some((best_current_index, _iou, best_current_bbox)) = best_match {
                                     next_tracked_objects[slot_idx] = Some(TrackedObject { bbox: best_current_bbox.clone(), animation_phase: tracked_obj.animation_phase + 0.05 + current_band_intensities[slot_idx] * 0.1 });
                                     current_frame_draw_info.push((slot_idx, best_current_index));
                                     matched_current_indices.insert(best_current_index);
                                 } else { info!("Lost track {}", slot_idx); }
                             }
                         }
                         tracked_objects = next_tracked_objects;

                        // --- 4c. Drawing Logic (CALL VISUALS MODULE) ---
                        for (slot_index, detection_index) in &current_frame_draw_info {
                            let slot_index = *slot_index;
                            let detection_index = *detection_index;

                             if let (Some(tracked_obj), Some(mask_to_draw), Some(bbox_to_draw)) =
                                (tracked_objects[slot_index].as_ref(), current_masks.get(detection_index), current_bboxes.get(detection_index))
                             {
                                let intensity = current_band_intensities[slot_index];
                                let mask_image = mask_to_draw.mask();
                                let bbox_rect = Rect::at(bbox_to_draw.xmin() as i32, bbox_to_draw.ymin() as i32)
                                                   .of_size(bbox_to_draw.width().max(1.0) as u32, bbox_to_draw.height().max(1.0) as u32);

                                // --- Call the visuals drawing function ---
                                visuals::draw_visuals(
                                    &mut display_image,
                                    mask_image,
                                    bbox_rect,
                                    slot_index,
                                    intensity,
                                    frame_count,
                                    tracked_obj.animation_phase,
                                    &mut rng, // Pass RNG
                                );

                                // --- Draw Bounding Box ---
                                let border_brightness = (80.0f32 + intensity * 175.0f32).clamp(0.0f32, 255.0f32) as u8;
                                let border_color = match slot_index {
                                    0 => Rgb([border_brightness, 0, 0]),
                                    1 => Rgb([0, border_brightness, 0]),
                                    2 => Rgb([0, 0, border_brightness]),
                                    _ => Rgb([border_brightness, border_brightness, border_brightness]),
                                };
                                drawing::draw_hollow_rect_mut(&mut display_image, bbox_rect, border_color);
                                if intensity > 0.7 {
                                    let expanded_rect = Rect::at(bbox_rect.left() - 1, bbox_rect.top() - 1)
                                                           .of_size(bbox_rect.width() + 2, bbox_rect.height() + 2);
                                    drawing::draw_hollow_rect_mut(&mut display_image, expanded_rect, border_color);
                                }
                             }
                        } // End draw loop
                    }
                }
                Err(e) => { warn!("Model forward pass failed: {}", e); }
            }


            // --- Send Final Image to UI ---
             let final_color_image = { let size = [display_image.width() as usize, display_image.height() as usize]; let pixels = display_image.into_raw(); ColorImage::from_rgb(size, &pixels) };
             match ui_sender.try_send(SegmentationThreadMsg::Frame(Arc::new(final_color_image))) { Ok(_) => { ctx.request_repaint(); } Err(TrySendError::Full(_)) => {} Err(TrySendError::Disconnected(_)) => { info!("UI disconnected."); stop_signal.store(true, Ordering::Relaxed); break; } }
             debug!("Seg loop: {:.2?}, Model: {:.2?}, Tracks: {}, AudioInt: [{:.2}, {:.2}, {:.2}]", loop_start_time.elapsed(), processing_time, tracked_objects.iter().filter(|o| o.is_some()).count(), current_band_intensities.get(0).cloned().unwrap_or(0.0), current_band_intensities.get(1).cloned().unwrap_or(0.0), current_band_intensities.get(2).cloned().unwrap_or(0.0));

        } else {
            thread::sleep(Duration::from_millis(5));
        }
    } // End while !stop_signal

    info!("Segmentation loop finishing.");
}