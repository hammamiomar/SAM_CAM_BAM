use fnv::FnvHashSet; // Faster HashSet for indices
use std::{
    // Use std::collections::HashSet if fnv is not added
    // collections::HashSet,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};
use crossbeam_channel::{
    Receiver as CrossbeamReceiver, Sender as CrossbeamSender, TryRecvError as CrossbeamTryRecvError,
    TrySendError,
};
use egui::ColorImage;
use image::{DynamicImage, GenericImageView, Pixel, Rgb, RgbImage};
use imageproc::drawing;
use log::{debug, error, info, warn};

use usls::{
    models::YOLO,
    Options, Ys, Y, Bbox, Nms // Import Nms for IoU
};

use crate::camera::CameraThreadMsg;
// Import Audio Analyzer
use crate::music::AudioAnalyzer;

// --- Message type for receiving user interactions ---
#[derive(Debug, Clone)]
pub enum UserInteractionMsg {
    SelectObject { slot: usize, x: f32, y: f32 },
    RemoveObjectAt { x: f32, y: f32 },
    ClearSelection,
    UpdateAudioPath(String),
}

// Message type SENDING to UI thread
#[derive(Debug)]
pub enum SegmentationThreadMsg {
    Frame(Arc<ColorImage>),
    Error(String),
    AudioLoaded(String)
}

// --- Constants ---
pub const MAX_TRACKS: usize = 3;
const IOU_THRESHOLD: f32 = 0.3;
const SELECTION_IOU_THRESHOLD: f32 = 0.7; // Higher threshold to prevent selecting already tracked items
pub const TRACK_COLORS: [Rgb<u8>; MAX_TRACKS] = [
    Rgb([255u8, 0u8, 0u8]),   // Red   (Slot 0 / Low Freq)
    Rgb([0u8, 255u8, 0u8]),   // Green (Slot 1 / Mid Freq)
    Rgb([0u8, 0u8, 255u8]),   // Blue  (Slot 2 / High Freq)
];
// --- End Constants ---

// --- TrackedObject Struct ---
#[derive(Debug, Clone)]
struct TrackedObject {
    bbox: Bbox, // Bbox from the *previous* frame for matching
}
// --- End TrackedObject Struct ---

pub fn start_segmentation_thread(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    user_interaction_receiver: CrossbeamReceiver<UserInteractionMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) -> JoinHandle<()> {
    info!("Spawning segmentation thread (FastSAM Multi-Object + Audio Viz)");
    thread::spawn(move || {
        segmentation_loop(
            ui_sender,
            camera_receiver,
            user_interaction_receiver,
            stop_signal,
            ctx,
            model_options,
        );
    })
}

fn segmentation_loop(
    ui_sender: CrossbeamSender<SegmentationThreadMsg>,
    camera_receiver: CrossbeamReceiver<CameraThreadMsg>,
    user_interaction_receiver: CrossbeamReceiver<UserInteractionMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
    model_options: Options,
) {
    info!("Segmentation loop started (FastSAM Multi-Object + Audio Viz).");

    // Initialize Model
    let model_result = YOLO::new(model_options);
    let mut model = match model_result {
        Ok(m) => { info!("FastSAM model loaded successfully."); m }
        Err(e) => {
            let error_msg = format!("Failed to load FastSAM model: {}", e);
            error!("{}", error_msg);
            let _ = ui_sender.send(SegmentationThreadMsg::Error(error_msg));
            ctx.request_repaint();
            return;
        }
    };

    // Initialize Audio Analyzer
    let mut audio_analyzer = AudioAnalyzer::new(); // Create instance

    // State
    let mut tracked_objects: Vec<Option<TrackedObject>> = vec![None; MAX_TRACKS];
    let mut pending_selection: Option<(usize, f32, f32)> = None; // (slot, x, y)
    let mut processing_time = Duration::from_secs(0);
    let default_intensities = vec![0.0; MAX_TRACKS];

    while !stop_signal.load(Ordering::Relaxed) {
        // Receive User Interactions
        loop {
            match user_interaction_receiver.try_recv() {
                Ok(UserInteractionMsg::SelectObject { slot, x, y }) => {
                    if slot < MAX_TRACKS {
                        info!("Received selection request for slot {}: ({}, {})", slot, x, y);
                        pending_selection = Some((slot, x, y));
                    } else {
                         warn!("Received invalid slot index {} for selection.", slot);
                    }
                }
                Ok(UserInteractionMsg::RemoveObjectAt { x, y }) => {
                    info!("Received removal request at: ({}, {})", x, y);
                    let mut found_slot = None;
                    for (slot_idx, tracked_opt) in tracked_objects.iter().enumerate() {
                        if let Some(tracked) = tracked_opt {
                             if x >= tracked.bbox.xmin() && x <= tracked.bbox.xmax() &&
                                y >= tracked.bbox.ymin() && y <= tracked.bbox.ymax()
                             {
                                 found_slot = Some(slot_idx);
                                 break;
                             }
                        }
                    }
                    if let Some(slot_idx) = found_slot {
                        info!("Removing object in slot {}.", slot_idx);
                        tracked_objects[slot_idx] = None;
                    } else {
                         info!("No tracked object found at removal point ({}, {}).", x, y);
                     }
                }
                Ok(UserInteractionMsg::ClearSelection) => {
                    info!("Clearing all selections.");
                    tracked_objects.fill(None);
                    pending_selection = None;
                }
                 Ok(UserInteractionMsg::UpdateAudioPath(path_str)) => {
                    info!("Received request to update audio path: {}", path_str);
                    match audio_analyzer.update_audio_source(&path_str) {
                        Ok(_) => {
                            info!("Successfully updated audio source.");
                            // --- Send Confirmation Back ---
                            let _ = ui_sender.send(SegmentationThreadMsg::AudioLoaded(path_str));
                            ctx.request_repaint(); // Notify UI of state change
                        }
                        Err(e) => {
                            let err_msg = format!("Audio Load Error: {}", e);
                            error!("{}", err_msg);
                            let _ = ui_sender.send(SegmentationThreadMsg::Error(err_msg));
                            ctx.request_repaint();
                        }
                    }
                }
                Err(CrossbeamTryRecvError::Empty) => break,
                Err(CrossbeamTryRecvError::Disconnected) => {
                    warn!("User interaction channel disconnected.");
                    break;
                }
            }
        }

        // Receive LATEST Frame from Camera Thread
        let mut latest_frame_arc: Option<Arc<RgbImage>> = None;
        loop {
            match camera_receiver.try_recv() {
                Ok(CameraThreadMsg::Frame(frame)) => { latest_frame_arc = Some(frame); }
                Ok(CameraThreadMsg::Error(err)) => {
                    warn!("Received error from camera thread: {}", err);
                    let _ = ui_sender.send(SegmentationThreadMsg::Error(format!("Camera Error: {}", err)));
                    ctx.request_repaint();
                }
                Err(CrossbeamTryRecvError::Empty) => break,
                Err(CrossbeamTryRecvError::Disconnected) => {
                    error!("Camera thread disconnected. Stopping segmentation loop.");
                    let _ = ui_sender.send(SegmentationThreadMsg::Error("Camera disconnected.".to_string()));
                    ctx.request_repaint();
                    stop_signal.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        if stop_signal.load(Ordering::Relaxed) { break; }

        // Get Audio Intensities
        let band_intensities = audio_analyzer.get_band_intensities();
        let band_intensities = if band_intensities.len() >= MAX_TRACKS {
            band_intensities[0..MAX_TRACKS].to_vec()
        } else {
            let mut padded = band_intensities;
            padded.resize(MAX_TRACKS, 0.0);
            padded
        };


        if let Some(frame_arc) = latest_frame_arc {
            let loop_start_time = Instant::now();
            let mut display_image = (*frame_arc).clone();
            let (orig_width, orig_height) = display_image.dimensions();
            let dynamic_image_input = DynamicImage::ImageRgb8((*frame_arc).clone());

            // Inference
            let proc_start = Instant::now();
            let results = model.forward(&[dynamic_image_input]);
            processing_time = proc_start.elapsed();

            let mut current_frame_draw_info: Vec<(usize, usize)> = Vec::new(); // (slot_index, detection_index)

            match results {
                Ok(ys) => {
                    if let Some(y) = ys.get(0) {
                        let current_bboxes = y.bboxes().unwrap_or_default();
                        let current_masks = y.masks().unwrap_or_default();

                        // 1. Attempt Selection if Pending
                        if let Some((slot, select_x, select_y)) = pending_selection.take() {
                            let mut best_candidate: Option<(usize, Bbox)> = None;
                            for (i, bbox) in current_bboxes.iter().enumerate() {
                                if select_x >= bbox.xmin() && select_x <= bbox.xmax() &&
                                   select_y >= bbox.ymin() && select_y <= bbox.ymax() {
                                    let overlaps_other_track = tracked_objects.iter().enumerate().any(
                                        |(other_slot, tracked_opt)| {
                                            if other_slot == slot { return false; }
                                            if let Some(tracked) = tracked_opt {
                                                 bbox.iou(&tracked.bbox) > SELECTION_IOU_THRESHOLD
                                             } else { false }
                                        }
                                    );
                                    if !overlaps_other_track {
                                        best_candidate = Some((i, bbox.clone()));
                                        break;
                                    } else {
                                         info!("Clicked object at index {} overlaps strongly with another track.", i);
                                    }
                                }
                            }
                            if let Some((candidate_index, candidate_bbox)) = best_candidate {
                                info!("Assigning object at index {} to slot {}", candidate_index, slot);
                                tracked_objects[slot] = Some(TrackedObject { bbox: candidate_bbox });
                                current_frame_draw_info.push((slot, candidate_index));
                            } else {
                                info!("No suitable new object found for slot {} at selection point ({}, {}).", slot, select_x, select_y);
                                // Ensure slot is cleared if replacing and no candidate found
                                tracked_objects[slot] = None;
                            }
                        }

                        // 2. Track Existing Objects
                        let mut next_tracked_objects: Vec<Option<TrackedObject>> = vec![None; MAX_TRACKS];
                        let mut matched_current_indices: FnvHashSet<usize> = FnvHashSet::default();

                        for slot_idx in 0..MAX_TRACKS {
                            if let Some(tracked_obj) = &tracked_objects[slot_idx] {
                                let mut best_match: Option<(usize, f32, Bbox)> = None;
                                for (i, current_bbox) in current_bboxes.iter().enumerate() {
                                    if matched_current_indices.contains(&i) { continue; }
                                    let iou = tracked_obj.bbox.iou(current_bbox);
                                    if iou > IOU_THRESHOLD {
                                        if let Some((_, current_best_iou, _)) = best_match {
                                            if iou > current_best_iou {
                                                best_match = Some((i, iou, current_bbox.clone()));
                                            }
                                        } else {
                                            best_match = Some((i, iou, current_bbox.clone()));
                                        }
                                    }
                                }
                                if let Some((best_current_index, best_iou, best_current_bbox)) = best_match {
                                    debug!("Tracked slot {} (IoU: {:.3}) to current index {}", slot_idx, best_iou, best_current_index);
                                    next_tracked_objects[slot_idx] = Some(TrackedObject { bbox: best_current_bbox.clone() }); // Clone here
                                    if !current_frame_draw_info.iter().any(|(s, _)| *s == slot_idx) {
                                        current_frame_draw_info.push((slot_idx, best_current_index));
                                    }
                                    matched_current_indices.insert(best_current_index);
                                } else {
                                    info!("Lost track of object in slot {}.", slot_idx);
                                }
                            }
                        }
                        tracked_objects = next_tracked_objects;

                        // 3. Drawing Logic
                        for (slot_index, detection_index) in current_frame_draw_info {
                            if detection_index >= current_masks.len() {
                                warn!("Draw index {} out of bounds for masks (len {})", detection_index, current_masks.len());
                                continue;
                            }
                             if let Some(mask_to_draw) = current_masks.get(detection_index) {
                                let intensity = band_intensities[slot_index];
                                let base_alpha = 0.1;
                                let max_additional_alpha = 0.7;
                                let final_alpha = (base_alpha + intensity * max_additional_alpha).clamp(0.05, 0.95);
                                let base_color_rgb = TRACK_COLORS[slot_index];
                                let overlay_alpha = final_alpha;
                                let mask_image = mask_to_draw.mask();

                                if mask_image.dimensions() == (orig_width, orig_height) {
                                    for y_coord in 0..orig_height {
                                        for x_coord in 0..orig_width {
                                            if let Some(mask_pixel) = mask_image.get_pixel_checked(x_coord, y_coord) {
                                                if mask_pixel[0] > 128 {
                                                    let current_pixel = display_image.get_pixel_mut(x_coord, y_coord);
                                                    let current_r = current_pixel[0] as f32; let color_r = base_color_rgb[0] as f32;
                                                    let current_g = current_pixel[1] as f32; let color_g = base_color_rgb[1] as f32;
                                                    let current_b = current_pixel[2] as f32; let color_b = base_color_rgb[2] as f32;
                                                    let new_r = ((current_r * (1.0-overlay_alpha)) + (color_r * overlay_alpha)).clamp(0.0, 255.0) as u8;
                                                    let new_g = ((current_g * (1.0-overlay_alpha)) + (color_g * overlay_alpha)).clamp(0.0, 255.0) as u8;
                                                    let new_b = ((current_b * (1.0-overlay_alpha)) + (color_b * overlay_alpha)).clamp(0.0, 255.0) as u8;
                                                    *current_pixel = Rgb([new_r, new_g, new_b]);
                                                }
                                            }
                                        }
                                    }
                                } else { warn!("Tracked mask (slot {}) dimension mismatch.", slot_index); }

                                // Draw Bbox
                                if detection_index >= current_bboxes.len() {
                                     warn!("Draw index {} out of bounds for bboxes (len {})", detection_index, current_bboxes.len());
                                     continue;
                                }
                                if let Some(bbox) = current_bboxes.get(detection_index) {
                                    let rect = imageproc::rect::Rect::at(bbox.xmin() as i32, bbox.ymin() as i32)
                                                        .of_size(bbox.width() as u32, bbox.height() as u32);
                                    let border_brightness = (100.0 + intensity * 155.0) as u8;
                                    let border_color = match slot_index {
                                        0 => Rgb([border_brightness, 0, 0]),
                                        1 => Rgb([0, border_brightness, 0]),
                                        2 => Rgb([0, 0, border_brightness]),
                                        _ => Rgb([border_brightness, border_brightness, border_brightness]), // Fallback
                                    };
                                    drawing::draw_hollow_rect_mut(&mut display_image, rect, border_color);
                                }
                             }
                        }
                    }
                }
                Err(e) => {
                    warn!("FastSAM model forward pass failed: {}", e);
                    let _ = ui_sender.send(SegmentationThreadMsg::Error(format!("Segmentation failed: {}", e)));
                    ctx.request_repaint();
                }
            }

            // Convert final image and Send to UI
            let final_color_image = {
                let size = [display_image.width() as usize, display_image.height() as usize];
                let pixels = display_image.into_raw();
                ColorImage::from_rgb(size, &pixels)
            };
            match ui_sender.try_send(SegmentationThreadMsg::Frame(Arc::new(final_color_image))) {
                Ok(_) => { ctx.request_repaint(); }
                Err(TrySendError::Full(_)) => { warn!("UI channel full. Dropping frame."); }
                Err(TrySendError::Disconnected(_)) => {
                    info!("UI receiver disconnected. Stopping segmentation loop.");
                    break;
                }
            }
            debug!("Seg loop: {:?}, Model: {:?}, Tracks: {}, AudioInt: [{:.2}, {:.2}, {:.2}]",
                   loop_start_time.elapsed(), processing_time, tracked_objects.iter().filter(|o| o.is_some()).count(),
                   band_intensities[0], band_intensities[1], band_intensities[2]);

        } else {
            thread::sleep(Duration::from_millis(5));
        }
    }

    info!("Segmentation loop finishing.");
}