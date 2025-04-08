use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use egui::{
    Align, Color32, ColorImage, ImageData, Layout, Pos2, Rect, Response, Sense, Stroke, // Added Color32, Stroke
    TextureHandle, TextureOptions, Vec2,
};
use log::{debug, error, info, warn};
use nokhwa::utils::{CameraIndex, Resolution};

use crate::{
    camera::{self, CameraThreadMsg},
    segmentation::{self, SegmentationThreadMsg, UserInteractionMsg, MAX_TRACKS, TRACK_COLORS}, // Import constants
};
use usls::Options;

const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);

pub struct WebcamAppUI {
    texture: Option<TextureHandle>,
    user_interaction_tx: Sender<UserInteractionMsg>,
    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,
    seg_thread_rx: Receiver<SegmentationThreadMsg>,
    camera_error: Option<String>,
    seg_error: Option<String>,
    camera_resolution: Option<Resolution>,
    texture_size: Option<Vec2>,
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
    // UI State for Multi-Selection
    current_selection_slot: usize, // Which track slot (0, 1, 2) is currently active for selection
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI with FastSAM (Multi-Object)");
        let camera_index = CameraIndex::Index(0);

        // Configure Model Options for FastSAM
        let device_str = "mps"; // Or "cpu", "cuda:0", etc.
        let dtype_str = "fp16"; // Or "fp32"
        let model_options = usls::Options::fastsam_s()
            .with_model_device(device_str.try_into().expect("Bad device string"))
            .with_model_dtype(dtype_str.try_into().unwrap_or(usls::DType::Fp32))
            .with_model_file("models/FastSAM-s.onnx") // ADJUST PATH
            .with_nc(1)
            .with_class_names(&["object"])
            .with_class_confs(&[0.35])
            .with_iou(0.45)
            .with_find_contours(false)
            .commit().expect("Failed to commit FastSAM model options");

        // Channels
        let (cam_to_seg_tx, cam_to_seg_rx) = unbounded::<CameraThreadMsg>();
        let (seg_to_ui_tx, seg_to_ui_rx) = bounded::<SegmentationThreadMsg>(1);
        let (user_interaction_tx, user_interaction_rx) = unbounded::<UserInteractionMsg>();

        // Stop Signals & Clones
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));
        let cam_stop_signal_clone = cam_stop_signal.clone();
        let cam_ctx_clone = cc.egui_ctx.clone();
        let seg_stop_signal_clone = seg_stop_signal.clone();
        let seg_ctx_clone = cc.egui_ctx.clone();

        // Start Threads
        let cam_thread_handle = Some(camera::start_camera_thread(
            camera_index, cam_to_seg_tx, cam_stop_signal_clone, cam_ctx_clone,
        ));
        let seg_thread_handle = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx, cam_to_seg_rx, user_interaction_rx,
            seg_stop_signal_clone, seg_ctx_clone, model_options,
        ));

        Self {
            texture: None,
            user_interaction_tx,
            cam_thread_handle,
            cam_stop_signal,
            seg_thread_handle,
            seg_stop_signal,
            seg_thread_rx: seg_to_ui_rx, // Assign receiver
            camera_error: None,
            seg_error: None,
            camera_resolution: None,
            texture_size: None,
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
            current_selection_slot: 0, // Default to selecting for the first slot
        }
    }

    fn update_fps_counter(&mut self) {
        self.frames_since_last_update += 1;
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_fps_update_time);

        if elapsed >= FPS_UPDATE_INTERVAL {
            let elapsed_secs = elapsed.as_secs_f32();
            self.last_calculated_fps = if elapsed_secs > 0.0 {
                self.frames_since_last_update as f32 / elapsed_secs
            } else {
                f32::INFINITY
            };
            self.frames_since_last_update = 0;
            self.last_fps_update_time = now;
        }
    }

    fn ui_pos_to_image_coords(
        &self,
        ui_pos: Pos2,
        image_rect: Rect,
        displayed_image_size: Vec2,
    ) -> Option<(f32, f32)> {
        if !image_rect.contains(ui_pos) { return None; }
        if let Some(original_res) = self.camera_resolution {
            let original_width = original_res.width() as f32;
            let original_height = original_res.height() as f32;
            if displayed_image_size.x <= 0.0 || displayed_image_size.y <= 0.0 {
                warn!("Displayed image size has zero dimension: {:?}", displayed_image_size);
                return None;
            }
            let relative_pos = ui_pos - image_rect.min;
            let norm_x = (relative_pos.x / displayed_image_size.x).clamp(0.0, 1.0);
            let norm_y = (relative_pos.y / displayed_image_size.y).clamp(0.0, 1.0);
            let image_x = norm_x * original_width;
            let image_y = norm_y * original_height;
            if image_x >= 0.0 && image_x < original_width && image_y >= 0.0 && image_y < original_height {
                 Some((image_x, image_y))
            } else {
                 warn!("Calculated image coords slightly out of bounds: ({}, {}), clamping might occur in seg thread.", image_x, image_y);
                 Some((image_x.clamp(0.0, original_width - 1.0), image_y.clamp(0.0, original_height-1.0)))
            }
        } else {
            None
        }
    }
}

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();

        // Process messages from SEGMENTATION thread
        loop {
            match self.seg_thread_rx.try_recv() {
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(processed_frame_arc) => {
                        let size = processed_frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);
                        if self.camera_resolution.is_none() {
                             self.camera_resolution = Some(Resolution::new(size[0] as u32, size[1] as u32));
                             info!("Received first frame, setting camera resolution to {}x{}", size[0], size[1]);
                        }
                        else if let Some(res) = self.camera_resolution {
                            if res.width() != size[0] as u32 || res.height() != size[1] as u32 {
                                warn!("Frame resolution ({:?}) differs from stored camera resolution ({:?}). Updating.", size, res);
                                self.camera_resolution = Some(Resolution::new(size[0] as u32, size[1] as u32));
                                let _ = self.user_interaction_tx.send(UserInteractionMsg::ClearSelection);
                            }
                        }
                        match self.texture {
                            Some(ref mut texture) => {
                                if self.texture_size.map_or(true, |s| s != frame_size_vec) {
                                     self.texture_size = Some(frame_size_vec);
                                }
                                texture.set(ImageData::Color(processed_frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                info!("Creating texture with size: {:?}", size);
                                let new_texture = ctx.load_texture( "webcam_stream", ImageData::Color(processed_frame_arc), TextureOptions::LINEAR, );
                                self.texture_size = Some(frame_size_vec);
                                self.texture = Some(new_texture);
                            }
                        }
                        self.seg_error = None; // Clear seg error on successful frame
                    }
                    SegmentationThreadMsg::Error(err) => {
                        error!("Segmentation Error: {}", err);
                        self.seg_error = Some(err);
                    }
                },
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    let err_msg = "Segmentation thread disconnected unexpectedly.".to_string();
                    error!("{}", err_msg);
                    self.seg_error = Some(err_msg);
                    if let Some(handle) = self.seg_thread_handle.take() {
                         if let Err(e) = handle.join() { self.seg_error = Some(format!("Segmentation thread panicked: {:?}", e)); }
                    }
                    break;
                 }
            }
        }


        // UI Definition
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                let is_web = cfg!(target_arch = "wasm32");
                if !is_web {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
                egui::widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::SidePanel::left("selection_panel")
            .resizable(false)
            .default_width(130.0)
            .show(ctx, |ui| {
                ui.heading("Track Control");
                ui.separator();
                ui.label("Select target slot:");

                // Radio buttons for each track slot
                for i in 0..MAX_TRACKS {
                    let color = TRACK_COLORS[i];
                    let egui_color = Color32::from_rgb(color[0], color[1], color[2]);

                    ui.horizontal(|ui| {
                        let is_selected = self.current_selection_slot == i;
                        if ui.radio(is_selected, "").clicked() {
                             self.current_selection_slot = i;
                         };

                        // Color swatch
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                        let stroke = if is_selected { Stroke::new(2.0, Color32::WHITE) } else { Stroke::NONE };
                        ui.painter().rect_filled(rect, 3.0, egui_color);
                        ui.painter().rect_stroke(rect, 3.0, stroke,egui::StrokeKind::Middle);

                        ui.label(format!("Track {}", i + 1));
                    });
                }
                 ui.separator();
                 ui.label("Left-Click: Assign to selected slot.");
                 ui.label("Right-Click: Remove object.");
                 ui.separator();

                // Clear All Selections Button
                 if ui.button("Clear All Selections").clicked() {
                    info!("Clear All Selections button clicked.");
                    match self.user_interaction_tx.send(UserInteractionMsg::ClearSelection) {
                        Ok(_) => debug!("Sent ClearSelection message."),
                        Err(e) => error!("Failed to send ClearSelection message: {}", e),
                    }
                 }

                 // Status Labels
                 ui.separator();
                 ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                 if let Some(res) = self.camera_resolution {
                     ui.label(format!("Cam Res: {}x{}", res.width(), res.height()));
                 } else if self.camera_error.is_none() && self.seg_error.is_none() {
                     ui.label("Cam Res: ...");
                 } else {
                     ui.label("Cam Res: Error");
                 }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.camera_error { ui.colored_label(egui::Color32::YELLOW, format!("Camera Status: {}", err)); }
            if let Some(err) = &self.seg_error { ui.colored_label(egui::Color32::RED, format!("Segmentation Status: {}", err)); }

            match &self.texture {
                Some(texture) => {
                    if let Some(tex_size) = self.texture_size {
                        // Calculate display size
                        let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
                        let available_width = ui.available_width();
                        let available_height = ui.available_height();
                        let mut image_width = available_width;
                        let mut image_height = available_width / aspect_ratio;
                        if image_height > available_height { image_height = available_height; image_width = available_height * aspect_ratio; }
                        let displayed_size = Vec2::new(image_width, image_height);

                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            let image_widget = egui::Image::new(texture)
                                .max_width(image_width)
                                .max_height(image_height)
                                .maintain_aspect_ratio(true)
                                .sense(Sense::click()); // Sense both clicks

                            let response: Response = ui.add(image_widget);

                            // Handle Left Click (Selection/Assignment)
                            if response.clicked_by(egui::PointerButton::Primary) {
                                if let Some(pointer_pos) = response.interact_pointer_pos() {
                                    if let Some((img_x, img_y)) = self.ui_pos_to_image_coords(pointer_pos, response.rect, displayed_size) {
                                        info!("Left Click at ({}, {}) for Slot {}", img_x, img_y, self.current_selection_slot);
                                        match self.user_interaction_tx.send(UserInteractionMsg::SelectObject { slot: self.current_selection_slot, x: img_x, y: img_y }) {
                                            Ok(_) => debug!("Sent SelectObject message for slot {}.", self.current_selection_slot),
                                            Err(e) => error!("Failed to send SelectObject message: {}", e),
                                        }
                                    } else { warn!("Could not convert left-click position."); }
                                }
                            }
                            // Handle Right Click (Removal)
                            else if response.clicked_by(egui::PointerButton::Secondary) {
                                if let Some(pointer_pos) = response.interact_pointer_pos() {
                                    if let Some((img_x, img_y)) = self.ui_pos_to_image_coords(pointer_pos, response.rect, displayed_size) {
                                        info!("Right Click at ({}, {}) for Removal", img_x, img_y);
                                        match self.user_interaction_tx.send(UserInteractionMsg::RemoveObjectAt { x: img_x, y: img_y }) {
                                            Ok(_) => debug!("Sent RemoveObjectAt message."),
                                            Err(e) => error!("Failed to send RemoveObjectAt message: {}", e),
                                        }
                                    } else { warn!("Could not convert right-click position."); }
                                }
                            }
                        });
                    } else { ui.label("Texture exists but size unknown."); }
                }
                 None if self.camera_error.is_none() && self.seg_error.is_none() => {
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0); ui.spinner(); ui.label("Initializing stream...");
                    });
                }
                 None => {
                    ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0);
                        ui.colored_label(egui::Color32::GRAY, "Stream unavailable due to errors.");
                    });
                }
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping threads...");
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.cam_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining camera thread: {:?}", e); }
             else { info!("Camera thread joined successfully."); }
        }
        if let Some(handle) = self.seg_thread_handle.take() {
             if let Err(e) = handle.join() { error!("Error joining segmentation thread: {:?}", e); }
             else { info!("Segmentation thread joined successfully."); }
        }
        info!("All threads stopped.");
    }
}