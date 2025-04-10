// src/ui.rs (Final Version - Fully Decoupled)
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle}, // Added thread for file dialog
    time::{Duration, Instant},
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use egui::{
    Align, Color32, ColorImage, ImageData, Layout, Pos2, Rect, Response, Sense, Stroke,
    TextureHandle, TextureOptions, Vec2,
};
use log::{debug, error, info, warn};
use nokhwa::utils::{CameraIndex, Resolution};

use crate::{
    camera::{self, CameraThreadMsg},
    segmentation::{self, SegmentationThreadMsg, UserInteractionMsg, MAX_TRACKS, TRACK_COLORS},
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
    current_selection_slot: usize,
    loaded_audio_path: Option<String>,
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI with FastSAM (Multi-Object + Audio)");
        let camera_index = CameraIndex::Index(0);

        // Configure Model Options for FastSAM
        let device_str = "mps";
        let dtype_str = "fp16";
        let model_options = match usls::Options::fastsam_s()
            .with_model_device(device_str.try_into().expect("Bad device string"))
            .with_model_dtype(dtype_str.try_into().unwrap_or(usls::DType::Fp32))
            .with_model_file("models/FastSAM-s.onnx") // ADJUST PATH
            .with_nc(1)
            .with_class_names(&["object"])
            .with_class_confs(&[0.35])
            .with_iou(0.45)
            .with_find_contours(true)
            .commit()
         {
            Ok(opts) => opts,
            Err(e) => panic!("Model options failed to commit: {}", e),
         };


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
            current_selection_slot: 0,
            loaded_audio_path: None,
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
            } else { f32::INFINITY };
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
             let relative_pos = ui_pos - image_rect.min; //position in image rect
             let norm_x = (relative_pos.x / displayed_image_size.x).clamp(0.0, 1.0); //displayed image is the image rect in the ui. so its gettingthe normalized coord for image
             let norm_y = (relative_pos.y / displayed_image_size.y).clamp(0.0, 1.0);
             let image_x = norm_x * original_width;
             let image_y = norm_y * original_height;
             if image_x >= 0.0 && image_x < original_width && image_y >= 0.0 && image_y < original_height {
                  Some((image_x, image_y))
             } else {
                  warn!("Calculated image coords slightly out of bounds: ({}, {}), clamping.", image_x, image_y);
                  Some((image_x.clamp(0.0, original_width - 1.0), image_y.clamp(0.0, original_height-1.0)))
             }
         } else { None } // Original resolution unknown
    }
}

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();

        // --- 1. Process incoming messages FIRST (mutates state) ---
        loop {
            match self.seg_thread_rx.try_recv() {
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(processed_frame_arc) => {
                        let size = processed_frame_arc.size;
                        let frame_size_vec = Vec2::new(size[0] as f32, size[1] as f32);
                        if self.camera_resolution.is_none() || //checks to see if need to assign camera resolution. if false or if mismatch with actual frame res
                           self.camera_resolution.map_or(false, |res| res.width() != size[0] as u32 || res.height() != size[1] as u32) {
                            let new_res = Resolution::new(size[0] as u32, size[1] as u32);
                            if self.camera_resolution.is_some() {
                                warn!("Frame resolution ({:?}) differs from stored camera resolution ({:?}). Updating.", size, self.camera_resolution.unwrap());
                                let _ = self.user_interaction_tx.send(UserInteractionMsg::ClearSelection);
                            } else {
                                info!("Received first frame, setting camera resolution to {}x{}", size[0], size[1]);
                            }
                            self.camera_resolution = Some(new_res);
                        }
                        match self.texture { //initialize texture to display, then after it will send to seg thread the different selections... to show the boxes n such
                            Some(ref mut texture) => {
                                if self.texture_size.map_or(true, |s| s != frame_size_vec) {
                                     self.texture_size = Some(frame_size_vec);
                                }
                                texture.set(ImageData::Color(processed_frame_arc), TextureOptions::LINEAR);
                            }
                            None => {
                                let new_texture = ctx.load_texture( "webcam_stream", ImageData::Color(processed_frame_arc), TextureOptions::LINEAR );
                                self.texture_size = Some(frame_size_vec);
                                self.texture = Some(new_texture);
                            }
                        }
                        self.seg_error = None;
                    }
                    SegmentationThreadMsg::Error(err) => {
                         if self.seg_error.as_ref() != Some(&err) {
                             error!("Segmentation Error: {}", err);
                             self.seg_error = Some(err);
                         }
                    }
                    SegmentationThreadMsg::AudioLoaded(path_str) => {
                         info!("UI received confirmation: Audio loaded from {}", path_str);
                         self.loaded_audio_path = Some(path_str); // MUTATE state here
                         self.seg_error = None;
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

        // --- 2. Prepare state for UI drawing ---
        let mut trigger_file_dialog = false;
        let mut clear_all_clicked = false;
        let mut interaction_error_this_frame: Option<String> = None;
        let mut new_selection_slot = self.current_selection_slot;

        // --- 3. Define and draw UI ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
             egui::menu::bar(ui, |ui| {
                 let is_web = cfg!(target_arch = "wasm32");
                 if !is_web { ui.menu_button("File", |ui| { if ui.button("Quit").clicked() { ctx.send_viewport_cmd(egui::ViewportCommand::Close); } }); ui.add_space(16.0); }
                 egui::widgets::global_theme_preference_buttons(ui);
             });
         });

        egui::SidePanel::left("control_panel")
            .resizable(false)
            .default_width(150.0)
            .show(ctx, |ui| {
                ui.heading("Track Control"); ui.separator(); ui.label("Select target slot:");
                ui.vertical(|ui| {
                    for i in 0..MAX_TRACKS {
                        ui.horizontal(|ui| {
                            let color = TRACK_COLORS[i];
                            let egui_color = Color32::from_rgb(color[0], color[1], color[2]);
                            let is_selected = new_selection_slot == i; // Use local var
                            if ui.radio(is_selected, "").clicked() {
                                 new_selection_slot = i; // Mutate local var
                             };
                            let (rect, _) = ui.allocate_exact_size(Vec2::new(16.0, 16.0), Sense::hover());
                            let stroke = if is_selected { Stroke::new(2.0, Color32::WHITE) } else { Stroke::NONE };
                            ui.painter().rect_filled(rect, 3.0, egui_color);
                            ui.painter().rect_stroke(rect, 3.0, stroke,egui::StrokeKind::Inside); // Corrected use
                            ui.selectable_label(is_selected, format!("Track {}", i + 1))
                                .on_hover_text(format!("Assign to Track {}", i+1));
                        });
                    }
                });
                ui.separator(); ui.label("Left-Click: Assign object."); ui.label("Right-Click: Remove object."); ui.separator();
                if ui.button("Clear All Selections").clicked() { clear_all_clicked = true; }
                ui.separator();

                ui.heading("Audio Control"); ui.separator();
                ui.horizontal(|ui| {
                    ui.label("File:");
                    if ui.button("📂").on_hover_text("Open Audio File").clicked() { trigger_file_dialog = true; }
                 });
                 ui.separator(); ui.label("Loaded Audio:");
                 if let Some(loaded_path) = &self.loaded_audio_path { // Immutable borrow OK
                     let display_path = if loaded_path.len() > 20 { format!("...{}", &loaded_path[loaded_path.len()-17..]) } else { loaded_path.clone() };
                     ui.label(&display_path).on_hover_text(loaded_path);
                 } else { ui.label("None"); }
                 ui.separator();

                 ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                 if let Some(res) = &self.camera_resolution { ui.label(format!("Cam Res: {}x{}", res.width(), res.height())); }
                 else if self.camera_error.is_none() && self.seg_error.is_none() { ui.label("Cam Res: ..."); }
                 else { ui.label("Cam Res: Error"); }
            });

        // Apply deferred radio button mutation
        self.current_selection_slot = new_selection_slot;

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(err) = &self.camera_error { ui.colored_label(egui::Color32::YELLOW, format!("Camera Status: {}", err)); }
            if let Some(err) = &self.seg_error { ui.colored_label(egui::Color32::RED, format!("Status: {}", err)); }

            match &self.texture { // Immutable borrow of self.texture
                Some(texture) => {
                    if let Some(tex_size) = self.texture_size { // Immutable borrow of self.texture_size
                        let aspect_ratio = if tex_size.y > 0.0 { tex_size.x / tex_size.y } else { 1.0 };
                        let available_width = ui.available_width(); let available_height = ui.available_height();
                        let mut image_width = available_width; let mut image_height = available_width / aspect_ratio;
                        if image_height > available_height { image_height = available_height; image_width = available_height * aspect_ratio; }
                        let displayed_size = Vec2::new(image_width, image_height);

                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            let image_widget = egui::Image::new(texture)
                                .max_width(image_width).max_height(image_height)
                                .maintain_aspect_ratio(true).sense(Sense::click());
                            let response: Response = ui.add(image_widget);

                            // --- Click Handlers: ONLY send messages, store potential errors locally ---
                            let tx = self.user_interaction_tx.clone(); // Clone sender
                            let maybe_coords = response.interact_pointer_pos()
                                .and_then(|pos| self.ui_pos_to_image_coords(pos, response.rect, displayed_size)); // Reads self.camera_resolution
                            let slot_to_use = self.current_selection_slot; // Reads self.current_selection_slot

                            if response.clicked_by(egui::PointerButton::Primary) {
                                if let Some((img_x, img_y)) = maybe_coords {
                                    info!("LClick S{} ({},{})", slot_to_use, img_x, img_y);
                                    if let Err(e) = tx.send(UserInteractionMsg::SelectObject { slot: slot_to_use, x: img_x, y: img_y }) {
                                        let err_msg = format!("Comm Err (Select): {}", e);
                                        error!("{}", err_msg);
                                        interaction_error_this_frame = Some(err_msg); // Store error locally
                                    }
                                } else { warn!("LClick pos convert failed."); }
                            } else if response.clicked_by(egui::PointerButton::Secondary) {
                                if let Some((img_x, img_y)) = maybe_coords {
                                     info!("RClick ({},{})", img_x, img_y);
                                     if let Err(e) = tx.send(UserInteractionMsg::RemoveObjectAt { x: img_x, y: img_y }) {
                                         let err_msg = format!("Comm Err (Remove): {}", e);
                                         error!("{}", err_msg);
                                         interaction_error_this_frame = Some(err_msg); // Store error locally
                                     }
                                } else { warn!("RClick pos convert failed."); }
                            }
                        }); // End ui.with_layout
                    } else { ui.label("Texture exists but size unknown."); }
                }
                 None if self.camera_error.is_none() && self.seg_error.is_none() => {
                     ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0); ui.spinner(); ui.label("Initializing stream...");
                    });
                 }
                 None => { // Error state
                    ui.with_layout(Layout::top_down(Align::Center), |ui| {
                        ui.add_space(ui.available_height() / 3.0);
                        ui.colored_label(egui::Color32::GRAY, "Stream unavailable due to errors.");
                    });
                 }
            } // End match &self.texture
        }); // End CentralPanel

        // --- 4. Process deferred actions ---
        if clear_all_clicked {
            info!("Sending deferred ClearSelection message.");
            if let Err(e) = self.user_interaction_tx.send(UserInteractionMsg::ClearSelection) {
                let err_msg = format!("Comm Err (Clear): {}", e);
                error!("{}", err_msg);
                interaction_error_this_frame = Some(err_msg); // Store error locally
            }
        }

        if trigger_file_dialog {
             let file_dialog_tx = self.user_interaction_tx.clone();
             std::thread::spawn(move || { // Closure only captures cloned sender
                 if let Some(path) = rfd::FileDialog::new().add_filter("Audio", &["mp3", "wav", "flac", "ogg"]).pick_file() {
                     if let Some(path_str) = path.to_str() {
                         info!("File dialog selected: {}", path_str);
                         if let Err(e) = file_dialog_tx.send(UserInteractionMsg::UpdateAudioPath(path_str.to_string())) {
                             error!("Failed to send selected audio path: {}", e);
                         }
                     } else { error!("Selected path not valid UTF-8: {:?}", path); }
                 } else { info!("File dialog cancelled."); }
             });
        }

        // --- 5. Apply interaction error state mutation LAST ---
        if let Some(err) = interaction_error_this_frame {
            // Only update if no more critical error is already present from message loop
            if self.seg_error.is_none() {
                self.seg_error = Some(err);
            }
        }

    } // --- End of update method ---

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        info!("Exit requested. Stopping threads...");
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);
        if let Some(handle) = self.cam_thread_handle.take() { if let Err(e) = handle.join() { error!("Cam join error: {:?}", e); } else { info!("Cam thread joined."); } }
        if let Some(handle) = self.seg_thread_handle.take() { if let Err(e) = handle.join() { error!("Seg join error: {:?}", e); } else { info!("Seg thread joined."); } }
        info!("All threads stopped.");
    }
}