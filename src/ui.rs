// src/ui.rs
use cpal::{traits::StreamTrait, Stream};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender, TryRecvError};
use egui::{widgets, Align, Color32, ImageData, Layout, TextureHandle, TextureOptions, Vec2};
use log::{error, info, warn};
use nokhwa::utils::{CameraIndex, Resolution};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crate::{
    camera::{self},
    live_audio,
    music::{self},
    segmentation::{self, SegmentationThreadMsg, UserInteractionSegMsg, MAX_TRACKS},
};

const FPS_UPDATE_INTERVAL: Duration = Duration::from_millis(500);
#[derive(Debug, Clone, PartialEq)]
enum LiveAudioStatus {
    Initializing,
    Running(u32, u16),
    Error(String),
    Disabled,
}

pub struct WebcamAppUI {
    texture: Option<TextureHandle>,
    seg_to_ui_rx: Receiver<SegmentationThreadMsg>,
    _user_interaction_tx: Sender<UserInteractionSegMsg>,
    cam_thread_handle: Option<JoinHandle<()>>,
    cam_stop_signal: Arc<AtomicBool>,
    seg_thread_handle: Option<JoinHandle<()>>,
    seg_stop_signal: Arc<AtomicBool>,
    audio_capture_thread_handle: Option<Stream>,
    audio_capture_stop_signal: Arc<AtomicBool>,
    audio_processor_thread_handle: Option<JoinHandle<()>>,
    audio_processor_stop_signal: Arc<AtomicBool>,
    camera_error: Option<String>,
    seg_error: Option<String>,
    live_audio_status: LiveAudioStatus,
    camera_resolution: Option<Resolution>,
    texture_size: Option<Vec2>,
    last_fps_update_time: Instant,
    frames_since_last_update: u32,
    last_calculated_fps: f32,
}

impl WebcamAppUI {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        info!("Initializing WebcamAppUI (Persistent Random Assignment Viz)"); // Log updated
        let camera_index = CameraIndex::Index(0);
        let device_str = "mps";
        let dtype_str = "fp16";
        let model_options = match usls::Options::fastsam_s()
            .with_model_device(device_str.try_into().expect("Bad device"))
            .with_model_dtype(dtype_str.try_into().unwrap_or(usls::DType::Fp32))
            .with_model_file("models/FastSAM-s.onnx")
            .with_nc(1)
            .with_class_names(&["object"])
            .with_class_confs(&[0.35])
            .with_iou(0.45)
            .with_find_contours(true)
            .commit()
        {
            Ok(o) => o,
            Err(e) => {
                panic!("Model opts failed: {}", e)
            }
        };
        let (cam_to_seg_tx, cam_to_seg_rx) = unbounded();
        let (seg_to_ui_tx, seg_to_ui_rx) = bounded(1);
        let (_user_interaction_tx, user_interaction_rx) = unbounded(); // Keep channel
        let (raw_samples_tx, raw_samples_rx) = bounded(10);
        let (intensities_tx, intensities_rx) = bounded(5);
        let cam_stop_signal = Arc::new(AtomicBool::new(false));
        let seg_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_capture_stop_signal = Arc::new(AtomicBool::new(false));
        let audio_processor_stop_signal = Arc::new(AtomicBool::new(false));
        let cam_stop_clone = cam_stop_signal.clone();
        let cam_ctx = cc.egui_ctx.clone();
        let seg_stop_clone = seg_stop_signal.clone();
        let seg_ctx = cc.egui_ctx.clone();
        let audio_cap_stop = audio_capture_stop_signal.clone();
        let audio_proc_stop = audio_processor_stop_signal.clone();
        let cam_thread = Some(camera::start_camera_thread(
            camera_index,
            cam_to_seg_tx,
            cam_stop_clone,
            cam_ctx,
        ));
        let initial_audio_status;
        let audio_capture_thread =
            match live_audio::start_audio_capture(raw_samples_tx, audio_cap_stop) {
                Ok((s, r, c)) => {
                    initial_audio_status = LiveAudioStatus::Running(r, c);
                    Some(s)
                }
                Err(e) => {
                    let m = format!("Audio capture failed: {}", e);
                    error!("{}", m);
                    initial_audio_status = LiveAudioStatus::Error(m);
                    None
                }
            };
        let audio_processor_thread = match initial_audio_status {
            LiveAudioStatus::Running(r, c) => {
                let mut p =
                    music::AudioProcessor::new(raw_samples_rx, intensities_tx, r, c, MAX_TRACKS);
                Some(std::thread::spawn(move || p.run(audio_proc_stop)))
            }
            _ => {
                warn!("No audio proc started.");
                None
            }
        };
        let seg_thread = Some(segmentation::start_segmentation_thread(
            seg_to_ui_tx,
            cam_to_seg_rx,
            user_interaction_rx,
            intensities_rx,
            seg_stop_clone,
            seg_ctx,
            model_options,
        )); // Pass user_interaction_rx

        Self {
            texture: None,
            seg_to_ui_rx,
            _user_interaction_tx, // Store sender
            cam_thread_handle: cam_thread,
            cam_stop_signal,
            seg_thread_handle: seg_thread,
            seg_stop_signal,
            audio_capture_thread_handle: audio_capture_thread,
            audio_capture_stop_signal,
            audio_processor_thread_handle: audio_processor_thread,
            audio_processor_stop_signal,
            camera_error: None,
            seg_error: None,
            live_audio_status: initial_audio_status,
            camera_resolution: None,
            texture_size: None,
            last_fps_update_time: Instant::now(),
            frames_since_last_update: 0,
            last_calculated_fps: 0.0,
        }
    }

    fn update_fps_counter(&mut self) {
        /* Unchanged */
        self.frames_since_last_update += 1;
        let n = Instant::now();
        let e = n.duration_since(self.last_fps_update_time);
        if e >= FPS_UPDATE_INTERVAL {
            self.last_calculated_fps =
                self.frames_since_last_update as f32 / e.as_secs_f32().max(0.001);
            self.frames_since_last_update = 0;
            self.last_fps_update_time = n;
        }
    }
}

impl eframe::App for WebcamAppUI {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps_counter();
        let mut received_frame_this_update = false;
        loop {
            match self.seg_to_ui_rx.try_recv() {
                Ok(msg) => match msg {
                    SegmentationThreadMsg::Frame(f) => {
                        received_frame_this_update = true;
                        let s = f.size;
                        let sz = Vec2::new(s[0] as f32, s[1] as f32);
                        if self.camera_resolution.map_or(true, |r| {
                            r.width() != s[0] as u32 || r.height() != s[1] as u32
                        }) {
                            self.camera_resolution =
                                Some(Resolution::new(s[0] as u32, s[1] as u32));
                            self.texture_size = Some(sz);
                        }
                        match self.texture {
                            Some(ref mut t) => t.set(ImageData::Color(f), TextureOptions::LINEAR),
                            None => {
                                self.texture = Some(ctx.load_texture(
                                    "vis",
                                    ImageData::Color(f),
                                    TextureOptions::LINEAR,
                                ))
                            }
                        }
                        self.seg_error = None;
                    }
                    SegmentationThreadMsg::Error(e) => {
                        if self.seg_error.as_ref() != Some(&e) {
                            self.seg_error = Some(e);
                        }
                    }
                },
                Err(TryRecvError::Empty) => {
                    break;
                }
                Err(TryRecvError::Disconnected) => {
                    let m = "Seg disconnected.".to_string();
                    if self.seg_error.is_none() {
                        self.seg_error = Some(m);
                    }
                    if let Some(h) = self.seg_thread_handle.take() {
                        let _ = h.join();
                    }
                    break;
                }
            }
        }

        // --- Simplified UI ---
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                if !cfg!(target_arch = "wasm32") {
                    ui.menu_button("File", |ui| {
                        if ui.button("Quit").clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    });
                    ui.add_space(16.0);
                }
                widgets::global_theme_preference_buttons(ui);
            });
        });

        egui::SidePanel::left("control_panel")
            .resizable(false)
            .default_width(180.0)
            .show(ctx, |ui| {
                ui.heading("Audio Status")
                    .on_hover_text("Status of live audio capture");
                ui.separator();
                match &self.live_audio_status {
                    /* Display status */
                    LiveAudioStatus::Initializing => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("Initializing...");
                        });
                    }
                    LiveAudioStatus::Running(r, c) => {
                        ui.label(format!("Capturing: {} Hz, {} ch", r, c));
                    }
                    LiveAudioStatus::Error(e) => {
                        ui.colored_label(Color32::RED, "Error").on_hover_text(e);
                    }
                    LiveAudioStatus::Disabled => {
                        ui.colored_label(Color32::GRAY, "Disabled/Stopped");
                    }
                }
                ui.separator();
                ui.heading("Info")
                    .on_hover_text("Performance and status details");
                ui.separator();
                ui.label(format!("UI FPS: {:.1}", self.last_calculated_fps));
                match &self.camera_resolution {
                    Some(r) => {
                        ui.label(format!("Cam Res: {}x{}", r.width(), r.height()));
                    }
                    None if self.camera_error.is_none()
                        && self.seg_error.is_none()
                        && self.texture.is_none() =>
                    {
                        ui.label("Cam Res: Initializing...");
                    }
                    None if self.camera_error.is_some() || self.seg_error.is_some() => {
                        ui.label("Cam Res: Error");
                    }
                    _ => {
                        ui.label("Cam Res: Waiting...");
                    }
                }
                if let Some(err) = &self.camera_error {
                    ui.separator();
                    ui.colored_label(Color32::YELLOW, "Camera Error:")
                        .on_hover_text(err);
                    ui.small(err);
                }
                if let Some(err) = &self.seg_error {
                    ui.separator();
                    ui.colored_label(Color32::RED, "Processing Error:")
                        .on_hover_text(err);
                    ui.small(err);
                }
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            match &self.texture {
                Some(texture) => {
                    if let Some(tex_size) = self.texture_size {
                        let ar = if tex_size.y > 0.0 {
                            tex_size.x / tex_size.y
                        } else {
                            1.0
                        };
                        let aw = ui.available_width();
                        let ah = ui.available_height();
                        let mut iw = aw;
                        let mut ih = aw / ar;
                        if ih > ah {
                            ih = ah;
                            iw = ah * ar;
                        }
                        let ds = Vec2::new(iw, ih);
                        ui.with_layout(Layout::top_down(Align::Center), |ui| {
                            let sized_texture = egui::load::SizedTexture::new(texture.id(), ds);
                            ui.add(egui::Image::new(sized_texture)); // Display image, no interaction
                        });
                    } else {
                        ui.centered_and_justified(|ui| ui.label("Texture size unknown."));
                    }
                }
                None if self.live_audio_status != LiveAudioStatus::Disabled
                    && self.camera_error.is_none()
                    && self.seg_error.is_none() =>
                {
                    ui.centered_and_justified(|ui| {
                        ui.spinner();
                        ui.label("Initializing stream...");
                    });
                }
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.colored_label(ui.visuals().error_fg_color, "Stream unavailable");
                        ui.label("Check status panel.");
                    });
                }
            }
        });

        if !received_frame_this_update {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    } 

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {

        info!("Exit requested...");
        self.cam_stop_signal.store(true, Ordering::Relaxed);
        self.seg_stop_signal.store(true, Ordering::Relaxed);
        self.audio_capture_stop_signal
            .store(true, Ordering::Relaxed);
        self.audio_processor_stop_signal
            .store(true, Ordering::Relaxed);
        info!("Stop signals sent.");
        if let Some(stream) = self.audio_capture_thread_handle.take() {
            if let Err(e) = stream.pause() {
                error!("Error pausing audio stream: {}", e);
            }
            drop(stream);
            info!("Audio stream dropped.");
        }
        if let Some(h) = self.cam_thread_handle.take() {
            info!("Joining cam...");
            if let Err(e) = h.join() {
                error!("Cam join err: {:?}", e);
            }
        }
        if let Some(h) = self.audio_processor_thread_handle.take() {
            info!("Joining audio proc...");
            if let Err(e) = h.join() {
                error!("Audio proc join err: {:?}", e);
            }
        }
        if let Some(h) = self.seg_thread_handle.take() {
            info!("Joining seg...");
            if let Err(e) = h.join() {
                error!("Seg join err: {:?}", e);
            }
        }
        info!("All threads stopped/joined.");
    }
}

trait CenteredJustified {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui));
}
impl CenteredJustified for egui::Ui {
    fn centered_and_justified(&mut self, add_contents: impl FnOnce(&mut egui::Ui)) {
        self.with_layout(egui::Layout::top_down(egui::Align::Center), |ui| {
            ui.add_space(ui.available_height() / 3.0);
            add_contents(ui);
        });
    }
}
