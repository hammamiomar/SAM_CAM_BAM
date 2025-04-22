// src/main.rs
#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod camera;
mod segmentation; 
mod ui;
mod music;
mod live_audio;
mod visuals;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Starting SAM_CAM_BAM (EdgeSAM Periodic)");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([300.0, 200.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SAM_CAM_BAM", 
        native_options,
        Box::new(|cc| Ok(Box::new(ui::WebcamAppUI::new(cc)))),
    )
}
