// src/main.rs
#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

// Declare the modules that main.rs will use
mod camera;
mod segmentation; // Ensure this is declared
mod ui;
mod music;
mod live_audio;
mod visuals;

// When compiling natively:
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
        "SAM_CAM_BAM Stream (EdgeSAM - Click)", // Updated title
        native_options,
        Box::new(|cc| Ok(Box::new(ui::WebcamAppUI::new(cc)))), // Use the struct from ui.rs
    )
}

// --- WASM main remains the same ---
#[cfg(target_arch = "wasm32")]
fn main() {
     // ... (wasm setup as before, likely won't work correctly) ...
}