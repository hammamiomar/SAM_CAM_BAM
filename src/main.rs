// src/main.rs
#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

mod camera;
mod segmentation;
mod ui;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("Starting SAM_CAM_BAM");

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([800.0, 600.0])
            .with_min_inner_size([300.0, 200.0]),
        ..Default::default()
    };

    eframe::run_native(
        "SAM_CAM_BAM", // Updated title
        native_options,
        // Box::new(|cc| Ok(Box::new(app::WebCamApp::new(cc)))), // Old way
        Box::new(|cc| Ok(Box::new(ui::WebcamAppUI::new(cc)))), // Use the struct from ui.rs
    )
}

// --- WASM main remains the same, but would also need to instantiate ui::WebcamAppUI ---
#[cfg(target_arch = "wasm32")]
fn main() {
     // ... (wasm setup as before) ...
     log::info!("Starting SAM_CAM_BAM (WASM)"); // Log WASM start

    wasm_bindgen_futures::spawn_local(async {
        let start_result = eframe::WebRunner::new()
            .start(
                "the_canvas_id", // Or your canvas ID
                web_options,
                 // Instantiate the UI struct here too, though camera part won't work easily
                 Box::new(|cc| Ok(Box::new(ui::WebcamAppUI::new(cc)))),
            )
            .await;
        // ... (rest of wasm setup) ...
    });
}