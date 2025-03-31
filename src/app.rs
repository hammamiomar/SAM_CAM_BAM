use std::sync::Arc;

use egui::{Color32, ColorImage, ImageData, TextureHandle, TextureOptions};
use image::{self, ImageBuffer, Rgb};
pub struct WebCamApp {
    // Example stuff:
    screen_texture: TextureHandle,

}


impl WebCamApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
       
       let screen_texture = cc.egui_ctx.load_texture(
        "webcam",
        ImageData::Color(Arc::new(ColorImage::new([1280,720], Color32::TRANSPARENT))),
    TextureOptions::default(),
    );
    Self { screen_texture: screen_texture }
    }

    pub fn capture_frame(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>>{
        let mut img = image::RgbImage::new(1280, 720);
        for x in 15..=17 {
            for y in 8..24 {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
                img.put_pixel(y, x, image::Rgb([255, 0, 0]));
            }
        }
        img

    }
}

impl eframe::App for WebCamApp {
    /// Called by the frame work to save state before shutdown.


    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::menu::bar(ui, |ui| {
                // NOTE: no File->Quit on web pages!
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

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::both().show(ui, |ui|{ 
            // The central panel the region left after adding TopPanel's and SidePanel's
            ui.heading("SAM_CAM_BAM");
                

            ui.horizontal(|ui| {
                ui.add(
                egui::Image::new(&self.screen_texture)
                    .max_height(1920.0)
                    .max_width(1080.0)
                    .rounding(10.0),
                );
            });
        });
        if ui.button("take a pic").clicked(){
            let img = self.capture_frame();
            self.screen_texture.set(
                ColorImage::from_rgb([1280, 720], &img.into_raw()),
                TextureOptions::default(),
            );
        };
        });
        
}
}