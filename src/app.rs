use std::{ptr::NonNull, sync::Arc};

use egui::{Color32, ColorImage, ImageData, TextureHandle, TextureOptions};
use image::{self, ImageBuffer, Rgb};
use nokhwa::{pixel_format::RgbFormat, utils::{CameraIndex, RequestedFormat, RequestedFormatType}, Camera};
pub struct WebCamApp {
    // Example stuff:
    screen_texture: TextureHandle,
    camera: Camera,

}


impl WebCamApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
       
    let screen_texture = cc.egui_ctx.load_texture(
        "webcam",
        ImageData::Color(Arc::new(ColorImage::new([1280,720], Color32::TRANSPARENT))),
        TextureOptions::default(),
        );

    //nokhwa::nokhwa_initialize(|work|println!("umm {work}"));
    // first camera in system
    let index = CameraIndex::Index(0); 
    // request the absolute highest resolution CameraFormat that can be decoded to RGB.
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestResolution);
    // make the camera
    let mut camera = Camera::new(index, requested).unwrap();
    camera.open_stream().unwrap();
        
    Self { screen_texture: screen_texture, camera }
    }

    pub fn capture_frame(&mut self) -> ImageBuffer<Rgb<u8>, Vec<u8>>{
        // get a frame
        let frame = self.camera.frame().unwrap();
        println!("Captured Single Frame of {}", frame.buffer().len());
        println!("Resolution {} ", frame.resolution());
        // decode into an ImageBuffer
        let decoded = frame.decode_image::<RgbFormat>();

        let img = match decoded {
            Ok(img)=> img,
            Err(e) => panic!("cant decode the image, {e} ")
        };
        println!("height is {} and width is{}",img.height(),img.width());
        return img

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
                ColorImage::from_rgb([1920, 1080], &img.into_raw()),
                TextureOptions::default(),
            );
        };
        });
        
}
}