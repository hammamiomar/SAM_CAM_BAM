// src/segmentation.rs
use std::{
    sync::{atomic::{AtomicBool, Ordering}, mpsc::{Receiver, Sender}, Arc}, 
    thread::{self,JoinHandle}};
use egui::ColorImage;
use image::{ImageBuffer, Rgb, RgbImage};
use log::{error, info}; // Use trace for potentially high-frequency logs

use usls::{models::YOLO, Annotator, DataLoader, Options};

use crate::camera::CameraThreadMsg;

pub enum SegmentationThreadMsg{
    Frame(Arc<ColorImage>),
    Error(String)
}

pub fn start_segmentation_thread(
    msg_sender: Sender<SegmentationThreadMsg>,
    msg_receiver: Receiver<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context,
) -> JoinHandle<()>{
    info!("Starting segmentation thread");
    thread::spawn(move || {
        segmentation_loop(msg_sender,msg_receiver,stop_signal,ctx);
    })
}

fn segmentation_loop(
    msg_sender: Sender<SegmentationThreadMsg>,
    msg_receiver: Receiver<CameraThreadMsg>,
    stop_signal: Arc<AtomicBool>,
    ctx: egui::Context
){
    let config = Options::fastsam_s()
        .with_model_device("mps".try_into().unwrap())
        .commit()
        .unwrap();

    let mut model = YOLO::new(config).unwrap();


    while !stop_signal.load(Ordering::Relaxed){
        match msg_receiver.try_recv(){
            Ok(msg) => match msg{
                CameraThreadMsg::Frame(frame) => {
                    let ys = model.forward(frame).unwrap();

                    // annotate
                    let annotator = Annotator::default()
                    .without_masks(true)
                    .with_bboxes_thickness(3)
                    .with_saveout("fastsam");
                    annotator.annotate(&xs, &ys);
                }
                CameraThreadMsg::Error(err) =>{
                    panic!();
                }
            }
            Err(_) => {
                error!("Camera oopsies..");
                break;
            }
        }
    }
}