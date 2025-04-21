// src/music.rs
use rustfft::{num_complex::Complex, FftPlanner};
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use log::{debug, error, info, warn};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration;


const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = FFT_SIZE / 4; // Determines how much overlap/update rate
const SMOOTHING_FACTOR: f32 = 0.15; // Adjust for more/less smoothing
// Adjust these bin boundaries based on sample rate and desired feel
const LOW_BAND_END_BIN: usize = 10;     // ~ 0-215 Hz for 44.1kHz
const MID_BAND_END_BIN: usize = 100;   // ~ 215-2150 Hz for 44.1kHz
// HIGH_BAND starts from MID_BAND_END_BIN to FFT_SIZE / 2

// How long to wait if no audio data comes in before checking stop signal again
const IDLE_SLEEP_DURATION: Duration = Duration::from_millis(10);

pub struct AudioProcessor {
    raw_sample_receiver: Receiver<Vec<f32>>,
    intensity_sender: Sender<Vec<f32>>,
    sample_rate: u32,
    channels: u16,
    fft_planner: FftPlanner<f32>,
    fft_buffer: Vec<Complex<f32>>,
    scratch_buffer: Vec<Complex<f32>>,
    window: Vec<f32>,
    // Buffer to hold incoming samples between FFT calculations
    sample_buffer: Vec<f32>,
    smoothed_intensities: Vec<f32>,
}

impl AudioProcessor {
    pub fn new(
        raw_sample_receiver: Receiver<Vec<f32>>,
        intensity_sender: Sender<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
        num_bands: usize, // Pass the number of bands expected (e.g., MAX_TRACKS)
    ) -> Self {
        info!(
            "Initializing AudioProcessor: SampleRate={}, Channels={}, Bands={}",
            sample_rate, channels, num_bands
        );
        // Pre-calculate Hanning window
        let window = apodize::hanning_iter(FFT_SIZE).map(|v| v as f32).collect();

        Self {
            raw_sample_receiver,
            intensity_sender,
            sample_rate,
            channels,
            fft_planner: FftPlanner::new(),
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            scratch_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE], // Scratch for FFT algorithm
            window,
            sample_buffer: Vec::with_capacity(FFT_SIZE * 2), // Capacity for ~2 FFT frames
            smoothed_intensities: vec![0.0; num_bands], // Initialize with zeros
        }
    }

    // This method will run in its own thread
    pub fn run(&mut self, stop_signal: Arc<AtomicBool>) {
        info!("AudioProcessor thread started.");
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        let mut last_send_time = std::time::Instant::now();

        while !stop_signal.load(Ordering::Relaxed) {
            // 1. Receive new audio samples
            match self.raw_sample_receiver.try_recv() {
                Ok(mut new_samples) => {
                    // Simple downmixing: average channels if necessary
                    if self.channels > 1 {
                        let num_frames = new_samples.len() / self.channels as usize;
                        let mut mono_samples = Vec::with_capacity(num_frames);
                        for frame_idx in 0..num_frames {
                            let start = frame_idx * self.channels as usize;
                            let end = start + self.channels as usize;
                            let frame = &new_samples[start..end];
                            let mono_sample = frame.iter().sum::<f32>() / self.channels as f32;
                            mono_samples.push(mono_sample);
                        }
                        self.sample_buffer.extend(mono_samples);
                    } else {
                        self.sample_buffer.extend(new_samples);
                    }
                }
                Err(TryRecvError::Empty) => {
                    // No new data, sleep briefly to avoid busy-waiting
                    thread::sleep(IDLE_SLEEP_DURATION);
                    // If buffer is empty and no new data, maybe send zeros after a timeout?
                    if self.sample_buffer.is_empty() && last_send_time.elapsed() > Duration::from_millis(100) {
                         self.smoothed_intensities.fill(0.0);
                         let _ = self.intensity_sender.try_send(self.smoothed_intensities.clone());
                         last_send_time = std::time::Instant::now();
                    }
                    continue; // Go back to check stop signal and try receiving again
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Audio capture channel disconnected. Stopping AudioProcessor.");
                    break; // Exit the loop
                }
            }

            // 2. Process buffer if enough samples are available
            while self.sample_buffer.len() >= FFT_SIZE {
                // Prepare the FFT input buffer
                for (i, sample) in self.sample_buffer[0..FFT_SIZE].iter().enumerate() {
                     // Apply window function
                     let windowed_sample = *sample * self.window.get(i).copied().unwrap_or(1.0);
                     self.fft_buffer[i] = Complex::new(windowed_sample, 0.0);
                }

                // Perform FFT
                // Use process_with_scratch for potentially better performance
                fft.process_with_scratch(&mut self.fft_buffer, &mut self.scratch_buffer);

                // Calculate magnitudes (only need the first half due to symmetry)
                let magnitudes: Vec<f32> = self.fft_buffer[0..FFT_SIZE / 2]
                    .iter()
                    .map(|c| c.norm()) // Magnitude = sqrt(real^2 + imag^2)
                    .collect();

                // Calculate band averages (adjust bins as needed)
                 let bin_width = self.sample_rate as f32 / FFT_SIZE as f32;
                 let low_end_bin = (5.0 / bin_width) as usize; // Start slightly above DC offset
                 let low_band_end_bin = (150.0 / bin_width) as usize; // Example: Bass up to 150 Hz
                 let mid_band_end_bin = (2000.0 / bin_width) as usize; // Example: Mids up to 2000 Hz
                 // Highs are from mid_band_end_bin to FFT_SIZE / 2

                 let calc_avg = |start_bin: usize, end_bin: usize| -> f32 {
                     let end_bin = end_bin.min(magnitudes.len()); // Ensure within bounds
                     if start_bin >= end_bin { return 0.0; }
                     let slice = &magnitudes[start_bin..end_bin];
                     let len = slice.len() as f32;
                     if len == 0.0 { 0.0 } else { slice.iter().sum::<f32>() / len }
                 };

                 let low_avg = calc_avg(low_end_bin.max(1), low_band_end_bin);
                 let mid_avg = calc_avg(low_band_end_bin, mid_band_end_bin);
                 let high_avg = calc_avg(mid_band_end_bin, FFT_SIZE / 2);

                 // Normalize and Smooth (consider perceptual loudness - sqrt is a simple approximation)
                 let max_mag = magnitudes.iter().fold(0.0f32, |a, &b| a.max(b));
                 let normalize = |avg: f32| if max_mag > 1e-4 { (avg / max_mag).sqrt().clamp(0.0, 1.0) } else { 0.0 }; // Add epsilon

                 let current_intensities = [
                     normalize(low_avg),
                     normalize(mid_avg),
                     normalize(high_avg),
                 ];

                 // Apply smoothing (Exponential Moving Average)
                 for i in 0..self.smoothed_intensities.len() {
                     self.smoothed_intensities[i] = self.smoothed_intensities[i] * (1.0 - SMOOTHING_FACTOR)
                                                    + current_intensities[i] * SMOOTHING_FACTOR;
                 }

                 // Send the smoothed intensities to the segmentation thread
                 // Use try_send to avoid blocking if the segmentation thread is slow
                 match self.intensity_sender.try_send(self.smoothed_intensities.clone()) {
                    Ok(_) => { /* Sent */ last_send_time = std::time::Instant::now(); }
                    Err(crossbeam_channel::TrySendError::Full(_)) => {
                        warn!("Intensity channel full, skipping send.");
                        // Don't update last_send_time, let the next successful send do it
                    }
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                         error!("Segmentation thread channel disconnected. Stopping AudioProcessor.");
                         stop_signal.store(true, Ordering::Relaxed); // Signal stop
                         break; // Exit inner loop
                     }
                 }

                // Remove HOP_SIZE samples from the beginning of the buffer to slide the window
                self.sample_buffer.drain(0..HOP_SIZE);
            } // End while sample_buffer.len() >= FFT_SIZE
            if stop_signal.load(Ordering::Relaxed){ break; } // Check stop signal after inner loop too
        } // End while !stop_signal

        info!("AudioProcessor thread finished.");
    }
}

// Helper for Hanning window (unchanged)
mod apodize {
    pub fn hanning_iter(len: usize) -> impl Iterator<Item = f64> {
        if len <= 1 {
             vec![1.0].into_iter()
         } else {
             (0..len).map(move |x| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * x as f64 / (len - 1) as f64).cos())).collect::<Vec<_>>().into_iter()
         }
    }
}