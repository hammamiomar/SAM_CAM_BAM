// src/music.rs
use rodio::{Decoder, Source};
use rustfft::{FftPlanner, num_complex::Complex};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use std::time::Duration;
use anyhow::{Context, Result, anyhow};
use log::{error, warn, info, debug};

const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = FFT_SIZE / 4;
const SMOOTHING_FACTOR: f32 = 0.2;
const LOW_BAND_END_BIN: usize = 10;
const MID_BAND_END_BIN: usize = 100;

pub struct AudioAnalyzer {
    decoder: Option<Decoder<BufReader<File>>>,
    sample_rate: u32,
    channels: u16,
    fft_planner: FftPlanner<f32>,
    fft_buffer: Vec<Complex<f32>>,
    scratch_buffer: Vec<Complex<f32>>,
    window: Vec<f32>,
    sample_buffer: Vec<f32>,
    smoothed_intensities: Vec<f32>,
}

impl AudioAnalyzer {
    pub fn new() -> Self {
        info!("Initializing AudioAnalyzer (no audio loaded initially).");
        Self {
            decoder: None,
            sample_rate: 0,
            channels: 0,
            fft_planner: FftPlanner::new(),
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            scratch_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            window: apodize::hanning_iter(FFT_SIZE).map(|v| v as f32).collect(),
            sample_buffer: Vec::with_capacity(FFT_SIZE * 2),
            smoothed_intensities: vec![0.0; 3],
        }
    }

    pub fn update_audio_source(&mut self, audio_file_path: &str) -> Result<()> {
        info!("Attempting to load audio source: {}", audio_file_path);
        self.decoder = None;
        self.sample_buffer.clear();
        self.smoothed_intensities.fill(0.0);

        let path = Path::new(audio_file_path);
        if !path.exists() {
            return Err(anyhow!("Audio file not found: {}", audio_file_path));
        }

        let file = File::open(path)
            .with_context(|| format!("Failed to open audio file: {}", audio_file_path))?;
        let decoder = Decoder::new(BufReader::new(file))
            .with_context(|| format!("Failed to create audio decoder for: {}", audio_file_path))?;

        self.sample_rate = decoder.sample_rate();
        self.channels = decoder.channels();
        info!("Audio loaded: {} Hz, {} channels", self.sample_rate, self.channels);

        if self.sample_rate < 8000 {
            warn!("Sample rate {} Hz is very low, frequency band analysis might be inaccurate.", self.sample_rate);
        }

        self.decoder = Some(decoder);
        Ok(())
    }

    // Revised get_next_samples using `if let` for clarity on mutable borrow
    fn get_next_samples(&mut self, count: usize) -> Result<Option<Vec<f32>>> {
        if let Some(mut decoder) = self.decoder.as_mut() { // decoder is &mut Decoder<_>
            let mut samples_collected = Vec::with_capacity(count);
            while samples_collected.len() < count {
                match decoder.next() { // Calling next() on &mut Decoder should be fine
                    Some(sample) => {
                        let sample_f32 = sample as f32 / i16::MAX as f32;
                        samples_collected.push(sample_f32);
                        for _ in 1..self.channels {
                            if decoder.next().is_none() { break; } // Also fine
                        }
                    }
                    None => {
                        debug!("End of audio stream reached in get_next_samples.");
                        // Signal EOF, let caller handle looping/reload
                        return Ok(None);
                    }
                }
            }
            // If loop completes, we collected 'count' samples
            Ok(Some(samples_collected))
        } else {
            // No decoder loaded
            Ok(None)
        }
    }


    pub fn get_band_intensities(&mut self) -> Vec<f32> {
        // If no decoder or error occurs, return default zeros
        let mut should_loop = false; // Flag to indicate EOF was reached

        // 1. Fill/Slide buffer
        if self.sample_buffer.len() < FFT_SIZE {
             match self.get_next_samples(FFT_SIZE - self.sample_buffer.len()) {
                Ok(Some(new_samples)) => self.sample_buffer.extend(new_samples),
                Ok(None) => { should_loop = true; } // Mark for loop/clear
                Err(e) => { error!("Error reading samples: {}", e); self.decoder = None; return vec![0.0; 3]; }
            }
            // If still not enough samples after trying to fill (e.g., very short file), return last known
             if self.sample_buffer.len() < FFT_SIZE && !should_loop {
                return self.smoothed_intensities.clone();
            }
        } else {
             // Slide window
             self.sample_buffer.drain(0..HOP_SIZE);
             match self.get_next_samples(HOP_SIZE) {
                Ok(Some(new_samples)) => self.sample_buffer.extend(new_samples),
                Ok(None) => { should_loop = true; } // Mark for loop/clear
                Err(e) => { error!("Error reading samples: {}", e); self.decoder = None; return vec![0.0; 3]; }
            }
            // If not enough samples after sliding, return last known
            if self.sample_buffer.len() < FFT_SIZE && !should_loop {
                return self.smoothed_intensities.clone();
            }
        }

        // Handle EOF -> loop audio by clearing buffer
        if should_loop {
             info!("Audio source reached end, looping (clearing buffer).");
             self.sample_buffer.clear();
             return vec![0.0; 3]; // Return zeros for this frame as buffer is reset
        }

        // Ensure buffer has exactly FFT_SIZE samples (should be guaranteed by logic above unless error occurred)
        if self.sample_buffer.len() != FFT_SIZE {
             error!("Sample buffer size ({}) != FFT size ({}). Returning defaults.", self.sample_buffer.len(), FFT_SIZE);
             return vec![0.0; 3];
        }

        // 2. Prepare buffer for FFT
        for (i, sample) in self.sample_buffer.iter().enumerate() {
            let windowed_sample = *sample * self.window.get(i).copied().unwrap_or(1.0);
            self.fft_buffer[i] = Complex::new(windowed_sample, 0.0);
        }

        // 3. Perform FFT
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        fft.process_with_scratch(&mut self.fft_buffer, &mut self.scratch_buffer);

        // 4. Calculate Magnitudes
        let magnitudes: Vec<f32> = self.fft_buffer[0..FFT_SIZE / 2]
            .iter()
            .map(|c| c.norm())
            .collect();

        // 5. Calculate Band Averages
        let calc_avg = |start_bin: usize, end_bin: usize| -> f32 {
            let end_bin = end_bin.min(magnitudes.len());
            if start_bin >= end_bin { return 0.0; }
            let slice = &magnitudes[start_bin..end_bin];
            let len = slice.len() as f32;
            if len == 0.0 { 0.0 } else { slice.iter().sum::<f32>() / len }
        };

        let low_avg = calc_avg(1, LOW_BAND_END_BIN);
        let mid_avg = calc_avg(LOW_BAND_END_BIN, MID_BAND_END_BIN);
        let high_avg = calc_avg(MID_BAND_END_BIN, FFT_SIZE / 2);

        // 6. Normalize and Smooth
        let max_mag = magnitudes.iter().fold(0.0f32, |a, &b| a.max(b));
        let normalize = |avg: f32| if max_mag > 1e-3 { (avg / max_mag).sqrt().clamp(0.0, 1.0) } else { 0.0 };

        let intensities = vec![
            normalize(low_avg),
            normalize(mid_avg),
            normalize(high_avg),
        ];

        for i in 0..3 {
            self.smoothed_intensities[i] = self.smoothed_intensities[i] * (1.0 - SMOOTHING_FACTOR)
                                           + intensities[i] * SMOOTHING_FACTOR;
        }

        self.smoothed_intensities.clone()
    }
}

// Helper for Hanning window
mod apodize {
    pub fn hanning_iter(len: usize) -> impl Iterator<Item = f64> {
        if len <= 1 {
             vec![1.0].into_iter()
         } else {
             (0..len).map(move |x| 0.5 * (1.0 - (2.0 * std::f64::consts::PI * x as f64 / (len - 1) as f64).cos())).collect::<Vec<_>>().into_iter()
         }
    }
}