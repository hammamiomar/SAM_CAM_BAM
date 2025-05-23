// src/music.rs
use crossbeam_channel::{Receiver, Sender, TryRecvError};
use log::{debug, error, info};
use rustfft::{num_complex::Complex, FftPlanner};
use std::collections::VecDeque;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::Duration; // For moving average normalization

const FFT_SIZE: usize = 512;
const HOP_SIZE: usize = FFT_SIZE / 4;
const SMOOTHING_FACTOR: f32 = 0.15;
const NORM_WINDOW_SIZE: usize = 50; // Number of frames for moving max window

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
    sample_buffer: Vec<f32>,
    smoothed_intensities: Vec<f32>,
    // --- State for adaptive normalization ---
    recent_max_low: VecDeque<f32>,
    recent_max_mid: VecDeque<f32>,
    recent_max_high: VecDeque<f32>,
    // --- Store bin indices ---
    low_bin_range: (usize, usize),
    mid_bin_range: (usize, usize),
    high_bin_range: (usize, usize),
}

impl AudioProcessor {
    pub fn new(
        raw_sample_receiver: Receiver<Vec<f32>>,
        intensity_sender: Sender<Vec<f32>>,
        sample_rate: u32,
        channels: u16,
        num_bands: usize,
    ) -> Self {
        info!(
            "Initializing AudioProcessor: SampleRate={}, Channels={}, Bands={}",
            sample_rate, channels, num_bands
        );
        let window = apodize::hanning_iter(FFT_SIZE).map(|v| v as f32).collect();

        // --- Calculate bin ranges based on sample rate ---
        let bin_width = sample_rate as f32 / FFT_SIZE as f32;
        // Adjust Hz ranges as needed
        let low_start_hz = 50.0; // Start a bit higher to avoid DC/rumble
        let low_end_hz = 180.0;
        let mid_start_hz = low_end_hz;
        let mid_end_hz = 3000.0; // Extend mids slightly?
        let high_start_hz = mid_end_hz;
        let high_end_hz = (sample_rate as f32 / 2.0) * 0.9; // Go up to 90% of Nyquist

        let low_bin_start = (low_start_hz / bin_width).round() as usize;
        let low_bin_end = (low_end_hz / bin_width).round() as usize;
        let mid_bin_start = (mid_start_hz / bin_width).round() as usize;
        let mid_bin_end = (mid_end_hz / bin_width).round() as usize;
        let high_bin_start = (high_start_hz / bin_width).round() as usize;
        let high_bin_end = (high_end_hz / bin_width).round().min((FFT_SIZE / 2) as f32) as usize; // Don't exceed max bin

        info!(
            "FFT Bin Ranges (Approx Hz): Bass [{:.1}-{:.1}], Mid [{:.1}-{:.1}], High [{:.1}-{:.1}]",
            low_bin_start as f32 * bin_width,
            low_bin_end as f32 * bin_width,
            mid_bin_start as f32 * bin_width,
            mid_bin_end as f32 * bin_width,
            high_bin_start as f32 * bin_width,
            high_bin_end as f32 * bin_width
        );

        Self {
            raw_sample_receiver,
            intensity_sender,
            sample_rate,
            channels,
            fft_planner: FftPlanner::new(),
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            scratch_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            window,
            sample_buffer: Vec::with_capacity(FFT_SIZE * 2),
            smoothed_intensities: vec![0.0; num_bands],
            // Initialize normalization windows with small non-zero value
            recent_max_low: VecDeque::from(vec![1e-3; NORM_WINDOW_SIZE]),
            recent_max_mid: VecDeque::from(vec![1e-3; NORM_WINDOW_SIZE]),
            recent_max_high: VecDeque::from(vec![1e-3; NORM_WINDOW_SIZE]),
            // Store bin ranges
            low_bin_range: (low_bin_start.max(1), low_bin_end), // Ensure start >= 1
            mid_bin_range: (mid_bin_start, mid_bin_end),
            high_bin_range: (high_bin_start, high_bin_end),
        }
    }

    // --- Helper to update and get moving maximum ---
    fn update_and_get_moving_max(window: &mut VecDeque<f32>, new_value: f32) -> f32 {
        window.push_back(new_value.max(1e-6)); // Add new value (ensure non-zero)
        if window.len() > NORM_WINDOW_SIZE {
            window.pop_front();
        }
        // Find the maximum in the current window
        window.iter().fold(0.0f32, |max, &val| max.max(val))
    }

    pub fn run(&mut self, stop_signal: Arc<AtomicBool>) {
        info!("AudioProcessor thread started.");
        let fft = self.fft_planner.plan_fft_forward(FFT_SIZE);
        let mut last_send_time = std::time::Instant::now();
        let mut frame_counter = 0; // For debug logging interval

        while !stop_signal.load(Ordering::Relaxed) {
            // 1. Receive samples (unchanged)
            match self.raw_sample_receiver.try_recv() {
                Ok(new_samples) => {
                    /* Downmix */
                    if self.channels > 1 {
                        let n = new_samples.len() / self.channels as usize;
                        let mut m = Vec::with_capacity(n);
                        for i in 0..n {
                            let s = i * self.channels as usize;
                            let o = new_samples[s..(s + self.channels as usize)]
                                .iter()
                                .sum::<f32>()
                                / self.channels as f32;
                            m.push(o);
                        }
                        self.sample_buffer.extend(m);
                    } else {
                        self.sample_buffer.extend(new_samples);
                    }
                }
                Err(TryRecvError::Empty) => {
                    /* Handle empty */
                    thread::sleep(IDLE_SLEEP_DURATION);
                    if self.sample_buffer.is_empty()
                        && last_send_time.elapsed() > Duration::from_millis(100)
                    {
                        self.smoothed_intensities.fill(0.0);
                        let _ = self
                            .intensity_sender
                            .try_send(self.smoothed_intensities.clone());
                        last_send_time = std::time::Instant::now();
                    }
                    continue;
                }
                Err(TryRecvError::Disconnected) => {
                    error!("Audio capture disconnected.");
                    break;
                }
            }

            // 2. Process buffer
            while self.sample_buffer.len() >= FFT_SIZE {
                frame_counter += 1;
                // Prepare FFT input (unchanged)
                for (i, s) in self.sample_buffer[0..FFT_SIZE].iter().enumerate() {
                    let w = s * self.window.get(i).copied().unwrap_or(1.0);
                    self.fft_buffer[i] = Complex::new(w, 0.0);
                }
                fft.process_with_scratch(&mut self.fft_buffer, &mut self.scratch_buffer);
                let magnitudes: Vec<f32> = self.fft_buffer[0..FFT_SIZE / 2]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .collect(); // Use norm_sqr (cheaper)

                // Calculate band averages (using stored ranges)
                let calc_avg = |start_bin: usize, end_bin: usize| -> f32 {
                    let start_bin = start_bin.min(magnitudes.len());
                    let end_bin = end_bin.min(magnitudes.len());
                    if start_bin >= end_bin {
                        return 0.0;
                    }
                    let slice = &magnitudes[start_bin..end_bin];
                    if slice.is_empty() {
                        0.0
                    } else {
                        slice.iter().sum::<f32>() / slice.len() as f32
                    }
                };

                let low_avg_sq = calc_avg(self.low_bin_range.0, self.low_bin_range.1);
                let mid_avg_sq = calc_avg(self.mid_bin_range.0, self.mid_bin_range.1);
                let high_avg_sq = calc_avg(self.high_bin_range.0, self.high_bin_range.1);

                // --- Adaptive Normalization ---
                let max_low = Self::update_and_get_moving_max(&mut self.recent_max_low, low_avg_sq);
                let max_mid = Self::update_and_get_moving_max(&mut self.recent_max_mid, mid_avg_sq);
                let max_high =
                    Self::update_and_get_moving_max(&mut self.recent_max_high, high_avg_sq);

                // Normalize each band relative to its own recent maximum
                // Take sqrt AFTER normalization for better perceptual scaling
                let norm_low = (low_avg_sq / max_low).sqrt().clamp(0.0, 1.0);
                let norm_mid = (mid_avg_sq / max_mid).sqrt().clamp(0.0, 1.0);
                let norm_high = (high_avg_sq / max_high).sqrt().clamp(0.0, 1.0);

                // Optional: Debug print occasionally
                if frame_counter % 100 == 0 {
                    // Print every 100 FFT frames approx
                    debug!(
                        "AvgSq (L,M,H): {:.4}, {:.4}, {:.4}",
                        low_avg_sq, mid_avg_sq, high_avg_sq
                    );
                    debug!(
                        "Max (L,M,H):   {:.4}, {:.4}, {:.4}",
                        max_low, max_mid, max_high
                    );
                    debug!(
                        "Norm (L,M,H):  {:.2}, {:.2}, {:.2}",
                        norm_low, norm_mid, norm_high
                    );
                }

                let current_intensities = [norm_low, norm_mid, norm_high];

                // Apply smoothing
                for i in 0..self
                    .smoothed_intensities
                    .len()
                    .min(current_intensities.len())
                {
                    self.smoothed_intensities[i] = self.smoothed_intensities[i]
                        * (1.0 - SMOOTHING_FACTOR)
                        + current_intensities[i] * SMOOTHING_FACTOR;
                }

                // Send smoothed intensities
                match self
                    .intensity_sender
                    .try_send(self.smoothed_intensities.clone())
                {
                    Ok(_) => {
                        last_send_time = std::time::Instant::now();
                    }
                    Err(crossbeam_channel::TrySendError::Full(_)) => {}
                    Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                        error!("Seg thread disconnected.");
                        stop_signal.store(true, Ordering::Relaxed);
                        break;
                    }
                }

                self.sample_buffer.drain(0..HOP_SIZE);
            }
            if stop_signal.load(Ordering::Relaxed) {
                break;
            }
        }
        info!("AudioProcessor thread finished.");
    }
}

mod apodize {
    pub fn hanning_iter(len: usize) -> impl Iterator<Item = f64> {
        if len <= 1 {
            vec![1.0].into_iter()
        } else {
            (0..len)
                .map(move |x| {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * x as f64 / (len - 1) as f64).cos())
                })
                .collect::<Vec<_>>()
                .into_iter()
        }
    }
}
