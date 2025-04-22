// src/live_audio.rs
use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{BuildStreamError, Device, SampleFormat, Stream, StreamConfig};
use crossbeam_channel::Sender;
use log::{error, info, warn};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const VIRTUAL_DEVICE_NAME: &str = "BlackHole"; //Using Blackhole for audio input/output

// Helper function
fn try_build_input_stream(
    device: &Device,
    raw_sample_sender: &Sender<Vec<f32>>,
    stop_signal: &Arc<AtomicBool>,
) -> Result<(Stream, u32, u16), anyhow::Error> {
    // Get the device name early for logging and cloning
    let device_name = device.name().unwrap_or_else(|_| "Unnamed Device".into());
    info!(
        "Attempting to find suitable input config for: {}",
        device_name
    );

    let mut supported_configs_range = device
        .supported_input_configs()
        .with_context(|| format!("Error querying supported input configs for {}", device_name))?;

    let supported_config = supported_configs_range
        .find(|c| c.sample_format() == SampleFormat::F32) 
        .or_else(|| {
            warn!(
                "F32 sample format not supported on {}, trying any supported format...",
                device_name
            );
            device
                .supported_input_configs()
                .ok()
                .and_then(|mut iter| iter.next())
        })
        .ok_or_else(|| anyhow!("No supported input config found for {}", device_name))?
        .with_max_sample_rate();

    if supported_config.sample_format() != SampleFormat::F32 {
        return Err(anyhow!(
            "Selected config for {} is not F32 ({:?}). Requires conversion.",
            device_name,
            supported_config.sample_format()
        ));
    }

    let config: StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0;
    let channels = config.channels;
    info!(
        "Selected config for {}: SampleRate={}, Channels={}, Format=F32",
        device_name, sample_rate, channels
    );

    // --- Clone variables needed for the closures ---
    let sender_clone = raw_sample_sender.clone();
    let stop_signal_clone = stop_signal.clone();
    let device_name_for_data_closure = device_name.clone(); 
    let device_name_for_err_closure = device_name.clone(); 

    // --- Error Callback Closure ---
    let err_fn = move |err| {
        error!(
            "An error occurred on the audio stream for {}: {}",
            device_name_for_err_closure, err
        );
    };
    
    // --- Data Callback Closure ---
    let data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
        if stop_signal_clone.load(Ordering::Relaxed) {
            return;
        }
        match sender_clone.try_send(data.to_vec()) {
            Ok(_) => { /* Sent */ }
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                warn!(
                    "Audio channel full, dropping data from {}.",
                    device_name_for_data_closure
                );
            }
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {}
        }
    };

    // --- Build the Stream ---
    let stream = device.build_input_stream(
        &config, data_fn, 
        err_fn, 
        None,    
    )?;

    info!("Audio stream built successfully for {}.", device_name);
    Ok((stream, sample_rate, channels))
}

pub fn start_audio_capture(
    raw_sample_sender: Sender<Vec<f32>>,
    stop_signal: Arc<AtomicBool>,
) -> Result<(Stream, u32, u16), anyhow::Error> {
    info!("Initializing audio capture...");

    let host = cpal::default_host();
    info!("Audio host: {}", host.id().name());

    let mut preferred_device: Option<Device> = None;

    // --- 1. Look for Virtual Device (e.g., BlackHole) ---
    info!(
        "Searching for virtual audio device containing '{}'...",
        VIRTUAL_DEVICE_NAME
    );
    match host.input_devices() {
        Ok(devices) => {
            for device in devices {
                if let Ok(name) = device.name() {
                    if name.contains(VIRTUAL_DEVICE_NAME) {
                        info!("Found potential virtual device: {}", name);
                        preferred_device = Some(device);
                        break; // Use the first one found
                    }
                }
            }
        }
        Err(e) => {
            warn!(
                "Error enumerating input devices: {}. Proceeding without virtual device search.",
                e
            );
        }
    }

    // --- 2. Try Preferred Device (BlackHole) if found ---
    if let Some(ref dev) = preferred_device {
        let dev_name = dev
            .name()
            .unwrap_or_else(|_| VIRTUAL_DEVICE_NAME.to_string());
        info!(
            "Attempting capture on preferred virtual device: {}",
            dev_name
        );
        match try_build_input_stream(dev, &raw_sample_sender, &stop_signal) {
            Ok(result) => {
                info!("Capture successful on virtual device '{}'.", dev_name);
                result
                    .0
                    .play()
                    .context("Failed to start audio stream on virtual device")?;
                info!("Audio stream started playing (capturing from virtual device).");
                return Ok(result);
            }
            Err(e) => {
                warn!(
                    "Failed to use preferred virtual device '{}': {}. Falling back.",
                    dev_name, e
                );
                // Proceed to default input fallback
            }
        }
    } else {
        info!(
            "Virtual device '{}' not found. Falling back to default input.",
            VIRTUAL_DEVICE_NAME
        );
        warn!("For loopback audio capture (computer output), install '{}' and configure Audio MIDI Setup.", VIRTUAL_DEVICE_NAME);
    }

    // --- 3. Fallback to Default Input Device (Microphone) ---
    let input_device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("No default input device found."))?;
    let dev_name = input_device
        .name()
        .unwrap_or_else(|_| "Default Input Device".into());
    info!(
        "Attempting capture on fallback default input device: {}",
        dev_name
    );
    match try_build_input_stream(&input_device, &raw_sample_sender, &stop_signal) {
        Ok(result) => {
            info!(
                "Capture successful on default input device (likely microphone): '{}'.",
                dev_name
            );
            result
                .0
                .play()
                .context("Failed to start audio stream on default input device")?;
            info!("Audio stream started playing (capturing from default input).");
            return Ok(result);
        }
        Err(e) => {
            error!(
                "Failed to capture on default input device '{}': {}",
                dev_name, e
            );
            if let Some(build_err) = e.downcast_ref::<BuildStreamError>() {
                if matches!(build_err, BuildStreamError::StreamConfigNotSupported) {
                    error!("(Reason: Stream type not supported - this often happens when trying loopback on devices that don't support it directly).");
                }
            }
            Err(e.context(format!(
                "Failed to build stream on default input device ({})",
                dev_name
            )))
        }
    }
}
