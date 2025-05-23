# SAM_CAM_BAM

An interactive music visualizer that combines computer vision and audio processing to create dynamic visual effects using webcam input and real-time audio analysis.

## Demo

🎥 **[Watch the demo video](https://youtu.be/13qkvewvES0)**

## Motivation

This project was created to:
- Learn Rust low-level programming and systems development
- Build a more interactive and engaging music visualizer
- Explore the intersection of computer vision and audio processing

## How It Works

SAM_CAM_BAM works by:
1. **Video Input**: Captures webcam images in real-time
2. **Segmentation**: Processes images through ONNX FastSAM for object segmentation
3. **Audio Analysis**: Splits audio into 3 frequency bands (bass, mids, highs)
4. **Visual Effects**: Draws dynamic masks on segmented objects based on music intensity
5. **Tracking**: Randomly selects and tracks different segments for varied visual effects

## Requirements

### macOS Setup (Currently Mac-only)
- **Camera Patch**: Run `cargo patch-crate` to enable macOS webcam support
- **Audio**: Install [BlackHole](https://existential.audio/blackhole/) for audio input/output routing
- **Note**: Currently configured specifically for Mac camera hardware

### Installation
```bash
# Apply the required patch for macOS webcam support
cargo patch-crate

# Run the application
cargo run --release
```

## Technical Details

- **Segmentation**: Uses ONNX FastSAM for real-time object segmentation
- **Audio Processing**: Real-time frequency band analysis (bass, mids, highs)
- **Tracking**: Random segment selection for dynamic visual variety
- **Framework**: Built with Rust using egui/eframe

## Future Work

- **Improved Tracking**: Enhance segmentation tracking algorithms
- **Web Deployment**: Use WebGPU for ONNX processing to deploy in browsers, allowing people to use it directly in their web browser

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.