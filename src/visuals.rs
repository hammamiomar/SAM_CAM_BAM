// src/visuals.rs
use image::{ImageBuffer, Luma, Pixel, Rgb, RgbImage};
use imageproc::rect::Rect;
use rand::Rng;
use std::f32::consts::PI;

// --- Constants ---
const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const BLACK: Rgb<u8> = Rgb([0, 0, 0]);

// --- Helper: Lerp Color (Linear Interpolation) ---
#[inline]
fn lerp_color(c1: Rgb<u8>, c2: Rgb<u8>, t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    Rgb([
        (c1[0] as f32 * (1.0 - t) + c2[0] as f32 * t).round() as u8,
        (c1[1] as f32 * (1.0 - t) + c2[1] as f32 * t).round() as u8,
        (c1[2] as f32 * (1.0 - t) + c2[2] as f32 * t).round() as u8,
    ])
}

// --- Helper: Check if inside mask ---
#[inline]
fn is_inside_mask(x: i32, y: i32, mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> bool {
    mask.get_pixel_checked(x as u32, y as u32)
        .map_or(false, |p| p[0] > 128)
}

// --- Bass Visualization ---
fn draw_bass_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32,
    _frame_count: u64, // Keep for potential future use
    _animation_phase: f32, // Keep for potential future use
    _rng: &mut impl Rng, // Keep for potential future use
) {
    let center_x = bbox_rect.left() + bbox_rect.width() as i32 / 2;
    let center_y = bbox_rect.top() + bbox_rect.height() as i32 / 2;
    let max_dist = (bbox_rect.width().max(bbox_rect.height()) as f32 * 0.7) + 1.0; // Diagonal distance approx

    // Palette: Deep Red -> Bright Red -> Orange -> Yellow
    let color_near = lerp_color(Rgb([255, 60, 0]), Rgb([255, 200, 0]), intensity.powi(2)); // Orange/Yellow pulse
    let color_far = lerp_color(Rgb([50, 0, 0]), Rgb([150, 0, 0]), intensity); // Dark to mid red

    // 1. Radial Gradient Fill (pulsing brightness/color)
    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                let dx = (x - center_x) as f32;
                let dy = (y - center_y) as f32;
                let dist = (dx * dx + dy * dy).sqrt();
                let gradient_t = (dist / max_dist).powf(0.8); // Non-linear falloff
                let base_viz_color = lerp_color(color_near, color_far, gradient_t);
                display_image.put_pixel(x as u32, y as u32, base_viz_color);
            }
        }
    }

    // 2. Thick Waveform Line (drawn over gradient)
    let wave_amplitude = bbox_rect.height() as f32 * 0.8 * intensity; // Thicker amplitude
    let wave_frequency = PI * 2.0 / (bbox_rect.width() as f32 * 0.8); // One cycle across 80% width
    let wave_phase = _animation_phase * 2.0; // Slow movement
    let line_color = lerp_color(Rgb([255, 150, 50]), WHITE, intensity); // Orange/White line
    let line_thickness = (1.0 + 4.0 * intensity) as i32; // Thickness from 1 to 5 pixels

    let wave_base_y = center_y;
    for x in bbox_rect.left()..bbox_rect.right() {
        let y_offset = (wave_amplitude * (x as f32 * wave_frequency + wave_phase).sin()) as i32;
        let y_center = wave_base_y + y_offset;

        // Draw thick line vertically
        for y_off in -line_thickness..=line_thickness {
             let y = y_center + y_off;
             // Simple anti-alias based on distance from center of thickness
             let thickness_t = 1.0 - (y_off as f32 / line_thickness as f32).abs();
             let current_line_color = lerp_color(BLACK, line_color, thickness_t.powi(2)); // Fade edge

             if is_inside_mask(x, y, mask_image) {
                 // Blend with the background gradient
                 let existing_pixel = display_image.get_pixel(x as u32, y as u32);
                 let final_color = lerp_color(*existing_pixel, current_line_color, 0.8 * thickness_t); // Blend 80% line color
                 display_image.put_pixel(x as u32, y as u32, final_color);
             }
        }
    }
}


// --- Mid Visualization ---
fn draw_mid_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32,
    frame_count: u64,
    animation_phase: f32,
    rng: &mut impl Rng,
) {
    // Palette: Green -> Cyan -> Yellow-Green
    let color1 = lerp_color(Rgb([0, 180, 50]), Rgb([0, 255, 150]), intensity); // Green -> Cyanish
    let color2 = lerp_color(Rgb([150, 255, 0]), Rgb([50, 200, 50]), intensity); // Yellow-Green -> Greenish

    // Angle for linear gradient based on time/intensity
    let angle_rad = (frame_count as f32 * 0.02 + intensity * PI).sin() * (PI / 4.0); // Oscillate +/- 45 deg
    let (sin_a, cos_a) = (angle_rad.sin(), angle_rad.cos());
    let center_x = (bbox_rect.left() + bbox_rect.width() as i32 / 2) as f32;
    let center_y = (bbox_rect.top() + bbox_rect.height() as i32 / 2) as f32;
    let max_proj_dist = (bbox_rect.width() as f32 * cos_a.abs() + bbox_rect.height() as f32 * sin_a.abs()) * 0.5 + 1.0;


    // 1. Linear Gradient Fill (dynamic angle)
    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                // Project pixel onto the gradient direction vector
                let rel_x = x as f32 - center_x;
                let rel_y = y as f32 - center_y;
                let projected_dist = rel_x * cos_a + rel_y * sin_a;
                let gradient_t = (projected_dist / max_proj_dist + 1.0) * 0.5; // Normalize to 0-1
                let base_viz_color = lerp_color(color1, color2, gradient_t.clamp(0.0, 1.0));
                display_image.put_pixel(x as u32, y as u32, base_viz_color);
            }
        }
    }

    // 2. Flowing Lines Texture Overlay
    let line_density = (intensity * 15.0).ceil() as i32; // More lines with intensity
    let line_speed = animation_phase * 20.0; // Faster movement
    let line_color = lerp_color(color2, WHITE, 0.6); // Use lighter color for lines

    for i in 0..line_density {
         // Horizontal flowing lines
         let flow_y_base = bbox_rect.height() as f32 * (i as f32 / line_density as f32);
         let flow_y_offset = (bbox_rect.height() as f32 * 0.1 * (line_speed * 0.1 + i as f32 * 0.5).sin());
         let flow_y = bbox_rect.top() + (flow_y_base + flow_y_offset) as i32;

         // Vertical flowing lines
         let flow_x_base = bbox_rect.width() as f32 * (i as f32 / line_density as f32);
         let flow_x_offset = (bbox_rect.width() as f32 * 0.1 * (line_speed * 0.08 + i as f32 * 0.7).cos());
         let flow_x = bbox_rect.left() + (flow_x_base + flow_x_offset) as i32;

         // Draw horizontal line segment
         for x in bbox_rect.left()..bbox_rect.right() {
             if flow_y >= bbox_rect.top() && flow_y < bbox_rect.bottom() && is_inside_mask(x, flow_y, mask_image) {
                 let existing = display_image.get_pixel(x as u32, flow_y as u32);
                 display_image.put_pixel(x as u32, flow_y as u32, lerp_color(*existing, line_color, 0.4)); // Blend weakly
             }
         }
         // Draw vertical line segment
          for y in bbox_rect.top()..bbox_rect.bottom() {
             if flow_x >= bbox_rect.left() && flow_x < bbox_rect.right() && is_inside_mask(flow_x, y, mask_image) {
                  let existing = display_image.get_pixel(flow_x as u32, y as u32);
                  display_image.put_pixel(flow_x as u32, y as u32, lerp_color(*existing, line_color, 0.4)); // Blend weakly
             }
         }
    }
}


// --- High Visualization ---
fn draw_high_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32,
    frame_count: u64, // Use frame_count for some dynamic behavior
    _animation_phase: f32, // Mark as unused
    rng: &mut impl Rng,
) {
    // Palette: Electric Blue -> Purple -> White flash
    let base_blue = Rgb([0, 50, 255]);
    let purple = Rgb([180, 0, 255]);
    let flash_color = WHITE; // Flash pure white

    // Base color shifts towards purple/white based on intensity
    let dynamic_base = lerp_color(base_blue, purple, intensity.powi(2));

    // 1. Optional: Keep a subtle noisy background fill
    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                let noise_factor = rng.gen::<f32>() * 0.2 - 0.1; // Less intense noise: -0.1 to +0.1
                let noise_intensity_factor = 1.0 + noise_factor * intensity;
                let r = (dynamic_base[0] as f32 * noise_intensity_factor).clamp(0.0, 255.0) as u8;
                let g = (dynamic_base[1] as f32 * noise_intensity_factor).clamp(0.0, 255.0) as u8;
                let b = (dynamic_base[2] as f32 * noise_intensity_factor).clamp(0.0, 255.0) as u8;
                // Blend slightly with black for depth
                let base_viz_color = lerp_color(BLACK, Rgb([r, g, b]), 0.8);
                display_image.put_pixel(x as u32, y as u32, base_viz_color);
            }
        }
    }

    // 2. Jagged Vertical Lines / EQ Bars
    let num_lines = (bbox_rect.width() as f32 / 4.0).max(3.0).min(50.0) as i32; // Lines every ~4 pixels, min 3, max 50
    let line_spacing = bbox_rect.width() as f32 / num_lines as f32;
    let max_height = bbox_rect.height() as f32;

    for i in 0..num_lines {
        let line_x = bbox_rect.left() + (i as f32 * line_spacing + line_spacing * 0.5) as i32;

        // Base height determined by intensity
        let base_height = max_height * intensity;

        // Add dynamic flicker/jaggedness based on intensity and frame count/randomness
        let flicker_intensity = intensity.powi(2); // Make flicker stronger at high intensity
        let random_flicker = rng.gen::<f32>() * 0.6 - 0.3; // Random height variation +/- 30%
        let time_flicker = (frame_count as f32 * 0.5 + i as f32 * 1.5).sin() * 0.2; // Temporal variation +/- 20%
        let total_flicker = (random_flicker + time_flicker) * flicker_intensity;

        let current_height = (base_height * (1.0 + total_flicker)).clamp(0.0, max_height);
        let line_top = bbox_rect.bottom() - current_height as i32; // Lines grow from bottom
        let line_bottom = bbox_rect.bottom();

        // Color transitions towards white flash based on intensity and flicker peak
        let color_intensity = (intensity + total_flicker.max(0.0) * 0.5).clamp(0.0, 1.0); // Boost color intensity with positive flicker
        let line_color = lerp_color(dynamic_base, flash_color, color_intensity.powi(3)); // Fast transition to white

        // Draw the vertical line segment
        for y in line_top..line_bottom {
            if is_inside_mask(line_x, y, mask_image) {
                // Blend with background for softer look? Or overwrite for sharpness? Let's overwrite.
                display_image.put_pixel(line_x as u32, y as u32, line_color);
                // Optional: Thicker line?
                 if is_inside_mask(line_x + 1, y, mask_image) {
                      display_image.put_pixel((line_x + 1) as u32, y as u32, line_color);
                 }
            }
        }
    }
}


// --- Main Public Function ---
pub fn draw_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    slot_index: usize,
    intensity: f32,
    frame_count: u64,
    animation_phase: f32,
    rng: &mut impl Rng,
) {
    // Ensure mask dimensions match (should be guaranteed by segmentation part)
    if mask_image.dimensions() != display_image.dimensions() {
        log::warn!("Visuals: Mask dimension mismatch!");
        return;
    }

    match slot_index {
        0 => draw_bass_visuals(display_image, mask_image, bbox_rect, intensity, frame_count, animation_phase, rng),
        1 => draw_mid_visuals(display_image, mask_image, bbox_rect, intensity, frame_count, animation_phase, rng),
        2 => draw_high_visuals(display_image, mask_image, bbox_rect, intensity, frame_count, animation_phase, rng),
        _ => { /* Should not happen */ }
    }
}