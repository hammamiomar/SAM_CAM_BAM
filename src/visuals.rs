// src/visuals.rs
use image::{ImageBuffer, Luma, Rgb, RgbImage}; 
use imageproc::rect::Rect;
use rand::Rng;
use std::f32::consts::PI;

const WHITE: Rgb<u8> = Rgb([255, 255, 255]);
const BLACK: Rgb<u8> = Rgb([0, 0, 0]);

#[inline]
fn lerp_color(c1: Rgb<u8>, c2: Rgb<u8>, t: f32) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    Rgb([
        (c1[0] as f32 * (1.0 - t) + c2[0] as f32 * t).round() as u8,
        (c1[1] as f32 * (1.0 - t) + c2[1] as f32 * t).round() as u8,
        (c1[2] as f32 * (1.0 - t) + c2[2] as f32 * t).round() as u8,
    ])
}

#[inline]
fn is_inside_mask(x: i32, y: i32, mask: &ImageBuffer<Luma<u8>, Vec<u8>>) -> bool {
    if x < 0 || y < 0 {
        return false;
    }
    mask.get_pixel_checked(x as u32, y as u32)
        .map_or(false, |p| p[0] > 128)
}

#[inline]
fn spatial_noise(x: f32, y: f32, seed: f32) -> f32 {
    let val = (x * 12.9898 + y * 78.233 + seed * 123.456).sin() * 43758.5453;
    val - val.floor()
}

// --- Bass Visualization - "Expanding Pulse / Heartbeat" 
fn draw_bass_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32,
    frame_count: u64,
    _animation_phase: f32,
    _rng: &mut impl Rng,
) {
    let center_x = bbox_rect.left() + bbox_rect.width() as i32 / 2;
    let center_y = bbox_rect.top() + bbox_rect.height() as i32 / 2;
    let max_dim = (bbox_rect.width().max(bbox_rect.height())) as f32;
    let color_low = Rgb([80, 0, 10]);
    let color_high = lerp_color(Rgb([255, 0, 0]), Rgb([255, 100, 0]), intensity);
    let pulse_speed = 0.02 + intensity * 0.05;
    let phase = (frame_count as f32 * pulse_speed) % 1.0;
    let ring_center_norm = phase;
    let ring_thickness_norm = (0.1 + intensity * 0.2).clamp(0.01, 0.5); // Ensure non-zero thickness
    let background_color = lerp_color(BLACK, color_low, intensity * 0.3);

    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                let dx = x as f32 - center_x as f32;
                let dy = y as f32 - center_y as f32;
                let dist_from_center = (dx * dx + dy * dy).sqrt();
                let dist_norm = dist_from_center / (max_dim * 0.5).max(1.0);
                let dist_from_ring_center = (dist_norm - ring_center_norm).abs();
                let ring_value = (1.0 - (dist_from_ring_center / ring_thickness_norm))
                    .clamp(0.0, 1.0)
                    .powi(2);
                let final_color = lerp_color(background_color, color_high, ring_value);
                display_image.put_pixel(x as u32, y as u32, final_color);
            }
        }
    }
}

// --- Mid Visualization - "Swirling Vortex / Galaxy"
fn draw_mid_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32, // 0.0 to 1.0
    frame_count: u64,
    animation_phase: f32, // Use for rotation base
    rng: &mut impl Rng,
) {
    let center_x = bbox_rect.left() as f32 + bbox_rect.width() as f32 / 2.0;
    let center_y = bbox_rect.top() as f32 + bbox_rect.height() as f32 / 2.0;
    let max_dist = (bbox_rect.width().max(bbox_rect.height()) as f32 * 0.7).max(1.0);

    // Palette: Greens, Blues, Purples swirling
    let color1 = Rgb([0, 200, 50]); // Green
    let color2 = Rgb([0, 150, 200]); // Cyan/Blue
    let color3 = Rgb([100, 50, 200]); // Purple

    // Intensity effects
    let rotation_speed = 0.01 + intensity * 0.05;
    let noise_amount = 0.1 + intensity * 0.4; // How much noise distorts the swirl
    let brightness_boost = intensity * 0.5; // Boost overall brightness

    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                // --- Calculate coordinates relative to center ---
                let rel_x = x as f32 - center_x;
                let rel_y = y as f32 - center_y;
                let dist = (rel_x * rel_x + rel_y * rel_y).sqrt();
                let mut angle = rel_y.atan2(rel_x); // Current angle

                // --- Add swirl based on distance and time/phase ---
                // Rotate more closer to the center, speed based on intensity
                let rotation_factor = (1.0 - (dist / max_dist)).powi(2); // Rotate more near center
                let rotation_amount =
                    animation_phase + frame_count as f32 * rotation_speed * rotation_factor;
                angle += rotation_amount;

                // --- Add noise distortion to angle and distance ---
                let noise_seed = frame_count as f32 * 0.01;
                let noise_val = spatial_noise(rel_x * 0.05, rel_y * 0.05, noise_seed);
                angle += (noise_val - 0.5) * PI * 0.3 * noise_amount; // Distort angle
                let noisy_dist = dist
                    * (1.0
                        + (spatial_noise(rel_x * 0.02, rel_y * 0.02, noise_seed + 10.0) - 0.5)
                            * 0.4
                            * noise_amount);

                // --- Map angle and distance to color ---
                let angle_norm = (angle / (2.0 * PI) + 10.0) % 1.0; // Normalize angle 0-1 (add offset to avoid seam issues)
                let dist_norm = (noisy_dist / max_dist).clamp(0.0, 1.0);

                // Blend colors based on angle
                let color_mix;
                if angle_norm < 0.333 {
                    color_mix = lerp_color(color1, color2, angle_norm / 0.333);
                } else if angle_norm < 0.666 {
                    color_mix = lerp_color(color2, color3, (angle_norm - 0.333) / 0.333);
                } else {
                    color_mix = lerp_color(color3, color1, (angle_norm - 0.666) / 0.333);
                }

                // Fade to black at edges and based on distance noise
                let fade = (1.0 - dist_norm.powf(1.5)) * (1.0 - (noise_val * 0.5 * noise_amount)); // Fade near edge and by noise
                let base_color = lerp_color(BLACK, color_mix, fade.clamp(0.0, 1.0));

                // Add brightness boost and random sparkles
                let mut final_color = lerp_color(base_color, WHITE, brightness_boost * fade); // Boost brightness towards center
                if rng.gen::<f32>() < 0.005 * intensity {
                    // Sparse sparkles
                    final_color = lerp_color(final_color, WHITE, 0.8);
                }

                display_image.put_pixel(x as u32, y as u32, final_color);
            }
        }
    }
}

// --- High Visualization - "Electric Static Field / Jagged Lines" 
fn draw_high_visuals(
    display_image: &mut RgbImage,
    mask_image: &ImageBuffer<Luma<u8>, Vec<u8>>,
    bbox_rect: Rect,
    intensity: f32,
    frame_count: u64,
    _animation_phase: f32,
    rng: &mut impl Rng,
) {
    let color_low = Rgb([0, 0, 50]);
    let color_mid = Rgb([100, 50, 255]);
    let color_high = WHITE;
    let field_intensity = intensity.powi(2);
    let line_intensity = intensity.sqrt();
    let noise_seed1 = frame_count as f32 * 0.1;
    let noise_seed2 = frame_count as f32 * -0.07;

    for y in bbox_rect.top()..bbox_rect.bottom() {
        for x in bbox_rect.left()..bbox_rect.right() {
            if is_inside_mask(x, y, mask_image) {
                let noise_val1 = spatial_noise(x as f32 * 0.08, y as f32 * 0.08, noise_seed1);
                let noise_val2 = spatial_noise(x as f32 * 0.03, y as f32 * 0.03, noise_seed2);
                let combined_noise = (noise_val1 * 0.6 + noise_val2 * 0.4 + rng.gen::<f32>() * 0.2
                    - 0.1)
                    .clamp(0.0, 1.0);
                let field_color =
                    lerp_color(color_low, color_mid, combined_noise * field_intensity * 1.5);

                let num_lines = 8.0 + line_intensity * 20.0;
                let line_phase = frame_count as f32 * 0.15;
                let line_y_norm =
                    (y as f32 / bbox_rect.height() as f32 * num_lines + line_phase) % 1.0; // Use bbox height
                let jag_noise_scale = 0.1;
                let jag_noise = spatial_noise(
                    x as f32 * jag_noise_scale,
                    y as f32 * jag_noise_scale,
                    noise_seed1 + 10.0,
                );
                let line_threshold = 0.05 + jag_noise * 0.1;
                let line_brightness = if line_y_norm < line_threshold {
                    (1.0 - line_y_norm / line_threshold) * line_intensity.powi(2) * 1.5
                } else {
                    0.0
                };

                let final_color =
                    lerp_color(field_color, color_high, line_brightness.clamp(0.0, 1.0));
                display_image.put_pixel(x as u32, y as u32, final_color);
            }
        }
    }
}

// --- Main Public Function --- RESTORED DISPATCHER ---
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
    // Dimension check (important!)
    if mask_image.dimensions() != display_image.dimensions() {
        // This might happen if camera resolution changes mid-stream before masks update?
        // log::warn!("Visuals: Mask ({:?}) and Display ({:?}) dimension mismatch!", mask_image.dimensions(), display_image.dimensions());
        return; // Skip drawing if dimensions mismatch
    }
    if bbox_rect.width() == 0 || bbox_rect.height() == 0 {
        return;
    } // Skip empty rects

    match slot_index {
        0 => draw_bass_visuals(
            display_image,
            mask_image,
            bbox_rect,
            intensity,
            frame_count,
            animation_phase,
            rng,
        ),
        1 => draw_mid_visuals(
            display_image,
            mask_image,
            bbox_rect,
            intensity,
            frame_count,
            animation_phase,
            rng,
        ),
        2 => draw_high_visuals(
            display_image,
            mask_image,
            bbox_rect,
            intensity,
            frame_count,
            animation_phase,
            rng,
        ),
        _ => {
            log::warn!("Visuals: Invalid slot {}", slot_index);
        }
    }
}
