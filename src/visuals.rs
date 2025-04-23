// src/visuals.rs
use crate::segmentation::VisualObjectData;
use egui::{epaint::{PathShape, PathStroke, CircleShape}, Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2};
// Import Rng trait for methods like gen_range
use rand::Rng;
use std::f32::consts::PI;

const WHITE: Color32 = Color32::WHITE;
const BLACK: Color32 = Color32::BLACK;
const TRANSPARENT: Color32 = Color32::TRANSPARENT;

#[inline]
fn lerp_color32(c1: Color32, c2: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgba_premultiplied(
        (c1.r() as f32 * (1.0 - t) + c2.r() as f32 * t).round() as u8,
        (c1.g() as f32 * (1.0 - t) + c2.g() as f32 * t).round() as u8,
        (c1.b() as f32 * (1.0 - t) + c2.b() as f32 * t).round() as u8,
        (c1.a() as f32 * (1.0 - t) + c2.a() as f32 * t).round() as u8,
    )
}

#[inline]
fn map_point(
    p: Pos2,
    img_w: f32,
    img_h: f32,
    image_rect: Rect,
) -> Pos2 {
    if img_w <= 0.0 || img_h <= 0.0 { return image_rect.center(); }
    let norm_x = p.x / img_w;
    let norm_y = p.y / img_h;
    image_rect.min + Vec2::new(norm_x * image_rect.width(), norm_y * image_rect.height())
}

fn map_contour(
    contour: &[Pos2],
    img_w: f32,
    img_h: f32,
    image_rect: Rect,
) -> Vec<Pos2> {
    contour.iter().map(|&p| map_point(p, img_w, img_h, image_rect)).collect()
}

#[inline]
fn spatial_noise(x: f32, y: f32, seed: f32) -> f32 {
    ( (x * 0.1 + seed).sin() + (y * 0.1 + seed * 1.2).cos() + (seed * 123.45).sin() ).cos().powi(2)
}

fn create_mask_path(
    contours: &[Vec<Pos2>],
    img_w: f32,
    img_h: f32,
    image_rect: Rect,
    fill_color: Color32,
    stroke: Stroke,
) -> Option<PathShape> {
    if contours.is_empty() || contours[0].is_empty() {
        return None;
    }
    let mapped_contours: Vec<Vec<Pos2>> = contours
        .iter()
        .map(|c| map_contour(c, img_w, img_h, image_rect))
        .filter(|c| c.len() > 2)
        .collect();
    if mapped_contours.is_empty() {
        return None;
    }
    let largest_contour = mapped_contours.iter().max_by_key(|c| c.len()).unwrap();
    let path_stroke = PathStroke::new(stroke.width, stroke.color);
    Some(PathShape {
        points: largest_contour.clone(),
        closed: true,
        fill: fill_color,
        stroke: path_stroke,
    })
}

fn draw_bass_visuals_epaint(
    painter: &Painter,
    image_rect: Rect,
    img_w: f32,
    img_h: f32,
    object: &VisualObjectData,
    frame_count: u64,
    _rng: &mut impl Rng, // Prefix unused variable
) {
    let color_low = Color32::from_rgb(80, 0, 10);
    let color_peak = lerp_color32(Color32::from_rgb(255, 0, 0), Color32::from_rgb(255, 100, 0), object.intensity);
    let background_color = lerp_color32(TRANSPARENT, color_low.linear_multiply(0.5), object.intensity * 0.6);

    let mask_path = create_mask_path(
        &object.contours, img_w, img_h, image_rect, background_color, Stroke::NONE,
    );

    if let Some(path) = mask_path {
        painter.add(Shape::Path(path));
        let pulse_speed = 0.02 + object.intensity * 0.05;
        let phase = (frame_count as f32 * pulse_speed + object.animation_phase) % 1.0;
        let ring_center_norm = phase;
        let ring_thickness_norm = (0.1 + object.intensity * 0.2).clamp(0.05, 0.5);
        let ring_outer_norm = (ring_center_norm + ring_thickness_norm * 0.5).min(1.0);
        let ring_inner_norm = (ring_center_norm - ring_thickness_norm * 0.5).max(0.0);
        let object_bbox_screen = Rect {
             min: map_point(object.bbox.min, img_w, img_h, image_rect),
             max: map_point(object.bbox.max, img_w, img_h, image_rect),
        };
        let center = object_bbox_screen.center();
        let max_radius_pixels = object_bbox_screen.size().min_elem() * 0.5;
        let outer_radius = max_radius_pixels * ring_outer_norm;
        let inner_radius = max_radius_pixels * ring_inner_norm;
        if outer_radius > inner_radius && outer_radius > 0.5 {
             let pulse_color = lerp_color32(color_low, color_peak, (1.0 - phase) * object.intensity.sqrt());
             let outer_circle = CircleShape::filled(center, outer_radius, pulse_color);
             painter.add(Shape::Circle(outer_circle));
             let inner_circle = CircleShape::filled(center, inner_radius, TRANSPARENT);
             painter.add(Shape::Circle(inner_circle));
        }
    }
}

fn draw_mid_visuals_epaint(
     painter: &Painter,
    image_rect: Rect,
    img_w: f32,
    img_h: f32,
    object: &VisualObjectData,
    frame_count: u64,
    _rng: &mut impl Rng, // Prefix unused variable
) {
    let color1 = Color32::from_rgb(0, 200, 50);
    let color2 = Color32::from_rgb(0, 150, 200);
    let color3 = Color32::from_rgb(100, 50, 200);
    let rotation_speed = 0.005 + object.intensity * 0.02;
    let drift_speed = 0.5 + object.intensity * 2.0;
    let particle_count = (50.0 + object.intensity * 250.0) as usize;
    let particle_radius = (1.0 + object.intensity * 3.0).clamp(0.5, 4.0);
    let brightness_boost = object.intensity * 0.4;
    let object_bbox_screen = Rect {
         min: map_point(object.bbox.min, img_w, img_h, image_rect),
         max: map_point(object.bbox.max, img_w, img_h, image_rect),
    };
    let center = object_bbox_screen.center();
    let max_dist = object_bbox_screen.size().max_elem() * 0.5;
    let mask_fill_color = BLACK.linear_multiply(0.3 + object.intensity * 0.3);
    let mask_path = create_mask_path(
         &object.contours, img_w, img_h, image_rect, mask_fill_color, Stroke::NONE
    );
    if let Some(path) = mask_path {
         painter.add(Shape::Path(path));
    }
    for i in 0..particle_count {
        let id = i as f32;
        let initial_angle = (id * 1.37) % (2.0 * PI);
        let initial_dist_norm = ((id / particle_count as f32).sqrt() + object.animation_phase * 0.1) % 1.0 ;
        let initial_dist = initial_dist_norm * max_dist;
        let swirl_factor = (1.0 - initial_dist_norm).powi(2);
        let current_angle = initial_angle + object.animation_phase + frame_count as f32 * rotation_speed * swirl_factor;
        let noise_seed = frame_count as f32 * 0.01 + id * 0.1;
        let noise_val = spatial_noise(current_angle.cos(), current_angle.sin(), noise_seed);
        let drift = (noise_val - 0.5) * drift_speed * (0.5 + object.intensity);
        let current_dist = (initial_dist + drift).max(0.0);
        let current_dist_clamped = current_dist.min(max_dist);
        let particle_pos = center + Vec2::angled(current_angle) * current_dist_clamped;
        if !object_bbox_screen.contains(particle_pos) { continue; }
        let angle_norm = (current_angle / (2.0 * PI) + 10.0) % 1.0;
        let color_mix = if angle_norm < 0.333 {
            lerp_color32(color1, color2, angle_norm / 0.333)
        } else if angle_norm < 0.666 {
            lerp_color32(color2, color3, (angle_norm - 0.333) / 0.333)
        } else {
            lerp_color32(color3, color1, (angle_norm - 0.666) / 0.333)
        };
        let fade = (1.0 - (current_dist_clamped / max_dist).powi(2)).clamp(0.0, 1.0);
        let base_color = lerp_color32(TRANSPARENT, color_mix, fade);
        let final_color = lerp_color32(base_color, WHITE, brightness_boost * fade);
        painter.circle_filled(particle_pos, particle_radius * fade, final_color);
    }
}

fn draw_high_visuals_epaint(
     painter: &Painter,
    image_rect: Rect,
    img_w: f32,
    img_h: f32,
    object: &VisualObjectData,
    frame_count: u64,
    rng: &mut impl Rng,
) {
    let color_low = Color32::from_rgb(0, 0, 50);
    let color_mid = Color32::from_rgb(100, 50, 255);
    let color_high = WHITE;
    let field_intensity = object.intensity.powi(2);
    let line_intensity = object.intensity.sqrt();
    let line_thickness = (0.5 + line_intensity * 1.5).clamp(0.5, 2.0);
    let num_lines = (10.0 + line_intensity * 50.0) as usize;
    let object_bbox_screen = Rect {
         min: map_point(object.bbox.min, img_w, img_h, image_rect),
         max: map_point(object.bbox.max, img_w, img_h, image_rect),
    };
    let base_fill = lerp_color32(TRANSPARENT, color_low, 0.5 + field_intensity * 0.5);
    let mask_path = create_mask_path(
         &object.contours, img_w, img_h, image_rect, base_fill, Stroke::NONE
    );
    if let Some(path) = mask_path {
         painter.add(Shape::Path(path));
    }
    let time_seed = frame_count as f32 * 0.1;
    let line_color = lerp_color32(color_mid, color_high, line_intensity);
    let stroke = Stroke::new(line_thickness, line_color);
    for i in 0..num_lines {
         let line_y_norm = (i as f32 / num_lines as f32 + time_seed * 0.05 + object.animation_phase * 0.1) % 1.0;
         let line_y = object_bbox_screen.min.y + line_y_norm * object_bbox_screen.height();
         let mut points = Vec::new();
         let num_segments = 20;
         for j in 0..=num_segments {
             let x_norm = j as f32 / num_segments as f32;
             let x = object_bbox_screen.min.x + x_norm * object_bbox_screen.width();
             let noise_x = x * 0.05;
             let noise_y = line_y * 0.05;
             let noise_val = spatial_noise(noise_x, noise_y, time_seed + i as f32 * 0.1);
             let jitter = (noise_val - 0.5) * object_bbox_screen.height() * 0.05 * (0.2 + line_intensity);
             let point_y = line_y + jitter;
             let clamped_point = Pos2::new(
                 x.clamp(image_rect.min.x, image_rect.max.x),
                 point_y.clamp(image_rect.min.y, image_rect.max.y)
             );
             if object_bbox_screen.contains(clamped_point) {
                points.push(clamped_point);
             }
         }
         if points.len() > 1 {
             painter.add(Shape::line(points, stroke));
         }
    }
    let spark_count = (line_intensity * 50.0) as usize;
    let spark_color = lerp_color32(line_color, WHITE, 0.7);
    for _ in 0..spark_count {
         if rng.gen::<f32>() < 0.5 {
             let spark_x = rng.gen_range(object_bbox_screen.min.x..=object_bbox_screen.max.x);
             let spark_y = rng.gen_range(object_bbox_screen.min.y..=object_bbox_screen.max.y);
             let spark_pos = Pos2::new(spark_x, spark_y);
             if object_bbox_screen.contains(spark_pos) {
                 painter.circle_filled(spark_pos, line_thickness * 0.5, spark_color);
             }
         }
    }
}

pub fn draw_object_visuals_epaint(
    painter: &Painter,
    image_rect: Rect,
    img_w: f32,
    img_h: f32,
    object: &VisualObjectData,
    frame_count: u64,
    rng: &mut impl Rng,
) {
    if img_w <= 0.0 || img_h <= 0.0 || image_rect.width() <= 0.0 || image_rect.height() <= 0.0 {
         log::warn!("Invalid dimensions provided to draw_object_visuals_epaint. Img: {}x{}, Rect: {:?}", img_w, img_h, image_rect);
        return;
    }
    if object.contours.is_empty() || object.contours[0].is_empty() {
        return;
    }
    let clipped_painter = painter.with_clip_rect(image_rect);
    match object.band_index {
        0 => draw_bass_visuals_epaint(
            &clipped_painter, image_rect, img_w, img_h, object, frame_count, rng,
        ),
        1 => draw_mid_visuals_epaint(
             &clipped_painter, image_rect, img_w, img_h, object, frame_count, rng,
        ),
        2 => draw_high_visuals_epaint(
             &clipped_painter, image_rect, img_w, img_h, object, frame_count, rng,
        ),
        _ => {
            log::warn!("Visuals: Invalid band index {}", object.band_index);
        }
    }
}