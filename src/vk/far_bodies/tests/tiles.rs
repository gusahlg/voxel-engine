use bytemuck::Zeroable;
use glam::{Quat, Vec3, Vec4};

use super::support::{
    Rng, kept_seeds, mapped_down, pack_table, placed, random_body, ray_at, tile_at,
    view_along_neg_z, view_pitched,
};
use crate::camera::{Camera3D, Lens};
use crate::color::LinearRgb;
use crate::far_body::{FarBody, FarShape, MAX_FAR_MAPS};
use crate::vk::far_bodies::cones::mapped_horizon_half;
use crate::vk::far_bodies::sky_draw::{SkyBodyPipe, SkyDraw};
use crate::vk::far_bodies::table::{FarTableGpu, MAX_FAR_TILES, nonzero_tiles, pack_table_cached};
use crate::vk::far_bodies::tiles::{
    TileFrames, TileSample, fill_tile_lists, framebuffer_ndc, tile_layout, tile_rect, tile_within,
    used_tiles,
};
use crate::vk::far_bodies::view::{FarView, ViewBasis, far_view};

fn shader_accepts(ray: Vec3, cone: [f32; 4], px: f32) -> bool {
    let bound = cone[3];
    if bound < 0.0 {
        return true;
    }
    let c = Vec3::new(cone[0], cone[1], cone[2]);
    let facing = ray.dot(c);
    let crossed = ray.cross(c).length_squared();
    let lim = bound + 3.0 * px;
    facing > 0.0 && crossed <= lim * lim
}

/// View ray at framebuffer pixel `(x, y)` (origin top-left, y down) and the
/// shader's `length(fwidth(ray))` from a one-pixel step.
fn ray_and_px(inv: &glam::Mat4, w: u32, h: u32, x: f32, y: f32) -> (Vec3, f32) {
    let ray = ray_at(inv, w, h, x, y);
    let dx = if x + 1.0 < w as f32 { 1.0 } else { -1.0 };
    let dy = if y + 1.0 < h as f32 { 1.0 } else { -1.0 };
    let rx = ray_at(inv, w, h, x + dx, y);
    let ry = ray_at(inv, w, h, x, y + dy);
    let fw = (rx - ray).abs() + (ry - ray).abs();
    (ray, fw.length().max(1e-6))
}

fn project_centre(dir: Vec3, view: &FarView) -> Option<(f32, f32)> {
    let len2 = dir.length_squared();
    if !(len2 > 0.0) {
        return None;
    }
    let dir = dir / len2.sqrt();
    let clip = view.view_proj * Vec4::new(dir.x, dir.y, dir.z, 0.0);
    if !(clip.w > 1e-4) || !clip.is_finite() {
        return None;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    let cx = (ndc_x * 0.5 + 0.5) * view.width as f32;
    let cy = (0.5 - ndc_y * 0.5) * view.height as f32;
    (cx.is_finite() && cy.is_finite()).then_some((cx, cy))
}

fn wall(dir: Vec3, seed: u32) -> FarBody {
    FarBody {
        dir,
        distance: 2.0,
        radius: 6.0,
        shape: FarShape::InnerSphere,
        rotation: Quat::IDENTITY,
        albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
        atmosphere: LinearRgb([0.0, 0.0, 0.0]),
        seed,
    }
}

#[test]
fn tile_lists_partition_in_order_and_clamp_partial_edges() {
    let mut table = FarTableGpu::zeroed();
    // 100×70 is not a multiple of 64: 2×2 tiles, the right column is 36 px
    // and the bottom row is 6 px.
    let width = 100u32;
    let height = 70u32;
    let tile_px = 64u32;
    let tiles_x = width.div_ceil(tile_px);
    let tiles_y = height.div_ceil(tile_px);
    assert_eq!((tiles_x, tiles_y), (2, 2));
    table.header = [3, tile_px, tiles_x, tiles_y];
    // Kept 0 is a sphere, kept 2 is a cube. Tile 1 carries the sphere,
    // tile 3 carries the cube.
    table.body[0].atmosphere[3] = 1.0;
    table.body[2].atmosphere[3] = 0.0;
    table.tile_mask[0] = 0;
    table.tile_mask[1] = 0b001;
    table.tile_mask[2] = 0;
    table.tile_mask[3] = 0b100;
    fill_tile_lists(&mut table, width, height);

    assert_eq!(table.list_header, [2, 2, width, height]);
    assert_eq!(&table.tile_index[..2], &[0, 2]);
    assert_eq!(table.tile_index[2], 1);
    assert_eq!(table.tile_index[3], 3);
    let n = used_tiles(&table);
    let n_base = table.list_header[0] as usize;
    let n_full = table.list_header[1] as usize;
    assert_eq!(n_base + n_full, n);
    let base = &table.tile_index[..n_base];
    let full = &table.tile_index[n_base..n];
    assert!(base.windows(2).all(|w| w[0] < w[1]));
    assert!(full.windows(2).all(|w| w[0] < w[1]));
    for &i in base {
        assert_eq!(table.tile_mask[i as usize], 0, "base tile {i}");
    }
    for &i in full {
        assert_ne!(table.tile_mask[i as usize], 0, "full tile {i}");
    }

    assert_eq!(
        tile_rect(0, tile_px, tiles_x, width, height),
        [0, 0, 64, 64]
    );
    assert_eq!(
        tile_rect(1, tile_px, tiles_x, width, height),
        [64, 0, 100, 64]
    );
    assert_eq!(
        tile_rect(2, tile_px, tiles_x, width, height),
        [0, 64, 64, 70]
    );
    assert_eq!(
        tile_rect(3, tile_px, tiles_x, width, height),
        [64, 64, 100, 70]
    );
    // The clamped corner sits on the same NDC edge as the fullscreen triangle.
    let [x0, y0, x1, y1] = tile_rect(3, tile_px, tiles_x, width, height);
    let _ = (x0, y0);
    let edge = framebuffer_ndc(x1 as f32, y1 as f32, width, height);
    assert_eq!(edge[0].to_bits(), 1.0f32.to_bits());
    assert_eq!(edge[1].to_bits(), (-1.0f32).to_bits());
    let origin = framebuffer_ndc(0.0, 0.0, width, height);
    assert_eq!(origin[0].to_bits(), (-1.0f32).to_bits());
    assert_eq!(origin[1].to_bits(), 1.0f32.to_bits());

    let draw = SkyDraw::from_table(&table, true);
    assert!(draw.quads);
    assert_eq!(
        (
            draw.n_base,
            draw.n_sphere,
            draw.n_heavy,
            table.list_header[1]
        ),
        (2, 1, 1, 2)
    );
    assert_eq!(draw.n_coarse, 0);
    assert_eq!(draw.n_coarse_far, 0);
    assert_eq!(draw.n_mapsolo, 0);
    assert_eq!(draw.body, SkyBodyPipe::NoMap);

    // No kept bodies: one body-free fullscreen triangle, even with a grid.
    table.header[0] = 0;
    let empty = SkyDraw::from_table(&table, true);
    assert!(!empty.quads);
    assert!(empty.base);

    // Culling off, or no tile grid: one fullscreen triangle. A kept sphere
    // uses the sphere pipeline; a kept cube uses the no-mapped pipeline.
    table.header[0] = 1;
    table.body[0].atmosphere[3] = 1.0;
    let unculled = SkyDraw::from_table(&table, false);
    assert!(!unculled.quads && !unculled.base);
    assert_eq!(unculled.body, SkyBodyPipe::Sphere);
    table.body[0].atmosphere[3] = 0.0;
    let cube = SkyDraw::from_table(&table, false);
    assert_eq!(cube.body, SkyBodyPipe::NoMap);
    assert_eq!(cube.n_mapsolo, 0);
    table.body[0].atmosphere[3] = 4.0;
    let mapped = SkyDraw::from_table(&table, false);
    assert_eq!(mapped.body, SkyBodyPipe::Full);
    assert_eq!(mapped.n_mapsolo, 0);
    table.header[1] = 0;
    let no_grid = SkyDraw::from_table(&table, true);
    assert!(!no_grid.quads && !no_grid.base);
    assert_eq!(no_grid.body, SkyBodyPipe::Full);
    assert_eq!(no_grid.n_mapsolo, 0);
}

#[test]
fn packed_view_lists_match_the_masks() {
    let width = 100u32;
    let height = 70u32;
    let view = view_along_neg_z(60.0, width, height);
    let body = placed(-Vec3::Z, 0.05, FarShape::Sphere, 1);
    let table = pack_table(std::slice::from_ref(&body), Some(view));
    assert_eq!(table.header[1], 64);
    let n = used_tiles(&table);
    assert!(n > 1);
    let n_base = table.list_header[0] as usize;
    let n_full = table.list_header[1] as usize;
    assert_eq!(n_base + n_full, n);
    assert_eq!(table.list_header[2], width);
    assert_eq!(table.list_header[3], height);
    assert_eq!(n_full as u64, nonzero_tiles(&table));
    let mut seen = vec![false; n];
    for (slot, &index) in table.tile_index[..n].iter().enumerate() {
        let index = index as usize;
        assert!(index < n && !seen[index], "index {index} repeated");
        seen[index] = true;
        let base = slot < n_base;
        assert_eq!(table.tile_mask[index] == 0, base);
        if slot > 0 && slot != n_base {
            assert!(table.tile_index[slot - 1] < table.tile_index[slot]);
        }
    }
    assert!(seen.iter().all(|s| *s));
    let corner = (n as u32) - 1;
    let rect = tile_rect(corner, table.header[1], table.header[2], width, height);
    assert_eq!(rect[2], width);
    assert_eq!(rect[3], height);
    assert!(rect[2] - rect[0] < 64 || rect[3] - rect[1] < 64);
}

#[test]
fn no_view_leaves_the_tile_header_clear() {
    let body = placed(Vec3::Z, 0.2, FarShape::Sphere, 1);
    let table = pack_table(std::slice::from_ref(&body), None);
    assert_eq!(table.header[0], 1);
    assert_eq!(table.header[1], 0);
    assert_eq!(table.header[2], 0);
    assert_eq!(table.header[3], 0);
    assert!(table.tile_mask.iter().all(|m| *m == 0));
}

#[test]
fn huge_extent_uses_128_px_tiles() {
    let w: u32 = 64 * 128;
    let h: u32 = 64 * 65;
    assert!(w.div_ceil(64) * h.div_ceil(64) > MAX_FAR_TILES as u32);
    let view = view_along_neg_z(60.0, w, h);
    let body = placed(-Vec3::Z, 0.05, FarShape::Sphere, 1);
    let table = pack_table(std::slice::from_ref(&body), Some(view));
    assert_eq!(table.header[1], 128);
    assert_eq!(table.header[2], w.div_ceil(128));
    assert_eq!(table.header[3], h.div_ceil(128));
    let tiles = table.header[2] as u64 * table.header[3] as u64;
    assert!(tiles <= MAX_FAR_TILES as u64);
    assert_eq!(used_tiles(&table) as u64, tiles);
}

#[test]
fn sentinel_paints_every_tile_and_bits_follow_kept_order() {
    let w = 3440u32;
    let h = 1440u32;
    let view = view_along_neg_z(60.0, w, h);
    let culled = placed(Vec3::Z, 0.04, FarShape::Sphere, 1);
    let axis = placed(-Vec3::Z, 0.02, FarShape::Sphere, 2);
    let right = placed(
        Vec3::new(0.45, 0.0, -1.0).normalize(),
        0.02,
        FarShape::Sphere,
        4,
    );
    let up = placed(
        Vec3::new(0.0, 0.35, -1.0).normalize(),
        0.02,
        FarShape::Sphere,
        5,
    );
    let table = pack_table(&[culled, axis, wall(Vec3::Z, 3), right, up], Some(view));
    assert_eq!(kept_seeds(&table), vec![2, 3, 4, 5]);
    assert_eq!(table.header[0], 4);
    assert_eq!(table.header[1], 64);
    assert_eq!(table.header[2], w.div_ceil(64));
    assert_eq!(table.header[3], h.div_ceil(64));
    assert_eq!(table.cone[1][3].to_bits(), (-1.0f32).to_bits());

    let tiles_x = table.header[2];
    let tiles_y = table.header[3];
    let tiles = tiles_x as usize * tiles_y as usize;
    let live = (1u32 << 4) - 1;
    let mut only_wall = false;
    let mut axis_cols = Vec::new();
    let mut right_cols = Vec::new();
    let mut up_rows = Vec::new();
    for (i, mask) in table.tile_mask[..tiles].iter().copied().enumerate() {
        assert_eq!(mask & !live, 0, "tile {i} has a bit past the kept count");
        assert!(mask & 0b0010 != 0, "sentinel missing from tile {i}");
        let mut bits = mask;
        let mut prev = -1i32;
        while bits != 0 {
            let b = bits.trailing_zeros() as i32;
            assert!(b > prev, "bits are not ascending in tile {i}");
            prev = b;
            bits &= bits - 1;
        }
        let has_axis = mask & 0b0001 != 0;
        let has_right = mask & 0b0100 != 0;
        let has_up = mask & 0b1000 != 0;
        if !has_axis && !has_right && !has_up {
            only_wall = true;
        }
        if has_axis && !has_right && !has_up {
            axis_cols.push((i as u32) % tiles_x);
        }
        if has_right && !has_axis && !has_up {
            right_cols.push((i as u32) % tiles_x);
        }
        if has_up && !has_axis && !has_right {
            up_rows.push((i as u32) / tiles_x);
        }
    }
    assert!(only_wall, "a small body painted every tile");
    let mean = |v: &[u32]| v.iter().sum::<u32>() as f32 / v.len() as f32;
    assert!(!axis_cols.is_empty(), "on-axis body has no exclusive tiles");
    assert!(!right_cols.is_empty(), "right body has no exclusive tiles");
    assert!(!up_rows.is_empty(), "up body has no exclusive tiles");
    let mid_x = tiles_x as f32 * 0.5;
    let mid_y = tiles_y as f32 * 0.5;
    assert!(
        (mean(&axis_cols) - mid_x).abs() < 2.0,
        "on-axis body not centred: {}",
        mean(&axis_cols)
    );
    assert!(
        mean(&right_cols) > mid_x + 2.0,
        "right body landed on the left: {}",
        mean(&right_cols)
    );
    assert!(
        mean(&up_rows) < mid_y - 2.0,
        "up body landed in the lower half: {}",
        mean(&up_rows)
    );

    let inv = view.view_proj.inverse();
    let (ray, px) = ray_and_px(&inv, w, h, w as f32 * 0.5, h as f32 * 0.5);
    assert!(ray.dot(-Vec3::Z) > 0.99, "centre ray {ray}");
    assert!(shader_accepts(ray, table.cone[0], px));
    let centre = tile_at(w as f32 * 0.5, h as f32 * 0.5, 64, tiles_x);
    assert!(table.tile_mask[centre] & 1 != 0);
}

fn push_local(pixels: &mut Vec<(f32, f32)>, cx: f32, cy: f32, w: u32, h: u32) {
    let wf = w as f32;
    let hf = h as f32;
    for radius in [0.0, 4.0, 12.0, 28.0, 60.0, 120.0, 240.0] {
        let n = if radius == 0.0 { 1 } else { 8 };
        for i in 0..n {
            let t = i as f32 / n as f32 * std::f32::consts::TAU;
            let x = cx + radius * t.cos();
            let y = cy + radius * t.sin();
            if x >= 0.5 && y >= 0.5 && x < wf - 0.5 && y < hf - 0.5 {
                pixels.push((x, y));
            }
        }
    }
}

#[test]
fn tile_rect_covers_every_pixel_the_cone_accepts() {
    let mut rng = Rng(0x71E5_0A11);
    let bases = [(3440u32, 1440u32), (1920u32, 1080u32)];
    let scales = [0.5f32, 2.0f32];
    let fovs = [60.0f32, 90.0f32, 120.0f32];
    let mut accepts = 0u32;
    for &(bw, bh) in &bases {
        for &scale in &scales {
            for &fovy in &fovs {
                let w = ((bw as f32) * scale).round().max(1.0) as u32;
                let h = ((bh as f32) * scale).round().max(1.0) as u32;
                let cam = {
                    let position = Vec3::new(
                        rng.range(-40.0, 40.0),
                        rng.range(-20.0, 20.0),
                        rng.range(-40.0, 40.0),
                    );
                    let forward = rng.unit_vec();
                    let hint = rng.unit_vec();
                    let mut up = hint - forward * hint.dot(forward);
                    if up.length_squared() < 1e-6 {
                        let alt = if forward.x.abs() < 0.9 {
                            Vec3::X
                        } else {
                            Vec3::Y
                        };
                        up = alt - forward * alt.dot(forward);
                    }
                    let up = up.normalize();
                    Camera3D {
                        position,
                        target: position + forward,
                        up,
                        fovy,
                        lens: Lens::Rectilinear,
                    }
                };
                let aspect = w as f32 / h as f32;
                let tan_half = (fovy.to_radians() * 0.5).tan();
                let view = far_view(tan_half, cam.view_proj(aspect), w, h);
                let bodies: Vec<FarBody> =
                    (0..12).map(|i| random_body(&mut rng, i as u32)).collect();
                let table = pack_table(&bodies, Some(view));
                let (tile, tiles_x, tiles_y) = tile_layout(w, h);
                assert_eq!(table.header[1], tile, "fov {fovy} {w}x{h}");
                assert_eq!(table.header[2], tiles_x);
                assert_eq!(table.header[3], tiles_y);
                let n = table.header[0] as usize;
                let tiles = tiles_x as usize * tiles_y as usize;
                assert_eq!(used_tiles(&table), tiles);
                for k in 0..n {
                    if table.cone[k][3] < 0.0 {
                        let bit = 1u32 << k;
                        assert!(
                            table.tile_mask[..tiles].iter().all(|m| m & bit != 0),
                            "sentinel bit {k} missed a tile at {w}x{h} fov {fovy}"
                        );
                    }
                }

                let mut pixels = Vec::new();
                let step = (tile / 2).max(1);
                let mut y = 0u32;
                while y < h {
                    let mut x = 0u32;
                    while x < w {
                        pixels.push((x as f32 + 0.5, y as f32 + 0.5));
                        x += step;
                    }
                    y += step;
                }
                pixels.push((w as f32 - 0.5, h as f32 - 0.5));
                for _ in 0..48 {
                    let x = (rng.next() % w) as f32 + 0.5;
                    let y = (rng.next() % h) as f32 + 0.5;
                    pixels.push((x, y));
                }
                let inv = view.view_proj.inverse();
                for k in 0..n {
                    if table.cone[k][3] < 0.0 {
                        continue;
                    }
                    let dir = Vec3::new(table.cone[k][0], table.cone[k][1], table.cone[k][2]);
                    let Some((cx, cy)) = project_centre(dir, &view) else {
                        continue;
                    };
                    if cx >= 1.0 && cy >= 1.0 && cx < w as f32 - 1.0 && cy < h as f32 - 1.0 {
                        let (ray, px) = ray_and_px(&inv, w, h, cx, cy);
                        assert!(
                            shader_accepts(ray, table.cone[k], px),
                            "centre ray missed cone {k} fov {fovy} {w}x{h} ray {ray} dir {dir}"
                        );
                    }
                    push_local(&mut pixels, cx, cy, w, h);
                }

                for (x, y) in pixels {
                    let (ray, px) = ray_and_px(&inv, w, h, x, y);
                    if !ray.is_finite() || !px.is_finite() {
                        continue;
                    }
                    let tile_i = tile_at(x, y, tile, tiles_x);
                    assert!(tile_i < tiles, "pixel {x},{y} tile {tile_i} of {tiles}");
                    let mask = table.tile_mask[tile_i];
                    for k in 0..n {
                        if table.cone[k][3] < 0.0 {
                            continue;
                        }
                        if !shader_accepts(ray, table.cone[k], px) {
                            continue;
                        }
                        accepts += 1;
                        let bit = 1u32 << k;
                        assert!(
                            mask & bit != 0,
                            "fov {fovy} scale {scale} {w}x{h} body {k} bound {} px {px} px_max {} pixel {x},{y} tile {tile_i} mask {mask:#x} ray {ray}",
                            table.cone[k][3],
                            view.px_max,
                        );
                    }
                }
            }
        }
    }
    assert!(accepts > 100, "cone test never passed, accepts {accepts}");
}

#[test]
fn tile_frames_follow_the_projection_not_the_turn() {
    let horizon = view_pitched(0.0, 60.0, 800, 600);
    let up = view_pitched(30.0, 60.0, 800, 600);
    let mut cache = TileFrames::empty();
    assert!(cache.rebuild_if_changed(&horizon));
    assert!(
        !cache.rebuild_if_changed(&up),
        "turning the camera rebuilt the tile frames"
    );
    let fresh = TileFrames::build(&up).expect("frames");
    assert_eq!(cache.samples.len(), fresh.samples.len());
    for (a, b) in cache.samples.iter().zip(&fresh.samples) {
        assert!((a.centre - b.centre).length() < 1e-4);
        // Focal length wobbles by an ulp as the basis turns, so compare
        // the radius as an angle. 1e-3 rad is well under a pixel.
        let ang = |s: &TileSample| {
            if s.sin_r > 1.0 {
                std::f32::consts::PI
            } else {
                s.sin_r.atan2(s.cos_r)
            }
        };
        assert!((ang(a) - ang(b)).abs() < 1e-3);
    }
    // A turn reuses the frames, and the packed mask matches a fresh pack.
    let planet = placed(-Vec3::Y, 0.99999, FarShape::Sphere, 1);
    let (from_cache, _) = pack_table_cached(
        std::slice::from_ref(&planet),
        Some(&up),
        Some(&cache),
        &[0.0; MAX_FAR_MAPS],
        &[0.0; MAX_FAR_MAPS],
        Vec3::Y,
    );
    let fresh_pack = pack_table(std::slice::from_ref(&planet), Some(up));
    let n = used_tiles(&fresh_pack);
    assert_eq!(&from_cache.tile_mask[..n], &fresh_pack.tile_mask[..n]);

    assert!(cache.rebuild_if_changed(&view_pitched(30.0, 90.0, 800, 600)));
    assert!(cache.rebuild_if_changed(&view_pitched(30.0, 90.0, 1600, 900)));
}

#[test]
fn home_planet_looking_up_leaves_most_tiles_clear() {
    // 3440×1440 is 54×23 tiles of 64 px, 1242 in all. At spawn every one of
    // them carried the home planet, because rho ≈ 0.99999 took the sentinel.
    let w = 3440u32;
    let h = 1440u32;
    let view = view_pitched(30.0, 70.0, w, h);
    let planet = placed(-Vec3::Y, 0.99999, FarShape::Sphere, 1);
    let table = pack_table(std::slice::from_ref(&planet), Some(view));
    assert_eq!(table.header[0], 1);
    assert_eq!(table.cone[0][3].to_bits(), 1.0f32.to_bits());
    let tiles_x = table.header[2];
    let tiles_y = table.header[3];
    let tiles = tiles_x as usize * tiles_y as usize;
    assert_eq!(tiles, 1242, "tile grid drifted from the measured frame");
    let painted = table.tile_mask[..tiles]
        .iter()
        .filter(|mask| *mask & 1 != 0)
        .count();
    assert!(
        painted > 0,
        "the planet was culled out of a view that sees it"
    );
    assert!(
        painted * 10 < tiles * 6,
        "painted {painted} of {tiles}, want under 60%"
    );
    let top = (tiles_x / 2) as usize;
    assert_eq!(
        table.tile_mask[top] & 1,
        0,
        "top-centre tile still carries the planet"
    );
    let bottom = ((tiles_y - 1) * tiles_x + tiles_x / 2) as usize;
    assert_ne!(
        table.tile_mask[bottom] & 1,
        0,
        "bottom-centre tile missed the planet"
    );
}

/// The per-pixel cone stays the hi-sphere sentinel. Tiles use the horizon
/// cone widened by the air limb, so a gate pixel and a limb pixel keep the
/// bit and the sky above that cone does not.
#[test]
fn mapped_horizon_tiles_cover_the_gate_and_leave_the_sky_clear() {
    let mut map_max = [0.0f32; MAX_FAR_MAPS];
    // (radius + max) / distance = 11/4 > 0.99, so cone.w is the sentinel.
    map_max[0] = 10.0;
    let distance = 4.0f32;
    let horizon = 0.04f32;
    // 0.15 rad of air is several 64 px tiles at 1440p, so the limb pulls
    // in tiles the raw horizon cone misses.
    let air = 0.6f32;
    let body = mapped_down(distance, 1.0, horizon, air);

    let wide = view_pitched(0.0, 70.0, 3440, 1440);
    let disabled = mapped_down(distance, 1.0, 1.0, air);
    let all = super::pack_table(std::slice::from_ref(&disabled), Some(wide), &map_max);
    assert_eq!(all.cone[0][3].to_bits(), (-1.0f32).to_bits());
    let wide_tiles = used_tiles(&all);
    assert_eq!(wide_tiles, 1242);
    assert!(
        all.tile_mask[..wide_tiles].iter().all(|mask| mask & 1 != 0),
        "horizon >= 1 must keep today's full-sky mask"
    );

    let level = super::pack_table(std::slice::from_ref(&body), Some(wide), &map_max);
    assert_eq!(level.header[0], 1);
    assert_eq!(
        level.cone[0][3].to_bits(),
        (-1.0f32).to_bits(),
        "the per-pixel cone is still the sentinel"
    );
    let painted = level.tile_mask[..wide_tiles]
        .iter()
        .filter(|mask| *mask & 1 != 0)
        .count();
    assert!(
        painted > 0 && painted < wide_tiles,
        "painted {painted} of {wide_tiles}"
    );

    let (sin_air, cos_air) =
        mapped_horizon_half(horizon, air, distance, wide.px_max).expect("widened cone");
    let (sin_raw, cos_raw) =
        mapped_horizon_half(horizon, 0.0, distance, wide.px_max).expect("raw cone");
    let frames = TileFrames::build(&wide).expect("frames");
    let basis = ViewBasis::from_view_proj(wide.view_proj).expect("basis");
    let dir_view = basis.to_view(-Vec3::Y);
    let mut limb_tiles = 0u32;
    let mut clear_tiles = 0u32;
    for (i, tile) in frames.samples.iter().enumerate() {
        let in_air = tile_within(tile, dir_view, sin_air, cos_air);
        let in_raw = tile_within(tile, dir_view, sin_raw, cos_raw);
        if in_air && !in_raw {
            assert_ne!(level.tile_mask[i] & 1, 0, "limb tile {i} missed the bit");
            limb_tiles += 1;
        }
        if !in_air {
            assert_eq!(level.tile_mask[i] & 1, 0, "sky tile {i} kept the bit");
            clear_tiles += 1;
        }
    }
    assert!(
        limb_tiles > 0,
        "the air limb painted no tile past the raw gate"
    );
    assert!(clear_tiles > 0, "no tile sits outside the horizon cone");

    // 85° up, 50° vertical fov: the whole view is above the widened cone.
    let steep = view_pitched(85.0, 50.0, 800, 600);
    let culled = super::pack_table(std::slice::from_ref(&body), Some(steep), &map_max);
    assert_eq!(culled.header[0], 0, "a sky-only view kept the planet");

    let mut gate_pixels = 0u32;
    let mut limb_pixels = 0u32;
    let frames_px = [
        (40.0f32, 60.0f32, 382u32, 160u32),
        (0.0, 70.0, 382, 160),
        (0.0, 90.0, 640, 360),
        (-25.0, 70.0, 640, 360),
        (-70.0, 60.0, 382, 160),
        (0.0, 70.0, 3440, 1440),
        (30.0, 70.0, 3440, 1440),
        (85.0, 50.0, 800, 600),
    ];
    for &(pitch, fovy, w, h) in &frames_px {
        let view = view_pitched(pitch, fovy, w, h);
        let table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        let Some((_, cos_b)) = mapped_horizon_half(horizon, air, distance, view.px_max) else {
            continue;
        };
        let kept = table.header[0] == 1;
        if kept {
            assert_eq!(table.cone[0][3].to_bits(), (-1.0f32).to_bits());
        }
        let tile_px = table.header[1];
        let tiles_x = table.header[2];
        let n_tiles = used_tiles(&table);
        let inv = view.view_proj.inverse();
        for y in 0..h {
            for x in 0..w {
                let ray = ray_at(&inv, w, h, x as f32 + 0.5, y as f32 + 0.5);
                if !ray.is_finite() {
                    continue;
                }
                // Shader gate: `dot(ray, -dir) <= horizon`. dir is −Y.
                let gate = ray.dot(Vec3::Y) <= horizon;
                let along = ray.dot(-Vec3::Y);
                let in_limb = along >= cos_b + 1.0e-4;
                if !gate && !in_limb {
                    continue;
                }
                assert!(
                    kept,
                    "pitch {pitch} fov {fovy} {w}x{h} dropped a ray inside the cone"
                );
                let tile_i = tile_at(x as f32 + 0.5, y as f32 + 0.5, tile_px, tiles_x);
                assert!(
                    tile_i < n_tiles && table.tile_mask[tile_i] & 1 != 0,
                    "pitch {pitch} fov {fovy} {w}x{h} pixel {x},{y} gate {gate} along {along} cos {cos_b}"
                );
                if gate {
                    gate_pixels += 1;
                } else {
                    limb_pixels += 1;
                }
            }
        }
    }
    assert!(gate_pixels > 1000, "gate scan never passed, {gate_pixels}");
    assert!(
        limb_pixels > 0,
        "no pixel sat in the air limb past the raw gate"
    );

    // Finite per-pixel sine. The horizon cone is wider, so a gate ray the
    // pixel cone rejects still carries the bit.
    map_max[0] = 0.0;
    let far = mapped_down(10.0, 1.0, -0.2, 0.05);
    let down = view_pitched(-40.0, 70.0, 640, 360);
    let finite = super::pack_table(std::slice::from_ref(&far), Some(down), &map_max);
    assert_eq!(finite.header[0], 1);
    let bound = finite.cone[0][3];
    assert!(
        bound > 0.0 && bound < 0.2,
        "expected the hi-sphere sine, got {bound}"
    );
    let inv = down.view_proj.inverse();
    let tile_px = finite.header[1];
    let tiles_x = finite.header[2];
    let mut gate = 0u32;
    let mut past_pixel_cone = 0u32;
    for y in 0..down.height {
        for x in 0..down.width {
            let (ray, px) = ray_and_px(
                &inv,
                down.width,
                down.height,
                x as f32 + 0.5,
                y as f32 + 0.5,
            );
            if !ray.is_finite() || ray.dot(Vec3::Y) > -0.2 {
                continue;
            }
            gate += 1;
            let tile_i = tile_at(x as f32 + 0.5, y as f32 + 0.5, tile_px, tiles_x);
            assert!(
                finite.tile_mask[tile_i] & 1 != 0,
                "gate pixel {x},{y} missed the horizon-cone bit"
            );
            if !shader_accepts(ray, finite.cone[0], px) {
                past_pixel_cone += 1;
            }
        }
    }
    assert!(gate > 100, "finite-cone view held no gate pixel");
    assert!(
        past_pixel_cone > 0,
        "the horizon cone was no wider than the per-pixel sine"
    );

    // horizon >= 1 with a finite sine keeps today's disc, not the whole sky.
    // The disc is ~6° around nadir, so the view has to look nearly straight down.
    let nadir = view_pitched(-85.0, 40.0, 640, 360);
    let off = mapped_down(10.0, 1.0, 1.0, 0.0);
    let today = super::pack_table(std::slice::from_ref(&off), Some(nadir), &map_max);
    assert_eq!(today.header[0], 1);
    assert!(today.cone[0][3] > 0.0);
    let n = used_tiles(&today);
    let painted_today = today.tile_mask[..n].iter().filter(|m| *m & 1 != 0).count();
    assert!(
        painted_today > 0 && painted_today < n,
        "horizon >= 1 painted {painted_today} of {n}"
    );
}

#[test]
fn big_sphere_tiles_cover_every_pixel_the_cone_accepts() {
    let rhos = [0.5f32, 0.9, 0.99, 0.99999];
    // Up, the horizon, and down. The body is the planet under the camera.
    let pitches = [50.0f32, 0.0, -80.0];
    let fovs = [60.0f32, 90.0, 120.0];
    // 16:9 and the 3440:1440 frame, small enough to visit every pixel.
    let extents = [(384u32, 216u32), (382u32, 160u32)];
    let mut accepts = 0u32;
    for &rho in &rhos {
        for &pitch in &pitches {
            for &fovy in &fovs {
                for &(w, h) in &extents {
                    let view = view_pitched(pitch, fovy, w, h);
                    let body = placed(-Vec3::Y, rho, FarShape::Sphere, 1);
                    let table = pack_table(std::slice::from_ref(&body), Some(view));
                    // A hemisphere under the camera meets a horizon or downward
                    // view. A tighter cone (rho 0.5) can miss the horizon.
                    if pitch < 0.0 || (pitch == 0.0 && rho >= 0.9) {
                        assert_eq!(
                            table.header[0], 1,
                            "rho {rho} pitch {pitch} fov {fovy} dropped a planet the view meets"
                        );
                    }
                    if table.header[0] == 0 {
                        continue;
                    }
                    let bound = table.cone[0][3];
                    assert!(bound >= 0.0, "rho {rho} took the sentinel ({bound})");
                    if 1.05 * rho < 1.0 {
                        assert!((bound - 1.05 * rho).abs() < 1e-5, "rho {rho} bound {bound}");
                    } else {
                        assert_eq!(bound.to_bits(), 1.0f32.to_bits());
                    }
                    let tile = table.header[1];
                    let tiles_x = table.header[2];
                    let tiles = tiles_x as usize * table.header[3] as usize;
                    let inv = view.view_proj.inverse();
                    for y in 0..h {
                        for x in 0..w {
                            let px_x = x as f32 + 0.5;
                            let px_y = y as f32 + 0.5;
                            let (ray, px) = ray_and_px(&inv, w, h, px_x, px_y);
                            if !ray.is_finite() || !px.is_finite() {
                                continue;
                            }
                            if !shader_accepts(ray, table.cone[0], px) {
                                continue;
                            }
                            accepts += 1;
                            let tile_i = tile_at(px_x, px_y, tile, tiles_x);
                            assert!(
                                tile_i < tiles && table.tile_mask[tile_i] & 1 != 0,
                                "rho {rho} pitch {pitch} fov {fovy} {w}x{h} pixel {px_x},{px_y} tile {tile_i} bound {bound} px {px} ray {ray}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        accepts > 100,
        "big-sphere cone test never passed, accepts {accepts}"
    );
}
