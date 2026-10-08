//! The per-frame tile classification as of d2f0e9f, frozen for
//! `tests/classify.rs`: the pack's mask stamping, the tile lists, the horizon
//! tile clear, the draw counts, the mapsolo and coarse partitions, and the
//! horizon candidate pick, with `far_cull_enabled()` and
//! `sky_mapsolo_enabled()` turned into parameters. Each function is the old
//! body with only that change. The geometry they call (`cone_bound` behind
//! its old switch, `body_meets_view`, `horizon_dip`, the tile and horizon
//! tests, `TileFrames`) was not changed by the cleanup, so the production
//! copies are used. Delete this file and the comparison once the cleanup
//! has shipped.

use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};
use crate::vk::far_bodies::cones::{
    add_angles, cone_bound, lo_disc_interior, mapped_horizon_half, mapped_interior_half, unit_dir,
};
use crate::vk::far_bodies::horizon::{horizon_axes, horizon_dip, tile_above_horizon};
use crate::vk::far_bodies::sky_draw::{SkyBodyPipe, SkyCoarseQuery, SkyDraw};
use crate::vk::far_bodies::table::{
    FarBodyGpu, FarTableGpu, HORIZON_BINS, HORIZON_TABLES, zeroed_table,
};
use crate::vk::far_bodies::tiles::{
    TileFrames, tile_layout, tile_strictly_inside, tile_within, used_tiles,
};
use crate::vk::far_bodies::view::{FarView, ViewBasis, body_meets_view};

fn rgb4(c: crate::color::LinearRgb) -> [f32; 4] {
    [c.0[0], c.0[1], c.0[2], 0.0]
}

fn old_pack_one(body: &FarBody) -> FarBodyGpu {
    let dir = body.dir;
    let q = body.rotation;
    let (shape, exponent, map_plus, horizon, air) = match body.shape {
        FarShape::Cube => (0.0, 0.0, 0, 0.0, 0.0),
        FarShape::Sphere => (1.0, 0.0, 0, 0.0, 0.0),
        FarShape::InnerSphere => (2.0, 0.0, 0, 0.0, 0.0),
        FarShape::Rounded { exponent } => (3.0, exponent, 0, 0.0, 0.0),
        FarShape::Mapped { map, horizon, air } => (4.0, 0.0, u32::from(map.0) + 1, horizon, air),
    };
    let mut albedo0 = rgb4(body.albedo[0]);
    let mut albedo1 = rgb4(body.albedo[1]);
    let mut albedo2 = rgb4(body.albedo[2]);
    albedo0[3] = horizon;
    albedo1[3] = air;
    albedo2[3] = body.distance;
    FarBodyGpu {
        dir_rho: [dir.x, dir.y, dir.z, body.radius / body.distance],
        rot: [q.x, q.y, q.z, q.w],
        albedo0,
        albedo1,
        albedo2,
        albedo3: rgb4(body.albedo[3]),
        albedo4: rgb4(body.albedo[4]),
        albedo5: rgb4(body.albedo[5]),
        atmosphere: [
            body.atmosphere.0[0],
            body.atmosphere.0[1],
            body.atmosphere.0[2],
            shape,
        ],
        seed: [body.seed, exponent.to_bits(), map_plus, 0],
    }
}

fn old_prime_horizon(table: &mut FarTableGpu) {
    table.horizon_id = [u32::MAX, u32::MAX, HORIZON_BINS as u32, 0];
    table.horizon_sin.fill(1.0);
}

pub(super) fn old_pack_table_cached(
    bodies: &[FarBody],
    view: Option<&FarView>,
    frames: Option<&TileFrames>,
    map_max: &[f32; MAX_FAR_MAPS],
    map_min: &[f32; MAX_FAR_MAPS],
    sky_up: glam::Vec3,
    cull: bool,
) -> (Box<FarTableGpu>, f32) {
    let mut table = zeroed_table();
    old_prime_horizon(&mut table);
    let n = bodies.len().min(MAX_FAR_BODIES);
    let dip = horizon_dip(&bodies[..n], sky_up, map_min);
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        // The old `cone_bound` opened with `if !far_cull_enabled() { return -1.0; }`.
        let bound = if cull {
            cone_bound(body, map_max)
        } else {
            -1.0
        };
        if cull {
            if let Some(view) = view {
                if !body_meets_view(body, bound, view) {
                    continue;
                }
            }
        }
        let d = body.dir;
        table.cone[kept] = [d.x, d.y, d.z, bound];
        table.body[kept] = old_pack_one(body);
        kept += 1;
    }
    table.header[0] = kept as u32;
    old_stamp_tiles(&mut table, view, frames, cull);
    (table, dip)
}

const RECT_SAFETY: f32 = 1.25;
const RECT_PAD_PX: f32 = 2.0;
const VIEW_PLANE_LIMIT_RAD: f32 = 80.0 * std::f32::consts::PI / 180.0;
const LARGE_CONE_SINE: f32 = 0.5;

enum TileCover {
    All,
    None,
    Rect {
        tx0: u32,
        ty0: u32,
        tx1: u32,
        ty1: u32,
    },
    Angular,
}

fn old_tile_cover(
    dir: glam::Vec3,
    bound: f32,
    view: &FarView,
    tile_px: u32,
    tiles_x: u32,
    tiles_y: u32,
    cull: bool,
) -> TileCover {
    if !cull || !(bound >= 0.0) || !bound.is_finite() {
        return TileCover::All;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !len2.is_finite() {
        return TileCover::All;
    }
    let dir = dir / len2.sqrt();
    let lim = bound + 3.0 * view.px_max;
    if !lim.is_finite() {
        return TileCover::All;
    }
    let a_sin = lim.min(1.0);
    let a = a_sin.asin();
    let clip = view.view_proj * glam::Vec4::new(dir.x, dir.y, dir.z, 0.0);
    let theta = if clip.is_finite() {
        clip.w.clamp(-1.0, 1.0).acos()
    } else {
        f32::NAN
    };
    let cos_outer = (theta + a).cos();
    let wide = bound >= LARGE_CONE_SINE || a_sin >= 1.0;
    let near_plane = !theta.is_finite()
        || theta + a >= VIEW_PLANE_LIMIT_RAD
        || !(cos_outer > 0.0)
        || !(clip.w > 0.0);
    if wide || near_plane {
        return TileCover::Angular;
    }
    let rows = view.view_proj.transpose();
    let f_px_x = 0.5 * view.width as f32 * rows.x_axis.truncate().length();
    let f_px_y = 0.5 * view.height as f32 * rows.y_axis.truncate().length();
    if !(f_px_x.is_finite() && f_px_y.is_finite()) || f_px_x <= 0.0 || f_px_y <= 0.0 {
        return TileCover::All;
    }
    let stretch = a / (cos_outer * cos_outer);
    let rx = RECT_SAFETY * stretch * f_px_x + RECT_PAD_PX;
    let ry = RECT_SAFETY * stretch * f_px_y + RECT_PAD_PX;
    if !(rx.is_finite() && ry.is_finite()) {
        return TileCover::All;
    }
    let ndc_x = clip.x / clip.w;
    let ndc_y = clip.y / clip.w;
    if !(ndc_x.is_finite() && ndc_y.is_finite()) {
        return TileCover::All;
    }
    let w = view.width as f32;
    let h = view.height as f32;
    let cx = (ndc_x * 0.5 + 0.5) * w;
    let cy = (0.5 - ndc_y * 0.5) * h;
    if !(cx.is_finite() && cy.is_finite()) {
        return TileCover::All;
    }
    let left = cx - rx;
    let right = cx + rx;
    let top = cy - ry;
    let bot = cy + ry;
    if right < 0.0 || bot < 0.0 || left >= w || top >= h {
        return TileCover::None;
    }
    let x0 = left.max(0.0).floor() as u32;
    let y0 = top.max(0.0).floor() as u32;
    let x1 = (right.floor().max(0.0) as u32).min(view.width - 1);
    let y1 = (bot.floor().max(0.0) as u32).min(view.height - 1);
    if x0 > x1 || y0 > y1 {
        return TileCover::None;
    }
    let tx0 = (x0 / tile_px).min(tiles_x - 1);
    let tx1 = (x1 / tile_px).min(tiles_x - 1);
    let ty0 = (y0 / tile_px).min(tiles_y - 1);
    let ty1 = (y1 / tile_px).min(tiles_y - 1);
    TileCover::Rect { tx0, ty0, tx1, ty1 }
}

fn old_stamp_tiles(
    table: &mut FarTableGpu,
    view: Option<&FarView>,
    cached: Option<&TileFrames>,
    cull: bool,
) {
    let Some(view) = view else {
        return;
    };
    let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
    table.header[1] = tile_px;
    table.header[2] = tiles_x;
    table.header[3] = tiles_y;
    let tile_count = tiles_x as usize * tiles_y as usize;
    let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
    let mut blanket = 0u32;
    let mut angular = [(0u32, glam::Vec3::ZERO, 0.0f32, 0.0f32); MAX_FAR_BODIES];
    let mut n_angular = 0usize;
    for k in 0..kept {
        let bit = 1u32 << k;
        let cone = table.cone[k];
        let dir = glam::Vec3::new(cone[0], cone[1], cone[2]);
        let gpu = &table.body[k];
        let shape = gpu.atmosphere[3];
        let horizon = gpu.albedo0[3];
        if (3.5..4.5).contains(&shape) && horizon < 1.0 {
            match mapped_horizon_half(horizon, gpu.albedo1[3], gpu.albedo2[3], view.px_max) {
                None => blanket |= bit,
                Some((sin_b, cos_b)) => {
                    angular[n_angular] = (bit, dir, sin_b, cos_b);
                    n_angular += 1;
                }
            }
            continue;
        }
        match old_tile_cover(dir, cone[3], view, tile_px, tiles_x, tiles_y, cull) {
            TileCover::All => blanket |= bit,
            TileCover::None => {}
            TileCover::Rect { tx0, ty0, tx1, ty1 } => {
                for ty in ty0..=ty1 {
                    let row = ty as usize * tiles_x as usize;
                    for tx in tx0..=tx1 {
                        table.tile_mask[row + tx as usize] |= bit;
                    }
                }
            }
            TileCover::Angular => {
                let sin_body = (cone[3] + 3.0 * view.px_max).clamp(0.0, 1.0);
                if !sin_body.is_finite() {
                    blanket |= bit;
                } else {
                    let cos_body = (1.0 - sin_body * sin_body).max(0.0).sqrt();
                    angular[n_angular] = (bit, dir, sin_body, cos_body);
                    n_angular += 1;
                }
            }
        }
    }
    if n_angular > 0 {
        old_paint_angular(table, view, cached, &angular[..n_angular], &mut blanket);
    }
    if blanket != 0 {
        for mask in &mut table.tile_mask[..tile_count] {
            *mask |= blanket;
        }
    }
    old_fill_tile_lists(table, view.width, view.height);
}

fn old_paint_angular(
    table: &mut FarTableGpu,
    view: &FarView,
    cached: Option<&TileFrames>,
    angular: &[(u32, glam::Vec3, f32, f32)],
    blanket: &mut u32,
) {
    let owned = if cached.is_some_and(|frames| frames.matches(view)) {
        None
    } else {
        TileFrames::build(view)
    };
    let Some(frames) = cached
        .filter(|frames| frames.matches(view))
        .or(owned.as_ref())
    else {
        for (bit, _, _, _) in angular {
            *blanket |= *bit;
        }
        return;
    };
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        for (bit, _, _, _) in angular {
            *blanket |= *bit;
        }
        return;
    };
    let n = frames.samples.len().min(table.tile_mask.len());
    for &(bit, dir, sin_body, cos_body) in angular {
        let len2 = dir.length_squared();
        if !(len2 > 0.0) || !len2.is_finite() {
            *blanket |= bit;
            continue;
        }
        let dir_view = basis.to_view(dir / len2.sqrt());
        let dir_len2 = dir_view.length_squared();
        if !(dir_len2 > 0.0) || !dir_view.is_finite() {
            *blanket |= bit;
            continue;
        }
        let dir_view = dir_view / dir_len2.sqrt();
        if !(sin_body.is_finite() && cos_body.is_finite()) {
            *blanket |= bit;
            continue;
        }
        for (i, tile) in frames.samples.iter().enumerate().take(n) {
            if tile_within(tile, dir_view, sin_body, cos_body) {
                table.tile_mask[i] |= bit;
            }
        }
    }
}

fn old_heavy_mask(table: &FarTableGpu) -> u32 {
    let n = (table.header[0] as usize).min(MAX_FAR_BODIES);
    let mut heavy = 0u32;
    for i in 0..n {
        let shape = table.body[i].atmosphere[3];
        let light = (0.5..1.5).contains(&shape) || (1.5..2.5).contains(&shape);
        if !light {
            heavy |= 1u32 << i;
        }
    }
    heavy
}

fn old_has_mapped(table: &FarTableGpu) -> bool {
    let n = (table.header[0] as usize).min(MAX_FAR_BODIES);
    (0..n).any(|i| {
        let shape = table.body[i].atmosphere[3];
        (3.5..4.5).contains(&shape)
    })
}

fn old_tile_split(table: &FarTableGpu) -> (u32, u32, u32) {
    let heavy = old_heavy_mask(table);
    let n = used_tiles(table);
    let mut n_base = 0u32;
    let mut n_sphere = 0u32;
    let mut n_heavy = 0u32;
    for mask in &table.tile_mask[..n] {
        if *mask == 0 {
            n_base += 1;
        } else if mask & heavy == 0 {
            n_sphere += 1;
        } else {
            n_heavy += 1;
        }
    }
    (n_base, n_sphere, n_heavy)
}

fn old_fill_tile_lists(table: &mut FarTableGpu, width: u32, height: u32) {
    let heavy = old_heavy_mask(table);
    let (n_base, n_sphere, n_heavy) = old_tile_split(table);
    let n = used_tiles(table);
    let mut base_i = 0u32;
    let mut sphere_i = n_base;
    let mut heavy_i = n_base + n_sphere;
    for (i, mask) in table.tile_mask[..n].iter().copied().enumerate() {
        if mask == 0 {
            table.tile_index[base_i as usize] = i as u32;
            base_i += 1;
        } else if mask & heavy == 0 {
            table.tile_index[sphere_i as usize] = i as u32;
            sphere_i += 1;
        } else {
            table.tile_index[heavy_i as usize] = i as u32;
            heavy_i += 1;
        }
    }
    table.list_header = [n_base, n_sphere + n_heavy, width, height];
}

fn old_clear_tiles_above_horizon(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    cull: bool,
) -> bool {
    if !cull || !frames.matches(view) {
        return false;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return false;
    };
    let n = used_tiles(table).min(frames.samples.len());
    let mut changed = false;
    for slot in 0..HORIZON_TABLES {
        let id = table.horizon_id[slot];
        if id == u32::MAX || id as usize >= MAX_FAR_BODIES {
            continue;
        }
        let gpu = &table.body[id as usize];
        let Some(dir) = unit_dir(glam::Vec3::new(
            gpu.dir_rho[0],
            gpu.dir_rho[1],
            gpu.dir_rho[2],
        )) else {
            continue;
        };
        let up = -dir;
        let Some((east, north)) = horizon_axes(up) else {
            continue;
        };
        let Some(up_v) = unit_dir(basis.to_view(up)) else {
            continue;
        };
        let Some(east_v) = unit_dir(basis.to_view(east)) else {
            continue;
        };
        let Some(north_v) = unit_dir(basis.to_view(north)) else {
            continue;
        };
        let mut bins = [1.0f32; HORIZON_BINS];
        let start = slot * HORIZON_BINS;
        bins.copy_from_slice(&table.horizon_sin[start..start + HORIZON_BINS]);
        let bit = 1u32 << id;
        for i in 0..n {
            if table.tile_mask[i] & bit == 0 {
                continue;
            }
            if tile_above_horizon(
                &frames.samples[i],
                &bins,
                up_v,
                east_v,
                north_v,
                view.px_max,
            ) {
                table.tile_mask[i] &= !bit;
                changed = true;
            }
        }
    }
    if changed {
        old_fill_tile_lists(table, view.width, view.height);
    }
    changed
}

fn old_widened_disc(cos_rim: f32, margin_rad: f32) -> Option<(f32, f32)> {
    if !(cos_rim.is_finite() && margin_rad.is_finite()) || !(-1.0..=1.0).contains(&cos_rim) {
        return None;
    }
    if margin_rad < 0.0 {
        return None;
    }
    let (sin_w, cos_w) = add_angles(cos_rim, margin_rad.cos());
    if !(sin_w.is_finite() && cos_w.is_finite()) || sin_w < 0.0 || sin_w > 1.0 {
        return None;
    }
    Some((sin_w, cos_w))
}

fn old_split_coarse_base(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    query: &SkyCoarseQuery,
) -> u32 {
    if query.stars {
        return 0;
    }
    let n_base = table.list_header[0] as usize;
    if n_base == 0 || !frames.matches(view) {
        return 0;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return 0;
    };
    let Some(sun) = unit_dir(query.sun_dir) else {
        return 0;
    };
    let Some(sun_view) = unit_dir(basis.to_view(sun)) else {
        return 0;
    };
    let Some(moon_view) = unit_dir(basis.to_view(-sun)) else {
        return 0;
    };
    let Some((sun_sin, sun_cos)) = old_widened_disc(query.sun_cos_rim, view.px_max) else {
        return 0;
    };
    let Some((moon_sin, moon_cos)) = old_widened_disc(query.moon_cos_rim, view.px_max) else {
        return 0;
    };
    let mut touch = Vec::with_capacity(n_base);
    for slot in 0..n_base {
        let index = table.tile_index[slot];
        let Some(tile) = frames.samples.get(index as usize) else {
            return 0;
        };
        touch.push(
            tile_within(tile, sun_view, sun_sin, sun_cos)
                || tile_within(tile, moon_view, moon_sin, moon_cos),
        );
    }
    let mut coarse = Vec::with_capacity(n_base);
    let mut fine = Vec::with_capacity(n_base);
    for (slot, hit) in touch.into_iter().enumerate() {
        let index = table.tile_index[slot];
        if hit {
            fine.push(index);
        } else {
            coarse.push(index);
        }
    }
    let n_coarse = coarse.len();
    for (slot, index) in coarse.into_iter().chain(fine).enumerate() {
        table.tile_index[slot] = index;
    }
    n_coarse as u32
}

fn old_mapped_rho_lo(gpu: &FarBodyGpu, map_min: &[f32; MAX_FAR_MAPS]) -> Option<f32> {
    let shape = gpu.atmosphere[3];
    if !(3.5..4.5).contains(&shape) {
        return None;
    }
    let map_plus = gpu.seed[2];
    if map_plus == 0 || map_plus as usize > MAX_FAR_MAPS {
        return None;
    }
    let distance = gpu.albedo2[3];
    let rho = gpu.dir_rho[3];
    if !(distance > 0.0) || !distance.is_finite() || !rho.is_finite() {
        return None;
    }
    let min_off = map_min[(map_plus as usize) - 1];
    if !min_off.is_finite() {
        return None;
    }
    let rho_lo = rho + min_off / distance;
    rho_lo.is_finite().then_some(rho_lo)
}

fn old_split_coarse_far(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    map_min: &[f32; MAX_FAR_MAPS],
) -> u32 {
    let (n_base, n_sphere, n_heavy) = old_tile_split(table);
    if n_heavy == 0 || !frames.matches(view) {
        return 0;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return 0;
    };
    let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
    let mut interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    let mut lo_interior: [Option<(glam::Vec3, f32, f32)>; MAX_FAR_BODIES] = [None; MAX_FAR_BODIES];
    for k in 0..kept {
        let gpu = &table.body[k];
        let shape = gpu.atmosphere[3];
        if !(3.5..4.5).contains(&shape) {
            continue;
        }
        let dir = glam::Vec3::new(table.cone[k][0], table.cone[k][1], table.cone[k][2]);
        let Some(dir) = unit_dir(dir) else {
            continue;
        };
        let Some(dir_view) = unit_dir(basis.to_view(dir)) else {
            continue;
        };
        let horizon = gpu.albedo0[3];
        if horizon < 1.0
            && let Some((sin_b, cos_b)) =
                mapped_interior_half(horizon, gpu.albedo1[3], gpu.albedo2[3], view.px_max)
        {
            interior[k] = Some((dir_view, sin_b, cos_b));
        }
        if let Some(rho_lo) = old_mapped_rho_lo(gpu, map_min)
            && let Some((sin_b, cos_b)) = lo_disc_interior(rho_lo, view.px_max)
        {
            lo_interior[k] = Some((dir_view, sin_b, cos_b));
        }
    }
    let start = (n_base + n_sphere) as usize;
    let end = start + n_heavy as usize;
    if end > table.tile_index.len() {
        return 0;
    }
    let mut take = Vec::with_capacity(n_heavy as usize);
    for slot in start..end {
        let index = table.tile_index[slot];
        let Some(tile) = frames.samples.get(index as usize) else {
            return 0;
        };
        let mask = table.tile_mask[index as usize];
        let inside = mask.count_ones() == 1 && {
            let k = mask.trailing_zeros() as usize;
            let in_horizon = match interior[k] {
                Some((dir_view, sin_b, cos_b)) => {
                    tile_strictly_inside(tile, dir_view, sin_b, cos_b)
                }
                None => false,
            };
            let in_lo = match lo_interior[k] {
                Some((dir_view, sin_b, cos_b)) => {
                    tile_strictly_inside(tile, dir_view, sin_b, cos_b)
                }
                None => false,
            };
            in_horizon || in_lo
        };
        take.push(inside);
    }
    let mut coarse = Vec::with_capacity(n_heavy as usize);
    let mut fine = Vec::with_capacity(n_heavy as usize);
    for (offset, inside) in take.into_iter().enumerate() {
        let index = table.tile_index[start + offset];
        if inside {
            coarse.push(index);
        } else {
            fine.push(index);
        }
    }
    let n_coarse = coarse.len();
    for (offset, index) in coarse.into_iter().chain(fine).enumerate() {
        table.tile_index[start + offset] = index;
    }
    n_coarse as u32
}

fn old_mask_is_mapsolo(table: &FarTableGpu, mask: u32) -> bool {
    if mask.count_ones() != 1 {
        return false;
    }
    let k = mask.trailing_zeros() as usize;
    let n = (table.header[0] as usize).min(MAX_FAR_BODIES);
    if k >= n {
        return false;
    }
    let shape = table.body[k].atmosphere[3];
    (3.5..4.5).contains(&shape)
}

fn old_count_mapsolo(table: &FarTableGpu, mapsolo: bool) -> u32 {
    if !mapsolo {
        return 0;
    }
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| old_mask_is_mapsolo(table, **mask))
        .count() as u32
}

fn old_partition_mapsolo(table: &mut FarTableGpu) -> u32 {
    let (n_base, n_sphere, n_heavy) = old_tile_split(table);
    if n_heavy == 0 {
        return 0;
    }
    let start = (n_base + n_sphere) as usize;
    let end = start + n_heavy as usize;
    if end > table.tile_index.len() {
        return 0;
    }
    let mut solo = Vec::with_capacity(n_heavy as usize);
    let mut rest = Vec::with_capacity(n_heavy as usize);
    for slot in start..end {
        let index = table.tile_index[slot];
        let mask = table.tile_mask.get(index as usize).copied().unwrap_or(0);
        if old_mask_is_mapsolo(table, mask) {
            solo.push(index);
        } else {
            rest.push(index);
        }
    }
    let n = solo.len();
    for (offset, index) in solo.into_iter().chain(rest).enumerate() {
        table.tile_index[start + offset] = index;
    }
    n as u32
}

fn old_body_pipe(table: &FarTableGpu) -> SkyBodyPipe {
    if old_has_mapped(table) {
        SkyBodyPipe::Full
    } else if old_heavy_mask(table) != 0 {
        SkyBodyPipe::NoMap
    } else {
        SkyBodyPipe::Sphere
    }
}

fn old_from_table(table: &FarTableGpu, cull: bool, mapsolo: bool) -> SkyDraw {
    let bodies = table.header[0];
    let tile_px = table.header[1];
    let tiles_x = table.header[2];
    let tiles_y = table.header[3];
    let body = old_body_pipe(table);
    let tiled = cull && tile_px != 0 && tiles_x != 0 && tiles_y != 0 && bodies != 0;
    if !tiled {
        return SkyDraw {
            base: bodies == 0,
            body,
            ..SkyDraw::default()
        };
    }
    let (n_base, n_sphere, n_heavy) = old_tile_split(table);
    SkyDraw {
        quads: true,
        body,
        n_base,
        n_sphere,
        n_heavy,
        n_mapsolo: old_count_mapsolo(table, mapsolo),
        ..SkyDraw::default()
    }
}

/// `(kept index, rho, map)` of the horizon tables `publish_horizons` built,
/// in table order.
pub(super) fn old_horizon_candidates(
    table: &FarTableGpu,
    map_max: &[f32; MAX_FAR_MAPS],
) -> Vec<(usize, f32, usize)> {
    let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
    struct Cand {
        index: usize,
        rho: f32,
        map: usize,
    }
    let mut cands = Vec::new();
    for k in 0..kept {
        let gpu = &table.body[k];
        let shape = gpu.atmosphere[3];
        if !(3.5..4.5).contains(&shape) {
            continue;
        }
        let map_plus = gpu.seed[2];
        if map_plus == 0 || map_plus as usize > MAX_FAR_MAPS {
            continue;
        }
        let map = (map_plus as usize) - 1;
        let distance = gpu.albedo2[3];
        let rho = gpu.dir_rho[3];
        let air = gpu.albedo1[3];
        let max_off = map_max[map];
        if !(distance > 0.0)
            || !distance.is_finite()
            || !rho.is_finite()
            || !air.is_finite()
            || !max_off.is_finite()
        {
            continue;
        }
        let radius = rho * distance;
        let reach = radius + max_off + air.max(0.0);
        if !(distance <= reach + distance * 1.0e-5) {
            continue;
        }
        cands.push(Cand { index: k, rho, map });
    }
    cands.sort_by(|a, b| b.rho.total_cmp(&a.rho).then(a.index.cmp(&b.index)));
    cands.truncate(HORIZON_TABLES);
    cands.iter().map(|c| (c.index, c.rho, c.map)).collect()
}

/// The host half of the old `FarBodyRing::write` after the horizon tables
/// are in: `publish_horizons`' tile clear, `SkyDraw::from_table`, then the
/// mapsolo and coarse partitions. `frames` were rebuilt for `view` first.
pub(super) fn old_classify(
    table: &mut FarTableGpu,
    view: Option<&FarView>,
    frames: &TileFrames,
    coarse: Option<&SkyCoarseQuery>,
    map_min: &[f32; MAX_FAR_MAPS],
    cull: bool,
    mapsolo: bool,
) -> SkyDraw {
    if let Some(view) = view
        && frames.matches(view)
    {
        old_clear_tiles_above_horizon(table, frames, view, cull);
    }
    let mut draw = old_from_table(table, cull, mapsolo);
    if draw.quads {
        if mapsolo {
            let n = old_partition_mapsolo(table);
            debug_assert_eq!(n, draw.n_mapsolo);
            draw.n_mapsolo = n;
        }
        if let (Some(query), Some(view)) = (coarse, view) {
            if frames.matches(view) {
                draw.n_coarse = old_split_coarse_base(table, frames, view, query);
                draw.n_coarse_far = old_split_coarse_far(table, frames, view, map_min);
            }
        }
    }
    draw
}
