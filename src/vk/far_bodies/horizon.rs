//! The geometric horizon dip, the azimuthal horizon frame and the tile test
//! against a published table, and the per-ring cache of built tables.

use super::cones::{add_angles, unit_dir};
use super::table::{FarTableGpu, HORIZON_BINS, HORIZON_TABLES};
use super::tiles::{TileRuns, TileSample, TileView, fill_tile_lists, used_tiles};
use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};

/// Index in `bodies` (the pre-frustum list [`horizon_dip`] walks) and the rho
/// that dip used. `None` when no outside body sits under the viewer.
pub(super) fn ground_pick(
    bodies: &[FarBody],
    sky_up: glam::Vec3,
    map_min: &[f32; MAX_FAR_MAPS],
) -> Option<(usize, f32)> {
    let (_, up, _) = crate::vk::uniforms::local_sky_basis(sky_up);
    let down = -up;
    let mut best = -1.0f32;
    let mut index = None;
    for (i, body) in bodies.iter().take(MAX_FAR_BODIES).enumerate() {
        let rho = horizon_rho(body, map_min);
        if !(rho.is_finite() && rho > 0.0 && rho < 1.0) {
            continue;
        }
        let dir = body.dir;
        let len2 = dir.length_squared();
        if !(len2 > 0.0) || !dir.is_finite() {
            continue;
        }
        if dir.dot(down) > 0.5 * len2.sqrt() && rho > best {
            best = rho;
            index = Some(i);
        }
    }
    index.map(|i| (i, best))
}

/// Sine of the geometric horizon dip below the local horizontal.
///
/// The ground body is the outside body (`0 < rho < 1`) with the largest rho
/// whose centre lies under the viewer: `dot(dir, -sky_up) > 0.5`. `rho` is
/// `radius/distance`. A [`FarShape::Mapped`] body uses the lo sphere instead,
/// `(radius + map_min[map]) / distance`: the deepest geometric horizon, so
/// the sky line stays under the drawn silhouette. An id past the table, or a
/// non-finite offset, uses offset 0 (the reference radius). An empty slot is
/// already 0. `sky_up` is the same local up [`crate::vk::uniforms::local_sky_basis`]
/// uses, so a zero or non-unit up matches the sky frame. No such body yields
/// `0`. Otherwise `s = sqrt(max(1 - rho², 0))`, clamped to `[0, 0.5]`.
///
/// Called from packing on the bodies that pack keeps (the list past `keep`,
/// not the frustum survivors). Fog and water evaluate `sky_radiance` too, so
/// the dip must not pop when the ground body leaves the sky frustum.
/// `map_min[i]` is map `i`'s minimum datum offset, the same table as the
/// lo-sphere disc.
pub(super) fn horizon_dip(
    bodies: &[FarBody],
    sky_up: glam::Vec3,
    map_min: &[f32; MAX_FAR_MAPS],
) -> f32 {
    let Some((_, rho)) = ground_pick(bodies, sky_up, map_min) else {
        return 0.0;
    };
    (1.0 - rho * rho).max(0.0).sqrt().clamp(0.0, 0.5)
}

/// `radius/distance`. A mapped body adds the map's minimum datum offset first
/// (the lo sphere). A missing or non-finite offset leaves the reference radius.
fn horizon_rho(body: &FarBody, map_min: &[f32; MAX_FAR_MAPS]) -> f32 {
    let mut radius = body.radius;
    if let FarShape::Mapped { map, .. } = body.shape {
        let slot = map.0 as usize;
        if slot < MAX_FAR_MAPS {
            let min_off = map_min[slot];
            if min_off.is_finite() {
                radius += min_off;
            }
        }
    }
    radius / body.distance
}

/// `(east, north)` for `up`. `ref` is +Y when `|up.y| < 0.9`, else +X.
/// `east = normalize(cross(ref, up))`, `north = cross(up, east)`.
pub(super) fn horizon_axes(up: glam::Vec3) -> Option<(glam::Vec3, glam::Vec3)> {
    let len2 = up.length_squared();
    if !(len2 > 0.0) || !up.is_finite() {
        return None;
    }
    let up = up / len2.sqrt();
    let reference = if up.y.abs() < 0.9 {
        glam::Vec3::Y
    } else {
        glam::Vec3::X
    };
    let east = reference.cross(up);
    let elen2 = east.length_squared();
    if !(elen2 > 1.0e-20) || !east.is_finite() {
        return None;
    }
    let east = east / elen2.sqrt();
    let north = up.cross(east);
    north.is_finite().then_some((east, north))
}

/// Azimuth of `dir` in `[0, 2π)`. `atan2(east, north)`, matching the shader.
pub(super) fn horizon_azimuth(
    dir: glam::Vec3,
    up: glam::Vec3,
    east: glam::Vec3,
    north: glam::Vec3,
) -> f32 {
    let horiz = dir - up * dir.dot(up);
    let az = horiz.dot(east).atan2(horiz.dot(north));
    if az < 0.0 {
        az + std::f32::consts::TAU
    } else {
        az
    }
}

pub(super) fn horizon_bin(az: f32) -> usize {
    let n = HORIZON_BINS as f32;
    let i = (az / std::f32::consts::TAU * n).floor() as i32;
    i.rem_euclid(HORIZON_BINS as i32) as usize
}

/// Azimuthal half-width of a cap of radius `beta` at polar angle `gamma`.
/// `None` when the cap contains either pole and therefore every bin.
fn azimuth_half(beta: f32, gamma: f32) -> Option<f32> {
    if !(beta >= 0.0 && gamma >= 0.0) || !beta.is_finite() || !gamma.is_finite() {
        return None;
    }
    if gamma <= beta + 1.0e-6 || gamma + beta >= std::f32::consts::PI - 1.0e-6 {
        return None;
    }
    let s = (beta.sin() / gamma.sin()).clamp(0.0, 1.0);
    Some(s.asin())
}

/// `true` when `ray` is strictly above `bins`. `ray` need not be unit.
#[cfg(test)]
pub(super) fn ray_above_horizon(
    bins: &[f32; HORIZON_BINS],
    up: glam::Vec3,
    east: glam::Vec3,
    north: glam::Vec3,
    ray: glam::Vec3,
) -> bool {
    let Some(ray) = unit_dir(ray) else {
        return false;
    };
    let Some(up) = unit_dir(up) else {
        return false;
    };
    let mu = ray.dot(up);
    let az = horizon_azimuth(ray, up, east, north);
    mu > bins[horizon_bin(az)]
}

/// The whole tile cone, widened by `px_margin` radians (2 px when that is
/// [`FarView::px_max`]), lies strictly above every bin its azimuth range
/// overlaps. The stored cone already carries the tile's angular radius.
pub(super) fn tile_above_horizon(
    tile: &TileSample,
    bins: &[f32; HORIZON_BINS],
    up: glam::Vec3,
    east: glam::Vec3,
    north: glam::Vec3,
    px_margin: f32,
) -> bool {
    if !(px_margin.is_finite() && (0.0..std::f32::consts::FRAC_PI_2).contains(&px_margin)) {
        return false;
    }
    let (sin_a, cos_a) = add_angles(tile.cos_r, px_margin.cos());
    if sin_a > 1.0 || !(sin_a.is_finite() && cos_a.is_finite()) {
        return false;
    }
    let mu = tile.centre.dot(up).clamp(-1.0, 1.0);
    let sin_g = (1.0 - mu * mu).max(0.0).sqrt();
    let min_elev = mu * cos_a - sin_g * sin_a;
    if !min_elev.is_finite() {
        return false;
    }
    let beta = sin_a.clamp(0.0, 1.0).asin();
    let gamma = mu.acos();
    let half = azimuth_half(beta, gamma).map(|h| h + std::f32::consts::TAU / HORIZON_BINS as f32);
    let az = horizon_azimuth(tile.centre, up, east, north);
    let max_bin = match half {
        None => bins.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        Some(h) => {
            let width = std::f32::consts::TAU / HORIZON_BINS as f32;
            let mut i0 = ((az - h) / width).floor() as i32;
            let i1 = ((az + h) / width).floor() as i32;
            if i1 - i0 >= HORIZON_BINS as i32 {
                bins.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            } else {
                let mut best = f32::NEG_INFINITY;
                while i0 <= i1 {
                    let idx = i0.rem_euclid(HORIZON_BINS as i32) as usize;
                    best = best.max(bins[idx]);
                    i0 += 1;
                }
                best
            }
        }
    };
    min_elev > max_bin
}

/// Drop a published body's bit from every tile whose cone is above that
/// body's table. Refills the tile lists when a bit changes and returns the
/// new runs; `None` leaves the lists as they were. Culling off
/// (`VOXEL_FAR_CULL=0`) does not call this and leaves the mask alone; the
/// shader test still runs.
pub(super) fn clear_tiles_above_horizon(
    table: &mut FarTableGpu,
    tiles: &TileView,
) -> Option<TileRuns> {
    let TileView {
        frames,
        view,
        basis,
    } = tiles;
    let n = used_tiles(table).min(frames.samples.len());
    let mut changed = false;
    for slot in 0..HORIZON_TABLES {
        let id = table.horizon_id[slot];
        if id == u32::MAX || id as usize >= MAX_FAR_BODIES {
            continue;
        }
        let Some(dir) = unit_dir(table.body[id as usize].dir()) else {
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
    changed.then(|| fill_tile_lists(table, view.width, view.height))
}

/// A Mapped body whose hi+air ball holds the eye: its kept index, reference
/// rho and far-map slot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct HorizonCand {
    pub(super) index: usize,
    pub(super) rho: f32,
    pub(super) map: usize,
}

/// The bodies that get a horizon table, in table order: the kept Mapped
/// bodies whose hi+air ball contains the eye, the largest reference rho
/// first (ties in kept order), at most [`HORIZON_TABLES`]. `map_max[i]` is
/// map `i`'s maximum datum offset. No allocation.
pub(super) fn horizon_candidates(
    table: &FarTableGpu,
    map_max: &[f32; MAX_FAR_MAPS],
) -> [Option<HorizonCand>; HORIZON_TABLES] {
    let mut all = [HorizonCand {
        index: 0,
        rho: 0.0,
        map: 0,
    }; MAX_FAR_BODIES];
    let mut n = 0usize;
    for (k, gpu) in table.body[..table.kept()].iter().enumerate() {
        if !gpu.is_mapped() {
            continue;
        }
        let Some(map) = gpu.map_index() else {
            continue;
        };
        let distance = gpu.distance();
        let rho = gpu.rho();
        let air = gpu.air();
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
        all[n] = HorizonCand { index: k, rho, map };
        n += 1;
    }
    // The kept indices are distinct, so the order is total and the unstable
    // sort gives the stable one.
    all[..n].sort_unstable_by(|a, b| b.rho.total_cmp(&a.rho).then(a.index.cmp(&b.index)));
    std::array::from_fn(|slot| all[..n].get(slot).copied())
}

/// Eye motion that keeps a cached table, as a fraction of the altitude
/// above the lo sphere.
const HORIZON_MOVE_FRAC: f32 = 0.001;

/// One cached horizon table, keyed by the map and the body that was built.
pub(super) struct HorizonSlot {
    pub(super) live: bool,
    pub(super) map: u32,
    pub(super) generation: u32,
    pub(super) radius_bits: u32,
    pub(super) air_bits: u32,
    pub(super) px_bits: u32,
    pub(super) rot_bits: [u32; 4],
    /// Centre → eye, world space, at the build.
    pub(super) eye: glam::Vec3,
    /// Altitude above the lo sphere at the build. The motion limit is
    /// [`HORIZON_MOVE_FRAC`] of this.
    pub(super) altitude: f32,
    pub(super) bins: [f32; HORIZON_BINS],
}

impl HorizonSlot {
    pub(super) fn empty() -> Self {
        Self {
            live: false,
            map: u32::MAX,
            generation: 0,
            radius_bits: 0,
            air_bits: 0,
            px_bits: 0,
            rot_bits: [0; 4],
            eye: glam::Vec3::ZERO,
            altitude: 0.0,
            bins: [1.0; HORIZON_BINS],
        }
    }
}

pub(super) struct HorizonCache {
    pub(super) slots: [HorizonSlot; HORIZON_TABLES],
}

impl HorizonCache {
    pub(super) fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| HorizonSlot::empty()),
        }
    }
}

/// `true` when `eye` is within 0.1% of the cached altitude above the lo
/// sphere and the body inputs still match. A non-positive cached altitude
/// keeps the table only when the eye has not moved.
pub(super) fn horizon_cache_hit(
    slot: &HorizonSlot,
    map: u32,
    generation: u32,
    radius_bits: u32,
    air_bits: u32,
    px_bits: u32,
    rot_bits: [u32; 4],
    eye: glam::Vec3,
) -> bool {
    if !slot.live
        || slot.map != map
        || slot.generation != generation
        || slot.radius_bits != radius_bits
        || slot.air_bits != air_bits
        || slot.px_bits != px_bits
        || slot.rot_bits != rot_bits
    {
        return false;
    }
    let moved = (slot.eye - eye).length();
    if !moved.is_finite() {
        return false;
    }
    let limit = if slot.altitude > 0.0 {
        HORIZON_MOVE_FRAC * slot.altitude
    } else {
        0.0
    };
    moved <= limit
}

pub(super) fn cache_dest(cache: &HorizonCache, used: &[bool; HORIZON_TABLES], map: u32) -> usize {
    if let Some(i) = cache
        .slots
        .iter()
        .enumerate()
        .position(|(i, slot)| !used[i] && slot.live && slot.map == map)
    {
        return i;
    }
    if let Some(i) = used.iter().position(|u| !*u) {
        return i;
    }
    0
}
