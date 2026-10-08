//! `VOXEL_SKY_DEBUG=1`: once a second, the ground body behind the horizon dip
//! and each published horizon table at the camera's forward azimuth.

use super::horizon::{ground_pick, horizon_axes, horizon_azimuth, horizon_bin};
use super::table::{FarTableGpu, HORIZON_BINS, HORIZON_TABLES};
use super::view::{FarView, ViewBasis};
use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};

/// Once a second while `VOXEL_SKY_DEBUG=1`. The dip is the value returned to
/// the UBO. `index` is the pre-frustum list [`horizon_dip`] walks. Horizon
/// lines are the kept-body slot the shader indexes, with that table's bin at
/// the camera's forward azimuth.
///
/// [`horizon_dip`]: super::horizon::horizon_dip
pub(super) fn log_sky_debug(
    bodies: &[FarBody],
    table: &FarTableGpu,
    view: Option<&FarView>,
    dip: f32,
    map_min: &[f32; MAX_FAR_MAPS],
    sky_up: glam::Vec3,
) {
    if !sky_debug_log_due() {
        return;
    }
    let n = bodies.len().min(MAX_FAR_BODIES);
    let listed = &bodies[..n];
    match ground_pick(listed, sky_up, map_min) {
        Some((index, rho)) => {
            let shape = sky_debug_shape(&listed[index].shape);
            eprintln!("sky-debug dip={dip} ground index={index} shape={shape} rho={rho}");
        }
        None => eprintln!("sky-debug dip={dip} ground=none"),
    }
    let forward = view.and_then(camera_forward);
    for slot in 0..HORIZON_TABLES {
        let kept = table.horizon_id[slot];
        if kept == u32::MAX {
            continue;
        }
        let kept_us = kept as usize;
        if kept_us >= MAX_FAR_BODIES {
            continue;
        }
        let start = slot * HORIZON_BINS;
        let bins = &table.horizon_sin[start..start + HORIZON_BINS];
        let mut lo = bins[0];
        let mut hi = bins[0];
        for v in &bins[1..] {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let gpu = &table.body[kept_us];
        let map_plus = gpu.seed[2];
        let map = map_plus.saturating_sub(1);
        let center = glam::Vec3::new(gpu.dir_rho[0], gpu.dir_rho[1], gpu.dir_rho[2]);
        match forward.and_then(|dir| horizon_forward_sample(center, dir, bins)) {
            Some((az, bin, sine)) => eprintln!(
                "sky-debug horizon slot={slot} kept={kept} map={map} min={lo} max={hi} forward_az={az} bin={bin} sin={sine}"
            ),
            None => eprintln!(
                "sky-debug horizon slot={slot} kept={kept} map={map} min={lo} max={hi} forward=none"
            ),
        }
    }
}

fn sky_debug_shape(shape: &FarShape) -> String {
    match *shape {
        FarShape::Cube => "cube".to_string(),
        FarShape::Sphere => "sphere".to_string(),
        FarShape::InnerSphere => "inner".to_string(),
        FarShape::Rounded { exponent } => format!("rounded({exponent})"),
        FarShape::Mapped { map, horizon, air } => {
            format!("mapped(map={}, horizon={horizon}, air={air})", map.0)
        }
    }
}

/// Look direction. Row 3 of `view_proj` points that way; [`ViewBasis::back`]
/// is the camera's +Z, opposite the look.
fn camera_forward(view: &FarView) -> Option<glam::Vec3> {
    ViewBasis::from_view_proj(view.view_proj).map(|basis| -basis.back)
}

/// `(azimuth, bin, sine)` of `forward` in the horizon frame whose up is
/// centre → eye (`-center`). The bin matches `far_above_horizon`.
fn horizon_forward_sample(
    center: glam::Vec3,
    forward: glam::Vec3,
    bins: &[f32],
) -> Option<(f32, usize, f32)> {
    let up = -center;
    let (east, north) = horizon_axes(up)?;
    let len2 = up.length_squared();
    let up = up / len2.sqrt();
    let az = horizon_azimuth(forward, up, east, north);
    let bin = horizon_bin(az);
    bins.get(bin).copied().map(|sine| (az, bin, sine))
}

fn sky_debug_log_due() -> bool {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static NEXT: Mutex<Option<Instant>> = Mutex::new(None);
    let Ok(mut next) = NEXT.lock() else {
        return false;
    };
    let now = Instant::now();
    if next.is_some_and(|t| now < t) {
        return false;
    }
    *next = Some(now + Duration::from_secs(1));
    true
}
