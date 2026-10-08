//! Angular extents of a far body: the per-pixel cone sine the shader rejects
//! with, and the half-angles of the mapped horizon, interior and lo-sphere
//! discs that the cull, the tile mask and the coarse split share.

use super::far_cull_enabled;
use super::table::FarBodyGpu;
use crate::far_body::{FarBody, FarShape, MAX_FAR_MAPS};

/// Sine of the angular radius the shader can draw (`s = ‖ray × center‖`), or
/// `-1` when a ray facing away from the centre can still draw something.
///
/// Re-checked against `far_bodies()`:
///
/// * **Sphere, camera outside (`rho < 1`).** The disc, the soft point and the
///   air rim all sit behind `if (!inner && !rounded && !mapped && facing <= 0) continue`.
///   Nothing in the back hemisphere is drawn, so the drawable set is the facing
///   hemisphere. The rim reaches `1.05 * rho`. Past sine 1 that is the whole
///   hemisphere: the pixel test with bound `1` is only `f > 0`, because
///   `|ray × dir| <= 1` always. Clamping there, instead of the `-1` sentinel,
///   is what stops a home planet (`rho ≈ 1`) from painting every tile.
/// * **Cube.** Same facing gate, so a back-hemisphere ray is not shaded. The
///   centre cone is still not a safe reject once the camera is inside or near
///   the corner sphere (`√3 * 1.035 * rho >= 0.99`): a rotated cube then
///   surrounds the camera, and a sine below 1 drops a face that still covers
///   the view. Sentinel only in that case; otherwise the sine is
///   `√3 * 1.035 * rho` (corners of the 1.035-grown cube, the rim's reach).
/// * **Rounded.** No facing gate. `far_ray_rounded` can meet a bulge on a ray
///   aimed away from the centre while the camera is still outside the solid,
///   and the air rim tests `facing < bestT` rather than `facing > 0` (the cone's
///   own `f <= 0` is what kills the antipodal ghost). Keep
///   `1.05 * rhoB >= 0.99 → -1`.
/// * **Inner sphere.** The far wall is every direction (`far_ray_inner` has no
///   facing reject, and the branch runs before the facing gate). Always `-1`.
/// * **Mapped.** No facing gate. The datum bulges out to the hi radius
///   `hi = (radius + map_max[map]) / distance` (`map_max[i]` is map `i`'s
///   maximum datum offset, or 0 when that map is unset). The air limb reaches
///   `hi + air/distance`: the air a ray can cross never extends past the hi
///   sphere plus air, and the shader's extra 3 px does not cover a thick
///   shell. A ray aimed away from the centre can still meet the body once
///   that sphere contains the camera. Sentinel when the widened bound is
///   `>= 0.99`; otherwise the sine is `hi + air/distance` (a negative radius
///   or air floors at 0, so it is not mistaken for the sentinel).
///
///   The sentinel is only the per-pixel reject. The tile mask uses the horizon
///   cone ([`mapped_horizon_half`]) whenever `horizon < 1`, including when this
///   bound is `-1`, so sky tiles above the limb do not take the mapped march.
///   `horizon >= 1` disables that cone and keeps today's tile coverage.
pub(super) fn cone_bound(body: &FarBody, map_max: &[f32; MAX_FAR_MAPS]) -> f32 {
    if !far_cull_enabled() {
        return -1.0;
    }
    let rho = body.radius / body.distance;
    match body.shape {
        FarShape::InnerSphere => -1.0,
        FarShape::Sphere => {
            let bound = 1.05 * rho;
            if !bound.is_finite() {
                -1.0
            } else if bound < 1.0 {
                bound
            } else {
                1.0
            }
        }
        FarShape::Cube => {
            let bound = 3.0f32.sqrt() * 1.035 * rho;
            if !bound.is_finite() || bound >= 0.99 {
                -1.0
            } else {
                bound
            }
        }
        FarShape::Rounded { exponent } => {
            let p = exponent.clamp(2.0, 32.0);
            let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
            let bound = 1.05 * rho_b;
            if !bound.is_finite() || bound >= 0.99 {
                -1.0
            } else {
                bound
            }
        }
        FarShape::Mapped { map, air, .. } => {
            let max_off = if (map.0 as usize) < MAX_FAR_MAPS {
                map_max[map.0 as usize]
            } else {
                0.0
            };
            let hi = ((body.radius + max_off) / body.distance).max(0.0);
            // World air, same space as `hi`. The 3 px margin on this sine is
            // not the shell.
            let shell = (air / body.distance).max(0.0);
            let bound = hi + shell;
            if !bound.is_finite() || bound >= 0.99 {
                -1.0
            } else {
                bound
            }
        }
    }
}

/// `(sin(a+b), cos(a+b))` from the cosines of two angles in `[0, π]`. A sum
/// past π (`sin < 0`) is stored as a radius that covers every direction
/// (`sin > 1` is the flag; a real sine never is).
pub(super) fn add_angles(cos_a: f32, cos_b: f32) -> (f32, f32) {
    let sin_a = (1.0 - cos_a * cos_a).max(0.0).sqrt();
    let sin_b = (1.0 - cos_b * cos_b).max(0.0).sqrt();
    let sin_r = sin_a * cos_b + cos_a * sin_b;
    let cos_r = (cos_a * cos_b - sin_a * sin_b).clamp(-1.0, 1.0);
    if sin_r < 0.0 {
        (2.0, -1.0)
    } else {
        (sin_r, cos_r)
    }
}

/// Half-angle of a mapped body's tile cone, as `(sin, cos)`.
///
/// The shader keeps a ray when `dot(ray, -dir) <= horizon`: a cone around
/// `+dir` of half-angle `π/2 + asin(horizon)`. `cos` of that angle is
/// `-horizon`, so a positive horizon (terrain above the plane normal to
/// `-dir`) is wider than a hemisphere and `cos` is negative. The air limb
/// widens the angle by `δ` with `sin δ = air/distance` (a negative thickness
/// floors at 0), then by the same 3 px sine pad as every other cone. `None`
/// paints every tile: `horizon >= 1`, a non-finite input, or a sum past π.
pub(super) fn mapped_horizon_half(
    horizon: f32,
    air: f32,
    distance: f32,
    px_max: f32,
) -> Option<(f32, f32)> {
    if !(horizon < 1.0) || !horizon.is_finite() {
        return None;
    }
    let cos_alpha = -horizon.clamp(-1.0, 1.0);
    let shell = if distance.is_finite() && distance > 0.0 && air.is_finite() {
        (air / distance).max(0.0)
    } else {
        return None;
    };
    let px = if px_max.is_finite() {
        3.0 * px_max.max(0.0)
    } else {
        return None;
    };
    let pad = shell + px;
    if !pad.is_finite() {
        return None;
    }
    // `δ = 90°` when the pad's sine saturates. A wider cone falls out of
    // [`add_angles`] as the full-sphere flag.
    let sin_delta = pad.min(1.0);
    let cos_delta = (1.0 - sin_delta * sin_delta).max(0.0).sqrt();
    let (sin_r, cos_r) = add_angles(cos_alpha, cos_delta);
    if sin_r > 1.0 || !sin_r.is_finite() || !cos_r.is_finite() {
        None
    } else {
        Some((sin_r, cos_r))
    }
}

/// Half-angle of the mapped disc inset by the air limb and 3 px, as `(sin, cos)`.
///
/// This is the horizon cone ([`mapped_horizon_half`] without the widening)
/// shrunk by `δ`, `sin δ = air/distance + 3 px`. A tile that lies strictly
/// inside it holds no silhouette, limb, or air rim. `None` when `horizon >= 1`
/// or the margin eats the cone (including a pad whose sine saturates).
pub(super) fn mapped_interior_half(
    horizon: f32,
    air: f32,
    distance: f32,
    px_max: f32,
) -> Option<(f32, f32)> {
    if !(horizon < 1.0) || !horizon.is_finite() {
        return None;
    }
    let h = horizon.clamp(-1.0, 1.0);
    let cos_alpha = -h;
    let sin_alpha = (1.0 - h * h).max(0.0).sqrt();
    let shell = if distance.is_finite() && distance > 0.0 && air.is_finite() {
        (air / distance).max(0.0)
    } else {
        return None;
    };
    let px = if px_max.is_finite() {
        3.0 * px_max.max(0.0)
    } else {
        return None;
    };
    let pad = shell + px;
    // A sine of 1 is 90°. Anything wider is not a margin we can subtract, and
    // the limb could sit anywhere inside that shell.
    if !pad.is_finite() || pad >= 1.0 {
        return None;
    }
    let sin_delta = pad;
    let cos_delta = (1.0 - sin_delta * sin_delta).max(0.0).sqrt();
    let sin_b = sin_alpha * cos_delta - cos_alpha * sin_delta;
    let cos_b = cos_alpha * cos_delta + sin_alpha * sin_delta;
    if !(sin_b > 0.0) || !sin_b.is_finite() || !cos_b.is_finite() {
        None
    } else {
        Some((sin_b, cos_b))
    }
}

/// Half-angle of the lo-sphere disc, shrunk by `px_margin` radians, as `(sin, cos)`.
///
/// The surface is everywhere at least the lo sphere, so with the eye outside
/// that sphere (`0 < rho_lo < 1`) every ray inside the disc of half-angle
/// `asin(rho_lo)` around the body centre hits the surface. `px_margin` is the
/// extra inset past the tile cone (2 px when it is [`FarView::px_max`]).
/// `None` when the eye is not strictly outside the lo sphere or the margin
/// eats the disc.
///
/// [`FarView::px_max`]: super::view::FarView::px_max
pub(super) fn lo_disc_interior(rho_lo: f32, px_margin: f32) -> Option<(f32, f32)> {
    if !(rho_lo > 0.0 && rho_lo < 1.0) || !rho_lo.is_finite() {
        return None;
    }
    if !px_margin.is_finite() || px_margin < 0.0 {
        return None;
    }
    let sin_a = rho_lo;
    let cos_a = (1.0 - sin_a * sin_a).max(0.0).sqrt();
    let sin_d = px_margin.sin();
    let cos_d = px_margin.cos();
    // A margin of 90° or more leaves no disc a tile can sit inside.
    if !(sin_d >= 0.0 && cos_d > 0.0) || !sin_d.is_finite() || !cos_d.is_finite() {
        return None;
    }
    let sin_b = sin_a * cos_d - cos_a * sin_d;
    let cos_b = cos_a * cos_d + sin_a * sin_d;
    if !(sin_b > 0.0) || !sin_b.is_finite() || !cos_b.is_finite() {
        None
    } else {
        Some((sin_b, cos_b))
    }
}

/// `radius/distance + min_offset/distance` for a mapped body. `None` when the
/// record is not mapped or the distance is unusable.
pub(super) fn mapped_rho_lo(gpu: &FarBodyGpu, map_min: &[f32; MAX_FAR_MAPS]) -> Option<f32> {
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

pub(super) fn unit_dir(v: glam::Vec3) -> Option<glam::Vec3> {
    let len2 = v.length_squared();
    if !(len2 > 0.0) || !v.is_finite() {
        return None;
    }
    Some(v / len2.sqrt())
}
