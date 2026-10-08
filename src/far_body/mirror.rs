//! Test-only CPU mirror of the ray tests in `shaders/far_body.slang`, the
//! references that check it, and fixtures shared with the tests in
//! `src/vk/far_bodies.rs`. The shader is the copy that runs. Constants written
//! out on both sides are pinned by `tests::mirror_constants_match_the_shader`.

use glam::{Quat, Vec3};

use super::far_map_basis;

/// A hit in normalised space. `t` is along the unit view ray. `normal` is world
/// space. `face` is the cube face (0 = +X … 5 = −Z), the dominant axis of a
/// rounded body's normal, and 0 for a sphere.
/// Test-only: the sky shader is the copy that runs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FarHit {
    pub t: f32,
    pub normal: Vec3,
    pub face: u32,
}

/// `radius/distance` in f32. Exact for distances the catalog actually uses.
pub(crate) fn normalised_radius(distance: f32, radius: f32) -> f32 {
    radius / distance
}

/// glam's `Quat * Vec3`, written out so the shader can use the same arithmetic.
/// Does not normalise: a packed unit quaternion is already unit.
pub(super) fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let b = Vec3::new(q.x, q.y, q.z);
    let w = q.w;
    v * (w * w - b.dot(b)) + b * (v.dot(b) * 2.0) + b.cross(v) * (w * 2.0)
}

pub(super) fn conjugate(q: Quat) -> Quat {
    Quat::from_xyzw(-q.x, -q.y, -q.z, q.w)
}

pub(super) const SLAB_EPS: f32 = 1e-8;
pub(super) const T_FAR: f32 = 1e30;

/// One axis of an AABB slab. `near_neg` is the −face of this axis.
fn slab_axis(
    o: f32,
    d: f32,
    rho: f32,
    t_min: &mut f32,
    t_max: &mut f32,
    face: &mut u32,
    axis: u32,
) -> bool {
    if d.abs() < SLAB_EPS {
        return o.abs() <= rho;
    }
    let inv = 1.0 / d;
    let t_neg = (-rho - o) * inv;
    let t_pos = (rho - o) * inv;
    let (t1, t2, near_neg) = if t_neg <= t_pos {
        (t_neg, t_pos, true)
    } else {
        (t_pos, t_neg, false)
    };
    if t1 > *t_min {
        *t_min = t1;
        *face = if near_neg { axis * 2 + 1 } else { axis * 2 };
    }
    if t2 < *t_max {
        *t_max = t2;
    }
    *t_min <= *t_max
}

/// Ray from the origin against a sphere centred on the unit `dir`.
/// `ray` and `dir` are unit. `rho` is the radius in that space, in (0, 1).
pub(crate) fn ray_sphere(ray: Vec3, dir: Vec3, rho: f32) -> Option<FarHit> {
    if !(rho > 0.0 && rho < 1.0) || !ray.is_finite() || !dir.is_finite() {
        return None;
    }
    let b = ray.dot(dir);
    let disc = b * b - (1.0 - rho * rho);
    if b <= 0.0 || disc < 0.0 {
        return None;
    }
    let t = b - disc.sqrt();
    if t <= 0.0 {
        return None;
    }
    Some(FarHit {
        t,
        normal: (ray * t - dir) / rho,
        face: 0,
    })
}

/// Ray from the origin against the far wall of a sphere the viewer is inside.
/// `dir` is unit, toward the centre. `distance` and `radius` are world units with
/// `distance < radius`. `t` is in the space where the centre sits at distance 1
/// (`t = b + sqrt(disc)`). The normal points toward the centre.
pub(crate) fn ray_inner_sphere(ray: Vec3, dir: Vec3, distance: f32, radius: f32) -> Option<FarHit> {
    if !(distance > 0.0 && radius > distance) || !ray.is_finite() || !dir.is_finite() {
        return None;
    }
    let rho = radius / distance;
    if !rho.is_finite() || !(rho > 1.0) {
        return None;
    }
    let b = ray.dot(dir);
    let disc = b * b - (1.0 - rho * rho);
    if disc < 0.0 {
        return None;
    }
    let t = b + disc.sqrt();
    if t <= 0.0 {
        return None;
    }
    Some(FarHit {
        t,
        normal: (dir - ray * t) / rho,
        face: 0,
    })
}

/// Ray from the origin against a cube of half-extent `rho` centred on unit `dir`.
/// `rotation` is body→world and is not re-normalised.
pub(crate) fn ray_cube(ray: Vec3, dir: Vec3, rho: f32, rotation: Quat) -> Option<FarHit> {
    if !(rho > 0.0 && rho < 1.0) || !ray.is_finite() || !dir.is_finite() {
        return None;
    }
    let inv = conjugate(rotation);
    let o = rotate(inv, -dir);
    let d = rotate(inv, ray);
    let mut t_min = 0.0;
    let mut t_max = T_FAR;
    let mut face = 0u32;
    if !slab_axis(o.x, d.x, rho, &mut t_min, &mut t_max, &mut face, 0)
        || !slab_axis(o.y, d.y, rho, &mut t_min, &mut t_max, &mut face, 1)
        || !slab_axis(o.z, d.z, rho, &mut t_min, &mut t_max, &mut face, 2)
        || t_min <= 0.0
    {
        return None;
    }
    let body_n = match face {
        0 => Vec3::X,
        1 => Vec3::NEG_X,
        2 => Vec3::Y,
        3 => Vec3::NEG_Y,
        4 => Vec3::Z,
        _ => Vec3::NEG_Z,
    };
    Some(FarHit {
        t: t_min,
        normal: rotate(rotation, body_n),
        face,
    })
}

/// Newton / secant steps on the rounded implicit. A radial hit lands in one step;
/// a graze spends the rest of the budget shrinking the bracket.
pub(super) const ROUNDED_STEPS: u32 = 6;
/// Relative pad on the enclosing sphere `rho · 3^(1/2 − 1/p)`.
pub(super) const ROUNDED_BOUND_PAD: f32 = 2.0e-4;
/// A closest-approach sample at most this factor outside the body is the hit.
pub(super) const ROUNDED_SNAP: f32 = 1.001;
/// Enclosing-sphere half-chord, relative to its radius, that is a graze.
pub(super) const ROUNDED_GRAZE: f32 = 0.02;
/// A grazing entry at most this factor outside the body is the sphere point.
pub(super) const ROUNDED_GRAZE_BAND: f32 = 1.01;
/// Relative band outside the body where the march stops.
pub(super) const ROUNDED_SETTLE: f32 = 2.0e-5;

/// Dominant axis of a body-space normal. Ties break toward X, then Y, then Z.
fn dom_face(n: Vec3) -> u32 {
    let a = n.abs();
    if a.x >= a.y && a.x >= a.z {
        if n.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if n.y >= 0.0 { 2 } else { 3 }
    } else if n.z >= 0.0 {
        4
    } else {
        5
    }
}

/// `s = ‖q‖_p` and its gradient. Zero at the origin.
pub(super) fn lp_grad(q: Vec3, p: f32) -> (f32, Vec3) {
    let ax = q.x.abs();
    let ay = q.y.abs();
    let az = q.z.abs();
    let m = ax.max(ay).max(az);
    if m == 0.0 {
        return (0.0, Vec3::ZERO);
    }
    let inv = 1.0 / m;
    let px = (ax * inv).powf(p);
    let py = (ay * inv).powf(p);
    let pz = (az * inv).powf(p);
    let sum = px + py + pz;
    if !(sum > 0.0) {
        return (0.0, Vec3::ZERO);
    }
    let s = m * sum.powf(1.0 / p);
    let mut g = Vec3::ZERO;
    if ax > 0.0 {
        g.x = px / sum * s / ax * q.x.signum();
    }
    if ay > 0.0 {
        g.y = py / sum * s / ay * q.y.signum();
    }
    if az > 0.0 {
        g.z = pz / sum * s / az * q.z.signum();
    }
    (s, g)
}

/// Hit at `t_hit` when that sample is outside the solid. Normal from the gradient.
fn rounded_at(
    ray_o: Vec3,
    ray_d: Vec3,
    rotation: Quat,
    t_hit: f32,
    p: f32,
    rho: f32,
) -> Option<FarHit> {
    if !(t_hit > 0.0) {
        return None;
    }
    let (s, g) = lp_grad(ray_o + ray_d * t_hit, p);
    if !(s >= rho) || !s.is_finite() {
        return None;
    }
    let glen = g.length();
    if !(glen > 1e-8) || !glen.is_finite() {
        return None;
    }
    let body_n = g / glen;
    Some(FarHit {
        t: t_hit,
        normal: rotate(rotation, body_n),
        face: dom_face(body_n),
    })
}

/// Ray from the origin against the superellipsoid of face radius `rho` centred
/// on unit `dir`. `exponent` is p ≥ 2. The enclosing sphere is
/// `rho · 3^(1/2 − 1/p)`, padded a hair so its entry stays outside the solid.
/// A miss that merely grazes that sphere, where the sphere lies on the body,
/// reports the sphere point rather than a hole.
pub(crate) fn ray_rounded(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    rotation: Quat,
    exponent: f32,
) -> Option<FarHit> {
    if !(rho > 0.0 && rho < 1.0)
        || !(exponent >= 2.0)
        || !exponent.is_finite()
        || !ray.is_finite()
        || !dir.is_finite()
    {
        return None;
    }
    let p = exponent.clamp(2.0, 32.0);
    // Exact bound, then a hair so `pow` cannot place the entry inside a corner.
    let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + ROUNDED_BOUND_PAD);
    let facing = ray.dot(dir);
    let disc = facing * facing - (1.0 - rho_b * rho_b);
    if !(disc >= 0.0) || !disc.is_finite() {
        return None;
    }
    let sd = disc.sqrt();
    let t_near = facing - sd;
    let t_far = facing + sd;
    if !(t_far > 0.0) {
        return None;
    }
    let inv = conjugate(rotation);
    let o = rotate(inv, -dir);
    let d = rotate(inv, ray);
    let t0 = t_near.max(0.0);
    let (s0, _) = lp_grad(o, p);
    if t0 == 0.0 && s0 <= rho {
        return None;
    }
    let t_c = facing.clamp(t0, t_far);
    let (s_c, _) = lp_grad(o + d * t_c, p);
    if s_c > rho {
        if s_c <= rho * ROUNDED_SNAP && t_c > 0.0 {
            if let Some(hit) = rounded_at(o, d, rotation, t_c, p, rho) {
                return Some(hit);
            }
        }
        // Grazing the enclosing sphere where that sphere meets the body.
        if t_near > 0.0 && sd <= rho_b * ROUNDED_GRAZE {
            let (s_b, _) = lp_grad(o + d * t_near, p);
            if s_b >= rho && s_b <= rho * ROUNDED_GRAZE_BAND {
                let n = (ray * t_near - dir) / rho_b;
                return Some(FarHit {
                    t: t_near,
                    normal: n,
                    face: dom_face(rotate(inv, n)),
                });
            }
        }
        return None;
    }
    let mut lo = t0;
    let mut hi = t_c;
    for _ in 0..ROUNDED_STEPS {
        let (s_l, g_l) = lp_grad(o + d * lo, p);
        if s_l >= rho && s_l <= rho * (1.0 + ROUNDED_SETTLE) {
            break;
        }
        if hi - lo < 1e-6 {
            break;
        }
        let slope = g_l.dot(d);
        let mut moved = false;
        if slope < -1e-8 {
            let t_n = lo - (s_l - rho) / slope;
            if t_n > lo && t_n < hi {
                let (s_n, _) = lp_grad(o + d * t_n, p);
                if s_n >= rho {
                    lo = t_n;
                    moved = true;
                } else {
                    hi = t_n;
                }
            }
        }
        if !moved {
            let (s_h, _) = lp_grad(o + d * hi, p);
            let mut t_f = 0.5 * (lo + hi);
            let span = s_l - s_h;
            if span.abs() > 1e-12 {
                let guess = lo + (hi - lo) * (s_l - rho) / span;
                if guess > lo && guess < hi {
                    t_f = guess;
                }
            }
            let (s_f, _) = lp_grad(o + d * t_f, p);
            if s_f >= rho {
                lo = t_f;
            } else {
                hi = t_f;
            }
        }
    }
    if let Some(hit) = rounded_at(o, d, rotation, lo, p, rho) {
        return Some(hit);
    }
    if t_near > 0.0 {
        let (s_b, _) = lp_grad(o + d * t_near, p);
        if s_b >= rho {
            let n = (ray * t_near - dir) / rho_b;
            return Some(FarHit {
                t: t_near,
                normal: n,
                face: dom_face(rotate(inv, n)),
            });
        }
    }
    None
}

/// Coefficients `c6 .. c0` of [`atan_approx`] as f32 bits, the `asfloat`
/// literals of `far_atan_approx`.
pub(super) const ATAN_BITS: [u32; 7] = [
    0x3c04_aa78,
    0xbd1a_a180,
    0x3dad_9679,
    0xbe0a_a033,
    0x3e4b_b99e,
    0xbeaa_a3a5,
    0x3f7f_fffb,
];

/// Odd degree-13 polynomial for `atan` on `[-1, 1]`.
///
/// A face of the equiangular chart only feeds this range (`dot(d, tu) /
/// dot(d, n)`), and the argument is clamped before the Horner step. Stepwise
/// f32 Horner (`p = p * u + c`, each multiply and add rounded) stays within
/// 6.61e-7 rad of `atan` on two million uniform samples, peaking near −0.972.
/// A datum cell is `π / (4 * 32)` ≈ 2.5e-2 rad, so the error is far below a cell.
pub(super) fn atan_approx(x: f32) -> f32 {
    let z = x.clamp(-1.0, 1.0);
    let u = z * z;
    let mut p = f32::from_bits(ATAN_BITS[0]);
    for bits in &ATAN_BITS[1..] {
        p = p * u + f32::from_bits(*bits);
    }
    z * p
}

/// Bilinear equiangular sample and the chart derivatives of that interpolant.
/// `g < 2` or a short slice is a flat zero. Four loads, matching the shader.
struct DatumSample {
    value: f32,
    /// `d(value) / dξ` inside the cell. Zero on the flat fallback.
    dr_dxi: f32,
    /// `d(value) / dη` inside the cell.
    dr_deta: f32,
}

fn sample_datum_d(g: u32, datum: &[f32], d: Vec3) -> DatumSample {
    let flat = DatumSample {
        value: 0.0,
        dr_dxi: 0.0,
        dr_deta: 0.0,
    };
    if g < 2 {
        return flat;
    }
    let gg = g as usize;
    let need = 6 * gg * gg;
    if datum.len() < need {
        return flat;
    }
    let face = dom_face(d) as usize;
    let (tu, n, tv) = far_map_basis(face);
    let den = d.dot(n);
    if den.abs() < 1e-8 {
        return flat;
    }
    let xi = (4.0 / std::f32::consts::PI) * atan_approx(d.dot(tu) / den);
    let eta = (4.0 / std::f32::consts::PI) * atan_approx(d.dot(tv) / den);
    let scale = 0.5 * (g - 1) as f32;
    let u = ((xi + 1.0) * scale).clamp(0.0, (g - 1) as f32);
    let v = ((eta + 1.0) * scale).clamp(0.0, (g - 1) as f32);
    let i0 = (u.floor() as u32).min(g - 1);
    let j0 = (v.floor() as u32).min(g - 1);
    let i1 = (i0 + 1).min(g - 1);
    let j1 = (j0 + 1).min(g - 1);
    let fu = u - i0 as f32;
    let fv = v - j0 as f32;
    let at = |j: u32, i: u32| datum[face * gg * gg + j as usize * gg + i as usize];
    let v00 = at(j0, i0);
    let v10 = at(j0, i1);
    let v01 = at(j1, i0);
    let v11 = at(j1, i1);
    let a = v00 + (v10 - v00) * fu;
    let b = v01 + (v11 - v01) * fu;
    // `u = (ξ + 1) * scale`, so the in-cell slope scales by the same factor.
    let dr_dfu = (v10 - v00) + ((v11 - v01) - (v10 - v00)) * fv;
    let dr_dfv = b - a;
    DatumSample {
        value: a + (b - a) * fv,
        dr_dxi: dr_dfu * scale,
        dr_deta: dr_dfv * scale,
    }
}

/// Manual bilinear of the equiangular datum. `g < 2` or a short slice is a flat
/// zero offset. Four loads, matching the shader (no hardware filtering).
pub(super) fn sample_datum(g: u32, datum: &[f32], d: Vec3) -> f32 {
    sample_datum_d(g, datum, d).value
}

/// Near and far roots of a sphere of radius `rho` centred on the unit `dir`.
fn sphere_roots(facing: f32, rho: f32) -> Option<(f32, f32)> {
    if !(rho > 0.0) || !rho.is_finite() {
        return None;
    }
    let disc = facing * facing - (1.0 - rho * rho);
    if disc < 0.0 || !disc.is_finite() {
        return None;
    }
    let sd = disc.sqrt();
    Some((facing - sd, facing + sd))
}

/// Finite-difference normal kept as the reference for the analytic one.
/// Two one-sided steps of half a datum cell. A flat datum is exactly radial.
pub(super) fn mapped_normal(
    rotation: Quat,
    u_world: Vec3,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
) -> Vec3 {
    let u = rotate(conjugate(rotation), u_world);
    let r_of = |dir: Vec3| rho + sample_datum(g, datum, dir) / distance;
    let cells = (g.max(2) - 1) as f32;
    let eps = std::f32::consts::PI / (4.0 * cells);
    let axis = if u.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let t1 = u.cross(axis).normalize_or_zero();
    let t2 = u.cross(t1);
    let r0 = r_of(u);
    let r1 = r_of((u + t1 * eps).normalize_or_zero());
    let r2 = r_of((u + t2 * eps).normalize_or_zero());
    let dr1 = (r1 - r0) / eps;
    let dr2 = (r2 - r0) / eps;
    let n_body = (u * r0 - t1 * dr1 - t2 * dr2).normalize_or_zero();
    rotate(rotation, n_body)
}

/// Outward normal of the bilinear patch at `u_world`.
///
/// `r(ξ, η)` is the interpolant already fetched for the hit. Differentiating
/// `p = r û` with `û ∥ n + tu tan(ξ π/4) + tv tan(η π/4)` gives the tangent
/// frame. On the +X face centre, `∂û/∂ξ × ∂û/∂η` points inward, so the outward
/// normal is `∂p/∂η × ∂p/∂ξ`. A flat patch is exactly radial.
pub(super) fn mapped_normal_analytic(
    rotation: Quat,
    u_world: Vec3,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
) -> Vec3 {
    let u = rotate(conjugate(rotation), u_world);
    let patch = sample_datum_d(g, datum, u);
    let r = rho + patch.value / distance;
    let dr_dxi = patch.dr_dxi / distance;
    let dr_deta = patch.dr_deta / distance;
    let face = dom_face(u) as usize;
    let (tu, n, tv) = far_map_basis(face);
    let den = u.dot(n);
    if den.abs() < 1e-8 || !(distance > 0.0) {
        return u_world;
    }
    let quarter = std::f32::consts::FRAC_PI_4;
    let xi = (4.0 / std::f32::consts::PI) * atan_approx(u.dot(tu) / den);
    let eta = (4.0 / std::f32::consts::PI) * atan_approx(u.dot(tv) / den);
    let tx = (xi * quarter).tan();
    let ty = (eta * quarter).tan();
    let dtx = quarter * (1.0 + tx * tx);
    let dty = quarter * (1.0 + ty * ty);
    let q = n + tu * tx + tv * ty;
    let s = q.length();
    if !(s > 1e-8) {
        return u_world;
    }
    let u_hat = q / s;
    let dq_dxi = tu * dtx;
    let dq_deta = tv * dty;
    let du_dxi = dq_dxi / s - u_hat * (u_hat.dot(dq_dxi) / s);
    let du_deta = dq_deta / s - u_hat * (u_hat.dot(dq_deta) / s);
    let dp_dxi = u_hat * dr_dxi + du_dxi * r;
    let dp_deta = u_hat * dr_deta + du_deta * r;
    let n_body = dp_deta.cross(dp_dxi).normalize_or_zero();
    if n_body.length_squared() < 1e-20 {
        return u_world;
    }
    rotate(rotation, n_body)
}

/// Coarse probes along the hi/lo bracket of [`ray_mapped`]. No shader runs
/// that march, so its caps below have no shader copy.
const MAPPED_COARSE_CAP: u32 = 16;
/// Secant / bisection steps after a sign change.
const MAPPED_REFINE_CAP: u32 = 24;
/// Entrance residual that already sits on the surface. A graze stays small
/// across a wide `t`, so the refinement stops on the bracket width instead.
/// Also the shallow-slope settle of [`fast_settled`], `FAR_MAP_FTOL` in the
/// shader.
pub(super) const MAPPED_F_TOL: f32 = 1e-8;
/// Relative width of the normalised-`t` bracket.
/// The `1e-7` floor keeps a root near the camera from asking for a sub-ulp
/// step. A `1e-4` floor left a hit a few metres out inside a bracket whose
/// relative width was several times 1e-6.
const MAPPED_T_TOL: f32 = 1e-7;
/// Interior probes of a coarse segment that could still hide a root. A
/// highland edge is negative only between two positive samples, and that
/// segment is not always the closest one.
const MAPPED_HUNT: u32 = 4;

/// Robust bracket march. Test-only, with no shader copy: the GPU march is
/// [`ray_mapped_fast`]. The tests compare the two inside the lo-sphere disc;
/// only [`ray_mapped_fast`] is held to the f64 reference.
///
/// Ray from the origin against a datum-mapped body. `dir` is the unit centre.
/// `rho` is `radius/distance`. Offsets are in the radius's unit.
///
/// A bounding-sphere miss and the horizon test return before any datum load.
/// Otherwise a sign-change search walks the bracket. The step is one datum
/// cell projected on the ray, and the count is capped. The closest approach
/// is always one of the probes. A segment whose samples are still within the
/// datum range of the surface is probed inside ([`MAPPED_HUNT`]): a highland
/// edge is negative only between those samples. A lo-sphere end with `f` on
/// zero is a graze of the minimum surface. The bracket is then
/// Anderson–Björck, and a step that fails to halve the width is bisected on
/// the next iteration. Stops when the width is [`MAPPED_T_TOL`] relative, or
/// when f32 can no longer split the interval. `horizon` skips
/// `dot(ray, -dir) > horizon`.
pub(crate) fn ray_mapped(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    horizon: f32,
    g: u32,
    datum: &[f32],
    min_off: f32,
    max_off: f32,
) -> Option<FarHit> {
    if !(rho > 0.0)
        || !(distance > 0.0)
        || !ray.is_finite()
        || !dir.is_finite()
        || !distance.is_finite()
    {
        return None;
    }
    if ray.dot(-dir) > horizon {
        return None;
    }
    let rho_lo = rho + min_off / distance;
    let rho_hi = rho + max_off / distance;
    let facing = ray.dot(dir);
    let (t_hi_in, t_hi_out) = sphere_roots(facing, rho_hi)?;
    if !(t_hi_out > 0.0) {
        return None;
    }
    let t_start = t_hi_in.max(0.0);
    let t_lo = sphere_roots(facing, rho_lo).and_then(|(t_lo, _)| (t_lo > 0.0).then_some(t_lo));
    let t_end = t_lo.map(|t| t.min(t_hi_out)).unwrap_or(t_hi_out);
    // `|p| - r` cancels in f32 when both are near 1, and a graze then moves
    // by an ulp over a tiny slope. `|p|^2 - r^2 = t²|ray|² - 2 t facing +
    // (|dir|² - 1) + (1 - r)(1 + r)`, divided by `|p| + r`, keeps that
    // difference. `1 - rho` is exact for a rho above one half.
    let ray2 = ray.length_squared();
    let dir2 = dir.length_squared();
    let f_at = |t: f32| -> f32 {
        let p = ray * t - dir;
        let rad = p.length();
        if rad < 1e-8 {
            return -rho;
        }
        let body = rotate(conjugate(rotation), p / rad);
        let h = sample_datum(g, datum, body);
        let r = rho + h / distance;
        let one_minus_r = (1.0 - rho) - h / distance;
        let diff_sq = t * t * ray2 - 2.0 * t * facing + (dir2 - 1.0) + one_minus_r * (1.0 + r);
        diff_sq / (rad + r)
    };
    let finish = |t: f32| -> Option<FarHit> {
        if !(t > 0.0) || !t.is_finite() {
            return None;
        }
        let p = ray * t - dir;
        let rad = p.length();
        if rad < 1e-8 {
            return None;
        }
        let u_world = p / rad;
        let body = rotate(conjugate(rotation), u_world);
        Some(FarHit {
            t,
            normal: mapped_normal_analytic(rotation, u_world, rho, distance, g, datum),
            face: dom_face(body),
        })
    };
    // Coincident lo/hi spheres, or a bracket thinner than the old entrance
    // test: the entrance is the surface.
    if !(t_end > t_start) || t_end - t_start < 1e-5 {
        let t_hit = if t_end > 0.0 { t_end } else { t_start };
        return finish(t_hit);
    }
    let f_start = f_at(t_start);
    if f_start < -1e-5 {
        return None;
    }
    // `t_start == 0` is the camera: the eye is inside the hi sphere, so the
    // entrance was clamped. `|f| <= 1e-5` is then a few hundred metres of
    // clearance on a planet-sized body, and a positive residual still has
    // the surface in front. Treating that clamp as the hit drops the ray.
    if f_start.abs() <= 1e-5 && t_start > 0.0 {
        return finish(t_start);
    }
    if t_start == 0.0 && f_start <= 0.0 {
        return None;
    }
    let cells = (g.max(2) - 1) as f32;
    let cell = std::f32::consts::FRAC_PI_2 / cells;
    let p0 = ray * t_start - dir;
    let rad0 = p0.length().max(rho);
    let sin_phi = if rad0 > 1e-8 {
        ray.cross(p0 / rad0).length()
    } else {
        0.0
    };
    let dt = cell * rad0 / sin_phi.max(0.05);
    let span = t_end - t_start;
    let n = ((span / dt.max(1e-8)).ceil() as u32).clamp(1, MAPPED_COARSE_CAP);
    let t_close = facing.clamp(t_start, t_end);
    let mut prev_t = t_start;
    let mut prev_f = f_start;
    let mut bracket: Option<(f32, f32, f32, f32)> = None;
    // A segment cannot cross zero if both samples are further out than the
    // whole datum range. Anything closer may hide a highland edge.
    let allowance = (max_off - min_off).abs() / distance;
    let hunt = |lo: f32, hi: f32, mut pf: f32, fhi: f32| -> Option<(f32, f32, f32, f32)> {
        if !(pf.min(fhi) <= allowance) {
            return None;
        }
        let mut pt = lo;
        for i in 1..=MAPPED_HUNT {
            let tm = lo + (hi - lo) * (i as f32 / (MAPPED_HUNT as f32 + 1.0));
            let fm = f_at(tm);
            if pf * fm <= 0.0 {
                return Some((pt, tm, pf, fm));
            }
            pt = tm;
            pf = fm;
        }
        (pf * fhi <= 0.0).then_some((pt, hi, pf, fhi))
    };
    for i in 1..=n {
        let tn = if i == n {
            t_end
        } else {
            t_start + span * (i as f32 / n as f32)
        };
        if t_close > prev_t + 1e-8 && t_close < tn - 1e-8 {
            let fc = f_at(t_close);
            if prev_f * fc <= 0.0 {
                bracket = Some((prev_t, t_close, prev_f, fc));
                break;
            }
            if let Some(found) = hunt(prev_t, t_close, prev_f, fc) {
                bracket = Some(found);
                break;
            }
            prev_t = t_close;
            prev_f = fc;
        }
        let ft = f_at(tn);
        if prev_f * ft <= 0.0 {
            bracket = Some((prev_t, tn, prev_f, ft));
            break;
        }
        if let Some(found) = hunt(prev_t, tn, prev_f, ft) {
            bracket = Some(found);
            break;
        }
        prev_t = tn;
        prev_f = ft;
    }
    let Some((mut a, mut b, mut fa, mut fb)) = bracket else {
        // No sign change. Entering the lo sphere with f on zero is a graze
        // of the minimum surface (a uniform grid of positive samples can
        // miss that). The hi-sphere exit is not: f is small there on a
        // highland far side that the ray never hit.
        return if t_lo.is_some() && prev_f <= 1e-5 {
            finish(prev_t)
        } else {
            None
        };
    };
    if a > b {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    // Chord weights stay separate from the true signs. `retained` is which
    // endpoint survived the previous secant (1 = a, 2 = b). Anderson–Björck
    // scales that weight only on the second retention in a row. A step that
    // does not halve the width forces a bisection next, which is what a
    // one-sided regula falsi fails to do on a long near-ground chord.
    let mut wa = fa;
    let mut wb = fb;
    let mut retained = 0i32;
    let mut force_bisect = false;
    for _ in 0..MAPPED_REFINE_CAP {
        let width = b - a;
        let scale = a.abs().min(b.abs()).max(1e-7);
        if !(width > MAPPED_T_TOL * scale) {
            break;
        }
        let den = wb - wa;
        let secant = if den.abs() > 1e-20 {
            b - wb * width / den
        } else {
            0.5 * (a + b)
        };
        let secant_ok = secant > a && secant < b;
        let bisect = force_bisect || !secant_ok;
        let c = if bisect { 0.5 * (a + b) } else { secant };
        force_bisect = false;
        if !(c > a && c < b) {
            break;
        }
        let fc = f_at(c);
        // `|f|` alone is not a root at a graze. Require the bracket to
        // already be inside the relative tolerance as well.
        if fc.abs() <= MAPPED_F_TOL && width <= MAPPED_T_TOL * scale.max(c.abs()) {
            return finish(c);
        }
        let old_width = width;
        if fa * fc <= 0.0 {
            let discarded = fb;
            b = c;
            fb = fc;
            wb = fc;
            if bisect {
                wa = fa;
                retained = 0;
            } else if retained == 1 {
                let mut m = if discarded.abs() > 1e-20 {
                    1.0 - fc / discarded
                } else {
                    0.5
                };
                if !(m > 0.0) || !m.is_finite() {
                    m = 0.5;
                }
                wa *= m;
            } else {
                wa = fa;
                retained = 1;
            }
        } else {
            let discarded = fa;
            a = c;
            fa = fc;
            wa = fc;
            if bisect {
                wb = fb;
                retained = 0;
            } else if retained == 2 {
                let mut m = if discarded.abs() > 1e-20 {
                    1.0 - fc / discarded
                } else {
                    0.5
                };
                if !(m > 0.0) || !m.is_finite() {
                    m = 0.5;
                }
                wb *= m;
            } else {
                wb = fb;
                retained = 2;
            }
        }
        if b - a > 0.5 * old_width {
            force_bisect = true;
        }
    }
    let mid = 0.5 * (a + b);
    let t_hit = if mid > a && mid < b {
        mid
    } else if fa.abs() <= fb.abs() {
        a
    } else {
        b
    };
    finish(t_hit)
}

/// Datum samples the march may spend. The analytic normal is one more fetch
/// at the hit, replacing the two finite-difference evaluations.
pub(super) const MAPPED_EVAL_CAP: u32 = 5;
struct MappedMarch<'a> {
    ray: Vec3,
    center: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &'a [f32],
    evals: u32,
}

impl MappedMarch<'_> {
    /// `(f, r, df/dt)` at `t`. `f` is the stable `|p| - r(dir(p))`. One datum
    /// evaluation. The slope is `u·ray`: inside the lo-sphere disc that term
    /// keeps the hit within 1e-6 relative of [`ray_mapped`], and the chart
    /// derivative is a live range the fast shader cannot afford.
    fn sample(&mut self, t: f32) -> (f32, f32, f32) {
        self.evals += 1;
        let p = self.ray * t - self.center;
        let rad = p.length();
        if rad < 1e-8 {
            return (-self.rho, 0.0, 0.0);
        }
        let u_world = p / rad;
        let body = rotate(conjugate(self.rotation), u_world);
        let value = sample_datum(self.g, self.datum, body);
        let r = self.rho + value / self.distance;
        // `|p| - r` cancels in f32 when both are near 1. The same
        // `|p|^2 - r^2` form as `ray_mapped` keeps a hit a few metres
        // out, which is the whole lo-disc interior at low altitude.
        let facing = self.ray.dot(self.center);
        let one_minus_r = (1.0 - self.rho) - value / self.distance;
        let diff_sq = t * t * self.ray.length_squared() - 2.0 * t * facing
            + (self.center.length_squared() - 1.0)
            + one_minus_r * (1.0 + r);
        let f = diff_sq / (rad + r);
        let slope = u_world.dot(self.ray);
        (f, r, slope)
    }

    fn open(&self) -> bool {
        self.evals < MAPPED_EVAL_CAP
    }
}

/// Near positive root of a sphere of radius `radius` centred on the unit dir.
fn positive_root(facing: f32, radius: f32) -> Option<f32> {
    let (t_near, t_far) = sphere_roots(facing, radius)?;
    if t_near > 0.0 {
        Some(t_near)
    } else if t_far > 0.0 {
        Some(t_far)
    } else {
        None
    }
}

/// Another Newton step would move `t` by less than 1e-8 relative, inside
/// the 1e-6 agreement band of the robust march. A shallow
/// slope falls back to [`MAPPED_F_TOL`]: a graze stays small across a wide
/// interval, and the step is not a meaningful distance.
fn fast_settled(t: f32, f: f32, slope: f32) -> bool {
    if slope.abs() > 1e-4 {
        (f / slope).abs() <= 1e-8 * t.abs().max(1e-12)
    } else {
        f.abs() <= MAPPED_F_TOL
    }
}

/// Illinois regula falsi inside an existing bracket. The retained endpoint's
/// weight is halved so a stuck side does not burn the remaining samples.
/// Stops at the eval cap, a tiny step, or [`fast_settled`]. The weights
/// used for the chord are not the values used to pick the final endpoint.
fn regula(march: &mut MappedMarch<'_>, mut a: f32, mut b: f32, mut fa: f32, mut fb: f32) -> f32 {
    let mut wa = fa;
    let mut wb = fb;
    while march.open() {
        let den = wb - wa;
        // Relative, not absolute: a 1e-7 bracket is a large fraction of a
        // hit that sits 10 m in front of a planet-sized body.
        let span = a.abs().max(b.abs()).max(1e-6);
        if den.abs() < 1e-20 || (b - a).abs() <= 1e-8 * span {
            break;
        }
        let tn = b - wb * (b - a) / den;
        if !(tn > a.min(b) && tn < a.max(b)) {
            break;
        }
        let (ft, _, st) = march.sample(tn);
        if fast_settled(tn, ft, st) {
            return tn;
        }
        // Shallow roots (grazes) make the chord lag. A Newton step from a
        // small residual uses the radial slope and lands on the root.
        if st.abs() > 1e-4 && ft.abs() < 2e-3 {
            let t2 = tn - ft / st;
            let lo = a.min(b);
            let hi = a.max(b);
            if t2 > lo && t2 < hi && (t2 - tn).abs() < 5e-3 {
                if !march.open() {
                    // Sphere curvature only. The datum's second derivative is
                    // small next to (1 - (u·ray)^2) / |p| on a shallow graze,
                    // and there is no sample left to measure it.
                    let p = march.ray * tn - march.center;
                    let rad = p.length();
                    let srad = (p / rad).dot(march.ray);
                    let curve = (1.0 - srad * srad) / rad;
                    let denom = st - ft * curve / (2.0 * st);
                    let t_h = if denom.abs() > 1e-4 && rad > 1e-8 {
                        tn - ft / denom
                    } else {
                        t2
                    };
                    return if t_h.is_finite() && (t_h - tn).abs() < 5e-3 {
                        t_h
                    } else {
                        t2
                    };
                }
                let (f2, _, s2) = march.sample(t2);
                if fast_settled(t2, f2, s2) {
                    return t2;
                }
                if s2.abs() > 1e-4 && f2.abs() < 1e-3 {
                    // Halley: fold in f'' from the two slopes so a shallow
                    // curve does not leave a residual after Newton. Accept
                    // it only inside the 1e-6 band; a wider step goes back
                    // into the bracket.
                    let d2 = if (t2 - tn).abs() > 1e-6 {
                        (s2 - st) / (t2 - tn)
                    } else {
                        0.0
                    };
                    let denom = s2 - f2 * d2 / (2.0 * s2);
                    let t3 = if denom.abs() > 1e-4 {
                        t2 - f2 / denom
                    } else {
                        t2 - f2 / s2
                    };
                    if t3.is_finite()
                        && t3 > lo
                        && t3 < hi
                        && (f2 / s2).abs() <= 1e-8 * t2.abs().max(1e-12)
                    {
                        return t3;
                    }
                }
                // Not there yet. Shrink the bracket onto the sampled pair.
                if ft * f2 <= 0.0 {
                    if tn < t2 {
                        a = tn;
                        b = t2;
                        fa = ft;
                        fb = f2;
                    } else {
                        a = t2;
                        b = tn;
                        fa = f2;
                        fb = ft;
                    }
                    wa = fa;
                    wb = fb;
                    continue;
                }
                if f2.abs() < ft.abs() {
                    if fa * f2 <= 0.0 {
                        b = t2;
                        fb = f2;
                        wb = f2;
                    } else {
                        a = t2;
                        fa = f2;
                        wa = f2;
                    }
                    continue;
                }
            }
        }
        if fa * ft <= 0.0 {
            b = tn;
            fb = ft;
            wb = ft;
            wa *= 0.5;
        } else {
            a = tn;
            fa = ft;
            wa = ft;
            wb *= 0.5;
        }
    }
    // The budget is spent. The next chord is a better estimate than either
    // endpoint and costs no further datum load.
    if !march.open() {
        let den = fb - fa;
        if den.abs() > 1e-20 {
            let tn = b - fb * (b - a) / den;
            if tn > a.min(b) && tn < a.max(b) {
                return tn;
            }
        }
    }
    if fa.abs() <= fb.abs() { a } else { b }
}

/// Cheap fixed-point march. Host mirror of `far_ray_mapped` in
/// `shaders/far_body.slang`: the two stay sample-for-sample identical, and
/// the shader is the copy that runs. The Illinois steps are this march's own
/// graze handler and stay inside [`MAPPED_EVAL_CAP`]; [`ray_mapped`] is not
/// called. `dir` is the unit centre.
/// `rho` is `radius/distance`. Offsets are in the radius's unit.
///
/// Starts at the lo-sphere entrance and sets `t` to the near intersection of
/// the ray with the sphere of radius `r(dir(p))`. Non-grazing rays settle in
/// two or three steps; a step smaller than 1e-8 relative returns early. A
/// graze that misses the lo sphere, or a fixed point that does not settle,
/// falls back to regula falsi.
/// The march takes at most [`MAPPED_EVAL_CAP`] datum samples. The normal is
/// the analytic patch derivative at the hit. A hi-sphere entrance clamped
/// to the camera is not a root: `|f| <= 1e-5` there is clearance, and a
/// positive residual keeps marching. `horizon` skips
/// `dot(ray, -dir) > horizon`.
pub(crate) fn ray_mapped_fast(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    horizon: f32,
    g: u32,
    datum: &[f32],
    min_off: f32,
    max_off: f32,
) -> Option<FarHit> {
    if !(rho > 0.0)
        || !(distance > 0.0)
        || !ray.is_finite()
        || !dir.is_finite()
        || !distance.is_finite()
    {
        return None;
    }
    if ray.dot(-dir) > horizon {
        return None;
    }
    let rho_lo = rho + min_off / distance;
    let rho_hi = rho + max_off / distance;
    let facing = ray.dot(dir);
    let (t_hi_in, t_hi_out) = sphere_roots(facing, rho_hi)?;
    if !(t_hi_out > 0.0) {
        return None;
    }
    let t_start = t_hi_in.max(0.0);
    let finish = |t: f32| -> Option<FarHit> {
        if !(t > 0.0) || !t.is_finite() {
            return None;
        }
        let p = ray * t - dir;
        let rad = p.length();
        if rad < 1e-8 {
            return None;
        }
        let u_world = p / rad;
        let body = rotate(conjugate(rotation), u_world);
        Some(FarHit {
            t,
            normal: mapped_normal_analytic(rotation, u_world, rho, distance, g, datum),
            face: dom_face(body),
        })
    };
    let mut march = MappedMarch {
        ray,
        center: dir,
        rho,
        distance,
        rotation,
        g,
        datum,
        evals: 0,
    };
    let lo_enter = sphere_roots(facing, rho_lo).and_then(|(t_lo, _)| (t_lo > 0.0).then_some(t_lo));
    if let Some(t_lo) = lo_enter {
        let b = t_lo.max(t_start);
        if b - t_start < 1e-5 {
            return finish(if t_lo > 0.0 { t_lo } else { t_start });
        }
        // Fixed-point pulls from the lo entrance: t <- ray ∩ sphere(r(dir)).
        // A sign change is a bracket. Two pulls plus the lo sample are the
        // non-grazing case; Illinois regula spends whatever is left. The hi
        // entrance is sampled only when those pulls never bracket, or when
        // the eye is already inside the hi sphere (t_start == 0), which the
        // old march treats as a miss when f < 0.
        let (mut f, mut r, mut slope) = march.sample(t_lo);
        if fast_settled(t_lo, f, slope) {
            return finish(t_lo);
        }
        let mut t = t_lo;
        let mut bracket: Option<(f32, f32, f32, f32)> = None;
        for _ in 0..2 {
            if !march.open() {
                break;
            }
            // The far root of a sphere that contains the camera is the back
            // of the planet. Step with the radial slope instead; that stays
            // on the near surface. A 1e-8 absolute stop froze hits near the
            // eye a ulp-scale step short of the root.
            let tn = match sphere_roots(facing, r) {
                Some((near, _)) if near > t_start && near < t_hi_out => near,
                _ if slope.abs() > 1e-4 && f.abs() < 0.05 => {
                    let ts = t - f / slope;
                    if ts > t_start && ts < t_hi_out {
                        ts
                    } else {
                        break;
                    }
                }
                _ => break,
            };
            if (tn - t).abs() <= 1e-12 {
                break;
            }
            if let Some((a, b, _, _)) = bracket
                && (tn <= a || tn >= b)
            {
                break;
            }
            let (ft, rt, st) = march.sample(tn);
            if f * ft <= 0.0 {
                bracket = Some(if t < tn {
                    (t, tn, f, ft)
                } else {
                    (tn, t, ft, f)
                });
            }
            if fast_settled(tn, ft, st) {
                return finish(tn);
            }
            t = tn;
            f = ft;
            r = rt;
            slope = st;
        }
        // The camera sample catches an eye inside the body. A bracket whose
        // near end is already outside does not need it, and the sample is
        // better spent tightening the root.
        let need_entrance = match bracket {
            None => true,
            Some((ba, bb, bfa, bfb)) => {
                let near_f = if ba <= bb { bfa } else { bfb };
                t_start == 0.0 && near_f < 0.0
            }
        };
        if need_entrance {
            if !march.open() {
                return finish(t);
            }
            let (fa, _, _) = march.sample(t_start);
            // Same camera clamp as [`ray_mapped`]: `|f| <= 1e-5` at `t == 0`
            // is clearance, not a root, and `finish(0)` would drop the ray.
            if t_start == 0.0 && fa.abs() <= 1e-5 {
                if fa <= 0.0 {
                    return None;
                }
            } else if fa.abs() <= 1e-5 {
                return finish(t_start);
            } else if fa < 0.0 {
                return None;
            }
            if bracket.is_none() {
                if fa * f <= 0.0 {
                    bracket = Some(if t_start < t {
                        (t_start, t, fa, f)
                    } else {
                        (t, t_start, f, fa)
                    });
                } else {
                    return finish(t);
                }
            }
        }
        let Some((a, b, fa, fb)) = bracket else {
            return finish(t);
        };
        return finish(regula(&mut march, a, b, fa, fb));
    }
    // No lo-sphere hit: the ray only clips the datum above the lo sphere.
    // Fixed-point pulls walk in from the hi entrance. A crossing becomes a
    // bracket. A shallow graze stalls with a small residual and a small
    // slope; one Newton step from that residual, then a second from the
    // sample, finishes it without another load. A short chord scan is the
    // last resort and only when a pull was not possible.
    let (mut f, mut r, mut slope) = march.sample(t_start);
    if f < -1e-5 {
        return None;
    }
    if f.abs() <= 1e-5 && t_start > 0.0 {
        return finish(t_start);
    }
    if t_start == 0.0 && f <= 0.0 {
        return None;
    }
    let f_enter = f;
    let mut t = t_start;
    let mut bracket: Option<(f32, f32, f32, f32)> = None;
    // One evaluation stays spare for the Newton check below.
    while march.evals + 1 < MAPPED_EVAL_CAP {
        let Some(tn) = positive_root(facing, r).filter(|tn| *tn > t_start && *tn <= t_hi_out)
        else {
            break;
        };
        if (tn - t).abs() <= 1e-8 {
            break;
        }
        let (ft, rt, st) = march.sample(tn);
        if f * ft <= 0.0 {
            bracket = Some(if t < tn {
                (t, tn, f, ft)
            } else {
                (tn, t, ft, f)
            });
            break;
        }
        if fast_settled(tn, ft, st) {
            return finish(tn);
        }
        t = tn;
        f = ft;
        r = rt;
        slope = st;
    }
    if bracket.is_none() && slope.abs() > 1e-4 && f.abs() < 0.05 {
        let ts = t - f / slope;
        if ts > t_start && ts < t_hi_out && (ts - t).abs() > 1e-8 && (ts - t).abs() < 0.05 {
            if march.open() {
                let (fs, _, ss) = march.sample(ts);
                if fs.abs() <= 1e-5 {
                    if ss.abs() > 1e-4 {
                        let t2 = ts - fs / ss;
                        if t2 > t_start && t2 < t_hi_out && (t2 - ts).abs() < 2e-3 {
                            return finish(t2);
                        }
                    }
                    return finish(ts);
                }
                if f * fs <= 0.0 {
                    bracket = Some(if t < ts {
                        (t, ts, f, fs)
                    } else {
                        (ts, t, fs, f)
                    });
                } else if fs.abs() < 1e-3 && ss.abs() > 1e-4 {
                    let t2 = ts - fs / ss;
                    if t2 > t_start && t2 < t_hi_out && (t2 - ts).abs() < 2e-3 {
                        return finish(t2);
                    }
                }
            } else if f.abs() < 1e-3 {
                return finish(ts);
            }
        }
    }
    if let Some((a, b, fa, fb)) = bracket {
        return finish(regula(&mut march, a, b, fa, fb));
    }
    let mut prev_t = t_start;
    let mut prev_f = f_enter;
    let left = MAPPED_EVAL_CAP.saturating_sub(march.evals);
    for i in 1..=left {
        if !march.open() {
            break;
        }
        let tn = t_start + (t_hi_out - t_start) * (i as f32 / (left as f32));
        let (ft, _, st) = march.sample(tn);
        if fast_settled(tn, ft, st) {
            return finish(tn);
        }
        if prev_f * ft <= 0.0 {
            return finish(regula(&mut march, prev_t, tn, prev_f, ft));
        }
        prev_t = tn;
        prev_f = ft;
    }
    None
}

/// Surface radius of the mapped body along `ray` at normalised `t`.
pub(super) fn mapped_radius_at(
    ray: Vec3,
    dir: Vec3,
    t: f32,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
) -> f32 {
    let p = ray * t - dir;
    let rad = p.length();
    if rad < 1e-8 {
        return 0.0;
    }
    rho + sample_datum(g, datum, rotate(conjugate(rotation), p / rad)) / distance
}

/// Samples on the cap-sphere chord. Fixed at 8, the shader loop bound.
pub(super) const LIMB_SAMPLES: usize = 8;

/// World metres from the closest approach. The ends are the cap chord.
/// Zero is the approach itself, so a march miss that dips under the datum
/// there still counts. The rest pack toward that approach: a uniform step
/// across the cap chord is coarser than the air. Lockstep with
/// `far_limb_delta` in `shaders/far_body.slang`.
pub(super) fn limb_delta_m(i: usize) -> f32 {
    match i {
        0 => -1.0e30,
        1 => -1_200_000.0,
        2 => -400_000.0,
        3 => 0.0,
        4 => 400_000.0,
        5 => 1_200_000.0,
        6 => 2_400_000.0,
        _ => 1.0e30,
    }
}

/// `true` when the eye's altitude above the bilinear datum in the eye's
/// direction is below `air`.
pub(super) fn eye_inside_mapped_air(
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    air: f32,
) -> bool {
    let radial = -dir;
    let rad = radial.length();
    if !(rad > 1e-8) || !(distance > 0.0) {
        return true;
    }
    let h = sample_datum(g, datum, rotate(conjugate(rotation), radial / rad));
    let altitude = distance - rho * distance - h;
    altitude < air
}

/// Forward chord through the sphere of normalised radius `rho_cap`.
/// `t` is in units of `distance`. `None` when the ray misses it.
pub(super) fn cap_chord(facing: f32, rho_cap: f32) -> Option<(f32, f32)> {
    if !(rho_cap > 0.0) || !rho_cap.is_finite() || !facing.is_finite() {
        return None;
    }
    let disc = facing * facing - (1.0 - rho_cap * rho_cap);
    if !(disc >= 0.0) || !disc.is_finite() {
        return None;
    }
    let sd = disc.sqrt();
    let t_far = facing + sd;
    if !(t_far > 0.0) {
        return None;
    }
    let t0 = (facing - sd).max(0.0);
    (t_far > t0).then_some((t0, t_far))
}

/// `(rad, r_air)` at normalised `t`. A sample on the centre is inside.
pub(super) fn limb_air_at(
    ray: Vec3,
    dir: Vec3,
    t: f32,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    air: f32,
) -> (f32, f32) {
    let p = ray * t - dir;
    let rad = p.length();
    if rad < 1e-8 {
        return (0.0, 1.0);
    }
    let h = sample_datum(g, datum, rotate(conjugate(rotation), p / rad));
    let r_air = rho + (h + air) / distance;
    (rad, r_air.max(0.0))
}

/// Fraction of `u` in `[0, 1]` where `qa u² + qb u + qc <= 0`.
fn limb_inside_fraction(qa: f32, qb: f32, qc: f32) -> f32 {
    if qa.abs() <= 1e-20 {
        if qb.abs() <= 1e-20 {
            return if qc <= 0.0 { 1.0 } else { 0.0 };
        }
        let u = -qc / qb;
        return if qb > 0.0 {
            u.clamp(0.0, 1.0)
        } else {
            (1.0 - u).clamp(0.0, 1.0)
        };
    }
    let disc = qb * qb - 4.0 * qa * qc;
    if !(disc >= 0.0) || !disc.is_finite() {
        return if qa > 0.0 { 0.0 } else { 1.0 };
    }
    let sd = disc.sqrt();
    let q = if qb >= 0.0 {
        -0.5 * (qb + sd)
    } else {
        -0.5 * (qb - sd)
    };
    let (u0, u1) = if q.abs() <= 1e-20 {
        let inv = 0.5 / qa;
        ((-qb - sd) * inv, (-qb + sd) * inv)
    } else {
        (q / qa, qc / q)
    };
    let lo = u0.min(u1);
    let hi = u0.max(u1);
    if qa > 0.0 {
        let a = lo.clamp(0.0, 1.0);
        let b = hi.clamp(0.0, 1.0);
        (b - a).max(0.0)
    } else {
        lo.clamp(0.0, 1.0) + (1.0 - hi).clamp(0.0, 1.0)
    }
}

/// Normalised length of `[t0, t1]` on which `|p|` is at or below the air
/// radius linearly interpolated from `r0` to `r1`. The quadratic matches the
/// measured `|p|` at both ends and the analytic curvature between them, so a
/// segment that ends inside the air is not dropped when the analytic radius
/// misses that end. Below the datum counts: that
/// radius already includes the interior, so a graze the march missed still
/// contributes.
pub(super) fn limb_segment_t(t0: f32, t1: f32, rad0: f32, rad1: f32, r0: f32, r1: f32) -> f32 {
    let dt = t1 - t0;
    if !(dt > 1e-20) {
        return 0.0;
    }
    let r0 = r0.max(0.0);
    let r1 = r1.max(0.0);
    let dr = r1 - r0;
    let qa = dt * dt - dr * dr;
    let qc = (rad0 - r0) * (rad0 + r0);
    let g1 = (rad1 - r1) * (rad1 + r1);
    let qb = g1 - qa - qc;
    let frac = limb_inside_fraction(qa, qb, qc);
    if frac.is_finite() { frac * dt } else { 0.0 }
}

/// World length of the air `ray` crosses. `0` when the eye is inside the
/// local air, or the cap-sphere chord never enters `R + datum + air`.
/// Host mirror of `far_mapped_air_chord`. The horizon table still widens
/// its shell by a pixel; this chord does not.
pub(super) fn ray_mapped_limb_chord(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
) -> f32 {
    if !(distance > 0.0) || !(air > 0.0) || !ray.is_finite() || !dir.is_finite() {
        return 0.0;
    }
    let facing = ray.dot(dir);
    if !(facing > 0.0) {
        return 0.0;
    }
    if eye_inside_mapped_air(dir, rho, distance, rotation, g, datum, air) {
        return 0.0;
    }
    let rho_cap = rho + (max_off + air) / distance;
    let Some((t0, t1)) = cap_chord(facing, rho_cap) else {
        return 0.0;
    };
    let ca = facing.clamp(t0, t1);
    let mut sum_t = 0.0f32;
    let mut prev_t = 0.0f32;
    let mut prev_rad = 0.0f32;
    let mut prev_r = 0.0f32;
    for i in 0..LIMB_SAMPLES {
        let t = (ca + limb_delta_m(i) / distance).clamp(t0, t1);
        let (rad, r_air) = limb_air_at(ray, dir, t, rho, distance, rotation, g, datum, air);
        if i > 0 {
            sum_t += limb_segment_t(prev_t, t, prev_rad, rad, prev_r, r_air);
        }
        prev_t = t;
        prev_rad = rad;
        prev_r = r_air;
    }
    if !(sum_t > 0.0) || !sum_t.is_finite() {
        return 0.0;
    }
    sum_t * distance
}

/// `true` when [`ray_mapped_limb_chord`] is positive. The limb is the air the
/// ray crosses, not a pixel-widened sphere.
pub(crate) fn ray_mapped_limb(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
) -> bool {
    ray_mapped_limb_chord(ray, dir, rho, distance, rotation, g, datum, max_off, air) > 0.0
}

// Fixtures shared by the tests in `tests.rs` and in `src/vk/far_bodies/tests/`.

/// Home-planet radius, in blocks. Datum offsets sit in `[-278_000, 1_040_000]`.
pub(crate) const HOME_RADIUS: f64 = 31_017_520.0;

/// Low-order height, then an affine map onto the home offset range.
/// Smooth on the scale of a face: a sum of a few harmonics, not noise.
pub(crate) fn home_datum(g: u32) -> Vec<f32> {
    let gg = g as usize;
    let mut raw = vec![0.0f64; 6 * gg * gg];
    for face in 0..6usize {
        let (tu, n, tv) = far_map_basis(face);
        let (tu, n, tv) = (tu.as_dvec3(), n.as_dvec3(), tv.as_dvec3());
        for j in 0..g {
            for i in 0..g {
                let edge = f64::from(g - 1);
                let xi = 2.0 * f64::from(i) / edge - 1.0;
                let eta = 2.0 * f64::from(j) / edge - 1.0;
                let quarter = std::f64::consts::FRAC_PI_4;
                let d = (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                let h = 0.55 * d.y
                    + 0.28 * (d.x * d.x - d.z * d.z)
                    + 0.22 * (2.0 * d.x * d.z)
                    + 0.14 * d.y * (1.0 - 3.0 * d.y * d.y);
                raw[face * gg * gg + j as usize * gg + i as usize] = h;
            }
        }
    }
    let lo = raw.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = raw.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_t = -278_000.0;
    let max_t = 1_040_000.0;
    let scale = (max_t - min_t) / (hi - lo);
    raw.iter()
        .map(|v| ((v - lo) * scale + min_t) as f32)
        .collect()
}

/// Angle from the centre direction. `0` looks straight down.
pub(crate) fn ray_from_nadir(dir: Vec3, azimuth: f32, angle: f32) -> Vec3 {
    let down = dir.normalize();
    let up = -down;
    let reference = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let east = reference.cross(up).normalize();
    let north = up.cross(east);
    let horiz = east * azimuth.sin() + north * azimuth.cos();
    (down * angle.cos() + horiz * angle.sin()).normalize()
}

/// Lowland everywhere, a broad highland in front of +X. The zenith sample
/// stays 0 so an eye on +Y is above the ground, not inside the bump.
pub(crate) fn horizon_highland(
    g: u32,
    beta0: f32,
    beta1: f32,
    az_half: f32,
    height: f32,
) -> Vec<f32> {
    let gg = g as usize;
    let mut datum = vec![0.0f32; 6 * gg * gg];
    for face in 0..6usize {
        let (tu, n, tv) = far_map_basis(face);
        for j in 0..g {
            for i in 0..g {
                let edge = (g - 1) as f32;
                let xi = 2.0 * i as f32 / edge - 1.0;
                let eta = 2.0 * j as f32 / edge - 1.0;
                let quarter = std::f32::consts::FRAC_PI_4;
                let d = (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                let beta = d.dot(Vec3::Y).clamp(-1.0, 1.0).acos();
                let az = d.z.atan2(d.x);
                if (beta0..=beta1).contains(&beta) && az.abs() <= az_half {
                    datum[face * gg * gg + j as usize * gg + i as usize] = height;
                }
            }
        }
    }
    datum
}

/// Direction of chart sample `(face, i, j)`. Same reconstruction `home_datum`
/// uses when it writes that sample.
pub(crate) fn chart_sample_dir(g: u32, face: usize, i: u32, j: u32) -> Vec3 {
    let (tu, n, tv) = far_map_basis(face);
    let edge = (g - 1) as f32;
    let xi = 2.0 * i as f32 / edge - 1.0;
    let eta = 2.0 * j as f32 / edge - 1.0;
    let quarter = std::f32::consts::FRAC_PI_4;
    (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize()
}

/// Lowest chart sample. The eye stands on this lowland.
pub(crate) fn lowland_up(g: u32, datum: &[f32]) -> Vec3 {
    let gg = g as usize;
    let mut best_h = f32::MAX;
    let mut best = Vec3::Y;
    for face in 0..6usize {
        for j in 0..g {
            for i in 0..g {
                let h = datum[face * gg * gg + j as usize * gg + i as usize];
                if h < best_h {
                    best_h = h;
                    best = chart_sample_dir(g, face, i, j);
                }
            }
        }
    }
    best
}

/// Unit direction of the lowest sample, and the bilinear height there.
pub(crate) fn lowland_foot(g: u32, datum: &[f32]) -> (Vec3, f32) {
    let up = lowland_up(g, datum);
    (up, sample_datum(g, datum, up))
}

/// Largest angle from nadir that `covered` accepts, by bisection over
/// `[1e-3, π]`. `None` when `1e-3` is not covered.
pub(super) fn limb_boundary_angle(covered: &impl Fn(f64) -> bool) -> Option<f64> {
    if !covered(1.0e-3) {
        return None;
    }
    let mut lo = 1.0e-3f64;
    let mut hi = std::f64::consts::PI;
    if covered(hi) {
        return Some(hi);
    }
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if covered(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some(0.5 * (lo + hi))
}

/// Angle from nadir of the outermost ray [`ray_mapped_limb`] still covers.
/// `None` when a ray just off nadir misses the air.
pub(crate) fn mapped_limb_top_angle(
    dir: Vec3,
    azimuth: f32,
    rho: f32,
    distance: f32,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
) -> Option<f64> {
    limb_boundary_angle(&|theta| {
        let ray = ray_from_nadir(dir, azimuth, theta as f32);
        ray_mapped_limb(
            ray,
            dir,
            rho,
            distance,
            Quat::IDENTITY,
            g,
            datum,
            max_off,
            air,
        )
    })
}
