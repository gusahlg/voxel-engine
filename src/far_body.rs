//! Scale-normalised far-body impostors. The ray tests here are the CPU mirror of
//! `shaders/far_body.slang`. Outside shapes sit at distance 1 along `dir`, with
//! radius `radius/distance`. An inner sphere is that same space with the viewer
//! inside (`distance < radius`); the hit is the far root.

use glam::{Quat, Vec3};

use crate::color::LinearRgb;

/// Bodies kept from one [`crate::Frame3D::set_far_bodies`] call. Extra entries are dropped.
pub const MAX_FAR_BODIES: usize = 32;

/// Sphere, a cube whose `radius` is the half-size, the inside of a sphere, or a
/// rounded cube.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum FarShape {
    Cube,
    #[default]
    Sphere,
    /// The viewer is inside the sphere. Every view ray hits the far wall.
    InnerSphere,
    /// `|x|^p + |y|^p + |z|^p = radius^p` in body space. `exponent` is p ≥ 2
    /// (2 is the sphere; large p approaches the cube). Face centres sit at
    /// `radius`; a unit direction `d` meets the surface at `radius / ‖d‖_p`.
    Rounded { exponent: f32 },
}

/// One body drawn in the sky pass.
///
/// `dir` points from the camera to the body centre. `distance` and `radius` share
/// a unit. A cube, sphere or rounded body is drawn only when `distance > radius`
/// (the viewer is outside the face sphere; a rounded body may still bulge past
/// the camera and the ray test drops those hits). [`FarShape::InnerSphere`] is
/// the opposite: `distance < radius`, and the far wall is the sky. `albedo` is
/// the six cube faces (+X, −X, +Y, −Y, +Z, −Z); a sphere uses `[0]` for land and
/// `[1]` for the second tone. A rounded body shades the face of its body-space
/// normal and uses the sphere's rim. A black `atmosphere` draws no rim.
/// `rotation` takes body space into world space (identity for an axis-aligned
/// cube or rounded body).
#[derive(Clone, Copy, Debug)]
pub struct FarBody {
    pub dir: Vec3,
    pub distance: f32,
    pub radius: f32,
    pub shape: FarShape,
    pub rotation: Quat,
    pub albedo: [LinearRgb; 6],
    pub atmosphere: LinearRgb,
    pub seed: u32,
}

impl Default for FarBody {
    fn default() -> Self {
        Self {
            dir: Vec3::Z,
            distance: 1.0,
            radius: 0.0,
            shape: FarShape::Sphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.0, 0.0, 0.0]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 0,
        }
    }
}

/// A hit in normalised space. `t` is along the unit view ray. `normal` is world
/// space. `face` is the cube face (0 = +X … 5 = −Z), the dominant axis of a
/// rounded body's normal, and 0 for a sphere.
/// Test-only: the sky shader is the copy that runs.
#[cfg(test)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct FarHit {
    pub t: f32,
    pub normal: Vec3,
    pub face: u32,
}

/// `radius/distance` in f32. Exact for distances the catalog actually uses.
#[cfg(test)]
pub(crate) fn normalised_radius(distance: f32, radius: f32) -> f32 {
    radius / distance
}

/// glam's `Quat * Vec3`, written out so the shader can use the same arithmetic.
/// Does not normalise: a packed unit quaternion is already unit.
#[cfg(test)]
fn rotate(q: Quat, v: Vec3) -> Vec3 {
    let b = Vec3::new(q.x, q.y, q.z);
    let w = q.w;
    v * (w * w - b.dot(b)) + b * (v.dot(b) * 2.0) + b.cross(v) * (w * 2.0)
}

#[cfg(test)]
fn conjugate(q: Quat) -> Quat {
    Quat::from_xyzw(-q.x, -q.y, -q.z, q.w)
}

#[cfg(test)]
const SLAB_EPS: f32 = 1e-8;
#[cfg(test)]
const T_FAR: f32 = 1e30;

/// One axis of an AABB slab. `near_neg` is the −face of this axis.
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
const ROUNDED_STEPS: u32 = 6;

/// Dominant axis of a body-space normal. Ties break toward X, then Y, then Z.
#[cfg(test)]
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
#[cfg(test)]
fn lp_grad(q: Vec3, p: f32) -> (f32, Vec3) {
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
#[cfg(test)]
fn rounded_at(ray_o: Vec3, ray_d: Vec3, rotation: Quat, t_hit: f32, p: f32, rho: f32) -> Option<FarHit> {
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
#[cfg(test)]
pub(crate) fn ray_rounded(ray: Vec3, dir: Vec3, rho: f32, rotation: Quat, exponent: f32) -> Option<FarHit> {
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
    let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
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
        if s_c <= rho * 1.001 && t_c > 0.0 {
            if let Some(hit) = rounded_at(o, d, rotation, t_c, p, rho) {
                return Some(hit);
            }
        }
        // Grazing the enclosing sphere where that sphere meets the body.
        if t_near > 0.0 && sd <= rho_b * 0.02 {
            let (s_b, _) = lp_grad(o + d * t_near, p);
            if s_b >= rho && s_b <= rho * 1.01 {
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
        if s_l >= rho && s_l <= rho * (1.0 + 2.0e-5) {
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

fn finite_rgb(c: LinearRgb) -> bool {
    c.0[0].is_finite() && c.0[1].is_finite() && c.0[2].is_finite()
}

fn keep(body: &FarBody) -> Option<FarBody> {
    if let FarShape::Rounded { exponent } = body.shape {
        if !(exponent >= 2.0) || !exponent.is_finite() {
            return None;
        }
    }
    let inside = matches!(body.shape, FarShape::InnerSphere);
    // Outside shapes reject a viewer inside the solid. The inner sphere is the
    // one shape that requires it, and it keeps distance and radius (not the
    // outside `radius/distance < 1` form) so that case stays explicit.
    let span_ok = if inside {
        body.distance > 0.0
            && body.distance < body.radius
            && (body.radius / body.distance).is_finite()
    } else {
        body.distance > body.radius
    };
    if !body.distance.is_finite() || !body.radius.is_finite() || !(body.radius > 0.0) || !span_ok {
        return None;
    }
    if !body.dir.is_finite() || body.dir.length_squared() == 0.0 {
        return None;
    }
    let q = body.rotation;
    if !q.x.is_finite()
        || !q.y.is_finite()
        || !q.z.is_finite()
        || !q.w.is_finite()
        || q.length_squared() == 0.0
    {
        return None;
    }
    if !finite_rgb(body.atmosphere) || body.albedo.iter().any(|c| !finite_rgb(*c)) {
        return None;
    }
    Some(FarBody {
        dir: body.dir.normalize(),
        rotation: q.normalize(),
        ..*body
    })
}

/// Keep the first [`MAX_FAR_BODIES`] entries that can be drawn, then sort far to near.
/// An inner sphere draws first: it is the sky, and the bodies inside it composite
/// over the wall. Equal distances then draw the smaller radius first so a shell
/// wins over a core it contains.
pub(crate) fn store(bodies: &[FarBody], out: &mut [FarBody; MAX_FAR_BODIES]) -> u32 {
    let mut n = 0usize;
    for body in bodies.iter().take(MAX_FAR_BODIES) {
        if let Some(kept) = keep(body) {
            out[n] = kept;
            n += 1;
        }
    }
    out[..n].sort_unstable_by(|a, b| {
        // `false` sorts before `true`, so the wall is the first draw.
        let back = |body: &FarBody| !matches!(body.shape, FarShape::InnerSphere);
        back(a).cmp(&back(b)).then(
            b.distance
                .total_cmp(&a.distance)
                .then(a.radius.total_cmp(&b.radius))
                .then(a.seed.cmp(&b.seed)),
        )
    });
    n as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere_at(dir: Vec3, distance: f32, radius: f32, seed: u32) -> FarBody {
        FarBody {
            dir,
            distance,
            radius,
            shape: FarShape::Sphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([1.0, 1.0, 1.0]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed,
        }
    }

    #[test]
    fn rotate_matches_glam() {
        let q = Quat::from_xyzw(0.2, -0.4, 0.1, 0.8).normalize();
        let v = Vec3::new(0.3, -1.2, 0.7);
        let got = rotate(q, v);
        let expect = q * v;
        assert!(
            (got - expect).length() < 1e-5,
            "hand rotate {got} vs glam {expect}"
        );
        let id = rotate(Quat::IDENTITY, v);
        assert_eq!(id.x.to_bits(), v.x.to_bits());
        assert_eq!(id.y.to_bits(), v.y.to_bits());
        assert_eq!(id.z.to_bits(), v.z.to_bits());
    }

    #[test]
    fn centre_ray_hits_sphere_and_cube() {
        let hit = ray_sphere(Vec3::Z, Vec3::Z, 0.25).unwrap();
        assert_eq!(hit.t.to_bits(), 0.75f32.to_bits());
        assert_eq!(hit.normal.z.to_bits(), (-1.0f32).to_bits());
        assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());

        let cube = ray_cube(Vec3::Z, Vec3::Z, 0.25, Quat::IDENTITY).unwrap();
        assert_eq!(cube.face, 5);
        assert_eq!(cube.t.to_bits(), 0.75f32.to_bits());
        assert_eq!(cube.normal.z.to_bits(), (-1.0f32).to_bits());
        let local = rotate(conjugate(Quat::IDENTITY), Vec3::Z * cube.t - Vec3::Z);
        assert_eq!(local.z.to_bits(), (-0.25f32).to_bits());
        assert_eq!(local.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(local.y.to_bits(), 0.0f32.to_bits());
    }

    #[test]
    fn yaw_half_turn_hits_the_opposite_cube_face() {
        // 180° about Y: (x, y, z) → (−x, y, −z). Exact, not `from_rotation_y(PI)`.
        let q = Quat::from_xyzw(0.0, 1.0, 0.0, 0.0);
        let hit = ray_cube(Vec3::Z, Vec3::Z, 0.25, q).unwrap();
        assert_eq!(hit.face, 4);
        assert_eq!(hit.t.to_bits(), 0.75f32.to_bits());
        assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(hit.normal.y.to_bits(), 0.0f32.to_bits());
        assert_eq!(hit.normal.z.to_bits(), (-1.0f32).to_bits());
        let local = rotate(conjugate(q), Vec3::Z * hit.t - Vec3::Z);
        assert_eq!(local.z.to_bits(), 0.25f32.to_bits());
    }

    #[test]
    fn misses_and_inside_draw_nothing() {
        assert!(ray_sphere(Vec3::X, Vec3::Z, 0.25).is_none());
        assert!(ray_cube(Vec3::X, Vec3::Z, 0.25, Quat::IDENTITY).is_none());
        assert!(ray_sphere(Vec3::Z, Vec3::Z, 1.0).is_none());
        assert!(ray_sphere(Vec3::Z, Vec3::Z, 1.5).is_none());
        assert!(ray_cube(Vec3::Z, Vec3::Z, 1.0, Quat::IDENTITY).is_none());
        assert!(ray_sphere(Vec3::Z, Vec3::Z, 0.0).is_none());
        assert!(ray_cube(Vec3::Z, Vec3::Z, 0.0, Quat::IDENTITY).is_none());
    }

    #[test]
    fn grazing_sphere_matches_the_disc() {
        let inside = Vec3::new(0.4, 0.0, (1.0 - 0.16f32).sqrt()).normalize();
        let hit = ray_sphere(inside, Vec3::Z, 0.5).unwrap();
        let err = (inside * hit.t - Vec3::Z).length() - 0.5;
        assert!(err.abs() < 1e-4, "{err}");
        let outside = Vec3::new(0.6, 0.0, (1.0 - 0.36f32).sqrt()).normalize();
        assert!(ray_sphere(outside, Vec3::Z, 0.5).is_none());
    }

    #[test]
    fn billion_block_distance_is_exact_in_normalised_space() {
        let rho = normalised_radius(1.0e9, 2.5e8);
        assert_eq!(rho.to_bits(), 0.25f32.to_bits());
        assert_eq!(
            normalised_radius(1.0e9, 5.0e8).to_bits(),
            0.5f32.to_bits()
        );
        let dir = Vec3::new(-0.2, 0.3, 0.8).normalize();
        // A small offset from the centre ray: still inside a rho of 0.25, for both shapes.
        let ray = (dir + Vec3::new(0.05, -0.02, 0.01)).normalize();
        let sphere_far = ray_sphere(ray, dir, rho).expect("sphere hit");
        let sphere_near = ray_sphere(ray, dir, 0.25).unwrap();
        assert_eq!(sphere_far.t.to_bits(), sphere_near.t.to_bits());
        assert_eq!(sphere_far.normal.x.to_bits(), sphere_near.normal.x.to_bits());
        assert_eq!(sphere_far.normal.y.to_bits(), sphere_near.normal.y.to_bits());
        assert_eq!(sphere_far.normal.z.to_bits(), sphere_near.normal.z.to_bits());
        let cube_far = ray_cube(ray, dir, rho, Quat::IDENTITY).unwrap();
        let cube_near = ray_cube(ray, dir, 0.25, Quat::IDENTITY).unwrap();
        assert_eq!(cube_far.t.to_bits(), cube_near.t.to_bits());
        assert_eq!(cube_far.face, cube_near.face);
        assert_eq!(cube_far.normal.x.to_bits(), cube_near.normal.x.to_bits());
        assert_eq!(cube_far.normal.y.to_bits(), cube_near.normal.y.to_bits());
        assert_eq!(cube_far.normal.z.to_bits(), cube_near.normal.z.to_bits());
        // The intersector never sees the world distance: a centre ray is 1 − rho.
        let centre = ray_sphere(Vec3::Z, Vec3::Z, rho).unwrap();
        assert_eq!(centre.t.to_bits(), (1.0 - rho).to_bits());
    }

    #[test]
    fn store_caps_sorts_and_drops_invalid() {
        let bodies: Vec<FarBody> = (0..40)
            .map(|i| sphere_at(Vec3::new(0.0, 0.0, 2.0), 100.0 + i as f32, 1.0, i))
            .collect();
        let mut out = [FarBody::default(); MAX_FAR_BODIES];
        let n = store(&bodies, &mut out);
        assert_eq!(n, 32);
        assert_eq!(out[0].distance, 131.0);
        assert_eq!(out[31].distance, 100.0);
        assert!(out.iter().take(n as usize).all(|b| (0..32).contains(&b.seed)));
        assert!((out[0].dir - Vec3::Z).length() < 1e-6);

        let mut nan = vec![
            sphere_at(Vec3::Z, 10.0, 1.0, 1),
            sphere_at(Vec3::Z, 20.0, 1.0, 2),
            sphere_at(Vec3::Z, 30.0, 1.0, 3),
        ];
        nan[1].distance = f32::NAN;
        assert_eq!(store(&nan, &mut out), 2);
        assert_eq!(out[0].distance, 30.0);
        assert_eq!(out[1].distance, 10.0);

        let inside = sphere_at(Vec3::Z, 1.0, 1.0, 4);
        assert_eq!(store(&[inside], &mut out), 0);
        let buried = sphere_at(Vec3::Z, 1.0, 2.0, 5);
        assert_eq!(store(&[buried], &mut out), 0);
        let mut wall = sphere_at(Vec3::Z, 2.0, 5.0, 7);
        wall.shape = FarShape::InnerSphere;
        assert_eq!(store(&[wall], &mut out), 1);
        assert_eq!(out[0].shape, FarShape::InnerSphere);
        assert!(out[0].distance < out[0].radius);
        // Outside the inner sphere is not this shape.
        wall.distance = 8.0;
        assert_eq!(store(&[wall], &mut out), 0);
        let mut zero_dir = sphere_at(Vec3::ZERO, 10.0, 1.0, 6);
        zero_dir.dir = Vec3::ZERO;
        assert_eq!(store(&[zero_dir], &mut out), 0);
    }

    #[test]
    fn inner_sphere_hits_every_ray_and_faces_the_centre() {
        let dir = Vec3::Y;
        let distance = 2.0;
        let radius = 5.0;
        let rho: f32 = radius / distance;
        assert_eq!(rho.to_bits(), 2.5f32.to_bits());
        let rays = [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(1.0, 0.3, -0.4).normalize(),
            Vec3::new(-0.2, -1.0, 0.5).normalize(),
        ];
        for ray in rays {
            let hit = ray_inner_sphere(ray, dir, distance, radius).expect("inside hits");
            let point = ray * hit.t;
            let err = (point - dir).length() - rho;
            assert!(err.abs() < 1e-4, "{ray:?} off the wall by {err}");
            let toward = (dir - point).normalize();
            assert!(
                hit.normal.dot(toward) > 0.999,
                "{ray:?} normal {:?} should point at the centre",
                hit.normal
            );
            assert!((hit.normal.length() - 1.0).abs() < 1e-4);
        }
        // Looking at the centre: t = b + sqrt(disc) = 1 + rho, normal back along dir.
        let b = 1.0f32;
        let disc = b * b - (1.0 - rho * rho);
        let t = b + disc.sqrt();
        let hit = ray_inner_sphere(Vec3::Y, dir, distance, radius).unwrap();
        assert_eq!(hit.t.to_bits(), t.to_bits());
        assert_eq!(hit.t.to_bits(), 3.5f32.to_bits());
        assert_eq!(hit.normal.y.to_bits(), (-1.0f32).to_bits());
        assert_eq!(hit.normal.x.to_bits(), 0.0f32.to_bits());
        assert_eq!(hit.face, 0);
        // The near root is behind the camera. This shape takes the far one.
        let near = b - disc.sqrt();
        assert!(near < 0.0);
        assert!(ray_inner_sphere(Vec3::Y, dir, 5.0, 2.0).is_none());
        assert!(ray_inner_sphere(Vec3::Y, dir, 2.0, 2.0).is_none());
        assert!(ray_inner_sphere(Vec3::Y, dir, 0.0, 2.0).is_none());
    }

    #[test]
    fn inner_sphere_draws_behind_the_bodies_inside_it() {
        let mut wall = sphere_at(Vec3::Z, 10.0, 40.0, 7);
        wall.shape = FarShape::InnerSphere;
        let core = sphere_at(Vec3::Z, 10.0, 1.0, 8);
        let far = sphere_at(Vec3::X, 100.0, 1.0, 9);
        let mut out = [FarBody::default(); MAX_FAR_BODIES];
        assert_eq!(store(&[core, far, wall], &mut out), 3);
        assert_eq!(out[0].shape, FarShape::InnerSphere);
        assert_eq!(out[1].distance, 100.0);
        assert_eq!(out[2].seed, 8);
    }

    fn axis_face(u: Vec3) -> Option<u32> {
        let a = u.abs();
        if a.x > 0.9 && a.y < 1e-4 && a.z < 1e-4 {
            Some(if u.x > 0.0 { 0 } else { 1 })
        } else if a.y > 0.9 && a.x < 1e-4 && a.z < 1e-4 {
            Some(if u.y > 0.0 { 2 } else { 3 })
        } else if a.z > 0.9 && a.x < 1e-4 && a.y < 1e-4 {
            Some(if u.z > 0.0 { 4 } else { 5 })
        } else {
            None
        }
    }

    /// Face centres, the three edge axes and the space diagonals.
    fn rounded_dirs() -> [Vec3; 16] {
        [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(-1.0, -1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
        ]
    }

    fn outside(q: Vec3, rho: f32, p: f32) -> bool {
        let (s, _) = lp_grad(q, p);
        s + rho * 1.0e-4 >= rho
    }

    #[test]
    fn rounded_hits_analytic_face_edge_and_corner() {
        let rho = 0.3f32;
        let tilt = Quat::from_xyzw(0.2, -0.4, 0.1, 0.8).normalize();
        for p in [2.0f32, 2.17, 10.0] {
            let corner = Vec3::ONE.normalize();
            let radial = rho / lp_grad(corner, p).0;
            let bound = rho * 3.0f32.powf(0.5 - 1.0 / p);
            assert!(
                (radial - bound).abs() < 1e-4,
                "p {p}: corner radius {radial} bound {bound}"
            );
            for rotation in [Quat::IDENTITY, tilt] {
                for dir in rounded_dirs() {
                    let u = dir.normalize();
                    let outward = rotate(rotation, u);
                    // Look straight down this body direction so the sample is the near hit.
                    let centre = -outward;
                    let reach = rho / lp_grad(u, p).0;
                    let point = centre + outward * reach;
                    let ray = -outward;
                    let hit = ray_rounded(ray, centre, rho, rotation, p)
                        .unwrap_or_else(|| panic!("p {p} dir {u:?} missed"));
                    let got = ray * hit.t;
                    let err = (got - point).length();
                    assert!(err < 2e-4, "p {p} dir {u:?} off by {err} (got {got}, want {point})");
                    assert!(
                        hit.normal.dot(outward) > 0.999,
                        "p {p} normal {:?} not along {outward:?}",
                        hit.normal
                    );
                    assert!((hit.normal.length() - 1.0).abs() < 1e-4);
                    let q = rotate(conjugate(rotation), got - centre);
                    assert!(outside(q, rho, p), "p {p} analytic hit inside");
                    if let Some(face) = axis_face(u) {
                        assert_eq!(hit.face, face, "p {p} face of {u:?}");
                    }
                }
            }
            // p = 2 is the sphere the closed form already tests.
            if p == 2.0 {
                let centre = Vec3::Z;
                let ray = Vec3::new(0.12, -0.05, 1.0).normalize();
                let sphere = ray_sphere(ray, centre, 0.4).unwrap();
                let rounded = ray_rounded(ray, centre, 0.4, Quat::IDENTITY, p).unwrap();
                assert!((sphere.t - rounded.t).abs() < 2e-4, "{} vs {}", sphere.t, rounded.t);
                assert!(sphere.normal.dot(rounded.normal) > 0.999);
            }
        }
    }

    #[test]
    fn rounded_grazing_never_hits_inside() {
        let perps = [
            Vec3::X,
            Vec3::Y,
            Vec3::new(1.0, 1.0, 0.0).normalize(),
            Vec3::new(-1.0, 2.0, 0.0).normalize(),
            Vec3::new(2.0, -0.5, 0.0).normalize(),
        ];
        for p in [2.0f32, 2.17, 10.0] {
            for rho in [0.35f32, 0.97] {
                let bound = rho * 3.0f32.powf(0.5 - 1.0 / p);
                let centre = Vec3::Z;
                for perp in perps {
                    for k in 0..18 {
                        let s = bound * (k as f32) / 16.0;
                        if s >= 1.0 {
                            continue;
                        }
                        let z = (1.0 - s * s).sqrt();
                        let ray = Vec3::new(perp.x * s, perp.y * s, z);
                        if let Some(hit) = ray_rounded(ray, centre, rho, Quat::IDENTITY, p) {
                            let q = rotate(conjugate(Quat::IDENTITY), ray * hit.t - centre);
                            assert!(
                                outside(q, rho, p),
                                "p {p} rho {rho} s {s} hit inside at {:?}",
                                ray * hit.t
                            );
                        }
                    }
                }
                // Past the enclosing sphere there is nothing to hit.
                if bound * 1.02 < 1.0 {
                    let s = bound * 1.02;
                    let ray = Vec3::new(s, 0.0, (1.0 - s * s).sqrt());
                    assert!(
                        ray_rounded(ray, centre, rho, Quat::IDENTITY, p).is_none(),
                        "p {p} rho {rho} ray outside the bound hit"
                    );
                }
            }
        }
    }

    #[test]
    fn rounded_exponent_is_kept_only_when_real() {
        let mut body = sphere_at(Vec3::Z, 4.0, 1.0, 3);
        body.shape = FarShape::Rounded { exponent: 2.17 };
        let mut out = [FarBody::default(); MAX_FAR_BODIES];
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 1);
        assert_eq!(out[0].shape, FarShape::Rounded { exponent: 2.17 });
        body.shape = FarShape::Rounded { exponent: 1.5 };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
        body.shape = FarShape::Rounded { exponent: f32::NAN };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
        body.shape = FarShape::Rounded { exponent: f32::INFINITY };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    }
}
