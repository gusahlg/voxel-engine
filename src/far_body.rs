//! Scale-normalised far-body impostors. The ray tests here are the CPU mirror of
//! `shaders/far_body.slang`. Outside shapes sit at distance 1 along `dir`, with
//! radius `radius/distance`. An inner sphere is that same space with the viewer
//! inside (`distance < radius`); the hit is the far root.

use glam::{Quat, Vec3};

use crate::color::LinearRgb;

/// Bodies kept from one [`crate::Frame3D::set_far_bodies`] call. Extra entries are dropped.
pub const MAX_FAR_BODIES: usize = 32;

/// Sphere, a cube whose `radius` is the half-size, or the inside of a sphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FarShape {
    Cube,
    #[default]
    Sphere,
    /// The viewer is inside the sphere. Every view ray hits the far wall.
    InnerSphere,
}

/// One body drawn in the sky pass.
///
/// `dir` points from the camera to the body centre. `distance` and `radius` share
/// a unit. A cube or sphere is drawn only when `distance > radius` (the viewer is
/// outside). [`FarShape::InnerSphere`] is the opposite: `distance < radius`, and
/// the far wall is the sky. `albedo` is the six cube faces (+X, −X, +Y, −Y, +Z, −Z);
/// a sphere uses `[0]` for land and `[1]` for the second tone. A black
/// `atmosphere` draws no rim. `rotation` takes body space into world space
/// (identity for an axis-aligned cube).
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
/// space. `face` is the cube face (0 = +X … 5 = −Z) and 0 for a sphere.
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

fn finite_rgb(c: LinearRgb) -> bool {
    c.0[0].is_finite() && c.0[1].is_finite() && c.0[2].is_finite()
}

fn keep(body: &FarBody) -> Option<FarBody> {
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
}
