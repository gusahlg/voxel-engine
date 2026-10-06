//! Scale-normalised far-body impostors. The ray tests here are the CPU mirror of
//! `shaders/far_body.slang`. Outside shapes sit at distance 1 along `dir`, with
//! radius `radius/distance`. An inner sphere is that same space with the viewer
//! inside (`distance < radius`); the hit is the far root.

use glam::{Quat, Vec3};

use crate::color::LinearRgb;

/// Bodies kept from one [`crate::Frame3D::set_far_bodies`] call. Extra entries are dropped.
pub const MAX_FAR_BODIES: usize = 32;

/// Datum-mapped planets the sky pass can hold at once.
pub const MAX_FAR_MAPS: usize = 8;

/// Slot of one datum and its albedo cube. Valid ids are `0..`[`MAX_FAR_MAPS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FarMapId(pub u8);

/// Sphere, a cube whose `radius` is the half-size, the inside of a sphere, a
/// rounded cube, or a datum-mapped planet.
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
    Rounded {
        exponent: f32,
    },
    /// Radius along body-space direction `d` is [`FarBody::radius`] plus the
    /// datum offset of `map`. Albedo comes from that map's cube faces.
    ///
    /// `horizon` is the sine of the highest elevation of any surface point
    /// seen from the eye, above the plane whose normal is `-dir`. The shader
    /// skips rays with `dot(ray, -dir) > horizon`. `1.0` disables the cull.
    /// `air` is the air-shell thickness in the same unit as `radius` (`0`
    /// draws no limb). A miss ray's limb is that shell on the surface radius
    /// at the ray's closest approach to the centre, not a shell on the
    /// datum's maximum.
    Mapped {
        map: FarMapId,
        horizon: f32,
        air: f32,
    },
}

/// Host description of one datum-mapped planet.
///
/// Installed with [`crate::Engine::set_far_map`]. The datum is equiangular on
/// the charts of [`far_map_basis`]: a body-space unit direction `d` on face
/// `f` has `xi = (4/π) atan(dot(d, tu) / dot(d, n))` and eta likewise with `tv`.
#[derive(Clone, Copy)]
pub struct FarMapDesc<'a> {
    /// Samples per face edge, edges included. `g` is in `2..=65`.
    pub datum_res: u32,
    /// `6 * g * g` offsets, in the same unit as [`FarBody::radius`]. Face order
    /// is +X, −X, +Y, −Y, +Z, −Z. Index `f * g * g + j * g + i`, with `i` along
    /// xi and `j` along eta.
    pub datum: &'a [f32],
    /// Cube-face edge in texels. A power of two in `1..=2048`, or `0` to keep
    /// the flat [`FarBody::albedo`] colours and upload no cube.
    pub albedo_size: u32,
}

/// Why [`crate::Engine::set_far_map`] or [`crate::Engine::set_far_map_face`] rejected a call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FarMapError {
    /// The id is outside `0..`[`MAX_FAR_MAPS`], or the slot has no map yet.
    BadId,
    /// Datum resolution, datum length, face index, or albedo size is not valid.
    BadSize,
    /// The albedo cube could not be allocated.
    ///
    /// [`Engine::set_far_map`](crate::Engine::set_far_map) does not return this.
    /// The render thread logs that failure once for the map id. With no
    /// complete cube it keeps the datum with flat per-face colours, as if
    /// `albedo_size` were 0. With a complete cube already on screen, that
    /// cube stays until a later `set_far_map`. The variant stays so existing
    /// matches still compile.
    OutOfMemory,
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
/// normal and uses the sphere's rim. A [`FarShape::Mapped`] body uses `radius`
/// as the reference radius; `albedo` is the flat fallback for a cube face whose
/// upload has not landed (or whose map has `albedo_size` 0), chosen by the
/// dominant body-space axis of the hit direction. `atmosphere` tints that
/// body's air limb, and `seed` is unused. A black `atmosphere` draws no rim.
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
#[cfg(test)]
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

/// Face basis `(tu, n, tv)` of the datum charts. Face order is +X, −X, +Y, −Y,
/// +Z, −Z. An index outside `0..6` returns the +Y basis.
pub fn far_map_basis(face: usize) -> (Vec3, Vec3, Vec3) {
    match face {
        0 => (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        1 => (Vec3::Y, Vec3::NEG_X, Vec3::Z),
        2 => (Vec3::X, Vec3::Y, Vec3::Z),
        3 => (Vec3::X, Vec3::NEG_Y, Vec3::NEG_Z),
        4 => (Vec3::X, Vec3::Z, Vec3::NEG_Y),
        5 => (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        _ => (Vec3::X, Vec3::Y, Vec3::Z),
    }
}

/// Body-space unit direction of the centre of texel `(x, y)` on albedo cube
/// face `face` of edge `size`. The orientation is Vulkan's cube-map face
/// table, so a caller never has to know it. `size` of 0 is treated as 1.
pub fn far_cube_texel_dir(face: usize, size: u32, x: u32, y: u32) -> Vec3 {
    let size = size.max(1) as f32;
    let s = (x as f32 + 0.5) / size;
    let t = (y as f32 + 0.5) / size;
    let uc = 2.0 * s - 1.0;
    let vc = 2.0 * t - 1.0;
    let d = match face {
        0 => Vec3::new(1.0, -vc, -uc),
        1 => Vec3::new(-1.0, -vc, uc),
        2 => Vec3::new(uc, 1.0, vc),
        3 => Vec3::new(uc, -1.0, -vc),
        4 => Vec3::new(uc, -vc, 1.0),
        _ => Vec3::new(-uc, -vc, -1.0),
    };
    d.normalize_or_zero()
}

/// `Ok` when `id` and `desc` can be installed. Does not touch the GPU.
pub(crate) fn validate_far_map(id: FarMapId, desc: &FarMapDesc<'_>) -> Result<(), FarMapError> {
    if id.0 as usize >= MAX_FAR_MAPS {
        return Err(FarMapError::BadId);
    }
    let g = desc.datum_res;
    if !(2..=65).contains(&g) {
        return Err(FarMapError::BadSize);
    }
    let n = 6usize * g as usize * g as usize;
    if desc.datum.len() != n {
        return Err(FarMapError::BadSize);
    }
    if desc.albedo_size != 0 && (desc.albedo_size > 2048 || !desc.albedo_size.is_power_of_two()) {
        return Err(FarMapError::BadSize);
    }
    Ok(())
}

/// `slot` is the installed albedo edge, or `None` when the id has no map.
/// `bytes` is the length of the RGBA8 upload.
pub(crate) fn validate_far_map_face(
    id: FarMapId,
    face: usize,
    bytes: usize,
    slot: Option<u32>,
) -> Result<(), FarMapError> {
    if id.0 as usize >= MAX_FAR_MAPS {
        return Err(FarMapError::BadId);
    }
    let Some(albedo_size) = slot else {
        return Err(FarMapError::BadId);
    };
    if face >= 6 || albedo_size == 0 {
        return Err(FarMapError::BadSize);
    }
    let expect = albedo_size as usize * albedo_size as usize * 4;
    if bytes != expect {
        return Err(FarMapError::BadSize);
    }
    Ok(())
}

fn keep(body: &FarBody) -> Option<FarBody> {
    match body.shape {
        FarShape::Rounded { exponent } => {
            if !(exponent >= 2.0) || !exponent.is_finite() {
                return None;
            }
        }
        FarShape::Mapped { map, horizon, air } => {
            if map.0 as usize >= MAX_FAR_MAPS
                || !horizon.is_finite()
                || !air.is_finite()
                || air < 0.0
            {
                return None;
            }
        }
        FarShape::Cube | FarShape::Sphere | FarShape::InnerSphere => {}
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

/// Odd degree-13 polynomial for `atan` on `[-1, 1]`.
///
/// A face of the equiangular chart only feeds this range (`dot(d, tu) /
/// dot(d, n)`), and the argument is clamped before the Horner step. Stepwise
/// f32 Horner (`p = p * u + c`, each multiply and add rounded) stays within
/// 6.61e-7 rad of `atan` on two million uniform samples, peaking near −0.972.
/// A datum cell is `π / (4 * 32)` ≈ 2.5e-2 rad, so the error is far below a cell.
#[cfg(test)]
fn atan_approx(x: f32) -> f32 {
    let z = x.clamp(-1.0, 1.0);
    let u = z * z;
    // c6 .. c0. Bits 3c04aa78 bd1aa180 3dad9679 be0aa033 3e4bb99e beaaa3a5 3f7ffffb.
    let mut p = f32::from_bits(0x3c04_aa78);
    p = p * u + f32::from_bits(0xbd1a_a180);
    p = p * u + f32::from_bits(0x3dad_9679);
    p = p * u + f32::from_bits(0xbe0a_a033);
    p = p * u + f32::from_bits(0x3e4b_b99e);
    p = p * u + f32::from_bits(0xbeaa_a3a5);
    p = p * u + f32::from_bits(0x3f7f_fffb);
    z * p
}

/// Bilinear equiangular sample and the chart derivatives of that interpolant.
/// `g < 2` or a short slice is a flat zero. Four loads, matching the shader.
#[cfg(test)]
struct DatumSample {
    value: f32,
    /// `d(value) / dξ` inside the cell. Zero on the flat fallback.
    dr_dxi: f32,
    /// `d(value) / dη` inside the cell.
    dr_deta: f32,
}

#[cfg(test)]
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
#[cfg(test)]
fn sample_datum(g: u32, datum: &[f32], d: Vec3) -> f32 {
    sample_datum_d(g, datum, d).value
}

/// Max of the four bilinear corners of the datum cell that contains `d`.
/// Matches `far_datum_cell_max`. The limb uses this so an edge cell keeps the
/// highland radius instead of the interpolated slope.
#[cfg(test)]
fn sample_datum_cell_max(g: u32, datum: &[f32], d: Vec3) -> f32 {
    if g < 2 {
        return 0.0;
    }
    let gg = g as usize;
    let need = 6 * gg * gg;
    if datum.len() < need {
        return 0.0;
    }
    let face = dom_face(d) as usize;
    let (tu, n, tv) = far_map_basis(face);
    let den = d.dot(n);
    if den.abs() < 1e-8 {
        return 0.0;
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
    let at = |j: u32, i: u32| datum[face * gg * gg + j as usize * gg + i as usize];
    at(j0, i0).max(at(j0, i1)).max(at(j1, i0)).max(at(j1, i1))
}

/// Near and far roots of a sphere of radius `rho` centred on the unit `dir`.
#[cfg(test)]
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
#[cfg(test)]
fn mapped_normal(
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
#[cfg(test)]
fn mapped_normal_analytic(
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

/// Previous regula-falsi march, kept so the fixed-point solver can be checked
/// against it on the same datum samples. A lo-sphere bracket takes 6 steps,
/// otherwise 12 even steps across the hi chord and 5 regula-falsi steps.
#[cfg(test)]
fn ray_mapped_regula(
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
    let f_at = |t: f32| -> f32 {
        let p = ray * t - dir;
        let rad = p.length();
        if rad < 1e-8 {
            return -rho;
        }
        let body = rotate(conjugate(rotation), p / rad);
        rad - (rho + sample_datum(g, datum, body) / distance)
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
            normal: mapped_normal(rotation, u_world, rho, distance, g, datum),
            face: dom_face(body),
        })
    };
    let lo_enter = sphere_roots(facing, rho_lo).and_then(|(t_lo, _)| (t_lo > 0.0).then_some(t_lo));
    if let Some(t_lo) = lo_enter {
        let b = t_lo.max(t_start);
        if b - t_start < 1e-5 {
            return finish(if t_lo > 0.0 { t_lo } else { t_start });
        }
        let mut a = t_start;
        let mut b = b;
        let mut fa = f_at(a);
        let mut fb = f_at(b);
        if fa.abs() <= 1e-5 {
            return finish(a);
        }
        if fa < 0.0 {
            return None;
        }
        for _ in 0..6 {
            let den = fb - fa;
            if den.abs() < 1e-20 || (b - a).abs() < 1e-7 {
                break;
            }
            let t = b - fb * (b - a) / den;
            if !(t > a.min(b) && t < a.max(b)) {
                break;
            }
            let ft = f_at(t);
            if fa * ft <= 0.0 {
                b = t;
                fb = ft;
            } else {
                a = t;
                fa = ft;
            }
        }
        return finish(if fa.abs() <= fb.abs() { a } else { b });
    }
    let mut prev_t = t_start;
    let mut prev_f = f_at(prev_t);
    if prev_f < -1e-5 {
        return None;
    }
    if prev_f.abs() <= 1e-5 {
        return finish(prev_t);
    }
    let mut bracket = None;
    for i in 1..=12 {
        let t = t_start + (t_hi_out - t_start) * (i as f32 / 12.0);
        let ft = f_at(t);
        if prev_f * ft <= 0.0 {
            bracket = Some((prev_t, t, prev_f, ft));
            break;
        }
        prev_t = t;
        prev_f = ft;
    }
    let Some((mut a, mut b, mut fa, mut fb)) = bracket else {
        return None;
    };
    for _ in 0..5 {
        let den = fb - fa;
        if den.abs() < 1e-20 || (b - a).abs() < 1e-7 {
            break;
        }
        let t = b - fb * (b - a) / den;
        if !(t > a.min(b) && t < a.max(b)) {
            break;
        }
        let ft = f_at(t);
        if fa * ft <= 0.0 {
            b = t;
            fb = ft;
        } else {
            a = t;
            fa = ft;
        }
    }
    finish(if fa.abs() <= fb.abs() { a } else { b })
}

/// Coarse probes along the hi/lo bracket. Matches `FAR_MAP_COARSE`.
#[cfg(test)]
const MAPPED_COARSE_CAP: u32 = 16;
/// Secant / bisection steps after a sign change. Matches `FAR_MAP_REFINE`.
#[cfg(test)]
const MAPPED_REFINE_CAP: u32 = 24;
/// Entrance residual that already sits on the surface. A graze stays small
/// across a wide `t`, so the refinement stops on the bracket width instead.
#[cfg(test)]
const MAPPED_F_TOL: f32 = 1e-8;
/// Relative width of the normalised-`t` bracket. Matches `FAR_MAP_TTOL`.
/// The `1e-7` floor keeps a root near the camera from asking for a sub-ulp
/// step. A `1e-4` floor left a hit a few metres out inside a bracket whose
/// relative width was several times 1e-6.
#[cfg(test)]
const MAPPED_T_TOL: f32 = 1e-7;
/// Interior probes of a coarse segment that could still hide a root.
/// Matches `FAR_MAP_HUNT`. A highland edge is negative only between two
/// positive samples, and that segment is not always the closest one.
#[cfg(test)]
const MAPPED_HUNT: u32 = 4;

/// Robust bracket march. Test-only reference beside the f64 intersection in
/// this module's tests. The GPU march is [`ray_mapped_fast`].
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
#[cfg(test)]
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
#[cfg(test)]
const MAPPED_EVAL_CAP: u32 = 5;
#[cfg(test)]
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

#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
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
#[cfg(test)]
fn mapped_radius_at(
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

/// Limb radius along `ray` at `t`: cell-max datum, so a highland edge keeps
/// the high corner instead of the interpolated slope.
#[cfg(test)]
fn mapped_limb_radius_at(
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
    rho + sample_datum_cell_max(g, datum, rotate(conjugate(rotation), p / rad)) / distance
}

/// Local air shell for a ray that missed the datum. Host mirror of
/// `far_mapped_limb`.
///
/// `r_surf` is the max cell-corner datum at closest approach and at the
/// hi-sphere entry and exit. An edge cell keeps the highland corner so the
/// interpolated slope cannot drop the limb into a notch. The outer radius
/// is `r_surf + max(air/distance, px)`, clamped to the air sphere
/// `rho + (max_off + air) / distance` that the horizon cone bounds. A
/// pixel-wide shell used to stick out past that sphere and get cut off in a
/// straight line. `s` is the impact parameter.
#[cfg(test)]
struct MappedShell {
    r_surf: f32,
    outer: f32,
    s: f32,
}

#[cfg(test)]
fn ray_mapped_shell(
    ray: Vec3,
    dir: Vec3,
    rho: f32,
    distance: f32,
    rotation: Quat,
    g: u32,
    datum: &[f32],
    max_off: f32,
    air: f32,
    px: f32,
) -> Option<MappedShell> {
    if !(distance > 0.0) || !ray.is_finite() || !dir.is_finite() {
        return None;
    }
    let facing = ray.dot(dir);
    if !(facing > 0.0) {
        return None;
    }
    let s = ray.cross(dir).length();
    let rho_cap = rho + (max_off + air) / distance;
    // Outside that sphere every local shell is missed. Skip the datum load.
    if !(rho_cap > 0.0 && s < rho_cap) {
        return None;
    }
    let r_at = |t: f32| mapped_limb_radius_at(ray, dir, t, rho, distance, rotation, g, datum);
    let mut r_surf = r_at(facing);
    if let Some((t_in, t_out)) = sphere_roots(facing, rho + max_off / distance) {
        if t_in > 0.0 {
            r_surf = r_surf.max(r_at(t_in));
        }
        if t_out > 0.0 {
            r_surf = r_surf.max(r_at(t_out));
        }
    }
    if !(r_surf > 0.0) {
        return None;
    }
    let shell = (air / distance).max(px);
    let outer = (r_surf + shell).min(rho_cap);
    (s < outer).then_some(MappedShell { r_surf, outer, s })
}

/// `true` when [`ray_mapped_shell`] covers `ray`. `px` is the pixel angle.
#[cfg(test)]
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
    px: f32,
) -> bool {
    ray_mapped_shell(
        ray, dir, rho, distance, rotation, g, datum, max_off, air, px,
    )
    .is_some()
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
        assert_eq!(normalised_radius(1.0e9, 5.0e8).to_bits(), 0.5f32.to_bits());
        let dir = Vec3::new(-0.2, 0.3, 0.8).normalize();
        // A small offset from the centre ray: still inside a rho of 0.25, for both shapes.
        let ray = (dir + Vec3::new(0.05, -0.02, 0.01)).normalize();
        let sphere_far = ray_sphere(ray, dir, rho).expect("sphere hit");
        let sphere_near = ray_sphere(ray, dir, 0.25).unwrap();
        assert_eq!(sphere_far.t.to_bits(), sphere_near.t.to_bits());
        assert_eq!(
            sphere_far.normal.x.to_bits(),
            sphere_near.normal.x.to_bits()
        );
        assert_eq!(
            sphere_far.normal.y.to_bits(),
            sphere_near.normal.y.to_bits()
        );
        assert_eq!(
            sphere_far.normal.z.to_bits(),
            sphere_near.normal.z.to_bits()
        );
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
        assert!(
            out.iter()
                .take(n as usize)
                .all(|b| (0..32).contains(&b.seed))
        );
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

    /// Each body is ray-tested with its centre at distance 1, so normalised
    /// `t` (and a rim's `facing`) cannot order two bodies. The sky shader
    /// compares `t * distance` for hits and `facing * distance` for rims and
    /// point blobs, and keeps the normalised `t` for shading. The inner
    /// sphere's far wall stays the background: a smaller world depth
    /// composites over it, and a hit past the shell does not.
    #[test]
    fn world_depth_orders_bodies_the_normalised_parameter_cannot() {
        let near = ray_sphere(Vec3::Z, Vec3::Z, 0.2).unwrap();
        let far = ray_sphere(Vec3::Z, Vec3::Z, 0.9).unwrap();
        let near_distance = 40.0;
        let far_distance = 4_000.0;
        assert!(near.t > far.t, "normalised t would pick the far body");
        let near_depth = near.t * near_distance;
        let far_depth = far.t * far_distance;
        assert!(near_depth < far_depth);
        assert!((near_depth - 32.0).abs() < 1e-3, "{near_depth}");
        assert!((far_depth - 400.0).abs() < 1e-2, "{far_depth}");

        let wall = ray_inner_sphere(Vec3::Z, Vec3::Z, 100.0, 500.0).unwrap();
        let wall_depth = wall.t * 100.0;
        assert!(
            (wall_depth - 600.0).abs() < 1e-2,
            "far wall world depth {wall_depth}"
        );
        assert!(wall.t > near.t && wall.t > far.t);
        let front = ray_sphere(Vec3::Z, Vec3::Z, 0.5).unwrap();
        assert!(front.t * 80.0 < wall_depth);
        assert!(
            front.t * 2_000.0 > wall_depth,
            "a body past the shell is behind the wall even though its normalised t is smaller"
        );

        // On this ray `facing` is 1. The close rim is still in front of the far hit.
        assert!(1.0 * near_distance < far_depth);
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
                    assert!(
                        err < 2e-4,
                        "p {p} dir {u:?} off by {err} (got {got}, want {point})"
                    );
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
                assert!(
                    (sphere.t - rounded.t).abs() < 2e-4,
                    "{} vs {}",
                    sphere.t,
                    rounded.t
                );
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
        body.shape = FarShape::Rounded {
            exponent: f32::INFINITY,
        };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    }

    #[test]
    fn far_map_basis_matches_the_chart_table() {
        let (x, y, z) = (Vec3::X, Vec3::Y, Vec3::Z);
        assert_eq!(far_map_basis(0), (-y, x, z));
        assert_eq!(far_map_basis(1), (y, -x, z));
        assert_eq!(far_map_basis(2), (x, y, z));
        assert_eq!(far_map_basis(3), (x, -y, -z));
        assert_eq!(far_map_basis(4), (x, z, -y));
        assert_eq!(far_map_basis(5), (x, -z, y));
    }

    #[test]
    fn far_cube_texel_dir_follows_the_vulkan_cube() {
        let normals = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ];
        let size = 8u32;
        for (face, n) in normals.iter().copied().enumerate() {
            for y in 0..size {
                for x in 0..size {
                    let d = far_cube_texel_dir(face, size, x, y);
                    assert!((d.length() - 1.0).abs() < 1e-5, "{d:?}");
                    let a = d.abs();
                    let axis = if a.x >= a.y && a.x >= a.z {
                        0
                    } else if a.y >= a.z {
                        1
                    } else {
                        2
                    };
                    assert_eq!(axis, face / 2, "face {face} texel {x},{y} dir {d}");
                    assert!(d.dot(n) > 0.0, "face {face} points away");
                    let mirror = far_cube_texel_dir(face, size, size - 1 - x, size - 1 - y);
                    let sum = d + mirror;
                    assert!(
                        sum.cross(n).length() < 1e-4,
                        "face {face} texel {x},{y} not symmetric: {sum}"
                    );
                }
            }
        }
        // Image origin of +X is the corner toward +Y and +Z.
        let corner = far_cube_texel_dir(0, 4096, 0, 0);
        let expect = Vec3::new(1.0, 1.0, 1.0).normalize();
        assert!((corner - expect).length() < 1e-3, "{corner} vs {expect}");
    }

    #[test]
    fn far_map_desc_rejects_a_bad_id_or_size() {
        let ok_datum = [0.0f32; 6 * 2 * 2];
        let ok = FarMapDesc {
            datum_res: 2,
            datum: &ok_datum,
            albedo_size: 4,
        };
        assert_eq!(validate_far_map(FarMapId(8), &ok), Err(FarMapError::BadId));
        assert_eq!(
            validate_far_map(FarMapId(255), &ok),
            Err(FarMapError::BadId)
        );
        assert!(validate_far_map(FarMapId(7), &ok).is_ok());

        let mut bad_g = ok;
        bad_g.datum_res = 1;
        assert_eq!(
            validate_far_map(FarMapId(0), &bad_g),
            Err(FarMapError::BadSize)
        );
        bad_g.datum_res = 66;
        assert_eq!(
            validate_far_map(FarMapId(0), &bad_g),
            Err(FarMapError::BadSize)
        );

        let short = FarMapDesc {
            datum_res: 2,
            datum: &[0.0f32; 6],
            albedo_size: 0,
        };
        assert_eq!(
            validate_far_map(FarMapId(0), &short),
            Err(FarMapError::BadSize)
        );

        let mut bad_albedo = ok;
        bad_albedo.albedo_size = 3;
        assert_eq!(
            validate_far_map(FarMapId(0), &bad_albedo),
            Err(FarMapError::BadSize)
        );
        bad_albedo.albedo_size = 4096;
        assert_eq!(
            validate_far_map(FarMapId(0), &bad_albedo),
            Err(FarMapError::BadSize)
        );
        bad_albedo.albedo_size = 0;
        assert!(validate_far_map(FarMapId(0), &bad_albedo).is_ok());
        bad_albedo.albedo_size = 2048;
        assert!(validate_far_map(FarMapId(0), &bad_albedo).is_ok());

        assert_eq!(
            validate_far_map_face(FarMapId(8), 0, 64, Some(4)),
            Err(FarMapError::BadId)
        );
        assert_eq!(
            validate_far_map_face(FarMapId(0), 0, 64, None),
            Err(FarMapError::BadId)
        );
        assert_eq!(
            validate_far_map_face(FarMapId(0), 6, 64, Some(4)),
            Err(FarMapError::BadSize)
        );
        assert_eq!(
            validate_far_map_face(FarMapId(0), 0, 16, Some(4)),
            Err(FarMapError::BadSize)
        );
        assert_eq!(
            validate_far_map_face(FarMapId(0), 0, 64, Some(0)),
            Err(FarMapError::BadSize)
        );
        assert!(validate_far_map_face(FarMapId(0), 5, 64, Some(4)).is_ok());
    }

    #[test]
    fn mapped_is_kept_only_with_a_real_horizon_and_air() {
        let mut body = sphere_at(Vec3::Z, 4.0, 1.0, 1);
        body.shape = FarShape::Mapped {
            map: FarMapId(3),
            horizon: 1.0,
            air: 0.0,
        };
        let mut out = [FarBody::default(); MAX_FAR_BODIES];
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 1);
        body.shape = FarShape::Mapped {
            map: FarMapId(8),
            horizon: 1.0,
            air: 0.0,
        };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
        body.shape = FarShape::Mapped {
            map: FarMapId(0),
            horizon: f32::NAN,
            air: 0.0,
        };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
        body.shape = FarShape::Mapped {
            map: FarMapId(0),
            horizon: 0.0,
            air: -0.01,
        };
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
        // Still an outside shape: the viewer must be beyond the reference radius.
        body.shape = FarShape::Mapped {
            map: FarMapId(0),
            horizon: 1.0,
            air: 0.2,
        };
        body.distance = 1.0;
        body.radius = 1.0;
        assert_eq!(store(std::slice::from_ref(&body), &mut out), 0);
    }

    #[test]
    fn flat_datum_matches_the_sphere() {
        let dir = Vec3::new(-0.2, 0.4, 0.8).normalize();
        let rho = 0.35f32;
        let distance = 8.0f32;
        let g = 4u32;
        let datum = vec![0.0f32; 6 * 16];
        let mut rays = vec![dir];
        for k in 0..12 {
            let ang = (k as f32) * 0.07;
            let ray = (dir + Vec3::new(ang.sin(), ang * 0.3, 0.0)).normalize();
            rays.push(ray);
        }
        rays.push(Vec3::X);
        rays.push(-dir);
        for ray in rays {
            let sphere = ray_sphere(ray, dir, rho);
            let mapped = ray_mapped(
                ray,
                dir,
                rho,
                distance,
                Quat::IDENTITY,
                1.0,
                g,
                &datum,
                0.0,
                0.0,
            );
            match (sphere, mapped) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    assert!((a.t - b.t).abs() < 1e-5, "{ray:?} t {} vs {}", a.t, b.t);
                    assert!(
                        (a.normal - b.normal).length() < 1e-5,
                        "{ray:?} n {:?} vs {:?}",
                        a.normal,
                        b.normal
                    );
                }
                other => panic!("{ray:?} disagreed: {other:?}"),
            }
        }
    }

    #[test]
    fn a_datum_bump_moves_the_hit_closer() {
        let g = 5u32;
        let mut datum = vec![0.0f32; 6 * 25];
        // Centre of the −Z face (the face the +Z centre ray meets).
        let face = 5usize;
        let i = 2usize;
        let j = 2usize;
        datum[face * 25 + j * 5 + i] = 0.2;
        let rho = 0.25f32;
        let distance = 4.0f32;
        let sphere = ray_sphere(Vec3::Z, Vec3::Z, rho).unwrap();
        let hit = ray_mapped(
            Vec3::Z,
            Vec3::Z,
            rho,
            distance,
            Quat::IDENTITY,
            1.0,
            g,
            &datum,
            0.0,
            0.2,
        )
        .expect("bump hits");
        assert!(
            hit.t < sphere.t - 1e-3,
            "bump t {} should be closer than sphere {}",
            hit.t,
            sphere.t
        );
    }

    #[test]
    fn a_limb_ray_that_misses_the_surface_is_a_miss() {
        // max offset 0, the rest sunk, so the hi sphere is the reference and
        // the surface sits inside it. Impact parameter between the two.
        let g = 2u32;
        let mut datum = vec![-0.4f32; 6 * 4];
        datum[2 * 4] = 0.0; // one +Y sample keeps max at 0
        let rho = 0.5f32;
        let distance = 2.0f32;
        let s = 0.45f32;
        let ray = Vec3::new(s, 0.0, (1.0 - s * s).sqrt());
        assert!(ray_sphere(ray, Vec3::Z, rho).is_some());
        assert!(
            ray_mapped(
                ray,
                Vec3::Z,
                rho,
                distance,
                Quat::IDENTITY,
                1.0,
                g,
                &datum,
                -0.4,
                0.0,
            )
            .is_none()
        );
    }

    #[test]
    fn horizon_cull_skips_rays_above_it() {
        let rho = 0.95f32;
        let datum = [0.0f32; 6 * 4];
        let centre = ray_mapped(
            Vec3::Z,
            Vec3::Z,
            rho,
            1.0,
            Quat::IDENTITY,
            -0.8,
            2,
            &datum,
            0.0,
            0.0,
        );
        assert!(centre.is_some());
        let s = (1.0f32 - 0.25).sqrt();
        let raised = Vec3::new(s, 0.0, 0.5);
        assert!(raised.dot(-Vec3::Z) > -0.8);
        assert!(ray_sphere(raised, Vec3::Z, rho).is_some());
        assert!(
            ray_mapped(
                raised,
                Vec3::Z,
                rho,
                1.0,
                Quat::IDENTITY,
                -0.8,
                2,
                &datum,
                0.0,
                0.0,
            )
            .is_none()
        );
        assert!(
            ray_mapped(
                raised,
                Vec3::Z,
                rho,
                1.0,
                Quat::IDENTITY,
                1.0,
                2,
                &datum,
                0.0,
                0.0,
            )
            .is_some()
        );
    }

    #[test]
    fn atan_approx_stays_under_a_datum_cell() {
        let mut max_err = 0.0f64;
        let n = 200_001i32;
        for i in 0..=n {
            let x = -1.0 + 2.0 * f64::from(i) / f64::from(n);
            let got = f64::from(atan_approx(x as f32));
            max_err = max_err.max((got - x.atan()).abs());
        }
        // The stepwise f32 Horner peaks near −0.972. Sweep that neighbourhood.
        for i in 0..20_000 {
            let x = -0.99 + 0.04 * f64::from(i) / 20_000.0;
            let got = f64::from(atan_approx(x as f32));
            max_err = max_err.max((got - x.atan()).abs());
        }
        assert!(
            max_err <= 6.7e-7,
            "atan approx err {max_err} exceeds 6.7e-7 rad"
        );
    }

    #[test]
    fn analytic_normal_matches_the_finite_difference() {
        let g = 33u32;
        let gg = g as usize;
        let mut datum = vec![0.0f32; 6 * gg * gg];
        for face in 0..6 {
            let (tu, n, tv) = far_map_basis(face);
            for j in 0..g {
                for i in 0..g {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    // Small enough that a half-cell finite difference stays within
                    // 1e-3 rad of the analytic slope, and large enough to tilt
                    // the normal off the radius.
                    let bump = 0.005 * (d.x * 1.3 + d.y * 0.7).sin() * (d.z * 1.1).cos();
                    datum[face * gg * gg + j as usize * gg + i as usize] = bump;
                }
            }
        }
        let rho = 0.4f32;
        let distance = 6.0f32;
        let rotation = Quat::from_axis_angle(Vec3::new(0.2, 0.5, 0.8).normalize(), 0.4);
        let mut worst = 0.0f32;
        let mut tilt = 0.0f32;
        for k in 0..64 {
            let z = -1.0 + 2.0 * (k as f32) / 63.0;
            let ang = k as f32 * 0.37;
            let r = (1.0 - z * z).max(0.0).sqrt();
            let dir = Vec3::new(r * ang.cos(), r * ang.sin(), z);
            let fd = mapped_normal(rotation, dir, rho, distance, g, &datum);
            let an = mapped_normal_analytic(rotation, dir, rho, distance, g, &datum);
            let err = fd.dot(an).clamp(-1.0, 1.0).acos();
            worst = worst.max(err);
            // `dir` is the world radius; the normal is world-space too.
            tilt = tilt.max(dir.dot(an).clamp(-1.0, 1.0).acos());
        }
        assert!(tilt > 1e-3, "datum did not tilt the normal ({tilt})");
        assert!(worst < 1e-3, "normal angle {worst} rad");
    }

    #[test]
    fn fixed_point_hits_match_regula() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        // Low-frequency height. A texel of white noise makes the fixed point
        // oscillate, and a planet datum does not look like that. Amplitude is
        // a few percent of the radius, frequencies a few radians, phases random.
        let g = 17u32;
        let gg = g as usize;
        let phase =
            |rng: &mut dyn FnMut() -> u32| (rng() as f32 / u32::MAX as f32) * std::f32::consts::TAU;
        let p1 = phase(&mut next);
        let p2 = phase(&mut next);
        let p3 = phase(&mut next);
        let mut datum = vec![0.0f32; 6 * gg * gg];
        for face in 0..6 {
            let (tu, n, tv) = far_map_basis(face);
            for j in 0..g {
                for i in 0..g {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    // A few percent of the radius. Steeper than terrain, flat
                    // enough that two or three fixed-point steps settle.
                    let bump = 0.05 * (d.x * 2.0 + p1).sin() * (d.y * 1.7 + p2).cos()
                        + 0.025 * (d.z * 3.0 + p3).sin();
                    datum[face * gg * gg + j as usize * gg + i as usize] = bump;
                }
            }
        }
        let unit = |rng: &mut dyn FnMut() -> u32| {
            let z = (rng() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            let ang = (rng() as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            let radial = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(radial * ang.cos(), radial * ang.sin(), z)
        };
        let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
        let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
        let rho = 0.45f32;
        let distance = 5.0f32;
        let mut hits = 0u32;
        let mut worst = 0.0f32;
        for _ in 0..400 {
            let dir = unit(&mut next);
            let axis = unit(&mut next);
            let turn = (next() as f32 / u32::MAX as f32) * 3.0;
            let rotation = Quat::from_axis_angle(axis, turn);
            let spread = (next() as f32 / u32::MAX as f32) * 0.7;
            let ray = (dir + unit(&mut next) * spread).normalize();
            let old = ray_mapped_regula(
                ray, dir, rho, distance, rotation, 1.0, g, &datum, min_off, max_off,
            );
            let new = ray_mapped(
                ray, dir, rho, distance, rotation, 1.0, g, &datum, min_off, max_off,
            );
            match (old, new) {
                (None, None) => {}
                (Some(a), Some(b)) => {
                    hits += 1;
                    worst = worst.max((a.t - b.t).abs());
                }
                (old, new) => {
                    panic!("hit mismatch old {old:?} new {new:?} ray {ray:?} dir {dir:?}")
                }
            }
        }
        // The 6-step regula is the coarser of the two. The safeguarded march
        // is held to the f64 reference elsewhere; here they must still name
        // the same hit, within a couple of millionths in normalised t.
        assert!(worst < 2e-6, "t delta {worst}");
        assert!(hits > 50, "only {hits} rays hit");
    }

    /// Home-planet radius, in blocks. Datum offsets sit in `[-278_000, 1_040_000]`.
    const HOME_RADIUS: f64 = 31_017_520.0;

    /// Low-order height, then an affine map onto the home offset range.
    /// Smooth on the scale of a face: a sum of a few harmonics, not noise.
    fn home_datum(g: u32) -> Vec<f32> {
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
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
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

    fn sphere_roots_f64(facing: f64, rho: f64) -> Option<(f64, f64)> {
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

    /// Residual of the shader's bilinear datum along an f64 ray.
    /// `t` is refined in f64; the height sample is the same f32 bilinear the
    /// march uses, including the equiangular polynomial.
    fn mapped_residual_f64(
        t: f64,
        ray: glam::DVec3,
        center: glam::DVec3,
        rho: f64,
        distance: f64,
        rotation: Quat,
        g: u32,
        datum: &[f32],
    ) -> f64 {
        let p = ray * t - center;
        let rad = p.length();
        if rad < 1e-12 {
            return -rho;
        }
        let u = p / rad;
        let u32 = Vec3::new(u.x as f32, u.y as f32, u.z as f32);
        let body = rotate(conjugate(rotation), u32);
        let height = f64::from(sample_datum(g, datum, body));
        rad - (rho + height / distance)
    }

    /// f64 reference intersection. Dense march inside
    /// `[max(t_hi_in, 0), min(t_lo_in, t_hi_out)]` (or `t_hi_out` when the ray
    /// misses the lo sphere), with a step well under one datum cell projected
    /// on the ray, then bisection to `1e-12` in normalised `t`.
    fn ray_mapped_reference(
        ray: Vec3,
        dir: Vec3,
        rho: f32,
        distance: f32,
        rotation: Quat,
        g: u32,
        datum: &[f32],
        min_off: f32,
        max_off: f32,
    ) -> Option<f64> {
        if !(rho > 0.0) || !(distance > 0.0) || !ray.is_finite() || !dir.is_finite() {
            return None;
        }
        let ray_d = ray.as_dvec3().normalize();
        let dir_d = dir.as_dvec3().normalize();
        let rho_d = f64::from(rho);
        let distance_d = f64::from(distance);
        let rho_lo = rho_d + f64::from(min_off) / distance_d;
        let rho_hi = rho_d + f64::from(max_off) / distance_d;
        let facing = ray_d.dot(dir_d);
        let (t_hi_in, t_hi_out) = sphere_roots_f64(facing, rho_hi)?;
        if !(t_hi_out > 0.0) {
            return None;
        }
        let t_start = t_hi_in.max(0.0);
        let t_lo =
            sphere_roots_f64(facing, rho_lo).and_then(|(t_lo, _)| (t_lo > 0.0).then_some(t_lo));
        let t_end = t_lo.map(|t| t.min(t_hi_out)).unwrap_or(t_hi_out);
        let f_at =
            |t: f64| mapped_residual_f64(t, ray_d, dir_d, rho_d, distance_d, rotation, g, datum);
        // Lo and hi spheres coincide on a constant datum: the entrance is the hit.
        if !(t_end > t_start) {
            let f0 = f_at(t_start);
            return (t_start > 0.0 && f0.abs() <= 1e-5).then_some(t_start);
        }
        let f_start = f_at(t_start);
        if f_start < -1e-5 {
            return None;
        }
        if f_start.abs() <= 1e-8 {
            return (t_start > 0.0).then_some(t_start);
        }
        // One datum cell is π / (2 (g-1)) radians. Along the ray that is
        // `cell * rad / sin(phi)`. A tenth of that cannot step over the
        // bilinear's single sign change inside a cell.
        let cells = f64::from(g.max(2) - 1);
        let cell = std::f64::consts::FRAC_PI_2 / cells;
        let mut t = t_start;
        let mut prev_f = f_start;
        let mut bracket: Option<(f64, f64)> = None;
        let mut guard = 0u32;
        while t < t_end && guard < 100_000 {
            guard += 1;
            let p = ray_d * t - dir_d;
            let rad = p.length().max(rho_d);
            let sin_phi = if rad > 1e-8 {
                ray_d.cross(p / rad).length()
            } else {
                0.0
            };
            let dt = (cell * rad / sin_phi.max(0.05)) * 0.1;
            let next = (t + dt.max(1e-8)).min(t_end);
            let ft = f_at(next);
            if prev_f * ft <= 0.0 {
                bracket = Some((t, next));
                break;
            }
            t = next;
            prev_f = ft;
        }
        let (mut a, mut b) = bracket?;
        for _ in 0..80 {
            if (b - a).abs() <= 1e-12 {
                break;
            }
            let mid = 0.5 * (a + b);
            let fm = f_at(mid);
            let fa = f_at(a);
            if fa * fm <= 0.0 {
                b = mid;
            } else {
                a = mid;
            }
        }
        let root = 0.5 * (a + b);
        (root > 0.0 && root.is_finite()).then_some(root)
    }

    #[test]
    fn reference_matches_spheres_of_constant_datum() {
        let g = 33u32;
        let gg = g as usize;
        let distance = 4.0f32;
        for offset in [0.0f32, 0.25] {
            let datum = vec![offset; 6 * gg * gg];
            let rho = 0.35f32;
            // Facing stays above the graze of rho 0.35 (disc needs |ray·dir| ≳ 0.94).
            for (ray, dir) in [
                (Vec3::Z, Vec3::Z),
                (Vec3::new(0.25, -0.1, 1.0).normalize(), Vec3::Z),
                (
                    Vec3::new(0.15, 0.05, 1.0).normalize(),
                    Vec3::new(0.05, -0.08, 1.0).normalize(),
                ),
            ] {
                // Same f64 ray the reference normalises, and the same radius
                // the residual subtracts (f64 division, not the f32 sphere).
                let facing = ray.as_dvec3().normalize().dot(dir.as_dvec3().normalize());
                let radius = f64::from(rho) + f64::from(offset) / f64::from(distance);
                let (t_near, _) = sphere_roots_f64(facing, radius).unwrap();
                let got = ray_mapped_reference(
                    ray,
                    dir,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    g,
                    &datum,
                    offset,
                    offset,
                )
                .unwrap_or_else(|| panic!("reference missed offset {offset} ray {ray:?}"));
                assert!(
                    (got - t_near).abs() < 1e-9,
                    "offset {offset} reference {got} analytic {t_near}"
                );
            }
        }
    }

    #[test]
    fn reference_straight_down_is_the_local_altitude() {
        let g = 33u32;
        let datum = home_datum(g);
        let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
        let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
        assert!(min_off > -278_001.0 && max_off < 1_040_001.0);
        assert!(max_off - min_off > 500_000.0, "harmonics collapsed");
        let up = Vec3::Y;
        let local = sample_datum(g, &datum, up);
        for altitude in [10_000.0f32, 50_000.0, 1.0e5, 1.0e7] {
            let distance = (HOME_RADIUS + f64::from(local) + f64::from(altitude)) as f32;
            let rho = (HOME_RADIUS as f32) / distance;
            let dir = -up;
            let got = ray_mapped_reference(
                dir,
                dir,
                rho,
                distance,
                Quat::IDENTITY,
                g,
                &datum,
                min_off,
                max_off,
            )
            .expect("straight down hits");
            // f32 radius/distance, not the f64 altitude ratio: R is not an f32 integer.
            let r = f64::from(rho) + f64::from(local) / f64::from(distance);
            let expect = 1.0 - r;
            assert!(
                (got - expect).abs() < 1e-9,
                "alt {altitude} reference {got} 1 - r(local) {expect}"
            );
        }
    }

    /// Angle from the centre direction. `0` looks straight down.
    fn ray_from_nadir(dir: Vec3, azimuth: f32, angle: f32) -> Vec3 {
        let down = dir.normalize();
        let up = -down;
        let reference = if up.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let east = reference.cross(up).normalize();
        let north = up.cross(east);
        let horiz = east * azimuth.sin() + north * azimuth.cos();
        (down * angle.cos() + horiz * angle.sin()).normalize()
    }

    /// Outermost reference hit, sweeping from nadir toward the anti-centre.
    fn reference_horizon_angle(
        dir: Vec3,
        azimuth: f32,
        rho: f32,
        distance: f32,
        g: u32,
        datum: &[f32],
        min_off: f32,
        max_off: f32,
    ) -> f64 {
        let hits = |angle: f64| {
            let ray = ray_from_nadir(dir, azimuth, angle as f32);
            ray_mapped_reference(
                ray,
                dir,
                rho,
                distance,
                Quat::IDENTITY,
                g,
                datum,
                min_off,
                max_off,
            )
            .is_some()
        };
        assert!(hits(0.0), "nadir missed");
        let mut lo = 0.0f64;
        let mut hi = std::f64::consts::PI;
        if hits(hi) {
            return hi;
        }
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if hits(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    /// Fixed-point march against the f64 reference, on the same home-datum
    /// sweep the bracket march passed at `2e-4` relative with no hit/miss
    /// disagreements.
    ///
    /// Measured on this sweep (208 rays: altitudes 10 km, 50 km, 100 km and
    /// 10,000 km, four azimuths, nadir through the horizon band):
    /// worst relative `|Δt|/t` = 8.133e-4, at 100 km, azimuth 0, 0.002 rad
    /// inside the reference horizon. Grazing horizon misses = 11, each the
    /// sample 2e-4 rad inside that horizon (the reference hits and the
    /// fixed-point march misses). A disagreement any farther away, or a
    /// fixed-point hit where the reference misses, fails. Misses that close
    /// to the horizon are the limb's job.
    #[test]
    fn mapped_solver_matches_the_reference_near_the_ground() {
        // Just above the measured 8.133e-4. The march is deterministic, so
        // the slack is one digit of the reported figure.
        const FAST_WORST_REL: f64 = 8.14e-4;
        /// The sweep's closest interior sample. Every measured miss lands here.
        const FAST_GRAZE_RAD: f64 = 2.0e-4;
        const FAST_GRAZING_MISSES: u32 = 11;

        let g = 33u32;
        let datum = home_datum(g);
        let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
        let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
        let local = sample_datum(g, &datum, Vec3::Y);
        let radius = HOME_RADIUS as f32;
        let mut worst = 0.0f64;
        let mut worst_where = String::new();
        let mut grazing_misses = 0u32;
        let mut other_misses = 0u32;
        let mut compared = 0u32;
        for altitude in [10_000.0f32, 50_000.0, 1.0e5, 1.0e7] {
            let distance = radius + local + altitude;
            let rho = radius / distance;
            let dir = -Vec3::Y;
            for azimuth in [0.0f32, 1.3, 2.6, 4.2] {
                let horizon = reference_horizon_angle(
                    dir, azimuth, rho, distance, g, &datum, min_off, max_off,
                );
                let mut angles = vec![0.0f64, 0.4, 0.8, 1.2217304763960306];
                for delta in [
                    -0.2, -0.05, -0.02, -0.01, -0.005, -0.002, -2.0e-4, 2.0e-4, 0.02,
                ] {
                    let angle = horizon + delta;
                    if angle > 0.0 && angle < std::f64::consts::PI {
                        angles.push(angle);
                    }
                }
                for angle in angles {
                    let ray = ray_from_nadir(dir, azimuth, angle as f32);
                    let reference = ray_mapped_reference(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    );
                    let got = ray_mapped_fast(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        1.0,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    );
                    let from_horizon = (angle - horizon).abs();
                    compared += 1;
                    match (reference, got) {
                        (None, None) => {}
                        (Some(t_ref), Some(hit)) => {
                            let rel = (f64::from(hit.t) - t_ref).abs() / t_ref;
                            if rel > worst {
                                worst = rel;
                                worst_where = format!(
                                    "alt {altitude} az {azimuth} angle {angle} \
                                     from_horizon {from_horizon} t {t_ref} got {}",
                                    hit.t
                                );
                            }
                        }
                        (Some(_), None) if from_horizon <= FAST_GRAZE_RAD => {
                            grazing_misses += 1;
                        }
                        (reference, got) => {
                            other_misses += 1;
                            if other_misses <= 8 {
                                eprintln!(
                                    "hit/miss alt {altitude} az {azimuth} angle {angle} \
                                     from_horizon {from_horizon} horizon {horizon} \
                                     ref {reference:?} got {got:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
        assert!(
            other_misses == 0 && grazing_misses <= FAST_GRAZING_MISSES && worst <= FAST_WORST_REL,
            "fixed-point march disagrees with the f64 reference: {other_misses} \
             non-grazing hit/miss mismatches over {compared} rays, \
             {grazing_misses} grazing horizon misses (bound {FAST_GRAZING_MISSES}, \
             within {FAST_GRAZE_RAD} rad), worst relative |Δt|/t = {worst} \
             (bound {FAST_WORST_REL}) at {worst_where}"
        );
    }

    /// Rays inside the lo-sphere disc hit far from a graze, so the five-sample
    /// march and the bracket reference name the same surface.
    #[test]
    fn fast_march_matches_robust_inside_the_lo_sphere_disc() {
        let g = 33u32;
        let datum = home_datum(g);
        let min_off = datum.iter().copied().fold(f32::MAX, f32::min);
        let max_off = datum.iter().copied().fold(f32::MIN, f32::max);
        let radius = HOME_RADIUS as f32;
        let local = sample_datum(g, &datum, Vec3::Y);
        let dir = -Vec3::Y;
        let mut worst = 0.0f64;
        let mut worst_where = String::new();
        let mut compared = 0u32;
        // Eye altitude above the local surface, 10 m through 1e6 m.
        for altitude in [10.0f32, 100.0, 1_000.0, 10_000.0, 1.0e5, 1.0e6] {
            let distance = radius + local + altitude;
            let rho = radius / distance;
            let rho_lo = rho + min_off / distance;
            assert!(
                rho_lo > 0.0 && rho_lo < 1.0,
                "alt {altitude} rho_lo {rho_lo} is not an exterior lo sphere"
            );
            // Half-angle of the lo-sphere disc. A unit ray at angle θ from the
            // centre hits that sphere when sin θ < rho_lo. Stay strictly inside.
            let limb = f64::from(rho_lo).asin();
            for azimuth in [0.0f32, 1.1, 2.4, 3.7, 5.0] {
                for frac in [0.0f64, 0.2, 0.45, 0.7, 0.9, 0.98] {
                    let angle = limb * frac;
                    let ray = ray_from_nadir(dir, azimuth, angle as f32);
                    let robust = ray_mapped(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        1.0,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    );
                    let fast = ray_mapped_fast(
                        ray,
                        dir,
                        rho,
                        distance,
                        Quat::IDENTITY,
                        1.0,
                        g,
                        &datum,
                        min_off,
                        max_off,
                    );
                    let (Some(robust), Some(fast)) = (robust, fast) else {
                        panic!(
                            "interior ray missed alt {altitude} az {azimuth} \
                             frac {frac} angle {angle} robust {robust:?} fast {fast:?}"
                        );
                    };
                    compared += 1;
                    let rel =
                        (f64::from(fast.t) - f64::from(robust.t)).abs() / f64::from(robust.t).abs();
                    if rel > worst {
                        worst = rel;
                        worst_where = format!(
                            "alt {altitude} az {azimuth} frac {frac} \
                             robust {} fast {}",
                            robust.t, fast.t
                        );
                    }
                }
            }
        }
        assert!(
            compared > 100 && worst <= 1.0e-6,
            "fast march disagrees inside the lo disc: {compared} rays, \
             worst relative |Δt|/t = {worst} at {worst_where}"
        );
    }

    /// Cube LOD the shader derives from the pixel footprint. `ndot` is
    /// `|ray · n|`. The face-centre texel is `(π/2) R / albedo_size`, and the
    /// result is clamped to the mip range `[0, log2(albedo_size)]`.
    fn mapped_albedo_lod(
        px: f32,
        t: f32,
        distance: f32,
        radius: f32,
        ndot: f32,
        albedo_size: u32,
    ) -> f32 {
        let nd = ndot.abs().max(0.05);
        let footprint = px * t * distance / nd;
        let texel = (std::f32::consts::FRAC_PI_2 * radius) / albedo_size as f32;
        let lod = (footprint / texel).max(1e-20).log2();
        let mip_max = (albedo_size as f32).log2();
        lod.clamp(0.0, mip_max)
    }

    #[test]
    fn albedo_lod_rises_toward_the_horizon() {
        let px = 0.01f32;
        let t = 0.5f32;
        let distance = 1.0e7f32;
        let radius = 3.101752e7f32;
        let albedo = 1024u32;
        let down = mapped_albedo_lod(px, t, distance, radius, 1.0, albedo);
        let graze = mapped_albedo_lod(px, t, distance, radius, 0.05, albedo);
        let flatter = mapped_albedo_lod(px, t, distance, radius, 0.01, albedo);
        let levels = std::f32::consts::LN_2;
        let expect = (20.0f32).ln() / levels;
        assert!(
            (graze - down - expect).abs() < 1e-4,
            "nadir {down} graze {graze} expected +{expect}"
        );
        assert!((flatter - graze).abs() < 1e-5, "ndot below 0.05 must clamp");
        assert!(down > 0.0 && graze < (albedo as f32).log2());
        let tiny = mapped_albedo_lod(1e-6, 1e-4, distance, radius, 1.0, albedo);
        let huge = mapped_albedo_lod(1.0, 10.0, distance, radius, 0.05, albedo);
        assert_eq!(tiny, 0.0);
        assert_eq!(huge, (albedo as f32).log2());
    }

    /// Lowland everywhere, a broad highland in front of +X. The zenith sample
    /// stays 0 so an eye on +Y is above the ground, not inside the bump.
    fn horizon_highland(g: u32, beta0: f32, beta1: f32, az_half: f32, height: f32) -> Vec<f32> {
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
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
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

    /// Closed-form tangent of the outer air sphere, or `1` when that sphere
    /// contains the camera (the game then disables the scalar cone).
    fn outer_horizon_sine(rho_cap: f32) -> f32 {
        if rho_cap > 0.0 && rho_cap < 1.0 {
            -(1.0 - rho_cap * rho_cap).max(0.0).sqrt()
        } else {
            1.0
        }
    }

    fn limb_chord(shell: &MappedShell) -> f32 {
        let outer_c = (shell.outer * shell.outer - shell.s * shell.s)
            .max(0.0)
            .sqrt();
        if shell.s >= shell.r_surf {
            2.0 * outer_c
        } else {
            let inner = (shell.r_surf * shell.r_surf - shell.s * shell.s)
                .max(0.0)
                .sqrt();
            2.0 * (outer_c - inner)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SilhouetteClass {
        Hit,
        Limb,
        Miss,
    }

    /// Highland at the horizon: every fixed-point ray hits, or gets a limb on
    /// the surface it grazes. A miss next to the silhouette is a hole. No
    /// limb past the air cap.
    #[test]
    fn highland_silhouette_is_continuous() {
        let g = 33u32;
        let radius = HOME_RADIUS as f32;
        let air = 20_000.0f32;
        let px = 1.5e-3f32;
        let dir = -Vec3::Y;
        let datum = horizon_highland(g, 0.03, 0.55, 0.45, 1_000_000.0);
        let max_off = datum.iter().copied().fold(0.0f32, f32::max);
        let min_off = 0.0f32;
        assert_eq!(sample_datum(g, &datum, Vec3::Y), 0.0);
        let mut inside_empty = 0u32;
        let mut limb_above = 0u32;
        let mut holes = 0u32;
        let mut gaps = 0u32;
        let mut gap_where = String::new();
        let mut limb_n = 0u32;
        let mut chord_jump = 0u32;
        for altitude in [50_000.0f32, 2_000_000.0] {
            let distance = radius + altitude;
            let rho = radius / distance;
            let rho_cap = rho + (max_off + air) / distance;
            let horizon = outer_horizon_sine(rho_cap);
            for azimuth in [0.0f32, 0.35, 0.55, 1.2] {
                let horizon_ang = reference_horizon_angle(
                    dir, azimuth, rho, distance, g, &datum, min_off, max_off,
                );
                let mut row: Vec<(SilhouetteClass, f32, f32, f32, f32)> = Vec::new();
                for k in -40..=80 {
                    let angle = horizon_ang + f64::from(k) * 2.0e-4;
                    if !(angle > 0.0 && angle < std::f64::consts::PI) {
                        continue;
                    }
                    let ray = ray_from_nadir(dir, azimuth, angle as f32);
                    let facing = ray.dot(dir);
                    let above = ray.dot(-dir) > horizon;
                    let hit = if above {
                        None
                    } else {
                        ray_mapped_fast(
                            ray,
                            dir,
                            rho,
                            distance,
                            Quat::IDENTITY,
                            horizon,
                            g,
                            &datum,
                            min_off,
                            max_off,
                        )
                    };
                    let shell = if above {
                        None
                    } else {
                        ray_mapped_shell(
                            ray,
                            dir,
                            rho,
                            distance,
                            Quat::IDENTITY,
                            g,
                            &datum,
                            max_off,
                            air,
                            px,
                        )
                    };
                    let s = ray.cross(dir).length();
                    let r_local = if facing > 0.0 {
                        mapped_radius_at(ray, dir, facing, rho, distance, Quat::IDENTITY, g, &datum)
                    } else {
                        0.0
                    };
                    let class = if hit.is_some() {
                        SilhouetteClass::Hit
                    } else if shell.is_some() {
                        SilhouetteClass::Limb
                    } else {
                        SilhouetteClass::Miss
                    };
                    if facing > 0.0 && s + 1e-5 < r_local && class == SilhouetteClass::Miss {
                        inside_empty += 1;
                    }
                    if let Some(sh) = &shell {
                        if class == SilhouetteClass::Limb && sh.s > rho_cap + 1e-5 {
                            limb_above += 1;
                        }
                    }
                    let chord = shell.as_ref().map(limb_chord).unwrap_or(0.0);
                    let outer = shell.as_ref().map(|sh| sh.outer).unwrap_or(0.0);
                    if class == SilhouetteClass::Limb {
                        limb_n += 1;
                    }
                    row.push((class, facing, chord, angle as f32, outer));
                }
                for w in row.windows(3) {
                    let (a, _, _, _, _) = w[0];
                    let (b, facing, _, ang, outer) = w[1];
                    let (c, _, _, _, _) = w[2];
                    if b == SilhouetteClass::Miss
                        && a != SilhouetteClass::Miss
                        && c != SilhouetteClass::Miss
                    {
                        holes += 1;
                    }
                    if a == SilhouetteClass::Hit && b == SilhouetteClass::Miss && facing > 0.0 {
                        gaps += 1;
                        if gap_where.is_empty() {
                            gap_where = format!(
                                "alt {altitude} az {azimuth} ang {ang} facing {facing} outer {outer} href {horizon_ang} {:?}->{:?}->{:?}",
                                a, b, c
                            );
                        }
                    }
                }
                for w in row.windows(2) {
                    let (a, _, ca, _, _) = w[0];
                    let (b, _, cb, _, _) = w[1];
                    if a == SilhouetteClass::Limb && b == SilhouetteClass::Limb {
                        let scale = ca.max(cb).max(px);
                        if (ca - cb).abs() > 8.0 * scale {
                            chord_jump += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(
            inside_empty, 0,
            "rays inside the local surface with no hit and no limb"
        );
        assert_eq!(limb_above, 0, "limb rays past the outer air sphere");
        assert_eq!(
            gaps, 0,
            "hit then miss with no limb between the surface and space ({gap_where})"
        );
        assert_eq!(holes, 0, "limb band with a miss hole in it");
        assert_eq!(chord_jump, 0, "limb chord jumped between adjacent rays");
        assert!(limb_n > 0, "expected some limb rays on the highland");
    }
}
