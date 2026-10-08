//! Scale-normalised far-body impostors. Outside shapes sit at distance 1 along
//! `dir`, with radius `radius/distance`. An inner sphere is that same space
//! with the viewer inside (`distance < radius`); the hit is the far root. The
//! ray tests run in `shaders/far_body.slang`; their CPU mirror is the
//! test-only `mirror` module.

use glam::{Quat, Vec3};

use crate::color::LinearRgb;
use crate::genconst::{FAR_ROUNDED_P_MAX, FAR_ROUNDED_P_MIN};

#[cfg(test)]
pub(crate) mod mirror;

/// Bodies kept from one [`crate::Frame3D::set_far_bodies`] call. Extra entries are dropped.
pub const MAX_FAR_BODIES: usize = 32;

/// Datum-mapped planets the sky pass can hold at once.
pub const MAX_FAR_MAPS: usize = 8;

// The sky shaders size the far table and the map arrays with the generated
// twins. A tile mask word has one bit per kept body.
const _: () = assert!(crate::genconst::MAX_FAR_BODIES as usize == MAX_FAR_BODIES);
const _: () = assert!(MAX_FAR_BODIES <= u32::BITS as usize);
const _: () = assert!(crate::genconst::MAX_FAR_MAPS as usize == MAX_FAR_MAPS);

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
    /// The sky draws p up to [`crate::genconst::FAR_ROUNDED_P_MAX`] (16384,
    /// whose corners sit at 0.99993 of a cube corner); a larger exponent draws
    /// that shape, air rim included.
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
    /// draws no limb). A miss ray's limb is the air that ray actually
    /// crosses, between the local datum and that datum plus `air`. No limb
    /// is drawn when the eye's altitude above the local datum is below `air`.
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

/// The exponent the sky draws a rounded body with: `exponent` clamped to
/// `[FAR_ROUNDED_P_MIN, FAR_ROUNDED_P_MAX]`. Twin of `far_round_p` in
/// `far_body.slang`.
pub(crate) fn rounded_p(exponent: f32) -> f32 {
    exponent.clamp(FAR_ROUNDED_P_MIN, FAR_ROUNDED_P_MAX)
}

/// Corner reach `rho · 3^(1/2 − 1/p)` of the drawn rounded body at the
/// clamped exponent: the enclosing sphere before its pad, and the air rim's
/// inner edge. Twin of `far_round_bound` in `far_body.slang`.
pub(crate) fn rounded_bound(rho: f32, exponent: f32) -> f32 {
    rho * 3.0f32.powf(0.5 - 1.0 / rounded_p(exponent))
}

fn keep(body: &FarBody) -> Option<FarBody> {
    match body.shape {
        FarShape::Rounded { exponent } => {
            if !(exponent >= FAR_ROUNDED_P_MIN) || !exponent.is_finite() {
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

#[cfg(test)]
pub(crate) mod tests;
