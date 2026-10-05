//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_body.slang` (160 bytes, std430).
//! A 16-byte cone sits in front of each record so a pixel can reject the body
//! before loading it. A tile mask follows the records: one bit per kept body,
//! one word per screen tile, so a sky pixel skips bodies that miss its tile.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

/// `VOXEL_FAR_CULL=0` disables the cone reject, the CPU frustum compaction, and
/// the per-tile mask (every tile keeps every surviving body). Any other value,
/// including unset, leaves culling on. Read once.
fn far_cull_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| !std::env::var("VOXEL_FAR_CULL").is_ok_and(|v| v == "0"))
}

/// One body on the GPU. Ten 16-byte lanes, 160 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct FarBodyGpu {
    pub dir_rho: [f32; 4],
    pub rot: [f32; 4],
    pub albedo0: [f32; 4],
    pub albedo1: [f32; 4],
    pub albedo2: [f32; 4],
    pub albedo3: [f32; 4],
    pub albedo4: [f32; 4],
    pub albedo5: [f32; 4],
    /// rgb rim tint, a = shape (0 cube, 1 sphere, 2 inner sphere, 3 rounded).
    pub atmosphere: [f32; 4],
    /// x = noise seed. y = rounded exponent as f32 bits (0 for other shapes).
    pub seed: [u32; 4],
}

/// Mask words in the GPU table. `shaders/far_body.slang` `tile_mask` is this
/// long. 8192 tiles cover 7680×4320 at 64 px; a larger render extent uses a
/// coarser tile so the count still fits.
const MAX_FAR_TILES: usize = 8192;
/// Preferred tile, in pixels. Coarsens when the grid would exceed [`MAX_FAR_TILES`].
const TILE_FINE_PX: u32 = 64;
const TILE_COARSE_PX: u32 = 128;

/// Header, cones, body records, then one mask word per screen tile.
/// `header[0]` is the live count. `header[1]` is the tile size in pixels,
/// `header[2]` is `tiles_x`, `header[3]` is `tiles_y` (all zero with no view).
/// `cone[i] = (dir.xyz, sine bound)` and a bound of `-1` means the pixel test
/// must not reject that body and every tile keeps bit `i`.
/// `tile_mask[ty * tiles_x + tx]` bit `k` is set when kept body `k` may cover
/// that tile. Only the `tiles_x * tiles_y` prefix is live.
///
/// `Pod` is implemented by hand: bytemuck's derive only covers arrays up to a
/// few dozen elements, and the mask is 8192 words. The layout is plain
/// `repr(C)` floats and uints with no padding.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct FarTableGpu {
    pub header: [u32; 4],
    pub cone: [[f32; 4]; MAX_FAR_BODIES],
    pub body: [FarBodyGpu; MAX_FAR_BODIES],
    pub tile_mask: [u32; MAX_FAR_TILES],
}

// SAFETY: every field is plain `f32`/`u32` bits, `repr(C)`, and the size is a
// multiple of the 16-byte alignment, so the all-zero bit pattern is valid and
// there is no padding to exclude from a byte copy.
unsafe impl Zeroable for FarTableGpu {}
unsafe impl Pod for FarTableGpu {}

const _: () = assert!(std::mem::size_of::<FarBodyGpu>() == 160);
const _: () = assert!(
    std::mem::size_of::<FarTableGpu>()
        == 16 + 16 * MAX_FAR_BODIES + 160 * MAX_FAR_BODIES + 4 * MAX_FAR_TILES
);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, cone) == 16);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, body) == 16 + 16 * MAX_FAR_BODIES);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, tile_mask) == 16 + 16 * MAX_FAR_BODIES + 160 * MAX_FAR_BODIES
);

/// Sine of the angular radius the shader can draw (`s = ‖ray × center‖`), or
/// `-1` when a ray facing away from the centre can still draw something.
///
/// Re-checked against `far_bodies()`:
///
/// * **Sphere, camera outside (`rho < 1`).** The disc, the soft point and the
///   air rim all sit behind `if (!inner && !rounded && facing <= 0) continue`.
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
fn cone_bound(body: &FarBody) -> f32 {
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
    }
}

/// What the sky pass needs to drop bodies that cannot meet this frame's view.
/// `view_proj` is the clean (unjittered) camera-relative matrix. `px_max` is an
/// upper bound on the pixel angle, with a factor of two over the projection.
/// `width` and `height` are the render extent the sky pass draws, the same one
/// `px_max` was measured against.
#[derive(Clone, Copy)]
pub(crate) struct FarView {
    pub view_proj: glam::Mat4,
    pub px_max: f32,
    pub width: u32,
    pub height: u32,
}

/// Vertical pixel angle `2 * tan(fovy/2) / height`, the horizontal equivalent
/// from the projection, the max of those, times two.
pub(crate) fn px_max(fovy_tan_half: f32, view_proj: glam::Mat4, width: u32, height: u32) -> f32 {
    let h = height.max(1) as f32;
    let w = width.max(1) as f32;
    let vert = 2.0 * fovy_tan_half / h;
    // Row 0 of `proj * view` has length `1 / tan(fovx/2)` (the view is rigid).
    let row0 = view_proj.transpose().x_axis.truncate().length().max(1e-20);
    let horiz = 2.0 / (row0 * w);
    vert.max(horiz) * 2.0
}

pub(crate) fn far_view(
    fovy_tan_half: f32,
    view_proj: glam::Mat4,
    width: u32,
    height: u32,
) -> FarView {
    let width = width.max(1);
    let height = height.max(1);
    FarView {
        view_proj,
        px_max: px_max(fovy_tan_half, view_proj, width, height),
        width,
        height,
    }
}

/// `(tile size, tiles_x, tiles_y)`. 64 px unless that grid exceeds
/// [`MAX_FAR_TILES`], then 128, then coarser powers of two. The chosen size is
/// what the shader reads from the header, so the mask never overruns the array.
fn tile_layout(width: u32, height: u32) -> (u32, u32, u32) {
    let width = width.max(1);
    let height = height.max(1);
    let mut tile = TILE_FINE_PX;
    loop {
        let tiles_x = width.div_ceil(tile);
        let tiles_y = height.div_ceil(tile);
        if tiles_x as u64 * tiles_y as u64 <= MAX_FAR_TILES as u64 {
            return (tile, tiles_x, tiles_y);
        }
        if tile >= width.max(height) {
            return (width.max(height), 1, 1);
        }
        let next = if tile < TILE_COARSE_PX {
            TILE_COARSE_PX
        } else {
            tile.saturating_mul(2)
        };
        tile = next.min(width.max(height)).max(tile + 1);
    }
}

/// Inward normals of the four side planes, for directions through the eye.
/// Gribb-Hartmann on the rows of `view_proj` (row3 ± row0, row3 ± row1); the
/// translation drops out because each side plane contains the camera.
fn side_normals(view_proj: glam::Mat4) -> [glam::Vec3; 4] {
    let rows = view_proj.transpose();
    let (r0, r1, r3) = (rows.x_axis, rows.y_axis, rows.w_axis);
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1].map(|plane| {
        let n = plane.truncate();
        let len = n.length();
        if len > 1e-20 {
            n / len
        } else {
            glam::Vec3::ZERO
        }
    })
}

/// `bound` is the cone's sine, or `-1` to always keep the body. A cone of sine
/// `a = min(bound + 3 px, 1)` lies outside an inward plane `n` when
/// `dot(n, dir) < -a`. `a >= 1` is a hemisphere (`f > 0` in the shader, since
/// `|ray × dir|` cannot exceed 1), not the whole sky: keep it only when that
/// hemisphere meets the frustum.
fn cone_in_view(dir: glam::Vec3, bound: f32, view: &FarView) -> bool {
    if !(bound >= 0.0) || !bound.is_finite() {
        return true;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !len2.is_finite() {
        return true;
    }
    let dir = dir / len2.sqrt();
    let a_sin = (bound + 3.0 * view.px_max).min(1.0);
    if !a_sin.is_finite() {
        return true;
    }
    if a_sin >= 1.0 {
        return hemisphere_meets_frustum(dir, view.view_proj);
    }
    for n in side_normals(view.view_proj) {
        if n.length_squared() == 0.0 {
            continue;
        }
        if n.dot(dir) < -a_sin {
            return false;
        }
    }
    true
}

/// The open hemisphere `dot(ray, dir) > 0` meets the view if any frustum corner
/// does. The four corners are the extreme rays of the side-plane cone, and
/// `dot` is linear, so the max over the frustum is a corner. A hair of slack
/// keeps a graze the pixel test might still accept.
fn hemisphere_meets_frustum(dir: glam::Vec3, view_proj: glam::Mat4) -> bool {
    let Some(inv) = view_proj.try_inverse() else {
        return true;
    };
    for y in [-1.0f32, 1.0] {
        for x in [-1.0f32, 1.0] {
            let ray = (inv * glam::Vec4::new(x, y, 0.0, 1.0)).truncate();
            let len2 = ray.length_squared();
            if !(len2 > 0.0) || !ray.is_finite() {
                return true;
            }
            if dir.dot(ray / len2.sqrt()) > -1.0e-4 {
                return true;
            }
        }
    }
    false
}

fn rgb4(c: crate::color::LinearRgb) -> [f32; 4] {
    [c.0[0], c.0[1], c.0[2], 0.0]
}

fn pack_one(body: &FarBody) -> FarBodyGpu {
    let dir = body.dir;
    let q = body.rotation;
    let (shape, exponent) = match body.shape {
        FarShape::Cube => (0.0, 0.0),
        FarShape::Sphere => (1.0, 0.0),
        FarShape::InnerSphere => (2.0, 0.0),
        FarShape::Rounded { exponent } => (3.0, exponent),
    };
    FarBodyGpu {
        dir_rho: [dir.x, dir.y, dir.z, body.radius / body.distance],
        rot: [q.x, q.y, q.z, q.w],
        albedo0: rgb4(body.albedo[0]),
        albedo1: rgb4(body.albedo[1]),
        albedo2: rgb4(body.albedo[2]),
        albedo3: rgb4(body.albedo[3]),
        albedo4: rgb4(body.albedo[4]),
        albedo5: rgb4(body.albedo[5]),
        atmosphere: [
            body.atmosphere.0[0],
            body.atmosphere.0[1],
            body.atmosphere.0[2],
            shape,
        ],
        seed: [body.seed, exponent.to_bits(), 0, 0],
    }
}

/// 1.25 on the angular pixel bound, then two pixels of pad.
const RECT_SAFETY: f32 = 1.25;
const RECT_PAD_PX: f32 = 2.0;
/// A screen rect is not trustworthy once the cone reaches within 10° of the
/// view plane (80° from the view axis). Those cones take the angular test.
const VIEW_PLANE_LIMIT_RAD: f32 = 80.0 * std::f32::consts::PI / 180.0;
/// Sine past which the disc-on-screen rect is a poor fit (`asin(0.5) = 30°`).
/// Wider cones, including a hemisphere at sine 1, take the angular test.
const LARGE_CONE_SINE: f32 = 0.5;
/// Slack on `dot >= cos(a_body + a_tile)` so a tile on the boundary is kept.
const ANGULAR_COS_EPS: f32 = 1.0e-5;
/// Focal lengths are rotation-invariant in exact arithmetic. A relative slop
/// absorbs the ulp wobble of `|row|` as the camera turns, and is far inside
/// one pixel, so a reused tile frame stays conservative.
const FOCAL_REL_EPS: f32 = 2.0e-5;

enum TileCover {
    /// Bit set in every tile.
    All,
    /// Fully off the render extent. The frustum cull should already have dropped it.
    None,
    /// Inclusive tile coordinates. Small cones only.
    Rect {
        tx0: u32,
        ty0: u32,
        tx1: u32,
        ty1: u32,
    },
    /// Cone too wide, or too close to the view plane, for the rect.
    Angular,
}

/// Orthonormal view axes and the projection scales, read off `view_proj`.
/// `right`/`up`/`back` are world-space. A view-space direction is
/// `(right · d, up · d, back · d)`, with `back` the camera's +Z (the camera
/// looks down −Z). `focal_x` is `1 / tan(fovx/2)`, `focal_y` is `1 / tan(fovy/2)`.
struct ViewBasis {
    right: glam::Vec3,
    up: glam::Vec3,
    back: glam::Vec3,
    focal_x: f32,
    focal_y: f32,
}

impl ViewBasis {
    fn from_view_proj(view_proj: glam::Mat4) -> Option<Self> {
        let rows = view_proj.transpose();
        let right = rows.x_axis.truncate();
        let up = rows.y_axis.truncate();
        // Row 3's xyz is the forward direction (camera looks down −Z).
        let forward = rows.w_axis.truncate();
        let focal_x = right.length();
        let focal_y = up.length();
        let forward_len = forward.length();
        if !(focal_x.is_finite() && focal_y.is_finite() && forward_len.is_finite())
            || focal_x <= 1e-12
            || focal_y <= 1e-12
            || forward_len <= 1e-12
        {
            return None;
        }
        Some(Self {
            right: right / focal_x,
            up: up / focal_y,
            back: -forward / forward_len,
            focal_x,
            focal_y,
        })
    }

    fn to_view(&self, world: glam::Vec3) -> glam::Vec3 {
        glam::Vec3::new(
            self.right.dot(world),
            self.up.dot(world),
            self.back.dot(world),
        )
    }

    /// View-space ray through NDC `(x, y)`. y is up, matching the sky unproject
    /// (`inv_view_proj * (ndc, 0, 1)`).
    fn ray_ndc(&self, ndc_x: f32, ndc_y: f32) -> glam::Vec3 {
        glam::Vec3::new(ndc_x / self.focal_x, ndc_y / self.focal_y, -1.0).normalize()
    }
}

/// Framebuffer pixel (origin top-left, y down) to GL NDC (y up).
fn pixel_ndc(x: f32, y: f32, width: f32, height: f32) -> (f32, f32) {
    ((x / width) * 2.0 - 1.0, 1.0 - (y / height) * 2.0)
}

fn focals_match(a: f32, b: f32) -> bool {
    if !(a.is_finite() && b.is_finite()) || a <= 0.0 || b <= 0.0 {
        return false;
    }
    (a - b).abs() <= a.max(b) * FOCAL_REL_EPS
}

/// One tile's view-space centre ray and the sine/cosine of its angular radius
/// (max angle from the centre to the four corner rays, plus one pixel).
struct TileSample {
    centre: glam::Vec3,
    sin_r: f32,
    cos_r: f32,
}

/// View-space tile cones for one projection. The directions do not depend on
/// the camera orientation, so the ring reuses them until the projection, the
/// render extent, or the tile size changes.
struct TileFrames {
    focal_x: f32,
    focal_y: f32,
    width: u32,
    height: u32,
    tile_px: u32,
    samples: Vec<TileSample>,
}

impl TileFrames {
    fn empty() -> Self {
        Self {
            focal_x: 0.0,
            focal_y: 0.0,
            width: 0,
            height: 0,
            tile_px: 0,
            samples: Vec::new(),
        }
    }

    fn matches(&self, view: &FarView) -> bool {
        let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
        let expect = tiles_x as usize * tiles_y as usize;
        if self.width != view.width
            || self.height != view.height
            || self.tile_px != tile_px
            || self.samples.len() != expect
        {
            return false;
        }
        let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
            return false;
        };
        focals_match(self.focal_x, basis.focal_x) && focals_match(self.focal_y, basis.focal_y)
    }

    /// `true` when the samples were derived again.
    fn rebuild_if_changed(&mut self, view: &FarView) -> bool {
        if self.matches(view) {
            return false;
        }
        *self = Self::build(view).unwrap_or_else(Self::empty);
        true
    }

    fn build(view: &FarView) -> Option<Self> {
        let basis = ViewBasis::from_view_proj(view.view_proj)?;
        let (tile_px, tiles_x, tiles_y) = tile_layout(view.width, view.height);
        let width = view.width.max(1);
        let height = view.height.max(1);
        let wf = width as f32;
        let hf = height as f32;
        let ray_at = |x: f32, y: f32| {
            let (nx, ny) = pixel_ndc(x, y, wf, hf);
            basis.ray_ndc(nx, ny)
        };
        let mut samples = Vec::with_capacity(tiles_x as usize * tiles_y as usize);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                let x0 = (tx * tile_px) as f32;
                let y0 = (ty * tile_px) as f32;
                let x1 = ((tx + 1) * tile_px).min(width) as f32;
                let y1 = ((ty + 1) * tile_px).min(height) as f32;
                let centre = ray_at(0.5 * (x0 + x1), 0.5 * (y0 + y1));
                let corners = [
                    ray_at(x0, y0),
                    ray_at(x1, y0),
                    ray_at(x0, y1),
                    ray_at(x1, y1),
                ];
                let mut cos_corner = 1.0f32;
                for corner in corners {
                    cos_corner = cos_corner.min(centre.dot(corner));
                }
                // One pixel past each corner, so a ray at the pixel edge and the
                // shader's derivative step stay inside the radius.
                let mut cos_px = 1.0f32;
                let steps = [
                    (x0, y0, -1.0, 0.0),
                    (x0, y0, 0.0, -1.0),
                    (x1, y0, 1.0, 0.0),
                    (x1, y0, 0.0, -1.0),
                    (x0, y1, -1.0, 0.0),
                    (x0, y1, 0.0, 1.0),
                    (x1, y1, 1.0, 0.0),
                    (x1, y1, 0.0, 1.0),
                ];
                for (x, y, dx, dy) in steps {
                    let a = ray_at(x, y);
                    let b = ray_at(x + dx, y + dy);
                    cos_px = cos_px.min(a.dot(b));
                }
                let (sin_r, cos_r) =
                    add_angles(cos_corner.clamp(-1.0, 1.0), cos_px.clamp(-1.0, 1.0));
                samples.push(TileSample {
                    centre,
                    sin_r,
                    cos_r,
                });
            }
        }
        Some(Self {
            focal_x: basis.focal_x,
            focal_y: basis.focal_y,
            width,
            height,
            tile_px,
            samples,
        })
    }
}

/// `(sin(a+b), cos(a+b))` from the cosines of two angles in `[0, π]`. A sum
/// past π (`sin < 0`) is stored as a radius that covers every direction
/// (`sin > 1` is the flag; a real sine never is).
fn add_angles(cos_a: f32, cos_b: f32) -> (f32, f32) {
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

/// `angle(tile centre, dir) <= a_body + a_tile`, compared in cosines.
/// `sin_body` / `cos_body` are the body's drawable half-angle.
fn tile_within(tile: &TileSample, dir_view: glam::Vec3, sin_body: f32, cos_body: f32) -> bool {
    if tile.sin_r > 1.0 {
        return true;
    }
    let sin_sum = sin_body * tile.cos_r + cos_body * tile.sin_r;
    let cos_sum = cos_body * tile.cos_r - sin_body * tile.sin_r;
    // Sum past π: every direction is inside the cone.
    if sin_sum < 0.0 {
        return true;
    }
    dir_view.dot(tile.centre) + ANGULAR_COS_EPS >= cos_sum
}

/// Screen tiles touched by kept body `dir`/`bound`. `bound` is the cone sine,
/// or `-1` to keep the body everywhere. The drawable angular radius is
/// `asin(min(bound + 3 px_max, 1))` on the unit centre direction. A wide cone,
/// or one that reaches within 10° of the view plane, is [`TileCover::Angular`]:
/// the screen rect is not a bound there.
fn tile_cover(
    dir: glam::Vec3,
    bound: f32,
    view: &FarView,
    tile_px: u32,
    tiles_x: u32,
    tiles_y: u32,
) -> TileCover {
    // `VOXEL_FAR_CULL=0` forces `cone_bound` to `-1`; either signal paints every tile.
    if !far_cull_enabled() || !(bound >= 0.0) || !bound.is_finite() {
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
    // Direction through the eye: translation in `view_proj` drops out at w = 0.
    // `clip.w` is cos(theta) for a unit direction (camera looks down −Z).
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
    // Pixel radius of an angular disc of radius `a` at angle `theta` from the
    // view axis. d(tan)/dθ = sec², largest at the outer edge. Per-axis focal
    // length, then 1.25 and 2 px.
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
    // Negative viewport: NDC y-up maps to framebuffer y-down, matching SV_Position.
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

/// Fill `header.yzw` and the live mask prefix. No view leaves the tile header
/// at zero; the shader then walks every kept body. `cached` is the ring's
/// view-space tile frames when they still match this projection.
fn stamp_tiles(table: &mut FarTableGpu, view: Option<&FarView>, cached: Option<&TileFrames>) {
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
    let mut angular = [(0u32, glam::Vec3::ZERO, 0.0f32); MAX_FAR_BODIES];
    let mut n_angular = 0usize;
    for k in 0..kept {
        let bit = 1u32 << k;
        let cone = table.cone[k];
        let dir = glam::Vec3::new(cone[0], cone[1], cone[2]);
        match tile_cover(dir, cone[3], view, tile_px, tiles_x, tiles_y) {
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
                angular[n_angular] = (bit, dir, cone[3]);
                n_angular += 1;
            }
        }
    }
    if n_angular > 0 {
        paint_angular(table, view, cached, &angular[..n_angular], &mut blanket);
    }
    if blanket != 0 {
        for mask in &mut table.tile_mask[..tile_count] {
            *mask |= blanket;
        }
    }
}

/// Set bit `k` on each tile whose view-space cone meets the body's.
/// `a_body = asin(min(bound + 3 px_max, 1))` (bound 1 is 90°). A missing frame
/// cache is built for this view; a projection that yields no basis paints the
/// body into every tile.
fn paint_angular(
    table: &mut FarTableGpu,
    view: &FarView,
    cached: Option<&TileFrames>,
    angular: &[(u32, glam::Vec3, f32)],
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
        for (bit, _, _) in angular {
            *blanket |= *bit;
        }
        return;
    };
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        for (bit, _, _) in angular {
            *blanket |= *bit;
        }
        return;
    };
    let n = frames.samples.len().min(table.tile_mask.len());
    for &(bit, dir, bound) in angular {
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
        let sin_body = (bound + 3.0 * view.px_max).clamp(0.0, 1.0);
        if !sin_body.is_finite() {
            *blanket |= bit;
            continue;
        }
        let cos_body = (1.0 - sin_body * sin_body).max(0.0).sqrt();
        for (i, tile) in frames.samples.iter().enumerate().take(n) {
            if tile_within(tile, dir_view, sin_body, cos_body) {
                table.tile_mask[i] |= bit;
            }
        }
    }
}

/// Live mask words (`tiles_x * tiles_y`), capped at the array.
fn used_tiles(table: &FarTableGpu) -> usize {
    let n = (table.header[2] as u64).saturating_mul(table.header[3] as u64);
    n.min(MAX_FAR_TILES as u64) as usize
}

/// Bytes the GPU reads this frame: the body table plus one mask per live tile.
/// The unused mask tail is not part of the match and is not rewritten.
fn table_bytes(table: &FarTableGpu) -> &[u8] {
    let len = std::mem::offset_of!(FarTableGpu, tile_mask)
        + used_tiles(table) * std::mem::size_of::<u32>();
    &bytemuck::bytes_of(table)[..len]
}

fn nonzero_tiles(table: &FarTableGpu) -> u64 {
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| **mask != 0)
        .count() as u64
}

/// Pack `bodies` in order. With culling on and a view, bodies whose cone cannot
/// meet the frustum are dropped; the rest stay in their original relative order
/// (the sky composite is order-dependent). `None` keeps every body. Each kept
/// body then sets its bit in the screen tiles its drawable cone can reach.
#[cfg(test)]
pub(crate) fn pack_table(bodies: &[FarBody], view: Option<FarView>) -> FarTableGpu {
    pack_table_cached(bodies, view.as_ref(), None)
}

fn pack_table_cached(
    bodies: &[FarBody],
    view: Option<&FarView>,
    frames: Option<&TileFrames>,
) -> FarTableGpu {
    let mut table = FarTableGpu::zeroed();
    let n = bodies.len().min(MAX_FAR_BODIES);
    let cull = far_cull_enabled();
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        let bound = cone_bound(body);
        if cull {
            if let Some(view) = view {
                if !cone_in_view(body.dir, bound, view) {
                    continue;
                }
            }
        }
        let d = body.dir;
        table.cone[kept] = [d.x, d.y, d.z, bound];
        table.body[kept] = pack_one(body);
        kept += 1;
    }
    table.header[0] = kept as u32;
    stamp_tiles(&mut table, view, frames);
    table
}

/// Per-slot far-body SSBO. Identical bytes skip the map write.
/// `tiles` is the view-space tile cones, rebuilt when the projection, the
/// render extent, or the tile size changes and reused across camera turns.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    last: PerSlot<Option<FarTableGpu>>,
    tiles: TileFrames,
}

impl FarBodyRing {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let size = std::mem::size_of::<FarTableGpu>() as u64;
        let make = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER);
            unsafe { b.maintain(instance, device, physical, size) };
            b
        };
        Self {
            bufs: PerSlot::new(std::array::from_fn(|_| make())),
            last: PerSlot::new(std::array::from_fn(|_| None)),
            tiles: TileFrames::empty(),
        }
    }

    pub(crate) fn write(&mut self, slot: FrameSlot, bodies: &[FarBody], view: Option<FarView>) {
        if let Some(view) = view.as_ref() {
            self.tiles.rebuild_if_changed(view);
        }
        let table = pack_table_cached(bodies, view.as_ref(), Some(&self.tiles));
        let offered = bodies.len().min(MAX_FAR_BODIES) as u64;
        crate::profile::gauge(crate::profile::Gauge::FarBodies, offered);
        crate::profile::gauge(crate::profile::Gauge::FarDrawn, u64::from(table.header[0]));
        crate::profile::gauge(crate::profile::Gauge::FarTiles, nonzero_tiles(&table));
        let bytes = table_bytes(&table);
        if self.last[slot]
            .as_ref()
            .is_some_and(|prev| table_bytes(prev) == bytes)
        {
            return;
        }
        unsafe { self.bufs[slot].write(0, bytes) };
        self.last[slot] = Some(table);
    }

    pub(crate) fn buffer(&self, slot: FrameSlot) -> vk::Buffer {
        self.bufs[slot]
            .bound()
            .expect("the far-body table is written before the sky pass binds it")
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        for buf in self.bufs.iter_mut() {
            unsafe { buf.destroy(device) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{Camera3D, Lens, WarpMap, WarpStrength};
    use crate::color::LinearRgb;
    use crate::far_body::{
        FarBody, FarShape, ray_cube, ray_inner_sphere, ray_rounded, ray_sphere, store,
    };
    use glam::{Quat, Vec3, Vec4};

    #[test]
    fn packs_a_billion_block_sphere_exactly() {
        let body = FarBody {
            dir: Vec3::new(0.0, 0.0, 4.0),
            distance: 1.0e9,
            radius: 2.5e8,
            shape: FarShape::Sphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.3, 0.4]); 6],
            atmosphere: LinearRgb([0.0, 0.1, 0.0]),
            seed: 0xA11CE,
        };
        let mut slot = [FarBody::default(); MAX_FAR_BODIES];
        let n = store(std::slice::from_ref(&body), &mut slot);
        assert_eq!(n, 1);
        let table = pack_table(&slot[..n as usize], None);
        assert_eq!(table.header[0], 1);
        assert_eq!(table.header[1], 0);
        let gpu = &table.body[0];
        assert_eq!(gpu.dir_rho[0].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.dir_rho[1].to_bits(), 0.0f32.to_bits());
        assert_eq!(gpu.dir_rho[2].to_bits(), 1.0f32.to_bits());
        assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
        assert_eq!(gpu.rot, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(gpu.seed[0], 0xA11CE);
        assert_eq!(gpu.atmosphere[3].to_bits(), 1.0f32.to_bits());
        assert_eq!(gpu.albedo0[0].to_bits(), 0.2f32.to_bits());

        let mut cube = body;
        cube.shape = FarShape::Cube;
        cube.distance = 4.0;
        cube.radius = 1.0;
        let packed = pack_one(&cube);
        assert_eq!(packed.atmosphere[3].to_bits(), 0.0f32.to_bits());
        assert_eq!(packed.dir_rho[3].to_bits(), 0.25f32.to_bits());

        let mut wall = body;
        wall.shape = FarShape::InnerSphere;
        wall.dir = Vec3::Z;
        wall.distance = 2.0;
        wall.radius = 5.0;
        let packed = pack_one(&wall);
        assert_eq!(packed.atmosphere[3].to_bits(), 2.0f32.to_bits());
        assert_eq!(packed.dir_rho[3].to_bits(), 2.5f32.to_bits());
        assert_eq!(packed.dir_rho[2].to_bits(), 1.0f32.to_bits());
        assert_eq!(packed.seed[1], 0);
    }

    #[test]
    fn packs_rounded_exponent_round_trip() {
        let body = FarBody {
            dir: Vec3::new(0.0, 0.0, 8.0),
            distance: 8.0,
            radius: 2.0,
            shape: FarShape::Rounded { exponent: 2.17 },
            rotation: Quat::from_xyzw(0.0, 1.0, 0.0, 0.0),
            albedo: [LinearRgb([0.4, 0.5, 0.6]); 6],
            atmosphere: LinearRgb([0.1, 0.2, 0.3]),
            seed: 0xB0D1,
        };
        let mut slot = [FarBody::default(); MAX_FAR_BODIES];
        let n = store(std::slice::from_ref(&body), &mut slot);
        assert_eq!(n, 1);
        assert_eq!(slot[0].shape, FarShape::Rounded { exponent: 2.17 });
        let table = pack_table(&slot[..n as usize], None);
        let gpu = &table.body[0];
        assert_eq!(gpu.atmosphere[3].to_bits(), 3.0f32.to_bits());
        assert_eq!(f32::from_bits(gpu.seed[1]).to_bits(), 2.17f32.to_bits());
        assert_eq!(gpu.seed[0], 0xB0D1);
        assert_eq!(gpu.seed[2], 0);
        assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
        assert_eq!(gpu.rot, [0.0, 1.0, 0.0, 0.0]);
        assert_eq!(std::mem::size_of::<FarBodyGpu>(), 160);

        let again = f32::from_bits(pack_one(&slot[0]).seed[1]);
        assert_eq!(again.to_bits(), 2.17f32.to_bits());
    }

    fn sample(shape: FarShape, distance: f32, radius: f32) -> FarBody {
        FarBody {
            dir: Vec3::Z,
            distance,
            radius,
            shape,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.3, 0.4]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 1,
        }
    }

    #[test]
    fn cone_bound_matches_what_the_shader_can_draw() {
        let sphere = sample(FarShape::Sphere, 1.0, 0.25);
        let table = pack_table(std::slice::from_ref(&sphere), None);
        assert_eq!(table.header[0], 1);
        assert_eq!(table.cone[0][0].to_bits(), 0.0f32.to_bits());
        assert_eq!(table.cone[0][1].to_bits(), 0.0f32.to_bits());
        assert_eq!(table.cone[0][2].to_bits(), 1.0f32.to_bits());
        assert_eq!(table.cone[0][3].to_bits(), (1.05f32 * 0.25).to_bits());
        assert_eq!(table.cone[0][0..3], table.body[0].dir_rho[0..3]);

        // 1.05 * rho < 1 stays the rim sine, even past the old 0.99 sentinel.
        let near = sample(FarShape::Sphere, 1.0, 0.95);
        let near_bound = 1.05f32 * 0.95;
        assert!(near_bound > 0.99 && near_bound < 1.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&near), None).cone[0][3].to_bits(),
            near_bound.to_bits()
        );
        // 1.05 * rho >= 1 is the facing hemisphere, never the sentinel.
        let home = sample(FarShape::Sphere, 1.0, 0.99999);
        assert!(1.05 * 0.99999 >= 1.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&home), None).cone[0][3].to_bits(),
            1.0f32.to_bits()
        );
        let over = sample(FarShape::Sphere, 1.0, 0.96);
        assert!(1.05 * 0.96 >= 1.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&over), None).cone[0][3].to_bits(),
            1.0f32.to_bits()
        );

        let cube = sample(FarShape::Cube, 1.0, 0.2);
        let cube_bound = 3.0f32.sqrt() * 1.035 * 0.2;
        assert!(cube_bound < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&cube), None).cone[0][3].to_bits(),
            cube_bound.to_bits()
        );
        // Inside the corner sphere the cube surrounds the camera: sentinel.
        let fat = sample(FarShape::Cube, 1.0, 0.6);
        let fat_bound = 3.0f32.sqrt() * 1.035 * 0.6;
        assert!(fat_bound >= 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&fat), None).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        let rounded = sample(FarShape::Rounded { exponent: 4.0 }, 1.0, 0.2);
        let p = 4.0f32;
        let rho_b = 0.2 * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
        assert_eq!(
            pack_table(std::slice::from_ref(&rounded), None).cone[0][3].to_bits(),
            (1.05 * rho_b).to_bits()
        );

        // The shader clamps the exponent to 32; the cone uses that same p.
        let steep = sample(FarShape::Rounded { exponent: 80.0 }, 1.0, 0.1);
        let p32 = 32.0f32;
        let rho_b32 = 0.1 * 3.0f32.powf(0.5 - 1.0 / p32) * (1.0 + 2.0e-4);
        assert!(1.05 * rho_b32 < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&steep), None).cone[0][3].to_bits(),
            (1.05 * rho_b32).to_bits()
        );

        let wall = sample(FarShape::InnerSphere, 2.0, 5.0);
        assert_eq!(
            pack_table(std::slice::from_ref(&wall), None).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        assert_eq!(std::mem::offset_of!(FarTableGpu, cone), 16);
        assert_eq!(
            std::mem::offset_of!(FarTableGpu, body),
            16 + 16 * MAX_FAR_BODIES
        );
    }

    fn view_along_neg_z(fovy: f32, width: u32, height: u32) -> FarView {
        view_pitched(0.0, fovy, width, height)
    }

    /// `pitch_deg` is elevation from the horizon: 0 looks along −Z, positive looks up.
    /// The eye is not at the origin, so a direction test that forgets the view
    /// rotation fails.
    fn view_pitched(pitch_deg: f32, fovy: f32, width: u32, height: u32) -> FarView {
        let pitch = pitch_deg.to_radians();
        let forward = Vec3::new(0.0, pitch.sin(), -pitch.cos());
        let up = if forward.cross(Vec3::Y).length_squared() < 1e-6 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let position = Vec3::new(12.0, 3.0, -4.0);
        let cam = Camera3D {
            position,
            target: position + forward,
            up,
            fovy,
            lens: Lens::Rectilinear,
        };
        let aspect = width as f32 / height as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        far_view(tan_half, cam.view_proj(aspect), width, height)
    }

    fn placed(dir: Vec3, rho: f32, shape: FarShape, seed: u32) -> FarBody {
        FarBody {
            dir,
            distance: 1.0,
            radius: rho,
            shape,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.1, 0.0, 0.0]),
            seed,
        }
    }

    fn kept_seeds(table: &FarTableGpu) -> Vec<u32> {
        (0..table.header[0] as usize)
            .map(|i| table.body[i].seed[0])
            .collect()
    }

    #[test]
    fn px_max_is_twice_the_larger_pixel_angle() {
        let fovy = 60.0f32;
        let w = 3440u32;
        let h = 1440u32;
        let aspect = w as f32 / h as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        let cam = Camera3D {
            position: Vec3::ZERO,
            target: -Vec3::Z,
            up: Vec3::Y,
            fovy,
            lens: Lens::Rectilinear,
        };
        let got = px_max(tan_half, cam.view_proj(aspect), w, h);
        let vert = 2.0 * tan_half / h as f32;
        let horiz = 2.0 * tan_half * aspect / w as f32;
        assert!((vert - horiz).abs() < 1e-5, "{vert} vs {horiz}");
        assert!((got - vert * 2.0).abs() < 1e-5 * got, "{got}");

        let moved = Camera3D {
            position: Vec3::new(80.0, -3.0, 12.0),
            target: Vec3::new(80.0, -3.0, 11.0),
            ..cam
        };
        let got_moved = px_max(tan_half, moved.view_proj(aspect), w, h);
        assert!((got - got_moved).abs() < 1e-4 * got);

        let lens = Lens::WideFov {
            strength: WarpStrength::new(1.5).unwrap(),
        };
        let wide_aspect = aspect * WarpMap::from_lens(lens).fov_scale();
        let wide_cam = Camera3D { lens, ..cam };
        let wide = px_max(tan_half, wide_cam.view_proj(wide_aspect), w, h);
        let wide_horiz = 2.0 * tan_half * wide_aspect / w as f32;
        assert!(wide_horiz > vert * 1.2);
        assert!(
            (wide - wide_horiz * 2.0).abs() < 1e-4 * wide,
            "{wide} vs {wide_horiz}"
        );
    }

    #[test]
    fn frustum_keeps_ahead_and_a_crossing_rim_and_drops_the_rest() {
        let view = view_along_neg_z(60.0, 1920, 1080);
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let back = -forward;
        let ahead = placed(forward, 0.05, FarShape::Sphere, 1);
        let behind = placed(back, 0.05, FarShape::Sphere, 2);

        let right = side_normals(view.view_proj)[1];
        assert!(
            right.dot(forward) > 0.0,
            "forward must sit inside the right plane"
        );
        let on_plane = (forward - right * forward.dot(right)).normalize();
        assert!(right.dot(on_plane).abs() < 1e-4);
        let bound = 1.05 * 0.02;
        let a_sin = (bound + 3.0 * view.px_max).min(1.0);
        let outward = -right;
        let beta_in = (a_sin * 0.5).asin();
        let crossing_dir = (on_plane * beta_in.cos() + outward * beta_in.sin()).normalize();
        assert!(right.dot(crossing_dir) < 0.0);
        assert!(right.dot(crossing_dir) >= -a_sin);
        let crossing = placed(crossing_dir, 0.02, FarShape::Sphere, 3);

        let beta_out = (a_sin + 0.08).min(0.95).asin();
        let outside_dir = (on_plane * beta_out.cos() + outward * beta_out.sin()).normalize();
        assert!(right.dot(outside_dir) < -a_sin);
        let outside = placed(outside_dir, 0.02, FarShape::Sphere, 4);

        let table = pack_table(&[behind, ahead, outside, crossing], Some(view));
        assert_eq!(kept_seeds(&table), vec![1, 3]);
        assert_eq!(table.header[0], 2);
    }

    #[test]
    fn hemisphere_cone_is_kept_only_when_it_meets_the_frustum() {
        let ahead = view_pitched(0.0, 60.0, 1920, 1080);
        // Bound 1 is the facing hemisphere. Straight behind misses a 60° frustum.
        let behind = placed(Vec3::Z, 0.99999, FarShape::Sphere, 1);
        let forward = placed(-Vec3::Z, 0.99999, FarShape::Sphere, 2);
        // Pure +X: the right half of the view is inside the hemisphere.
        let side = placed(Vec3::X, 0.99999, FarShape::Sphere, 3);
        let table = pack_table(&[behind, forward, side], Some(ahead));
        assert_eq!(kept_seeds(&table), vec![2, 3]);
        assert_eq!(table.cone[0][3].to_bits(), 1.0f32.to_bits());
        assert_eq!(table.cone[1][3].to_bits(), 1.0f32.to_bits());

        // Looking steeply up, the whole frustum is above the horizon, so the
        // planet's lower hemisphere does not meet it.
        let up = view_pitched(70.0, 60.0, 1280, 720);
        let planet = placed(-Vec3::Y, 0.99999, FarShape::Sphere, 4);
        let dropped = pack_table(std::slice::from_ref(&planet), Some(up));
        assert_eq!(dropped.header[0], 0);

        // Same planet, looking down: the hemisphere covers the view.
        let down = view_pitched(-70.0, 60.0, 1280, 720);
        let kept = pack_table(std::slice::from_ref(&planet), Some(down));
        assert_eq!(kept.header[0], 1);
        assert_eq!(kept.cone[0][3].to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn inner_sphere_and_near_camera_rounded_are_always_kept() {
        let view = view_along_neg_z(50.0, 1280, 720);
        let back = Vec3::new(0.0, 0.0, 1.0);
        let wall = FarBody {
            dir: back,
            distance: 2.0,
            radius: 6.0,
            shape: FarShape::InnerSphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 7,
        };
        let rho = 0.96f32;
        let p = 2.0f32;
        let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
        assert!(1.05 * rho_b >= 0.99);
        let round = placed(back, rho, FarShape::Rounded { exponent: p }, 8);
        let table = pack_table(&[wall, round], Some(view));
        assert_eq!(table.header[0], 2);
        assert_eq!(kept_seeds(&table), vec![7, 8]);
        assert_eq!(table.cone[0][3].to_bits(), (-1.0f32).to_bits());
        assert_eq!(table.cone[1][3].to_bits(), (-1.0f32).to_bits());
    }

    #[test]
    fn compaction_preserves_relative_order_and_the_header_count() {
        let view = view_along_neg_z(60.0, 800, 600);
        let forward = Vec3::new(0.0, 0.0, -1.0);
        let back = -forward;
        let wall = FarBody {
            dir: back,
            distance: 1.5,
            radius: 4.0,
            shape: FarShape::InnerSphere,
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.0, 0.0, 0.0]),
            seed: 13,
        };
        let bodies = [
            placed(back, 0.04, FarShape::Sphere, 10),
            placed(forward, 0.04, FarShape::Cube, 11),
            placed(back, 0.05, FarShape::Sphere, 12),
            wall,
            placed(forward, 0.03, FarShape::Rounded { exponent: 6.0 }, 14),
        ];
        let table = pack_table(&bodies, Some(view));
        assert_eq!(table.header[0], 3);
        assert_eq!(kept_seeds(&table), vec![11, 13, 14]);
        assert_eq!(table.cone[0][2].to_bits(), forward.z.to_bits());
        assert_eq!(table.body[0].dir_rho[2].to_bits(), forward.z.to_bits());
        assert_eq!(table.cone[2][2].to_bits(), forward.z.to_bits());
    }

    struct Rng(u32);

    impl Rng {
        fn next(&mut self) -> u32 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            self.0 = if x == 0 { 0xA5A5_A5A5 } else { x };
            self.0
        }

        fn unit_f32(&mut self) -> f32 {
            (self.next() >> 8) as f32 * (1.0 / 16_777_216.0)
        }

        fn range(&mut self, a: f32, b: f32) -> f32 {
            a + (b - a) * self.unit_f32()
        }

        fn unit_vec(&mut self) -> Vec3 {
            let z = self.range(-1.0, 1.0);
            let t = self.range(0.0, std::f32::consts::TAU);
            let r = (1.0 - z * z).max(0.0).sqrt();
            Vec3::new(r * t.cos(), r * t.sin(), z)
        }

        fn quat(&mut self) -> Quat {
            Quat::from_axis_angle(self.unit_vec(), self.range(0.0, std::f32::consts::TAU))
        }
    }

    fn random_body(rng: &mut Rng, seed: u32) -> FarBody {
        let dir = rng.unit_vec();
        let (shape, distance, radius) = match rng.next() % 4 {
            0 => (FarShape::Sphere, 1.0, rng.range(0.002, 0.9)),
            1 => (FarShape::Cube, 1.0, rng.range(0.002, 0.9)),
            2 => (
                FarShape::Rounded {
                    exponent: rng.range(2.0, 32.0),
                },
                1.0,
                rng.range(0.002, 0.85),
            ),
            _ => (FarShape::InnerSphere, 2.0, 5.0),
        };
        FarBody {
            dir,
            distance,
            radius,
            shape,
            rotation: rng.quat(),
            albedo: [LinearRgb([0.4, 0.3, 0.2]); 6],
            atmosphere: LinearRgb([0.05, 0.08, 0.1]),
            seed,
        }
    }

    fn pixel_rejects(ray: Vec3, dir: Vec3, bound: f32, px: f32) -> bool {
        if bound < 0.0 {
            return false;
        }
        let dir = dir.normalize();
        let facing = ray.dot(dir);
        let s2 = ray.cross(dir).length_squared();
        let lim = bound + 3.0 * px;
        facing <= 0.0 || s2 > lim * lim
    }

    fn solid_hit(ray: Vec3, body: &FarBody) -> bool {
        let dir = body.dir.normalize();
        let rho = body.radius / body.distance;
        match body.shape {
            FarShape::Sphere => ray_sphere(ray, dir, rho).is_some(),
            FarShape::Cube => ray_cube(ray, dir, rho, body.rotation).is_some(),
            FarShape::Rounded { exponent } => {
                ray_rounded(ray, dir, rho, body.rotation, exponent).is_some()
            }
            FarShape::InnerSphere => {
                ray_inner_sphere(ray, dir, body.distance, body.radius).is_some()
            }
        }
    }

    /// Front contribution only. `facing <= 0` is past any rim the shader keeps
    /// after the cone reject (the rounded antipodal ghost is not a real rim).
    fn beyond_rim(ray: Vec3, body: &FarBody, px: f32) -> bool {
        let dir = body.dir.normalize();
        let rho = body.radius / body.distance;
        let facing = ray.dot(dir);
        let s = ray.cross(dir).length();
        match body.shape {
            FarShape::Sphere => {
                let rim = rho + (0.05 * rho).max(1.25 * px);
                facing <= 0.0 || s >= rim
            }
            FarShape::Rounded { exponent } => {
                let p = exponent.clamp(2.0, 32.0);
                let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p);
                let rim = rho_b + (0.05 * rho_b).max(1.25 * px);
                facing <= 0.0 || s >= rim
            }
            FarShape::Cube => {
                if facing <= 0.0 {
                    return true;
                }
                let rho_rim = rho * 1.035 + 0.5 * px;
                if !(rho_rim > 0.0 && rho_rim < 1.0) {
                    return true;
                }
                ray_cube(ray, dir, rho_rim, body.rotation).is_none()
            }
            FarShape::InnerSphere => false,
        }
    }

    #[test]
    fn cull_and_cone_reject_only_drop_rays_with_no_contribution() {
        let mut rng = Rng(0xF4A5_C011);
        let width = 3440u32;
        let height = 1440u32;
        let fovy = 70.0f32;
        let cam = Camera3D {
            position: Vec3::new(4.0, -2.0, 9.0),
            target: Vec3::new(4.0, -2.0, 8.0),
            up: Vec3::Y,
            fovy,
            lens: Lens::Rectilinear,
        };
        let aspect = width as f32 / height as f32;
        let tan_half = (fovy.to_radians() * 0.5).tan();
        let view_proj = cam.view_proj(aspect);
        let view = far_view(tan_half, view_proj, width, height);
        let vert = 2.0 * tan_half / height as f32;
        let row0 = view_proj.transpose().x_axis.truncate().length();
        let horiz = 2.0 / (row0 * width as f32);
        let px_values = [vert, horiz, vert.hypot(horiz)];

        let inv = view_proj.inverse();
        let mut rays = Vec::new();
        let push_ndc = |rays: &mut Vec<Vec3>, x: f32, y: f32| {
            let ray = (inv * Vec4::new(x, y, 0.0, 1.0)).truncate().normalize();
            let inside = side_normals(view_proj)
                .iter()
                .all(|n| n.length_squared() == 0.0 || n.dot(ray) >= -1e-4);
            if inside {
                rays.push(ray);
            }
        };
        push_ndc(&mut rays, 0.0, 0.0);
        for _ in 0..48 {
            push_ndc(&mut rays, rng.range(-1.0, 1.0), rng.range(-1.0, 1.0));
        }
        assert!(rays.len() >= 30, "rays inside the frustum: {}", rays.len());

        // The table keeps at most MAX_FAR_BODIES. Anything past that is omitted
        // by capacity, not by the frustum test, so each pack stays within the cap.
        for _batch in 0..3 {
            let bodies: Vec<FarBody> = (0..MAX_FAR_BODIES)
                .map(|i| random_body(&mut rng, i as u32))
                .collect();
            let table = pack_table(&bodies, Some(view));
            let kept = kept_seeds(&table);

            for body in &bodies {
                let culled = !kept.contains(&body.seed);
                let bound = cone_bound(body);
                for ray in &rays {
                    for px in px_values {
                        if !(culled || pixel_rejects(*ray, body.dir, bound, px)) {
                            continue;
                        }
                        let hit = solid_hit(*ray, body);
                        let past = matches!(body.shape, FarShape::InnerSphere)
                            || beyond_rim(*ray, body, px);
                        let dir = body.dir.normalize();
                        assert!(
                            !hit && past,
                            "seed {} {:?} culled {culled} bound {bound} px {px} hit {hit} past {past} facing {} s {}",
                            body.seed,
                            body.shape,
                            ray.dot(dir),
                            ray.cross(dir).length(),
                        );
                    }
                }
            }
        }
    }

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

    fn ray_at(inv: &glam::Mat4, w: u32, h: u32, x: f32, y: f32) -> Vec3 {
        let ndc_x = (x / w as f32) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / h as f32) * 2.0;
        (*inv * Vec4::new(ndc_x, ndc_y, 0.0, 1.0))
            .truncate()
            .normalize()
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

    fn tile_at(x: f32, y: f32, tile: u32, tiles_x: u32) -> usize {
        let tx = (x.max(0.0) as u32) / tile;
        let ty = (y.max(0.0) as u32) / tile;
        (ty * tiles_x + tx) as usize
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
    fn shader_tile_mask_length_matches_the_host_cap() {
        let src = include_str!("../../shaders/far_body.slang");
        let needle = format!("uint tile_mask[{MAX_FAR_TILES}]");
        assert!(
            src.contains(&needle),
            "shader tile_mask length drifted from {MAX_FAR_TILES}"
        );
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
    fn identical_used_prefix_ignores_the_mask_tail() {
        let view = view_along_neg_z(60.0, 800, 600);
        let body = placed(-Vec3::Z, 0.05, FarShape::Sphere, 1);
        let mut a = pack_table(std::slice::from_ref(&body), Some(view));
        let b = a;
        let n = used_tiles(&a);
        assert!(n > 0 && n < MAX_FAR_TILES);
        assert_eq!(table_bytes(&a), table_bytes(&b));
        a.tile_mask[n] = 0xFFFF_FFFF;
        assert_eq!(table_bytes(&a), table_bytes(&b));
        a.tile_mask[0] ^= 1;
        assert_ne!(table_bytes(&a), table_bytes(&b));
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
    fn view_space_tile_ray_matches_the_sky_unproject() {
        let view = view_pitched(25.0, 75.0, 800, 600);
        let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
        let inv = view.view_proj.inverse();
        for &(x, y) in &[
            (0.0, 0.0),
            (400.0, 300.0),
            (799.0, 599.0),
            (10.5, 20.5),
            (0.0, 599.0),
        ] {
            let (nx, ny) = pixel_ndc(x, y, 800.0, 600.0);
            let view_ray = basis.ray_ndc(nx, ny);
            let world = ray_at(&inv, 800, 600, x, y);
            let from_world = basis.to_view(world).normalize();
            assert!(
                (view_ray - from_world).length() < 1e-4,
                "pixel {x},{y} analytic {view_ray} unproject {from_world}"
            );
        }
        // The centre of the framebuffer looks along the camera forward, view −Z.
        let centre = basis.ray_ndc(0.0, 0.0);
        assert!(
            (centre - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5,
            "centre ray {centre}"
        );
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
        let from_cache = pack_table_cached(std::slice::from_ref(&planet), Some(&up), Some(&cache));
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
}
