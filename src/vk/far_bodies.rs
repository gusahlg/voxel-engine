//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_table.slang` (160 bytes, std430).
//! A 16-byte cone sits in front of each record so a pixel can reject the body
//! before loading it. A tile mask follows the records: one bit per kept body,
//! one word per screen tile, so a sky pixel skips bodies that miss its tile.
//! After the mask, compact tile-index lists feed the instanced tile-quad
//! vertex shader: mask == 0, then sphere-only tiles, then tiles that meet a
//! cube, rounded or mapped body. When a coarse-shading query is passed, the
//! base run is stably split into tiles the sun and moon discs miss, then the
//! rest, and the heavy run is split into mapped-interior tiles, then the rest.
//! A mapped-interior tile lies inside the horizon disc or, with the eye
//! outside the lo sphere, inside that sphere's disc. That prefix is drawn
//! at 2×2. The edge band stays at full rate. Both use the fixed-point march.
//! Two azimuthal horizon tables follow the body records. Each is 256 sines
//! of elevation around the local up of a Mapped body the eye is inside the
//! hi+air ball of. A tile whose cone sits above its table loses that body's
//! bit; the shader skips a ray above the same table.
//! `list_header.x` stays the whole base count; the shader still indexes by
//! instance id.

use ash::vk;
use bytemuck::{Pod, Zeroable};

use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};
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
    /// rgb rim / air tint. w = shape: 0 cube, 1 sphere, 2 inner sphere,
    /// 3 rounded, 4 mapped.
    pub atmosphere: [f32; 4],
    /// x = noise seed (unused for mapped). y = rounded exponent as f32 bits
    /// (0 otherwise). z = map id + 1 (0 = not mapped). w = 0.
    ///
    /// `albedo2.w` is the body's world distance for every shape. The sky
    /// shader orders hits by `t * distance` and rims and point blobs by
    /// `facing * distance`, and keeps the normalised `t` for shading.
    /// Mapped spare lanes: `albedo0.w` = horizon, `albedo1.w` = air thickness
    /// in radius units. The other `.w` lanes stay 0. `dir_rho.w` stays the
    /// reference `radius/distance`, not the datum hi radius.
    pub seed: [u32; 4],
}

/// Mask words in the GPU table. `shaders/far_body.slang` `tile_mask` is this
/// long. 8192 tiles cover 7680×4320 at 64 px; a larger render extent uses a
/// coarser tile so the count still fits.
const MAX_FAR_TILES: usize = 8192;
/// Preferred tile, in pixels. Coarsens when the grid would exceed [`MAX_FAR_TILES`].
const TILE_FINE_PX: u32 = 64;
const TILE_COARSE_PX: u32 = 128;

/// Header, cones, body records, horizon tables, then one mask word per screen tile.
/// `header[0]` is the live count. `header[1]` is the tile size in pixels,
/// `header[2]` is `tiles_x`, `header[3]` is `tiles_y` (all zero with no view).
/// `cone[i] = (dir.xyz, sine bound)` and a bound of `-1` means the pixel test
/// must not reject that body and every tile keeps bit `i`.
/// `horizon_id.x` and `.y` are the kept-body index of each azimuth table, or
/// `u32::MAX` when that table is unused. `.z` is [`HORIZON_BINS`]. `.w` is 0.
/// `horizon_sin` is table 0 then table 1: bin `b` is an upper bound on the
/// sine of elevation (above the plane perpendicular to centre→eye) of any
/// surface or air-shell point in that azimuth sector. The frame matches
/// `shaders/far_table.slang`. An unused table is filled with 1, so it never
/// rejects. `tile_mask[ty * tiles_x + tx]` bit `k` is set when kept body `k`
/// may cover that tile. Only the `tiles_x * tiles_y` prefix is live.
///
/// `list_header` is `(n_base, n_full, width, height)`. `tile_index` holds the
/// base tiles (mask == 0), then the full tiles. The base run is ascending
/// row-major unless a coarse query has stably partitioned it into the
/// disc-free prefix and the rest. `list_header.x` is still every base tile.
/// The sky vertex shader reads this tail; `far_bodies` does not.
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
    /// `.x` / `.y` kept-body index or `u32::MAX`. `.z` = [`HORIZON_BINS`].
    pub horizon_id: [u32; 4],
    /// Table 0 in `0..HORIZON_BINS`, table 1 after it.
    pub horizon_sin: [f32; HORIZON_BINS * HORIZON_TABLES],
    pub tile_mask: [u32; MAX_FAR_TILES],
    pub list_header: [u32; 4],
    pub tile_index: [u32; MAX_FAR_TILES],
}

// SAFETY: every field is plain `f32`/`u32` bits, `repr(C)`, and the size is a
// multiple of the 16-byte alignment, so the all-zero bit pattern is valid and
// there is no padding to exclude from a byte copy.
unsafe impl Zeroable for FarTableGpu {}
unsafe impl Pod for FarTableGpu {}

/// Bins in one azimuthal horizon table. Matches `horizon_sin` in
/// `shaders/far_table.slang` (two tables, back to back).
const HORIZON_BINS: usize = 256;
/// Tables published per frame. The two Mapped bodies with the largest rho.
const HORIZON_TABLES: usize = 2;
/// Added to every stored sine. A sample sitting on the bound stays inside.
const HORIZON_SIN_PAD: f32 = 1.0e-4;
/// Eye motion that keeps a cached table, as a fraction of the altitude
/// above the lo sphere.
const HORIZON_MOVE_FRAC: f32 = 0.001;
/// Bilinear 4×4 splits of a datum cell whose cap reaches the up axis.
/// Six levels take a ~1.5e6-block cell (g = 33) down to ~370 m.
const HORIZON_SPLIT_DEPTH: u32 = 6;
/// Sub-cells along one chart edge.
const HORIZON_SPLIT_N: u32 = 4;
/// Fine-chart stride of one datum cell, `SPLIT_N ^ SPLIT_DEPTH`. Every
/// split corner lands on this grid, so the per-frame walk does not call `tan`.
const HORIZON_FINE_STRIDE: u32 = 4096;
const _: () = assert!(HORIZON_FINE_STRIDE == 4u32.pow(HORIZON_SPLIT_DEPTH));
/// Extra azimuth half-width. Covers the fast `atan` (under 1e-4 rad) twice,
/// once for the bin centre and once for the half-angle.
const HORIZON_AZ_PAD: f32 = 2.0e-4;
/// Patch visits before further splits fall back to the corner-max bound.
/// The axis chain and a one-level straddle stay far under this. A 4×4 of
/// every cell would not.
const HORIZON_PATCH_BUDGET: u32 = 8192;

const _: () = assert!(std::mem::size_of::<FarBodyGpu>() == 160);
const _: () = assert!(
    std::mem::size_of::<FarTableGpu>()
        == 16
            + 16 * MAX_FAR_BODIES
            + 160 * MAX_FAR_BODIES
            + 16
            + 4 * HORIZON_BINS * HORIZON_TABLES
            + 4 * MAX_FAR_TILES
            + 16
            + 4 * MAX_FAR_TILES
);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, cone) == 16);
const _: () = assert!(std::mem::offset_of!(FarTableGpu, body) == 16 + 16 * MAX_FAR_BODIES);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, horizon_id)
        == 16 + 16 * MAX_FAR_BODIES + 160 * MAX_FAR_BODIES
);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, horizon_sin)
        == std::mem::offset_of!(FarTableGpu, horizon_id) + 16
);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, tile_mask)
        == std::mem::offset_of!(FarTableGpu, horizon_sin) + 4 * HORIZON_BINS * HORIZON_TABLES
);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, list_header)
        == std::mem::offset_of!(FarTableGpu, tile_mask) + 4 * MAX_FAR_TILES
);
const _: () = assert!(
    std::mem::offset_of!(FarTableGpu, tile_index)
        == std::mem::offset_of!(FarTableGpu, list_header) + 16
);

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
///   `hi + air/distance`: the shader sits on the local surface, which is at
///   most the hi sphere, and the shader's extra 3 px does not cover a thick
///   shell. A ray aimed away from the centre can still meet the body once
///   that sphere contains the camera. Sentinel when the widened bound is
///   `>= 0.99`; otherwise the sine is `hi + air/distance` (a negative radius
///   or air floors at 0, so it is not mistaken for the sentinel).
///
///   The sentinel is only the per-pixel reject. The tile mask uses the horizon
///   cone ([`mapped_horizon_half`]) whenever `horizon < 1`, including when this
///   bound is `-1`, so sky tiles above the limb do not take the mapped march.
///   `horizon >= 1` disables that cone and keeps today's tile coverage.
fn cone_bound(body: &FarBody, map_max: &[f32; MAX_FAR_MAPS]) -> f32 {
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
    // Every shape, not only mapped. The shader's depth compare reads this lane.
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

/// Half-angle of a mapped body's tile cone, as `(sin, cos)`.
///
/// The shader keeps a ray when `dot(ray, -dir) <= horizon`: a cone around
/// `+dir` of half-angle `π/2 + asin(horizon)`. `cos` of that angle is
/// `-horizon`, so a positive horizon (terrain above the plane normal to
/// `-dir`) is wider than a hemisphere and `cos` is negative. The air limb
/// widens the angle by `δ` with `sin δ = air/distance` (a negative thickness
/// floors at 0), then by the same 3 px sine pad as every other cone. `None`
/// paints every tile: `horizon >= 1`, a non-finite input, or a sum past π.
fn mapped_horizon_half(horizon: f32, air: f32, distance: f32, px_max: f32) -> Option<(f32, f32)> {
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
fn mapped_interior_half(horizon: f32, air: f32, distance: f32, px_max: f32) -> Option<(f32, f32)> {
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
fn lo_disc_interior(rho_lo: f32, px_margin: f32) -> Option<(f32, f32)> {
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
fn mapped_rho_lo(gpu: &FarBodyGpu, map_min: &[f32; MAX_FAR_MAPS]) -> Option<f32> {
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

/// Maximum of `dot(unit ray, dir)` on the view's direction cone.
///
/// `dir` is unit. The maximum on a convex spherical polygon is 1 when `dir`
/// lies inside, otherwise on an edge. An edge's maximum is a corner or the
/// point where the great circle passes closest to `dir`, so the four corners
/// alone miss a graze through the middle of a side. `None` keeps the body.
fn max_dir_dot_on_frustum(dir: glam::Vec3, view_proj: glam::Mat4) -> Option<f32> {
    let inv = view_proj.try_inverse()?;
    let mut corners = [glam::Vec3::ZERO; 4];
    let mut i = 0;
    for y in [-1.0f32, 1.0] {
        for x in [-1.0f32, 1.0] {
            let ray = (inv * glam::Vec4::new(x, y, 0.0, 1.0)).truncate();
            let len2 = ray.length_squared();
            if !(len2 > 0.0) || !ray.is_finite() {
                return None;
            }
            corners[i] = ray / len2.sqrt();
            i += 1;
        }
    }
    let clip = view_proj * glam::Vec4::new(dir.x, dir.y, dir.z, 0.0);
    if clip.is_finite() && clip.w > 0.0 {
        let slack = 1.0e-4 * clip.w.abs();
        if clip.x.abs() <= clip.w + slack && clip.y.abs() <= clip.w + slack {
            return Some(1.0);
        }
    }
    let mut best = f32::NEG_INFINITY;
    for corner in corners {
        best = best.max(dir.dot(corner));
    }
    // NDC order from the loops above: (-1,-1), (1,-1), (-1,1), (1,1).
    for (ia, ib) in [(0usize, 1usize), (2, 3), (0, 2), (1, 3)] {
        let (a, b) = (corners[ia], corners[ib]);
        let n = a.cross(b);
        let nlen2 = n.length_squared();
        if !(nlen2 > 1.0e-20) {
            continue;
        }
        let n = n / nlen2.sqrt();
        let proj = dir - n * dir.dot(n);
        let plen2 = proj.length_squared();
        if !(plen2 > 1.0e-20) {
            continue;
        }
        let u = proj / plen2.sqrt();
        let ab = a.cross(b);
        let on_arc = a.cross(u).dot(ab) >= -1.0e-5 && u.cross(b).dot(ab) >= -1.0e-5;
        if on_arc {
            best = best.max(dir.dot(u));
        }
    }
    best.is_finite().then_some(best)
}

/// The widened horizon cone meets the view. A missing inverse keeps the body.
fn horizon_cone_meets_frustum(dir: glam::Vec3, cos_alpha: f32, view_proj: glam::Mat4) -> bool {
    let len2 = dir.length_squared();
    if !(len2 > 0.0) || !dir.is_finite() || !cos_alpha.is_finite() {
        return true;
    }
    let dir = dir / len2.sqrt();
    match max_dir_dot_on_frustum(dir, view_proj) {
        Some(max_dot) => max_dot >= cos_alpha - 1.0e-4,
        None => true,
    }
}

/// Frustum keep. A mapped body with `horizon < 1` uses the widened horizon
/// cone, including when [`cone_bound`] is the per-pixel sentinel. `horizon >= 1`
/// and every other shape keep the pixel-cone test.
fn body_meets_view(body: &FarBody, bound: f32, view: &FarView) -> bool {
    if let FarShape::Mapped { horizon, air, .. } = body.shape {
        if horizon < 1.0 {
            return match mapped_horizon_half(horizon, air, body.distance, view.px_max) {
                None => true,
                Some((_, cos_alpha)) => {
                    horizon_cone_meets_frustum(body.dir, cos_alpha, view.view_proj)
                }
            };
        }
    }
    cone_in_view(body.dir, bound, view)
}

/// `angle(tile centre, dir) <= a_body + a_tile`, compared in cosines.
/// `sin_body` / `cos_body` are the body's drawable half-angle. `cos_body` is
/// negative when the half-angle is wider than a hemisphere.
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

/// The whole tile cone lies strictly inside the body's half-angle.
///
/// `angle(centre, dir) + a_tile < a_body`. A tile that only touches the
/// boundary stays out, so the limb and the silhouette stay at 1×1.
fn tile_strictly_inside(
    tile: &TileSample,
    dir_view: glam::Vec3,
    sin_body: f32,
    cos_body: f32,
) -> bool {
    if tile.sin_r > 1.0 || !(tile.sin_r.is_finite() && tile.cos_r.is_finite()) {
        return false;
    }
    let sin_g = sin_body * tile.cos_r - cos_body * tile.sin_r;
    let cos_g = cos_body * tile.cos_r + sin_body * tile.sin_r;
    if !(sin_g > 0.0) || !sin_g.is_finite() || !cos_g.is_finite() {
        return false;
    }
    dir_view.dot(tile.centre) > cos_g + ANGULAR_COS_EPS
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
    // `(bit, dir, sin, cos)` of the drawable half-angle. `cos` may be negative.
    let mut angular = [(0u32, glam::Vec3::ZERO, 0.0f32, 0.0f32); MAX_FAR_BODIES];
    let mut n_angular = 0usize;
    for k in 0..kept {
        let bit = 1u32 << k;
        let cone = table.cone[k];
        let dir = glam::Vec3::new(cone[0], cone[1], cone[2]);
        let gpu = &table.body[k];
        let shape = gpu.atmosphere[3];
        let horizon = gpu.albedo0[3];
        // Mapped + a real horizon: the tile cone is the horizon gate, even
        // when `cone.w` is the per-pixel sentinel. `horizon >= 1` falls
        // through to that sentinel and paints every tile.
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
        paint_angular(table, view, cached, &angular[..n_angular], &mut blanket);
    }
    if blanket != 0 {
        for mask in &mut table.tile_mask[..tile_count] {
            *mask |= blanket;
        }
    }
    fill_tile_lists(table, view.width, view.height);
}

/// Set bit `k` on each tile whose view-space cone meets the body's half-angle
/// `(sin, cos)`. `cos` is negative past 90°. A missing frame cache is built
/// for this view; a projection that yields no basis paints the body into
/// every tile.
fn paint_angular(
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

/// Live mask words (`tiles_x * tiles_y`), capped at the array.
fn used_tiles(table: &FarTableGpu) -> usize {
    let n = (table.header[2] as u64).saturating_mul(table.header[3] as u64);
    n.min(MAX_FAR_TILES as u64) as usize
}

/// Screen rect of tile `index` (`ty * tiles_x + tx`), in pixels, clamped to
/// the render extent. The right and bottom tiles are shorter when `width` or
/// `height` is not a multiple of `tile_px`. Matches `sky_tile.vert`.
#[cfg(test)]
fn tile_rect(index: u32, tile_px: u32, tiles_x: u32, width: u32, height: u32) -> [u32; 4] {
    debug_assert!(tiles_x > 0);
    let tx = index % tiles_x;
    let ty = index / tiles_x;
    let x0 = tx.saturating_mul(tile_px);
    let y0 = ty.saturating_mul(tile_px);
    let x1 = x0.saturating_add(tile_px).min(width);
    let y1 = y0.saturating_add(tile_px).min(height);
    [x0, y0, x1, y1]
}

/// Inverse of the scene pass's negative-height viewport. NDC y is up,
/// framebuffer y is down. Matches `sky_tile.vert` and the fullscreen
/// triangle's clip xy at the same sample.
#[cfg(test)]
fn framebuffer_ndc(xf: f32, yf: f32, width: u32, height: u32) -> [f32; 2] {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    [(xf / w) * 2.0 - 1.0, 1.0 - (yf / h) * 2.0]
}

/// Kept bodies whose shader is not the sphere/inner pair. Bit i is kept index i.
/// Shape 0 is the cube. An unrecognised shape stays on the full march.
fn heavy_mask(table: &FarTableGpu) -> u32 {
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

fn has_mapped(table: &FarTableGpu) -> bool {
    let n = (table.header[0] as usize).min(MAX_FAR_BODIES);
    (0..n).any(|i| {
        let shape = table.body[i].atmosphere[3];
        (3.5..4.5).contains(&shape)
    })
}

/// `(n_base, n_sphere, n_heavy)` over the live tiles. Sphere tiles have a
/// non-zero mask that misses every heavy body.
fn tile_split(table: &FarTableGpu) -> (u32, u32, u32) {
    let heavy = heavy_mask(table);
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

/// Partition the live tiles into mask == 0, then sphere-only, then heavy.
/// Each run is ascending row-major. `list_header.y` is `n_sphere + n_heavy`.
fn fill_tile_lists(table: &mut FarTableGpu, width: u32, height: u32) {
    let heavy = heavy_mask(table);
    let (n_base, n_sphere, n_heavy) = tile_split(table);
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

/// Disc and star inputs for [`split_coarse_base`]. The rims are the push
/// constant's outer cosines (`SkyParams::disc_rims`), not the solid cores.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SkyCoarseQuery {
    /// World-space sun direction, the same vector the push constant normalises.
    /// A zero or non-finite direction shades every base tile at 1×1.
    pub sun_dir: glam::Vec3,
    pub sun_cos_rim: f32,
    pub moon_cos_rim: f32,
    /// `max(night, star_floor) * stars_gain > 0` in `sky.frag`.
    pub stars: bool,
}

/// Same gate as `sky.frag`: stars draw when `max(night, star_floor) * gain`
/// is positive. A non-finite factor counts as stars on, so no tile goes coarse.
pub(crate) fn stars_drawn(night: f32, star_floor: f32, stars_gain: f32) -> bool {
    if !(night.is_finite() && star_floor.is_finite() && stars_gain.is_finite()) {
        return true;
    }
    night.max(star_floor) * stars_gain > 0.0
}

/// Widen a disc-rim cosine by `margin_rad`. `None` when the inputs are not a
/// finite cone, or the widened angle covers the sphere (every tile is touched).
fn widened_disc(cos_rim: f32, margin_rad: f32) -> Option<(f32, f32)> {
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

fn unit_dir(v: glam::Vec3) -> Option<glam::Vec3> {
    let len2 = v.length_squared();
    if !(len2 > 0.0) || !v.is_finite() {
        return None;
    }
    Some(v / len2.sqrt())
}

/// Stably partition the base prefix into tiles neither disc can touch, then
/// the rest. Sphere and heavy runs are left where [`fill_tile_lists`] put
/// them. Returns the coarse count. Stars, a missing frame, or an unusable
/// disc leave the prefix unchanged and return 0.
///
/// `px_max` is twice the larger-axis pixel angle, so the margin is 2 px on
/// top of the tile cone (which already reaches 1 px past its corners).
fn split_coarse_base(
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
    let Some((sun_sin, sun_cos)) = widened_disc(query.sun_cos_rim, view.px_max) else {
        return 0;
    };
    let Some((moon_sin, moon_cos)) = widened_disc(query.moon_cos_rim, view.px_max) else {
        return 0;
    };
    // Classify before writing, so a bad index leaves the run untouched.
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

/// Stably partition the heavy run into mapped-interior tiles, then the rest.
/// Returns the coarse count. Both runs use the fixed-point march. The prefix
/// is drawn at 2×2; the suffix, where the silhouette and the limb live, stays
/// at full rate. A tile qualifies when its mask is exactly one
/// mapped body (no other body, and so no nearer body, can cover it) and its
/// cone lies strictly inside one of:
///
/// * the horizon disc inset by the air limb and 3 px, when `horizon < 1`;
/// * the lo-sphere disc inset by 2 px, when the eye is outside that sphere.
///   The tile cone already carries the tile's angular radius, so the inset
///   is the rest of the margin.
///
/// Every pixel of that tile is a surface hit, and the hit replaces the sky
/// colour. Stars and the sun/moon discs are composited before far bodies, so
/// they do not keep the tile at 1×1 — that gate is only for empty base-sky
/// tiles ([`split_coarse_base`]). A missing frame leaves the run unchanged
/// and returns 0. `map_min[i]` is map `i`'s minimum datum offset. `query` is
/// the same value the base split uses; the interior does not read it.
fn split_coarse_far(
    table: &mut FarTableGpu,
    frames: &TileFrames,
    view: &FarView,
    _query: &SkyCoarseQuery,
    map_min: &[f32; MAX_FAR_MAPS],
) -> u32 {
    let (n_base, n_sphere, n_heavy) = tile_split(table);
    if n_heavy == 0 || !frames.matches(view) {
        return 0;
    }
    let Some(basis) = ViewBasis::from_view_proj(view.view_proj) else {
        return 0;
    };
    let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
    // Horizon-disc cone, then lo-sphere disc. Either one admits the tile.
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
        // `px_max` is already two pixel-angles, the 2 px inset.
        if let Some(rho_lo) = mapped_rho_lo(gpu, map_min)
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
    // Classify before writing, so a bad index leaves the run untouched.
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

/// Which body fragment a draw needs. `Full` has every shape. `NoMap` drops the
/// mapped march. `Sphere` keeps spheres and inner spheres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkyBodyPipe {
    Full,
    NoMap,
    Sphere,
}

/// How `record_sky` draws this frame. Tile quads only when the mask is a real
/// per-tile classification: a view, culling on, a non-zero tile size, and at
/// least one kept body. Otherwise one fullscreen triangle. No kept bodies use
/// the body-free pipeline. A frame with no mapped body uses `NoMap`, and a
/// frame of only spheres and inner spheres uses `Sphere`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SkyDraw {
    pub quads: bool,
    /// Fullscreen triangle uses the body-free pipeline.
    pub base: bool,
    /// Fullscreen body pipeline, or the pipeline for heavy tiles.
    pub body: SkyBodyPipe,
    pub n_base: u32,
    /// Tiles whose mask is only spheres and inner spheres.
    pub n_sphere: u32,
    /// Tiles whose mask meets a cube, rounded or mapped body.
    pub n_heavy: u32,
    /// `n_sphere + n_heavy`. `far.tiles`.
    pub n_full: u32,
    /// Coarse-eligible prefix of the base run. `sky.coarse`. Zero unless a
    /// query actually split the prefix.
    pub n_coarse: u32,
    /// Coarse-eligible prefix of the heavy run. `sky.coarse_far`. Zero unless
    /// a query actually split that run.
    pub n_coarse_far: u32,
}

impl Default for SkyDraw {
    fn default() -> Self {
        Self {
            quads: false,
            base: false,
            body: SkyBodyPipe::Full,
            n_base: 0,
            n_sphere: 0,
            n_heavy: 0,
            n_full: 0,
            n_coarse: 0,
            n_coarse_far: 0,
        }
    }
}

impl SkyDraw {
    fn body_pipe(table: &FarTableGpu) -> SkyBodyPipe {
        if has_mapped(table) {
            SkyBodyPipe::Full
        } else if heavy_mask(table) != 0 {
            SkyBodyPipe::NoMap
        } else {
            SkyBodyPipe::Sphere
        }
    }

    fn from_table(table: &FarTableGpu, cull: bool) -> Self {
        let bodies = table.header[0];
        let tile_px = table.header[1];
        let tiles_x = table.header[2];
        let tiles_y = table.header[3];
        let (n_base, n_sphere, n_heavy) = tile_split(table);
        let n_full = n_sphere + n_heavy;
        let tiled = cull && tile_px != 0 && tiles_x != 0 && tiles_y != 0 && bodies != 0;
        if tiled {
            Self {
                quads: true,
                base: false,
                body: Self::body_pipe(table),
                n_base,
                n_sphere,
                n_heavy,
                n_full,
                n_coarse: 0,
                n_coarse_far: 0,
            }
        } else {
            Self {
                quads: false,
                base: bodies == 0,
                body: Self::body_pipe(table),
                n_base: 0,
                n_sphere: 0,
                n_heavy: 0,
                n_full: 0,
                n_coarse: 0,
                n_coarse_far: 0,
            }
        }
    }
}

/// Bytes the GPU reads this frame: the body table plus one mask per live tile.
/// The unused mask tail is not part of the match and is not rewritten.
fn table_bytes(table: &FarTableGpu) -> &[u8] {
    let len = std::mem::offset_of!(FarTableGpu, tile_mask)
        + used_tiles(table) * std::mem::size_of::<u32>();
    &bytemuck::bytes_of(table)[..len]
}

/// `list_header` plus the live index prefix (`n_base + n_full` words).
fn list_bytes(table: &FarTableGpu) -> &[u8] {
    let start = std::mem::offset_of!(FarTableGpu, list_header);
    let n = (table.list_header[0] as usize)
        .saturating_add(table.list_header[1] as usize)
        .min(MAX_FAR_TILES);
    let len = std::mem::size_of::<[u32; 4]>() + n * std::mem::size_of::<u32>();
    &bytemuck::bytes_of(table)[start..start + len]
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
/// `map_max[i]` is map `i`'s maximum datum offset, used for a mapped body's
/// per-pixel cone. That sine is the hi radius plus `air/distance` (`-1` when
/// that reach is at least 0.99). A mapped body with `horizon < 1` paints tiles
/// from the horizon cone instead, so the sentinel does not cover the sky.
#[cfg(test)]
pub(crate) fn pack_table(
    bodies: &[FarBody],
    view: Option<FarView>,
    map_max: &[f32; MAX_FAR_MAPS],
) -> FarTableGpu {
    // The table bytes do not depend on the dip. Callers that need the sine
    // go through [`pack_table_cached`] with the real minimum offsets.
    *pack_table_cached(
        bodies,
        view.as_ref(),
        None,
        map_max,
        &[0.0; MAX_FAR_MAPS],
        glam::Vec3::Y,
    )
    .0
}

/// The table is 72 KB. `Box::new(FarTableGpu::zeroed())` would build that on
/// the stack, and the render thread already holds `Renderer` there.
fn zeroed_table() -> Box<FarTableGpu> {
    let layout = std::alloc::Layout::new::<FarTableGpu>();
    // SAFETY: `FarTableGpu` is `Zeroable`, so the zeroed allocation is a valid
    // value. The pointer is the exact layout `Box` will free.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<FarTableGpu>();
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    unsafe { Box::from_raw(ptr) }
}

/// Sine of the geometric horizon dip below the local horizontal.
///
/// The ground body is the outside body (`0 < rho < 1`) with the largest rho
/// whose centre lies under the viewer: `dot(dir, -sky_up) > 0.5`. `rho` is
/// `radius/distance`. A [`FarShape::Mapped`] body uses the lo sphere instead,
/// `(radius + map_min[map]) / distance`: the deepest geometric horizon, so
/// the sky line stays under the drawn silhouette. An id past the table, or a
/// non-finite offset, uses offset 0 (the reference radius). An empty slot is
/// already 0. `sky_up` is the same local up [`super::uniforms::local_sky_basis`]
/// uses, so a zero or non-unit up matches the sky frame. No such body yields
/// `0`. Otherwise `s = sqrt(max(1 - rho², 0))`, clamped to `[0, 0.5]`.
///
/// Called from packing on the bodies that pack keeps (the list past `keep`,
/// not the frustum survivors). Fog and water evaluate `sky_radiance` too, so
/// the dip must not pop when the ground body leaves the sky frustum.
/// `map_min[i]` is map `i`'s minimum datum offset, the same table as the
/// lo-sphere disc.
pub(crate) fn horizon_dip(
    bodies: &[FarBody],
    sky_up: glam::Vec3,
    map_min: &[f32; MAX_FAR_MAPS],
) -> f32 {
    let (_, up, _) = super::uniforms::local_sky_basis(sky_up);
    let down = -up;
    let mut best = -1.0f32;
    for body in bodies.iter().take(MAX_FAR_BODIES) {
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
        }
    }
    if !(best >= 0.0) {
        return 0.0;
    }
    (1.0 - best * best).max(0.0).sqrt().clamp(0.0, 0.5)
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
fn horizon_axes(up: glam::Vec3) -> Option<(glam::Vec3, glam::Vec3)> {
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
fn horizon_azimuth(dir: glam::Vec3, up: glam::Vec3, east: glam::Vec3, north: glam::Vec3) -> f32 {
    let horiz = dir - up * dir.dot(up);
    let az = horiz.dot(east).atan2(horiz.dot(north));
    if az < 0.0 {
        az + std::f32::consts::TAU
    } else {
        az
    }
}

fn horizon_bin(az: f32) -> usize {
    let n = HORIZON_BINS as f32;
    let i = (az / std::f32::consts::TAU * n).floor() as i32;
    i.rem_euclid(HORIZON_BINS as i32) as usize
}

/// Sine of the elevation of a point at normalised radius `rho` whose
/// centre-direction has `dot(s, up) = mu`. The eye is the origin and the
/// centre is `-up`. Highest radius is the higher elevation: the derivative
/// in `rho` is non-negative. The derivative in `mu` peaks at `mu = rho`
/// when `rho < 1`, and keeps rising up to the pole when `rho >= 1`.
fn horizon_elev_sin(rho: f32, mu: f32) -> f32 {
    let mu = mu.clamp(-1.0, 1.0);
    if !(rho > 0.0) || !rho.is_finite() {
        return -1.0;
    }
    let disc = (1.0 - rho) * (1.0 - rho) + 2.0 * rho * (1.0 - mu);
    if !(disc > 1.0e-20) {
        return 1.0;
    }
    let s = (rho * mu - 1.0) / disc.sqrt();
    if s.is_finite() {
        s.clamp(-1.0, 1.0)
    } else {
        1.0
    }
}

/// Upper bound on elevation over the cap. `mu*` is `rho` when the cap
/// straddles the limb (`mu = rho`), otherwise the endpoint closer to it.
fn cap_elev(rho: f32, mu_far: f32, mu_near: f32) -> f32 {
    let mu = if rho >= 1.0 {
        mu_near
    } else {
        rho.clamp(mu_far, mu_near)
    };
    horizon_elev_sin(rho, mu)
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

fn chart_dir(face: usize, xi: f32, eta: f32) -> Option<glam::Vec3> {
    let (tu, n, tv) = crate::far_body::far_map_basis(face);
    let a = (xi * std::f32::consts::FRAC_PI_4).tan();
    let b = (eta * std::f32::consts::FRAC_PI_4).tan();
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let d = n + tu * a + tv * b;
    unit_dir(d)
}

fn raise_bins(bins: &mut [f32; HORIZON_BINS], az: f32, half: Option<f32>, elev: f32) {
    if !elev.is_finite() {
        return;
    }
    let elev = (elev + HORIZON_SIN_PAD).clamp(-1.0, 1.0);
    let Some(half) = half else {
        for bin in bins.iter_mut() {
            *bin = bin.max(elev);
        }
        return;
    };
    let width = std::f32::consts::TAU / HORIZON_BINS as f32;
    let mut i0 = ((az - half) / width).floor() as i32;
    let i1 = ((az + half) / width).floor() as i32;
    if i1 - i0 >= HORIZON_BINS as i32 {
        for bin in bins.iter_mut() {
            *bin = bin.max(elev);
        }
        return;
    }
    while i0 <= i1 {
        let idx = i0.rem_euclid(HORIZON_BINS as i32) as usize;
        bins[idx] = bins[idx].max(elev);
        i0 += 1;
    }
}

fn horizon_bilerp(h00: f32, h10: f32, h01: f32, h11: f32, u: f32, v: f32) -> f32 {
    let a = h00 + (h10 - h00) * u;
    let b = h01 + (h11 - h01) * u;
    a + (b - a) * v
}

/// `atan` on `[0, 1]`. Degree-9 odd polynomial, max abs error under 3e-5 rad.
#[inline(always)]
fn fast_atan(z: f32) -> f32 {
    let z = z.abs();
    let u = z * z;
    let mut p = 0.024597976f32;
    p = p * u + -0.09351313;
    p = p * u + 0.18633223;
    p = p * u + -0.33197755;
    p = p * u + 0.99998576;
    z * p
}

/// `atan2` via one [`fast_atan`] on a reduced `[0, 1]` argument.
#[inline(always)]
fn fast_atan2(y: f32, x: f32) -> f32 {
    let ax = x.abs();
    let ay = y.abs();
    if !(ax + ay > 0.0) {
        return 0.0;
    }
    let a = if ax >= ay {
        fast_atan(ay / ax)
    } else {
        std::f32::consts::FRAC_PI_2 - fast_atan(ax / ay)
    };
    let a = if x < 0.0 { std::f32::consts::PI - a } else { a };
    if y < 0.0 { -a } else { a }
}

/// `asin` on `[0, 1]` via `atan2(s, sqrt(1-s²))`.
#[inline(always)]
fn fast_asin(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    let c = (1.0 - s * s).max(0.0).sqrt();
    let a = fast_atan2(s, c);
    if a < 0.0 { -a } else { a }
}

/// `(cos, sin)` of `beta + 1e-5`, where `beta = acos(min_cos)`.
///
/// The extra tenth of a milliradian keeps the f32 cos/sin identity from
/// shrinking the cap inside the true corner angle.
#[inline]
fn horizon_cap_axes(min_cos: f32) -> (f32, f32) {
    let cos_b = min_cos.clamp(-1.0, 1.0);
    let sin_b = (1.0 - cos_b * cos_b).max(0.0).sqrt();
    let d = 1.0e-5f32;
    let c = cos_b - sin_b * d;
    let s = (sin_b + cos_b * d).max(0.0);
    let len = (c * c + s * s).sqrt();
    if !(len > 0.0) {
        return (1.0, 0.0);
    }
    ((c / len).clamp(-1.0, 1.0), (s / len).clamp(0.0, 1.0))
}

/// `(mu_far, mu_near)` without `acos`. `cos_b` / `sin_b` are a cap half-angle.
#[inline(always)]
fn cap_mu_fast(mu: f32, cos_b: f32, sin_b: f32, sin_g: f32) -> (f32, f32) {
    let near_c = (mu * cos_b + sin_g * sin_b).clamp(-1.0, 1.0);
    let far_c = (mu * cos_b - sin_g * sin_b).clamp(-1.0, 1.0);
    let mu_near = if mu >= cos_b { 1.0 } else { near_c };
    let mu_far = if mu <= -cos_b { -1.0 } else { far_c };
    (mu_far.min(mu_near), mu_near)
}

/// One datum cell on the equiangular chart. Built once per `g`.
struct HorizonCellGeom {
    centre: glam::Vec3,
    cos_beta: f32,
    sin_beta: f32,
}

struct HorizonGeom {
    cells: u32,
    stride: u32,
    /// `tan(ξ π/4)` at `ξ = 2k/fine - 1`, `k = 0..=cells*stride`.
    tan_q: Vec<f32>,
    /// Face-major, then row `j`, then column `i`. `sin_beta < 0` is unusable.
    cell: Vec<HorizonCellGeom>,
}

fn build_horizon_geom(g: u32) -> HorizonGeom {
    let cells = g.max(2) - 1;
    let stride = HORIZON_FINE_STRIDE;
    let fine = cells * stride;
    let mut tan_q = Vec::with_capacity(fine as usize + 1);
    let denom = fine as f32;
    for k in 0..=fine {
        let xi = 2.0 * k as f32 / denom - 1.0;
        tan_q.push((xi * std::f32::consts::FRAC_PI_4).tan());
    }
    let mut cell = Vec::with_capacity(6 * cells as usize * cells as usize);
    for face in 0..6usize {
        let (tu, n, tv) = crate::far_body::far_map_basis(face);
        for j in 0..cells {
            for i in 0..cells {
                let ku0 = i * stride;
                let kv0 = j * stride;
                let corner = |ku: u32, kv: u32| {
                    let a = tan_q[ku as usize];
                    let b = tan_q[kv as usize];
                    unit_dir(n + tu * a + tv * b)
                };
                let c00 = corner(ku0, kv0);
                let c10 = corner(ku0 + stride, kv0);
                let c01 = corner(ku0, kv0 + stride);
                let c11 = corner(ku0 + stride, kv0 + stride);
                let centre = corner(ku0 + stride / 2, kv0 + stride / 2);
                let geom = match (centre, c00, c10, c01, c11) {
                    (Some(centre), Some(a), Some(b), Some(c), Some(d)) => {
                        let min_cos = centre
                            .dot(a)
                            .min(centre.dot(b))
                            .min(centre.dot(c))
                            .min(centre.dot(d));
                        let (cos_beta, sin_beta) = horizon_cap_axes(min_cos);
                        HorizonCellGeom {
                            centre,
                            cos_beta,
                            sin_beta,
                        }
                    }
                    _ => HorizonCellGeom {
                        centre: glam::Vec3::ZERO,
                        cos_beta: 0.0,
                        sin_beta: -1.0,
                    },
                };
                cell.push(geom);
            }
        }
    }
    HorizonGeom {
        cells,
        stride,
        tan_q,
        cell,
    }
}

fn horizon_geom(g: u32) -> Option<&'static HorizonGeom> {
    if !(2..=65).contains(&g) {
        return None;
    }
    use std::sync::OnceLock;
    static SLOTS: OnceLock<Box<[OnceLock<HorizonGeom>]>> = OnceLock::new();
    let slots = SLOTS.get_or_init(|| {
        let mut v = Vec::with_capacity(66);
        for _ in 0..66 {
            v.push(OnceLock::new());
        }
        v.into_boxed_slice()
    });
    Some(slots[g as usize].get_or_init(|| build_horizon_geom(g)))
}

/// Split a patch whose cap contains the up axis, or sits within one cap-radius
/// of it, and a patch whose terrain corners straddle the eye radius.
///
/// `cos_b` / `sin_b` are the cap half-angle. The axis cap paints every bin, so
/// it recurses to [`HORIZON_SPLIT_DEPTH`]. A cap that only comes close is split
/// twice: past that, each level multiplies the patch count by four and the
/// walk covers the globe. A far straddle is split once. Uniform terrain already
/// has its max at a corner. Splitting every cell whose shell contains the
/// camera would be the whole planet (the pixel shell is thicker than 50 km)
/// and does not fit in the frame.
fn horizon_should_split(
    depth: u32,
    mu: f32,
    cos_b: f32,
    sin_b: f32,
    radius: f32,
    min_off: f32,
    max_off: f32,
    distance: f32,
) -> bool {
    if depth >= HORIZON_SPLIT_DEPTH || !(cos_b.abs() <= 1.0) || !mu.is_finite() || sin_b < 0.0 {
        return false;
    }
    let mu = mu.clamp(-1.0, 1.0);
    // gamma <= beta + 1e-6, with cos decreasing on [0, π].
    horizon_reaches_axis(depth, mu, cos_b, sin_b)
        || horizon_straddle(depth, radius, min_off, max_off, distance)
}

/// Cap contains the up axis, or sits within one cap-radius of it.
///
/// The axis cap paints every bin, so it recurses to [`HORIZON_SPLIT_DEPTH`].
/// A cap that only comes close is split twice: past that each level multiplies
/// the patch count by four.
#[inline]
fn horizon_reaches_axis(depth: u32, mu: f32, cos_b: f32, sin_b: f32) -> bool {
    if depth >= HORIZON_SPLIT_DEPTH || !(cos_b.abs() <= 1.0) || !mu.is_finite() || sin_b < 0.0 {
        return false;
    }
    let mu = mu.clamp(-1.0, 1.0);
    // gamma <= beta + 1e-6, with cos decreasing on [0, π].
    let cos_contains = cos_b - sin_b * 1.0e-6;
    if mu >= cos_contains {
        return true;
    }
    if depth >= 2 {
        return false;
    }
    if cos_b <= 0.0 {
        return true;
    }
    let cos_2b = 2.0 * cos_b * cos_b - 1.0;
    let sin_2b = 2.0 * sin_b * cos_b;
    mu >= cos_2b - sin_2b * 1.0e-6
}

/// Terrain corners on opposite sides of the eye radius. One 4×4 split; the
/// sub-cell max stays at a corner. The shell is not part of this test: once
/// it contains the camera every cell would split.
#[inline]
fn horizon_straddle(depth: u32, radius: f32, min_off: f32, max_off: f32, distance: f32) -> bool {
    if depth >= 1 {
        return false;
    }
    let r_max = radius + max_off;
    let r_min = radius + min_off;
    r_max >= distance && r_min < distance
}

/// Elevation sine of one datum patch.
///
/// A patch that contains the up axis and whose terrain is below the eye does
/// not take `theta_min = 0` at the shell radius. That sample is the zenith
/// spike — the shell above the camera — and it is not a shaded ray
/// (`facing > 0` stays below the horizontal). Smearing it paints every bin
/// with sine 1, which is what a coarse cell under the eye did. The patch
/// contributes the smooth-sphere limb of its own radius instead: negative
/// while the eye is outside that sphere, and 0 once the shell contains the
/// eye. `theta_min = 0` is used only when the terrain itself reaches the eye.
///
/// A patch that does not contain the axis still drops a positive shell spike
/// when the terrain is below the eye. The drawable limb is the horizontal.
fn horizon_patch_elev(
    rho_outer: f32,
    rho_terrain: f32,
    mu_far: f32,
    mu_near: f32,
    contains_axis: bool,
) -> f32 {
    if contains_axis && rho_terrain < 1.0 {
        if rho_outer < 1.0 {
            return -(1.0 - rho_outer * rho_outer).max(0.0).sqrt();
        }
        return 0.0;
    }
    if rho_outer >= 1.0 && rho_terrain < 1.0 {
        let spiked = cap_elev(rho_outer, mu_far, mu_near);
        if spiked > 0.0 {
            return cap_elev(rho_terrain, mu_far, mu_near).max(0.0);
        }
        return spiked;
    }
    cap_elev(rho_outer, mu_far, mu_near)
}

/// Cheap upper bound on [`horizon_patch_elev`] for a patch of this `max_off`.
/// Depends only on the radius, so a cell at or under the bound cannot raise a
/// bin the axis cap has already filled.
#[inline(always)]
fn horizon_elev_upper(radius: f32, max_off: f32, shell: f32, distance: f32) -> f32 {
    let rho_t = (radius + max_off) / distance;
    if !rho_t.is_finite() || rho_t >= 1.0 {
        return 1.0;
    }
    let rho_o = (radius + max_off + shell) / distance;
    if !(rho_o > 0.0) || !rho_o.is_finite() {
        return -1.0;
    }
    if rho_o >= 1.0 {
        return 0.0;
    }
    -(1.0 - rho_o * rho_o).max(0.0).sqrt()
}

struct HorizonAccum<'a> {
    bins: &'a mut [f32; HORIZON_BINS],
    geom: &'a HorizonGeom,
    /// Eye axes in body space (`R⁻¹` of the world axes). Azimuth is
    /// `atan2(centre·east, centre·north)` and matches [`horizon_azimuth`].
    up_b: glam::Vec3,
    east_b: glam::Vec3,
    north_b: glam::Vec3,
    radius: f32,
    distance: f32,
    shell: f32,
    bin_w: f32,
    /// Bins already at or above this sine cannot be raised by a patch whose
    /// upper bound is lower. `-∞` until the axis cap has been written.
    floor: f32,
    visited: u32,
    truncated: bool,
}

impl HorizonAccum<'_> {
    fn dir_k(
        &self,
        basis: (glam::Vec3, glam::Vec3, glam::Vec3),
        ku: u32,
        kv: u32,
    ) -> Option<glam::Vec3> {
        let a = self.geom.tan_q[ku as usize];
        let b = self.geom.tan_q[kv as usize];
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let (tu, n, tv) = basis;
        unit_dir(n + tu * a + tv * b)
    }

    /// Chart midpoint. Spans coarser than one fine step land on the tan grid.
    fn dir_mid(
        &self,
        face: usize,
        basis: (glam::Vec3, glam::Vec3, glam::Vec3),
        ku0: u32,
        kv0: u32,
        ku1: u32,
        kv1: u32,
    ) -> Option<glam::Vec3> {
        if ku1 > ku0 + 1 && kv1 > kv0 + 1 {
            return self.dir_k(basis, (ku0 + ku1) / 2, (kv0 + kv1) / 2);
        }
        let fine = self.geom.cells * self.geom.stride;
        let mid = |k0: u32, k1: u32| {
            let t = 0.5 * (k0 as f32 + k1 as f32);
            2.0 * t / fine as f32 - 1.0
        };
        chart_dir(face, mid(ku0, ku1), mid(kv0, kv1))
    }

    #[inline(always)]
    fn commit(&mut self, centre: glam::Vec3, cos_b: f32, sin_b: f32, max_off: f32) {
        let mu = centre.dot(self.up_b).clamp(-1.0, 1.0);
        let sin_g = (1.0 - mu * mu).max(0.0).sqrt();
        let rho_outer = (self.radius + max_off + self.shell) / self.distance;
        let rho_terrain = (self.radius + max_off) / self.distance;
        if !(rho_outer > 0.0) || !rho_outer.is_finite() {
            return;
        }
        let (mu_far, mu_near) = cap_mu_fast(mu, cos_b, sin_b, sin_g);
        let cos_contains = cos_b - sin_b * 1.0e-6;
        let contains = mu >= cos_contains;
        let south = mu <= -cos_contains;
        let elev = horizon_patch_elev(rho_outer, rho_terrain, mu_far, mu_near, contains);
        let half = if contains || south || !(sin_g > 1.0e-8) {
            None
        } else {
            let ratio = (sin_b / sin_g).clamp(0.0, 1.0);
            Some(fast_asin(ratio) + self.bin_w + HORIZON_AZ_PAD)
        };
        let az = fast_atan2(centre.dot(self.east_b), centre.dot(self.north_b));
        raise_bins(self.bins, az, half, elev);
    }

    fn patch(
        &mut self,
        face: usize,
        ku0: u32,
        kv0: u32,
        ku1: u32,
        kv1: u32,
        h00: f32,
        h10: f32,
        h01: f32,
        h11: f32,
        depth: u32,
    ) {
        self.visited += 1;
        if !h00.is_finite() || !h10.is_finite() || !h01.is_finite() || !h11.is_finite() {
            return;
        }
        let max_off = h00.max(h10).max(h01).max(h11);
        let min_off = h00.min(h10).min(h01).min(h11);
        if self.floor.is_finite()
            && horizon_elev_upper(self.radius, max_off, self.shell, self.distance) + HORIZON_SIN_PAD
                <= self.floor
        {
            return;
        }
        let basis = crate::far_body::far_map_basis(face);
        let (Some(c00), Some(c10), Some(c01), Some(c11), Some(centre)) = (
            self.dir_k(basis, ku0, kv0),
            self.dir_k(basis, ku1, kv0),
            self.dir_k(basis, ku0, kv1),
            self.dir_k(basis, ku1, kv1),
            self.dir_mid(face, basis, ku0, kv0, ku1, kv1),
        ) else {
            return;
        };
        let min_cos = centre
            .dot(c00)
            .min(centre.dot(c10))
            .min(centre.dot(c01))
            .min(centre.dot(c11));
        let (cos_b, sin_b) = horizon_cap_axes(min_cos);
        // Corners closer than an f32 ulp share one direction. Deeper 4×4
        // splits stay on that point and only multiply the patch count.
        if min_cos >= 1.0 - 1.0e-7 {
            self.commit(centre, cos_b, sin_b, max_off);
            return;
        }
        let mu = centre.dot(self.up_b);
        let want = horizon_should_split(
            depth,
            mu,
            cos_b,
            sin_b,
            self.radius,
            min_off,
            max_off,
            self.distance,
        );
        let split = want && self.visited < HORIZON_PATCH_BUDGET;
        if want && !split {
            self.truncated = true;
        }
        if split {
            let step_u = (ku1 - ku0) / HORIZON_SPLIT_N;
            let step_v = (kv1 - kv0) / HORIZON_SPLIT_N;
            if step_u == 0 || step_v == 0 {
                self.commit(centre, cos_b, sin_b, max_off);
                return;
            }
            let nf = HORIZON_SPLIT_N as f32;
            for sv in 0..HORIZON_SPLIT_N {
                let v0 = sv as f32 / nf;
                let v1 = (sv + 1) as f32 / nf;
                let kv_a = kv0 + step_v * sv;
                let kv_b = kv_a + step_v;
                for su in 0..HORIZON_SPLIT_N {
                    let u0 = su as f32 / nf;
                    let u1 = (su + 1) as f32 / nf;
                    let ku_a = ku0 + step_u * su;
                    let ku_b = ku_a + step_u;
                    self.patch(
                        face,
                        ku_a,
                        kv_a,
                        ku_b,
                        kv_b,
                        horizon_bilerp(h00, h10, h01, h11, u0, v0),
                        horizon_bilerp(h00, h10, h01, h11, u1, v0),
                        horizon_bilerp(h00, h10, h01, h11, u0, v1),
                        horizon_bilerp(h00, h10, h01, h11, u1, v1),
                        depth + 1,
                    );
                }
            }
            return;
        }
        self.commit(centre, cos_b, sin_b, max_off);
    }
}

/// 256-bin horizon table for one Mapped body.
///
/// `up` is the unit centre→eye direction. A cell's radius is the max of its
/// four corner offsets (exact for bilinear) plus `radius` plus the shell
/// `max(air, px * distance)`. `px` is an upper bound on the shader's pixel
/// angle, because the limb shell is `max(air/distance, px)`. Directions of
/// the cell lie in a cap of the half-diagonal angle around the chart-midpoint
/// direction. The cap's highest elevation at that radius is written into
/// every azimuth bin the cap overlaps; a cap that contains the up axis
/// covers every bin. An unwritten bin becomes 1 so it cannot reject.
///
/// A datum cell at g = 33 is ~1.5e6 blocks wide. The cell under the eye can
/// hold a corner above the camera while the ground underfoot is far below, and
/// the air shell can contain the eye on its own. That cap used to write sine 1
/// into every bin. Those cells are split bilinearly into 4×4 sub-cells
/// ([`horizon_should_split`]); a sub-cell's max stays at a corner. Chart
/// directions for a given `g` are cached. Cells that cannot rise above the
/// sine the axis cap wrote are skipped.
fn build_horizon_bins(
    g: u32,
    datum: &[f32],
    rotation: glam::Quat,
    up: glam::Vec3,
    radius: f32,
    distance: f32,
    air: f32,
    px: f32,
) -> [f32; HORIZON_BINS] {
    let full = [1.0; HORIZON_BINS];
    if g < 2 || !(distance > 0.0) || !distance.is_finite() || !radius.is_finite() {
        return full;
    }
    let gg = g as usize;
    let n = 6 * gg * gg;
    if datum.len() < n || datum.iter().take(n).any(|v| !v.is_finite()) {
        return full;
    }
    let Some(up) = unit_dir(up) else {
        return full;
    };
    let Some((east, north)) = horizon_axes(up) else {
        return full;
    };
    let rotation = if rotation.is_finite() && rotation.length_squared() > 0.0 {
        rotation.normalize()
    } else {
        glam::Quat::IDENTITY
    };
    let shell = air.max(0.0).max(px.max(0.0) * distance);
    let mut bins = [-2.0f32; HORIZON_BINS];
    let bin_w = std::f32::consts::TAU / HORIZON_BINS as f32;
    let cells = g - 1;
    let owned;
    let geom: &HorizonGeom = match horizon_geom(g) {
        Some(cached) => cached,
        None => {
            owned = build_horizon_geom(g);
            &owned
        }
    };
    let inv = rotation.conjugate();
    let mut accum = HorizonAccum {
        bins: &mut bins,
        geom,
        up_b: inv * up,
        east_b: inv * east,
        north_b: inv * north,
        radius,
        distance,
        shell,
        bin_w,
        floor: f32::NEG_INFINITY,
        visited: 0,
        truncated: false,
    };
    let stride = geom.stride;
    // Axis caps first. One of them paints every bin, and that sine is the
    // floor the other cells have to beat.
    for face in 0..6usize {
        let face_base = face * gg * gg;
        for j in 0..cells {
            let row = face_base + j as usize * gg;
            for i in 0..cells {
                let geom_i = (face * cells as usize + j as usize) * cells as usize + i as usize;
                let sample = &geom.cell[geom_i];
                if sample.sin_beta >= 0.0
                    && !horizon_reaches_axis(
                        0,
                        sample.centre.dot(accum.up_b),
                        sample.cos_beta,
                        sample.sin_beta,
                    )
                {
                    continue;
                }
                let h00 = datum[row + i as usize];
                let h10 = datum[row + i as usize + 1];
                let h01 = datum[row + gg + i as usize];
                let h11 = datum[row + gg + i as usize + 1];
                let ku0 = i * stride;
                let kv0 = j * stride;
                accum.patch(
                    face,
                    ku0,
                    kv0,
                    ku0 + stride,
                    kv0 + stride,
                    h00,
                    h10,
                    h01,
                    h11,
                    0,
                );
            }
        }
    }
    let mut floor = f32::INFINITY;
    for bin in accum.bins.iter() {
        if *bin < -1.5 {
            floor = f32::NEG_INFINITY;
            break;
        }
        floor = floor.min(*bin);
    }
    accum.floor = floor;
    for face in 0..6usize {
        let face_base = face * gg * gg;
        for j in 0..cells {
            let row = face_base + j as usize * gg;
            for i in 0..cells {
                let geom_i = (face * cells as usize + j as usize) * cells as usize + i as usize;
                let sample = &geom.cell[geom_i];
                let h00 = datum[row + i as usize];
                let h10 = datum[row + i as usize + 1];
                let h01 = datum[row + gg + i as usize];
                let h11 = datum[row + gg + i as usize + 1];
                let max_off = h00.max(h10).max(h01).max(h11);
                if sample.sin_beta >= 0.0
                    && horizon_elev_upper(radius, max_off, shell, distance) + HORIZON_SIN_PAD
                        <= floor
                {
                    continue;
                }
                let mu = sample.centre.dot(accum.up_b);
                if sample.sin_beta < 0.0
                    || horizon_reaches_axis(0, mu, sample.cos_beta, sample.sin_beta)
                {
                    continue;
                }
                let min_off = h00.min(h10).min(h01).min(h11);
                if horizon_straddle(0, radius, min_off, max_off, distance) {
                    let ku0 = i * stride;
                    let kv0 = j * stride;
                    accum.patch(
                        face,
                        ku0,
                        kv0,
                        ku0 + stride,
                        kv0 + stride,
                        h00,
                        h10,
                        h01,
                        h11,
                        0,
                    );
                } else {
                    accum.commit(sample.centre, sample.cos_beta, sample.sin_beta, max_off);
                }
            }
        }
    }
    debug_assert!(
        !accum.truncated,
        "horizon split stopped after {} patches",
        accum.visited
    );
    for bin in &mut bins {
        if *bin < -1.5 {
            *bin = 1.0;
        }
    }
    bins
}

/// `true` when `ray` is strictly above `bins`. `ray` need not be unit.
fn ray_above_horizon(
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
fn tile_above_horizon(
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
/// body's table. Refills the tile lists when a bit changes. Culling off
/// leaves the mask alone; the shader test still runs.
fn clear_tiles_above_horizon(table: &mut FarTableGpu, frames: &TileFrames, view: &FarView) -> bool {
    if !far_cull_enabled() || !frames.matches(view) {
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
        fill_tile_lists(table, view.width, view.height);
    }
    changed
}

/// One cached horizon table, keyed by the map and the body that was built.
struct HorizonSlot {
    live: bool,
    map: u32,
    generation: u32,
    radius_bits: u32,
    air_bits: u32,
    px_bits: u32,
    rot_bits: [u32; 4],
    /// Centre → eye, world space, at the build.
    eye: glam::Vec3,
    /// Altitude above the lo sphere at the build. The motion limit is
    /// [`HORIZON_MOVE_FRAC`] of this.
    altitude: f32,
    bins: [f32; HORIZON_BINS],
}

impl HorizonSlot {
    fn empty() -> Self {
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

struct HorizonCache {
    slots: [HorizonSlot; HORIZON_TABLES],
}

impl HorizonCache {
    fn new() -> Self {
        Self {
            slots: std::array::from_fn(|_| HorizonSlot::empty()),
        }
    }
}

/// `true` when `eye` is within 0.1% of the cached altitude above the lo
/// sphere and the body inputs still match. A non-positive cached altitude
/// keeps the table only when the eye has not moved.
fn horizon_cache_hit(
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

fn cache_dest(cache: &HorizonCache, used: &[bool; HORIZON_TABLES], map: u32) -> usize {
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

/// Sentinel ids and sines. A zeroed table would name body 0 and reject every
/// ray above the horizontal.
fn prime_horizon(table: &mut FarTableGpu) {
    table.horizon_id = [u32::MAX, u32::MAX, HORIZON_BINS as u32, 0];
    table.horizon_sin.fill(1.0);
}

fn pack_table_cached(
    bodies: &[FarBody],
    view: Option<&FarView>,
    frames: Option<&TileFrames>,
    map_max: &[f32; MAX_FAR_MAPS],
    map_min: &[f32; MAX_FAR_MAPS],
    sky_up: glam::Vec3,
) -> (Box<FarTableGpu>, f32) {
    let mut table = zeroed_table();
    prime_horizon(&mut table);
    let n = bodies.len().min(MAX_FAR_BODIES);
    // Before the cull: see [`horizon_dip`].
    let dip = horizon_dip(&bodies[..n], sky_up, map_min);
    let cull = far_cull_enabled();
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        let bound = cone_bound(body, map_max);
        if cull {
            if let Some(view) = view {
                if !body_meets_view(body, bound, view) {
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
    (table, dip)
}

/// Per-slot far-body SSBO. Identical bytes skip the map write.
/// `tiles` is the view-space tile cones, rebuilt when the projection, the
/// render extent, or the tile size changes and reused across camera turns.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    /// Previous upload. Boxed: three inline tables would add ~210 KB to
    /// `Renderer`, which lives on the render thread's default stack.
    last: PerSlot<Option<Box<FarTableGpu>>>,
    tiles: TileFrames,
    draw: PerSlot<SkyDraw>,
    /// Reused across frames. Rebuilt when the eye moves more than 0.1% of
    /// its altitude above the lo sphere, or the body's rotation, radius,
    /// air, pixel angle, or datum generation changes.
    horizon: HorizonCache,
    datum_scratch: Vec<f32>,
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
            draw: PerSlot::new(std::array::from_fn(|_| SkyDraw::default())),
            horizon: HorizonCache::new(),
            datum_scratch: Vec::new(),
        }
    }

    /// Draw the sky recorded for `slot` after [`Self::write`].
    pub(crate) fn sky_draw(&self, slot: FrameSlot) -> SkyDraw {
        self.draw[slot]
    }

    /// Fill the two horizon tables from the datum and drop tiles that sit
    /// entirely above them. At most two Mapped bodies, the ones with the
    /// largest reference rho whose hi+air ball contains the eye. A cache hit
    /// skips the cell walk.
    fn publish_horizons(
        &mut self,
        table: &mut FarTableGpu,
        view: Option<&FarView>,
        maps: &super::far_maps::FarMaps,
        map_max: &[f32; MAX_FAR_MAPS],
        map_min: &[f32; MAX_FAR_MAPS],
    ) {
        let kept = (table.header[0] as usize).min(MAX_FAR_BODIES);
        let px = view.map(|v| v.px_max).unwrap_or(0.0);
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

        let mut used = [false; HORIZON_TABLES];
        for (out_slot, cand) in cands.iter().enumerate() {
            let gpu = table.body[cand.index];
            let distance = gpu.albedo2[3];
            let radius = gpu.dir_rho[3] * distance;
            let air = gpu.albedo1[3].max(0.0);
            let min_off = map_min[cand.map];
            let altitude = distance - (radius + min_off);
            let Some(dir) = unit_dir(glam::Vec3::new(
                gpu.dir_rho[0],
                gpu.dir_rho[1],
                gpu.dir_rho[2],
            )) else {
                continue;
            };
            let eye = -dir * distance;
            let rot_bits = [
                gpu.rot[0].to_bits(),
                gpu.rot[1].to_bits(),
                gpu.rot[2].to_bits(),
                gpu.rot[3].to_bits(),
            ];
            let Some((g_stamp, generation)) = maps.map_stamp(cand.map) else {
                continue;
            };
            if g_stamp < 2 {
                continue;
            }
            let map_id = cand.map as u32;
            let px_bits = px.to_bits();
            let radius_bits = radius.to_bits();
            let air_bits = air.to_bits();
            let hit = self.horizon.slots.iter().position(|slot| {
                horizon_cache_hit(
                    slot,
                    map_id,
                    generation,
                    radius_bits,
                    air_bits,
                    px_bits,
                    rot_bits,
                    eye,
                )
            });
            let bins = if let Some(i) = hit {
                used[i] = true;
                self.horizon.slots[i].bins
            } else {
                let Some(g) = maps.copy_datum(cand.map, &mut self.datum_scratch) else {
                    continue;
                };
                let rotation =
                    glam::Quat::from_xyzw(gpu.rot[0], gpu.rot[1], gpu.rot[2], gpu.rot[3]);
                let built = build_horizon_bins(
                    g,
                    &self.datum_scratch,
                    rotation,
                    -dir,
                    radius,
                    distance,
                    air,
                    px,
                );
                let dest = cache_dest(&self.horizon, &used, map_id);
                self.horizon.slots[dest] = HorizonSlot {
                    live: true,
                    map: map_id,
                    generation,
                    radius_bits,
                    air_bits,
                    px_bits,
                    rot_bits,
                    eye,
                    altitude,
                    bins: built,
                };
                used[dest] = true;
                built
            };
            table.horizon_id[out_slot] = cand.index as u32;
            let start = out_slot * HORIZON_BINS;
            table.horizon_sin[start..start + HORIZON_BINS].copy_from_slice(&bins);
        }
        if let Some(view) = view
            && self.tiles.matches(view)
        {
            clear_tiles_above_horizon(table, &self.tiles, view);
        }
    }

    /// Pack and upload. Returns the horizon-dip sine for `sky_bitangent.w`
    /// ([`horizon_dip`]), including when the table bytes are unchanged.
    pub(crate) fn write(
        &mut self,
        slot: FrameSlot,
        bodies: &[FarBody],
        view: Option<FarView>,
        maps: &super::far_maps::FarMaps,
        coarse: Option<SkyCoarseQuery>,
        sky_up: glam::Vec3,
    ) -> f32 {
        if let Some(view) = view.as_ref() {
            self.tiles.rebuild_if_changed(view);
        }
        let map_max = maps.max_offsets();
        let map_min = maps.min_offsets();
        let (mut table, dip) = pack_table_cached(
            bodies,
            view.as_ref(),
            Some(&self.tiles),
            &map_max,
            &map_min,
            sky_up,
        );
        self.publish_horizons(&mut table, view.as_ref(), maps, &map_max, &map_min);
        let offered = bodies.len().min(MAX_FAR_BODIES) as u64;
        crate::profile::gauge(crate::profile::Gauge::FarBodies, offered);
        crate::profile::gauge(crate::profile::Gauge::FarDrawn, u64::from(table.header[0]));
        // `far.tiles` is the full-pipeline tile count (mask != 0).
        debug_assert_eq!(u64::from(table.list_header[1]), nonzero_tiles(&table));
        crate::profile::gauge(
            crate::profile::Gauge::FarTiles,
            u64::from(table.list_header[1]),
        );
        let mut draw = SkyDraw::from_table(&table, far_cull_enabled());
        // Fullscreen sky (no tile grid) stays on the 1×1 triangle. The split
        // rewrites the uploaded index prefix, so it runs before the byte match.
        if draw.quads {
            if let (Some(query), Some(view)) = (coarse.as_ref(), view.as_ref()) {
                if self.tiles.matches(view) {
                    draw.n_coarse = split_coarse_base(&mut table, &self.tiles, view, query);
                    draw.n_coarse_far =
                        split_coarse_far(&mut table, &self.tiles, view, query, &map_min);
                }
            }
        }
        crate::profile::gauge(crate::profile::Gauge::SkyCoarse, u64::from(draw.n_coarse));
        crate::profile::gauge(
            crate::profile::Gauge::SkyCoarseFar,
            u64::from(draw.n_coarse_far),
        );
        self.draw[slot] = draw;
        let bytes = table_bytes(&table);
        let lists = list_bytes(&table);
        if self.last[slot]
            .as_ref()
            .is_some_and(|prev| table_bytes(prev) == bytes && list_bytes(prev) == lists)
        {
            return dip;
        }
        unsafe {
            self.bufs[slot].write(0, bytes);
            self.bufs[slot].write(std::mem::offset_of!(FarTableGpu, list_header) as u64, lists);
        }
        self.last[slot] = Some(table);
        dip
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
        FarBody, FarMapId, FarShape, ray_cube, ray_inner_sphere, ray_mapped, ray_mapped_fast,
        ray_mapped_limb, ray_rounded, ray_sphere, store,
    };

    fn pack_table(bodies: &[FarBody], view: Option<FarView>) -> FarTableGpu {
        super::pack_table(bodies, view, &[0.0; MAX_FAR_MAPS])
    }

    /// Reference-radius dip. A zero minimum-offset table leaves every shape,
    /// including Mapped, on `radius/distance`.
    fn horizon_dip(bodies: &[FarBody], sky_up: Vec3) -> f32 {
        super::horizon_dip(bodies, sky_up, &[0.0; MAX_FAR_MAPS])
    }
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
    fn every_shape_packs_world_distance_in_albedo2_w() {
        let shapes = [
            (FarShape::Sphere, 12.5, 1.0),
            (FarShape::Cube, 12.5, 1.0),
            (FarShape::InnerSphere, 2.0, 8.0),
            (FarShape::Rounded { exponent: 3.0 }, 12.5, 1.0),
            (
                FarShape::Mapped {
                    map: FarMapId(1),
                    horizon: 0.1,
                    air: 0.2,
                },
                12.5,
                1.0,
            ),
        ];
        for (shape, distance, radius) in shapes {
            let gpu = super::pack_one(&sample(shape, distance, radius));
            assert_eq!(
                gpu.albedo2[3].to_bits(),
                distance.to_bits(),
                "albedo2.w is the world distance"
            );
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

        let mut map_max = [0.0f32; MAX_FAR_MAPS];
        map_max[3] = 0.1;
        let mapped = sample(
            FarShape::Mapped {
                map: FarMapId(3),
                horizon: -0.25,
                air: 0.4,
            },
            4.0,
            1.0,
        );
        let table = super::pack_table(std::slice::from_ref(&mapped), None, &map_max);
        let hi = ((1.0f32 + 0.1) / 4.0).max(0.0);
        let shell = (0.4f32 / 4.0).max(0.0);
        assert!(hi + shell < 0.99);
        assert_eq!(table.cone[0][3].to_bits(), (hi + shell).to_bits());
        let gpu = &table.body[0];
        assert_eq!(gpu.atmosphere[3].to_bits(), 4.0f32.to_bits());
        assert_eq!(gpu.seed[2], 4);
        assert_eq!(gpu.seed[1], 0);
        assert_eq!(gpu.seed[3], 0);
        assert_eq!(gpu.albedo0[3].to_bits(), (-0.25f32).to_bits());
        assert_eq!(gpu.albedo1[3].to_bits(), 0.4f32.to_bits());
        assert_eq!(gpu.albedo2[3].to_bits(), 4.0f32.to_bits());
        assert_eq!(gpu.dir_rho[3].to_bits(), 0.25f32.to_bits());
        map_max[3] = 10.0;
        assert_eq!(
            super::pack_table(std::slice::from_ref(&mapped), None, &map_max).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        assert_eq!(std::mem::offset_of!(FarTableGpu, cone), 16);
        assert_eq!(
            std::mem::offset_of!(FarTableGpu, body),
            16 + 16 * MAX_FAR_BODIES
        );
    }

    #[test]
    fn mapped_cone_widens_by_the_air_shell() {
        let mut map_max = [0.0f32; MAX_FAR_MAPS];
        map_max[2] = 30.0;
        let distance = 1_000.0f32;
        let radius = 200.0f32;
        let air = 40.0f32;
        let hi = ((radius + map_max[2]) / distance).max(0.0);
        let shell = (air / distance).max(0.0);
        assert!(hi + shell < 0.99, "precondition {hi} {shell}");
        // 1080p at 60° is about a milliradian per pixel. This shell is many
        // times the shader's 3 px margin, so the margin cannot stand in for it.
        let px = 2.0 * (60.0f32.to_radians() * 0.5).tan() / 1080.0;
        assert!(
            shell > 3.0 * px,
            "shell {shell} must exceed the 3 px margin {px}"
        );

        let bare = sample(
            FarShape::Mapped {
                map: FarMapId(2),
                horizon: 1.0,
                air: 0.0,
            },
            distance,
            radius,
        );
        assert_eq!(
            super::pack_table(std::slice::from_ref(&bare), None, &map_max).cone[0][3].to_bits(),
            hi.to_bits(),
            "no air leaves the hi radius"
        );

        let thick = sample(
            FarShape::Mapped {
                map: FarMapId(2),
                horizon: 1.0,
                air,
            },
            distance,
            radius,
        );
        assert_eq!(
            super::pack_table(std::slice::from_ref(&thick), None, &map_max).cone[0][3].to_bits(),
            (hi + shell).to_bits()
        );

        // hi itself is inside the sine limit; the shell crosses it.
        let close_distance = 100.0f32;
        let close_radius = 50.0f32;
        let close_air = 25.0f32;
        let close_hi = ((close_radius + map_max[2]) / close_distance).max(0.0);
        let close_bound = close_hi + (close_air / close_distance).max(0.0);
        assert!(
            close_hi < 0.99 && close_bound >= 0.99,
            "{close_hi} {close_bound}"
        );
        let close = sample(
            FarShape::Mapped {
                map: FarMapId(2),
                horizon: 1.0,
                air: close_air,
            },
            close_distance,
            close_radius,
        );
        assert_eq!(
            super::pack_table(std::slice::from_ref(&close), None, &map_max).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
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

    /// `rho = R/(R+h)` straight down. The exact dip is `sqrt(1-rho²)`; for
    /// small `h` that is `sqrt(2h/R)`.
    fn dip_below(rho: f32, up: Vec3) -> f32 {
        horizon_dip(
            std::slice::from_ref(&placed(-up, rho, FarShape::Sphere, 1)),
            up,
        )
    }

    #[test]
    fn horizon_dip_is_zero_without_a_ground_body() {
        assert_eq!(horizon_dip(&[], Vec3::Y).to_bits(), 0.0f32.to_bits());
        // Above the viewer, and level with the horizon: neither is ground.
        let above = placed(Vec3::Y, 0.9, FarShape::Sphere, 1);
        let sideways = placed(Vec3::X, 0.9, FarShape::Sphere, 2);
        // Just outside the 60° cone (`dot` a hair under 0.5). The compare is strict.
        let edge = placed(
            Vec3::new(0.87, -0.5, 0.0),
            0.9,
            FarShape::Mapped {
                map: crate::far_body::FarMapId(0),
                horizon: 1.0,
                air: 0.0,
            },
            3,
        );
        assert_eq!(horizon_dip(&[above], Vec3::Y).to_bits(), 0.0f32.to_bits());
        assert_eq!(
            horizon_dip(&[sideways], Vec3::Y).to_bits(),
            0.0f32.to_bits()
        );
        assert_eq!(horizon_dip(&[edge], Vec3::Y).to_bits(), 0.0f32.to_bits());
        // Inside the body (rho >= 1), including an inner sphere underfoot.
        let inside = placed(-Vec3::Y, 1.5, FarShape::InnerSphere, 4);
        assert_eq!(horizon_dip(&[inside], Vec3::Y).to_bits(), 0.0f32.to_bits());
        // A body above must not win over "no ground", even with a larger rho
        // than a body that fails the outside test.
        assert_eq!(
            horizon_dip(&[above, inside], Vec3::Y).to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn horizon_dip_of_the_body_below_matches_sqrt_2h_over_r() {
        let r = 6_371_000.0f32;
        let h = 10_000.0f32;
        let rho = r / (r + h);
        let s = dip_below(rho, Vec3::Y);
        let approx = (2.0 * h / r).sqrt();
        let exact = (1.0 - rho * rho).max(0.0).sqrt();
        assert!((s - exact).abs() <= exact * 1e-6, "s {s} exact {exact}");
        assert!(
            (s - approx).abs() < 1e-4,
            "small-h dip {s} should track sqrt(2h/R) {approx}"
        );
        assert!(s < 0.5, "10 km on an Earth-sized body is under the clamp");
        // Non-unit up, and a ground body that is not world -Y.
        let up = Vec3::Z * 4.0;
        let s_z = dip_below(rho, up);
        assert_eq!(s_z.to_bits(), s.to_bits());
        assert_eq!(
            horizon_dip(
                std::slice::from_ref(&placed(-Vec3::Y, rho, FarShape::Sphere, 1)),
                up
            )
            .to_bits(),
            0.0f32.to_bits()
        );
    }

    #[test]
    fn horizon_dip_ignores_bodies_above_or_sideways_and_clamps() {
        let ground = placed(-Vec3::Y, 0.98, FarShape::Sphere, 1);
        let above = placed(Vec3::Y, 0.99, FarShape::Sphere, 2);
        let side = placed(Vec3::X, 0.995, FarShape::Cube, 3);
        let s = horizon_dip(&[above, side, ground], Vec3::Y);
        let expect = (1.0 - 0.98 * 0.98f32).sqrt();
        assert!((s - expect).abs() < 1e-6, "{s} vs {expect}");
        // Largest rho wins, not the largest dip. The tiny body would clamp.
        let tiny = placed(
            Vec3::new(0.0, -1.0, 0.1).normalize(),
            0.2,
            FarShape::Sphere,
            4,
        );
        let s_large = horizon_dip(&[tiny, ground], Vec3::Y);
        assert!(
            (s_large - expect).abs() < 1e-6,
            "largest rho, got {s_large}"
        );
        // A small disc underfoot: sqrt(1-rho²) > 0.5, so the lane saturates.
        let s_clamp = dip_below(0.2, Vec3::Y);
        assert_eq!(s_clamp.to_bits(), 0.5f32.to_bits());
        // Just inside the 60° cone still counts.
        let inside_cone = placed(
            Vec3::new((1.0 - 0.6 * 0.6f32).sqrt(), -0.6, 0.0),
            0.97,
            FarShape::Sphere,
            5,
        );
        let s_cone = horizon_dip(&[inside_cone], Vec3::Y);
        let cone_expect = (1.0 - 0.97 * 0.97f32).sqrt();
        assert!((s_cone - cone_expect).abs() < 1e-6, "{s_cone}");
    }

    #[test]
    fn pack_reports_the_same_horizon_dip() {
        let bodies = [
            placed(Vec3::Y, 0.4, FarShape::Sphere, 1),
            placed(-Vec3::Y, 0.96, FarShape::Sphere, 2),
        ];
        let map_min = [0.0; MAX_FAR_MAPS];
        let (table, dip) =
            pack_table_cached(&bodies, None, None, &[0.0; MAX_FAR_MAPS], &map_min, Vec3::Y);
        assert_eq!(table.header[0], 2);
        assert_eq!(dip.to_bits(), horizon_dip(&bodies, Vec3::Y).to_bits());
        assert!(dip > 0.0 && dip < 0.5);
    }

    /// Lowlands sit inside the reference sphere, so the sky dip has to follow
    /// the lo sphere or the gradient edge draws above the silhouette.
    #[test]
    fn horizon_dip_of_a_mapped_ground_body_uses_the_lo_sphere() {
        let radius = 31_000_000.0f32;
        let distance = radius + 50_000.0;
        let min_off = -278_000.0f32;
        let mut map_min = [0.0f32; MAX_FAR_MAPS];
        map_min[3] = min_off;
        let mapped = |map: u8| FarBody {
            dir: -Vec3::Y,
            distance,
            radius,
            shape: FarShape::Mapped {
                map: FarMapId(map),
                horizon: 0.2,
                air: 12.0,
            },
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.1, 0.0, 0.0]),
            seed: 7,
        };
        let reference = FarBody {
            shape: FarShape::Sphere,
            ..mapped(3)
        };
        let lo_sphere = FarBody {
            radius: radius + min_off,
            shape: FarShape::Sphere,
            ..mapped(3)
        };

        let dip_mapped = super::horizon_dip(std::slice::from_ref(&mapped(3)), Vec3::Y, &map_min);
        let dip_ref = super::horizon_dip(std::slice::from_ref(&reference), Vec3::Y, &map_min);
        let dip_lo = super::horizon_dip(std::slice::from_ref(&lo_sphere), Vec3::Y, &map_min);
        assert!(
            dip_mapped > dip_ref,
            "lo sphere {dip_mapped} should dip past the reference sphere {dip_ref}"
        );
        assert_eq!(dip_mapped.to_bits(), dip_lo.to_bits());
        let rho = radius / distance;
        let exact = (1.0 - rho * rho).max(0.0).sqrt();
        assert_eq!(dip_ref.to_bits(), exact.to_bits());

        // A sphere does not read the offset table.
        let mut noisy = [123_456.0f32; MAX_FAR_MAPS];
        noisy[3] = min_off;
        assert_eq!(
            super::horizon_dip(std::slice::from_ref(&reference), Vec3::Y, &noisy).to_bits(),
            dip_ref.to_bits()
        );

        // Unknown id, and a cleared slot whose minimum is still 0.
        assert_eq!(
            super::horizon_dip(
                std::slice::from_ref(&mapped(MAX_FAR_MAPS as u8)),
                Vec3::Y,
                &map_min
            )
            .to_bits(),
            dip_ref.to_bits()
        );
        assert_eq!(
            super::horizon_dip(std::slice::from_ref(&mapped(0)), Vec3::Y, &map_min).to_bits(),
            dip_ref.to_bits()
        );
        // A non-finite offset is not a datum; stay on the reference radius.
        map_min[3] = f32::NAN;
        assert_eq!(
            super::horizon_dip(std::slice::from_ref(&mapped(3)), Vec3::Y, &map_min).to_bits(),
            dip_ref.to_bits()
        );

        map_min[3] = min_off;
        let (_table, packed) = pack_table_cached(
            std::slice::from_ref(&mapped(3)),
            None,
            None,
            &[0.0; MAX_FAR_MAPS],
            &map_min,
            Vec3::Y,
        );
        assert_eq!(packed.to_bits(), dip_mapped.to_bits());
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
            FarShape::Mapped { horizon, .. } => {
                let datum = [0.0f32; 24];
                ray_mapped(
                    ray,
                    dir,
                    rho,
                    body.distance,
                    body.rotation,
                    horizon,
                    2,
                    &datum,
                    0.0,
                    0.0,
                )
                .is_some()
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
            FarShape::Mapped { horizon, air, .. } => {
                if ray.dot(-dir) > horizon {
                    return true;
                }
                let shell = rho + (air / body.distance).max(px);
                facing <= 0.0 || s >= shell
            }
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
                let bound = cone_bound(body, &[0.0; MAX_FAR_MAPS]);
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
        let src = include_str!("../../shaders/far_table.slang");
        let mask = format!("uint tile_mask[{MAX_FAR_TILES}]");
        let index = format!("uint tile_index[{MAX_FAR_TILES}]");
        assert!(
            src.contains(&mask),
            "shader tile_mask length drifted from {MAX_FAR_TILES}"
        );
        assert!(
            src.contains(&index),
            "shader tile_index length drifted from {MAX_FAR_TILES}"
        );
        assert!(src.contains("uint4 list_header"));
        assert!(
            src.contains("float horizon_sin[512]"),
            "horizon table length drifted from 2 * {HORIZON_BINS}"
        );
        assert!(src.contains("uint4 horizon_id"));
        let vert = include_str!("../../shaders/sky_tile.vert.slang");
        assert!(
            vert.contains("(xf / float(width)) * 2.0 - 1.0"),
            "tile ndc x drifted from framebuffer_ndc"
        );
        assert!(
            vert.contains("1.0 - (yf / float(height)) * 2.0"),
            "tile ndc y drifted from framebuffer_ndc"
        );
        assert!(vert.contains("min(x0 + tilePx, width)"));
    }

    #[test]
    fn tile_lists_partition_in_order_and_clamp_partial_edges() {
        let mut table = FarTableGpu::zeroed();
        // 100×70 is not a multiple of 64: 2×2 tiles, the right column is 36 px
        // and the bottom row is 6 px.
        let width = 100u32;
        let height = 70u32;
        let tile_px = 64u32;
        let tiles_x = width.div_ceil(tile_px);
        let tiles_y = height.div_ceil(tile_px);
        assert_eq!((tiles_x, tiles_y), (2, 2));
        table.header = [3, tile_px, tiles_x, tiles_y];
        // Kept 0 is a sphere, kept 2 is a cube. Tile 1 carries the sphere,
        // tile 3 carries the cube.
        table.body[0].atmosphere[3] = 1.0;
        table.body[2].atmosphere[3] = 0.0;
        table.tile_mask[0] = 0;
        table.tile_mask[1] = 0b001;
        table.tile_mask[2] = 0;
        table.tile_mask[3] = 0b100;
        fill_tile_lists(&mut table, width, height);

        assert_eq!(table.list_header, [2, 2, width, height]);
        assert_eq!(&table.tile_index[..2], &[0, 2]);
        assert_eq!(table.tile_index[2], 1);
        assert_eq!(table.tile_index[3], 3);
        let n = used_tiles(&table);
        let n_base = table.list_header[0] as usize;
        let n_full = table.list_header[1] as usize;
        assert_eq!(n_base + n_full, n);
        let base = &table.tile_index[..n_base];
        let full = &table.tile_index[n_base..n];
        assert!(base.windows(2).all(|w| w[0] < w[1]));
        assert!(full.windows(2).all(|w| w[0] < w[1]));
        for &i in base {
            assert_eq!(table.tile_mask[i as usize], 0, "base tile {i}");
        }
        for &i in full {
            assert_ne!(table.tile_mask[i as usize], 0, "full tile {i}");
        }

        assert_eq!(
            tile_rect(0, tile_px, tiles_x, width, height),
            [0, 0, 64, 64]
        );
        assert_eq!(
            tile_rect(1, tile_px, tiles_x, width, height),
            [64, 0, 100, 64]
        );
        assert_eq!(
            tile_rect(2, tile_px, tiles_x, width, height),
            [0, 64, 64, 70]
        );
        assert_eq!(
            tile_rect(3, tile_px, tiles_x, width, height),
            [64, 64, 100, 70]
        );
        // The clamped corner sits on the same NDC edge as the fullscreen triangle.
        let [x0, y0, x1, y1] = tile_rect(3, tile_px, tiles_x, width, height);
        let _ = (x0, y0);
        let edge = framebuffer_ndc(x1 as f32, y1 as f32, width, height);
        assert_eq!(edge[0].to_bits(), 1.0f32.to_bits());
        assert_eq!(edge[1].to_bits(), (-1.0f32).to_bits());
        let origin = framebuffer_ndc(0.0, 0.0, width, height);
        assert_eq!(origin[0].to_bits(), (-1.0f32).to_bits());
        assert_eq!(origin[1].to_bits(), 1.0f32.to_bits());

        let draw = SkyDraw::from_table(&table, true);
        assert!(draw.quads);
        assert_eq!(
            (draw.n_base, draw.n_sphere, draw.n_heavy, draw.n_full),
            (2, 1, 1, 2)
        );
        assert_eq!(draw.n_coarse, 0);
        assert_eq!(draw.n_coarse_far, 0);
        assert_eq!(draw.body, SkyBodyPipe::NoMap);

        // No kept bodies: one body-free fullscreen triangle, even with a grid.
        table.header[0] = 0;
        let empty = SkyDraw::from_table(&table, true);
        assert!(!empty.quads);
        assert!(empty.base);

        // Culling off, or no tile grid: one fullscreen triangle. A kept sphere
        // uses the sphere pipeline; a kept cube uses the no-mapped pipeline.
        table.header[0] = 1;
        table.body[0].atmosphere[3] = 1.0;
        let unculled = SkyDraw::from_table(&table, false);
        assert!(!unculled.quads && !unculled.base);
        assert_eq!(unculled.body, SkyBodyPipe::Sphere);
        table.body[0].atmosphere[3] = 0.0;
        let cube = SkyDraw::from_table(&table, false);
        assert_eq!(cube.body, SkyBodyPipe::NoMap);
        table.body[0].atmosphere[3] = 4.0;
        let mapped = SkyDraw::from_table(&table, false);
        assert_eq!(mapped.body, SkyBodyPipe::Full);
        table.header[1] = 0;
        let no_grid = SkyDraw::from_table(&table, true);
        assert!(!no_grid.quads && !no_grid.base);
        assert_eq!(no_grid.body, SkyBodyPipe::Full);
    }

    #[test]
    fn packed_view_lists_match_the_masks() {
        let width = 100u32;
        let height = 70u32;
        let view = view_along_neg_z(60.0, width, height);
        let body = placed(-Vec3::Z, 0.05, FarShape::Sphere, 1);
        let table = pack_table(std::slice::from_ref(&body), Some(view));
        assert_eq!(table.header[1], 64);
        let n = used_tiles(&table);
        assert!(n > 1);
        let n_base = table.list_header[0] as usize;
        let n_full = table.list_header[1] as usize;
        assert_eq!(n_base + n_full, n);
        assert_eq!(table.list_header[2], width);
        assert_eq!(table.list_header[3], height);
        assert_eq!(n_full as u64, nonzero_tiles(&table));
        let mut seen = vec![false; n];
        for (slot, &index) in table.tile_index[..n].iter().enumerate() {
            let index = index as usize;
            assert!(index < n && !seen[index], "index {index} repeated");
            seen[index] = true;
            let base = slot < n_base;
            assert_eq!(table.tile_mask[index] == 0, base);
            if slot > 0 && slot != n_base {
                assert!(table.tile_index[slot - 1] < table.tile_index[slot]);
            }
        }
        assert!(seen.iter().all(|s| *s));
        let corner = (n as u32) - 1;
        let rect = tile_rect(corner, table.header[1], table.header[2], width, height);
        assert_eq!(rect[2], width);
        assert_eq!(rect[3], height);
        assert!(rect[2] - rect[0] < 64 || rect[3] - rect[1] < 64);
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
        let (from_cache, _) = pack_table_cached(
            std::slice::from_ref(&planet),
            Some(&up),
            Some(&cache),
            &[0.0; MAX_FAR_MAPS],
            &[0.0; MAX_FAR_MAPS],
            Vec3::Y,
        );
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

    fn mapped_down(distance: f32, radius: f32, horizon: f32, air: f32) -> FarBody {
        FarBody {
            dir: -Vec3::Y,
            distance,
            radius,
            shape: FarShape::Mapped {
                map: FarMapId(0),
                horizon,
                air,
            },
            rotation: Quat::IDENTITY,
            albedo: [LinearRgb([0.2, 0.2, 0.2]); 6],
            atmosphere: LinearRgb([0.1, 0.2, 0.3]),
            seed: 7,
        }
    }

    /// The per-pixel cone stays the hi-sphere sentinel. Tiles use the horizon
    /// cone widened by the air limb, so a gate pixel and a limb pixel keep the
    /// bit and the sky above that cone does not.
    #[test]
    fn mapped_horizon_tiles_cover_the_gate_and_leave_the_sky_clear() {
        let mut map_max = [0.0f32; MAX_FAR_MAPS];
        // (radius + max) / distance = 11/4 > 0.99, so cone.w is the sentinel.
        map_max[0] = 10.0;
        let distance = 4.0f32;
        let horizon = 0.04f32;
        // 0.15 rad of air is several 64 px tiles at 1440p, so the limb pulls
        // in tiles the raw horizon cone misses.
        let air = 0.6f32;
        let body = mapped_down(distance, 1.0, horizon, air);

        let wide = view_pitched(0.0, 70.0, 3440, 1440);
        let disabled = mapped_down(distance, 1.0, 1.0, air);
        let all = super::pack_table(std::slice::from_ref(&disabled), Some(wide), &map_max);
        assert_eq!(all.cone[0][3].to_bits(), (-1.0f32).to_bits());
        let wide_tiles = used_tiles(&all);
        assert_eq!(wide_tiles, 1242);
        assert!(
            all.tile_mask[..wide_tiles].iter().all(|mask| mask & 1 != 0),
            "horizon >= 1 must keep today's full-sky mask"
        );

        let level = super::pack_table(std::slice::from_ref(&body), Some(wide), &map_max);
        assert_eq!(level.header[0], 1);
        assert_eq!(
            level.cone[0][3].to_bits(),
            (-1.0f32).to_bits(),
            "the per-pixel cone is still the sentinel"
        );
        let painted = level.tile_mask[..wide_tiles]
            .iter()
            .filter(|mask| *mask & 1 != 0)
            .count();
        assert!(
            painted > 0 && painted < wide_tiles,
            "painted {painted} of {wide_tiles}"
        );

        let (sin_air, cos_air) =
            mapped_horizon_half(horizon, air, distance, wide.px_max).expect("widened cone");
        let (sin_raw, cos_raw) =
            mapped_horizon_half(horizon, 0.0, distance, wide.px_max).expect("raw cone");
        let frames = TileFrames::build(&wide).expect("frames");
        let basis = ViewBasis::from_view_proj(wide.view_proj).expect("basis");
        let dir_view = basis.to_view(-Vec3::Y);
        let mut limb_tiles = 0u32;
        let mut clear_tiles = 0u32;
        for (i, tile) in frames.samples.iter().enumerate() {
            let in_air = tile_within(tile, dir_view, sin_air, cos_air);
            let in_raw = tile_within(tile, dir_view, sin_raw, cos_raw);
            if in_air && !in_raw {
                assert_ne!(level.tile_mask[i] & 1, 0, "limb tile {i} missed the bit");
                limb_tiles += 1;
            }
            if !in_air {
                assert_eq!(level.tile_mask[i] & 1, 0, "sky tile {i} kept the bit");
                clear_tiles += 1;
            }
        }
        assert!(
            limb_tiles > 0,
            "the air limb painted no tile past the raw gate"
        );
        assert!(clear_tiles > 0, "no tile sits outside the horizon cone");

        // 85° up, 50° vertical fov: the whole view is above the widened cone.
        let steep = view_pitched(85.0, 50.0, 800, 600);
        let culled = super::pack_table(std::slice::from_ref(&body), Some(steep), &map_max);
        assert_eq!(culled.header[0], 0, "a sky-only view kept the planet");

        let mut gate_pixels = 0u32;
        let mut limb_pixels = 0u32;
        let frames_px = [
            (40.0f32, 60.0f32, 382u32, 160u32),
            (0.0, 70.0, 382, 160),
            (0.0, 90.0, 640, 360),
            (-25.0, 70.0, 640, 360),
            (-70.0, 60.0, 382, 160),
            (0.0, 70.0, 3440, 1440),
            (30.0, 70.0, 3440, 1440),
            (85.0, 50.0, 800, 600),
        ];
        for &(pitch, fovy, w, h) in &frames_px {
            let view = view_pitched(pitch, fovy, w, h);
            let table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
            let Some((_, cos_b)) = mapped_horizon_half(horizon, air, distance, view.px_max) else {
                continue;
            };
            let kept = table.header[0] == 1;
            if kept {
                assert_eq!(table.cone[0][3].to_bits(), (-1.0f32).to_bits());
            }
            let tile_px = table.header[1];
            let tiles_x = table.header[2];
            let n_tiles = used_tiles(&table);
            let inv = view.view_proj.inverse();
            for y in 0..h {
                for x in 0..w {
                    let ray = ray_at(&inv, w, h, x as f32 + 0.5, y as f32 + 0.5);
                    if !ray.is_finite() {
                        continue;
                    }
                    // Shader gate: `dot(ray, -dir) <= horizon`. dir is −Y.
                    let gate = ray.dot(Vec3::Y) <= horizon;
                    let along = ray.dot(-Vec3::Y);
                    let in_limb = along >= cos_b + 1.0e-4;
                    if !gate && !in_limb {
                        continue;
                    }
                    assert!(
                        kept,
                        "pitch {pitch} fov {fovy} {w}x{h} dropped a ray inside the cone"
                    );
                    let tile_i = tile_at(x as f32 + 0.5, y as f32 + 0.5, tile_px, tiles_x);
                    assert!(
                        tile_i < n_tiles && table.tile_mask[tile_i] & 1 != 0,
                        "pitch {pitch} fov {fovy} {w}x{h} pixel {x},{y} gate {gate} along {along} cos {cos_b}"
                    );
                    if gate {
                        gate_pixels += 1;
                    } else {
                        limb_pixels += 1;
                    }
                }
            }
        }
        assert!(gate_pixels > 1000, "gate scan never passed, {gate_pixels}");
        assert!(
            limb_pixels > 0,
            "no pixel sat in the air limb past the raw gate"
        );

        // Finite per-pixel sine. The horizon cone is wider, so a gate ray the
        // pixel cone rejects still carries the bit.
        map_max[0] = 0.0;
        let far = mapped_down(10.0, 1.0, -0.2, 0.05);
        let down = view_pitched(-40.0, 70.0, 640, 360);
        let finite = super::pack_table(std::slice::from_ref(&far), Some(down), &map_max);
        assert_eq!(finite.header[0], 1);
        let bound = finite.cone[0][3];
        assert!(
            bound > 0.0 && bound < 0.2,
            "expected the hi-sphere sine, got {bound}"
        );
        let inv = down.view_proj.inverse();
        let tile_px = finite.header[1];
        let tiles_x = finite.header[2];
        let mut gate = 0u32;
        let mut past_pixel_cone = 0u32;
        for y in 0..down.height {
            for x in 0..down.width {
                let (ray, px) = ray_and_px(
                    &inv,
                    down.width,
                    down.height,
                    x as f32 + 0.5,
                    y as f32 + 0.5,
                );
                if !ray.is_finite() || ray.dot(Vec3::Y) > -0.2 {
                    continue;
                }
                gate += 1;
                let tile_i = tile_at(x as f32 + 0.5, y as f32 + 0.5, tile_px, tiles_x);
                assert!(
                    finite.tile_mask[tile_i] & 1 != 0,
                    "gate pixel {x},{y} missed the horizon-cone bit"
                );
                if !shader_accepts(ray, finite.cone[0], px) {
                    past_pixel_cone += 1;
                }
            }
        }
        assert!(gate > 100, "finite-cone view held no gate pixel");
        assert!(
            past_pixel_cone > 0,
            "the horizon cone was no wider than the per-pixel sine"
        );

        // horizon >= 1 with a finite sine keeps today's disc, not the whole sky.
        // The disc is ~6° around nadir, so the view has to look nearly straight down.
        let nadir = view_pitched(-85.0, 40.0, 640, 360);
        let off = mapped_down(10.0, 1.0, 1.0, 0.0);
        let today = super::pack_table(std::slice::from_ref(&off), Some(nadir), &map_max);
        assert_eq!(today.header[0], 1);
        assert!(today.cone[0][3] > 0.0);
        let n = used_tiles(&today);
        let painted_today = today.tile_mask[..n].iter().filter(|m| *m & 1 != 0).count();
        assert!(
            painted_today > 0 && painted_today < n,
            "horizon >= 1 painted {painted_today} of {n}"
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

    fn coarse_query(sun: Vec3, stars: bool) -> SkyCoarseQuery {
        let (sun_cos_rim, moon_cos_rim) = crate::vk::pipeline::SkyParams::disc_rims(0.03);
        SkyCoarseQuery {
            sun_dir: sun,
            sun_cos_rim,
            moon_cos_rim,
            stars,
        }
    }

    /// 320×180 at 60° is 5×3 tiles of 64 px. The screen centre sits in tile 7.
    fn coarse_grid() -> (Box<FarTableGpu>, FarView, TileFrames) {
        let width = 320u32;
        let height = 180u32;
        let view = view_along_neg_z(60.0, width, height);
        let frames = TileFrames::build(&view).expect("view basis");
        assert!(frames.matches(&view));
        let (tile_px, tiles_x, tiles_y) = tile_layout(width, height);
        assert_eq!((tile_px, tiles_x, tiles_y), (64, 5, 3));
        let mut table = zeroed_table();
        table.header = [0, tile_px, tiles_x, tiles_y];
        fill_tile_lists(&mut table, width, height);
        let n = (tiles_x * tiles_y) as usize;
        assert_eq!(table.list_header[0] as usize, n);
        assert!(table.tile_index[..n].windows(2).all(|w| w[0] < w[1]));
        (table, view, frames)
    }

    fn assert_disc_split(sun: Vec3, center_coarse: bool) {
        let (mut table, view, frames) = coarse_grid();
        let n_base = table.list_header[0];
        let before = table.tile_index[..n_base as usize].to_vec();
        let header = table.list_header;
        let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(sun, false));
        assert_eq!(table.list_header, header);
        assert!(n_coarse > 0 && n_coarse < n_base);
        let coarse = &table.tile_index[..n_coarse as usize];
        let fine = &table.tile_index[n_coarse as usize..n_base as usize];
        assert!(coarse.windows(2).all(|w| w[0] < w[1]));
        assert!(fine.windows(2).all(|w| w[0] < w[1]));
        let mut merged = table.tile_index[..n_base as usize].to_vec();
        merged.sort_unstable();
        assert_eq!(merged, before);
        let center = 7u32;
        assert_eq!(tile_at(160.0, 90.0, 64, 5), center as usize);
        let center_in_coarse = coarse.contains(&center);
        assert_eq!(center_in_coarse, center_coarse);
        assert!(coarse.contains(&0), "corner tile must stay coarse");
        assert!(!fine.contains(&0));
    }

    #[test]
    fn coarse_base_excludes_a_tile_a_disc_touches() {
        // Sun on the view axis: the centre tile meets the sun disc, the moon
        // is behind the camera. Corner tile 0 is many degrees off both discs.
        assert_disc_split(-Vec3::Z, false);
        // Sun behind the camera puts the moon on the view axis.
        assert_disc_split(Vec3::Z, false);
    }

    #[test]
    fn coarse_base_takes_every_tile_when_both_discs_miss() {
        // +Y is 90° off a 60° view along −Z. Both discs sit outside the frustum.
        let (mut table, view, frames) = coarse_grid();
        let n_base = table.list_header[0];
        let before = table.tile_index[..n_base as usize].to_vec();
        let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(Vec3::Y, false));
        assert_eq!(n_coarse, n_base);
        assert_eq!(&table.tile_index[..n_base as usize], before.as_slice());
    }

    #[test]
    fn stars_disable_coarse_and_leave_the_base_run() {
        let (mut table, view, frames) = coarse_grid();
        let n_base = table.list_header[0] as usize;
        let before = table.tile_index[..n_base].to_vec();
        let n_coarse = split_coarse_base(&mut table, &frames, &view, &coarse_query(-Vec3::Z, true));
        assert_eq!(n_coarse, 0);
        assert_eq!(&table.tile_index[..n_base], before.as_slice());
        // Day, no star floor: the product is zero even with the gain left on.
        assert!(!stars_drawn(0.0, 0.0, 1.0));
        assert!(stars_drawn(1.0, 0.0, 1.0));
        assert!(stars_drawn(0.0, 0.4, 1.0));
        assert!(!stars_drawn(1.0, 1.0, 0.0));
        assert!(stars_drawn(f32::NAN, 0.0, 1.0));
        // A non-finite rim shades nothing coarse and does not reorder.
        let (mut table, view, frames) = coarse_grid();
        let mut query = coarse_query(-Vec3::Z, false);
        query.sun_cos_rim = f32::NAN;
        let n_coarse = split_coarse_base(&mut table, &frames, &view, &query);
        assert_eq!(n_coarse, 0);
        assert_eq!(&table.tile_index[..n_base], before.as_slice());
    }

    #[test]
    fn coarse_split_keeps_the_three_way_partition() {
        let (mut table, view, frames) = coarse_grid();
        table.header[0] = 2;
        table.body[0].atmosphere[3] = 1.0;
        table.body[1].atmosphere[3] = 0.0;
        // Tile 1 is sphere-only, tile 3 meets a cube. The centre (tile 7) stays
        // base, so the sun on −Z pulls it into the fine suffix and the base
        // order actually changes.
        table.tile_mask[1] = 0b001;
        table.tile_mask[3] = 0b010;
        fill_tile_lists(&mut table, view.width, view.height);
        let n_base = table.list_header[0] as usize;
        let n_full = table.list_header[1] as usize;
        assert_eq!(n_base + n_full, 15);
        assert_eq!(n_full, 2);
        let base_before = table.tile_index[..n_base].to_vec();
        let full_before = table.tile_index[n_base..n_base + n_full].to_vec();
        let header = table.list_header;
        let n_coarse =
            split_coarse_base(&mut table, &frames, &view, &coarse_query(-Vec3::Z, false));
        assert!(n_coarse > 0 && (n_coarse as usize) < n_base);
        assert_eq!(table.list_header, header);
        assert_eq!(
            &table.tile_index[n_base..n_base + n_full],
            full_before.as_slice()
        );
        let mut merged = table.tile_index[..n_base].to_vec();
        merged.sort_unstable();
        assert_eq!(merged, base_before);
        let coarse = &table.tile_index[..n_coarse as usize];
        let fine = &table.tile_index[n_coarse as usize..n_base];
        assert!(coarse.windows(2).all(|w| w[0] < w[1]));
        assert!(fine.windows(2).all(|w| w[0] < w[1]));
        assert!(fine.contains(&7), "the sunlit centre tile stays at 1x1");
        for &index in coarse {
            assert_eq!(table.tile_mask[index as usize], 0);
        }
        for &index in fine {
            assert_eq!(table.tile_mask[index as usize], 0);
        }
    }

    fn heavy_run(table: &FarTableGpu) -> (usize, usize) {
        let (n_base, n_sphere, n_heavy) = tile_split(table);
        let start = (n_base + n_sphere) as usize;
        (start, start + n_heavy as usize)
    }

    #[test]
    fn coarse_far_is_the_mapped_interior_only() {
        let view = view_pitched(0.0, 90.0, 1280, 720);
        let frames = TileFrames::build(&view).expect("frames");
        let map_max = [0.0f32; MAX_FAR_MAPS];
        let body = mapped_down(4.0, 1.0, 0.0, 0.0);
        let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        assert_eq!(table.header[0], 1);
        let (start, end) = heavy_run(&table);
        let n_heavy = (end - start) as u32;
        assert!(n_heavy > 1, "heavy {n_heavy}");
        let before = table.tile_index[start..end].to_vec();
        let header = table.list_header;
        // Sun behind the camera. The moon sits on the view axis, on the horizon,
        // so it only knocks out tiles near the limb.
        let map_min = [0.0f32; MAX_FAR_MAPS];
        let n = split_coarse_far(
            &mut table,
            &frames,
            &view,
            &coarse_query(Vec3::Z, false),
            &map_min,
        );
        assert!(n > 0 && n < n_heavy, "coarse {n} of {n_heavy}");
        assert_eq!(table.list_header, header);
        let coarse = &table.tile_index[start..start + n as usize];
        let fine = &table.tile_index[start + n as usize..end];
        assert!(coarse.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(fine.windows(2).all(|pair| pair[0] < pair[1]));
        let mut merged = table.tile_index[start..end].to_vec();
        merged.sort_unstable();
        let mut expect = before.clone();
        expect.sort_unstable();
        assert_eq!(merged, expect);

        let (sin_b, cos_b) =
            mapped_interior_half(0.0, 0.0, 4.0, view.px_max).expect("interior cone");
        let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
        let dir_view = basis.to_view(-Vec3::Y).normalize();
        let mut boundary = 0u32;
        for &index in &before {
            let tile = &frames.samples[index as usize];
            let inside = tile_strictly_inside(tile, dir_view, sin_b, cos_b);
            if inside {
                assert!(
                    coarse.contains(&index),
                    "interior tile {index} stayed at 1x1"
                );
            } else {
                assert!(fine.contains(&index), "limb tile {index} went coarse");
                boundary += 1;
            }
        }
        assert!(boundary > 0, "every heavy tile was interior");

        // horizon >= 1 disables the horizon disc. The lo sphere still admits
        // tiles: min offset 0 makes rho_lo = rho = 0.75, and a downward view
        // puts the centre inside that disc while the corners stay out.
        let disabled = mapped_down(4.0, 3.0, 1.0, 0.0);
        let down = view_pitched(-50.0, 70.0, 640, 360);
        let down_frames = TileFrames::build(&down).expect("down frames");
        let mut off = super::pack_table(std::slice::from_ref(&disabled), Some(down), &map_max);
        let (off_start, off_end) = heavy_run(&off);
        assert!(off_end > off_start, "horizon >= 1 painted no heavy tile");
        let off_before = off.tile_index[off_start..off_end].to_vec();
        let n_lo = split_coarse_far(
            &mut off,
            &down_frames,
            &down,
            &coarse_query(Vec3::Y, false),
            &map_min,
        );
        assert!(n_lo > 0 && (n_lo as usize) < off_end - off_start);
        let (sin_lo, cos_lo) = lo_disc_interior(0.75, down.px_max).expect("lo disc");
        let down_basis = ViewBasis::from_view_proj(down.view_proj).expect("down basis");
        let down_dir = down_basis.to_view(-Vec3::Y).normalize();
        let lo_coarse = &off.tile_index[off_start..off_start + n_lo as usize];
        let lo_fine = &off.tile_index[off_start + n_lo as usize..off_end];
        for &index in &off_before {
            let tile = &down_frames.samples[index as usize];
            // Stars and discs do not knock an interior tile out: the hit
            // covers them. The sun sits on +Y here and would have done so.
            let inside = tile_strictly_inside(tile, down_dir, sin_lo, cos_lo);
            if inside {
                assert!(
                    lo_coarse.contains(&index),
                    "lo-disc tile {index} stayed 1x1"
                );
            } else {
                assert!(lo_fine.contains(&index), "outside tile {index} went coarse");
            }
        }

        // A second body on the same tiles drops those tiles out of the prefix.
        // Bottom-centre of this view, inside the mapped hemisphere and on screen.
        let companion = placed(Vec3::new(0.0, -1.0, -1.0), 0.15, FarShape::Sphere, 2);
        let mut both = super::pack_table(&[body, companion], Some(view), &map_max);
        assert_eq!(both.header[0], 2);
        let n_both = split_coarse_far(
            &mut both,
            &frames,
            &view,
            &coarse_query(Vec3::Z, false),
            &map_min,
        );
        assert!(n_both > 0, "the companion erased every interior tile");
        let (both_start, both_end) = heavy_run(&both);
        let both_coarse = &both.tile_index[both_start..both_start + n_both as usize];
        let mut shared = 0u32;
        for &index in &both.tile_index[both_start..both_end] {
            let mask = both.tile_mask[index as usize];
            if mask.count_ones() != 1 {
                shared += 1;
                assert!(
                    !both_coarse.contains(&index),
                    "shared tile {index} mask {mask:#x} went coarse"
                );
            }
        }
        assert!(shared > 0, "the companion shared no tile");
        for &index in both_coarse {
            assert_eq!(both.tile_mask[index as usize], 1);
        }

        // A sun disc aimed down the body used to empty the prefix. The surface
        // hit replaces the disc, so the interior still goes coarse.
        let mut covered = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        let mut query = coarse_query(-Vec3::Y, true);
        query.sun_cos_rim = 0.0;
        assert!(
            split_coarse_far(&mut covered, &frames, &view, &query, &map_min) > 0,
            "a sun over the interior suppressed every coarse tile"
        );
    }

    /// Daytime with the star floor up, and the sun disc covering the ground
    /// body's interior. Tiles below the horizon still join the 2×2 run.
    #[test]
    fn stars_and_a_sun_in_view_still_coarse_the_interior() {
        let view = view_pitched(0.0, 90.0, 1280, 720);
        let frames = TileFrames::build(&view).expect("frames");
        let map_max = [0.0f32; MAX_FAR_MAPS];
        let map_min = [0.0f32; MAX_FAR_MAPS];
        let body = mapped_down(4.0, 1.0, 0.0, 0.0);
        let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        let (start, end) = heavy_run(&table);
        assert!(end > start);
        let mut query = coarse_query(-Vec3::Y, true);
        // cos 0 is a 90° rim: the sun disc meets every tile that sees the body.
        query.sun_cos_rim = 0.0;
        let n = split_coarse_far(&mut table, &frames, &view, &query, &map_min);
        assert!(n > 0, "stars and the sun left the interior at 1x1");
        let (sin_b, cos_b) =
            mapped_interior_half(0.0, 0.0, 4.0, view.px_max).expect("interior cone");
        let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
        let dir_view = basis.to_view(-Vec3::Y).normalize();
        let coarse = &table.tile_index[start..start + n as usize];
        let mut interiors = 0u32;
        for &index in &table.tile_index[start..end] {
            let tile = &frames.samples[index as usize];
            if tile_strictly_inside(tile, dir_view, sin_b, cos_b) {
                interiors += 1;
                assert!(
                    coarse.contains(&index),
                    "interior tile {index} stayed at 1x1 under stars and the sun"
                );
            }
        }
        assert!(interiors > 0, "no tile sat below the horizon");
    }

    /// Every pixel of a lo-sphere interior tile meets the datum. Altitudes run
    /// from 10 m to 1e6 m above the lo sphere; the horizon lane stays at 1 so
    /// only the lo disc can admit a tile. Stars stay off here; a disc no longer
    /// removes an interior tile.
    #[test]
    fn lo_interior_pixels_hit_the_mapped_surface() {
        let radius = 31_017_520.0f32;
        let g = 9u32;
        let gg = g as usize;
        let mut datum = vec![0.0f32; 6 * gg * gg];
        for face in 0..6usize {
            let (tu, n, tv) = crate::far_body::far_map_basis(face);
            for j in 0..g {
                for i in 0..g {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    let bump = radius
                        * (0.004 * (d.x * 3.0).sin() * (d.y * 2.0 + 0.4).cos()
                            + 0.002 * (d.z * 5.0).sin());
                    datum[face * gg * gg + j as usize * gg + i as usize] = bump;
                }
            }
        }
        let min_off = datum.iter().copied().fold(f32::INFINITY, f32::min);
        let max_off = datum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            min_off < 0.0 && max_off > 0.0,
            "relief {min_off}..{max_off}"
        );
        // Point the deepest sample at the eye. The eye is then `altitude`
        // above the surface as well as above the lo sphere, so it is outside
        // the star body and a lo-disc ray has a forward hit.
        let mut valley = Vec3::Y;
        let mut valley_off = f32::INFINITY;
        for face in 0..6usize {
            let (tu, n, tv) = crate::far_body::far_map_basis(face);
            for j in 0..g {
                for i in 0..g {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    let off = datum[face * gg * gg + j as usize * gg + i as usize];
                    if off < valley_off {
                        valley_off = off;
                        valley = d;
                    }
                }
            }
        }
        let valley_rot = Quat::from_rotation_arc(valley, Vec3::Y);
        let mut map_min = [0.0f32; MAX_FAR_MAPS];
        let mut map_max = [0.0f32; MAX_FAR_MAPS];
        map_min[0] = min_off;
        map_max[0] = max_off;

        let mut state = 0xA11C_E5EDu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let unit = |rng: &mut dyn FnMut() -> u32| rng() as f32 / u32::MAX as f32;
        // 10 m through 1e6 m, plus a couple of draws inside that range.
        let mut altitudes = vec![10.0f32, 1_000_000.0];
        for _ in 0..6 {
            let t = unit(&mut next);
            // Log span so both the near ground and the far approach show up.
            let log_h = 10.0f32.ln() + t * ((1.0e6f32).ln() - 10.0f32.ln());
            altitudes.push(log_h.exp());
        }
        let width = 192u32;
        let height = 108u32;
        let mut checked = 0u32;
        let mut interiors = 0u32;
        for (trial, &altitude) in altitudes.iter().enumerate() {
            let pitch = -75.0 + unit(&mut next) * 50.0;
            let yaw = unit(&mut next) * std::f32::consts::TAU;
            let distance = radius + min_off + altitude;
            assert!(distance > 0.0);
            let mut body = mapped_down(distance, radius, 1.0, 0.0);
            // Yaw around the valley so the relief turns under a fixed eye.
            body.rotation = Quat::from_rotation_y(yaw) * valley_rot;
            let rho_lo = (radius + min_off) / distance;
            assert!(
                rho_lo > 0.0 && rho_lo < 1.0,
                "alt {altitude} rho_lo {rho_lo}"
            );
            let view = view_pitched(pitch, 70.0, width, height);
            let Some(frames) = TileFrames::build(&view) else {
                continue;
            };
            let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
            if table.header[0] == 0 {
                continue;
            }
            let (start, end) = heavy_run(&table);
            if end <= start {
                continue;
            }
            let n = split_coarse_far(
                &mut table,
                &frames,
                &view,
                &coarse_query(Vec3::X, false),
                &map_min,
            );
            if n == 0 {
                continue;
            }
            interiors += n;
            let inv = view.view_proj.inverse();
            let tile_px = table.header[1];
            let tiles_x = table.header[2];
            let rho = radius / distance;
            for &index in &table.tile_index[start..start + n as usize] {
                let [x0, y0, x1, y1] = tile_rect(index, tile_px, tiles_x, width, height);
                for y in y0..y1 {
                    for x in x0..x1 {
                        let ray = ray_at(&inv, width, height, x as f32 + 0.5, y as f32 + 0.5);
                        if !ray.is_finite() {
                            continue;
                        }
                        let hit = ray_mapped_fast(
                            ray,
                            body.dir,
                            rho,
                            distance,
                            body.rotation,
                            1.0,
                            g,
                            &datum,
                            min_off,
                            max_off,
                        );
                        let facing = ray.dot(body.dir);
                        let lo_disc = facing * facing - (1.0 - rho_lo * rho_lo);
                        assert!(
                            hit.is_some(),
                            "trial {trial} alt {altitude} pitch {pitch:.1} pixel {x},{y} tile {index} missed; facing {facing} rho {rho} rho_lo {rho_lo} lo_disc {lo_disc} ray {ray:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(interiors > 0, "no lo-sphere interior tile in the sweep");
        assert!(checked > 1000, "checked only {checked} interior pixels");
    }

    #[test]
    fn packed_horizon_tables_do_not_name_a_body() {
        let table = pack_table(&[], None);
        assert_eq!(table.horizon_id[0], u32::MAX);
        assert_eq!(table.horizon_id[1], u32::MAX);
        assert_eq!(table.horizon_id[2], HORIZON_BINS as u32);
        assert!(table.horizon_sin.iter().all(|s| *s == 1.0));
    }

    #[test]
    fn flat_horizon_puts_the_zenith_above_and_keeps_the_nadir() {
        let g = 2u32;
        let datum = vec![0.0f32; 24];
        let radius = 1_000.0f32;
        let air = 10.0f32;
        let distance = radius + air + 100.0;
        let up = Vec3::Y;
        let bins = build_horizon_bins(g, &datum, Quat::IDENTITY, up, radius, distance, air, 0.0);
        let (east, north) = horizon_axes(up).expect("axes");
        assert!(ray_above_horizon(&bins, up, east, north, up));
        assert!(!ray_above_horizon(&bins, up, east, north, -up));
        let rho = radius / distance;
        let hit = ray_mapped(
            -up,
            -up,
            rho,
            distance,
            Quat::IDENTITY,
            1.0,
            g,
            &datum,
            0.0,
            0.0,
        );
        assert!(
            hit.is_some(),
            "nadir misses a flat sphere outside the hi shell"
        );
    }

    #[test]
    fn horizon_cache_keeps_a_small_move_and_drops_rotation() {
        let mut slot = HorizonSlot::empty();
        slot.live = true;
        slot.map = 1;
        slot.generation = 4;
        slot.radius_bits = 10.0f32.to_bits();
        slot.air_bits = 2.0f32.to_bits();
        slot.px_bits = 0.0f32.to_bits();
        slot.rot_bits = [0, 0, 0, 1.0f32.to_bits()];
        slot.eye = Vec3::new(0.0, 1_000.0, 0.0);
        slot.altitude = 1_000.0;
        let eye = slot.eye;
        let rot = slot.rot_bits;
        assert!(horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            rot,
            eye,
        ));
        // 0.05% of the altitude stays cached. 0.2% rebuilds.
        let near = eye + Vec3::Y * (0.0005 * slot.altitude);
        assert!(horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            rot,
            near,
        ));
        let far = eye + Vec3::Y * (0.002 * slot.altitude);
        assert!(!horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            rot,
            far,
        ));
        let mut spun = rot;
        spun[0] ^= 1;
        assert!(!horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            spun,
            eye,
        ));
        slot.altitude = 0.0;
        assert!(horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            rot,
            eye,
        ));
        assert!(!horizon_cache_hit(
            &slot,
            1,
            4,
            slot.radius_bits,
            slot.air_bits,
            slot.px_bits,
            rot,
            near,
        ));
    }

    /// `horizon` is 1, so the scalar cone paints every tile. The eye is inside
    /// the hi+air ball only because one cell is a mountain; the local ground
    /// is below the eye. Tiles looking up lose the bit. The downward tile keeps it.
    #[test]
    fn upward_tiles_lose_the_bit_above_the_horizon_table() {
        let radius = 10_000.0f32;
        let air = 10.0f32;
        let altitude = 100.0f32;
        let g = 9u32;
        let gg = g as usize;
        let mut datum = vec![0.0f32; 6 * gg * gg];
        // +X face centre cell. Far from a view that looks along −Z.
        let mountain = 2_000.0f32;
        datum[0 * gg * gg + 4 * gg + 4] = mountain;
        let distance = radius + altitude;
        assert!(distance < radius + mountain + air);
        let mut body = mapped_down(distance, radius, 1.0, air);
        body.rotation = Quat::IDENTITY;
        let mut map_max = [0.0f32; MAX_FAR_MAPS];
        map_max[0] = mountain;
        let view = view_pitched(0.0, 70.0, 640, 360);
        let frames = TileFrames::build(&view).expect("tiles");
        let mut table = super::pack_table(std::slice::from_ref(&body), Some(view), &map_max);
        assert_eq!(table.header[0], 1);
        assert!(
            table.tile_mask[..frames.samples.len()]
                .iter()
                .all(|m| *m == 1),
            "horizon 1 inside the hi ball paints every tile"
        );
        let up = -body.dir;
        let bins = build_horizon_bins(
            g,
            &datum,
            body.rotation,
            up,
            radius,
            distance,
            air,
            view.px_max,
        );
        table.horizon_id[0] = 0;
        table.horizon_sin[..HORIZON_BINS].copy_from_slice(&bins);
        assert!(clear_tiles_above_horizon(&mut table, &frames, &view));
        let basis = ViewBasis::from_view_proj(view.view_proj).expect("basis");
        let up_v = basis.to_view(up).normalize();
        let mut highest = 0usize;
        let mut lowest = 0usize;
        let mut hi = f32::NEG_INFINITY;
        let mut lo = f32::INFINITY;
        for (i, tile) in frames.samples.iter().enumerate() {
            let mu = tile.centre.dot(up_v);
            if mu > hi {
                hi = mu;
                highest = i;
            }
            if mu < lo {
                lo = mu;
                lowest = i;
            }
        }
        assert_eq!(
            table.tile_mask[highest], 0,
            "upward tile mu {hi} kept the bit"
        );
        assert_eq!(
            table.tile_mask[lowest] & 1,
            1,
            "downward tile mu {lo} lost the bit"
        );
    }

    /// No ray that hits the surface or the limb is classified above the table.
    /// Relief is a few percent of the radius. Eyes run from 10 m to 2e6 m
    /// above the lo sphere. `horizon` is 1 so the scalar plane does not reject.
    #[test]
    fn no_hit_or_limb_is_above_the_horizon_table() {
        let radius = 31_017_520.0f32;
        let mut state = 0xC0FF_EE01u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let unit = |rng: &mut dyn FnMut() -> u32| rng() as f32 / u32::MAX as f32;
        let mut above_ok = 0u32;
        let mut hits = 0u32;
        for trial in 0..4u32 {
            let g = if trial % 2 == 0 { 5u32 } else { 7u32 };
            let gg = g as usize;
            let mut datum = vec![0.0f32; 6 * gg * gg];
            let amp = 0.04 + unit(&mut next) * 0.04;
            for face in 0..6usize {
                let (tu, n, tv) = crate::far_body::far_map_basis(face);
                for j in 0..g {
                    for i in 0..g {
                        let edge = (g - 1) as f32;
                        let xi = 2.0 * i as f32 / edge - 1.0;
                        let eta = 2.0 * j as f32 / edge - 1.0;
                        let quarter = std::f32::consts::FRAC_PI_4;
                        let d = (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan())
                            .normalize();
                        let bump = radius
                            * amp
                            * ((d.x * 3.0 + trial as f32).sin() * (d.y * 2.0).cos()
                                + 0.5 * (d.z * 5.0 + 0.7).sin());
                        datum[face * gg * gg + j as usize * gg + i as usize] = bump;
                    }
                }
            }
            let min_off = datum.iter().copied().fold(f32::INFINITY, f32::min);
            let max_off = datum.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            assert!(min_off < 0.0 && max_off > min_off + radius * 0.02);
            let air = radius * 0.001;
            let yaw = unit(&mut next) * std::f32::consts::TAU;
            let rotation = Quat::from_rotation_y(yaw);
            let px = 2.0e-4f32;
            for alt_i in 0..6 {
                let t = alt_i as f32 / 5.0;
                let log_h = 10.0f32.ln() + t * ((2.0e6f32).ln() - 10.0f32.ln());
                let altitude = log_h.exp();
                let distance = radius + min_off + altitude;
                let rho = radius / distance;
                let dir = -Vec3::Y;
                let up = Vec3::Y;
                let bins = build_horizon_bins(g, &datum, rotation, up, radius, distance, air, px);
                let (east, north) = horizon_axes(up).expect("axes");
                let mut rays = Vec::new();
                rays.push(-up);
                rays.push(up);
                for _ in 0..24 {
                    let x = unit(&mut next) * 2.0 - 1.0;
                    let y = unit(&mut next) * 2.0 - 1.0;
                    let z = unit(&mut next) * 2.0 - 1.0;
                    let v = Vec3::new(x, y, z);
                    if v.length_squared() > 1.0e-6 {
                        rays.push(v.normalize());
                    }
                }
                for ray in rays {
                    let above = ray_above_horizon(&bins, up, east, north, ray);
                    let hit = ray_mapped(
                        ray, dir, rho, distance, rotation, 1.0, g, &datum, min_off, max_off,
                    );
                    let limb = ray_mapped_limb(
                        ray, dir, rho, distance, rotation, g, &datum, max_off, air, px,
                    );
                    if hit.is_some() {
                        hits += 1;
                    }
                    if above {
                        assert!(
                            hit.is_none() && !limb,
                            "trial {trial} alt {altitude} above a hit={hit:?} limb {limb} ray {ray:?}"
                        );
                        above_ok += 1;
                    }
                }
            }
        }
        assert!(hits > 20, "the sweep barely hit the surface ({hits})");

        // Flat underfoot, one mountain on +X. The eye is outside the local
        // air shell and still inside the mountain's hi ball, so the zenith
        // clears the table while a hit on the mountain does not.
        let g = 9u32;
        let gg = g as usize;
        let mut datum = vec![0.0f32; 6 * gg * gg];
        let mountain = radius * 0.1;
        datum[4 * gg + 4] = mountain;
        let air = 20_000.0f32;
        let rotation = Quat::IDENTITY;
        for altitude in [50_000.0f32, 2.0e6] {
            let distance = radius + altitude;
            assert!(distance > radius + air && distance < radius + mountain + air);
            let up = Vec3::Y;
            let bins = build_horizon_bins(g, &datum, rotation, up, radius, distance, air, 0.0);
            let (east, north) = horizon_axes(up).expect("axes");
            assert!(
                ray_above_horizon(&bins, up, east, north, up),
                "zenith at {altitude} was not above the local ground"
            );
            above_ok += 1;
            let rho = radius / distance;
            for ray in [-up, up, Vec3::X, Vec3::new(1.0, -0.2, 0.0).normalize()] {
                let above = ray_above_horizon(&bins, up, east, north, ray);
                let hit = ray_mapped(
                    ray, -up, rho, distance, rotation, 1.0, g, &datum, 0.0, mountain,
                );
                let limb = ray_mapped_limb(
                    ray, -up, rho, distance, rotation, g, &datum, mountain, air, 0.0,
                );
                if above {
                    assert!(
                        hit.is_none() && !limb,
                        "mountain alt {altitude} above a hit ray {ray:?} limb {limb}"
                    );
                }
            }
        }
        assert!(
            above_ok > 0,
            "the zenith was never above a directional horizon"
        );
    }

    /// Home-planet numbers: R = 31,017,520, g = 33, air 20,000, lowland under
    /// the eye, a +1M highland 25–30M blocks away toward +X. `px` is the
    /// 3440×1440, 70° sky margin.
    fn home_horizon_inputs(altitude: f32) -> (f32, f32, f32, f32, Vec<f32>) {
        let radius = 31_017_520.0f32;
        let g = 33u32;
        let gg = g as usize;
        let mut datum = vec![0.0f32; 6 * gg * gg];
        let mut highland = 0u32;
        for face in 0..6usize {
            let (tu, n, tv) = crate::far_body::far_map_basis(face);
            for j in 0..g {
                for i in 0..g {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let d =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    let arc = d.dot(Vec3::Y).clamp(-1.0, 1.0).acos() * radius;
                    // Same atan2(east, north) as the table. 0 is +X.
                    let az = d.z.atan2(d.x);
                    if (25.0e6..=30.0e6).contains(&arc) && az.abs() <= 0.5 {
                        datum[face * gg * gg + j as usize * gg + i as usize] = 1_000_000.0;
                        highland += 1;
                    }
                }
            }
        }
        assert!(highland > 8, "highland covered only {highland} samples");
        let view = view_pitched(0.0, 70.0, 3440, 1440);
        (radius, radius + altitude, 20_000.0, view.px_max, datum)
    }

    /// (a) Open-sky bins stay under +0.1, so the table actually rejects sky.
    /// (b) No surface or limb hit on this datum is classified above it.
    #[test]
    fn open_sky_bins_reject_above_a_distant_highland() {
        let up = Vec3::Y;
        let (east, north) = horizon_axes(up).expect("axes");
        let open_az = horizon_azimuth(-Vec3::X, up, east, north);
        let open_bin = horizon_bin(open_az);
        for altitude in [10.0f32, 10_000.0, 50_000.0] {
            let (radius, distance, air, px, datum) = home_horizon_inputs(altitude);
            let g = 33u32;
            let bins = build_horizon_bins(g, &datum, Quat::IDENTITY, up, radius, distance, air, px);
            for delta in -16..=16 {
                let idx = (open_bin as i32 + delta).rem_euclid(HORIZON_BINS as i32) as usize;
                assert!(
                    bins[idx] < 0.1,
                    "alt {altitude} open bin {idx} sine {} (px {px})",
                    bins[idx]
                );
            }
            assert!(
                ray_above_horizon(&bins, up, east, north, up),
                "alt {altitude} zenith was not rejected"
            );
            let down_open = (-up * 0.5 - Vec3::X * 0.8660254).normalize();
            assert!(
                !ray_above_horizon(&bins, up, east, north, down_open),
                "alt {altitude} a ray 30° below the horizon was rejected; bin {}",
                bins[open_bin]
            );
            let up_open = (up * 0.5 - Vec3::X * 0.8660254).normalize();
            assert!(
                ray_above_horizon(&bins, up, east, north, up_open),
                "alt {altitude} open sky 30° up was kept; bin {}",
                bins[open_bin]
            );

            let rho = radius / distance;
            let min_off = 0.0f32;
            let max_off = 1_000_000.0f32;
            let mut hits = 0u32;
            let mut rays = vec![-up, Vec3::X, -Vec3::X, Vec3::Z, -Vec3::Z];
            let gg = g as usize;
            for face in 0..6usize {
                let (tu, n, tv) = crate::far_body::far_map_basis(face);
                for (j, i) in [(8u32, 16u32), (16, 16), (24, 16), (16, 8), (16, 24)] {
                    let edge = (g - 1) as f32;
                    let xi = 2.0 * i as f32 / edge - 1.0;
                    let eta = 2.0 * j as f32 / edge - 1.0;
                    let quarter = std::f32::consts::FRAC_PI_4;
                    let s =
                        (n + tu * (xi * quarter).tan() + tv * (eta * quarter).tan()).normalize();
                    let off = datum[face * gg * gg + j as usize * gg + i as usize];
                    let point = s * (radius + off);
                    let eye = up * distance;
                    let to = point - eye;
                    if to.length_squared() > 1.0 {
                        rays.push(to.normalize());
                    }
                }
            }
            for ray in rays {
                let above = ray_above_horizon(&bins, up, east, north, ray);
                let hit = ray_mapped(
                    ray,
                    -up,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    1.0,
                    g,
                    &datum,
                    min_off,
                    max_off,
                );
                let limb = ray_mapped_limb(
                    ray,
                    -up,
                    rho,
                    distance,
                    Quat::IDENTITY,
                    g,
                    &datum,
                    max_off,
                    air,
                    px,
                );
                if hit.is_some() || limb {
                    hits += 1;
                    assert!(
                        !above,
                        "alt {altitude} rejected a hit={hit:?} limb {limb} ray {ray:?}"
                    );
                }
            }
            assert!(hits > 0, "alt {altitude} never hit the datum or the limb");
        }
    }

    /// Per-frame cost of the g = 33 table. Release budget is 0.2 ms; debug is
    /// only a backstop so a 4^6 fan-out still fails the suite.
    #[test]
    fn horizon_table_at_g33_builds_within_a_fifth_of_a_millisecond() {
        let up = Vec3::Y;
        let built = [10.0f32, 10_000.0, 50_000.0].map(home_horizon_inputs);
        let time_one = |radius, distance, air, px, datum: &Vec<f32>| {
            std::hint::black_box(build_horizon_bins(
                33,
                datum,
                Quat::IDENTITY,
                up,
                radius,
                distance,
                air,
                px,
            ))
        };
        for (radius, distance, air, px, datum) in &built {
            time_one(*radius, *distance, *air, *px, datum);
        }
        let iters = if cfg!(debug_assertions) { 2 } else { 20 };
        let rounds = if cfg!(debug_assertions) { 1 } else { 5 };
        let mut best = f64::MAX;
        for _ in 0..rounds {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                for (radius, distance, air, px, datum) in &built {
                    time_one(*radius, *distance, *air, *px, datum);
                }
            }
            let each = start.elapsed().as_nanos() as f64 / (iters as f64 * built.len() as f64);
            best = best.min(each);
        }
        eprintln!("horizon g=33 build {best:.0} ns");
        let limit = if cfg!(debug_assertions) {
            80_000_000.0
        } else {
            200_000.0
        };
        assert!(
            best < limit,
            "horizon build {best:.0} ns exceeds {limit:.0}"
        );
    }

    /// The azimuth pad is twice this error, so a fast angle cannot open a gap
    /// in the bin range.
    #[test]
    fn fast_atan_stays_inside_the_azimuth_pad() {
        let mut worst = 0.0f32;
        let ang = |d: f32| {
            let a = d.abs();
            a.min(std::f32::consts::TAU - a)
        };
        for i in 0..=20_000 {
            let z = i as f32 / 20_000.0;
            worst = worst.max(ang(fast_atan(z) - z.atan()));
        }
        for i in 0..64 {
            for j in 0..64 {
                let x = (i as f32 - 32.0) / 8.0;
                let y = (j as f32 - 32.0) / 8.0;
                if x == 0.0 && y == 0.0 {
                    continue;
                }
                worst = worst.max(ang(fast_atan2(y, x) - y.atan2(x)));
            }
        }
        for i in 0..=20_000 {
            let s = i as f32 / 20_000.0;
            worst = worst.max(ang(fast_asin(s) - s.asin()));
        }
        assert!(
            worst * 2.0 < HORIZON_AZ_PAD,
            "fast angle error {worst} rad, pad {HORIZON_AZ_PAD}"
        );
    }
}
