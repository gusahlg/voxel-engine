//! Host ring of far-body records for the sky pass (set 0, binding 2).
//! One coherent buffer per frame-in-flight, written before the sky draw.
//! The record matches `FarGpu` in `shaders/far_body.slang` (160 bytes, std430).
//! A 16-byte cone sits in front of each record so a pixel can reject the body
//! before loading it. A tile mask follows the records: one bit per kept body,
//! one word per screen tile, so a sky pixel skips bodies that miss its tile.

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
/// `-1` when a ray facing away can still hit. Matches `far_bodies()`: the
/// sphere and rounded rims reach `1.05` times the body, the cube's corners
/// sit at `√3` times the enlarged half-extent, a mapped body uses the hi
/// radius `(radius + max datum offset) / distance`, and an inner sphere is
/// the sky. `map_max[i]` is that map's maximum datum offset (0 if unset).
fn cone_bound(body: &FarBody, map_max: &[f32; MAX_FAR_MAPS]) -> f32 {
    if !far_cull_enabled() {
        return -1.0;
    }
    let rho = body.radius / body.distance;
    let bound = match body.shape {
        FarShape::InnerSphere => return -1.0,
        FarShape::Sphere => 1.05 * rho,
        FarShape::Cube => 3.0f32.sqrt() * 1.035 * rho,
        FarShape::Rounded { exponent } => {
            let p = exponent.clamp(2.0, 32.0);
            let rho_b = rho * 3.0f32.powf(0.5 - 1.0 / p) * (1.0 + 2.0e-4);
            1.05 * rho_b
        }
        FarShape::Mapped { map, .. } => {
            let max_off = if (map.0 as usize) < MAX_FAR_MAPS {
                map_max[map.0 as usize]
            } else {
                0.0
            };
            ((body.radius + max_off) / body.distance).max(0.0)
        }
    };
    // The camera is inside, or nearly inside, the bounding sphere: a ray
    // facing away can still meet the body, so the cone must not reject it.
    if !bound.is_finite() || bound >= 0.99 {
        -1.0
    } else {
        bound
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
/// `dot(n, dir) < -a`. `a >= 1` covers the view, so the body stays.
fn cone_in_view(dir: glam::Vec3, bound: f32, view: &FarView) -> bool {
    if bound < 0.0 {
        return true;
    }
    let a_sin = (bound + 3.0 * view.px_max).min(1.0);
    if a_sin >= 1.0 {
        return true;
    }
    let len2 = dir.length_squared();
    if !(len2 > 0.0) {
        return true;
    }
    let dir = dir / len2.sqrt();
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
/// A cone that reaches within 10° of the view plane (80° from the view axis)
/// is painted into every tile. A screen rect is not trustworthy that close.
const VIEW_PLANE_LIMIT_RAD: f32 = 80.0 * std::f32::consts::PI / 180.0;

enum TileCover {
    /// Bit set in every tile.
    All,
    /// Fully off the render extent. The frustum cull should already have dropped it.
    None,
    /// Inclusive tile coordinates.
    Rect {
        tx0: u32,
        ty0: u32,
        tx1: u32,
        ty1: u32,
    },
}

/// Screen tiles touched by kept body `dir`/`bound`. `bound` is the cone sine,
/// or `-1` to keep the body everywhere. The drawable angular radius is
/// `asin(min(bound + 3 px_max, 1))` on the unit centre direction.
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
    if !lim.is_finite() || lim >= 1.0 {
        return TileCover::All;
    }
    let a = lim.asin();
    // Direction through the eye: translation in `view_proj` drops out at w = 0.
    // `clip.w` is cos(theta) for a unit direction (camera looks down −Z).
    let clip = view.view_proj * glam::Vec4::new(dir.x, dir.y, dir.z, 0.0);
    if !clip.is_finite() {
        return TileCover::All;
    }
    let theta = clip.w.clamp(-1.0, 1.0).acos();
    if !theta.is_finite() || theta + a >= VIEW_PLANE_LIMIT_RAD {
        return TileCover::All;
    }
    let cos_outer = (theta + a).cos();
    if !(cos_outer > 0.0) || !(clip.w > 0.0) {
        return TileCover::All;
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
/// at zero; the shader then walks every kept body.
fn stamp_tiles(table: &mut FarTableGpu, view: Option<&FarView>) {
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
        }
    }
    if blanket != 0 {
        for mask in &mut table.tile_mask[..tile_count] {
            *mask |= blanket;
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
/// `map_max[i]` is map `i`'s maximum datum offset, used for a mapped body's
/// hi-radius cone (`-1` when that sphere contains the camera, which paints
/// every tile).
pub(crate) fn pack_table(
    bodies: &[FarBody],
    view: Option<FarView>,
    map_max: &[f32; MAX_FAR_MAPS],
) -> FarTableGpu {
    let mut table = FarTableGpu::zeroed();
    let n = bodies.len().min(MAX_FAR_BODIES);
    let cull = far_cull_enabled();
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        let bound = cone_bound(body, map_max);
        if cull {
            if let Some(view) = view.as_ref() {
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
    stamp_tiles(&mut table, view.as_ref());
    table
}

/// Per-slot far-body SSBO. Identical bytes skip the map write.
pub(crate) struct FarBodyRing {
    bufs: PerSlot<HostBuffer>,
    last: PerSlot<Option<FarTableGpu>>,
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
        }
    }

    pub(crate) fn write(
        &mut self,
        slot: FrameSlot,
        bodies: &[FarBody],
        view: Option<FarView>,
        map_max: &[f32; MAX_FAR_MAPS],
    ) {
        let table = pack_table(bodies, view, map_max);
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
        FarBody, FarMapId, FarShape, ray_cube, ray_inner_sphere, ray_mapped, ray_rounded,
        ray_sphere, store,
    };

    fn pack_table(bodies: &[FarBody], view: Option<FarView>) -> FarTableGpu {
        super::pack_table(bodies, view, &[0.0; MAX_FAR_MAPS])
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

        // 1.05 * rho >= 0.99: the camera is nearly inside the bounding sphere.
        let inside = sample(FarShape::Sphere, 1.0, 0.95);
        assert!(1.05 * 0.95 >= 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&inside), None).cone[0][3].to_bits(),
            (-1.0f32).to_bits()
        );

        let cube = sample(FarShape::Cube, 1.0, 0.2);
        let cube_bound = 3.0f32.sqrt() * 1.035 * 0.2;
        assert!(cube_bound < 0.99);
        assert_eq!(
            pack_table(std::slice::from_ref(&cube), None).cone[0][3].to_bits(),
            cube_bound.to_bits()
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
        assert_eq!(table.cone[0][3].to_bits(), ((1.0f32 + 0.1) / 4.0).to_bits());
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

    fn view_along_neg_z(fovy: f32, width: u32, height: u32) -> FarView {
        let cam = Camera3D {
            position: Vec3::new(12.0, 3.0, -4.0),
            target: Vec3::new(12.0, 3.0, -5.0),
            up: Vec3::Y,
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
}
