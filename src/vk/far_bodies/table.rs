//! The far table the sky pass reads: the record layouts and their offsets,
//! per-body packing, the whole-table pack, and the byte ranges the ring
//! uploads.

use bytemuck::{Pod, Zeroable};

use super::cones::cone_bound;
use super::horizon::horizon_dip;
#[cfg(test)]
use super::tiles::TileFrames;
use super::tiles::{TileRuns, TileView, stamp_tiles, used_tiles};
use super::view::{FarView, body_meets_view};
use crate::far_body::{FarBody, FarShape, MAX_FAR_BODIES, MAX_FAR_MAPS};

/// One body on the GPU. Ten 16-byte lanes, 160 bytes.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(super) struct FarBodyGpu {
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

/// The shape codes in `FarBodyGpu::atmosphere.w`, as `FarGpu` in
/// `shaders/far_table.slang` lists them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ShapeCode {
    Cube,
    Sphere,
    InnerSphere,
    Rounded,
    Mapped,
}

impl ShapeCode {
    pub(super) const ALL: [Self; 5] = [
        Self::Cube,
        Self::Sphere,
        Self::InnerSphere,
        Self::Rounded,
        Self::Mapped,
    ];

    /// The lane value [`pack_one`] writes.
    pub(super) const fn lane(self) -> f32 {
        match self {
            Self::Cube => 0.0,
            Self::Sphere => 1.0,
            Self::InnerSphere => 2.0,
            Self::Rounded => 3.0,
            Self::Mapped => 4.0,
        }
    }
}

/// Named reads of the packed lanes. [`pack_one`] is the one writer.
impl FarBodyGpu {
    /// The code whose `[lane - 0.5, lane + 0.5)` holds `atmosphere.w`, or
    /// `None` for any other value (NaN included). Such a body is heavy.
    pub(super) fn shape(&self) -> Option<ShapeCode> {
        let w = self.atmosphere[3];
        ShapeCode::ALL
            .into_iter()
            .find(|code| ((code.lane() - 0.5)..(code.lane() + 0.5)).contains(&w))
    }

    pub(super) fn is_mapped(&self) -> bool {
        self.shape() == Some(ShapeCode::Mapped)
    }

    pub(super) fn is_rounded(&self) -> bool {
        self.shape() == Some(ShapeCode::Rounded)
    }

    /// A sphere or an inner sphere, the pair the sphere fragment draws.
    pub(super) fn is_light(&self) -> bool {
        matches!(
            self.shape(),
            Some(ShapeCode::Sphere | ShapeCode::InnerSphere)
        )
    }

    /// Centre direction, `dir_rho.xyz`.
    pub(super) fn dir(&self) -> glam::Vec3 {
        glam::Vec3::new(self.dir_rho[0], self.dir_rho[1], self.dir_rho[2])
    }

    /// Reference `radius / distance`, `dir_rho.w`.
    pub(super) fn rho(&self) -> f32 {
        self.dir_rho[3]
    }

    /// Mapped horizon, `albedo0.w`. 0 for the other shapes.
    pub(super) fn horizon(&self) -> f32 {
        self.albedo0[3]
    }

    /// Mapped air-shell thickness in the unit of `radius`, `albedo1.w`. 0
    /// for the other shapes.
    pub(super) fn air(&self) -> f32 {
        self.albedo1[3]
    }

    /// World distance, `albedo2.w`, every shape.
    pub(super) fn distance(&self) -> f32 {
        self.albedo2[3]
    }

    /// Far-map slot from `seed.z` (map id + 1). `None` when the body is not
    /// mapped (0) or the id is past [`MAX_FAR_MAPS`].
    pub(super) fn map_index(&self) -> Option<usize> {
        let map_plus = self.seed[2] as usize;
        (1..=MAX_FAR_MAPS).contains(&map_plus).then(|| map_plus - 1)
    }
}

/// Mask words in the GPU table. `tile_mask` in `shaders/far_table.slang` is
/// sized by the generated twin. 8192 tiles cover 7680×4320 at 64 px; a larger
/// render extent uses a coarser tile so the count still fits.
pub(super) const MAX_FAR_TILES: usize = 8192;
const _: () = assert!(crate::genconst::MAX_FAR_TILES as usize == MAX_FAR_TILES);

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
/// `list_header` is `(n_base, n_full, width, height)`. `tile_index` holds
/// three runs, each ascending row-major as [`fill_tile_lists`] writes them:
/// mask == 0, then tiles whose bits are only spheres and inner spheres, then
/// the rest (the heavy run). `n_full` is the last two runs. The ring may then
/// stably reorder the base run (the disc-free coarse prefix first) and the
/// heavy run (the coarse mapped interior first, then, with mapsolo on, the
/// other single-Mapped tiles; with roundsolo on, the single-Rounded tiles
/// last) so each sky pipeline draws one contiguous run.
/// `list_header.x` is still every base tile. The sky tile vertex shader reads
/// this tail; the `far_bodies()` fragment loop does not.
///
/// `Pod` is implemented by hand: bytemuck's derive only covers arrays up to a
/// few dozen elements, and the mask is 8192 words. The layout is plain
/// `repr(C)` floats and uints with no padding.
///
/// [`fill_tile_lists`]: super::tiles::fill_tile_lists
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq)]
pub(super) struct FarTableGpu {
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

impl FarTableGpu {
    /// Live body count: `header.x`, capped at [`MAX_FAR_BODIES`].
    pub(super) fn kept(&self) -> usize {
        (self.header[0] as usize).min(MAX_FAR_BODIES)
    }

    /// Bit `k` is set when kept body `k` passes `pred`.
    pub(super) fn kept_bits(&self, pred: impl Fn(&FarBodyGpu) -> bool) -> u32 {
        let mut bits = 0u32;
        for (k, body) in self.body[..self.kept()].iter().enumerate() {
            if pred(body) {
                bits |= 1u32 << k;
            }
        }
        bits
    }

    /// The cone direction of kept body `k`, `cone[k].xyz`.
    pub(super) fn cone_dir(&self, k: usize) -> glam::Vec3 {
        let cone = self.cone[k];
        glam::Vec3::new(cone[0], cone[1], cone[2])
    }
}

/// Bins in one azimuthal horizon table. `horizon_sin` in
/// `shaders/far_table.slang` holds two tables, back to back, sized by the
/// generated twin.
pub(super) const HORIZON_BINS: usize = 256;
const _: () = assert!(crate::genconst::FAR_HORIZON_BINS as usize == HORIZON_BINS);
/// Tables published per frame. The two Mapped bodies with the largest rho.
pub(super) const HORIZON_TABLES: usize = 2;

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

fn rgb4(c: crate::color::LinearRgb) -> [f32; 4] {
    [c.0[0], c.0[1], c.0[2], 0.0]
}

pub(super) fn pack_one(body: &FarBody) -> FarBodyGpu {
    let dir = body.dir;
    let q = body.rotation;
    let (shape, exponent, map_plus, horizon, air) = match body.shape {
        FarShape::Cube => (ShapeCode::Cube, 0.0, 0, 0.0, 0.0),
        FarShape::Sphere => (ShapeCode::Sphere, 0.0, 0, 0.0, 0.0),
        FarShape::InnerSphere => (ShapeCode::InnerSphere, 0.0, 0, 0.0, 0.0),
        FarShape::Rounded { exponent } => (ShapeCode::Rounded, exponent, 0, 0.0, 0.0),
        FarShape::Mapped { map, horizon, air } => {
            let map_plus = u32::from(map.0) + 1;
            (ShapeCode::Mapped, 0.0, map_plus, horizon, air)
        }
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
            shape.lane(),
        ],
        seed: [body.seed, exponent.to_bits(), map_plus, 0],
    }
}

/// Bytes the GPU reads this frame: the body table plus one mask per live tile.
/// The unused mask tail is not part of the match and is not rewritten.
pub(super) fn table_bytes(table: &FarTableGpu) -> &[u8] {
    let len = std::mem::offset_of!(FarTableGpu, tile_mask)
        + used_tiles(table) * std::mem::size_of::<u32>();
    &bytemuck::bytes_of(table)[..len]
}

/// `list_header` plus the live index prefix (`n_base + n_full` words).
pub(super) fn list_bytes(table: &FarTableGpu) -> &[u8] {
    let start = std::mem::offset_of!(FarTableGpu, list_header);
    let n = (table.list_header[0] as usize)
        .saturating_add(table.list_header[1] as usize)
        .min(MAX_FAR_TILES);
    let len = std::mem::size_of::<[u32; 4]>() + n * std::mem::size_of::<u32>();
    &bytemuck::bytes_of(table)[start..start + len]
}

pub(super) fn nonzero_tiles(table: &FarTableGpu) -> u64 {
    table.tile_mask[..used_tiles(table)]
        .iter()
        .filter(|mask| **mask != 0)
        .count() as u64
}

/// [`pack_table_cached`] with no tile frames and a zero minimum-offset table,
/// keeping the table.
#[cfg(test)]
pub(super) fn pack_table(
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
pub(super) fn zeroed_table() -> Box<FarTableGpu> {
    let layout = std::alloc::Layout::new::<FarTableGpu>();
    // SAFETY: `FarTableGpu` is `Zeroable`, so the zeroed allocation is a valid
    // value. The pointer is the exact layout `Box` will free.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<FarTableGpu>();
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    unsafe { Box::from_raw(ptr) }
}

/// Sentinel ids and sines. A zeroed table would name body 0 and reject every
/// ray above the horizontal.
fn prime_horizon(table: &mut FarTableGpu) {
    table.horizon_id = [u32::MAX, u32::MAX, HORIZON_BINS as u32, 0];
    table.horizon_sin.fill(1.0);
}

/// [`pack_table_into`] a fresh table with culling on. `frames` are used when
/// they match `view`; otherwise the angular cones build their own.
#[cfg(test)]
pub(super) fn pack_table_cached(
    bodies: &[FarBody],
    view: Option<&FarView>,
    frames: Option<&TileFrames>,
    map_max: &[f32; MAX_FAR_MAPS],
    map_min: &[f32; MAX_FAR_MAPS],
    sky_up: glam::Vec3,
) -> (Box<FarTableGpu>, f32) {
    let mut table = zeroed_table();
    let tiles = view
        .zip(frames)
        .and_then(|(view, frames)| frames.view(view));
    let (dip, _) = pack_table_into(
        &mut table,
        bodies,
        view,
        tiles.as_ref(),
        map_max,
        map_min,
        sky_up,
        true,
    );
    (table, dip)
}

/// Pack `bodies` in order into `table`, which is overwritten whole. With
/// culling on and a view, bodies whose cone cannot meet the frustum are
/// dropped; the rest stay in their original relative order (the sky
/// composite is order-dependent). No view keeps every body. Each kept body
/// then sets its bit in the screen tiles its drawable cone can reach.
/// `map_max[i]` is map `i`'s maximum datum offset, used for a mapped body's
/// per-pixel cone. That sine is the hi radius plus `air/distance` (`-1` when
/// that reach is at least 0.99). A mapped body with `horizon < 1` paints tiles
/// from the horizon cone instead, so the sentinel does not cover the sky.
///
/// `cull` off (`VOXEL_FAR_CULL=0`) keeps every body, stores the `-1` cone,
/// and puts every kept body in every live tile. `tiles` are the ring's tile
/// frames when they match `view`. Returns the horizon-dip sine and the runs
/// the tile lists were filled with.
#[allow(clippy::too_many_arguments)]
pub(super) fn pack_table_into(
    table: &mut FarTableGpu,
    bodies: &[FarBody],
    view: Option<&FarView>,
    tiles: Option<&TileView>,
    map_max: &[f32; MAX_FAR_MAPS],
    map_min: &[f32; MAX_FAR_MAPS],
    sky_up: glam::Vec3,
    cull: bool,
) -> (f32, TileRuns) {
    // Every byte: a reused table must not carry the last frame's masks,
    // bodies or index tail.
    bytemuck::bytes_of_mut(table).fill(0);
    prime_horizon(table);
    let n = bodies.len().min(MAX_FAR_BODIES);
    // Before the cull: see [`horizon_dip`].
    let dip = horizon_dip(&bodies[..n], sky_up, map_min);
    let mut kept = 0usize;
    for body in bodies.iter().take(n) {
        let bound = if cull {
            cone_bound(body, map_max)
        } else {
            -1.0
        };
        if cull
            && let Some(view) = view
            && !body_meets_view(body, bound, view)
        {
            continue;
        }
        let d = body.dir;
        table.cone[kept] = [d.x, d.y, d.z, bound];
        table.body[kept] = pack_one(body);
        kept += 1;
    }
    table.header[0] = kept as u32;
    let runs = stamp_tiles(table, view, tiles, cull);
    (dip, runs)
}
