use super::arena::ArenaDirectory;
#[cfg(test)]
use super::buffers::MeshRecord;
use super::buffers::{DrawIndexedIndirect, MESH_FLAG_FACE_RUNS};
use super::pipeline::EyeSplit;
use crate::camera::Frustum;
use crate::switches::{Switch, parse_or};

/// Camera emission groups, in partition-table order. Mirrored by cull.comp.slang.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub(crate) enum Group {
    /// Full-res (scale <= 1) Opaque: no-`discard` fragment module.
    Opaque = 0,
    Cutout = 1,
    /// Coarse-LOD (scale > 1) Opaque: the box-clip `discard` module.
    OpaqueLod = 2,
    /// Full-res caged mesh (any non-blend pass, scale <= 1).
    Caged = 3,
    /// Coarse-LOD caged mesh (scale > 1).
    CagedLod = 4,
}
/// Camera groups (Opaque, Cutout, OpaqueLod, Caged, CagedLod). Bucketed.
pub(crate) const CAMERA_GROUPS: usize = 5;
/// Shadow Near/Far. Unbucketed; sized from full-res Opaque plus full-res caged.
pub(crate) const SHADOW_GROUPS: usize = 2;
/// Camera groups + shadow groups.
pub(crate) const GROUPS: usize = CAMERA_GROUPS + SHADOW_GROUPS;
/// Front-to-back buckets on camera groups only (shadows stay unbucketed).
pub(crate) const BUCKETS: usize = crate::genconst::CULL_DISTANCE_BUCKETS as usize;
/// Inclusive lower edges of buckets 1..=3 for full-res groups
/// (`cull.comp.slang` `distance_bucket` with scale 1). Coarse-LOD groups
/// multiply these by [`lod_bucket_scale`].
pub(crate) const CULL_BUCKET_SPLITS: [f32; 3] = [
    crate::genconst::CULL_BUCKET_SPLIT_0,
    crate::genconst::CULL_BUCKET_SPLIT_1,
    crate::genconst::CULL_BUCKET_SPLIT_2,
];
/// Default coarse-LOD split multiplier. Edges become 512 / 2048 / 8192.
/// `VOXEL_LOD_BUCKET_SCALE=1` restores the full-res edges.
pub(crate) const DEFAULT_LOD_BUCKET_SCALE: f32 = 32.0;
/// Squared full-res edges. The splits are powers of two, so `split²` is exact
/// and is the [`sqrt_ge_threshold`] of `split`: `d² >= split²` matches the
/// shader's `length >= split` exactly and the hot path skips the sqrt.
const FULL_BUCKET_EDGE_SQ: [f32; 3] = [
    crate::genconst::CULL_BUCKET_SPLIT_0 * crate::genconst::CULL_BUCKET_SPLIT_0,
    crate::genconst::CULL_BUCKET_SPLIT_1 * crate::genconst::CULL_BUCKET_SPLIT_1,
    crate::genconst::CULL_BUCKET_SPLIT_2 * crate::genconst::CULL_BUCKET_SPLIT_2,
];
/// Max contiguous face-runs the GPU cull emits per camera mesh. Per axis the
/// camera is in {+, −, both}; upload order +X,+Y,+Z,−X,−Y,−Z keeps same-sign
/// faces adjacent, so an outside camera sees ≤3 maximal contiguous runs.
pub(crate) const MAX_FACE_RUNS: u32 = 3;
/// Live-count lanes, one per camera group.
pub(crate) const LANES: usize = CAMERA_GROUPS;
/// Size of VkDrawIndexedIndirectCommand.
pub(crate) const CMD_STRIDE: u64 = 20;
pub(crate) const WORKGROUP: u32 = crate::genconst::CULL_WORKGROUP;
/// Profiling-only geometry histogram: per camera group `[draws, index_count]`.
pub(crate) const STATS_COUNT: usize = CAMERA_GROUPS * 2;
pub(crate) const STATS_BYTES: u64 = (STATS_COUNT * size_of::<u32>()) as u64;
/// Push-constant flag: fill the stats histogram.
pub(crate) const FLAG_STATS: u32 = crate::genconst::CULL_FLAG_STATS;
/// `CullParams.flags` bit: emit face runs.
pub(crate) const FLAG_FACE_RUNS: u32 = crate::genconst::CULL_FLAG_FACE_RUNS;
/// Live camera-group records at or below this count skip the GPU cull and
/// emit the same commands on the CPU. Overridable via `VOXEL_CPU_CULL_MAX`.
///
/// Forcing the CPU path was +5 % at 567 meshes on an RTX 3070 and +25 % on an
/// RTX 4060 with a Ryzen 5 5500; at 2025 meshes it was +8 % on the Ryzen box
/// and -3 % on the i5 box. 1024 keeps the win without paying on fast GPUs.
const CPU_CULL_MAX: u32 = 1024;

const _: () = assert!(crate::genconst::CULL_DISTANCE_BUCKETS == 4);
const _: () = assert!(crate::genconst::CULL_CAMERA_GROUPS == CAMERA_GROUPS as u32);
const _: () = assert!(Group::OpaqueLod as u32 == crate::genconst::CULL_OPAQUE_LOD_GROUP);
const _: () = assert!(Group::Caged as u32 == crate::genconst::CULL_CAGED_GROUP);
const _: () = assert!(Group::CagedLod as u32 == crate::genconst::CULL_CAGED_LOD_GROUP);
const _: () = assert!(GROUPS == CAMERA_GROUPS + SHADOW_GROUPS);
const _: () = assert!(Group::CagedLod as usize + 1 == CAMERA_GROUPS);
const _: () = assert!(MESH_FLAG_FACE_RUNS == crate::genconst::MESH_FLAG_FACE_RUNS);

/// Camera-group index. Caged draws take their own groups so the flat pipelines
/// stay free of the bent varying.
#[inline(always)]
fn camera_group(pass: u32, lod: bool, caged: bool) -> u32 {
    if caged {
        if lod {
            Group::CagedLod as u32
        } else {
            Group::Caged as u32
        }
    } else if pass == 0 && lod {
        Group::OpaqueLod as u32
    } else {
        pass
    }
}

#[inline(always)]
fn is_lod_group(group: u32) -> bool {
    group == Group::OpaqueLod as u32 || group == Group::CagedLod as u32
}

/// True when every fragment of a camera-relative AABB would be discarded by
/// the coarse-LOD box clip (`mesh3d.frag.slang`): `abs(p - centre) < half` on
/// every axis. A non-positive half covers nothing (`min >= max` on that axis).
/// The skip is exact: the farthest corner from `centre` on each axis must sit
/// strictly inside `half`. `centre == 0` matches the centred `abs` test.
/// Mirrored by `lod_aabb_inside_box` in `cull.comp.slang`.
fn lod_aabb_inside_box(mn: [f32; 3], mx: [f32; 3], centre: [f32; 3], half: [f32; 3]) -> bool {
    if half[0] <= 0.0 || half[1] <= 0.0 || half[2] <= 0.0 {
        return false;
    }
    let far = |i: usize| (mn[i] - centre[i]).abs().max((mx[i] - centre[i]).abs());
    far(0) < half[0] && far(1) < half[1] && far(2) < half[2]
}

/// `VOXEL_LOD_BUCKET_SCALE` when it parses as a finite positive float,
/// otherwise [`DEFAULT_LOD_BUCKET_SCALE`]. Read once.
pub(crate) fn lod_bucket_scale() -> f32 {
    VOXEL_LOD_BUCKET_SCALE.get()
}

/// Read by [`lod_bucket_scale`].
pub(crate) static VOXEL_LOD_BUCKET_SCALE: Switch<f32> =
    Switch::new("VOXEL_LOD_BUCKET_SCALE", parse_lod_bucket_scale);

fn parse_lod_bucket_scale(raw: Option<&str>) -> f32 {
    raw.and_then(|s| s.parse().ok())
        .filter(|s: &f32| s.is_finite() && *s > 0.0)
        .unwrap_or(DEFAULT_LOD_BUCKET_SCALE)
}

/// Linear bucket edges for `scale`. Scale 1 is the full-res splits.
pub(crate) fn bucket_splits(scale: f32) -> [f32; 3] {
    CULL_BUCKET_SPLITS.map(|s| s * scale)
}

/// Split multiplier for one camera group. Coarse-LOD groups use
/// [`lod_bucket_scale`]; every other group uses 1.
pub(crate) fn group_bucket_scale(group: u32) -> f32 {
    if is_lod_group(group) {
        lod_bucket_scale()
    } else {
        1.0
    }
}

/// `d²` thresholds for `scale`: `d² >= t` exactly when the shader's
/// `length(centre) >= split * scale`. Scale 1 returns the const
/// [`FULL_BUCKET_EDGE_SQ`]; any other scale (once per cull, not per mesh)
/// goes through [`sqrt_ge_threshold`].
fn bucket_edge_sq(scale: f32) -> [f32; 3] {
    if scale == 1.0 {
        return FULL_BUCKET_EDGE_SQ;
    }
    bucket_splits(scale).map(sqrt_ge_threshold)
}

/// Smallest `t` with `sqrt(t) >= edge`. f32 sqrt is correctly rounded and
/// monotonic, so `d2 >= t` is exactly `sqrt(d2) >= edge`. A power-of-two edge
/// gives `edge * edge`. Other edges can be an ulp off: at `edge = 20`
/// (`VOXEL_LOD_BUCKET_SCALE=1.25`) the float just below 400 has sqrt 20, so
/// `d2 >= edge * edge` would put that centre one bucket nearer than the
/// shader's `length >= edge`.
fn sqrt_ge_threshold(edge: f32) -> f32 {
    let mut t = edge * edge;
    while t.sqrt() < edge {
        t = t.next_up();
    }
    while t > 0.0 && t.next_down().sqrt() >= edge {
        t = t.next_down();
    }
    t
}

/// Camera-distance bucket of an AABB centre, matching `cull.comp.slang`
/// `distance_bucket(length, scale)`.
#[cfg(test)]
#[inline(always)]
fn distance_bucket(dist: f32) -> u32 {
    distance_bucket_scaled(dist, 1.0)
}

#[cfg(test)]
fn distance_bucket_scaled(dist: f32, scale: f32) -> u32 {
    distance_bucket_sq(dist * dist, bucket_edge_sq(scale))
}

#[inline(always)]
fn distance_bucket_sq(d2: f32, edge_sq: [f32; 3]) -> u32 {
    u32::from(d2 >= edge_sq[0]) + u32::from(d2 >= edge_sq[1]) + u32::from(d2 >= edge_sq[2])
}

/// `VOXEL_CPU_CULL_MAX` when it parses as a `u32`, otherwise
/// [`CPU_CULL_MAX`]. Read once; [`FaceCull::resolve`] calls this every frame.
///
/// [`FaceCull::resolve`]: super::cull::FaceCull::resolve
pub(crate) fn cpu_cull_max() -> u32 {
    VOXEL_CPU_CULL_MAX.get()
}

/// Read by [`cpu_cull_max`].
pub(crate) static VOXEL_CPU_CULL_MAX: Switch<u32> =
    Switch::new("VOXEL_CPU_CULL_MAX", |raw| parse_or(raw, CPU_CULL_MAX));

/// P-vertex test matching `outside_plane` in `cull.comp.slang` and
/// [`Frustum::intersects_aabb`]. Branch-free: `max(n·mn, n·mx)` selects the
/// p-vertex for each axis without a sign compare (valid when `mn <= mx`).
#[inline(always)]
fn outside_plane(plane: [f32; 4], mn: [f32; 3], mx: [f32; 3]) -> bool {
    let d = f32::max(plane[0] * mn[0], plane[0] * mx[0])
        + f32::max(plane[1] * mn[1], plane[1] * mx[1])
        + f32::max(plane[2] * mn[2], plane[2] * mx[2])
        + plane[3];
    d < 0.0
}

#[inline(always)]
fn aabb_in_planes(planes: &[[f32; 4]; 5], mn: [f32; 3], mx: [f32; 3]) -> bool {
    u32::from(outside_plane(planes[0], mn, mx))
        | u32::from(outside_plane(planes[1], mn, mx))
        | u32::from(outside_plane(planes[2], mn, mx))
        | u32::from(outside_plane(planes[3], mn, mx))
        | u32::from(outside_plane(planes[4], mn, mx))
        == 0
}

/// Legacy p-vertex used by [`cpu_cull_legacy`]: the shader's `?:` form.
#[cfg(test)]
fn outside_plane_select(plane: [f32; 4], mn: [f32; 3], mx: [f32; 3]) -> bool {
    let corner = [
        if plane[0] >= 0.0 { mx[0] } else { mn[0] },
        if plane[1] >= 0.0 { mx[1] } else { mn[1] },
        if plane[2] >= 0.0 { mx[2] } else { mn[2] },
    ];
    plane[0] * corner[0] + plane[1] * corner[1] + plane[2] * corner[2] + plane[3] < 0.0
}

#[cfg(test)]
fn aabb_in_planes_select(planes: &[[f32; 4]], mn: [f32; 3], mx: [f32; 3]) -> bool {
    planes.iter().all(|p| !outside_plane_select(*p, mn, mx))
}

fn slot_visible(visible: &[u32], slot: u32) -> bool {
    visible
        .get((slot >> 5) as usize)
        .is_some_and(|w| w & (1 << (slot & 31)) != 0)
}

/// Persistent CPU-cull staging: per-partition command lists and counts stay in
/// cache-resident host memory across frames (inner capacity retained).
/// Host-visible (write-combined) buffers are filled with one contiguous copy
/// per partition, never by scattered per-slot stores.
#[derive(Default)]
pub(crate) struct CpuCullScratch {
    pub part_cmds: Vec<Vec<DrawIndexedIndirect>>,
    pub counts: Vec<u32>,
    live_vis: Vec<u32>,
}

#[cfg(test)]
fn flatten_part_cmds(
    part_cmds: &[Vec<DrawIndexedIndirect>],
    partitions: &[PartitionGpu],
) -> Vec<DrawIndexedIndirect> {
    let total: usize = partitions.iter().map(|p| p.capacity as usize).sum();
    let mut cmds = vec![bytemuck::Zeroable::zeroed(); total];
    for (i, p) in partitions.iter().enumerate() {
        let src = &part_cmds[i];
        let start = p.offset as usize;
        cmds[start..start + src.len()].copy_from_slice(src);
    }
    cmds
}

/// Camera-relative AABB and decoded scale, matching the shader's
/// `offset / mn / mx / scale` reconstruction.
#[cfg(test)]
fn cam_relative_aabb(rec: &MeshRecord, eye: EyeSplit) -> ([f32; 3], [f32; 3], f32) {
    let scale = rec.detail_scale();
    let r = eye.rel(rec.block);
    let offset = [
        r[0] + rec.local_off[0],
        r[1] + rec.local_off[1],
        r[2] + rec.local_off[2],
    ];
    (
        [
            rec.aabb_min[0] * scale + offset[0],
            rec.aabb_min[1] * scale + offset[1],
            rec.aabb_min[2] * scale + offset[2],
        ],
        [
            rec.aabb_max[0] * scale + offset[0],
            rec.aabb_max[1] * scale + offset[1],
            rec.aabb_max[2] * scale + offset[2],
        ],
        scale,
    )
}

/// Per-axis face visibility, matching `cull.comp.slang`: +axis iff `mn[axis] < 0`,
/// −axis iff `mx[axis] > 0`. Flat meshes only. Caged slots use
/// [`cage_direction_vis`].
fn face_vis(mn: [f32; 3], mx: [f32; 3]) -> [bool; 6] {
    [
        mn[0] < 0.0,
        mn[1] < 0.0,
        mn[2] < 0.0,
        mx[0] > 0.0,
        mx[1] > 0.0,
        mx[2] > 0.0,
    ]
}

/// The cull reads `vis` only; tests assert the rest.
#[derive(Clone, Copy, Debug)]
struct CageFaceVis {
    /// Upload order +X,+Y,+Z,−X,−Y,−Z.
    vis: [bool; 6],
    /// Max corner error of corners 3,5,6,7, in min-edge lengths.
    #[cfg_attr(not(test), allow(dead_code))]
    d: f32,
    /// `2d + bias`, the margin on each `t_cam` test.
    #[cfg_attr(not(test), allow(dead_code))]
    eps: f32,
    /// The camera in the cage's affine frame.
    #[cfg_attr(not(test), allow(dead_code))]
    t_cam: [f32; 3],
}

#[inline(always)]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn len3(a: [f32; 3]) -> f32 {
    dot3(a, a).sqrt()
}

/// Camera-relative cage corners. `corners` are anchor-relative; the offset is
/// the vertex shader's `(anchor − cam_block) − cam_frac`.
#[inline(always)]
fn cam_relative_corners(corners: [[f32; 3]; 8], anchor: [i32; 3], eye: EyeSplit) -> [[f32; 3]; 8] {
    let d = eye.rel(anchor);
    std::array::from_fn(|i| {
        [
            corners[i][0] + d[0],
            corners[i][1] + d[1],
            corners[i][2] + d[2],
        ]
    })
}

/// Per-direction visibility of a caged mesh, or `None` when the affine frame
/// is singular (draw the mesh whole). `corners` are camera-relative.
/// Mirrors `cage_direction_mask` in `cull.comp.slang`, with the same generated
/// cutoff (`CULL_CAGE_DET_REL`) and margin bias (`CULL_CAGE_VIS_BIAS`).
///
/// `E = [P1−P0, P2−P0, P4−P0]`, `t_cam = E⁻¹(−P0)`. A local +X face is
/// front-facing when `t_cam.x` is past that face's parameter; testing the
/// full cage range `[0, 1]` (plus the non-affinity margin) is conservative.
fn cage_direction_vis(p: [[f32; 3]; 8]) -> Option<CageFaceVis> {
    let e0 = sub3(p[1], p[0]);
    let e1 = sub3(p[2], p[0]);
    let e2 = sub3(p[4], p[0]);
    let c12 = cross3(e1, e2);
    let c20 = cross3(e2, e0);
    let c01 = cross3(e0, e1);
    let det = dot3(e0, c12);
    let l0 = len3(e0);
    let l1 = len3(e1);
    let l2 = len3(e2);
    let vol = l0 * l1 * l2;
    if !(det.abs() > crate::genconst::CULL_CAGE_DET_REL * vol.max(1.0)) {
        return None;
    }
    let b = [-p[0][0], -p[0][1], -p[0][2]];
    let inv = 1.0 / det;
    let t_cam = [dot3(b, c12) * inv, dot3(b, c20) * inv, dot3(b, c01) * inv];
    let min_edge = l0.min(l1).min(l2);
    // Left to right like the shader's `P0 + e0 + e1`: `(P0 + e0) + e1`.
    let err = |idx: usize, predict: [f32; 3]| len3(sub3(p[idx], predict));
    let p01 = add3(p[0], e0);
    let dev = err(3, add3(p01, e1))
        .max(err(5, add3(p01, e2)))
        .max(err(6, add3(add3(p[0], e1), e2)))
        .max(err(7, add3(add3(p01, e1), e2)));
    let d = dev / min_edge;
    let eps = 2.0 * d + crate::genconst::CULL_CAGE_VIS_BIAS;
    Some(CageFaceVis {
        vis: [
            t_cam[0] > -eps,
            t_cam[1] > -eps,
            t_cam[2] > -eps,
            t_cam[0] < 1.0 + eps,
            t_cam[1] < 1.0 + eps,
            t_cam[2] < 1.0 + eps,
        ],
        d,
        eps,
        t_cam,
    })
}

/// Cumulative index bounds `b[0..=6]` from packed u16 quad pairs (upload
/// slots 0|1, 2|3, 4|5), matching the shader's `b[7]` loop.
fn packed_face_bounds(face_quads: [u32; 3]) -> [u32; 7] {
    let mut b = [0u32; 7];
    for i in 0..3 {
        let packed = face_quads[i];
        b[i * 2 + 1] = b[i * 2] + 6 * (packed & 0xFFFF);
        b[i * 2 + 2] = b[i * 2 + 1] + 6 * (packed >> 16);
    }
    b
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FaceRun {
    first_index: u32,
    index_count: u32,
}

/// Merge consecutive visible non-empty face buckets. At most [`MAX_FACE_RUNS`]
/// runs; mirrors the shader's `start = ~0u` loop exactly.
fn merge_face_runs(vis: [bool; 6], bounds: [u32; 7], out: &mut [FaceRun; 3]) -> usize {
    let mut n = 0;
    let mut start = u32::MAX;
    for k in 0..6u32 {
        let occ = vis[k as usize] && bounds[k as usize + 1] != bounds[k as usize];
        if occ {
            if start == u32::MAX {
                start = k;
            }
        } else if start != u32::MAX {
            out[n] = FaceRun {
                first_index: bounds[start as usize],
                index_count: bounds[k as usize] - bounds[start as usize],
            };
            n += 1;
            start = u32::MAX;
        }
    }
    if start != u32::MAX {
        out[n] = FaceRun {
            first_index: bounds[start as usize],
            index_count: bounds[6] - bounds[start as usize],
        };
        n += 1;
    }
    n
}

/// Where the cull appends a partition's commands. Counts every command, as
/// the shader's atomic does, and stores those within the partition's
/// capacity.
trait CmdSink {
    fn emit(&mut self, part: usize, cmd: DrawIndexedIndirect);
}

/// [`cpu_cull_into`]'s sink: one list per partition.
struct PartLists<'a> {
    partitions: &'a [PartitionGpu],
    part_cmds: &'a mut [Vec<DrawIndexedIndirect>],
    counts: &'a mut [u32],
}

impl CmdSink for PartLists<'_> {
    #[inline(always)]
    fn emit(&mut self, part: usize, cmd: DrawIndexedIndirect) {
        emit_part(part, cmd, self.partitions, self.part_cmds, self.counts);
    }
}

/// [`PartLists::emit`] on loose slices. [`cull_solid`] calls it directly:
/// there a sink struct held across the loop measured about 1 ns per slot
/// slower.
#[inline(always)]
fn emit_part(
    part: usize,
    cmd: DrawIndexedIndirect,
    partitions: &[PartitionGpu],
    part_cmds: &mut [Vec<DrawIndexedIndirect>],
    counts: &mut [u32],
) {
    let i = counts[part];
    counts[part] = i + 1;
    if i < partitions[part].capacity {
        part_cmds[part].push(cmd);
    }
}

/// The legacy mirror's sink: one array, each partition at its offset.
#[cfg(test)]
struct FlatCmds<'a> {
    partitions: &'a [PartitionGpu],
    cmds: &'a mut [DrawIndexedIndirect],
    counts: &'a mut [u32],
}

#[cfg(test)]
impl CmdSink for FlatCmds<'_> {
    #[inline(always)]
    fn emit(&mut self, part: usize, cmd: DrawIndexedIndirect) {
        let i = self.counts[part];
        self.counts[part] += 1;
        if i < self.partitions[part].capacity {
            self.cmds[(self.partitions[part].offset + i) as usize] = cmd;
        }
    }
}

/// Camera-relative box of one SoA slot, rounded as `cull.comp.slang` does:
/// `offset = (block - cam_block) - cam_frac + local_off`, then
/// `aabb * scale + offset` (`aabb` is stored scaled; the scale is a power of
/// two, so that product is exact). A caged slot has a zero offset and its
/// corner box: the shader's `corners_aabb + ((anchor - cam_block) - cam_frac)`
/// (`(anchor - cam_block) - cam_frac` is never `-0`, so adding the zero
/// changes no bit).
#[inline(always)]
fn cam_relative_soa(
    aabb: [f32; 6],
    local_off: [f32; 3],
    block: [i32; 3],
    eye: EyeSplit,
) -> ([f32; 3], [f32; 3]) {
    let r = eye.rel(block);
    let d = [
        r[0] + local_off[0],
        r[1] + local_off[1],
        r[2] + local_off[2],
    ];
    (
        [aabb[0] + d[0], aabb[1] + d[1], aabb[2] + d[2]],
        [aabb[3] + d[0], aabb[4] + d[1], aabb[5] + d[2]],
    )
}

/// Precompute the per-frame live ∩ (visible ∨ shadows) bitset. Arrival is
/// tested in the hot loop so a constant-true `is_arrived` can be DCE'd.
fn build_live_vis(
    live_bits: &[u32],
    visible: &[u32],
    slot_count: u32,
    shadow: bool,
    out: &mut Vec<u32>,
) {
    let words = slot_count.div_ceil(32) as usize;
    out.clear();
    out.resize(words, 0);
    if words == 0 {
        return;
    }
    if shadow {
        let n = words.min(live_bits.len());
        out[..n].copy_from_slice(&live_bits[..n]);
    } else {
        let n = words.min(live_bits.len()).min(visible.len());
        for i in 0..n {
            out[i] = live_bits[i] & visible[i];
        }
    }
    let rem = slot_count % 32;
    if rem != 0 {
        out[words - 1] &= (1u32 << rem) - 1;
    }
}

/// One frame's cull inputs besides the slot tables: what `CullParams` carries
/// to the shader.
#[derive(Clone, Copy)]
pub(crate) struct CullView<'a> {
    pub camera: &'a Frustum,
    /// The two cascade frusta, on frames that regenerate shadows.
    pub shadow: Option<&'a [Frustum; 2]>,
    pub eye: EyeSplit,
    /// Slots `0..slot_count` are culled.
    pub slot_count: u32,
    /// Full-res coverage box: a coarse-LOD box strictly inside it is not
    /// drawn. A non-positive half covers nothing.
    pub half: [f32; 3],
    pub centre: [f32; 3],
    /// Emit face runs for meshes that carry them.
    pub face_cull: bool,
}

fn frustum_planes(frustum: &Frustum) -> [[f32; 4]; 5] {
    frustum.planes().map(|p| p.to_array())
}

/// Camera-partition inputs, fixed for one cull.
#[derive(Clone, Copy)]
struct CameraBuckets {
    half: [f32; 3],
    centre: [f32; 3],
    /// Squared coarse-LOD edges, [`bucket_edge_sq`] of [`lod_bucket_scale`].
    lod_sq: [f32; 3],
    arena_count: usize,
}

impl CameraBuckets {
    fn new(view: &CullView, arena_count: usize) -> Self {
        Self {
            half: view.half,
            centre: view.centre,
            lod_sq: bucket_edge_sq(lod_bucket_scale()),
            arena_count,
        }
    }

    /// Camera partition of a box that passed the frustum, or `None` for a
    /// coarse-LOD box the clip would discard whole. Buckets on the box
    /// centre's squared distance against the group's edges.
    #[inline(always)]
    fn partition(&self, group: u32, arena: u32, mn: [f32; 3], mx: [f32; 3]) -> Option<usize> {
        let lod = is_lod_group(group);
        if lod && lod_aabb_inside_box(mn, mx, self.centre, self.half) {
            return None;
        }
        let cx = 0.5 * (mn[0] + mx[0]);
        let cy = 0.5 * (mn[1] + mx[1]);
        let cz = 0.5 * (mn[2] + mx[2]);
        let edges = if lod {
            self.lod_sq
        } else {
            FULL_BUCKET_EDGE_SQ
        };
        let bucket = distance_bucket_sq(cx * cx + cy * cy + cz * cz, edges);
        // `camera_part` in u32, as the shader computes it.
        let part = (group * self.arena_count as u32 + arena) * BUCKETS as u32 + bucket;
        debug_assert_eq!(
            part as usize,
            camera_part(
                group as usize,
                arena as usize,
                bucket as usize,
                self.arena_count
            )
        );
        Some(part as usize)
    }
}

/// Face-run directions of a camera draw, or `None` to draw it whole. A caged
/// slot (`corners` anchor-relative, `block` its anchor) tests the cage's
/// affine frame and is `None` when that frame is singular. A flat slot tests
/// its camera-relative box per axis and never reads `corners`.
#[inline(always)]
fn direction_vis(
    caged: bool,
    corners: &[[f32; 3]; 8],
    block: [i32; 3],
    eye: EyeSplit,
    mn: [f32; 3],
    mx: [f32; 3],
) -> Option<[bool; 6]> {
    if caged {
        cage_direction_vis(cam_relative_corners(*corners, block, eye)).map(|v| v.vis)
    } else {
        Some(face_vis(mn, mx))
    }
}

/// Emits one camera draw into `part` and counts each command in `group`'s
/// stats: the merged runs of the visible directions when `runs` carries them
/// with the slot's packed face-quad counts, else the whole mesh.
#[inline(always)]
fn emit_camera(
    out: &mut impl CmdSink,
    part: usize,
    cmd: DrawIndexedIndirect,
    runs: Option<([bool; 6], [u32; 3])>,
    group: u32,
    stats: &mut [u32; STATS_COUNT],
) {
    let Some((vis, face_quads)) = runs else {
        out.emit(part, cmd);
        count_draw(stats, group, cmd.index_count);
        return;
    };
    let mut merged = [FaceRun::default(); MAX_FACE_RUNS as usize];
    let n = merge_face_runs(vis, packed_face_bounds(face_quads), &mut merged);
    for run in merged.iter().take(n) {
        out.emit(
            part,
            DrawIndexedIndirect {
                first_index: run.first_index,
                index_count: run.index_count,
                ..cmd
            },
        );
        count_draw(stats, group, run.index_count);
    }
}

#[inline(always)]
fn count_draw(stats: &mut [u32; STATS_COUNT], group: u32, index_count: u32) {
    stats[group as usize * 2] += 1;
    stats[group as usize * 2 + 1] += index_count;
}

/// Host re-implementation of `computeMain` in `cull.comp.slang`. Appends each
/// partition's `DrawCmd`s to its own `scratch.part_cmds` list and counts them
/// in `scratch.counts`, past capacity included as the shader's atomics do
/// (both keep their allocations across frames).
pub(crate) fn cpu_cull_into(
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    view: &CullView,
    scratch: &mut CpuCullScratch,
) -> [u32; STATS_COUNT] {
    let npart = partitions.len();
    if scratch.part_cmds.len() < npart {
        scratch.part_cmds.resize_with(npart, Vec::new);
    }
    for v in &mut scratch.part_cmds[..npart] {
        v.clear();
    }
    scratch.counts.clear();
    scratch.counts.resize(npart, 0);

    let shadow_on = view.shadow.is_some();
    build_live_vis(
        dir.live_bits(),
        visible,
        view.slot_count,
        shadow_on,
        &mut scratch.live_vis,
    );

    let frame = SlotFrame {
        dir,
        visible,
        live_vis: &scratch.live_vis,
        cam_planes: frustum_planes(view.camera),
        shadow_planes: view
            .shadow
            .map(|f| [frustum_planes(&f[0]), frustum_planes(&f[1])]),
        eye: view.eye,
        slot_count: view.slot_count,
        buckets: CameraBuckets::new(view, dir.arena_count()),
    };
    let out = PartLists {
        partitions,
        part_cmds: &mut scratch.part_cmds,
        counts: &mut scratch.counts,
    };
    match (view.face_cull, shadow_on) {
        (false, false) => cull_solid(&frame, is_arrived, out),
        (false, true) => cull_slots::<false, true>(&frame, is_arrived, out),
        (true, false) => cull_slots::<true, false>(&frame, is_arrived, out),
        (true, true) => cull_slots::<true, true>(&frame, is_arrived, out),
    }
}

/// [`cpu_cull_into`] with the commands placed at their partition offsets.
#[cfg(test)]
pub(crate) fn cpu_cull(
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    view: &CullView,
) -> (Vec<DrawIndexedIndirect>, Vec<u32>, [u32; STATS_COUNT]) {
    let mut scratch = CpuCullScratch::default();
    let stats = cpu_cull_into(dir, is_arrived, visible, partitions, view, &mut scratch);
    (
        flatten_part_cmds(&scratch.part_cmds, partitions),
        scratch.counts,
        stats,
    )
}

/// Record-walking mirror of `computeMain`: the records' own boxes and fields
/// and the shader's `?:` p-vertex, with [`cull_slots`]'s partition, direction
/// and emission steps. The emission reference for the old-vs-new test.
#[cfg(test)]
fn cpu_cull_legacy(
    records: &[MeshRecord],
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    view: &CullView,
) -> (Vec<DrawIndexedIndirect>, Vec<u32>, [u32; STATS_COUNT]) {
    let total: usize = partitions.iter().map(|p| p.capacity as usize).sum();
    let mut cmds = vec![bytemuck::Zeroable::zeroed(); total];
    let mut counts = vec![0u32; partitions.len()];
    let mut stats = [0u32; STATS_COUNT];
    let cam_planes = frustum_planes(view.camera);
    let shadow_planes = view
        .shadow
        .map(|f| [frustum_planes(&f[0]), frustum_planes(&f[1])]);
    let buckets = CameraBuckets::new(view, dir.arena_count());
    let eye = view.eye;
    let mut out = FlatCmds {
        partitions,
        cmds: &mut cmds,
        counts: &mut counts,
    };

    for slot in 0..view.slot_count {
        if !is_arrived(slot) {
            continue;
        }
        let word = dir.arena_word(slot as usize);
        if word == 0 {
            continue;
        }
        let cam_visible = slot_visible(visible, slot);
        if !cam_visible && view.shadow.is_none() {
            continue;
        }
        let Some(rec) = records.get(slot as usize) else {
            continue;
        };
        let pass = (rec.detail_pass >> crate::genconst::DETAIL_GPU_BITS) & 3;
        if pass > 1 {
            continue;
        }
        let i = slot as usize;
        let bits = dir.cull_bits().get(i).copied().unwrap_or(0);
        let caged = super::arena::cull_bits_caged(bits);
        let block = dir.cull_blocks().get(i).copied().unwrap_or([0; 3]);
        let (mn, mx, scale) = if caged {
            let aabb = dir.cull_aabbs().get(i).copied().unwrap_or([0.0; 6]);
            let (mn, mx) = cam_relative_soa(aabb, [0.0; 3], block, eye);
            (mn, mx, rec.detail_scale())
        } else {
            cam_relative_aabb(rec, eye)
        };
        let arena = word - 1;
        let cmd = DrawIndexedIndirect {
            index_count: rec.index_count,
            instance_count: 1,
            first_index: 0,
            vertex_offset: rec.vertex_offset,
            first_instance: slot,
        };

        if cam_visible && aabb_in_planes_select(&cam_planes, mn, mx) {
            let group = camera_group(pass, scale > 1.0, caged);
            if let Some(part) = buckets.partition(group, arena, mn, mx) {
                let runs = if view.face_cull && (rec.flags & MESH_FLAG_FACE_RUNS) != 0 {
                    let corners = dir.cull_corners().get(i).unwrap_or(&[[0.0; 3]; 8]);
                    direction_vis(caged, corners, block, eye, mn, mx)
                        .map(|vis| (vis, rec.face_quads))
                } else {
                    None
                };
                emit_camera(&mut out, part, cmd, runs, group, &mut stats);
            }
        }

        if pass == 0
            && scale <= 1.0
            && let Some(planes) = &shadow_planes
        {
            for (c, plane) in planes.iter().enumerate() {
                if aabb_in_planes_select(plane, mn, mx) {
                    out.emit(shadow_part(c, arena as usize, buckets.arena_count), cmd);
                }
            }
        }
    }
    (cmds, counts, stats)
}

/// What the slot loop reads besides the directory's SoA, built once per cull.
struct SlotFrame<'a> {
    dir: &'a ArenaDirectory,
    /// Camera visibility. Read per slot only when shadows are on.
    visible: &'a [u32],
    /// [`build_live_vis`]: live ∩ (visible ∨ shadows).
    live_vis: &'a [u32],
    cam_planes: [[f32; 4]; 5],
    shadow_planes: Option<[[[f32; 4]; 5]; 2]>,
    eye: EyeSplit,
    slot_count: u32,
    buckets: CameraBuckets,
}

/// `cull_slots::<false, false>` written for the Fast/Minimum path: the
/// frustum test runs before any cull bit is decoded, a camera reject ends the
/// slot, and commands go straight to [`emit_part`]. The generic loop measured
/// about 6 % slower per slot here.
fn cull_solid(
    frame: &SlotFrame,
    is_arrived: impl Fn(u32) -> bool,
    out: PartLists,
) -> [u32; STATS_COUNT] {
    let PartLists {
        partitions,
        part_cmds,
        counts,
    } = out;
    let mut stats = [0u32; STATS_COUNT];
    let dir = frame.dir;
    let aabbs = dir.cull_aabbs();
    let local_offs = dir.cull_local_offs();
    let blocks = dir.cull_blocks();
    let bits_soa = dir.cull_bits();
    let index_counts = dir.cull_index_counts();
    let vertex_offsets = dir.cull_vertex_offsets();
    let live_vis = frame.live_vis;
    let cam_planes = &frame.cam_planes;
    let eye = frame.eye;
    let buckets = frame.buckets;

    for slot in 0..frame.slot_count {
        let w = (slot >> 5) as usize;
        if unsafe { *live_vis.get_unchecked(w) } & (1u32 << (slot & 31)) == 0 {
            continue;
        }
        if !is_arrived(slot) {
            continue;
        }
        let i = slot as usize;
        let bits = unsafe { *bits_soa.get_unchecked(i) };
        if bits == 0 {
            continue;
        }
        let aabb = unsafe { *aabbs.get_unchecked(i) };
        let local_off = unsafe { *local_offs.get_unchecked(i) };
        let block = unsafe { *blocks.get_unchecked(i) };
        let (mn, mx) = cam_relative_soa(aabb, local_off, block, eye);
        if !aabb_in_planes(cam_planes, mn, mx) {
            continue;
        }
        let group = camera_group(
            super::arena::cull_bits_pass(bits),
            super::arena::cull_bits_lod(bits),
            super::arena::cull_bits_caged(bits),
        );
        let arena = super::arena::cull_bits_arena(bits) - 1;
        let Some(part) = buckets.partition(group, arena, mn, mx) else {
            continue;
        };
        let cmd = DrawIndexedIndirect {
            index_count: unsafe { *index_counts.get_unchecked(i) },
            instance_count: 1,
            first_index: 0,
            vertex_offset: unsafe { *vertex_offsets.get_unchecked(i) },
            first_instance: slot,
        };
        emit_part(part, cmd, partitions, part_cmds, counts);
        count_draw(&mut stats, group, cmd.index_count);
    }
    stats
}

/// The SoA slot loop, one instance per face-run/shadow pair (the pair with
/// neither runs [`cull_solid`]).
fn cull_slots<const FACE: bool, const SHADOW: bool>(
    frame: &SlotFrame,
    is_arrived: impl Fn(u32) -> bool,
    mut out: PartLists,
) -> [u32; STATS_COUNT] {
    let mut stats = [0u32; STATS_COUNT];
    let dir = frame.dir;
    let aabbs = dir.cull_aabbs();
    let local_offs = dir.cull_local_offs();
    let blocks = dir.cull_blocks();
    let bits_soa = dir.cull_bits();
    let index_counts = dir.cull_index_counts();
    let vertex_offsets = dir.cull_vertex_offsets();
    let face_quads = dir.cull_face_quads();
    let corners_soa = dir.cull_corners();
    let live_vis = frame.live_vis;
    let visible = frame.visible;
    let cam_planes = frame.cam_planes;
    let shadow_planes = frame.shadow_planes;
    let eye = frame.eye;
    let buckets = frame.buckets;

    for slot in 0..frame.slot_count {
        let w = (slot >> 5) as usize;
        if unsafe { *live_vis.get_unchecked(w) } & (1u32 << (slot & 31)) == 0 {
            continue;
        }
        if !is_arrived(slot) {
            continue;
        }
        let i = slot as usize;
        let bits = unsafe { *bits_soa.get_unchecked(i) };
        if bits == 0 {
            continue;
        }
        let aabb = unsafe { *aabbs.get_unchecked(i) };
        let local_off = unsafe { *local_offs.get_unchecked(i) };
        let block = unsafe { *blocks.get_unchecked(i) };
        let (mn, mx) = cam_relative_soa(aabb, local_off, block, eye);
        let pass = super::arena::cull_bits_pass(bits);
        let lod = super::arena::cull_bits_lod(bits);
        let caged = super::arena::cull_bits_caged(bits);
        let arena = super::arena::cull_bits_arena(bits) - 1;

        // Without shadows `live_vis` already holds the camera visibility.
        let mut camera = None;
        if (!SHADOW || slot_visible(visible, slot)) && aabb_in_planes(&cam_planes, mn, mx) {
            let group = camera_group(pass, lod, caged);
            camera = buckets
                .partition(group, arena, mn, mx)
                .map(|part| (group, part));
        }
        // Shadow casters: full-res Opaque only, always whole.
        let mut sh0 = false;
        let mut sh1 = false;
        if SHADOW
            && pass == 0
            && !lod
            && let Some(planes) = &shadow_planes
        {
            sh0 = aabb_in_planes(&planes[0], mn, mx);
            sh1 = aabb_in_planes(&planes[1], mn, mx);
        }
        if camera.is_none() && !sh0 && !sh1 {
            continue;
        }

        let cmd = DrawIndexedIndirect {
            index_count: unsafe { *index_counts.get_unchecked(i) },
            instance_count: 1,
            first_index: 0,
            vertex_offset: unsafe { *vertex_offsets.get_unchecked(i) },
            first_instance: slot,
        };

        if let Some((group, part)) = camera {
            // `None` draws the mesh whole: face runs off, or a singular cage.
            let runs = if FACE && super::arena::cull_bits_face(bits) {
                let corners = unsafe { corners_soa.get_unchecked(i) };
                direction_vis(caged, corners, block, eye, mn, mx)
                    .map(|vis| (vis, unsafe { *face_quads.get_unchecked(i) }))
            } else {
                None
            };
            emit_camera(&mut out, part, cmd, runs, group, &mut stats);
        }
        if sh0 {
            out.emit(shadow_part(0, arena as usize, buckets.arena_count), cmd);
        }
        if sh1 {
            out.emit(shadow_part(1, arena as usize, buckets.arena_count), cmd);
        }
    }
    stats
}

/// Partition index for a camera (group, arena, bucket) triple.
#[inline]
pub(crate) fn camera_part(group: usize, arena: usize, bucket: usize, arena_count: usize) -> usize {
    debug_assert!(group < CAMERA_GROUPS);
    debug_assert!(bucket < BUCKETS);
    (group * arena_count + arena) * BUCKETS + bucket
}

/// Partition index for a shadow (cascade, arena) pair. Cascades are unbucketed.
#[inline]
pub(crate) fn shadow_part(cascade: usize, arena: usize, arena_count: usize) -> usize {
    debug_assert!(cascade < SHADOW_GROUPS);
    CAMERA_GROUPS * arena_count * BUCKETS + cascade * arena_count + arena
}

/// Partition table length for `arena_count` live arena rows.
pub(crate) fn partition_count(arena_count: usize) -> usize {
    arena_count * (CAMERA_GROUPS * BUCKETS + SHADOW_GROUPS)
}

/// Indirect-count draw calls a recorder would issue for `group`: partitions
/// with `capacity > 0`. Zero-capacity buckets (unreachable distance, empty
/// live lane) are skipped — including those the GPU still writes a 0 count
/// into. Matches the calls `record_groups` in [`super::scene_pass`] issues.
pub(crate) fn group_indirect_calls(
    partitions: &[PartitionGpu],
    group: Group,
    arena_count: usize,
) -> u32 {
    if arena_count == 0 {
        return 0;
    }
    let span = arena_count * BUCKETS;
    let base = group as usize * span;
    let end = base + span;
    if end > partitions.len() {
        return 0;
    }
    partitions[base..end]
        .iter()
        .filter(|p| p.capacity > 0)
        .count() as u32
}

/// GPU Partition struct.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PartitionGpu {
    pub offset: u32,
    pub capacity: u32,
}
const _: () = assert!(size_of::<PartitionGpu>() == 8);
const _: () = assert!(std::mem::offset_of!(PartitionGpu, offset) == 0);
const _: () = assert!(std::mem::offset_of!(PartitionGpu, capacity) == 4);

impl Group {
    /// Camera-group partition-table order (shadows are unbucketed after this).
    pub(crate) const ALL: [Group; CAMERA_GROUPS] = [
        Group::Opaque,
        Group::Cutout,
        Group::OpaqueLod,
        Group::Caged,
        Group::CagedLod,
    ];
}

#[cfg(test)]
mod shader_mirror;

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use ash::vk;
    use ash::vk::Handle;

    use super::super::arena::MeshAabb;
    use super::*;
    use crate::mesh::Pass;

    fn buf(raw: u64) -> vk::Buffer {
        vk::Buffer::from_raw(raw)
    }

    const G1: NonZeroU32 = NonZeroU32::new(1).unwrap();
    fn origin_eye() -> EyeSplit {
        EyeSplit {
            block: [0; 3],
            _pad0: 0,
            frac: [0.0; 3],
            _pad1: 0.0,
        }
    }

    /// The tests' view: eye at the origin, clip box centred on it.
    fn view<'a>(
        camera: &'a Frustum,
        shadow: Option<&'a [Frustum; 2]>,
        slot_count: u32,
        half: [f32; 3],
        face_cull: bool,
    ) -> CullView<'a> {
        CullView {
            camera,
            shadow,
            eye: origin_eye(),
            slot_count,
            half,
            centre: [0.0; 3],
            face_cull,
        }
    }

    const FULL: bool = false;
    const LOD: bool = true;

    #[test]
    fn group_order_is_the_partition_table_order() {
        for (i, g) in Group::ALL.iter().enumerate() {
            assert_eq!(*g as usize, i);
        }
        assert_eq!(CAMERA_GROUPS, Group::ALL.len());
    }

    fn centred(mn: [f32; 3], mx: [f32; 3], half: [f32; 3]) -> bool {
        lod_aabb_inside_box(mn, mx, [0.0; 3], half)
    }

    /// The pre-offset test: farthest corner from the origin, strictly inside.
    fn centred_abs(mn: [f32; 3], mx: [f32; 3], half: [f32; 3]) -> bool {
        if half[0] <= 0.0 || half[1] <= 0.0 || half[2] <= 0.0 {
            return false;
        }
        mn[0].abs().max(mx[0].abs()) < half[0]
            && mn[1].abs().max(mx[1].abs()) < half[1]
            && mn[2].abs().max(mx[2].abs()) < half[2]
    }

    #[test]
    fn lod_aabb_inside_box_matches_fragment_discard() {
        let half = [10.0, 8.0, 10.0];
        let cases = [
            ([-3.0, -2.0, -4.0], [1.0, 2.0, 2.0]),
            ([-1.0, -1.0, -1.0], [8.0, 1.0, 8.0]),
            ([0.0, -1.0, 0.0], [6.0, 1.0, 8.0]),
            ([-3.0, -9.0, -4.0], [3.0, 1.0, 4.0]),
            ([-1.0, -1.0, -1.0], [11.0, 1.0, 1.0]),
            ([0.0, 0.0, 0.0], [10.0, 1.0, 1.0]),
            ([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]),
        ];
        for (mn, mx) in cases {
            assert_eq!(
                centred(mn, mx, half),
                centred_abs(mn, mx, half),
                "centre 0 diverged for {mn:?}..{mx:?}"
            );
        }
        // Fully inside: farthest corner (3, 2, 4).
        assert!(centred([-3.0, -2.0, -4.0], [1.0, 2.0, 2.0], half));
        // A corner the old cylinder rejected (length ~11.3) is inside the box.
        assert!(centred([-1.0, -1.0, -1.0], [8.0, 1.0, 8.0], half));
        // |x| = 6 and |z| = 8 are both inside 10.
        assert!(centred([0.0, -1.0, 0.0], [6.0, 1.0, 8.0], half));
        // |y| = 9 is outside half.y = 8.
        assert!(!centred([-3.0, -9.0, -4.0], [3.0, 1.0, 4.0], half));
        // Outside on X.
        assert!(!centred([-1.0, -1.0, -1.0], [11.0, 1.0, 1.0], half));
        // On the face (abs == half) is not strictly inside.
        assert!(!centred([0.0, 0.0, 0.0], [10.0, 1.0, 1.0], half));
        // A non-positive component covers nothing.
        for closed in [[0.0, 8.0, 10.0], [10.0, 0.0, 10.0], [10.0, 8.0, 0.0]] {
            assert!(!centred([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], closed));
        }
    }

    #[test]
    fn lod_aabb_inside_box_offset_covers_the_open_box_only() {
        // [min, max] = (7, 13) × (−9, 1) × (−2, 6).
        let centre = [10.0, -4.0, 2.0];
        let half = [3.0, 5.0, 4.0];
        let inside = |mn: [f32; 3], mx: [f32; 3]| lod_aabb_inside_box(mn, mx, centre, half);

        assert!(inside([8.0, -8.0, -1.0], [12.0, 0.0, 5.0]));
        // Same AABB is outside the centred box of this half.
        assert!(!centred([8.0, -8.0, -1.0], [12.0, 0.0, 5.0], half));

        // Straddling each face (the face itself is not strictly inside).
        assert!(!inside([8.0, -1.0, 0.0], [13.0, 0.0, 1.0]), "+X");
        assert!(!inside([7.0, -1.0, 0.0], [12.0, 0.0, 1.0]), "-X");
        assert!(!inside([8.0, -1.0, 0.0], [12.0, 1.0, 1.0]), "+Y");
        assert!(!inside([8.0, -9.0, 0.0], [12.0, 0.0, 1.0]), "-Y");
        assert!(!inside([8.0, -1.0, 0.0], [12.0, 0.0, 6.0]), "+Z");
        assert!(!inside([8.0, -1.0, -2.0], [12.0, 0.0, 1.0]), "-Z");

        // Completely outside one face.
        assert!(!inside([13.0, -1.0, 0.0], [20.0, 0.0, 1.0]));
        assert!(!inside([-4.0, -1.0, 0.0], [6.0, 0.0, 1.0]));

        // min >= max on an axis → non-positive half → covers nothing.
        let degenerate = |min: [f32; 3], max: [f32; 3]| {
            let centre = std::array::from_fn(|i| (min[i] + max[i]) * 0.5);
            let half = std::array::from_fn(|i| (max[i] - min[i]) * 0.5);
            assert!(
                half[0] <= 0.0 || half[1] <= 0.0 || half[2] <= 0.0,
                "expected an empty axis"
            );
            assert!(!lod_aabb_inside_box([-100.0; 3], [100.0; 3], centre, half));
        };
        degenerate([2.0, -1.0, -1.0], [2.0, 1.0, 1.0]);
        degenerate([5.0, 0.0, 0.0], [1.0, 2.0, 2.0]);
        degenerate([-1.0, 4.0, -1.0], [1.0, 4.0, 1.0]);
        degenerate([-1.0, -1.0, 3.0], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn distance_bucket_splits_match_genconst_edges() {
        assert_eq!(distance_bucket(0.0), 0);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_0 - 0.01),
            0
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_0), 1);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_1 - 0.01),
            1
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_1), 2);
        assert_eq!(
            distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_2 - 0.01),
            2
        );
        assert_eq!(distance_bucket(crate::genconst::CULL_BUCKET_SPLIT_2), 3);
        assert_eq!(distance_bucket(1.0e6), 3);
    }

    #[test]
    fn lod_bucket_scale_one_matches_full_res_and_default_is_thirty_two() {
        assert_eq!(parse_lod_bucket_scale(None), DEFAULT_LOD_BUCKET_SCALE);
        assert_eq!(parse_lod_bucket_scale(Some("32")), 32.0);
        assert_eq!(parse_lod_bucket_scale(Some("1")), 1.0);
        assert_eq!(parse_lod_bucket_scale(Some("1.5")), 1.5);
        assert_eq!(parse_lod_bucket_scale(Some("0")), DEFAULT_LOD_BUCKET_SCALE);
        assert_eq!(parse_lod_bucket_scale(Some("-2")), DEFAULT_LOD_BUCKET_SCALE);
        assert_eq!(
            parse_lod_bucket_scale(Some("nan")),
            DEFAULT_LOD_BUCKET_SCALE
        );
        assert_eq!(
            parse_lod_bucket_scale(Some("nope")),
            DEFAULT_LOD_BUCKET_SCALE
        );
        assert_eq!(parse_lod_bucket_scale(Some("")), DEFAULT_LOD_BUCKET_SCALE);
        assert_eq!(
            lod_bucket_scale(),
            parse_lod_bucket_scale(std::env::var("VOXEL_LOD_BUCKET_SCALE").ok().as_deref())
        );
        assert_eq!(group_bucket_scale(Group::Opaque as u32), 1.0);
        assert_eq!(group_bucket_scale(Group::Cutout as u32), 1.0);
        assert_eq!(group_bucket_scale(Group::Caged as u32), 1.0);
        assert_eq!(
            group_bucket_scale(Group::OpaqueLod as u32),
            lod_bucket_scale()
        );
        assert_eq!(
            group_bucket_scale(Group::CagedLod as u32),
            lod_bucket_scale()
        );

        let dists = [
            0.0, 15.99, 16.0, 63.99, 64.0, 255.99, 256.0, 511.99, 512.0, 2047.99, 2048.0, 8191.99,
            8192.0, 1.0e6,
        ];
        for dist in dists {
            assert_eq!(
                distance_bucket_scaled(dist, 1.0),
                distance_bucket(dist),
                "scale 1 dist={dist}"
            );
        }
        assert_eq!(bucket_splits(32.0), [512.0, 2048.0, 8192.0]);
        assert_eq!(bucket_edge_sq(1.0), FULL_BUCKET_EDGE_SQ);
        assert_eq!(
            bucket_edge_sq(32.0),
            [512.0 * 512.0, 2048.0 * 2048.0, 8192.0 * 8192.0]
        );
        assert_eq!(distance_bucket_scaled(511.99, 32.0), 0);
        assert_eq!(distance_bucket_scaled(512.0, 32.0), 1);
        assert_eq!(distance_bucket_scaled(2047.99, 32.0), 1);
        assert_eq!(distance_bucket_scaled(2048.0, 32.0), 2);
        assert_eq!(distance_bucket_scaled(8191.99, 32.0), 2);
        assert_eq!(distance_bucket_scaled(8192.0, 32.0), 3);
        assert_eq!(distance_bucket_scaled(1.0e6, 32.0), 3);
    }

    fn look_neg_z() -> Frustum {
        let cam = crate::camera::Camera3D {
            position: glam::Vec3::ZERO,
            target: glam::Vec3::new(0.0, 0.0, -1.0),
            up: glam::Vec3::Y,
            fovy: 60.0,
            lens: crate::camera::Lens::Rectilinear,
        };
        Frustum::from_view_proj(&cam.view_proj(16.0 / 9.0))
    }

    fn opaque_rec(mn: [f32; 3], mx: [f32; 3]) -> MeshRecord {
        MeshRecord {
            block: [0; 3],
            detail_pass: u32::from(crate::mesh::Detail::FULL.to_gpu_bits()),
            local_off: [0.0; 3],
            cage: 0,
            aabb_min: mn,
            index_count: 36,
            aabb_max: mx,
            vertex_offset: 7,
            face_quads: [0; 3],
            flags: 0,
        }
    }

    fn face_rec(mn: [f32; 3], mx: [f32; 3]) -> MeshRecord {
        let mut rec = opaque_rec(mn, mx);
        rec.face_quads = [1 | (1 << 16), 1 | (1 << 16), 1 | (1 << 16)];
        rec.flags = MESH_FLAG_FACE_RUNS;
        rec
    }

    fn run_cpu(
        dir: &mut ArenaDirectory,
        records: &[MeshRecord],
        visible: &[u32],
        camera: &Frustum,
        face_cull: bool,
        half: [f32; 3],
    ) -> (
        Vec<PartitionGpu>,
        Vec<DrawIndexedIndirect>,
        Vec<u32>,
        [u32; STATS_COUNT],
    ) {
        let eye = origin_eye();
        for (i, rec) in records.iter().enumerate() {
            dir.note_cull_draw(i as u32, rec);
        }
        let mut parts = Vec::new();
        let runs = if face_cull { MAX_FACE_RUNS } else { 1 };
        dir.partitions_into(&mut parts, runs, Some(eye));
        let view = view(camera, None, dir.live_end(), half, face_cull);
        let (cmds, counts, stats) = cpu_cull(dir, |_| true, visible, &parts, &view);
        (parts, cmds, counts, stats)
    }

    fn part_cmds<'a>(
        parts: &[PartitionGpu],
        cmds: &'a [DrawIndexedIndirect],
        counts: &[u32],
        idx: usize,
    ) -> &'a [DrawIndexedIndirect] {
        let n = counts[idx].min(parts[idx].capacity) as usize;
        let start = parts[idx].offset as usize;
        &cmds[start..start + n]
    }

    #[test]
    fn cpu_cull_emits_a_mesh_in_front_and_skips_one_behind() {
        let camera = look_neg_z();
        let front = opaque_rec([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
        let behind = opaque_rec([-1.0, -1.0, 9.0], [1.0, 1.0, 11.0]);
        assert!(camera.intersects_aabb(
            glam::Vec3::from(front.aabb_min),
            glam::Vec3::from(front.aabb_max)
        ));
        assert!(!camera.intersects_aabb(
            glam::Vec3::from(behind.aabb_min),
            glam::Vec3::from(behind.aabb_max)
        ));

        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&front),
        );
        let (parts, cmds, counts, stats) =
            run_cpu(&mut dir, &[front], &[1], &camera, false, [0.0; 3]);
        let idx = camera_part(0, 0, 0, 1);
        assert_eq!(
            part_cmds(&parts, &cmds, &counts, idx),
            [DrawIndexedIndirect {
                index_count: 36,
                instance_count: 1,
                first_index: 0,
                vertex_offset: 7,
                first_instance: 0,
            }]
        );
        assert_eq!(stats[0], 1);
        assert_eq!(stats[1], 36);

        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&behind),
        );
        let (_, _, counts, stats) = run_cpu(&mut dir, &[behind], &[1], &camera, false, [0.0; 3]);
        assert!(counts.iter().all(|&c| c == 0));
        assert_eq!(stats, [0; STATS_COUNT]);
    }

    #[test]
    fn cpu_cull_keeps_a_box_that_straddles_the_near_plane() {
        let camera = look_neg_z();
        let rec = opaque_rec([-1.0, -1.0, -5.0], [1.0, 1.0, 5.0]);
        assert!(camera.intersects_aabb(
            glam::Vec3::from(rec.aabb_min),
            glam::Vec3::from(rec.aabb_max)
        ));
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&rec),
        );
        let (parts, cmds, counts, _) = run_cpu(&mut dir, &[rec], &[1], &camera, false, [0.0; 3]);
        let idx = camera_part(0, 0, 0, 1);
        assert_eq!(part_cmds(&parts, &cmds, &counts, idx).len(), 1);
    }

    #[test]
    fn cpu_cull_honours_the_visible_bitset() {
        let camera = look_neg_z();
        let rec = opaque_rec([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&rec),
        );
        let (_, _, counts, _) = run_cpu(&mut dir, &[rec], &[0], &camera, false, [0.0; 3]);
        assert!(counts.iter().all(|&c| c == 0));
    }

    #[test]
    fn cpu_cull_merges_face_runs_for_a_known_visibility_pattern() {
        let camera = look_neg_z();
        // In front, entirely in −X of the camera. vis = [T,T,T,F,T,F]:
        // +X +Y +Z merge, −Y is a second run, −X and −Z empty.
        let rec = face_rec([-10.0, -1.0, -15.0], [-5.0, 1.0, -10.0]);
        let vis = face_vis(rec.aabb_min, rec.aabb_max);
        assert_eq!(vis, [true, true, true, false, true, false]);
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&rec),
        );
        let (parts, cmds, counts, stats) = run_cpu(&mut dir, &[rec], &[1], &camera, true, [0.0; 3]);
        let idx = camera_part(0, 0, 0, 1);
        let got = part_cmds(&parts, &cmds, &counts, idx);
        assert_eq!(
            got,
            [
                DrawIndexedIndirect {
                    index_count: 18,
                    instance_count: 1,
                    first_index: 0,
                    vertex_offset: 7,
                    first_instance: 0,
                },
                DrawIndexedIndirect {
                    index_count: 6,
                    instance_count: 1,
                    first_index: 24,
                    vertex_offset: 7,
                    first_instance: 0,
                },
            ]
        );
        assert_eq!(stats[0], 2);
        assert_eq!(stats[1], 24);
    }

    #[test]
    fn cpu_cull_falls_back_to_a_whole_mesh_command_when_the_flag_is_clear() {
        let camera = look_neg_z();
        let rec = opaque_rec([-10.0, -1.0, -15.0], [-5.0, 1.0, -10.0]);
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            FULL,
            MeshAabb::from_record(&rec),
        );
        let (parts, cmds, counts, _) = run_cpu(&mut dir, &[rec], &[1], &camera, true, [0.0; 3]);
        let idx = camera_part(0, 0, 0, 1);
        assert_eq!(
            part_cmds(&parts, &cmds, &counts, idx),
            [DrawIndexedIndirect {
                index_count: 36,
                instance_count: 1,
                first_index: 0,
                vertex_offset: 7,
                first_instance: 0,
            }]
        );
    }

    #[test]
    fn cpu_and_independent_bucket_run_logic_agree() {
        // Independent: count how many splits a distance has crossed.
        let bucket_naive = |dist: f32, scale: f32| {
            CULL_BUCKET_SPLITS
                .iter()
                .filter(|&&s| dist >= s * scale)
                .count()
                .min(BUCKETS - 1) as u32
        };
        let dists = [
            0.0, 15.99, 16.0, 63.99, 64.0, 255.99, 256.0, 511.99, 512.0, 2047.99, 2048.0, 8191.99,
            8192.0, 1.0e6,
        ];
        for dist in dists {
            assert_eq!(
                distance_bucket(dist),
                bucket_naive(dist, 1.0),
                "dist={dist}"
            );
            for scale in [1.0, 32.0] {
                assert_eq!(
                    distance_bucket_scaled(dist, scale),
                    bucket_naive(dist, scale),
                    "dist={dist} scale={scale}"
                );
            }
        }
        // Independent: walk occupied faces and group consecutive runs.
        let runs_naive = |vis: [bool; 6], bounds: [u32; 7]| {
            let occ: [bool; 6] = std::array::from_fn(|k| vis[k] && bounds[k + 1] != bounds[k]);
            let mut out = Vec::new();
            let mut k = 0;
            while k < 6 {
                if !occ[k] {
                    k += 1;
                    continue;
                }
                let start = k;
                while k < 6 && occ[k] {
                    k += 1;
                }
                out.push(FaceRun {
                    first_index: bounds[start],
                    index_count: bounds[k] - bounds[start],
                });
            }
            out
        };
        let bounds = packed_face_bounds([1 | (1 << 16), 1 | (1 << 16), 1 | (1 << 16)]);
        assert_eq!(bounds, [0, 6, 12, 18, 24, 30, 36]);
        // Empty +X (bit 0) so some patterns have a hole.
        let bounds_hole = packed_face_bounds([0 | (1 << 16), 1 | (1 << 16), 1 | (1 << 16)]);
        for vis_bits in 0..64u8 {
            let vis = std::array::from_fn(|i| vis_bits & (1 << i) != 0);
            for b in [bounds, bounds_hole] {
                let mut got = [FaceRun {
                    first_index: 0,
                    index_count: 0,
                }; 3];
                let n = merge_face_runs(vis, b, &mut got);
                assert_eq!(
                    got[..n],
                    runs_naive(vis, b)[..],
                    "vis={vis_bits} bounds={b:?}"
                );
            }
        }
    }

    #[test]
    fn cpu_cull_skips_lod_meshes_fully_inside_the_box() {
        let camera = look_neg_z();
        let mut rec = opaque_rec([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
        rec.detail_pass = u32::from(crate::mesh::Detail(1).to_gpu_bits());
        let (mn, mx, scale) = cam_relative_aabb(&rec, origin_eye());
        assert!(scale > 1.0);
        assert!(lod_aabb_inside_box(mn, mx, [0.0; 3], [100.0; 3]));
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            LOD,
            MeshAabb::from_record(&rec),
        );
        let (_, _, counts, stats) = run_cpu(&mut dir, &[rec], &[1], &camera, false, [100.0; 3]);
        assert!(counts.iter().all(|&c| c == 0));
        assert_eq!(stats, [0; STATS_COUNT]);
    }

    /// Centre on the view axis at `dist` blocks in front of `look_neg_z`.
    fn front_box(dist: f32) -> ([f32; 3], [f32; 3]) {
        let z = -dist;
        ([-0.5, -0.5, z - 0.5], [0.5, 0.5, z + 0.5])
    }

    fn assert_only_bucket(
        parts: &[PartitionGpu],
        cmds: &[DrawIndexedIndirect],
        counts: &[u32],
        group: Group,
        arena: usize,
        arenas: usize,
        bucket: u32,
    ) {
        for b in 0..BUCKETS {
            let got = part_cmds(
                parts,
                cmds,
                counts,
                camera_part(group as usize, arena, b, arenas),
            );
            if b == bucket as usize {
                assert_eq!(got.len(), 1, "{group:?} arena {arena} bucket {b}");
            } else {
                assert!(
                    got.is_empty(),
                    "{group:?} arena {arena} bucket {b} held {}",
                    got.len()
                );
            }
        }
    }

    #[test]
    fn cpu_cull_buckets_lod_groups_with_the_scaled_splits() {
        let camera = look_neg_z();
        let scale = lod_bucket_scale();
        // 300 is past the full-res last edge (256). At the default scale 32 it
        // is still inside the first LOD edge (512). 3000 sits in LOD bucket 2
        // (2048..8192) and still in the full-res last bucket.
        let dists = [300.0, 3000.0];
        let mut records = Vec::new();
        let mut dir = ArenaDirectory::new();
        for (arena, &dist) in dists.iter().enumerate() {
            let (mn, mx) = front_box(dist);
            let aabb = MeshAabb {
                block: [0; 3],
                min: mn,
                max: mx,
                local_off: [0.0; 3],
            };
            let full = opaque_rec(mn, mx);
            let mut lod = opaque_rec(mn, mx);
            lod.detail_pass = u32::from(crate::mesh::Detail(1).to_gpu_bits());
            let mut caged = lod;
            caged.cage = 1;
            let base = records.len() as u32;
            dir.note_upload(base, G1, buf(arena as u64 + 1), Pass::Opaque, FULL, aabb);
            dir.note_upload(base + 1, G1, buf(arena as u64 + 1), Pass::Opaque, LOD, aabb);
            dir.note_upload_caged(
                base + 2,
                G1,
                buf(arena as u64 + 1),
                Pass::Opaque,
                LOD,
                aabb,
                aabb.box_corners(),
            );
            records.extend([full, lod, caged]);
        }
        let (parts, cmds, counts, stats) =
            run_cpu(&mut dir, &records, &[0b0011_1111], &camera, false, [0.0; 3]);
        let arenas = dists.len();
        assert_eq!(dir.arena_count(), arenas);
        for (arena, &dist) in dists.iter().enumerate() {
            let full_b = distance_bucket(dist);
            let lod_b = distance_bucket_scaled(dist, scale);
            assert_only_bucket(&parts, &cmds, &counts, Group::Opaque, arena, arenas, full_b);
            assert_only_bucket(
                &parts,
                &cmds,
                &counts,
                Group::OpaqueLod,
                arena,
                arenas,
                lod_b,
            );
            assert_only_bucket(
                &parts,
                &cmds,
                &counts,
                Group::CagedLod,
                arena,
                arenas,
                lod_b,
            );
            if scale > 1.0 {
                assert!(
                    lod_b < full_b,
                    "dist {dist}: LOD bucket {lod_b} should precede full-res {full_b}"
                );
            } else {
                assert_eq!(lod_b, full_b, "scale 1 keeps one set of edges");
            }
        }
        assert_eq!(stats[Group::Opaque as usize * 2], 2);
        assert_eq!(stats[Group::OpaqueLod as usize * 2], 2);
        assert_eq!(stats[Group::CagedLod as usize * 2], 2);
    }

    #[test]
    fn cpu_cull_max_defaults_to_1024_and_parses_env() {
        assert_eq!(CPU_CULL_MAX, 1024);
        assert_eq!(
            std::env::var("VOXEL_CPU_CULL_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(CPU_CULL_MAX),
            cpu_cull_max()
        );
    }

    #[test]
    fn branch_free_pvertex_matches_select_for_valid_aabbs() {
        let camera = look_neg_z();
        let planes = camera.planes().map(|p| p.to_array());
        let boxes = [
            ([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]),
            ([-1.0, -1.0, 9.0], [1.0, 1.0, 11.0]),
            ([-1.0, -1.0, -5.0], [1.0, 1.0, 5.0]),
            ([-10.0, -1.0, -15.0], [-5.0, 1.0, -10.0]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            ([-100.0, -50.0, -200.0], [100.0, 50.0, -4.0]),
        ];
        for (mn, mx) in boxes {
            assert_eq!(
                aabb_in_planes(&planes, mn, mx),
                aabb_in_planes_select(&planes, mn, mx),
                "mn={mn:?} mx={mx:?}"
            );
        }
    }

    fn assert_same_emission(
        parts: &[PartitionGpu],
        a_cmds: &[DrawIndexedIndirect],
        a_counts: &[u32],
        a_stats: [u32; STATS_COUNT],
        b_cmds: &[DrawIndexedIndirect],
        b_counts: &[u32],
        b_stats: [u32; STATS_COUNT],
    ) {
        assert_eq!(a_counts, b_counts, "per-partition counts");
        assert_eq!(a_stats, b_stats, "stats histogram");
        assert_eq!(a_counts.len(), parts.len());
        for (i, p) in parts.iter().enumerate() {
            let n = a_counts[i].min(p.capacity) as usize;
            let start = p.offset as usize;
            assert_eq!(
                &a_cmds[start..start + n],
                &b_cmds[start..start + n],
                "partition {i} commands"
            );
        }
    }

    fn rec_pass(mn: [f32; 3], mx: [f32; 3], pass: Pass, lod: bool) -> MeshRecord {
        let mut rec = opaque_rec(mn, mx);
        let detail = if lod {
            crate::mesh::Detail(1)
        } else {
            crate::mesh::Detail::FULL
        };
        rec.detail_pass =
            u32::from(detail.to_gpu_bits()) | ((pass as u32) << crate::genconst::DETAIL_GPU_BITS);
        rec
    }

    /// Mixed synthetic slot table: holes, Blend, LOD, face-runs, Cutout,
    /// two arenas, some hidden, some not-yet-arrived. Slot 48 is a Y-rotated
    /// caged mesh so the agreement covers cage direction runs.
    fn synthetic_slots() -> (ArenaDirectory, Vec<MeshRecord>, Vec<u32>, Vec<bool>) {
        const N: usize = 64;
        let mut dir = ArenaDirectory::new();
        let mut records = vec![bytemuck::Zeroable::zeroed(); N];
        let mut arrived = vec![true; N];
        let mut vis_bits = vec![0u32; N.div_ceil(32)];
        for slot in 0..N as u32 {
            vis_bits[(slot >> 5) as usize] |= 1 << (slot & 31);
        }
        for slot in 0..N as u32 {
            let i = slot as usize;
            let z = -10.0 - (slot % 17) as f32 * 8.0;
            let mn = [-1.0, -1.0, z - 1.0];
            let mx = [1.0, 1.0, z + 1.0];
            // Slot 48: Y-rotated cage in front. Its direction mask is not the
            // AABB mask, so a path that skips the cage frame disagrees.
            let (rec, corners) = if slot == 48 {
                let corners = y90_cage_at([0.0, 0.0, -40.0]);
                let (min, max) = crate::cage::corner_aabb(corners);
                let mut rec = face_rec(min, max);
                rec.cage = 1;
                (rec, Some(corners))
            } else {
                (
                    match slot % 7 {
                        0 => opaque_rec([mn[0], mn[1], 9.0], [mx[0], mx[1], 11.0]), // behind
                        1 => rec_pass(mn, mx, Pass::Cutout, false),
                        2 => rec_pass(mn, mx, Pass::Opaque, true),
                        3 => face_rec(mn, mx),
                        4 => rec_pass(mn, mx, Pass::Blend, false),
                        5 if slot % 5 == 0 => continue, // hole
                        _ => opaque_rec(mn, mx),
                    },
                    None,
                )
            };
            records[i] = rec;
            let lod = rec.detail_scale() > 1.0;
            let buf_id = 1 + u64::from(slot % 2);
            if let Some(corners) = corners {
                let aabb = MeshAabb {
                    block: rec.block,
                    min: rec.aabb_min,
                    max: rec.aabb_max,
                    local_off: [0.0; 3],
                };
                dir.note_upload_caged(slot, G1, buf(buf_id), rec.pass(), lod, aabb, corners);
            } else {
                dir.note_upload(
                    slot,
                    G1,
                    buf(buf_id),
                    rec.pass(),
                    lod,
                    MeshAabb::from_record(&rec),
                );
            }
            if slot % 11 == 0 {
                vis_bits[(slot >> 5) as usize] &= !(1 << (slot & 31));
            }
            if slot % 13 == 0 {
                arrived[i] = false;
            }
        }
        for (i, rec) in records.iter().enumerate() {
            dir.note_cull_draw(i as u32, rec);
        }
        (dir, records, vis_bits, arrived)
    }

    #[test]
    fn cpu_cull_new_matches_legacy_emission() {
        let camera = look_neg_z();
        let shadow_frusta = [look_neg_z(), look_neg_z()];
        let (mut dir, records, visible, arrived) = synthetic_slots();
        let is_arrived = |s: u32| arrived.get(s as usize).copied().unwrap_or(false);
        let eye = origin_eye();
        for face_cull in [false, true] {
            for with_shadow in [false, true] {
                for half in [[0.0, 0.0, 0.0], [40.0, 12.0, 25.0]] {
                    let mut parts = Vec::new();
                    let runs = if face_cull { MAX_FACE_RUNS } else { 1 };
                    dir.partitions_into(&mut parts, runs, Some(eye));
                    let shadow = with_shadow.then_some(&shadow_frusta);
                    let view = view(&camera, shadow, dir.live_end(), half, face_cull);
                    let (old_cmds, old_counts, old_stats) =
                        cpu_cull_legacy(&records, &dir, is_arrived, &visible, &parts, &view);
                    let (new_cmds, new_counts, new_stats) =
                        cpu_cull(&dir, is_arrived, &visible, &parts, &view);
                    assert_same_emission(
                        &parts,
                        &old_cmds,
                        &old_counts,
                        old_stats,
                        &new_cmds,
                        &new_counts,
                        new_stats,
                    );
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn cpu_cull_timing_10k_slots() {
        const N: usize = 10_000;
        const WARMUP: u32 = 20;
        const ITERS: u32 = 200;
        let camera = look_neg_z();
        let eye = origin_eye();
        let mut dir = ArenaDirectory::new();
        let mut records = Vec::with_capacity(N);
        for slot in 0..N as u32 {
            let rec = opaque_rec([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
            dir.note_upload(
                slot,
                G1,
                buf(1),
                Pass::Opaque,
                FULL,
                MeshAabb::from_record(&rec),
            );
            records.push(rec);
        }
        for (i, rec) in records.iter().enumerate() {
            dir.note_cull_draw(i as u32, rec);
        }
        let visible = vec![u32::MAX; N.div_ceil(32)];
        let mut parts = Vec::new();
        dir.partitions_into(&mut parts, 1, Some(eye));
        let view = view(&camera, None, dir.live_end(), [0.0; 3], false);
        let mut scratch = CpuCullScratch::default();

        fn time_ns(iters: u32, mut body: impl FnMut()) -> f64 {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                body();
            }
            start.elapsed().as_nanos() as f64 / f64::from(iters)
        }

        for _ in 0..WARMUP {
            let _ = cpu_cull_legacy(&records, &dir, |_| true, &visible, &parts, &view);
            let _ = cpu_cull_into(&dir, |_| true, &visible, &parts, &view, &mut scratch);
        }

        // Include the host-visible fill: old path memcpy'd the whole sparse
        // command buffer (including unused capacity); new copies each live
        // partition range only.
        let total: usize = parts.iter().map(|p| p.capacity as usize).sum();
        let mut wc_cmds = vec![0u8; total * CMD_STRIDE as usize];
        let mut wc_counts = vec![0u8; parts.len() * 4];

        let old_ns = time_ns(ITERS, || {
            let (cmds, counts, stats) =
                cpu_cull_legacy(&records, &dir, |_| true, &visible, &parts, &view);
            let cmd_bytes: &[u8] = bytemuck::cast_slice(&cmds);
            wc_cmds[..cmd_bytes.len()].copy_from_slice(cmd_bytes);
            let count_bytes: &[u8] = bytemuck::cast_slice(&counts);
            wc_counts[..count_bytes.len()].copy_from_slice(count_bytes);
            std::hint::black_box((&wc_cmds, &wc_counts, stats));
        });
        let new_ns = time_ns(ITERS, || {
            let stats = cpu_cull_into(&dir, |_| true, &visible, &parts, &view, &mut scratch);
            for (i, p) in parts.iter().enumerate() {
                let src = &scratch.part_cmds[i];
                if src.is_empty() {
                    continue;
                }
                let bytes: &[u8] = bytemuck::cast_slice(src);
                let start = p.offset as usize * CMD_STRIDE as usize;
                wc_cmds[start..start + bytes.len()].copy_from_slice(bytes);
            }
            let count_bytes: &[u8] = bytemuck::cast_slice(&scratch.counts);
            wc_counts[..count_bytes.len()].copy_from_slice(count_bytes);
            std::hint::black_box((&wc_cmds, &wc_counts, stats));
        });
        let old_per = old_ns / N as f64;
        let new_per = new_ns / N as f64;
        println!(
            "cpu_cull timing: legacy {old_per:.2} ns/slot, new {new_per:.2} ns/slot (10k slots, {ITERS} iters)"
        );
        assert!(
            new_per < old_per,
            "new path should be cheaper: {old_per:.2} -> {new_per:.2} ns/slot"
        );
    }

    #[test]
    fn caged_slot_culls_by_its_corner_box_and_draws_whole_with_the_flag_clear() {
        let camera = look_neg_z();
        let front = MeshAabb {
            block: [0; 3],
            min: [-1.0, -1.0, -11.0],
            max: [1.0, 1.0, -9.0],
            local_off: [0.0; 3],
        };
        let behind = MeshAabb {
            block: [0; 3],
            min: [-1.0, -1.0, 9.0],
            max: [1.0, 1.0, 11.0],
            local_off: [0.0; 3],
        };
        // In the frustum, and strictly inside a 10-block clip box.
        let inside = MeshAabb {
            block: [0; 3],
            min: [-1.0, -1.0, -6.0],
            max: [1.0, 1.0, -4.0],
            local_off: [0.0; 3],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, false, front);
        dir.note_upload_caged(
            1,
            G1,
            buf(1),
            Pass::Opaque,
            false,
            front,
            front.box_corners(),
        );
        dir.note_upload_caged(
            2,
            G1,
            buf(1),
            Pass::Opaque,
            false,
            behind,
            behind.box_corners(),
        );
        dir.note_upload_caged(
            3,
            G1,
            buf(1),
            Pass::Opaque,
            true,
            inside,
            inside.box_corners(),
        );

        let mut kept = opaque_rec(behind.min, behind.max);
        kept.cage = 1;
        kept.face_quads = [1 | (1 << 16), 1 | (1 << 16), 1 | (1 << 16)];
        // Face runs are on for this cull, but a clear MESH_FLAG_FACE_RUNS
        // draws a caged mesh whole like a flat one. A set flag would split it
        // by cage direction (caged_face_runs_agree_between_paths_...).
        kept.flags = 0;
        let mut hidden = opaque_rec(front.min, front.max);
        hidden.cage = 2;
        let mut lod = opaque_rec(inside.min, inside.max);
        lod.cage = 3;
        let records = [opaque_rec(front.min, front.max), kept, hidden, lod];
        let (parts, cmds, counts, stats) = run_cpu(
            &mut dir,
            &records,
            &[0b1111],
            &camera,
            true,
            [10.0, 10.0, 10.0],
        );
        let opaque = part_cmds(
            &parts,
            &cmds,
            &counts,
            camera_part(Group::Opaque as usize, 0, 0, 1),
        );
        assert_eq!(opaque.len(), 1);
        assert_eq!(opaque[0].first_instance, 0);
        let caged = part_cmds(
            &parts,
            &cmds,
            &counts,
            camera_part(Group::Caged as usize, 0, 0, 1),
        );
        assert_eq!(caged.len(), 1, "corner box in front is kept");
        assert_eq!(caged[0].first_instance, 1);
        assert_eq!(caged[0].first_index, 0, "flag clear: drawn whole");
        assert_eq!(caged[0].index_count, 36);
        assert!(
            cmds.iter().all(|c| c.first_instance != 2),
            "a corner box behind the camera is culled even when the local box is in front"
        );
        let lod_cmds = part_cmds(
            &parts,
            &cmds,
            &counts,
            camera_part(Group::CagedLod as usize, 0, 0, 1),
        );
        assert!(
            lod_cmds.is_empty(),
            "a caged LOD mesh inside the clip box is not drawn"
        );
        assert_eq!(stats[Group::Caged as usize * 2], 1);
        assert_eq!(stats[Group::Caged as usize * 2 + 1], 36);
    }

    /// Half-8 cube, 90° about Y: `(x, y, z) → (z, y, −x)`, then translated so
    /// the cube centre sits at `center`. Local +X becomes world −Z.
    fn y90_cage_at(center: [f32; 3]) -> [[f32; 3]; 8] {
        let half = 8.0;
        std::array::from_fn(|i| {
            let local = [
                if i & 1 == 0 { -half } else { half },
                if i & 2 == 0 { -half } else { half },
                if i & 4 == 0 { -half } else { half },
            ];
            [
                local[2] + center[0],
                local[1] + center[1],
                -local[0] + center[2],
            ]
        })
    }

    fn aa_corners(mn: [f32; 3], mx: [f32; 3]) -> [[f32; 3]; 8] {
        std::array::from_fn(|i| {
            [
                if i & 1 == 0 { mn[0] } else { mx[0] },
                if i & 2 == 0 { mn[1] } else { mx[1] },
                if i & 4 == 0 { mn[2] } else { mx[2] },
            ]
        })
    }

    fn cam_shift(corners: [[f32; 3]; 8], cam: [f32; 3]) -> [[f32; 3]; 8] {
        std::array::from_fn(|i| sub3(corners[i], cam))
    }

    fn rotated_cage(q: glam::Quat, center: glam::Vec3) -> [[f32; 3]; 8] {
        let half = 8.0;
        std::array::from_fn(|i| {
            let local = glam::Vec3::new(
                if i & 1 == 0 { -half } else { half },
                if i & 2 == 0 { -half } else { half },
                if i & 4 == 0 { -half } else { half },
            );
            (q * local + center).to_array()
        })
    }

    /// Four parameter-corners of the plane `t[axis] = s`. Free axes are
    /// `(axis+1)%3` then `(axis+2)%3`, so `cross(+a, +b)` is the +face normal.
    fn plane_corners(corners: [[f32; 3]; 8], axis: usize, s: f32) -> [glam::Vec3; 4] {
        let a = (axis + 1) % 3;
        let b = (axis + 2) % 3;
        std::array::from_fn(|i| {
            let mut t = [0.0; 3];
            t[axis] = s;
            t[a] = if i & 1 == 0 { 0.0 } else { 1.0 };
            t[b] = if i & 2 == 0 { 0.0 } else { 1.0 };
            glam::Vec3::from(crate::cage::trilinear(corners, t))
        })
    }

    /// Camera at the origin, in front of triangle `(a, b, c)`. Degenerate
    /// area is not a front face.
    fn tri_faces_camera(a: glam::Vec3, b: glam::Vec3, c: glam::Vec3) -> bool {
        let n = (b - a).cross(c - a);
        let area = n.length();
        area >= 1.0e-4 && n.dot(a) < -1.0e-4 * area
    }

    /// Upload slot `slot` (+X,+Y,+Z,−X,−Y,−Z) has a front-facing triangle on
    /// some plane `s = i/16`. +faces wind `(A,B,D)` and `(A,D,C)`; −faces flip.
    fn direction_has_front_triangle(corners: [[f32; 3]; 8], slot: usize) -> bool {
        let axis = slot % 3;
        let neg = slot >= 3;
        for i in 0..=16 {
            let p = plane_corners(corners, axis, i as f32 / 16.0);
            let front = if neg {
                tri_faces_camera(p[0], p[2], p[3]) || tri_faces_camera(p[0], p[3], p[1])
            } else {
                tri_faces_camera(p[0], p[1], p[3]) || tri_faces_camera(p[0], p[3], p[2])
            };
            if front {
                return true;
            }
        }
        false
    }

    /// A culled direction has no front-facing sample. Singular frames draw
    /// whole, so they are not checked here.
    fn assert_cull_keeps_front_triangles(corners: [[f32; 3]; 8]) {
        let Some(got) = cage_direction_vis(corners) else {
            return;
        };
        for slot in 0..6 {
            if got.vis[slot] {
                continue;
            }
            let axis = slot % 3;
            let neg = slot >= 3;
            for i in 0..=16 {
                let s = i as f32 / 16.0;
                let p = plane_corners(corners, axis, s);
                let hit = if neg {
                    tri_faces_camera(p[0], p[2], p[3]) || tri_faces_camera(p[0], p[3], p[1])
                } else {
                    tri_faces_camera(p[0], p[1], p[3]) || tri_faces_camera(p[0], p[3], p[2])
                };
                if hit {
                    let e0 = sub3(corners[1], corners[0]);
                    let e1 = sub3(corners[2], corners[0]);
                    let e2 = sub3(corners[4], corners[0]);
                    panic!(
                        "culled slot {slot} s={s} faces the camera (d={}, vis={:?})\nP0={:?} e0={:?} e1={:?} e2={:?}\ntri={:?}",
                        got.d, got.vis, corners[0], e0, e1, e2, p
                    );
                }
            }
        }
    }

    /// Affine cages: the kept set is exactly the directions with a front face.
    fn assert_affine_vis_matches_triangles(corners: [[f32; 3]; 8]) {
        let got = cage_direction_vis(corners).expect("affine frame");
        assert!(got.d < 1.0e-4, "affine cage d={}", got.d);
        for slot in 0..6 {
            assert_eq!(
                got.vis[slot],
                direction_has_front_triangle(corners, slot),
                "slot {slot} vis={:?} d={}",
                got.vis,
                got.d
            );
        }
    }

    fn face_near_camera(mn: [f32; 3], mx: [f32; 3]) -> bool {
        (0..3).any(|k| (-1.0..1.0).contains(&mn[k]) || (-1.0..1.0).contains(&mx[k]))
    }

    #[test]
    fn cage_direction_vis_matches_aabb_keeps_inside_and_follows_rotation() {
        let samples = [-80.0, -20.0, 0.0, 20.0, 80.0];
        let half = 8.0;
        for x in samples {
            for y in samples {
                for z in samples {
                    let mn = [x - half, y - half, z - half];
                    let mx = [x + half, y + half, z + half];
                    if face_near_camera(mn, mx) {
                        continue;
                    }
                    let corners = aa_corners(mn, mx);
                    let got = cage_direction_vis(corners).unwrap();
                    assert_eq!(got.vis, face_vis(mn, mx), "centre ({x}, {y}, {z})");
                    assert!(got.d < 1.0e-5, "d={}", got.d);
                    assert_affine_vis_matches_triangles(corners);
                }
            }
        }

        // On the +X plane the bias keeps the bucket; the AABB test drops it.
        let on_plane = aa_corners([0.0, -8.0, -8.0], [16.0, 8.0, 8.0]);
        assert!(cage_direction_vis(on_plane).unwrap().vis[0]);
        assert!(!face_vis([0.0, -8.0, -8.0], [16.0, 8.0, 8.0])[0]);

        let inside = aa_corners([-8.0; 3], [8.0; 3]);
        assert_eq!(cage_direction_vis(inside).unwrap().vis, [true; 6]);
        assert_affine_vis_matches_triangles(inside);

        let y90 = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let mapped = y90 * glam::Vec3::X;
        assert!(
            (mapped - glam::Vec3::new(0.0, 0.0, -1.0)).length() < 1.0e-5,
            "glam Y+90 sent +X to {mapped}"
        );
        let y_front = y90_cage_at([0.0, 0.0, -40.0]);
        for i in 0..8 {
            let d = sub3(
                y_front[i],
                rotated_cage(y90, glam::Vec3::new(0.0, 0.0, -40.0))[i],
            );
            assert!(len3(d) < 1.0e-4, "corner {i} disagrees with glam");
        }
        let got = cage_direction_vis(y_front).unwrap();
        assert_eq!(got.vis, [false, true, true, true, true, true]);
        assert!(got.d < 1.0e-5, "d={}", got.d);
        let (mn, mx) = crate::cage::corner_aabb(y_front);
        // A Y-rotated cube has the same world AABB; that mask is the wrong one.
        assert_ne!(got.vis, face_vis(mn, mx));
        assert_affine_vis_matches_triangles(y_front);

        let y_back = y90_cage_at([0.0, 0.0, 40.0]);
        assert_eq!(
            cage_direction_vis(y_back).unwrap().vis,
            [true, true, true, false, true, true]
        );
        assert_affine_vis_matches_triangles(y_back);

        let y_in = y90_cage_at([0.0, 0.0, 0.0]);
        assert_eq!(cage_direction_vis(y_in).unwrap().vis, [true; 6]);
        assert_affine_vis_matches_triangles(y_in);

        // 30° about (1,1,1). Camera sits 40 blocks from the centre along the
        // rotated local +X, so t_cam ≈ (3, 0.5, 0.5).
        let axis = glam::Vec3::new(1.0, 1.0, 1.0).normalize();
        let q = glam::Quat::from_axis_angle(axis, 30.0_f32.to_radians());
        let local_px = q * glam::Vec3::X;
        let plus = rotated_cage(q, -local_px * 40.0);
        let got = cage_direction_vis(plus).unwrap();
        assert_eq!(got.vis, [true, true, true, false, true, true]);
        assert!(got.d < 1.0e-4, "d={}", got.d);
        assert!(dot3((-local_px * 40.0).to_array(), sub3(plus[1], plus[0])) < 0.0);
        assert_affine_vis_matches_triangles(plus);

        let minus = rotated_cage(q, local_px * 40.0);
        assert_eq!(
            cage_direction_vis(minus).unwrap().vis,
            [false, true, true, true, true, true]
        );
        assert_affine_vis_matches_triangles(minus);
    }

    #[test]
    fn bent_cage_widens_eps_and_never_culls_a_front_face() {
        let flat = aa_corners([0.0; 3], [16.0; 3]);
        let mut bent = flat;
        bent[3] = add3(bent[3], [0.0, 6.0, 0.0]);
        bent[5] = add3(bent[5], [4.0, 0.0, -3.0]);
        bent[6] = add3(bent[6], [-2.0, 5.0, 4.0]);
        bent[7] = add3(bent[7], [3.0, -8.0, 6.0]);
        let expect_d = 109.0_f32.sqrt() / 16.0;
        let got = cage_direction_vis(bent).unwrap();
        assert!(
            (got.d - expect_d).abs() < 1.0e-5,
            "d={} expect {expect_d}",
            got.d
        );
        assert!(got.d > 0.2);

        // Corner 7 only, offset (0, 8, 0): d = 8/16 exactly.
        let mut poke = flat;
        poke[7][1] += 8.0;
        let poke_d = cage_direction_vis(poke).unwrap().d;
        assert!((poke_d - 0.5).abs() < 1.0e-6, "d={poke_d}");

        // Flat at camera-relative mn.x = 0.05 culls +X (t.x = -0.003125, eps = 0.001).
        // Both bends keep it.
        let cam = [-0.05, 8.0, 8.0];
        let flat_vis = cage_direction_vis(cam_shift(flat, cam)).unwrap();
        assert!(flat_vis.d < 1.0e-5);
        assert!(!flat_vis.vis[0], "flat should cull +X");
        assert!(cage_direction_vis(cam_shift(bent, cam)).unwrap().vis[0]);
        assert!(cage_direction_vis(cam_shift(poke, cam)).unwrap().vis[0]);

        // Neighbourhood of the cage, step 4. Both strong bends keep every
        // sampled front face here; the flat cage matches triangle-for-triangle
        // except on the 1e-3 band around a face plane.
        for x in (-16..=32).step_by(4) {
            for y in (-16..=32).step_by(4) {
                for z in (-16..=32).step_by(4) {
                    let at = [x as f32, y as f32, z as f32];
                    let flat_rel = cam_shift(flat, at);
                    assert_cull_keeps_front_triangles(flat_rel);
                    assert_cull_keeps_front_triangles(cam_shift(bent, at));
                    assert_cull_keeps_front_triangles(cam_shift(poke, at));
                    // Skip the 1e-3 band, where the bias keeps a direction the
                    // triangle test (strict half-space) does not call front.
                    let on_plane = (0..3).any(|k| at[k].abs() < 1.0 || (at[k] - 16.0).abs() < 1.0);
                    if !on_plane {
                        assert_affine_vis_matches_triangles(flat_rel);
                    }
                }
            }
        }

        // One-block kink (d = 1/16, eps ≈ 0.126). Checked through the view
        // distances of a chunk: still no culled front face.
        let mut kink = flat;
        kink[7][1] += 1.0;
        let kink_d = cage_direction_vis(kink).unwrap().d;
        assert!((kink_d - 1.0 / 16.0).abs() < 1.0e-6, "d={kink_d}");
        let far = [-40.0, -4.0, 8.0, 24.0, 64.0];
        for x in far {
            for y in far {
                for z in far {
                    assert_cull_keeps_front_triangles(cam_shift(kink, [x, y, z]));
                    assert_cull_keeps_front_triangles(cam_shift(flat, [x, y, z]));
                }
            }
        }
        for x in (-48..=64).step_by(4) {
            for y in (-48..=64).step_by(4) {
                for z in (-48..=64).step_by(4) {
                    let at = [x as f32, y as f32, z as f32];
                    assert_cull_keeps_front_triangles(cam_shift(kink, at));
                }
            }
        }

        // Affine centre (corners 0, 1, 2, 4) puts t_cam at 0.5: all six stay.
        let e0 = sub3(bent[1], bent[0]);
        let e1 = sub3(bent[2], bent[0]);
        let e2 = sub3(bent[4], bent[0]);
        let mid = add3(
            bent[0],
            [
                0.5 * (e0[0] + e1[0] + e2[0]),
                0.5 * (e0[1] + e1[1] + e2[1]),
                0.5 * (e0[2] + e1[2] + e2[2]),
            ],
        );
        assert_eq!(
            cage_direction_vis(cam_shift(bent, mid)).unwrap().vis,
            [true; 6]
        );
        assert_eq!(
            cage_direction_vis(cam_shift(flat, mid)).unwrap().vis,
            [true; 6]
        );
        assert_eq!(
            cage_direction_vis(cam_shift(poke, mid)).unwrap().vis,
            [true; 6]
        );

        let mut dead = flat;
        dead[1] = dead[0];
        assert!(cage_direction_vis(dead).is_none());
        assert!(cage_direction_vis([[0.0; 3]; 8]).is_none());
    }

    fn caged_camera_cmds<'a>(
        parts: &[PartitionGpu],
        cmds: &'a [DrawIndexedIndirect],
        counts: &[u32],
        slot: u32,
    ) -> Vec<&'a DrawIndexedIndirect> {
        let mut out = Vec::new();
        for bucket in 0..BUCKETS {
            let idx = camera_part(Group::Caged as usize, 0, bucket, 1);
            for cmd in part_cmds(parts, cmds, counts, idx) {
                if cmd.first_instance == slot {
                    out.push(cmd);
                }
            }
        }
        out
    }

    #[test]
    fn caged_face_runs_agree_between_paths_and_shadows_stay_whole() {
        let camera = look_neg_z();
        let shadow_frusta = [look_neg_z(), look_neg_z()];
        let eye = origin_eye();

        let side_mn = [-10.0, -1.0, -15.0];
        let side_mx = [-5.0, 1.0, -10.0];
        let mut side = face_rec(side_mn, side_mx);
        side.cage = 1;
        let side_corners = aa_corners(side_mn, side_mx);
        assert_eq!(
            cage_direction_vis(side_corners).unwrap().vis,
            [true, true, true, false, true, false]
        );

        let y_corners = y90_cage_at([0.0, 0.0, -40.0]);
        let (y_mn, y_mx) = crate::cage::corner_aabb(y_corners);
        let mut rotated = face_rec(y_mn, y_mx);
        rotated.cage = 2;
        rotated.vertex_offset = 11;
        assert_eq!(
            cage_direction_vis(y_corners).unwrap().vis,
            [false, true, true, true, true, true]
        );
        // AABB face_vis of this box is [T,T,T,T,T,F] (one run 0..30).
        assert_ne!(
            cage_direction_vis(y_corners).unwrap().vis,
            face_vis(y_mn, y_mx)
        );

        let mut singular = face_rec([-1.0, -1.0, -12.0], [1.0, 1.0, -8.0]);
        singular.cage = 3;
        singular.vertex_offset = 13;
        assert!(cage_direction_vis([[0.0; 3]; 8]).is_none());

        let records = [side, rotated, singular];
        let mut dir = ArenaDirectory::new();
        dir.note_upload_caged(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            false,
            MeshAabb {
                block: [0; 3],
                min: side_mn,
                max: side_mx,
                local_off: [0.0; 3],
            },
            side_corners,
        );
        dir.note_upload_caged(
            1,
            G1,
            buf(1),
            Pass::Opaque,
            false,
            MeshAabb {
                block: [0; 3],
                min: y_mn,
                max: y_mx,
                local_off: [0.0; 3],
            },
            y_corners,
        );
        dir.note_upload_caged(
            2,
            G1,
            buf(1),
            Pass::Opaque,
            false,
            MeshAabb {
                block: [0; 3],
                min: singular.aabb_min,
                max: singular.aabb_max,
                local_off: [0.0; 3],
            },
            [[0.0; 3]; 8],
        );
        for (i, rec) in records.iter().enumerate() {
            dir.note_cull_draw(i as u32, rec);
        }

        let mut parts = Vec::new();
        dir.partitions_into(&mut parts, MAX_FACE_RUNS, Some(eye));
        let visible = [0b111u32];
        let view = view(
            &camera,
            Some(&shadow_frusta),
            dir.live_end(),
            [0.0; 3],
            true,
        );
        let (new_cmds, new_counts, new_stats) = cpu_cull(&dir, |_| true, &visible, &parts, &view);
        let (old_cmds, old_counts, old_stats) =
            cpu_cull_legacy(&records, &dir, |_| true, &visible, &parts, &view);
        assert_same_emission(
            &parts,
            &old_cmds,
            &old_counts,
            old_stats,
            &new_cmds,
            &new_counts,
            new_stats,
        );
        assert_eq!(new_stats[Group::Caged as usize * 2], 4);
        assert_eq!(new_stats[Group::Caged as usize * 2 + 1], 18 + 6 + 30 + 36);

        let side_cmds = caged_camera_cmds(&parts, &new_cmds, &new_counts, 0);
        assert_eq!(side_cmds.len(), 2, "side box should emit two runs");
        assert_eq!(side_cmds[0].first_index, 0);
        assert_eq!(side_cmds[0].index_count, 18);
        assert_eq!(side_cmds[1].first_index, 24);
        assert_eq!(side_cmds[1].index_count, 6);

        let rot_cmds = caged_camera_cmds(&parts, &new_cmds, &new_counts, 1);
        assert_eq!(rot_cmds.len(), 1, "rotated cage should emit one run");
        assert_eq!(rot_cmds[0].first_index, 6);
        assert_eq!(rot_cmds[0].index_count, 30);
        assert_eq!(rot_cmds[0].vertex_offset, 11);

        let whole = caged_camera_cmds(&parts, &new_cmds, &new_counts, 2);
        assert_eq!(whole.len(), 1, "singular frame draws the mesh whole");
        assert_eq!(whole[0].first_index, 0);
        assert_eq!(whole[0].index_count, 36);
        assert_eq!(whole[0].vertex_offset, 13);

        for cascade in 0..2 {
            let idx = shadow_part(cascade, 0, 1);
            let sh = part_cmds(&parts, &new_cmds, &new_counts, idx);
            assert_eq!(sh.len(), 3, "cascade {cascade} keeps every caster whole");
            let mut seen = [false; 3];
            for cmd in sh {
                assert_eq!(cmd.first_index, 0);
                assert_eq!(cmd.index_count, 36);
                seen[cmd.first_instance as usize] = true;
            }
            assert_eq!(seen, [true; 3]);
        }
    }
}
