use super::arena::ArenaDirectory;
#[cfg(test)]
use super::buffers::MESH_FLAG_FACE_RUNS;
use super::buffers::{DrawIndexedIndirect, MeshRecord};
use super::pipeline::EyeSplit;
use crate::camera::Frustum;

/// Camera emission groups, in partition-table order. Mirrored by cull.comp.slang.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(usize)]
pub(crate) enum Group {
    /// Full-res (scale <= 1) Opaque: no-`discard` fragment module.
    Opaque = 0,
    Cutout = 1,
    /// Coarse-LOD (scale > 1) Opaque: the slab-clip `discard` module.
    OpaqueLod = 2,
}
/// Camera groups (Opaque, Cutout, OpaqueLod). Bucketed.
pub(crate) const CAMERA_GROUPS: usize = 3;
/// Shadow Near/Far. Unbucketed; sized from the full-res Opaque live count.
pub(crate) const SHADOW_GROUPS: usize = 2;
/// Camera groups + shadow groups.
pub(crate) const GROUPS: usize = CAMERA_GROUPS + SHADOW_GROUPS;
/// Front-to-back buckets on camera groups only (shadows stay unbucketed).
pub(crate) const BUCKETS: usize = crate::genconst::CULL_DISTANCE_BUCKETS as usize;
/// Inclusive lower edges of buckets 1..=3 (`cull.comp.slang` `distance_bucket`).
pub(crate) const CULL_BUCKET_SPLITS: [f32; 3] = [
    crate::genconst::CULL_BUCKET_SPLIT_0,
    crate::genconst::CULL_BUCKET_SPLIT_1,
    crate::genconst::CULL_BUCKET_SPLIT_2,
];
/// Max contiguous face-runs the GPU cull emits per camera mesh. Per axis the
/// camera is in {+, −, both}; upload order +X,+Y,+Z,−X,−Y,−Z keeps same-sign
/// faces adjacent, so an outside camera sees ≤3 maximal contiguous runs.
pub(crate) const MAX_FACE_RUNS: u32 = 3;
/// Live-count lanes: [full-res Opaque, Cutout, LOD Opaque].
pub(crate) const LANES: usize = CAMERA_GROUPS;
/// Size of VkDrawIndexedIndirectCommand.
pub(crate) const CMD_STRIDE: u64 = 20;
pub(crate) const WORKGROUP: u32 = crate::genconst::CULL_WORKGROUP;
/// Profiling-only geometry histogram: per camera group `[draws, index_count]`.
pub(crate) const STATS_COUNT: usize = CAMERA_GROUPS * 2;
pub(crate) const STATS_BYTES: u64 = (STATS_COUNT * size_of::<u32>()) as u64;
pub(crate) const FLAG_STATS: u32 = 1;
/// Live camera-group records at or below this count skip the GPU cull and
/// emit the same commands on the CPU. Overridable via `VOXEL_CPU_CULL_MAX`.
///
/// Forcing the CPU path was +5 % at 567 meshes on an RTX 3070 and +25 % on an
/// RTX 4060 with a Ryzen 5 5500; at 2025 meshes it was +8 % on the Ryzen box
/// and -3 % on the i5 box. 1024 keeps the win without paying on fast GPUs.
const CPU_CULL_MAX: u32 = 1024;

const _: () = assert!(crate::genconst::CULL_DISTANCE_BUCKETS == 4);
const _: () = assert!(crate::genconst::CULL_CAMERA_GROUPS == CAMERA_GROUPS as u32);
const _: () = assert!(GROUPS == CAMERA_GROUPS + SHADOW_GROUPS);
const _: () = assert!(Group::OpaqueLod as usize + 1 == CAMERA_GROUPS);

/// True when every fragment of a camera-relative AABB would be discarded by
/// the coarse-LOD slab clip (`mesh3d.frag.slang`). Both extents must be
/// active (`clip_v == 0` makes the fragment `inside_v` false). Horizontal
/// test uses the farthest xz corner (max |x|, max |z|) so a skip is exact
/// for the per-fragment `length(world.xz) < clip` test. Mirrored by
/// `lod_aabb_inside_slab` in `cull.comp.slang`.
fn lod_aabb_inside_slab(mn: [f32; 3], mx: [f32; 3], clip: f32, clip_v: f32) -> bool {
    if !(clip > 0.0 && clip_v > 0.0) {
        return false;
    }
    let far_x = mn[0].abs().max(mx[0].abs());
    let far_z = mn[2].abs().max(mx[2].abs());
    let inside_h = (far_x * far_x + far_z * far_z).sqrt() < clip;
    let inside_v = mn[1].abs().max(mx[1].abs()) < clip_v;
    inside_h && inside_v
}

/// Camera-distance bucket of an AABB centre, matching `cull.comp.slang`.
/// Splits are 16/64/256 (powers of two), so comparing `dist²` against `split²`
/// is exact and skips the `sqrt`.
#[cfg(test)]
#[inline(always)]
fn distance_bucket(dist: f32) -> u32 {
    distance_bucket_sq(dist * dist)
}

#[inline(always)]
fn distance_bucket_sq(d2: f32) -> u32 {
    const S0: f32 = crate::genconst::CULL_BUCKET_SPLIT_0;
    const S1: f32 = crate::genconst::CULL_BUCKET_SPLIT_1;
    const S2: f32 = crate::genconst::CULL_BUCKET_SPLIT_2;
    u32::from(d2 >= S0 * S0) + u32::from(d2 >= S1 * S1) + u32::from(d2 >= S2 * S2)
}

pub(crate) fn cpu_cull_max() -> u32 {
    std::env::var("VOXEL_CPU_CULL_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CPU_CULL_MAX)
}

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
    let offset = [
        (rec.block[0] - eye.block[0]) as f32 - eye.frac[0] + rec.local_off[0],
        (rec.block[1] - eye.block[1]) as f32 - eye.frac[1] + rec.local_off[1],
        (rec.block[2] - eye.block[2]) as f32 - eye.frac[2] + rec.local_off[2],
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
/// −axis iff `mx[axis] > 0`. +Y subtracts the planet-curvature droop.
fn face_vis(mn: [f32; 3], mx: [f32; 3]) -> [bool; 6] {
    let far_x = mn[0].abs().max(mx[0].abs());
    let far_z = mn[2].abs().max(mx[2].abs());
    let droop = ((far_x * far_x + far_z * far_z) * crate::genconst::CURVE_INV_2R)
        .min(crate::genconst::CURVE_MAX_DROP);
    [
        mn[0] < 0.0,
        (mn[1] - droop) < 0.0,
        mn[2] < 0.0,
        mx[0] > 0.0,
        mx[1] > 0.0,
        mx[2] > 0.0,
    ]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

#[cfg(test)]
fn emit_cmd(
    part: usize,
    cmd: DrawIndexedIndirect,
    partitions: &[PartitionGpu],
    cmds: &mut [DrawIndexedIndirect],
    counts: &mut [u32],
) {
    let i = counts[part];
    counts[part] += 1;
    if i < partitions[part].capacity {
        cmds[(partitions[part].offset + i) as usize] = cmd;
    }
}

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

#[inline(always)]
fn cam_relative_soa(
    aabb: [f32; 6],
    block: [i32; 3],
    eye_block: [i32; 3],
    eye_frac: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let dx = (block[0] - eye_block[0]) as f32 - eye_frac[0];
    let dy = (block[1] - eye_block[1]) as f32 - eye_frac[1];
    let dz = (block[2] - eye_block[2]) as f32 - eye_frac[2];
    (
        [aabb[0] + dx, aabb[1] + dy, aabb[2] + dz],
        [aabb[3] + dx, aabb[4] + dy, aabb[5] + dz],
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

/// Host re-implementation of `computeMain` in `cull.comp.slang`. Writes
/// `DrawCmd`s at partition offsets and per-partition counts into `scratch`
/// (capacity retained across frames).
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_cull_into(
    _records: &[MeshRecord],
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    camera: &Frustum,
    shadow: Option<&[Frustum; 2]>,
    eye: EyeSplit,
    slot_count: u32,
    clip: f32,
    clip_v: f32,
    face_cull: bool,
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

    let shadow_on = shadow.is_some();
    build_live_vis(
        dir.live_bits(),
        visible,
        slot_count,
        shadow_on,
        &mut scratch.live_vis,
    );

    let cam_planes = camera.planes().map(|p| p.to_array());
    let shadow_planes = shadow.map(|frusta| {
        [
            frusta[0].planes().map(|p| p.to_array()),
            frusta[1].planes().map(|p| p.to_array()),
        ]
    });

    match (face_cull, shadow_on) {
        (false, false) => cull_fast_solid(
            dir,
            is_arrived,
            partitions,
            &cam_planes,
            clip,
            clip_v,
            slot_count,
            &scratch.live_vis,
            &mut scratch.part_cmds,
            &mut scratch.counts,
            eye.block,
            eye.frac,
        ),
        (false, true) => cull_slots::<false, true>(
            dir,
            is_arrived,
            partitions,
            &cam_planes,
            shadow_planes.as_ref(),
            visible,
            eye,
            clip,
            clip_v,
            slot_count,
            &scratch.live_vis,
            &mut scratch.part_cmds,
            &mut scratch.counts,
        ),
        (true, false) => cull_slots::<true, false>(
            dir,
            is_arrived,
            partitions,
            &cam_planes,
            None,
            visible,
            eye,
            clip,
            clip_v,
            slot_count,
            &scratch.live_vis,
            &mut scratch.part_cmds,
            &mut scratch.counts,
        ),
        (true, true) => cull_slots::<true, true>(
            dir,
            is_arrived,
            partitions,
            &cam_planes,
            shadow_planes.as_ref(),
            visible,
            eye,
            clip,
            clip_v,
            slot_count,
            &scratch.live_vis,
            &mut scratch.part_cmds,
            &mut scratch.counts,
        ),
    }
}

/// Host re-implementation of `computeMain` in `cull.comp.slang`. Writes
/// `DrawCmd`s at partition offsets and per-partition counts.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_cull(
    records: &[MeshRecord],
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    camera: &Frustum,
    shadow: Option<&[Frustum; 2]>,
    eye: EyeSplit,
    slot_count: u32,
    clip: f32,
    clip_v: f32,
    face_cull: bool,
) -> (Vec<DrawIndexedIndirect>, Vec<u32>, [u32; STATS_COUNT]) {
    let mut scratch = CpuCullScratch::default();
    let stats = cpu_cull_into(
        records,
        dir,
        is_arrived,
        visible,
        partitions,
        camera,
        shadow,
        eye,
        slot_count,
        clip,
        clip_v,
        face_cull,
        &mut scratch,
    );
    (
        flatten_part_cmds(&scratch.part_cmds, partitions),
        scratch.counts,
        stats,
    )
}

/// Pre-SoA CPU cull: walks `0..slot_count`, loads the 80-byte [`MeshRecord`],
/// and uses the shader's `?:` p-vertex. Kept as the emission reference for
/// the old-vs-new test.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn cpu_cull_legacy(
    records: &[MeshRecord],
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    visible: &[u32],
    partitions: &[PartitionGpu],
    camera: &Frustum,
    shadow: Option<&[Frustum; 2]>,
    eye: EyeSplit,
    slot_count: u32,
    clip: f32,
    clip_v: f32,
    face_cull: bool,
) -> (Vec<DrawIndexedIndirect>, Vec<u32>, [u32; STATS_COUNT]) {
    let total: usize = partitions.iter().map(|p| p.capacity as usize).sum();
    let mut cmds = vec![bytemuck::Zeroable::zeroed(); total];
    let mut counts = vec![0u32; partitions.len()];
    let mut stats = [0u32; STATS_COUNT];
    let cam_planes = camera.planes().map(|p| p.to_array());
    let shadow_planes = shadow.map(|frusta| {
        [
            frusta[0].planes().map(|p| p.to_array()),
            frusta[1].planes().map(|p| p.to_array()),
        ]
    });
    let arena_count = dir.arena_count() as u32;

    for slot in 0..slot_count {
        if !is_arrived(slot) {
            continue;
        }
        let word = dir.arena_word(slot as usize);
        if word == 0 {
            continue;
        }
        let cam_visible = slot_visible(visible, slot);
        if !cam_visible && shadow.is_none() {
            continue;
        }
        let Some(rec) = records.get(slot as usize) else {
            continue;
        };
        let pass = (rec.detail_pass >> 4) & 3;
        if pass > 1 {
            continue;
        }
        let (mn, mx, scale) = cam_relative_aabb(rec, eye);
        let arena = word - 1;
        let cmd = DrawIndexedIndirect {
            index_count: rec.index_count,
            instance_count: 1,
            first_index: 0,
            vertex_offset: rec.vertex_offset,
            first_instance: slot,
        };

        if cam_visible && aabb_in_planes_select(&cam_planes, mn, mx) {
            let group = if pass == 0 && scale > 1.0 { 2 } else { pass };
            if group != 2 || !lod_aabb_inside_slab(mn, mx, clip, clip_v) {
                let cx = 0.5 * (mn[0] + mx[0]);
                let cy = 0.5 * (mn[1] + mx[1]);
                let cz = 0.5 * (mn[2] + mx[2]);
                let dist = (cx * cx + cy * cy + cz * cz).sqrt();
                let bucket = distance_bucket(dist);
                let part = ((group * arena_count + arena) * crate::genconst::CULL_DISTANCE_BUCKETS
                    + bucket) as usize;
                if !face_cull || (rec.flags & MESH_FLAG_FACE_RUNS) == 0 {
                    emit_cmd(part, cmd, partitions, &mut cmds, &mut counts);
                    stats[group as usize * 2] += 1;
                    stats[group as usize * 2 + 1] += cmd.index_count;
                } else {
                    let vis = face_vis(mn, mx);
                    let bounds = packed_face_bounds(rec.face_quads);
                    let mut runs = [FaceRun {
                        first_index: 0,
                        index_count: 0,
                    }; 3];
                    let n = merge_face_runs(vis, bounds, &mut runs);
                    for run in runs.iter().take(n) {
                        let mut drawn = cmd;
                        drawn.first_index = run.first_index;
                        drawn.index_count = run.index_count;
                        emit_cmd(part, drawn, partitions, &mut cmds, &mut counts);
                        stats[group as usize * 2] += 1;
                        stats[group as usize * 2 + 1] += drawn.index_count;
                    }
                }
            }
        }

        if pass == 0
            && scale <= 1.0
            && let Some(planes) = &shadow_planes
        {
            let shadow_base =
                CAMERA_GROUPS as u32 * arena_count * crate::genconst::CULL_DISTANCE_BUCKETS;
            for (c, plane) in planes.iter().enumerate() {
                if aabb_in_planes_select(plane, mn, mx) {
                    emit_cmd(
                        (shadow_base + c as u32 * arena_count + arena) as usize,
                        cmd,
                        partitions,
                        &mut cmds,
                        &mut counts,
                    );
                }
            }
        }
    }
    (cmds, counts, stats)
}

/// Solid (no face-runs, no shadows) hot path: no per-slot Option, no MeshRecord.
#[allow(clippy::too_many_arguments)]
fn cull_fast_solid(
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    partitions: &[PartitionGpu],
    cam_planes: &[[f32; 4]; 5],
    clip: f32,
    clip_v: f32,
    slot_count: u32,
    live_vis: &[u32],
    part_cmds: &mut [Vec<DrawIndexedIndirect>],
    counts: &mut [u32],
    eye_block: [i32; 3],
    eye_frac: [f32; 3],
) -> [u32; STATS_COUNT] {
    let mut stats = [0u32; STATS_COUNT];
    let aabbs = dir.cull_aabbs();
    let blocks = dir.cull_blocks();
    let bits_soa = dir.cull_bits();
    let index_counts = dir.cull_index_counts();
    let vertex_offsets = dir.cull_vertex_offsets();
    let arena_count = dir.arena_count() as u32;
    let buckets = crate::genconst::CULL_DISTANCE_BUCKETS;

    for slot in 0..slot_count {
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
        let block = unsafe { *blocks.get_unchecked(i) };
        let (mn, mx) = cam_relative_soa(aabb, block, eye_block, eye_frac);
        if !aabb_in_planes(cam_planes, mn, mx) {
            continue;
        }
        let pass = super::arena::cull_bits_pass(bits);
        let lod = super::arena::cull_bits_lod(bits);
        let group = if pass == 0 && lod { 2 } else { pass };
        if group == 2 && lod_aabb_inside_slab(mn, mx, clip, clip_v) {
            continue;
        }
        let cx = 0.5 * (mn[0] + mx[0]);
        let cy = 0.5 * (mn[1] + mx[1]);
        let cz = 0.5 * (mn[2] + mx[2]);
        let bucket = distance_bucket_sq(cx * cx + cy * cy + cz * cz);
        let arena = super::arena::cull_bits_arena(bits) - 1;
        let part = ((group * arena_count + arena) * buckets + bucket) as usize;
        let cmd = DrawIndexedIndirect {
            index_count: unsafe { *index_counts.get_unchecked(i) },
            instance_count: 1,
            first_index: 0,
            vertex_offset: unsafe { *vertex_offsets.get_unchecked(i) },
            first_instance: slot,
        };
        emit_part(part, cmd, partitions, part_cmds, counts);
        stats[group as usize * 2] += 1;
        stats[group as usize * 2 + 1] += cmd.index_count;
    }
    stats
}

#[allow(clippy::too_many_arguments)]
fn cull_slots<const FACE: bool, const SHADOW: bool>(
    dir: &ArenaDirectory,
    is_arrived: impl Fn(u32) -> bool,
    partitions: &[PartitionGpu],
    cam_planes: &[[f32; 4]; 5],
    shadow_planes: Option<&[[[f32; 4]; 5]; 2]>,
    visible: &[u32],
    eye: EyeSplit,
    clip: f32,
    clip_v: f32,
    slot_count: u32,
    live_vis: &[u32],
    part_cmds: &mut [Vec<DrawIndexedIndirect>],
    counts: &mut [u32],
) -> [u32; STATS_COUNT] {
    let mut stats = [0u32; STATS_COUNT];
    let aabbs = dir.cull_aabbs();
    let blocks = dir.cull_blocks();
    let bits_soa = dir.cull_bits();
    let index_counts = dir.cull_index_counts();
    let vertex_offsets = dir.cull_vertex_offsets();
    let face_quads = dir.cull_face_quads();
    let eye_block = eye.block;
    let eye_frac = eye.frac;
    let arena_count = dir.arena_count() as u32;
    let buckets = crate::genconst::CULL_DISTANCE_BUCKETS;
    let shadow_base = CAMERA_GROUPS as u32 * arena_count * buckets;

    for slot in 0..slot_count {
        let w = (slot >> 5) as usize;
        let bit = 1u32 << (slot & 31);
        if unsafe { live_vis.get_unchecked(w) } & bit == 0 {
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
        let block = unsafe { *blocks.get_unchecked(i) };
        let (mn, mx) = cam_relative_soa(aabb, block, eye_block, eye_frac);
        let pass = super::arena::cull_bits_pass(bits);
        let lod = super::arena::cull_bits_lod(bits);
        let arena = super::arena::cull_bits_arena(bits) - 1;
        let cam_visible = !SHADOW || slot_visible(visible, slot);

        let mut cam_ok = false;
        let mut group = 0u32;
        let mut part = 0usize;
        if cam_visible && aabb_in_planes(cam_planes, mn, mx) {
            group = if pass == 0 && lod { 2 } else { pass };
            if group != 2 || !lod_aabb_inside_slab(mn, mx, clip, clip_v) {
                let cx = 0.5 * (mn[0] + mx[0]);
                let cy = 0.5 * (mn[1] + mx[1]);
                let cz = 0.5 * (mn[2] + mx[2]);
                let bucket = distance_bucket_sq(cx * cx + cy * cy + cz * cz);
                part = ((group * arena_count + arena) * buckets + bucket) as usize;
                cam_ok = true;
            }
        }
        let mut sh0 = false;
        let mut sh1 = false;
        if SHADOW
            && pass == 0
            && !lod
            && let Some(planes) = shadow_planes
        {
            sh0 = aabb_in_planes(&planes[0], mn, mx);
            sh1 = aabb_in_planes(&planes[1], mn, mx);
        }
        if !cam_ok && !sh0 && !sh1 {
            continue;
        }

        let cmd = DrawIndexedIndirect {
            index_count: unsafe { *index_counts.get_unchecked(i) },
            instance_count: 1,
            first_index: 0,
            vertex_offset: unsafe { *vertex_offsets.get_unchecked(i) },
            first_instance: slot,
        };

        if cam_ok {
            if FACE && super::arena::cull_bits_face(bits) {
                let vis = face_vis(mn, mx);
                let bounds = packed_face_bounds(unsafe { *face_quads.get_unchecked(i) });
                let mut runs = [FaceRun {
                    first_index: 0,
                    index_count: 0,
                }; 3];
                let n_runs = merge_face_runs(vis, bounds, &mut runs);
                for run in runs.iter().take(n_runs) {
                    let mut drawn = cmd;
                    drawn.first_index = run.first_index;
                    drawn.index_count = run.index_count;
                    emit_part(part, drawn, partitions, part_cmds, counts);
                    stats[group as usize * 2] += 1;
                    stats[group as usize * 2 + 1] += drawn.index_count;
                }
            } else {
                emit_part(part, cmd, partitions, part_cmds, counts);
                stats[group as usize * 2] += 1;
                stats[group as usize * 2 + 1] += cmd.index_count;
            }
        }

        if sh0 {
            emit_part(
                (shadow_base + arena) as usize,
                cmd,
                partitions,
                part_cmds,
                counts,
            );
        }
        if sh1 {
            emit_part(
                (shadow_base + arena_count + arena) as usize,
                cmd,
                partitions,
                part_cmds,
                counts,
            );
        }
    }
    stats
}

/// Partition index for a camera (pass, arena, bucket) triple.
pub(crate) fn camera_part(group: usize, arena: usize, bucket: usize, arena_count: usize) -> usize {
    debug_assert!(group < CAMERA_GROUPS);
    debug_assert!(bucket < BUCKETS);
    (group * arena_count + arena) * BUCKETS + bucket
}

/// Partition index for a shadow (cascade, arena) pair. Cascades are unbucketed.
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
/// into. Matches [`super::scene_pass`] `record_group_indirect_count`.
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
    pub(crate) const ALL: [Group; CAMERA_GROUPS] = [Group::Opaque, Group::Cutout, Group::OpaqueLod];
}

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

    const FULL: bool = false;
    const LOD: bool = true;

    #[test]
    fn group_order_is_the_partition_table_order() {
        for (i, g) in Group::ALL.iter().enumerate() {
            assert_eq!(*g as usize, i);
        }
        assert_eq!(CAMERA_GROUPS, Group::ALL.len());
    }

    #[test]
    fn lod_aabb_inside_slab_matches_fragment_discard() {
        // Fully inside: farthest xz = (3, 4) length 5 < 10; |y| = 2 < 8.
        assert!(lod_aabb_inside_slab(
            [-3.0, -2.0, -4.0],
            [1.0, 2.0, 2.0],
            10.0,
            8.0
        ));

        // Straddling the circle: origin is inside, farthest (8, 8) length ~11.3 > 10.
        assert!(!lod_aabb_inside_slab(
            [-1.0, -1.0, -1.0],
            [8.0, 1.0, 8.0],
            10.0,
            8.0
        ));
        // On the circle (6-8-10) is not strictly inside; skip only if farthest is inside.
        assert!(!lod_aabb_inside_slab(
            [0.0, -1.0, 0.0],
            [6.0, 1.0, 8.0],
            10.0,
            8.0
        ));

        // Inside horizontally (length 5 < 10) but |y| = 9 is outside clip_v = 8.
        assert!(!lod_aabb_inside_slab(
            [-3.0, -9.0, -4.0],
            [3.0, 1.0, 4.0],
            10.0,
            8.0
        ));

        // clip == 0 disables the fragment discard, so never skip.
        assert!(!lod_aabb_inside_slab(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            0.0,
            8.0
        ));
        // clip_v == 0 likewise (inside_v is then false in the fragment shader).
        assert!(!lod_aabb_inside_slab(
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            10.0,
            0.0
        ));
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
            _pad: 0,
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
        clip: f32,
        clip_v: f32,
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
        let (cmds, counts, stats) = cpu_cull(
            records,
            dir,
            |_| true,
            visible,
            &parts,
            camera,
            None,
            eye,
            dir.live_end(),
            clip,
            clip_v,
            face_cull,
        );
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
            run_cpu(&mut dir, &[front], &[1], &camera, false, 0.0, 0.0);
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
        let (_, _, counts, stats) = run_cpu(&mut dir, &[behind], &[1], &camera, false, 0.0, 0.0);
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
        let (parts, cmds, counts, _) = run_cpu(&mut dir, &[rec], &[1], &camera, false, 0.0, 0.0);
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
        let (_, _, counts, _) = run_cpu(&mut dir, &[rec], &[0], &camera, false, 0.0, 0.0);
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
        let (parts, cmds, counts, stats) = run_cpu(&mut dir, &[rec], &[1], &camera, true, 0.0, 0.0);
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
        let (parts, cmds, counts, _) = run_cpu(&mut dir, &[rec], &[1], &camera, true, 0.0, 0.0);
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
        let bucket_naive = |dist: f32| {
            CULL_BUCKET_SPLITS
                .iter()
                .filter(|&&s| dist >= s)
                .count()
                .min(BUCKETS - 1) as u32
        };
        for dist in [0.0, 15.99, 16.0, 63.99, 64.0, 255.99, 256.0, 1.0e6] {
            assert_eq!(distance_bucket(dist), bucket_naive(dist), "dist={dist}");
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
    fn cpu_cull_skips_lod_meshes_fully_inside_the_slab() {
        let camera = look_neg_z();
        let mut rec = opaque_rec([-1.0, -1.0, -11.0], [1.0, 1.0, -9.0]);
        rec.detail_pass = u32::from(crate::mesh::Detail(1).to_gpu_bits());
        let (mn, mx, scale) = cam_relative_aabb(&rec, origin_eye());
        assert!(scale > 1.0);
        assert!(lod_aabb_inside_slab(mn, mx, 100.0, 100.0));
        let mut dir = ArenaDirectory::new();
        dir.note_upload(
            0,
            G1,
            buf(1),
            Pass::Opaque,
            LOD,
            MeshAabb::from_record(&rec),
        );
        let (_, _, counts, stats) = run_cpu(&mut dir, &[rec], &[1], &camera, false, 100.0, 100.0);
        assert!(counts.iter().all(|&c| c == 0));
        assert_eq!(stats, [0; STATS_COUNT]);
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
        rec.detail_pass = u32::from(detail.to_gpu_bits()) | ((pass as u32) << 4);
        rec
    }

    /// Mixed synthetic slot table: holes, Blend, LOD, face-runs, Cutout,
    /// two arenas, some hidden, some not-yet-arrived.
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
            let rec = match slot % 7 {
                0 => opaque_rec([mn[0], mn[1], 9.0], [mx[0], mx[1], 11.0]), // behind
                1 => rec_pass(mn, mx, Pass::Cutout, false),
                2 => rec_pass(mn, mx, Pass::Opaque, true),
                3 => face_rec(mn, mx),
                4 => rec_pass(mn, mx, Pass::Blend, false),
                5 if slot % 5 == 0 => continue, // hole
                _ => opaque_rec(mn, mx),
            };
            records[i] = rec;
            let lod = rec.detail_scale() > 1.0;
            let buf_id = 1 + u64::from(slot % 2);
            dir.note_upload(
                slot,
                G1,
                buf(buf_id),
                rec.pass(),
                lod,
                MeshAabb::from_record(&rec),
            );
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
                for (clip, clip_v) in [(0.0, 0.0), (40.0, 40.0)] {
                    let mut parts = Vec::new();
                    let runs = if face_cull { MAX_FACE_RUNS } else { 1 };
                    dir.partitions_into(&mut parts, runs, Some(eye));
                    let shadow = with_shadow.then_some(&shadow_frusta);
                    let (old_cmds, old_counts, old_stats) = cpu_cull_legacy(
                        &records,
                        &dir,
                        is_arrived,
                        &visible,
                        &parts,
                        &camera,
                        shadow,
                        eye,
                        dir.live_end(),
                        clip,
                        clip_v,
                        face_cull,
                    );
                    let (new_cmds, new_counts, new_stats) = cpu_cull(
                        &records,
                        &dir,
                        is_arrived,
                        &visible,
                        &parts,
                        &camera,
                        shadow,
                        eye,
                        dir.live_end(),
                        clip,
                        clip_v,
                        face_cull,
                    );
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
        let slot_count = dir.live_end();
        let mut scratch = CpuCullScratch::default();

        fn time_ns(iters: u32, mut body: impl FnMut()) -> f64 {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                body();
            }
            start.elapsed().as_nanos() as f64 / f64::from(iters)
        }

        for _ in 0..WARMUP {
            let _ = cpu_cull_legacy(
                &records,
                &dir,
                |_| true,
                &visible,
                &parts,
                &camera,
                None,
                eye,
                slot_count,
                0.0,
                0.0,
                false,
            );
            let _ = cpu_cull_into(
                &records,
                &dir,
                |_| true,
                &visible,
                &parts,
                &camera,
                None,
                eye,
                slot_count,
                0.0,
                0.0,
                false,
                &mut scratch,
            );
        }

        // Include the host-visible fill: old path memcpy'd the whole sparse
        // command buffer (including unused capacity); new copies each live
        // partition range only.
        let total: usize = parts.iter().map(|p| p.capacity as usize).sum();
        let mut wc_cmds = vec![0u8; total * CMD_STRIDE as usize];
        let mut wc_counts = vec![0u8; parts.len() * 4];

        let old_ns = time_ns(ITERS, || {
            let (cmds, counts, stats) = cpu_cull_legacy(
                &records,
                &dir,
                |_| true,
                &visible,
                &parts,
                &camera,
                None,
                eye,
                slot_count,
                0.0,
                0.0,
                false,
            );
            let cmd_bytes: &[u8] = bytemuck::cast_slice(&cmds);
            wc_cmds[..cmd_bytes.len()].copy_from_slice(cmd_bytes);
            let count_bytes: &[u8] = bytemuck::cast_slice(&counts);
            wc_counts[..count_bytes.len()].copy_from_slice(count_bytes);
            std::hint::black_box((&wc_cmds, &wc_counts, stats));
        });
        let new_ns = time_ns(ITERS, || {
            let stats = cpu_cull_into(
                &records,
                &dir,
                |_| true,
                &visible,
                &parts,
                &camera,
                None,
                eye,
                slot_count,
                0.0,
                0.0,
                false,
                &mut scratch,
            );
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
}
