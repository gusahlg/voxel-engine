//! GPU draw-command emission: cull compute shader + indirect-count.
//! One dispatch per frame frustum-tests each mesh and appends commands
//! per-(camera-group, arena, distance-bucket) and per-(cascade, arena) for
//! shadows. When occlusion is on, camera draws are also tested against the
//! previous frame's Hi-Z pyramid (shadow emission is unaffected). Blend uses
//! CPU path; immediates untouched. A small live camera-group count skips the
//! dispatch and writes the same commands on the host.
//!
//! Camera groups (bucketed): full-res Opaque, Cutout, coarse-LOD Opaque
//! (`scale > 1`). The LOD split exists so full-res opaque draws bind a
//! fragment module with no `discard` (early depth write) while only the LOD
//! partition pays for the slab clip. Coarse-LOD meshes whose camera-relative
//! AABB lies entirely inside that slab are not emitted (every fragment would
//! be discarded). Shadow Near/Far stay unbucketed and reuse the full-res
//! Opaque live count.

use std::num::NonZeroU32;

use ash::vk;

use glam::{Mat4, Vec3};

use super::alloc::{find_memory_type, try_find_memory_type};
use super::buffers::{
    DrawIndexedIndirect, FRAMES_IN_FLIGHT, HostBuffer, MESH_FLAG_FACE_RUNS, MeshRecord,
    RecordBuffers,
};
use super::pass;
use super::pipeline::EyeSplit;
use crate::camera::Frustum;
use crate::mesh::Pass;

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;
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
const CULL_BUCKET_SPLITS: [f32; 3] = [
    crate::genconst::CULL_BUCKET_SPLIT_0,
    crate::genconst::CULL_BUCKET_SPLIT_1,
    crate::genconst::CULL_BUCKET_SPLIT_2,
];
/// Max contiguous face-runs the GPU cull emits per camera mesh. Per axis the
/// camera is in {+, −, both}; upload order +X,+Y,+Z,−X,−Y,−Z keeps same-sign
/// faces adjacent, so an outside camera sees ≤3 maximal contiguous runs.
pub(crate) const MAX_FACE_RUNS: u32 = 3;
/// Live-count lanes: [full-res Opaque, Cutout, LOD Opaque].
const LANES: usize = CAMERA_GROUPS;
/// Size of VkDrawIndexedIndirectCommand.
pub(crate) const CMD_STRIDE: u64 = 20;
const WORKGROUP: u32 = crate::genconst::CULL_WORKGROUP;
/// Profiling-only geometry histogram: per camera group `[draws, index_count]`,
/// then a trailing occluded-mesh count.
const STATS_COUNT: usize = CAMERA_GROUPS * 2 + 1;
const STATS_OCC: usize = CAMERA_GROUPS * 2;
const _: () = assert!(STATS_OCC + 1 == STATS_COUNT);
const STATS_BYTES: u64 = (STATS_COUNT * size_of::<u32>()) as u64;
const FLAG_STATS: u32 = 1;
/// Live camera-group records at or below this count skip the GPU cull and
/// emit the same commands on the CPU. Overridable via `VOXEL_CPU_CULL_MAX`.
///
/// Forcing the CPU path was +5 % at 567 meshes on an RTX 3070 and +25 % on an
/// RTX 4060 with a Ryzen 5 5500; at 2025 meshes it was +8 % on the Ryzen box
/// and -3 % on the i5 box. 1024 keeps the win without paying on fast GPUs.
const CPU_CULL_MAX: u32 = 1024;

static CULL_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull.comp.spv"));
static CULL_COMP_WAVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/cull_wave.comp.spv"));

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
/// Favour drawing: cull only when the mesh's nearest reversed-Z is strictly
/// farther than the farthest occluder in the rect by this margin. Twin of
/// `CULL_OCC_EPS` in `cull.comp.slang`.
const OCC_EPS: f32 = crate::genconst::CULL_OCC_EPS;

/// Mip where the UV rect spans at most 2×2 texels of the pyramid. `ceil` so
/// a split goes coarser (safer: a larger footprint's MIN is farther). Twin of
/// `occ_mip_for_rect` in `cull.comp.slang`.
pub(crate) fn occ_mip_for_rect(span_uv: [f32; 2], level0: [u32; 2], mip_count: u32) -> u32 {
    let span_tex = (span_uv[0] * level0[0] as f32).max(span_uv[1] * level0[1] as f32);
    let mip_f = (span_tex * 0.5).max(1.0).log2().ceil();
    (mip_f as u32).min(mip_count.saturating_sub(1))
}

/// Reversed-Z compare: hide iff `nearest_z` is farther than `occ_min` by
/// [`OCC_EPS`]. Twin of `occ_hidden_z` in `cull.comp.slang`.
pub(crate) fn occ_hidden(nearest_z: f32, occ_min: f32) -> bool {
    nearest_z < occ_min - OCC_EPS
}

/// Project 8 camera-relative AABB corners with `view_proj`. `None` (keep the
/// mesh) if any corner has w ≤ 0 or the clamped UV rect is degenerate.
/// Otherwise `(uv_min, uv_max, nearest_z)` with uv in [0,1] and nearest_z the
/// max of z/w (reversed-Z nearer).
pub(crate) fn occ_screen_rect(
    mn: [f32; 3],
    mx: [f32; 3],
    view_proj: &Mat4,
) -> Option<([f32; 2], [f32; 2], f32)> {
    let mut nearest_z = 0.0f32;
    let mut uv_min = [1.0f32, 1.0];
    let mut uv_max = [0.0f32, 0.0];
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                let p = Vec3::new(
                    if x != 0 { mx[0] } else { mn[0] },
                    if y != 0 { mx[1] } else { mn[1] },
                    if z != 0 { mx[2] } else { mn[2] },
                );
                let clip = *view_proj * p.extend(1.0);
                if clip.w <= 0.0 {
                    return None;
                }
                let ndc = clip.truncate() / clip.w;
                let uv = [ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5];
                uv_min[0] = uv_min[0].min(uv[0]);
                uv_min[1] = uv_min[1].min(uv[1]);
                uv_max[0] = uv_max[0].max(uv[0]);
                uv_max[1] = uv_max[1].max(uv[1]);
                nearest_z = nearest_z.max(ndc.z);
            }
        }
    }
    uv_min[0] = uv_min[0].clamp(0.0, 1.0);
    uv_min[1] = uv_min[1].clamp(0.0, 1.0);
    uv_max[0] = uv_max[0].clamp(0.0, 1.0);
    uv_max[1] = uv_max[1].clamp(0.0, 1.0);
    if uv_max[0] - uv_min[0] <= 0.0 || uv_max[1] - uv_min[1] <= 0.0 {
        return None;
    }
    Some((uv_min, uv_max, nearest_z))
}

/// Inclusive texel pair covering `uv` at `mip`, plus the MIN of those four
/// samples (the farthest occluder). Host twin of the shader gather.
pub(crate) fn occ_gather_min(
    uv_min: [f32; 2],
    uv_max: [f32; 2],
    level0: [u32; 2],
    mip: u32,
    sample: impl Fn(i32, i32) -> f32,
) -> f32 {
    let mip_size = [
        ((level0[0] as f32) / 2f32.powi(mip as i32)).max(1.0),
        ((level0[1] as f32) / 2f32.powi(mip as i32)).max(1.0),
    ];
    let mut i0 = [
        (uv_min[0] * mip_size[0]).floor() as i32,
        (uv_min[1] * mip_size[1]).floor() as i32,
    ];
    let mut i1 = [
        (uv_max[0] * mip_size[0]).ceil() as i32 - 1,
        (uv_max[1] * mip_size[1]).ceil() as i32 - 1,
    ];
    i1[0] = i1[0].max(i0[0]);
    i1[1] = i1[1].max(i0[1]);
    let dim = [mip_size[0] as i32, mip_size[1] as i32];
    i0[0] = i0[0].clamp(0, dim[0] - 1);
    i0[1] = i0[1].clamp(0, dim[1] - 1);
    i1[0] = i1[0].clamp(0, dim[0] - 1);
    i1[1] = i1[1].clamp(0, dim[1] - 1);
    sample(i0[0], i0[1])
        .min(sample(i1[0], i0[1]))
        .min(sample(i0[0], i1[1]))
        .min(sample(i1[0], i1[1]))
}

fn distance_bucket(dist: f32) -> u32 {
    let mut b = 0u32;
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_0 {
        b = 1;
    }
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_1 {
        b = 2;
    }
    if dist >= crate::genconst::CULL_BUCKET_SPLIT_2 {
        b = 3;
    }
    b.min(crate::genconst::CULL_DISTANCE_BUCKETS - 1)
}

fn cpu_cull_max() -> u32 {
    std::env::var("VOXEL_CPU_CULL_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CPU_CULL_MAX)
}

/// P-vertex test matching `outside_plane` in `cull.comp.slang` and
/// [`Frustum::intersects_aabb`].
fn outside_plane(plane: [f32; 4], mn: [f32; 3], mx: [f32; 3]) -> bool {
    let corner = [
        if plane[0] >= 0.0 { mx[0] } else { mn[0] },
        if plane[1] >= 0.0 { mx[1] } else { mn[1] },
        if plane[2] >= 0.0 { mx[2] } else { mn[2] },
    ];
    plane[0] * corner[0] + plane[1] * corner[1] + plane[2] * corner[2] + plane[3] < 0.0
}

fn aabb_in_planes(planes: &[[f32; 4]], mn: [f32; 3], mx: [f32; 3]) -> bool {
    planes.iter().all(|p| !outside_plane(*p, mn, mx))
}

fn slot_visible(visible: &[u32], slot: u32) -> bool {
    visible
        .get((slot >> 5) as usize)
        .is_some_and(|w| w & (1 << (slot & 31)) != 0)
}

/// Camera-relative AABB and decoded scale, matching the shader's
/// `offset / mn / mx / scale` reconstruction.
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

/// Host re-implementation of `computeMain` in `cull.comp.slang`. Writes
/// `DrawCmd`s at partition offsets and per-partition counts.
fn cpu_cull(
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

        if cam_visible && aabb_in_planes(&cam_planes, mn, mx) {
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

        if pass == 0 && scale <= 1.0 {
            if let Some(planes) = &shadow_planes {
                let shadow_base =
                    CAMERA_GROUPS as u32 * arena_count * crate::genconst::CULL_DISTANCE_BUCKETS;
                for c in 0..2 {
                    if aabb_in_planes(&planes[c], mn, mx) {
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
    }
    (cmds, counts, stats)
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

/// GPU CullParams struct.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullParamsGpu {
    cam_planes: [[f32; 4]; 5],
    shadow_planes: [[f32; 4]; 10],
    cam_block: [i32; 3],
    slot_count: u32,
    cam_frac: [f32; 3],
    arena_count: u32,
    shadow_enabled: u32,
    flags: u32,
    clip: f32,
    clip_v: f32,
    prev_view_proj: [[f32; 4]; 4],
    hiz_mips: u32,
    hiz_w: u32,
    hiz_h: u32,
    occ_flags: u32,
}
// std140: prev_view_proj is 16-aligned after clip_v (288); occ tail is 16 bytes.
const _: () = assert!(size_of::<CullParamsGpu>() == 368);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_planes) == 0);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_planes) == 80);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_block) == 240);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, slot_count) == 252);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, cam_frac) == 256);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, arena_count) == 268);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, shadow_enabled) == 272);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, flags) == 276);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, clip) == 280);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, clip_v) == 284);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, prev_view_proj) == 288);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, hiz_mips) == 352);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, hiz_w) == 356);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, hiz_h) == 360);
const _: () = assert!(std::mem::offset_of!(CullParamsGpu, occ_flags) == 364);

/// Occlusion inputs for one GPU cull dispatch. `enabled` is false on the
/// first frame after create/resize/invalidation (pyramid is cleared to 0,
/// which never culls) and when the occlusion flag is off.
#[derive(Clone, Copy)]
pub(crate) struct OccParams {
    pub view_proj: Mat4,
    pub mips: u32,
    pub level0: vk::Extent2D,
    pub enabled: bool,
}

impl OccParams {
    pub fn disabled(level0: vk::Extent2D, mips: u32) -> Self {
        Self {
            view_proj: Mat4::IDENTITY,
            mips: mips.max(1),
            level0,
            enabled: false,
        }
    }
}

/// Pyramid view+sampler bound at cull set 0 binding 8. Always a valid
/// `SHADER_READ_ONLY` image (cleared to 0 when history is empty).
#[derive(Clone, Copy)]
pub(crate) struct HizSample {
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}

/// Mesh AABB in world/block space: `aabb_min/max * scale + local_off`, relative
/// to `block`. The GPU cull buckets by AABB-centre distance; a centre always
/// lies inside its box, so a union of these boxes bounds every possible centre.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MeshAabb {
    pub block: [i32; 3],
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl MeshAabb {
    const ZERO: Self = Self {
        block: [0; 3],
        min: [0.0; 3],
        max: [0.0; 3],
    };

    pub(crate) fn from_record(rec: &MeshRecord) -> Self {
        let s = rec.detail_scale();
        Self {
            block: rec.block,
            min: [
                rec.aabb_min[0] * s + rec.local_off[0],
                rec.aabb_min[1] * s + rec.local_off[1],
                rec.aabb_min[2] * s + rec.local_off[2],
            ],
            max: [
                rec.aabb_max[0] * s + rec.local_off[0],
                rec.aabb_max[1] * s + rec.local_off[1],
                rec.aabb_max[2] * s + rec.local_off[2],
            ],
        }
    }
}

/// Conservative union of live camera-group AABBs for one arena, stored relative
/// to integer `origin`. `dirty` means a free/drain removed a contributor and
/// the union must be rebuilt from still-live slots before it is queried.
#[derive(Clone, Copy)]
struct ArenaUnion {
    origin: [i32; 3],
    min: [f32; 3],
    max: [f32; 3],
    dirty: bool,
    valid: bool,
}

impl ArenaUnion {
    const EMPTY: Self = Self {
        origin: [0; 3],
        min: [0.0; 3],
        max: [0.0; 3],
        dirty: false,
        valid: false,
    };

    fn from_aabb(aabb: MeshAabb) -> Self {
        Self {
            origin: aabb.block,
            min: aabb.min,
            max: aabb.max,
            dirty: false,
            valid: true,
        }
    }

    fn expand(&mut self, aabb: MeshAabb) {
        if !self.valid {
            *self = Self::from_aabb(aabb);
            return;
        }
        let d = [
            (aabb.block[0] - self.origin[0]) as f32,
            (aabb.block[1] - self.origin[1]) as f32,
            (aabb.block[2] - self.origin[2]) as f32,
        ];
        for i in 0..3 {
            self.min[i] = self.min[i].min(aabb.min[i] + d[i]);
            self.max[i] = self.max[i].max(aabb.max[i] + d[i]);
        }
    }

    /// Camera-distance range of every point of the union box (hence every
    /// possible AABB centre). Matches the shader's `(block - cam_block) - frac`
    /// placement so a single-mesh union agrees with `distance_bucket`.
    fn cam_dist_range(&self, eye: EyeSplit) -> (f32, f32) {
        let off = [
            (self.origin[0] - eye.block[0]) as f32 - eye.frac[0],
            (self.origin[1] - eye.block[1]) as f32 - eye.frac[1],
            (self.origin[2] - eye.block[2]) as f32 - eye.frac[2],
        ];
        aabb_center_dist_range(
            [
                self.min[0] + off[0],
                self.min[1] + off[1],
                self.min[2] + off[2],
            ],
            [
                self.max[0] + off[0],
                self.max[1] + off[1],
                self.max[2] + off[2],
            ],
        )
    }
}

/// Nearest and farthest origin-distance of any point in the AABB. Used as
/// centre-distance bounds: the union box *is* the set of possible centres.
fn aabb_center_dist_range(mn: [f32; 3], mx: [f32; 3]) -> (f32, f32) {
    let mn = Vec3::from_array(mn);
    let mx = Vec3::from_array(mx);
    let closest = Vec3::ZERO.clamp(mn, mx);
    let farthest = Vec3::new(
        if mn.x.abs() > mx.x.abs() { mn.x } else { mx.x },
        if mn.y.abs() > mx.y.abs() { mn.y } else { mx.y },
        if mn.z.abs() > mx.z.abs() { mn.z } else { mx.z },
    );
    (closest.length(), farthest.length())
}

/// True when a centre at some distance in `[dmin, dmax]` can land in `bucket`.
/// Bucket `b` is `[split_{b-1}, split_b)` with `split_{-1} = 0` and the last
/// bucket unbounded on the right (`dist >= split` assigns the higher bucket).
fn bucket_intersects(bucket: usize, dmin: f32, dmax: f32) -> bool {
    debug_assert!(bucket < BUCKETS);
    let lo = if bucket == 0 {
        0.0
    } else {
        CULL_BUCKET_SPLITS[bucket - 1]
    };
    match CULL_BUCKET_SPLITS.get(bucket) {
        Some(&hi) => dmax >= lo && dmin < hi,
        None => dmax >= lo,
    }
}

/// Arena registry with live counts per (arena, lane).
pub(crate) struct ArenaDirectory {
    buffers: Vec<vk::Buffer>,
    /// Live [Opaque, Cutout, OpaqueLod] counts per arena (shadow reuses Opaque).
    live: Vec<[u32; LANES]>,
    /// Reference count per arena; zero = reusable.
    refs: Vec<u32>,
    /// Slot placement (arena, lane, gen) for free decrement / re-laning.
    slots: Vec<Option<(u32, Option<usize>, NonZeroU32)>>,
    /// Per-slot world AABB, parallel to `slots` (only valid when the slot is live).
    aabbs: Vec<MeshAabb>,
    /// Conservative union of live camera-group AABBs per arena.
    unions: Vec<ArenaUnion>,
    /// The registered Blend slots (unordered). The CPU Blend re-source walks
    /// exactly this set, so its cost scales with the transparent meshes, not
    /// with the whole slot table.
    blend: Vec<u32>,
    /// Position + 1 of a slot in `blend` (0 = not a Blend slot); O(1) removal.
    blend_pos: Vec<u32>,
    /// One past the highest registered slot: the cull dispatch and its
    /// visibility mask stop here rather than at the table's high-water mark
    /// when the tail has been freed.
    live_end: u32,
}

impl ArenaDirectory {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            live: Vec::new(),
            refs: Vec::new(),
            slots: Vec::new(),
            aabbs: Vec::new(),
            unions: Vec::new(),
            blend: Vec::new(),
            blend_pos: Vec::new(),
            live_end: 0,
        }
    }

    /// Registers an upload: interns the arena block (reusing a drained row),
    /// bumps its counts, and returns the arena index for the slot's word.
    /// `lod` is the record's `scale > 1` (the cull shader's LOD test).
    /// `aabb` grows the arena's conservative union (camera-group slots only).
    pub fn note_upload(
        &mut self,
        slot: u32,
        generation: NonZeroU32,
        buffer: vk::Buffer,
        pass: Pass,
        lod: bool,
        aabb: MeshAabb,
    ) -> u32 {
        if let Some(Some((old_arena, _, _))) = self.slots.get(slot as usize).copied() {
            // Re-register without a free: drop the old box out of the union.
            self.mark_union_dirty(old_arena as usize);
        }
        let hit = (0..self.buffers.len())
            .find(|&i| self.refs[i] > 0 && self.buffers[i] == buffer)
            .or_else(|| {
                let reuse = self.refs.iter().position(|&r| r == 0);
                if let Some(i) = reuse {
                    self.buffers[i] = buffer;
                    debug_assert_eq!(self.live[i], [0; LANES], "drained row kept live counts");
                    self.unions[i] = ArenaUnion::EMPTY;
                }
                reuse
            });
        let arena = match hit {
            Some(i) => i as u32,
            None => {
                self.buffers.push(buffer);
                self.live.push([0; LANES]);
                self.refs.push(0);
                self.unions.push(ArenaUnion::EMPTY);
                (self.buffers.len() - 1) as u32
            }
        };
        self.refs[arena as usize] += 1;
        let lane = group_lane(pass, lod);
        if let Some(lane) = lane {
            self.live[arena as usize][lane] += 1;
        }
        let n = slot as usize + 1;
        if self.slots.len() < n {
            self.slots.resize(n, None);
            self.aabbs.resize(n, MeshAabb::ZERO);
        }
        self.slots[slot as usize] = Some((arena, lane, generation));
        self.aabbs[slot as usize] = aabb;
        if lane.is_some() {
            self.grow_union(arena as usize, aabb);
        }
        self.set_blend(slot, pass == Pass::Blend);
        self.live_end = self.live_end.max(slot + 1);
        arena
    }

    /// Adds `slot` to (or removes it from) the Blend set; idempotent either way.
    fn set_blend(&mut self, slot: u32, on: bool) {
        let i = slot as usize;
        if self.blend_pos.len() <= i {
            self.blend_pos.resize(i + 1, 0);
        }
        let pos = self.blend_pos[i];
        if on && pos == 0 {
            self.blend.push(slot);
            self.blend_pos[i] = self.blend.len() as u32;
        } else if !on && pos != 0 {
            let at = (pos - 1) as usize;
            self.blend.swap_remove(at);
            self.blend_pos[i] = 0;
            if let Some(&moved) = self.blend.get(at) {
                self.blend_pos[moved as usize] = pos;
            }
        }
    }

    /// The registered Blend slots, in no particular order.
    pub fn blend_slots(&self) -> &[u32] {
        &self.blend
    }

    /// One past the highest registered slot (0 when nothing is registered):
    /// every slot at or beyond it has a dead arena word.
    pub fn live_end(&self) -> u32 {
        self.live_end
    }

    /// Re-lanes a resident slot whose record was recomposed (a mover's
    /// placement patch may change its detail): moves its live count so the
    /// partition capacities keep matching what the cull shader emits.
    /// Grows the arena union with the new AABB (stale-large until a free).
    pub fn note_record(&mut self, slot: u32, pass: Pass, lod: bool, aabb: MeshAabb) {
        let Some(Some((arena, lane, _))) = self.slots.get(slot as usize).copied() else {
            return;
        };
        let new_lane = group_lane(pass, lod);
        if lane != new_lane {
            if let Some(old) = lane {
                self.live[arena as usize][old] -= 1;
            }
            if let Some(new) = new_lane {
                self.live[arena as usize][new] += 1;
            }
            if let Some(Some((_, slot_lane, _))) = self.slots.get_mut(slot as usize) {
                *slot_lane = new_lane;
            }
        }
        self.aabbs[slot as usize] = aabb;
        if new_lane.is_some() {
            self.grow_union(arena as usize, aabb);
        }
        self.set_blend(slot, pass == Pass::Blend);
    }

    /// Register a free with generation check.
    pub fn note_free(&mut self, slot: u32, generation: NonZeroU32) {
        let Some(Some((arena, lane, stored_gen))) =
            self.slots.get_mut(slot as usize).map(Option::take)
        else {
            return;
        };
        if stored_gen != generation {
            // Stale free for a newer generation; restore slot.
            self.slots[slot as usize] = Some((arena, lane, stored_gen));
            return;
        }
        self.refs[arena as usize] -= 1;
        if let Some(lane) = lane {
            self.live[arena as usize][lane] -= 1;
        }
        if self.refs[arena as usize] == 0 {
            self.unions[arena as usize] = ArenaUnion::EMPTY;
        } else {
            self.mark_union_dirty(arena as usize);
        }
        self.set_blend(slot, false);
        if slot + 1 == self.live_end {
            // The tail died: retreat to the next registered slot. Amortised
            // O(1) — each dead slot is stepped over once per retreat.
            self.live_end = self.slots[..slot as usize]
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |i| i as u32 + 1);
        }
    }

    pub fn arena_buffer(&self, arena: usize) -> vk::Buffer {
        self.buffers[arena]
    }

    /// Get arena word for cull shader (0 = dead, else arena+1).
    pub fn arena_word(&self, slot: usize) -> u32 {
        self.slots
            .get(slot)
            .copied()
            .flatten()
            .map_or(0, |(a, _, _)| a + 1)
    }

    pub fn arena_count(&self) -> usize {
        self.buffers.len()
    }

    /// Live camera-group records (Opaque + Cutout + OpaqueLod) across every
    /// arena. Blend is excluded. Used to choose the CPU cull path.
    fn camera_live(&self) -> u32 {
        self.live.iter().map(|l| l.iter().sum::<u32>()).sum()
    }

    fn mark_union_dirty(&mut self, arena: usize) {
        if let Some(u) = self.unions.get_mut(arena) {
            u.dirty = true;
        }
    }

    fn grow_union(&mut self, arena: usize, aabb: MeshAabb) {
        let u = &mut self.unions[arena];
        if u.dirty {
            return;
        }
        u.expand(aabb);
    }

    fn recompute_union(&mut self, arena: usize) {
        let mut acc = ArenaUnion::EMPTY;
        for (i, slot) in self.slots.iter().enumerate() {
            let Some((a, lane, _)) = *slot else {
                continue;
            };
            if a as usize != arena || lane.is_none() {
                continue;
            }
            acc.expand(self.aabbs[i]);
        }
        acc.dirty = false;
        self.unions[arena] = acc;
    }

    fn bucket_reachable(&self, arena: usize, bucket: usize, eye: EyeSplit) -> bool {
        let u = &self.unions[arena];
        if !u.valid {
            // No tracked union: keep the bucket (never drop a reachable mesh).
            return true;
        }
        let (dmin, dmax) = u.cam_dist_range(eye);
        bucket_intersects(bucket, dmin, dmax)
    }

    /// Fills `parts` with the group-major partition table, reusing its
    /// allocation, and returns the total command count.
    ///
    /// Camera groups (Opaque, Cutout, OpaqueLod) emit K distance buckets per
    /// arena, each sized to `live * runs_per_mesh` (worst case: every mesh lands
    /// in one bucket and emits that many face-runs) unless `eye` is set and the
    /// arena union AABB cannot reach that bucket — then capacity is 0 so the
    /// draw loop skips the call. Shadow groups (Near, Far) stay ×1 (whole-mesh
    /// cmd) and reuse the full-res Opaque live count — a caster may land in
    /// both cascades.
    fn partitions_into(
        &mut self,
        parts: &mut Vec<PartitionGpu>,
        runs_per_mesh: u32,
        eye: Option<EyeSplit>,
    ) -> u32 {
        let a = self.live.len();
        for arena in 0..a {
            if self.unions[arena].dirty {
                self.recompute_union(arena);
            }
        }
        parts.clear();
        parts.reserve(partition_count(a));
        let mut offset = 0u32;
        for group in Group::ALL {
            let lane = group as usize;
            for arena in 0..a {
                let full = self.live[arena][lane] * runs_per_mesh;
                for bucket in 0..BUCKETS {
                    let capacity = match eye {
                        _ if full == 0 => 0,
                        None => full,
                        Some(e) if self.bucket_reachable(arena, bucket, e) => full,
                        Some(_) => 0,
                    };
                    parts.push(PartitionGpu { offset, capacity });
                    offset += capacity;
                }
            }
        }
        for _cascade in 0..SHADOW_GROUPS {
            for arena in 0..a {
                let capacity = self.live[arena][0];
                parts.push(PartitionGpu { offset, capacity });
                offset += capacity;
            }
        }
        offset
    }

    /// Get group-major partition table (group, arena pairs). No eye: every
    /// live lane keeps all K buckets (used by tests that do not exercise
    /// distance-bucket zeroing).
    #[cfg(test)]
    fn partitions(&mut self) -> (Vec<PartitionGpu>, u32) {
        let mut parts = Vec::new();
        let total = self.partitions_into(&mut parts, 1, None);
        (parts, total)
    }
}

impl Group {
    /// Camera-group partition-table order (shadows are unbucketed after this).
    pub(crate) const ALL: [Group; CAMERA_GROUPS] = [Group::Opaque, Group::Cutout, Group::OpaqueLod];
}

/// Get live-count lane for a (pass, lod) record (Blend returns None).
fn group_lane(pass: Pass, lod: bool) -> Option<usize> {
    match pass {
        Pass::Opaque if lod => Some(2),
        Pass::Opaque => Some(0),
        Pass::Cutout => Some(1),
        Pass::Blend => None,
    }
}

/// Device-local grow-only buffer for GPU scratch.
struct DeviceBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: u64,
    usage: vk::BufferUsageFlags,
}

impl DeviceBuffer {
    fn new(usage: vk::BufferUsageFlags) -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 0,
            usage,
        }
    }

    fn bound(&self) -> Option<vk::Buffer> {
        (self.buffer != vk::Buffer::null()).then_some(self.buffer)
    }

    /// Grow to at least `needed` bytes. Safe after fence is waited.
    unsafe fn ensure(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) {
        if needed <= self.capacity {
            return;
        }
        unsafe {
            self.destroy(device);
            let capacity = needed.next_power_of_two().max(4096);
            let buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(capacity)
                        .usage(self.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("create cull buffer");
            let reqs = device.get_buffer_memory_requirements(buffer);
            let memory_props = instance.get_physical_device_memory_properties(physical);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(find_memory_type(
                            &memory_props,
                            reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )),
                    None,
                )
                .expect("allocate cull buffer memory");
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind cull buffer memory");
            self.buffer = buffer;
            self.memory = memory;
            self.capacity = capacity;
        }
    }

    unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
            self.buffer = vk::Buffer::null();
            self.memory = vk::DeviceMemory::null();
            self.capacity = 0;
        }
    }
}

/// Per-frame cull result (commands, counts, partition table).
pub(crate) struct CullFrame {
    pub commands: vk::Buffer,
    pub counts: vk::Buffer,
    pub partitions: Vec<PartitionGpu>,
    pub arena_count: usize,
    pub slot_count: u32,
    /// Host-written commands/counts; [`CullState::record`] is a no-op.
    pub cpu: bool,
}

pub(crate) struct CullState {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    params: [HostBuffer; SLOTS],
    parts: [HostBuffer; SLOTS],
    visible: [HostBuffer; SLOTS],
    commands: [DeviceBuffer; SLOTS],
    counts: [DeviceBuffer; SLOTS],
    cpu_cmds: [HostBuffer; SLOTS],
    cpu_counts: [HostBuffer; SLOTS],
    stats: [StatsReadback; SLOTS],
    /// Recycled partition table when [`Self::prepare`] returns `None`, so a
    /// frame with nothing to cull does not drop last frame's allocation.
    spare_parts: Vec<PartitionGpu>,
    /// Per-direction face-run culling. On by default; follows
    /// [`crate::Engine::set_cull_faces`].
    face_cull: bool,
}

/// Host-visible copy of the per-slot geometry histogram, fence-safe to read
/// after the slot's timeline wait. Same mechanism as the VRS mix buffer.
struct StatsReadback {
    gpu: vk::Buffer,
    gpu_memory: vk::DeviceMemory,
    cpu: vk::Buffer,
    cpu_memory: vk::DeviceMemory,
    mapped: *mut u32,
}

impl CullState {
    pub fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        cache: vk::PipelineCache,
        wave_atomics: bool,
    ) -> Self {
        // Bindings match cull.comp.slang.
        let storage = |binding: u32| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        };
        let bindings = [
            storage(0),
            storage(1),
            storage(2),
            storage(3),
            storage(4),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            storage(6),
            storage(7),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .expect("create cull set layout")
        };
        let set_layouts = [set_layout];
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(4)];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .expect("create cull pipeline layout")
        };
        let bytes = if wave_atomics {
            CULL_COMP_WAVE
        } else {
            CULL_COMP
        };
        let pipeline = pass::compute_pipeline(
            device,
            cache,
            layout,
            bytes,
            if wave_atomics { "cull-wave" } else { "cull" },
        );
        Self {
            set_layout,
            layout,
            pipeline,
            params: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::UNIFORM_BUFFER)),
            parts: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER)),
            visible: std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER)),
            commands: std::array::from_fn(|_| {
                DeviceBuffer::new(
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                )
            }),
            counts: std::array::from_fn(|_| {
                DeviceBuffer::new(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::INDIRECT_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
            }),
            cpu_cmds: std::array::from_fn(|_| {
                HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER)
            }),
            cpu_counts: std::array::from_fn(|_| {
                HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER)
            }),
            stats: std::array::from_fn(|_| StatsReadback::new(device, memory_props)),
            spare_parts: Vec::new(),
            face_cull: true,
        }
    }

    /// Applied between frames; [`Self::prepare`] snapshots the value so a
    /// toggle cannot size partitions for one run count and advertise the other.
    pub fn set_face_cull(&mut self, on: bool) {
        self.face_cull = on;
    }

    /// Last completed histogram for `slot`: `[draws0, idx0, draws1, idx1, draws2, idx2, occ]`.
    pub fn stats(&self, slot: usize) -> [u32; STATS_COUNT] {
        unsafe {
            let p = self.stats[slot].mapped;
            std::array::from_fn(|i| *p.add(i))
        }
    }

    /// CPU-zero the mapped histogram so a skipped cull publishes zeros.
    pub fn clear_stats_cpu(&self, slot: usize) {
        unsafe { std::ptr::write_bytes(self.stats[slot].mapped, 0, STATS_COUNT) };
    }

    /// Prepare buffers and params for cull dispatch. Safe after fence is waited.
    ///
    /// `slot_count` bounds the dispatch (the caller trims it to the directory's
    /// live end); `visible` must cover it. `partitions` is last frame's table
    /// handed back for reuse (its contents are discarded). `clip` / `clip_v`
    /// are the full-res slab extents (`DrawLists::lod_clip`, `lod_clip_v`);
    /// 0 disables, matching the mesh3d push constants.
    ///
    /// When the directory's camera-group live count is at most `CPU_CULL_MAX`
    /// (or `VOXEL_CPU_CULL_MAX`), commands and counts are written to host-visible
    /// buffers here and no compute work is recorded.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prepare(
        &mut self,
        slot: usize,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        dir: &mut ArenaDirectory,
        records: RecordBuffers,
        host_records: &[MeshRecord],
        is_arrived: impl Fn(u32) -> bool,
        slot_count: u32,
        camera: &Frustum,
        shadow: Option<&[Frustum; 2]>,
        eye: super::pipeline::EyeSplit,
        clip: f32,
        clip_v: f32,
        visible: &[u32],
        mut partitions: Vec<PartitionGpu>,
        occ: OccParams,
    ) -> Option<CullFrame> {
        debug_assert!(slot_count <= records.slots, "slot_count exceeds the table");
        debug_assert!(visible.len() >= slot_count.div_ceil(32) as usize);
        if partitions.capacity() == 0 {
            partitions = std::mem::take(&mut self.spare_parts);
        }
        // One snapshot for both partition capacity and CullParams.flags so a
        // mid-frame toggle cannot size runs for one value and advertise the other.
        let face_cull = self.face_cull;
        let runs_per_mesh = if face_cull { MAX_FACE_RUNS } else { 1 };
        let total = dir.partitions_into(&mut partitions, runs_per_mesh, Some(eye));
        if partitions.is_empty() || total == 0 {
            self.spare_parts = partitions;
            return None;
        }
        if dir.camera_live() <= cpu_cull_max() {
            let (cmds, counts, stats_hist) = cpu_cull(
                host_records,
                dir,
                is_arrived,
                visible,
                &partitions,
                camera,
                shadow,
                eye,
                slot_count,
                clip,
                clip_v,
                face_cull,
            );
            unsafe {
                let cb = &mut self.cpu_cmds[slot];
                cb.maintain(instance, device, physical, u64::from(total) * CMD_STRIDE);
                cb.write(0, bytemuck::cast_slice(&cmds));
                let nb = &mut self.cpu_counts[slot];
                nb.maintain(instance, device, physical, (partitions.len() * 4) as u64);
                nb.write(0, bytemuck::cast_slice(&counts));
                std::ptr::copy_nonoverlapping(
                    stats_hist.as_ptr(),
                    self.stats[slot].mapped,
                    STATS_COUNT,
                );
            }
            return Some(CullFrame {
                commands: self.cpu_cmds[slot].bound()?,
                counts: self.cpu_counts[slot].bound()?,
                partitions,
                arena_count: dir.arena_count(),
                slot_count,
                cpu: true,
            });
        }
        let mut params = CullParamsGpu {
            cam_planes: camera.planes().map(|p| p.to_array()),
            shadow_planes: [[0.0; 4]; 10],
            cam_block: eye.block,
            slot_count,
            cam_frac: eye.frac,
            arena_count: dir.arena_count() as u32,
            shadow_enabled: shadow.is_some() as u32,
            flags: u32::from(face_cull),
            clip,
            clip_v,
            prev_view_proj: occ.view_proj.to_cols_array_2d(),
            hiz_mips: occ.mips.max(1),
            hiz_w: occ.level0.width.max(1),
            hiz_h: occ.level0.height.max(1),
            occ_flags: u32::from(occ.enabled),
        };
        if let Some(frusta) = shadow {
            for (c, f) in frusta.iter().enumerate() {
                for (p, plane) in f.planes().iter().enumerate() {
                    params.shadow_planes[c * 5 + p] = plane.to_array();
                }
            }
        }
        let part_bytes: &[u8] = bytemuck::cast_slice(&partitions);
        unsafe {
            let pb = &mut self.params[slot];
            pb.maintain(
                instance,
                device,
                physical,
                size_of::<CullParamsGpu>() as u64,
            );
            pb.write(0, bytemuck::bytes_of(&params));
            let tb = &mut self.parts[slot];
            tb.maintain(instance, device, physical, part_bytes.len() as u64);
            tb.write(0, part_bytes);
            let vis_bytes: &[u8] = bytemuck::cast_slice(visible);
            let vb = &mut self.visible[slot];
            vb.maintain(instance, device, physical, vis_bytes.len() as u64);
            vb.write(0, vis_bytes);
            self.commands[slot].ensure(instance, device, physical, u64::from(total) * CMD_STRIDE);
            self.counts[slot].ensure(instance, device, physical, (partitions.len() * 4) as u64);
        }
        Some(CullFrame {
            commands: self.commands[slot].bound()?,
            counts: self.counts[slot].bound()?,
            partitions,
            arena_count: dir.arena_count(),
            slot_count,
            cpu: false,
        })
    }

    /// Record cull dispatch (zeros counts, executes cull, fences writes).
    /// Geometry stats fill/atomics/copy run only while profiling.
    /// A CPU-culled frame records no compute work: commands and counts were
    /// written to host-coherent memory in [`Self::prepare`] after the slot wait.
    /// Returns whether this recorded GPU commands (fill/dispatch/barrier).
    pub unsafe fn record(
        &self,
        device: &ash::Device,
        push: &ash::khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        records: RecordBuffers,
        frame: &CullFrame,
        hiz: HizSample,
    ) -> bool {
        if frame.cpu {
            return false;
        }
        let stats = crate::profile::is_enabled();
        unsafe {
            device.cmd_fill_buffer(cmd, frame.counts, 0, vk::WHOLE_SIZE, 0);
            if stats {
                device.cmd_fill_buffer(cmd, self.stats[slot].gpu, 0, STATS_BYTES, 0);
            }
            // CLEAR / TRANSFER_WRITE → COMPUTE / SHADER_STORAGE_{READ,WRITE}
            // covers the counts fill and, when profiling, the stats fill.
            let to_compute = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(
                    vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
                )];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_compute),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            let info = |buffer: vk::Buffer| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            };
            // Binding order matches cull.comp.slang.
            let infos = [
                info(records.records),
                info(records.arenas),
                info(
                    self.parts[slot]
                        .bound()
                        .expect("partitions were just written"),
                ),
                info(frame.commands),
                info(frame.counts),
                info(self.params[slot].bound().expect("params were just written")),
                info(
                    self.visible[slot]
                        .bound()
                        .expect("visibility was just written"),
                ),
                info(self.stats[slot].gpu),
            ];
            let hiz_info = [vk::DescriptorImageInfo::default()
                .sampler(hiz.sampler)
                .image_view(hiz.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let mut writes: [vk::WriteDescriptorSet; 9] = std::array::from_fn(|i| {
                vk::WriteDescriptorSet::default()
                    .dst_binding(i as u32)
                    .descriptor_type(if i == 5 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(std::slice::from_ref(&infos[i.min(7)]))
            });
            writes[8] = vk::WriteDescriptorSet::default()
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&hiz_info);
            push.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &writes,
            );
            let flags = if stats { FLAG_STATS } else { 0 };
            device.cmd_push_constants(
                cmd,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&flags),
            );
            device.cmd_dispatch(cmd, frame.slot_count.div_ceil(WORKGROUP), 1, 1);

            // COMPUTE / SHADER_STORAGE_WRITE → DRAW_INDIRECT / INDIRECT_COMMAND_READ,
            // and when profiling also COPY / TRANSFER_READ for the stats copy.
            let mut dst_stage = vk::PipelineStageFlags2::DRAW_INDIRECT;
            let mut dst_access = vk::AccessFlags2::INDIRECT_COMMAND_READ;
            if stats {
                dst_stage |= vk::PipelineStageFlags2::COPY;
                dst_access |= vk::AccessFlags2::TRANSFER_READ;
            }
            let to_draws = [vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(dst_stage)
                .dst_access_mask(dst_access)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_draws),
            );
            if stats {
                device.cmd_copy_buffer(
                    cmd,
                    self.stats[slot].gpu,
                    self.stats[slot].cpu,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: STATS_BYTES,
                    }],
                );
                // COPY / TRANSFER_WRITE → HOST / HOST_READ. Mapped read is after
                // this slot's timeline wait (one cycle later).
                let copy_to_host = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&copy_to_host),
                );
            }
        }
        true
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            for b in self
                .params
                .iter_mut()
                .chain(&mut self.parts)
                .chain(&mut self.visible)
                .chain(&mut self.cpu_cmds)
                .chain(&mut self.cpu_counts)
            {
                b.destroy(device);
            }
            for b in self.commands.iter_mut().chain(&mut self.counts) {
                b.destroy(device);
            }
            for s in &self.stats {
                s.destroy(device);
            }
        }
    }
}

impl StatsReadback {
    fn new(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties) -> Self {
        let gpu = create_buffer(
            device,
            memory_props,
            STATS_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let (cpu, cpu_memory, mapped) = create_mapped_buffer(
            device,
            memory_props,
            STATS_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        Self {
            gpu: gpu.0,
            gpu_memory: gpu.1,
            cpu,
            cpu_memory,
            mapped,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.cpu_memory);
            device.destroy_buffer(self.cpu, None);
            device.free_memory(self.cpu_memory, None);
            device.destroy_buffer(self.gpu, None);
            device.free_memory(self.gpu_memory, None);
        }
    }
}

fn create_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create cull stats buffer")
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(find_memory_type(
                        memory_props,
                        reqs.memory_type_bits,
                        props,
                    )),
                None,
            )
            .expect("allocate cull stats buffer")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind cull stats buffer");
    }
    (buffer, memory)
}

fn create_mapped_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory, *mut u32) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create cull stats readback")
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let plain = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = try_find_memory_type(memory_props, reqs.memory_type_bits, cached)
        .unwrap_or_else(|| find_memory_type(memory_props, reqs.memory_type_bits, plain));
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index),
                None,
            )
            .expect("allocate cull stats readback")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind cull stats readback");
    }
    let mapped = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("map cull stats readback") as *mut u32
    };
    unsafe { std::ptr::write_bytes(mapped, 0, STATS_COUNT) };
    (buffer, memory, mapped)
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn buf(raw: u64) -> vk::Buffer {
        vk::Buffer::from_raw(raw)
    }

    fn genr(v: u32) -> NonZeroU32 {
        NonZeroU32::new(v).unwrap()
    }
    const G1: NonZeroU32 = NonZeroU32::new(1).unwrap();
    const UNIT: MeshAabb = MeshAabb {
        block: [0; 3],
        min: [0.0; 3],
        max: [1.0; 3],
    };

    fn origin_eye() -> EyeSplit {
        EyeSplit {
            block: [0; 3],
            _pad0: 0,
            frac: [0.0; 3],
            _pad1: 0.0,
        }
    }

    #[test]
    fn empty_directory_has_no_partitions() {
        let mut dir = ArenaDirectory::new();
        let (parts, total) = dir.partitions();
        assert!(parts.is_empty());
        assert_eq!(total, 0);
    }

    const FULL: bool = false;
    const LOD: bool = true;

    #[test]
    fn single_arena_single_opaque_upload_produces_exact_partition() {
        let mut dir = ArenaDirectory::new();
        let arena = dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        assert_eq!(arena, 0);
        assert_eq!(dir.arena_count(), 1);
        let (parts, total) = dir.partitions();
        // 3 camera groups * K buckets + 2 shadow groups, one arena.
        assert_eq!(parts.len(), partition_count(1));
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(Group::Opaque as usize, 0, bucket, 1)],
                PartitionGpu {
                    offset: bucket as u32,
                    capacity: 1
                }
            );
            assert_eq!(
                parts[camera_part(Group::Cutout as usize, 0, bucket, 1)].capacity,
                0,
                "cutout buckets stay empty"
            );
            assert_eq!(
                parts[camera_part(Group::OpaqueLod as usize, 0, bucket, 1)].capacity,
                0,
                "lod buckets stay empty"
            );
        }
        // Shadow Near/Far reuse full-res Opaque's live count, unbucketed.
        // Command offsets skip empty Cutout/LOD buckets (capacity 0).
        assert_eq!(
            parts[shadow_part(0, 0, 1)],
            PartitionGpu {
                offset: BUCKETS as u32,
                capacity: 1
            }
        );
        assert_eq!(
            parts[shadow_part(1, 0, 1)],
            PartitionGpu {
                offset: BUCKETS as u32 + 1,
                capacity: 1
            }
        );
        // K opaque camera slots + 2 shadow slots.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn lod_opaque_uploads_take_their_own_group_and_never_cast() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, LOD, UNIT);
        dir.note_upload(1, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(2, G1, buf(1), Pass::Opaque, LOD, UNIT);
        let (parts, total) = dir.partitions();
        assert_eq!(
            parts[camera_part(Group::Opaque as usize, 0, 0, 1)].capacity,
            1
        );
        assert_eq!(
            parts[camera_part(Group::Cutout as usize, 0, 0, 1)].capacity,
            0
        );
        // Shadow casters are the full-res set only (cull.comp: scale <= 1).
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
        assert_eq!(
            parts[camera_part(Group::OpaqueLod as usize, 0, 0, 1)].capacity,
            2
        );
        // Opaque K + LOD 2K + 2 shadow.
        assert_eq!(total, (BUCKETS + BUCKETS * 2 + 2) as u32);
    }

    #[test]
    fn blend_uploads_do_not_occupy_a_cull_lane() {
        // Blend never reaches the GPU cull (CPU-sorted path); its records still
        // register a reference (for reuse bookkeeping) but no live count.
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Blend, FULL, UNIT);
        let (parts, total) = dir.partitions();
        assert!(parts.iter().all(|p| p.capacity == 0));
        assert_eq!(total, 0);
    }

    #[test]
    fn partitions_are_group_major_offsets_accumulate_across_arenas() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT); // arena 0: 1 opaque
        dir.note_upload(1, G1, buf(1), Pass::Opaque, FULL, UNIT); // arena 0: 2 opaque (same buffer)
        dir.note_upload(2, G1, buf(2), Pass::Cutout, FULL, UNIT); // arena 1: 1 cutout
        dir.note_upload(3, G1, buf(2), Pass::Opaque, LOD, UNIT); // arena 1: 1 LOD opaque
        assert_eq!(dir.arena_count(), 2);
        let (parts, total) = dir.partitions();
        assert_eq!(parts.len(), partition_count(2));
        // Opaque a0: K buckets, each capacity 2, offsets 0,2,4,6.
        for bucket in 0..BUCKETS {
            let p = parts[camera_part(Group::Opaque as usize, 0, bucket, 2)];
            assert_eq!(p.capacity, 2);
            assert_eq!(p.offset, (bucket * 2) as u32);
        }
        // Opaque a1: empty.
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(Group::Opaque as usize, 1, bucket, 2)].capacity,
                0
            );
        }
        // Cutout a0 empty, a1 capacity 1 across K buckets.
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(Group::Cutout as usize, 0, bucket, 2)].capacity,
                0
            );
            assert_eq!(
                parts[camera_part(Group::Cutout as usize, 1, bucket, 2)].capacity,
                1
            );
        }
        // LOD a0 empty, a1 capacity 1.
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(Group::OpaqueLod as usize, 0, bucket, 2)].capacity,
                0
            );
            assert_eq!(
                parts[camera_part(Group::OpaqueLod as usize, 1, bucket, 2)].capacity,
                1
            );
        }
        // Shadows unbucketed, reuse full-res Opaque live counts (a0=2, a1=0).
        assert_eq!(parts[shadow_part(0, 0, 2)].capacity, 2);
        assert_eq!(parts[shadow_part(0, 1, 2)].capacity, 0);
        assert_eq!(parts[shadow_part(1, 0, 2)].capacity, 2);
        assert_eq!(parts[shadow_part(1, 1, 2)].capacity, 0);
        // Opaque: K*2, Cutout: K*1, LOD: K*1, ShadowNear: 2, ShadowFar: 2.
        assert_eq!(total, (BUCKETS * 2 + BUCKETS + BUCKETS + 2 + 2) as u32);
    }

    #[test]
    fn note_free_decrements_live_count_and_capacity_shrinks() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_free(0, G1);
        let (parts, total) = dir.partitions();
        assert_eq!(parts[camera_part(0, 0, 0, 1)].capacity, 1);
        // K camera slots + 2 shadow slots, each sized off the remaining live count.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn note_free_on_last_reference_drains_the_arena_row_for_reuse() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_free(0, G1);
        assert_eq!(dir.arena_count(), 1); // row kept, but refs == 0 now
        // A fresh upload reuses the drained row instead of growing the table.
        let arena = dir.note_upload(1, G1, buf(2), Pass::Cutout, FULL, UNIT);
        assert_eq!(arena, 0, "drained row should be reused, not appended");
        assert_eq!(dir.arena_count(), 1);
    }

    #[test]
    fn note_upload_matches_a_still_live_buffer_instead_of_reusing_a_drained_row() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        // A second upload to the SAME live buffer must hit the existing row, not
        // mint a new one (this is how one arena block accrues multiple meshes).
        let arena = dir.note_upload(1, G1, buf(1), Pass::Cutout, FULL, UNIT);
        assert_eq!(arena, 0);
        assert_eq!(dir.arena_count(), 1);
        let (parts, _) = dir.partitions();
        assert_eq!(parts[camera_part(0, 0, 0, 1)].capacity, 1); // Opaque
        assert_eq!(parts[camera_part(1, 0, 0, 1)].capacity, 1); // Cutout
    }

    #[test]
    fn note_free_with_stale_generation_is_a_no_op() {
        // A slot freed then immediately re-uploaded (new generation) must not
        // have a late/duplicate free for the OLD generation decrement its
        // still-live count out from under it.
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_free(0, genr(1)); // real free: drains the slot
        dir.note_upload(0, genr(2), buf(2), Pass::Cutout, FULL, UNIT); // reused, new generation
        dir.note_free(0, genr(1)); // stale duplicate: must be ignored
        let (parts, total) = dir.partitions();
        assert_eq!(
            parts[camera_part(1, 0, 0, 1)].capacity,
            1,
            "Cutout slot must still be live"
        );
        assert_eq!(
            total, BUCKETS as u32,
            "stale free must not have drained the reused slot"
        );
    }

    #[test]
    fn blend_set_tracks_uploads_and_frees_only_for_blend_slots() {
        let mut dir = ArenaDirectory::new();
        assert!(dir.blend_slots().is_empty());
        dir.note_upload(3, G1, buf(1), Pass::Blend, FULL, UNIT);
        dir.note_upload(7, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(9, G1, buf(2), Pass::Blend, FULL, UNIT);
        dir.note_upload(12, G1, buf(2), Pass::Blend, FULL, UNIT);
        let mut got = dir.blend_slots().to_vec();
        got.sort_unstable();
        assert_eq!(got, [3, 9, 12]);
        // Removing from the middle (swap_remove) keeps the moved slot findable.
        dir.note_free(9, G1);
        let mut got = dir.blend_slots().to_vec();
        got.sort_unstable();
        assert_eq!(got, [3, 12]);
        dir.note_free(12, G1);
        assert_eq!(dir.blend_slots(), [3]);
        // A stale free never touches the set; a real one drains it.
        dir.note_free(3, genr(2));
        assert_eq!(dir.blend_slots(), [3]);
        dir.note_free(3, G1);
        assert!(dir.blend_slots().is_empty());
        // Re-registering a drained slot as Blend re-adds it exactly once.
        dir.note_upload(3, genr(2), buf(1), Pass::Blend, FULL, UNIT);
        dir.note_upload(3, genr(2), buf(1), Pass::Blend, FULL, UNIT);
        assert_eq!(dir.blend_slots(), [3]);
        // Re-registering it as Opaque (without a free in between) removes it.
        dir.note_upload(3, genr(3), buf(1), Pass::Opaque, FULL, UNIT);
        assert!(dir.blend_slots().is_empty());
    }

    #[test]
    fn partitions_into_reuses_the_callers_allocation() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        let mut parts = Vec::with_capacity(64);
        let ptr = parts.as_ptr();
        let total = dir.partitions_into(&mut parts, 1, None);
        assert_eq!(parts.as_ptr(), ptr);
        assert_eq!(parts.len(), partition_count(1));
        // K camera (opaque) slots + 2 unbucketed shadow slots.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn partitions_scale_camera_capacity_by_runs_per_mesh() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        let mut parts = Vec::new();
        let total = dir.partitions_into(&mut parts, 3, None);
        for bucket in 0..BUCKETS {
            assert_eq!(
                parts[camera_part(Group::Opaque as usize, 0, bucket, 1)].capacity,
                3
            );
        }
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
        assert_eq!(parts[shadow_part(1, 0, 1)].capacity, 1);
        assert_eq!(total, BUCKETS as u32 * 3 + 2);
    }

    #[test]
    fn live_end_follows_the_highest_registered_slot() {
        let mut dir = ArenaDirectory::new();
        assert_eq!(dir.live_end(), 0);
        dir.note_upload(4, G1, buf(1), Pass::Opaque, FULL, UNIT);
        assert_eq!(dir.live_end(), 5);
        dir.note_upload(40, G1, buf(1), Pass::Blend, FULL, UNIT);
        dir.note_upload(20, G1, buf(1), Pass::Cutout, FULL, UNIT);
        assert_eq!(dir.live_end(), 41);
        // Freeing below the top leaves it; freeing the top retreats past the
        // dead gap to the next registered slot.
        dir.note_free(20, G1);
        assert_eq!(dir.live_end(), 41);
        dir.note_free(40, G1);
        assert_eq!(dir.live_end(), 5);
        // A stale free of the top is ignored.
        dir.note_free(4, genr(2));
        assert_eq!(dir.live_end(), 5);
        dir.note_free(4, G1);
        assert_eq!(dir.live_end(), 0);
    }

    #[test]
    fn note_record_moves_a_recomposed_slot_between_lanes() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        // A mover recomposed at a coarser detail migrates to the LOD lane...
        dir.note_record(0, Pass::Opaque, LOD, UNIT);
        let (parts, _) = dir.partitions();
        assert_eq!(
            parts[camera_part(Group::Opaque as usize, 0, 0, 1)].capacity,
            0
        );
        assert_eq!(
            parts[camera_part(Group::OpaqueLod as usize, 0, 0, 1)].capacity,
            1
        );
        // ...and back; an unchanged lane is a no-op; a free still balances.
        dir.note_record(0, Pass::Opaque, FULL, UNIT);
        dir.note_record(0, Pass::Opaque, FULL, UNIT);
        let (parts, _) = dir.partitions();
        assert_eq!(
            parts[camera_part(Group::Opaque as usize, 0, 0, 1)].capacity,
            1
        );
        assert_eq!(
            parts[camera_part(Group::OpaqueLod as usize, 0, 0, 1)].capacity,
            0
        );
        dir.note_free(0, G1);
        let (parts, total) = dir.partitions();
        assert!(parts.iter().all(|p| p.capacity == 0));
        assert_eq!(total, 0);
        // A non-resident slot is ignored.
        dir.note_record(7, Pass::Opaque, LOD, UNIT);
        assert_eq!(dir.partitions().1, 0);
    }

    #[test]
    fn group_lane_maps_camera_passes_and_excludes_blend() {
        assert_eq!(group_lane(Pass::Opaque, FULL), Some(0));
        assert_eq!(group_lane(Pass::Opaque, LOD), Some(2));
        assert_eq!(group_lane(Pass::Cutout, FULL), Some(1));
        assert_eq!(group_lane(Pass::Cutout, LOD), Some(1));
        assert_eq!(group_lane(Pass::Blend, FULL), None);
        assert_eq!(group_lane(Pass::Blend, LOD), None);
    }

    #[test]
    fn group_order_is_the_partition_table_order() {
        for (i, g) in Group::ALL.iter().enumerate() {
            assert_eq!(*g as usize, i);
        }
        assert_eq!(CAMERA_GROUPS, Group::ALL.len());
    }

    #[test]
    fn stats_histogram_is_two_u32s_per_camera_group() {
        assert_eq!(STATS_COUNT, 7);
        assert_eq!(STATS_OCC, 6);
        assert_eq!(STATS_BYTES, 28);
        assert_eq!(FLAG_STATS, 1);
    }

    #[test]
    fn cull_params_std140_tail_is_slab_extents() {
        assert_eq!(size_of::<CullParamsGpu>(), 368);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, flags), 276);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, clip), 280);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, clip_v), 284);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, prev_view_proj), 288);
        assert_eq!(std::mem::offset_of!(CullParamsGpu, occ_flags), 364);
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

    #[test]
    fn camera_buckets_are_k_wide_shadows_are_unbucketed() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        let (parts, _) = dir.partitions();
        assert_eq!(
            GROUPS, 5,
            "Opaque, Cutout, OpaqueLod, ShadowNear, ShadowFar"
        );
        assert_eq!(parts.len(), CAMERA_GROUPS * BUCKETS + SHADOW_GROUPS);
        // Adjacent camera buckets of the same arena share capacity but not offset.
        let b0 = parts[camera_part(0, 0, 0, 1)];
        let b1 = parts[camera_part(0, 0, 1, 1)];
        assert_eq!(b0.capacity, b1.capacity);
        assert_eq!(b1.offset, b0.offset + b0.capacity);
        // Shadow groups occupy one partition per arena, not K.
        assert_eq!(
            shadow_part(1, 0, 1) - shadow_part(0, 0, 1),
            1,
            "cascades are adjacent unbucketed partitions"
        );
    }

    fn partitions_at(dir: &mut ArenaDirectory, eye: EyeSplit) -> (Vec<PartitionGpu>, u32) {
        let mut parts = Vec::new();
        let total = dir.partitions_into(&mut parts, 1, Some(eye));
        (parts, total)
    }

    fn opaque_caps(parts: &[PartitionGpu], arena: usize, n: usize) -> [u32; BUCKETS] {
        std::array::from_fn(|b| parts[camera_part(Group::Opaque as usize, arena, b, n)].capacity)
    }

    #[test]
    fn bucket_intersects_matches_distance_bucket_edges() {
        let s0 = crate::genconst::CULL_BUCKET_SPLIT_0;
        let s1 = crate::genconst::CULL_BUCKET_SPLIT_1;
        let s2 = crate::genconst::CULL_BUCKET_SPLIT_2;
        // A point in bucket 0 cannot reach 1..=3.
        assert!(bucket_intersects(0, 0.0, s0 - 0.01));
        assert!(!bucket_intersects(1, 0.0, s0 - 0.01));
        // Exactly on a split lands in the higher bucket (shader: dist >= split).
        assert!(!bucket_intersects(0, s0, s0));
        assert!(bucket_intersects(1, s0, s0));
        assert!(!bucket_intersects(2, s0, s0));
        // A range that straddles a split keeps both sides.
        assert!(bucket_intersects(0, s0 - 1.0, s0));
        assert!(bucket_intersects(1, s0 - 1.0, s0));
        assert!(bucket_intersects(2, s1, s2));
        assert!(bucket_intersects(3, s1, s2));
        assert!(!bucket_intersects(1, s1, s2));
        // Unbounded last bucket.
        assert!(bucket_intersects(3, s2, 1.0e6));
        assert!(!bucket_intersects(2, s2, 1.0e6));
    }

    #[test]
    fn near_mesh_zeros_every_camera_bucket_but_zero_shadows_untouched() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        let (parts, total) = partitions_at(&mut dir, origin_eye());
        assert_eq!(opaque_caps(&parts, 0, 1), [1, 0, 0, 0]);
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
        assert_eq!(parts[shadow_part(1, 0, 1)].capacity, 1);
        // One camera slot + two unbucketed shadow slots.
        assert_eq!(total, 3);
    }

    #[test]
    fn far_mesh_zeros_every_camera_bucket_but_the_last_shadows_untouched() {
        let far = MeshAabb {
            block: [0, 0, 300],
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, far);
        let (parts, total) = partitions_at(&mut dir, origin_eye());
        assert_eq!(opaque_caps(&parts, 0, 1), [0, 0, 0, 1]);
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
        assert_eq!(parts[shadow_part(1, 0, 1)].capacity, 1);
        assert_eq!(total, 3);
    }

    #[test]
    fn spanning_union_keeps_every_bucket_the_box_can_reach() {
        // A long box along Z from the eye to past split 0 (16), not to split 1 (64).
        let span = MeshAabb {
            block: [0; 3],
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 20.0],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, span);
        let (parts, _) = partitions_at(&mut dir, origin_eye());
        assert_eq!(opaque_caps(&parts, 0, 1), [1, 1, 0, 0]);
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
    }

    #[test]
    fn free_dirties_the_union_so_a_far_companion_stops_keeping_far_buckets() {
        let far = MeshAabb {
            block: [0, 0, 300],
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(1), Pass::Opaque, FULL, far);
        let (parts, _) = partitions_at(&mut dir, origin_eye());
        // Union of near+far covers every bucket.
        assert_eq!(opaque_caps(&parts, 0, 1), [2, 2, 2, 2]);
        dir.note_free(1, G1);
        let (parts, total) = partitions_at(&mut dir, origin_eye());
        assert_eq!(opaque_caps(&parts, 0, 1), [1, 0, 0, 0]);
        assert_eq!(parts[shadow_part(0, 0, 1)].capacity, 1);
        assert_eq!(total, 3);
    }

    #[test]
    fn each_arena_zeros_buckets_from_its_own_union() {
        let far = MeshAabb {
            block: [0, 0, 300],
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(2), Pass::Opaque, FULL, far);
        let (parts, _) = partitions_at(&mut dir, origin_eye());
        assert_eq!(opaque_caps(&parts, 0, 2), [1, 0, 0, 0]);
        assert_eq!(
            [
                parts[camera_part(0, 1, 0, 2)].capacity,
                parts[camera_part(0, 1, 1, 2)].capacity,
                parts[camera_part(0, 1, 2, 2)].capacity,
                parts[camera_part(0, 1, 3, 2)].capacity,
            ],
            [0, 0, 0, 1]
        );
        assert_eq!(parts[shadow_part(0, 0, 2)].capacity, 1);
        assert_eq!(parts[shadow_part(0, 1, 2)].capacity, 1);
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
    fn camera_live_sums_camera_lanes_and_excludes_blend() {
        let mut dir = ArenaDirectory::new();
        assert_eq!(dir.camera_live(), 0);
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(1), Pass::Cutout, FULL, UNIT);
        dir.note_upload(2, G1, buf(1), Pass::Opaque, LOD, UNIT);
        dir.note_upload(3, G1, buf(1), Pass::Blend, FULL, UNIT);
        assert_eq!(dir.camera_live(), 3);
        dir.note_free(0, G1);
        assert_eq!(dir.camera_live(), 2);
    }

    #[test]
    fn occ_mip_picks_a_level_where_the_rect_is_at_most_2x2() {
        let level0 = [64, 64];
        // 2 texels at mip 0 → mip 0 (exactly 2×2).
        assert_eq!(occ_mip_for_rect([2.0 / 64.0, 2.0 / 64.0], level0, 7), 0);
        // Just over 2 texels → mip 1.
        assert_eq!(occ_mip_for_rect([2.1 / 64.0, 1.0 / 64.0], level0, 7), 1);
        // 4 texels → mip 1 (2 texels there).
        assert_eq!(occ_mip_for_rect([4.0 / 64.0, 4.0 / 64.0], level0, 7), 1);
        // 4.1 texels → mip 2.
        assert_eq!(occ_mip_for_rect([4.1 / 64.0, 1.0 / 64.0], level0, 7), 2);
        // Tiny rect stays at mip 0; oversize clamps to last mip.
        assert_eq!(occ_mip_for_rect([0.5 / 64.0, 0.5 / 64.0], level0, 7), 0);
        assert_eq!(occ_mip_for_rect([1.0, 1.0], level0, 3), 2);
    }

    #[test]
    fn occ_hidden_culls_behind_a_nearer_occluder_and_keeps_empty_depth() {
        // Reversed-Z: 0.9 nearer than 0.4. Mesh nearest 0.4 is farther than
        // occluder 0.9 → culled.
        assert!(occ_hidden(0.4, 0.9));
        // Empty pyramid (cleared to 0 = far) never culls.
        assert!(!occ_hidden(0.4, 0.0));
        assert!(!occ_hidden(0.0, 0.0));
    }

    #[test]
    fn occ_hidden_epsilon_favours_drawing() {
        let occ = 0.5;
        // Exactly occ - eps is NOT strictly farther → keep.
        assert!(!occ_hidden(occ - OCC_EPS, occ));
        // A hair past the margin → cull.
        assert!(occ_hidden(occ - OCC_EPS - 1.0e-6, occ));
        // Equal depths → keep.
        assert!(!occ_hidden(occ, occ));
    }

    #[test]
    fn occ_screen_rect_keeps_a_box_that_crosses_the_near_plane() {
        // Identity clip: w = 1 for every corner, so this path is the
        // w <= 0 keep. A translation that puts one corner behind the
        // viewer (w <= 0 after a perspective-like row) uses w = z of the
        // homogeneous result: the last row of this matrix copies z into w.
        let behind = Mat4::from_cols_array_2d(&[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);
        // Corners at z = ±1: z = -1 gives w = -1.
        assert_eq!(
            occ_screen_rect([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], &behind),
            None
        );
    }

    #[test]
    fn occ_gather_then_compare_culls_a_box_fully_behind_a_nearer_occluder() {
        // Unit box in front, projected by a matrix that maps to a small
        // on-screen rect at z/w = 0.2. A 1-texel occluder of 0.9 is nearer.
        let vp = Mat4::from_cols_array_2d(&[
            [0.1, 0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0, 0.0],
            [0.0, 0.0, 0.2, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let (uv_min, uv_max, nearest_z) =
            occ_screen_rect([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], &vp).expect("in front");
        assert!((nearest_z - 0.2).abs() < 1e-5);
        let mip = occ_mip_for_rect([uv_max[0] - uv_min[0], uv_max[1] - uv_min[1]], [64, 64], 7);
        let occ_min = occ_gather_min(uv_min, uv_max, [64, 64], mip, |_x, _y| 0.9);
        assert!(occ_hidden(nearest_z, occ_min));
        let empty = occ_gather_min(uv_min, uv_max, [64, 64], mip, |_x, _y| 0.0);
        assert!(!occ_hidden(nearest_z, empty));
    }
}
