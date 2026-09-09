use super::arena::ArenaDirectory;
use super::buffers::{DrawIndexedIndirect, MESH_FLAG_FACE_RUNS, MeshRecord};
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

pub(crate) fn cpu_cull_max() -> u32 {
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
}
