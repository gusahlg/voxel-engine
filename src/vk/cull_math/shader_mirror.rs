//! Host mirror of `shaders/cull.comp.slang`, and the tests that hold the CPU
//! cull to it.
//!
//! [`slang`] follows the shader function by function and statement by
//! statement, in f32 and in the shader's operation order: round to nearest,
//! no fused multiply-add, correctly rounded `sqrt` and division. That is the
//! arithmetic the shader writes down. A device may fuse `a * b + c` or
//! approximate `length` (Vulkan allows both), so parity here is with the
//! shader's arithmetic, not with one GPU. The CPU cull must match it bit for
//! bit: a box one ulp off flips a mesh on a frustum plane between the two
//! paths at `CPU_CULL_MAX`.

use std::num::NonZeroU32;

use ash::vk;
use ash::vk::Handle;
use glam::{IVec3, Vec3, Vec4};

use super::super::arena::{ArenaDirectory, MeshAabb};
use super::super::buffers::{DrawIndexedIndirect, MESH_FLAG_FACE_RUNS, MeshRecord};
use super::super::pipeline::EyeSplit;
use super::{
    CULL_BUCKET_SPLITS, CpuCullScratch, CullView, FLAG_FACE_RUNS, FLAG_STATS, FULL_BUCKET_EDGE_SQ,
    Group, MAX_FACE_RUNS, PartitionGpu, SHADOW_GROUPS, STATS_COUNT, bucket_edge_sq, bucket_splits,
    cage_direction_vis, cam_relative_corners, cam_relative_soa, cpu_cull_into, distance_bucket_sq,
    flatten_part_cmds, lod_bucket_scale, sqrt_ge_threshold,
};
use crate::cage::CageGpu;
use crate::camera::{Camera3D, Frustum, Lens};
use crate::mesh::{Detail, Pass};

/// `cull.comp.slang`, one Rust function per Slang function.
mod slang {
    use glam::{IVec3, Vec3, Vec4};

    use super::super::super::buffers::{DrawIndexedIndirect, MeshRecord};
    use super::super::{PartitionGpu, STATS_COUNT};
    use crate::cage::CageGpu;
    use crate::genconst::{
        CULL_BUCKET_SPLIT_0, CULL_BUCKET_SPLIT_1, CULL_BUCKET_SPLIT_2, CULL_CAGE_DET_REL,
        CULL_CAGE_VIS_BIAS, CULL_CAGED_GROUP, CULL_CAGED_LOD_GROUP, CULL_CAMERA_GROUPS,
        CULL_DISTANCE_BUCKETS, CULL_FLAG_FACE_RUNS, CULL_FLAG_STATS, CULL_OPAQUE_LOD_GROUP,
        CULL_WORKGROUP, DETAIL_GPU_BIAS, DETAIL_GPU_BITS, MESH_FLAG_FACE_RUNS,
    };

    // Literals the shader writes inline. `mirror_literals_match_the_shader`
    // reads each one back out of the source, and checks that the shader
    // reads the generated constants above where this mirror does.
    /// `cage_direction_mask`: `uint mask = 0x40u`, the usable-mask bit.
    pub(super) const CAGE_MASK_USABLE: u32 = 0x40;
    /// Shadow cascades (`c < 2`) and planes per frustum (`p < 5`).
    pub(super) const CASCADES: usize = 2;
    pub(super) const PLANES: usize = 5;
    /// `b[i * 2u] + 6u * (packed & 0xFFFFu)`.
    pub(super) const INDICES_PER_QUAD: u32 = 6;
    pub(super) const QUAD_MASK: u32 = 0xFFFF;

    /// `CullParams`, binding 5. The std140 pads are left out.
    #[derive(Clone, Copy)]
    pub(super) struct CullParams {
        pub cam_planes: [Vec4; PLANES],
        pub shadow_planes: [Vec4; CASCADES * PLANES],
        pub cam_block: IVec3,
        pub slot_count: u32,
        pub cam_frac: Vec3,
        pub arena_count: u32,
        pub shadow_enabled: u32,
        pub flags: u32,
        pub half_x: f32,
        pub half_y: f32,
        pub half_z: f32,
        pub centre_x: f32,
        pub centre_y: f32,
        pub centre_z: f32,
        pub lod_bucket_scale: f32,
    }

    /// The dispatch's descriptor bindings and push constant. `commands`,
    /// `counts` and `stats` are written; `counts` starts zero-filled.
    pub(super) struct Bindings<'a> {
        pub records: &'a [MeshRecord],
        pub slot_arena: &'a [u32],
        pub partitions: &'a [PartitionGpu],
        pub commands: Vec<DrawIndexedIndirect>,
        pub counts: Vec<u32>,
        pub params: CullParams,
        pub visible: &'a [u32],
        pub stats: [u32; STATS_COUNT],
        pub cages: &'a [CageGpu],
        pub pc_flags: u32,
    }

    fn dot(a: Vec3, b: Vec3) -> f32 {
        a.x * b.x + a.y * b.y + a.z * b.z
    }

    pub(super) fn length(v: Vec3) -> f32 {
        dot(v, v).sqrt()
    }

    fn cross(a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }

    fn xyz(v: [f32; 4]) -> Vec3 {
        Vec3::new(v[0], v[1], v[2])
    }

    /// `float3(a - b)` for `int3`: the subtraction wraps.
    fn float3_sub(a: [i32; 3], b: IVec3) -> Vec3 {
        IVec3::from_array(a).wrapping_sub(b).as_vec3()
    }

    /// `common.slang`.
    pub(super) fn detail_scale(detail_pass: u32) -> f32 {
        f32::from_bits(
            ((detail_pass & ((1u32 << DETAIL_GPU_BITS) - 1)) + (127 - DETAIL_GPU_BIAS)) << 23,
        )
    }

    /// `cage.slang`.
    fn cage_corners_aabb(c: &CageGpu) -> (Vec3, Vec3) {
        let mut mn = xyz(c.corners[0]);
        let mut mx = mn;
        for i in 1..8 {
            mn = mn.min(xyz(c.corners[i]));
            mx = mx.max(xyz(c.corners[i]));
        }
        (mn, mx)
    }

    pub(super) fn outside_plane(plane: Vec4, mn: Vec3, mx: Vec3) -> bool {
        let corner = Vec3::new(
            if plane.x >= 0.0 { mx.x } else { mn.x },
            if plane.y >= 0.0 { mx.y } else { mn.y },
            if plane.z >= 0.0 { mx.z } else { mn.z },
        );
        dot(plane.truncate(), corner) + plane.w < 0.0
    }

    pub(super) fn distance_bucket(dist: f32, scale: f32) -> u32 {
        let mut b = 0;
        if dist >= CULL_BUCKET_SPLIT_0 * scale {
            b = 1;
        }
        if dist >= CULL_BUCKET_SPLIT_1 * scale {
            b = 2;
        }
        if dist >= CULL_BUCKET_SPLIT_2 * scale {
            b = 3;
        }
        b.min(CULL_DISTANCE_BUCKETS - 1)
    }

    pub(super) fn lod_aabb_inside_box(
        mn: Vec3,
        mx: Vec3,
        centre: Vec3,
        half_x: f32,
        half_y: f32,
        half_z: f32,
    ) -> bool {
        if !(half_x > 0.0 && half_y > 0.0 && half_z > 0.0) {
            return false;
        }
        let far = (mn - centre).abs().max((mx - centre).abs());
        far.x < half_x && far.y < half_y && far.z < half_z
    }

    /// `cage_direction_mask` up to the mask: `(t_cam, eps)`, or `None` where
    /// the shader returns `0u` for a singular frame (a NaN `det` included,
    /// hence the negated compare).
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    pub(super) fn cage_direction_frame(c: &CageGpu, params: &CullParams) -> Option<(Vec3, f32)> {
        let o = float3_sub(c.anchor, params.cam_block) - params.cam_frac;
        let p0 = xyz(c.corners[0]) + o;
        let p1 = xyz(c.corners[1]) + o;
        let p2 = xyz(c.corners[2]) + o;
        let p3 = xyz(c.corners[3]) + o;
        let p4 = xyz(c.corners[4]) + o;
        let p5 = xyz(c.corners[5]) + o;
        let p6 = xyz(c.corners[6]) + o;
        let p7 = xyz(c.corners[7]) + o;
        let e0 = p1 - p0;
        let e1 = p2 - p0;
        let e2 = p4 - p0;
        let c12 = cross(e1, e2);
        let c20 = cross(e2, e0);
        let c01 = cross(e0, e1);
        let det = dot(e0, c12);
        let l0 = length(e0);
        let l1 = length(e1);
        let l2 = length(e2);
        let vol = l0 * l1 * l2;
        if !(det.abs() > CULL_CAGE_DET_REL * vol.max(1.0)) {
            return None;
        }
        let b = -p0;
        let inv = 1.0 / det;
        let t_cam = Vec3::new(dot(b, c12), dot(b, c20), dot(b, c01)) * inv;
        let min_edge = l0.min(l1.min(l2));
        let mut dev = length(p3 - (p0 + e0 + e1));
        dev = dev.max(length(p5 - (p0 + e0 + e2)));
        dev = dev.max(length(p6 - (p0 + e1 + e2)));
        dev = dev.max(length(p7 - (p0 + e0 + e1 + e2)));
        let eps = 2.0 * (dev / min_edge) + CULL_CAGE_VIS_BIAS;
        Some((t_cam, eps))
    }

    pub(super) fn cage_direction_mask(c: &CageGpu, params: &CullParams) -> u32 {
        let Some((t_cam, eps)) = cage_direction_frame(c, params) else {
            return 0;
        };
        let mut mask = CAGE_MASK_USABLE;
        if t_cam.x > -eps {
            mask |= 1;
        }
        if t_cam.y > -eps {
            mask |= 2;
        }
        if t_cam.z > -eps {
            mask |= 4;
        }
        if t_cam.x < 1.0 + eps {
            mask |= 8;
        }
        if t_cam.y < 1.0 + eps {
            mask |= 16;
        }
        if t_cam.z < 1.0 + eps {
            mask |= 32;
        }
        mask
    }

    /// The non-wave `emit`.
    fn emit(b: &mut Bindings, part: u32, cmd: DrawIndexedIndirect) {
        let part = part as usize;
        let i = b.counts[part];
        b.counts[part] = i + 1;
        if i < b.partitions[part].capacity {
            b.commands[(b.partitions[part].offset + i) as usize] = cmd;
        }
    }

    /// `computeMain`'s camera-relative `mn` / `mx`.
    pub(super) fn slot_box(b: &Bindings, rec: &MeshRecord, scale: f32) -> (Vec3, Vec3) {
        let mut mn;
        let mut mx;
        let caged = rec.cage != 0;
        if caged {
            let cg = &b.cages[rec.cage as usize];
            (mn, mx) = cage_corners_aabb(cg);
            let o = float3_sub(cg.anchor, b.params.cam_block) - b.params.cam_frac;
            mn += o;
            mx += o;
        } else {
            let offset = float3_sub(rec.block, b.params.cam_block) - b.params.cam_frac
                + Vec3::from_array(rec.local_off);
            mn = Vec3::from_array(rec.aabb_min) * scale + offset;
            mx = Vec3::from_array(rec.aabb_max) * scale + offset;
        }
        (mn, mx)
    }

    fn count_stats(b: &mut Bindings, group: u32, index_count: u32) {
        if (b.pc_flags & CULL_FLAG_STATS) != 0 {
            b.stats[(group * 2) as usize] += 1;
            b.stats[(group * 2 + 1) as usize] += index_count;
        }
    }

    fn compute_main(b: &mut Bindings, id_x: u32) {
        let slot = id_x;
        if slot >= b.params.slot_count {
            return;
        }
        let word = b.slot_arena[slot as usize];
        if word == 0 {
            return; // dead slot
        }
        let cam_visible = (b.visible[(slot >> 5) as usize] & (1u32 << (slot & 31))) != 0;
        if !cam_visible && b.params.shadow_enabled == 0 {
            return;
        }

        let arena = word - 1;
        let rec = b.records[slot as usize];
        let pass = (rec.detail_pass >> DETAIL_GPU_BITS) & 3;
        if pass > 1 {
            return; // Blend: CPU-sorted path
        }

        let scale = detail_scale(rec.detail_pass);
        let caged = rec.cage != 0;
        let (mn, mx) = slot_box(b, &rec, scale);

        let cmd = DrawIndexedIndirect {
            index_count: rec.index_count,
            instance_count: 1,
            first_index: 0,
            vertex_offset: rec.vertex_offset,
            first_instance: slot,
        };

        if cam_visible {
            let mut in_frustum = true;
            for p in 0..PLANES {
                in_frustum = in_frustum && !outside_plane(b.params.cam_planes[p], mn, mx);
            }
            if in_frustum {
                let group = if caged {
                    if scale > 1.0 {
                        CULL_CAGED_LOD_GROUP
                    } else {
                        CULL_CAGED_GROUP
                    }
                } else if pass == 0 && scale > 1.0 {
                    CULL_OPAQUE_LOD_GROUP
                } else {
                    pass
                };
                let lod_group = group == CULL_OPAQUE_LOD_GROUP || group == CULL_CAGED_LOD_GROUP;
                let p = b.params;
                let lod_centre = Vec3::new(p.centre_x, p.centre_y, p.centre_z);
                if !lod_group
                    || !lod_aabb_inside_box(mn, mx, lod_centre, p.half_x, p.half_y, p.half_z)
                {
                    let dist = length(0.5 * (mn + mx));
                    let bucket_scale = if lod_group { p.lod_bucket_scale } else { 1.0 };
                    let bucket = distance_bucket(dist, bucket_scale);
                    let part = (group * p.arena_count + arena) * CULL_DISTANCE_BUCKETS + bucket;
                    let mut face_runs = (p.flags & CULL_FLAG_FACE_RUNS) != 0
                        && (rec.flags & MESH_FLAG_FACE_RUNS) != 0;
                    let mut vis = [false; 6];
                    if face_runs && caged {
                        // Singular frame (mask 0) draws whole. Otherwise bits 0..5.
                        let mask = cage_direction_mask(&b.cages[rec.cage as usize], &p);
                        if mask == 0 {
                            face_runs = false;
                        } else {
                            for (k, v) in vis.iter_mut().enumerate() {
                                *v = (mask & (1u32 << k)) != 0;
                            }
                        }
                    } else if face_runs {
                        vis = [
                            mn.x < 0.0,
                            mn.y < 0.0,
                            mn.z < 0.0,
                            mx.x > 0.0,
                            mx.y > 0.0,
                            mx.z > 0.0,
                        ];
                    }
                    if !face_runs {
                        emit(b, part, cmd);
                        count_stats(b, group, cmd.index_count);
                    } else {
                        let mut bounds = [0u32; 7];
                        for i in 0..3 {
                            let packed = rec.face_quads[i];
                            bounds[i * 2 + 1] =
                                bounds[i * 2] + INDICES_PER_QUAD * (packed & QUAD_MASK);
                            bounds[i * 2 + 2] =
                                bounds[i * 2 + 1] + INDICES_PER_QUAD * (packed >> 16);
                        }
                        let mut start = !0u32;
                        for k in 0..6usize {
                            let occ = vis[k] && (bounds[k + 1] != bounds[k]);
                            if occ {
                                if start == !0u32 {
                                    start = k as u32;
                                }
                            } else if start != !0u32 {
                                let mut run = cmd;
                                run.first_index = bounds[start as usize];
                                run.index_count = bounds[k] - bounds[start as usize];
                                emit(b, part, run);
                                count_stats(b, group, run.index_count);
                                start = !0u32;
                            }
                        }
                        if start != !0u32 {
                            let mut run = cmd;
                            run.first_index = bounds[start as usize];
                            run.index_count = bounds[6] - bounds[start as usize];
                            emit(b, part, run);
                            count_stats(b, group, run.index_count);
                        }
                    }
                }
            }
        }

        // Shadow casters: full-res opaque only.
        if pass == 0 && scale <= 1.0 && b.params.shadow_enabled != 0 {
            let shadow_base = CULL_CAMERA_GROUPS * b.params.arena_count * CULL_DISTANCE_BUCKETS;
            for c in 0..CASCADES {
                let mut cast = true;
                for p in 0..PLANES {
                    cast = cast && !outside_plane(b.params.shadow_planes[c * PLANES + p], mn, mx);
                }
                if cast {
                    emit(
                        b,
                        shadow_base + c as u32 * b.params.arena_count + arena,
                        cmd,
                    );
                }
            }
        }
    }

    /// One thread per id over `ceil(slot_count / CULL_WORKGROUP)` groups, in
    /// id order. The GPU runs them in any order; a partition's commands are
    /// compared as a set.
    pub(super) fn dispatch(b: &mut Bindings) {
        let threads = b.params.slot_count.div_ceil(CULL_WORKGROUP) * CULL_WORKGROUP;
        for id in 0..threads {
            compute_main(b, id);
        }
    }
}

const G1: NonZeroU32 = NonZeroU32::new(1).unwrap();

/// SplitMix64: deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.below(from.len() as u32) as usize]
    }

    fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }

    /// `[0, 1)`, 24 random mantissa bits.
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform in `-spread ..= spread`.
    fn spread(&mut self, spread: i32) -> i32 {
        let width = 2 * i64::from(spread) + 1;
        ((self.next() % width as u64) as i64 - i64::from(spread)) as i32
    }

    fn unit_vec(&mut self) -> Vec3 {
        loop {
            let v = Vec3::new(
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
                self.range(-1.0, 1.0),
            );
            let len = v.length();
            if len > 0.1 && len <= 1.0 {
                return v / len;
            }
        }
    }
}

/// One frame's cull inputs, registered the way `apply_upload_mesh` does.
struct Scene {
    records: Vec<MeshRecord>,
    /// Index 0 is the zero cage, as in the GPU table.
    cages: Vec<CageGpu>,
    dir: ArenaDirectory,
    arrived: Vec<bool>,
    visible: Vec<u32>,
    eye: EyeSplit,
    camera: Frustum,
    shadow: [Frustum; 2],
    half: [f32; 3],
    centre: [f32; 3],
}

impl Scene {
    fn slot_count(&self) -> u32 {
        self.dir.live_end()
    }

    /// `RecordTable` gates a slot's word to 0 until its copy has arrived.
    fn slot_arena(&self) -> Vec<u32> {
        (0..self.records.len())
            .map(|s| {
                if self.arrived[s] {
                    self.dir.arena_word(s)
                } else {
                    0
                }
            })
            .collect()
    }

    /// The `CullParams` that `CullState::prepare` uploads for these inputs.
    fn params(&self, face_cull: bool, shadow_on: bool) -> slang::CullParams {
        let mut shadow_planes = [Vec4::ZERO; slang::CASCADES * slang::PLANES];
        if shadow_on {
            for (c, f) in self.shadow.iter().enumerate() {
                for (p, plane) in f.planes().iter().enumerate() {
                    shadow_planes[c * slang::PLANES + p] = *plane;
                }
            }
        }
        slang::CullParams {
            cam_planes: self.camera.planes(),
            shadow_planes,
            cam_block: IVec3::from_array(self.eye.block),
            slot_count: self.slot_count(),
            cam_frac: Vec3::from_array(self.eye.frac),
            arena_count: self.dir.arena_count() as u32,
            shadow_enabled: u32::from(shadow_on),
            flags: if face_cull { FLAG_FACE_RUNS } else { 0 },
            half_x: self.half[0],
            half_y: self.half[1],
            half_z: self.half[2],
            centre_x: self.centre[0],
            centre_y: self.centre[1],
            centre_z: self.centre[2],
            lod_bucket_scale: lod_bucket_scale(),
        }
    }

    fn bindings<'a>(
        &'a self,
        slot_arena: &'a [u32],
        partitions: &'a [PartitionGpu],
        face_cull: bool,
        shadow_on: bool,
    ) -> slang::Bindings<'a> {
        let total: usize = partitions.iter().map(|p| p.capacity as usize).sum();
        slang::Bindings {
            records: &self.records,
            slot_arena,
            partitions,
            commands: vec![bytemuck::Zeroable::zeroed(); total],
            counts: vec![0; partitions.len()],
            params: self.params(face_cull, shadow_on),
            visible: &self.visible,
            stats: [0; STATS_COUNT],
            cages: &self.cages,
            pc_flags: FLAG_STATS,
        }
    }

    /// Slots the shader reaches past its early returns (some camera or
    /// shadow test runs on them).
    fn culled_slots(&self) -> Vec<u32> {
        let slot_arena = self.slot_arena();
        (0..self.slot_count())
            .filter(|&s| {
                slot_arena[s as usize] != 0
                    && self.records[s as usize].pass() != Pass::Blend
                    && self.dir.cull_bits()[s as usize] != 0
            })
            .collect()
    }
}

fn random_block(rng: &mut Rng) -> [i32; 3] {
    let spread = rng.pick(&[4, 64, 4096, 1 << 22, i32::MAX]);
    std::array::from_fn(|_| rng.spread(spread))
}

/// A cage around `[0, size]³`: rotated, shifted, often bent, sometimes
/// collapsed (a singular frame).
fn random_cage(rng: &mut Rng, anchor: [i32; 3]) -> CageGpu {
    let size = rng.range(1.0, 32.0);
    let rot = glam::Quat::from_axis_angle(rng.unit_vec(), rng.range(0.0, std::f32::consts::TAU));
    let shift = Vec3::new(
        rng.range(-8.0, 8.0),
        rng.range(-8.0, 8.0),
        rng.range(-8.0, 8.0),
    );
    let mut corners: [Vec3; 8] = std::array::from_fn(|i| {
        let local = Vec3::new(
            if i & 1 != 0 { size } else { 0.0 },
            if i & 2 != 0 { size } else { 0.0 },
            if i & 4 != 0 { size } else { 0.0 },
        );
        rot * local + shift
    });
    if rng.chance(0.5) {
        for _ in 0..1 + rng.below(3) {
            let i = rng.below(8) as usize;
            corners[i] += rng.unit_vec() * rng.range(0.0, 0.3 * size);
        }
    }
    match rng.below(20) {
        0 => corners = [shift; 8],
        1 => {
            for c in &mut corners {
                c.y = shift.y;
            }
        }
        _ => {}
    }
    CageGpu::from_corners(IVec3::from_array(anchor), corners)
}

/// A local box coordinate: whole blocks, or an arbitrary mantissa.
fn random_coord(rng: &mut Rng) -> f32 {
    if rng.chance(0.5) {
        rng.below(17) as f32
    } else {
        rng.range(0.0, 16.0)
    }
}

fn random_record(rng: &mut Rng, eye: [i32; 3], cages: &mut Vec<CageGpu>) -> MeshRecord {
    let pass = rng.pick(&[
        Pass::Opaque,
        Pass::Opaque,
        Pass::Opaque,
        Pass::Cutout,
        Pass::Cutout,
        Pass::Blend,
    ]);
    let detail = Detail(rng.pick(&[-2, -1, 0, 0, 0, 1, 2, 3, 5]));
    let spread = rng.pick(&[16, 300, 5000, 1 << 20]);
    let block = std::array::from_fn(|k| eye[k].wrapping_add(rng.spread(spread)));
    let mut aabb_min = [0.0; 3];
    let mut aabb_max = [0.0; 3];
    for k in 0..3 {
        let a = random_coord(rng);
        let b = if rng.chance(0.1) {
            a
        } else {
            random_coord(rng)
        };
        aabb_min[k] = a.min(b);
        aabb_max[k] = a.max(b);
    }
    let local_off = if rng.chance(0.4) {
        [0.0; 3]
    } else {
        std::array::from_fn(|_| rng.range(-2.0, 2.0))
    };
    let mut face_quads = [0u32; 3];
    let mut quads = 0;
    for packed in &mut face_quads {
        let lo = if rng.chance(0.25) { 0 } else { rng.below(6) };
        let hi = if rng.chance(0.25) { 0 } else { rng.below(6) };
        *packed = lo | (hi << 16);
        quads += lo + hi;
    }
    let cage = if rng.chance(0.3) {
        let anchor = std::array::from_fn(|k| block[k].wrapping_add(rng.spread(4)));
        cages.push(random_cage(rng, anchor));
        (cages.len() - 1) as u32
    } else {
        0
    };
    MeshRecord {
        block,
        detail_pass: u32::from(detail.to_gpu_bits())
            | ((pass as u32) << crate::genconst::DETAIL_GPU_BITS),
        local_off,
        cage,
        aabb_min,
        index_count: 6 * quads.max(1),
        aabb_max,
        vertex_offset: rng.spread(1000),
        face_quads,
        flags: if rng.chance(0.8) {
            MESH_FLAG_FACE_RUNS
        } else {
            0
        },
    }
}

/// `apply_upload_mesh`: the record's box, or its cage's corner box.
fn register(dir: &mut ArenaDirectory, slot: u32, rec: &MeshRecord, cages: &[CageGpu], buf: u64) {
    let buffer = vk::Buffer::from_raw(buf);
    let lod = rec.detail_scale() > 1.0;
    if rec.cage != 0 {
        let entry = cages[rec.cage as usize];
        let aabb = MeshAabb::from_cage(&entry);
        dir.note_upload_caged(slot, G1, buffer, rec.pass(), lod, aabb, entry.corners3());
    } else {
        dir.note_upload(
            slot,
            G1,
            buffer,
            rec.pass(),
            lod,
            MeshAabb::from_record(rec),
        );
    }
    dir.note_cull_draw(slot, rec);
}

fn perspective(rng: &mut Rng, position: Vec3) -> Frustum {
    let forward = rng.unit_vec();
    let up = if forward.y.abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let cam = Camera3D {
        position,
        target: position + forward,
        up,
        fovy: rng.range(20.0, 120.0),
        lens: Lens::Rectilinear,
    };
    Frustum::from_view_proj(&cam.view_proj(rng.range(0.5, 2.5)))
}

fn random_planes(rng: &mut Rng) -> Frustum {
    Frustum::from_planes(std::array::from_fn(|_| {
        rng.unit_vec().extend(rng.range(-300.0, 300.0))
    }))
}

/// Keeps every box: zero normal, positive distance.
const OPEN: Vec4 = Vec4::new(0.0, 0.0, 0.0, 1.0);

/// A plane along one axis that the box touches: its p-vertex gives
/// `dot + w == 0`, inside by the strict `< 0` test, and a box one ulp
/// smaller on that side is outside.
fn touching_plane(rng: &mut Rng, mn: Vec3, mx: Vec3) -> Vec4 {
    let axis = rng.below(3) as usize;
    let mut n = Vec3::ZERO;
    if rng.chance(0.5) {
        n[axis] = 1.0;
        n.extend(-mx[axis])
    } else {
        n[axis] = -1.0;
        n.extend(mn[axis])
    }
}

fn random_scene(rng: &mut Rng) -> Scene {
    let eye = EyeSplit {
        block: random_block(rng),
        _pad0: 0,
        frac: [rng.unit(), rng.unit(), rng.unit()],
        _pad1: 0.0,
    };
    let n = 1 + rng.below(96) as usize;
    let mut records = vec![bytemuck::Zeroable::zeroed(); n];
    let mut cages = vec![CageGpu::ZERO];
    let mut dir = ArenaDirectory::new();
    let mut arrived = vec![true; n];
    let mut visible = vec![0u32; n.div_ceil(32)];
    for slot in 0..n as u32 {
        if rng.chance(0.85) {
            visible[(slot >> 5) as usize] |= 1 << (slot & 31);
        }
        arrived[slot as usize] = rng.chance(0.9);
        if rng.chance(0.08) {
            continue; // never registered
        }
        let rec = random_record(rng, eye.block, &mut cages);
        records[slot as usize] = rec;
        register(&mut dir, slot, &rec, &cages, 1 + u64::from(rng.below(3)));
        if rng.chance(0.05) {
            dir.note_free(slot, G1);
        }
    }
    let (half, centre) = if rng.chance(0.3) {
        ([0.0; 3], [0.0; 3])
    } else {
        (
            std::array::from_fn(|_| rng.range(0.0, 300.0)),
            std::array::from_fn(|_| rng.range(-100.0, 100.0)),
        )
    };
    let shadow_at = rng.unit_vec() * rng.range(0.0, 200.0);
    let mut scene = Scene {
        records,
        cages,
        dir,
        arrived,
        visible,
        eye,
        camera: perspective(rng, Vec3::ZERO),
        shadow: [perspective(rng, shadow_at), random_planes(rng)],
        half,
        centre,
    };
    if rng.chance(0.6) {
        snap_to_slots(rng, &mut scene);
    }
    scene
}

/// Puts planes and the LOD box exactly on the shader's box of random slots,
/// so a CPU box one ulp off changes what is drawn.
fn snap_to_slots(rng: &mut Rng, scene: &mut Scene) {
    let slots = scene.culled_slots();
    if slots.is_empty() {
        return;
    }
    let slot_arena = scene.slot_arena();
    let b = scene.bindings(&slot_arena, &[], false, false);
    let shader_box = |s: u32| {
        let rec = &scene.records[s as usize];
        slang::slot_box(&b, rec, slang::detail_scale(rec.detail_pass))
    };
    let mut cam = if rng.chance(0.5) {
        [OPEN; 5]
    } else {
        scene.camera.planes()
    };
    let mut shadow = [scene.shadow[0].planes(), [OPEN; 5]];
    for _ in 0..1 + rng.below(3) {
        let (mn, mx) = shader_box(rng.pick(&slots));
        cam[rng.below(5) as usize] = touching_plane(rng, mn, mx);
        let (mn, mx) = shader_box(rng.pick(&slots));
        let cascade = rng.below(2) as usize;
        shadow[cascade][rng.below(5) as usize] = touching_plane(rng, mn, mx);
    }
    // The LOD skip is strict too: one axis exactly at the farthest corner.
    let lod: Vec<u32> = slots
        .iter()
        .copied()
        .filter(|&s| scene.records[s as usize].detail_scale() > 1.0)
        .collect();
    let lod_box = (!lod.is_empty()).then(|| shader_box(rng.pick(&lod)));
    drop(b);
    if let Some((mn, mx)) = lod_box {
        let mid = 0.5 * (mn + mx);
        let centre = mid + Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), 0.0);
        let far = (mn - centre).abs().max((mx - centre).abs());
        let tight = rng.below(3) as usize;
        let mut half = far + Vec3::ONE;
        half[tight] = far[tight];
        scene.centre = centre.to_array();
        scene.half = half.to_array();
    }
    scene.camera = Frustum::from_planes(cam);
    scene.shadow = shadow.map(Frustum::from_planes);
}

/// Per-slot intermediates: the camera-relative box, the distance bucket at
/// several scales, and a caged slot's frame, each bit for bit.
fn assert_slot_math_matches(scene: &Scene, seed: u64) {
    let slot_arena = scene.slot_arena();
    let b = scene.bindings(&slot_arena, &[], true, false);
    let eye = scene.eye;
    let scales = [1.0, lod_bucket_scale(), 1.25, 1.5, 3.0, 0.7, 100.3];
    for slot in scene.culled_slots() {
        let i = slot as usize;
        let rec = &scene.records[i];
        let (smn, smx) = slang::slot_box(&b, rec, slang::detail_scale(rec.detail_pass));
        let (mn, mx) = cam_relative_soa(
            scene.dir.cull_aabbs()[i],
            scene.dir.cull_local_offs()[i],
            scene.dir.cull_blocks()[i],
            eye,
        );
        let bits = |v: [f32; 3]| v.map(f32::to_bits);
        assert_eq!(
            bits(mn),
            bits(smn.to_array()),
            "seed {seed} slot {slot} min"
        );
        assert_eq!(
            bits(mx),
            bits(smx.to_array()),
            "seed {seed} slot {slot} max"
        );

        let c = [
            0.5 * (mn[0] + mx[0]),
            0.5 * (mn[1] + mx[1]),
            0.5 * (mn[2] + mx[2]),
        ];
        let d2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
        let dist = slang::length(0.5 * (smn + smx));
        for scale in scales {
            assert_eq!(
                distance_bucket_sq(d2, bucket_edge_sq(scale)),
                slang::distance_bucket(dist, scale),
                "seed {seed} slot {slot} bucket at scale {scale}"
            );
        }

        if rec.cage != 0 {
            let corners =
                cam_relative_corners(scene.dir.cull_corners()[i], scene.dir.cull_blocks()[i], eye);
            let cage = &scene.cages[rec.cage as usize];
            let cpu = cage_direction_vis(corners);
            let shader = slang::cage_direction_frame(cage, &b.params);
            assert_eq!(
                cpu.is_some(),
                shader.is_some(),
                "seed {seed} slot {slot} singular frame"
            );
            if let (Some(cpu), Some((t_cam, eps))) = (cpu, shader) {
                assert_eq!(
                    bits(cpu.t_cam),
                    bits(t_cam.to_array()),
                    "seed {seed} slot {slot} t_cam"
                );
                assert_eq!(
                    cpu.eps.to_bits(),
                    eps.to_bits(),
                    "seed {seed} slot {slot} eps"
                );
                let mask = slang::cage_direction_mask(cage, &b.params);
                let vis: [bool; 6] = std::array::from_fn(|k| mask & (1 << k) != 0);
                assert_eq!(cpu.vis, vis, "seed {seed} slot {slot} direction mask");
            }
        }
    }
}

/// A partition's stored commands, in a fixed order: the GPU appends them in
/// any order.
fn stored(
    parts: &[PartitionGpu],
    cmds: &[DrawIndexedIndirect],
    counts: &[u32],
    i: usize,
) -> Vec<(u32, u32, u32, i32, u32)> {
    let n = counts[i].min(parts[i].capacity) as usize;
    let start = parts[i].offset as usize;
    let mut out: Vec<_> = cmds[start..start + n]
        .iter()
        .map(|c| {
            (
                c.first_instance,
                c.first_index,
                c.index_count,
                c.vertex_offset,
                c.instance_count,
            )
        })
        .collect();
    out.sort_unstable();
    out
}

/// Runs `cpu_cull_into` and the shader mirror on one scene, every
/// face-run/shadow combination, and compares counts, the geometry histogram
/// and each partition's commands.
fn assert_emission_matches(scene: &mut Scene, seed: u64) {
    let slot_arena = scene.slot_arena();
    let mut scratch = CpuCullScratch::default();
    for face_cull in [false, true] {
        for shadow_on in [false, true] {
            let mut parts = Vec::new();
            let runs = if face_cull { MAX_FACE_RUNS } else { 1 };
            scene.dir.partitions_into(&mut parts, runs, Some(scene.eye));
            let view = CullView {
                camera: &scene.camera,
                shadow: shadow_on.then_some(&scene.shadow),
                eye: scene.eye,
                slot_count: scene.slot_count(),
                half: scene.half,
                centre: scene.centre,
                face_cull,
            };
            let stats = cpu_cull_into(
                &scene.dir,
                |s| scene.arrived[s as usize],
                &scene.visible,
                &parts,
                &view,
                &mut scratch,
            );
            let cmds = flatten_part_cmds(&scratch.part_cmds, &parts);

            let mut b = scene.bindings(&slot_arena, &parts, face_cull, shadow_on);
            slang::dispatch(&mut b);

            let case = format!("seed {seed} face_cull {face_cull} shadow {shadow_on}");
            assert_eq!(scratch.counts, b.counts, "{case}: partition counts");
            assert_eq!(stats, b.stats, "{case}: stats histogram");
            for i in 0..parts.len() {
                assert_eq!(
                    stored(&parts, &cmds, &scratch.counts, i),
                    stored(&parts, &b.commands, &b.counts, i),
                    "{case}: partition {i} commands"
                );
            }
        }
    }
}

#[test]
fn cpu_cull_matches_the_shader_on_random_scenes() {
    const SCENES: u64 = 400;
    for seed in 0..SCENES {
        let mut rng = Rng(seed);
        let mut scene = random_scene(&mut rng);
        assert_slot_math_matches(&scene, seed);
        assert_emission_matches(&mut scene, seed);
    }
}

/// A mover whose box the old CPU order, `(aabb * scale + local_off) +
/// ((block - cam) - frac)`, rounded differently from the shader.
#[test]
fn a_moved_mesh_box_rounds_like_the_shader() {
    let mut rec: MeshRecord = bytemuck::Zeroable::zeroed();
    rec.block = [1000, -3, 77];
    rec.detail_pass = u32::from(Detail::FULL.to_gpu_bits());
    rec.local_off = [0.1, 0.3, -0.7];
    rec.aabb_min = [0.7, 3.3, 1.1];
    rec.aabb_max = [15.9, 4.9, 2.6];
    let eye = EyeSplit {
        block: [0, 5, 3],
        _pad0: 0,
        frac: [0.3, 0.6, 0.9],
        _pad1: 0.0,
    };
    let mut dir = ArenaDirectory::new();
    register(&mut dir, 0, &rec, &[CageGpu::ZERO], 1);
    let (mn, _) = cam_relative_soa(
        dir.cull_aabbs()[0],
        dir.cull_local_offs()[0],
        dir.cull_blocks()[0],
        eye,
    );
    let rel = |k: usize| (rec.block[k] - eye.block[k]) as f32 - eye.frac[k];
    let shader: [f32; 3] = std::array::from_fn(|k| rec.aabb_min[k] + (rel(k) + rec.local_off[k]));
    let old: [f32; 3] = std::array::from_fn(|k| (rec.aabb_min[k] + rec.local_off[k]) + rel(k));
    assert_eq!(mn, shader);
    assert_ne!(old, shader, "this case no longer separates the two orders");
}

#[test]
fn bucket_thresholds_match_the_shader_length_at_every_edge() {
    for scale in [
        1.0,
        32.0,
        1.25,
        1.5,
        3.0,
        0.7,
        1.1,
        100.3,
        lod_bucket_scale(),
    ] {
        let edges = bucket_edge_sq(scale);
        for split in bucket_splits(scale) {
            let mut d2 = split * split;
            for _ in 0..64 {
                d2 = d2.next_down();
            }
            for _ in 0..128 {
                assert_eq!(
                    distance_bucket_sq(d2, edges),
                    slang::distance_bucket(d2.sqrt(), scale),
                    "scale {scale} d2 {d2}"
                );
                d2 = d2.next_up();
            }
        }
    }
    // Edge 20 (16 × 1.25): the float below 400 already has sqrt 20.
    assert!(sqrt_ge_threshold(20.0) < 400.0);
    assert_eq!(sqrt_ge_threshold(20.0).sqrt(), 20.0);
    assert!(sqrt_ge_threshold(20.0).next_down().sqrt() < 20.0);
    // Power-of-two splits keep their exact squares.
    assert_eq!(
        CULL_BUCKET_SPLITS.map(sqrt_ge_threshold),
        FULL_BUCKET_EDGE_SQ
    );
}

/// The body of the Slang function whose signature starts with `head`.
fn shader_fn<'a>(src: &'a str, head: &str) -> &'a str {
    let at = src
        .find(head)
        .unwrap_or_else(|| panic!("cull.comp.slang has no `{head}`"));
    let open = at + src[at..].find('{').expect("function body");
    let mut depth = 0u32;
    for (i, c) in src[open..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return &src[open..=open + i];
                }
            }
            _ => {}
        }
    }
    panic!("`{head}` has no closing brace")
}

/// The literal or identifier after each `marker` in `text`. At least one.
fn literals_after<'a>(text: &'a str, marker: &str) -> Vec<&'a str> {
    let found: Vec<&str> = text
        .match_indices(marker)
        .map(|(at, _)| {
            let rest = &text[at + marker.len()..];
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-' | '+')))
                .unwrap_or(rest.len());
            &rest[..end]
        })
        .collect();
    assert!(!found.is_empty(), "cull.comp.slang has no `{marker}`");
    found
}

fn uint(lit: &str) -> u32 {
    let lit = lit.trim_end_matches('u');
    match lit.strip_prefix("0x") {
        Some(hex) => u32::from_str_radix(hex, 16),
        None => lit.parse(),
    }
    .unwrap_or_else(|_| panic!("`{lit}` is not an integer"))
}

/// Every literal the mirror copies by hand, read back out of the shader, and
/// every generated constant the mirror and the CPU cull share with it, found
/// by name where the shader uses it.
#[test]
fn mirror_literals_match_the_shader() {
    let src = include_str!("../../../shaders/cull.comp.slang");

    let cage = shader_fn(src, "uint cage_direction_mask(Cage c)");
    assert_eq!(literals_after(cage, "abs(det) > "), ["CULL_CAGE_DET_REL"]);
    assert_eq!(
        literals_after(cage, "(dev / min_edge) + "),
        ["CULL_CAGE_VIS_BIAS"]
    );
    assert_eq!(
        literals_after(cage, "uint mask = ")
            .into_iter()
            .map(uint)
            .collect::<Vec<_>>(),
        [slang::CAGE_MASK_USABLE]
    );

    let main = shader_fn(src, "void computeMain(");
    assert_eq!(
        literals_after(main, "(pass == 0u && scale > 1.0) ? "),
        ["CULL_OPAQUE_LOD_GROUP"]
    );
    assert_eq!(
        literals_after(main, "bool lod_group = group == "),
        ["CULL_OPAQUE_LOD_GROUP"]
    );
    assert_eq!(
        Group::OpaqueLod as u32,
        crate::genconst::CULL_OPAQUE_LOD_GROUP
    );
    assert_eq!(
        literals_after(main, "(params.flags & "),
        ["CULL_FLAG_FACE_RUNS"]
    );
    assert_eq!(FLAG_FACE_RUNS, crate::genconst::CULL_FLAG_FACE_RUNS);
    assert_eq!(
        literals_after(main, "(rec.flags & "),
        ["MESH_FLAG_FACE_RUNS"]
    );
    assert_eq!(MESH_FLAG_FACE_RUNS, crate::genconst::MESH_FLAG_FACE_RUNS);
    assert_eq!(literals_after(main, "(pc.flags & "), ["CULL_FLAG_STATS"; 3]);
    assert_eq!(FLAG_STATS, crate::genconst::CULL_FLAG_STATS);
    let uints =
        |marker: &str| -> Vec<u32> { literals_after(main, marker).into_iter().map(uint).collect() };
    assert_eq!(uints("for (int c = 0; c < "), [slang::CASCADES as u32]);
    assert_eq!(slang::CASCADES, SHADOW_GROUPS);
    assert_eq!(uints("for (int p = 0; p < "), [slang::PLANES as u32; 2]);
    assert_eq!(uints("params.shadow_planes[c * "), [slang::PLANES as u32]);
    assert_eq!(uints("b[i * 2u] + "), [slang::INDICES_PER_QUAD]);
    assert_eq!(uints("b[i * 2u + 1u] + "), [slang::INDICES_PER_QUAD]);
    assert_eq!(uints("(packed & "), [slang::QUAD_MASK]);
}
