use ash::vk;

use super::cull::ArenaDirectory;
use super::handles::{DrawDyn, MeshMeta};
use super::host_buffer::HostBuffer;
use super::mesh_residency::MeshResidency;
use crate::mesh::{Detail, Pass};
use crate::rev::FRAMES_IN_FLIGHT;

/// Persistent per-mesh record, indexed by slot, mirrored in shaders.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshRecord {
    pub block: [i32; 3],
    pub detail_pass: u32,
    pub local_off: [f32; 3],
    pub _pad: u32,
    pub aabb_min: [f32; 3],
    pub index_count: u32,
    pub aabb_max: [f32; 3],
    pub vertex_offset: i32,
    /// Packed u16 quad counts for upload slots (0|1, 2|3, 4|5).
    pub face_quads: [u32; 3],
    /// Bit 0 ([`MESH_FLAG_FACE_RUNS`]): `face_quads` are valid u16 counts.
    /// Clear → the cull shader emits a whole-mesh draw for this record.
    pub flags: u32,
}

/// `MeshRecord::flags` bit 0: packed `face_quads` fit in u16. Clear on overflow
/// so GPU face-run culling falls back to a whole-mesh command for that mesh.
pub(crate) const MESH_FLAG_FACE_RUNS: u32 = 1;

// Stride must match the vertex shaders exactly; layout drift corrupts every draw.
const _: () = assert!(std::mem::size_of::<MeshRecord>() == 80);
const _: () = assert!(std::mem::offset_of!(MeshRecord, face_quads) == 64);
const _: () = assert!(std::mem::offset_of!(MeshRecord, flags) == 76);

impl MeshRecord {
    /// Decode pass bits from detail_pass.
    pub(crate) fn pass(&self) -> Pass {
        match (self.detail_pass >> 4) & 3 {
            0 => Pass::Opaque,
            1 => Pass::Cutout,
            _ => Pass::Blend,
        }
    }

    /// Decode per-draw scale from biased detail field.
    pub(crate) fn detail_scale(&self) -> f32 {
        Detail::from_gpu_bits((self.detail_pass & 0xF) as u8).scale()
    }

    /// Compose a GPU record from mesh metadata and placement.
    pub(crate) fn compose(meta: &MeshMeta, p: crate::mesh::MeshPlacement) -> Self {
        let (face_quads, flags) = Self::pack_face_quads(&meta.bounds);
        Self {
            block: p.block.to_array(),
            // Detail in bits 0..4, pass in bits 4..6.
            detail_pass: u32::from(p.detail.to_gpu_bits()) | ((meta.pass as u32) << 4),
            local_off: p.local_off.to_array(),
            _pad: 0,
            aabb_min: meta.aabb_min.to_array(),
            index_count: meta.bounds[6],
            aabb_max: meta.aabb_max.to_array(),
            vertex_offset: meta.vertex_offset,
            face_quads,
            flags,
        }
    }

    /// Pack upload-order bucket quad counts as u16 pairs. Overflowing buckets
    /// wrap in the packed words; [`MESH_FLAG_FACE_RUNS`] stays clear so the
    /// cull shader emits a whole-mesh draw instead of corrupted ranges.
    fn pack_face_quads(bounds: &[u32; 7]) -> ([u32; 3], u32) {
        let mut packed = [0u32; 3];
        let mut face_runs = true;
        for (k, slot) in packed.iter_mut().enumerate() {
            let q0 = (bounds[k * 2 + 1] - bounds[k * 2]) / 6;
            let q1 = (bounds[k * 2 + 2] - bounds[k * 2 + 1]) / 6;
            if q0 > u32::from(u16::MAX) || q1 > u32::from(u16::MAX) {
                face_runs = false;
            }
            debug_assert!((q0 <= u32::from(u16::MAX) && q1 <= u32::from(u16::MAX)) || !face_runs);
            *slot = (q0 & u32::from(u16::MAX)) | ((q1 & u32::from(u16::MAX)) << 16);
        }
        (packed, u32::from(face_runs) * MESH_FLAG_FACE_RUNS)
    }
}

/// Persistent record store with deferred writes per frame.
pub(crate) struct RecordTable {
    records: Vec<MeshRecord>,
    dyns: Vec<DrawDyn>,
    gpu: [RecordCopy; FRAMES_IN_FLIGHT as usize],
    /// Incremented when drawable meshes change (signals shadow cache).
    occluder_rev: u64,
}

struct RecordCopy {
    records: HostBuffer,
    dyns: HostBuffer,
    arenas: HostBuffer,
    /// Slots to flush on next upload.
    dirty: Vec<u32>,
}

/// Flushed buffers for descriptor pushes.
#[derive(Clone, Copy)]
pub(crate) struct RecordBuffers {
    pub records: vk::Buffer,
    pub dyns: vk::Buffer,
    pub arenas: vk::Buffer,
    /// Table length in slots.
    pub slots: u32,
}

impl RecordTable {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
            dyns: Vec::new(),
            gpu: std::array::from_fn(|_| RecordCopy {
                records: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                dyns: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                arenas: HostBuffer::new(vk::BufferUsageFlags::STORAGE_BUFFER),
                dirty: Vec::new(),
            }),
            occluder_rev: 0,
        }
    }

    /// The current occluder-set revision, read by the shadow cache each frame.
    pub(crate) fn occluder_rev(&self) -> u64 {
        self.occluder_rev
    }

    fn mark(&mut self, slot: u32) {
        for copy in &mut self.gpu {
            copy.dirty.push(slot);
        }
    }

    /// Install a freshly-uploaded mesh's record.
    pub fn install(&mut self, slot: u32, record: MeshRecord) {
        let n = slot as usize + 1;
        if self.records.len() < n {
            self.records.resize(n, bytemuck::Zeroable::zeroed());
            self.dyns.resize(n, DrawDyn::resting());
        }
        self.records[slot as usize] = record;
        self.dyns[slot as usize] = DrawDyn::resting();
        self.occluder_rev += 1;
        self.mark(slot);
    }

    /// Mark freed slot dirty to read its dead arena word.
    pub fn clear_arena(&mut self, slot: u32) {
        if (slot as usize) < self.records.len() {
            self.occluder_rev += 1;
            self.mark(slot);
        }
    }

    /// Mark arrived slots dirty to re-read their arena words.
    pub fn mark_arrived(&mut self, slots: &[u32]) {
        if !slots.is_empty() {
            self.occluder_rev += 1;
        }
        for &slot in slots {
            self.mark(slot);
        }
    }

    /// Reads a slot's record. `None` for a slot no mesh ever occupied.
    pub fn record(&self, slot: u32) -> Option<&MeshRecord> {
        self.records.get(slot as usize)
    }

    /// Host mirror of every slot's [`MeshRecord`], indexed by slot. The CPU
    /// cull reads this; dead slots are skipped via the directory's arena word.
    pub fn records(&self) -> &[MeshRecord] {
        &self.records
    }

    /// Replaces a mover's record (recomposed main-side); the dyn lane is
    /// untouched so a mover keeps its style.
    pub fn set_record(&mut self, slot: u32, record: MeshRecord) {
        let Some(rec) = self.records.get_mut(slot as usize) else {
            return;
        };
        *rec = record;
        self.occluder_rev += 1; // recomposed geometry moves the occluder
        self.mark(slot);
    }

    /// Patch the dynamic style.
    pub fn set_dyn(&mut self, slot: u32, dyn_lane: DrawDyn) {
        let Some(d) = self.dyns.get_mut(slot as usize) else {
            return;
        };
        *d = dyn_lane;
        self.mark(slot);
    }

    /// Flush slot's pending writes. Must run after fence is waited.
    pub unsafe fn flush(
        &mut self,
        slot: usize,
        dir: &ArenaDirectory,
        mesh_res: &MeshResidency,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Option<RecordBuffers> {
        let copy = &mut self.gpu[slot];
        let rec_bytes: &[u8] = bytemuck::cast_slice(&self.records);
        let dyn_bytes: &[u8] = bytemuck::cast_slice(&self.dyns);
        const ARENA: usize = std::mem::size_of::<u32>();
        let arena_len = (self.records.len() * ARENA) as u64;
        let arena_word = |s: usize| {
            if mesh_res.is_arrived(s as u32) {
                dir.arena_word(s)
            } else {
                0
            }
        };
        unsafe {
            let grew = copy
                .records
                .maintain(instance, device, physical, rec_bytes.len() as u64)
                | copy
                    .dyns
                    .maintain(instance, device, physical, dyn_bytes.len() as u64)
                | copy.arenas.maintain(instance, device, physical, arena_len);
            if rec_bytes.is_empty() {
                return None;
            }
            if grew {
                // Buffer reallocated; rewrite all contents.
                let arena_words: Vec<u32> = (0..self.records.len()).map(arena_word).collect();
                copy.records.write(0, rec_bytes);
                copy.dyns.write(0, dyn_bytes);
                copy.arenas.write(0, bytemuck::cast_slice(&arena_words));
            } else {
                const REC: usize = std::mem::size_of::<MeshRecord>();
                const DYN: usize = std::mem::size_of::<DrawDyn>();
                for &s in &copy.dirty {
                    let s = s as usize;
                    copy.records
                        .write((s * REC) as u64, &rec_bytes[s * REC..(s + 1) * REC]);
                    copy.dyns
                        .write((s * DYN) as u64, &dyn_bytes[s * DYN..(s + 1) * DYN]);
                    copy.arenas
                        .write((s * ARENA) as u64, &arena_word(s).to_ne_bytes());
                }
            }
        }
        copy.dirty.clear();
        Some(RecordBuffers {
            records: copy.records.bound()?,
            dyns: copy.dyns.bound()?,
            arenas: copy.arenas.bound()?,
            slots: self.records.len() as u32,
        })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for copy in &mut self.gpu {
            unsafe {
                copy.records.destroy(device);
                copy.dyns.destroy(device);
                copy.arenas.destroy(device);
            }
        }
    }
}

/// Pod mirror of VkDrawIndexedIndirectCommand for HostBuffer writes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    /// Slot index in the SSBO.
    pub first_instance: u32,
}

// Layout must match `VkDrawIndexedIndirectCommand` field-for-field.
const _: () = {
    use ash::vk::DrawIndexedIndirectCommand as Ash;
    assert!(std::mem::size_of::<DrawIndexedIndirect>() == std::mem::size_of::<Ash>());
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, index_count)
            == std::mem::offset_of!(Ash, index_count)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, instance_count)
            == std::mem::offset_of!(Ash, instance_count)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, first_index)
            == std::mem::offset_of!(Ash, first_index)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, vertex_offset)
            == std::mem::offset_of!(Ash, vertex_offset)
    );
    assert!(
        std::mem::offset_of!(DrawIndexedIndirect, first_instance)
            == std::mem::offset_of!(Ash, first_instance)
    );
};

#[cfg(test)]
mod tests {
    /// Verify detail_pass encoding/decoding is consistent.
    #[test]
    fn compose_then_detail_scale_matches_placement_scale() {
        use super::super::handles::{DrawDyn, MeshMeta, PlacementState};
        use super::{MESH_FLAG_FACE_RUNS, MeshRecord};
        use crate::mesh::{Detail, MeshPlacement};
        for k in -2..=13i8 {
            let detail = Detail(k);
            let meta = MeshMeta {
                aabb_min: glam::Vec3::ZERO,
                aabb_max: glam::Vec3::ONE,
                bounds: [0; 7],
                vertex_offset: 0,
                pass: crate::mesh::Pass::Opaque,
                placement: PlacementState::Pinned,
                dyn_lane: DrawDyn::resting(),
            };
            let p = MeshPlacement::terrain(glam::IVec3::ZERO, detail);
            let rec = MeshRecord::compose(&meta, p);
            assert_eq!(
                rec.detail_scale(),
                detail.scale(),
                "biased detail_pass must decode to the placement's scale (k={k})"
            );
            assert_eq!(rec.pass(), crate::mesh::Pass::Opaque, "pass bits intact");
            assert_eq!(
                rec.flags & MESH_FLAG_FACE_RUNS,
                MESH_FLAG_FACE_RUNS,
                "empty buckets fit in u16 so face-runs stay advertised"
            );
        }
    }

    #[test]
    fn compose_clears_face_runs_when_a_bucket_exceeds_u16() {
        use super::super::handles::{DrawDyn, MeshMeta, PlacementState};
        use super::super::mesh_resident::index_bounds_from_quad_counts;
        use super::{MESH_FLAG_FACE_RUNS, MeshRecord};
        use crate::mesh::{Detail, FACE_UPLOAD_ORDER, MeshPlacement};
        let mut counts = [0u32; 6];
        // Upload slot 2 overflows; the other five directions are empty.
        counts[FACE_UPLOAD_ORDER[2]] = u32::from(u16::MAX) + 1;
        let bounds = index_bounds_from_quad_counts(counts);
        let huge = counts[FACE_UPLOAD_ORDER[2]] * 6;
        let meta = MeshMeta {
            aabb_min: glam::Vec3::ZERO,
            aabb_max: glam::Vec3::ONE,
            bounds,
            vertex_offset: 12,
            pass: crate::mesh::Pass::Opaque,
            placement: PlacementState::Pinned,
            dyn_lane: DrawDyn::resting(),
        };
        let rec = MeshRecord::compose(
            &meta,
            MeshPlacement::terrain(glam::IVec3::ZERO, Detail::FULL),
        );
        assert_eq!(
            rec.flags & MESH_FLAG_FACE_RUNS,
            0,
            "overflow must not advertise packed face-runs"
        );
        assert_eq!(
            rec.index_count, huge,
            "index range stays whole-mesh (bounds[0]..bounds[6])"
        );
        assert_eq!(rec.vertex_offset, 12);
    }

    /// Old upload: vertices in insertion order, six index buckets of the
    /// shared-IBO pattern `[b, b+1, b+2, b, b+2, b+3]`, then one 4-vertex copy
    /// per quad in [`crate::mesh::FACE_UPLOAD_ORDER`].
    fn permute_old(quads: &[[crate::mesh::MeshVertex; 4]]) -> Vec<crate::mesh::MeshVertex> {
        use crate::mesh::FACE_UPLOAD_ORDER;
        let mut vertices = Vec::new();
        let mut buckets: [Vec<u32>; 6] = std::array::from_fn(|_| Vec::new());
        for corners in quads {
            let dir = corners[0].normal() as usize;
            let base = vertices.len() as u32;
            vertices.extend_from_slice(corners);
            buckets[dir].extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        let mut out = Vec::new();
        for &dir in &FACE_UPLOAD_ORDER {
            for quad in buckets[dir].chunks_exact(6) {
                let b = quad[0];
                assert_eq!(
                    *quad,
                    [b, b + 1, b + 2, b, b + 2, b + 3],
                    "reference permuter assumes the shared-IBO pattern"
                );
                out.extend_from_slice(&vertices[b as usize..b as usize + 4]);
            }
        }
        out
    }

    #[test]
    fn upload_layout_matches_old_index_permutation() {
        use super::super::mesh_resident::{
            index_bounds_from_quad_counts, write_vertices_upload_order,
        };
        use super::{MESH_FLAG_FACE_RUNS, MeshRecord};
        use crate::mesh::FACE_UPLOAD_ORDER;
        use crate::mesh::{Ao, Light, MeshData, MeshVertex, Normal, Pass};

        fn tagged_quad(normal: Normal, tag: u8) -> [MeshVertex; 4] {
            std::array::from_fn(|i| {
                MeshVertex::new(
                    [i as u8, tag, 0],
                    normal,
                    u16::from(tag),
                    Ao::NONE,
                    Light::FULL,
                    false,
                )
            })
        }

        // Mixed directions, one empty (NegX), two PosX quads so within-bucket
        // order is visible. Insertion order is not upload order.
        let quads = [
            tagged_quad(Normal::NegY, 1),
            tagged_quad(Normal::PosX, 2),
            tagged_quad(Normal::PosZ, 3),
            tagged_quad(Normal::PosX, 4),
            tagged_quad(Normal::PosY, 5),
            tagged_quad(Normal::NegZ, 6),
        ];
        let mut data = MeshData::new(Pass::Opaque);
        for q in quads {
            data.quad(q);
        }

        let expected = permute_old(&quads);
        assert_eq!(
            expected.len(),
            4 * data.quad_counts().iter().sum::<u32>() as usize,
            "vertex count stays 4 * quads"
        );
        assert_eq!(data.vertices(), expected);

        let mut buf = vec![0u8; data.vertex_bytes()];
        let written = unsafe { write_vertices_upload_order(&data, buf.as_mut_ptr()) };
        assert_eq!(written, data.vertex_bytes());
        let got: &[MeshVertex] = bytemuck::cast_slice(&buf);
        assert_eq!(got, expected.as_slice());

        let counts = data.quad_counts();
        assert_eq!(counts[Normal::PosX as usize], 2);
        assert_eq!(counts[Normal::NegX as usize], 0);
        assert_eq!(counts[Normal::PosY as usize], 1);
        assert_eq!(counts[Normal::NegY as usize], 1);
        assert_eq!(counts[Normal::PosZ as usize], 1);
        assert_eq!(counts[Normal::NegZ as usize], 1);

        let bounds = index_bounds_from_quad_counts(counts);
        // FACE_UPLOAD_ORDER: +X,+Y,+Z,−X,−Y,−Z → 2,1,1,0,1,1 quads.
        assert_eq!(bounds, [0, 12, 18, 24, 24, 30, 36]);
        let (face_quads, flags) = MeshRecord::pack_face_quads(&bounds);
        assert_eq!(flags, MESH_FLAG_FACE_RUNS);
        assert_eq!(
            face_quads,
            [
                counts[FACE_UPLOAD_ORDER[0]] | (counts[FACE_UPLOAD_ORDER[1]] << 16),
                counts[FACE_UPLOAD_ORDER[2]] | (counts[FACE_UPLOAD_ORDER[3]] << 16),
                counts[FACE_UPLOAD_ORDER[4]] | (counts[FACE_UPLOAD_ORDER[5]] << 16),
            ]
        );
    }
}
