use std::num::NonZeroU32;

use ash::vk;
use glam::Vec3;

use super::buffers::{MESH_FLAG_FACE_RUNS, MeshRecord};
use super::cull_math::{
    BUCKETS, CULL_BUCKET_SPLITS, Group, LANES, PartitionGpu, SHADOW_GROUPS, partition_count,
};
use super::pipeline::EyeSplit;
use crate::mesh::Pass;

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
/// the union must be rebuilt from the arena's member list before it is queried.
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

/// Packed CPU-cull bits: arena word in [0, 20), pass in [20, 22), LOD in bit 22,
/// face-runs flag in bit 23. Zero means the slot is dead to the CPU cull
/// (freed or Blend).
const CULL_ARENA_MASK: u32 = 0x000F_FFFF;
const CULL_PASS_SHIFT: u32 = 20;
const CULL_LOD_BIT: u32 = 1 << 22;
const CULL_FACE_BIT: u32 = 1 << 23;

#[inline]
fn pack_cull_bits(arena_word: u32, pass: Pass, lod: bool) -> u32 {
    debug_assert!(arena_word != 0 && arena_word <= CULL_ARENA_MASK);
    let pass_u = pass as u32;
    (arena_word & CULL_ARENA_MASK)
        | ((pass_u & 3) << CULL_PASS_SHIFT)
        | (u32::from(lod) * CULL_LOD_BIT)
}

#[inline]
pub(crate) fn cull_bits_arena(bits: u32) -> u32 {
    bits & CULL_ARENA_MASK
}

#[inline]
pub(crate) fn cull_bits_pass(bits: u32) -> u32 {
    (bits >> CULL_PASS_SHIFT) & 3
}

#[inline]
pub(crate) fn cull_bits_lod(bits: u32) -> bool {
    bits & CULL_LOD_BIT != 0
}

#[inline]
pub(crate) fn cull_bits_face(bits: u32) -> bool {
    bits & CULL_FACE_BIT != 0
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
    /// CPU-cull SoA: block-relative AABB `[min.xyz, max.xyz]`, parallel to `slots`.
    cull_aabb: Vec<[f32; 6]>,
    /// CPU-cull SoA: integer block of each slot's AABB, parallel to `slots`.
    cull_block: Vec<[i32; 3]>,
    /// CPU-cull SoA: packed arena/pass/lod (0 = dead to cull). Parallel to `slots`.
    cull_bits: Vec<u32>,
    /// CPU-cull SoA: draw index count, parallel to `slots`.
    cull_index_count: Vec<u32>,
    /// CPU-cull SoA: vertex offset, parallel to `slots`.
    cull_vertex_offset: Vec<i32>,
    /// CPU-cull SoA: packed face-quad counts, parallel to `slots`.
    cull_face_quads: Vec<[u32; 3]>,
    /// Per-slot live-to-cull bitset (camera-group slots only). Rebuilt into the
    /// per-frame live+visible mask; maintained on register/free.
    live_bits: Vec<u32>,
    /// Conservative union of live camera-group AABBs per arena.
    unions: Vec<ArenaUnion>,
    /// Camera-group slots per arena (unordered). Union recompute walks these
    /// instead of the whole slot table.
    members: Vec<Vec<u32>>,
    /// Position + 1 of a slot in its arena's `members` (0 = not a member).
    member_pos: Vec<u32>,
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
    /// Directory rows with `refs > 0`.
    occupied: u32,
    stats: Option<super::handles::MeshStatsShared>,
}

impl ArenaDirectory {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            live: Vec::new(),
            refs: Vec::new(),
            slots: Vec::new(),
            aabbs: Vec::new(),
            cull_aabb: Vec::new(),
            cull_block: Vec::new(),
            cull_bits: Vec::new(),
            cull_index_count: Vec::new(),
            cull_vertex_offset: Vec::new(),
            cull_face_quads: Vec::new(),
            live_bits: Vec::new(),
            unions: Vec::new(),
            members: Vec::new(),
            member_pos: Vec::new(),
            blend: Vec::new(),
            blend_pos: Vec::new(),
            live_end: 0,
            occupied: 0,
            stats: None,
        }
    }

    pub fn attach_stats(&mut self, stats: super::handles::MeshStatsShared) {
        self.stats = Some(stats);
        self.publish();
    }

    fn publish(&self) {
        if let Some(stats) = &self.stats {
            stats.store_arenas(self.occupied);
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
        if let Some(Some((old_arena, old_lane, _))) = self.slots.get(slot as usize).copied() {
            // Re-register without a free: drop the old box out of the union.
            if old_lane.is_some() {
                self.set_member(old_arena as usize, slot, false);
            }
            self.mark_union_dirty(old_arena as usize);
        }
        let hit = (0..self.buffers.len())
            .find(|&i| self.refs[i] > 0 && self.buffers[i] == buffer)
            .or_else(|| {
                let reuse = self.refs.iter().position(|&r| r == 0);
                if let Some(i) = reuse {
                    self.buffers[i] = buffer;
                    debug_assert_eq!(self.live[i], [0; LANES], "drained row kept live counts");
                    debug_assert!(
                        self.members[i].is_empty(),
                        "drained row kept camera-group members"
                    );
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
                self.members.push(Vec::new());
                (self.buffers.len() - 1) as u32
            }
        };
        self.refs[arena as usize] += 1;
        if self.refs[arena as usize] == 1 {
            self.occupied += 1;
        }
        let lane = group_lane(pass, lod);
        if let Some(lane) = lane {
            self.live[arena as usize][lane] += 1;
        }
        let n = slot as usize + 1;
        if self.slots.len() < n {
            self.slots.resize(n, None);
            self.aabbs.resize(n, MeshAabb::ZERO);
            self.cull_aabb.resize(n, [0.0; 6]);
            self.cull_block.resize(n, [0; 3]);
            self.cull_bits.resize(n, 0);
            self.cull_index_count.resize(n, 0);
            self.cull_vertex_offset.resize(n, 0);
            self.cull_face_quads.resize(n, [0; 3]);
            self.live_bits.resize(n.div_ceil(32), 0);
        }
        self.slots[slot as usize] = Some((arena, lane, generation));
        self.aabbs[slot as usize] = aabb;
        self.write_cull_soa(slot, arena, lane, pass, lod, aabb);
        if lane.is_some() {
            self.set_member(arena as usize, slot, true);
            self.grow_union(arena as usize, aabb);
        }
        self.set_blend(slot, pass == Pass::Blend);
        self.live_end = self.live_end.max(slot + 1);
        self.publish();
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

    /// Adds `slot` to (or removes it from) `arena`'s camera-group member list;
    /// swap-remove, same scheme as [`Self::set_blend`].
    fn set_member(&mut self, arena: usize, slot: u32, on: bool) {
        let i = slot as usize;
        if self.member_pos.len() <= i {
            self.member_pos.resize(i + 1, 0);
        }
        let pos = self.member_pos[i];
        if on && pos == 0 {
            self.members[arena].push(slot);
            self.member_pos[i] = self.members[arena].len() as u32;
        } else if !on && pos != 0 {
            let at = (pos - 1) as usize;
            self.members[arena].swap_remove(at);
            self.member_pos[i] = 0;
            if let Some(&moved) = self.members[arena].get(at) {
                self.member_pos[moved as usize] = pos;
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
            match (lane.is_some(), new_lane.is_some()) {
                (true, false) => {
                    self.set_member(arena as usize, slot, false);
                    self.mark_union_dirty(arena as usize);
                }
                (false, true) => self.set_member(arena as usize, slot, true),
                _ => {}
            }
        }
        self.aabbs[slot as usize] = aabb;
        self.write_cull_soa(slot, arena, new_lane, pass, lod, aabb);
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
        if self.refs[arena as usize] == 0 {
            self.occupied = self.occupied.saturating_sub(1);
        }
        if let Some(lane) = lane {
            self.live[arena as usize][lane] -= 1;
            self.set_member(arena as usize, slot, false);
        }
        if self.refs[arena as usize] == 0 || self.members[arena as usize].is_empty() {
            self.unions[arena as usize] = ArenaUnion::EMPTY;
        } else if lane.is_some() {
            self.mark_union_dirty(arena as usize);
        }
        self.set_blend(slot, false);
        self.clear_cull_soa(slot);
        if slot + 1 == self.live_end {
            // The tail died: retreat to the next registered slot. Amortised
            // O(1) — each dead slot is stepped over once per retreat.
            self.live_end = self.slots[..slot as usize]
                .iter()
                .rposition(Option::is_some)
                .map_or(0, |i| i as u32 + 1);
        }
        self.publish();
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

    /// Block-relative AABB `[min.xyz, max.xyz]` for the CPU cull hot loop.
    pub(crate) fn cull_aabbs(&self) -> &[[f32; 6]] {
        &self.cull_aabb
    }

    /// Integer block of each slot's AABB, parallel to [`Self::cull_aabbs`].
    pub(crate) fn cull_blocks(&self) -> &[[i32; 3]] {
        &self.cull_block
    }

    /// Packed arena/pass/lod bits, parallel to [`Self::cull_aabbs`].
    pub(crate) fn cull_bits(&self) -> &[u32] {
        &self.cull_bits
    }

    /// Draw index counts, parallel to [`Self::cull_aabbs`].
    pub(crate) fn cull_index_counts(&self) -> &[u32] {
        &self.cull_index_count
    }

    /// Vertex offsets, parallel to [`Self::cull_aabbs`].
    pub(crate) fn cull_vertex_offsets(&self) -> &[i32] {
        &self.cull_vertex_offset
    }

    /// Packed face-quad counts, parallel to [`Self::cull_aabbs`].
    pub(crate) fn cull_face_quads(&self) -> &[[u32; 3]] {
        &self.cull_face_quads
    }

    /// Camera-group live bitset, maintained on register/free.
    pub(crate) fn live_bits(&self) -> &[u32] {
        &self.live_bits
    }

    /// Copies draw-emission fields from a mesh record into the CPU-cull SoA.
    /// Call after [`Self::note_upload`] / [`Self::note_record`].
    pub(crate) fn note_cull_draw(&mut self, slot: u32, rec: &MeshRecord) {
        let i = slot as usize;
        if i >= self.cull_index_count.len() {
            return;
        }
        self.cull_index_count[i] = rec.index_count;
        self.cull_vertex_offset[i] = rec.vertex_offset;
        self.cull_face_quads[i] = rec.face_quads;
        if rec.flags & MESH_FLAG_FACE_RUNS != 0 {
            self.cull_bits[i] |= CULL_FACE_BIT;
        } else {
            self.cull_bits[i] &= !CULL_FACE_BIT;
        }
    }

    fn write_cull_soa(
        &mut self,
        slot: u32,
        arena: u32,
        lane: Option<usize>,
        pass: Pass,
        lod: bool,
        aabb: MeshAabb,
    ) {
        let i = slot as usize;
        self.cull_aabb[i] = [
            aabb.min[0],
            aabb.min[1],
            aabb.min[2],
            aabb.max[0],
            aabb.max[1],
            aabb.max[2],
        ];
        self.cull_block[i] = aabb.block;
        // Blend (no lane) is dead to the CPU cull: the shader skips pass > 1.
        // Preserve the face-runs bit; [`Self::note_cull_draw`] sets it.
        let face = self.cull_bits[i] & CULL_FACE_BIT;
        self.cull_bits[i] = if lane.is_some() {
            pack_cull_bits(arena + 1, pass, lod) | face
        } else {
            0
        };
        self.set_live_bit(slot, lane.is_some());
    }

    fn clear_cull_soa(&mut self, slot: u32) {
        let i = slot as usize;
        if i < self.cull_bits.len() {
            self.cull_bits[i] = 0;
        }
        self.set_live_bit(slot, false);
    }

    fn set_live_bit(&mut self, slot: u32, on: bool) {
        let i = (slot >> 5) as usize;
        let b = 1u32 << (slot & 31);
        if self.live_bits.len() <= i {
            self.live_bits.resize(i + 1, 0);
        }
        if on {
            self.live_bits[i] |= b;
        } else {
            self.live_bits[i] &= !b;
        }
    }

    pub fn arena_count(&self) -> usize {
        self.buffers.len()
    }

    #[cfg(test)]
    pub fn occupied_arenas(&self) -> u32 {
        self.occupied
    }

    /// Live camera-group records (Opaque + Cutout + OpaqueLod) across every
    /// arena. Blend is excluded. Used to choose the CPU cull path.
    pub(crate) fn camera_live(&self) -> u32 {
        self.live.iter().map(|l| l.iter().sum::<u32>()).sum()
    }

    fn mark_union_dirty(&mut self, arena: usize) {
        if let Some(u) = self.unions.get_mut(arena) {
            u.dirty = true;
        }
    }

    fn grow_union(&mut self, arena: usize, aabb: MeshAabb) {
        // Expand even when dirty: a fresh upload can sit outside the stale
        // box (farther than, or off-axis from, the freed contributor). A
        // later query must not zero that mesh's bucket before recompute.
        // Restore dirty afterwards: `expand` of an invalid union goes through
        // `from_aabb`, which writes dirty=false and would skip the member-list
        // rebuild while other slots still contribute.
        let u = &mut self.unions[arena];
        let was_dirty = u.dirty;
        u.expand(aabb);
        if was_dirty {
            u.dirty = true;
        }
    }

    fn recompute_union(&mut self, arena: usize) {
        let mut acc = ArenaUnion::EMPTY;
        for i in 0..self.members[arena].len() {
            let slot = self.members[arena][i] as usize;
            acc.expand(self.aabbs[slot]);
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
    pub(crate) fn partitions_into(
        &mut self,
        parts: &mut Vec<PartitionGpu>,
        runs_per_mesh: u32,
        eye: Option<EyeSplit>,
    ) -> u32 {
        let a = self.live.len();
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
                        Some(e) => {
                            // Dirty unions are rebuilt once per frame, and only
                            // when a camera partition will query them.
                            if self.unions[arena].dirty {
                                self.recompute_union(arena);
                            }
                            if self.bucket_reachable(arena, bucket, e) {
                                full
                            } else {
                                0
                            }
                        }
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

/// Get live-count lane for a (pass, lod) record (Blend returns None).
fn group_lane(pass: Pass, lod: bool) -> Option<usize> {
    match pass {
        Pass::Opaque if lod => Some(2),
        Pass::Opaque => Some(0),
        Pass::Cutout => Some(1),
        Pass::Blend => None,
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::super::cull_math::{
        CAMERA_GROUPS, GROUPS, camera_part, group_indirect_calls, shadow_part,
    };
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
        assert_eq!(dir.occupied_arenas(), 1);
        assert_eq!(dir.live_end(), 2);
        dir.note_free(0, G1);
        assert_eq!(dir.occupied_arenas(), 1);
        assert_eq!(dir.live_end(), 2);
        let (parts, total) = dir.partitions();
        assert_eq!(parts[camera_part(0, 0, 0, 1)].capacity, 1);
        // K camera slots + 2 shadow slots, each sized off the remaining live count.
        assert_eq!(total, BUCKETS as u32 + 2);
    }

    #[test]
    fn note_free_on_last_reference_drains_the_arena_row_for_reuse() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        assert_eq!(dir.occupied_arenas(), 1);
        dir.note_free(0, G1);
        assert_eq!(dir.occupied_arenas(), 0);
        assert_eq!(dir.live_end(), 0);
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
        assert_eq!(dir.occupied_arenas(), 0);
        dir.note_upload(0, genr(2), buf(2), Pass::Cutout, FULL, UNIT); // reused, new generation
        assert_eq!(dir.occupied_arenas(), 1);
        assert_eq!(dir.live_end(), 1);
        dir.note_free(0, genr(1)); // stale duplicate: must be ignored
        assert_eq!(dir.occupied_arenas(), 1);
        assert_eq!(dir.live_end(), 1);
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
        assert_eq!(group_indirect_calls(&parts, Group::Opaque, 1), 1);
        assert_eq!(group_indirect_calls(&parts, Group::Cutout, 1), 0);
        assert_eq!(group_indirect_calls(&parts, Group::OpaqueLod, 1), 0);
    }

    #[test]
    fn group_indirect_calls_counts_nonzero_capacity_partitions() {
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(2), Pass::Opaque, LOD, UNIT);
        let (parts, _) = partitions_at(&mut dir, origin_eye());
        // Two arenas: near full-res + near LOD. Each keeps bucket 0 only.
        assert_eq!(dir.arena_count(), 2);
        assert_eq!(group_indirect_calls(&parts, Group::Opaque, 2), 1);
        assert_eq!(group_indirect_calls(&parts, Group::OpaqueLod, 2), 1);
        assert_eq!(group_indirect_calls(&parts, Group::Cutout, 2), 0);
        // Empty table / zero arenas.
        assert_eq!(group_indirect_calls(&[], Group::Opaque, 0), 0);
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
    fn upload_after_free_in_the_same_arena_is_not_culled_by_a_stale_union() {
        // Free dirties the union; grow_union used to no-op while dirty, so a
        // later upload outside the remaining box could have its buckets
        // zeroed. Recompute must walk this arena's members, including the
        // new slot — not a full slot-table scan and not the stale box.
        let far = MeshAabb {
            block: [0, 0, 300],
            min: [0.0; 3],
            max: [1.0; 3],
        };
        let mut dir = ArenaDirectory::new();
        dir.note_upload(0, G1, buf(1), Pass::Opaque, FULL, UNIT);
        dir.note_upload(1, G1, buf(1), Pass::Opaque, FULL, far);
        dir.note_free(1, G1);
        dir.note_upload(2, G1, buf(1), Pass::Opaque, FULL, far);
        let (parts, _) = partitions_at(&mut dir, origin_eye());
        // Two live meshes; missing the new slot would zero far buckets.
        assert_eq!(opaque_caps(&parts, 0, 1), [2, 2, 2, 2]);
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
}
