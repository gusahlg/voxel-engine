use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use glam::Vec3;

use crate::mesh::{MeshHandle, Pass};

/// Snapshot of mesh-slot and arena occupancy. Cheap to copy; constructed by
/// [`Engine::mesh_stats`](crate::Engine::mesh_stats). `#[non_exhaustive]` so
/// fields can be added without breaking callers.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct MeshStats {
    /// Currently registered mesh slots (main-thread handle table).
    pub live_slots: u32,
    /// Live slots per [`Pass`] (Opaque, Cutout, Blend).
    pub live_per_pass: [u32; 3],
    /// One past the highest registered slot — the value cull loops iterate to.
    pub slot_high_water: u32,
    /// CPU-cull live-count threshold after the `VOXEL_CPU_CULL_MAX` override.
    pub cpu_cull_max: u32,
    /// Occupied arena-directory rows (`refs > 0`).
    pub arenas: u32,
    /// Allocated device-arena capacity in bytes (mesh `GpuAllocator` blocks).
    pub arena_bytes: u64,
    /// Suballocated device-arena bytes currently in use.
    pub arena_bytes_used: u64,
}

/// Published by the handle table / arena directory / mesh allocator, read by
/// [`crate::Engine::mesh_stats`]. Same lock-free pattern as
/// [`super::exposure::ExposureShared`].
#[derive(Clone)]
pub struct MeshStatsShared(Arc<MeshStatsInner>);

struct MeshStatsInner {
    live_slots: AtomicU32,
    live_pass: [AtomicU32; 3],
    slot_high_water: AtomicU32,
    cpu_cull_max: u32,
    arenas: AtomicU32,
    arena_bytes: AtomicU64,
    arena_bytes_used: AtomicU64,
}

impl MeshStatsShared {
    pub(crate) fn new(cpu_cull_max: u32) -> Self {
        crate::profile::mesh_sets(0, cpu_cull_max, 0);
        Self(Arc::new(MeshStatsInner {
            live_slots: AtomicU32::new(0),
            live_pass: [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)],
            slot_high_water: AtomicU32::new(0),
            cpu_cull_max,
            arenas: AtomicU32::new(0),
            arena_bytes: AtomicU64::new(0),
            arena_bytes_used: AtomicU64::new(0),
        }))
    }

    pub(crate) fn store_slots(&self, live: u32, per_pass: [u32; 3], high_water: u32) {
        self.0.live_slots.store(live, Ordering::Relaxed);
        for (dst, src) in self.0.live_pass.iter().zip(per_pass) {
            dst.store(src, Ordering::Relaxed);
        }
        self.0.slot_high_water.store(high_water, Ordering::Relaxed);
        crate::profile::mesh_sets(
            live,
            self.0.cpu_cull_max,
            self.0.arenas.load(Ordering::Relaxed),
        );
    }

    pub(crate) fn store_arenas(&self, arenas: u32) {
        self.0.arenas.store(arenas, Ordering::Relaxed);
        crate::profile::mesh_sets(
            self.0.live_slots.load(Ordering::Relaxed),
            self.0.cpu_cull_max,
            arenas,
        );
    }

    pub(crate) fn store_bytes(&self, cap: u64, used: u64) {
        self.0.arena_bytes.store(cap, Ordering::Relaxed);
        self.0.arena_bytes_used.store(used, Ordering::Relaxed);
    }

    pub fn load(&self) -> MeshStats {
        MeshStats {
            live_slots: self.0.live_slots.load(Ordering::Relaxed),
            live_per_pass: [
                self.0.live_pass[0].load(Ordering::Relaxed),
                self.0.live_pass[1].load(Ordering::Relaxed),
                self.0.live_pass[2].load(Ordering::Relaxed),
            ],
            slot_high_water: self.0.slot_high_water.load(Ordering::Relaxed),
            cpu_cull_max: self.0.cpu_cull_max,
            arenas: self.0.arenas.load(Ordering::Relaxed),
            arena_bytes: self.0.arena_bytes.load(Ordering::Relaxed),
            arena_bytes_used: self.0.arena_bytes_used.load(Ordering::Relaxed),
        }
    }
}

/// Metadata that reports a [`Pass`] so the allocator can keep per-pass counts.
pub(crate) trait SlotPass {
    fn slot_pass(&self) -> Option<Pass>;
}

impl SlotPass for u32 {
    fn slot_pass(&self) -> Option<Pass> {
        None
    }
}

/// Bridge letting [`HandleAllocator`] mint handles generically.
pub(crate) trait GpuHandle: Copy {
    fn from_parts(slot: u32, generation: NonZeroU32) -> Self;
    fn slot(self) -> u32;
    fn generation(self) -> NonZeroU32;
}

impl GpuHandle for MeshHandle {
    fn from_parts(slot: u32, generation: NonZeroU32) -> Self {
        MeshHandle { slot, generation }
    }
    fn slot(self) -> u32 {
        self.slot
    }
    fn generation(self) -> NonZeroU32 {
        self.generation
    }
}

/// Bumps a 1-based generation, skipping the reserved 0 niche on wrap so a
/// recycled slot never reuses a live handle's generation (and never hits 0).
fn bump_generation(g: NonZeroU32) -> NonZeroU32 {
    NonZeroU32::new(g.get().wrapping_add(1)).unwrap_or(NonZeroU32::MIN)
}

/// The single main-thread authority for handle identity + culling metadata.
/// Mints generational handles, recycles freed slots, and answers record-time
/// metadata lookups. Holds NO Vulkan resources — those live render-side in the
/// residency mirror ([`MeshResidency`]).
pub(crate) struct HandleAllocator<H: GpuHandle, M> {
    meta: Vec<Option<M>>,
    /// 1-based; bumped (never to 0) when a slot is freed.
    generations: Vec<NonZeroU32>,
    free: Vec<u32>,
    live: usize,
    live_per_pass: [u32; 3],
    /// One past the highest occupied slot (retreats when the tail is freed).
    high_water: u32,
    stats: Option<MeshStatsShared>,
    _marker: std::marker::PhantomData<fn() -> H>,
}

impl<H: GpuHandle, M: Copy + SlotPass> HandleAllocator<H, M> {
    pub fn new() -> Self {
        Self {
            meta: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            live: 0,
            live_per_pass: [0; 3],
            high_water: 0,
            stats: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn attach_stats(&mut self, stats: MeshStatsShared) {
        self.stats = Some(stats);
        self.publish();
    }

    #[cfg(test)]
    pub fn live_count(&self) -> usize {
        self.live
    }

    #[cfg(test)]
    pub fn live_per_pass(&self) -> [u32; 3] {
        self.live_per_pass
    }

    #[cfg(test)]
    pub fn slot_high_water(&self) -> u32 {
        self.high_water
    }

    fn bump_pass(&mut self, pass: Option<Pass>, delta: i32) {
        let Some(pass) = pass else {
            return;
        };
        let i = pass as usize;
        if delta > 0 {
            self.live_per_pass[i] = self.live_per_pass[i].saturating_add(delta as u32);
        } else {
            self.live_per_pass[i] = self.live_per_pass[i].saturating_sub((-delta) as u32);
        }
    }

    fn publish(&self) {
        if let Some(stats) = &self.stats {
            stats.store_slots(self.live as u32, self.live_per_pass, self.high_water);
        }
    }

    /// Assigns `meta` to a fresh or recycled slot and mints its handle.
    pub fn alloc_slot(&mut self, meta: M) -> H {
        self.live += 1;
        self.bump_pass(meta.slot_pass(), 1);
        let h = match self.free.pop() {
            Some(i) => {
                self.meta[i as usize] = Some(meta);
                H::from_parts(i, self.generations[i as usize])
            }
            None => {
                let i = self.meta.len() as u32;
                self.meta.push(Some(meta));
                self.generations.push(NonZeroU32::MIN);
                H::from_parts(i, NonZeroU32::MIN)
            }
        };
        self.high_water = self.high_water.max(h.slot() + 1);
        self.publish();
        h
    }

    /// Frees `h`'s slot (gen-checked): bumps the generation and recycles the
    /// slot. Returns false for a stale or double free.
    pub fn free_slot(&mut self, h: H) -> bool {
        let slot = h.slot() as usize;
        let Some(&generation) = self.generations.get(slot) else {
            return false;
        };
        if generation != h.generation() {
            return false;
        }
        if let Some(meta) = self.meta[slot].take() {
            self.bump_pass(meta.slot_pass(), -1);
            self.generations[slot] = bump_generation(generation);
            self.free.push(h.slot());
            self.live -= 1;
            if h.slot() + 1 == self.high_water {
                self.high_water = self.meta[..slot]
                    .iter()
                    .rposition(Option::is_some)
                    .map_or(0, |i| i as u32 + 1);
            }
            self.publish();
            true
        } else {
            false
        }
    }

    /// Mutable metadata with generation check.
    pub fn meta_mut(&mut self, h: H) -> Option<&mut M> {
        if *self.generations.get(h.slot() as usize)? != h.generation() {
            return None;
        }
        self.meta.get_mut(h.slot() as usize)?.as_mut()
    }
}

pub(crate) type MeshHandles = HandleAllocator<MeshHandle, MeshMeta>;

/// Main-owned, `Send + Copy` culling/draw metadata for one mesh — NO Vulkan
/// handles. The record path reads this to frustum-cull and embeds the draw
/// params into the snapshot.
#[derive(Clone, Copy)]
pub(crate) struct MeshMeta {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    /// Seven local (0-based) index boundaries into the shared quad IBO:
    /// `bounds[k]..bounds[k+1]` is upload-order face `k`'s range (cumulative
    /// `6*quads`) and `bounds[0]..bounds[6]` the whole mesh.
    /// `bounds[0]` is always 0; always increasing (see [`build_mesh_resident`]).
    pub bounds: [u32; 7],
    /// First vertex (in vertices from block start); the command's `vertex_offset`.
    pub vertex_offset: i32,
    pub pass: Pass,
    /// GPU record placement on the main side.
    pub placement: PlacementState,
    /// GPU dyn lane cache; patched only on change.
    pub dyn_lane: DrawDyn,
}

/// Mesh placement sync strategy: pinned (immutable) or tracked (recovered and patched).
#[derive(Clone, Copy)]
pub(crate) enum PlacementState {
    /// Immutable at upload (terrain).
    Pinned,
    /// Recovered and patched on drift (movers).
    Tracked(Option<crate::mesh::MeshPlacement>),
}

/// Per-mesh dynamic style, patched on change.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawDyn {
    pub mode: u32,
    pub flat_rgba: u32,
}

impl SlotPass for MeshMeta {
    fn slot_pass(&self) -> Option<Pass> {
        Some(self.pass)
    }
}

const _: () = assert!(std::mem::size_of::<DrawDyn>() == 8);

impl DrawDyn {
    /// Rest state: plain textured.
    pub fn resting() -> Self {
        Self {
            mode: 0,
            flat_rgba: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HandleAllocator, SlotPass};
    use crate::mesh::{MeshHandle, Pass};

    #[test]
    fn mesh_handle_option_has_niche() {
        // NonZeroU32 generation gives Option<MeshHandle> a niche → 8 bytes, so
        // the streaming lane's millions of Option<MeshHandle> stay compact.
        assert_eq!(std::mem::size_of::<MeshHandle>(), 8);
        assert_eq!(
            std::mem::size_of::<Option<MeshHandle>>(),
            std::mem::size_of::<MeshHandle>()
        );
    }

    #[test]
    fn handle_allocator_reuses_slot_with_bumped_nonzero_generation() {
        // M = u32 stands in for MeshMeta: this exercises identity only.
        let mut a: HandleAllocator<MeshHandle, u32> = HandleAllocator::new();
        let h0 = a.alloc_slot(10);
        assert_eq!(h0.slot, 0);
        assert_eq!(h0.generation.get(), 1, "generations are 1-based");
        assert_eq!(a.meta_mut(h0).copied(), Some(10));
        assert_eq!(a.live_count(), 1);

        assert!(a.free_slot(h0));
        // A stale handle resolves to nothing after its slot is freed.
        assert_eq!(a.meta_mut(h0).copied(), None);
        // Double free is rejected (generation already moved on).
        assert!(!a.free_slot(h0));
        assert_eq!(a.live_count(), 0);

        // Realloc reuses slot 0 with a bumped, still-nonzero generation.
        let h1 = a.alloc_slot(20);
        assert_eq!(h1.slot, 0);
        assert_eq!(h1.generation.get(), 2);
        assert_eq!(a.meta_mut(h1).copied(), Some(20));
        // The old handle still doesn't alias the reused slot.
        assert_eq!(a.meta_mut(h0).copied(), None);
        assert_eq!(a.live_count(), 1);
        assert_eq!(a.slot_high_water(), 1);
    }

    #[derive(Clone, Copy)]
    struct PassMeta(Pass);

    impl SlotPass for PassMeta {
        fn slot_pass(&self) -> Option<Pass> {
            Some(self.0)
        }
    }

    #[test]
    fn handle_allocator_live_per_pass_and_high_water_stay_exact() {
        let mut a: HandleAllocator<MeshHandle, PassMeta> = HandleAllocator::new();
        let o0 = a.alloc_slot(PassMeta(Pass::Opaque));
        let c0 = a.alloc_slot(PassMeta(Pass::Cutout));
        let b0 = a.alloc_slot(PassMeta(Pass::Blend));
        assert_eq!(a.live_count(), 3);
        assert_eq!(a.live_per_pass(), [1, 1, 1]);
        assert_eq!(a.slot_high_water(), 3);

        assert!(a.free_slot(c0));
        assert_eq!(a.live_count(), 2);
        assert_eq!(a.live_per_pass(), [1, 0, 1]);
        assert_eq!(
            a.slot_high_water(),
            3,
            "hole in the middle does not retreat"
        );

        assert!(a.free_slot(b0));
        assert_eq!(a.live_per_pass(), [1, 0, 0]);
        assert_eq!(a.slot_high_water(), 1, "tail free retreats to live end");

        // Regenerate slot 0: free then alloc — counts stay exact.
        assert!(a.free_slot(o0));
        assert_eq!(a.live_count(), 0);
        assert_eq!(a.live_per_pass(), [0, 0, 0]);
        assert_eq!(a.slot_high_water(), 0);
        let o1 = a.alloc_slot(PassMeta(Pass::Opaque));
        assert_eq!(o1.slot, 0);
        assert_eq!(a.live_count(), 1);
        assert_eq!(a.live_per_pass(), [1, 0, 0]);
        assert_eq!(a.slot_high_water(), 1);
        assert!(!a.free_slot(o0), "stale generation is not a live slot");
        assert_eq!(a.live_count(), 1);
        assert_eq!(a.live_per_pass(), [1, 0, 0]);
    }
}
