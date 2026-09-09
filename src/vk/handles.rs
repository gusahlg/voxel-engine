use std::num::NonZeroU32;

use glam::Vec3;

use crate::mesh::{MeshHandle, Pass};

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
    _marker: std::marker::PhantomData<fn() -> H>,
}

impl<H: GpuHandle, M: Copy> HandleAllocator<H, M> {
    pub fn new() -> Self {
        Self {
            meta: Vec::new(),
            generations: Vec::new(),
            free: Vec::new(),
            live: 0,
            _marker: std::marker::PhantomData,
        }
    }

    #[cfg(test)]
    pub fn live_count(&self) -> usize {
        self.live
    }

    /// Assigns `meta` to a fresh or recycled slot and mints its handle.
    pub fn alloc_slot(&mut self, meta: M) -> H {
        self.live += 1;
        match self.free.pop() {
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
        }
    }

    /// Frees `h`'s slot (gen-checked): bumps the generation and recycles the
    /// slot. Returns false for a stale or double free.
    pub fn free_slot(&mut self, h: H) -> bool {
        let slot = h.slot() as usize;
        let Some(generation) = self.generations.get_mut(slot) else {
            return false;
        };
        if *generation != h.generation() {
            return false;
        }
        if self.meta[slot].take().is_some() {
            *generation = bump_generation(*generation);
            self.free.push(h.slot());
            self.live -= 1;
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
    use super::HandleAllocator;
    use crate::mesh::MeshHandle;

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
    }
}
