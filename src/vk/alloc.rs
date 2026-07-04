//! Block-suballocating GPU buffer allocator.
//!
//! One `vkAllocateMemory` per mesh would exhaust `maxMemoryAllocationCount`
//! with per-chunk voxel meshes, so this allocator owns a small number of
//! large blocks (one `vk::Buffer` + one `vk::DeviceMemory` each) and hands
//! out sub-ranges via a first-fit free list.
//!
//! Two pools:
//! - DEVICE: vertex/index/transfer-dst buffers in DEVICE_LOCAL memory,
//!   preferring a type that is also HOST_VISIBLE | HOST_COHERENT (unified
//!   memory, e.g. Apple Silicon) so mesh data can be written directly.
//! - STAGING: transfer-src buffers in HOST_VISIBLE | HOST_COHERENT memory.
//!
//! Host-visible blocks are mapped once at creation and never unmapped;
//! `Allocation::mapped` points at the allocation's first byte.

use std::ptr::NonNull;

use ash::vk;

/// Default block size; larger allocations get a dedicated, larger block.
const BLOCK_SIZE: u64 = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// FreeList — pure offset/size suballocator, no Vulkan types (unit-testable).
// ---------------------------------------------------------------------------

/// A contiguous free byte range inside a block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FreeRange {
    offset: u64,
    size: u64,
}

/// Sorted free ranges, first-fit with alignment, split on alloc, coalesce
/// adjacent ranges on free.
struct FreeList {
    capacity: u64,
    used: u64,
    /// Sorted by offset; adjacent ranges are always coalesced.
    free: Vec<FreeRange>,
}

impl FreeList {
    fn new(capacity: u64) -> Self {
        Self {
            capacity,
            used: 0,
            free: vec![FreeRange { offset: 0, size: capacity }],
        }
    }

    fn used(&self) -> u64 {
        self.used
    }

    /// First-fit allocation of `size` bytes at a multiple of `align`.
    /// Returns the offset, or `None` if no hole is large enough.
    fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        debug_assert!(size > 0, "zero-size suballocation");
        let size = size.max(1);
        let align = align.max(1);
        for i in 0..self.free.len() {
            let range = self.free[i];
            let aligned = range.offset.next_multiple_of(align);
            let pad = aligned - range.offset;
            if range.size < pad || range.size - pad < size {
                continue;
            }
            let tail = range.size - pad - size;
            // Alignment padding stays on the free list so no bytes are lost.
            match (pad > 0, tail > 0) {
                (false, false) => {
                    self.free.remove(i);
                }
                (true, false) => self.free[i].size = pad,
                (false, true) => {
                    self.free[i] = FreeRange { offset: aligned + size, size: tail };
                }
                (true, true) => {
                    self.free[i].size = pad;
                    self.free
                        .insert(i + 1, FreeRange { offset: aligned + size, size: tail });
                }
            }
            self.used += size;
            return Some(aligned);
        }
        None
    }

    /// Returns a previously allocated range, coalescing with any adjacent
    /// free neighbours.
    fn free(&mut self, offset: u64, size: u64) {
        debug_assert!(size > 0, "zero-size free");
        let size = size.max(1);
        debug_assert!(offset + size <= self.capacity, "free out of bounds");
        debug_assert!(self.used >= size, "free exceeds used bytes");
        let idx = self.free.partition_point(|r| r.offset < offset);
        let merges_prev =
            idx > 0 && self.free[idx - 1].offset + self.free[idx - 1].size == offset;
        let merges_next = idx < self.free.len() && offset + size == self.free[idx].offset;
        match (merges_prev, merges_next) {
            (true, true) => {
                let next = self.free.remove(idx);
                self.free[idx - 1].size += size + next.size;
            }
            (true, false) => self.free[idx - 1].size += size,
            (false, true) => {
                self.free[idx].offset = offset;
                self.free[idx].size += size;
            }
            (false, false) => self.free.insert(idx, FreeRange { offset, size }),
        }
        self.used = self.used.saturating_sub(size);
    }
}

// ---------------------------------------------------------------------------
// Vulkan layer
// ---------------------------------------------------------------------------

/// Identifies which pool an allocation came from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Pool {
    Device,
    Staging,
}

/// A sub-range of a pooled buffer. Bind `buffer` with `offset`; `mapped`
/// points at this allocation's bytes when the block is host-visible.
///
/// Deliberately neither `Copy` nor `Clone`: `GpuAllocator::free` consumes it,
/// which makes double-frees a compile error.
#[derive(Debug)]
pub struct Allocation {
    /// The owning block's buffer — bind with `offset`.
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub size: u64,
    /// This allocation's bytes when the block memory is host-visible.
    pub mapped: Option<NonNull<u8>>,
    block: usize,
    pool: Pool,
}

/// One `vk::Buffer` + `vk::DeviceMemory` pair, suballocated by a `FreeList`.
struct Block {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Actual `vkAllocateMemory` size (>= the buffer size).
    memory_size: u64,
    /// Base pointer of the persistent mapping, when host-visible.
    mapped: Option<NonNull<u8>>,
    free_list: FreeList,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct AllocatorStats {
    pub device_blocks: usize,
    pub device_reserved: u64,
    pub device_used: u64,
    pub staging_blocks: usize,
    pub staging_reserved: u64,
    pub staging_used: u64,
}

pub struct GpuAllocator {
    memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Candidate memory type indices for the device pool, best first.
    device_type_prefs: Vec<u32>,
    /// Candidate memory type indices for the staging pool, best first.
    staging_type_prefs: Vec<u32>,
    /// Whether the device pool's preferred type is host-visible.
    device_host_visible: bool,
    device_blocks: Vec<Block>,
    staging_blocks: Vec<Block>,
}

impl GpuAllocator {
    /// Captures memory properties; performs no allocations yet.
    pub unsafe fn new(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Self {
        let memory_props =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let types_with = |flags: vk::MemoryPropertyFlags| -> Vec<u32> {
            (0..memory_props.memory_type_count)
                .filter(|&i| {
                    memory_props.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
                })
                .collect()
        };

        let unified = vk::MemoryPropertyFlags::DEVICE_LOCAL
            | vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT;
        let host_coherent =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        // Device pool: unified-memory types first, then plain DEVICE_LOCAL.
        let mut device_type_prefs = types_with(unified);
        for i in types_with(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            if !device_type_prefs.contains(&i) {
                device_type_prefs.push(i);
            }
        }
        let staging_type_prefs = types_with(host_coherent);

        let device_host_visible = device_type_prefs.first().is_some_and(|&i| {
            memory_props.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        });

        if device_type_prefs.is_empty() {
            // Spec guarantees a DEVICE_LOCAL type exists, but stay graceful:
            // block creation will fall back to any host-visible type.
            log::warn!("no DEVICE_LOCAL memory type reported by the driver");
        }
        if staging_type_prefs.is_empty() {
            log::warn!("no HOST_VISIBLE|HOST_COHERENT memory type reported by the driver");
        }

        Self {
            memory_props,
            device_type_prefs,
            staging_type_prefs,
            device_host_visible,
            device_blocks: Vec::new(),
            staging_blocks: Vec::new(),
        }
    }

    /// True when the DEVICE pool's memory type is HOST_VISIBLE (unified
    /// memory): mesh data can be written through `Allocation::mapped`
    /// directly, no staging copy needed.
    pub fn unified_memory(&self) -> bool {
        self.device_host_visible
    }

    /// Suballocates from the DEVICE pool (vertex/index/transfer-dst usage).
    pub unsafe fn alloc_device(
        &mut self,
        device: &ash::Device,
        size: u64,
        align: u64,
    ) -> Result<Allocation, vk::Result> {
        let usage = vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST;
        unsafe {
            alloc_from_pool(
                &mut self.device_blocks,
                &self.device_type_prefs,
                &self.memory_props,
                usage,
                Pool::Device,
                device,
                size,
                align,
            )
        }
    }

    /// Suballocates from the STAGING pool (transfer-src usage, host-visible).
    pub unsafe fn alloc_staging(
        &mut self,
        device: &ash::Device,
        size: u64,
        align: u64,
    ) -> Result<Allocation, vk::Result> {
        unsafe {
            alloc_from_pool(
                &mut self.staging_blocks,
                &self.staging_type_prefs,
                &self.memory_props,
                vk::BufferUsageFlags::TRANSFER_SRC,
                Pool::Staging,
                device,
                size,
                align,
            )
        }
    }

    /// Returns the range to its block's free list. Empty blocks are kept —
    /// chunk churn reuses them almost immediately.
    pub unsafe fn free(&mut self, allocation: Allocation) {
        let blocks = match allocation.pool {
            Pool::Device => &mut self.device_blocks,
            Pool::Staging => &mut self.staging_blocks,
        };
        let block = blocks
            .get_mut(allocation.block)
            .expect("allocation refers to an unknown block (freed after destroy?)");
        debug_assert_eq!(
            block.buffer, allocation.buffer,
            "allocation/block buffer mismatch"
        );
        block.free_list.free(allocation.offset, allocation.size);
    }

    /// Destroy all blocks. Caller guarantees the GPU is idle.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for block in self
            .device_blocks
            .drain(..)
            .chain(self.staging_blocks.drain(..))
        {
            // Freeing the memory implicitly unmaps the persistent mapping.
            unsafe {
                device.destroy_buffer(block.buffer, None);
                device.free_memory(block.memory, None);
            }
        }
    }

    pub fn stats(&self) -> AllocatorStats {
        let mut stats = AllocatorStats::default();
        for block in &self.device_blocks {
            stats.device_blocks += 1;
            stats.device_reserved += block.memory_size;
            stats.device_used += block.free_list.used();
        }
        for block in &self.staging_blocks {
            stats.staging_blocks += 1;
            stats.staging_reserved += block.memory_size;
            stats.staging_used += block.free_list.used();
        }
        stats
    }
}

/// Allocates from `blocks`, creating a new block when no existing block has
/// a large enough hole.
#[allow(clippy::too_many_arguments)]
unsafe fn alloc_from_pool(
    blocks: &mut Vec<Block>,
    type_prefs: &[u32],
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    usage: vk::BufferUsageFlags,
    pool: Pool,
    device: &ash::Device,
    size: u64,
    align: u64,
) -> Result<Allocation, vk::Result> {
    debug_assert!(size > 0, "zero-size GPU allocation");
    let size = size.max(1);

    for (index, block) in blocks.iter_mut().enumerate() {
        if let Some(offset) = block.free_list.alloc(size, align) {
            return Ok(make_allocation(block, index, offset, size, pool));
        }
    }

    let block = unsafe { create_block(device, memory_props, type_prefs, usage, size)? };
    blocks.push(block);
    let index = blocks.len() - 1;
    let block = &mut blocks[index];
    let offset = block
        .free_list
        .alloc(size, align)
        .expect("fresh block must fit the allocation that sized it");
    Ok(make_allocation(block, index, offset, size, pool))
}

fn make_allocation(block: &Block, index: usize, offset: u64, size: u64, pool: Pool) -> Allocation {
    let mapped = block.mapped.map(|base| {
        // SAFETY: `offset + size` fits in the block, whose whole range is
        // mapped, so the derived pointer is in bounds and non-null.
        unsafe { NonNull::new_unchecked(base.as_ptr().add(offset as usize)) }
    });
    Allocation {
        buffer: block.buffer,
        offset,
        size,
        mapped,
        block: index,
        pool,
    }
}

/// Creates a block of at least `min_size` bytes (rounded up to BLOCK_SIZE),
/// bound at offset 0 and persistently mapped when the chosen memory type is
/// host-visible.
unsafe fn create_block(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_prefs: &[u32],
    usage: vk::BufferUsageFlags,
    min_size: u64,
) -> Result<Block, vk::Result> {
    let buffer_size = min_size.max(BLOCK_SIZE);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let Some(type_index) = pick_memory_type(memory_props, type_prefs, requirements.memory_type_bits)
    else {
        unsafe { device.destroy_buffer(buffer, None) };
        log::error!(
            "no compatible memory type for pool buffer (memory_type_bits = {:#b})",
            requirements.memory_type_bits
        );
        return Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY);
    };
    let host_visible = memory_props.memory_types[type_index as usize]
        .property_flags
        .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    let memory = match unsafe { device.allocate_memory(&alloc_info, None) } {
        Ok(memory) => memory,
        Err(err) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(err);
        }
    };

    if let Err(err) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
        }
        return Err(err);
    }

    // Map once for the block's lifetime; never unmapped.
    let mapped = if host_visible {
        match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
        {
            Ok(ptr) => NonNull::new(ptr.cast::<u8>()),
            Err(err) => {
                unsafe {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                }
                return Err(err);
            }
        }
    } else {
        None
    };

    log::debug!(
        "created {} MiB gpu block (memory type {type_index}, host_visible = {host_visible})",
        requirements.size / (1024 * 1024),
    );

    Ok(Block {
        buffer,
        memory,
        memory_size: requirements.size,
        mapped,
        // Suballocate only within the buffer's extent, not the (possibly
        // slightly larger) memory allocation.
        free_list: FreeList::new(buffer_size),
    })
}

/// Picks the first preferred type allowed by `memory_type_bits`, falling
/// back to any compatible HOST_VISIBLE type.
fn pick_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_prefs: &[u32],
    memory_type_bits: u32,
) -> Option<u32> {
    for &i in type_prefs {
        if memory_type_bits & (1 << i) != 0 {
            return Some(i);
        }
    }
    let fallback = (0..memory_props.memory_type_count).find(|&i| {
        memory_type_bits & (1 << i) != 0
            && memory_props.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    });
    if let Some(i) = fallback {
        log::warn!("preferred memory types unavailable; falling back to host-visible type {i}");
    }
    fallback
}

// ---------------------------------------------------------------------------
// Tests (FreeList only — no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{FreeList, FreeRange};

    #[test]
    fn alloc_free_roundtrip() {
        let mut fl = FreeList::new(1024);
        let a = fl.alloc(100, 1).expect("fits");
        assert_eq!(a, 0);
        assert_eq!(fl.used(), 100);
        fl.free(a, 100);
        assert_eq!(fl.used(), 0);
        assert_eq!(fl.free, vec![FreeRange { offset: 0, size: 1024 }]);
        // The full capacity is allocatable again.
        assert_eq!(fl.alloc(1024, 1), Some(0));
    }

    #[test]
    fn alignment_rounds_up() {
        let mut fl = FreeList::new(1024);
        assert_eq!(fl.alloc(10, 256), Some(0)); // offset 0 satisfies any align
        assert_eq!(fl.alloc(10, 256), Some(256));
        assert_eq!(fl.alloc(10, 256), Some(512));
        // Padding is kept free, not counted as used.
        assert_eq!(fl.used(), 30);
    }

    #[test]
    fn first_fit_picks_earliest_hole() {
        let mut fl = FreeList::new(1024);
        let a = fl.alloc(100, 1).unwrap(); // [0, 100)
        let _b = fl.alloc(100, 1).unwrap(); // [100, 200)
        let c = fl.alloc(100, 1).unwrap(); // [200, 300)
        fl.free(a, 100);
        fl.free(c, 100); // holes: [0, 100) and [200, 1024)
        assert_eq!(fl.alloc(50, 1), Some(0)); // earliest sufficient hole wins
    }

    #[test]
    fn coalesce_left() {
        let mut fl = FreeList::new(300);
        let a = fl.alloc(100, 1).unwrap();
        let b = fl.alloc(100, 1).unwrap();
        let _c = fl.alloc(100, 1).unwrap();
        fl.free(a, 100); // [0, 100)
        fl.free(b, 100); // merges into the left neighbour
        assert_eq!(fl.free, vec![FreeRange { offset: 0, size: 200 }]);
    }

    #[test]
    fn coalesce_right() {
        let mut fl = FreeList::new(300);
        let a = fl.alloc(100, 1).unwrap();
        let b = fl.alloc(100, 1).unwrap();
        let _c = fl.alloc(100, 1).unwrap();
        fl.free(b, 100); // [100, 200)
        fl.free(a, 100); // merges into the right neighbour
        assert_eq!(fl.free, vec![FreeRange { offset: 0, size: 200 }]);
    }

    #[test]
    fn coalesce_both_sides() {
        let mut fl = FreeList::new(300);
        let a = fl.alloc(100, 1).unwrap();
        let b = fl.alloc(100, 1).unwrap();
        let c = fl.alloc(100, 1).unwrap();
        fl.free(a, 100);
        fl.free(c, 100); // [0, 100) and [200, 300)
        assert_eq!(fl.free.len(), 2);
        fl.free(b, 100); // bridges both neighbours
        assert_eq!(fl.free, vec![FreeRange { offset: 0, size: 300 }]);
        assert_eq!(fl.used(), 0);
    }

    #[test]
    fn exhaustion_returns_none() {
        let mut fl = FreeList::new(256);
        assert_eq!(fl.alloc(300, 1), None);
        assert_eq!(fl.alloc(256, 1), Some(0));
        assert_eq!(fl.alloc(1, 1), None);

        // Fits by raw size but not once alignment padding is applied.
        let mut fl = FreeList::new(300);
        let _a = fl.alloc(100, 1).unwrap();
        assert_eq!(fl.alloc(200, 256), None); // would need [256, 456)
    }

    #[test]
    fn interleaved_accounting_stays_consistent() {
        let mut fl = FreeList::new(4096);
        let mut live: Vec<(u64, u64)> = Vec::new();
        let mut expected_used = 0u64;
        // Deterministic interleaving of allocs and frees.
        for step in 0..200u64 {
            if step % 3 == 2 && !live.is_empty() {
                let idx = (step as usize * 7) % live.len();
                let (offset, size) = live.remove(idx);
                fl.free(offset, size);
                expected_used -= size;
            } else {
                let size = 16 + (step * 37) % 240;
                let align = [1u64, 16, 64, 256][(step % 4) as usize];
                if let Some(offset) = fl.alloc(size, align) {
                    assert_eq!(offset % align, 0, "misaligned offset at step {step}");
                    live.push((offset, size));
                    expected_used += size;
                }
            }
            assert_eq!(fl.used(), expected_used, "used-bytes drift at step {step}");
        }
        for (offset, size) in live.drain(..) {
            fl.free(offset, size);
            expected_used -= size;
            assert_eq!(fl.used(), expected_used);
        }
        assert_eq!(fl.used(), 0);
        assert_eq!(fl.free, vec![FreeRange { offset: 0, size: 4096 }]);
    }
}
