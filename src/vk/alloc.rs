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

use super::device::{BudgetSnapshot, MemoryBudget};

/// Growth block size; larger allocations get a dedicated, larger block.
const BLOCK_SIZE: u64 = 64 * 1024 * 1024;

/// A pool's FIRST block is smaller: a small world (or the menu) holds a few
/// MB of live meshes, and a 64 MiB opening reservation was most of the app's
/// idle GPU footprint. Growth past it still comes in [`BLOCK_SIZE`] strides,
/// so a big world pays one extra allocation, ever.
const FIRST_BLOCK_SIZE: u64 = 16 * 1024 * 1024;

/// Consecutive `shrink_device` ticks a device block must stay completely
/// free before it is returned to the driver. At a per-frame cadence this is
/// several seconds of sustained emptiness, so a block that briefly drains and
/// refills (a player leaving and re-entering a region) is never released and
/// then immediately recreated — the settling window is the anti-thrash guard.
const DEVICE_SHRINK_SETTLE_TICKS: u32 = 300;

/// Whether the DEVICE pool is host-visible (unified memory).
/// Decided at construction and never changes.
#[derive(Clone, Copy)]
struct UnifiedMemory(bool);

/// A deferred `VK_EXT_memory_budget` query.
/// `snapshot()` runs the driver query on demand at block creation, not per-allocation.
#[derive(Clone)]
struct BudgetQuery {
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    token: MemoryBudget,
}

impl BudgetQuery {
    fn snapshot(self) -> BudgetSnapshot {
        unsafe { self.token.query(&self.instance, self.physical) }
    }
}

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
            free: vec![FreeRange {
                offset: 0,
                size: capacity,
            }],
        }
    }

    fn used(&self) -> u64 {
        self.used
    }

    /// True when no bytes are allocated.
    fn is_empty(&self) -> bool {
        self.used == 0
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
                    self.free[i] = FreeRange {
                        offset: aligned + size,
                        size: tail,
                    };
                }
                (true, true) => {
                    self.free[i].size = pad;
                    self.free.insert(
                        i + 1,
                        FreeRange {
                            offset: aligned + size,
                            size: tail,
                        },
                    );
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
        let merges_prev = idx > 0 && self.free[idx - 1].offset + self.free[idx - 1].size == offset;
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

/// A persistently-mapped device pointer. Vulkan persistent mappings are
/// process-wide valid; this pointer is only ever written/freed on the main
/// thread and never dereferenced on the render thread, so crossing threads is
/// sound. Wrapping [`NonNull`] this way is what makes [`Allocation`] `Send`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MappedPtr(NonNull<u8>);

// SAFETY: see the type doc — the pointer is only ever used on the main thread.
unsafe impl Send for MappedPtr {}

impl MappedPtr {
    pub(crate) fn as_ptr(self) -> *mut u8 {
        self.0.as_ptr()
    }
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
    pub mapped: Option<MappedPtr>,
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
    /// Consecutive `shrink_device` ticks this block has been completely free.
    /// Reset to 0 whenever it holds any live suballocation. Only meaningful
    /// for the device pool; staging shrink ignores it.
    empty_ticks: u32,
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
    /// Retained for budget queries at block-creation time.
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    /// Enables `VK_EXT_memory_budget` queries, if available.
    memory_budget: Option<MemoryBudget>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    /// Preferred memory type indices for the device pool, in order.
    device_type_prefs: Vec<u32>,
    /// Preferred memory type indices for the staging pool, in order.
    staging_type_prefs: Vec<u32>,
    /// Whether the device pool is host-visible (unified memory).
    unified: UnifiedMemory,
    /// Slot vectors allowing reuse of destroyed block indices.
    device_blocks: Vec<Option<Block>>,
    staging_blocks: Vec<Option<Block>>,
}

impl GpuAllocator {
    /// Captures memory properties; performs no allocations yet.
    pub unsafe fn new(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        memory_budget: Option<MemoryBudget>,
    ) -> Self {
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
        // A DEVICE_LOCAL|HOST_VISIBLE type only counts as "unified" when its
        // heap is genuinely large: discrete GPUs without resizable BAR expose
        // a ~256 MiB DEVICE_LOCAL|HOST_VISIBLE window that must NOT become the
        // main mesh pool (it would exhaust long before VRAM does).
        let heap_of = |i: u32| memory_props.memory_types[i as usize].heap_index as usize;
        let big_enough = |i: u32| {
            const MIN_UNIFIED_HEAP: u64 = 1 << 30; // 1 GiB
            memory_props.memory_heaps[heap_of(i)].size >= MIN_UNIFIED_HEAP
        };
        let mut device_type_prefs: Vec<u32> = types_with(unified)
            .into_iter()
            .filter(|&i| big_enough(i))
            .collect();
        for i in types_with(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            if !device_type_prefs.contains(&i) {
                device_type_prefs.push(i);
            }
        }
        let staging_type_prefs = types_with(host_coherent);

        let unified = UnifiedMemory(device_type_prefs.first().is_some_and(|&i| {
            memory_props.memory_types[i as usize]
                .property_flags
                .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        }));

        if device_type_prefs.is_empty() {
            // Spec guarantees a DEVICE_LOCAL type exists, but stay graceful:
            // block creation will fall back to any host-visible type.
            log::warn!("no DEVICE_LOCAL memory type reported by the driver");
        }
        if staging_type_prefs.is_empty() {
            log::warn!("no HOST_VISIBLE|HOST_COHERENT memory type reported by the driver");
        }

        Self {
            instance: instance.clone(),
            physical: physical_device,
            memory_budget,
            memory_props,
            device_type_prefs,
            staging_type_prefs,
            unified,
            device_blocks: Vec::new(),
            staging_blocks: Vec::new(),
        }
    }

    /// True when the DEVICE pool's memory type is HOST_VISIBLE (unified
    /// memory): mesh data can be written through `Allocation::mapped`
    /// directly, no staging copy needed.
    pub fn unified_memory(&self) -> bool {
        self.unified.0
    }

    /// Build a budget query for this block creation.
    fn budget_query(&self) -> Option<BudgetQuery> {
        self.memory_budget.map(|token| BudgetQuery {
            instance: self.instance.clone(),
            physical: self.physical,
            token,
        })
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
        let budget = self.budget_query();
        unsafe {
            alloc_from_pool(
                &mut self.device_blocks,
                &self.device_type_prefs,
                &self.memory_props,
                budget,
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
        let budget = self.budget_query();
        unsafe {
            alloc_from_pool(
                &mut self.staging_blocks,
                &self.staging_type_prefs,
                &self.memory_props,
                budget,
                vk::BufferUsageFlags::TRANSFER_SRC,
                Pool::Staging,
                device,
                size,
                align,
            )
        }
    }

    /// Returns the range to its block's free list.
    pub unsafe fn free(&mut self, allocation: Allocation) {
        let blocks = match allocation.pool {
            Pool::Device => &mut self.device_blocks,
            Pool::Staging => &mut self.staging_blocks,
        };
        let block = blocks
            .get_mut(allocation.block)
            .and_then(|slot| slot.as_mut())
            .expect("allocation refers to an unknown block (freed after destroy?)");
        debug_assert_eq!(
            block.buffer, allocation.buffer,
            "allocation/block buffer mismatch"
        );
        block.free_list.free(allocation.offset, allocation.size);
    }

    /// Destroys completely-free staging blocks beyond the first.
    /// Only applies on discrete GPUs (unified-memory devices skip staging).
    ///
    /// Safety: an empty block has no live allocations.
    pub unsafe fn shrink_staging(&mut self, device: &ash::Device) {
        for index in empty_blocks_beyond_first(&self.staging_blocks) {
            let block = self.staging_blocks[index]
                .take()
                .expect("selected slot holds a block");
            // Freeing the memory implicitly unmaps the persistent mapping.
            unsafe {
                device.destroy_buffer(block.buffer, None);
                device.free_memory(block.memory, None);
            }
            log::debug!(
                "destroyed empty staging block ({} MiB)",
                block.memory_size / (1024 * 1024)
            );
        }
    }

    /// Returns settled-empty device blocks to the driver, bounding the pool's
    /// high-water mark over a long session. Call once per frame.
    ///
    /// Meshing churn (the LOD pyramid multiplies it) makes the device pool
    /// grow to a transient peak; without this it would retain every block ever
    /// touched forever. A block is released only after it has been *completely*
    /// free for [`DEVICE_SHRINK_SETTLE_TICKS`] consecutive calls, keeping one
    /// empty block warm to absorb the next spike.
    ///
    /// Safety / deferred-free interaction: an empty `free_list` means the block
    /// holds zero live suballocations. Every suballocation only reaches
    /// `free()` — and thus the free_list — *after* the caller's deferred-free
    /// window (frames-in-flight) has elapsed, so no in-flight command buffer
    /// can still reference any byte of an empty block. Its slot is set to
    /// `None`; because it was empty, no outstanding `Allocation` names it, so
    /// reusing the slot later cannot alias a live handle.
    pub unsafe fn shrink_device(&mut self, device: &ash::Device) {
        for index in settled_empty_blocks(&mut self.device_blocks, DEVICE_SHRINK_SETTLE_TICKS) {
            let block = self.device_blocks[index]
                .take()
                .expect("selected slot holds a block");
            // Freeing the memory implicitly unmaps any persistent mapping.
            unsafe {
                device.destroy_buffer(block.buffer, None);
                device.free_memory(block.memory, None);
            }
            log::debug!(
                "released settled-empty device block ({} MiB)",
                block.memory_size / (1024 * 1024)
            );
        }
    }

    /// Destroy all blocks (GPU must be idle).
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        for block in self
            .device_blocks
            .drain(..)
            .chain(self.staging_blocks.drain(..))
            .flatten()
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
        for block in self.device_blocks.iter().flatten() {
            stats.device_blocks += 1;
            stats.device_reserved += block.memory_size;
            stats.device_used += block.free_list.used();
        }
        for block in self.staging_blocks.iter().flatten() {
            stats.staging_blocks += 1;
            stats.staging_reserved += block.memory_size;
            stats.staging_used += block.free_list.used();
        }
        stats
    }
}

/// Indices of completely-free blocks, excluding the first one (kept warm).
/// Returns an empty Vec in typical steady state (at most one empty block).
fn empty_blocks_beyond_first(blocks: &[Option<Block>]) -> Vec<usize> {
    let mut extra = Vec::new();
    let mut kept_one = false;
    for (index, slot) in blocks.iter().enumerate() {
        let Some(block) = slot else { continue };
        if !block.free_list.is_empty() {
            continue;
        }
        if kept_one {
            extra.push(index);
        } else {
            kept_one = true;
        }
    }
    extra
}

/// Advances each block's empty-settling counter and returns the indices of
/// blocks that have stayed completely free for at least `settle_ticks`
/// consecutive calls, always keeping the first still-empty block warm.
///
/// A block holding any live suballocation resets its counter to 0. Because the
/// counter advances only while a block is empty, a block that momentarily
/// drains and refills is never selected — this is what stops release/recreate
/// thrash. Returns an empty Vec in typical steady state.
fn settled_empty_blocks(blocks: &mut [Option<Block>], settle_ticks: u32) -> Vec<usize> {
    let mut extra = Vec::new();
    let mut kept_warm = false;
    for (index, slot) in blocks.iter_mut().enumerate() {
        let Some(block) = slot else { continue };
        if !block.free_list.is_empty() {
            block.empty_ticks = 0;
            continue;
        }
        block.empty_ticks = block.empty_ticks.saturating_add(1);
        if !kept_warm {
            kept_warm = true;
        } else if block.empty_ticks >= settle_ticks {
            extra.push(index);
        }
    }
    extra
}

/// Allocates from `blocks`, creating a new block when no existing block has
/// a large enough hole.
#[allow(clippy::too_many_arguments)]
unsafe fn alloc_from_pool(
    blocks: &mut Vec<Option<Block>>,
    type_prefs: &[u32],
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    budget: Option<BudgetQuery>,
    usage: vk::BufferUsageFlags,
    pool: Pool,
    device: &ash::Device,
    size: u64,
    align: u64,
) -> Result<Allocation, vk::Result> {
    debug_assert!(size > 0, "zero-size GPU allocation");
    let size = size.max(1);

    for (index, slot) in blocks.iter_mut().enumerate() {
        let Some(block) = slot else { continue };
        if let Some(offset) = block.free_list.alloc(size, align) {
            return Ok(make_allocation(block, index, offset, size, pool));
        }
    }

    // The pool's first-ever block opens small (see FIRST_BLOCK_SIZE).
    let floor = if blocks.iter().flatten().next().is_none() { FIRST_BLOCK_SIZE } else { BLOCK_SIZE };
    let block =
        unsafe { create_block(device, memory_props, type_prefs, budget, usage, size, floor)? };
    // Reuse a destroyed block's slot so existing Allocation indices stay valid.
    let index = match blocks.iter().position(|slot| slot.is_none()) {
        Some(index) => {
            blocks[index] = Some(block);
            index
        }
        None => {
            blocks.push(Some(block));
            blocks.len() - 1
        }
    };
    let block = blocks[index].as_mut().expect("slot was just filled");
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
        MappedPtr(unsafe { NonNull::new_unchecked(base.as_ptr().add(offset as usize)) })
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

/// Creates a block of at least `min_size` bytes (rounded up to `size_floor` —
/// [`FIRST_BLOCK_SIZE`] or [`BLOCK_SIZE`]), bound at offset 0 and persistently
/// mapped when the chosen memory type is host-visible.
unsafe fn create_block(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_prefs: &[u32],
    budget: Option<BudgetQuery>,
    usage: vk::BufferUsageFlags,
    min_size: u64,
    size_floor: u64,
) -> Result<Block, vk::Result> {
    let buffer_size = min_size.max(size_floor);
    let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

    let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

    let Some(type_index) = pick_memory_type(
        memory_props,
        type_prefs,
        requirements.memory_type_bits,
        budget.map(BudgetQuery::snapshot),
        requirements.size,
    ) else {
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
        match unsafe { device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) } {
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
        empty_ticks: 0,
    })
}

/// Selects a memory type matching `type_filter` and `properties`.
pub fn find_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> u32 {
    try_find_memory_type(memory_props, type_filter, properties)
        .expect("No suitable memory type")
}

/// [`find_memory_type`] without the panic: `None` when no type matches, so a
/// caller can express a PREFERENCE ladder (e.g. HOST_CACHED for CPU-read
/// buffers, falling back to plain coherent memory where the device has none).
pub fn try_find_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..memory_props.memory_type_count).find(|&i| {
        let suitable = (type_filter & (1 << i)) != 0;
        let has_props = memory_props.memory_types[i as usize]
            .property_flags
            .contains(properties);
        suitable && has_props
    })
}

/// Picks a memory type from the preference list, respecting budget constraints.
/// Falls back to less-preferred types if needed; budget limits are advisory.
fn pick_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_prefs: &[u32],
    memory_type_bits: u32,
    budget: Option<BudgetSnapshot>,
    alloc_size: u64,
) -> Option<u32> {
    // Compatible types in preference order, then any host-visible fallback.
    let mut candidates: Vec<u32> = type_prefs
        .iter()
        .copied()
        .filter(|&i| memory_type_bits & (1 << i) != 0)
        .collect();
    for i in 0..memory_props.memory_type_count {
        let compatible = memory_type_bits & (1 << i) != 0;
        let host_visible = memory_props.memory_types[i as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
        if compatible && host_visible && !candidates.contains(&i) {
            candidates.push(i);
        }
    }

    let within_budget = |i: u32| match budget {
        None => true,
        Some(snapshot) => {
            let heap = memory_props.memory_types[i as usize].heap_index as usize;
            snapshot.heap_usage[heap].saturating_add(alloc_size) <= snapshot.heap_budget[heap]
        }
    };

    // Prefer a type with budget headroom; fall back to the most-preferred anyway.
    if let Some(&i) = candidates.iter().find(|&&i| within_budget(i)) {
        return Some(i);
    }
    if let Some(&i) = candidates.first() {
        if budget.is_some() {
            log::warn!(
                "all compatible memory types report over budget for a {} MiB block; \
                 attempting type {i} anyway",
                alloc_size / (1024 * 1024)
            );
        } else {
            log::warn!("preferred memory types unavailable; falling back to type {i}");
        }
        return Some(i);
    }
    None
}

// ---------------------------------------------------------------------------
// Tests (FreeList only — no GPU required)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::device::BudgetSnapshot;
    use super::{
        Block, FreeList, FreeRange, empty_blocks_beyond_first, pick_memory_type,
        settled_empty_blocks,
    };
    use ash::vk;

    /// Two memory types on two heaps: type 0 (preferred, heap 0) and a
    /// host-visible type 1 (fallback, heap 1). Both compatible with any buffer.
    fn two_heap_props() -> vk::PhysicalDeviceMemoryProperties {
        let mut props = vk::PhysicalDeviceMemoryProperties::default();
        props.memory_type_count = 2;
        props.memory_types[0] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            heap_index: 0,
        };
        props.memory_types[1] = vk::MemoryType {
            property_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT,
            heap_index: 1,
        };
        props.memory_heap_count = 2;
        props
    }

    #[test]
    fn budget_skips_full_heap() {
        let props = two_heap_props();
        let prefs = [0u32]; // prefer type 0 (heap 0)
        let bits = 0b11; // both types compatible

        // No budget info: always the preferred type.
        assert_eq!(pick_memory_type(&props, &prefs, bits, None, 64), Some(0));

        // Heap 0 has headroom: preferred type wins.
        let ok = BudgetSnapshot {
            heap_budget: [1000; vk::MAX_MEMORY_HEAPS],
            heap_usage: [0; vk::MAX_MEMORY_HEAPS],
        };
        assert_eq!(
            pick_memory_type(&props, &prefs, bits, Some(ok), 64),
            Some(0)
        );

        // Heap 0 over budget, heap 1 has room: degrade to the host-visible type.
        let mut tight = BudgetSnapshot {
            heap_budget: [1000; vk::MAX_MEMORY_HEAPS],
            heap_usage: [0; vk::MAX_MEMORY_HEAPS],
        };
        tight.heap_usage[0] = 990; // 990 + 64 > 1000
        assert_eq!(
            pick_memory_type(&props, &prefs, bits, Some(tight), 64),
            Some(1)
        );

        // Every heap over budget: fall back to the most-preferred type anyway
        // (budget is advisory; the OOM path is the real backstop).
        let full = BudgetSnapshot {
            heap_budget: [10; vk::MAX_MEMORY_HEAPS],
            heap_usage: [10; vk::MAX_MEMORY_HEAPS],
        };
        assert_eq!(
            pick_memory_type(&props, &prefs, bits, Some(full), 64),
            Some(0)
        );
    }

    /// A GPU-less block (null handles) for pure bookkeeping tests.
    fn dummy_block(capacity: u64, used: u64) -> Option<Block> {
        let mut free_list = FreeList::new(capacity);
        if used > 0 {
            free_list.alloc(used, 1).expect("fits");
        }
        Some(Block {
            buffer: ash::vk::Buffer::null(),
            memory: ash::vk::DeviceMemory::null(),
            memory_size: capacity,
            mapped: None,
            free_list,
            empty_ticks: 0,
        })
    }

    #[test]
    fn free_list_empty_detection() {
        let mut fl = FreeList::new(1024);
        assert!(fl.is_empty());
        let a = fl.alloc(64, 1).expect("fits");
        assert!(!fl.is_empty());
        fl.free(a, 64);
        assert!(fl.is_empty());
        // Fully free again means the whole capacity is one range.
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 1024
            }]
        );
    }

    #[test]
    fn staging_shrink_keeps_one_warm() {
        // No blocks / a single empty block: nothing to destroy.
        assert!(empty_blocks_beyond_first(&[]).is_empty());
        assert!(empty_blocks_beyond_first(&[dummy_block(1024, 0)]).is_empty());

        // In-use blocks are never selected, however many are empty.
        let blocks = vec![dummy_block(1024, 100), dummy_block(1024, 1)];
        assert!(empty_blocks_beyond_first(&blocks).is_empty());

        // More than one empty: the FIRST empty stays warm, the rest go.
        let blocks = vec![
            dummy_block(1024, 100), // in use — kept
            dummy_block(1024, 0),   // first empty — kept warm
            dummy_block(1024, 0),   // destroyed
            None,                   // already-destroyed slot — skipped
            dummy_block(1024, 0),   // destroyed
        ];
        assert_eq!(empty_blocks_beyond_first(&blocks), vec![2, 4]);
    }

    #[test]
    fn device_shrink_settles_before_release() {
        // Two empty blocks beyond a warm one; a third in use.
        let mut blocks = vec![
            dummy_block(1024, 100), // in use — never selected
            dummy_block(1024, 0),   // first empty — kept warm
            dummy_block(1024, 0),   // eligible after settling
        ];

        // Below threshold: settling, nothing released yet.
        for _ in 0..2 {
            assert!(settled_empty_blocks(&mut blocks, 3).is_empty());
        }
        // Third consecutive empty tick crosses the window.
        assert_eq!(settled_empty_blocks(&mut blocks, 3), vec![2]);
    }

    #[test]
    fn device_shrink_refill_resets_counter() {
        let mut blocks = vec![
            dummy_block(1024, 0), // kept warm
            dummy_block(1024, 0), // candidate
        ];
        // Accumulate settling ticks just shy of release.
        for _ in 0..2 {
            assert!(settled_empty_blocks(&mut blocks, 3).is_empty());
        }
        // The candidate briefly refills: its counter must reset.
        blocks[1].as_mut().unwrap().free_list.alloc(64, 1).unwrap();
        assert!(settled_empty_blocks(&mut blocks, 3).is_empty());
        blocks[1].as_mut().unwrap().free_list.free(0, 64);
        // Starts settling from zero again — no premature release.
        for _ in 0..2 {
            assert!(settled_empty_blocks(&mut blocks, 3).is_empty());
        }
        assert_eq!(settled_empty_blocks(&mut blocks, 3), vec![1]);
    }

    #[test]
    fn alloc_free_roundtrip() {
        let mut fl = FreeList::new(1024);
        let a = fl.alloc(100, 1).expect("fits");
        assert_eq!(a, 0);
        assert_eq!(fl.used(), 100);
        fl.free(a, 100);
        assert_eq!(fl.used(), 0);
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 1024
            }]
        );
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
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 200
            }]
        );
    }

    #[test]
    fn coalesce_right() {
        let mut fl = FreeList::new(300);
        let a = fl.alloc(100, 1).unwrap();
        let b = fl.alloc(100, 1).unwrap();
        let _c = fl.alloc(100, 1).unwrap();
        fl.free(b, 100); // [100, 200)
        fl.free(a, 100); // merges into the right neighbour
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 200
            }]
        );
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
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 300
            }]
        );
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
        assert_eq!(
            fl.free,
            vec![FreeRange {
                offset: 0,
                size: 4096
            }]
        );
    }
}
