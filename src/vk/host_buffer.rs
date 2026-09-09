use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;

use super::alloc::try_find_memory_type;

/// Smallest immediate-buffer capacity (also the floor the decay stops at).
const IMM_MIN_CAPACITY: u64 = 64 * 1024;
/// Decay window for capacity shrinking.
const IMM_SHRINK_WINDOW: u32 = 600;

/// Ceiling on [`HostBuffer`] bytes placed in a *small* BAR heap. Discrete GPUs
/// without ReBAR expose only a ~256 MiB `DEVICE_LOCAL | HOST_VISIBLE` window;
/// host buffers take a bounded slice of it and the rest stay in system memory.
/// ReBAR / unified heaps (>= [`SMALL_BAR_HEAP`]) are used without this cap.
const HOST_BAR_CAP: u64 = 64 << 20;
/// Same threshold [`super::alloc`] uses to tell a real unified/ReBAR heap from
/// a discrete GPU's small BAR window.
const SMALL_BAR_HEAP: u64 = 1 << 30;
/// Bytes currently charged against [`HOST_BAR_CAP`] (small-BAR devices only).
pub(crate) static HOST_BAR_BYTES: AtomicU64 = AtomicU64::new(0);

/// Host-visible, host-coherent — the property set every [`HostBuffer`] write
/// relies on (persistent mapping, no explicit flush).
pub(crate) const HOST_COHERENT: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
    vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
        | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
);

fn heap_size(memory_props: &vk::PhysicalDeviceMemoryProperties, type_index: u32) -> u64 {
    let heap = memory_props.memory_types[type_index as usize].heap_index as usize;
    memory_props.memory_heaps[heap].size
}

/// Picks the memory type for a [`HostBuffer`] of `size` bytes: the first BAR
/// type (`DEVICE_LOCAL` on top of [`HOST_COHERENT`]) when that heap is large
/// (ReBAR / unified) or when `bar_used + size` stays under [`HOST_BAR_CAP`]
/// on a small BAR; else the first plain host-coherent type.
///
/// The `bool` is whether the pick *is* the BAR type (allocation failure then
/// falls back to system memory). Charging the cap is a separate decision at
/// allocate time: only small BAR heaps consume [`HOST_BAR_BYTES`].
pub(crate) fn host_buffer_memory_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    size: u64,
    bar_used: u64,
) -> Option<(u32, bool)> {
    if let Some(i) = try_find_memory_type(
        memory_props,
        type_filter,
        HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL,
    ) {
        let small = heap_size(memory_props, i) < SMALL_BAR_HEAP;
        if !small || bar_used.saturating_add(size) <= HOST_BAR_CAP {
            return Some((i, true));
        }
    }
    try_find_memory_type(memory_props, type_filter, HOST_COHERENT).map(|i| (i, false))
}

pub(crate) fn bar_charge(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_index: u32,
    size: u64,
) -> u64 {
    if heap_size(memory_props, type_index) < SMALL_BAR_HEAP {
        size
    } else {
        0
    }
}

/// A growable host-visible buffer written each frame, one per frame-in-flight.
/// Used for immediate geometry and indirect commands.
pub struct HostBuffer {
    /// Null until first write; use [`Self::bound`] to obtain safely.
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
    /// Bytes this buffer holds against [`HOST_BAR_CAP`] (0 = system memory).
    bar_bytes: u64,
    usage: vk::BufferUsageFlags,
    /// Peak need in decay window.
    window_peak: u64,
    /// Frame count in decay window.
    window_frames: u32,
}

impl HostBuffer {
    /// Get the buffer handle, or `None` if unallocated.
    pub fn bound(&self) -> Option<vk::Buffer> {
        (self.buffer != vk::Buffer::null()).then_some(self.buffer)
    }

    pub fn new(usage: vk::BufferUsageFlags) -> Self {
        Self {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null_mut(),
            capacity: 0,
            bar_bytes: 0,
            usage,
            window_peak: 0,
            window_frames: 0,
        }
    }

    /// Maintain capacity and shrink if needed. Call after fence is waited.
    pub unsafe fn maintain(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) -> bool {
        let mut changed = false;
        if needed > self.window_peak {
            self.window_peak = needed;
        }
        self.window_frames += 1;
        if self.window_frames >= IMM_SHRINK_WINDOW {
            let peak = self.window_peak;
            self.window_frames = 0;
            self.window_peak = 0;
            if let Some(target) = shrink_capacity(self.capacity, peak) {
                unsafe {
                    self.destroy(device);
                    changed = true;
                    if target > 0 {
                        self.ensure_capacity(instance, device, physical, target);
                    }
                }
            }
        }
        if needed > 0 {
            changed |= unsafe { self.ensure_capacity(instance, device, physical, needed) };
        }
        changed
    }

    unsafe fn ensure_capacity(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        needed: u64,
    ) -> bool {
        if needed <= self.capacity {
            return false;
        }
        let new_capacity = needed.next_power_of_two().max(IMM_MIN_CAPACITY);
        unsafe {
            self.destroy(device);

            let memory_props = instance.get_physical_device_memory_properties(physical);
            let info = vk::BufferCreateInfo::default()
                .size(new_capacity)
                .usage(self.usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device
                .create_buffer(&info, None)
                .expect("create host buffer");
            let req = device.get_buffer_memory_requirements(buffer);
            // BAR first (charged against the cap), then system memory. A BAR
            // allocation that the driver refuses anyway (the window is shared
            // with everything else) falls back the same way.
            let bar_used = HOST_BAR_BYTES.load(Ordering::Relaxed);
            let (type_index, is_bar) =
                host_buffer_memory_type(&memory_props, req.memory_type_bits, req.size, bar_used)
                    .expect("no HOST_VISIBLE | HOST_COHERENT memory type for a host buffer");
            let allocate = |type_index: u32| {
                device.allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(type_index),
                    None,
                )
            };
            let (memory, bar_bytes) = match allocate(type_index) {
                Ok(memory) => (
                    memory,
                    if is_bar {
                        bar_charge(&memory_props, type_index, req.size)
                    } else {
                        0
                    },
                ),
                Err(err) if is_bar => {
                    log::debug!(
                        "BAR host buffer allocation refused ({err:?}); using system memory"
                    );
                    let fallback =
                        try_find_memory_type(&memory_props, req.memory_type_bits, HOST_COHERENT)
                            .expect(
                                "no HOST_VISIBLE | HOST_COHERENT memory type for a host buffer",
                            );
                    (allocate(fallback).expect("allocate host buffer memory"), 0)
                }
                Err(err) => panic!("allocate host buffer memory: {err:?}"),
            };
            HOST_BAR_BYTES.fetch_add(bar_bytes, Ordering::Relaxed);
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind host buffer memory");
            let mapped = device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("Failed to map immediate buffer") as *mut u8;

            self.buffer = buffer;
            self.memory = memory;
            self.mapped = mapped;
            self.capacity = new_capacity;
            self.bar_bytes = bar_bytes;
        }
        true
    }

    pub unsafe fn write(&mut self, offset: u64, bytes: &[u8]) {
        assert!(
            offset
                .checked_add(bytes.len() as u64)
                .is_some_and(|end| end <= self.capacity)
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.mapped.add(offset as usize),
                bytes.len(),
            );
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
            HOST_BAR_BYTES.fetch_sub(self.bar_bytes, Ordering::Relaxed);
            self.buffer = vk::Buffer::null();
            self.memory = vk::DeviceMemory::null();
            self.mapped = std::ptr::null_mut();
            self.capacity = 0;
            self.bar_bytes = 0;
        }
    }
}

fn shrink_capacity(capacity: u64, peak: u64) -> Option<u64> {
    if capacity <= IMM_MIN_CAPACITY {
        return None; // already at (or below) the floor
    }
    if peak == 0 {
        return Some(0);
    }
    (capacity > peak.saturating_mul(4)).then(|| peak.saturating_mul(2))
}

#[cfg(test)]
mod tests {
    use super::{
        HOST_BAR_CAP, HOST_COHERENT, IMM_MIN_CAPACITY, host_buffer_memory_type, shrink_capacity,
    };
    use ash::vk;

    fn props(
        types: &[(vk::MemoryPropertyFlags, u32)],
        heap_sizes: &[u64],
    ) -> vk::PhysicalDeviceMemoryProperties {
        let mut p = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            memory_heap_count: heap_sizes.len() as u32,
            ..Default::default()
        };
        for (i, &(property_flags, heap_index)) in types.iter().enumerate() {
            p.memory_types[i] = vk::MemoryType {
                property_flags,
                heap_index,
            };
        }
        for (i, &size) in heap_sizes.iter().enumerate() {
            p.memory_heaps[i].size = size;
        }
        p
    }

    #[test]
    fn host_buffers_prefer_the_bar_type_until_the_cap_and_fall_back_to_system_memory() {
        // Discrete layout: device-local VRAM, system host-coherent, then a
        // small BAR window (no ReBAR).
        let bar = HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let discrete = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 2),
            ],
            &[8 << 30, 16 << 30, 256 << 20],
        );
        let all = 0b111;
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, 0),
            Some((2, true))
        );
        // At the cap the same request lands in system memory.
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, HOST_BAR_CAP),
            Some((1, false))
        );
        assert_eq!(
            host_buffer_memory_type(&discrete, all, 1 << 20, HOST_BAR_CAP - (1 << 20)),
            Some((2, true))
        );
        // A request larger than the cap skips the small BAR even when unused.
        assert_eq!(
            host_buffer_memory_type(&discrete, all, HOST_BAR_CAP + 1, 0),
            Some((1, false))
        );
        // A type filter excluding the BAR type skips it regardless of headroom.
        assert_eq!(
            host_buffer_memory_type(&discrete, 0b011, 1 << 20, 0),
            Some((1, false))
        );
        // ReBAR: the BAR heap is the full VRAM window, so the cap does not apply.
        let rebar = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
                (bar, 0),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(
            host_buffer_memory_type(&rebar, all, 1 << 20, HOST_BAR_CAP),
            Some((2, true))
        );
        // No BAR type at all: plain host-coherent.
        let no_bar = props(
            &[
                (vk::MemoryPropertyFlags::DEVICE_LOCAL, 0),
                (HOST_COHERENT, 1),
            ],
            &[8 << 30, 16 << 30],
        );
        assert_eq!(
            host_buffer_memory_type(&no_bar, 0b11, 4096, 0),
            Some((1, false))
        );
        // Device-local only (no host-visible type): nothing suitable.
        let dl = props(&[(vk::MemoryPropertyFlags::DEVICE_LOCAL, 0)], &[8 << 30]);
        assert_eq!(host_buffer_memory_type(&dl, 0b1, 4096, 0), None);
    }

    #[test]
    fn shrink_decay_rules() {
        // At or below the floor: never shrink, even when idle.
        assert_eq!(shrink_capacity(IMM_MIN_CAPACITY, 0), None);
        assert_eq!(shrink_capacity(0, 0), None);
        // A whole window with zero usage: destroy outright.
        assert_eq!(shrink_capacity(1 << 20, 0), Some(0));
        // Capacity within 4x of the mark: keep.
        assert_eq!(shrink_capacity(1 << 20, 1 << 18), None); // exactly 4x
        assert_eq!(shrink_capacity(1 << 20, (1 << 18) + 1), None);
        assert_eq!(shrink_capacity(1 << 20, 1 << 19), None);
        // Way oversized: recreate at 2x the mark.
        assert_eq!(shrink_capacity(1 << 20, (1 << 18) - 1), Some((1 << 19) - 2));
        assert_eq!(shrink_capacity(16 << 20, 100 << 10), Some(200 << 10));
        // The 2x target is always strictly below the old capacity.
        let target = shrink_capacity(16 << 20, 100 << 10).unwrap();
        assert!(target.next_power_of_two().max(IMM_MIN_CAPACITY) < 16 << 20);
    }
}
