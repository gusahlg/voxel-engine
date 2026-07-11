//! Per-frame uniform buffer ring (set 0, binding 2) — the engine-side home of
//! `UboRing`. Two host-visible, host-coherent, persistently
//! mapped buffers (one per frame-in-flight), each exactly the size of the wire
//! struct. Written once per frame before recording; bound by push descriptor
//! alongside the offsets SSBO (binding 0) and block texture (binding 1).
//!
//! `HostBuffer` wraps each slot's buffer handle with its persistent mapping,
//! which `write` requires for coherent copies (a bare `ash::vk::Buffer` has no
//! mapped pointer). Indexed by `FrameSlot` to prevent raw-usize confusion.

use ash::vk;

use crate::skeleton::{FrameSlot, FrameUniformsGpu, PerSlot};
use crate::vk::buffers::HostBuffer;

/// The per-frame UBO ring. Indexed only by [`FrameSlot`] (the parity type),
/// so raw-usize slot confusion is inexpressible here.
pub(crate) struct UboRing {
    bufs: PerSlot<HostBuffer>,
}

impl UboRing {
    /// Allocate both slots' UBOs, each sized to the wire struct. [`HostBuffer`]
    /// is `HOST_VISIBLE | HOST_COHERENT` by construction, so no flush is ever
    /// needed and the skeleton's "assert coherent at creation" requirement is
    /// satisfied structurally. Call at renderer init (GPU idle), which is what
    /// [`HostBuffer::maintain`] requires.
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let size = size_of::<FrameUniformsGpu>() as u64;
        let make = || {
            let mut b = HostBuffer::new(vk::BufferUsageFlags::UNIFORM_BUFFER);
            // GPU is idle during init; `maintain` allocates + persistently maps.
            unsafe { b.maintain(instance, device, physical, size) };
            b
        };
        Self {
            bufs: PerSlot::new([make(), make()]),
        }
    }

    /// Copy this frame's uniforms into `slot`'s mapped buffer. Coherent memory:
    /// the write is visible to the GPU with no explicit flush.
    pub(crate) fn write(&mut self, slot: FrameSlot, u: &FrameUniformsGpu) {
        unsafe { self.bufs[slot].write(0, bytemuck::bytes_of(u)) };
    }

    /// The buffer bound at set 0, binding 2 for `slot`. The per-frame UBO is
    /// written unconditionally every frame before any pass reads it, so it is
    /// always allocated here.
    pub(crate) fn buffer(&self, slot: FrameSlot) -> vk::Buffer {
        self.bufs[slot]
            .bound()
            .expect("the per-frame UBO is written every frame before it is bound")
    }

    pub(crate) unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.bufs[FrameSlot::new(0)].destroy(device);
            self.bufs[FrameSlot::new(1)].destroy(device);
        }
    }
}
