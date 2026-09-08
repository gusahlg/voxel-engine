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

use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

pub const FRAME_UNIFORMS_SET: u32 = 0;
pub const FRAME_UNIFORMS_BINDING: u32 = 2;

// FrameUniformsGpu is generated from build.rs; edit the lane_table there.
include!(concat!(env!("OUT_DIR"), "/gen_frame_uniforms.rs"));

impl FrameUniformsGpu {
    /// Neutral fully-lit default.
    pub fn full_bright() -> Self {
        let mut n = <Self as bytemuck::Zeroable>::zeroed();
        n.sun_dir_elev = [0.0, 1.0, 0.0, std::f32::consts::FRAC_PI_2];
        n.candle[3] = 1.0; // ambient
        n.exposure_dither[0] = 1.0; // exposure
        n.extras[0] = 1.0; // stars
        n.prepare_derived();
        n
    }

    /// Fill engine-derived sky lanes: unit `sun_dir_elev.xyz`, `extras.y` = glow_pow,
    /// `extras.z` = 0.5 + turbidity. Does not touch `extras.x` (stars gain).
    /// Called at the producer→GPU write so fog/water/sky agree without a per-pixel
    /// normalize / smoothstep / lerp of per-frame constants.
    pub fn prepare_derived(&mut self) {
        let x = self.sun_dir_elev[0];
        let y = self.sun_dir_elev[1];
        let z = self.sun_dir_elev[2];
        let len = (x * x + y * y + z * z).sqrt();
        if len > 1e-8 {
            let inv = 1.0 / len;
            self.sun_dir_elev[0] = x * inv;
            self.sun_dir_elev[1] = y * inv;
            self.sun_dir_elev[2] = z * inv;
        }
        let t = glow_smoothstep(
            crate::genconst::GLOW_EDGE0,
            crate::genconst::GLOW_EDGE1,
            self.sun_dir_elev[3],
        );
        self.extras[1] =
            crate::genconst::GLOW_POW_SUNSET * (1.0 - t) + crate::genconst::GLOW_POW_DAY * t;
        self.extras[2] = 0.5 + self.zenith[3];
    }
}

/// GLSL `smoothstep(edge0, edge1, x)`: clamp then Hermite cubic.
fn glow_smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Bumped when lane_table layout or extras.yz semantics change.
pub const FRAME_UNIFORMS_VERSION: u32 = 4;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_derived_normalizes_sun_and_fills_glow() {
        let mut u = FrameUniformsGpu::full_bright();
        u.sun_dir_elev = [3.0, 0.0, 4.0, crate::genconst::GLOW_EDGE1];
        u.zenith[3] = 1.5;
        u.extras[0] = 0.25;
        u.prepare_derived();
        let dir = glam::Vec3::new(u.sun_dir_elev[0], u.sun_dir_elev[1], u.sun_dir_elev[2]);
        assert!((dir.length() - 1.0).abs() < 1e-5);
        assert!((u.sun_dir_elev[0] - 0.6).abs() < 1e-5);
        assert!((u.sun_dir_elev[2] - 0.8).abs() < 1e-5);
        assert_eq!(u.extras[0], 0.25);
        assert!((u.extras[1] - crate::genconst::GLOW_POW_DAY).abs() < 1e-5);
        assert!((u.extras[2] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn prepare_derived_sunset_glow_at_or_below_horizon() {
        let mut u = FrameUniformsGpu::full_bright();
        u.sun_dir_elev = [0.0, 1.0, 0.0, crate::genconst::GLOW_EDGE0];
        u.prepare_derived();
        assert!((u.extras[1] - crate::genconst::GLOW_POW_SUNSET).abs() < 1e-5);
        u.sun_dir_elev[3] = crate::genconst::GLOW_EDGE0 - 0.5;
        u.prepare_derived();
        assert!((u.extras[1] - crate::genconst::GLOW_POW_SUNSET).abs() < 1e-5);
    }
}
