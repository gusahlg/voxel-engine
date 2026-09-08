//! Per-frame uniform buffer ring (set 0, binding 2) — the engine-side home of
//! `UboRing`. Two host-visible, host-coherent, persistently
//! mapped buffers (one per frame-in-flight), each the size of the GPU struct
//! (public [`FrameUniformsGpu`] plus the engine-derived tail). Written once per
//! frame before recording; bound by push descriptor alongside the offsets SSBO
//! (binding 0) and block texture (binding 1).
//!
//! `HostBuffer` wraps each slot's buffer handle with its persistent mapping,
//! which `write` requires for coherent copies (a bare `ash::vk::Buffer` has no
//! mapped pointer). Indexed by `FrameSlot` to prevent raw-usize confusion.

use ash::vk;
use glam::Vec3;

use crate::genconst::{GLOW_EDGE0, GLOW_EDGE1, GLOW_POW_DAY, GLOW_POW_SUNSET};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

pub const FRAME_UNIFORMS_SET: u32 = 0;
pub const FRAME_UNIFORMS_BINDING: u32 = 2;

// FrameUniformsGpu / FrameUniformsExt are generated from build.rs; edit the
// lane tables there.
include!(concat!(env!("OUT_DIR"), "/gen_frame_uniforms.rs"));

impl FrameUniformsGpu {
    /// Neutral fully-lit default.
    pub fn full_bright() -> Self {
        let mut n = <Self as bytemuck::Zeroable>::zeroed();
        n.sun_dir_elev = [0.0, 1.0, 0.0, std::f32::consts::FRAC_PI_2];
        n.candle[3] = 1.0; // ambient
        n.exposure_dither[0] = 1.0; // exposure
        n.extras[0] = 1.0; // stars
        n
    }
}

impl FrameUniformsExt {
    /// Hoist per-frame uniform-only math the shaders used to recompute every
    /// fragment: ambient floor colour, sky-halo exponent, halo tint×scale, and
    /// the shadow-fallback day factor. Mirrors `common.slang`.
    fn derive(u: FrameUniformsGpu) -> Self {
        let zenith = Vec3::new(u.zenith[0], u.zenith[1], u.zenith[2]);
        let floor = u.candle[3];
        let zl = zenith.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        let ambient = if zl > 1e-6 {
            zenith * (floor / zl)
        } else {
            Vec3::splat(floor)
        };
        let t = {
            let x = ((u.sun_dir_elev[3] - GLOW_EDGE0) / (GLOW_EDGE1 - GLOW_EDGE0)).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let glow_pow = GLOW_POW_SUNSET + (GLOW_POW_DAY - GLOW_POW_SUNSET) * t;
        let glow_rgb = Vec3::new(u.light[0], u.light[1], u.light[2]) * (0.5 + u.zenith[3]);
        let day = (u.light[3] * 2.0 - 1.0).abs();
        Self {
            base: u,
            ambient_glow: [ambient.x, ambient.y, ambient.z, glow_pow],
            glow_day: [glow_rgb.x, glow_rgb.y, glow_rgb.z, day],
        }
    }
}

// Bumped when the GPU `FrameUniforms` layout changes (public prefix or derived tail).
pub const FRAME_UNIFORMS_VERSION: u32 = 4;

/// The per-frame UBO ring. Indexed only by [`FrameSlot`] (the parity type),
/// so raw-usize slot confusion is inexpressible here.
pub(crate) struct UboRing {
    bufs: PerSlot<HostBuffer>,
}

impl UboRing {
    /// Allocate both slots' UBOs, each sized to the GPU struct (public wire +
    /// derived tail). [`HostBuffer`] is `HOST_VISIBLE | HOST_COHERENT` by
    /// construction, so no flush is ever needed and the skeleton's "assert
    /// coherent at creation" requirement is satisfied structurally. Call at
    /// renderer init (GPU idle), which is what [`HostBuffer::maintain`] requires.
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
    ) -> Self {
        let size = size_of::<FrameUniformsExt>() as u64;
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

    /// Copy this frame's uniforms into `slot`'s mapped buffer, appending the
    /// derived tail. Coherent memory: the write is visible to the GPU with no
    /// explicit flush.
    pub(crate) fn write(&mut self, slot: FrameSlot, u: &FrameUniformsGpu) {
        let ext = FrameUniformsExt::derive(*u);
        unsafe { self.bufs[slot].write(0, bytemuck::bytes_of(&ext)) };
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

    fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    }

    /// Public wire stays 8 float4s so the game's `From<&FrameSnapshot>` layout
    /// cannot silently grow; derived lanes live past that prefix.
    #[test]
    fn public_wire_stays_eight_lanes() {
        assert_eq!(size_of::<FrameUniformsGpu>(), 128);
        assert_eq!(size_of::<FrameUniformsExt>(), 160);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, ambient_glow), 128);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, glow_day), 144);
    }

    #[test]
    fn derived_lanes_match_shader_formulas() {
        let mut u = FrameUniformsGpu::full_bright();
        u.zenith = [0.09, 0.22, 0.45, 2.0];
        u.candle[3] = 0.30;
        u.light = [1.25, 1.15, 1.0, 1.0];
        u.sun_dir_elev[3] = 0.2;

        let ext = FrameUniformsExt::derive(u);
        let luma = 0.2126 * 0.09 + 0.7152 * 0.22 + 0.0722 * 0.45;
        let scale = 0.30 / luma;
        for i in 0..3 {
            assert!(
                (ext.ambient_glow[i] - u.zenith[i] * scale).abs() < 1e-6,
                "ambient[{i}]"
            );
        }
        let t = smoothstep(GLOW_EDGE0, GLOW_EDGE1, 0.2);
        let glow_pow = GLOW_POW_SUNSET + (GLOW_POW_DAY - GLOW_POW_SUNSET) * t;
        assert!((ext.ambient_glow[3] - glow_pow).abs() < 1e-6);
        let glow_scale = 0.5 + 2.0;
        assert!((ext.glow_day[0] - 1.25 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[1] - 1.15 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[2] - 1.0 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn derived_ambient_falls_back_when_zenith_is_black() {
        let u = FrameUniformsGpu::full_bright();
        // full_bright leaves zenith at zero, candle.w = 1.
        let ext = FrameUniformsExt::derive(u);
        assert_eq!(&ext.ambient_glow[..3], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn derived_day_factor_tracks_night_and_noon() {
        let mut u = FrameUniformsGpu::full_bright();
        u.light[3] = 0.0;
        assert!((FrameUniformsExt::derive(u).glow_day[3] - 1.0).abs() < 1e-7);
        u.light[3] = 0.5;
        assert!(FrameUniformsExt::derive(u).glow_day[3].abs() < 1e-7);
        u.light[3] = 1.0;
        assert!((FrameUniformsExt::derive(u).glow_day[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn derived_glow_pow_is_sunset_below_edge0_and_day_above_edge1() {
        let mut u = FrameUniformsGpu::full_bright();
        u.sun_dir_elev[3] = GLOW_EDGE0 - 1.0;
        assert!((FrameUniformsExt::derive(u).ambient_glow[3] - GLOW_POW_SUNSET).abs() < 1e-6);
        u.sun_dir_elev[3] = GLOW_EDGE1 + 1.0;
        assert!((FrameUniformsExt::derive(u).ambient_glow[3] - GLOW_POW_DAY).abs() < 1e-6);
    }
}
