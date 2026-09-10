//! Per-frame uniform buffer ring (set 0, binding 2) — the engine-side home of
//! `UboRing`. One host-visible, host-coherent, persistently
//! mapped buffer per frame-in-flight, each the size of the GPU struct
//! (public [`FrameUniformsGpu`] plus the engine-derived tail). Written once per
//! frame before recording; bound by push descriptor alongside the offsets SSBO
//! (binding 0), block texture (binding 1), and material table (binding 7).
//!
//! `HostBuffer` wraps each slot's buffer handle with its persistent mapping,
//! which `write` requires for coherent copies (a bare `ash::vk::Buffer` has no
//! mapped pointer). Indexed by `FrameSlot` to prevent raw-usize confusion.

use ash::vk;
use glam::Vec3;

use crate::engine::RenderFlags;
use crate::genconst::{
    GLOW_EDGE0, GLOW_EDGE1, GLOW_POW_DAY, GLOW_POW_SUNSET, SHADOW_BOUNCE_TINT, SHADOW_SKY_AMBIENT,
};
use crate::rev::{FrameSlot, PerSlot};
use crate::vk::buffers::HostBuffer;

/// Packed into `shadow_bounce.w` as `f32::from_bits`; shaders `asuint` the lane.
pub(crate) const LANE_BIT_SHADOWS: u32 = 1;
pub(crate) const LANE_BIT_BLOCKLIGHT: u32 = 2;
pub(crate) const LANE_BIT_AMBIENT: u32 = 4;

pub(crate) fn lane_enable_bits(f: &RenderFlags) -> f32 {
    let mut bits = 0u32;
    if f.shadows {
        bits |= LANE_BIT_SHADOWS;
    }
    if f.blocklight {
        bits |= LANE_BIT_BLOCKLIGHT;
    }
    if f.ambient {
        bits |= LANE_BIT_AMBIENT;
    }
    f32::from_bits(bits)
}

/// Compile-time lean opaque/LOD fragment: every optional lighting lane is off
/// and fog is off. Matches the runtime path `shadow_bounce.w` bits == 0 and
/// `frame.horizon.w == 0`.
pub(crate) fn mesh_lean(flags: &RenderFlags) -> bool {
    lane_enable_bits(flags).to_bits() == 0 && !flags.fog
}

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

/// Rec.709 relative luminance. Twin of `luma709` in `common.slang`.
fn luma709(c: Vec3) -> f32 {
    c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

impl FrameUniformsExt {
    /// Hoist per-frame uniform-only math the shaders used to recompute every
    /// fragment: ambient floor colour, sky-halo exponent, halo tint×scale,
    /// the shadow-fallback day factor, the sky-dome shadow fill, and the
    /// shadows/blocklight/ambient lane-enable bits in `shadow_bounce.w`.
    /// Mirrors `common.slang`.
    pub(crate) fn derive(u: FrameUniformsGpu, flags: RenderFlags) -> Self {
        let zenith = Vec3::new(u.zenith[0], u.zenith[1], u.zenith[2]);
        let light = Vec3::new(u.light[0], u.light[1], u.light[2]);
        let floor = u.candle[3];
        let zl = luma709(zenith);
        let ambient = if zl > 1e-6 {
            zenith * (floor / zl)
        } else {
            Vec3::splat(floor)
        };
        let t = glow_smoothstep(GLOW_EDGE0, GLOW_EDGE1, u.sun_dir_elev[3]);
        let glow_pow = GLOW_POW_SUNSET + (GLOW_POW_DAY - GLOW_POW_SUNSET) * t;
        let glow_rgb = light * (0.5 + u.zenith[3]);
        let day = (u.light[3] * 2.0 - 1.0).abs();
        let bounce_src = if zl > 1e-6 {
            light.lerp(zenith * (luma709(light) / zl), SHADOW_BOUNCE_TINT)
        } else {
            light
        };
        let bounce = bounce_src * SHADOW_SKY_AMBIENT;
        Self {
            base: u,
            ambient_glow: [ambient.x, ambient.y, ambient.z, glow_pow],
            glow_day: [glow_rgb.x, glow_rgb.y, glow_rgb.z, day],
            shadow_bounce: [bounce.x, bounce.y, bounce.z, lane_enable_bits(&flags)],
        }
    }
}

// Bumped when the GPU `FrameUniforms` layout changes (public prefix extras.yz
// sky lanes, or the engine-derived tail).
pub const FRAME_UNIFORMS_VERSION: u32 = 6;

/// The per-frame UBO ring. Indexed only by [`FrameSlot`],
/// so raw-usize slot confusion is inexpressible here.
pub(crate) struct UboRing {
    bufs: PerSlot<HostBuffer>,
    last: PerSlot<Option<FrameUniformsExt>>,
    last_gpu: PerSlot<Option<(FrameUniformsGpu, RenderFlags)>>,
}

impl UboRing {
    /// Allocate every slot's UBO, each sized to the GPU struct (public wire +
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
            bufs: PerSlot::new(std::array::from_fn(|_| make())),
            last: PerSlot::new(std::array::from_fn(|_| None)),
            last_gpu: PerSlot::new(std::array::from_fn(|_| None)),
        }
    }

    /// Copy this frame's already-derived uniforms (176-byte `FrameUniformsExt`)
    /// into `slot`'s mapped buffer. Coherent memory: the write is visible to
    /// the GPU with no explicit flush. `prepare_derived` runs once on the
    /// producer (begin_3d / full_bright); `FrameUniformsExt::derive` runs once
    /// on the render thread before this write. Identical bytes for this slot
    /// skip the map write.
    pub(crate) fn write(&mut self, slot: FrameSlot, ext: &FrameUniformsExt) {
        if self.last[slot].as_ref() == Some(ext) {
            return;
        }
        unsafe { self.bufs[slot].write(0, bytemuck::bytes_of(ext)) };
        self.last[slot] = Some(*ext);
    }

    /// Derive the engine tail and write, skipping both when this slot already
    /// holds `u` (sky/lighting-dependent work independent of jittered view-proj).
    pub(crate) fn write_from_gpu(
        &mut self,
        slot: FrameSlot,
        u: FrameUniformsGpu,
        flags: RenderFlags,
    ) {
        if self.last_gpu[slot] == Some((u, flags)) {
            return;
        }
        self.write(slot, &FrameUniformsExt::derive(u, flags));
        self.last_gpu[slot] = Some((u, flags));
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
        for buf in self.bufs.iter_mut() {
            unsafe { buf.destroy(device) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RenderFlags;

    fn derive_default(u: FrameUniformsGpu) -> FrameUniformsExt {
        FrameUniformsExt::derive(u, RenderFlags::default())
    }

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

    /// Public wire stays 8 float4s so the game's `From<&FrameSnapshot>` layout
    /// cannot silently grow; derived lanes live past that prefix.
    #[test]
    fn public_wire_stays_eight_lanes() {
        assert_eq!(size_of::<FrameUniformsGpu>(), 128);
        assert_eq!(size_of::<FrameUniformsExt>(), 176);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, ambient_glow), 128);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, glow_day), 144);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, shadow_bounce), 160);
    }

    #[test]
    fn derived_lanes_match_shader_formulas() {
        let mut u = FrameUniformsGpu::full_bright();
        u.zenith = [0.09, 0.22, 0.45, 2.0];
        u.candle[3] = 0.30;
        u.light = [1.25, 1.15, 1.0, 1.0];
        u.sun_dir_elev[3] = 0.2;

        let ext = derive_default(u);
        let luma = luma709(Vec3::new(0.09, 0.22, 0.45));
        let scale = 0.30 / luma;
        for i in 0..3 {
            assert!(
                (ext.ambient_glow[i] - u.zenith[i] * scale).abs() < 1e-6,
                "ambient[{i}]"
            );
        }
        let t = glow_smoothstep(GLOW_EDGE0, GLOW_EDGE1, 0.2);
        let glow_pow = GLOW_POW_SUNSET + (GLOW_POW_DAY - GLOW_POW_SUNSET) * t;
        assert!((ext.ambient_glow[3] - glow_pow).abs() < 1e-6);
        let glow_scale = 0.5 + 2.0;
        assert!((ext.glow_day[0] - 1.25 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[1] - 1.15 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[2] - 1.0 * glow_scale).abs() < 1e-6);
        assert!((ext.glow_day[3] - 1.0).abs() < 1e-7);

        let light = Vec3::new(1.25, 1.15, 1.0);
        let zenith = Vec3::new(0.09, 0.22, 0.45);
        let bounce_src = light.lerp(
            zenith * (luma709(light) / luma709(zenith)),
            SHADOW_BOUNCE_TINT,
        );
        let bounce = bounce_src * SHADOW_SKY_AMBIENT;
        for i in 0..3 {
            assert!(
                (ext.shadow_bounce[i] - bounce[i]).abs() < 1e-6,
                "shadow_bounce[{i}]"
            );
        }
        assert_eq!(
            ext.shadow_bounce[3].to_bits(),
            lane_enable_bits(&RenderFlags::default()).to_bits()
        );
    }

    #[test]
    fn mesh_lean_when_every_optional_lane_is_off() {
        let all_off = RenderFlags {
            shadows: false,
            blocklight: false,
            ambient: false,
            fog: false,
            ..RenderFlags::default()
        };
        assert!(mesh_lean(&all_off));
        assert_eq!(lane_enable_bits(&all_off).to_bits(), 0);
        assert!(!mesh_lean(&RenderFlags {
            fog: true,
            ..all_off
        }));
        assert!(!mesh_lean(&RenderFlags {
            shadows: true,
            ..all_off
        }));
        assert!(!mesh_lean(&RenderFlags {
            blocklight: true,
            ..all_off
        }));
        assert!(!mesh_lean(&RenderFlags {
            ambient: true,
            ..all_off
        }));
        assert!(!mesh_lean(&RenderFlags::default()));
    }

    #[test]
    fn lane_enable_bits_pack_shadows_blocklight_ambient() {
        let mut f = RenderFlags::default();
        f.shadows = false;
        f.blocklight = false;
        f.ambient = false;
        assert_eq!(lane_enable_bits(&f).to_bits(), 0);
        f.shadows = true;
        assert_eq!(lane_enable_bits(&f).to_bits(), LANE_BIT_SHADOWS);
        f.blocklight = true;
        assert_eq!(
            lane_enable_bits(&f).to_bits(),
            LANE_BIT_SHADOWS | LANE_BIT_BLOCKLIGHT
        );
        f.ambient = true;
        assert_eq!(
            lane_enable_bits(&f).to_bits(),
            LANE_BIT_SHADOWS | LANE_BIT_BLOCKLIGHT | LANE_BIT_AMBIENT
        );
        let ext = FrameUniformsExt::derive(FrameUniformsGpu::full_bright(), f);
        assert_eq!(
            ext.shadow_bounce[3].to_bits(),
            lane_enable_bits(&f).to_bits()
        );
    }

    /// `SHADOW_BOUNCE_TINT == 0` ⇒ lane is `SHADOW_SKY_AMBIENT * light.rgb`.
    /// Skipped when this build selected the new (non-zero) tint.
    #[test]
    fn shadow_bounce_is_sun_scaled_when_tint_is_zero() {
        if SHADOW_BOUNCE_TINT.abs() > 1e-8 {
            return;
        }
        let mut u = FrameUniformsGpu::full_bright();
        u.light = [1.25, 1.15, 1.0, 1.0];
        u.zenith = [0.09, 0.22, 0.45, 2.0];
        let ext = derive_default(u);
        for i in 0..3 {
            assert!(
                (ext.shadow_bounce[i] - u.light[i] * SHADOW_SKY_AMBIENT).abs() < 1e-6,
                "shadow_bounce[{i}]"
            );
        }
        assert_eq!(
            ext.shadow_bounce[3].to_bits(),
            lane_enable_bits(&RenderFlags::default()).to_bits()
        );
    }

    #[test]
    fn derived_ambient_falls_back_when_zenith_is_black() {
        let u = FrameUniformsGpu::full_bright();
        // full_bright leaves zenith at zero, candle.w = 1.
        let ext = derive_default(u);
        assert_eq!(&ext.ambient_glow[..3], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn derived_day_factor_tracks_night_and_noon() {
        let mut u = FrameUniformsGpu::full_bright();
        u.light[3] = 0.0;
        assert!((derive_default(u).glow_day[3] - 1.0).abs() < 1e-7);
        u.light[3] = 0.5;
        assert!(derive_default(u).glow_day[3].abs() < 1e-7);
        u.light[3] = 1.0;
        assert!((derive_default(u).glow_day[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn derived_glow_pow_is_sunset_below_edge0_and_day_above_edge1() {
        let mut u = FrameUniformsGpu::full_bright();
        u.sun_dir_elev[3] = GLOW_EDGE0 - 1.0;
        assert!((derive_default(u).ambient_glow[3] - GLOW_POW_SUNSET).abs() < 1e-6);
        u.sun_dir_elev[3] = GLOW_EDGE1 + 1.0;
        assert!((derive_default(u).ambient_glow[3] - GLOW_POW_DAY).abs() < 1e-6);
    }
}
