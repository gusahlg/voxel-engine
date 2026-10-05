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

/// Per-frame local sky frame. `None` on [`crate::frame::DrawLists`] means
/// [`LocalFrame::default`]: up `+Y`, altitude `0`.
#[derive(Clone, Copy, PartialEq)]
pub(crate) struct LocalFrame {
    pub up: Vec3,
    pub altitude: f32,
}

impl Default for LocalFrame {
    fn default() -> Self {
        Self { up: Vec3::Y, altitude: 0.0 }
    }
}

/// Orthonormal sky basis `(tangent, up, bitangent)`.
///
/// `tangent` is the world axis least aligned with `up` (ties break toward X,
/// then Y, then Z), rejected into the plane. `bitangent = tangent × up`.
/// Exact `+Y` yields the identity basis, so the old world-Y sky math is unchanged.
pub fn local_sky_basis(up: Vec3) -> (Vec3, Vec3, Vec3) {
    let len2 = up.length_squared();
    if len2 <= 1e-12 {
        return (Vec3::X, Vec3::Y, Vec3::Z);
    }
    let up = up * len2.sqrt().recip();
    let (ax, ay, az) = (up.x.abs(), up.y.abs(), up.z.abs());
    let axis = if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let tangent = axis - up * axis.dot(up);
    let tlen2 = tangent.length_squared();
    debug_assert!(tlen2 > 1e-12, "least-aligned axis is never parallel to up");
    let tangent = tangent * tlen2.sqrt().recip();
    let bitangent = tangent.cross(up);
    (tangent, up, bitangent)
}

impl FrameUniformsExt {
    /// Hoist per-frame uniform-only math the shaders used to recompute every
    /// fragment: ambient floor colour, sky-halo exponent, halo tint×scale,
    /// the shadow-fallback day factor, the sky-dome shadow fill, and the
    /// shadows/blocklight/ambient lane-enable bits in `shadow_bounce.w`,
    /// and the local sky basis. Mirrors `common.slang`.
    pub(crate) fn derive(u: FrameUniformsGpu, flags: RenderFlags, local: LocalFrame) -> Self {
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
        let (tangent, up, bitangent) = local_sky_basis(local.up);
        Self {
            base: u,
            ambient_glow: [ambient.x, ambient.y, ambient.z, glow_pow],
            glow_day: [glow_rgb.x, glow_rgb.y, glow_rgb.z, day],
            shadow_bounce: [bounce.x, bounce.y, bounce.z, lane_enable_bits(&flags)],
            sky_tangent: [tangent.x, tangent.y, tangent.z, 0.0],
            sky_up: [up.x, up.y, up.z, local.altitude],
            sky_bitangent: [bitangent.x, bitangent.y, bitangent.z, 0.0],
            morph_eye_block: [0; 4],
            morph_eye_frac: [0.0; 4],
            morph_band: [[0.0; 4]; 16],
        }
    }

    /// Copy the LOD-morph tail. `derive` leaves it zero (morphing off).
    pub(crate) fn apply_lod_morph(&mut self, morph: LodMorphGpu) {
        self.morph_eye_block = morph.eye_block;
        self.morph_eye_frac = morph.eye_frac;
        self.morph_band = morph.bands;
    }
}

/// Bit 0 of `morph_eye_block.w`: this frame's LOD morph bands are live.
pub(crate) const LOD_MORPH_FLAG: i32 = 1;

/// Engine-derived LOD morph block appended after the float4 lanes.
/// xyz of `eye_block` / `eye_frac` are the camera-style split of `set_lod_morph`'s
/// eye; `eye_block.w` bit 0 is [`LOD_MORPH_FLAG`]. `bands[k]` is `(half.xyz, start)`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LodMorphGpu {
    pub eye_block: [i32; 4],
    pub eye_frac: [f32; 4],
    pub bands: [[f32; 4]; 16],
}

// Bumped when the GPU `FrameUniforms` layout changes (public prefix or the
// engine-derived tail). 8 adds the LOD morph tail (eye split + 16 bands).
pub const FRAME_UNIFORMS_VERSION: u32 = 8;

/// The per-frame UBO ring. Indexed only by [`FrameSlot`],
/// so raw-usize slot confusion is inexpressible here.
pub(crate) struct UboRing {
    bufs: PerSlot<HostBuffer>,
    last: PerSlot<Option<FrameUniformsExt>>,
    last_gpu: PerSlot<Option<(FrameUniformsGpu, RenderFlags, LocalFrame, LodMorphGpu)>>,
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

    /// Copy this frame's already-derived uniforms (`FrameUniformsExt`, public
    /// wire plus the basis tail) into `slot`'s mapped buffer. Coherent memory:
    /// the write is visible to the GPU with no explicit flush. `prepare_derived`
    /// runs once on the producer (begin_3d / full_bright); `FrameUniformsExt::derive`
    /// runs once on the render thread before this write. Identical bytes for
    /// this slot skip the map write.
    pub(crate) fn write(&mut self, slot: FrameSlot, ext: &FrameUniformsExt) {
        if self.last[slot].as_ref() == Some(ext) {
            return;
        }
        unsafe { self.bufs[slot].write(0, bytemuck::bytes_of(ext)) };
        self.last[slot] = Some(*ext);
    }

    /// Derive the engine tail and write, skipping both when this slot already
    /// holds `u` and `morph` (sky/lighting-dependent work independent of jittered view-proj).
    pub(crate) fn write_from_gpu(
        &mut self,
        slot: FrameSlot,
        u: FrameUniformsGpu,
        flags: RenderFlags,
        local: LocalFrame,
        morph: LodMorphGpu,
    ) {
        if self.last_gpu[slot] == Some((u, flags, local, morph)) {
            return;
        }
        let mut ext = FrameUniformsExt::derive(u, flags, local);
        ext.apply_lod_morph(morph);
        self.write(slot, &ext);
        self.last_gpu[slot] = Some((u, flags, local, morph));
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
        FrameUniformsExt::derive(u, RenderFlags::default(), LocalFrame::default())
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
        assert_eq!(size_of::<FrameUniformsExt>(), 512);
        assert_eq!(size_of::<LodMorphGpu>(), 288);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, ambient_glow), 128);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, glow_day), 144);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, shadow_bounce), 160);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, sky_tangent), 176);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, sky_up), 192);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, sky_bitangent), 208);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, morph_eye_block), 224);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, morph_eye_frac), 240);
        assert_eq!(std::mem::offset_of!(FrameUniformsExt, morph_band), 256);
        assert_eq!(size_of::<FrameUniformsExt>() % 16, 0);
        assert!(size_of::<FrameUniformsExt>() <= 16384);
    }

    #[test]
    fn derive_zeroes_lod_morph_and_apply_writes_the_tail() {
        let ext = derive_default(FrameUniformsGpu::full_bright());
        assert_eq!(ext.morph_eye_block, [0; 4]);
        assert_eq!(ext.morph_eye_frac, [0.0; 4]);
        assert_eq!(ext.morph_band, [[0.0; 4]; 16]);
        assert_eq!(ext.base.extras[0], 1.0);
        let mut morph = LodMorphGpu::default();
        morph.eye_block = [9, -3, 4, LOD_MORPH_FLAG];
        morph.eye_frac = [0.25, 0.5, 0.75, 0.0];
        morph.bands[3] = [10.0, 20.0, 30.0, 0.5];
        let mut ext = ext;
        ext.apply_lod_morph(morph);
        assert_eq!(ext.morph_eye_block, morph.eye_block);
        assert_eq!(ext.morph_eye_frac, morph.eye_frac);
        assert_eq!(ext.morph_band[3], morph.bands[3]);
        assert_eq!(ext.morph_band[2], [0.0; 4]);
        assert_eq!(ext.base.extras[0], 1.0);
    }

    #[test]
    fn plus_y_basis_is_the_identity() {
        let (t, u, b) = local_sky_basis(Vec3::Y);
        assert_eq!(t, Vec3::X);
        assert_eq!(u, Vec3::Y);
        assert_eq!(b, Vec3::Z);
        let ray = Vec3::new(0.2, 0.9, -0.3);
        let local = Vec3::new(ray.dot(t), ray.dot(u), ray.dot(b));
        assert_eq!(local, ray);
    }

    #[test]
    fn basis_is_right_handed_on_every_axis_and_a_diagonal() {
        for up in [
            Vec3::X,
            -Vec3::X,
            -Vec3::Y,
            Vec3::Z,
            -Vec3::Z,
            Vec3::ONE.normalize(),
            Vec3::new(0.2, 0.9, -0.1),
        ] {
            let (t, u, b) = local_sky_basis(up);
            assert!((u - up.normalize()).length() < 1e-5, "up {up:?}");
            assert!((t.length() - 1.0).abs() < 1e-5 && (b.length() - 1.0).abs() < 1e-5);
            assert!(t.dot(u).abs() < 1e-5 && b.dot(u).abs() < 1e-5 && t.dot(b).abs() < 1e-5);
            assert!((t.cross(u) - b).length() < 1e-5, "right-handed {up:?}");
        }
        let (t, u, b) = local_sky_basis(Vec3::X);
        assert_eq!(u, Vec3::X);
        assert_eq!(t, Vec3::Y);
        assert_eq!(b, -Vec3::Z);
        // A ray straight up the +X face is local +Y, so the old `.y` sky math stands.
        assert!((Vec3::X.dot(u) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn zero_up_falls_back_to_plus_y() {
        assert_eq!(local_sky_basis(Vec3::ZERO), local_sky_basis(Vec3::Y));
        assert_eq!(local_sky_basis(Vec3::Y * 4.0), local_sky_basis(Vec3::Y));
    }

    #[test]
    fn derive_writes_the_basis_and_leaves_anim_w_alone() {
        let mut u = FrameUniformsGpu::full_bright();
        u.anim[3] = 42.0;
        let ext = FrameUniformsExt::derive(
            u,
            RenderFlags::default(),
            LocalFrame { up: Vec3::Y, altitude: 7.0 },
        );
        assert_eq!(ext.sky_tangent, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(ext.sky_up, [0.0, 1.0, 0.0, 7.0]);
        assert_eq!(ext.sky_bitangent, [0.0, 0.0, 1.0, 0.0]);
        assert_eq!(ext.base.anim[3], 42.0);
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
        let ext = FrameUniformsExt::derive(FrameUniformsGpu::full_bright(), f, LocalFrame::default());
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
