//! Temporal anti-aliasing / upsampling: history images, reprojection, present resolve.
//!
//! The scene is rendered with a per-frame sub-pixel jitter (Halton(2,3), applied
//! ONLY to the mesh view-proj at push-constant packing — [`super::jittered_clip`]).
//! Raster pixel = clean position + `jitter_px` (pixel-y-down), so render texel
//! `t` (centre `t+0.5`) holds the scene at the unjittered position
//! `s_t = t + 0.5 - jitter_px`. The fused present-time tonemap (`-DTAA_FUSED`)
//! reconstructs each output pixel from a 3×3 of those texels with a Gaussian
//! in output pixels (UE σ = 0.47, `w = exp(-2.29 |d_out|²)`, `d_out` is
//! `(s_t - p)` scaled by `output_extent / render_extent` per axis). At ratio 1
//! that is the previous 0.47-render-px kernel; when upsampling it stays 0.47
//! output px so `cur` is not a 2×-wide low-pass. Neighbourhood-clamps in YCoCg,
//! reprojects the previous *presented* frame with 5-tap Catmull-Rom history
//! (Jimenez 2016), and blends with current-frame weight `(1-blend)*w_max`. A
//! Karis 2014 velocity-weighted boost is a runtime push-constant defaulting to
//! 1.0 (off) so camera rotation does not re-inject the low-res `cur`; override
//! once at startup with `VOXEL_TAA_MOTION_BOOST` / `VOXEL_TAA_MOTION_PX`.
//!
//! No full-resolution TAA compute pass runs per rendered frame. TAA work happens
//! only on presented frames (mailbox drops skip it). History at swapchain extent
//! makes TAA the render-scale up/downsampler.
//!
//! One ping-pong pair total: presents are serialized by the copy timeline
//! value. Recreate / resize / TAA toggle / the first present clear
//! `history_valid` so no garbage bleeds in.
//!
//! Reprojection is depth-aware: the host f64-composes `prev * inv(cur)` into a
//! single push-constant matrix relating this presented camera to the previously
//! presented camera (not the previous rendered frame). Depth is point-sampled
//! from the render-res sampleable depth, which already rests in
//! [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`].

use ash::vk;
use glam::{DMat4, DVec3, Mat4, Vec2};

use super::image::{
    AllocError, ImageDesc, ImageResource, LayoutUse, create_image_array, image_purpose,
};

/// The view-projection without jitter. Jittered matrix is applied privately
/// at push-constant packing only so culling and TAA reprojection stay stable.
#[derive(Clone, Copy, Debug)]
pub struct CleanViewProj(pub Mat4);

/// Sub-pixel camera jitter in pixels (±0.5). Converted to NDC privately.
#[derive(Clone, Copy, Debug, Default)]
pub struct JitterOffset(pub Vec2);

impl JitterOffset {
    pub const ZERO: JitterOffset = JitterOffset(Vec2::ZERO);
}

/// Length of the jitter sequence (shared with shaders).
pub const TEMPORAL_SEQ_LEN: u64 = 16;

/// Halton(2,3) − 0.5 sequence from generated constants.
pub fn jitter_at(frame_index: u64) -> JitterOffset {
    let e = crate::genconst::HALTON_23[(frame_index % TEMPORAL_SEQ_LEN) as usize];
    JitterOffset(Vec2::new(e[0], e[1]))
}

/// History format matches the HDR target (linear, not sRGB).
pub const TAA_HISTORY_FORMAT: ash::vk::Format = ash::vk::Format::R16G16B16A16_SFLOAT;

/// Legacy compute-pass bindings, kept for the public `skeleton` re-export.
/// The fused tonemap set is HDR=0, spill=1, history=2, depth=3.
#[deprecated(
    note = "TAA now resolves in the present-time tonemap fragment; these compute-pass bindings are unused"
)]
pub const TAA_RESOLVE_CURRENT_BINDING: u32 = 0;
#[deprecated(
    note = "TAA now resolves in the present-time tonemap fragment; these compute-pass bindings are unused"
)]
pub const TAA_RESOLVE_HISTORY_BINDING: u32 = 1;

/// Fused tonemap descriptor bindings (`layout_tonemap_taa`).
pub(crate) const TONEMAP_TAA_HDR_BINDING: u32 = 0;
pub(crate) const TONEMAP_TAA_SPILL_BINDING: u32 = 1;
pub(crate) const TONEMAP_TAA_HISTORY_BINDING: u32 = 2;
pub(crate) const TONEMAP_TAA_DEPTH_BINDING: u32 = 3;

/// Reprojection inputs. `prev` is clean (un-jittered) to avoid ghosting.
pub struct Reprojection {
    pub prev: CleanViewProj,
    pub camera_delta: DVec3,
}

impl Reprojection {
    /// Compose `prev * translate(-delta) * inverse(cur)` in f64, then narrow.
    pub fn matrix(&self, cur: Mat4) -> Mat4 {
        compose_reproj(self.prev.0, self.camera_delta, cur)
    }
}

fn compose_reproj(prev_vp: Mat4, camera_delta: DVec3, cur: Mat4) -> Mat4 {
    let prev = prev_vp.as_dmat4() * DMat4::from_translation(-camera_delta);
    (prev * cur.as_dmat4().inverse()).as_mat4()
}

/// History feedback weight from genconst; fraction of clamped history kept
/// each present.
use crate::genconst::HISTORY_BLEND;

/// Default Karis velocity-weighted current-frame boost. 1.0 disables the
/// term (`lerp(cur_w, cur_w, …)` is a no-op). Override with
/// `VOXEL_TAA_MOTION_BOOST`.
const DEFAULT_MOTION_BOOST: f32 = 1.0;
/// Output-pixel velocity at which the (optional) motion boost saturates.
/// Override with `VOXEL_TAA_MOTION_PX`.
const DEFAULT_MOTION_PX: f32 = 8.0;

fn parse_f32_or(raw: Option<&str>, default: f32) -> f32 {
    raw.and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    parse_f32_or(std::env::var(name).ok().as_deref(), default)
}

/// Camera of the frame being presented, used to compose the reprojection
/// matrix against the previously *presented* camera.
pub(crate) struct TaaPresent {
    pub view_proj: Mat4,
    pub eye: DVec3,
    pub jitter: Vec2,
}

/// Push constants for `tonemap.frag` compiled with `-DTAA_FUSED`. Layout is
/// mirrored one-to-one in Slang. Fits the 128-byte min `maxPushConstantsSize`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TonemapTaaPush {
    pub reproj: [[f32; 4]; 4],
    pub render_extent: [f32; 2],
    pub jitter_px: [f32; 2],
    pub output_extent: [f32; 2],
    pub exposure: f32,
    pub s: f32,
    pub atan_s: f32,
    pub vignette: f32,
    pub blend: f32,
    pub history_valid: u32,
    pub depth_valid: u32,
    pub motion_boost: f32,
    pub motion_px: f32,
}

const _: () = assert!(size_of::<TonemapTaaPush>() <= 128);
const _: () = assert!(size_of::<TonemapTaaPush>() == 124);
const _: () = assert!(std::mem::offset_of!(TonemapTaaPush, output_extent) == 80);
const _: () = assert!(std::mem::offset_of!(TonemapTaaPush, motion_boost) == 116);

fn create_history_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
) -> Result<ImageResource, AllocError> {
    let desc = ImageDesc {
        extent,
        format: TAA_HISTORY_FORMAT,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        layers: 1,
        aspect: vk::ImageAspectFlags::COLOR,
        samples: vk::SampleCountFlags::TYPE_1,
    };
    ImageResource::create(
        device,
        memory_props,
        &desc,
        &image_purpose("TAA history", extent, vk::SampleCountFlags::TYPE_1),
    )
}

/// Render-thread owner of the swapchain-sized history pair and the previous
/// presented camera. The fused resolve itself lives in the tonemap pipeline.
///
/// Validation cases (all by construction):
/// - **First present / after recreate**: both images are UNDEFINED; the write
///   side is discarded into COLOR_ATTACHMENT, the read side is promoted to
///   SHADER_READ_ONLY so the descriptor is valid, and `history_valid` is 0 so
///   the shader does not sample it.
/// - **TAA toggled** (`set_flags`): [`TaaState::invalidate_history`] clears
///   `valid` and `prev`; the next present is a first-present.
/// - **Resize / swapchain recreate**: history is rebuilt at the new swapchain
///   extent and invalidated (contents discarded; reconverges).
/// - **Render-scale change**: history stays swapchain-sized so the images are
///   kept; [`TaaState::invalidate_history`] still drops temporal state so the
///   new reconstruction kernel does not mix with the previous sample grid.
/// - **MSAA on**: depth is the SAMPLE_ZERO resolve target, already resting in
///   [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`]; sampled with no further transition.
/// - **`render_scale != 1`**: history is swapchain-sized, current/depth are
///   render-sized; `source_uv = warpSampleUv(uv_out)` maps the two spaces, and
///   the 3×3 Gaussian reconstructs (or downsamples) at the output pixel.
/// - **Wide FOV**: history is stored in presented space, so a reprojected
///   source uv is converted with `unwarpSampleUv` before the history tap.
pub(crate) struct TaaState {
    /// Ping-pong integrator (single logical history, NOT per-slot). Presents
    /// are serialized by the copy timeline, so one pair is enough.
    history: [ImageResource; 2],
    /// Index of the image holding the last *presented* resolved output (this
    /// present's history source); the other is written this present, then
    /// becomes the source.
    read_idx: usize,
    extent: vk::Extent2D,
    /// False until at least one present has populated `read_idx` (and after a
    /// resize or TAA toggle discards history).
    valid: bool,
    /// Previous *presented* frame's view-proj (without jitter) + render-space-
    /// origin world position (f64), for reprojection.
    prev: Option<(Mat4, DVec3)>,
    /// Karis velocity-weighted current-frame boost. 1.0 = off (default).
    motion_boost: f32,
    /// Output-pixel velocity that saturates `motion_boost`. Default 8.0.
    motion_px: f32,
}

impl TaaState {
    pub(crate) fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        swapchain_extent: vk::Extent2D,
    ) -> Result<TaaState, AllocError> {
        Ok(TaaState {
            history: create_image_array(device, || {
                create_history_image(device, memory_props, swapchain_extent)
            })?,
            read_idx: 0,
            extent: swapchain_extent,
            valid: false,
            prev: None,
            motion_boost: env_f32("VOXEL_TAA_MOTION_BOOST", DEFAULT_MOTION_BOOST),
            motion_px: env_f32("VOXEL_TAA_MOTION_PX", DEFAULT_MOTION_PX),
        })
    }

    /// Rebuild history images after swapchain recreate/resize (contents
    /// discarded, reconverges). Same swapchain extent (render-scale-only
    /// apply) keeps the images and only invalidates temporal state.
    ///
    /// On allocation failure the previous images are left in place and
    /// temporal state is invalidated so the next present does not sample
    /// a mismatched history.
    pub(crate) fn recreate(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        swapchain_extent: vk::Extent2D,
    ) -> Result<(), AllocError> {
        if self.extent.width != swapchain_extent.width
            || self.extent.height != swapchain_extent.height
        {
            let history = match create_image_array(device, || {
                create_history_image(device, memory_props, swapchain_extent)
            }) {
                Ok(history) => history,
                Err(err) => {
                    self.invalidate_history();
                    return Err(err);
                }
            };
            for h in &self.history {
                unsafe { h.destroy(device) };
            }
            self.history = history;
            self.read_idx = 0;
            self.extent = swapchain_extent;
        }
        self.invalidate_history();
        Ok(())
    }

    /// Reset temporal state (called on TAA toggle to prevent history ghosting).
    pub(crate) fn invalidate_history(&mut self) {
        self.valid = false;
        self.prev = None;
    }

    pub(super) fn write_view(&self) -> vk::ImageView {
        self.history[1 - self.read_idx].view()
    }

    pub(super) fn read_view(&self) -> vk::ImageView {
        self.history[self.read_idx].view()
    }

    /// Compose the present-time reprojection matrix against the previously
    /// presented camera. Identity when this is the first present after an
    /// invalidate (shader ignores it via `history_valid`).
    pub(super) fn reprojection(&self, cur: Mat4, eye: DVec3) -> Mat4 {
        match self.prev {
            Some((prev_vp, prev_eye)) => Reprojection {
                prev: CleanViewProj(prev_vp),
                camera_delta: prev_eye - eye,
            }
            .matrix(cur),
            None => Mat4::IDENTITY,
        }
    }

    pub(super) fn history_valid(&self) -> bool {
        self.valid
    }

    /// Write-history → COLOR_ATTACHMENT (discard: fully overwritten) and, if
    /// the read side is still UNDEFINED, promote it to SHADER_READ_ONLY so the
    /// fused shader's history sampler is a valid descriptor even when
    /// `history_valid` is 0.
    pub(super) fn history_pre_barriers(
        &mut self,
    ) -> (
        vk::ImageMemoryBarrier2<'static>,
        Option<vk::ImageMemoryBarrier2<'static>>,
    ) {
        let r = self.read_idx;
        let w = 1 - r;
        let write = self.history[w].barrier_to(LayoutUse::ColorAttachmentWrite, true);
        let read = if self.history[r].layout() != vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            Some(self.history[r].barrier_to(LayoutUse::FragmentSampled, true))
        } else {
            None
        };
        (write, read)
    }

    /// Write-history COLOR_ATTACHMENT → SHADER_READ_ONLY (next present's read).
    pub(super) fn history_post_barrier(&mut self) -> vk::ImageMemoryBarrier2<'static> {
        let w = 1 - self.read_idx;
        self.history[w].barrier_to(LayoutUse::FragmentSampled, false)
    }

    /// After the present copy is recorded: the just-written image becomes the
    /// next present's history, and this presented camera is stored as `prev`.
    pub(super) fn finish_present(&mut self, cur: Mat4, eye: DVec3) {
        self.read_idx = 1 - self.read_idx;
        self.valid = true;
        self.prev = Some((cur, eye));
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            for h in &self.history {
                h.destroy(device);
            }
        }
    }
}

impl super::Renderer {
    /// Push constants for the fused TAA tonemap draw of this present.
    pub(super) fn tonemap_taa_push(
        &self,
        taa: &TaaPresent,
        warp: crate::camera::WarpPush,
    ) -> TonemapTaaPush {
        let reproj = self.taa.reprojection(taa.view_proj, taa.eye);
        let render = self.render_extent;
        let output = self.swapchain.extent;
        TonemapTaaPush {
            reproj: reproj.to_cols_array_2d(),
            render_extent: [render.width as f32, render.height as f32],
            jitter_px: taa.jitter.to_array(),
            output_extent: [output.width as f32, output.height as f32],
            exposure: warp.exposure,
            s: warp.s,
            atan_s: warp.atan_s,
            vignette: warp.vignette,
            blend: HISTORY_BLEND,
            history_valid: self.taa.history_valid() as u32,
            // Depth is sampleable at every sample count (MSAA resolves it).
            depth_valid: 1,
            motion_boost: self.taa.motion_boost,
            motion_px: self.taa.motion_px,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Jitter sequence is centered and bounded (±0.5 px).
    #[test]
    fn jitter_sequence_centered_and_bounded() {
        let mut sum = Vec2::ZERO;
        for f in 0..TEMPORAL_SEQ_LEN {
            let j = jitter_at(f).0;
            assert!(j.x.abs() <= 0.5 && j.y.abs() <= 0.5);
            sum += j;
        }
        assert!(sum.length() < 0.1 * TEMPORAL_SEQ_LEN as f32);
    }

    /// Same camera → identity reprojection (f64 compose, then narrow).
    #[test]
    fn compose_reproj_identity_when_camera_unmoved() {
        let vp = Mat4::from_translation(glam::Vec3::new(10.0, 2.0, -4.0));
        let eye = DVec3::new(10.0, 2.0, -4.0);
        let m = compose_reproj(vp, DVec3::ZERO, vp);
        let ident = Reprojection {
            prev: CleanViewProj(vp),
            camera_delta: eye - eye,
        }
        .matrix(vp);
        let m_cols = m.to_cols_array_2d();
        let ident_cols = ident.to_cols_array_2d();
        for col in 0..4 {
            for row in 0..4 {
                let expected = if col == row { 1.0 } else { 0.0 };
                assert!((m_cols[col][row] - expected).abs() < 1e-5);
                assert!((ident_cols[col][row] - expected).abs() < 1e-5);
            }
        }
    }

    /// Identity prev/cur with a camera delta is a translation by -delta.
    #[test]
    fn compose_reproj_applies_camera_delta_as_translation() {
        let cur = Mat4::IDENTITY;
        let prev = Mat4::IDENTITY;
        let delta = DVec3::new(1.0 + 1e-8, 0.0, 0.0);
        let m = compose_reproj(prev, delta, cur);
        // translate(-delta) on identity prev, times identity inv(cur) → translation -delta.
        let t = m.w_axis;
        assert!((t.x + delta.x as f32).abs() < 1e-6);
        assert!(t.y.abs() < 1e-6 && t.z.abs() < 1e-6);
    }

    #[test]
    fn fused_push_fits_vulkan_minimum() {
        assert!(size_of::<TonemapTaaPush>() <= 128);
        assert_eq!(size_of::<TonemapTaaPush>(), 124);
        assert_eq!(std::mem::offset_of!(TonemapTaaPush, motion_boost), 116);
        assert_eq!(std::mem::offset_of!(TonemapTaaPush, motion_px), 120);
    }

    /// Raster texel `t` holds `s_t = t + 0.5 - jitter_px`. Nearest texel of
    /// continuous render-space `p` is `tc = floor(p + jitter_px)`. Sample
    /// distance is measured in output pixels:
    /// `d_out = (s_t - p) * (output_extent / render_extent)`;
    /// `w = exp(-2.29 |d_out|²)` (UE σ = 0.47 output px). At ratio 1, `d_out`
    /// equals `s_t - p` (bit-identical to the old render-space kernel).
    #[test]
    fn taau_reconstruction_nearest_texel_and_weight() {
        let p = Vec2::new(10.5, 10.5);
        let jitter = Vec2::new(0.25, -0.25);
        let tc = (p + jitter).floor();
        assert_eq!(tc, Vec2::new(10.0, 10.0));
        // Same rule as the depth tap: `dp = floor((p - 0.5) + jitter + 0.5)`.
        let px = p - Vec2::splat(0.5);
        let dp = (px + jitter + Vec2::splat(0.5)).floor();
        assert_eq!(dp, tc);

        let s_t = tc + Vec2::splat(0.5) - jitter;
        let d = s_t - p;
        let d2 = d.dot(d);
        assert!((d2 - 0.125).abs() < 1e-6);

        // Ratio 1: d_out = d. 0.5 / 0.47² is UE's documented 2.29.
        let k_from_sigma = 0.5 / (0.47f32 * 0.47);
        assert!((k_from_sigma - 2.29).abs() < 0.03);
        let w = (-2.29f32 * d2).exp();
        assert!((w - (-k_from_sigma * d2).exp()).abs() < 1e-2);
        assert!(w > 0.0 && w <= 1.0);

        // Upsample 2×: one render texel is two output px, so a neighbour is
        // discarded instead of forming a 2×-wide low-pass (w_max ≈ 1).
        let d_nb = Vec2::new(1.0, 0.0);
        let w_nb_1x = (-2.29f32 * d_nb.dot(d_nb)).exp();
        let d_nb_up = d_nb * 2.0;
        let w_nb_up = (-2.29f32 * d_nb_up.dot(d_nb_up)).exp();
        assert!(w_nb_1x > 0.05);
        assert!(w_nb_up < 0.01);
        assert!(w_nb_up < w_nb_1x);
    }

    /// 1D Catmull-Rom (Keys cubic, a = −0.5), matching `historyCatmullRom`.
    fn catmull_rom_w(f: f32) -> [f32; 4] {
        [
            f * (-0.5 + f * (1.0 - 0.5 * f)),
            1.0 + f * f * (-2.5 + 1.5 * f),
            f * (0.5 + f * (2.0 - 1.5 * f)),
            f * f * (-0.5 + 0.5 * f),
        ]
    }

    /// 5-tap (drop corners, renormalise) at a texel centre is the identity;
    /// remaining weights after dropping corners still sum near 1.
    #[test]
    fn catmull_rom_5tap_texel_centre_is_identity() {
        let [w0, w1, w2, w3] = catmull_rom_w(0.0);
        assert!((w1 - 1.0).abs() < 1e-6);
        assert!(w0.abs() < 1e-6 && w2.abs() < 1e-6 && w3.abs() < 1e-6);

        let [wx0, wx1, wx2, wx3] = catmull_rom_w(0.5);
        let [wy0, wy1, wy2, wy3] = catmull_rom_w(0.5);
        let w12x = wx1 + wx2;
        let w12y = wy1 + wy2;
        let taps = [w12x * wy0, wx0 * w12y, w12x * w12y, wx3 * w12y, w12x * wy3];
        let sum: f32 = taps.iter().sum();
        assert!(sum > 0.9 && sum < 1.0);
        let renorm: f32 = taps.iter().map(|w| w / sum).sum();
        assert!((renorm - 1.0).abs() < 1e-5);
    }

    /// Karis 2014 velocity-weighted feedback: boost 1.0 (default) is a no-op
    /// at any v; boost > 1 with v ≥ motion_px refreshes faster (clamped to 1).
    #[test]
    fn velocity_weighted_feedback_static_vs_moving() {
        let cur_w = 1.0 - crate::genconst::HISTORY_BLEND;
        let apply = |v: f32, boost: f32, px: f32| {
            let t = (v / px).clamp(0.0, 1.0);
            let boosted = (cur_w * boost).min(1.0);
            cur_w * (1.0 - t) + boosted * t
        };
        let px = DEFAULT_MOTION_PX;
        // Default (boost 1.0): identity at any velocity.
        assert!((apply(0.0, DEFAULT_MOTION_BOOST, px) - cur_w).abs() < 1e-6);
        assert!((apply(px, DEFAULT_MOTION_BOOST, px) - cur_w).abs() < 1e-6);
        assert!((apply(100.0, DEFAULT_MOTION_BOOST, px) - cur_w).abs() < 1e-6);
        // Opt-in boost 4.0: motion refreshes faster, static is unchanged.
        let boost = 4.0;
        assert!((apply(0.0, boost, px) - cur_w).abs() < 1e-6);
        assert!((apply(px, boost, px) - (cur_w * boost).min(1.0)).abs() < 1e-6);
        assert!(apply(px, boost, px) > apply(0.0, boost, px));
        assert!(apply(px, boost, px) <= 1.0);
    }

    #[test]
    fn motion_env_parse_f32_ignores_invalid() {
        assert_eq!(parse_f32_or(None, 1.0), 1.0);
        assert_eq!(parse_f32_or(Some("4.0"), 1.0), 4.0);
        assert_eq!(parse_f32_or(Some("8"), 1.0), 8.0);
        assert_eq!(parse_f32_or(Some("nope"), 1.0), 1.0);
        assert_eq!(parse_f32_or(Some(""), 8.0), 8.0);
        assert_eq!(parse_f32_or(Some("  "), 8.0), 8.0);
    }
}
