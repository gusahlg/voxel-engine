//! Temporal anti-aliasing: history images, reprojection, and present-time resolve.
//!
//! The scene is rendered with a per-frame sub-pixel jitter (Halton(2,3), applied
//! ONLY to the mesh view-proj at push-constant packing — [`super::jittered_clip`]).
//! The present-time tonemap (`-DTAA_FUSED`) integrates jittered frames into a
//! stable image at **swapchain resolution**: it reprojects the previous
//! *presented* frame, neighbourhood-clamps in YCoCg, blends, writes the new
//! history as a second colour attachment, and tonemaps the resolved colour.
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

use super::image::{ImageDesc, ImageResource, LayoutUse};

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
pub const TAA_RESOLVE_CURRENT_BINDING: u32 = 0;
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
    pub exposure: f32,
    pub s: f32,
    pub atan_s: f32,
    pub vignette: f32,
    pub blend: f32,
    pub history_valid: u32,
    pub depth_valid: u32,
    pub _pad: u32,
}

const _: () = assert!(size_of::<TonemapTaaPush>() <= 128);
const _: () = assert!(size_of::<TonemapTaaPush>() == 112);

fn create_history_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
) -> ImageResource {
    let desc = ImageDesc {
        extent,
        format: TAA_HISTORY_FORMAT,
        usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        layers: 1,
        aspect: vk::ImageAspectFlags::COLOR,
        samples: vk::SampleCountFlags::TYPE_1,
    };
    ImageResource::create(device, memory_props, &desc)
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
/// - **MSAA on**: depth is the SAMPLE_ZERO resolve target, already resting in
///   [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`]; sampled with no further transition.
/// - **`render_scale != 1`**: history is swapchain-sized, current/depth are
///   render-sized; `source_uv = warpSampleUv(uv_out)` maps the two spaces.
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
}

impl TaaState {
    pub(crate) fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        swapchain_extent: vk::Extent2D,
    ) -> TaaState {
        TaaState {
            history: std::array::from_fn(|_| {
                create_history_image(device, memory_props, swapchain_extent)
            }),
            read_idx: 0,
            extent: swapchain_extent,
            valid: false,
            prev: None,
        }
    }

    /// Rebuild history images after swapchain recreate/resize (contents
    /// discarded, reconverges).
    pub(crate) fn recreate(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        swapchain_extent: vk::Extent2D,
    ) {
        for h in &self.history {
            unsafe { h.destroy(device) };
        }
        self.history =
            std::array::from_fn(|_| create_history_image(device, memory_props, swapchain_extent));
        self.read_idx = 0;
        self.extent = swapchain_extent;
        self.valid = false;
        self.prev = None;
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
        let extent = self.render_extent;
        TonemapTaaPush {
            reproj: reproj.to_cols_array_2d(),
            render_extent: [extent.width as f32, extent.height as f32],
            jitter_px: taa.jitter.to_array(),
            exposure: warp.exposure,
            s: warp.s,
            atan_s: warp.atan_s,
            vignette: warp.vignette,
            blend: HISTORY_BLEND,
            history_valid: self.taa.history_valid() as u32,
            // Depth is sampleable at every sample count (MSAA resolves it).
            depth_valid: 1,
            _pad: 0,
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
    }
}
