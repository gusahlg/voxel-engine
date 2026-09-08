//! Temporal anti-aliasing resolve: reprojection, neighbourhood clamping, and blending
//! to stabilize jittered frames.
//!
//! The scene is rendered with a per-frame sub-pixel jitter (Halton(2,3), applied
//! ONLY to the mesh view-proj at push-constant packing — [`super::jittered_clip`]).
//! This pass integrates the jittered frames into a stable image: it reprojects the
//! previous resolved frame into the current view, neighbourhood-clamps it against
//! the current frame to kill ghosting, blends, and writes BOTH the history
//! integrator and the resolved HDR that exposure meters and tonemap reads.
//!
//! This pass runs AFTER the main HDR resolve and BEFORE `record_exposure_pass`,
//! so exposure meters the stabilized image. TAA itself runs every frame (history
//! must not freeze on mailbox drops); bloom/exposure are present-only.
//!
//! History is a single ping-pong integrator, independent of the 2FIF slots: each
//! frame reads the image written last frame and writes the other. Persistent;
//! recreated on resize with contents discarded (history reconverges).
//!
//! Reprojection is depth-aware: the host f64-composes `prev * inv(cur)` into a
//! single push-constant matrix (no per-slot UBO). Depth is point-sampled.

use ash::vk;
use glam::{DMat4, DVec3, Mat4, Vec2};

use super::image::{ImageDesc, ImageResource, LayoutUse};
use super::targets::HDR_COLOR_FORMAT;
use crate::rev::FrameSlot;

use super::pass;

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

/// TAA runs after HDR resolve, before exposure metering. Wrong pass order
/// causes flicker without compile error. Test with `taa_static_hold` (fixed camera)
/// to verify temporal stability.
pub const TAA_RESOLVE_CURRENT_BINDING: u32 = 0;
pub const TAA_RESOLVE_HISTORY_BINDING: u32 = 1;

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

const TAA_RESOLVE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/taa_resolve.comp.spv"));

const TAA_RESOLVE_OUTPUT_BINDING: u32 = 2;
const TAA_RESOLVE_DEPTH_BINDING: u32 = 3;

const TAA_TILE: u32 = crate::genconst::TAA_TILE;

/// History feedback weight from genconst; fraction of clamped history kept each frame.
use crate::genconst::HISTORY_BLEND;

/// Push constants for `taa_resolve.comp`. One f64-composed reprojection matrix;
/// Vulkan min `maxPushConstantsSize` is 128 B.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaPush {
    reproj: [[f32; 4]; 4],
    dim: [u32; 2],
    /// Raster jitter in pixels (point-sampled depth at the jittered pixel).
    jitter_px: [f32; 2],
    blend: f32,
    history_valid: u32,
    /// 0 under the far-plane fallback (no sampleable depth).
    depth_valid: u32,
}

const _: () = assert!(size_of::<TaaPush>() <= 128);

fn create_hdr_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
) -> ImageResource {
    let desc = ImageDesc {
        extent,
        format: HDR_COLOR_FORMAT,
        usage: vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC,
        layers: 1,
        aspect: vk::ImageAspectFlags::COLOR,
        samples: vk::SampleCountFlags::TYPE_1,
    };
    ImageResource::create(device, memory_props, &desc)
}

/// The resolve compute pipeline plus the samplers it reads current/history/depth through.
struct TaaCompute {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
    depth_sampler: vk::Sampler,
}

impl TaaCompute {
    fn new(device: &ash::Device, cache: vk::PipelineCache) -> TaaCompute {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(TAA_RESOLVE_CURRENT_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(TAA_RESOLVE_HISTORY_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(TAA_RESOLVE_OUTPUT_BINDING)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(TAA_RESOLVE_DEPTH_BINDING)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        ];
        let (set_layout, layout) =
            pass::push_descriptor_layouts(device, &bindings, size_of::<TaaPush>() as u32, "taa");
        let pipeline = pass::compute_pipeline(device, cache, layout, TAA_RESOLVE_COMP, "taa");
        let sampler = pass::linear_clamp_sampler(device, "taa");
        let depth_sampler = pass::nearest_clamp_sampler(device, "taa depth");

        TaaCompute {
            pipeline,
            layout,
            set_layout,
            sampler,
            depth_sampler,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler(self.depth_sampler, None);
        }
    }
}

/// Render-thread owner of the whole TAA resolve pass.
pub(crate) struct TaaState {
    compute: TaaCompute,
    /// Ping-pong integrator (single logical history, NOT per-slot). Each
    /// image tracks its own layout (folded from a separate `hist_layout`
    /// array into `ImageResource` itself — one writer, not two).
    history: [ImageResource; 2],
    /// Index of the image holding LAST frame's resolved output (this frame's
    /// history source); the other is written this frame, then becomes the source.
    read_idx: usize,
    extent: vk::Extent2D,
    /// False until at least one frame has populated `read_idx` (and after a
    /// resize discards history): the shader then integrates from the current
    /// frame alone so no garbage/black history bleeds in.
    valid: bool,
    /// Previous frame's view-proj (without jitter) + render-space-origin world position
    /// (f64 — the delta is computed BEFORE any narrowing), for reprojection.
    prev: Option<(Mat4, DVec3)>,
}

impl TaaState {
    pub(crate) fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        render_extent: vk::Extent2D,
        cache: vk::PipelineCache,
    ) -> TaaState {
        TaaState {
            compute: TaaCompute::new(device, cache),
            history: std::array::from_fn(|_| create_hdr_image(device, memory_props, render_extent)),
            read_idx: 0,
            extent: render_extent,
            valid: false,
            prev: None,
        }
    }

    /// Rebuild history images after resize (contents discarded, reconverges).
    pub(crate) fn recreate(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        render_extent: vk::Extent2D,
    ) {
        for h in &self.history {
            unsafe { h.destroy(device) };
        }
        self.history =
            std::array::from_fn(|_| create_hdr_image(device, memory_props, render_extent));
        self.read_idx = 0;
        self.extent = render_extent;
        self.valid = false;
        self.prev = None;
    }

    /// The history image at index `i`.
    pub(super) fn history_image(&self, i: usize) -> (vk::Image, vk::ImageView) {
        (self.history[i].image(), self.history[i].view())
    }

    /// Reset temporal state (called on TAA toggle to prevent history ghosting).
    pub(crate) fn invalidate_history(&mut self) {
        self.valid = false;
        self.prev = None;
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.compute.destroy(device);
            for h in &self.history {
                h.destroy(device);
            }
        }
    }
}

impl super::Renderer {
    /// Record TAA resolve (after HDR, before exposure). Outputs the stabilized HDR
    /// for exposure/tonemap to sample directly. One pre-dispatch barrier and one
    /// post-dispatch barrier cover offscreen and both history images. Depth
    /// already rests in [`super::SAMPLEABLE_DEPTH_REST_LAYOUT`] from
    /// `RenderPass::end` and is sampled with no further transition.
    pub(crate) fn record_taa_pass(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        clean_view_proj: Mat4,
        eye: DVec3,
        jitter_px: Vec2,
    ) {
        let taa = &mut self.taa;
        let r = taa.read_idx;
        let w = 1 - r;

        let reproj_mat = match taa.prev {
            Some((prev_vp, prev_eye)) => Reprojection {
                prev: CleanViewProj(prev_vp),
                camera_delta: prev_eye - eye,
            }
            .matrix(clean_view_proj),
            None => Mat4::IDENTITY,
        };

        let extent = taa.extent;
        let offscreen = &self.targets.offscreen[slot.index()];
        // Depth-aware reprojection reads this frame's depth (already in
        // SAMPLEABLE_DEPTH_REST_LAYOUT from RenderPass::end). Under MSAA that
        // is the single-sample resolve of the geometry pass; single-sampled it
        // is the depth buffer directly.
        let depth = self.targets.sampleable_depth(slot.index());
        let device = &self.device.device;
        unsafe {
            let hist_r = taa.history[r].barrier_to(LayoutUse::ComputeSampledRead, false);
            let hist_w = taa.history[w].barrier_to(LayoutUse::ComputeStorageWrite, true);
            // Offscreen: src COLOR_ATTACHMENT_OUTPUT / COLOR_ATTACHMENT_WRITE
            // (end_deferred left it in COLOR_ATTACHMENT). Dst COMPUTE_SHADER /
            // SHADER_SAMPLED_READ. Old COLOR_ATTACHMENT → SHADER_READ_ONLY.
            // Depth is not in this batch: it already rests in SHADER_READ_ONLY.
            let pre = [
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(offscreen.image())
                    .subresource_range(super::color_range()),
                hist_r,
                hist_w,
            ];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, taa.compute.pipeline);
            let cur_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.sampler)
                .image_view(offscreen.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let hist_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.sampler)
                .image_view(taa.history[r].view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let out_info = [vk::DescriptorImageInfo::default()
                .image_view(taa.history[w].view())
                .image_layout(vk::ImageLayout::GENERAL)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.depth_sampler)
                .image_view(depth.view())
                .image_layout(super::SAMPLEABLE_DEPTH_REST_LAYOUT)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(TAA_RESOLVE_CURRENT_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&cur_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(TAA_RESOLVE_HISTORY_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&hist_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(TAA_RESOLVE_OUTPUT_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&out_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(TAA_RESOLVE_DEPTH_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                taa.compute.layout,
                0,
                &writes,
            );

            let push = TaaPush {
                reproj: reproj_mat.to_cols_array_2d(),
                dim: [extent.width, extent.height],
                jitter_px: jitter_px.to_array(),
                blend: HISTORY_BLEND,
                history_valid: taa.valid as u32,
                // Depth is now sampleable at every sample count (MSAA resolves it).
                depth_valid: 1,
            };
            device.cmd_push_constants(
                cmd,
                taa.compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(
                cmd,
                extent.width.div_ceil(TAA_TILE),
                extent.height.div_ceil(TAA_TILE),
                1,
            );

            // History write → sampled for exposure/tonemap. Depth stays at rest.
            let hist_pub = [taa.history[w].barrier_to(LayoutUse::SampledAfterComputeWrite, false)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&hist_pub),
            );
        }

        taa.read_idx = w;
        taa.valid = true;
        taa.prev = Some((clean_view_proj, eye));
        self.slots[slot].hdr_source = super::HdrSource::TaaHistory(w);
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
}
