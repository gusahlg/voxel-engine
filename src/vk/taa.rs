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
//! so exposure meters the stabilized image.
//!
//! History is a single ping-pong integrator, independent of the 2FIF slots: each
//! frame reads the image written last frame and writes the other. Persistent;
//! recreated on resize with contents discarded (history reconverges).
//!
//! Reprojection is depth-aware: the current pixel is unprojected at its real depth
//! through the current inverse view-proj and reprojected by the packed
//! previous clip transform, so both rotation and translation reproject correctly.
//! Single-sampled only (MSAA depth can't be sampled by this pass); the MSAA fallback
//! is the old far-plane path, and disocclusions still fall to the neighbourhood clamp.

use ash::vk;
use glam::{DVec3, Mat4, Vec2};

use super::alloc::find_memory_type;
use super::buffers::FRAMES_IN_FLIGHT;
use super::image::{ImageDesc, ImageResource, LayoutUse};
use super::targets::HDR_COLOR_FORMAT;
use crate::rev::FrameSlot;

use super::pass;

/// The view-projection without jitter. Jittered matrix is applied privately
/// at push-constant packing only. Keeps VRS fingerprinting stable.
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
pub const TAA_RESOLVE_REPROJ_BINDING: u32 = 2;

/// Reprojection inputs. `prev` is clean (un-jittered) to avoid ghosting.
pub struct Reprojection {
    pub prev: CleanViewProj,
    pub camera_delta: DVec3,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReprojectionGpu {
    pub prev_view_proj: [[f32; 4]; 4],
    /// xyz = camera delta, w unused.
    pub camera_delta: [f32; 4],
}

const _: () = assert!(size_of::<ReprojectionGpu>() == 80);

impl Reprojection {
    /// Compose the previous clip transform with the camera delta, then narrow to f32.
    pub fn pack(&self) -> ReprojectionGpu {
        let d = self.camera_delta.as_vec3();
        let m = self.prev.0 * Mat4::from_translation(-d);
        ReprojectionGpu {
            prev_view_proj: m.to_cols_array_2d(),
            camera_delta: [d.x, d.y, d.z, 0.0],
        }
    }
}

const TAA_RESOLVE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/taa_resolve.comp.spv"));

/// Output storage-image binding (inputs 0/1/2 are fixed).
const TAA_RESOLVE_OUTPUT_BINDING: u32 = 3;

/// Depth binding for depth-aware reprojection.
const TAA_RESOLVE_DEPTH_BINDING: u32 = 4;

/// History feedback weight from genconst; fraction of clamped history kept each frame.
use crate::genconst::HISTORY_BLEND;

/// Push constants for `taa_resolve.comp` (the inverse view-proj without jitter).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaPush {
    inv_cur_view_proj: [[f32; 4]; 4],
    dim: [u32; 2],
    /// Raster jitter in pixels (samples depth at uv + jitter for clean center ray).
    jitter_px: [f32; 2],
    blend: f32,
    history_valid: u32,
    /// 0 under MSAA (falls back to far-plane reprojection).
    depth_valid: u32,
}

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
        mips: 1,
        layers: 1,
        aspect: vk::ImageAspectFlags::COLOR,
        samples: vk::SampleCountFlags::TYPE_1,
    };
    ImageResource::create(device, memory_props, &desc)
}

/// One slot's host-visible reprojection UBO (`ReprojectionGpu`, 80 B). Per-slot
/// so a 2FIF in-flight frame never reads a half-written buffer.
struct ReprojUbo {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut ReprojectionGpu,
}

impl ReprojUbo {
    fn new(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties) -> ReprojUbo {
        let buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size_of::<ReprojectionGpu>() as u64)
                        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("create taa reproj ubo")
        };
        let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(find_memory_type(
                            memory_props,
                            reqs.memory_type_bits,
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )),
                    None,
                )
                .expect("allocate taa reproj ubo memory")
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind taa reproj ubo memory");
        }
        let mapped = unsafe {
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map taa reproj ubo") as *mut ReprojectionGpu
        };
        ReprojUbo {
            buffer,
            memory,
            mapped,
        }
    }

    fn write(&self, r: &ReprojectionGpu) {
        unsafe { std::ptr::write(self.mapped, *r) };
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// The resolve compute pipeline plus the sampler it reads current/history through.
struct TaaCompute {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
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
                .binding(TAA_RESOLVE_REPROJ_BINDING)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
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

        TaaCompute {
            pipeline,
            layout,
            set_layout,
            sampler,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
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
    reproj: [ReprojUbo; FRAMES_IN_FLIGHT as usize],
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
            reproj: std::array::from_fn(|_| ReprojUbo::new(device, memory_props)),
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
            for u in &self.reproj {
                u.destroy(device);
            }
        }
    }
}

// SAFETY: the mapped UBO pointers are only dereferenced on the render thread
// (this state lives inside the render-thread-owned `Renderer`); `Send` is needed
// only because `Renderer` is constructed on and moved to that thread.
unsafe impl Send for TaaState {}

impl super::Renderer {
    /// Record TAA resolve (after HDR, before exposure). Outputs the stabilized HDR
    /// for exposure/tonemap to sample directly.
    pub(crate) fn record_taa_pass(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        clean_view_proj: Mat4,
        eye: DVec3,
        jitter_px: Vec2,
    ) {
        // Captured before the &mut borrow below: the frame-wide depth layout
        // (RENDERING_LOCAL_READ when the water-absorption path is active).
        let depth_layout = self.depth_pass_layout();
        let taa = &mut self.taa;
        let r = taa.read_idx;
        let w = 1 - r;

        let reproj_gpu = match taa.prev {
            Some((prev_vp, prev_eye)) => Reprojection {
                prev: CleanViewProj(prev_vp),
                camera_delta: prev_eye - eye,
            }
            .pack(),
            None => Reprojection {
                prev: CleanViewProj(clean_view_proj),
                camera_delta: DVec3::ZERO,
            }
            .pack(),
        };
        taa.reproj[slot.index()].write(&reproj_gpu);

        let extent = taa.extent;
        let offscreen = &self.targets.offscreen[slot.index()];
        // Depth-aware reprojection reads this frame's depth (same command buffer,
        // ordered by barrier). Under MSAA that is the single-sample resolve of
        // the geometry pass; single-sampled it is the depth buffer directly.
        let depth = self.targets.sampleable_depth(slot.index());
        let device = &self.device.device;
        unsafe {
            let mut pre = vec![
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(offscreen.image())
                    .subresource_range(super::color_range()),
            ];
            // Depth → shader-read. Under MSAA the producer is the render pass's
            // SAMPLE_ZERO resolve (LATE_FRAGMENT_TESTS/DEPTH_STENCIL_ATTACHMENT_WRITE),
            // covered by this src; restored to `depth_layout` after the dispatch.
            pre.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(
                        vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                            | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    )
                    .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(depth_layout)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(depth.image())
                    .subresource_range(super::depth_range()),
            );
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );
            taa.history[r].transition(device, cmd, LayoutUse::ComputeSampledRead);
            taa.history[w].transition_discard(device, cmd, LayoutUse::ComputeStorageWrite);

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
                .sampler(taa.compute.sampler)
                .image_view(depth.view())
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let reproj_info = [vk::DescriptorBufferInfo::default()
                .buffer(taa.reproj[slot.index()].buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
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
                    .dst_binding(TAA_RESOLVE_REPROJ_BINDING)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&reproj_info),
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
                inv_cur_view_proj: clean_view_proj.inverse().to_cols_array_2d(),
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
            device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);

            taa.history[w].transition(device, cmd, LayoutUse::SampledAfterComputeWrite);
            // Depth back to its attachment layout for the next pass / godray read.
            let post = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                        | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(depth_layout)
                .image(depth.image())
                .subresource_range(super::depth_range())];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&post),
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

    /// Verify TAA's footprint interferes with itself (sanity check).
    #[test]
    fn taa_interferes_with_itself() {
        let m = manifest();
        assert!(crate::producer::interferes(&m.footprint, &m.footprint));
    }

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
}
