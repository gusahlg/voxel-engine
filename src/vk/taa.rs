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
use glam::{DVec3, Mat4};

use super::alloc::find_memory_type;
use super::buffers::FRAMES_IN_FLIGHT;
use super::targets::{HDR_COLOR_FORMAT, ImageResources};
use crate::skeleton::{
    CleanViewProj, FrameSlot, Reprojection, ReprojectionGpu, TAA_RESOLVE_CURRENT_BINDING,
    TAA_RESOLVE_HISTORY_BINDING, TAA_RESOLVE_REPROJ_BINDING,
};

const TAA_RESOLVE_COMP: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/taa_resolve.comp.spv"));

/// Output storage-image binding. The three INPUT bindings are fixed
/// (current=0, history=1, reproj UBO=2); the resolved-history write target lives
/// on the same dedicated set at the next free slot (the fixed assignments are
/// untouched — this only names the output the compute stage writes).
const TAA_RESOLVE_OUTPUT_BINDING: u32 = 3;

/// This frame's depth for depth-aware reprojection. Slot 4: the fixed
/// 0/1/2 assignments are untouched, 3 is the output.
const TAA_RESOLVE_DEPTH_BINDING: u32 = 4;

/// History feedback weight from genconst; fraction of clamped history kept each frame.
use crate::genconst::HISTORY_BLEND;

fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("invalid embedded SPIR-V");
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .expect("create taa shader module")
    }
}

/// Push constants for `taa_resolve.comp` (the inverse view-proj without jitter).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TaaPush {
    inv_cur_view_proj: [[f32; 4]; 4],
    dim: [u32; 2],
    blend: f32,
    history_valid: u32,
    /// 0 under MSAA (multisampled depth can't be sampled): far-plane fallback.
    depth_valid: u32,
}

fn create_hdr_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
) -> ImageResources {
    let image = unsafe {
        device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(HDR_COLOR_FORMAT)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    // STORAGE: compute writes the resolved history. SAMPLED: next
                    // frame reprojects/samples it. TRANSFER_SRC: copied into the
                    // offscreen HDR for exposure + tonemap.
                    .usage(
                        vk::ImageUsageFlags::STORAGE
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            )
            .expect("create taa history image")
    };
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(find_memory_type(
                        memory_props,
                        reqs.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )),
                None,
            )
            .expect("allocate taa history memory")
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("bind taa history memory");
    }
    let view = unsafe {
        device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(HDR_COLOR_FORMAT)
                    .subresource_range(super::color_range()),
                None,
            )
            .expect("create taa history view")
    };
    ImageResources {
        image,
        memory,
        view,
    }
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
        let set_layout = unsafe {
            device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
                        .bindings(&bindings),
                    None,
                )
                .expect("create taa set layout")
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<TaaPush>() as u32)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .expect("create taa pipeline layout")
        };

        let module = create_shader_module(device, TAA_RESOLVE_COMP);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .module(module)
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    cache,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|(_, err)| err)
                .expect("create taa compute pipeline")[0]
        };
        unsafe { device.destroy_shader_module(module, None) };

        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                    None,
                )
                .expect("create taa sampler")
        };

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
    /// Ping-pong integrator (single logical history, NOT per-slot).
    history: [ImageResources; 2],
    hist_layout: [vk::ImageLayout; 2],
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
            hist_layout: [vk::ImageLayout::UNDEFINED; 2],
            read_idx: 0,
            reproj: std::array::from_fn(|_| ReprojUbo::new(device, memory_props)),
            extent: render_extent,
            valid: false,
            prev: None,
        }
    }

    /// Rebuild the extent-dependent history images after a resize; contents are
    /// discarded (history reconverges), so `valid`/`prev` reset. The caller
    /// has already `device_wait_idle`'d. The compute pipeline + per-slot UBOs are
    /// extent-independent and kept.
    pub(crate) fn recreate(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        render_extent: vk::Extent2D,
    ) {
        for h in &self.history {
            unsafe { h.destroy(device) };
        }
        self.history = std::array::from_fn(|_| create_hdr_image(device, memory_props, render_extent));
        self.hist_layout = [vk::ImageLayout::UNDEFINED; 2];
        self.read_idx = 0;
        self.extent = render_extent;
        self.valid = false;
        self.prev = None;
    }

    /// The history image at `i` — the finalized-HDR source `hdr_of` resolves
    /// when the resolve wrote this frame's output there.
    pub(super) fn history_image(&self, i: usize) -> (vk::Image, vk::ImageView) {
        (self.history[i].image, self.history[i].view)
    }

    /// Drop the temporal state without touching GPU resources: the next
    /// resolve integrates from the current frame alone. Called on a TAA flag
    /// toggle (E-03) — the pre-toggle history is a stale scene that must not
    /// ghost into the first re-enabled frame. The tracked image layouts are
    /// left alone: they describe the images' REAL states regardless of
    /// validity, and the next pass's barriers depend on them being truthful.
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

/// The REAL prior use of a history image, derived from its tracked layout —
/// the `src` half of that image's next barrier. History images are SHARED
/// across submissions (not per-frame-slot), so these dependencies genuinely
/// order against the previous frame; declaring a wrong source (the old code
/// claimed a compute storage write for an image last TRANSFER-read, and NO
/// dependency at all for an image the previous frame sampled) leaves a
/// cross-submission hazard sync validation cannot always see (E-02).
///
/// Reads need execution-only ordering, so `src_access` is `NONE` throughout:
/// each state's underlying storage WRITE was already made available by the
/// barrier that entered that state, and barrier chaining carries it forward.
fn history_src(last: vk::ImageLayout) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    match last {
        // Fresh or post-resize: no prior access to order against.
        vk::ImageLayout::UNDEFINED => (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE),
        // Legacy state (pre copy-back-removal sessions only); kept so the
        // mapping stays total over anything an old frame could have left.
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::NONE)
        }
        // Last uses: the previous resolve SAMPLED it as history (compute),
        // and — as the published frame HDR — exposure/bloom (compute) and the
        // tonemap present-copy (fragment) sampled it too.
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => (
            vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
            vk::AccessFlags2::NONE,
        ),
        other => unreachable!("untracked TAA history layout {other:?}"),
    }
}

impl super::Renderer {
    /// Record the TAA resolve. Called AFTER the HDR resolve and BEFORE `record_exposure_pass`,
    /// only when a 3D scene was drawn. Reads the current offscreen HDR + reprojected history
    /// and writes the stabilized image into the history integrator, which it then PUBLISHES
    /// as the slot's `hdr_source` — exposure, bloom, and the tonemap present-copy sample it
    /// directly (the old full-res copy-back into the offscreen is gone). The offscreen is
    /// left in `SHADER_READ_ONLY` (this pass's own read state).
    ///
    /// `clean_view_proj` is the camera matrix without jitter. `eye` is the world position of
    /// the render-space origin in f64 ([`DrawLists::eye`](crate::frame::DrawLists)):
    /// under camera-at-origin the frame-to-frame eye delta IS the entire translation,
    /// so it must arrive here un-narrowed.
    pub(crate) fn record_taa_pass(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
        clean_view_proj: Mat4,
        eye: DVec3,
    ) {
        // Captured before the &mut borrow below: the frame-wide depth layout
        // (RENDERING_LOCAL_READ when the water-absorption path is active).
        let depth_layout = self.depth_pass_layout();
        let taa = &mut self.taa;
        let r = taa.read_idx;
        let w = 1 - r;

        // Reprojection: the delta is a TRUE f64 subtract of world eyes;
        // narrowing happens only inside `Reprojection::pack`. First frame (no
        // prev): identity, unused because `valid` is false.
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
        // Depth-aware reprojection reads THIS frame's depth — same
        // command buffer as the write, ordered by the barrier below. Only
        // possible single-sampled; the shader's far-plane fallback covers MSAA.
        let depth_ok = self.targets.samples == vk::SampleCountFlags::TYPE_1;
        let depth = &self.targets.depth[slot.index()];
        let device = &self.device.device;
        // The src half of each history barrier comes from the image's TRACKED
        // prior state — never from an assumed pipeline shape (E-02).
        let (r_src_stage, r_src_access) = history_src(taa.hist_layout[r]);
        let (w_src_stage, w_src_access) = history_src(taa.hist_layout[w]);
        unsafe {
            // Inputs → shader-read; output → general (contents discarded).
            let mut pre = vec![
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(offscreen.image)
                    .subresource_range(super::color_range()),
                // History source: last frame's copy-out TRANSFER-read it (or
                // it is fresh); order this frame's sampling after that.
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(r_src_stage)
                    .src_access_mask(r_src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .old_layout(taa.hist_layout[r])
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image(taa.history[r].image)
                    .subresource_range(super::color_range()),
                // Becoming output: the PREVIOUS frame sampled this image as
                // its history source — the storage write must wait for that
                // read (write-after-read across submissions). Contents are
                // fully overwritten, so the layout may still discard from
                // UNDEFINED; the dependency is what was missing.
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(w_src_stage)
                    .src_access_mask(w_src_access)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(taa.history[w].image)
                    .subresource_range(super::color_range()),
            ];
            if depth_ok {
                // Mirror record_vrs_generate's transition; restored to
                // DEPTH_ATTACHMENT_OPTIMAL after the dispatch (both the next
                // main pass and the VRS classifier expect it there).
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
                        .image(depth.image)
                        .subresource_range(super::depth_range()),
                );
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, taa.compute.pipeline);
            let cur_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.sampler)
                .image_view(offscreen.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let hist_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.sampler)
                .image_view(taa.history[r].view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let out_info = [vk::DescriptorImageInfo::default()
                .image_view(taa.history[w].view)
                .image_layout(vk::ImageLayout::GENERAL)];
            // Under MSAA the depth binding still needs a VALID single-sample
            // image: bind the (already shader-read) offscreen as a dummy; the
            // shader never samples it because `depth_valid == 0`.
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(taa.compute.sampler)
                .image_view(if depth_ok { depth.view } else { offscreen.view })
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
                blend: HISTORY_BLEND,
                history_valid: taa.valid as u32,
                depth_valid: depth_ok as u32,
            };
            device.cmd_push_constants(
                cmd,
                taa.compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);

            // The resolve output IS the frame's final HDR: publish it as the
            // slot's `hdr_source` and hand it straight to the downstream
            // samplers (exposure/bloom compute, tonemap fragment). The old
            // path copied it back into the offscreen — a full-res read+write
            // (~2x frame size) every frame, bought nothing, and is gone.
            let mut post = vec![vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(
                    vk::PipelineStageFlags2::COMPUTE_SHADER
                        | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                )
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(taa.history[w].image)
                .subresource_range(super::color_range())];
            if depth_ok {
                post.push(
                    vk::ImageMemoryBarrier2::default()
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
                        .image(depth.image)
                        .subresource_range(super::depth_range()),
                );
            }
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&post),
            );
        }

        // Both images rest in SHADER_READ_ONLY: the written one is this
        // frame's published HDR (sampled by exposure/bloom/tonemap) and next
        // frame's history source; the just-read one becomes the write target.
        taa.hist_layout[w] = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        taa.hist_layout[r] = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        taa.read_idx = w;
        taa.valid = true;
        taa.prev = Some((clean_view_proj, eye));
        self.hdr_source[slot.index()] = super::HdrSource::TaaHistory(w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E-02: the barrier source is the image's REAL last use, per tracked
    /// layout — a transfer read for last frame's resolve source, a compute
    /// sample for last frame's history input, nothing for a fresh image.
    #[test]
    fn history_barrier_sources_match_the_tracked_prior_use() {
        assert_eq!(
            history_src(vk::ImageLayout::UNDEFINED),
            (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
        );
        assert_eq!(
            history_src(vk::ImageLayout::TRANSFER_SRC_OPTIMAL),
            (vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::NONE)
        );
        assert_eq!(
            history_src(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::NONE
            )
        );
    }

    /// The ping-pong state machine only ever leaves images in the three
    /// layouts `history_src` tracks, from first frame through steady state —
    /// so the barrier-source mapping is total over reachable states.
    #[test]
    fn ping_pong_layouts_stay_within_the_tracked_states() {
        let mut layouts = [vk::ImageLayout::UNDEFINED; 2];
        let mut read_idx = 0usize;
        for _ in 0..8 {
            let r = read_idx;
            let w = 1 - r;
            // Both barriers must resolve (would panic on an untracked state).
            let _ = history_src(layouts[r]);
            let _ = history_src(layouts[w]);
            // The pass's exit states (both rest sampled since the copy-back
            // removal: the output IS the published frame HDR).
            layouts[w] = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            layouts[r] = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            read_idx = w;
        }
        // Steady state: both images rest sampled.
        assert_eq!(layouts[read_idx], vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        assert_eq!(layouts[1 - read_idx], vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    }
}
