/// The Vulkan renderer: instance, device, swapchain, render targets, pipelines,
/// GPU memory, and frame loop. Vulkan 1.3 with dynamic rendering + synchronization2;
/// `FRAMES_IN_FLIGHT` command buffers in flight; reversed-Z depth; optional MSAA with resolve.
///
/// Rendering and presentation decouple: frames render into offscreen images and
/// present only when a swapchain image is available (mailbox). On macOS, vsync
/// paces at refresh via presentation backpressure; vsync off uncaps the loop.
pub(crate) mod alloc;
pub(crate) mod arena;
pub(crate) mod block_textures;
pub(crate) mod bloom;
pub(crate) mod buffers;
pub(crate) mod compute;
pub(crate) mod cull;
pub(crate) mod cull_math;
pub(crate) mod depth;
pub(crate) mod device;
pub(crate) mod draw_prep;
pub(crate) mod exposure;
pub(crate) mod frame_loop;
pub(crate) mod gpu_timer;
pub(crate) mod handles;
pub(crate) mod host_buffer;
pub(crate) mod image;
pub(crate) mod image_upload;
pub(crate) mod instance;
pub(crate) mod materials;
pub(crate) mod mesh3d_desc;
pub(crate) mod mesh_residency;
pub(crate) mod mesh_resident;
pub(crate) mod mesh_staging;
pub(crate) mod minimap;
pub(crate) mod pass;
pub(crate) mod pipeline;
pub(crate) mod present;
pub(crate) mod quad_ibo;
pub(crate) mod records;
pub(crate) mod recreate;
pub(crate) mod render_client;
pub(crate) mod retire;
pub(crate) mod scene_pass;
pub(crate) mod shadow;
pub(crate) mod sky;
pub(crate) mod submit;
pub(crate) mod swapchain;
pub(crate) mod taa;
pub(crate) mod targets;
pub(crate) mod texture;
pub(crate) mod timeline;
pub(crate) mod transfer;
pub(crate) mod uniforms;
pub(crate) mod vertex_input;
pub(crate) mod vrs;

use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::mpsc::Sender;

use ash::{khr, vk};

use crate::mesh::Pass;
use crate::skeleton::{FrameSlot, PerSlot};
use block_textures::BlockTextures;
use buffers::{DrawIndexedIndirect, FRAMES_IN_FLIGHT, GpuResident, HostBuffer, MeshResidency};
use compute::{ComputeLane, ComputeRuntime};
use device::Device;
use frame_loop::{DrawEntry, DrawRun, PendingSubmit};
use gpu_timer::{GpuPipeStats, GpuTimer};
use image::{AllocError, ImageDesc, ImageResource};
use instance::InstanceBundle;
use materials::MaterialTable;
use mesh_staging::MeshStagingPool;
use minimap::MinimapTexture;
use pipeline::Pipelines;
use render_client::{Capture, DeviceCaps, DeviceLeftovers, InitReply, RenderConfig, RenderReturn};
use swapchain::Swapchain;
use targets::{RenderTargets, walk_ladder};
use texture::FontAtlas;
use timeline::{BinarySemaphore, Timeline, TimelineValue};
use transfer::TransferLane;

pub(super) use depth::{PrevDepthTrack, SAMPLEABLE_DEPTH_REST_LAYOUT, sampleable_depth_consumed};
pub(crate) use present::HdrReadable;

/// Recoverable environmental events raised by acquire/present. `OutOfDate`
/// and `SurfaceLost` drive the existing swapchain-recreate flow. `DeviceLost`
/// is classified here for completeness but has NO recovery path — it is
/// surfaced only so callers panic on it explicitly rather than swallowing it.
enum Env {
    OutOfDate,
    SurfaceLost,
    DeviceLost,
}

impl Env {
    /// Classifies an acquire/present error. `None` for errors that are unrecoverable.
    fn classify(err: vk::Result) -> Option<Env> {
        match err {
            vk::Result::ERROR_OUT_OF_DATE_KHR => Some(Env::OutOfDate),
            vk::Result::ERROR_SURFACE_LOST_KHR => Some(Env::SurfaceLost),
            vk::Result::ERROR_DEVICE_LOST => Some(Env::DeviceLost),
            _ => None,
        }
    }
}

struct SlotState {
    cmd: vk::CommandBuffer,
    /// Signaled by acquire, waited by present copy. Reused only after the
    /// previous copy retires. Binary (WSI doesn't support timeline semaphores).
    image_available: BinarySemaphore,
    /// Timeline value the slot's render submit signals; waited before reusing
    /// the slot. Seeded to `TimelineValue::START` so frame 0 doesn't block.
    render_value: TimelineValue,
    /// Timeline value the slot's present copy signals; waited before rendering
    /// into the slot the copy reads. Seeded to `TimelineValue::START`.
    copy_value: TimelineValue,
    imm: HostBuffer,
    indirect: HostBuffer,
    /// This slot's rate image was classified at the end of a previous use
    /// (layout GENERAL) and may be bound as a shading-rate attachment. False
    /// after create/recreate so the first scene pass of a slot skips VRS.
    vrs_ready: bool,
    /// History image holds a raw classification from a previous VRS dispatch.
    vrs_history: bool,
}

/// Minimap texture edge length in texels.
pub(crate) const MINIMAP_SIZE: u32 = 256;

pub(crate) struct Renderer {
    instance: InstanceBundle,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    device: Device,

    /// Mesh residency mirrors (render-side).
    mesh_res: MeshResidency,
    /// Freed allocation channel.
    ret: Sender<RenderReturn>,
    /// Swapchain size.
    size: vk::Extent2D,

    swapchain: Swapchain,
    targets: RenderTargets,
    pipelines: Pipelines,
    /// Pipeline cache.
    pipeline_cache: vk::PipelineCache,
    atlas: FontAtlas,
    block_textures: BlockTextures,
    /// Retired textures.
    retired_textures: buffers::RetireQueue<block_textures::RetiredBlockTextures>,
    /// Per-layer material descriptors (fixed 16384-entry SSBO).
    materials: MaterialTable,
    /// Minimap texture.
    minimap: MinimapTexture,

    /// Per-slot state (command buffer, sync, readiness bits).
    slots: PerSlot<SlotState>,
    /// Copy submit semaphores.
    /// Binary because the WSI rejects timeline semaphores.
    present_semaphores: Vec<BinarySemaphore>,
    /// Persistent per-mesh record/dyn SSBOs (slot-indexed via `first_instance`).
    pub(crate) records: buffers::RecordTable,
    /// The record/dyn/arena buffers flushed for the recording slot this frame;
    /// the mesh passes' descriptor pushes read them. `None` while no mesh exists.
    record_buffers: Option<buffers::RecordBuffers>,
    /// Dirty cache for shadow depth.
    shadow_cache: shadow::ShadowCache,
    /// GPU draw-command emission.
    cull: cull::CullState,
    /// Arena registry with live counts.
    arena_dir: cull::ArenaDirectory,
    /// Cull output for this frame; GPU sources opaque/cutout/shadow from here.
    cull_frame: Option<cull::CullFrame>,
    /// App visibility mask (gates GPU and CPU cull).
    visible_mask: Vec<u32>,
    /// Shared quad index buffer.
    quad_ibo: buffers::QuadIbo,
    /// 3D pipeline descriptor layout.
    mesh3d_set_layout: vk::DescriptorSetLayout,
    /// Per-frame uniforms.
    ubo_ring: uniforms::UboRing,
    /// Shadow pass.
    shadow: shadow::ShadowPass,
    /// Exposure metering.
    exposure: exposure::ExposureState,
    /// Bloom pipelines.
    bloom: bloom::BloomState,
    /// Cloud-LUT compute pipeline.
    sky_cloud: sky::SkyCloudState,
    /// TAA state.
    taa: taa::TaaState,
    /// Per-slot "did this record store sampleable depth / was it sampled".
    prev_depth: PrevDepthTrack,
    /// 1×1 depth bound at mesh3d binding 5 when previous-frame depth is invalid.
    prev_depth_dummy: ImageResource,

    /// Resolved Blend draws (CPU path scratch).
    draw_scratch: Vec<DrawEntry>,
    /// Blend indirect commands.
    draw_commands: Vec<DrawIndexedIndirect>,
    /// Blend draw runs.
    draw_runs: Vec<DrawRun>,

    /// Feature flags.
    flags: crate::engine::RenderFlags,

    /// Present copy command buffer.
    copy_cmd: vk::CommandBuffer,
    /// Render and present timeline.
    timeline: Timeline,
    /// Transfer queue for staging copies.
    transfer_lane: TransferLane,
    /// Compute job lane (dedicated / second queue / same-queue fallback).
    compute_lane: ComputeLane,
    /// Shared with main: staging rings, job queue, poll.
    compute: Arc<ComputeRuntime>,
    /// Worker-writable mesh staging pool (also cloned to main via InitReply).
    mesh_staging: Arc<MeshStagingPool>,
    /// Transfer-lane value this graphics submission waits on: last frame's
    /// deferred mesh copies, this frame's quad-IBO grow, and/or in-place
    /// block-texture layer uploads. Stages are the first consumers of that
    /// batch (vertex input for meshes, fragment shader for textures).
    pending_transfer_wait: Option<(TimelineValue, vk::PipelineStageFlags2)>,
    /// Last present copy timeline value.
    last_copy_value: TimelineValue,
    /// Last render timeline value.
    last_render_value: TimelineValue,
    /// Offscreen slot for in-flight copy.
    copy_slot: Option<usize>,

    /// Pending screenshot/capture.
    pending_capture: Option<Capture>,

    slot: usize,

    vsync: Pending<bool>,
    msaa: Pending<SampleCount>,
    needs_recreate: bool,
    /// Render target scale relative to window.
    render_scale: Pending<f32>,
    /// Offscreen render extent.
    render_extent: vk::Extent2D,
    /// Last present time (for vsync-off pacing).
    last_present: std::time::Instant,
    present_interval: std::time::Duration,
    gpu_timer: GpuTimer,
    pipe_stats: GpuPipeStats,
    /// `VOXEL_BENCH_EMPTY=K` (K ≥ 1): that many empty command buffers per
    /// frame in one submit, no present. Zero disables the experiment.
    empty_submit: u32,
    /// Extra primary CBs for `empty_submit` > 1: `(K-1) * FRAMES_IN_FLIGHT`,
    /// indexed `slot * (K-1) + i`. The slot's usual `cmd` is the first of K.
    empty_extra: Box<[vk::CommandBuffer]>,
    /// Uncapped unpresented frames waiting to share one `vkQueueSubmit2`.
    pending_submits: Vec<PendingSubmit>,
    /// Runtime batch limit (`VOXEL_SUBMIT_BATCH`, default [`crate::rev::SUBMIT_BATCH_MAX`]).
    submit_batch_limit: usize,
    /// Eager-flush the pending batch while slot waits are blocking (GPU-bound).
    gpu_bound: frame_loop::GpuBoundState,
}

impl Renderer {
    /// Builds the renderer ON the render thread from the main-created instance +
    /// surface (a `!Send` window handle never crosses). Returns the renderer and
    /// the [`InitReply`] main uses to build its allocator. The window itself
    /// stays on main.
    ///
    /// Render-target allocation failure is returned (not panicked) so the
    /// render thread can send [`Err`] to main instead of dying.
    pub(crate) fn build(
        instance: InstanceBundle,
        surface_loader: khr::surface::Instance,
        surface: vk::SurfaceKHR,
        cfg: RenderConfig,
        ret: Sender<RenderReturn>,
    ) -> Result<(Self, InitReply), AllocError> {
        let RenderConfig {
            vsync,
            msaa,
            render_scale,
            size: win_size,
            present_interval,
            flags,
        } = cfg;
        let render_scale = Scale::new(render_scale).as_f32();

        let device = Device::new(
            &instance.entry,
            &instance.instance,
            &surface_loader,
            surface,
        );
        let mut transfer_lane = unsafe {
            TransferLane::new(
                &device.device,
                device.transfer_family,
                device.transfer_queue,
                device.transfer_tier,
            )
        };
        log::info!(
            "transfer lane: {:?} (family {})",
            transfer_lane.tier(),
            transfer_lane.family(),
        );
        let mesh_res = MeshResidency::new();

        let size = vk::Extent2D {
            width: win_size.width,
            height: win_size.height,
        };
        let swapchain = Swapchain::new(
            &instance.instance,
            &device,
            &surface_loader,
            surface,
            size,
            vsync,
            vk::SwapchainKHR::null(),
        );

        let requested_msaa = resolve_msaa(msaa, device.max_msaa(), "requested");
        let requested_scale = render_scale;
        let memory_props = unsafe {
            instance
                .instance
                .get_physical_device_memory_properties(device.physical)
        };
        let mut first_err = None;
        let walked = walk_ladder(requested_msaa, requested_scale, |msaa, scale| {
            RenderTargets::new(
                &instance.instance,
                &device.device,
                device.physical,
                scaled_extent(swapchain.extent, scale),
                msaa,
                device.fragment_shading_rate.as_ref(),
            )
            .inspect_err(|err| {
                if first_err.is_none() {
                    first_err = Some(err.clone());
                }
            })
        });
        let (targets, msaa, render_scale) = match walked {
            Some((targets, msaa, scale)) => {
                if msaa != requested_msaa || (scale - requested_scale).abs() > f32::EPSILON {
                    log::warn!(
                        "renderer: render targets fell back to MSAA {} / render scale {} (requested {} / {})",
                        msaa.as_u32(),
                        scale,
                        requested_msaa.as_u32(),
                        requested_scale,
                    );
                }
                (targets, msaa, scale)
            }
            None => {
                abort_build(
                    instance,
                    surface_loader,
                    surface,
                    device,
                    transfer_lane,
                    swapchain,
                    None,
                    None,
                );
                return Err(first_err.expect("request rung was attempted"));
            }
        };
        let render_extent = scaled_extent(swapchain.extent, render_scale);
        let taa = match taa::TaaState::new(&device.device, &memory_props, swapchain.extent) {
            Ok(taa) => taa,
            Err(err) => {
                abort_build(
                    instance,
                    surface_loader,
                    surface,
                    device,
                    transfer_lane,
                    swapchain,
                    Some(targets),
                    None,
                );
                return Err(err);
            }
        };
        log::info!("HDR color format: {:?}", targets.color_format);

        let atlas = FontAtlas::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.graphics_family,
            device.command_pool,
            &mut transfer_lane,
        );

        // Default 1x1 white block texture array (before Pipelines::new: its
        // persistent set layout feeds layout_3d).
        let block_tex = BlockTextures::new_default(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.graphics_family,
            device.command_pool,
            &mut transfer_lane,
            device.anisotropy,
            device.max_image_array_layers,
        );
        let materials = materials::MaterialTable::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.graphics_family,
            device.command_pool,
            &mut transfer_lane,
        );
        let mesh3d_set_layout = buffers::create_mesh3d_set_layout(&device.device);

        let minimap = MinimapTexture::new(
            &instance.instance,
            &device.device,
            device.physical,
            device.graphics_queue,
            device.command_pool,
            MINIMAP_SIZE,
            crate::color::Color::BLACK,
        );

        let pipeline_cache = create_pipeline_cache(&device.device);
        let pipelines = Pipelines::new(
            &device.device,
            pipeline_cache,
            targets.color_format,
            swapchain.format,
            targets.depth_format,
            targets.samples,
            atlas.set_layout,
            mesh3d_set_layout,
            device.fragment_shading_rate.as_ref(),
            device.independent_blend,
        );

        // Per-slot command buffers plus one extra for the present copy.
        let cmd_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32 + 1);
        let mut cmds = unsafe {
            device
                .device
                .allocate_command_buffers(&cmd_info)
                .expect("Failed to allocate command buffers")
        };
        let copy_cmd = cmds.pop().expect("command buffer allocation");
        let empty_submit = empty_submit_count();
        let empty_extra = if empty_submit > 1 {
            let n = (empty_submit - 1) * FRAMES_IN_FLIGHT as u32;
            let info = vk::CommandBufferAllocateInfo::default()
                .command_pool(device.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(n);
            unsafe {
                device
                    .device
                    .allocate_command_buffers(&info)
                    .expect("Failed to allocate empty-submit command buffers")
            }
            .into_boxed_slice()
        } else {
            Box::from([])
        };
        let timeline = unsafe { Timeline::new(&device.device) };
        let compute_lane = unsafe {
            ComputeLane::new(
                &device.device,
                device.compute_family,
                device.compute_queue,
                device.compute_tier,
            )
        };
        log::info!(
            "compute lane: {:?} (family {})",
            compute_lane.tier(),
            compute_lane.family(),
        );
        let props = unsafe {
            instance
                .instance
                .get_physical_device_properties(device.physical)
        };
        let compute = unsafe {
            ComputeRuntime::new(
                &instance.instance,
                &device.device,
                device.physical,
                device.min_storage_buffer_offset_alignment.max(16),
                timeline.semaphore(),
                compute_lane.semaphore_opt(),
                device.timestamps_supported,
                device.timestamp_period_ns,
                device.host_query_reset,
                compute::compute_limits(&props),
            )
        };
        let mut cmds = cmds.into_iter();
        let slots = PerSlot::new(std::array::from_fn(|_| SlotState {
            cmd: cmds.next().expect("per-slot command buffer"),
            image_available: unsafe { BinarySemaphore::new(&device.device) },
            render_value: TimelineValue::START,
            copy_value: TimelineValue::START,
            imm: HostBuffer::new(vk::BufferUsageFlags::VERTEX_BUFFER),
            indirect: HostBuffer::new(vk::BufferUsageFlags::INDIRECT_BUFFER),
            vrs_ready: false,
            vrs_history: false,
        }));

        let present_semaphores = create_present_semaphores(&device.device, swapchain.images.len());
        let ubo_ring = uniforms::UboRing::new(&instance.instance, &device.device, device.physical);

        let shadow = shadow::ShadowPass::new(
            &instance.instance,
            &device.device,
            device.physical,
            pipeline_cache,
            pipelines.layout_3d,
            pipelines.layout_debug,
        );
        let exposure = exposure::ExposureState::new(
            &device.device,
            &memory_props,
            render_extent,
            pipeline_cache,
        );
        let bloom = bloom::BloomState::new(&device.device, &memory_props, pipeline_cache);
        let sky_cloud = sky::SkyCloudState::new(&device.device, pipeline_cache);

        let gpu_timer = GpuTimer::new(
            &device.device,
            device.timestamps_supported,
            device.timestamp_period_ns,
            device.host_query_reset,
        );
        let pipe_stats = GpuPipeStats::new(
            &device.device,
            device.pipeline_statistics_query && crate::profile::is_enabled(),
            device.host_query_reset,
        );
        if crate::profile::is_enabled() && !device.pipeline_statistics_query {
            log::warn!(
                "profiler: pipelineStatisticsQuery unsupported; skipping frag/prims/overdraw gauges"
            );
        }

        let caps = DeviceCaps {
            max_msaa: device.max_msaa(),
            max_texture_layers: device.max_image_array_layers,
            device_name: unsafe {
                instance
                    .instance
                    .get_physical_device_properties(device.physical)
                    .device_name_as_c_str()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "<unknown>".into())
            },
            device_local_bytes: Device::device_local_bytes(&memory_props),
            device_local_heap_size: Device::largest_device_local_heap(&memory_props)
                .map(|(_, size)| size)
                .unwrap_or(0),
            supports_vrs: device.fragment_shading_rate.is_some(),
            vrs_texel_size: device
                .fragment_shading_rate
                .as_ref()
                .map(|fsr| (fsr.texel_size.width, fsr.texel_size.height)),
            supports_pipeline_stats: device.pipeline_statistics_query,
        };
        let mesh_staging = unsafe {
            MeshStagingPool::new(
                &instance.instance,
                &device.device,
                device.physical,
                mesh_staging::mesh_staging_bytes(),
            )
        };
        let reply = InitReply {
            instance: instance.instance.clone(),
            physical: device.physical,
            memory_budget: device.memory_budget,
            device: device.device.clone(),
            caps,
            msaa: msaa.as_u32(),
            render_scale,
            exposure: exposure.shared(),
            gpu_load: gpu_timer.load_shared(),
            mesh_stats: handles::MeshStatsShared::new(cull_math::cpu_cull_max()),
            mesh_staging: Arc::clone(&mesh_staging),
            compute: Arc::clone(&compute),
        };

        let cull = cull::CullState::new(
            &device.device,
            &memory_props,
            pipeline_cache,
            device.cull_wave_atomics,
        );
        // GPU-driven emission: opaque/cutout/shadow draws are always emitted
        // by the cull dispatch, so the device must support drawIndirectCount.
        // Device selection enforces this; this assert makes mis-selection fail
        // loudly here rather than silently mis-render.
        assert!(
            device.draw_indirect_count,
            "GPU cull requires drawIndirectCount; device selection must require it"
        );
        let prev_depth_dummy = ImageResource::create(
            &device.device,
            &memory_props,
            &ImageDesc {
                extent: vk::Extent2D {
                    width: 1,
                    height: 1,
                },
                format: targets.depth_format,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                layers: 1,
                aspect: vk::ImageAspectFlags::DEPTH,
                samples: vk::SampleCountFlags::TYPE_1,
            },
            "prev-depth dummy",
        )
        .expect("prev-depth dummy");
        let renderer = Self {
            instance,
            surface_loader,
            surface,
            device,
            mesh_res,
            ret,
            size,
            swapchain,
            targets,
            pipelines,
            pipeline_cache,
            atlas,
            block_textures: block_tex,
            retired_textures: buffers::RetireQueue::new(),
            materials,
            minimap,
            slots,
            present_semaphores,
            records: buffers::RecordTable::new(),
            record_buffers: None,
            shadow_cache: shadow::ShadowCache::new(),
            cull,
            arena_dir: {
                let mut dir = cull::ArenaDirectory::new();
                dir.attach_stats(reply.mesh_stats.clone());
                dir
            },
            cull_frame: None,
            visible_mask: Vec::new(),
            quad_ibo: buffers::QuadIbo::new(),
            mesh3d_set_layout,
            ubo_ring,
            shadow,
            exposure,
            bloom,
            sky_cloud,
            taa,
            prev_depth: PrevDepthTrack::new(),
            prev_depth_dummy,
            draw_scratch: Vec::new(),
            flags,
            draw_commands: Vec::new(),
            draw_runs: Vec::new(),
            copy_cmd,
            timeline,
            transfer_lane,
            compute_lane,
            compute,
            mesh_staging,
            pending_transfer_wait: None,
            last_copy_value: TimelineValue::START,
            last_render_value: TimelineValue::START,
            copy_slot: None,
            pending_capture: None,
            slot: 0,
            vsync: Pending::new(vsync),
            msaa: Pending::new(msaa),
            needs_recreate: false,
            render_scale: Pending::new(render_scale),
            render_extent,
            last_present: std::time::Instant::now(),
            present_interval,
            gpu_timer,
            pipe_stats,
            empty_submit,
            empty_extra,
            pending_submits: Vec::with_capacity(crate::rev::SUBMIT_BATCH_MAX),
            submit_batch_limit: frame_loop::submit_batch_limit(),
            gpu_bound: frame_loop::GpuBoundState::default(),
        };
        Ok((renderer, reply))
    }

    /// Handle window resize and flag swapchain rebuild.
    pub(crate) fn on_resize(&mut self, size: winit::dpi::PhysicalSize<u32>) {
        self.size = vk::Extent2D {
            width: size.width,
            height: size.height,
        };
        self.needs_recreate = true;
    }

    // Setters driven by RenderCmd; getters cached main-side in RenderClient.

    pub fn set_vsync(&mut self, on: bool) {
        if self.vsync.set(on) {
            self.needs_recreate = true;
        }
    }

    pub fn set_msaa(&mut self, samples: u32) -> u32 {
        let resolved = resolve_msaa(samples, self.device.max_msaa(), "set_msaa");
        if self.msaa.set(resolved) {
            self.needs_recreate = true;
        }
        resolved.as_u32()
    }

    /// Get mesh3d pipeline for the pass.
    fn mesh_pipeline_for(&self, pass: Pass) -> vk::Pipeline {
        self.pipelines.pipeline_for(pass)
    }

    /// Replace feature flags (safe mid-run).
    pub fn set_flags(&mut self, flags: crate::engine::RenderFlags) {
        if self.flags == flags {
            return;
        }
        // Flag transitions reset temporal state to avoid stale cached values.
        if self.flags.exposure && !flags.exposure {
            self.exposure.reset();
        } else if !self.flags.exposure && flags.exposure {
            self.exposure.rearm();
        }
        if self.flags.taa != flags.taa {
            self.taa.invalidate_history();
        }
        if self.flags.bloom && !flags.bloom {
            // Next presented frame must re-clear the stale pyramid to black.
            for chain in &mut self.targets.bloom {
                chain.cleared = false;
            }
        }
        self.flags = flags;
    }

    /// GPU face-run culling. Takes effect at the next cull prepare so partition
    /// capacity and the cull-params flag always agree for a frame.
    pub fn set_cull_faces(&mut self, on: bool) {
        self.cull.set_face_cull(on);
    }

    /// Set render scale; returns clamped value.
    pub fn set_render_scale(&mut self, scale: f32) -> f32 {
        let clamped = Scale::new(scale).get();
        if (clamped - self.render_scale.effective()).abs() > f32::EPSILON {
            self.render_scale.queue(clamped);
            self.needs_recreate = true;
        }
        clamped
    }

    /// Installs a main-built mesh resident into the residency mirror (from the
    /// ordered command stream). Identity/meta live on main; the mirror carries
    /// only the device buffer + staged copy.
    pub(crate) fn apply_upload_mesh(
        &mut self,
        slot: u32,
        generation: NonZeroU32,
        quads: u32,
        resident: GpuResident,
        record: buffers::MeshRecord,
    ) {
        // Grow the shared quad IBO to index this mesh before its draws record.
        self.quad_ibo.require(quads);
        self.arena_dir.note_upload(
            slot,
            generation,
            resident.buffer(),
            record.pass(),
            record.detail_scale() > 1.0,
            cull::MeshAabb::from_record(&record),
        );
        self.arena_dir.note_cull_draw(slot, &record);
        self.mesh_res.apply_upload(slot, generation, resident);
        self.records.install(slot, record);
    }

    /// Replaces a mover's recomposed record, keeping the cull lane counts in
    /// step should its detail (LOD lane) have changed.
    pub(crate) fn apply_set_record(&mut self, slot: u32, record: buffers::MeshRecord) {
        self.arena_dir.note_record(
            slot,
            record.pass(),
            record.detail_scale() > 1.0,
            cull::MeshAabb::from_record(&record),
        );
        self.arena_dir.note_cull_draw(slot, &record);
        self.records.set_record(slot, record);
    }

    /// Set one word of the visibility mask.
    pub(crate) fn set_visible_word(&mut self, word: u32, bits: u32) {
        let i = word as usize;
        if self.visible_mask.len() <= i {
            self.visible_mask.resize(i + 1, 0);
        }
        self.visible_mask[i] = bits;
    }

    /// Retire a freed mesh resident.
    pub(crate) fn apply_free_mesh(&mut self, slot: u32, generation: NonZeroU32) {
        // Pending frames may still draw this mesh, submitted or not. Stamp
        // with `last_reserved` so the retire queue covers the newest frame
        // that could reference it without flushing the submit batch.
        // `RetireQueue::collect` compares against the completed counter and
        // leaves a not-yet-signalled stamp queued.
        self.arena_dir.note_free(slot, generation);
        self.records.clear_arena(slot);
        self.mesh_res
            .apply_free(slot, generation, self.timeline.last_reserved());
    }

    /// Queue screenshot capture to path.
    pub fn request_capture(&mut self, capture: Capture) {
        // Surface without TRANSFER_SRC cannot screenshot.
        if !self.swapchain.screenshot_capable {
            log::error!("screenshot refused: surface lacks TRANSFER_SRC swapchain usage");
            if let Some(reply) = capture.reply {
                let _ = reply.send(Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "surface does not support screenshot copies",
                )));
            }
            return;
        }
        if let Some(prev) = self.pending_capture.replace(capture) {
            if let Some(reply) = prev.reply {
                let _ = reply.send(Err(std::io::Error::new(
                    std::io::ErrorKind::Interrupted,
                    "capture superseded by a newer request",
                )));
            }
        }
    }

    /// Replace block texture array. Same texel size within capacity uploads
    /// only new/changed layers on the transfer lane (no idle wait). Size
    /// change or capacity overflow reallocates and retires the old image.
    pub fn set_block_textures(&mut self, size: u32, layers: &[Vec<u8>]) {
        let cap = self.device.max_image_array_layers as usize;
        let layers = if layers.len() > cap {
            log::error!(
                "set_block_textures: {} layers exceeds the device cap of {cap}; truncating",
                layers.len()
            );
            &layers[..cap]
        } else {
            layers
        };
        if self
            .block_textures
            .can_update_in_place(size, layers.len() as u32)
        {
            self.block_textures.queue_set(layers);
            log::debug!(
                "block textures in-place: {} used / {} cap of {}x{}",
                self.block_textures.layers,
                self.block_textures.capacity(),
                self.block_textures.size,
                self.block_textures.size,
            );
            return;
        }
        // Pending frames sample the old array; submit them so the grow copy's
        // graphics-queue barrier (and the retire stamp) covers them. No idle wait.
        self.flush_pending_submits();
        self.block_textures.queue_grow(size, layers.to_vec());
        log::debug!(
            "block textures grow queued: {} layers of {}x{} (bound {} used / {} cap of {}x{})",
            layers.len(),
            size,
            size,
            self.block_textures.layers,
            self.block_textures.capacity(),
            self.block_textures.size,
            self.block_textures.size,
        );
    }

    /// Replace the per-layer material table. Index = layer id; unused tail
    /// slots return to the array-layer default. Transfer-lane upload, no idle
    /// wait; the buffer is fixed at 16384 entries.
    pub fn set_material_descs(&mut self, descs: &[crate::MaterialDesc]) {
        self.materials.queue_set(descs);
        log::debug!(
            "material descs set: {} used / {} cap",
            self.materials.used(),
            crate::MATERIAL_DESC_CAPACITY,
        );
    }

    /// Append descriptors at the current used count. Excess past the 14-bit
    /// layer cap is dropped.
    pub fn append_material_descs(&mut self, descs: &[crate::MaterialDesc]) {
        self.materials.queue_append(descs);
        log::debug!(
            "material descs append: {} used / {} cap",
            self.materials.used(),
            crate::MATERIAL_DESC_CAPACITY,
        );
    }

    /// Append layers at the current texel size. Fits-in-capacity uploads only
    /// the new layers; overflow reallocates through [`Self::set_block_textures`].
    pub fn append_block_textures(&mut self, layers: &[Vec<u8>]) {
        let limit = self.device.max_image_array_layers;
        match self.block_textures.try_append(layers, limit) {
            Ok(()) => {
                log::debug!(
                    "block textures append: {} used / {} cap of {}x{}",
                    self.block_textures.layers,
                    self.block_textures.capacity(),
                    self.block_textures.size,
                    self.block_textures.size,
                );
            }
            Err(()) => {
                let mut all = self.block_textures.palette().to_vec();
                all.extend(layers.iter().cloned());
                let size = self.block_textures.size;
                self.set_block_textures(size, &all);
            }
        }
    }

    /// Uploads minimap pixels to staging buffer (synced per-slot).
    pub fn update_minimap(&mut self, rgba: &[u8]) {
        self.minimap.update(rgba);
    }

    pub fn update_minimap_rect(&mut self, x: u32, y: u32, w: u32, h: u32, rgba: &[u8]) {
        self.minimap.update_rect(x, y, w, h, rgba);
    }
}

impl Renderer {
    /// Destroys every render-owned resource (GPU idle first) and hands the
    /// device/instance/surface back to main, which destroys the allocator
    /// buffers and then `vkDestroyDevice` in the correct order. Consuming `self`
    /// (rather than `Drop`) is what lets those fields move out to main.
    pub(crate) fn teardown(mut self) -> DeviceLeftovers {
        self.flush_pending_submits();
        unsafe {
            let device = &self.device.device;
            let _ = device.device_wait_idle();

            self.pipelines.destroy(device);
            save_pipeline_cache(device, self.pipeline_cache);
            device.destroy_pipeline_cache(self.pipeline_cache, None);
            self.atlas.destroy(device);
            self.minimap.destroy(device);
            device.destroy_descriptor_set_layout(self.mesh3d_set_layout, None);
            self.ubo_ring.destroy(device);
            self.shadow.destroy(device);
            self.exposure.destroy(device);
            self.bloom.destroy(device);
            self.sky_cloud.destroy(device);
            self.taa.destroy(device);
            self.prev_depth_dummy.destroy(device);
            self.block_textures.destroy(device);
            self.retired_textures
                .collect_all(|mut tex| tex.destroy(device));
            self.materials.destroy(device);
            self.gpu_timer.destroy(device);
            self.pipe_stats.destroy(device);
            self.targets.destroy(device);
            self.records.destroy(device);
            self.cull.destroy(device);
            self.quad_ibo.destroy(device);
            // The residents' allocations belong to the main-owned allocator
            // (destroyed there after this returns); just drop them — no Vulkan
            // calls, GPU already idle. Staging leases return to the pool ring
            // here, then the pool buffer itself is destroyed.
            self.mesh_res.destroy_all(&mut |_a| {});
            self.mesh_staging.destroy(device);
            self.compute.destroy(device);
            self.compute_lane.destroy(device);
            for &sem in &self.present_semaphores {
                sem.destroy(device);
            }
            self.timeline.destroy(device);
            self.transfer_lane.destroy(device);
            for slot in 0..FRAMES_IN_FLIGHT as usize {
                let s = &mut self.slots[FrameSlot::new(slot)];
                s.imm.destroy(device);
                s.indirect.destroy(device);
                s.image_available.destroy(device);
            }
            self.swapchain.destroy(device);
        }
        DeviceLeftovers {
            instance: self.instance,
            surface_loader: self.surface_loader,
            surface: self.surface,
            device: self.device,
        }
    }
}

/// Profiling experiment: `VOXEL_BENCH_EMPTY=K` (integer K ≥ 1) records K
/// distinct empty command buffers per frame (begin/end only; re-recording one
/// primary K times is not valid), submits them in **one** `vkQueueSubmit2`
/// (one `VkSubmitInfo2` with K `VkCommandBufferSubmitInfo`s and the usual
/// single timeline signal), and skips present. Isolates whether the ~21 µs
/// per-submission floor is per submit or per command buffer. Unset / `0` /
/// non-integer disables. Read once at renderer creation. Not a public API.
fn empty_submit_count() -> u32 {
    static COUNT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *COUNT.get_or_init(|| {
        let Ok(v) = std::env::var("VOXEL_BENCH_EMPTY") else {
            return 0;
        };
        v.parse::<u32>().ok().filter(|&k| k >= 1).unwrap_or(0)
    })
}

/// Clamp range for render-resolution scale (0.25x to 2.0x). Re-exported from
/// crate root so settings UI and renderer stay in sync.
pub const RENDER_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=2.0;

/// Render-resolution scale relative to the window, clamped to
/// [`RENDER_SCALE_RANGE`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Scale(f32);

impl Scale {
    /// Clamps `value` into the supported [`RENDER_SCALE_RANGE`].
    pub fn new(value: f32) -> Self {
        Scale(value.clamp(*RENDER_SCALE_RANGE.start(), *RENDER_SCALE_RANGE.end()))
    }

    /// The clamped scale factor.
    pub fn get(self) -> f32 {
        self.0
    }

    /// Alias for [`Scale::get`].
    pub fn as_f32(self) -> f32 {
        self.0
    }
}

/// A value plus an optional change queued to apply at the next frame boundary.
/// Reads go through [`effective`](Self::effective), so a getter can never
/// forget to account for a pending change — the footgun of parallel
/// `current`/`pending` fields.
struct Pending<T> {
    current: T,
    pending: Option<T>,
}

impl<T: Copy + PartialEq> Pending<T> {
    fn new(current: T) -> Self {
        Self {
            current,
            pending: None,
        }
    }

    /// The value including any queued change.
    fn effective(&self) -> T {
        self.pending.unwrap_or(self.current)
    }

    /// The currently-applied value, ignoring any queued change. Use where the
    /// live GPU state (not the requested one) matters, e.g. present pacing
    /// before the swapchain is rebuilt.
    fn current(&self) -> T {
        self.current
    }

    /// Queues `next` unconditionally (caller owns the change test).
    fn queue(&mut self, next: T) {
        self.pending = Some(next);
    }

    /// Queues `next` if it differs from the effective value; returns whether
    /// it did, so callers can flag a recreate.
    fn set(&mut self, next: T) -> bool {
        let changed = next != self.effective();
        if changed {
            self.pending = Some(next);
        }
        changed
    }

    /// Applies any queued change; returns whether one was applied.
    fn commit(&mut self) -> bool {
        match self.pending.take() {
            Some(v) => {
                self.current = v;
                true
            }
            None => false,
        }
    }
}

/// A validated MSAA sample count (powers of two only: 1, 2, 4, 8).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SampleCount {
    X1,
    X2,
    X4,
    X8,
}

impl SampleCount {
    /// The sample count as a `u32` (1, 2, 4, or 8).
    pub fn as_u32(self) -> u32 {
        match self {
            SampleCount::X1 => 1,
            SampleCount::X2 => 2,
            SampleCount::X4 => 4,
            SampleCount::X8 => 8,
        }
    }

    /// The corresponding Vulkan sample-count flag.
    pub fn as_flags(self) -> vk::SampleCountFlags {
        match self {
            SampleCount::X1 => vk::SampleCountFlags::TYPE_1,
            SampleCount::X2 => vk::SampleCountFlags::TYPE_2,
            SampleCount::X4 => vk::SampleCountFlags::TYPE_4,
            SampleCount::X8 => vk::SampleCountFlags::TYPE_8,
        }
    }

    /// Rounds an arbitrary `u32` DOWN to the nearest valid {1,2,4,8} bucket.
    fn bucket(value: u32) -> SampleCount {
        match value {
            0 | 1 => SampleCount::X1,
            2..=3 => SampleCount::X2,
            4..=7 => SampleCount::X4,
            _ => SampleCount::X8,
        }
    }

    /// Finds the largest supported count <= max, clamped to {1,2,4,8}. Returns
    /// (count, changed) so callers can log downgrades.
    pub fn nearest_supported(requested: u32, max: u32) -> (SampleCount, bool) {
        let mut count = SampleCount::bucket(requested);
        let cap = SampleCount::bucket(max);
        if count.as_u32() > cap.as_u32() {
            count = cap;
        }
        (count, count.as_u32() != requested)
    }
}

/// Resolves an MSAA request to a supported {1,2,4,8} count, logging any
/// downgrade. `context` labels the log line.
fn resolve_msaa(requested: u32, max: u32, context: &str) -> SampleCount {
    let (count, changed) = SampleCount::nearest_supported(requested, max);
    if changed {
        log::debug!(
            "MSAA ({context}): requested {requested}x -> using {}x (max {max}x)",
            count.as_u32(),
        );
    }
    count
}

/// Pipeline cache path: OS temp dir for per-user write access.
fn pipeline_cache_path() -> std::path::PathBuf {
    std::env::temp_dir().join("voxel_engine_pipeline.cache")
}

/// Creates pipeline cache, seeded from disk if available. Invalid data falls back to empty.
fn create_pipeline_cache(device: &ash::Device) -> vk::PipelineCache {
    let data = std::fs::read(pipeline_cache_path()).unwrap_or_default();
    if !data.is_empty() {
        let info = vk::PipelineCacheCreateInfo::default().initial_data(&data);
        if let Ok(cache) = unsafe { device.create_pipeline_cache(&info, None) } {
            log::debug!("pipeline cache loaded ({} bytes)", data.len());
            return cache;
        }
        log::warn!("saved pipeline cache rejected; starting empty");
    }
    unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
        .expect("Failed to create pipeline cache")
}

/// Best-effort write-back of the pipeline cache; a failure only costs the
/// next run's warm start.
fn save_pipeline_cache(device: &ash::Device, cache: vk::PipelineCache) {
    match unsafe { device.get_pipeline_cache_data(cache) } {
        Ok(data) if !data.is_empty() => {
            if let Err(err) = std::fs::write(pipeline_cache_path(), &data) {
                log::debug!("pipeline cache not saved: {err}");
            }
        }
        Ok(_) => {}
        Err(err) => log::debug!("pipeline cache data unavailable: {err:?}"),
    }
}

fn create_present_semaphores(device: &ash::Device, count: usize) -> Vec<BinarySemaphore> {
    (0..count)
        .map(|_| unsafe { BinarySemaphore::new(device) })
        .collect()
}

/// Tear down Vulkan objects created before a render-target allocation failure
/// in [`Renderer::build`]. The render thread owns instance/surface/device at
/// this point; main never sees them.
#[allow(clippy::too_many_arguments)]
fn abort_build(
    mut instance: InstanceBundle,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    mut device: Device,
    mut transfer_lane: TransferLane,
    mut swapchain: Swapchain,
    mut targets: Option<RenderTargets>,
    taa: Option<taa::TaaState>,
) {
    unsafe {
        if let Some(taa) = &taa {
            taa.destroy(&device.device);
        }
        if let Some(targets) = &mut targets {
            targets.destroy(&device.device);
        }
        swapchain.destroy(&device.device);
        transfer_lane.destroy(&device.device);
        device.destroy();
        surface_loader.destroy_surface(surface, None);
        instance.destroy();
    }
}

/// Clamps an MSAA request to a supported {1,2,4,8} sample count (as a `u32`),
/// mirroring [`Renderer::set_msaa`] so the client can clamp locally.
pub(crate) fn clamp_msaa(requested: u32, max: u32) -> u32 {
    SampleCount::nearest_supported(requested, max).0.as_u32()
}

/// Display refresh interval; falls back to 60 Hz if unavailable.
pub(crate) fn display_refresh_interval(window: &winit::window::Window) -> std::time::Duration {
    let millihertz = window
        .current_monitor()
        .and_then(|m| m.refresh_rate_millihertz())
        .filter(|&mhz| mhz > 0) // Some(0) = unknown on some X11/VM backends
        .unwrap_or(60_000);
    std::time::Duration::from_secs_f64(1000.0 / millihertz as f64)
}

fn scaled_extent(extent: vk::Extent2D, scale: f32) -> vk::Extent2D {
    vk::Extent2D {
        width: ((extent.width as f32 * scale) as u32).max(1),
        height: ((extent.height as f32 * scale) as u32).max(1),
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn depth_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_clamps_into_range() {
        assert_eq!(Scale::new(1.0).get(), 1.0);
        assert_eq!(Scale::new(0.5).as_f32(), 0.5);
        // Below the floor and above the ceiling clamp to the bounds.
        assert_eq!(Scale::new(0.0).get(), 0.25);
        assert_eq!(Scale::new(-5.0).get(), 0.25);
        assert_eq!(Scale::new(10.0).get(), 2.0);
        assert_eq!(Scale::new(2.0).get(), 2.0);
        assert_eq!(Scale::new(0.25).get(), 0.25);
    }

    #[test]
    fn sample_count_conversions() {
        for (count, n, flag) in [
            (SampleCount::X1, 1, vk::SampleCountFlags::TYPE_1),
            (SampleCount::X2, 2, vk::SampleCountFlags::TYPE_2),
            (SampleCount::X4, 4, vk::SampleCountFlags::TYPE_4),
            (SampleCount::X8, 8, vk::SampleCountFlags::TYPE_8),
        ] {
            assert_eq!(count.as_u32(), n);
            assert_eq!(count.as_flags(), flag);
        }
    }

    #[test]
    fn nearest_supported_exact_values_are_unchanged() {
        for n in [1, 2, 4, 8] {
            let (count, changed) = SampleCount::nearest_supported(n, 8);
            assert_eq!(count.as_u32(), n);
            assert!(!changed, "{n} is exact and within cap");
        }
    }

    #[test]
    fn nearest_supported_rounds_odd_down_and_flags_change() {
        // Odd / non-power-of-two values round DOWN to the nearest bucket and
        // report changed = true (the log-worthy downgrade case).
        for (req, expected) in [(0, 1), (3, 2), (5, 4), (7, 4), (16, 8), (100, 8)] {
            let (count, changed) = SampleCount::nearest_supported(req, 8);
            assert_eq!(count.as_u32(), expected, "requested {req}");
            assert!(changed, "requested {req} was downgraded");
        }
    }

    #[test]
    fn nearest_supported_caps_at_max_and_flags_change() {
        // Hardware cap: 8x requested but only 4x supported -> 4x, changed.
        let (count, changed) = SampleCount::nearest_supported(8, 4);
        assert_eq!(count.as_u32(), 4);
        assert!(changed);

        // Cap of 1x (no MSAA support) forces X1.
        let (count, changed) = SampleCount::nearest_supported(4, 1);
        assert_eq!(count.as_u32(), 1);
        assert!(changed);

        // Requested already at the cap: unchanged.
        let (count, changed) = SampleCount::nearest_supported(4, 4);
        assert_eq!(count.as_u32(), 4);
        assert!(!changed);
    }
}
