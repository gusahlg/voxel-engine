//! Scene exposure metering — compute-based reduction with temporal smoothing.
//!
//! A compute pass reduces the linear-HDR offscreen to a small per-tile buffer of
//! mean `log2(luma)` (16×16 texels/tile, `exposure_reduce.comp`). The CPU averages
//! the tile means after the slot's fence, maps the geometric-mean luma through the
//! exposure curve, and temporally smooths the result — yielding the exposure
//! multiplier the tonemap pass applies before its tone curve.
//!
//! All of this package's engine code lives here (not `vk/mod.rs`): the pipeline,
//! the double-buffered readback ring, `Renderer::record_exposure_pass`, and
//! `Engine::exposure_for_compose` are added as `impl` blocks on types owned by
//! other modules — Rust permits inherent impls from any module of the crate.
//!
//! MERGE NOTES (orchestrator wires these; they are the only touches outside this
//! file, deliberately left to the merge per the shared-tree rules):
//!  * `vk/mod.rs` declares this module: `mod exposure;` (re-export `ExposureState`
//!    / `ExposureShared` as needed), and `skeleton.rs`'s opaque `ExposureRing`
//!    (+ its inherent impl) is dropped in favour of the real one here.
//!  * `Renderer` gains a field `exposure: ExposureState`, built in `Renderer::new`
//!    with `ExposureState::new(device, memory_props, render_extent, cache)` and
//!    rebuilt on resize; destroyed in `Renderer::destroy`.
//!  * `draw_frame` calls `self.record_exposure_pass(cmd, FrameSlot::new(slot))`
//!    AFTER the HDR geometry/sky pass and BEFORE the tonemap pass.
//!  * `Engine` gains `exposure_shared: ExposureShared`, a clone of
//!    `ExposureState::shared()`, handed across at construction so the main-thread
//!    `compose()` can read the render thread's latest metered value.
//!  * `build.rs` compiles `shaders/exposure_reduce.comp.slang` → `exposure_reduce.comp.spv`.
//!  * The tonemap push (`camera::WarpPush`) gains a trailing `dither_phase: f32`,
//!    and `vk/mod.rs` sources the tonemap `exposure` from `exposure_shared` and
//!    the `dither_phase` from `DITHER_PHASE_16[frame % 16]` (== FrameUniforms
//!    `exposure_dither.y`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

use super::alloc::find_memory_type;
use super::buffers::FRAMES_IN_FLIGHT;
use crate::engine::Engine;
use crate::genconst;
use crate::skeleton::{Exposure, ExposureRead, ExposureWrite, FrameSlot};

const EXPOSURE_REDUCE_COMP: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/exposure_reduce.comp.spv"));

/// One tile = this many HDR texels per side (mirrors `TILE` in the shader).
const TILE: u32 = 16;

/// EV-space exposure curve with fixed point at day luma; constants from build.rs.
fn exposure_curve(log2_luma: f32) -> f32 {
    (genconst::EXPOSURE_EV_SLOPE * (genconst::EXPOSURE_L_DAY_LOG2 - log2_luma))
        .exp2()
        .clamp(genconst::EXPOSURE_CLAMP_LO, genconst::EXPOSURE_CLAMP_HI)
}

/// Local SPIR-V module builder (pipeline.rs's is private to its module).
fn create_shader_module(device: &ash::Device, bytes: &[u8]) -> vk::ShaderModule {
    let code =
        ash::util::read_spv(&mut std::io::Cursor::new(bytes)).expect("invalid embedded SPIR-V");
    unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .expect("create exposure shader module")
    }
}

/// Push constants for `exposure_reduce.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ExposurePush {
    hdr_dim: [u32; 2],
    tiles: [u32; 2],
}

// Double-buffered tile-mean buffers for per-slot GPU readback.

/// One slot's host-visible, persistently-mapped tile-mean buffer (mean log2
/// luma per tile). Host-coherent so no flush/invalidate is needed on readback.
struct TileMeans {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut f32,
}

impl TileMeans {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        tile_count: usize,
    ) -> TileMeans {
        let size = (tile_count.max(1) * size_of::<f32>()) as u64;
        let buffer = unsafe {
            device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        // Written by the compute reduction, read back by the CPU.
                        .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .expect("create exposure tile-mean buffer")
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
                .expect("allocate exposure tile-mean memory")
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind exposure tile-mean memory");
        }
        let mapped = unsafe {
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map exposure tile-mean memory") as *mut f32
        };
        // Start neutral (log2(1.0) == 0) so the very first frame reads a sane value.
        unsafe { std::ptr::write_bytes(mapped, 0, tile_count.max(1)) };
        TileMeans {
            buffer,
            memory,
            mapped,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.memory);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Per-slot exposure readback. The compute reduction writes THIS frame's slot
/// buffer; the CPU reads THE SAME slot — whose previous write belongs to the
/// frame two-back, the exact frame whose fence [`wait_slot_and_reclaim`] just
/// waited. Reading `s.other()` (the old rule) touched the immediately previous
/// frame's buffer, which can still be in flight: a torn, finite-but-wrong
/// readback. The read happens at record time, BEFORE this frame's dispatch is
/// submitted, so read-then-overwrite on one buffer is race-free by
/// construction. The parity rule lives in exactly one function
/// ([`slot_parity`], used by [`ExposureRing::views`]); callers cannot mix the
/// views.
///
/// [`wait_slot_and_reclaim`]: super::Renderer::wait_slot_and_reclaim
pub struct ExposureRing {
    slots: [TileMeans; FRAMES_IN_FLIGHT as usize],
    tile_count: usize,
}

/// The slot-index parity rule as a pure `(write, read)` mapping — extracted so
/// the fence-safety argument is unit-testable without a device: the read index
/// must name a buffer whose last GPU write is fence-proven complete, and the
/// only such buffer under 2-frames-in-flight is the waited slot's own.
fn slot_parity(waited: FrameSlot) -> (usize, usize) {
    (waited.index(), waited.index())
}

impl ExposureRing {
    fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        tile_count: usize,
    ) -> ExposureRing {
        ExposureRing {
            slots: std::array::from_fn(|_| TileMeans::new(device, memory_props, tile_count)),
            tile_count,
        }
    }

    /// The parity resolver: the CPU reads the waited slot's buffer (its value
    /// is from the frame two-back, fence-proven), then the compute pass
    /// overwrites that same buffer. Baked HERE so no call site can mix views.
    pub fn views(&self, s: FrameSlot) -> (ExposureWrite<'_>, ExposureRead<'_>) {
        let (write, read) = slot_parity(s);
        (
            ExposureWrite {
                buf: &self.slots[write].buffer,
            },
            ExposureRead {
                buf: &self.slots[read].buffer,
            },
        )
    }

    /// CPU readback of the buffer a `views(_).1` read-view points at, plus the
    /// exposure curve and temporal smoothing. Returns the exposure the next
    /// `compose()` should carry. Host-coherent memory → the mapped floats are up
    /// to date once the slot's fence has cleared (the caller's responsibility).
    pub fn metered(&self, read: ExposureRead<'_>, dt: f32, prev: Exposure) -> Exposure {
        let slot = self
            .slots
            .iter()
            .find(|t| t.buffer == *read.buf)
            .expect("ExposureRead must originate from this ring");

        // Average tile means = mean log2 luma; exp2 → geometric-mean scene luma.
        let mut sum = 0.0f32;
        for i in 0..self.tile_count {
            sum += unsafe { *slot.mapped.add(i) };
        }
        let mean_log2 = if self.tile_count > 0 {
            sum / self.tile_count as f32
        } else {
            0.0
        };
        // Defense in depth: the read buffer's last write is fence-proven
        // complete (the parity resolver reads the waited slot), so a torn read
        // should be impossible — but a NaN from a degenerate HDR frame (e.g.
        // an all-inf tile) must still not poison the filter forever. Hold
        // `prev` for a frame instead.
        if !mean_log2.is_finite() {
            return prev;
        }
        let target = exposure_curve(mean_log2).max(1e-3); // exposure is a positive multiplier
        // mix(new, prev, exp(-dt·1.25)): large dt → mostly new (fast adapt on a
        // long frame), small dt → mostly prev (smooth). Frame-rate independent.
        let k = (-dt * 1.25).exp();
        Exposure(target * (1.0 - k) + prev.0 * k)
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        for slot in &self.slots {
            unsafe { slot.destroy(device) };
        }
    }
}

// SAFETY: the mapped pointers are only dereferenced on the render thread (the
// ring lives inside the render-thread-owned `Renderer`); `Send` is needed only
// because `Renderer` is constructed on and moved to that thread.
unsafe impl Send for ExposureRing {}

// ============================================================================
// Compute pipeline
// ============================================================================

/// The metering compute pipeline plus the sampler it reads the HDR target
/// through. Set 0 is push-descriptor: binding 0 = HDR (combined image sampler),
/// binding 1 = tile-mean storage buffer.
struct ExposureCompute {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    sampler: vk::Sampler,
}

impl ExposureCompute {
    fn new(device: &ash::Device, cache: vk::PipelineCache) -> ExposureCompute {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
                .expect("create exposure set layout")
        };
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(size_of::<ExposurePush>() as u32)];
        let set_layouts = [set_layout];
        let layout = unsafe {
            device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&set_layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .expect("create exposure pipeline layout")
        };

        let module = create_shader_module(device, EXPOSURE_REDUCE_COMP);
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .module(module)
            .name(c"main")
            .stage(vk::ShaderStageFlags::COMPUTE);
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(cache, &[info], None)
                .map_err(|(_, err)| err)
                .expect("create exposure compute pipeline")[0]
        };
        unsafe { device.destroy_shader_module(module, None) };

        // Linear clamp: tile means smooth a little over the box; edge clamp keeps
        // the border tiles from wrapping.
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
                .expect("create exposure HDR sampler")
        };

        ExposureCompute {
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

// ============================================================================
// Cross-thread published exposure
// ============================================================================

/// The latest metered exposure, published by the render thread and read by the
/// main thread's `compose()`. A single `f32` in an atomic — exposure changes
/// slowly and a torn-free scalar is all `compose` needs.
#[derive(Clone)]
pub struct ExposureShared(Arc<AtomicU32>);

impl ExposureShared {
    fn new() -> ExposureShared {
        ExposureShared(Arc::new(AtomicU32::new(Exposure::DEFAULT.0.to_bits())))
    }
    fn store(&self, e: Exposure) {
        self.0.store(e.0.to_bits(), Ordering::Relaxed);
    }
    pub fn load(&self) -> Exposure {
        Exposure(f32::from_bits(self.0.load(Ordering::Relaxed)))
    }
}

// ============================================================================
// ExposureState — render-thread owner of the whole pass
// ============================================================================

pub(crate) struct ExposureState {
    compute: ExposureCompute,
    ring: ExposureRing,
    /// Tile-grid dimensions = ceil(hdr_dim / TILE).
    tiles: vk::Extent2D,
    hdr_dim: vk::Extent2D,
    /// Last smoothed exposure (render-thread state, fed back into `metered`).
    exposure: Exposure,
    /// Wall-clock of the previous metering, for the temporal-smoothing `dt`
    /// (the render thread owns its own cadence; there is no game `dt` here).
    last: std::time::Instant,
    shared: ExposureShared,
}

impl ExposureState {
    pub(crate) fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        render_extent: vk::Extent2D,
        cache: vk::PipelineCache,
    ) -> ExposureState {
        let tiles = vk::Extent2D {
            width: render_extent.width.div_ceil(TILE).max(1),
            height: render_extent.height.div_ceil(TILE).max(1),
        };
        let tile_count = (tiles.width * tiles.height) as usize;
        ExposureState {
            compute: ExposureCompute::new(device, cache),
            ring: ExposureRing::new(device, memory_props, tile_count),
            tiles,
            hdr_dim: render_extent,
            exposure: Exposure::DEFAULT,
            last: std::time::Instant::now(),
            shared: ExposureShared::new(),
        }
    }

    /// Clone of the published cell for the main thread (`Engine`).
    pub(crate) fn shared(&self) -> ExposureShared {
        self.shared.clone()
    }

    /// Metering was switched OFF: the public contract says exposure pins to
    /// 1.0, so publish [`Exposure::DEFAULT`] everywhere the stale meter could
    /// otherwise linger (tonemap reads `current()`, `compose()` reads the
    /// shared cell) and reset the smoothing clock.
    pub(crate) fn reset(&mut self) {
        self.exposure = Exposure::DEFAULT;
        self.shared.store(Exposure::DEFAULT);
        self.last = std::time::Instant::now();
    }

    /// Metering was switched back ON: restart the smoothing clock so the first
    /// `dt` measures from the re-enable, not from whenever metering last ran.
    pub(crate) fn rearm(&mut self) {
        self.last = std::time::Instant::now();
    }

    /// The render thread's latest metered+smoothed exposure — the multiplier the
    /// tonemap pass applies before its curve. Same value published to
    /// [`ExposureShared`], read directly here since tonemap runs render-side.
    pub(crate) fn current(&self) -> Exposure {
        self.exposure
    }

    /// Rebuild the extent-dependent GPU resources (the tile-mean ring) after a
    /// resize. The compute pipeline is extent-independent (dims arrive by push),
    /// and the published `ExposureShared` cell + smoothed `exposure` are kept so
    /// the main thread's `compose()` reads an unbroken value. The caller has
    /// already `device_wait_idle`'d.
    pub(crate) fn recreate(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        render_extent: vk::Extent2D,
    ) {
        let tiles = vk::Extent2D {
            width: render_extent.width.div_ceil(TILE).max(1),
            height: render_extent.height.div_ceil(TILE).max(1),
        };
        let tile_count = (tiles.width * tiles.height) as usize;
        unsafe { self.ring.destroy(device) };
        self.ring = ExposureRing::new(device, memory_props, tile_count);
        self.tiles = tiles;
        self.hdr_dim = render_extent;
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            self.compute.destroy(device);
            self.ring.destroy(device);
        }
    }
}

// ============================================================================
// Seams (impls on foreign-owned types — see module MERGE NOTES)
// ============================================================================

impl super::Renderer {
    /// Record the metering reduction and fold the previous result into the
    /// published exposure. Inserted by the orchestrator AFTER the HDR pass and
    /// BEFORE tonemap. `slot` is this frame's slot; `views(slot)` reads and
    /// then overwrites the SAME slot's buffer — the read value is the one this
    /// slot's fence (already waited this frame) proves complete, and the read
    /// happens at record time, before the overwriting dispatch is submitted.
    ///
    /// This pass owns the offscreen's `COLOR_ATTACHMENT → SHADER_READ_ONLY`
    /// transition (it samples the HDR for the reduction and leaves it sampled),
    /// so it returns the [`super::HdrReadable`] proof the tonemap present-copy
    /// requires.
    pub(crate) fn record_exposure_pass(
        &mut self,
        cmd: vk::CommandBuffer,
        slot: FrameSlot,
    ) -> super::HdrReadable {
        let device = &self.device.device;
        let exp = &mut self.exposure;

        // Parity resolver: read the waited slot's buffer (frame N-2's result,
        // fence-proven), then let this frame's dispatch overwrite it.
        let now = std::time::Instant::now();
        let dt = now.duration_since(exp.last).as_secs_f32();
        exp.last = now;
        let metered = {
            let (_write, read) = exp.ring.views(slot);
            exp.ring.metered(read, dt, exp.exposure)
        };
        exp.exposure = metered;
        exp.shared.store(metered);
        // The buffer the compute dispatch fills this frame (read next frame).
        let write_buf = *exp.ring.views(slot).0.buf;

        let hdr = &self.targets.offscreen[slot.index()];
        let tiles = exp.tiles;
        unsafe {
            // HDR color-attachment write → compute sampled read. The buffer is
            // host-coherent + freshly fence-cleared, so no pre-barrier is needed
            // for the write target beyond the frame fence already waited.
            let pre = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(hdr.image)
                .subresource_range(super::color_range())];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&pre),
            );

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, exp.compute.pipeline);
            let hdr_info = [vk::DescriptorImageInfo::default()
                .sampler(exp.compute.sampler)
                .image_view(hdr.view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let buf_info = [vk::DescriptorBufferInfo::default()
                .buffer(write_buf)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&hdr_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                exp.compute.layout,
                0,
                &writes,
            );

            let push = ExposurePush {
                hdr_dim: [exp.hdr_dim.width, exp.hdr_dim.height],
                tiles: [tiles.width, tiles.height],
            };
            device.cmd_push_constants(
                cmd,
                exp.compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, tiles.width.div_ceil(8), tiles.height.div_ceil(8), 1);

            // Compute storage write → host read (next cycle) and HDR back to
            // color-attachment layout for the tonemap sample.
            let post = [
                vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ),
            ];
            let img_post = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(hdr.image)
                .subresource_range(super::color_range())];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default()
                    .memory_barriers(&post)
                    .image_memory_barriers(&img_post),
            );
        }
        // The offscreen is now in SHADER_READ_ONLY for the tonemap present-copy.
        super::HdrReadable::new(slot.index())
    }
}

impl Engine {
    /// The exposure `compose()` folds into this frame's `FrameSnapshot`:
    /// the render thread's latest metered+smoothed value — or the pinned
    /// [`Exposure::DEFAULT`] whenever metering is disabled, so the public
    /// "off pins exposure at 1.0" contract holds on the main thread too (the
    /// render-side reset covers the tonemap; this covers `compose()`). The
    /// temporal smoothing (which needs the readback cadence) already happened
    /// render-side, so `dt` is unused here — the seam keeps it for symmetry
    /// with `metered`.
    pub fn exposure_for_compose(&mut self, _dt: f32) -> Exposure {
        if !self.flags.exposure {
            return Exposure::DEFAULT;
        }
        self.exposure_shared.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fence-safety argument for the readback, as a pure timeline model:
    /// frame `i` records on slot `i % 2` after waiting THAT slot's fence, so
    /// the only buffer whose last GPU write is provably complete is the waited
    /// slot's own (written by frame `i - 2`). Reading any buffer written more
    /// recently — the old `s.other()` rule read frame `i - 1`'s — races an
    /// in-flight dispatch.
    #[test]
    fn readback_reads_only_fence_proven_buffers() {
        let mut last_writer: [Option<usize>; FRAMES_IN_FLIGHT as usize] = [None, None];
        for frame in 0..64 {
            let slot = FrameSlot::new(frame % FRAMES_IN_FLIGHT as usize);
            let (write, read) = slot_parity(slot);
            if let Some(writer) = last_writer[read] {
                assert!(
                    writer + FRAMES_IN_FLIGHT as usize <= frame,
                    "frame {frame} reads a buffer last written by frame {writer}, \
                     which the slot fence does not prove complete"
                );
            }
            last_writer[write] = Some(frame);
        }
    }
}
