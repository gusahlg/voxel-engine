/// RGBA8 texture array with layer-capacity headroom. Layer 0 is always white.
/// In-place palette growth uploads only new/changed layers on the transfer
/// lane; the sampled view is not recreated. Descriptor set lives in the
/// Renderer; texture swaps only rewrite the set, no pipeline rebuild needed.
use ash::vk;

use super::alloc::find_memory_type;
use super::buffers::RetireQueue;
use super::device::Anisotropy;
use super::image_upload::{ImageUpload, upload_image};
use super::timeline::{Timeline, TimelineValue};
use super::transfer::TransferLane;

/// Floor on allocated array layers so mid-game palette growth does not
/// recreate the image. Capped by `limits.maxImageArrayLayers`.
pub(crate) const MIN_LAYER_CAPACITY: u32 = 64;

/// Stages that first sample the block texture array (mesh3d fragment).
pub(crate) const BLOCK_TEXTURE_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::FRAGMENT_SHADER;

pub struct BlockTextures {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    /// Shader-visible used layer count (`layer` in [`crate::MeshVertex`]).
    pub layers: u32,
    pub size: u32,
    capacity: u32,
    mip_levels: u32,
    /// CPU palette; used for diffs and for a capacity-overflow rebuild.
    layer_pixels: Vec<Vec<u8>>,
    /// High-water of layers written on the GPU (UNDEFINED vs SHADER_READ).
    written: u32,
    pending: Vec<PendingLayer>,
    command_pool: vk::CommandPool,
    anisotropy: Option<Anisotropy>,
    max_layers: u32,
    pending_grow: Option<PendingGrow>,
    staging_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    transfer_retire: RetireQueue<(vk::Buffer, vk::DeviceMemory)>,
    /// Dedicated-family overwrite release: throwaway graphics timeline + CB.
    /// Not the render timeline — that would reserve a value after the frame
    /// already reserved its signal and submit out of order.
    release_cmds: RetireQueue<(Timeline, vk::CommandBuffer)>,
}

struct PendingLayer {
    index: u32,
    pixels: Vec<u8>,
}

struct PendingGrow {
    size: u32,
    layers: Vec<Vec<u8>>,
    /// Layers GPU-copied from the bound image (`0` if the texel size changed).
    copied: u32,
    staging: Vec<PendingLayer>,
}

/// GPU objects of a superseded array; destroyed after the timeline value of
/// the last frame that sampled (or copied from) it.
pub(crate) struct RetiredBlockTextures {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

impl RetiredBlockTextures {
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Result of flushing pending uploads/grows into the next frame's command buffer.
pub(crate) struct TextureFlush {
    pub transfer_wait: Option<TimelineValue>,
    pub retire: Option<(TimelineValue, RetiredBlockTextures)>,
}

impl TextureFlush {
    fn none() -> Self {
        Self {
            transfer_wait: None,
            retire: None,
        }
    }
}

/// Allocated array-layer count: next power of two ≥ `requested`, at least
/// [`MIN_LAYER_CAPACITY`], never above `device_limit`.
pub(crate) fn layer_capacity(requested: u32, device_limit: u32) -> u32 {
    let requested = requested.max(1);
    let limit = device_limit.max(1);
    if requested >= limit {
        return limit;
    }
    requested
        .next_power_of_two()
        .max(MIN_LAYER_CAPACITY)
        .min(limit)
}

/// Indices in `new` whose bytes differ from `old` (append is `old.len()..`).
/// Shrinking `new` does not list dropped tail indices — those layers stay on
/// the GPU unused.
pub(crate) fn changed_layer_indices(old: &[Vec<u8>], new: &[Vec<u8>]) -> Vec<u32> {
    let mut out = Vec::new();
    for (i, layer) in new.iter().enumerate() {
        if old.get(i) != Some(layer) {
            out.push(i as u32);
        }
    }
    out
}

/// How a grow splits work: GPU-copy `copied` prefix layers (same texel size),
/// staging-upload the returned indices (tail, plus any changed prefix).
pub(crate) fn grow_plan(
    old_size: u32,
    old_written: u32,
    old_pixels: &[Vec<u8>],
    new_size: u32,
    new_layers: &[Vec<u8>],
) -> (u32, Vec<u32>) {
    let copied = if new_size == old_size {
        old_written.min(new_layers.len() as u32)
    } else {
        0
    };
    let mut staging = Vec::new();
    for (i, layer) in new_layers.iter().enumerate() {
        let idx = i as u32;
        if idx < copied {
            if old_pixels.get(i) != Some(layer) {
                staging.push(idx);
            }
        } else {
            staging.push(idx);
        }
    }
    (copied, staging)
}

/// Inclusive-contiguous runs `(base, count)` from a set of layer indices.
pub(crate) fn consecutive_runs(mut indices: Vec<u32>) -> Vec<(u32, u32)> {
    if indices.is_empty() {
        return Vec::new();
    }
    indices.sort_unstable();
    indices.dedup();
    let mut runs = Vec::new();
    let mut start = indices[0];
    let mut prev = indices[0];
    for &i in &indices[1..] {
        if i == prev + 1 {
            prev = i;
        } else {
            runs.push((start, prev - start + 1));
            start = i;
            prev = i;
        }
    }
    runs.push((start, prev - start + 1));
    runs
}

fn color_range(mip_levels: u32, base_layer: u32, layer_count: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mip_levels,
        base_array_layer: base_layer,
        layer_count,
    }
}

impl BlockTextures {
    /// 1x1, one all-white layer — the init-time placeholder, with capacity
    /// headroom so the first palette appends do not recreate the image
    /// unless the texel size changes.
    #[allow(clippy::too_many_arguments)]
    pub fn new_default(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        command_pool: vk::CommandPool,
        lane: &mut TransferLane,
        anisotropy: Option<Anisotropy>,
        max_layers: u32,
    ) -> Self {
        Self::upload(
            instance,
            device,
            physical,
            graphics_queue,
            graphics_family,
            command_pool,
            lane,
            anisotropy,
            1,
            &[vec![255, 255, 255, 255]],
            max_layers,
        )
    }

    /// Uploads `layers` RGBA8 images of `size`x`size` as a device-local
    /// texture array with a full CPU-built mip chain per layer. The image is
    /// created with [`layer_capacity`] array layers; only `layers.len()` are
    /// written. Blocks until the copy completes (init / size-change / overflow).
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        command_pool: vk::CommandPool,
        lane: &mut TransferLane,
        anisotropy: Option<Anisotropy>,
        size: u32,
        layers: &[Vec<u8>],
        max_layers: u32,
    ) -> Self {
        assert!(size >= 1, "block texture size must be >= 1");
        assert!(!layers.is_empty(), "block texture array needs >= 1 layer");
        let layer_bytes = size as usize * size as usize * 4;
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                layer_bytes,
                "layer {i}: expected {size}x{size} RGBA8 = {layer_bytes} bytes"
            );
        }
        let layer_count = layers.len() as u32;
        let capacity = layer_capacity(layer_count, max_layers);
        let mip_levels = 32 - size.leading_zeros(); // log2 size, clamped to 1x1

        // CPU mip chains, then packed mip-major so each mip level is one
        // buffer->image copy covering all *used* layers.
        let chains: Vec<Vec<Vec<u8>>> = layers
            .iter()
            .map(|base| build_mip_chain(base, size, mip_levels))
            .collect();
        let mut staging_data = Vec::new();
        let mut mip_offsets = Vec::with_capacity(mip_levels as usize);
        for mip in 0..mip_levels as usize {
            mip_offsets.push(staging_data.len() as u64);
            for chain in &chains {
                staging_data.extend_from_slice(&chain[mip]);
            }
        }
        let regions: Vec<vk::BufferImageCopy> = (0..mip_levels)
            .map(|mip| {
                let extent = (size >> mip).max(1);
                vk::BufferImageCopy::default()
                    .buffer_offset(mip_offsets[mip as usize])
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: mip,
                        base_array_layer: 0,
                        layer_count,
                    })
                    .image_extent(vk::Extent3D {
                        width: extent,
                        height: extent,
                        depth: 1,
                    })
            })
            .collect();
        let (image, memory, view) = upload_image(
            instance,
            device,
            physical,
            graphics_queue,
            graphics_family,
            command_pool,
            lane,
            &ImageUpload {
                extent: vk::Extent2D {
                    width: size,
                    height: size,
                },
                // sRGB: the sampler hardware-decodes texels to linear light, and
                // bilinear/mip/aniso filtering happens in linear. The tonemap
                // pass owns the OETF back to display.
                format: vk::Format::R8G8B8A8_SRGB,
                mip_levels,
                array_layers: capacity,
                view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                bytes: &staging_data,
                regions: &regions,
            },
        );

        // NEAREST texels (crisp voxel look), LINEAR between mips, REPEAT so
        // greedy-meshed quads tile per block. Anisotropy (when supported) takes
        // multiple NEAREST footprint samples, killing grazing-angle shimmer on
        // distant terrain without softening the blocky look.
        let mut sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);
        if let Some(a) = anisotropy {
            sampler_info = sampler_info
                .anisotropy_enable(true)
                .max_anisotropy(a.clamp(8.0));
        }
        let sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create block texture sampler")
        };

        Self {
            image,
            memory,
            view,
            sampler,
            layers: layer_count,
            size,
            capacity,
            mip_levels,
            layer_pixels: layers.to_vec(),
            written: layer_count,
            pending: Vec::new(),
            command_pool,
            anisotropy,
            max_layers,
            pending_grow: None,
            staging_retire: RetireQueue::new(),
            transfer_retire: RetireQueue::new(),
            release_cmds: RetireQueue::new(),
        }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn palette(&self) -> &[Vec<u8>] {
        &self.layer_pixels
    }

    /// Same texel size and `layer_count` within the allocated capacity.
    /// False while a grow is queued so a later set coalesces into that grow.
    pub fn can_update_in_place(&self, size: u32, layer_count: u32) -> bool {
        self.pending_grow.is_none()
            && size == self.size
            && layer_count >= 1
            && layer_count <= self.capacity
    }

    /// Pending writes to layers the GPU already sampled (need overwrite sync).
    pub fn has_overwrite_pending(&self) -> bool {
        self.pending.iter().any(|p| p.index < self.written)
    }

    pub fn has_garbage(&self) -> bool {
        !self.staging_retire.is_empty()
            || !self.transfer_retire.is_empty()
            || !self.release_cmds.is_empty()
    }

    /// Diff `layers` against the bound palette and queue only changed indices.
    pub fn queue_set(&mut self, layers: &[Vec<u8>]) {
        let layer_bytes = self.size as usize * self.size as usize * 4;
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                layer_bytes,
                "layer {i}: expected {}x{} RGBA8 = {layer_bytes} bytes",
                self.size,
                self.size
            );
        }
        let changed = changed_layer_indices(&self.layer_pixels, layers);
        for &i in &changed {
            self.queue_pending(i, layers[i as usize].clone());
        }
        self.layer_pixels = layers.to_vec();
        self.layers = layers.len() as u32;
        self.pending.retain(|p| p.index < self.layers);
    }

    /// Queue a realloc (texel-size change or capacity overflow). Applied at
    /// the next frame: GPU-copy existing layers when the size matches, upload
    /// the rest, switch the sampled view, retire the old image on the timeline.
    pub fn queue_grow(&mut self, size: u32, layers: Vec<Vec<u8>>) {
        assert!(size >= 1, "block texture size must be >= 1");
        assert!(!layers.is_empty(), "block texture array needs >= 1 layer");
        let layer_bytes = size as usize * size as usize * 4;
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                layer_bytes,
                "layer {i}: expected {size}x{size} RGBA8 = {layer_bytes} bytes"
            );
        }
        let (copied, staging_idx) =
            grow_plan(self.size, self.written, &self.layer_pixels, size, &layers);
        let staging = staging_idx
            .into_iter()
            .map(|index| PendingLayer {
                index,
                pixels: layers[index as usize].clone(),
            })
            .collect();
        self.pending.clear();
        self.layers = layers.len() as u32;
        self.layer_pixels = layers.clone();
        self.pending_grow = Some(PendingGrow {
            size,
            layers,
            copied,
            staging,
        });
    }

    /// Append `new_layers` at the current texel size. `Ok` if they fit in
    /// capacity (after clamping to `device_limit`). `Err` if a bigger image
    /// is required; the palette is left unchanged so the caller can rebuild.
    pub fn try_append(&mut self, new_layers: &[Vec<u8>], device_limit: u32) -> Result<(), ()> {
        if new_layers.is_empty() {
            return Ok(());
        }
        if let Some(grow) = &mut self.pending_grow {
            let layer_bytes = grow.size as usize * grow.size as usize * 4;
            for (i, layer) in new_layers.iter().enumerate() {
                assert_eq!(
                    layer.len(),
                    layer_bytes,
                    "append layer {i}: expected {}x{} RGBA8 = {layer_bytes} bytes",
                    grow.size,
                    grow.size
                );
            }
            let used = grow.layers.len() as u32;
            let room = device_limit.saturating_sub(used) as usize;
            if new_layers.len() > room {
                log::error!(
                    "append_block_textures: {} layers would exceed the device cap of {device_limit}; truncating",
                    used as usize + new_layers.len()
                );
            }
            let start = grow.layers.len() as u32;
            for (i, layer) in new_layers.iter().take(room).enumerate() {
                grow.staging.push(PendingLayer {
                    index: start + i as u32,
                    pixels: layer.clone(),
                });
                grow.layers.push(layer.clone());
            }
            self.layer_pixels = grow.layers.clone();
            self.layers = grow.layers.len() as u32;
            return Ok(());
        }
        let layer_bytes = self.size as usize * self.size as usize * 4;
        for (i, layer) in new_layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                layer_bytes,
                "append layer {i}: expected {}x{} RGBA8 = {layer_bytes} bytes",
                self.size,
                self.size
            );
        }
        let room = device_limit.saturating_sub(self.layers) as usize;
        if new_layers.len() > room {
            log::error!(
                "append_block_textures: {} layers would exceed the device cap of {device_limit}; truncating",
                self.layers as usize + new_layers.len()
            );
        }
        let new_layers = &new_layers[..new_layers.len().min(room)];
        if new_layers.is_empty() {
            return Ok(());
        }
        let new_used = self.layers + new_layers.len() as u32;
        if new_used > self.capacity {
            return Err(());
        }
        let start = self.layers;
        for (i, layer) in new_layers.iter().enumerate() {
            self.queue_pending(start + i as u32, layer.clone());
        }
        self.layer_pixels.extend(new_layers.iter().cloned());
        self.layers = new_used;
        Ok(())
    }

    fn queue_pending(&mut self, index: u32, pixels: Vec<u8>) {
        if let Some(p) = self.pending.iter_mut().find(|p| p.index == index) {
            p.pixels = pixels;
        } else {
            self.pending.push(PendingLayer { index, pixels });
        }
    }

    /// Record pending per-layer copies on the transfer lane (or the frame
    /// command buffer on `SameQueueFallback`). Grows that exceed capacity run
    /// on the graphics command buffer (`vkCmdCopyImage` of existing layers).
    /// No host wait.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn flush(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        lane: &mut TransferLane,
        graphics_cmd: vk::CommandBuffer,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        graphics_timeline: &Timeline,
        last_render_value: TimelineValue,
        done_at: TimelineValue,
    ) -> TextureFlush {
        if let Some(grow) = self.pending_grow.take() {
            return unsafe {
                self.flush_grow(instance, device, physical, graphics_cmd, done_at, grow)
            };
        }
        if self.pending.is_empty() {
            return TextureFlush::none();
        }
        let mut pending = std::mem::take(&mut self.pending);
        pending.sort_by_key(|p| p.index);
        let overwrite: Vec<u32> = pending
            .iter()
            .filter(|p| p.index < self.written)
            .map(|p| p.index)
            .collect();
        let fresh: Vec<u32> = pending
            .iter()
            .filter(|p| p.index >= self.written)
            .map(|p| p.index)
            .collect();

        let packed = pack_pending(self.size, self.mip_levels, &pending);
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let (staging, staging_mem) =
            unsafe { create_filled_staging(device, &memory_props, &packed.bytes) };

        let separate_queue = lane.is_separate_queue();
        let needs_qfot = lane.needs_ownership_transfer();
        let mip_levels = self.mip_levels;
        let image = self.image;

        let extra_wait = if !overwrite.is_empty() && needs_qfot {
            Some(unsafe {
                self.submit_overwrite_release(
                    device,
                    graphics_queue,
                    graphics_family,
                    lane.family(),
                    done_at,
                    &overwrite,
                )
            })
        } else if !overwrite.is_empty() && separate_queue {
            Some((
                graphics_timeline.semaphore(),
                last_render_value,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ))
        } else {
            None
        };

        let lane_batch = separate_queue.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());

        let mut to_dst = Vec::new();
        if needs_qfot {
            for (base, count) in consecutive_runs(overwrite.clone()) {
                to_dst.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .src_queue_family_index(graphics_family)
                        .dst_queue_family_index(lane.family())
                        .image(image)
                        .subresource_range(color_range(mip_levels, base, count)),
                );
            }
        } else {
            let src_stage = if separate_queue {
                vk::PipelineStageFlags2::NONE
            } else {
                vk::PipelineStageFlags2::FRAGMENT_SHADER
            };
            let src_access = if separate_queue {
                vk::AccessFlags2::NONE
            } else {
                vk::AccessFlags2::SHADER_SAMPLED_READ
            };
            for (base, count) in consecutive_runs(overwrite.clone()) {
                to_dst.push(
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(src_stage)
                        .src_access_mask(src_access)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .image(image)
                        .subresource_range(color_range(mip_levels, base, count)),
                );
            }
        }
        for (base, count) in consecutive_runs(fresh.clone()) {
            to_dst.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(image)
                    .subresource_range(color_range(mip_levels, base, count)),
            );
        }
        unsafe {
            if !to_dst.is_empty() {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_dst),
                );
            }
            for (_index, regions) in &packed.per_layer {
                device.cmd_copy_buffer_to_image(
                    record_cmd,
                    staging,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                );
            }
        }

        let all_indices: Vec<u32> = pending.iter().map(|p| p.index).collect();
        let mut to_sampled = Vec::new();
        for (base, count) in consecutive_runs(all_indices) {
            let mut b = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(color_range(mip_levels, base, count));
            if needs_qfot {
                b = b
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .src_queue_family_index(lane.family())
                    .dst_queue_family_index(graphics_family);
            } else if separate_queue {
                b = b
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE);
            } else {
                b = b
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ);
            }
            to_sampled.push(b);
        }
        unsafe {
            if !to_sampled.is_empty() {
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
                );
            }
        }

        let arrived_at = if let Some(lane_batch) = lane_batch {
            let value = unsafe { lane.submit_after(device, lane_batch, extra_wait) };
            if needs_qfot {
                let mut acquire = Vec::new();
                let uploaded: Vec<u32> = pending.iter().map(|p| p.index).collect();
                for (base, count) in consecutive_runs(uploaded) {
                    acquire.push(
                        vk::ImageMemoryBarrier2::default()
                            .src_stage_mask(vk::PipelineStageFlags2::NONE)
                            .src_access_mask(vk::AccessFlags2::NONE)
                            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                            .src_queue_family_index(lane.family())
                            .dst_queue_family_index(graphics_family)
                            .image(image)
                            .subresource_range(color_range(mip_levels, base, count)),
                    );
                }
                unsafe {
                    if !acquire.is_empty() {
                        device.cmd_pipeline_barrier2(
                            graphics_cmd,
                            &vk::DependencyInfo::default().image_memory_barriers(&acquire),
                        );
                    }
                }
            }
            self.transfer_retire.push(value, (staging, staging_mem));
            Some(value)
        } else {
            self.staging_retire.push(done_at, (staging, staging_mem));
            None
        };

        if let Some(max_idx) = pending.iter().map(|p| p.index).max() {
            self.written = self.written.max(max_idx + 1);
        }
        TextureFlush {
            transfer_wait: arrived_at,
            retire: None,
        }
    }

    /// Create a larger (or differently sized) array, copy existing layers
    /// GPU-side when the texel size matches, upload the rest from staging,
    /// and switch the sampled view. Old image is retired after `done_at`.
    unsafe fn flush_grow(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_cmd: vk::CommandBuffer,
        done_at: TimelineValue,
        grow: PendingGrow,
    ) -> TextureFlush {
        let new_used = grow.layers.len() as u32;
        let new_capacity = layer_capacity(new_used, self.max_layers);
        let (new_image, new_memory, new_view, new_sampler, new_mips) = create_gpu_array(
            instance,
            device,
            physical,
            self.anisotropy,
            grow.size,
            new_capacity,
        );

        let copied = grow.copied.min(new_used);
        let staging_layers = grow.staging;

        // New image: UNDEFINED → TRANSFER_DST on every layer this grow writes.
        let mut barriers = vec![
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(new_image)
                .subresource_range(color_range(new_mips, 0, new_used)),
        ];
        if copied > 0 {
            // In-flight frames sample the old array as SHADER_READ. This later
            // graphics CB's barrier waits that fragment work (same-queue
            // submission order) before TRANSFER_SRC.
            barriers.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .image(self.image)
                    .subresource_range(color_range(self.mip_levels, 0, copied)),
            );
        }
        unsafe {
            device.cmd_pipeline_barrier2(
                graphics_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            if copied > 0 {
                let copies = image_copy_mips(self.size, self.mip_levels, copied);
                device.cmd_copy_image(
                    graphics_cmd,
                    self.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &copies,
                );
            }
        }

        let packed = if staging_layers.is_empty() {
            None
        } else {
            Some(pack_pending(grow.size, new_mips, &staging_layers))
        };
        if let Some(packed) = &packed {
            let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
            let (staging, staging_mem) =
                unsafe { create_filled_staging(device, &memory_props, &packed.bytes) };
            unsafe {
                for (_index, regions) in &packed.per_layer {
                    device.cmd_copy_buffer_to_image(
                        graphics_cmd,
                        staging,
                        new_image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        regions,
                    );
                }
            }
            self.staging_retire.push(done_at, (staging, staging_mem));
        }

        let to_sampled = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(new_image)
            .subresource_range(color_range(new_mips, 0, new_used))];
        unsafe {
            device.cmd_pipeline_barrier2(
                graphics_cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }

        let old = RetiredBlockTextures {
            image: self.image,
            memory: self.memory,
            view: self.view,
            sampler: self.sampler,
        };
        self.image = new_image;
        self.memory = new_memory;
        self.view = new_view;
        self.sampler = new_sampler;
        self.size = grow.size;
        self.capacity = new_capacity;
        self.mip_levels = new_mips;
        self.layers = new_used;
        self.layer_pixels = grow.layers;
        self.written = new_used;

        TextureFlush {
            transfer_wait: None,
            retire: Some((done_at, old)),
        }
    }

    /// Dedicated-family overwrite: release sampled layers on graphics so the
    /// transfer queue can acquire them. Submitted (not host-waited) after
    /// in-flight frames that sampled those layers.
    unsafe fn submit_overwrite_release(
        &mut self,
        device: &ash::Device,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        transfer_family: u32,
        done_at: TimelineValue,
        overwrite: &[u32],
    ) -> (vk::Semaphore, TimelineValue, vk::PipelineStageFlags2) {
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc)
                .expect("Failed to allocate block-texture release command buffer")[0]
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .expect("Failed to begin block-texture release command buffer");
        }
        let mut release = Vec::new();
        for (base, count) in consecutive_runs(overwrite.to_vec()) {
            release.push(
                vk::ImageMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                    .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::NONE)
                    .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_queue_family_index(graphics_family)
                    .dst_queue_family_index(transfer_family)
                    .image(self.image)
                    .subresource_range(color_range(self.mip_levels, base, count)),
            );
        }
        unsafe {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&release),
            );
            device
                .end_command_buffer(cmd)
                .expect("Failed to end block-texture release command buffer");
        }
        let mut tmp = unsafe { Timeline::new(device) };
        let rs = tmp.begin_render(cmd);
        let completion = unsafe { rs.submit(device, graphics_queue, &tmp, None) };
        let sem = tmp.semaphore();
        let value = completion.value();
        self.release_cmds.push(done_at, (tmp, cmd));
        (sem, value, vk::PipelineStageFlags2::COPY)
    }

    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        self.staging_retire
            .collect(current, |(buffer, memory)| unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
        let pool = self.command_pool;
        self.release_cmds
            .collect(current, |(timeline, cmd)| unsafe {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
    }

    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        self.transfer_retire
            .collect(current, |(buffer, memory)| unsafe {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.staging_retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            self.transfer_retire.collect_all(|(buffer, memory)| {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
            });
            let pool = self.command_pool;
            self.release_cmds.collect_all(|(timeline, cmd)| {
                timeline.destroy(device);
                device.free_command_buffers(pool, &[cmd]);
            });
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

fn create_gpu_array(
    instance: &ash::Instance,
    device: &ash::Device,
    physical: vk::PhysicalDevice,
    anisotropy: Option<Anisotropy>,
    size: u32,
    capacity: u32,
) -> (vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler, u32) {
    let mip_levels = 32 - size.leading_zeros();
    let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(mip_levels)
        .array_layers(capacity)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        )
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        device
            .create_image(&image_info, None)
            .expect("Failed to create block texture image")
    };
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(find_memory_type(
            &memory_props,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        ));
    let memory = unsafe {
        device
            .allocate_memory(&alloc_info, None)
            .expect("Failed to allocate block texture memory")
    };
    unsafe {
        device
            .bind_image_memory(image, memory, 0)
            .expect("Failed to bind block texture memory");
    }
    let view_range = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mip_levels,
        base_array_layer: 0,
        layer_count: capacity,
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(view_range);
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("Failed to create block texture view")
    };
    let sampler = create_sampler(device, anisotropy, mip_levels);
    (image, memory, view, sampler, mip_levels)
}

fn create_sampler(
    device: &ash::Device,
    anisotropy: Option<Anisotropy>,
    mip_levels: u32,
) -> vk::Sampler {
    let mut sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .min_lod(0.0)
        .max_lod(mip_levels as f32);
    if let Some(a) = anisotropy {
        sampler_info = sampler_info
            .anisotropy_enable(true)
            .max_anisotropy(a.clamp(8.0));
    }
    unsafe {
        device
            .create_sampler(&sampler_info, None)
            .expect("Failed to create block texture sampler")
    }
}

fn image_copy_mips(size: u32, mip_levels: u32, layer_count: u32) -> Vec<vk::ImageCopy> {
    (0..mip_levels)
        .map(|mip| {
            let extent = (size >> mip).max(1);
            vk::ImageCopy::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count,
                })
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: 0,
                    layer_count,
                })
                .extent(vk::Extent3D {
                    width: extent,
                    height: extent,
                    depth: 1,
                })
        })
        .collect()
}

struct PackedUpload {
    bytes: Vec<u8>,
    per_layer: Vec<(u32, Vec<vk::BufferImageCopy>)>,
}

fn pack_pending(size: u32, mip_levels: u32, pending: &[PendingLayer]) -> PackedUpload {
    let mut bytes = Vec::new();
    let mut per_layer = Vec::with_capacity(pending.len());
    for p in pending {
        let chain = build_mip_chain(&p.pixels, size, mip_levels);
        let mut regions = Vec::with_capacity(mip_levels as usize);
        for (mip, mip_pixels) in chain.iter().enumerate() {
            let offset = bytes.len() as u64;
            bytes.extend_from_slice(mip_pixels);
            let extent = (size >> mip).max(1);
            regions.push(
                vk::BufferImageCopy::default()
                    .buffer_offset(offset)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: mip as u32,
                        base_array_layer: p.index,
                        layer_count: 1,
                    })
                    .image_extent(vk::Extent3D {
                        width: extent,
                        height: extent,
                        depth: 1,
                    }),
            );
        }
        per_layer.push((p.index, regions));
    }
    PackedUpload { bytes, per_layer }
}

unsafe fn create_filled_staging(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    bytes: &[u8],
) -> (vk::Buffer, vk::DeviceMemory) {
    let size = bytes.len() as vk::DeviceSize;
    let staging_info = vk::BufferCreateInfo::default()
        .size(size.max(1))
        .usage(vk::BufferUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    unsafe {
        let staging = device
            .create_buffer(&staging_info, None)
            .expect("Failed to create block-texture staging buffer");
        let req = device.get_buffer_memory_requirements(staging);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(find_memory_type(
                memory_props,
                req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        let memory = device
            .allocate_memory(&alloc, None)
            .expect("Failed to allocate block-texture staging memory");
        device
            .bind_buffer_memory(staging, memory, 0)
            .expect("Failed to bind block-texture staging memory");
        if !bytes.is_empty() {
            let ptr = device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .expect("Failed to map block-texture staging memory");
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            device.unmap_memory(memory);
        }
        (staging, memory)
    }
}

fn srgb_to_linear(v: u8) -> f32 {
    let f = v as f32 / 255.0;
    if f <= 0.04045 {
        f / 12.92
    } else {
        ((f + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> u8 {
    let f = if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (f * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

fn build_mip_chain(base: &[u8], size: u32, levels: u32) -> Vec<Vec<u8>> {
    let mut mips = Vec::with_capacity(levels as usize);
    mips.push(base.to_vec());
    let mut w = size as usize;
    for _ in 1..levels {
        let prev = mips.last().unwrap();
        let nw = (w / 2).max(1);
        let mut next = vec![0u8; nw * nw * 4];
        for y in 0..nw {
            // Clamp handles odd dimensions (non-power-of-two sizes).
            let y0 = (y * 2).min(w - 1);
            let y1 = (y * 2 + 1).min(w - 1);
            for x in 0..nw {
                let x0 = (x * 2).min(w - 1);
                let x1 = (x * 2 + 1).min(w - 1);
                let idx = [y0 * w + x0, y0 * w + x1, y1 * w + x0, y1 * w + x1];
                for c in 0..3 {
                    let sum: f32 = idx.iter().map(|&i| srgb_to_linear(prev[i * 4 + c])).sum();
                    next[(y * nw + x) * 4 + c] = linear_to_srgb(sum / 4.0);
                }
                let alpha: u32 = idx.iter().map(|&i| prev[i * 4 + 3] as u32).sum();
                next[(y * nw + x) * 4 + 3] = ((alpha + 2) / 4) as u8;
            }
        }
        mips.push(next);
        w = nw;
    }
    mips
}

#[cfg(test)]
mod tests {
    use super::{
        MIN_LAYER_CAPACITY, build_mip_chain, changed_layer_indices, consecutive_runs, grow_plan,
        layer_capacity,
    };

    #[test]
    fn capacity_is_next_power_of_two_at_least_64_capped_by_device() {
        assert_eq!(layer_capacity(1, 2048), MIN_LAYER_CAPACITY);
        assert_eq!(layer_capacity(63, 2048), 64);
        assert_eq!(layer_capacity(64, 2048), 64);
        assert_eq!(layer_capacity(65, 2048), 128);
        assert_eq!(layer_capacity(100, 2048), 128);
        assert_eq!(layer_capacity(200, 2048), 256);
        assert_eq!(layer_capacity(1, 32), 32);
        assert_eq!(layer_capacity(100, 96), 96);
        assert_eq!(layer_capacity(2048, 2048), 2048);
        assert_eq!(layer_capacity(3000, 2048), 2048);
        assert_eq!(layer_capacity(0, 2048), 64);
    }

    #[test]
    fn layer_diff_detects_appends_and_in_place_changes() {
        let a = vec![vec![1, 2, 3, 4]];
        let b = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        assert_eq!(changed_layer_indices(&a, &b), vec![1]);
        let c = vec![vec![9, 9, 9, 9], vec![5, 6, 7, 8]];
        assert_eq!(changed_layer_indices(&a, &c), vec![0, 1]);
        assert!(changed_layer_indices(&b, &b).is_empty());
        let shorter = vec![vec![1, 2, 3, 4]];
        assert!(
            changed_layer_indices(&b, &shorter).is_empty(),
            "equal prefix on shrink is not a changed layer"
        );
        assert_eq!(
            changed_layer_indices(&b, &[vec![0, 0, 0, 0]]),
            vec![0],
            "changed prefix on shrink"
        );
    }

    #[test]
    fn grow_plan_copies_prefix_and_stages_tail_or_size_change() {
        let old = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let append = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8], vec![9, 9, 9, 9]];
        assert_eq!(grow_plan(16, 2, &old, 16, &append), (2, vec![2]));
        let changed = vec![vec![0, 0, 0, 0], vec![5, 6, 7, 8], vec![9, 9, 9, 9]];
        assert_eq!(grow_plan(16, 2, &old, 16, &changed), (2, vec![0, 2]));
        assert_eq!(
            grow_plan(1, 1, &[vec![255, 255, 255, 255]], 16, &append),
            (0, vec![0, 1, 2]),
            "texel-size change cannot GPU-copy"
        );
    }

    #[test]
    fn consecutive_runs_groups_sparse_indices() {
        assert!(consecutive_runs(vec![]).is_empty());
        assert_eq!(consecutive_runs(vec![3]), vec![(3, 1)]);
        assert_eq!(
            consecutive_runs(vec![2, 3, 4, 7, 8, 10]),
            vec![(2, 3), (7, 2), (10, 1)]
        );
        assert_eq!(consecutive_runs(vec![5, 5, 4]), vec![(4, 2)]);
    }

    #[test]
    fn mip_chain_halves_to_one() {
        let base = vec![255u8; 16 * 16 * 4];
        let chain = build_mip_chain(&base, 16, 5);
        assert_eq!(chain.len(), 5);
        let sizes: Vec<usize> = chain.iter().map(|m| m.len()).collect();
        assert_eq!(sizes, vec![16 * 16 * 4, 8 * 8 * 4, 4 * 4 * 4, 2 * 2 * 4, 4]);
        // White stays white through the box filter.
        assert!(chain.iter().all(|m| m.iter().all(|&b| b == 255)));
    }

    #[test]
    fn mip_chain_averages_2x2() {
        // RGB averages in LINEAR light; a constant channel round-trips to itself
        // (within rounding). Alpha is linear coverage and averages arithmetically.
        let mut base = vec![0u8; 2 * 2 * 4];
        for t in 0..4 {
            base[t * 4] = 100; // r constant across the 2x2
        }
        base[3] = 0; // alpha: 0, 100, 100, 200 -> arithmetic mean 100
        base[7] = 100;
        base[11] = 100;
        base[15] = 200;
        let chain = build_mip_chain(&base, 2, 2);
        assert_eq!(chain[1].len(), 4);
        assert!(
            (chain[1][0] as i32 - 100).abs() <= 1,
            "linear mean of a constant"
        );
        assert_eq!(chain[1][1], 0);
        assert_eq!(chain[1][3], 100, "alpha arithmetic mean");
    }

    #[test]
    fn mip_linear_mean_is_brighter_than_gamma_mean() {
        // 0 and 255 average to mid-gray in linear (~188 sRGB), well above the
        // naive gamma mean of ~128 — the whole point of linear downsampling.
        let mut base = vec![0u8; 2 * 2 * 4];
        for t in 0..4 {
            base[t * 4] = if t < 2 { 0 } else { 255 };
        }
        let chain = build_mip_chain(&base, 2, 2);
        assert!(chain[1][0] > 150, "got {}", chain[1][0]);
    }

    #[test]
    fn mip_chain_single_texel() {
        let base = vec![7u8, 8, 9, 10];
        let chain = build_mip_chain(&base, 1, 1);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0], base);
    }
}
