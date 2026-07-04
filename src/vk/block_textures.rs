/// The block texture array: one RGBA8 `TEXTURE_2D_ARRAY` sampled by the 3D
/// pipelines (vertex `color.a` selects the layer). Mip chains are built
/// CPU-side with a box filter and uploaded in one blocking submit, mirroring
/// the font atlas' one-shot pattern. Layer 0 is white by contract; the
/// default array created at init is a single white 1x1 layer so everything
/// renders before the first `set_block_textures` call.
///
/// The descriptor set layout/pool/set live in the `Renderer` (created once,
/// never rebuilt); swapping textures only rewrites the set via
/// `write_descriptor`, so pipelines never need rebuilding.
use ash::vk;

use super::targets::find_memory_type;

pub struct BlockTextures {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub layers: u32,
    pub size: u32,
}

impl BlockTextures {
    /// 1x1, one all-white layer — the init-time placeholder.
    pub fn new_default(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
    ) -> Self {
        Self::upload(
            instance,
            device,
            physical,
            graphics_queue,
            command_pool,
            1,
            &[vec![255, 255, 255, 255]],
        )
    }

    /// Uploads `layers` RGBA8 images of `size`x`size` as a device-local
    /// texture array with a full CPU-built mip chain per layer. Blocks until
    /// the copy completes.
    pub fn upload(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        command_pool: vk::CommandPool,
        size: u32,
        layers: &[Vec<u8>],
    ) -> Self {
        assert!(size >= 1, "block texture size must be >= 1");
        assert!(!layers.is_empty(), "block texture array needs >= 1 layer");
        let layer_bytes = (size * size * 4) as usize;
        for (i, layer) in layers.iter().enumerate() {
            assert_eq!(
                layer.len(),
                layer_bytes,
                "layer {i}: expected {size}x{size} RGBA8 = {layer_bytes} bytes"
            );
        }
        let layer_count = layers.len() as u32;
        let mip_levels = 32 - size.leading_zeros(); // floor(log2) + 1, down to 1x1

        // CPU mip chains, then packed mip-major so each mip level is one
        // buffer->image copy covering all layers.
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
        let total = staging_data.len() as vk::DeviceSize;

        // Image
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            })
            .mip_levels(mip_levels)
            .array_layers(layer_count)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
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
                instance,
                physical,
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

        // Staging buffer (temporary, destroyed after the blocking upload).
        let staging_info = vk::BufferCreateInfo::default()
            .size(total)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging = unsafe {
            device
                .create_buffer(&staging_info, None)
                .expect("Failed to create staging buffer")
        };
        let staging_req = unsafe { device.get_buffer_memory_requirements(staging) };
        let staging_alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(staging_req.size)
            .memory_type_index(find_memory_type(
                instance,
                physical,
                staging_req.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        let staging_memory = unsafe {
            device
                .allocate_memory(&staging_alloc, None)
                .expect("Failed to allocate staging memory")
        };
        unsafe {
            device
                .bind_buffer_memory(staging, staging_memory, 0)
                .expect("Failed to bind staging memory");
            let ptr = device
                .map_memory(staging_memory, 0, total, vk::MemoryMapFlags::empty())
                .expect("Failed to map staging memory");
            std::ptr::copy_nonoverlapping(
                staging_data.as_ptr(),
                ptr as *mut u8,
                staging_data.len(),
            );
            device.unmap_memory(staging_memory);
        }

        // One-time upload: UNDEFINED -> TRANSFER_DST -> copies -> SHADER_READ.
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe {
            device
                .allocate_command_buffers(&alloc)
                .expect("Failed to allocate upload command buffer")[0]
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let subresource = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: mip_levels,
            base_array_layer: 0,
            layer_count,
        };
        unsafe {
            device
                .begin_command_buffer(cmd, &begin)
                .expect("Failed to begin upload command buffer");

            let to_transfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image)
                .subresource_range(subresource)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
            );

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
            device.cmd_copy_buffer_to_image(
                cmd,
                staging,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );

            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(image)
                .subresource_range(subresource)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );

            device
                .end_command_buffer(cmd)
                .expect("Failed to end upload command buffer");

            let buffers = [cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&buffers);
            device
                .queue_submit(graphics_queue, &[submit], vk::Fence::null())
                .expect("Failed to submit block texture upload");
            device
                .queue_wait_idle(graphics_queue)
                .expect("Block texture upload wait failed");

            device.free_command_buffers(command_pool, &buffers);
            device.destroy_buffer(staging, None);
            device.free_memory(staging_memory, None);
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(subresource);
        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("Failed to create block texture view")
        };

        // NEAREST texels (crisp voxel look), LINEAR between mips, REPEAT so
        // greedy-meshed quads tile per block.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .min_lod(0.0)
            .max_lod(mip_levels as f32);
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
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Creates the persistent descriptor machinery for the block texture array:
/// binding 0 = combined image sampler, fragment stage. Lives as long as the
/// renderer; texture swaps only rewrite the set.
pub fn create_descriptor(
    device: &ash::Device,
) -> (
    vk::DescriptorSetLayout,
    vk::DescriptorPool,
    vk::DescriptorSet,
) {
    let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe {
        device
            .create_descriptor_set_layout(&layout_info, None)
            .expect("Failed to create block texture set layout")
    };

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let pool = unsafe {
        device
            .create_descriptor_pool(&pool_info, None)
            .expect("Failed to create block texture descriptor pool")
    };

    let layouts = [set_layout];
    let set_alloc = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let set = unsafe {
        device
            .allocate_descriptor_sets(&set_alloc)
            .expect("Failed to allocate block texture descriptor set")[0]
    };

    (set_layout, pool, set)
}

/// Points the persistent descriptor set at `textures`. Only call when the
/// GPU cannot be using the set (init, or after `device_wait_idle`).
pub fn write_descriptor(device: &ash::Device, set: vk::DescriptorSet, textures: &BlockTextures) {
    let image_infos = [vk::DescriptorImageInfo::default()
        .sampler(textures.sampler)
        .image_view(textures.view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
    let writes = [vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos)];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Full mip chain for one RGBA8 layer: level 0 is `base`, each further level
/// is a 2x2 box filter (rounded average) of the previous, down to 1x1.
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
                for c in 0..4 {
                    let sum = prev[(y0 * w + x0) * 4 + c] as u32
                        + prev[(y0 * w + x1) * 4 + c] as u32
                        + prev[(y1 * w + x0) * 4 + c] as u32
                        + prev[(y1 * w + x1) * 4 + c] as u32;
                    next[(y * nw + x) * 4 + c] = ((sum + 2) / 4) as u8;
                }
            }
        }
        mips.push(next);
        w = nw;
    }
    mips
}

#[cfg(test)]
mod tests {
    use super::build_mip_chain;

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
        // 2x2 texels: r = 0, 100, 100, 200 -> avg 100 (all other channels 0).
        let mut base = vec![0u8; 2 * 2 * 4];
        base[0] = 0;
        base[4] = 100;
        base[8] = 100;
        base[12] = 200;
        let chain = build_mip_chain(&base, 2, 2);
        assert_eq!(chain[1].len(), 4);
        assert_eq!(chain[1][0], 100);
        assert_eq!(chain[1][1], 0);
    }

    #[test]
    fn mip_chain_single_texel() {
        let base = vec![7u8, 8, 9, 10];
        let chain = build_mip_chain(&base, 1, 1);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0], base);
    }
}
