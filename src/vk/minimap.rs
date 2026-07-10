/// The minimap texture: a per-slot double-buffered RGBA image sampled by the
/// `tris2d_tex` pipeline. The app rebuilds the pixels on a throttle and calls
/// [`crate::Engine::update_minimap`]; each frame slot re-uploads only when its
/// copy is behind the latest `version`, so one refresh costs at most two copies.
use ash::vk;

use super::alloc::find_memory_type;
use super::buffers::{FRAMES_IN_FLIGHT, HostBuffer};
use super::image_upload::push_combined_image_sampler;
use crate::color::Color;

/// The one subresource every barrier/view here addresses: the single color mip.
const COLOR_SUBRESOURCE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};

/// One per-slot GPU image plus its current layout (tracked so `sync` emits the
/// right transition on first vs. subsequent uploads).
struct Img {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    layout: vk::ImageLayout,
}

pub(crate) struct MinimapTexture {
    /// Per-slot double buffer: a single image would be a write-after-read hazard
    /// across in-flight frames; per-slot images make every write safe under the
    /// existing timeline wait with no extra sync.
    images: [Img; FRAMES_IN_FLIGHT as usize],
    /// Per-slot staging buffer for pixel uploads.
    staging: [HostBuffer; FRAMES_IN_FLIGHT as usize],
    /// LINEAR filter with CLAMP_TO_EDGE addressing.
    sampler: vk::Sampler,
    size: u32,
    /// Cached RGBA pixels, reused across refreshes.
    pixels: Vec<u8>,
    /// Bumped by `update`; each slot syncs when its version is stale.
    version: u64,
    uploaded: [u64; FRAMES_IN_FLIGHT as usize],
}

impl MinimapTexture {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        _queue: vk::Queue,
        _pool: vk::CommandPool,
        size: u32,
        void: Color,
    ) -> Self {
        assert!(size >= 1, "minimap size must be >= 1");
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let byte_len = (size * size * 4) as u64;

        let make_img = || {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(vk::Extent3D {
                    width: size,
                    height: size,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let image = unsafe {
                device
                    .create_image(&image_info, None)
                    .expect("Failed to create minimap image")
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
                    .expect("Failed to allocate minimap image memory")
            };
            unsafe {
                device
                    .bind_image_memory(image, memory, 0)
                    .expect("Failed to bind minimap image memory");
            }
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .subresource_range(COLOR_SUBRESOURCE);
            let view = unsafe {
                device
                    .create_image_view(&view_info, None)
                    .expect("Failed to create minimap image view")
            };
            Img {
                image,
                memory,
                view,
                layout: vk::ImageLayout::UNDEFINED,
            }
        };
        let images = [make_img(), make_img()];

        // Staging buffers never change size — size them up-front so `sync`, which
        // has no `instance`/`physical`, only needs to `write` into them.
        let mut staging =
            std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::TRANSFER_SRC));
        for s in &mut staging {
            unsafe { s.maintain(instance, device, physical, byte_len) };
        }

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create minimap sampler")
        };

        // Visible checkerboard so the pipeline/descriptor/rotation can be
        // smoke-tested before the app feeds real pixels.
        let white = Color::rgb(255, 255, 255);
        let mut pixels = vec![0u8; byte_len as usize];
        for y in 0..size {
            for x in 0..size {
                let c = if ((x / 16) + (y / 16)) % 2 == 0 {
                    void
                } else {
                    white
                };
                let i = ((y * size + x) * 4) as usize;
                pixels[i..i + 4].copy_from_slice(&[c.r, c.g, c.b, c.a]);
            }
        }

        Self {
            images,
            staging,
            sampler,
            size,
            pixels,
            version: 1,
            uploaded: [0; FRAMES_IN_FLIGHT as usize],
        }
    }

    /// Updates pixels and bumps version.
    pub fn update(&mut self, rgba: &[u8]) {
        assert_eq!(rgba.len(), (self.size * self.size * 4) as usize);
        self.pixels.copy_from_slice(rgba);
        self.version += 1;
    }

    /// If this slot's uploaded version is behind `version`, records a staging →
    /// image copy (with the layout transitions the sampler needs) on the live
    /// frame command buffer. No-op when the slot is already current.
    pub unsafe fn sync(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if self.uploaded[slot] == self.version {
            return;
        }
        let img = &mut self.images[slot];
        unsafe {
            self.staging[slot].write(0, &self.pixels);

            // The first upload transitions out of UNDEFINED (nothing to wait on);
            // later uploads overwrite an image the fragment shader last sampled.
            let (src_stage, src_access) = match img.layout {
                vk::ImageLayout::UNDEFINED => {
                    (vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE)
                }
                _ => (
                    vk::PipelineStageFlags2::FRAGMENT_SHADER,
                    vk::AccessFlags2::SHADER_SAMPLED_READ,
                ),
            };
            let to_transfer = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(src_stage)
                .src_access_mask(src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .old_layout(img.layout)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(img.image)
                .subresource_range(COLOR_SUBRESOURCE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
            );

            let region = [vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(vk::Extent3D {
                    width: self.size,
                    height: self.size,
                    depth: 1,
                })];
            device.cmd_copy_buffer_to_image(
                cmd,
                self.staging[slot].buffer,
                img.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );

            let to_sampled = [vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(img.image)
                .subresource_range(COLOR_SUBRESOURCE)];
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_sampled),
            );
        }
        img.layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        self.uploaded[slot] = self.version;
    }

    /// Pushes this slot's minimap descriptor for the pipeline.
    pub fn push_descriptor(
        &self,
        push: &ash::khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        slot: usize,
    ) {
        push_combined_image_sampler(push, cmd, layout, 0, self.sampler, self.images[slot].view);
    }

    /// Whether a texture has ever been uploaded (nothing to draw before the
    /// first app refresh).
    pub fn ready(&self) -> bool {
        self.version > 0
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            for img in &mut self.images {
                device.destroy_image_view(img.view, None);
                device.destroy_image(img.image, None);
                device.free_memory(img.memory, None);
            }
            for s in &mut self.staging {
                s.destroy(device);
            }
        }
    }
}
