/// Per-slot minimap texture with versioned upload.
use ash::vk;

use super::buffers::{FRAMES_IN_FLIGHT, HostBuffer};
use super::image::{ImageDesc, ImageResource, LayoutUse};
use super::image_upload::push_combined_image_sampler;
use crate::color::Color;

pub(crate) struct MinimapTexture {
    /// Per-slot texture images.
    images: [ImageResource; FRAMES_IN_FLIGHT as usize],
    staging: [HostBuffer; FRAMES_IN_FLIGHT as usize],
    sampler: vk::Sampler,
    size: u32,
    pixels: Vec<u8>,
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
            ImageResource::create(
                device,
                &memory_props,
                &ImageDesc {
                    extent: vk::Extent2D {
                        width: size,
                        height: size,
                    },
                    format: vk::Format::R8G8B8A8_UNORM,
                    usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                    mips: 1,
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
            )
        };
        let images = [make_img(), make_img()];

        // Pre-allocate staging buffers.
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

    pub fn update(&mut self, rgba: &[u8]) {
        assert_eq!(rgba.len(), (self.size * self.size * 4) as usize);
        self.pixels.copy_from_slice(rgba);
        self.version += 1;
    }

    pub unsafe fn sync(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if self.uploaded[slot] == self.version {
            return;
        }
        let img = &mut self.images[slot];
        unsafe {
            self.staging[slot].write(0, &self.pixels);

            img.transition(device, cmd, LayoutUse::TransferDst);

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
                self.staging[slot]
                    .bound()
                    .expect("minimap staging is written before this upload copy"),
                img.image(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &region,
            );

            img.transition(device, cmd, LayoutUse::FragmentSampledAfterTransfer);
        }
        self.uploaded[slot] = self.version;
    }

    pub fn push_descriptor(
        &self,
        push: &ash::khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        slot: usize,
    ) {
        push_combined_image_sampler(push, cmd, layout, 0, self.sampler, self.images[slot].view());
    }

    pub fn ready(&self) -> bool {
        self.version > 0
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            for img in &self.images {
                img.destroy(device);
            }
            for s in &mut self.staging {
                s.destroy(device);
            }
        }
    }
}
