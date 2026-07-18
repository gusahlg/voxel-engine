/// The font atlas: a single R8 texture built from the embedded 8x8 font,
/// uploaded once at startup, sampled NEAREST by the 2D pipeline. Owns the
/// push-descriptor set layout; the image is pushed at record time.
use ash::{khr, vk};

use super::image_upload::{
    ImageUpload, create_sampler_set_layout, push_combined_image_sampler, upload_image,
};
use super::transfer::TransferLane;
use crate::font;

pub struct FontAtlas {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub set_layout: vk::DescriptorSetLayout,
}

impl FontAtlas {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        graphics_queue: vk::Queue,
        graphics_family: u32,
        command_pool: vk::CommandPool,
        lane: &mut TransferLane,
    ) -> Self {
        let pixels = font::build_atlas();
        let extent = vk::Extent2D {
            width: font::ATLAS_WIDTH,
            height: font::ATLAS_HEIGHT,
        };

        // Single R8 blob, one mip, one layer: one copy region covering it all.
        let regions = [vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })];
        let (image, memory, view) = upload_image(
            instance,
            device,
            physical,
            graphics_queue,
            graphics_family,
            command_pool,
            lane,
            &ImageUpload {
                extent,
                format: vk::Format::R8_UNORM,
                mip_levels: 1,
                array_layers: 1,
                view_type: vk::ImageViewType::TYPE_2D,
                bytes: &pixels,
                regions: &regions,
            },
        );

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create font atlas sampler")
        };

        // Push-descriptor layout: binding 0 = combined image sampler, fragment.
        let set_layout = create_sampler_set_layout(device);

        Self {
            image,
            memory,
            view,
            sampler,
            set_layout,
        }
    }

    /// Pushes the atlas image into `set_index` of the bound 2D layout.
    pub fn push_descriptor(
        &self,
        push: &khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        layout: vk::PipelineLayout,
        set_index: u32,
    ) {
        push_combined_image_sampler(push, cmd, layout, set_index, self.sampler, self.view);
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
