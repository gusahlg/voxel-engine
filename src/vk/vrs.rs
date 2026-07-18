//! Variable-rate shading: per-slot rate image and classifier pipeline.

use ash::vk;

use super::buffers::FRAMES_IN_FLIGHT;
use super::device::FragmentShadingRate;
use super::image::{ImageDesc, ImageResource};

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VrsPush {
    pub d_threshold: f32,
    pub texel_w: u32,
    pub texel_h: u32,
}

pub(crate) struct RateAttachment {
    pub view: vk::ImageView,
    pub texel_size: vk::Extent2D,
}

pub(crate) struct Vrs {
    /// Texel size for VRS attachment; shared by all pipelines.
    pub texel_size: vk::Extent2D,
    tiles: vk::Extent2D,
    images: [ImageResource; SLOTS],
}

impl Vrs {
    pub fn new(
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        fsr: &FragmentShadingRate,
        render_extent: vk::Extent2D,
    ) -> Vrs {
        let texel_size = fsr.texel_size;
        let tiles = vk::Extent2D {
            width: render_extent.width.div_ceil(texel_size.width).max(1),
            height: render_extent.height.div_ceil(texel_size.height).max(1),
        };
        let images = std::array::from_fn(|_| create_rate_image(device, memory_props, tiles));
        Vrs {
            texel_size,
            tiles,
            images,
        }
    }

    pub fn tiles(&self) -> vk::Extent2D {
        self.tiles
    }

    pub fn view(&self, slot: usize) -> vk::ImageView {
        self.images[slot].view()
    }

    pub fn image(&self, slot: usize) -> vk::Image {
        self.images[slot].image()
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for img in &self.images {
                img.destroy(device);
            }
        }
    }
}

fn create_rate_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    tiles: vk::Extent2D,
) -> ImageResource {
    let desc = ImageDesc {
        extent: tiles,
        format: vk::Format::R8_UINT,
        usage: vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
        mips: 1,
        layers: 1,
        aspect: vk::ImageAspectFlags::COLOR,
        samples: vk::SampleCountFlags::TYPE_1,
    };
    ImageResource::create(device, memory_props, &desc)
}

