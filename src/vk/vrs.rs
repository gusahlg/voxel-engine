//! Variable-rate shading: per-slot rate image, history, and classifier.

use ash::vk;

use super::alloc::{find_memory_type, try_find_memory_type};
use super::buffers::FRAMES_IN_FLIGHT;
use super::device::FragmentShadingRate;
use super::image::{ImageDesc, ImageResource};

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

pub(crate) const FLAG_ALLOW_4X4: u32 = 1 << 0;
pub(crate) const FLAG_USE_HISTORY: u32 = 1 << 1;

const MIX_COUNT: usize = 3;
pub(crate) const MIX_BYTES: u64 = (MIX_COUNT * size_of::<u32>()) as u64;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct VrsPush {
    pub d_threshold: f32,
    pub texel_w: u32,
    pub texel_h: u32,
    pub tiles_x: u32,
    pub tiles_y: u32,
    pub depth_w: u32,
    pub depth_h: u32,
    pub flags: u32,
}

pub(crate) struct RateAttachment {
    pub view: vk::ImageView,
    pub texel_size: vk::Extent2D,
}

/// Host-visible copy of the per-slot tile-mix histogram, fence-safe to read
/// after the slot's timeline wait.
struct MixReadback {
    gpu: vk::Buffer,
    gpu_memory: vk::DeviceMemory,
    cpu: vk::Buffer,
    cpu_memory: vk::DeviceMemory,
    mapped: *mut u32,
}

pub(crate) struct Vrs {
    /// Texel size for VRS attachment; shared by all pipelines.
    pub texel_size: vk::Extent2D,
    tiles: vk::Extent2D,
    images: [ImageResource; SLOTS],
    history: [ImageResource; SLOTS],
    mix: [MixReadback; SLOTS],
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
        let images = std::array::from_fn(|_| {
            create_r8_image(
                device,
                memory_props,
                tiles,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
            )
        });
        let history = std::array::from_fn(|_| {
            create_r8_image(device, memory_props, tiles, vk::ImageUsageFlags::STORAGE)
        });
        let mix = std::array::from_fn(|_| MixReadback::new(device, memory_props));
        Vrs {
            texel_size,
            tiles,
            images,
            history,
            mix,
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

    pub fn history_view(&self, slot: usize) -> vk::ImageView {
        self.history[slot].view()
    }

    pub fn history_image(&self, slot: usize) -> vk::Image {
        self.history[slot].image()
    }

    pub fn mix_gpu(&self, slot: usize) -> vk::Buffer {
        self.mix[slot].gpu
    }

    pub fn mix_cpu(&self, slot: usize) -> vk::Buffer {
        self.mix[slot].cpu
    }

    /// Last completed histogram for `slot`: `[1x1, 2x2, 4x4]`.
    pub fn mix(&self, slot: usize) -> [u32; MIX_COUNT] {
        unsafe {
            let p = self.mix[slot].mapped;
            [*p, *p.add(1), *p.add(2)]
        }
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for img in &self.images {
                img.destroy(device);
            }
            for img in &self.history {
                img.destroy(device);
            }
            for mix in &self.mix {
                mix.destroy(device);
            }
        }
    }
}

impl MixReadback {
    fn new(device: &ash::Device, memory_props: &vk::PhysicalDeviceMemoryProperties) -> Self {
        let gpu = create_buffer(
            device,
            memory_props,
            MIX_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let (cpu, cpu_memory, mapped) = create_mapped_buffer(
            device,
            memory_props,
            MIX_BYTES,
            vk::BufferUsageFlags::TRANSFER_DST,
        );
        Self {
            gpu: gpu.0,
            gpu_memory: gpu.1,
            cpu,
            cpu_memory,
            mapped,
        }
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.cpu_memory);
            device.destroy_buffer(self.cpu, None);
            device.free_memory(self.cpu_memory, None);
            device.destroy_buffer(self.gpu, None);
            device.free_memory(self.gpu_memory, None);
        }
    }
}

fn create_r8_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    tiles: vk::Extent2D,
    usage: vk::ImageUsageFlags,
) -> ImageResource {
    ImageResource::create(
        device,
        memory_props,
        &ImageDesc {
            extent: tiles,
            format: vk::Format::R8_UINT,
            usage,
            layers: 1,
            aspect: vk::ImageAspectFlags::COLOR,
            samples: vk::SampleCountFlags::TYPE_1,
        },
    )
}

fn create_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> (vk::Buffer, vk::DeviceMemory) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create VRS mix buffer")
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
                        props,
                    )),
                None,
            )
            .expect("allocate VRS mix buffer")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind VRS mix buffer");
    }
    (buffer, memory)
}

fn create_mapped_buffer(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u64,
    usage: vk::BufferUsageFlags,
) -> (vk::Buffer, vk::DeviceMemory, *mut u32) {
    let buffer = unsafe {
        device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .expect("create VRS mix readback")
    };
    let reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let cached = vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED;
    let plain = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let type_index = try_find_memory_type(memory_props, reqs.memory_type_bits, cached)
        .unwrap_or_else(|| find_memory_type(memory_props, reqs.memory_type_bits, plain));
    let memory = unsafe {
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(type_index),
                None,
            )
            .expect("allocate VRS mix readback")
    };
    unsafe {
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind VRS mix readback");
    }
    let mapped = unsafe {
        device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            .expect("map VRS mix readback") as *mut u32
    };
    unsafe { std::ptr::write_bytes(mapped, 0, MIX_COUNT) };
    (buffer, memory, mapped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SKY_DEPTH: f32 = 1.0e-6;
    const RATE_1X1: u32 = 0;
    const RATE_2X2: u32 = (1 << 2) | 1;
    const RATE_4X4: u32 = (2 << 2) | 2;

    fn classify_tile(dmin: f32, dmax: f32, d_threshold: f32, allow_4x4: bool) -> u32 {
        if dmax < SKY_DEPTH {
            return if allow_4x4 { RATE_4X4 } else { RATE_2X2 };
        }
        let far = dmax < d_threshold;
        let flat = (dmax - dmin) < (d_threshold * 0.5);
        if far && flat { RATE_2X2 } else { RATE_1X1 }
    }

    fn conservative_rate(raw: u32, prev_raw: u32, neighbor_full: bool) -> u32 {
        if raw == RATE_1X1 || prev_raw == RATE_1X1 || neighbor_full {
            RATE_1X1
        } else {
            raw
        }
    }

    #[test]
    fn sky_is_coarse_and_uses_4x4_when_advertised() {
        assert_eq!(classify_tile(0.0, 0.0, 0.01, false), RATE_2X2);
        assert_eq!(classify_tile(0.0, 0.0, 0.01, true), RATE_4X4);
        assert_eq!(classify_tile(0.0, SKY_DEPTH * 0.5, 0.01, true), RATE_4X4);
    }

    #[test]
    fn far_flat_terrain_stays_2x2_even_when_4x4_exists() {
        assert_eq!(classify_tile(0.001, 0.002, 0.01, true), RATE_2X2);
        assert_eq!(classify_tile(0.001, 0.002, 0.01, false), RATE_2X2);
    }

    #[test]
    fn near_or_discontinuous_tiles_are_full_rate() {
        assert_eq!(classify_tile(0.5, 0.6, 0.01, true), RATE_1X1);
        // Far but a silhouette crosses the tile (range > half the threshold).
        assert_eq!(classify_tile(0.0, 0.009, 0.01, true), RATE_1X1);
    }

    #[test]
    fn conservative_holds_last_near_and_dilates_full_rate() {
        assert_eq!(conservative_rate(RATE_4X4, RATE_1X1, false), RATE_1X1);
        assert_eq!(conservative_rate(RATE_2X2, RATE_2X2, true), RATE_1X1);
        assert_eq!(conservative_rate(RATE_4X4, RATE_2X2, false), RATE_4X4);
        assert_eq!(conservative_rate(RATE_2X2, RATE_4X4, false), RATE_2X2);
        // Finer is always allowed.
        assert_eq!(conservative_rate(RATE_1X1, RATE_4X4, false), RATE_1X1);
    }

    #[test]
    fn rate_encoding_matches_vk_fragment_size_pack() {
        assert_eq!(RATE_1X1, 0);
        assert_eq!(RATE_2X2, 0b0101);
        assert_eq!(RATE_4X4, 0b1010);
    }

    #[test]
    fn push_layout_is_tight_u32s() {
        assert_eq!(size_of::<VrsPush>(), 32);
    }
}
