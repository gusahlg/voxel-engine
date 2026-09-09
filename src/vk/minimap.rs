/// Per-slot minimap texture with versioned upload.
use ash::vk;

use super::buffers::{FRAMES_IN_FLIGHT, HostBuffer};
use super::image::{ImageDesc, ImageResource, LayoutUse};
use super::image_upload::push_combined_image_sampler;
use super::pass;
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
    /// Texel subrect to upload for the current `version` (`w==size && h==size` is full).
    dirty: (u32, u32, u32, u32),
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
                    layers: 1,
                    aspect: vk::ImageAspectFlags::COLOR,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
                "minimap",
            )
            .expect("minimap image")
        };
        let images = std::array::from_fn(|_| make_img());

        // Pre-allocate staging buffers.
        let mut staging =
            std::array::from_fn(|_| HostBuffer::new(vk::BufferUsageFlags::TRANSFER_SRC));
        for s in &mut staging {
            unsafe { s.maintain(instance, device, physical, byte_len) };
        }

        let sampler = pass::linear_clamp_sampler(device, "minimap");

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
            dirty: (0, 0, size, size),
        }
    }

    pub fn update(&mut self, rgba: &[u8]) {
        assert_eq!(rgba.len(), (self.size * self.size * 4) as usize);
        self.pixels.copy_from_slice(rgba);
        self.version += 1;
        self.dirty = (0, 0, self.size, self.size);
    }

    pub fn update_rect(&mut self, x: u32, y: u32, w: u32, h: u32, rgba: &[u8]) {
        assert!(
            x.saturating_add(w) <= self.size && y.saturating_add(h) <= self.size,
            "minimap rect out of bounds"
        );
        assert_eq!(rgba.len(), (w * h * 4) as usize);
        for row in 0..h {
            let dst = (((y + row) * self.size + x) * 4) as usize;
            let src = (row * w * 4) as usize;
            let n = (w * 4) as usize;
            self.pixels[dst..dst + n].copy_from_slice(&rgba[src..src + n]);
        }
        self.version += 1;
        self.dirty = union_rect(self.dirty, (x, y, w, h));
    }

    /// Returns whether this recorded a buffer-to-image upload.
    pub unsafe fn sync(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) -> bool {
        if self.uploaded[slot] == self.version {
            return false;
        }
        let img = &mut self.images[slot];
        let (x, y, w, h) = self.dirty;
        unsafe {
            // Tightly packed dirty rect into staging; bufferRowLength is the
            // rect width so the GPU reads packed rows (transfer copy, not a
            // full 256 KiB rewrite).
            let packed_len = (w * h * 4) as usize;
            if w == self.size && h == self.size && x == 0 && y == 0 {
                self.staging[slot].write(0, &self.pixels);
            } else {
                let mut packed = vec![0u8; packed_len];
                for row in 0..h {
                    let src = (((y + row) * self.size + x) * 4) as usize;
                    let dst = (row * w * 4) as usize;
                    let n = (w * 4) as usize;
                    packed[dst..dst + n].copy_from_slice(&self.pixels[src..src + n]);
                }
                self.staging[slot].write(0, &packed);
            }

            img.transition(device, cmd, LayoutUse::TransferDst);

            let region = [vk::BufferImageCopy::default()
                .buffer_row_length(w)
                .buffer_image_height(h)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_offset(vk::Offset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: w,
                    height: h,
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
        true
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

fn union_rect(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
    let (ax, ay, aw, ah) = a;
    let (bx, by, bw, bh) = b;
    if aw == 0 || ah == 0 {
        return b;
    }
    if bw == 0 || bh == 0 {
        return a;
    }
    let x = ax.min(bx);
    let y = ay.min(by);
    let x2 = (ax + aw).max(bx + bw);
    let y2 = (ay + ah).max(by + bh);
    (x, y, x2 - x, y2 - y)
}
