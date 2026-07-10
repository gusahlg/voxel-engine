/// RGBA8 texture array, mip-chained CPU-side and uploaded in one blocking
/// submit. Layer 0 is always white. Descriptor set lives in the Renderer;
/// texture swaps only rewrite the set, no pipeline rebuild needed.
use ash::vk;

use super::device::Anisotropy;
use super::image_upload::{ImageUpload, upload_image};

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
        anisotropy: Option<Anisotropy>,
    ) -> Self {
        Self::upload(
            instance,
            device,
            physical,
            graphics_queue,
            command_pool,
            anisotropy,
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
        anisotropy: Option<Anisotropy>,
        size: u32,
        layers: &[Vec<u8>],
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
        let mip_levels = 32 - size.leading_zeros(); // log2 size, clamped to 1x1

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
        // One buffer->image copy region per mip level, each covering all
        // array layers (the staging blob is packed mip-major above).
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
            command_pool,
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
                array_layers: layer_count,
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
            sampler_info = sampler_info.anisotropy_enable(true).max_anisotropy(a.clamp(8.0));
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
        assert!((chain[1][0] as i32 - 100).abs() <= 1, "linear mean of a constant");
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
