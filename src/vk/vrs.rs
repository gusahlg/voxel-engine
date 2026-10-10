//! Variable-rate shading: per-slot rate image, history, and classifier.

use ash::vk;

use super::alloc::GpuCpuReadback;
use super::buffers::FRAMES_IN_FLIGHT;
use super::device::FragmentShadingRate;
use super::image::{AllocError, ImageDesc, ImageResource, create_image_array, image_purpose};
use super::{SAMPLEABLE_DEPTH_REST_LAYOUT, color_range};
use crate::skeleton::FrameSlot;

const SLOTS: usize = FRAMES_IN_FLIGHT as usize;

pub(crate) const FLAG_ALLOW_4X4: u32 = 1 << 0;
pub(crate) const FLAG_USE_HISTORY: u32 = 1 << 1;
pub(crate) const FLAG_WRITE_MIX: u32 = 1 << 2;

/// `VrsPush::flags` for one classify dispatch: the bits `vrs.comp.slang`
/// tests in `pc.flags`.
fn classify_flags(allow_4x4: bool, use_history: bool, write_mix: bool) -> u32 {
    let mut flags = 0u32;
    if allow_4x4 {
        flags |= FLAG_ALLOW_4X4;
    }
    if use_history {
        flags |= FLAG_USE_HISTORY;
    }
    if write_mix {
        flags |= FLAG_WRITE_MIX;
    }
    flags
}

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
struct MixReadback(GpuCpuReadback);

impl std::ops::Deref for MixReadback {
    type Target = GpuCpuReadback;
    fn deref(&self) -> &GpuCpuReadback {
        &self.0
    }
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
    ) -> Result<Vrs, AllocError> {
        let texel_size = fsr.texel_size;
        let tiles = vk::Extent2D {
            width: render_extent.width.div_ceil(texel_size.width).max(1),
            height: render_extent.height.div_ceil(texel_size.height).max(1),
        };
        let rate_purpose = image_purpose("VRS rate", tiles, vk::SampleCountFlags::TYPE_1);
        let history_purpose = image_purpose("VRS history", tiles, vk::SampleCountFlags::TYPE_1);
        let images = create_image_array(device, || {
            create_r8_image(
                device,
                memory_props,
                tiles,
                vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR,
                &rate_purpose,
            )
        })?;
        let history = match create_image_array(device, || {
            create_r8_image(
                device,
                memory_props,
                tiles,
                vk::ImageUsageFlags::STORAGE,
                &history_purpose,
            )
        }) {
            Ok(history) => history,
            Err(err) => {
                for img in &images {
                    unsafe { img.destroy(device) };
                }
                return Err(err);
            }
        };
        let mix = std::array::from_fn(|_| MixReadback::new(device, memory_props));
        Ok(Vrs {
            texel_size,
            tiles,
            images,
            history,
            mix,
        })
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
        Self(GpuCpuReadback::new(
            device,
            memory_props,
            MIX_COUNT,
            "VRS mix buffer",
            "VRS mix readback",
        ))
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { self.0.destroy(device) };
    }
}

fn create_r8_image(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    tiles: vk::Extent2D,
    usage: vk::ImageUsageFlags,
    purpose: &str,
) -> Result<ImageResource, AllocError> {
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
        purpose,
    )
}

impl super::Renderer {
    /// Rate-attachment view/texel size for the scene-pass begin. The image is
    /// still in GENERAL from the previous classify of this slot; the caller
    /// joins [`Self::vrs_rate_to_attachment_barrier`] into the begin batch.
    pub(super) fn vrs_rate_attachment(&self, slot: usize) -> RateAttachment {
        let vrs = self.targets.vrs.as_ref().expect("use_vrs implies vrs");
        RateAttachment {
            view: vrs.view(slot),
            texel_size: vrs.texel_size,
        }
    }

    /// GENERAL → `FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL`.
    ///
    /// Src: last classify of this slot (`COMPUTE_SHADER` / `SHADER_STORAGE_WRITE`).
    /// The slot timeline wait makes that submit complete; this barrier is the
    /// layout transition plus queue-side availability for the shading-rate
    /// attachment read. Dst: `FRAGMENT_SHADING_RATE_ATTACHMENT_KHR` /
    /// `FRAGMENT_SHADING_RATE_ATTACHMENT_READ`. Joins the scene-pass begin batch.
    pub(super) fn vrs_rate_to_attachment_barrier(
        &self,
        slot: usize,
    ) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("use_vrs implies vrs");
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR)
            .dst_access_mask(vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
            .image(vrs.image(slot))
            .subresource_range(color_range())
    }

    /// Rate image → GENERAL for the end-of-frame classify.
    ///
    /// If this slot bound the rate image as a shading-rate attachment (`vrs_ready`),
    /// src is the FSR read (`FRAGMENT_SHADING_RATE_ATTACHMENT_KHR` /
    /// `FRAGMENT_SHADING_RATE_ATTACHMENT_READ`) and old layout is FSR_OPTIMAL.
    /// First classify after create/recreate: image is still UNDEFINED, src NONE.
    /// Dst: `COMPUTE_SHADER` / `SHADER_STORAGE_WRITE`. Contents are rewritten, so
    /// UNDEFINED would also be a valid old layout on the FSR path; we keep FSR
    /// so the execution dependency on the attachment read is explicit.
    pub(super) fn vrs_rate_to_general_barrier(&self, slot: usize) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let used = self.slots[FrameSlot::new(slot)].vrs_ready;
        if used {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR)
                .src_access_mask(vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(vrs.image(slot))
                .subresource_range(color_range())
        } else {
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(vrs.image(slot))
                .subresource_range(color_range())
        }
    }

    /// History image → GENERAL for the end-of-frame classify.
    ///
    /// With history: src is the previous classify write (`COMPUTE_SHADER` /
    /// `SHADER_STORAGE_WRITE`), old GENERAL. First classify: UNDEFINED, src NONE.
    /// Dst: `COMPUTE_SHADER` / `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE`.
    pub(super) fn vrs_history_to_general_barrier(
        &self,
        slot: usize,
    ) -> vk::ImageMemoryBarrier2<'_> {
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let use_history = self.slots[FrameSlot::new(slot)].vrs_history;
        vk::ImageMemoryBarrier2::default()
            .src_stage_mask(if use_history {
                vk::PipelineStageFlags2::COMPUTE_SHADER
            } else {
                vk::PipelineStageFlags2::NONE
            })
            .src_access_mask(if use_history {
                vk::AccessFlags2::SHADER_STORAGE_WRITE
            } else {
                vk::AccessFlags2::NONE
            })
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
            .old_layout(if use_history {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::UNDEFINED
            })
            .new_layout(vk::ImageLayout::GENERAL)
            .image(vrs.history_image(slot))
            .subresource_range(color_range())
    }

    /// Zeros the mix histogram. Returns whether a CLEAR→COMPUTE memory barrier
    /// must join the upcoming pipeline barrier (only when profiling).
    pub(super) unsafe fn record_vrs_mix_fill(&self, cmd: vk::CommandBuffer, slot: usize) -> bool {
        if !crate::profile::is_enabled() {
            return false;
        }
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        unsafe {
            self.device
                .device
                .cmd_fill_buffer(cmd, vrs.mix_gpu(slot), 0, MIX_BYTES, 0);
        }
        true
    }

    /// CLEAR / TRANSFER_WRITE → COMPUTE / SHADER_STORAGE_{READ,WRITE} for the
    /// mix fill. Only issued when [`Self::record_vrs_mix_fill`] returned true.
    pub(super) fn vrs_mix_fill_memory_barrier() -> vk::MemoryBarrier2<'static> {
        vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::CLEAR)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            )
    }

    /// Dispatch the classifier. Depth already rests in
    /// [`SAMPLEABLE_DEPTH_REST_LAYOUT`]; rate + history are already GENERAL
    /// (joined into the post-scene barrier). The rate image stays GENERAL
    /// until the next scene-pass begin of this slot. Mix fill/copy/host
    /// barriers run only while profiling.
    ///
    /// `depth_ms` ([`super::SceneDepthUse::ClassifyMs`]): read sample 0 of the
    /// stored MS depth with the `vrs_ms` pipeline instead of the resolve.
    ///
    /// Only called when classifying, so both `vrs` and `vrs_compute` are
    /// present. `cmd` must be recording, outside a render pass.
    pub(super) unsafe fn record_vrs_generate(
        &self,
        cmd: vk::CommandBuffer,
        slot: usize,
        d_threshold: f32,
        depth_ms: bool,
    ) {
        let device = &self.device.device;
        let vrs = self.targets.vrs.as_ref().expect("classify_vrs implies vrs");
        let compute = self
            .pipelines
            .vrs_compute
            .as_ref()
            .expect("classify_vrs implies vrs_compute");
        // Depth is already in SAMPLEABLE_DEPTH_REST_LAYOUT (the post-scene
        // rest barrier). Sampleable: the single-sample resolve under MSAA,
        // else the depth image itself. Classify-only MSAA: the stored MS depth
        // attachment, read through the sample-0 pipeline (same layout).
        let (pipeline, depth_view) = if depth_ms {
            (
                compute
                    .pipeline_ms
                    .expect("ClassifyMs implies the vrs_ms pipeline"),
                self.targets.depth[slot].view(),
            )
        } else {
            (compute.pipeline, self.targets.sampleable_depth(slot).view())
        };
        let tiles = vrs.tiles();
        let use_history = self.slots[FrameSlot::new(slot)].vrs_history;
        let allow_4x4 = self
            .device
            .fragment_shading_rate
            .as_ref()
            .is_some_and(|f| f.allow_4x4(self.targets.samples));
        let write_mix = crate::profile::is_enabled();
        let flags = classify_flags(allow_4x4, use_history, write_mix);
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            let depth_info = [vk::DescriptorImageInfo::default()
                .sampler(compute.depth_sampler)
                .image_view(depth_view)
                .image_layout(SAMPLEABLE_DEPTH_REST_LAYOUT)];
            let rate_info = [vk::DescriptorImageInfo::default()
                .image_view(vrs.view(slot))
                .image_layout(vk::ImageLayout::GENERAL)];
            let history_info = [vk::DescriptorImageInfo::default()
                .image_view(vrs.history_view(slot))
                .image_layout(vk::ImageLayout::GENERAL)];
            let mix_info = [vk::DescriptorBufferInfo::default()
                .buffer(vrs.mix_gpu(slot))
                .offset(0)
                .range(MIX_BYTES)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&rate_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&history_info),
                vk::WriteDescriptorSet::default()
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&mix_info),
            ];
            self.device.push_descriptor.cmd_push_descriptor_set(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                compute.layout,
                0,
                &writes,
            );

            let push = VrsPush {
                d_threshold,
                texel_w: vrs.texel_size.width,
                texel_h: vrs.texel_size.height,
                tiles_x: tiles.width,
                tiles_y: tiles.height,
                depth_w: self.render_extent.width,
                depth_h: self.render_extent.height,
                flags,
            };
            device.cmd_push_constants(
                cmd,
                compute.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&push),
            );
            device.cmd_dispatch(cmd, tiles.width.div_ceil(8), tiles.height.div_ceil(8), 1);

            if write_mix {
                // COMPUTE / SHADER_STORAGE_WRITE → COPY / TRANSFER_READ.
                let mix_to_copy = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&mix_to_copy),
                );
                // The mapped read happens after this slot's timeline wait
                // (one cycle later).
                vrs.mix[slot].record_copy_to_host(device, cmd, MIX_BYTES);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The classifier only runs on the GPU, so these tests run it as
    //! `vrs.comp.slang` spells it. [`Shader::read`] matches the shader's
    //! `classify` and the rate / history / mix tail of `computeMain` against
    //! fixed templates, so any change to that logic fails here, and takes
    //! every literal (sky depth, flatness factor, rate codes, flag bits) from
    //! the source. The host side the shader reads ([`classify_flags`],
    //! [`VrsPush`], the mix lane order) is checked against the same text.

    use super::*;

    const SHADER: &str = include_str!("../../shaders/vrs.comp.slang");

    /// `classify(dmin, dmax)`; each `{}` is a literal [`Shader::read`] takes.
    const CLASSIFY: &str = "
        if (dmax < {}) {
            return ((pc.flags & {}) != 0) ? uint({}) : uint({});
        }
        bool farTile = dmax < pc.dThreshold;
        bool flat = (dmax - dmin) < (pc.dThreshold * {});
        return (farTile && flat) ? uint({}) : {};";

    /// `computeMain` from the tile's own class to the end: one-tile dilation,
    /// history, the written rate, and the mix histogram.
    const TILE: &str = "
        uint raw = classify(dmin, dmax);
        bool neighborFull = false;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0)
                    continue;
                uint n = uint(ly + 1 + dy) * 10u + uint(lx + 1 + dx);
                if (classify(s_dmin[n], s_dmax[n]) == {})
                    neighborFull = true;
            }
        }
        uint prev = uint({});
        if ((pc.flags & {}) != 0) {
            prev = history[gid.xy];
        }
        uint rate = raw;
        if (raw == {} || prev == {} || neighborFull) {
            rate = {};
        }
        rateOut[gid.xy] = rate;
        history[gid.xy] = raw;
        if ((pc.flags & {}) != 0) {
            uint mixIdx = rate == {} ? 0u : (rate == uint({}) ? 1u : 2u);
            uint unused;
            InterlockedAdd(mixCounts[mixIdx], 1, unused);
        }";

    /// Whitespace-normalised source without `//` comments.
    fn squash(src: &str) -> String {
        src.lines()
            .map(|line| line.split("//").next().unwrap_or(""))
            .flat_map(str::split_whitespace)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Matches `src` against `template` (both squashed). Each `{}` captures
    /// the text up to the next fixed part; every fixed part must match.
    fn captures(src: &str, template: &str) -> Vec<String> {
        let src = squash(src);
        let template = squash(template);
        let mut parts = template.split("{}");
        let head = parts.next().unwrap_or("");
        let mut rest = src
            .strip_prefix(head)
            .unwrap_or_else(|| panic!("vrs.comp.slang drifted: `{src}` does not start `{head}`"));
        let mut out = Vec::new();
        for part in parts {
            if part.is_empty() {
                out.push(rest.to_string());
                rest = "";
                continue;
            }
            let at = rest
                .find(part)
                .unwrap_or_else(|| panic!("vrs.comp.slang drifted: no `{part}` in `{rest}`"));
            out.push(rest[..at].to_string());
            rest = &rest[at + part.len()..];
        }
        assert!(rest.is_empty(), "vrs.comp.slang drifted: trailing `{rest}`");
        out
    }

    /// Body of the function `signature` opens, braces balanced.
    fn body(signature: &str) -> &'static str {
        let start = SHADER
            .find(signature)
            .unwrap_or_else(|| panic!("vrs.comp.slang has no `{signature}`"));
        let open = start + SHADER[start..].find('{').expect("a function body");
        let mut depth = 0usize;
        for (i, c) in SHADER[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return &SHADER[open + 1..open + i];
                    }
                }
                _ => {}
            }
        }
        panic!("`{signature}` has an unbalanced body");
    }

    fn uint(lit: &str) -> u32 {
        lit.trim_end_matches('u')
            .parse()
            .unwrap_or_else(|_| panic!("`{lit}` is not a uint literal"))
    }

    fn float(lit: &str) -> f32 {
        lit.parse()
            .unwrap_or_else(|_| panic!("`{lit}` is not a float literal"))
    }

    /// A rate code as the shader writes it: `0u` or `(Wu << S) | Hu`.
    fn rate_code(expr: &str) -> u32 {
        let Some((wide, high)) = expr.split_once(" | ") else {
            return uint(expr);
        };
        let (w, shift) = wide
            .strip_prefix('(')
            .and_then(|e| e.strip_suffix(')'))
            .and_then(|e| e.split_once(" << "))
            .unwrap_or_else(|| panic!("`{expr}` is not a rate code"));
        (uint(w) << uint(shift)) | uint(high)
    }

    /// Vulkan's shading-rate attachment texel for a `w`×`h` fragment:
    /// `(log2(w) << 2) | log2(h)`.
    fn vk_rate(w: u32, h: u32) -> u32 {
        (w.ilog2() << 2) | h.ilog2()
    }

    /// The classifier as `vrs.comp.slang` has it, every literal read from it.
    struct Shader {
        sky_depth: f32,
        allow_4x4: u32,
        sky_wide: u32,
        sky_narrow: u32,
        flat: f32,
        far_flat: u32,
        near: u32,
        neighbor_full: u32,
        no_history: u32,
        use_history: u32,
        raw_full: u32,
        prev_full: u32,
        full: u32,
        write_mix: u32,
        mix_full: u32,
        mix_2x2: u32,
    }

    impl Shader {
        fn read() -> Self {
            let c = captures(body("uint classify(float dmin, float dmax)"), CLASSIFY);
            let main = body("void computeMain(");
            let tail = &main[main
                .find("uint raw = classify")
                .expect("computeMain classifies its tile")..];
            let t = captures(tail, TILE);
            Shader {
                sky_depth: float(&c[0]),
                allow_4x4: uint(&c[1]),
                sky_wide: rate_code(&c[2]),
                sky_narrow: rate_code(&c[3]),
                flat: float(&c[4]),
                far_flat: rate_code(&c[5]),
                near: rate_code(&c[6]),
                neighbor_full: rate_code(&t[0]),
                no_history: rate_code(&t[1]),
                use_history: uint(&t[2]),
                raw_full: rate_code(&t[3]),
                prev_full: rate_code(&t[4]),
                full: rate_code(&t[5]),
                write_mix: uint(&t[6]),
                mix_full: rate_code(&t[7]),
                mix_2x2: rate_code(&t[8]),
            }
        }

        /// `classify(dmin, dmax)` with `pc.dThreshold` and `pc.flags`.
        fn classify(&self, dmin: f32, dmax: f32, d_threshold: f32, flags: u32) -> u32 {
            if dmax < self.sky_depth {
                return if flags & self.allow_4x4 != 0 {
                    self.sky_wide
                } else {
                    self.sky_narrow
                };
            }
            let far_tile = dmax < d_threshold;
            let flat = (dmax - dmin) < (d_threshold * self.flat);
            if far_tile && flat {
                self.far_flat
            } else {
                self.near
            }
        }

        /// The rate `computeMain` writes for a tile of class `raw` with these
        /// neighbour classes and history texel.
        fn rate(&self, raw: u32, neighbors: [u32; 8], history: u32, flags: u32) -> u32 {
            let neighbor_full = neighbors.contains(&self.neighbor_full);
            let prev = if flags & self.use_history != 0 {
                history
            } else {
                self.no_history
            };
            if raw == self.raw_full || prev == self.prev_full || neighbor_full {
                self.full
            } else {
                raw
            }
        }

        /// The `mixCounts` lane `computeMain` counts `rate` in.
        fn mix_lane(&self, rate: u32) -> usize {
            if rate == self.mix_full {
                0
            } else if rate == self.mix_2x2 {
                1
            } else {
                2
            }
        }
    }

    const D_THRESHOLD: f32 = 0.01;

    #[test]
    fn sky_is_coarse_and_uses_4x4_when_advertised() {
        let s = Shader::read();
        let narrow = classify_flags(false, false, false);
        let wide = classify_flags(true, false, false);
        assert_eq!(s.classify(0.0, 0.0, D_THRESHOLD, narrow), vk_rate(2, 2));
        assert_eq!(s.classify(0.0, 0.0, D_THRESHOLD, wide), vk_rate(4, 4));
        assert_eq!(
            s.classify(0.0, s.sky_depth * 0.5, D_THRESHOLD, wide),
            vk_rate(4, 4)
        );
    }

    #[test]
    fn far_flat_terrain_stays_2x2_even_when_4x4_exists() {
        let s = Shader::read();
        for allow_4x4 in [false, true] {
            let flags = classify_flags(allow_4x4, false, false);
            assert_eq!(s.classify(0.001, 0.002, D_THRESHOLD, flags), vk_rate(2, 2));
        }
    }

    #[test]
    fn near_or_discontinuous_tiles_are_full_rate() {
        let s = Shader::read();
        let flags = classify_flags(true, false, false);
        assert_eq!(s.classify(0.5, 0.6, D_THRESHOLD, flags), vk_rate(1, 1));
        // Far but a silhouette crosses the tile (range > half the threshold).
        assert_eq!(s.classify(0.0, 0.009, D_THRESHOLD, flags), vk_rate(1, 1));
    }

    #[test]
    fn conservative_holds_last_near_and_dilates_full_rate() {
        let s = Shader::read();
        let (r1, r2, r4) = (vk_rate(1, 1), vk_rate(2, 2), vk_rate(4, 4));
        let history = classify_flags(true, true, false);
        let coarse = [r4; 8];
        // A tile that was near at the last classify is not coarsened.
        assert_eq!(s.rate(r4, coarse, r1, history), r1);
        // A full-rate neighbour dilates into the tile.
        let mut one_near = [r2; 8];
        one_near[3] = r1;
        assert_eq!(s.rate(r2, one_near, r2, history), r1);
        assert_eq!(s.rate(r4, coarse, r2, history), r4);
        assert_eq!(s.rate(r2, coarse, r4, history), r2);
        // Finer is always allowed.
        assert_eq!(s.rate(r1, coarse, r4, history), r1);
        // First classify after create/recreate: the history texel is not read.
        assert_eq!(
            s.rate(r4, coarse, r1, classify_flags(true, false, false)),
            r4
        );
    }

    #[test]
    fn rate_encoding_matches_vk_fragment_size_pack() {
        let s = Shader::read();
        assert_eq!(
            (s.sky_wide, s.sky_narrow, s.far_flat, s.near),
            (vk_rate(4, 4), vk_rate(2, 2), vk_rate(2, 2), vk_rate(1, 1)),
        );
        // Dilation, history hold and the written rate all mean 1x1.
        for code in [s.neighbor_full, s.raw_full, s.prev_full, s.full] {
            assert_eq!(code, vk_rate(1, 1));
        }
        assert_eq!(s.no_history, vk_rate(2, 2), "no history holds nothing");
        assert_eq!(
            (vk_rate(1, 1), vk_rate(2, 2), vk_rate(4, 4)),
            (0, 0b0101, 0b1010)
        );
    }

    #[test]
    fn mix_lanes_are_1x1_2x2_4x4() {
        // `Vrs::mix` and the Vrs1x1 / Vrs2x2 / Vrs4x4 gauges read this order.
        let s = Shader::read();
        let lanes = [vk_rate(1, 1), vk_rate(2, 2), vk_rate(4, 4)].map(|r| s.mix_lane(r));
        assert_eq!(lanes, [0, 1, 2]);
        assert_eq!(MIX_BYTES, (lanes.len() * size_of::<u32>()) as u64);
    }

    #[test]
    fn push_layout_is_tight_u32s() {
        // The shader's `Push` block, in order, against the Rust field each
        // member lands in.
        let block = squash(SHADER);
        let block = block
            .split_once("struct Push {")
            .and_then(|(_, rest)| rest.split_once("};"))
            .expect("vrs.comp.slang declares `struct Push`")
            .0;
        let members: Vec<(&str, &str)> = block
            .split(';')
            .map(str::trim)
            .filter(|m| !m.is_empty())
            .map(|m| m.split_once(' ').expect("`type name`"))
            .collect();
        let rust = [
            (
                "float",
                "dThreshold",
                std::mem::offset_of!(VrsPush, d_threshold),
            ),
            ("uint", "texelW", std::mem::offset_of!(VrsPush, texel_w)),
            ("uint", "texelH", std::mem::offset_of!(VrsPush, texel_h)),
            ("uint", "tilesX", std::mem::offset_of!(VrsPush, tiles_x)),
            ("uint", "tilesY", std::mem::offset_of!(VrsPush, tiles_y)),
            ("uint", "depthW", std::mem::offset_of!(VrsPush, depth_w)),
            ("uint", "depthH", std::mem::offset_of!(VrsPush, depth_h)),
            ("uint", "flags", std::mem::offset_of!(VrsPush, flags)),
        ];
        assert_eq!(
            members,
            rust.iter()
                .map(|&(ty, name, _)| (ty, name))
                .collect::<Vec<_>>()
        );
        for (i, &(_, name, offset)) in rust.iter().enumerate() {
            assert_eq!(offset, i * 4, "{name}");
        }
        assert_eq!(size_of::<VrsPush>(), rust.len() * 4);
    }

    #[test]
    fn flag_bits_do_not_overlap() {
        let s = Shader::read();
        assert_eq!(classify_flags(true, false, false), s.allow_4x4);
        assert_eq!(classify_flags(false, true, false), s.use_history);
        assert_eq!(classify_flags(false, false, true), s.write_mix);
        assert_eq!(classify_flags(false, false, false), 0);
        let all = classify_flags(true, true, true);
        assert_eq!(all, s.allow_4x4 | s.use_history | s.write_mix);
        assert_eq!(all.count_ones(), 3);
    }
}
