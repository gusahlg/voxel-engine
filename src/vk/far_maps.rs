//! Datum storage and albedo cubes for [`crate::FarShape::Mapped`].
//!
//! One device-local buffer holds eight headers plus every datum. Cube images
//! are created on `set_map`. If that allocation fails, the failure is logged
//! once for the map id. With no complete cube on screen the slot keeps its
//! datum with `albedo_size` 0, so the shader uses the flat per-face colours;
//! a complete cube stays up. [`crate::FarMapError::OutOfMemory`] is not
//! returned to the caller. Face bytes and mip blits are recorded into the
//! frame command buffer, which is the queue that can blit; a face whose cube
//! is missing is dropped. The datum copy follows the block-texture / material
//! transfer lane (timeline wait, queue-family ownership when the families
//! differ, staging retired on that timeline). Nothing here waits on the GPU.
//!
//! A second `set_map` on a slot whose sampled cube already has all six faces
//! landed keeps that cube on the descriptor and in the header. The new cube
//! is pending: uploads fill it, and it becomes the sampled cube only once
//! every new face and its mips have signaled the graphics timeline. The old
//! cube is then retired at the swap frame's timeline value. The datum is
//! written on the call, not at the swap. `albedo_size` 0 drops both cubes
//! on that call.

use ash::vk;

use super::alloc::{create_buffer, create_filled_staging};
use super::buffers::RetireQueue;
use super::image::{allocate_and_bind_image, color_range};
use super::mesh_residency::{CopyBarrier, CopyConsumer, copy_barrier};
use super::timeline::TimelineValue;
use super::transfer::{TransferCtx, UploadRetire};
use crate::far_body::{FarMapError, MAX_FAR_MAPS};

/// Stages that first read the datum buffer and the albedo cubes (sky fragment).
pub(crate) const FAR_MAP_CONSUMER_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::FRAGMENT_SHADER;

/// Copy-barrier consumer of the datum buffer: sky-fragment storage reads.
const DATUM_CONSUMER: CopyConsumer = (
    FAR_MAP_CONSUMER_STAGES,
    vk::AccessFlags2::SHADER_STORAGE_READ,
);

const MAX_G: usize = 65;
/// Floats reserved for one map, including the padding past a smaller `g`.
const SLOT_FLOATS: usize = 6 * MAX_G * MAX_G;
const HEADER_UINTS: usize = MAX_FAR_MAPS * 8;
const BUFFER_UINTS: usize = HEADER_UINTS + MAX_FAR_MAPS * SLOT_FLOATS;
/// Six cube faces, bit `f` for face `f`.
const FACE_MASK: u32 = 0b11_1111;

struct GpuCube {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    mips: u32,
    size: u32,
    /// Recorded transition out of `UNDEFINED`. The next commands in that
    /// command buffer see `SHADER_READ_ONLY`.
    cleared: bool,
}

impl GpuCube {
    unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// Where [`SlotCubes::install`] puts a new cube.
enum CubeInstall {
    /// `albedo_size` 0. Both cubes are dropped and the shader uses flat colours.
    Flat,
    /// No complete cube is on screen. The new cube is sampled as faces land.
    Immediate,
    /// A complete cube is on screen. The new cube stays off-screen until its
    /// six faces have landed.
    Pending,
}

fn cube_install(active_landed: Option<u32>, albedo_size: u32) -> CubeInstall {
    if albedo_size == 0 {
        CubeInstall::Flat
    } else if active_landed.is_some_and(|bits| bits & FACE_MASK == FACE_MASK) {
        CubeInstall::Pending
    } else {
        CubeInstall::Immediate
    }
}

/// Cubes [`SlotCubes::install`] or [`SlotCubes::clear`] removed. The caller
/// retires them after the frames that might still sample them.
#[derive(Debug, Default, PartialEq, Eq)]
struct Retired<A> {
    active: Option<A>,
    pending: Option<A>,
}

impl<A> Retired<A> {
    fn drain(self, mut f: impl FnMut(A)) {
        if let Some(cube) = self.active {
            f(cube);
        }
        if let Some(cube) = self.pending {
            f(cube);
        }
    }
}

struct Tracked<A> {
    size: u32,
    /// Faces whose uploads are visible to later draws. Bit `f` is face `f`.
    landed: u32,
    /// `set_map` generation that owns this cube. A face stamped with another
    /// generation is not for this image.
    generation: u32,
    cube: A,
}

/// Sampled cube plus an optional replacement. `A` is the device image;
/// tests pass an id. No Vulkan types.
///
/// `land_face` means that face's upload, mips included, is visible to later
/// draws. The device code calls it in the recording frame for the sampled
/// cube (that frame's barriers cover the texels) and, for a pending cube,
/// only after the frame that recorded the upload has signaled the graphics
/// timeline.
struct SlotCubes<A> {
    active: Option<Tracked<A>>,
    pending: Option<Tracked<A>>,
}

impl<A> SlotCubes<A> {
    fn new() -> Self {
        Self {
            active: None,
            pending: None,
        }
    }

    /// `cube` is `None` when `size` is 0 or the image could not be allocated.
    /// A failed allocation of a pending cube drops any cube already pending
    /// and leaves the sampled one in place.
    #[must_use]
    fn install(&mut self, size: u32, cube: Option<A>, generation: u32) -> Retired<A> {
        let kind = cube_install(self.active.as_ref().map(|c| c.landed), size);
        match kind {
            CubeInstall::Flat => {
                debug_assert!(cube.is_none(), "albedo size 0 allocates no cube");
                self.take_both()
            }
            CubeInstall::Immediate => {
                let retired = self.take_both();
                if let Some(cube) = cube {
                    self.active = Some(Tracked {
                        size,
                        landed: 0,
                        generation,
                        cube,
                    });
                }
                retired
            }
            CubeInstall::Pending => {
                let pending = self.pending.take().map(|c| c.cube);
                if let Some(cube) = cube {
                    self.pending = Some(Tracked {
                        size,
                        landed: 0,
                        generation,
                        cube,
                    });
                }
                Retired {
                    active: None,
                    pending,
                }
            }
        }
    }

    #[must_use]
    fn clear(&mut self) -> Retired<A> {
        self.take_both()
    }

    fn take_both(&mut self) -> Retired<A> {
        Retired {
            active: self.active.take().map(|c| c.cube),
            pending: self.pending.take().map(|c| c.cube),
        }
    }

    /// Landed mask published to the shader. The pending cube's bits are not
    /// included.
    fn sampled_landed(&self) -> u32 {
        self.active
            .as_ref()
            .map(|c| c.landed & FACE_MASK)
            .unwrap_or(0)
    }

    /// Edge of the sampled cube, or 0 when the shader should use flat colours.
    fn sampled_size(&self) -> u32 {
        self.active.as_ref().map(|c| c.size).unwrap_or(0)
    }

    fn has_pending(&self) -> bool {
        self.pending.is_some()
    }

    fn active_cube(&self) -> Option<&A> {
        self.active.as_ref().map(|c| &c.cube)
    }

    /// Cube that receives uploads: the pending replacement when one exists.
    fn upload(&self) -> Option<&Tracked<A>> {
        if let Some(pending) = self.pending.as_ref() {
            Some(pending)
        } else {
            self.active.as_ref()
        }
    }

    fn upload_mut(&mut self) -> Option<&mut Tracked<A>> {
        if let Some(pending) = self.pending.as_mut() {
            Some(pending)
        } else {
            self.active.as_mut()
        }
    }

    fn accepts(&self, generation: u32) -> bool {
        self.upload()
            .is_some_and(|cube| cube.generation == generation)
    }

    /// Set `face`'s bit on the upload target. When that completes a pending
    /// cube, it becomes the sampled cube and the previous one is returned.
    fn land_face(&mut self, face: u32) -> Option<A> {
        if face >= 6 {
            return None;
        }
        {
            let Some(target) = self.upload_mut() else {
                return None;
            };
            target.landed |= 1 << face;
        }
        let complete = self
            .pending
            .as_ref()
            .is_some_and(|cube| cube.landed & FACE_MASK == FACE_MASK);
        if !complete {
            return None;
        }
        let next = self.pending.take().expect("pending cube completed");
        self.active.replace(next).map(|cube| cube.cube)
    }
}

struct Slot {
    g: u32,
    min_off: f32,
    max_off: f32,
    /// Bumped on every `set_map` and `clear`. Queued faces stamp it; a face
    /// whose generation no longer matches is dropped.
    generation: u32,
    cubes: SlotCubes<GpuCube>,
}

struct PendingFace {
    id: u8,
    face: u32,
    generation: u32,
    bytes: Box<[u8]>,
}

/// A pending-cube face recorded into the frame that signals `done_at`.
/// Mips are in that same command buffer, so the signal covers them.
struct InflightFace {
    id: u8,
    face: u32,
    generation: u32,
    done_at: TimelineValue,
}

pub(crate) struct FarMaps {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    /// Header (8 × 8 uints) then `MAX_FAR_MAPS` datum slots. Floats are stored
    /// as their bits. Uploaded whole when `buffer_dirty`.
    words: Vec<u32>,
    buffer_dirty: bool,
    /// `true` after the first copy has handed the buffer to graphics, so the
    /// next overwrite has to order against fragment reads.
    buffer_on_graphics: bool,
    sampler: vk::Sampler,
    dummy: GpuCube,
    slots: [Slot; MAX_FAR_MAPS],
    pending_faces: Vec<PendingFace>,
    /// Pending-cube faces whose blits are in a frame that has not signaled
    /// yet. Applied by [`Self::promote_landed`].
    inflight_faces: Vec<InflightFace>,
    pending_retire: Vec<GpuCube>,
    image_retire: RetireQueue<GpuCube>,
    /// Face and datum staging, and the dedicated-family datum release.
    retire: UploadRetire,
    /// Bit i is set after map i's cube allocation has failed and been logged.
    /// Cleared when a later `set_map` installs a cube or an explicit
    /// `albedo_size` of 0, so the next failure logs again. Not per frame.
    oom_logged: u8,
}

impl FarMaps {
    pub(crate) fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        command_pool: vk::CommandPool,
    ) -> Self {
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let fmt = unsafe {
            instance.get_physical_device_format_properties(physical, vk::Format::R8G8B8A8_SRGB)
        };
        let need = vk::FormatFeatureFlags::SAMPLED_IMAGE
            | vk::FormatFeatureFlags::BLIT_SRC
            | vk::FormatFeatureFlags::BLIT_DST
            | vk::FormatFeatureFlags::TRANSFER_SRC
            | vk::FormatFeatureFlags::TRANSFER_DST;
        assert!(
            fmt.optimal_tiling_features.contains(need),
            "R8G8B8A8_SRGB cannot blit far-map cubes: {:?}",
            fmt.optimal_tiling_features
        );
        let (buffer, memory) = create_buffer(
            device,
            &memory_props,
            (BUFFER_UINTS * 4) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            "far-map datum",
        );
        let sampler = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                        .min_lod(0.0)
                        .max_lod(vk::LOD_CLAMP_NONE),
                    None,
                )
                .expect("create far-map albedo sampler")
        };
        let dummy = unsafe { create_cube(device, &memory_props, 1).expect("far-map dummy cube") };
        let slot = || Slot {
            g: 0,
            min_off: 0.0,
            max_off: 0.0,
            generation: 1,
            cubes: SlotCubes::new(),
        };
        Self {
            buffer,
            memory,
            words: vec![0; BUFFER_UINTS],
            buffer_dirty: true,
            buffer_on_graphics: false,
            sampler,
            dummy,
            slots: std::array::from_fn(|_| slot()),
            pending_faces: Vec::new(),
            inflight_faces: Vec::new(),
            pending_retire: Vec::new(),
            image_retire: RetireQueue::new(),
            retire: UploadRetire::new(command_pool, "far-map"),
            oom_logged: 0,
        }
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    pub(crate) fn sampler(&self) -> vk::Sampler {
        self.sampler
    }

    /// Sampled cube view for `id`, or the 1×1 dummy when that slot has no
    /// albedo on screen. A pending replacement is not returned here.
    pub(crate) fn view(&self, id: usize) -> vk::ImageView {
        self.slots
            .get(id)
            .and_then(|s| s.cubes.active_cube())
            .map(|c| c.view)
            .unwrap_or(self.dummy.view)
    }

    /// Maximum datum offset of each map, for the hi-radius cone. An empty
    /// slot reports 0.
    pub(crate) fn max_offsets(&self) -> [f32; MAX_FAR_MAPS] {
        let mut out = [0.0; MAX_FAR_MAPS];
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.g >= 2 {
                out[i] = slot.max_off;
            }
        }
        out
    }

    /// Minimum datum offset of each map, for the lo-sphere disc. An empty
    /// slot reports 0, the same flat datum the shader samples.
    pub(crate) fn min_offsets(&self) -> [f32; MAX_FAR_MAPS] {
        let mut out = [0.0; MAX_FAR_MAPS];
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.g >= 2 {
                out[i] = slot.min_off;
            }
        }
        out
    }

    /// Grid size and `set_map` generation. `None` when `id` is out of range.
    /// An empty slot has `g == 0` and still has a generation.
    pub(crate) fn map_stamp(&self, id: usize) -> Option<(u32, u32)> {
        let slot = self.slots.get(id)?;
        Some((slot.g, slot.generation))
    }

    /// Copy the live datum (`6 * g * g` floats, face-major) into `out`.
    /// `None` when the slot is empty (`g < 2`); the shader then samples a
    /// flat zero and no horizon table is published for it.
    pub(crate) fn copy_datum(&self, id: usize, out: &mut Vec<f32>) -> Option<u32> {
        let slot = self.slots.get(id)?;
        if slot.g < 2 {
            return None;
        }
        let g = slot.g;
        let n = 6 * g as usize * g as usize;
        let base = HEADER_UINTS + id * SLOT_FLOATS;
        if base + n > self.words.len() {
            return None;
        }
        out.clear();
        out.reserve(n);
        for i in 0..n {
            out.push(f32::from_bits(self.words[base + i]));
        }
        Some(g)
    }

    pub(crate) fn has_garbage(&self) -> bool {
        !self.image_retire.is_empty() || self.retire.has_garbage()
    }

    /// A datum overwrite of a buffer graphics has already read, a face blit
    /// onto a cube a previous frame may still be sampling, or a finished
    /// replacement that will republish the shared header. The frame loop
    /// submits pending batches first so `last_render_value` covers them.
    pub(crate) fn has_overwrite_pending(&self) -> bool {
        if self.buffer_dirty && self.buffer_on_graphics {
            return true;
        }
        // The swap writes the header. Previous frames, including an
        // unsubmitted batch, may still be reading the old words.
        if self.buffer_on_graphics && self.pending_swap_is_recorded() {
            return true;
        }
        self.pending_faces.iter().any(|face| {
            self.slots.get(face.id as usize).is_some_and(|slot| {
                slot.generation == face.generation
                    && !slot.cubes.has_pending()
                    && slot
                        .cubes
                        .active
                        .as_ref()
                        .is_some_and(|cube| cube.generation == face.generation && cube.cube.cleared)
            })
        })
    }

    /// All six faces of some pending cube are recorded, so a later promote
    /// may swap and rewrite the header.
    fn pending_swap_is_recorded(&self) -> bool {
        self.slots.iter().enumerate().any(|(id, slot)| {
            let Some(pending) = slot.cubes.pending.as_ref() else {
                return false;
            };
            let mut mask = pending.landed;
            for face in &self.inflight_faces {
                if face.id as usize == id && face.generation == pending.generation && face.face < 6
                {
                    mask |= 1 << face.face;
                }
            }
            mask & FACE_MASK == FACE_MASK
        })
    }

    /// Install `datum` and, when `albedo_size > 0`, a cube. The datum (and
    /// its min/max) is written on this call, including while the new cube is
    /// only pending. The header's landed mask and albedo edge stay on the
    /// sampled cube until a pending cube swaps in.
    ///
    /// A slot with a sampled cube whose six faces have landed keeps that cube
    /// on screen. The new cube is pending; [`Self::queue_face`] fills it, and
    /// [`Self::promote_landed`] swaps once those uploads have signaled the
    /// graphics timeline. A second call replaces the pending cube. A slot
    /// with no cube, or a cube still missing a face, is replaced immediately
    /// and the previous images are retired at the next flush. `albedo_size`
    /// 0 drops both cubes. No GPU wait.
    ///
    /// A failed cube allocation is logged once per map id (until a later call
    /// installs a cube or requests `albedo_size` 0). With no complete cube
    /// the datum is kept with `albedo_size` 0. With one, that cube stays.
    /// [`FarMapError::OutOfMemory`] is not returned. [`Self::queue_face`]
    /// then drops faces for this generation because no cube owns it.
    pub(crate) fn set_map(
        &mut self,
        device: &ash::Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        id: u8,
        g: u32,
        datum: &[f32],
        albedo_size: u32,
    ) -> Result<(), FarMapError> {
        if id as usize >= MAX_FAR_MAPS {
            return Err(FarMapError::BadId);
        }
        let slot_i = id as usize;
        let bit = 1u8 << id;
        let active_landed = self.slots[slot_i]
            .cubes
            .active
            .as_ref()
            .map(|cube| cube.landed);
        let kind = cube_install(active_landed, albedo_size);
        let cube = if albedo_size == 0 {
            self.oom_logged &= !bit;
            None
        } else {
            match unsafe { create_cube(device, memory_props, albedo_size) } {
                Ok(cube) => {
                    self.oom_logged &= !bit;
                    Some(cube)
                }
                Err(FarMapError::OutOfMemory) => {
                    if self.oom_logged & bit == 0 {
                        self.oom_logged |= bit;
                        if matches!(kind, CubeInstall::Pending) {
                            log::warn!(
                                "far map {id}: albedo cube allocation failed; keeping the cube on screen until the next set_far_map"
                            );
                        } else {
                            log::warn!(
                                "far map {id}: albedo cube allocation failed; drawing flat per-face colours until the next set_far_map"
                            );
                        }
                    }
                    None
                }
                Err(err) => return Err(err),
            }
        };
        let generation = {
            let slot = &mut self.slots[slot_i];
            slot.generation = slot.generation.wrapping_add(1);
            slot.generation
        };
        let retired = self.slots[slot_i]
            .cubes
            .install(albedo_size, cube, generation);
        retired.drain(|old| self.pending_retire.push(old));
        let slot = &mut self.slots[slot_i];
        slot.g = g;
        let base = HEADER_UINTS + slot_i * SLOT_FLOATS;
        for w in &mut self.words[base..base + SLOT_FLOATS] {
            *w = 0;
        }
        let mut min_off = f32::INFINITY;
        let mut max_off = f32::NEG_INFINITY;
        let n = datum.len().min(SLOT_FLOATS);
        for (i, v) in datum.iter().take(n).enumerate() {
            let v = if v.is_finite() { *v } else { 0.0 };
            min_off = min_off.min(v);
            max_off = max_off.max(v);
            self.words[base + i] = v.to_bits();
        }
        if !min_off.is_finite() {
            min_off = 0.0;
            max_off = 0.0;
        }
        slot.min_off = min_off;
        slot.max_off = max_off;
        self.write_header(slot_i);
        self.buffer_dirty = true;
        Ok(())
    }

    /// Queue one face for the cube currently accepting uploads (the pending
    /// replacement when one exists, otherwise the sampled cube). Dropped at
    /// flush if `generation` is stale, the cube is gone, or the byte length
    /// does not match that cube. Does not wait.
    pub(crate) fn queue_face(&mut self, id: u8, face: u32, bytes: Box<[u8]>) {
        let Some(slot) = self.slots.get(id as usize) else {
            return;
        };
        self.pending_faces.push(PendingFace {
            id,
            face,
            generation: slot.generation,
            bytes,
        });
    }

    /// Drop the slot, sampled cube and pending cube. Both images are retired
    /// at the next flush. The header reports an empty map (`g == 0`) so the
    /// shader's datum read is zero.
    pub(crate) fn clear(&mut self, id: u8) {
        if id as usize >= MAX_FAR_MAPS {
            return;
        }
        // The next failed allocation of this id should log again.
        self.oom_logged &= !(1u8 << id);
        let retired = self.slots[id as usize].cubes.clear();
        retired.drain(|old| self.pending_retire.push(old));
        let slot = &mut self.slots[id as usize];
        slot.generation = slot.generation.wrapping_add(1);
        slot.g = 0;
        slot.min_off = 0.0;
        slot.max_off = 0.0;
        let base = HEADER_UINTS + id as usize * SLOT_FLOATS;
        for w in &mut self.words[base..base + SLOT_FLOATS] {
            *w = 0;
        }
        self.write_header(id as usize);
        self.buffer_dirty = true;
    }

    fn write_header(&mut self, id: usize) {
        let slot = &self.slots[id];
        let b = id * 8;
        self.words[b] = slot.g;
        self.words[b + 1] = (id * SLOT_FLOATS) as u32;
        self.words[b + 2] = slot.min_off.to_bits();
        self.words[b + 3] = slot.max_off.to_bits();
        self.words[b + 4] = slot.cubes.sampled_landed();
        self.words[b + 5] = slot.cubes.sampled_size();
        self.words[b + 6] = 0;
        self.words[b + 7] = 0;
    }

    /// Record pending clears, face blits, and the datum copy. Returns the
    /// transfer-lane value the graphics submit must wait on, if the copy left
    /// this command buffer.
    pub unsafe fn flush(&mut self, ctx: &mut TransferCtx<'_>) -> Option<TimelineValue> {
        let device = ctx.device;
        let graphics_cmd = ctx.graphics_cmd;
        let done_at = ctx.done_at;
        unsafe {
            if !self.inflight_faces.is_empty() {
                let current = ctx.graphics_timeline.counter(device);
                self.promote_landed(current);
            }
            if !self.dummy.cleared {
                clear_cube(device, graphics_cmd, &self.dummy);
                self.dummy.cleared = true;
            }
            for slot in &mut self.slots {
                if let Some(cube) = slot.cubes.active.as_mut() {
                    prepare_cube(device, graphics_cmd, &mut cube.cube);
                }
                if let Some(cube) = slot.cubes.pending.as_mut() {
                    prepare_cube(device, graphics_cmd, &mut cube.cube);
                }
            }
            let face_staging = self.record_faces(ctx);
            if let Some(staging) = face_staging {
                self.retire.retire_on_graphics(done_at, staging);
            }
            let wait = if self.buffer_dirty {
                self.upload_buffer(ctx)
            } else {
                None
            };
            for cube in self.pending_retire.drain(..) {
                self.image_retire.push(done_at, cube);
            }
            wait
        }
    }

    /// Move pending-cube faces whose recording frame has signaled onto the
    /// slot. The sixth such face swaps that cube onto the descriptor and the
    /// header; the old cube is retired with this frame.
    fn promote_landed(&mut self, current: TimelineValue) {
        let mut i = 0;
        while i < self.inflight_faces.len() {
            if self.inflight_faces[i].done_at > current {
                i += 1;
                continue;
            }
            let face = self.inflight_faces.swap_remove(i);
            self.promote_face(face);
        }
    }

    fn promote_face(&mut self, face: InflightFace) {
        let id = face.id as usize;
        let (retired, publish) = {
            let Some(slot) = self.slots.get_mut(id) else {
                return;
            };
            if slot.generation != face.generation || !slot.cubes.accepts(face.generation) {
                return;
            }
            let targets_pending = slot.cubes.has_pending();
            let retired = slot.cubes.land_face(face.face);
            let publish = !targets_pending || !slot.cubes.has_pending();
            (retired, publish)
        };
        if let Some(old) = retired {
            self.pending_retire.push(old);
        }
        if publish {
            self.write_header(id);
            self.buffer_dirty = true;
        }
    }

    /// Copy each current face into mip 0 and blit the chain, on `cmd` (the
    /// graphics buffer: a transfer-only queue cannot blit). A face on the
    /// sampled cube sets its landed bit now, so this frame's shader sees it.
    /// A face on a pending cube waits in [`Self::inflight_faces`] until
    /// `done_at` has signaled, mips included. Returns the staging buffer,
    /// retired by the caller at `done_at`.
    unsafe fn record_faces(
        &mut self,
        ctx: &TransferCtx<'_>,
    ) -> Option<(vk::Buffer, vk::DeviceMemory)> {
        let device = ctx.device;
        let cmd = ctx.graphics_cmd;
        let done_at = ctx.done_at;
        let faces = std::mem::take(&mut self.pending_faces);
        if faces.is_empty() {
            return None;
        }
        let mut packed: Vec<u8> = Vec::new();
        struct Job {
            id: usize,
            face: u32,
            offset: u64,
        }
        let mut jobs: Vec<Job> = Vec::new();
        for face in faces {
            let id = face.id as usize;
            let Some(slot) = self.slots.get(id) else {
                continue;
            };
            if slot.generation != face.generation || face.face >= 6 {
                continue;
            }
            let Some(tracked) = slot.cubes.upload() else {
                continue;
            };
            if tracked.generation != face.generation || !tracked.cube.cleared {
                continue;
            }
            let expect = tracked.cube.size as usize * tracked.cube.size as usize * 4;
            if face.bytes.len() != expect {
                continue;
            }
            let offset = packed.len() as u64;
            packed.extend_from_slice(&face.bytes);
            jobs.push(Job {
                id,
                face: face.face,
                offset,
            });
        }
        if jobs.is_empty() {
            return None;
        }
        let (staging, staging_mem) =
            create_filled_staging(device, &ctx.memory_props(), &packed, "far-map staging");
        for job in &jobs {
            let (image, mips, size) = {
                let cube = &self.slots[job.id]
                    .cubes
                    .upload()
                    .expect("face job has a cube")
                    .cube;
                (cube.image, cube.mips, cube.size)
            };
            unsafe {
                blit_face(
                    device, cmd, image, mips, size, job.face, staging, job.offset,
                )
            };
            let id = job.id;
            if self.slots[id].cubes.has_pending() {
                // The pending image is not sampled yet. Land the bit only
                // once this frame, mips included, has signaled.
                self.inflight_faces.push(InflightFace {
                    id: id as u8,
                    face: job.face,
                    generation: self.slots[id].generation,
                    done_at,
                });
            } else {
                let retired = self.slots[id].cubes.land_face(job.face);
                if let Some(old) = retired {
                    self.pending_retire.push(old);
                }
                self.write_header(id);
                self.buffer_dirty = true;
            }
        }
        Some((staging, staging_mem))
    }

    unsafe fn upload_buffer(&mut self, ctx: &mut TransferCtx<'_>) -> Option<TimelineValue> {
        self.buffer_dirty = false;
        let device = ctx.device;
        let graphics_cmd = ctx.graphics_cmd;
        let graphics_family = ctx.graphics_family;
        let done_at = ctx.done_at;
        let bytes: &[u8] = bytemuck::cast_slice(&self.words);
        let (staging, staging_mem) =
            create_filled_staging(device, &ctx.memory_props(), bytes, "far-map staging");
        let separate = ctx.lane.is_separate_queue();
        let qfot = ctx.lane.needs_ownership_transfer();
        let buffer = self.buffer;
        let size = bytes.len() as u64;
        let reads = vk::AccessFlags2::SHADER_STORAGE_READ;

        let extra_wait = if qfot && self.buffer_on_graphics {
            // Release the datum earlier frames read so the lane can acquire it.
            let release = [vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .src_queue_family_index(graphics_family)
                .dst_queue_family_index(ctx.lane.family())
                .buffer(buffer)
                .size(size)];
            Some(unsafe {
                self.retire.submit_release(
                    device,
                    ctx.graphics_queue,
                    done_at,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                )
            })
        } else if separate && self.buffer_on_graphics {
            Some((
                ctx.graphics_timeline.semaphore(),
                ctx.last_render_value,
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
            ))
        } else {
            None
        };

        let lane = &mut *ctx.lane;
        let lane_batch = separate.then(|| unsafe { lane.begin(device) });
        let record_cmd = lane_batch.as_ref().map_or(graphics_cmd, |b| b.cmd());
        unsafe {
            if !separate && self.buffer_on_graphics {
                let to_copy = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
                    .src_access_mask(reads)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .buffer(buffer)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_copy),
                );
            } else if qfot && self.buffer_on_graphics {
                let acquire = [vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_queue_family_index(graphics_family)
                    .dst_queue_family_index(lane.family())
                    .buffer(buffer)
                    .size(size)];
                device.cmd_pipeline_barrier2(
                    record_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                );
            }
            device.cmd_copy_buffer(
                record_cmd,
                staging,
                buffer,
                &[vk::BufferCopy::default().size(size)],
            );
        }

        let arrived = if let Some(lane_batch) = lane_batch {
            if qfot {
                let release = [copy_barrier(
                    buffer,
                    0,
                    size,
                    DATUM_CONSUMER,
                    CopyBarrier::Release {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                unsafe {
                    device.cmd_pipeline_barrier2(
                        record_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&release),
                    );
                }
            }
            let value = unsafe { lane.submit_after(device, lane_batch, extra_wait) };
            if qfot {
                let acquire = [copy_barrier(
                    buffer,
                    0,
                    size,
                    DATUM_CONSUMER,
                    CopyBarrier::Acquire {
                        src_family: lane.family(),
                        dst_family: graphics_family,
                    },
                )];
                unsafe {
                    device.cmd_pipeline_barrier2(
                        graphics_cmd,
                        &vk::DependencyInfo::default().buffer_memory_barriers(&acquire),
                    );
                }
            }
            self.retire.retire_on_lane(value, (staging, staging_mem));
            Some(value)
        } else {
            let to_shader = [copy_barrier(
                buffer,
                0,
                size,
                DATUM_CONSUMER,
                CopyBarrier::Draw,
            )];
            unsafe {
                device.cmd_pipeline_barrier2(
                    graphics_cmd,
                    &vk::DependencyInfo::default().buffer_memory_barriers(&to_shader),
                );
            }
            self.retire
                .retire_on_graphics(done_at, (staging, staging_mem));
            None
        };
        self.buffer_on_graphics = true;
        arrived
    }

    pub unsafe fn collect(&mut self, device: &ash::Device, current: TimelineValue) {
        unsafe {
            self.retire.collect(device, current);
            self.image_retire
                .collect(current, |cube| cube.destroy(device));
        }
    }

    pub unsafe fn collect_transfer(&mut self, device: &ash::Device, current: TimelineValue) {
        unsafe { self.retire.collect_transfer(device, current) };
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.retire.destroy(device);
            self.image_retire.collect_all(|cube| cube.destroy(device));
            for cube in self.pending_retire.drain(..) {
                cube.destroy(device);
            }
            self.dummy.destroy(device);
            for slot in &mut self.slots {
                let retired = slot.cubes.clear();
                if let Some(cube) = retired.active {
                    cube.destroy(device);
                }
                if let Some(cube) = retired.pending {
                    cube.destroy(device);
                }
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn image_barrier(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    range: vk::ImageSubresourceRange,
) {
    let barrier = [vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(range)];
    unsafe {
        device.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().image_memory_barriers(&barrier),
        );
    }
}

/// Cube of `size`² faces, full mip chain, still `UNDEFINED`. `Err` only when
/// the image memory cannot be allocated; the image is destroyed on that path.
unsafe fn create_cube(
    device: &ash::Device,
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    size: u32,
) -> Result<GpuCube, FarMapError> {
    let mips = size.trailing_zeros() + 1;
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .extent(vk::Extent3D {
            width: size,
            height: size,
            depth: 1,
        })
        .mip_levels(mips)
        .array_layers(6)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe {
        device
            .create_image(&info, None)
            .expect("create far-map albedo cube")
    };
    let memory = match allocate_and_bind_image(device, memory_props, image, "far-map albedo cube") {
        Ok(memory) => memory,
        Err(_) => {
            unsafe { device.destroy_image(image, None) };
            return Err(FarMapError::OutOfMemory);
        }
    };
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::CUBE)
        .format(vk::Format::R8G8B8A8_SRGB)
        .subresource_range(color_range(0, mips, 0, 6));
    let view = unsafe {
        device
            .create_image_view(&view_info, None)
            .expect("create far-map albedo cube view")
    };
    Ok(GpuCube {
        image,
        memory,
        view,
        mips,
        size,
        cleared: false,
    })
}

/// First use of a cube: `UNDEFINED` → cleared black → `SHADER_READ`.
unsafe fn prepare_cube(device: &ash::Device, cmd: vk::CommandBuffer, cube: &mut GpuCube) {
    if cube.cleared {
        return;
    }
    unsafe { clear_cube(device, cmd, cube) };
    cube.cleared = true;
}

/// Every mip of every face: `UNDEFINED` → cleared black → `SHADER_READ`.
/// A later face upload may sample a face it did not write, so none may stay
/// `UNDEFINED`.
unsafe fn clear_cube(device: &ash::Device, cmd: vk::CommandBuffer, cube: &GpuCube) {
    let range = color_range(0, cube.mips, 0, 6);
    unsafe {
        image_barrier(
            device,
            cmd,
            cube.image,
            vk::PipelineStageFlags2::NONE,
            vk::AccessFlags2::NONE,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            range,
        );
        device.cmd_clear_color_image(
            cmd,
            cube.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &vk::ClearColorValue { float32: [0.0; 4] },
            &[range],
        );
        image_barrier(
            device,
            cmd,
            cube.image,
            vk::PipelineStageFlags2::CLEAR,
            vk::AccessFlags2::TRANSFER_WRITE,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            range,
        );
    }
}

/// Copy mip 0 of `face` from `staging` and blit the rest of the chain.
/// The cube is `SHADER_READ` on entry (after [`clear_cube`]) and on return.
#[allow(clippy::too_many_arguments)]
unsafe fn blit_face(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
    size: u32,
    face: u32,
    staging: vk::Buffer,
    offset: u64,
) {
    let mip0 = color_range(0, 1, face, 1);
    unsafe {
        image_barrier(
            device,
            cmd,
            image,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::PipelineStageFlags2::COPY,
            vk::AccessFlags2::TRANSFER_WRITE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip0,
        );
        let region = vk::BufferImageCopy::default()
            .buffer_offset(offset)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: face,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: size,
                height: size,
                depth: 1,
            });
        device.cmd_copy_buffer_to_image(
            cmd,
            staging,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[region],
        );
        let mut src_w = size;
        for mip in 0..mips.saturating_sub(1) {
            let dst_w = (src_w / 2).max(1);
            let src_range = color_range(mip, 1, face, 1);
            let dst_range = color_range(mip + 1, 1, face, 1);
            image_barrier(
                device,
                cmd,
                image,
                vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                src_range,
            );
            image_barrier(
                device,
                cmd,
                image,
                FAR_MAP_CONSUMER_STAGES,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                dst_range,
            );
            let blit = vk::ImageBlit::default()
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip,
                    base_array_layer: face,
                    layer_count: 1,
                })
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: src_w as i32,
                        y: src_w as i32,
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: mip + 1,
                    base_array_layer: face,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_w as i32,
                        y: dst_w as i32,
                        z: 1,
                    },
                ]);
            device.cmd_blit_image(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[blit],
                vk::Filter::LINEAR,
            );
            image_barrier(
                device,
                cmd,
                image,
                vk::PipelineStageFlags2::BLIT,
                vk::AccessFlags2::TRANSFER_READ,
                FAR_MAP_CONSUMER_STAGES,
                vk::AccessFlags2::SHADER_SAMPLED_READ,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_range,
            );
            src_w = dst_w;
        }
        let last = color_range(mips.saturating_sub(1), 1, face, 1);
        image_barrier(
            device,
            cmd,
            image,
            vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::BLIT,
            vk::AccessFlags2::TRANSFER_WRITE,
            FAR_MAP_CONSUMER_STAGES,
            vk::AccessFlags2::SHADER_SAMPLED_READ,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            last,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{DATUM_CONSUMER, FACE_MASK, Retired, SlotCubes};
    use ash::vk;

    #[test]
    fn datum_copy_consumer_is_fragment_storage_reads() {
        assert_eq!(
            DATUM_CONSUMER,
            (
                vk::PipelineStageFlags2::FRAGMENT_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            )
        );
    }

    fn complete(id: u32, size: u32, generation: u32) -> SlotCubes<u32> {
        let mut cubes = SlotCubes::new();
        assert_eq!(
            cubes.install(size, Some(id), generation),
            Retired::default()
        );
        for face in 0..6 {
            assert_eq!(cubes.land_face(face), None);
        }
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert!(!cubes.has_pending());
        cubes
    }

    #[test]
    fn first_install_shows_each_face_as_it_lands() {
        let mut cubes = SlotCubes::new();
        assert_eq!(cubes.install(64, Some(7), 1), Retired::default());
        assert_eq!(cubes.sampled_size(), 64);
        assert_eq!(cubes.sampled_landed(), 0);
        let mut mask = 0;
        for face in 0..6 {
            assert_eq!(cubes.land_face(face), None);
            mask |= 1 << face;
            assert_eq!(cubes.sampled_landed(), mask);
            assert_eq!(cubes.sampled_size(), 64);
            assert!(!cubes.has_pending());
        }
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
    }

    #[test]
    fn replacement_keeps_the_old_mask_until_six_new_faces_land() {
        let mut cubes = complete(1, 256, 1);
        assert_eq!(cubes.install(1024, Some(2), 2), Retired::default());
        assert!(cubes.has_pending());
        assert_eq!(cubes.sampled_size(), 256);
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert_eq!(cubes.upload().map(|c| c.size), Some(1024));
        for face in 0..5 {
            assert_eq!(cubes.land_face(face), None);
            assert_eq!(cubes.sampled_size(), 256);
            assert_eq!(cubes.sampled_landed(), FACE_MASK);
            assert!(cubes.has_pending());
        }
        assert_eq!(cubes.land_face(5), Some(1));
        assert!(!cubes.has_pending());
        assert_eq!(cubes.sampled_size(), 1024);
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert_eq!(cubes.active_cube().copied(), Some(2));
        // A repeat of the last face stays on the cube now on screen.
        assert_eq!(cubes.land_face(5), None);
        assert_eq!(cubes.sampled_size(), 1024);
    }

    #[test]
    fn replacement_of_an_incomplete_cube_is_immediate() {
        let mut cubes = SlotCubes::new();
        assert_eq!(cubes.install(256, Some(1), 1), Retired::default());
        assert_eq!(cubes.land_face(0), None);
        assert_eq!(cubes.land_face(3), None);
        assert_eq!(cubes.sampled_landed(), (1 << 0) | (1 << 3));
        assert_eq!(
            cubes.install(1024, Some(2), 2),
            Retired {
                active: Some(1),
                pending: None,
            }
        );
        assert!(!cubes.has_pending());
        assert_eq!(cubes.sampled_size(), 1024);
        assert_eq!(cubes.sampled_landed(), 0);
        assert_eq!(cubes.land_face(2), None);
        assert_eq!(cubes.sampled_landed(), 1 << 2);
        assert_eq!(cubes.sampled_size(), 1024);
    }

    #[test]
    fn clear_drops_the_sampled_cube_and_the_pending_cube() {
        let mut cubes = complete(1, 256, 1);
        assert_eq!(cubes.install(1024, Some(2), 2), Retired::default());
        assert_eq!(cubes.land_face(0), None);
        assert_eq!(
            cubes.clear(),
            Retired {
                active: Some(1),
                pending: Some(2),
            }
        );
        assert_eq!(cubes.sampled_size(), 0);
        assert_eq!(cubes.sampled_landed(), 0);
        assert!(!cubes.has_pending());
        assert!(cubes.upload().is_none());
        assert_eq!(cubes.land_face(0), None);
    }

    #[test]
    fn albedo_size_zero_drops_both_cubes_immediately() {
        let mut cubes = complete(1, 256, 1);
        assert_eq!(cubes.install(1024, Some(2), 2), Retired::default());
        assert_eq!(cubes.land_face(4), None);
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert_eq!(
            cubes.install(0, None, 3),
            Retired {
                active: Some(1),
                pending: Some(2),
            }
        );
        assert_eq!(cubes.sampled_size(), 0);
        assert_eq!(cubes.sampled_landed(), 0);
        assert!(!cubes.has_pending());
        assert!(cubes.upload().is_none());
    }

    #[test]
    fn a_second_replacement_discards_the_pending_cube() {
        let mut cubes = complete(1, 256, 1);
        assert_eq!(cubes.install(512, Some(2), 2), Retired::default());
        assert_eq!(cubes.land_face(0), None);
        assert_eq!(cubes.land_face(1), None);
        assert_eq!(
            cubes.install(1024, Some(3), 3),
            Retired {
                active: None,
                pending: Some(2),
            }
        );
        assert_eq!(cubes.sampled_size(), 256);
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert!(cubes.accepts(3));
        assert!(!cubes.accepts(2));
        for face in 0..5 {
            assert_eq!(cubes.land_face(face), None);
            assert_eq!(cubes.sampled_size(), 256);
        }
        assert_eq!(cubes.land_face(5), Some(1));
        assert_eq!(cubes.sampled_size(), 1024);
        assert_eq!(cubes.active_cube().copied(), Some(3));
    }

    #[test]
    fn failed_allocation_keeps_a_complete_cube() {
        let mut cubes = complete(1, 256, 1);
        assert_eq!(cubes.install(512, Some(2), 2), Retired::default());
        assert_eq!(
            cubes.install(1024, None, 3),
            Retired {
                active: None,
                pending: Some(2),
            }
        );
        assert!(!cubes.has_pending());
        assert_eq!(cubes.sampled_size(), 256);
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
        assert_eq!(cubes.active_cube().copied(), Some(1));
        assert!(cubes.accepts(1));
        assert!(!cubes.accepts(3));
        // Faces for the failed generation must not move the sampled mask.
        assert_eq!(cubes.sampled_landed(), FACE_MASK);
    }

    #[test]
    fn failed_allocation_of_an_incomplete_cube_drops_to_flat() {
        let mut cubes = SlotCubes::new();
        assert_eq!(cubes.install(256, Some(1), 1), Retired::default());
        assert_eq!(cubes.land_face(0), None);
        assert_eq!(
            cubes.install(1024, None, 2),
            Retired {
                active: Some(1),
                pending: None,
            }
        );
        assert_eq!(cubes.sampled_size(), 0);
        assert_eq!(cubes.sampled_landed(), 0);
        assert!(cubes.upload().is_none());
        assert!(!cubes.accepts(2));
    }
}
