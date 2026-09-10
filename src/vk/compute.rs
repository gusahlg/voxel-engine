//! General GPU compute-job facility for deterministic world generation
//! (and later meshing). The game owns the shaders; the engine owns pipelines,
//! staging, submission, and readback.
//!
//! # Tiers
//! Preference order, matching the transfer lane:
//! 1. **Dedicated family** — a queue family with `COMPUTE` and without
//!    `GRAPHICS` (async compute). Jobs are recorded and submitted as they
//!    arrive, capped at 32 in-flight jobs and by outstanding readback bytes.
//! 2. **Second queue, same family** — another queue index in the graphics
//!    family. Same submit-as-they-arrive behaviour; no queue-family ownership
//!    transfer.
//! 3. **Same-queue fallback** — jobs record into the frame command buffer
//!    *before* the scene, at most 200 µs of estimated GPU time per frame.
//!
//! An empty queue records nothing, so an idle frame is untouched.
//!
//! # Budget
//! On the same-queue fallback, each kind starts at 200 µs estimated GPU time.
//! Timestamps update that estimate (EMA), clamped to 50..=2000 µs. A pure
//! policy function decides how many queued jobs fit; the first job always
//! runs so a single expensive dispatch cannot stall the queue.
//!
//! # Determinism
//! Game shaders must be **32-bit integer only** (wrapping arithmetic), use
//! **no shared memory**, and **guard out-of-range indices**. The engine does
//! not prove this at pipeline creation. Bit-identical CPU/GPU results are
//! the game's contract; [`crate::Engine::run_compute_blocking`] exists for
//! that parity test.

use std::collections::VecDeque;
use std::ffi::CString;
use std::io::Cursor;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ash::khr;
use ash::vk;

use super::alloc::try_find_memory_type;
use super::buffers::HOST_COHERENT;
use super::mesh_staging::{StagingRegion, StagingRing, Stamp, sysmem_staging_type};
use super::pass;
use super::timeline::{Timeline, TimelineValue};
use super::transfer::{BatchRing, LaneRecording, Tier};

/// In-flight GPU job cap on the dedicated / second-queue tiers.
pub(crate) const MAX_IN_FLIGHT: usize = 32;
/// Same-queue fallback: pack jobs into this many microseconds of estimated
/// GPU time per frame.
pub(crate) const FALLBACK_FRAME_BUDGET_US: u32 = 200;
/// Initial per-kind GPU-time estimate (microseconds).
pub(crate) const BUDGET_START_US: u32 = 200;
pub(crate) const BUDGET_MIN_US: u32 = 50;
pub(crate) const BUDGET_MAX_US: u32 = 2000;

const MAX_PUSH_BYTES: u32 = 128;
const MAX_INPUTS: u32 = 2;
const QUERY_PAIRS: usize = 64;

/// Default host-cached input staging ring (16 MiB). One job for the first
/// user is ~1.18 MiB of input; override with `VOXEL_COMPUTE_INPUT_MB`.
pub const COMPUTE_INPUT_BYTES: u64 = 16 << 20;
/// Default host-cached readback ring (32 MiB). One job is ~524 KiB of
/// output; ~32 in-flight during a streaming burst. Override with
/// `VOXEL_COMPUTE_READBACK_MB`.
pub const COMPUTE_READBACK_BYTES: u64 = 32 << 20;

const INPUT_ENV: &str = "VOXEL_COMPUTE_INPUT_MB";
const READBACK_ENV: &str = "VOXEL_COMPUTE_READBACK_MB";

/// SPIR-V for [`crate::Engine::example_compute_desc`].
pub const EXAMPLE_COMPUTE_SPIRV: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/compute_example.comp.spv"));

/// Pipeline description for [`crate::Engine::register_compute`].
#[derive(Clone, Copy, Debug)]
pub struct ComputeDesc<'a> {
    /// SPIR-V module bytes (endian-independent; parsed with `ash::util::read_spv`).
    pub spirv: &'a [u8],
    /// SPIR-V entry point name (`"main"` for Slang-compiled modules).
    pub entry: &'a str,
    /// Push-constant size in bytes, `0..=128`, multiple of 4.
    pub push_bytes: u32,
    /// Read-only storage buffers at bindings `0..inputs` (`0..=2`). Binding
    /// `inputs` is the output storage buffer.
    pub inputs: u32,
    /// Maximum output size a job of this kind may request.
    pub output_bytes_max: u32,
    /// Shader workgroup size (`[numthreads]`). Validated against device
    /// limits; dispatch counts are per job.
    pub workgroup: [u32; 3],
}

impl ComputeDesc<'static> {
    /// Integer-hash example used by the roundtrip test.
    pub fn example() -> Self {
        Self {
            spirv: EXAMPLE_COMPUTE_SPIRV,
            entry: "main",
            push_bytes: 8,
            inputs: 1,
            output_bytes_max: 4096 * 4,
            workgroup: [64, 1, 1],
        }
    }
}

/// Small `Copy` handle returned by [`crate::Engine::register_compute`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComputeKind(u32);

impl ComputeKind {
    pub fn raw(self) -> u32 {
        self.0
    }
}

/// Monotonic job identifier returned by [`ComputeQueue::submit`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(u64);

impl JobId {
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// One compute dispatch. `inputs[i]` is the storage buffer at binding `i`.
pub struct ComputeJob<'a> {
    pub kind: ComputeKind,
    pub push: &'a [u8],
    pub inputs: [Option<ComputeInput>; 2],
    pub output_bytes: u32,
    pub dispatch: [u32; 3],
}

/// Errors from compute registration and submit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// No compute support (or the compute rings failed to allocate). The
    /// game should fall back to CPU.
    NoCompute,
    /// Invalid [`ComputeDesc`] or job (push size, bindings, SPIR-V, kind).
    Invalid(&'static str),
    /// Pipeline or layout creation failed.
    Pipeline,
    /// `output_bytes` exceeds the kind's `output_bytes_max`, or is zero.
    OutputTooLarge,
    /// A required input was `None` or its region is no longer valid.
    InputGone,
    /// Input or readback ring is full. Retry later.
    Busy,
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCompute => write!(f, "device does not support compute jobs"),
            Self::Invalid(msg) => write!(f, "invalid compute desc: {msg}"),
            Self::Pipeline => write!(f, "compute pipeline creation failed"),
            Self::OutputTooLarge => write!(f, "compute job output exceeds kind maximum"),
            Self::InputGone => write!(f, "compute job input region is missing"),
            Self::Busy => write!(f, "compute staging or readback ring is full"),
        }
    }
}

impl std::error::Error for EngineError {}

pub(crate) struct ComputeDescOwned {
    pub spirv: Box<[u8]>,
    pub entry: String,
    pub push_bytes: u32,
    pub inputs: u32,
    pub output_bytes_max: u32,
    pub workgroup: [u32; 3],
}

impl ComputeDesc<'_> {
    pub(crate) fn owned(self) -> ComputeDescOwned {
        ComputeDescOwned {
            spirv: self.spirv.to_vec().into_boxed_slice(),
            entry: self.entry.to_owned(),
            push_bytes: self.push_bytes,
            inputs: self.inputs,
            output_bytes_max: self.output_bytes_max,
            workgroup: self.workgroup,
        }
    }
}

pub(crate) fn parse_mb(s: &str) -> Option<u64> {
    let mb: u64 = s.trim().parse().ok()?;
    Some(mb.saturating_mul(1 << 20))
}

fn ring_bytes(env: &str, default: u64) -> u64 {
    match std::env::var(env) {
        Ok(s) => parse_mb(&s).unwrap_or_else(|| {
            log::warn!("invalid {env}={s:?}; using {} MiB", default / (1 << 20));
            default
        }),
        Err(_) => default,
    }
}

pub(crate) fn compute_input_bytes() -> u64 {
    ring_bytes(INPUT_ENV, COMPUTE_INPUT_BYTES)
}

pub(crate) fn compute_readback_bytes() -> u64 {
    ring_bytes(READBACK_ENV, COMPUTE_READBACK_BYTES)
}

/// How many of `costs_us` (queue order) fit in `budget_us`.
///
/// Empty input → 0. The first job is always taken so a single expensive
/// dispatch cannot stall the queue; further jobs pack while the running
/// sum stays `<= budget_us`.
pub(crate) fn jobs_fitting_budget(budget_us: u32, costs_us: &[u32]) -> usize {
    if costs_us.is_empty() {
        return 0;
    }
    let mut used = 0u32;
    let mut n = 0usize;
    for &cost in costs_us {
        let cost = cost.max(1);
        if n == 0 {
            used = cost;
            n = 1;
            continue;
        }
        if used.saturating_add(cost) > budget_us {
            break;
        }
        used = used.saturating_add(cost);
        n += 1;
    }
    n
}

/// EMA of `prev` and `measured`, clamped to [`BUDGET_MIN_US`]..=[`BUDGET_MAX_US`].
pub(crate) fn adapt_estimate_us(prev: u32, measured: u32) -> u32 {
    let mixed = (u64::from(prev) + u64::from(measured)) / 2;
    (mixed as u32).clamp(BUDGET_MIN_US, BUDGET_MAX_US)
}

/// Queue-family pick for the compute lane. Pure (no Vulkan calls).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ComputeQueuePick {
    pub tier: Tier,
    pub family: u32,
    pub queue_index: u32,
    /// Graphics-family queues the device must request (1..=3).
    pub graphics_queues: u32,
}

/// `dedicated_compute` is `(family, queue_count)` for a COMPUTE && !GRAPHICS
/// family, if any. `transfer_family` is the family the transfer queue uses.
pub(crate) fn pick_compute_queue(
    graphics_family: u32,
    graphics_queue_count: u32,
    transfer_tier: Tier,
    transfer_family: u32,
    transfer_queue_count: u32,
    dedicated_compute: Option<(u32, u32)>,
) -> ComputeQueuePick {
    let gfx_for_transfer: u32 = if transfer_tier == Tier::SecondQueueSameFamily {
        2
    } else {
        1
    };

    if let Some((cf, cq)) = dedicated_compute
        && cf != graphics_family
    {
        if cf == transfer_family && transfer_tier == Tier::DedicatedFamily {
            if cq >= 2 && transfer_queue_count >= 2 {
                return ComputeQueuePick {
                    tier: Tier::DedicatedFamily,
                    family: cf,
                    queue_index: 1,
                    graphics_queues: gfx_for_transfer,
                };
            }
        } else {
            return ComputeQueuePick {
                tier: Tier::DedicatedFamily,
                family: cf,
                queue_index: 0,
                graphics_queues: gfx_for_transfer,
            };
        }
    }

    if graphics_queue_count > gfx_for_transfer {
        ComputeQueuePick {
            tier: Tier::SecondQueueSameFamily,
            family: graphics_family,
            queue_index: gfx_for_transfer,
            graphics_queues: gfx_for_transfer + 1,
        }
    } else {
        ComputeQueuePick {
            tier: Tier::SameQueueFallback,
            family: graphics_family,
            queue_index: 0,
            graphics_queues: gfx_for_transfer,
        }
    }
}

struct LaneResources {
    pool: vk::CommandPool,
    ring: BatchRing<vk::CommandBuffer>,
    timeline: Timeline,
}

/// Compute analogue of [`super::transfer::TransferLane`].
pub(crate) struct ComputeLane {
    tier: Tier,
    family: u32,
    queue: vk::Queue,
    resources: Option<LaneResources>,
}

impl ComputeLane {
    pub unsafe fn new(device: &ash::Device, family: u32, queue: vk::Queue, tier: Tier) -> Self {
        let resources = (tier != Tier::SameQueueFallback).then(|| unsafe {
            let pool_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create compute command pool");
            LaneResources {
                pool,
                ring: BatchRing::new(),
                timeline: Timeline::new(device),
            }
        });
        Self {
            tier,
            family,
            queue,
            resources,
        }
    }

    pub fn tier(&self) -> Tier {
        self.tier
    }

    pub fn family(&self) -> u32 {
        self.family
    }

    pub fn is_separate_queue(&self) -> bool {
        self.resources.is_some()
    }

    pub unsafe fn begin(&mut self, device: &ash::Device) -> LaneRecording {
        let res = self
            .resources
            .as_mut()
            .expect("ComputeLane::begin requires a separate queue");
        let recycled = res.ring.take_free().or_else(|| {
            let completed = unsafe { res.timeline.counter(device) };
            res.ring.reclaim(completed);
            res.ring.take_free()
        });
        let cmd = match recycled {
            Some(cmd) => cmd,
            None => {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(res.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
                    .expect("Failed to allocate compute command buffer")[0];
                res.ring.push_recording(cmd);
                log::debug!("compute lane grew to {} command buffers", res.ring.len());
                cmd
            }
        };
        unsafe {
            device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .expect("compute command buffer reset failed");
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(cmd, &begin)
                .expect("begin compute command buffer failed");
        }
        LaneRecording::from_cmd(cmd)
    }

    pub unsafe fn submit(&mut self, device: &ash::Device, batch: LaneRecording) -> TimelineValue {
        let res = self
            .resources
            .as_mut()
            .expect("ComputeLane::submit requires a separate queue");
        let cmd = batch.cmd();
        unsafe {
            device
                .end_command_buffer(cmd)
                .expect("end compute command buffer failed");
        }
        let rs = res.timeline.begin_render(cmd);
        let value = rs.value();
        let completion = unsafe { rs.submit(device, self.queue, &res.timeline, None) };
        debug_assert_eq!(completion.value(), value);
        res.ring.submitted(cmd, value);
        value
    }

    pub fn semaphore_opt(&self) -> Option<vk::Semaphore> {
        self.resources.as_ref().map(|r| r.timeline.semaphore())
    }

    #[allow(dead_code)]
    pub fn semaphore(&self) -> vk::Semaphore {
        self.semaphore_opt()
            .expect("ComputeLane::semaphore requires a separate queue")
    }

    #[allow(dead_code)]
    pub unsafe fn wait(&self, device: &ash::Device, value: TimelineValue) {
        let res = self
            .resources
            .as_ref()
            .expect("ComputeLane::wait requires a separate queue");
        unsafe { res.timeline.wait(device, value) };
    }

    #[allow(dead_code)]
    pub unsafe fn counter(&self, device: &ash::Device) -> Option<TimelineValue> {
        let res = self.resources.as_ref()?;
        Some(unsafe { res.timeline.counter(device) })
    }

    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        if let Some(res) = self.resources.take() {
            unsafe {
                res.timeline.destroy(device);
                device.destroy_command_pool(res.pool, None);
            }
        }
    }
}

struct HostRing {
    ring: StagingRing,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: Option<NonNull<u8>>,
    /// True when the mapping is HOST_CACHED | DEVICE_LOCAL (cheap GPU write).
    is_direct: bool,
    destroyed: AtomicBool,
    _pin: Option<Box<[u8]>>,
}

// SAFETY: persistent mapping is process-wide; disjoint regions are written by
// the workers that acquired them, and GPU reads start only after submit.
unsafe impl Send for HostRing {}
unsafe impl Sync for HostRing {}

impl HostRing {
    fn disabled() -> Self {
        Self {
            ring: StagingRing::new(0, 16),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            mapped: None,
            is_direct: false,
            destroyed: AtomicBool::new(true),
            _pin: None,
        }
    }

    unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        prefer_direct: bool,
    ) -> Arc<Self> {
        if size == 0 {
            return Arc::new(Self::disabled());
        }
        let align = align.max(16);
        let size = size.next_multiple_of(align).max(align);
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe {
            device
                .create_buffer(&info, None)
                .expect("create compute host ring buffer")
        };
        let req = unsafe { device.get_buffer_memory_requirements(buffer) };
        let type_index = if prefer_direct {
            cheap_direct_type(&memory_props, req.memory_type_bits)
                .or_else(|| sysmem_staging_type(&memory_props, req.memory_type_bits))
        } else {
            sysmem_staging_type(&memory_props, req.memory_type_bits)
        }
        .expect("no HOST_VISIBLE | HOST_COHERENT memory type for compute ring");
        let memory = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(type_index),
                    None,
                )
                .expect("allocate compute host ring memory")
        };
        unsafe {
            device
                .bind_buffer_memory(buffer, memory, 0)
                .expect("bind compute host ring memory");
        }
        let mapped = unsafe {
            device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .expect("map compute host ring") as *mut u8
        };
        let flags = memory_props.memory_types[type_index as usize].property_flags;
        let is_direct = flags.contains(
            HOST_COHERENT
                | vk::MemoryPropertyFlags::HOST_CACHED
                | vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        log::info!(
            "compute ring: {} MiB usage={usage:?} type {type_index} direct={is_direct} ({flags:?})",
            size / (1024 * 1024)
        );
        Arc::new(Self {
            ring: StagingRing::new(size, align),
            buffer,
            memory,
            mapped: NonNull::new(mapped),
            is_direct,
            destroyed: AtomicBool::new(false),
            _pin: None,
        })
    }

    fn acquire(self: &Arc<Self>, bytes: usize) -> Option<(StagingRegion, *mut u8)> {
        let mapped = self.mapped?;
        let region = self.ring.acquire(bytes)?;
        Some((region, unsafe {
            mapped.as_ptr().add(region.offset as usize)
        }))
    }

    fn copy_out(&self, region: StagingRegion, bytes: u32) -> Box<[u8]> {
        let mapped = self.mapped.expect("compute readback is mapped");
        let n = bytes as usize;
        let src =
            unsafe { std::slice::from_raw_parts(mapped.as_ptr().add(region.offset as usize), n) };
        src.to_vec().into_boxed_slice()
    }

    fn reclaim(&self, render: u64, transfer: Option<u64>) {
        self.ring.reclaim(render, transfer);
    }

    unsafe fn destroy(&self, device: &ash::Device) {
        if self.destroyed.swap(true, Ordering::AcqRel) {
            return;
        }
        if self.buffer != vk::Buffer::null() {
            unsafe {
                device.destroy_buffer(self.buffer, None);
                device.free_memory(self.memory, None);
            }
        }
    }
}

/// HOST_VISIBLE | HOST_COHERENT | HOST_CACHED | DEVICE_LOCAL — UMA-class
/// memory where the shader can write the readback ring directly.
fn cheap_direct_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
) -> Option<u32> {
    let want = HOST_COHERENT
        | vk::MemoryPropertyFlags::HOST_CACHED
        | vk::MemoryPropertyFlags::DEVICE_LOCAL;
    try_find_memory_type(memory_props, type_filter, want)
}

fn device_local_type(
    memory_props: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
) -> Option<u32> {
    let n = memory_props.memory_type_count;
    (0..n)
        .find(|&i| {
            if type_filter & (1 << i) == 0 {
                return false;
            }
            let flags = memory_props.memory_types[i as usize].property_flags;
            flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
                && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
        })
        .or_else(|| {
            try_find_memory_type(
                memory_props,
                type_filter,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
        })
}

/// Cheap `Clone` handle workers use to acquire compute input regions.
#[derive(Clone)]
pub struct ComputeStager {
    ring: Arc<HostRing>,
}

impl ComputeStager {
    /// Non-blocking acquire of a host-mapped region. `None` if the pool is
    /// exhausted (the job retries later).
    pub fn acquire(&self, bytes: usize) -> Option<ComputeInput> {
        let (region, ptr) = HostRing::acquire(&self.ring, bytes)?;
        Some(ComputeInput {
            ring: Arc::clone(&self.ring),
            region,
            requested: bytes,
            ptr,
            consumed: false,
        })
    }

    pub fn capacity_bytes(&self) -> u64 {
        self.ring.ring.capacity()
    }
}

/// A mapped compute-input region. Write with [`Self::bytes`] or [`Self::write`].
/// [`Drop`] returns it to the ring without a GPU wait.
#[must_use = "dropping a ComputeInput releases the region; pass it to ComputeQueue::submit to keep the bytes"]
pub struct ComputeInput {
    ring: Arc<HostRing>,
    region: StagingRegion,
    requested: usize,
    ptr: *mut u8,
    consumed: bool,
}

// SAFETY: the mapped range is exclusively owned by this region until release.
unsafe impl Send for ComputeInput {}

impl ComputeInput {
    /// Mapped bytes of the acquire request (not the aligned reservation).
    pub fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.requested) }
    }

    /// Copy `data` into the region. Returns `false` if it would not fit.
    pub fn write(&mut self, data: &[u8]) -> bool {
        let dst = self.bytes();
        if data.len() > dst.len() {
            return false;
        }
        dst[..data.len()].copy_from_slice(data);
        true
    }

    pub fn release(self) {
        drop(self);
    }

    fn into_lease(mut self) -> InputLease {
        self.consumed = true;
        InputLease {
            ring: Arc::clone(&self.ring),
            region: self.region,
        }
    }
}

impl Drop for ComputeInput {
    fn drop(&mut self) {
        if !self.consumed {
            self.ring.ring.release(self.region);
        }
    }
}

struct InputLease {
    ring: Arc<HostRing>,
    region: StagingRegion,
}

impl InputLease {
    fn stamp(self, stamp: Stamp) {
        self.ring.ring.stamp(self.region, stamp);
        std::mem::forget(self);
    }
}

impl Drop for InputLease {
    fn drop(&mut self) {
        self.ring.ring.release(self.region);
    }
}

/// Cheap `Clone` handle: submit from any thread.
#[derive(Clone)]
pub struct ComputeQueue {
    inner: Arc<ComputeRuntime>,
}

impl ComputeQueue {
    /// Queue `job` for the render thread. Returns [`EngineError::Busy`] when
    /// the readback ring cannot hold the output (caller retries later).
    pub fn submit(&self, job: ComputeJob<'_>) -> Result<JobId, EngineError> {
        self.inner.submit_job(job)
    }
}

struct KindState {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    set_layout: vk::DescriptorSetLayout,
    push_bytes: u32,
    inputs: u32,
    output_bytes_max: u32,
    estimate_us: AtomicU32,
}

struct KindMeta {
    push_bytes: u32,
    inputs: u32,
    output_bytes_max: u32,
}

pub(crate) struct QueuedJob {
    id: JobId,
    kind: ComputeKind,
    push: Box<[u8]>,
    inputs: [Option<InputLease>; 2],
    output: StagingRegion,
    output_bytes: u32,
    dispatch: [u32; 3],
}

struct InFlight {
    id: JobId,
    kind: ComputeKind,
    value: TimelineValue,
    on_compute_lane: bool,
    gpu_done: bool,
    output: StagingRegion,
    output_bytes: u32,
    inputs: [Option<InputLease>; 2],
    timestamp_pair: Option<u32>,
}

struct Scratch {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

pub(crate) struct ComputeLimits {
    max_work_group_invocations: u32,
    max_work_group_size: [u32; 3],
    max_work_group_count: [u32; 3],
    max_push: u32,
}

/// Shared between main (poll / submit / stager) and the render thread.
pub(crate) struct ComputeRuntime {
    enabled: bool,
    device: ash::Device,
    graphics_timeline: vk::Semaphore,
    compute_timeline: Option<vk::Semaphore>,
    input: Arc<HostRing>,
    readback: Arc<HostRing>,
    scratch: Option<Scratch>,
    /// Shader writes the mapped readback buffer directly (UMA/cached BAR).
    direct: bool,
    kinds: Mutex<Vec<KindState>>,
    queue: Mutex<VecDeque<QueuedJob>>,
    inflight: Mutex<VecDeque<InFlight>>,
    /// Already copied out, waiting for the next [`ComputeRuntime::poll`].
    stash: Mutex<VecDeque<(JobId, Box<[u8]>)>>,
    next_id: AtomicU64,
    pending: AtomicUsize,
    gpu_inflight: AtomicUsize,
    limits: ComputeLimits,
    query_pool: vk::QueryPool,
    query_free: Mutex<Vec<u32>>,
    timestamp_period_ns: f32,
    host_query_reset: bool,
}

impl ComputeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        align: u64,
        graphics_timeline: vk::Semaphore,
        compute_timeline: Option<vk::Semaphore>,
        timestamps: bool,
        timestamp_period_ns: f32,
        host_query_reset: bool,
        limits: ComputeLimits,
    ) -> Arc<Self> {
        let input_size = compute_input_bytes();
        let readback_size = compute_readback_bytes();
        let memory_props = unsafe { instance.get_physical_device_memory_properties(physical) };

        let input = unsafe {
            HostRing::new(
                instance,
                device,
                physical,
                input_size,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                align,
                false,
            )
        };

        let readback = unsafe {
            HostRing::new(
                instance,
                device,
                physical,
                readback_size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                align,
                true,
            )
        };
        let direct = readback.is_direct;

        let scratch = if direct || readback.buffer == vk::Buffer::null() {
            None
        } else {
            let size = readback.ring.capacity().max(align);
            let info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = unsafe {
                device
                    .create_buffer(&info, None)
                    .expect("create compute scratch buffer")
            };
            let req = unsafe { device.get_buffer_memory_requirements(buffer) };
            let type_index = device_local_type(&memory_props, req.memory_type_bits)
                .expect("no DEVICE_LOCAL memory type for compute scratch");
            let memory = unsafe {
                device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo::default()
                            .allocation_size(req.size)
                            .memory_type_index(type_index),
                        None,
                    )
                    .expect("allocate compute scratch")
            };
            unsafe {
                device
                    .bind_buffer_memory(buffer, memory, 0)
                    .expect("bind compute scratch");
            }
            log::info!(
                "compute scratch: {} MiB device-local (shader write → copy to readback)",
                size / (1024 * 1024)
            );
            Some(Scratch { buffer, memory })
        };

        let (query_pool, query_free) = if timestamps {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count((QUERY_PAIRS * 2) as u32);
            let pool = unsafe {
                device
                    .create_query_pool(&info, None)
                    .expect("create compute timestamp pool")
            };
            let free: Vec<u32> = (0..QUERY_PAIRS as u32).collect();
            (pool, free)
        } else {
            (vk::QueryPool::null(), Vec::new())
        };

        let enabled = input.buffer != vk::Buffer::null() && readback.buffer != vk::Buffer::null();
        if !enabled {
            log::warn!("compute: rings disabled; register_compute will return NoCompute");
        } else {
            log::info!(
                "compute: direct_write={direct} input={} MiB readback={} MiB",
                input.ring.capacity() / (1024 * 1024),
                readback.ring.capacity() / (1024 * 1024),
            );
        }

        Arc::new(Self {
            enabled,
            device: device.clone(),
            graphics_timeline,
            compute_timeline,
            input,
            readback,
            scratch,
            direct,
            kinds: Mutex::new(Vec::new()),
            queue: Mutex::new(VecDeque::new()),
            inflight: Mutex::new(VecDeque::new()),
            stash: Mutex::new(VecDeque::new()),
            next_id: AtomicU64::new(1),
            pending: AtomicUsize::new(0),
            gpu_inflight: AtomicUsize::new(0),
            limits,
            query_pool,
            query_free: Mutex::new(query_free),
            timestamp_period_ns,
            host_query_reset,
        })
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn stager(self: &Arc<Self>) -> ComputeStager {
        ComputeStager {
            ring: Arc::clone(&self.input),
        }
    }

    pub(crate) fn queue_handle(self: &Arc<Self>) -> ComputeQueue {
        ComputeQueue {
            inner: Arc::clone(self),
        }
    }

    pub(crate) fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }

    pub(crate) fn queued_is_empty(&self) -> bool {
        self.queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }

    fn lock_kinds(&self) -> std::sync::MutexGuard<'_, Vec<KindState>> {
        self.kinds.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_queue(&self) -> std::sync::MutexGuard<'_, VecDeque<QueuedJob>> {
        self.queue.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_inflight(&self) -> std::sync::MutexGuard<'_, VecDeque<InFlight>> {
        self.inflight.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn kind_meta(&self, kind: ComputeKind) -> Option<KindMeta> {
        let kinds = self.lock_kinds();
        let k = kinds.get(kind.0 as usize)?;
        Some(KindMeta {
            push_bytes: k.push_bytes,
            inputs: k.inputs,
            output_bytes_max: k.output_bytes_max,
        })
    }

    fn estimate_us(&self, kind: ComputeKind) -> u32 {
        self.lock_kinds()
            .get(kind.0 as usize)
            .map(|k| k.estimate_us.load(Ordering::Relaxed))
            .unwrap_or(BUDGET_START_US)
    }

    pub(crate) fn register(
        &self,
        device: &ash::Device,
        cache: vk::PipelineCache,
        desc: &ComputeDescOwned,
    ) -> Result<ComputeKind, EngineError> {
        if !self.enabled {
            return Err(EngineError::NoCompute);
        }
        validate_desc(desc, &self.limits, self.readback.ring.capacity())?;
        let bindings = descriptor_bindings(desc.inputs);
        let (set_layout, layout) = pass::push_descriptor_layouts(
            device,
            &bindings,
            vk::ShaderStageFlags::COMPUTE,
            desc.push_bytes,
            "compute-job",
        );
        let pipeline = try_compute_pipeline(
            device,
            cache,
            layout,
            &desc.spirv,
            &desc.entry,
            "compute-job",
        )
        .inspect_err(|_| unsafe {
            device.destroy_pipeline_layout(layout, None);
            device.destroy_descriptor_set_layout(set_layout, None);
        })?;
        let mut kinds = self.lock_kinds();
        let id = kinds.len() as u32;
        kinds.push(KindState {
            pipeline,
            layout,
            set_layout,
            push_bytes: desc.push_bytes,
            inputs: desc.inputs,
            output_bytes_max: desc.output_bytes_max,
            estimate_us: AtomicU32::new(BUDGET_START_US),
        });
        log::info!(
            "compute kind {id}: inputs={} push={} out_max={} wg={:?}",
            desc.inputs,
            desc.push_bytes,
            desc.output_bytes_max,
            desc.workgroup
        );
        Ok(ComputeKind(id))
    }

    fn submit_job(&self, job: ComputeJob<'_>) -> Result<JobId, EngineError> {
        if !self.enabled {
            return Err(EngineError::NoCompute);
        }
        let meta = self
            .kind_meta(job.kind)
            .ok_or(EngineError::Invalid("kind"))?;
        if job.output_bytes == 0 || job.output_bytes > meta.output_bytes_max {
            return Err(EngineError::OutputTooLarge);
        }
        if job.push.len() != meta.push_bytes as usize {
            return Err(EngineError::Invalid("push"));
        }
        if job
            .dispatch
            .iter()
            .zip(self.limits.max_work_group_count)
            .any(|(&d, m)| d > m)
        {
            return Err(EngineError::Invalid("dispatch"));
        }
        if job
            .inputs
            .iter()
            .take(meta.inputs as usize)
            .any(Option::is_none)
        {
            return Err(EngineError::InputGone);
        }
        let Some((output, _)) = HostRing::acquire(&self.readback, job.output_bytes as usize) else {
            return Err(EngineError::Busy);
        };
        let inputs = job.inputs.map(|inp| inp.map(ComputeInput::into_lease));
        let id = JobId(self.next_id.fetch_add(1, Ordering::Relaxed));
        self.lock_queue().push_back(QueuedJob {
            id,
            kind: job.kind,
            push: job.push.to_vec().into_boxed_slice(),
            inputs,
            output,
            output_bytes: job.output_bytes,
            dispatch: job.dispatch,
        });
        self.pending.fetch_add(1, Ordering::Relaxed);
        Ok(id)
    }

    fn probe_counters(&self) -> (u64, Option<u64>) {
        let render = unsafe {
            self.device
                .get_semaphore_counter_value(self.graphics_timeline)
                .expect("compute: graphics timeline counter")
        };
        let compute = self.compute_timeline.map(|sem| unsafe {
            self.device
                .get_semaphore_counter_value(sem)
                .expect("compute: compute timeline counter")
        });
        (render, compute)
    }

    /// Mark GPU-complete jobs (no waits). Updates kind estimates from timestamps.
    pub(crate) fn reap(&self) {
        let (render, compute) = self.probe_counters();
        let mut inflight = self.lock_inflight();
        let mut kinds = self.lock_kinds();
        let mut free = self.query_free.lock().unwrap_or_else(|e| e.into_inner());
        for job in inflight.iter_mut() {
            if job.gpu_done {
                continue;
            }
            let done = if job.on_compute_lane {
                compute.is_some_and(|c| c >= job.value.raw())
            } else {
                render >= job.value.raw()
            };
            if !done {
                continue;
            }
            job.gpu_done = true;
            self.gpu_inflight.fetch_sub(1, Ordering::Relaxed);
            if let Some(pair) = job.timestamp_pair.take()
                && self.query_pool != vk::QueryPool::null()
            {
                let q0 = pair * 2;
                let mut data = [0u64; 2];
                let ok = unsafe {
                    self.device.get_query_pool_results(
                        self.query_pool,
                        q0,
                        &mut data,
                        vk::QueryResultFlags::TYPE_64,
                    )
                };
                if ok.is_ok() && data[1] >= data[0] {
                    let ticks = data[1] - data[0];
                    let us = (ticks as f64 * f64::from(self.timestamp_period_ns) / 1000.0).round()
                        as u32;
                    if let Some(kind) = kinds.get_mut(job.kind.0 as usize) {
                        let prev = kind.estimate_us.load(Ordering::Relaxed);
                        kind.estimate_us
                            .store(adapt_estimate_us(prev, us.max(1)), Ordering::Relaxed);
                    }
                }
                if self.host_query_reset {
                    unsafe { self.device.reset_query_pool(self.query_pool, q0, 2) };
                }
                free.push(pair);
            }
        }
    }

    pub(crate) fn poll(&self) -> Vec<(JobId, Box<[u8]>)> {
        self.reap();
        let (render, compute) = self.probe_counters();
        let mut out = Vec::new();
        {
            let mut stash = self.stash.lock().unwrap_or_else(|e| e.into_inner());
            out.extend(stash.drain(..));
        }
        let mut inflight = self.lock_inflight();
        while let Some(front) = inflight.front() {
            if !front.gpu_done {
                break;
            }
            let job = inflight.pop_front().unwrap();
            let bytes = self.readback.copy_out(job.output, job.output_bytes);
            let stamp = if job.on_compute_lane {
                Stamp::Transfer(job.value.raw())
            } else {
                Stamp::Render(job.value.raw())
            };
            self.readback.ring.stamp(job.output, stamp);
            for lease in job.inputs.into_iter().flatten() {
                lease.stamp(stamp);
            }
            self.pending.fetch_sub(1, Ordering::Relaxed);
            out.push((job.id, bytes));
        }
        drop(inflight);
        self.readback.reclaim(render, compute);
        self.input.reclaim(render, compute);
        out
    }

    /// Copy out `id` if it has completed; leave other completed jobs in the
    /// stash so a later [`Self::poll`] still returns them in order.
    pub(crate) fn take_completed(&self, id: JobId) -> Option<Box<[u8]>> {
        let all = self.poll();
        let mut found = None;
        let mut stash = self.stash.lock().unwrap_or_else(|e| e.into_inner());
        for (jid, bytes) in all {
            if jid == id {
                found = Some(bytes);
            } else {
                stash.push_back((jid, bytes));
            }
        }
        found
    }

    fn take_query_pair(&self) -> Option<u32> {
        self.query_free
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .pop()
    }

    fn push_inflight(&self, job: InFlight) {
        self.gpu_inflight.fetch_add(1, Ordering::Relaxed);
        self.lock_inflight().push_back(job);
    }

    pub(crate) fn flush_async(
        &self,
        device: &ash::Device,
        push: &khr::push_descriptor::Device,
        lane: &mut ComputeLane,
    ) {
        if !lane.is_separate_queue() {
            return;
        }
        self.reap();
        loop {
            if self.gpu_inflight.load(Ordering::Relaxed) >= MAX_IN_FLIGHT {
                break;
            }
            let Some(job) = self.lock_queue().pop_front() else {
                break;
            };
            let batch = unsafe { lane.begin(device) };
            let cmd = batch.cmd();
            let pair = self.take_query_pair();
            self.record_one(device, push, cmd, &job, pair);
            let value = unsafe { lane.submit(device, batch) };
            self.push_inflight(InFlight {
                id: job.id,
                kind: job.kind,
                value,
                on_compute_lane: true,
                gpu_done: false,
                output: job.output,
                output_bytes: job.output_bytes,
                inputs: job.inputs,
                timestamp_pair: pair,
            });
        }
    }

    pub(crate) fn record_budgeted(
        &self,
        device: &ash::Device,
        push: &khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        render_value: TimelineValue,
    ) {
        self.reap();
        let n = {
            let q = self.lock_queue();
            if q.is_empty() {
                return;
            }
            let costs: Vec<u32> = q.iter().map(|j| self.estimate_us(j.kind)).collect();
            jobs_fitting_budget(FALLBACK_FRAME_BUDGET_US, &costs)
        };
        if n == 0 {
            return;
        }
        let jobs: Vec<QueuedJob> = {
            let mut q = self.lock_queue();
            let n = n.min(q.len());
            q.drain(..n).collect()
        };
        for job in jobs {
            let pair = self.take_query_pair();
            self.record_one(device, push, cmd, &job, pair);
            self.push_inflight(InFlight {
                id: job.id,
                kind: job.kind,
                value: render_value,
                on_compute_lane: false,
                gpu_done: false,
                output: job.output,
                output_bytes: job.output_bytes,
                inputs: job.inputs,
                timestamp_pair: pair,
            });
        }
    }

    pub(crate) fn take_all_queued(&self) -> Vec<QueuedJob> {
        self.lock_queue().drain(..).collect()
    }

    fn record_one(
        &self,
        device: &ash::Device,
        push: &khr::push_descriptor::Device,
        cmd: vk::CommandBuffer,
        job: &QueuedJob,
        timestamp_pair: Option<u32>,
    ) {
        let kinds = self.lock_kinds();
        let Some(kind) = kinds.get(job.kind.0 as usize) else {
            return;
        };
        let pipeline = kind.pipeline;
        let layout = kind.layout;
        let push_bytes = kind.push_bytes;
        let n_inputs = kind.inputs;
        drop(kinds);

        unsafe {
            if let Some(pair) = timestamp_pair
                && self.query_pool != vk::QueryPool::null()
            {
                let q0 = pair * 2;
                if self.host_query_reset {
                    device.reset_query_pool(self.query_pool, q0, 2);
                } else {
                    device.cmd_reset_query_pool(cmd, self.query_pool, q0, 2);
                }
                device.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::COMPUTE_SHADER,
                    self.query_pool,
                    q0,
                );
            }

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            if push_bytes > 0 {
                let mut tmp = [0u8; MAX_PUSH_BYTES as usize];
                let n = job.push.len().min(MAX_PUSH_BYTES as usize);
                tmp[..n].copy_from_slice(&job.push[..n]);
                device.cmd_push_constants(
                    cmd,
                    layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &tmp[..push_bytes as usize],
                );
            }

            let out_buf = if self.direct {
                self.readback.buffer
            } else {
                self.scratch
                    .as_ref()
                    .map(|s| s.buffer)
                    .unwrap_or(self.readback.buffer)
            };
            let out_info = vk::DescriptorBufferInfo::default()
                .buffer(out_buf)
                .offset(job.output.offset)
                .range(job.output.len.max(1));
            let in0 = job.inputs[0].as_ref().map(|l| {
                vk::DescriptorBufferInfo::default()
                    .buffer(self.input.buffer)
                    .offset(l.region.offset)
                    .range(l.region.len.max(1))
            });
            let in1 = job.inputs[1].as_ref().map(|l| {
                vk::DescriptorBufferInfo::default()
                    .buffer(self.input.buffer)
                    .offset(l.region.offset)
                    .range(l.region.len.max(1))
            });

            match n_inputs {
                0 => {
                    let writes = [vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(&out_info))];
                    push.cmd_push_descriptor_set(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &writes,
                    );
                }
                1 => {
                    let i0 = in0.expect("input 0");
                    let infos = [i0, out_info];
                    let writes = [
                        vk::WriteDescriptorSet::default()
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[0..1]),
                        vk::WriteDescriptorSet::default()
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[1..2]),
                    ];
                    push.cmd_push_descriptor_set(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &writes,
                    );
                }
                _ => {
                    let i0 = in0.expect("input 0");
                    let i1 = in1.expect("input 1");
                    let infos = [i0, i1, out_info];
                    let writes = [
                        vk::WriteDescriptorSet::default()
                            .dst_binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[0..1]),
                        vk::WriteDescriptorSet::default()
                            .dst_binding(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[1..2]),
                        vk::WriteDescriptorSet::default()
                            .dst_binding(2)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&infos[2..3]),
                    ];
                    push.cmd_push_descriptor_set(
                        cmd,
                        vk::PipelineBindPoint::COMPUTE,
                        layout,
                        0,
                        &writes,
                    );
                }
            }

            device.cmd_dispatch(cmd, job.dispatch[0], job.dispatch[1], job.dispatch[2]);

            if !self.direct {
                if let Some(scratch) = &self.scratch {
                    let to_copy = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)];
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&to_copy),
                    );
                    let region = [vk::BufferCopy::default()
                        .src_offset(job.output.offset)
                        .dst_offset(job.output.offset)
                        .size(u64::from(job.output_bytes))];
                    device.cmd_copy_buffer(cmd, scratch.buffer, self.readback.buffer, &region);
                    let to_host = [vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::COPY)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                        .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                    device.cmd_pipeline_barrier2(
                        cmd,
                        &vk::DependencyInfo::default().memory_barriers(&to_host),
                    );
                }
            } else {
                let to_host = [vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::HOST)
                    .dst_access_mask(vk::AccessFlags2::HOST_READ)];
                device.cmd_pipeline_barrier2(
                    cmd,
                    &vk::DependencyInfo::default().memory_barriers(&to_host),
                );
            }

            if let Some(pair) = timestamp_pair
                && self.query_pool != vk::QueryPool::null()
            {
                device.cmd_write_timestamp2(
                    cmd,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                    self.query_pool,
                    pair * 2 + 1,
                );
            }
        }
    }

    pub(crate) unsafe fn destroy(&self, device: &ash::Device) {
        let mut kinds = self.lock_kinds();
        for k in kinds.drain(..) {
            unsafe {
                device.destroy_pipeline(k.pipeline, None);
                device.destroy_pipeline_layout(k.layout, None);
                device.destroy_descriptor_set_layout(k.set_layout, None);
            }
        }
        if self.query_pool != vk::QueryPool::null() {
            unsafe { device.destroy_query_pool(self.query_pool, None) };
        }
        if let Some(s) = &self.scratch {
            unsafe {
                device.destroy_buffer(s.buffer, None);
                device.free_memory(s.memory, None);
            }
        }
        unsafe {
            self.input.destroy(device);
            self.readback.destroy(device);
        }
    }
}

fn descriptor_bindings(inputs: u32) -> Vec<vk::DescriptorSetLayoutBinding<'static>> {
    let mut b = Vec::with_capacity(inputs as usize + 1);
    for i in 0..inputs {
        b.push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(i)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
        );
    }
    b.push(
        vk::DescriptorSetLayoutBinding::default()
            .binding(inputs)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    );
    b
}

fn validate_desc(
    desc: &ComputeDescOwned,
    limits: &ComputeLimits,
    readback_cap: u64,
) -> Result<(), EngineError> {
    if desc.spirv.is_empty() {
        return Err(EngineError::Invalid("spirv"));
    }
    if desc.entry.is_empty() {
        return Err(EngineError::Invalid("entry"));
    }
    if desc.push_bytes > MAX_PUSH_BYTES.min(limits.max_push) || !desc.push_bytes.is_multiple_of(4) {
        return Err(EngineError::Invalid("push_bytes"));
    }
    if desc.inputs > MAX_INPUTS {
        return Err(EngineError::Invalid("inputs"));
    }
    if desc.output_bytes_max == 0 || u64::from(desc.output_bytes_max) > readback_cap {
        return Err(EngineError::Invalid("output_bytes_max"));
    }
    let wg = desc.workgroup;
    if wg[0] == 0 || wg[1] == 0 || wg[2] == 0 {
        return Err(EngineError::Invalid("workgroup"));
    }
    if wg
        .iter()
        .zip(limits.max_work_group_size)
        .any(|(&w, m)| w > m)
    {
        return Err(EngineError::Invalid("workgroup"));
    }
    let invocs = wg[0].saturating_mul(wg[1]).saturating_mul(wg[2]);
    if invocs > limits.max_work_group_invocations {
        return Err(EngineError::Invalid("workgroup"));
    }
    Ok(())
}

fn try_compute_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    layout: vk::PipelineLayout,
    bytes: &[u8],
    entry: &str,
    label: &str,
) -> Result<vk::Pipeline, EngineError> {
    let code =
        ash::util::read_spv(&mut Cursor::new(bytes)).map_err(|_| EngineError::Invalid("spirv"))?;
    let module = unsafe {
        device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)
            .map_err(|_| EngineError::Pipeline)?
    };
    let name = CString::new(entry).map_err(|_| EngineError::Invalid("entry"))?;
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .module(module)
        .name(&name)
        .stage(vk::ShaderStageFlags::COMPUTE);
    let result = unsafe {
        device.create_compute_pipelines(
            cache,
            &[vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout)],
            None,
        )
    };
    unsafe { device.destroy_shader_module(module, None) };
    match result {
        Ok(p) => Ok(p[0]),
        Err((_, err)) => {
            log::error!("create {label} compute pipeline: {err:?}");
            Err(EngineError::Pipeline)
        }
    }
}

impl super::Renderer {
    pub(crate) fn register_compute(
        &mut self,
        desc: ComputeDescOwned,
    ) -> Result<ComputeKind, EngineError> {
        self.compute
            .register(&self.device.device, self.pipeline_cache, &desc)
    }

    /// Dedicated / second-queue: submit queued jobs now. No-op on fallback
    /// or when the queue is empty.
    pub(crate) fn flush_compute_async(&mut self) {
        if self.compute.queued_is_empty() && self.compute.gpu_inflight.load(Ordering::Relaxed) == 0
        {
            return;
        }
        self.compute.flush_async(
            &self.device.device,
            &self.device.push_descriptor,
            &mut self.compute_lane,
        );
    }

    /// Same-queue fallback: record a budgeted prefix into `cmd`. No-op when
    /// the queue is empty (idle frames are untouched).
    pub(crate) fn record_compute_fallback(
        &mut self,
        cmd: vk::CommandBuffer,
        render_value: TimelineValue,
    ) {
        if self.compute_lane.is_separate_queue() || self.compute.queued_is_empty() {
            return;
        }
        self.compute.record_budgeted(
            &self.device.device,
            &self.device.push_descriptor,
            cmd,
            render_value,
        );
    }

    /// Drain every queued job and wait. Used by
    /// [`crate::Engine::run_compute_blocking`].
    pub(crate) fn flush_compute_blocking(&mut self) {
        self.flush_pending_submits();
        let device = self.device.device.clone();
        unsafe {
            let _ = device.device_wait_idle();
        }
        self.compute.reap();
        if self.compute_lane.is_separate_queue() {
            loop {
                if self.compute.queued_is_empty() {
                    break;
                }
                self.flush_compute_async();
                unsafe {
                    let _ = device.device_wait_idle();
                }
                self.compute.reap();
            }
            return;
        }
        let jobs = self.compute.take_all_queued();
        if jobs.is_empty() {
            return;
        }
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.device.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = unsafe { device.allocate_command_buffers(&alloc) }
            .expect("compute blocking command buffer")[0];
        let rs = self.timeline.begin_render(cmd);
        let value = rs.value();
        unsafe {
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .expect("begin blocking compute");
        }
        for job in &jobs {
            self.compute
                .record_one(&device, &self.device.push_descriptor, cmd, job, None);
        }
        unsafe {
            device
                .end_command_buffer(cmd)
                .expect("end blocking compute");
            let _ = rs.submit(&device, self.device.graphics_queue, &self.timeline, None);
            self.timeline.wait(&device, value);
            device.free_command_buffers(self.device.command_pool, &[cmd]);
        }
        for job in jobs {
            self.compute.push_inflight(InFlight {
                id: job.id,
                kind: job.kind,
                value,
                on_compute_lane: false,
                gpu_done: false,
                output: job.output,
                output_bytes: job.output_bytes,
                inputs: job.inputs,
                timestamp_pair: None,
            });
        }
        self.compute.reap();
    }
}

pub(crate) fn compute_limits(props: &vk::PhysicalDeviceProperties) -> ComputeLimits {
    ComputeLimits {
        max_work_group_invocations: props.limits.max_compute_work_group_invocations,
        max_work_group_size: props.limits.max_compute_work_group_size,
        max_work_group_count: props.limits.max_compute_work_group_count,
        max_push: props.limits.max_push_constants_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk::mesh_staging::StagingRing;

    #[test]
    fn budget_empty_is_zero() {
        assert_eq!(jobs_fitting_budget(200, &[]), 0);
    }

    #[test]
    fn budget_packs_until_full() {
        assert_eq!(jobs_fitting_budget(200, &[100, 100]), 2);
        assert_eq!(jobs_fitting_budget(200, &[100, 100, 100]), 2);
        assert_eq!(jobs_fitting_budget(200, &[50, 50, 50, 50]), 4);
        assert_eq!(jobs_fitting_budget(200, &[200]), 1);
        assert_eq!(jobs_fitting_budget(200, &[200, 1]), 1);
    }

    #[test]
    fn budget_always_takes_the_first_job() {
        assert_eq!(jobs_fitting_budget(200, &[500]), 1);
        assert_eq!(jobs_fitting_budget(200, &[500, 50]), 1);
        assert_eq!(jobs_fitting_budget(50, &[200, 50]), 1);
    }

    #[test]
    fn adapt_estimate_ema_and_clamp() {
        assert_eq!(adapt_estimate_us(200, 50), 125);
        assert_eq!(adapt_estimate_us(200, 4000), BUDGET_MAX_US);
        assert_eq!(adapt_estimate_us(50, 50), BUDGET_MIN_US);
        assert_eq!(adapt_estimate_us(2000, 2000), BUDGET_MAX_US);
        assert_eq!(adapt_estimate_us(BUDGET_START_US, BUDGET_START_US), 200);
    }

    #[test]
    fn job_ids_are_monotonic_and_fifo() {
        let next = AtomicU64::new(1);
        let mut q = VecDeque::new();
        let alloc = || JobId(next.fetch_add(1, Ordering::Relaxed));
        let a = alloc();
        q.push_back(a);
        let b = alloc();
        q.push_back(b);
        let c = alloc();
        q.push_back(c);
        assert!(a < b && b < c);
        assert_eq!(a.raw(), 1);
        let drained: Vec<_> = q.drain(..).collect();
        assert_eq!(drained, vec![a, b, c]);
    }

    #[test]
    fn readback_ring_reclaim_with_fake_timeline() {
        let ring = StagingRing::new(16, 16);
        let a = ring.acquire(16).unwrap();
        ring.stamp(a, Stamp::Render(2));
        ring.reclaim(1, None);
        assert!(
            ring.acquire(16).is_none(),
            "region waits for the fake timeline"
        );
        ring.reclaim(2, None);
        let b = ring.acquire(16).expect("reclaimed at timeline 2");
        assert_eq!(b.offset, 0);
        ring.stamp(b, Stamp::Transfer(4));
        ring.reclaim(100, None);
        assert!(
            ring.acquire(16).is_none(),
            "Transfer stamp ignores the render counter"
        );
        ring.reclaim(0, Some(4));
        assert!(ring.acquire(16).is_some());
    }

    #[test]
    fn parse_compute_mb_env() {
        assert_eq!(parse_mb("16"), Some(16 << 20));
        assert_eq!(parse_mb(" 32 "), Some(32 << 20));
        assert_eq!(parse_mb("0"), Some(0));
        assert_eq!(parse_mb(""), None);
        assert_eq!(parse_mb("nope"), None);
    }

    #[test]
    fn pick_dedicated_compute_family() {
        let p = pick_compute_queue(0, 16, Tier::DedicatedFamily, 1, 2, Some((2, 8)));
        assert_eq!(p.tier, Tier::DedicatedFamily);
        assert_eq!(p.family, 2);
        assert_eq!(p.queue_index, 0);
        assert_eq!(p.graphics_queues, 1);
    }

    #[test]
    fn pick_compute_shares_transfer_family_second_queue() {
        let p = pick_compute_queue(0, 16, Tier::DedicatedFamily, 1, 2, Some((1, 2)));
        assert_eq!(p.tier, Tier::DedicatedFamily);
        assert_eq!(p.family, 1);
        assert_eq!(p.queue_index, 1);
    }

    #[test]
    fn pick_second_graphics_queue_when_transfer_took_the_first_extra() {
        let p = pick_compute_queue(0, 16, Tier::SecondQueueSameFamily, 0, 16, None);
        assert_eq!(p.tier, Tier::SecondQueueSameFamily);
        assert_eq!(p.queue_index, 2);
        assert_eq!(p.graphics_queues, 3);
    }

    #[test]
    fn pick_second_graphics_queue_when_transfer_is_dedicated() {
        let p = pick_compute_queue(0, 16, Tier::DedicatedFamily, 1, 2, None);
        assert_eq!(p.tier, Tier::SecondQueueSameFamily);
        assert_eq!(p.queue_index, 1);
        assert_eq!(p.graphics_queues, 2);
    }

    #[test]
    fn pick_same_queue_fallback() {
        let p = pick_compute_queue(0, 1, Tier::SameQueueFallback, 0, 1, None);
        assert_eq!(p.tier, Tier::SameQueueFallback);
        assert_eq!(p.queue_index, 0);
        assert_eq!(p.graphics_queues, 1);
    }

    #[test]
    fn stager_is_send_sync_input_is_send() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync>() {}
        send_sync::<ComputeStager>();
        send_sync::<ComputeQueue>();
        send::<ComputeInput>();
        send::<JobId>();
        send::<ComputeKind>();
    }
}
