//! Timestamp-query GPU pass timer (`GpuTimer`, `GpuPass`) and its stamps.
//! Split out of `mod.rs` so later work can touch profiling without opening
//! the renderer setup.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use ash::vk;

use super::buffers::FRAMES_IN_FLIGHT;

/// A GPU render pass boundary, in record order. The variant ordinal indexes the
/// per-pass accumulator (matches the tracking in [`crate::profile::Meter`]).
/// Every span of the render command buffer ends at one of these, so the
/// stamps sum to the whole GPU frame (the present copy is timed apart, see
/// [`GpuTimer::end_copy`]).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum GpuPass {
    /// Staged mesh copies (graphics-queue tiers) + the minimap upload.
    Copies,
    /// Cull compute: counts fill, dispatch, DRAW_INDIRECT barrier.
    Cull,
    /// The cascaded shadow-map pass (stamped only on regenerating frames).
    ShadowMap,
    /// Scene-pass begin: attachment transitions + `cmd_begin_rendering` clears.
    Clear,
    /// Full-res opaque (camera group 0, all distance buckets).
    OpaqueFull,
    /// Coarse-LOD opaque (camera group 2); recorded after full-res so the
    /// LOD skirt is mostly depth-rejected. Before cutout.
    OpaqueLod,
    /// Cutout (camera group 1); recorded after both opaque partitions.
    Cutout,
    Sky,
    Cubes,
    Lines,
    Shadows,
    Transparent,
    /// HUD overlay. Drawn in the present copy (`GpuTonemap`); the scene-pass
    /// stamp accounts 0 so the report still lists this meter.
    Overlay,
    /// End of the scene pass: `cmd_end_rendering` (where the MSAA color
    /// resolve executes) and the offscreen/depth-rest finalize transitions.
    Resolve,
    /// The VRS classify dispatch (end of frame, after depth rests; stamped
    /// only when it runs).
    Vrs,
    /// Retired TAA compute stamp. Resolve is fused into the present-time
    /// tonemap (`GpuTonemap`); this variant is never marked and reports 0 so
    /// `GpuPass` ordinals stay stable.
    Taa,
    /// Exposure metering reduce + finalize (stamped only when it runs).
    Exposure,
    /// The bloom chain + quarter-res spill dispatch — the render-command tail.
    /// Without this closing stamp everything after the last boundary silently
    /// vanishes from the report.
    Bloom,
}

impl GpuPass {
    pub(super) const ALL: [GpuPass; 18] = [
        GpuPass::Copies,
        GpuPass::Cull,
        GpuPass::ShadowMap,
        GpuPass::Clear,
        GpuPass::OpaqueFull,
        GpuPass::OpaqueLod,
        GpuPass::Cutout,
        GpuPass::Sky,
        GpuPass::Cubes,
        GpuPass::Lines,
        GpuPass::Shadows,
        GpuPass::Transparent,
        GpuPass::Overlay,
        GpuPass::Resolve,
        GpuPass::Vrs,
        GpuPass::Taa,
        GpuPass::Exposure,
        GpuPass::Bloom,
    ];
    pub(super) const COUNT: usize = Self::ALL.len();

    pub(super) fn meter(self) -> crate::profile::Meter {
        use crate::profile::Meter;
        match self {
            GpuPass::Copies => Meter::GpuCopies,
            GpuPass::Cull => Meter::GpuCull,
            GpuPass::ShadowMap => Meter::GpuShadowMap,
            GpuPass::Clear => Meter::GpuClear,
            GpuPass::OpaqueFull => Meter::GpuOpaqueFull,
            GpuPass::OpaqueLod => Meter::GpuOpaqueLod,
            GpuPass::Cutout => Meter::GpuCutout,
            GpuPass::Sky => Meter::GpuSky,
            GpuPass::Cubes => Meter::GpuCubes,
            GpuPass::Lines => Meter::GpuLines,
            GpuPass::Shadows => Meter::GpuShadows,
            GpuPass::Transparent => Meter::GpuTransparent,
            GpuPass::Overlay => Meter::GpuOverlay,
            GpuPass::Resolve => Meter::GpuResolve,
            GpuPass::Vrs => Meter::GpuVrs,
            GpuPass::Taa => Meter::GpuTaa,
            GpuPass::Exposure => Meter::GpuExposure,
            GpuPass::Bloom => Meter::GpuBloom,
        }
    }

    /// Combined opaque span (full-res + coarse LOD + cutout). Not a stamped
    /// pass — summed from the three group stamps at readback so the `opaque`
    /// meter stays comparable with reports that predate the split.
    pub(super) fn opaque_ms(passes: &[f64; Self::COUNT]) -> f64 {
        passes[Self::OpaqueFull as usize]
            + passes[Self::OpaqueLod as usize]
            + passes[Self::Cutout as usize]
    }
}

/// One start timestamp plus one boundary per pass.
const GPU_STAMPS: usize = GpuPass::COUNT + 1;
/// The present copy's start/end pair lives after the per-slot render ranges.
/// One pair suffices: a new copy is only recorded once the previous one has
/// retired (`decide_present` probes/waits it), so its stamps are read first.
const COPY_STAMP_BASE: u32 = (GPU_STAMPS * FRAMES_IN_FLIGHT as usize) as u32;
/// Two frame-boundary stamps per slot (TOP / BOTTOM), after the copy pair.
/// Recorded only after [`crate::Engine::enable_gpu_load`]; the profiler
/// stamps (`VOXEL_PROFILE`) use a separate range and are unaffected.
const LOAD_STAMP_BASE: u32 = COPY_STAMP_BASE + 2;
const LOAD_STAMPS: u32 = 2;
const QUERY_COUNT: u32 = LOAD_STAMP_BASE + LOAD_STAMPS * FRAMES_IN_FLIGHT as u32;

fn load_query(slot: usize, i: u32) -> u32 {
    LOAD_STAMP_BASE + slot as u32 * LOAD_STAMPS + i
}

/// Last completed frame's GPU busy time and idle gap before that submit.
#[derive(Clone, Copy, Debug)]
pub struct GpuLoad {
    pub frame_ms: f32,
    pub gap_ms: f32,
}

/// Published by the render thread, read by [`crate::Engine::gpu_load`].
/// Two `f32` bit-patterns, same pattern as [`super::exposure::ExposureShared`].
/// Recording is off until [`crate::Engine::enable_gpu_load`]; `load` is `None`
/// while disabled even if a previous enable left values in the atomics.
#[derive(Clone)]
pub struct GpuLoadShared {
    frame: Arc<AtomicU32>,
    gap: Arc<AtomicU32>,
    ready: Arc<AtomicBool>,
    enabled: Arc<AtomicBool>,
}

impl GpuLoadShared {
    fn new() -> Self {
        Self {
            frame: Arc::new(AtomicU32::new(0)),
            gap: Arc::new(AtomicU32::new(0)),
            ready: Arc::new(AtomicBool::new(false)),
            enabled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub(crate) fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub(crate) fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Release);
        if !on {
            self.ready.store(false, Ordering::Release);
        }
    }

    fn store(&self, frame_ms: f32, gap_ms: f32) {
        if !self.enabled.load(Ordering::Relaxed) {
            return;
        }
        self.frame.store(frame_ms.to_bits(), Ordering::Relaxed);
        self.gap.store(gap_ms.to_bits(), Ordering::Relaxed);
        self.ready.store(true, Ordering::Release);
    }

    pub fn load(&self) -> Option<GpuLoad> {
        if !self.enabled.load(Ordering::Acquire) {
            return None;
        }
        if !self.ready.load(Ordering::Acquire) {
            return None;
        }
        Some(GpuLoad {
            frame_ms: f32::from_bits(self.frame.load(Ordering::Relaxed)),
            gap_ms: f32::from_bits(self.gap.load(Ordering::Relaxed)),
        })
    }
}

/// Per-pass GPU timing via a timestamp query pool: a start timestamp plus one
/// after each recorded pass. Only the passes that actually run write a stamp,
/// and the label written alongside each stamp keeps deltas attributable even
/// when a frame skips passes (no 3D, VRS off). A slot's results are read one
/// reuse cycle later (`FRAMES_IN_FLIGHT` frames), after its fence is waited, so
/// the read never stalls — and because that wait is in render order, consecutive
/// `read_into` calls are consecutive rendered frames (possibly different slots).
/// Their timestamps share the device clock, so `start(N) - end(N-1)` is the idle
/// gap before this submit.
///
/// `count`/`label` are [`Cell`]s so a mark needs only `&self`: the render pass
/// holds an immutable `&Renderer` while recording, and all timer state is
/// touched on the single render thread. A null pool (hardware without timestamp
/// support) makes every method a no-op.
pub(super) struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per tick (`limits.timestampPeriod`).
    period_ns: f32,
    /// `hostQueryReset` (Vulkan 1.2). Host-reset after the slot fence wait;
    /// otherwise `vkCmdResetQueryPool` outside the render pass.
    host_reset: bool,
    /// Whether each slot holds completed timestamps to read back.
    primed: [bool; FRAMES_IN_FLIGHT as usize],
    /// Stamps written for each slot's most recent recording (incl. the start).
    count: [std::cell::Cell<u32>; FRAMES_IN_FLIGHT as usize],
    /// The pass that ended at each stamp (index `i` labels the span `i-1..i`).
    label: [[std::cell::Cell<GpuPass>; GPU_STAMPS]; FRAMES_IN_FLIGHT as usize],
    /// Armed when the open span recorded GPU commands. [`Self::mark`] writes a
    /// stamp only when this is set; otherwise the pass accounts 0.
    pending: [std::cell::Cell<bool>; FRAMES_IN_FLIGHT as usize],
    /// Whether the present-copy pair holds a completed range to read back.
    copy_primed: bool,
    /// Last stamp of the previously *read* render submit (raw ticks), used to
    /// compute the idle gap before the next readable frame. Cleared when a
    /// readback is unavailable so a later start is not compared across a hole.
    prev_end: Option<u64>,
    load_primed: [bool; FRAMES_IN_FLIGHT as usize],
    /// `begin_load` actually wrote this slot's TOP stamp; `end_load` must
    /// close that pair even if the setter flips mid-record.
    load_open: [std::cell::Cell<bool>; FRAMES_IN_FLIGHT as usize],
    load_prev_end: Option<u64>,
    load: GpuLoadShared,
}

/// Device-time gap (ms) from the previous render submit's last stamp to this
/// submit's first stamp. `period_ns` is `VkPhysicalDeviceLimits::timestampPeriod`.
///
/// A start that precedes the previous end is GPU overlap (the next command
/// buffer began before the last one drained) — zero idle, not a 64-bit wrap.
/// Session-length 64-bit timestamp clocks do not wrap.
pub(super) fn idle_gap_ms(prev_end: u64, this_start: u64, period_ns: f32) -> f64 {
    if this_start <= prev_end {
        0.0
    } else {
        this_start.wrapping_sub(prev_end) as f64 * period_ns as f64 / 1.0e6
    }
}

impl GpuTimer {
    pub(super) fn new(
        device: &ash::Device,
        supported: bool,
        period_ns: f32,
        host_reset: bool,
    ) -> Self {
        let pool = if supported {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(QUERY_COUNT);
            unsafe {
                device
                    .create_query_pool(&info, None)
                    .expect("Failed to create timestamp query pool")
            }
        } else {
            vk::QueryPool::null()
        };
        Self {
            pool,
            period_ns,
            host_reset,
            primed: [false; FRAMES_IN_FLIGHT as usize],
            count: std::array::from_fn(|_| std::cell::Cell::new(0)),
            label: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::cell::Cell::new(GpuPass::OpaqueFull))
            }),
            pending: std::array::from_fn(|_| std::cell::Cell::new(false)),
            copy_primed: false,
            prev_end: None,
            load_primed: [false; FRAMES_IN_FLIGHT as usize],
            load_open: std::array::from_fn(|_| std::cell::Cell::new(false)),
            load_prev_end: None,
            load: GpuLoadShared::new(),
        }
    }

    pub(super) fn load_shared(&self) -> GpuLoadShared {
        self.load.clone()
    }

    fn enabled(&self) -> bool {
        self.pool != vk::QueryPool::null()
    }

    /// Reads `slot`'s prior render-pass per-pass durations (ms), adding each to
    /// `sink`. The caller must have waited `slot`'s fence, so the result is
    /// ready without a GPU stall. Returns the summed render-pass time and the
    /// idle gap before this submit (`None` on the first readable frame).
    ///
    /// Reuses the timestamps already fetched for the per-pass spans — no extra
    /// query readback. An unavailable result (too few stamps, or the pool
    /// read failing) drops the stored previous end so the next successful
    /// frame does not treat skipped GPU work as idle.
    pub(super) unsafe fn read_into(
        &mut self,
        device: &ash::Device,
        slot: usize,
        sink: &mut [f64],
    ) -> Option<(f64, Option<f64>)> {
        if !self.enabled() || !self.primed[slot] {
            return None;
        }
        let n = self.count[slot].get() as usize;
        if n < 2 {
            self.prev_end = None;
            return None;
        }
        let mut ts = [0u64; GPU_STAMPS];
        let read = unsafe {
            device.get_query_pool_results(
                self.pool,
                slot as u32 * GPU_STAMPS as u32,
                &mut ts[..n],
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if read.is_err() {
            self.prev_end = None;
            return None;
        }
        let mut total = 0.0;
        for i in 1..n {
            let ms = ts[i].wrapping_sub(ts[i - 1]) as f64 * self.period_ns as f64 / 1.0e6;
            sink[self.label[slot][i].get() as usize] += ms;
            total += ms;
        }
        // `ts[0]` is the TOP_OF_PIPE `begin`; `ts[n-1]` is the last BOTTOM_OF_PIPE `mark`.
        let gap = self
            .prev_end
            .map(|end| idle_gap_ms(end, ts[0], self.period_ns));
        self.prev_end = Some(ts[n - 1]);
        Some((total, gap))
    }

    fn reset_queries(&self, device: &ash::Device, cmd: vk::CommandBuffer, first: u32, count: u32) {
        unsafe {
            if self.host_reset {
                device.reset_query_pool(self.pool, first, count);
            } else {
                device.cmd_reset_query_pool(cmd, self.pool, first, count);
            }
        }
    }

    /// Resets `slot`'s queries and writes the start timestamp. Must be recorded
    /// outside any render pass. Host-resets the pool when the device has
    /// `hostQueryReset`; otherwise `vkCmdResetQueryPool`.
    pub(super) unsafe fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if !self.enabled() {
            return;
        }
        let base = slot as u32 * GPU_STAMPS as u32;
        self.reset_queries(device, cmd, base, GPU_STAMPS as u32);
        unsafe {
            device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, base);
        }
        self.count[slot].set(1);
        self.pending[slot].set(false);
    }

    /// Arm the next [`Self::mark`]: the open span recorded GPU commands.
    /// Cheap no-op when timestamps are off.
    pub(super) fn recorded(&self, slot: usize) {
        if self.enabled() {
            self.pending[slot].set(true);
        }
    }

    /// Writes a boundary timestamp closing `pass` for `slot` if the pass
    /// recorded GPU work ([`Self::recorded`]). Otherwise a no-op: the pass
    /// accounts 0 at readback (the sink starts at 0; `GpuPass::ALL` is still
    /// fed so the profiler report format is unchanged).
    pub(super) unsafe fn mark(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        pass: GpuPass,
    ) {
        if !self.enabled() || !self.pending[slot].replace(false) {
            return;
        }
        let i = self.count[slot].get();
        if i as usize >= GPU_STAMPS {
            return;
        }
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.pool,
                slot as u32 * GPU_STAMPS as u32 + i,
            );
        }
        self.label[slot][i as usize].set(pass);
        self.count[slot].set(i + 1);
    }

    /// Reads this slot's previous load pair (TOP/BOTTOM) after its fence wait
    /// and publishes [`GpuLoadShared`]. `None` until the first successful
    /// readback; a hole drops `load_prev_end` so the next gap is not invented.
    /// Skipped while [`crate::Engine::enable_gpu_load`] is off (drops a primed
    /// unread pair so a later enable does not publish a stale sample).
    pub(super) unsafe fn read_load(&mut self, device: &ash::Device, slot: usize) {
        if !self.enabled() || !self.load_primed[slot] {
            return;
        }
        if !self.load.is_enabled() {
            self.load_primed[slot] = false;
            self.load_prev_end = None;
            return;
        }
        let mut ts = [0u64; 2];
        let read = unsafe {
            device.get_query_pool_results(
                self.pool,
                load_query(slot, 0),
                &mut ts,
                vk::QueryResultFlags::TYPE_64,
            )
        };
        if read.is_err() {
            self.load_prev_end = None;
            return;
        }
        let frame_ms = ts[1].wrapping_sub(ts[0]) as f64 * self.period_ns as f64 / 1.0e6;
        let gap_ms = self
            .load_prev_end
            .map(|end| idle_gap_ms(end, ts[0], self.period_ns))
            .unwrap_or(0.0);
        self.load_prev_end = Some(ts[1]);
        self.load.store(frame_ms as f32, gap_ms as f32);
    }

    /// Resets the load pair and writes TOP_OF_PIPE. Recorded only when
    /// [`crate::Engine::enable_gpu_load`] is on — two stamps, host query reset
    /// when available. The profiler's own stamps are a separate range.
    pub(super) unsafe fn begin_load(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) {
        if !self.enabled() || !self.load.is_enabled() {
            return;
        }
        self.reset_queries(device, cmd, load_query(slot, 0), LOAD_STAMPS);
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.pool,
                load_query(slot, 0),
            );
        }
        self.load_open[slot].set(true);
    }

    /// Writes BOTTOM_OF_PIPE and marks the pair readable next cycle.
    /// Closes a pair `begin_load` actually opened, even if the setter flipped
    /// off mid-record (an unmatched TOP would leave the query incomplete).
    pub(super) unsafe fn end_load(
        &mut self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
    ) {
        if !self.enabled() || !self.load_open[slot].get() {
            return;
        }
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.pool,
                load_query(slot, 1),
            );
        }
        self.load_open[slot].set(false);
        self.load_primed[slot] = true;
    }

    /// Marks `slot` readable next cycle. Call after the render pass ends.
    pub(super) fn finish(&mut self, slot: usize) {
        if self.enabled() {
            self.primed[slot] = true;
        }
    }

    /// Reads the previous present copy's duration (ms). The caller must know
    /// that copy has retired (a new copy is only recorded once it has), so the
    /// read never stalls; `None` before the first copy or without timestamps.
    pub(super) unsafe fn read_copy(&self, device: &ash::Device) -> Option<f64> {
        if !self.enabled() || !self.copy_primed {
            return None;
        }
        let mut ts = [0u64; 2];
        unsafe {
            device.get_query_pool_results(
                self.pool,
                COPY_STAMP_BASE,
                &mut ts,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .ok()?;
        Some(ts[1].wrapping_sub(ts[0]) as f64 * self.period_ns as f64 / 1.0e6)
    }

    /// Resets the present-copy pair and writes its start stamp. Recorded on the
    /// copy command buffer, outside any render pass, after [`Self::read_copy`].
    /// Host-resets when the device has `hostQueryReset`.
    pub(super) unsafe fn begin_copy(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if !self.enabled() {
            return;
        }
        self.reset_queries(device, cmd, COPY_STAMP_BASE, 2);
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::TOP_OF_PIPE,
                self.pool,
                COPY_STAMP_BASE,
            );
        }
    }

    /// Writes the present-copy end stamp (after the last barrier, before the
    /// command buffer ends) and marks the pair readable by the next copy.
    pub(super) unsafe fn end_copy(&mut self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if !self.enabled() {
            return;
        }
        unsafe {
            device.cmd_write_timestamp2(
                cmd,
                vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                self.pool,
                COPY_STAMP_BASE + 1,
            );
        }
        self.copy_primed = true;
    }

    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.enabled() {
            unsafe { device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}

/// Pipeline-statistics pass, in record order. The ordinal indexes the per-slot
/// query (`slot * COUNT + pass`).
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum PipeStatPass {
    OpaqueFull,
    OpaqueLod,
    Cutout,
    Sky,
    Transparent,
}

impl PipeStatPass {
    pub(super) const COUNT: usize = 5;
}

/// One query writes counters in bit-order of the enabled flags: IA primitives
/// (free extra), clipping primitives, then fragment shader invocations.
/// `repr(C)` so `get_query_pool_results` can use this as the per-query stride
/// (ash's query_count is `data.len()`, stride is `size_of::<T>`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PipeStatRow {
    _ia_prims: u64,
    clip_prims: u64,
    frag_invocs: u64,
}
fn pipe_stat_flags() -> vk::QueryPipelineStatisticFlags {
    vk::QueryPipelineStatisticFlags::INPUT_ASSEMBLY_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::CLIPPING_PRIMITIVES
        | vk::QueryPipelineStatisticFlags::FRAGMENT_SHADER_INVOCATIONS
}
const PIPE_QUERY_COUNT: u32 = (PipeStatPass::COUNT * FRAMES_IN_FLIGHT as usize) as u32;

/// Delayed-slot index of `pass` in `slot`'s query range.
pub(super) fn pipe_stat_query(slot: usize, pass: PipeStatPass) -> u32 {
    slot as u32 * PipeStatPass::COUNT as u32 + pass as u32
}

/// Full-res overdraw: fragment invocations per render-extent pixel.
pub(super) fn overdraw_ratio(frag_full: u64, width: u32, height: u32) -> f64 {
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 {
        0.0
    } else {
        frag_full as f64 / pixels as f64
    }
}

/// Per-pass `VK_QUERY_TYPE_PIPELINE_STATISTICS` pool (fragment invocations +
/// clipping primitives). Same delayed-slot readback as [`GpuTimer`]: a slot is
/// read after its fence wait, then reset (host reset when the device has it,
/// otherwise `vkCmdResetQueryPool` outside the render pass).
pub(super) struct GpuPipeStats {
    pool: vk::QueryPool,
    host_reset: bool,
    primed: [bool; FRAMES_IN_FLIGHT as usize],
}

impl GpuPipeStats {
    pub(super) fn new(device: &ash::Device, supported: bool, host_reset: bool) -> Self {
        let pool = if supported {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::PIPELINE_STATISTICS)
                .query_count(PIPE_QUERY_COUNT)
                .pipeline_statistics(pipe_stat_flags());
            unsafe {
                device
                    .create_query_pool(&info, None)
                    .expect("Failed to create pipeline statistics query pool")
            }
        } else {
            vk::QueryPool::null()
        };
        Self {
            pool,
            host_reset,
            primed: [false; FRAMES_IN_FLIGHT as usize],
        }
    }

    fn enabled(&self) -> bool {
        self.pool != vk::QueryPool::null()
    }

    /// Reads `slot`'s prior counters. Caller has waited the slot fence.
    /// Returns `(frag[5], prims_full)` or `None` if the slot was never recorded
    /// or the read failed (unsupported / not ready).
    pub(super) unsafe fn read_into(
        &mut self,
        device: &ash::Device,
        slot: usize,
    ) -> Option<([u64; PipeStatPass::COUNT], u64)> {
        if !self.enabled() || !self.primed[slot] {
            return None;
        }
        let mut raw = [PipeStatRow::default(); PipeStatPass::COUNT];
        let first = pipe_stat_query(slot, PipeStatPass::OpaqueFull);
        let read = unsafe {
            device.get_query_pool_results(self.pool, first, &mut raw, vk::QueryResultFlags::TYPE_64)
        };
        if read.is_err() {
            return None;
        }
        let mut frag = [0u64; PipeStatPass::COUNT];
        for pass in 0..PipeStatPass::COUNT {
            frag[pass] = raw[pass].frag_invocs;
        }
        let prims_full = raw[PipeStatPass::OpaqueFull as usize].clip_prims;
        Some((frag, prims_full))
    }

    /// Reset `slot`'s queries before recording. Host reset after the fence wait
    /// when available; otherwise a command-buffer reset outside the render pass.
    pub(super) unsafe fn prepare(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if !self.enabled() {
            return;
        }
        let first = pipe_stat_query(slot, PipeStatPass::OpaqueFull);
        let count = PipeStatPass::COUNT as u32;
        unsafe {
            if self.host_reset {
                device.reset_query_pool(self.pool, first, count);
            } else {
                device.cmd_reset_query_pool(cmd, self.pool, first, count);
            }
        }
    }

    pub(super) unsafe fn begin_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        pass: PipeStatPass,
    ) {
        if !self.enabled() {
            return;
        }
        unsafe {
            device.cmd_begin_query(
                cmd,
                self.pool,
                pipe_stat_query(slot, pass),
                vk::QueryControlFlags::empty(),
            );
        }
    }

    pub(super) unsafe fn end_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        pass: PipeStatPass,
    ) {
        if !self.enabled() {
            return;
        }
        unsafe {
            device.cmd_end_query(cmd, self.pool, pipe_stat_query(slot, pass));
        }
    }

    pub(super) fn finish(&mut self, slot: usize) {
        if self.enabled() {
            self.primed[slot] = true;
        }
    }

    pub(super) unsafe fn destroy(&mut self, device: &ash::Device) {
        if self.enabled() {
            unsafe { device.destroy_query_pool(self.pool, None) };
            self.pool = vk::QueryPool::null();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_pass_ordinals_index_all_in_record_order() {
        // `GpuPass as usize` indexes the per-pass readback sink and ALL is the
        // record order; every pass also needs a distinct profile meter.
        let mut meters = Vec::new();
        for (i, pass) in GpuPass::ALL.into_iter().enumerate() {
            assert_eq!(pass as usize, i, "{pass:?} is out of order in GpuPass::ALL");
            meters.push(pass.meter() as usize);
        }
        meters.sort_unstable();
        meters.dedup();
        assert_eq!(meters.len(), GpuPass::COUNT);
        assert_eq!(GPU_STAMPS, GpuPass::COUNT + 1);
        assert_eq!(
            QUERY_COUNT as usize,
            GPU_STAMPS * FRAMES_IN_FLIGHT as usize + 2 + 2 * FRAMES_IN_FLIGHT as usize
        );
        // Opaque split: stamped after each camera group, in draw order
        // (full-res, coarse LOD, then cutout). Combined `opaque` is summed
        // at readback and is not a GpuPass.
        assert_eq!(GpuPass::OpaqueFull as usize, GpuPass::Clear as usize + 1);
        assert_eq!(
            GpuPass::OpaqueLod as usize,
            GpuPass::OpaqueFull as usize + 1
        );
        assert_eq!(GpuPass::Cutout as usize, GpuPass::OpaqueLod as usize + 1);
        assert_eq!(GpuPass::Sky as usize, GpuPass::Cutout as usize + 1);
    }

    #[test]
    fn opaque_ms_sums_the_three_group_stamps() {
        let mut passes = [0.0f64; GpuPass::COUNT];
        passes[GpuPass::OpaqueFull as usize] = 0.03;
        passes[GpuPass::OpaqueLod as usize] = 0.05;
        passes[GpuPass::Cutout as usize] = 0.01;
        assert!((GpuPass::opaque_ms(&passes) - 0.09).abs() < 1e-12);
    }

    #[test]
    fn gpu_load_disabled_by_default() {
        let shared = GpuLoadShared::new();
        assert!(!shared.is_enabled());
        assert!(shared.load().is_none());
        shared.store(1.5, 0.25);
        assert!(
            shared.load().is_none(),
            "store must not publish while disabled"
        );
        shared.set_enabled(true);
        assert!(shared.load().is_none(), "enabling does not invent a sample");
        shared.store(1.5, 0.25);
        let g = shared.load().expect("enabled + stored");
        assert_eq!(g.frame_ms, 1.5);
        assert_eq!(g.gap_ms, 0.25);
        shared.set_enabled(false);
        assert!(shared.load().is_none());
        shared.set_enabled(true);
        assert!(
            shared.load().is_none(),
            "disable clears ready; re-enable waits for a fresh store"
        );
    }

    #[test]
    fn idle_gap_ms_converts_ticks_via_period() {
        // 1 ns/tick: 1_000_000 ticks = 1 ms.
        assert!((idle_gap_ms(10, 10 + 1_000_000, 1.0) - 1.0).abs() < 1e-12);
        // 1000 ns/tick (1 µs): 1000 ticks = 1 ms.
        assert!((idle_gap_ms(0, 1000, 1000.0) - 1.0).abs() < 1e-12);
        // Back-to-back stamps: zero idle.
        assert_eq!(idle_gap_ms(42, 42, 1.0), 0.0);
        // This submit started before the previous end (GPU overlap): zero idle.
        assert_eq!(idle_gap_ms(100, 50, 1.0), 0.0);
    }

    #[test]
    fn pipe_stat_query_is_slot_delayed_linear() {
        assert_eq!(pipe_stat_query(0, PipeStatPass::OpaqueFull), 0);
        assert_eq!(pipe_stat_query(0, PipeStatPass::Transparent), 4);
        assert_eq!(
            pipe_stat_query(1, PipeStatPass::OpaqueFull),
            PipeStatPass::COUNT as u32
        );
        assert_eq!(
            pipe_stat_query(1, PipeStatPass::Sky),
            PipeStatPass::COUNT as u32 + PipeStatPass::Sky as u32
        );
        assert_eq!(
            PIPE_QUERY_COUNT as usize,
            PipeStatPass::COUNT * FRAMES_IN_FLIGHT as usize
        );
        // Counter layout inside one query: IA prims, clip prims, frag invocs.
        assert_eq!(std::mem::size_of::<PipeStatRow>(), 3 * 8);
    }

    #[test]
    fn overdraw_ratio_divides_by_extent_pixels() {
        assert_eq!(overdraw_ratio(0, 100, 100), 0.0);
        assert_eq!(overdraw_ratio(10_000, 100, 100), 1.0);
        assert!((overdraw_ratio(25_000, 100, 100) - 2.5).abs() < 1e-12);
        assert_eq!(overdraw_ratio(99, 0, 10), 0.0);
        assert_eq!(overdraw_ratio(99, 10, 0), 0.0);
    }
}
