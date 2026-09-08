//! Timestamp-query GPU pass timer (`GpuTimer`, `GpuPass`) and its stamps.
//! Split out of `mod.rs` so later work can touch profiling without opening
//! the renderer setup.

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
    Opaque,
    Sky,
    Cubes,
    Lines,
    Shadows,
    Transparent,
    Overlay,
    /// End of the scene pass: `cmd_end_rendering` (where the MSAA color
    /// resolve executes) and the offscreen/depth-rest finalize transitions.
    Resolve,
    /// The VRS classify dispatch (end of frame, after depth rests; stamped
    /// only when it runs).
    Vrs,
    /// The TAA resolve compute (stamped only when it runs).
    Taa,
    /// Exposure metering reduce + finalize (stamped only when it runs).
    Exposure,
    /// The bloom chain — the render-command tail. Without this closing stamp
    /// everything after the last boundary silently vanishes from the report.
    Bloom,
}

impl GpuPass {
    pub(super) const ALL: [GpuPass; 16] = [
        GpuPass::Copies,
        GpuPass::Cull,
        GpuPass::ShadowMap,
        GpuPass::Clear,
        GpuPass::Opaque,
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
            GpuPass::Opaque => Meter::GpuOpaque,
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
}

/// One start timestamp plus one boundary per pass.
const GPU_STAMPS: usize = GpuPass::COUNT + 1;
/// The present copy's start/end pair lives after the per-slot render ranges.
/// One pair suffices: a new copy is only recorded once the previous one has
/// retired (`decide_present` probes/waits it), so its stamps are read first.
const COPY_STAMP_BASE: u32 = (GPU_STAMPS * FRAMES_IN_FLIGHT as usize) as u32;
const QUERY_COUNT: u32 = COPY_STAMP_BASE + 2;

/// Per-pass GPU timing via a timestamp query pool: a start timestamp plus one
/// after each recorded pass. Only the passes that actually run write a stamp,
/// and the label written alongside each stamp keeps deltas attributable even
/// when a frame skips passes (no 3D, VRS off). A slot's results are read one
/// cycle later, after its fence is waited, so the read never stalls — and
/// because that wait is in render order, consecutive `read_into` calls are
/// consecutive rendered frames (possibly different slots). Their timestamps
/// share the device clock, so `start(N) - end(N-1)` is the idle gap before
/// this submit.
///
/// `count`/`label` are [`Cell`]s so a mark needs only `&self`: the render pass
/// holds an immutable `&Renderer` while recording, and all timer state is
/// touched on the single render thread. A null pool (hardware without timestamp
/// support) makes every method a no-op.
pub(super) struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per tick (`limits.timestampPeriod`).
    period_ns: f32,
    /// Whether each slot holds completed timestamps to read back.
    primed: [bool; FRAMES_IN_FLIGHT as usize],
    /// Stamps written for each slot's most recent recording (incl. the start).
    count: [std::cell::Cell<u32>; FRAMES_IN_FLIGHT as usize],
    /// The pass that ended at each stamp (index `i` labels the span `i-1..i`).
    label: [[std::cell::Cell<GpuPass>; GPU_STAMPS]; FRAMES_IN_FLIGHT as usize],
    /// Whether the present-copy pair holds a completed range to read back.
    copy_primed: bool,
    /// Last stamp of the previously *read* render submit (raw ticks), used to
    /// compute the idle gap before the next readable frame. Cleared when a
    /// readback is unavailable so a later start is not compared across a hole.
    prev_end: Option<u64>,
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
    pub(super) fn new(device: &ash::Device, supported: bool, period_ns: f32) -> Self {
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
            primed: [false; FRAMES_IN_FLIGHT as usize],
            count: std::array::from_fn(|_| std::cell::Cell::new(0)),
            label: std::array::from_fn(|_| {
                std::array::from_fn(|_| std::cell::Cell::new(GpuPass::Opaque))
            }),
            copy_primed: false,
            prev_end: None,
        }
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

    /// Resets `slot`'s queries and writes the start timestamp. Must be recorded
    /// outside any render pass.
    pub(super) unsafe fn begin(&self, device: &ash::Device, cmd: vk::CommandBuffer, slot: usize) {
        if !self.enabled() {
            return;
        }
        let base = slot as u32 * GPU_STAMPS as u32;
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, base, GPU_STAMPS as u32);
            device.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::TOP_OF_PIPE, self.pool, base);
        }
        self.count[slot].set(1);
    }

    /// Writes a boundary timestamp closing `pass` for `slot`. Recorded inside
    /// the render pass; needs only `&self` (interior-mutable bookkeeping).
    pub(super) unsafe fn mark(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        slot: usize,
        pass: GpuPass,
    ) {
        if !self.enabled() {
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
    pub(super) unsafe fn begin_copy(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if !self.enabled() {
            return;
        }
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, COPY_STAMP_BASE, 2);
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
            GPU_STAMPS * FRAMES_IN_FLIGHT as usize + 2
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
}
