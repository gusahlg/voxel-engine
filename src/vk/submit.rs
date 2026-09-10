//! Submit batching for uncapped unpresented frames: pending ring, flush policy,
//! and empty-submit bench path. Split out of `frame_loop` (move only).

use ash::vk;

use crate::skeleton::FrameSlot;

use super::Renderer;
use super::buffers::{FRAMES_IN_FLIGHT, SUBMIT_BATCH_MAX};
use super::render_client::RenderReturn;
use super::timeline::{RENDER_SIGNAL_STAGES, RenderSubmit, TimelineValue};

/// Slot whose `render_value` [`Renderer::wait_slot_and_reclaim`] waits before
/// recording into `slot`.
///
/// Uncapped: wait only the slot being reused, so all [`FRAMES_IN_FLIGHT`]
/// command buffers can sit on the GPU and hide submit latency. Vsync: wait the
/// slot that is two frames old so the effective depth stays 2 — a third
/// in-flight frame would add a full refresh of latency. Timeline values are
/// monotone, so that wait also proves `slot` itself is idle.
fn reclaim_wait_slot(slot: usize, vsync: bool) -> usize {
    if vsync {
        (slot + FRAMES_IN_FLIGHT as usize - 2) % FRAMES_IN_FLIGHT as usize
    } else {
        slot
    }
}

/// Recorded but not yet submitted: one entry per deferred unpresented frame.
pub(crate) struct PendingSubmit {
    slot: usize,
    cmd: vk::CommandBuffer,
    extra_wait: Option<(TimelineValue, vk::PipelineStageFlags2)>,
    /// Value reserved by `begin_render`; the batch signals the last entry's.
    signal: TimelineValue,
}

pub(crate) fn fold_transfer_wait(
    a: Option<(TimelineValue, vk::PipelineStageFlags2)>,
    b: Option<(TimelineValue, vk::PipelineStageFlags2)>,
) -> Option<(TimelineValue, vk::PipelineStageFlags2)> {
    match (a, b) {
        (Some((v, s)), Some((w, t))) => Some((v.max(w), s | t)),
        (a, b) => a.or(b),
    }
}

/// Whether the just-recorded frame may join the pending batch (`Defer`) or
/// must be submitted this call (`Flush`). `pending_count` includes the current
/// frame. A presented / vsync-on frame always `Flush`es (the caller submits
/// any already-pending batch first, then this frame on its own).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubmitBatchAction {
    Defer,
    Flush,
}

fn submit_batch_action(
    uncapped: bool,
    present: bool,
    pending_count: usize,
    limit: usize,
) -> SubmitBatchAction {
    if !uncapped || present || pending_count >= limit {
        SubmitBatchAction::Flush
    } else {
        SubmitBatchAction::Defer
    }
}

/// True when `wait_slot` still has an unsubmitted command buffer. Waiting
/// then would block on a value that has not been queued (or, if `render_value`
/// was left at the slot's previous use, return immediately and reset a CB
/// still in the pending list).
fn pending_blocks_wait(pending_slots: impl IntoIterator<Item = usize>, wait_slot: usize) -> bool {
    pending_slots.into_iter().any(|s| s == wait_slot)
}

/// Consecutive short slot waits that clear [`GpuBoundState`].
const GPU_BOUND_CLEAR_AFTER: u8 = 8;
/// Slot-wait duration that counts as GPU-bound (eager flush). Shorter waits
/// (a few microseconds while the in-flight batch finishes inside the submit
/// floor) stay batched. Overridden by `VOXEL_EAGER_FLUSH_US` (read once).
const EAGER_FLUSH_WAIT_US: u64 = 15;

/// Adaptive submit-batch flush: eager while the GPU is the bottleneck so the
/// next batch is in-flight before the CPU records, batched otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GpuBoundState {
    gpu_bound: bool,
    idle_waits: u8,
}

impl GpuBoundState {
    /// Sets GPU-bound when the slot wait took at least [`eager_flush_wait`].
    /// Shorter waits count as idle for the [`GPU_BOUND_CLEAR_AFTER`] streak.
    fn note_wait(&mut self, waited: std::time::Duration) {
        if waited >= eager_flush_wait() {
            self.gpu_bound = true;
            self.idle_waits = 0;
        } else if self.gpu_bound {
            self.idle_waits = self.idle_waits.saturating_add(1);
            if self.idle_waits >= GPU_BOUND_CLEAR_AFTER {
                self.gpu_bound = false;
                self.idle_waits = 0;
            }
        }
    }
}

/// `VOXEL_SUBMIT_BATCH` (integer ≥ 1): command buffers per `vkQueueSubmit2`
/// for unpresented uncapped frames. Unset uses [`SUBMIT_BATCH_MAX`]. Clamped
/// to `1..=FRAMES_IN_FLIGHT-1` so the ring always has a free slot. Read once
/// at renderer creation.
pub(crate) fn submit_batch_limit() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        let parsed = std::env::var("VOXEL_SUBMIT_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(SUBMIT_BATCH_MAX);
        parsed.clamp(1, FRAMES_IN_FLIGHT as usize - 1)
    })
}

/// Slot-wait duration that enables eager flush. `VOXEL_EAGER_FLUSH_US` if
/// set and parseable, otherwise [`EAGER_FLUSH_WAIT_US`]. Read once.
fn eager_flush_wait() -> std::time::Duration {
    static THRESHOLD: std::sync::OnceLock<std::time::Duration> = std::sync::OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        let us = std::env::var("VOXEL_EAGER_FLUSH_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(EAGER_FLUSH_WAIT_US);
        std::time::Duration::from_micros(us)
    })
}

impl Renderer {
    /// Empty command-buffer submit used by `VOXEL_BENCH_EMPTY=K`: wait the slot,
    /// record K distinct empty primaries (begin/end only), one `vkQueueSubmit2`
    /// with K command buffers and the usual single timeline signal, skip present.
    /// Slot wait + timeline signal keep shutdown and FIF reuse intact.
    pub(super) fn draw_empty_submit(&mut self) {
        let slot = self.slot;
        crate::profile::count(crate::profile::Counter::Rendered);
        self.wait_slot_and_reclaim(slot);
        let k = self.empty_submit.max(1) as usize;
        let primary = self.slots[FrameSlot::new(slot)].cmd;
        let extra_n = k - 1;
        let extra_base = slot * extra_n;
        let mut cmds = Vec::with_capacity(k);
        cmds.push(primary);
        if extra_n > 0 {
            cmds.extend_from_slice(&self.empty_extra[extra_base..extra_base + extra_n]);
        }
        unsafe {
            let device = &self.device.device;
            for &cmd in &cmds {
                device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                    .expect("command buffer reset failed");
                device
                    .begin_command_buffer(
                        cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .expect("begin command buffer failed");
                device
                    .end_command_buffer(cmd)
                    .expect("end command buffer failed");
            }
        }
        let rs = self.timeline.begin_render(cmds[0]);
        {
            let _p = crate::profile::scope(crate::profile::Meter::Submit);
            let extra_wait = self
                .pending_transfer_wait
                .take()
                .map(|(value, stages)| (self.transfer_lane.semaphore(), value, stages));
            let completion = unsafe {
                rs.submit_bufs(
                    &self.device.device,
                    self.device.graphics_queue,
                    &self.timeline,
                    &cmds,
                    extra_wait,
                    RENDER_SIGNAL_STAGES,
                )
            };
            self.slots[FrameSlot::new(slot)].render_value = completion.value();
            self.last_render_value = completion.value();
            crate::profile::count(crate::profile::Counter::Submits);
        }
        self.slot = (self.slot + 1) % FRAMES_IN_FLIGHT as usize;
    }
    /// Waits until the slot's last render has completed (GPU is done with its
    /// command buffer and immediate buffer), then reclaims retired GPU memory
    /// whose last possible use the timeline has reached. Publishes the
    /// one-cycle-late VRS mix and cull geometry gauges after the wait.
    ///
    /// With vsync on, waits [`reclaim_wait_slot`] (one slot earlier than the
    /// reuse target) so the effective in-flight depth stays 2.
    ///
    /// Flushes a still-pending batch that occupies `slot` before waiting:
    /// the wait must never target a timeline value that has not been submitted,
    /// and a deferred frame's host-visible slot (uniforms, cull buffers, query
    /// readback) must not be overwritten until its batch signals.
    ///
    /// When the GPU is the bottleneck ([`GpuBoundState`]: last slot wait
    /// ≥ [`eager_flush_wait`]), flushes any deferred batch unconditionally so
    /// the GPU stays fed while the CPU records. When CPU-bound, flush only if
    /// the wait would block, preserving two-frames-per-submit.
    pub(super) fn wait_slot_and_reclaim(&mut self, slot: usize) {
        let vsync = self.vsync.current();
        let wait_slot = reclaim_wait_slot(slot, vsync);
        // Never wait on a timeline value that has not been submitted: if this
        // slot (or the vsync wait-slot) is still in the pending batch, flush
        // first. With `FRAMES_IN_FLIGHT >= SUBMIT_BATCH_MAX + 1` this is a
        // safety net, not the steady-state path.
        if pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), wait_slot)
            || pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), slot)
        {
            self.flush_pending_submits();
        }
        assert!(
            !pending_blocks_wait(self.pending_submits.iter().map(|p| p.slot), wait_slot),
            "wait_slot_and_reclaim must not wait on an unsubmitted frame"
        );
        let value = self.slots[FrameSlot::new(wait_slot)].render_value;
        let completed = unsafe { self.timeline.counter(&self.device.device) };
        let would_block = completed < value;
        // GPU-bound: flush before waiting so the next batch is queued while
        // this slot completes. CPU-bound: flush only if this wait would block.
        if self.gpu_bound.gpu_bound || would_block {
            self.flush_pending_submits();
        }
        let device = &self.device.device;
        unsafe {
            {
                let _p = crate::profile::scope(crate::profile::Meter::Fence);
                let wait_start = std::time::Instant::now();
                if vsync {
                    self.timeline.wait(device, value);
                } else {
                    const FENCE_SPIN_BUDGET: std::time::Duration =
                        std::time::Duration::from_micros(200);
                    self.timeline.wait_spin(device, value, FENCE_SPIN_BUDGET);
                }
                self.gpu_bound.note_wait(wait_start.elapsed());
            }
            self.publish_vrs_mix(slot);
            self.publish_cull_stats(slot);
            // Staging-pool reclaim is once per frame even when the allocator
            // retire queues are empty: stale worker builds never touch those.
            let transfer_current = self.transfer_lane.counter(device);
            self.mesh_staging.reclaim(
                self.slots[FrameSlot::new(wait_slot)].render_value,
                transfer_current,
            );
            // Nothing retired (the steady state): skip the two counter reads
            // and the queue drains, which would find nothing to reclaim.
            if !self.mesh_res.has_garbage()
                && self.retired_textures.is_empty()
                && !self.quad_ibo.has_garbage()
                && !self.block_textures.has_garbage()
                && !self.materials.has_garbage()
            {
                return;
            }
            let _p = crate::profile::scope(crate::profile::Meter::Reclaim);
            let current = self.timeline.counter(device);
            // Retired allocations return to the main-owned allocator freelist;
            // staging-block shrink happens main-side after it reclaims them.
            let ret = &self.ret;
            self.mesh_res
                .collect(current, &mut |a| drop(ret.send(RenderReturn::FreeAlloc(a))));
            self.retired_textures
                .collect(current, |mut tex| tex.destroy(device));
            self.block_textures.collect(device, current);
            self.materials.collect(device, current);
            // Superseded quad IBO buffers are render-owned raw buffers (not
            // allocator suballocations), so destroy them here rather than shipping
            // them back to main's freelist.
            self.quad_ibo.collect(device, current);
            // Staging submitted on a separate transfer queue is
            // reclaimed against the LANE's own timeline, not the render
            // one — a non-blocking probe (never waited: nothing here may
            // stall this reclaim pass on the transfer queue's progress).
            if let Some(transfer_current) = transfer_current {
                self.mesh_res.collect_transfer(transfer_current, &mut |a| {
                    drop(ret.send(RenderReturn::FreeAlloc(a)))
                });
                self.quad_ibo.collect_transfer(device, transfer_current);
                self.block_textures
                    .collect_transfer(device, transfer_current);
                self.materials.collect_transfer(device, transfer_current);
            }
        }
    }
    /// Submits the recorded command buffer, or defers it into the pending
    /// batch. Uncapped unpresented frames join the batch until it reaches
    /// the runtime limit (`VOXEL_SUBMIT_BATCH`) or a flush condition. A
    /// presented / recreate / vsync-on frame flushes any pending batch first,
    /// then submits on its own (batch-size-1 semantics).
    ///
    /// Transfer-lane waits captured at record time ride the submit that
    /// actually queues the command buffer: a batch waits on the max value
    /// among its frames.
    pub(super) fn submit_or_defer(&mut self, rs: RenderSubmit, slot: usize, present: bool) {
        let extra_wait = self.pending_transfer_wait.take();
        let uncapped = !self.vsync.current();
        // Presented / recreate / vsync: pending batch first, then this frame
        // alone. `submit_batch_action` returns Flush for these, so the second
        // flush submits the just-pushed frame as its own `vkQueueSubmit2`.
        if present || !uncapped {
            self.flush_pending_submits();
        }
        let (signal, cmd) = rs.into_parts();
        self.pending_submits.push(PendingSubmit {
            slot,
            cmd,
            extra_wait,
            signal,
        });
        if submit_batch_action(
            uncapped,
            present,
            self.pending_submits.len(),
            self.submit_batch_limit,
        ) == SubmitBatchAction::Flush
        {
            self.flush_pending_submits();
        }
    }

    /// One `vkQueueSubmit2` for every pending command buffer, in frame order,
    /// with a single timeline signal at the last frame's reserved value.
    /// Every slot in the batch records that value as `render_value`, so
    /// reclamation waits for the whole batch. No-op when the list is empty.
    pub(super) fn flush_pending_submits(&mut self) {
        if self.pending_submits.is_empty() {
            return;
        }
        let extra_wait = self
            .pending_submits
            .iter()
            .filter_map(|p| p.extra_wait)
            .fold(None, |acc, w| fold_transfer_wait(acc, Some(w)))
            .map(|(value, stages)| (self.transfer_lane.semaphore(), value, stages));
        let signal = self
            .pending_submits
            .last()
            .expect("non-empty pending batch")
            .signal;
        let cmds: Vec<vk::CommandBuffer> = self.pending_submits.iter().map(|p| p.cmd).collect();
        let completion = unsafe {
            self.timeline.submit_render(
                &self.device.device,
                self.device.graphics_queue,
                &cmds,
                signal,
                extra_wait,
                RENDER_SIGNAL_STAGES,
            )
        };
        crate::profile::count(crate::profile::Counter::Submits);
        let done = completion.value();
        for p in self.pending_submits.drain(..) {
            self.slots[FrameSlot::new(p.slot)].render_value = done;
        }
        self.last_render_value = done;
    }
}
#[cfg(test)]
mod tests {
    use super::{
        FRAMES_IN_FLIGHT, GPU_BOUND_CLEAR_AFTER, GpuBoundState, SUBMIT_BATCH_MAX,
        SubmitBatchAction, pending_blocks_wait, reclaim_wait_slot, submit_batch_action,
    };

    #[test]
    fn reclaim_wait_slot_stays_at_two_deep_when_vsync() {
        for slot in 0..FRAMES_IN_FLIGHT as usize {
            assert_eq!(reclaim_wait_slot(slot, false), slot);
            let waited = reclaim_wait_slot(slot, true);
            // Two frames old: not the slot being reused (unless FIF == 2) and
            // not the just-submitted previous slot.
            let prev = (slot + FRAMES_IN_FLIGHT as usize - 1) % FRAMES_IN_FLIGHT as usize;
            assert_ne!(waited, prev, "vsync must not serialize to 1 in flight");
            assert_eq!(
                waited,
                (slot + FRAMES_IN_FLIGHT as usize - 2) % FRAMES_IN_FLIGHT as usize
            );
        }
    }

    #[test]
    fn submit_batch_action_defers_uncapped_unpresented_until_limit() {
        let limit = SUBMIT_BATCH_MAX;
        assert_eq!(
            submit_batch_action(true, false, 1, limit),
            SubmitBatchAction::Defer,
            "first unpresented frame of a batch of {limit} must defer"
        );
        assert_eq!(
            submit_batch_action(true, false, limit, limit),
            SubmitBatchAction::Flush,
            "reaching the limit must flush"
        );
        assert_eq!(
            submit_batch_action(true, false, limit + 1, limit),
            SubmitBatchAction::Flush
        );
    }

    #[test]
    fn submit_batch_action_flush_when_presented_or_capped_or_limit_one() {
        assert_eq!(
            submit_batch_action(true, true, 1, 2),
            SubmitBatchAction::Flush,
            "a presented frame never joins a batch"
        );
        assert_eq!(
            submit_batch_action(false, false, 1, 2),
            SubmitBatchAction::Flush,
            "vsync/capped is batch-size-1"
        );
        assert_eq!(
            submit_batch_action(true, false, 1, 1),
            SubmitBatchAction::Flush,
            "VOXEL_SUBMIT_BATCH=1 is today's per-frame submit"
        );
    }

    #[test]
    fn fif_covers_a_full_batch_plus_the_slot_being_recorded() {
        assert!(FRAMES_IN_FLIGHT as usize > SUBMIT_BATCH_MAX);
    }

    /// Apply the wait-path flush rules: safety-net if `wait_slot` / `slot` is
    /// still pending, then flush the rest of the batch when GPU-bound or when
    /// the wait would block (`!slot_reached`).
    fn apply_wait_path_flush(
        pending: &mut Vec<usize>,
        wait_slot: usize,
        slot: usize,
        slot_reached: bool,
        gpu_bound: bool,
    ) {
        if pending_blocks_wait(pending.iter().copied(), wait_slot)
            || pending_blocks_wait(pending.iter().copied(), slot)
        {
            pending.clear();
        }
        if gpu_bound || !slot_reached {
            pending.clear();
        }
    }

    /// Walk the slot ring under the production defer/flush rules.
    ///
    /// After the wait-path flush the slot being reused must be free —
    /// otherwise the wait would target a value that has not been submitted
    /// (or, worse, the slot's previous already-signalled value and reset an
    /// unsubmitted command buffer).
    fn simulate_ring(
        uncapped: bool,
        limit: usize,
        present: impl Fn(usize) -> bool,
        slot_reached: bool,
        gpu_bound: bool,
    ) {
        let fif = FRAMES_IN_FLIGHT as usize;
        let mut pending: Vec<usize> = Vec::new();
        let mut slot = 0usize;
        for i in 0..64 {
            let wait_slot = reclaim_wait_slot(slot, !uncapped);
            apply_wait_path_flush(&mut pending, wait_slot, slot, slot_reached, gpu_bound);
            if gpu_bound || !slot_reached {
                assert!(
                    pending.is_empty(),
                    "gpu-bound or blocking wait must flush pending (frame {i}, slot {slot})"
                );
            }
            assert!(
                !pending_blocks_wait(pending.iter().copied(), slot),
                "slot {slot} still pending at reuse (frame {i}, pending {pending:?}); \
                 FRAMES_IN_FLIGHT must be SUBMIT_BATCH_MAX + 1"
            );
            let will_present = present(i);
            if will_present || !uncapped {
                pending.clear();
            }
            pending.push(slot);
            if submit_batch_action(uncapped, will_present, pending.len(), limit)
                == SubmitBatchAction::Flush
            {
                pending.clear();
            }
            slot = (slot + 1) % fif;
        }
    }

    #[test]
    fn slot_ring_stays_available_under_uncapped_batching() {
        for slot_reached in [true, false] {
            for gpu_bound in [false, true] {
                simulate_ring(
                    true,
                    SUBMIT_BATCH_MAX,
                    |i| i % 5 == 0,
                    slot_reached,
                    gpu_bound,
                );
                simulate_ring(true, SUBMIT_BATCH_MAX, |_| false, slot_reached, gpu_bound);
                simulate_ring(true, 1, |_| false, slot_reached, gpu_bound);
                simulate_ring(false, SUBMIT_BATCH_MAX, |_| false, slot_reached, gpu_bound);
                simulate_ring(true, SUBMIT_BATCH_MAX, |_| true, slot_reached, gpu_bound);
            }
        }
    }

    #[test]
    fn pending_slot_is_not_available_until_flushed() {
        assert!(!pending_blocks_wait(std::iter::empty(), 0));
        assert!(pending_blocks_wait([0], 0));
        assert!(!pending_blocks_wait([0], 1));
        assert!(pending_blocks_wait([0, 1], 1));
        // Safety-net flush: once the blocking slot is submitted, reuse is
        // legal (the subsequent timeline wait covers the batch signal).
        let mut pending = vec![0, 1];
        assert!(pending_blocks_wait(pending.iter().copied(), 0));
        apply_wait_path_flush(&mut pending, 0, 0, true, false);
        assert!(!pending_blocks_wait(pending.iter().copied(), 0));
        assert!(
            pending.is_empty(),
            "safety-net flush submits the slot before wait_spin"
        );
    }

    #[test]
    fn a_pending_batch_as_large_as_the_ring_occupies_the_next_slot() {
        // Why FRAMES_IN_FLIGHT >= SUBMIT_BATCH_MAX + 1: a pending list of FIF
        // frames would occupy the slot about to be reused, and the wait path
        // would have to flush before waiting (the safety net in
        // `wait_slot_and_reclaim`). The extra slot keeps that path cold.
        let fif = FRAMES_IN_FLIGHT as usize;
        let pending: Vec<usize> = (0..fif).collect();
        assert!(
            pending_blocks_wait(pending.iter().copied(), 0),
            "reusing slot 0 while it is still pending is illegal"
        );
        let slack: Vec<usize> = (0..fif - 1).collect();
        assert!(
            !pending_blocks_wait(slack.iter().copied(), fif - 1),
            "a batch of FIF-1 leaves the next ring slot free"
        );
    }

    #[test]
    fn wait_keeps_pending_when_the_slot_is_already_complete() {
        let mut pending = vec![0];
        apply_wait_path_flush(&mut pending, 1, 1, true, false);
        assert_eq!(
            pending,
            vec![0],
            "already-reached slot must not flush a deferred batch"
        );
    }

    #[test]
    fn wait_flushes_pending_when_the_slot_wait_would_block() {
        let mut pending = vec![0];
        apply_wait_path_flush(&mut pending, 1, 1, false, false);
        assert!(
            pending.is_empty(),
            "blocking wait must flush the deferred batch first"
        );
    }

    #[test]
    fn wait_flushes_pending_while_gpu_bound_even_if_slot_is_complete() {
        let mut pending = vec![0];
        apply_wait_path_flush(&mut pending, 1, 1, true, true);
        assert!(
            pending.is_empty(),
            "gpu-bound frames flush the deferred batch before wait"
        );
    }

    #[test]
    fn gpu_bound_sets_on_long_wait_and_clears_after_short_waits() {
        let mut s = GpuBoundState::default();
        assert!(!s.gpu_bound);
        let short = std::time::Duration::from_micros(5);
        let long = std::time::Duration::from_micros(30);
        s.note_wait(short);
        assert!(!s.gpu_bound, "a 5 us wait does not set gpu_bound");
        s.note_wait(long);
        assert!(s.gpu_bound, "a 30 us wait sets gpu_bound");
        assert_eq!(s.idle_waits, 0);
        s.note_wait(long);
        assert!(s.gpu_bound);
        assert_eq!(s.idle_waits, 0, "another long wait resets the idle streak");
        for i in 1..GPU_BOUND_CLEAR_AFTER {
            s.note_wait(short);
            assert!(s.gpu_bound, "still bound after {i} short waits");
        }
        s.note_wait(short);
        assert!(
            !s.gpu_bound,
            "clears after {GPU_BOUND_CLEAR_AFTER} short waits"
        );
        assert_eq!(s.idle_waits, 0);
        s.note_wait(short);
        assert!(!s.gpu_bound);
    }
}
