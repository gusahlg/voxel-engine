//! Monotone revisions: single counter type shared by CPU and GPU timelines.

/// Monotone revision for timeline-semaphore synchronization.
/// [`crate::vk::timeline`] aliases TimelineValue to this type.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Rev(pub u64);

impl Rev {
    /// Initial revision (also initial timeline value).
    pub const START: Rev = Rev(0);

    pub fn raw(self) -> u64 {
        self.0
    }
}

/// GPU command-buffer / per-slot resource ring depth.
///
/// 3 pipelines the submit-to-completion round trip (doorbell + scheduling,
/// ~10 µs on an RTX 3070) behind the previous frame's execution. With 2 slots
/// every frame waits for N−1 before recording N+1, so that latency sits on
/// the critical path and tiny frames cannot exceed ~1 / (GPU work + submit).
/// 3 adds one frame of input-to-photon latency — negligible above 1000 FPS;
/// with vsync on, the frame loop waits one slot earlier in
/// `wait_slot_and_reclaim` so the effective depth stays 2 (a third in-flight
/// frame would add a full refresh of latency). Must be at least
/// [`SUBMIT_BATCH_MAX`] + 1 so an unpresented batch can sit unsubmitted
/// without wrapping onto a slot whose command buffer is still pending.
/// Tune here; every per-slot array and the main/render frame pool derive from
/// this constant.
///
/// Each extra slot duplicates offscreen color and depth, the MSAA resolve
/// target, bloom chain, spill, cloud LUT, VRS rate/history, cull output, UBO,
/// immediates, and a primary command buffer. Raise only when
/// [`SUBMIT_BATCH_MAX`] needs another slot of slack.
pub const FRAMES_IN_FLIGHT: u64 = 3;
const _: () = assert!(FRAMES_IN_FLIGHT >= 2);

/// Max unpresented frames coalesced into one `vkQueueSubmit2` when vsync is
/// off. Default 2; runtime `VOXEL_SUBMIT_BATCH` (1 = submit every frame, as
/// before batching). A presented frame always submits on its own after any
/// pending batch. [`FRAMES_IN_FLIGHT`] must cover a full batch plus the slot
/// being recorded next.
pub const SUBMIT_BATCH_MAX: usize = 2;
const _: () = assert!(FRAMES_IN_FLIGHT as usize >= SUBMIT_BATCH_MAX + 1);

/// Frame slot index in `0..FRAMES_IN_FLIGHT`; type-safe prevents raw-usize indexing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameSlot(u8);

impl FrameSlot {
    /// Next slot in the in-flight ring (historically "the other" of two slots).
    #[must_use]
    pub fn other(self) -> FrameSlot {
        FrameSlot((self.0 + 1) % FRAMES_IN_FLIGHT as u8)
    }
    /// Crate-internal: create a slot from index.
    pub(crate) fn new(index: usize) -> FrameSlot {
        debug_assert!(index < FRAMES_IN_FLIGHT as usize);
        FrameSlot(index as u8)
    }
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Per-frame-in-flight resources, indexable only by FrameSlot (no usize impl).
pub struct PerSlot<T>([T; FRAMES_IN_FLIGHT as usize]);

impl<T> PerSlot<T> {
    pub fn new(slots: [T; FRAMES_IN_FLIGHT as usize]) -> Self {
        PerSlot(slots)
    }

    /// Iterate every in-flight slot for lifecycle passes.
    #[allow(dead_code)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
    }

    /// Mutable iterate every in-flight slot for teardown.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.0.iter_mut()
    }
}

impl<T> std::ops::Index<FrameSlot> for PerSlot<T> {
    type Output = T;
    fn index(&self, s: FrameSlot) -> &T {
        &self.0[s.index()]
    }
}

impl<T> std::ops::IndexMut<FrameSlot> for PerSlot<T> {
    fn index_mut(&mut self, s: FrameSlot) -> &mut T {
        &mut self.0[s.index()]
    }
}

/// Value with revision timestamp; staleness is orderable.
pub struct Stamped2<T> {
    pub value: T,
    pub at: Rev,
}

/// Keyed reactive cache: recomputes entries when revision advances.
pub struct DerivedMap<K, V> {
    entries: std::collections::HashMap<K, Stamped2<V>>,
}

impl<K, V> Default for DerivedMap<K, V> {
    fn default() -> Self {
        Self {
            entries: std::collections::HashMap::new(),
        }
    }
}

impl<K: Eq + std::hash::Hash + Clone, V> DerivedMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_recompute(&mut self, key: K, rev: Rev, compute: impl FnOnce() -> V) -> &V {
        let stale = self.entries.get(&key).is_none_or(|s| s.at < rev);
        if stale {
            self.entries.insert(
                key.clone(),
                Stamped2 {
                    value: compute(),
                    at: rev,
                },
            );
        }
        &self.entries[&key].value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_map_recomputes_only_when_rev_advances() {
        let mut calls = 0;
        let mut m: DerivedMap<u32, u32> = DerivedMap::new();

        assert_eq!(
            *m.get_or_recompute(1, Rev(5), || {
                calls += 1;
                10
            }),
            10
        );
        assert_eq!(calls, 1, "first get always computes");

        assert_eq!(
            *m.get_or_recompute(1, Rev(5), || {
                calls += 1;
                20
            }),
            10
        );
        assert_eq!(calls, 1, "same rev must not recompute");

        assert_eq!(
            *m.get_or_recompute(1, Rev(6), || {
                calls += 1;
                30
            }),
            30
        );
        assert_eq!(calls, 2, "newer rev must recompute");

        // Different keys are independent.
        assert_eq!(
            *m.get_or_recompute(2, Rev(0), || {
                calls += 1;
                99
            }),
            99
        );
        assert_eq!(calls, 3);
    }

    #[test]
    fn rev_is_ordered_by_join_under_max() {
        let a = Rev(3);
        let b = Rev(7);
        assert!(a < b);
        assert_eq!(a.max(b), b);
    }

    /// The ring walk visits every slot once and returns to the start.
    #[test]
    fn frame_slot_other_walks_the_ring() {
        let start = FrameSlot::new(0);
        let mut s = start;
        let mut seen = [false; FRAMES_IN_FLIGHT as usize];
        for _ in 0..FRAMES_IN_FLIGHT {
            assert!(
                !seen[s.index()],
                "ring must not repeat a slot before wrapping"
            );
            seen[s.index()] = true;
            s = s.other();
        }
        assert_eq!(s, start);
        assert!(seen.iter().all(|&v| v));
        assert_ne!(start.other(), start);
    }
}
