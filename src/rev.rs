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

/// Frame slot index (0 or 1); type-safe prevents raw-usize indexing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrameSlot(u8);

impl FrameSlot {
    /// The other frame (2FIF: exactly two slots).
    #[must_use]
    pub fn other(self) -> FrameSlot {
        FrameSlot(1 - self.0)
    }
    /// Crate-internal: create a slot from index.
    pub(crate) fn new(index: usize) -> FrameSlot {
        debug_assert!(index < 2);
        FrameSlot(index as u8)
    }
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Per-frame-in-flight resources, indexable only by FrameSlot (no usize impl).
pub struct PerSlot<T>([T; 2]);

impl<T> PerSlot<T> {
    pub fn new(pair: [T; 2]) -> Self {
        PerSlot(pair)
    }

    /// Iterate both frames for lifecycle passes.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &T> {
        self.0.iter()
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

    /// Parity resolver is its own inverse and never aliases.
    #[test]
    fn frame_slot_other_is_involution() {
        let a = FrameSlot::new(0);
        assert_eq!(a.other().other(), a);
        assert_ne!(a.other(), a);
    }
}
