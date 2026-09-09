use super::timeline::TimelineValue;

/// A deferred-reclaim queue: items stamped with their last possible GPU use.
/// [`collect`](Self::collect) only reclaims items the GPU has provably passed.
/// Allocator-agnostic: yields items for the caller to free.
pub struct RetireQueue<T> {
    entries: std::collections::VecDeque<(TimelineValue, T)>,
}

impl<T> RetireQueue<T> {
    pub fn new() -> Self {
        Self {
            entries: std::collections::VecDeque::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Retires `item`, stamped with the timeline value that last could
    /// reference it.
    pub fn push(&mut self, done_at: TimelineValue, item: T) {
        self.entries.push_back((done_at, item));
    }

    /// Drains entries whose GPU use has completed, calling `f` on each.
    ///
    /// `current` is the completed (signalled) timeline counter. A stamp that
    /// is reserved but not yet signalled compares greater than `current` and
    /// stays queued; this never waits.
    pub fn collect(&mut self, current: TimelineValue, mut f: impl FnMut(T)) {
        while let Some((stamp, _)) = self.entries.front() {
            if *stamp > current {
                break;
            }
            let (_, item) = self.entries.pop_front().unwrap();
            f(item);
        }
    }

    /// Drains everything, calling `f` on each item.
    pub fn collect_all(&mut self, mut f: impl FnMut(T)) {
        for (_, item) in self.entries.drain(..) {
            f(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::timeline::TimelineValue;
    use super::RetireQueue;

    #[test]
    fn retire_queue_reclaims_when_the_timeline_reaches_the_stamp() {
        // An entry stamped at value N reclaims once the timeline counter has
        // reached N (stamp <= current).
        let v = TimelineValue::from_raw_for_test;
        let mut q: RetireQueue<u32> = RetireQueue::new();
        assert!(q.is_empty());
        q.push(v(1), 100);
        q.push(v(2), 101);
        q.push(v(4), 103);
        assert!(!q.is_empty());

        // current = 0: nothing has completed yet.
        let mut freed = Vec::new();
        q.collect(v(0), |x| freed.push(x));
        assert_eq!(freed, Vec::<u32>::new());

        // current = 1: stamp 1 drains; stamp 2 stays.
        let mut freed = Vec::new();
        q.collect(v(1), |x| freed.push(x));
        assert_eq!(freed, vec![100]);

        // current = 3: stamp 2 (2 <= 3) drains; stamp 4 stays.
        let mut freed = Vec::new();
        q.collect(v(3), |x| freed.push(x));
        assert_eq!(freed, vec![101]);

        // collect_all drains the remainder (stamp 4, not yet reached).
        let mut freed = Vec::new();
        q.collect_all(|x| freed.push(x));
        assert_eq!(freed, vec![103]);
        assert!(q.is_empty());
    }

    #[test]
    fn retire_queue_holds_a_reserved_but_unsubmitted_stamp() {
        // A free stamped at last_reserved (frame reserved, not yet submitted)
        // must not reclaim until the completed counter reaches that value.
        // collect only compares; it must not wait.
        let v = TimelineValue::from_raw_for_test;
        let mut q: RetireQueue<u32> = RetireQueue::new();
        let reserved = v(5);
        q.push(reserved, 42);

        let mut freed = Vec::new();
        q.collect(v(3), |x| freed.push(x));
        assert_eq!(freed, Vec::<u32>::new(), "unsubmitted stamp is not done");
        assert!(!q.is_empty());

        let mut freed = Vec::new();
        q.collect(v(4), |x| freed.push(x));
        assert!(
            freed.is_empty(),
            "still waiting for the reserved value itself"
        );

        let mut freed = Vec::new();
        q.collect(reserved, |x| freed.push(x));
        assert_eq!(freed, vec![42]);
        assert!(q.is_empty());
    }
}
