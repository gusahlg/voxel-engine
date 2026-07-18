//! The producer manifest: a declarative description of what a producer
//! reads/writes, when it runs, and what it costs. From one [`Producer`]
//! declaration a later scheduler phase derives queue placement, pomset
//! edges, barriers, dispatch bounds, eviction participation, and law tests —
//! so the shape here must cover every producer's footprint, not just today's.

/// Signed detail level. A brick is always 16³ cells; its world extent is
/// 16·2^k. k=0: one cell = one world unit. k=-1: half blocks. k>0: LOD
/// aggregates. INVARIANT: nothing anywhere assumes k >= 0.
///
/// The canonical type: `mesh::Detail` re-exports this rather than defining
/// its own, and the app's `ident` module does the same — one type, not two
/// structurally identical ones.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Detail(pub i8);

impl Detail {
    /// Full-resolution: one cell = one world unit (k=0, scale 1.0).
    pub const FULL: Detail = Detail(0);

    /// Encodable range at the GPU boundary: `k ∈ −2..=13` biased into an
    /// unsigned 4-bit field. The bias is single-sourced through genconst so the
    /// shaders that decode `detail_pass` share the exact value.
    const GPU_BIAS: i8 = crate::genconst::DETAIL_GPU_BIAS as i8;
    const GPU_MIN: i8 = -Self::GPU_BIAS;
    const GPU_MAX: i8 = 15 - Self::GPU_BIAS;

    /// A detail level from an unsigned LOD step (`k = level`, non-negative).
    /// Negative levels (half-block edits) are constructed via [`Detail`] directly.
    /// Clamped to `GPU_MAX` BEFORE the signed cast: public inputs are
    /// untrusted, and clamping an already-cast `i8` cannot undo the
    /// 128..=255 wrap to negative — silently past `to_gpu_bits`'s
    /// debug-only assert (compiled out in release).
    pub fn new(level: u8) -> Detail {
        Detail(level.min(Self::GPU_MAX as u8) as i8)
    }

    /// Per-draw uniform scale: `2^k` world units per cell. `exp2`, not a shift,
    /// so it is defined for negative `k` too: `scale(-1) == 0.5`, `scale(-2) == 0.25`.
    pub fn scale(self) -> f32 {
        2f32.powi(self.0 as i32)
    }

    /// Bias `k` into the unsigned 4-bit field records/shaders store.
    pub fn to_gpu_bits(self) -> u8 {
        debug_assert!(
            (Self::GPU_MIN..=Self::GPU_MAX).contains(&self.0),
            "Detail {} outside the GPU-encodable range {}..={}",
            self.0,
            Self::GPU_MIN,
            Self::GPU_MAX
        );
        (self.0 + Self::GPU_BIAS) as u8
    }

    /// Inverse of [`Self::to_gpu_bits`].
    pub fn from_gpu_bits(bits: u8) -> Detail {
        debug_assert!(bits <= 15, "GPU detail field {bits} does not fit 4 bits");
        Detail(bits as i8 - Self::GPU_BIAS)
    }
}

/// A minimal axis-aligned cell-space region selector — the smallest shape
/// that lets [`interferes`] test real overlap now. Arbitrary selector
/// geometry (rings, spheres, frustum slices) lands with the scheduler.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegionSelector {
    pub min: [i64; 3],
    pub max: [i64; 3],
}

impl RegionSelector {
    pub fn overlaps(&self, other: &RegionSelector) -> bool {
        (0..3).all(|i| self.min[i] <= other.max[i] && other.min[i] <= self.max[i])
    }
}

/// Opaque store identity for [`FootprintKey::Keyed`] producers (net peers,
/// sim systems, autosave generation, …).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StoreId(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SourceId(pub u32);

/// Where a producer reads/writes. `Global` (⊤) and non-spatial `Keyed` keys
/// are first-class, not special cases.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FootprintKey {
    Region {
        sel: RegionSelector,
        level: Detail,
    },
    /// Net peers, sim systems, autosave generation, …
    Keyed {
        store: StoreId,
    },
    /// ⊤: frame images, scalar clocks.
    Global,
}

fn keys_overlap(a: &FootprintKey, b: &FootprintKey) -> bool {
    match (a, b) {
        (FootprintKey::Global, _) | (_, FootprintKey::Global) => true,
        (FootprintKey::Keyed { store: s1 }, FootprintKey::Keyed { store: s2 }) => s1 == s2,
        (FootprintKey::Region { sel: s1, .. }, FootprintKey::Region { sel: s2, .. }) => {
            s1.overlaps(s2)
        }
        _ => false, // a Keyed store and a Region never alias
    }
}

pub struct Footprint {
    pub reads: Vec<FootprintKey>,
    pub writes: Vec<FootprintKey>,
}

impl Footprint {
    fn all(&self) -> impl Iterator<Item = &FootprintKey> {
        self.reads.iter().chain(self.writes.iter())
    }
}

/// `p # q ⟺ footprints overlap ∧ one writes`. Symmetric by construction: true
/// iff either side writes into something the other side touches.
pub fn interferes(p: &Footprint, q: &Footprint) -> bool {
    let writes_into = |writer: &Footprint, other: &Footprint| {
        writer
            .writes
            .iter()
            .any(|w| other.all().any(|k| keys_overlap(w, k)))
    };
    writes_into(p, q) || writes_into(q, p)
}

pub enum Cadence {
    Frame,
    FixedTick,
    Hz(u16),
    OnRevision(SourceId),
    Once,
}

#[derive(Clone, Copy, Debug)]
pub enum Budget {
    Bytes(u32),
    Dispatches(u16),
    Millis(f32),
}

/// THE manifest. From this one declaration a later phase derives queue
/// placement, pomset edges, Sync2 barriers, timeline waits/signals, dispatch
/// bounds, eviction participation, law tests, and a profiler row.
pub struct Producer {
    pub name: &'static str,
    pub footprint: Footprint,
    pub cadence: Cadence,
    /// A forward-progress floor is declared by the scheduler that consumes
    /// this, not encoded in [`Budget`] itself.
    pub budget: Budget,
}

/// What a producer's run accomplished this tick. `UpTo` stamps the output at
/// a revision; `Partial` reports bounded remaining work (the resumable-BFS /
/// worklist case); `Idle` means nothing was due.
pub enum Progress {
    UpTo(crate::rev::Rev),
    Partial { remaining: u32 },
    Idle,
}

/// The scheduler's clock view for one tick. Fixed-tick catch-up is capped
/// here, not in a lane-local accumulator, so drift stays bounded.
pub struct Clocks {
    pub frame_dt: f32,
    /// Number of fixed-tick steps due this call; capped by the scheduler.
    pub fixed_ticks_due: u8,
}

/// Quiescence as an observable, not a per-lane counter: `quiescent` is true
/// iff every producer was `Idle` or already stamped at its current input
/// revisions this tick.
pub struct TickReport {
    pub quiescent: bool,
    pub spent: Budget,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(min: [i64; 3], max: [i64; 3]) -> FootprintKey {
        FootprintKey::Region {
            sel: RegionSelector { min, max },
            level: Detail(0),
        }
    }

    fn footprint(reads: Vec<FootprintKey>, writes: Vec<FootprintKey>) -> Footprint {
        Footprint { reads, writes }
    }

    #[test]
    fn region_overlap_without_a_write_does_not_interfere() {
        let a = footprint(vec![region([0, 0, 0], [4, 4, 4])], vec![]);
        let b = footprint(vec![region([2, 2, 2], [6, 6, 6])], vec![]);
        assert!(
            !interferes(&a, &b),
            "two readers of the same region never interfere"
        );
    }

    #[test]
    fn region_overlap_with_a_write_interferes() {
        let a = footprint(vec![], vec![region([0, 0, 0], [4, 4, 4])]);
        let b = footprint(vec![region([2, 2, 2], [6, 6, 6])], vec![]);
        assert!(interferes(&a, &b));
        assert!(interferes(&b, &a), "interference is symmetric");
    }

    #[test]
    fn disjoint_regions_never_interfere_even_if_both_write() {
        let a = footprint(vec![], vec![region([0, 0, 0], [1, 1, 1])]);
        let b = footprint(vec![], vec![region([10, 10, 10], [11, 11, 11])]);
        assert!(!interferes(&a, &b));
    }

    #[test]
    fn global_interferes_with_everything_written() {
        let global_writer = footprint(vec![], vec![FootprintKey::Global]);
        let region_reader = footprint(vec![region([0, 0, 0], [1, 1, 1])], vec![]);
        assert!(interferes(&global_writer, &region_reader));

        let keyed_reader = footprint(vec![FootprintKey::Keyed { store: StoreId(1) }], vec![]);
        assert!(interferes(&global_writer, &keyed_reader));
    }

    #[test]
    fn two_global_readers_do_not_interfere() {
        let a = footprint(vec![FootprintKey::Global], vec![]);
        let b = footprint(vec![FootprintKey::Global], vec![]);
        assert!(!interferes(&a, &b));
    }

    #[test]
    fn keyed_interferes_only_on_matching_store_id() {
        let writer = footprint(vec![], vec![FootprintKey::Keyed { store: StoreId(7) }]);
        let same = footprint(vec![FootprintKey::Keyed { store: StoreId(7) }], vec![]);
        let other = footprint(vec![FootprintKey::Keyed { store: StoreId(8) }], vec![]);
        assert!(interferes(&writer, &same));
        assert!(!interferes(&writer, &other));
    }

    #[test]
    fn keyed_and_region_never_alias() {
        let a = footprint(vec![], vec![FootprintKey::Keyed { store: StoreId(1) }]);
        let b = footprint(vec![region([0, 0, 0], [1, 1, 1])], vec![]);
        assert!(!interferes(&a, &b));
    }

    #[test]
    fn detail_gpu_bits_roundtrip_across_encodable_range() {
        for k in -2..=13i8 {
            let d = Detail(k);
            let bits = d.to_gpu_bits();
            assert!(bits <= 15);
            assert_eq!(Detail::from_gpu_bits(bits), d);
        }
    }

    #[test]
    fn detail_scale_is_exp2_including_negative_levels() {
        assert_eq!(Detail(-2).scale(), 0.25);
        assert_eq!(Detail(-1).scale(), 0.5);
        assert_eq!(Detail::FULL.scale(), 1.0);
        assert_eq!(Detail(0).scale(), 1.0);
        assert_eq!(Detail(3).scale(), 8.0);
    }

    #[test]
    fn detail_new_clamps_out_of_range_public_input() {
        assert_eq!(Detail::new(200), Detail(Detail::GPU_MAX));
        assert_eq!(Detail::new(255), Detail(Detail::GPU_MAX));
    }
}
