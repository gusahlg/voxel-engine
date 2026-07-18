//! Unified per-frame profiler: the single sink every subsystem feeds so one
//! report shows where the frame budget goes across CPU, GPU, and worker threads.
//!
//! One flat [`Meter`] enum names every timed stage; each is grouped into a
//! [`Tier`]. Storage is a global array of atomics (nanoseconds + sample count),
//! so a meter can be fed from ANY thread — the main thread (CPU scopes), the
//! render thread (GPU pass timings read back from timestamps), or the chunk
//! worker pool (job timings) — without locking. `Meter as usize` indexes the
//! arrays, so a label can never drift from the time it names.
//!
//! Everything is normalized to **milliseconds per frame** in the report, so the
//! three domains are directly comparable:
//! - CPU tiers run sequentially on the main thread, so `sim + list + submit`
//!   ≈ the main-thread frame cost.
//! - GPU runs asynchronously to the CPU; its total is the GPU frame cost.
//! - Workers run in parallel off the critical path; their ms/frame is *offered
//!   load* — if it exceeds the frame wall-time, the backlog grows and far
//!   terrain lags behind the player.
//!
//! Gated by `VOXEL_PROFILE`: disabled → every entry point is a cheap no-op.

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// A timed stage. Ordinal indexes the accumulator arrays; grouped by [`tier`].
///
/// [`tier`]: Meter::tier
#[derive(Clone, Copy)]
pub enum Meter {
    // Tier::CpuSim — simulation (App::frame → Game::update)
    NetEvents,
    Physics,
    StreamDrain,
    StreamLight,
    StreamMesh,
    StreamTiles,
    StreamOcclusion,
    // Tier::CpuList — render-list build (Game::draw)
    ListSky,
    ListWorld,
    ListHud,
    // Tier::CpuSubmit — submit-side CPU (draw_frame)
    Fence,
    Acquire,
    Upload,
    Pack,
    Record,
    // Record sub-stages: the CPU cost of recording each pass inside `Record`.
    // Substages (like the tile ones) — excluded from the submit tier total and
    // printed on their own breakdown line, so they never double-count `Record`.
    RecShadow,
    RecMesh,
    RecSky,
    RecImmediate,
    RecOverlay,
    RecTransitions,
    Submit,
    Present,
    // Tier::Gpu — GPU render passes (timestamp readback)
    GpuOpaque,
    GpuSky,
    GpuCubes,
    GpuLines,
    GpuShadows,
    GpuTransparent,
    GpuOverlay,
    /// End of the scene pass: `cmd_end_rendering` (the 8x-MSAA color resolve
    /// lands here), the offscreen finalize transitions, and TAA/exposure.
    GpuResolve,
    /// The render-command tail after the resolve: the bloom chain.
    GpuPost,
    // Tier::Workers — off-thread chunk jobs; the tile stages are sub-timings
    WorkGenerate,
    WorkMesh,
    WorkLight,
    WorkTile,
    TileSample,
    TileMesh,
}

impl Meter {
    const ALL: [Meter; 38] = [
        Meter::NetEvents,
        Meter::Physics,
        Meter::StreamDrain,
        Meter::StreamLight,
        Meter::StreamMesh,
        Meter::StreamTiles,
        Meter::StreamOcclusion,
        Meter::ListSky,
        Meter::ListWorld,
        Meter::ListHud,
        Meter::Fence,
        Meter::Acquire,
        Meter::Upload,
        Meter::Pack,
        Meter::Record,
        Meter::RecShadow,
        Meter::RecMesh,
        Meter::RecSky,
        Meter::RecImmediate,
        Meter::RecOverlay,
        Meter::RecTransitions,
        Meter::Submit,
        Meter::Present,
        Meter::GpuOpaque,
        Meter::GpuSky,
        Meter::GpuCubes,
        Meter::GpuLines,
        Meter::GpuShadows,
        Meter::GpuTransparent,
        Meter::GpuOverlay,
        Meter::GpuResolve,
        Meter::GpuPost,
        Meter::WorkGenerate,
        Meter::WorkMesh,
        Meter::WorkLight,
        Meter::WorkTile,
        Meter::TileSample,
        Meter::TileMesh,
    ];
    const COUNT: usize = Self::ALL.len();

    fn label(self) -> &'static str {
        match self {
            Meter::NetEvents => "net",
            Meter::Physics => "physics",
            Meter::StreamDrain => "stream.drain",
            Meter::StreamLight => "stream.light",
            Meter::StreamMesh => "stream.mesh",
            Meter::StreamTiles => "stream.tiles",
            Meter::StreamOcclusion => "stream.occ",
            Meter::ListSky => "list.sky",
            Meter::ListWorld => "list.world",
            Meter::ListHud => "list.hud",
            Meter::Fence => "fence",
            Meter::Acquire => "acquire",
            Meter::Upload => "upload",
            Meter::Pack => "pack",
            Meter::Record => "record",
            Meter::RecShadow => "rec.shadow",
            Meter::RecMesh => "rec.mesh",
            Meter::RecSky => "rec.sky",
            Meter::RecImmediate => "rec.imm",
            Meter::RecOverlay => "rec.2d",
            Meter::RecTransitions => "rec.trans",
            Meter::Submit => "submit",
            Meter::Present => "present",
            Meter::GpuOpaque => "opaque",
            Meter::GpuSky => "sky",
            Meter::GpuCubes => "cubes",
            Meter::GpuLines => "lines",
            Meter::GpuShadows => "shadows",
            Meter::GpuTransparent => "transparent",
            Meter::GpuOverlay => "overlay",
            Meter::GpuResolve => "resolve",
            Meter::GpuPost => "post",
            Meter::WorkGenerate => "generate",
            Meter::WorkMesh => "mesh",
            Meter::WorkLight => "light",
            Meter::WorkTile => "tile",
            Meter::TileSample => "tile.sample",
            Meter::TileMesh => "tile.mesh",
        }
    }

    fn tier(self) -> Tier {
        match self {
            Meter::NetEvents
            | Meter::Physics
            | Meter::StreamDrain
            | Meter::StreamLight
            | Meter::StreamMesh
            | Meter::StreamTiles
            | Meter::StreamOcclusion => Tier::CpuSim,
            Meter::ListSky | Meter::ListWorld | Meter::ListHud => Tier::CpuList,
            Meter::Fence
            | Meter::Acquire
            | Meter::Upload
            | Meter::Pack
            | Meter::Record
            | Meter::RecShadow
            | Meter::RecMesh
            | Meter::RecSky
            | Meter::RecImmediate
            | Meter::RecOverlay
            | Meter::RecTransitions
            | Meter::Submit
            | Meter::Present => Tier::CpuSubmit,
            Meter::GpuOpaque
            | Meter::GpuSky
            | Meter::GpuCubes
            | Meter::GpuLines
            | Meter::GpuShadows
            | Meter::GpuTransparent
            | Meter::GpuOverlay
            | Meter::GpuResolve
            | Meter::GpuPost => Tier::Gpu,
            Meter::WorkGenerate
            | Meter::WorkMesh
            | Meter::WorkLight
            | Meter::WorkTile
            | Meter::TileSample
            | Meter::TileMesh => Tier::Workers,
        }
    }
}

/// A sampled *count* (not a duration): the last value set this frame, reported
/// as-is. Answers "how big is the set a stage iterates" — the size-vs-cost check
/// a timing meter can't express (e.g. why `list.world` scales).
#[derive(Clone, Copy)]
pub enum Gauge {
    WorldChunks,
    WorldChunksLive,
    WorldTiles,
    WorldSkins,
    UploadBytes,
    DrawsPacked,
}

impl Gauge {
    const ALL: [Gauge; 6] = [
        Gauge::WorldChunks,
        Gauge::WorldChunksLive,
        Gauge::WorldTiles,
        Gauge::WorldSkins,
        Gauge::UploadBytes,
        Gauge::DrawsPacked,
    ];
    const COUNT: usize = Self::ALL.len();

    fn label(self) -> &'static str {
        match self {
            Gauge::WorldChunks => "chunks",
            Gauge::WorldChunksLive => "live",
            Gauge::WorldTiles => "tiles",
            Gauge::WorldSkins => "skins",
            Gauge::UploadBytes => "upload.bytes",
            Gauge::DrawsPacked => "draws.packed",
        }
    }
}

/// Last-set value per gauge (overwritten each frame, never accumulated).
static GAUGES: [AtomicU64; Gauge::COUNT] = [const { AtomicU64::new(0) }; Gauge::COUNT];

/// Record the current value of a gauge. Cheap no-op when profiling is off.
pub fn gauge(g: Gauge, value: u64) {
    if enabled() {
        GAUGES[g as usize].store(value, Ordering::Relaxed);
    }
}

/// A group of meters sharing an interpretation (see the module docs).
#[derive(Clone, Copy, PartialEq)]
enum Tier {
    CpuSim,
    CpuList,
    CpuSubmit,
    Gpu,
    Workers,
}

impl Tier {
    const ALL: [Tier; 5] = [
        Tier::CpuSim,
        Tier::CpuList,
        Tier::CpuSubmit,
        Tier::Gpu,
        Tier::Workers,
    ];

    fn label(self) -> &'static str {
        match self {
            Tier::CpuSim => "sim",
            Tier::CpuList => "list",
            Tier::CpuSubmit => "submit",
            Tier::Gpu => "gpu",
            Tier::Workers => "workers",
        }
    }
}

/// Frames per reporting window (~3.3s at 72fps).
const WINDOW: u64 = 240;

/// Windowed accumulators, reset after each report. Every counter is atomic so
/// worker threads and the render thread can feed the same sink the main thread
/// reads. `frames` is the shared denominator for ms/frame.
struct Meters {
    nanos: [AtomicU64; Meter::COUNT],
    count: [AtomicU64; Meter::COUNT],
    frames: AtomicU64,
}

static METERS: Meters = Meters {
    nanos: [const { AtomicU64::new(0) }; Meter::COUNT],
    count: [const { AtomicU64::new(0) }; Meter::COUNT],
    frames: AtomicU64::new(0),
};

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VOXEL_PROFILE").is_ok_and(|v| v != "0"))
}

/// Whether profiling is active. Lets a subsystem skip expensive collection (GPU
/// timestamp readback) entirely when disabled.
pub fn is_enabled() -> bool {
    enabled()
}

/// RAII scope: records `start.elapsed()` into its meter on drop. `start` is
/// `None` when profiling is disabled, making the drop a no-op.
#[must_use]
pub struct Guard {
    meter: Meter,
    start: Option<Instant>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(start) = self.start {
            add(self.meter, start.elapsed());
        }
    }
}

/// Open a timing scope for `meter`; the time lands when the guard drops.
pub fn scope(meter: Meter) -> Guard {
    Guard {
        meter,
        start: enabled().then(Instant::now),
    }
}

/// Feed one sample directly (for stages already timed elsewhere — GPU passes,
/// worker jobs). Callable from any thread.
pub fn add(meter: Meter, elapsed: Duration) {
    if !enabled() {
        return;
    }
    METERS.nanos[meter as usize].fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
    METERS.count[meter as usize].fetch_add(1, Ordering::Relaxed);
}

/// Feed one sample measured in milliseconds (GPU timestamps are read as ms).
pub fn add_ms(meter: Meter, ms: f64) {
    if ms.is_finite() && ms > 0.0 {
        add(meter, Duration::from_secs_f64(ms / 1000.0));
    }
}

/// Max seconds a window may run before flushing, regardless of frame count.
/// During a stall the frame cap alone could take a long time to fill, so this
/// forces a timely report while the badness is still on screen. Overridable
/// via `VOXEL_PROFILE_FLUSH_MS`.
fn flush_secs() -> f64 {
    static SECS: OnceLock<f64> = OnceLock::new();
    *SECS.get_or_init(|| {
        std::env::var("VOXEL_PROFILE_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|ms| ms / 1000.0)
            .unwrap_or(1.0)
    })
}

/// End of frame (main thread). Counts a frame, tracks the worst single-frame
/// period, and flushes the report at whichever comes first: [`WINDOW`] frames or
/// [`flush_secs`] of wall time — so a stall reports promptly instead of being
/// smeared across a slow 240-frame window.
pub fn frame_end() {
    if !enabled() {
        return;
    }
    let now = Instant::now();
    // Per-frame period, from the previous frame_end. Feeds the worst-frame peak
    // so a transient spike inside an otherwise-fast window is still surfaced.
    let dt = LAST_FRAME.replace(Some(now)).map(|p| now.duration_since(p));
    if let Some(dt) = dt {
        WORST_MS.with(|w| w.set(w.get().max(dt.as_secs_f64() * 1000.0)));
    }
    let frames = METERS.frames.fetch_add(1, Ordering::Relaxed) + 1;
    let aged = WINDOW_START
        .with(|s| s.get())
        .is_some_and(|start| now.duration_since(start).as_secs_f64() >= flush_secs());
    if frames < WINDOW && !aged {
        return;
    }
    METERS.frames.store(0, Ordering::Relaxed);
    report(frames);
}

thread_local! {
    /// Wall-clock start of the current window, for real ms/frame and fps. Only
    /// touched by `report`/`frame_end` on the main thread.
    static WINDOW_START: Cell<Option<Instant>> = const { Cell::new(None) };
    /// End of the previous frame, for the per-frame period.
    static LAST_FRAME: Cell<Option<Instant>> = const { Cell::new(None) };
    /// Worst single-frame period (ms) seen this window; reset by `report`.
    static WORST_MS: Cell<f64> = const { Cell::new(0.0) };
}

fn report(frames: u64) {
    let wall = WINDOW_START.replace(Some(Instant::now()));
    let f = frames as f64;

    // Swap-read every meter (reset for the next window). ms/frame and per-sample
    // ms are both derived here so the caller sees a stable snapshot.
    let mut ms_per_frame = [0.0f64; Meter::COUNT];
    let mut ms_per_sample = [0.0f64; Meter::COUNT];
    let mut per_frame_count = [0.0f64; Meter::COUNT];
    for m in Meter::ALL {
        let ns = METERS.nanos[m as usize].swap(0, Ordering::Relaxed) as f64;
        let c = METERS.count[m as usize].swap(0, Ordering::Relaxed) as f64;
        ms_per_frame[m as usize] = ns / f / 1.0e6;
        ms_per_sample[m as usize] = if c > 0.0 { ns / c / 1.0e6 } else { 0.0 };
        per_frame_count[m as usize] = c / f;
    }

    // Header: real frame period (hence fps) when we have a prior window mark,
    // plus the window's worst single frame — a stall that lasted only a few
    // frames shows here even when the average stays fast.
    let worst = WORST_MS.replace(0.0);
    let mut header = format!("profile {frames}f");
    if let Some(secs) = wall.map(|w| w.elapsed().as_secs_f64()) {
        header.push_str(&format!(
            " {:.2}ms/f {:.0}fps worst {:.1}ms",
            secs * 1000.0 / f,
            f / secs,
            worst,
        ));
    }
    let tier_total = |t: Tier| -> f64 {
        Meter::ALL
            .into_iter()
            .filter(|m| m.tier() == t && !is_substage(*m))
            .map(|m| ms_per_frame[m as usize])
            .sum()
    };
    let cpu = tier_total(Tier::CpuSim) + tier_total(Tier::CpuList) + tier_total(Tier::CpuSubmit);
    header.push_str(&format!(
        " | cpu {:.2} (sim {:.2} list {:.2} submit {:.2}) gpu {:.2} work {:.2}",
        cpu,
        tier_total(Tier::CpuSim),
        tier_total(Tier::CpuList),
        tier_total(Tier::CpuSubmit),
        tier_total(Tier::Gpu),
        tier_total(Tier::Workers),
    ));
    // `eprintln!`, not `log::info!`: `VOXEL_PROFILE` is an explicit opt-in, so
    // the report prints unconditionally rather than also depending on the
    // env_logger level (`RUST_LOG=info`).
    eprintln!("{header}");

    // One line per tier, meters sorted hottest-first. Workers report ms/job and
    // jobs/frame (they are off-thread), with the tile sub-stages appended.
    for tier in Tier::ALL {
        let mut meters: Vec<Meter> = Meter::ALL
            .into_iter()
            .filter(|m| m.tier() == tier && !is_substage(*m))
            .collect();
        meters.sort_unstable_by(|a, b| {
            ms_per_frame[*b as usize].total_cmp(&ms_per_frame[*a as usize])
        });
        let mut line = format!("  {:<7}:", tier.label());
        for m in meters {
            if tier == Tier::Workers {
                line.push_str(&format!(
                    " {} {:.2}ms/job ×{:.2}/f (={:.2}/f)",
                    m.label(),
                    ms_per_sample[m as usize],
                    per_frame_count[m as usize],
                    ms_per_frame[m as usize],
                ));
            } else {
                line.push_str(&format!(" {} {:.2}", m.label(), ms_per_frame[m as usize]));
            }
        }
        if tier == Tier::Workers {
            line.push_str(&format!(
                " [tile.sample {:.2} tile.mesh {:.2} ms/job]",
                ms_per_sample[Meter::TileSample as usize],
                ms_per_sample[Meter::TileMesh as usize],
            ));
        }
        eprintln!("{line}");
    }

    // Set sizes: the iterated-set counts behind the CPU list cost. `list.world`
    // scales with these, so a jump here — not a per-item regression — is what
    // makes it spike. Last-sampled values, not windowed averages.
    let mut sline = String::from("  sets   :");
    for g in Gauge::ALL {
        sline.push_str(&format!(
            " {} {}",
            g.label(),
            GAUGES[g as usize].load(Ordering::Relaxed)
        ));
    }
    eprintln!("{sline}");

    // Record breakdown: the CPU sub-costs inside `Record`, hottest-first. These
    // are substages (not in the submit tier total); this line explains where the
    // `record` number goes without double-counting it.
    let mut rec: Vec<Meter> = [
        Meter::RecShadow,
        Meter::RecMesh,
        Meter::RecSky,
        Meter::RecImmediate,
        Meter::RecOverlay,
        Meter::RecTransitions,
    ]
    .into_iter()
    .collect();
    rec.sort_unstable_by(|a, b| ms_per_frame[*b as usize].total_cmp(&ms_per_frame[*a as usize]));
    let mut rline = String::from("  record :");
    for m in rec {
        rline.push_str(&format!(" {} {:.2}", m.label(), ms_per_frame[m as usize]));
    }
    eprintln!("{rline}");
}

/// Tile sub-stage meters are reported inline in the workers line, not as their
/// own tier entries (they double-count `tile`'s wall-time).
fn is_substage(m: Meter) -> bool {
    matches!(
        m,
        Meter::TileSample
            | Meter::TileMesh
            | Meter::RecShadow
            | Meter::RecMesh
            | Meter::RecSky
            | Meter::RecImmediate
            | Meter::RecOverlay
            | Meter::RecTransitions
    )
}
