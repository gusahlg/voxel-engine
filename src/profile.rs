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
//! - The wait tier is time a thread spent BLOCKED (not working): the main
//!   thread in the frame-pool handoff, the render thread in its timeline
//!   waits. It is reported apart from the CPU tiers so `submit` stays pure
//!   render-thread work.
//! - GPU runs asynchronously to the CPU; its total is the GPU frame cost, per
//!   RENDERED frame (the render thread may coalesce main-thread frames). The
//!   tonemap present copy runs only on presented frames; its meter carries the
//!   per-rendered-frame share, with the per-presented cost alongside.
//!   `gap` is device time between the previous render submit's union-stage end
//!   stamp and this submit's `TOP_OF_PIPE` start: idle GPU plus submit/command-
//!   processor overhead. It is not part of the gpu total; the header's `idle N%`
//!   is the window average of `gap / (gap + gpu_frame)`.
//! - Workers run in parallel off the critical path; their ms/frame is *offered
//!   load* — if it exceeds the frame wall-time, the backlog grows and far
//!   terrain lags behind the player.
//!
//! Gated by `VOXEL_PROFILE`: disabled → every entry point is a cheap no-op.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
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
    Acquire,
    Upload,
    Pack,
    Record,
    // Record sub-stages: the CPU cost of recording each pass inside `Record`.
    // Substages (like the tile ones) — excluded from the submit tier total and
    // printed in brackets after it, so they never double-count `Record`.
    RecShadow,
    RecCull,
    RecMesh,
    RecSky,
    RecImmediate,
    RecOverlay,
    RecTransitions,
    Submit,
    Present,
    /// Retired-resource reclaim after the slot fence (render thread).
    Reclaim,
    // Tier::Wait — time blocked, not working
    /// Main thread blocked in the frame-pool handoff (`take_frame`): the
    /// present-pacing wait for the render thread to return a snapshot.
    WaitFrame,
    /// Render thread blocked on the slot's previous render (timeline wait).
    Fence,
    /// Render thread blocked on an in-flight present copy (slot reuse, forced
    /// capture).
    WaitCopy,
    /// Render thread blocked pacing to the display: the vsync copy wait or a
    /// blocking drawable acquire.
    WaitVsync,
    // Tier::Gpu — GPU render passes (timestamp readback), in record order
    /// Staged mesh copies + minimap upload ahead of the cull.
    GpuCopies,
    /// Cull compute: counts fill, dispatch, and the DRAW_INDIRECT barrier.
    GpuCull,
    /// Cascaded shadow-map pass (only on regenerating frames).
    GpuShadowMap,
    /// VRS classify dispatch (only with a rate image and primed depth).
    GpuVrs,
    /// Scene-pass begin: attachment transitions + `cmd_begin_rendering` clears.
    GpuClear,
    /// Combined opaque span (full-res + cutout + coarse LOD), summed at GPU
    /// timestamp readback. The three group meters are substages of this.
    GpuOpaque,
    /// Full-res opaque (camera group 0). Substage of [`Meter::GpuOpaque`].
    GpuOpaqueFull,
    /// Cutout (camera group 1). Substage of [`Meter::GpuOpaque`].
    GpuCutout,
    /// Coarse-LOD opaque (camera group 2). Substage of [`Meter::GpuOpaque`].
    GpuOpaqueLod,
    GpuSky,
    GpuCubes,
    GpuLines,
    GpuShadows,
    GpuTransparent,
    GpuOverlay,
    /// End of the scene pass: `cmd_end_rendering` (the MSAA color resolve lands
    /// here) and the offscreen finalize transitions.
    GpuResolve,
    /// Retired TAA compute resolve. Fused into [`Meter::GpuTonemap`] at present
    /// time; left in the enum so ordinals stay stable (reports 0).
    GpuTaa,
    /// Exposure metering reduce + finalize.
    GpuExposure,
    /// The bloom chain + quarter-res spill (bloom composite + godrays) — the
    /// render-command tail.
    GpuBloom,
    /// The present copy (tonemap + fused TAA + warp + 2D overlay) — a separate
    /// submit that runs only on presented frames. Bloom composite and godrays
    /// live in the bloom span's spill dispatch.
    GpuTonemap,
    /// Device-time gap before this render submit: idle GPU plus submit /
    /// command-processor overhead (`TOP_OF_PIPE(N) - union-end(N-1)` on the
    /// device clock). Pass ends are stamped at each pass's last real stage,
    /// never `BOTTOM_OF_PIPE`.
    GpuGap,
    // Tier::Workers — off-thread chunk jobs; the tile stages are sub-timings
    WorkGenerate,
    WorkMesh,
    WorkLight,
    WorkTile,
    TileSample,
    TileMesh,
}

impl Meter {
    const ALL: [Meter; 55] = [
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
        Meter::Acquire,
        Meter::Upload,
        Meter::Pack,
        Meter::Record,
        Meter::RecShadow,
        Meter::RecCull,
        Meter::RecMesh,
        Meter::RecSky,
        Meter::RecImmediate,
        Meter::RecOverlay,
        Meter::RecTransitions,
        Meter::Submit,
        Meter::Present,
        Meter::Reclaim,
        Meter::WaitFrame,
        Meter::Fence,
        Meter::WaitCopy,
        Meter::WaitVsync,
        Meter::GpuCopies,
        Meter::GpuCull,
        Meter::GpuShadowMap,
        Meter::GpuVrs,
        Meter::GpuClear,
        Meter::GpuOpaque,
        Meter::GpuOpaqueFull,
        Meter::GpuCutout,
        Meter::GpuOpaqueLod,
        Meter::GpuSky,
        Meter::GpuCubes,
        Meter::GpuLines,
        Meter::GpuShadows,
        Meter::GpuTransparent,
        Meter::GpuOverlay,
        Meter::GpuResolve,
        Meter::GpuTaa,
        Meter::GpuExposure,
        Meter::GpuBloom,
        Meter::GpuTonemap,
        Meter::GpuGap,
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
            Meter::Acquire => "acquire",
            Meter::Upload => "upload",
            Meter::Pack => "pack",
            Meter::Record => "record",
            Meter::RecShadow => "rec.shadow",
            Meter::RecCull => "rec.cull",
            Meter::RecMesh => "rec.mesh",
            Meter::RecSky => "rec.sky",
            Meter::RecImmediate => "rec.imm",
            Meter::RecOverlay => "rec.2d",
            Meter::RecTransitions => "rec.trans",
            Meter::Submit => "submit",
            Meter::Present => "present",
            Meter::Reclaim => "reclaim",
            Meter::WaitFrame => "frame",
            Meter::Fence => "fence",
            Meter::WaitCopy => "copy",
            Meter::WaitVsync => "vsync",
            Meter::GpuCopies => "copies",
            Meter::GpuCull => "cull",
            Meter::GpuShadowMap => "shadowmap",
            Meter::GpuVrs => "vrs",
            Meter::GpuClear => "clear",
            Meter::GpuOpaque => "opaque",
            Meter::GpuOpaqueFull => "opaque.full",
            Meter::GpuCutout => "cutout",
            Meter::GpuOpaqueLod => "opaque.lod",
            Meter::GpuSky => "sky",
            Meter::GpuCubes => "cubes",
            Meter::GpuLines => "lines",
            Meter::GpuShadows => "shadows",
            Meter::GpuTransparent => "transparent",
            Meter::GpuOverlay => "overlay",
            Meter::GpuResolve => "resolve",
            Meter::GpuTaa => "taa",
            Meter::GpuExposure => "exposure",
            Meter::GpuBloom => "bloom",
            Meter::GpuTonemap => "tonemap",
            Meter::GpuGap => "gap",
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
            Meter::Acquire
            | Meter::Upload
            | Meter::Pack
            | Meter::Record
            | Meter::RecShadow
            | Meter::RecCull
            | Meter::RecMesh
            | Meter::RecSky
            | Meter::RecImmediate
            | Meter::RecOverlay
            | Meter::RecTransitions
            | Meter::Submit
            | Meter::Present
            | Meter::Reclaim => Tier::CpuSubmit,
            Meter::WaitFrame | Meter::Fence | Meter::WaitCopy | Meter::WaitVsync => Tier::Wait,
            Meter::GpuCopies
            | Meter::GpuCull
            | Meter::GpuShadowMap
            | Meter::GpuVrs
            | Meter::GpuClear
            | Meter::GpuOpaque
            | Meter::GpuOpaqueFull
            | Meter::GpuCutout
            | Meter::GpuOpaqueLod
            | Meter::GpuSky
            | Meter::GpuCubes
            | Meter::GpuLines
            | Meter::GpuShadows
            | Meter::GpuTransparent
            | Meter::GpuOverlay
            | Meter::GpuResolve
            | Meter::GpuTaa
            | Meter::GpuExposure
            | Meter::GpuBloom
            | Meter::GpuTonemap
            | Meter::GpuGap => Tier::Gpu,
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
    DrawsFull,
    DrawsCutout,
    DrawsLod,
    DrawsBlend,
    /// `vkCmdDrawIndexedIndirectCount` calls recorded for full-res opaque
    /// (partitions with capacity > 0; GPU count may still be zero).
    CallsFull,
    /// Same for coarse-LOD opaque.
    CallsLod,
    /// Draw calls recorded for Blend (CPU-sorted indirect / multi-draw).
    CallsBlend,
    /// Live arena rows in the directory this frame.
    Arenas,
    TrisFull,
    TrisCutout,
    TrisLod,
    Vrs1x1,
    Vrs2x2,
    Vrs4x4,
    FragFull,
    FragLod,
    FragCutout,
    FragBlend,
    FragSky,
    PrimsFull,
    /// Worker/main `MeshStager::acquire` successes since the last flush.
    PoolAcquires,
    /// Pooled `vkCmdCopyBuffer` regions submitted this frame.
    PoolCopies,
    /// Frames a pooled mesh sat in `pending` before `is_arrived` (1 = flushed
    /// the same render-loop iteration it was applied; >1 = budget-deferred).
    PoolArrivalFrames,
    /// Staged uploads that had to scan the region because no AABB was recorded.
    PoolAabbFallback,
}

impl Gauge {
    const ALL: [Gauge; 30] = [
        Gauge::WorldChunks,
        Gauge::WorldChunksLive,
        Gauge::WorldTiles,
        Gauge::WorldSkins,
        Gauge::UploadBytes,
        Gauge::DrawsPacked,
        Gauge::DrawsFull,
        Gauge::DrawsCutout,
        Gauge::DrawsLod,
        Gauge::DrawsBlend,
        Gauge::CallsFull,
        Gauge::CallsLod,
        Gauge::CallsBlend,
        Gauge::Arenas,
        Gauge::TrisFull,
        Gauge::TrisCutout,
        Gauge::TrisLod,
        Gauge::Vrs1x1,
        Gauge::Vrs2x2,
        Gauge::Vrs4x4,
        Gauge::FragFull,
        Gauge::FragLod,
        Gauge::FragCutout,
        Gauge::FragBlend,
        Gauge::FragSky,
        Gauge::PrimsFull,
        Gauge::PoolAcquires,
        Gauge::PoolCopies,
        Gauge::PoolArrivalFrames,
        Gauge::PoolAabbFallback,
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
            Gauge::DrawsFull => "draws.full",
            Gauge::DrawsCutout => "draws.cutout",
            Gauge::DrawsLod => "draws.lod",
            Gauge::DrawsBlend => "draws.blend",
            Gauge::CallsFull => "calls.full",
            Gauge::CallsLod => "calls.lod",
            Gauge::CallsBlend => "calls.blend",
            Gauge::Arenas => "arenas",
            Gauge::TrisFull => "tris.full",
            Gauge::TrisCutout => "tris.cutout",
            Gauge::TrisLod => "tris.lod",
            Gauge::Vrs1x1 => "vrs.1x1",
            Gauge::Vrs2x2 => "vrs.2x2",
            Gauge::Vrs4x4 => "vrs.4x4",
            Gauge::FragFull => "frag.full",
            Gauge::FragLod => "frag.lod",
            Gauge::FragCutout => "frag.cutout",
            Gauge::FragBlend => "frag.blend",
            Gauge::FragSky => "frag.sky",
            Gauge::PrimsFull => "prims.full",
            Gauge::PoolAcquires => "pool.acq",
            Gauge::PoolCopies => "pool.copies",
            Gauge::PoolArrivalFrames => "pool.arrive",
            Gauge::PoolAabbFallback => "pool.aabb",
        }
    }
}

/// Last-set value per gauge (overwritten each frame, never accumulated).
static GAUGES: [AtomicU64; Gauge::COUNT] = [const { AtomicU64::new(0) }; Gauge::COUNT];
/// Full-res overdraw (`frag.full / pixels`), stored as f64 bits. Printed with
/// two decimals on the `sets` line. Zero when pipe stats are unpublished.
static OVERDRAW_FULL: AtomicU64 = AtomicU64::new(0);
/// `slots N/M arenas A` on the sets line (`M` = cpu_cull_max).
static MESH_SLOTS_LIVE: AtomicU64 = AtomicU64::new(0);
static MESH_SLOTS_MAX: AtomicU64 = AtomicU64::new(0);
static MESH_ARENAS: AtomicU64 = AtomicU64::new(0);

/// Record the current value of a gauge. Cheap no-op when profiling is off.
pub fn gauge(g: Gauge, value: u64) {
    if enabled() {
        GAUGES[g as usize].store(value, Ordering::Relaxed);
    }
}

/// Record full-res overdraw (`frag.full / render-extent pixels`).
/// Record mesh slot occupancy for the profiler's `sets` line.
pub fn mesh_sets(live: u32, cpu_cull_max: u32, arenas: u32) {
    if enabled() {
        MESH_SLOTS_LIVE.store(u64::from(live), Ordering::Relaxed);
        MESH_SLOTS_MAX.store(u64::from(cpu_cull_max), Ordering::Relaxed);
        MESH_ARENAS.store(u64::from(arenas), Ordering::Relaxed);
    }
}

pub fn overdraw_full(ratio: f64) {
    if enabled() && ratio.is_finite() {
        OVERDRAW_FULL.store(ratio.to_bits(), Ordering::Relaxed);
    }
}

/// A group of meters sharing an interpretation (see the module docs).
#[derive(Clone, Copy, PartialEq)]
enum Tier {
    CpuSim,
    CpuList,
    CpuSubmit,
    Wait,
    Gpu,
    Workers,
}

impl Tier {
    const ALL: [Tier; 6] = [
        Tier::CpuSim,
        Tier::CpuList,
        Tier::CpuSubmit,
        Tier::Wait,
        Tier::Gpu,
        Tier::Workers,
    ];

    fn label(self) -> &'static str {
        match self {
            Tier::CpuSim => "sim",
            Tier::CpuList => "list",
            Tier::CpuSubmit => "submit",
            Tier::Wait => "wait",
            Tier::Gpu => "gpu",
            Tier::Workers => "workers",
        }
    }
}

/// A per-window event count (reset with the meters). `Rendered` is the GPU
/// tier's denominator; `Presented` is the tonemap meter's.
#[derive(Clone, Copy)]
pub enum Counter {
    /// Frames the render thread actually drew (`draw_frame` past its early outs).
    Rendered,
    /// Frames that reached a present copy + `vkQueuePresentKHR`.
    Presented,
    /// Queued game frames dropped by the render loop (kept only the newest).
    Coalesced,
    /// Render `vkQueueSubmit2` calls (a batched submit of N command buffers
    /// counts as one). Compared with [`Self::Rendered`] this is how much
    /// uncapped submission batching coalesced.
    Submits,
}

impl Counter {
    const COUNT: usize = 4;
}

static COUNTERS: [AtomicU64; Counter::COUNT] = [const { AtomicU64::new(0) }; Counter::COUNT];

/// Count one event. Cheap no-op when profiling is off.
pub fn count(c: Counter) {
    if enabled() {
        COUNTERS[c as usize].fetch_add(1, Ordering::Relaxed);
    }
}

/// Per-rendered-frame GPU totals (render command buffer, ms) fed by the render
/// thread's timestamp readback; drained by `report` for the p50/p95 header.
/// Capped so a runaway window can never grow it unboundedly.
static GPU_FRAMES: Mutex<Vec<f32>> = Mutex::new(Vec::new());
/// Per-frame `gap / (gap + gpu_frame)` for the header's `idle N%`. Same cap.
static GPU_IDLE_FRACS: Mutex<Vec<f32>> = Mutex::new(Vec::new());
const GPU_FRAME_CAP: usize = 4096;

/// Record one rendered frame's total GPU time (ms). Cheap no-op when off.
pub fn gpu_frame_ms(ms: f64) {
    if !enabled() || !ms.is_finite() || ms < 0.0 {
        return;
    }
    let mut samples = GPU_FRAMES.lock().unwrap_or_else(|e| e.into_inner());
    if samples.len() < GPU_FRAME_CAP {
        samples.push(ms as f32);
    }
}

/// Record the idle gap (ms) before this GPU frame. Also accumulates the
/// windowed idle fraction `gap / (gap + gpu_frame)` for the header.
/// Cheap no-op when profiling is off.
pub fn gpu_gap_ms(gap_ms: f64, frame_ms: f64) {
    if !enabled() || !gap_ms.is_finite() || gap_ms < 0.0 {
        return;
    }
    add_ms(Meter::GpuGap, gap_ms);
    if !frame_ms.is_finite() || frame_ms < 0.0 {
        return;
    }
    let den = gap_ms + frame_ms;
    if den <= 0.0 {
        return;
    }
    let mut samples = GPU_IDLE_FRACS.lock().unwrap_or_else(|e| e.into_inner());
    if samples.len() < GPU_FRAME_CAP {
        samples.push((gap_ms / den) as f32);
    }
}

/// Nearest-rank quantile of `samples` (sorted in place). `None` when empty.
fn quantile(samples: &mut [f32], q: f64) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    samples.sort_unstable_by(f32::total_cmp);
    let idx = ((samples.len() - 1) as f64 * q).round() as usize;
    Some(samples[idx.min(samples.len() - 1)] as f64)
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

impl Guard {
    /// Closes the running segment under the current meter and keeps timing
    /// under `meter` from now. Lets one scope hand a blocking wait in its
    /// middle to a wait meter without nesting (which would double-count).
    pub fn split(&mut self, meter: Meter) {
        if let Some(start) = self.start {
            let now = Instant::now();
            add(self.meter, now.duration_since(start));
            self.start = Some(now);
        }
        self.meter = meter;
    }
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
    let rendered = COUNTERS[Counter::Rendered as usize].swap(0, Ordering::Relaxed);
    let presented = COUNTERS[Counter::Presented as usize].swap(0, Ordering::Relaxed);
    let coalesced = COUNTERS[Counter::Coalesced as usize].swap(0, Ordering::Relaxed);
    let submits = COUNTERS[Counter::Submits as usize].swap(0, Ordering::Relaxed);
    // GPU meters are per RENDERED frame (the render thread may coalesce);
    // without a rendered count (no timestamps, minimized) fall back to frames.
    let gpu_f = if rendered > 0 { rendered as f64 } else { f };

    // Swap-read every meter (reset for the next window). ms/frame and per-sample
    // ms are both derived here so the caller sees a stable snapshot.
    let mut ms_per_frame = [0.0f64; Meter::COUNT];
    let mut ms_per_sample = [0.0f64; Meter::COUNT];
    let mut per_frame_count = [0.0f64; Meter::COUNT];
    for m in Meter::ALL {
        let ns = METERS.nanos[m as usize].swap(0, Ordering::Relaxed) as f64;
        let c = METERS.count[m as usize].swap(0, Ordering::Relaxed) as f64;
        let den = if m.tier() == Tier::Gpu { gpu_f } else { f };
        ms_per_frame[m as usize] = ns / den / 1.0e6;
        ms_per_sample[m as usize] = if c > 0.0 { ns / c / 1.0e6 } else { 0.0 };
        per_frame_count[m as usize] = c / f;
    }
    let mut gpu_frames = std::mem::take(&mut *GPU_FRAMES.lock().unwrap_or_else(|e| e.into_inner()));
    let gpu_idle_fracs =
        std::mem::take(&mut *GPU_IDLE_FRACS.lock().unwrap_or_else(|e| e.into_inner()));

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
    // The wait tier split by thread: `frame` is the main thread's, the rest the
    // render thread's. Neither is work, so neither joins `cpu`.
    let wait_main = ms_per_frame[Meter::WaitFrame as usize];
    header.push_str(&format!(
        " | cpu {:.2} (sim {:.2} list {:.2} submit {:.2}) wait {:.2} (main {:.2} render {:.2}) gpu {:.2}",
        cpu,
        tier_total(Tier::CpuSim),
        tier_total(Tier::CpuList),
        tier_total(Tier::CpuSubmit),
        tier_total(Tier::Wait),
        wait_main,
        tier_total(Tier::Wait) - wait_main,
        tier_total(Tier::Gpu),
    ));
    // Per-frame GPU distribution (render command buffer totals): the average
    // above hides a bimodal frame mix (e.g. shadow-map regeneration frames).
    if let (Some(p50), Some(p95)) = (
        quantile(&mut gpu_frames, 0.5),
        quantile(&mut gpu_frames, 0.95),
    ) {
        header.push_str(&format!(" (p50 {p50:.2} p95 {p95:.2})"));
    }
    if !gpu_idle_fracs.is_empty() {
        let idle =
            gpu_idle_fracs.iter().map(|x| f64::from(*x)).sum::<f64>() / gpu_idle_fracs.len() as f64;
        header.push_str(&format!(" idle {:.0}%", idle * 100.0));
    }
    header.push_str(&format!(
        " work {:.2} | rendered {rendered} coalesced {coalesced} submits {submits} presented {presented}",
        tier_total(Tier::Workers),
    ));
    // `eprintln!`, not `log::info!`: `VOXEL_PROFILE` is an explicit opt-in, so
    // the report prints unconditionally rather than also depending on the
    // env_logger level (`RUST_LOG=info`).
    eprintln!("{header}");

    // One line per tier, meters sorted hottest-first. Workers report ms/job and
    // jobs/frame (they are off-thread), with the tile sub-stages appended; the
    // submit line appends the record sub-stages the same way.
    let sorted = |mut meters: Vec<Meter>| {
        meters.sort_unstable_by(|a, b| {
            ms_per_frame[*b as usize].total_cmp(&ms_per_frame[*a as usize])
        });
        meters
    };
    for tier in Tier::ALL {
        let meters = sorted(
            Meter::ALL
                .into_iter()
                .filter(|m| m.tier() == tier && !is_substage(*m))
                .collect(),
        );
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
            // The present copy runs once per PRESENTED frame: its per-rendered
            // share is what the gpu total sums; the real per-present cost is
            // the number a tonemap change moves.
            if matches!(m, Meter::GpuTonemap) && ms_per_sample[m as usize] > 0.0 {
                line.push_str(&format!(" ({:.2}/present)", ms_per_sample[m as usize]));
            }
            // Opaque group split: substages of `opaque` (not in the gpu total),
            // hottest-first, so the combined number stays comparable.
            if matches!(m, Meter::GpuOpaque) {
                let split = sorted(vec![
                    Meter::GpuOpaqueFull,
                    Meter::GpuCutout,
                    Meter::GpuOpaqueLod,
                ]);
                line.push_str(" [");
                for (i, s) in split.into_iter().enumerate() {
                    if i > 0 {
                        line.push(' ');
                    }
                    line.push_str(&format!("{} {:.2}", s.label(), ms_per_frame[s as usize]));
                }
                line.push(']');
            }
        }
        if tier == Tier::Gpu {
            // Idle gap is GPU device time but not GPU *work*; keep it off the
            // hottest-first pass list (and the gpu total) and pin it at the end.
            line.push_str(&format!(
                " {} {:.2}",
                Meter::GpuGap.label(),
                ms_per_frame[Meter::GpuGap as usize],
            ));
        }
        if tier == Tier::Workers {
            line.push_str(&format!(
                " [tile.sample {:.2} tile.mesh {:.2} ms/job]",
                ms_per_sample[Meter::TileSample as usize],
                ms_per_sample[Meter::TileMesh as usize],
            ));
        }
        if tier == Tier::CpuSubmit {
            // Record breakdown: the CPU sub-costs inside `record`, hottest-first.
            // Substages (not in the submit total), so this explains where the
            // `record` number goes without double-counting it.
            let rec = sorted(vec![
                Meter::RecShadow,
                Meter::RecCull,
                Meter::RecMesh,
                Meter::RecSky,
                Meter::RecImmediate,
                Meter::RecOverlay,
                Meter::RecTransitions,
            ]);
            line.push_str(" [");
            for (i, m) in rec.into_iter().enumerate() {
                if i > 0 {
                    line.push(' ');
                }
                line.push_str(&format!("{} {:.2}", m.label(), ms_per_frame[m as usize]));
            }
            line.push(']');
        }
        eprintln!("{line}");
    }

    // Set sizes: the iterated-set counts behind the CPU list cost. `list.world`
    // scales with these, so a jump here — not a per-item regression — is what
    // makes it spike. Last-sampled values, not windowed averages.
    let mut sline = format!(
        "  sets   : slots {}/{} arenas {}",
        MESH_SLOTS_LIVE.load(Ordering::Relaxed),
        MESH_SLOTS_MAX.load(Ordering::Relaxed),
        MESH_ARENAS.load(Ordering::Relaxed),
    );
    for g in Gauge::ALL {
        sline.push_str(&format!(
            " {} {}",
            g.label(),
            GAUGES[g as usize].load(Ordering::Relaxed)
        ));
    }
    sline.push_str(&format!(
        " overdraw.full {:.2}",
        f64::from_bits(OVERDRAW_FULL.load(Ordering::Relaxed))
    ));
    eprintln!("{sline}");
}

/// Tile/record/opaque-group sub-stage meters are reported inline, not as their
/// own tier entries (they double-count a parent). `GpuGap` is idle between
/// submits — reported at the end of the gpu line and as `idle N%` in the
/// header, never in the gpu total.
fn is_substage(m: Meter) -> bool {
    matches!(
        m,
        Meter::TileSample
            | Meter::TileMesh
            | Meter::RecShadow
            | Meter::RecCull
            | Meter::RecMesh
            | Meter::RecSky
            | Meter::RecImmediate
            | Meter::RecOverlay
            | Meter::RecTransitions
            | Meter::GpuGap
            | Meter::GpuOpaqueFull
            | Meter::GpuCutout
            | Meter::GpuOpaqueLod
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_submits_is_the_fourth_window_count() {
        assert_eq!(Counter::Rendered as usize, 0);
        assert_eq!(Counter::Presented as usize, 1);
        assert_eq!(Counter::Coalesced as usize, 2);
        assert_eq!(Counter::Submits as usize, 3);
        assert_eq!(Counter::COUNT, 4);
    }

    #[test]
    fn gauge_ordinals_index_all_in_order() {
        for (i, g) in Gauge::ALL.into_iter().enumerate() {
            assert_eq!(g as usize, i, "{} is out of order in Gauge::ALL", g.label());
        }
    }

    #[test]
    fn gauge_labels_are_unique() {
        let mut labels: Vec<&str> = Gauge::ALL.into_iter().map(Gauge::label).collect();
        let n = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), n, "duplicate Gauge label");
    }

    #[test]
    fn meter_ordinals_index_all_in_order() {
        // `Meter as usize` indexes the accumulators; ALL must list every variant
        // at its own ordinal or a label drifts from the time it names.
        for (i, m) in Meter::ALL.into_iter().enumerate() {
            assert_eq!(m as usize, i, "{} is out of order in Meter::ALL", m.label());
        }
    }

    #[test]
    fn meter_labels_are_unique_within_a_tier() {
        for tier in Tier::ALL {
            let mut labels: Vec<&str> = Meter::ALL
                .into_iter()
                .filter(|m| m.tier() == tier)
                .map(Meter::label)
                .collect();
            let n = labels.len();
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), n, "duplicate label in tier {}", tier.label());
        }
    }

    #[test]
    fn opaque_group_meters_are_gpu_substages() {
        assert!(matches!(Meter::GpuOpaque.tier(), Tier::Gpu));
        assert!(!is_substage(Meter::GpuOpaque));
        for m in [Meter::GpuOpaqueFull, Meter::GpuCutout, Meter::GpuOpaqueLod] {
            assert!(
                matches!(m.tier(), Tier::Gpu),
                "{} should be a GPU meter",
                m.label()
            );
            assert!(
                is_substage(m),
                "{} should not join the gpu total",
                m.label()
            );
        }
    }

    #[test]
    fn every_tier_has_a_meter_and_wait_holds_only_waits() {
        for tier in Tier::ALL {
            assert!(Meter::ALL.iter().any(|m| m.tier() == tier));
        }
        let waits: Vec<Meter> = Meter::ALL
            .into_iter()
            .filter(|m| m.tier() == Tier::Wait)
            .collect();
        assert!(waits.iter().all(|m| matches!(
            m,
            Meter::WaitFrame | Meter::Fence | Meter::WaitCopy | Meter::WaitVsync
        )));
        assert_eq!(waits.len(), 4);
    }

    #[test]
    fn quantile_is_nearest_rank() {
        let mut empty: [f32; 0] = [];
        assert_eq!(quantile(&mut empty, 0.5), None);
        let mut one = [3.0f32];
        assert_eq!(quantile(&mut one, 0.95), Some(3.0));
        // Unsorted 1..=100: p50 lands on 51, p95 on 95 (nearest rank).
        let mut v: Vec<f32> = (1..=100).rev().map(|x| x as f32).collect();
        assert_eq!(quantile(&mut v, 0.5), Some(51.0));
        assert_eq!(quantile(&mut v, 0.95), Some(95.0));
        assert_eq!(quantile(&mut v, 1.0), Some(100.0));
        assert_eq!(quantile(&mut v, 0.0), Some(1.0));
    }

    #[test]
    fn split_on_a_disabled_guard_only_retargets() {
        // With profiling off the guard carries no start; `split` must stay a
        // no-op apart from retargeting the meter (no panic, nothing recorded).
        let mut g = Guard {
            meter: Meter::Acquire,
            start: None,
        };
        g.split(Meter::WaitCopy);
        assert!(matches!(g.meter, Meter::WaitCopy));
        assert!(g.start.is_none());
    }
}
