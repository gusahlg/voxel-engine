//! Deterministic, blocking screenshot capture.
//!
//! [`Engine::screenshot`](crate::Engine::screenshot) is async and picks a
//! timestamped name; the golden harness instead needs a capture that lands at
//! an exact path and blocks until the PNG is on disk. This module reuses the
//! existing readback plumbing (`request_screenshot` → present-copy readback →
//! `write_png`) and adds only the *blocking* wrapper plus a PNG decoder, so the
//! game crate calls `voxel_engine::{screenshot_to, load_png, Screenshot}` and
//! stays codec-free.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::Engine;

/// A decoded screenshot: tonemapped 8-bit sRGB, tightly-packed RGBA
/// (`width * height * 4` bytes), row-major from the top-left — the present
/// orientation, i.e. exactly what the player sees.
pub struct Screenshot {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// How long [`screenshot_to`] waits for the readback PNG before giving up. The
/// capture normally lands within a few present cycles; this only bounds a stuck
/// render thread.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(20);

/// Blocking, deterministic variant of [`Engine::screenshot`](crate::Engine::screenshot):
/// captures the next presented frame to exactly `path` (post-tonemap 8-bit
/// sRGB, top-left origin) and returns only once the PNG is fully written.
///
/// The capture targets the engine's *last submitted* scene: the caller records
/// and finishes the frame it wants (drops its [`Frame`](crate::Frame)), then
/// calls this. The request is queued, then the retained scene is re-presented
/// so the pending capture latches real terrain rather than a blank frame; the
/// write is atomic (temp + rename), so the returned path always names a
/// complete file.
pub fn screenshot_to(engine: &mut Engine, path: &Path) -> std::io::Result<()> {
    // Start from a clean slate so the poll below detects *this* capture's write,
    // never a stale file from a previous run.
    if path.exists() {
        std::fs::remove_file(path)?;
    }
    engine.request_screenshot_to(path.to_path_buf());

    let deadline = Instant::now() + CAPTURE_TIMEOUT;
    // A couple of presents flush the request through the render thread and the
    // 2-frames-in-flight pipeline; after that the async writer owns the file.
    let mut presents_left = 3u32;
    while Instant::now() < deadline {
        if presents_left > 0 {
            engine.present_last();
            presents_left -= 1;
        }
        if path.exists() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        format!("screenshot_to timed out waiting for {}", path.display()),
    ))
}

/// Decodes a PNG written by [`screenshot_to`] into a [`Screenshot`]. Errors on
/// anything that is not an 8-bit RGBA PNG.
pub fn load_png(path: &Path) -> std::io::Result<Screenshot> {
    let (width, height, rgba) = crate::screenshot::read_png(path)?;
    Ok(Screenshot {
        width,
        height,
        rgba,
    })
}
