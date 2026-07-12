//! Deterministic, blocking screenshot capture.
//!
//! [`Engine::screenshot`](crate::Engine::screenshot) is async and picks a
//! timestamped name; the golden harness instead needs a capture that lands at
//! an exact path and blocks until the PNG is on disk. This module reuses the
//! existing readback plumbing (`request_capture` → present-copy readback →
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

/// Liveness backstop for [`screenshot_to`]: the forced present lands within one
/// or two present cycles, so this deadline only trips if the swapchain is truly
/// dead (e.g. window destroyed). It is NOT the primary completion signal — that
/// is the reply channel carrying the real write outcome.
const CAPTURE_TIMEOUT: Duration = Duration::from_secs(20);

/// Blocking, deterministic variant of [`Engine::screenshot`](crate::Engine::screenshot):
/// captures the next presented frame to exactly `path` (post-tonemap 8-bit
/// sRGB, top-left origin) and returns only once the PNG is fully written — or a
/// real error from the encode/write.
///
/// The capture targets the engine's *last submitted* scene: the caller records
/// and finishes the frame it wants (drops its [`Frame`](crate::Frame)), then
/// calls this. The request forces the next present (the pacer cannot drop it),
/// so re-presenting the retained scene once latches real terrain rather than a
/// blank frame; completion (and any encode error) arrives over the reply
/// channel, not by polling the filesystem.
pub fn screenshot_to(engine: &mut Engine, path: &Path) -> std::io::Result<()> {
    let rx = engine.request_capture(path.to_path_buf());
    let deadline = Instant::now() + CAPTURE_TIMEOUT;
    // Drive frames until the forced present latches the capture and the writer
    // replies. One present_last is enough on the happy path; the loop only
    // repeats if a swapchain recreate ate the first present.
    while Instant::now() < deadline {
        engine.present_last();
        match rx.recv_timeout(Duration::from_millis(50)) {
            Ok(result) => return result,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                return Err(std::io::Error::other("capture reply channel disconnected"));
            }
        }
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
