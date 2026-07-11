//! Screenshot file naming and PNG encoding. The Vulkan layer copies the
//! presented frame into a host buffer and swizzles it to RGBA8; the helpers
//! here decide *where* it lands and write the PNG.

use std::path::{Path, PathBuf};

/// Next non-colliding screenshot path under a cwd-relative `screenshots/`
/// directory (created on demand). Named `watt-<UTC-timestamp>.png`, with a
/// `-1`, `-2`, … suffix appended only if that exact name already exists, so a
/// capture never overwrites an earlier one. Returns `None` if the directory
/// cannot be created.
pub(crate) fn next_path() -> Option<PathBuf> {
    let dir = PathBuf::from("screenshots");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        log::error!("cannot create screenshots directory: {e}");
        return None;
    }
    let stamp = timestamp();
    let mut path = dir.join(format!("watt-{stamp}.png"));
    let mut n = 1u32;
    while path.exists() {
        path = dir.join(format!("watt-{stamp}-{n}.png"));
        n += 1;
    }
    Some(path)
}

/// `YYYYMMDD-HHMMSS` in UTC. Falls back to `00000000-000000` if the system
/// clock predates the epoch (never in practice).
fn timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, s) = civil_from_epoch(secs);
    format!("{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}")
}

/// UTC epoch seconds → (year, month, day, hour, minute, second) via Howard
/// Hinnant's `civil_from_days`, avoiding a calendar dependency for one string.
fn civil_from_epoch(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (secs / 86_400) as i64;
    let tod = (secs % 86_400) as u32;
    let (hour, minute, second) = (tod / 3600, (tod % 3600) / 60, tod % 60);

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = if month <= 2 { year + 1 } else { year };
    (year, month, day, hour, minute, second)
}

/// Writes tightly-packed RGBA8 pixels (`width * height * 4` bytes, top-down)
/// as an 8-bit PNG.
///
/// Atomic: the encode goes to a sibling `<name>.tmp` which is then renamed onto
/// `path`. A reader polling for `path` (the blocking [`crate::screenshot_to`]
/// capture) therefore never observes a half-written file.
pub(crate) fn write_png(
    path: &Path,
    width: u32,
    height: u32,
    rgba: &[u8],
) -> Result<(), png::EncodingError> {
    let tmp = path.with_extension("png.tmp");
    {
        let file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        let mut encoder = png::Encoder::new(file, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.write_header()?.write_image_data(rgba)?;
    }
    std::fs::rename(&tmp, path).map_err(png::EncodingError::from)
}

/// Decodes an RGBA8 PNG written by [`write_png`] into flat, tightly-packed
/// `(width, height, rgba)`. Used by [`crate::load_png`]; kept here beside the
/// encoder so the `png` codec has one home.
pub(crate) fn read_png(path: &Path) -> std::io::Result<(u32, u32, Vec<u8>)> {
    let decoder = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
    let mut reader = decoder
        .read_info()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    buf.truncate(info.buffer_size());
    // `screenshot_to` always writes RGBA8; reject anything else loudly rather
    // than hand back mis-shaped pixels.
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "expected 8-bit RGBA PNG, got {:?}/{:?}",
                info.color_type, info.bit_depth
            ),
        ));
    }
    Ok((info.width, info.height, buf))
}
