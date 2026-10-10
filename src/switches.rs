//! Environment switches: every `VOXEL_*` variable the engine, its build and
//! the demo read, and the helpers the engine reads its own with.
//!
//! A runtime switch is a [`Switch`] static next to the code it configures.
//! It is parsed on first use and cached for the process, so a hot path pays
//! one atomic load. An unset or non-Unicode value reaches the parser as
//! `None`, so the parser's answer for `None` is the default.
//! [`log_non_default`] runs at renderer creation. It logs one `switches:`
//! info line listing every runtime switch whose value differs from its
//! default, as `NAME=value` with the value as set. At defaults it logs
//! nothing.
//!
//! Flags come in three kinds. *On unless `0`*: only the exact string `0`
//! turns it off. *On if set, not `0`*: any other value turns it on, empty
//! included. *On if `1`*: only the exact string `1` turns it on. Numbers are
//! parsed untrimmed, and a value that does not parse is the default. MiB
//! sizes allow surrounding whitespace and log a warning when invalid.
//!
//! ## Runtime (cached; logged when not at the default)
//!
//! | Switch | Type | Default | Meaning | Read in |
//! |---|---|---|---|---|
//! | `VOXEL_PROFILE` | flag, on if set, not `0` | off | profiler meters and the periodic report | `profile.rs` |
//! | `VOXEL_PROFILE_FLUSH_MS` | `f64` ms | `1000` | longest wall time a profiler window runs before it reports | `profile.rs` |
//! | `VOXEL_SHADER_STATS` | `0`, `1` or `2`; any other value is `1` | `0` | `1` prints pipeline-executable statistics, `2` also writes internal representations | `vk/shader_stats.rs` |
//! | `VOXEL_SHADER_STATS_DIR` | path (`var_os`; logged when set) | `shader_stats` | where `VOXEL_SHADER_STATS=2` writes | `vk/shader_stats.rs` |
//! | `VOXEL_SKY_DEBUG` | flag, on if `1` | off | false-colours the sky by source and logs the horizon pick once a second | `vk/uniforms.rs` |
//! | `VOXEL_SKY_COARSE` | flag, on unless `0` | on | 2×2 sky tiles where the device supports them | `vk/device.rs` |
//! | `VOXEL_FAR_CULL` | flag, on unless `0` | on | far-body cone reject, frustum compaction and tile masks | `vk/far_bodies/mod.rs` |
//! | `VOXEL_SKY_MAPSOLO` | flag, on unless `0` | on | loop-free fragment for heavy tiles that hold exactly one Mapped body | `vk/far_bodies/mod.rs` |
//! | `VOXEL_SKY_ROUNDSOLO` | flag, on unless `0` | on | loop-free fragment for heavy tiles that hold exactly one Rounded body | `vk/far_bodies/mod.rs` |
//! | `VOXEL_LOD_BUCKET_SCALE` | `f32`, finite and > 0 | `32` | coarse-LOD distance-bucket multiplier | `vk/cull_math.rs` |
//! | `VOXEL_CPU_CULL_MAX` | `u32` | `1024` | camera live-mesh count up to which the CPU culls instead of the GPU | `vk/cull_math.rs` |
//! | `VOXEL_SUBMIT_BATCH` | `usize`, clamped to `1..=FRAMES_IN_FLIGHT - 1` | `2` | command buffers per submit for unpresented uncapped frames | `vk/submit.rs` |
//! | `VOXEL_EAGER_FLUSH_US` | `u64` µs | `15` | slot wait that turns on eager flush | `vk/submit.rs` |
//! | `VOXEL_BENCH_EMPTY` | `u32`, `0` is off | `0` | experiment: K empty command buffers in one submit per frame, no present | `vk/mod.rs` |
//! | `VOXEL_HDR_11BIT` | flag, on if set, not `0` | off | parked experiment: try a packed 11-bit HDR offscreen | `vk/targets.rs` |
//! | `VOXEL_TAA_MOTION_BOOST` | `f32` | `1` (no boost) | velocity-weighted current-frame boost in TAA | `vk/taa.rs` |
//! | `VOXEL_TAA_MOTION_PX` | `f32` | `8` | output-pixel velocity at which the boost saturates | `vk/taa.rs` |
//! | `VOXEL_MESH_STAGING_MB` | MiB | `32` | host mesh staging ring; `0` disables it | `vk/mesh_staging.rs` |
//! | `VOXEL_COMPUTE_INPUT_MB` | MiB | `16` | compute input staging ring | `vk/compute.rs` |
//! | `VOXEL_COMPUTE_READBACK_MB` | MiB | `32` | compute readback ring | `vk/compute.rs` |
//! | `VOXEL_ENGINE_VALIDATION` | flag, off for `0`, `false` or `off` (any case) | on in debug builds, off in release | Khronos validation layer | `vk/instance.rs` |
//! | `VOXEL_EVENTLOOP_ANY_THREAD` | presence (`var_os`) | unset | builds the event loop off the main thread (test aid for `compute_roundtrip`) | `engine.rs` |
//!
//! ## Build time (baked into the build; not logged)
//!
//! | Switch | Type | Default | Meaning | Read in |
//! |---|---|---|---|---|
//! | `VOXEL_LIGHT_LEGACY` | `0` or `1`; anything else fails the build | `0` | September lighting constants (the legacy look) | `build.rs` |
//! | `VOXEL_BUILD_PROBE` | presence | unset | also compiles the dev-only probe shaders | `build.rs` |
//! | `VOXEL_ENGINE_REFRESH_SHADER_FALLBACKS` | `0` or `1` | `0` | rewrites `shaders_spv/` with the pinned `slangc` | `crates/slang-build` |
//! | `SLANGC` | path | `slangc` on `PATH` | the shader compiler | `crates/slang-build` |
//!
//! ## Demo (`src/bin/demo.rs` only; not logged)
//!
//! | Switch | Type | Default | Meaning | Read in |
//! |---|---|---|---|---|
//! | `VOXEL_DEMO_FAR` | presence | unset | far-body showcase | `bin/demo.rs` |
//! | `VOXEL_DEMO_MAPPED_RHO` | `f32` in `(0, 1)` | `0.12` | showcase planet rho; an override also turns on its horizon cull | `bin/demo.rs` |
//! | `VOXEL_DEMO_ROUNDED_P` | `f32 >= 2`, or `cube` | `4` | showcase rounded exponent; `cube` draws a `FarShape::Cube` of the same size | `bin/demo.rs` |
//! | `VOXEL_DEMO_ROUNDED_RHO` | `f32` in `(0, 1)` | `0.06` | showcase rounded rho; about `0.6` is a close-up whose tiles hold that body alone | `bin/demo.rs` |
//! | `VOXEL_DEMO_RECREATE_CYCLE` | presence | unset | cycles vsync, MSAA, scale and fullscreen, then quits | `bin/demo.rs` |
//! | `VOXEL_DEMO_UPLOAD_CYCLE` | `0` or empty is off; `1` or `all`; or a comma list of `far`, `set`, `append`, `mat` | off | repeats far-map, block-texture and material uploads every ~60 frames | `bin/demo.rs` |
//! | `VOXEL_DEMO_VRS` | flag, on if set, not `0` | off | starts with VRS on | `bin/demo.rs` |
//! | `VOXEL_DEMO_MSAA` | `u32` | `1` | starting MSAA count | `bin/demo.rs` |
//! | `VOXEL_DEMO_FULLSCREEN` | flag, on if set, not `0` | off | starts fullscreen | `bin/demo.rs` |
//! | `VOXEL_WARP` | `f32` | `0` | starting cylindrical warp strength | `bin/demo.rs` |
//! | `VOXEL_AUTOSHOT` | presence | unset | one screenshot after warm-up, then quits | `bin/demo.rs` |

use std::str::FromStr;
use std::sync::OnceLock;

/// One environment variable, parsed on first use and cached for the process.
pub(crate) struct Switch<T> {
    name: &'static str,
    parser: fn(Option<&str>) -> T,
    value: OnceLock<T>,
}

impl<T: Copy> Switch<T> {
    /// `parser` gets the value as set, or `None` when the variable is unset
    /// or not Unicode.
    pub(crate) const fn new(name: &'static str, parser: fn(Option<&str>) -> T) -> Self {
        Self {
            name,
            parser,
            value: OnceLock::new(),
        }
    }

    /// The parsed value. The environment is read once per process.
    #[inline]
    pub(crate) fn get(&self) -> T {
        *self
            .value
            .get_or_init(|| (self.parser)(std::env::var(self.name).ok().as_deref()))
    }

    /// What `raw` parses to. `parse(None)` is the default.
    pub(crate) fn parse(&self, raw: Option<&str>) -> T {
        (self.parser)(raw)
    }
}

/// On unless the value is exactly `0`. Unset is on.
pub(crate) fn on_unless_zero(raw: Option<&str>) -> bool {
    raw != Some("0")
}

/// On when set to anything but exactly `0`, empty included. Unset is off.
pub(crate) fn on_if_set_nonzero(raw: Option<&str>) -> bool {
    raw.is_some_and(|v| v != "0")
}

/// On only when the value is exactly `1`.
pub(crate) fn on_if_one(raw: Option<&str>) -> bool {
    raw == Some("1")
}

/// `raw` parsed as `T`, untrimmed. Unset or unparsable is `default`.
pub(crate) fn parse_or<T: FromStr>(raw: Option<&str>, default: T) -> T {
    raw.and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Whole MiB as bytes (saturating). Surrounding whitespace is allowed.
pub(crate) fn parse_mib(s: &str) -> Option<u64> {
    let mb: u64 = s.trim().parse().ok()?;
    Some(mb.saturating_mul(1 << 20))
}

/// [`parse_mib`] of `raw`. Unset is `default`; an invalid value logs a
/// warning naming `name` and is `default`.
pub(crate) fn mib_or(name: &str, raw: Option<&str>, default: u64) -> u64 {
    match raw {
        Some(s) => parse_mib(s).unwrap_or_else(|| {
            log::warn!("invalid {name}={s:?}; using {} MiB", default / (1 << 20));
            default
        }),
        None => default,
    }
}

/// A runtime switch [`log_non_default`] checks.
trait Logged: Sync {
    fn name(&self) -> &'static str;

    /// True when the value in effect differs from the default.
    fn non_default(&self) -> bool;

    /// The value as set, for the log line.
    fn raw(&self) -> Option<String> {
        std::env::var_os(self.name()).map(|v| v.to_string_lossy().into_owned())
    }
}

impl<T: Copy + PartialEq + Send + Sync> Logged for Switch<T> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn non_default(&self) -> bool {
        self.get() != self.parse(None)
    }
}

/// A switch the engine reads by presence with `var_os`: set at all is not
/// the default.
struct Presence(&'static str);

impl Logged for Presence {
    fn name(&self) -> &'static str {
        self.0
    }

    fn non_default(&self) -> bool {
        std::env::var_os(self.0).is_some()
    }
}

/// Every runtime switch, in the order of the module table.
static RUNTIME: [&dyn Logged; 22] = [
    &crate::profile::VOXEL_PROFILE,
    &crate::profile::VOXEL_PROFILE_FLUSH_MS,
    &crate::vk::shader_stats::VOXEL_SHADER_STATS,
    &Presence("VOXEL_SHADER_STATS_DIR"),
    &crate::vk::uniforms::VOXEL_SKY_DEBUG,
    &crate::vk::device::VOXEL_SKY_COARSE,
    &crate::vk::far_bodies::VOXEL_FAR_CULL,
    &crate::vk::far_bodies::VOXEL_SKY_MAPSOLO,
    &crate::vk::far_bodies::VOXEL_SKY_ROUNDSOLO,
    &crate::vk::cull_math::VOXEL_LOD_BUCKET_SCALE,
    &crate::vk::cull_math::VOXEL_CPU_CULL_MAX,
    &crate::vk::submit::VOXEL_SUBMIT_BATCH,
    &crate::vk::submit::VOXEL_EAGER_FLUSH_US,
    &crate::vk::VOXEL_BENCH_EMPTY,
    &crate::vk::targets::VOXEL_HDR_11BIT,
    &crate::vk::taa::VOXEL_TAA_MOTION_BOOST,
    &crate::vk::taa::VOXEL_TAA_MOTION_PX,
    &crate::vk::mesh_staging::VOXEL_MESH_STAGING_MB,
    &crate::vk::compute::VOXEL_COMPUTE_INPUT_MB,
    &crate::vk::compute::VOXEL_COMPUTE_READBACK_MB,
    &crate::vk::instance::VOXEL_ENGINE_VALIDATION,
    &Presence("VOXEL_EVENTLOOP_ANY_THREAD"),
];

/// `NAME=value` for each of `entries` not at its default, space separated.
/// A value that is empty or holds whitespace is quoted. Empty when every
/// entry is at its default.
fn summary(entries: &[&dyn Logged]) -> String {
    let mut line = String::new();
    for entry in entries.iter().filter(|e| e.non_default()) {
        let raw = entry.raw().unwrap_or_default();
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(entry.name());
        line.push('=');
        if raw.is_empty() || raw.contains(char::is_whitespace) {
            line.push_str(&format!("{raw:?}"));
        } else {
            line.push_str(&raw);
        }
    }
    line
}

/// Logs one `switches:` info line naming every runtime switch not at its
/// default. Logs nothing when all are at their defaults. Called at renderer
/// creation.
pub(crate) fn log_non_default() {
    let line = summary(&RUNTIME);
    if !line.is_empty() {
        log::info!("switches: {line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Debug;
    use std::time::Duration;

    /// Inputs every parser is checked against.
    const SAMPLES: [Option<&str>; 24] = [
        None,
        Some(""),
        Some(" "),
        Some("0"),
        Some("1"),
        Some("2"),
        Some(" 1"),
        Some("1 "),
        Some("01"),
        Some("-1"),
        Some("off"),
        Some("OFF"),
        Some("false"),
        Some("False"),
        Some("yes"),
        Some("nan"),
        Some("inf"),
        Some("1.5"),
        Some("64"),
        Some(" 64 "),
        Some("abc"),
        Some("4294967296"),
        Some("18446744073709551615"),
        Some("18446744073709551616"),
    ];

    /// `switch` parses every sample like `old`, and `None` is `default`.
    /// Compared through `Debug` so NaN matches NaN.
    fn same<T: Copy + Debug>(switch: &Switch<T>, default: T, old: impl Fn(Option<&str>) -> T) {
        assert_eq!(
            format!("{:?}", switch.parse(None)),
            format!("{default:?}"),
            "{} default",
            switch.name
        );
        for raw in SAMPLES {
            assert_eq!(
                format!("{:?}", switch.parse(raw)),
                format!("{:?}", old(raw)),
                "{}={raw:?}",
                switch.name
            );
        }
    }

    #[test]
    fn flag_kinds() {
        for (raw, unless_zero, set_nonzero, one) in [
            (None, true, false, false),
            (Some(""), true, true, false),
            (Some("0"), false, false, false),
            (Some("1"), true, true, true),
            (Some("2"), true, true, false),
            (Some(" 0"), true, true, false),
            (Some("00"), true, true, false),
            (Some(" 1"), true, true, false),
            (Some("off"), true, true, false),
            (Some("true"), true, true, false),
        ] {
            assert_eq!(on_unless_zero(raw), unless_zero, "{raw:?}");
            assert_eq!(on_if_set_nonzero(raw), set_nonzero, "{raw:?}");
            assert_eq!(on_if_one(raw), one, "{raw:?}");
        }
    }

    #[test]
    fn parse_or_is_untrimmed_and_falls_back() {
        assert_eq!(parse_or(None, 7u32), 7);
        assert_eq!(parse_or(Some("12"), 7u32), 12);
        assert_eq!(parse_or(Some(" 12"), 7u32), 7);
        assert_eq!(parse_or(Some("-1"), 7u32), 7);
        assert_eq!(parse_or(Some("4294967296"), 7u32), 7);
        assert_eq!(parse_or(Some(""), 7u32), 7);
        assert_eq!(parse_or(Some("4.0"), 1.0f32), 4.0);
        assert_eq!(parse_or(Some("8"), 1.0f32), 8.0);
        assert_eq!(parse_or(Some("nope"), 1.0f32), 1.0);
        assert_eq!(parse_or(Some("  "), 8.0f32), 8.0);
        assert!(parse_or(Some("nan"), 1.0f64).is_nan());
        assert_eq!(parse_or(Some("-0.5"), 1.0f64), -0.5);
    }

    #[test]
    fn mib_trims_saturates_and_falls_back() {
        assert_eq!(parse_mib("16"), Some(16 << 20));
        assert_eq!(parse_mib(" 32 "), Some(32 << 20));
        assert_eq!(parse_mib(" 0 "), Some(0));
        assert_eq!(parse_mib("1"), Some(1 << 20));
        assert_eq!(parse_mib(""), None);
        assert_eq!(parse_mib("nope"), None);
        assert_eq!(parse_mib("-1"), None);
        assert_eq!(parse_mib("18446744073709551615"), Some(u64::MAX));
        assert_eq!(mib_or("X", None, 5 << 20), 5 << 20);
        assert_eq!(mib_or("X", Some("64"), 5 << 20), 64 << 20);
        assert_eq!(mib_or("X", Some("0"), 5 << 20), 0);
        assert_eq!(mib_or("X", Some("lots"), 5 << 20), 5 << 20);
    }

    /// Each runtime switch against a frozen copy of the expression it
    /// replaced (`std::env::var(name)` turned into `raw`), with the default
    /// the module table documents. The copies stay verbatim, lints aside.
    #[test]
    #[allow(clippy::redundant_guards, clippy::nonminimal_bool)]
    fn runtime_switches_keep_their_parse() {
        use crate::vk;
        let old_mib = |raw: Option<&str>, default: u64| match raw {
            Some(s) => s
                .trim()
                .parse::<u64>()
                .ok()
                .map(|mb| mb.saturating_mul(1 << 20))
                .unwrap_or(default),
            None => default,
        };
        same(&crate::profile::VOXEL_PROFILE, false, |raw| {
            raw.is_some_and(|v| v != "0")
        });
        same(&crate::profile::VOXEL_PROFILE_FLUSH_MS, 1.0, |raw| {
            raw.and_then(|v| v.parse::<f64>().ok())
                .map(|ms| ms / 1000.0)
                .unwrap_or(1.0)
        });
        same(&vk::shader_stats::VOXEL_SHADER_STATS, 0, |raw| match raw {
            Some(v) if v == "0" => 0,
            Some(v) if v == "2" => 2,
            Some(_) => 1,
            None => 0,
        });
        same(&vk::uniforms::VOXEL_SKY_DEBUG, false, |raw| {
            raw.is_some_and(|v| v == "1")
        });
        for on in [
            &vk::device::VOXEL_SKY_COARSE,
            &vk::far_bodies::VOXEL_FAR_CULL,
            &vk::far_bodies::VOXEL_SKY_MAPSOLO,
            &vk::far_bodies::VOXEL_SKY_ROUNDSOLO,
        ] {
            same(on, true, |raw| !raw.is_some_and(|v| v == "0"));
        }
        same(&vk::cull_math::VOXEL_LOD_BUCKET_SCALE, 32.0, |raw| {
            raw.and_then(|s| s.parse().ok())
                .filter(|s: &f32| s.is_finite() && *s > 0.0)
                .unwrap_or(32.0)
        });
        same(&vk::cull_math::VOXEL_CPU_CULL_MAX, 1024, |raw| {
            raw.and_then(|v| v.parse().ok()).unwrap_or(1024)
        });
        same(&vk::submit::VOXEL_SUBMIT_BATCH, 2, |raw| {
            raw.and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(2)
                .clamp(1, 2)
        });
        same(
            &vk::submit::VOXEL_EAGER_FLUSH_US,
            Duration::from_micros(15),
            |raw| Duration::from_micros(raw.and_then(|v| v.parse::<u64>().ok()).unwrap_or(15)),
        );
        same(&vk::VOXEL_BENCH_EMPTY, 0, |raw| {
            let Some(v) = raw else {
                return 0;
            };
            v.parse::<u32>().ok().filter(|&k| k >= 1).unwrap_or(0)
        });
        same(&vk::targets::VOXEL_HDR_11BIT, false, |raw| {
            raw.is_some_and(|v| v != "0")
        });
        same(&vk::taa::VOXEL_TAA_MOTION_BOOST, 1.0, |raw| {
            raw.and_then(|s| s.parse().ok()).unwrap_or(1.0)
        });
        same(&vk::taa::VOXEL_TAA_MOTION_PX, 8.0, |raw| {
            raw.and_then(|s| s.parse().ok()).unwrap_or(8.0)
        });
        same(&vk::mesh_staging::VOXEL_MESH_STAGING_MB, 32 << 20, |raw| {
            old_mib(raw, 32 << 20)
        });
        same(&vk::compute::VOXEL_COMPUTE_INPUT_MB, 16 << 20, |raw| {
            old_mib(raw, 16 << 20)
        });
        same(&vk::compute::VOXEL_COMPUTE_READBACK_MB, 32 << 20, |raw| {
            old_mib(raw, 32 << 20)
        });
        same(
            &vk::instance::VOXEL_ENGINE_VALIDATION,
            cfg!(debug_assertions),
            |raw| match raw {
                Some(v) => {
                    v != "0" && !v.eq_ignore_ascii_case("false") && !v.eq_ignore_ascii_case("off")
                }
                None => cfg!(debug_assertions),
            },
        );
    }

    /// Switch names in the `## Runtime` table, in order.
    fn runtime_table_names() -> Vec<&'static str> {
        let src = include_str!("switches.rs");
        let start = src.find("//! ## Runtime").expect("runtime table heading");
        let end = start
            + src[start..]
                .find("//! ## Build")
                .expect("build table heading");
        src[start..end]
            .lines()
            .filter_map(|l| l.strip_prefix("//! | `"))
            .filter_map(|l| l.split('`').next())
            .collect()
    }

    #[test]
    fn runtime_table_lists_every_logged_switch_in_order() {
        let logged: Vec<&str> = RUNTIME.iter().map(|e| e.name()).collect();
        assert_eq!(runtime_table_names(), logged);
        let mut unique = logged.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), logged.len(), "a switch is listed twice");
        assert!(logged.iter().all(|n| n.starts_with("VOXEL_")));
    }

    struct Fake(&'static str, Option<&'static str>);

    impl Logged for Fake {
        fn name(&self) -> &'static str {
            self.0
        }

        fn non_default(&self) -> bool {
            self.1.is_some()
        }

        fn raw(&self) -> Option<String> {
            self.1.map(str::to_owned)
        }
    }

    #[test]
    fn summary_lists_only_non_default_switches() {
        assert_eq!(summary(&[]), "");
        assert_eq!(
            summary(&[&Fake("VOXEL_A", None), &Fake("VOXEL_B", None)]),
            ""
        );
        assert_eq!(
            summary(&[
                &Fake("VOXEL_A", Some("0")),
                &Fake("VOXEL_B", None),
                &Fake("VOXEL_C", Some("1.5")),
            ]),
            "VOXEL_A=0 VOXEL_C=1.5"
        );
        assert_eq!(
            summary(&[&Fake("VOXEL_A", Some("")), &Fake("VOXEL_B", Some(" 64 "))]),
            r#"VOXEL_A="" VOXEL_B=" 64 ""#
        );
    }
}
