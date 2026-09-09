// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Version substring the engine and game must share when writing fallbacks.
///
/// `nix develop` provides `shader-slang` whose `slangc -v` banner is
/// `v2025.22.1-nixpkgs`. Ordinary compiles accept any `slangc` (the
/// fingerprint includes the banner so a bump rebuilds); refreshing
/// `shaders_spv/` requires this pin so committed SPIR-V stays byte-stable.
pub const PINNED_SLANGC_VERSION: &str = "2025.22.1";

/// Environment variable that, when set, overrides PATH lookup for `slangc`.
pub const SLANGC_ENV: &str = "SLANGC";

/// The shader toolchain found on PATH (or [`SLANGC_ENV`]). `None` from
/// [`detect_toolchain`] means `slangc` is missing and committed `.spv`
/// fallbacks must be used verbatim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Toolchain {
    /// Binary invoked as `slangc` (the string `"slangc"` when resolved via PATH,
    /// or the `SLANGC` override).
    pub slangc: PathBuf,
    /// `slangc -v` banner; part of the compile-cache key so a toolchain bump
    /// recompiles everything even when no source changed.
    pub slangc_version: String,
    /// `spirv-val` on PATH (vulkan-tools in the nix shell): every freshly
    /// compiled module is validated and the build fails on invalid SPIR-V.
    pub spirv_val: bool,
}

/// Locate `slangc` the way a `nix develop` shell does: honour `SLANGC` if
/// set, otherwise run the `slangc` on `PATH`. Returns `None` when the binary
/// is missing or `-v` fails.
pub fn detect_toolchain() -> Option<Toolchain> {
    let slangc = slangc_binary();
    let output = Command::new(&slangc).arg("-v").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let slangc_version = parse_version_banner(&output.stdout, &output.stderr);
    let spirv_val = spirv_val_available();
    Some(Toolchain {
        slangc,
        slangc_version,
        spirv_val,
    })
}

/// `true` when `spirv-val --version` succeeds.
pub fn spirv_val_available() -> bool {
    Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Concatenate `slangc -v` streams and trim. slangc prints its version on
/// stderr; both streams are taken to be safe.
pub fn parse_version_banner(stdout: &[u8], stderr: &[u8]) -> String {
    let mut slangc_version = String::from_utf8_lossy(stdout).into_owned();
    slangc_version.push_str(&String::from_utf8_lossy(stderr));
    slangc_version.trim().to_owned()
}

/// `true` when `banner` contains the pinned version substring.
pub fn banner_matches_pin(banner: &str, pin: &str) -> bool {
    banner.contains(pin)
}

pub(crate) fn slangc_binary() -> PathBuf {
    match env::var_os(SLANGC_ENV) {
        Some(explicit) => PathBuf::from(explicit),
        None => PathBuf::from("slangc"),
    }
}

pub(crate) fn pin_error(refresh_env: &str, pin: &str, banner: &str) -> String {
    format!("{refresh_env}=1 requires slangc {pin} (nix develop); found {banner}")
}

pub(crate) fn require_pin(
    toolchain: &Toolchain,
    refresh_env: &str,
    pin: &str,
) -> Result<(), String> {
    if banner_matches_pin(&toolchain.slangc_version, pin) {
        Ok(())
    } else {
        Err(pin_error(refresh_env, pin, &toolchain.slangc_version))
    }
}

/// Read a `0`/`1` cargo env flag. Missing is `false`. Any other value panics
/// with the same message the engine `build.rs` used.
pub fn env_flag(name: &str) -> bool {
    env_flag_from(name, env::var(name))
}

pub(crate) fn env_flag_from(name: &str, var: Result<String, env::VarError>) -> bool {
    match var {
        Err(env::VarError::NotPresent) => false,
        Ok(value) if value == "0" => false,
        Ok(value) if value == "1" => true,
        Ok(value) => panic!("{name} must be 0 or 1, got {value:?}"),
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must be valid UTF-8 and either 0 or 1"),
    }
}

pub(crate) fn cargo_rerun_env(name: &str) {
    println!("cargo:rerun-if-env-changed={name}");
}

pub(crate) fn cargo_rerun_path(path: &Path) {
    println!("cargo:rerun-if-changed={}", path.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_version_banner_trims_and_joins_streams() {
        assert_eq!(
            parse_version_banner(b"  v2025.22.1-nixpkgs\n", b""),
            "v2025.22.1-nixpkgs"
        );
        assert_eq!(
            parse_version_banner(b"", b"v2025.22.1-nixpkgs\n"),
            "v2025.22.1-nixpkgs"
        );
        assert_eq!(parse_version_banner(b"pre-", b"post\n"), "pre-post");
    }

    #[test]
    fn nixpkgs_banner_matches_pin() {
        assert!(banner_matches_pin(
            "v2025.22.1-nixpkgs",
            PINNED_SLANGC_VERSION
        ));
        assert!(banner_matches_pin("2025.22.1", PINNED_SLANGC_VERSION));
        assert!(!banner_matches_pin("v2024.1.0", PINNED_SLANGC_VERSION));
        assert!(!banner_matches_pin("", PINNED_SLANGC_VERSION));
    }

    #[test]
    fn env_flag_from_accepts_0_1_and_absent() {
        assert!(!env_flag_from("X", Err(env::VarError::NotPresent)));
        assert!(!env_flag_from("X", Ok("0".into())));
        assert!(env_flag_from("X", Ok("1".into())));
    }

    #[test]
    #[should_panic(expected = "X must be 0 or 1, got \"2\"")]
    fn env_flag_from_rejects_other_values() {
        let _ = env_flag_from("X", Ok("2".into()));
    }

    #[test]
    fn require_pin_reports_refresh_env_and_found_banner() {
        let tc = Toolchain {
            slangc: PathBuf::from("slangc"),
            slangc_version: "v1.0.0".into(),
            spirv_val: false,
        };
        let err = require_pin(&tc, "REFRESH", "2025.22.1").unwrap_err();
        assert!(err.contains("REFRESH=1 requires slangc 2025.22.1"));
        assert!(err.contains("v1.0.0"));
    }
}
