// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pinned Slang → SPIR-V compile helper used by `voxel_engine` and, via a
//! path dependency, by the game that ships its own Slang sources.
//!
//! # What this crate does
//!
//! [`compile_all`] runs `slangc` the same way the engine `build.rs` used to:
//!
//! - locate `slangc` on `PATH` (or `$SLANGC`), which is how `nix develop`
//!   exposes the pinned `shader-slang` package;
//! - invoke it with a SPIR-V profile, `-O2`, column-major matrices, `-D`
//!   defines, extra args, and `-I` include dirs;
//! - fingerprint the toolchain banner + argv + source/include bytes so
//!   unchanged modules are not rebuilt;
//! - re-validate with `spirv-val` when that tool is on PATH;
//! - copy committed `.spv` fallbacks when `slangc` is missing, and fail
//!   loudly when a fallback is missing or not SPIR-V;
//! - when [`Options::refresh`] is set (the engine uses
//!   `VOXEL_ENGINE_REFRESH_SHADER_FALLBACKS=1`), require the pinned slangc
//!   version, copy compiled modules into the fallback directory, and verify
//!   the committed bytes are identical to the compile.
//!
//! Generated constants, uniform-lane tables, and the engine shader list stay
//! in the engine `build.rs`. This crate is only the compiler driver.
//!
//! # Example (`build.rs`)
//!
//! ```no_run
//! use std::path::{Path, PathBuf};
//! use voxel_slang_build::{Options, ShaderJob, Stage, compile_all, env_flag};
//!
//! let mut opts = Options::new(
//!     PathBuf::from(std::env::var("OUT_DIR").unwrap()),
//!     "shaders_spv",
//! );
//! opts.refresh = env_flag(opts.refresh_env);
//! opts.include_dirs.push(PathBuf::from("shaders"));
//! let jobs = [ShaderJob {
//!     src: Path::new("shaders/mesh3d.vert.slang"),
//!     stage: Stage::Vertex,
//!     entry: "vertexMain",
//!     defines: &[],
//!     extra_args: &[],
//!     out_name: "mesh3d.vert.spv",
//! }];
//! compile_all(&jobs, &opts).unwrap_or_else(|e| panic!("{e}"));
//! ```

#![forbid(unsafe_code)]

mod compile;
mod error;
mod includes;
mod spirv;
mod toolchain;

use std::{
    env,
    path::{Path, PathBuf},
};

pub use compile::{compile_all, copy_if_changed, define_arg, slangc_args};
pub use error::BuildError;
pub use includes::include_closure;
pub use spirv::{
    SPIRV_1_3, SPIRV_1_6, SPIRV_MAGIC, SpirvInfo, inspect_spirv, run_spirv_val, validate_spirv,
    validate_spirv_with_env,
};
pub use toolchain::{
    PINNED_SLANGC_VERSION, SLANGC_ENV, Toolchain, banner_matches_pin, detect_toolchain, env_flag,
    parse_version_banner, spirv_val_available,
};

/// Shipping SPIR-V profile. Vulkan 1.3 guarantees SPIR-V 1.6, and at 1.6 Slang
/// lowers `discard` to `OpDemoteToHelperInvocation` (the quad keeps its
/// derivatives; core-1.3 feature `shaderDemoteToHelperInvocation`) instead of
/// `OpKill`.
pub const SPIRV_PROFILE: &str = "spirv_1_6";
/// `spirv-val` environment for freshly compiled modules.
pub const SPIRV_VAL_TARGET_ENV: &str = "vulkan1.3";
/// Slang optimisation level. With the pinned toolchain (2025.22.1) `-O3`
/// emits byte-identical modules to `-O2`, so take the cheaper compile.
pub const SLANG_OPT_LEVEL: &str = "-O2";
/// Env var that copies compiled modules into the fallback directory. Ordinary
/// builds never rewrite those files.
pub const DEFAULT_REFRESH_ENV: &str = "VOXEL_ENGINE_REFRESH_SHADER_FALLBACKS";

/// Shader stage passed to `slangc -stage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    /// `slangc -stage vertex`
    Vertex,
    /// `slangc -stage fragment`
    Fragment,
    /// `slangc -stage compute`
    Compute,
}

impl Stage {
    /// The `slangc -stage` token.
    pub const fn as_str(self) -> &'static str {
        match self {
            Stage::Vertex => "vertex",
            Stage::Fragment => "fragment",
            Stage::Compute => "compute",
        }
    }
}

/// One compile unit: a Slang source, its entry point, `-D` defines, and the
/// `OUT_DIR` / fallback file name.
#[derive(Clone, Copy, Debug)]
pub struct ShaderJob<'a> {
    /// Slang source path, relative to the package whose `build.rs` is running.
    pub src: &'a Path,
    pub stage: Stage,
    pub entry: &'a str,
    /// `(name, value)` pairs emitted as `-Dname` or `-Dname=value`.
    pub defines: &'a [(&'a str, Option<&'a str>)],
    /// Extra `slangc` args after the defines (`-capability`, …).
    pub extra_args: &'a [&'a str],
    /// File name written to [`Options::out_dir`] and, on refresh, to
    /// [`Options::fallback_dir`] (`mesh3d.vert.spv`, …).
    pub out_name: &'a str,
}

impl<'a> ShaderJob<'a> {
    /// Job with no defines and no extra args.
    pub const fn new(src: &'a Path, stage: Stage, entry: &'a str, out_name: &'a str) -> Self {
        Self {
            src,
            stage,
            entry,
            defines: &[],
            extra_args: &[],
            out_name,
        }
    }
}

/// Compile options shared by every [`ShaderJob`] in a [`compile_all`] call.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct Options {
    /// Destination directory. The engine passes `OUT_DIR`.
    pub out_dir: PathBuf,
    /// Directory of committed `.spv` fallbacks (`shaders_spv/` for the engine).
    pub fallback_dir: PathBuf,
    /// Extra `-I` paths for `#include` / `import`.
    pub include_dirs: Vec<PathBuf>,
    /// Env var that, when `1`, the engine sets [`Self::refresh`] from.
    /// Also emitted as `cargo:rerun-if-env-changed`.
    pub refresh_env: &'static str,
    /// When `true`, copy compiled modules into [`Self::fallback_dir`] after a
    /// successful compile and require [`Self::pinned_version`].
    pub refresh: bool,
    /// `-profile` token (engine: [`SPIRV_PROFILE`]).
    pub spirv_profile: &'static str,
    /// Optimisation flag (engine: [`SLANG_OPT_LEVEL`]). `None` omits it
    /// (probe shaders).
    pub opt_level: Option<&'static str>,
    /// `spirv-val --target-env` (engine: [`SPIRV_VAL_TARGET_ENV`]).
    pub spirv_val_target_env: &'static str,
    /// Required `slangc -v` substring when [`Self::refresh`] is set.
    /// Engine: [`PINNED_SLANGC_VERSION`]. `None` skips the pin check.
    pub pinned_version: Option<&'static str>,
    /// Run `spirv-val` on freshly compiled modules when the tool is on PATH.
    pub validate: bool,
}

impl Options {
    /// Engine / game defaults: SPIR-V 1.6, `-O2`, vulkan1.3 `spirv-val`,
    /// pinned slangc [`PINNED_SLANGC_VERSION`], refresh via
    /// [`DEFAULT_REFRESH_ENV`].
    pub fn new(out_dir: impl Into<PathBuf>, fallback_dir: impl Into<PathBuf>) -> Self {
        Self {
            out_dir: out_dir.into(),
            fallback_dir: fallback_dir.into(),
            include_dirs: Vec::new(),
            refresh_env: DEFAULT_REFRESH_ENV,
            refresh: false,
            spirv_profile: SPIRV_PROFILE,
            opt_level: Some(SLANG_OPT_LEVEL),
            spirv_val_target_env: SPIRV_VAL_TARGET_ENV,
            pinned_version: Some(PINNED_SLANGC_VERSION),
            validate: true,
        }
    }
}

impl Default for Options {
    fn default() -> Self {
        Self::new(
            env::var_os("OUT_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(".")),
            "shaders_spv",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_tokens_match_slangc() {
        assert_eq!(Stage::Vertex.as_str(), "vertex");
        assert_eq!(Stage::Fragment.as_str(), "fragment");
        assert_eq!(Stage::Compute.as_str(), "compute");
    }

    #[test]
    fn shader_job_new_has_empty_defines() {
        let job = ShaderJob::new(
            Path::new("a.slang"),
            Stage::Vertex,
            "vertexMain",
            "a.vert.spv",
        );
        assert!(job.defines.is_empty());
        assert!(job.extra_args.is_empty());
    }

    #[test]
    fn options_new_matches_engine_defaults() {
        let opts = Options::new("out", "shaders_spv");
        assert_eq!(opts.spirv_profile, "spirv_1_6");
        assert_eq!(opts.opt_level, Some("-O2"));
        assert_eq!(opts.spirv_val_target_env, "vulkan1.3");
        assert_eq!(opts.refresh_env, "VOXEL_ENGINE_REFRESH_SHADER_FALLBACKS");
        assert_eq!(opts.pinned_version, Some("2025.22.1"));
        assert!(opts.validate);
        assert!(!opts.refresh);
        assert!(opts.include_dirs.is_empty());
    }
}
