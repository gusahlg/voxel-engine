// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
};

use crate::{
    BuildError, Options, ShaderJob,
    includes::include_closure,
    toolchain::{Toolchain, cargo_rerun_env, detect_toolchain, require_pin},
};

/// 64-bit FNV-1a: stable across Rust versions (unlike `DefaultHasher`), so a
/// cache written by one toolchain stays meaningful to the next.
pub(crate) fn fnv1a(bytes: &[u8], mut hash: u64) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Format a `-D` argument: `-DNAME` or `-DNAME=value`.
pub fn define_arg(name: &str, value: Option<&str>) -> String {
    match value {
        None => format!("-D{name}"),
        Some(v) => format!("-D{name}={v}"),
    }
}

pub(crate) fn path_arg(path: &Path) -> String {
    path.to_str()
        .unwrap_or_else(|| panic!("shader path must be UTF-8: {}", path.display()))
        .to_owned()
}

/// Exact `slangc` argv (without `-o`) for `job` under `opts`.
pub fn slangc_args(job: &ShaderJob<'_>, opts: &Options) -> Vec<String> {
    let mut args = vec![
        path_arg(job.src),
        "-target".into(),
        "spirv".into(),
        "-profile".into(),
        opts.spirv_profile.to_owned(),
        "-entry".into(),
        job.entry.to_owned(),
        "-stage".into(),
        job.stage.as_str().to_owned(),
        "-matrix-layout-column-major".into(),
    ];
    if let Some(opt) = opts.opt_level {
        args.push(opt.to_owned());
    }
    for (name, value) in job.defines {
        args.push(define_arg(name, *value));
    }
    args.extend(job.extra_args.iter().map(|a| (*a).to_owned()));
    for dir in &opts.include_dirs {
        args.push("-I".into());
        args.push(path_arg(dir));
    }
    args
}

/// Cache key for one compile: toolchain banner, exact argv, validator, and the
/// path + contents of the source and every transitive include.
pub(crate) fn fingerprint(
    toolchain: &Toolchain,
    args: &[String],
    src: &Path,
    include_dirs: &[PathBuf],
    spirv_val_target_env: &str,
) -> String {
    let mut hash = fnv1a(toolchain.slangc_version.as_bytes(), 0xcbf2_9ce4_8422_2325);
    hash = fnv1a(spirv_val_target_env.as_bytes(), hash);
    hash = fnv1a(&[u8::from(toolchain.spirv_val)], hash);
    for arg in args {
        hash = fnv1a(arg.as_bytes(), hash);
        hash = fnv1a(b"\0", hash);
    }
    for path in include_closure(src, include_dirs) {
        hash = fnv1a(path.to_string_lossy().as_bytes(), hash);
        hash = fnv1a(b"\0", hash);
        let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        hash = fnv1a(&bytes, hash);
        hash = fnv1a(b"\0", hash);
    }
    format!("{hash:016x}")
}

/// Write `fresh` to `fallback` only when the committed bytes differ, so a
/// refresh that produces an identical module leaves `shaders_spv/` untouched.
pub fn copy_if_changed(compiled: &Path, fallback: &Path) {
    let fresh = fs::read(compiled)
        .unwrap_or_else(|e| panic!("read compiled shader {}: {e}", compiled.display()));
    match fs::read(fallback) {
        Ok(existing) if existing == fresh => return,
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => panic!("read shader fallback {}: {e}", fallback.display()),
    }
    fs::write(fallback, fresh)
        .unwrap_or_else(|e| panic!("write shader fallback {}: {e}", fallback.display()));
}

pub(crate) fn compile_one(
    toolchain: Option<&Toolchain>,
    job: &ShaderJob<'_>,
    opts: &Options,
) -> Result<PathBuf, String> {
    let out_path = opts.out_dir.join(job.out_name);
    let fallback_path = opts.fallback_dir.join(job.out_name);
    // Fingerprint of the last successful (compiled + validated) build of
    // `out_path`; skip the compile when nothing feeding it has changed.
    let stamp_path = opts.out_dir.join(format!("{}.fingerprint", job.out_name));

    let Some(toolchain) = toolchain else {
        if !fallback_path.exists() {
            return Err(format!(
                "slangc not found and no prebuilt {} — install Slang or restore shaders_spv/",
                fallback_path.display()
            ));
        }
        // Stale fallback: a committed .spv that is not a SPIR-V module at all.
        // Without a compiler we cannot rebuild, so fail loudly rather than
        // embedding garbage.
        let bytes = fs::read(&fallback_path)
            .map_err(|e| format!("read shader fallback {}: {e}", fallback_path.display()))?;
        if let Err(e) = crate::inspect_spirv(&bytes) {
            return Err(format!(
                "slangc not found and stale prebuilt {} ({e}) — install Slang or restore shaders_spv/",
                fallback_path.display()
            ));
        }
        fs::copy(&fallback_path, &out_path).map_err(|e| {
            format!(
                "copy {} -> {}: {e}",
                fallback_path.display(),
                out_path.display()
            )
        })?;
        // The fallback is not a compile of the current source: forget any
        // stamp so a later toolchain install rebuilds instead of trusting it.
        let _ = fs::remove_file(&stamp_path);
        return Ok(out_path);
    };

    let args = slangc_args(job, opts);
    let fingerprint = fingerprint(
        toolchain,
        &args,
        job.src,
        &opts.include_dirs,
        opts.spirv_val_target_env,
    );
    if out_path.exists() && fs::read_to_string(&stamp_path).is_ok_and(|s| s == fingerprint) {
        return Ok(out_path);
    }
    // Never leave a stale stamp next to a module we are about to rewrite.
    let _ = fs::remove_file(&stamp_path);

    let output = Command::new(&toolchain.slangc)
        .args(&args)
        .arg("-o")
        .arg(&out_path)
        .output()
        .map_err(|e| format!("failed to run slangc for {}: {e}", job.src.display()))?;
    if !output.status.success() {
        return Err(format!(
            "slangc failed while compiling {} (entry {})\n--- stdout ---\n{}\n--- stderr ---\n{}",
            job.src.display(),
            job.entry,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    if opts.validate && toolchain.spirv_val {
        crate::run_spirv_val(&out_path, opts.spirv_val_target_env).map_err(|stderr| {
            format!(
                "spirv-val rejected {} ({} from {}):\n{stderr}",
                job.out_name,
                job.entry,
                job.src.display()
            )
        })?;
    }

    fs::write(&stamp_path, fingerprint)
        .map_err(|e| format!("write {}: {e}", stamp_path.display()))?;
    Ok(out_path)
}

/// Compile every job on a bounded worker pool (one `slangc` process each).
/// Failures are collected and reported together so one broken shader does not
/// hide the diagnostics of another.
///
/// Writes `{out_name}` into [`Options::out_dir`]. When `slangc` is missing,
/// copies committed fallbacks from [`Options::fallback_dir`]. When
/// [`Options::refresh`] is set, copies successful compiles back into the
/// fallback directory after every job has finished (so a compiler failure
/// cannot leave a half-refreshed inventory).
pub fn compile_all(jobs: &[ShaderJob<'_>], opts: &Options) -> Result<Vec<PathBuf>, BuildError> {
    cargo_rerun_env(opts.refresh_env);
    cargo_rerun_env(crate::toolchain::SLANGC_ENV);
    crate::includes::emit_rerun_for_jobs(jobs, &opts.include_dirs, &opts.fallback_dir);

    let toolchain = detect_toolchain();
    if opts.refresh {
        let Some(toolchain) = toolchain.as_ref() else {
            return Err(BuildError::msg(format!(
                "{}=1 requires slangc",
                opts.refresh_env
            )));
        };
        if let Some(pin) = opts.pinned_version {
            require_pin(toolchain, opts.refresh_env, pin).map_err(BuildError::msg)?;
        }
        fs::create_dir_all(&opts.fallback_dir).expect("create shader fallback directory");
    }

    if jobs.is_empty() {
        return Ok(Vec::new());
    }

    let workers = thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len())
        .max(1);
    let next = AtomicUsize::new(0);
    let failures: Vec<String> = thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| {
                    let mut errors = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(job) = jobs.get(index) else {
                            break;
                        };
                        if let Err(e) = compile_one(toolchain.as_ref(), job, opts) {
                            errors.push(e);
                        }
                    }
                    errors
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("shader compile worker panicked"))
            .collect()
    });
    if !failures.is_empty() {
        return Err(BuildError::msg(format!(
            "shader compilation failed:\n\n{}",
            failures.join("\n\n")
        )));
    }

    // Defer every source-tree write until all requested shaders have compiled.
    if opts.refresh {
        for job in jobs {
            copy_if_changed(
                &opts.out_dir.join(job.out_name),
                &opts.fallback_dir.join(job.out_name),
            );
            let compiled = fs::read(opts.out_dir.join(job.out_name)).unwrap_or_else(|e| {
                panic!(
                    "read compiled shader {}: {e}",
                    opts.out_dir.join(job.out_name).display()
                )
            });
            let fallback = fs::read(opts.fallback_dir.join(job.out_name)).unwrap_or_else(|e| {
                panic!(
                    "read shader fallback {}: {e}",
                    opts.fallback_dir.join(job.out_name).display()
                )
            });
            assert!(
                compiled == fallback,
                "refreshed fallback {} is not identical to the compiled module",
                job.out_name
            );
        }
    }

    Ok(jobs
        .iter()
        .map(|job| opts.out_dir.join(job.out_name))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Stage, inspect_spirv};
    use std::path::Path;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir()
            .join("voxel_slang_build_compile")
            .join(name);
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn dummy_spirv() -> Vec<u8> {
        let words = [crate::SPIRV_MAGIC, crate::SPIRV_1_6, 0, 0, 0];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    #[test]
    fn define_arg_formats_name_and_optional_value() {
        assert_eq!(
            define_arg("WATER_DEPTH_ABSORPTION", None),
            "-DWATER_DEPTH_ABSORPTION"
        );
        assert_eq!(define_arg("N", Some("1")), "-DN=1");
    }

    #[test]
    fn slangc_args_match_engine_shipping_invocation() {
        let job = ShaderJob {
            src: Path::new("shaders/mesh3d.frag.slang"),
            stage: Stage::Fragment,
            entry: "fragmentMain",
            defines: &[("MESH3D_OPAQUE", None), ("MESH3D_LOD", None)],
            extra_args: &[],
            out_name: "mesh3d_lod.frag.spv",
        };
        let opts = Options::new("out", "shaders_spv");
        let args = slangc_args(&job, &opts);
        assert_eq!(
            args,
            [
                "shaders/mesh3d.frag.slang",
                "-target",
                "spirv",
                "-profile",
                "spirv_1_6",
                "-entry",
                "fragmentMain",
                "-stage",
                "fragment",
                "-matrix-layout-column-major",
                "-O2",
                "-DMESH3D_OPAQUE",
                "-DMESH3D_LOD",
            ]
        );
    }

    #[test]
    fn slangc_args_append_extra_args_include_dirs_and_skip_opt() {
        let job = ShaderJob {
            src: Path::new("shaders/cull.comp.slang"),
            stage: Stage::Compute,
            entry: "computeMain",
            defines: &[("USE_WAVE_ATOMICS", None)],
            extra_args: &["-capability", "subgroup_basic_ballot"],
            out_name: "cull_wave.comp.spv",
        };
        let mut opts = Options::new("out", "shaders_spv");
        opts.opt_level = None;
        opts.include_dirs = vec![PathBuf::from("shaders")];
        opts.spirv_profile = "spirv_1_5";
        let args = slangc_args(&job, &opts);
        assert_eq!(
            args,
            [
                "shaders/cull.comp.slang",
                "-target",
                "spirv",
                "-profile",
                "spirv_1_5",
                "-entry",
                "computeMain",
                "-stage",
                "compute",
                "-matrix-layout-column-major",
                "-DUSE_WAVE_ATOMICS",
                "-capability",
                "subgroup_basic_ballot",
                "-I",
                "shaders",
            ]
        );
    }

    #[test]
    fn fnv1a_is_stable() {
        assert_eq!(
            fnv1a(b"hello", 0xcbf2_9ce4_8422_2325),
            fnv1a(b"hello", 0xcbf2_9ce4_8422_2325)
        );
        assert_ne!(
            fnv1a(b"hello", 0xcbf2_9ce4_8422_2325),
            fnv1a(b"world", 0xcbf2_9ce4_8422_2325)
        );
    }

    #[test]
    fn fingerprint_changes_when_source_changes() {
        let dir = scratch("fp");
        let src = dir.join("a.slang");
        fs::write(&src, "void main() {}\n").unwrap();
        let tc = Toolchain {
            slangc: PathBuf::from("slangc"),
            slangc_version: "v2025.22.1-nixpkgs".into(),
            spirv_val: true,
        };
        let args = vec!["a.slang".into(), "-O2".into()];
        let a = fingerprint(&tc, &args, &src, &[], "vulkan1.3");
        let b = fingerprint(&tc, &args, &src, &[], "vulkan1.3");
        assert_eq!(a, b);
        fs::write(&src, "void main() { }\n").unwrap();
        let c = fingerprint(&tc, &args, &src, &[], "vulkan1.3");
        assert_ne!(a, c);
    }

    #[test]
    fn compile_one_copies_fallback_when_compiler_absent() {
        let dir = scratch("fb-ok");
        let out = dir.join("out");
        let fb = dir.join("fb");
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&fb).unwrap();
        let spv = dummy_spirv();
        fs::write(fb.join("tri.vert.spv"), &spv).unwrap();
        let opts = Options::new(&out, &fb);
        let job = ShaderJob {
            src: Path::new("missing.vert.slang"),
            stage: Stage::Vertex,
            entry: "vertexMain",
            defines: &[],
            extra_args: &[],
            out_name: "tri.vert.spv",
        };
        let path = compile_one(None, &job, &opts).unwrap();
        assert_eq!(fs::read(&path).unwrap(), spv);
        assert!(!out.join("tri.vert.spv.fingerprint").exists());
    }

    #[test]
    fn compile_one_fails_when_fallback_missing() {
        let dir = scratch("fb-miss");
        let opts = Options::new(dir.join("out"), dir.join("fb"));
        fs::create_dir_all(&opts.out_dir).unwrap();
        fs::create_dir_all(&opts.fallback_dir).unwrap();
        let job = ShaderJob {
            src: Path::new("missing.vert.slang"),
            stage: Stage::Vertex,
            entry: "vertexMain",
            defines: &[],
            extra_args: &[],
            out_name: "tri.vert.spv",
        };
        let err = compile_one(None, &job, &opts).unwrap_err();
        assert!(err.contains("slangc not found and no prebuilt"));
        assert!(err.contains("tri.vert.spv"));
    }

    #[test]
    fn compile_one_fails_when_fallback_is_stale_garbage() {
        let dir = scratch("fb-stale");
        let out = dir.join("out");
        let fb = dir.join("fb");
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&fb).unwrap();
        fs::write(fb.join("tri.vert.spv"), b"not spirv").unwrap();
        let opts = Options::new(&out, &fb);
        let job = ShaderJob {
            src: Path::new("missing.vert.slang"),
            stage: Stage::Vertex,
            entry: "vertexMain",
            defines: &[],
            extra_args: &[],
            out_name: "tri.vert.spv",
        };
        let err = compile_one(None, &job, &opts).unwrap_err();
        assert!(err.contains("stale prebuilt"));
        assert!(err.contains("tri.vert.spv"));
    }

    #[test]
    fn copy_if_changed_skips_identical_and_writes_when_different() {
        let dir = scratch("copy");
        let compiled = dir.join("compiled.spv");
        let fallback = dir.join("fallback.spv");
        let bytes = dummy_spirv();
        fs::write(&compiled, &bytes).unwrap();
        fs::write(&fallback, &bytes).unwrap();
        let before = fs::metadata(&fallback).unwrap().modified().unwrap();
        copy_if_changed(&compiled, &fallback);
        let after = fs::metadata(&fallback).unwrap().modified().unwrap();
        assert_eq!(before, after);

        let mut other = bytes.clone();
        other[8] = 1;
        fs::write(&compiled, &other).unwrap();
        copy_if_changed(&compiled, &fallback);
        assert_eq!(fs::read(&fallback).unwrap(), other);
        inspect_spirv(&other).unwrap();
    }

    #[test]
    fn compile_all_refresh_without_slangc_fails() {
        if detect_toolchain().is_some() {
            // Force the refresh-requires-compiler check by using a refresh
            // env that compile_all reads only for the error string; the
            // function still calls detect_toolchain. When slangc *is* on
            // PATH this test cannot observe the missing-compiler branch.
            return;
        }
        let dir = scratch("refresh-noslangc");
        let mut opts = Options::new(dir.join("out"), dir.join("fb"));
        opts.refresh = true;
        fs::create_dir_all(&opts.out_dir).unwrap();
        let err = compile_all(&[], &opts).unwrap_err();
        assert!(err.to_string().contains("=1 requires slangc"));
    }

    #[test]
    fn compile_trivial_compute_when_slangc_available() {
        let Some(_) = detect_toolchain() else {
            eprintln!("slangc not on PATH: skipping compile_trivial_compute");
            return;
        };
        let dir = scratch("real-compile");
        let src = dir.join("trivial.comp.slang");
        fs::write(
            &src,
            r#"[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain() {}
"#,
        )
        .unwrap();
        let out = dir.join("out");
        let fb = dir.join("fb");
        fs::create_dir_all(&out).unwrap();
        fs::create_dir_all(&fb).unwrap();
        let mut opts = Options::new(&out, &fb);
        opts.refresh = true;
        let job = ShaderJob {
            src: &src,
            stage: Stage::Compute,
            entry: "computeMain",
            defines: &[],
            extra_args: &[],
            out_name: "trivial.comp.spv",
        };
        let paths = compile_all(&[job], &opts).unwrap();
        assert_eq!(paths.len(), 1);
        let bytes = fs::read(&paths[0]).unwrap();
        let info = inspect_spirv(&bytes).unwrap();
        assert_eq!(info.version, crate::SPIRV_1_6);
        assert_eq!(fs::read(fb.join("trivial.comp.spv")).unwrap(), bytes);
    }
}
