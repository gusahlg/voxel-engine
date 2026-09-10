use std::{
    env, fs,
    path::{Path, PathBuf},
};

use voxel_slang_build::{Options, ShaderJob, Stage, compile_all, detect_toolchain, env_flag};

fn job(
    src: &'static str,
    stage: Stage,
    entry: &'static str,
    out: &'static str,
) -> ShaderJob<'static> {
    ShaderJob::new(Path::new(src), stage, entry, out)
}

/// Engine shader list plus the define-gated variants. Output names in
/// `OUT_DIR` / `shaders_spv/` must stay stable.
fn shipping_jobs() -> Vec<ShaderJob<'static>> {
    let mut jobs = vec![
        job(
            "shaders/mesh3d.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "mesh3d.vert.spv",
        ),
        job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d.frag.spv",
        ),
        job(
            "shaders/debug.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "debug.vert.spv",
        ),
        job(
            "shaders/debug.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "debug.frag.spv",
        ),
        job(
            "shaders/sky.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "sky.vert.spv",
        ),
        job(
            "shaders/sky.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "sky.frag.spv",
        ),
        job(
            "shaders/sky_cloud.comp.slang",
            Stage::Compute,
            "computeMain",
            "sky_cloud.comp.spv",
        ),
        job(
            "shaders/tonemap.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "tonemap.vert.spv",
        ),
        job(
            "shaders/tonemap.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "tonemap.frag.spv",
        ),
        job(
            "shaders/tris2d.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "tris2d.vert.spv",
        ),
        job(
            "shaders/tris2d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "tris2d.frag.spv",
        ),
        job(
            "shaders/tris2d_tex.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "tris2d_tex.frag.spv",
        ),
        job(
            "shaders/vrs.comp.slang",
            Stage::Compute,
            "computeMain",
            "vrs.comp.spv",
        ),
        job(
            "shaders/shadow_depth.vert.slang",
            Stage::Vertex,
            "vertexMain",
            "shadow_depth.vert.spv",
        ),
        job(
            "shaders/exposure_reduce.comp.slang",
            Stage::Compute,
            "computeMain",
            "exposure_reduce.comp.spv",
        ),
        job(
            "shaders/cull.comp.slang",
            Stage::Compute,
            "computeMain",
            "cull.comp.spv",
        ),
        // Bloom: two entry points from one source (threshold + downsample-chain).
        job(
            "shaders/bloom.comp.slang",
            Stage::Compute,
            "threshold",
            "bloom_threshold.comp.spv",
        ),
        job(
            "shaders/bloom.comp.slang",
            Stage::Compute,
            "downsample",
            "bloom_downsample.comp.spv",
        ),
        // Quarter-res bloom-composite + godrays, sampled once by tonemap.
        job(
            "shaders/spill.comp.slang",
            Stage::Compute,
            "computeMain",
            "spill.comp.spv",
        ),
        job(
            "shaders/compute_example.comp.slang",
            Stage::Compute,
            "computeMain",
            "compute_example.comp.spv",
        ),
    ];
    // Water depth-absorption path. Samples the previous frame's depth
    // (set 0 binding 5) + Δd-driven body tint, compiled only into
    // `mesh3d_transparent_absorb`. The default variant stays the interim-tint
    // fallback.
    jobs.push(ShaderJob {
        defines: &[("WATER_DEPTH_ABSORPTION", None)],
        ..job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d_water.frag.spv",
        )
    });
    // Opaque-only mesh3d.frag: strips water/absorb and the LOD-slab `discard`.
    jobs.push(ShaderJob {
        defines: &[("MESH3D_OPAQUE", None)],
        ..job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d_opaque.frag.spv",
        )
    });
    // Coarse-LOD opaque mesh3d.frag: opaque ALU diet plus the slab-clip `discard`.
    jobs.push(ShaderJob {
        defines: &[("MESH3D_OPAQUE", None), ("MESH3D_LOD", None)],
        ..job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d_lod.frag.spv",
        )
    });
    // Opaque mesh3d.frag with every optional lighting lane compiled out.
    jobs.push(ShaderJob {
        defines: &[("MESH3D_OPAQUE", None), ("MESH3D_LEAN", None)],
        ..job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d_opaque_lean.frag.spv",
        )
    });
    // Coarse-LOD opaque plus the same lane-off diet as the opaque lean variant.
    jobs.push(ShaderJob {
        defines: &[
            ("MESH3D_OPAQUE", None),
            ("MESH3D_LOD", None),
            ("MESH3D_LEAN", None),
        ],
        ..job(
            "shaders/mesh3d.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "mesh3d_lod_lean.frag.spv",
        )
    });
    // Present-time TAA tonemap fragment (`-DTAA_FUSED`).
    jobs.push(ShaderJob {
        defines: &[("TAA_FUSED", None)],
        ..job(
            "shaders/tonemap.frag.slang",
            Stage::Fragment,
            "fragmentMain",
            "tonemap_taa.frag.spv",
        )
    });
    jobs
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=VOXEL_BUILD_PROBE");
    println!("cargo:rerun-if-env-changed=VOXEL_LIGHT_LEGACY");
    // Emitting any rerun-if-changed replaces cargo's default "rerun if any
    // package file changed", so build.rs itself must be listed explicitly —
    // otherwise edits to the CONSTS table below would not regenerate outputs.
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut opts = Options::new(out_dir.clone(), "shaders_spv");
    opts.refresh = env_flag(opts.refresh_env);

    // Single-source the constants shared by Rust and Slang BEFORE compiling any
    // shader: common.slang `#include`s the generated .slang, so it must exist
    // first. See generate_shared_constants for the drift-kill rationale.
    generate_shared_constants(&out_dir);

    // Ensure tunables use generated includes, not hand-written constants.
    lint_slang_constants();

    let jobs = shipping_jobs();
    compile_all(&jobs, &opts).unwrap_or_else(|e| panic!("{e}"));
    // Wave-aggregated InterlockedAdd variant of the cull shader. Not in SHADERS:
    // shaders_spv/ keeps the plain-atomic module (no subgroup caps) as the
    // no-slangc fallback; this file lives only in OUT_DIR. See compile_cull_wave.
    compile_cull_wave(&opts);

    // Substrate probe: compute shaders for BDA, QUAD, STORAGE, and occupancy tests.
    // Gated behind VOXEL_BUILD_PROBE to avoid requiring extended SPIR-V profile.
    if env::var("VOXEL_BUILD_PROBE").is_ok() {
        compile_probe(&opts);
    }
}

/// Compile probe shaders at SPIR-V 1.5 for BDA and subgroup-quad ops.
fn compile_probe(base: &Options) {
    assert!(
        detect_toolchain().is_some(),
        "VOXEL_BUILD_PROBE requires slangc"
    );
    const PROBE_CAPS: &[&str] = &[
        "-capability",
        "spvGroupNonUniformQuad",
        "-capability",
        "spvPhysicalStorageBufferAddresses",
    ];
    let probe_shaders = [
        ShaderJob {
            extra_args: PROBE_CAPS,
            ..job(
                "shaders/probe_bda.comp.slang",
                Stage::Compute,
                "computeMain",
                "probe_bda.comp.spv",
            )
        },
        ShaderJob {
            extra_args: PROBE_CAPS,
            ..job(
                "shaders/probe_quad.comp.slang",
                Stage::Compute,
                "computeMain",
                "probe_quad.comp.spv",
            )
        },
        ShaderJob {
            extra_args: PROBE_CAPS,
            ..job(
                "shaders/probe_storage.comp.slang",
                Stage::Compute,
                "computeMain",
                "probe_storage.comp.spv",
            )
        },
        ShaderJob {
            extra_args: PROBE_CAPS,
            ..job(
                "shaders/probe_occupancy.comp.slang",
                Stage::Compute,
                "computeMain",
                "probe_occupancy.comp.spv",
            )
        },
    ];
    let mut opts = base.clone();
    opts.refresh = false;
    opts.spirv_profile = "spirv_1_5";
    opts.opt_level = None;
    opts.validate = false;
    compile_all(&probe_shaders, &opts).unwrap_or_else(|e| panic!("{e}"));
}

// Slang constant lint: prevent duplicates of tunable values.
// Allow only math facts as hand-written constants, never tunables.

/// Names allowed as hand-written constants because they're math facts, not tunables.
/// Each entry must carry a justification.
const SLANG_CONST_ALLOWLIST: &[(&str, &str)] = &[(
    "FACE_NORMAL",
    "the 6 unit cube-face normals are fixed by the packed-vertex normal \
         index convention (mesh.rs), not a tunable value",
)];

/// Packed-vertex / AO names that `build_table()` must emit and that unpack
/// shaders must reference so a clone cannot silently re-hardcode the bits.
const PACKING_CONST_NAMES: &[&str] = &[
    "SHIFT_X",
    "SHIFT_Y",
    "SHIFT_Z",
    "SHIFT_NORMAL",
    "SHIFT_LAYER",
    "SHIFT_AO",
    "SHIFT_SKY",
    "SHIFT_BLOCK",
    "SHIFT_WATER",
    "SHIFT_MICRO_X",
    "SHIFT_MICRO_Y",
    "SHIFT_MICRO_Z",
    "MASK_COORD",
    "MASK_NORMAL",
    "MASK_LAYER",
    "MASK_AO",
    "MASK_LIGHT",
    "MASK_MICRO",
    "AO_MIN",
    "AO_STEP",
];

/// Lint static consts; allowlist-exempt those referencing generated symbols.
fn lint_slang_constants() {
    let shaders_dir = Path::new("shaders");
    let mut violations = Vec::new();

    for entry in fs::read_dir(shaders_dir).expect("read shaders dir") {
        let entry = entry.expect("read shaders dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("slang") {
            continue;
        }
        lint_slang_file(&path, &mut violations);
    }

    lint_packing_constants(&mut violations);

    if !violations.is_empty() {
        panic!(
            "Slang constant lint failed: hand-written numeric `static const` \
             outside the genconst-generated include. Route tunables through \
             build.rs's build_table(), or add a justified entry to \
             SLANG_CONST_ALLOWLIST for a pure math constant.\n\n{}",
            violations.join("\n")
        );
    }
}

/// Require the packed-vertex / AO names in `build_table()` and in the shaders
/// that unpack vertices, so C4's previous gap (inline hex / 0.4+0.2) cannot
/// return without failing this lint.
fn lint_packing_constants(violations: &mut Vec<String>) {
    let table = build_table();
    let table_names: std::collections::HashSet<&str> = table.iter().map(|d| d.name).collect();
    for name in PACKING_CONST_NAMES {
        if !table_names.contains(name) {
            violations.push(format!(
                "build_table() is missing packing/AO constant `{name}`"
            ));
        }
    }

    const UNPACK_FILES: &[(&str, &[&str])] = &[
        ("shaders/mesh3d.vert.slang", PACKING_CONST_NAMES),
        (
            "shaders/shadow_depth.vert.slang",
            &["SHIFT_X", "SHIFT_Y", "SHIFT_Z", "MASK_COORD"],
        ),
    ];
    for (path, required) in UNPACK_FILES {
        let src = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read {path} for packing-const lint: {e}"));
        for name in *required {
            if !src.contains(name) {
                violations.push(format!(
                    "{path}: unpack shader does not reference generated `{name}`"
                ));
            }
        }
    }
}

fn lint_slang_file(path: &Path, violations: &mut Vec<String>) {
    let src = fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let lines: Vec<&str> = src.lines().collect();

    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        if !trimmed.starts_with("static const") {
            i += 1;
            continue;
        }
        let decl_line = i + 1; // 1-based line numbers for error messages

        // Accumulate multi-line array initializers until the terminating `;`.
        let mut stmt = String::new();
        loop {
            stmt.push_str(lines[i]);
            stmt.push('\n');
            if lines[i].contains(';') || i + 1 >= lines.len() {
                break;
            }
            i += 1;
        }

        // Extract name after type, stripping array suffix if present.
        let after_type = trimmed
            .trim_start_matches("static const")
            .trim_start()
            .split_once(char::is_whitespace)
            .map(|x| x.1)
            .unwrap_or("");
        let name: String = after_type
            .trim_start()
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        let rhs = stmt.split_once('=').map(|x| x.1).unwrap_or("");
        let has_numeric_literal = rhs.char_indices().any(|(idx, c)| {
            c.is_ascii_digit() && !rhs[..idx].ends_with(|p: char| p.is_alphanumeric() || p == '_')
        });

        if has_numeric_literal && !SLANG_CONST_ALLOWLIST.iter().any(|(n, _)| *n == name) {
            violations.push(format!(
                "{}:{decl_line}: static const `{name}` has a hand-written numeric literal \
                 and is not in SLANG_CONST_ALLOWLIST",
                path.display()
            ));
        }

        i += 1;
    }
}

/// Wave-aggregated InterlockedAdd variant of the cull shader. Lives only in
/// OUT_DIR: shaders_spv/ keeps the plain-atomic module (no subgroup caps).
///
/// Without slangc, `cull_wave.comp.spv` is a copy of the plain `cull.comp.spv`.
/// That copy is semantically safe: it is a correct plain-atomic module (same
/// bindings, same push constants, same per-thread InterlockedAdd emit). The
/// Rust side (`device.cull_wave_atomics`) selects `CULL_COMP_WAVE` vs
/// `CULL_COMP` from GPU subgroup BASIC+BALLOT, not by inspecting SPIR-V.
/// Selecting the copy on a wave-capable GPU is equivalent to selecting
/// `CULL_COMP` — results match; only the wave aggregation is skipped. The
/// copy does not declare GroupNonUniform, so validation does not require
/// subgroup features. A dedicated `shaders_spv/cull_wave.comp.spv` is not
/// needed; unlike `-DTAA_FUSED`, a plain clone is not a different interface.
fn compile_cull_wave(base: &Options) {
    let cull_wave = ShaderJob {
        defines: &[("USE_WAVE_ATOMICS", None)],
        extra_args: &["-capability", "subgroup_basic_ballot"],
        ..job(
            "shaders/cull.comp.slang",
            Stage::Compute,
            "computeMain",
            "cull_wave.comp.spv",
        )
    };
    if detect_toolchain().is_some() {
        let mut opts = base.clone();
        opts.refresh = false;
        compile_all(&[cull_wave], &opts).unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    // Plain-atomic clone: see the safety argument on this function.
    let plain = base.out_dir.join("cull.comp.spv");
    let wave = base.out_dir.join(cull_wave.out_name);
    fs::copy(&plain, &wave)
        .unwrap_or_else(|e| panic!("copy plain cull module to {}: {e}", wave.display()));
}

// Shared constants: single source of truth for Rust and Slang.
//
// One table (`build_table`) is rendered into two files that MUST agree:
//   * $OUT_DIR/gen_constants.rs            — `pub const` items (voxel_engine::genconst)
//   * shaders/generated/shader_constants.slang — `static const` items
// Any constant that appears in both CPU-side sky/lighting math and its shader
// twin lives here so CPU↔GPU drift becomes impossible-to-forget instead of a
// silent fog seam. The .slang is `#include`d by common.slang.

enum Val {
    Scalar(f32),
    UInt(u32),
    /// `u32` in Rust (same as [`Val::UInt`]); `static const int` in Slang so
    /// decimal shift literals lower to the same signed SPIR-V constants as
    /// the old untyped `>> 15` forms.
    Int(u32),
    Arr(Vec<f32>),
    Arr2(Vec<[f32; 2]>),
}

struct Def {
    name: &'static str,
    /// Provenance / units; emitted as doc /// and //.
    doc: &'static str,
    val: Val,
}

/// Round-trippable f32 literal. Rust's Debug formatting is shortest-round-trip;
/// the same text is a valid Rust literal and (with an `f` suffix) a valid Slang
/// float literal.
fn lit(x: f32) -> String {
    format!("{x:?}")
}

fn build_table() -> Vec<Def> {
    // VOXEL_LIGHT_LEGACY=1 selects constant values that make Steps 2–4 of the
    // lighting look algebraically identical to the pre-change shaders. Unset
    // (the default) is the new sky-tinted / AO-shaped / blended-cascade look.
    // Step 1 (varying diet) is bit-identical either way.
    let light_legacy = env_flag("VOXEL_LIGHT_LEGACY");

    // Precompute the sRGB decode table once here so CPU and shader agree exactly.
    let mut srgb = Vec::with_capacity(256);
    for v in 0u32..256 {
        let s = v as f32 / 255.0;
        let lin = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        srgb.push(lin);
    }

    // Skip index 0 so no jitter sample lands exactly on the pixel centre.
    let radical_inverse = |mut i: u32, base: u32| -> f32 {
        let mut f = 1.0f32;
        let mut r = 0.0f32;
        while i > 0 {
            f /= base as f32;
            r += f * (i % base) as f32;
            i /= base;
        }
        r
    };
    let mut halton = Vec::with_capacity(16);
    for k in 0u32..16 {
        let i = k + 1;
        halton.push([radical_inverse(i, 2) - 0.5, radical_inverse(i, 3) - 0.5]);
    }

    // Autoexposure curve fitted in EV space (gentle log-linear response, no cliff).
    // Key anchors: exposure(day_luma)=1.0 (day image must match golden), exposure(cave_luma)=6.05.
    // NEVER re-bless day if the fit drifts; it breaks image match.
    let l_day = 0.2114f32;
    let l_cave = 0.00289f32;
    let exposure_cave = 6.05f32;
    let exposure_l_day_log2 = l_day.log2();
    let exposure_ev_slope = exposure_cave.log2() / (exposure_l_day_log2 - l_cave.log2());

    vec![
        Def {
            name: "SRGB8_TO_LINEAR",
            doc: "Exact float sRGB EOTF decode table, indexed by 8-bit code (0..=255).\n[0] == 0.0, [255] == 1.0, strictly monotone increasing.",
            val: Val::Arr(srgb),
        },
        Def {
            name: "HALTON_23",
            doc: "Halton(2,3) low-discrepancy jitter offsets in [-0.5, 0.5), index 1..=16.\nMean ~= 0 per axis. TAA sub-pixel jitter source.",
            val: Val::Arr2(halton),
        },
        Def {
            name: "HISTORY_BLEND",
            doc: "TAA history feedback weight: fraction of the (reprojected, variance-clamped)\nhistory kept each present. Higher = steadier but slower to react. Read by vk/taa.rs\nand pushed into the fused tonemap. Reduced 0.95→0.92: animated water waves and\nclouds demand faster per-frame reactivity. 0.92 (~8% new sample) reduces ghosting\nwhile maintaining temporal coherence. Range [0.85–0.98].",
            val: Val::Scalar(0.92),
        },
        Def {
            name: "VARIANCE_GAMMA",
            doc: "TAA neighbourhood variance-clamp width in std-devs: history is clamped to\nYCoCg mean +/- VARIANCE_GAMMA*stddev of the unweighted 3x3 current taps. Wider =\nsteadier (less crawl) but more ghosting. Read by tonemap.frag (TAA_FUSED).",
            val: Val::Scalar(1.25),
        },
        Def {
            name: "CANDLE_HIGH_MUL",
            doc: "Blocklight candle curve: candle = x*sqrt(x) + (CANDLE_HIGH_MUL*x)^CANDLE_HIGH_POW,\nclamped to CANDLE_CLAMP. x = block light level in [0,1].",
            val: Val::Scalar(1.17),
        },
        Def {
            name: "CANDLE_HIGH_POW",
            doc: "Candle curve high-end exponent. Tentative, pending tuning.",
            val: Val::Scalar(6.0),
        },
        Def {
            name: "CANDLE_CLAMP",
            doc: "Candle curve output clamp ceiling. Tentative, pending tuning.",
            val: Val::Scalar(4.0),
        },
        Def {
            name: "MICRO_STEP",
            doc: "Per-vertex anti-z-fight nudge in local units. Applied before scale so it\ninherits LOD's 2^k scale. Read by mesh3d.vert.",
            val: Val::Scalar(0.01),
        },
        // Packed mesh vertex bit layout. Twin of mesh.rs pack/unpack; the Slang
        // unpack in mesh3d.vert / shadow_depth.vert must use these names.
        Def {
            name: "SHIFT_X",
            doc: "Packed vertex word 0: x-coordinate shift. 5-bit field, values 0..=16.",
            val: Val::Int(0),
        },
        Def {
            name: "SHIFT_Y",
            doc: "Packed vertex word 0: y-coordinate shift. 5-bit field, values 0..=16.",
            val: Val::Int(5),
        },
        Def {
            name: "SHIFT_Z",
            doc: "Packed vertex word 0: z-coordinate shift. 5-bit field, values 0..=16.",
            val: Val::Int(10),
        },
        Def {
            name: "SHIFT_NORMAL",
            doc: "Packed vertex word 0: face-normal index shift. 3 bits: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z.",
            val: Val::Int(15),
        },
        Def {
            name: "SHIFT_LAYER",
            doc: "Packed vertex word 0: texture-array layer shift. 14 bits, word 0's remaining span.",
            val: Val::Int(18),
        },
        Def {
            name: "SHIFT_AO",
            doc: "Packed vertex word 1: AO level shift. 2 bits, 0..=3; 3 = no occlusion.",
            val: Val::Int(0),
        },
        Def {
            name: "SHIFT_SKY",
            doc: "Packed vertex word 1: skylight shift. 4 bits, 0..=15.",
            val: Val::Int(2),
        },
        Def {
            name: "SHIFT_BLOCK",
            doc: "Packed vertex word 1: blocklight shift. 4 bits, 0..=15.",
            val: Val::Int(6),
        },
        Def {
            name: "SHIFT_WATER",
            doc: "Packed vertex word 1: water material bit. Set by the mesher for liquid faces.",
            val: Val::Int(10),
        },
        Def {
            name: "SHIFT_MICRO_X",
            doc: "Packed vertex word 1: x micro-offset shift. 2-bit two's-complement, -2..=1.",
            val: Val::Int(11),
        },
        Def {
            name: "SHIFT_MICRO_Y",
            doc: "Packed vertex word 1: y micro-offset shift. 2-bit two's-complement, -2..=1.",
            val: Val::Int(13),
        },
        Def {
            name: "SHIFT_MICRO_Z",
            doc: "Packed vertex word 1: z micro-offset shift. 2-bit two's-complement, -2..=1.",
            val: Val::Int(15),
        },
        Def {
            name: "MASK_COORD",
            doc: "Packed vertex coordinate mask (5 bits). Holds 0..=16; 17..=31 store without\ncorrupting a neighbour.",
            val: Val::UInt(0x1F),
        },
        Def {
            name: "MASK_NORMAL",
            doc: "Packed vertex face-normal index mask (3 bits).",
            val: Val::UInt(0x7),
        },
        Def {
            name: "MASK_LAYER",
            doc: "Packed vertex texture-array layer mask (14 bits, 16384 layers).",
            val: Val::UInt(0x3FFF),
        },
        Def {
            name: "MASK_AO",
            doc: "Packed vertex AO-level mask (2 bits).",
            val: Val::UInt(0x3),
        },
        Def {
            name: "MASK_LIGHT",
            doc: "Packed vertex skylight / blocklight mask (4 bits).",
            val: Val::UInt(0xF),
        },
        Def {
            name: "MASK_MICRO",
            doc: "Packed vertex per-axis micro-offset mask (2 bits).",
            val: Val::UInt(0x3),
        },
        Def {
            name: "AO_MIN",
            doc: "Baked AO level-0 diffuse multiplier. ao_factor = AO_MIN + ao_level * AO_STEP\n(levels 0..=3 → 0.4..=1.0). Read by mesh3d.vert.",
            val: Val::Scalar(0.4),
        },
        Def {
            name: "AO_STEP",
            doc: "Baked AO per-level step. See AO_MIN.",
            val: Val::Scalar(0.2),
        },
        // Sun disc core/rim radii, tuned to match the sun's real angular size.
        Def {
            name: "SUN_DISC_CORE",
            doc: "Sun-disc solid core as a multiple of the sun angular radius.\nLarger core keeps sun defined against brighter night sky. Complements SUN_DISC_RIM tightening.",
            val: Val::Scalar(1.12),
        },
        // Far-field shadow fallback steepness: only near-full sky access keeps
        // direct light beyond the cascades.
        Def {
            name: "SHADOW_FALLBACK_POW",
            doc: "Exponent on visible_sky in the beyond-cascades shadow fallback.",
            val: Val::Scalar(10.0),
        },
        Def {
            name: "SHADOW_SKY_AMBIENT",
            doc: "Floor on the skylight's lit factor under sun shadow: the sky DOME still\nlights a sun-shadowed surface (blue-sky bounce), so shadow can attenuate\nskylight only down to this fraction — never to the black pit that erased\nall material detail in shadowed cliffs. Scales with sky_amount, so caves\n(sky_amount 0) stay dark; only outdoor shadow gains the floor.",
            val: Val::Scalar(0.22),
        },
        Def {
            name: "CASCADE_BLEND_FRAC",
            doc: "Fraction of the near split over which near/far PCF cross-fade; 0 = hard\nswitch. New look 0.15; VOXEL_LIGHT_LEGACY=1 keeps the hard 64 m cut.",
            val: Val::Scalar(if light_legacy { 0.0 } else { 0.15 }),
        },
        Def {
            name: "SHADOW_BOUNCE_TINT",
            doc: "How far the sun-shadow fill tints from sun colour toward luma-matched\nzenith colour. 0 = fill is SHADOW_SKY_AMBIENT × sun colour (legacy);\n0.75 pulls the fill toward sky-blue so shadowed ground is sky-lit, not warm.",
            val: Val::Scalar(if light_legacy { 0.0 } else { 0.75 }),
        },
        Def {
            name: "AO_DIRECT",
            doc: "Fraction of baked AO applied to the direct sun term. Dome/ambient/blocklight\nkeep full AO. 1.0 = AO darkens direct sun as much as ambient (legacy,\ndouble-darkens creases the cascade already shades); 0.5 is the new look.",
            val: Val::Scalar(if light_legacy { 1.0 } else { 0.5 }),
        },
        Def {
            name: "SHADOW_FAR_MIN_RADIUS",
            doc: "Lower clamp of the far cascade split when fitted to DrawLists::lod_clip.\n96 >= 64·1.15 + 16 m fade band, so the blend band and SHADOW_LIMIT fade\nnever overlap the near split. Legacy 256 keeps the fixed far radius.",
            val: Val::Scalar(if light_legacy { 256.0 } else { 96.0 }),
        },
        Def {
            name: "SHADOW_RESOLUTION",
            doc: "Cascade shadow map edge in texels. Mirror of vk/targets.rs (asserted equal\nthere); lets the PCF derive texel size without a per-fragment GetDimensions.",
            val: Val::Scalar(2048.0),
        },
        Def {
            name: "CULL_WORKGROUP",
            doc: "GPU cull dispatch workgroup width (one thread per mesh slot). The CPU-side\ndispatch partition math (vk/cull.rs) must divide by the same value.",
            val: Val::UInt(64),
        },
        Def {
            name: "CULL_DISTANCE_BUCKETS",
            doc: "Front-to-back approximate-order buckets per camera (pass, arena) partition.\nShadow groups stay unbucketed. CPU partition math and the cull shader must agree.",
            val: Val::UInt(4),
        },
        Def {
            name: "CULL_CAMERA_GROUPS",
            doc: "Camera cull groups: full-res Opaque, Cutout, coarse-LOD Opaque.\nMust match vk::cull::CAMERA_GROUPS; the shader uses this for partition indexing.",
            val: Val::UInt(3),
        },
        Def {
            name: "CULL_BUCKET_SPLIT_0",
            doc: "Camera-distance edge (metres from AABB centre) between cull buckets 0 and 1.\nBucket 0 is nearest; record_mesh_indirect_count draws 0..K-1 near-to-far.",
            val: Val::Scalar(16.0),
        },
        Def {
            name: "CULL_BUCKET_SPLIT_1",
            doc: "Camera-distance edge (metres) between cull buckets 1 and 2.",
            val: Val::Scalar(64.0),
        },
        Def {
            name: "CULL_BUCKET_SPLIT_2",
            doc: "Camera-distance edge (metres) between cull buckets 2 and 3 (farthest).",
            val: Val::Scalar(256.0),
        },
        Def {
            name: "EXPOSURE_TILE",
            doc: "Exposure metering tile edge in HDR texels. The CPU-side tile-grid dimensions\n(vk/exposure.rs) are ceil(hdr_dim / EXPOSURE_TILE) and must agree.",
            val: Val::UInt(16),
        },
        Def {
            name: "TAA_TILE",
            doc: "Retired TAA compute workgroup edge; the resolve now runs in the present-time\ntonemap fragment (no groupshared tile). Kept so generated constants stay stable.",
            val: Val::UInt(16),
        },
        Def {
            name: "SUN_DISC_RIM",
            doc: "Sun-disc soft rim (fade-to-zero edge) as a multiple of the sun angular radius.\nTighter rim keeps sun crisp against busier night sky. Complements brighter stars [4.2, 4.5, 5.2] and larger moon.",
            val: Val::Scalar(1.5),
        },
        // Sunset-glow widening: the sky sun-halo exponent lerps between these two
        // as sun elevation crosses [GLOW_EDGE0, GLOW_EDGE1], so low sun gets a
        // wide golden-hour halo while high sun keeps the fixed-8.0 noon look.
        // CPU twin (documentation/parity): the `GLOW` Curve in src/sky/palette.rs.
        Def {
            name: "GLOW_POW_SUNSET",
            doc: "Sky sun-halo exponent at low sun (small ⇒ wide golden-hour halo). Blended to\nGLOW_POW_DAY over sun elevation (sun_dir_elev.w, [-1,1]).",
            val: Val::Scalar(3.5),
        },
        Def {
            name: "GLOW_POW_DAY",
            doc: "Sky sun-halo exponent at high sun: preserves the tight noon halo (the former fixed 8.0).",
            val: Val::Scalar(8.0),
        },
        Def {
            name: "GLOW_EDGE0",
            doc: "Glow-exponent smoothstep low edge on sun elevation. Twin: palette::GLOW.edge0.",
            val: Val::Scalar(0.0),
        },
        Def {
            name: "GLOW_EDGE1",
            doc: "Glow-exponent smoothstep high edge; at/above it the noon halo is reproduced exactly.\nTwin: palette::GLOW.edge1.",
            val: Val::Scalar(0.4),
        },
        // Wrap period for the `anim` lane: animated shaders read a world-time
        // seconds phase and a camera-space UV, both folded into [0, PERIOD) on the
        // CPU so f32 keeps precision arbitrarily far from the origin / far into a
        // day. Round value in BOTH seconds (time) and metres (space).
        Def {
            name: "ANIM_PERIOD",
            doc: "Wrap period for the `anim` FrameUniforms lane, in seconds (anim_time)\nand metres (camera UV). Folds world time/space into [0, ANIM_PERIOD)\nCPU-side so f32 never loses phase precision far from the origin.",
            val: Val::Scalar(512.0),
        },
        // Autoexposure make-up curve constants (fitted above). CPU-only today —
        // exposure metering runs render-side in Rust (vk/exposure.rs) — but the
        // generated table is the ONE home for tuned/fitted constants, so they live
        // here even though no shader reads them yet.
        Def {
            name: "EXPOSURE_L_DAY_LOG2",
            doc: "Autoexposure fixed point: exposure=1.0 at day luma (golden image must match).\nRead by vk/exposure.rs.",
            val: Val::Scalar(exposure_l_day_log2),
        },
        Def {
            name: "EXPOSURE_EV_SLOPE",
            doc: "Autoexposure EV-space slope (fitted so exposure(cave)=6.05). Gives gentle\ndarken (no cliff like old linear fit). See EXPOSURE_L_DAY_LOG2.",
            val: Val::Scalar(exposure_ev_slope),
        },
        Def {
            name: "EXPOSURE_CLAMP_LO",
            doc: "Floor on exposure multiplier (EV-space curve is always positive). Bounds\npathological over-metering; no longer a cliff band-aid.",
            val: Val::Scalar(0.25),
        },
        Def {
            name: "EXPOSURE_CLAMP_HI",
            doc: "Ceiling on the exposure multiplier so a near-black frame can't blow up unbounded.",
            val: Val::Scalar(8.0),
        },
        // Bloom: threshold → downsample → quarter-res spill (golden-spiral + godrays).
        Def {
            name: "BLOOM_THRESHOLD_LO",
            doc: "Bloom soft-knee low edge on exposed luma (luma·exposure). Below this the\npixel contributes no spill. Read by vk/bloom.rs → bloom.comp.",
            val: Val::Scalar(0.85),
        },
        Def {
            name: "BLOOM_THRESHOLD_HI",
            doc: "Bloom soft-knee high edge: at/above this luma the spill weight is full.",
            val: Val::Scalar(1.0),
        },
        Def {
            name: "BLOOM_THRESHOLD_SCALE",
            doc: "Bloom spill scale applied to the thresholded colour, keeping\nthe stored spill well below the source so BLOOM_STRENGTH stays the only knob.",
            val: Val::Scalar(0.5),
        },
        Def {
            name: "BLOOM_STRENGTH",
            doc: "Bloom composite weight before tonemap sigmoid. Subtle spill off bright sources.",
            val: Val::Scalar(0.04),
        },
        Def {
            name: "BLOOM_SPIRAL_SAMPLES",
            doc: "Golden-angle spiral tap count for the bloom upsample-composite. Taps\nsample the mip chain at BLOOM_SPIRAL_LOD.",
            val: Val::Scalar(6.0),
        },
        Def {
            name: "BLOOM_SPIRAL_LOD",
            doc: "Mip level the bloom spiral samples the (already downsampled) chain at — the\nwide soft blur comes from the pyramid; the spiral just spreads and de-aliases it.\nQuarter-res chain, so LOD 1 is eighth-res.",
            val: Val::Scalar(1.0),
        },
        Def {
            name: "BLOOM_MAX_MIPS",
            doc: "Bloom pyramid mip cap. The spill pass samples only BLOOM_SPIRAL_LOD, so the\nchain stops at that level (base + LOD). CPU (vk/targets.rs) must agree.",
            val: Val::UInt(2),
        },
        Def {
            name: "BLOOM_SPIRAL_RADIUS",
            doc: "Bloom spiral radius in output uv (isotropic). Small — the mip chain already\ncarries the wide blur, so this only softens the seams between taps.",
            val: Val::Scalar(0.08),
        },
        Def {
            name: "SPILL_FACTOR",
            doc: "Spill image is 1/SPILL_FACTOR of the render extent on each axis (quarter-res\nat 4). CPU image create (vk/targets.rs) and the spill dispatch must agree.",
            val: Val::UInt(4),
        },
        Def {
            name: "SPILL_WG",
            doc: "Spill compute workgroup edge. CPU dispatch divides the spill extent by this;\nthe shader's [numthreads] uses the same value.",
            val: Val::UInt(8),
        },
        // Screen-space godrays: dithered march toward sun, composite veil in the spill pass.
        Def {
            name: "GODRAY_SAMPLES",
            doc: "March taps from each pixel toward the sun's screen position. Low (4); a\nfixed half-step start-offset centres the first sample.\nCast to int in the shader.",
            val: Val::Scalar(4.0),
        },
        Def {
            name: "GODRAY_GLOW_EXP",
            doc: "Sunward falloff exponent. Higher tightens rays around sun.",
            val: Val::Scalar(6.0),
        },
        Def {
            name: "GODRAY_STRENGTH",
            doc: "Godray veil composite weight in the quarter-res spill (before tonemap sigmoid).",
            val: Val::Scalar(0.6),
        },
        Def {
            name: "GODRAY_MAX_LEN",
            doc: "Max march length AND sunward falloff radius, in output uv (does double duty:\nthe march covers up to this distance toward the sun, and the glow fades to zero\nbeyond it). Isotropic uv space (elliptical on non-square frames — imperceptible\non a soft veil).",
            val: Val::Scalar(0.6),
        },
        // Moon + stars. The moon reuses sun_disc at the anti-solar point,
        // gated by night (1 - day_night_mix); stars are a procedural hash grid
        // derived purely from the world-space ray (static, deterministic — no
        // wall-clock, so goldens stay bit-comparable).
        Def {
            name: "MOON_TINT",
            doc: "Moon disc linear radiance (additive, HDR). Cool desaturated — no moon-phase\nstate exists, so a single fixed tint. Dimmer than the sun so it reads as a\nsoft night disc, not a second sun.",
            val: Val::Arr(vec![0.45, 0.5, 0.62]),
        },
        Def {
            name: "MOON_RADIUS_SCALE",
            doc: "Moon angular radius as a multiple of the sun's (pc.sun.w): a touch smaller.\nLarger moon for visibility against voxel terrain; complements brighter stars. Range [0.75–1.0].",
            val: Val::Scalar(0.92),
        },
        Def {
            name: "STAR_COLOR",
            doc: "Star point linear radiance (additive, HDR); brightness folded in. Cool white.\nBrighter stars for presence against voxel terrain; still cool-toned. Range [3.0–5.5] per channel.",
            val: Val::Arr(vec![4.2, 4.5, 5.2]),
        },
        Def {
            name: "STAR_DENSITY",
            doc: "Star hash-grid resolution: cells across the hemispheric ray projection.\nHigher ⇒ more, smaller cells (denser field). Slightly more presence without sparse\nfeel; correlates with STAR_THRESHOLD for ~5.5% population. Range [250–400].",
            val: Val::Scalar(328.0),
        },
        Def {
            name: "STAR_THRESHOLD",
            doc: "Per-cell hash cutoff for a star to exist; fraction of lit cells is\n1 - STAR_THRESHOLD (sparse). Maintains sparse field (~5.5%) while correlating\nwith STAR_DENSITY bump.",
            val: Val::Scalar(0.953),
        },
        Def {
            name: "STAR_SHARP",
            doc: "Star point sharpness. Higher gives smaller, crisper points.",
            val: Val::Scalar(72.0),
        },
        Def {
            name: "STAR_MOON_FADE",
            doc: "cos(angle) toward the moon above which stars fade out, so the field never\noverlaps the moon disc. Near 1 ⇒ only a tight halo is cleared.",
            val: Val::Scalar(0.9995),
        },
        Def {
            name: "STAR_HORIZON_FADE",
            doc: "Star fade toward horizon (hides stars in fog band).",
            val: Val::Scalar(8.0),
        },
        // Slab clouds: 2-plane volumetric slab in sky.frag, world-anchored so pinned day is deterministic.
        Def {
            name: "CLOUD_BOTTOM",
            doc: "World-y (metres) of the lower slab plane. Clouds render only when the\ncamera (anim.w) is below the slab and the view ray rises into it.",
            val: Val::Scalar(180.0),
        },
        Def {
            name: "CLOUD_THICKNESS",
            doc: "Vertical slab thickness (metres); the upper plane is CLOUD_BOTTOM + this.",
            val: Val::Scalar(60.0),
        },
        Def {
            name: "CLOUD_STEPS",
            doc: "Raymarch steps between the two slab-plane intersections (N=10).\nCast to int in the shader.",
            val: Val::Scalar(10.0),
        },
        Def {
            name: "CLOUD_NOISE_SCALE",
            doc: "Cloud noise spatial frequency (1/metres). World-anchored so pinned day is deterministic.",
            val: Val::Scalar(0.0009),
        },
        Def {
            name: "CLOUD_OCTAVE2_SCALE",
            doc: "Second value-noise octave frequency, as a multiple of CLOUD_NOISE_SCALE.\nSampled on swapped zx so the two octaves don't align into stripes.",
            val: Val::Scalar(2.0),
        },
        Def {
            name: "CLOUD_WIND",
            doc: "Cloud drift in noise units per second: coord += CLOUD_WIND * anim_time.\nDeterministic at a pinned day (anim_time is derived from the pinned clock).",
            val: Val::Arr(vec![0.02, 0.007]),
        },
        Def {
            name: "CLOUD_COVERAGE",
            doc: "Density threshold. Higher makes sky sparser.",
            val: Val::Scalar(0.58),
        },
        Def {
            name: "CLOUD_OPACITY",
            doc: "Per-step opacity gain applied to remapped density in the front-to-back\naccumulation. Tuned so a dense column reads opaque within CLOUD_STEPS steps.",
            val: Val::Scalar(0.55),
        },
        Def {
            name: "CLOUD_LIGHT",
            doc: "Weight of the direct-light lane (frame.light.rgb) in the lit cloud colour.",
            val: Val::Scalar(1.0),
        },
        Def {
            name: "CLOUD_AMBIENT",
            doc: "Weight of the sky/zenith lane (frame.zenith.rgb) in the lit cloud colour.",
            val: Val::Scalar(0.6),
        },
        Def {
            name: "CLOUD_DARK",
            doc: "Interior darkening: cloud colour × lerp(1, CLOUD_DARK, sqrt(density)), so\ndense cores read darker than thin edges (mix(bright,dark,sqrt(cv))).\nLighter cores prevent muddy interior appearance.",
            val: Val::Scalar(0.35),
        },
        Def {
            name: "CLOUD_SILVER",
            doc: "Silver-lining gain: mix(col, col*CLOUD_SILVER, (1-cv^0.25)*bright^2) where\nbright = pow(dot(ray,sun),3) — bright sunward rim on thin cloud edges.",
            val: Val::Scalar(13.0),
        },
        Def {
            name: "CLOUD_HORIZON_OFFSET",
            doc: "Horizon hide: ray.y below this fades clouds out (they don't reach the\nhorizon band where the sky reads as fog). clamp((y-0.06)*5).",
            val: Val::Scalar(0.06),
        },
        Def {
            name: "CLOUD_HORIZON_FADE",
            doc: "Horizon-hide steepness: saturate((ray.y - CLOUD_HORIZON_OFFSET) * this).",
            val: Val::Scalar(5.0),
        },
        // Direction-space cloud LUT: one compute dispatch per sky frame marches
        // the slab into an octahedral upper-hemisphere map; the full-res sky
        // fragment takes one bilinear tap. 256 is enough — cloud noise is
        // ~1 km scale — and stays at the cheap end of the 256..512 quality band.
        Def {
            name: "SKY_CLOUD_LUT_SIZE",
            doc: "Edge length of the square octahedral cloud LUT (texels). Quality band\n[256, 512]; CPU dispatch and the compute shader must agree.",
            val: Val::UInt(256),
        },
        Def {
            name: "SKY_CLOUD_LUT_WG",
            doc: "Cloud-LUT compute workgroup edge. CPU dispatch divides SKY_CLOUD_LUT_SIZE\nby this; the shader's [numthreads] uses the same value.",
            val: Val::UInt(8),
        },
        // Water: animated waves + reflection + glint + interim tint, world-anchored for deterministic goldens.
        Def {
            name: "WATER_WAVE_FREQ",
            doc: "Wave value-noise spatial frequency (1/metres) at the base octave.\nHigher ⇒ smaller, tighter ripples.",
            val: Val::Scalar(0.35),
        },
        Def {
            name: "WATER_WAVE_AMP",
            doc: "Wave normal perturbation strength: how far the height gradient tilts the\nflat water normal. Higher ⇒ choppier reflections.",
            val: Val::Scalar(0.18),
        },
        Def {
            name: "WATER_WAVE_SPEED",
            doc: "Wave scroll speed (cycles/second). Applied to anim_time for deterministic goldens.",
            val: Val::Scalar(1.4),
        },
        Def {
            name: "WATER_FRESNEL_F0",
            doc: "Water reflectance at normal incidence (grazing lifts reflection to 1).",
            val: Val::Scalar(0.02),
        },
        Def {
            name: "WATER_GLINT",
            doc: "Sun-glint intensity: multiplies the reused sun_disc sampled along the\nreflection ray, an HDR specular highlight on the waves.",
            val: Val::Scalar(2.0),
        },
        Def {
            name: "WATER_GLINT_RADIUS",
            doc: "Angular radius (radians) of the water sun-glint disc — wider than the sky\nsun so wave-perturbed normals scatter it into a soft sparkle.",
            val: Val::Scalar(0.06),
        },
        Def {
            name: "WATER_DEEP",
            doc: "Deep-water linear tint the body colour is pulled toward (interim absorption\nstand-in until true depth-difference absorption lands).",
            val: Val::Arr(vec![0.02, 0.09, 0.13]),
        },
        Def {
            name: "WATER_BODY_MIX",
            doc: "How far the lit block colour is mixed toward WATER_DEEP for the water body\n(0 = keep block colour, 1 = full deep tint). Fallback mix used when previous-frame\ndepth is invalid, or MSAA is on (absorb pipeline not built); true depth-difference\nabsorption (WATER_ABS) supersedes it when the previous depth is sampled.",
            val: Val::Scalar(0.6),
        },
        // Water depth-difference absorption: driven by water column thickness from previous-frame depth.
        Def {
            name: "WATER_ABS",
            doc: "Water absorption coefficient (1/metres²). Larger saturates in shallower water.",
            val: Val::Scalar(0.22),
        },
        Def {
            name: "WATER_ABS_BASE",
            doc: "Absorption curve base. Keeps thin water sheets mostly transparent.",
            val: Val::Scalar(1.125),
        },
        // Unified source of truth for previously-duplicated shader constants.
        Def {
            name: "Z_NEAR",
            doc: "Reversed-Z near-plane distance. Single source for the projection\nmatrix (camera.rs, which re-exports this) and the water depth-absorption\npass's forward-distance reconstruction (mesh3d.frag.slang's `WATER_Z_NEAR`\nalias) — previously two independently hand-written 0.05s.",
            val: Val::Scalar(0.05),
        },
        Def {
            name: "CURVE_INV_2R",
            doc: "Gameplay-tuned planet curvature: inverse of 2x the visual planet\nradius (~300 km). Read by mesh3d.vert's horizon droop (CURVE_MAX_DROP-clamped,\npresentation-only — does not affect gameplay).",
            val: Val::Scalar(1.0 / (2.0 * 300_000.0)),
        },
        Def {
            name: "CURVE_MAX_DROP",
            doc: "Planet-curvature horizon droop clamp, in metres (reached ~50 km out).\nBounds mesh3d.vert's CURVE_INV_2R droop so distant vertices can't overflow\nor fold the horizon.",
            val: Val::Scalar(4096.0),
        },
        Def {
            name: "LUMA_FLOOR",
            doc: "Exposure metering luma floor: keeps log2(luma) finite on black tiles.\nRead by exposure_reduce.comp.",
            val: Val::Scalar(1e-4),
        },
        // Detail level bias for GPU encoding (shared with shaders).
        Def {
            name: "DETAIL_GPU_BIAS",
            doc: "Bias added to the signed detail level k before it is stored in the\n4-bit detail field of MeshRecord.detail_pass. Decode: 2^k = exp2(bits - bias).",
            val: Val::UInt(2),
        },
        Def {
            name: "GLOW_SCALE",
            doc: "HDR emissive scale for MaterialDesc.glow: after lighting, add\n(glow/255) * GLOW_SCALE * albedo. Default 4.0. Read by mesh3d.frag via material.slang.",
            val: Val::Scalar(4.0),
        },
    ]
}

// Per-frame uniform block: single source of truth for the std140 lane layout
// shared by `voxel_engine::skeleton::FrameUniformsGpu` (Rust wire form) and
// `common.slang`'s `FrameUniforms`. One table drives BOTH so a lane can never
// drift between CPU and GPU; the generated Rust also emits offset static asserts
// so any reorder is a compile error.
//   * $OUT_DIR/gen_frame_uniforms.rs        — included by skeleton.rs
//   * shaders/generated/frame_uniforms.slang — #included by common.slang
// Every lane is one float4; semantics are smuggled into `.w`/spare channels.

struct Lane {
    name: &'static str,
    /// Per-channel meaning; emitted as a doc /// (Rust) and // (Slang) comment.
    doc: &'static str,
}

fn lane_table() -> Vec<Lane> {
    vec![
        Lane {
            name: "sun_dir_elev",
            doc: "xyz = sun direction (unit, engine-normalized in prepare_derived),\nw = sun elevation (game: sun_dir.y in [-1,1]; drives the glow_pow lerp).",
        },
        Lane {
            name: "light",
            doc: "rgb = direct light color (linear), w = day_night_mix.",
        },
        Lane {
            name: "zenith",
            doc: "rgb = zenith color (linear), w = turbidity.",
        },
        Lane {
            name: "horizon",
            doc: "rgb = horizon color (linear), w = fog density.",
        },
        Lane {
            name: "candle",
            doc: "rgb = blocklight (candle) color (linear), w = ambient floor luma.",
        },
        Lane {
            name: "exposure_dither",
            doc: "x = exposure, y = reserved zero (post-effect dither removed), zw = TAA jitter\nin pixels (informational; jitter is applied via the matrix — zero until enabled).",
        },
        Lane {
            name: "extras",
            doc: "x = stars gain (1 = night starfield renders, 0 = skipped — the\n`RenderFlags::stars` gate). y = glow_pow, z = glow_scale (0.5+turbidity):\nengine-derived sky_radiance terms, filled by FrameUniformsGpu::prepare_derived.\nw reserved (debug-flat may overwrite the whole lane).",
        },
        Lane {
            name: "anim",
            doc: "x = anim_time = world-time seconds mod ANIM_PERIOD; yz = fract(camera_world.xz\n/ ANIM_PERIOD); w = camera world-y (metres, bounded ⇒ no wrap) for the cloud\nslab in sky.frag.",
        },
    ]
}

/// Engine-derived tail appended after [`lane_table`] in the GPU UBO / Slang
/// `FrameUniforms`. Not part of the public `FrameUniformsGpu` wire the game
/// writes — the renderer fills these from the public lanes once per frame.
fn derived_lane_table() -> Vec<Lane> {
    vec![
        Lane {
            name: "ambient_glow",
            doc: "Engine-derived (not in FrameUniformsGpu). rgb = ambient_floor_color\n(zenith rescaled to candle.w luma; grayscale candle.w when zenith luma is 0);\nw = sky halo exponent (GLOW_POW_SUNSET..DAY smoothstep on sun_dir_elev.w).",
        },
        Lane {
            name: "glow_day",
            doc: "Engine-derived. rgb = light.rgb * (0.5 + turbidity) (sky-halo tint×scale);\nw = abs(2*day_night_mix - 1) (shadow_fallback day factor).",
        },
        Lane {
            name: "shadow_bounce",
            doc: "Engine-derived. rgb = SHADOW_SKY_AMBIENT * lerp(light.rgb, zenith.rgb *\nluma709(light)/luma709(zenith), SHADOW_BOUNCE_TINT); light.rgb when zenith\nluma is 0. w = asfloat lane-enable bits (shadows=1, blocklight=2, ambient=4).",
        },
    ]
}

fn emit_generated_spdx(s: &mut String) {
    // REUSE-IgnoreStart
    s.push_str("// SPDX-FileCopyrightText: 2026 voxel-engine contributors\n");
    s.push_str("// SPDX-License-Identifier: MIT OR Apache-2.0\n\n");
    // REUSE-IgnoreEnd
}

fn emit_rust_uniforms(public: &[Lane], derived: &[Lane]) -> String {
    let mut s = String::new();
    emit_generated_spdx(&mut s);
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: lane_table() / derived_lane_table() in build.rs.\n");
    s.push_str("// Twin file: shaders/generated/frame_uniforms.slang (public then derived).\n\n");
    s.push_str("/// WIRE form of the game's `FrameSnapshot`. Every lane is a `[f32; 4]` so\n");
    s.push_str("/// std140 and scalar layouts cannot diverge. Colors are LINEAR f32\n");
    s.push_str("/// (unclamped — HDR palettes survive). The layout below IS the contract with\n");
    s.push_str(
        "/// `common.slang`'s `FrameUniforms` prefix; the offset asserts below are the enforcement.\n",
    );
    s.push_str("///\n");
    s.push_str("/// Constructed ONLY via the game crate's `From<&FrameSnapshot>` impl.\n");
    s.push_str("/// Engine-derived extra lanes live on [`FrameUniformsExt`], not here.\n");
    s.push_str("#[repr(C)]\n");
    s.push_str("#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]\n");
    s.push_str("pub struct FrameUniformsGpu {\n");
    for l in public {
        doc_lines(l.doc, "    /// ", &mut s);
        s.push_str(&format!("    pub {}: [f32; 4],\n", l.name));
    }
    s.push_str("}\n\n");
    s.push_str(&format!(
        "const _: () = assert!(size_of::<FrameUniformsGpu>() == {});\n",
        public.len() * 16
    ));
    for (i, l) in public.iter().enumerate() {
        s.push_str(&format!(
            "const _: () = assert!(std::mem::offset_of!(FrameUniformsGpu, {}) == {});\n",
            l.name,
            i * 16
        ));
    }
    s.push('\n');
    s.push_str("/// GPU UBO: public [`FrameUniformsGpu`] plus the engine-derived tail.\n");
    s.push_str("/// The game still writes `FrameUniformsGpu`; the renderer appends the rest.\n");
    s.push_str("#[repr(C)]\n");
    s.push_str("#[derive(Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]\n");
    s.push_str("pub(crate) struct FrameUniformsExt {\n");
    s.push_str("    /// Public per-frame lanes the game wrote.\n");
    s.push_str("    pub base: FrameUniformsGpu,\n");
    for l in derived {
        doc_lines(l.doc, "    /// ", &mut s);
        s.push_str(&format!("    pub {}: [f32; 4],\n", l.name));
    }
    s.push_str("}\n\n");
    s.push_str(&format!(
        "const _: () = assert!(size_of::<FrameUniformsExt>() == {});\n",
        (public.len() + derived.len()) * 16
    ));
    s.push_str("const _: () = assert!(std::mem::offset_of!(FrameUniformsExt, base) == 0);\n");
    for (i, l) in derived.iter().enumerate() {
        s.push_str(&format!(
            "const _: () = assert!(std::mem::offset_of!(FrameUniformsExt, {}) == {});\n",
            l.name,
            (public.len() + i) * 16
        ));
    }
    s
}

fn emit_slang_uniforms(public: &[Lane], derived: &[Lane]) -> String {
    let mut s = String::new();
    emit_generated_spdx(&mut s);
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: lane_table() / derived_lane_table() in build.rs.\n");
    s.push_str("// Twin file: voxel_engine::skeleton::FrameUniformsGpu (public prefix) plus\n");
    s.push_str("// the engine-derived tail on FrameUniformsExt (same extra lanes/order).\n");
    s.push_str(
        "// Consumed via: #include \"generated/frame_uniforms.slang\" (from common.slang).\n\n",
    );
    s.push_str("// Per-frame uniform block. Public lanes match FrameUniformsGpu; the tail is\n");
    s.push_str("// filled by the renderer (vk/uniforms.rs) and is not part of the game wire.\n");
    s.push_str("struct FrameUniforms\n{\n");
    for l in public {
        doc_lines(l.doc, "    // ", &mut s);
        s.push_str(&format!("    float4 {};\n", l.name));
    }
    if !derived.is_empty() {
        s.push_str("    // --- engine-derived tail (not in FrameUniformsGpu) ---\n");
        for l in derived {
            doc_lines(l.doc, "    // ", &mut s);
            s.push_str(&format!("    float4 {};\n", l.name));
        }
    }
    s.push_str("};\n");
    s
}

fn doc_lines(doc: &str, prefix: &str, out: &mut String) {
    for line in doc.lines() {
        out.push_str(prefix);
        out.push_str(line);
        out.push('\n');
    }
}

fn emit_rust(defs: &[Def]) -> String {
    let mut s = String::new();
    emit_generated_spdx(&mut s);
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: the build_table() function in build.rs.\n");
    s.push_str("// Twin file: shaders/generated/shader_constants.slang (same names).\n\n");
    for d in defs {
        doc_lines(d.doc, "/// ", &mut s);
        match &d.val {
            Val::Scalar(x) => {
                s.push_str(&format!("pub const {}: f32 = {};\n\n", d.name, lit(*x)));
            }
            Val::UInt(x) | Val::Int(x) => {
                s.push_str(&format!("pub const {}: u32 = {};\n\n", d.name, x));
            }
            Val::Arr(v) => {
                s.push_str(&format!("pub const {}: [f32; {}] = [\n", d.name, v.len()));
                for chunk in v.chunks(8) {
                    s.push_str("    ");
                    for x in chunk {
                        s.push_str(&lit(*x));
                        s.push_str(", ");
                    }
                    s.push('\n');
                }
                s.push_str("];\n\n");
            }
            Val::Arr2(v) => {
                s.push_str(&format!(
                    "pub const {}: [[f32; 2]; {}] = [\n",
                    d.name,
                    v.len()
                ));
                for p in v {
                    s.push_str(&format!("    [{}, {}],\n", lit(p[0]), lit(p[1])));
                }
                s.push_str("];\n\n");
            }
        }
    }
    s
}

fn emit_slang(defs: &[Def]) -> String {
    let mut s = String::new();
    emit_generated_spdx(&mut s);
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: the build_table() function in build.rs.\n");
    s.push_str("// Twin file: voxel_engine::genconst (Rust, same names).\n");
    s.push_str(
        "// Consumed via: #include \"generated/shader_constants.slang\" (from common.slang).\n\n",
    );
    for d in defs {
        doc_lines(d.doc, "// ", &mut s);
        match &d.val {
            Val::Scalar(x) => {
                s.push_str(&format!(
                    "static const float {} = {}f;\n\n",
                    d.name,
                    lit(*x)
                ));
            }
            Val::UInt(x) => {
                s.push_str(&format!("static const uint {} = {};\n\n", d.name, x));
            }
            Val::Int(x) => {
                s.push_str(&format!("static const int {} = {};\n\n", d.name, x));
            }
            Val::Arr(v) => {
                s.push_str(&format!(
                    "static const float {}[{}] = {{\n",
                    d.name,
                    v.len()
                ));
                for chunk in v.chunks(8) {
                    s.push_str("    ");
                    for x in chunk {
                        s.push_str(&lit(*x));
                        s.push_str("f, ");
                    }
                    s.push('\n');
                }
                s.push_str("};\n\n");
            }
            Val::Arr2(v) => {
                s.push_str(&format!(
                    "static const float2 {}[{}] = {{\n",
                    d.name,
                    v.len()
                ));
                for p in v {
                    s.push_str(&format!("    float2({}f, {}f),\n", lit(p[0]), lit(p[1])));
                }
                s.push_str("};\n\n");
            }
        }
    }
    s
}

/// Write only when content differs, so `rerun-if-changed=shaders` (which sees
/// the generated .slang) cannot spin into a rebuild loop and stale outputs
/// never survive a table edit.
fn write_if_changed(path: &Path, content: &str) {
    let differs = fs::read_to_string(path)
        .map(|old| old != content)
        .unwrap_or(true);
    if differs {
        fs::write(path, content).unwrap();
    }
}

fn generate_shared_constants(out_dir: &Path) {
    let defs = build_table();
    write_if_changed(&out_dir.join("gen_constants.rs"), &emit_rust(&defs));

    let gen_dir = Path::new("shaders/generated");
    fs::create_dir_all(gen_dir).unwrap();
    write_if_changed(&gen_dir.join("shader_constants.slang"), &emit_slang(&defs));

    let public = lane_table();
    let derived = derived_lane_table();
    write_if_changed(
        &out_dir.join("gen_frame_uniforms.rs"),
        &emit_rust_uniforms(&public, &derived),
    );
    write_if_changed(
        &gen_dir.join("frame_uniforms.slang"),
        &emit_slang_uniforms(&public, &derived),
    );
}
