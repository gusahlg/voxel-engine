use std::{env, fs, path::Path, path::PathBuf, process::Command};

struct Shader<'a> {
    src: &'a str,
    stage: &'a str,
    entry: &'a str,
    dst: &'a str,
}

const SHADERS: &[Shader] = &[
    Shader {
        src: "shaders/mesh3d.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "mesh3d.vert.spv",
    },
    Shader {
        src: "shaders/mesh3d.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "mesh3d.frag.spv",
    },
    Shader {
        src: "shaders/surface3d.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "surface3d.vert.spv",
    },
    Shader {
        src: "shaders/surface3d.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "surface3d.frag.spv",
    },
    Shader {
        src: "shaders/debug.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "debug.vert.spv",
    },
    Shader {
        src: "shaders/debug.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "debug.frag.spv",
    },
    Shader {
        src: "shaders/sky.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "sky.vert.spv",
    },
    Shader {
        src: "shaders/sky.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "sky.frag.spv",
    },
    Shader {
        src: "shaders/tonemap.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "tonemap.vert.spv",
    },
    Shader {
        src: "shaders/tonemap.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "tonemap.frag.spv",
    },
    Shader {
        src: "shaders/tris2d.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "tris2d.vert.spv",
    },
    Shader {
        src: "shaders/tris2d.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "tris2d.frag.spv",
    },
    Shader {
        src: "shaders/tris2d_tex.frag.slang",
        stage: "fragment",
        entry: "fragmentMain",
        dst: "tris2d_tex.frag.spv",
    },
    Shader {
        src: "shaders/vrs.comp.slang",
        stage: "compute",
        entry: "computeMain",
        dst: "vrs.comp.spv",
    },
    Shader {
        src: "shaders/shadow_depth.vert.slang",
        stage: "vertex",
        entry: "vertexMain",
        dst: "shadow_depth.vert.spv",
    },
    Shader {
        src: "shaders/exposure_reduce.comp.slang",
        stage: "compute",
        entry: "computeMain",
        dst: "exposure_reduce.comp.spv",
    },
    Shader {
        src: "shaders/taa_resolve.comp.slang",
        stage: "compute",
        entry: "computeMain",
        dst: "taa_resolve.comp.spv",
    },
];

fn have_slangc() -> bool {
    Command::new("slangc")
        .arg("-v")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-changed=shaders_spv");
    // Emitting any rerun-if-changed replaces cargo's default "rerun if any
    // package file changed", so build.rs itself must be listed explicitly —
    // otherwise edits to the CONSTS table below would not regenerate outputs.
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Single-source the constants shared by Rust and Slang BEFORE compiling any
    // shader: common.slang `#include`s the generated .slang, so it must exist
    // first. See generate_shared_constants for the drift-kill rationale.
    generate_shared_constants(&out_dir);

    // Checked-in fallback so the crate builds without a Slang toolchain
    // (e.g. inside `nix build` sandboxes). Refreshed whenever slangc is around.
    let fallback_dir = Path::new("shaders_spv");
    fs::create_dir_all(fallback_dir).unwrap();

    let slangc = have_slangc();

    for shader in SHADERS {
        compile(slangc, &out_dir, fallback_dir, shader, &[]);
    }

    // display-space sigmoid in place of PBR Neutral + OETF. Selected at pipeline
    // creation by WATT_TONEMAP=makeup.
    compile(
        slangc,
        &out_dir,
        fallback_dir,
        &Shader {
            src: "shaders/tonemap.frag.slang",
            stage: "fragment",
            entry: "fragmentMain",
            dst: "tonemap_makeup.frag.spv",
        },
        &["-DMAKEUP_SIGMOID"],
    );
}

fn compile(
    slangc: bool,
    out_dir: &Path,
    fallback_dir: &Path,
    shader: &Shader,
    defines: &[&str],
) {
    let out_path = out_dir.join(shader.dst);
    let fallback_path = fallback_dir.join(shader.dst);

    if !slangc {
        assert!(
            fallback_path.exists(),
            "slangc not found and no prebuilt {} — install Slang or restore shaders_spv/",
            fallback_path.display()
        );
        fs::copy(&fallback_path, &out_path).unwrap();
        return;
    }

    let output = Command::new("slangc")
        .args([
            shader.src,
            "-target",
            "spirv",
            "-profile",
            "spirv_1_3",
            "-entry",
            shader.entry,
            "-stage",
            shader.stage,
            "-matrix-layout-column-major",
        ])
        .args(defines)
        .arg("-o")
        .arg(&out_path)
        .output()
        .expect("failed to run slangc");

    if !output.status.success() {
        eprintln!("slangc failed while compiling {}", shader.src);
        eprintln!(
            "--- stdout ---\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
        eprintln!(
            "--- stderr ---\n{}",
            String::from_utf8_lossy(&output.stderr)
        );
        panic!("shader compilation failed");
    }

    let _ = fs::copy(&out_path, &fallback_path);
}

// ---------------------------------------------------------------------------
// Shared constants: single source of truth for Rust and Slang.
//
// One table (`build_table`) is rendered into two files that MUST agree:
//   * $OUT_DIR/gen_constants.rs            — `pub const` items (voxel_engine::genconst)
//   * shaders/generated/shader_constants.slang — `static const` items
// Any constant that appears in both CPU-side sky/lighting math and its shader
// twin lives here so CPU↔GPU drift becomes impossible-to-forget instead of a
// silent fog seam. The .slang is `#include`d by common.slang.
// ---------------------------------------------------------------------------

enum Val {
    Scalar(f32),
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
        halton.push([
            radical_inverse(i, 2) - 0.5,
            radical_inverse(i, 3) - 0.5,
        ]);
    }

    // Bit-reverse the phase order so cycling through it over time doesn't
    // step linearly (which would show up as visible dither banding).
    let mut dither = Vec::with_capacity(16);
    for k in 0u32..16 {
        let mut rev = 0u32;
        for b in 0..4 {
            rev |= ((k >> b) & 1) << (3 - b);
        }
        dither.push(rev as f32 / 16.0);
    }

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
            name: "DITHER_PHASE_16",
            doc: "Ordered dither phases: 4-bit bit-reversal permutation of k/16.",
            val: Val::Arr(dither),
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
            name: "POST_TONEMAP_DITHER_GAIN",
            doc: "Post-tonemap dither amplitude in output space (1 LSB of 8-bit = 1/255).\nAmplitude must stay exposure-invariant.",
            val: Val::Scalar(1.0 / 255.0),
        },
        // Sun disc core/rim radii, tuned to match the sun's real angular size.
        Def {
            name: "SUN_DISC_CORE",
            doc: "Sun-disc solid core as a multiple of the sun angular radius.",
            val: Val::Scalar(1.0),
        },
        // Far-field shadow fallback steepness: only near-full sky access keeps
        // direct light beyond the cascades.
        Def {
            name: "SHADOW_FALLBACK_POW",
            doc: "Exponent on visible_sky in the beyond-cascades shadow fallback.",
            val: Val::Scalar(10.0),
        },
        Def {
            name: "SUN_DISC_RIM",
            doc: "Sun-disc soft rim (fade-to-zero edge) as a multiple of the sun angular radius.",
            val: Val::Scalar(2.0),
        },
    ]
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
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: the build_table() function in build.rs.\n");
    s.push_str("// Twin file: shaders/generated/shader_constants.slang (same names).\n\n");
    for d in defs {
        doc_lines(d.doc, "/// ", &mut s);
        match &d.val {
            Val::Scalar(x) => {
                s.push_str(&format!("pub const {}: f32 = {};\n\n", d.name, lit(*x)));
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
                s.push_str(&format!("pub const {}: [[f32; 2]; {}] = [\n", d.name, v.len()));
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
    s.push_str("// @generated by voxel-engine/build.rs — DO NOT EDIT.\n");
    s.push_str("// Source of truth: the build_table() function in build.rs.\n");
    s.push_str("// Twin file: voxel_engine::genconst (Rust, same names).\n");
    s.push_str("// Consumed via: #include \"generated/shader_constants.slang\" (from common.slang).\n\n");
    for d in defs {
        doc_lines(d.doc, "// ", &mut s);
        match &d.val {
            Val::Scalar(x) => {
                s.push_str(&format!("static const float {} = {}f;\n\n", d.name, lit(*x)));
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
    let differs = fs::read_to_string(path).map(|old| old != content).unwrap_or(true);
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
}
