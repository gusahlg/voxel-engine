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

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // Checked-in fallback so the crate builds without a Slang toolchain
    // (e.g. inside `nix build` sandboxes). Refreshed whenever slangc is around.
    let fallback_dir = Path::new("shaders_spv");
    fs::create_dir_all(fallback_dir).unwrap();

    let slangc = have_slangc();

    for shader in SHADERS {
        let out_path = out_dir.join(shader.dst);
        let fallback_path = fallback_dir.join(shader.dst);

        if !slangc {
            assert!(
                fallback_path.exists(),
                "slangc not found and no prebuilt {} — install Slang or restore shaders_spv/",
                fallback_path.display()
            );
            fs::copy(&fallback_path, &out_path).unwrap();
            continue;
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
                "-o",
            ])
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
}
