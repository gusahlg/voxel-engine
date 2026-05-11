use std::{fs, path::Path, process::Command};

struct Shader<'a> {
    src: &'a str,
    stage: &'a str,
    entry: &'a str,
    dst: &'a str,
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=PATH");

    let out_dir = Path::new("shaders_spv");
    fs::create_dir_all(out_dir).unwrap();

    let shaders = [
        Shader {
            src: "shaders/tri.vert.slang",
            stage: "vertex",
            entry: "vertexMain",
            dst: "tri.vert.spv",
        },
        Shader {
            src: "shaders/tri.frag.slang",
            stage: "fragment",
            entry: "fragmentMain",
            dst: "tri.frag.spv",
        },
    ];

    for shader in shaders {
        let dst_path = out_dir.join(shader.dst);

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
                "-o",
            ])
            .arg(&dst_path)
            .output()
            .expect("failed to run slangc; is Slang installed and in PATH?");

        if !output.status.success() {
            eprintln!("slangc failed while compiling {}", shader.src);
            eprintln!("stage: {}", shader.stage);
            eprintln!("entry: {}", shader.entry);
            eprintln!("output: {}", dst_path.display());
            eprintln!();
            eprintln!("--- stdout ---");
            eprintln!("{}", String::from_utf8_lossy(&output.stdout));
            eprintln!("--- stderr ---");
            eprintln!("{}", String::from_utf8_lossy(&output.stderr));

            panic!("shader compilation failed");
        }
    }
}
