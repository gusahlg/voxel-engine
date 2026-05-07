use std::{fs, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = Path::new("shaders_spv");
    fs::create_dir_all(out_dir).unwrap();

    for (src, stage, dst_name) in [
        ("shaders/tri.vert.slang", "vertex", "tri.vert.spv"),
        ("shaders/tri.frag.slang", "fragment", "tri.frag.spv"),
    ] {
        let dst = out_dir.join(dst_name);

        let status = Command::new("slangc")
            .args([
                src,
                "-target",
                "spirv",
                "-profile",
                "spirv_1_3",
                "-entry",
                "main",
                "-stage",
                stage,
                "-o",
            ])
            .arg(&dst)
            .status()
            .expect("failed to run slangc (is Slang installed?)");

        assert!(status.success(), "slangc failed on {}", src);
    }

    println!("cargo:rerun-if-env-changed=PATH");
}
