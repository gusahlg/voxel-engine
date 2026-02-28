use std::{fs, path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=shaders");

    let out_dir = Path::new("shaders_spv");
    fs::create_dir_all(out_dir).unwrap();

    for (src, stage) in [("shaders/tri.vert", "vert"), ("shaders/tri.frag", "frag")] {
        let dst = out_dir.join(format!("tri.{}.spv", stage));

        let status = Command::new("glslc")
            .args([src, "-o"])
            .arg(&dst)
            .status()
            .expect("failed to run glslc (is shaderc installed?)");

        assert!(status.success(), "glslc failed on {}", src);
    }

    println!("cargo:rerun-if-env-changed=PATH");
}
