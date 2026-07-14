//! All-module SPIR-V validation gate (E-08 follow-through): every tracked
//! fallback in `shaders_spv/` must pass `spirv-val` at the shipping Vulkan
//! target. This is what keeps an invalid module (like the old
//! `mesh3d_water.frag.spv` scalar `OpImageRead`) from sitting silently in the
//! tree behind a disabled pipeline. Skips — loudly — when `spirv-val` is not
//! on PATH, so plain `cargo test` outside the dev shell still passes; run
//! inside `nix develop` for the full gate.

use std::process::Command;

#[test]
fn all_tracked_spirv_modules_pass_spirv_val() {
    if Command::new("spirv-val").arg("--version").output().is_err() {
        eprintln!("spirv-val not on PATH: SKIPPING the all-module SPIR-V gate");
        return;
    }
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders_spv");
    let mut checked = 0;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("shaders_spv exists") {
        let path = entry.expect("read shaders_spv entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("spv") {
            continue;
        }
        checked += 1;
        let out = Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.3")
            .arg(&path)
            .output()
            .expect("run spirv-val");
        if !out.status.success() {
            failures.push(format!(
                "{}: {}",
                path.file_name().unwrap().to_string_lossy(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    }
    assert!(checked > 10, "expected the full module inventory, found {checked}");
    assert!(
        failures.is_empty(),
        "invalid SPIR-V modules in the tracked inventory:\n{}",
        failures.join("\n")
    );
}
