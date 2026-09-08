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
    assert!(
        checked > 10,
        "expected the full module inventory, found {checked}"
    );
    assert!(
        failures.is_empty(),
        "invalid SPIR-V modules in the tracked inventory:\n{}",
        failures.join("\n")
    );
}

/// The modules the engine actually links come from `OUT_DIR` — freshly
/// compiled by build.rs when `slangc` is on PATH, copied from `shaders_spv/`
/// otherwise. Either way each must be a well-formed SPIR-V module that
/// `spirv-val` accepts for Vulkan 1.3, and the `DemoteToHelperInvocation`
/// capability (what `discard` lowers to at SPIR-V 1.6) may only appear in
/// fragment modules: it is the one capability `vk/device.rs` has to opt into.
#[test]
fn linked_spirv_modules_are_valid_for_vulkan_1_3() {
    const OP_CAPABILITY: u32 = 17;
    const CAPABILITY_DEMOTE_TO_HELPER_INVOCATION: u32 = 5379;
    const SPIRV_MAGIC: u32 = 0x0723_0203;
    const SPIRV_1_3: u32 = 0x0001_0300;
    const SPIRV_1_6: u32 = 0x0001_0600;
    let have_spirv_val = Command::new("spirv-val").arg("--version").output().is_ok();
    let have_slangc = Command::new("slangc").arg("-v").output().is_ok();
    if !have_spirv_val {
        eprintln!("spirv-val not on PATH: only checking headers/capabilities of OUT_DIR modules");
    }
    let dir = std::path::Path::new(env!("OUT_DIR"));
    let mut checked = 0;
    let mut saw_demote = false;
    let mut failures = Vec::new();
    for entry in std::fs::read_dir(dir).expect("OUT_DIR exists") {
        let path = entry.expect("read OUT_DIR entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("spv") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        // Probe shaders (VOXEL_BUILD_PROBE) are not the shipping inventory.
        if name.starts_with("probe_") {
            continue;
        }
        checked += 1;
        let bytes = std::fs::read(&path).expect("read module");
        let words: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        if words.len() < 5 || words[0] != SPIRV_MAGIC {
            failures.push(format!("{name}: not a SPIR-V module"));
            continue;
        }
        // Fresh compiles target SPIR-V 1.6; the checked-in fallbacks may lag.
        let version = words[1];
        let version_ok = if have_slangc {
            version == SPIRV_1_6
        } else {
            version == SPIRV_1_3 || version == SPIRV_1_6
        };
        if !version_ok {
            failures.push(format!("{name}: unexpected SPIR-V version {version:#x}"));
        }
        let mut demote = false;
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            let opcode = words[i] & 0xffff;
            if count == 0 {
                failures.push(format!("{name}: zero-length instruction at word {i}"));
                break;
            }
            if opcode == OP_CAPABILITY
                && words.get(i + 1) == Some(&CAPABILITY_DEMOTE_TO_HELPER_INVOCATION)
            {
                demote = true;
            }
            i += count;
        }
        if demote {
            saw_demote = true;
            if !name.contains(".frag.") {
                failures.push(format!(
                    "{name}: DemoteToHelperInvocation capability outside a fragment module"
                ));
            }
        }
        if have_spirv_val {
            let out = Command::new("spirv-val")
                .arg("--target-env")
                .arg("vulkan1.3")
                .arg(&path)
                .output()
                .expect("run spirv-val");
            if !out.status.success() {
                failures.push(format!(
                    "{name}: {}",
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
            }
        }
    }
    assert!(
        checked > 10,
        "expected the full module inventory in OUT_DIR, found {checked}"
    );
    if have_slangc {
        assert!(
            saw_demote,
            "slangc SPIR-V 1.6 must lower mesh3d `discard` to DemoteToHelperInvocation"
        );
    }
    assert!(
        failures.is_empty(),
        "linked SPIR-V modules failed the Vulkan 1.3 gate:\n{}",
        failures.join("\n")
    );
}
