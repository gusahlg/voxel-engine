//! All-module SPIR-V validation gate (E-08 follow-through): every tracked
//! fallback in `shaders_spv/` must pass `spirv-val` at the shipping Vulkan
//! target. This is what keeps an invalid module (like the old
//! `mesh3d_water.frag.spv` scalar `OpImageRead`) from sitting silently in the
//! tree behind a disabled pipeline. Skips — loudly — when `spirv-val` is not
//! on PATH, so plain `cargo test` outside the dev shell still passes; run
//! inside `nix develop` for the full gate.

use voxel_slang_build::{
    SPIRV_1_3, SPIRV_1_6, detect_toolchain, inspect_spirv, spirv_val_available, validate_spirv,
};

#[test]
fn all_tracked_spirv_modules_pass_spirv_val() {
    if !spirv_val_available() {
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
        let bytes = std::fs::read(&path).expect("read module");
        if let Err(e) = validate_spirv(&bytes) {
            failures.push(format!(
                "{}: {e}",
                path.file_name().unwrap().to_string_lossy()
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
    let have_spirv_val = spirv_val_available();
    let have_slangc = detect_toolchain().is_some();
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
        let info = match inspect_spirv(&bytes) {
            Ok(info) => info,
            Err(e) => {
                failures.push(format!("{name}: {e}"));
                continue;
            }
        };
        // Fresh compiles target SPIR-V 1.6; the checked-in fallbacks may lag.
        let version_ok = if have_slangc {
            info.version == SPIRV_1_6
        } else {
            info.version == SPIRV_1_3 || info.version == SPIRV_1_6
        };
        if !version_ok {
            failures.push(format!(
                "{name}: unexpected SPIR-V version {:#x}",
                info.version
            ));
        }
        if info.demote_to_helper_invocation {
            saw_demote = true;
            if !name.contains(".frag.") {
                failures.push(format!(
                    "{name}: DemoteToHelperInvocation capability outside a fragment module"
                ));
            }
        }
        if have_spirv_val {
            if let Err(e) = validate_spirv(&bytes) {
                failures.push(format!("{name}: {e}"));
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
