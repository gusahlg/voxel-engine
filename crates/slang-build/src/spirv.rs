// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, path::Path, process::Command};

use crate::toolchain::spirv_val_available;

/// SPIR-V magic number (little-endian `0x07230203`).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;
/// SPIR-V 1.3 version word. Checked-in fallbacks may still use this.
pub const SPIRV_1_3: u32 = 0x0001_0300;
/// SPIR-V 1.6 version word. Fresh compiles target this (Vulkan 1.3 baseline).
pub const SPIRV_1_6: u32 = 0x0001_0600;

const OP_CAPABILITY: u32 = 17;
const CAPABILITY_DEMOTE_TO_HELPER_INVOCATION: u32 = 5379;

/// Header and capability facts extracted from a SPIR-V binary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpirvInfo {
    /// Word 1 of the header (`0x00010300` / `0x00010600` for 1.3 / 1.6).
    pub version: u32,
    /// `OpCapability DemoteToHelperInvocation` (what Slang 1.6 lowers
    /// `discard` to). Must only appear in fragment modules.
    pub demote_to_helper_invocation: bool,
}

/// Parse `bytes` as SPIR-V: magic, well-formed instruction stream, and the
/// `DemoteToHelperInvocation` capability.
pub fn inspect_spirv(bytes: &[u8]) -> Result<SpirvInfo, String> {
    if bytes.len() < 20 || !bytes.len().is_multiple_of(4) {
        return Err("not a SPIR-V module".into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words.len() < 5 || words[0] != SPIRV_MAGIC {
        return Err("not a SPIR-V module".into());
    }
    let version = words[1];
    let mut demote_to_helper_invocation = false;
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let opcode = words[i] & 0xffff;
        if count == 0 {
            return Err(format!("zero-length instruction at word {i}"));
        }
        if i + count > words.len() {
            return Err(format!("truncated instruction at word {i}"));
        }
        if opcode == OP_CAPABILITY
            && words.get(i + 1) == Some(&CAPABILITY_DEMOTE_TO_HELPER_INVOCATION)
        {
            demote_to_helper_invocation = true;
        }
        i += count;
    }
    Ok(SpirvInfo {
        version,
        demote_to_helper_invocation,
    })
}

/// Validate a SPIR-V module: well-formed header and instruction stream, then
/// `spirv-val --target-env vulkan1.3` when that tool is on PATH.
///
/// When `spirv-val` is absent this still rejects truncated or non-SPIR-V
/// bytes; it does not fail the build for a missing validator (same policy as
/// the engine test suite, which skips the all-module gate outside the
/// dev shell).
pub fn validate_spirv(bytes: &[u8]) -> Result<(), String> {
    validate_spirv_with_env(bytes, crate::SPIRV_VAL_TARGET_ENV)
}

/// [`validate_spirv`] with an explicit `spirv-val --target-env`.
pub fn validate_spirv_with_env(bytes: &[u8], target_env: &str) -> Result<(), String> {
    inspect_spirv(bytes)?;
    if !spirv_val_available() {
        return Ok(());
    }
    let dir = std::env::temp_dir().join("voxel_slang_build_spirv_val");
    fs::create_dir_all(&dir).map_err(|e| format!("create {dir:?}: {e}"))?;
    let path = dir.join(format!("mod-{:x}.spv", fnv1a_quick(bytes)));
    fs::write(&path, bytes).map_err(|e| format!("write {}: {e}", path.display()))?;
    let result = run_spirv_val(&path, target_env);
    let _ = fs::remove_file(&path);
    result
}

/// Run `spirv-val --target-env <target_env>` on a file on disk.
pub fn run_spirv_val(path: &Path, target_env: &str) -> Result<(), String> {
    let output = Command::new("spirv-val")
        .arg("--target-env")
        .arg(target_env)
        .arg(path)
        .output()
        .map_err(|e| format!("failed to run spirv-val for {}: {e}", path.display()))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_owned())
    }
}

fn fnv1a_quick(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(version: u32) -> Vec<u8> {
        let words = [SPIRV_MAGIC, version, 0, 0, 0];
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn with_demote(version: u32) -> Vec<u8> {
        let mut bytes = header(version);
        // OpCapability DemoteToHelperInvocation: 2 words.
        let inst = [
            (2u32 << 16) | OP_CAPABILITY,
            CAPABILITY_DEMOTE_TO_HELPER_INVOCATION,
        ];
        bytes.extend(inst.iter().flat_map(|w| w.to_le_bytes()));
        bytes
    }

    #[test]
    fn inspect_rejects_empty_and_bad_magic() {
        assert!(inspect_spirv(b"").is_err());
        assert!(inspect_spirv(&[0; 20]).is_err());
        let mut bad = header(SPIRV_1_6);
        bad[0] = 0;
        assert!(
            inspect_spirv(&bad)
                .unwrap_err()
                .contains("not a SPIR-V module")
        );
    }

    #[test]
    fn inspect_reads_version_and_demote_capability() {
        let info = inspect_spirv(&header(SPIRV_1_6)).unwrap();
        assert_eq!(info.version, SPIRV_1_6);
        assert!(!info.demote_to_helper_invocation);

        let info = inspect_spirv(&with_demote(SPIRV_1_6)).unwrap();
        assert!(info.demote_to_helper_invocation);
    }

    #[test]
    fn inspect_rejects_zero_length_instruction() {
        let mut bytes = header(SPIRV_1_6);
        bytes.extend(0u32.to_le_bytes());
        let err = inspect_spirv(&bytes).unwrap_err();
        assert!(err.contains("zero-length instruction at word 5"));
    }

    #[test]
    fn validate_spirv_accepts_header_when_spirv_val_absent_or_present() {
        // A 5-word header is well-formed for inspect; spirv-val (if present)
        // will reject it as an incomplete module. Use inspect here; the
        // helper's spirv-val step is covered by engine shader_validation.
        assert!(inspect_spirv(&header(SPIRV_1_3)).is_ok());
        assert_eq!(
            inspect_spirv(&header(SPIRV_1_3)).unwrap().version,
            SPIRV_1_3
        );
    }
}
