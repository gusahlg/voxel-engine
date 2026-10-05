//! Optional compiled-shader statistics (`VOXEL_SHADER_STATS`).
//!
//! Unset (or `0`): nothing here runs and `VK_KHR_pipeline_executable_properties`
//! is not enabled. `1` prints one stderr line per pipeline executable.
//! `2` also writes internal representations under `VOXEL_SHADER_STATS_DIR`
//! (default `./shader_stats`).

use std::ffi::{CStr, FromBytesUntilNulError};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ash::vk;

pub(crate) type Loader = ash::khr::pipeline_executable_properties::Device;

/// `0` off, `1` statistics, `2` statistics plus internal representations.
pub(crate) fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VOXEL_SHADER_STATS") {
        Ok(v) if v == "0" => 0,
        Ok(v) if v == "2" => 2,
        Ok(_) => 1,
        Err(_) => 0,
    })
}

pub(crate) fn enabled() -> bool {
    mode() != 0
}

/// Flags to OR into pipeline create info. Empty unless the extension is live.
pub(crate) fn capture_flags(loader: Option<&Loader>) -> vk::PipelineCreateFlags {
    if loader.is_none() {
        return vk::PipelineCreateFlags::empty();
    }
    let mut flags = vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR;
    if mode() >= 2 {
        flags |= vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR;
    }
    flags
}

pub(crate) fn report_pipeline_stats(
    device: &ash::Device,
    loader: Option<&Loader>,
    pipeline: vk::Pipeline,
    name: &str,
) {
    let Some(loader) = loader else {
        return;
    };
    let _ = device.handle();
    let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
    let executables = match unsafe { loader.get_pipeline_executable_properties(&info) } {
        Ok(v) => v,
        Err(err) => {
            eprintln!("shader-stats {name}: properties query failed: {err}");
            return;
        }
    };
    for (index, exec) in executables.iter().enumerate() {
        let stage = stage_label(exec.stages);
        let exec_name = lossy_cstr(exec.name_as_c_str());
        let exec_info = vk::PipelineExecutableInfoKHR::default()
            .pipeline(pipeline)
            .executable_index(index as u32);
        let stats = match unsafe { loader.get_pipeline_executable_statistics(&exec_info) } {
            Ok(v) => v,
            Err(err) => {
                eprintln!(
                    "shader-stats {name} {stage} {exec_name}: statistics query failed: {err}"
                );
                continue;
            }
        };
        let mut line = format!("shader-stats {name} {stage} {exec_name}:");
        for stat in &stats {
            let stat_name = stat_name_token(&lossy_cstr(stat.name_as_c_str()));
            let value = format_stat(stat.format, stat.value);
            line.push(' ');
            line.push_str(&stat_name);
            line.push('=');
            line.push_str(&value);
        }
        eprintln!("{line}");
        if mode() >= 2 {
            write_internal(loader, &exec_info, name, &stage, index, &exec_name);
        }
    }
}

fn write_internal(
    loader: &Loader,
    exec_info: &vk::PipelineExecutableInfoKHR<'_>,
    pipeline: &str,
    stage: &str,
    index: usize,
    exec_name: &str,
) {
    let mut reps =
        match unsafe { loader.get_pipeline_executable_internal_representations(exec_info) } {
            Ok(v) => v,
            Err(err) => {
                eprintln!("shader-stats {pipeline}: internal representations query failed: {err}");
                return;
            }
        };
    if reps.is_empty() {
        note_ir_unavailable();
        return;
    }
    let mut storage: Vec<Vec<u8>> = reps.iter().map(|r| vec![0u8; r.data_size]).collect();
    for (rep, buf) in reps.iter_mut().zip(storage.iter_mut()) {
        if !buf.is_empty() {
            rep.p_data = buf.as_mut_ptr().cast();
        }
    }
    let mut count = reps.len() as u32;
    let result = unsafe {
        (loader
            .fp()
            .get_pipeline_executable_internal_representations_khr)(
            loader.device(),
            exec_info,
            &mut count,
            reps.as_mut_ptr(),
        )
    };
    if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
        eprintln!("shader-stats {pipeline}: internal representations read failed: {result}");
        return;
    }
    let dir = stats_dir();
    if let Err(err) = std::fs::create_dir_all(dir) {
        eprintln!("shader-stats: create {}: {err}", dir.display());
        return;
    }
    for (rep_index, (rep, buf)) in reps.iter().zip(storage.iter()).enumerate() {
        let rep_name = lossy_cstr(rep.name_as_c_str());
        let file = dir.join(format!(
            "{}_{}_{}.txt",
            sanitize(pipeline),
            sanitize(stage),
            sanitize(&format!("{exec_name}_{rep_index}_{rep_name}"))
        ));
        let body = if rep.is_text == vk::TRUE {
            let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
            String::from_utf8_lossy(&buf[..end]).into_owned()
        } else {
            format!(
                "binary representation, {} bytes (index {index})\n",
                buf.len()
            )
        };
        if let Err(err) = std::fs::write(&file, body) {
            eprintln!("shader-stats: write {}: {err}", file.display());
        }
    }
}

fn note_ir_unavailable() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!("shader-stats: internal representations unavailable");
    });
}

fn stats_dir() -> &'static Path {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(|| match std::env::var_os("VOXEL_SHADER_STATS_DIR") {
        Some(v) => PathBuf::from(v),
        None => PathBuf::from("shader_stats"),
    })
}

fn lossy_cstr(name: Result<&CStr, FromBytesUntilNulError>) -> String {
    name.map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|_| "?".into())
}

pub(crate) fn stat_name_token(name: &str) -> String {
    name.replace(' ', "_")
}

fn format_stat(
    format: vk::PipelineExecutableStatisticFormatKHR,
    value: vk::PipelineExecutableStatisticValueKHR,
) -> String {
    unsafe {
        if format == vk::PipelineExecutableStatisticFormatKHR::BOOL32 {
            if value.b32 == vk::TRUE {
                "true".to_string()
            } else {
                "false".to_string()
            }
        } else if format == vk::PipelineExecutableStatisticFormatKHR::INT64 {
            value.i64.to_string()
        } else if format == vk::PipelineExecutableStatisticFormatKHR::UINT64 {
            value.u64.to_string()
        } else if format == vk::PipelineExecutableStatisticFormatKHR::FLOAT64 {
            value.f64.to_string()
        } else {
            format!("?({format:?})")
        }
    }
}

pub(crate) fn stage_label(stages: vk::ShaderStageFlags) -> String {
    const BITS: &[(vk::ShaderStageFlags, &str)] = &[
        (vk::ShaderStageFlags::VERTEX, "vertex"),
        (vk::ShaderStageFlags::TESSELLATION_CONTROL, "tess_control"),
        (vk::ShaderStageFlags::TESSELLATION_EVALUATION, "tess_eval"),
        (vk::ShaderStageFlags::GEOMETRY, "geometry"),
        (vk::ShaderStageFlags::FRAGMENT, "fragment"),
        (vk::ShaderStageFlags::COMPUTE, "compute"),
        (vk::ShaderStageFlags::TASK_EXT, "task"),
        (vk::ShaderStageFlags::MESH_EXT, "mesh"),
    ];
    let mut parts = Vec::new();
    for (bit, name) in BITS {
        if stages.contains(*bit) {
            parts.push(*name);
        }
    }
    if parts.is_empty() {
        "unknown".into()
    } else {
        parts.join("+")
    }
}

pub(crate) fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '.' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() { "_".into() } else { out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_names_replace_spaces() {
        assert_eq!(stat_name_token("Register Count"), "Register_Count");
        assert_eq!(stat_name_token("binary_size"), "binary_size");
    }

    #[test]
    fn stage_labels_join_bits() {
        assert_eq!(stage_label(vk::ShaderStageFlags::FRAGMENT), "fragment");
        assert_eq!(
            stage_label(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            "vertex+fragment"
        );
    }

    #[test]
    fn sanitize_filename_pieces() {
        assert_eq!(sanitize("mesh3d_opaque"), "mesh3d_opaque");
        assert_eq!(sanitize("bloom threshold"), "bloom_threshold");
        assert_eq!(sanitize("a/b"), "a_b");
    }

    #[test]
    fn formats_stat_values() {
        let b = vk::PipelineExecutableStatisticValueKHR { b32: vk::TRUE };
        assert_eq!(
            format_stat(vk::PipelineExecutableStatisticFormatKHR::BOOL32, b),
            "true"
        );
        let i = vk::PipelineExecutableStatisticValueKHR { i64: -3 };
        assert_eq!(
            format_stat(vk::PipelineExecutableStatisticFormatKHR::INT64, i),
            "-3"
        );
        let u = vk::PipelineExecutableStatisticValueKHR { u64: 42 };
        assert_eq!(
            format_stat(vk::PipelineExecutableStatisticFormatKHR::UINT64, u),
            "42"
        );
        let f = vk::PipelineExecutableStatisticValueKHR { f64: 1.5 };
        assert_eq!(
            format_stat(vk::PipelineExecutableStatisticFormatKHR::FLOAT64, f),
            "1.5"
        );
    }
}
