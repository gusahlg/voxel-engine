// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use crate::toolchain::cargo_rerun_path;

/// Transitive `#include "..."` closure of `src` (Slang resolves quoted
/// includes relative to the including file, then each `-I` dir), in
/// deterministic first-visit order. Files that do not exist are skipped:
/// slangc reports those itself.
pub fn include_closure(src: &Path, include_dirs: &[PathBuf]) -> Vec<PathBuf> {
    let mut order = Vec::new();
    let mut seen = HashSet::new();
    let mut stack = vec![src.to_path_buf()];
    while let Some(path) = stack.pop() {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        if !seen.insert(path.clone()) {
            continue;
        }
        let dir = path.parent().unwrap_or(Path::new(""));
        // Push in reverse so includes are visited in source order.
        let mut includes: Vec<PathBuf> = text
            .lines()
            .filter_map(|line| {
                quoted_include(line).map(|name| resolve_include(dir, name, include_dirs))
            })
            .collect();
        includes.reverse();
        stack.extend(includes);
        order.push(path);
    }
    order
}

fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix("#include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let (name, _) = rest.split_once('"')?;
    Some(name)
}

fn resolve_include(from_dir: &Path, name: &str, include_dirs: &[PathBuf]) -> PathBuf {
    let relative = from_dir.join(name);
    if relative.exists() {
        return relative;
    }
    for dir in include_dirs {
        let candidate = dir.join(name);
        if candidate.exists() {
            return candidate;
        }
    }
    relative
}

pub(crate) fn emit_rerun_for_jobs(
    jobs: &[crate::ShaderJob<'_>],
    include_dirs: &[PathBuf],
    fallback_dir: &Path,
) {
    cargo_rerun_path(fallback_dir);
    let mut seen = HashSet::new();
    for job in jobs {
        if seen.insert(job.src.to_path_buf()) {
            cargo_rerun_path(job.src);
        }
        for path in include_closure(job.src, include_dirs) {
            if seen.insert(path.clone()) {
                cargo_rerun_path(&path);
            }
        }
    }
    for dir in include_dirs {
        cargo_rerun_path(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let p = std::env::temp_dir()
            .join("voxel_slang_build_includes")
            .join(name);
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn quoted_include_parses_and_ignores_angle_brackets() {
        assert_eq!(
            quoted_include("  #include \"foo.slang\""),
            Some("foo.slang")
        );
        assert_eq!(quoted_include("#include <foo.slang>"), None);
        assert_eq!(quoted_include("// #include \"foo.slang\""), None);
        assert_eq!(quoted_include("const int x = 1;"), None);
    }

    #[test]
    fn include_closure_walks_quoted_includes_in_source_order() {
        let dir = scratch("walk");
        let a = dir.join("a.slang");
        let b = dir.join("b.slang");
        let c = dir.join("c.slang");
        fs::write(&a, "#include \"b.slang\"\n#include \"c.slang\"\n").unwrap();
        fs::write(&b, "// b\n").unwrap();
        fs::write(&c, "// c\n").unwrap();
        let got = include_closure(&a, &[]);
        assert_eq!(got, vec![a, b, c]);
    }

    #[test]
    fn include_closure_skips_missing_files() {
        let dir = scratch("missing");
        let a = dir.join("a.slang");
        fs::write(&a, "#include \"nope.slang\"\n").unwrap();
        let got = include_closure(&a, &[]);
        assert_eq!(got, vec![a]);
    }

    #[test]
    fn include_closure_searches_include_dirs_when_relative_is_absent() {
        let root = scratch("idirs");
        let src_dir = root.join("src");
        let inc_dir = root.join("inc");
        fs::create_dir_all(&src_dir).unwrap();
        fs::create_dir_all(&inc_dir).unwrap();
        let src = src_dir.join("main.slang");
        let header = inc_dir.join("shared.slang");
        fs::write(&src, "#include \"shared.slang\"\n").unwrap();
        fs::write(&header, "// shared\n").unwrap();
        let got = include_closure(&src, &[inc_dir]);
        assert_eq!(got, vec![src, header]);
    }

    #[test]
    fn include_closure_prefers_relative_over_include_dirs() {
        let root = scratch("prefer");
        let src_dir = root.join("src");
        let inc_dir = root.join("inc");
        fs::create_dir_all(&src_dir).unwrap();
        fs::create_dir_all(&inc_dir).unwrap();
        let src = src_dir.join("main.slang");
        let local = src_dir.join("shared.slang");
        let other = inc_dir.join("shared.slang");
        fs::write(&src, "#include \"shared.slang\"\n").unwrap();
        fs::write(&local, "// local\n").unwrap();
        fs::write(&other, "// other\n").unwrap();
        let got = include_closure(&src, &[inc_dir]);
        assert_eq!(got, vec![src, local]);
    }
}
