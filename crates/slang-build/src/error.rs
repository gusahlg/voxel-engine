// SPDX-FileCopyrightText: 2026 voxel-engine contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fmt;

/// Failure from [`compile_all`](crate::compile_all) or a helper it calls.
///
/// Messages match the engine `build.rs` diagnostics so a game using this
/// crate sees the same text for missing fallbacks, a missing compiler, or a
/// rejected module.
#[derive(Debug)]
pub struct BuildError {
    message: String,
}

impl BuildError {
    pub(crate) fn msg(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BuildError {}

impl From<String> for BuildError {
    fn from(message: String) -> Self {
        Self { message }
    }
}
