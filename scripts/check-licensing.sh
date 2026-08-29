#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 voxel-engine contributors
# SPDX-License-Identifier: MIT OR Apache-2.0

set -euo pipefail

reuse lint
cargo deny check licenses bans sources
python3 tools/gen_font.py --check src/font.rs
