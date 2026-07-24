#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

reuse lint
cargo deny check licenses bans sources
python3 tools/gen_font.py --check src/font.rs
