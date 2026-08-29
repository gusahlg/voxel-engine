<!--
SPDX-FileCopyrightText: 2026 voxel-engine contributors
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Third-party material

This file records material included in or required to build `voxel_engine` that
is not owned by the voxel-engine contributors.

## Embedded `font8x8_basic` glyph data

- **Location:** the `GLYPHS` table in `src/font.rs`
- **Author/compiler:** Daniel Hepper `<daniel@hepper.net>`
- **Earlier source:** public-domain IBM VGA fonts collected by Marcel Sondaar
- **Upstream:** <https://github.com/dhepper/font8x8>
- **Upstream revision:** `8e279d2d864e79128e96188a6b9526cfa3fbfef9`
- **Upstream file:** <https://github.com/dhepper/font8x8/blob/8e279d2d864e79128e96188a6b9526cfa3fbfef9/font8x8_basic.h>
- **Upstream file SHA-256:**
  `49d8df366296b203ca3211bc0672cf2a762135bf12710735b6292756b19dffd5`
- **Licence/status:** Public Domain, recorded as
  `LicenseRef-font8x8-Public-Domain`
- **Use:** printable ASCII glyphs 32 through 126

`tools/gen_font.py` downloads that immutable revision (or accepts a local
copy), verifies the full-file hash, extracts printable ASCII, and can check the
embedded Rust table:

```sh
python3 tools/gen_font.py --check src/font.rs
```

The glyph data retains its public-domain status. The surrounding Rust
atlas-building code is `MIT OR Apache-2.0`.

## Rust dependencies

`Cargo.lock` is the exact dependency record. All current package sources use
the checksummed crates.io registry; there are no Git dependencies. The locked
direct dependencies report these SPDX expressions:

| Package | Locked version | Licence expression |
| --- | ---: | --- |
| `ash` | 0.38.0+1.3.281 | `MIT OR Apache-2.0` |
| `ash-window` | 0.13.0 | `MIT OR Apache-2.0` |
| `winit` | 0.30.12 | `Apache-2.0` |
| `raw-window-handle` | 0.6.2 | `MIT OR Apache-2.0 OR Zlib` |
| `glam` | 0.32.1 | `MIT OR Apache-2.0` |
| `bytemuck` | 1.25.0 | `Zlib OR Apache-2.0 OR MIT` |
| `log` | 0.4.29 | `MIT OR Apache-2.0` |
| `env_logger` | 0.11.9 | `MIT OR Apache-2.0` |
| `png` | 0.17.16 | `MIT OR Apache-2.0` |

This table is not a complete transitive licence report. Before each release,
run an all-target `cargo deny`/SBOM audit and ship the required third-party
licence texts and notices beside binaries.

At the 2026-07-24 audit snapshot, the full locked graph passed
`cargo deny check licenses bans sources`. All packages satisfied the
free-licence allowlist and registry/Git-source rules; duplicate dependency
versions remain warnings.

## Slang shader compiler

The Nix development environment currently resolves `shader-slang` 2026.5.2,
reported by Nixpkgs as `Apache-2.0` with `LLVM-exception`. It is a build tool,
not vendored into this repository. The checked-in SPIR-V files are generated
from this project's Slang source and are `MIT OR Apache-2.0` like that source.
