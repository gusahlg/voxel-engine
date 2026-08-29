<!--
SPDX-FileCopyrightText: 2026 voxel-engine contributors
SPDX-License-Identifier: MIT OR Apache-2.0
-->

# Contributing

Thank you for improving `voxel_engine`.

## Inbound equals outbound

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed under `MIT OR Apache-2.0` as described in
[LICENSE.md](LICENSE.md), without any additional terms or conditions. You
retain your copyright.

The project does not require copyright assignment or a CLA.

## Developer Certificate of Origin

Every commit must certify the [Developer Certificate of Origin 1.1](DCO). Add a
sign-off made with your real identity:

```sh
git commit --signoff
```

This adds:

```text
Signed-off-by: Your Name <your.email@example.com>
```

The sign-off is a certification, not merely attribution. Do not sign for work
you lack the right to submit. Preserve valid sign-offs while rebasing.

## Provenance

For every contribution:

- identify imported, adapted, generated or tool-assisted material;
- preserve upstream copyright and licence notices;
- provide the upstream URL and exact revision for imported source;
- include the preferred editable source and reproduction steps for generated
  files; and
- do not submit proprietary material, binary-only dependencies or content with
  unknown redistribution rights.

If an AI or other code-generation tool materially assisted a contribution,
record that fact in the pull request. The human submitter remains responsible
for reviewing the result, verifying provenance and making the DCO
certification.

## Development

Run the relevant checks before submitting:

```sh
cargo fmt --check
cargo clippy --all-targets
cargo test
```

Shader changes should also be compiled with the pinned Slang toolchain and
validated. Checked-in SPIR-V is a fallback generated from `shaders/`; do not
edit it directly. Refresh it with
`nix develop --command env VOXEL_ENGINE_REFRESH_SHADER_FALLBACKS=1 cargo check --locked`,
review the generated diff, then run `cargo test --test shader_validation`.
Changes to generated shader constants belong in `build.rs`.

Licensing metadata and dependency policy are checked with
`scripts/check-licensing.sh` (`reuse lint`, `cargo deny check`, and the font
provenance check); `nix flake check` runs `reuse lint` as well.

## Review

Pull requests should describe:

- what changed and why;
- the checks run;
- visible or performance impact;
- Vulkan/platform assumptions; and
- any third-party, generated or tool-assisted inputs.

Security-sensitive reports should follow [SECURITY.md](SECURITY.md), not a
public issue.
