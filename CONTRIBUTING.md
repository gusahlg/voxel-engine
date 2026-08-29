<!--
SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# Contributing

Thank you for improving Project Watt Cubed and `voxel_engine`.

## Inbound equals outbound

By contributing, you agree that your software contribution is licensed under
`AGPL-3.0-or-later` and your creative-content contribution is licensed under
`CC-BY-SA-4.0`, unless the file clearly records another approved compatible
licence. You retain your copyright. You grant no separate proprietary
relicensing permission.

The project does not require copyright assignment or a proprietary-relicensing
CLA.

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

When the repository enables them, contributors must also pass `reuse lint`,
`cargo deny check`, shader validation and reproducibility checks.

## Review

Pull requests should describe:

- what changed and why;
- the checks run;
- visible or performance impact;
- Vulkan/platform assumptions; and
- any third-party, generated or tool-assisted inputs.

Security-sensitive reports should follow [SECURITY.md](SECURITY.md), not a
public issue.
