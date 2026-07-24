<!--
SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
SPDX-License-Identifier: CC-BY-SA-4.0
-->

# voxel_engine licensing audit

Audit date: 2026-07-24

## Scope and status

This is a technical ownership and provenance audit of the separate
`voxel-engine` repository. It is not legal advice. The audit identified two
human contributors: gusahlg (including the `guahlg` alias with the same email)
and Restitutor. Both made substantial surviving software contributions.

The licensing-transition commit must not be pushed or published until each
affected human contributor's direct written confirmation has been retained in
the project records. Future contributions use DCO 1.1 sign-offs; no historical
commit contains a DCO sign-off, so the new policy is prospective.

## Method

The audit used:

```sh
git shortlog --summary --email --all
git log --format='%aN <%aE>' --all
git log --numstat --all
git blame -C -C --line-porcelain
```

It also searched current and historical files for third-party notices,
generated outputs, binary files, source URLs, copied/adapted markers, licence
metadata, and contribution trailers.

The history contains 57 commits by gusahlg/guahlg and 9 by Restitutor.
Move-aware current-tree attribution is approximately 9,600 surviving lines for
gusahlg/guahlg and 13,600 for Restitutor. These figures show contribution
magnitude; they are not a legal allocation of copyright.

## Supplied patch

Commit `6c247bf1cad6d8209c3f712e37117082e02e95f2`, titled “Apply supplied
voxel-engine patch,” changed 45 files. Its metadata does not name the supplier.
During this audit, the maintainer confirmed that they personally supplied it.
Retain that statement with the licensing-transition records.

## Third-party and generated material

- `src/font.rs` contains public-domain `font8x8_basic` glyph data surrounded by
  project-owned AGPL code. The exact upstream revision, source hash, custom
  SPDX LicenseRef, and deterministic verification tool are recorded in
  [THIRD_PARTY.md](THIRD_PARTY.md).
- The 19 checked-in SPIR-V files are generated from project-owned Slang source.
  Each binary has an AGPL `.license` sidecar, and `build.rs` emits SPDX headers
  into generated textual source.
- Cargo packages use the checksummed crates.io registry. They retain their
  original free-software licences; all-target dependency checks remain a
  release requirement.
- `Cargo.lock`, `flake.lock`, generated Slang includes, and SPIR-V are generated
  outputs whose preferred source and build logic are present in the tree.

## Confirmation record

Before pushing the transition, retain a durable copy of each contributor's
direct statement, including date, identity, scope, and record location:

> I confirm that I have the right to license my contributions to Project Watt
> Cubed and voxel-engine under GNU AGPL version 3 or any later version. For
> artistic and documentation contributions, I agree to Creative Commons
> Attribution-ShareAlike 4.0.

Do not publish private correspondence or personal details beyond what is
necessary for a reliable internal record.

The supplied licensing plan identifies the project lead as under 18. Before
publishing the transition, have a parent or guardian participate in the
project lead's initial licensing declaration and seek Swedish
intellectual-property advice where practical. This technical audit is not a
substitute for that review.
