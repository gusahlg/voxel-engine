#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Project Watt Cubed contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reproduce and verify the printable-ASCII font8x8 Rust table."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys
import urllib.request

UPSTREAM_REVISION = "8e279d2d864e79128e96188a6b9526cfa3fbfef9"
UPSTREAM_URL = (
    "https://raw.githubusercontent.com/dhepper/font8x8/"
    f"{UPSTREAM_REVISION}/font8x8_basic.h"
)
UPSTREAM_SHA256 = (
    "49d8df366296b203ca3211bc0672cf2a762135bf12710735b6292756b19dffd5"
)
HEX_BYTE = re.compile(r"0x([0-9A-Fa-f]{2})")
CODEPOINT = re.compile(r"U\+([0-9A-Fa-f]{4})")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        help="read the pinned font8x8_basic.h from this path instead of HTTPS",
    )
    parser.add_argument(
        "--check",
        type=pathlib.Path,
        help="verify the GLYPHS snippet in this Rust file instead of printing rows",
    )
    return parser.parse_args()


def load_header(path: pathlib.Path | None) -> bytes:
    if path is not None:
        data = path.read_bytes()
    else:
        request = urllib.request.Request(
            UPSTREAM_URL, headers={"User-Agent": "voxel-engine-font-generator"}
        )
        with urllib.request.urlopen(request) as response:
            data = response.read()

    actual = hashlib.sha256(data).hexdigest()
    if actual != UPSTREAM_SHA256:
        raise ValueError(
            f"font8x8_basic.h SHA-256 mismatch: expected {UPSTREAM_SHA256}, "
            f"got {actual}"
        )
    return data


def parse_header(data: bytes) -> list[tuple[int, ...]]:
    rows: dict[int, tuple[int, ...]] = {}
    for line in data.decode("utf-8").splitlines():
        point = CODEPOINT.search(line)
        values = tuple(int(value, 16) for value in HEX_BYTE.findall(line))
        if point is not None and len(values) == 8:
            rows[int(point.group(1), 16)] = values

    missing = [point for point in range(32, 127) if point not in rows]
    if missing:
        raise ValueError(f"upstream header is missing code points: {missing}")
    return [rows[point] for point in range(32, 127)]


def parse_rust(path: pathlib.Path) -> list[tuple[int, ...]]:
    source = path.read_text(encoding="utf-8")
    try:
        snippet = source.split("// SPDX-SnippetBegin", 1)[1].split(
            "// SPDX-SnippetEnd", 1
        )[0]
    except IndexError as error:
        raise ValueError(f"{path} has no font SPDX snippet") from error

    rows = [
        tuple(int(value, 16) for value in HEX_BYTE.findall(line))
        for line in snippet.splitlines()
    ]
    return [row for row in rows if len(row) == 8]


def render(rows: list[tuple[int, ...]]) -> str:
    output = []
    for point, row in zip(range(32, 127), rows, strict=True):
        values = ", ".join(f"0x{value:02X}" for value in row)
        label = "space" if point == 32 else chr(point)
        output.append(f"    [{values}], // {point} {label}")
    return "\n".join(output)


def main() -> int:
    args = arguments()
    expected = parse_header(load_header(args.input))
    if args.check is not None:
        actual = parse_rust(args.check)
        if actual != expected:
            print(f"{args.check}: embedded glyph table differs from upstream", file=sys.stderr)
            return 1
        print(
            f"{args.check}: 95 glyphs match font8x8 {UPSTREAM_REVISION} "
            f"({UPSTREAM_SHA256})"
        )
        return 0

    print(render(expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
