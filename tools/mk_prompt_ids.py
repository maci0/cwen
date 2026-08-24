#!/usr/bin/env python3
"""Write prompt1.ids: the pinned BOS id 248044 as raw little-endian int32.

Shared by tools/test_speed_gates.sh and tools/profile_*.sh so the pin lives
in one place instead of a copy-pasted python -c per script. Python tools that
need the same prompt (bench_toks, idea_bench, e2e_ref) import BOS/write_ids
from here rather than restating the id.
"""

import struct
from pathlib import Path

BOS = 248044


def write_ids(path: Path, token: int = BOS) -> None:
    path.write_bytes(struct.pack("<i", token))


def main() -> int:
    write_ids(Path(__file__).resolve().parents[1] / "prompt1.ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
