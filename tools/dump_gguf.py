#!/usr/bin/env python3
"""List tensors in a GGUF file (no full load)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gguf_util import GGUF, TYPE_NAMES

INTERESTING_KEYS = (
    "general.architecture",
    "general.name",
    "general.file_type",
    "qwen3next.context_length",
    "qwen3.context_length",
    "llama.context_length",
)


def main() -> int:
    ap = argparse.ArgumentParser(description="List tensors in a GGUF file (no full load).")
    ap.add_argument(
        "gguf",
        nargs="?",
        default="model/Qwen3.8-27B-Q4_0.gguf",
        help="GGUF path (default: model/Qwen3.8-27B-Q4_0.gguf)",
    )
    args = ap.parse_args()
    path = Path(args.gguf)
    if not path.is_file():
        print(f"missing {path}", file=sys.stderr)
        return 1
    try:
        g = GGUF.open(path)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"file={path} version={g.version} tensors={len(g.tensors)} kv={len(g.kv)}")
    for key, val in g.kv.items():
        if (
            key in INTERESTING_KEYS
            or key.endswith(".block_count")
            or key.endswith(".embedding_length")
        ):
            print(f"  kv {key} = {val}")
    for i, t in enumerate(g.tensors.values()):
        name = TYPE_NAMES.get(t.typ, str(t.typ))
        print(f"{i:4d}  {name:6s}  dims={t.ne}  off={t.off}  {t.name}")
    print(f"listed {len(g.tensors)} tensors")
    g.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BrokenPipeError:
        # reader closed the pipe early (`| head`): die quietly instead of
        # tracebacking; devnull keeps the interpreter's shutdown flush silent
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(1) from None
