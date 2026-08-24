#!/usr/bin/env python3
"""Encode text to int32 token-id blob for ./run."""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Encode text to an int32 token-id blob for ./run.")
    ap.add_argument("text", nargs="?", default="Hello", help="text to encode")
    ap.add_argument("-o", "--out", default="prompt.ids", help="output int32 blob path")
    ap.add_argument(
        "--model", default="model", help="tokenizer source: local dir or hub id (default: model/)"
    )
    ap.add_argument("--chat", action="store_true", help="wrap as user chat turn")
    args = ap.parse_args()

    # argv arrives via surrogateescape when the shell hands over non-UTF-8
    # bytes; tokenizers rejects those with a cryptic TypeError, so gate here.
    try:
        args.text.encode("utf-8")
    except UnicodeEncodeError:
        print("error: text is not valid UTF-8 (undecodable bytes in argument)", file=sys.stderr)
        return 1

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("transformers not installed; run 'make setup'", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat:
        messages = [{"role": "user", "content": args.text}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok.encode(text, add_special_tokens=False)
    else:
        ids = tok.encode(args.text, add_special_tokens=True)

    Path(args.out).write_bytes(struct.pack(f"{len(ids)}i", *ids))
    print(f"wrote {len(ids)} tokens -> {args.out}")
    print(ids[:32], ("..." if len(ids) > 32 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
