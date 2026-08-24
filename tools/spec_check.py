#!/usr/bin/env python3
"""Greedy-lossless check for CWEN_SPEC block speculation.

Runs ./run twice on repetitive prompts, with and without CWEN_SPEC=1, and
requires identical token streams (greedy verification is lossless by
construction, so any divergence is a bug). Also prints decode tok/s for both
modes as a quick A/B.

Model load dominates wall time (~25 s each), so defaults are one prompt,
modest n_predict; raise --repeats/--n-predict for real benches.
"""

from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

# strict engine-stdout contract lives here; keep one copy
from bench_toks import parse_token_stream

# same corpus text spec_e2e uses, so both gates exercise the identical prompt
from spec_e2e import REPEAT_TEXT


def build_prompt_ids(repeats: int) -> list[int]:
    """Repeated text with the last instance cut short: models tend to continue
    the pattern, which is what feeds the n-gram lookup."""
    try:
        from transformers import AutoTokenizer
    except ImportError as err:
        print("transformers not installed; run 'make setup'", file=sys.stderr)
        raise SystemExit(1) from err
    tok = AutoTokenizer.from_pretrained("model")
    full = tok.encode(REPEAT_TEXT, add_special_tokens=False)
    part = tok.encode(REPEAT_TEXT[: len(REPEAT_TEXT) // 2], add_special_tokens=False)
    return full * repeats + part


def write_repetitive_prompt(path: Path, repeats: int) -> None:
    ids = build_prompt_ids(repeats)
    path.write_bytes(struct.pack(f"{len(ids)}i", *ids))


def run_once(
    run: Path, model: Path, prompt: Path, n_pred: int, spec: bool
) -> tuple[float, list[int], str]:
    env = dict(os.environ)
    env["CWEN_SPEC"] = "1" if spec else "0"
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(run), str(model), str(prompt), str(n_pred)],
        capture_output=True,
        text=True,
        env=env,
        # explicit decode: run.c stderr can carry non-ASCII path bytes
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"run failed rc={r.returncode}\n{r.stderr}")
    toks = parse_token_stream(r.stdout or "", str(run))
    return wall, toks, (r.stderr or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Greedy-lossless check for CWEN_SPEC block speculation."
    )
    ap.add_argument("--run", type=Path, default=Path("./run"), help="path to the run binary")
    ap.add_argument(
        "--model", type=Path, default=Path("model/Qwen3.8-27B-Q4_0.gguf"), help="GGUF model path"
    )
    ap.add_argument(
        "--prompt", type=Path, default=Path("spec_check.ids"), help="prompt ids file to write"
    )
    ap.add_argument("--repeats", type=int, default=8, help="prompt repetition count")
    ap.add_argument("--n-predict", type=int, default=48, help="tokens to generate per mode")
    args = ap.parse_args()

    write_repetitive_prompt(args.prompt, args.repeats)
    print(f"prompt: {args.prompt} ({args.repeats}x repeated text)")
    args.run = args.run.resolve()

    try:
        wall_plain, toks_plain, err_plain = run_once(
            args.run, args.model, args.prompt, args.n_predict, spec=False
        )
        wall_spec, toks_spec, err_spec = run_once(
            args.run, args.model, args.prompt, args.n_predict, spec=True
        )
    except (RuntimeError, OSError) as e:
        print(f"spec_check: {e}", file=sys.stderr)
        return 1

    ok = toks_plain == toks_spec and len(toks_plain) == args.n_predict
    print(
        f"plain : {len(toks_plain)} tokens, {wall_plain:.1f}s wall "
        f"({(args.n_predict - 1) / max(wall_plain - 25, 0.5):.2f} tok/s est)"
    )
    print(
        f"spec  : {len(toks_spec)} tokens, {wall_spec:.1f}s wall "
        f"({(args.n_predict - 1) / max(wall_spec - 25, 0.5):.2f} tok/s est)"
    )
    for line in err_spec.splitlines():
        if line.startswith("spec:"):
            print(f"spec stats: {line[6:]}")

    if not ok:
        bad = next(
            (i for i, (a, b) in enumerate(zip(toks_plain, toks_spec, strict=False)) if a != b),
            min(len(toks_plain), len(toks_spec)),
        )
        print(
            f"FAIL: streams diverge at index {bad}: "
            f"plain={toks_plain[bad : bad + 4]} spec={toks_spec[bad : bad + 4]}"
        )
        return 1
    print("PASS: greedy token streams identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
