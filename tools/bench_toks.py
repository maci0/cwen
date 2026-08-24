#!/usr/bin/env python3
"""End-to-end tok/s for cwen run binary.

Reports:
  - whole-process tok/s (includes mmap load + prefill)
  - decode-only tok/s via (wall_N - wall_M)/(N-M)

Default --omp 16 mirrors production (cwen_tune.h ships 16 threads, one CCD);
override only when deliberately A/B-ing thread counts.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path


def parse_token_stream(raw: str, source: str) -> list[int]:
    """Strict decode of the engine's stdout: whitespace-separated decimal ids.

    The engine prints nothing else on stdout. Filtering junk out silently would
    let a corrupted or truncated stream shrink the pinned-chain comparison and
    the tok/s numbers instead of failing them.
    """
    toks: list[int] = []
    for field in raw.split():
        if not re.fullmatch(r"[0-9]+", field):
            raise RuntimeError(
                f"{source}: non-token stdout field {field!r} (raw head: {raw[:200]!r})"
            )
        toks.append(int(field))
    return toks


def run_once(
    run: Path, model: Path, prompt: Path, n_pred: int, env: dict[str, str], cwd: Path
) -> tuple[float, list[int], str]:
    t0 = time.perf_counter()
    r = subprocess.run(
        [str(run), str(model), str(prompt), str(n_pred)],
        capture_output=True,
        text=True,
        # run.c echoes user paths into stderr; decode explicitly so a non-ASCII
        # or non-UTF-8 filename can never crash the harness on LC_ALL=C
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(cwd),
    )
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        raise RuntimeError(f"run failed rc={r.returncode}\n{r.stderr}\n{r.stdout}")
    toks = parse_token_stream(r.stdout or "", str(run))
    return wall, toks, (r.stderr or "").strip()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="End-to-end tok/s for the cwen run binary (whole-process + decode-only)."
    )
    ap.add_argument("--run", default="run", help="path to the run binary")
    ap.add_argument("--model", default="model/Qwen3.8-27B-Q4_0.gguf", help="GGUF model path")
    ap.add_argument("--prompt", default="prompt1.ids", help="int32 prompt ids (written if missing)")
    ap.add_argument("--ns", default="2,4,8", help="comma list of n_predict values")
    ap.add_argument("--trials", type=int, default=3, help="runs per n_predict value")
    ap.add_argument("--omp", type=int, default=16, help="OMP_NUM_THREADS (one CCD on 9950X)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run = root / args.run
    model = root / args.model
    prompt = root / args.prompt
    if not prompt.exists():
        prompt.write_bytes(struct.pack("i", 248044))

    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.omp),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_DYNAMIC": "false",
        }
    )
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    print(
        f"tok/s bench  OMP={args.omp} trials={args.trials} ns={ns}",
        flush=True,
    )
    print(f"binary={run} model={model.name}", flush=True)

    medians: dict[int, float] = {}
    last_toks: list[int] = []
    for n in ns:
        walls: list[float] = []
        try:
            for t in range(args.trials):
                wall, toks, _err = run_once(run, model, prompt, n, env, root)
                walls.append(wall)
                last_toks = toks
                print(
                    f"  n={n} trial={t + 1}/{args.trials} wall={wall:.3f}s "
                    f"whole={n / wall:.3f} tok/s tokens={toks}",
                    flush=True,
                )
        except (RuntimeError, OSError) as e:
            print(f"bench_toks: {e}", file=sys.stderr, flush=True)
            return 1
        walls.sort()
        med = statistics.median(walls)
        medians[n] = med
        print(
            f"  >> n={n} median_wall={med:.3f}s whole_tok/s={n / med:.3f} "
            f"trials={[round(w, 3) for w in walls]}",
            flush=True,
        )

    # decode-only from largest vs smallest n
    if len(ns) >= 2:
        lo, hi = min(ns), max(ns)
        wlo, whi = medians[lo], medians[hi]
        if whi > wlo and hi > lo:
            spt = (whi - wlo) / (hi - lo)
            print(
                f"\ndecode-only ≈ (wall_{hi}-wall_{lo})/({hi}-{lo}) "
                f"= {spt:.4f} s/tok  →  {1.0 / spt:.3f} tok/s",
                flush=True,
            )
    print(f"last tokens: {last_toks}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
