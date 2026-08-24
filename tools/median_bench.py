#!/usr/bin/env python3
"""Run N independent bench_q4_gemv processes; report median ms/iter under fixed OMP."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path


def one_trial(
    bench: Path, model: Path, gdir: Path, iters: int, omp: int
) -> tuple[str, float, bool]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    r = subprocess.run(
        [str(bench), str(model), str(gdir), str(iters)],
        capture_output=True,
        text=True,
        # explicit decode: stderr can carry non-ASCII model/golden path bytes
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(
        r"(PASS|FAIL)\s+(\S+)\s+.*\s+([0-9.]+)\s+ms/iter",
        out,
    )
    if not m:
        raise RuntimeError(f"parse fail:\n{out}")
    return m.group(2), float(m.group(3)), m.group(1) == "PASS"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run N independent bench_q4_gemv processes; report median ms/iter."
    )
    ap.add_argument("--bench", default="bench_q4_gemv", help="path to the bench binary")
    ap.add_argument("--model", default="model/Qwen3.8-27B-Q4_0.gguf", help="GGUF model path")
    ap.add_argument("--golden", required=True, help="golden/<tensor>/ dir to bench")
    ap.add_argument("--trials", type=int, default=5, help="independent processes to run")
    ap.add_argument("--iters", type=int, default=10, help="timed iterations per process")
    ap.add_argument("--omp", type=int, default=16, help="OMP_NUM_THREADS (one CCD on 9950X)")
    ap.add_argument("--label", default="", help="name shown in the summary line")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    bench = root / args.bench
    times = []
    ok_all = True
    name = "?"
    try:
        for t in range(args.trials):
            name, ms, ok = one_trial(
                bench, root / args.model, root / args.golden, args.iters, args.omp
            )
            times.append(ms)
            ok_all = ok_all and ok
            print(
                f"  trial {t + 1}/{args.trials}: {ms:.3f} ms  {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
    except (RuntimeError, OSError) as e:
        print(f"median_bench: {e}", file=sys.stderr, flush=True)
        return 1
    times.sort()
    med = statistics.median(times)
    print(
        f"{'PASS' if ok_all else 'FAIL'} {args.label or name}  "
        f"median={med:.3f} ms  trials={times}  OMP={args.omp}",
        flush=True,
    )
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
