#!/usr/bin/env python3
"""A/B every compile-time idea vs baseline. Correctness (PASS) + median ms.

Usage:
  .venv/bin/python tools/idea_bench.py
  .venv/bin/python tools/idea_bench.py --toks   # also decode-only tok/s for winners
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from mk_prompt_ids import write_ids

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "model" / "Qwen3.8-27B-Q4_0.gguf"
BENCH = ROOT / "bench_idea"  # per-idea binary name
OUT = ROOT / "golden" / "idea_bench.json"

GOLDENS = [
    ("golden/blk_0_attn_gate_weight", "Q4_0"),
    ("golden/blk_0_ffn_down_weight", "Q4_1"),
    ("golden/blk_0_ssm_out_weight", "Q5_K"),
]

# (name, extra -D flags). baseline first.
# Flags retired from run.c (measured, rejected or shipped unconditionally):
# MULTIROW, VNNI, COPY_X, BG_STREAM, POPULATE, WARM, HELP_PF, MLP_TEAM
# (one-team mlp is now the only implementation).
IDEAS = [
    # baseline = measured-default win PAIR_GEMV off (fair A/B).
    ("baseline", ["-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("fast_silu", ["-DCWEN_IDEA_FAST_SILU=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("madvise", ["-DCWEN_IDEA_MADVISE=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("ccd_pin", ["-DCWEN_IDEA_CCD=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("pf_t0", ["-DCWEN_IDEA_PF_T0=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("no_pf", ["-DCWEN_IDEA_NO_PF=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("pipe_pf", ["-DCWEN_IDEA_PIPE_PF=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("collapse", ["-DCWEN_IDEA_COLLAPSE=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("serial_mlp", ["-DCWEN_IDEA_SERIAL_MLP=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("gdn_omp", ["-DCWEN_IDEA_GDN_OMP=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("pair_gemv", ["-DCWEN_IDEA_PAIR_GEMV=1"]),
    ("pf_one", ["-DCWEN_IDEA_PF_ONE=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("madv_seq", ["-DCWEN_IDEA_MADV_SEQ=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("no_silu_omp", ["-DCWEN_IDEA_NO_SILU_OMP=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    ("conv_omp", ["-DCWEN_IDEA_CONV_OMP=1", "-DCWEN_IDEA_PAIR_GEMV=0"]),
    (
        "gdn_pair_nsilu",
        ["-DCWEN_IDEA_GDN_OMP=1", "-DCWEN_IDEA_PAIR_GEMV=1", "-DCWEN_IDEA_NO_SILU_OMP=1"],
    ),
]


def gcc_compile(extra: list[str], flags: list[str], out_bin: Path) -> tuple[bool, str]:
    cmd = [
        "gcc",
        "-O3",
        "-std=c11",
        "-Wno-unused-function",
        "-fopenmp",
        "-mavx2",
        "-mfma",
        "-mf16c",
        "-fno-math-errno",
        "-fno-trapping-math",
        "-fomit-frame-pointer",
        "-mavx512f",
        "-mavx512bw",
        "-mavx512vl",
        "-mavx512dq",
        "-mavx512vnni",
        "-march=native",
        "-DCWEN_AVX512",
        *extra,
        *flags,
        "-o",
        str(out_bin),
        str(ROOT / "run.c"),
        "-lm",
    ]
    r = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return r.returncode == 0, (r.stderr or "") + (r.stdout or "")


def one_bench(bin_path: Path, gdir: Path, iters: int, omp: int) -> tuple[bool, float, str]:
    """Return (ok, ms/iter, failure detail tail)."""
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(omp),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
            "OMP_DYNAMIC": "false",
        }
    )
    r = subprocess.run(
        [str(bin_path), str(MODEL), str(gdir), str(iters)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(PASS|FAIL)\s+\S+\s+.*\s+([0-9.]+)\s+ms/iter", out)
    if not m:
        return False, 1e9, out[-500:]
    ok = m.group(1) == "PASS"
    return ok, float(m.group(2)), "" if ok else out[-500:]


def median_bench(bin_path: Path, gdir: Path, trials: int, iters: int, omp: int) -> dict:
    times = []
    ok_all = True
    detail = ""
    for _ in range(trials):
        ok, ms, d = one_bench(bin_path, gdir, iters, omp)
        ok_all = ok_all and ok
        if not ok:
            detail = d
        times.append(ms)
    if not ok_all:
        # keep the reason visible next to the FAIL row instead of dropping it
        print(f"  bench failure detail ({gdir.name}): {detail}", file=sys.stderr)
    times.sort()
    return {
        "ok": ok_all,
        "median": statistics.median(times),
        "trials": times,
    }


def tok_s(bin_path: Path, omp: int) -> float:
    """decode-only via wall(n=8)-wall(n=2)."""
    prompt = ROOT / "prompt1.ids"
    if not prompt.exists():
        write_ids(prompt)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(omp),
            "OMP_PROC_BIND": "close",
            "OMP_PLACES": "cores",
        }
    )

    def wall(n: int) -> float:
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(bin_path), str(MODEL), str(prompt), str(n)],
            capture_output=True,
            env=env,
            cwd=str(ROOT),
        )
        wall = time.perf_counter() - t0
        if r.returncode != 0:
            # a crashed run would otherwise shrink w8 toward w2 and surface
            # as a plausible-looking 0.0 tok/s
            err = (r.stderr or b"")[-500:]
            raise RuntimeError(f"{bin_path.name} failed rc={r.returncode}: {err!r}")
        return wall

    # warm
    wall(2)
    w2 = statistics.median([wall(2) for _ in range(2)])
    w8 = statistics.median([wall(8) for _ in range(2)])
    if w8 <= w2:
        return 0.0
    return 6.0 / (w8 - w2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="A/B every compile-time CWEN_IDEA_* flag vs baseline: goldens + median ms."
    )
    ap.add_argument("--trials", type=int, default=3, help="median_bench trials per kernel")
    ap.add_argument("--iters", type=int, default=10, help="timed iterations per trial")
    ap.add_argument("--omp", type=int, default=16, help="OMP_NUM_THREADS (one CCD on 9950X)")
    ap.add_argument(
        "--toks",
        action="store_true",
        help="also decode-only tok/s for kernels that pass all goldens",
    )
    ap.add_argument("--only", default="", help="comma names to run")
    args = ap.parse_args()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    known = {name for name, _ in IDEAS}
    unknown = sorted(only - known)
    if unknown:
        print(f"unknown idea(s) {', '.join(unknown)}; have {sorted(known)}", file=sys.stderr)
        return 2

    results: list[dict[str, Any]] = []
    print(f"idea_bench  trials={args.trials} iters={args.iters} omp={args.omp}", flush=True)

    for name, flags in IDEAS:
        if only and name not in only:
            continue
        bin_path = ROOT / f"bench_idea_{name}"
        print(f"\n=== {name}  flags={flags} ===", flush=True)
        ok, log = gcc_compile(["-Wall", "-DCWEN_BENCH_Q4_GEMV"], flags, bin_path)
        if not ok:
            print(f"  COMPILE FAIL:\n{log[-800:]}", flush=True)
            results.append({"name": name, "flags": flags, "compile": False, "log": log[-500:]})
            continue
        row: dict[str, Any] = {"name": name, "flags": flags, "compile": True, "kernels": {}}
        all_ok = True
        for grel, label in GOLDENS:
            # Q4_1/Q5 unaffected by multirow/vnni; still measure
            m = median_bench(bin_path, ROOT / grel, args.trials, args.iters, args.omp)
            row["kernels"][label] = m
            all_ok = all_ok and m["ok"]
            st = "PASS" if m["ok"] else "FAIL"
            print(f"  {st} {label:6s} median={m['median']:.3f} ms  {m['trials']}", flush=True)
        row["all_ok"] = all_ok
        if args.toks and all_ok:
            rbin = ROOT / f"run_idea_{name}"
            rok, rlog = gcc_compile([], flags, rbin)
            if rok:
                try:
                    tps = tok_s(rbin, args.omp)
                except RuntimeError as e:
                    print(f"  tok/s measurement failed: {e}", flush=True)
                else:
                    row["tok_s"] = tps
                    print(f"  decode tok/s ≈ {tps:.3f}", flush=True)
            else:
                print(f"  run compile fail: {rlog[-200:]}", flush=True)
        results.append(row)

    # summary table vs baseline
    base = next((r for r in results if r.get("name") == "baseline" and r.get("compile")), None)
    print("\n======== SUMMARY vs baseline ========", flush=True)
    print(
        f"{'idea':12s} {'ok':4s} {'Q4_0':>8s} {'Q4_1':>8s} {'Q5_K':>8s} {'vsQ4':>7s} {'vsQ5':>7s}",
        flush=True,
    )
    for r in results:
        if not r.get("compile"):
            print(f"{r['name']:12s} COMPILE_FAIL", flush=True)
            continue
        k = r["kernels"]

        def med(lab, kernels=k):
            return kernels.get(lab, {}).get("median", float("nan"))

        q0, q1, q5 = med("Q4_0"), med("Q4_1"), med("Q5_K")
        if base and base.get("kernels"):
            b0 = base["kernels"]["Q4_0"]["median"]
            b5 = base["kernels"]["Q5_K"]["median"]
            s0 = b0 / q0 if q0 > 0 else 0
            s5 = b5 / q5 if q5 > 0 else 0
        else:
            s0 = s5 = 1.0
        ok_txt = "YES" if r.get("all_ok") else "NO"
        print(
            f"{r['name']:12s} {ok_txt:4s} {q0:8.3f} {q1:8.3f} {q5:8.3f} {s0:6.2f}x {s5:6.2f}x",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)

    # cleanup idea binaries optional keep for debug
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
