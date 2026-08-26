#!/usr/bin/env python3
"""Check the GA's symbolic thr = f(M, K) expression tree.

eval_expr and expr_to_c must agree (the Python evaluator is what fitness
scores; the C string is what lands in cwen_tune.h), and restore_expr is a
parser over golden/ga_log/best.json, so it has to reject malformed input
rather than pass it through to the C emitter.

Usage: .venv/bin/python tools/ga_expr_check.py
"""

from __future__ import annotations

import json
import random
import subprocess
import tempfile
from pathlib import Path

from ga_evolve import Expr, eval_expr, expr_to_c, rand_expr, restore_expr, simplify_const_expr

ROOT = Path(__file__).resolve().parents[1]
CASES = [(1, 32), (7, 5120), (32, 5120), (5120, 5120), (17408, 5120), (248320, 5120)]


def c_values(exprs: list[Expr]) -> list[list[int]]:
    """Compile each expression as C and evaluate it over CASES."""
    lines = ["#include <stdio.h>", "int main(void){int M,K;"]
    for e in exprs:
        for m, k in CASES:
            lines.append(f'  M={m};K={k};printf("%lld\\n", (long long)({expr_to_c(e)}));')
    lines.append("  return 0;}")
    with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as d:
        src, exe = Path(d) / "e.c", Path(d) / "e"
        src.write_text("\n".join(lines))
        subprocess.run(["gcc", "-O0", "-o", str(exe), str(src)], check=True)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    nums = [int(v) for v in out.split()]
    n = len(CASES)
    return [nums[i * n : (i + 1) * n] for i in range(len(exprs))]


def main() -> int:
    (ROOT / ".scratch").mkdir(exist_ok=True)
    rng = random.Random(11)
    exprs = [simplify_const_expr(rand_expr(rng, 0, max_depth=3)) for _ in range(60)]

    fails = 0
    for e, got in zip(exprs, c_values(exprs), strict=True):
        want = [eval_expr(e, m, k) for m, k in CASES]
        if want != got:
            print(f"FAIL eval/C mismatch for {e!r}: python={want} c={got}")
            fails += 1

    # JSON round trip: tuples decode as lists and must rebuild identically
    for e in exprs:
        back = restore_expr(json.loads(json.dumps(e)))
        if back != e:
            print(f"FAIL round trip {e!r} -> {back!r}")
            fails += 1

    for bad in ([], ["nope", 1, 2], ["+", 1], True, None, {"op": "+"}, ["+", 1, None]):
        try:
            restore_expr(bad)
        except ValueError:
            continue
        print(f"FAIL restore_expr accepted {bad!r}")
        fails += 1

    if fails:
        print(f"FAIL ga expr: {fails} problems")
        return 1
    print(f"PASS ga expr: {len(exprs)} trees, eval/C agree, parser rejects malformed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
