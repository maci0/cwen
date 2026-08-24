#!/usr/bin/env python3
"""Compare C dump y.bin vs y_ref.bin under absolute/relative tol."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path


def load_f32(path: Path) -> list[float]:
    b = path.read_bytes()
    n = len(b) // 4
    return list(struct.unpack(f"{n}f", b))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare C dump y.bin vs y_ref.bin under absolute/relative tol."
    )
    ap.add_argument("dir", type=Path, help="golden/<tensor>/ with y_ref.bin and y_c.bin")
    ap.add_argument("--atol", type=float, default=5e-3, help="abs cap in the soft band above 1e-4")
    args = ap.parse_args()
    missing = [p.name for p in (args.dir / "y_ref.bin", args.dir / "y_c.bin") if not p.is_file()]
    if missing:
        print(
            f"FAIL {args.dir}: missing {', '.join(missing)} "
            "(run bench_q4_gemv MODEL DIR first to write y_c.bin)",
            file=sys.stderr,
        )
        return 1
    y_ref = load_f32(args.dir / "y_ref.bin")
    y_c = load_f32(args.dir / "y_c.bin")
    if len(y_ref) != len(y_c):
        print(f"FAIL len ref={len(y_ref)} c={len(y_c)}")
        return 1
    # NaN/inf must fail outright: every d > ma comparison below is False for a
    # NaN diff, so without this gate a poisoned kernel output would PASS.
    if not all(math.isfinite(v) for v in y_c) or not all(math.isfinite(v) for v in y_ref):
        print(f"FAIL {args.dir.name}: non-finite value in y_ref.bin/y_c.bin")
        return 1
    ma = mr = 0.0
    bi = 0
    for i, (a, b) in enumerate(zip(y_ref, y_c, strict=True)):
        d = abs(a - b)
        if d > ma:
            ma = d
            bi = i
        r = d / max(abs(a), abs(b), 1e-8)
        if r > mr:
            mr = r
    meta = {}
    mp = args.dir / "meta.json"
    if mp.is_file():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    # Mirror bench_q4_gemv's internal gate (run.c): tiny abs error passes
    # outright; in the soft band abs is capped by --atol and rel must stay
    # small. Absolute error is the real signal for Q4; rel blows up near zero.
    ok = ma <= 1e-4 or (ma <= args.atol and mr <= 0.02)
    print(
        f"{'PASS' if ok else 'FAIL'} {args.dir.name}  "
        f"max_abs={ma:.6g} (i={bi}) max_rel={mr:.6g}  "
        f"tol_abs={args.atol}  meta={meta.get('name', '')}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
