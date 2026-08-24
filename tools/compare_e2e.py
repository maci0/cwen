#!/usr/bin/env python3
"""Compare golden/e2e (python) vs golden/e2e_c (C dumps)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare e2e dumps: numpy reference vs C (residuals + logits)."
    )
    ap.add_argument("--ref", default="golden/e2e", help="python reference dump dir (make e2e)")
    ap.add_argument("--c", default="golden/e2e_c", help="C dump dir (make e2e)")
    ap.add_argument("--atol", type=float, default=5e-4, help="per-tensor max_abs tolerance")
    ap.add_argument("--min-cos", type=float, default=0.999, help="per-tensor minimum cosine")
    args = ap.parse_args()
    ref, cdir = Path(args.ref), Path(args.c)
    # Required baseline: with no embed.bin on either side the loop below would
    # compare nothing and return success.
    missing = [str(p / "embed.bin") for p in (ref, cdir) if not (p / "embed.bin").is_file()]
    if missing:
        print(f"FAIL missing {', '.join(missing)} (regenerate dumps: make e2e)", file=sys.stderr)
        return 1
    names = ["embed.bin"] + sorted(p.name for p in ref.glob("layer*.bin"))
    has_ref_logits = (ref / "logits.bin").is_file()
    has_c_logits = (cdir / "logits.bin").is_file()
    if has_ref_logits != has_c_logits:
        # One side dumping logits and the other not is a stale or mismatched
        # dump pair, not a reason to drop the lm_head from the comparison.
        have, lack = (ref, cdir) if has_ref_logits else (cdir, ref)
        # Fatal pre-check like the missing-embed gate above: nothing gets
        # compared, so it reports on stderr with the verdict lines.
        print(f"FAIL logits.bin in {have} but not {lack} (regenerate both dumps)", file=sys.stderr)
        return 1
    if has_ref_logits:
        names.append("logits.bin")
    bad = 0
    for name in names:
        if not (cdir / name).is_file():
            print(f"FAIL {name} missing in {cdir}")
            bad += 1
            continue
        a = np.fromfile(ref / name, dtype=np.float32)
        b = np.fromfile(cdir / name, dtype=np.float32)
        if a.shape != b.shape:
            print(f"FAIL {name} shape {a.shape} vs {b.shape}")
            bad += 1
            continue
        d = np.abs(a - b)
        ma = float(d.max())
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        ok = ma <= args.atol and cos >= args.min_cos
        print(f"{'PASS' if ok else 'FAIL'} {name:14s} max_abs={ma:.6g} cos={cos:.8f}")
        bad += 0 if ok else 1
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
