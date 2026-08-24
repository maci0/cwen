#!/usr/bin/env python3
"""Compare full-stack python ref vs C: residuals, logits, generated tokens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_f32(p: Path) -> np.ndarray:
    return np.fromfile(p, dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def need(p: Path) -> bool:
    """Required artifact: absence means the suite would compare nothing."""
    if p.is_file():
        return True
    # Fatal pre-check (main aborts below): stderr, like accept.py/compare_e2e.py;
    # stdout stays reserved for the per-section verdict report.
    print(f"FAIL missing {p} (rebuild dumps: make e2e-full-py / e2e-full-c)", file=sys.stderr)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare full-stack dumps: numpy reference vs C "
        "(residuals, logits, generated tokens)."
    )
    ap.add_argument(
        "--ref", default="golden/e2e_full", help="python reference dump dir (make e2e-full-py)"
    )
    ap.add_argument(
        "--c", dest="cdir", default="golden/e2e_full_c", help="C dump dir (make e2e-full-c)"
    )
    args = ap.parse_args()
    REF = Path(args.ref)
    C = Path(args.cdir)
    bad = 0

    # Required baseline on both sides (generators always write these); without
    # them every section below would SKIP and the run would pass vacuously.
    ref_logits = REF / (
        "logits_prefill.bin" if (REF / "logits_prefill.bin").is_file() else "logits.bin"
    )
    required = [
        REF / "embed.bin",
        C / "embed.bin",
        ref_logits,
        C / "logits_pos00.bin",
        REF / "meta.json",
        C / "tokens.txt",
    ]
    if not all(need(p) for p in required):
        return 1

    print("=== prefill residuals ===")
    for name in ["embed.bin"] + [f"layer{i:02d}.bin" for i in (0, 3, 15, 31, 47, 63)]:
        rp, cp = REF / name, C / name
        has_ref, has_c = rp.is_file(), cp.is_file()
        if has_ref != has_c:
            # Both dumps come from paired full-stack runs: one side holding the
            # artifact and the other not means a stale or truncated dump, and
            # silently skipping it would shrink coverage while still passing.
            have, lack = (REF, C) if has_ref else (C, REF)
            print(f"FAIL {name} in {have} but not {lack} (regenerate both dumps)")
            bad += 1
            continue
        if not has_ref:
            print(f"SKIP {name}")
            continue
        a, b = load_f32(rp), load_f32(cp)
        ma = float(np.max(np.abs(a - b)))
        c = cos(a, b)
        ok = ma < 1e-3 and c > 0.999
        print(f"{'PASS' if ok else 'FAIL'} {name:14s} max_abs={ma:.6g} cos={c:.8f}")
        bad += 0 if ok else 1

    print("=== prefill logits ===")
    rp = ref_logits
    cp = C / "logits_pos00.bin"
    if rp.is_file() and cp.is_file():
        a, b = load_f32(rp), load_f32(cp)
        c = cos(a, b)
        t1a, t1b = int(np.argmax(a)), int(np.argmax(b))
        ok = c > 0.999 and t1a == t1b
        print(
            f"{'PASS' if ok else 'FAIL'} logits cos={c:.8f} top1={t1a}/{t1b} "
            f"max_abs={float(np.max(np.abs(a - b))):.6g}"
        )
        bad += 0 if ok else 1
    else:
        print("SKIP logits_prefill")

    print("=== multi-token gen (C argmax chain) ===")
    tok_path = C / "tokens.txt"
    if tok_path.is_file():
        toks = [int(x) for x in tok_path.read_text(encoding="utf-8").split()]
        for i, t in enumerate(toks):
            lp = C / f"logits_pos{i:02d}.bin"
            if not lp.is_file():
                # tokens.txt and the per-position logits come from the same
                # dump run; a missing file is truncation, not a partial ref.
                print(f"FAIL logits_pos{i:02d} missing from {C} (token {t} unverifiable)")
                bad += 1
                continue
            top = int(np.argmax(load_f32(lp)))
            ok = top == t
            print(f"{'PASS' if ok else 'FAIL'} gen step {i}: argmax={top} token={t}")
            bad += 0 if ok else 1

        # python generated list (e2e_ref always writes it with --gen > 0)
        meta = json.loads((REF / "meta.json").read_text(encoding="utf-8"))
        py = meta.get("generated") or []
        if py:
            ok = py == toks
            print(f"{'PASS' if ok else 'FAIL'} tokens py={py} C={toks}")
            bad += 0 if ok else 1
        else:
            print("FAIL meta.json has no generated list")
            bad += 1
        # gen logits cos if both exist
        for gi in range(len(toks)):
            rp = REF / f"logits_gen{gi:02d}.bin"
            # C dumps pos{gi+1} after forwarding token gi; that forward's
            # logits choose token gi+1, exactly what python's logits_gen{gi}
            # holds after generating its first token
            cp = C / f"logits_pos{gi + 1:02d}.bin"
            has_ref, has_c = rp.is_file(), cp.is_file()
            if has_ref != has_c:
                print(
                    f"FAIL logits_gen{gi:02d}: ref={has_ref} c={has_c} "
                    "(generators ran with mismatched --gen, or a dump is truncated)"
                )
                bad += 1
                continue
            if not has_ref:
                print(f"SKIP logits_gen{gi:02d}")
                continue
            a, b = load_f32(rp), load_f32(cp)
            c = cos(a, b)
            ok = c > 0.999 and int(np.argmax(a)) == int(np.argmax(b))
            print(
                f"{'PASS' if ok else 'FAIL'} logits_gen{gi:02d} cos={c:.8f} "
                f"top={int(np.argmax(a))}/{int(np.argmax(b))}"
            )
            bad += 0 if ok else 1
    else:
        print("SKIP tokens.txt")
        bad += 1

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
