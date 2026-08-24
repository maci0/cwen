#!/usr/bin/env python3
"""Generate GEMV goldens from the on-disk GGUF for C to check against."""

from __future__ import annotations

import argparse
import json
import random
import struct
import sys
from array import array
from pathlib import Path
from typing import TYPE_CHECKING

from gguf_util import GGUF
from ref_kernels import gemv_stream

if TYPE_CHECKING:
    from numpy.typing import NDArray


def write_f32(path: Path, xs: NDArray | array[float] | list[float]) -> None:
    if isinstance(xs, list):
        path.write_bytes(struct.pack(f"{len(xs)}f", *xs))
    else:
        path.write_bytes(xs.tobytes())  # ndarray / array path (already float32)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate GEMV goldens from the on-disk GGUF for C to check against."
    )
    ap.add_argument("--model", default="model/Qwen3.8-27B-Q4_0.gguf", help="GGUF model path")
    ap.add_argument("--out", default="golden", help="output root for golden dirs")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed for activation vectors")
    ap.add_argument(
        "--tensors",
        nargs="*",
        help="tensor names to dump (default: the verify set)",
        default=[
            "blk.0.attn_gate.weight",  # Q4_0 -> CWENR v4 solo split (T_Q4_0RS)
            "blk.0.ffn_gate.weight",  # Q4_0 -> CWENR v4 interleaved pair side
            "blk.0.ffn_down.weight",  # Q4_1 17408x5120
            "blk.0.ssm_out.weight",  # Q5_K
            "output.weight",  # Q6_K lm_head
        ],
    )
    args = ap.parse_args()
    model = Path(args.model)
    if not model.is_file():
        print(
            f"gen_golden: missing {args.model} (fetch the pinned weights: tools/download.sh)",
            file=sys.stderr,
        )
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"opening {args.model} ...", flush=True)
    g = GGUF.open(model)
    meta_all = []
    missing = []
    for name in args.tensors:
        t = g.tensors.get(name)
        if t is None:
            print(f"MISSING {name}", file=sys.stderr, flush=True)
            missing.append(name)
            continue
        K, M = t.ne0, t.ne1
        print(f"gemv {name} type={t.typ} [{K},{M}]", flush=True)
        x = [rng.uniform(-1.0, 1.0) for _ in range(K)]
        y = gemv_stream(t, x)
        safe = name.replace(".", "_")
        d = out / safe
        d.mkdir(exist_ok=True)
        write_f32(d / "x.bin", x)
        write_f32(d / "y_ref.bin", y)
        meta = {"name": name, "type": t.typ, "ne0": K, "ne1": M, "seed": args.seed}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        meta_all.append(meta)
        print(f"  wrote {d}  y[0]={y[0]:.6g} y[1]={y[1]:.6g}", flush=True)

    (out / "index.json").write_text(json.dumps(meta_all, indent=2), encoding="utf-8")
    g.close()
    if missing:
        print(
            f"gen_golden: {len(missing)}/{len(args.tensors)} requested tensors missing "
            f"in {args.model}: {', '.join(missing)}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
