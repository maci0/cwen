#!/usr/bin/env python3
"""Validate the paired split-Q8 (Q8SI) container + dual-dot math offline.

Reimplements run.c dot_q8si in numpy over the real layers.0.mlp_gu /
layers.0.attn_kv payloads, then checks:
  1. dequant rows A/B match safetensors gate/up / k/v rows (quant error only)
  2. dual-dot outputs for synthetic x match per-row matmul

Usage: .venv/bin/python tools/q8si_check.py [spec] [safetensors]
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
from pack_dflash import drafter_safetensors


def spec_tensor(data: bytes, entries: dict, name: str):
    ne0, ne1, typ, off, nb = entries[name]
    assert typ == 4, f"{name}: not Q8SI"
    rb = ne0 + (ne0 // 32) * 2
    blob = np.frombuffer(data[off : off + nb], dtype=np.uint8).reshape(ne1, rb)
    q = blob[:, :ne0].astype(np.int8).reshape(ne1, -1, 32).astype(np.float32)
    sc = (
        np.frombuffer(blob[:, ne0:].tobytes(), dtype="<u2")
        .view(np.float16)
        .astype(np.float32)
        .reshape(ne1, ne0 // 32)
    )
    rows = (q * sc[:, :, None]).reshape(ne1, ne0)
    return rows[0::2], rows[1::2]


def st_rows(path: Path, key: str) -> np.ndarray:
    with path.open("rb") as sf:
        n = struct.unpack("<Q", sf.read(8))[0]
        hdr = json.loads(sf.read(n))
        m = hdr[key]
        beg, end = m["data_offsets"]
        sf.seek(8 + n + beg)
        u16 = np.frombuffer(sf.read(end - beg), dtype="<u2").astype("<u4")
        return (u16 << np.uint32(16)).view("<f4").reshape(m["shape"])


def main() -> int:
    spec = Path(sys.argv[1] if len(sys.argv) > 1 else "model/dflash2.spec")
    stp = Path(sys.argv[2]) if len(sys.argv) > 2 else drafter_safetensors()
    data = spec.read_bytes()
    with spec.open("rb") as f:
        f.read(12)
        entries = {}
        for _ in range(struct.unpack("<I", data[8:12])[0]):
            nm = f.read(64)
            e = struct.unpack("<IIQQQ", f.read(32))
            entries[nm.split(b"\0")[0].decode()] = e

    fails = 0
    H = 5120
    rng = np.random.default_rng(7)
    x = rng.integers(0, 89, size=H).astype(np.float32) / 89.0 - 0.5

    for pair, akey, bkey in [
        ("layers.0.mlp_gu", "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight"),
        (
            "layers.0.attn_kv",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
        ),
    ]:
        if pair not in entries:
            print(f"SKIP {pair} (separate tensors)")
            continue
        A, B = spec_tensor(data, entries, pair)
        Ga = st_rows(stp, akey)
        Gb = st_rows(stp, bkey)
        da = np.abs(A[:3] - Ga[:3]).max()
        db = np.abs(B[:3] - Gb[:3]).max()
        ya = A[:3] @ x
        yb = B[:3] @ x
        ya_ref = Ga[:3] @ x
        yb_ref = Gb[:3] @ x
        ok = (
            da < 0.05
            and db < 0.05
            and np.abs(ya - ya_ref).max() < 0.5
            and np.abs(yb - yb_ref).max() < 0.5
        )
        print(
            f"{pair}: dequantA={da:.4g} dequantB={db:.4g} "
            f"yA={np.array2string(ya, precision=3)} yB={np.array2string(yb, precision=3)} "
            f"{'OK' if ok else 'FAIL'}"
        )
        if not ok:
            fails += 1
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
