#!/usr/bin/env python3
"""Offline GGUF Q4_0 → CWENR sidecar.

v4 (default): v3 solo + interleaved dual-mat pairs
  pairs: ffn_gate+ffn_up, attn_k+attn_v (same shape)
  per pair row/block: qsA[16]|qsB[16], scA[f16]|scB[f16]
  one sequential DRAM stream for dual-mat gemv.

v3: split qs[16*nb*ne1] + f16 scales[nb*ne1] per tensor
v2: packed 20B {qs[16], f32 d}

  magic[8]     = b"CWENR001"
  version u32  = 4
  n_tensors u32
  data_base u64
  stamp u64    = tag u32 (b"CWEN") | src_pages u32; src_pages = ceil(gguf_bytes/4096).
                 run.c drops a stamped sidecar whose source GGUF size no longer
                 matches, so a replaced GGUF can never serve stale weights.
                 All zeros = legacy untagged sidecar (trusted as before).
  directory: n × 128 B
    name[96], ne0, ne1, qs_off, sc_off, flags u32, pad u32
    flags: 0=solo, 1=interleaved A (primary), 2=interleaved B (partner, same offs)

Usage:
  .venv/bin/python tools/repack_q4.py model/Qwen3.8-27B-Q4_0.gguf
  .venv/bin/python tools/repack_q4.py model/Qwen3.8-27B-Q4_0.gguf --sidecar-version 3
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

from gguf_util import GGUF, QK4, T_Q4_0

MAGIC = b"CWENR001"
ENT_SIZE = 128
F_SOLO = 0
F_IL_A = 1
F_IL_B = 2
STAMP_TAG = 0x4E455743  # b"CWEN"; pairs with CWENR_STAMP_TAG in run.c

# (primary_suffix, partner_suffix) — primary written first as IL_A
PAIR_SUFFIXES = (
    ("ffn_gate.weight", "ffn_up.weight"),
    ("attn_k.weight", "attn_v.weight"),
)


def align_up(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def default_out(gguf: Path) -> Path:
    if gguf.suffix.lower() == ".gguf":
        return gguf.with_suffix(".cwenr")
    return Path(str(gguf) + ".cwenr")


def pack_v2_tensor(raw: memoryview, ne0: int, ne1: int) -> bytes:
    nb = ne0 // QK4
    src = np.frombuffer(raw, dtype=np.uint8).reshape(ne1, nb, 18)
    out = np.zeros((ne1, nb, 20), dtype=np.uint8)
    out[:, :, 0:16] = src[:, :, 2:18]
    d_f32 = src[:, :, 0:2].copy().view(np.dtype("<f2")).astype(np.float32).reshape(ne1, nb)
    out[:, :, 16:20] = np.ascontiguousarray(d_f32).view(np.uint8).reshape(ne1, nb, 4)
    return out.reshape(-1).tobytes()


def pack_v3_tensor(raw: memoryview, ne0: int, ne1: int) -> tuple[bytes, bytes]:
    """Return (qs_blob, sc_f16_blob)."""
    nb = ne0 // QK4
    src = np.frombuffer(raw, dtype=np.uint8).reshape(ne1, nb, 18)
    qs = np.ascontiguousarray(src[:, :, 2:18]).reshape(-1).tobytes()
    sc = np.ascontiguousarray(src[:, :, 0:2]).reshape(-1).tobytes()
    return qs, sc


def pack_v4_pair(raw_a: memoryview, raw_b: memoryview, ne0: int, ne1: int) -> tuple[bytes, bytes]:
    """Interleave two same-shape Q4_0 tensors: (qsA|qsB)*nblk, (scA|scB)*nblk."""
    nb = ne0 // QK4
    sa = np.frombuffer(raw_a, dtype=np.uint8).reshape(ne1, nb, 18)
    sb = np.frombuffer(raw_b, dtype=np.uint8).reshape(ne1, nb, 18)
    # qs: (ne1, nb, 2, 16) then flatten
    qs = np.empty((ne1, nb, 2, 16), dtype=np.uint8)
    qs[:, :, 0, :] = sa[:, :, 2:18]
    qs[:, :, 1, :] = sb[:, :, 2:18]
    sc = np.empty((ne1, nb, 2, 2), dtype=np.uint8)
    sc[:, :, 0, :] = sa[:, :, 0:2]
    sc[:, :, 1, :] = sb[:, :, 0:2]
    return qs.reshape(-1).tobytes(), sc.reshape(-1).tobytes()


def partner_name(name: str, a_suf: str, b_suf: str) -> str | None:
    if not name.endswith(a_suf):
        return None
    return name[: -len(a_suf)] + b_suf


def build_plan(q4: list, ver: int) -> list[dict]:
    """Return ordered list of write plans (solo or pair)."""
    by_name = dict(q4)
    used: set[str] = set()
    plans: list[dict] = []

    if ver == 4:
        for name, t in q4:
            if name in used:
                continue
            paired = False
            for a_suf, b_suf in PAIR_SUFFIXES:
                pn = partner_name(name, a_suf, b_suf)
                if pn is None or pn not in by_name:
                    continue
                ta, tb = t, by_name[pn]
                if ta.ne0 != tb.ne0 or ta.ne1 != tb.ne1:
                    print(f"  pair shape mismatch {name} vs {pn}", flush=True)
                    break
                if ta.ne0 % QK4:
                    break
                plans.append(
                    {
                        "kind": "pair",
                        "a": name,
                        "b": pn,
                        "ta": ta,
                        "tb": tb,
                        "ne0": ta.ne0,
                        "ne1": ta.ne1,
                    }
                )
                used.add(name)
                used.add(pn)
                paired = True
                break
            if paired:
                continue
            if t.ne0 % QK4:
                print(f"  skip {name}: ne0={t.ne0}", flush=True)
                continue
            plans.append({"kind": "solo", "name": name, "t": t, "ne0": t.ne0, "ne1": t.ne1})
            used.add(name)
    else:
        for name, t in q4:
            if t.ne0 % QK4:
                print(f"  skip {name}: ne0={t.ne0}", flush=True)
                continue
            plans.append({"kind": "solo", "name": name, "t": t, "ne0": t.ne0, "ne1": t.ne1})
    return plans


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline Q4_0 → CWENR")
    ap.add_argument("gguf", type=Path, help="source GGUF with Q4_0 tensors")
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="sidecar path (default: GGUF path with .cwenr suffix)",
    )
    ap.add_argument(
        "--sidecar-version",
        type=int,
        choices=(2, 3, 4),
        default=4,
        help="CWENR sidecar format version to write",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the layout plan, write nothing")
    args = ap.parse_args()
    if not args.gguf.is_file():
        print(
            f"repack_q4: missing {args.gguf} (fetch the pinned weights: tools/download.sh)",
            file=sys.stderr,
        )
        return 1
    out = args.out or default_out(args.gguf)
    ver = args.sidecar_version

    print(f"open {args.gguf}  → CWENR v{ver}", flush=True)
    g = GGUF.open(args.gguf)
    q4 = sorted(
        [(n, t) for n, t in g.tensors.items() if t.typ == T_Q4_0],
        key=lambda x: x[0],
    )
    print(f"Q4_0 tensors: {len(q4)}", flush=True)
    plans = build_plan(q4, ver)
    npairs = sum(1 for p in plans if p["kind"] == "pair")
    nsolo = sum(1 for p in plans if p["kind"] == "solo")
    print(f"plans: {len(plans)}  pairs={npairs} solo={nsolo}", flush=True)

    # directory entries + payload layout
    # each entry: (name, ne0, ne1, qs_off, sc_off, flags, plan_ref, is_b)
    entries: list[tuple] = []
    off = 0
    total = 0
    for pl in plans:
        ne0, ne1 = pl["ne0"], pl["ne1"]
        nblk = (ne0 // QK4) * ne1
        if ver == 2:
            nbytes = nblk * 20
            off = align_up(off, 64)
            entries.append((pl["name"], ne0, ne1, off, nbytes, F_SOLO, pl, False, nbytes, 0))
            off += nbytes
            total += nbytes
            continue
        if pl["kind"] == "pair" and ver == 4:
            qs_nb, sc_nb = nblk * 32, nblk * 4  # interleaved
            off = align_up(off, 64)
            qs_off = off
            off += qs_nb
            off = align_up(off, 64)
            sc_off = off
            off += sc_nb
            entries.append((pl["a"], ne0, ne1, qs_off, sc_off, F_IL_A, pl, False, qs_nb, sc_nb))
            entries.append((pl["b"], ne0, ne1, qs_off, sc_off, F_IL_B, pl, True, qs_nb, sc_nb))
            total += qs_nb + sc_nb
        else:
            qs_nb, sc_nb = nblk * 16, nblk * 2
            off = align_up(off, 64)
            qs_off = off
            off += qs_nb
            off = align_up(off, 64)
            sc_off = off
            off += sc_nb
            nm = pl["name"] if pl["kind"] == "solo" else pl["a"]
            entries.append((nm, ne0, ne1, qs_off, sc_off, F_SOLO, pl, False, qs_nb, sc_nb))
            total += qs_nb + sc_nb

    print(f"payload ≈ {total / (1024**3):.2f} GiB  → {out}", flush=True)
    if args.dry_run:
        return 0

    header_size = 32
    dir_bytes = len(entries) * ENT_SIZE
    data_base = align_up(header_size + dir_bytes, 64)
    src_pages = (args.gguf.stat().st_size + 4095) // 4096
    tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<IIQ", ver, len(entries), data_base))
            f.write(struct.pack("<II", STAMP_TAG, src_pages))
            for ent in entries:
                name, ne0, ne1, a, b, flags = ent[:6]
                nb = name.encode("utf-8")
                if len(nb) >= 96:
                    raise ValueError(name)
                rec = bytearray(ENT_SIZE)
                rec[0 : len(nb)] = nb
                if ver == 2:
                    struct.pack_into("<iiQQII", rec, 96, ne0, ne1, a, b, 0, 0)
                else:
                    struct.pack_into("<iiQQII", rec, 96, ne0, ne1, a, b, flags, 0)
                f.write(rec)
            pad = data_base - f.tell()
            if pad:
                f.write(b"\0" * pad)

            for i, ent in enumerate(entries):
                name, ne0, ne1, a, b, flags, pl, _is_b, qs_nb, sc_nb = ent
                if ver == 2:
                    target = data_base + a
                    if f.tell() < target:
                        f.write(b"\0" * (target - f.tell()))
                    blob = pack_v2_tensor(pl["t"].data, ne0, ne1)
                    assert len(blob) == b
                    f.write(blob)
                elif flags == F_IL_B:
                    pass  # payload already written by IL_A
                else:
                    if flags == F_IL_A:
                        qs, sc = pack_v4_pair(pl["ta"].data, pl["tb"].data, ne0, ne1)
                    else:
                        t = pl["t"] if pl["kind"] == "solo" else pl["ta"]
                        qs, sc = pack_v3_tensor(t.data, ne0, ne1)
                    assert len(qs) == qs_nb and len(sc) == sc_nb
                    for blob, doff in ((qs, a), (sc, b)):
                        target = data_base + doff
                        if f.tell() < target:
                            f.write(b"\0" * (target - f.tell()))
                        f.write(blob)
                if (i + 1) % 25 == 0 or i + 1 == len(entries):
                    print(f"  {i + 1}/{len(entries)}  {name}", flush=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, out)
    except BaseException:
        # A failed repack must not strand a multi-GiB .cwenr.tmp next to the
        # real sidecar: disk-full, Ctrl-C, and bad payloads all land here.
        tmp.unlink(missing_ok=True)
        raise
    print(
        f"done {out}  {out.stat().st_size / (1024**3):.2f} GiB"
        f"  n={len(entries)} v{ver} pairs={npairs}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
