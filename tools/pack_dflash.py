#!/usr/bin/env python3
"""Pack the DFlash2 drafter safetensors into cwen's .spec container.

Container type ids mirror run.c kernels (0=F32, 1=Q4_0R, 2=Q8_0). This packer
emits Q8_0 for every matmul tensor (attention/MLP projections, selector
codebooks, conv kernel projections) and F32 for norms and conv base kernels;
Q8_0 is required in practice because 4-bit drafter weights lose argmax
agreement with the Q4_0 target (acceptance drops to zero). Quantization
formulas follow ggml exactly so the C dequant paths reproduce training
numerics within quant error:

  q4_0: per 32 block, d = signed max(x) / -8, q = round(x/d)+8 clamped to
        [0,15], low nibble first in each byte pair. Supported by the format;
        not emitted by this packer.
  q8_0: per 32 block, d = amax / 127, q = round(x/d) as int8.

Usage:
  .venv/bin/python tools/pack_dflash.py [SRC_SAFETENSORS] [-o model/dflash2.spec]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

DRAFTER_REPO = "models--incoai--Qwen3.8-27B-DFlash2"
DRAFTER_SNAPSHOT = "dedf8df68adfb1afeaf7b7480c0a0243108177b4"


def drafter_safetensors() -> Path:
    """Default drafter checkpoint in the HF cache, honouring HF_HOME."""
    cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface")
    return cache / "hub" / DRAFTER_REPO / "snapshots" / DRAFTER_SNAPSHOT / "model.safetensors"


MAGIC = b"DFSP"
VERSION = 1
T_F32, T_Q4R, T_Q8, T_Q8S, T_Q8SI = 0, 1, 2, 3, 4
# Q8S: split stream [int8 weights][f16 scales per 32-block row] - same values
# as Q8 blocks, arranged for contiguous SIMD loads (CWENR philosophy).
# Q8SI: two matrices sharing an input, physical rows alternating A,B; one
# dual pass emits both outputs per weight stream.


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """BF16 safetensors -> dict of float32 arrays shaped as stored."""
    out = {}
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if meta["dtype"] != "BF16":
                raise SystemExit(f"{name}: unsupported dtype {meta['dtype']}")
            beg, end = meta["data_offsets"]
            f.seek(base + beg)
            u16 = np.frombuffer(f.read(end - beg), dtype="<u2").astype("<u4")
            f32 = (u16 << np.uint32(16)).view("<f4")
            out[name] = f32.reshape(meta["shape"])
    return out


def pack_q4r(x: np.ndarray) -> bytes:
    """Row-major [ne1, ne0] float32 -> Q4_0R blocks (qs[16] then f32 d)."""
    ne1, ne0 = x.shape
    assert ne0 % 32 == 0, ne0
    xb = x.reshape(ne1, ne0 // 32, 32).astype(np.float32)
    # ggml q4_0: the scale uses the signed value of largest magnitude,
    # d = signed_max / -8, so dequant (q-8)*d reproduces the signs.
    signed_max = np.where(
        np.abs(xb.max(axis=2, keepdims=True)) >= np.abs(xb.min(axis=2, keepdims=True)),
        xb.max(axis=2, keepdims=True),
        xb.min(axis=2, keepdims=True),
    )
    d = signed_max / np.float32(-8.0)
    inv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    q = np.clip(np.rint(xb * inv + np.float32(8.5)), 0, 15).astype(np.uint8)
    lo = q[:, :, 0::2]
    hi = q[:, :, 1::2]
    qs = lo | (hi << np.uint8(4))  # [ne1, nb, 16]
    nb = ne0 // 32
    out = np.empty((ne1, nb, 20), dtype=np.uint8)
    out[:, :, :16] = qs
    out[:, :, 16:] = d.astype("<f4").view(np.uint8)
    return out.tobytes()


def pack_q8(x: np.ndarray) -> bytes:
    ne1, ne0 = x.shape
    assert ne0 % 32 == 0, ne0
    xb = x.reshape(ne1, ne0 // 32, 32).astype(np.float32)
    d = np.abs(xb).max(axis=2, keepdims=True) / np.float32(127.0)
    inv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    q = np.clip(np.rint(xb * inv), -127, 127).astype(np.int8)
    dh = d.astype("<f2").view(np.uint8)  # true f32->f16 convert, LE bytes
    out = np.empty((ne1, ne0 // 32, 34), dtype=np.uint8)
    out[:, :, :2] = dh
    out[:, :, 2:] = q
    return out.tobytes()


def pack_q8s(x: np.ndarray) -> bytes:
    """Split-stream Q8: [int8 weights row-major][f16 scales per 32-block]."""
    ne1, ne0 = x.shape
    nblk = ne0 // 32
    xb = x.reshape(ne1, nblk, 32).astype(np.float32)
    d = np.abs(xb).max(axis=2, keepdims=True) / np.float32(127.0)
    inv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
    q = np.clip(np.rint(xb * inv), -127, 127).astype(np.int8)
    sc = d.astype("<f2").view(np.uint8).reshape(ne1, nblk * 2)
    out = np.zeros((ne1, ne0 + nblk * 2), dtype=np.uint8)
    out[:, :ne0] = q.reshape(ne1, ne0)
    out[:, ne0:] = sc
    return out.tobytes()


def pack_q8si(a: np.ndarray, b: np.ndarray) -> bytes:
    """Pair-interleaved split Q8: physical rows alternate A,B, each a complete
    payload (int8 weights followed by its own f16 scales)."""
    na, ne0 = a.shape
    nb_, ne0b = b.shape
    assert na == nb_ and ne0 == ne0b and ne0 % 32 == 0
    nblk = ne0 // 32
    rb = ne0 + nblk * 2

    def rows(x: np.ndarray) -> np.ndarray:
        xb = x.reshape(na, nblk, 32).astype(np.float32)
        d = np.abs(xb).max(axis=2, keepdims=True) / np.float32(127.0)
        inv = np.where(d != 0, np.float32(1.0) / d, np.float32(0.0))
        q = np.clip(np.rint(xb * inv), -127, 127).astype(np.int8)
        sc = d.astype("<f2").view(np.uint8).reshape(na, nblk * 2)
        out = np.zeros((na, rb), dtype=np.uint8)
        out[:, :ne0] = q.reshape(na, ne0)
        out[:, ne0:] = sc
        return out

    ra, rbw = rows(a), rows(b)
    inter = np.zeros((na * 2, rb), dtype=np.uint8)
    inter[0::2] = ra
    inter[1::2] = rbw
    return inter.tobytes()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pack the DFlash2 drafter safetensors into a .spec container."
    )
    ap.add_argument(
        "src",
        nargs="?",
        default=str(drafter_safetensors()),
        help="drafter safetensors path",
    )
    ap.add_argument("-o", "--out", default="model/dflash2.spec", help="output .spec path")
    ap.add_argument(
        "--prec",
        choices=["q8", "mixed", "q4"],
        default="q8",
        help="drafter matmul precision. q8 recommended: q4 "
        "acceptance measured at zero on this model",
    )
    ap.add_argument(
        "--layout",
        choices=["blocks", "split"],
        default="blocks",
        help="split = CWENR-style streams (Q8S singles + Q8SI gate/up and "
        "k/v pairs); requires a run binary with split-Q8 support",
    )
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        print(
            f"pack_dflash: missing {src} "
            "(download incoai/Qwen3.8-27B-DFlash2, see README 'Trained DFlash2 drafter')",
            file=sys.stderr,
        )
        return 1
    tensors = load_safetensors(src)

    # precision presets: q8 recommended (q4 acceptance measured at zero),
    # mixed keeps attention projections exact while shrinking the MLPs
    big = {"q8": T_Q8, "mixed": T_Q8, "q4": T_Q4R}[args.prec]
    mlp = {"q8": T_Q8, "mixed": T_Q4R, "q4": T_Q4R}[args.prec]

    plan: list[tuple[str, int, int, int, np.ndarray]] = []

    def add(name: str, arr: np.ndarray, typ: int):
        a = np.ascontiguousarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        plan.append((name, a.shape[1], a.shape[0], typ, a))

    add("fc", tensors["fc.weight"], T_Q8S if args.layout == "split" else big)
    add("hidden_norm", tensors["hidden_norm.weight"].reshape(1, 5120), T_F32)
    add("norm", tensors["norm.weight"].reshape(1, 5120), T_F32)
    add("sel.pred", tensors["candidate_selector.predecessor_codebook"], T_Q8)
    add("sel.succ", tensors["candidate_selector.successor_codebook"], T_Q8)
    add("sel.hproj", tensors["candidate_selector.hidden_projection.weight"], T_Q8)
    for li in range(5):
        p = f"layers.{li}."
        q_typ = T_Q8S if args.layout == "split" else big
        add(p + "q_proj", tensors[p + "self_attn.q_proj.weight"], q_typ)
        if args.layout == "split":
            kv = np.concatenate(
                [tensors[p + "self_attn.k_proj.weight"], tensors[p + "self_attn.v_proj.weight"]]
            )
            add(p + "attn_kv", kv, T_Q8SI)
        else:
            add(p + "k_proj", tensors[p + "self_attn.k_proj.weight"], big)
            add(p + "v_proj", tensors[p + "self_attn.v_proj.weight"], big)
        o_typ = T_Q8S if args.layout == "split" else big
        add(p + "o_proj", tensors[p + "self_attn.o_proj.weight"], o_typ)
        if args.layout == "split":
            gu = np.concatenate(
                [tensors[p + "mlp.gate_proj.weight"], tensors[p + "mlp.up_proj.weight"]]
            )
            add(p + "mlp_gu", gu, T_Q8SI)
        else:
            add(p + "gate", tensors[p + "mlp.gate_proj.weight"], mlp)
            add(p + "up", tensors[p + "mlp.up_proj.weight"], mlp)
        dn_typ = T_Q8S if args.layout == "split" else mlp
        add(p + "down", tensors[p + "mlp.down_proj.weight"], dn_typ)
        add(p + "ln1", tensors[p + "input_layernorm.weight"].reshape(1, 5120), T_F32)
        add(p + "ln2", tensors[p + "post_attention_layernorm.weight"].reshape(1, 5120), T_F32)
        add(p + "qn", tensors[p + "self_attn.q_norm.weight"].reshape(1, 128), T_F32)
        add(p + "kn", tensors[p + "self_attn.k_norm.weight"].reshape(1, 128), T_F32)
        add(
            p + "attn_conv_base", tensors[p + "attention_conv.base_kernel"].reshape(1, 20480), T_F32
        )
        add(p + "mlp_conv_base", tensors[p + "mlp_conv.base_kernel"].reshape(1, 20480), T_F32)
        add(p + "attn_conv_proj", tensors[p + "attention_conv.kernel_projection.weight"], T_Q8)
        add(p + "mlp_conv_proj", tensors[p + "mlp_conv.kernel_projection.weight"], T_Q8)

    out_path = Path(args.out)
    header_end = 12 + len(plan) * 96  # magic+counts + 96B/tensor entry
    payload_off = (header_end + 63) & ~63  # 64B-align blob start
    entries = []
    blobs = []
    off = payload_off
    for name, ne0, ne1, typ, arr in plan:
        if typ == T_Q8SI:
            half = arr.shape[0] // 2
            data = pack_q8si(arr[:half], arr[half:])
        else:
            ser = {
                T_F32: lambda a: a.astype("<f4").tobytes(),
                T_Q4R: pack_q4r,
                T_Q8: pack_q8,
                T_Q8S: pack_q8s,
            }[typ]
            data = ser(arr)
        pad = (-len(data)) % 64
        data += b"\0" * pad
        entries.append((name, ne0, ne1, typ, off, len(data) - pad))
        blobs.append(data)
        off += len(data)

    with out_path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", VERSION, len(entries)))
        for name, ne0, ne1, typ, o, nb in entries:
            nm = name.encode("utf-8")
            if len(nm) > 63:
                # run.c reads a fixed 64B name field and strcmps it verbatim;
                # a silent byte-truncation could split a UTF-8 sequence and
                # yield an entry that can never match. Fail here instead.
                raise ValueError(name)
            nm = nm.ljust(64, b"\0")
            f.write(nm + struct.pack("<IIQQ", ne0, ne1, typ, o))
            f.write(struct.pack("<Q", nb))
        assert f.tell() == header_end, (f.tell(), header_end)
        f.write(b"\0" * (payload_off - header_end))
        for b in blobs:
            f.write(b)
    print(
        f"wrote {out_path}: {len(entries)} tensors, "
        f"{out_path.stat().st_size / 2**20:.1f} MiB payload"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
