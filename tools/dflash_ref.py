#!/usr/bin/env python3
"""Numpy reference for the DFlash2 drafter forward; diffs against C dumps.

Reads a CWEN_DF_DUMP directory (meta.bin, hout.bin) plus the drafter
safetensors, recomputes the drafter window in float32 numpy, and reports
max-abs deltas on the final normed hidden states per slot.

Usage: .venv/bin/python tools/dflash_ref.py <dumpdir> <safetensors>
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

EPS = 1e-6


def load_st(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            beg, end = meta["data_offsets"]
            f.seek(8 + n + beg)
            u16 = np.frombuffer(f.read(end - beg), dtype="<u2").astype("<u4")
            out[name] = (u16 << np.uint32(16)).view("<f4").reshape(meta["shape"])
    return out


def rms(x: np.ndarray, w: np.ndarray, eps: float = EPS) -> np.ndarray:
    ss = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ss + eps) * w


def rope_std(x: np.ndarray, pos: int, theta: float = 1e7) -> np.ndarray:
    """x [heads,128]; rotate-half pairing i<->i+64."""
    hd = x.shape[-1]
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float32) / hd)
    ang = np.float32(pos) * inv
    c, s = np.cos(ang), np.sin(ang)
    a, b = x[..., : hd // 2], x[..., hd // 2 :]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


def conv_window(x: np.ndarray, dyn: np.ndarray, base: np.ndarray) -> np.ndarray:
    """x [L,H]; dyn [L,k,g]; base [k,H]; zero pad left."""
    L, H = x.shape
    k = base.shape[0]
    g = dyn.shape[-1]
    gs = H // g
    xb = x.reshape(L, g, gs)
    out = np.zeros_like(xb)
    for t in range(k):
        xs = np.zeros_like(xb)
        if t < L:
            xs[t:] = xb[: L - t]
        out += (base[t].reshape(1, g, gs) + dyn[:, t][:, :, None]) * xs
    return out.reshape(L, H)


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w.T


def main() -> int:
    dump = Path(sys.argv[1])
    st = load_st(Path(sys.argv[2]))
    raw = (dump / "meta.bin").read_bytes()
    P, bs, E, H = struct.unpack("<4i", raw[:16])
    meta = np.frombuffer(raw[16:], dtype="<f4")
    ctx = meta[: P * H].reshape(P, H)
    off = P * H
    x0 = meta[off + bs : off + bs + bs * H].reshape(bs, H)
    del meta

    hout_ref = np.fromfile(dump / "hout.bin", dtype="<f4").reshape(E, H)

    win = 2048

    def taps_of(name: str, li: int) -> np.ndarray:
        return st[f"layers.{li}.{name}"]

    x = x0.copy()
    for li in range(5):
        p = f"layers.{li}."
        resid = x
        ln = rms(x, st[p + "input_layernorm.weight"])
        dyn = linear(ln, st[p + "attention_conv.kernel_projection.weight"])
        dyn = dyn.reshape(ln.shape[0], 2, 2, 320)
        base = st[p + "attention_conv.base_kernel"]
        cin = conv_window(ln, dyn[:, 0], base[0])
        L = cin.shape[0]
        q = linear(cin, st[p + "self_attn.q_proj.weight"]).reshape(L, 32, 128)
        kc = linear(ctx, st[p + "self_attn.k_proj.weight"]).reshape(P, 8, 128)
        kw = linear(cin, st[p + "self_attn.k_proj.weight"]).reshape(L, 8, 128)
        vc = linear(ctx, st[p + "self_attn.v_proj.weight"]).reshape(P, 8, 128)
        vw = linear(cin, st[p + "self_attn.v_proj.weight"]).reshape(L, 8, 128)
        q = rms(q, st[p + "self_attn.q_norm.weight"])
        k_all = rms(np.concatenate([kc, kw], axis=0), st[p + "self_attn.k_norm.weight"])
        v_all = np.concatenate([vc, vw], axis=0)
        for i in range(L):
            q[i] = rope_std(q[i], P + i)
        for j in range(P + L):
            k_all[j] = rope_std(k_all[j], j)
        ao = np.zeros((L, 32, 128), dtype=np.float32)
        for i in range(L):
            qpos = P + i
            lo = max(0, qpos - (win - 1))
            for h in range(32):
                hk = h // 4
                ks = k_all[lo : P + L, hk]
                sc = ks @ q[i, h] * (128**-0.5)
                sc = np.exp(sc - sc.max())
                sc /= sc.sum()
                ao[i, h] = sc @ v_all[lo : P + L, hk]
        ao = linear(ao.reshape(L, 4096), st[p + "self_attn.o_proj.weight"])
        cout = conv_window(ao, dyn[:, 1], base[1])
        x = resid + cout
        resid = x
        ln2 = rms(x, st[p + "post_attention_layernorm.weight"])
        dynm = linear(ln2, st[p + "mlp_conv.kernel_projection.weight"])
        dynm = dynm.reshape(ln2.shape[0], 2, 2, 320)
        mbase = st[p + "mlp_conv.base_kernel"]
        min_ = conv_window(ln2, dynm[:, 0], mbase[0])
        gate = linear(min_, st[p + "mlp.gate_proj.weight"])
        up = linear(min_, st[p + "mlp.up_proj.weight"])
        act = gate * (1.0 / (1.0 + np.exp(-gate))) * up
        down = linear(act, st[p + "mlp.down_proj.weight"])
        cout2 = conv_window(down, dynm[:, 1], mbase[1])
        x = resid + cout2
        print(f"layer {li} done", flush=True)

    hn = rms(x, st["norm.weight"])[1:]  # slots 1..bs-1
    d = np.abs(hn - hout_ref)
    print(f"hout max_abs={d.max():.6g} mean={d.mean():.3g}")
    for s in range(E):
        print(f"  slot {s + 1}: max={d[s].max():.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
