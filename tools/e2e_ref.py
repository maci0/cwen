#!/usr/bin/env python3
"""
Independent e2e reference matching run.c for one token, first N layers.
Writes golden/e2e/{embed,layer00,...,logits}.bin float32 vectors.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

from gguf_util import GGUF
from mk_prompt_ids import BOS
from ref_kernels import dequant_row, gemv_np

H, INTER, L = 5120, 17408, 64
NH, NKV, HD, NROT = 24, 4, 256, 64
LKH, LVH, LSD, CONV_K, QKV_DIM = 16, 48, 128, 4, 10240
FULL_INT = 4
RMS_EPS = 1e-6
ROPE_THETA = 1e7
ROPE_SEC = (11, 11, 10)


def rmsnorm(x, w):
    ss = float(np.dot(x, x))
    s = 1.0 / math.sqrt(ss / len(x) + RMS_EPS)
    return w * x * s


def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def softplus(x):
    # exp only on the x<=20 branch so large x cannot overflow to inf
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def embed_token(g, token: int) -> np.ndarray:
    t = g.tensors["token_embd.weight"]
    # row = token, length H, Q4_0
    return np.asarray(dequant_row(t, token), dtype=np.float32)


def rope_apply(q: np.ndarray, n_heads: int, pos: int) -> None:
    """In-place RoPE on [n_heads, HD], first NROT dims, section pairs."""
    for h in range(n_heads):
        v = q[h]
        idx = 0
        for n in ROPE_SEC:
            for _p in range(n):
                freq = 1.0 / (ROPE_THETA ** (idx / NROT))
                ang = pos * freq
                c, s = math.cos(ang), math.sin(ang)
                i0, i1 = idx * 2, idx * 2 + 1
                if i1 < NROT:
                    a, b = float(v[i0]), float(v[i1])
                    v[i0] = a * c - b * s
                    v[i1] = a * s + b * c
                idx += 1


def conv1d_update(mixed: np.ndarray, cs: np.ndarray, kw: np.ndarray) -> np.ndarray:
    """kw shape (QKV_DIM, CONV_K) from weight [4, 10240] stored row-major ne0=4."""
    out = np.empty(QKV_DIM, np.float32)
    for c in range(QKV_DIM):
        acc = 0.0
        for k in range(CONV_K - 1):
            acc += cs[c, k] * kw[c, k]
        acc += mixed[c] * kw[c, CONV_K - 1]
        out[c] = acc
        cs[c, :-1] = cs[c, 1:]
        cs[c, -1] = mixed[c]
    return silu(out)


def gdn_step(qkv, z, ab, bb, S, wn) -> np.ndarray:
    q0 = qkv[: LKH * LSD].reshape(LKH, LSD).copy()
    k0 = qkv[LKH * LSD : 2 * LKH * LSD].reshape(LKH, LSD).copy()
    v0 = qkv[2 * LKH * LSD :].reshape(LVH, LSD)
    # l2norm rows
    for r in range(LKH):
        q0[r] /= math.sqrt(float(np.dot(q0[r], q0[r])) + RMS_EPS)
        k0[r] /= math.sqrt(float(np.dot(k0[r], k0[r])) + RMS_EPS)
    scale = 1.0 / math.sqrt(LSD)
    out = np.empty((LVH, LSD), np.float32)
    for hv in range(LVH):
        hk = hv % LKH
        q, k, v = q0[hk], k0[hk], v0[hv]
        Sh = S[hv]
        g = math.exp(float(ab[hv]))
        beta = float(bb[hv])
        Sh *= g
        # C computes sk[j] = sum_i S[i*LSD+j]*k[i]; with row-major S that is k @ S
        sk = k @ Sh
        delta = beta * (v - sk)
        # S[i,j] += k[i]*delta[j]
        Sh += np.outer(k, delta)
        out[hv] = (q @ Sh) * scale
    # gated rmsnorm
    for hv in range(LVH):
        o = out[hv]
        ss = float(np.dot(o, o))
        s = 1.0 / math.sqrt(ss / LSD + RMS_EPS)
        zg = silu(z[hv])
        out[hv] = wn * o * s * zg
    return out.reshape(-1)


def mlp(g, layer: int, xb: np.ndarray) -> np.ndarray:
    p = f"blk.{layer}."
    gate = gemv_np(g.tensors[p + "ffn_gate.weight"], xb)
    up = gemv_np(g.tensors[p + "ffn_up.weight"], xb)
    h = silu(gate) * up
    return gemv_np(g.tensors[p + "ffn_down.weight"], h)


def layer_linear(g, layer: int, x: np.ndarray, S, cs) -> np.ndarray:
    p = f"blk.{layer}."
    wn = np.frombuffer(g.tensors[p + "attn_norm.weight"].data, dtype=np.float32).copy()
    xb = rmsnorm(x, wn)
    qkv = gemv_np(g.tensors[p + "attn_qkv.weight"], xb)
    z = gemv_np(g.tensors[p + "attn_gate.weight"], xb).reshape(LVH, LSD)
    alpha = gemv_np(g.tensors[p + "ssm_alpha.weight"], xb)
    beta = gemv_np(g.tensors[p + "ssm_beta.weight"], xb)
    dt = np.frombuffer(g.tensors[p + "ssm_dt.bias"].data, dtype=np.float32).copy()
    A = np.frombuffer(g.tensors[p + "ssm_a"].data, dtype=np.float32).copy()
    ab = A * softplus(alpha + dt)
    bb = sigmoid(beta)
    kw_t = g.tensors[p + "ssm_conv1d.weight"]
    # [4, 10240] F32 → (10240, 4)
    kw = np.frombuffer(kw_t.data, dtype=np.float32).reshape(QKV_DIM, CONV_K)
    qkv = conv1d_update(qkv, cs[layer], kw)
    sn = np.frombuffer(g.tensors[p + "ssm_norm.weight"].data, dtype=np.float32).copy()
    core = gdn_step(qkv, z, ab, bb, S[layer], sn)
    xb2 = gemv_np(g.tensors[p + "ssm_out.weight"], core)
    x = x + xb2
    wn2 = np.frombuffer(g.tensors[p + "post_attention_norm.weight"].data, dtype=np.float32).copy()
    xb = rmsnorm(x, wn2)
    x = x + mlp(g, layer, xb)
    return x


def layer_full(g, layer: int, x: np.ndarray, Kc, Vc, pos: int) -> np.ndarray:
    p = f"blk.{layer}."
    wn = np.frombuffer(g.tensors[p + "attn_norm.weight"].data, dtype=np.float32).copy()
    xb = rmsnorm(x, wn)
    qfull = gemv_np(g.tensors[p + "attn_q.weight"], xb).reshape(NH, 2, HD)
    q = qfull[:, 0, :].copy()
    gate = qfull[:, 1, :].copy()
    k = gemv_np(g.tensors[p + "attn_k.weight"], xb).reshape(NKV, HD)
    v = gemv_np(g.tensors[p + "attn_v.weight"], xb).reshape(NKV, HD)
    qn = np.frombuffer(g.tensors[p + "attn_q_norm.weight"].data, dtype=np.float32).copy()
    kn = np.frombuffer(g.tensors[p + "attn_k_norm.weight"].data, dtype=np.float32).copy()
    for h in range(NH):
        s = 1.0 / math.sqrt(float(np.dot(q[h], q[h])) / HD + RMS_EPS)
        q[h] = q[h] * s * qn
    for h in range(NKV):
        s = 1.0 / math.sqrt(float(np.dot(k[h], k[h])) / HD + RMS_EPS)
        k[h] = k[h] * s * kn
    rope_apply(q, NH, pos)
    rope_apply(k, NKV, pos)
    Kc[layer, pos] = k
    Vc[layer, pos] = v
    scale = 1.0 / math.sqrt(HD)
    kv_mul = NH // NKV
    yatt = np.zeros((NH, HD), np.float32)
    for h in range(NH):
        hkv = h // kv_mul
        scores = np.array(
            [float(np.dot(q[h], Kc[layer, t, hkv])) * scale for t in range(pos + 1)],
            np.float32,
        )
        scores = np.exp(scores - scores.max())
        scores /= scores.sum()
        o = np.zeros(HD, np.float32)
        for t in range(pos + 1):
            o += scores[t] * Vc[layer, t, hkv]
        yatt[h] = o * sigmoid(gate[h])
    xb2 = gemv_np(g.tensors[p + "attn_output.weight"], yatt.reshape(-1))
    x = x + xb2
    wn2 = np.frombuffer(g.tensors[p + "post_attention_norm.weight"].data, dtype=np.float32).copy()
    xb = rmsnorm(x, wn2)
    x = x + mlp(g, layer, xb)
    return x


def write_f32(path: Path, v: np.ndarray) -> None:
    path.write_bytes(np.asarray(v, dtype=np.float32).tobytes())


def forward_layers(g, x, S, cs, Kc, Vc, pos: int, nL: int, dump_prefix: Path | None):
    """Run layers 0..nL-1; optionally dump residual after each layer."""
    for layer in range(nL):
        is_lin = ((layer + 1) % FULL_INT) != 0
        x = layer_linear(g, layer, x, S, cs) if is_lin else layer_full(g, layer, x, Kc, Vc, pos)
        if dump_prefix is not None:
            write_f32(dump_prefix / f"layer{layer:02d}.bin", x)
        if layer % 8 == 0 or layer + 1 == nL:
            print(f"  layer {layer} x[0]={x[0]:.6g} mean={x.mean():.6g}", flush=True)
    return x


def logits_from_x(g, x: np.ndarray) -> np.ndarray:
    wn = np.frombuffer(g.tensors["output_norm.weight"].data, dtype=np.float32).copy()
    xb = rmsnorm(x, wn)
    return gemv_np(g.tensors["output.weight"], xb)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Independent e2e reference matching run.c: writes float32 "
        "residual/logit dumps for compare_e2e*.py."
    )
    ap.add_argument("--model", default="model/Qwen3.8-27B-Q4_0.gguf", help="GGUF model path")
    ap.add_argument("--token", type=int, default=BOS, help="first prompt token")
    ap.add_argument("--layers", type=int, default=4, help="run first N layers (0=all 64)")
    ap.add_argument("--out", default="golden/e2e", help="output directory for the dumps")
    ap.add_argument("--logits", action="store_true", help="compute full lm_head")
    ap.add_argument("--gen", type=int, default=0, help="extra decode steps after prompt (argmax)")
    ap.add_argument("--prompt-ids", type=Path, default=None, help="binary int32 prompt file")
    args = ap.parse_args()
    if not Path(args.model).is_file():
        print(
            f"e2e_ref: missing {args.model} (fetch the pinned weights: tools/download.sh)",
            file=sys.stderr,
        )
        return 1
    if args.prompt_ids is not None and not args.prompt_ids.is_file():
        print(
            f"e2e_ref: missing {args.prompt_ids} (encode text with tools/tok.py)", file=sys.stderr
        )
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("open", args.model, flush=True)
    g = GGUF.open(args.model)
    nL = L if args.layers <= 0 else min(args.layers, L)

    if args.prompt_ids:
        raw = args.prompt_ids.read_bytes()
        prompt = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    else:
        prompt = [args.token]

    max_seq = max(8, len(prompt) + args.gen + 1)
    S = np.zeros((nL, LVH, LSD, LSD), np.float32)
    cs = np.zeros((nL, QKV_DIM, CONV_K - 1), np.float32)
    Kc = np.zeros((nL, max_seq, NKV, HD), np.float32)
    Vc = np.zeros((nL, max_seq, NKV, HD), np.float32)

    generated: list[int] = []
    all_tokens = list(prompt)
    pos = 0
    last_logits = None

    for ti, tok in enumerate(prompt):
        print(f"prefill token[{ti}]={tok} layers={nL}", flush=True)
        x = embed_token(g, tok)
        if ti == 0:
            write_f32(out / "embed.bin", x)
            print("embed", float(x[0]), float(x.mean()), flush=True)
        dump = out if (ti == 0) else None  # dump layer residual only for first token
        x = forward_layers(g, x, S, cs, Kc, Vc, pos, nL, dump)
        if nL == L or args.logits or args.gen > 0:
            last_logits = logits_from_x(g, x)
            if ti == 0:
                write_f32(out / "logits_prefill.bin", last_logits)
        if ti + 1 < len(prompt):
            pos += 1

    # decode steps (requires full stack for meaningful match to C)
    for gi in range(args.gen):
        if last_logits is None:
            last_logits = logits_from_x(g, x)
        nxt = int(np.argmax(last_logits))
        generated.append(nxt)
        all_tokens.append(nxt)
        print(f"gen[{gi}] -> {nxt}", flush=True)
        pos += 1
        x = embed_token(g, nxt)
        x = forward_layers(g, x, S, cs, Kc, Vc, pos, nL, None)
        last_logits = logits_from_x(g, x)
        write_f32(out / f"logits_gen{gi:02d}.bin", last_logits)

    meta = {
        "prompt": prompt,
        "generated": generated,
        "all_tokens": all_tokens,
        "layers": nL,
        "note": "independent numpy ref matching run.c algorithms",
    }
    if last_logits is not None:
        top = np.argpartition(-last_logits, 8)[:8]
        top = top[np.argsort(-last_logits[top])]
        meta["top8"] = [int(i) for i in top]
        meta["top8_vals"] = [float(last_logits[i]) for i in top]
        print("top8", meta["top8"], flush=True)
        if args.logits or nL == L:
            write_f32(out / "logits.bin", last_logits)

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
