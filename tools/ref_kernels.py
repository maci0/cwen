#!/usr/bin/env python3
"""Reference dequant + GEMV matching run.c / ggml row layout."""

from __future__ import annotations

import struct
from array import array
from typing import TYPE_CHECKING

from gguf_util import QK4, QK_K, SIZEOF, T_F32, T_Q4_0, T_Q4_1, T_Q5_K, T_Q6_K, Tensor

if TYPE_CHECKING:
    import numpy as np


def f16_to_f32(h: int) -> float:
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    if e == 0:
        if f == 0:
            return -0.0 if s else 0.0
        # match C: ldexpf((float)f, -24)
        x = f * (2.0**-24)
        return -x if s else x
    if e == 31:
        return float("nan") if f else (-float("inf") if s else float("inf"))
    # rebuild float32 bits
    u = (s << 31) | ((e - 15 + 127) << 23) | (f << 13)
    return struct.unpack("<f", struct.pack("<I", u))[0]


def get_scale_min_k4(j: int, q: bytes) -> tuple[int, int]:
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
    m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4)
    return d, m


def s8(v: int) -> int:
    """Signed int8 like the C (int8_t) cast."""
    v &= 0xFF
    return v - 256 if v >= 128 else v


def dequant_row_q4_0(row: memoryview, n: int) -> list[float]:
    assert n % QK4 == 0
    y = [0.0] * n
    nb = n // QK4
    for i in range(nb):
        base = i * 18
        d = f16_to_f32(struct.unpack_from("<H", row, base)[0])
        qs = row[base + 2 : base + 18]
        for j in range(QK4 // 2):
            x0 = (qs[j] & 0xF) - 8
            x1 = (qs[j] >> 4) - 8
            y[i * QK4 + j] = x0 * d
            y[i * QK4 + j + QK4 // 2] = x1 * d
    return y


def dequant_row_q4_1(row: memoryview, n: int) -> list[float]:
    assert n % QK4 == 0
    y = [0.0] * n
    for i in range(n // QK4):
        base = i * 20
        d = f16_to_f32(struct.unpack_from("<H", row, base)[0])
        m = f16_to_f32(struct.unpack_from("<H", row, base + 2)[0])
        qs = row[base + 4 : base + 20]
        for j in range(QK4 // 2):
            x0 = qs[j] & 0xF
            x1 = qs[j] >> 4
            y[i * QK4 + j] = x0 * d + m
            y[i * QK4 + j + QK4 // 2] = x1 * d + m
    return y


def dequant_row_q5_K(row: memoryview, n: int) -> list[float]:
    assert n % QK_K == 0
    y: list[float] = []
    bsz = 2 + 2 + 12 + QK_K // 8 + QK_K // 2
    for i in range(n // QK_K):
        base = i * bsz
        d = f16_to_f32(struct.unpack_from("<H", row, base)[0])
        dmin = f16_to_f32(struct.unpack_from("<H", row, base + 2)[0])
        scales = bytes(row[base + 4 : base + 16])
        qh = bytes(row[base + 16 : base + 16 + QK_K // 8])
        ql = bytes(row[base + 16 + QK_K // 8 : base + bsz])
        is_ = 0
        u1, u2 = 1, 2
        ql_off = 0
        for _j in range(0, QK_K, 64):
            sc, m = get_scale_min_k4(is_ + 0, scales)
            d1, m1 = d * sc, dmin * m
            sc, m = get_scale_min_k4(is_ + 1, scales)
            d2, m2 = d * sc, dmin * m
            for qi in range(32):
                y.append(d1 * ((ql[ql_off + qi] & 0xF) + (16 if (qh[qi] & u1) else 0)) - m1)
            for qi in range(32):
                y.append(d2 * ((ql[ql_off + qi] >> 4) + (16 if (qh[qi] & u2) else 0)) - m2)
            ql_off += 32
            is_ += 2
            u1 <<= 2
            u2 <<= 2
    return y


def dequant_row_q6_K(row: memoryview, n: int) -> list[float]:
    assert n % QK_K == 0
    y: list[float] = []
    bsz = QK_K // 2 + QK_K // 4 + QK_K // 16 + 2
    for i in range(n // QK_K):
        base = i * bsz
        ql = bytes(row[base : base + QK_K // 2])
        qh = bytes(row[base + QK_K // 2 : base + QK_K // 2 + QK_K // 4])
        sc = list(struct.unpack_from("<" + "b" * (QK_K // 16), row, base + QK_K // 2 + QK_K // 4))
        d = f16_to_f32(struct.unpack_from("<H", row, base + QK_K // 2 + QK_K // 4 + QK_K // 16)[0])
        ql_o = qh_o = sc_o = 0
        for _n0 in range(0, QK_K, 128):
            block = [0.0] * 128
            for qi in range(32):
                is_ = qi // 16
                q1 = s8((ql[ql_o + qi] & 0xF) | (((qh[qh_o + qi] >> 0) & 3) << 4)) - 32
                q2 = s8((ql[ql_o + qi + 32] & 0xF) | (((qh[qh_o + qi] >> 2) & 3) << 4)) - 32
                q3 = s8((ql[ql_o + qi] >> 4) | (((qh[qh_o + qi] >> 4) & 3) << 4)) - 32
                q4 = s8((ql[ql_o + qi + 32] >> 4) | (((qh[qh_o + qi] >> 6) & 3) << 4)) - 32
                block[qi] = d * sc[sc_o + is_ + 0] * q1
                block[qi + 32] = d * sc[sc_o + is_ + 2] * q2
                block[qi + 64] = d * sc[sc_o + is_ + 4] * q3
                block[qi + 96] = d * sc[sc_o + is_ + 6] * q4
            y.extend(block)
            ql_o += 64
            qh_o += 32
            sc_o += 8
    return y


def dequant_row(t: Tensor, i: int) -> list[float]:
    row = t.row_ptr(i)
    n = t.ne0
    if t.typ == T_F32:
        return list(struct.unpack_from("<" + "f" * n, row, 0))
    if t.typ == T_Q4_0:
        return dequant_row_q4_0(row, n)
    if t.typ == T_Q4_1:
        return dequant_row_q4_1(row, n)
    if t.typ == T_Q5_K:
        return dequant_row_q5_K(row, n)
    if t.typ == T_Q6_K:
        return dequant_row_q6_K(row, n)
    raise ValueError(t.typ)


def gemv_stream(t: Tensor, x: list[float]) -> array:
    y = gemv_np(t, x)
    out = array("f", [0.0]) * len(y)
    for i, v in enumerate(y):
        out[i] = float(v)
    return out


def gemv_np(t: Tensor, x: np.ndarray | list[float]) -> np.ndarray:
    """Fast path: dequant (streamed for huge Q6) then matvec."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    assert x.shape[0] == t.ne0
    if t.typ == T_Q6_K:
        return gemv_q6_k_batched(t, x)
    if t.typ == T_Q5_K:
        return gemv_q5_k_batched(t, x)
    W = dequant_matrix_np(t)
    return W @ x


def q6_k_block_bytes() -> int:
    return SIZEOF[T_Q6_K]  # 210


def gemv_q6_k_batched(t: Tensor, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Stream Q6_K rows in batches (full f32 lm_head is ~5 GiB)."""
    import numpy as np

    K, M = t.ne0, t.ne1
    bsz = q6_k_block_bytes()
    nb = K // QK_K
    raw = np.frombuffer(t.data, dtype=np.uint8).reshape(M, nb, bsz)
    y = np.empty(M, np.float32)
    for i0 in range(0, M, batch):
        i1 = min(M, i0 + batch)
        W = dequant_q6_rows(raw[i0:i1], K)
        y[i0:i1] = W @ x
    return y


def _s8m32(u16: np.ndarray) -> np.ndarray:
    import numpy as np

    u = u16.astype(np.int16)
    return np.where(u >= 128, u - 256, u).astype(np.float32) - 32.0


def dequant_q6_rows(raw: np.ndarray, K: int) -> np.ndarray:
    """raw: (B, nb, 210) uint8 → (B, K) float32. Vectorized over rows and lanes."""
    import numpy as np

    B, nb, bsz = raw.shape
    assert bsz == q6_k_block_bytes() and nb * QK_K == K
    W = np.empty((B, K), np.float32)
    ql = raw[:, :, 0:128]
    qh = raw[:, :, 128:192]
    sc = raw[:, :, 192:208].view(np.int8).astype(np.float32)  # (B, nb, 16)
    d = raw[:, :, 208:210].copy().view("<f2").astype(np.float32)[:, :, 0]  # (B, nb)

    for bi in range(nb):
        base = bi * QK_K
        d_b = d[:, bi : bi + 1]  # (B, 1)
        for sub in range(2):
            ql_s = ql[:, bi, sub * 64 : sub * 64 + 64]  # (B, 64)
            qh_s = qh[:, bi, sub * 32 : sub * 32 + 32].astype(np.uint16)  # (B, 32)
            sc_s = sc[:, bi, sub * 8 : sub * 8 + 8]  # (B, 8)
            # scales per lane l: is=l//16 → sc[:, is], sc[:, is+2], ...
            is_idx = np.arange(32) // 16  # (32,)
            sc0 = sc_s[:, is_idx]  # (B, 32)
            sc2 = sc_s[:, is_idx + 2]
            sc4 = sc_s[:, is_idx + 4]
            sc6 = sc_s[:, is_idx + 6]

            q1 = _s8m32((ql_s[:, :32] & 0xF) | ((qh_s & 3) << 4))
            q2 = _s8m32((ql_s[:, 32:] & 0xF) | (((qh_s >> 2) & 3) << 4))
            q3 = _s8m32((ql_s[:, :32] >> 4) | (((qh_s >> 4) & 3) << 4))
            q4 = _s8m32((ql_s[:, 32:] >> 4) | (((qh_s >> 6) & 3) << 4))

            off = base + sub * 128
            W[:, off : off + 32] = d_b * sc0 * q1
            W[:, off + 32 : off + 64] = d_b * sc2 * q2
            W[:, off + 64 : off + 96] = d_b * sc4 * q3
            W[:, off + 96 : off + 128] = d_b * sc6 * q4
    return W


def gemv_q5_k_batched(t: Tensor, x: np.ndarray, batch: int = 2048) -> np.ndarray:
    import numpy as np

    K, M = t.ne0, t.ne1
    y = np.empty(M, np.float32)
    for i0 in range(0, M, batch):
        i1 = min(M, i0 + batch)
        W = np.empty((i1 - i0, K), np.float32)
        for j, i in enumerate(range(i0, i1)):
            W[j] = dequant_row(t, i)
        y[i0:i1] = W @ x
    return y


def dequant_matrix_np(t: Tensor) -> np.ndarray:
    import numpy as np

    K, M = t.ne0, t.ne1
    if t.typ == T_F32:
        return np.frombuffer(t.data, dtype=np.float32).reshape(M, K).copy()
    if t.typ == T_Q4_0:
        nb = K // QK4
        raw = np.frombuffer(t.data, dtype=np.uint8).reshape(M, nb, 18)
        d = raw[:, :, 0:2].copy().view("<f2").astype(np.float32)[:, :, 0]
        qs = raw[:, :, 2:]
        lo = (qs & 0x0F).astype(np.int16) - 8
        hi = (qs >> 4).astype(np.int16) - 8
        W = np.empty((M, K), np.float32)
        for b in range(nb):
            W[:, b * 32 : b * 32 + 16] = lo[:, b, :] * d[:, b : b + 1]
            W[:, b * 32 + 16 : b * 32 + 32] = hi[:, b, :] * d[:, b : b + 1]
        return W
    if t.typ == T_Q4_1:
        nb = K // QK4
        raw = np.frombuffer(t.data, dtype=np.uint8).reshape(M, nb, 20)
        d = raw[:, :, 0:2].copy().view("<f2").astype(np.float32)[:, :, 0]
        m = raw[:, :, 2:4].copy().view("<f2").astype(np.float32)[:, :, 0]
        qs = raw[:, :, 4:]
        lo = (qs & 0x0F).astype(np.float32)
        hi = (qs >> 4).astype(np.float32)
        W = np.empty((M, K), np.float32)
        for b in range(nb):
            W[:, b * 32 : b * 32 + 16] = lo[:, b, :] * d[:, b : b + 1] + m[:, b : b + 1]
            W[:, b * 32 + 16 : b * 32 + 32] = hi[:, b, :] * d[:, b : b + 1] + m[:, b : b + 1]
        return W
    if t.typ == T_Q6_K:
        # only for small M tests — prefer gemv_q6_k_batched
        bsz = q6_k_block_bytes()
        nb = K // QK_K
        raw = np.frombuffer(t.data, dtype=np.uint8).reshape(M, nb, bsz)
        return dequant_q6_rows(raw, K)
    W = np.empty((M, K), np.float32)
    for i in range(M):
        W[i] = dequant_row(t, i)
    return W
