#!/usr/bin/env python3
"""Minimal GGUF mmap reader (metadata + tensor views)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# ggml type ids
T_F32, T_F16, T_Q4_0, T_Q4_1 = 0, 1, 2, 3
T_Q8_0 = 8
T_Q4_K, T_Q5_K, T_Q6_K = 12, 13, 14

# ggml type id -> display name (for dumps; ids beyond the constants above are
# listed so a listing tool never prints a bare number for a known quant)
TYPE_NAMES = {
    T_F32: "F32",
    T_F16: "F16",
    T_Q4_0: "Q4_0",
    T_Q4_1: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    T_Q8_0: "Q8_0",
    10: "Q2_K",
    11: "Q3_K",
    T_Q4_K: "Q4_K",
    T_Q5_K: "Q5_K",
    T_Q6_K: "Q6_K",
    30: "BF16",
}

QK4, QK_K = 32, 256
SIZEOF = {
    T_F32: 4,
    T_F16: 2,
    T_Q4_0: 18,  # per block of 32
    T_Q4_1: 20,
    T_Q8_0: 34,  # f16 + int8[32]
    T_Q5_K: 2 + 2 + 12 + QK_K // 8 + QK_K // 2,  # 176
    T_Q6_K: QK_K // 2 + QK_K // 4 + QK_K // 16 + 2,  # 210
}


def _row_nbytes(typ: int, ne0: int) -> int:
    if typ == T_F32:
        return ne0 * 4
    if typ == T_Q4_0:
        assert ne0 % QK4 == 0
        return (ne0 // QK4) * 18
    if typ == T_Q4_1:
        assert ne0 % QK4 == 0
        return (ne0 // QK4) * 20
    if typ == T_Q8_0:
        assert ne0 % QK4 == 0
        return (ne0 // QK4) * 34
    if typ == T_Q5_K:
        assert ne0 % QK_K == 0
        return (ne0 // QK_K) * SIZEOF[T_Q5_K]
    if typ == T_Q6_K:
        assert ne0 % QK_K == 0
        return (ne0 // QK_K) * SIZEOF[T_Q6_K]
    raise ValueError(f"unsupported type {typ}")


@dataclass
class Tensor:
    name: str
    typ: int
    ne: list[int]
    off: int  # byte offset from the start of the data section
    data: memoryview  # raw bytes of tensor body

    @property
    def ne0(self) -> int:
        return int(self.ne[0])

    @property
    def ne1(self) -> int:
        return int(self.ne[1]) if len(self.ne) > 1 else 1

    def row_bytes(self) -> int:
        return _row_nbytes(self.typ, self.ne0)

    def row_ptr(self, i: int) -> memoryview:
        rb = self.row_bytes()
        off = i * rb
        return self.data[off : off + rb]


def _read_str(mv: memoryview, off: int) -> tuple[str, int]:
    # Bounds-checked like run.c's loader ("truncated"/"str long"): slicing
    # clamps silently, so without these a cut-off file yields a short string
    # plus a bogus running offset instead of a named error. Strict UTF-8 is
    # per spec; the wrapper only adds location context to the failure.
    if off + 8 > len(mv):
        raise ValueError(f"truncated gguf string header at {off}")
    (n,) = struct.unpack_from("<Q", mv, off)
    off += 8
    if n > len(mv) - off:
        raise ValueError(f"truncated gguf string at {off}: {n}B past end of file")
    try:
        s = bytes(mv[off : off + n]).decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"gguf string at {off} is not valid UTF-8") from e
    return s, off + n


_VAL_FMT = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_val(mv: memoryview, off: int, t: int) -> tuple[Any, int]:
    """Decode one non-array KV value; arrays are skipped by the caller."""
    if t == 8:
        return _read_str(mv, off)
    fmt = _VAL_FMT.get(t)
    if fmt is None:
        raise ValueError(f"kv type {t}")
    (v,) = struct.unpack_from(fmt, mv, off)
    return v, off + struct.calcsize(fmt)


def _skip_val(mv: memoryview, off: int, t: int, depth: int = 0) -> int:
    # Same depth-64 cap as run.c's skip_val: hostile deep array nesting must
    # fail with a named error instead of exhausting the interpreter stack.
    if depth > 64:
        raise ValueError("kv nest too deep")
    if t in (0, 1, 7):
        return off + 1
    if t in (2, 3):
        return off + 2
    if t in (4, 5, 6):
        return off + 4
    if t in (10, 11, 12):
        return off + 8
    if t == 8:
        (n,) = struct.unpack_from("<Q", mv, off)
        return off + 8 + n
    if t == 9:
        (at,) = struct.unpack_from("<I", mv, off)
        (n,) = struct.unpack_from("<Q", mv, off + 4)
        off += 12
        for _ in range(n):
            off = _skip_val(mv, off, at, depth + 1)
        return off
    raise ValueError(f"kv type {t}")


@dataclass
class GGUF:
    path: Path
    version: int
    kv: dict[str, Any]
    tensors: dict[str, Tensor]
    _mm: Any

    @classmethod
    def open(cls, path: str | Path) -> GGUF:
        import mmap as mm_mod

        path = Path(path)
        # f must outlive the mmap (close() closes both); a with-block would
        # tear down the fd while the mapping is still live.
        f = open(path, "rb")  # noqa: SIM115
        mm_file = mm_mod.mmap(f.fileno(), 0, access=mm_mod.ACCESS_READ)
        # keep f open via attachment
        view = memoryview(mm_file)
        # Declared up front so the failure path below can drop every local
        # that exports the mmap: the active exception's traceback keeps this
        # frame alive through the handler, so only clearing here works.
        kv: dict[str, Any] = {}
        tensors: dict[str, Tensor] = {}
        try:
            if bytes(view[:4]) != GGUF_MAGIC:
                raise ValueError("not GGUF")
            off = 4
            (version,) = struct.unpack_from("<I", view, off)
            off += 4
            n_tensors, n_kv = struct.unpack_from("<QQ", view, off)
            off += 16
            for _ in range(n_kv):
                key, off = _read_str(view, off)
                (vt,) = struct.unpack_from("<I", view, off)
                off += 4
                # retain small scalar-ish values; skip huge tokenizer arrays by
                # size only so the mmap never materializes megabyte lists
                start = off
                if vt == 9:
                    off = _skip_val(view, off, vt)
                    kv[key] = f"<array skipped {off - start}B>"
                else:
                    val, off = _read_val(view, off, vt)
                    kv[key] = val

            infos: list[tuple[str, int, list[int], int]] = []
            for _ in range(n_tensors):
                name, off = _read_str(view, off)
                (n_dims,) = struct.unpack_from("<I", view, off)
                off += 4
                # Same n_dims <= 4 bound as run.c's loader: the format string
                # below is built from this count, so an unbounded u32 would
                # let a hostile file drive a multi-GiB allocation.
                if n_dims > 4:
                    raise ValueError(f"gguf tensor {name}: n_dims {n_dims} > 4")
                dims = list(struct.unpack_from("<" + "Q" * n_dims, view, off))
                off += 8 * n_dims
                (typ,) = struct.unpack_from("<I", view, off)
                off += 4
                (to,) = struct.unpack_from("<Q", view, off)
                off += 8
                infos.append((name, typ, dims, to))

            data_off = (off + 31) & ~31
            for name, typ, dims, to in infos:
                ne0 = int(dims[0])
                ne1 = int(dims[1]) if len(dims) > 1 else 1
                if typ == T_F32:
                    n = 1
                    for d in dims:
                        n *= int(d)
                    nbytes = n * 4
                elif len(dims) == 1:
                    nbytes = _row_nbytes(typ, ne0)
                else:
                    nbytes = _row_nbytes(typ, ne0) * ne1
                base = data_off + to
                tensors[name] = Tensor(
                    name=name, typ=typ, ne=list(dims), off=to, data=view[base : base + nbytes]
                )

            return cls(path=path, version=version, kv=kv, tensors=tensors, _mm=(f, mm_file))
        except BaseException as e:
            # A malformed file raises before the GGUF exists, so close() never
            # runs; release both handles here or every caller that catches
            # ValueError to report bad input strands an fd plus a mapping.
            # The in-flight traceback pins every frame it crossed, and the
            # _read/_skip helpers hold the mmap view as their `mv` argument,
            # so scrub those locals first or mm_file.close() below fails with
            # BufferError instead of letting the parse error propagate.
            tb = e.__traceback__
            while tb:
                loc = tb.tb_frame.f_locals
                # locals cannot be deleted from the frame proxy, only rebound
                loc["mv"] = None
                loc["view"] = None
                tb = tb.tb_next
            del view
            tensors.clear()  # Tensor.data views pin the mmap
            try:
                mm_file.close()
            except BufferError:
                import gc

                gc.collect()  # collect dropped views; a held view re-raises below
                mm_file.close()
            f.close()
            raise

    def close(self) -> None:
        f, mm_file = self._mm
        # Drop our tensor views; live numpy views exported from the mmap
        # would otherwise make mm_file.close() raise BufferError.
        empty = memoryview(b"")
        for t in self.tensors.values():
            t.data = empty
        try:
            mm_file.close()
        except BufferError:
            import gc

            gc.collect()  # collect dropped views; a held view re-raises below
            mm_file.close()
        f.close()
