#!/usr/bin/env python3
"""Seed corpus for tools/fuzz_loader.c (make fuzz-seeds).

Hand-built minimal GGUF, CWENR, DFlash .spec, and NGC2 ngram-cache headers
that pass the early magic/version checks so the mutator starts near the deep
parse loops instead of at byte 0, plus request-frame byte streams covering
every framing decision of the server protocol. Deterministic: no model files
needed.
"""

import argparse
import pathlib
import struct


def gguf(tensors=(), kv=()):
    out = bytearray(b"GGUF")
    out += struct.pack("<I", 3)  # version
    out += struct.pack("<Q", len(tensors))  # n_tensors
    out += struct.pack("<Q", len(kv))  # n_kv
    for key, typ, val in kv:
        kb = key.encode("utf-8")
        out += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", typ)
        if typ == 4:  # f32
            out += struct.pack("<f", val)
        elif typ == 8:  # string
            vb = str(val).encode("utf-8")
            out += struct.pack("<Q", len(vb)) + vb
        elif typ == 9:  # array: elem type + count + elems
            et, vals = val
            out += struct.pack("<I", et) + struct.pack("<Q", len(vals))
            for x in vals:
                out += struct.pack("<B", x) if et == 0 else struct.pack("<I", x)
        elif typ == 10:  # u64
            out += struct.pack("<Q", val)
        else:
            raise ValueError(typ)
    for name, dims, gtype, off in tensors:
        nb = name.encode("utf-8")
        out += struct.pack("<Q", len(nb)) + nb
        out += struct.pack("<I", len(dims))
        for d in dims:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", gtype) + struct.pack("<Q", off)
    return bytes(out)


def cwenr(ver, entries, data_base):
    out = bytearray(b"CWENR001")
    out += struct.pack("<II", ver, len(entries))
    out += struct.pack("<Q", data_base)
    for name, ne0, ne1, a, b, flags in entries:
        nb = name.encode("utf-8")
        if len(nb) >= 96:
            # same contract as tools/repack_q4.py: a silent byte-truncation
            # could split a UTF-8 sequence and yield a never-matching name.
            raise ValueError(name)
        e = bytearray(nb)
        e += bytes(96 - len(e))
        e += struct.pack("<iiQQ", ne0, ne1, a, b)
        e += struct.pack("<IQ", flags, 0)
        out += e
    return bytes(out)


def dfsp(entries):
    """DFlash .spec container: header + 96B entries + 64B-aligned payload.

    Entry: name[64], u32 ne0, u32 ne1, u64 type (0=F32 1=Q4_0R 2=Q8_0),
    u64 offset, u64 nbytes; offsets are absolute from file start."""
    header_end = 12 + len(entries) * 96
    payload_off = (header_end + 63) & ~63
    out = bytearray(b"DFSP")
    out += struct.pack("<II", 1, len(entries))
    off = payload_off
    body = bytearray()
    for name, ne0, ne1, typ, data in entries:
        nm = name.encode("utf-8")
        if len(nm) >= 64:
            raise ValueError(name)
        out += nm + bytes(64 - len(nm))
        out += struct.pack("<IIQQ", ne0, ne1, typ, off)
        out += struct.pack("<Q", len(data))
        body += data
        pad = (-len(data)) % 64
        body += bytes(pad)
        off += len(data) + pad
    assert len(out) == header_end
    return bytes(out) + bytes(payload_off - header_end) + bytes(body)


def frames(*words):
    """Request-frame stream: the wire is all little-endian u32 (frame headers
    <n_prompt><n_gen> and prompt token ids)."""
    return struct.pack(f"<{len(words)}I", *words)


EOF_SENTINEL = 0xFFFFFFFF
# Model geometry from run.c: token ids live in [0,V); server_read_frame
# validates both frame header words against the context window (g_ctx,
# CWEN_CTX, default 4096), not the MAX_SEQ enum ceiling.
V = 248320
G_CTX = 4096

NKEY = 16  # run.c Scfg_n_key default; ng_load refuses maps from other configs


def ngc2(entries, vocab=V, n_key=NKEY, used=None):
    """NGC2 ngram-cache file: "NGC2", u32 n_key, u32 vocab, u32 used, then
    records of {i32 key[n_key], i32 tok, i32 cnt}."""
    out = bytearray(b"NGC2")
    out += struct.pack("<III", n_key, vocab, len(entries) if used is None else used)
    for key, tok, cnt in entries:
        assert len(key) == n_key, "key must be exactly n_key ints"
        for t in key:
            out += struct.pack("<i", t)
        out += struct.pack("<ii", tok, cnt)
    return bytes(out)


def ngkey(*ints):
    return list(ints) + [0] * (NKEY - len(ints))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Deterministic hand-built GGUF/CWENR/DFSP format heads and "
            "request-frame streams for the fuzz mutator."
        )
    )
    ap.add_argument("--out", default="tools/fuzz_corpus", help="output directory for the corpus")
    args = ap.parse_args()
    outd = pathlib.Path(args.out)
    outd.mkdir(parents=True, exist_ok=True)

    seeds = {
        # bare magic + version only; rejection paths behind the magic check
        "seed_gguf_magic.gguf": b"GGUF" + struct.pack("<IQ", 3, 0),
        # header with one kv string and one plausible tensor entry
        "seed_gguf_kv_str.gguf": gguf(
            kv=[("general.name", 8, "cwen")],
            tensors=[("token_embd.weight", [1024, 64], 2, 0)],
        ),
        # nested array kv (skip_val recursion) plus two tensor entries
        "seed_gguf_nested_arr.gguf": gguf(
            kv=[("test.arr", 9, (0, [1, 2, 3]))],
            tensors=[
                ("blk.0.attn_norm.weight", [1024], 0, 64),
                ("output_norm.weight", [1024], 0, 8192),
            ],
        ),
        # truncated mid-tensor-table: every read must stay bounds-checked
        "seed_gguf_trunc.gguf": gguf(
            tensors=[("blk.0.ffn_gate.weight", [1024, 3584], 2, 0)],
        )[:-6],
        # CWENR v4 directory head (bind path needs a matching GGUF too)
        "seed_cwenr_v4.cwenr": cwenr(
            4,
            [("blk.1.attn_qkv.weight", 6144, 1024, 0, 2048, 1)],
            data_base=32 + 128,
        ),
        # DFlash .spec: empty container -> walks the post-loop geometry checks
        "seed_dfsp_empty.spec": dfsp([]),
        # one bindable layer tensor (F32, one QK4 row) + payload
        "seed_dfsp_one.spec": dfsp(
            [("layers.0.qn", 32, 1, 0, b"\x11" * 128)],
        ),
        # count declares two entries but only one is present: truncated header
        "seed_dfsp_trunc.spec": dfsp(
            [("layers.0.ln1", 5120, 1, 0, b"\x22" * 20480)],
        )[:-96],
        # two accepted frames (incl. the V-1 token boundary), EOF sentinel,
        # trailing bytes the parser must never touch after close
        "seed_frame_ok.stream": frames(3, 2, 1, 2, 3, 1, 1, V - 1, EOF_SENTINEL) + b"JUNK",
        # every rejection flavor, then a valid frame to prove stream alignment:
        # n_prompt=0 (zero-width drain), out-of-range token, n_gen>G_CTX
        # (4B drain), accept, sentinel
        "seed_frame_reject.stream": frames(
            0, 1, 2, 1, 5, 999_999_999, 1, G_CTX + 1, 0, 1, 1, 42, EOF_SENTINEL
        ),
        # EOF mid-payload: declared 4 tokens, only 2 present -> close, no reply
        "seed_frame_trunc_payload.stream": frames(4, 1, 7, 8),
        # oversized header declares a 4097*4 B drain but only 8 filler bytes
        # follow: truncated drain consumes to EOF and closes
        "seed_frame_trunc_drain.stream": frames(G_CTX + 1, 1, 0xDEAD, 0xBEEF),
        # regression unit (fuzzer-found): header words legal under the MAX_SEQ
        # enum but over the default context bound -> reject + drain to EOF;
        # pins the g_ctx (not MAX_SEQ) validation bound
        "seed_frame_ctx_reject.stream": frames(67, 11874),
        # largest legal frame: exactly G_CTX prompt tokens must be accepted so
        # the boundary itself stays pinned on both sides
        "seed_frame_ctx_accept.stream": frames(G_CTX, 1)
        + struct.pack(f"<{G_CTX}I", *([0] * G_CTX)),
        # NGC2 cache: distinct entries on the [0,V) token boundaries plus a
        # duplicate-key pair exercising the count-merge path (3+5 -> 8)
        "seed_ngc2_ok.ngc": ngc2(
            [
                (ngkey(11, 22, 33), 0, 3),
                (ngkey(44, 55, 66), V - 1, 1),
                (ngkey(11, 22, 33), V - 2, 5),
                (ngkey(77), 42, 2),
            ]
        ),
        # header declares three records but the tail is cut mid-record:
        # the loader must stop cleanly at the truncation
        "seed_ngc2_trunc.ngc": ngc2(
            [
                (ngkey(1, 2, 3), 10, 1),
                (ngkey(4, 5, 6), 11, 2),
                (ngkey(7, 8, 9), 12, 1),
            ]
        )[:-80],
        # wrong vocab in the header: config-mismatch rejection path
        "seed_ngc2_foreign.ngc": ngc2([(ngkey(9, 9, 9), 5, 1)], vocab=1024),
    }
    for name, blob in seeds.items():
        (outd / name).write_bytes(blob)
        print(f"{outd / name}  {len(blob)} B")


if __name__ == "__main__":
    main()
