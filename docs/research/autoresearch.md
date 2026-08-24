# Autoresearch: Qwen3.8-27B CPU decode

| Field | Value |
|-------|-------|
| Started | 2026-08-16 |
| Target | `decode_tok_s` (higher is better) |
| Gate | deterministic argmax tokens; no golden reuse from Qwen3.6 |
| Environment | local `/home/maci/Desktop/cwen` |
| Host | AMD Ryzen 9 9950X (16c/32t, AVX-512 + VNNI) |
| Weights | `model/Qwen3.8-27B-Q4_0.gguf` (16056478688 B, GGUF v3, `qwen35`) |
| Max iterations | 100 (**reached**) |
| Bench | `AVX512=1 OMP=16 tools/autoresearch.sh` → `tools/bench_toks.py --ns 2,8 --trials 3` |

Hard rules: AGENTS.md fusion style (no `gemv2`/`gemv3`). Keep a change only if tok/s rises **and** the token chain is unchanged (or a documented MTP path that still verifies against the base argmax).

## Baseline

**2.635 tok/s** (I01, gcc 16.2.1 AVX512, OMP=16, 2026-08-16).

Tokens: `17 15 17 15 95859 17 15 17` (BOS 248044, argmax). Historical 3.6: ~2.73 tok/s with CWENR, tokens `2 653`.

## Iteration log (narrative)

### I00 / I01 baseline
Kept as reference. Cold first n=2 was 4.06s; median drops it. **2.635 tok/s**

### I02 clang-22 AVX512
**2.427 tok/s**. Revert. gcc stays default.

### I03 GDN AVX-512 16-wide + 2-row
**2.759 tok/s**. Keep. Tokens match.

### I04 mask-pad + Cephes silu512
**2.723 decode-only**, n=8 wall improved. Tokens match. Keep as quality-neutral SIMD hygiene. See padding note below.

### I05–I100 (continued)

Best honest keep remains **I06 2.792 tok/s** (GDN OpenMP over 48 heads). Tokens stayed `17 15 17 15 95859 17 15 17` except I64 FAST_SILU (fail).

Late-loop n=8 walls drifted from ~4.04s to ~4.6–4.7s (thermal/variance). Several 2.9–3.2 decode-only prints were **slow n=2**, not faster decode. Judge n=8 wall.

See `autoresearch.jsonl` for the full 100-iter log and `outputs/qwen38-autoresearch-handoff.md`.

## Padding (quality-neutral vs not)

Safe: zero extra SIMD lanes on **linear** ops (dot, residual, rms sum-of-squares, scale). Extra `0*x` does not change the result. Use mask-store so padded writes never land in live memory.

Not safe: pad a **nonlinear** lane (`exp`, `silu`, `softmax`, `softplus`) and then reduce / write it back as if it were real. `silu(0)=0` happens to be OK; `exp(0)=1` is not if you sum it.

This model: H=5120, I=17408, LSD=128, QKV=10240, V=248320 are all multiples of 16. Scalar tails were already dead code. The remaining scalar cost is `expf` in SiLU, which padding cannot remove.

## GGUF facts (this file, dumped 2026-08-16)

- `qwen35.block_count = 65` (64 text layers + MTP `blk.64`)
- tensors = 866
- types: F32×456, Q4_0×352, Q4_1×8, Q5_K×48, Q6_K×1 (`output.weight`), **Q8_0×1** (`blk.64.nextn.eh_proj.weight`)
- MTP tensors (ignored by current `L=64` bind):
  - `blk.64.attn_{q,k,v,output}` + q/k norms + SwiGLU (full-attn style)
  - `blk.64.nextn.eh_proj.weight` Q8_0 `[10240, 5120]`
  - `blk.64.nextn.{enorm,hnorm,shared_head_norm}.weight` F32 `[5120]`
- Engine binds layers `0..63` only. Extra tensors are parsed and left unused.
- Leftover `model/Qwen3.6-27B-Q4_0.cwenr` does **not** bind (name-matched sidecar).

## Paper / engine notes (2026-08-16)

### arXiv 2608.03893 — Cross-Model KV Cache Transfer

Heo et al., NVIDIA, 2026-08-04. https://arxiv.org/abs/2608.03893

Closed-form per-head ridge map so a **different-size family member** can skip re-prefill. Needs matched KV heads/dim, 500-seq calibration, 1–3B mapper params.

**Decision: defer.** This engine is single-model decode. The paper does not speed one-token Qwen3.8-27B decode. Revisit only if we add a 3.8-family cascade.

Useful adjacent ideas from the same paper: RoPE-stripped key space, attention-output cosine as a quality metric, cross-layer KV sharing citations (Brandon 2024 CLA). Cross-layer KV **inside** 3.8 would change numerics; not a free tok/s win.

### Colibri (JustVugg/colibri)

Pure-C MoE multitier (VRAM/RAM/NVMe). Takeaways that can land here:

- Measure end-to-end, never silently change precision
- One-layer-ahead prefetch as a **policy**, not a promise (already tried `PIPE_PF` / `MADVISE`; re-try only with a new shape)
- Compressed state / exact-forward gate
- MTP + grammar drafts: Colibri measured a **32% loss** when expert-hit is high and acceptance is poor. Same risk for our MTP: draft must beat one extra full layer.

### ds4 (antirez/ds4)

Narrow DeepSeek V4 / GLM 5.2 engine. Takeaways:

- Asymmetric quant: keep shared / router / head high precision, crush bulk experts
- Offline imatrix + mixed-layer splice
- SSD streaming + KV store (long context, not our 2048 decode)
- Custom Q8 / IQ2 CUDA paths; CPU dots still ggml-family

For us: Q4_0 already matches the bulk. `ssm_out` is Q5_K, `lm_head` Q6_K. Next lever is **Q8_0 MTP proj** + maybe requant `ffn_down` Q4_1 → Q4_0 (quality gate).

### TokenSpeed (lightseekorg/tokenspeed)

GPU agentic engine. Day-0 Qwen3.8 / Qwen3.5 configs. Scheduler + MLA kernels, not CPU GEMV. Confirms 3.8 is still `qwen3_5` hybrid + MTP. No CPU kernel to copy.

### C-Kernel-Engine GDN writeup

https://c-kernel-engine.github.io/C-Kernel-Engine/deltanet-deep-dive.html

GDN at `d=128` is **compute-bound**, SIMD-friendly. Their AVX-512 path: 16-wide FMA, 2-row unroll, `_mm512_reduce_add_ps`. Our `gdn_step` started AVX2 8-wide; the AVX-512 16-wide + 2-row path landed as I03 (kept).

They keep kernels single-thread; we OMP over the 48 heads (`CWEN_IDEA_GDN_OMP`, now default-on; I06 keep on 3.8).

## Idea backlog (try / skip)

| ID | Idea | Source | Expected | Status |
|----|------|--------|----------|--------|
| I01 | gcc AVX512 baseline, OMP=16 | current tree | lock number | queued |
| I02 | clang-22 `-march=native` AVX512 | compiler sweep | ±5% | queued |
| I03 | GDN AVX-512 16-wide + 2-row | C-Kernel-Engine | +3–8% if GDN is visible | queued |
| I04 | `CWEN_IDEA_GDN_OMP=1` | existing flag | maybe + if 48 heads starve | queued |
| I05 | 3.8 CWENR v4 sidecar | existing `repack_q4.py` | + if Q4_0 GGUF layout loses BW | queued |
| I06 | Q6_K lm_head wider unroll | OPTIMIZE.md 43–57 ms | +2–5% | queued |
| I07 | Q8_0 GEMV for `eh_proj` | GGUF type 8 | needed for MTP, not e2e alone | queued |
| I08 | MTP nextn draft + verify | GGUF blk.64, Colibri | + if accept>1.3 else lose | queued |
| I09 | KV cache Q8/Q4 | user request, KVQuant line | tiny at seq=1–8; maybe later @2048 | deferred short-seq |
| I10 | Cross-model KV map 2608.03893 | user paper | wrong target | skip |
| I11 | `-ffast-math` / `-Ofast` | flags | reject if tokens change | queued |
| I12 | gcc vs clang LTO / znver5 | flags | cheap | queued |
| I13 | Fuse GDN out-norm + silu(z) SIMD | local | small | queued |
| I14 | Prefetch next layer during GDN | existing `bg_stream` | already mixed; re-measure 3.8 | queued |
| I15 | Q4_1 `ffn_down` → Q4_0 requant | ds4-style | quality risk | later |
| I16 | INT8/VNNI activation quant | prior `VNNI=1` FAIL | only if we can pass goldens | later |
| I17 | Skip unused MTP pages (`MADV_DONTNEED` blk.64) | Colibri placement | maybe faster first token only | queued |
| I18 | Raise `MAX_SEQ` / shrink unused KV alloc | local | memory, not tok/s at n=8 | later |

## Iteration log

See `autoresearch.jsonl` and `CHANGELOG.md`. Narrative updates append below after each keep/revert.

## Handoff

`outputs/qwen38-autoresearch-handoff.md`
