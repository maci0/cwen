# cwen autonomous optimization process

Target: one kernel at a time. Correctness before speed. Zen 3+ = **AVX2 baseline**; Zen 4/5 = `make AVX512=1`.

## Current baselines (2026-07-14, Ryzen 9 9950X, **Qwen3.6 era**; do not read as 3.8 numbers. 3.8 loop results live in `CHANGELOG.md` / `autoresearch.md`)

**Protocol:** 5 independent process trials × 10 timed iters, report **median**.  
`tools/median_bench.py`. Build: `make AVX512=1 bench-q4_gemv`. **OMP=16** (one CCD; 24/32 thrash L3).

| Kernel | Prior (GA/24 thr) | **Now (SIMD+cache)** | Notes |
|--------|-------------------|----------------------|-------|
| Q4_0 `attn_gate` | 0.086–0.099 | **0.095** | dual-acc AVX512; BW-bound |
| Q4_1 `ffn_down` | 0.498–0.603 | **0.468** | AVX512 Q4_1 path |
| Q5_K `ssm_out` | 0.632–0.857 | **0.285** | **~2.2–3×** SIMD FMA |
| Q6_K `output.weight` | ~59 | **57** | partial SIMD unpack |

Goldens: `golden/blk_0_*`, `golden/output_weight`. All PASS. Tokens `2 653`.

**tok/s (decode-only, wall diff n=8 vs n=2):** **~2.73** (was ~2.3). Whole-process @ n=8 ~2.0 tok/s (includes mmap).

## GA + symbolic regression

```bash
make AVX512=1
make ga                    # or: .venv/bin/python tools/ga_evolve.py --gens 6 --pop 12 --avx512
# best genome -> cwen_tune.h + golden/ga_log/best.json
make AVX512=1 bench-q4_gemv
.venv/bin/python tools/median_bench.py --golden golden/blk_0_attn_gate_weight
```

**What evolves**

| Gene | Where | Notes |
|------|-------|-------|
| `CWEN_OMP_THRESH_EXPR(M,K)` | compile, symbolic tree | parallel when `M > expr` **and** block-work `M*(K/32) >= 4096`; ops `+ - * // >> max min` over `M,K,const` |
| `CWEN_PREFETCH` | compile | row prefetch distance in gemv |
| `CWEN_Q4_UNROLL` | compile | 1 or 2 block unroll (AVX2 path) |
| `CWEN_Q4_PF_BLOCKS` | compile | Q4 block prefetch (AVX-512 path) |
| `CWEN_OMP_THREADS` | compile default / env | precedence: `OMP_NUM_THREADS` > env `CWEN_OMP_THREADS` > baked default |

Fitness = weighted sum of ms/iter over Q4_0 + Q4_1 + Q5_K goldens (FAIL = huge penalty).

**Best genome (seed 42, 6 gens × pop 12)**

```
thr_expr = 64
threads  = 24
schedule = static,32
prefetch = 4
unroll   = 2
q4_pf    = 4
fitness  = 1.472  (baseline seed ~1.85)
```

Symbolic search mostly confirmed a **constant** threshold of 64 is near-optimal for these shapes; gains came from **thread count**, **static chunk 32**, and **prefetch=4**. Pathological thr expressions (`M`, `K`, `M*16`) correctly scored poorly (serial / wrong gate).

## Kernel contracts (freeze before editing)

| Kernel | Inputs (globals) | Output | Same-work microbench |
|--------|------------------|--------|----------------------|
| `gemv` / `dot_q4_0` | `x[K]`, W Q4_0, `M,K` | `y[M]` | `golden/blk_0_attn_gate_weight` |
| `gemv` / `dot_q4_1` | `x[K]`, W Q4_1 | `y[M]` | `golden/blk_0_ffn_down_weight` |
| `gemv` / `dot_q5_K` | `x[K]`, W Q5_K | `y[M]` | `golden/blk_0_ssm_out_weight` |
| `gemv` / `dot_q6_K` | `x[H]`, `output.weight` | logits | optional lm_head |
| `gdn_step` | q,k,v,g,β,S | o,S | layer0 linear |
| `rmsnorm` | `x[H]`, `w[H]` | `xb[H]` | H=5120 |

**Same work:** fixed shapes, fixed iteration count, warm cache.

## Loop

```bash
make AVX512=1 bench-q4_gemv
./bench_q4_gemv model/Qwen3.8-27B-Q4_0.gguf golden/blk_0_attn_gate_weight 20
tools/opt_loop.sh q4_gemv
tools/test_speed_gates.sh   # gemv PASS + deterministic tokens
make ga                     # re-evolve knobs after kernel code changes
```

`accept.py` / bench: max_abs primary (Q4 noise).

## Applied optimizations (this iteration)

1. **SIMD Q4_0 nibble unpack** + F16C scale + dual-block FMA (AVX2)
2. **AVX-512** 16-wide dequant+FMA; **4-acc + 2-block unroll** (hide FMA latency)
3. **AVX-512 Q4_1** path (was AVX2-only)
4. **Q5_K / Q6_K AVX2 FMA** (were pure scalar; Q5 ~2–3×)
5. **NTA prefetch** on weights (`__builtin_prefetch(...,0,0)`); T0 on activations
6. **64B-aligned** working vectors; `restrict` + `__attribute__((hot))`
7. **OpenMP static** over rows; default **16 threads** (one CCD; measured better than 24)
8. **MADV_HUGEPAGE** on mmap + load-time residency (`MAP_POPULATE` / `MADV_POPULATE_READ` / touch); GGUF warm defers when a CWENR sidecar owns Q4_0
9. **Compiler:** `-flto -fno-math-errno -fno-trapping-math -fomit-frame-pointer -mavx512vnni`
10. **GDN** column matvec tiled by 8; residual add SIMD; AVX2 argmax
11. **GA + symbolic thr** (`tools/ga_evolve.py` → `cwen_tune.h`)

## Idea A/B results (2026-07-14, tools/idea_bench.py)

Protocol: `make`-style AVX512 build, OMP=16, 3–5×10-iter medians on goldens; decode tok/s via wall(n=8)−wall(n=2).

| Idea | flag | golden | Q4_0 med | decode tok/s | Verdict |
|------|------|--------|----------|--------------|---------|
| baseline | (none) | PASS | 0.079–0.090 | **2.55** | keep |
| multirow 2 | `MULTIROW=2` | PASS | ~1.02× Q4 | n/a | no e2e win |
| multirow 4 | `MULTIROW=4` | PASS | ~1.07–1.12× Q4 micro | **2.41 worse** | **reject** (BW thrash) |
| Q8+int madd | `VNNI=1` | **FAIL** Q4 | 0.5× slower | n/a | **reject** |
| fast silu poly | `FAST_SILU=1` | gemv PASS | n/a | tokens **wrong** (`2 198 0 271`) | **reject** |
| madvise next layer | `MADVISE=1` | PASS | noise | ~2.50 | **reject** (no gain) |
| CCD pin `{0}:16:1` | `CCD=1` | PASS | ~same | **2.59** | **keep (default)** |
| weight prefetch T0 | `PF_T0=1` | PASS | slower | n/a | **reject** (NTA better) |
| no prefetch | `NO_PF=1` | PASS | slower | n/a | **reject** |
| copy act buffer | `COPY_X=1` | PASS | ~same | n/a | **reject** |

Re-run matrix:
```bash
.venv/bin/python tools/idea_bench.py --trials 5 --omp 16
.venv/bin/python tools/idea_bench.py --only ccd_pin,pair_gemv --toks
```

**Takeaway:** model is DRAM-bound. Sharing activations (multirow) helps the Q4 microbench slightly but **hurts e2e** (more concurrent weight streams). Integer Q8 path needs tighter scales to pass goldens. Poly silu breaks argmax chain. Only **CCD pinning** is a free, correct e2e win.

Thread scaling (Q4_0 gemv): 1→16 nearly linear; **24/32 slower** → pure DRAM BW + CCD.

## Offline weight repack (CWENR v2 → v4)

CPU layout is built **offline** (no at-load heap convert). Current: **v4** (`make repack` passes `--sidecar-version 4`).

```bash
make repack
# → model/Qwen3.8-27B-Q4_0.cwenr (sidecar next to the GGUF; rebuild after download)
# auto-mmap next to .gguf; override: CWEN_REPACK=/path/file.cwenr
```

| | GGUF Q4_0 | CWENR v2/v3 | **CWENR v4 (current)** | CWENR v1 32B pad (rejected) |
|--|-----------|-------------|------------------------|------------------------------|
| Block | 18 B `{f16 d, qs[16]}` | v2: packed 20 B `{qs[16], f32 d}`; v3: split qs + f16 scales | v3 solo split + **interleaved dual-mat pairs** (`ffn_gate`+`ffn_up`, `attn_k`+`attn_v`: per block `qsA|qsB`, `scA|scB`) | 32 B + pad |
| Hot path | f16 + loadu | split streams (`T_Q4_0RS`) | one sequential stream for dual-mat gemv (`T_Q4_0RSI`) | aligned but +78% bytes |
| File size | (in 15 GiB gguf) | ~13.6 GiB est. | **12.44 GiB** (2026-08-16 sidecar) | ~22 GiB, slower e2e |

Fusion note: the old standalone `gemv2(ffn_gate, ffn_up)` API is gone. Fusion now lives at the call sites per AGENTS.md: `gemv_pair(qkv, gate_z)` and the one-team `mlp` (unconditional; the two-team variant was measured worse and removed).

Tool: `tools/repack_q4.py`. Loader: `load_cwenr()` accepts `ver` 2/3/4 (v2 gets a runtime split; v4 flags mark interleaved A/B partners). No sidecar → plain GGUF.

## Flamegraph-driven opts (2026-07-14 night)

From `golden/profile/flames/` (Q4 dequant + GOMP wait + Q6):

| Change | Why (hardware) |
|--------|----------------|
| `OMP_WAIT_POLICY=passive` + `GOMP_SPINCOUNT=100` | Workers were spinning in libgomp while one core ran GDN/rmsnorm; free cores for turbo |
| Work-based OMP: `M*(K/32) >= 2048` | Amortize fork cost; tiny mats stay serial. Threshold later raised to **4096** (`gemv_use_omp`, "cut medium-mat forks"; GA section matches) |
| Q4_0R **4-block** unroll + pipelined loads | Zen5 dual FMA; hide L2 latency |
| Q6_K dual-acc, no stack temps | Was ~10% of samples spilling to stack |

| Metric | Before | After |
|--------|--------|-------|
| Q4_0 median ms | 0.095 | **0.084** |
| Q6_K median ms | ~57 | **~43** |
| decode tok/s | 2.31 | **2.42** |
| tokens | 2 653 | 2 653 |

Evidence: `/tmp/grok-goal-*/implementer/{baseline,after}.txt`.

## Low-level profile (perf + eBPF + Zen5 PMU)

See **`golden/profile/REPORT.md`**. Re-run: `tools/profile_lowlevel.sh 4`.

| Finding | Evidence |
|---------|----------|
| **~49%** cycles in Q4_0 `gemv` workers | `perf record` → `gemv._omp_fn.0` |
| **~31%** cycles in **libgomp barrier wait** | samples in `libgomp/.../wait.h` |
| **~12%** Q6_K, ~5% Q5_K | other `gemv._omp_fn.*` |
| IPC **1.33**, L1d miss **4.6%**, branch miss **0.1%** | `perf stat` |
| DRAM **~19 GB/s** (not peak); remote CCD ~0 | `ls_any_fills_from_sys.*` |
| **~500 `GOMP_parallel` / forward**, **1.55 s** in GOMP over 3-tok run | `bpftrace` uprobe |

**Profile-driven next target:** fewer OpenMP regions (layer/token-scoped parallel or persistent pool), not more Q4 FMA.

## DRAM / MC experiments (2026-07-14)

Outside-box ideas while DRAM sits under peak (~19–22 GB/s) and GDN is serial
(L3-hot S, workers passive). Protocol: idea_bench OMP=16, gemv goldens + decode
tok/s via wall(n=8)−wall(n=2). Evidence: `SCRATCH/dram_mc/`.

| Idea | flag | decode tok/s | Verdict |
|------|------|--------------|---------|
| baseline (pre) | n/a | **2.40–2.43** | |
| **bg_stream** | `BG_STREAM=1` | **2.46–2.53** | **keep as flag** (default off; pair gemv superseded it) |
| pipe_pf (page-walk l+1) | `PIPE_PF=1` | 2.41 | reject |
| bg+pipe | both | 2.43 | reject (pipe cancels) |
| MADV_POPULATE_READ | `POPULATE=1` | 2.32 | reject (load tax) |
| MADV_COLLAPSE | `COLLAPSE=1` | **1.86** | **reject hard** |
| full warm maps | `WARM=1` | 2.36 | reject |
| serial mlp (unfuse) | `SERIAL_MLP=1` | 2.41 | reject (fusion stays) |
| madvise WILLNEED l+1 | `MADVISE=1` | ~2.50 (prior) | reject (no win) |

**What works:** during `gdn_step`, every head touches 64 KiB of the *next* layer’s
head mat (qkv/wq) at 64 B stride (~3 MiB over LVH=48). MC was idle while S lived
in L3; streaming next weights overlaps DRAM with GDN compute. Tokens still
`2 653`. Prefer a lean `run.c` (no hard LOC cap).

Later note (2026-08): `POPULATE=1` / `WARM=1` **are** the shipped defaults in
`run.c` despite the rejects above (those were measured before the CWENR path
existed). Residency is paid once at load (`mmap_resident_ex`); on the CWENR path
the GGUF warm is deferred and rebound Q4 pages are dropped
(`warm_live_gguf_tensors` + `MADV_DONTNEED`).

**What does not:** load-time full populate/warm (hurts or no decode win);
THP collapse (catastrophic); unfusing mlp (two separate OMP teams worse);
extra page-walk at layer start (competes with current layer). Rank/bank/channel
mapping is not controllable from pure C without PA knowledge (BIOS/DIMM only).
Ceiling remains bytes×DRAM; bg_stream only improves *utilization* of idle MC.

Re-run:
```bash
.venv/bin/python tools/idea_bench.py --only baseline,pipe_pf,serial_mlp --toks --omp 16
```

## What not to do

- Full f32 dequant of the model
- OpenMP on tiny M (ssm_alpha etc.): thr expression must keep them serial
- Multi-row Q4 microkernel without re-bench (proved slower here)
- MADV_WILLNEED of entire 15 GiB each start (superseded: load-time residency is now the default, see the later note in "DRAM / MC experiments")
- MADV_COLLAPSE on multi-GB weight maps (measured multi-tok/s regression)
- Claiming >~6 tok/s without explaining timing bug
- Running `make verify` without `AVX512=1` if you want max-speed numbers (rebuilds AVX2-only)
