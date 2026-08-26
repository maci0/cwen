# cwen Handoff

## What this is

Pure-C inference engine for Qwen3.8-27B on CPU (AMD Zen 3+, AVX2 baseline /
AVX-512 optional). Single-file engine (`run.c`, ~4900 lines) with mmap Q4 GGUF
loading, speculative decoding (four drafter types), long-context support, and
a comprehensive measurement infrastructure.

Repo: https://github.com/maci0/cwen

## Current state (all lossless, CI green)

| Feature | Status | Key env / CLI |
|---|---|---|
| Serial decode | ✅ production | default |
| N-gram block speculation | ✅ shipped | `CWEN_SPEC=1` or `-d N` |
| Persistent n-gram cache | ✅ shipped | `CWEN_NGRAM_CACHE=FILE` |
| DFlash2 trained drafter | ✅ shipped | `CWEN_DFLASH=model/dflash2.spec` |
| **MTP nextn drafter** | ✅ **shipped** | auto-detected; `CWEN_MTP=0` opts out |
| Adaptive draft sizing | ✅ shipped (now AIMD) | automatic |
| Memory residency mode | ✅ shipped | `CWEN_RESIDENCY=1` |
| Long context (32k + YaRN) | ✅ shipped | `CWEN_CTX=N` + `CWEN_ROPE_YARN=o,f` |
| Runtime prefetch knobs | ✅ shipped | `CWEN_PF_T0` / `CWEN_NO_PF` / `CWEN_PIPE_PF` |
| Unpack-once batched GEMV | ✅ shipped, 1.6-2.1x | AVX-512, `2 ≤ B ≤ 8`, Q4_0RS + Q4_0RSI |

Drafter priority: `CWEN_DFLASH` > MTP nextn > n-gram map > n-gram scan.

## What was done this session

### PR18 MTP finished: acceptance 0 → 2.9-8.0 kept/cycle

The previous session left the nextn drafter structurally wired but at zero
acceptance. Root cause was not one bug but nine. The biggest: **`mtp_capture_hidden()`
was never called**, so every draft was conditioned on a zeroed hidden state.
Behind it sat a forward pass written against the *drafter's* geometry
(`DL_NH=32 / DL_HD=128`) instead of the target's (`NH=24 / NKV=4 / HD=256`),
`wk` and `wv` both writing the same buffer, a missing per-head q/gate split
(so the output gate never applied), the DFlash rotate-half RoPE table instead
of the target's sectioned `rope_apply`, V read out of `NKc` at a bogus offset,
`silu_mul` feeding the wrong buffer to `ffn_down`, and a double final norm.

Rewritten against `ref/qwen35.cpp`'s `graph_mtp`, which is the authoritative
reference for this layer. The model that made it work:

> Nextn stream slot *j* pairs token *t_j* with the target's final normed
> hidden *h_{j-1}* and predicts *t_{j+1}*. Slot 0 has no predecessor hidden,
> so the stream starts at slot 1 and attention runs over `[1..pos]`. RoPE is
> relative, so indexing by the input token's own position rather than the
> trained-time shift-by-one leaves scores unchanged.

Because slot *j*'s K/V depend only on `(t_j, h_{j-1})` and not on slot *j-1*'s
*output*, commits are order-independent per slot; only chained *drafting* is
sequential (each draft recurses on the nextn layer's own output).

Measured acceptance (`tools/spec_e2e.py --quick`, `-d 8`, all streams
byte-identical to plain decode):

| case | kept/cycle | full accepts |
|---|---|---|
| count | 8.00 | 3/3 |
| repeat | 7.33 | 2/3 |
| code | 5.00 | 3/4 |
| strawberry | 4.20 | 3/5 |
| prose | 1.43 | 4/7 |

Same suite, wall-clock, three resident engines rotating frames so every arm
samples the same load window (loadavg ~42, 24 generated tokens per frame, so
prefill dominates and these understate steady-state decode). Before column is
the same suite run at the start of this session, after the nextn forward was
correct but before the driver and kernel fixes:

| case | plain s | `-d 8` | `-d 3` | (before: `-d 8` / `-d 3`) |
|---|---|---|---|---|
| repeat | 69.6 | **1.15x** | **1.16x** | 0.90x / 0.85x |
| strawberry | 30.3 | **1.10x** | **1.37x** | 1.02x / 1.44x |
| code | 61.7 | **1.15x** | **1.26x** | 0.85x / 0.77x |
| count | 70.8 | **1.12x** | 1.08x | 1.09x / 0.86x |
| prose | 35.8 | 0.90x | 0.93x | 0.74x / 0.68x |

`prose` is the drafter-hostile case (1.43 kept/cycle): the rejection cost
dominates, exactly as DFlash2 does on drifting text.

### Two speculation-driver fixes that MTP exposed

1. **Short walks no longer cost k+1 full forwards.** A rejected tail restored
   the snapshot and re-ran the kept prefix through `forward_ex` once per
   token. It now re-scores that prefix with a single `forward_block(blk,k+1,0)`:
   weights stream once either way, and `ustar` already came from `Blogits`, so
   the lm_head sweeps are skipped too.
2. **Adaptive sizing is AIMD on full accepts, not acceptance rate.** The old
   rolling-rate rule rewarded exactly the draft lengths that short-walk every
   cycle (draft 6, keep 5 rates at 83% and grows E, while every cycle pays a
   rollback). It now drops to what the target took (`E_cap = max(min_draft, k)`)
   and probes upward only after four clean cycles.

Applied together, these took a repeat prompt at `-d 8` from 62.6s to 35.0s
(AVX2, same load window); neither number is attributable to one alone.

Also: `DL_BLOCK` (a DFlash2 walk limit) no longer caps the verify block for
other drafters; MTP can use the full `SPEC_BMAX-1`.

### The AVX-512 build did not compile

`dot_q8s` and `dot_q8si` carried half-finished 512-bit branches (`__m256`
scales into `_mm512_mul_ps`, an undeclared `acc`, 8-byte loads converted as if
16-wide). Both are Q8S/Q8SI split-container kernels, the layout already
measured *slower* than blocks, so the branches were deleted rather than
rewritten; AVX-512 builds use the AVX2 kernel there. `make AVX512=1` is clean
under `-Werror` again. (The previous CHANGELOG claims these were fixed on
2026-08-25; they were not.)

### Quiet-window guard

`tools/measure_when_quiet.sh -l LOAD -w SECONDS -- cmd ...` blocks until the
1-minute loadavg drops below the ceiling (default: half the core count), then
execs the command. Exits 75 if the window never opened, so a caller can tell
"never ran" from "ran and failed".

## Known issues / remaining work

### Unpack-once batched GEMV (`dot_q4_bcol`), done

`gemvb` called a per-column dot B times per row, redoing the nibble split, the
int8→f32 widen and the scale multiply each time; only the weight *load* was
shared. It also had no interleaved path at all: `T_Q4_0RSI` (164 of 352
tensors, including the two widest matrices in the model, `ffn_gate`/`ffn_up` at
17408x5120) fell through to the generic per-column `gemv_row` loop.

Both layouts now dequantize each 32-weight block once into two `__m512` and FMA
them against every column. One accumulator per column keeps register pressure
inside 32 zmm for `B ≤ 8`; wider blocks and non-AVX-512 builds keep the old
path. `-DCWEN_NO_BCOL` compiles it out, which is how the A/B below was run.

Min-of-R, alternating binaries, `bench_spec` at loadavg ~45, read the ratios,
not the absolutes. `B=1` is the control: it does not take the new path, and it
does not move.

| tensor | layout | B=1 | B=2 | B=4 | B=8 |
|---|---|---|---|---|---|
| `blk.0.attn_qkv` | Q4_0RS | 1.07x | 2.15x | 1.13x | **1.61x** |
| `blk.0.ffn_gate` | Q4_0RSI | 1.03x | 1.22x | 1.75x | **2.07x** |

Still on the old path and worth the same treatment: `T_Q4_1` (`ffn_down`, 8
tensors) and `T_Q6_K` (`output.weight`, the lm_head every drafted token pays).

### An AVX-512 Q4 kernel needs its *activation* vector 64-byte aligned

`dot_q4_0rs_avx512` loads the activation with `_mm512_load_ps`, so every buffer
reaching `gemv` as `x` must carry `__attribute__((aligned(64)))`, which is why
every activation global in the engine already has it. `mtp_step`'s statics did
not, and **only `-flto` exposed it**: without LTO the aligned load never
materialized and the AVX-512 build passed every test; with LTO it faulted in
`dot_q4_0rs_2row` on `ln`, 32 bytes off a cache line. A green AVX2 build says
nothing about this class of bug, check AVX-512 *with* the production flag set.

### Beware: `perf stat` loses the OpenMP threads on the decode path

`perf stat -e instructions:u ./run ... -d 8` reports ~21.7G instructions and
5s of task-clock for a run that burns ~400 CPU-seconds, the counts are
inherited by only part of the process and are *stable across configurations
that do genuinely different work*, which makes them look trustworthy. Do not
A/B kernels this way. Wall-clock min-of-R on `bench_spec`, or the interleaved
`tools/spec_e2e.py`, are the measurements that hold up.

### MTP cost split (`CWEN_SPEC_DEBUG=1` prints it)

Per drafted token (AVX2 build, loaded box): nextn layer **8.7 ms**, shared lm_head
**27.8 ms**. The head, not the layer, sets the draft budget, DFlash2 sidesteps
it with a top-16 selector walk, and the same trick is the obvious MTP follow-up.
Commits cost one nextn step per verified position (also 8.7 ms), prefill
included; those are batchable (slot K/V are mutually independent) for roughly
another 5% but nobody has needed it yet.

### Quiet-window benchmarking (still the gating constraint)

Every absolute tok/s number in this repo needs loadavg < 12. Under load 21 the
verify-block ceiling measures 1.46x at B=8; the documented quiet-box figure is
3.91x. **The block sweep, not the drafter, is 90% of a drafted cycle**, at
`-d 4` with full acceptance the split is block 0.36 s/token, drafts 29 ms,
commits 9 ms, against 0.50 s/token serial. So MTP's payoff is gated by machine
load far more than by acceptance. Use `tools/measure_when_quiet.sh`.

## Architecture

```
run.c (~4900 lines)
├── GGUF loader (mmap, all quant types)
├── CWENR sidecar loader (split/interleaved Q4_0 layout)
├── .spec container loader (drafter weights: Q8S/Q8SI/F32)
├── Target model: qwen3.5 hybrid (GDN + full attn, 64 layers)
│   ├── GEMV kernels: dot_q4_0rs_2row, dot_q4_0rsi, dot_q6_K, dot_q5_K...
│   ├── gemvb: batched verify pass (+ dot_q4_bcol unpack-once, B<=8)
│   └── forward_ex / forward_block: serial and batched paths
├── Speculative decoding driver (generate_tokens_spec)
│   ├── Drafter priority: DFlash2 > MTP > ngram-map > ngram-scan
│   ├── Verify walk: greedy prefix + bonus token
│   ├── Snapshot/rollback; short walks re-score via one block sweep
│   └── AIMD E_cap on full accepts
├── DFlash2 drafter (dflash_draft): 5-layer transformer, eh_proj injection,
│   grouped dynamic convs, top-16 selector walk
├── MTP nextn drafter (mtp_step/mtp_commit/mtp_draft): blk.64 as a full-attn
│   decoder block behind the nextn projection, own K/V cache, shared lm_head
├── N-gram counted map (ng_draft_map): chained lookups, persistent NGC2 format
└── Frame server (CWEN_SERVER=1): binary stdin/stdout protocol
```

## Key files

| File | Role |
|---|---|
| `run.c` | entire engine |
| `cwen_tune.h` | GA-tuned constants |
| `ref/qwen35.cpp` | llama.cpp reference graph, authoritative for `graph_mtp` |
| `tools/measure_when_quiet.sh` | loadavg gate for any measurement |
| `tools/pack_dflash.py` | safetensors → `.spec` container for DFlash2 |
| `tools/spec_e2e.py` | interleaved A/B lossless gate (3 engines resident) |
| `tools/spec_check.py` | quick single-prompt lossless check |
| `tools/spec_sweep.py` | OFAT parameter sweep with drift guard |
| `bench_spec` | microbench: gemvb scaling+correctness, snapshot cost, block sweep |
| `docs/DESIGN.md` | architecture, decisions K1-K19, technique comparison |

## How to test

```bash
# MTP lossless gate (drafter is auto-detected from blk.64)
CWEN_SPEC=1 ./run model/Qwen3.8-27B-Q4_0.gguf spec_rep.ids 48 -d 8
# byte-identical to: ./run model/Qwen3.8-27B-Q4_0.gguf spec_rep.ids 48

# full e2e suite (interleaved servers, all drafters lossless)
.venv/bin/python tools/spec_e2e.py --quick

# microbench (also the gemvb correctness gate)
make AVX512=1 bench-spec && ./bench_spec model/Qwen3.8-27B-Q4_0.gguf 2

# full gate (goldens + pinned chain + determinism)
tools/test_speed_gates.sh outputs/
```

## Environment notes

- Box: AMD Ryzen 9 9950X (Zen 5), 32 threads, 126 GB RAM, shared with other tenants
- Load average swings 10–80; use min-of-R timing and interleaved A/B
- Model: `model/Qwen3.8-27B-Q4_0.gguf` (15 GiB) + CWENR sidecar (12.7 GiB resident)
- Drafter: `model/dflash2.spec` (~2 GiB), optional, MTP needs no sidecar
- Build: `make` (AVX2) or `make AVX512=1` (peak); both compile clean under `-Werror`
