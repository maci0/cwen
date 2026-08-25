# cwen — Handoff

## What this is

Pure-C inference engine for Qwen3.8-27B on CPU (AMD Zen 3+, AVX2 baseline /
AVX-512 optional). Single-file engine (`run.c`, ~4700 lines) with mmap Q4 GGUF
loading, speculative decoding (three drafter types), long-context support, and
a comprehensive measurement infrastructure.

Repo: https://github.com/maci0/cwen

## Current state (all lossless, CI green)

| Feature | Status | Key env / CLI |
|---|---|---|
| Serial decode | ✅ production | default |
| N-gram block speculation | ✅ shipped | `CWEN_SPEC=1` or `-d N` |
| Persistent n-gram cache | ✅ shipped | `CWEN_NGRAM_CACHE=FILE` |
| DFlash2 trained drafter | ✅ shipped | `CWEN_DFLASH=model/dflash2.spec` |
| Adaptive draft sizing | ✅ shipped | automatic (rolling acceptance window) |
| Memory residency mode | ✅ shipped | `CWEN_RESIDENCY=1` |
| Long context (32k + YaRN) | ✅ shipped | `CWEN_CTX=N` + `CWEN_ROPE_YARN=o,f` |
| Runtime prefetch knobs | ✅ shipped | `CWEN_PF_T0` / `CWEN_NO_PF` / `CWEN_PIPE_PF` |
| MTP nextn layer (PR18) | 🔄 structural only | auto-detected; acceptance at zero |

Measured (loaded box, AVX-512): serial ~2.8 tok/s quiet-box documented;
DFlash2 repeat-heavy **1.66x wall** (5.71 avg kept/cycle); n-gram
1.4–3.6x on matching workloads.

## Architecture

```
run.c (~4900 lines)
├── GGUF loader (mmap, all quant types)
├── CWENR sidecar loader (split/interleaved Q4_0 layout)
├── .spec container loader (drafter weights: Q8S/Q8SI/F32)
├── Target model: qwen3.5 hybrid (GDN + full attn, 64 layers)
│   ├── GEMV kernels: dot_q4_0rs_2row, dot_q4_0rsi, dot_q6_K, dot_q5_K...
│   ├── gemvb: batched verify pass (weights stream once per B columns)
│   └── forward_ex / forward_block: serial and batched paths
├── Speculative decoding driver (generate_tokens_spec)
│   ├── Drafter priority: DFlash2 > MTP > ngram-map > ngram-scan
│   ├── Verify walk: greedy prefix + bonus token
│   ├── Snapshot/rollback for rejected tails
│   └── Adaptive E_cap from rolling acceptance window
├── DFlash2 drafter (dflash_draft): 5-layer transformer, eh_proj injection,
│   grouped dynamic convs, top-16 selector walk
├── MTP nextn layer (mtp_draft): single decoder layer autoregressive,
│   eh_proj concat(embed, final_hidden), gated attention — STRUCTURAL ONLY
├── N-gram counted map (ng_draft_map): chained lookups, persistent NGC2 format
└── Frame server (CWEN_SERVER=1): binary stdin/stdout protocol
```

## Key files

| File | Role |
|---|---|
| `run.c` | entire engine |
| `cwen_tune.h` | GA-tuned constants |
| `tools/pack_dflash.py` | safetensors → `.spec` container for DFlash2 |
| `tools/spec_e2e.py` | interleaved A/B lossless gate (3 engines resident) |
| `tools/spec_check.py` | quick single-prompt lossless check |
| `tools/spec_sweep.py` | OFAT parameter sweep with drift guard |
| `tools/q8si_check.py` | Q8SI container numeric validation |
| `bench_spec` | microbench: gemvb scaling, snapshot cost, block sweep |
| `docs/DESIGN.md` | architecture, decisions K1-K18, technique comparison |
| `docs/assets/*.svg` | charts: acceptance, layout A/B, ceiling, precision |

## How to test

```bash
# Quick lossless gate
CWEN_DFLASH=model/dflash2.spec CWEN_SPEC=1 \
  ./run model/Qwen3.8-27B-Q4_0.gguf spec_rep.ids 48 -d 8
# Output must be byte-identical to serial decode

# Full e2e suite (interleaved servers)
.venv/bin/python tools/spec_e2e.py --quick

# Microbench
make bench-spec && ./bench_spec model/Qwen3.8-27B-Q4_0.gguf 2

# Full gate (goldens + pinned chain + determinism)
tools/test_speed_gates.sh outputs/
```

## What was done this session

### Shipped
1. **Adaptive draft sizing**: rolling 8-cycle acceptance window shrinks/grows
   E_cap. Strawberry unchanged (5.71); drifting workloads get fewer wasted sweeps.
2. **CWEN_RESIDENCY=1**: THP hints on heap arenas, mlock weight mmap (15.3 GiB),
   prefault large allocations, runtime prefetch knobs (`CWEN_PF_T0`, `CWEN_NO_PF`,
   `CWEN_PIPE_FP`). From cachelm L3-residency learnings.
3. **Interactive flamegraph**: perf → FlameGraph SVG at
   `docs/assets/flamegraph-dflash2.svg`. 88% self-time in four dot kernels.
4. **Comparison charts**: acceptance by workload, layout A/B, block ceiling,
   precision study — generated SVGs in `docs/assets/`.
5. **Split-Q8SI container experiment**: implemented, measured (0.19–0.60x blocks),
   kept as experimental `--layout split`. Negative result documented.

### Debugged and fixed
- Packer per-row payload slicing bug (ASan caught double-increment in df_q8_row)
- Broken AVX512 branches in dot_q8s/dot_q8si (__m256 passed where __m512 expected)
- dflash_commit ctx-K/V matvec missing kv-pair branch (was calling gemv on unbound tensors)
- gcc13 LTO false-positive OOB in argmax_of (pointer-walk fix)
- env_bool restructured so every path returns

## Known issues / remaining work

### PR18 MTP: acceptance tuning (HIGH PRIORITY)
The structural MTP support compiles and runs losslessly, but the drafter's
proposals have zero acceptance. The nextn layer's hidden states are numerically
incorrect. Debugging approach (same as DFlash2):

1. Add CWEN_DF_DUMP-style hooks to mtp_step() capturing per-layer intermediates
2. Write a numpy reference implementing the nextn layer forward
3. Bisect: embed → enorm/hnorm → eh_proj → attention (QK-norm, RoPE, GQA) → MLP
4. Most likely culprits: head-dim mapping in the output gate, RoPE convention,
   or enorm/hnorm application order

The blk.64 tensor shapes confirm: wq [5120→12288] = query(24×256) + gate(24×256);
wk/wv [5120→1024] = 4 KV heads × 256; o_proj [6144→5120]. Same geometry as the
target's full-attn layers.

### Unpack-once batched GEMV
Currently gemvb calls gemv_row/dot functions B times per weight row-pair,
re-doing nibble extraction each time. An unpack-once approach would dequantize
into registers once then FMA against all B columns. Multi-column experiment
showed no wall-clock improvement because L1-resident weights make re-reads free;
the real gain requires avoiding the ALU extraction entirely (new intrinsic
kernels). Estimated ceiling: block sweep 2.97x → ~5x.

### Quiet-window benchmarking
All absolute tok/s numbers need load <12. Guard script:
`outputs/spec/measure_when_quiet.sh` fires automatically when quiet.
`sweep` tool also has a built-in drift guard.

## Performance summary

| Config | tok/s | Conditions |
|---|---|---|
| Serial decode baseline | ~2.8 | quiet box, AVX-512 |
| + DFlash2 (repeat-heavy) | ~4.5-5 est | 1.66x measured same-window |
| + DFlash2 (drifting) | ~0.8x | rejection cost dominates |
| + n-gram (pattern match) | 1.4-3.6x | workload-gated |
| Verify-block ceiling B=8 | 3.91x per sweep | bench_spec min-of-R |

## Environment notes

- Box: AMD Ryzen 9 9950X (Zen 5), 32 threads, 126 GB RAM, shared with other tenants
- Load average swings 10–80; use min-of-R timing and interleaved A/B
- Model: `model/Qwen3.8-27B-Q4_0.gguf` (15 GiB) + CWENR sidecar (12.7 GiB resident)
- Drafter: `model/dflash2.spec` (~2 GiB, Q8_0 block layout)
- Build: `make` (AVX2) or `make AVX512=1` (peak); both must compile clean under -Werror
