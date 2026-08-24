# Changelog

Lab notebook for the Qwen3.8-27B decode-throughput loop.

## 2026-08-24: drafter paired split-Q8 (.spec v2)

- `.spec` container version bumped 1 -> 2: the packer now emits split-Q8
  tensor types (3=Q8S single matrix, 4=Q8SI two matrices paired row-wise) so
  `df_layer` streams each weight once per verify block instead of running two
  gemv passes (k/v land in one `attn_kv`, gate/up in one `mlp_gu`). Engines
  built before this change reject v2 files with `dflash: unsupported version`
  (previously they would have failed per-tensor with
  `unknown tensor type`); the loader still reads every v1 file unchanged, and
  re-packing with the new `tools/pack_dflash.py` is the only migration step.
- **Removed** `pack_dflash.py --prec q8|mixed|q4`: q8 was the only preset
  that produced a usable drafter (measured acceptance: mixed 0.33, q4 0.00),
  so the flag only made broken containers. Scripts invoking the packer must
  drop the flag; output precision is now always split/paired Q8.
- Removed `tools/dflash_ref.py` (numpy reference for drafter dumps; no longer
  referenced by any gate).

## 2026-08-24: spec context-cap room gate + cache-load hardening

- Speculative decode no longer emits the bonus token when a full-accept block
  lands flush at the `CWEN_CTX` cap: serial decode never emits without a free
  slot, so spec could previously return one token more than the window held.
  Token streams are otherwise unchanged.
- `ng_load` (NGC2 cache) now rejects out-of-vocab key ids while reading, so a
  corrupt or foreign cache file cannot plant entries that never match but
  survive every future save.
- New reproducible-build gate: `make verify-reproducible`
  (`tools/check_reproducible.sh`) rebuilds ./run in a second build dir under
  different path/locale/TZ/SOURCE_DATE_EPOCH and requires byte-identical
  binaries.

## 2026-08-24: context-cap truncation signal

- Both decode drivers (serial and speculative) used to stop silently when the
  context window filled mid-generation: a caller asking for more tokens than the
  window had room for got a short stream indistinguishable from a complete
  answer (greedy argmax has no EOS, so a short count has exactly one cause).
  They now print `cwen: context window full (CWEN_CTX=N); emitted X of Y
  requested tokens` on stderr when the count falls short, mirroring the existing
  prompt-truncation warning in main(). Token streams are unchanged; verified
  byte-identical stdout before/after on fit, partial, and zero-room cases.

## 2026-08-24: n-gram cache correctness pass

- Full n-gram maps no longer freeze: `ng_evict_one` drops a least-hit entry
  (llama.cpp ngram-cache eviction style) so long sessions and repeated runs
  keep learning past the 2^18-entry cap. Equal-count victims rotate through
  the table instead of hammering one corner; backward-shift deletion keeps
  linear-probe chains intact.
- Persisted map format bumped to NGC2 (`CWEN_NGRAM_CACHE`): the header now
  carries n_key plus vocab size, so a map never merges across configs or
  models. Old or corrupt cache files now degrade to a cold start with a note
  instead of exiting the process; truncated tails keep what loaded. Saves go
  through `<path>.tmp` + rename so crashes cannot leave torn files.
- Harness `.scratch/test_ngcache.c`: fills to cap, forces thousands of
  evictions with probe-chain integrity checks, LFU survival, chained draft
  proposals, save/load roundtrip, and corrupt/stale/truncated file handling.

## 2026-08-23: ngram-cache, long context (YaRN), drafter precision study

- **n-gram counted map** (`CWEN_NGRAM_CACHE=FILE`, llama.cpp ngram-cache style): FNV-hashed key(N confirmed tokens) -> {most recent continuation, hits}, updated from history as it grows; drafts via chained lookups, history scan kept as longer-proposal fallback; binary persistence loaded at startup / saved at exit. Round trip proven: run 2 loaded 31 entries and drafted a full 8-token accept on its first cycle. Lossless throughout.
- **Long context**: compile-time cap raised to 32768; `CWEN_CTX=N` sizes KV + drafter caches at runtime (every bound checks the runtime cap; strides shared). **Opt-in YaRN** (`CWEN_ROPE_YARN=orig,factor[,bf,bs]`) implements transformers `_compute_yarn_parameters` on this file's pinned inv axis (theta^(-idx/64), llama.cpp-verified): correction-dim ramp + get_mscale cos/sin scaling; defaults bit-stable. Verified: goldens green, 8.4k-token prefill at ctx 8192 completes, plain-vs-DFlash2 streams IDENTICAL, factor 16 measurably changes outputs.
- **Drafter precision study** (`pack_dflash.py --prec q8|mixed|q4`): strawberry 48 tok acceptance - q8 avg kept 5.71 (5/7 full), mixed (attn q8 + mlp q4) 0.33, q4 0.00. All three byte-identical output vs serial. Conclusion: the DFlash2 drafter needs ~Q8 everywhere; do not ship q4 drafts.

## 2026-08-24: split-Q8 drafter container experiment (negative result)

- `pack_dflash.py --layout split`: CWENR-style split streams for the drafter (Q8S singles, Q8SI gate/up + k/v pairs). Loader + kernels (dot_q8s, dot_q8si, df_dual_gemvb) verified numerically exact; lossless gate passes.
- **Measured: split layout is 0.19-0.60x block-Q8 on decode tok/s** (interleaved A/B, best-of-3). At Q4 the split wins because nibble unpacking dominates and separating scales enables pure-nibble streams; at Q8 the block format's inline f16 scale already sits next to its qs bytes — one load serves both. Split turns every scale read into a distant second stream that defeats prefetching.
- Verdict: blocks is correct and optimal at Q8; split kept behind `--layout split` as a lossless-but-slower option. Do not use for production drafter containers.
- Also fixed: dflash_commit ctx-K/V matvec now routes through df_dual_gemvb when pairs are bound (was calling gemv on unbound separate k/v tensors); loader accepts manifest types 3/4 with per-type byte accounting.

## 2026-08-23: speculation technique comparison documented

- DESIGN.md "Speculative decoding landscape": baseline / MTP / DFlash / DFlash2 side by side (published Qwen3.8-27B acceptance lengths: MTP 4.28 mean vs DFlash2 4.80; DFlash superseded) plus every cwen-measured number: serial ~2.8 tok/s quiet-box, ngram 1.4-3.6x pattern workloads, DFlash2 Q8_0 avg kept 5.2-6.0 and 1.66x repeat-heavy wall (0.8x drifting), verify ceiling B=2/4/8 = 1.46x/2.69x/3.91x.
- Recorded the Q4-drafter trap and the two standing headroom items (unpack-once batched GEMV kernels; adaptive draft sizing). MTP flagged as PR18 candidate: `blk.64.nextn.*` tensors already ship in the loaded GGUF; nobody has published an MTP-vs-DFlash2 CPU comparison.
- n-gram counted map + persistent cache (llama.cpp ngram-cache style) noted as open option in DESIGN.md; not built.

## 2026-08-23: PR17 trained DFlash2 drafter (CWEN_DFLASH)

- `tools/pack_dflash.py`: packs `incoai/Qwen3.8-27B-DFlash2` safetensors (1.92B BF16) into `model/dflash2.spec`, a manifest+payload container; drafter matmuls as **Q8_0**, norms/conv bases F32. Q4_0 for the drafter is a trap: acceptance dropped to literally zero (the drafter must reproduce the target's exact argmax; 4-bit noise flips top-1 every time). Q8_0 restored ~5-6 kept tokens per drafted cycle.
- run.c: `.spec` loader with geometry checks, rotate-half RoPE tables (theta 1e7, distinct from target mRoPE), grouped dynamic depthwise convs (`prepare`/`finish` halves from one projection of the sublayer input, zero pad at window start), incremental context injection (fc[25600->5120]+hidden_norm once per committed token; per-layer K/V cached post-norm+rope), bidirectional noise-window attention over [ctx | window] with 2048 sliding mask, selector greedy walk (top-16, <A*proj(h),B> edges); tap capture feeds commits from both serial and block forwards.
- Lossless gate: byte-identical streams vs serial decode on strawberry/repeat/loop prompts including short-walk rejections.
- Measured (loaded box): strawberry repeat 64 tok **1.66x** vs plain (7/9 full accepts, avg kept 6.0); drifting workloads with frequent rejections land ~0.8x (snapshot+replay cost); cooldown softens but does not eliminate.
- Debugging notes kept: ASan caught a double-advance in df_q8_row; a long "divergence" hunt was prolonged by comparison scripts with their own uint8-wrap and double-increment bugs; C scalar == numpy scalar exactly once both were fixed.

## 2026-08-23: config fail-fast pass

- `CWEN_SPEC_DEBUG` is parsed once in `spec_config_init` with the other spec knobs: invalid values exit with a named error before the model load (they used to be ignored), and the server loop no longer re-reads environ on every request frame.
- `CWEN_OMP_THREADS` below 1 exits with a named error instead of silently clamping to 1, same silent-reinterpretation class the strict env parsers were written to kill.
- `./run --help` now lists `CWEN_DUMP_LOGITS` and `CWEN_SPEC_DEBUG`; both were in the README table but had drifted out of the help text.

## 2026-08-23 — harness trust boundaries

- Engine stdout is now parsed strictly by `tools/bench_toks.py` and `tools/spec_check.py`: every whitespace-separated field must be a decimal token id or the run fails naming the field. The old filter silently dropped non-numeric junk, so a corrupted stream shrank the pinned-chain comparison (test_speed_gates EXPECT) and the spec losslessness gate instead of failing them; empty output stays valid (n_predict=0 prefill).
- `tools/spec_e2e.py` Server: stderr is drained continuously by a daemon thread (an undrained stderr pipe could block the engine mid-frame once it wrote ~64 KiB of summaries/traces; reachable today via an inherited CWEN_SPEC_DEBUG=1), reply reads are deadline-bounded (`--timeout`, default 900 s, also bounds startup), and engine death mid-reply now fails with exit status plus a stderr tail instead of a bare struct.error. A wedged engine gets a short shutdown grace after its frame times out.
- Verified against stub engines (roundtrip, mid-frame death, hang timeout, stderr flood >64 KiB, exit before ready) and one real `./run ... prompt1.ids 4` decode through the strict parser.

## 2026-08-23 — spec microbench + e2e suite

- `tools/spec_sweep.py`: OFAT parameter sweep over resident servers (decode-only tok/s via `(t(n)-t(1))/(n-1)` so prefill cancels; min-of-reps; per-frame acceptance stats attributed by frame order; drift probe warns when box load moves >25% between bookends). First attempt was aborted mid-run by a load spike (0.5 tok/s plain at load 44); sweet-spot sweep still pending a quiet window.

- `bench_spec` (`make bench-spec`, `-DCWEN_BENCH_SPEC`): three sections. GEMVB checks `gemvb` output against B separate `gemv` calls (exact) and reports ms/iter + weight GB/s for qkv/gate/down/ssm_out/lm_head at B in {1,2,4,8}; SNAP times the parallel snapshot save+load pair (~45-60 ms, the per-drafted-cycle fixed cost); BLOCK sweeps verify-block vs serial forwards, arms alternating rep-by-rep with min-of-R (sequential phases and means both tracked machine drift, not code).
- Findings: naive `gemvb` scales near-linearly with B without unpack-once kernels; row-pair fast paths recovered the B=1 regression. AVX512 sweep: B=2 1.46x, B=4 2.69x, B=8 3.91x block-vs-serial.
- `tools/spec_e2e.py`: multi-prompt lossless gate over the K16 frame server; engines resident simultaneously with frames rotating per case to share load windows. Quick suite PASS twice: all frames byte-identical across plain/d8/d3; drafting engages on repetitive cases only; non-drafting cases sit at parity within box noise.
- AVX512 coverage: `tools/test_speed_gates.sh` green (goldens 4/4, pinned chain exact, deterministic); spec lossless re-verified under AVX512 including forced rejections (`CWEN_SPEC_NGRAM_N=4 CWEN_SPEC_MIN_DRAFT=1` on the drifting prompt: 8 drafted, 4 full accept, 4 short walks through snapshot rollback+replay, streams identical).
- Driver now takes the plain serial step when no drafts are proposed (undrafted cycles no longer pay block machinery); spec stats line counts full accepts in avg kept.
- `env_bool` restructured so every path returns (clang fuzz build warned: the exit reroute macro hides noreturn).

## 2026-08-23 — block speculation (DFlash-style verify + n-gram drafter)

- `CWEN_SPEC=1` (or CLI `-d/--draft-tokens N`, alias `--spec-draft-n-max`, llama.cpp-style): greedy block speculation with a zero-weight prompt-lookup drafter. One batched forward (`forward_block`) scores `[pending token | drafts]`; every weight matrix streams once per block via new `gemvb` (row-hot across all B activations). Verify walk accepts the longest argmax-matching prefix plus the target's bonus token; short walks restore a GDN snapshot (`Srec`+`Cstate`) and replay only the kept prefix serially.
- Lossless gate: token streams byte-identical to serial decode on all checks run (generic prompt 32 tok, strawberry repeat 96 tok at max_draft 3 and 8; `diff` clean). Drafter fires only on history repeats; cooldown after repeated full rejects keeps non-repetitive work near break-even.
- `tools/spec_check.py`: A/B equivalence runner (plain vs spec streams must be identical).
- Knobs: `CWEN_SPEC_NGRAM_N` (16), `CWEN_SPEC_MAX_DRAFT` (8), `CWEN_SPEC_MIN_DRAFT` (2), `CWEN_SPEC_COOLDOWN` (8); CLI flag wins over env and implies `CWEN_SPEC=1`.
- Known bounds: dumps cover prefill only under spec; trained DFlash2 drafter slots in behind `ngram_draft`'s contract (DESIGN.md PR17) once drafter weights are packaged.

## 2026-08-23 — sidecar freshness stamp (cache-correctness pass)

- CWENR header reserved[8] is now a source stamp: `{u32 "CWEN", u32 src_pages}`; `repack_q4.py` writes it, `load_cwenr` enforces it (stale sidecar dropped with a message, full-GGUF fallback). Replacing a GGUF without repacking used to silently serve old weights from the sidecar.
- `model/Qwen3.8-27B-Q4_0.cwenr` stamped in place (bytes 24..31); Qwen3.6 sidecar left untagged (source GGUF no longer on disk, trusted legacy).
- Verified: tokens identical across sidecar-bound / stale-rejected / untagged paths (`0 198 2 220`); gemv goldens PASS; fuzz build compiles (stamp gate bypassed under `CWEN_FUZZ_LOADER` since the harness pairs one blob as both files).

## 2026-08-16 — I100 cap

Loop reached 100 iterations (jsonl iters 0–99). Tokens stayed `17 15 17 15 95859 17 15 17` except I64 FAST_SILU (quality fail).

Best honest keep: **I06 GDN_OMP=1 + I03 AVX-512 GDN** at **2.792 tok/s** (n=8 ~4.04s). Later reconfirms drifted to ~4.6–4.7s n=8 from machine heat/variance. Do not treat 2.899/3.037/3.228 prints as wins; they were slow-n=2 artifacts.

## 2026-08-16 — I05–I21 (resume)

- I05 CWENR v4 sidecar 12.44 GiB: tokens match, decode 2.682 (no metric win; keep layout)
- I06 `CWEN_IDEA_GDN_OMP=1`: **2.792 tok/s keep** (best so far)
- I07 `-funroll-loops`: 2.756 revert
- I08–I11 OMP 8/12/20/24: all worse than 16
- I12 CONV_OMP: 2.685 revert
- I13 aligned `_mm512_load_ps`: 2.566 revert loads; `aligned_alloc` kept
- I14 PIPE_PF: 2.730 revert
- I15–I21 schedule sweep: `dynamic,32` first print 2.899 was n=2 artifact; confirm 2.693. Keep `static,32`
- RoPE sin/cos table added (quality-neutral). Next: fuse ssm_alpha/beta

## 2026-08-16 — I04 pad + vector SiLU

- Masked zero-pad (`load_pad16`/`store_pad16`) on rmsnorm, l2norm, residual, f32 gemv, SiLU
- Cephes-style `exp512` / `silu512` (not FAST_SILU)
- Tokens unchanged `[17, 15, 17, 15, 95859, 17, 15, 17]`
- decode_tok_s **2.723** (n=8 median 3.970s, better than I03 4.119s; decode-only lower because n=2 also sped up)
- Production dims already divide 16; pad path is unused except the predicted-true branch

## 2026-08-16 — I03 GDN AVX-512

- 16-wide FMA + 2-row unroll in `gdn_step` (C-Kernel-Engine)
- decode_tok_s **2.759** (was 2.635). Tokens match. **keep**

## 2026-08-16 — I02 clang-22

- 2.427 tok/s. Tokens match. **revert** to gcc 16.2.1

## 2026-08-16 — baseline I01

- gcc 16.2.1, `make AVX512=1`, OMP=16, schedule `static,32`
- decode_tok_s = **2.635** (n=2 median 1.781s, n=8 median 4.057s)
- tokens `[17, 15, 17, 15, 95859, 17, 15, 17]` deterministic across 6 runs
- BOS-only prompt; chain is the coherence oracle, not a chat eval
- Historical 3.6 ~2.73 tok/s with CWENR; 3.8 has no sidecar yet

## 2026-08-16 — session start

- Weights present: `model/Qwen3.8-27B-Q4_0.gguf` (15G).
- GGUF: `qwen35`, 65 blocks, MTP `blk.64.nextn.*`, one Q8_0 tensor (`eh_proj`).
- Engine still binds 64 text layers. MTP unused.
- Paper 2608.03893 logged as **skip** for this metric (cross-model KV, not single-model decode).
- MTP added to the try-list (I07/I08).
- Baseline not measured yet.
