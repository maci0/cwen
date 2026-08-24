# cwen threat model

Starter model of the attack surface as built. Every claim carries a file reference so
the next pass can re-verify it against code. Individual vulnerabilities are out of
scope here: they belong to sec-review, which this document aims.

- Last reviewed: 2026-08-24 (against run.c at 4964 lines, working tree incl. the
  split-Q8 drafter changes)
- Owner: unset (organizational: to be assigned)
- Review cadence: unset (organizational: to be set)

## Scope

`run.c` is a single-process, local CLI inference engine compiled three ways from one
translation unit (`run`, `bench_q4_gemv`, `bench_spec`; Makefile:121-122,179-181,185-187)
plus a libFuzzer build (`fuzz_loader`, Makefile:159-162). There is no network listener,
no HTTP/RPC surface, and no stored credentials in the repo (`rg -in 'api_key|secret|
password|credential'` over run.c, tools/, Makefile, README.md, DESIGN.md finds nothing;
re-verified 2026-08-24). The attackable inputs are binary model files (GGUF, CWENR
sidecar, DFlash `.spec` drafter container, NGC2 n-gram persistence file), the binary
prompt file, environment variables, CLI arguments, and the optional stdin frame
protocol. Deployment surface is one dev workstation plus a GitHub Actions CI job that
builds, lints, smoke-tests, and rebuild-compares (`.github/workflows/ci.yml`; actions
pinned by commit SHA); there are no containers, services, or deploy artifacts declaring
anything internet-facing.

## Risk-ranked summary

| # | Risk | Boundary | Status |
|---|------|----------|--------|
| R1 | Memory corruption from a hostile GGUF/CWENR/.spec/NGC2 file leads to code execution under the user's privileges | filesystem -> loader | Mitigated in depth (bounds checks, shape pinning, differential fuzzing over all four parsers plus the cache round trip); residual risk in kernel indexing paths |
| R2 | Substituted or poisoned weight/drafter file accepted by the delivery path | supply chain -> disk -> loader | Target weights mitigated at download time (SHA-256 pin, tools/download.sh:15); residual: sidecar binding is page-count only, and the drafter `.spec` has no integrity link at all (impact bounded by lossless verify, see Gaps #3) |
| R3 | Resource exhaustion via the stdin server loop: unthrottled full-context frames on a ~13-15 GiB-resident process | parent process -> server_loop | Partially mitigated: validated framing, per-frame caps tied to `CWEN_CTX` (default 4096, ceiling 32768); no quota/backpressure/audit |
| R4 | `CWEN_DUMP` makes the process write dump files into an arbitrary writable directory; `CWEN_NGRAM_CACHE` gets clobbered by the atexit save | environment -> process -> filesystem | Accepted (requires local env control already); dump dir preflighted (run.c:2485-2497), cache save is tmp+rename (run.c:4371-4388) |
| R5 | Tampered golden dumps mask correctness regressions or steer kernel changes | filesystem -> verify tooling | Gap: goldens unauthenticated |

## Assets

| Asset | Location | Impact if compromised |
|-------|----------|----------------------|
| Host process integrity | `./run` process running with user privileges | Loader memory corruption = arbitrary code execution under the invoking user |
| Target weights (~15 GiB GGUF + CWENR sidecar) | `model/Qwen3.8-27B-Q4_0.gguf`, sibling `.cwenr` | Public open weights: confidentiality low; integrity high (poisoned weights silently alter every output) |
| Drafter weights (.spec container) | `model/dflash2.spec`, producer tools/pack_dflash.py | Integrity affects speed only, not outputs: every draft is verified against target argmax before emission (run.c:4654-4661, contract 4572-4584); a hostile drafter can degrade throughput, not corrupt results |
| N-gram persistence file (NGC2) | path from `CWEN_NGRAM_CACHE`, format run.c:4367-4370 | Same bound as the drafter: planted continuations are proposals, always verified (run.c:4654-4661); worst case is throughput skew and a poisoned file that round-trips across runs |
| Output streams (generated token ids, e2e dumps, server reply frames) | stdout, `golden/e2e*_c/` | Research results corrupted; dumps feed accept/reject decisions |
| Golden reference data | `golden/` dirs, consumed by `bench_q4_gemv` and `tools/accept.py` | Tampering masks regressions in the SIMD kernels |
| Compute/availability | ~12.7-15+ GiB resident after load (README.md:75) | Long load + fault-in cost per invocation; a crash loop wastes CCD time |
| Secrets | none found in code or tools | n/a |

## Entry points

All enumerated from code; none are network listeners.

| Entry point | Code | Input trust |
|-------------|------|-------------|
| CLI argv: `[MODEL] [IDS_FILE] [N_PREDICT]` plus `-h/--help`, `-d/--draft-tokens/--spec-draft-n-max N`; unknown options and >3 positionals rejected | production `main` run.c:4804-4832; strict range parse `arg_int_range` run.c:2451-2461 | Untrusted paths; numeric args strictly parsed and bounded |
| GGUF container parser (magic, version, kv walk, tensor table, offsets, extents) | `load_gguf` run.c:3034-3107 | Untrusted file from disk |
| CWENR sidecar parser (v2/v3/v4 directory, staleness stamp, offset binding) | `load_cwenr` run.c:3233-3390; invoked from `load_model` run.c:3546-3555; sidecar path from `CWEN_REPACK` env or derived suffix `cwenr_path_for` run.c:3114-3129 | Untrusted file from disk |
| DFlash2 `.spec` drafter container ("DFSP" header, v1/v2 version gate, 96B entry walk, offset/nbytes binding, geometry pinning incl. paired `mlp_gu`/`attn_kv` tensors), gated by `CWEN_DFLASH` env | gate run.c:4843-4844; loader `load_dflash` run.c:3415-3544; called from main run.c:4859-4860; producer tools/pack_dflash.py (emits v2, types 0-4) | Untrusted file from disk; cannot alter outputs (see Assets), only parse safely or not |
| NGC2 n-gram persistence file ("NGC2" magic, header config match, record walk), gated by `CWEN_NGRAM_CACHE` + `CWEN_SPEC=1` without `CWEN_DFLASH` | gate/inert-knob notice run.c:4845-4851; load at startup run.c:4909-4915; `ng_load` run.c:4393-4439; save-at-exit `ng_save_atexit` run.c:4390-4392 | Untrusted file from disk; malformed/stale configs degrade to a cold start (run.c:4400-4417), never fail the run |
| Prompt token file: raw little-endian int32 ids | main run.c:4923-4949 | Untrusted; capped at runtime context `g_ctx` during read (run.c:4926), per-id range-checked run.c:4927-4931; size-vs-count warnings for trailing/oversized files run.c:4938-4945 |
| Stdin server mode (strict bool `CWEN_SERVER`): request frames `<u32 n_prompt><u32 n_gen><n_prompt*i32>`, `0xffffffff` header word = EOF sentinel; replies `<u32 n_out><n_out*i32>` | gate run.c:4840-4841 (`env_bool` run.c:2466-2475); frame parser `server_read_frame` run.c:3929-3947 (helpers `rd_u32f` 3897-3901, `drain_u32f` 3908-3915, `first_bad_token` 3918-3922); reply writer run.c:4735-4741; loop `server_loop` run.c:4747-4761 | Untrusted peer controlling stdin/stdout |
| Environment variables: `CWEN_DUMP`/`CWEN_DUMP_LAYERS`/`CWEN_DUMP_LOGITS` (run.c:2477-2517, preflight 2485-2497), `CWEN_REPACK` (run.c:3115-3119), `CWEN_SPEC*` knobs incl. `CWEN_SPEC_DEBUG` (`spec_config_init` run.c:3768-3800), `CWEN_DFLASH` (run.c:4843), `CWEN_SERVER` (run.c:4841), `CWEN_CTX` + `CWEN_ROPE_YARN` (`rope_env_init` run.c:3804-3831), `CWEN_NGRAM_CACHE` (run.c:4848-4851, 4909-4915), `CWEN_OMP_THREADS` (run.c:3871-3885), `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`/`OMP_PROC_BIND`/`OMP_PLACES` (read by libgomp, defaulted at run.c:3843-3865); diagnostic presence-only knob `CWEN_DF_TOP` read mid-run in `dflash_draft` (run.c:2980-2989) | strict parsers: `env_int` run.c:2437-2447, `env_bool` run.c:2466-2475 | Local user control only; set-but-invalid values exit with named errors instead of silent fallbacks (`CWEN_DF_TOP` is the one truthiness-style knob; stderr diagnostics only) |
| One-shot self re-exec: exports missing OMP defaults plus `CWEN_OMPREEXEC` marker, then `execv(/proc/self/exe, argv)` | `cwen_omp_init` run.c:3840-3866 (re-exec block 3843-3865) | Replays own binary and argv once; marker prevents loops; same user, same image |
| Debug dump writer | `dump_f32` run.c:2420-2433 writes `%s/%s.bin` under the `CWEN_DUMP` dir; filenames are code-controlled literals/format strings (e.g. `"embed.bin"`, `"layer%02d.bin"`, `"logits_pos%02d.bin"`; run.c:2537-2543, 2556-2583) | Reverse flow: process -> filesystem |
| NGC2 writer (reverse flow) | `ng_save` run.c:4371-4388 writes `<path>.tmp` then renames over the `CWEN_NGRAM_CACHE` path; registered via `atexit` run.c:4915 | Reverse flow: process -> filesystem |
| Bench harness golden dir: `meta.json` parsed by strstr/atoi (`read_meta_name` run.c:3969-3980), fixture preflight (`golden_preflight` run.c:3985-4001), x.bin/y_ref.bin floats | bench_q4_gemv main run.c:4002+ (built with `-DCWEN_BENCH_Q4_GEMV`, Makefile:179-181) | Trusted-ish (own goldens), parsed defensively anyway |
| Spec microbench: `[MODEL] [ITERS]` only, no external data files | bench_spec main run.c:4070+ (built with `-DCWEN_BENCH_SPEC`, Makefile:185-187) | Model file only |
| libFuzzer harness feeding arbitrary bytes as GGUF, CWENR, `.spec`, frame stream, AND NGC2 cache; frame stream cross-checked against an independent decoder; NGC2 checked by save/reload round-trip diff; unique mkstemp fixture files | tools/fuzz_loader.c:28-34 (harness hooks), :57-70 (mkstemp fixtures), :83-124 (reference decoder + divergence aborts), :126-144 (five-way parse, round trip, post-load invariant aborts); engine-side hooks run.c:4441+ (`cw_fuzz_ngcache` 4458+, round-trip contract 4448-4457); build/run Makefile:159-174; seeds incl. DFSP and NGC2 heads tools/gen_fuzz_seeds.py:73-127 | Deliberately hostile input |
| Python/shell tooling: HF download with SHA-256 pin, repack with source stamp, drafter packing, benches | tools/download.sh:15 (pin), :18-29 (check), :31-36 (already-present verify), :44-52 (network fetch), :53-58 (post-download delete on mismatch); stamp written tools/repack_q4.py:242,250; drafter packer resolves its safetensors by cache path only tools/pack_dflash.py:149-154,164-172 (no content pin, see Gaps #3); subprocess runners tools/ga_evolve.py:333,342 | Developer-invoked; network only during download |

## Trust boundaries and data flow

1. **filesystem -> loader** (primary): model GGUF/CWENR/.spec/NGC2 and prompt ids cross
   into the process via mmap/read. Validation points: the parsers listed above. This is
   the boundary where untrusted bytes first become pointers and sizes.
2. **supply chain -> disk**: target weights arrive via `hf_hub_download`
   (tools/download.sh:44-52). Content is pinned to a SHA-256 recorded in the script
   (tools/download.sh:15); the check runs on an already-present file (:31-36) and again
   after download, deleting mismatches (:53-58). TLS protects the transfer; the pin
   protects the content choice. The drafter `.spec` is produced locally by
   tools/pack_dflash.py from a safetensors snapshot whose default path encodes a commit
   hash but whose bytes are not hash-pinned (:149-154,164-172). Adjacent dependency
   surfaces: Python tool deps install from a hash-checked lock (`make setup`,
   Makefile:97-104, lock regenerated 106-115), CI actions are pinned by commit SHA
   (.github/workflows/ci.yml), and the compiled binary itself has a rebuild-and-compare
   gate (`verify-reproducible`, tools/check_reproducible.sh, Makefile:232-235, wired
   into CI).
3. **parent process -> server_loop**: whoever supplies stdin/stdout of the process.
   Frame validation (run.c:3929-3947) is the only control on the engine side; there is
   no authentication, authorization, or rate limiting at this boundary. The reference
   client bounds itself, not the engine: deadline-bounded reply reads
   (tools/spec_e2e.py:91-130) and continuous stderr drain (:88-91,126-130). If this
   process is ever placed behind a socket wrapper, that wrapper becomes the
   authentication point and none exists today.
4. **environment -> process**: env vars select the sidecar path, drafter path, dump
   destination, cache path, speculation knobs, context size, and thread counts. Env
   control implies local execution rights, so this is a low privilege-transition
   boundary; it still feeds file-write paths (R4). Special cases: when OMP knobs are
   unset, the process re-executes its own image once with exported defaults
   (run.c:3843-3865) before any model I/O; `CWEN_CTX` raises the per-frame work ceiling
   shared with the server (run.c:3806-3809); `CWEN_DF_TOP` is read mid-run rather than
   once at startup (run.c:2980).
5. **process -> filesystem (reverse flow)**: `dump_f32` writes into any directory named
   by `CWEN_DUMP` (preflighted to exist and be writable, run.c:2485-2497); filenames are
   code-controlled (run.c:2537-2583), so no traversal, but existing files at those names
   are clobbered. Writes are fail-fast on short write/close error (run.c:2420-2433).
   `ng_save` similarly overwrites `<CWEN_NGRAM_CACHE>` via tmp+rename (run.c:4371-4388).
6. **build/tooling -> runtime**: `tools/repack_q4.py` produces the CWENR consumed by
   the C parser (and stamps it with the source GGUF's page count,
   tools/repack_q4.py:242,250); `tools/pack_dflash.py` produces the `.spec`;
   `tools/gen_golden.py` produces goldens consumed by verification. A tampered producer
   poisons everything downstream without touching run.c.

Privilege transitions: none documented because none exist; the process never drops or
raises privileges, never forks workers with a different identity. The OMP re-exec
(run.c:3843-3865) re-runs the same binary as the same user with only environment
additions.

Secrets flow: none. No credentials enter, live in, or leave the codebase (verified by
search noted under Scope).

## Threats per boundary

### filesystem -> loader (R1, STRIDE: tampering, information disclosure, elevation of privilege)

A hostile model/cache file is crafted to corrupt memory during parse or bind bad extents
for the kernels. Concrete classes tied to this code:

GGUF container:

- Truncation and cursor overrun past the mmap: countered by the bounds-checked cursor
  `adv()` (run.c:2996-2998) used by all readers.
- Oversized strings smashing stack buffers: countered by u64-cap compare in `rd_str`
  (run.c:3002-3006).
- Deeply nested kv arrays recursing to stack exhaustion: countered by depth limit 64
  (run.c:3007-3021).
- Unknown GGUF versions misparsed: rejected outright, v2/v3 only (run.c:3050-3052).
- Absurd tensor counts/dimension counts causing downstream overflow: countered by
  table cap (run.c:3055), n_dims cap (run.c:3063), and zero/INT_MAX rejection
  (run.c:3064-3072).
- GGUF claiming internal sidecar/drafter-only type tags (T_Q4_0R/RS/RSI/Q8S/Q8SI, the
  100+ range): rejected at parse; they would otherwise pass the extent check with NULL
  scales and fault the first gemv (run.c:3073-3079).
- Tensor offsets pointing outside the map: countered by offset containment plus the
  product-cannot-overflow extent check (run.c:3091-3101).

CWENR sidecar:

- Malformed headers: magic/version/count/dir-fit validation (run.c:3247, 3274-3282).
- Entry offsets escaping the map: remainder-form containment that cannot wrap
  (run.c:3298-3303 v3/v4, 3336-3339 v2), exact-size match for v2 payloads
  (run.c:3340-3341), shared entry guards (name known, source is Q4_0, shapes equal:
  run.c:3219-3231).
- Stale sidecar served over a changed model: page-count stamp dropped when mismatched
  (run.c:3250-3273; producer stamp tools/repack_q4.py:242,250).

DFlash `.spec` container (v1 and v2 accepted; v2 adds split-Q8 types 3/4):

- Bad magic/version/truncated entries: run.c:3425, 3428-3430, 3436.
- Unknown tensor types: mapped through a closed set and rejected otherwise
  (run.c:3460-3462); v1 readers reject the whole file at the version gate instead of
  misreading v2 entries (run.c:3428-3430).
- Zero/huge/misaligned dims: positive-int cast bounds plus QK4 row alignment
  (run.c:3447-3448).
- Offsets/nbytes escaping the map: wrap-safe remainder-form containment
  (run.c:3449-3453) plus exact nbytes-vs-declared-geometry match via the shared
  `dflash_tensor_bytes` (run.c:3404-3414, 3454-3459; one arm sizes all three Q8
  flavors at 34B per 32 weights, so the split-Q8 scale pointers derived at
  run.c:3464-3465 stay inside the matched extent).
- Unknown or duplicate tensor names: rejected (run.c:3467-3486 layer-name parse and
  name table, 3487-3498 globals).
- Declared geometry disagreeing with compile-time drafter dims: every consumed tensor
  pinned exactly after the walk (run.c:3501-3543), now including the paired containers
  `mlp_gu` ([H x 2I]) and `attn_kv` ([H x 2*DL_KV]) as alternatives to solo gate/up/k/v
  (run.c:3507-3519), raw-F32 norm rows (3524-3529), fixed-block Q8_0 selector tables
  indexed by token id (3535-3541), and the hproj gemv output width (3543).

NGC2 cache file:

- Wrong magic or a foreign config (n_key/vocab mismatch): degrades to a cold start
  instead of merging foreign continuations (run.c:4409-4417).
- Truncated header/tail: ignored (run.c:4404-4408, 4426-4428).
- Out-of-range keys, continuation ids, or non-positive counts: skipped per record
  (run.c:4427-4433), so a corrupt file cannot plant entries that persist through every
  future save.
- Oversized `used` count: bounded by the live map capacity, loop breaks when full
  (run.c:4425).

Residual (all four parsers): kernel-side indexing itself is not independently validated;
it trusts load-time shape pinning (`rebind_layers_from_tens` run.c:3170-3213 for the
target, the geometry blocks above for the drafter). Types whose byte size `row_bytes`
cannot compute are excluded from the GGUF extent check and must never be read
(run.c:3095-3100; the internal-only T_Q8SI also returns 0 there but never flows through
GGUF, which rejects the whole 100+ tag range); any future kernel reading such a tensor
reintroduces OOB. Load-time shape pinning is the single control class carrying several
high-impact threats (see Single points of failure).

### supply chain -> disk (R2, tampering)

An attacker who controls the mirror, proxy cache, or local `model/` file tries to
substitute a weight file. The download path rejects this for target weights: the script
records a SHA-256 for the exact artifact (tools/download.sh:15) and verifies before
keeping an existing file and after every download, removing mismatches
(tools/download.sh:31-36,53-58). Substituting weights therefore requires either a
SHA-256 collision, write access to `model/` after verification (same authority as
R5-class local tampering), or talking a developer into deliberately updating the pin
(a documented, visible decision: tools/download.sh:10-13 comments).

Residual gaps: the CWENR sidecar is bound to its source GGUF only by a page-count
stamp checked when present (run.c:3250-3273); the stamp is optional (legacy untagged
sidecars are trusted as-is) and page counts are trivially matched by a crafted pair.
The drafter `.spec` has no integrity link to anything: tools/pack_dflash.py resolves
its safetensors source by cache path only (:149-154), without content verification.
Impact differs sharply between the two: poisoned target weights (GGUF/sidecar pair)
silently redefine the model's function, while a poisoned drafter cannot change a
single emitted token (every draft must match target argmax to be kept, run.c:4654-4661);
its worst case is pathological proposals that drag decoding below serial speed.

### parent -> server_loop (R3, denial of service, repudiation)

Each accepted frame forces `reset_state()` plus prefill up to the runtime context cap
(default 4096, raisable to MAX_SEQ=32768 via `CWEN_CTX`) and generation of up to n_gen
tokens (server_loop run.c:4747-4761, generate_tokens run.c:4711-4729, cap set at
run.c:3806-3809, MAX_SEQ at run.c:83). The framing layer is hardened and spec-pinned:
count caps against `g_ctx` (run.c:3934), payload drain on rejection so the stream stays
aligned (run.c:3908-3915, 3936), EOF sentinel handling (run.c:3933), token range
validation inside the shared parser (run.c:3940-3944), and differential fuzzing against
an independent decoder written from the protocol text (tools/fuzz_loader.c:83-124).
What remains unmitigated is economic: no per-peer quota, no backpressure, and no audit
record. A hostile pipe owner can pin a CPU core indefinitely, and afterwards there is
nothing to investigate beyond stderr progress/validation lines (run.c:3935, 3942,
4738, 4920).

### supply chain -> goldens / tooling producers (R5, tampering)

Goldens and CWENR/.spec artifacts are plain files with no integrity metadata; whoever
can write `golden/` decides what `make verify` accepts (Makefile:198-209 compares only
against those files). Impact limited to development-time wrong conclusions, hence lower
rank. Tooling-side control that exists: engine stdout is decoded strictly as decimal
token ids by the gate runners, so a corrupted stream fails the comparison instead of
silently shrinking it (tools/bench_toks.py:25-38; reused by tools/spec_check.py:24,67).

## Mitigations map (existing controls)

| Control | Covers | Reference |
|---------|--------|-----------|
| Bounds-checked read cursor on every GGUF access | truncation/OOB reads | run.c:2996-2998 |
| String length cap before memcpy | stack smash via kv keys/tensor names | run.c:3002-3006 |
| KV nesting depth limit | stack exhaustion | run.c:3007-3021 |
| Version/table-cap/dim rejection + non-overflowing extent math | integer overflow to OOB | run.c:3050-3055, 3063-3072, 3091-3101 |
| Reserved internal type tags rejected at GGUF parse | NULL-scales fault on first gemv | run.c:3073-3079 |
| Tensor offset containment vs mmap length | arbitrary-offset OOB | run.c:3091-3101 |
| Exact shape pinning of every consumed target weight/vector | kernel OOB from mismatched dims | run.c:3170-3213 |
| Weight type contracts (F32 vs matmul types, embed dispatch, lm_head dispatch) | garbage-type misinterpretation, silent stale logits | run.c:3133-3150, 3171-3185 |
| CWENR header validation (magic/version/count/dir-fit) | malformed sidecar headers | run.c:3247, 3274-3282 |
| CWENR per-entry OOB + exact-size checks (remainder form) | sidecar offset attacks | run.c:3298-3303, 3336-3341 |
| CWENR staleness stamp vs source GGUF page count | stale sidecar served over changed model | run.c:3250-3273; producer tools/repack_q4.py:242,250 |
| DFlash header/version/truncation/type/dim/offset/nbytes validation | malformed .spec containers | run.c:3425-3462 |
| DFlash unknown-name/duplicate rejection + full geometry pinning (incl. paired mlp_gu/attn_kv) | drafter kernel OOB from mismatched shapes | run.c:3467-3498, 3501-3543 |
| Lossless draft verify (draft kept iff equals target argmax) | poisoned drafter or poisoned n-gram cache altering outputs | run.c:4572-4584, 4654-4661 |
| Hard token-range check at use site (not just parse sites) | OOB embedding lookup via token id | run.c:1813-1817, 2713-2714 |
| Shared frame parser: count caps, reject-drain alignment, EOF sentinel | frame-driven buffer overrun, stream desync | run.c:3897-3947 |
| Strict CLI integer parsing with named errors | argv misuse/silent fallbacks | run.c:2451-2461 |
| Strict env parses (`env_int`, `env_bool`) | silent atoi fallbacks, truthiness bugs (CWEN_SERVER=0 starting the server) | run.c:2437-2475 |
| Speculation knob range validation, parsed once pre-load | absurd draft/cooldown configs; no per-frame env re-read | run.c:3768-3800 |
| Context-window env validation (`CWEN_CTX` in [64, MAX_SEQ]) | allocation-size and sequence-bound disagreement | run.c:3804-3809 |
| `CWEN_DUMP` directory preflight (exists, writable) before the expensive load | typo'd dump dir failing mid-run | run.c:2485-2497 |
| One-shot marked self re-exec (loop-proof via `CWEN_OMPREEXEC`) | OMP env never reaching libgomp constructor | run.c:3843-3865 |
| Fail-fast output writes (reply frames, dumps) | silent token/data loss on I/O failure | run.c:2420-2433, 4735-4741 |
| NGC2 magic + config-match rejection (cold start on mismatch) | foreign-config cache silently merged | run.c:4409-4417 |
| NGC2 per-record key/token/count range validation | corrupt cache planting unmatched-but-persistent entries | run.c:4427-4433 |
| NGC2 atomic save (tmp + rename, short-write abort) | torn cache file on crash/interrupt | run.c:4371-4388 |
| Golden fixture preflight (existence, regular file, minimum size) | late confusing failures on stale/missing goldens | run.c:3985-4001 |
| Defensive meta.json parsing in bench (bounded copy, alloc checks) | hostile golden dir | run.c:3969-3980 |
| libFuzzer + ASan/UBSan over all four untrusted parsers plus the NGC2 save/load round trip | regression detection on parser bugs | tools/fuzz_loader.c:126-144, Makefile:155-174, tools/gen_fuzz_seeds.py:73-127 |
| Differential fuzzing of frame parser vs independent spec-written decoder | parser/protocol drift becoming a desync bug | tools/fuzz_loader.c:77-124 |
| Differential fuzzing of NGC2 load/save/load round trip (entry sets + hit counts diffed) | cache serialization drift decaying silently | run.c:4448-4457, tools/fuzz_loader.c:10-18,135 |
| Post-load invariant aborts in harness (pointer escape, bound-tensor containment) | silent bad binds surviving parse | tools/fuzz_loader.c:142-143; run.c fuzz hooks 4441+ |
| mkstemp fixture files in harness | symlink-clobber of fixed /tmp names (CWE-377/59) | tools/fuzz_loader.c:45-70 |
| Strict decimal-token decode of engine stdout in gate runners | corrupted stream shrinking gate comparisons | tools/bench_toks.py:25-38 (single copy, reused by tools/spec_check.py:24,67) |
| Deadline-bounded reply reads + continuous stderr drain in reference client | hung engine wedging the client / blocking mid-frame on pipe fill | tools/spec_e2e.py:88-130 |
| SHA-256 content pin on downloaded target weights | substituted/poisoned downloads | tools/download.sh:15,31-36,53-58 |
| Hash-checked Python dependency lock (direct + transitive) | poisoned tool deps | Makefile:97-115 |
| Reproducible-build gate (byte-identical ./run across path/locale/TZ/SOURCE_DATE_EPOCH), enforced in CI too | build-environment tamper / nondeterministic artifacts | tools/check_reproducible.sh, Makefile:232-235, .github/workflows/ci.yml |
| CI actions pinned by commit SHA | action-supply-chain substitution | .github/workflows/ci.yml (checkout, setup-uv steps) |

Claims made elsewhere were checked: DESIGN.md "Security & operability"
(DESIGN.md:394-401) says local-only CLI, bounds-checked offsets, fuzzing over
GGUF/CWENR/DFlash-.spec/frame parsing plus the NGC2 cache round trip, the
dump/server/spec env surfaces, and page-in behavior; all match the code above. No
security claim was found that the code contradicts.

Single points of failure: load-time shape pinning (`rebind_layers_from_tens`
run.c:3170-3213 plus the drafter geometry block run.c:3501-3543) is the one control
class standing between hostile files and every kernel-side memory access; the SHA-256
pin (tools/download.sh:15) is the one control standing between the delivery path and
every experiment downstream. Both are listed here so their failure modes get first
attention in sec-review passes.

## Gaps (unmitigated, ranked)

1. R3: no quota, backpressure, or audit trail on server frames (run.c:4747-4761).
   Exploitability high for anyone already holding stdin; impact limited to the
   process's own resources, scaled by the operator's chosen `CWEN_CTX`.
2. R1 residual: `row_bytes==0` types bypass the GGUF extent check by design and depend
   on a "never read" invariant enforced nowhere mechanically (run.c:3095-3100);
   kernel-side indexing relies entirely on load-time shape pinning. Fuzzing covers the
   four parsers and the cache round trip, not the gemv kernels themselves.
3. R2 residual: CWENR-to-GGUF binding is a page-count stamp, present optionally
   (run.c:3250-3273); a crafted GGUF+sidecar pair with matching page count binds
   cleanly and redefines the function. The drafter `.spec` has no binding at all and
   its producer does not content-pin the safetensors input
   (tools/pack_dflash.py:149-154,164-172); bounded by the lossless verify to
   throughput damage only (run.c:4654-4661).
4. R5: goldens and CWENR/.spec artifacts carry no integrity metadata; producers and
   consumers trust the filesystem.
5. No SECURITY.md exists: there is no documented disclosure contact or supported-version
   statement, and no path from "vulnerability reported" to "fix shipped". Not invented
   here; recording its absence.

## Abuse cases

Single-user research CLI, so classic multi-tenant abuse does not apply. Documented
hostile-but-authorized scenarios:

- **Compute squandering via valid frames**: a peer holding the server's stdin sends
  repeated maximal frames (`n_prompt = n_gen = g_ctx`); each is individually legal,
  passes validation (run.c:3934-3946), and costs seconds of full-model compute plus
  recurrent-state scrubbing in `reset_state` (called run.c:4754). No quota intervenes.
- **Pinned-artifact swap (function-defining)**: someone with write access to `model/`
  replaces the GGUF and regenerates or retags the sidecar so the page-count stamp
  matches (run.c:3250-3273); the loader cannot distinguish this from a legitimate
  repack. Every downstream result is silently whatever the attacker chose. The
  download-time SHA-256 pin (tools/download.sh:15) does not help because it runs only
  in the download script.
- **Drafter sabotage (throughput-only)**: same actor swaps `model/dflash2.spec` for a
  crafted container proposing one-token drafts every cycle; all loads and verifies
  cleanly (geometry checks run.c:3501-3543 do not constrain proposal quality), every
  draft is rejected at the argmax compare (run.c:4654-4661), and decoding pays snapshot +
  replay overhead below serial speed. Outputs stay byte-identical.
- **N-gram cache poisoning (throughput-only, sticky)**: an actor plants a crafted NGC2
  file at the `CWEN_NGRAM_CACHE` path with valid magic/config (run.c:4409-4417) and
  in-range keys carrying attacker-chosen hit counts (accepted at run.c:4427-4433,
  counts merge with an INT_MAX clamp at run.c:4288-4306). Every proposal is still
  verified (run.c:4654-4661), so outputs stay byte-identical; the damage is skewed
  speculation and a poisoned map that re-saves itself at exit (run.c:4390-4392),
  surviving across runs until the file is deleted.
- **Golden laundering**: someone with write access to `golden/` replaces reference
  floats so a broken kernel change still prints PASS (Makefile:198-209 compares only
  against those files).

Trust placed in client-side enforcement: none found; there is no client-facing policy
layer to bypass.

## Response readiness (note only)

- Audit trail: none. Server mode emits only stderr progress/validation lines
  (run.c:3935, 3942, 4738, 4920); there is no request log to investigate incidents
  from.
- Disclosure path: absent (no SECURITY.md). Fuzz crashes land in `fuzz_out/artifacts/`
  (Makefile:169-174), which is the closest thing to a vulnerability intake today.
