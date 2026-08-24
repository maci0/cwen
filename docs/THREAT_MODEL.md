# cwen threat model

Starter model of the attack surface as built. Every claim carries a file reference so
the next pass can re-verify it against code. Individual vulnerabilities are out of
scope here: they belong to sec-review, which this document aims.

- Last reviewed: 2026-08-24 (against run.c at 4298 lines)
- Owner: unset (organizational: to be assigned)
- Review cadence: unset (organizational: to be set)

## Scope

`run.c` is a single-process, local CLI inference engine compiled three ways from one
translation unit (`run`, `bench_q4_gemv`, `bench_spec`; Makefile:109-110,165-167,171-173)
plus a libFuzzer build (`fuzz_loader`, Makefile:145-148). There is no network listener,
no HTTP/RPC surface, and no stored credentials in the repo (`rg -in 'api_key|secret|
password|credential'` over run.c, tools/, Makefile, README.md, DESIGN.md finds
nothing; re-verified 2026-08-24). The attackable inputs are binary model files (GGUF,
CWENR sidecar, DFlash `.spec` drafter container), the binary prompt file, environment
variables, CLI arguments, and the optional stdin frame protocol. Deployment surface is
one dev workstation; there are no containers, services, or CI artifacts declaring
otherwise.

## Risk-ranked summary

| # | Risk | Boundary | Status |
|---|------|----------|--------|
| R1 | Memory corruption from a hostile GGUF/CWENR/.spec model file leads to code execution under the user's privileges | filesystem -> loader | Mitigated in depth (bounds checks, shape pinning, differential fuzzing); residual risk in kernel indexing paths |
| R2 | Substituted or poisoned weight/drafter file accepted by the download path | supply chain -> disk -> loader | Target weights mitigated at download time (SHA-256 pin, tools/download.sh:15); residual: sidecar binding is page-count only, and the drafter `.spec` has no integrity link at all (impact bounded by lossless verify, see Gaps #3) |
| R3 | Resource exhaustion via the stdin server loop: unthrottled full-context frames on a ~15 GiB-resident process | parent process -> server_loop | Partially mitigated: frame caps + validated framing exist, no quota/audit |
| R4 | `CWEN_DUMP` makes the process write dump files into an arbitrary writable directory | environment -> process -> filesystem | Accepted (requires local env control already) |
| R5 | Tampered golden dumps mask correctness regressions or steer kernel changes | filesystem -> verify tooling | Gap: goldens unauthenticated |

## Assets

| Asset | Location | Impact if compromised |
|-------|----------|----------------------|
| Host process integrity | `./run` process running with user privileges | Loader memory corruption = arbitrary code execution under the invoking user |
| Target weights (~16 GB GGUF + CWENR sidecar) | `model/Qwen3.8-27B-Q4_0.gguf`, sibling `.cwenr` | Public open weights: confidentiality low; integrity high (poisoned weights silently alter every output) |
| Drafter weights (.spec container) | `model/dflash2.spec`, producer tools/pack_dflash.py | Integrity affects speed only, not outputs: every draft is verified against target argmax before emission (run.c:4023-4026, 4081); a hostile drafter can degrade throughput, not corrupt results |
| Output streams (generated token ids, e2e dumps, server reply frames) | stdout, `golden/e2e_*_c/` | Research results corrupted; dumps feed accept/reject decisions |
| Golden reference data | `golden/` dirs, consumed by `bench_q4_gemv` and `tools/accept.py` | Tampering masks regressions in the SIMD kernels |
| Compute/availability | ~15 GiB resident after load (README.md:184) | Long load + fault-in cost per invocation; a crash loop wastes CCD time |
| Secrets | none found in code or tools | n/a |

## Entry points

All enumerated from code; none are network listeners.

| Entry point | Code | Input trust |
|-------------|------|-------------|
| CLI argv: `[MODEL] [IDS_FILE] [N_PREDICT]` plus `-h/--help`, `-d/--draft-tokens/--spec-draft-n-max N`; unknown options and >3 positionals rejected | production `main` run.c:4211-4297 (arg loop 4217-4239); strict range parse `arg_int_range` run.c:2289-2299 | Untrusted paths; numeric args strictly parsed and bounded |
| GGUF container parser (magic, version, kv walk, tensor table, offsets, extents) | `load_gguf` run.c:2837-2910 | Untrusted file from disk |
| CWENR sidecar parser (v2/v3/v4 directory, staleness stamp, offset binding) | `load_cwenr` run.c:3036-3184; invoked from `load_model` run.c:3320-3334; sidecar path from `CWEN_REPACK` env or derived suffix `cwenr_path_for` run.c:2917-2932 | Untrusted file from disk |
| DFlash2 `.spec` drafter container ("DFSP" header, entry walk, offset/nbytes binding, geometry pinning), gated by `CWEN_DFLASH` env | gate run.c:4248-4249; loader `load_dflash` run.c:3204-3318; called from main run.c:4257; producer tools/pack_dflash.py | Untrusted file from disk; cannot alter outputs (see Assets), only parse safely or not |
| Prompt token file: raw little-endian int32 ids | run.c:4265-4291 | Untrusted; per-id range-checked during read run.c:4268-4272; capped at MAX_SEQ; size-vs-count warnings for trailing/oversized files run.c:4280-4286 |
| Stdin server mode (strict bool `CWEN_SERVER`): request frames `<u32 n_prompt><u32 n_gen><n_prompt*i32>`, `0xffffffff` header word = EOF sentinel; replies `<u32 n_out><n_out*i32>` | gate run.c:4246-4247 (`env_bool` run.c:2304-2313); frame parser `server_read_frame` run.c:3659-3677; loop `server_loop` run.c:4158-4173; reply writer run.c:4146-4156 | Untrusted peer controlling stdin/stdout |
| Environment variables: `CWEN_DUMP`, `CWEN_DUMP_LAYERS`, `CWEN_DUMP_LOGITS` (run.c:2315-2339), `CWEN_REPACK` (run.c:2918), `CWEN_SPEC*` knobs incl. `CWEN_SPEC_DEBUG` (`spec_config_init` run.c:3529-3561), `CWEN_DFLASH` (run.c:4248), `CWEN_SERVER` (run.c:4247), `CWEN_OMP_THREADS` (run.c:3605-3615), `OMP_WAIT_POLICY`/`GOMP_SPINCOUNT`/`OMP_PROC_BIND`/`OMP_PLACES` (read by libgomp, defaulted at run.c:3574-3594) | strict parsers: `env_int` run.c:2275-2285, `env_bool` run.c:2304-2313 | Local user control only; set-but-invalid values exit with named errors instead of silent fallbacks |
| One-shot self re-exec: exports missing OMP defaults plus `CWEN_OMPREEXEC` marker, then `execv(/proc/self/exe, argv)` | `cwen_omp_init` run.c:3570-3618 (re-exec block 3574-3594) | Replays own binary and argv once; marker prevents loops; same user, same image |
| Debug dump writer | `dump_f32` run.c:2258-2271 writes `%s/%s.bin` under the `CWEN_DUMP` dir; filenames are code-controlled literals/format strings (e.g. `"logits.bin"`, `"layer%02d.bin"`, run.c:2363-2405) | Reverse flow: process -> filesystem |
| Bench harness golden dir: `meta.json` parsed by strstr/atoi (`read_meta_name` run.c:3699-3712), fixture preflight (`golden_preflight` run.c:3715-3731), x.bin/y_ref.bin floats | bench_q4_gemv main run.c:3732-3796 (built with `-DCWEN_BENCH_Q4_GEMV`, Makefile:165-167) | Trusted-ish (own goldens), parsed defensively anyway |
| Spec microbench: `[MODEL] [ITERS]` only, no external data files | bench_spec main run.c:3830+ (built with `-DCWEN_BENCH_SPEC`, Makefile:171-173) | Model file only |
| libFuzzer harness feeding arbitrary bytes as GGUF, CWENR, `.spec`, AND frame stream; frame stream cross-checked against an independent decoder; unique mkstemp fixture files | tools/fuzz_loader.c:24-29 (harness hooks), :39-64 (mkstemp fixtures), :77-118 (reference decoder + divergence abort), :120-133 (four-way parse, post-load invariant aborts); build/run Makefile:145-160; seeds incl. DFSP heads tools/gen_fuzz_seeds.py:66-148 | Deliberately hostile input |
| Python/shell tooling: HF download with SHA-256 pin, repack with source stamp, drafter packing, benches | tools/download.sh:15 (pin), :17-31 (check), :33-39 (already-present verify), :45-53 (network fetch), :54-58 (post-download delete on mismatch); stamp written tools/repack_q4.py:230,238; drafter packer tools/pack_dflash.py:95-120 (no content pin, see Gaps #3); subprocess runners tools/bench_toks.py, tools/ga_evolve.py:350-356 | Developer-invoked; network only during download |

## Trust boundaries and data flow

1. **filesystem -> loader** (primary): model GGUF/CWENR/.spec and prompt ids cross into
   the process via mmap/read. Validation points: the parsers listed above. This is the
   boundary where untrusted bytes first become pointers and sizes.
2. **supply chain -> disk**: target weights arrive via `hf_hub_download`
   (tools/download.sh:45-53). Content is pinned to a SHA-256 recorded in the script
   (tools/download.sh:15); the check runs on an already-present file (:33-39) and again
   after download, deleting mismatches (:54-58). TLS protects the transfer; the pin
   protects the content choice. The drafter `.spec` is produced locally by
   tools/pack_dflash.py from a safetensors snapshot whose path encodes a commit hash but
   whose bytes are not hash-pinned (:99-105).
3. **parent process -> server_loop**: whoever supplies stdin/stdout of the process.
   Frame validation (run.c:3659-3677) is the only control on the engine side; there is
   no authentication, authorization, or rate limiting at this boundary. The reference
   client bounds itself, not the engine: deadline-bounded reply reads
   (tools/spec_e2e.py:104,130-140) and continuous stderr drain (:88-91,126-127). If this
   process is ever placed behind a socket wrapper, that wrapper becomes the
   authentication point and none exists today.
4. **environment -> process**: env vars select the sidecar path, drafter path, dump
   destination, speculation knobs, and thread counts. Env control implies local
   execution rights, so this is a low privilege-transition boundary; it still feeds
   file-write paths (R4). Special case: when OMP knobs are unset, the process
   re-executes its own image once with exported defaults (run.c:3574-3594) before any
   model I/O.
5. **process -> filesystem (reverse flow)**: `dump_f32` writes into any directory named
   by `CWEN_DUMP`; filenames are code-controlled (run.c:2363-2405), so no traversal,
   but existing files at those names are clobbered. Writes are fail-fast on short
   write/close error (run.c:2263-2270).
6. **build/tooling -> runtime**: `tools/repack_q4.py` produces the CWENR consumed by
   the C parser (and stamps it with the source GGUF's page count, tools/repack_q4.py:230,238);
   `tools/pack_dflash.py` produces the `.spec`; `tools/gen_golden.py` produces goldens
   consumed by verification. A tampered producer poisons everything downstream without
   touching run.c.

Privilege transitions: none documented because none exist; the process never drops or
raises privileges, never forks workers with a different identity. The OMP re-exec
(run.c:3574-3594) re-runs the same binary as the same user with only environment
additions.

Secrets flow: none. No credentials enter, live in, or leave the codebase (verified by
search noted under Scope).

## Threats per boundary

### filesystem -> loader (R1, STRIDE: tampering, information disclosure, elevation of privilege)

A hostile model file is crafted to corrupt memory during parse or bind bad extents for
the kernels. Concrete classes tied to this code:

GGUF container:

- Truncation and cursor overrun past the mmap: countered by the bounds-checked cursor
  `adv()` (run.c:2799-2802) used by all readers.
- Oversized strings smashing stack buffers: countered by u64-cap compare in `rd_str`
  (run.c:2805-2809).
- Deeply nested kv arrays recursing to stack exhaustion: countered by depth limit 64
  (run.c:2810-2811).
- Unknown GGUF versions misparsed: rejected outright, v2/v3 only (run.c:2852-2855).
- Absurd tensor counts/dimension counts causing downstream overflow: countered by
  table cap (run.c:2858), n_dims cap (run.c:2864-2866), and zero/INT_MAX rejection
  (run.c:2867-2875).
- GGUF claiming internal sidecar-only type tags (T_Q4_0R/RS/RSI): rejected at parse;
  they would otherwise pass the extent check with NULL scales and fault the first gemv
  (run.c:2877-2882).
- Tensor offsets pointing outside the map: countered by offset containment plus the
  product-cannot-overflow extent check (run.c:2894-2904).

CWENR sidecar:

- Malformed headers: magic/version/count/dir-fit validation (run.c:3050-3085).
- Entry offsets escaping the map: remainder-form containment that cannot wrap
  (run.c:3101-3106 v3/v4, 3139-3142 v2), exact-size match for v2 payloads
  (run.c:3143-3144), shared entry guards (name known, source is Q4_0, shapes equal:
  run.c:3017-3035).

DFlash `.spec` container:

- Bad magic/version/truncated entries: run.c:3214-3217, 3222-3223.
- Unknown tensor types: capped at the three defined ones (run.c:3231).
- Zero/huge/misaligned dims: positive-int cast bounds plus QK4 row alignment
  (run.c:3235-3236).
- Offsets/nbytes escaping the map: wrap-safe remainder-form containment
  (run.c:3237-3241) plus exact nbytes-vs-declared-geometry match via the shared
  `dflash_tensor_bytes` (run.c:3194-3203, 3242-3247).
- Unknown or duplicate tensor names: rejected (run.c:3251-3254 layer-name parse,
  3267-3268, 3279-3280).
- Declared geometry disagreeing with compile-time drafter dims: every consumed tensor
  pinned exactly after the walk (run.c:3284-3317), including raw-F32 norm rows
  (3300-3308), fixed-block Q8_0 selector tables indexed by token id (3309-3315), and
  the hproj gemv output width (3316-3317).

Residual (all three parsers): kernel-side indexing itself is not independently
validated; it trusts load-time shape pinning (`rebind_layers_from_tens`
run.c:2973-3016 for the target, the geometry block above for the drafter). Types whose
byte size `row_bytes` cannot compute are excluded from the extent check and must never
be read (run.c:2898-2904); any future kernel reading such a tensor reintroduces OOB.
This is the single control carrying several high-impact threats (see Single points of
failure).

### supply chain -> disk (R2, tampering)

An attacker who controls the mirror, proxy cache, or local `model/` file tries to
substitute a weight file. The download path rejects this for target weights: the script
records a SHA-256 for the exact artifact (tools/download.sh:15) and verifies before
keeping an existing file and after every download, removing mismatches (tools/download.sh:33-39,
54-58). Substituting weights therefore requires either a SHA-256 collision, write
access to `model/` after verification (same authority as R5-class local tampering),
or talking a developer into deliberately updating the pin (a documented, visible
decision: tools/download.sh:10-14 comments).

Residual gaps: the CWENR sidecar is bound to its source GGUF only by a page-count
stamp checked when present (run.c:3053-3076); the stamp is optional (legacy untagged
sidecars are trusted as-is) and page counts are trivially matched by a crafted pair.
The drafter `.spec` has no integrity link to anything: tools/pack_dflash.py resolves
its safetensors source by cache path only (:99-105), without content verification.
Impact differs sharply between the two: poisoned target weights (GGUF/sidecar pair)
silently redefine the model's function, while a poisoned drafter cannot change a
single emitted token (every draft must match target argmax to be kept, run.c:4081);
its worst case is pathological proposals that drag decoding below serial speed.

### parent -> server_loop (R3, denial of service, repudiation)

Each accepted frame forces `reset_state()` plus prefill up to MAX_SEQ=4096 tokens and
generation of up to n_gen tokens (run.c:4158-4173, `generate_tokens` run.c:4123-4140).
The framing layer is hardened and spec-pinned: count caps at MAX_SEQ (run.c:3664-3668),
payload drain on rejection so the stream stays aligned (run.c:3633-3645, 3666),
EOF sentinel handling (run.c:3626-3631, 3663), token range validation inside the shared
parser (run.c:3647-3652, 3669-3674), and differential fuzzing against an independent
decoder written from the protocol text (tools/fuzz_loader.c:77-118). What remains
unmitigated is economic: no per-peer quota, no backpressure, and no audit record.
A hostile pipe owner can pin a CPU core indefinitely, and afterwards there is nothing
to investigate beyond stderr progress/validation lines (run.c:3665, 3672, 4149, 4262).

### supply chain -> goldens / tooling producers (R5, tampering)

Goldens and CWENR/.spec artifacts are plain files with no integrity metadata; whoever
can write `golden/` decides what `make verify` accepts (Makefile:184-195). Impact
limited to development-time wrong conclusions, hence lower rank. Tooling-side control
that exists: engine stdout is decoded strictly as decimal token ids by the gate
runners, so a corrupted stream fails the comparison instead of silently shrinking it
(tools/bench_toks.py:23-32; reused by tools/spec_check.py:24).

## Mitigations map (existing controls)

| Control | Covers | Reference |
|---------|--------|-----------|
| Bounds-checked read cursor on every GGUF access | truncation/OOB reads | run.c:2799-2802 |
| String length cap before memcpy | stack smash via kv keys/tensor names | run.c:2805-2809 |
| KV nesting depth limit | stack exhaustion | run.c:2810-2811 |
| Version/table-cap/dim rejection + non-overflowing extent math | integer overflow to OOB | run.c:2852-2855, 2858, 2864-2875, 2894-2904 |
| Reserved internal type tags rejected at GGUF parse | NULL-scales fault on first gemv | run.c:2877-2882 |
| Tensor offset containment vs mmap length | arbitrary-offset OOB | run.c:2894-2904 |
| Exact shape pinning of every consumed target weight/vector | kernel OOB from mismatched dims | run.c:2973-3016 |
| Weight type contracts (F32 vs matmul types, embed dispatch, lm_head dispatch) | garbage-type misinterpretation, silent stale logits | run.c:2936-2953, 2975-2988 |
| CWENR header validation (magic/version/count/dir-fit) | malformed sidecar headers | run.c:3050-3085 |
| CWENR per-entry OOB + exact-size checks (remainder form) | sidecar offset attacks | run.c:3101-3106, 3139-3144 |
| CWENR staleness stamp vs source GGUF page count | stale sidecar served over changed model | run.c:3053-3076; producer tools/repack_q4.py:230,238 |
| DFlash header/version/truncation/type/dim/offset/nbytes validation | malformed .spec containers | run.c:3214-3247 |
| DFlash unknown-name/duplicate rejection + full geometry pinning | drafter kernel OOB from mismatched shapes | run.c:3251-3280, 3284-3317 |
| Lossless draft verify (draft kept iff equals target argmax) | poisoned drafter altering outputs | run.c:4023-4026, 4081 |
| SHA-256 content pin on downloaded target weights | substituted/poisoned downloads | tools/download.sh:15,33-39,54-58 |
| Hard token-range check at use site (not just parse sites) | OOB embedding lookup via token id | run.c:1675-1678, 2527 |
| Shared frame parser: count caps, reject-drain alignment, EOF sentinel | frame-driven buffer overrun, stream desync | run.c:3626-3677 |
| Strict CLI integer parsing with named errors | argv misuse/silent fallbacks | run.c:2289-2299 |
| Strict env parses (`env_int`, `env_bool`) | silent atoi fallbacks, truthiness bugs (CWEN_SERVER=0 starting the server) | run.c:2275-2313 |
| Speculation knob range validation, parsed once pre-load | absurd draft/cooldown configs; no per-frame env re-read | run.c:3529-3561 |
| One-shot marked self re-exec (loop-proof via `CWEN_OMPREEXEC`) | OMP env never reaching libgomp constructor | run.c:3574-3594 |
| Fail-fast output writes (reply frames, dumps) | silent token/data loss on I/O failure | run.c:2258-2271, 4146-4156 |
| Golden fixture preflight (existence, regular file, minimum size) | late confusing failures on stale/missing goldens | run.c:3715-3731 |
| Defensive meta.json parsing in bench (bounded copy, alloc checks) | hostile golden dir | run.c:3699-3712 |
| libFuzzer + ASan/UBSan over all four untrusted parsers (GGUF, CWENR, .spec, frames) | regression detection on parser bugs | tools/fuzz_loader.c:120-133, Makefile:145-160, tools/gen_fuzz_seeds.py:66-148 |
| Differential fuzzing of frame parser vs independent spec-written decoder | parser/protocol drift becoming a desync bug | tools/fuzz_loader.c:71-118 |
| Post-load invariant aborts in harness (pointer escape, bound-tensor containment) | silent bad binds surviving parse | tools/fuzz_loader.c:131; run.c:3416-3434 |
| mkstemp fixture files in harness | symlink-clobber of fixed /tmp names (CWE-377/59) | tools/fuzz_loader.c:39-64 |
| Strict decimal-token decode of engine stdout in gate runners | corrupted stream shrinking gate comparisons | tools/bench_toks.py:23-32 (single copy, reused by tools/spec_check.py:24) |
| Deadline-bounded reply reads + continuous stderr drain in reference client | hung engine wedging the client / blocking mid-frame on pipe fill | tools/spec_e2e.py:88-107,126-140 |

Claims made elsewhere were checked: DESIGN.md "Security & operability" (DESIGN.md:393-401)
says local-only CLI, bounds-checked offsets, fuzzing over GGUF/CWENR/DFlash-.spec/frame
parsing, the dump/server/spec env surfaces, and page-in behavior; all match the code
above. No security claim was found that the code contradicts.

Single points of failure: load-time shape pinning (`rebind_layers_from_tens`
run.c:2973-3016 plus the drafter geometry block run.c:3284-3317) is the one control
class standing between hostile files and every kernel-side memory access; the SHA-256
pin (tools/download.sh:15) is the one control standing between the delivery path and
every experiment downstream. Both are listed here so their failure modes get first
attention in sec-review passes.

## Gaps (unmitigated, ranked)

1. R3: no quota, backpressure, or audit trail on server frames (run.c:4158-4173).
   Exploitability high for anyone already holding stdin; impact limited to the
   process's own resources.
2. R1 residual: `row_bytes==0` types bypass the extent check by design and depend on a
   "never read" invariant enforced nowhere mechanically (run.c:2898-2904); kernel-side
   indexing relies entirely on load-time shape pinning. Fuzzing covers the four parsers,
   not the gemv kernels themselves.
3. R2 residual: CWENR-to-GGUF binding is a page-count stamp, present optionally
   (run.c:3053-3076); a crafted GGUF+sidecar pair with matching page count binds
   cleanly and redefines the function. The drafter `.spec` has no binding at all and
   its producer does not content-pin the safetensors input (tools/pack_dflash.py:99-105);
   bounded by the lossless verify to throughput damage only (run.c:4081).
4. R5: goldens and CWENR/.spec artifacts carry no integrity metadata; producers and
   consumers trust the filesystem.
5. No SECURITY.md exists: there is no documented disclosure contact or supported-version
   statement, and no path from "vulnerability reported" to "fix shipped". Not invented
   here; recording its absence.

## Abuse cases

Single-user research CLI, so classic multi-tenant abuse does not apply. Documented
hostile-but-authorized scenarios:

- **Compute squandering via valid frames**: a peer holding the server's stdin sends
  repeated maximal frames (`n_prompt = n_gen = MAX_SEQ`); each is individually legal,
  passes validation (run.c:3664-3674), and costs seconds of full-model compute plus
  recurrent-state scrubbing in `reset_state` (run.c:3520-3524). No quota intervenes.
- **Pinned-artifact swap (function-defining)**: someone with write access to `model/`
  replaces the GGUF and regenerates or retags the sidecar so the page-count stamp
  matches (run.c:3053-3076); the loader cannot distinguish this from a legitimate
  repack. Every downstream result is silently whatever the attacker chose. The
  download-time SHA-256 pin (tools/download.sh:15) does not help because it runs only
  in the download script.
- **Drafter sabotage (throughput-only)**: same actor swaps `model/dflash2.spec` for a
  crafted container proposing one-token drafts every cycle; all loads and verifies
  cleanly (geometry checks run.c:3284-3317 do not constrain proposal quality), every
  draft is rejected at the argmax compare (run.c:4081), and decoding pays snapshot +
  replay overhead below serial speed. Outputs stay byte-identical.
- **Golden laundering**: someone with write access to `golden/` replaces reference
  floats so a broken kernel change still prints PASS (Makefile:184-195 compares only
  against those files).

Trust placed in client-side enforcement: none found; there is no client-facing policy
layer to bypass.

## Response readiness (note only)

- Audit trail: none. Server mode emits only stderr progress/validation lines
  (run.c:3665, 3672, 4149, 4262); there is no request log to investigate incidents
  from.
- Disclosure path: absent (no SECURITY.md). Fuzz crashes land in `fuzz_out/artifacts/`
  (Makefile:157-160), which is the closest thing to a vulnerability intake today.
