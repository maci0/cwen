# GNU Make predefines CC=cc, so ?= never fires; honor an explicit override
# (`make CC=clang`), otherwise default to gcc as documented in the README.
ifeq ($(origin CC),default)
  CC := gcc
endif
# Default: AVX2 (Zen3+). Max speed on Zen4/5: make AVX512=1
# -fno-math-errno/-fno-trapping-math: free libm; -flto: cross-TU inline (single file still helps IPO)
# -fomit-frame-pointer: one more reg on x86-64 hot loops
CFLAGS  ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -Wshadow=compatible-local -Wcast-align \
           -Wstrict-prototypes -Wmissing-prototypes -Wdouble-promotion \
           -Wvla -Wwrite-strings -Wredundant-decls -Wundef -Wnull-dereference \
           -Werror -fopenmp -mavx2 -mfma -mf16c \
           -fno-math-errno -fno-trapping-math -fomit-frame-pointer -flto \
           -falign-functions=32 -falign-loops=32 \
           -ffile-prefix-map=$(CURDIR)=.
LDFLAGS ?= -fopenmp -lm -flto
MODEL   ?= model/Qwen3.8-27B-Q4_0.gguf
PY      := .venv/bin/python

ifeq ($(AVX512),1)
  # znver5: VNNI/BF16/VBMI available; we use FMA512 + wider dequant today
  CFLAGS += -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -march=native -DCWEN_AVX512
endif

# Diagnostics raised after the tree proved clean under -Werror on gcc across
# all build configs (default, AVX512=1, bench targets). Portable classes stay
# unguarded so clang builds pick them up too; GCC-only classes are guarded to
# keep unknown-warning-option from tripping -Werror under `make CC=clang`.
CFLAGS += -Wcast-qual -Wformat=2 -Wdate-time
ifeq ($(findstring clang,$(shell $(CC) --version)),)
CFLAGS += -Wlogical-op -Wduplicated-cond -Wduplicated-branches \
          -Wjump-misses-init -Walloc-zero -Warith-conversion \
          -Warray-bounds=2 -Wuse-after-free=3
endif

# Profiling build used by tools/profile_flames.sh and tools/profile_lowlevel.sh:
# debug info + frame pointers for perf stacks, always the AVX512 path.
PROF_CFLAGS  = -O3 -std=c11 -g -fno-omit-frame-pointer -Werror -fopenmp \
               -mavx2 -mfma -mf16c \
               -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vnni -march=native \
               -fno-math-errno -fno-trapping-math -DCWEN_AVX512 \
               -ffile-prefix-map=$(CURDIR)=.
PROF_LDFLAGS = -fopenmp -lm

# Rebuild when flags change (AVX512=1 toggle, CFLAGS/CC edit), not just sources.
# The stamp file records the last-used command line; dependents rebuild iff it moved.
BUILD_STAMP      := .build-flags
BUILD_STAMP_PROF := .build-flags-prof

$(BUILD_STAMP): FORCE
	@printf '%s\n' 'cc=$(CC) | cflags=$(CFLAGS) | ldflags=$(LDFLAGS)' > $@.tmp
	@if cmp -s $@ $@.tmp; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

$(BUILD_STAMP_PROF): FORCE
	@printf '%s\n' 'prof cc=$(CC) | cflags=$(PROF_CFLAGS) | ldflags=$(PROF_LDFLAGS)' > $@.tmp
	@if cmp -s $@ $@.tmp; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

FORCE:
.PHONY: FORCE

.DEFAULT_GOAL := all

.PHONY: all help setup lock pycheck clean check-loc lint gate golden verify verify-tsan verify-reproducible e2e verify-e2e e2e-full-c e2e-full-py verify-e2e-full ga bench-toks bench-spec idea-bench repack fuzz-seeds fuzz-run buildinfo bench-q4_gemv run_prof fuzz

all: run

# Discoverability: every contributor entry point, one line each.
help:
	@echo "cwen dev commands"
	@echo "  make setup        create .venv + install Python deps (hash-checked requirements.lock)"
	@echo "  make lock         regenerate requirements.lock from requirements.txt"
	@echo "  make              build ./run; add AVX512=1 for Zen4/5 peak"
	@echo "  make verify       gemv goldens vs $(MODEL) (fast correctness loop)"
	@echo "  make verify-tsan  gemv goldens under ThreadSanitizer (needs clang)"
	@echo "  make verify-reproducible  rebuild-and-compare: bit-identical ./run across path/locale/TZ"
	@echo "  make gate         full gate: goldens + pinned decode chain (tools/test_speed_gates.sh)"
	@echo "  make golden       regenerate the goldens verify checks (after weight change)"
	@echo "  make bench-q4_gemv build the gemv bench alone"
	@echo "  ./bench_q4_gemv MODEL golden/DIR ITERS   single-kernel check"
	@echo "  make bench-spec   build the block-speculation microbench"
	@echo "  ./bench_spec MODEL ITERS   gemvb/snapshot/block-sweep numbers"
	@echo "  make verify-e2e       4-layer residual compare C vs numpy ref"
	@echo "  make verify-e2e-full  full-stack compare (slow: builds both dumps)"
	@echo "  make e2e-full-c       C-side dump only (golden/e2e_full_c); used to re-pin the decode gate"
	@echo "  make e2e-full-py      numpy-ref dump only (golden/e2e_full)"
	@echo "  make repack       Q4_0 -> CWENR sidecar for $(MODEL)"
	@echo "  tools/download.sh fetch the GGUF into model/"
	@echo "  make ga           GA tune cwen_tune.h (long)"
	@echo "  make bench-toks   end-to-end tok/s of ./run"
	@echo "  make idea-bench   A/B every CWEN_IDEA_* flag"
	@echo "  make fuzz / fuzz-seeds / fuzz-run   loader fuzzing (needs clang)"
	@echo "  make run_prof     profiling build for tools/profile_*.sh"
	@echo "  make buildinfo    print toolchain + flags snapshot"
	@echo "  make check-loc    run.c line count (informational)"
	@echo "  make lint         static analysis: shellcheck + cppcheck + ruff (venv) + mypy"
	@echo "  make clean        remove built binaries"

# One-time dev env: project-local .venv via uv, installed from the hash-checked
# lock so every wheel (direct + transitive) is verified against its sha256.
setup:
	@test -f requirements.lock || { echo "cwen: requirements.lock missing; run 'make lock' first" >&2; exit 1; }
	@command -v uv >/dev/null || { echo "cwen: 'uv' not found on PATH; install it first (https://docs.astral.sh/uv/getting-started/installation/)" >&2; exit 1; }
	@test -x .venv/bin/python || uv venv .venv
	uv pip install --python .venv/bin/python -r requirements.lock
	@echo "dev env ready: .venv"

# Regenerate the lock after editing requirements.txt: direct pins carry over,
# transitives are frozen at constraints.txt so re-resolving cannot drift the
# validated tree. Resolution targets the pinned interpreter from
# .python-version so the lock does not depend on whichever Python the invoking
# host happens to have. Diff-review the moved versions before committing;
# hashes are rewritten for every entry.
lock:
	uv pip compile requirements.txt --constraints constraints.txt \
	  --generate-hashes \
	  --python-version "$$(cat .python-version)" -o requirements.lock

# Preflight: name the problem instead of ".venv/bin/python: No such file or directory".
pycheck:
	@test -x .venv/bin/python || { echo "cwen: no .venv found; run 'make setup' first"; exit 1; }

run: run.c cwen_tune.h $(BUILD_STAMP)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# Profiling binary: same flags the profile scripts used inline, now in one place.
run_prof: run.c cwen_tune.h $(BUILD_STAMP_PROF)
	$(CC) $(PROF_CFLAGS) -o $@ $< $(PROF_LDFLAGS)

# Recorded build environment: toolchain + flags snapshot for reproducibility.
buildinfo:
	@printf 'cc=%s (%s)\n' "$$($(CC) -dumpfullversion 2>/dev/null || $(CC) -dumpversion)" "$(CC)"
	@printf 'cflags=%s\n' '$(CFLAGS)'
	@printf 'ldflags=%s\n' '$(LDFLAGS)'
	@printf 'model=%s\n' '$(MODEL)'

# Informational only (no hard cap).
check-loc:
	@echo "run.c: $$(wc -l < run.c) lines"

# Static analysis gate: shellcheck + cppcheck(warning,performance,portability)
# + ruff (lint + format check) + mypy. All must stay green.
# shellcheck/cppcheck/mypy are system packages; ruff comes from requirements.txt via make setup.
lint: | pycheck
	@shellcheck -S style tools/*.sh
	@cppcheck -q --enable=warning,performance,portability --check-level=exhaustive \
	  --std=c11 --platform=unix64 --error-exitcode=1 run.c tools/fuzz_loader.c
	@$(PY) -m ruff check tools/
	@$(PY) -m ruff format --check tools/
	@mypy
	@echo "lint OK"

clean:
	rm -f run bench_q4_gemv bench_spec fuzz_loader run_prof bench_q4_gemv_tsan \
	  $(BUILD_STAMP) $(BUILD_STAMP_PROF)

# libFuzzer harness for every binary parser surface (GGUF, CWENR sidecar,
# DFlash .spec, request frames, NGC2 cache round trip); needs clang.
FUZZ_CC  ?= clang
FUZZ_OUT ?= fuzz_out
fuzz: run.c tools/fuzz_loader.c cwen_tune.h
	$(FUZZ_CC) -O1 -g -std=c11 -mavx2 -mfma -mf16c \
	  -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined \
	  -DCWEN_FUZZ_LOADER -o fuzz_loader run.c tools/fuzz_loader.c -lm

# Deterministic hand-built format heads (GGUF / CWENR / DFlash .spec) and
# request-frame streams for the mutator.
fuzz-seeds: | pycheck
	$(PY) tools/gen_fuzz_seeds.py --out tools/fuzz_corpus

# Short guided run over seeds + grown corpus; artifacts under fuzz_out/.
fuzz-run: fuzz
	mkdir -p $(FUZZ_OUT)/corpus $(FUZZ_OUT)/artifacts tools/fuzz_corpus
	./fuzz_loader $(FUZZ_OUT)/corpus tools/fuzz_corpus \
	  -artifact_prefix=$(FUZZ_OUT)/artifacts/ -max_total_time=60 \
	  -rss_limit_mb=4096 -malloc_limit_mb=2048 -timeout=5 -close_fd_mask=3


# Bench harness compiles only the gemv slice of run.c; the rest of the file
# is unused by construction here, so drop just that one warning class.
bench-q4_gemv: run.c cwen_tune.h $(BUILD_STAMP)
	$(CC) $(CFLAGS) -Wno-unused-function -DCWEN_BENCH_Q4_GEMV \
	  -o bench_q4_gemv run.c $(LDFLAGS)

# Block-speculation microbench: gemvb scaling + correctness, snapshot
# rollback cost, verify-block vs serial sweep. Same TU, spec slice.
bench-spec: run.c cwen_tune.h $(BUILD_STAMP)
	$(CC) $(CFLAGS) -Wno-unused-function -DCWEN_BENCH_SPEC \
	  -o bench_spec run.c $(LDFLAGS)

# The verify set: four layer-0 gemvs + output.weight (Q6_K). ffn_gate is the
# one interleaved-pair tensor (CWENR v4 T_Q4_0RSI side A): without it no
# golden exercises the RSI read path.
golden: | pycheck
	$(PY) tools/gen_golden.py --out golden --tensors \
	  blk.0.attn_gate.weight blk.0.ffn_gate.weight blk.0.ffn_down.weight \
	  blk.0.ssm_out.weight output.weight

# Same set the durable gate iterates: four layer-0 gemvs + output.weight (Q6_K)
verify: bench-q4_gemv | pycheck
	@ndirs=0; \
	for d in golden/blk_* golden/output_weight; do \
	  [ -d "$$d" ] || continue; \
	  ndirs=$$((ndirs+1)); \
	  ./bench_q4_gemv $(MODEL) $$d 3 || exit 1; \
	  $(PY) tools/accept.py $$d || exit 1; \
	done; \
	if [ "$$ndirs" -eq 0 ]; then \
	  echo "verify: no golden dirs matched; run 'make golden' first" >&2; exit 1; \
	fi
	@echo "gemv goldens OK"

# Same golden set as 'verify' under ThreadSanitizer (needs clang): the bench
# binary is instrumented, libgomp's own suspend/resume traffic is silenced via
# tools/tsan-omp.supp; any other report exits nonzero and fails the target.
bench-q4_gemv_tsan: run.c cwen_tune.h
	clang -O1 -g -std=c11 -fopenmp -mavx2 -mfma -mf16c \
	  -fsanitize=thread -DCWEN_BENCH_Q4_GEMV -Wno-unused-function \
	  -o bench_q4_gemv_tsan $< -lm

verify-tsan: bench-q4_gemv_tsan
	@ndirs=0; \
	for d in golden/blk_* golden/output_weight; do \
	  [ -d "$$d" ] || continue; \
	  ndirs=$$((ndirs+1)); \
	  TSAN_OPTIONS="suppressions=tools/tsan-omp.supp" \
	    ./bench_q4_gemv_tsan $(MODEL) $$d 1 || exit 1; \
	done; \
	if [ "$$ndirs" -eq 0 ]; then \
	  echo "verify-tsan: no golden dirs matched; run 'make golden' first" >&2; exit 1; \
	fi
	@echo "gemv goldens OK (tsan)"

# Independent rebuild-and-compare gate: same source, different build dir,
# locale, TZ, and SOURCE_DATE_EPOCH must produce a byte-identical ./run.
verify-reproducible:
	tools/check_reproducible.sh

# Durable speed+correctness gate (AVX512): gemv goldens + pinned argmax chain
# + run-to-run determinism. Builds its own binaries; logs under outputs/gates/.
gate:
	tools/test_speed_gates.sh outputs/gates

# GA + symbolic tune: tools/ga_evolve.py writes cwen_tune.h
ga: | pycheck
	$(PY) tools/ga_evolve.py --gens 6 --pop 12 --iters 5 --trials 1 --avx512

# End-to-end tok/s (needs built run with AVX512 for peak)
bench-toks: run | pycheck
	$(PY) tools/bench_toks.py --trials 3 --ns 2,4,8

# A/B every CWEN_IDEA_* flag (gemv goldens + optional --toks)
idea-bench: | pycheck
	$(PY) tools/idea_bench.py --trials 3 --omp 16

# Offline Q4_0 → CWENR v4 (solo split + interleaved gate/up, k/v pairs)
repack: | pycheck
	$(PY) tools/repack_q4.py $(MODEL) --sidecar-version 4

# Partial e2e (4 layers residual)
e2e: run | pycheck
	$(PY) tools/e2e_ref.py --layers 4 --token 248044 --out golden/e2e
	rm -rf golden/e2e_c && mkdir -p golden/e2e_c
	$(PY) tools/mk_prompt_ids.py
	CWEN_DUMP=golden/e2e_c CWEN_DUMP_LAYERS=4 ./run $(MODEL) prompt1.ids 0

verify-e2e: e2e | pycheck
	$(PY) tools/compare_e2e.py --ref golden/e2e --c golden/e2e_c
	@echo "e2e goldens OK"

# Full stack: 64 layers + logits + 2 decode tokens
# C dump is fast (~5–20s). Python ref is slower (numpy); skip if golden/e2e_full already present.
e2e-full-c: run | pycheck
	rm -rf golden/e2e_full_c && mkdir -p golden/e2e_full_c
	$(PY) tools/mk_prompt_ids.py
	CWEN_DUMP=golden/e2e_full_c ./run $(MODEL) prompt1.ids 2 > golden/e2e_full_c/tokens.txt

e2e-full-py: | pycheck
	$(PY) tools/e2e_ref.py --layers 0 --token 248044 --logits --gen 2 --out golden/e2e_full

verify-e2e-full: | pycheck
	$(PY) tools/compare_e2e_full.py
	@echo "e2e-full OK"
