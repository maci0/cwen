#!/usr/bin/env bash
# Durable speed+correctness gate: real gemv binary vs on-disk goldens + short decode.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${1:-.}"
mkdir -p "$OUT"

make AVX512=1 >/dev/null
make AVX512=1 bench-q4_gemv >/dev/null

fail=0
echo "=== gemv goldens $(date -Iseconds) ===" | tee "$OUT/test_gemv.log"
ndirs=0
# Same set make verify iterates: four layer-0 gemvs (incl. the interleaved
# ffn_gate pair side) + output.weight (Q6_K);
# skipping the lm_head dir would leave the Q6_K kernel path untested here.
for d in golden/blk_* golden/output_weight; do
  [ -d "$d" ] || continue
  ndirs=$((ndirs+1))
  ./bench_q4_gemv model/Qwen3.8-27B-Q4_0.gguf "$d" 10 | tee -a "$OUT/test_gemv.log" || fail=1
done

# assert all PASS (log text and exit status); empty golden set must not pass vacuously
if [ "$ndirs" -eq 0 ]; then
  echo "no golden dirs matched golden/blk_*" >&2
  exit 1
fi
if grep -q '^FAIL' "$OUT/test_gemv.log"; then
  echo "gemv FAIL" >&2
  exit 1
fi
npass=$(grep -c '^PASS' "$OUT/test_gemv.log" || true)
echo "pass_lines $npass/$ndirs"
if [ "$npass" -ne "$ndirs" ]; then
  echo "gemv PASS count $npass != golden dirs $ndirs" >&2
  exit 1
fi
if [ "$fail" -ne 0 ]; then
  echo "gemv bench exited nonzero" >&2
  exit 1
fi

# short decode: pinned argmax chain + run-to-run determinism.
# EXPECT is the recorded Qwen3.8-27B argmax chain for BOS 248044 (see
# autoresearch.md / CHANGELOG.md). Re-pin only after a weight change, via
# `make e2e-full-c` (a self-consistent run alone is not evidence).
EXPECT="17 15 17 15 95859 17 15 17"
python3 tools/mk_prompt_ids.py
t1=$(./run model/Qwen3.8-27B-Q4_0.gguf prompt1.ids 8 2>"$OUT/test_run.err") \
  || { echo "run failed"; cat "$OUT/test_run.err" >&2; exit 1; }
t2=$(./run model/Qwen3.8-27B-Q4_0.gguf prompt1.ids 8 2>/dev/null) \
  || { echo "run (repeat) failed"; exit 1; }
grep -Eq '^cwenr:.*(bound|split)' "$OUT/test_run.err" \
  || { echo "cwenr sidecar not bound; gate did not exercise CWENR path" >&2; exit 1; }
echo "tokens1=$t1" | tee "$OUT/test_tokens.log"
echo "tokens2=$t2" | tee -a "$OUT/test_tokens.log"
if [ "$t1" != "$EXPECT" ]; then
  echo "token chain mismatch: got [$t1] want [$EXPECT]" >&2
  exit 1
fi
if [ "$t1" != "$t2" ]; then
  echo "token nondeterminism: [$t1] vs [$t2]" >&2
  exit 1
fi
echo "OK speed gates"
exit 0
