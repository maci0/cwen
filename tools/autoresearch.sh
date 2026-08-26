#!/usr/bin/env bash
# Decode-throughput bench for the Qwen3.8-27B autoresearch loop.
# Prints DECODE_TOK_S=... and TOKENS=... for the logger.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CC="${CC:-gcc}"
AVX512="${AVX512:-1}"
OMP="${OMP:-16}"
TRIALS="${TRIALS:-3}"
NS="${NS:-2,8}"
MODEL="${MODEL:-model/Qwen3.8-27B-Q4_0.gguf}"
RUN_BIN="${RUN_BIN:-run}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing $MODEL" >&2
  exit 1
fi
PY=.venv/bin/python
if [[ ! -x "$PY" ]]; then
  echo "cwen: no .venv found; run 'make setup' first" >&2
  exit 1
fi

make AVX512="$AVX512" CC="$CC" >/dev/null
"$PY" tools/mk_prompt_ids.py

export OMP_NUM_THREADS="$OMP"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"
export GOMP_SPINCOUNT="${GOMP_SPINCOUNT:-100}"

mkdir -p .scratch
LOG=$(mktemp "$ROOT/.scratch/autoresearch.XXXXXX")
trap 'rm -f "$LOG"' EXIT
"$PY" tools/bench_toks.py \
  --run "$RUN_BIN" --model "$MODEL" --prompt prompt1.ids \
  --ns "$NS" --trials "$TRIALS" --omp "$OMP" | tee "$LOG"

# Anchor on the ASCII prefix and the trailing "tok/s": the middle of the line
# carries non-ASCII glyphs, so byte-matching across it is not safe. LC_ALL=C
# keeps sed on bytes rather than failing on invalid multibyte sequences.
decode=$(LC_ALL=C sed -n \
  's/^decode-only.*[[:space:]]\([0-9.][0-9.]*\)[[:space:]][[:space:]]*tok\/s[[:space:]]*$/\1/p' \
  "$LOG" | tail -1)
decode=${decode:-nan}
toks=$(grep -Eo 'last tokens: \[[0-9, ]+\]' "$LOG" | tail -1 || true)
echo "DECODE_TOK_S=${decode}"
echo "TOKENS_LINE=${toks}"
