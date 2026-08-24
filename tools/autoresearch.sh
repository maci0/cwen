#!/usr/bin/env bash
# Decode-throughput bench for the Qwen3.8-27B autoresearch loop.
# Prints DECODE_TOK_S=... and TOKENS=... for the logger.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
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

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT
"$PY" tools/bench_toks.py \
  --run "$RUN_BIN" --model "$MODEL" --prompt prompt1.ids \
  --ns "$NS" --trials "$TRIALS" --omp "$OMP" | tee "$LOG"

# parse the decode-only line by its ASCII prefix; its tail carries non-ASCII
# glyphs, so never byte-match past the prefix
decode=$(grep '^decode-only' "$LOG" | grep -Eo '[0-9.]+ tok/s' | tail -1 \
         | grep -Eo '^[0-9.]+' | head -1 || true)
if [[ -z "${decode:-}" ]]; then
  decode=$("$PY" - <<'PY' "$LOG"
import re,sys
t=open(sys.argv[1],encoding="utf-8",errors="replace").read()
m=re.search(r"^decode-only.*\s([0-9.]+)\s+tok/s\s*$", t, re.M)
print(m.group(1) if m else "nan")
PY
)
fi
toks=$(grep -Eo 'last tokens: \[[0-9, ]+\]' "$LOG" | tail -1 || true)
echo "DECODE_TOK_S=${decode}"
echo "TOKENS_LINE=${toks}"
