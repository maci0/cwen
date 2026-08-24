#!/usr/bin/env bash
# Build run_prof, perf-record 8 tokens, emit SVG flamecharts + index.html
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=golden/profile/flames
# Pinned upstream commit (master head 2024-10-20); bump deliberately, not blindly.
FG_REPO=https://github.com/brendangregg/FlameGraph.git
FG_REF=41fee1f99f9276008b7cd112fca19dc3ea84ac32
FG="${FLAMEGRAPH_DIR:-$HOME/.cache/cwen-profile/FlameGraph}"
mkdir -p "$OUT"
if [[ ! -x "$FG/flamegraph.pl" ]]; then
  # An interrupted clone leaves a partial dir that blocks every re-clone;
  # without flamegraph.pl it is not a usable checkout, so replace it.
  rm -rf "$FG"
  mkdir -p "$(dirname "$FG")"
  git clone --quiet "$FG_REPO" "$FG"
  git -C "$FG" checkout --quiet --detach "$FG_REF"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}" OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
make run_prof
python3 tools/mk_prompt_ids.py
perf record -g --call-graph dwarf,16384 -F 997 -o "$OUT/perf.data" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf prompt1.ids "${1:-8}" >/dev/null
perf script -i "$OUT/perf.data" | "$FG/stackcollapse-perf.pl" --all > "$OUT/out.folded"
echo "folded -> $OUT (run prior python dashboard recipe or re-open session script)"
"$FG/flamegraph.pl" --title "cwen CPU" --width 1920 --colors hot --bgcolors '#0d1117' \
  "$OUT/out.folded" > "$OUT/quick-flame.svg"
echo "wrote $OUT/quick-flame.svg"
