#!/usr/bin/env bash
# Record a perf profile of the DFlash2 speculative workload. Feed the result
# to tools/profile_flames.sh to render the flamegraph.
set -euo pipefail

root=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
[[ -f run.c ]] || { echo "$0: run.c not found under $root" >&2; exit 1; }

OUT=outputs/spec/profile
mkdir -p "$OUT"

# -O2 with frame pointers, not the -O3 production flags: perf needs the stacks,
# and profiling a debug build would move the hot path entirely.
gcc -O2 -g -fno-omit-frame-pointer -std=c11 \
    -Wno-unused-function -fopenmp -mavx512f -mavx512bw -mavx512vl \
    -mavx512dq -mavx512vnni -march=native \
    -DCWEN_AVX512 -o run_prof run.c -lm

echo "recording @ 99Hz: DFlash2 workload"
perf record -F 99 -g --call-graph dwarf -o "$OUT/perf.data" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf spec_rep.ids 128 -d 8
echo "done: $OUT/perf.data"
