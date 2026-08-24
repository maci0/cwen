#!/usr/bin/env bash
set -e
cd /home/maci/Desktop/Research/cwen
OUT=outputs/spec/profile
mkdir -p "$OUT"

gcc -O2 -g -fno-omit-frame-pointer -std=c11 \
    -Wno-unused-function -fopenmp -mavx512f -mavx512bw -mavx512vl \
    -mavx512dq -mavx512vnni -march=native \
    -DCWEN_AVX512 -o run_prof run.c -lm

echo "recording 60s @ 99Hz: DFlash2 workload"
perf record -F 99 -g --call-graph dwarf -o "$OUT/perf.data" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf spec_rep.ids 128 -d 8
echo "done"
