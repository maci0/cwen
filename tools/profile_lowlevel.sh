#!/usr/bin/env bash
# Low-level profile of cwen decode: perf + optional bpftrace.
# Usage: tools/profile_lowlevel.sh [n_predict]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
N="${1:-4}"
OUT=golden/profile
mkdir -p "$OUT"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_DYNAMIC=false

echo "== build run_prof (-g -fno-omit-frame-pointer) =="
make run_prof

python3 tools/mk_prompt_ids.py

echo "== perf stat (generic) =="
perf stat -e cycles,instructions,branches,branch-misses,page-faults,\
L1-dcache-loads,L1-dcache-load-misses \
  -o "$OUT/perf_stat.txt" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf prompt1.ids "$N" \
  >"$OUT/run_stdout.txt" 2>"$OUT/run_stderr.txt" || true
cat "$OUT/perf_stat.txt"

echo "== perf stat (AMD Zen fills) =="
perf stat -e cycles,instructions,\
ls_any_fills_from_sys.dram_io_all,ls_any_fills_from_sys.local_l2,\
ls_any_fills_from_sys.remote_cache,ls_dispatch.ld_dispatch,\
ls_l1_d_tlb_miss.all,ls_l1_d_tlb_miss.all_l2_miss \
  -o "$OUT/perf_amd_mem.txt" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf prompt1.ids "$N" \
  >/dev/null 2>&1 || true
cat "$OUT/perf_amd_mem.txt"

echo "== perf record (dwarf stacks) =="
perf record -g --call-graph dwarf,8192 -F 999 -o "$OUT/perf.data" -- \
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf prompt1.ids "$N" \
  >/dev/null 2>&1
perf report -i "$OUT/perf.data" --stdio --no-children -n --percent-limit 0.5 \
  | tee "$OUT/perf_report.txt" | head -50

if command -v bpftrace >/dev/null && sudo -n true 2>/dev/null; then
  echo "== bpftrace GOMP_parallel timing (8s window) =="
  BPF_PROG='
uprobe:/usr/lib/libgomp.so.1:GOMP_parallel { @enter[tid]=nsecs; @calls=count(); }
uretprobe:/usr/lib/libgomp.so.1:GOMP_parallel {
  if (@enter[tid]) { @tot=sum(nsecs-@enter[tid]); @lat=hist(nsecs-@enter[tid]); delete(@enter[tid]); }
}
interval:s:8 { print(@calls); print(@tot); print(@lat); exit(); }
'
  # redirects must run as root: sudo does not elevate the caller's redirections
  sudo -n sh -c 'exec bpftrace -e "$1" >"$2" 2>"$3"' sh \
    "$BPF_PROG" "$OUT/bpf_gomp.txt" "$OUT/bpf_gomp.err" &
  BP=$!
  sleep 0.4
  ./run_prof model/Qwen3.8-27B-Q4_0.gguf prompt1.ids 3 >/dev/null 2>&1 || true
  wait "$BP" 2>/dev/null || true
  cat "$OUT/bpf_gomp.txt"
else
  echo "(skip bpftrace: need sudo -n and bpftrace)"
fi

echo "== done; see $OUT/ and $OUT/REPORT.md =="
