#!/usr/bin/env bash
# Autonomous single-kernel accept loop helper.
# Usage: tools/opt_loop.sh q4_gemv [golden/dir]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
kernel="${1:-q4_gemv}"
gdir="${2:-golden/blk_0_attn_gate_weight}"

case "$kernel" in
  q4_gemv)
    make bench-q4_gemv
    log=$(mktemp)
    trap 'rm -f "$log"' EXIT
    ./bench_q4_gemv model/Qwen3.8-27B-Q4_0.gguf "$gdir" 10 | tee "$log"
    .venv/bin/python tools/accept.py "$gdir"
    echo "Edit run.c gemv/dot_q4_0, re-run this script. Keep PASS + better ms/iter."
    ;;
  *)
    echo "unknown kernel: $kernel" >&2
    exit 1
    ;;
esac
