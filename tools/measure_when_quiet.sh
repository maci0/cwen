#!/bin/sh
# Run a command only in a quiet load window, so absolute tok/s numbers mean
# something on a shared box. Load swings 10-80 here; anything measured above
# ~1x cores is scheduler noise, not the engine.
#
#   tools/measure_when_quiet.sh [-l LOAD] [-w SECONDS] -- cmd [args...]
#
# -l  1-minute loadavg ceiling (default: half the core count)
# -w  give up after this many seconds of waiting (default 3600, 0 = forever)
#
# Exits 75 (EX_TEMPFAIL) if the window never opened, so a caller can tell
# "never ran" apart from "ran and failed".
set -eu

cores=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)
limit=$((cores / 2))
wait_s=3600

while [ $# -gt 0 ]; do
  case $1 in
    -l) limit=$2; shift 2 ;;
    -w) wait_s=$2; shift 2 ;;
    --) shift; break ;;
    *)  echo "usage: $0 [-l LOAD] [-w SECONDS] -- cmd [args...]" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "$0: no command given" >&2; exit 2; }

waited=0
while :; do
  load=$(cut -d' ' -f1 /proc/loadavg)
  # integer compare: /proc/loadavg prints two decimals, so scale both by 100
  if [ "$(echo "$load" | tr -d .)" -le $((limit * 100)) ]; then
    echo "quiet: load $load <= $limit after ${waited}s" >&2
    exec "$@"
  fi
  [ "$wait_s" -eq 0 ] || [ "$waited" -lt "$wait_s" ] || {
    echo "$0: load still $load (> $limit) after ${waited}s; not measuring" >&2
    exit 75
  }
  sleep 15
  waited=$((waited + 15))
done
