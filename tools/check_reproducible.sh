#!/usr/bin/env bash
# Independent rebuild-and-compare gate: ./run must come out bit-identical when
# rebuilt from a different absolute path under a pinned locale, timezone, and
# SOURCE_DATE_EPOCH. Catches -ffile-prefix-map gaps, __DATE__/__TIME__ use,
# and LTO nondeterminism. Work area lives under gitignored outputs/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT="${OUT:-outputs/repro}"
trap 'rm -rf "$OUT"' EXIT

make clean >/dev/null
make >/dev/null
a=$(sha256sum run | awk '{print $1}')

mkdir -p "$OUT/build2"
cp run.c cwen_tune.h Makefile "$OUT/build2/"
(cd "$OUT/build2" && LC_ALL=C TZ=Asia/Tokyo SOURCE_DATE_EPOCH=946684800 make >/dev/null)
b=$(sha256sum "$OUT/build2/run" | awk '{print $1}')

if [ "$a" != "$b" ]; then
  echo "reproducibility check FAILED:" >&2
  echo "  first:  $a" >&2
  echo "  second: $b" >&2
  exit 1
fi
echo "reproducible: sha256 $a"
