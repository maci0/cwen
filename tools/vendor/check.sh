#!/bin/sh
# Verify vendored sources still match their pinned upstream revision, so a
# local edit cannot masquerade as pristine upstream. Needs network; run it by
# hand, not from CI.
set -eu

root=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
rev=41fee1f99f9276008b7cd112fca19dc3ea84ac32
base="https://raw.githubusercontent.com/brendangregg/FlameGraph/$rev"

mkdir -p "$root/.scratch"
tmp=$(mktemp -d "$root/.scratch/vendorcheck.XXXXXX")
trap 'rm -rf "$tmp"' EXIT INT TERM

rc=0
check() { # remote-path local-path
  curl -fsS -o "$tmp/dl" "$base/$1"
  if cmp -s "$tmp/dl" "$root/$2"; then
    echo "ok    $2"
  else
    echo "DRIFT $2 differs from FlameGraph@$rev" >&2
    diff -u "$tmp/dl" "$root/$2" >&2 || true
    rc=1
  fi
}

check flamegraph.pl          tools/flamegraph.pl
check stackcollapse-perf.pl  tools/stackcollapse-perf.pl
check docs/cddl1.txt         tools/vendor/cddl1.txt
exit $rc
