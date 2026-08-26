# Vendored third-party sources

Everything listed here is copied verbatim from upstream. The build must not
depend on a network fetch, so the files sit in the tree; this manifest records
where they came from and what, if anything, was changed locally.

## FlameGraph

- Upstream: https://github.com/brendangregg/FlameGraph
- Pinned revision: `41fee1f99f9276008b7cd112fca19dc3ea84ac32`
- License: CDDL-1.0, text in `cddl1.txt` (the path the file headers point at)
- Files: `../flamegraph.pl`, `../stackcollapse-perf.pl`
- Local patches: none

Verify the pin and that no local edits crept in:

```sh
tools/vendor/check.sh
```

The check fetches the pinned revision and diffs it against the tree. It needs
network; it is not part of `make lint` or CI for that reason.
