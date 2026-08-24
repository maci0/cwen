#!/usr/bin/env python3
"""Parameter sweep for CWEN_SPEC block speculation on this box.

For every config, one resident engine (K16 frame server) measures decode-only
tok/s on each workload via (t(n)-t(1))/(n-1): both frames pay the same
prefill, so it cancels. Baseline arm is plain decode. Acceptance stats are
aggregated over the timed frames.

Slow: one model load per config plus 2*reps frames per workload. Run inside a
quiet window; the table is only as honest as the load average.

Usage: .venv/bin/python tools/spec_sweep.py [--quick] [--workloads a,b]
"""

from __future__ import annotations

import argparse
import sys
import time

from spec_e2e import SPEC_STATS, Server, build_corpus

# OFAT around the defaults (key=16, max_draft=8, min_draft=2, cooldown=8)
CONFIGS: list[tuple[str, dict[str, str]]] = [
    ("plain", {}),
    ("d2", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "2"}),
    ("d3", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "3"}),
    ("d4", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "4"}),
    ("d6", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "6"}),
    ("d8", {"CWEN_SPEC": "1"}),
    ("d12", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "12"}),
    ("n8-d8", {"CWEN_SPEC": "1", "CWEN_SPEC_NGRAM_N": "8"}),
    ("n24-d8", {"CWEN_SPEC": "1", "CWEN_SPEC_NGRAM_N": "24"}),
    ("n4-min1-d8", {"CWEN_SPEC": "1", "CWEN_SPEC_NGRAM_N": "4", "CWEN_SPEC_MIN_DRAFT": "1"}),
    ("min1-d8", {"CWEN_SPEC": "1", "CWEN_SPEC_MIN_DRAFT": "1"}),
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep CWEN_SPEC drafting knobs over the workload corpus."
    )
    ap.add_argument("--run", type=str, default="./run", help="path to the run binary")
    ap.add_argument(
        "--model", type=str, default="model/Qwen3.8-27B-Q4_0.gguf", help="GGUF model path"
    )
    ap.add_argument(
        "--workloads",
        type=str,
        default="strawberry,repeat",
        help="comma-separated corpus names from spec_e2e",
    )
    ap.add_argument(
        "--n-gen", type=int, default=None, help="timed tokens (default 48, or 32 with --quick)"
    )
    ap.add_argument("--quick", action="store_true", help="one rep per arm and shorter generations")
    args = ap.parse_args()
    n_gen = args.n_gen or (32 if args.quick else 48)
    reps = 1 if args.quick else 2

    corpus = build_corpus()
    names = [w.strip() for w in args.workloads.split(",") if w.strip()]
    for name in names:
        if name not in corpus:
            print(f"unknown workload '{name}'; have {sorted(corpus)}", file=sys.stderr)
            return 2

    # Drift guard: plain decoded twice at arm's length; if box load moved
    # them apart, every ratio below measures the tenants, not the knobs.
    print(f"load guard: measuring plain drift ({time.strftime('%H:%M:%S')})", flush=True)
    probe = Server(args.run, args.model, "drift-probe", {})
    try:
        ids = corpus[names[0]][0]
        _, ta = probe.frame(ids, 1)
        _, tb = probe.frame(ids, n_gen)
        s_a = (n_gen - 1) / (tb - ta)
        _, ta = probe.frame(ids, 1)
        _, tb = probe.frame(ids, n_gen)
        s_b = (n_gen - 1) / (tb - ta)
    finally:
        probe.close()
    drift = abs(s_a - s_b) / max(s_a, s_b)
    print(f"plain {s_a:.2f} vs {s_b:.2f} tok/s -> drift {drift:.0%}", flush=True)
    if drift > 0.25:
        print(
            "WARNING: machine drifted >25% between probes; ratios will be "
            "noise. Rerun in a quieter window.",
            flush=True,
        )

    # rows[config][workload] = [tok_s, stat lines over timed frames]
    rows: dict[str, dict[str, list]] = {}
    for label, env in CONFIGS:
        srv = Server(args.run, args.model, label, env)
        rows[label] = {name: [0.0, []] for name in names}
        try:
            for name in names:
                ids = corpus[name][0]
                for _ in range(reps):
                    _, t1 = srv.frame(ids, 1)
                    _, tn = srv.frame(ids, n_gen)
                    sps = (n_gen - 1) / (tn - t1)
                    if sps > rows[label][name][0]:
                        rows[label][name][0] = sps
                    print(f"[{label}] {name}: best {rows[label][name][0]:.2f} tok/s", flush=True)
        finally:
            tail = srv.close()
            # one summary line per generate, in frame order; frames alternate
            # warm(1)/timed(n) per rep, so timed lines sit at odd positions
            timed = [m.group(0) for m in SPEC_STATS.finditer(tail)][1::2]
            for i, name in enumerate(names):
                got = timed[i * reps : (i + 1) * reps]
                rows[label][name][1] = got

    base = rows["plain"]
    hdr = f"{'config':<13}{'draft%':>8}{'full':>6}{'kept':>7}" + "".join(
        f"{w + ' tok/s':>13}{w + ' x':>9}" for w in names
    )
    print("\n" + hdr)
    best_label, best_score = "plain", [-1.0] * len(names)
    for label, _ in CONFIGS:
        cells_parts: list[str] = []
        score = []
        for name in names:
            sps, st = rows[label][name]
            drafted = fulls = cyc_total = 0
            kept_w = 0.0
            for line in st:
                m = SPEC_STATS.match(line)
                if not m:
                    continue
                cyc = int(m.group(1))
                d = int(m.group(2))
                drafted += d
                fulls += int(m.group(3))
                cyc_total += cyc
                kept_w += float(m.group(5)) * d
            bx = base[name][0]
            x = sps / bx if bx else 0.0
            score.append(x)
            if st:
                dpct = 100.0 * drafted / cyc_total if cyc_total else 0.0
                kavg = kept_w / drafted if drafted else 0.0
                cells_parts.append(f"{dpct:>7.0f}%{fulls:>6}{kavg:>7.1f}{sps:13.2f}{x:9.2f}")
            else:
                cells_parts.append(f"{'-':>21}{sps:13.2f}{x:9.2f}")
        if label != "plain" and all(s > best_score[i] for i, s in enumerate(score)):
            best_label, best_score = label, list(score)
        print(f"{label:<13}" + "".join(cells_parts))
    rec = ", ".join(f"{w}={x:.2f}x" for w, x in zip(names, best_score, strict=True))
    print(f"\nsweet spot on this run: {best_label} ({rec})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
