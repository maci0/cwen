#!/usr/bin/env python3
"""Genetic algorithm + symbolic regression for cwen kernel knobs.

Evolves:
  - compile-time: CWEN_PREFETCH, CWEN_Q4_UNROLL, CWEN_Q4_PF_BLOCKS,
                  CWEN_OMP_THRESH_EXPR(M,K)  (symbolic tree)
  - runtime:      OMP_NUM_THREADS

Fitness = weighted sum of gemv ms/iter over golden kernels (FAIL -> huge).
Correctness is a hard gate (bench must print PASS).

Note: gemv pragmas hardcode schedule(static); there is no schedule knob.

Usage:
  .venv/bin/python tools/ga_evolve.py --gens 8 --pop 12 --avx512
  .venv/bin/python tools/ga_evolve.py --apply-best   # write best from log
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TUNE_H = ROOT / "cwen_tune.h"
BENCH = ROOT / "bench_q4_gemv"
MODEL = ROOT / "model" / "Qwen3.8-27B-Q4_0.gguf"
LOG_DIR = ROOT / "golden" / "ga_log"

# Kernels used during evolution (skip huge lm_head for speed).
DEFAULT_GOLDENS = [
    ("golden/blk_0_attn_gate_weight", 1.0),  # Q4_0
    ("golden/blk_0_ffn_down_weight", 1.5),  # Q4_1 larger
    ("golden/blk_0_ssm_out_weight", 1.0),  # Q5_K
]

# ---- symbolic expression trees for thr = f(M, K) ----
# Node: int const | "M" | "K" | (op, left, right)
# ops: +, -, *, //, >>, max, min  (// and >> safe)

OPS = ("+", "-", "*", "//", ">>", "max", "min")
CONST_POOL = (0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096)


def rand_leaf(rng: random.Random) -> Any:
    r = rng.random()
    if r < 0.35:
        return "M"
    if r < 0.70:
        return "K"
    return rng.choice(CONST_POOL)


def rand_expr(rng: random.Random, depth: int = 0, max_depth: int = 3) -> Any:
    if depth >= max_depth or rng.random() < 0.4:
        return rand_leaf(rng)
    op = rng.choice(OPS)
    return (op, rand_expr(rng, depth + 1, max_depth), rand_expr(rng, depth + 1, max_depth))


def eval_expr(node: Any, M: int, K: int) -> int:
    if isinstance(node, int):
        return node
    if node == "M":
        return M
    if node == "K":
        return K
    op, a, b = node
    av, bv = eval_expr(a, M, K), eval_expr(b, M, K)
    if op == "+":
        return av + bv
    if op == "-":
        return av - bv
    if op == "*":
        return av * bv
    if op == "//":
        # match C int division (trunc toward zero), not Python floor
        if bv == 0:
            return av
        q = abs(av) // abs(bv)
        return q if (av < 0) == (bv < 0) else -q
    if op == ">>":
        s = max(0, min(20, bv))
        return av >> s
    if op == "max":
        return max(av, bv)
    if op == "min":
        return min(av, bv)
    raise ValueError(op)


def expr_to_c(node: Any) -> str:
    """Emit a C integer expression using (M) and (K)."""
    if isinstance(node, int):
        return str(node)
    if node == "M":
        return "(M)"
    if node == "K":
        return "(K)"
    op, a, b = node
    ca, cb = expr_to_c(a), expr_to_c(b)
    if op == "+":
        return f"(({ca})+({cb}))"
    if op == "-":
        return f"(({ca})-({cb}))"
    if op == "*":
        return f"(({ca})*({cb}))"
    if op == "//":
        # safe div
        return f"(({cb})!=0?({ca})/({cb}):({ca}))"
    if op == ">>":
        return f"(({ca})>>(({cb})<0?0:(({cb})>20?20:({cb}))))"
    if op == "max":
        return f"(({ca})>({cb})?({ca}):({cb}))"
    if op == "min":
        return f"(({ca})<({cb})?({ca}):({cb}))"
    raise ValueError(op)


def mutate_expr(node: Any, rng: random.Random, p: float = 0.25) -> Any:
    if rng.random() < p:
        return rand_expr(rng, 0, max_depth=rng.randint(1, 3))
    if isinstance(node, (int, str)):
        if rng.random() < 0.5:
            return rand_leaf(rng)
        return node
    op, a, b = node
    if rng.random() < 0.2:
        op = rng.choice(OPS)
    return (op, mutate_expr(a, rng, p), mutate_expr(b, rng, p))


def crossover_expr(a: Any, b: Any, rng: random.Random) -> Any:
    if isinstance(a, (int, str)) or isinstance(b, (int, str)):
        return a if rng.random() < 0.5 else b
    if rng.random() < 0.5:
        return (a[0], crossover_expr(a[1], b[1] if isinstance(b, tuple) else b, rng), a[2])
    return (
        b[0] if isinstance(b, tuple) else a[0],
        a[1],
        crossover_expr(a[2], b[2] if isinstance(b, tuple) else b, rng),
    )


def simplify_const_expr(node: Any) -> Any:
    """Fold pure-const subtrees."""
    if isinstance(node, (int, str)):
        return node
    op, a, b = node
    a, b = simplify_const_expr(a), simplify_const_expr(b)
    if isinstance(a, int) and isinstance(b, int):
        try:
            return eval_expr((op, a, b), 0, 0)
        except Exception:
            return (op, a, b)
    return (op, a, b)


# ---- genome ----

PREFETCH_CHOICES = (0, 1, 2, 4, 8, 16, 32)
UNROLL_CHOICES = (1, 2, 4)
PF_BLOCKS_CHOICES = (4, 8, 12, 16, 20, 24, 32)  # Q4_0R dual-distance PF
THREADS_CHOICES = (8, 12, 16, 20, 24)  # one-CCD focus


@dataclass
class Genome:
    prefetch: int = 2
    q4_unroll: int = 2
    q4_pf_blocks: int = 4
    omp_threads: int = 16
    thr_expr: Any = 64  # symbolic tree or int
    # fitness cache
    fitness: float = field(default=float("inf"), compare=False)
    per_kernel: dict = field(default_factory=dict, compare=False)
    ok: bool = field(default=False, compare=False)

    def compile_key(self) -> str:
        return json.dumps(
            {
                "pf": self.prefetch,
                "u": self.q4_unroll,
                "qpf": self.q4_pf_blocks,
                "expr": self.thr_expr,
            },
            sort_keys=True,
            default=str,
        )

    def runtime_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.omp_threads)
        env["OMP_DYNAMIC"] = "false"
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "{0}:16:1"  # one CCD on 9950X
        env["OMP_WAIT_POLICY"] = "passive"
        env["GOMP_SPINCOUNT"] = "100"
        return env

    def to_dict(self) -> dict:
        return {
            "prefetch": self.prefetch,
            "q4_unroll": self.q4_unroll,
            "q4_pf_blocks": self.q4_pf_blocks,
            "omp_threads": self.omp_threads,
            "thr_expr": self.thr_expr,
            "thr_c": expr_to_c(simplify_const_expr(self.thr_expr)),
            "fitness": self.fitness,
            "per_kernel": self.per_kernel,
            "ok": self.ok,
        }


def random_genome(rng: random.Random) -> Genome:
    # Prefer simple thr expressions early
    if rng.random() < 0.5:
        thr: Any = rng.choice((16, 32, 48, 64, 96, 128, 256))
    else:
        thr = simplify_const_expr(rand_expr(rng, 0, max_depth=rng.randint(1, 3)))
    return Genome(
        prefetch=rng.choice(PREFETCH_CHOICES),
        q4_unroll=rng.choice(UNROLL_CHOICES),
        q4_pf_blocks=rng.choice(PF_BLOCKS_CHOICES),
        omp_threads=rng.choice(THREADS_CHOICES),
        thr_expr=thr,
    )


def mutate(g: Genome, rng: random.Random, p: float = 0.3) -> Genome:
    ng = Genome(
        prefetch=g.prefetch,
        q4_unroll=g.q4_unroll,
        q4_pf_blocks=g.q4_pf_blocks,
        omp_threads=g.omp_threads,
        thr_expr=g.thr_expr,
    )
    if rng.random() < p:
        ng.prefetch = rng.choice(PREFETCH_CHOICES)
    if rng.random() < p:
        ng.q4_unroll = rng.choice(UNROLL_CHOICES)
    if rng.random() < p:
        ng.q4_pf_blocks = rng.choice(PF_BLOCKS_CHOICES)
    if rng.random() < p:
        ng.omp_threads = rng.choice(THREADS_CHOICES)
    if rng.random() < p:
        if isinstance(ng.thr_expr, int) and rng.random() < 0.4:
            ng.thr_expr = rng.choice((16, 32, 48, 64, 96, 128, 256, 512))
        else:
            ng.thr_expr = simplify_const_expr(mutate_expr(ng.thr_expr, rng))
    return ng


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    return Genome(
        prefetch=a.prefetch if rng.random() < 0.5 else b.prefetch,
        q4_unroll=a.q4_unroll if rng.random() < 0.5 else b.q4_unroll,
        q4_pf_blocks=a.q4_pf_blocks if rng.random() < 0.5 else b.q4_pf_blocks,
        omp_threads=a.omp_threads if rng.random() < 0.5 else b.omp_threads,
        thr_expr=crossover_expr(a.thr_expr, b.thr_expr, rng)
        if rng.random() < 0.5
        else (a.thr_expr if rng.random() < 0.5 else b.thr_expr),
    )


def write_tune_h(g: Genome, path: Path = TUNE_H) -> None:
    expr = simplify_const_expr(g.thr_expr)
    c_expr = expr_to_c(expr)
    # sanity: evaluate at a few shapes so we don't emit pathological thr
    for M, K in ((6144, 5120), (5120, 17408), (5120, 6144), (32, 5120)):
        try:
            v = eval_expr(expr, M, K)
            if abs(v) > 10_000_000:
                c_expr = "64"
                expr = 64
                break
        except Exception:
            c_expr = "64"
            expr = 64
            break
    body = f"""/* Auto-tuned knobs (GA / symbolic search). Safe defaults if missing. */
#ifndef CWEN_TUNE_H
#define CWEN_TUNE_H

#ifndef CWEN_OMP_THREADS
#define CWEN_OMP_THREADS {int(g.omp_threads)}
#endif
/* Symbolic OMP gate: parallel when M > EXPR(M,K). Evolved by GA. */
#ifndef CWEN_OMP_THRESH_EXPR
#define CWEN_OMP_THRESH_EXPR(M, K) ({c_expr})
#endif
#ifndef CWEN_PREFETCH
#define CWEN_PREFETCH {int(g.prefetch)}
#endif
#ifndef CWEN_Q4_UNROLL
#define CWEN_Q4_UNROLL {int(g.q4_unroll)}
#endif
#ifndef CWEN_Q4_PF_BLOCKS
#define CWEN_Q4_PF_BLOCKS {int(g.q4_pf_blocks)}
#endif

#endif
"""
    # Atomic replace: cwen_tune.h is compiled into every build, so a crash or
    # full disk mid-write must never strand a truncated header. Same contract
    # as tools/repack_q4.py; re-running after a failure converges.
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(body)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def compile_bench(avx512: bool, jobs: int = 0) -> None:
    cmd = ["make", "bench-q4_gemv"]
    if avx512:
        cmd.append("AVX512=1")
    r = subprocess.run(
        cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stdout}\n{r.stderr}")


def one_bench(gdir: Path, env: dict[str, str], iters: int) -> tuple[bool, float, str, str]:
    """Return (ok, ms/iter, kernel label, failure detail tail)."""
    r = subprocess.run(
        [str(BENCH), str(MODEL), str(gdir), str(iters)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(ROOT),
    )
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(PASS|FAIL)\s+(\S+)\s+.*\s+([0-9.]+)\s+ms/iter", out)
    if not m:
        return False, 1e9, "", out[-500:]
    ok = m.group(1) == "PASS"
    return ok, float(m.group(3)), m.group(2), "" if ok else out[-500:]


_active_compile_key: str | None = None


def ensure_compiled(g: Genome, avx512: bool, compile_cache: dict[str, bool]) -> bool:
    """Write tune header and rebuild only when compile genes change."""
    global _active_compile_key
    ck = g.compile_key()
    if ck in compile_cache and not compile_cache[ck]:
        return False
    if ck == _active_compile_key and BENCH.exists():
        return True
    write_tune_h(g)
    try:
        compile_bench(avx512)
        compile_cache[ck] = True
        _active_compile_key = ck
        return True
    except RuntimeError as e:
        print(f"  compile FAIL: {e}", flush=True)
        compile_cache[ck] = False
        return False


def evaluate(
    g: Genome,
    goldens: list[tuple[str, float]],
    avx512: bool,
    iters: int,
    trials: int,
    compile_cache: dict[str, bool],
) -> Genome:
    if not ensure_compiled(g, avx512, compile_cache):
        g.fitness = 1e12
        g.ok = False
        return g

    env = g.runtime_env()
    total = 0.0
    ok_all = True
    per: dict[str, float] = {}
    for rel, w in goldens:
        gdir = ROOT / rel
        times: list[float] = []
        for _ in range(trials):
            ok, ms, name, detail = one_bench(gdir, env, iters)
            if not ok:
                ok_all = False
                times.append(1e6)
                # A bench that dies or FAILs must say why: during a long GA
                # run a silently swallowed failure looks identical to a slow
                # genome and poisons every later generation.
                print(
                    f"  bench FAIL {gdir.name} (kernel={name}): {detail}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                times.append(ms)
        med = statistics.median(times)
        per[rel] = med
        total += w * med
    g.per_kernel = per
    g.fitness = total if ok_all else total + 1e6
    g.ok = ok_all
    return g


def seed_baseline() -> Genome:
    return Genome(
        prefetch=2,
        q4_unroll=2,
        q4_pf_blocks=4,
        omp_threads=16,
        thr_expr=64,
    )


def tournament(pop: list[Genome], rng: random.Random, k: int = 3) -> Genome:
    picks = rng.sample(pop, min(k, len(pop)))
    return min(picks, key=lambda x: x.fitness)


def run_ga(args: argparse.Namespace) -> Genome:
    rng = random.Random(args.seed)
    goldens = DEFAULT_GOLDENS
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    compile_cache: dict[str, bool] = {}

    # population: baseline + random
    pop: list[Genome] = [seed_baseline()]
    # known good-ish seeds
    for thr in (32, 64, 128):
        for nt in (8, 16, 32):
            pop.append(
                Genome(
                    prefetch=2,
                    q4_unroll=2,
                    q4_pf_blocks=4,
                    omp_threads=nt,
                    thr_expr=thr,
                )
            )
    while len(pop) < args.pop:
        pop.append(random_genome(rng))
    pop = pop[: args.pop]

    history: list[dict] = []
    best = seed_baseline()
    best.fitness = float("inf")

    print(
        f"GA start: pop={args.pop} gens={args.gens} iters={args.iters} "
        f"trials={args.trials} avx512={args.avx512}",
        flush=True,
    )
    t0 = time.monotonic()

    for gen in range(args.gens):
        print(f"\n=== gen {gen}/{args.gens - 1} ===", flush=True)
        for i, ind in enumerate(pop):
            if ind.fitness < float("inf") and gen > 0 and ind.ok:
                # already evaluated and kept
                continue
            evaluate(ind, goldens, args.avx512, args.iters, args.trials, compile_cache)
            pk = " ".join(f"{Path(k).name}={v:.3f}" for k, v in ind.per_kernel.items())
            print(
                f"  [{i:02d}] fit={ind.fitness:.3f} ok={ind.ok} "
                f"nt={ind.omp_threads} pf={ind.prefetch} u={ind.q4_unroll} "
                f"qpf={ind.q4_pf_blocks} "
                f"thr={expr_to_c(simplify_const_expr(ind.thr_expr))} | {pk}",
                flush=True,
            )
            if ind.fitness < best.fitness:
                best = Genome(
                    **{
                        k: getattr(ind, k)
                        for k in (
                            "prefetch",
                            "q4_unroll",
                            "q4_pf_blocks",
                            "omp_threads",
                            "thr_expr",
                        )
                    }
                )
                best.fitness = ind.fitness
                best.per_kernel = dict(ind.per_kernel)
                best.ok = ind.ok

        history.append({"gen": gen, "best": best.to_dict(), "pop": [p.to_dict() for p in pop]})
        (LOG_DIR / "history.json").write_text(
            json.dumps(history, indent=2, default=str), encoding="utf-8"
        )
        (LOG_DIR / "best.json").write_text(
            json.dumps(best.to_dict(), indent=2, default=str), encoding="utf-8"
        )

        if gen == args.gens - 1:
            break

        # next generation
        new_pop: list[Genome] = [best]  # elitism
        while len(new_pop) < args.pop:
            if rng.random() < 0.15:
                child = random_genome(rng)
            else:
                p1, p2 = tournament(pop, rng), tournament(pop, rng)
                child = crossover(p1, p2, rng)
                child = mutate(child, rng, p=args.mutate)
            # reset fitness so re-evaluated
            child.fitness = float("inf")
            child.ok = False
            new_pop.append(child)
        pop = new_pop

    # final re-eval best with more trials
    print("\n=== final refine (more trials) ===", flush=True)
    evaluate(best, goldens, args.avx512, max(args.iters, 10), max(args.trials, 5), compile_cache)
    write_tune_h(best)
    compile_bench(args.avx512)
    (LOG_DIR / "best.json").write_text(
        json.dumps(best.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    elapsed = time.monotonic() - t0
    print(
        f"\nBEST fitness={best.fitness:.3f} ok={best.ok}  elapsed={elapsed:.1f}s\n"
        f"  thr_expr = {expr_to_c(simplify_const_expr(best.thr_expr))}\n"
        f"  threads={best.omp_threads}\n"
        f"  prefetch={best.prefetch} unroll={best.q4_unroll} q4_pf={best.q4_pf_blocks}\n"
        f"  kernels={best.per_kernel}\n"
        f"  wrote {TUNE_H}",
        flush=True,
    )
    return best


def apply_best() -> None:
    path = LOG_DIR / "best.json"
    if not path.exists():
        print("no best.json; run ga first", file=sys.stderr)
        sys.exit(1)
    d = json.loads(path.read_text(encoding="utf-8"))
    # thr_expr may be nested lists from JSON
    g = Genome(
        prefetch=int(d["prefetch"]),
        q4_unroll=int(d["q4_unroll"]),
        q4_pf_blocks=int(d["q4_pf_blocks"]),
        omp_threads=int(d["omp_threads"]),
        thr_expr=restore_expr(d["thr_expr"]),
    )
    write_tune_h(g)
    print(f"applied best -> {TUNE_H}")
    print(json.dumps(g.to_dict(), indent=2, default=str))


def restore_expr(node: Any) -> Any:
    if isinstance(node, list):
        if len(node) == 3 and isinstance(node[0], str) and node[0] in OPS:
            return (node[0], restore_expr(node[1]), restore_expr(node[2]))
        return node
    return node


def main() -> int:
    ap = argparse.ArgumentParser(description="GA + symbolic thr for cwen")
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=10)
    ap.add_argument("--iters", type=int, default=5, help="bench iters during evolve")
    ap.add_argument("--trials", type=int, default=1, help="trials per kernel during evolve")
    ap.add_argument("--mutate", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--avx512",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compile kernels with -mavx512 (Zen4/5 peak); --no-avx512 to disable",
    )
    ap.add_argument("--apply-best", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run: pop=4 gens=2")
    args = ap.parse_args()
    if args.smoke:
        args.pop, args.gens, args.iters, args.trials = 4, 2, 3, 1
    if args.apply_best:
        apply_best()
        return 0
    if not MODEL.exists():
        print(f"missing model {MODEL}", file=sys.stderr)
        return 1
    run_ga(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
