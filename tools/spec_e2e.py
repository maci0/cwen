#!/usr/bin/env python3
"""End-to-end gate for CWEN_SPEC block speculation, over the K16 frame server.

Loads the model once per mode (plain / spec configs), then pushes a corpus of
prompts through binary stdin/stdout frames. Hard gates:

  - every frame exits cleanly and emits exactly n_gen tokens
  - spec token streams are identical to plain (greedy is lossless)
  - drafter sanity: drafted cycles > 0 on repetitive cases

Soft reporting: wall-clock speedup per case. Exit code 0 iff all hard gates
pass.

Usage: .venv/bin/python tools/spec_e2e.py [--quick] [--n-gen N]
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import selectors
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

REPEAT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
)
CODE_TEXT = (
    "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n\nresult = add(1, 2)\n"
)

Case = tuple[list[int], bool]  # (prompt ids, drafting expected)


def build_corpus() -> dict[str, Case]:
    try:
        from transformers import AutoTokenizer
    except ImportError as err:
        print("pip/uv: install transformers tokenizers", file=sys.stderr)
        raise SystemExit(1) from err
    tok = AutoTokenizer.from_pretrained("model")

    def enc(text: str) -> list[int]:
        return tok.encode(text, add_special_tokens=False)

    full = enc(REPEAT_TEXT)
    # truncated last repeat: models tend to continue the pattern
    repeat = full * 5 + enc(REPEAT_TEXT[: len(REPEAT_TEXT) // 2])
    chat = tok.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Repeat the word strawberry exactly 60 times, "
                "separated by single spaces.",
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # tokenize=False returns the rendered prompt string; the annotation is
    # wider (tokenize=True shapes), so pin it before encoding.
    assert isinstance(chat, str)
    return {
        "repeat": (repeat, True),
        "strawberry": (enc(chat), True),
        "code": (enc(CODE_TEXT * 4), False),
        "count": (enc(" ".join(str(i) for i in range(1, 65))), False),
        "prose": (
            enc(
                "In 1543, Copernicus published his heliocentric model. "
                "Vesalius published his anatomy atlas the same year. "
                "Both works relied on printing presses in Basel and Venice."
            ),
            False,
        ),
    }


class Server:
    """One resident engine speaking the K16 binary frame protocol.

    stderr is drained continuously by a daemon thread: the engine writes a
    spec summary per frame (and per-cycle traces under CWEN_SPEC_DEBUG), and
    an undrained stderr pipe would eventually block the engine mid-frame
    while this client blocks on stdout. Reply reads are deadline-bounded so a
    hung or dead engine fails with its exit status and stderr tail instead of
    hanging forever or raising a bare struct.error.
    """

    READY_MARKER = b"waiting on stdin"

    def __init__(
        self,
        run: Path,
        model: Path,
        label: str,
        env_extra: dict[str, str],
        timeout: float = 900.0,
    ):
        self.label = label
        self.timeout = timeout
        env = dict(os.environ)
        env["CWEN_SERVER"] = "1"
        env.update(env_extra)
        self.proc = subprocess.Popen(
            [str(run), str(model)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._wedged = False
        assert self.proc.stdin and self.proc.stdout and self.proc.stderr
        # Bind the PIPE streams once: every later use goes through these
        # non-optional names instead of re-widening to IO[bytes] | None.
        self._in = self.proc.stdin
        self._out = self.proc.stdout
        self._err = self.proc.stderr
        self._stderr_chunks: list[bytes] = []
        self._drain = threading.Thread(target=self._drain_stderr, daemon=True)
        self._drain.start()
        # startup logs several lines (omp tuning, load progress); poll the
        # collected stderr for the ready marker wherever it lands
        deadline = time.monotonic() + timeout
        try:
            while True:
                if self.READY_MARKER in self.stderr_bytes():
                    break
                if not self._drain.is_alive():
                    raise RuntimeError(f"{label}: engine exited before ready\n{self.stderr_text()}")
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"{label}: engine not ready after {timeout:.0f}s\n{self.stderr_text()}"
                    )
                time.sleep(0.05)
        except BaseException:
            # A half-constructed Server is invisible to callers' finally
            # blocks; without this teardown a timed-out or interrupted start
            # strands a resident engine holding the entire model in RAM.
            self._kill_engine()
            raise

    def _kill_engine(self) -> None:
        """Unconditional teardown for a failed startup: EOF first so a healthy
        engine exits cleanly, then SIGKILL for one that never became ready."""
        proc = self.proc
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
        self._drain.join(timeout=5)

    def _drain_stderr(self) -> None:
        while True:
            chunk = os.read(self._err.fileno(), 65536)
            if not chunk:
                return
            self._stderr_chunks.append(chunk)

    def stderr_bytes(self) -> bytes:
        return b"".join(self._stderr_chunks)

    def stderr_text(self, tail: int | None = None) -> str:
        data = self.stderr_bytes() if tail is None else self.stderr_bytes()[-tail:]
        # explicit decode: run.c stderr can carry non-ASCII path bytes
        return data.decode("utf-8", errors="replace")

    def _fail(self, msg: str) -> str:
        """Diagnosed failure text: what broke, engine status, recent stderr."""
        rc = self.proc.poll()
        return f"{self.label}: {msg}; engine rc={rc}; stderr tail:\n{self.stderr_text(tail=2000)}"

    def _read_exact(self, n: int, what: str) -> bytes:
        """Read exactly n reply bytes or fail with a diagnosis before deadline."""
        deadline = time.monotonic() + self.timeout
        sel = selectors.DefaultSelector()
        sel.register(self._out.fileno(), selectors.EVENT_READ)
        try:
            buf = bytearray()
            while len(buf) < n:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._wedged = True
                    raise RuntimeError(
                        self._fail(
                            f"timeout after {self.timeout:.0f}s waiting for {what} "
                            f"(got {len(buf)}/{n} bytes)"
                        )
                    )
                if not sel.select(timeout=remaining):
                    continue  # loop re-checks the deadline
                chunk = os.read(self._out.fileno(), n - len(buf))
                if not chunk:
                    raise RuntimeError(
                        self._fail(f"engine closed the pipe mid-{what} (got {len(buf)}/{n} bytes)")
                    )
                buf.extend(chunk)
            return bytes(buf)
        finally:
            sel.close()

    def frame(self, ids: list[int], n_gen: int) -> tuple[list[int], float]:
        t0 = time.perf_counter()
        try:
            self._in.write(struct.pack("<2I", len(ids), n_gen))
            self._in.write(struct.pack(f"<{len(ids)}i", *ids))
            self._in.flush()
        except BrokenPipeError:
            raise RuntimeError(self._fail("engine closed stdin before the request")) from None
        (n_out,) = struct.unpack("<I", self._read_exact(4, "reply header"))
        if n_out == 0 or n_out > 4096:
            raise RuntimeError(self._fail(f"frame rejected (n_out={n_out})"))
        raw = self._read_exact(4 * n_out, "reply payload")
        wall = time.perf_counter() - t0
        return list(struct.unpack(f"<{n_out}i", raw)), wall

    def close(self) -> str:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except BrokenPipeError:
            pass
        # a wedged engine (timed-out frame) gets no long grace period; the
        # real engine exits promptly on EOF, so healthy shutdown keeps 60 s
        try:
            self.proc.wait(timeout=5 if self._wedged else 60)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=60)
        self._drain.join(timeout=5)
        return self.stderr_text()


SPEC_STATS = re.compile(
    r"spec: (\d+) cycles \((\d+) drafted, (\d+) full accept, (\d+) short; avg kept ([\d.]+)\)"
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="A/B CWEN_SPEC block drafting vs plain decode over the prompt corpus."
    )
    ap.add_argument("--run", type=Path, default=Path("./run"), help="path to the run binary")
    ap.add_argument(
        "--model", type=Path, default=Path("model/Qwen3.8-27B-Q4_0.gguf"), help="GGUF model path"
    )
    ap.add_argument(
        "--n-gen", type=int, default=None, help="tokens per case (default 48, or 24 with --quick)"
    )
    ap.add_argument("--quick", action="store_true", help="shorter generations (24 vs 48 tokens)")
    ap.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="seconds allowed for engine startup and each frame (default 900)",
    )
    args = ap.parse_args()
    n_gen = args.n_gen or (24 if args.quick else 48)

    corpus = build_corpus()
    modes = [
        ("plain", {}),
        ("spec-d8", {"CWEN_SPEC": "1"}),
        ("spec-d3", {"CWEN_SPEC": "1", "CWEN_SPEC_MAX_DRAFT": "3"}),
    ]
    servers: dict[str, Server] = {}
    results: dict[tuple[str, str], tuple[list[int], float]] = {}
    tails: dict[str, str] = {}
    try:
        # all engines resident at once; frames rotate per case so every arm
        # samples the same load window (sequential phases measured machine
        # drift, not speculation)
        for label, env in modes:
            t0 = time.perf_counter()
            servers[label] = Server(
                args.run.resolve(), args.model.resolve(), label, env, timeout=args.timeout
            )
            print(f"[{label}] model up in {time.perf_counter() - t0:.0f}s", flush=True)
        for case_i, name in enumerate(corpus):
            for rot in range(len(modes)):
                label = modes[(case_i + rot) % len(modes)][0]
                results[(label, name)] = servers[label].frame(corpus[name][0], n_gen)
                print(f"[{label}] {name}: done ({results[(label, name)][1]:.1f}s)", flush=True)
    finally:
        for lbl, srv in servers.items():
            tails[lbl] = srv.close()

    fails = 0
    hdr = f"{'case':<12} {'plain s':>9} {'d8 s':>9} {'d8 x':>6} {'d3 s':>9} {'d3 x':>6}  gates"
    print("\n" + hdr)
    # the engine prints one spec summary per generate, in frame order, so
    # matches zip 1:1 with corpus order for each spec server
    stats_by_label: dict[str, list[re.Match[str]]] = {}
    for lbl in ("spec-d8", "spec-d3"):
        stats_by_label[lbl] = list(SPEC_STATS.finditer(tails.get(lbl, "")))
        if len(stats_by_label[lbl]) != len(corpus):
            print(
                f"WARN: {lbl}: {len(stats_by_label[lbl])} spec summaries for {len(corpus)} frames"
            )

    for idx, (name, (_, expect_draft)) in enumerate(corpus.items()):
        ptoks, pwall = results[("plain", name)]
        gates = []
        if len(ptoks) != n_gen:
            gates.append(f"plain-len={len(ptoks)}")
            fails += 1
        row = f"{name:<12} {pwall:9.1f}"
        for lbl, short in (("spec-d8", "d8"), ("spec-d3", "d3")):
            stoks, swall = results[(lbl, name)]
            if stoks != ptoks:
                bad = next(
                    (i for i, (a, b) in enumerate(zip(ptoks, stoks, strict=False)) if a != b),
                    None,
                )
                gates.append(f"{short}-DIVERGE@{bad}")
                fails += 1
            x = pwall / swall if swall else 0.0
            row += f" {swall:9.1f} {x:6.2f}"
        if expect_draft:
            stats = stats_by_label["spec-d8"]
            drafted = int(stats[idx].group(2)) if idx < len(stats) else 0
            if drafted > 0:
                gates.append("draft-ok")
            else:
                gates.append("NO-DRAFTING")
                fails += 1
        print(row + ("  " + ", ".join(gates) if gates else ""))

    for lbl in ("spec-d8", "spec-d3"):
        lines = stats_by_label[lbl]
        if lines:
            print(f"\n{lbl} acceptance per case:")
            for name, m in zip(corpus, lines, strict=True):
                print(f"  {name:<12} {m.group(0)}")

    if fails:
        print(f"\nFAIL: {fails} hard-gate violation(s)")
        return 1
    print("\nPASS: all frames clean; spec streams identical to plain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
