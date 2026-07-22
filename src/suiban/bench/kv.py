"""`suiban bench kv` — KV V-cache quality benchmark on YOUR hardware.

For each V-cache config (V in {tq4_0, tq3_0, q4_0, q8_0}, K=q8_0 always — the
plan-frozen matrix) the runner measures:

(a) real perplexity via the fork's own `llama-perplexity` tool (subprocess against
    the installed TurboQuant binary's sibling, e.g. ~/.bonsai/bin/cuda/llama-perplexity)
    over a bundled synthetic corpus — absolute PPL on synthetic text is meaningless,
    the DELTAS vs the q8_0 baseline are the signal. When the installed binary has no
    llama-perplexity next to it (prebuilts ship without tools) the column reads n/a
    with the reason — never faked;
(b) a needle test: a fact planted at 10/50/90% depth of a long synthetic context at
    4k/8k/16k tokens (16k only when the slot ctx fits on this hardware — sizes that
    do not fit are reported as "not run", honestly, not as failures);
(c) a multi-turn agentic replay: a canned ~30-turn synthetic transcript is replayed
    with only the probe answers generated live, scoring whether facts from the early
    turns survive a conversation-length quantized V cache (bench/replay.py).

Probing: TurboQuant configs are skipped up front (standard fallback notice) when the
installed binary has no TQ kernels; every other config is validated by actually
launching llama-server with the type — a binary that rejects `--cache-type-v X`
fails the health probe and the config is reported as skipped, never faked. The slot
context is chosen by a descending ladder: the largest ctx that boots wins, and needle
sizes beyond it are labeled "not run".

The report lands at ~/.bonsai/reports/bench-kv-<date>.md with an honest
"measured on your hardware" header. The harness logic (math, skip handling, report
rendering) is unit-tested against the mock backend, which reports FIXED FIXTURE
values — clearly labeled, never presented as measurements. The real-slot path is
exercised on real weights via `suiban bench kv`.
"""

from __future__ import annotations

import contextlib
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import httpx

from suiban.bench.replay import REPLAY_PROBES, probe_messages, score_replay
from suiban.bench.sample_text import SAMPLE_TEXT
from suiban.effort import sampling_for, thinking_payload_fields
from suiban.kv import KvState
from suiban.llama import binary as binary_mod
from suiban.llama.backend import RealBackend, SlotBackend, build_server_flags
from suiban.sched.planner import PlannedSlot

V_TYPES: tuple[str, ...] = ("tq4_0", "tq3_0", "q4_0", "q8_0")
BASELINE_V = "q8_0"
K_TYPE = "q8_0"
TURBOQUANT_V_TYPES = ("tq4_0", "tq3_0")

NEEDLE_SIZES: tuple[int, ...] = (4096, 8192, 16384)  # haystack sizes (estimated tokens)
NEEDLE_DEPTHS: tuple[int, ...] = (10, 50, 90)
NEEDLE_FACT = "The secret bonsai passphrase is korokke-42."
NEEDLE_QUESTION = (
    "Somewhere in the text above, a secret bonsai passphrase is stated. "
    "What is it? Answer with the passphrase only."
)
NEEDLE_ANSWER = "korokke-42"

BENCH_PORT = 8799

# llama-perplexity settings: 6 x 2048-token chunks of the synthetic corpus.
PPL_CHUNK_CTX = 2048
PPL_CHUNKS = 6
PPL_TIMEOUT_S = 3600.0

# The standard fallback notice text (mirrors kv.py's resolve_kv_state wording).
TQ_SKIP_REASON = (
    "TurboQuant kernels not present in installed binary; skipped. Run: suiban install turboquant"
)

_FILLER_SENTENCES = (
    "The gardener checks the drainage holes of every pot before the first watering.",
    "Copper wire holds a branch in place until the wood remembers the new shape.",
    "Repotting happens in early spring, just as the buds begin to swell.",
    "A shallow tray of gravel keeps the humidity near the foliage through winter.",
    "Deadwood is preserved with care so the trunk keeps its story visible.",
)


class BenchBackend(Protocol):
    """What the harness needs from a slot."""

    async def perplexity(self) -> tuple[tuple[float, float] | None, str | None]:
        """((PPL, err) or None, note explaining an n/a)."""
        ...

    def max_haystack_tokens(self) -> int: ...

    async def ask(self, context: str, question: str) -> str: ...

    async def chat(self, messages: list[dict[str, str]]) -> str: ...


@dataclass
class ConfigResult:
    v_type: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None
    ppl: tuple[float, float] | None = None  # (PPL, +/- err) from llama-perplexity
    ppl_note: str | None = None  # why ppl is None (honest n/a), if it is
    # size -> depth -> retrieved?
    needle: dict[int, dict[int, bool]] = field(default_factory=dict)
    # size -> honest reason it was not run (e.g. slot ctx)
    needle_skipped: dict[int, str] = field(default_factory=dict)
    replay: tuple[int, int] | None = None  # (probes passed, probes total)

    @property
    def needle_passed(self) -> int:
        return sum(1 for per_size in self.needle.values() for ok in per_size.values() if ok)


# -- harness math ------------------------------------------------------------
def build_haystack(depth_pct: int, *, target_tokens: int = 4096) -> str:
    """Deterministic long context with NEEDLE_FACT planted at ~depth_pct% of its
    length. Token count is estimated at 4 chars/token (same estimate the memory
    layer uses — good enough for placement, and the report says 'synthetic')."""
    target_chars = target_tokens * 4
    sentences: list[str] = []
    length = 0
    i = 0
    while length < target_chars:
        sentence = _FILLER_SENTENCES[i % len(_FILLER_SENTENCES)]
        sentences.append(sentence)
        length += len(sentence) + 1
        i += 1
    insert_at = max(0, min(len(sentences) - 1, round(len(sentences) * depth_pct / 100)))
    sentences.insert(insert_at, NEEDLE_FACT)
    return " ".join(sentences)


def build_ppl_corpus(*, chunks: int = PPL_CHUNKS, chunk_ctx: int = PPL_CHUNK_CTX) -> str:
    """Deterministic synthetic corpus for llama-perplexity: enough estimated tokens
    for `chunks` x `chunk_ctx`. Real English words (from the bundled public-domain
    sample) deterministically shuffled into pseudo-sentences: repeated canned
    sentences would be memorized in-context (measured PPL ~1.0 — zero
    discriminative power for cache-quality deltas), so the corpus must stay
    UNPREDICTABLE while remaining reproducible. Absolute PPL on such text is
    meaningless by construction; only the deltas between V-cache configs matter."""
    target_chars = int(chunks * chunk_ctx * 4 * 1.25)  # 25% margin over the estimate
    raw = (SAMPLE_TEXT + " " + " ".join(_FILLER_SENTENCES)).split()
    words = sorted({w.strip(".,;:!?()—“”") for w in raw} - {""})
    rng = random.Random(20260721)
    parts: list[str] = []
    length = 0
    while length < target_chars:
        sentence = " ".join(rng.choice(words) for _ in range(rng.randint(6, 14))) + "."
        parts.append(sentence)
        length += len(sentence) + 1
    return " ".join(parts)


def required_slot_ctx(haystack_tokens: int) -> int:
    """Slot ctx needed to run a haystack of the given ESTIMATED token count: the
    4 chars/token estimate can undershoot, so budget 25% headroom plus room for
    the question and the answer."""
    return int(haystack_tokens * 1.25) + 1024


def slot_ctx_ladder(haystack_sizes: tuple[int, ...]) -> list[int]:
    """Descending ctx candidates, one per haystack size: the largest that boots
    on this hardware wins and caps which needle sizes run."""
    return [required_slot_ctx(size) for size in sorted(haystack_sizes, reverse=True)]


def max_haystack_for_ctx(slot_ctx: int, haystack_sizes: tuple[int, ...]) -> int:
    fitting = [s for s in haystack_sizes if required_slot_ctx(s) <= slot_ctx]
    return max(fitting) if fitting else 0


def parse_final_ppl(output: str) -> tuple[float, float] | None:
    """Parse llama-perplexity's 'Final estimate: PPL = 186.7474 +/- 5.47562'."""
    match = re.search(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", output)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def delta_vs_baseline(results: list[ConfigResult]) -> dict[str, float | None]:
    """PPL delta of each config vs the q8_0 baseline, in percent (None when either
    side is unavailable). Positive delta = worse (higher perplexity) than baseline."""
    baseline = next(
        (r.ppl for r in results if r.v_type == BASELINE_V and r.status == "ok"),
        None,
    )
    out: dict[str, float | None] = {}
    for result in results:
        if baseline is None or result.ppl is None:
            out[result.v_type] = None
        else:
            out[result.v_type] = round((result.ppl[0] - baseline[0]) / baseline[0] * 100, 2)
    return out


async def run_kv_bench(
    slot_provider: Callable[[str], contextlib.AbstractAsyncContextManager],
    *,
    haystack_sizes: tuple[int, ...] = NEEDLE_SIZES,
    depths: tuple[int, ...] = NEEDLE_DEPTHS,
    run_replay: bool = True,
) -> list[ConfigResult]:
    """Run the matrix. `slot_provider(v_type)` is an async context manager yielding
    `(backend | None, skip_reason)`; a None backend records a skipped config. Any
    exception inside a config becomes status "failed" — the matrix always finishes."""
    results: list[ConfigResult] = []
    for v_type in V_TYPES:
        try:
            async with slot_provider(v_type) as (backend, skip_reason):
                if backend is None:
                    results.append(
                        ConfigResult(v_type, "skipped", reason=skip_reason or "unavailable")
                    )
                    continue
                ppl, ppl_note = await backend.perplexity()
                needle: dict[int, dict[int, bool]] = {}
                needle_skipped: dict[int, str] = {}
                max_tokens = backend.max_haystack_tokens()
                for size in haystack_sizes:
                    if size > max_tokens:
                        needle_skipped[size] = (
                            f"not run: slot ctx on this hardware fits haystacks up to "
                            f"{max_tokens} tokens"
                        )
                        continue
                    needle[size] = {}
                    for depth in depths:
                        context = build_haystack(depth, target_tokens=size)
                        answer = await backend.ask(context, NEEDLE_QUESTION)
                        needle[size][depth] = NEEDLE_ANSWER in answer
                replay_score: tuple[int, int] | None = None
                if run_replay:
                    answers = [
                        await backend.chat(probe_messages(prefix, question))
                        for prefix, question, _ in REPLAY_PROBES
                    ]
                    replay_score = score_replay(answers)
                results.append(
                    ConfigResult(
                        v_type,
                        "ok",
                        ppl=ppl,
                        ppl_note=ppl_note,
                        needle=needle,
                        needle_skipped=needle_skipped,
                        replay=replay_score,
                    )
                )
        except Exception as exc:  # noqa: BLE001 - one broken config must not kill the run
            results.append(ConfigResult(v_type, "failed", reason=f"{type(exc).__name__}: {exc}"))
    return results


# -- report ------------------------------------------------------------------
def _needle_cell(result: ConfigResult, size: int, depths: tuple[int, ...]) -> str:
    if size in result.needle_skipped:
        return "not run*"
    per_size = result.needle.get(size)
    if not per_size:
        return "—"
    passed = sum(1 for ok in per_size.values() if ok)
    if passed == len(depths):
        return f"pass {passed}/{len(depths)}"
    failed_depths = ",".join(str(d) for d in depths if not per_size.get(d))
    return f"FAIL {passed}/{len(depths)} (at {failed_depths}%)"


def render_report(
    results: list[ConfigResult],
    *,
    machine_line: str,
    mock: bool,
    date: str | None = None,
    haystack_sizes: tuple[int, ...] = NEEDLE_SIZES,
    depths: tuple[int, ...] = NEEDLE_DEPTHS,
) -> str:
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    deltas = delta_vs_baseline(results)
    lines = [
        f"# suiban bench kv — {date}",
        "",
        f"Measured on your hardware: {machine_line}.",
        "These are LOCAL measurements of this machine, binary and weights — not "
        "official benchmarks. PPL comes from the fork's `llama-perplexity` over a "
        f"bundled SYNTHETIC corpus ({PPL_CHUNKS} x {PPL_CHUNK_CTX}-token chunks): "
        "absolute values are meaningless, the deltas vs the q8_0 baseline are the "
        "signal. The needle test plants a fact at 10/50/90% depth of a synthetic "
        "context; the replay test replays a canned ~30-turn transcript and checks "
        "that early-turn facts still come back at the end.",
        "",
    ]
    if mock:
        lines += [
            "**MOCK MODE** — the mock backend reports fixed fixture values. Nothing "
            "in this report is a measurement.",
            "",
        ]
    header = (
        "| V cache (K=q8_0) | status | PPL (llama-perplexity) | delta vs q8_0 | "
        + " | ".join(f"needle {s // 1024}k" for s in haystack_sizes)
        + " | replay |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (5 + len(haystack_sizes)))
    for result in results:
        if result.status == "ok":
            ppl_s = f"{result.ppl[0]:.2f} ± {result.ppl[1]:.3g}" if result.ppl else "n/a†"
            delta = deltas.get(result.v_type)
            delta_s = f"{delta:+.2f}%" if delta is not None else "n/a"
            needles = [_needle_cell(result, s, depths) for s in haystack_sizes]
            replay_s = f"{result.replay[0]}/{result.replay[1]}" if result.replay else "—"
            lines.append(
                f"| {result.v_type} | ok | {ppl_s} | {delta_s} | "
                + " | ".join(needles)
                + f" | {replay_s} |"
            )
        else:
            lines.append(
                f"| {result.v_type} | {result.status} | — | — | "
                + " | ".join("—" for _ in haystack_sizes)
                + " | — |"
            )
    skipped = [r for r in results if r.status != "ok"]
    if skipped:
        lines += ["", "## Skipped / failed configs", ""]
        for result in skipped:
            lines.append(f"- `{result.v_type}` — {result.status}: {result.reason}")
    # Configs can differ in ctx (a bigger V type may only boot at a smaller
    # rung), so collect the distinct reasons across all results.
    not_run_reasons = sorted(
        {reason.removeprefix("not run: ") for r in results for reason in r.needle_skipped.values()}
    )
    if not_run_reasons:
        lines += [
            "",
            "\\* needle sizes not run: "
            + "; ".join(not_run_reasons)
            + " (larger cards run the full ladder).",
        ]
    ppl_notes = sorted({r.ppl_note for r in results if r.status == "ok" and r.ppl_note})
    if ppl_notes:
        lines += ["", "† PPL n/a: " + "; ".join(ppl_notes) + "."]
    lines.append("")
    return "\n".join(lines)


# -- mock backend (fixtures, used by unit tests and SUIBAN_LLAMA_MOCK runs) ---
# FIXTURES for harness tests and mock-mode smoke runs. These are NOT measurements
# of anything; the report's MOCK MODE banner says so.
MOCK_FIXTURE_PPL: dict[str, tuple[float, float]] = {
    "q8_0": (10.00, 0.10),
    "q4_0": (10.10, 0.10),
    "tq4_0": (10.05, 0.10),
    "tq3_0": (10.20, 0.11),
}


class MockBenchBackend:
    """Deterministic stand-in: fixed fixture PPL, needle and replay always
    retrievable (the mock has no cache to degrade)."""

    def __init__(self, v_type: str) -> None:
        self._v_type = v_type

    async def perplexity(self) -> tuple[tuple[float, float] | None, str | None]:
        return MOCK_FIXTURE_PPL[self._v_type], None

    def max_haystack_tokens(self) -> int:
        return max(NEEDLE_SIZES)

    async def ask(self, context: str, question: str) -> str:
        return NEEDLE_ANSWER if NEEDLE_FACT in context else "I do not know."

    async def chat(self, messages: list[dict[str, str]]) -> str:
        question = messages[-1]["content"]
        transcript = " ".join(m["content"] for m in messages[:-1]).lower()
        for _, probe_question, expected in REPLAY_PROBES:
            if probe_question == question:
                return expected if expected.lower() in transcript else "I do not know."
        return "I do not know."


# -- llama-perplexity runner ---------------------------------------------------
def find_perplexity_binary(server_binary: Path) -> Path | None:
    """The fork's llama-perplexity tool next to the resolved llama-server binary
    (the TurboQuant install promotes the whole tool set into ~/.bonsai/bin/<backend>)."""
    candidate = server_binary.parent / "llama-perplexity"
    return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None


def measure_perplexity(
    server_binary: Path,
    model_path: Path,
    v_type: str,
    *,
    gpu: bool,
    k_type: str = K_TYPE,
    timeout: float = PPL_TIMEOUT_S,
) -> tuple[tuple[float, float] | None, str | None]:
    """Run llama-perplexity for one V-cache config. Returns ((ppl, err), None) or
    (None, honest_reason). Never raises: PPL is one column of the report, and the
    needle/replay measurements must proceed without it.

    Security seam (audit next session): spawns the installed binary from
    ~/.bonsai/bin/<backend>/ (same trust root as RealBackend's llama-server
    spawn) with a fixed argv — no shell, no user-controlled strings beyond the
    resolved model path; corpus goes through a private tempfile."""
    ppl_binary = find_perplexity_binary(server_binary)
    if ppl_binary is None:
        return None, (
            "no llama-perplexity next to the installed llama-server (prebuilt binaries "
            "ship without tools; `suiban install turboquant` builds them)"
        )
    env = dict(os.environ)
    # Same shared-lib treatment RealBackend gives llama-server (backend.py):
    lib_key = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
    env[lib_key] = str(ppl_binary.parent)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write(build_ppl_corpus())
        corpus_path = Path(handle.name)
    cmd = [
        str(ppl_binary),
        "-m",
        str(model_path),
        "-f",
        str(corpus_path),
        "-c",
        str(PPL_CHUNK_CTX),
        "--chunks",
        str(PPL_CHUNKS),
        "-fa",
        "on",
        "-ctk",
        k_type,
        "-ctv",
        v_type,
        "-ngl",
        "999" if gpu else "0",
        "--no-warmup",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, f"llama-perplexity timed out after {timeout:.0f}s"
    except OSError as exc:
        return None, f"llama-perplexity failed to start: {exc}"
    finally:
        corpus_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        return None, f"llama-perplexity exited {proc.returncode} for --cache-type-v {v_type}"
    parsed = parse_final_ppl(proc.stdout + proc.stderr)
    if parsed is None:
        return None, "llama-perplexity produced no 'Final estimate' line"
    return parsed, None


# -- real slot backend --------------------------------------------------------
class SlotBench:
    """BenchBackend over a live llama-server slot, with the PPL measured up front
    (llama-perplexity must not run while the server holds the model in VRAM)."""

    def __init__(
        self,
        backend: SlotBackend,
        model: str,
        *,
        ppl: tuple[float, float] | None,
        ppl_note: str | None,
        slot_ctx: int,
        haystack_sizes: tuple[int, ...] = NEEDLE_SIZES,
    ) -> None:
        self._backend = backend
        self._model = model
        self._ppl = ppl
        self._ppl_note = ppl_note
        self._max_haystack = max_haystack_for_ctx(slot_ctx, haystack_sizes)

    async def perplexity(self) -> tuple[tuple[float, float] | None, str | None]:
        return self._ppl, self._ppl_note

    def max_haystack_tokens(self) -> int:
        return self._max_haystack

    async def ask(self, context: str, question: str) -> str:
        return await self.chat([{"role": "user", "content": f"{context}\n\n{question}"}])

    async def chat(self, messages: list[dict[str, str]]) -> str:
        sampling = sampling_for(self._model)
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": 0.0,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            **thinking_payload_fields(0),
            "max_tokens": 64,
        }
        try:
            async with self._backend.client() as client:
                response = await client.post("/v1/chat/completions", json=payload, timeout=600.0)
                response.raise_for_status()
                return response.json()["choices"][0]["message"].get("content") or ""
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
            return ""


def real_slot_provider(
    *,
    compute_backend: str,
    family: str,
    model: str = "bonsai-27b",
    port: int = BENCH_PORT,
    use_mock: bool = False,
    haystack_sizes: tuple[int, ...] = NEEDLE_SIZES,
):
    """slot_provider factory for run_kv_bench: measures llama-perplexity offline,
    then spins the orchestrator model once per config with the config's
    --cache-type-v (largest slot ctx that boots wins), tears it down after."""
    from suiban.installer import models as model_store

    @contextlib.asynccontextmanager
    async def provide(v_type: str) -> AsyncIterator[tuple[BenchBackend | None, str | None]]:
        if v_type in TURBOQUANT_V_TYPES and not (
            use_mock or binary_mod.turboquant_installed(compute_backend)
        ):
            yield None, TQ_SKIP_REASON
            return
        if use_mock:
            yield MockBenchBackend(v_type), None
            return

        binary = binary_mod.resolve_server_binary(compute_backend)  # raises BonsaiError
        model_path = model_store.resolve_model_path(model, family)

        # PPL first: llama-perplexity needs the VRAM the slot is about to take.
        ppl, ppl_note = measure_perplexity(binary, model_path, v_type, gpu=compute_backend != "cpu")

        kv = KvState(
            k_type=K_TYPE,
            v_type=v_type,
            enabled=v_type in TURBOQUANT_V_TYPES,
            preset="recommended",
            backend_supported=True,
            fallback_active=False,
            fallback_reason=None,
        )
        real: RealBackend | None = None
        planned: PlannedSlot | None = None
        for ctx in slot_ctx_ladder(haystack_sizes):
            planned = PlannedSlot(
                slot_id="bench",
                role="orchestrator",
                model=model,
                family=family,
                ctx=ctx,
                gpu=0 if compute_backend != "cpu" else None,
                port=port,
                vram_mb=0,
            )
            flags = build_server_flags(planned, kv, model_path=model_path)
            candidate = RealBackend(planned, binary=binary, flags=flags)
            await candidate.start()
            if planned.state == "ready":
                real = candidate
                break
            # Did not boot at this ctx (VRAM or type rejection) — try smaller.
            await candidate.stop()
        if real is None or planned is None:
            # Every ladder rung failed: the binary refused the config (bad
            # --cache-type-v, missing kernel, FA constraint) or cannot boot at all.
            yield (
                None,
                (
                    f"llama-server did not become healthy with --cache-type-v {v_type} "
                    f"at any ctx in {slot_ctx_ladder(haystack_sizes)} "
                    "(type validation probe failed)"
                ),
            )
            return
        try:
            yield (
                SlotBench(
                    real,
                    model,
                    ppl=ppl,
                    ppl_note=ppl_note,
                    slot_ctx=planned.ctx,
                    haystack_sizes=haystack_sizes,
                ),
                None,
            )
        finally:
            await real.stop()

    return provide


def report_path(reports_dir: Path, date: str | None = None) -> Path:
    date = date or datetime.now(UTC).strftime("%Y-%m-%d")
    return reports_dir / f"bench-kv-{date}.md"
