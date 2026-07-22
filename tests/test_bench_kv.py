"""bench kv harness: haystack/corpus construction, llama-perplexity parsing and
fallback, ctx ladder, matrix runner with skips/failures, baseline deltas, replay
scoring, report rendering, and the TurboQuant probe short-circuit."""

from __future__ import annotations

import contextlib

from suiban.bench.kv import (
    BASELINE_V,
    MOCK_FIXTURE_PPL,
    NEEDLE_ANSWER,
    NEEDLE_DEPTHS,
    NEEDLE_FACT,
    NEEDLE_SIZES,
    PPL_CHUNK_CTX,
    PPL_CHUNKS,
    TQ_SKIP_REASON,
    V_TYPES,
    ConfigResult,
    MockBenchBackend,
    build_haystack,
    build_ppl_corpus,
    delta_vs_baseline,
    find_perplexity_binary,
    max_haystack_for_ctx,
    measure_perplexity,
    parse_final_ppl,
    real_slot_provider,
    render_report,
    report_path,
    required_slot_ctx,
    run_kv_bench,
    slot_ctx_ladder,
)
from suiban.bench.replay import (
    REPLAY_PROBES,
    REPLAY_TURNS,
    probe_messages,
    score_replay,
)
from suiban.bench.sample_text import SAMPLE_TEXT


# -- haystack / corpus --------------------------------------------------------
def test_haystack_plants_needle_at_requested_depth() -> None:
    for depth in NEEDLE_DEPTHS:
        haystack = build_haystack(depth, target_tokens=2048)
        assert NEEDLE_FACT in haystack
        position = haystack.index(NEEDLE_FACT) / len(haystack)
        assert abs(position - depth / 100) < 0.15, (depth, position)
    # Deterministic: same inputs, same haystack.
    assert build_haystack(50) == build_haystack(50)


def test_sample_text_is_nontrivial() -> None:
    assert len(SAMPLE_TEXT) > 2000  # ~1K-token public-domain sample
    assert "Four score and seven years ago" in SAMPLE_TEXT


def test_ppl_corpus_is_deterministic_large_and_unpredictable() -> None:
    corpus = build_ppl_corpus()
    assert corpus == build_ppl_corpus()
    # 4 chars/token estimate with 25% margin for chunks x chunk_ctx tokens
    assert len(corpus) >= PPL_CHUNKS * PPL_CHUNK_CTX * 4
    # real words from the bundled sample pool
    assert "the" in corpus.split()
    # shuffled pseudo-sentences, not canned repetition: a memorizable corpus
    # measures nothing (PPL ~1.0), so near-duplicate sentences must be rare
    sentences = corpus.split(".")
    assert len(set(sentences)) > 0.95 * len(sentences)


# -- llama-perplexity parsing / fallback --------------------------------------
def test_parse_final_ppl_real_fork_line() -> None:
    out = "....\nFinal estimate: PPL = 186.7474 +/- 5.47562\n"
    assert parse_final_ppl(out) == (186.7474, 5.47562)


def test_parse_final_ppl_missing_line_is_none() -> None:
    assert parse_final_ppl("llama_perplexity: tokenizing the input ..") is None


def test_find_perplexity_binary_requires_sibling(tmp_path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("")
    assert find_perplexity_binary(server) is None
    sibling = tmp_path / "llama-perplexity"
    sibling.write_text("#!/bin/sh\n")
    sibling.chmod(0o755)
    assert find_perplexity_binary(server) == sibling


def test_measure_perplexity_missing_tool_is_honest_note(tmp_path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("")
    ppl, note = measure_perplexity(server, tmp_path / "model.gguf", "tq4_0", gpu=False)
    assert ppl is None
    assert "no llama-perplexity" in note


def test_measure_perplexity_parses_tool_output(tmp_path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("")
    fake = tmp_path / "llama-perplexity"
    fake.write_text("#!/bin/sh\necho 'Final estimate: PPL = 12.3400 +/- 0.5000'\n")
    fake.chmod(0o755)
    ppl, note = measure_perplexity(server, tmp_path / "model.gguf", "tq4_0", gpu=False)
    assert ppl == (12.34, 0.5)
    assert note is None


def test_measure_perplexity_nonzero_exit_is_honest_note(tmp_path) -> None:
    server = tmp_path / "llama-server"
    server.write_text("")
    fake = tmp_path / "llama-perplexity"
    fake.write_text("#!/bin/sh\nexit 7\n")
    fake.chmod(0o755)
    ppl, note = measure_perplexity(server, tmp_path / "model.gguf", "tq4_0", gpu=False)
    assert ppl is None
    assert "exited 7" in note


# -- ctx ladder ----------------------------------------------------------------
def test_slot_ctx_ladder_descends_and_fits_sizes() -> None:
    ladder = slot_ctx_ladder(NEEDLE_SIZES)
    assert ladder == sorted(ladder, reverse=True)
    for size in NEEDLE_SIZES:
        assert required_slot_ctx(size) in ladder
        # 25% haystack-estimate headroom plus question/answer room
        assert required_slot_ctx(size) >= size + 1024


def test_max_haystack_for_ctx() -> None:
    assert max_haystack_for_ctx(required_slot_ctx(16384), NEEDLE_SIZES) == 16384
    assert max_haystack_for_ctx(required_slot_ctx(8192), NEEDLE_SIZES) == 8192
    assert max_haystack_for_ctx(required_slot_ctx(4096), NEEDLE_SIZES) == 4096
    assert max_haystack_for_ctx(1024, NEEDLE_SIZES) == 0


# -- replay fixture ------------------------------------------------------------
def test_replay_fixture_shape() -> None:
    assert len(REPLAY_TURNS) == 30
    roles = [t["role"] for t in REPLAY_TURNS]
    assert roles == ["user", "assistant"] * 15  # strict alternation
    for prefix, question, expected in REPLAY_PROBES:
        assert 0 < prefix <= len(REPLAY_TURNS)
        assert REPLAY_TURNS[prefix - 1]["role"] == "assistant"  # probe appends a user turn
        transcript = " ".join(t["content"] for t in REPLAY_TURNS[:prefix])
        assert expected.lower() in transcript.lower(), (question, expected)
        messages = probe_messages(prefix, question)
        assert messages[-1] == {"role": "user", "content": question}


def test_score_replay() -> None:
    expected = [probe[2] for probe in REPLAY_PROBES]
    assert score_replay(expected) == (len(REPLAY_PROBES), len(REPLAY_PROBES))
    wrong = ["nope"] * len(REPLAY_PROBES)
    assert score_replay(wrong) == (0, len(REPLAY_PROBES))
    mixed = [expected[0], *wrong[1:]]
    assert score_replay(mixed) == (1, len(REPLAY_PROBES))


# -- matrix runner ------------------------------------------------------------
def _provider(backends: dict):
    """backends: v_type -> BenchBackend | ("skip", reason) | ("raise", exc)."""

    @contextlib.asynccontextmanager
    async def provide(v_type: str):
        entry = backends[v_type]
        if isinstance(entry, tuple) and entry[0] == "skip":
            yield None, entry[1]
        elif isinstance(entry, tuple) and entry[0] == "raise":
            raise entry[1]
        else:
            yield entry, None

    return provide


class WeakNeedleBackend(MockBenchBackend):
    """Fails retrieval at deep positions — for exercising the FAIL path."""

    async def ask(self, context: str, question: str) -> str:
        position = context.index(NEEDLE_FACT) / len(context)
        return NEEDLE_ANSWER if position < 0.5 else "no idea"


class SmallCtxBackend(MockBenchBackend):
    """Slot ctx only fits the smallest haystack — for the not-run path."""

    def max_haystack_tokens(self) -> int:
        return 4096


class NoPplBackend(MockBenchBackend):
    """The installed binary has no llama-perplexity — honest n/a."""

    async def perplexity(self):
        return None, "no llama-perplexity next to the installed llama-server"


async def test_run_kv_bench_matrix_with_skips_and_failures() -> None:
    provider = _provider(
        {
            "tq4_0": MockBenchBackend("tq4_0"),
            "tq3_0": ("skip", TQ_SKIP_REASON),
            "q4_0": ("raise", RuntimeError("server exploded")),
            "q8_0": MockBenchBackend("q8_0"),
        }
    )
    results = await run_kv_bench(provider, haystack_sizes=(1024,))
    by_type = {r.v_type: r for r in results}
    assert [r.v_type for r in results] == list(V_TYPES)  # full matrix, always

    assert by_type["tq4_0"].status == "ok"
    assert by_type["tq4_0"].ppl == MOCK_FIXTURE_PPL["tq4_0"]
    assert by_type["tq4_0"].needle == {1024: {d: True for d in NEEDLE_DEPTHS}}
    assert by_type["tq4_0"].replay == (len(REPLAY_PROBES), len(REPLAY_PROBES))

    assert by_type["tq3_0"].status == "skipped"
    assert by_type["tq3_0"].reason == TQ_SKIP_REASON

    assert by_type["q4_0"].status == "failed"
    assert "server exploded" in by_type["q4_0"].reason

    assert by_type["q8_0"].status == "ok"

    deltas = delta_vs_baseline(results)
    expected = round(
        (MOCK_FIXTURE_PPL["tq4_0"][0] - MOCK_FIXTURE_PPL[BASELINE_V][0])
        / MOCK_FIXTURE_PPL[BASELINE_V][0]
        * 100,
        2,
    )
    assert deltas["tq4_0"] == expected
    assert deltas["q8_0"] == 0.0
    assert deltas["tq3_0"] is None  # skipped config has no number — never faked


async def test_needle_failures_are_recorded() -> None:
    provider = _provider(
        {
            "tq4_0": WeakNeedleBackend("tq4_0"),
            "tq3_0": ("skip", "x"),
            "q4_0": ("skip", "x"),
            "q8_0": MockBenchBackend("q8_0"),
        }
    )
    results = await run_kv_bench(provider, haystack_sizes=(1024,))
    weak = next(r for r in results if r.v_type == "tq4_0")
    assert weak.needle[1024][10] is True
    assert weak.needle[1024][90] is False
    assert weak.needle_passed < len(NEEDLE_DEPTHS)


async def test_oversized_haystacks_are_not_run_with_reason() -> None:
    provider = _provider(
        {
            "tq4_0": SmallCtxBackend("tq4_0"),
            "tq3_0": ("skip", "x"),
            "q4_0": ("skip", "x"),
            "q8_0": SmallCtxBackend("q8_0"),
        }
    )
    results = await run_kv_bench(provider, haystack_sizes=NEEDLE_SIZES)
    small = next(r for r in results if r.v_type == "tq4_0")
    assert sorted(small.needle) == [4096]
    assert sorted(small.needle_skipped) == [8192, 16384]
    assert "slot ctx" in small.needle_skipped[16384]


async def test_replay_can_be_disabled() -> None:
    provider = _provider(
        {
            "tq4_0": MockBenchBackend("tq4_0"),
            "tq3_0": ("skip", "x"),
            "q4_0": ("skip", "x"),
            "q8_0": MockBenchBackend("q8_0"),
        }
    )
    results = await run_kv_bench(provider, haystack_sizes=(1024,), run_replay=False)
    assert all(r.replay is None for r in results)


def test_delta_without_baseline_is_none() -> None:
    results = [
        ConfigResult("tq4_0", "ok", ppl=(10.0, 0.1)),
        ConfigResult("q8_0", "failed", reason="no baseline"),
    ]
    assert delta_vs_baseline(results) == {"tq4_0": None, "q8_0": None}


# -- report -------------------------------------------------------------------
def _sample_results() -> list[ConfigResult]:
    return [
        ConfigResult(
            "tq4_0",
            "ok",
            ppl=MOCK_FIXTURE_PPL["tq4_0"],
            needle={
                4096: {10: True, 50: True, 90: False},
                8192: {10: True, 50: True, 90: True},
            },
            needle_skipped={
                16384: "not run: slot ctx on this hardware fits haystacks up to 8192 tokens"
            },
            replay=(5, 6),
        ),
        ConfigResult("tq3_0", "skipped", reason=TQ_SKIP_REASON),
        ConfigResult(
            "q8_0",
            "ok",
            ppl=None,
            ppl_note="no llama-perplexity next to the installed llama-server",
            needle={
                4096: {10: True, 50: True, 90: True},
                8192: {10: True, 50: True, 90: True},
            },
            needle_skipped={
                16384: "not run: slot ctx on this hardware fits haystacks up to 8192 tokens"
            },
            replay=(6, 6),
        ),
    ]


def test_render_report_is_honest() -> None:
    report = render_report(
        _sample_results(),
        machine_line="Fake GPU 0 (24576 MiB); backend cuda; binary prism-b9596-9fcaed7",
        mock=False,
        date="2026-07-21",
    )
    assert "# suiban bench kv — 2026-07-21" in report
    assert "Measured on your hardware" in report
    assert "not official benchmarks" in report
    assert "MOCK MODE" not in report
    assert "| tq4_0 | ok |" in report
    assert "FAIL 2/3 (at 90%)" in report  # the failed 90% needle is visible
    assert "pass 3/3" in report
    assert "not run*" in report  # 16k did not fit — labeled, not faked
    assert "slot ctx" in report
    assert "5/6" in report  # replay column
    assert "n/a†" in report  # q8_0 without llama-perplexity stays honest
    assert "no llama-perplexity" in report
    assert "## Skipped / failed configs" in report
    assert TQ_SKIP_REASON in report


def test_render_report_deltas_when_baseline_present() -> None:
    results = [
        ConfigResult("tq4_0", "ok", ppl=(10.05, 0.1), needle={4096: {10: True}}, replay=(6, 6)),
        ConfigResult("q8_0", "ok", ppl=(10.00, 0.1), needle={4096: {10: True}}, replay=(6, 6)),
    ]
    report = render_report(
        results, machine_line="m", mock=False, haystack_sizes=(4096,), depths=(10,)
    )
    assert "+0.50%" in report
    assert "+0.00%" in report


def test_render_report_mock_banner() -> None:
    report = render_report(_sample_results(), machine_line="mock", mock=True)
    assert "MOCK MODE" in report
    assert "Nothing in this report is a measurement" in report


def test_report_path_uses_date(tmp_path) -> None:
    path = report_path(tmp_path, date="2026-07-21")
    assert path.name == "bench-kv-2026-07-21.md"


# -- real provider probes ------------------------------------------------------
async def test_real_provider_skips_turboquant_without_kernels(bonsai_home) -> None:
    # No TURBOQUANT marker exists under the (tmp) bonsai home -> TQ configs skip
    # up front with the standard fallback notice, touching no binary.
    provider = real_slot_provider(compute_backend="cpu", family="ternary", use_mock=False)
    async with provider("tq4_0") as (backend, reason):
        assert backend is None
        assert reason == TQ_SKIP_REASON


async def test_real_provider_mock_mode_yields_fixture_backend(bonsai_home) -> None:
    provider = real_slot_provider(compute_backend="cpu", family="ternary", use_mock=True)
    async with provider("tq4_0") as (backend, reason):
        assert reason is None
        assert await backend.perplexity() == (MOCK_FIXTURE_PPL["tq4_0"], None)
        assert backend.max_haystack_tokens() == max(NEEDLE_SIZES)


async def test_real_provider_missing_binary_becomes_failed_result(bonsai_home) -> None:
    provider = real_slot_provider(compute_backend="cpu", family="ternary", use_mock=False)
    results = await run_kv_bench(provider, haystack_sizes=(1024,))
    by_type = {r.v_type: r for r in results}
    assert by_type["tq4_0"].status == "skipped"  # kernels absent -> standard notice
    assert by_type["q8_0"].status == "failed"  # no binary installed in the tmp home
    assert "llama-server" in by_type["q8_0"].reason
