"""Compression-fidelity MECHANICS (memory/fidelity.py + compression.py), modelless.

The honest modelless claim: facts that the summarizer KEEPS in its output survive the
pipeline — the middle-span selection, the summary fold, and re-compression never lose
them. A fake summarizer that echoes exactly the planted probes it finds in its input
stands in for the model; whether the REAL model keeps facts is measured by the
SUIBAN_LIVE_FIDELITY=1 harness (test_fidelity_live.py) against the live stack."""

from __future__ import annotations

from suiban.memory import compression as comp
from suiban.memory import fidelity


def _echo_summarizer(facts: list[fidelity.PlantedFact]):
    """Keeps every planted probe it can see in the input, drops everything else —
    the mechanical stand-in for a perfectly fact-disciplined utility model."""

    async def summarize(text: str) -> str:
        kept = fidelity.surviving_probes(text, facts)
        return "Summary of older turns. Facts kept: " + "; ".join(kept)

    return summarize


def test_generator_is_deterministic_and_probes_unique() -> None:
    facts = fidelity.plant_facts(14)
    assert facts == fidelity.plant_facts(14)
    assert len({f.probe for f in facts}) == 14
    for fact in facts:
        assert fact.probe in fact.statement

    conversation = fidelity.synthetic_conversation(facts, filler_turns=5)
    early = " ".join(m["content"] for m in conversation[: 2 * len(facts)])
    assert fidelity.survival(early, facts) == 1.0  # every fact planted early
    tail = " ".join(m["content"] for m in conversation[2 * len(facts) :])
    assert fidelity.survival(tail, facts) == 0.0  # filler contains none


async def test_facts_in_summarizer_output_survive_compression() -> None:
    facts = fidelity.plant_facts(10)
    conversation = fidelity.synthetic_conversation(facts, filler_turns=60)
    messages = [{"role": "system", "content": "mode prompt"}, *conversation]
    slot_ctx = 8192  # conversation estimates well past 70% of this

    result = await comp.compress(messages, slot_ctx, _echo_summarizer(facts))
    assert result is not None
    summary = next(
        m["content"]
        for m in result.messages
        if str(m.get("content", "")).startswith(comp.SUMMARY_PREFIX)
    )
    rate = fidelity.survival(summary, facts)
    assert rate >= fidelity.MIN_SURVIVAL, f"survival {rate} after first compression"
    # With the echo summarizer the mechanics are lossless — anything below 1.0 here
    # means the pipeline itself dropped a fact, not the model.
    assert rate == 1.0


async def test_facts_survive_folding_and_recompression() -> None:
    """The second compression folds the previous summary into the new middle span;
    facts already inside the summary must pass through the summarizer input again."""
    facts = fidelity.plant_facts(10)
    conversation = fidelity.synthetic_conversation(facts, filler_turns=60)
    messages = [{"role": "system", "content": "mode prompt"}, *conversation]
    summarize = _echo_summarizer(facts)

    first = await comp.compress(messages, 8192, summarize)
    assert first is not None
    # The conversation keeps growing with pure filler; compress again (and again).
    grown = [*first.messages, *fidelity.synthetic_conversation([], filler_turns=40)]
    second = await comp.compress(grown, 8192, summarize)
    assert second is not None
    summaries = [
        m for m in second.messages if str(m.get("content", "")).startswith(comp.SUMMARY_PREFIX)
    ]
    assert len(summaries) == 1  # folded, not stacked
    rate = fidelity.survival(summaries[0]["content"], facts)
    assert rate >= fidelity.MIN_SURVIVAL
    assert rate == 1.0  # mechanics are lossless with a fact-keeping summarizer


async def test_protected_tail_facts_never_pass_through_the_summarizer() -> None:
    """Facts stated within the protected recent window stay VERBATIM — they are
    never entrusted to the summarizer at all."""
    facts = fidelity.plant_facts(4)
    filler = fidelity.synthetic_conversation([], filler_turns=30)
    tail_facts = [{"role": "user", "content": f.statement} for f in facts]
    messages = [{"role": "system", "content": "mode prompt"}, *filler, *tail_facts]

    seen_by_summarizer: list[str] = []

    async def spy_summarize(text: str) -> str:
        seen_by_summarizer.append(text)
        return "summary with no facts"

    result = await comp.compress(messages, 4096, spy_summarize)
    assert result is not None
    assert fidelity.survival("\n".join(seen_by_summarizer), facts) == 0.0
    verbatim_tail = " ".join(str(m.get("content", "")) for m in result.messages[-4:])
    assert fidelity.survival(verbatim_tail, facts) == 1.0
