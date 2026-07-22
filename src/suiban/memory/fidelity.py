"""Compression-fidelity instrumentation: planted-fact synthetic conversations.

Used two ways (docs/memory.md §5):

- MECHANICS tests (tests/test_fidelity.py, modelless): a fake summarizer that echoes
  the key phrases it finds proves that any fact PRESENT IN THE SUMMARIZER OUTPUT
  survives folding and re-compression — the pipeline never loses what the model kept.
  That is the honest modelless claim; whether the MODEL keeps the facts is a model
  property.
- REAL-MODEL harness (tests/test_fidelity_live.py, behind SUIBAN_LIVE_FIDELITY=1):
  sends the planted-fact transcript through the live stack's utility model with the
  production SUMMARIZE_SYSTEM_PROMPT and measures fact survival by plain string
  containment (no judge model). The prompt in compression.py was tuned against this
  survival number; the harness asserts >= MIN_SURVIVAL.

Survival is deliberately dumb: casefolded substring containment of short distinctive
probes. No embeddings, no scoring model — a fact either made it into the summary
text or it did not.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SURVIVAL = 0.80

# (probe, statement-template). Probes are distinctive strings whose survival is
# checked by containment; statements plant them in early conversation turns. The
# mix mirrors what compression must not lose: names, numbers, decisions, ids,
# paths, versions, error strings, dates, amounts, URLs.
_FACT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("Dr. Yamane", "The reviewer for the sensor paper is Dr. Yamane from the Kyoto lab."),
    ("TCK-4821", "The blocking ticket for the release is [TCK-4821], owned by platform."),
    ("Friday 09:00 UTC", "We decided the deploy window is Friday 09:00 UTC, no exceptions."),
    ("0x7f3a9c", "The corrupted firmware image has checksum 0x7f3a9c."),
    ("/var/lib/bonsai/queue", "The stuck jobs live under /var/lib/bonsai/queue on the worker."),
    ("v2.19.3", "Rolling back to v2.19.3 fixed the regression, so we pinned it."),
    ("ECONNRESET", "The gateway fails with ECONNRESET roughly every four hours."),
    ("March 14th", "The audit deadline is March 14th; drafts are due a week before."),
    ("4,750 euros", "The approved hardware budget is 4,750 euros for the quarter."),
    (
        "https://status.example.net/incident/88",
        "The postmortem link is https://status.example.net/incident/88.",
    ),
)


@dataclass(frozen=True)
class PlantedFact:
    probe: str  # distinctive string; survival == containment of this
    statement: str  # the sentence planted early in the conversation


def plant_facts(n: int = 10) -> list[PlantedFact]:
    """Deterministic facts, cycling the templates with a numeric suffix past 10 so
    probes stay unique at any n."""
    facts: list[PlantedFact] = []
    for i in range(n):
        probe, statement = _FACT_TEMPLATES[i % len(_FACT_TEMPLATES)]
        if i >= len(_FACT_TEMPLATES):
            unique = f"{probe} (case {i + 1})"
            statement = statement.replace(probe, unique)
            probe = unique
        facts.append(PlantedFact(probe=probe, statement=statement))
    return facts


def synthetic_conversation(
    facts: list[PlantedFact], filler_turns: int = 30, filler_chars: int = 280
) -> list[dict]:
    """Planted-fact conversation: every fact is stated (and acknowledged) in the
    EARLY turns, then `filler_turns` user/assistant pairs of chatty noise follow —
    so compression's middle span holds the facts and the protected tail holds only
    filler, the worst case for fact survival."""
    messages: list[dict] = []
    for fact in facts:
        messages.append({"role": "user", "content": f"For the record: {fact.statement}"})
        messages.append({"role": "assistant", "content": "Noted, I will remember that."})
    filler = (
        "Anyway, let us keep chatting about nothing in particular; the weather keeps "
        "changing and there is always another tangent to wander down. "
    )
    filler = (filler * (filler_chars // len(filler) + 1))[:filler_chars]
    for i in range(filler_turns):
        messages.append({"role": "user", "content": f"(filler turn {i + 1}) {filler}"})
        messages.append({"role": "assistant", "content": f"(filler reply {i + 1}) {filler}"})
    return messages


def survival(text: str, facts: list[PlantedFact]) -> float:
    """Fraction of facts whose probe appears in `text` (casefolded containment)."""
    if not facts:
        return 1.0
    haystack = text.casefold()
    kept = sum(1 for fact in facts if fact.probe.casefold() in haystack)
    return kept / len(facts)


def surviving_probes(text: str, facts: list[PlantedFact]) -> list[str]:
    haystack = text.casefold()
    return [fact.probe for fact in facts if fact.probe.casefold() in haystack]
