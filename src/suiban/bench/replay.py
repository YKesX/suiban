"""Canned multi-turn agentic replay fixture for `suiban bench kv`.

A ~30-turn synthetic assistant session with concrete facts planted in the early
turns. The bench replays the transcript against a live slot (all prior turns
are canned, deterministic text — only the probe answers are generated) and
scores answer stability: does the model still retrieve turn-3 facts at turn 28
with a quantized V cache holding the whole conversation?

This is a SYNTHETIC, canned fixture (the report says so). It measures
multi-turn KV retention, not general model quality.
"""

from __future__ import annotations

# Facts planted early and probed late. Kept boring on purpose: retrieval, not
# reasoning, is what the KV cache can break.
FACT_CODENAME = "aogiri"
FACT_BUDGET = "4200 euro"
FACT_DATE = "October 9"
FACT_VENDOR = "Kiyomizu Pots"
FACT_SKUS = "17"
FACT_DISCOUNT = "MOSS15"

_U = "user"
_A = "assistant"

# 30 canned turns (15 exchanges). Assistant turns are fixed text so the replay
# is deterministic; the model never generates them.
REPLAY_TURNS: list[dict[str, str]] = [
    {"role": _U, "content": "I need help planning the autumn launch of our bonsai starter kits."},
    {
        "role": _A,
        "content": "Happy to help. Let's collect the basics first: internal codename, "
        "budget, target date, suppliers, and the product list.",
    },
    {"role": _U, "content": f"Internal codename is '{FACT_CODENAME}'. Please use it from now on."},
    {"role": _A, "content": f"Noted: the launch is codenamed '{FACT_CODENAME}'."},
    {"role": _U, "content": f"Marketing budget is {FACT_BUDGET} total. Hard cap."},
    {"role": _A, "content": f"Recorded a hard budget cap of {FACT_BUDGET}."},
    {"role": _U, "content": f"Launch date: {FACT_DATE}. The web shop goes live that morning."},
    {"role": _A, "content": f"Launch date set to {FACT_DATE}, shop live in the morning."},
    {
        "role": _U,
        "content": f"Ceramics come from {FACT_VENDOR}; they need four weeks lead time.",
    },
    {"role": _A, "content": f"Supplier logged: {FACT_VENDOR}, four weeks lead time."},
    {"role": _U, "content": f"The catalog has {FACT_SKUS} SKUs at launch, no more."},
    {"role": _A, "content": f"Catalog frozen at {FACT_SKUS} SKUs for launch."},
    {
        "role": _U,
        "content": f"Early-bird discount code is {FACT_DISCOUNT}, 15 percent, first week only.",
    },
    {"role": _A, "content": f"Discount code {FACT_DISCOUNT} recorded: 15 percent, first week."},
    {"role": _U, "content": "Draft a one-line teaser for the newsletter."},
    {
        "role": _A,
        "content": "Teaser: 'Tiny trees, big calm - our bonsai starter kits arrive this autumn.'",
    },
    {"role": _U, "content": "Good. What should the packaging checklist cover?"},
    {
        "role": _A,
        "content": "Packaging checklist: pot wrap, soil bag seal, tool roll, care card, "
        "outer box drop test, and a QR code to the care guide.",
    },
    {"role": _U, "content": "Add a task to photograph every kit before the shop goes live."},
    {
        "role": _A,
        "content": "Added: product photography for every kit, due before the shop goes live.",
    },
    {"role": _U, "content": "Which channels do we announce on first?"},
    {
        "role": _A,
        "content": "Suggested order: newsletter first, then the blog, then social channels "
        "on launch morning.",
    },
    {"role": _U, "content": "The care guide needs a section on winter dormancy. Note that."},
    {"role": _A, "content": "Noted: add a winter dormancy section to the care guide."},
    {"role": _U, "content": "Someone proposed bundling a humidity tray. Thoughts?"},
    {
        "role": _A,
        "content": "A humidity tray fits the starter positioning; it raises unit cost, so "
        "it should stay within the agreed budget cap.",
    },
    {"role": _U, "content": "Let's also plan a small press kit for local garden magazines."},
    {
        "role": _A,
        "content": "Press kit plan: two photos, a fact sheet, founder quote, and a sample "
        "discount for editors.",
    },
    {"role": _U, "content": "Summarize where we stand in one sentence."},
    {
        "role": _A,
        "content": "Launch planning is on track: naming, budget, date, supplier, catalog "
        "and promotion basics are all locked.",
    },
]

# Probes: (transcript prefix length, question, expected substring in the answer).
# Prefix lengths point at assistant-turn boundaries so the conversation stays
# alternating when the probe question is appended.
REPLAY_PROBES: list[tuple[int, str, str]] = [
    (
        30,
        "What is the internal codename of the launch? Answer with the codename only.",
        FACT_CODENAME,
    ),
    (30, "What is the total marketing budget? Answer with the amount only.", "4200"),
    (30, "On which date does the web shop go live? Answer with the date only.", FACT_DATE),
    (30, "Which supplier provides the ceramics? Answer with the supplier name only.", FACT_VENDOR),
    (30, "How many SKUs does the catalog have at launch? Answer with the number only.", FACT_SKUS),
    (30, "What is the early-bird discount code? Answer with the code only.", FACT_DISCOUNT),
]


def probe_messages(prefix_len: int, question: str) -> list[dict[str, str]]:
    """The canned transcript prefix plus the probe question as a final user turn."""
    return [*REPLAY_TURNS[:prefix_len], {"role": _U, "content": question}]


def score_replay(answers: list[str]) -> tuple[int, int]:
    """(passed, total) for answers aligned with REPLAY_PROBES."""
    passed = 0
    for answer, (_, _, expected) in zip(answers, REPLAY_PROBES, strict=True):
        if expected.lower() in (answer or "").lower():
            passed += 1
    return passed, len(REPLAY_PROBES)
