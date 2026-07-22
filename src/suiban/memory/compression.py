"""In-context compression at ~70% of the slot context (docs/memory.md §5).

Token counts are ESTIMATED (chars/4 + a small per-message overhead) — suiban does not
ship a tokenizer, and the estimate only has to be good enough to trigger comfortably
before the context is actually full. TODO(v1.1): use llama-server's /tokenize endpoint
for exact counts once the live-wiring pass connects real slots.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

TRIGGER_FRACTION = 0.70
CHARS_PER_TOKEN = 4
PER_MESSAGE_OVERHEAD_TOKENS = 4
# Floor of the protected verbatim tail; the effective window is adaptive — see
# keep_recent_messages().
KEEP_RECENT_MESSAGES = 4

SUMMARY_PREFIX = "Rolling conversation summary (older turns condensed):"

# The resident utility model's summarization prompt (chat routing binds it to the
# utility slot). Tuned for compression FIDELITY: planted-fact tests
# (tests/test_fidelity.py mechanics + the SUIBAN_LIVE_FIDELITY=1 real-model harness)
# measure whether concrete facts survive; the explicit keep-list below is the tuned
# discipline — vague "condense this" prompts drop names and numbers first, which is
# exactly what recall needs most.
SUMMARIZE_SYSTEM_PROMPT = (
    "You condense conversation transcripts. Reply with ONLY the condensed result — "
    "no preamble, no commentary. Preserve VERBATIM every concrete fact: names, "
    "numbers, dates, amounts, versions, file paths, URLs, error messages, decisions "
    "made, and any [ids], ticket codes, or other identifiers. Drop pleasantries, "
    "repetition, and filler first; never drop a specific detail to save space. When "
    "unsure whether something is a fact, keep it."
)


def wrap_fold_input(transcript: str) -> str:
    """Frame the transcript so the instruction comes LAST.

    Measured live (SUIBAN_LIVE_FIDELITY harness, 1-bit 1.7B): a raw role-prefixed
    dialogue as the user message makes the small model continue the conversational
    pattern ('Noted, I will remember that.') instead of summarizing — 0/10 fact
    survival. Delimiting the transcript and restating the task after it flips
    recency bias in our favor."""
    return (
        "<transcript>\n"
        f"{transcript}\n"
        "</transcript>\n"
        "Condense the transcript above now. Keep every concrete fact verbatim "
        "(names, numbers, dates, amounts, versions, paths, URLs, errors, decisions, "
        "[ids]). Reply with ONLY the condensed result."
    )


SummarizeFn = Callable[[str], Awaitable[str]]


def keep_recent_messages(slot_ctx: int) -> int:
    """Adaptive verbatim window: the protected recent tail scales with the slot
    context — 4 messages below 16K ctx, 6 at 16K, 8 at 32K and above. Small contexts
    need the room; big contexts can afford more verbatim recency."""
    if slot_ctx >= 32768:
        return 8
    if slot_ctx >= 16384:
        return 6
    return KEEP_RECENT_MESSAGES


def message_text(message: dict) -> str:
    """Flatten OpenAI message content (string or multimodal parts) to text."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, dict) and part.get("type") == "image_url":
                parts.append("[image]")
        return "\n".join(parts)
    return ""


def estimate_message_tokens(message: dict) -> int:
    """Per-message token estimate (the single source of truth `estimate_tokens` sums).
    Exposed so the overflow guard can maintain a running total incrementally instead of
    re-summing the whole conversation after every trim (memory/injection.py)."""
    total = len(message_text(message)) // CHARS_PER_TOKEN + PER_MESSAGE_OVERHEAD_TOKENS
    for call in message.get("tool_calls") or []:
        total += len(str(call)) // CHARS_PER_TOKEN
    return total


def estimate_tokens(messages: list[dict]) -> int:
    return sum(estimate_message_tokens(m) for m in messages)


def usage_fraction(messages: list[dict], slot_ctx: int) -> float:
    if slot_ctx <= 0:
        return 0.0
    return estimate_tokens(messages) / slot_ctx


def should_compress(messages: list[dict], slot_ctx: int) -> bool:
    return usage_fraction(messages, slot_ctx) >= TRIGGER_FRACTION


def _split(
    messages: list[dict], keep_recent: int = KEEP_RECENT_MESSAGES
) -> tuple[list[dict], list[dict], list[dict]]:
    """(leading system messages, compressible middle, kept recent tail).

    A previous rolling summary (system message with SUMMARY_PREFIX) is NOT part of the
    protected head — it belongs to the middle so subsequent compressions fold into one
    summary instead of stacking new ones."""
    head_end = 0
    while (
        head_end < len(messages)
        and messages[head_end].get("role") == "system"
        and not str(messages[head_end].get("content", "")).startswith(SUMMARY_PREFIX)
    ):
        head_end += 1
    body = messages[head_end:]
    keep = min(keep_recent, len(body))
    return list(messages[:head_end]), body[: len(body) - keep], body[len(body) - keep :]


@dataclass(frozen=True)
class CompressionResult:
    messages: list[dict]
    trigger_pct: float
    messages_summarized: int


async def compress(
    messages: list[dict], slot_ctx: int, summarize: SummarizeFn
) -> CompressionResult | None:
    """Replace the oldest span with a rolling summary produced by the resident utility
    model. Returns None when nothing (or too little) is compressible. The replaced
    messages are already verbatim in the archive — nothing is lost on disk."""
    if not should_compress(messages, slot_ctx):
        return None
    head, middle, tail = _split(messages, keep_recent_messages(slot_ctx))
    if len(middle) < 2:
        return None  # nothing meaningful to fold; the tail is protected
    transcript = "\n".join(f"{m.get('role', '?')}: {message_text(m)}" for m in middle)
    summary = (await summarize(wrap_fold_input(transcript))).strip()
    summary_message = {"role": "system", "content": f"{SUMMARY_PREFIX}\n{summary}"}
    trigger_pct = round(usage_fraction(messages, slot_ctx) * 100, 1)
    return CompressionResult(
        messages=[*head, summary_message, *tail],
        trigger_pct=trigger_pct,
        messages_summarized=len(middle),
    )
