"""Session auto-titling (api.md §5 behavior note, additive 2026-07-21b).

After the FIRST completed exchange of a session whose title is still null, the
resident utility slot (thinking off) generates a concise title (<= 6 words, no quotes,
no trailing punctuation) and the store persists it. Titling is fire-and-forget: it is
scheduled as a background task after the response is on its way, and ANY failure just
leaves the title null — a chat must never break because a title could not be made.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from suiban.memory.store import MemoryStore

logger = logging.getLogger(__name__)

TITLE_SYSTEM_PROMPT = (
    "You title conversations. Reply with ONLY a concise title for the conversation — "
    "at most six words, no quotes, no trailing punctuation."
)
TITLE_MAX_WORDS = 6
TITLE_INPUT_MAX_CHARS = 2000
TITLE_TIMEOUT_S = 60.0

# (transcript excerpt) -> raw model output
GenerateTitleFn = Callable[[str], Awaitable[str]]

# Strong references to in-flight titling tasks (create_task alone is GC-bait).
_TASKS: set[asyncio.Task] = set()


def clean_title(raw: str) -> str:
    """Enforce the title rules on model output: collapse whitespace, strip wrapping
    quotes and trailing punctuation, clip to TITLE_MAX_WORDS."""
    title = " ".join(raw.split())
    title = title.strip("\"'“”‘’`")
    title = title.rstrip(".!?,;:").strip()
    return " ".join(title.split()[:TITLE_MAX_WORDS])


def title_input(messages: list[dict]) -> str:
    """The first user/assistant exchange, flattened and truncated — enough signal for
    a six-word title without resending a whole transcript."""
    parts: list[str] = []
    for role in ("user", "assistant"):
        text = next((str(m.get("content") or "") for m in messages if m.get("role") == role), "")
        if text:
            parts.append(f"{role}: {text}")
    return "\n".join(parts)[:TITLE_INPUT_MAX_CHARS]


async def maybe_title_session(
    store: MemoryStore, generate: GenerateTitleFn, session_id: str
) -> None:
    """Title the session iff its title is still null. Failures are logged and leave
    the title null — never raised."""
    try:
        transcript = store.session_transcript(session_id)
        if transcript is None or transcript["session"]["title"] is not None:
            return
        if not transcript["messages"]:
            return
        raw = await generate(title_input(transcript["messages"]))
        title = clean_title(raw)
        if title:
            store.set_session_title(session_id, title)
    except asyncio.CancelledError:  # app shutdown — an untitled session is fine
        raise
    except Exception as exc:  # noqa: BLE001 - titling must never crash a chat
        logger.warning("auto-titling session %s failed: %s", session_id, exc)


def schedule_titling(store: MemoryStore, generate: GenerateTitleFn, session_id: str) -> None:
    """Fire-and-forget background titling (runs after the response is sent)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop (sync callers) — titling is best-effort
    task = loop.create_task(maybe_title_session(store, generate, session_id))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def cancel_pending() -> None:
    """App-shutdown hygiene: stop any titling still in flight."""
    for task in list(_TASKS):
        task.cancel()
    for task in list(_TASKS):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _TASKS.clear()
