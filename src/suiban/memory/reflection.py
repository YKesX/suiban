"""Post-task reflection (api.md §11 memory notes, additive 2026-07-21c).

After a completed chat/code exchange on the 27B ORCHESTRATOR — never workers, never
utility, never external providers; the chat router gates on slot role and the write
path re-checks it — a background, failure-tolerant completion asks whether the
exchange revealed a durable user fact or preference. The model either calls the
existing `memory_write` tool (server-enforced 27B-only, docs/memory.md §7) or answers
"none".

Kept cheap on purpose (8 GB-tier sanity): thinking off, small max_tokens, and at most
one reflection per session per REFLECTION_EVERY_N_EXCHANGES completed exchanges
(in-memory counter — a restart resets it, which only means one extra reflection).
Nothing extra is archived; any failure is logged and forgotten — a chat must never
break because reflection could not run.

Execution is registry-generic: whatever write tools the caller's registry holds can
be called (the chat router still passes memory_write only). One exception to the
single-completion rule: a skill_save/skill_improve rejected by the frontmatter
validator earns exactly ONE retry with the validator's message appended, then quiet
surrender (see reflect_once).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable

from suiban.effort import Sampling, thinking_payload_fields
from suiban.memory.compression import message_text
from suiban.memory.skills import SKILL_REJECTION_PREFIX
from suiban.tools.base import ToolContext, ToolResult
from suiban.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

REFLECTION_MODES = ("chat", "code")
REFLECTION_EVERY_N_EXCHANGES = 3  # reflect on exchange 1, then 4, 7, ...
REFLECTION_MAX_TOKENS = 256
REFLECTION_TIMEOUT_S = 60.0
EXCHANGE_MAX_CHARS = 3000
# Cap on the number of distinct sessions the in-memory exchange counter tracks. The
# counter is a per-session rate-limit key with no natural eviction — a long-lived
# server that sees many session_ids would otherwise grow it without bound (audit
# 2026-07-22). Bounded as an LRU: evicting an idle session's counter only means it
# reflects once more if it ever comes back, the same benign cost as a server restart.
MAX_TRACKED_SESSIONS = 4096

REFLECTION_SYSTEM_PROMPT = (
    "You are reflecting on the finished exchange below, after the fact. If it "
    "revealed a DURABLE user fact or preference that future sessions genuinely need "
    "(identity, standing preferences, decisions), call memory_write ONCE with a "
    "distilled entry. Otherwise reply with exactly: none. Never store secrets, "
    "one-off task details, or anything the user asked to forget."
)

# (payload) -> completion response; the chat router binds this to the orchestrator
# slot. Tests inject scripted callables.
CompleteFn = Callable[[dict], Awaitable[dict]]

# Strong references to in-flight reflection tasks (create_task alone is GC-bait).
_TASKS: set[asyncio.Task] = set()
# session_id -> completed-exchange count (the rate-limit key; in-memory only). Bounded
# LRU (MAX_TRACKED_SESSIONS): most-recently-touched at the end, oldest evicted first.
_EXCHANGE_COUNTS: OrderedDict[str, int] = OrderedDict()


def reset() -> None:
    """Fresh counters (app startup / tests) — the rate limit is in-memory only."""
    _EXCHANGE_COUNTS.clear()


def should_reflect(session_id: str) -> bool:
    """Count one completed exchange for the session; True on the 1st and then every
    REFLECTION_EVERY_N_EXCHANGES-th (1, 4, 7, ... with N=3). The counter is a bounded
    LRU: the touched session moves to the newest slot and, once more than
    MAX_TRACKED_SESSIONS sessions are tracked, the oldest is evicted."""
    count = _EXCHANGE_COUNTS.get(session_id, 0) + 1
    _EXCHANGE_COUNTS[session_id] = count
    _EXCHANGE_COUNTS.move_to_end(session_id)
    while len(_EXCHANGE_COUNTS) > MAX_TRACKED_SESSIONS:
        _EXCHANGE_COUNTS.popitem(last=False)
    return count % REFLECTION_EVERY_N_EXCHANGES == 1


def exchange_digest(messages: list[dict], final_text: str) -> str:
    """The latest user message plus the assistant's reply, truncated — enough signal
    for a durable-fact check without resending the whole conversation."""
    latest = next((message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")
    return f"user: {latest}\nassistant: {final_text}"[:EXCHANGE_MAX_CHARS]


SKILL_WRITE_TOOLS = ("skill_save", "skill_improve")


async def _execute_write_calls(
    registry: ToolRegistry, ctx: ToolContext, message: dict
) -> list[tuple[dict, ToolResult]]:
    """Run every tool call the reflection completion made that exists in ITS
    registry (calls to anything else are dropped as noise — the registry is the
    whitelist). Returns (call, result) pairs so the caller can inspect rejections."""
    executed: list[tuple[dict, ToolResult]] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        name = function.get("name") or ""
        if registry.get(name) is None:
            continue
        raw = function.get("arguments")
        if isinstance(raw, str):
            try:
                args = json.loads(raw or "{}")
            except ValueError:
                continue
        else:
            args = raw or {}
        if isinstance(args, dict) and not registry.validate_args(name, args):
            executed.append((call, await registry.run(name, args, ctx)))
    return executed


def _skill_rejection(executed: list[tuple[dict, ToolResult]]) -> tuple[dict, ToolResult] | None:
    """The first skill_save/skill_improve call rejected by the SCHEMA validator
    (recognized by the stable rejection prefix). Other errors — role enforcement,
    unknown tools, crashes — never trigger a retry: re-asking cannot fix them."""
    for call, result in executed:
        name = (call.get("function") or {}).get("name")
        if (
            name in SKILL_WRITE_TOOLS
            and result.status == "error"
            and result.content.startswith(SKILL_REJECTION_PREFIX)
        ):
            return call, result
    return None


async def reflect_once(
    complete: CompleteFn,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    model: str,
    sampling: Sampling,
    exchange_text: str,
) -> None:
    """ONE reflection completion (thinking off, small max_tokens) followed by the
    execution of any tool call it made that is in its registry. A plain-text answer
    ("none" or anything else) writes nothing — by design.

    Exactly one follow-up happens in exactly one case: a skill_save/skill_improve
    rejected by the frontmatter validator. The validator's full error message is
    appended (as the tool result of the failed call) and the model gets ONE chance
    to resend a corrected SKILL.md; a second rejection gives up quietly. No other
    error earns a retry, and nothing extra is archived either way."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REFLECTION_SYSTEM_PROMPT},
            {"role": "user", "content": exchange_text},
        ],
        "stream": False,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "max_tokens": REFLECTION_MAX_TOKENS,
        "tools": registry.openai_tools(),
        "tool_choice": "auto",
        **thinking_payload_fields(0),
    }
    response = await complete(payload)
    message = (response.get("choices") or [{}])[0].get("message") or {}
    executed = await _execute_write_calls(registry, ctx, message)

    rejected = _skill_rejection(executed)
    if rejected is None:
        return
    call, result = rejected
    retry_payload = {
        **payload,
        "messages": [
            *payload["messages"],
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": call.get("id", ""), "content": result.content},
        ],
    }
    retry_response = await complete(retry_payload)
    retry_message = (retry_response.get("choices") or [{}])[0].get("message") or {}
    await _execute_write_calls(registry, ctx, retry_message)  # 2nd rejection: give up


async def _reflect_safe(
    complete: CompleteFn,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    model: str,
    sampling: Sampling,
    exchange_text: str,
) -> None:
    try:
        await reflect_once(
            complete, registry, ctx, model=model, sampling=sampling, exchange_text=exchange_text
        )
    except asyncio.CancelledError:  # app shutdown — a skipped reflection is fine
        raise
    except Exception as exc:  # noqa: BLE001 - reflection must never crash anything
        logger.warning("post-task reflection failed for %s: %s", ctx.session_id, exc)


def schedule_reflection(
    complete: CompleteFn,
    registry: ToolRegistry,
    ctx: ToolContext,
    *,
    model: str,
    sampling: Sampling,
    session_id: str,
    exchange_text: str,
) -> None:
    """Fire-and-forget background reflection, rate-limited per session. The counter
    ticks on EVERY completed exchange; only the 1st and every Nth actually reflect."""
    if not should_reflect(session_id):
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return  # no event loop (sync callers) — reflection is best-effort
    task = loop.create_task(
        _reflect_safe(
            complete, registry, ctx, model=model, sampling=sampling, exchange_text=exchange_text
        )
    )
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)


async def cancel_pending() -> None:
    """App-shutdown hygiene: stop any reflection still in flight."""
    for task in list(_TASKS):
        task.cancel()
    for task in list(_TASKS):
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    _TASKS.clear()
