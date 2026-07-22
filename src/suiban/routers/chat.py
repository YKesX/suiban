"""POST /v1/chat/completions — OpenAI-compatible chat + bonsai extensions (api.md §1).

Two execution paths:

- **Pass-through** (the request carries its own `tools`, a `tool_choice`, or a
  `response_format`): suiban forwards to the routed slot and returns/streams the
  slot's OpenAI-shaped answer — tool calls come back to the CLIENT (OpenAI semantics;
  `delta.tool_calls` in the default stream). Grammar constraint happens in
  llama-server (--jinja + json_schema/GBNF).
- **Agentic** (no client tools): the mode's system prompt + server-side tool registry
  drive the ReAct loop (agent/loop.py); the default stream envelope synthesizes
  byte-compatible OpenAI chunks from the loop's text deltas, `stream_events:true`
  streams the rich envelope.

Model routing: `bonsai-auto` -> orchestrator slot; a concrete non-resident model is a
409 (`model_not_resident`) — no mid-run loads. Image parts require the routed slot to
be the vision 27B, else 400 with a clear message.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Coroutine
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.requests import ClientDisconnect

from suiban.agent import events as ev
from suiban.agent.events import AgentEvent
from suiban.agent.loop import AgentLoop, BackendChat, tool_result_cap
from suiban.app import AppState, state_of
from suiban.config import Effort
from suiban.effort import (
    max_tool_iterations,
    sampling_for,
    thinking_budget,
    thinking_payload_fields,
)
from suiban.errors import BonsaiError
from suiban.llama.manager import LlamaSlot
from suiban.memory import compression as comp
from suiban.memory import injection as inj
from suiban.memory import reflection, titling
from suiban.memory.injection import MEMORY_CONTEXT_HEADER, PROJECT_CONTEXT_HEADER
from suiban.memory.service import MemoryService
from suiban.modes.registry import CHAT_MODES, MODES, system_prompt
from suiban.modes.ultra import UltraRun, UltraWorker
from suiban.providers.registry import EFFORT_TEMPERATURE, ProviderState
from suiban.sched.budget import MODELS
from suiban.tools.base import ToolContext
from suiban.tools.memory_tools import MemoryWriteTool
from suiban.tools.registry import ToolRegistry, build_registry, build_worker_registry

logger = logging.getLogger(__name__)

router = APIRouter()

_ROLES = ("system", "user", "assistant", "tool")
EFFORTS = ("low", "mid", "high", "xhigh", "max")

# session_id is client-supplied (api.md §1) and joined under ~/.bonsai/work to form the
# fs/shell tool jail root — an unsanitized id like "../../etc" would relocate the jail
# outside the bonsai home. The RAW id still keys the archive DB (parameterized SQL,
# safe); only the ON-DISK directory name is derived through this safe transform.
_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def safe_workdir_name(session_id: str) -> str:
    """Filesystem-safe directory name for a session's workdir jail. A strict
    [A-Za-z0-9_-] id (no separators, no dots) is used verbatim so existing sessions
    (`tg-<id>`, `anon-<hex>`, UUIDs) keep their directory; anything else — traversal
    attempts, unicode, empty — maps to a stable sha256 digest that can never escape."""
    if session_id and _SAFE_SESSION_RE.match(session_id):
        return session_id
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


# -- request parsing ---------------------------------------------------------
class ChatRequest:
    """Validated OpenAI subset + bonsai extensions. Unknown OpenAI fields are
    tolerated and ignored (third-party clients send plenty); the honored subset is
    checked strictly with contract-envelope 400s. `effort_default` is the settings
    override for requests that carry no effort (req.effort > settings.effort_default
    > the mode's default)."""

    def __init__(self, body: dict, *, effort_default: Effort | None = None) -> None:
        def bad(message: str, code: str = "validation_error") -> BonsaiError:
            return BonsaiError(400, message, code=code)

        model = body.get("model")
        if not isinstance(model, str) or not model:
            raise bad("'model' is required (bonsai-auto or an id from /v1/models)")
        self.model = model

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise bad("'messages' must be a non-empty array")
        for i, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") not in _ROLES:
                raise bad(f"messages[{i}].role must be one of {', '.join(_ROLES)}")
            content = message.get("content")
            if content is not None and not isinstance(content, str | list):
                raise bad(f"messages[{i}].content must be a string or an array of parts")
        self.messages: list[dict] = messages

        self.stream = bool(body.get("stream", False))
        self.stream_events = bool(body.get("stream_events", False))
        self.temperature = _opt_number(body, "temperature", 0.0, 2.0)
        self.top_p = _opt_number(body, "top_p", 0.0, 1.0)
        self.top_k = _opt_int(body, "top_k", 1, 1000)
        self.max_tokens = _opt_int(body, "max_tokens", 1, 10_000_000)

        stop = body.get("stop")
        if stop is not None and not isinstance(stop, str | list):
            raise bad("'stop' must be a string or an array of strings")
        self.stop = stop

        tools = body.get("tools")
        if tools is not None and not isinstance(tools, list):
            raise bad("'tools' must be an array of tool definitions")
        self.tools = tools
        self.tool_choice = body.get("tool_choice")
        response_format = body.get("response_format")
        if response_format is not None and not isinstance(response_format, dict):
            raise bad("'response_format' must be an object")
        self.response_format = response_format

        mode = body.get("mode", "chat")
        if mode == "deep_research":
            raise bad(
                "deep_research is not a chat mode — submit it as an async job: "
                "POST /v1/jobs {type: 'deep_research', query: ...}",
                code="mode_not_chat",
            )
        if mode not in CHAT_MODES:
            raise bad(f"mode must be one of {', '.join(CHAT_MODES)}; got {mode!r}")
        self.mode = mode

        # Provider-prefixed ids route to an external provider (api.md §1, additive
        # 2026-07-21c); external sessions are chat-only — no code/ultra loops on
        # backends we cannot schedule, grammar-constrain, or jail.
        self.external = "/" in self.model
        if self.external and self.mode != "chat":
            raise bad(
                f"external model {self.model!r} supports mode 'chat' only; got mode {self.mode!r}",
                code="external_model_mode",
            )

        effort = body.get("effort")
        if effort is not None and effort not in EFFORTS:
            raise bad(f"effort must be one of {', '.join(EFFORTS)}; got {effort!r}")
        self.effort = effort or effort_default or MODES[self.mode].default_effort

        session_id = body.get("session_id")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise bad("'session_id' must be a non-empty string")
        self.session_id = session_id

        project_id = body.get("project_id")
        if project_id is not None and (not isinstance(project_id, str) or not project_id):
            raise bad("'project_id' must be a non-empty string")
        self.project_id = project_id

        workdir = body.get("workdir")
        if workdir is not None:
            if not isinstance(workdir, str) or not workdir:
                raise bad("'workdir' must be a non-empty string", code="workdir_invalid")
            if self.mode != "code":
                raise bad(
                    f"'workdir' is only valid with mode 'code'; got mode {self.mode!r}",
                    code="workdir_invalid",
                )
        self.workdir = workdir

        # auto_confirm (api.md 2026-07-22b): code/ultra only — bypasses the confirm gate
        # for destructive shell commands AND file mutations. 400 in chat mode.
        auto_confirm = body.get("auto_confirm", False)
        if not isinstance(auto_confirm, bool):
            raise bad("'auto_confirm' must be a boolean")
        if auto_confirm and self.mode not in ("code", "ultra"):
            raise bad(
                f"'auto_confirm' is only valid with mode 'code' or 'ultra'; got mode {self.mode!r}",
                code="auto_confirm_mode",
            )
        self.auto_confirm = auto_confirm

        # Set by the endpoint after validation against the live config home.
        self.workdir_path: Path | None = None
        # Client identity (X-Bonsai-Client header): dai|sentei|other. Selects the
        # identity overlay merged into the system prompt; the endpoint sets it.
        self.client: str = "other"
        # Names of skills injected into this request's context (_inject_skill_context)
        # — flipped to verified when the run completes successfully.
        self.injected_skills: list[str] = []

    @property
    def workdir_persist(self) -> str | None:
        """What to remember on the session row: the validated custom workdir only —
        the default per-session jail is never persisted."""
        return str(self.workdir_path) if self.workdir_path is not None else None

    @property
    def has_images(self) -> bool:
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False

    @property
    def passthrough(self) -> bool:
        return self.tools is not None or self.response_format is not None


def _opt_number(body: dict, key: str, lo: float, hi: float) -> float | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not lo <= value <= hi:
        raise BonsaiError(400, f"'{key}' must be a number in [{lo}, {hi}]", code="validation_error")
    return float(value)


def _opt_int(body: dict, key: str, lo: int, hi: int) -> int | None:
    value = body.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise BonsaiError(
            400, f"'{key}' must be an integer in [{lo}, {hi}]", code="validation_error"
        )
    return int(value)


# -- routing -----------------------------------------------------------------
def _route_slot(state: AppState, req: ChatRequest) -> LlamaSlot:
    loadout = state.loadout
    if req.model == "bonsai-auto":
        planned = loadout.orchestrator
        if planned is None:  # cannot happen with the current planner; stay honest
            raise BonsaiError(500, "no orchestrator slot in the loadout", code="no_orchestrator")
    elif req.model in MODELS:
        planned = loadout.slot_for_model(req.model)
        if planned is None:
            raise BonsaiError(
                409,
                f"model {req.model} is not resident in the active loadout "
                "(no mid-run loads; see /v1/models for resident models)",
                code="model_not_resident",
            )
    else:
        raise BonsaiError(
            400,
            f"unknown model {req.model!r}; use bonsai-auto or an id from /v1/models",
            code="unknown_model",
        )

    if req.has_images and not (planned.model == "bonsai-27b" and planned.mmproj):
        raise BonsaiError(
            400,
            "image parts require the vision-capable 27B orchestrator; this request "
            f"routes to {planned.model} (slot {planned.slot_id}). Send images with "
            "model bonsai-auto or bonsai-27b on a loadout where the 27B is resident.",
            code="vision_unavailable",
        )

    slot = state.manager.slot(planned.slot_id)
    if slot is None or slot.state != "ready":
        raise BonsaiError(
            409,
            f"slot {planned.slot_id} ({planned.model}) is not ready "
            f"(state: {slot.state if slot else 'missing'})",
            code="slot_unavailable",
        )
    return slot


def _utility_slot(state: AppState) -> LlamaSlot | None:
    for slot_id in ("utility", "orchestrator"):
        slot = state.manager.slot(slot_id)
        if slot is not None and slot.state == "ready":
            return slot
    return None


# -- client-disconnect abort + per-slot queueing ------------------------------
# Starlette cancels STREAMING response generators when the client drops, so those
# paths abort naturally (cancellation propagates into the in-flight httpx call and
# llama-server sees the connection close). Non-streaming handlers get no such signal:
# without the watcher below, a closed dai tab left the whole agent run decoding on a
# dead socket (observed live — orphaned Ultra runs burning GPU for minutes).
CLIENT_DISCONNECT_POLL_S = 1.0
# stream_events runs that wait on a busy slot for longer than this emit ONE
# "queued behind N" notice; sub-second waits stay silent (nothing actionable).
QUEUE_NOTICE_AFTER_S = 1.0


async def await_watching_disconnect(request: Request, coro: Coroutine) -> Any:
    """Await `coro` unless the client disconnects first (asyncio.wait pattern).

    On disconnect the backend task is CANCELLED — the CancelledError propagates into
    the in-flight httpx call, the connection to llama-server closes, and the slot
    stops decoding — then ClientDisconnect is raised (app.py answers it with an
    empty 204 nobody will read)."""
    task = asyncio.ensure_future(coro)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=CLIENT_DISCONNECT_POLL_S)
            if task in done:
                return task.result()
            if await request.is_disconnected():
                raise ClientDisconnect
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# -- projects (api.md §9: validation + excerpt injection) --------------------
# Injection HEADERS live in memory/injection.py (one owner: the overflow guard must
# recognize exactly the delimiters the injectors write); they are re-exported above.
PROJECT_CONTEXT_FRACTION = 0.10  # of the slot ctx (token estimate, chars/4)
PROJECT_DOC_LIMIT = 4


def _require_project(state: AppState, project_id: str | None) -> None:
    if project_id is None:
        return
    if state.memory.store.get_project(project_id) is None:
        raise BonsaiError(404, f"no such project: {project_id}", code="project_not_found")


def _latest_user_text(messages: list[dict]) -> str:
    return next((comp.message_text(m) for m in reversed(messages) if m.get("role") == "user"), "")


def coalesce_system_messages(messages: list[dict]) -> list[dict]:
    """Fold every system message into ONE leading system message.

    The Bonsai chat template hard-rejects anything else — verified live against the
    pinned fork slot: a second system message or a system message after a user turn
    400s with raise_exception('System message must be first'). Injected context
    blocks (project docs, memory recall), the mode prompt, and any client-sent
    system messages therefore merge in encounter order, joined by blank lines."""
    systems = [m.get("content") or "" for m in messages if m.get("role") == "system"]
    leading = 1 if messages and messages[0].get("role") == "system" else 0
    if len(systems) == leading:  # zero systems, or exactly one already in front
        return list(messages)
    rest = [m for m in messages if m.get("role") != "system"]
    joined = "\n\n".join(s for s in systems if s)
    return [{"role": "system", "content": joined}, *rest]


def _inject_project_context(state: AppState, req: ChatRequest, ctx_tokens: int) -> None:
    """Prepend a clearly-delimited system message with the top FTS5 doc excerpts for
    the latest user message. Budgeted like compression (chars/4 token estimate) so the
    block can never crowd the context; injected before compression runs so the
    trigger math accounts for it. `ctx_tokens` is the routed slot's context (or the
    external stand-in — external providers don't tell us theirs)."""
    if req.project_id is None:
        return
    latest = _latest_user_text(req.messages)
    if not latest.strip():
        return
    hits = state.memory.store.search_project_docs(req.project_id, latest, limit=PROJECT_DOC_LIMIT)
    if not hits:
        return
    budget_chars = int(ctx_tokens * PROJECT_CONTEXT_FRACTION) * comp.CHARS_PER_TOKEN
    blocks: list[str] = []
    used = 0
    for hit in hits:
        block = f"<<<doc: {hit['title']}>>>\n{hit['excerpt']}\n<<<end doc>>>"
        if blocks and used + len(block) > budget_chars:
            break
        blocks.append(block[:budget_chars])
        used += len(blocks[-1])
    content = PROJECT_CONTEXT_HEADER + "\n" + "\n".join(blocks)
    req.messages = [{"role": "system", "content": content}, *req.messages]


# -- automatic memory recall (api.md §11 notes, additive 2026-07-21c) ---------
MEMORY_CONTEXT_FRACTION = 0.05  # of the slot ctx (token estimate, chars/4)
MEMORY_HIT_LIMIT = 3


def _inject_memory_context(state: AppState, req: ChatRequest, ctx_tokens: int) -> None:
    """Light automatic recall: top FTS5 MEMORY hits (identity/state/archive entries
    — never raw transcripts) for the latest user message, injected as ONE delimited
    system message, budget-capped exactly like the project-doc injection. Chat/code
    only; skipped when nothing matches. Deliberate transcript digs stay behind the
    memory_search/session_search tools (docs/memory.md §3)."""
    if req.mode not in ("chat", "code"):
        return
    latest = _latest_user_text(req.messages)
    if not latest.strip():
        return
    hits = state.memory.store.search(latest, limit=MEMORY_HIT_LIMIT)
    if not hits:
        return
    budget_chars = int(ctx_tokens * MEMORY_CONTEXT_FRACTION) * comp.CHARS_PER_TOKEN
    blocks: list[str] = []
    used = 0
    for hit in hits:
        entry = hit["entry"]
        block = (
            f"<<<memory {entry['id']} ({entry['layer']}) {entry['title']}>>>\n"
            f"{hit['snippet']}\n<<<end memory>>>"
        )
        if blocks and used + len(block) > budget_chars:
            break
        blocks.append(block[:budget_chars])
        used += len(blocks[-1])
    content = MEMORY_CONTEXT_HEADER + "\n" + "\n".join(blocks)
    req.messages = [{"role": "system", "content": content}, *req.messages]


# -- client identity (api.md §1 X-Bonsai-Client, additive 2026-07-22b) --------
_CLIENTS = ("dai", "sentei", "other")


def _client_identity(header: str | None) -> str:
    """Normalize the X-Bonsai-Client header to dai|sentei|other (default other)."""
    value = (header or "").strip().lower()
    return value if value in _CLIENTS else "other"


def _inject_identity_context(state: AppState, req: ChatRequest) -> None:
    """Prepend the base identity.md PLUS the matching client overlay as a leading system
    message (api.md §1, additive 2026-07-22b): sentei → the coding overlay, dai → the
    general overlay, other/unknown → base only. Coalesced into the single system prompt
    downstream (coalesce_system_messages). Trusted persona content — NOT a delimited
    untrusted block. Skipped only when there is no identity content at all."""
    files = state.memory.files
    base = files.identity().strip()
    overlay = files.client_overlay(req.client).strip()
    blocks = [b for b in (base, overlay) if b]
    if not blocks:
        return
    req.messages = [{"role": "system", "content": "\n\n".join(blocks)}, *req.messages]


# -- skill injection (docs/memory.md §6: verified-first, [unverified] labeled) --
def _inject_skill_context(state: AppState, req: ChatRequest, ctx_tokens: int) -> None:
    """Inject skills whose name/description match the latest user message as ONE
    delimited system message — verified skills first, unverified ones labeled
    `[unverified]` (memory/injection.py owns selection, labeling, and budget).
    Local AGENTIC chat/code runs only: pass-through requests run the client's tools
    (skills would be dead weight) and external models never verify anything. The
    injected names are remembered on the request; a successful run flips them to
    verified (_mark_skills_verified)."""
    if req.passthrough or req.mode not in ("chat", "code"):
        return
    latest = _latest_user_text(req.messages)
    if not latest.strip():
        return
    content, names = inj.build_skill_context(state.memory.skills.list(), latest, ctx_tokens)
    if content is None:
        return
    req.injected_skills = names
    req.messages = [{"role": "system", "content": content}, *req.messages]


def _mark_skills_verified(state: AppState, req: ChatRequest, finish_reason: str | None) -> None:
    """A run that USED injected skills and completed successfully verifies them
    (persisted in the skills meta store; docs/memory.md §6). Error finishes verify
    nothing — an unproven skill stays labeled until a clean run."""
    if finish_reason == "error":
        return
    for name in req.injected_skills:
        state.memory.skills.mark_verified(name)


# -- code-mode workdir (api.md §1 `workdir`, additive 2026-07-21b) -----------
def validate_workdir(home: Path, raw: str, *, origin: str = "workdir") -> Path:
    """Enforce the api.md workdir rules: absolute path, exists, is a directory, and
    not inside ~/.bonsai (symlinks resolved FIRST, so a link cannot smuggle the jail
    into the home). Returns the resolved path — it becomes the jail root, and the
    fs/shell/git_ro symlink-escape defenses (tools/fs.resolve_in_jail) apply to it
    exactly as they do to the default per-session dir."""

    def bad(reason: str) -> BonsaiError:
        return BonsaiError(400, f"invalid {origin} {raw!r}: {reason}", code="workdir_invalid")

    candidate = Path(raw)
    if not candidate.is_absolute():
        raise bad("the path must be absolute")
    resolved = candidate.resolve()
    if not resolved.exists():
        raise bad("the directory does not exist")
    if not resolved.is_dir():
        raise bad("the path is not a directory")
    home_resolved = home.resolve()
    if resolved == home_resolved or home_resolved in resolved.parents:
        raise bad(f"the bonsai home ({home_resolved}) is off limits to session tools")
    return resolved


def _validate_request_workdir(state: AppState, req: ChatRequest) -> None:
    if req.workdir is not None:
        req.workdir_path = validate_workdir(state.config.home, req.workdir)


def _session_workdir(state: AppState, req: ChatRequest, session_id: str) -> Path:
    """The jail root for this run: an explicit (validated) code-mode workdir wins; a
    continued code session reuses its remembered workdir; everything else gets the
    default per-session dir under ~/.bonsai/work/."""
    if req.workdir_path is not None:
        return req.workdir_path
    if req.mode == "code" and req.session_id:
        stored = state.memory.store.session_workdir(req.session_id)
        if stored is not None:
            try:
                return validate_workdir(
                    state.config.home, stored, origin="the session's remembered workdir"
                )
            except BonsaiError as exc:
                raise BonsaiError(
                    400,
                    f"{exc.message} — pass a new 'workdir' to repoint this session",
                    code="workdir_invalid",
                ) from exc
    workdir = state.config.home / "work" / safe_workdir_name(session_id)
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def _make_summarizer(state: AppState):
    """Rolling-summary / recall-digest function backed by the resident utility slot
    (the orchestrator itself on CPU-only loadouts)."""
    slot = _utility_slot(state)
    if slot is None:
        return None
    sampling = sampling_for(slot.model)

    async def summarize(text: str) -> str:
        payload = {
            "model": slot.model,
            "messages": [
                # The fidelity-tuned prompt (memory/compression.py) — planted-fact
                # tests measure what it must not lose.
                {"role": "system", "content": comp.SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            **thinking_payload_fields(0),
        }
        chat = BackendChat(slot.backend)
        response = await chat.complete(payload, timeout=120.0)
        return response["choices"][0]["message"].get("content") or ""

    return summarize


def _make_titler(state: AppState):
    """Session-title generator on the resident utility slot, thinking OFF (session
    auto-titling, additive 2026-07-21b). None when no usable slot is ready."""
    slot = _utility_slot(state)
    if slot is None:
        return None
    sampling = sampling_for(slot.model)

    async def generate(text: str) -> str:
        payload = {
            "model": slot.model,
            "messages": [
                {"role": "system", "content": titling.TITLE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "stream": False,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            **thinking_payload_fields(0),
        }
        chat = BackendChat(slot.backend)
        response = await chat.complete(payload, timeout=titling.TITLE_TIMEOUT_S)
        return response["choices"][0]["message"].get("content") or ""

    return generate


def _schedule_auto_title(state: AppState, session_id: str | None) -> None:
    """Background auto-titling after a completed exchange; no-op for anonymous
    sessions, sessions already titled, or loadouts with no ready slot."""
    if not session_id:
        return
    generate = _make_titler(state)
    if generate is None:
        return
    titling.schedule_titling(state.memory.store, generate, session_id)


def _schedule_reflection(
    state: AppState, req: ChatRequest, slot: LlamaSlot, final_text: str
) -> None:
    """Post-task reflection (api.md §11 notes, additive 2026-07-21c): background,
    rate-limited, ORCHESTRATOR-only. Worker/utility slots never reflect (the
    one-writer rule), external sessions never reach this (they have no slot), ultra
    is excluded by mode, and anonymous exchanges are skipped — the once-per-N rate
    limit is keyed by session."""
    if not req.session_id or req.mode not in reflection.REFLECTION_MODES:
        return
    if slot.role != "orchestrator":
        return
    chat = BackendChat(slot.backend)

    async def complete(payload: dict) -> dict:
        return await chat.complete(payload, timeout=reflection.REFLECTION_TIMEOUT_S)

    # A registry holding ONLY memory_write: the reflection call can persist a durable
    # fact or answer "none" — nothing else is even in its schema.
    registry = ToolRegistry([MemoryWriteTool(state.memory)])
    ctx = ToolContext(
        session_id=req.session_id,
        workdir=state.config.home / "work" / safe_workdir_name(req.session_id),
        role=slot.role,
        mode=req.mode,
    )
    reflection.schedule_reflection(
        complete,
        registry,
        ctx,
        model=slot.model,
        sampling=sampling_for(slot.model),
        session_id=req.session_id,
        exchange_text=reflection.exchange_digest(req.messages, final_text),
    )


# -- session recording -------------------------------------------------------
def _new_request_messages(memory: MemoryService, session_id: str, messages: list[dict]) -> list:
    """Which request messages to archive: on a fresh session, all non-system ones; on
    a continued session, the tail after the last assistant message (OpenAI clients
    resend full history — earlier messages are already recorded)."""
    transcript = memory.store.session_transcript(session_id)
    fresh = transcript is None or not transcript["messages"]
    non_system = [m for m in messages if m.get("role") != "system"]
    if fresh:
        return non_system
    last_assistant = -1
    for i, message in enumerate(non_system):
        if message.get("role") == "assistant":
            last_assistant = i
    return non_system[last_assistant + 1 :]


def _record_exchange(
    memory: MemoryService,
    session_id: str | None,
    mode: str,
    request_messages: list[dict],
    tool_messages: list[dict],
    assistant_text: str,
    project_id: str | None = None,
    workdir: str | None = None,
) -> None:
    if not session_id:
        return
    new_messages = _new_request_messages(memory, session_id, request_messages)
    memory.store.ensure_session(session_id, mode, project_id, workdir)
    for message in new_messages:
        memory.store.add_message(
            session_id, message.get("role", "user"), comp.message_text(message)
        )
    for message in tool_messages:
        memory.store.add_message(session_id, "tool", message.get("content", ""))
    if assistant_text:
        memory.store.add_message(session_id, "assistant", assistant_text)


# -- OpenAI response shaping -------------------------------------------------
def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _usage_payload(usage: dict) -> dict:
    out = {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
    }
    if usage.get("thinking_tokens"):
        out["thinking_tokens"] = usage["thinking_tokens"]
    return out


def _bonsai_ext(req: ChatRequest, slot: LlamaSlot) -> dict:
    return {
        "mode": req.mode,
        "effort": req.effort,
        "slot": slot.slot_id,
        "session_id": req.session_id,
    }


def _openai_chunk(completion_id: str, model: str, delta: dict, finish: str | None = None) -> str:
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# -- the endpoint ------------------------------------------------------------
@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    state = state_of(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")

    # Activity tracking: /v1/system/apply commits staged settings only at idle, so
    # every chat run — including the full lifetime of a stream — counts as activity.
    state.activity.begin()
    try:
        req = ChatRequest(body, effort_default=state.config.settings.effort_default)
        req.client = _client_identity(request.headers.get("x-bonsai-client"))
        _require_project(state, req.project_id)
        _validate_request_workdir(state, req)
        if req.external:
            provider, model = state.providers.resolve(req.model)
            _inject_project_context(state, req, EXTERNAL_CONTEXT_TOKENS)
            _inject_memory_context(state, req, EXTERNAL_CONTEXT_TOKENS)
            _inject_identity_context(state, req)
            response = await _run_external(state, req, provider, model)
        else:
            # Lazy residency (api.md 2026-07-22c): warm the planned loadout on demand
            # before routing to a slot. A cold start blocks until the slots are healthy
            # (reusing start_all + the health wait); the returned flag surfaces a
            # `warming_up` notice on rich streams so the slow first token has a reason.
            warming = await state.load.ensure_loaded()
            slot = _route_slot(state, req)
            # Bounded per-slot queue: reject HERE (before any response is
            # committed) so streaming requests can still 429 cleanly.
            slot.gate.check_capacity(slot.slot_id)
            _inject_project_context(state, req, slot.planned.ctx)
            _inject_memory_context(state, req, slot.planned.ctx)
            _inject_skill_context(state, req, slot.planned.ctx)
            _inject_identity_context(state, req)
            if req.passthrough:
                response = await _run_passthrough(state, req, slot, request, warming=warming)
            else:
                response = await _run_agentic(state, req, slot, request, warming=warming)
    except BaseException:
        state.activity.end()
        raise

    if isinstance(response, StreamingResponse):
        inner = response.body_iterator

        async def tracked() -> AsyncIterator:
            try:
                async for chunk in inner:
                    yield chunk
            finally:
                state.activity.end()

        response.body_iterator = tracked()
        return response
    state.activity.end()
    return response


# -- lazy-residency warming notice (api.md 2026-07-22c) ----------------------
def _warming_notice() -> AgentEvent:
    """The `warming_up` notice a cold start prepends to a rich stream: the loadout is
    being brought resident before the first token."""
    return ev.notice(
        "info",
        "warming_up",
        "warming up the model loadout (cold start); the first token may take a moment",
    )


# -- pass-through path -------------------------------------------------------
def _base_payload(req: ChatRequest, slot: LlamaSlot, messages: list[dict]) -> dict:
    sampling = sampling_for(slot.model)
    payload: dict = {
        "model": slot.model,
        "messages": coalesce_system_messages(messages),
        "temperature": req.temperature if req.temperature is not None else sampling.temperature,
        "top_p": req.top_p if req.top_p is not None else sampling.top_p,
        "top_k": req.top_k if req.top_k is not None else sampling.top_k,
        **thinking_payload_fields(thinking_budget(req.effort, slot.planned.ctx)),
    }
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.stop is not None:
        payload["stop"] = req.stop
    return payload


async def _run_passthrough(
    state: AppState,
    req: ChatRequest,
    slot: LlamaSlot,
    request: Request,
    *,
    warming: bool = False,
):
    payload = _base_payload(req, slot, req.messages)
    if req.tools is not None:
        payload["tools"] = req.tools
    if req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice
    if req.response_format is not None:
        payload["response_format"] = req.response_format

    if req.stream and not req.stream_events:
        # Byte-compatible OpenAI stream: proxy the slot's SSE verbatim
        # (incl. delta.tool_calls when the model calls a client tool). Client
        # disconnect cancels this generator (starlette), closing the slot stream.
        payload["stream"] = True

        async def proxy() -> AsyncIterator[bytes]:
            async with (
                slot.gate.hold(),
                slot.backend.client() as client,
                client.stream(
                    "POST", "/v1/chat/completions", json=payload, timeout=300.0
                ) as response,
            ):
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(proxy(), media_type="text/event-stream")

    payload["stream"] = False
    chat = BackendChat(slot.backend)

    async def gated_complete() -> dict:
        async with slot.gate.hold():
            return await chat.complete(payload, timeout=300.0)

    try:
        response = await await_watching_disconnect(request, gated_complete())
    except (httpx.HTTPError, ValueError) as exc:
        raise BonsaiError(500, f"backend request failed: {exc}", code="backend_error") from exc

    message = response["choices"][0]["message"]
    _record_exchange(
        state.memory,
        req.session_id,
        req.mode,
        req.messages,
        [],
        message.get("content") or "",
        req.project_id,
        req.workdir_persist,
    )
    _schedule_auto_title(state, req.session_id)
    _schedule_reflection(state, req, slot, message.get("content") or "")

    if req.stream and req.stream_events:
        return StreamingResponse(
            _rich_envelope_from_completion(response, warming=warming),
            media_type="text/event-stream",
        )

    response["bonsai"] = _bonsai_ext(req, slot)
    return JSONResponse(response)


def _rich_envelope_from_completion(response: dict, *, warming: bool = False) -> AsyncIterator[str]:
    """stream_events envelope synthesized from a completed (non-streaming) OpenAI
    response — pass-through and external sessions share it: deltas, the CLIENT's
    tool calls (not executed here — they belong to the caller), usage, done. A cold-start
    `warming` flag leads with the `warming_up` notice (api.md 2026-07-22c)."""

    async def rich() -> AsyncIterator[str]:
        if warming:
            yield _warming_notice().as_sse()
        message = response["choices"][0]["message"]
        if message.get("content"):
            yield ev.delta(message["content"]).as_sse()
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                arguments = {"_raw": function.get("arguments")}
            yield ev.tool_call(call.get("id", ""), function.get("name", ""), arguments).as_sse()
        usage = response.get("usage") or {}
        yield ev.usage(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("thinking_tokens", 0),
        ).as_sse()
        yield ev.done(response["choices"][0].get("finish_reason") or "stop").as_sse()
        yield "data: [DONE]\n\n"

    return rich()


# -- external sessions (api.md §1 + §11, additive 2026-07-21c) ----------------
# External providers do not tell us their context size; injections budget against
# this honest stand-in instead of a slot ctx.
EXTERNAL_CONTEXT_TOKENS = 8192
EXTERNAL_TIMEOUT_S = 300.0


def _external_payload(req: ChatRequest, model: str) -> dict:
    """The OpenAI-style body sent upstream. Effort maps to SAMPLING ONLY: a monotone
    temperature default when the request sets none (api.md §1). bonsai extension
    fields (mode/effort/session_id/...) and the fork's `chat_template_kwargs` are
    NEVER sent — the provider gets a plain OpenAI request; client tools pass
    through untouched."""
    payload: dict = {
        "model": model,
        "messages": req.messages,
        "temperature": (
            req.temperature if req.temperature is not None else EFFORT_TEMPERATURE[req.effort]
        ),
    }
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.top_k is not None:
        payload["top_k"] = req.top_k
    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens
    if req.stop is not None:
        payload["stop"] = req.stop
    if req.tools is not None:
        payload["tools"] = req.tools
    if req.tool_choice is not None:
        payload["tool_choice"] = req.tool_choice
    if req.response_format is not None:
        payload["response_format"] = req.response_format
    return payload


def _external_bonsai_ext(req: ChatRequest, provider: ProviderState) -> dict:
    # Same keys as _bonsai_ext. "slot" names what served the request — an external
    # session has no VRAM slot, so the provider name is the honest value.
    return {
        "mode": req.mode,
        "effort": req.effort,
        "slot": provider.name,
        "session_id": req.session_id,
    }


def _stream_text_from_sse(sse_text: str) -> str:
    """Assistant text accumulated from a buffered OpenAI chunk stream — for the
    session archive; the bytes themselves were already proxied verbatim."""
    parts: list[str] = []
    for line in sse_text.splitlines():
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        for choice in chunk.get("choices") or []:
            delta = (choice.get("delta") or {}).get("content")
            if isinstance(delta, str):
                parts.append(delta)
    return "".join(parts)


async def _run_external(state: AppState, req: ChatRequest, provider: ProviderState, model: str):
    """Proxy one chat exchange to an external provider (mode "chat" only, enforced
    at request parse). Memory/project injection already happened; archiving and
    auto-titling (on OUR utility slot) work as for any session. NO agent loop, NO
    VRAM scheduling, NO thinking control, NO post-task reflection — reflection is an
    orchestrator capability and external models are not the orchestrator."""
    payload = _external_payload(req, model)
    url = f"{provider.base_url}/v1/chat/completions"
    headers = provider.auth_headers()

    def record(assistant_text: str) -> None:
        _record_exchange(
            state.memory,
            req.session_id,
            req.mode,
            req.messages,
            [],
            assistant_text,
            req.project_id,
            None,
        )
        _schedule_auto_title(state, req.session_id)

    if req.stream and not req.stream_events:
        # Byte-compatible OpenAI stream: proxy the provider's SSE verbatim, buffer a
        # copy so the exchange still lands in the session archive afterwards.
        payload["stream"] = True

        async def proxy() -> AsyncIterator[bytes]:
            buffer = bytearray()
            try:
                async with (
                    state.providers.client() as client,
                    client.stream(
                        "POST", url, json=payload, headers=headers, timeout=EXTERNAL_TIMEOUT_S
                    ) as response,
                ):
                    async for chunk in response.aiter_bytes():
                        buffer.extend(chunk)
                        yield chunk
            finally:
                record(_stream_text_from_sse(buffer.decode("utf-8", errors="replace")))

        return StreamingResponse(proxy(), media_type="text/event-stream")

    payload["stream"] = False
    try:
        async with state.providers.client() as client:
            response = await client.post(
                url, json=payload, headers=headers, timeout=EXTERNAL_TIMEOUT_S
            )
        response.raise_for_status()
        body = response.json()
        message = body["choices"][0]["message"]
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise BonsaiError(
            500,
            f"external provider {provider.name!r} request failed: {exc}",
            code="provider_error",
        ) from exc

    record(message.get("content") or "")

    if req.stream and req.stream_events:
        # Rich envelope over the provider's completed answer — same synthesis as a
        # pass-through call (client tool calls stream as tool_call events).
        return StreamingResponse(
            _rich_envelope_from_completion(body), media_type="text/event-stream"
        )

    body["bonsai"] = _external_bonsai_ext(req, provider)
    return JSONResponse(body)


# -- agentic path ------------------------------------------------------------
async def _prepare_loop(
    state: AppState, req: ChatRequest, slot: LlamaSlot
) -> tuple[AgentLoop, list[AgentEvent]]:
    """Build the loop (system prompt, compression, overflow guard, registry).
    Returns the loop plus the preparation events (compression / context_trimmed
    notice) to emit first on rich streams."""
    session_id = req.session_id or f"anon-{uuid.uuid4().hex[:12]}"
    workdir = _session_workdir(state, req, session_id)

    summarize = _make_summarizer(state)
    messages = coalesce_system_messages(
        [{"role": "system", "content": system_prompt(req.mode)}, *req.messages]
    )

    prep_events: list[AgentEvent] = []
    # Auto-compression (api.md 2026-07-22b): on by default, gated by chat.auto_compress.
    # Fires at ~70% of the slot ctx (memory/compression.py); the compression event
    # surfaces on rich streams so the client can show "older turns condensed".
    if summarize is not None and state.config.settings.chat.auto_compress:
        try:
            result = await comp.compress(messages, slot.planned.ctx, summarize)
        except (httpx.HTTPError, ValueError, KeyError):
            result = None  # compression is an optimization — never fail the request
        if result is not None:
            messages = result.messages
            prep_events.append(ev.compression(result.trigger_pct, result.messages_summarized))

    # Overflow guard (docs/memory.md §5): if the estimate still exceeds ~90% of the
    # slot ctx AFTER compression (or compression was unavailable), trim injected
    # blocks first, then the oldest unprotected tail — and SAY so. llama-server is
    # never handed an over-ctx request silently (non-stream clients get the log line;
    # stream_events clients get the notice).
    messages, trim_report = inj.enforce_context_budget(messages, slot.planned.ctx)
    if trim_report is not None:
        logger.warning("context trimmed for session %s: %s", session_id, trim_report.describe())
        prep_events.append(ev.notice("warn", "context_trimmed", trim_report.describe()))

    registry = build_registry(
        req.mode,
        slot.role,
        memory=state.memory,
        browse_t2_available=state.loadout.capabilities(state.config.settings)["browse_t2"],
        summarize=summarize,
        # MCP tools (api.md 2026-07-21d): connected servers' namespaced tools join
        # chat/code runs (the only modes prepared here). Fetched per run, so a
        # crashed server's tools disappear from the next run automatically.
        extra_tools=state.mcp.tools() if state.mcp is not None else None,
    )
    ctx = ToolContext(
        session_id=session_id,
        workdir=workdir,
        role=slot.role,
        mode=req.mode,
        auto_confirm=req.auto_confirm,
    )
    loop = AgentLoop(
        BackendChat(slot.backend),
        model=slot.model,
        registry=registry,
        ctx=ctx,
        messages=messages,
        sampling=sampling_for(slot.model),
        thinking_budget_tokens=thinking_budget(req.effort, slot.planned.ctx),
        max_iterations=max_tool_iterations(req.effort),
        max_tokens=req.max_tokens,
        stop=req.stop,
        tool_result_max_chars=tool_result_cap(slot.planned.ctx),
    )
    return loop, prep_events


def _prepare_ultra(state: AppState, req: ChatRequest, slot: LlamaSlot) -> UltraRun:
    """Ultra mode: plan (grammar-constrained) -> contained sub-agents on worker
    slots (sequential on the orchestrator when none are ready) -> synthesis.
    Sub-agents get fresh contexts and the focused WORKER_TOOLSET — never memory or
    skill writes (modes/ultra.py). Dispatch is coordinated with SLAP; the validated
    transcript is persisted to the in-process trace store for /v1/slap/trace."""
    # Local import keeps the SLAP wiring inside this function (single-owner boundary).
    from suiban.slap import trace_store

    session_id = req.session_id or f"anon-{uuid.uuid4().hex[:12]}"
    workdir = state.config.home / "work" / safe_workdir_name(session_id)
    workdir.mkdir(parents=True, exist_ok=True)
    summarize = _make_summarizer(state)

    def _dispatch_target(managed: LlamaSlot) -> UltraWorker:
        # Capability facts advertised over SLAP (model/family/quant/ctx/backend/workload).
        return UltraWorker(
            managed.slot_id,
            managed.model,
            managed.planned.ctx,
            BackendChat(managed.backend),
            family="bonsai",
            quant=managed.planned.quant,
            backend=state.compute_backend,
            workload=float(managed.gate.queue_depth),
        )

    # Worker slots are NOT gated inside the Ultra dispatch: the run effectively owns
    # them, and a direct chat routed to a worker model would interleave at
    # llama-server's own request queue (serialized there, just without our 429
    # bound). TODO(v1.1): hold worker gates per sub-task so direct worker chats
    # queue behind Ultra explicitly.
    workers = [
        _dispatch_target(s)
        for s in state.manager.slots
        if s.role == "worker" and s.state == "ready"
    ]
    return UltraRun(
        orchestrator=_dispatch_target(slot),
        workers=workers,
        registry_factory=lambda: build_worker_registry(memory=state.memory, summarize=summarize),
        # Workers inherit the request's auto_confirm: with the bypass on they may mutate
        # files autonomously; with it off, worker fs_write/fs_undo stay confirm-gated and
        # simply fail (no interactive confirmer inside a contained sub-agent) — which
        # matches SLAP's rule that workers propose artifacts rather than mutate canonical
        # state. Human authority is exercised once, at the Ultra request.
        tool_ctx_factory=lambda: ToolContext(
            session_id=session_id,
            workdir=workdir,
            role="worker",
            mode="ultra",
            auto_confirm=req.auto_confirm,
        ),
        messages=req.messages,
        effort=req.effort,
        max_tokens=req.max_tokens,
        session_id=session_id,
        workdir=workdir,
        trace_sink=trace_store().record,
        # SLAP toggle (api.md 2026-07-22c): when off, dispatch runs the plain
        # structured-dict path — no SLAP messages built or recorded, trace stays empty.
        slap_enabled=state.config.settings.slap.enabled,
    )


async def _run_agentic(
    state: AppState,
    req: ChatRequest,
    slot: LlamaSlot,
    request: Request,
    *,
    warming: bool = False,
):
    loop: AgentLoop | UltraRun
    if req.mode == "ultra":
        loop = _prepare_ultra(state, req, slot)
        prep_events: list[AgentEvent] = []
    else:
        loop, prep_events = await _prepare_loop(state, req, slot)
    # A cold start leads the rich stream with the warming_up notice (api.md 2026-07-22c).
    if warming:
        prep_events = [_warming_notice(), *prep_events]

    def record() -> None:
        _record_exchange(
            state.memory,
            req.session_id,
            req.mode,
            req.messages,
            loop.tool_messages,
            loop.final_text,
            req.project_id,
            req.workdir_persist,
        )
        _mark_skills_verified(state, req, loop.finish_reason)
        _schedule_auto_title(state, req.session_id)
        _schedule_reflection(state, req, slot, loop.final_text)

    if not req.stream:

        async def consume() -> None:
            # The whole agentic run holds the slot gate (per-slot serialization);
            # the disconnect watcher cancels it if the client goes away.
            async with slot.gate.hold():
                async for _event in loop.run():
                    pass  # events are for streams; non-streaming wants the aggregate

        await await_watching_disconnect(request, consume())
        if loop.finish_reason == "error":
            raise BonsaiError(500, loop.error_message or "agent run failed", code="backend_error")
        record()
        return JSONResponse(
            {
                "id": _completion_id(),
                "object": "chat.completion",
                "created": 0,
                "model": slot.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": loop.final_text},
                        "finish_reason": loop.finish_reason,
                    }
                ],
                "usage": _usage_payload(loop.total_usage),
                "bonsai": _bonsai_ext(req, slot),
            }
        )

    if req.stream_events:

        async def rich() -> AsyncIterator[str]:
            for prep_event in prep_events:
                yield prep_event.as_sse()
            # Slot gate with a visible queue: a real wait (> QUEUE_NOTICE_AFTER_S)
            # emits ONE notice so the client can say "queued behind N" instead of
            # showing a silent stall. wait_for's cancel forfeits our queue spot,
            # which is fine: the queue is bounded (capacity-checked at the endpoint)
            # and every waiter is an equivalent chat.
            ahead = slot.gate.queue_depth
            try:
                await asyncio.wait_for(slot.gate.acquire(), timeout=QUEUE_NOTICE_AFTER_S)
            except TimeoutError:
                yield ev.notice(
                    "info",
                    "slot_queued",
                    f"queued behind {ahead} run(s) on slot {slot.slot_id}",
                ).as_sse()
                await slot.gate.acquire()
            try:
                async for event in loop.run():
                    yield event.as_sse()
                record()
                yield "data: [DONE]\n\n"
            finally:
                slot.gate.release()

        return StreamingResponse(rich(), media_type="text/event-stream")

    # Default stream envelope: byte-compatible OpenAI chunks synthesized from the
    # loop's text deltas. Internal tool activity is NOT surfaced as delta.tool_calls —
    # those chunks mean "the CLIENT should run this tool" in OpenAI semantics, and
    # these tools are ours; clients wanting tool visibility use stream_events.
    completion_id = _completion_id()

    async def openai_stream() -> AsyncIterator[str]:
        yield _openai_chunk(completion_id, slot.model, {"role": "assistant", "content": ""})
        # Same per-slot serialization as every local run; no queue notice here —
        # OpenAI chunks have no envelope for one (stream_events does).
        async with slot.gate.hold():
            async for event in loop.run():
                if event.type == "delta":
                    yield _openai_chunk(
                        completion_id, slot.model, {"content": event.payload["text"]}
                    )
        finish = loop.finish_reason if loop.finish_reason != "error" else "stop"
        final = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": slot.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
            "usage": _usage_payload(loop.total_usage),
        }
        yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
        record()
        yield "data: [DONE]\n\n"

    return StreamingResponse(openai_stream(), media_type="text/event-stream")


# -- internal pipeline entry (scheduled runs) --------------------------------
async def run_internal_chat(
    state: AppState,
    *,
    prompt: str,
    mode: str,
    effort: str,
    session_id: str,
    project_id: str | None = None,
) -> tuple[str, str | None]:
    """Run one prompt through the SAME pipeline as an HTTP chat request — routing,
    project validation/injection, agent loop, archive, auto-titling, activity
    tracking — and return (final text, error or None). Scheduled runs use this so a
    scheduled session is indistinguishable from a normal one (api.md §10)."""
    body: dict = {
        "model": "bonsai-auto",
        "messages": [{"role": "user", "content": prompt}],
        "mode": mode,
        "effort": effort,
        "session_id": session_id,
    }
    if project_id is not None:
        body["project_id"] = project_id
    req = ChatRequest(body)
    state.activity.begin()
    try:
        _require_project(state, req.project_id)
        _validate_request_workdir(state, req)
        # Lazy residency (api.md 2026-07-22c): a scheduled run warms the loadout on
        # demand, exactly like an interactive request.
        await state.load.ensure_loaded()
        slot = _route_slot(state, req)
        _inject_project_context(state, req, slot.planned.ctx)
        _inject_memory_context(state, req, slot.planned.ctx)
        _inject_skill_context(state, req, slot.planned.ctx)
        _inject_identity_context(state, req)  # client "other" -> base identity only
        loop, _prep_events = await _prepare_loop(state, req, slot)
        # Scheduled runs are background work: they hold the slot gate like any run
        # but skip the capacity 429 — waiting longer beats failing a cron job.
        async with slot.gate.hold():
            async for _event in loop.run():
                pass
        _record_exchange(
            state.memory,
            req.session_id,
            req.mode,
            req.messages,
            loop.tool_messages,
            loop.final_text,
            req.project_id,
            req.workdir_persist,
        )
        _mark_skills_verified(state, req, loop.finish_reason)
        _schedule_auto_title(state, req.session_id)
        _schedule_reflection(state, req, slot, loop.final_text)
        if loop.finish_reason == "error":
            return loop.final_text, loop.error_message or "agent run failed"
        return loop.final_text, None
    finally:
        state.activity.end()
