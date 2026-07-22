"""POST /v1/memory/sessions/import — import another tool's chats (api.md §5,
additive 2026-07-22b).

Parses an OpenAI / claude.ai / claude-code / generic export into archived sessions (they
then appear under GET /v1/memory/sessions and restore like any session). With
`compress:true` the resident utility model condenses each long import into a single seed
summary message, so a resumed session starts inside context. A payload that does not
match the provider shape is a 400 `import_unrecognized`; parsing lives in
memory/importers.py (pure, no network — the utility model is only touched for compress).
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Request

from suiban.app import state_of
from suiban.errors import BonsaiError
from suiban.memory import compression as comp
from suiban.memory import importers
from suiban.memory.ids import ulid
from suiban.routers.chat import _make_summarizer

router = APIRouter()

_MODES = ("chat", "code")


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")
    return body


@router.post("/v1/memory/sessions/import")
async def import_sessions(request: Request) -> dict:
    state = state_of(request)
    body = await _json_body(request)

    provider = body.get("provider")
    if provider not in importers.PROVIDERS:
        raise BonsaiError(
            400,
            f"'provider' must be one of {', '.join(importers.PROVIDERS)}; got {provider!r}",
            code="validation_error",
        )
    if "data" not in body:
        raise BonsaiError(
            400, "'data' is required (the provider's export payload)", code="validation_error"
        )
    mode = body.get("mode", "chat")
    if mode not in _MODES:
        raise BonsaiError(
            400, f"'mode' must be one of {', '.join(_MODES)}; got {mode!r}", code="validation_error"
        )
    compress = body.get("compress", False)
    if not isinstance(compress, bool):
        raise BonsaiError(400, "'compress' must be a boolean", code="validation_error")

    try:
        parsed_sessions = importers.parse_import(provider, body["data"])
    except importers.ImportUnrecognized as exc:
        raise BonsaiError(400, str(exc), code="import_unrecognized") from exc

    # Compression is a user-requested utility-model operation, so warm the lazily-
    # resident loadout on demand (api.md 2026-07-22c) before building the summarizer;
    # None (still no ready slot) makes compress degrade honestly to a verbatim import
    # rather than failing. A verbatim import never touches a model, so it never loads.
    if compress:
        await state.load.ensure_loaded()
    summarize = _make_summarizer(state) if compress else None
    store = state.memory.store

    imported: list[dict] = []
    for parsed in parsed_sessions:
        messages = parsed.messages
        if compress and summarize is not None and len(messages) >= 2:
            try:
                folded = await summarize(comp.wrap_fold_input(importers.transcript_text(messages)))
            except (httpx.HTTPError, ValueError, KeyError):
                folded = ""  # compression is an optimization — never fail the import
            summary = folded.strip()
            if summary:
                messages = [{"role": "system", "content": f"{comp.SUMMARY_PREFIX}\n{summary}"}]

        session_id = f"import-{ulid()}"
        store.ensure_session(session_id, mode)
        for message in messages:
            store.add_message(session_id, message["role"], message["content"])
        title = parsed.title or importers.default_title(parsed.messages)
        if title:
            store.set_session_title(session_id, title)
        imported.append({"id": session_id, "title": title, "message_count": len(messages)})

    return {"imported": imported}
