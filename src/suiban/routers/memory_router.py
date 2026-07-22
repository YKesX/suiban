"""/v1/memory — entries, search, state files, sessions (docs/api.md §5).

Fixed paths (search/state/sessions) are declared before /v1/memory/{entry_id} so they
match first. HTTP writes are human/client actions. State files — identity.md included
— are edited over HTTP via PUT /v1/memory/state/{name} (additive 2026-07-21b); the
identity ENTRY surface stays read-only so there is exactly one edit path.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from suiban.app import state_of
from suiban.errors import BonsaiError

router = APIRouter()

_LAYERS = ("identity", "state", "archive")


class _CreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: str
    title: str = Field(min_length=1)
    content: str
    tags: list[str] | None = None


class _UpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1)
    content: str | None = None
    tags: list[str] | None = None


class _StateUpdateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: str


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")
    return body


def _validate(model: type[BaseModel], body: dict) -> BaseModel:
    try:
        return model.model_validate(body)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise BonsaiError(
            400, f"invalid request: {loc}: {first['msg']}", code="validation_error"
        ) from exc


@router.get("/v1/memory/search")
async def memory_search(request: Request, q: str = "", limit: int = 12) -> dict:
    memory = state_of(request).memory
    if not q:
        return {"results": []}
    return {"results": memory.store.search(q, limit=max(1, min(limit, 50)))}


@router.get("/v1/memory/state")
async def memory_state(request: Request) -> dict:
    memory = state_of(request).memory
    return {"files": [f.as_dict() for f in memory.files.all_files()]}


@router.put("/v1/memory/state/{name}")
async def memory_state_update(request: Request, name: str) -> dict:
    """Overwrite an EXISTING bounded state file (identity.md included). 404 for names
    outside the known set (nothing new is creatable here), 400 `state_file_too_large`
    above max_bytes (additive 2026-07-21b)."""
    memory = state_of(request).memory
    body = _validate(_StateUpdateBody, await _json_body(request))
    return memory.update_state_file(name, body.content).as_dict()


@router.delete("/v1/memory/state/{name}", status_code=204)
async def memory_state_delete(request: Request, name: str) -> Response:
    """Delete a bounded state file (additive 2026-07-22d). 404 for unknown names, 400
    `identity_read_only` for identity.md and the client overlays (never deletable)."""
    memory = state_of(request).memory
    memory.delete_state_file(name)
    return Response(status_code=204)


@router.get("/v1/memory/sessions")
async def memory_sessions(
    request: Request,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
    project_id: str | None = None,
    mode: str | None = None,
) -> dict:
    memory = state_of(request).memory
    sessions = memory.store.list_sessions(
        q,
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
        project_id=project_id,
        mode=mode,
    )
    return {"sessions": sessions}


@router.get("/v1/memory/sessions/{session_id}")
async def memory_session_transcript(request: Request, session_id: str) -> dict:
    memory = state_of(request).memory
    transcript = memory.store.session_transcript(session_id)
    if transcript is None:
        raise BonsaiError(404, f"no such session: {session_id}", code="session_not_found")
    return transcript


@router.delete("/v1/memory/sessions/{session_id}", status_code=204)
async def memory_session_delete(request: Request, session_id: str) -> Response:
    """Delete an archived session/chat and its transcript (additive 2026-07-22d).
    404 `session_not_found` for an unknown id."""
    memory = state_of(request).memory
    memory.delete_session(session_id)
    return Response(status_code=204)


@router.get("/v1/memory")
async def memory_list(
    request: Request, layer: str | None = None, limit: int = 50, offset: int = 0
) -> dict:
    memory = state_of(request).memory
    if layer is not None and layer not in _LAYERS:
        raise BonsaiError(
            400, f"layer must be one of {', '.join(_LAYERS)}; got {layer!r}", code="layer_unknown"
        )
    entries, total = memory.store.list_entries(
        layer, limit=max(1, min(limit, 200)), offset=max(0, offset)
    )
    return {"entries": [e.as_dict() for e in entries], "total": total}


@router.post("/v1/memory", status_code=201)
async def memory_create(request: Request) -> dict:
    memory = state_of(request).memory
    body = _validate(_CreateBody, await _json_body(request))
    entry = memory.create_entry(body.layer, body.title, body.content, body.tags)
    return entry.as_dict()


@router.put("/v1/memory/{entry_id}")
async def memory_update(request: Request, entry_id: str) -> dict:
    memory = state_of(request).memory
    body = _validate(_UpdateBody, await _json_body(request))
    entry = memory.update_entry(entry_id, title=body.title, content=body.content, tags=body.tags)
    return entry.as_dict()


@router.delete("/v1/memory/{entry_id}", status_code=204)
async def memory_delete(request: Request, entry_id: str) -> Response:
    memory = state_of(request).memory
    memory.delete_entry(entry_id)
    return Response(status_code=204)
