"""/v1/skills — agentskills.io-compatible skills (docs/api.md §6).

HTTP PUT is a human action: source becomes "human", version increments. Model-driven
creation/improvement is NOT on this surface — it happens only inside the 27B's
post-task reflection (server-enforced in the tool registry).
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from suiban.app import state_of
from suiban.errors import BonsaiError
from suiban.memory.skill_import import SOURCES, SkillImportError, import_skills
from suiban.memory.skills import SKILL_NAME_RE

router = APIRouter()


class _PutBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str | None = None
    content: str = Field(min_length=1)


@router.get("/v1/skills")
async def skills_list(request: Request) -> dict:
    memory = state_of(request).memory
    return {"skills": [s.as_dict(with_content=False) for s in memory.skills.list()]}


@router.post("/v1/skills/import")
async def skills_import(request: Request) -> dict:
    """Import agentskills.io SKILL.md skills from another ecosystem (api.md 2026-07-22c).
    `{ "source": "openclaw"|"hermes"|"path", "path"? }` → `{ "imported": [ { "name" } ],
    "skipped": [ { "name", "reason" } ] }`. A malformed skill is skipped with a reason;
    an unscannable source (a `path` that does not exist) is a clean 400, never a crash."""
    memory = state_of(request).memory
    try:
        raw = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(raw, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")
    source = raw.get("source")
    if source not in SOURCES:
        raise BonsaiError(
            400,
            f"'source' must be one of {', '.join(SOURCES)}; got {source!r}",
            code="validation_error",
        )
    path = raw.get("path")
    if path is not None and not isinstance(path, str):
        raise BonsaiError(400, "'path' must be a string", code="validation_error")
    if source == "path" and not path:
        raise BonsaiError(400, "'path' is required when source is 'path'", code="validation_error")
    try:
        # user_home defaults to the real home (~/.openclaw, ~/.hermes live there); the
        # "path" source names an explicit directory and does not consult it.
        result = import_skills(memory.skills, source, path)
    except SkillImportError as exc:
        raise BonsaiError(400, str(exc), code="import_source_unavailable") from exc
    return {
        "imported": [{"name": name} for name in result.imported],
        "skipped": result.skipped,
    }


@router.get("/v1/skills/{name}")
async def skills_get(request: Request, name: str) -> dict:
    memory = state_of(request).memory
    skill = memory.skills.get(name)
    if skill is None:
        raise BonsaiError(404, f"no such skill: {name}", code="skill_not_found")
    return skill.as_dict()


@router.put("/v1/skills/{name}")
async def skills_put(request: Request, name: str) -> dict:
    memory = state_of(request).memory
    if not SKILL_NAME_RE.match(name):
        raise BonsaiError(
            400, f"skill name must be kebab-case (got {name!r})", code="skill_name_invalid"
        )
    try:
        raw = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    try:
        body = _PutBody.model_validate(raw)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise BonsaiError(
            400, f"invalid request: {loc}: {first['msg']}", code="validation_error"
        ) from exc
    skill = memory.skills.put(name, body.content, source="human", description=body.description)
    return skill.as_dict()


@router.delete("/v1/skills/{name}", status_code=204)
async def skills_delete(request: Request, name: str) -> Response:
    memory = state_of(request).memory
    if not memory.skills.delete(name):
        raise BonsaiError(404, f"no such skill: {name}", code="skill_not_found")
    return Response(status_code=204)
