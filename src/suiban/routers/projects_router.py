"""/v1/projects — project CRUD + plain-text knowledge docs (docs/api.md §9, additive
2026-07-21b).

Projects group sessions and carry docs searched via FTS5 (never embeddings); chat
requests carrying `project_id` get matching excerpts injected (routers/chat.py) and
their sessions list under the project. Deleting a project keeps member sessions alive
with `project_id` cleared.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from suiban.app import state_of
from suiban.errors import BonsaiError
from suiban.memory.store import MemoryStore

router = APIRouter()


class _CreateProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str = ""


class _PatchProject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1)
    description: str | None = None


class _CreateDoc(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
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


def _store(request: Request) -> MemoryStore:
    return state_of(request).memory.store


def _get_project_or_404(store: MemoryStore, project_id: str) -> dict:
    project = store.get_project(project_id)
    if project is None:
        raise BonsaiError(404, f"no such project: {project_id}", code="project_not_found")
    return project


@router.get("/v1/projects")
async def list_projects(request: Request) -> dict:
    return {"projects": _store(request).list_projects()}


@router.post("/v1/projects", status_code=201)
async def create_project(request: Request) -> dict:
    store = _store(request)
    body = _validate(_CreateProject, await _json_body(request))
    return store.add_project(body.name, body.description)


@router.get("/v1/projects/{project_id}")
async def get_project(request: Request, project_id: str) -> dict:
    return _get_project_or_404(_store(request), project_id)


@router.patch("/v1/projects/{project_id}")
async def patch_project(request: Request, project_id: str) -> dict:
    store = _store(request)
    _get_project_or_404(store, project_id)
    body = _validate(_PatchProject, await _json_body(request))
    updated = store.update_project(project_id, name=body.name, description=body.description)
    assert updated is not None
    return updated


@router.delete("/v1/projects/{project_id}", status_code=204)
async def delete_project(request: Request, project_id: str) -> Response:
    store = _store(request)
    _get_project_or_404(store, project_id)
    store.delete_project(project_id)
    return Response(status_code=204)


@router.get("/v1/projects/{project_id}/docs")
async def list_docs(request: Request, project_id: str) -> dict:
    store = _store(request)
    _get_project_or_404(store, project_id)
    return {"docs": store.list_project_docs(project_id)}


@router.post("/v1/projects/{project_id}/docs", status_code=201)
async def create_doc(request: Request, project_id: str) -> dict:
    store = _store(request)
    _get_project_or_404(store, project_id)
    body = _validate(_CreateDoc, await _json_body(request))
    return store.add_project_doc(project_id, body.title, body.content)


@router.get("/v1/projects/{project_id}/docs/{doc_id}")
async def get_doc(request: Request, project_id: str, doc_id: str) -> dict:
    store = _store(request)
    _get_project_or_404(store, project_id)
    doc = store.get_project_doc(project_id, doc_id)
    if doc is None:
        raise BonsaiError(404, f"no such doc: {doc_id}", code="project_doc_not_found")
    return doc


@router.delete("/v1/projects/{project_id}/docs/{doc_id}", status_code=204)
async def delete_doc(request: Request, project_id: str, doc_id: str) -> Response:
    store = _store(request)
    _get_project_or_404(store, project_id)
    if not store.delete_project_doc(project_id, doc_id):
        raise BonsaiError(404, f"no such doc: {doc_id}", code="project_doc_not_found")
    return Response(status_code=204)
