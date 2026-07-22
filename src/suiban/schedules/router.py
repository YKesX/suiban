"""/v1/schedules — scheduled-run CRUD + run-now (docs/api.md §10, additive
2026-07-21b). Validation is manual (ChatRequest-style): the cadence rules are
cross-field and deserve exact contract-envelope 400s."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Request, Response

from suiban.app import AppState, state_of
from suiban.errors import BonsaiError
from suiban.modes.registry import MODES
from suiban.schedules.runner import Scheduler
from suiban.schedules.store import (
    Cadence,
    Schedule,
    compute_next_run,
    iso_utc,
    new_schedule_id,
    now_iso,
    validate_cadence,
)

router = APIRouter()

SCHEDULE_MODES = ("chat", "code")
EFFORTS = ("low", "mid", "high", "xhigh", "max")

_FIELDS = ("name", "prompt", "mode", "effort", "project_id", "cadence", "enabled")


def _bad(message: str, code: str = "validation_error") -> BonsaiError:
    return BonsaiError(400, message, code=code)


async def _json_body(request: Request) -> dict:
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")
    return body


def _scheduler(request: Request) -> Scheduler:
    scheduler = state_of(request).scheduler
    assert scheduler is not None  # wired in the app lifespan
    return scheduler


def _get_or_404(scheduler: Scheduler, schedule_id: str) -> Schedule:
    schedule = scheduler.store.get(schedule_id)
    if schedule is None:
        raise BonsaiError(404, f"no such schedule: {schedule_id}", code="schedule_not_found")
    return schedule


def _require_project(state: AppState, project_id: str | None) -> None:
    if project_id is None:
        return
    if state.memory.store.get_project(project_id) is None:
        raise BonsaiError(404, f"no such project: {project_id}", code="project_not_found")


def _validate_fields(body: dict) -> dict:
    """Shared field validation for create + patch. Returns only the fields present."""
    unknown = set(body) - set(_FIELDS)
    if unknown:
        raise _bad(f"unknown fields: {', '.join(sorted(unknown))}")
    out: dict = {}
    for key in ("name", "prompt"):
        if key in body:
            value = body[key]
            if not isinstance(value, str) or not value.strip():
                raise _bad(f"'{key}' must be a non-empty string")
            out[key] = value.strip()
    if "mode" in body:
        if body["mode"] not in SCHEDULE_MODES:
            raise _bad(f"mode must be one of {', '.join(SCHEDULE_MODES)}; got {body['mode']!r}")
        out["mode"] = body["mode"]
    if "effort" in body:
        if body["effort"] not in EFFORTS:
            raise _bad(f"effort must be one of {', '.join(EFFORTS)}; got {body['effort']!r}")
        out["effort"] = body["effort"]
    if "project_id" in body:
        value = body["project_id"]
        if value is not None and (not isinstance(value, str) or not value):
            raise _bad("'project_id' must be a non-empty string or null")
        out["project_id"] = value
    if "cadence" in body:
        out["cadence"] = validate_cadence(body["cadence"])
    if "enabled" in body:
        if not isinstance(body["enabled"], bool):
            raise _bad("'enabled' must be a boolean")
        out["enabled"] = body["enabled"]
    return out


@router.get("/v1/schedules")
async def list_schedules(request: Request) -> dict:
    return {"schedules": [s.as_dict() for s in _scheduler(request).store.list()]}


@router.post("/v1/schedules", status_code=201)
async def create_schedule(request: Request) -> dict:
    state = state_of(request)
    scheduler = _scheduler(request)
    fields = _validate_fields(await _json_body(request))
    for required in ("name", "prompt", "cadence"):
        if required not in fields:
            raise _bad(f"'{required}' is required")
    mode = fields.get("mode", "chat")
    cadence: Cadence = fields["cadence"]
    _require_project(state, fields.get("project_id"))
    schedule = Schedule(
        id=new_schedule_id(),
        name=fields["name"],
        prompt=fields["prompt"],
        mode=mode,
        effort=fields.get("effort", MODES[mode].default_effort),
        project_id=fields.get("project_id"),
        cadence=cadence,
        enabled=fields.get("enabled", True),
        created_at=now_iso(),
        last_run_at=None,
        next_run_at=iso_utc(compute_next_run(cadence, datetime.now().astimezone())),
        last_session_id=None,
        last_error=None,
    )
    scheduler.store.add(schedule)
    return schedule.as_dict()


@router.get("/v1/schedules/{schedule_id}")
async def get_schedule(request: Request, schedule_id: str) -> dict:
    return _get_or_404(_scheduler(request), schedule_id).as_dict()


@router.patch("/v1/schedules/{schedule_id}")
async def patch_schedule(request: Request, schedule_id: str) -> dict:
    state = state_of(request)
    scheduler = _scheduler(request)
    current = _get_or_404(scheduler, schedule_id)
    fields = _validate_fields(await _json_body(request))
    if "project_id" in fields:
        _require_project(state, fields["project_id"])
    if "cadence" in fields or fields.get("enabled") is True:
        # A new cadence — or a re-enable that could otherwise fire on a stale
        # next_run_at — recomputes the next run from now.
        cadence = fields.get("cadence", current.cadence)
        fields["next_run_at"] = iso_utc(compute_next_run(cadence, datetime.now().astimezone()))
    updated = scheduler.store.update(schedule_id, **fields)
    assert updated is not None
    return updated.as_dict()


@router.delete("/v1/schedules/{schedule_id}", status_code=204)
async def delete_schedule(request: Request, schedule_id: str) -> Response:
    scheduler = _scheduler(request)
    _get_or_404(scheduler, schedule_id)
    scheduler.store.delete(schedule_id)
    return Response(status_code=204)


@router.post("/v1/schedules/{schedule_id}/run", status_code=202)
async def run_schedule_now(request: Request, schedule_id: str) -> dict:
    return {"session_id": _scheduler(request).run_now(schedule_id)}
