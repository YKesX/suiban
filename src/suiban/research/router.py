"""/v1/jobs — deep research, exactly per api.md §3.

Everything user-visible here is coarse: state, stage, percent, and (once completed)
the report markdown. No queries, no URLs, no drafts — that is a product rule.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

from suiban.app import state_of
from suiban.effort import EFFORT_LEVELS
from suiban.errors import BonsaiError
from suiban.modes.registry import MODES
from suiban.research.jobs import Job, JobManager

router = APIRouter()


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _manager(request: Request) -> JobManager:
    return state_of(request).jobs


def _get_job_or_404(manager: JobManager, job_id: str) -> Job:
    job = manager.get(job_id)
    if job is None:
        raise BonsaiError(404, f"no such job: {job_id}", code="job_not_found")
    return job


@router.post("/v1/jobs", status_code=202)
async def create_job(request: Request) -> dict:
    manager = _manager(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "request body must be a JSON object", code="invalid_json")

    job_type = body.get("type")
    if job_type != "deep_research":
        raise BonsaiError(
            400,
            f"unknown job type {job_type!r}; the only v1 job type is 'deep_research'",
            code="job_type_unknown",
        )
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        raise BonsaiError(400, "'query' must be a non-empty string", code="validation_error")
    effort = body.get("effort")
    if effort is not None and effort not in EFFORT_LEVELS:
        raise BonsaiError(
            400,
            f"effort must be one of {', '.join(EFFORT_LEVELS)}; got {effort!r}",
            code="validation_error",
        )

    # Same default chain as chat requests: body effort > settings.effort_default >
    # the mode's default (effort_default wired, refinement pass).
    resolved = (
        effort
        or state_of(request).config.settings.effort_default
        or MODES["deep_research"].default_effort
    )
    job = manager.submit(query.strip(), resolved)
    return {"id": job.id, "state": job.state}


@router.get("/v1/jobs")
async def list_jobs(request: Request) -> dict:
    return {"jobs": [job.status_dict() for job in _manager(request).list()]}


@router.get("/v1/jobs/{job_id}")
async def get_job(request: Request, job_id: str) -> dict:
    return _get_job_or_404(_manager(request), job_id).status_dict()


@router.get("/v1/jobs/{job_id}/events")
async def job_events(request: Request, job_id: str) -> StreamingResponse:
    manager = _manager(request)
    job = _get_job_or_404(manager, job_id)

    async def stream() -> AsyncIterator[str]:
        # Snapshot first so a late subscriber syncs, then live changes; the stream
        # closes after a terminal state (api.md).
        current = manager.get(job_id) or job
        if current.stage is not None and not current.terminal:
            yield _sse({"type": "progress", "stage": current.stage, "percent": current.percent})
        yield _sse({"type": "state", "state": current.state})
        if current.terminal:
            return
        queue = manager.subscribe(job_id)
        try:
            while True:
                payload = await queue.get()
                yield _sse(payload)
                if payload.get("type") == "state" and payload.get("state") in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    return
        finally:
            manager.unsubscribe(job_id, queue)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/v1/jobs/{job_id}/report")
async def job_report(request: Request, job_id: str) -> Response:
    manager = _manager(request)
    job = _get_job_or_404(manager, job_id)
    path = manager.report_path(job_id)
    if job.state != "completed" or not path.is_file():
        raise BonsaiError(
            404,
            f"job {job_id} has no report (state: {job.state}); reports exist once "
            "the job is completed",
            code="report_not_ready",
        )
    return Response(path.read_text(encoding="utf-8"), media_type="text/markdown")


@router.delete("/v1/jobs/{job_id}")
async def cancel_job(request: Request, job_id: str) -> dict:
    # Awaited: the response means the pipeline task actually unwound (bounded) and
    # the in-flight llama-server request was aborted — see JobManager.cancel.
    job = await _manager(request).cancel(job_id)
    return {"id": job.id, "state": job.state}
