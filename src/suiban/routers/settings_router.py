"""/v1/settings — GET current+staged, PATCH stages only (apply commits)."""

from __future__ import annotations

from fastapi import APIRouter, Request

from suiban.app import state_of
from suiban.errors import BonsaiError

router = APIRouter()


@router.get("/v1/settings")
async def get_settings(request: Request) -> dict:
    state = state_of(request)
    staged = state.config.staged_settings()
    return {
        "current": state.config.settings.public_dict(),
        "staged": staged.public_dict() if staged is not None else None,
    }


@router.patch("/v1/settings")
async def patch_settings(request: Request) -> dict:
    state = state_of(request)
    try:
        body = await request.json()
    except ValueError as exc:
        raise BonsaiError(400, "request body must be JSON", code="invalid_json") from exc
    if not isinstance(body, dict):
        raise BonsaiError(400, "PATCH body must be a JSON object", code="invalid_json")
    state.config.stage(body)
    staged = state.config.staged_settings()
    assert staged is not None
    return {
        "current": state.config.settings.public_dict(),
        "staged": staged.public_dict(),
    }
