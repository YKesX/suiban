"""/v1/modes — mode metadata (never the prompt text), per docs/api.md §7."""

from __future__ import annotations

from fastapi import APIRouter

from suiban.modes.registry import MODES, get_mode

router = APIRouter()


@router.get("/v1/modes")
async def list_modes() -> dict:
    return {"modes": [mode.as_dict() for mode in MODES.values()]}


@router.get("/v1/modes/{name}")
async def get_mode_route(name: str) -> dict:
    return get_mode(name).as_dict()
