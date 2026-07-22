"""GET /v1/mcp/connectors — the built-in MCP connector catalog (api.md 2026-07-22c).

Returns every curated connector with its `enabled` flag reflecting the current
`mcp_connectors` settings. Enabling one is a `PATCH /v1/settings { "mcp_connectors":
[...] }` + `POST /v1/system/apply` — on apply the connector is wired into the same
McpManager as a custom stdio server (mcp/catalog.py, mcp/manager.py). This is a
read-only, non-inference route: it never warms the lazy loadout.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from suiban.app import state_of
from suiban.mcp.catalog import catalog_view

router = APIRouter()


@router.get("/v1/mcp/connectors")
async def list_connectors(request: Request) -> dict:
    state = state_of(request)
    return {"connectors": catalog_view(state.config.settings.mcp_connectors)}
