"""/v1/slap — SLAP agent-protocol observability (api.md §12, additive 2026-07-22b).

Read-only. Ultra coordinates its sub-agents with SLAP internally; these endpoints expose
the protocol version/operations, the vendored per-operation JSON schemas, and the
validated agent-to-agent transcript of a completed Ultra run (coarse — no worker
internals, and the volatile per-agent system prompts are never recorded).
"""

from __future__ import annotations

from fastapi import APIRouter

from suiban import slap
from suiban.errors import BonsaiError

router = APIRouter()


@router.get("/v1/slap")
async def slap_info() -> dict:
    return {
        "version": slap.VERSION,
        "profiles": list(slap.PROFILES),
        "operations": list(slap.OPERATIONS),
    }


@router.get("/v1/slap/schema/{operation}")
async def slap_schema(operation: str) -> dict:
    # Only the nine operations are addressable here; the shared `envelope` base schema
    # (which load_schema also accepts) is not one of them → 404, per api.md §12.
    if operation not in slap.OPERATIONS:
        raise BonsaiError(
            404, f"no such SLAP operation: {operation!r}", code="slap_operation_not_found"
        )
    return slap.load_schema(operation)


@router.get("/v1/slap/trace/{session_id}")
async def slap_trace(session_id: str) -> dict:
    return {"messages": slap.trace_store().get(session_id)}
