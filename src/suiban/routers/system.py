"""/v1/system — status, budget, health, apply, search_test."""

from __future__ import annotations

from fastapi import APIRouter, Request

from suiban import __version__
from suiban.app import state_of
from suiban.errors import BonsaiError
from suiban.mcp.catalog import combined_mcp_servers
from suiban.search import build_search_provider

router = APIRouter()

# search_test (api.md §11): a harmless default query when the body omits one.
DEFAULT_SEARCH_TEST_QUERY = "bonsai tree care"
SEARCH_TEST_MAX_RESULTS = 3


@router.get("/v1/system")
async def get_system(request: Request) -> dict:
    state = state_of(request)
    settings = state.config.settings
    loadout = state.loadout
    # Refresh live GPU usage; planning stays frozen (loadout fixed at run start).
    try:
        snapshot = state.telemetry_provider.snapshot()
        state.telemetry = snapshot
    except Exception:
        snapshot = state.telemetry  # stale beats crashing
    return {
        "version": __version__,
        "uptime_s": state.uptime_s,
        "gpus": [g.as_dict() for g in snapshot.gpus] if snapshot.gpus else None,
        "telemetry_source": snapshot.source,
        "loadout": {
            **loadout.as_dict(),
            "slots": [s.as_dict() for s in loadout.slots],
        },
        "capabilities": loadout.capabilities(settings),
        "kv": state.kv.as_dict(),
        "quant_family": {
            "configured": loadout.family_configured,
            "effective": loadout.family_effective,
            "degraded": loadout.family_degraded,
            "reason": loadout.family_reason,
        },
        "dspark": {
            "enabled": settings.dspark_enabled,
            "available": state.compute_backend == "cuda",
        },
        # Lazy / keep-alive residency (api.md 2026-07-22c). state reflects the
        # LoadController: cold (nothing resident) | loading | ready | idle_unloading.
        "runtime": {
            "keep_alive": settings.runtime.keep_alive,
            "models_loaded": state.load.models_loaded,
            "state": state.load.state,
        },
        "jobs_active": state.jobs.active,
        "security": {
            "auth_required": state.auth_required,
            "remote_agentic": settings.server.remote_agentic,
            "telegram_paired": bool(settings.gateways.telegram.allowed_chat_ids),
        },
        "notices": [n.as_dict() for n in state.notices()],
    }


@router.get("/v1/system/budget")
async def get_budget(request: Request) -> dict:
    state = state_of(request)
    loadout = state.loadout
    ctx_by_model = {s.model: s.ctx for s in loadout.slots}
    rows = state.budget.table_rows(
        loadout.family_effective, state.kv.k_type, state.kv.v_type, ctx_by_model
    )
    return {"measured": state.budget.has_measurements, "rows": rows}


@router.get("/v1/system/health")
async def get_health(request: Request) -> dict:
    state = state_of(request)
    manager = state.manager
    ready, total = manager.slots_ready, manager.slots_total
    binary_ok = True
    models_ok = True
    for notice in state.notices():
        if notice.code == "binary_missing":
            binary_ok = False
        if notice.code == "model_missing":
            models_ok = False
    telemetry_ok = state.telemetry.source != "ram" or state.telemetry.gpus is None
    if total == 0 or not binary_ok or not models_ok:
        status = "degraded"
    elif not state.load.models_loaded:
        # Lazy residency (api.md 2026-07-22c): a cold or idle-unloaded server is
        # HEALTHY — it warms the loadout on the first inference request. Zero slots
        # resident is the zero-friction default, not a fault.
        status = "ok"
    elif ready == total:
        status = "ok"
    elif ready == 0:
        status = "degraded" if any(s.state == "failed" for s in manager.slots) else "starting"
    else:
        status = "degraded"
    return {
        "status": status,
        "checks": {
            "binary": binary_ok,
            "models": models_ok,
            "telemetry": telemetry_ok,
            "slots_ready": ready,
            "slots_total": total,
        },
    }


@router.post("/v1/system/apply")
async def post_apply(request: Request) -> dict:
    state = state_of(request)
    staged = state.config.staged
    if staged is not None and "quant_family" in staged and state.manager.family_download_active:
        raise BonsaiError(
            409,
            "cannot switch quant family while its download is still running",
            code="family_download_active",
        )
    if staged is not None and not state.is_idle():
        # Never mid-run (api.md): defer the commit to the next idle transition
        # (ActivityTracker idle callback / job-terminal listener fire it).
        state.apply_pending = True
        requires_restart, pending_until_idle = state.config.pending_effects()
        return {
            "applied": False,
            "requires_restart": requires_restart,
            "pending_until_idle": pending_until_idle,
        }
    # Idle (or nothing staged): commit now. Loadout-affecting keys (quant_family,
    # loadout, server, gateways) replan/rebind only at the next boot —
    # requires_restart in the response says exactly that.
    state.apply_pending = False
    result = state.config.apply()
    if result["applied"]:
        # Provider model lists refresh on apply (api.md §11). refresh() never raises
        # — an unreachable provider becomes a notice, not an error here.
        await state.providers.refresh(state.config.settings.providers)
        # Catalog connectors commit at idle without a restart (api.md 2026-07-22c):
        # bring the MCP manager in line with the committed mcp_servers + connectors.
        if state.mcp is not None:
            await state.mcp.resync(combined_mcp_servers(state.config.settings))
    return result


@router.post("/v1/system/search_test")
async def post_search_test(request: Request) -> dict:
    """Test the configured web-search provider (api.md §11). Never throws: every
    failure — bad config, transport error, a broken parse — comes back as
    `ok: false` with an honest `error` string. Powers the settings "test" button."""
    state = state_of(request)
    try:
        body = await request.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    query = body.get("query")
    if not isinstance(query, str) or not query.strip():
        query = DEFAULT_SEARCH_TEST_QUERY

    search_settings = state.config.settings.search
    provider_name = search_settings.provider
    try:
        provider = build_search_provider(search_settings)
        results = await provider.search(query, SEARCH_TEST_MAX_RESULTS)
    except Exception as exc:  # noqa: BLE001 - the test button reports, never raises
        return {"ok": False, "provider": provider_name, "results": [], "error": str(exc)}
    if not results:
        return {
            "ok": False,
            "provider": provider_name,
            "results": [],
            "error": "search returned no results (misconfigured provider or a broken parse)",
        }
    return {
        "ok": True,
        "provider": provider_name,
        "results": [{"title": r.title, "url": r.url} for r in results[:SEARCH_TEST_MAX_RESULTS]],
        "error": None,
    }
