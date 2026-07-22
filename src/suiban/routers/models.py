"""GET /v1/models — OpenAI-compatible list plus bonsai metadata."""

from __future__ import annotations

from fastapi import APIRouter, Request

from suiban.app import state_of
from suiban.installer import models as model_store
from suiban.sched.budget import MAX_CTX, MODELS, QUANT_NAME

router = APIRouter()


@router.get("/v1/models")
async def list_models(request: Request) -> dict:
    state = state_of(request)
    loadout = state.loadout
    data = []
    for model in MODELS:
        slot = loadout.slot_for_model(model)
        resident = slot is not None
        family = slot.family if slot else state.config.settings.quant_family
        role = slot.role if slot else "none"
        if resident and model == "bonsai-27b" and loadout.utility_shared_with_orchestrator:
            role = "orchestrator"  # CPU-only: orchestrator also serves utility duty
        data.append(
            {
                "id": model,
                "object": "model",
                "owned_by": "prism-ml",
                "bonsai": {
                    "family": family,
                    "quant": QUANT_NAME[family],
                    "role": role,
                    "resident": resident,
                    "ctx": slot.ctx if slot else MAX_CTX[model],
                    "vision": model == "bonsai-27b",
                    "downloaded_families": model_store.downloaded_families(model),
                },
            }
        )
    # External provider models append after the bonsai family (api.md §2, additive
    # 2026-07-21c): "<provider>/<model>", bonsai.external true, resident = provider
    # reachability at the last refresh.
    data.extend(state.providers.model_entries())
    return {"object": "list", "data": data}
