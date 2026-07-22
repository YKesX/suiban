"""/v1/gateways/whatsapp — QR device-linking (api.md §8, changed 2026-07-22b).

WhatsApp links via the WhatsApp Web multi-device protocol: enable the gateway, then
scan a QR with the phone (Settings → Linked Devices). These endpoints drive that flow;
outbound pings then reach the linked device. When the gateway is disabled there is no
built gateway object, so the endpoints report the honest `unlinked` state.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from suiban.app import state_of

router = APIRouter()


@router.get("/v1/gateways/whatsapp/qr")
async def whatsapp_qr(request: Request) -> dict:
    """`{ state: unlinked|awaiting_scan|linked, qr, qr_ascii }`. Poll while
    `awaiting_scan`; `qr` clears once `linked`. Disabled gateway → `unlinked`."""
    gateway = state_of(request).whatsapp
    if gateway is None:
        return {"state": "unlinked", "qr": None, "qr_ascii": None}
    return gateway.qr_state()


@router.post("/v1/gateways/whatsapp/unlink")
async def whatsapp_unlink(request: Request) -> dict:
    """Forget the linked device session → `{ state: unlinked }` (idempotent; a
    disabled/absent gateway is already unlinked)."""
    gateway = state_of(request).whatsapp
    if gateway is None:
        return {"state": "unlinked"}
    return gateway.unlink()
