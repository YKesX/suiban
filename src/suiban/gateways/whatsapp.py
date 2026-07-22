"""WhatsApp gateway: QR device-linking + OUTBOUND-only notification pings
(changed 2026-07-22b — was the Cloud-API-token gateway).

Linking works like WhatsApp Web / Linked Devices: enable the gateway, fetch a QR
(`GET /v1/gateways/whatsapp/qr`), scan it with the phone, and the device session lives
under `~/.bonsai/whatsapp/` (never a repo). No Cloud-API token, no phone-number id.
Once linked, the same events Telegram pings — deep-research completions and
scheduled-run results — reach WhatsApp too, sent through the linked session. Nothing is
ever received (inbound relay is TODO(v1.2)), consistent with suiban's loopback posture.

Pluggable link backend: if the optional native library `neonize` is importable we drive
a real WhatsApp Web multi-device session through it; otherwise a stub backend stands in.
EITHER WAY we always render a REAL, scannable QR from the pairing string with the
`qrcode` pure-Python lib (terminal ASCII + the data string).

HONESTY / KNOWN_ISSUE: without a live WhatsApp account the link handshake and the send
path CANNOT be exercised end to end here — the stub backend produces a real QR but no
real session, so scanning it does not complete a link, and `send` has nowhere to go.
`neonize` is an OPTIONAL native dependency, unverified against live WhatsApp in this
build. The state machine, QR rendering, unlink, and notify no-op-when-unlinked behavior
ARE fully exercised; the real link+send is `TODO(v1.2)` (see docs/gateways.md and the
repo KNOWN_ISSUES note).
"""

from __future__ import annotations

import asyncio
import io
import logging
import secrets
from collections.abc import Callable
from pathlib import Path

import qrcode

from suiban.config import Settings
from suiban.sched.planner import Notice

logger = logging.getLogger(__name__)

# The stub session credential file: its presence marks a device as linked. A real
# neonize backend would write/read its own session material alongside it.
SESSION_FILE_NAME = "session.json"


def render_qr_ascii(data: str) -> str:
    """A real, scannable terminal QR for `data`, rendered with the pure-Python `qrcode`
    lib (no Pillow needed for ASCII). Same encoding a phone camera reads off the screen —
    the DATA is what differs between the real and stub backends, not the rendering."""
    qr = qrcode.QRCode(border=1)
    qr.add_data(data)
    qr.make(fit=True)
    buffer = io.StringIO()
    qr.print_ascii(out=buffer)
    return buffer.getvalue()


class StubLinkBackend:
    """Stand-in link backend used when `neonize` is not installed.

    Produces a stable placeholder pairing string (so a REAL QR renders), but there is no
    live WhatsApp session behind it: scanning the QR does not link, and `send` raises.
    This is deliberately honest — the alternative would be pretending an unlinked gateway
    can deliver messages."""

    name = "stub"

    def __init__(self) -> None:
        # Stable for the process so polling the QR endpoint returns the same code.
        self._ref = secrets.token_urlsafe(16)

    def pairing_string(self) -> str:
        return f"bonsai-whatsapp-link:{self._ref}"

    async def send(self, to_number: str, text: str) -> None:
        raise RuntimeError(
            "WhatsApp is not linked to a live session (no neonize backend / device link). "
            "Link a device by scanning the QR from GET /v1/gateways/whatsapp/qr first."
        )


def _load_link_backend(state_dir: Path):
    """Pick the link backend: the real neonize-backed session if the optional native
    library is importable, else the honest stub. neonize is NEVER a hard dependency."""
    try:
        import neonize  # noqa: F401  (optional native dep; presence check only)
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return StubLinkBackend()
    # TODO(v1.2): drive a real WhatsApp Web multi-device session through neonize
    # (persist/restore the linked-device session under state_dir, surface its pairing
    # QR, and route send() through it). Unverified against live WhatsApp in this build,
    # so we fall back to the stub's honest behavior until it is wired and tested.
    logger.info("neonize is present but the live WhatsApp backend is not wired yet (TODO v1.2)")
    return StubLinkBackend()


class WhatsAppGateway:
    """QR-linked, outbound-only WhatsApp gateway.

    State machine (api.md 2026-07-22b): `linked` when a device session exists;
    `awaiting_scan` while a QR is being shown; the DISABLED case (`unlinked`) is handled
    by the router when no gateway is built. `notices` is the shared /v1/system notice
    list (send failures surface there, deduplicated by code)."""

    available = True

    def __init__(
        self,
        *,
        to_number: str,
        linked: bool,
        state_dir: Path,
        notices: list[Notice] | None = None,
        backend=None,
        persist_linked: Callable[[bool], None] | None = None,
    ) -> None:
        self._to_number = to_number
        self._state_dir = state_dir
        self._notices = notices if notices is not None else []
        self._backend = backend if backend is not None else _load_link_backend(state_dir)
        self._persist_linked = persist_linked
        self._session_file = state_dir / SESSION_FILE_NAME
        self._linked = linked or self._session_file.exists()
        self._started = False

    @property
    def running(self) -> bool:
        return self._started

    @property
    def linked(self) -> bool:
        return self._linked

    async def start(self) -> None:
        """Outbound-only: nothing to poll or bind — arm notify(). A real backend would
        also resume its persisted session here."""
        self._started = True
        logger.info(
            "whatsapp gateway: armed (%s, backend=%s)",
            "linked" if self._linked else "awaiting device link",
            self._backend.name,
        )

    async def stop(self) -> None:  # symmetric with start(); idempotent
        self._started = False

    # -- QR linking --------------------------------------------------------
    def qr_state(self) -> dict:
        """The GET /v1/gateways/whatsapp/qr body. Once linked the QR clears; otherwise
        a real, scannable QR (ASCII + the raw data string) is returned to poll against."""
        if self._linked:
            return {"state": "linked", "qr": None, "qr_ascii": None}
        pairing = self._backend.pairing_string()
        return {"state": "awaiting_scan", "qr": pairing, "qr_ascii": render_qr_ascii(pairing)}

    def unlink(self) -> dict:
        """Forget the linked device session (POST /v1/gateways/whatsapp/unlink)."""
        self._linked = False
        self._session_file.unlink(missing_ok=True)
        if self._persist_linked is not None:
            self._persist_linked(False)
        logger.info("whatsapp gateway: device unlinked; session forgotten")
        return {"state": "unlinked"}

    # -- outbound notifications --------------------------------------------
    def notify(self, kind: str, title: str, summary: str) -> None:
        """The generalized notification hook (research completions, scheduled runs):
        one fire-and-forget text to the linked device. No-ops cleanly when the gateway
        is not started, not linked, or has no recipient configured."""
        if not self._started or not self._linked or not self._to_number:
            return
        text = f"{title}: {summary}" if summary else title
        logger.debug("whatsapp gateway: %s notification: %s", kind, title)
        asyncio.get_running_loop().create_task(self._send_safe(text))

    async def send_text(self, text: str) -> None:
        """One outbound text through the linked session; raises on any failure."""
        await self._backend.send(self._to_number, text)

    async def _send_safe(self, text: str) -> None:
        try:
            await self.send_text(text)
        except Exception as exc:  # noqa: BLE001 - a failed ping must never crash anything
            logger.warning("whatsapp gateway: send failed: %s", exc)
            self._notices[:] = [n for n in self._notices if n.code != "whatsapp_send_failed"]
            self._notices.append(
                Notice(
                    "warn",
                    "whatsapp_send_failed",
                    f"WhatsApp notification could not be sent: {exc}. Re-link the device "
                    "(Settings → WhatsApp → scan QR) and check gateways.whatsapp.to_number.",
                )
            )


def build_whatsapp_gateway(
    settings: Settings,
    *,
    notices: list[Notice],
    state_dir: Path,
    backend=None,
    persist_linked: Callable[[bool], None] | None = None,
) -> WhatsAppGateway | None:
    """Build the WhatsApp gateway when enabled; None (silent) when disabled. Enabled but
    not yet linked is a VALID state — the gateway exists to serve the QR-link flow — so
    unlike the old token gateway there is no 'missing credential' short-circuit."""
    wa = settings.gateways.whatsapp
    if not wa.enabled:
        return None
    state_dir.mkdir(parents=True, exist_ok=True)
    return WhatsAppGateway(
        to_number=wa.to_number,
        linked=wa.linked,
        state_dir=state_dir,
        notices=notices,
        backend=backend,
        persist_linked=persist_linked,
    )
