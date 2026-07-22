"""WhatsApp QR device-linking gateway (changed 2026-07-22b): build decision, the QR
state machine (unlinked/awaiting_scan/linked) with a REAL rendered QR, unlink, and the
outbound notify hook. No network anywhere and no live WhatsApp — the send path is
exercised through an injected fake backend (link+send against live WhatsApp is
unverified by design; see docs/gateways.md)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from suiban.config import Settings
from suiban.gateways.whatsapp import (
    StubLinkBackend,
    WhatsAppGateway,
    build_whatsapp_gateway,
    render_qr_ascii,
)
from suiban.sched.planner import Notice


def _settings(
    enabled: bool = True, linked: bool = False, to_number: str = "15551234567"
) -> Settings:
    return Settings.model_validate(
        {"gateways": {"whatsapp": {"enabled": enabled, "linked": linked, "to_number": to_number}}}
    )


class FakeBackend:
    """Records sends; never touches the network. `fail=True` makes send() raise."""

    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    def pairing_string(self) -> str:
        return "bonsai-whatsapp-link:fake-ref"

    async def send(self, to_number: str, text: str) -> None:
        if self.fail:
            raise RuntimeError("no live session")
        self.sent.append((to_number, text))


def _gateway(
    tmp_path: Path, *, linked: bool = False, to_number: str = "15551234567", backend=None
) -> tuple[WhatsAppGateway, list[Notice]]:
    notices: list[Notice] = []
    gateway = WhatsAppGateway(
        to_number=to_number,
        linked=linked,
        state_dir=tmp_path / "whatsapp",
        notices=notices,
        backend=backend if backend is not None else FakeBackend(),
    )
    return gateway, notices


# -- build decision logic -----------------------------------------------------
def test_build_disabled_is_silent(tmp_path: Path) -> None:
    notices: list[Notice] = []
    assert (
        build_whatsapp_gateway(_settings(enabled=False), notices=notices, state_dir=tmp_path / "wa")
        is None
    )
    assert notices == []


def test_build_enabled_returns_gateway_even_when_unlinked(tmp_path: Path) -> None:
    """Enabled-but-not-yet-linked is valid: the gateway exists to serve the QR flow."""
    notices: list[Notice] = []
    gateway = build_whatsapp_gateway(_settings(), notices=notices, state_dir=tmp_path / "wa")
    assert isinstance(gateway, WhatsAppGateway)
    assert notices == []
    assert gateway.running is False
    assert gateway.linked is False


# -- QR rendering + state machine ---------------------------------------------
def test_render_qr_ascii_is_a_real_scannable_qr() -> None:
    ascii_qr = render_qr_ascii("bonsai-whatsapp-link:sample-pairing-string")
    assert "█" in ascii_qr  # real QR block glyphs, not a placeholder string
    assert len(ascii_qr.splitlines()) > 8  # a QR grid, not one line


def test_qr_state_awaiting_scan_returns_a_real_qr(tmp_path: Path) -> None:
    gateway, _ = _gateway(tmp_path, backend=FakeBackend())
    body = gateway.qr_state()
    assert body["state"] == "awaiting_scan"
    assert body["qr"] == "bonsai-whatsapp-link:fake-ref"
    assert "█" in body["qr_ascii"]


def test_qr_state_linked_clears_the_qr(tmp_path: Path) -> None:
    gateway, _ = _gateway(tmp_path, linked=True)
    assert gateway.qr_state() == {"state": "linked", "qr": None, "qr_ascii": None}


def test_stub_backend_renders_a_real_qr_from_its_pairing_string() -> None:
    """Even without neonize the default backend yields a scannable QR (honest: it does
    not complete a live link)."""
    backend = StubLinkBackend()
    assert "█" in render_qr_ascii(backend.pairing_string())


# -- unlink -------------------------------------------------------------------
def test_unlink_forgets_the_session_and_persists(tmp_path: Path) -> None:
    persisted: list[bool] = []
    state_dir = tmp_path / "whatsapp"
    state_dir.mkdir()
    (state_dir / "session.json").write_text("{}")  # a "linked" session on disk
    gateway = WhatsAppGateway(
        to_number="1",
        linked=True,
        state_dir=state_dir,
        backend=FakeBackend(),
        persist_linked=persisted.append,
    )
    assert gateway.linked is True
    assert gateway.unlink() == {"state": "unlinked"}
    assert gateway.linked is False
    assert not (state_dir / "session.json").exists()
    assert persisted == [False]


# -- outbound notify hook -----------------------------------------------------
async def test_notify_sends_when_linked_and_started(tmp_path: Path) -> None:
    backend = FakeBackend()
    gateway, _ = _gateway(tmp_path, linked=True, backend=backend)
    await gateway.start()
    gateway.notify("research", "Deep research finished: sky", "it is blue")
    gateway.notify("schedule", "title only", "")
    await asyncio.sleep(0.05)
    assert ("15551234567", "Deep research finished: sky: it is blue") in backend.sent
    assert ("15551234567", "title only") in backend.sent


async def test_notify_noops_when_unlinked(tmp_path: Path) -> None:
    backend = FakeBackend()
    gateway, _ = _gateway(tmp_path, linked=False, backend=backend)
    await gateway.start()  # started, but not linked
    gateway.notify("research", "t", "s")
    await asyncio.sleep(0.05)
    assert backend.sent == []  # nothing sent while unlinked


async def test_notify_noops_when_not_started(tmp_path: Path) -> None:
    backend = FakeBackend()
    gateway, _ = _gateway(tmp_path, linked=True, backend=backend)
    gateway.notify("research", "t", "s")  # never started
    await asyncio.sleep(0.05)
    assert backend.sent == []
    await gateway.stop()  # idempotent


async def test_notify_noops_without_a_recipient(tmp_path: Path) -> None:
    backend = FakeBackend()
    gateway, _ = _gateway(tmp_path, linked=True, to_number="", backend=backend)
    await gateway.start()
    gateway.notify("research", "t", "s")
    await asyncio.sleep(0.05)
    assert backend.sent == []


async def test_failed_send_surfaces_a_notice_never_raises(tmp_path: Path) -> None:
    gateway, notices = _gateway(tmp_path, linked=True, backend=FakeBackend(fail=True))
    await gateway.start()
    gateway.notify("research", "Deep research finished", "the query")
    await asyncio.sleep(0.05)
    assert [n.code for n in notices] == ["whatsapp_send_failed"]
    # A second failure replaces the notice instead of stacking duplicates.
    gateway.notify("schedule", "Scheduled run failed", "boom")
    await asyncio.sleep(0.05)
    assert [n.code for n in notices] == ["whatsapp_send_failed"]


# -- app wiring ---------------------------------------------------------------
def test_notify_gateways_fans_out_to_both_gateways(client: TestClient) -> None:
    """AppState.notify_gateways is the single hook research jobs and the scheduler fire;
    every configured gateway gets the ping."""
    state = client.app.state.bonsai
    calls: list[tuple[str, str, str, str]] = []
    original_gateway, original_whatsapp = state.gateway, state.whatsapp
    try:
        state.gateway = SimpleNamespace(
            notify=lambda kind, title, summary: calls.append(("telegram", kind, title, summary))
        )
        state.whatsapp = SimpleNamespace(
            notify=lambda kind, title, summary: calls.append(("whatsapp", kind, title, summary))
        )
        state.notify_gateways("schedule", "Scheduled run finished: digest", "all quiet")
    finally:  # the lifespan teardown calls .stop() on whatever is attached
        state.gateway, state.whatsapp = original_gateway, original_whatsapp
    assert calls == [
        ("telegram", "schedule", "Scheduled run finished: digest", "all quiet"),
        ("whatsapp", "schedule", "Scheduled run finished: digest", "all quiet"),
    ]


def test_whatsapp_disabled_by_default_in_app(client: TestClient) -> None:
    assert client.app.state.bonsai.whatsapp is None


# -- QR endpoints over HTTP ---------------------------------------------------
def test_qr_endpoint_reports_unlinked_when_disabled(client: TestClient) -> None:
    body = client.get("/v1/gateways/whatsapp/qr").json()
    assert body == {"state": "unlinked", "qr": None, "qr_ascii": None}
    assert client.post("/v1/gateways/whatsapp/unlink").json() == {"state": "unlinked"}


def test_qr_endpoints_drive_an_enabled_gateway(client: TestClient, tmp_path: Path) -> None:
    state = client.app.state.bonsai
    gateway = WhatsAppGateway(
        to_number="1", linked=False, state_dir=tmp_path / "wa", backend=FakeBackend()
    )
    original = state.whatsapp
    try:
        state.whatsapp = gateway
        qr = client.get("/v1/gateways/whatsapp/qr").json()
        assert qr["state"] == "awaiting_scan"
        assert "█" in qr["qr_ascii"]
        assert client.post("/v1/gateways/whatsapp/unlink").json() == {"state": "unlinked"}
    finally:
        state.whatsapp = original
