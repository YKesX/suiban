"""Non-loopback API auth (api.md 2026-07-22 security, H6).

Loopback binds stay open (unchanged zero-friction default). A non-loopback bind mints
and persists an auth token and requires `Authorization: Bearer <token>` on every route
except GET /v1/system/health. TestClient simulates each bind by writing server.host.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import FakeTelemetry
from suiban.app import create_app
from suiban.config import ConfigManager, host_is_loopback


def _app(home: Path, host: str = "127.0.0.1"):
    if host != "127.0.0.1":
        cfg = ConfigManager(home)
        cfg.load()
        cfg.stage({"server": {"host": host}})
        cfg.apply()
    return create_app(
        home=home,
        telemetry_provider=FakeTelemetry([24 * 1024]),
        compute_backend="cuda",
        use_mock=True,
    )


def test_host_is_loopback_classification() -> None:
    for h in ("127.0.0.1", "localhost", "::1", "127.0.0.5", "[::1]"):
        assert host_is_loopback(h), h
    for h in ("0.0.0.0", "192.168.1.10", "10.0.0.5", "::", "example.com"):
        assert not host_is_loopback(h), h


def test_loopback_bind_requires_no_auth_and_mints_no_token(bonsai_home: Path) -> None:
    with TestClient(_app(bonsai_home)) as client:
        body = client.get("/v1/system").json()
        assert body["security"]["auth_required"] is False
        settings = client.get("/v1/settings").json()["current"]
        assert settings["server"]["auth_token_set"] is False  # no token on loopback


def test_non_loopback_bind_requires_bearer_token(bonsai_home: Path) -> None:
    with TestClient(_app(bonsai_home, host="0.0.0.0")) as client:
        # health is always reachable (readiness probe) — no auth.
        assert client.get("/v1/system/health").status_code == 200
        # everything else 401s without a token, in the contract envelope + version header.
        resp = client.get("/v1/system")
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"
        assert set(resp.json()["error"]) == {"type", "message", "code"}
        assert resp.headers["X-Bonsai-Api-Version"] == "1"
        # the token was auto-generated + persisted at the non-loopback apply.
        token = client.app.state.bonsai.config.settings.server.auth_token
        assert token
        ok = client.get("/v1/system", headers={"Authorization": f"Bearer {token}"})
        assert ok.status_code == 200
        assert ok.json()["security"]["auth_required"] is True
        # a wrong / missing token still 401s.
        bad = client.get("/v1/system", headers={"Authorization": "Bearer wrong"})
        assert bad.status_code == 401
        assert client.post("/v1/system/apply").status_code == 401
        # the token is never echoed — only auth_token_set.
        body = client.get("/v1/settings", headers={"Authorization": f"Bearer {token}"}).json()
        assert body["current"]["server"]["auth_token_set"] is True
        assert "auth_token" not in body["current"]["server"]
        assert token not in json.dumps(body)
