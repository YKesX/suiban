"""End-to-end app behavior against the mock llama backend (SUIBAN_LLAMA_MOCK)."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from conftest import FakeTelemetry
from suiban.app import create_app


def test_startup_plans_24gb_loadout_but_holds_models_cold(client: TestClient) -> None:
    """Lazy residency (api.md 2026-07-22c): the loadout is PLANNED at boot but NO slots
    are started — serve comes up healthy with zero models resident."""
    body = client.get("/v1/system").json()
    assert body["loadout"]["tier"] == "24gb"
    slots = body["loadout"]["slots"]
    assert [s["slot_id"] for s in slots] == ["orchestrator", "utility", "worker-1", "worker-2"]
    # Not started yet: every planned slot reads "cold", not "ready".
    assert all(s["state"] == "cold" for s in slots)
    assert body["runtime"] == {"keep_alive": "5", "models_loaded": False, "state": "cold"}
    assert body["kv"]["v_type"] == "tq4_0"
    assert body["quant_family"] == {
        "configured": "ternary",
        "effective": "ternary",
        "degraded": False,
        "reason": None,
    }
    assert body["gpus"][0]["vram_total_mb"] == 24 * 1024
    assert body["capabilities"]["ultra_parallel"] is True
    assert body["dspark"] == {"enabled": False, "available": True}


def test_first_inference_warms_the_loadout(client: TestClient) -> None:
    """A cold start warms the planned slots on the first inference request; afterwards
    the loadout is resident and /v1/system reports it ready."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    body = client.get("/v1/system").json()
    assert body["runtime"]["models_loaded"] is True
    assert body["runtime"]["state"] == "ready"
    assert all(s["state"] == "ready" for s in body["loadout"]["slots"])


def test_health_ok_when_cold_and_when_warm(client: TestClient) -> None:
    # Cold (lazy) is healthy: zero resident slots is the zero-friction default.
    body = client.get("/v1/system/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["slots_ready"] == 0
    assert body["checks"]["slots_total"] == 4
    # After a warming request every slot is ready.
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    body = client.get("/v1/system/health").json()
    assert body["status"] == "ok"
    assert body["checks"]["slots_ready"] == body["checks"]["slots_total"] == 4


def test_root_landing_ready_when_healthy(client: TestClient) -> None:
    # GET / is the human landing page: a browser used to get a bare 404 here.
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    body = resp.text
    assert "Suiban is ready" in body
    assert "https://github.com/YKesX/dai" in body
    assert "https://github.com/YKesX/sentei" in body


def test_landing_html_both_states() -> None:
    from suiban.app import _landing_html

    ready = _landing_html(True)
    assert "Suiban is ready" in ready
    assert "YKesX/dai" in ready and "YKesX/sentei" in ready
    not_ready = _landing_html(False)
    assert "Suiban is not ready" in not_ready
    assert "Suiban is ready" not in not_ready


def test_settings_staging_flow_over_http(client: TestClient) -> None:
    resp = client.patch("/v1/settings", json={"kv": {"preset": "aggressive"}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["current"]["kv"]["preset"] == "recommended"  # unchanged until apply
    assert body["staged"]["kv"]["preset"] == "aggressive"

    resp = client.post("/v1/system/apply")
    assert resp.json()["applied"] is True
    # kv maps to llama-server launch flags: honestly reported as requires_restart
    # (api.md 2026-07-21d), no longer mislabeled pending_until_idle.
    assert resp.json()["requires_restart"] == ["kv"]
    assert resp.json()["pending_until_idle"] == []

    body = client.get("/v1/settings").json()
    assert body["current"]["kv"]["preset"] == "aggressive"
    assert body["staged"] is None


def test_settings_secret_never_echoed(client: TestClient) -> None:
    client.patch(
        "/v1/settings", json={"gateways": {"telegram": {"enabled": True, "token": "42:top"}}}
    )
    client.post("/v1/system/apply")
    body = client.get("/v1/settings").json()
    telegram = body["current"]["gateways"]["telegram"]
    assert telegram == {
        "enabled": True,
        "token_set": True,
        "allowed_chat_ids": [],
        "require_pairing": True,
        "rate_limit_per_min": 20,
    }
    assert "42:top" not in json.dumps(body)


def test_settings_invalid_patch_is_400_envelope(client: TestClient) -> None:
    resp = client.patch("/v1/settings", json={"quant_family": "3bit"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"
    # nothing got staged
    assert client.get("/v1/settings").json()["staged"] is None


def test_mock_chat_roundtrip_through_manager(client: TestClient) -> None:
    """The slot seam: ask the manager for a client, talk OpenAI to the mock slot."""
    import anyio

    # Lazy residency: warm the loadout first (a chat request), then the manager has a
    # live orchestrator slot to hand out a client for.
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "hi"}]},
    )
    manager = client.app.state.bonsai.manager

    async def roundtrip() -> dict:
        async with manager.client_for("orchestrator") as slot_client:
            resp = await slot_client.post(
                "/v1/chat/completions",
                json={"model": "bonsai-27b", "messages": [{"role": "user", "content": "hello"}]},
            )
            return resp.json()

    body = anyio.run(roundtrip)
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "deterministic" in body["choices"][0]["message"]["content"]


def test_cpu_only_app(bonsai_home: Path) -> None:
    app = create_app(
        home=bonsai_home,
        telemetry_provider=FakeTelemetry(None, ram_mb=62 * 1024),
        compute_backend="cpu",
        use_mock=True,
    )
    with TestClient(app) as client:
        body = client.get("/v1/system").json()
        assert body["gpus"] is None
        assert body["telemetry_source"] == "ram"
        assert body["loadout"]["tier"] == "cpu"
        assert body["dspark"]["available"] is False
        models = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
        assert models["bonsai-27b"]["bonsai"]["role"] == "orchestrator"
        assert models["bonsai-8b"]["bonsai"]["role"] == "none"
        assert models["bonsai-8b"]["bonsai"]["resident"] is False


def test_real_backend_boot_degrades_honestly_without_binaries(bonsai_home: Path) -> None:
    """Lifecycle with use_mock=False on a machine with nothing installed: serve still
    boots (cold, healthy — lazy residency), and a KV notice already tells the truth. The
    first inference ATTEMPT warms the loadout, its slots fail VISIBLY, and /v1/system then
    carries the binary_missing notice + failed slot states + degraded health. Never a
    crash — a 409 on the attempt, not a 500."""
    app = create_app(
        home=bonsai_home,
        telemetry_provider=FakeTelemetry([24 * 1024]),
        compute_backend="cuda",
        use_mock=False,
    )
    with TestClient(app) as client:
        # Cold boot: KV fallback already resolved (not slot-dependent), but no load
        # attempted yet, so serve is healthy and no binary_missing notice exists.
        body = client.get("/v1/system").json()
        codes = {n["code"] for n in body["notices"]}
        assert "turboquant_prebuilt_fallback" in codes  # no TURBOQUANT marker installed
        assert "binary_missing" not in codes
        assert body["kv"]["v_type"] == "q8_0"
        assert body["kv"]["turboquant"]["fallback_active"] is True
        assert body["runtime"]["state"] == "cold"
        assert client.get("/v1/system/health").json()["status"] == "ok"

        # The first inference attempt warms the loadout; with no binaries every slot
        # fails to launch and the request is a clean 409 (never a crash).
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 409

        body = client.get("/v1/system").json()
        codes = {n["code"] for n in body["notices"]}
        assert "binary_missing" in codes
        assert body["loadout"]["slots"], "the loadout is still planned and reported"
        assert all(s["state"] == "failed" for s in body["loadout"]["slots"])
        health = client.get("/v1/system/health").json()
        assert health["status"] == "degraded"
        assert health["checks"]["binary"] is False


def test_12gb_app_reports_family_degradation(bonsai_home: Path) -> None:
    app = create_app(
        home=bonsai_home,
        telemetry_provider=FakeTelemetry([12 * 1024]),
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as client:
        body = client.get("/v1/system").json()
        qf = body["quant_family"]
        assert qf["configured"] == "ternary"
        assert qf["effective"] == "1bit"
        assert qf["degraded"] is True
        assert qf["reason"]
        assert any(n["code"] == "family_degraded" for n in body["notices"])
