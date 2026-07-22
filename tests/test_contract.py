"""Contract tests: EVERY route of docs/api.md exists with the right methods, every
response carries X-Bonsai-Api-Version, every non-2xx body is the error envelope, and
implemented endpoints match the documented shapes.
"""

from __future__ import annotations

import re

import pytest
from fastapi.routing import APIRoute

# (method, path-template) — path params normalized to {}.
CONTRACT_ROUTES: list[tuple[str, str]] = [
    ("POST", "/v1/chat/completions"),
    ("GET", "/v1/models"),
    ("POST", "/v1/jobs"),
    ("GET", "/v1/jobs"),
    ("GET", "/v1/jobs/{}"),
    ("GET", "/v1/jobs/{}/events"),
    ("GET", "/v1/jobs/{}/report"),
    ("DELETE", "/v1/jobs/{}"),
    ("GET", "/v1/system"),
    ("GET", "/v1/system/budget"),
    ("GET", "/v1/system/health"),
    ("POST", "/v1/system/apply"),
    ("POST", "/v1/system/search_test"),  # additive 2026-07-21c: web-search test
    ("GET", "/v1/memory"),
    ("GET", "/v1/memory/search"),
    ("POST", "/v1/memory"),
    ("PUT", "/v1/memory/{}"),
    ("DELETE", "/v1/memory/{}"),
    ("GET", "/v1/memory/state"),
    ("PUT", "/v1/memory/state/{}"),  # additive 2026-07-21b: state-file editing
    ("DELETE", "/v1/memory/state/{}"),  # additive 2026-07-22d: state-file delete
    ("GET", "/v1/memory/sessions"),
    ("GET", "/v1/memory/sessions/{}"),
    ("DELETE", "/v1/memory/sessions/{}"),  # additive 2026-07-22d: chat delete
    ("POST", "/v1/memory/sessions/import"),  # additive 2026-07-22b: import chats
    ("GET", "/v1/skills"),
    ("POST", "/v1/skills/import"),  # additive 2026-07-22c: import agentskills.io skills
    ("GET", "/v1/skills/{}"),
    ("PUT", "/v1/skills/{}"),
    ("DELETE", "/v1/skills/{}"),
    ("GET", "/v1/modes"),
    ("GET", "/v1/modes/{}"),
    ("GET", "/v1/settings"),
    ("PATCH", "/v1/settings"),
    # -- additive 2026-07-21b: projects + schedules --------------------------
    ("GET", "/v1/projects"),
    ("POST", "/v1/projects"),
    ("GET", "/v1/projects/{}"),
    ("PATCH", "/v1/projects/{}"),
    ("DELETE", "/v1/projects/{}"),
    ("GET", "/v1/projects/{}/docs"),
    ("POST", "/v1/projects/{}/docs"),
    ("GET", "/v1/projects/{}/docs/{}"),
    ("DELETE", "/v1/projects/{}/docs/{}"),
    ("GET", "/v1/schedules"),
    ("POST", "/v1/schedules"),
    ("GET", "/v1/schedules/{}"),
    ("PATCH", "/v1/schedules/{}"),
    ("DELETE", "/v1/schedules/{}"),
    ("POST", "/v1/schedules/{}/run"),
    # SLAP protocol observability (api.md §12, additive 2026-07-22b).
    ("GET", "/v1/slap"),
    ("GET", "/v1/slap/schema/{}"),
    ("GET", "/v1/slap/trace/{}"),
    # WhatsApp QR device-linking (api.md §8, changed 2026-07-22b).
    ("GET", "/v1/gateways/whatsapp/qr"),
    ("POST", "/v1/gateways/whatsapp/unlink"),
    # MCP connector catalog (api.md 2026-07-22c).
    ("GET", "/v1/mcp/connectors"),
]


def _normalize(path: str) -> str:
    return re.sub(r"\{[^}]*\}", "{}", path)


def _walk(routes) -> list[APIRoute]:
    """Flatten APIRoutes. Newer FastAPI wraps include_router() results in lazy
    _IncludedRouter objects exposing the original APIRouter as `original_router`."""
    out: list[APIRoute] = []
    for route in routes:
        if isinstance(route, APIRoute):
            out.append(route)
        elif hasattr(route, "original_router"):
            out.extend(_walk(route.original_router.routes))
        elif hasattr(route, "routes"):
            out.extend(_walk(route.routes))
    return out


def registered_routes(app) -> set[tuple[str, str]]:
    out = set()
    for route in _walk(app.routes):
        for method in route.methods - {"HEAD", "OPTIONS"}:
            out.add((method, _normalize(route.path)))
    return out


@pytest.mark.parametrize(("method", "path"), CONTRACT_ROUTES)
def test_every_contract_route_is_registered(client, method: str, path: str) -> None:
    assert (method, path) in registered_routes(client.app)


def test_no_extra_v1_surface(client) -> None:
    """The frozen contract is also an upper bound: no undocumented /v1 routes."""
    assert registered_routes(client.app) == set(CONTRACT_ROUTES)


def test_api_version_header_on_success_and_errors(client) -> None:
    assert client.get("/v1/models").headers["X-Bonsai-Api-Version"] == "1"
    assert client.get("/v1/nope").headers["X-Bonsai-Api-Version"] == "1"
    assert client.post("/v1/chat/completions", json={}).headers["X-Bonsai-Api-Version"] == "1"


def test_error_envelope_on_unknown_route(client) -> None:
    resp = client.get("/v1/nope")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"type", "message", "code"}
    assert body["error"]["type"] == "not_found_error"


def test_no_501s_remain_anywhere(client) -> None:
    """Every contract route is implemented now: nothing may answer 501. Requests are
    deliberately minimal — 400/404 responses are fine, 'not implemented' is not."""
    for method, path in CONTRACT_ROUTES:
        concrete = path.replace("{}", "nope")
        resp = client.request(
            method, concrete, json={} if method in ("POST", "PUT", "PATCH") else None
        )
        assert resp.status_code != 501, f"{method} {concrete} still stubbed"
        if resp.status_code >= 400:
            assert resp.json()["error"]["code"] != "not_implemented", f"{method} {concrete}"


def test_jobs_endpoints_real_shapes(client) -> None:
    """Deep research is implemented: POST 202 -> JobStatus lifecycle (full pipeline
    behavior is covered in test_research.py; this asserts the contract shapes)."""
    created = client.post("/v1/jobs", json={"type": "deep_research", "query": "shape check"})
    assert created.status_code == 202
    assert set(created.json()) == {"id", "state"}
    job_id = created.json()["id"]
    assert job_id.startswith("job_")
    assert created.json()["state"] == "queued"

    listing = client.get("/v1/jobs").json()
    assert set(listing) == {"jobs"}
    status = client.get(f"/v1/jobs/{job_id}").json()
    assert set(status) == {
        "id",
        "type",
        "query",
        "state",
        "stage",
        "percent",
        "created_at",
        "started_at",
        "finished_at",
        "error",
    }
    assert status["state"] in ("queued", "running", "completed", "failed", "cancelled")

    assert client.post("/v1/jobs", json={"type": "espresso", "query": "x"}).status_code == 400
    assert client.get("/v1/jobs/job_nope").status_code == 404

    cancelled = client.delete(f"/v1/jobs/{job_id}")
    assert cancelled.status_code == 200
    assert set(cancelled.json()) == {"id", "state"}


def test_chat_completions_real_shape(client) -> None:
    """Chat is implemented now: OpenAI chat.completion + bonsai extension block."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(body["usage"])
    assert set(body["bonsai"]) == {"mode", "effort", "slot", "session_id"}
    assert body["bonsai"]["slot"] == "orchestrator"


def test_memory_endpoints_real_shapes(client) -> None:
    body = client.get("/v1/memory").json()
    assert set(body) == {"entries", "total"}
    created = client.post(
        "/v1/memory",
        json={"layer": "archive", "title": "contract check", "content": "shape test entry"},
    )
    assert created.status_code == 201
    entry = created.json()
    assert set(entry) == {
        "id",
        "layer",
        "title",
        "content",
        "tags",
        "created_at",
        "updated_at",
        "source_session",
    }
    assert entry["id"].startswith("mem_")

    updated = client.put(f"/v1/memory/{entry['id']}", json={"title": "renamed"})
    assert updated.status_code == 200
    assert updated.json()["title"] == "renamed"

    search = client.get("/v1/memory/search", params={"q": "shape test"}).json()
    assert set(search) == {"results"}
    assert search["results"], "the created entry must be findable"
    assert set(search["results"][0]) == {"entry", "score", "snippet"}

    state = client.get("/v1/memory/state").json()
    assert set(state) == {"files"}
    for f in state["files"]:
        assert set(f) == {"name", "content", "bytes", "max_bytes"}
    # identity.md is part of the state payload (api.md §5).
    assert "identity.md" in {f["name"] for f in state["files"]}

    sessions = client.get("/v1/memory/sessions").json()
    assert set(sessions) == {"sessions"}
    missing = client.get("/v1/memory/sessions/nope")
    assert missing.status_code == 404

    deleted = client.delete(f"/v1/memory/{entry['id']}")
    assert deleted.status_code == 204


def test_memory_state_put_real_shapes(client) -> None:
    """PUT /v1/memory/state/{name} (additive 2026-07-21b): update-only, byte-capped,
    identity.md included; edits show up in GET and in FTS recall."""
    updated = client.put(
        "/v1/memory/state/identity.md", json={"content": "# identity\n\nI prefer terse replies."}
    )
    assert updated.status_code == 200
    body = updated.json()
    assert set(body) == {"name", "content", "bytes", "max_bytes"}
    assert body["name"] == "identity.md"
    assert body["bytes"] == len(body["content"].encode())

    files = {f["name"]: f for f in client.get("/v1/memory/state").json()["files"]}
    assert files["identity.md"]["content"] == "# identity\n\nI prefer terse replies."

    # The edit is searchable immediately (the identity mirror re-indexed).
    hits = client.get("/v1/memory/search", params={"q": "terse replies"}).json()["results"]
    assert any(h["entry"]["layer"] == "identity" for h in hits)

    # Unknown names are 404 — new files are not creatable through this route.
    missing = client.put("/v1/memory/state/new-file.md", json={"content": "x"})
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "state_file_unknown"

    # Oversized content is a 400 with the contract code.
    max_bytes = files["identity.md"]["max_bytes"]
    too_big = client.put("/v1/memory/state/identity.md", json={"content": "x" * (max_bytes + 1)})
    assert too_big.status_code == 400
    assert too_big.json()["error"]["code"] == "state_file_too_large"


def test_memory_sessions_shape_includes_project_id(client) -> None:
    """Sessions list rows carry project_id (additive 2026-07-21b) — null when the
    session is not bound to a project."""
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "shape check"}],
            "session_id": "sess-shape",
        },
    )
    sessions = client.get("/v1/memory/sessions").json()["sessions"]
    assert sessions
    assert set(sessions[0]) == {
        "id",
        "title",
        "mode",
        "project_id",
        "started_at",
        "ended_at",
        "message_count",
    }
    assert sessions[0]["project_id"] is None


def test_projects_endpoints_real_shapes(client) -> None:
    created = client.post("/v1/projects", json={"name": "contract", "description": "d"})
    assert created.status_code == 201
    project = created.json()
    assert set(project) == {
        "id",
        "name",
        "description",
        "created_at",
        "session_count",
        "doc_count",
    }
    assert project["id"].startswith("proj_")
    assert project["session_count"] == 0 and project["doc_count"] == 0

    listing = client.get("/v1/projects").json()
    assert set(listing) == {"projects"}
    got = client.get(f"/v1/projects/{project['id']}")
    assert got.status_code == 200
    patched = client.patch(f"/v1/projects/{project['id']}", json={"name": "renamed"})
    assert patched.status_code == 200
    assert patched.json()["name"] == "renamed"

    doc = client.post(
        f"/v1/projects/{project['id']}/docs", json={"title": "notes", "content": "text body"}
    )
    assert doc.status_code == 201
    assert set(doc.json()) == {"id", "title", "bytes", "created_at", "content"}
    doc_id = doc.json()["id"]
    docs = client.get(f"/v1/projects/{project['id']}/docs").json()
    assert set(docs) == {"docs"}
    assert set(docs["docs"][0]) == {"id", "title", "bytes", "created_at"}  # no content in list
    single = client.get(f"/v1/projects/{project['id']}/docs/{doc_id}").json()
    assert single["content"] == "text body"

    assert client.delete(f"/v1/projects/{project['id']}/docs/{doc_id}").status_code == 204
    assert client.delete(f"/v1/projects/{project['id']}").status_code == 204
    missing = client.get(f"/v1/projects/{project['id']}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "project_not_found"


def test_schedules_endpoints_real_shapes(client) -> None:
    created = client.post(
        "/v1/schedules",
        json={
            "name": "daily digest",
            "prompt": "summarize the day",
            "cadence": {"kind": "daily", "time": "07:30"},
        },
    )
    assert created.status_code == 201
    schedule = created.json()
    assert set(schedule) == {
        "id",
        "name",
        "prompt",
        "mode",
        "effort",
        "project_id",
        "cadence",
        "enabled",
        "created_at",
        "last_run_at",
        "next_run_at",
        "last_session_id",
        "last_error",
    }
    assert schedule["id"].startswith("sched_")
    assert schedule["cadence"] == {"kind": "daily", "time": "07:30"}
    assert schedule["enabled"] is True
    assert schedule["next_run_at"] is not None

    listing = client.get("/v1/schedules").json()
    assert set(listing) == {"schedules"}
    assert client.get(f"/v1/schedules/{schedule['id']}").status_code == 200
    patched = client.patch(f"/v1/schedules/{schedule['id']}", json={"enabled": False})
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False

    run = client.post(f"/v1/schedules/{schedule['id']}/run")
    assert run.status_code == 202
    assert set(run.json()) == {"session_id"}

    assert client.delete(f"/v1/schedules/{schedule['id']}").status_code == 204
    missing = client.get(f"/v1/schedules/{schedule['id']}")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "schedule_not_found"


def test_skills_endpoints_real_shapes(client) -> None:
    body = client.get("/v1/skills").json()
    assert set(body) == {"skills"}

    put = client.put(
        "/v1/skills/contract-check",
        json={"content": "---\nname: contract-check\ndescription: shape test\n---\n\n# steps\n"},
    )
    assert put.status_code == 200
    skill = put.json()
    # `verified` is the 2026-07-21 refinement's additive optional field on the
    # Skill object (False until a successful run used the skill).
    assert set(skill) == {
        "name",
        "description",
        "version",
        "updated_at",
        "source",
        "content",
        "verified",
    }
    assert skill["source"] == "human"
    assert skill["version"] == 1
    assert skill["verified"] is False

    got = client.get("/v1/skills/contract-check")
    assert got.status_code == 200
    listing = client.get("/v1/skills").json()["skills"]
    assert all("content" not in s for s in listing)

    assert client.delete("/v1/skills/contract-check").status_code == 204
    assert client.get("/v1/skills/contract-check").status_code == 404


def test_modes_endpoints_real_shapes(client) -> None:
    body = client.get("/v1/modes").json()
    assert set(body) == {"modes"}
    names = {m["name"] for m in body["modes"]}
    assert names == {"chat", "code", "ultra", "deep_research"}
    for mode in body["modes"]:
        assert set(mode) == {
            "name",
            "description",
            "system_prompt_version",
            "tools",
            "default_effort",
            "endpoint",
        }
    single = client.get("/v1/modes/code").json()
    assert single["name"] == "code"
    assert single["endpoint"] == "/v1/chat/completions"
    assert client.get("/v1/modes/nope").status_code == 404


def test_models_shape(client) -> None:
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    # No providers are enabled in the fixture, so no external ("<name>/<model>")
    # entries appear — the external shape is covered in test_providers.py.
    assert ids == {"bonsai-27b", "bonsai-8b", "bonsai-4b", "bonsai-1.7b"}
    for model in body["data"]:
        assert model["object"] == "model"
        assert model["owned_by"] == "prism-ml"
        bonsai = model["bonsai"]
        assert set(bonsai) == {
            "family",
            "quant",
            "role",
            "resident",
            "ctx",
            "vision",
            "downloaded_families",
        }
        assert bonsai["role"] in ("orchestrator", "worker", "utility", "none")


def test_system_shape(client) -> None:
    body = client.get("/v1/system").json()
    assert set(body) == {
        "version",
        "uptime_s",
        "gpus",
        "telemetry_source",
        "loadout",
        "capabilities",
        "kv",
        "quant_family",
        "dspark",
        "runtime",  # additive 2026-07-22c: lazy/keep-alive residency
        "jobs_active",
        "security",  # additive 2026-07-22: auth/remote_agentic/telegram_paired
        "notices",
    }
    assert set(body["runtime"]) == {"keep_alive", "models_loaded", "state"}
    assert body["runtime"]["state"] in ("cold", "loading", "ready", "idle_unloading")
    assert isinstance(body["runtime"]["models_loaded"], bool)
    assert set(body["security"]) == {"auth_required", "remote_agentic", "telegram_paired"}
    assert body["security"] == {
        "auth_required": False,  # loopback bind (the test client)
        "remote_agentic": False,
        "telegram_paired": False,
    }
    assert set(body["loadout"]) == {"planned_at", "tier", "slots", "headroom_mb"}
    assert set(body["capabilities"]) == {"vision", "browse_t2", "skill_writes", "ultra_parallel"}
    assert set(body["kv"]) == {"k_type", "v_type", "turboquant"}
    assert set(body["kv"]["turboquant"]) == {
        "enabled",
        "preset",
        "backend_supported",
        "fallback_active",
        "fallback_reason",
    }
    assert set(body["quant_family"]) == {"configured", "effective", "degraded", "reason"}
    assert set(body["dspark"]) == {"enabled", "available"}
    for notice in body["notices"]:
        assert set(notice) == {"level", "code", "message"}


def test_system_budget_shape(client) -> None:
    body = client.get("/v1/system/budget").json()
    assert set(body) == {"measured", "rows"}
    assert isinstance(body["measured"], bool)
    assert len(body["rows"]) == 4


def test_system_health_shape(client) -> None:
    resp = client.get("/v1/system/health")
    assert resp.status_code == 200  # 200 always; status carries the truth
    body = resp.json()
    assert set(body) == {"status", "checks"}
    assert body["status"] in ("ok", "starting", "degraded")
    assert set(body["checks"]) == {"binary", "models", "telemetry", "slots_ready", "slots_total"}


def test_settings_shape(client) -> None:
    body = client.get("/v1/settings").json()
    assert set(body) == {"current", "staged"}
    assert body["staged"] is None
    # Residency, SLAP and the connector-catalog reference are all part of the settings
    # shape (api.md 2026-07-22c).
    assert {"runtime", "slap", "mcp_connectors"} <= set(body["current"])
    assert body["current"]["mcp_connectors"] == []  # none referenced by default
    assert body["current"]["kv"] == {"turboquant_enabled": True, "preset": "recommended"}
    # Gateways carry only the *_set secret indicators (api.md §8, whatsapp additive);
    # telegram also carries its inbound-auth fields (api.md 2026-07-22 security).
    assert body["current"]["gateways"]["telegram"] == {
        "enabled": False,
        "token_set": False,
        "allowed_chat_ids": [],
        "require_pairing": True,
        "rate_limit_per_min": 20,
    }
    # WhatsApp is QR-linked now (api.md 2026-07-22b): no secret, just link state.
    assert body["current"]["gateways"]["whatsapp"] == {
        "enabled": False,
        "linked": False,
        "to_number": "",
    }
    # Providers + search (additive 2026-07-21c): api_key is write-only -> api_key_set.
    assert body["current"]["providers"] == [
        {
            "name": "ollama",
            "kind": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "enabled": False,
            "api_key_set": False,
        }
    ]
    assert body["current"]["search"] == {
        "provider": "duckduckgo",
        "base_url": "",
        "api_key_set": False,
    }


def test_mcp_connectors_shape(client) -> None:
    """GET /v1/mcp/connectors (api.md 2026-07-22c): every catalog entry carries EXACTLY
    the documented fields, and `enabled` (the authority clients render) reflects
    settings.mcp_connectors — none enabled in the fixture."""
    connectors = client.get("/v1/mcp/connectors").json()["connectors"]
    assert connectors  # the curated catalog is non-empty
    for connector in connectors:
        assert set(connector) == {
            "id",
            "name",
            "description",
            "command",
            "args",
            "requires_path",
            "enabled",
        }
        assert isinstance(connector["args"], list)
        assert isinstance(connector["requires_path"], bool)
    assert all(connector["enabled"] is False for connector in connectors)


def test_system_search_test_shape(client) -> None:
    """POST /v1/system/search_test never throws (api.md §11): with the offline test
    transport the duckduckgo default fails, and that failure is REPORTED, not
    raised."""
    resp = client.post("/v1/system/search_test", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"ok", "provider", "results", "error"}
    assert body["ok"] is False
    assert body["provider"] == "duckduckgo"
    assert body["results"] == []
    assert isinstance(body["error"], str) and body["error"]
