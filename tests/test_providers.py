"""External providers (api.md §11 + §1 + §2, additive 2026-07-21c): registry polling
incl. unreachable handling, /v1/models external entries, the chat routing matrix
(external_model_mode, model_not_found), envelope passthrough for both stream shapes,
effort-to-sampling-only mapping, client-tool passthrough, and archiving/titling for
external sessions. Every transport is an injected httpx.MockTransport — no network."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from suiban.config import ProviderSettings
from suiban.errors import BonsaiError
from suiban.llama.mock_server import MOCK_SESSION_TITLE
from suiban.providers.registry import EFFORT_TEMPERATURE, ProviderRegistry

OLLAMA_MODELS = ["llama3.2", "qwen3:4b"]
CORP_MODELS = ["gpt-oss"]

EXTERNAL_ANSWER = "Answer from the external provider."


def _models_response(ids: list[str]) -> httpx.Response:
    return httpx.Response(
        200, json={"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}
    )


def _completion_response(model: str, body: dict) -> httpx.Response:
    if body.get("tools") and body.get("tool_choice") == "required":
        message: dict = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-ext-1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": EXTERNAL_ANSWER}
        finish = "stop"
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-ext-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        },
    )


def _stream_response(model: str) -> httpx.Response:
    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": "chatcmpl-ext-1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    sse = (
        chunk({"role": "assistant", "content": ""})
        + chunk({"content": "Answer from "})
        + chunk({"content": "the external provider."})
        + chunk({}, "stop")
        + "data: [DONE]\n\n"
    )
    return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})


@pytest.fixture
def upstream(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    """Two enabled providers behind one MockTransport: 'ollama' (keyless, preset
    base_url) serving OLLAMA_MODELS and 'corp' (openai kind, api_key) serving
    CORP_MODELS. Returns the upstream request log."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        is_corp = request.url.host == "llm.corp.test"
        models = CORP_MODELS if is_corp else OLLAMA_MODELS
        if request.url.path == "/v1/models":
            return _models_response(models)
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            if body.get("stream"):
                return _stream_response(body.get("model", ""))
            return _completion_response(body.get("model", ""), body)
        return httpx.Response(404, json={"error": "no such route"})

    monkeypatch.setattr(
        "suiban.providers.registry._default_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    resp = client.patch(
        "/v1/settings",
        json={
            "providers": [
                {"name": "ollama", "kind": "ollama", "enabled": True},
                {
                    "name": "corp",
                    "kind": "openai",
                    "base_url": "https://llm.corp.test",
                    "enabled": True,
                    "api_key": "sk-corp-secret",
                },
            ]
        },
    )
    assert resp.status_code == 200
    applied = client.post("/v1/system/apply").json()
    assert applied["applied"] is True
    return requests


# -- registry unit ------------------------------------------------------------
def _settings(name: str = "p", **kwargs) -> ProviderSettings:
    return ProviderSettings.model_validate(
        {"name": name, "kind": "openai", "base_url": "http://prov.test", "enabled": True, **kwargs}
    )


async def test_registry_polls_with_bearer_and_caches_models() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _models_response(["m1", "m2"])

    registry = ProviderRegistry(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await registry.refresh([_settings(api_key="sk-x"), _settings(name="off", enabled=False)])
    assert [str(r.url) for r in seen] == ["http://prov.test/v1/models"]  # disabled: skipped
    assert seen[0].headers["Authorization"] == "Bearer sk-x"
    state, model = registry.resolve("p/m2")
    assert (state.name, model, state.reachable) == ("p", "m2", True)
    assert registry.notices() == []


async def test_registry_keyless_provider_sends_no_auth_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _models_response(["m1"])

    registry = ProviderRegistry(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await registry.refresh([_settings()])
    assert "authorization" not in {k.lower() for k in seen[0].headers}


async def test_registry_unreachable_marks_state_keeps_cache_and_notices() -> None:
    healthy = True

    def handler(request: httpx.Request) -> httpx.Response:
        if not healthy:
            raise httpx.ConnectError("connection refused")
        return _models_response(["m1"])

    registry = ProviderRegistry(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await registry.refresh([_settings()])
    assert registry.states[0].reachable is True

    healthy = False
    await registry.refresh([_settings()])  # never raises
    state = registry.states[0]
    assert state.reachable is False
    assert state.models == ["m1"]  # last known list survives the outage
    assert [n.code for n in registry.notices()] == ["provider_unreachable"]
    # Entries stay listed, resident:false (api.md §2).
    entries = registry.model_entries()
    assert entries[0]["id"] == "p/m1"
    assert entries[0]["bonsai"]["resident"] is False
    # The cached model still routes (the outage may be transient)...
    registry.resolve("p/m1")
    # ...but unknown models mention the outage in the 404.
    with pytest.raises(BonsaiError) as err:
        registry.resolve("p/other")
    assert err.value.code == "model_not_found"
    assert "unreachable" in err.value.message


async def test_registry_resolve_unknown_provider_is_404() -> None:
    registry = ProviderRegistry()
    with pytest.raises(BonsaiError) as err:
        registry.resolve("ghost/model")
    assert err.value.status == 404
    assert err.value.code == "model_not_found"


def test_ollama_kind_presets_base_url() -> None:
    assert (
        ProviderSettings.model_validate({"name": "o", "kind": "ollama"}).base_url
        == "http://127.0.0.1:11434"
    )
    explicit = ProviderSettings.model_validate(
        {"name": "o", "kind": "ollama", "base_url": "http://10.0.0.5:11434"}
    )
    assert explicit.base_url == "http://10.0.0.5:11434"


# -- /v1/models surface -------------------------------------------------------
def test_models_appends_external_entries(client: TestClient, upstream) -> None:
    body = client.get("/v1/models").json()
    by_id = {m["id"]: m for m in body["data"]}
    # The bonsai family is untouched and first.
    assert [m["id"] for m in body["data"][:4]] == [
        "bonsai-27b",
        "bonsai-8b",
        "bonsai-4b",
        "bonsai-1.7b",
    ]
    entry = by_id["ollama/llama3.2"]
    assert entry["object"] == "model"
    assert entry["owned_by"] == "ollama"
    bonsai = entry["bonsai"]
    assert bonsai["external"] is True
    assert bonsai["provider"] == "ollama"
    assert bonsai["role"] == "none"
    assert bonsai["resident"] is True
    assert bonsai["family"] is None and bonsai["quant"] is None and bonsai["ctx"] is None
    assert "corp/gpt-oss" in by_id
    # The ollama poll went to the preset base_url (keyless), corp with its Bearer.
    polls = {str(r.url): r for r in upstream if r.url.path == "/v1/models"}
    assert "http://127.0.0.1:11434/v1/models" in polls
    corp_poll = polls["https://llm.corp.test/v1/models"]
    assert corp_poll.headers["Authorization"] == "Bearer sk-corp-secret"


# -- chat routing matrix ------------------------------------------------------
def _chat_body(model: str, **kwargs) -> dict:
    return {"model": model, "messages": [{"role": "user", "content": "hello"}], **kwargs}


def test_external_model_requires_chat_mode(client: TestClient, upstream) -> None:
    for mode in ("code", "ultra"):
        resp = client.post("/v1/chat/completions", json=_chat_body("ollama/llama3.2", mode=mode))
        assert resp.status_code == 400, mode
        assert resp.json()["error"]["code"] == "external_model_mode"


def test_unknown_provider_or_model_is_404(client: TestClient, upstream) -> None:
    for model in ("ghost/llama3.2", "ollama/ghost-model"):
        resp = client.post("/v1/chat/completions", json=_chat_body(model))
        assert resp.status_code == 404, model
        assert resp.json()["error"]["code"] == "model_not_found"


def test_slash_model_without_any_provider_is_404(client: TestClient) -> None:
    resp = client.post("/v1/chat/completions", json=_chat_body("ollama/llama3.2"))
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "model_not_found"


def test_external_non_streaming_envelope_and_upstream_payload(client: TestClient, upstream) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json=_chat_body("ollama/llama3.2", session_id="sess-ext-1", effort="low"),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == EXTERNAL_ANSWER
    assert body["bonsai"] == {
        "mode": "chat",
        "effort": "low",
        "slot": "ollama",
        "session_id": "sess-ext-1",
    }

    sent = json.loads(next(r for r in upstream if r.url.path == "/v1/chat/completions").content)
    # The provider sees the BARE model id and a plain OpenAI request: effort maps to
    # sampling only — no chat_template_kwargs, no bonsai fields (api.md §1).
    assert sent["model"] == "llama3.2"
    assert sent["temperature"] == EFFORT_TEMPERATURE["low"]
    for forbidden in ("chat_template_kwargs", "mode", "effort", "session_id", "stream_events"):
        assert forbidden not in sent, forbidden


def test_external_explicit_temperature_beats_the_effort_ladder(
    client: TestClient, upstream
) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json=_chat_body("ollama/llama3.2", effort="max", temperature=0.05),
    )
    assert resp.status_code == 200
    sent = json.loads(next(r for r in upstream if r.url.path == "/v1/chat/completions").content)
    assert sent["temperature"] == 0.05


def test_external_default_stream_proxies_openai_chunks_and_archives(
    client: TestClient, upstream
) -> None:
    session_id = "sess-ext-stream"
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_chat_body("ollama/llama3.2", stream=True, session_id=session_id),
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        raw = resp.read().decode()

    lines = [line for line in raw.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
    assert text == EXTERNAL_ANSWER
    # The streamed exchange landed in the session archive.
    transcript = client.get(f"/v1/memory/sessions/{session_id}").json()
    roles = [m["role"] for m in transcript["messages"]]
    assert roles == ["user", "assistant"]
    assert transcript["messages"][1]["content"] == EXTERNAL_ANSWER


def test_external_stream_events_envelope(client: TestClient, upstream) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_chat_body("ollama/llama3.2", stream=True, stream_events=True),
    ) as resp:
        assert resp.status_code == 200
        raw = resp.read().decode()
    lines = [line[len("data: ") :] for line in raw.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    events = [json.loads(line) for line in lines[:-1]]
    types = [e["type"] for e in events]
    assert types == ["delta", "usage", "done"]
    assert events[0]["text"] == EXTERNAL_ANSWER
    assert events[-1]["finish_reason"] == "stop"


def test_external_client_tools_pass_through(client: TestClient, upstream) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    resp = client.post(
        "/v1/chat/completions",
        json=_chat_body("ollama/llama3.2", tools=tools, tool_choice="required"),
    )
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    sent = json.loads(next(r for r in upstream if r.url.path == "/v1/chat/completions").content)
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "required"


def test_external_session_archives_and_titles_on_our_utility_slot(
    client: TestClient, upstream
) -> None:
    # Lazy residency (api.md 2026-07-22c): an external chat never warms local models, so
    # auto-titling — a local utility-slot op — only runs when the loadout is already
    # resident. Warm it once with a local chat, then the external session titles on our
    # utility slot exactly as before.
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "warm"}]},
    )
    session_id = "sess-ext-title"
    resp = client.post(
        "/v1/chat/completions", json=_chat_body("ollama/llama3.2", session_id=session_id)
    )
    assert resp.status_code == 200
    transcript = client.get(f"/v1/memory/sessions/{session_id}").json()
    assert [m["role"] for m in transcript["messages"]] == ["user", "assistant"]
    assert transcript["messages"][1]["content"] == EXTERNAL_ANSWER
    # Auto-titling runs on OUR utility slot (the mock backend), not the provider.
    deadline = time.monotonic() + 10
    title = None
    while time.monotonic() < deadline:
        title = client.get(f"/v1/memory/sessions/{session_id}").json()["session"]["title"]
        if title:
            break
        time.sleep(0.02)
    assert title == MOCK_SESSION_TITLE


def test_external_project_binding_and_doc_injection(client: TestClient, upstream) -> None:
    project_id = client.post("/v1/projects", json={"name": "ext-proj"}).json()["id"]
    client.post(
        f"/v1/projects/{project_id}/docs",
        json={"title": "zeppelin status", "content": "the zeppelin refactor is half done"},
    )
    resp = client.post(
        "/v1/chat/completions",
        json=_chat_body(
            "ollama/llama3.2",
            session_id="sess-ext-proj",
            project_id=project_id,
            **{"messages": [{"role": "user", "content": "how is the zeppelin refactor going?"}]},
        ),
    )
    assert resp.status_code == 200
    sent = json.loads(next(r for r in upstream if r.url.path == "/v1/chat/completions").content)
    injected = [
        m["content"]
        for m in sent["messages"]
        if m["role"] == "system" and "<<<doc: zeppelin status>>>" in str(m.get("content"))
    ]
    assert injected, "project excerpts must be injected into the upstream request"
    # Unknown project still 404s before any upstream call.
    bad = client.post(
        "/v1/chat/completions", json=_chat_body("ollama/llama3.2", project_id="proj_nope")
    )
    assert bad.status_code == 404
    assert bad.json()["error"]["code"] == "project_not_found"
    # The session lists under the project.
    sessions = client.get("/v1/memory/sessions", params={"project_id": project_id}).json()
    assert [s["id"] for s in sessions["sessions"]] == ["sess-ext-proj"]


def test_provider_unreachable_surfaces_notice_in_system(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(
        "suiban.providers.registry._default_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client.patch(
        "/v1/settings",
        json={"providers": [{"name": "ollama", "kind": "ollama", "enabled": True}]},
    )
    assert client.post("/v1/system/apply").json()["applied"] is True
    notices = client.get("/v1/system").json()["notices"]
    assert any(n["code"] == "provider_unreachable" for n in notices)
    # Nothing to list, and chat to it is an honest 404 — never a crash.
    ids = {m["id"] for m in client.get("/v1/models").json()["data"]}
    assert not any("/" in i for i in ids)
    resp = client.post("/v1/chat/completions", json=_chat_body("ollama/llama3.2"))
    assert resp.status_code == 404


def test_suiban_default_routing_untouched(client: TestClient, upstream) -> None:
    """bonsai-auto still routes to the local orchestrator with providers enabled."""
    resp = client.post("/v1/chat/completions", json=_chat_body("bonsai-auto"))
    assert resp.status_code == 200
    assert resp.json()["bonsai"]["slot"] == "orchestrator"


def test_external_sessions_never_schedule_reflection(
    client: TestClient, upstream, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-task reflection is an orchestrator capability — external models never
    trigger it (api.md §11 notes)."""
    from suiban.memory import reflection

    scheduled: list[str] = []
    monkeypatch.setattr(
        reflection,
        "schedule_reflection",
        lambda *args, **kwargs: scheduled.append(kwargs.get("session_id", "?")),
    )
    resp = client.post(
        "/v1/chat/completions", json=_chat_body("ollama/llama3.2", session_id="sess-ext-refl")
    )
    assert resp.status_code == 200
    assert scheduled == []
