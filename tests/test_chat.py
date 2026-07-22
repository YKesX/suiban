"""POST /v1/chat/completions against the mock seam: validation, routing, both stream
envelopes, pass-through tool calls, session recording, the code-mode workdir,
client-disconnect abort, per-slot serialization, and the effort-default chain."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect
from starlette.requests import Request as StarletteRequest

from suiban.agent.loop import BackendChat
from suiban.errors import BonsaiError
from suiban.llama.manager import SlotGate
from suiban.llama.mock_server import MOCK_COMPLETION_TEXT
from suiban.routers.chat import (
    MEMORY_CONTEXT_HEADER,
    await_watching_disconnect,
    validate_workdir,
)

TINY_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="


def _sse_payloads(text: str) -> list:
    """Parse SSE data lines; the [DONE] sentinel is returned as the string 'DONE'."""
    out = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        out.append("DONE" if payload == "[DONE]" else json.loads(payload))
    return out


def _warm_loadout(client: TestClient) -> None:
    """Lazy residency (api.md 2026-07-22c): warm the planned slots with one trivial chat
    so a test can grab a slot gate / patch the backend on a resident loadout."""
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "warm"}]},
    )
    assert resp.status_code == 200


def _stream_events(client: TestClient, content: str = "hi") -> list:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": content}],
            "stream": True,
            "stream_events": True,
        },
    ) as resp:
        assert resp.status_code == 200
        return _sse_payloads(resp.read().decode())


def _warming_codes(payloads: list) -> list:
    return [
        p["code"]
        for p in payloads
        if p != "DONE" and p.get("type") == "notice" and p.get("code") == "warming_up"
    ]


def test_cold_start_rich_stream_emits_warming_up_then_warm_does_not(client: TestClient) -> None:
    """A cold start leads the rich stream with a warming_up notice (api.md 2026-07-22c);
    once the loadout is resident, later requests carry no such notice."""
    first = _stream_events(client)
    assert _warming_codes(first) == ["warming_up"]
    second = _stream_events(client)
    assert _warming_codes(second) == []


# -- validation --------------------------------------------------------------
def test_missing_model_is_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_empty_messages_is_400(client: TestClient) -> None:
    resp = client.post("/v1/chat/completions", json={"model": "bonsai-auto", "messages": []})
    assert resp.status_code == 400


def test_bad_mode_effort_and_deep_research_are_400(client: TestClient) -> None:
    base = {"model": "bonsai-auto", "messages": [{"role": "user", "content": "x"}]}
    assert client.post("/v1/chat/completions", json={**base, "mode": "zen"}).status_code == 400
    assert client.post("/v1/chat/completions", json={**base, "effort": "ultra"}).status_code == 400
    resp = client.post("/v1/chat/completions", json={**base, "mode": "deep_research"})
    assert resp.status_code == 400
    assert "/v1/jobs" in resp.json()["error"]["message"]


def test_unknown_openai_fields_are_tolerated(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "x"}],
            "presence_penalty": 0.5,
            "n": 1,
            "user": "abc",
        },
    )
    assert resp.status_code == 200


# -- routing -----------------------------------------------------------------
def test_non_resident_model_is_409(client: TestClient) -> None:
    # The 24 GB fixture loadout is 27b + 4b utility + 2x8b workers; 1.7b is installed
    # upstream but not resident.
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-1.7b", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 409
    body = resp.json()["error"]
    assert body["type"] == "conflict_error"
    assert body["code"] == "model_not_resident"


def test_unknown_model_is_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_model"


def test_resident_worker_model_is_routable(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-8b", "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["bonsai"]["slot"].startswith("worker")


# -- images ------------------------------------------------------------------
def _image_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": {"url": TINY_PNG}},
            ],
        }
    ]


def test_images_route_to_27b_orchestrator(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions", json={"model": "bonsai-auto", "messages": _image_messages()}
    )
    assert resp.status_code == 200
    assert resp.json()["bonsai"]["slot"] == "orchestrator"


def test_images_to_non_vision_model_is_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions", json={"model": "bonsai-8b", "messages": _image_messages()}
    )
    assert resp.status_code == 400
    body = resp.json()["error"]
    assert body["code"] == "vision_unavailable"
    assert "27B" in body["message"]


# -- non-streaming -----------------------------------------------------------
def test_non_streaming_openai_shape_and_bonsai_ext(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hello"}],
            "mode": "code",
            "effort": "low",
            "session_id": "sess-ext",
        },
    )
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["choices"][0]["message"]["content"] == MOCK_COMPLETION_TEXT
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    assert body["bonsai"] == {
        "mode": "code",
        "effort": "low",
        "slot": "orchestrator",
        "session_id": "sess-ext",
    }


# -- default stream envelope (byte-compatible OpenAI) ------------------------
def test_default_stream_is_openai_chunks(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        payloads = _sse_payloads(resp.read().decode())

    assert payloads[-1] == "DONE"
    chunks = payloads[:-1]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert MOCK_COMPLETION_TEXT in text
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert "usage" in chunks[-1]


# -- rich stream envelope ----------------------------------------------------
def test_stream_events_envelope(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_events": True,
        },
    ) as resp:
        assert resp.status_code == 200
        payloads = _sse_payloads(resp.read().decode())

    assert payloads[-1] == "DONE"
    events = payloads[:-1]
    types = [e["type"] for e in events]
    assert types[-1] == "done"
    assert "delta" in types
    assert "usage" in types
    known = {
        "delta",
        "thinking_status",
        "tool_call",
        "tool_result",
        "plan",
        "agent_spawn",
        "agent_result",
        "compression",
        "notice",
        "usage",
        "done",
        "error",
    }
    assert set(types) <= known
    done = events[-1]
    assert done["finish_reason"] in ("stop", "length", "tool_calls", "cancelled", "error")
    usage = next(e for e in events if e["type"] == "usage")
    assert set(usage) == {"type", "prompt_tokens", "completion_tokens", "thinking_tokens"}


# -- pass-through (client tools) --------------------------------------------
CLIENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]


def test_client_tools_pass_through_non_streaming(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": CLIENT_TOOLS,
            "tool_choice": "required",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    calls = choice["message"]["tool_calls"]
    assert calls[0]["function"]["name"] == "get_weather"
    assert body["bonsai"]["slot"] == "orchestrator"


def test_client_tools_stream_delta_tool_calls(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "weather in tokyo?"}],
            "tools": CLIENT_TOOLS,
            "tool_choice": "required",
            "stream": True,
        },
    ) as resp:
        assert resp.status_code == 200
        payloads = _sse_payloads(resp.read().decode())

    assert payloads[-1] == "DONE"
    chunks = payloads[:-1]
    tool_deltas = [c for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
    assert tool_deltas, "OpenAI stream must carry delta.tool_calls"
    name = tool_deltas[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
    assert name == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# -- session recording -------------------------------------------------------
def test_session_recorded_into_archive(client: TestClient) -> None:
    session_id = "sess-record-1"
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "remember the magic word xyzzy"}],
            "session_id": session_id,
        },
    )
    transcript = client.get(f"/v1/memory/sessions/{session_id}").json()
    roles = [m["role"] for m in transcript["messages"]]
    assert roles == ["user", "assistant"]
    assert transcript["messages"][0]["content"] == "remember the magic word xyzzy"
    assert transcript["session"]["mode"] == "chat"

    # Continue the session OpenAI-style (full history resent): only the tail is new.
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [
                {"role": "user", "content": "remember the magic word xyzzy"},
                {"role": "assistant", "content": MOCK_COMPLETION_TEXT},
                {"role": "user", "content": "what was the magic word?"},
            ],
            "session_id": session_id,
        },
    )
    transcript = client.get(f"/v1/memory/sessions/{session_id}").json()
    contents = [m["content"] for m in transcript["messages"]]
    assert contents.count("remember the magic word xyzzy") == 1  # not duplicated
    assert "what was the magic word?" in contents

    # ... and the transcript is searchable.
    found = client.get("/v1/memory/sessions", params={"q": "xyzzy"}).json()["sessions"]
    assert [s["id"] for s in found] == [session_id]


def test_no_session_id_records_nothing(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "ephemeral"}]},
    )
    sessions = client.get("/v1/memory/sessions").json()["sessions"]
    assert sessions == []


# -- code-mode workdir (api.md §1 `workdir`, additive 2026-07-21b) ------------
def _code_body(workdir: str | None = None, session_id: str | None = None) -> dict:
    body: dict = {
        "model": "bonsai-auto",
        "messages": [{"role": "user", "content": "fix the tests"}],
        "mode": "code",
    }
    if workdir is not None:
        body["workdir"] = workdir
    if session_id is not None:
        body["session_id"] = session_id
    return body


def test_workdir_requires_code_mode(client: TestClient, tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    body = _code_body(str(project))
    body["mode"] = "chat"
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "workdir_invalid"
    assert "code" in resp.json()["error"]["message"]


def test_workdir_validation_matrix(client: TestClient, tmp_path: Path, bonsai_home: Path) -> None:
    a_file = tmp_path / "file.txt"
    a_file.write_text("not a dir")
    inside_home = bonsai_home / "work" / "sneaky"
    inside_home.mkdir(parents=True)
    for bad, reason_bit in [
        ("relative/dir", "absolute"),
        (str(tmp_path / "missing"), "does not exist"),
        (str(a_file), "not a directory"),
        (str(inside_home), "off limits"),
        (str(bonsai_home), "off limits"),
    ]:
        resp = client.post("/v1/chat/completions", json=_code_body(bad))
        assert resp.status_code == 400, bad
        error = resp.json()["error"]
        assert error["code"] == "workdir_invalid", bad
        assert reason_bit in error["message"], bad


def test_validate_workdir_rejects_symlink_into_home(tmp_path: Path) -> None:
    """Symlinks are resolved BEFORE the home check — a link cannot smuggle the jail
    into ~/.bonsai."""
    home = tmp_path / "home"
    (home / "inner").mkdir(parents=True)
    link = tmp_path / "innocent"
    link.symlink_to(home / "inner")
    with pytest.raises(BonsaiError) as err:
        validate_workdir(home, str(link))
    assert err.value.code == "workdir_invalid"
    # ... and a symlink to a legitimate directory resolves to its target.
    real = tmp_path / "real"
    real.mkdir()
    ok_link = tmp_path / "ok"
    ok_link.symlink_to(real)
    assert validate_workdir(home, str(ok_link)) == real.resolve()


def test_workdir_is_used_and_remembered_on_the_session(
    client: TestClient, tmp_path: Path, bonsai_home: Path
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    session_id = "sess-workdir-1"

    resp = client.post("/v1/chat/completions", json=_code_body(str(project), session_id))
    assert resp.status_code == 200
    store = client.app.state.bonsai.memory.store
    assert store.session_workdir(session_id) == str(project.resolve())
    # The custom jail replaced the default one — nothing created under ~/.bonsai/work.
    assert not (bonsai_home / "work" / session_id).exists()

    # Continuation WITHOUT the field: the session remembers its workdir.
    resp = client.post("/v1/chat/completions", json=_code_body(session_id=session_id))
    assert resp.status_code == 200
    assert not (bonsai_home / "work" / session_id).exists()
    assert store.session_workdir(session_id) == str(project.resolve())


def test_workdir_remembered_but_gone_is_400(
    client: TestClient, tmp_path: Path, bonsai_home: Path
) -> None:
    project = tmp_path / "vanishing"
    project.mkdir()
    session_id = "sess-workdir-2"
    assert (
        client.post("/v1/chat/completions", json=_code_body(str(project), session_id)).status_code
        == 200
    )

    project.rmdir()
    resp = client.post("/v1/chat/completions", json=_code_body(session_id=session_id))
    assert resp.status_code == 400
    error = resp.json()["error"]
    assert error["code"] == "workdir_invalid"
    assert "repoint" in error["message"]  # the fix is actionable: pass a new workdir

    # Repointing works and is remembered.
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    resp = client.post("/v1/chat/completions", json=_code_body(str(replacement), session_id))
    assert resp.status_code == 200
    store = client.app.state.bonsai.memory.store
    assert store.session_workdir(session_id) == str(replacement.resolve())


def test_default_sessions_still_get_the_per_session_jail(
    client: TestClient, bonsai_home: Path
) -> None:
    session_id = "sess-default-jail"
    resp = client.post("/v1/chat/completions", json=_code_body(session_id=session_id))
    assert resp.status_code == 200
    assert (bonsai_home / "work" / session_id).is_dir()
    assert client.app.state.bonsai.memory.store.session_workdir(session_id) is None


# -- automatic memory injection (api.md §11 notes, additive 2026-07-21c) ------
def _spy_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    payloads: list[dict] = []
    original = BackendChat.complete

    async def spy(self, payload: dict, timeout: float) -> dict:
        payloads.append(payload)
        return await original(self, payload, timeout)

    monkeypatch.setattr(BackendChat, "complete", spy)
    return payloads


def _memory_blocks(payloads: list[dict]) -> list[str]:
    return [
        m["content"]
        for p in payloads
        for m in p["messages"]
        if isinstance(m.get("content"), str) and MEMORY_CONTEXT_HEADER in m["content"]
    ]


def test_matching_memory_hits_are_injected_in_chat_and_code(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(
        "/v1/memory",
        json={
            "layer": "archive",
            "title": "favorite color",
            "content": "the user's favorite color is vermilion",
        },
    )
    payloads = _spy_payloads(monkeypatch)
    for mode, session in (("chat", "sess-mem-chat"), ("code", "sess-mem-code")):
        payloads.clear()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "remind me about vermilion"}],
                "mode": mode,
                "effort": "low",
                "session_id": session,
            },
        )
        assert resp.status_code == 200
        injected = _memory_blocks(payloads)
        assert injected, f"memory hits must be injected in mode {mode}"
        assert "<<<memory mem_" in injected[0]
        assert "vermilion" in injected[0]
        assert "<<<end memory>>>" in injected[0]
        # The injected block is a system message — never archived.
        transcript = client.get(f"/v1/memory/sessions/{session}").json()
        assert all(MEMORY_CONTEXT_HEADER not in m["content"] for m in transcript["messages"])


def test_no_matching_memory_injects_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _spy_payloads(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "xylophonic quandary"}],
            "session_id": "sess-mem-none",
        },
    )
    assert resp.status_code == 200
    assert _memory_blocks(payloads) == []


def test_ultra_mode_gets_no_automatic_memory_injection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    client.post(
        "/v1/memory",
        json={"layer": "archive", "title": "t", "content": "ultramarine pigment history"},
    )
    payloads = _spy_payloads(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "tell me about ultramarine pigment"}],
            "mode": "ultra",
            "effort": "low",
        },
    )
    assert resp.status_code == 200
    assert _memory_blocks(payloads) == []


# -- client identity overlays (api.md 2026-07-22b) ---------------------------
def _leading_system(payloads: list[dict]) -> str:
    for p in payloads:
        messages = p["messages"]
        if messages and messages[0].get("role") == "system":
            return messages[0]["content"]
    return ""


def test_client_identity_overlay_injected(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _spy_payloads(monkeypatch)
    for header, marker in (
        ("sentei", "In the terminal (sentei)"),
        ("dai", "In the desktop app (dai)"),
    ):
        payloads.clear()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "hi"}],
                "effort": "low",
                "session_id": f"id-{header}",
            },
            headers={"X-Bonsai-Client": header},
        )
        assert resp.status_code == 200
        system = _leading_system(payloads)
        assert "honest to a fault" in system  # base identity.md merged in
        assert marker in system  # the matching client overlay merged on top


def test_unknown_client_gets_base_identity_only(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = _spy_payloads(monkeypatch)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "effort": "low",
            "session_id": "id-other",
        },
        headers={"X-Bonsai-Client": "other"},
    )
    assert resp.status_code == 200
    system = _leading_system(payloads)
    assert "honest to a fault" in system  # base persona present
    assert "In the terminal (sentei)" not in system  # no overlay for other/unknown
    assert "In the desktop app (dai)" not in system


# -- auto_confirm (api.md 2026-07-22b: code/ultra only) ----------------------
def test_auto_confirm_rejected_in_chat_mode(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "auto_confirm": True,
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "auto_confirm_mode"


@pytest.mark.parametrize("mode", ["code", "ultra"])
def test_auto_confirm_accepted_in_code_and_ultra(client: TestClient, mode: str) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "mode": mode,
            "effort": "low",
            "auto_confirm": True,
        },
    )
    assert resp.status_code == 200


def test_auto_confirm_must_be_boolean(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "mode": "code",
            "auto_confirm": "yes",
        },
    )
    assert resp.status_code == 400


# -- compression end to end --------------------------------------------------
def test_oversized_history_triggers_compression_event(client: TestClient) -> None:
    # Orchestrator ctx is 32768 in the 24 GB fixture; ~24k estimated tokens of history
    # crosses the 70% trigger. The utility slot (mock) produces the rolling summary.
    history = [{"role": "user", "content": f"turn {i} " + "x" * 4000} for i in range(24)]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [*history, {"role": "user", "content": "so, where were we?"}],
            "stream": True,
            "stream_events": True,
        },
    ) as resp:
        assert resp.status_code == 200
        payloads = _sse_payloads(resp.read().decode())

    events = [p for p in payloads if p != "DONE"]
    compressions = [e for e in events if e["type"] == "compression"]
    assert len(compressions) == 1
    assert compressions[0]["trigger_pct"] >= 70
    assert compressions[0]["messages_summarized"] > 0
    # The run still completed normally after compression.
    assert events[-1]["type"] == "done"
    assert any(e["type"] == "delta" for e in events)


def test_auto_compress_off_disables_compression(client: TestClient) -> None:
    """chat.auto_compress=false keeps the full history verbatim — no compression event
    even over the 70% trigger (api.md 2026-07-22b)."""
    assert client.patch("/v1/settings", json={"chat": {"auto_compress": False}}).status_code == 200
    assert client.post("/v1/system/apply").json()["applied"] is True

    history = [{"role": "user", "content": f"turn {i} " + "x" * 4000} for i in range(24)]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [*history, {"role": "user", "content": "so, where were we?"}],
            "stream": True,
            "stream_events": True,
        },
    ) as resp:
        assert resp.status_code == 200
        payloads = _sse_payloads(resp.read().decode())
    events = [p for p in payloads if p != "DONE"]
    assert not [e for e in events if e["type"] == "compression"]
    assert events[-1]["type"] == "done"


# -- system-message coalescing (Bonsai template: ONE system message, FIRST) ---
def test_coalesce_system_messages() -> None:
    from suiban.routers.chat import coalesce_system_messages

    # zero or one-leading system: untouched
    plain = [{"role": "user", "content": "hi"}]
    assert coalesce_system_messages(plain) == plain
    led = [{"role": "system", "content": "a"}, {"role": "user", "content": "hi"}]
    assert coalesce_system_messages(led) == led
    # injected + mode prompt + client system all fold into one leading message
    messy = [
        {"role": "system", "content": "mode prompt"},
        {"role": "system", "content": "memory block"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "late client system"},
    ]
    out = coalesce_system_messages(messy)
    assert [m["role"] for m in out] == ["system", "user"]
    assert out[0]["content"] == "mode prompt\n\nmemory block\n\nlate client system"


# -- client-disconnect abort (deep-detail pass) -------------------------------
def _fake_request(*, disconnected: bool) -> StarletteRequest:
    """A real starlette Request over a scripted ASGI receive channel. The connected
    variant's receive parks forever — is_disconnected()'s pre-cancelled scope
    abandons it and reports False, exactly like a live keep-alive socket."""

    async def receive_disconnect():
        return {"type": "http.disconnect"}

    async def receive_never():
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []}
    return StarletteRequest(scope, receive_disconnect if disconnected else receive_never)


async def test_watching_disconnect_returns_result_when_client_stays() -> None:
    async def fast_backend() -> str:
        await asyncio.sleep(0)
        return "answer"

    assert await await_watching_disconnect(_fake_request(disconnected=False), fast_backend()) == (
        "answer"
    )


async def test_watching_disconnect_cancels_slow_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ranked-gap fix: a gone client CANCELS the backend call (llama-server sees
    the drop) instead of letting the run burn GPU on a dead socket."""
    monkeypatch.setattr("suiban.routers.chat.CLIENT_DISCONNECT_POLL_S", 0.02)
    cancelled = asyncio.Event()

    async def slow_backend() -> str:
        try:
            await asyncio.sleep(60)  # a fake slow llama-server completion
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    with pytest.raises(ClientDisconnect):
        await asyncio.wait_for(
            await_watching_disconnect(_fake_request(disconnected=True), slow_backend()), 5
        )
    assert cancelled.is_set(), "the backend task must be cancelled on disconnect"


async def test_watching_disconnect_propagates_backend_errors() -> None:
    async def failing_backend() -> str:
        raise ValueError("backend boom")

    with pytest.raises(ValueError, match="backend boom"):
        await await_watching_disconnect(_fake_request(disconnected=False), failing_backend())


# -- per-slot serialization (SlotGate) ----------------------------------------
async def test_slot_gate_serializes_and_bounds_queue() -> None:
    gate = SlotGate()
    assert not gate.busy and gate.queue_depth == 0
    await gate.acquire()
    assert gate.busy and gate.queue_depth == 1
    gate.check_capacity("orchestrator")  # holder alone: still room

    waiters = [asyncio.create_task(gate.acquire()) for _ in range(SlotGate.MAX_QUEUE)]
    await asyncio.sleep(0)  # let them join the queue
    assert gate.queue_depth == 1 + SlotGate.MAX_QUEUE
    with pytest.raises(BonsaiError) as excinfo:
        gate.check_capacity("orchestrator")
    assert excinfo.value.status == 429
    assert excinfo.value.code == "slot_queue_full"

    # FIFO drain: each release admits exactly one waiter.
    for expected_left in range(SlotGate.MAX_QUEUE, 0, -1):
        gate.release()
        await asyncio.sleep(0.01)
        assert gate.busy
        assert gate.queue_depth == expected_left
    gate.release()
    assert not gate.busy and gate.queue_depth == 0
    await asyncio.gather(*waiters)


def test_chat_queue_overflow_is_429_overloaded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One running + MAX_QUEUE waiting chats fill the orchestrator gate; the next
    request answers 429 overloaded_error without touching the backend."""
    _warm_loadout(client)  # lazy residency: resident before we patch/grab the gate
    release = asyncio.Event()
    original = BackendChat.complete

    async def blocked(self, payload: dict, timeout: float) -> dict:
        await release.wait()
        return await original(self, payload, timeout)

    monkeypatch.setattr(BackendChat, "complete", blocked)
    gate = client.app.state.bonsai.manager.slot("orchestrator").gate

    def post_chat() -> int:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "queue me"}],
                "effort": "low",
            },
        )
        return resp.status_code

    def wait_for(predicate, deadline_s: float = 5.0) -> None:
        import time

        end = time.monotonic() + deadline_s
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition never became true")

    threads = []
    results: list[int] = []

    def run_and_record() -> None:
        results.append(post_chat())

    # 1 holder + MAX_QUEUE waiters, joined deterministically via the gate itself.
    for expected_depth in range(1, SlotGate.MAX_QUEUE + 2):
        thread = threading.Thread(target=run_and_record)
        thread.start()
        threads.append(thread)
        wait_for(lambda d=expected_depth: gate.queue_depth >= d)

    overflow = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "one too many"}],
            "effort": "low",
        },
    )
    assert overflow.status_code == 429
    body = overflow.json()["error"]
    assert body["type"] == "overloaded_error"
    assert body["code"] == "slot_queue_full"

    # Unblock: the queued runs all complete (serialized, not lost).
    client.portal.call(release.set)
    for thread in threads:
        thread.join(timeout=10)
    assert results == [200] * (SlotGate.MAX_QUEUE + 1)
    assert not gate.busy


def test_stream_events_waiting_chat_gets_queue_notice(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream_events chat that actually waits (> the notice threshold) sees ONE
    'queued behind N' notice before its run starts; the holder sees none."""
    _warm_loadout(client)  # lazy residency: resident before we patch/grab the gate
    monkeypatch.setattr("suiban.routers.chat.QUEUE_NOTICE_AFTER_S", 0.05)
    release = asyncio.Event()
    original = BackendChat.complete

    async def blocked(self, payload: dict, timeout: float) -> dict:
        await release.wait()
        return await original(self, payload, timeout)

    monkeypatch.setattr(BackendChat, "complete", blocked)
    gate = client.app.state.bonsai.manager.slot("orchestrator").gate

    holder_status: list[int] = []

    def holder() -> None:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "hold the slot"}],
                "effort": "low",
            },
        )
        holder_status.append(resp.status_code)

    import time

    holder_thread = threading.Thread(target=holder)
    holder_thread.start()
    end = time.monotonic() + 5
    while time.monotonic() < end and not gate.busy:
        time.sleep(0.01)
    assert gate.busy

    waiter_payloads: list = []

    def waiter() -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "queued run"}],
                "effort": "low",
                "stream": True,
                "stream_events": True,
            },
        ) as resp:
            waiter_payloads.extend(_sse_payloads(resp.read().decode()))

    waiter_thread = threading.Thread(target=waiter)
    waiter_thread.start()
    # Give the waiter time to join the queue and cross the notice threshold,
    # then release the holder so both runs complete.
    end = time.monotonic() + 5
    while time.monotonic() < end and gate.queue_depth < 2:
        time.sleep(0.01)
    time.sleep(0.15)
    client.portal.call(release.set)
    holder_thread.join(timeout=10)
    waiter_thread.join(timeout=10)

    assert holder_status == [200]
    notices = [
        p
        for p in waiter_payloads
        if p != "DONE" and p["type"] == "notice" and p["code"] == "slot_queued"
    ]
    assert len(notices) == 1
    assert "queued behind 1 run(s) on slot orchestrator" in notices[0]["message"]
    assert waiter_payloads[-1] == "DONE"


# -- effort default chain (req.effort > settings.effort_default > mode) -------
def test_effort_chain_over_http(client: TestClient) -> None:
    def chat_effort(body_extra: dict) -> str:
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "hi"}],
                **body_extra,
            },
        )
        assert resp.status_code == 200
        return resp.json()["bonsai"]["effort"]

    # Unset effort_default: the mode defaults hold.
    assert chat_effort({"mode": "chat"}) == "mid"
    assert chat_effort({"mode": "code"}) == "high"

    # A configured effort_default overrides every mode default...
    client.patch("/v1/settings", json={"effort_default": "low"})
    assert client.post("/v1/system/apply").json()["applied"] is True
    assert chat_effort({"mode": "chat"}) == "low"
    assert chat_effort({"mode": "code"}) == "low"
    # ...but an explicit request effort still wins.
    assert chat_effort({"mode": "code", "effort": "xhigh"}) == "xhigh"


def test_effort_default_applies_to_research_jobs(client: TestClient) -> None:
    client.patch("/v1/settings", json={"effort_default": "low"})
    assert client.post("/v1/system/apply").json()["applied"] is True
    job_id = client.post("/v1/jobs", json={"type": "deep_research", "query": "effort?"}).json()[
        "id"
    ]
    # effort is internal (not in the JobStatus shape) — check the stored row.
    assert client.app.state.bonsai.jobs.get(job_id).effort == "low"
    explicit = client.post(
        "/v1/jobs", json={"type": "deep_research", "query": "q2", "effort": "high"}
    )
    # The first job may still be active; only assert when admitted.
    if explicit.status_code == 202:
        assert client.app.state.bonsai.jobs.get(explicit.json()["id"]).effort == "high"


def test_stream_error_event_is_nested() -> None:
    """Regression: a flat {'type': error_type} payload used to overwrite the
    envelope type so clients never saw an 'error' event (live 400 debugging)."""
    import json as _json

    from suiban.agent import events as ev

    e = ev.error("server_error", "boom")
    wire = _json.loads(e.as_sse().removeprefix("data: "))
    assert wire["type"] == "error"
    assert wire["error"] == {"type": "server_error", "message": "boom"}
