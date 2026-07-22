"""Mode registry + /v1/modes router: versioned prompts, toolsets, defaults."""

from __future__ import annotations

from fastapi.testclient import TestClient

from suiban.modes.registry import CHAT_MODES, MODES, system_prompt
from suiban.tools.registry import mode_tool_names


def test_all_modes_have_versioned_prompt_headers() -> None:
    for name, mode in MODES.items():
        version = mode.prompt_version()
        # v2 (audit 2026-07-22): every mode prompt gained the untrusted-content rule.
        # ultra advanced to v3 when it gained the volatile per-agent system-prompt
        # guidance (SLAP wave 4); the invariant is a parseable "name@N" header, N >= 2.
        expected_name = name if name != "deep_research" else "research"
        assert version.startswith(f"{expected_name}@"), version
        assert int(version.rsplit("@", 1)[1]) >= 2, version
        text = system_prompt(name)
        assert text.startswith("# ")
        assert len(text) > 500, f"{name} prompt is suspiciously short"


def test_chat_modes_vs_job_modes() -> None:
    assert set(CHAT_MODES) == {"chat", "code", "ultra"}
    assert MODES["deep_research"].endpoint == "/v1/jobs"
    for name in CHAT_MODES:
        assert MODES[name].endpoint == "/v1/chat/completions"


def test_mode_toolsets_reflect_write_gating() -> None:
    # chat/code list the orchestrator write tools; ultra/deep_research never do.
    for name in ("chat", "code"):
        assert {"memory_write", "skill_save", "skill_improve"} <= set(mode_tool_names(name))
    for name in ("ultra", "deep_research"):
        assert not {"memory_write", "skill_save", "skill_improve"} & set(mode_tool_names(name))
    assert "shell" in mode_tool_names("code")
    assert "shell" not in mode_tool_names("chat")
    assert "browse_t1" in mode_tool_names("deep_research")


def test_modes_router_matches_registry(client: TestClient) -> None:
    listed = {m["name"]: m for m in client.get("/v1/modes").json()["modes"]}
    assert set(listed) == set(MODES)
    for name, mode in MODES.items():
        entry = listed[name]
        assert entry["default_effort"] == mode.default_effort
        assert entry["tools"] == mode.tools
        assert entry["system_prompt_version"] == mode.prompt_version()
        # Prompt text is never exposed on the HTTP surface.
        assert "prompt" not in {k.lower() for k in entry} - {"system_prompt_version"}
        single = client.get(f"/v1/modes/{name}").json()
        assert single == entry


def test_modes_router_404_shape(client: TestClient) -> None:
    resp = client.get("/v1/modes/zen")
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"
