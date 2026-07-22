"""Regression transcript fixtures: canned multi-step tool conversations (JSON files
under tests/fixtures/transcripts/) replayed through the real AgentLoop against a
scripted backend. Guards the loop's end-to-end behavior — repair semantics, counters,
event stream shape — against refactors, including a 20-step long-haul completion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from suiban.agent.loop import AgentLoop
from suiban.effort import Sampling
from suiban.tools.base import Tool, ToolContext, ToolResult
from suiban.tools.registry import ToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "transcripts"
FIXTURE_NAMES = ["happy", "malformed_then_repaired", "repair_exhausted", "long_haul_20"]
SAMPLING = Sampling(temperature=0.5, top_p=0.85, top_k=20)


class ScriptedChat:
    """Stands in for BackendChat: returns the fixture's responses in order."""

    def __init__(self, responses: list[dict]) -> None:
        self.remaining = list(responses)
        self.payloads: list[dict] = []

    async def complete(self, payload: dict, timeout: float) -> dict:
        self.payloads.append(payload)
        return self.remaining.pop(0)


class StepTool(Tool):
    name = "step"
    description = "Record one step note."
    parameters = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": ["note"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.notes: list[str] = []

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.notes.append(args["note"])
        return ToolResult("ok", f"recorded: {args['note']}")


def replay_setup(name: str, tmp_path: Path) -> tuple[dict, StepTool, ScriptedChat, AgentLoop]:
    fixture = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    tool = StepTool()
    chat = ScriptedChat(fixture["responses"])
    loop = AgentLoop(
        chat,  # duck-typed BackendChat
        model="bonsai-27b",
        registry=ToolRegistry([tool]),
        ctx=ToolContext(session_id="s1", workdir=tmp_path, role="orchestrator"),
        messages=[{"role": "user", "content": "go"}],
        sampling=SAMPLING,
        thinking_budget_tokens=0,
        max_iterations=fixture["max_iterations"],
    )
    return fixture, tool, chat, loop


@pytest.mark.parametrize("name", FIXTURE_NAMES)
async def test_transcript_fixture_replays(name: str, tmp_path: Path) -> None:
    fixture, tool, chat, loop = replay_setup(name, tmp_path)
    events = [event async for event in loop.run()]
    expect = fixture["expect"]

    assert loop.final_text == expect["final_text"]
    assert loop.finish_reason == expect["finish_reason"]
    statuses = [e.payload["status"] for e in events if e.type == "tool_result"]
    assert statuses == expect["tool_result_statuses"]
    assert tool.notes == expect["executed_notes"]
    assert loop.tool_stats == expect["tool_stats"]

    usage = next(e for e in events if e.type == "usage")
    if expect["usage_has_malformed_fields"]:
        stats = expect["tool_stats"]
        assert usage.payload["malformed_calls"] == stats["malformed_calls"]
        assert usage.payload["repaired_calls"] == stats["repaired_calls"]
        assert usage.payload["abandoned_calls"] == stats["abandoned_calls"]
    else:
        assert set(usage.payload) == {"prompt_tokens", "completion_tokens", "thinking_tokens"}

    assert chat.remaining == []  # the whole conversation was consumed


async def test_long_haul_20_completes_end_to_end(tmp_path: Path) -> None:
    """The headline assertion: a 20-step tool task runs to full completion — every
    step executes exactly once, in order, and the run ends with a normal answer
    (no ceiling notice, no errors)."""
    _fixture, tool, chat, loop = replay_setup("long_haul_20", tmp_path)
    events = [event async for event in loop.run()]

    assert tool.notes == [f"step-{i:02d}" for i in range(1, 21)]
    results = [e for e in events if e.type == "tool_result"]
    assert len(results) == 20
    assert all(r.payload["status"] == "ok" for r in results)
    assert not [e for e in events if e.type in ("notice", "error")]
    done = events[-1]
    assert done.type == "done"
    assert done.payload["finish_reason"] == "stop"
    assert loop.final_text == "all 20 steps recorded — task complete"
    # Every step carried the tool schemas (grammar plumbing held for all 21 calls).
    assert len(chat.payloads) == 21
    assert all("tools" in p for p in chat.payloads)
