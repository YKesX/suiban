"""Agent loop behavior against a scripted backend: happy path, malformed-then-repaired
tool calls, repair exhaustion (graceful step failure, per-RUN budget), the
malformed-rate counters, and the iteration ceiling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from suiban.agent.loop import MAX_REPAIR_ATTEMPTS, AgentLoop
from suiban.effort import Sampling
from suiban.tools.base import Tool, ToolContext, ToolResult
from suiban.tools.registry import ToolRegistry

SAMPLING = Sampling(temperature=0.5, top_p=0.85, top_k=20)


class ScriptedChat:
    """Stands in for BackendChat: returns queued responses, records payloads."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict] = []

    async def complete(self, payload: dict, timeout: float) -> dict:
        self.payloads.append(payload)
        return self._responses.pop(0)


class EchoTool(Tool):
    name = "echo"
    description = "Echo the given text back."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        self.calls.append(args)
        return ToolResult("ok", f"echo: {args['text']}")


def text_response(text: str, *, thinking: int = 0) -> dict:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}
    if thinking:
        usage["thinking_tokens"] = thinking
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": usage,
    }


def tool_call_response(name: str, arguments: dict | str) -> dict:
    raw = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": name, "arguments": raw},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def make_loop(chat: ScriptedChat, tool: Tool, *, max_iterations: int = 8) -> AgentLoop:
    return AgentLoop(
        chat,  # duck-typed BackendChat
        model="bonsai-27b",
        registry=ToolRegistry([tool]),
        ctx=ToolContext(session_id="s1", workdir=Path("/tmp"), role="orchestrator"),
        messages=[{"role": "user", "content": "go"}],
        sampling=SAMPLING,
        thinking_budget_tokens=0,
        max_iterations=max_iterations,
    )


async def collect(loop: AgentLoop) -> list:
    return [event async for event in loop.run()]


async def test_happy_path_tool_call_then_answer() -> None:
    tool = EchoTool()
    chat = ScriptedChat(
        [tool_call_response("echo", {"text": "hi"}), text_response("done!", thinking=7)]
    )
    loop = make_loop(chat, tool)
    events = await collect(loop)

    types = [e.type for e in events]
    assert types == ["tool_call", "tool_result", "thinking_status", "delta", "usage", "done"]
    assert tool.calls == [{"text": "hi"}]
    assert events[0].payload == {"id": "call_1", "name": "echo", "arguments": {"text": "hi"}}
    assert events[1].payload["status"] == "ok"
    # confirm_token is a denial-only key: the happy path must not carry it at all.
    assert "confirm_token" not in events[1].payload
    assert loop.final_text == "done!"
    assert loop.finish_reason == "stop"
    assert loop.total_usage == {"prompt_tokens": 20, "completion_tokens": 10, "thinking_tokens": 7}
    # Grammar constraint plumbing: every step carried the tool schemas + tool_choice.
    assert all("tools" in p and "tool_choice" in p for p in chat.payloads)
    # The tool result went back to the model as a role:"tool" message.
    assert chat.payloads[1]["messages"][-1]["role"] == "tool"


async def test_malformed_call_is_repaired() -> None:
    tool = EchoTool()
    chat = ScriptedChat(
        [
            tool_call_response("echo", {"wrong_key": 1}),  # schema violation
            tool_call_response("echo", {"text": "fixed"}),  # repaired by the model
            text_response("recovered"),
        ]
    )
    loop = make_loop(chat, tool)
    events = await collect(loop)

    statuses = [e.payload.get("status") for e in events if e.type == "tool_result"]
    assert statuses == ["error", "ok"]  # rejected once, then the repaired call ran
    assert tool.calls == [{"text": "fixed"}]
    # The repair prompt carried the validation error back to the model.
    repair_message = chat.payloads[1]["messages"][-1]
    assert repair_message["role"] == "tool"
    assert "invalid arguments" in repair_message["content"].lower() or "rejected" in (
        repair_message["content"].lower()
    )
    assert loop.final_text == "recovered"


async def test_repair_exhaustion_fails_step_gracefully() -> None:
    tool = EchoTool()
    bad = tool_call_response("echo", "not json {{")
    chat = ScriptedChat([bad, bad, bad, text_response("gave up on the tool, answering anyway")])
    loop = make_loop(chat, tool)
    events = await collect(loop)

    results = [e for e in events if e.type == "tool_result"]
    assert len(results) == MAX_REPAIR_ATTEMPTS + 1
    assert all(r.payload["status"] == "error" for r in results)
    assert "repair" in results[0].payload["summary"].lower()
    assert "failed after" in results[-1].payload["summary"].lower()
    assert tool.calls == []  # the malformed call never executed
    assert loop.finish_reason == "stop"  # ... and the run still completed
    assert loop.final_text == "gave up on the tool, answering anyway"
    assert loop.tool_stats == {
        "tool_calls": 3,
        "malformed_calls": 3,
        "repaired_calls": 0,
        "abandoned_calls": 1,
    }


async def test_repair_budget_is_per_run_not_per_episode() -> None:
    """Alternating malformed/valid calls must NOT refill the budget: with
    MAX_REPAIR_ATTEMPTS=2, the third malformed call is abandoned even though valid
    calls executed in between (the old per-episode counter reset on every success)."""
    tool = EchoTool()
    chat = ScriptedChat(
        [
            tool_call_response("echo", {"bogus": 1}),  # malformed -> repair 1/2
            tool_call_response("echo", {"text": "ok-1"}),  # valid (repaired)
            tool_call_response("echo", {"bogus": 2}),  # malformed -> repair 2/2
            tool_call_response("echo", {"text": "ok-2"}),  # valid (repaired)
            tool_call_response("echo", {"bogus": 3}),  # malformed -> budget gone: abandoned
            text_response("finished despite the flaky tool calls"),
        ]
    )
    loop = make_loop(chat, tool)
    events = await collect(loop)

    statuses = [e.payload["status"] for e in events if e.type == "tool_result"]
    assert statuses == ["error", "ok", "error", "ok", "error"]
    # The third malformed call was abandoned outright, not offered another repair.
    last = [e for e in events if e.type == "tool_result"][-1]
    assert "failed after" in last.payload["summary"]
    assert tool.calls == [{"text": "ok-1"}, {"text": "ok-2"}]
    assert loop.tool_stats == {
        "tool_calls": 5,
        "malformed_calls": 3,
        "repaired_calls": 2,
        "abandoned_calls": 1,
    }
    # The counters ride the usage event (api.md optional additive fields).
    usage = next(e for e in events if e.type == "usage")
    assert usage.payload["malformed_calls"] == 3
    assert usage.payload["repaired_calls"] == 2
    assert usage.payload["abandoned_calls"] == 1


async def test_usage_event_omits_counters_on_clean_runs() -> None:
    """Clean runs keep the original three-field usage payload — the counters are
    additive fields present only when a run actually had malformed calls."""
    tool = EchoTool()
    chat = ScriptedChat([tool_call_response("echo", {"text": "hi"}), text_response("done")])
    loop = make_loop(chat, tool)
    events = await collect(loop)

    usage = next(e for e in events if e.type == "usage")
    assert set(usage.payload) == {"prompt_tokens", "completion_tokens", "thinking_tokens"}
    assert loop.tool_stats["malformed_calls"] == 0


async def test_iteration_ceiling_forces_toolless_wrapup() -> None:
    tool = EchoTool()
    chat = ScriptedChat(
        [
            tool_call_response("echo", {"text": "1"}),
            tool_call_response("echo", {"text": "2"}),
            text_response("best effort answer"),  # the forced wrap-up call
        ]
    )
    loop = make_loop(chat, tool, max_iterations=2)
    events = await collect(loop)

    notices = [e for e in events if e.type == "notice"]
    assert len(notices) == 1
    assert notices[0].payload["code"] == "tool_iteration_ceiling"
    assert loop.finish_reason == "length"
    assert loop.final_text == "best effort answer"
    # The wrap-up call forbids tools.
    assert chat.payloads[-1]["tool_choice"] == "none"


async def test_denied_tool_result_event_carries_confirm_token() -> None:
    """api.md (v1 additive, 2026-07-21): tool_result with status "denied" carries the
    single-use confirm_token; it must survive into the SSE payload verbatim."""

    class GateTool(Tool):
        name = "gate"
        description = "Always denies, issuing a confirmation token."
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return ToolResult(
                "denied",
                "needs user confirmation; re-run with the confirm_token",
                summary="confirmation required: gate",
                confirm_token="tok-abc123",
            )

    chat = ScriptedChat([tool_call_response("gate", {}), text_response("asked the user")])
    loop = make_loop(chat, GateTool())
    events = await collect(loop)

    result = next(e for e in events if e.type == "tool_result")
    assert result.payload["status"] == "denied"
    assert result.payload["confirm_token"] == "tok-abc123"
    assert result.payload["summary"]  # the instructing summary text is kept alongside
    # ... and the key survives SSE serialization (this is what routers/chat.py emits).
    wire = json.loads(result.as_sse()[len("data: ") :])
    assert wire["confirm_token"] == "tok-abc123"
    assert wire["type"] == "tool_result"


async def test_backend_failure_aborts_with_error_event() -> None:
    class ExplodingChat:
        async def complete(self, payload: dict, timeout: float) -> dict:
            raise ValueError("boom")

    loop = AgentLoop(
        ExplodingChat(),
        model="bonsai-27b",
        registry=ToolRegistry([]),
        ctx=ToolContext(session_id="s1", workdir=Path("/tmp")),
        messages=[{"role": "user", "content": "go"}],
        sampling=SAMPLING,
        thinking_budget_tokens=0,
        max_iterations=4,
    )
    events = [event async for event in loop.run()]
    assert [e.type for e in events] == ["error", "done"]
    assert events[1].payload["finish_reason"] == "error"
    assert loop.finish_reason == "error"


def test_tool_result_cap_math() -> None:
    from suiban.agent.loop import tool_result_cap

    assert tool_result_cap(8192) == 6553  # 20% of 8K ctx at 4 chars/token
    assert tool_result_cap(16384) == 13107
    assert tool_result_cap(512) == 2000  # floor


async def test_fat_tool_result_is_truncated_for_the_model() -> None:
    """Regression: a 40K-char browse_t1-sized result blew past the slot ctx on the
    next step and llama-server 400'd the run (observed live in Ultra sub-tasks).
    Only the model-visible tool message is trimmed."""

    class FatTool(EchoTool):
        async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
            return ToolResult("ok", "x" * 40_000, summary="fat page")

    chat = ScriptedChat([tool_call_response("echo", {"text": "hi"}), text_response("done")])
    tool = FatTool()
    loop = AgentLoop(
        chat,
        model="bonsai-27b",
        registry=ToolRegistry([tool]),
        ctx=ToolContext(session_id="s1", workdir=Path("/tmp"), role="orchestrator"),
        messages=[{"role": "user", "content": "go"}],
        sampling=SAMPLING,
        thinking_budget_tokens=0,
        max_iterations=8,
        tool_result_max_chars=6553,
    )
    async for _ in loop.run():
        pass
    tool_msgs = [m for m in loop.tool_messages if m["name"] == "echo"]
    assert tool_msgs, "tool message expected"
    content = tool_msgs[0]["content"]
    assert len(content) < 7000
    assert "[tool result truncated: 6553 of 40000 chars" in content
