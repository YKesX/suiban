"""The stream_events envelope: typed constructors for every api.md event type.

Wire format (stream_events:true): each SSE line is `data: {"type": ...}`; the stream
terminates with {"type":"done"} then `data: [DONE]`. Clients MUST ignore unknown
types — the union may grow additively within v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentEvent:
    type: str
    payload: dict[str, Any]

    def as_dict(self) -> dict:
        return {"type": self.type, **self.payload}

    def as_sse(self) -> str:
        return f"data: {json.dumps(self.as_dict(), ensure_ascii=False)}\n\n"


def delta(text: str) -> AgentEvent:
    return AgentEvent("delta", {"text": text})


def thinking_status(phase: str, thinking_tokens: int) -> AgentEvent:
    return AgentEvent("thinking_status", {"phase": phase, "thinking_tokens": thinking_tokens})


def tool_call(call_id: str, name: str, arguments: dict) -> AgentEvent:
    return AgentEvent("tool_call", {"id": call_id, "name": name, "arguments": arguments})


def tool_result(
    call_id: str, name: str, status: str, summary: str, confirm_token: str | None = None
) -> AgentEvent:
    """`confirm_token` appears in the payload only when set — api.md sends it solely
    with status "denied"; ok/error events must not carry the key at all."""
    payload: dict[str, Any] = {"id": call_id, "name": name, "status": status, "summary": summary}
    if confirm_token is not None:
        payload["confirm_token"] = confirm_token
    return AgentEvent("tool_result", payload)


def plan(steps: list[str]) -> AgentEvent:
    return AgentEvent("plan", {"steps": steps})


def agent_spawn(agent_id: str, model: str, task: str, effort: str) -> AgentEvent:
    return AgentEvent(
        "agent_spawn", {"agent_id": agent_id, "model": model, "task": task, "effort": effort}
    )


def agent_result(agent_id: str, status: str, summary: str) -> AgentEvent:
    return AgentEvent("agent_result", {"agent_id": agent_id, "status": status, "summary": summary})


def compression(trigger_pct: float, messages_summarized: int) -> AgentEvent:
    return AgentEvent(
        "compression",
        {"trigger_pct": trigger_pct, "messages_summarized": messages_summarized},
    )


def notice(level: str, code: str, message: str) -> AgentEvent:
    return AgentEvent("notice", {"level": level, "code": code, "message": message})


def usage(
    prompt_tokens: int,
    completion_tokens: int,
    thinking_tokens: int,
    *,
    malformed_calls: int | None = None,
    repaired_calls: int | None = None,
    abandoned_calls: int | None = None,
) -> AgentEvent:
    """The counters are optional additive fields (api.md, 2026-07-21d): agentic runs
    include them only when the run had malformed tool calls — clean runs and
    non-agentic paths keep the original three-field payload."""
    payload: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thinking_tokens": thinking_tokens,
    }
    if malformed_calls is not None:
        payload["malformed_calls"] = malformed_calls
    if repaired_calls is not None:
        payload["repaired_calls"] = repaired_calls
    if abandoned_calls is not None:
        payload["abandoned_calls"] = abandoned_calls
    return AgentEvent("usage", payload)


def done(finish_reason: str) -> AgentEvent:
    return AgentEvent("done", {"finish_reason": finish_reason})


def error(error_type: str, message: str) -> AgentEvent:
    # Nested on purpose: a flat {"type": error_type} would overwrite the envelope's
    # own "type" in as_dict() and the event would stop being an "error" event on the
    # wire (observed live: clients saw {"type": "server_error"} and ignored it).
    return AgentEvent("error", {"error": {"type": error_type, "message": message}})
