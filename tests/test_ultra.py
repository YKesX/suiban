"""Ultra mode: grammar-constrained plan, contained sub-agents on scripted workers,
worker-tool isolation, sequential fallback, effort inheritance + caps, sub-task
timeouts, graceful degradation, and the HTTP path against the mock backend."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from suiban import slap
from suiban.memory.service import MemoryService
from suiban.modes.ultra import (
    SEQUENTIAL_EFFORT_CAP,
    SEQUENTIAL_MAX_SUBTASKS,
    SUBTASK_TIMEOUT_HIGH_S,
    SUBTASK_TIMEOUT_LOW_S,
    UltraRun,
    UltraWorker,
    subtask_timeout_s,
    worker_effort,
    worker_prompt,
)
from suiban.tools.base import ToolContext
from suiban.tools.registry import (
    WORKER_TOOLSET,
    WRITE_TOOL_NAMES,
    build_worker_registry,
)


class ScriptedChat:
    """BackendChat stand-in: queued responses, recorded payloads."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict] = []

    async def complete(self, payload: dict, timeout: float) -> dict:
        self.payloads.append(payload)
        if not self._responses:
            raise AssertionError("scripted chat ran out of responses")
        return self._responses.pop(0)


def text_response(text: str) -> dict:
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def plan_response(subtasks: list[dict]) -> dict:
    return text_response(json.dumps({"subtasks": subtasks}))


def plan_with_prompts(subtasks: list[tuple[str, str, str]]) -> dict:
    """A plan whose sub-tasks each carry a volatile per-agent system_prompt."""
    return plan_response([{"title": t, "brief": b, "system_prompt": sp} for t, b, sp in subtasks])


def worker_report_response(status: str = "ok", summary: str = "done", output: str = "out") -> dict:
    return text_response(json.dumps({"status": status, "summary": summary, "output": output}))


@pytest.fixture
def memory(tmp_path: Path) -> MemoryService:
    service = MemoryService(tmp_path / "home")
    service.startup()
    yield service
    service.close()


def make_run(
    memory: MemoryService,
    tmp_path: Path,
    orch: ScriptedChat,
    workers: list[ScriptedChat],
    effort: str = "high",
) -> UltraRun:
    workdir = tmp_path / "jail"
    workdir.mkdir(exist_ok=True)
    return UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[
            UltraWorker(f"worker-{i + 1}", "bonsai-8b", 16384, chat)
            for i, chat in enumerate(workers)
        ],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="ultra-test", workdir=workdir, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "build the thing"}],
        effort=effort,
    )


async def collect(run: UltraRun) -> list:
    return [event async for event in run.run()]


# -- happy path: parallel dispatch on two workers -----------------------------
async def test_parallel_dispatch_two_workers(memory: MemoryService, tmp_path: Path) -> None:
    orch = ScriptedChat(
        [
            plan_response(
                [
                    {"title": "part one", "brief": "do part one"},
                    {"title": "part two", "brief": "do part two"},
                ]
            ),
            text_response("final synthesis"),
        ]
    )
    workers = [
        ScriptedChat([text_response("draft one"), worker_report_response(output="result-1")]),
        ScriptedChat([text_response("draft two"), worker_report_response(output="result-2")]),
    ]
    run = make_run(memory, tmp_path, orch, workers)
    events = await collect(run)

    types = [e.type for e in events]
    assert types[0] == "plan"
    assert events[0].payload["steps"] == ["part one", "part two"]
    spawns = [e for e in events if e.type == "agent_spawn"]
    results = [e for e in events if e.type == "agent_result"]
    assert len(spawns) == 2 and len(results) == 2
    assert {s.payload["agent_id"] for s in spawns} == {"agent-1", "agent-2"}
    assert all(s.payload["effort"] == "high" for s in spawns)  # inherit the request effort
    assert all(r.payload["status"] == "ok" for r in results)

    # Every spawn precedes its own result.
    def index_of(event_type: str, agent_id: str) -> int:
        return next(
            i
            for i, e in enumerate(events)
            if e.type == event_type and e.payload["agent_id"] == agent_id
        )

    for agent_id in ("agent-1", "agent-2"):
        assert index_of("agent_spawn", agent_id) < index_of("agent_result", agent_id)
    assert types[-3:] == ["delta", "usage", "done"]
    assert run.final_text == "final synthesis"
    assert run.finish_reason == "stop"
    # Both workers actually ran (fresh contexts: system prompt is the worker prompt).
    for chat in workers:
        first = chat.payloads[0]["messages"][0]
        assert first["role"] == "system"
        assert first["content"] == worker_prompt()


# -- worker tool isolation (THE containment assertion) ------------------------
async def test_workers_never_receive_write_tools(memory: MemoryService, tmp_path: Path) -> None:
    orch = ScriptedChat(
        [
            plan_response([{"title": "t", "brief": "b"}]),
            text_response("synth"),
        ]
    )
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    await collect(run)

    # Registry level: the worker registry cannot contain write tools.
    registry = build_worker_registry(memory=memory)
    assert set(registry.names) == set(WORKER_TOOLSET)
    assert not set(WRITE_TOOL_NAMES) & set(registry.names)

    # Wire level: no payload sent to the worker slot ever carried a write-tool
    # schema — the grammar-constrained decoder could not have emitted such a call.
    tool_payloads = [p for p in worker.payloads if "tools" in p]
    assert tool_payloads, "the worker loop must pass its toolset for grammar constraint"
    for payload in tool_payloads:
        names = {t["function"]["name"] for t in payload["tools"]}
        assert not set(WRITE_TOOL_NAMES) & names
        assert names <= set(WORKER_TOOLSET)


# -- effort inheritance / caps / timeouts (deep-detail pass) ------------------
def test_worker_effort_inherits_and_caps() -> None:
    # Parallel: inherit the request effort untouched.
    for effort in ("low", "mid", "high", "xhigh", "max"):
        assert worker_effort(effort, parallel=True) == effort
    # Sequential: capped at "mid" — a single slot cannot afford xhigh per sub-task.
    assert worker_effort("low", parallel=False) == "low"
    assert worker_effort("mid", parallel=False) == "mid"
    assert worker_effort("high", parallel=False) == SEQUENTIAL_EFFORT_CAP
    assert worker_effort("xhigh", parallel=False) == SEQUENTIAL_EFFORT_CAP
    assert worker_effort("max", parallel=False) == SEQUENTIAL_EFFORT_CAP


def test_subtask_timeout_scales_with_effort() -> None:
    assert subtask_timeout_s("low") == SUBTASK_TIMEOUT_LOW_S
    assert subtask_timeout_s("mid") == SUBTASK_TIMEOUT_LOW_S
    assert subtask_timeout_s("high") == SUBTASK_TIMEOUT_HIGH_S
    assert subtask_timeout_s("xhigh") == SUBTASK_TIMEOUT_HIGH_S
    assert subtask_timeout_s("max") == SUBTASK_TIMEOUT_HIGH_S


async def test_parallel_workers_inherit_low_effort_thinking_off(
    memory: MemoryService, tmp_path: Path
) -> None:
    """effort=low reaches the worker payloads: thinking off (enable_thinking False)."""
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker], effort="low")
    events = await collect(run)
    spawn = next(e for e in events if e.type == "agent_spawn")
    assert spawn.payload["effort"] == "low"
    loop_payload = next(p for p in worker.payloads if "tools" in p)
    assert loop_payload["chat_template_kwargs"]["enable_thinking"] is False


async def test_plan_truncated_to_worker_count_in_parallel(
    memory: MemoryService, tmp_path: Path
) -> None:
    orch = ScriptedChat(
        [
            plan_response(
                [
                    {"title": "one", "brief": "b1"},
                    {"title": "two", "brief": "b2"},
                    {"title": "three", "brief": "b3"},
                ]
            ),
            text_response("synth"),
        ]
    )
    workers = [
        ScriptedChat([text_response("d1"), worker_report_response()]),
        ScriptedChat([text_response("d2"), worker_report_response()]),
    ]
    run = make_run(memory, tmp_path, orch, workers)
    events = await collect(run)
    plan_event = next(e for e in events if e.type == "plan")
    assert plan_event.payload["steps"] == ["one", "two"]  # cap = worker count
    assert any(e.type == "notice" and e.payload["code"] == "ultra_plan_truncated" for e in events)
    assert len([e for e in events if e.type == "agent_result"]) == 2


async def test_subtask_timeout_becomes_structured_failure(
    memory: MemoryService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-task past its wall-clock budget is CANCELLED (the fake backend sees the
    CancelledError, like an aborted httpx call) and reported as a failed
    agent_result plus a timeout notice; synthesis still happens."""
    monkeypatch.setattr("suiban.modes.ultra.SUBTASK_TIMEOUT_LOW_S", 0.05)
    cancelled = asyncio.Event()

    class HangingChat:
        async def complete(self, payload: dict, timeout: float) -> dict:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("unreachable")

    orch = ScriptedChat(
        [
            plan_response([{"title": "stuck", "brief": "b"}]),
            text_response("synthesis works around the timeout"),
        ]
    )
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, HangingChat())],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=tmp_path, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
        effort="mid",
    )
    events = await collect(run)
    assert cancelled.is_set(), "the in-flight backend call must be cancelled"
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "failed"
    assert "timed out" in result.payload["summary"]
    timeout_notice = next(
        e for e in events if e.type == "notice" and e.payload["code"] == "ultra_subtask_timeout"
    )
    assert "budget" in timeout_notice.payload["message"]
    assert run.final_text == "synthesis works around the timeout"
    assert run.finish_reason == "stop"
    # The synthesis prompt saw the honest failure status.
    synthesis = orch.payloads[-1]["messages"][-1]["content"]
    assert "Status: failed" in synthesis


# -- sequential fallback ------------------------------------------------------
async def test_sequential_fallback_without_workers(memory: MemoryService, tmp_path: Path) -> None:
    orch = ScriptedChat(
        [
            plan_response([{"title": "solo", "brief": "do it"}]),
            text_response("worker-on-orchestrator draft"),  # sub-agent loop step
            worker_report_response(summary="did it", output="res"),  # structured report
            text_response("sequential synthesis"),
        ]
    )
    run = make_run(memory, tmp_path, orch, workers=[])
    events = await collect(run)

    notices = [e for e in events if e.type == "notice"]
    assert any(n.payload["code"] == "ultra_sequential" for n in notices)
    # Requested "high" degrades to the sequential cap — with a visible reason.
    capped = next(n for n in notices if n.payload["code"] == "ultra_effort_capped")
    assert SEQUENTIAL_EFFORT_CAP in capped.payload["message"]
    assert [e.type for e in events if e.type in ("agent_spawn", "agent_result")] == [
        "agent_spawn",
        "agent_result",
    ]
    # The sub-agent ran on the orchestrator slot but stayed contained.
    spawn = next(e for e in events if e.type == "agent_spawn")
    assert spawn.payload["model"] == "bonsai-27b"
    assert spawn.payload["effort"] == SEQUENTIAL_EFFORT_CAP
    loop_payload = next(p for p in orch.payloads if "tools" in p)
    names = {t["function"]["name"] for t in loop_payload["tools"]}
    assert not set(WRITE_TOOL_NAMES) & names
    assert run.final_text == "sequential synthesis"


async def test_sequential_low_effort_not_capped_no_notice(
    memory: MemoryService, tmp_path: Path
) -> None:
    orch = ScriptedChat(
        [
            plan_response([{"title": "solo", "brief": "do it"}]),
            text_response("draft"),
            worker_report_response(),
            text_response("synth"),
        ]
    )
    run = make_run(memory, tmp_path, orch, workers=[], effort="low")
    events = await collect(run)
    assert not any(
        e.type == "notice" and e.payload["code"] == "ultra_effort_capped" for e in events
    )
    spawn = next(e for e in events if e.type == "agent_spawn")
    assert spawn.payload["effort"] == "low"


async def test_sequential_subtask_cap_is_three(memory: MemoryService, tmp_path: Path) -> None:
    plan = [{"title": f"t{i}", "brief": f"b{i}"} for i in range(1, 5)]  # 4 planned
    per_subtask = [
        [text_response(f"draft {i}"), worker_report_response(summary=f"done {i}")]
        for i in range(1, 4)
    ]
    orch = ScriptedChat(
        [
            plan_response(plan),
            *[response for pair in per_subtask for response in pair],
            text_response("synth"),
        ]
    )
    run = make_run(memory, tmp_path, orch, workers=[])
    events = await collect(run)
    plan_event = next(e for e in events if e.type == "plan")
    assert plan_event.payload["steps"] == ["t1", "t2", "t3"]
    assert SEQUENTIAL_MAX_SUBTASKS == 3
    assert len([e for e in events if e.type == "agent_result"]) == 3
    assert any(e.type == "notice" and e.payload["code"] == "ultra_plan_truncated" for e in events)


# -- plan failure falls back to a single contained sub-task -------------------
async def test_unparseable_plan_falls_back_gracefully(
    memory: MemoryService, tmp_path: Path
) -> None:
    orch = ScriptedChat(
        [
            text_response("not json"),  # plan attempt
            text_response("still not json"),  # repair 1
            text_response("nope"),  # repair 2
            text_response("synthesis anyway"),
        ]
    )
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    events = await collect(run)

    plan_event = next(e for e in events if e.type == "plan")
    assert plan_event.payload["steps"] == ["complete the task"]
    assert any(e.type == "notice" and e.payload["code"] == "ultra_plan_fallback" for e in events)
    # The fallback brief carries the user task text.
    brief = worker.payloads[0]["messages"][1]["content"]
    assert "build the thing" in brief
    assert run.finish_reason == "stop"


# -- worker process death becomes a failed agent_result (chaos, mock seam) ----
async def test_worker_death_is_structured_failure(memory: MemoryService, tmp_path: Path) -> None:
    """A worker slot dying mid-run surfaces on the chat seam as httpx.ConnectError
    (connection refused to a dead llama-server). The sub-agent loop turns it into a
    graceful error finish; ultra reports a failed agent_result and synthesizes on."""

    class DeadSlotChat:
        async def complete(self, payload: dict, timeout: float) -> dict:
            raise httpx.ConnectError("All connection attempts failed")

    orch = ScriptedChat(
        [
            plan_response([{"title": "victim", "brief": "b"}]),
            text_response("synthesis notes the dead worker"),
        ]
    )
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, DeadSlotChat())],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=tmp_path, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
    )
    events = await collect(run)
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "failed"
    assert "backend request failed" in result.payload["summary"]
    assert events[-1].type == "done"
    assert run.finish_reason == "stop"  # the run completed with an honest synthesis
    synthesis = orch.payloads[-1]["messages"][-1]["content"]
    assert "Status: failed" in synthesis


# -- a crashed/failed worker becomes a failed agent_result --------------------
async def test_failed_worker_is_reported_not_fatal(memory: MemoryService, tmp_path: Path) -> None:
    class ExplodingChat:
        async def complete(self, payload: dict, timeout: float) -> dict:
            raise ValueError("worker boom")

    orch = ScriptedChat(
        [
            plan_response([{"title": "doomed", "brief": "b"}]),
            text_response("synthesis notes the failure"),
        ]
    )
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, ExplodingChat())],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=tmp_path, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
    )
    events = await collect(run)
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "failed"
    assert events[-1].type == "done"
    assert run.finish_reason == "stop"  # the run itself completed with a synthesis


# -- structured report failure degrades to the loop's own answer --------------
async def test_worker_report_parse_failure_uses_loop_text(
    memory: MemoryService, tmp_path: Path
) -> None:
    orch = ScriptedChat(
        [
            plan_response([{"title": "t", "brief": "b"}]),
            text_response("synth"),
        ]
    )
    worker = ScriptedChat(
        [
            text_response("the actual deliverable"),
            text_response("not json"),  # report attempt
            text_response("not json"),  # repair 1
            text_response("not json"),  # repair 2
        ]
    )
    run = make_run(memory, tmp_path, orch, [worker])
    events = await collect(run)
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "ok"
    assert "the actual deliverable" in result.payload["summary"]


# -- HTTP path over the mock backend ------------------------------------------
def test_ultra_mode_over_http_stream_events(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "do a big thing"}],
            "mode": "ultra",
            "stream": True,
            "stream_events": True,
        },
    ) as resp:
        assert resp.status_code == 200
        raw = resp.read().decode()

    payloads = [
        json.loads(line[len("data: ") :])
        for line in raw.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    types = [p["type"] for p in payloads]
    assert "plan" in types
    assert "agent_spawn" in types and "agent_result" in types
    assert types[-1] == "done"
    spawn = next(p for p in payloads if p["type"] == "agent_spawn")
    # agent_spawn/agent_result now carry the SLAP task_id (api.md §12, 2026-07-22b).
    assert set(spawn) == {"type", "agent_id", "model", "task", "effort", "task_id"}
    assert spawn["task_id"].startswith("T")
    # The 24 GB fixture has workers (parallel): sub-tasks inherit the mode's
    # default effort ("high" for ultra) instead of the old pinned xhigh.
    assert spawn["effort"] == "high"
    result = next(p for p in payloads if p["type"] == "agent_result")
    assert set(result) == {"type", "agent_id", "status", "summary", "task_id"}
    assert result["task_id"] == spawn["task_id"]


def test_ultra_mode_over_http_non_streaming(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "do a big thing"}],
            "mode": "ultra",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"]
    assert body["bonsai"]["mode"] == "ultra"


# -- SLAP: volatile per-agent system prompts ----------------------------------
async def test_plan_system_prompt_is_used_by_worker(memory: MemoryService, tmp_path: Path) -> None:
    """A sub-task's volatile system_prompt becomes the worker's system message,
    replacing the static fallback."""
    prompt = "You are a meticulous SQL migration reviewer. Done = every DDL is reversible."
    orch = ScriptedChat([plan_with_prompts([("t", "b", prompt)]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    await collect(run)

    system_message = worker.payloads[0]["messages"][0]
    assert system_message["role"] == "system"
    assert system_message["content"] == prompt
    assert system_message["content"] != worker_prompt()


async def test_volatile_system_prompt_does_not_leak_between_workers(
    memory: MemoryService, tmp_path: Path
) -> None:
    """Each worker sees ONLY its own volatile prompt; a prompt never crosses to the
    other agent's traffic."""
    sp1 = "SYSTEM-PROMPT-ALPHA: you are the alpha specialist."
    sp2 = "SYSTEM-PROMPT-BETA: you are the beta specialist."
    orch = ScriptedChat(
        [
            plan_with_prompts([("one", "b1", sp1), ("two", "b2", sp2)]),
            text_response("synth"),
        ]
    )
    workers = [
        ScriptedChat([text_response("d1"), worker_report_response()]),
        ScriptedChat([text_response("d2"), worker_report_response()]),
    ]
    run = make_run(memory, tmp_path, orch, workers)
    await collect(run)

    # Both prompts were used, each by exactly one worker (pool order is nondeterministic).
    systems = {w.payloads[0]["messages"][0]["content"] for w in workers}
    assert systems == {sp1, sp2}
    # No leak: each worker's ENTIRE traffic contains exactly one of the two prompts.
    for worker in workers:
        blob = json.dumps(worker.payloads)
        assert (sp1 in blob) != (sp2 in blob)


async def test_volatile_system_prompt_not_persisted_to_trace_or_memory(
    memory: MemoryService, tmp_path: Path
) -> None:
    """The volatile prompt reaches the worker but is stripped from the recorded assign,
    absent from the whole SLAP trace, and never archived (tool_messages)."""
    prompt = "SECRET-VOLATILE-PROMPT: discarded when the agent finishes."
    orch = ScriptedChat([plan_with_prompts([("t", "b", prompt)]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    await collect(run)

    assert worker.payloads[0]["messages"][0]["content"] == prompt  # used for this agent
    assign = next(m for m in run.slap_messages if m["operation"] == "assign")
    assert "system_prompt" not in assign  # redacted from the recorded assign
    assert prompt not in json.dumps(run.slap_messages)  # nowhere in the trace
    assert prompt not in json.dumps(run.tool_messages)  # never archived


async def test_fallback_prompt_when_plan_omits_system_prompt(
    memory: MemoryService, tmp_path: Path
) -> None:
    """When the orchestrator omits a system_prompt, the worker uses the static
    ultra_worker fallback (and the assign still validates and records)."""
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("s")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    await collect(run)

    assert worker.payloads[0]["messages"][0]["content"] == worker_prompt()
    assign = next(m for m in run.slap_messages if m["operation"] == "assign")
    assert "system_prompt" not in assign
    assert slap.validate_message(assign) == []


# -- SLAP: transcript (assign / result / decide / capability) -----------------
async def test_slap_transcript_validates_and_covers_operations(
    memory: MemoryService, tmp_path: Path
) -> None:
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response(output="deliverable")])
    run = make_run(memory, tmp_path, orch, [worker])
    await collect(run)

    ops = [m["operation"] for m in run.slap_messages]
    assert "capability" in ops
    assert "assign" in ops
    assert "result" in ops
    assert "decide" in ops
    for message in run.slap_messages:
        assert slap.validate_message(message) == []
    # Every recorded message carries the run's task ids and a unique message id.
    ids = [m["message_id"] for m in run.slap_messages]
    assert len(ids) == len(set(ids))
    result = next(m for m in run.slap_messages if m["operation"] == "result")
    assert result["status"] == "completed"
    assert result["claims"][0]["evidence"]  # every claim carries evidence
    decide = next(m for m in run.slap_messages if m["operation"] == "decide")
    assert decide["decision"] == "accept"


async def test_capability_advertises_slot_facts(memory: MemoryService, tmp_path: Path) -> None:
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("s")])
    worker_chat = ScriptedChat([text_response("draft"), worker_report_response()])
    workdir = tmp_path / "jail"
    workdir.mkdir(exist_ok=True)
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[
            UltraWorker(
                "worker-1",
                "bonsai-8b",
                16384,
                worker_chat,
                family="bonsai",
                quant="Q2_0",
                backend="cuda",
                workload=1.0,
            )
        ],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=workdir, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
    )
    await collect(run)
    cap = next(m for m in run.slap_messages if m["operation"] == "capability")
    assert cap["model"] == "bonsai-8b"
    assert cap["quantization"] == "Q2_0"
    assert cap["backend"] == "cuda"
    assert cap["context_limit"] == 16384
    assert slap.validate_message(cap) == []


async def test_agent_events_carry_task_id(memory: MemoryService, tmp_path: Path) -> None:
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("s")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    events = await collect(run)
    spawn = next(e for e in events if e.type == "agent_spawn")
    result = next(e for e in events if e.type == "agent_result")
    assert spawn.payload["task_id"] == "T1"
    assert result.payload["task_id"] == "T1"


async def test_invalid_slap_message_degrades_gracefully(
    memory: MemoryService, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SLAP message that fails validation emits a slap_degraded notice and the run
    falls back to the structured path (static prompt, valid agent_result) — never a
    crash, and the invalid message is not recorded."""

    def broken_assign(**kwargs: object) -> dict:
        # Missing required role/goal + an unexpected property → schema-invalid.
        return {
            "protocol": "SLAP",
            "version": "1.0",
            "operation": "assign",
            "message_id": "Mx",
            "task_id": "T1",
            "bogus": True,
        }

    monkeypatch.setattr("suiban.slap.build_assign", broken_assign)
    orch = ScriptedChat([plan_with_prompts([("t", "b", "vp")]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = make_run(memory, tmp_path, orch, [worker])
    events = await collect(run)

    assert any(e.type == "notice" and e.payload["code"] == "slap_degraded" for e in events)
    assert run.final_text == "synth"
    assert run.finish_reason == "stop"
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "ok"
    # The invalid assign was not recorded; the worker fell back to the static prompt.
    assert all(m["operation"] != "assign" for m in run.slap_messages)
    assert worker.payloads[0]["messages"][0]["content"] == worker_prompt()


# -- SLAP toggle (settings.slap.enabled=false → plain structured-dict path) ----
async def test_slap_disabled_routes_the_plain_dict_path(
    memory: MemoryService, tmp_path: Path
) -> None:
    """With SLAP off the dispatch still runs (plan → contained sub-agent → synthesis),
    but no SLAP messages are built or recorded, and there is NO slap_degraded notice —
    SLAP is off by choice, not broken."""
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    workdir = tmp_path / "jail"
    workdir.mkdir(exist_ok=True)
    captured: dict[str, list] = {}
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, worker)],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=workdir, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
        session_id="slap-off",
        trace_sink=lambda sid, msgs: captured.__setitem__(sid, msgs),
        slap_enabled=False,
    )
    events = await collect(run)

    assert run.slap_messages == []  # nothing recorded
    assert captured.get("slap-off") == []  # the trace sink got an empty transcript
    assert not any(e.type == "notice" and e.payload["code"] == "slap_degraded" for e in events)
    # The dispatch itself is unchanged: the sub-agent ran and synthesis produced an answer.
    assert any(e.type == "agent_spawn" for e in events)
    result = next(e for e in events if e.type == "agent_result")
    assert result.payload["status"] == "ok"
    assert run.final_text == "synth"
    assert run.finish_reason == "stop"


async def test_slap_disabled_still_uses_volatile_prompt(
    memory: MemoryService, tmp_path: Path
) -> None:
    """The plain dict path keeps the plan's per-agent roles: a volatile system_prompt is
    still the worker's system message even though no SLAP assign is built."""
    prompt = "ROLE: you are the dict-path specialist for THIS sub-task."
    orch = ScriptedChat([plan_with_prompts([("t", "b", prompt)]), text_response("synth")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, worker)],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="s", workdir=tmp_path, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
        slap_enabled=False,
    )
    await collect(run)
    assert worker.payloads[0]["messages"][0]["content"] == prompt
    assert run.slap_messages == []


def test_slap_toggle_over_http_trace_empty_schema_still_served(client: TestClient) -> None:
    """settings.slap.enabled=false: /v1/slap still serves the protocol + schema, but a
    disabled Ultra run records no trace (api.md 2026-07-22c)."""
    client.patch("/v1/settings", json={"slap": {"enabled": False}})
    assert client.post("/v1/system/apply").json()["applied"] is True

    # The schema surface is unchanged.
    info = client.get("/v1/slap").json()
    assert info["operations"]
    assert client.get("/v1/slap/schema/assign").status_code == 200

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "do a big thing"}],
            "mode": "ultra",
            "session_id": "slap-off-http",
        },
    )
    assert resp.status_code == 200
    assert client.get("/v1/slap/trace/slap-off-http").json()["messages"] == []


def test_slap_enabled_over_http_records_trace(client: TestClient) -> None:
    """The default (SLAP on): the same Ultra run DOES record a validated transcript, so
    the disabled case above is a real contrast, not a vacuous pass."""
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "do a big thing"}],
            "mode": "ultra",
            "session_id": "slap-on-http",
        },
    )
    assert resp.status_code == 200
    messages = client.get("/v1/slap/trace/slap-on-http").json()["messages"]
    assert messages
    assert all(slap.validate_message(m) == [] for m in messages)


async def test_trace_sink_receives_validated_transcript(
    memory: MemoryService, tmp_path: Path
) -> None:
    """The trace_sink is handed the validated message list keyed by session_id."""
    captured: dict[str, list] = {}
    orch = ScriptedChat([plan_response([{"title": "t", "brief": "b"}]), text_response("s")])
    worker = ScriptedChat([text_response("draft"), worker_report_response()])
    workdir = tmp_path / "jail"
    workdir.mkdir(exist_ok=True)
    run = UltraRun(
        orchestrator=UltraWorker("orchestrator", "bonsai-27b", 32768, orch),
        workers=[UltraWorker("worker-1", "bonsai-8b", 16384, worker)],
        registry_factory=lambda: build_worker_registry(memory=memory),
        tool_ctx_factory=lambda: ToolContext(
            session_id="sess-abc", workdir=workdir, role="worker", mode="ultra"
        ),
        messages=[{"role": "user", "content": "go"}],
        session_id="sess-abc",
        trace_sink=lambda sid, msgs: captured.__setitem__(sid, msgs),
    )
    await collect(run)
    assert "sess-abc" in captured
    assert [m["operation"] for m in captured["sess-abc"]]
    assert all(slap.validate_message(m) == [] for m in captured["sess-abc"])
