"""Post-task reflection (api.md §11 notes, additive 2026-07-21c): trigger cadence
(once per session per N=3 exchanges), the "none" path, the memory_write path with the
one-writer rule intact, failure tolerance, and the worker/external/ultra/anonymous
exclusions at the chat router."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban.effort import Sampling
from suiban.memory import reflection
from suiban.memory.service import MemoryService
from suiban.memory.skills import SKILL_REJECTION_PREFIX
from suiban.tools.base import ToolContext
from suiban.tools.memory_tools import MemoryWriteTool, SkillSaveTool
from suiban.tools.registry import ToolRegistry

SAMPLING = Sampling(temperature=0.7, top_p=0.95, top_k=20)


@pytest.fixture(autouse=True)
def fresh_counters():
    reflection.reset()
    yield
    reflection.reset()


def _memory(tmp_path: Path) -> MemoryService:
    service = MemoryService(tmp_path / "home")
    service.startup()
    return service


def _registry(memory: MemoryService) -> ToolRegistry:
    return ToolRegistry([MemoryWriteTool(memory)])


def _ctx(tmp_path: Path, role: str = "orchestrator") -> ToolContext:
    return ToolContext(session_id="sess-r", workdir=tmp_path, role=role, mode="chat")


def _text_response(text: str) -> dict:
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ]
    }


def _tool_call_response(arguments: str) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-r1",
                            "type": "function",
                            "function": {"name": "memory_write", "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


# -- cadence ------------------------------------------------------------------
def test_should_reflect_first_then_every_third_per_session() -> None:
    fired = [reflection.should_reflect("sess-a") for _ in range(7)]
    assert fired == [True, False, False, True, False, False, True]
    # A different session has its own counter.
    assert reflection.should_reflect("sess-b") is True


def test_reset_clears_counters() -> None:
    assert reflection.should_reflect("sess-a") is True
    assert reflection.should_reflect("sess-a") is False
    reflection.reset()
    assert reflection.should_reflect("sess-a") is True


def test_exchange_counter_is_bounded_lru() -> None:
    """The per-session exchange counter cannot grow without bound (audit 2026-07-22):
    it is an LRU capped at MAX_TRACKED_SESSIONS, evicting the oldest session first."""
    cap = reflection.MAX_TRACKED_SESSIONS
    for i in range(cap):
        reflection.should_reflect(f"s{i}")
    assert len(reflection._EXCHANGE_COUNTS) == cap

    # Overflow by a handful; length stays pinned and the OLDEST sessions are gone.
    for i in range(cap, cap + 5):
        reflection.should_reflect(f"s{i}")
    assert len(reflection._EXCHANGE_COUNTS) == cap
    assert "s0" not in reflection._EXCHANGE_COUNTS
    assert "s4" not in reflection._EXCHANGE_COUNTS
    assert f"s{cap + 4}" in reflection._EXCHANGE_COUNTS

    # An evicted session simply starts fresh — it reflects again on its next exchange,
    # the same benign "one extra reflection" cost as a restart.
    assert reflection.should_reflect("s0") is True


def test_lru_touch_protects_recently_used_session() -> None:
    """Touching a session moves it to the newest slot, so it survives an eviction that
    would otherwise claim it as oldest."""
    cap = reflection.MAX_TRACKED_SESSIONS
    for i in range(cap):
        reflection.should_reflect(f"s{i}")
    reflection.should_reflect("s0")  # touch the oldest -> now newest
    reflection.should_reflect("new-session")  # forces one eviction
    assert len(reflection._EXCHANGE_COUNTS) == cap
    assert "s0" in reflection._EXCHANGE_COUNTS  # protected by the touch
    assert "s1" not in reflection._EXCHANGE_COUNTS  # evicted instead


async def test_schedule_reflection_rate_limits_actual_calls(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    calls: list[dict] = []

    async def complete(payload: dict) -> dict:
        calls.append(payload)
        return _text_response("none")

    for _ in range(6):  # six completed exchanges -> reflections on 1 and 4 only
        reflection.schedule_reflection(
            complete,
            _registry(memory),
            _ctx(tmp_path),
            model="bonsai-27b",
            sampling=SAMPLING,
            session_id="sess-r",
            exchange_text="user: hi\nassistant: hello",
        )
    await asyncio.sleep(0.05)
    assert len(calls) == 2
    memory.close()


# -- the reflection call itself ----------------------------------------------
async def test_reflect_once_payload_is_cheap_and_tool_scoped(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    seen: list[dict] = []

    async def complete(payload: dict) -> dict:
        seen.append(payload)
        return _text_response("none")

    await reflection.reflect_once(
        complete,
        _registry(memory),
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="user: x\nassistant: y",
    )
    payload = seen[0]
    # Thinking OFF, small max_tokens, and ONLY memory_write in the tool schema.
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["max_tokens"] == reflection.REFLECTION_MAX_TOKENS
    assert [t["function"]["name"] for t in payload["tools"]] == ["memory_write"]
    assert payload["messages"][0]["content"] == reflection.REFLECTION_SYSTEM_PROMPT
    memory.close()


async def test_none_answer_writes_nothing(tmp_path: Path) -> None:
    memory = _memory(tmp_path)

    async def complete(payload: dict) -> dict:
        return _text_response("none")

    await reflection.reflect_once(
        complete,
        _registry(memory),
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="user: x\nassistant: y",
    )
    entries, total = memory.store.list_entries(layer="archive")
    assert total == 0
    memory.close()


async def test_memory_write_call_persists_a_durable_fact(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    memory.store.ensure_session("sess-r", "chat")
    arguments = (
        '{"layer": "archive", "title": "Prefers metric units",'
        ' "content": "The user always wants metric units in answers."}'
    )

    async def complete(payload: dict) -> dict:
        return _tool_call_response(arguments)

    await reflection.reflect_once(
        complete,
        _registry(memory),
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="user: use metric please\nassistant: noted",
    )
    entries, total = memory.store.list_entries(layer="archive")
    assert total == 1
    assert entries[0].title == "Prefers metric units"
    assert entries[0].source_session == "sess-r"
    memory.close()


async def test_worker_role_write_is_refused_by_the_service(tmp_path: Path) -> None:
    """Defense in depth: even if a reflection somehow ran with a worker-role ctx,
    the memory service's one-writer check refuses the write."""
    memory = _memory(tmp_path)

    async def complete(payload: dict) -> dict:
        return _tool_call_response('{"layer": "archive", "title": "t", "content": "c"}')

    await reflection.reflect_once(
        complete,
        _registry(memory),
        _ctx(tmp_path, role="worker"),
        model="bonsai-8b",
        sampling=SAMPLING,
        exchange_text="user: x\nassistant: y",
    )
    _, total = memory.store.list_entries(layer="archive")
    assert total == 0
    memory.close()


async def test_bad_arguments_and_failures_never_raise(tmp_path: Path) -> None:
    memory = _memory(tmp_path)

    async def bad_json(payload: dict) -> dict:
        return _tool_call_response("{not json")

    await reflection.reflect_once(
        complete=bad_json,
        registry=_registry(memory),
        ctx=_ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="x",
    )
    _, total = memory.store.list_entries(layer="archive")
    assert total == 0

    async def boom(payload: dict) -> dict:
        raise RuntimeError("backend down")

    # The scheduled (safe) wrapper survives a crashing completion.
    reflection.schedule_reflection(
        boom,
        _registry(memory),
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        session_id="sess-crash",
        exchange_text="x",
    )
    await asyncio.sleep(0.05)
    memory.close()


# -- skill-validation retry (reject once, retry once, then give up) -----------
def _skill_call_response(name: str, arguments: dict) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-skill-1",
                            "type": "function",
                            "function": {
                                "name": "skill_save",
                                "arguments": json.dumps({"name": name, **arguments}),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


VALID_SKILL = "---\nname: tidy-commits\ndescription: how to write tidy commits\n---\n\n# steps\n"
INVALID_SKILL = "# no frontmatter at all"


async def test_skill_rejection_retries_once_with_validator_error(tmp_path: Path) -> None:
    """Invalid SKILL.md -> structured rejection -> ONE follow-up completion carrying
    the validator's message -> the corrected call is executed."""
    memory = _memory(tmp_path)
    registry = ToolRegistry([SkillSaveTool(memory)])
    payloads: list[dict] = []

    async def complete(payload: dict) -> dict:
        payloads.append(payload)
        if len(payloads) == 1:
            return _skill_call_response("tidy-commits", {"content": INVALID_SKILL})
        return _skill_call_response("tidy-commits", {"content": VALID_SKILL})

    await reflection.reflect_once(
        complete,
        registry,
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="user: x\nassistant: y",
    )
    assert len(payloads) == 2
    # The retry appends the failed assistant call plus its rejection as a tool msg.
    retry_messages = payloads[1]["messages"]
    assert retry_messages[-2]["tool_calls"][0]["id"] == "call-skill-1"
    assert retry_messages[-1]["role"] == "tool"
    assert retry_messages[-1]["content"].startswith(SKILL_REJECTION_PREFIX)
    saved = memory.skills.get("tidy-commits")
    assert saved is not None and saved.source == "learned"
    memory.close()


async def test_skill_rejection_gives_up_quietly_after_second_failure(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    registry = ToolRegistry([SkillSaveTool(memory)])
    payloads: list[dict] = []

    async def complete(payload: dict) -> dict:
        payloads.append(payload)
        return _skill_call_response("tidy-commits", {"content": INVALID_SKILL})

    await reflection.reflect_once(  # must not raise
        complete,
        registry,
        _ctx(tmp_path),
        model="bonsai-27b",
        sampling=SAMPLING,
        exchange_text="x",
    )
    assert len(payloads) == 2  # exactly one retry, never a third attempt
    assert memory.skills.get("tidy-commits") is None
    memory.close()


async def test_non_skill_errors_never_trigger_a_retry(tmp_path: Path) -> None:
    """A memory_write refused by role enforcement is an error a retry cannot fix —
    exactly ONE completion happens."""
    memory = _memory(tmp_path)
    payloads: list[dict] = []

    async def complete(payload: dict) -> dict:
        payloads.append(payload)
        return _tool_call_response('{"layer": "archive", "title": "t", "content": "c"}')

    await reflection.reflect_once(
        complete,
        _registry(memory),
        _ctx(tmp_path, role="worker"),
        model="bonsai-8b",
        sampling=SAMPLING,
        exchange_text="x",
    )
    assert len(payloads) == 1
    memory.close()


def test_exchange_digest_uses_latest_user_message_and_truncates() -> None:
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "latest question"},
    ]
    digest = reflection.exchange_digest(messages, "final answer")
    assert digest == "user: latest question\nassistant: final answer"
    long = reflection.exchange_digest([{"role": "user", "content": "x" * 10_000}], "y")
    assert len(long) == reflection.EXCHANGE_MAX_CHARS


# -- router gating (who reflects, who never does) -----------------------------
def _spy_scheduler(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    scheduled: list[dict] = []

    def spy(complete, registry, ctx, *, model, sampling, session_id, exchange_text) -> None:
        scheduled.append({"model": model, "session_id": session_id, "role": ctx.role})

    monkeypatch.setattr(reflection, "schedule_reflection", spy)
    return scheduled


def test_orchestrator_chat_and_code_schedule_reflection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled = _spy_scheduler(monkeypatch)
    for mode in ("chat", "code"):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "hello"}],
                "mode": mode,
                "session_id": f"sess-refl-{mode}",
            },
        )
        assert resp.status_code == 200
    assert [s["session_id"] for s in scheduled] == ["sess-refl-chat", "sess-refl-code"]
    assert all(s["model"] == "bonsai-27b" and s["role"] == "orchestrator" for s in scheduled)


def test_workers_ultra_and_anonymous_never_schedule_reflection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    scheduled = _spy_scheduler(monkeypatch)
    # Worker slot (bonsai-8b routes to a worker in the 24 GB fixture).
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-8b",
            "messages": [{"role": "user", "content": "x"}],
            "session_id": "sess-worker",
        },
    )
    # Ultra mode (orchestrator slot, but reflection is chat/code only).
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "x"}],
            "mode": "ultra",
            "effort": "low",
            "session_id": "sess-ultra",
        },
    )
    # Anonymous exchange (no session key to rate-limit on).
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "x"}]},
    )
    assert scheduled == []
