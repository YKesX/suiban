"""Overflow guard (memory/injection.py): the trim ladder — injected blocks first
(lowest-scored last blocks first, memory recall before skills before project docs),
then the oldest unprotected tail — plus the `context_trimmed` notice over HTTP."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from suiban.memory import injection as inj
from suiban.memory.compression import estimate_tokens, keep_recent_messages


def _msg(role: str, chars: int, text: str = "x") -> dict:
    return {"role": role, "content": (text * chars)[:chars]}


def _injected(header: str, blocks: list[str]) -> dict:
    return {"role": "system", "content": header + "\n" + "\n".join(blocks)}


def _block(kind: str, label: str, body: str) -> str:
    return f"<<<{kind} {label}>>>\n{body}\n<<<end {kind}>>>"


def test_no_trim_when_under_limit() -> None:
    messages = [_msg("system", 100), _msg("user", 100)]
    out, report = inj.enforce_context_budget(messages, 8192)
    assert report is None
    assert out is messages  # untouched, not even copied


def test_injected_blocks_drop_lowest_scored_first() -> None:
    """Blocks inside an injection are bm25-ordered best-first; the guard pops from
    the END, so the best block survives the longest."""
    blocks = [
        _block("memory", "mem_best (archive) t", "BEST " * 100),
        _block("memory", "mem_mid (archive) t", "MID " * 100),
        _block("memory", "mem_worst (archive) t", "WORST " * 100),
    ]
    # ctx tuned so dropping the two worst blocks is enough (drop 1 -> 1021 tokens, still
    # over; drop 2 -> 908 tokens, fits under the 945 limit).
    messages = [_injected(inj.MEMORY_CONTEXT_HEADER, blocks), _msg("user", 2600)]
    slot_ctx = 1050  # limit 945 tokens
    out, report = inj.enforce_context_budget(messages, slot_ctx)
    assert report is not None
    assert report.dropped_blocks == 2
    assert report.dropped_messages == 0
    assert not report.still_over
    content = out[0]["content"]
    assert "mem_best" in content
    assert "mem_worst" not in content and "mem_mid" not in content
    assert content.startswith(inj.MEMORY_CONTEXT_HEADER)
    assert out[1]["content"] == messages[1]["content"]  # tail untouched


def test_trim_ladder_memory_then_skills_then_project_then_tail() -> None:
    memory_msg = _injected(
        inj.MEMORY_CONTEXT_HEADER, [_block("memory", "mem_1 (archive) t", "M " * 200)]
    )
    skill_msg = _injected(inj.SKILL_CONTEXT_HEADER, [_block("skill", "some-skill v1", "S " * 200)])
    project_msg = _injected(inj.PROJECT_CONTEXT_HEADER, [_block("doc", "spec", "P " * 200)])
    head = {"role": "system", "content": "mode prompt"}
    old_tail = [_msg("user", 400, "old-a "), _msg("assistant", 400, "old-b ")]
    recent = [_msg("user", 200, "recent ") for _ in range(4)]
    messages = [memory_msg, skill_msg, project_msg, head, *old_tail, *recent]

    # A limit low enough that every injection AND the pre-window tail must go.
    slot_ctx = 320  # limit 288 tokens
    out, report = inj.enforce_context_budget(messages, slot_ctx)
    assert report is not None
    contents = [m["content"] for m in out]
    assert not any(c.startswith(inj.MEMORY_CONTEXT_HEADER) for c in contents)
    assert not any(c.startswith(inj.SKILL_CONTEXT_HEADER) for c in contents)
    assert not any(c.startswith(inj.PROJECT_CONTEXT_HEADER) for c in contents)
    assert report.dropped_injections == 3
    assert report.dropped_messages == 2  # both old tail messages
    # The protected window and the system head survive.
    assert "mode prompt" in contents
    assert sum(1 for m in out if m["role"] != "system") == 4
    assert estimate_tokens(out) <= report.limit_tokens
    assert not report.still_over


def test_partial_ladder_stops_as_soon_as_it_fits() -> None:
    """When dropping the memory injection alone suffices, skills/project stay."""
    memory_msg = _injected(
        inj.MEMORY_CONTEXT_HEADER, [_block("memory", "mem_1 (archive) t", "M " * 400)]
    )
    project_msg = _injected(inj.PROJECT_CONTEXT_HEADER, [_block("doc", "spec", "P " * 20)])
    messages = [memory_msg, project_msg, _msg("user", 600)]
    out, report = inj.enforce_context_budget(messages, 400)
    assert report is not None
    contents = [m["content"] for m in out]
    assert any(c.startswith(inj.PROJECT_CONTEXT_HEADER) for c in contents)
    assert not any(c.startswith(inj.MEMORY_CONTEXT_HEADER) for c in contents)
    assert report.dropped_messages == 0


def test_still_over_when_only_protected_window_remains() -> None:
    """A giant protected tail cannot be trimmed further — the report says so
    honestly instead of pretending the request fits."""
    messages = [_msg("user", 40_000) for _ in range(4)]
    out, report = inj.enforce_context_budget(messages, 8192)
    assert report is not None
    assert report.still_over
    assert len(out) == 4  # protected window kept
    assert "still over" in report.describe()


# -- single-pass equivalence (audit 2026-07-22 perf rewrite) -----------------
def _reference_enforce(messages: list[dict], slot_ctx: int):
    """The pre-audit O(M^2) enforce_context_budget, verbatim — re-estimates the whole
    message list after every popped block and every deleted message. The single-pass
    rewrite must produce byte-identical output and an identical TrimReport."""
    limit = int(slot_ctx * inj.HARD_LIMIT_FRACTION)
    tokens_before = estimate_tokens(messages)
    if slot_ctx <= 0 or tokens_before <= limit:
        return messages, None
    out = [dict(m) for m in messages]
    dropped_blocks = dropped_injections = dropped_messages = 0
    for header in inj.TRIM_ORDER:
        index = next((i for i, m in enumerate(out) if inj._injected_header(m) == header), None)
        if index is None:
            continue
        head, blocks = inj._split_injected(out[index]["content"])
        while blocks and estimate_tokens(out) > limit:
            blocks.pop()
            dropped_blocks += 1
            if blocks:
                out[index]["content"] = head + "\n" + "\n".join(blocks)
            else:
                del out[index]
                dropped_injections += 1
                break
        if estimate_tokens(out) <= limit:
            break
    keep = keep_recent_messages(slot_ctx)
    while estimate_tokens(out) > limit:
        body_indexes = [i for i, m in enumerate(out) if m.get("role") != "system"]
        if len(body_indexes) <= keep:
            break
        del out[body_indexes[0]]
        dropped_messages += 1
    tokens_after = estimate_tokens(out)
    return out, inj.TrimReport(
        dropped_blocks=dropped_blocks,
        dropped_injections=dropped_injections,
        dropped_messages=dropped_messages,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        limit_tokens=limit,
        still_over=tokens_after > limit,
    )


def _synth_conversation(n_body: int) -> list[dict]:
    mem = (
        inj.MEMORY_CONTEXT_HEADER
        + "\n"
        + "\n".join(_block("memory", f"m{i} (archive) note", "recall " * 40) for i in range(6))
    )
    skills = (
        inj.SKILL_CONTEXT_HEADER
        + "\n"
        + "\n".join(_block("skill", f"skill-{i} v1", "procedure " * 40) for i in range(2))
    )
    project = (
        inj.PROJECT_CONTEXT_HEADER
        + "\n"
        + "\n".join(_block("doc", f"spec-{i}", "excerpt " * 40) for i in range(3))
    )
    messages = [
        {"role": "system", "content": "mode prompt " * 20},
        {"role": "system", "content": mem},
        {"role": "system", "content": skills},
        {"role": "system", "content": project},
    ]
    for i in range(n_body):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"turn {i}: " + ("lorem ipsum dolor " * 30)})
    return messages


@pytest.mark.parametrize("slot_ctx", [400, 2048, 8192, 65536, 200000])
def test_single_pass_matches_reference_on_400_message_conversation(slot_ctx: int) -> None:
    """The single-pass guard is byte-for-byte equal to the O(M^2) reference across the
    whole trim ladder (both rungs) and the no-trim early return, on a 400-message
    conversation — at ctx values that trim everything, some, and nothing."""
    messages = _synth_conversation(400)
    ref_out, ref_report = _reference_enforce(messages, slot_ctx)
    new_out, new_report = inj.enforce_context_budget(messages, slot_ctx)
    assert [m["content"] for m in new_out] == [m["content"] for m in ref_out]
    assert [m.get("role") for m in new_out] == [m.get("role") for m in ref_out]
    assert new_report == ref_report
    # The running total the single pass maintains must equal a fresh full estimate.
    if new_report is not None:
        assert new_report.tokens_after == estimate_tokens(new_out)


def test_single_pass_does_not_mutate_input() -> None:
    """Trimming copies before editing: the caller's list and its dicts are untouched."""
    messages = _synth_conversation(400)
    before = [dict(m) for m in messages]
    inj.enforce_context_budget(messages, 2048)
    assert messages == before


def test_context_trimmed_notice_reaches_the_stream(client: TestClient) -> None:
    """End-to-end: an oversized protected tail triggers the guard inside
    _prepare_loop and the stream carries a context_trimmed notice."""
    # 9 large user turns: compression protects the last 8 (32K ctx) which alone
    # exceed 90% of the slot context, so the guard must speak.
    messages = [
        {"role": "user", "content": f"turn {i}: " + ("lorem ipsum " * 12_000)} for i in range(9)
    ]
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": messages,
            "stream": True,
            "stream_events": True,
            "effort": "low",
        },
    ) as response:
        assert response.status_code == 200
        events = []
        for line in response.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: ") :]))
    notices = [
        e for e in events if e.get("type") == "notice" and e.get("code") == "context_trimmed"
    ]
    assert notices, f"no context_trimmed notice in {[e.get('type') for e in events]}"
    assert notices[0]["level"] == "warn"
    assert "context estimate" in notices[0]["message"]
