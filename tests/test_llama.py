"""llama layer: server flags, mock backend behavior, crash backoff, binary resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from suiban import paths
from suiban.config import KvSettings
from suiban.kv import resolve_kv_state
from suiban.llama.backend import MockBackend, build_server_flags, restart_backoff_s
from suiban.llama.binary import (
    PRISM_RELEASE_TAG,
    BinaryMissing,
    release_matches_pin,
    resolve_server_binary,
    turboquant_installed,
)
from suiban.sched.planner import PlannedSlot


def make_slot(**overrides) -> PlannedSlot:
    defaults = dict(
        slot_id="orchestrator",
        role="orchestrator",
        model="bonsai-27b",
        family="ternary",
        ctx=32768,
        gpu=0,
        port=8701,
        vram_mb=9536,
        mmproj=True,
        dspark=False,
    )
    defaults.update(overrides)
    return PlannedSlot(**defaults)


KV = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=True)


def test_flags_27b_full() -> None:
    flags = build_server_flags(
        make_slot(),
        KV,
        model_path=Path("/m/27b.gguf"),
        mmproj_path=Path("/m/mmproj.gguf"),
    )
    text = " ".join(flags)
    assert "--host 127.0.0.1" in text
    assert "--port 8701" in text
    assert "-c 32768" in text
    assert "-ngl 999" in text
    assert "--jinja" in text
    # Pinned fork's -fa REQUIRES a value (on|off|auto); quantized KV forces "on".
    assert "-fa on" in text
    assert flags[flags.index("-fa") + 1] == "on"
    assert "--cache-type-k q8_0" in text
    assert "--cache-type-v tq4_0" in text
    # Slot-wide thinking ceiling: min(xhigh 24576, 40% of ctx 32768) = 13107. The fork
    # ignores per-request budget fields, so this launch flag is the only hard cap.
    assert "--reasoning-budget 13107" in text
    # Log verbosity 4: the KV-cache summary line the hybrid-attention probe parses
    # is INFO-level and filtered at the fork's default verbosity 3.
    assert "-lv 4" in text
    assert "--mmproj /m/mmproj.gguf" in text
    assert "-md" not in flags


def test_flags_dspark_draft_model() -> None:
    flags = build_server_flags(
        make_slot(dspark=True),
        KV,
        model_path=Path("/m/27b.gguf"),
        mmproj_path=Path("/m/mmproj.gguf"),
        draft_model_path=Path("/m/draft.gguf"),
    )
    assert "-md" in flags
    assert flags[flags.index("-md") + 1] == "/m/draft.gguf"


def test_flags_cpu_slot_and_f16_no_fa() -> None:
    kv_f16 = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=False)
    flags = build_server_flags(
        make_slot(gpu=None, mmproj=False), kv_f16, model_path=Path("/m/27b.gguf")
    )
    text = " ".join(flags)
    assert "-ngl 0" in text
    assert "-fa" not in flags  # FA unusable -> f16/f16, no flag
    assert "--cache-type-v f16" in text


def test_flags_mmproj_required_when_slot_wants_it() -> None:
    with pytest.raises(ValueError):
        build_server_flags(make_slot(), KV, model_path=Path("/m/27b.gguf"))


def test_restart_backoff_progression() -> None:
    assert [restart_backoff_s(n) for n in range(6)] == [1, 2, 4, 8, 16, 30]
    assert restart_backoff_s(10) == 30  # capped


async def test_mock_backend_lifecycle_and_canned_completion() -> None:
    slot = make_slot(mmproj=False)
    backend = MockBackend(slot)
    await backend.start()
    assert slot.state == "ready"
    assert await backend.healthy()

    async with backend.client() as client:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "bonsai-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["content"].startswith("This is a deterministic")
        assert body["created"] == 0  # deterministic on purpose

        # same request -> same id (deterministic fingerprint)
        resp2 = await client.post(
            "/v1/chat/completions",
            json={"model": "bonsai-27b", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp2.json()["id"] == body["id"]

    await backend.stop()
    assert slot.state == "stopped"


async def test_mock_backend_streaming_is_openai_shaped() -> None:
    backend = MockBackend(make_slot(mmproj=False))
    await backend.start()
    async with (
        backend.client() as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "bonsai-27b",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as resp,
    ):
        assert resp.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in resp.aiter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[len("data: ") :]) for line in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "deterministic canned completion" in text
    await backend.stop()


def test_binary_resolution_missing_gives_remediation(bonsai_home: Path) -> None:
    with pytest.raises(BinaryMissing) as exc_info:
        resolve_server_binary("cuda")
    assert "suiban install binaries" in exc_info.value.message


def test_binary_markers(bonsai_home: Path) -> None:
    bin_dir = paths.bin_dir("cuda")
    bin_dir.mkdir(parents=True)
    (bin_dir / "llama-server").write_bytes(b"#!fake")
    assert resolve_server_binary("cuda").name == "llama-server"
    assert release_matches_pin("cuda") is False  # no RELEASE marker yet
    (bin_dir / "RELEASE").write_text(PRISM_RELEASE_TAG + "\n")
    assert release_matches_pin("cuda") is True
    assert turboquant_installed("cuda") is False
    (bin_dir / "TURBOQUANT").write_text("tq4_0 tq3_0\n")
    assert turboquant_installed("cuda") is True
