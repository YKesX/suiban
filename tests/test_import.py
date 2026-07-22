"""Chat import (api.md 2026-07-22b): the memory/importers.py parsers per provider, and
the POST /v1/memory/sessions/import endpoint (create archived sessions, mode filter,
compress-to-seed, and the 400 import_unrecognized shape guard). No network — the
compress path drives the mock utility slot."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from suiban.memory import importers
from suiban.memory.compression import SUMMARY_PREFIX

# -- provider export fixtures (representative shapes) -------------------------
GENERIC = {
    "title": "My chat",
    "messages": [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi, how can I help?"},
    ],
}

OPENAI = [
    {
        "title": "OpenAI export",
        "mapping": {
            "root": {"id": "root", "message": None, "parent": None, "children": ["a"]},
            "b": {
                "id": "b",
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "text", "parts": ["4"]},
                    "create_time": 2.0,
                },
                "parent": "a",
                "children": [],
            },
            "a": {
                "id": "a",
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["what is 2+2"]},
                    "create_time": 1.0,
                },
                "parent": "root",
                "children": ["b"],
            },
        },
    }
]

CLAUDE = [
    {
        "name": "Claude chat",
        "chat_messages": [
            {"sender": "human", "text": "hey claude"},
            {"sender": "assistant", "text": "hello!"},
        ],
    }
]

CLAUDE_CODE = (
    '{"role": "user", "content": "fix the failing test"}\n'
    '{"role": "assistant", "content": "found it: an off-by-one in the loop"}\n'
)


# -- parser unit tests --------------------------------------------------------
def test_generic_parser() -> None:
    [session] = importers.parse_import("generic", GENERIC)
    assert session.title == "My chat"
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert session.messages[0]["content"] == "hello there"


def test_openai_parser_orders_by_create_time_and_skips_empty_root() -> None:
    [session] = importers.parse_import("openai", OPENAI)
    assert session.title == "OpenAI export"
    # Ordered by create_time despite the out-of-order mapping; the empty root is dropped.
    assert [m["content"] for m in session.messages] == ["what is 2+2", "4"]


def test_claude_parser_maps_human_to_user() -> None:
    [session] = importers.parse_import("claude", CLAUDE)
    assert session.title == "Claude chat"
    assert session.messages[0] == {"role": "user", "content": "hey claude"}
    assert session.messages[1]["role"] == "assistant"


def test_claude_code_parser_reads_jsonl() -> None:
    [session] = importers.parse_import("claude-code", CLAUDE_CODE)
    assert session.title is None
    assert [m["role"] for m in session.messages] == ["user", "assistant"]
    assert "off-by-one" in session.messages[1]["content"]


@pytest.mark.parametrize(
    ("provider", "data"),
    [
        ("generic", {"nope": 1}),
        ("openai", {"totally": "wrong"}),
        ("claude", [{"no_chat_messages": True}]),
        ("claude-code", 12345),
        ("claude-code", "not json at all"),
    ],
)
def test_shape_mismatch_raises_import_unrecognized(provider: str, data: object) -> None:
    with pytest.raises(importers.ImportUnrecognized):
        importers.parse_import(provider, data)


# -- HTTP endpoint ------------------------------------------------------------
def _import(client: TestClient, body: dict):
    return client.post("/v1/memory/sessions/import", json=body)


def test_import_creates_archived_sessions(client: TestClient) -> None:
    resp = _import(client, {"provider": "generic", "data": GENERIC})
    assert resp.status_code == 200
    imported = resp.json()["imported"]
    assert len(imported) == 1
    row = imported[0]
    assert set(row) == {"id", "title", "message_count"}
    assert row["title"] == "My chat"
    assert row["message_count"] == 2

    # The session is now in the archive and restorable.
    transcript = client.get(f"/v1/memory/sessions/{row['id']}").json()
    assert transcript["session"]["mode"] == "chat"
    assert [m["role"] for m in transcript["messages"]] == ["user", "assistant"]


def test_import_honors_mode_and_lists_under_it(client: TestClient) -> None:
    row = _import(client, {"provider": "claude", "data": CLAUDE, "mode": "code"}).json()[
        "imported"
    ][0]
    sessions = client.get("/v1/memory/sessions", params={"mode": "code"}).json()["sessions"]
    assert row["id"] in {s["id"] for s in sessions}
    assert row["id"] not in {
        s["id"]
        for s in client.get("/v1/memory/sessions", params={"mode": "chat"}).json()["sessions"]
    }


def test_import_compress_condenses_to_a_seed_summary(client: TestClient) -> None:
    row = _import(client, {"provider": "generic", "data": GENERIC, "compress": True}).json()[
        "imported"
    ][0]
    # Compression folds the transcript into ONE seed summary message.
    assert row["message_count"] == 1
    transcript = client.get(f"/v1/memory/sessions/{row['id']}").json()
    assert len(transcript["messages"]) == 1
    assert transcript["messages"][0]["content"].startswith(SUMMARY_PREFIX)


def test_import_derives_a_title_when_absent(client: TestClient) -> None:
    row = _import(client, {"provider": "claude-code", "data": CLAUDE_CODE}).json()["imported"][0]
    assert row["title"] == "fix the failing test"  # first user line


def test_import_unrecognized_is_400(client: TestClient) -> None:
    resp = _import(client, {"provider": "openai", "data": {"not": "an export"}})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "import_unrecognized"


@pytest.mark.parametrize(
    "body",
    [
        {"data": GENERIC},  # missing provider
        {"provider": "aol", "data": GENERIC},  # unknown provider
        {"provider": "generic"},  # missing data
        {"provider": "generic", "data": GENERIC, "mode": "ultra"},  # bad mode
        {"provider": "generic", "data": GENERIC, "compress": "yes"},  # bad compress
    ],
)
def test_import_validation_400s(client: TestClient, body: dict) -> None:
    resp = _import(client, body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] in ("validation_error", "import_unrecognized")
