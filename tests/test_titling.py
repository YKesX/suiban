"""Session auto-titling: title cleanup rules, the only-when-null guard, failure
tolerance, and the end-to-end background path on the mock backend (chat, code,
ultra)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban.llama.mock_server import MOCK_SESSION_TITLE
from suiban.memory.store import MemoryStore
from suiban.memory.titling import clean_title, maybe_title_session, title_input


# -- clean_title rules --------------------------------------------------------
def test_clean_title_strips_quotes_punctuation_and_whitespace() -> None:
    assert clean_title('"Fixing the Zeppelin Refactor."') == "Fixing the Zeppelin Refactor"
    assert clean_title("  spaced   out\ntitle!  ") == "spaced out title"
    assert clean_title("'single quoted…?'") == "single quoted…"
    assert clean_title("“smart quotes”") == "smart quotes"


def test_clean_title_clips_to_six_words() -> None:
    assert clean_title("one two three four five six seven eight") == "one two three four five six"


def test_clean_title_empty_input_stays_empty() -> None:
    assert clean_title("   ") == ""
    assert clean_title('"...."') == ""


def test_title_input_uses_first_exchange_only() -> None:
    messages = [
        {"role": "user", "content": "first question", "created_at": "t"},
        {"role": "assistant", "content": "first answer", "created_at": "t"},
        {"role": "user", "content": "second question", "created_at": "t"},
    ]
    text = title_input(messages)
    assert "first question" in text and "first answer" in text
    assert "second question" not in text


# -- maybe_title_session ------------------------------------------------------
@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.sqlite")
    yield s
    s.close()


def _seed_session(store: MemoryStore, session_id: str = "sess-t") -> None:
    store.ensure_session(session_id, "chat")
    store.add_message(session_id, "user", "how do I prune a bonsai?")
    store.add_message(session_id, "assistant", "carefully, with clean shears")


async def test_titles_a_null_titled_session(store: MemoryStore) -> None:
    _seed_session(store)
    seen: list[str] = []

    async def generate(text: str) -> str:
        seen.append(text)
        return '"Pruning a Bonsai Tree."'

    await maybe_title_session(store, generate, "sess-t")
    transcript = store.session_transcript("sess-t")
    assert transcript is not None
    assert transcript["session"]["title"] == "Pruning a Bonsai Tree"
    assert "how do I prune" in seen[0]


async def test_only_titles_when_title_is_null(store: MemoryStore) -> None:
    _seed_session(store)
    store.set_session_title("sess-t", "Existing Title")

    async def generate(text: str) -> str:  # pragma: no cover - must not be called
        raise AssertionError("generate must not run for an already-titled session")

    await maybe_title_session(store, generate, "sess-t")
    assert store.session_transcript("sess-t")["session"]["title"] == "Existing Title"


async def test_failure_leaves_title_null_and_never_raises(store: MemoryStore) -> None:
    _seed_session(store)

    async def generate(text: str) -> str:
        raise RuntimeError("utility slot went away")

    await maybe_title_session(store, generate, "sess-t")  # must not raise
    assert store.session_transcript("sess-t")["session"]["title"] is None

    # An all-punctuation reply cleans to empty: title stays null too.
    async def garbage(text: str) -> str:
        return '"!!!"'

    await maybe_title_session(store, garbage, "sess-t")
    assert store.session_transcript("sess-t")["session"]["title"] is None


async def test_unknown_or_empty_sessions_are_noops(store: MemoryStore) -> None:
    async def generate(text: str) -> str:  # pragma: no cover - must not be called
        raise AssertionError("generate must not run without a first exchange")

    await maybe_title_session(store, generate, "sess-missing")
    store.ensure_session("sess-empty", "chat")
    await maybe_title_session(store, generate, "sess-empty")
    assert store.session_transcript("sess-empty")["session"]["title"] is None


# -- end to end on the mock backend ------------------------------------------
def _wait_for_title(client: TestClient, session_id: str, deadline_s: float = 10.0) -> str:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        title = client.get(f"/v1/memory/sessions/{session_id}").json()["session"]["title"]
        if title:
            return title
        time.sleep(0.02)
    raise AssertionError(f"session {session_id} never got a title")


@pytest.mark.parametrize("mode", ["chat", "code", "ultra"])
def test_first_exchange_titles_the_session(client: TestClient, mode: str) -> None:
    session_id = f"sess-title-{mode}"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hello there"}],
            "mode": mode,
            "effort": "low",
            "session_id": session_id,
        },
    )
    assert resp.status_code == 200
    assert _wait_for_title(client, session_id) == MOCK_SESSION_TITLE


def test_anonymous_chats_never_title(client: TestClient) -> None:
    client.post(
        "/v1/chat/completions",
        json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "no session"}]},
    )
    assert client.get("/v1/memory/sessions").json()["sessions"] == []
