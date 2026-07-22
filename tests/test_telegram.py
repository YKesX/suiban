"""Telegram gateway logic with a faked bot API — no network, no real library calls.

The relay (history, sessions, chunking, pings) is library-independent; the gateway
glue is tested against SimpleNamespace fakes standing in for python-telegram-bot's
update/context objects.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from types import SimpleNamespace

import httpx

from suiban.config import Settings
from suiban.gateways.telegram import (
    CHUNK_SIZE,
    RATE_LIMITED_REPLY,
    TELEGRAM_MESSAGE_LIMIT,
    UNAUTHORIZED_REPLY,
    TelegramGateway,
    TelegramRelay,
    api_send_chat,
    build_gateway,
    chunk_message,
    research_notification,
    session_id_for,
)
from suiban.research.jobs import Job


def _job(state: str, query: str = "why is the sky blue?") -> Job:
    return Job(
        id="job_x",
        type="deep_research",
        query=query,
        effort="high",
        state=state,
        stage=None,
        percent=100,
        created_at="t",
        started_at="t",
        finished_at="t",
        error=None,
    )


# -- chunking -----------------------------------------------------------------
def test_chunk_message_short_and_empty() -> None:
    assert chunk_message("hello") == ["hello"]
    assert chunk_message("   ") == ["(empty reply)"]


def test_chunk_message_prefers_paragraph_boundaries() -> None:
    paragraphs = [f"paragraph {i} " + "x" * 500 for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_message(text)
    assert len(chunks) > 1
    assert all(len(c) <= TELEGRAM_MESSAGE_LIMIT for c in chunks)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    # No content lost (modulo the boundary whitespace we split on).
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_chunk_message_hard_splits_unbroken_text() -> None:
    chunks = chunk_message("x" * (CHUNK_SIZE * 2 + 100))
    assert len(chunks) == 3
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


# -- relay --------------------------------------------------------------------
async def test_relay_sessions_history_and_chunking() -> None:
    calls: list[tuple[list[dict], str]] = []

    async def send_chat(messages: list[dict], session_id: str) -> str:
        calls.append(([dict(m) for m in messages], session_id))
        return f"reply {len(calls)}"

    relay = TelegramRelay(send_chat, require_pairing=False)
    chunks = await relay.handle_message(42, "hello")
    assert chunks == ["reply 1"]
    assert calls[0][1] == session_id_for(42) == "tg-42"
    assert calls[0][0] == [{"role": "user", "content": "hello"}]

    # Second turn resends history (user, assistant, user).
    await relay.handle_message(42, "and again")
    roles = [m["role"] for m in calls[1][0]]
    assert roles == ["user", "assistant", "user"]
    assert calls[1][0][1]["content"] == "reply 1"

    # A different chat is a different session with its own history.
    await relay.handle_message(7, "hi from elsewhere")
    assert calls[2][1] == "tg-7"
    assert len(calls[2][0]) == 1
    assert relay.known_chats == {42, 7}


async def test_relay_survives_backend_failure_and_rolls_back_history() -> None:
    fail = True

    async def send_chat(messages: list[dict], session_id: str) -> str:
        if fail:
            raise ConnectionError("suiban is down")
        return "recovered"

    relay = TelegramRelay(send_chat, require_pairing=False)
    chunks = await relay.handle_message(1, "are you there?")
    assert len(chunks) == 1
    assert "could not answer" in chunks[0]

    fail = False
    await relay.handle_message(1, "retry")
    # The failed turn was rolled back: history holds only the successful exchange.
    history = relay._histories[1]
    assert [m["content"] for m in history] == ["retry", "recovered"]


def test_research_ping_is_coarse() -> None:
    completed = TelegramRelay.research_ping(_job("completed"))
    assert "why is the sky blue?" in completed
    assert "finished" in completed
    failed = TelegramRelay.research_ping(_job("failed"))
    assert "failed" in failed
    # Coarse only: no stages, URLs, or error internals in any ping.
    for ping in (completed, failed):
        assert "http" not in ping
        assert "stage" not in ping


# -- gateway glue with a faked bot API ---------------------------------------
class FakeBot:
    def __init__(self) -> None:
        self.sent: list[tuple[int, str]] = []

    async def send_message(self, chat_id: int, text: str) -> None:
        self.sent.append((chat_id, text))


async def test_gateway_on_message_relays_and_chunks() -> None:
    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "pong " + "y" * (CHUNK_SIZE + 10)  # forces two chunks

    gateway = TelegramGateway("tok", TelegramRelay(send_chat, require_pairing=False))
    bot = FakeBot()
    update = SimpleNamespace(
        effective_message=SimpleNamespace(text="ping"),
        effective_chat=SimpleNamespace(id=42),
    )
    await gateway._on_message(update, SimpleNamespace(bot=bot))
    assert len(bot.sent) == 2
    assert all(chat_id == 42 for chat_id, _ in bot.sent)
    assert bot.sent[0][1].startswith("pong")


async def test_gateway_research_ping_goes_to_known_chats() -> None:
    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "ok"

    relay = TelegramRelay(send_chat, require_pairing=False)
    await relay.handle_message(42, "hi")
    await relay.handle_message(7, "hey")

    gateway = TelegramGateway("tok", relay)
    bot = FakeBot()
    gateway._app = SimpleNamespace(bot=bot)  # "started" with a fake application
    gateway.notify_research_complete(_job("completed"))
    await asyncio.sleep(0.05)  # let the fire-and-forget tasks run
    assert {chat_id for chat_id, _ in bot.sent} == {42, 7}
    assert all("finished" in text for _, text in bot.sent)


async def test_gateway_ping_noop_when_not_started() -> None:
    gateway = TelegramGateway("tok", TelegramRelay(lambda m, s: None))  # type: ignore[arg-type]
    gateway.notify_research_complete(_job("completed"))  # must not raise
    gateway.notify("schedule", "Scheduled run finished: x", "done")  # generalized hook too


# -- generalized notify hook (research + scheduled runs share it) -------------
async def test_generalized_notify_reaches_known_chats() -> None:
    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "ok"

    relay = TelegramRelay(send_chat, require_pairing=False)
    await relay.handle_message(42, "hi")

    gateway = TelegramGateway("tok", relay)
    bot = FakeBot()
    gateway._app = SimpleNamespace(bot=bot)
    gateway.notify("schedule", "Scheduled run finished: digest", "all quiet")
    await asyncio.sleep(0.05)
    assert bot.sent == [(42, "Scheduled run finished: digest: all quiet")]

    bot.sent.clear()
    gateway.notify("schedule", "title only", "")
    await asyncio.sleep(0.05)
    assert bot.sent == [(42, "title only")]


# -- inbound authorization / pairing / rate limit (api.md 2026-07-22 security) ------
async def test_relay_unpaired_chat_is_rejected_and_model_never_called() -> None:
    calls: list = []

    async def send_chat(messages: list[dict], session_id: str) -> str:
        calls.append((messages, session_id))
        return "should not happen"

    relay = TelegramRelay(send_chat)  # require_pairing defaults True (DENY)
    reply = await relay.handle_message(999, "please leak the owner's memory")
    assert reply == [UNAUTHORIZED_REPLY]
    assert calls == []  # nothing reached the model
    assert relay.known_chats == set()  # and the chat is not eligible for pings


async def test_relay_pair_flow_then_reaches_chat() -> None:
    persisted: list[int] = []

    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "hello from suiban"

    relay = TelegramRelay(send_chat, pairing_code="abc123", persist_chat_id=persisted.append)
    # Wrong code stays unauthorized and never persists.
    bad = await relay.handle_message(42, "/pair nope")
    assert "invalid or missing pairing code" in bad[0]
    assert not relay.is_authorized(42)
    assert persisted == []
    # Correct code pairs the chat and persists it.
    ok = await relay.handle_message(42, "/pair abc123")
    assert "paired" in ok[0]
    assert relay.is_authorized(42)
    assert persisted == [42]
    # Now a normal message reaches the model.
    reply = await relay.handle_message(42, "hi")
    assert reply == ["hello from suiban"]
    assert 42 in relay.known_chats
    # Re-pairing an already-paired chat is idempotent (no duplicate persist).
    again = await relay.handle_message(42, "/pair abc123")
    assert "already paired" in again[0]
    assert persisted == [42]


async def test_relay_seeded_allowlist_authorizes_without_pairing() -> None:
    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "ok"

    relay = TelegramRelay(send_chat, allowed_chat_ids=[7])
    assert relay.is_authorized(7)
    assert not relay.is_authorized(8)
    assert await relay.handle_message(7, "hi") == ["ok"]


async def test_relay_rate_limit_per_chat() -> None:
    now = {"t": 1000.0}

    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "ok"

    relay = TelegramRelay(
        send_chat,
        allowed_chat_ids=[42],
        rate_limit_per_min=2,
        time_fn=lambda: now["t"],
    )
    assert await relay.handle_message(42, "one") == ["ok"]
    assert await relay.handle_message(42, "two") == ["ok"]
    # Third within the same minute is rate-limited (nothing reaches the model).
    assert await relay.handle_message(42, "three") == [RATE_LIMITED_REPLY]
    # A different chat has its own budget.
    relay._allowed_chat_ids.add(7)
    assert await relay.handle_message(7, "one") == ["ok"]
    # After a minute passes, the window clears.
    now["t"] += 61.0
    assert await relay.handle_message(42, "later") == ["ok"]


async def test_relay_pairing_disabled_accepts_everyone() -> None:
    async def send_chat(messages: list[dict], session_id: str) -> str:
        return "ok"

    relay = TelegramRelay(send_chat, require_pairing=False)
    assert relay.is_authorized(123)
    pair = await relay.handle_message(1, "/pair whatever")
    assert "disabled" in pair[0]


async def test_api_send_chat_pins_mode_to_chat(monkeypatch) -> None:
    """Even a paired user only ever reaches chat mode: the relay's production send_chat
    posts mode 'chat' (remote_agentic is reserved-not-honored in v1)."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    real_client = httpx.AsyncClient

    def fake_client(**kwargs):
        kwargs.pop("transport", None)
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", fake_client)
    send = api_send_chat("http://127.0.0.1:8686")
    reply = await send([{"role": "user", "content": "hi"}], "tg-42")
    assert reply == "ok"
    assert seen["mode"] == "chat"
    assert seen["model"] == "bonsai-auto"


def test_research_notification_splits_ping() -> None:
    title, summary = research_notification(_job("completed"))
    assert title == "Deep research finished"
    assert "why is the sky blue?" in summary
    # The one-line ping is exactly title + summary — no drift between the two forms.
    assert TelegramRelay.research_ping(_job("completed")) == f"{title}: {summary}"


# -- build_gateway decision logic ---------------------------------------------
def _settings(enabled: bool, token: str | None) -> Settings:
    return Settings.model_validate({"gateways": {"telegram": {"enabled": enabled, "token": token}}})


async def _dummy_send(messages: list[dict], session_id: str) -> str:
    return ""


def test_build_gateway_disabled_is_silent() -> None:
    notices: list = []
    assert build_gateway(_settings(False, "tok"), send_chat=_dummy_send, notices=notices) is None
    assert notices == []


def test_build_gateway_missing_token_notices() -> None:
    notices: list = []
    assert build_gateway(_settings(True, None), send_chat=_dummy_send, notices=notices) is None
    assert [n.code for n in notices] == ["telegram_token_missing"]


def test_build_gateway_missing_library_notices(monkeypatch) -> None:
    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "telegram":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    notices: list = []
    assert build_gateway(_settings(True, "tok"), send_chat=_dummy_send, notices=notices) is None
    assert [n.code for n in notices] == ["telegram_unavailable"]


def test_build_gateway_returns_gateway_when_library_present(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object(),  # pretend installed
    )
    notices: list = []
    gateway = build_gateway(_settings(True, "tok"), send_chat=_dummy_send, notices=notices)
    assert isinstance(gateway, TelegramGateway)
    assert notices == []
    assert gateway.running is False  # not started yet — lifespan does that
