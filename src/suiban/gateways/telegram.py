"""Telegram gateway: long-polling chat relay + research-complete pings.

Transport rules (v1, plan-frozen):
- LONG POLLING ONLY — outbound HTTPS to api.telegram.org, no webhooks, no open ports.
- The bot token is a write-only secret from ~/.bonsai/config.toml
  (`[gateways.telegram]`); /v1/settings never echoes it (`token_set` only).
- python-telegram-bot is an optional extra (`suiban[gateways]`); when it is missing
  the gateway declines to start with a notice — never a crash.

The relay is an ordinary API client: each incoming message becomes a
POST /v1/chat/completions against the local server with a per-chat session id
(`tg-<chat_id>`), so gateway conversations get the same memory/session treatment as
any other client and the frozen contract stays the only coordination point. Replies
are non-streamed and sent as chunked messages (Telegram's 4096-char limit) — simple
and robust over streamed message-editing, which is a v1.1 nicety.
TODO(v1.1): streamed replies via edited-message updates, rate-limit aware.

Per-chat history is kept in memory (last HISTORY_LIMIT turns) and resent with each
request, OpenAI-style; the archive layer de-duplicates. TODO(v1.1): rebuild gateway
history from /v1/memory/sessions/{id} after a restart instead of starting blank.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import secrets
import time
from collections import deque
from collections.abc import Awaitable, Callable

import httpx

from suiban.config import Settings
from suiban.sched.planner import Notice

logger = logging.getLogger(__name__)

TELEGRAM_MESSAGE_LIMIT = 4096
CHUNK_SIZE = 3900  # margin under the hard limit for markdown fences etc.
HISTORY_LIMIT = 20  # messages (user + assistant) resent per chat
CHAT_TIMEOUT_S = 300.0
DEFAULT_RATE_LIMIT_PER_MIN = 20

# Inbound-authorization replies (api.md 2026-07-22 security). An unpaired chat gets
# exactly ONE of these and nothing reaches the model.
PAIR_COMMAND = "/pair"
UNAUTHORIZED_REPLY = (
    "not authorized. Run '/pair <code>' with the one-time code printed in the suiban "
    "server console to link this chat."
)
RATE_LIMITED_REPLY = "rate limit reached (too many messages this minute). Try again shortly."

# (messages, session_id) -> assistant reply text
SendChatFn = Callable[[list[dict], str], Awaitable[str]]
# Persist a newly-paired chat id (ConfigManager.add_telegram_chat_id in production).
PersistChatFn = Callable[[int], None]


def _is_pair_command(text: str) -> bool:
    """True when `text` is a `/pair` command (tolerating `/pair@botname` group syntax)."""
    first = text.split(maxsplit=1)[0] if text else ""
    return first.split("@", 1)[0].lower() == PAIR_COMMAND


def chunk_message(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split a reply into Telegram-sized chunks, preferring paragraph then line
    boundaries so code blocks and lists break as cleanly as possible."""
    text = text.strip() or "(empty reply)"
    chunks: list[str] = []
    while len(text) > size:
        cut = text.rfind("\n\n", 0, size)
        if cut < size // 2:
            cut = text.rfind("\n", 0, size)
        if cut < size // 2:
            cut = size
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def session_id_for(chat_id: int) -> str:
    return f"tg-{chat_id}"


def research_notification(job) -> tuple[str, str]:
    """(title, summary) for the generalized notify hook — coarse only: job state +
    the user's own query, nothing internal (no stages, no sources, no URLs)."""
    if job.state == "completed":
        return "Deep research finished", f"{job.query!r}. Read the report in dai, or: sentei jobs"
    return f"Deep research {job.state}", f"{job.query!r}."


class TelegramRelay:
    """Library-independent core: history, sessions, chunking, ping text. All the
    logic that deserves tests lives here; TelegramGateway is transport glue."""

    def __init__(
        self,
        send_chat: SendChatFn,
        *,
        history_limit: int = HISTORY_LIMIT,
        allowed_chat_ids: list[int] | set[int] | None = None,
        require_pairing: bool = True,
        rate_limit_per_min: int = DEFAULT_RATE_LIMIT_PER_MIN,
        persist_chat_id: PersistChatFn | None = None,
        pairing_code: str | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._send_chat = send_chat
        self._history_limit = history_limit
        self._histories: dict[int, deque[dict]] = {}
        self.known_chats: set[int] = set()
        # Inbound authorization (api.md 2026-07-22 security). Default DENY.
        self._allowed_chat_ids: set[int] = set(allowed_chat_ids or ())
        self.require_pairing = require_pairing
        self._rate_limit_per_min = rate_limit_per_min
        self._persist_chat_id = persist_chat_id
        # One-time pairing code, printed to the SERVER console at gateway start — never
        # sent over Telegram. token_hex(4) = 32 bits; combined with the per-chat rate
        # limit, brute force over Telegram is infeasible.
        self.pairing_code = pairing_code or secrets.token_hex(4)
        self._time_fn = time_fn
        self._rate_events: dict[int, deque[float]] = {}

    def _history(self, chat_id: int) -> deque[dict]:
        return self._histories.setdefault(chat_id, deque(maxlen=self._history_limit))

    # -- authorization / pairing / rate limiting --------------------------
    def is_authorized(self, chat_id: int) -> bool:
        if not self.require_pairing:
            return True
        return chat_id in self._allowed_chat_ids

    def _handle_pair(self, chat_id: int, text: str) -> str:
        if not self.require_pairing:
            return "pairing is disabled; this bot already accepts every chat."
        if chat_id in self._allowed_chat_ids:
            return "this chat is already paired with suiban."
        parts = text.split(maxsplit=1)
        code = parts[1].strip() if len(parts) > 1 else ""
        if code and secrets.compare_digest(code, self.pairing_code):
            self._allowed_chat_ids.add(chat_id)
            if self._persist_chat_id is not None:
                try:
                    self._persist_chat_id(chat_id)
                except Exception:  # noqa: BLE001 - persistence must not crash the relay
                    logger.exception("telegram relay: failed to persist paired chat %s", chat_id)
            logger.info("telegram relay: chat %s paired", chat_id)
            return "paired — this chat can now talk to suiban."
        return (
            "invalid or missing pairing code. Run '/pair <code>' with the one-time code "
            "printed in the suiban server console at startup."
        )

    def _rate_limit_ok(self, chat_id: int) -> bool:
        if self._rate_limit_per_min <= 0:
            return True
        now = self._time_fn()
        window = self._rate_events.setdefault(chat_id, deque())
        cutoff = now - 60.0
        while window and window[0] <= cutoff:
            window.popleft()
        if len(window) >= self._rate_limit_per_min:
            return False
        window.append(now)
        return True

    async def handle_message(self, chat_id: int, text: str) -> list[str]:
        """One incoming message -> reply chunks. `/pair` is handled first (it is how a
        chat authorizes); an unpaired chat gets a single not-authorized reply and
        nothing reaches the model; a paired chat is rate-limited per minute. Backend
        errors become an honest apology chunk — the gateway never goes silent or
        crashes on a bad turn."""
        if _is_pair_command(text.strip()):
            return [self._handle_pair(chat_id, text.strip())]
        if not self.is_authorized(chat_id):
            return [UNAUTHORIZED_REPLY]
        if not self._rate_limit_ok(chat_id):
            return [RATE_LIMITED_REPLY]
        # Only authorized chats join known_chats: notification pings never reach an
        # unpaired chat.
        self.known_chats.add(chat_id)
        history = self._history(chat_id)
        history.append({"role": "user", "content": text})
        try:
            reply = await self._send_chat(list(history), session_id_for(chat_id))
        except Exception as exc:  # noqa: BLE001 - relay survives any backend failure
            logger.warning("telegram relay: chat request failed: %s", exc)
            history.pop()  # the turn never happened; don't poison the history
            return [f"suiban could not answer right now ({type(exc).__name__}). Try again."]
        history.append({"role": "assistant", "content": reply})
        return chunk_message(reply)

    @staticmethod
    def research_ping(job) -> str:
        """The one-line coarse completion ping (title + summary joined)."""
        title, summary = research_notification(job)
        return f"{title}: {summary}"


def api_send_chat(api_base: str, *, timeout_s: float = CHAT_TIMEOUT_S) -> SendChatFn:
    """SendChatFn talking to the local suiban server over the frozen contract."""

    async def send(messages: list[dict], session_id: str) -> str:
        async with httpx.AsyncClient(base_url=api_base, timeout=timeout_s) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "bonsai-auto",
                    "messages": messages,
                    "mode": "chat",
                    "session_id": session_id,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"].get("content") or ""

    return send


class TelegramGateway:
    """python-telegram-bot glue: long-polling Application bound to a TelegramRelay.

    start()/stop() are called from the app lifespan; both are idempotent and both
    swallow transport teardown noise (shutdown must never hang the server)."""

    def __init__(self, token: str, relay: TelegramRelay) -> None:
        self._token = token
        self.relay = relay
        self._app = None  # telegram.ext.Application once started

    @property
    def running(self) -> bool:
        return self._app is not None

    async def start(self) -> None:
        if self._app is not None:
            return
        # Lazy import: the core must run without the optional extra installed.
        from telegram.ext import Application, MessageHandler, filters

        application = Application.builder().token(self._token).build()
        # filters.TEXT (not ~COMMAND): `/pair` must reach the relay so a chat can
        # authorize itself; the relay routes it before any model call.
        application.add_handler(MessageHandler(filters.TEXT, self._on_message))
        await application.initialize()
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)
        self._app = application
        logger.info("telegram gateway: long polling started")
        if self.relay.require_pairing:
            logger.warning(
                "telegram gateway: pairing REQUIRED. To authorize a chat, send "
                "'/pair %s' to the bot FROM that chat. This code is printed here only, "
                "never over Telegram, and is valid until the next restart.",
                self.relay.pairing_code,
            )
        else:
            logger.warning(
                "telegram gateway: pairing is DISABLED "
                "(gateways.telegram.require_pairing=false) — every chat that messages "
                "the bot can drive it. Re-enable pairing unless this bot is private."
            )

    async def stop(self) -> None:
        application, self._app = self._app, None
        if application is None:
            return
        for step in (
            application.updater.stop,
            application.stop,
            application.shutdown,
        ):
            try:
                await step()
            except Exception:  # noqa: BLE001 - teardown must always complete
                logger.exception("telegram gateway: shutdown step failed")
        logger.info("telegram gateway: stopped")

    async def _on_message(self, update, context) -> None:
        message = update.effective_message
        chat = update.effective_chat
        if message is None or chat is None or not message.text:
            return
        for chunk in await self.relay.handle_message(chat.id, message.text):
            await context.bot.send_message(chat_id=chat.id, text=chunk)

    def notify(self, kind: str, title: str, summary: str) -> None:
        """The generalized notification hook (research completions, scheduled runs):
        fire-and-forget pings to every chat seen this run. `kind` is a stable machine
        string ("research", "schedule") — Telegram sends one text either way; other
        gateways may route on it."""
        if self._app is None or not self.relay.known_chats:
            return
        text = f"{title}: {summary}" if summary else title
        logger.debug("telegram gateway: %s notification: %s", kind, title)
        for chat_id in list(self.relay.known_chats):
            asyncio.get_running_loop().create_task(self._send_safe(chat_id, text))

    def notify_research_complete(self, job) -> None:
        """Research-completion ping via the generalized hook (JobManager listener)."""
        self.notify("research", *research_notification(job))

    async def _send_safe(self, chat_id: int, text: str) -> None:
        try:
            await self._app.bot.send_message(chat_id=chat_id, text=text)
        except Exception as exc:  # noqa: BLE001 - a dead chat must not raise
            logger.warning("telegram gateway: ping to chat %s failed: %s", chat_id, exc)


def build_gateway(
    settings: Settings,
    *,
    send_chat: SendChatFn,
    notices: list[Notice],
    persist_chat_id: PersistChatFn | None = None,
) -> TelegramGateway | None:
    """Decide whether the Telegram gateway can start; explain honestly when not.
    `persist_chat_id` (ConfigManager.add_telegram_chat_id) records a paired chat id in
    config.toml so authorization survives restarts (api.md 2026-07-22 security)."""
    telegram_settings = settings.gateways.telegram
    if not telegram_settings.enabled:
        return None
    if not telegram_settings.token:
        notices.append(
            Notice(
                "warn",
                "telegram_token_missing",
                "Telegram gateway is enabled but no token is set. PATCH /v1/settings "
                "with gateways.telegram.token (write-only) and restart.",
            )
        )
        return None
    if importlib.util.find_spec("telegram") is None:
        notices.append(
            Notice(
                "warn",
                "telegram_unavailable",
                "Telegram gateway is enabled but python-telegram-bot is not installed. "
                "Install the extra: uv pip install 'suiban[gateways]'",
            )
        )
        return None
    relay = TelegramRelay(
        send_chat,
        allowed_chat_ids=telegram_settings.allowed_chat_ids,
        require_pairing=telegram_settings.require_pairing,
        rate_limit_per_min=telegram_settings.rate_limit_per_min,
        persist_chat_id=persist_chat_id,
    )
    return TelegramGateway(telegram_settings.token, relay)
