# Gateways

Messaging gateways let you talk to your bonsai stack from a phone. v1 ships two:
**Telegram** (long-polling chat relay + notification pings) and **WhatsApp**
(QR device-linked, outbound-only notification pings; changed 2026-07-22b). Gateways are
ordinary API clients. They speak the frozen HTTP contract ([api.md](api.md)) to the
local server and nothing else. Both fire from the same generalized notify hook:
deep-research completions and scheduled-run results ping every configured gateway.

## Telegram

### What it does

- **Chat relay.** Every text message to your bot becomes a
  `POST /v1/chat/completions` (mode `chat`, model `bonsai-auto`) with a per-chat
  session id (`tg-<chat_id>`), so conversations from Telegram get the same memory,
  session archive and compression treatment as any other client. Replies are sent
  as complete messages, chunked under Telegram's 4096-character limit.
  TODO(v1.1): streamed replies via edited-message updates (rate-limit aware);
  v1 deliberately ships the simple, robust variant.
- **Research pings.** When a deep-research job reaches a terminal state, every chat
  that has talked to the bot since this server started gets a coarse ping (job state
  + your own query text, never stages, sources or internals; see
  [research.md](research.md)). TODO(v1.1): persist known chats across restarts.

### Transport: long polling, no webhooks

The gateway makes outbound HTTPS connections to `api.telegram.org` and polls for
updates. It never opens a port, never registers a webhook and never needs a public
address, consistent with suiban's loopback-only posture. This is a v1 design rule,
not a limitation to be fixed.

### Setup

1. Create a bot with Telegram's `@BotFather`; it gives you a token of the form
   `123456:ABC-...`.
2. Install the optional dependency (the core runs without it):

   ```
   uv pip install 'suiban[gateways]'
   ```

3. Configure the token and enable the gateway, either over HTTP:

   ```
   PATCH /v1/settings   {"gateways": {"telegram": {"enabled": true, "token": "123456:ABC-..."}}}
   POST  /v1/system/apply
   ```

   or by editing `~/.bonsai/config.toml` while suiban is stopped:

   ```toml
   [gateways.telegram]
   enabled = true
   token = "123456:ABC-..."
   ```

4. Restart suiban. Gateway changes are `requires_restart`: the gateway starts and
   stops with the app lifespan (TODO(v1.1): hot start/stop at apply time).

### Token handling (write-only secret)

The token lives only in `~/.bonsai/config.toml`, never in a repo, never in logs.
Over HTTP it is **write-only**: `PATCH /v1/settings` accepts it, but `GET
/v1/settings` only ever reports `"token_set": true`. Treat the token like a
password; anyone holding it controls your bot.

### Failure behavior

The gateway degrades, it never takes the server down:

- enabled without a token → server runs, `telegram_token_missing` notice in
  `GET /v1/system`;
- `python-telegram-bot` not installed → `telegram_unavailable` notice;
- gateway crashes at startup → `telegram_start_failed` notice, chat serving
  continues;
- a failed reply to one message → an apology message in that chat, and the failed
  turn is dropped from the relay history so it cannot poison the next request.

Note: anyone who can message your bot can chat with your stack. Telegram bots are
open by default. If that matters to you, restrict the bot with BotFather's privacy
settings or keep the gateway disabled. TODO(v1.1): an allowlist of chat ids in
`gateways.telegram`.

## WhatsApp: QR device-linking, outbound notifications only

**Changed 2026-07-22b.** WhatsApp no longer uses a Cloud-API token. It links a device
via the WhatsApp Web multi-device protocol (you scan a QR with your phone, exactly like
WhatsApp Web / Linked Devices) and the linked session then sends outbound pings. There
is **no secret** in the config: the whole gateway is `{enabled, linked, to_number}`.

### What it does (and deliberately does not)

- **QR device-linking.** Enable the gateway, fetch a QR
  (`GET /v1/gateways/whatsapp/qr`), scan it in WhatsApp → Settings → Linked Devices.
  The linked-device session lives under `~/.bonsai/whatsapp/`, never in a repo.
  `POST /v1/gateways/whatsapp/unlink` forgets it.
- **Notification pings.** Once linked, deep-research completions and scheduled-run
  results are sent as text to ONE configured number (`to_number`) through the linked
  session. Pings are coarse: job state + your own query, or schedule name + a one-line
  summary, never stages, sources or internals.
- **No inbound chat relay.** `TODO(v1.2)`: inbound relay stays out of v1, consistent
  with suiban's no-open-ports posture.

### The QR endpoints

`GET /v1/gateways/whatsapp/qr` → `{ state, qr, qr_ascii }`:

- `state` is `unlinked` (gateway disabled / no link in progress), `awaiting_scan`
  (a QR is shown; poll this endpoint until it flips) or `linked` (`qr`/`qr_ascii`
  clear to `null`).
- `qr` is the raw pairing string to render as a QR; `qr_ascii` is a ready-to-print
  terminal QR (both are produced from the SAME pairing string with the `qrcode` lib).

`POST /v1/gateways/whatsapp/unlink` → `{ "state": "unlinked" }` (idempotent).

### Setup

1. Install the optional dependencies (the core runs without them):

   ```
   uv pip install 'suiban[gateways]'
   ```

   This pulls `qrcode` (renders the link QR) and, optionally, you may also install
   `neonize` for the live WhatsApp Web backend (an optional **native** dependency; see
   the honesty note below).
2. Enable the gateway and set the recipient, over HTTP:

   ```
   PATCH /v1/settings   {"gateways": {"whatsapp": {"enabled": true, "to_number": "15551234567"}}}
   POST  /v1/system/apply
   ```

   or by editing `~/.bonsai/config.toml` while suiban is stopped:

   ```toml
   [gateways.whatsapp]
   enabled = true
   to_number = "15551234567"
   ```

3. Restart suiban (gateway changes are `requires_restart`, same as Telegram), then scan
   the QR from `GET /v1/gateways/whatsapp/qr` with your phone.

### Honesty note / KNOWN_ISSUE

Without a live WhatsApp account the link handshake and the send path **cannot be
exercised end to end** in this build. If `neonize` is not installed a stub backend is
used: it renders a **real, scannable QR**, but scanning it does not complete a link and
there is no live session to send through. `neonize` is an **optional native dependency**,
and the live link+send path is **unverified against live WhatsApp**, tracked as
`TODO(v1.2)` in `gateways/whatsapp.py` and in the repo `KNOWN_ISSUES.md`. The state
machine, QR rendering, unlink and notify-when-unlinked behavior are fully tested.

### Failure behavior

The gateway degrades, it never takes the server down and never fails a job:

- disabled → no gateway is built; `GET /v1/gateways/whatsapp/qr` reports `unlinked`.
- enabled but not yet linked → `awaiting_scan`; `notify()` no-ops (nothing to send to).
- a failed send (no live session, bad number) → logged AND surfaced as a
  `whatsapp_send_failed` notice; the research job or scheduled run that triggered the
  ping is untouched.
