# Known issues: suiban

Honest, public list of the sharp edges in this release. None of these is a secret
foot-gun: each is either a documented boundary, a platform gap or a `v1.1` item.
suiban's default posture is **loopback-only, single-user, on your own machine**.
Most items below only matter when you deviate from that.

## Security boundaries

### The shell tool is not a security sandbox
`mode: "code"` exposes a `shell` tool that runs commands with the **same privileges
as the suiban process**. The tool jails filesystem paths to the session workdir and
scrubs secret-bearing environment variables (anything matching
`*TOKEN*`/`*KEY*`/`BONSAI*`/`TELEGRAM*`/`HF*`) from the child environment, and it
keeps a denylist, but shell metacharacters mean a denylist is **not a boundary**,
and it is not treated as one. A model (or a prompt-injection payload that reaches an
agentic run) can run arbitrary commands as you.

- **Mitigation now:** run suiban as an unprivileged user, ideally in a container or a
  dedicated account; do not run it as root; keep the default loopback bind.
- **`TODO(v1.1):`** a real OS sandbox for the tool child (`bwrap` / Landlock on
  Linux, sandbox-exec on macOS). The seam exists; the enforcement does not yet.

### MCP servers run unsandboxed, with your privileges
Configured MCP stdio servers (`settings.mcp`) are launched as child processes with
your privileges and are **fully trusted**: the MCP config is a trust boundary, not a
sandbox. Add only servers you trust; a malicious or compromised MCP server can do
anything your user account can.

### LAN / non-loopback binding requires the bearer token
The default bind is `127.0.0.1` and stays **auth-free** (zero-friction local use).
If you set `server.host` to a non-loopback address (e.g. `0.0.0.0`), suiban
auto-generates a bearer token at first apply and then **requires**
`Authorization: Bearer <token>` on every request except `GET /v1/system/health`
(401 otherwise). See the README security notes and api.md (2026-07-22). Notes:

- CORS is **not** an authorization boundary and is not treated as one; the bearer
  check is.
- The token lives in `~/.bonsai/config.toml` (write-only over the API). `chmod 600`
  that file. There is no transport encryption. Put suiban behind a TLS reverse proxy
  or a VPN/WireGuard tunnel if you expose it beyond localhost.

### Remote agentic access is off by default
The Telegram gateway is pinned to `chat` mode and, even so, defaults to a chat-ID
allowlist with one-time pairing (`gateways.telegram.require_pairing = true`) and a
per-chat rate limit. `server.remote_agentic` is `false` by default; leaving it off is
the safe choice. Turning it on widens what a paired remote user can drive. Do it only
for chats you control. Never expose the gateway token; it stays in `~/.bonsai`.

## Download integrity

Binary and model downloads **are** integrity-checked, and this release ships the
checked-in SHA-256 manifest that makes that real:

- **Fork binaries:** every release archive is verified against
  `src/suiban/installer/assets_sha256.json` (SHA-256 digests captured from the GitHub
  releases API for the pinned tag `prism-b9596-9fcaed7`) before extraction; a mismatch
  is a hard failure. Every asset the installer can select for a supported
  os/backend/arch has a checked-in digest (a regression test asserts this), so a real
  install never silently falls back to the un-verified path.
- **Model weights:** `huggingface_hub` verifies each file's own etag/SHA against the
  HF repo during download (transport integrity). On top of that, suiban's pinned
  per-file byte sizes are a tripwire: a size that deviates >2% is now a **hard
  failure**, not a warning.

There is no signature/provenance chain beyond TLS + these hashes; the hashes are only
as trustworthy as this repository. If you bump the pinned fork tag, regenerate the
manifest from the releases API in the same change.

## Platform / feature gaps

### macOS can orphan `llama-server` on a hard kill
Child `llama-server` processes are reaped via `PR_SET_PDEATHSIG`, which is
**Linux-only**. On macOS, if suiban is killed with `SIGKILL` (not a clean shutdown),
a `llama-server` subprocess can be orphaned and keep holding VRAM. Clean shutdowns and
`SIGTERM` are handled on all platforms. Workaround on macOS: check for and kill stray
`llama-server` processes after a hard kill. `TODO(v1.1):` a macOS-native reaper.

### WhatsApp link + send is unverified against live WhatsApp
The WhatsApp gateway is QR device-linked (changed 2026-07-22b): enable it, scan the QR
from `GET /v1/gateways/whatsapp/qr` and outbound pings go through the linked session.
The link backend is pluggable (the real one uses `neonize`, an **optional native
dependency**), but if `neonize` is absent a stub backend stands in: it renders a **real,
scannable QR**, but scanning it does **not** complete a live link and there is no session
to send through. The live link handshake and outbound send path are therefore
**unverified against live WhatsApp** in this build. The state machine, QR rendering,
unlink and notify-when-unlinked behavior are fully tested. `TODO(v1.2):` wire and verify
the neonize backend end to end. Telegram remains the verified messaging path.

### Playwright tier-2 browsing is not wired
The agentic `browse_t2` (headful/JS browsing via a sandboxed Playwright profile) is a
declared seam that is **inert in v1**. The tool is not connected to a live browser.
Tier-1 fetch-and-read browsing works. `TODO(v1.1):` wire tier-2.

### Ternary vs 1-bit family auto-degrades under 12 GB
On GPUs below ~12 GB VRAM, a loadout that cannot fit the ternary (`Q2_0`) family for
the chosen sizes auto-selects the smaller 1-bit (`Q1_0`) family (or smaller model
sizes) to fit. This is surfaced as a `notice`, never silent, but it means the model
quality on an 8 GB card is the 1-bit family's: a real, disclosed trade, not a bug.

## Static-analysis note

`bandit -ll` reports `B608` (hardcoded-SQL) on the FTS5 / schedules / research query
builders. These are false positives: the interpolated fragments are fixed internal
column-name constants, never user input and every value is bound through `?`
placeholders. FTS5 match queries additionally strip operators before quoting. No
user-controlled string is ever concatenated into SQL.
