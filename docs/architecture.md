# suiban architecture

How the bonsai stack is put together, and why. This document describes design that is
law for v1 (decisions here are settled; see the repo `CLAUDE.md` conventions) and is
honest about what is *not* finished. The HTTP surface itself is specified in
[api.md](api.md), which is frozen; nothing in this file adds to or overrides it.

## 1. Three repos, one contract

The stack is three independent, independently publishable repositories:

| Repo | What it is | Stack |
|---|---|---|
| `suiban` | Inference & orchestration core (this repo) | Python 3.11+, FastAPI, uv |
| `dai` | Desktop GUI | Tauri 2, React, TypeScript, Tailwind |
| `sentei` | CLI | Python, Typer, Rich |

**The one law:** [`suiban/docs/api.md`](api.md) is the frozen v1 contract and the ONLY
coordination point. dai and sentei talk to suiban exclusively over HTTP
(`http://127.0.0.1:8686` by default, configurable in `~/.bonsai/config.toml`). There are
no cross-repo imports, no shared code and no relative paths between repos. A feature
that needs a contract change lands in api.md first (additive only within v1, with a
changelog entry), then in code. Because the chat endpoint is strictly OpenAI-compatible,
any third-party OpenAI client is a fourth, unplanned-for consumer, which is exactly the
discipline the freeze enforces.

## 2. Runtime topology

```
   dai (GUI)          sentei (CLI)      any OpenAI client       Telegram servers
      |                    |                    |                     ^
      |                    |                    |                     | long-poll
      +---------+----------+--------------------+                     | (outbound only)
                |                                                     |
                v   HTTP :8686  --  frozen contract: docs/api.md      |
+---------------------------------------------------------------------------------+
|  suiban (FastAPI, single process)                                   |           |
|                                                                     |           |
|   api/ routes ---- modes/  chat . code . ultra . deep_research      |           |
|        |           (versioned prompt files)                gateways/telegram.py |
|        v                                                                        |
|   agent/  ReAct loop -- grammar-constrained tool calls, repair <=2 per run      |
|        |                                                                        |
|   tools/  fs (+undo) . shell (confirm-gated) . git (ro) . browse t1/t2 .        |
|           recall . mcp/ (stdio servers -> namespaced tools)                     |
|        |                                                                        |
|   memory/  identity . state . FTS5 archive . skills   (27B-only writes)         |
|        |                                                                        |
|   sched/  telemetry (NVML -> rocm-smi -> Metal -> RAM) . budget . loadout       |
|        |                                                                        |
|   llama/  port pool . health . restart . SlotGate queue . stderr ring . flags   |
+-------+----------------+-----------------+------------------+-------------------+
        |                |                 |                  |
        v                v                 v                  v
  llama-server     llama-server      llama-server       llama-server
  orchestrator     utility           worker-1           worker-2
  bonsai-27b       bonsai-4b         bonsai-8b          bonsai-8b
  (+mmproj)        :8702             :8703              :8704
  :8701  GPU 0     GPU 0             GPU 1*             GPU 1*
                                     (* GPU 1 when present, else GPU 0)
```

Slots are `llama-server` subprocesses from the pinned PrismML llama.cpp fork
(`PrismML-Eng/llama.cpp`, branch `prism`, release `prism-b9596-9fcaed7`). suiban is the
only thing that talks to them; clients never see slot ports.

## 3. suiban internals

### 3.1 `llama/`: process manager

- Resolves the fork binary for the current OS/backend, including the fork's release
  asset naming scheme (`llama-prism-<build>-<sha>-bin-<os>-<backend>-<arch>`) and the
  Windows deviation (mainline-style names plus a separate cudart zip; the installer
  handles both).
- Owns a port pool for slots, performs health checks and restarts crashed slots
  (a restart never changes the planned loadout). A slot whose restart loop gives up
  surfaces a `slot_failed` notice even hours after startup.
- Managed flags per slot: `--jinja` (ChatML + tool templates), `--mmproj` (27B only,
  vision projector), `--reasoning-budget` (thinking default), `--cache-type-k/v`
  (KV quantization), `-fa on` (flash attention, required for quantized V; the pinned
  fork's `-fa` takes a mandatory `on|off|auto` value), `-md` (DSpark speculative
  drafter, opt-in, default off) and `-lv 4` (log verbosity: the KV-cache summary
  line below is filtered at the fork's default verbosity 3).
- llama-server stderr goes to a bounded ring buffer (last ~200 lines) instead of
  `/dev/null`: the tail is quoted into slot-failure notices (the only diagnostic when
  a slot dies), and startup lines feed the **hybrid-attention runtime probe**: the
  `llama_kv_cache: size = … N layers …` line is parsed and, for the 27B, checked
  against the expected 16-of-64 hybrid allocation. A mismatch emits a
  `kv_layers_mismatch` notice (the VRAM plan assumes 16 layers); an unrecognized log
  format degrades to "unknown", never to a crash.
- **Per-slot serialization (SlotGate).** A slot's llama-server effectively decodes one
  request at a time, so suiban serializes explicitly instead of letting concurrent
  runs starve each other into timeouts: one run holds a slot's gate, at most 4 more
  wait and beyond that the chat endpoint answers 429 `overloaded_error`
  (`slot_queue_full`). A `stream_events` chat that actually waits longer than ~1 s
  gets a `slot_queued` notice ("queued behind N"). Research jobs and scheduled runs
  wait without the 429 cap (background work queues, it does not fail); research
  additionally locks per pipeline *step*, not per job; see
  [research.md](research.md).
- **Client-disconnect abort.** Streaming responses abort via starlette's generator
  cancellation when the client drops. Non-streaming chat runs poll
  `request.is_disconnected()` alongside the backend task and cancel it on disconnect,
  so a closed tab stops the GPU instead of orphaning the run.
- **Lazy / keep-alive residency (`llama/load_controller.py`, api.md 2026-07-22c).**
  ollama-style: `serve` **plans** the loadout at boot but starts **no** slots, so suiban
  comes up healthy with zero VRAM in use. A `LoadController` owns residency from there:
  `ensure_loaded()` runs at the top of every inference path (chat/completions incl.
  ultra, deep-research jobs, scheduled runs and compress-on-import) and warms the
  planned slots on demand (reusing `start_all` + the health wait) before routing. A cold
  start leads a rich stream with a `warming_up` notice, then simply waits for the slots
  to become healthy. A background reaper unloads the whole loadout (freeing VRAM) after
  `runtime.keep_alive` idle minutes, but **never mid-generation**: an in-flight chat or a
  running research job counts as busy (the same idle test `/v1/system/apply` uses), and
  the residency lock serializes warm-ups against unloads. `keep_alive` parses `"24/7"` /
  `"0"` / `"always"` (and any non-positive value) as *stay hot forever*; a positive
  minutes value is the idle window (default `"5"`). It is read live by the reaper, so an
  applied change takes effect at the next idle moment with no restart. `GET /v1/system`
  reports `runtime: { keep_alive, models_loaded, state }` with `state ∈
  cold|loading|ready|idle_unloading`; a not-yet-resident slot reads `cold` honestly, and
  a slot that *failed* to launch keeps its `failed` state (the failure never gets papered
  over). External-provider chats never trigger a local warm-up (using Ollama/OpenAI must
  not spin up local GPU), so their background housekeeping (auto-titling on the utility
  slot) runs only while the loadout is already resident, degrading to "no title" when
  cold rather than forcing a load.

### 3.2 `sched/`: telemetry, budget, loadout

**Telemetry abstraction.** Probed in order: NVML (NVIDIA) → `rocm-smi` (AMD) → Metal
(Apple) → psutil RAM fallback. `GET /v1/system` reports `telemetry_source`, and `gpus`
is `null` on CPU-only machines; clients must not assume GPU telemetry exists.

**Budget bootstrap: analytic → measured.** Loadout planning needs to know what each
model costs before anything is loaded. On a fresh install the planner uses an *analytic
prior*: Hugging Face weight file sizes + the KV bytes/token derivation (see
[hardware.md](hardware.md)) + a fixed buffer estimate per model. After the first real
launch, the measured ctx-independent costs (buffers, and weights when they deviate) are
written to `~/.bonsai/budget.json`, keyed by model + family (KV is never "measured"
because it scales exactly with context) and **measured values override the analytic
prior from then on**. `GET /v1/system/budget` labels every row `analytic` or `measured`
so clients can show which numbers are real.

**Loadout planner.** A loadout is: one orchestrator slot (27B, degraded to the 1-bit
family on small cards, never absent on GPU tiers), one **permanent utility slot** (4B;
1.7B on the 8 GB tier; on CPU-only the orchestrator doubles as utility) and zero or
more worker slots. Workers degrade along a fixed ladder as VRAM shrinks:

```
2 x 8B  ->  8B + 4B  ->  2 x 4B  ->  1 x 4B  ->  1 x 1.7B  ->  none (sequential Ultra)
```

A safety margin is always reserved. Placement on multi-GPU machines: orchestrator (and
utility) on GPU 0, workers on GPU 1 when present.

**The never-mid-run rule.** The loadout is chosen at run start and is immutable while
runs are active. Models are never loaded or unloaded mid-run; requesting a non-resident
model is a 409 (`model_not_resident`); staged settings apply only at the next idle
moment (`POST /v1/system/apply`). This trades flexibility for the thing local users
actually feel: no surprise VRAM thrash and no mid-conversation model swaps.

**Degrade ladders (never crash, never silent).** Two more ladders exist beside the
worker ladder, and every step on any ladder emits a `notice` (SSE event and
`/v1/system.notices`):

- *Quant family:* configured `ternary` degrades to effective `1bit` where the ternary
  27B cannot fit (12 GB / 8 GB tiers). `/v1/system.quant_family` reports
  `configured` vs `effective` with `degraded` and `reason`.
- *KV cache:* K=`q8_0` + V=`TQ4_0` (default) or V=`TQ3_0` (aggressive preset) →
  TurboQuant kernels or flash attention missing → K/V=`q8_0` → flash attention unusable
  → K/V=`f16`. Unchecking the TurboQuant disclaimer toggle also selects K/V=`q8_0`.
  Full detail in [turboquant.md](turboquant.md).

### 3.3 `effort.py`: the effort ladder

| Effort | `thinking_budget_tokens` | Max tool iterations |
|---|---|---|
| `low` | 0 (thinking off) | 8 |
| `mid` | 4,096 | 16 |
| `high` | 12,288 | 32 |
| `xhigh` | 24,576 | 48 |
| `max` | −1 (unlimited) | 64 |

Every thinking budget is capped at `min(budget, 40% of the slot's context)` so thinking
can never starve the answer. Ultra sub-tasks inherit the *request* effort; on
sequential tiers (no worker slots) that inheritance is capped at `mid`: xhigh
thinking per sub-task on a single slot multiplies wall-clock time with no parallelism
payoff (measured live: a trivial sequential Ultra ran past 10 minutes before this
cap). A request that carries no `effort` resolves through
`settings.effort_default` (optional override) and then the mode's default. Sampling
follows the model cards: 27B temp 0.7 / top-p 0.95 / top-k 20; 8B, 4B and 1.7B
temp 0.5 / top-p 0.85 / top-k 20. Note the small models default thinking *off*
upstream; high-effort workers turn it on; quality of small-model thinking is
verified during the integration phase (honest TODO, see section 5).

### 3.4 `modes/`: chat, code, ultra, deep research

Mode system prompts are versioned markdown files (`src/suiban/modes/prompts/*.md`);
`GET /v1/modes` exposes name, description, `system_prompt_version`, tool list and
default effort, never the prompt text itself.

- **chat**: direct conversation with the orchestrator; memory recall on demand.
- **code**: plan → act → verify. Emits a `plan` event before acting; filesystem, shell
  (confirm-gated end-to-end, the HTTP surface never executes arbitrary commands
  directly) and read-only git tools; diffs and tool activity stream as events.
- **ultra**: the orchestrator decomposes the task and spawns contained sub-agents on
  worker slots (`agent_spawn` / `agent_result` events). Sub-tasks inherit the request
  effort (capped at `mid` when sequential), are capped in number (worker count when
  parallel, 3 sequential) and run under per-sub-task wall-clock timeouts (~240 s at
  low/mid effort, ~480 s above); a timed-out or crashed sub-agent becomes a `failed`
  `agent_result` plus a notice, and the orchestrator synthesizes around it. Every
  degrade on this path (sequential fallback, effort cap, plan truncation, timeout)
  emits a notice with a one-line reason. Containment is structural, not prompt-based:
  workers get no skill/memory write tools, no vision, no tier-2 browsing. On tiers
  with no worker slots, Ultra still works but sub-tasks run one at a time on the
  orchestrator (v1 behavior), `ultra_parallel` is reported `false` and a notice says
  so. Ultra coordinates its sub-agents with **SLAP** (api.md §12); the `slap.enabled`
  toggle (api.md 2026-07-22c, default on) turns that off: dispatch then runs the plain
  structured-dict path (no SLAP messages built or recorded, so `/v1/slap/trace` is empty
  for those runs, though `/v1/slap` still serves the protocol + schemas). The per-agent
  volatile system prompts and the `agent_spawn`/`agent_result` events are unchanged
  either way; the toggle is read per Ultra request.
- **deep_research**, not a chat mode: an async job on `/v1/jobs` (typically 15–40 min).
  **Product rule, not a gap:** progress is coarse only: a stage label plus a percent.
  Internal queries, URLs, drafts and sub-agent chatter are never streamed to users.
  Completion can trigger gateway notifications (e.g. a Telegram ping).

### 3.5 `agent/`: the loop

A ReAct-style loop, grammar-constrained everywhere: every tool call is decoded under a
`json_schema`/GBNF grammar by `llama-server`, so tool arguments parse by construction.
When a step still fails validation semantically, the loop retries with a repair prompt
(the validation error fed back). The repair budget is **per run** (at most two
repairs total, regardless of valid calls in between), after which malformed calls are
recorded as graceful failures (`tool_result` with `status:"error"`) rather than
aborting the run. The loop counts malformed/repaired/abandoned calls per run; the
counters ride the `usage` stream event as optional fields and land in the log, so the
malformed-call rate of a model/prompt combination is measurable. Canned multi-step
transcripts (happy, repaired, exhausted, 20-step long-haul) replay through the real
loop as regression fixtures (`tests/fixtures/transcripts/`). Iteration ceilings come
from the effort ladder above.

### 3.6 `memory/`: four layers, one writer

Four layers: `identity` (human-owned file; human-editable over HTTP via
`PUT /v1/memory/state/identity.md`, never model-written), `state` (small bounded
files), `archive` (SQLite FTS5: sessions, transcripts, distilled entries) and
`skills` (agentskills.io-compatible markdown). Recall is tool-driven (`memory_search`
+ `session_search`, in chat AND code modes) plus one narrow automatic path (additive
2026-07-21c): the top FTS5 memory hits for the latest user message are injected as a
single delimited, budget-capped system block; see memory.md §3 for the exact rules.
Compression triggers at ~70% of slot context using the resident utility model.
**Only the 27B orchestrator writes memories and skills**, during post-task
reflection (rate-limited, background, failure-tolerant; memory.md §7), and this is
enforced server-side by slot role, not by prompt text. Workers and utility models
read skills; they can never write them. Full specification: [memory.md](memory.md).

### 3.7 `tools/`: capability-gated

Tools are MCP-compatible in shape: filesystem, shell (confirm-gated), git (read-only),
browse tier 1 (fetch + readability extraction), browse tier 2 (Playwright, sandboxed,
never given credentials) and memory/skill recall. Tier-2 browsing, vision and
skill/memory writes are capability-gated on a resident 27B; `/v1/system.capabilities`
is the truth clients render.

**Code-mode edit safety.** `fs_write` computes a unified diff of the before-state vs
the new content BEFORE applying it. The diff rides the `tool_result` content, so
clients (dai's tool feed) show exactly what an edit changed as it happens. Every
applied edit is journaled under `<workdir>/.suiban-undo/` (numbered entries with the
full prior content, last 20 kept); `fs_undo` reverts the most recent edit or a
specific one by number, consuming its entry. The journal dir is hidden from `fs_list`
and refused as an `fs_write` target. Undo entries are written before the edit
applies, so even an interrupted write leaves the prior state recoverable.

**Confirm gate for file mutations (api.md 2026-07-22b).** File mutations (`fs_write`,
`fs_undo`) share the destructive-shell confirm gate: the first call is refused with a
`tool_result` `status:"denied"`, a single-use `confirm_token` **bound to the exact
operation** (a `fs_write` token is bound to `(path, content)`; a `fs_undo` token to the
edit's sequence number) and the unified diff. The client renders it Approve/Decline,
and the model re-runs the identical operation with the token to apply. Nothing touches
disk before confirmation, so a denial is a true no-op. `fs_read`/`fs_list` are never
gated. This is the same mechanism, tokens and single-use/operation-binding discipline
as the shell gate: `fs_write`/`fs_undo` simply join `rm` and friends behind it.

**`auto_confirm` bypass (api.md 2026-07-22b, code/ultra only).** A `ChatRequest` may set
`auto_confirm:true` (400 in chat mode); it flows through the registry into the
`ToolContext.auto_confirm` flag, and both `shell` and `fs` skip the gate for that run.
Destructive commands and file mutations execute without a `denied`/`confirm_token`
round-trip. This is a deliberately dangerous power-user setting: every auto-confirmed
destructive/mutating action is logged (`logger.warning` with the command/path) so the
bypass is never silent.

**MCP servers (`mcp/`, api.md additive 2026-07-21d).** Users can attach external MCP
servers (stdio transport only in v1; `settings.mcp_servers[]`, requires_restart;
they start/stop with the app lifespan like gateways). suiban's client speaks
JSON-RPC 2.0 over the server subprocess's stdin/stdout: `initialize` handshake
(spec rev 2025-06-18) → `notifications/initialized` → `tools/list` →
`tools/call` with per-call timeouts. Connected servers' tools join chat/code runs
namespaced `mcp_<server>_<tool>`, their JSON schemas passed through to the model
verbatim; text content is extracted from results. A server that fails to start or
crashes mid-run becomes an `mcp_server_failed` notice and its tools vanish from
subsequent runs, never a crash and an in-flight call degrades to a tool-shaped
error the loop survives. `/v1/modes` keeps listing built-in tools only: the MCP set
is per-config and dynamic, not part of a mode's definition. Scope is tools-only in
v1: resources, prompts and sampling are `TODO(v1.1)` (no current mode consumes
them). Verified against the public `@modelcontextprotocol/server-everything` and
`server-filesystem` servers; CI exercises a bundled stdlib-only fixture server.

**MCP connector catalog (`mcp/catalog.py`, api.md 2026-07-22c).** On top of the free-form
custom servers, a curated one-click catalog of well-known MCP servers (filesystem, git,
fetch, memory, everything, sequential-thinking, time: the reference/community set the
openclaw and hermes ecosystems reference in their optional-mcps). `GET /v1/mcp/connectors`
lists it with each connector's `enabled` flag per `settings.mcp_connectors` (a list of
`{ id, enabled }`, distinct from `mcp_servers`). Enabling one resolves it into an
`McpServerSettings` and wires it into the SAME manager: identical transport, namespacing
and failure handling; custom servers win any id collision so a user entry is never
shadowed. Unlike custom `mcp_servers` (requires_restart), connectors are
`pending_until_idle`: the manager re-syncs (`resync()`) on apply so an enabled connector
starts without a full restart, leaving unchanged servers running untouched. A connector
that needs a launch path (filesystem) defaults its root to the user's home.
`TODO(v1.1): per-connector path in the { id, enabled } shape`.

### 3.8 `gateways/`

Telegram: long-polling (outbound only, no inbound port, nothing to expose), chat
relay plus notification pings. The bot token lives only in `~/.bonsai/config.toml`
and is write-only over the API (`token_set: true` is all GET ever shows). WhatsApp
(QR device-linked, changed 2026-07-22b) is outbound-only: it links via the WhatsApp Web
multi-device protocol (scan the QR from `GET /v1/gateways/whatsapp/qr`; linked session
under `~/.bonsai/whatsapp/`, never a repo), so there is **no secret**: settings are just
`{enabled, linked, to_number}`. The link backend is pluggable (`neonize` for the real
session, an optional native dep; a stub renders a real QR when it is absent). Honest
boundary: without a live WhatsApp account the link+send path is unverified; see
docs/gateways.md and KNOWN_ISSUES.md. Inbound relay is TODO(v1.2).

### 3.9 `providers/`: the external-inference boundary (additive 2026-07-21c)

Users can register external OpenAI-compatible endpoints (settings `providers[]`:
`ollama` for a local Ollama, `openai` for any generic compatible server; `api_key`
write-only like every secret). The boundary is deliberately narrow and honest:

- **suiban never manages their lifecycle or VRAM.** It does not start, stop,
  schedule or budget an external provider: it only polls `{base_url}/v1/models`
  (short timeout, on boot and after `/v1/system/apply`) and caches the model list.
  A failed poll marks the provider unreachable with a `provider_unreachable` notice
  and keeps the last known list (`resident: false` in `/v1/models`); it never
  crashes or blocks anything.
- **External sessions are proxies, not agent runs.** `<provider>/<model>` ids route
  the chat to the provider as a plain OpenAI request: `mode:"chat"` only (400
  `external_model_mode` otherwise), no thinking control, no grammar guarantees, no
  server-side tool loop: client tools pass through untouched, and effort degrades
  to a sampling (temperature) default. bonsai extension fields and fork-specific
  request fields are never sent upstream.
- **The local stack stays primary.** Memory/project injection, session archiving
  and auto-titling (on OUR utility slot) work for external sessions; post-task
  reflection does not: memory writes remain a 27B-orchestrator capability.
  `bonsai-auto` and the local loadout are untouched defaults.

Search providers for deep research (settings `search`, `suiban/search/`) sit behind
the same kind of seam: one `search(query, count)` protocol, five transports, all
injectable, tested without the network; see [research.md](research.md).

## 4. Security posture (v1)

Loopback bind by default; on a loopback bind there is deliberately **no HTTP auth**:
that is the zero-friction local default, unchanged. Everything runs and stays local:
models, memory, skills, transcripts under `~/.bonsai/`. No telemetry leaves the machine.
Outbound network only for: installer downloads, the browse tools when invoked and
gateways when enabled.

The pre-publication audit (2026-07-22) hardened every path that a non-loopback bind, a
gateway or fetched/untrusted content could reach (api.md 2026-07-22 security entry):

- **Non-loopback bind = auth required.** When `server.host` is not a loopback address
  (`config.host_is_loopback`), suiban mints and persists `server.auth_token` on first
  such bind (printed to the console once, write-only over HTTP; GET shows only
  `auth_token_set`) and requires `Authorization: Bearer <token>` on every route except
  `GET /v1/system/health` (401 `unauthorized` otherwise). The bind host is the
  effective one (`serve --host` cannot silently bypass the gate). `GET /v1/system`
  reports `security.{auth_required, remote_agentic, telegram_paired}`.
- **Telegram front door.** Inbound is default-DENY: only `allowed_chat_ids` reach the
  model. A chat pairs by sending `/pair <code>` with a one-time code printed to the
  server console at gateway start (never sent over Telegram); paired ids persist to
  config. Per-chat rate limit (`rate_limit_per_min`, default 20). Even a paired user is
  pinned to chat mode; `server.remote_agentic` (default off) is reserved and **not yet
  honored**: a gateway is never wired to the shell in v1.
- **SSRF.** `browse_t1` resolves the target host and refuses any address that is
  loopback/private/link-local/reserved/multicast/unspecified, follows redirects
  manually (max 5) and re-checks the host on every hop: a public URL that 302s to
  `169.254.169.254` is blocked at the hop.
- **Prompt injection.** External tool output (browse, `fs_read`, `git_ro`, MCP results)
  enters the conversation wrapped in a delimited UNTRUSTED block; the skill-context
  header and every mode prompt state that fetched/file/skill content is DATA, never
  instructions. The destructive-shell **confirm gate remains the boundary**: no
  fetched page or file can drive an unconfirmed destructive shell call.
- **Jails.** `session_id` is sanitized before any filesystem join (the archive DB keeps
  the raw id, parameterized), so a `../` id cannot relocate the fs/shell jail. `fs_read`
  /`fs_write` open the final path with `O_NOFOLLOW` (closes the resolve-then-act TOCTOU
  for the final component). `git_ro` pins `GIT_CEILING_DIRECTORIES` so discovery cannot
  read a repo above the jail. The shell subprocess env is scrubbed of secret-bearing
  variables (`*TOKEN*/*KEY*/*SECRET*/*PASSWORD*`, `BONSAI_/TELEGRAM_/HF_/HUGGING/AWS_`).

Configured MCP servers are local subprocesses running with the user's privileges.
`~/.bonsai/config.toml` is the trust boundary for what they may do (and they may use the
network themselves). Honest remaining limits (KNOWN_ISSUES, v1.1): the shell denylist is
best-effort, not a sandbox (bwrap/landlock is the real fix); the fs O_NOFOLLOW guard
does not cover intermediate-component swaps (openat2 `RESOLVE_BENEATH` is the real fix);
unverified skill bodies are labelled and fenced but not content-sanitized.

## 5. Known seams in v1 (honest list)

- **Metal TurboQuant kernels**: `TODO(v1.1)`. CUDA and CPU only in v1; Apple Silicon
  runs K/V=`q8_0` with a visible notice.
- **Vulkan / ROCm TurboQuant**: out of scope for v1; `suiban install turboquant` warns
  and skips on those backends (fallback ladder applies).
- **WhatsApp inbound relay**, `TODO(v1.2)`: needs a Meta webhook on a public HTTPS
  endpoint (outbound pings ship in v1; Telegram is the two-way gateway).
- **Browse tier 2 (Playwright)**: implemented behind the capability gate, but sandbox
  hardening is finalized during the integration phase, not before.
- **Small-model thinking above `low`**: upstream defaults thinking off for
  8B/4B/1.7B; Ultra workers inherit the request effort and enable thinking at `mid`
  and above; quality is verified at integration
  (`TODO(v1.1): revisit if quality regresses`).
- **Nearest-size model fallback**: download-fallback logic is built but dormant; both
  families of all four sizes exist upstream today.
- **Benchmark numbers**: VRAM/KV figures in docs are analytic priors until *your*
  machine's first launch measures them. The reference 8 GB machine has been measured:
  buffers and the booted loadout in [hardware.md](hardware.md) §6, `suiban bench kv`
  quality battery and decode-speed tables in [benchmarks.md](benchmarks.md), all
  labeled one-machine. Docs say which numbers are which; we do not publish invented
  benchmarks.
