# Changelog: suiban

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: SemVer.

## [Unreleased]

### Added
- Feature wave 5 (api.md 2026-07-22c, all additive):
  - **Lazy / keep-alive model residency (ollama-style).** `serve` now plans the loadout
    at boot but starts **no** slots: suiban comes up healthy with zero VRAM in use. A new
    `LoadController` (`llama/load_controller.py`) warms the planned slots on the first
    inference request (chat/completions incl. ultra, deep-research jobs, scheduled runs
    and compress-on-import) via the existing `start_all` + health wait, and a background
    reaper unloads the whole loadout after `runtime.keep_alive` idle minutes, **never
    mid-generation** (an in-flight chat or running job counts as busy). A cold start leads
    a rich stream with a `warming_up` notice. `keep_alive`: `"24/7"`/`"0"`/`"always"` (or
    any non-positive value) stays hot forever; a positive minutes value is the idle window
    (default `"5"`), read live by the reaper so an applied change needs no restart.
  - **`GET /v1/system` → `runtime`.** New `runtime: { keep_alive, models_loaded, state }`
    with `state ∈ cold|loading|ready|idle_unloading`; a not-yet-resident slot reports
    `cold`, and `/v1/system/health` treats a cold/idle-unloaded server as healthy (`ok`).
    A slot that *failed* to launch keeps its honest `failed` state through it all.
  - **`slap.enabled` toggle** (default true). Off routes Ultra dispatch through the plain
    structured-dict path (no SLAP messages built or recorded) so `/v1/slap/trace` is
    empty for those runs; `/v1/slap` still serves the protocol version + per-operation
    schemas. Per-agent volatile prompts and `agent_spawn`/`agent_result` events unchanged.
  - **Settings.** `runtime.keep_alive` (str|int, default `"5"`), `slap.enabled` (bool,
    default true) and `mcp_connectors` (list of `{ id, enabled }` referencing the
    built-in connector catalog, distinct from custom `mcp_servers`). All three commit at
    the next idle moment (`pending_until_idle`), never a restart; added to `public_dict`.
  - **Behavior note (honest).** External-provider chats never trigger a local warm-up
    (using Ollama/OpenAI must not spin up local GPU), so their background auto-titling on
    the utility slot runs only while the loadout is already resident, degrading to "no
    title" when cold rather than forcing a load.
  - **MCP connector catalog (one-click, on top of custom `mcp_servers`).** A curated
    built-in catalog of well-known MCP servers (`mcp/catalog.py`): filesystem, git, fetch,
    memory, everything, sequential-thinking, time: the same reference/community servers
    the openclaw and hermes ecosystems reference in their optional-mcps. `GET
    /v1/mcp/connectors` returns the catalog with each connector's `enabled` flag per
    `settings.mcp_connectors`. Enabling one (`PATCH settings.mcp_connectors` + apply)
    resolves it into an `McpServerSettings` and wires it into the SAME `McpManager` as a
    custom stdio server: identical wave-2 stdio client, `mcp_<id>_<tool>` namespacing and
    `mcp_server_failed`-notice failure handling. Custom `mcp_servers` keep working and win
    any id/name collision. The manager gains `resync()`/`resync_soon()` so a connector
    committed at idle takes effect without a process restart (mirrors provider re-polling);
    an unchanged, still-alive server is left running untouched. A `requires_path` connector
    (filesystem) defaults its root to the user's home. `TODO(v1.1):` per-connector path in
    the `{ id, enabled }` settings shape.
  - **Skill import from other agentskills.io ecosystems.** `POST /v1/skills/import
    { source: "openclaw"|"hermes"|"path", path? }` → `{ imported: [ { name } ], skipped:
    [ { name, reason } ] }` and `suiban skills import <openclaw|hermes|PATH>`
    (`memory/skill_import.py`). Scans a skills directory for `<name>/SKILL.md`, validates
    each with the SAME agentskills.io validator the model-write path uses and copies the
    good ones (SKILL.md + `scripts/` + supporting files) into `~/.bonsai/skills/<name>/`
    marked `source: "imported"`, `verified: false`. Malformed skills are skipped and
    reported, never a crash; a `path` that does not exist is a clean 400
    (`import_source_unavailable`). Known sources: openclaw → `~/.openclaw/workspace/skills`
    (+ a repo `.agents/skills`), hermes → `~/.hermes/skills` (+ a repo
    `optional-skills/<cat>/<name>`), `path` → any directory (scanned recursively). Because
    suiban skills ARE this SKILL.md format, they are portable both ways.
  - **NOTICE.** Credited openclaw (github.com/openclaw/openclaw, MIT) and Hermes
    (github.com/nousresearch/hermes-agent, MIT) for SKILL.md skill-format interoperability
    via the agentskills.io standard (no code copied).
- Feature wave 4 (api.md 2026-07-22b, all additive except the pre-1.0 WhatsApp settings
  reshape):
  - **Confirm gate for file mutations.** `fs_write` and `fs_undo` now share the
    destructive-shell confirm gate: the first call is refused with `status:"denied"`, a
    single-use `confirm_token` bound to the exact operation (write → `(path, content)`;
    undo → the edit's sequence number) and the unified diff; the model re-runs with the
    token to apply. Nothing touches disk before confirmation. `fs_read`/`fs_list` stay
    ungated.
  - **`auto_confirm` bypass.** `ChatRequest.auto_confirm` (bool, code/ultra only, 400 in
    chat) threads a bypass flag through the registry into `ToolContext`; destructive
    shell AND file mutations then run without the gate. Every auto-confirmed action is
    logged (`logger.warning` with command/path), never silent.
  - **Client identities.** The `X-Bonsai-Client` header (`dai`/`sentei`/`other`) selects
    an identity overlay merged into the system prompt on top of the base `identity.md`
    (sentei → coding overlay, dai → general overlay, other → base only). The three
    identity files seed into `~/.bonsai/memory/` on first run; the overlays
    (`identity-dai.md`/`identity-sentei.md`) are editable state files served by
    `GET /v1/memory/state` and `PUT /v1/memory/state/{name}`.
  - **Import chats.** `POST /v1/memory/sessions/import` (providers `openai` · `claude` ·
    `claude-code` · `generic`) parses another tool's export into archived sessions;
    `compress:true` condenses long imports into a seed summary via the utility model.
    Shape mismatch → 400 `import_unrecognized`. Pure offline parsers in
    `memory/importers.py`.
  - **Sessions mode filter.** `GET /v1/memory/sessions?mode=chat|code` filters by mode so
    dai's Chat and Code tabs show separate recents.
  - **WhatsApp QR device-linking** (replaces the Cloud-API-token gateway). Links via the
    WhatsApp Web multi-device protocol: `GET /v1/gateways/whatsapp/qr`
    (`unlinked`/`awaiting_scan`/`linked` + a real `qrcode`-rendered terminal QR) and
    `POST /v1/gateways/whatsapp/unlink`. Settings become `whatsapp.{enabled, linked,
    to_number}` (no secret). Pluggable link backend (`neonize`, optional native dep; a
    stub renders a real QR when absent). HONEST: link+send is unverified against live
    WhatsApp; see KNOWN_ISSUES.md and docs/gateways.md (`TODO(v1.2)`).
  - **Auto-compression toggle.** New `chat.auto_compress` setting (default `true`) gates
    the ~70%-context rolling compression; the compression event still surfaces on rich
    streams.

### Security
- Pre-publication security audit (2026-07-22), all additive (api.md 2026-07-22 entry;
  loopback default behavior unchanged):
  - **Non-loopback API auth.** Binding `server.host` to a non-loopback address
    auto-generates + persists a write-only `server.auth_token` (printed to the console
    once) and requires `Authorization: Bearer <token>` on every route except
    `GET /v1/system/health` (401 `unauthorized`). Loopback binds stay open.
    `serve --host` is treated as the effective bind so it cannot bypass the gate.
    `GET /v1/system` gains `security.{auth_required, remote_agentic, telegram_paired}`.
  - **Telegram inbound authorization.** Default-DENY: only `allowed_chat_ids` reach the
    model; a chat pairs with `/pair <code>` (one-time code printed to the server
    console, never sent over Telegram), paired ids persist to config. Per-chat rate
    limit (`rate_limit_per_min`, default 20). `require_pairing` default true. Gateways
    stay pinned to chat mode; `server.remote_agentic` is reserved and not yet honored.
  - **SSRF hardening in `browse_t1`.** Resolves the host and rejects
    loopback/private/link-local/reserved/multicast/unspecified addresses; follows
    redirects manually (max 5), re-checking the host on every hop.
  - **Prompt-injection defense.** External tool output (browse/`fs_read`/`git_ro`/MCP)
    is wrapped in delimited UNTRUSTED blocks; the skill-context header and every mode
    prompt (v2) state that fetched/file/skill content is data, never instructions. The
    destructive-shell confirm gate remains the boundary.
  - **Jail hardening.** `session_id` sanitized before any filesystem join (traversal
    can't relocate the fs/shell jail); `fs_read`/`fs_write` open with `O_NOFOLLOW`;
    `git_ro` pins `GIT_CEILING_DIRECTORIES` to the jail; the shell subprocess env is
    scrubbed of secret-bearing variables. Research job ids widened to 128-bit.

### Fixed
- Robustness / bug-hunt pass (audit 2026-07-22): every hostile input that used to
  produce a raw traceback or a 500 now degrades to a clean `BonsaiError`, a logged
  warning or an honest failure state. A `hypothesis` fuzz suite
  (`tests/test_fuzz.py`) + hostile-env suite (`tests/test_hostile_env.py`) pin the
  behavior.
  - **FTS5 query builder NUL crash.** A NUL (or other C0 control) byte in a search
    query (e.g. `?q=a%00b`) survived `fts_query` into the quoted `MATCH` expression
    and made SQLite raise `OperationalError('unterminated string')`, a 500 on every
    memory/session/project search. Control bytes are now stripped from tokens
    (`\x00-\x1f`, `\x7f`); the unicode61 tokenizer discarded them anyway, so no
    searchable content is lost.
  - **Malformed `config.toml` / `staged.toml`.** A hand-edited config with a TOML
    syntax error (or saved in a non-UTF-8 encoding) raised a bare
    `TOMLDecodeError`/`UnicodeDecodeError` out of `serve`/`doctor`. Now a clean
    `BonsaiError(config_invalid_toml)` naming the file and the remedy ("fix the TOML
    syntax, or delete it to regenerate defaults").
  - **`serve` port pre-flight.** Binding an occupied port surfaced uvicorn's raw
    `[Errno 98]`. A bind probe now prints "port N is already in use (another suiban?
    change server.port in ~/.bonsai/config.toml)" and exits 1.
  - **SQLite `database is locked`.** The memory, jobs and schedules stores each open
    their own connection to the shared `memory.sqlite`; under concurrent writes a
    second connection could see "database is locked". All three now set
    `PRAGMA busy_timeout=5000` (writers queue instead of erroring), and the chat
    archive write (`MemoryStore.add_message`) downgrades a lock to a logged warning so
    the chat response still returns (a non-lock `OperationalError` still surfaces).

### Performance
- TTFT hot-path pass (audit 2026-07-22, measured, see
  [`docs/benchmarks.md`](docs/benchmarks.md) "TTFT hot path + soak"). Two functions
  that ran on every chat request in `_prepare_loop` were profiled, rewritten, and
  re-measured on identical data; behavior is byte-for-byte unchanged (equivalence
  tests pin it).
  - **`SkillStore.list()` no longer re-reads the disk every request.** It globbed and
    re-parsed every `SKILL.md` + `meta.json` on each call (via `_inject_skill_context`).
    Now the parsed list is cached and invalidated by a cheap stat-only signature (a
    process-local write generation bumped by `put`/`delete`/`mark_verified`, plus each
    skill's `SKILL.md` mtime+size and `meta.json` mtime), so newly saved, deleted,
    re-verified, hand-dropped and out-of-band-edited skills all still appear on the
    next call. Measured ≈4.9× faster (3241 → 666 µs/call at 50 skills).
  - **`enforce_context_budget` is single-pass, no longer O(M²).** It re-estimated the
    whole message list after every popped block and every deleted message. It now
    computes per-message token estimates once and maintains a running total, adjusting
    incrementally. The trim ladder and the `context_trimmed` notice are identical.
    Measured ≈97× faster on a 400-message conversation (16.1 ms → 166 µs/call).
  - In-context compression's pre-first-token utility-model round-trip is confirmed
    inherent (gated at the 70% trigger, best-effort wrapped) and deliberately left in
    place, documented as a correctness cost, not overhead to remove.
- **Bounded per-session reflection counter (leak hunt).**
  `reflection._EXCHANGE_COUNTS` was an unbounded `dict` keyed by `session_id`; a
  long-lived server seeing many sessions grew it without bound. It is now a bounded LRU
  (`OrderedDict`, cap 4096, oldest evicted). Verified the llama-backend stderr ring and
  MCP stderr tail are already bounded `deque(maxlen=...)` (no change).

### Added
- Initial v1 development: FastAPI core, llama-server process manager, VRAM-aware
  scheduler with measured budget table, effort ladder, chat/code/ultra/deep-research
  modes, grammar-constrained agentic loop, FTS5 memory engine + skills, Telegram
  gateway, TurboQuant KV-cache patchset (TQ4_0/TQ3_0) for the PrismML llama.cpp fork,
  `suiban bench kv`, bootstrap installer.
- `tool_result` SSE events carry the single-use `confirm_token` when a confirm-gated
  tool denies an operation (api.md v1 additive change, 2026-07-21); the key is absent
  on `ok`/`error` results.

- `suiban install turboquant` now finishes the job: after building, it builds
  `llama-server` itself and promotes it plus the fork's shared libraries into
  `~/.bonsai/bin/<backend>/`, writing the `TURBOQUANT` marker (previously the CLI
  claimed to "swap the binary" but no swap step existed).
- First-launch VRAM measurement is wired: real slot launches are bracketed by
  telemetry snapshots and the machine-dependent buffer cost (delta minus exact
  weights/KV math) is persisted to `~/.bonsai/budget.json`; implausible deltas
  (< 80% of weights) are discarded. Measured on the reference 8 GB machine:
  27B buffers 931 MiB (prior 1229), 1.7B 269 MiB (prior 614).
- Wave 2 (api.md additive 2026-07-21b): **projects** (`/v1/projects` CRUD + plain-text
  knowledge docs, FTS5-searched and injected into member sessions; chat `project_id`;
  `/v1/memory/sessions?project_id=` filter), **schedules** (`/v1/schedules` CRUD +
  run-now; daily/weekly/interval cadences; runs are ordinary archived chat sessions),
  **session auto-titling** (utility model, thinking off, fire-and-forget after the
  first exchange), **state-file editing over HTTP** (`PUT /v1/memory/state/{name}`,
  identity.md included: 400 `state_file_too_large` above the cap, 404 outside the
  known set, edits re-mirrored into FTS immediately), **code-mode `workdir`** (jail
  the session's fs/shell/git_ro tools to a validated user directory; remembered on
  the session row for continuations; 400 `workdir_invalid` otherwise) and the
  **WhatsApp outbound gateway** (Business Cloud API text pings for research
  completions and scheduled runs; `access_token` write-only like the Telegram token;
  send failures surface a `whatsapp_send_failed` notice; inbound relay is
  TODO(v1.2), needs a public webhook).

- Wave 3 (api.md additive 2026-07-21c): **external providers** (settings
  `providers[]`: `ollama`/`openai` kinds, `api_key` write-only; enabled providers
  are polled for `{base_url}/v1/models` on boot and after apply, cached and listed
  in `GET /v1/models` as `<name>/<model>` with `bonsai.external`; chat routes
  provider-prefixed ids as plain OpenAI proxies, mode `chat` only (400
  `external_model_mode`), effort maps to sampling only, both stream envelopes,
  client tools pass through, memory/project injection + archiving + auto-titling
  work; unreachable providers surface a `provider_unreachable` notice and their
  models 404 honestly), **pluggable web search for deep research** (settings
  `search`: duckduckgo keyless default with an honestly-fragile HTML scrape,
  searxng, brave, tavily, serper; the gather stage now searches the plan's
  sub-questions and fetches the top results via browse_t1; total search failure
  degrades to the old plan-URL behavior with a note at the top of the report;
  `POST /v1/system/search_test` powers the settings test button and never throws)
  and **reflection + recall** (post-task reflection on the orchestrator after
  chat/code exchanges: background, thinking off, small max_tokens, at most once
  per session per 3 exchanges, memory_write-or-"none", workers/external/ultra/
  anonymous excluded; `session_search` tool added and recall registered in BOTH
  chat and code toolsets; light automatic injection of top FTS5 memory hits for
  the latest user message, delimited and budget-capped like project docs).

- TurboQuant rigor + perf pass (deep-detail session):
  - **Patch 0007** (clean-room, like the rest of the CUDA patchset): warp-shfl
    butterfly fast path for the flash-attention vec kernel's TurboQuant V dequant, the 0006 TODO. Measured on the 8 GB tier (1-bit 27B, K=q8_0): decode at 16K
    prefilled depth 6.51 → 19.47 t/s (tq4_0, 2.99×) and 5.99 → 18.62 t/s (tq3_0,
    3.11×), numerics unchanged within fp32 round-off. Lane pattern verified
    warp-wide at runtime (`__all_sync`); closed-form path kept as fallback and for
    HIP. Full before/after tables: `docs/benchmarks.md` (new).
  - `vendor/run_kernel_tests.py` grew property-style rigor: heavy-tailed
    (Student-t df=3) and per-channel-outlier (×100 spikes) distributions over 10
    seeds with MEASURED envelopes (honest finding: total MSE does not degrade, but
    under channel outliers the non-outlier channels reconstruct worse than zeroing,
    asserted as a band, not hidden), head-dim rows 64/96/128/256 layout checks
    and CPU-vs-CUDA-algorithm parity on every new case.
  - Compiled-CUDA numeric validation ON the GPU (stage 7 +
    `vendor/tools/tq_cuda_numeric.cpp`): dequantize rows, SET_ROWS KV-write
    quantize and FLASH_ATTN_EXT vec/prefill paths run on-device against the CPU
    reference; measured max deviations recorded in `vendor/README.md` (which also
    lost its stale "not yet exercised on a GPU" claims: CUDA E2E was validated in
    the integration pass and re-validated here).
  - `suiban bench kv` battery: real `llama-perplexity` wired in as a subprocess
    against the installed binary (replaces the n/a OpenAI logprob probe; honest
    n/a note when a binary ships without the tool), needle ladder 4k/8k/16k with
    ctx-honest "not run" labels (largest slot ctx that boots wins) and a canned
    ~30-turn agentic replay fixture scored on answer stability.
    `suiban install turboquant` now promotes `llama-perplexity` alongside
    `llama-server`.
  - `apply_patches.py` idempotency fix: applied-detection now uses the top of the
    patch stack (a later patch editing an earlier patch's lines used to break
    per-patch reverse-apply detection and abort re-runs).

- Release hygiene (pre-publication audit, workstream R):
  - **Download integrity for the fork binaries (H7):** a checked-in SHA-256 manifest
    (`installer/assets_sha256.json`, digests captured from the GitHub releases API for
    the pinned tag `prism-b9596-9fcaed7`) is now verified after each archive download;
    a mismatch is a hard failure (`asset_sha256_mismatch`) and the archive is deleted
    before extraction. A regression test asserts every asset the installer can select
    for a supported os/backend/arch has a checked-in digest, so no real install falls
    back to the un-verified path.
  - `NOTICE` file at the repo root (Apache-2.0 §4(d) attribution): PrismML Bonsai
    models + llama.cpp fork, ggml/llama.cpp (MIT), TurboQuant (arXiv:2504.19874), the
    MIT `Aaryan-Kapoor/llama.cpp` TQ3_0 port, the `ggml-org/llama.cpp#20969` community
    discussion, the (unlicensed, cite-only) community CUDA gist, agentskills.io, the
    Hermes-inspired memory design and third-party Python deps.
  - `KNOWN_ISSUES.md` at the repo root: honest, public boundaries (shell tool is not a
    sandbox; MCP servers run unsandboxed; LAN bind requires the bearer token; download
    integrity model; macOS pdeathsig gap; Playwright tier-2 inert; 8 GB family
    auto-degrade).
  - Dependency audits run clean: `pip-audit` (no known vulnerabilities) and
    `bandit -ll` (0 High; the Medium `B608` hits are false positives: constant
    column-name interpolation with `?`-bound values, documented in `KNOWN_ISSUES.md`).
- Wave 4 (api.md additive 2026-07-22b): **SLAP** (Structured Lightweight Agent
  Protocol) is now how Ultra coordinates its sub-agents.
  - Self-contained vendored implementation in `suiban/slap/`: the 10 canonical JSON
    schemas copied byte-identically from the separate `slap` repo (no cross-repo import: suiban vendors SLAP exactly as it vendors `api.md`), plus schema
    load/validate (clean errors, no jsonschema dep) and builders for all nine
    operations. A drift-guard test asserts the vendored schemas stay byte-identical to
    the canonical repo.
  - **Ultra rewrite** (`modes/ultra.py`): dispatch is represented as SLAP messages:
    each executing slot advertises a `capability`, the orchestrator emits a validated
    `assign` per sub-task, each worker returns a `result` (evidence-linked claims,
    risks, confidence) and synthesis emits a `decide`. Every message validates against
    the vendored schemas before use; a validation failure degrades to the prior
    structured-dict path with a `slap_degraded` notice, never a crash. All wave-2
    latency bounds are unchanged (sequential caps, effort inheritance, per-sub-task
    timeouts, tool_result cap).
  - **Volatile per-agent system prompts:** the plan may carry a per-sub-task
    `system_prompt` the orchestrator writes on the fly; it is the worker's system
    message for that one agent lifetime, then discarded: never reused across agents,
    never archived to memory and stripped from the `assign` before it is recorded in
    the trace. `ultra_worker.md` is now the fallback used when a sub-task omits a
    system prompt (ultra planning prompt bumped to v3 to instruct writing one).
  - `agent_spawn`/`agent_result` stream events now carry the SLAP `task_id`.
  - **`/v1/slap` observability** (`routers/slap_router.py`): `GET /v1/slap`
    (version/profiles/operations), `GET /v1/slap/schema/{operation}` (vendored schema,
    404 unknown), `GET /v1/slap/trace/{session_id}` (a completed Ultra run's validated
    transcript, empty list if none). Traces live in a bounded in-process store (memory
    is Phase C, no new table).

### Changed
- **Model-weight download integrity hardened (H7):** a byte-size deviation >2% from
  the pinned recon table is now a **hard failure** (`model_size_mismatch`, the file is
  deleted and never recorded), not a warning: a truncated or swapped weight must
  never be loaded. `huggingface_hub`'s own etag/SHA remains the transport integrity
  source; the size table is the tripwire on top of it.
- `installer/models.py` wraps `hf_hub_download` so a disk-full / permission / network
  `OSError` surfaces as a clean `BonsaiError` (`model_download_failed`) with a fix hint
  instead of a raw traceback out of `suiban install models`.

### Fixed
- llama-server flags: the pinned fork's `-fa`/`--flash-attn` takes a mandatory
  `on|off|auto` value, so quantized-KV slots now pass `-fa on` instead of bare `-fa`
  (which would have swallowed the next argv token). Verified against tag
  `prism-b9596-9fcaed7` `common/arg.cpp`.
- Thinking controls now match verified fork behavior: the per-request
  `thinking_budget_tokens` field from the model docs is ignored by llama-server, so
  effort now drives the Qwen-style `chat_template_kwargs.enable_thinking` per request
  (low ⇒ off) with the numeric ceiling enforced slot-wide via `--reasoning-budget`
  (xhigh budget bounded by 40% of slot ctx). Graded per-request budgets: TODO(v1.1).
- HF model table corrected against the live tree API: ternary lives in separate
  `prism-ml/Ternary-Bonsai-*-gguf` repos (not the same repo as 1-bit); file picking is
  exact-suffix so the unsupported `PQ2_0` / mainline `Q2_0_g64` variants can never be
  selected; exact byte sizes recorded; DSpark drafter downloadable via
  `install models --dspark` (opt-in).
- Release asset resolver rewritten against the real asset list: linux CUDA assets are
  `linux-cuda-12.4|12.8-…tar.gz` while CPU/Vulkan/ROCm use `ubuntu-…` stems; Windows
  CUDA uses the odd `llama-prism-b1-<sha>` prefix + separate cudart zip; archives are
  `.tar.gz` on linux/macos (extractor now handles tar + recreates the load-bearing
  soname symlinks that flattening dropped).
- Asset downloads retry with HTTP-Range resume (4 attempts, 120 s read timeout): release archives are hundreds of MB and residential links stall.
- `RealBackend` sets `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` to the binary's directory
  when spawning llama-server (the fork's shared libs live there; spawn failed without
  it).
- `suiban bench kv` benchmarks the 27B family actually on disk (with a printed note)
  when the configured family is absent, matching the planner's degradation instead of
  erroring on tiers that run 1-bit.
- CORS middleware for first-party UIs: vite dev (`localhost:5173`) and the Tauri
  webview origins (`tauri://localhost`, `http://tauri.localhost`) can now call the
  loopback API from a browser context (previously every dai fetch failed CORS).

- Refinement pass (scheduler/ultra/jobs correctness):
  - **Ultra latency bounded** (top-ranked live gap: a trivial sequential Ultra ran
    past 10 minutes at pinned xhigh). Sub-tasks now inherit the REQUEST effort,
    capped at `mid` on sequential tiers; sub-task count capped (worker count when
    parallel, 3 sequential); per-sub-task wall-clock timeouts (240 s low/mid, 480 s
    high+) cancel the sub-agent and surface a `failed` `agent_result` plus an
    `ultra_subtask_timeout` notice the orchestrator synthesizes around. Every
    degrade (sequential fallback, effort cap, plan truncation, timeout) emits a
    notice with a one-line reason.
  - **Client-disconnect abort**: non-streaming chat runs poll
    `request.is_disconnected()` alongside the backend task and cancel it when the
    client is gone (streaming paths already abort via generator cancellation): a closed tab no longer orphans a GPU-burning run.
  - **Per-slot serialization (SlotGate)**: one run holds a slot, at most 4 wait,
    beyond that 429 `overloaded_error` (`slot_queue_full`); `stream_events` chats
    that wait >1 s get a `slot_queued` ("queued behind N") notice. Research jobs
    take the orchestrator gate per pipeline STEP so chats interleave with a running
    job (fairness choice documented in research/wiring.py + docs/research.md);
    scheduled runs wait uncapped.
  - **Research cancel is real**: DELETE awaits the task unwind (bounded ~10 s) so
    the in-flight llama-server request is aborted before the response; submits
    during a still-unwinding task answer 429 (no double-use).
  - **Slot stderr ring buffer** (last ~200 lines, was DEVNULL): tails are quoted
    into `slot_failed` notices (including the new restart-give-up notice), and the
    startup log feeds a **hybrid-attention runtime probe**: llama-server now runs
    at `-lv 4` and the `llama_kv_cache: size = … N layers` line (format captured
    live from the pinned fork) is checked against the 27B's expected 16-of-64
    hybrid allocation; mismatch emits `kv_layers_mismatch`, never a crash.
  - **`effort_default` wired** (was dead): request default resolves
    `req.effort` > `settings.effort_default` > mode default, for chat AND research
    submits. The setting is now optional/unset by default so per-mode defaults
    keep working until a user explicitly overrides them.
  - Chaos tests: a stub llama-server executable exercises RealBackend
    spawn/health/stderr/crash-restart/backoff/give-up/SIGKILL for real (no GPU);
    planner matrices extended to [24,8], [8,8], 3-GPU and table-driven per-tier
    assertions; NVML/rocm-smi providers unit-tested against mocked drivers.

- `kv` and `dspark_enabled` reclassified as `requires_restart` in
  `/v1/system/apply` responses (api.md additive 2026-07-21d): they map to
  llama-server launch flags and slots never relaunch mid-process, so the old
  `pending_until_idle` label was untrue.

- Refinement pass (loop/tools + MCP):
  - **Repair budget is per RUN** (was per episode): at most 2 repair prompts total
    per agentic run: alternating malformed/valid calls no longer refill it; further
    malformed calls are abandoned as graceful step failures. The loop now counts
    malformed/repaired/abandoned calls per run, logs them and surfaces them as
    optional `usage`-event fields (api.md additive 2026-07-21d) so malformed-call
    rates are measurable from transcripts. Regression transcript fixtures
    (`tests/fixtures/transcripts/`: happy, malformed-then-repaired,
    repair-exhausted, 20-step long-haul) replay through the real loop; the 20-step
    fixture asserts end-to-end task completion.
  - **MCP stdio client** (`suiban/mcp/`, api.md §8 additive 2026-07-21d):
    `settings.mcp_servers[]` ({name kebab-case, command, args, enabled};
    requires_restart, servers start/stop with the app lifespan like gateways).
    JSON-RPC 2.0 over the server subprocess's stdio: initialize handshake (spec rev
    2025-06-18), notifications/initialized, tools/list (paginated), tools/call with
    per-call timeout and text-content extraction. Connected servers' tools join
    chat/code runs namespaced `mcp_<server>_<tool>` with their JSON schemas passed
    through verbatim; a failed start or mid-run crash emits an `mcp_server_failed`
    notice and removes the server's tools, never a crash. Verified live against
    `@modelcontextprotocol/server-everything` (13 tools; echo round-trip) and
    `@modelcontextprotocol/server-filesystem /tmp` (14 tools; list_directory) via
    npx; CI runs a bundled stdlib-only fixture server as a real subprocess.
  - **Code-mode edit safety**: `fs_write` now returns a unified diff (before-state
    vs new content, computed BEFORE applying) in the tool_result, and journals every
    edit under `<workdir>/.suiban-undo/` (prior content, last 20 entries, written
    before the edit applies). New `fs_undo` tool (code mode) reverts the last edit
    or one by number; the journal dir is hidden from `fs_list` and refused as a
    write target.
  - **Sandbox light pass** (full security audit is next session; seams labeled with
    AUDIT SEAM comments): destructive-shell detection now also catches
    `find -delete`, deletion primitives interpreters reach for
    (unlink/rmtree/os.remove/fs.rmSync) and interpreter `-c`/`-e` one-liners that
    spawn subprocesses; the Playwright stub pins profile AND downloads dirs under
    `~/.bonsai/browser/`; the fs resolve-then-act TOCTOU and the MCP
    config-as-trust-boundary are documented for the audit.

- Refinement pass (memory / skills / compression):
  - **Skill schema validation**: model-driven `skill_save`/`skill_improve` now
    validate agentskills.io frontmatter (closed `---` block of key: value pairs,
    kebab-case `name` matching the skill, non-empty `description`) and reject
    invalid files with a structured `400 skill_invalid`; the reflection path
    retries exactly once with the validator's full error appended, then gives up
    quietly. Human writes (HTTP PUT, hand-dropped dirs) stay lenient as before.
  - **Skill verification lifecycle**: skills carry a persisted `verified` flag
    (additive optional field on the api.md Skill object): false on every content
    write, flipped true when a run that had the skill injected completes
    successfully. New skill injection for local agentic chat/code runs: up to 2
    name/description-matching skills as one delimited system block,
    verified-first, unverified entries labeled `[unverified]`.
  - **Compression fidelity**: the utility summarizer prompt now carries an
    explicit keep-list (names, numbers, dates, paths, URLs, error messages,
    decisions, [ids]); a planted-fact generator (`memory/fidelity.py`) backs
    modelless mechanics tests (facts kept by the summarizer survive folding and
    re-compression losslessly; protected-tail facts never reach the summarizer)
    plus an opt-in real-model harness (`SUIBAN_LIVE_FIDELITY=1`) asserting >= 80%
    fact survival on the live stack.
  - **Adaptive verbatim window**: compression's protected recent tail scales with
    the slot context (4 below 16K, 6 at 16K, 8 at 32K+).
  - **Context-overflow guard**: requests estimated past 90% of the slot context
    after compression are trimmed on a fixed ladder: injected memory blocks, then
    skill blocks, then project-doc blocks (lowest-scored first within each), then
    the oldest messages beyond the protected window, with a `context_trimmed`
    notice (stream) / warning log (non-stream); llama-server is never handed an
    over-context request silently.
  - **FTS5 relevance**: all FTS tables migrated to
    `unicode61 remove_diacritics 2` with `prefix='2 3'` indexes (two-way diacritic
    folding; >= 3-char query tokens become prefix terms, replacing porter
    stemming); ~64-char snippet windows; idempotent shadow-table rebuild for
    existing databases; a 10-case relevance eval set (diacritics, prefixes,
    multi-term) pinned in tests.

### Documentation
- Refinement pass (docs / install honesty):
  - `docs/hardware.md` buffer priors reconciled with `sched/budget.py`'s real
    `BUFFER_PRIOR_GIB` values (27B 1.2 GiB, others 0.6 GiB; the old 1.19/0.66/0.34/
    0.25 table matched nothing in code) and all tier totals recomputed (24 GB ≈18.5,
    16 GB ≈14.9, 12 GB ≈10.8, 8 GB ≈7.6 GiB analytic). New one-machine "Measured on
    real hardware" section: measured buffers from `budget.json` (27B 970 MiB / 1.7B
    248 MiB, with first-launch 931/269 noted as normal drift), the booted 16K-ctx
    8 GB loadout arithmetic (predicted within ~20 MiB of slot-reported VRAM), and
    decode speeds cited from `docs/benchmarks.md`.
  - budget.json keying corrected in hardware.md and architecture.md: measured
    overrides are keyed model+family (buffers/weights only; KV is always exact
    analytic math), not (model, family, ctx, kv_config) as previously claimed.
  - README quickstart made copy-paste-true: `bootstrap.sh` alone only syncs the
    venv, so the install-binaries / install-models steps (with real download sizes)
    now appear in order with `suiban doctor` as the gate; features list extended
    with the shipped MCP servers, external providers and research web search.
  - `bootstrap.sh` gained `--full`: doctor + install binaries + install models,
    interactively, prompting before every large download, ending on a doctor gate.
  - Architecture diagram now names the SlotGate queue; the stale "benchmarks await
    the integration phase" seam bullet replaced with pointers to the measured
    hardware.md §6 / benchmarks.md tables; `suiban bench kv` no longer labeled
    "(integration phase)" in hardware.md.
  - Added `CONTRIBUTING.md` (dev setup, checks, contract law, secrets rules, PR
    expectations) and `FAQ.md` (seeded from real questions: serving-but-chats-fail,
    sequential Ultra on 8 GB, why K stays q8_0, TurboQuant-less prebuilts, CUDA 12
    runtime vs 13). Removed the README's phantom overview-screenshot reference.
  - Attribution completed (pre-publication audit): the RHT-substitution consensus is
    now cited by name (the llama.cpp community discussion `ggml-org/llama.cpp#20969`) in `docs/turboquant.md` and `vendor/README.md` (previously "community
    discussion consensus" generically); and `docs/memory.md` now credits the
    Hermes-inspired layered on-demand memory design (inspiration only, no borrowed code, matching the new `NOTICE`).
