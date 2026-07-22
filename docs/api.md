# suiban HTTP API: v1 (FROZEN)

This document is the **single coordination point** between suiban, dai and sentei.
Clients speak plain HTTP to suiban; no other integration mechanism exists.

- Base URL: `http://127.0.0.1:8686` (host/port from `~/.bonsai/config.toml`; suiban
  binds loopback only by default; see Security notes).
- Every response carries header `X-Bonsai-Api-Version: 1`.
- Request/response bodies: `application/json` unless stated (SSE streams:
  `text/event-stream`; research reports: `text/markdown`).
- **Freeze policy:** v1 shapes below may gain *optional* fields (additive, documented in
  the Changelog at the bottom). Nothing is renamed, removed or retyped within v1.
- Authentication: none in v1 (loopback bind). Any `Authorization` header is ignored.

## Error envelope (all endpoints)

Non-2xx responses:

```json
{ "error": { "type": "invalid_request_error", "message": "human-readable detail", "code": "kv_preset_unknown" } }
```

`type` ∈ `invalid_request_error` (400) · `not_found_error` (404) · `conflict_error`
(409, e.g. apply while a run is active) · `overloaded_error` (429, loadout busy) ·
`server_error` (500). `code` is a stable machine string; may be `null`.

---

## 1. `POST /v1/chat/completions`: OpenAI-compatible chat

Strict OpenAI Chat Completions compatibility, plus optional bonsai extensions. Third-party
OpenAI clients work unmodified and get byte-compatible behavior.

**Request (OpenAI subset honored):** `model`, `messages` (string content or multimodal
parts; `image_url` parts accept `data:` base64 URIs; images are ALWAYS routed to the
27B orchestrator), `stream`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`stop`, `tools`, `tool_choice`, `response_format` (incl. `json_schema`).

`model` values: `bonsai-auto` (recommended: the orchestrator of the active loadout),
or a concrete id from `/v1/models` (`bonsai-27b`, `bonsai-8b`, `bonsai-4b`,
`bonsai-1.7b`). Requesting a non-resident model is a 409 (`model_not_resident`); no
mid-run loads. **External models** (additive 2026-07-21c): provider-prefixed ids from
`/v1/models` (e.g. `ollama/llama3.2`) route the session to that configured provider.
External sessions are `mode:"chat"` only (400 `external_model_mode` otherwise), still
get memory injection/archiving/titling, but NO VRAM scheduling, thinking control or
grammar guarantees; effort maps to sampling only, tools pass through only if the
provider supports them. suiban's own loadout stays the default (`bonsai-auto`).

**Bonsai extension fields (all optional):**

| Field | Type | Meaning |
|---|---|---|
| `mode` | `"chat"` \| `"code"` \| `"ultra"` | Orchestration mode. Default `"chat"`. Deep research is NOT a mode here; use `/v1/jobs`. |
| `effort` | `"low"` \| `"mid"` \| `"high"` \| `"xhigh"` \| `"max"` | Thinking budget + tool-loop ceiling. Default from mode. |
| `session_id` | string | Continuity handle for memory injection, compression and archive. Client-generated UUID; reuse to continue a conversation. |
| `stream_events` | bool | With `stream:true`: switch SSE payloads to the rich envelope below. Default `false` (pure OpenAI chunks). |
| `project_id` | string | Optional (additive 2026-07-21b). Binds the session to a project: relevant excerpts from the project's knowledge docs (FTS5) are injected, and the session lists under the project. Unknown id → 404 `project_not_found`. |
| `workdir` | string | Optional, `mode:"code"` only (additive 2026-07-21b). Absolute path to an existing directory; the session's fs/shell tools are jailed there instead of the default per-session workdir. Invalid → 400 `workdir_invalid`. |
| `auto_confirm` | bool | Optional, `mode:"code"`/`"ultra"` only (additive 2026-07-22b). Default `false`. When `true`, the session BYPASSES the confirmation gate for destructive shell commands AND file mutations; tools run without emitting `denied`/`confirm_token`. This is a DANGEROUS power-user setting; the server logs every auto-confirmed destructive action. Ignored in chat mode. |

**`X-Bonsai-Client` request header** (additive 2026-07-22b): `dai` · `sentei` · `other`
(default `other`). Selects the client-identity overlay merged into the system prompt
(`sentei` gets the coding-focused identity, `dai` the general one) on top of the base
`identity.md`. Overlays are editable state files (`identity-<client>.md`), see §5.

**Confirmation gate (extended 2026-07-22b):** the `denied` + `confirm_token` flow (see
"Destructive-op confirmation flow") now covers **file mutations** (`fs_write`, `fs_undo`)
as well as destructive shell commands. The `tool_result` denial carries a unified diff in
`summary` for file edits. `auto_confirm:true` skips it. Clients (dai/sentei) render the
diff/command and Approve/Decline.

**Non-streaming response:** standard OpenAI `chat.completion` object. Extension:
`"bonsai": { "mode", "effort", "slot": "orchestrator", "session_id" }`.
`usage` gains optional `thinking_tokens`.

**Streaming (`stream:true`, default envelope):** standard OpenAI `chat.completion.chunk`
SSE (`data: {...}\n\n`, terminated by `data: [DONE]`). Tool calls stream as OpenAI
`delta.tool_calls`.

**Streaming with `stream_events:true`:** each SSE line is `data: {"type": ...}`;
terminated by `{"type":"done"}` then `data: [DONE]`. Event types (union; clients MUST
ignore unknown types):

| type | payload fields |
|---|---|
| `delta` | `text` |
| `thinking_status` | `phase` (`"thinking"`\|`"answering"`), `thinking_tokens` (int so far) |
| `tool_call` | `id`, `name`, `arguments` (object) |
| `tool_result` | `id`, `name`, `status` (`"ok"`\|`"error"`\|`"denied"`), `summary` (string, truncated), `confirm_token` (string, only when `status:"denied"`) |
| `plan` | `steps` (string[]), code mode plan before acting |
| `agent_spawn` | `agent_id`, `model`, `task` (one-line), `effort` |
| `agent_result` | `agent_id`, `status` (`"ok"`\|`"failed"`), `summary` |
| `compression` | `trigger_pct` (number), `messages_summarized` (int) |
| `notice` | `level` (`"info"`\|`"warn"`), `code`, `message`, e.g. TurboQuant fallback, family degradation |
| `usage` | `prompt_tokens`, `completion_tokens`, `thinking_tokens`; optional `malformed_calls`, `repaired_calls`, `abandoned_calls` (additive 2026-07-21d; present only when an agentic run had malformed tool calls) |
| `done` | `finish_reason` (`"stop"`\|`"length"`\|`"tool_calls"`\|`"cancelled"`\|`"error"`) |
| `error` | `error` (object: `type`, `message`); stream aborts after this |

## 2. `GET /v1/models`

OpenAI-compatible list plus metadata:

```json
{ "object": "list", "data": [ {
  "id": "bonsai-27b", "object": "model", "owned_by": "prism-ml",
  "bonsai": { "family": "ternary", "quant": "Q2_0", "role": "orchestrator",
              "resident": true, "ctx": 32768, "vision": true,
              "downloaded_families": ["ternary", "1bit"] } } ] }
```

`role` ∈ `orchestrator` · `worker` · `utility` · `none` (installed, not in loadout).

External provider models (additive 2026-07-21c) are appended with ids
`<provider>/<model>` and `"bonsai": { "external": true, "provider": "<name>",
"role": "none", "resident": <provider reachable at last refresh> }`; other bonsai
fields are best-effort or null for external entries.

## 3. `/v1/jobs`: deep research

- `POST /v1/jobs` body `{ "type": "deep_research", "query": string, "effort": effort? }`
  → **202** `{ "id": "job_...", "state": "queued" }`. Unknown `type` → 400.
- `GET /v1/jobs` → `{ "jobs": [JobStatus] }` (newest first).
- `GET /v1/jobs/{id}` → JobStatus:
  `{ "id", "type", "query", "state": "queued"|"running"|"completed"|"failed"|"cancelled",
     "stage": string|null, "percent": 0-100, "created_at", "started_at", "finished_at",
     "error": string|null }`.
  `stage` is COARSE ("collecting sources", "cross-checking", "writing report").
  Internals (queries, URLs, drafts, sub-agent chatter) are never exposed. This is a
  product rule, not a gap.
- `GET /v1/jobs/{id}/events` streams SSE: `data: {"type":"progress","stage":...,"percent":...}`
  on change, `data: {"type":"state","state":...}` on transitions; closes after terminal
  state.
- `GET /v1/jobs/{id}/report` returns `text/markdown` (404 until `completed`).
- `DELETE /v1/jobs/{id}` → `{ "id", "state": "cancelled" }` (idempotent).

Completion also triggers gateway notifications (e.g. Telegram ping) if configured.

## 4. `/v1/system`

### `GET /v1/system`

```json
{
  "version": "0.1.0",
  "uptime_s": 1234,
  "gpus": [ { "index": 0, "name": "RTX 4090", "vram_total_mb": 24564,
              "vram_used_mb": 18211, "source": "nvml" } ],
  "telemetry_source": "nvml",
  "loadout": {
    "planned_at": "2026-07-21T12:00:00Z",
    "tier": "24gb",
    "slots": [ { "slot_id": "orchestrator", "role": "orchestrator",
                 "model": "bonsai-27b", "family": "ternary", "quant": "Q2_0",
                 "ctx": 32768, "gpu": 0, "port": 8701, "state": "ready",
                 "vram_mb": 9530, "mmproj": true, "dspark": false } ],
    "headroom_mb": 5600
  },
  "capabilities": { "vision": true, "browse_t2": true, "skill_writes": true,
                    "ultra_parallel": true },
  "kv": { "k_type": "q8_0", "v_type": "tq4_0",
          "turboquant": { "enabled": true, "preset": "recommended",
                          "backend_supported": true, "fallback_active": false,
                          "fallback_reason": null } },
  "quant_family": { "configured": "ternary", "effective": "ternary",
                    "degraded": false, "reason": null },
  "dspark": { "enabled": false, "available": true },
  "jobs_active": 0,
  "notices": [ { "level": "warn", "code": "turboquant_prebuilt_fallback",
                 "message": "TurboQuant kernels not present in prebuilt binary; using q8_0/q8_0. Run: suiban install turboquant" } ]
}
```

`gpus` is `null` when no GPU telemetry exists (CPU-only); `telemetry_source` ∈
`nvml` · `rocm-smi` · `metal` · `ram` (CPU-only fallback). `kv.v_type` ∈ `tq4_0` ·
`tq3_0` · `q8_0` · `f16`. `turboquant.preset` ∈ `recommended` (TQ4) · `aggressive`
(TQ3) · `off`.

### `GET /v1/system/budget`

`{ "measured": bool, "rows": [ { "model", "family", "ctx", "kv_config",
"weights_mb", "kv_mb", "buffers_mb", "total_mb", "source": "analytic"|"measured" } ] }`

### `GET /v1/system/health`

`{ "status": "ok"|"starting"|"degraded", "checks": { "binary": true, "models": true,
"telemetry": true, "slots_ready": 3, "slots_total": 3 } }`. 200 always (status carries
the truth); used as the readiness probe.

### `POST /v1/system/apply`

Commits staged settings (see `/v1/settings`). Takes effect at the next idle moment,
**never mid-run**. → `{ "applied": bool, "requires_restart": ["quant_family"],
"pending_until_idle": ["kv"] }`. 409 `conflict_error` if a hard-blocked change is staged
(e.g. family switch while its download is still running).

## 5. `/v1/memory`

MemoryEntry: `{ "id": "mem_...", "layer": "identity"|"state"|"archive", "title",
"content", "tags": string[], "created_at", "updated_at", "source_session": string|null }`

- `GET /v1/memory?layer=&limit=&offset=` → `{ "entries": [...], "total": n }`
- `GET /v1/memory/search?q=&limit=` → `{ "results": [ { "entry": MemoryEntry,
  "score": number, "snippet": string } ] }` (FTS5 `bm25` scoring)
- `POST /v1/memory` `{ "layer": "state"|"archive", "title", "content", "tags"? }` → 201 MemoryEntry
- `PUT /v1/memory/{id}` `{ "title"?, "content"?, "tags"? }` → MemoryEntry
- `DELETE /v1/memory/{id}` → 204
- `GET /v1/memory/state` → `{ "files": [ { "name", "content", "bytes", "max_bytes" } ] }`
  (the bounded state files, verbatim; includes `identity.md` and the client overlays
  `identity-dai.md` / `identity-sentei.md`)
- `PUT /v1/memory/state/{name}` `{ "content" }` → the updated file object (additive
  2026-07-21b; human editing of `identity.md`, the client overlays and state files over
  HTTP; 400 `state_file_too_large` above `max_bytes`, 404 for unknown names; no new
  files are creatable through this route)
- `DELETE /v1/memory/state/{name}` → 204 (additive 2026-07-22d; removes one bounded state
  file and its FTS mirror; 404 `state_file_unknown` for names outside the known set; 400
  `identity_read_only` for `identity.md` and the client overlays, which are never
  deletable over HTTP)
- `GET /v1/memory/sessions?q=&mode=&limit=&offset=&project_id=` → `{ "sessions": [ { "id",
  "title", "mode", "project_id", "started_at", "ended_at", "message_count" } ] }` (`q`
  searches FTS5 archive; `mode`=`chat`|`code` filters, additive 2026-07-22b, so dai's
  Chat and Code tabs show separate recents). `title` starts null and is auto-generated by
  the utility model after the first exchange.
- `GET /v1/memory/sessions/{id}` → `{ "session": {...}, "messages": [ { "role",
  "content", "created_at" } ] }`; powers session restore/resume in dai/sentei.
- `DELETE /v1/memory/sessions/{id}` → 204 (additive 2026-07-22d; deletes an archived
  session/chat and its transcript, keeping the FTS index in step; 404 `session_not_found`
  for an unknown id). Powers "remove chat" in dai's recents and memory browser.
- `POST /v1/memory/sessions/import` (additive 2026-07-22b) `{ "provider":
  "openai"|"claude"|"claude-code"|"generic", "data": <export payload>, "mode":
  "chat"|"code"?, "compress": bool? }` → `{ "imported": [ { "id", "title",
  "message_count" } ] }`. Parses another provider's exported conversation(s) into archived
  sessions (400 `import_unrecognized` if the payload does not match the provider shape).
  With `compress:true` the utility model condenses long imports into a seed summary so a
  resumed session starts inside context. Used by dai's "import chats" and sentei's
  `resume-claude`.

## 6. `/v1/skills`

Skill: `{ "name": "kebab-case", "description", "version": int, "updated_at",
"source": "seed"|"learned"|"human", "content": "<agentskills.io-compatible markdown>" }`

- `GET /v1/skills` → `{ "skills": [Skill-without-content] }`
- `GET /v1/skills/{name}` → Skill
- `PUT /v1/skills/{name}` `{ "description"?, "content" }` → Skill (`source` becomes
  `human`, `version` increments)
- `DELETE /v1/skills/{name}` → 204

Model-driven skill creation/improvement is NOT on this surface. It happens only inside
the 27B's post-task reflection (server-enforced).

## 7. `/v1/modes`

- `GET /v1/modes` → `{ "modes": [ { "name": "chat"|"code"|"ultra"|"deep_research",
  "description", "system_prompt_version": "code@3", "tools": string[],
  "default_effort": effort, "endpoint": "/v1/chat/completions"|"/v1/jobs" } ] }`
- `GET /v1/modes/{name}` → same single object. Prompt text itself is not exposed.

## 8. `/v1/settings`

`GET /v1/settings` → current + staged:

```json
{ "current": Settings, "staged": Settings|null }
```

Settings:

```json
{
  "quant_family": "ternary",
  "kv": { "turboquant_enabled": true, "preset": "recommended" },
  "chat": { "auto_compress": true },
  "dspark_enabled": false,
  "effort_default": "mid",
  "loadout": { "prefer_workers": 2, "worker_ctx": 16384, "orchestrator_ctx": 32768 },
  "browse": { "tier2_enabled": true },
  "providers": [
    { "name": "ollama", "kind": "ollama", "base_url": "http://127.0.0.1:11434",
      "enabled": false, "api_key_set": false }
  ],
  "search": { "provider": "duckduckgo", "base_url": "", "api_key_set": false },
  "mcp_servers": [
    { "name": "filesystem", "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allow"],
      "enabled": false }
  ],
  "gateways": {
    "telegram": { "enabled": false, "token_set": true },
    "whatsapp": { "enabled": false, "linked": false, "to_number": "" }
  },
  "server": { "host": "127.0.0.1", "port": 8686 }
}
```

`PATCH /v1/settings` (any subset, deep-merged) → **stages only**; nothing changes until
`POST /v1/system/apply`. Secrets (`gateways.telegram.token`) are **write-only**: accepted
in PATCH, never echoed; only `token_set` appears in GET.

**WhatsApp is QR-linked (changed 2026-07-22b), no tokens.** It links via the WhatsApp
Web multi-device protocol: enable the gateway, then scan a QR code with your phone
(Settings → Linked Devices), exactly like WhatsApp Web. No Cloud-API token or phone-number
ID. Endpoints:
- `GET /v1/gateways/whatsapp/qr` → `{ "state": "unlinked"|"awaiting_scan"|"linked",
  "qr": "<string to render as a QR>"|null, "qr_ascii": "<terminal QR>"|null }`. Poll
  while `awaiting_scan`; `qr` clears once `linked`.
- `POST /v1/gateways/whatsapp/unlink` → `{ "state": "unlinked" }` (forgets the session).
Outbound-only in this version (research-complete + scheduled-run pings to `to_number`);
inbound relay is `TODO(v1.2)`. The linked-device session lives under `~/.bonsai`, never in
a repo.

## 12. `/v1/slap`: agent protocol observability (additive 2026-07-22b)

suiban's Ultra mode coordinates its sub-agents with **SLAP** (Structured Lightweight Agent
Protocol; canonical spec + JSON schemas in the separate `slap` repo). These read-only
endpoints expose the protocol for inspection; the messages themselves flow internally and
surface to clients through the existing `agent_spawn`/`agent_result` stream events (which
now carry the SLAP `task_id`).
- `GET /v1/slap` → `{ "version": "1.0", "profiles": ["orchestrator","worker","utility"],
  "operations": ["assign","result","review","decide","error","cancel","heartbeat",
  "capability","status"] }`
- `GET /v1/slap/schema/{operation}` → the JSON Schema for that operation (404 for unknown).
- `GET /v1/slap/trace/{session_id}` → `{ "messages": [SLAP message] }` for a completed
  Ultra run (the validated agent-to-agent transcript; coarse, no worker internals).

## 9. `/v1/projects` (additive 2026-07-21b)

Project: `{ "id": "proj_...", "name", "description", "created_at", "session_count",
"doc_count" }`. A project groups sessions and carries a knowledge base of plain-text
docs, searched via FTS5 (never embeddings) and injected into member sessions on demand.

- `GET /v1/projects` → `{ "projects": [Project] }`
- `POST /v1/projects` `{ "name", "description"? }` → 201 Project
- `GET /v1/projects/{id}` → Project · `PATCH /v1/projects/{id}` `{ "name"?,
  "description"? }` → Project · `DELETE /v1/projects/{id}` → 204 (member sessions
  survive with `project_id` cleared)
- `GET /v1/projects/{id}/docs` → `{ "docs": [ { "id", "title", "bytes",
  "created_at" } ] }`
- `POST /v1/projects/{id}/docs` `{ "title", "content" }` → 201 doc (with content)
- `GET /v1/projects/{id}/docs/{doc_id}` → doc with content ·
  `DELETE /v1/projects/{id}/docs/{doc_id}` → 204

## 10. `/v1/schedules` (additive 2026-07-21b)

Schedule: `{ "id": "sched_...", "name", "prompt", "mode": "chat"|"code",
"effort": effort, "project_id": string|null,
"cadence": { "kind": "daily"|"weekly"|"interval", "time": "HH:MM"?,
"weekday": 0-6?, "every_minutes": int? }, "enabled": bool, "created_at",
"last_run_at", "next_run_at", "last_session_id", "last_error": string|null }`

- `GET /v1/schedules` → `{ "schedules": [Schedule] }` · `POST /v1/schedules` (name,
  prompt, cadence required) → 201 · `GET/PATCH /v1/schedules/{id}` ·
  `DELETE /v1/schedules/{id}` → 204 · `POST /v1/schedules/{id}/run` (run now) → 202
  `{ "session_id" }`
- Runs execute as ordinary chat sessions (archived, auto-titled, `project_id`
  honored); completion triggers gateway notifications. Times are server-local;
  `weekly` requires `weekday` + `time`, `interval` requires `every_minutes` ≥ 5.

## 11. External providers & web search (additive 2026-07-21c)

**Providers** (settings `providers[]`): external OpenAI-compatible inference layers.
`kind` ∈ `ollama` (presets `base_url http://127.0.0.1:11434`, keyless) · `openai`
(generic OpenAI-compatible endpoint; `api_key` write-only → `api_key_set`). Enabled
providers are polled for their model list (`{base_url}/v1/models`) and appear in
`GET /v1/models` as `<name>/<model>`; chat routing per §1. suiban never manages their
lifecycle or VRAM (honest boundary, documented in architecture.md).

**Web search** (settings `search`): the provider deep research uses for its gather
stage. `provider` ∈ `duckduckgo` (keyless default, best-effort HTML endpoint) ·
`searxng` (`base_url` required) · `brave` · `tavily` · `serper` (each `api_key`
write-only). `POST /v1/system/search_test` `{ "query"? }` → `{ "ok": bool,
"provider", "results": [ { "title", "url" } ] (≤3 on ok), "error": string|null }`, which
powers the settings "test" button; never throws, reports honestly.

**Memory behavior notes (same date):** the 27B orchestrator's post-task reflection may
write user memories after chat/code exchanges (write enforcement unchanged: workers
never). Recall is available in BOTH chat and code modes: `memory_search` +
session-archive search tools, plus light automatic injection of top FTS5 memory hits
relevant to the latest user message.

---

## Security notes

- Loopback bind by default; changing `server.host` is an explicit user action and the
  docs warn there is no auth in v1 (`TODO(v1.1): token auth for non-loopback binds`).
- The shell tool inside code mode is confirm-gated end-to-end; the HTTP surface never
  executes arbitrary commands directly.

### Destructive-op confirmation flow

Destructive tool invocations (e.g. shell `rm`) are denied server-side: the loop emits
`tool_result` with `status:"denied"` and a single-use `confirm_token` bound to the exact
command. The client shows the user a confirmation; on approval it sends a normal
follow-up user message (same `session_id`) telling the assistant to proceed with that
token. The model then re-invokes the tool with a `confirm_token` argument. Tokens are
single-use and command-bound; there is no request-level confirm field.

## Changelog

- **v1 (frozen)**, initial contract: chat/completions (+mode/effort/session_id/
  stream_events), models, jobs (deep research), system (+budget/health/apply), memory
  (+state/sessions incl. transcript), skills, modes, settings (staged).
- **v1, additive (2026-07-21)**: `tool_result` events gain optional `confirm_token`
  (present only with `status:"denied"`); documented the destructive-op confirmation
  flow above. No existing field changed.
- **v1, additive (2026-07-21b)**: projects (`/v1/projects` + chat `project_id` +
  sessions filter), schedules (`/v1/schedules`), state-file editing
  (`PUT /v1/memory/state/{name}`, supersedes the earlier "identity is read-only over
  HTTP" note), code-mode `workdir` chat field, session auto-titling behavior,
  WhatsApp outbound gateway settings. All additive; no existing field changed.
- **v1, additive (2026-07-21c)**: external inference providers (settings
  `providers[]`, provider-prefixed model ids, chat routing rules), pluggable web
  search for deep research (settings `search`, `POST /v1/system/search_test`),
  reflection/recall behavior notes (user memories from normal chats; recall in chat
  AND code modes). All additive; no existing field changed.
- **v1, additive (2026-07-21d)**: MCP servers: settings gain
  `mcp_servers: [ { "name": kebab-case, "command": string, "args": string[],
  "enabled": bool } ]` (stdio transport only in v1; requires_restart). Tools from
  connected servers appear to the model namespaced `mcp_<server>_<tool>` and surface
  in `tool_call`/`tool_result` stream events like built-ins. A failed server start is
  a `notice` (`mcp_server_failed`), never a crash. The `usage` stream event gains
  optional `malformed_calls`/`repaired_calls`/`abandoned_calls` counters, present
  only when an agentic run had malformed tool calls (the loop's per-run repair
  budget surfaced). Also documented: the stream `error`
  event's nested payload (bug-fix clarification), and `kv`/`dspark_enabled`
  reclassified as `requires_restart` in `/v1/system/apply` responses (they never
  applied at idle; the report now tells the truth). All additive.
- **v1, additive (2026-07-21e)**: Skill objects gain optional `verified: bool`
  (false until a run that used the injected skill completes successfully; every
  content write resets it); stream `notice` events may carry code `context_trimmed`
  (context-overflow guard trimmed injected blocks or old turns rather than sending an
  over-context request). All additive; no existing field changed.
- **v1, additive (2026-07-22b)**: code/ultra `auto_confirm` bypass field;
  `X-Bonsai-Client` header + client identity overlays (`identity-dai.md`/
  `identity-sentei.md`); confirmation gate extended to file mutations (`fs_write`/
  `fs_undo`); `GET /v1/memory/sessions?mode=` filter; `POST /v1/memory/sessions/import`;
  WhatsApp switched from Cloud-API tokens to QR device-linking (`GET
  /v1/gateways/whatsapp/qr`, `POST .../unlink`; settings `whatsapp.{enabled,linked,
  to_number}`); `/v1/slap` protocol-observability endpoints (Ultra now coordinates
  sub-agents via SLAP; `agent_spawn`/`agent_result` carry `task_id`). All additive; the
  WhatsApp settings shape changed pre-1.0 (no external consumer of the old token fields).
  Settings also gain `chat.auto_compress` (bool, default true; gates the automatic
  ~70%-context rolling compression; requires-idle to apply).
- **v1, additive (2026-07-22c)**: lazy/keep-alive model residency (ollama-style):
  `serve` holds no models until the first inference request; the loadout unloads after
  idle. Settings gain `runtime.keep_alive` (string `"24/7"`/`"0"` = stay hot, or a
  minutes integer; default 5). `GET /v1/system` gains `runtime: { keep_alive,
  models_loaded: bool, state: "cold"|"loading"|"ready"|"idle_unloading" }`. Any inference
  request auto-loads the loadout (a `notice` `warming_up` may precede the first token on a
  cold start); non-inference routes never load. Settings also gain `slap.enabled` (bool,
  default true; off routes Ultra through the plain-dict path) and `mcp_connectors` (a
  list of `{ id, enabled }` referencing the built-in catalog at `GET /v1/mcp/connectors`,
  distinct from custom `mcp_servers`). `GET /v1/mcp/connectors` returns
  `{ "connectors": [ { "id", "name", "description", "command", "args": string[],
  "requires_path": bool, "enabled": bool } ] }`; `enabled` reflects
  `settings.mcp_connectors` and is the authority clients render (do not derive it
  separately). New: `POST /v1/skills/import` `{ "source":
  "openclaw"|"hermes"|"path", "path"? }` → `{ "imported": [ { "name" } ], "skipped":
  [...] }` imports agentskills.io `SKILL.md` directories from the named ecosystem's skills
  folder (or a given path). All additive.
- **v1, additive (2026-07-22, security)**: settings gain `server.auth_token_set`
  (read-only bool in GET; the token is write-only and auto-generated at first
  non-loopback bind), `server.remote_agentic` (bool, default false), and
  `gateways.telegram.{allowed_chat_ids: int[], require_pairing: bool (default true),
  rate_limit_per_min: int}`. **Auth:** when `server.host` is not a loopback address,
  every request except `GET /v1/system/health` requires `Authorization: Bearer
  <token>` → 401 `unauthorized` otherwise; loopback binds stay open (unchanged
  default). `GET /v1/system` gains `security: { auth_required, remote_agentic,
  telegram_paired }`. All additive; loopback default behavior is unchanged.
- **v1, additive (2026-07-22d)**: two DELETE routes for manual cleanup from dai/sentei:
  `DELETE /v1/memory/sessions/{id}` (remove an archived chat and its transcript; 404
  `session_not_found`) and `DELETE /v1/memory/state/{name}` (remove a bounded state file;
  404 `state_file_unknown`, 400 `identity_read_only` for `identity.md` and the client
  overlays). New routes only; no existing shape changed. Per-entry `DELETE /v1/memory/{id}`
  was already in v1.
