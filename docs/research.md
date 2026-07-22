# Deep research

How suiban runs deep research: an **async job**, never a chat mode. The HTTP surface
is `/v1/jobs`, frozen in [api.md](api.md) §3; this document explains the job model,
the pipeline behind it and the product rules that shape both.

## The coarse-progress product rule

Users see exactly three things about a running job: its **state**, a coarse **stage**
string and a **percent**. The queries the model planned, the URLs it fetched, the
pages it read, its cross-check notes and report drafts are internal working state and
**never appear in any API response, SSE event or gateway ping**. This is a product
rule, not a missing feature: half-formed research narrated live reads as authority
before it has earned any. The finished report, with its citations and its
limitations section, is the deliverable; the process stays inside the machine.

Concretely: `stage` is only ever one of `planning queries`, `collecting sources`,
`cross-checking`, `writing report` (and `null` once terminal), and the SSE stream
emits only `progress` and `state` events.

## Job model

Jobs are rows in a `jobs` table inside the same SQLite database as memory
(`~/.bonsai/memory/memory.sqlite`, one local database, no extra infrastructure).

States and transitions:

```
queued -> running -> completed
                  \-> failed
queued/running ----> cancelled     (DELETE /v1/jobs/{id})
```

- `POST /v1/jobs {type:"deep_research", query, effort?}` → **202** `{id, state:"queued"}`.
  The job starts immediately (queued is momentary in v1; see concurrency below).
- `GET /v1/jobs` lists newest first; `GET /v1/jobs/{id}` returns the JobStatus shape.
- `GET /v1/jobs/{id}/events` is SSE: a snapshot on subscribe, then `progress` on
  change and `state` on transitions; the stream closes after a terminal state.
- `GET /v1/jobs/{id}/report` returns `text/markdown`: **404 until `completed`**.
  Reports persist at `~/.bonsai/reports/<job_id>.md`.
- `DELETE /v1/jobs/{id}` cancels, idempotently: cancelling an active or
  already-cancelled job returns `state:"cancelled"`; cancelling a job that already
  finished is a no-op that returns its truthful terminal state (we do not relabel a
  completed job "cancelled"). The cancel is real, not cosmetic: the pipeline task is
  cancelled AND its unwind is awaited (bounded, ~10 s) before the DELETE returns, so
  the in-flight llama-server request was actually aborted. In the rare case the
  unwind exceeds the bound, the job row is already terminal but a new submit answers
  429 until the old task truly finished. The one-job invariant covers teardown too.
- Jobs found `queued`/`running` at startup were orphaned by a previous process and
  are marked `failed` with an explicit error.

Completion (and failure/cancellation) also triggers gateway notifications when a
gateway is configured; see [gateways.md](gateways.md).

## Concurrency: one job at a time (v1)

`POST /v1/jobs` while any job is queued or running returns **429**
(`overloaded_error`, code `research_job_active`). This is honest, not lazy: the
pipeline runs on the orchestrator slot of a single local GPU loadout, and a second
concurrent research run would silently starve the first (and every chat request)
rather than fail cleanly. TODO(v1.1): a real queue (accept N, run 1) once the
scheduler can reserve slot time for background jobs.

## The pipeline

Each job runs this staged pipeline on the orchestrator slot (progress spans in
parentheses):

1. **planning queries** (0–10%): a grammar-constrained completion (json_schema →
   GBNF) produces the sub-questions and concrete source URLs. If the plan cannot be
   parsed after repair retries, the pipeline degrades to researching the bare query
   with no pre-planned sources, and the report says so.
2. **collecting sources** (10–55%), additive 2026-07-21c: the sub-questions become
   web-search queries on the configured search provider (settings `search`, see
   below); the top result URLs (deduped, capped at 6) are fetched with the tier-1
   fetcher (plain HTTP + readability extraction); when tier-2 browsing is available
   (resident 27B + setting enabled) it is the fallback for pages tier 1 cannot
   read. The searched URLs and their result titles feed the report's citations.
   Failed fetches become "unavailable" notes, not crashes. If search fails ENTIRELY
   (every query errored, or nothing came back), the stage falls back to the plan's
   model-proposed URLs (the pre-search behavior) and a note at the top of the
   report says so honestly.
3. **cross-checking** (55–80%): claims are sorted into corroborated / disputed /
   single-source / unverifiable against the gathered material. Internal only.
4. **writing report** (80–100%), the final markdown: answer first, body by
   sub-question with inline citations and a limitations section. An empty
   synthesis fails the job rather than completing with a blank report.

## Search providers (settings `search`, additive 2026-07-21c)

One provider is configured at a time (`search.provider`; `api_key` is write-only →
`api_key_set` in GET, exactly like the gateway tokens):

| Provider | Needs | Transport |
|---|---|---|
| `duckduckgo` | nothing (default) | best-effort scrape of `html.duckduckgo.com/html`, honestly fragile: non-API markup, may rate-limit/captcha; a broken parse yields zero results, never a crash |
| `searxng` | `search.base_url` (your instance, JSON format enabled) | `{base_url}/search?q=&format=json` |
| `brave` | `search.api_key` | `api.search.brave.com/res/v1/web/search`, `X-Subscription-Token` |
| `tavily` | `search.api_key` | `POST api.tavily.com/search`, key in the body |
| `serper` | `search.api_key` | `POST google.serper.dev/search`, `X-API-KEY` |

`POST /v1/system/search_test {query?}` runs one search on the configured provider
and answers `{ok, provider, results (≤3, title+url only), error}`. It never throws,
and an empty result set reports `ok:false` (a test button that says "ok" on zero
results would be lying). This powers the settings "test" button in dai/sentei.

Honest limits: with the keyless duckduckgo default, coverage depends on a scraper
with no API contract; the keyed providers (or a self-hosted searxng) are the
reliable path. Search queries and result URLs remain internal working state. The
coarse-progress rule above is unchanged.

## Interaction with the run lifecycle

An active job counts as activity: `/v1/system/apply` never commits staged settings
while a research job is queued or running (`applied:false`, and the commit fires on
the transition to idle). `jobs_active` in `GET /v1/system` reflects the live count.

**Fairness with interactive chats (deliberate design).** A job runs on the
orchestrator slot, but it does not own it: the pipeline acquires the slot's gate per
*step* (one completion) and releases it between steps (`research/wiring.py`).
Holding the slot for the job's 15-40 minutes would starve every interactive chat into
300 s timeouts; with per-step locking a chat waits at most one research completion,
and the job waits its turn behind queued chats between steps. The trade: research
wall-clock time stretches under interactive load, which is the right direction to
degrade: background work absorbs latency, foreground work does not. The job also
skips the chat queue's 429 capacity check: background work waits, it does not fail.
