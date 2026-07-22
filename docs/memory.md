# Memory & skills

suiban's persistence design: four layers, one writer, zero embeddings. Everything is
plain files and one SQLite database under `~/.bonsai/`, inspectable with a text editor
and `sqlite3`, wiped with `rm`. There are no vector databases and no embedding models
anywhere in the stack; retrieval is SQLite **FTS5** plus the resident utility model.

The HTTP surface for all of this is frozen in [api.md](api.md) (§5 `/v1/memory`,
§6 `/v1/skills`). This document is the internal specification behind that surface; if
the implementation ever needs to deviate, this file must be updated in the same change.

**Design lineage (inspiration, not borrowed code).** The layered, on-demand recall
model here, cheap always-available layers (identity, state) plus retrieval that is
*fetched by tool call when needed* rather than always prepended into context, is
inspired by the Hermes memory architecture. The inspiration is conceptual: the
four-layer split and the visible, tool-driven recall approach were informed by it,
while the storage (plain files + SQLite FTS5, no embeddings), the one-writer
enforcement (§7) and every line of the implementation are suiban's own. No code was
borrowed. This credit is also recorded in the repository `NOTICE` file.

## 1. The four layers

| Layer | Storage | Written by | Purpose |
|---|---|---|---|
| `identity` | `~/.bonsai/memory/identity.md` | **human only** (text editor · `PUT /v1/memory/state/identity.md`) | Who the user is, standing preferences. No model ever writes it. |
| `state` | `~/.bonsai/memory/state/*.md` (bounded files) | 27B reflection · human via API | Current facts that change: active projects, decisions, open threads. |
| `archive` | `~/.bonsai/memory/memory.sqlite` | 27B reflection · session recorder · human via API | Full session transcripts + distilled long-term entries. FTS5-indexed. |
| `skills` | `~/.bonsai/skills/<name>/SKILL.md` | **27B reflection · human via API** | Reusable how-to procedures. agentskills.io-compatible markdown. |

**State files are bounded.** Every state file has a byte budget (`max_bytes`, 8 KiB in
v1, reported verbatim by `GET /v1/memory/state`, identity.md included, same budget).
A MODEL write that would exceed the budget is **auto-compacted oldest-first**: leading
paragraphs are dropped until the content fits (state files grow by appending, so the
top is the oldest; the newest bytes always survive). The write tool's description
tells the 27B to write sparingly, but the byte cap is enforced mechanically, not by
model cooperation. State stays small enough to be cheap to recall and impossible to
hoard. State and identity files on disk are the source of truth; they are mirrored
into the database on startup and on every write through suiban (a by-hand edit to the
files while the server runs is picked up at the next startup) so that one FTS5 index
covers every layer.

**Editing state files over HTTP (additive 2026-07-21b).** `PUT
/v1/memory/state/{name}` overwrites ONE existing bounded file, `identity.md`
included, which is therefore no longer read-only over HTTP (it remains off limits to
every model). The route is deliberately narrow:

- `{name}` is a bare filename matched against the known set (`identity.md` +
  existing `state/*.md`); anything else (unknown names, traversal strings, new
  filenames) is a 404 (`state_file_unknown`). New files are not creatable here
  (`POST /v1/memory` with `layer: "state"` creates state files).
- A HUMAN edit above `max_bytes` is rejected with 400 (`state_file_too_large`):
  rejected loudly, never silently compacted; auto-compaction applies only to
  model-driven appends.
- The edit is re-mirrored into the FTS index immediately, so recall (identity
  included) sees it without a restart.

**Client identities (additive 2026-07-22b).** The base `identity.md` is joined by two
**client overlays** (`identity-dai.md` and `identity-sentei.md`) seeded from packaged
copies into `~/.bonsai/memory/` on first run (existing files are never overwritten).
Each request carries an `X-Bonsai-Client` header (`dai` · `sentei` · `other`, default
`other`); suiban injects the base identity **plus** the matching overlay into the system
prompt (coalesced into the single leading system message):

- `sentei` → the coding-focused overlay (sharp terminal pair-programmer);
- `dai` → the general overlay (calm desktop generalist);
- `other`/unknown → base `identity.md` only.

The overlays are ordinary editable state files: they appear in `GET /v1/memory/state`
and are edited via `PUT /v1/memory/state/identity-dai.md` / `identity-sentei.md` under
the same byte cap. Unlike the base `identity.md`, the overlays are **not** mirrored into
FTS recall. They are persona injected per client, not searchable facts, so a `dai`
overlay never bleeds into a `sentei` session's recall.

**Importing chats (additive 2026-07-22b).** `POST /v1/memory/sessions/import` parses
another tool's exported conversation(s) into archived sessions (they then list under
`GET /v1/memory/sessions` and restore like any session). Providers: `openai` (the
ChatGPT `conversations.json` mapping export), `claude` (the claude.ai `chat_messages`
export), `claude-code` (a `~/.claude` project JSONL transcript, one object per line with
role/content) and `generic` (`{title?, messages:[{role, content}]}`). Parsing is pure
and offline (`memory/importers.py`); a payload that does not match the provider shape is
a 400 `import_unrecognized`. With `compress:true` the resident utility model condenses
each long import into a single seed summary message (the `Rolling conversation summary`
shape from §5), so a resumed session starts inside context instead of replaying the
whole transcript. `mode` (default `chat`) tags the created sessions for the
`?mode=chat|code` sessions filter. dai's Chat and Code tabs then show separate recents.

## 2. SQLite schema (v1)

One database: `~/.bonsai/memory/memory.sqlite`, kept beside the files it mirrors.
WAL mode, `synchronous=NORMAL`.

```sql
-- Long-term entries. archive rows are canonical here; identity/state rows are
-- read-through mirrors of the files (rebuilt on startup / write). archive ids are
-- "mem_" + ULID (time-sortable); mirror rows use deterministic ids
-- ("mem_file_" + sha256(layer/name)[:12]) so re-mirroring is idempotent.
CREATE TABLE memory_entries (
  id             TEXT PRIMARY KEY,
  layer          TEXT NOT NULL CHECK (layer IN ('identity','state','archive')),
  title          TEXT NOT NULL,
  content        TEXT NOT NULL,
  tags           TEXT NOT NULL DEFAULT '[]',  -- JSON array of strings
  created_at     TEXT NOT NULL,               -- ISO-8601 UTC
  updated_at     TEXT NOT NULL,
  source_session TEXT REFERENCES sessions(id) -- NULL for human/mirrored entries
);

CREATE VIRTUAL TABLE memory_fts USING fts5(
  title, content, tags,
  content='memory_entries', content_rowid='rowid',
  tokenize='unicode61 remove_diacritics 2', prefix='2 3'
);
-- external-content triggers keep memory_fts in sync (AFTER INSERT/UPDATE/DELETE;
-- messages_fts below needs only INSERT/DELETE — messages are never updated)

-- Session archive: every conversation, verbatim.
CREATE TABLE sessions (
  id            TEXT PRIMARY KEY,             -- client-supplied session_id (UUID)
  title         TEXT,                         -- nullable; see auto-titling note below
  mode          TEXT NOT NULL,                -- chat | code | ultra
  project_id    TEXT,                         -- api.md §9 project binding (nullable)
  workdir       TEXT,                         -- remembered code-mode jail root
                                              -- (internal; never in the API shape)
  started_at    TEXT NOT NULL,
  ended_at      TEXT,
  message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE messages (
  id         INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role       TEXT NOT NULL,                   -- user | assistant | tool
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE messages_fts USING fts5(
  content,
  content='messages', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2', prefix='2 3'
);
```

**Tokenizer (2026-07-21 refinement).** All FTS tables (memory, messages, project
docs) use `unicode61 remove_diacritics 2` (diacritics fold BOTH ways, so `cafe`
finds *café* and `café` finds *cafe*) with `prefix='2 3'` indexes. Query tokens of
≥ 3 chars are issued as prefix terms (`"kuber"*` finds *kubernetes*, `"deploy"*`
finds *deployed*), which replaces what porter stemming used to give without its
false conflations; 1–2-char tokens stay exact. Databases created with the old
`porter unicode61` tables are migrated on open: the shadow tables are dropped,
recreated with the new spec and repopulated via FTS5 `rebuild`; the rebuild is
idempotent and the content tables are untouched. Result snippets use a ~13-token (≈ 64-char) window;
project-doc excerpts stay wider (32 tokens) because they feed injection, not a
result list. A 10-case query→expected-entry eval set (diacritics, prefixes,
multi-term) lives in `tests/test_memory.py` and pins this behavior.

Search (`GET /v1/memory/search`, `GET /v1/memory/sessions?q=`) ranks with FTS5's
built-in `bm25()`; the `score` and `snippet` in the API response come straight from
`bm25()` and `snippet()`. Scores are raw bm25 values (more negative = better match),
passed through without rescaling. Free-text queries are tokenized and OR-joined into
the MATCH expression (OR for recall; bm25 sorts the good hits up). No embeddings, by
decision. FTS5 is transparent, fast on CPU and its failure mode (a miss) is visible
rather than weird.

`sessions.title` starts `NULL` and is auto-generated once (additive 2026-07-21b):
after the first completed exchange of a still-untitled session, the resident utility
model (thinking off) produces a ≤ 6-word title in a fire-and-forget background task.
Any failure just leaves the title `NULL`. Clients must still render untitled
sessions by id/first message.

## 3. Injection policy: tools first, one narrow automatic path

The original v1 rule was "nothing is silently injected". The 2026-07-21c additive
contract softens it by exactly one narrow path, and the reasons for restraint still
govern its shape:

1. Local contexts are expensive: see the KV math in [hardware.md](hardware.md); a
   permanently-injected memory blob is per-token VRAM burned on every request.
2. Visible recall is debuggable recall. When the model "remembers" something, the user
   can see exactly what was fetched and from where.

What holds: no memory layer is wholesale-prepended to conversations, and recall is
primarily tool-driven (`memory_search` + `session_search`, available in BOTH chat and
code modes), visible in the event stream as ordinary `tool_call`/`tool_result`
pairs.

**Light automatic injection (additive 2026-07-21c):** for chat and code exchanges,
the top FTS5 MEMORY hits (bm25 over `memory_fts`, entries only, never raw
transcripts) for the *latest user message* are prepended as ONE clearly-delimited
system message (`<<<memory mem_… (layer) title>>> … <<<end memory>>>`), snippets
only, budget-capped at ~5% of the slot context (chars/4 estimate, the same
budgeting as the project-doc injection), skipped entirely when nothing matches.
Ultra and deep-research runs get no automatic injection. External provider sessions
get the same injection (with a fixed stand-in budget; external context sizes are
unknown).

**Skill injection (2026-07-21 refinement):** local agentic chat/code runs also get
up to 2 skills whose *name/description* tokens match the latest user message
(stopword-filtered overlap, deliberately dumb and inspectable, like everything
else here), injected as ONE `<<<skill name vN>>> … <<<end skill>>>` system block,
budget-capped at ~8% of the slot context. **Verified skills are ordered first and
unverified ones are labeled `[unverified]`** in the injected text. That label plus
ordering is how "prefer verified skills" reaches the model (see §6 for the
verification lifecycle). Pass-through requests, ultra, deep research and external
sessions get no skill injection. All injection delimiters are owned by
`memory/injection.py` so the overflow guard (§5) can recognize exactly what it may
trim.

## 4. Recall flow

```
orchestrator issues memory_search(query, limit?)
    -> FTS5 bm25 over memory_fts + messages_fts   (top-k per index; k=12 default,
       capped at 20)
    -> few hits (≤6 combined): the raw hit lines are returned as-is —
       "[mem_…] (layer) title: snippet" / "[session …] role: snippet"
    -> more hits: the resident utility model condenses them to a short
       query-focused digest, dropping irrelevant hits, keeping the [ids]
    -> orchestrator may follow up with a direct fetch of a full entry /
       session transcript by id
```

`session_search(query, limit?)` (additive 2026-07-21c) is the dedicated
session-archive dig: FTS5 bm25 over `messages_fts` alone, returning
`[session <id>] role: snippet` lines, for "what did we discuss about X?" questions
where the model then follows up on a session id. Registered in both chat and code
toolsets, every role (reads were never restricted; writes still are, §7).

The utility model (4B; 1.7B on the 8 GB tier; the orchestrator itself on CPU-only) is
resident in every loadout precisely so recall and compression never require a model
load. When no utility summarizer is available, the raw hit list is returned regardless
of size, degraded, never broken. Ranking quality is bm25's, good enough with decent
queries, and the digest step filters false positives. `TODO(v1.1): query expansion by
the utility model if recall precision proves weak in real use.`

## 5. Compression (~70% of context)

Compression runs at **request preparation** (before the agent loop starts), whenever
the incoming history is at **≥ 70% of the slot's context**:

1. Token counts are ESTIMATED: chars/4 plus a small per-message overhead; suiban
   ships no tokenizer, and the estimate only has to trigger comfortably early.
   `TODO(v1.1): exact counts via llama-server's /tokenize once the live-wiring pass
   connects real slots.`
2. The compressible span is everything between the leading system prompt(s) and the
   **adaptive protected tail**: the last 4 messages below 16K ctx, 6 at 16K, 8 at
   32K and above (2026-07-21 refinement: small contexts need the room, big contexts
   can afford more verbatim recency). A previous rolling summary counts as part of
   the span, so subsequent compressions fold into ONE summary instead of stacking.
   Spans of fewer than 2 messages are left alone.
3. The resident utility model produces the rolling summary, which replaces the span as
   a system message (`"Rolling conversation summary (older turns condensed):"`).
4. The replaced messages are already in the archive verbatim (the session recorder
   writes as the conversation happens), so compression loses nothing on disk.
5. The client sees a `compression` SSE event: `{ "trigger_pct": 70.4,
   "messages_summarized": 18 }`. Compression is felt, never hidden.
6. Compression is an optimization, never a gate: if the summarizer call fails, the
   request proceeds uncompressed.

Threshold rationale: at 70% there is still room for the summary, the reply and the
thinking budget (which is itself capped at 40% of ctx; see the effort ladder in
[architecture.md](architecture.md)).

**Overflow guard (2026-07-21 refinement).** Compression is an optimization; the
guard is the backstop. After compression (or when compression could not run), if the
estimated request still exceeds **90% of the slot context**, `memory/injection.py`
trims in a fixed ladder: injected blocks first (memory recall, then skills, then
project docs, dropping each injection's LAST (lowest-bm25) blocks first), then the
oldest non-system messages beyond the adaptive protected tail. The run emits a
`notice` event (`code: "context_trimmed"`, level `warn`) describing what was
dropped; non-streaming clients get the same text as a server log line. If even the
protected minimum exceeds the limit (one giant message), the notice says so. The
request proceeds, but llama-server is **never handed an over-context request
silently**. The system head (mode prompt, rolling summary) is never trimmed.

**Compression fidelity (2026-07-21 refinement).** The summarizer prompt
(`SUMMARIZE_SYSTEM_PROMPT`, memory/compression.py) carries an explicit keep-list
(names, numbers, dates, paths, URLs, error messages, decisions, `[ids]`) because
vague "condense this" prompts drop exactly what recall needs most. Two test layers
keep this honest (`memory/fidelity.py` plants distinctive facts early in a
synthetic conversation, filler after and scores survival by plain string
containment; no judge model):

- *Mechanics, modelless* (`tests/test_fidelity.py`): with a summarizer that echoes
  the facts it sees, facts survive compression, folding and re-compression at
  100%. The pipeline never loses what the model kept, and protected-tail facts
  never pass through the summarizer at all.
- *Real model, opt-in* (`tests/test_fidelity_live.py`, `SUIBAN_LIVE_FIDELITY=1`
  against the live stack): the resident utility model must keep ≥ 80% of planted
  facts. What the mechanics tests cannot honestly claim, that the MODEL keeps
  facts, this harness measures.

## 6. Skills

Skills are directories under `~/.bonsai/skills/<name>/` with a `SKILL.md` whose
frontmatter is **agentskills.io-compatible**: `name` and `description` are required;
suiban requires nothing else in the file:

```markdown
---
name: changelog-entry
description: Add a keepachangelog-style entry to CHANGELOG.md for the current change,
  matching the project's existing tense and category conventions.
---

# Writing a changelog entry

1. Read the existing CHANGELOG.md; note tense, category names, and linking style.
2. Classify the change: Added / Changed / Fixed / Removed.
3. Draft one entry line; mention user-visible behavior, not internals.
4. Insert under [Unreleased]; never rewrite released sections.
```

Version (`version`, integer), provenance (`source`: `seed` | `learned` | `human`) and
`updated_at` are tracked by suiban *outside* the file (in `<name>/meta.json` beside
the SKILL.md) so the markdown stays portable to any agentskills.io consumer. A skill
directory dropped in by hand (no meta.json) is treated as `source: "human"`,
version 1. `GET /v1/skills` lists them; `PUT` marks `source: "human"` and bumps
`version`. `source: "seed"` is reserved: `TODO(v1.1): ship a starter skill pack at
install time; nothing writes "seed" yet.`

**Who reads, who writes.** Every model in the loadout reads skills, workers included.
Creation and improvement of skills happens in exactly one place: the 27B orchestrator's
post-task reflection, where it may distill a new skill or refine an existing one
(`source: "learned"`, version bump). Nothing model-driven is exposed over HTTP for this;
see api.md §6.

**Import / portability (2026-07-22c).** Because a suiban skill IS an agentskills.io
`SKILL.md` directory, skills are portable BOTH ways: any other agentskills.io tool's
`<name>/SKILL.md` directory imports here unchanged, and a suiban skill directory drops
straight into those tools. `POST /v1/skills/import` and `suiban skills import
<openclaw|hermes|PATH>` (`memory/skill_import.py`) scan a source's skills folder, validate
each skill with the SAME frontmatter validator as the model-write path
(`validate_skill_markdown`) and copy the valid ones (SKILL.md plus any `scripts/` and
supporting files) into `~/.bonsai/skills/<name>/`. An imported skill is marked
`source: "imported"`, `verified: false` (unproven until a run uses it, exactly like a
learned skill). A malformed skill is *skipped* and returned in a `skipped: [{name,
reason}]` list. The import never crashes on one bad file; a source that cannot be scanned
at all (a `path` that does not exist) is a clean `400 import_source_unavailable`. Known
sources: `openclaw` (`~/.openclaw/workspace/skills`, plus a checked-out repo's
`.agents/skills` when a path is given), `hermes` (`~/.hermes/skills`, plus a repo's
`optional-skills/<cat>/<name>` tree) and a bare `path` (any directory, scanned
recursively for `SKILL.md`). Both openclaw and hermes-agent are MIT-licensed
agentskills.io skill sets; only their skills are read, no code (see NOTICE). Re-importing
a name replaces the directory and resets provenance to imported/unverified.

**Schema validation on model writes (2026-07-21 refinement).** `skill_save` /
`skill_improve` validate the frontmatter before anything touches disk: a closed
`---` block of `key: value` pairs (the minimal YAML subset), `name` present +
kebab-case + equal to the skill name, `description` present and non-empty. An
invalid file is a structured rejection (`400 skill_invalid`, message prefixed
`invalid skill` listing every validator error). The reflection path retries exactly
ONCE (the rejection is appended as the failed call's tool result and the model may
resend a corrected SKILL.md), then gives up quietly. Human writes (HTTP `PUT`,
hand-dropped directories) stay lenient: bare content still gets synthesized
frontmatter, and whatever a human left on disk is tolerated. Only the model path is
strict. A model that cannot produce two frontmatter keys should not be teaching
skills.

**Verification lifecycle (2026-07-21 refinement).** meta.json also tracks
`verified` (surfaced on the api.md Skill object as an additive optional field):

- Every content write (save, improve, human PUT, hand-dropped dir) starts or
  resets it to `false`: new instructions are unproven.
- A run that *used* the skill (it was injected into the run's context, §3) and
  completed successfully flips it to `true`. Error finishes verify nothing.
- Injection prefers verified skills (ordered first) and labels the rest
  `[unverified]`, so an unproven skill is visible as such to the model and, via
  `/v1/skills`, to humans.

## 7. The write-enforcement rule

**Only the 27B orchestrator writes memories and skills.** This is enforced in the
server, not the prompt:

- Worker and utility slots' tool schemas simply do not contain memory- or skill-write
  tools; the grammar-constrained decoder cannot emit a call that is not in the schema.
- Defense in depth: the internal write path checks the originating slot's `role` and
  rejects anything that is not `orchestrator`. A misrouted call fails loudly.
- Reflection (the only model-driven write moment) runs on the orchestrator after the
  task completes, never inside worker loops.
- HTTP writes (`POST /v1/memory`, `PUT /v1/memory/state/{name}`,
  `PUT /v1/skills/{name}`) are human/client actions and are labeled as such
  (`source: "human"`). `identity` is human-editable over HTTP via
  `PUT /v1/memory/state/identity.md` (additive 2026-07-21b, this supersedes the
  earlier "identity is read-only over HTTP" rule) but stays untouchable by every
  model: the entry surface (`PUT/DELETE /v1/memory/{id}`) still refuses identity
  rows, and no tool schema contains an identity write.

**Post-task reflection cadence (additive 2026-07-21c).** After a completed chat or
code exchange driven by the ORCHESTRATOR slot, suiban runs a background reflection
completion asking whether the exchange revealed a durable user fact or preference;
the model either calls `memory_write` (its only available tool in that call) or
answers "none". The mechanics, chosen for 8 GB-tier sanity:

- Rate limit: at most once per session per **3** completed exchanges (the 1st, 4th,
  7th, …; in-memory counter, reset on restart). Anonymous exchanges (no
  `session_id`) never reflect. There is no session to key the limit on.
- Cheap by construction: thinking off, `max_tokens` 256, one completion, no
  follow-up round, nothing extra archived.
- Failure-tolerant: any error is logged and dropped. A chat can never break
  because reflection could not run.
- Exclusions are structural: worker/utility slots never reach the reflection path
  (and the write path re-checks role anyway), ultra runs are excluded by mode, and
  external provider sessions never reflect. External models are not the
  orchestrator, so they can never write memory.

Rationale: the smallest models are the most suggestible; letting an 1.7B worker write
long-term memory is how a stack poisons its own well. One writer, and the largest one,
keeps persistent state trustworthy.

## 8. Privacy & wiping

Everything lives under `~/.bonsai/`, nothing inside any repo and nothing leaves the
machine (no telemetry; network only for installer downloads, the browse tools when
invoked and gateways when enabled):

```
~/.bonsai/
├── config.toml          # settings incl. any gateway token (chmod 600 recommended)
├── staged.toml          # staged-but-not-applied settings (see api.md §8)
├── budget.json          # measured VRAM footprints (see hardware.md)
├── bin/<backend>/       # fork binaries, per backend (cuda, cpu, ...)
├── models/<family>/     # GGUF weights + mmproj, per quant family
├── logs/
├── memory/
│   ├── identity.md         # yours; edit by hand or via PUT /v1/memory/state/identity.md
│   ├── identity-dai.md     # dai client identity overlay (editable, api.md 2026-07-22b)
│   ├── identity-sentei.md  # sentei client identity overlay (editable)
│   ├── state/              # bounded state files
│   └── memory.sqlite       # archive + transcripts + FTS index
├── skills/<name>/          # SKILL.md + meta.json
├── whatsapp/               # linked WhatsApp device session (QR-linked gateway)
├── work/<session_id>/      # default per-session tool jail (a code-mode session may
│                           # instead be jailed to a user-chosen workdir; api.md §1)
├── reports/                # deep-research reports (<job_id>.md), bench reports
└── browser/profile/        # sandboxed Playwright profile (tier-2 browsing only)
```

To wipe:

- **All memory of you:** delete `~/.bonsai/memory/` (identity, state, transcripts,
  entries: everything; recreated empty on next start).
- **Transcripts/entries only:** delete `~/.bonsai/memory/memory.sqlite` (keeps
  identity.md and state files; mirrors and index rebuild on next start).
- **Learned skills:** delete directories under `~/.bonsai/skills/`.
- **Everything including models:** delete `~/.bonsai/`.

Deleting an entry over the API (`DELETE /v1/memory/{id}`) removes the row and its FTS
index entries. Deleting individual *sessions* is not on the v1 HTTP surface (api.md is
frozen). Wipe the database file instead. And note honestly: SQLite in WAL mode does
not scrub freed pages immediately, so for hard deletion of sensitive content, wipe the
file rather than relying on row deletes.
