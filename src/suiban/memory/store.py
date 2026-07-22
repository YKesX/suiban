"""SQLite FTS5 archive: memory entries, sessions, messages (docs/memory.md §2).

No embeddings anywhere — retrieval is FTS5 `bm25()` (raw scores: more negative =
better match, passed straight through per the doc) plus `snippet()`. sqlite3 calls are
synchronous; at local single-user scale that is fine, and a lock keeps it safe from
concurrent request handlers.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection, OperationalError, connect

from suiban.memory.ids import memory_id, new_doc_id, new_project_id

logger = logging.getLogger(__name__)

# Milliseconds a write waits for the DB lock before giving up (audit 2026-07-22). The
# jobs/schedules/memory stores each open their OWN connection to the one memory.sqlite,
# so under concurrent writes a second connection can see "database is locked"; WAL +
# this timeout makes writers queue instead of erroring. Set explicitly on every
# connection (not left to sqlite3.connect's default) so it is visible and testable.
BUSY_TIMEOUT_MS = 5000

# Tokenizer + prefix spec shared by every FTS table (2026-07-21 refinement):
# unicode61 with two-way diacritic folding (café == cafe in query AND content) and
# prefix indexes so fts_query's trailing-star prefix terms stay index-backed. The
# porter stemmer is gone — prefix matching covers the common inflection case
# ("deploy"* matches deployed) without stemming's false conflations. The literal
# string below doubles as the migration marker (_migrate checks sqlite_master for it).
FTS_TOKENIZE = "unicode61 remove_diacritics 2"
FTS_PREFIX = "2 3"

# snippet()/'excerpt' windows are in TOKENS: ~13 tokens ≈ a 64-char window at the
# ~5 chars/token of English prose — enough to show a match in context without
# flooding recall results. Project-doc excerpts stay wider (32) on purpose: they are
# an injection source, not a result list.
SNIPPET_TOKENS = 13
DOC_EXCERPT_TOKENS = 32

_FTS_TABLES: dict[str, str] = {
    "memory_fts": f"""CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  title, content, tags,
  content='memory_entries', content_rowid='rowid',
  tokenize='{FTS_TOKENIZE}', prefix='{FTS_PREFIX}'
)""",
    "messages_fts": f"""CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  content,
  content='messages', content_rowid='id',
  tokenize='{FTS_TOKENIZE}', prefix='{FTS_PREFIX}'
)""",
    "project_docs_fts": f"""CREATE VIRTUAL TABLE IF NOT EXISTS project_docs_fts USING fts5(
  title, content,
  content='project_docs', content_rowid='rowid',
  tokenize='{FTS_TOKENIZE}', prefix='{FTS_PREFIX}'
)""",
}

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS memory_entries (
  id             TEXT PRIMARY KEY,
  layer          TEXT NOT NULL CHECK (layer IN ('identity','state','archive')),
  title          TEXT NOT NULL,
  content        TEXT NOT NULL,
  tags           TEXT NOT NULL DEFAULT '[]',
  created_at     TEXT NOT NULL,
  updated_at     TEXT NOT NULL,
  source_session TEXT REFERENCES sessions(id)
);

{_FTS_TABLES["memory_fts"]};

CREATE TRIGGER IF NOT EXISTS memory_entries_ai AFTER INSERT ON memory_entries BEGIN
  INSERT INTO memory_fts(rowid, title, content, tags)
  VALUES (new.rowid, new.title, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memory_entries_ad AFTER DELETE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, title, content, tags)
  VALUES ('delete', old.rowid, old.title, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memory_entries_au AFTER UPDATE ON memory_entries BEGIN
  INSERT INTO memory_fts(memory_fts, rowid, title, content, tags)
  VALUES ('delete', old.rowid, old.title, old.content, old.tags);
  INSERT INTO memory_fts(rowid, title, content, tags)
  VALUES (new.rowid, new.title, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS sessions (
  id            TEXT PRIMARY KEY,
  title         TEXT,
  mode          TEXT NOT NULL,
  project_id    TEXT,
  workdir       TEXT,
  started_at    TEXT NOT NULL,
  ended_at      TEXT,
  message_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role       TEXT NOT NULL,
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL
);

{_FTS_TABLES["messages_fts"]};

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
  INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TABLE IF NOT EXISTS projects (
  id          TEXT PRIMARY KEY,
  name        TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS project_docs (
  id         TEXT PRIMARY KEY,
  project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  title      TEXT NOT NULL,
  content    TEXT NOT NULL,
  created_at TEXT NOT NULL
);

{_FTS_TABLES["project_docs_fts"]};

CREATE TRIGGER IF NOT EXISTS project_docs_ai AFTER INSERT ON project_docs BEGIN
  INSERT INTO project_docs_fts(rowid, title, content)
  VALUES (new.rowid, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS project_docs_ad AFTER DELETE ON project_docs BEGIN
  INSERT INTO project_docs_fts(project_docs_fts, rowid, title, content)
  VALUES ('delete', old.rowid, old.title, old.content);
END;
"""


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# Tokens at least this long become prefix terms ("kuber"* matches kubernetes);
# 1–2-char tokens stay exact — prefixing those would match half the index.
PREFIX_MIN_CHARS = 3


def fts_query(user_query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression: quoted tokens OR-joined
    (OR for recall; bm25 ranking sorts the good hits up). Tokens of >=
    PREFIX_MIN_CHARS chars are prefix terms (trailing *), backed by the prefix='2 3'
    indexes — so partial words and un-stemmed inflections still hit.

    The token class also drops C0 control characters and DEL (\\x00-\\x1f, \\x7f): a
    NUL byte reaching the quoted MATCH expression makes SQLite's C string API read it
    as an unterminated string and raise OperationalError (audit 2026-07-22 — a query
    like `?q=a%00b` would otherwise 500 every FTS search). unicode61 treats these as
    separators anyway, so nothing searchable is lost."""
    tokens = re.findall(r"[^\s\x00-\x1f\x7f\"'()*:^]+", user_query)
    return " OR ".join(f'"{t}"*' if len(t) >= PREFIX_MIN_CHARS else f'"{t}"' for t in tokens)


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    layer: str
    title: str
    content: str
    tags: list[str]
    created_at: str
    updated_at: str
    source_session: str | None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "layer": self.layer,
            "title": self.title,
            "content": self.content,
            "tags": self.tags,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_session": self.source_session,
        }


def _row_to_entry(row) -> MemoryEntry:
    return MemoryEntry(
        id=row[0],
        layer=row[1],
        title=row[2],
        content=row[3],
        tags=json.loads(row[4]),
        created_at=row[5],
        updated_at=row[6],
        source_session=row[7],
    )


_ENTRY_COLS = "id, layer, title, content, tags, created_at, updated_at, source_session"

_SESSION_COLS = "s.id, s.title, s.mode, s.project_id, s.started_at, s.ended_at, s.message_count"


def _row_to_session(row) -> dict:
    return {
        "id": row[0],
        "title": row[1],
        "mode": row[2],
        "project_id": row[3],
        "started_at": row[4],
        "ended_at": row[5],
        "message_count": row[6],
    }


# The api.md §9 Project shape: counts are computed live (cheap at local scale).
_PROJECT_COLS = (
    "p.id, p.name, p.description, p.created_at, "
    "(SELECT COUNT(*) FROM sessions s WHERE s.project_id = p.id), "
    "(SELECT COUNT(*) FROM project_docs d WHERE d.project_id = p.id)"
)


def _row_to_project(row) -> dict:
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "created_at": row[3],
        "session_count": row[4],
        "doc_count": row[5],
    }


class MemoryStore:
    """One SQLite database (WAL, synchronous=NORMAL) behind a process-wide lock."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Connection = connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Lightweight migration guards, all idempotent.

        - wave 2 (additive 2026-07-21b): sessions gained nullable project_id and
          workdir columns (workdir is internal — the code-mode jail root a session
          remembers; it is never part of the API session shape).
        - FTS relevance (2026-07-21 refinement): any FTS shadow table created with
          the old `porter unicode61` tokenizer is dropped, recreated with the
          FTS_TOKENIZE/FTS_PREFIX spec, and repopulated from its content table via
          FTS5's 'rebuild' command. Detection reads sqlite_master, so a database
          already on the new spec is untouched — safe to run on every startup."""
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(sessions)").fetchall()}
        if "project_id" not in columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN project_id TEXT")
        if "workdir" not in columns:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN workdir TEXT")
        for table, create_sql in _FTS_TABLES.items():
            row = self._conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if row is None or FTS_TOKENIZE in (row[0] or ""):
                continue  # missing (created fresh by _SCHEMA) or already migrated
            # The content-table triggers reference these tables by name and survive
            # the drop; the recreate below lands before anything can fire them.
            self._conn.execute(f"DROP TABLE {table}")
            self._conn.execute(create_sql)
            self._conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- entries -----------------------------------------------------------
    def add_entry(
        self,
        layer: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        *,
        source_session: str | None = None,
        entry_id: str | None = None,
    ) -> MemoryEntry:
        now = _now_iso()
        entry = MemoryEntry(
            id=entry_id or memory_id(),
            layer=layer,
            title=title,
            content=content,
            tags=tags or [],
            created_at=now,
            updated_at=now,
            source_session=source_session,
        )
        with self._lock:
            self._conn.execute(
                f"INSERT INTO memory_entries ({_ENTRY_COLS}) VALUES (?,?,?,?,?,?,?,?)",
                (
                    entry.id,
                    entry.layer,
                    entry.title,
                    entry.content,
                    json.dumps(entry.tags),
                    entry.created_at,
                    entry.updated_at,
                    entry.source_session,
                ),
            )
            self._conn.commit()
        return entry

    def get_entry(self, entry_id: str) -> MemoryEntry | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_ENTRY_COLS} FROM memory_entries WHERE id = ?", (entry_id,)
            ).fetchone()
        return _row_to_entry(row) if row else None

    def update_entry(
        self,
        entry_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry | None:
        current = self.get_entry(entry_id)
        if current is None:
            return None
        new_title = title if title is not None else current.title
        new_content = content if content is not None else current.content
        new_tags = tags if tags is not None else current.tags
        now = _now_iso()
        with self._lock:
            self._conn.execute(
                "UPDATE memory_entries SET title=?, content=?, tags=?, updated_at=? WHERE id=?",
                (new_title, new_content, json.dumps(new_tags), now, entry_id),
            )
            self._conn.commit()
        return self.get_entry(entry_id)

    def delete_entry(self, entry_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def delete_layer_mirror(self, layer: str) -> None:
        """Drop all mirror rows of a file-backed layer (before re-mirroring)."""
        with self._lock:
            self._conn.execute("DELETE FROM memory_entries WHERE layer = ?", (layer,))
            self._conn.commit()

    def list_entries(
        self, layer: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[MemoryEntry], int]:
        where = "WHERE layer = ?" if layer else ""
        params: tuple = (layer,) if layer else ()
        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM memory_entries {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT {_ENTRY_COLS} FROM memory_entries {where} "
                "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ).fetchall()
        return [_row_to_entry(r) for r in rows], total

    def search(self, query: str, limit: int = 12) -> list[dict]:
        """bm25-ranked search over memory_fts. Result dicts match api.md
        /v1/memory/search: {entry, score, snippet}."""
        match = fts_query(query)
        if not match:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {', '.join('e.' + c for c in _ENTRY_COLS.split(', '))}, "
                "bm25(memory_fts) AS score, "
                f"snippet(memory_fts, 1, '[', ']', '…', {SNIPPET_TOKENS}) AS snip "
                "FROM memory_fts JOIN memory_entries e ON e.rowid = memory_fts.rowid "
                "WHERE memory_fts MATCH ? ORDER BY score LIMIT ?",
                (match, limit),
            ).fetchall()
        return [
            {"entry": _row_to_entry(row[:8]).as_dict(), "score": row[8], "snippet": row[9]}
            for row in rows
        ]

    # -- sessions ----------------------------------------------------------
    def ensure_session(
        self,
        session_id: str,
        mode: str,
        project_id: str | None = None,
        workdir: str | None = None,
    ) -> None:
        """Create the session row if new; a continued session keeps its original mode
        and started_at. A project_id (re)binds the session to that project; a workdir
        (re)pins the session's code-mode tool jail (internal column, api.md §1)."""
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions (id, mode, started_at) VALUES (?,?,?)",
                (session_id, mode, _now_iso()),
            )
            if project_id is not None:
                self._conn.execute(
                    "UPDATE sessions SET project_id = ? WHERE id = ?", (project_id, session_id)
                )
            if workdir is not None:
                self._conn.execute(
                    "UPDATE sessions SET workdir = ? WHERE id = ?", (workdir, session_id)
                )
            self._conn.commit()

    def session_workdir(self, session_id: str) -> str | None:
        """The session's remembered code-mode workdir (None: default per-session jail)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT workdir FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row[0] if row else None

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Archive one chat message. This is on the chat hot path, so a locked
        database must DEGRADE, never 500 (audit 2026-07-22): busy_timeout already
        gives writers BUSY_TIMEOUT_MS to acquire the lock; if that still fails under
        extreme contention, log and drop THIS archive write so the chat response
        still returns. Non-lock OperationalErrors (corrupt db, disk I/O) still
        surface — only 'database is locked' is downgraded."""
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                    (session_id, role, content, _now_iso()),
                )
                self._conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1, ended_at = ? "
                    "WHERE id = ?",
                    (_now_iso(), session_id),
                )
                self._conn.commit()
        except OperationalError as exc:
            if "locked" not in str(exc).lower():
                raise
            logger.warning(
                "memory archive write for session %r skipped (database busy): %s",
                session_id,
                exc,
            )

    def set_session_title(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            self._conn.commit()

    def list_sessions(
        self,
        query: str | None = None,
        limit: int = 50,
        offset: int = 0,
        project_id: str | None = None,
        mode: str | None = None,
    ) -> list[dict]:
        conditions: list[str] = []
        params: list = []
        if query:
            match = fts_query(query)
            if not match:
                return []
            conditions.append(
                "s.id IN (SELECT DISTINCT m.session_id FROM messages_fts "
                "JOIN messages m ON m.id = messages_fts.rowid WHERE messages_fts MATCH ?)"
            )
            params.append(match)
        if project_id is not None:
            conditions.append("s.project_id = ?")
            params.append(project_id)
        # mode=chat|code filter (api.md 2026-07-22b): dai's Chat and Code tabs show
        # separate recents.
        if mode is not None:
            conditions.append("s.mode = ?")
            params.append(mode)
        where = f"WHERE {' AND '.join(conditions)} " if conditions else ""
        sql = (
            f"SELECT {_SESSION_COLS} FROM sessions s {where}"
            "ORDER BY s.started_at DESC LIMIT ? OFFSET ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, (*params, limit, offset)).fetchall()
        return [_row_to_session(r) for r in rows]

    def session_transcript(self, session_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_SESSION_COLS} FROM sessions s WHERE s.id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            messages = self._conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return {
            "session": _row_to_session(row),
            "messages": [{"role": m[0], "content": m[1], "created_at": m[2]} for m in messages],
        }

    def delete_session(self, session_id: str) -> bool:
        """Delete an archived session and its messages. Returns False if unknown.

        `messages_fts` is kept in step by the `messages_ad` trigger as rows leave
        `messages`. Any archive entry that cited this session as its source has the
        back-reference nulled first, so the delete never trips the (ON) foreign key.
        """
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                return False
            self._conn.execute(
                "UPDATE memory_entries SET source_session = NULL WHERE source_session = ?",
                (session_id,),
            )
            self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        return True

    def search_messages(self, query: str, limit: int = 12) -> list[dict]:
        """bm25-ranked hits from session transcripts (for the recall tool)."""
        match = fts_query(query)
        if not match:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT m.session_id, m.role, m.content, bm25(messages_fts) AS score, "
                f"snippet(messages_fts, 0, '[', ']', '…', {SNIPPET_TOKENS}) AS snip "
                "FROM messages_fts JOIN messages m ON m.id = messages_fts.rowid "
                "WHERE messages_fts MATCH ? ORDER BY score LIMIT ?",
                (match, limit),
            ).fetchall()
        return [
            {"session_id": r[0], "role": r[1], "content": r[2], "score": r[3], "snippet": r[4]}
            for r in rows
        ]

    # -- projects (api.md §9, additive 2026-07-21b) -------------------------
    def add_project(self, name: str, description: str = "") -> dict:
        new_id = new_project_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO projects (id, name, description, created_at) VALUES (?,?,?,?)",
                (new_id, name, description, _now_iso()),
            )
            self._conn.commit()
        project = self.get_project(new_id)
        assert project is not None
        return project

    def get_project(self, project_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_PROJECT_COLS} FROM projects p WHERE p.id = ?", (project_id,)
            ).fetchone()
        return _row_to_project(row) if row else None

    def list_projects(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_PROJECT_COLS} FROM projects p ORDER BY p.created_at DESC, p.id DESC"
            ).fetchall()
        return [_row_to_project(r) for r in rows]

    def update_project(
        self, project_id: str, *, name: str | None = None, description: str | None = None
    ) -> dict | None:
        current = self.get_project(project_id)
        if current is None:
            return None
        new_name = name if name is not None else current["name"]
        new_description = description if description is not None else current["description"]
        with self._lock:
            self._conn.execute(
                "UPDATE projects SET name = ?, description = ? WHERE id = ?",
                (new_name, new_description, project_id),
            )
            self._conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        """Delete a project and its docs (FK cascade). Member sessions survive with
        project_id cleared (api.md §9)."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET project_id = NULL WHERE project_id = ?", (project_id,)
            )
            cursor = self._conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    # -- project docs -------------------------------------------------------
    def add_project_doc(self, project_id: str, title: str, content: str) -> dict:
        new_id = new_doc_id()
        with self._lock:
            self._conn.execute(
                "INSERT INTO project_docs (id, project_id, title, content, created_at) "
                "VALUES (?,?,?,?,?)",
                (new_id, project_id, title, content, _now_iso()),
            )
            self._conn.commit()
        doc = self.get_project_doc(project_id, new_id)
        assert doc is not None
        return doc

    def list_project_docs(self, project_id: str) -> list[dict]:
        """Doc metadata only (no content) — the GET /v1/projects/{id}/docs shape."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, content, created_at FROM project_docs "
                "WHERE project_id = ? ORDER BY created_at DESC, id DESC",
                (project_id,),
            ).fetchall()
        return [
            {"id": r[0], "title": r[1], "bytes": len(r[2].encode("utf-8")), "created_at": r[3]}
            for r in rows
        ]

    def get_project_doc(self, project_id: str, doc_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, title, content, created_at FROM project_docs "
                "WHERE project_id = ? AND id = ?",
                (project_id, doc_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "title": row[1],
            "bytes": len(row[2].encode("utf-8")),
            "created_at": row[3],
            "content": row[2],
        }

    def delete_project_doc(self, project_id: str, doc_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM project_docs WHERE project_id = ? AND id = ?", (project_id, doc_id)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def search_project_docs(self, project_id: str, query: str, limit: int = 4) -> list[dict]:
        """bm25-ranked excerpts from ONE project's docs — the chat injection source."""
        match = fts_query(query)
        if not match:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT d.id, d.title, bm25(project_docs_fts) AS score, "
                f"snippet(project_docs_fts, 1, '', '', '…', {DOC_EXCERPT_TOKENS}) AS excerpt "
                "FROM project_docs_fts JOIN project_docs d ON d.rowid = project_docs_fts.rowid "
                "WHERE project_docs_fts MATCH ? AND d.project_id = ? "
                "ORDER BY score LIMIT ?",
                (match, project_id, limit),
            ).fetchall()
        return [{"doc_id": r[0], "title": r[1], "score": r[2], "excerpt": r[3]} for r in rows]
