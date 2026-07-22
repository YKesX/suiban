"""Job model, SQLite persistence, and the in-process job manager.

Jobs live in a `jobs` table inside the same SQLite database as memory
(~/.bonsai/memory/memory.sqlite) — one local database, per plan. The JobStore opens
its own WAL connection; the JobManager runs jobs as asyncio tasks in the app's event
loop, enforces the v1 concurrency limit (max 1 research job — an honest local-GPU
constraint, see docs/research.md), fans out coarse progress to SSE subscribers, and
fires completion listeners (gateway pings, idle-apply commits).

Fairness with interactive chats: a job does NOT own the orchestrator slot for its
15-40 minute lifetime. The run_job pipeline acquires the slot's gate per pipeline
STEP (one completion at a time — research/wiring.py), so chats interleave with a
running job at step granularity instead of hitting 300 s timeouts blind.

Cancellation is synchronous-ish on purpose: cancel() cancels the task AND awaits its
unwind (bounded by CANCEL_UNWIND_TIMEOUT_S) so the in-flight llama-server request is
actually aborted before the DELETE returns; submit() refuses (429) while a previous
task is still unwinding — the single-job invariant covers teardown, not just the
running state.

JobStatus over HTTP exposes EXACTLY the api.md shape: id, type, query, state, stage,
percent, created_at, started_at, finished_at, error. Internals (effort, report path)
stay out of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from sqlite3 import Connection, connect

from suiban.errors import BonsaiError

logger = logging.getLogger(__name__)

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("completed", "failed", "cancelled")

# How long cancel() waits for the cancelled task to actually unwind (finally blocks,
# httpx teardown). A task still alive after this is logged and left to finish dying
# on its own; the job row is already terminal either way.
CANCEL_UNWIND_TIMEOUT_S = 10.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id          TEXT PRIMARY KEY,
  type        TEXT NOT NULL,
  query       TEXT NOT NULL,
  effort      TEXT NOT NULL,
  state       TEXT NOT NULL CHECK (state IN
                ('queued','running','completed','failed','cancelled')),
  stage       TEXT,
  percent     INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT NOT NULL,
  started_at  TEXT,
  finished_at TEXT,
  error       TEXT
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_job_id() -> str:
    # 128-bit random id (audit 2026-07-22): the previous 48-bit (uuid4[:12]) id was
    # guessable enough to matter once job ids gate report access.
    return f"job_{secrets.token_hex(16)}"


@dataclass(frozen=True)
class Job:
    id: str
    type: str
    query: str
    effort: str
    state: str
    stage: str | None
    percent: int
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def status_dict(self) -> dict:
        """The api.md JobStatus shape — coarse fields only."""
        return {
            "id": self.id,
            "type": self.type,
            "query": self.query,
            "state": self.state,
            "stage": self.stage,
            "percent": self.percent,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


_COLS = "id, type, query, effort, state, stage, percent, created_at, started_at, finished_at, error"


def _row_to_job(row) -> Job:
    return Job(*row)


class JobStore:
    """jobs table access (own connection into the shared memory.sqlite, WAL)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Connection = connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Wait up to 5s for the write lock instead of erroring immediately: this
            # connection shares memory.sqlite with the memory + schedule stores (audit
            # 2026-07-22, see memory/store.py BUSY_TIMEOUT_MS).
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, job: Job) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO jobs ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job.id,
                    job.type,
                    job.query,
                    job.effort,
                    job.state,
                    job.stage,
                    job.percent,
                    job.created_at,
                    job.started_at,
                    job.finished_at,
                    job.error,
                ),
            )
            self._conn.commit()

    def update(self, job_id: str, **fields) -> Job | None:
        if fields:
            sets = ", ".join(f"{key} = ?" for key in fields)
            with self._lock:
                self._conn.execute(
                    f"UPDATE jobs SET {sets} WHERE id = ?", (*fields.values(), job_id)
                )
                self._conn.commit()
        return self.get(job_id)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(f"SELECT {_COLS} FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def list(self, limit: int = 100) -> list[Job]:
        """Newest first (api.md)."""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def active_count(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')"
            ).fetchone()
        return int(row[0])

    def fail_orphans(self, reason: str) -> int:
        """Jobs left queued/running by a previous process are dead — say so."""
        now = _now_iso()
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE jobs SET state='failed', error=?, finished_at=? "
                "WHERE state IN ('queued','running')",
                (reason, now),
            )
            self._conn.commit()
        return cursor.rowcount


class JobCancelled(Exception):
    """Raised inside a job task when cancellation was requested."""


class JobManager:
    """Owns job execution: one asyncio task per job, v1 limit of one active research
    job, SSE fan-out, and terminal-state listeners."""

    def __init__(
        self,
        store: JobStore,
        *,
        run_job: Callable,  # async (job: Job, progress: async (stage, percent)) -> str (report md)
        reports_dir: Path,
    ) -> None:
        self._store = store
        self._run_job = run_job
        self._reports_dir = reports_dir
        self._tasks: dict[str, asyncio.Task] = {}
        self._watchers: dict[str, list[asyncio.Queue]] = {}
        self._listeners: list[Callable[[Job], None]] = []

    # -- lifecycle ---------------------------------------------------------
    def startup(self) -> None:
        orphaned = self._store.fail_orphans("suiban restarted while this job was in flight")
        if orphaned:
            logger.warning("marked %d orphaned research job(s) failed on startup", orphaned)

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for task in list(self._tasks.values()):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

    # -- listeners / watchers ----------------------------------------------
    def add_listener(self, callback: Callable[[Job], None]) -> None:
        """Called (synchronously, in the event loop) on every terminal transition."""
        self._listeners.append(callback)

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._watchers.setdefault(job_id, []).append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        watchers = self._watchers.get(job_id, [])
        if queue in watchers:
            watchers.remove(queue)
        if not watchers:
            self._watchers.pop(job_id, None)

    def _notify(self, job_id: str, payload: dict) -> None:
        for queue in self._watchers.get(job_id, []):
            queue.put_nowait(payload)

    def _fire_listeners(self, job: Job) -> None:
        for callback in self._listeners:
            try:
                callback(job)
            except Exception:  # noqa: BLE001 - a broken listener must not break jobs
                logger.exception("job listener failed for %s", job.id)

    # -- accessors ----------------------------------------------------------
    @property
    def active(self) -> int:
        return self._store.active_count()

    def get(self, job_id: str) -> Job | None:
        return self._store.get(job_id)

    def list(self) -> list[Job]:
        return self._store.list()

    def report_path(self, job_id: str) -> Path:
        return self._reports_dir / f"{job_id}.md"

    # -- submission ---------------------------------------------------------
    def submit(self, query: str, effort: str) -> Job:
        if self._store.active_count() > 0:
            raise BonsaiError(
                429,
                "a deep-research job is already active; v1 runs at most one at a time "
                "(the loadout is a single local GPU). Wait for it or DELETE it first.",
                code="research_job_active",
            )
        if any(not task.done() for task in self._tasks.values()):
            # Terminal in the store but the task is still unwinding (cancel() timed
            # out, or a concurrent cancel is mid-await): admitting a new job now
            # would double-use the orchestrator. 429, same code — retry in seconds.
            raise BonsaiError(
                429,
                "the previous research job is still shutting down; retry in a moment",
                code="research_job_active",
            )
        job = Job(
            id=new_job_id(),
            type="deep_research",
            query=query,
            effort=effort,
            state="queued",
            stage=None,
            percent=0,
            created_at=_now_iso(),
            started_at=None,
            finished_at=None,
            error=None,
        )
        self._store.add(job)
        self._tasks[job.id] = asyncio.get_running_loop().create_task(self._execute(job.id))
        return job

    async def cancel(self, job_id: str) -> Job:
        """Idempotent cancel: active jobs transition to cancelled; repeating the call
        keeps returning cancelled; a job that already finished keeps its truthful
        terminal state (docs/research.md). The terminal write happens HERE (not in
        the task's cancellation handler) so cancelling a task that never got to run
        still records honestly.

        The task's unwind is AWAITED (bounded): task.cancel() raises CancelledError
        at the pipeline's current await — usually an httpx call to llama-server,
        which aborts the in-flight request — and only once the task finished (or
        CANCEL_UNWIND_TIMEOUT_S passed) does this return, so a DELETE that answered
        "cancelled" means the GPU actually stopped, not "will stop eventually"."""
        job = self._store.get(job_id)
        if job is None:
            raise BonsaiError(404, f"no such job: {job_id}", code="job_not_found")
        if job.terminal:
            return job
        updated = self._set_terminal(job_id, "cancelled")
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
            # asyncio.wait, not wait_for: on timeout it just reports the task as
            # pending instead of force-cancelling and BLOCKING until the task dies
            # (wait_for's timeout only fires after cancellation completes — a
            # wedged unwind would hang the DELETE indefinitely). A task still
            # pending here stays tracked in _tasks, and submit() refuses new jobs
            # until it actually finishes.
            _done, pending = await asyncio.wait({task}, timeout=CANCEL_UNWIND_TIMEOUT_S)
            if pending:
                logger.warning(
                    "research job %s: task did not unwind within %.0fs of cancel",
                    job_id,
                    CANCEL_UNWIND_TIMEOUT_S,
                )
        return updated if updated is not None else replace(job, state="cancelled")

    # -- execution ----------------------------------------------------------
    def _set_terminal(self, job_id: str, state: str, error: str | None = None) -> Job | None:
        """Record a terminal state exactly once; later attempts (cancel-vs-complete
        races, the task's own CancelledError handler) are no-ops."""
        current = self._store.get(job_id)
        if current is None or current.terminal:
            return current
        job = self._store.update(
            job_id,
            state=state,
            error=error,
            finished_at=_now_iso(),
            **({"percent": 100, "stage": None} if state == "completed" else {}),
        )
        self._notify(job_id, {"type": "state", "state": state})
        if job is not None:
            self._fire_listeners(job)
        return job

    async def _execute(self, job_id: str) -> None:
        job = self._store.update(job_id, state="running", started_at=_now_iso())
        assert job is not None
        self._notify(job_id, {"type": "state", "state": "running"})

        last: tuple[str, int] | None = None

        async def progress(stage: str, percent: int) -> None:
            nonlocal last
            percent = max(0, min(100, int(percent)))
            if last == (stage, percent):
                return  # SSE fires on change only (api.md)
            last = (stage, percent)
            self._store.update(job_id, stage=stage, percent=percent)
            self._notify(job_id, {"type": "progress", "stage": stage, "percent": percent})

        try:
            report_md = await self._run_job(job, progress)
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            self.report_path(job_id).write_text(report_md, encoding="utf-8")
            self._set_terminal(job_id, "completed")
        except asyncio.CancelledError:
            self._set_terminal(job_id, "cancelled")
            # Do not re-raise: cancellation of the JOB is a normal outcome; the task
            # itself finishes cleanly so shutdown() does not log it as a failure.
        except Exception as exc:  # noqa: BLE001 - a failed job is a state, not a crash
            logger.exception("research job %s failed", job_id)
            self._set_terminal(job_id, "failed", error=str(exc))
        finally:
            self._tasks.pop(job_id, None)
