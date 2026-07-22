"""Schedule model, cadence validation + next-run math, and SQLite persistence.

Schedules live in a `schedules` table inside the same SQLite database as memory
(~/.bonsai/memory/memory.sqlite) — one local database, like research jobs. Cadence
times are SERVER-LOCAL wall clock (api.md §10): "07:30" means 07:30 on this machine.
Computed timestamps are stored/exposed as UTC ISO like every other timestamp.

Cadence rules (contract): `daily` needs `time`; `weekly` needs `weekday` (0=Monday …
6=Sunday, Python convention) + `time`; `interval` needs `every_minutes` >= 5.
"""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from sqlite3 import Connection, connect

from suiban.errors import BonsaiError

CADENCE_KINDS = ("daily", "weekly", "interval")
MIN_INTERVAL_MINUTES = 5

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
  id                   TEXT PRIMARY KEY,
  name                 TEXT NOT NULL,
  prompt               TEXT NOT NULL,
  mode                 TEXT NOT NULL CHECK (mode IN ('chat','code')),
  effort               TEXT NOT NULL,
  project_id           TEXT,
  cadence_kind         TEXT NOT NULL CHECK (cadence_kind IN ('daily','weekly','interval')),
  cadence_time         TEXT,
  cadence_weekday      INTEGER,
  cadence_every_minutes INTEGER,
  enabled              INTEGER NOT NULL DEFAULT 1,
  created_at           TEXT NOT NULL,
  last_run_at          TEXT,
  next_run_at          TEXT,
  last_session_id      TEXT,
  last_error           TEXT
);
"""


def new_schedule_id() -> str:
    return f"sched_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_utc(moment: datetime) -> str:
    """Aware datetime -> the UTC ISO format used across the API."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso_utc(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class Cadence:
    kind: str
    time: str | None = None
    weekday: int | None = None
    every_minutes: int | None = None

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind}
        if self.time is not None:
            out["time"] = self.time
        if self.weekday is not None:
            out["weekday"] = self.weekday
        if self.every_minutes is not None:
            out["every_minutes"] = self.every_minutes
        return out


def validate_cadence(raw: object) -> Cadence:
    """Contract cadence rules -> Cadence, or a 400 `cadence_invalid` envelope."""

    def bad(message: str) -> BonsaiError:
        return BonsaiError(400, f"invalid cadence: {message}", code="cadence_invalid")

    if not isinstance(raw, dict):
        raise bad("must be an object {kind, ...}")
    unknown = set(raw) - {"kind", "time", "weekday", "every_minutes"}
    if unknown:
        raise bad(f"unknown fields: {', '.join(sorted(unknown))}")
    kind = raw.get("kind")
    if kind not in CADENCE_KINDS:
        raise bad(f"kind must be one of {', '.join(CADENCE_KINDS)}; got {kind!r}")

    time_value = raw.get("time")
    if time_value is not None and (
        not isinstance(time_value, str) or not _TIME_RE.match(time_value)
    ):
        raise bad(f"time must be 'HH:MM' (24h); got {time_value!r}")

    if kind == "interval":
        every = raw.get("every_minutes")
        if isinstance(every, bool) or not isinstance(every, int) or every < MIN_INTERVAL_MINUTES:
            raise bad(f"interval requires every_minutes >= {MIN_INTERVAL_MINUTES}; got {every!r}")
        return Cadence(kind="interval", every_minutes=every)

    if time_value is None:
        raise bad(f"{kind} requires a time ('HH:MM')")
    if kind == "weekly":
        weekday = raw.get("weekday")
        if isinstance(weekday, bool) or not isinstance(weekday, int) or not 0 <= weekday <= 6:
            raise bad(f"weekly requires weekday 0-6 (0=Monday); got {weekday!r}")
        return Cadence(kind="weekly", time=time_value, weekday=weekday)
    return Cadence(kind="daily", time=time_value)


def compute_next_run(cadence: Cadence, after: datetime) -> datetime:
    """The next fire time strictly after `after` (an AWARE server-local datetime)."""
    if cadence.kind == "interval":
        assert cadence.every_minutes is not None
        return after + timedelta(minutes=cadence.every_minutes)
    assert cadence.time is not None
    hour, minute = (int(part) for part in cadence.time.split(":"))
    candidate = after.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if cadence.kind == "daily":
        if candidate <= after:
            candidate += timedelta(days=1)
        return candidate
    assert cadence.weekday is not None
    candidate += timedelta(days=(cadence.weekday - after.weekday()) % 7)
    if candidate <= after:
        candidate += timedelta(days=7)
    return candidate


@dataclass(frozen=True)
class Schedule:
    id: str
    name: str
    prompt: str
    mode: str
    effort: str
    project_id: str | None
    cadence: Cadence
    enabled: bool
    created_at: str
    last_run_at: str | None
    next_run_at: str | None
    last_session_id: str | None
    last_error: str | None

    def as_dict(self) -> dict:
        """The api.md §10 Schedule shape."""
        return {
            "id": self.id,
            "name": self.name,
            "prompt": self.prompt,
            "mode": self.mode,
            "effort": self.effort,
            "project_id": self.project_id,
            "cadence": self.cadence.as_dict(),
            "enabled": self.enabled,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "next_run_at": self.next_run_at,
            "last_session_id": self.last_session_id,
            "last_error": self.last_error,
        }

    def with_cadence(self, cadence: Cadence) -> Schedule:
        return replace(self, cadence=cadence)


_COLS = (
    "id, name, prompt, mode, effort, project_id, cadence_kind, cadence_time, "
    "cadence_weekday, cadence_every_minutes, enabled, created_at, last_run_at, "
    "next_run_at, last_session_id, last_error"
)


def _row_to_schedule(row) -> Schedule:
    return Schedule(
        id=row[0],
        name=row[1],
        prompt=row[2],
        mode=row[3],
        effort=row[4],
        project_id=row[5],
        cadence=Cadence(kind=row[6], time=row[7], weekday=row[8], every_minutes=row[9]),
        enabled=bool(row[10]),
        created_at=row[11],
        last_run_at=row[12],
        next_run_at=row[13],
        last_session_id=row[14],
        last_error=row[15],
    )


class ScheduleStore:
    """schedules table access (own connection into the shared memory.sqlite, WAL)."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Connection = connect(str(db_path), check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            # Wait up to 5s for the write lock instead of erroring immediately: this
            # connection shares memory.sqlite with the memory + job stores (audit
            # 2026-07-22, see memory/store.py BUSY_TIMEOUT_MS).
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def add(self, schedule: Schedule) -> None:
        cadence = schedule.cadence
        with self._lock:
            self._conn.execute(
                f"INSERT INTO schedules ({_COLS}) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    schedule.id,
                    schedule.name,
                    schedule.prompt,
                    schedule.mode,
                    schedule.effort,
                    schedule.project_id,
                    cadence.kind,
                    cadence.time,
                    cadence.weekday,
                    cadence.every_minutes,
                    int(schedule.enabled),
                    schedule.created_at,
                    schedule.last_run_at,
                    schedule.next_run_at,
                    schedule.last_session_id,
                    schedule.last_error,
                ),
            )
            self._conn.commit()

    def get(self, schedule_id: str) -> Schedule | None:
        with self._lock:
            row = self._conn.execute(
                f"SELECT {_COLS} FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()
        return _row_to_schedule(row) if row else None

    def list(self) -> list[Schedule]:
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_COLS} FROM schedules ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return [_row_to_schedule(r) for r in rows]

    def update(self, schedule_id: str, **fields) -> Schedule | None:
        """Update Schedule-level fields; `cadence` expands to its four columns."""
        columns: dict = {}
        for key, value in fields.items():
            if key == "cadence":
                columns["cadence_kind"] = value.kind
                columns["cadence_time"] = value.time
                columns["cadence_weekday"] = value.weekday
                columns["cadence_every_minutes"] = value.every_minutes
            elif key == "enabled":
                columns["enabled"] = int(value)
            else:
                columns[key] = value
        if columns:
            sets = ", ".join(f"{name} = ?" for name in columns)
            with self._lock:
                self._conn.execute(
                    f"UPDATE schedules SET {sets} WHERE id = ?", (*columns.values(), schedule_id)
                )
                self._conn.commit()
        return self.get(schedule_id)

    def delete(self, schedule_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM schedules WHERE id = ?", (schedule_id,))
            self._conn.commit()
        return cursor.rowcount > 0
