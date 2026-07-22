"""Hostile-environment tests (audit 2026-07-22, workstream B).

Deterministic, no real GPU / network / weights. Each asserts that a broken
environment — garbage config, a dead llama-server, a taken port, a locked database,
hostile unicode, concurrent writers — turns into a clean BonsaiError / logged
degrade / honest failure state, never a raw traceback or a 500.

The disk-full path (installer/models.py `hf_hub_download` -> OSError -> BonsaiError
`model_download_failed`) is covered by tests/test_installer.py (added by workstream R);
not duplicated here.
"""

from __future__ import annotations

import socket
import sqlite3
import threading
from pathlib import Path

import pytest

from suiban.cli import probe_bind
from suiban.config import ConfigManager
from suiban.errors import BonsaiError
from suiban.memory.store import BUSY_TIMEOUT_MS, MemoryStore
from suiban.research.jobs import Job, JobStore
from suiban.routers.chat import safe_workdir_name
from suiban.schedules.store import Cadence, Schedule, ScheduleStore


# -- garbage config.toml ------------------------------------------------------
def test_garbage_config_toml_is_clean_bonsai_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_text("this is [not = valid == toml ][[[\n", encoding="utf-8")
    with pytest.raises(BonsaiError) as exc_info:
        ConfigManager(home).load()
    assert exc_info.value.code == "config_invalid_toml"
    assert str(home / "config.toml") in exc_info.value.message
    assert "delete it" in exc_info.value.message  # names the remedy, not a traceback


def test_non_utf8_config_toml_is_clean_bonsai_error(tmp_path: Path) -> None:
    """A config.toml hand-saved in a non-UTF-8 encoding is a clean error, not a
    UnicodeDecodeError traceback out of read_text."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.toml").write_bytes(b"\xff\xfe host = 'x'")  # invalid UTF-8 lead bytes
    with pytest.raises(BonsaiError) as exc_info:
        ConfigManager(home).load()
    assert exc_info.value.code == "config_invalid_toml"


def test_garbage_staged_toml_is_clean_bonsai_error(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    ConfigManager(home).load()  # writes a valid config.toml
    (home / "staged.toml").write_text("= = broken\n", encoding="utf-8")
    with pytest.raises(BonsaiError) as exc_info:
        ConfigManager(home).load()
    assert exc_info.value.code == "config_invalid_toml"
    assert "staged.toml" in exc_info.value.message


# -- port already in use ------------------------------------------------------
def test_probe_bind_reports_port_in_use() -> None:
    """The serve() pre-flight bind probe reports an occupied port with a suiban
    remedy instead of letting uvicorn raise [Errno 98]."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        message = probe_bind("127.0.0.1", port)
        assert message is not None
        assert f"port {port} is already in use" in message
        assert "server.port" in message
    finally:
        holder.close()


def test_probe_bind_free_port_returns_none() -> None:
    # Port 0 asks the OS for any free port — always bindable, so the probe is clear.
    assert probe_bind("127.0.0.1", 0) is None


# -- dead llama-server (truncated/corrupt weights surrogate) ------------------
# A real corrupt GGUF makes llama-server die at load; RealBackend must surface that as
# slot state "failed" plus a stderr diagnostic, never hang or crash. We simulate the
# dead process with a tiny stub that exits at startup (the same seam the corrupt-weight
# path hits). This mirrors tests/test_llama_chaos.py without its full harness.
_DYING_STUB = """#!/usr/bin/env python3
import sys
print("error: failed to load model: invalid magic (corrupt GGUF)", file=sys.stderr, flush=True)
sys.exit(1)
"""


async def test_realbackend_dead_process_is_failed_with_stderr(tmp_path: Path) -> None:
    from suiban.llama.backend import RealBackend
    from suiban.sched.planner import PlannedSlot

    stub = tmp_path / "stub-llama-server"
    stub.write_text(_DYING_STUB, encoding="utf-8")
    stub.chmod(0o755)
    with socket.socket() as probe_sock:
        probe_sock.bind(("127.0.0.1", 0))
        port = probe_sock.getsockname()[1]
    planned = PlannedSlot(
        slot_id="worker-1",
        role="worker",
        model="bonsai-8b",
        family="ternary",
        ctx=8192,
        gpu=None,
        port=port,
        vram_mb=0,
    )
    backend = RealBackend(planned, binary=stub, flags=["--port", str(port)])
    await backend.start()
    try:
        assert planned.state == "failed"
        import asyncio

        await asyncio.sleep(0.05)  # let the stderr drain task catch the line
        tail = backend.stderr_tail()
        assert any("corrupt GGUF" in line for line in tail)
    finally:
        await backend.stop()


# -- busy_timeout is set on all three stores ----------------------------------
def test_busy_timeout_set_on_all_stores(tmp_path: Path) -> None:
    """Every store sharing memory.sqlite sets PRAGMA busy_timeout explicitly so a
    second writer waits for the lock instead of erroring immediately."""
    db = tmp_path / "memory.sqlite"
    stores = [MemoryStore(db), JobStore(db), ScheduleStore(db)]
    try:
        for store in stores:
            value = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
            assert value == BUSY_TIMEOUT_MS == 5000
    finally:
        for store in stores:
            store.close()


# -- locked-database degrade on the chat hot path -----------------------------
class _LockedConn:
    """A connection stand-in whose writes always raise 'database is locked'."""

    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    def commit(self):  # pragma: no cover - never reached, execute raises first
        pass


class _BrokenConn:
    def execute(self, *args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    def commit(self):  # pragma: no cover
        pass


def test_add_message_degrades_on_locked_db(tmp_path: Path) -> None:
    """A locked database on add_message is logged and dropped — the chat response
    still returns (no 500). Only 'database is locked' is downgraded."""
    store = MemoryStore(tmp_path / "memory.sqlite")
    try:
        store._conn = _LockedConn()  # type: ignore[assignment]
        # Must NOT raise — the archive write is best-effort on a busy db.
        store.add_message("s1", "user", "盆栽")
    finally:
        store._lock = threading.Lock()  # detach the fake before close()


def test_add_message_reraises_non_lock_operational_error(tmp_path: Path) -> None:
    """A genuine OperationalError (corrupt db, disk I/O) still surfaces — the degrade
    is scoped to lock contention only, not a mask over real corruption."""
    store = MemoryStore(tmp_path / "memory.sqlite")
    store._conn = _BrokenConn()  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError):
        store.add_message("s1", "user", "x")


# -- hostile unicode round-trips ----------------------------------------------
def test_unicode_session_and_message_round_trip(tmp_path: Path) -> None:
    """A 盆栽 session id + message content survives the archive (parameterized SQL)
    and the FTS recall path; the on-disk workdir name is derived safely."""
    store = MemoryStore(tmp_path / "memory.sqlite")
    try:
        session_id = "盆栽-会話-🌸"
        store.ensure_session(session_id, "chat")
        store.add_message(session_id, "user", "盆栽 は a bonsai 🌸 tree")
        transcript = store.session_transcript(session_id)
        assert transcript is not None
        assert transcript["messages"][0]["content"] == "盆栽 は a bonsai 🌸 tree"
        # FTS recall over the same content (fts_query neutralizes any hostile bytes).
        hits = store.search_messages("盆栽")
        assert any(h["session_id"] == session_id for h in hits)
    finally:
        store.close()


def test_safe_workdir_name_neutralizes_hostile_ids() -> None:
    # A strict, already-safe id is used verbatim (existing sessions keep their dir).
    assert safe_workdir_name("tg-12345") == "tg-12345"
    assert safe_workdir_name("anon-deadbeef") == "anon-deadbeef"
    # Traversal / unicode / empty -> a stable 16-char sha256 digest that cannot escape.
    for hostile in ["../../etc", "盆栽-session", "", "a/b\\c", "..", "with space"]:
        name = safe_workdir_name(hostile)
        assert len(name) == 16 and name.isalnum()
        assert "/" not in name and "\\" not in name and ".." not in name


# NOTE (KNOWN_ISSUES, Windows-only — cannot be tested on this Linux CI): the
# safe_workdir_name character class [A-Za-z0-9_-] admits Windows reserved DEVICE names
# (CON, NUL, PRN, AUX, COM1-9, LPT1-9) verbatim, and a trailing dot/space is legal on
# POSIX but stripped/illegal on Windows. On Windows, a session named "CON" would fail
# to mkdir its workdir jail. POSIX (the shipped target) is unaffected. Documented, not
# fixed here (no Windows to verify a fix against).
def test_safe_workdir_name_windows_reserved_is_passthrough_today() -> None:
    """Documents current behavior: 'CON' matches the safe class and passes through,
    which is a Windows-only KNOWN_ISSUE (see note above)."""
    assert safe_workdir_name("CON") == "CON"


# -- concurrency: no 'database is locked' escapes with busy_timeout ------------
def test_concurrent_writers_no_database_locked(tmp_path: Path) -> None:
    """N threads write across all three stores (memory add_message + jobs.update +
    schedules.update) against ONE memory.sqlite behind three separate connections.
    With WAL + busy_timeout=5000 no writer sees 'database is locked'."""
    db = tmp_path / "memory.sqlite"
    mem = MemoryStore(db)
    jobs = JobStore(db)
    scheds = ScheduleStore(db)

    mem.ensure_session("s1", "chat")
    jobs.add(
        Job(
            id="job_x",
            type="deep_research",
            query="q",
            effort="mid",
            state="running",
            stage=None,
            percent=0,
            created_at="2026-07-22T00:00:00Z",
            started_at="2026-07-22T00:00:00Z",
            finished_at=None,
            error=None,
        )
    )
    scheds.add(
        Schedule(
            id="sched_x",
            name="n",
            prompt="p",
            mode="chat",
            effort="mid",
            project_id=None,
            cadence=Cadence(kind="daily", time="07:30"),
            enabled=True,
            created_at="2026-07-22T00:00:00Z",
            last_run_at=None,
            next_run_at=None,
            last_session_id=None,
            last_error=None,
        )
    )

    n_threads = 8
    iters = 25
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def mem_writer() -> None:
        barrier.wait()
        for k in range(iters):
            try:
                mem.add_message("s1", "user", f"msg {k} 盆栽")
            except Exception as exc:  # noqa: BLE001 - collect for the assertion
                errors.append(exc)

    def job_writer() -> None:
        barrier.wait()
        for k in range(iters):
            try:
                jobs.update("job_x", percent=k % 100, stage=f"stage-{k}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    def sched_writer() -> None:
        barrier.wait()
        for k in range(iters):
            try:
                scheds.update("sched_x", last_error=f"tick {k}")
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

    workers = [mem_writer, job_writer, sched_writer]
    threads = [threading.Thread(target=workers[i % len(workers)]) for i in range(n_threads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
    finally:
        mem.close()
        jobs.close()
        scheds.close()

    locked = [e for e in errors if "locked" in str(e).lower()]
    assert not locked, f"database-locked errors escaped despite busy_timeout: {locked}"
    assert not errors, f"unexpected writer errors: {errors}"
