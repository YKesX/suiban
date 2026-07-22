"""The scheduler: an asyncio lifespan task that fires due schedules through the
internal chat pipeline.

Rules (api.md §10 + plan): check interval ~20s; next_run_at is computed against the
server-local wall clock; a schedule with a run already in flight is SKIPPED (never
overlapped); disabled schedules never fire. Each run is an ordinary chat session
(mode chat|code, effort, project_id honored, archived + auto-titled) executed by the
injected `run_schedule` seam; afterwards last_run_at / last_session_id / last_error
are recorded and the generalized gateway hook `notify(kind, title, summary)` fires —
the same hook research-job completions use.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING

from suiban.errors import BonsaiError
from suiban.schedules.store import (
    Schedule,
    ScheduleStore,
    compute_next_run,
    iso_utc,
    parse_iso_utc,
)

if TYPE_CHECKING:
    from suiban.app import AppState

logger = logging.getLogger(__name__)

CHECK_INTERVAL_S = 20.0
NOTIFY_SUMMARY_MAX_CHARS = 200

# async (schedule, session_id) -> (final assistant text, error or None)
RunScheduleFn = Callable[[Schedule, str], Awaitable[tuple[str, str | None]]]
NotifyFn = Callable[[str, str, str], None]


def new_run_session_id() -> str:
    return f"sched-{uuid.uuid4().hex[:12]}"


def _summary_line(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) > NOTIFY_SUMMARY_MAX_CHARS:
        flat = flat[: NOTIFY_SUMMARY_MAX_CHARS - 1] + "…"
    return flat or "(empty reply)"


class Scheduler:
    """Owns schedule execution. The HTTP router does CRUD against `store` and calls
    `run_now`; the app lifespan calls start()/shutdown()."""

    def __init__(
        self,
        store: ScheduleStore,
        *,
        run_schedule: RunScheduleFn,
        notify: NotifyFn,
        check_interval_s: float = CHECK_INTERVAL_S,
    ) -> None:
        self.store = store
        self._run_schedule = run_schedule
        self._notify = notify
        self._check_interval_s = check_interval_s
        self._loop_task: asyncio.Task | None = None
        self._running: dict[str, asyncio.Task] = {}  # schedule id -> in-flight run

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._loop_task is None:
            self._loop_task = asyncio.get_running_loop().create_task(self._loop())

    async def shutdown(self) -> None:
        tasks = [t for t in (self._loop_task, *self._running.values()) if t is not None]
        self._loop_task = None
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._running.clear()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._check_interval_s)
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - the scheduler must outlive any bad tick
                logger.exception("scheduler tick failed")

    # -- firing ------------------------------------------------------------
    def running(self, schedule_id: str) -> bool:
        return schedule_id in self._running

    def tick(self, now: datetime | None = None) -> list[str]:
        """Fire every due schedule (enabled, next_run_at passed, nothing in flight).
        Returns the session ids of launched runs (tests use them)."""
        moment = now if now is not None else datetime.now().astimezone()
        launched: list[str] = []
        for schedule in self.store.list():
            if not schedule.enabled or schedule.next_run_at is None:
                continue
            if schedule.id in self._running:
                continue  # a run is in flight — never overlap
            if parse_iso_utc(schedule.next_run_at) > moment:
                continue
            launched.append(self._launch(schedule))
        return launched

    def run_now(self, schedule_id: str) -> str:
        """POST /v1/schedules/{id}/run — immediate run, returns the session id."""
        schedule = self.store.get(schedule_id)
        if schedule is None:
            raise BonsaiError(404, f"no such schedule: {schedule_id}", code="schedule_not_found")
        if schedule.id in self._running:
            raise BonsaiError(
                409,
                f"schedule {schedule_id} already has a run in flight; runs never overlap",
                code="schedule_run_active",
            )
        return self._launch(schedule)

    def _launch(self, schedule: Schedule) -> str:
        session_id = new_run_session_id()
        task = asyncio.get_running_loop().create_task(self._execute(schedule.id, session_id))
        self._running[schedule.id] = task
        task.add_done_callback(lambda _t, sid=schedule.id: self._running.pop(sid, None))
        return session_id

    async def _execute(self, schedule_id: str, session_id: str) -> None:
        schedule = self.store.get(schedule_id)
        if schedule is None:  # deleted between launch and execution
            return
        text = ""
        error: str | None = None
        try:
            text, error = await self._run_schedule(schedule, session_id)
        except asyncio.CancelledError:
            raise  # shutdown — record nothing half-true
        except Exception as exc:  # noqa: BLE001 - a failed run is a state, not a crash
            logger.exception("scheduled run %s failed", schedule_id)
            error = str(exc)
        finished = datetime.now().astimezone()
        current = self.store.get(schedule_id)
        if current is None:  # deleted while running — nothing left to record
            return
        self.store.update(
            schedule_id,
            last_run_at=iso_utc(finished),
            last_session_id=session_id,
            last_error=error,
            # current.cadence, not the launch-time snapshot: a PATCH mid-run wins.
            next_run_at=iso_utc(compute_next_run(current.cadence, finished)),
        )
        try:
            if error is None:
                self._notify(
                    "schedule", f"Scheduled run finished: {schedule.name}", _summary_line(text)
                )
            else:
                self._notify(
                    "schedule", f"Scheduled run failed: {schedule.name}", _summary_line(error)
                )
        except Exception:  # noqa: BLE001 - a broken notifier must not break schedules
            logger.exception("schedule notification failed for %s", schedule_id)


def make_run_schedule(state: AppState) -> RunScheduleFn:
    """The production RunScheduleFn: execute through the internal chat pipeline so a
    scheduled run is indistinguishable from a normal session."""

    async def run(schedule: Schedule, session_id: str) -> tuple[str, str | None]:
        # Deferred import: routers.chat imports suiban.app at module level.
        from suiban.routers.chat import run_internal_chat

        return await run_internal_chat(
            state,
            prompt=schedule.prompt,
            mode=schedule.mode,
            effort=schedule.effort,
            session_id=session_id,
            project_id=schedule.project_id,
        )

    return run
