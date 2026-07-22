"""Schedules (api.md §10): cadence validation + next-run math (weekly wrap, interval
floor), the scheduler (due firing, in-flight skip, disabled schedules, failure
recording, notify hook), and the /v1/schedules HTTP surface incl. run-now on the
mock backend."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban.errors import BonsaiError
from suiban.schedules.runner import Scheduler
from suiban.schedules.store import (
    Cadence,
    Schedule,
    ScheduleStore,
    compute_next_run,
    iso_utc,
    new_schedule_id,
    now_iso,
    parse_iso_utc,
    validate_cadence,
)

TZ = timezone(timedelta(hours=3))
# 2026-07-21 is a Tuesday (weekday 1).
TUESDAY_10 = datetime(2026, 7, 21, 10, 0, tzinfo=TZ)


# -- cadence validation -------------------------------------------------------
def test_validate_cadence_accepts_contract_shapes() -> None:
    assert validate_cadence({"kind": "daily", "time": "07:30"}) == Cadence("daily", time="07:30")
    assert validate_cadence({"kind": "weekly", "time": "09:00", "weekday": 0}) == Cadence(
        "weekly", time="09:00", weekday=0
    )
    assert validate_cadence({"kind": "interval", "every_minutes": 5}) == Cadence(
        "interval", every_minutes=5
    )


@pytest.mark.parametrize(
    "raw",
    [
        "not a dict",
        {"kind": "hourly"},
        {"kind": "daily"},  # daily needs time
        {"kind": "daily", "time": "25:00"},
        {"kind": "daily", "time": "07:5"},
        {"kind": "weekly", "time": "09:00"},  # weekly needs weekday
        {"kind": "weekly", "weekday": 1},  # ... and time
        {"kind": "weekly", "time": "09:00", "weekday": 7},
        {"kind": "weekly", "time": "09:00", "weekday": True},
        {"kind": "interval"},
        {"kind": "interval", "every_minutes": 4},  # floor is 5
        {"kind": "interval", "every_minutes": "5"},
        {"kind": "daily", "time": "07:30", "bogus": 1},
    ],
)
def test_validate_cadence_rejects(raw) -> None:
    with pytest.raises(BonsaiError) as err:
        validate_cadence(raw)
    assert err.value.status == 400
    assert err.value.code == "cadence_invalid"


# -- next-run math ------------------------------------------------------------
def test_next_run_daily_before_and_after_time() -> None:
    later_today = compute_next_run(Cadence("daily", time="10:30"), TUESDAY_10)
    assert later_today == TUESDAY_10.replace(hour=10, minute=30)
    tomorrow = compute_next_run(Cadence("daily", time="09:00"), TUESDAY_10)
    assert tomorrow == (TUESDAY_10 + timedelta(days=1)).replace(hour=9, minute=0)
    # Exactly-now fires strictly AFTER now: next day.
    exact = compute_next_run(Cadence("daily", time="10:00"), TUESDAY_10)
    assert exact == (TUESDAY_10 + timedelta(days=1)).replace(hour=10, minute=0)


def test_next_run_weekly_same_week_and_wrap() -> None:
    # Friday (4) is later this week.
    friday = compute_next_run(Cadence("weekly", time="09:00", weekday=4), TUESDAY_10)
    assert friday == (TUESDAY_10 + timedelta(days=3)).replace(hour=9, minute=0)
    assert friday.weekday() == 4
    # Monday (0) already passed: wraps to NEXT week.
    monday = compute_next_run(Cadence("weekly", time="09:00", weekday=0), TUESDAY_10)
    assert monday == (TUESDAY_10 + timedelta(days=6)).replace(hour=9, minute=0)
    assert monday.weekday() == 0
    # Same weekday, earlier time: wraps a full week.
    tuesday = compute_next_run(Cadence("weekly", time="09:00", weekday=1), TUESDAY_10)
    assert tuesday == (TUESDAY_10 + timedelta(days=7)).replace(hour=9, minute=0)
    # Same weekday, later time: today.
    tuesday_late = compute_next_run(Cadence("weekly", time="23:00", weekday=1), TUESDAY_10)
    assert tuesday_late == TUESDAY_10.replace(hour=23, minute=0)


def test_next_run_interval_adds_minutes() -> None:
    assert compute_next_run(
        Cadence("interval", every_minutes=45), TUESDAY_10
    ) == TUESDAY_10 + timedelta(minutes=45)


# -- scheduler ---------------------------------------------------------------
def _schedule(
    *,
    schedule_id: str | None = None,
    enabled: bool = True,
    next_run_at: str | None = None,
    cadence: Cadence | None = None,
) -> Schedule:
    return Schedule(
        id=schedule_id or new_schedule_id(),
        name="digest",
        prompt="summarize the day",
        mode="chat",
        effort="low",
        project_id=None,
        cadence=cadence or Cadence("interval", every_minutes=5),
        enabled=enabled,
        created_at=now_iso(),
        last_run_at=None,
        next_run_at=next_run_at,
        last_session_id=None,
        last_error=None,
    )


@pytest.fixture
def schedule_store(tmp_path: Path):
    store = ScheduleStore(tmp_path / "db.sqlite")
    yield store
    store.close()


def _past() -> str:
    return iso_utc(datetime.now(UTC) - timedelta(minutes=1))


async def test_scheduler_fires_due_and_records(schedule_store: ScheduleStore) -> None:
    runs: list[tuple[str, str]] = []
    pings: list[tuple[str, str, str]] = []

    async def run(schedule: Schedule, session_id: str) -> tuple[str, str | None]:
        runs.append((schedule.id, session_id))
        return "the day was quiet", None

    scheduler = Scheduler(schedule_store, run_schedule=run, notify=lambda *a: pings.append(a))
    schedule = _schedule(next_run_at=_past())
    schedule_store.add(schedule)

    launched = scheduler.tick()
    assert len(launched) == 1
    await asyncio.gather(*scheduler._running.values())

    assert runs == [(schedule.id, launched[0])]
    updated = schedule_store.get(schedule.id)
    assert updated is not None
    assert updated.last_session_id == launched[0]
    assert updated.last_run_at is not None
    assert updated.last_error is None
    assert parse_iso_utc(updated.next_run_at) > datetime.now(UTC)  # recomputed forward
    assert pings == [("schedule", "Scheduled run finished: digest", "the day was quiet")]

    # Not due any more: another tick launches nothing.
    assert scheduler.tick() == []
    await scheduler.shutdown()


async def test_scheduler_skips_disabled_and_in_flight(schedule_store: ScheduleStore) -> None:
    release = asyncio.Event()
    runs: list[str] = []

    async def slow_run(schedule: Schedule, session_id: str) -> tuple[str, str | None]:
        runs.append(session_id)
        await release.wait()
        return "done", None

    scheduler = Scheduler(schedule_store, run_schedule=slow_run, notify=lambda *a: None)
    disabled = _schedule(enabled=False, next_run_at=_past())
    active = _schedule(next_run_at=_past())
    schedule_store.add(disabled)
    schedule_store.add(active)

    launched = scheduler.tick()
    assert len(launched) == 1  # the disabled schedule NEVER fires
    assert scheduler.running(active.id)
    assert scheduler.tick() == []  # in flight -> skipped, never overlapped
    with pytest.raises(BonsaiError) as err:
        scheduler.run_now(active.id)
    assert err.value.status == 409
    assert err.value.code == "schedule_run_active"

    release.set()
    await asyncio.gather(*scheduler._running.values())
    assert runs == launched
    assert schedule_store.get(disabled.id).last_run_at is None
    await scheduler.shutdown()


async def test_scheduler_records_failures_and_notifies(schedule_store: ScheduleStore) -> None:
    pings: list[tuple[str, str, str]] = []

    async def bad_run(schedule: Schedule, session_id: str) -> tuple[str, str | None]:
        raise RuntimeError("slot exploded")

    scheduler = Scheduler(schedule_store, run_schedule=bad_run, notify=lambda *a: pings.append(a))
    schedule = _schedule(next_run_at=_past())
    schedule_store.add(schedule)
    scheduler.tick()
    await asyncio.gather(*scheduler._running.values())

    updated = schedule_store.get(schedule.id)
    assert updated is not None
    assert updated.last_error == "slot exploded"
    assert updated.last_run_at is not None
    assert updated.next_run_at is not None  # still scheduled forward — no dead schedules
    assert pings == [("schedule", "Scheduled run failed: digest", "slot exploded")]
    await scheduler.shutdown()


async def test_run_now_unknown_schedule_is_404(schedule_store: ScheduleStore) -> None:
    scheduler = Scheduler(schedule_store, run_schedule=lambda s, sid: None, notify=lambda *a: None)
    with pytest.raises(BonsaiError) as err:
        scheduler.run_now("sched_nope")
    assert err.value.status == 404
    await scheduler.shutdown()


# -- HTTP surface -------------------------------------------------------------
def test_schedule_http_validation(client: TestClient) -> None:
    base = {
        "name": "n",
        "prompt": "p",
        "cadence": {"kind": "interval", "every_minutes": 5},
    }
    assert client.post("/v1/schedules", json={}).status_code == 400
    assert client.post("/v1/schedules", json={**base, "name": " "}).status_code == 400
    without_prompt = {k: v for k, v in base.items() if k != "prompt"}
    assert client.post("/v1/schedules", json=without_prompt).status_code == 400
    without_cadence = {k: v for k, v in base.items() if k != "cadence"}
    assert client.post("/v1/schedules", json=without_cadence).status_code == 400
    assert client.post("/v1/schedules", json={**base, "mode": "ultra"}).status_code == 400
    assert client.post("/v1/schedules", json={**base, "effort": "turbo"}).status_code == 400
    assert client.post("/v1/schedules", json={**base, "bogus": 1}).status_code == 400
    floor = client.post(
        "/v1/schedules", json={**base, "cadence": {"kind": "interval", "every_minutes": 4}}
    )
    assert floor.status_code == 400
    assert floor.json()["error"]["code"] == "cadence_invalid"
    weekly = client.post(
        "/v1/schedules", json={**base, "cadence": {"kind": "weekly", "time": "09:00"}}
    )
    assert weekly.status_code == 400

    unknown_project = client.post("/v1/schedules", json={**base, "project_id": "proj_nope"})
    assert unknown_project.status_code == 404
    assert unknown_project.json()["error"]["code"] == "project_not_found"

    assert client.get("/v1/schedules/sched_nope").status_code == 404
    assert client.patch("/v1/schedules/sched_nope", json={}).status_code == 404
    assert client.delete("/v1/schedules/sched_nope").status_code == 404
    assert client.post("/v1/schedules/sched_nope/run").status_code == 404


def test_schedule_defaults_and_patch_recompute(client: TestClient) -> None:
    created = client.post(
        "/v1/schedules",
        json={
            "name": "weekly report",
            "prompt": "write the weekly report",
            "mode": "code",
            "cadence": {"kind": "weekly", "time": "09:00", "weekday": 0},
        },
    ).json()
    assert created["mode"] == "code"
    assert created["effort"] == "high"  # code-mode default effort
    assert created["cadence"] == {"kind": "weekly", "time": "09:00", "weekday": 0}
    next_before = created["next_run_at"]
    assert parse_iso_utc(next_before).astimezone().weekday() == 0

    patched = client.patch(
        f"/v1/schedules/{created['id']}",
        json={"cadence": {"kind": "interval", "every_minutes": 30}},
    ).json()
    assert patched["cadence"] == {"kind": "interval", "every_minutes": 30}
    assert patched["next_run_at"] != next_before  # recomputed from now
    delta = parse_iso_utc(patched["next_run_at"]) - datetime.now(UTC)
    assert timedelta(minutes=28) < delta <= timedelta(minutes=30)


def _wait(predicate, deadline_s: float = 10.0):
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("condition never became true")


def test_run_now_executes_archives_and_titles_on_mock(client: TestClient) -> None:
    project_id = client.post("/v1/projects", json={"name": "sched-proj"}).json()["id"]
    schedule = client.post(
        "/v1/schedules",
        json={
            "name": "digest",
            "prompt": "summarize what happened today",
            "effort": "low",
            "project_id": project_id,
            "cadence": {"kind": "daily", "time": "06:00"},
        },
    ).json()

    run = client.post(f"/v1/schedules/{schedule['id']}/run")
    assert run.status_code == 202
    session_id = run.json()["session_id"]

    def fetch_transcript():
        resp = client.get(f"/v1/memory/sessions/{session_id}")
        return resp.json() if resp.status_code == 200 else None

    # The run lands in the archive as a normal session, bound to the project.
    transcript = _wait(fetch_transcript)
    roles = [m["role"] for m in transcript["messages"]]
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert transcript["messages"][0]["content"] == "summarize what happened today"
    assert transcript["session"]["project_id"] == project_id

    def fetch_ran_schedule():
        body = client.get(f"/v1/schedules/{schedule['id']}").json()
        return body if body["last_run_at"] else None

    # Schedule bookkeeping caught up.
    updated = _wait(fetch_ran_schedule)
    assert updated["last_session_id"] == session_id
    assert updated["last_error"] is None

    # ... and the session gets auto-titled by the utility slot (mock).
    titled = _wait(
        lambda: client.get(f"/v1/memory/sessions/{session_id}").json()["session"]["title"] or None
    )
    assert titled == "Mock conversation title"
