"""Lazy / keep-alive model residency (api.md 2026-07-22c): the LoadController warms the
planned loadout on demand and unloads it after idle keep-alive — but never while busy.

Deterministic: the mock backend loads/unloads in-process, an injected clock drives the
idle window, and the reaper's `_reap_once` is exercised directly so nothing depends on
wall-clock timers.
"""

from __future__ import annotations

import asyncio

import pytest

from suiban.config import KvSettings
from suiban.kv import resolve_kv_state
from suiban.llama.load_controller import LoadController, parse_keep_alive
from suiban.llama.manager import LlamaManager
from suiban.sched.planner import Loadout, PlannedSlot

KV = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=True)


class FakeClock:
    """Monotonic-style clock the tests advance by hand."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def make_loadout() -> Loadout:
    slots = [
        PlannedSlot(
            slot_id="orchestrator",
            role="orchestrator",
            model="bonsai-27b",
            family="ternary",
            ctx=32768,
            gpu=0,
            port=8701,
            vram_mb=9000,
            mmproj=True,
        ),
        PlannedSlot(
            slot_id="worker-1",
            role="worker",
            model="bonsai-8b",
            family="ternary",
            ctx=16384,
            gpu=0,
            port=8702,
            vram_mb=4000,
        ),
    ]
    return Loadout(
        planned_at="2026-07-22T00:00:00Z",
        tier="24gb",
        slots=slots,
        headroom_mb=2000,
        family_configured="ternary",
        family_effective="ternary",
        family_degraded=False,
        family_reason=None,
    )


def make_controller(
    *,
    busy: dict,
    keep_alive: dict,
    clock: FakeClock,
) -> tuple[LoadController, LlamaManager]:
    manager = LlamaManager(make_loadout(), KV, compute_backend="cuda", use_mock=True)
    controller = LoadController(
        manager,
        is_busy=lambda: busy["v"],
        keep_alive=lambda: keep_alive["v"],
        clock=clock,
    )
    return controller, manager


# -- keep_alive parsing -------------------------------------------------------
def test_parse_keep_alive_stay_hot_tokens() -> None:
    for token in ("24/7", "0", "always", "ALWAYS", "-1", " 24/7 "):
        assert parse_keep_alive(token) is None
    assert parse_keep_alive(0) is None
    assert parse_keep_alive(-3) is None


def test_parse_keep_alive_minutes() -> None:
    assert parse_keep_alive("5") == 5.0
    assert parse_keep_alive(10) == 10.0
    assert parse_keep_alive("0.5") == 0.5


def test_parse_keep_alive_garbage_stays_hot() -> None:
    # Unparseable => never unload (keep serving beats thrashing on a typo).
    assert parse_keep_alive("soon") is None
    assert parse_keep_alive("") is None


# -- boot: nothing resident ---------------------------------------------------
def test_boot_holds_the_loadout_cold() -> None:
    clock = FakeClock()
    controller, manager = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)
    # No slots STARTED at boot: the manager registered none, and the planned slots read
    # "cold" (not "planned"/"ready").
    assert manager.slots == []
    assert controller.models_loaded is False
    assert controller.state == "cold"
    assert all(s.state == "cold" for s in manager.loadout.slots)


# -- ensure_loaded ------------------------------------------------------------
async def test_ensure_loaded_starts_slots_once() -> None:
    clock = FakeClock()
    controller, manager = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)
    cold_start = await controller.ensure_loaded()
    assert cold_start is True
    assert controller.models_loaded is True
    assert controller.state == "ready"
    assert manager.slot("orchestrator").state == "ready"
    assert manager.slot("worker-1").state == "ready"
    # Idempotent: a second call is a no-op (already resident) and reports no cold start.
    assert await controller.ensure_loaded() is False


async def test_ensure_loaded_stamps_activity() -> None:
    clock = FakeClock()
    controller, _ = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)
    await controller.ensure_loaded()
    clock.advance(3600)
    controller.touch()  # a later inference stamps activity
    # Reaper must measure idle from the touch, not from the load.
    assert await controller._reap_once() is False


async def test_concurrent_ensure_loaded_cold_start_runs_start_all_once() -> None:
    """A burst of concurrent ensure_loaded() on a cold controller warms the loadout
    exactly ONCE: the lock + double-check serialize the cold start, so start_all runs a
    single time and only one caller observes the cold-start transition."""
    clock = FakeClock()
    controller, manager = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)

    real_start_all = manager.start_all
    calls = {"n": 0}

    async def slow_start_all() -> None:
        calls["n"] += 1
        await asyncio.sleep(0.02)  # hold the lock so the burst piles up behind it
        await real_start_all()

    manager.start_all = slow_start_all  # type: ignore[method-assign]

    results = await asyncio.gather(*(controller.ensure_loaded() for _ in range(8)))

    assert calls["n"] == 1  # start_all fired exactly once despite 8 concurrent callers
    assert sum(1 for r in results if r is True) == 1  # one cold start, the rest no-ops
    assert controller.models_loaded is True
    assert controller.state == "ready"


# -- idle reaper --------------------------------------------------------------
async def test_reaper_unloads_after_keep_alive_when_idle() -> None:
    clock = FakeClock()
    busy = {"v": False}
    controller, manager = make_controller(busy=busy, keep_alive={"v": "5"}, clock=clock)
    await controller.ensure_loaded()

    # Inside the window: nothing unloads.
    clock.advance(4 * 60)
    assert await controller._reap_once() is False
    assert controller.models_loaded is True

    # Past the window and idle: the loadout unloads, freeing every slot.
    clock.advance(2 * 60)
    assert await controller._reap_once() is True
    assert controller.models_loaded is False
    assert controller.state == "cold"
    assert all(s.state == "cold" for s in manager.loadout.slots)


async def test_reaper_never_unloads_while_busy() -> None:
    clock = FakeClock()
    busy = {"v": True}  # an in-flight chat / running job
    controller, _ = make_controller(busy=busy, keep_alive={"v": "5"}, clock=clock)
    await controller.ensure_loaded()
    clock.advance(60 * 60)  # long past the window
    assert await controller._reap_once() is False
    assert controller.models_loaded is True  # never unload mid-generation


async def test_reaper_does_not_unload_while_a_research_job_is_active() -> None:
    """The residency-busy predicate counts a running deep-research job (jobs.active > 0)
    as busy, so the idle reaper never unloads mid-job even long past the keep-alive
    window; once the job finishes, the next pass may unload."""
    clock = FakeClock()
    jobs = {"active": 1}  # one running deep-research job, no chat in flight
    manager = LlamaManager(make_loadout(), KV, compute_backend="cuda", use_mock=True)
    controller = LoadController(
        manager,
        is_busy=lambda: jobs["active"] > 0,  # mirrors app.py _residency_busy
        keep_alive=lambda: "5",
        clock=clock,
    )
    await controller.ensure_loaded()

    clock.advance(60 * 60)  # far past the 5-minute idle window
    assert await controller._reap_once() is False  # the active job pins the loadout
    assert controller.models_loaded is True

    jobs["active"] = 0  # the job finishes
    assert await controller._reap_once() is True  # now the idle loadout unloads
    assert controller.models_loaded is False


async def test_keep_alive_24_7_never_unloads() -> None:
    clock = FakeClock()
    controller, _ = make_controller(busy={"v": False}, keep_alive={"v": "24/7"}, clock=clock)
    await controller.ensure_loaded()
    clock.advance(365 * 24 * 60 * 60)  # a year idle
    assert await controller._reap_once() is False
    assert controller.models_loaded is True


async def test_reaper_reloads_after_an_idle_unload() -> None:
    clock = FakeClock()
    controller, manager = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)
    await controller.ensure_loaded()
    clock.advance(10 * 60)
    assert await controller._reap_once() is True
    assert controller.models_loaded is False
    # A new request re-warms the same planned loadout.
    assert await controller.ensure_loaded() is True
    assert controller.models_loaded is True
    assert manager.slot("orchestrator").state == "ready"


async def test_keep_alive_is_read_live() -> None:
    clock = FakeClock()
    keep_alive = {"v": "24/7"}
    controller, _ = make_controller(busy={"v": False}, keep_alive=keep_alive, clock=clock)
    await controller.ensure_loaded()
    clock.advance(60 * 60)
    assert await controller._reap_once() is False  # stay-hot
    # Flip keep_alive live (as an applied settings change would): the reaper honors it
    # on the very next pass, no restart.
    keep_alive["v"] = "5"
    assert await controller._reap_once() is True
    assert controller.models_loaded is False


# -- _mark_cold preserves a genuinely-failed slot's honest state --------------
def test_mark_cold_preserves_failed_slots() -> None:
    """_mark_cold is defensive: it resets healthy planned slots to cold but leaves a
    genuinely-failed slot's state intact, so a launch failure never gets papered over as
    'cold'."""
    clock = FakeClock()
    controller, manager = make_controller(busy={"v": False}, keep_alive={"v": "5"}, clock=clock)
    manager.loadout.slots[0].state = "ready"
    manager.loadout.slots[1].state = "failed"
    controller._mark_cold()
    assert manager.loadout.slots[0].state == "cold"
    assert manager.loadout.slots[1].state == "failed"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
