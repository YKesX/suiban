"""Lazy / keep-alive model residency (api.md 2026-07-22c, ollama-style).

`serve` plans the loadout at boot but starts NO slots. The LoadController owns residency
from there: `ensure_loaded()` starts the planned slots on demand (at the top of every
inference path, before routing) and stamps `last_activity`; a background reaper unloads
the whole loadout — freeing VRAM — after `runtime.keep_alive` idle minutes, but NEVER
mid-generation (an in-flight chat or a running research job counts as busy).

The controller is the single owner of the loaded/loading/unloading state, so
`GET /v1/system` can report `runtime: { keep_alive, models_loaded, state }` honestly and
a not-yet-loaded slot reads "cold" rather than pretending to be ready.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

from suiban.llama.manager import LlamaManager

logger = logging.getLogger(__name__)

# How often the idle reaper wakes to consider unloading. Short enough that a 5-minute
# keep-alive unloads promptly, cheap enough to run forever; tests inject a tiny value.
REAPER_INTERVAL_S = 30.0

# Tokens that mean "stay resident forever" (case-insensitive).
_NEVER_UNLOAD_TOKENS = frozenset({"24/7", "0", "always", "-1"})

# State the controller reports to /v1/system. "cold" = nothing resident (boot, or after
# an idle unload); "loading" = warming the loadout; "ready" = resident; "idle_unloading"
# = releasing the loadout right now.
COLD = "cold"
LOADING = "loading"
READY = "ready"
IDLE_UNLOADING = "idle_unloading"


def parse_keep_alive(value: str | int) -> float | None:
    """Idle minutes before an unload, or None to stay hot forever.

    "24/7"/"0"/"always"/"-1" (any case) and any value <= 0 mean never unload; a positive
    minutes integer, or a string of minutes, is that many minutes. Anything unparseable
    also means never unload — the safe direction: keep serving rather than thrash the
    loadout on a typo (a notice-worthy misconfig, never a crash)."""
    token = str(value).strip().lower()
    if token in _NEVER_UNLOAD_TOKENS:
        return None
    try:
        minutes = float(token)
    except ValueError:
        return None
    return minutes if minutes > 0 else None


class LoadController:
    """Owns model residency for one running server. Concurrency-safe: `ensure_loaded`
    and the reaper serialize through one lock, so a cold-start burst warms the loadout
    exactly once and an unload never races a warm-up."""

    def __init__(
        self,
        manager: LlamaManager,
        *,
        is_busy: Callable[[], bool],
        keep_alive: Callable[[], str | int],
        clock: Callable[[], float] = time.monotonic,
        reaper_interval_s: float = REAPER_INTERVAL_S,
    ) -> None:
        self._manager = manager
        self._is_busy = is_busy
        self._keep_alive = keep_alive
        self._clock = clock
        self._reaper_interval_s = reaper_interval_s
        self._lock = asyncio.Lock()
        self._loaded = False
        self._loading = False
        self._unloading = False
        self._last_activity = clock()
        self._reaper_task: asyncio.Task | None = None
        # Boot with the planned slots reading "cold": the loadout is planned, nothing is
        # resident yet (they flip to ready/failed on the first ensure_loaded).
        self._mark_cold()

    # -- reported state ----------------------------------------------------
    @property
    def models_loaded(self) -> bool:
        return self._loaded

    @property
    def state(self) -> str:
        if self._unloading:
            return IDLE_UNLOADING
        if self._loading:
            return LOADING
        if self._loaded:
            return READY
        return COLD

    def _mark_cold(self) -> None:
        """Reflect an unloaded loadout on the planned slots. Only slots that are NOT in a
        failed state are reset — a slot that failed to launch keeps its honest 'failed'
        so /v1/system still shows the failure."""
        for slot in self._manager.loadout.slots:
            if slot.state != "failed":
                slot.state = COLD

    # -- residency ---------------------------------------------------------
    def touch(self) -> None:
        """Reset the idle timer. Called at the top of every inference and on the
        activity-idle transition, so keep-alive counts idle time, not wall-clock."""
        self._last_activity = self._clock()

    async def ensure_loaded(self) -> bool:
        """Start the planned slots if not resident; stamp activity. Returns True when a
        cold start actually happened (the caller surfaces a `warming_up` notice on rich
        streams), False when the loadout was already resident. Idempotent and
        concurrency-safe; waits out any in-progress unload, then reloads."""
        if self._loaded and not self._unloading:
            self.touch()
            return False
        async with self._lock:
            if self._loaded and not self._unloading:
                self.touch()
                return False
            self._loading = True
            try:
                logger.info("residency: warming the planned loadout (cold start)")
                await self._manager.start_all()
                self._loaded = True
                self.touch()
                return True
            finally:
                self._loading = False

    async def _reap_once(self) -> bool:
        """One reaper pass: unload the loadout iff keep-alive has a finite window, the
        loadout is resident and settled, the system is idle (no in-flight chat/job), and
        it has been idle past the window. Returns True when it unloaded."""
        idle_minutes = parse_keep_alive(self._keep_alive())
        if idle_minutes is None:  # stay hot forever
            return False
        if not self._loaded or self._loading or self._unloading:
            return False
        if self._is_busy():
            return False
        if self._clock() - self._last_activity < idle_minutes * 60:
            return False
        async with self._lock:
            # Re-check under the lock: a request may have loaded/touched, or made the
            # system busy, between the fast pre-check and acquiring the lock. NEVER
            # unload mid-generation — the busy check is the guard.
            if not self._loaded or self._is_busy():
                return False
            if self._clock() - self._last_activity < idle_minutes * 60:
                return False
            self._unloading = True
            logger.info("residency: unloading the idle loadout (keep_alive elapsed)")
            try:
                await self._manager.shutdown()
                self._loaded = False
                self._mark_cold()
            finally:
                self._unloading = False
        return True

    # -- reaper lifecycle --------------------------------------------------
    def start(self) -> None:
        """Launch the background idle reaper (with the app lifespan)."""
        if self._reaper_task is None:
            self._reaper_task = asyncio.create_task(self._reaper_loop())

    async def stop(self) -> None:
        """Cancel the reaper. The loadout itself is torn down by manager.shutdown() in
        the lifespan finally block (harmless whether or not it was resident)."""
        if self._reaper_task is not None:
            self._reaper_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reaper_task
            self._reaper_task = None

    async def _reaper_loop(self) -> None:
        while True:
            await asyncio.sleep(self._reaper_interval_s)
            try:
                await self._reap_once()
            except Exception:  # noqa: BLE001 - a reaper hiccup must never kill the server
                logger.exception("residency: idle-reaper pass failed")
