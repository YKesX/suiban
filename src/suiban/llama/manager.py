"""Slot lifecycle manager: builds backends for a planned loadout, starts/stops them,
hands out per-slot OpenAI-compatible clients, and serializes access per slot
(SlotGate) — llama-server decodes one request at a time per slot, so suiban queues
fairly instead of letting concurrent runs time each other out blindly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import httpx

from suiban.errors import BonsaiError
from suiban.installer import models as model_store
from suiban.kv import KvState
from suiban.llama import backend as backend_mod
from suiban.llama.backend import (
    EXPECTED_KV_LAYERS,
    MockBackend,
    RealBackend,
    SlotBackend,
    build_server_flags,
)
from suiban.llama.binary import resolve_server_binary
from suiban.sched.budget import BudgetProvider
from suiban.sched.planner import Loadout, Notice, PlannedSlot

logger = logging.getLogger(__name__)

# How long the manager waits for the async stderr drain to catch up before the
# hybrid-attention probe reads the ring (one retry — the KV summary line is printed
# well before /health goes 200, so this is belt-and-braces, not a poll loop).
_KV_PROBE_RETRY_S = 0.1


class SlotGate:
    """Serializes runs on one slot with a bounded wait queue.

    A slot's llama-server effectively decodes one request at a time; unserialized
    concurrent chats would silently share tokens until the 300 s step timeout kills
    one of them. The gate makes the contention explicit: one holder, at most
    MAX_QUEUE waiters, and a 429 `overloaded_error` (api.md error table) beyond that.
    Interactive chats hold the gate for a whole run; research holds it per pipeline
    STEP (see research/wiring.py) so chats interleave with a running job.
    """

    MAX_QUEUE = 4

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._waiting = 0

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    @property
    def queue_depth(self) -> int:
        """Runs ahead of a new arrival: the holder plus everyone already waiting."""
        return (1 if self._lock.locked() else 0) + self._waiting

    def check_capacity(self, slot_id: str) -> None:
        """429 when the wait queue is full. Called by the CHAT entry point before
        joining; internal waiters (research steps, scheduled runs) skip the check —
        they wait however long it takes, but still count toward the depth."""
        if self._lock.locked() and self._waiting >= self.MAX_QUEUE:
            raise BonsaiError(
                429,
                f"slot {slot_id} is running a request with {self._waiting} more "
                "queued (the queue is full); retry shortly",
                code="slot_queue_full",
            )

    async def acquire(self) -> None:
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1

    def release(self) -> None:
        self._lock.release()

    @contextlib.asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        await self.acquire()
        try:
            yield
        finally:
            self.release()


@dataclass
class LlamaSlot:
    """A managed llama-server slot: plan + live backend/process."""

    slot_id: str
    role: str
    model: str
    port: int
    gpu: int | None
    planned: PlannedSlot
    backend: SlotBackend
    process: asyncio.subprocess.Process | None = field(default=None)
    gate: SlotGate = field(default_factory=SlotGate)

    @property
    def state(self) -> str:
        return self.planned.state


class LlamaManager:
    def __init__(
        self,
        loadout: Loadout,
        kv: KvState,
        *,
        compute_backend: str = "cpu",
        use_mock: bool | None = None,
        budget: BudgetProvider | None = None,
        used_vram_mb: Callable[[int], int | None] | None = None,
    ) -> None:
        self._loadout = loadout
        self._kv = kv
        self._compute_backend = compute_backend
        self._use_mock = backend_mod.mock_enabled() if use_mock is None else use_mock
        self._slots: dict[str, LlamaSlot] = {}
        self.notices: list[Notice] = []
        # First-launch VRAM measurement: with both hooks present, real slot launches
        # are bracketed by used-VRAM snapshots and the machine-dependent buffer cost
        # (delta minus exact weights/KV math) is persisted to ~/.bonsai/budget.json.
        self._budget = budget
        self._used_vram_mb = used_vram_mb

    # -- construction ------------------------------------------------------
    def _on_slot_giveup(self, message: str) -> None:
        """RealBackend hook: a slot whose crash-restart loop gave up (possibly hours
        after startup) still surfaces in /v1/system.notices.

        AUDIT SEAM (security pass, next session): messages here and in start_all
        quote llama-server stderr tails, which can contain local absolute paths
        (model files under the bonsai home). They surface only via the loopback-only
        /v1/system API today; revisit if non-loopback binds gain auth in v1.1."""
        self.notices.append(Notice("warn", "slot_failed", message))

    def _make_backend(self, planned: PlannedSlot) -> SlotBackend:
        if self._use_mock:
            return MockBackend(planned)
        binary = resolve_server_binary(self._compute_backend)
        model_path = model_store.resolve_model_path(planned.model, planned.family)
        mmproj_path = model_store.resolve_mmproj_path(planned.family) if planned.mmproj else None
        draft_path = model_store.resolve_dspark_path(planned.family) if planned.dspark else None
        flags = build_server_flags(
            planned,
            self._kv,
            model_path=model_path,
            mmproj_path=mmproj_path,
            draft_model_path=draft_path,
        )
        return RealBackend(planned, binary=binary, flags=flags, on_giveup=self._on_slot_giveup)

    # -- lifecycle ---------------------------------------------------------
    async def start_all(self) -> None:
        """Start every planned slot. Failure of one slot never crashes the server —
        it surfaces as slot state 'failed' plus a notice."""
        for planned in self._loadout.slots:
            try:
                slot_backend = self._make_backend(planned)
            except BonsaiError as exc:
                planned.state = "failed"
                self.notices.append(Notice("warn", exc.code or "slot_failed", exc.message))
                logger.warning("slot %s cannot start: %s", planned.slot_id, exc.message)
                # Still register the slot so /v1/system reports it honestly.
                self._slots[planned.slot_id] = LlamaSlot(
                    slot_id=planned.slot_id,
                    role=planned.role,
                    model=planned.model,
                    port=planned.port,
                    gpu=planned.gpu,
                    planned=planned,
                    backend=MockBackend(planned) if self._use_mock else _NullBackend(planned),
                )
                continue
            slot = LlamaSlot(
                slot_id=planned.slot_id,
                role=planned.role,
                model=planned.model,
                port=planned.port,
                gpu=planned.gpu,
                planned=planned,
                backend=slot_backend,
            )
            self._slots[planned.slot_id] = slot
            measure = (
                isinstance(slot_backend, RealBackend)
                and planned.gpu is not None
                and self._budget is not None
                and self._used_vram_mb is not None
            )
            before_mb = self._used_vram_mb(planned.gpu) if measure else None
            await slot_backend.start()
            if isinstance(slot_backend, RealBackend):
                slot.process = slot_backend.process
            if measure and planned.state == "ready" and before_mb is not None:
                after_mb = self._used_vram_mb(planned.gpu)
                if after_mb is not None and after_mb > before_mb:
                    self._record_measurement(planned, after_mb - before_mb)
            if planned.state == "ready":
                await self._check_kv_layers(slot)
            else:
                message = f"slot {planned.slot_id} ({planned.model}) failed to become healthy."
                if isinstance(slot_backend, RealBackend):
                    tail = slot_backend.stderr_tail()
                    if tail:
                        message += " Last stderr: " + " | ".join(tail)
                self.notices.append(Notice("warn", "slot_failed", message))

    async def _check_kv_layers(self, slot: LlamaSlot) -> None:
        """Hybrid-attention runtime probe: the 27B must allocate KV on 16 of its 64
        layers (the whole VRAM plan assumes it). Parsed from llama-server startup
        stderr (backend.kv_layer_count); a mismatch is a `kv_layers_mismatch` notice,
        an unparseable log is a debug entry — never a crash either way."""
        backend = slot.backend
        if not isinstance(backend, RealBackend):
            return
        expected = EXPECTED_KV_LAYERS.get(slot.model)
        if expected is None:
            return
        observed = backend.kv_layer_count()
        if observed is None:
            # The async stderr drain may still be catching up right after /health
            # went 200; give it one beat and re-read.
            await asyncio.sleep(_KV_PROBE_RETRY_S)
            observed = backend.kv_layer_count()
        if observed is None:
            logger.debug(
                "slot %s: no llama_kv_cache summary line found in startup stderr; "
                "hybrid-attention probe skipped",
                slot.slot_id,
            )
            return
        if observed != expected:
            self.notices.append(
                Notice(
                    "warn",
                    "kv_layers_mismatch",
                    f"slot {slot.slot_id} ({slot.model}): llama-server allocated KV "
                    f"cache on {observed} layers, expected {expected} (hybrid "
                    "attention). VRAM use may exceed the planned budget; check the "
                    "pinned fork build.",
                )
            )

    def _record_measurement(self, planned: PlannedSlot, delta_mb: int) -> None:
        """Persist the machine-dependent buffer cost from a real launch's VRAM delta.

        Weights and KV are exact math (file bytes, block layouts); the compute/graph
        buffers are the only genuinely unknown term, so buffers = delta - the exact
        parts. Deltas smaller than 80% of the weights mean something else touched the
        GPU mid-launch — those measurements are discarded as unreliable."""
        assert self._budget is not None
        cost = self._budget.slot_cost(
            planned.model,
            planned.family,
            planned.ctx,
            k_type=self._kv.k_type,
            v_type=self._kv.v_type,
            with_mmproj=planned.mmproj,
            with_dspark=planned.dspark,
        )
        if delta_mb < int(cost.weights_mb * 0.8):
            logger.warning(
                "slot %s: VRAM delta %d MiB below weights prior %d MiB — discarding measurement",
                planned.slot_id,
                delta_mb,
                cost.weights_mb,
            )
            return
        buffers_mb = max(64, delta_mb - cost.weights_mb - cost.kv_mb - cost.extras_mb)
        self._budget.record_measured(planned.model, planned.family, buffers_mb=buffers_mb)
        logger.info(
            "slot %s measured: %d MiB total -> buffers_mb=%d persisted",
            planned.slot_id,
            delta_mb,
            buffers_mb,
        )

    async def shutdown(self) -> None:
        await asyncio.gather(
            *(slot.backend.stop() for slot in self._slots.values()), return_exceptions=True
        )

    # -- access ------------------------------------------------------------
    @property
    def loadout(self) -> Loadout:
        """The planned loadout this manager serves (fixed at run start). The residency
        controller reads its planned slots to reflect a cold/unloaded state honestly."""
        return self._loadout

    @property
    def slots(self) -> list[LlamaSlot]:
        return list(self._slots.values())

    def slot(self, slot_id: str) -> LlamaSlot | None:
        return self._slots.get(slot_id)

    def client_for(self, slot_id: str) -> httpx.AsyncClient:
        slot = self._slots.get(slot_id)
        if slot is None:
            raise BonsaiError(404, f"no such slot: {slot_id}", code="slot_not_found")
        return slot.backend.client()

    @property
    def slots_ready(self) -> int:
        return sum(1 for s in self._slots.values() if s.state == "ready")

    @property
    def slots_total(self) -> int:
        return len(self._loadout.slots)

    @property
    def family_download_active(self) -> bool:
        """Hard-block condition for /v1/system/apply (409 on family switch while its
        download runs). TODO(v1.1): wire to the installer's download job state once
        downloads become long-running server-side jobs; Stage 1 installs are CLI-only.
        """
        return False


class _NullBackend(SlotBackend):
    """Placeholder backend for slots that could not be constructed (missing binary or
    weights): keeps the slot visible in /v1/system without pretending it can serve."""

    async def start(self) -> None:  # pragma: no cover - never started
        self.slot.state = "failed"

    async def stop(self) -> None:
        return None

    def client(self) -> httpx.AsyncClient:
        raise BonsaiError(
            409,
            f"slot {self.slot.slot_id} is not available (missing binary or weights)",
            code="slot_unavailable",
        )
