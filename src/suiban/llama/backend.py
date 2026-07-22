"""Slot backends: the seam between suiban and llama-server processes.

RealBackend spawns/monitors an actual llama-server from the pinned fork; MockBackend
(SUIBAN_LLAMA_MOCK=1) serves the deterministic in-process responder over an ASGI
transport. Consumers never build URLs themselves — they ask a backend for a client.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import os
import re
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path

import httpx

from suiban.effort import slot_reasoning_budget
from suiban.kv import KvState
from suiban.llama.mock_server import build_mock_app
from suiban.sched.planner import PlannedSlot

logger = logging.getLogger(__name__)

MOCK_ENV_VAR = "SUIBAN_LLAMA_MOCK"

# Generous on purpose: a 27B cold load with a 16K KV allocation on a laptop GPU was
# observed to exceed 120 s — a timeout here strands a loading model as "failed" while
# its process holds VRAM.
HEALTH_TIMEOUT_S = 420.0
HEALTH_POLL_INTERVAL_S = 0.25
SHUTDOWN_GRACE_S = 10.0
MAX_RESTART_ATTEMPTS = 5

# llama-server stderr is kept in a bounded ring (was DEVNULL): the last lines are the
# only diagnostic when a slot dies, and the startup lines carry the KV-cache layout
# the hybrid-attention probe parses. ~200 lines is a full startup at -lv 4 (observed
# 212 on the pinned fork) without unbounded growth on chatty runs.
STDERR_RING_LINES = 200
STDERR_TAIL_LINES = 5  # lines quoted into slot-failure notices

# Hybrid-attention runtime probe: at -lv 4 the pinned fork (prism-b9596-9fcaed7)
# prints ONE KV-cache summary line per allocation. Observed live on this machine
# (1.7B, 28 attention layers, K=q8_0/V=tq4_0):
#   0.00.712.413 I llama_kv_cache: size =  182.00 MiB (  4096 cells,  28 layers,
#       4/1 seqs), K (q8_0):  119.00 MiB, V (tq4_0):   63.00 MiB
# The regex is deliberately tolerant (any prefix, flexible spacing) so log-format
# drift degrades to "count unknown", never to a crash.
KV_CACHE_LAYERS_RE = re.compile(r"llama_kv_cache:\s+size\s*=.*?,\s*(\d+)\s+layers")

# The 27B's hybrid attention allocates KV on 16 of its 64 layers; a full-64
# allocation means the fork silently lost the hybrid pattern (and ~4x the planned
# KV VRAM). Only the 27B has this contract; the small models are all-layer.
EXPECTED_KV_LAYERS = {"bonsai-27b": 16}


def mock_enabled() -> bool:
    return os.environ.get(MOCK_ENV_VAR) == "1"


def restart_backoff_s(attempt: int) -> float:
    """Exponential backoff for crash restarts: 1, 2, 4, 8, 16 (capped 30) seconds."""
    return float(min(2**attempt, 30))


def _linux_pdeathsig() -> None:
    """Child-side hook: die when the parent dies, even on SIGKILL.

    A hard-killed suiban must never leave llama-server orphans squatting on VRAM
    (observed in integration: two orphans held 6.3 GiB and OOM'd the next launch).
    Linux-only (prctl); on other platforms the graceful shutdown path still reaps.
    TODO(v1.1): macOS equivalent (kqueue parent watch)."""
    if sys.platform != "linux":
        return
    import ctypes
    import signal as _signal

    PR_SET_PDEATHSIG = 1
    # Best-effort: on failure the graceful shutdown path still reaps children.
    with contextlib.suppress(OSError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, _signal.SIGTERM)


def build_server_flags(
    slot: PlannedSlot,
    kv: KvState,
    *,
    model_path: Path,
    mmproj_path: Path | None = None,
    draft_model_path: Path | None = None,
) -> list[str]:
    """llama-server argv (without the binary itself) for one slot.

    Managed flags per plan: --jinja (ChatML tool templates), --mmproj (27B vision),
    --reasoning-budget, --cache-type-k/-v, -fa on (quantized KV), -md (DSpark opt-in),
    -ngl, -c, --port, --host 127.0.0.1.
    """
    flags = [
        "-m",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(slot.port),
        "-c",
        str(slot.ctx),
        "-ngl",
        "999" if slot.gpu is not None else "0",
        "--jinja",
        # Slot-wide thinking ceiling (xhigh budget bounded by 40% of ctx). Per-request
        # control is on/off only via chat_template_kwargs.enable_thinking — the fork
        # ignores per-request budget fields (verified live; see effort.py).
        "--reasoning-budget",
        str(slot_reasoning_budget(slot.ctx)),
        "--cache-type-k",
        kv.k_type,
        "--cache-type-v",
        kv.v_type,
        # Log verbosity 4: the INFO-level `llama_kv_cache: size = ... N layers` line
        # (which the hybrid-attention runtime probe parses) is filtered out at the
        # fork's default verbosity 3. Verified live; DEBUG per-layer lines need 5.
        "-lv",
        "4",
    ]
    if kv.k_type != "f16" or kv.v_type != "f16":
        # Quantized KV hard-requires flash attention. Verified against the pinned fork
        # (tag prism-b9596-9fcaed7, common/arg.cpp): `-fa`/`--flash-attn` takes a
        # REQUIRED value in {on, off, auto} — bare `-fa` would swallow the next argv
        # token. Force "on"; f16/f16 loadouts omit the flag (fork default: auto).
        flags += ["-fa", "on"]
    if slot.mmproj:
        if mmproj_path is None:
            raise ValueError(f"slot {slot.slot_id} wants mmproj but no mmproj_path given")
        flags += ["--mmproj", str(mmproj_path)]
    if slot.dspark:
        if draft_model_path is None:
            raise ValueError(f"slot {slot.slot_id} wants dspark but no draft model given")
        flags += ["-md", str(draft_model_path)]
    return flags


class SlotBackend(abc.ABC):
    """Lifecycle + client access for one slot."""

    def __init__(self, slot: PlannedSlot) -> None:
        self.slot = slot

    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def stop(self) -> None: ...

    @abc.abstractmethod
    def client(self) -> httpx.AsyncClient:
        """OpenAI-compatible HTTP client bound to this slot."""

    async def healthy(self) -> bool:
        try:
            async with self.client() as client:
                resp = await client.get("/health", timeout=5.0)
                return resp.status_code == 200
        except httpx.HTTPError:
            return False


class MockBackend(SlotBackend):
    """In-process deterministic responder; no subprocess, no network."""

    def __init__(self, slot: PlannedSlot) -> None:
        super().__init__(slot)
        self._app = build_mock_app(slot_model=slot.model)
        self._started = False

    async def start(self) -> None:
        self._started = True
        self.slot.state = "ready"

    async def stop(self) -> None:
        self._started = False
        self.slot.state = "stopped"

    def client(self) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=self._app)
        return httpx.AsyncClient(transport=transport, base_url="http://suiban-mock")


ProcessFactory = Callable[[list[str]], "asyncio.subprocess.Process"]


class RealBackend(SlotBackend):
    """Spawns llama-server, polls /health, restarts on crash with backoff.

    stderr is drained into a bounded ring buffer (STDERR_RING_LINES) instead of
    DEVNULL: `stderr_tail()` feeds slot-failure notices and `kv_layer_count()` powers
    the hybrid-attention runtime probe. `on_giveup` (optional) is called with a
    one-line reason when the crash-restart loop exhausts its attempts, so the manager
    can surface a notice for a slot that died long after startup."""

    def __init__(
        self,
        slot: PlannedSlot,
        *,
        binary: Path,
        flags: list[str],
        env: dict[str, str] | None = None,
        on_giveup: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(slot)
        self._binary = binary
        self._flags = flags
        self._env = env
        self._on_giveup = on_giveup
        self.process: asyncio.subprocess.Process | None = None
        self._watcher: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_ring: deque[str] = deque(maxlen=STDERR_RING_LINES)
        self._stopping = False
        self._restart_attempts = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.slot.port}"

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url)

    async def _spawn(self) -> asyncio.subprocess.Process:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)
        # The fork's shared libs (libggml*, libllama*) sit next to llama-server in
        # ~/.bonsai/bin/<backend>/ — the loader needs that directory on its path.
        lib_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        bin_parent = str(Path(self._binary).parent)
        env[lib_var] = bin_parent + (os.pathsep + env[lib_var] if env.get(lib_var) else "")
        if self.slot.gpu is not None:
            # Pin the process to its planned GPU.
            env["CUDA_VISIBLE_DEVICES"] = str(self.slot.gpu)
        process = await asyncio.create_subprocess_exec(
            str(self._binary),
            *self._flags,
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=_linux_pdeathsig,
        )
        # One drain task per spawn; it ends at pipe EOF when the process dies. The
        # ring survives restarts (marked below) so pre-crash lines stay diagnosable.
        self._stderr_task = asyncio.create_task(self._drain_stderr(process.stderr))
        return process

    async def _drain_stderr(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None:  # pragma: no cover - PIPE always yields a reader
            return
        while True:
            try:
                line = await stream.readline()
            except (ValueError, OSError):  # over-long line or a torn-down transport
                continue
            if not line:
                return
            self._stderr_ring.append(line.decode("utf-8", errors="replace").rstrip())

    def stderr_tail(self, n: int = STDERR_TAIL_LINES) -> list[str]:
        """The last n stderr lines — the only diagnostic when llama-server dies."""
        return list(self._stderr_ring)[-n:]

    def kv_layer_count(self) -> int | None:
        """Layers the KV cache was allocated on, parsed from startup stderr — the
        LAST summary line wins (restarts re-allocate). None when no line matched
        (format drift or a not-yet-drained pipe): the probe degrades to 'unknown',
        it never guesses and never crashes."""
        for line in reversed(self._stderr_ring):
            match = KV_CACHE_LAYERS_RE.search(line)
            if match is not None:
                return int(match.group(1))
        return None

    async def _wait_healthy(self, timeout: float = HEALTH_TIMEOUT_S) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            if self.process is not None and self.process.returncode is not None:
                return False  # died during startup
            if await self.healthy():
                return True
            await asyncio.sleep(HEALTH_POLL_INTERVAL_S)
        return False

    async def start(self) -> None:
        self._stopping = False
        self.slot.state = "starting"
        self.process = await self._spawn()
        if await self._wait_healthy():
            self.slot.state = "ready"
            self._restart_attempts = 0
            self._watcher = asyncio.create_task(self._watch())
        else:
            self.slot.state = "failed"
            await self._kill()

    def _give_up(self, reason: str) -> None:
        tail = self.stderr_tail()
        if tail:
            reason += " Last stderr: " + " | ".join(tail)
        logger.error("slot %s: %s", self.slot.slot_id, reason)
        self.slot.state = "failed"
        if self._on_giveup is not None:
            self._on_giveup(f"slot {self.slot.slot_id} ({self.slot.model}): {reason}")

    async def _watch(self) -> None:
        """Restart-on-crash with exponential backoff; gives up after MAX attempts."""
        while True:
            assert self.process is not None
            await self.process.wait()
            if self._stopping:
                return
            if self._restart_attempts >= MAX_RESTART_ATTEMPTS:
                self._give_up(f"llama-server crashed {self._restart_attempts} times; giving up.")
                return
            delay = restart_backoff_s(self._restart_attempts)
            self._restart_attempts += 1
            logger.warning(
                "slot %s: llama-server exited unexpectedly; restart %d/%d in %.0fs",
                self.slot.slot_id,
                self._restart_attempts,
                MAX_RESTART_ATTEMPTS,
                delay,
            )
            self.slot.state = "starting"
            await asyncio.sleep(delay)
            self._stderr_ring.append(
                f"--- suiban: restart {self._restart_attempts}/{MAX_RESTART_ATTEMPTS} ---"
            )
            self.process = await self._spawn()
            if await self._wait_healthy():
                self.slot.state = "ready"
            else:
                await self._kill()
                self._give_up("llama-server did not become healthy after a crash restart.")
                return

    async def stop(self) -> None:
        """Graceful shutdown: SIGTERM, wait, then SIGKILL."""
        self._stopping = True
        if self._watcher is not None:
            self._watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watcher
            self._watcher = None
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=SHUTDOWN_GRACE_S)
            except TimeoutError:
                await self._kill()
        if self._stderr_task is not None:
            # The pipe hits EOF once the process is dead; bounded, but never hang
            # shutdown on a wedged transport.
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(self._stderr_task, timeout=2.0)
            self._stderr_task = None
        self.slot.state = "stopped"

    async def _kill(self) -> None:
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            with contextlib.suppress(ProcessLookupError):
                await self.process.wait()
