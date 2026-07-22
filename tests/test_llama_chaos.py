"""Chaos tests: a stub llama-server executable (a tiny python script) exercises
RealBackend's REAL process management — spawn, health polling, stderr ring, crash
restart with backoff, give-up, SIGKILL escalation, shutdown — with no GPU, no model
weights, and no fork binary. The stub serves /health, mimics the pinned fork's
startup stderr (including the `llama_kv_cache: size = ... N layers` line the
hybrid-attention probe parses), and dies on command (the tests kill its process).
"""

from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path

import pytest

from suiban.config import KvSettings
from suiban.kv import resolve_kv_state
from suiban.llama.backend import RealBackend
from suiban.llama.manager import LlamaManager, LlamaSlot
from suiban.sched.planner import Loadout, PlannedSlot

KV = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=True)

# The stub prints stderr in the fork's observed format; --kv-layers controls the
# summary line so both the match and mismatch probe paths are exercised.
STUB_SOURCE = '''#!/usr/bin/env python3
"""Stub llama-server for suiban chaos tests (tests/test_llama_chaos.py)."""
import json
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


port = int(arg("--port", "0"))
kv_layers = arg("--kv-layers")

if "--exit-immediately" in sys.argv:
    print("0.00.050.000 E load_model: stub dying at startup (scripted)",
          file=sys.stderr, flush=True)
    sys.exit(7)
if "--ignore-sigterm" in sys.argv:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

print("0.00.100.000 I srv  llama_server: loading model", file=sys.stderr, flush=True)
if kv_layers is not None:
    print("0.00.200.000 I llama_kv_cache: size =  182.00 MiB (  4096 cells,  "
          + kv_layers
          + " layers,  4/1 seqs), K (q8_0):  119.00 MiB, V (tq4_0):   63.00 MiB",
          file=sys.stderr, flush=True)
print("0.00.300.000 I srv  llama_server: server is listening", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
'''


@pytest.fixture
def stub_binary(tmp_path: Path) -> Path:
    path = tmp_path / "stub-llama-server"
    path.write_text(STUB_SOURCE, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def fast_timings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backoff/grace read the module globals at call time — shrink them so chaos
    scenarios run in milliseconds, not the production 1-30 s ladder."""
    monkeypatch.setattr("suiban.llama.backend.restart_backoff_s", lambda attempt: 0.01)
    monkeypatch.setattr("suiban.llama.backend.SHUTDOWN_GRACE_S", 0.3)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def make_planned(port: int, model: str = "bonsai-8b") -> PlannedSlot:
    return PlannedSlot(
        slot_id="worker-1",
        role="worker",
        model=model,
        family="ternary",
        ctx=8192,
        gpu=None,  # no CUDA_VISIBLE_DEVICES pinning for the stub
        port=port,
        vram_mb=0,
    )


def make_backend(stub: Path, planned: PlannedSlot, *extra_flags: str, **kwargs) -> RealBackend:
    flags = ["--port", str(planned.port), *extra_flags]
    return RealBackend(planned, binary=stub, flags=flags, **kwargs)


async def wait_for_state(planned: PlannedSlot, state: str, deadline_s: float = 10.0) -> None:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if planned.state == state:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"slot never reached {state!r} (state: {planned.state})")


# -- spawn / health / stderr ring / shutdown ----------------------------------
async def test_spawn_health_stderr_ring_and_graceful_stop(stub_binary: Path) -> None:
    planned = make_planned(free_port())
    backend = make_backend(stub_binary, planned, "--kv-layers", "28")
    await backend.start()
    try:
        assert planned.state == "ready"
        assert await backend.healthy()
        # stderr went to the ring, not DEVNULL; the KV summary line parses.
        await asyncio.sleep(0.05)  # let the drain task catch up
        tail = backend.stderr_tail(10)
        assert any("loading model" in line for line in tail)
        assert backend.kv_layer_count() == 28
    finally:
        await backend.stop()
    assert planned.state == "stopped"
    assert backend.process is not None
    assert backend.process.returncode is not None  # actually reaped


async def test_startup_death_is_failed_with_stderr_diagnostic(stub_binary: Path) -> None:
    planned = make_planned(free_port())
    backend = make_backend(stub_binary, planned, "--exit-immediately")
    await backend.start()
    assert planned.state == "failed"
    await asyncio.sleep(0.05)
    assert any("stub dying at startup" in line for line in backend.stderr_tail())


# -- crash restart / backoff / give-up ----------------------------------------
async def test_crash_restart_recovers_with_marker(stub_binary: Path) -> None:
    planned = make_planned(free_port())
    backend = make_backend(stub_binary, planned, "--kv-layers", "28")
    await backend.start()
    try:
        assert planned.state == "ready"
        first_pid = backend.process.pid
        backend.process.kill()  # die on command
        # Wait for the watcher to respawn (new pid), then for health to pass —
        # polling "ready" alone could race the still-ready pre-crash state.
        end = time.monotonic() + 10
        while time.monotonic() < end and backend.process.pid == first_pid:
            await asyncio.sleep(0.02)
        assert backend.process.pid != first_pid  # a real respawn, same port
        await wait_for_state(planned, "ready")
        assert await backend.healthy()
        assert any("suiban: restart 1/" in line for line in backend.stderr_tail(30))
    finally:
        await backend.stop()


async def test_repeated_crashes_give_up_with_notice(
    stub_binary: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("suiban.llama.backend.MAX_RESTART_ATTEMPTS", 0)
    giveups: list[str] = []
    planned = make_planned(free_port())
    backend = make_backend(stub_binary, planned, on_giveup=giveups.append)
    await backend.start()
    try:
        assert planned.state == "ready"
        backend.process.kill()
        await wait_for_state(planned, "failed")
        assert len(giveups) == 1
        assert "giving up" in giveups[0]
        assert "worker-1" in giveups[0]
        assert "Last stderr:" in giveups[0]  # the ring feeds the notice
    finally:
        await backend.stop()


async def test_sigterm_ignoring_server_is_sigkilled(stub_binary: Path) -> None:
    planned = make_planned(free_port())
    backend = make_backend(stub_binary, planned, "--ignore-sigterm")
    await backend.start()
    assert planned.state == "ready"
    start = time.monotonic()
    await backend.stop()
    assert planned.state == "stopped"
    assert backend.process.returncode == -9  # grace expired -> SIGKILL
    assert time.monotonic() - start < 5


# -- hybrid-attention runtime probe (manager notice) --------------------------
def make_manager() -> LlamaManager:
    loadout = Loadout(
        planned_at="2026-07-21T00:00:00Z",
        tier="8gb",
        slots=[],
        headroom_mb=0,
        family_configured="ternary",
        family_effective="ternary",
        family_degraded=False,
        family_reason=None,
    )
    return LlamaManager(loadout, KV, compute_backend="cpu", use_mock=False)


async def run_probe(stub_binary: Path, *extra_flags: str) -> LlamaManager:
    planned = make_planned(free_port(), model="bonsai-27b")
    backend = make_backend(stub_binary, planned, *extra_flags)
    slot = LlamaSlot(
        slot_id=planned.slot_id,
        role=planned.role,
        model=planned.model,
        port=planned.port,
        gpu=planned.gpu,
        planned=planned,
        backend=backend,
    )
    manager = make_manager()
    await backend.start()
    try:
        assert planned.state == "ready"
        await manager._check_kv_layers(slot)
    finally:
        await backend.stop()
    return manager


async def test_kv_probe_matching_layers_no_notice(stub_binary: Path) -> None:
    manager = await run_probe(stub_binary, "--kv-layers", "16")  # the 27B contract
    assert manager.notices == []


async def test_kv_probe_mismatch_emits_notice(stub_binary: Path) -> None:
    manager = await run_probe(stub_binary, "--kv-layers", "64")  # hybrid pattern lost
    codes = [n.code for n in manager.notices]
    assert codes == ["kv_layers_mismatch"]
    message = manager.notices[0].message
    assert "64" in message and "16" in message


async def test_kv_probe_unparseable_log_stays_silent(stub_binary: Path) -> None:
    manager = await run_probe(stub_binary)  # stub prints no KV summary line
    assert manager.notices == []
