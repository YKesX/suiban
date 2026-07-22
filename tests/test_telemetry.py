"""Telemetry provider selection, the CPU-only (ram) fallback, and the real
NVML / rocm-smi parsing paths against mocked drivers."""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

from suiban.sched.telemetry import (
    NvmlProvider,
    RamProvider,
    RocmSmiProvider,
    TelemetrySnapshot,
    pick_provider,
)


class FailingProvider:
    name = "nvml"

    def probe(self) -> bool:
        return False

    def snapshot(self) -> TelemetrySnapshot:  # pragma: no cover
        raise AssertionError("must not be called")


class ExplodingProvider:
    name = "rocm-smi"

    def probe(self) -> bool:
        raise RuntimeError("driver soup")

    def snapshot(self) -> TelemetrySnapshot:  # pragma: no cover
        raise AssertionError("must not be called")


def test_fallback_to_ram_when_gpu_providers_fail() -> None:
    provider = pick_provider((FailingProvider(), ExplodingProvider(), RamProvider()))
    assert provider.name == "ram"
    snapshot = provider.snapshot()
    assert snapshot.gpus is None  # contract: gpus nullable on CPU-only
    assert snapshot.source == "ram"
    assert snapshot.ram_total_mb > 0


def test_first_successful_provider_wins() -> None:
    class Working:
        name = "nvml"

        def probe(self) -> bool:
            return True

        def snapshot(self) -> TelemetrySnapshot:
            return TelemetrySnapshot(source="nvml", gpus=(), ram_total_mb=1024)

    provider = pick_provider((FailingProvider(), Working(), RamProvider()))
    assert isinstance(provider, Working)


def test_empty_provider_list_still_returns_ram() -> None:
    assert pick_provider(()).name == "ram"


# -- NvmlProvider against a mocked pynvml module ------------------------------
def _fake_pynvml(gpus: list[tuple[object, int, int]], *, fail_init: bool = False):
    """A stand-in pynvml module: gpus is [(name, total_bytes, used_bytes)]. Names
    may be bytes (older NVML bindings) or str (newer) — both appear in the wild."""
    module = types.ModuleType("pynvml")
    state = {"initialized": False, "shutdowns": 0}

    def nvml_init():
        if fail_init:
            raise RuntimeError("driver not loaded")
        state["initialized"] = True

    def nvml_shutdown():
        state["shutdowns"] += 1

    class _Mem:
        def __init__(self, total: int, used: int) -> None:
            self.total = total
            self.used = used

    module.nvmlInit = nvml_init
    module.nvmlShutdown = nvml_shutdown
    module.nvmlDeviceGetCount = lambda: len(gpus)
    module.nvmlDeviceGetHandleByIndex = lambda i: i
    module.nvmlDeviceGetMemoryInfo = lambda handle: _Mem(gpus[handle][1], gpus[handle][2])
    module.nvmlDeviceGetName = lambda handle: gpus[handle][0]
    module._state = state
    return module


def test_nvml_provider_parses_mocked_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = 1024**3
    fake = _fake_pynvml(
        [
            (b"NVIDIA GeForce RTX 4090", 24 * gib, 3 * gib),  # bytes name (old bindings)
            ("NVIDIA RTX A2000", 8 * gib, 1 * gib),  # str name (new bindings)
        ]
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    provider = NvmlProvider()
    assert provider.probe() is True
    snapshot = provider.snapshot()
    assert snapshot.source == "nvml"
    assert snapshot.ram_total_mb > 0
    assert [g.index for g in snapshot.gpus] == [0, 1]
    first, second = snapshot.gpus
    assert first.name == "NVIDIA GeForce RTX 4090"  # bytes decoded
    assert first.vram_total_mb == 24 * 1024
    assert first.vram_used_mb == 3 * 1024
    assert second.name == "NVIDIA RTX A2000"
    assert second.source == "nvml"
    # NVML is shut down after probe AND after snapshot (no leaked handles).
    assert fake._state["shutdowns"] == 2


def test_nvml_probe_false_on_driver_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([], fail_init=True))
    assert NvmlProvider().probe() is False


def test_nvml_probe_false_on_zero_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", _fake_pynvml([]))
    assert NvmlProvider().probe() is False


# -- RocmSmiProvider against canned subprocess output -------------------------
_ROCM_JSON = (
    '{"card1": {"VRAM Total Memory (B)": "17163091968",'
    ' "VRAM Total Used Memory (B)": "1073741824"},'
    ' "card0": {"VRAM Total Memory (B)": "25753026560",'
    ' "VRAM Total Used Memory (B)": "2147483648"}}'
)


def test_rocm_smi_provider_parses_canned_output(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=_ROCM_JSON, stderr="")

    monkeypatch.setattr("suiban.sched.telemetry.subprocess.run", fake_run)
    snapshot = RocmSmiProvider().snapshot()
    assert calls == [["rocm-smi", "--showmeminfo", "vram", "--json"]]
    assert snapshot.source == "rocm-smi"
    # Cards sorted by key: card0 first regardless of JSON order.
    card0, card1 = snapshot.gpus
    assert (card0.index, card1.index) == (0, 1)
    assert "card0" in card0.name and "card1" in card1.name
    assert card0.vram_total_mb == 25753026560 // (1024 * 1024)
    assert card0.vram_used_mb == 2048
    assert card1.vram_total_mb == 17163091968 // (1024 * 1024)
    assert card1.vram_used_mb == 1024


def test_rocm_smi_probe_uses_which(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("suiban.sched.telemetry.shutil.which", lambda name: "/usr/bin/rocm-smi")
    assert RocmSmiProvider().probe() is True
    monkeypatch.setattr("suiban.sched.telemetry.shutil.which", lambda name: None)
    assert RocmSmiProvider().probe() is False
