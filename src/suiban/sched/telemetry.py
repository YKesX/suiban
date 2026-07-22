"""GPU telemetry abstraction.

Provider order: nvml -> rocm-smi -> metal -> ram (psutil fallback, gpus=None).
`/v1/system.gpus` is nullable by contract — CPU-only machines report gpus=null with
telemetry_source="ram". Never crash on a missing driver; fall through instead.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Protocol

import psutil


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    vram_total_mb: int
    vram_used_mb: int
    source: str

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "source": self.source,
        }


@dataclass(frozen=True)
class TelemetrySnapshot:
    source: str  # nvml | rocm-smi | metal | ram
    gpus: tuple[GpuInfo, ...] | None  # None on CPU-only (contract: gpus nullable)
    ram_total_mb: int


class TelemetryProvider(Protocol):
    name: str

    def probe(self) -> bool: ...

    def snapshot(self) -> TelemetrySnapshot: ...


def _ram_total_mb() -> int:
    return int(psutil.virtual_memory().total // (1024 * 1024))


class NvmlProvider:
    name = "nvml"

    def probe(self) -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            ok = pynvml.nvmlDeviceGetCount() > 0
            pynvml.nvmlShutdown()
            return ok
        except Exception:
            return False

    def snapshot(self) -> TelemetrySnapshot:
        import pynvml

        pynvml.nvmlInit()
        try:
            gpus = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                gpus.append(
                    GpuInfo(
                        index=i,
                        name=name,
                        vram_total_mb=int(mem.total // (1024 * 1024)),
                        vram_used_mb=int(mem.used // (1024 * 1024)),
                        source=self.name,
                    )
                )
        finally:
            pynvml.nvmlShutdown()
        return TelemetrySnapshot(source=self.name, gpus=tuple(gpus), ram_total_mb=_ram_total_mb())


class RocmSmiProvider:
    name = "rocm-smi"

    def probe(self) -> bool:
        return shutil.which("rocm-smi") is not None

    def snapshot(self) -> TelemetrySnapshot:
        out = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        data = json.loads(out)
        gpus = []
        for i, (card, fields) in enumerate(sorted(data.items())):
            total = int(fields.get("VRAM Total Memory (B)", 0))
            used = int(fields.get("VRAM Total Used Memory (B)", 0))
            gpus.append(
                GpuInfo(
                    index=i,
                    # TODO(v1.1): pull the marketing name via --showproductname; the
                    # meminfo JSON only carries the card key.
                    name=f"AMD GPU ({card})",
                    vram_total_mb=total // (1024 * 1024),
                    vram_used_mb=used // (1024 * 1024),
                    source=self.name,
                )
            )
        return TelemetrySnapshot(source=self.name, gpus=tuple(gpus), ram_total_mb=_ram_total_mb())


class MetalProvider:
    """Apple Silicon: unified memory, so 'VRAM' is system memory.

    TODO(v1.1): query IOKit for the real recommendedMaxWorkingSetSize instead of
    reporting total unified RAM.
    """

    name = "metal"

    def probe(self) -> bool:
        return sys.platform == "darwin"

    def snapshot(self) -> TelemetrySnapshot:
        vm = psutil.virtual_memory()
        gpu = GpuInfo(
            index=0,
            name="Apple Silicon (unified memory)",
            vram_total_mb=int(vm.total // (1024 * 1024)),
            vram_used_mb=int(vm.used // (1024 * 1024)),
            source=self.name,
        )
        return TelemetrySnapshot(source=self.name, gpus=(gpu,), ram_total_mb=_ram_total_mb())


class RamProvider:
    """CPU-only fallback: no GPUs (gpus=null in /v1/system), RAM budget only."""

    name = "ram"

    def probe(self) -> bool:
        return True

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(source=self.name, gpus=None, ram_total_mb=_ram_total_mb())


DEFAULT_PROVIDERS: tuple[type, ...] = (NvmlProvider, RocmSmiProvider, MetalProvider, RamProvider)


def pick_provider(providers: tuple[TelemetryProvider, ...] | None = None) -> TelemetryProvider:
    """First provider whose probe succeeds; RamProvider always succeeds last."""
    candidates: tuple[TelemetryProvider, ...] = (
        providers if providers is not None else tuple(cls() for cls in DEFAULT_PROVIDERS)
    )
    for provider in candidates:
        try:
            if provider.probe():
                return provider
        except Exception:
            continue
    return RamProvider()
