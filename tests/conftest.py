"""Shared fixtures. Everything runs modelless: SUIBAN_HOME points at a tmp dir and the
llama layer uses the deterministic in-process mock backend (SUIBAN_LLAMA_MOCK)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from suiban.app import create_app
from suiban.sched.telemetry import GpuInfo, TelemetrySnapshot


@pytest.fixture(autouse=True)
def bonsai_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "bonsai-home"
    monkeypatch.setenv("SUIBAN_HOME", str(home))
    monkeypatch.setenv("SUIBAN_LLAMA_MOCK", "1")
    return home


@pytest.fixture(autouse=True)
def offline_default_transports(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may touch the network: the search and provider default clients are
    replaced with a transport that refuses every request. Tests that want traffic
    inject their own client_factory (httpx.MockTransport) instead."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"network disabled in tests ({request.url})")

    def offline_client() -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(refuse))

    monkeypatch.setattr("suiban.search.providers._default_client", offline_client)
    monkeypatch.setattr("suiban.providers.registry._default_client", offline_client)


class FakeTelemetry:
    """Deterministic telemetry provider for tests."""

    def __init__(
        self,
        gpus_mb: list[int] | None,
        *,
        ram_mb: int = 63 * 1024,
        source: str = "nvml",
    ) -> None:
        self.name = source
        self._source = source
        self._ram_mb = ram_mb
        if gpus_mb is None:
            self._gpus: tuple[GpuInfo, ...] | None = None
            self._source = "ram"
            self.name = "ram"
        else:
            self._gpus = tuple(
                GpuInfo(
                    index=i,
                    name=f"Fake GPU {i}",
                    vram_total_mb=total,
                    vram_used_mb=0,
                    source=source,
                )
                for i, total in enumerate(gpus_mb)
            )

    def probe(self) -> bool:
        return True

    def snapshot(self) -> TelemetrySnapshot:
        return TelemetrySnapshot(source=self._source, gpus=self._gpus, ram_total_mb=self._ram_mb)


@pytest.fixture
def telemetry_24gb() -> FakeTelemetry:
    return FakeTelemetry([24 * 1024])


@pytest.fixture
def client(bonsai_home: Path, telemetry_24gb: FakeTelemetry):
    """App with a fake 24 GB GPU, CUDA backend, mock llama slots."""
    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as test_client:
        yield test_client
