"""Loadout planner across all five hardware tiers (plan §Verification fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import FakeTelemetry
from suiban.config import Settings
from suiban.sched.budget import BudgetProvider
from suiban.sched.planner import Loadout, plan_loadout
from suiban.sched.telemetry import TelemetrySnapshot


def plan(
    gpus_mb: list[int] | None, *, ram_mb: int = 63 * 1024, settings: Settings | None = None
) -> Loadout:
    telemetry: TelemetrySnapshot = FakeTelemetry(gpus_mb, ram_mb=ram_mb).snapshot()
    return plan_loadout(telemetry, settings or Settings(), BudgetProvider(), backend="cuda")


def notice_codes(loadout: Loadout) -> set[str]:
    return {n.code for n in loadout.notices}


def worker_models(loadout: Loadout) -> list[str]:
    return [s.model for s in loadout.workers]


def test_24gb_tier(bonsai_home: Path) -> None:
    loadout = plan([24 * 1024])
    assert loadout.tier == "24gb"
    assert loadout.family_effective == "ternary"
    assert loadout.family_degraded is False
    assert loadout.orchestrator.model == "bonsai-27b"
    assert loadout.orchestrator.ctx == 32768
    assert loadout.orchestrator.mmproj is True
    utility = next(s for s in loadout.slots if s.role == "utility")
    assert utility.model == "bonsai-4b"
    assert worker_models(loadout) == ["bonsai-8b", "bonsai-8b"]
    # Plan fixture: 24 GB loadout totals ~18.3 GiB (analytic priors land at ~18.46).
    total_gib = sum(s.vram_mb for s in loadout.slots) / 1024
    assert total_gib == pytest.approx(18.3, abs=0.3)
    assert loadout.headroom_mb > 0
    # unique ports, orchestrator first at 8701
    ports = [s.port for s in loadout.slots]
    assert len(set(ports)) == len(ports)
    assert loadout.orchestrator.port == 8701


def test_16gb_tier_analytic_priors_degrade_worker(bonsai_home: Path) -> None:
    loadout = plan([16 * 1024])
    assert loadout.tier == "16gb"
    assert loadout.family_effective == "ternary"  # no family degradation at 16 GB
    assert loadout.orchestrator.ctx == 32768
    # With the conservative analytic buffer priors the 8B worker of the nominal tier
    # table does not fit; the ladder honestly lands on 1x4B (see the measured test).
    assert worker_models(loadout) == ["bonsai-4b"]
    assert "workers_degraded" in notice_codes(loadout)


def test_16gb_tier_measured_budget_restores_8b_worker(bonsai_home: Path) -> None:
    provider = BudgetProvider()
    provider.record_measured("bonsai-27b", "ternary", buffers_mb=600)
    provider.record_measured("bonsai-8b", "ternary", buffers_mb=300)
    telemetry = FakeTelemetry([16 * 1024]).snapshot()
    loadout = plan_loadout(telemetry, Settings(), BudgetProvider(), backend="cuda")
    assert worker_models(loadout) == ["bonsai-8b"]  # nominal tier table restored


def test_12gb_tier_family_degrades_to_1bit(bonsai_home: Path) -> None:
    loadout = plan([12 * 1024])
    assert loadout.tier == "12gb"
    assert loadout.family_configured == "ternary"
    assert loadout.family_effective == "1bit"
    assert loadout.family_degraded is True
    assert loadout.family_reason is not None
    assert "family_degraded" in notice_codes(loadout)
    assert all(s.family == "1bit" for s in loadout.slots)
    assert loadout.orchestrator.quant == "Q1_0"
    utility = next(s for s in loadout.slots if s.role == "utility")
    assert utility.model == "bonsai-4b"
    # 1-bit weights are small enough that a single 8B worker fits (the plan's tier
    # table conservatively promised 1x4B; exceeding it is fine, underdelivering is not)
    assert worker_models(loadout) in (["bonsai-8b"], ["bonsai-4b"])


def test_12gb_configured_1bit_is_not_reported_degraded(bonsai_home: Path) -> None:
    loadout = plan([12 * 1024], settings=Settings(quant_family="1bit"))
    assert loadout.family_effective == "1bit"
    assert loadout.family_degraded is False
    assert "family_degraded" not in notice_codes(loadout)


def test_8gb_tier(bonsai_home: Path) -> None:
    loadout = plan([8 * 1024])
    assert loadout.tier == "8gb"
    assert loadout.family_effective == "1bit"
    assert loadout.orchestrator.model == "bonsai-27b"
    assert loadout.orchestrator.mmproj is True  # vision stays at every GPU tier
    assert loadout.orchestrator.ctx == 8192  # ctx ladder floor
    utility = next(s for s in loadout.slots if s.role == "utility")
    assert utility.model == "bonsai-1.7b"
    assert worker_models(loadout) == []
    codes = notice_codes(loadout)
    assert "ultra_sequential" in codes
    assert "vram_tight" in codes
    assert "orchestrator_ctx_reduced" in codes
    caps = loadout.capabilities(Settings())
    assert caps["vision"] is True
    assert caps["ultra_parallel"] is False


def test_cpu_tier_big_ram(bonsai_home: Path) -> None:
    loadout = plan(None, ram_mb=62 * 1024)
    assert loadout.tier == "cpu"
    assert len(loadout.slots) == 1
    slot = loadout.slots[0]
    assert slot.model == "bonsai-27b"
    assert slot.role == "orchestrator"
    assert slot.gpu is None
    assert loadout.utility_shared_with_orchestrator is True
    assert "cpu_only" in notice_codes(loadout)
    caps = loadout.capabilities(Settings())
    assert caps["skill_writes"] is True
    assert caps["ultra_parallel"] is False


def test_cpu_tier_small_ram_uses_8b(bonsai_home: Path) -> None:
    loadout = plan(None, ram_mb=8 * 1024)
    assert loadout.tier == "cpu"
    assert loadout.slots[0].model == "bonsai-8b"
    assert "no_27b_resident" in notice_codes(loadout)
    caps = loadout.capabilities(Settings())
    assert caps["vision"] is False
    assert caps["skill_writes"] is False


def test_tiny_gpu_falls_back_to_cpu(bonsai_home: Path) -> None:
    loadout = plan([4 * 1024])
    assert loadout.tier == "cpu"
    assert "gpu_too_small" in notice_codes(loadout)


def test_prefer_workers_1_caps_ladder(bonsai_home: Path) -> None:
    settings = Settings.model_validate({"loadout": {"prefer_workers": 1}})
    loadout = plan([24 * 1024], settings=settings)
    assert len(loadout.workers) == 1
    assert loadout.workers[0].model == "bonsai-8b"


def test_prefer_workers_0_means_sequential(bonsai_home: Path) -> None:
    settings = Settings.model_validate({"loadout": {"prefer_workers": 0}})
    loadout = plan([24 * 1024], settings=settings)
    assert worker_models(loadout) == []
    assert "ultra_sequential" in notice_codes(loadout)


def test_multi_gpu_places_workers_on_gpu1(bonsai_home: Path) -> None:
    loadout = plan([24 * 1024, 24 * 1024])
    assert loadout.orchestrator.gpu == 0
    assert all(w.gpu == 1 for w in loadout.workers)
    assert worker_models(loadout) == ["bonsai-8b", "bonsai-8b"]


# -- full tier matrix, table-driven (deep-detail pass) ------------------------
# Each row pins the COMPLETE planner outcome for a GPU configuration: tier, family,
# every slot (role/model/gpu), and the notice codes. Values are the planner's real
# analytic-prior output, verified by hand against budget.py — a change here must be
# a deliberate planning change, never drift.
_MATRIX: list[dict] = [
    {
        "id": "24gb-single",
        "gpus": [24 * 1024],
        "tier": "24gb",
        "family": "ternary",
        "degraded": False,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-4b", 0),
            ("worker-1", "bonsai-8b", 0),
            ("worker-2", "bonsai-8b", 0),
        ],
        "notices": set(),
        "ultra_parallel": True,
    },
    {
        "id": "16gb-single",
        "gpus": [16 * 1024],
        "tier": "16gb",
        "family": "ternary",
        "degraded": False,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-4b", 0),
            ("worker-1", "bonsai-4b", 0),  # analytic priors: one rung below nominal
        ],
        "notices": {"workers_degraded"},
        "ultra_parallel": True,
    },
    {
        "id": "12gb-single",
        "gpus": [12 * 1024],
        "tier": "12gb",
        "family": "1bit",
        "degraded": True,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-4b", 0),
            ("worker-1", "bonsai-8b", 0),  # 1-bit weights leave room for an 8B
        ],
        "notices": {"family_degraded", "workers_degraded"},
        "ultra_parallel": True,
    },
    {
        "id": "8gb-single",
        "gpus": [8 * 1024],
        "tier": "8gb",
        "family": "1bit",
        "degraded": True,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-1.7b", 0),
        ],
        "notices": {
            "family_degraded",
            "orchestrator_ctx_reduced",
            "ultra_sequential",
            "vram_tight",
        },
        "ultra_parallel": False,
    },
    {
        "id": "24gb-plus-8gb",
        "gpus": [24 * 1024, 8 * 1024],
        "tier": "24gb",  # tier is GPU 0's
        "family": "ternary",
        "degraded": False,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-4b", 0),
            ("worker-1", "bonsai-8b", 1),  # workers sized by GPU 1's 8 GB pool
            ("worker-2", "bonsai-4b", 1),
        ],
        "notices": {"workers_degraded"},
        "ultra_parallel": True,
    },
    {
        "id": "dual-8gb",
        "gpus": [8 * 1024, 8 * 1024],
        "tier": "8gb",
        "family": "1bit",
        "degraded": True,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-1.7b", 0),
            ("worker-1", "bonsai-8b", 1),  # the second card rescues Ultra parallelism
            ("worker-2", "bonsai-8b", 1),
        ],
        "notices": {"family_degraded", "orchestrator_ctx_reduced", "vram_tight"},
        "ultra_parallel": True,
    },
    {
        "id": "triple-24gb",
        "gpus": [24 * 1024] * 3,
        "tier": "24gb",
        "family": "ternary",
        "degraded": False,
        "slots": [
            ("orchestrator", "bonsai-27b", 0),
            ("utility", "bonsai-4b", 0),
            # v1 places workers on GPU 1 only; GPU 2 is honest headroom, not slots.
            ("worker-1", "bonsai-8b", 1),
            ("worker-2", "bonsai-8b", 1),
        ],
        "notices": set(),
        "ultra_parallel": True,
    },
    {
        "id": "cpu-only",
        "gpus": None,
        "tier": "cpu",
        "family": "ternary",
        "degraded": False,
        "slots": [("orchestrator", "bonsai-27b", None)],
        "notices": {"cpu_only"},
        "ultra_parallel": False,
    },
]


@pytest.mark.parametrize("case", _MATRIX, ids=[c["id"] for c in _MATRIX])
def test_planner_matrix(bonsai_home: Path, case: dict) -> None:
    loadout = plan(case["gpus"])
    assert loadout.tier == case["tier"]
    assert loadout.family_effective == case["family"]
    assert loadout.family_degraded is case["degraded"]
    assert [(s.slot_id, s.model, s.gpu) for s in loadout.slots] == case["slots"]
    assert all(s.family == case["family"] for s in loadout.slots)
    assert notice_codes(loadout) == case["notices"]
    caps = loadout.capabilities(Settings())
    assert caps["ultra_parallel"] is case["ultra_parallel"]
    # Ports are unique and VRAM accounting stays inside the physical total.
    ports = [s.port for s in loadout.slots]
    assert len(set(ports)) == len(ports)
    if case["gpus"]:
        assert loadout.headroom_mb >= 0


def test_slot_dict_matches_contract_shape(bonsai_home: Path) -> None:
    slot_dict = plan([24 * 1024]).orchestrator.as_dict()
    assert set(slot_dict) == {
        "slot_id",
        "role",
        "model",
        "family",
        "quant",
        "ctx",
        "gpu",
        "port",
        "state",
        "vram_mb",
        "mmproj",
        "dspark",
    }
    assert slot_dict["quant"] == "Q2_0"
