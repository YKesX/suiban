"""Analytic budget table vs the plan's verified numbers, plus measured overrides."""

from __future__ import annotations

from pathlib import Path

import pytest

from suiban.sched.budget import (
    BudgetProvider,
    kv_gib,
    kv_kib_per_token,
)


# Plan fixtures: KiB/token at K=q8_0 + V=TQ4_0.
@pytest.mark.parametrize(
    ("model", "expected_kib"),
    [
        ("bonsai-27b", 26.0),
        ("bonsai-8b", 58.5),
        ("bonsai-4b", 58.5),
        ("bonsai-1.7b", 45.5),
    ],
)
def test_kv_kib_per_token_default_preset(model: str, expected_kib: float) -> None:
    assert kv_kib_per_token(model) == expected_kib


def test_27b_kv_at_32k_is_0_8125_gib() -> None:
    assert kv_gib("bonsai-27b", 32768) == 0.8125


def test_8b_kv_at_16k_matches_plan() -> None:
    # 58.5 KiB/tok * 16384 tokens = 0.9140625 GiB (plan: ~0.91)
    assert kv_gib("bonsai-8b", 16384) == pytest.approx(0.9140625)


def test_aggressive_preset_tq3() -> None:
    # K=q8_0 (1.0625) + V=TQ3_0 (0.4375) = 1.5 B/elem -> 27B: 16384*1.5/1024 = 24.0
    assert kv_kib_per_token("bonsai-27b", "q8_0", "tq3_0") == 24.0


def test_f16_fallback_costs_more() -> None:
    assert kv_kib_per_token("bonsai-27b", "f16", "f16") == 64.0


def test_slot_cost_27b_ternary(bonsai_home: Path) -> None:
    cost = BudgetProvider().slot_cost("bonsai-27b", "ternary", 32768, with_mmproj=True)
    assert cost.weights_mb == round(6.67 * 1024)
    assert cost.kv_mb == 832  # 0.8125 GiB
    assert cost.buffers_mb == round(1.2 * 1024)
    assert cost.extras_mb == round(0.63 * 1024)
    assert cost.source == "analytic"
    assert cost.kv_config == "K=q8_0,V=tq4_0"


def test_measured_override_roundtrip(bonsai_home: Path) -> None:
    provider = BudgetProvider()
    assert not provider.has_measurements
    provider.record_measured("bonsai-27b", "ternary", weights_mb=6700, buffers_mb=512)
    # a fresh provider reads the persisted file
    fresh = BudgetProvider()
    assert fresh.has_measurements
    cost = fresh.slot_cost("bonsai-27b", "ternary", 32768)
    assert cost.weights_mb == 6700
    assert cost.buffers_mb == 512
    assert cost.source == "measured"
    # unmeasured models keep analytic priors
    assert fresh.slot_cost("bonsai-8b", "ternary", 16384).source == "analytic"


def test_corrupt_budget_file_falls_back(bonsai_home: Path) -> None:
    bonsai_home.mkdir(parents=True, exist_ok=True)
    (bonsai_home / "budget.json").write_text("{not json", encoding="utf-8")
    provider = BudgetProvider()
    assert not provider.has_measurements  # graceful, no crash


def test_table_rows_shape(bonsai_home: Path) -> None:
    rows = BudgetProvider().table_rows(
        "ternary", "q8_0", "tq4_0", {"bonsai-27b": 32768, "bonsai-4b": 8192}
    )
    assert len(rows) == 4
    for row in rows:
        assert set(row) == {
            "model",
            "family",
            "ctx",
            "kv_config",
            "weights_mb",
            "kv_mb",
            "buffers_mb",
            "total_mb",
            "source",
        }
        assert row["total_mb"] == row["weights_mb"] + row["kv_mb"] + row["buffers_mb"]
