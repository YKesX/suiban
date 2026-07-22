"""Effort ladder: thinking budgets (with the 40% ctx cap), tool iterations, sampling."""

from __future__ import annotations

import pytest

from suiban.effort import (
    EFFORT_LEVELS,
    max_tool_iterations,
    sampling_for,
    thinking_budget,
)


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", 0), ("mid", 4096), ("high", 12288), ("xhigh", 24576)],
)
def test_thinking_budgets_uncapped(effort: str, expected: int) -> None:
    # 65536 ctx -> 40% cap is 26214, above every finite rung
    assert thinking_budget(effort, 65536) == expected


def test_max_effort_resolves_to_the_cap() -> None:
    # -1 (unlimited) always becomes 40% of slot ctx
    assert thinking_budget("max", 32768) == 13107
    assert thinking_budget("max", 8192) == 3276


def test_cap_applies_to_finite_budgets() -> None:
    # mid=4096 but 40% of 8192 = 3276 caps it
    assert thinking_budget("mid", 8192) == 3276
    # xhigh=24576 but 40% of 32768 = 13107 caps it
    assert thinking_budget("xhigh", 32768) == 13107


def test_low_is_always_zero() -> None:
    assert thinking_budget("low", 2048) == 0
    assert thinking_budget("low", 262144) == 0


@pytest.mark.parametrize(
    ("effort", "expected"),
    [("low", 8), ("mid", 16), ("high", 32), ("xhigh", 48), ("max", 64)],
)
def test_tool_iterations(effort: str, expected: int) -> None:
    assert max_tool_iterations(effort) == expected


def test_ladder_is_complete() -> None:
    assert EFFORT_LEVELS == ("low", "mid", "high", "xhigh", "max")


def test_sampling_27b() -> None:
    s = sampling_for("bonsai-27b")
    assert (s.temperature, s.top_p, s.top_k) == (0.7, 0.95, 20)


@pytest.mark.parametrize("model", ["bonsai-8b", "bonsai-4b", "bonsai-1.7b"])
def test_sampling_small_models(model: str) -> None:
    s = sampling_for(model)
    assert (s.temperature, s.top_p, s.top_k) == (0.5, 0.85, 20)
