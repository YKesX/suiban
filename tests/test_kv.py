"""KV fallback chain: never crash, never silently degrade."""

from __future__ import annotations

from suiban.config import KvSettings
from suiban.kv import resolve_kv_state


def test_happy_path_recommended() -> None:
    state = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=True)
    assert (state.k_type, state.v_type) == ("q8_0", "tq4_0")
    assert state.fallback_active is False
    assert state.notice is None
    assert state.as_dict()["turboquant"] == {
        "enabled": True,
        "preset": "recommended",
        "backend_supported": True,
        "fallback_active": False,
        "fallback_reason": None,
    }


def test_aggressive_preset_uses_tq3() -> None:
    state = resolve_kv_state(
        KvSettings(preset="aggressive"), backend_supported=True, fa_available=True
    )
    assert state.v_type == "tq3_0"


def test_disclaimer_toggle_off_is_config_not_fallback() -> None:
    state = resolve_kv_state(
        KvSettings(turboquant_enabled=False), backend_supported=True, fa_available=True
    )
    assert (state.k_type, state.v_type) == ("q8_0", "q8_0")
    assert state.fallback_active is False
    assert state.notice is None


def test_missing_kernels_fall_back_to_q8_with_notice() -> None:
    state = resolve_kv_state(KvSettings(), backend_supported=False, fa_available=True)
    assert (state.k_type, state.v_type) == ("q8_0", "q8_0")
    assert state.fallback_active is True
    assert state.notice is not None
    assert state.notice.code == "turboquant_prebuilt_fallback"
    assert "suiban install turboquant" in state.fallback_reason


def test_no_fa_falls_back_to_f16_with_notice() -> None:
    state = resolve_kv_state(KvSettings(), backend_supported=True, fa_available=False)
    assert (state.k_type, state.v_type) == ("f16", "f16")
    assert state.fallback_active is True
    assert state.notice.code == "kv_fa_fallback"
