"""KV-cache type resolution with the graceful fallback chain (plan-frozen):

    V=TQ4_0/TQ3_0 (TurboQuant)
      -> kernels missing in the installed binary:  K=q8_0, V=q8_0  (+ notice)
      -> flash attention unusable:                 K=f16,  V=f16   (+ notice)

Disclaimer toggle off (kv.turboquant_enabled=false) or preset "off" -> K/V=q8_0/q8_0
(that is configuration, not a fallback). Never crash, never silently degrade.
"""

from __future__ import annotations

from dataclasses import dataclass

from suiban.config import KvSettings
from suiban.sched.planner import Notice

_PRESET_V_TYPE = {"recommended": "tq4_0", "aggressive": "tq3_0"}


@dataclass(frozen=True)
class KvState:
    k_type: str
    v_type: str
    enabled: bool
    preset: str
    backend_supported: bool
    fallback_active: bool
    fallback_reason: str | None
    notice: Notice | None = None

    def as_dict(self) -> dict:
        """The /v1/system `kv` object."""
        return {
            "k_type": self.k_type,
            "v_type": self.v_type,
            "turboquant": {
                "enabled": self.enabled,
                "preset": self.preset,
                "backend_supported": self.backend_supported,
                "fallback_active": self.fallback_active,
                "fallback_reason": self.fallback_reason,
            },
        }


def resolve_kv_state(
    settings: KvSettings, *, backend_supported: bool, fa_available: bool
) -> KvState:
    if not fa_available:
        # Quantized V hard-requires flash attention; without FA nothing quantized works.
        reason = "flash attention unavailable on this backend; KV cache runs at f16/f16."
        return KvState(
            k_type="f16",
            v_type="f16",
            enabled=settings.turboquant_enabled,
            preset=settings.preset,
            backend_supported=backend_supported,
            fallback_active=True,
            fallback_reason=reason,
            notice=Notice("warn", "kv_fa_fallback", reason),
        )

    if not settings.turboquant_enabled or settings.preset == "off":
        return KvState(
            k_type="q8_0",
            v_type="q8_0",
            enabled=False,
            preset=settings.preset,
            backend_supported=backend_supported,
            fallback_active=False,
            fallback_reason=None,
        )

    if not backend_supported:
        reason = (
            "TurboQuant kernels not present in prebuilt binary; using q8_0/q8_0. "
            "Run: suiban install turboquant"
        )
        return KvState(
            k_type="q8_0",
            v_type="q8_0",
            enabled=True,
            preset=settings.preset,
            backend_supported=False,
            fallback_active=True,
            fallback_reason=reason,
            notice=Notice("warn", "turboquant_prebuilt_fallback", reason),
        )

    return KvState(
        k_type="q8_0",
        v_type=_PRESET_V_TYPE[settings.preset],
        enabled=True,
        preset=settings.preset,
        backend_supported=True,
        fallback_active=False,
        fallback_reason=None,
    )
