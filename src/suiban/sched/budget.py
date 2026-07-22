"""VRAM budget math: analytic priors from verified model facts, plus measured
overrides persisted at ~/.bonsai/budget.json after first real launch.

Verified constants (plan Recon; do not "improve" them):
- KV bytes/element: f16 2.0 · q8_0 1.0625 · tq4_0 0.5625 (18 B/32) · tq3_0 0.4375 (14 B/32)
- KV elements/token (per K and per V): 27B 16x4x256 · 8B 36x8x128 · 4B 36x8x128 ·
  1.7B 28x8x128  ->  K=q8_0 + V=TQ4_0 gives 26.0 / 58.5 / 58.5 / 45.5 KiB/token.
- Weight sizes (GiB): 27B 6.67/3.54 · 8B 2.03/1.08 · 4B 1.00/0.53 · 1.7B 0.43/0.23
  (ternary Q2_0 / 1-bit Q1_0). mmproj (27B vision, Q8_0): 629 MB ~= 0.63 GiB.
- Buffer priors: 1.2 GiB (27B) / 0.6 GiB (others). Deliberately conservative;
  measured values replace them after the first real launch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from suiban import paths

MIB = 1024 * 1024
GIB = 1024 * MIB

MODELS: tuple[str, ...] = ("bonsai-27b", "bonsai-8b", "bonsai-4b", "bonsai-1.7b")

KV_BYTES_PER_ELEMENT: dict[str, float] = {
    "f16": 2.0,
    "q8_0": 1.0625,
    "tq4_0": 0.5625,
    "tq3_0": 0.4375,
}

# layers x kv_heads x head_dim, per K cache and per V cache each.
KV_ELEMENTS_PER_TOKEN: dict[str, int] = {
    "bonsai-27b": 16 * 4 * 256,  # 64L hybrid, 16 full-attention layers
    "bonsai-8b": 36 * 8 * 128,
    "bonsai-4b": 36 * 8 * 128,
    "bonsai-1.7b": 28 * 8 * 128,
}

WEIGHTS_GIB: dict[str, dict[str, float]] = {
    "bonsai-27b": {"ternary": 6.67, "1bit": 3.54},
    "bonsai-8b": {"ternary": 2.03, "1bit": 1.08},
    "bonsai-4b": {"ternary": 1.00, "1bit": 0.53},
    "bonsai-1.7b": {"ternary": 0.43, "1bit": 0.23},
}

BUFFER_PRIOR_GIB: dict[str, float] = {
    "bonsai-27b": 1.2,
    "bonsai-8b": 0.6,
    "bonsai-4b": 0.6,
    "bonsai-1.7b": 0.6,
}

MMPROJ_GIB = 0.63  # 629 MB Q8_0, 27B only
DSPARK_GIB = 1.8  # ~1.8 GB CUDA drafter, opt-in; measured value replaces this prior

MAX_CTX: dict[str, int] = {
    "bonsai-27b": 262_144,
    "bonsai-8b": 65_536,
    "bonsai-4b": 32_768,
    "bonsai-1.7b": 32_768,
}

QUANT_NAME: dict[str, str] = {"ternary": "Q2_0", "1bit": "Q1_0"}


def kv_kib_per_token(model: str, k_type: str = "q8_0", v_type: str = "tq4_0") -> float:
    elems = KV_ELEMENTS_PER_TOKEN[model]
    bytes_per_token = elems * (KV_BYTES_PER_ELEMENT[k_type] + KV_BYTES_PER_ELEMENT[v_type])
    return bytes_per_token / 1024


def kv_gib(model: str, ctx: int, k_type: str = "q8_0", v_type: str = "tq4_0") -> float:
    # KiB/token * tokens -> KiB; / 1024^2 -> GiB
    return kv_kib_per_token(model, k_type, v_type) * ctx / (1024 * 1024)


@dataclass(frozen=True)
class SlotCost:
    """Analytic (or measured-override) VRAM cost of one llama-server slot, in MiB."""

    model: str
    family: str
    ctx: int
    k_type: str
    v_type: str
    weights_mb: int
    kv_mb: int
    buffers_mb: int
    extras_mb: int  # mmproj + dspark
    source: str  # "analytic" | "measured"

    @property
    def total_mb(self) -> int:
        return self.weights_mb + self.kv_mb + self.buffers_mb + self.extras_mb

    @property
    def total_gib(self) -> float:
        return self.total_mb / 1024

    @property
    def kv_config(self) -> str:
        return f"K={self.k_type},V={self.v_type}"


class BudgetProvider:
    """Analytic prior table with measured overrides from ~/.bonsai/budget.json.

    budget.json shape (written after first real launch, read here):
        { "measured": { "<model>/<family>": { "weights_mb": int?, "buffers_mb": int? } } }
    Missing keys fall back to the analytic prior — partial measurements are fine.
    """

    def __init__(self, budget_file: Path | None = None) -> None:
        self._file = budget_file or paths.budget_path()
        self._measured: dict[str, dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        if not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            measured = data.get("measured", {})
            if isinstance(measured, dict):
                self._measured = {
                    str(k): {mk: int(mv) for mk, mv in v.items()}
                    for k, v in measured.items()
                    if isinstance(v, dict)
                }
        except (ValueError, OSError):
            # Corrupt budget file: fall back to analytic priors, never crash.
            self._measured = {}

    @property
    def has_measurements(self) -> bool:
        return bool(self._measured)

    def record_measured(
        self,
        model: str,
        family: str,
        *,
        weights_mb: int | None = None,
        buffers_mb: int | None = None,
    ) -> None:
        """Persist a measured override (called after a real slot launch)."""
        key = f"{model}/{family}"
        entry = self._measured.setdefault(key, {})
        if weights_mb is not None:
            entry["weights_mb"] = weights_mb
        if buffers_mb is not None:
            entry["buffers_mb"] = buffers_mb
        self._file.parent.mkdir(parents=True, exist_ok=True)
        self._file.write_text(json.dumps({"measured": self._measured}, indent=2), encoding="utf-8")

    def slot_cost(
        self,
        model: str,
        family: str,
        ctx: int,
        *,
        k_type: str = "q8_0",
        v_type: str = "tq4_0",
        with_mmproj: bool = False,
        with_dspark: bool = False,
    ) -> SlotCost:
        override = self._measured.get(f"{model}/{family}", {})
        weights_mb = override.get("weights_mb", round(WEIGHTS_GIB[model][family] * 1024))
        buffers_mb = override.get("buffers_mb", round(BUFFER_PRIOR_GIB[model] * 1024))
        kv_mb = round(kv_gib(model, ctx, k_type, v_type) * 1024)
        extras_mb = 0
        if with_mmproj:
            extras_mb += round(MMPROJ_GIB * 1024)
        if with_dspark:
            extras_mb += round(DSPARK_GIB * 1024)
        source = "measured" if override else "analytic"
        return SlotCost(
            model=model,
            family=family,
            ctx=ctx,
            k_type=k_type,
            v_type=v_type,
            weights_mb=weights_mb,
            kv_mb=kv_mb,
            buffers_mb=buffers_mb,
            extras_mb=extras_mb,
            source=source,
        )

    def table_rows(
        self, family: str, k_type: str, v_type: str, ctx_by_model: dict[str, int]
    ) -> list[dict]:
        """Rows for GET /v1/system/budget."""
        rows = []
        for model in MODELS:
            ctx = ctx_by_model.get(model, min(MAX_CTX[model], 16384))
            cost = self.slot_cost(
                model,
                family,
                ctx,
                k_type=k_type,
                v_type=v_type,
                with_mmproj=(model == "bonsai-27b"),
            )
            rows.append(
                {
                    "model": model,
                    "family": family,
                    "ctx": ctx,
                    "kv_config": cost.kv_config,
                    "weights_mb": cost.weights_mb,
                    "kv_mb": cost.kv_mb,
                    "buffers_mb": cost.buffers_mb + cost.extras_mb,
                    "total_mb": cost.total_mb,
                    "source": cost.source,
                }
            )
        return rows
