"""Loadout planner. Runs once at run start — the loadout NEVER changes mid-run.

Rules (plan-frozen):
- Permanent utility slot: 4B (1.7B on the 8 GB tier; shares the orchestrator slot on
  CPU-only). The utility model is resident in every GPU loadout.
- Worker degrade ladder: 2x8B -> 8B+4B -> 2x4B -> 1x4B -> 1x1.7B -> none (Ultra then
  runs sequentially, with a notice).
- 27B family degradation: configured ternary degrades to 1bit on GPUs <= 12 GB, with a
  notice; the configured value is preserved in settings.
- Safety margin: max(1.5 GiB, 8% of VRAM). Mandatory slots (orchestrator + utility) may
  eat into the margin as a last resort (tight-fit notice) — never crash, never silently
  drop the 27B.
- CPU-only: single orchestrator sized by RAM (27B if >= 16 GiB RAM else 8B), utility
  duty shared by the orchestrator, no workers.

TODO(v1.1): the analytic buffer priors (1.2/0.6 GiB) are deliberately conservative, so
on 16 GB / 12 GB the ladder can land one rung below the README tier table (4B instead
of 8B worker, 1.7B instead of 4B). Measured overrides in ~/.bonsai/budget.json restore
the nominal composition after the first real launch — see tests/test_planner.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from suiban.config import Settings
from suiban.sched.budget import MAX_CTX, QUANT_NAME, BudgetProvider, SlotCost
from suiban.sched.telemetry import TelemetrySnapshot

PORT_POOL_START = 8701

# GiB thresholds (MiB) for tier labels; family degradation applies below 14 GiB.
_TIER_THRESHOLDS_MB = (
    (20 * 1024, "24gb"),
    (14 * 1024, "16gb"),
    (10 * 1024, "12gb"),
    (6 * 1024, "8gb"),
)
_FAMILY_DEGRADE_BELOW_MB = 14 * 1024
_MIN_GPU_MB = 6 * 1024

# Worker rungs, walked top-down; first rung that fits wins. The plan writes the ladder
# as 2x8B -> 8B+4B -> 2x4B -> 1x4B -> 1x1.7B -> none, but its 16 GB tier table lands on
# a single 8B worker — so the 1x8B rung is inserted where it belongs by VRAM demand
# (between 2x4B and 1x4B) to keep the ladder monotone and the tier table reachable.
_WORKER_LADDER: tuple[tuple[str, ...], ...] = (
    ("bonsai-8b", "bonsai-8b"),
    ("bonsai-8b", "bonsai-4b"),
    ("bonsai-4b", "bonsai-4b"),
    ("bonsai-8b",),
    ("bonsai-4b",),
    ("bonsai-1.7b",),
    (),
)

_ORCH_CTX_LADDER_FLOOR = 8192


@dataclass(frozen=True)
class Notice:
    level: str  # "info" | "warn"
    code: str
    message: str

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "message": self.message}


@dataclass
class PlannedSlot:
    slot_id: str
    role: str  # orchestrator | worker | utility
    model: str
    family: str
    ctx: int
    gpu: int | None
    port: int
    vram_mb: int
    mmproj: bool = False
    dspark: bool = False
    state: str = "planned"  # manager moves it: starting -> ready | failed

    @property
    def quant(self) -> str:
        return QUANT_NAME[self.family]

    def as_dict(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "role": self.role,
            "model": self.model,
            "family": self.family,
            "quant": self.quant,
            "ctx": self.ctx,
            "gpu": self.gpu,
            "port": self.port,
            "state": self.state,
            "vram_mb": self.vram_mb,
            "mmproj": self.mmproj,
            "dspark": self.dspark,
        }


@dataclass
class Loadout:
    planned_at: str
    tier: str  # 24gb | 16gb | 12gb | 8gb | cpu
    slots: list[PlannedSlot]
    headroom_mb: int
    family_configured: str
    family_effective: str
    family_degraded: bool
    family_reason: str | None
    utility_shared_with_orchestrator: bool = False
    notices: list[Notice] = field(default_factory=list)

    @property
    def orchestrator(self) -> PlannedSlot | None:
        return next((s for s in self.slots if s.role == "orchestrator"), None)

    @property
    def workers(self) -> list[PlannedSlot]:
        return [s for s in self.slots if s.role == "worker"]

    def slot_for_model(self, model: str) -> PlannedSlot | None:
        return next((s for s in self.slots if s.model == model), None)

    def capabilities(self, settings: Settings) -> dict:
        has_27b = any(s.model == "bonsai-27b" for s in self.slots)
        return {
            "vision": has_27b and any(s.mmproj for s in self.slots),
            "browse_t2": has_27b and settings.browse.tier2_enabled,
            "skill_writes": has_27b,
            "ultra_parallel": len(self.workers) > 0,
        }

    def as_dict(self) -> dict:
        return {
            "planned_at": self.planned_at,
            "tier": self.tier,
            "slots": [s.as_dict() for s in self.slots],
            "headroom_mb": self.headroom_mb,
        }


def _tier_for(vram_mb: int) -> str | None:
    for threshold, label in _TIER_THRESHOLDS_MB:
        if vram_mb >= threshold:
            return label
    return None


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_loadout(
    telemetry: TelemetrySnapshot,
    settings: Settings,
    budget: BudgetProvider,
    *,
    backend: str = "cpu",
    k_type: str = "q8_0",
    v_type: str = "tq4_0",
) -> Loadout:
    """Produce the fixed loadout for this run. Pure function of its inputs."""
    if not telemetry.gpus:
        return _plan_cpu(telemetry, settings, budget, k_type, v_type)

    gpu0 = telemetry.gpus[0]
    tier = _tier_for(gpu0.vram_total_mb)
    if tier is None:
        loadout = _plan_cpu(telemetry, settings, budget, k_type, v_type)
        loadout.notices.insert(
            0,
            Notice(
                "warn",
                "gpu_too_small",
                f"GPU 0 has {gpu0.vram_total_mb} MiB VRAM (< {_MIN_GPU_MB} MiB); "
                "falling back to CPU-only loadout.",
            ),
        )
        return loadout

    notices: list[Notice] = []
    configured = settings.quant_family
    effective = configured
    degraded = False
    reason: str | None = None
    if gpu0.vram_total_mb < _FAMILY_DEGRADE_BELOW_MB and configured == "ternary":
        effective, degraded = "1bit", True
        reason = (
            f"GPU 0 VRAM {gpu0.vram_total_mb} MiB <= 12 GB tier: 27B runs the 1-bit "
            "(Q1_0) family so it stays resident. Configured family is unchanged."
        )
        notices.append(Notice("warn", "family_degraded", reason))

    margin_mb = max(int(1.5 * 1024), int(gpu0.vram_total_mb * 0.08))
    avail0 = gpu0.vram_total_mb - margin_mb

    dspark = settings.dspark_enabled and backend == "cuda"
    if settings.dspark_enabled and backend != "cuda":
        notices.append(
            Notice("info", "dspark_unavailable", "DSpark drafter is CUDA-only; ignoring toggle.")
        )

    # -- mandatory slots: orchestrator (27B, mmproj) + permanent utility ---------
    utility_model = "bonsai-1.7b" if tier == "8gb" else "bonsai-4b"
    utility_cost = budget.slot_cost(utility_model, effective, 8192, k_type=k_type, v_type=v_type)

    orch_ctx_ladder = sorted(
        {
            min(settings.loadout.orchestrator_ctx, MAX_CTX["bonsai-27b"]),
            16384,
            _ORCH_CTX_LADDER_FLOOR,
        },
        reverse=True,
    )
    orch_ctx_ladder = [c for c in orch_ctx_ladder if c <= settings.loadout.orchestrator_ctx]

    orch_cost: SlotCost | None = None
    tight_fit = False
    for ctx in orch_ctx_ladder:
        candidate = budget.slot_cost(
            "bonsai-27b",
            effective,
            ctx,
            k_type=k_type,
            v_type=v_type,
            with_mmproj=True,
            with_dspark=dspark,
        )
        if candidate.total_mb + utility_cost.total_mb <= avail0:
            orch_cost = candidate
            break
    if orch_cost is None:
        # Last resort: smallest ctx, allowed to eat into the safety margin (but never
        # past physical VRAM). Graceful degradation over crashing.
        candidate = budget.slot_cost(
            "bonsai-27b",
            effective,
            _ORCH_CTX_LADDER_FLOOR,
            k_type=k_type,
            v_type=v_type,
            with_mmproj=True,
            with_dspark=dspark,
        )
        if candidate.total_mb + utility_cost.total_mb <= gpu0.vram_total_mb:
            orch_cost = candidate
            tight_fit = True
            notices.append(
                Notice(
                    "warn",
                    "vram_tight",
                    "Orchestrator + utility fit inside physical VRAM but eat into the "
                    "safety margin; expect little headroom for long prompts.",
                )
            )
        else:
            loadout = _plan_cpu(telemetry, settings, budget, k_type, v_type)
            loadout.notices.insert(
                0,
                Notice(
                    "warn",
                    "loadout_overflow",
                    "Even the minimum GPU loadout does not fit VRAM; using CPU-only.",
                ),
            )
            return loadout
    if orch_cost.ctx < settings.loadout.orchestrator_ctx:
        notices.append(
            Notice(
                "warn",
                "orchestrator_ctx_reduced",
                f"Orchestrator context reduced to {orch_cost.ctx} (configured "
                f"{settings.loadout.orchestrator_ctx}) to fit VRAM.",
            )
        )

    # -- workers: walk the ladder ------------------------------------------------
    multi_gpu = len(telemetry.gpus) > 1
    if multi_gpu:
        gpu1 = telemetry.gpus[1]
        worker_pool_mb = gpu1.vram_total_mb - max(int(1.5 * 1024), int(gpu1.vram_total_mb * 0.08))
        worker_gpu = 1
    else:
        mandatory_mb = orch_cost.total_mb + utility_cost.total_mb
        worker_pool_mb = 0 if tight_fit else avail0 - mandatory_mb
        worker_gpu = 0

    worker_ctx = min(settings.loadout.worker_ctx, MAX_CTX["bonsai-1.7b"])
    chosen_rung: tuple[str, ...] = ()
    chosen_costs: list[SlotCost] = []
    ladder = [r for r in _WORKER_LADDER if len(r) <= settings.loadout.prefer_workers]
    for rung_index, rung in enumerate(ladder):
        costs = [
            budget.slot_cost(
                m, effective, min(worker_ctx, MAX_CTX[m]), k_type=k_type, v_type=v_type
            )
            for m in rung
        ]
        if sum(c.total_mb for c in costs) <= worker_pool_mb:
            chosen_rung, chosen_costs = rung, costs
            if rung_index > 0 and rung:
                notices.append(
                    Notice(
                        "info",
                        "workers_degraded",
                        f"Worker loadout degraded to {'+'.join(rung)} to fit VRAM.",
                    )
                )
            break
    if not chosen_rung:
        notices.append(
            Notice(
                "warn",
                "ultra_sequential",
                "No VRAM headroom for worker slots; Ultra mode runs sub-tasks "
                "sequentially on the orchestrator.",
            )
        )

    # -- assemble ---------------------------------------------------------------
    port = PORT_POOL_START
    slots: list[PlannedSlot] = []
    slots.append(
        PlannedSlot(
            slot_id="orchestrator",
            role="orchestrator",
            model="bonsai-27b",
            family=effective,
            ctx=orch_cost.ctx,
            gpu=0,
            port=port,
            vram_mb=orch_cost.total_mb,
            mmproj=True,
            dspark=dspark,
        )
    )
    port += 1
    slots.append(
        PlannedSlot(
            slot_id="utility",
            role="utility",
            model=utility_model,
            family=effective,
            ctx=utility_cost.ctx,
            gpu=0,
            port=port,
            vram_mb=utility_cost.total_mb,
        )
    )
    port += 1
    for i, cost in enumerate(chosen_costs, start=1):
        slots.append(
            PlannedSlot(
                slot_id=f"worker-{i}",
                role="worker",
                model=cost.model,
                family=effective,
                ctx=cost.ctx,
                gpu=worker_gpu,
                port=port,
                vram_mb=cost.total_mb,
            )
        )
        port += 1

    total_vram_mb = sum(g.vram_total_mb for g in telemetry.gpus)
    headroom_mb = total_vram_mb - sum(s.vram_mb for s in slots)
    return Loadout(
        planned_at=_now_iso(),
        tier=tier,
        slots=slots,
        headroom_mb=headroom_mb,
        family_configured=configured,
        family_effective=effective,
        family_degraded=degraded,
        family_reason=reason,
        notices=notices,
    )


def _plan_cpu(
    telemetry: TelemetrySnapshot,
    settings: Settings,
    budget: BudgetProvider,
    k_type: str,
    v_type: str,
) -> Loadout:
    ram_gib = telemetry.ram_total_mb / 1024
    model = "bonsai-27b" if ram_gib >= 16 else "bonsai-8b"
    family = settings.quant_family
    ctx = min(settings.loadout.orchestrator_ctx, MAX_CTX[model])
    cost = budget.slot_cost(
        model, family, ctx, k_type=k_type, v_type=v_type, with_mmproj=(model == "bonsai-27b")
    )
    notices = [
        Notice(
            "info",
            "cpu_only",
            f"No GPU telemetry; running {model} on CPU. The orchestrator also serves "
            "utility duty (compression/recall); no worker slots.",
        )
    ]
    if model == "bonsai-8b":
        notices.append(
            Notice(
                "warn",
                "no_27b_resident",
                "RAM < 16 GiB: 8B orchestrator only — vision, tier-2 browsing and "
                "skill/memory writes are unavailable (they require a resident 27B).",
            )
        )
    slot = PlannedSlot(
        slot_id="orchestrator",
        role="orchestrator",
        model=model,
        family=family,
        ctx=ctx,
        gpu=None,
        port=PORT_POOL_START,
        vram_mb=cost.total_mb,
        mmproj=(model == "bonsai-27b"),
    )
    return Loadout(
        planned_at=_now_iso(),
        tier="cpu",
        slots=[slot],
        headroom_mb=max(telemetry.ram_total_mb - cost.total_mb, 0),
        family_configured=family,
        family_effective=family,
        family_degraded=False,
        family_reason=None,
        utility_shared_with_orchestrator=True,
        notices=notices,
    )
