"""Effort ladder (plan-frozen):

effort  thinking_budget_tokens  max tool iterations
low     0                       8
mid     4096                    16
high    12288                   32
xhigh   24576                   48
max     -1 (unlimited)          64

Thinking budgets are always capped at 40% of the slot's context window. Sampling comes
from the model docs: 27B temp 0.7 / top-p 0.95 / top-k 20; 8B/4B/1.7B 0.5 / 0.85 / 20.
"""

from __future__ import annotations

from dataclasses import dataclass

from suiban.config import Effort

_LADDER: dict[Effort, tuple[int, int]] = {
    "low": (0, 8),
    "mid": (4096, 16),
    "high": (12288, 32),
    "xhigh": (24576, 48),
    "max": (-1, 64),
}

EFFORT_LEVELS: tuple[Effort, ...] = ("low", "mid", "high", "xhigh", "max")

THINKING_CTX_FRACTION = 0.4


@dataclass(frozen=True)
class Sampling:
    temperature: float
    top_p: float
    top_k: int


_SAMPLING_27B = Sampling(temperature=0.7, top_p=0.95, top_k=20)
_SAMPLING_SMALL = Sampling(temperature=0.5, top_p=0.85, top_k=20)


def thinking_budget(effort: Effort, slot_ctx: int) -> int:
    """Thinking token budget for an effort level, capped at 40% of the slot context.

    `max` effort (-1 = unlimited upstream) resolves to the cap itself, so a request can
    never think past 40% of its slot's context.
    """
    raw, _ = _LADDER[effort]
    cap = int(slot_ctx * THINKING_CTX_FRACTION)
    if raw < 0:
        return cap
    return min(raw, cap)


def thinking_payload_fields(thinking_budget_tokens: int) -> dict:
    """Request fields that ACTUALLY control thinking on the pinned fork.

    Verified live against tag prism-b9596-9fcaed7 + the Bonsai chat template
    (2026-07-21): a per-request `thinking_budget_tokens` field (from the model docs) is
    ignored by llama-server; the working per-request control is the Qwen-style template
    kwarg `chat_template_kwargs.enable_thinking`. Graded per-request budgets therefore
    degrade to on/off here, and the numeric ceiling is enforced slot-wide via the
    `--reasoning-budget` launch flag (see llama/backend.py).
    TODO(v1.1): restore graded per-request budgets if the fork honors a request field.
    """
    return {"chat_template_kwargs": {"enable_thinking": thinking_budget_tokens != 0}}


def slot_reasoning_budget(slot_ctx: int) -> int:
    """Slot-wide thinking ceiling for the --reasoning-budget launch flag: the xhigh
    budget bounded by 40% of the slot context (also bounds effort=max)."""
    return min(_LADDER["xhigh"][0], int(slot_ctx * THINKING_CTX_FRACTION))


def max_tool_iterations(effort: Effort) -> int:
    return _LADDER[effort][1]


def sampling_for(model: str) -> Sampling:
    """Per-model-card sampling defaults. `model` is a bonsai model id (bonsai-27b...)."""
    return _SAMPLING_27B if model == "bonsai-27b" else _SAMPLING_SMALL
