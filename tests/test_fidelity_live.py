"""REAL-MODEL compression-fidelity harness — opt-in, never part of the normal suite.

Run against the LIVE stack (suiban serving real llama-server slots):

    SUIBAN_LIVE_FIDELITY=1 uv run pytest tests/test_fidelity_live.py -q -s

It builds the planted-fact transcript (memory/fidelity.py), asks the resident
UTILITY model (role "utility" from /v1/models; the orchestrator as fallback) to
condense it with the production SUMMARIZE_SYSTEM_PROMPT, and scores fact survival by
plain string containment — asserting >= MIN_SURVIVAL (80%). Per-fact results print
with -s so a failing prompt tune shows exactly which facts died.

Honesty notes: the request goes through /v1/chat/completions (mode chat, effort
low), so the mode prompt is coalesced in front of the summarizer prompt — a slight
difference from the internal summarizer call, which sends the summarizer prompt
alone. The internal path cannot be reached over HTTP, and testing through the public
surface keeps this harness runnable against any live stack. SUIBAN_LIVE_URL
overrides the default base URL (never a repo-baked machine path)."""

from __future__ import annotations

import os

import httpx
import pytest

from suiban.memory import fidelity
from suiban.memory.compression import SUMMARIZE_SYSTEM_PROMPT, wrap_fold_input

pytestmark = pytest.mark.skipif(
    os.environ.get("SUIBAN_LIVE_FIDELITY") != "1",
    reason="live real-model harness; set SUIBAN_LIVE_FIDELITY=1 to run against the live stack",
)

BASE_URL = os.environ.get("SUIBAN_LIVE_URL", "http://127.0.0.1:8686")
REQUEST_TIMEOUT_S = 600.0  # small models on small GPUs take their time


def _utility_model(client: httpx.Client) -> str:
    response = client.get(f"{BASE_URL}/v1/models")
    response.raise_for_status()
    models = response.json()["data"]
    for role in ("utility", "orchestrator"):
        for model in models:
            bonsai = model.get("bonsai") or {}
            if bonsai.get("role") == role and bonsai.get("resident"):
                return model["id"]
    raise AssertionError(f"no resident utility/orchestrator model in {BASE_URL}/v1/models")


def test_live_fact_survival_meets_threshold() -> None:
    facts = fidelity.plant_facts(10)
    conversation = fidelity.synthetic_conversation(facts, filler_turns=30)
    transcript = "\n".join(f"{m['role']}: {m['content']}" for m in conversation)

    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        model = _utility_model(client)
        response = client.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                    {"role": "user", "content": wrap_fold_input(transcript)},
                ],
                "mode": "chat",
                "effort": "low",
                "stream": False,
            },
        )
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"] or ""

    kept = fidelity.surviving_probes(summary, facts)
    rate = fidelity.survival(summary, facts)
    print(f"\nmodel: {model}  survival: {rate:.0%} ({len(kept)}/{len(facts)})")
    for fact in facts:
        marker = "KEPT" if fact.probe in kept else "LOST"
        print(f"  [{marker}] {fact.probe}")
    assert rate >= fidelity.MIN_SURVIVAL, (
        f"real-model fact survival {rate:.0%} < {fidelity.MIN_SURVIVAL:.0%} — "
        f"lost: {[f.probe for f in facts if f.probe not in kept]}; "
        "tune SUMMARIZE_SYSTEM_PROMPT (memory/compression.py) and rerun"
    )
