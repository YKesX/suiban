"""Bind the research engine to the live app: orchestrator slot + browse fetchers.

Kept separate from engine.py so the pipeline stays seam-injected (tests run it on
scripted chat + canned pages) while the app gets a one-call factory.

Fairness choice (documented, deliberate): the job acquires the orchestrator's
SlotGate per pipeline STEP — one completion — not for the whole job. A research run
lasts 15-40 minutes; holding the slot that long would starve every interactive chat
into 300 s timeouts (the pre-refinement live failure). Per-step locking means a chat
waits at most one research completion (bounded by RESEARCH_STEP_TIMEOUT_S), and the
job waits its turn behind queued chats between steps — research latency stretches
under interactive load, which is the right direction to degrade. The job skips the
gate's capacity 429: background work waits, it does not fail.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from suiban.agent.loop import BackendChat
from suiban.config import SearchSettings
from suiban.effort import sampling_for, thinking_budget
from suiban.modes.registry import system_prompt
from suiban.research.engine import FetchFn, ResearchEngine
from suiban.research.jobs import Job
from suiban.search import build_search_provider
from suiban.tools.base import ToolContext
from suiban.tools.browse import BrowseT1Tool, BrowseT2Tool

RESEARCH_STEP_TIMEOUT_S = 300.0


def make_fetch(workdir: Path, *, t2_available: bool) -> FetchFn:
    """Tier-1 fetch with tier-2 fallback (when the capability exists). Failures come
    back as (False, reason) — the engine records them, it never crashes."""
    tier1 = BrowseT1Tool()
    tier2 = BrowseT2Tool() if t2_available else None
    ctx = ToolContext(
        session_id="research", workdir=workdir, role="orchestrator", mode="deep_research"
    )

    async def fetch(url: str) -> tuple[bool, str]:
        result = await tier1.run({"url": url}, ctx)
        if result.status == "ok":
            return True, result.content
        if tier2 is not None:
            t2_result = await tier2.run({"url": url}, ctx)
            if t2_result.status == "ok":
                return True, t2_result.content
        return False, result.content

    return fetch


def make_run_job(
    llama_manager,
    config_home: Path,
    capabilities: dict,
    *,
    search_settings: Callable[[], SearchSettings] | None = None,
    ensure_loaded: Callable[[], Awaitable[bool]] | None = None,
):
    """The JobManager `run_job` callable: builds a fresh engine per job against the
    CURRENT orchestrator slot (slots never change mid-run, but a restart between
    jobs may have replaced the backend object). `search_settings` is read per job so
    an applied settings change takes effect on the next run without a restart.
    `ensure_loaded` warms the lazily-resident loadout before the job needs the
    orchestrator (api.md 2026-07-22c); None (tests) skips the hook."""
    workdir = config_home / "work" / "research"

    async def run_job(job: Job, progress) -> str:
        # Lazy residency (api.md 2026-07-22c): a research job auto-warms the loadout, so
        # a cold server serves a submitted job without a manual load first.
        if ensure_loaded is not None:
            await ensure_loaded()
        slot = llama_manager.slot("orchestrator")
        if slot is None or slot.state != "ready":
            raise RuntimeError(
                f"orchestrator slot is not ready (state: {slot.state if slot else 'missing'}); "
                "deep research needs the orchestrator"
            )
        workdir.mkdir(parents=True, exist_ok=True)
        chat = BackendChat(slot.backend)

        async def complete(payload: dict) -> dict:
            # Per-STEP slot gate (module docstring): held for one completion only,
            # so interactive chats interleave with the running job.
            async with slot.gate.hold():
                return await chat.complete(payload, timeout=RESEARCH_STEP_TIMEOUT_S)

        search_fn = None
        search_provider_name = ""
        if search_settings is not None:
            provider = build_search_provider(search_settings())
            search_fn = provider.search
            search_provider_name = provider.name
        engine = ResearchEngine(
            complete=complete,
            fetch=make_fetch(workdir, t2_available=bool(capabilities.get("browse_t2"))),
            model=slot.model,
            sampling=sampling_for(slot.model),
            thinking_budget_tokens=thinking_budget(job.effort, slot.planned.ctx),  # type: ignore[arg-type]
            search=search_fn,
            search_provider=search_provider_name,
        )
        return await engine.run(job.query, progress, prompt=system_prompt("deep_research"))

    return run_job
