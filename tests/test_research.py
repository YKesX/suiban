"""Deep research: engine pipeline on canned pages + scripted chat, job store/manager
lifecycle, the /v1/jobs HTTP surface (202/list/status/SSE/report/cancel/429), and the
idle-gated settings apply."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban.app import create_app
from suiban.effort import Sampling
from suiban.errors import BonsaiError
from suiban.research.engine import (
    STAGE_COLLECT,
    STAGE_CROSSCHECK,
    STAGE_PLAN,
    STAGE_WRITE,
    ResearchEngine,
)
from suiban.research.jobs import Job, JobManager, JobStore, new_job_id
from suiban.search import SearchError, SearchResult

SAMPLING = Sampling(temperature=0.7, top_p=0.95, top_k=20)

COARSE_STAGES = {STAGE_PLAN, STAGE_COLLECT, STAGE_CROSSCHECK, STAGE_WRITE}

CANNED_PAGES = {
    "https://a.example/paper": "Alpha paper: the sky is blue due to Rayleigh scattering.",
    "https://b.example/docs": "Beta docs: scattering strength scales with 1/lambda^4.",
}


class ScriptedComplete:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.payloads: list[dict] = []

    async def __call__(self, payload: dict) -> dict:
        self.payloads.append(payload)
        if not self._responses:
            raise AssertionError("scripted complete ran out of responses")
        return self._responses.pop(0)


def text_response(text: str) -> dict:
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5},
    }


async def canned_fetch(url: str) -> tuple[bool, str]:
    if url in CANNED_PAGES:
        return True, CANNED_PAGES[url]
    return False, f"HTTP 404 for {url}"


def make_engine(complete) -> ResearchEngine:
    return ResearchEngine(
        complete=complete,
        fetch=canned_fetch,
        model="bonsai-27b",
        sampling=SAMPLING,
        thinking_budget_tokens=0,
    )


class ProgressRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    async def __call__(self, stage: str, percent: int) -> None:
        self.events.append((stage, percent))


# -- engine pipeline ----------------------------------------------------------
async def test_engine_pipeline_on_canned_pages() -> None:
    plan_json = json.dumps(
        {
            "subquestions": ["why is the sky blue?"],
            "sources": [
                {"url": "https://a.example/paper"},
                {"url": "https://b.example/docs"},
                {"url": "https://c.example/missing"},
            ],
        }
    )
    complete = ScriptedComplete(
        [
            text_response(plan_json),
            text_response("cross-check: both sources agree; the third is unavailable."),
            text_response("# Report\n\nThe sky is blue because of Rayleigh scattering."),
        ]
    )
    progress = ProgressRecorder()
    report = await make_engine(complete).run(
        "why is the sky blue?", progress, prompt="research prompt"
    )

    assert report.startswith("# Report")
    stages = [s for s, _ in progress.events]
    assert set(stages) <= COARSE_STAGES  # coarse strings only — no internals
    assert stages[0] == STAGE_PLAN
    assert STAGE_COLLECT in stages and STAGE_CROSSCHECK in stages
    assert stages[-1] == STAGE_WRITE
    percents = [p for _, p in progress.events]
    assert percents == sorted(percents)  # monotone
    assert percents[-1] == 100
    # The gathered source text reached the cross-check step; the failed fetch was
    # recorded honestly.
    crosscheck_payload = complete.payloads[1]["messages"][-1]["content"]
    assert "Rayleigh scattering" in crosscheck_payload
    assert "UNAVAILABLE" in crosscheck_payload


async def test_engine_plan_failure_degrades_to_bare_query() -> None:
    complete = ScriptedComplete(
        [
            text_response("not json"),  # plan
            text_response("still bad"),  # repair 1
            text_response("no"),  # repair 2
            text_response("cross-check with no sources"),
            text_response("# Report\n\nNo sources could be retrieved; honesty section."),
        ]
    )
    progress = ProgressRecorder()
    report = await make_engine(complete).run("obscure question", progress, prompt="p")
    assert "No sources" in report
    # The report step was told the honest coverage.
    report_payload = complete.payloads[-1]["messages"][-1]["content"]
    assert "No sources could be planned or retrieved" in report_payload


async def test_engine_empty_report_raises() -> None:
    complete = ScriptedComplete(
        [
            text_response(json.dumps({"subquestions": ["q"], "sources": []})),
            text_response("analysis"),
            text_response(""),  # empty report -> the job must FAIL, not complete
        ]
    )
    with pytest.raises(RuntimeError):
        await make_engine(complete).run("q", ProgressRecorder(), prompt="p")


# -- engine gather via web search (additive 2026-07-21c) ----------------------
def _search_engine(complete, search) -> ResearchEngine:
    return ResearchEngine(
        complete=complete,
        fetch=canned_fetch,
        model="bonsai-27b",
        sampling=SAMPLING,
        thinking_budget_tokens=0,
        search=search,
        search_provider="duckduckgo",
    )


async def test_engine_gather_uses_search_results_for_sources_and_citations() -> None:
    plan_json = json.dumps(
        {
            "subquestions": ["why is the sky blue?", "how does wavelength matter?"],
            "sources": [{"url": "https://model-proposed.example/ignored"}],
        }
    )
    complete = ScriptedComplete(
        [
            text_response(plan_json),
            text_response("cross-check over searched sources"),
            text_response("# Report\n\nCited from searched sources."),
        ]
    )
    queries: list[tuple[str, int]] = []

    async def search(query: str, count: int) -> list[SearchResult]:
        queries.append((query, count))
        return [
            SearchResult(title="Alpha paper", url="https://a.example/paper", snippet="s"),
            SearchResult(title="Beta docs", url="https://b.example/docs", snippet="s"),
        ]

    progress = ProgressRecorder()
    report = await _search_engine(complete, search).run(
        "why is the sky blue?", progress, prompt="p"
    )

    # Sub-questions became the search queries; coarse stages only, still monotone.
    assert [q for q, _ in queries] == ["why is the sky blue?", "how does wavelength matter?"]
    stages = [s for s, _ in progress.events]
    assert set(stages) <= COARSE_STAGES
    percents = [p for _, p in progress.events]
    assert percents == sorted(percents)

    # The searched URLs (deduped) fed the gather + citations; the model-proposed URL
    # was not fetched; no degrade note — search worked.
    crosscheck_payload = complete.payloads[1]["messages"][-1]["content"]
    assert "https://a.example/paper" in crosscheck_payload
    assert "(Alpha paper)" in crosscheck_payload  # search titles reach the digest
    assert "model-proposed.example" not in crosscheck_payload
    assert "Rayleigh scattering" in crosscheck_payload  # fetched page content
    assert not report.startswith(">")
    assert report.startswith("# Report")


async def test_engine_search_total_failure_degrades_with_report_note() -> None:
    plan_json = json.dumps(
        {
            "subquestions": ["why is the sky blue?"],
            "sources": [{"url": "https://a.example/paper"}],
        }
    )
    complete = ScriptedComplete(
        [
            text_response(plan_json),
            text_response("cross-check over planned sources"),
            text_response("# Report\n\nFrom planned sources only."),
        ]
    )

    async def search(query: str, count: int) -> list[SearchResult]:
        raise SearchError("duckduckgo request failed: connection refused")

    report = await _search_engine(complete, search).run(
        "why is the sky blue?", ProgressRecorder(), prompt="p"
    )
    # Honest degrade: the note is prepended mechanically, the plan URLs were fetched.
    assert report.startswith("> **Note:** web search (duckduckgo) was unavailable")
    assert "connection refused" in report.splitlines()[0]
    assert "# Report" in report
    crosscheck_payload = complete.payloads[1]["messages"][-1]["content"]
    assert "https://a.example/paper" in crosscheck_payload


async def test_engine_search_empty_results_falls_back_too() -> None:
    plan_json = json.dumps({"subquestions": ["q1"], "sources": [{"url": "https://b.example/docs"}]})
    complete = ScriptedComplete(
        [
            text_response(plan_json),
            text_response("cross-check"),
            text_response("# Report\n\nBody."),
        ]
    )

    async def search(query: str, count: int) -> list[SearchResult]:
        return []

    report = await _search_engine(complete, search).run("q1", ProgressRecorder(), prompt="p")
    assert "search returned no results" in report.splitlines()[0]
    assert "https://b.example/docs" in complete.payloads[1]["messages"][-1]["content"]


async def test_engine_without_search_seam_behaves_as_before_no_note() -> None:
    plan_json = json.dumps({"subquestions": ["q"], "sources": [{"url": "https://a.example/paper"}]})
    complete = ScriptedComplete(
        [text_response(plan_json), text_response("cc"), text_response("# Report\n\nBody.")]
    )
    report = await make_engine(complete).run("q", ProgressRecorder(), prompt="p")
    assert report.startswith("# Report")  # no degrade note: unsearched IS configured


# -- job store ----------------------------------------------------------------
def _job(job_id: str | None = None, state: str = "queued") -> Job:
    return Job(
        id=job_id or new_job_id(),
        type="deep_research",
        query="q",
        effort="high",
        state=state,
        stage=None,
        percent=0,
        created_at="2026-07-21T00:00:00Z",
        started_at=None,
        finished_at=None,
        error=None,
    )


def test_job_store_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "db.sqlite")
    job = _job()
    store.add(job)
    assert store.get(job.id).state == "queued"
    assert store.active_count() == 1
    updated = store.update(job.id, state="running", stage="collecting sources", percent=30)
    assert (updated.state, updated.stage, updated.percent) == ("running", "collecting sources", 30)
    orphaned = store.fail_orphans("restarted")
    assert orphaned == 1
    assert store.get(job.id).state == "failed"
    assert store.active_count() == 0
    store.close()


# -- job manager --------------------------------------------------------------
async def test_manager_lifecycle_and_single_job_limit(tmp_path: Path) -> None:
    release = asyncio.Event()
    seen: list[dict] = []

    async def slow_run(job: Job, progress) -> str:
        await progress("collecting sources", 25)
        await release.wait()
        return "# report md"

    manager = JobManager(
        JobStore(tmp_path / "db.sqlite"), run_job=slow_run, reports_dir=tmp_path / "reports"
    )
    terminal: list[Job] = []
    manager.add_listener(terminal.append)

    job = manager.submit("q1", "high")
    assert job.state == "queued"
    queue = manager.subscribe(job.id)
    with pytest.raises(BonsaiError) as excinfo:
        manager.submit("q2", "high")
    assert excinfo.value.status == 429  # max 1 research job in v1

    await asyncio.sleep(0.05)  # let the task start and progress
    assert manager.get(job.id).state == "running"
    release.set()
    for _ in range(100):
        await asyncio.sleep(0.01)
        if manager.get(job.id).state == "completed":
            break
    final = manager.get(job.id)
    assert final.state == "completed"
    assert final.percent == 100 and final.stage is None
    assert manager.report_path(job.id).read_text(encoding="utf-8") == "# report md"
    assert [j.state for j in terminal] == ["completed"]

    while not queue.empty():
        seen.append(queue.get_nowait())
    kinds = [(e["type"], e.get("state") or e.get("stage")) for e in seen]
    assert ("state", "running") in kinds
    assert ("progress", "collecting sources") in kinds
    assert kinds[-1] == ("state", "completed")

    # A fresh submit works now, and cancel is idempotent.
    job2 = manager.submit("q3", "high")
    cancelled = await manager.cancel(job2.id)
    assert cancelled.state == "cancelled"
    # cancel() awaited the unwind: the task is gone by the time it returned.
    assert manager.get(job2.id).state == "cancelled"
    assert (await manager.cancel(job2.id)).state == "cancelled"  # idempotent
    # Cancelling a COMPLETED job keeps the truthful terminal state.
    assert (await manager.cancel(job.id)).state == "completed"
    await manager.shutdown()


# -- cancel unwinding (deep-detail pass) --------------------------------------
async def test_cancel_awaits_unwind_and_aborts_in_flight_call(tmp_path: Path) -> None:
    """DELETE semantics: cancel() returns only after the pipeline task unwound —
    the in-flight backend call (the sleep below stands in for an httpx request to
    llama-server) received its CancelledError and teardown completed."""
    started = asyncio.Event()
    aborted = asyncio.Event()
    unwound = asyncio.Event()

    async def run_job(job: Job, progress) -> str:
        await progress("collecting sources", 10)
        started.set()
        try:
            await asyncio.sleep(60)  # the in-flight "httpx call"
        except asyncio.CancelledError:
            aborted.set()
            await asyncio.sleep(0.05)  # teardown work during the unwind
            unwound.set()
            raise
        return "# never"

    manager = JobManager(
        JobStore(tmp_path / "db.sqlite"), run_job=run_job, reports_dir=tmp_path / "reports"
    )
    job = manager.submit("q", "high")
    await asyncio.wait_for(started.wait(), 5)
    result = await manager.cancel(job.id)
    assert result.state == "cancelled"
    assert aborted.is_set(), "the in-flight call must be aborted via task cancellation"
    assert unwound.is_set(), "cancel() must not return before the unwind finished"
    assert manager.get(job.id).state == "cancelled"
    await manager.shutdown()


async def test_submit_during_wedged_unwind_is_429_then_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No double-use: while a cancelled task is still unwinding (here: wedged past
    the bounded cancel wait), a new submit 429s; once the task actually finishes,
    submits work again."""
    monkeypatch.setattr("suiban.research.jobs.CANCEL_UNWIND_TIMEOUT_S", 0.1)
    release = asyncio.Event()
    entered = asyncio.Event()

    async def run_job(job: Job, progress) -> str:
        if release.is_set():
            return "# second job"
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            # Wedged teardown: swallow repeated cancels until the test releases it.
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(0.01)
            raise
        return "# never"

    manager = JobManager(
        JobStore(tmp_path / "db.sqlite"), run_job=run_job, reports_dir=tmp_path / "reports"
    )
    job = manager.submit("q", "high")
    await asyncio.wait_for(entered.wait(), 5)
    result = await manager.cancel(job.id)  # returns after the bounded wait expires
    assert result.state == "cancelled"

    with pytest.raises(BonsaiError) as excinfo:
        manager.submit("q2", "high")
    assert excinfo.value.status == 429
    assert excinfo.value.code == "research_job_active"

    release.set()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if not manager._tasks:
            break
    assert not manager._tasks, "the wedged task must eventually unwind and deregister"
    job2 = manager.submit("q2", "high")  # admitted again
    for _ in range(200):
        await asyncio.sleep(0.01)
        if manager.get(job2.id).terminal:
            break
    assert manager.get(job2.id).state == "completed"
    await manager.shutdown()


# -- HTTP surface (mock backend end to end) -----------------------------------
def _wait_for_state(client: TestClient, job_id: str, state: str, deadline_s: float = 10.0) -> dict:
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        body = client.get(f"/v1/jobs/{job_id}").json()
        if body["state"] == state:
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {state}: {body}")


def test_job_http_lifecycle_completes_on_mock(client: TestClient, bonsai_home: Path) -> None:
    created = client.post(
        "/v1/jobs", json={"type": "deep_research", "query": "what is bonsai?", "effort": "high"}
    )
    assert created.status_code == 202
    job_id = created.json()["id"]

    status = _wait_for_state(client, job_id, "completed")
    assert status["percent"] == 100
    assert status["error"] is None

    # Internals never leak: the status is exactly the JobStatus shape and any stage
    # ever exposed is one of the coarse strings.
    assert set(status) == {
        "id",
        "type",
        "query",
        "state",
        "stage",
        "percent",
        "created_at",
        "started_at",
        "finished_at",
        "error",
    }

    report = client.get(f"/v1/jobs/{job_id}/report")
    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert report.text  # mock-synthesized markdown
    assert (bonsai_home / "reports" / f"{job_id}.md").is_file()

    # SSE on a terminal job: immediate state event, then the stream closes.
    with client.stream("GET", f"/v1/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    events = [json.loads(line[len("data: ") :]) for line in lines]
    assert events[-1] == {"type": "state", "state": "completed"}

    listing = client.get("/v1/jobs").json()["jobs"]
    assert listing[0]["id"] == job_id  # newest first


def test_job_http_running_report_404_429_cancel_and_idle_apply(
    bonsai_home: Path, telemetry_24gb, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def slow_run(job, progress):
        await progress("collecting sources", 20)
        await asyncio.sleep(60)
        return "# never"

    monkeypatch.setattr("suiban.app.make_run_job", lambda *args, **kwargs: slow_run)
    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as client:
        job_id = client.post("/v1/jobs", json={"type": "deep_research", "query": "slow"}).json()[
            "id"
        ]
        _wait_for_state(client, job_id, "running")

        # report is 404 until completed
        report = client.get(f"/v1/jobs/{job_id}/report")
        assert report.status_code == 404
        assert report.json()["error"]["type"] == "not_found_error"

        # concurrency: one research job in v1
        second = client.post("/v1/jobs", json={"type": "deep_research", "query": "another"})
        assert second.status_code == 429
        assert second.json()["error"]["type"] == "overloaded_error"

        # /v1/system reflects the running job; apply is deferred while busy
        assert client.get("/v1/system").json()["jobs_active"] == 1
        client.patch("/v1/settings", json={"kv": {"preset": "aggressive"}})
        applied = client.post("/v1/system/apply").json()
        assert applied["applied"] is False
        # kv reports requires_restart now (api.md 2026-07-21d); the COMMIT to
        # config.toml still waits for idle like any deferred apply.
        assert applied["requires_restart"] == ["kv"]
        assert applied["pending_until_idle"] == []
        assert client.get("/v1/settings").json()["current"]["kv"]["preset"] == "recommended"

        # cancel (idempotent), then the deferred apply commits at idle
        first = client.delete(f"/v1/jobs/{job_id}")
        assert first.status_code == 200
        assert first.json() == {"id": job_id, "state": "cancelled"}
        _wait_for_state(client, job_id, "cancelled")
        assert client.delete(f"/v1/jobs/{job_id}").json()["state"] == "cancelled"

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if client.get("/v1/settings").json()["current"]["kv"]["preset"] == "aggressive":
                break
            time.sleep(0.02)
        assert client.get("/v1/settings").json()["current"]["kv"]["preset"] == "aggressive"
        assert client.get("/v1/settings").json()["staged"] is None


def test_research_job_locks_per_step_and_chats_interleave(
    bonsai_home: Path, telemetry_24gb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fairness choice (research/wiring.py docstring): a running job holds the
    orchestrator gate per pipeline STEP only. A chat issued mid-job completes while
    the job is STILL running — with whole-job locking it would have waited for the
    terminal state. Also asserts the gate really is held during job completions."""
    from suiban.agent.loop import BackendChat

    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as client:
        # Lazy residency (api.md 2026-07-22c): warm the loadout so the orchestrator gate
        # exists to observe, THEN patch the backend and run the job (which no-op-reloads).
        client.post(
            "/v1/chat/completions",
            json={"model": "bonsai-auto", "messages": [{"role": "user", "content": "warm"}]},
        )
        gate = client.app.state.bonsai.manager.slot("orchestrator").gate

        gate_seen_busy: list[bool] = []
        original = BackendChat.complete

        async def slow(self, payload: dict, timeout: float) -> dict:
            gate_seen_busy.append(gate.busy)
            await asyncio.sleep(0.2)
            return await original(self, payload, timeout)

        monkeypatch.setattr(BackendChat, "complete", slow)

        job_id = client.post(
            "/v1/jobs", json={"type": "deep_research", "query": "fairness check"}
        ).json()["id"]
        _wait_for_state(client, job_id, "running")

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "bonsai-auto",
                "messages": [{"role": "user", "content": "quick question"}],
                "effort": "low",
            },
        )
        assert resp.status_code == 200
        # The chat finished while the job was still mid-pipeline: step-level locking.
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "running"
        _wait_for_state(client, job_id, "completed")
    # Research completions ran with the orchestrator gate held (serialized).
    assert gate_seen_busy and all(gate_seen_busy)


def test_job_validation_errors(client: TestClient) -> None:
    assert client.post("/v1/jobs", json={"type": "deep_research"}).status_code == 400
    assert client.post("/v1/jobs", json={"type": "deep_research", "query": " "}).status_code == 400
    resp = client.post("/v1/jobs", json={"type": "deep_research", "query": "q", "effort": "turbo"})
    assert resp.status_code == 400
    assert client.get("/v1/jobs/job_missing").status_code == 404
    assert client.delete("/v1/jobs/job_missing").status_code == 404
    assert client.get("/v1/jobs/job_missing/events").status_code == 404
