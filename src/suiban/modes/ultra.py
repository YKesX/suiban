"""Ultra mode: orchestrator-planned multi-agent execution, coordinated with SLAP.

Flow: the orchestrator produces a grammar-constrained decomposition plan (json_schema
response_format → GBNF in llama-server), each sub-task runs as a short-lived contained
sub-agent on a worker slot — fresh context, focused toolset (WORKER_TOOLSET),
structured result envelope — and the orchestrator synthesizes the final answer from
the collected results. With no ready worker slots (capabilities.ultra_parallel =
false) sub-tasks run sequentially on the orchestrator slot with the SAME containment,
plus an `ultra_sequential` notice.

Dispatch is represented as SLAP messages (api.md §12; vendored protocol in
suiban/slap/): each executing slot advertises a `capability`, the orchestrator emits an
`assign` per sub-task, each worker returns a `result`, and synthesis emits a `decide`.
Every message is validated against the vendored schemas before use; a validation failure
degrades gracefully — the run falls back to the structured-dict path and emits a
`slap_degraded` notice, never crashing. The validated sequence is persisted per run (an
in-process store, not memory) so `GET /v1/slap/trace/{session_id}` can return it.

VOLATILE PER-AGENT SYSTEM PROMPTS: the plan may carry a per-sub-task `system_prompt` the
orchestrator writes on the fly. When present (and its `assign` validates) it is the
worker's system message for that ONE agent lifetime; it is discarded when the agent
finishes — never reused across agents, never archived to memory, and stripped from the
`assign` before it is recorded in the trace. The static ultra_worker.md prompt is the
FALLBACK used whenever the orchestrator omits a system prompt (or its assign degrades).

Latency is bounded, not hoped for: sub-tasks inherit the REQUEST effort (capped at
"mid" on sequential tiers — a single slot at 15 tok/s cannot afford xhigh thinking per
sub-task), the sub-task count is capped (worker count when parallel, 3 sequential),
and every sub-task runs under a wall-clock timeout scaled by its effort. A timed-out
or crashed sub-agent becomes a structured failure (`agent_result` status "failed" plus
a notice) that the orchestrator synthesizes around; every degrade on this path emits a
notice with a one-line reason.

Containment is structural: sub-agent registries come from build_worker_registry(), so
memory-write and skill tools are never in the schema a worker's grammar can decode,
their ToolContext role is "worker" (the memory service re-checks it), and worker
internals (tool chatter, drafts) are never streamed — clients see only the api.md
`agent_spawn` / `agent_result` events (which now carry the SLAP `task_id`).

Every stage degrades gracefully: an unparseable plan falls back to a single contained
sub-task (with a notice), a crashed sub-agent becomes a failed agent_result, and the
synthesis step reports failures honestly instead of hiding them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx

from suiban import slap
from suiban.agent import events
from suiban.agent.events import AgentEvent
from suiban.agent.loop import AgentLoop, tool_result_cap
from suiban.agent.structured import request_structured
from suiban.config import Effort
from suiban.effort import (
    EFFORT_LEVELS,
    max_tool_iterations,
    sampling_for,
    thinking_budget,
    thinking_payload_fields,
)
from suiban.memory.compression import message_text
from suiban.modes.registry import PROMPTS_DIR, system_prompt
from suiban.tools.base import ToolContext
from suiban.tools.registry import ToolRegistry

DEFAULT_STEP_TIMEOUT_S = 300.0

# Sub-task count caps: with workers each sub-task gets its own slot, so the honest cap
# is the worker count; sequentially every extra sub-task is a full extra pass on ONE
# slot, so the cap is fixed at 3 (measured live: an uncapped sequential Ultra ran a
# trivial question past 10 minutes — ranked gap #1 of the refinement pass).
SEQUENTIAL_MAX_SUBTASKS = 3
# Sequential sub-tasks never think above "mid": xhigh thinking per sub-task on a
# single 15 tok/s slot multiplies wall-clock time without a parallelism payoff.
SEQUENTIAL_EFFORT_CAP: Effort = "mid"
# Per-sub-task wall-clock budgets (the whole contained sub-agent run, tools included).
SUBTASK_TIMEOUT_LOW_S = 240.0  # low / mid effort
SUBTASK_TIMEOUT_HIGH_S = 480.0  # high / xhigh / max effort

# The SLAP root task id for a run: the whole request. Sub-tasks are T1, T2, ... under it.
ROOT_TASK_ID = "T0"


def worker_effort(request_effort: Effort, *, parallel: bool) -> Effort:
    """Sub-task effort: inherit the request effort; cap at SEQUENTIAL_EFFORT_CAP when
    sub-tasks share the orchestrator slot (no worker slots in the loadout)."""
    if parallel:
        return request_effort
    if EFFORT_LEVELS.index(request_effort) > EFFORT_LEVELS.index(SEQUENTIAL_EFFORT_CAP):
        return SEQUENTIAL_EFFORT_CAP
    return request_effort


def subtask_timeout_s(effort: Effort) -> float:
    """Wall-clock budget for one contained sub-agent run, by its effective effort."""
    return SUBTASK_TIMEOUT_LOW_S if effort in ("low", "mid") else SUBTASK_TIMEOUT_HIGH_S


def git_base_revision(workdir: Path | None) -> str | None:
    """A `git:<branch>:<sha>` base-revision reference when `workdir` is a git repo, else
    None. Pure filesystem (reads .git/HEAD + the ref) — no subprocess, always bounded."""
    if workdir is None:
        return None
    git_dir = workdir / ".git"
    if not git_dir.is_dir():
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref:"):
        ref = head[4:].strip()
        branch = ref.rsplit("/", 1)[-1]
        try:
            sha = (git_dir / ref).read_text(encoding="utf-8").strip()
        except OSError:
            return f"git:{branch}"  # unborn / packed ref: the branch alone is honest
        return f"git:{branch}:{sha[:12]}"
    return f"git:{head[:12]}"  # detached HEAD holds the sha directly


PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1},
                    "brief": {"type": "string", "minLength": 1},
                    # Volatile per-agent system prompt the orchestrator writes for THIS
                    # sub-task. Optional: when omitted, the worker falls back to the
                    # static ultra_worker.md prompt.
                    "system_prompt": {"type": "string", "minLength": 1},
                },
                "required": ["title", "brief"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}

WORKER_RESULT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok", "failed"]},
        "summary": {"type": "string", "minLength": 1},
        "output": {"type": "string"},
    },
    "required": ["status", "summary", "output"],
    "additionalProperties": False,
}


def plan_instruction(cap: int) -> str:
    return (
        f"Decompose the task above into 1-{cap} independent sub-tasks for contained "
        "worker models. Each sub-task needs a short title, a context-complete brief "
        "(every fact, file path, constraint and deliverable format pasted in — the "
        "worker sees nothing but its brief), and a crisp task-scoped system_prompt: a "
        "few lines that put THIS worker in the right role for THIS sub-task (its "
        "expertise, standards, and what 'done' means), written fresh for this task and "
        "used only for this one agent. Keep judgment and cross-cutting work for "
        "yourself; delegate only self-contained execution."
    )


_RESULT_INSTRUCTION = (
    'Report your result for this sub-task now. status is "ok" only if the deliverable is '
    'complete and you verified it against the brief, otherwise "failed". summary is one '
    "honest line. output is the full deliverable (or what exists of it). Use only facts "
    "from your work above."
)


@lru_cache(maxsize=1)
def worker_prompt() -> str:
    """The ultra_worker system prompt (versioned file, like every mode prompt). Used as
    the FALLBACK when a sub-task carries no volatile system prompt."""
    return (PROMPTS_DIR / "ultra_worker.md").read_text(encoding="utf-8")


@dataclass
class UltraWorker:
    """A dispatch target: a slot's model + an OpenAI-compatible chat seam, plus the
    capability facts it advertises over SLAP (model/family/quant/ctx/backend/workload)."""

    slot_id: str
    model: str
    ctx: int
    chat: Any  # BackendChat-like: async complete(payload, timeout) -> dict
    family: str = "bonsai"
    quant: str = ""
    backend: str = ""
    workload: float = 0.0


@dataclass
class WorkerResult:
    status: str  # "ok" | "failed"
    summary: str
    output: str


@dataclass
class _Subtask:
    title: str
    brief: str
    agent_id: str
    task_id: str
    system_prompt: str | None = None
    result: WorkerResult | None = field(default=None)


class UltraRun:
    """One Ultra request. Duck-types AgentLoop's surface for the chat router:
    `run()` yields stream_events; afterwards final_text / finish_reason /
    total_usage / tool_messages hold the aggregate. `slap_messages` holds the validated
    SLAP transcript (also handed to `trace_sink` for /v1/slap/trace)."""

    def __init__(
        self,
        *,
        orchestrator: UltraWorker,
        workers: list[UltraWorker],
        registry_factory: Callable[[], ToolRegistry],
        tool_ctx_factory: Callable[[], ToolContext],
        messages: list[dict],
        effort: Effort = "high",
        max_tokens: int | None = None,
        step_timeout_s: float = DEFAULT_STEP_TIMEOUT_S,
        session_id: str = "ultra",
        workdir: Path | None = None,
        trace_sink: Callable[[str, list[dict]], None] | None = None,
        slap_enabled: bool = True,
    ) -> None:
        self._orch = orchestrator
        self._workers = workers
        self._registry_factory = registry_factory
        self._tool_ctx_factory = tool_ctx_factory
        self._messages = list(messages)
        self._effort = effort
        self._max_tokens = max_tokens
        self._step_timeout = step_timeout_s
        self._parallel = bool(workers)
        self._worker_effort = worker_effort(effort, parallel=self._parallel)
        self._subtask_cap = len(workers) if self._parallel else SEQUENTIAL_MAX_SUBTASKS
        self._subtask_timeout = subtask_timeout_s(self._worker_effort)

        # -- SLAP trace state -------------------------------------------------
        # slap_enabled (api.md 2026-07-22c): off routes dispatch through the plain
        # structured-dict path — no SLAP messages are built, validated, or recorded, so
        # slap_messages stays empty and /v1/slap/trace serves nothing for this run. The
        # dispatch itself (plan → contained sub-agents → synthesis) is unchanged.
        self._slap_enabled = slap_enabled
        self._session_id = session_id
        self._workdir = workdir
        self._trace_sink = trace_sink
        self.slap_messages: list[dict] = []
        self._message_counter = 0

        self.final_text: str = ""
        self.finish_reason: str = "stop"
        self.error_message: str | None = None
        self.total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thinking_tokens": 0,
        }
        self.tool_messages: list[dict] = []  # ultra archives only the synthesis

    # -- helpers -----------------------------------------------------------
    async def _orch_complete(self, payload: dict) -> dict:
        return await self._orch.chat.complete(payload, timeout=self._step_timeout)

    def _merge_usage(self, usage: dict) -> None:
        for key in self.total_usage:
            self.total_usage[key] += int(usage.get(key, 0))

    def _task_messages(self) -> list[dict]:
        return [m for m in self._messages if m.get("role") != "system"]

    def _orch_payload(self, messages: list[dict], *, thinking: int) -> dict:
        sampling = sampling_for(self._orch.model)
        return {
            "model": self._orch.model,
            "messages": messages,
            "stream": False,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            **thinking_payload_fields(thinking),
        }

    # -- SLAP plumbing -----------------------------------------------------
    def _next_message_id(self) -> str:
        self._message_counter += 1
        return f"M{self._message_counter}"

    def _record_slap(
        self, message: dict, *, kind: str, record: dict | None = None
    ) -> AgentEvent | None:
        """Validate a SLAP message against the vendored schema. On success, append it
        (or a redacted `record` variant) to the trace and return None. On failure,
        record nothing and return a `slap_degraded` notice so the caller can surface the
        graceful fallback — the run never crashes on an invalid message.

        When SLAP is disabled (settings.slap.enabled=false) this is a no-op: nothing is
        built into the trace, and there is no degrade notice — SLAP is off by choice, not
        broken."""
        if not self._slap_enabled:
            return None
        errors = slap.validate_message(message)
        if errors:
            return events.notice(
                "warn",
                "slap_degraded",
                f"SLAP {kind} message failed schema validation ({errors[0]}); "
                "continuing on the structured path.",
            )
        self.slap_messages.append(record if record is not None else message)
        return None

    def _flush_trace(self) -> None:
        if self._trace_sink is not None:
            self._trace_sink(self._session_id, self.slap_messages)

    def _advertise_capabilities(self, slots: list[UltraWorker]) -> list[AgentEvent]:
        """Each executing slot advertises a SLAP `capability` (recorded in the trace).
        Routing is unchanged for v1 — this is the measured-capability record."""
        notices: list[AgentEvent] = []
        for worker in slots:
            capability = slap.build_capability(
                message_id=self._next_message_id(),
                task_id=ROOT_TASK_ID,
                model=worker.model,
                model_family=worker.family or None,
                quantization=worker.quant or None,
                context_limit=worker.ctx,
                backend=worker.backend or None,
                current_workload=worker.workload,
            )
            notice = self._record_slap(capability, kind="capability")
            if notice is not None:
                notices.append(notice)
        return notices

    def _dispatch_assign(self, subtask: _Subtask) -> tuple[str, AgentEvent | None]:
        """Build + validate + record the `assign` for one sub-task; return the system
        prompt the worker should use and an optional degrade notice.

        The volatile per-agent system prompt (when present and the assign validates) is
        the worker's system message; it is STRIPPED from the copy recorded in the trace,
        so it lives only for this agent's lifetime. On a validation failure, or when the
        plan omitted a system prompt, the worker falls back to the static ultra_worker
        prompt.

        With SLAP disabled (settings.slap.enabled=false) no assign is built or recorded;
        the worker still gets its volatile per-agent prompt (or the static fallback), so
        the plain structured-dict dispatch keeps the plan's per-agent roles."""
        if not self._slap_enabled:
            return subtask.system_prompt or worker_prompt(), None
        assign = slap.build_assign(
            message_id=self._next_message_id(),
            task_id=subtask.task_id,
            parent_task=ROOT_TASK_ID,
            role="worker",
            goal=subtask.title,
            base_revision=git_base_revision(self._workdir),
            expected_artifacts=["result"],
            system_prompt=subtask.system_prompt,
            limits={
                "maximum_rounds": max_tool_iterations(self._worker_effort),
                "maximum_seconds": int(self._subtask_timeout),
            },
        )
        # Redact the volatile system prompt before it is ever recorded/persisted.
        redacted = {k: v for k, v in assign.items() if k != "system_prompt"}
        notice = self._record_slap(assign, kind="assign", record=redacted)
        if notice is None and assign.get("system_prompt"):
            return assign["system_prompt"], None
        return worker_prompt(), notice

    def _record_result(self, subtask: _Subtask) -> AgentEvent | None:
        """Build + validate + record the worker's SLAP `result` message from its
        structured WorkerResult (every claim carries an evidence reference)."""
        result = subtask.result or WorkerResult("failed", "never ran", "")
        completed = result.status == "ok"
        artifact_ref = f"result:{subtask.task_id}"
        artifacts = [artifact_ref] if result.output else None
        claims = None
        if completed and result.summary:
            claims = [{"claim": result.summary, "evidence": [artifact_ref]}]
        risks = None
        if not completed:
            risks = [
                f"sub-task '{subtask.title}' did not complete; synthesis proceeds "
                "without this piece"
            ]
        message = slap.build_result(
            message_id=self._next_message_id(),
            task_id=subtask.task_id,
            parent_task=ROOT_TASK_ID,
            status="completed" if completed else "failed",
            artifacts=artifacts,
            claims=claims,
            risks=risks,
            summary=result.summary or None,
            # Coarse status-derived signal, not a measured score.
            confidence=0.8 if completed else 0.1,
        )
        return self._record_slap(message, kind="result")

    def _record_decide(self, subtasks: list[_Subtask]) -> AgentEvent | None:
        """Synthesis emits a SLAP `decide`: which sub-task artifacts were accepted."""
        accepted = [f"result:{s.task_id}" for s in subtasks if s.result and s.result.status == "ok"]
        rejected = [
            f"result:{s.task_id}" for s in subtasks if not (s.result and s.result.status == "ok")
        ]
        message = slap.build_decide(
            message_id=self._next_message_id(),
            task_id=ROOT_TASK_ID,
            decision="accept" if accepted else "escalate",
            accepted_artifacts=accepted or None,
            rejected_artifacts=rejected or None,
            reason=(
                f"{len(accepted)}/{len(subtasks)} sub-task artifact(s) accepted; "
                "synthesized the final deliverable from the verified pieces."
            ),
        )
        return self._record_slap(message, kind="decide")

    def _spawn_event(self, subtask: _Subtask, model: str) -> AgentEvent:
        """agent_spawn carrying the SLAP task_id (api.md §12)."""
        base = events.agent_spawn(subtask.agent_id, model, subtask.title, self._worker_effort)
        return AgentEvent(base.type, {**base.payload, "task_id": subtask.task_id})

    def _result_event(self, subtask: _Subtask, result: WorkerResult) -> AgentEvent:
        """agent_result carrying the SLAP task_id (api.md §12)."""
        base = events.agent_result(subtask.agent_id, result.status, result.summary)
        return AgentEvent(base.type, {**base.payload, "task_id": subtask.task_id})

    # -- stage 1: plan -----------------------------------------------------
    async def _plan(self) -> tuple[list[_Subtask], list[AgentEvent]]:
        messages = [
            {"role": "system", "content": system_prompt("ultra")},
            *self._task_messages(),
            {"role": "user", "content": plan_instruction(self._subtask_cap)},
        ]
        # Structured calls run with thinking off: the grammar constrains the output
        # and a deterministic emission is worth more than deliberation here.
        result = await request_structured(
            self._orch_complete,
            self._orch_payload(messages, thinking=0),
            PLAN_SCHEMA,
            schema_name="ultra_plan",
        )
        self._merge_usage(result.usage)
        raw = (result.data or {}).get("subtasks") or []
        if raw:
            notices: list[AgentEvent] = []
            if len(raw) > self._subtask_cap:
                notices.append(
                    events.notice(
                        "info",
                        "ultra_plan_truncated",
                        f"Plan produced {len(raw)} sub-tasks; running the first "
                        f"{self._subtask_cap} (the cap for this loadout) to keep the "
                        "run bounded.",
                    )
                )
            subtasks = [
                _Subtask(
                    title=s["title"],
                    brief=s["brief"],
                    agent_id=f"agent-{i + 1}",
                    task_id=f"T{i + 1}",
                    system_prompt=s.get("system_prompt"),
                )
                for i, s in enumerate(raw[: self._subtask_cap])
            ]
            return subtasks, notices
        # Graceful fallback: the whole task as one contained sub-task (no volatile
        # system prompt → the worker uses the static fallback prompt).
        task_text = "\n\n".join(message_text(m) for m in self._task_messages() if message_text(m))
        notice = events.notice(
            "warn",
            "ultra_plan_fallback",
            "Decomposition plan could not be produced "
            f"({result.error or 'empty plan'}); running the task as a single "
            "contained sub-task.",
        )
        fallback = _Subtask(
            title="complete the task", brief=task_text, agent_id="agent-1", task_id="T1"
        )
        return [fallback], [notice]

    # -- stage 2: dispatch -------------------------------------------------
    async def _run_subtask(
        self, worker: UltraWorker, subtask: _Subtask, system_content: str
    ) -> WorkerResult:
        """One contained sub-agent: fresh context, focused toolset, the run's
        sub-task effort, structured result envelope. `system_content` is the volatile
        per-agent system prompt (or the fallback) — local to this loop, discarded when
        it returns; it never enters tool_messages, memory, or the trace."""
        brief = f"Sub-task: {subtask.title}\n\n{subtask.brief}"
        loop = AgentLoop(
            worker.chat,
            model=worker.model,
            registry=self._registry_factory(),
            ctx=self._tool_ctx_factory(),
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": brief},
            ],
            sampling=sampling_for(worker.model),
            thinking_budget_tokens=thinking_budget(self._worker_effort, worker.ctx),
            max_iterations=max_tool_iterations(self._worker_effort),
            step_timeout_s=self._step_timeout,
            tool_result_max_chars=tool_result_cap(worker.ctx),
        )
        async for _event in loop.run():
            pass  # worker internals stay contained; only agent_spawn/agent_result surface
        self._merge_usage(loop.total_usage)
        if loop.finish_reason == "error":
            return WorkerResult("failed", loop.error_message or "worker backend failed", "")

        sampling = sampling_for(worker.model)
        report = await request_structured(
            lambda p: worker.chat.complete(p, timeout=self._step_timeout),
            {
                "model": worker.model,
                "messages": [*loop.messages, {"role": "user", "content": _RESULT_INSTRUCTION}],
                "stream": False,
                "temperature": sampling.temperature,
                "top_p": sampling.top_p,
                "top_k": sampling.top_k,
                **thinking_payload_fields(0),
            },
            WORKER_RESULT_SCHEMA,
            schema_name="ultra_worker_result",
        )
        self._merge_usage(report.usage)
        if report.data is not None:
            return WorkerResult(
                report.data["status"], report.data["summary"], report.data["output"]
            )
        # Graceful: build the envelope from the loop's own final answer.
        text = loop.final_text
        summary = text.strip().splitlines()[0][:120] if text.strip() else "no output produced"
        return WorkerResult("ok" if text.strip() else "failed", summary, text)

    async def _run_subtask_bounded(
        self, worker: UltraWorker, subtask: _Subtask, system_content: str
    ) -> tuple[WorkerResult, AgentEvent | None]:
        """Run one sub-task under its wall-clock budget. Never raises: a timeout or a
        crashed sub-agent becomes a structured failure (plus a notice for timeouts)
        that the orchestrator synthesizes around. wait_for CANCELS the sub-agent on
        timeout, so the in-flight llama-server request is dropped, not orphaned."""
        try:
            result = await asyncio.wait_for(
                self._run_subtask(worker, subtask, system_content), timeout=self._subtask_timeout
            )
            return result, None
        except TimeoutError:
            budget = int(self._subtask_timeout)
            notice = events.notice(
                "warn",
                "ultra_subtask_timeout",
                f"Sub-task '{subtask.title}' exceeded its {budget}s budget "
                f"(effort {self._worker_effort}) and was cancelled; the orchestrator "
                "synthesizes around the failure.",
            )
            return WorkerResult("failed", f"timed out after {budget}s", ""), notice
        except Exception as exc:  # noqa: BLE001 - a broken sub-agent must not kill the run
            return WorkerResult("failed", f"sub-agent crashed: {exc!r}", ""), None

    async def _dispatch_parallel(self, subtasks: list[_Subtask]) -> AsyncIterator[AgentEvent]:
        """Run sub-tasks concurrently across the worker-slot pool; spawn/result (and
        assign/result-degrade/timeout notices) events are emitted as they actually
        happen (interleaved across agents)."""
        done_marker = object()  # one per finished sub-task; drives the drain loop
        queue: asyncio.Queue = asyncio.Queue()
        pool: asyncio.Queue[UltraWorker] = asyncio.Queue()
        for worker in self._workers:
            pool.put_nowait(worker)

        async def run_one(subtask: _Subtask) -> None:
            system_content, assign_notice = self._dispatch_assign(subtask)
            if assign_notice is not None:
                await queue.put(assign_notice)
            worker = await pool.get()
            await queue.put(self._spawn_event(subtask, worker.model))
            try:
                result, notice = await self._run_subtask_bounded(worker, subtask, system_content)
            finally:
                pool.put_nowait(worker)
            subtask.result = result
            result_notice = self._record_result(subtask)
            if notice is not None:
                await queue.put(notice)
            if result_notice is not None:
                await queue.put(result_notice)
            await queue.put(self._result_event(subtask, result))
            await queue.put(done_marker)

        tasks = [asyncio.create_task(run_one(s)) for s in subtasks]
        finished = 0
        while finished < len(subtasks):
            item = await queue.get()
            if item is done_marker:
                finished += 1
            else:
                yield item
        await asyncio.gather(*tasks)

    async def _dispatch_sequential(self, subtasks: list[_Subtask]) -> AsyncIterator[AgentEvent]:
        """No worker slots: same contained sub-agents, one at a time on the
        orchestrator slot (role/toolset containment unchanged)."""
        for subtask in subtasks:
            system_content, assign_notice = self._dispatch_assign(subtask)
            if assign_notice is not None:
                yield assign_notice
            yield self._spawn_event(subtask, self._orch.model)
            result, notice = await self._run_subtask_bounded(self._orch, subtask, system_content)
            subtask.result = result
            result_notice = self._record_result(subtask)
            if notice is not None:
                yield notice
            if result_notice is not None:
                yield result_notice
            yield self._result_event(subtask, result)

    # -- stage 3: synthesize -----------------------------------------------
    def _synthesis_message(self, subtasks: list[_Subtask]) -> dict:
        sections = []
        for i, subtask in enumerate(subtasks, 1):
            result = subtask.result or WorkerResult("failed", "never ran", "")
            sections.append(
                f"### Sub-task {i}: {subtask.title}\n"
                f"Brief: {subtask.brief}\n"
                f"Status: {result.status}\n"
                f"Worker summary: {result.summary}\n"
                f"Worker output:\n{result.output or '(none)'}"
            )
        body = "\n\n".join(sections)
        return {
            "role": "user",
            "content": (
                "All sub-tasks have run; the worker results follow. Verify each against "
                "its brief, then produce the final deliverable in a single voice. Treat "
                "worker output as a junior's draft — fix or redo what is wrong, and state "
                "honestly anything that failed or remains unverified.\n\n" + body
            ),
        }

    # -- the run -----------------------------------------------------------
    async def run(self) -> AsyncIterator[AgentEvent]:
        subtasks, plan_notices = await self._plan()
        yield events.plan([s.title for s in subtasks])
        for plan_notice in plan_notices:
            yield plan_notice

        # Each executing slot advertises its capability into the SLAP trace.
        executors = self._workers if self._parallel else [self._orch]
        for cap_notice in self._advertise_capabilities(executors):
            yield cap_notice

        if self._workers:
            async for event in self._dispatch_parallel(subtasks):
                yield event
        else:
            yield events.notice(
                "warn",
                "ultra_sequential",
                "No worker slots in this loadout; Ultra sub-tasks run sequentially "
                "on the orchestrator.",
            )
            if self._worker_effort != self._effort:
                yield events.notice(
                    "info",
                    "ultra_effort_capped",
                    f"Sequential Ultra caps sub-task effort at '{self._worker_effort}' "
                    f"(requested '{self._effort}') so the run stays bounded on a "
                    "single slot.",
                )
            async for event in self._dispatch_sequential(subtasks):
                yield event

        # Synthesis decision (recorded before the final llama call so the trace is
        # persisted even if synthesis errors), then persist the validated transcript.
        decide_notice = self._record_decide(subtasks)
        if decide_notice is not None:
            yield decide_notice
        self._flush_trace()

        messages = [
            {"role": "system", "content": system_prompt("ultra")},
            *self._task_messages(),
            self._synthesis_message(subtasks),
        ]
        payload = self._orch_payload(
            messages, thinking=thinking_budget(self._effort, self._orch.ctx)
        )
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        try:
            response = await self._orch_complete(payload)
            message = response["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            self.error_message = f"synthesis failed: {exc}"
            self.finish_reason = "error"
            yield events.error("server_error", self.error_message)
            yield events.done("error")
            return
        self._merge_usage(response.get("usage") or {})

        self.final_text = message.get("content") or ""
        self.finish_reason = response["choices"][0].get("finish_reason") or "stop"
        if self.final_text:
            yield events.delta(self.final_text)
        yield events.usage(**self.total_usage)
        yield events.done(self.finish_reason)
