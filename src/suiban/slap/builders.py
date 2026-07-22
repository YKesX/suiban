"""Builders for the nine SLAP operations.

Each builder returns a plain dict shaped to its vendored schema; callers validate with
`validate_message` before use. Optional fields set to None are dropped so messages stay
compact and satisfy the schemas' `additionalProperties: false`. Message/task ids are the
caller's responsibility (Ultra assigns them deterministically per run) — the protocol
does not mint them.
"""

from __future__ import annotations

from typing import Any

from suiban.slap.protocol import VERSION


def _envelope(
    operation: str,
    *,
    message_id: str,
    task_id: str,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "protocol": "SLAP",
        "version": VERSION,
        "message_id": message_id,
        "task_id": task_id,
        "operation": operation,
    }
    if parent_task is not None:
        env["parent_task"] = parent_task
    if depends_on is not None:
        env["depends_on"] = depends_on
    if created_at is not None:
        env["created_at"] = created_at
    return env


def _set(envelope: dict[str, Any], /, **fields: Any) -> dict[str, Any]:
    """Attach operation fields to the envelope, dropping any that are None. The first
    parameter is positional-only so operation fields named `message` (build_error) do
    not collide with it."""
    for key, value in fields.items():
        if value is not None:
            envelope[key] = value
    return envelope


def build_assign(
    *,
    message_id: str,
    task_id: str,
    role: str,
    goal: str,
    base_revision: str | None = None,
    scope: list[str] | None = None,
    inputs: list[str] | None = None,
    expected_artifacts: list[str] | None = None,
    checks: list[str] | None = None,
    system_prompt: str | None = None,
    limits: dict[str, Any] | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """orchestrator → worker: delegate a bounded task.

    `system_prompt` is the volatile per-agent system prompt the orchestrator generates
    for THIS task; it is not persisted and is discarded when the agent finishes.
    """
    msg = _envelope(
        "assign",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(
        msg,
        role=role,
        goal=goal,
        base_revision=base_revision,
        scope=scope,
        inputs=inputs,
        expected_artifacts=expected_artifacts,
        checks=checks,
        system_prompt=system_prompt,
        limits=limits,
    )


def build_result(
    *,
    message_id: str,
    task_id: str,
    status: str,
    artifacts: list[str] | None = None,
    claims: list[dict[str, Any]] | None = None,
    risks: list[str] | None = None,
    summary: str | None = None,
    confidence: float | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """worker → orchestrator: report outcome. Every claim must carry ≥1 evidence ref."""
    msg = _envelope(
        "result",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(
        msg,
        status=status,
        artifacts=artifacts,
        claims=claims,
        risks=risks,
        summary=summary,
        confidence=confidence,
    )


def build_review(
    *,
    message_id: str,
    task_id: str,
    target: str,
    status: str,
    findings: list[dict[str, Any]] | None = None,
    confidence: float | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """reviewer → orchestrator: verdict on an artifact."""
    msg = _envelope(
        "review",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(msg, target=target, status=status, findings=findings, confidence=confidence)


def build_decide(
    *,
    message_id: str,
    task_id: str,
    decision: str,
    accepted_artifacts: list[str] | None = None,
    rejected_artifacts: list[str] | None = None,
    next_tasks: list[str] | None = None,
    reason: str | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """orchestrator → *: accept / reject / retry / escalate."""
    msg = _envelope(
        "decide",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(
        msg,
        decision=decision,
        accepted_artifacts=accepted_artifacts,
        rejected_artifacts=rejected_artifacts,
        next_tasks=next_tasks,
        reason=reason,
    )


def build_error(
    *,
    message_id: str,
    task_id: str,
    code: str,
    message: str | None = None,
    retriable: bool | None = None,
    partial_artifacts: list[str] | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """any → orchestrator: structured failure."""
    msg = _envelope(
        "error",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(
        msg, code=code, message=message, retriable=retriable, partial_artifacts=partial_artifacts
    )


def build_cancel(
    *,
    message_id: str,
    task_id: str,
    reason: str | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """orchestrator → worker: stop a task."""
    msg = _envelope(
        "cancel",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(msg, reason=reason)


def build_heartbeat(
    *,
    message_id: str,
    task_id: str,
    progress: float | None = None,
    note: str | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """worker → orchestrator: liveness + coarse progress."""
    msg = _envelope(
        "heartbeat",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(msg, progress=progress, note=note)


def build_capability(
    *,
    message_id: str,
    task_id: str,
    model: str,
    model_family: str | None = None,
    parameters: str | None = None,
    quantization: str | None = None,
    context_limit: int | None = None,
    modalities: list[str] | None = None,
    tools: list[str] | None = None,
    structured_output_reliability: float | None = None,
    latency_ms_estimate: float | None = None,
    memory_mb_estimate: float | None = None,
    compute_cost_estimate: float | None = None,
    task_categories: list[str] | None = None,
    max_task_complexity: str | None = None,
    current_workload: float | None = None,
    backend: str | None = None,
    location: str | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """agent → scheduler: advertise what this instance can do (all optional but model)."""
    msg = _envelope(
        "capability",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(
        msg,
        model=model,
        model_family=model_family,
        parameters=parameters,
        quantization=quantization,
        context_limit=context_limit,
        modalities=modalities,
        tools=tools,
        structured_output_reliability=structured_output_reliability,
        latency_ms_estimate=latency_ms_estimate,
        memory_mb_estimate=memory_mb_estimate,
        compute_cost_estimate=compute_cost_estimate,
        task_categories=task_categories,
        max_task_complexity=max_task_complexity,
        current_workload=current_workload,
        backend=backend,
        location=location,
    )


def build_status(
    *,
    message_id: str,
    task_id: str,
    task_state: str,
    agents: list[dict[str, Any]] | None = None,
    note: str | None = None,
    parent_task: str | None = None,
    depends_on: list[str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """scheduler/agent → *: task-graph / agent-state snapshot."""
    msg = _envelope(
        "status",
        message_id=message_id,
        task_id=task_id,
        parent_task=parent_task,
        depends_on=depends_on,
        created_at=created_at,
    )
    return _set(msg, task_state=task_state, agents=agents, note=note)
