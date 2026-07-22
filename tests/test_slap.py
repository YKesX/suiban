"""SLAP vendored-protocol tests: schema parity (drift guard), loading, validation with
clean errors, builders for all nine operations, the in-process trace store, and the
read-only /v1/slap observability endpoints (api.md §12)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban import slap

# The canonical schemas live in the sibling `slap` repo (workspace/slap/schemas). This
# is a dev-workspace drift guard, not a runtime coupling — suiban vendors its own copy
# and never imports the slap package; the test skips when the sibling repo is absent
# (e.g. a standalone suiban checkout).
_SLAP_REPO_SCHEMAS = Path(__file__).resolve().parents[2] / "slap" / "schemas"

_ALL_SCHEMA_FILES = (*slap.OPERATIONS, "envelope")


# -- drift guard: vendored schemas are byte-identical to the canonical repo ----
@pytest.mark.skipif(
    not _SLAP_REPO_SCHEMAS.is_dir(), reason="sibling slap repo not checked out alongside suiban"
)
def test_vendored_schemas_are_byte_identical_to_slap_repo() -> None:
    for name in _ALL_SCHEMA_FILES:
        vendored = (slap.SCHEMAS_DIR / f"{name}.json").read_bytes()
        canonical = (_SLAP_REPO_SCHEMAS / f"{name}.json").read_bytes()
        assert vendored == canonical, f"{name}.json drifted from the canonical slap repo"


def test_all_ten_schemas_are_vendored() -> None:
    for name in _ALL_SCHEMA_FILES:
        assert (slap.SCHEMAS_DIR / f"{name}.json").is_file()
    assert len(_ALL_SCHEMA_FILES) == 10


# -- loading -------------------------------------------------------------------
def test_load_schema_and_schemas() -> None:
    assert slap.load_schema("assign")["properties"]["operation"]["const"] == "assign"
    schemas = slap.load_schemas()
    assert set(schemas) == set(slap.OPERATIONS)  # envelope is the base, not an operation
    with pytest.raises(KeyError):
        slap.load_schema("bogus")


def test_protocol_constants() -> None:
    assert slap.VERSION == "1.0"
    assert len(slap.OPERATIONS) == 9
    assert "orchestrator" in slap.PROFILES


# -- builders ------------------------------------------------------------------
def test_all_builders_produce_valid_messages() -> None:
    messages = [
        slap.build_assign(message_id="M1", task_id="T1", role="worker", goal="do it"),
        slap.build_result(message_id="M2", task_id="T1", status="completed"),
        slap.build_review(message_id="M3", task_id="T1", target="patch:1", status="approved"),
        slap.build_decide(message_id="M4", task_id="T1", decision="accept"),
        slap.build_error(message_id="M5", task_id="T1", code="timeout"),
        slap.build_cancel(message_id="M6", task_id="T1"),
        slap.build_heartbeat(message_id="M7", task_id="T1", progress=0.5),
        slap.build_capability(message_id="M8", task_id="T1", model="bonsai-8b"),
        slap.build_status(message_id="M9", task_id="T1", task_state="running"),
    ]
    for message in messages:
        assert slap.validate_message(message) == [], (
            message["operation"],
            slap.validate_message(message),
        )
    assert {m["operation"] for m in messages} == set(slap.OPERATIONS)


def test_builder_drops_optional_none_fields() -> None:
    message = slap.build_assign(message_id="M1", task_id="T1", role="worker", goal="g")
    for optional in ("scope", "inputs", "system_prompt", "limits", "base_revision", "checks"):
        assert optional not in message


def test_assign_with_all_fields_validates() -> None:
    message = slap.build_assign(
        message_id="M1",
        task_id="T42",
        parent_task="T0",
        role="implementer",
        goal="Prevent concurrent token refresh corruption",
        base_revision="git:main:a81f39",
        scope=["src/auth.py"],
        inputs=["issue:42"],
        expected_artifacts=["patch", "test_report"],
        checks=["pytest tests/test_auth.py"],
        system_prompt="You are a careful concurrency reviewer.",
        limits={"maximum_files": 2, "maximum_rounds": 2, "maximum_seconds": 900},
    )
    assert slap.validate_message(message) == []
    assert message["system_prompt"] == "You are a careful concurrency reviewer."


def test_result_with_evidence_linked_claims_validates() -> None:
    message = slap.build_result(
        message_id="M2",
        task_id="T42",
        status="completed",
        artifacts=["patch:T42.patch"],
        claims=[{"claim": "token refresh is serialized", "evidence": ["patch:T42.patch"]}],
        risks=["lock contention"],
        confidence=0.84,
    )
    assert slap.validate_message(message) == []


# -- validation: clean errors --------------------------------------------------
def test_validate_message_rejects_non_dict() -> None:
    errors = slap.validate_message("not a message")
    assert errors and "object" in errors[0]


def test_validate_message_rejects_non_slap_protocol() -> None:
    errors = slap.validate_message(
        {
            "protocol": "NOTSLAP",
            "version": "1.0",
            "operation": "cancel",
            "message_id": "M",
            "task_id": "T",
        }
    )
    assert any("protocol" in e for e in errors)


def test_validate_message_rejects_unknown_operation() -> None:
    errors = slap.validate_message(
        {
            "protocol": "SLAP",
            "version": "1.0",
            "operation": "frobnicate",
            "message_id": "M",
            "task_id": "T",
        }
    )
    assert any("operation" in e for e in errors)


def test_validate_message_rejects_unsupported_major_version() -> None:
    errors = slap.validate_message(
        {
            "protocol": "SLAP",
            "version": "2.0",
            "operation": "cancel",
            "message_id": "M",
            "task_id": "T",
        }
    )
    assert any("major version" in e for e in errors)


def test_validate_message_reports_missing_required_fields() -> None:
    errors = slap.validate_message(
        {
            "protocol": "SLAP",
            "version": "1.0",
            "operation": "assign",
            "message_id": "M",
            "task_id": "T",
        }
    )
    assert any("role" in e for e in errors)
    assert any("goal" in e for e in errors)


def test_validate_rejects_unexpected_property() -> None:
    message = slap.build_capability(message_id="M", task_id="T", model="x")
    message["surprise"] = 1
    assert any("surprise" in e for e in slap.validate_message(message))


def test_validate_rejects_bad_version_pattern() -> None:
    message = slap.build_cancel(message_id="M", task_id="T")
    message["version"] = "1"  # not major.minor
    assert any("pattern" in e for e in slap.validate_message(message))


def test_claim_without_evidence_is_invalid() -> None:
    message = slap.build_result(
        message_id="M",
        task_id="T",
        status="completed",
        claims=[{"claim": "unsupported", "evidence": []}],
    )
    assert slap.validate_message(message)  # evidence minItems 1


def test_is_valid_helper() -> None:
    assert slap.is_valid(slap.build_cancel(message_id="M", task_id="T"))
    assert not slap.is_valid({"protocol": "SLAP"})


# -- trace store ---------------------------------------------------------------
def test_trace_store_record_get_reset() -> None:
    store = slap.SlapTraceStore(max_sessions=2)
    store.record("a", [{"operation": "assign"}])
    assert store.get("a") == [{"operation": "assign"}]
    assert store.get("missing") == []

    # get() returns a deep copy: mutating it cannot corrupt the store.
    borrowed = store.get("a")
    borrowed.append({"x": 1})
    assert store.get("a") == [{"operation": "assign"}]

    # Bounded FIFO eviction past max_sessions.
    store.record("b", [])
    store.record("c", [])
    assert store.get("a") == []  # oldest evicted
    store.reset()
    assert store.get("b") == []


# -- /v1/slap endpoints --------------------------------------------------------
def test_slap_info_endpoint(client: TestClient) -> None:
    resp = client.get("/v1/slap")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == "1.0"
    assert body["profiles"] == ["orchestrator", "worker", "utility"]
    assert body["operations"] == list(slap.OPERATIONS)


def test_slap_schema_endpoint_returns_vendored_schema(client: TestClient) -> None:
    resp = client.get("/v1/slap/schema/assign")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["$id"].endswith("assign.json")
    assert schema["properties"]["operation"]["const"] == "assign"


def test_slap_schema_endpoint_unknown_is_404(client: TestClient) -> None:
    resp = client.get("/v1/slap/schema/frobnicate")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "slap_operation_not_found"


def test_slap_schema_endpoint_envelope_is_404(client: TestClient) -> None:
    # envelope is the shared base, not one of the nine operations.
    assert client.get("/v1/slap/schema/envelope").status_code == 404


def test_slap_trace_unknown_session_is_empty(client: TestClient) -> None:
    resp = client.get("/v1/slap/trace/never-ran-this")
    assert resp.status_code == 200
    assert resp.json() == {"messages": []}


def test_slap_trace_after_ultra_run(client: TestClient) -> None:
    slap.trace_store().reset()
    session_id = "slap-trace-session-1"
    run = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "do a big thing"}],
            "mode": "ultra",
            "session_id": session_id,
        },
    )
    assert run.status_code == 200

    resp = client.get(f"/v1/slap/trace/{session_id}")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert messages
    ops = {m["operation"] for m in messages}
    assert {"capability", "assign", "result", "decide"} <= ops
    # Volatile per-agent system prompts are never persisted to the trace.
    assert all("system_prompt" not in m for m in messages)
    for message in messages:
        assert slap.validate_message(message) == []
