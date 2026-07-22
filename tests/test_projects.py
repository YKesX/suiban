"""Projects (api.md §9): router CRUD + error codes, scoped FTS5 doc search, the
sessions.project_id migration guard, chat integration (validation, session binding,
excerpt injection), and the sessions project_id filter."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from suiban.agent.loop import BackendChat
from suiban.memory.store import MemoryStore
from suiban.routers.chat import PROJECT_CONTEXT_HEADER


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.sqlite")
    yield s
    s.close()


# -- store: CRUD + scoped FTS -------------------------------------------------
def test_project_roundtrip_and_counts(store: MemoryStore) -> None:
    project = store.add_project("bonsai", "the local stack")
    assert project["id"].startswith("proj_")
    assert project["session_count"] == 0 and project["doc_count"] == 0

    store.add_project_doc(project["id"], "vram notes", "the 27b fits in 10 GB ternary")
    store.ensure_session("sess-a", "chat", project["id"])
    fetched = store.get_project(project["id"])
    assert fetched is not None
    assert fetched["doc_count"] == 1 and fetched["session_count"] == 1

    updated = store.update_project(project["id"], description="renamed")
    assert updated is not None and updated["description"] == "renamed"
    assert [p["id"] for p in store.list_projects()] == [project["id"]]


def test_project_doc_search_is_scoped_to_the_project(store: MemoryStore) -> None:
    mine = store.add_project("mine")
    other = store.add_project("other")
    doc = store.add_project_doc(mine["id"], "deploy runbook", "deploy on friday mornings only")
    store.add_project_doc(other["id"], "deploy junk", "deploy whenever, chaos reigns")

    hits = store.search_project_docs(mine["id"], "deploy friday")
    assert [h["doc_id"] for h in hits] == [doc["id"]]
    assert "friday" in hits[0]["excerpt"]
    # Operator injection never raises (same fts_query guard as memory search).
    assert isinstance(store.search_project_docs(mine["id"], '"AND (NOT* ^:'), list)


def test_delete_project_cascades_docs_and_clears_sessions(store: MemoryStore) -> None:
    project = store.add_project("doomed")
    store.add_project_doc(project["id"], "doc", "searchable zeppelin content")
    store.ensure_session("sess-survivor", "chat", project["id"])

    assert store.delete_project(project["id"])
    assert store.get_project(project["id"]) is None
    assert store.search_project_docs(project["id"], "zeppelin") == []
    # The member session survives with project_id cleared (api.md §9).
    sessions = store.list_sessions()
    assert [s["id"] for s in sessions] == ["sess-survivor"]
    assert sessions[0]["project_id"] is None


def test_sessions_project_id_migration_guard(tmp_path: Path) -> None:
    """A database created before projects landed gains the column on open."""
    db_path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT, mode TEXT NOT NULL, "
        "started_at TEXT NOT NULL, ended_at TEXT, message_count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO sessions (id, mode, started_at) VALUES ('legacy', 'chat', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    try:
        sessions = store.list_sessions()
        assert [s["id"] for s in sessions] == ["legacy"]
        assert sessions[0]["project_id"] is None
        project = store.add_project("late")
        store.ensure_session("legacy", "chat", project["id"])
        assert store.list_sessions(project_id=project["id"])[0]["id"] == "legacy"
    finally:
        store.close()


def test_list_sessions_project_filter_combines_with_query(store: MemoryStore) -> None:
    project = store.add_project("filtered")
    store.ensure_session("in-project", "chat", project["id"])
    store.add_message("in-project", "user", "the xylophone plan")
    store.ensure_session("outside", "chat")
    store.add_message("outside", "user", "the xylophone plan elsewhere")

    assert {s["id"] for s in store.list_sessions(query="xylophone")} == {"in-project", "outside"}
    filtered = store.list_sessions(query="xylophone", project_id=project["id"])
    assert [s["id"] for s in filtered] == ["in-project"]
    assert store.list_sessions(project_id="proj_nope") == []


# -- HTTP surface -------------------------------------------------------------
def test_projects_http_validation_and_404s(client: TestClient) -> None:
    assert client.post("/v1/projects", json={}).status_code == 400
    assert client.post("/v1/projects", json={"name": ""}).status_code == 400
    assert client.post("/v1/projects", json={"name": "x", "bogus": 1}).status_code == 400

    for method, path in (
        ("GET", "/v1/projects/proj_nope"),
        ("PATCH", "/v1/projects/proj_nope"),
        ("DELETE", "/v1/projects/proj_nope"),
        ("GET", "/v1/projects/proj_nope/docs"),
        ("POST", "/v1/projects/proj_nope/docs"),
    ):
        resp = client.request(method, path, json={"name": "x", "title": "t", "content": "c"})
        assert resp.status_code == 404, path
        assert resp.json()["error"]["code"] == "project_not_found"

    project_id = client.post("/v1/projects", json={"name": "p"}).json()["id"]
    missing_doc = client.get(f"/v1/projects/{project_id}/docs/doc_nope")
    assert missing_doc.status_code == 404
    assert missing_doc.json()["error"]["code"] == "project_doc_not_found"
    assert client.delete(f"/v1/projects/{project_id}/docs/doc_nope").status_code == 404


# -- chat integration ---------------------------------------------------------
def test_chat_with_unknown_project_is_404(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "hi"}],
            "project_id": "proj_nope",
        },
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "project_not_found"


def test_chat_binds_session_and_injects_doc_excerpts(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = client.post("/v1/projects", json={"name": "refactor"}).json()["id"]
    client.post(
        f"/v1/projects/{project_id}/docs",
        json={
            "title": "zeppelin status",
            "content": "the zeppelin refactor is half done; blockers are in the gondola module",
        },
    )

    payloads: list[dict] = []
    original = BackendChat.complete

    async def spy(self, payload: dict, timeout: float) -> dict:
        payloads.append(payload)
        return await original(self, payload, timeout)

    monkeypatch.setattr(BackendChat, "complete", spy)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "how is the zeppelin refactor going?"}],
            "session_id": "sess-proj",
            "project_id": project_id,
        },
    )
    assert resp.status_code == 200

    # The injected system message is clearly delimited and carries the doc excerpt.
    injected = [
        m
        for p in payloads
        for m in p["messages"]
        # Coalesced into the single leading system message (Bonsai template rule).
        if isinstance(m.get("content"), str) and PROJECT_CONTEXT_HEADER in m["content"]
    ]
    assert injected, "project excerpts must be injected into the model request"
    assert "<<<doc: zeppelin status>>>" in injected[0]["content"]
    assert "zeppelin refactor" in injected[0]["content"]
    assert "<<<end doc>>>" in injected[0]["content"]

    # The session is bound to the project and listed under the filter.
    sessions = client.get("/v1/memory/sessions", params={"project_id": project_id}).json()
    assert [s["id"] for s in sessions["sessions"]] == ["sess-proj"]
    assert sessions["sessions"][0]["project_id"] == project_id
    transcript = client.get("/v1/memory/sessions/sess-proj").json()
    assert transcript["session"]["project_id"] == project_id
    # ... and the injected block was NOT archived (system messages never are).
    assert all(PROJECT_CONTEXT_HEADER not in m["content"] for m in transcript["messages"])

    assert client.get(f"/v1/projects/{project_id}").json()["session_count"] == 1


def test_chat_without_matching_docs_injects_nothing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_id = client.post("/v1/projects", json={"name": "empty"}).json()["id"]

    payloads: list[dict] = []
    original = BackendChat.complete

    async def spy(self, payload: dict, timeout: float) -> dict:
        payloads.append(payload)
        return await original(self, payload, timeout)

    monkeypatch.setattr(BackendChat, "complete", spy)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "anything at all"}],
            "session_id": "sess-empty-proj",
            "project_id": project_id,
        },
    )
    assert resp.status_code == 200
    assert all(
        not (isinstance(m.get("content"), str) and m["content"].startswith(PROJECT_CONTEXT_HEADER))
        for p in payloads
        for m in p["messages"]
    )
    # The session still binds to the project.
    sessions = client.get("/v1/memory/sessions", params={"project_id": project_id}).json()
    assert [s["id"] for s in sessions["sessions"]] == ["sess-empty-proj"]
