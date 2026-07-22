"""Skill injection + verification lifecycle (docs/memory.md §6): matching skills are
injected verified-first with [unverified] labels; a successful run that used a skill
flips it to verified in the meta store; /v1/skills exposes the flag."""

from __future__ import annotations

from fastapi.testclient import TestClient

from suiban.memory import injection as inj
from suiban.memory.skills import Skill


def _skill(name: str, description: str, *, verified: bool = False, version: int = 1) -> Skill:
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n# steps for {name}\n"
    return Skill(
        name=name,
        description=description,
        version=version,
        updated_at="2026-07-21T00:00:00Z",
        source="learned",
        content=content,
        verified=verified,
    )


# -- selection ordering -------------------------------------------------------
def test_select_skills_orders_verified_first_then_overlap() -> None:
    skills = [
        _skill("changelog-entry", "write a changelog entry", verified=False),
        _skill("release-notes", "changelog and release notes for a release", verified=True),
        _skill("unrelated-skill", "grill the perfect halloumi"),
    ]
    chosen = inj.select_skills(skills, "please add a changelog entry for the release", limit=3)
    assert [s.name for s in chosen] == ["release-notes", "changelog-entry"]
    # The verified skill is ordered first; the non-matching one is absent entirely.
    assert chosen[0].verified is True


def test_select_skills_no_match_no_injection() -> None:
    skills = [_skill("changelog-entry", "write a changelog entry")]
    assert inj.select_skills(skills, "what is the weather like") == []
    assert inj.build_skill_context(skills, "what is the weather like", 8192) == (None, [])


# -- block format -------------------------------------------------------------
def test_skill_block_labels_unverified_and_strips_frontmatter() -> None:
    unverified = inj.skill_block(_skill("changelog-entry", "write one", version=2))
    assert unverified.startswith("<<<skill changelog-entry v2 [unverified]>>>")
    assert "---" not in unverified  # frontmatter stripped; body + description only
    assert "# steps for changelog-entry" in unverified
    assert unverified.endswith("<<<end skill>>>")

    verified = inj.skill_block(_skill("changelog-entry", "write one", verified=True))
    assert "[unverified]" not in verified


def test_build_skill_context_budget_caps_blocks() -> None:
    big = _skill("big-skill", "a matching keyword skill")
    big = Skill(**{**big.__dict__, "content": big.content + "x" * 100_000})
    small = _skill("keyword-helper", "matching keyword helper", verified=True)
    content, names = inj.build_skill_context([big, small], "keyword", 1024)
    assert content is not None
    # Verified first; the huge unverified block no longer fits after it.
    assert names == ["keyword-helper"]
    assert len(content) < 10_000


# -- end-to-end verification over HTTP ---------------------------------------
def test_successful_run_verifies_injected_skill(client: TestClient) -> None:
    put = client.put(
        "/v1/skills/changelog-entry",
        json={
            "content": "---\nname: changelog-entry\ndescription: how to write a "
            "changelog entry\n---\n\n# steps\n1. do it\n"
        },
    )
    assert put.status_code == 200
    assert put.json()["verified"] is False

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "please write a changelog entry"}],
            "mode": "chat",
            "session_id": "sess-skill-verify",
        },
    )
    assert resp.status_code == 200

    got = client.get("/v1/skills/changelog-entry").json()
    assert got["verified"] is True
    listing = client.get("/v1/skills").json()["skills"]
    assert listing[0]["verified"] is True  # exposed without content too


def test_unmatched_skill_stays_unverified(client: TestClient) -> None:
    client.put(
        "/v1/skills/halloumi-grilling",
        json={
            "content": "---\nname: halloumi-grilling\ndescription: grill the perfect "
            "halloumi\n---\n\n# steps\n"
        },
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "summarize this python traceback"}],
            "mode": "chat",
            "session_id": "sess-no-skill",
        },
    )
    assert resp.status_code == 200
    assert client.get("/v1/skills/halloumi-grilling").json()["verified"] is False


def test_human_edit_resets_verification(client: TestClient) -> None:
    content = "---\nname: reset-check\ndescription: reset check skill\n---\n\n# v1\n"
    client.put("/v1/skills/reset-check", json={"content": content})
    # Verify via a matching successful run.
    client.post(
        "/v1/chat/completions",
        json={
            "model": "bonsai-auto",
            "messages": [{"role": "user", "content": "run the reset check skill please"}],
            "mode": "chat",
            "session_id": "sess-reset-check",
        },
    )
    assert client.get("/v1/skills/reset-check").json()["verified"] is True
    # A content rewrite (version bump) is unproven again.
    client.put("/v1/skills/reset-check", json={"content": content.replace("# v1", "# v2")})
    got = client.get("/v1/skills/reset-check").json()
    assert got["version"] == 2
    assert got["verified"] is False
