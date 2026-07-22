"""Skill import (api.md 2026-07-22c): scan an ecosystem's SKILL.md directories, validate
each with the shared agentskills.io validator, copy the good ones into the store
(source="imported", verified=false), skip the malformed ones with a reason. Covers the
pure importer, the openclaw/hermes source resolution, and the POST /v1/skills/import
route (happy path + a path that does not exist)."""

from __future__ import annotations

from pathlib import Path

import pytest

import suiban.memory.skill_import as skill_import
from suiban.memory.skill_import import SkillImportError, import_skills
from suiban.memory.skills import SkillStore

GOOD = "---\nname: {name}\ndescription: An imported {name} skill.\n---\n\n# {name}\n1. do it\n"
MALFORMED = "this file has no frontmatter at all\njust prose\n"


def _good(name: str) -> str:
    return GOOD.format(name=name)


def _write_skill(root: Path, name: str, content: str, extra: dict[str, str] | None = None) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for rel, text in (extra or {}).items():
        target = skill_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return skill_dir


# -- pure importer -----------------------------------------------------------
def test_import_from_path_imports_good_skips_malformed(tmp_path: Path) -> None:
    src = tmp_path / "external"
    _write_skill(src, "good-skill", _good("good-skill"), extra={"scripts/run.sh": "echo hi\n"})
    _write_skill(src, "bad-skill", MALFORMED)

    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "path", str(src))

    assert result.imported == ["good-skill"]
    assert [s["name"] for s in result.skipped] == ["bad-skill"]
    assert result.skipped[0]["reason"]  # a non-empty explanation, not a crash

    # The copied skill is in the store, with its scripts/ and imported provenance.
    skill = store.get("good-skill")
    assert skill is not None
    assert skill.source == "imported"
    assert skill.verified is False
    assert (tmp_path / "store" / "good-skill" / "scripts" / "run.sh").is_file()
    assert "good-skill" in {s.name for s in store.list()}


def test_import_path_that_does_not_exist_raises(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "store")
    with pytest.raises(SkillImportError):
        import_skills(store, "path", str(tmp_path / "nope"))


def test_import_path_requires_a_path(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "store")
    with pytest.raises(SkillImportError):
        import_skills(store, "path", None)


def test_import_openclaw_scans_workspace_and_repo(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_skill(home / ".openclaw" / "workspace" / "skills", "ws-skill", _good("ws-skill"))
    repo = tmp_path / "repo"
    _write_skill(repo / ".agents" / "skills", "repo-skill", _good("repo-skill"))

    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "openclaw", str(repo), user_home=home)

    assert set(result.imported) == {"ws-skill", "repo-skill"}
    assert store.get("ws-skill") is not None
    assert store.get("repo-skill") is not None


def test_import_hermes_scans_home_and_nested_optional_skills(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _write_skill(home / ".hermes" / "skills", "hz-skill", _good("hz-skill"))
    repo = tmp_path / "repo"
    # optional-skills/<category>/<name>/SKILL.md — the recursive scan handles the nesting.
    _write_skill(repo / "optional-skills" / "coding", "opt-skill", _good("opt-skill"))

    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "hermes", str(repo), user_home=home)

    assert set(result.imported) == {"hz-skill", "opt-skill"}


def test_import_missing_ecosystem_dirs_is_empty_not_an_error(tmp_path: Path) -> None:
    """openclaw/hermes with nothing installed and no path: empty result, never a raise."""
    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "openclaw", None, user_home=tmp_path / "empty-home")
    assert result.imported == []
    assert result.skipped == []


def test_reimport_replaces_and_resets_verification(tmp_path: Path) -> None:
    src = tmp_path / "external"
    _write_skill(src, "again", _good("again"))
    store = SkillStore(tmp_path / "store")

    import_skills(store, "path", str(src))
    store.mark_verified("again")
    assert store.get("again").verified is True

    # A re-import resets provenance (imported, unverified) — new content is unproven.
    result = import_skills(store, "path", str(src))
    assert result.imported == ["again"]
    assert store.get("again").verified is False
    assert store.get("again").source == "imported"


# -- POST /v1/skills/import --------------------------------------------------
def test_import_endpoint_happy_path(client, tmp_path: Path) -> None:
    src = tmp_path / "external"
    _write_skill(src, "endpoint-skill", _good("endpoint-skill"))

    resp = client.post("/v1/skills/import", json={"source": "path", "path": str(src)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["imported"] == [{"name": "endpoint-skill"}]
    assert body["skipped"] == []

    # The imported skill is now visible on the /v1/skills surface.
    listed = {s["name"] for s in client.get("/v1/skills").json()["skills"]}
    assert "endpoint-skill" in listed
    got = client.get("/v1/skills/endpoint-skill").json()
    assert got["source"] == "imported"
    assert got["verified"] is False


def test_import_endpoint_reports_skipped(client, tmp_path: Path) -> None:
    src = tmp_path / "external"
    _write_skill(src, "ok-skill", _good("ok-skill"))
    _write_skill(src, "broken-skill", MALFORMED)

    body = client.post("/v1/skills/import", json={"source": "path", "path": str(src)}).json()
    assert body["imported"] == [{"name": "ok-skill"}]
    assert len(body["skipped"]) == 1
    assert body["skipped"][0]["name"] == "broken-skill"
    assert body["skipped"][0]["reason"]


def test_import_endpoint_path_missing_is_clean_400(client, tmp_path: Path) -> None:
    resp = client.post(
        "/v1/skills/import", json={"source": "path", "path": str(tmp_path / "does-not-exist")}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "import_source_unavailable"


def test_import_endpoint_rejects_unknown_source(client) -> None:
    resp = client.post("/v1/skills/import", json={"source": "nope"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "validation_error"


def test_import_endpoint_path_source_requires_path(client) -> None:
    resp = client.post("/v1/skills/import", json={"source": "path"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "validation_error"


# -- SEC-2: the `path` scan is bounded, never the whole filesystem -----------
def test_import_path_filesystem_root_is_refused(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "store")
    with pytest.raises(SkillImportError, match="filesystem root"):
        import_skills(store, "path", "/")


def test_import_path_home_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    store = SkillStore(home / ".bonsai" / "skills")
    with pytest.raises(SkillImportError, match="home directory"):
        import_skills(store, "path", str(home), user_home=home)


def test_import_path_ancestor_of_bonsai_home_is_refused(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".bonsai" / "skills").mkdir(parents=True)
    store = SkillStore(home / ".bonsai" / "skills")
    # tmp_path is an ancestor of ~/.bonsai: scanning it would sweep the store itself.
    with pytest.raises(SkillImportError, match="bonsai home"):
        import_skills(store, "path", str(tmp_path), user_home=home)


def test_import_endpoint_filesystem_root_is_clean_400(client) -> None:
    resp = client.post("/v1/skills/import", json={"source": "path", "path": "/"})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "import_source_unavailable"


def test_import_path_too_many_dirs_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skill_import, "MAX_SCAN_DIRS", 5)
    src = tmp_path / "external"
    for i in range(12):
        (src / f"d{i}").mkdir(parents=True)
    store = SkillStore(tmp_path / "store")
    with pytest.raises(SkillImportError, match="more than 5 directories"):
        import_skills(store, "path", str(src))


def test_import_path_depth_cap_prunes_deep_skills(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skill_import, "MAX_SCAN_DEPTH", 2)
    src = tmp_path / "external"
    _write_skill(src, "shallow", _good("shallow"))  # depth 1: imported
    _write_skill(src / "a" / "b" / "c", "deep", _good("deep"))  # depth 4: pruned
    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "path", str(src))
    assert result.imported == ["shallow"]
    assert store.get("deep") is None


def test_import_path_skill_count_cap_stops_discovery(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(skill_import, "MAX_IMPORT_SKILLS", 2)
    src = tmp_path / "external"
    for i in range(5):
        _write_skill(src, f"skill-{i}", _good(f"skill-{i}"))
    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "path", str(src))
    assert len(result.imported) == 2  # discovery stopped at the cap


# -- SEC-3: symlinks in a source skill dir are skipped, never followed -------
def test_import_skips_symlinks_in_skill_dir(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET EXTERNAL DATA\n", encoding="utf-8")
    src = tmp_path / "external"
    skill_dir = _write_skill(src, "leaky", _good("leaky"))
    (skill_dir / "stolen.txt").symlink_to(secret)  # hostile: points outside the skill dir

    store = SkillStore(tmp_path / "store")
    result = import_skills(store, "path", str(src))

    assert result.imported == ["leaky"]
    dest = tmp_path / "store" / "leaky"
    # The symlink was dropped: no stolen.txt copied, no external contents pulled in.
    assert not (dest / "stolen.txt").exists()
    assert "TOP SECRET" not in (dest / "SKILL.md").read_text()


# -- T4: a traversal / slash-bearing skill name is refused at the import layer
@pytest.mark.parametrize("bad", ["../evil", "a/b", "..", "nested/../escape", "with space"])
def test_import_skill_dir_rejects_traversal_names(tmp_path: Path, bad: str) -> None:
    src = tmp_path / "external"
    _write_skill(src, "legit", _good("legit"))
    store = SkillStore(tmp_path / "store")
    with pytest.raises(ValueError):
        store.import_skill_dir(bad, src / "legit")
    # Nothing escaped the store root.
    assert not (tmp_path / "evil").exists()
    assert not (tmp_path / "store" / "evil").exists()
