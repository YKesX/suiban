"""Memory layer: FTS5 roundtrip + ranking sanity, session archive, bounded state
files, compression trigger math + rolling summary, skills store, write enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from suiban.errors import BonsaiError
from suiban.memory import compression as comp
from suiban.memory.service import MemoryService
from suiban.memory.skills import (
    SKILL_REJECTION_PREFIX,
    SkillStore,
    parse_frontmatter,
    validate_skill_markdown,
)
from suiban.memory.state_files import StateFiles, compact_oldest
from suiban.memory.store import MemoryStore, fts_query


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "memory.sqlite")
    yield s
    s.close()


@pytest.fixture
def service(tmp_path: Path):
    s = MemoryService(tmp_path / "home")
    s.startup()
    yield s
    s.close()


# -- FTS5 roundtrip + ranking -----------------------------------------------
def test_entry_roundtrip_and_fts_search(store: MemoryStore) -> None:
    entry = store.add_entry("archive", "kubernetes upgrade", "cluster moved to v1.30", ["infra"])
    assert entry.id.startswith("mem_")

    fetched = store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.title == "kubernetes upgrade"
    assert fetched.tags == ["infra"]

    hits = store.search("kubernetes")
    assert len(hits) == 1
    assert hits[0]["entry"]["id"] == entry.id
    assert isinstance(hits[0]["score"], float)
    assert "kubernetes" in hits[0]["snippet"].lower() or "cluster" in hits[0]["snippet"]

    updated = store.update_entry(entry.id, content="cluster moved to v1.31")
    assert updated is not None
    assert updated.content.endswith("v1.31")
    assert store.search("cluster")  # FTS index followed the update

    assert store.delete_entry(entry.id)
    assert store.search("kubernetes") == []  # ... and the delete


def test_bm25_ranking_sanity(store: MemoryStore) -> None:
    heavy = store.add_entry(
        "archive",
        "postgres tuning notes",
        "postgres postgres tuning: shared_buffers, postgres vacuum thresholds",
    )
    light = store.add_entry(
        "archive",
        "misc notes",
        "a long unrelated note that mentions postgres exactly once among many other "
        "words about cooking, travel plans, and a book list",
    )
    hits = store.search("postgres")
    assert [h["entry"]["id"] for h in hits] == [heavy.id, light.id]
    # bm25: more negative = better; the doc dominated by the term ranks first.
    assert hits[0]["score"] < hits[1]["score"]


def test_fts_query_survives_operator_injection(store: MemoryStore) -> None:
    store.add_entry("archive", "quotes", 'he said "AND NOT" loudly')
    # Raw FTS operators / broken quotes must not raise.
    assert isinstance(store.search('"AND (NOT* ^:'), list)


def test_fts_query_survives_nul_and_control_bytes(store: MemoryStore) -> None:
    """Regression (audit 2026-07-22): a NUL byte in the query used to reach the quoted
    MATCH expression and make SQLite raise OperationalError('unterminated string') —
    a 500 on every FTS search. Control bytes are now stripped from tokens."""
    store.add_entry("archive", "bonsai", "盆栽 deployment notes")
    # A NUL between letters must not raise and must still tokenize the halves.
    assert fts_query("a\x00b") == '"a" OR "b"'
    assert fts_query("\x00\x1f\x7f") == ""
    for hostile in ["a\x00b", "deploy\x00ment", "盆栽\x00", "\x00", "\x08\x1f"]:
        assert isinstance(store.search(hostile), list)


def test_fts_query_prefix_terms() -> None:
    # Tokens >= 3 chars become prefix terms; 1-2 char tokens stay exact.
    assert fts_query("kubernetes up") == '"kubernetes"* OR "up"'
    assert fts_query("") == ""


# -- FTS relevance eval set (unicode61 remove_diacritics 2 + prefix indexes) ---
def _seed_eval_entries(store: MemoryStore) -> dict[str, str]:
    """A small realistic corpus; returns {label: entry_id}."""
    entries = {
        "k8s": ("kubernetes upgrade", "cluster moved to v1.30, kubelet flags rewritten", ["infra"]),
        "cafe": ("café shortlist", "the café near Malmö station has the best cardamom buns", []),
        "pg": ("postgres tuning", "postgres vacuum thresholds and shared_buffers sizing", ["db"]),
        "deploy": ("deploy decisions", "we deployed on friday; deployment window is 09:00", []),
        "zure": ("azure billing", "the azure invoice arrives on the 3rd", ["billing"]),
        "jose": ("contact notes", "José prefers async updates over calls", []),
        "misc": ("reading list", "a book about typography and one about rivers", []),
    }
    return {
        label: store.add_entry("archive", title, content, tags).id
        for label, (title, content, tags) in entries.items()
    }


@pytest.mark.parametrize(
    ("query", "expected_label"),
    [
        ("kubernetes", "k8s"),  # 1. plain exact term
        ("kuber", "k8s"),  # 2. prefix query (prefix indexes)
        ("kubelet flags", "k8s"),  # 3. multi-term, both in content
        ("cafe", "cafe"),  # 4. plain query hits diacritic content (café)
        ("café", "cafe"),  # 5. diacritic query hits diacritic content
        ("malmo buns", "cafe"),  # 6. diacritic folding + multi-term
        ("postgres vacuum", "pg"),  # 7. multi-term ranks the dense doc first
        ("deployment", "deploy"),  # 8. inflection via its own prefix match
        ("deplo", "deploy"),  # 9. mid-word prefix
        ("jose", "jose"),  # 10. diacritic folding in names (José)
    ],
)
def test_fts_relevance_eval_set(store: MemoryStore, query: str, expected_label: str) -> None:
    ids = _seed_eval_entries(store)
    hits = store.search(query)
    assert hits, f"no hits for {query!r}"
    assert hits[0]["entry"]["id"] == ids[expected_label], (
        f"{query!r}: expected {expected_label} first, got {[h['entry']['title'] for h in hits]}"
    )


def test_fts_migration_rebuilds_old_porter_tables(tmp_path: Path) -> None:
    """A database created with the pre-refinement `porter unicode61` FTS tables is
    rebuilt idempotently: rows survive, diacritic folding works, no duplicates."""
    import sqlite3

    db_path = tmp_path / "memory.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE memory_entries (
          id TEXT PRIMARY KEY, layer TEXT NOT NULL, title TEXT NOT NULL,
          content TEXT NOT NULL, tags TEXT NOT NULL DEFAULT '[]',
          created_at TEXT NOT NULL, updated_at TEXT NOT NULL, source_session TEXT
        );
        CREATE VIRTUAL TABLE memory_fts USING fts5(
          title, content, tags,
          content='memory_entries', content_rowid='rowid',
          tokenize='porter unicode61'
        );
        CREATE TRIGGER memory_entries_ai AFTER INSERT ON memory_entries BEGIN
          INSERT INTO memory_fts(rowid, title, content, tags)
          VALUES (new.rowid, new.title, new.content, new.tags);
        END;
        INSERT INTO memory_entries VALUES
          ('mem_old1', 'archive', 'café notes', 'the café serves rooibos', '[]',
           '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', NULL);
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore(db_path)
    try:
        sql = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='memory_fts'"
        ).fetchone()[0]
        assert "remove_diacritics 2" in sql
        assert "porter" not in sql
        hits = store.search("cafe")  # diacritic folding proves the rebuild indexed
        assert [h["entry"]["id"] for h in hits] == ["mem_old1"]
    finally:
        store.close()

    # Idempotent: a second open leaves everything intact (no duplicate rows).
    store = MemoryStore(db_path)
    try:
        assert len(store.search("cafe")) == 1
    finally:
        store.close()


# -- sessions ---------------------------------------------------------------
def test_session_archive_and_transcript(store: MemoryStore) -> None:
    store.ensure_session("sess-1", "chat")
    store.add_message("sess-1", "user", "what did we decide about the deploy window")
    store.add_message("sess-1", "assistant", "we decided friday mornings only")

    transcript = store.session_transcript("sess-1")
    assert transcript is not None
    assert transcript["session"]["mode"] == "chat"
    assert transcript["session"]["message_count"] == 2
    assert [m["role"] for m in transcript["messages"]] == ["user", "assistant"]

    sessions = store.list_sessions()
    assert sessions[0]["id"] == "sess-1"
    assert store.list_sessions(query="deploy window")[0]["id"] == "sess-1"
    assert store.list_sessions(query="zeppelin") == []

    hits = store.search_messages("deploy window")
    assert hits and hits[0]["session_id"] == "sess-1"

    # ensure_session on an existing id keeps the original mode.
    store.ensure_session("sess-1", "code")
    transcript = store.session_transcript("sess-1")
    assert transcript is not None
    assert transcript["session"]["mode"] == "chat"


def test_delete_session_removes_transcript_and_fts(store: MemoryStore) -> None:
    store.ensure_session("sess-del", "chat")
    store.add_message("sess-del", "user", "remember the zephyr migration")
    store.ensure_session("sess-keep", "chat")
    store.add_message("sess-keep", "user", "the aurora rollout")

    assert store.delete_session("sess-del") is True
    assert store.session_transcript("sess-del") is None
    assert all(s["id"] != "sess-del" for s in store.list_sessions())
    # the messages_fts mirror follows the delete (via the messages_ad trigger)
    assert store.search_messages("zephyr") == []
    # unrelated sessions and their index survive
    assert store.session_transcript("sess-keep") is not None
    assert store.search_messages("aurora")
    # deleting an unknown id is a no-op, not an error
    assert store.delete_session("sess-del") is False


def test_delete_session_survives_archive_backreference(store: MemoryStore) -> None:
    # an archive entry citing the session as its source must not block the delete
    store.ensure_session("sess-src", "chat")
    store.add_message("sess-src", "user", "the sequoia benchmark")
    entry = store.add_entry(
        "archive", "sequoia note", "benchmark held", source_session="sess-src"
    )
    assert store.delete_session("sess-src") is True
    # the entry is kept, its dangling back-reference nulled
    kept = store.get_entry(entry.id)
    assert kept is not None and kept.source_session is None


def test_service_delete_state_file_and_identity_guard(service: MemoryService) -> None:
    service.files.write_state("scratch-note", "a throwaway working note")
    service.remirror_files()
    assert any(f.name == "scratch-note.md" for f in service.files.all_files())

    service.delete_state_file("scratch-note.md")
    assert all(f.name != "scratch-note.md" for f in service.files.all_files())

    # identity files are never deletable over HTTP
    with pytest.raises(BonsaiError) as ident:
        service.delete_state_file("identity.md")
    assert ident.value.code == "identity_read_only"
    assert any(f.name == "identity.md" for f in service.files.all_files())

    # unknown names 404 rather than touching the filesystem
    with pytest.raises(BonsaiError) as unknown:
        service.delete_state_file("../../etc/passwd")
    assert unknown.value.code == "state_file_unknown"


# -- bounded state files -----------------------------------------------------
def test_compact_oldest_drops_leading_paragraphs() -> None:
    old = "oldest paragraph " * 20
    mid = "middle paragraph " * 20
    new = "newest paragraph"
    content = f"{old}\n\n{mid}\n\n{new}"
    cap = len(f"{mid}\n\n{new}".encode()) + 10
    result = compact_oldest(content, cap)
    assert "oldest" not in result
    assert result.endswith("newest paragraph")


def test_compact_oldest_hard_trims_single_giant_paragraph() -> None:
    content = "x" * 1000 + "TAIL"
    result = compact_oldest(content, 100)
    assert len(result.encode()) <= 100
    assert result.endswith("TAIL")


def test_state_files_capped_and_listed(tmp_path: Path) -> None:
    files = StateFiles(tmp_path / "memory", max_bytes=64)
    files.ensure()
    files.write_state("projects", "para one\n\npara two")
    files.append_state("projects", "para three that pushes the file over the tiny cap xxxx")
    listed = files.files()
    assert len(listed) == 1
    assert listed[0].name == "projects.md"
    assert listed[0].bytes <= 64
    assert listed[0].max_bytes == 64
    assert "para three" in listed[0].content  # newest survived
    assert "para one" not in listed[0].content  # oldest compacted away


# -- compression -------------------------------------------------------------
def _msg(role: str, chars: int) -> dict:
    return {"role": role, "content": "x" * chars}


def test_trigger_math_estimates_and_threshold() -> None:
    # 4 chars/token + 4 overhead: 100 messages x 400 chars ~= 10400 tokens.
    messages = [_msg("user", 400) for _ in range(100)]
    estimate = comp.estimate_tokens(messages)
    assert estimate == 100 * (100 + 4)
    # 70% of 16384 = 11468.8 -> 10400 is below; a 8192-ctx slot is over.
    assert not comp.should_compress(messages, 16384)
    assert comp.should_compress(messages, 8192)


async def test_compress_replaces_middle_and_reports() -> None:
    calls: list[str] = []

    async def summarize(text: str) -> str:
        calls.append(text)
        return "SUMMARY OF OLD TURNS"

    messages = [{"role": "system", "content": "prompt"}] + [_msg("user", 400) for _ in range(20)]
    result = await comp.compress(messages, 2000, summarize)
    assert result is not None
    assert result.messages_summarized == 20 - comp.KEEP_RECENT_MESSAGES
    assert result.trigger_pct > 70
    assert len(calls) == 1

    out = result.messages
    assert out[0]["content"] == "prompt"  # system head intact
    assert out[1]["content"].startswith(comp.SUMMARY_PREFIX)
    assert "SUMMARY OF OLD TURNS" in out[1]["content"]
    assert len(out) == 1 + 1 + comp.KEEP_RECENT_MESSAGES


async def test_second_compression_folds_previous_summary() -> None:
    async def summarize(text: str) -> str:
        return "ROLLED"

    messages = [{"role": "system", "content": "prompt"}] + [_msg("user", 400) for _ in range(20)]
    first = await comp.compress(messages, 2000, summarize)
    assert first is not None
    grown = first.messages + [_msg("user", 2000) for _ in range(10)]
    second = await comp.compress(grown, 2000, summarize)
    assert second is not None
    summaries = [
        m for m in second.messages if str(m.get("content", "")).startswith(comp.SUMMARY_PREFIX)
    ]
    assert len(summaries) == 1  # folded, not stacked


async def test_no_compression_below_threshold() -> None:
    async def summarize(text: str) -> str:  # pragma: no cover - must not be called
        raise AssertionError("summarize must not run below the threshold")

    messages = [_msg("user", 40)]
    assert await comp.compress(messages, 32768, summarize) is None


# -- adaptive verbatim window -------------------------------------------------
@pytest.mark.parametrize(
    ("slot_ctx", "expected"),
    [(4096, 4), (8192, 4), (16383, 4), (16384, 6), (32767, 6), (32768, 8), (131072, 8)],
)
def test_keep_recent_messages_scales_with_ctx(slot_ctx: int, expected: int) -> None:
    assert comp.keep_recent_messages(slot_ctx) == expected


async def test_compress_uses_adaptive_window() -> None:
    async def summarize(text: str) -> str:
        return "S"

    # 16K ctx -> protect 6; make the history big enough to trigger at 16384.
    messages = [{"role": "system", "content": "prompt"}] + [_msg("user", 3000) for _ in range(20)]
    result = await comp.compress(messages, 16384, summarize)
    assert result is not None
    assert result.messages_summarized == 20 - 6
    assert len(result.messages) == 1 + 1 + 6  # head + summary + protected 6


# -- skills ------------------------------------------------------------------
SKILL_MD = """---
name: changelog-entry
description: Add a keepachangelog-style entry matching project conventions.
---

# Writing a changelog entry

1. Read the existing CHANGELOG.md.
"""


def test_frontmatter_parsing() -> None:
    front = parse_frontmatter(SKILL_MD)
    assert front["name"] == "changelog-entry"
    assert front["description"].startswith("Add a keepachangelog")
    assert parse_frontmatter("no frontmatter here") == {}
    assert parse_frontmatter("---\nname: x\n(never closed)") == {}


def test_skill_store_lifecycle(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    skills.ensure()

    created = skills.put("changelog-entry", SKILL_MD, source="human")
    assert created.version == 1
    assert created.source == "human"
    assert created.description.startswith("Add a keepachangelog")

    improved = skills.put("changelog-entry", SKILL_MD + "\n2. More.", source="learned")
    assert improved.version == 2
    assert improved.source == "learned"

    listed = skills.list()
    assert [s.name for s in listed] == ["changelog-entry"]
    # SKILL.md stays portable: version/source live outside the markdown.
    on_disk = (tmp_path / "skills" / "changelog-entry" / "SKILL.md").read_text()
    assert "version" not in parse_frontmatter(on_disk)

    with pytest.raises(ValueError):
        skills.put("Not Kebab", "x", source="human")

    assert skills.delete("changelog-entry")
    assert skills.get("changelog-entry") is None
    assert not skills.delete("changelog-entry")


def test_skill_list_cache_reflects_writes_and_external_changes(tmp_path: Path) -> None:
    """The mtime/stat-invalidated list() cache (audit 2026-07-22) must never go stale:
    a newly saved skill, a delete, a verification flip, AND an out-of-band hand edit on
    disk all show up on the next list()."""
    skills = SkillStore(tmp_path / "skills")
    skills.ensure()
    assert skills.list() == []

    # In-process write invalidates (write-generation bump).
    skills.put("alpha-skill", SKILL_MD.replace("changelog-entry", "alpha-skill"), source="human")
    assert [s.name for s in skills.list()] == ["alpha-skill"]
    # Warm the cache, then confirm a second skill appears (new subdir → signature moves).
    skills.list()
    skills.put("beta-skill", SKILL_MD.replace("changelog-entry", "beta-skill"), source="human")
    assert [s.name for s in skills.list()] == ["alpha-skill", "beta-skill"]

    # A verification flip is visible on the next list().
    skills.list()
    assert skills.mark_verified("alpha-skill") is True
    listed = {s.name: s.verified for s in skills.list()}
    assert listed["alpha-skill"] is True and listed["beta-skill"] is False

    # A delete is visible.
    skills.list()
    assert skills.delete("beta-skill")
    assert [s.name for s in skills.list()] == ["alpha-skill"]

    # An out-of-band hand edit of an existing SKILL.md (bypassing the store) is picked
    # up: the SKILL.md mtime/size is part of the cache signature.
    skills.list()  # warm
    alpha_md = tmp_path / "skills" / "alpha-skill" / "SKILL.md"
    alpha_md.write_text(
        "---\nname: alpha-skill\ndescription: HAND EDITED on disk\n---\n\n# new body\n",
        encoding="utf-8",
    )
    refreshed = {s.name: s.description for s in skills.list()}
    assert refreshed["alpha-skill"] == "HAND EDITED on disk"

    # A hand-dropped new skill dir (never through put) also appears.
    skills.list()  # warm
    dropped = tmp_path / "skills" / "gamma-skill"
    dropped.mkdir()
    (dropped / "SKILL.md").write_text(
        "---\nname: gamma-skill\ndescription: dropped in by hand\n---\n\nbody\n", encoding="utf-8"
    )
    assert [s.name for s in skills.list()] == ["alpha-skill", "gamma-skill"]


def test_skill_without_frontmatter_gets_synthesized_header(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    skill = skills.put("bare-skill", "# Just steps\n1. do it", source="human", description="d")
    front = parse_frontmatter(skill.content)
    assert front["name"] == "bare-skill"
    assert front["description"] == "d"


# -- skill schema validation (model writes only) ------------------------------
def test_validate_skill_markdown_accepts_agentskills_files() -> None:
    assert validate_skill_markdown("changelog-entry", SKILL_MD) == []
    # Indented continuation lines are the minimal-YAML subset and are fine.
    multiline = "---\nname: multi-line\ndescription: first part\n  continued here\n---\n\nbody\n"
    assert validate_skill_markdown("multi-line", multiline) == []


@pytest.mark.parametrize(
    ("name", "content", "expected_fragment"),
    [
        ("no-frontmatter", "# just a body", "must start with a '---'"),
        ("unclosed", "---\nname: unclosed\ndescription: d", "never closed"),
        ("missing-name", "---\ndescription: d\n---\nbody", "missing the required 'name'"),
        ("missing-desc", "---\nname: missing-desc\n---\nx", "missing the required 'description'"),
        ("empty-desc", "---\nname: empty-desc\ndescription:\n---\nbody", "must be non-empty"),
        ("mismatch", "---\nname: other-name\ndescription: d\n---\nbody", "must match"),
        ("bad-case", "---\nname: Bad Case\ndescription: d\n---\nbody", "must be kebab-case"),
        ("bad-line", "---\nname: bad-line\ndescription: d\nnot a pair\n---\nbody", "key: value"),
        ("dupe", "---\nname: dupe\nname: dupe\ndescription: d\n---\nbody", "duplicate"),
    ],
)
def test_validate_skill_markdown_rejections(
    name: str, content: str, expected_fragment: str
) -> None:
    errors = validate_skill_markdown(name, content)
    assert errors, f"expected rejection for {name}"
    assert any(expected_fragment in e for e in errors), errors


def test_model_save_skill_rejects_invalid_frontmatter(service: MemoryService) -> None:
    with pytest.raises(BonsaiError) as err:
        service.model_save_skill("orchestrator", "broken-skill", "# no frontmatter at all")
    assert err.value.status == 400
    assert err.value.code == "skill_invalid"
    assert err.value.message.startswith(SKILL_REJECTION_PREFIX)
    assert service.skills.get("broken-skill") is None

    # A valid file passes and lands as "learned".
    valid = "---\nname: good-skill\ndescription: does the thing\n---\n\n# steps\n"
    skill = service.model_save_skill("orchestrator", "good-skill", valid)
    assert skill.source == "learned"


def test_human_put_stays_lenient(service: MemoryService) -> None:
    """HTTP PUT / hand-dropped files are NOT schema-gated: bare content gets
    synthesized frontmatter exactly as before (only the model path is strict)."""
    skill = service.skills.put("hand-rolled", "# steps only", source="human", description="d")
    assert parse_frontmatter(skill.content)["name"] == "hand-rolled"


# -- skill verification lifecycle ---------------------------------------------
def test_skill_verified_lifecycle(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    skills.ensure()

    created = skills.put("changelog-entry", SKILL_MD, source="learned")
    assert created.verified is False  # every new skill is unproven
    assert created.as_dict()["verified"] is False
    assert created.as_dict(with_content=False)["verified"] is False

    assert skills.mark_verified("changelog-entry") is True
    fetched = skills.get("changelog-entry")
    assert fetched is not None and fetched.verified is True
    # Idempotent, and content/version untouched.
    assert skills.mark_verified("changelog-entry") is True
    fetched = skills.get("changelog-entry")
    assert fetched is not None and fetched.version == 1

    # An improve (content write) resets verification: new instructions, new proof.
    improved = skills.put("changelog-entry", SKILL_MD + "\n2. More.", source="learned")
    assert improved.version == 2
    assert improved.verified is False

    assert skills.mark_verified("no-such-skill") is False


def test_hand_dropped_skill_dir_is_unverified(tmp_path: Path) -> None:
    skills = SkillStore(tmp_path / "skills")
    skill_dir = tmp_path / "skills" / "dropped-in"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: dropped-in\ndescription: d\n---\nbody")
    skill = skills.get("dropped-in")
    assert skill is not None
    assert skill.verified is False


# -- service: mirroring + write enforcement ---------------------------------
def test_identity_and_state_are_mirrored_into_fts(service: MemoryService) -> None:
    (service.files._identity_file).write_text("# identity\n\nI prefer tabs over spaces.\n")
    service.files.write_state("open threads", "the zeppelin refactor is half done")
    service.remirror_files()

    identity_hits = [
        h for h in service.store.search("tabs spaces") if h["entry"]["layer"] == "identity"
    ]
    assert identity_hits
    state_hits = [h for h in service.store.search("zeppelin") if h["entry"]["layer"] == "state"]
    assert state_hits

    # Idempotent: re-mirroring does not duplicate rows.
    before = service.store.list_entries("state")[1]
    service.remirror_files()
    assert service.store.list_entries("state")[1] == before


def test_identity_entry_surface_stays_read_only(service: MemoryService) -> None:
    """Identity is edited ONLY via update_state_file (PUT /v1/memory/state/identity.md);
    the entry surface still refuses it so there is exactly one edit path."""
    service.remirror_files()
    identity_entries, _ = service.store.list_entries("identity")
    assert identity_entries
    with pytest.raises(BonsaiError) as err:
        service.update_entry(identity_entries[0].id, title="hijack")
    assert err.value.status == 400
    with pytest.raises(BonsaiError):
        service.delete_entry(identity_entries[0].id)
    with pytest.raises(BonsaiError):
        service.create_entry("identity", "x", "y")


# -- state-file editing over HTTP (PUT /v1/memory/state/{name}) ---------------
def test_update_state_file_identity_and_state_roundtrip(service: MemoryService) -> None:
    service.files.write_state("open threads", "the zeppelin refactor is half done")
    service.remirror_files()

    updated = service.update_state_file("identity.md", "# identity\n\nI answer in haiku.")
    assert updated.name == "identity.md"
    assert updated.bytes == len(updated.content.encode())
    assert service.files.identity() == "# identity\n\nI answer in haiku."
    # The mirror re-indexed immediately: recall sees the edit without a restart.
    assert any(h["entry"]["layer"] == "identity" for h in service.store.search("answer in haiku"))

    updated = service.update_state_file("open-threads.md", "the zeppelin refactor SHIPPED")
    assert updated.name == "open-threads.md"
    files = {f.name: f for f in service.files.all_files()}
    assert files["open-threads.md"].content == "the zeppelin refactor SHIPPED"
    assert "identity.md" in files  # all_files() = identity + state


@pytest.mark.parametrize("name", ["nope.md", "../identity.md", "state/../../etc/passwd", ""])
def test_update_state_file_unknown_names_are_404(service: MemoryService, name: str) -> None:
    """The known set is bare filenames of existing files — traversal strings can never
    name a path, and nothing new is creatable."""
    with pytest.raises(BonsaiError) as err:
        service.update_state_file(name, "content")
    assert err.value.status == 404
    assert err.value.code == "state_file_unknown"
    assert not (service.files._dir.parent / "etc").exists()


def test_update_state_file_rejects_oversized_content(service: MemoryService) -> None:
    before = service.files.identity()
    huge = "x" * (service.files.max_bytes + 1)
    with pytest.raises(BonsaiError) as err:
        service.update_state_file("identity.md", huge)
    assert err.value.status == 400
    assert err.value.code == "state_file_too_large"
    assert service.files.identity() == before  # rejected loudly, never compacted


# -- client identities (api.md 2026-07-22b) ----------------------------------
def test_identity_files_are_seeded_on_first_run(service: MemoryService) -> None:
    """startup() seeds identity.md + both client overlays from the packaged copies."""
    files = {f.name for f in service.files.all_files()}
    assert {"identity.md", "identity-dai.md", "identity-sentei.md"} <= files
    # The seeds are the real personas (not the placeholder template).
    assert "orchestrator" in service.files.identity().lower()
    assert service.files.client_overlay("sentei")  # non-empty coding overlay
    assert service.files.client_overlay("dai")  # non-empty general overlay
    assert service.files.client_overlay("other") == ""  # unknown/other -> no overlay


def test_client_overlays_are_editable_state_files(service: MemoryService) -> None:
    updated = service.update_state_file("identity-sentei.md", "# sentei\n\nBe terse.")
    assert updated.name == "identity-sentei.md"
    assert service.files.client_overlay("sentei") == "# sentei\n\nBe terse."
    # It lives directly in the memory dir (like identity.md), never compacted into state/.
    assert (service.files._dir / "identity-sentei.md").exists()
    assert "identity-sentei.md" not in {f.name for f in service.files.files()}


def test_overlay_edit_is_rejected_oversized(service: MemoryService) -> None:
    with pytest.raises(BonsaiError) as err:
        service.update_state_file("identity-dai.md", "x" * (service.files.max_bytes + 1))
    assert err.value.code == "state_file_too_large"


# -- sessions mode filter (api.md 2026-07-22b) -------------------------------
def test_list_sessions_filters_by_mode(store: MemoryStore) -> None:
    store.ensure_session("c1", "chat")
    store.ensure_session("k1", "code")
    store.ensure_session("k2", "code")
    for sid in ("c1", "k1", "k2"):
        store.add_message(sid, "user", "hi")

    chat_ids = {s["id"] for s in store.list_sessions(mode="chat")}
    code_ids = {s["id"] for s in store.list_sessions(mode="code")}
    assert chat_ids == {"c1"}
    assert code_ids == {"k1", "k2"}
    assert len(store.list_sessions()) == 3  # no filter -> all


def test_state_entry_writes_go_through_bounded_file(service: MemoryService) -> None:
    entry = service.create_entry("state", "active project", "building the bonsai stack")
    assert entry.layer == "state"
    files = {f.name: f for f in service.files.files()}
    assert "active-project.md" in files
    assert "bonsai stack" in files["active-project.md"].content

    service.delete_entry(entry.id)
    assert "active-project.md" not in {f.name for f in service.files.files()}


def test_model_writes_enforced_to_orchestrator_role(service: MemoryService) -> None:
    entry = service.model_write_memory("orchestrator", "archive", "t", "c")
    assert entry.layer == "archive"
    for role in ("worker", "utility"):
        with pytest.raises(BonsaiError) as err:
            service.model_write_memory(role, "archive", "t", "c")
        assert err.value.code == "writer_role_required"
        with pytest.raises(BonsaiError):
            service.model_save_skill(role, "some-skill", "# x")
