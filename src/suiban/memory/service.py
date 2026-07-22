"""MemoryService: the facade over the four layers (docs/memory.md).

Files on disk (identity + state) are the source of truth; they are mirrored into the
SQLite store on startup and on every change so ONE FTS5 index covers every layer.
Mirror rows use deterministic ids (mem_file_<slug>) so re-mirroring is idempotent.

Write enforcement: `require_writer_role()` rejects any model-driven write whose slot
role is not `orchestrator` — defense in depth behind the registry-level gating.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from suiban.errors import BonsaiError
from suiban.memory.skills import Skill, SkillStore, skill_rejection, validate_skill_markdown
from suiban.memory.state_files import IDENTITY_NAME, StateFile, StateFiles, slugify
from suiban.memory.store import MemoryEntry, MemoryStore

WRITER_ROLE = "orchestrator"


def _mirror_id(layer: str, name: str) -> str:
    digest = hashlib.sha256(f"{layer}/{name}".encode()).hexdigest()[:12]
    return f"mem_file_{digest}"


def require_writer_role(role: str, action: str) -> None:
    """Server-side enforcement of the one-writer rule (docs/memory.md §7)."""
    if role != WRITER_ROLE:
        raise BonsaiError(
            409,
            f"{action} is restricted to the 27B orchestrator slot; a {role!r} slot attempted it.",
            code="writer_role_required",
        )


class MemoryService:
    def __init__(self, home: Path) -> None:
        self._memory_dir = home / "memory"
        self.files = StateFiles(self._memory_dir)
        self.store = MemoryStore(self._memory_dir / "memory.sqlite")
        self.skills = SkillStore(home / "skills")

    def startup(self) -> None:
        self.files.ensure()
        self.skills.ensure()
        self.remirror_files()

    def close(self) -> None:
        self.store.close()

    # -- file mirroring ----------------------------------------------------
    def remirror_files(self) -> None:
        """Rebuild the identity/state mirror rows from the files (idempotent)."""
        self.store.delete_layer_mirror("identity")
        self.store.delete_layer_mirror("state")
        identity = self.files.identity()
        if identity.strip():
            self.store.add_entry(
                "identity", "identity.md", identity, entry_id=_mirror_id("identity", "identity.md")
            )
        for state_file in self.files.files():
            self.store.add_entry(
                "state",
                state_file.name,
                state_file.content,
                entry_id=_mirror_id("state", state_file.name),
            )

    # -- entries (HTTP + tool surface) -------------------------------------
    def create_entry(
        self,
        layer: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        *,
        source_session: str | None = None,
    ) -> MemoryEntry:
        if layer == "state":
            # State lives in the bounded file; the DB row is a mirror.
            state_file = self.files.write_state(title, content)
            self.store.delete_entry(_mirror_id("state", state_file.name))
            return self.store.add_entry(
                "state",
                state_file.name,
                state_file.content,
                tags,
                source_session=source_session,
                entry_id=_mirror_id("state", state_file.name),
            )
        if layer == "archive":
            return self.store.add_entry(
                "archive", title, content, tags, source_session=source_session
            )
        raise BonsaiError(
            400,
            f"layer must be 'state' or 'archive' (identity is edited via "
            f"PUT /v1/memory/state/{IDENTITY_NAME}, never created), got {layer!r}",
            code="layer_not_writable",
        )

    # -- bounded state files over HTTP (PUT /v1/memory/state/{name}) ---------
    def update_state_file(self, name: str, content: str) -> StateFile:
        """Overwrite ONE existing bounded state file (identity.md included) with
        human-authored content.

        `name` must be a bare filename from the known set (identity.md + existing
        state/*.md) — anything else is a 404, so traversal strings can never name a
        path and no new files are creatable through this route. Oversized content is
        a 400 (`state_file_too_large`): a human edit is rejected loudly, never
        silently compacted (compaction is for model-driven appends)."""
        known = {f.name for f in self.files.all_files()}
        if name not in known:
            raise BonsaiError(
                404,
                f"no such state file: {name!r} (known: {', '.join(sorted(known))}; "
                "new files are not creatable over HTTP)",
                code="state_file_unknown",
            )
        size = len(content.encode("utf-8"))
        if size > self.files.max_bytes:
            raise BonsaiError(
                400,
                f"content is {size} bytes; state files are capped at "
                f"{self.files.max_bytes} bytes — trim it and retry",
                code="state_file_too_large",
            )
        if self.files.is_identity_file(name):
            # identity.md and the client overlays (identity-dai.md / identity-sentei.md)
            # are written verbatim to the memory dir, never compacted into state/.
            updated = self.files.write_identity_file(name, content)
        else:
            updated = self.files.write_state(Path(name).stem, content)
        # Refresh the FTS mirrors so recall (identity injection included) sees the
        # edit immediately, not at the next startup.
        self.remirror_files()
        return updated

    def update_entry(
        self,
        entry_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        entry = self.store.get_entry(entry_id)
        if entry is None:
            raise BonsaiError(404, f"no such memory entry: {entry_id}", code="memory_not_found")
        if entry.layer == "identity":
            raise BonsaiError(
                400,
                f"identity is not editable via the entry surface; use "
                f"PUT /v1/memory/state/{IDENTITY_NAME} (or a text editor on "
                "~/.bonsai/memory/identity.md)",
                code="identity_read_only",
            )
        if entry.layer == "state" and content is not None:
            state_file = self.files.write_state(Path(entry.title).stem, content)
            content = state_file.content  # post-compaction truth
        updated = self.store.update_entry(entry_id, title=title, content=content, tags=tags)
        assert updated is not None
        return updated

    def delete_entry(self, entry_id: str) -> None:
        entry = self.store.get_entry(entry_id)
        if entry is None:
            raise BonsaiError(404, f"no such memory entry: {entry_id}", code="memory_not_found")
        if entry.layer == "identity":
            raise BonsaiError(
                400,
                "identity is never deletable over HTTP; clear it via "
                f"PUT /v1/memory/state/{IDENTITY_NAME} or edit the file",
                code="identity_read_only",
            )
        if entry.layer == "state":
            self.files.delete_state(Path(entry.title).stem)
        self.store.delete_entry(entry_id)

    def delete_state_file(self, name: str) -> None:
        """Delete ONE bounded state file (DELETE /v1/memory/state/{name}).

        `name` must be a bare filename from the known set; anything else is a 404 so
        traversal strings can never name a path. Identity files (identity.md and the
        client overlays) are never deletable over HTTP — the same protection PUT gives
        their contents. The FTS mirror entry is dropped alongside the file."""
        known = {f.name for f in self.files.all_files()}
        if name not in known:
            raise BonsaiError(
                404,
                f"no such state file: {name!r} (known: {', '.join(sorted(known))})",
                code="state_file_unknown",
            )
        if self.files.is_identity_file(name):
            raise BonsaiError(
                400,
                f"{name} is an identity file and is never deletable over HTTP; "
                f"clear its contents via PUT /v1/memory/state/{name} instead",
                code="identity_read_only",
            )
        self.files.delete_state(Path(name).stem)
        self.store.delete_entry(_mirror_id("state", name))

    def delete_session(self, session_id: str) -> None:
        """Delete an archived session/chat (DELETE /v1/memory/sessions/{id})."""
        if not self.store.delete_session(session_id):
            raise BonsaiError(404, f"no such session: {session_id}", code="session_not_found")

    # -- model-driven writes (27B reflection only) --------------------------
    def model_write_memory(
        self,
        role: str,
        layer: str,
        title: str,
        content: str,
        tags: list[str] | None = None,
        *,
        session_id: str | None = None,
    ) -> MemoryEntry:
        require_writer_role(role, "memory_write")
        source = session_id if session_id and self.store.session_exists(session_id) else None
        return self.create_entry(layer, title, content, tags, source_session=source)

    def model_save_skill(self, role: str, name: str, content: str) -> Skill:
        """Model-driven skill write (skill_save / skill_improve): 27B-only AND
        schema-validated — an invalid SKILL.md is a structured 400 (`skill_invalid`)
        whose message (stable `invalid skill` prefix) the reflection retry recognizes
        and feeds back to the model once. Human writes (HTTP PUT) stay lenient."""
        require_writer_role(role, "skill_save")
        errors = validate_skill_markdown(name, content)
        if errors:
            raise BonsaiError(400, skill_rejection(name, errors), code="skill_invalid")
        return self.skills.put(name, content, source="learned")

    # -- state slug helper ---------------------------------------------------
    @staticmethod
    def state_name_for(title: str) -> str:
        return slugify(title)
