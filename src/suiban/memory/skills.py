"""Skills: agentskills.io-compatible markdown under ~/.bonsai/skills/<name>/SKILL.md.

The SKILL.md stays portable — only `name` and `description` frontmatter are required.
Version / provenance / updated_at are suiban bookkeeping and live OUTSIDE the markdown,
in <name>/meta.json (docs/memory.md §6).

Frontmatter parsing is a minimal YAML subset (key: value + indented continuations) —
enough for agentskills.io files. TODO(v1.1): swap in a real YAML parser if skills in
the wild use richer frontmatter than key/value pairs.

Model-driven writes are schema-validated (validate_skill_markdown) and rejected with
a structured message (skill_rejection) the reflection retry can recognize; human
writes stay lenient. meta.json additionally tracks `verified` — False on every
content write, flipped by mark_verified() when a run that used the skill succeeds.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SKILL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Every model-path rejection starts with this exact prefix so the reflection retry
# (memory/reflection.py) can recognize a schema rejection in a ToolResult without the
# tool layer having to grow a structured error channel.
SKILL_REJECTION_PREFIX = "invalid skill"

_FRONTMATTER_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")


def _ignore_symlinks(dirpath: str, names: list[str]) -> set[str]:
    """copytree `ignore` callback (SEC-3): drop every symlinked entry so a hostile source
    skill directory cannot pull external file contents (e.g. `data -> /etc/passwd`) into
    the store. Symlinks are skipped entirely rather than copied or followed."""
    return {name for name in names if os.path.islink(os.path.join(dirpath, name))}


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_frontmatter(content: str) -> dict[str, str]:
    """Extract simple `key: value` pairs (with indented continuation lines) from a
    leading `---` frontmatter block. Returns {} when there is none."""
    if not content.startswith("---"):
        return {}
    lines = content.splitlines()
    fields: dict[str, str] = {}
    current_key: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        if line[:1].isspace() and current_key:
            fields[current_key] += " " + line.strip()
            continue
        match = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", line)
        if match:
            current_key = match.group(1)
            fields[current_key] = match.group(2).strip()
    return {}  # no closing --- => not valid frontmatter


def validate_skill_markdown(name: str, content: str) -> list[str]:
    """agentskills.io-compatible frontmatter validation for MODEL-driven writes
    (skill_save / skill_improve). Returns a list of human-readable errors; empty
    means valid. Checks: a closed `---` frontmatter block whose every line is a
    `key: value` pair (or an indented continuation — the minimal YAML subset this
    module parses), `name` present + kebab-case + matching the skill's directory
    name, `description` present and non-empty.

    Human writes (HTTP PUT, hand-dropped directories) stay lenient by design —
    SkillStore.put synthesizes minimal frontmatter for bare content and the store
    tolerates whatever a human left on disk. Only the model path is strict: a model
    that cannot produce two frontmatter keys should not be teaching skills."""
    errors: list[str] = []
    if not SKILL_NAME_RE.match(name):
        errors.append(
            f"skill name must be kebab-case (lowercase a-z/0-9 words joined by '-'), got {name!r}"
        )
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        errors.append(
            "content must start with a '---' frontmatter block containing 'name' and 'description'"
        )
        return errors
    closing = next((i for i, line in enumerate(lines[1:], start=1) if line.strip() == "---"), None)
    if closing is None:
        errors.append("frontmatter block is never closed — add a '---' line after the fields")
        return errors
    fields: dict[str, str] = {}
    current_key: str | None = None
    for lineno, line in enumerate(lines[1:closing], start=2):
        if not line.strip():
            current_key = None
            continue
        if line[:1].isspace():
            if current_key is None:
                errors.append(
                    f"frontmatter line {lineno} is indented but continues no key: {line.strip()!r}"
                )
            continue
        match = _FRONTMATTER_KEY_RE.match(line)
        if match is None:
            errors.append(f"frontmatter line {lineno} is not a 'key: value' pair: {line.strip()!r}")
            current_key = None
            continue
        current_key = match.group(1)
        if current_key in fields:
            errors.append(f"duplicate frontmatter key {current_key!r}")
        fields[current_key] = match.group(2).strip()
    front_name = fields.get("name")
    if front_name is None:
        errors.append("frontmatter is missing the required 'name' field")
    elif not SKILL_NAME_RE.match(front_name):
        errors.append(f"frontmatter 'name' must be kebab-case, got {front_name!r}")
    elif front_name != name:
        errors.append(f"frontmatter 'name' ({front_name!r}) must match the skill name ({name!r})")
    if "description" not in fields:
        errors.append("frontmatter is missing the required 'description' field")
    elif not parse_frontmatter(content).get("description", "").strip():
        # parse_frontmatter merges indented continuation lines; only a description
        # empty AFTER merging is genuinely empty.
        errors.append("frontmatter 'description' must be non-empty")
    return errors


def skill_rejection(name: str, errors: list[str]) -> str:
    """The structured rejection message for an invalid model-driven skill write —
    stable prefix (SKILL_REJECTION_PREFIX), every validator error, and the fix
    instruction, so a retry has everything it needs in one string."""
    return (
        f"{SKILL_REJECTION_PREFIX} {name!r}: {'; '.join(errors)}. "
        "Resend the FULL SKILL.md starting with a closed '---' frontmatter block "
        "whose 'name' matches the skill name and whose 'description' is non-empty."
    )


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: int
    updated_at: str
    source: str  # "seed" | "learned" | "human"
    content: str
    # False until a run that actually USED the skill (injected into its context)
    # completed successfully; every content write resets it (docs/memory.md §6).
    # Additive optional field on the api.md Skill object.
    verified: bool = False

    def as_dict(self, *, with_content: bool = True) -> dict:
        out = {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "updated_at": self.updated_at,
            "source": self.source,
            "verified": self.verified,
        }
        if with_content:
            out["content"] = self.content
        return out


class SkillStore:
    def __init__(self, skills_dir: Path) -> None:
        self._dir = skills_dir
        # list() cache (audit 2026-07-22): the chat hot path calls list() on EVERY
        # request (via _inject_skill_context), which globbed + re-read + re-parsed every
        # SKILL.md and meta.json each time. We now cache the parsed list and invalidate
        # on a cheap stat-only signature. The signature mixes a process-local write
        # generation (bumped by every in-process mutation — always correct regardless of
        # mtime granularity) with per-skill (SKILL.md mtime+size, meta.json mtime) so
        # hand edits on disk that bypass the store are still picked up without a read.
        self._cache: list[Skill] | None = None
        self._cache_sig: tuple | None = None
        self._write_gen = 0

    def ensure(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def _invalidate(self) -> None:
        """Force the next list() to rebuild after an in-process write/delete."""
        self._write_gen += 1

    def _signature(self) -> tuple:
        """Stat-only fingerprint of the skills dir — no file reads or parsing. Changes
        whenever a skill is added, removed, or rewritten (the SKILL.md/meta.json
        mtime/size move), and whenever an in-process mutation bumps _write_gen."""
        parts: list[tuple[str, int, int, int]] = []
        try:
            with os.scandir(self._dir) as it:
                subdirs = sorted(entry.name for entry in it if entry.is_dir())
        except FileNotFoundError:
            return (self._write_gen,)
        for name in subdirs:
            try:
                skill_stat = (self._dir / name / "SKILL.md").stat()
            except (FileNotFoundError, NotADirectoryError):
                continue  # a dir without SKILL.md is not a skill (list() skips it too)
            try:
                meta_mtime = (self._dir / name / "meta.json").stat().st_mtime_ns
            except FileNotFoundError:
                meta_mtime = 0
            parts.append((name, skill_stat.st_mtime_ns, skill_stat.st_size, meta_mtime))
        return (self._write_gen, tuple(parts))

    def _skill_file(self, name: str) -> Path:
        return self._dir / name / "SKILL.md"

    def _meta_file(self, name: str) -> Path:
        return self._dir / name / "meta.json"

    def _load_meta(self, name: str) -> dict:
        meta_path = self._meta_file(name)
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                pass  # corrupt meta: fall through to defaults, never crash
        # A skill dir without meta was dropped in by a human — unverified until used.
        return {"version": 1, "source": "human", "updated_at": _now_iso(), "verified": False}

    def list(self) -> list[Skill]:
        self.ensure()
        signature = self._signature()
        if self._cache is not None and signature == self._cache_sig:
            return list(self._cache)  # fresh list, shared immutable Skill objects
        out = []
        for skill_file in sorted(self._dir.glob("*/SKILL.md")):
            skill = self.get(skill_file.parent.name)
            if skill is not None:
                out.append(skill)
        self._cache = out
        self._cache_sig = signature
        return list(out)

    def get(self, name: str) -> Skill | None:
        skill_file = self._skill_file(name)
        if not SKILL_NAME_RE.match(name) or not skill_file.is_file():
            return None
        content = skill_file.read_text(encoding="utf-8")
        front = parse_frontmatter(content)
        meta = self._load_meta(name)
        return Skill(
            name=name,
            description=front.get("description", ""),
            version=int(meta.get("version", 1)),
            updated_at=str(meta.get("updated_at", _now_iso())),
            source=str(meta.get("source", "human")),
            content=content,
            verified=bool(meta.get("verified", False)),
        )

    def put(self, name: str, content: str, *, source: str, description: str | None = None) -> Skill:
        """Create or update a skill. `source` is who wrote it: "human" (HTTP PUT) or
        "learned" (27B reflection). Version increments on every update."""
        if not SKILL_NAME_RE.match(name):
            raise ValueError(f"skill name must be kebab-case: {name!r}")
        existing = self.get(name)
        version = existing.version + 1 if existing else 1
        skill_dir = self._dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        if not content.startswith("---"):
            # Keep the file agentskills.io-compatible: synthesize minimal frontmatter.
            desc = description or (existing.description if existing else name)
            content = f"---\nname: {name}\ndescription: {desc}\n---\n\n{content}"
        self._skill_file(name).write_text(content, encoding="utf-8")
        # Every content write resets verification: new or changed instructions are
        # unproven until a run that used them completes successfully.
        meta = {"version": version, "source": source, "updated_at": _now_iso(), "verified": False}
        self._meta_file(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._invalidate()
        skill = self.get(name)
        assert skill is not None
        if description and not skill.description:
            skill = Skill(**{**skill.__dict__, "description": description})
        return skill

    def import_skill_dir(self, name: str, src_dir: Path, *, source: str = "imported") -> Skill:
        """Copy a whole agentskills.io skill directory (`SKILL.md` + `scripts/` + any
        supporting files) from `src_dir` into ~/.bonsai/skills/<name>/, replacing an
        existing directory of that name, then write meta.json (source, version 1,
        verified=false). Used by skill import (memory/skill_import.py); the caller has
        already validated the SKILL.md frontmatter. `source` defaults to "imported" so
        an imported skill's provenance is visible in `GET /v1/skills` and unproven until
        a run uses it."""
        if not SKILL_NAME_RE.match(name):
            raise ValueError(f"skill name must be kebab-case: {name!r}")
        dest = self._dir / name
        src = src_dir.resolve()
        if src == dest.resolve():
            raise ValueError(f"cannot import skill {name!r} from its own store directory")
        self.ensure()
        if dest.exists():
            shutil.rmtree(dest)
        # SEC-3: never follow symlinks in a (possibly hostile) source skill dir into the
        # store. `ignore` drops every symlinked entry so a `link -> /etc/passwd` cannot
        # smuggle external file contents into ~/.bonsai/skills/.
        shutil.copytree(src, dest, ignore=_ignore_symlinks)
        if not self._skill_file(name).is_file():
            # The source SKILL.md was itself a symlink and was skipped: reject cleanly so
            # the importer reports it (skipped) rather than crashing on a missing file.
            shutil.rmtree(dest, ignore_errors=True)
            raise ValueError(f"skill {name!r}: SKILL.md is a symlink and is not imported")
        meta = {"version": 1, "source": source, "updated_at": _now_iso(), "verified": False}
        self._meta_file(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._invalidate()
        skill = self.get(name)
        assert skill is not None
        return skill

    def mark_verified(self, name: str) -> bool:
        """Flip a skill to verified: a run that had this skill injected completed
        successfully (docs/memory.md §6). Persisted in meta.json (the skills meta
        store) — content and updated_at are untouched, verification is bookkeeping,
        not an edit. Returns False for unknown names."""
        if not SKILL_NAME_RE.match(name) or not self._skill_file(name).is_file():
            return False
        meta = self._load_meta(name)
        if not meta.get("verified"):
            meta["verified"] = True
            self._meta_file(name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self._invalidate()
        return True

    def delete(self, name: str) -> bool:
        skill_dir = self._dir / name
        if not SKILL_NAME_RE.match(name) or not skill_dir.is_dir():
            return False
        for child in sorted(skill_dir.rglob("*"), reverse=True):
            child.unlink() if child.is_file() else child.rmdir()
        skill_dir.rmdir()
        self._invalidate()
        return True
