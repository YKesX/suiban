"""Import agentskills.io SKILL.md skill directories from other ecosystems (api.md
2026-07-22c, POST /v1/skills/import + `suiban skills import`).

suiban skills ARE agentskills.io `SKILL.md` directory skills, so they are portable BOTH
ways: any `<name>/SKILL.md` directory another agentskills.io tool ships imports here, and
a suiban skill directory drops straight into those tools. This module scans a source's
skills folder, validates each skill's frontmatter with the SAME validator the model-write
path uses (`validate_skill_markdown`), copies the good ones into ~/.bonsai/skills/<name>/
(source="imported", verified=false), and reports the malformed ones in a `skipped` list
rather than crashing.

Known sources (both verified as agentskills.io SKILL.md skill sets, both MIT — see
NOTICE): openclaw (github.com/openclaw/openclaw) and hermes (github.com/nousresearch/
hermes-agent). A bare `path` imports any directory the user names, scanned recursively.
No code is copied from either project — only their SKILL.md skills, via the shared
agentskills.io format.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from suiban.memory.skills import SkillStore, validate_skill_markdown

SOURCES = ("openclaw", "hermes", "path")

# The per-user bonsai home directory name (~/.bonsai). A `path` import must never scan a
# directory that CONTAINS this, since that would sweep the skills store (and usually the
# whole home) into the importer.
BONSAI_DIR_NAME = ".bonsai"

# Scan bounds (SEC-2): a `path` import scanning an over-broad root (a filesystem root, the
# home, a deep tree) must be refused or bounded, never allowed to rglob the whole disk.
# Read live so tests can tighten them via monkeypatch.
MAX_SCAN_DIRS = 5000  # directories visited before the scan is refused
MAX_IMPORT_SKILLS = 200  # skill directories collected before discovery stops
MAX_SCAN_DEPTH = 12  # directory levels below the root that are descended into


class SkillImportError(Exception):
    """A source could not be scanned at all (e.g. a `path` that does not exist, or a root
    so broad it would sweep the filesystem). Distinct from a malformed individual skill,
    which is skipped and reported, not raised."""


@dataclass
class ImportResult:
    imported: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # {"name": ..., "reason": ...}


def _source_roots(source: str, path: str | None, user_home: Path) -> list[Path]:
    """The directories to scan for a given source. `user_home` (defaulting to the real
    home) locates the per-ecosystem skill folders; it is injectable so tests never touch
    a developer's real ~/.openclaw or ~/.hermes."""
    if source == "openclaw":
        # The openclaw workspace skills, plus a checked-out repo's .agents/skills.
        roots = [user_home / ".openclaw" / "workspace" / "skills"]
        if path:
            roots.append(Path(path) / ".agents" / "skills")
        return roots
    if source == "hermes":
        # The hermes skills dir, plus a repo's optional-skills/<cat>/<name> tree.
        roots = [user_home / ".hermes" / "skills"]
        if path:
            roots.append(Path(path) / "optional-skills")
        return roots
    if source == "path":
        if not path:
            raise SkillImportError("source 'path' requires a directory path")
        root = Path(path).expanduser()
        _reject_unbounded_root(root, user_home)
        if not root.is_dir():
            raise SkillImportError(f"no such directory: {path}")
        return [root]
    raise SkillImportError(
        f"unknown import source {source!r}; expected one of {', '.join(SOURCES)}"
    )


def _reject_unbounded_root(root: Path, user_home: Path) -> None:
    """Refuse a `path` source whose root is so broad the scan would sweep the filesystem:
    a filesystem root (`/`, `C:\\`), the user's home (`$HOME` / `~`), or any ancestor of
    the bonsai home (which would pull the skills store itself into the importer)."""
    resolved = root.resolve()
    home = user_home.resolve()
    bonsai_home = (home / BONSAI_DIR_NAME).resolve()
    if resolved.parent == resolved:
        raise SkillImportError(
            f"refusing to import from a filesystem root ({resolved}); point at a skills folder"
        )
    if resolved == home:
        raise SkillImportError(
            f"refusing to import from the home directory ({resolved}); point at a skills folder"
        )
    if bonsai_home.is_relative_to(resolved):
        raise SkillImportError(
            f"refusing to import from {resolved}: it contains the bonsai home "
            f"({bonsai_home}); point at a specific skills folder instead"
        )


def _discover_skill_dirs(roots: list[Path]) -> list[Path]:
    """Every `<name>/SKILL.md` under the given roots (recursive: handles the flat
    <name>/SKILL.md layout and hermes' optional-skills/<cat>/<name>/SKILL.md alike).
    De-duplicated by skill directory; a missing root contributes nothing.

    Bounded (SEC-2): the walk descends at most MAX_SCAN_DEPTH levels, visits at most
    MAX_SCAN_DIRS directories (a broader tree is refused with a clear error rather than
    scanned), collects at most MAX_IMPORT_SKILLS skills, never follows symlinked
    directories, and visits directory entries in sorted order for a deterministic result."""
    seen: set[Path] = set()
    found: list[Path] = []
    dirs_scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        base = root.resolve()
        base_depth = len(base.parts)
        for dirpath, dirnames, filenames in os.walk(base):  # followlinks=False by default
            dirs_scanned += 1
            if dirs_scanned > MAX_SCAN_DIRS:
                raise SkillImportError(
                    f"refusing to scan {base}: more than {MAX_SCAN_DIRS} directories "
                    "(point at a specific skills folder, not a broad tree)"
                )
            dirnames.sort()
            if len(Path(dirpath).parts) - base_depth >= MAX_SCAN_DEPTH:
                dirnames[:] = []  # stop descending past the depth cap
            if "SKILL.md" in filenames:
                skill_dir = Path(dirpath).resolve()
                if skill_dir not in seen:
                    seen.add(skill_dir)
                    found.append(skill_dir)
                    if len(found) >= MAX_IMPORT_SKILLS:
                        return found
    return found


def import_skills(
    store: SkillStore,
    source: str,
    path: str | None = None,
    *,
    user_home: Path | None = None,
) -> ImportResult:
    """Scan `source` for `<name>/SKILL.md` skills, validate each, copy the valid ones
    into the store (source="imported", verified=false), and skip the malformed ones with
    a reason. Raises SkillImportError only when the source itself cannot be scanned (an
    absent `path`); a bad individual skill is reported in `skipped`, never raised."""
    home = user_home if user_home is not None else Path.home()
    roots = _source_roots(source, path, home)
    result = ImportResult()
    for skill_dir in _discover_skill_dirs(roots):
        name = skill_dir.name
        try:
            content = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            result.skipped.append({"name": name, "reason": f"could not read SKILL.md: {exc}"})
            continue
        errors = validate_skill_markdown(name, content)
        if errors:
            result.skipped.append({"name": name, "reason": "; ".join(errors)})
            continue
        try:
            store.import_skill_dir(name, skill_dir, source="imported")
        except (OSError, ValueError) as exc:
            result.skipped.append({"name": name, "reason": f"copy failed: {exc}"})
            continue
        result.imported.append(name)
    return result
