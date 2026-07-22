"""Identity + bounded state files under ~/.bonsai/memory/ (docs/memory.md §1).

State files are byte-capped: a write that would exceed `max_bytes` is compacted by
dropping the OLDEST content first (paragraphs from the top of the file — state files
are appended to over time, so the top is the oldest). Files on disk are the source of
truth; the service mirrors them into the FTS index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

STATE_MAX_BYTES = 8 * 1024

IDENTITY_TEMPLATE = """\
# identity

<!-- Who you are and your standing preferences, in your own words.
     This file is yours: suiban reads it (recall tools, plus a small budget-capped
     excerpt when it matches your latest message — docs/memory.md §3) but no MODEL
     ever writes it. Edit with any text editor, or over HTTP via
     PUT /v1/memory/state/identity.md. -->
"""

IDENTITY_NAME = "identity.md"

# Client-identity overlays (api.md 2026-07-22b): the base identity plus the overlay
# matching the request's X-Bonsai-Client header are injected into the system prompt.
# sentei → the coding overlay, dai → the general overlay, other/unknown → base only.
CLIENT_OVERLAY_NAMES: tuple[str, ...] = ("identity-dai.md", "identity-sentei.md")
IDENTITY_FILE_NAMES: tuple[str, ...] = (IDENTITY_NAME, *CLIENT_OVERLAY_NAMES)
_CLIENT_OVERLAY_FOR = {"dai": "identity-dai.md", "sentei": "identity-sentei.md"}

# Packaged seed copies (real personas, not placeholders) shipped in the repo; copied
# into ~/.bonsai/memory/ on first run. Editable afterwards, so seeds never overwrite.
IDENTITY_SEED_DIR = Path(__file__).parent / "identities"


def slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "untitled"


def compact_oldest(content: str, max_bytes: int) -> str:
    """Drop oldest (leading) paragraphs until the content fits max_bytes. If a single
    paragraph is still too big, hard-trim its head — the newest bytes always survive."""
    if len(content.encode("utf-8")) <= max_bytes:
        return content
    paragraphs = content.split("\n\n")
    while len(paragraphs) > 1 and len("\n\n".join(paragraphs).encode("utf-8")) > max_bytes:
        paragraphs.pop(0)
    result = "\n\n".join(paragraphs)
    encoded = result.encode("utf-8")
    if len(encoded) > max_bytes:
        result = encoded[-max_bytes:].decode("utf-8", errors="ignore")
    return result


@dataclass(frozen=True)
class StateFile:
    name: str
    content: str
    bytes: int
    max_bytes: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "content": self.content,
            "bytes": self.bytes,
            "max_bytes": self.max_bytes,
        }


class StateFiles:
    """The memory/ file layer: identity.md + state/*.md."""

    def __init__(self, memory_dir: Path, max_bytes: int = STATE_MAX_BYTES) -> None:
        self._dir = memory_dir
        self._state_dir = memory_dir / "state"
        self._identity_file = memory_dir / "identity.md"
        self.max_bytes = max_bytes

    def ensure(self) -> None:
        """First-run seeding: copy the packaged identity.md + the client overlays into
        ~/.bonsai/memory/ (api.md 2026-07-22b). Existing files are never overwritten —
        the seeds are a starting point, then the files are the user's to edit."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        for name in IDENTITY_FILE_NAMES:
            dest = self._dir / name
            if dest.exists():
                continue
            seed = IDENTITY_SEED_DIR / name
            if seed.exists():
                dest.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            elif name == IDENTITY_NAME:  # overlays are optional; base always exists
                dest.write_text(IDENTITY_TEMPLATE, encoding="utf-8")

    # -- identity + client overlays ----------------------------------------
    def _top_level_file(self, name: str) -> StateFile:
        """One identity-family file (identity.md / identity-<client>.md) in the bounded
        StateFile shape — they share the state byte budget so GET /v1/memory/state is
        uniform. Missing file → empty content."""
        path = self._dir / name
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        return StateFile(
            name=name,
            content=content,
            bytes=len(content.encode("utf-8")),
            max_bytes=self.max_bytes,
        )

    def identity(self) -> str:
        if not self._identity_file.exists():
            return ""
        return self._identity_file.read_text(encoding="utf-8")

    def identity_file(self) -> StateFile:
        return self._top_level_file(IDENTITY_NAME)

    def overlay_files(self) -> list[StateFile]:
        """The client-identity overlays that exist on disk, in stable order — served by
        GET /v1/memory/state and editable via PUT /v1/memory/state/{name}."""
        return [self._top_level_file(n) for n in CLIENT_OVERLAY_NAMES if (self._dir / n).exists()]

    def client_overlay(self, client: str) -> str:
        """The overlay content for a client (dai/sentei), or "" for other/unknown."""
        name = _CLIENT_OVERLAY_FOR.get(client)
        if name is None:
            return ""
        path = self._dir / name
        return path.read_text(encoding="utf-8") if path.exists() else ""

    @staticmethod
    def is_identity_file(name: str) -> bool:
        return name in IDENTITY_FILE_NAMES

    def write_identity_file(self, name: str, content: str) -> StateFile:
        """Overwrite an identity-family file verbatim (identity.md or a client overlay).
        Callers enforce the byte cap — a human edit is rejected oversized, never silently
        compacted (compaction is for model-driven state appends)."""
        self.ensure()
        (self._dir / name).write_text(content, encoding="utf-8")
        return self._top_level_file(name)

    def write_identity(self, content: str) -> StateFile:
        """Overwrite identity.md verbatim (thin wrapper over write_identity_file)."""
        return self.write_identity_file(IDENTITY_NAME, content)

    # -- state -------------------------------------------------------------
    def state_path(self, name: str) -> Path:
        return self._state_dir / f"{slugify(name)}.md"

    def write_state(self, name: str, content: str) -> StateFile:
        """Write a state file, compacting oldest content to stay under the byte cap."""
        self.ensure()
        compacted = compact_oldest(content, self.max_bytes)
        path = self.state_path(name)
        path.write_text(compacted, encoding="utf-8")
        return StateFile(
            name=path.name,
            content=compacted,
            bytes=len(compacted.encode("utf-8")),
            max_bytes=self.max_bytes,
        )

    def append_state(self, name: str, addition: str) -> StateFile:
        """Append to a state file (newest at the bottom), compacting from the top."""
        path = self.state_path(name)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        merged = f"{existing.rstrip()}\n\n{addition.strip()}\n" if existing.strip() else addition
        return self.write_state(name, merged)

    def delete_state(self, name: str) -> bool:
        path = self.state_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def files(self) -> list[StateFile]:
        """All bounded state files, verbatim (state/*.md only — identity.md is a
        separate layer; all_files() prepends it for the HTTP payload)."""
        self.ensure()
        out = []
        for path in sorted(self._state_dir.glob("*.md")):
            content = path.read_text(encoding="utf-8")
            out.append(
                StateFile(
                    name=path.name,
                    content=content,
                    bytes=len(content.encode("utf-8")),
                    max_bytes=self.max_bytes,
                )
            )
        return out

    def all_files(self) -> list[StateFile]:
        """identity.md + the client overlays + every state file — the GET
        /v1/memory/state payload and the known-name set for PUT /v1/memory/state/{name}
        (bare filenames only; nothing outside this list is addressable, so traversal
        cannot name a path)."""
        return [self.identity_file(), *self.overlay_files(), *self.files()]
