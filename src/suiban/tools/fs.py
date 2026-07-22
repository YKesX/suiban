"""Filesystem tools, jailed to the per-session workdir.

Every path is resolved (symlinks followed) and must land inside the jail root —
absolute paths, `..` traversal, and symlink escapes all fail with a clear error.

Code-mode edit safety (docs/architecture.md §3.7): `fs_write` computes a unified diff
of before-state vs the new content BEFORE applying it — the diff rides the tool_result
content so clients (dai's tool feed) show exactly what changed — and journals the
prior state under `<workdir>/.suiban-undo/` (last MAX_UNDO_ENTRIES edits). `fs_undo`
reverts the most recent journaled edit, or a specific one by index. The journal dir is
excluded from `fs_list` output and refused as an `fs_write` target.

Confirm gate (api.md 2026-07-22b): file MUTATIONS (`fs_write`, `fs_undo`) are refused
on the first call with a `denied` status + a single-use, operation-bound `confirm_token`
and the unified diff — exactly like a destructive shell command. The model re-runs the
identical operation with the token to apply; nothing touches disk before confirmation.
`fs_read`/`fs_list` are never gated. A session with `auto_confirm` (ToolContext bypass
flag; code/ultra only) skips the gate — every such mutation is logged (logger.warning).

Security (audit 2026-07-22): resolve-then-act TOCTOU — a path that passed
`resolve_in_jail` can be swapped for a symlink between the check and the read/write that
follows. The final open now uses O_NOFOLLOW (`_read_bytes_nofollow` /
`_write_text_nofollow`), so a swapped-in symlink at the target fails instead of being
followed out of the jail. KNOWN_ISSUE: this is not full openat2(RESOLVE_BENEATH) —
intermediate-component swaps and platforms without O_NOFOLLOW (Windows) still rely on
the resolve check; a landlock/openat2 jail is the v1.1 hardening.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Any

from suiban.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

READ_MAX_BYTES = 256 * 1024
LIST_MAX_ENTRIES = 500

UNDO_DIR_NAME = ".suiban-undo"
MAX_UNDO_ENTRIES = 20
DIFF_MAX_CHARS = 8_000

# O_NOFOLLOW on the final open closes the resolve-then-act TOCTOU (audit 2026-07-22): a
# path that passed resolve_in_jail can be swapped for a symlink before the read/write —
# opening the (already-resolved) target with O_NOFOLLOW makes such a swap fail (ELOOP)
# instead of following the link out of the jail. O_NOFOLLOW is 0 (a no-op) on platforms
# lacking it (Windows); there the resolve check remains the only guard. This is not full
# openat2(RESOLVE_BENEATH) — intermediate-component swaps are still a KNOWN_ISSUE for v1.1.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def resolve_in_jail(root: Path, user_path: str) -> Path:
    """Resolve `user_path` against the jail root; raise ValueError on escape."""
    root = root.resolve()
    candidate = Path(user_path)
    joined = candidate if candidate.is_absolute() else root / candidate
    resolved = joined.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes the session workdir: {user_path}")
    return resolved


def _read_bytes_nofollow(path: Path) -> bytes:
    """Read a file, refusing to follow a symlink at the final component (O_NOFOLLOW)."""
    fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    try:
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _write_text_nofollow(path: Path, text: str) -> None:
    """Write a file, refusing to follow a symlink at the final component (O_NOFOLLOW)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | _O_NOFOLLOW, 0o644)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(text.encode("utf-8"))
    finally:
        os.close(fd)


def _in_undo_dir(root: Path, resolved: Path) -> bool:
    try:
        return UNDO_DIR_NAME in resolved.relative_to(root).parts
    except ValueError:  # not under root — resolve_in_jail already rejects this
        return False


def unified_diff_text(before: str, after: str, rel_path: str) -> str:
    """Unified diff (a/<path> vs b/<path>), the same shape `diff -u` / git emit."""
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
        lineterm="",
    )
    return "\n".join(diff_lines)


def _diff_counts(diff: str) -> tuple[int, int]:
    lines = diff.splitlines()
    added = sum(1 for ln in lines if ln.startswith("+") and not ln.startswith("+++"))
    removed = sum(1 for ln in lines if ln.startswith("-") and not ln.startswith("---"))
    return added, removed


def _clip_diff(diff: str) -> str:
    if len(diff) > DIFF_MAX_CHARS:
        return diff[:DIFF_MAX_CHARS] + f"\n… [diff truncated at {DIFF_MAX_CHARS} chars]"
    return diff


# -- confirm-gate signatures (api.md 2026-07-22b) ------------------------------
# A confirmation token is bound to the EXACT operation it approves (single use),
# exactly like the destructive-shell gate binds its token to the exact command:
# a token issued for one (path, content) cannot approve a different write, and a
# token for undoing edit #3 cannot approve undoing edit #4.
def _write_signature(rel: str, content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"fs_write:{rel}:{digest}"


def _undo_signature(seq: object) -> str:
    return f"fs_undo:{seq}"


# -- undo journal --------------------------------------------------------------
def _undo_entry_files(root: Path) -> list[Path]:
    undo = root / UNDO_DIR_NAME
    if not undo.is_dir():
        return []
    return sorted((p for p in undo.glob("*.json") if p.stem.isdigit()), key=lambda p: int(p.stem))


def record_undo_entry(root: Path, rel_path: str, prior_content: str | None) -> int:
    """Journal the PRE-edit state (called before the edit is applied, so even a crash
    mid-write leaves the prior content recoverable). Keeps the last MAX_UNDO_ENTRIES
    entries; returns the new entry's sequence number."""
    undo = root / UNDO_DIR_NAME
    undo.mkdir(parents=True, exist_ok=True)
    existing = _undo_entry_files(root)
    seq = (int(existing[-1].stem) + 1) if existing else 1
    entry = {
        "seq": seq,
        "path": rel_path,
        "existed": prior_content is not None,
        "prior_content": prior_content,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (undo / f"{seq:04d}.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
    for stale in existing[: max(0, len(existing) + 1 - MAX_UNDO_ENTRIES)]:
        stale.unlink(missing_ok=True)
    return seq


def load_undo_entries(root: Path) -> list[tuple[Path, dict]]:
    """(file, parsed entry) pairs, oldest first; unparseable entries are skipped."""
    out: list[tuple[Path, dict]] = []
    for file in _undo_entry_files(root):
        try:
            entry = json.loads(file.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(entry, dict) and "path" in entry:
            out.append((file, entry))
    return out


# -- tools ---------------------------------------------------------------------
class FsReadTool(Tool):
    name = "fs_read"
    description = "Read a text file inside the session workdir. Returns the file content."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Path relative to the workdir."}},
        "required": ["path"],
        "additionalProperties": False,
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = resolve_in_jail(ctx.workdir, args["path"])
        except ValueError as exc:
            return ToolResult("error", str(exc))
        if not target.is_file():
            return ToolResult("error", f"not a file: {args['path']}")
        try:
            data = _read_bytes_nofollow(target)
        except OSError as exc:
            return ToolResult("error", f"could not read {args['path']}: {exc}")
        truncated = len(data) > READ_MAX_BYTES
        text = data[:READ_MAX_BYTES].decode("utf-8", errors="replace")
        if truncated:
            text += f"\n… [truncated at {READ_MAX_BYTES} bytes of {len(data)}]"
        return ToolResult("ok", text, summary=f"read {args['path']} ({len(data)} bytes)")


class FsWriteTool(Tool):
    name = "fs_write"
    description = (
        "Write (create or overwrite) a text file inside the session workdir. "
        "Parent directories are created as needed. The result includes a unified "
        "diff of the change; every edit is journaled and revertible via fs_undo. "
        "File writes are confirm-gated: the first call is refused with status "
        "'denied' and a confirm_token; after the user approves, re-run the exact "
        "same path and content with that confirm_token to apply."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the workdir."},
            "content": {"type": "string", "description": "Full new file content."},
            "confirm_token": {
                "type": "string",
                "description": "Token from a previous 'denied' result, after user approval.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = resolve_in_jail(ctx.workdir, args["path"])
        except ValueError as exc:
            return ToolResult("error", str(exc))
        root = ctx.workdir.resolve()
        if target == root:
            return ToolResult("error", "refusing to write the workdir root itself")
        if _in_undo_dir(root, target):
            return ToolResult(
                "error", f"refusing to write inside the edit journal ({UNDO_DIR_NAME}/)"
            )
        content: str = args["content"]

        # Diff-before-apply: capture the before-state and compute the unified diff
        # FIRST, so the change is fully described (and journaled) before a byte moves.
        prior: str | None = None
        if target.is_file():
            try:
                prior = _read_bytes_nofollow(target).decode("utf-8", errors="replace")
            except OSError as exc:
                return ToolResult("error", f"could not read {args['path']}: {exc}")
        rel = target.relative_to(root).as_posix()
        diff = unified_diff_text(prior if prior is not None else "", content, rel)
        added, removed = _diff_counts(diff)
        verb = "created" if prior is None else "rewrote"

        # Confirm gate (api.md 2026-07-22b): a file mutation is refused on the first
        # call — like a destructive shell command — with the unified diff so the client
        # can render it for Approve/Decline. Nothing has been journaled or written yet,
        # so a denial leaves the workdir untouched. auto_confirm bypasses it (logged).
        if ctx.auto_confirm:
            logger.warning(
                "auto_confirm: fs_write applied WITHOUT confirmation (session %s): %s",
                ctx.session_id,
                rel,
            )
        else:
            signature = _write_signature(rel, content)
            token = args.get("confirm_token")
            confirmed = bool(token) and ctx.confirm_tokens.pop(token, None) == signature
            if not confirmed:
                new_token = secrets.token_urlsafe(12)
                ctx.confirm_tokens[new_token] = signature
                instruction = (
                    f"This file write ({verb} {rel}, +{added} -{removed}) needs user "
                    "confirmation before it is applied. Ask the user; if they approve, "
                    "re-run fs_write with the same path and content plus "
                    f'confirm_token "{new_token}".'
                )
                body = f"{instruction}\n\n{_clip_diff(diff)}" if diff else instruction
                summary = f"confirmation required: {verb} {rel} (+{added} -{removed})\n{diff}"
                return ToolResult("denied", body, summary=summary, confirm_token=new_token)

        seq = record_undo_entry(root, rel, prior)

        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_text_nofollow(target, content)
        except OSError as exc:
            return ToolResult("error", f"could not write {args['path']}: {exc}")

        header = (
            f"{verb} {rel} ({len(content.encode('utf-8'))} bytes, +{added} -{removed}; "
            f"revert with fs_undo, edit #{seq})"
        )
        body = f"{header}\n\n{_clip_diff(diff)}" if diff else f"{header}\n\n(no content change)"
        return ToolResult("ok", body, summary=f"{verb} {rel} (+{added} -{removed})")


class FsListTool(Tool):
    name = "fs_list"
    description = "List directory entries inside the session workdir (non-recursive)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory relative to the workdir. Default: the workdir root.",
            }
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            target = resolve_in_jail(ctx.workdir, args.get("path", "."))
        except ValueError as exc:
            return ToolResult("error", str(exc))
        if not target.is_dir():
            return ToolResult("error", f"not a directory: {args.get('path', '.')}")
        # The undo journal is session plumbing, not workspace content — hidden.
        entries = sorted(
            (p for p in target.iterdir() if p.name != UNDO_DIR_NAME),
            key=lambda p: (p.is_file(), p.name),
        )
        lines = []
        for entry in entries[:LIST_MAX_ENTRIES]:
            suffix = "/" if entry.is_dir() else f"  ({entry.stat().st_size} bytes)"
            lines.append(f"{entry.name}{suffix}")
        if len(entries) > LIST_MAX_ENTRIES:
            lines.append(f"… {len(entries) - LIST_MAX_ENTRIES} more entries")
        body = "\n".join(lines) if lines else "(empty directory)"
        return ToolResult("ok", body, summary=f"listed {len(entries)} entries")


class FsUndoTool(Tool):
    name = "fs_undo"
    description = (
        "Revert a file edit made by fs_write in this session. Without arguments "
        "reverts the most recent edit; pass `index` (the edit number shown in "
        "fs_write results) to revert a specific one. The journal keeps the last "
        f"{MAX_UNDO_ENTRIES} edits; each entry can be reverted once. Like fs_write, "
        "an undo is confirm-gated: the first call is refused with status 'denied' "
        "and a confirm_token; re-run for the same edit with that token to apply."
    )
    parameters = {
        "type": "object",
        "properties": {
            "index": {
                "type": "integer",
                "minimum": 1,
                "description": "Edit number to revert. Default: the most recent edit.",
            },
            "confirm_token": {
                "type": "string",
                "description": "Token from a previous 'denied' result, after user approval.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        root = ctx.workdir.resolve()
        entries = load_undo_entries(root)
        if not entries:
            return ToolResult("error", "nothing to undo: the edit journal is empty")

        if "index" in args:
            wanted = int(args["index"])
            match = [(f, e) for f, e in entries if e.get("seq") == wanted]
            if not match:
                available = ", ".join(str(e.get("seq")) for _, e in entries)
                return ToolResult("error", f"no journaled edit #{wanted} (available: {available})")
            entry_file, entry = match[0]
        else:
            entry_file, entry = entries[-1]

        rel = str(entry["path"])
        seq = entry.get("seq", "?")
        try:
            target = resolve_in_jail(ctx.workdir, rel)
        except ValueError as exc:  # defensive: a journal entry can't point outside
            return ToolResult("error", f"journal entry #{seq} is invalid: {exc}")

        current = ""
        if target.is_file():
            try:
                current = _read_bytes_nofollow(target).decode("utf-8", errors="replace")
            except OSError:
                current = ""  # unreadable (e.g. swapped symlink): the diff shows empty

        # Describe the revert (delete vs restore) and its diff WITHOUT touching disk,
        # so the confirm gate can present it and a denial is a no-op.
        existed = entry.get("existed", False)
        prior = str(entry.get("prior_content") or "")
        if not existed:
            action = f"deleted {rel} (undid its creation)"
            diff = unified_diff_text(current, "", rel)
        else:
            action = f"restored {rel} to its pre-edit content"
            diff = unified_diff_text(current, prior, rel)

        # Confirm gate (api.md 2026-07-22b): the token is bound to THIS edit's seq, so
        # approving an undo of #3 cannot silently undo a different edit. auto_confirm
        # bypasses it (logged).
        if ctx.auto_confirm:
            logger.warning(
                "auto_confirm: fs_undo applied WITHOUT confirmation (session %s): edit #%s %s",
                ctx.session_id,
                seq,
                rel,
            )
        else:
            signature = _undo_signature(seq)
            token = args.get("confirm_token")
            confirmed = bool(token) and ctx.confirm_tokens.pop(token, None) == signature
            if not confirmed:
                new_token = secrets.token_urlsafe(12)
                ctx.confirm_tokens[new_token] = signature
                instruction = (
                    f"Undoing edit #{seq} ({action}) needs user confirmation before it "
                    "is applied. Ask the user; if they approve, re-run fs_undo for the "
                    f'same edit plus confirm_token "{new_token}".'
                )
                body = f"{instruction}\n\n{_clip_diff(diff)}" if diff else instruction
                summary = f"confirmation required: undo edit #{seq} ({rel})\n{diff}"
                return ToolResult("denied", body, summary=summary, confirm_token=new_token)

        if not existed:
            # The edit created the file — undoing it deletes the file.
            target.unlink(missing_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _write_text_nofollow(target, prior)
            except OSError as exc:
                return ToolResult("error", f"could not restore {rel}: {exc}")

        entry_file.unlink(missing_ok=True)  # consumed: an entry reverts once
        body = f"undid edit #{seq}: {action}\n\n{_clip_diff(diff)}"
        return ToolResult("ok", body, summary=f"undid edit #{seq}: {rel}")
