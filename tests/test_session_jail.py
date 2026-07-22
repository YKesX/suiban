"""session_id path-traversal containment (audit 2026-07-22, H2).

A client-supplied session_id is joined under ~/.bonsai/work to form the fs/shell jail
root; `safe_workdir_name` must keep the on-disk directory inside the jail regardless of
the id, while leaving legitimate ids untouched (so existing sessions keep their dir).
"""

from __future__ import annotations

from pathlib import Path

from suiban.routers.chat import safe_workdir_name


def test_legit_session_ids_pass_through_unchanged() -> None:
    for sid in ("tg-42", "anon-deadbeef00", "sess_123", "A-Z_0-9", "uuid-1234-5678"):
        assert safe_workdir_name(sid) == sid


def test_traversal_and_hostile_ids_cannot_escape_the_jail() -> None:
    root = Path("/home/user/.bonsai/work").resolve()
    for evil in ("../../etc", "../../../etc/passwd", "a/b/c", "..", "/etc/shadow", "", "盆栽", "."):
        name = safe_workdir_name(evil)
        assert "/" not in name and "\\" not in name and ".." not in name
        workdir = (root / name).resolve()
        assert root in workdir.parents  # the derived workdir stays under the jail root
        assert workdir.parent == root
