"""Filesystem locations. Everything mutable lives under the bonsai home (~/.bonsai).

The repo itself never contains machine state; tests point SUIBAN_HOME at a tmp dir.
"""

from __future__ import annotations

import os
from pathlib import Path


def bonsai_home() -> Path:
    """Root for all local state: config, staged settings, binaries, models, budget."""
    override = os.environ.get("SUIBAN_HOME")
    if override:
        return Path(override)
    return Path.home() / ".bonsai"


def config_path() -> Path:
    return bonsai_home() / "config.toml"


def staged_path() -> Path:
    return bonsai_home() / "staged.toml"


def budget_path() -> Path:
    return bonsai_home() / "budget.json"


def bin_dir(backend: str) -> Path:
    return bonsai_home() / "bin" / backend


def models_dir(family: str) -> Path:
    return bonsai_home() / "models" / family


def memory_dir() -> Path:
    """Memory root: identity.md, state/, memory.sqlite (see docs/memory.md)."""
    return bonsai_home() / "memory"


def reports_dir() -> Path:
    """Deep-research reports (<job_id>.md) and bench reports (bench-kv-<date>.md)."""
    return bonsai_home() / "reports"


def skills_dir() -> Path:
    """agentskills.io-compatible skill directories: <name>/SKILL.md."""
    return bonsai_home() / "skills"


def work_dir(session_id: str) -> Path:
    """Per-session tool workdir — the fs/shell jail root."""
    return bonsai_home() / "work" / session_id


def browser_profile_dir() -> Path:
    """Sandboxed Playwright profile for tier-2 browsing (never the user's browser)."""
    return bonsai_home() / "browser" / "profile"


def browser_downloads_dir() -> Path:
    """Pinned download target for tier-2 browsing: anything a page manages to download
    lands here (quarantined, inspectable) — never in ~/Downloads or the session
    workdir."""
    return bonsai_home() / "browser" / "downloads"
