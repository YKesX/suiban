"""SLAP protocol constants + vendored-schema loading.

The 10 JSON schemas in `schemas/` are a byte-identical vendored copy of the canonical
`slap` repo (see `schemas/README.md`). suiban implements SLAP independently — no import
from the slap package — the same way it implements `docs/api.md` for its HTTP clients.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

SCHEMAS_DIR = Path(__file__).parent / "schemas"

# Protocol version this vendored copy targets (envelope `version`, SemVer major.minor).
VERSION = "1.0"

# The nine operations (SLAP.md §"The nine operations"). `envelope` is the shared base,
# not an operation.
OPERATIONS: tuple[str, ...] = (
    "assign",
    "result",
    "review",
    "decide",
    "error",
    "cancel",
    "heartbeat",
    "capability",
    "status",
)

# Conformance profiles suiban advertises over /v1/slap (SLAP.md §Profiles). suiban's
# Ultra loadout runs an orchestrator, workers, and a utility model, so it advertises
# those three; the full profile list also includes "minimal" and "extended".
PROFILES: tuple[str, ...] = ("orchestrator", "worker", "utility")


@cache
def load_schema(operation: str) -> dict:
    """The vendored JSON Schema for one operation (or the shared `envelope`).

    Raises KeyError for an unknown operation so callers can turn it into a clean 404 /
    validation error rather than a stray FileNotFoundError.
    """
    if operation != "envelope" and operation not in OPERATIONS:
        raise KeyError(operation)
    path = SCHEMAS_DIR / f"{operation}.json"
    if not path.is_file():
        raise KeyError(operation)
    return json.loads(path.read_text(encoding="utf-8"))


def load_schemas() -> dict[str, dict]:
    """Every vendored operation schema keyed by operation name (envelope excluded)."""
    return {op: load_schema(op) for op in OPERATIONS}
