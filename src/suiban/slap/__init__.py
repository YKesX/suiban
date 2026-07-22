"""Vendored, self-contained implementation of SLAP (Structured Lightweight Agent
Protocol) 1.0.

The canonical spec and JSON schemas live in the separate `slap` repo. suiban does NOT
import that package: it vendors the 10 schemas (`schemas/`, byte-identical — see
`schemas/README.md`) and implements load / validate / build here, exactly as it
implements `docs/api.md` independently of its HTTP clients. Ultra mode (`modes/ultra.py`)
is the first consumer; `/v1/slap` exposes the protocol for inspection.
"""

from __future__ import annotations

from suiban.slap.builders import (
    build_assign,
    build_cancel,
    build_capability,
    build_decide,
    build_error,
    build_heartbeat,
    build_result,
    build_review,
    build_status,
)
from suiban.slap.protocol import (
    OPERATIONS,
    PROFILES,
    SCHEMAS_DIR,
    VERSION,
    load_schema,
    load_schemas,
)
from suiban.slap.trace import SlapTraceStore, trace_store
from suiban.slap.validation import is_valid, validate_instance, validate_message

__all__ = [
    "OPERATIONS",
    "PROFILES",
    "SCHEMAS_DIR",
    "VERSION",
    "SlapTraceStore",
    "build_assign",
    "build_cancel",
    "build_capability",
    "build_decide",
    "build_error",
    "build_heartbeat",
    "build_result",
    "build_review",
    "build_status",
    "is_valid",
    "load_schema",
    "load_schemas",
    "trace_store",
    "validate_instance",
    "validate_message",
]
