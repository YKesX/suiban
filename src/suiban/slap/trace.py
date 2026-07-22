"""In-process SLAP trace store (memory/ is Phase C — no new memory table).

One Ultra run records its validated SLAP message sequence here keyed by `session_id`;
`GET /v1/slap/trace/{session_id}` reads it back. Process-local and bounded: it survives
only for the server's lifetime and evicts the oldest sessions past `max_sessions`. The
volatile per-agent system prompts are stripped before a message is recorded (see
`modes/ultra.py`), so nothing here retains them.
"""

from __future__ import annotations

import copy
from collections import OrderedDict


class SlapTraceStore:
    """Bounded, per-session store of validated SLAP messages."""

    def __init__(self, max_sessions: int = 128) -> None:
        self._by_session: OrderedDict[str, list[dict]] = OrderedDict()
        self._max_sessions = max_sessions

    def record(self, session_id: str, messages: list[dict]) -> None:
        """Replace a session's trace with `messages` (a deep copy is stored)."""
        self._by_session.pop(session_id, None)
        self._by_session[session_id] = copy.deepcopy(messages)
        while len(self._by_session) > self._max_sessions:
            self._by_session.popitem(last=False)

    def get(self, session_id: str) -> list[dict]:
        """A session's recorded messages, or an empty list if none (api.md §12)."""
        return copy.deepcopy(self._by_session.get(session_id, []))

    def reset(self) -> None:
        self._by_session.clear()


_STORE = SlapTraceStore()


def trace_store() -> SlapTraceStore:
    """The process-global SLAP trace store."""
    return _STORE
