"""Sortable entry ids: "mem_" + a ULID-style string.

26 chars of Crockford base32 — 48-bit millisecond timestamp + 80 random bits — so ids
sort by creation time lexicographically. Implemented locally (~20 lines) instead of
adding a ulid dependency.
"""

from __future__ import annotations

import secrets
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def _base32(value: int, length: int) -> str:
    chars = []
    for _ in range(length):
        chars.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def ulid() -> str:
    timestamp_ms = int(time.time() * 1000)
    randomness = secrets.randbits(80)
    return _base32(timestamp_ms, 10) + _base32(randomness, 16)


def memory_id() -> str:
    return f"mem_{ulid()}"


def new_project_id() -> str:
    return f"proj_{ulid()}"


def new_doc_id() -> str:
    return f"doc_{ulid()}"
