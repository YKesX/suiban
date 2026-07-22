"""Memory & skills: four layers, one writer, zero embeddings (docs/memory.md).

- identity: ~/.bonsai/memory/identity.md (human-owned; editable over HTTP via
  PUT /v1/memory/state/identity.md — no model ever writes it)
- state:    ~/.bonsai/memory/state/*.md (bounded files, oldest-content compaction)
- archive:  ~/.bonsai/memory/memory.sqlite (sessions + messages + entries, FTS5)
- skills:   ~/.bonsai/skills/<name>/SKILL.md (agentskills.io-compatible markdown)

MemoryService is the facade everything else uses; write enforcement (27B orchestrator
only) lives in the tool layer AND in the service (defense in depth).
"""

from suiban.memory.service import MemoryService

__all__ = ["MemoryService"]
