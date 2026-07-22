"""Memory & skill tools.

READ (`memory_search`) is available to every registry. WRITES (`memory_write`,
`skill_save`, `skill_improve`) exist ONLY in orchestrator-role registries built for
chat/code post-task reflection — see tools/registry.py — and each write tool ALSO
re-checks the slot role via the service (defense in depth, docs/memory.md §7).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from suiban.errors import BonsaiError
from suiban.memory.service import MemoryService
from suiban.tools.base import Tool, ToolContext, ToolResult

SummarizeFn = Callable[[str], Awaitable[str]]

# Above this many combined hits the raw dump is noise: the resident utility model
# condenses it to a query-focused digest (ids kept so the model can fetch details).
DIGEST_THRESHOLD = 6


class MemorySearchTool(Tool):
    name = "memory_search"
    description = (
        "Search long-term memory (identity, state, distilled entries) and past session "
        "transcripts. FTS5 keyword search — use distinctive words from what you are "
        "trying to recall."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryService, summarize: SummarizeFn | None = None) -> None:
        self._memory = memory
        self._summarize = summarize

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query: str = args["query"]
        limit = int(args.get("limit", 12))
        entry_hits = self._memory.store.search(query, limit=limit)
        message_hits = self._memory.store.search_messages(query, limit=limit)
        if not entry_hits and not message_hits:
            return ToolResult("ok", f"no memory hits for: {query}", summary="0 hits")

        lines = []
        for hit in entry_hits:
            entry = hit["entry"]
            lines.append(f"[{entry['id']}] ({entry['layer']}) {entry['title']}: {hit['snippet']}")
        for hit in message_hits:
            lines.append(f"[session {hit['session_id']}] {hit['role']}: {hit['snippet']}")
        body = "\n".join(lines)
        total = len(entry_hits) + len(message_hits)

        if total > DIGEST_THRESHOLD and self._summarize is not None:
            digest = await self._summarize(
                f"Condense these memory-search hits into a short digest answering the "
                f"query {query!r}. Keep the [ids] of anything you mention; drop "
                f"irrelevant hits.\n\n{body}"
            )
            body = digest.strip() or body
        return ToolResult("ok", body, summary=f"{total} hits for {query!r}")


class SessionSearchTool(Tool):
    """Recall over the session archive alone (api.md §11 notes, additive
    2026-07-21c). memory_search already mixes transcript hits in; this tool exists
    for deliberate archive digs — "what did we discuss about X?" — returning
    snippets with their session ids so the model can follow up on one id."""

    name = "session_search"
    description = (
        "Search past session transcripts (the conversation archive). FTS5 keyword "
        "search — use distinctive words. Returns matching snippets with their "
        "session ids; follow up on an id when you need that conversation's detail."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query: str = args["query"]
        limit = int(args.get("limit", 12))
        hits = self._memory.store.search_messages(query, limit=limit)
        if not hits:
            return ToolResult("ok", f"no session-archive hits for: {query}", summary="0 hits")
        lines = [f"[session {hit['session_id']}] {hit['role']}: {hit['snippet']}" for hit in hits]
        return ToolResult("ok", "\n".join(lines), summary=f"{len(hits)} session hits for {query!r}")


class MemoryWriteTool(Tool):
    name = "memory_write"
    description = (
        "Persist something durable to memory. layer 'state' = current facts that "
        "change (bounded file, oldest content compacts away); layer 'archive' = "
        "distilled long-term entries. Write sparingly — memory is for what future "
        "sessions genuinely need."
    )
    parameters = {
        "type": "object",
        "properties": {
            "layer": {"type": "string", "enum": ["state", "archive"]},
            "title": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["layer", "title", "content"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            entry = self._memory.model_write_memory(
                ctx.role,
                args["layer"],
                args["title"],
                args["content"],
                args.get("tags"),
                session_id=ctx.session_id,
            )
        except BonsaiError as exc:
            return ToolResult("error", exc.message)
        return ToolResult(
            "ok",
            f"saved {entry.layer} memory {entry.id}: {entry.title}",
            summary=f"memory saved: {entry.title}",
        )


class SkillSaveTool(Tool):
    name = "skill_save"
    description = (
        "Save a NEW reusable skill (agentskills.io markdown with name/description "
        "frontmatter). Only for procedures that worked and will recur; not for "
        "one-off task notes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "kebab-case skill name."},
            "content": {"type": "string", "minLength": 1},
        },
        "required": ["name", "content"],
        "additionalProperties": False,
    }

    def __init__(self, memory: MemoryService) -> None:
        self._memory = memory

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._memory.skills.get(args["name"]) is not None:
            return ToolResult(
                "error",
                f"skill {args['name']!r} already exists — use skill_improve to refine it",
            )
        return self._save(args, ctx)

    def _save(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            skill = self._memory.model_save_skill(ctx.role, args["name"], args["content"])
        except BonsaiError as exc:
            return ToolResult("error", exc.message)
        except ValueError as exc:
            return ToolResult("error", str(exc))
        return ToolResult(
            "ok",
            f"skill {skill.name} saved (version {skill.version})",
            summary=f"skill saved: {skill.name} v{skill.version}",
        )


class SkillImproveTool(SkillSaveTool):
    name = "skill_improve"
    description = (
        "Improve an EXISTING skill with what this task taught you. Provide the full "
        "revised SKILL.md content; the version increments."
    )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._memory.skills.get(args["name"]) is None:
            return ToolResult(
                "error", f"skill {args['name']!r} does not exist — use skill_save to create it"
            )
        return self._save(args, ctx)
