"""Tool registry: what a given (mode, slot role, capabilities) combination may call.

THE enforcement point for the one-writer rule: memory_write / skill_save /
skill_improve are added ONLY when the driving slot's role is `orchestrator` AND the
mode is chat or code (the post-task reflection path). Worker/utility registries never
contain them, so the grammar-constrained decoder cannot even emit such a call
(docs/memory.md §7). browse_t2 is added only when the loadout capability allows it.
"""

from __future__ import annotations

import asyncio
from typing import Any

from suiban.memory.service import MemoryService
from suiban.tools import schema as schema_mod
from suiban.tools.base import Tool, ToolContext, ToolResult
from suiban.tools.browse import BrowseT1Tool, BrowseT2Tool
from suiban.tools.fs import FsListTool, FsReadTool, FsUndoTool, FsWriteTool
from suiban.tools.git_ro import GitRoTool
from suiban.tools.memory_tools import (
    MemorySearchTool,
    MemoryWriteTool,
    SessionSearchTool,
    SkillImproveTool,
    SkillSaveTool,
    SummarizeFn,
)
from suiban.tools.plan import PlanTool
from suiban.tools.shell import ShellTool

WRITE_TOOL_MODES = ("chat", "code")


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def openai_tools(self) -> list[dict]:
        """The `tools` array passed to llama-server on every request — this is what the
        grammar constrains decoding against."""
        return [t.openai_schema() for t in self._tools.values()]

    def validate_args(self, name: str, args: Any) -> list[str]:
        tool = self._tools.get(name)
        if tool is None:
            return [f"unknown tool: {name!r} (available: {', '.join(self._tools) or 'none'})"]
        if not isinstance(args, dict):
            return [f"arguments must be a JSON object, got {type(args).__name__}"]
        return schema_mod.validate(args, tool.parameters)

    async def run(self, name: str, args: dict, ctx: ToolContext) -> ToolResult:
        """Execute a tool with its timeout ceiling; failures become error results,
        never exceptions — a broken tool must not kill the agent loop."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("error", f"unknown tool: {name!r}")
        try:
            return await asyncio.wait_for(tool.run(args, ctx), timeout=tool.timeout_s)
        except TimeoutError:
            return ToolResult("error", f"tool {name} timed out after {tool.timeout_s:.0f}s")
        except Exception as exc:  # noqa: BLE001 - the loop must survive any tool bug
            return ToolResult("error", f"tool {name} crashed: {exc!r}")


# Static mode -> base toolset map (write tools appended per role, see build_registry).
# Also the source for /v1/modes `tools` listings (modes/registry.py).
# ultra lists the toolset its contained sub-agents run with (modes/ultra.py); the
# dispatcher itself does not run a tool loop — it plans (grammar-constrained),
# dispatches, and synthesizes.
MODE_TOOLSETS: dict[str, tuple[str, ...]] = {
    # Recall lives in BOTH chat and code (api.md §11 notes, additive 2026-07-21c):
    # memory_search plus the dedicated session-archive search.
    "chat": ("memory_search", "session_search", "browse_t1", "browse_t2"),
    "code": (
        "plan",
        "fs_read",
        "fs_write",
        "fs_list",
        "fs_undo",  # revert journaled fs_write edits (docs/architecture.md §3.7)
        "shell",
        "git_ro",
        "memory_search",
        "session_search",
    ),
    "ultra": ("plan", "fs_read", "fs_write", "fs_list", "shell", "git_ro", "memory_search"),
    "deep_research": ("browse_t1", "browse_t2", "memory_search"),
}

# What an Ultra sub-agent may call: files in the session jail, confirm-gated shell,
# read-only git, browse t1, memory READS. Never write tools, never browse_t2, never
# vision — containment is structural (docs/memory.md §7).
WORKER_TOOLSET: tuple[str, ...] = (
    "fs_read",
    "fs_write",
    "fs_list",
    "shell",
    "git_ro",
    "browse_t1",
    "memory_search",
)

WRITE_TOOL_NAMES: tuple[str, ...] = ("memory_write", "skill_save", "skill_improve")


def mode_tool_names(mode: str, *, with_writes: bool = True) -> list[str]:
    names = list(MODE_TOOLSETS.get(mode, ()))
    if with_writes and mode in WRITE_TOOL_MODES:
        names.extend(WRITE_TOOL_NAMES)
    return names


def build_registry(
    mode: str,
    role: str,
    *,
    memory: MemoryService,
    browse_t2_available: bool = False,
    summarize: SummarizeFn | None = None,
    extra_tools: list[Tool] | None = None,
) -> ToolRegistry:
    """Build the registry for one agent run.

    `role` is the slot role driving the loop. Only `orchestrator` + chat/code gets the
    memory/skill write tools; browse_t2 needs the loadout capability (resident 27B +
    setting enabled) — callers pass /v1/system.capabilities.browse_t2.

    `extra_tools` is the MCP seam (api.md 2026-07-21d): connected servers' namespaced
    `mcp_<server>_<tool>` tools, appended per run so a crashed server's tools vanish
    from the NEXT run without touching this registry. The caller (routers/chat.py)
    passes them for chat/code runs only; `/v1/modes` keeps listing built-ins — the
    MCP set is dynamic per-config, not part of the mode definition.
    """
    available: dict[str, Tool] = {
        "memory_search": MemorySearchTool(memory, summarize),
        "session_search": SessionSearchTool(memory),
        "browse_t1": BrowseT1Tool(),
        "plan": PlanTool(),
        "fs_read": FsReadTool(),
        "fs_write": FsWriteTool(),
        "fs_list": FsListTool(),
        "fs_undo": FsUndoTool(),
        "shell": ShellTool(),
        "git_ro": GitRoTool(),
    }
    if browse_t2_available:
        available["browse_t2"] = BrowseT2Tool()

    tools = [available[n] for n in MODE_TOOLSETS.get(mode, ()) if n in available]

    if role == "orchestrator" and mode in WRITE_TOOL_MODES:
        tools.append(MemoryWriteTool(memory))
        tools.append(SkillSaveTool(memory))
        tools.append(SkillImproveTool(memory))

    if extra_tools:
        # mcp_-prefixed names cannot collide with built-ins; last-in wins in the
        # registry dict, so built-ins are appended first deliberately.
        tools.extend(extra_tools)

    return ToolRegistry(tools)


def build_worker_registry(
    *, memory: MemoryService, summarize: SummarizeFn | None = None
) -> ToolRegistry:
    """Registry for one Ultra sub-agent (WORKER_TOOLSET). Write tools can never appear
    here — the grammar-constrained decoder on a worker slot cannot even emit a
    memory_write/skill_save/skill_improve call, and the memory service re-checks the
    role as defense in depth."""
    available: dict[str, Tool] = {
        "fs_read": FsReadTool(),
        "fs_write": FsWriteTool(),
        "fs_list": FsListTool(),
        "shell": ShellTool(),
        "git_ro": GitRoTool(),
        "browse_t1": BrowseT1Tool(),
        "memory_search": MemorySearchTool(memory, summarize),
    }
    return ToolRegistry([available[name] for name in WORKER_TOOLSET])
