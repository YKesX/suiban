"""MCP-compatible tool layer: name + description + JSON-schema params + async run().

`registry.build_registry()` is the single place tool capability is decided — memory and
skill WRITE tools exist only in orchestrator-role registries (server-enforced, see
docs/memory.md §7); browse tier 2 exists only when the loadout capability allows it.
"""

from suiban.tools.base import Tool, ToolContext, ToolResult
from suiban.tools.registry import ToolRegistry, build_registry

__all__ = ["Tool", "ToolContext", "ToolResult", "ToolRegistry", "build_registry"]
