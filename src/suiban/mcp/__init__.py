"""MCP (Model Context Protocol) stdio client — api.md §8, additive 2026-07-21d;
connector catalog added 2026-07-22c.

v1 scope, deliberately narrow: stdio transport only, tools only (no resources,
prompts, sampling, or roots). Connected servers' tools appear to the model namespaced
`mcp_<server>_<tool>` with their JSON schemas passed through verbatim. Curated one-click
connectors (catalog.py) resolve into the same manager as custom stdio servers.
"""

from suiban.mcp.catalog import (
    CATALOG,
    Connector,
    catalog_view,
    combined_mcp_servers,
    resolve_connectors,
)
from suiban.mcp.client import McpClient, McpError
from suiban.mcp.manager import McpManager, McpTool

__all__ = [
    "CATALOG",
    "Connector",
    "McpClient",
    "McpError",
    "McpManager",
    "McpTool",
    "catalog_view",
    "combined_mcp_servers",
    "resolve_connectors",
]
