"""Built-in MCP connector catalog (api.md 2026-07-22c).

A curated, one-click list of well-known MCP servers on TOP of the free-form custom
`mcp_servers`. Settings carry only `{ id, enabled }` per connector (`mcp_connectors`);
the command/args/description live here so enabling one is a single toggle. An enabled
connector is resolved into an `McpServerSettings` and wired into the same `McpManager`
as a custom stdio server — identical transport (the wave-2 stdio client), identical
namespacing (`mcp_<id>_<tool>`), identical failure handling (a bad server is a notice,
never a crash).

The entries below are real, well-known MCP servers: the reference `@modelcontextprotocol`
set plus the common community `uvx` servers. These are the same servers the openclaw and
hermes agent ecosystems reference in their `optional-mcps`; enabling them here is the
suiban-native path (docs/memory.md). Because connectors carry only `{ id, enabled }`, a
connector whose launch needs a path (filesystem) defaults that path to the user's home
directory — TODO(v1.1): per-connector path config in the `mcp_connectors` shape so the
filesystem root is user-chosen rather than defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from suiban.config import McpConnectorSettings, McpServerSettings


@dataclass(frozen=True)
class Connector:
    """One catalog entry. `args` is the launch template; when `requires_path` is set the
    resolved server appends a directory argument (defaulted to the user's home) so the
    server has a root to operate on."""

    id: str
    name: str
    description: str
    command: str
    args: list[str] = field(default_factory=list)
    requires_path: bool = False

    def to_server_settings(self, *, default_path: Path) -> McpServerSettings:
        args = list(self.args)
        if self.requires_path:
            args.append(str(default_path))
        # id is validated kebab-case (McpConnectorSettings) == McpServerSettings.name
        # pattern, so it namespaces the server's tools as mcp_<id>_<tool> directly.
        return McpServerSettings(name=self.id, command=self.command, args=args, enabled=True)


# The curated catalog. All entries are real MCP servers; none is spawned unless the user
# enables it. npx/uvx are resolved from PATH at spawn time (a missing runtime becomes an
# mcp_server_failed notice, never a crash — McpManager handles it).
CATALOG: list[Connector] = [
    Connector(
        id="filesystem",
        name="Filesystem",
        description="Read, write, and search files under an allowed directory.",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem"],
        requires_path=True,
    ),
    Connector(
        id="git",
        name="Git",
        description="Inspect and operate on Git repositories (status, diff, log, commit).",
        command="uvx",
        args=["mcp-server-git"],
    ),
    Connector(
        id="fetch",
        name="Fetch",
        description="Fetch a URL and return its content as markdown for the model to read.",
        command="uvx",
        args=["mcp-server-fetch"],
    ),
    Connector(
        id="memory",
        name="Knowledge-graph memory",
        description="A persistent knowledge-graph memory server (entities and relations).",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-memory"],
    ),
    Connector(
        id="everything",
        name="Everything (reference)",
        description="The MCP reference server exercising every protocol feature — for testing.",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-everything"],
    ),
    Connector(
        id="sequential-thinking",
        name="Sequential thinking",
        description="A structured step-by-step reasoning scratchpad tool.",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
    ),
    Connector(
        id="time",
        name="Time",
        description="Current time and timezone conversion utilities.",
        command="uvx",
        args=["mcp-server-time"],
    ),
]


def _default_path() -> Path:
    """Directory handed to a `requires_path` connector when none is configured. The
    user's home is computed at runtime — never a machine-specific path in the repo."""
    return Path.home()


def resolve_connectors(
    selections: list[McpConnectorSettings],
    *,
    catalog: list[Connector] | None = None,
    default_path: Path | None = None,
) -> list[McpServerSettings]:
    """Turn the ENABLED `mcp_connectors` selections into `McpServerSettings`. Unknown
    ids (a catalog that shrank between versions) are ignored; disabled selections are
    skipped. `catalog` is injectable for tests (a fake fixture entry); it defaults to
    the module `CATALOG` read live so a monkeypatch takes effect."""
    entries = CATALOG if catalog is None else catalog
    by_id = {c.id: c for c in entries}
    path = default_path if default_path is not None else _default_path()
    out: list[McpServerSettings] = []
    for selection in selections:
        if not selection.enabled:
            continue
        connector = by_id.get(selection.id)
        if connector is None:
            continue
        out.append(connector.to_server_settings(default_path=path))
    return out


def catalog_view(
    selections: list[McpConnectorSettings], *, catalog: list[Connector] | None = None
) -> list[dict]:
    """The GET /v1/mcp/connectors payload: every catalog entry with its `enabled` flag
    reflecting the current `mcp_connectors` settings (api.md 2026-07-22c)."""
    entries = CATALOG if catalog is None else catalog
    enabled_ids = {s.id for s in selections if s.enabled}
    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "command": c.command,
            "args": list(c.args),
            "requires_path": c.requires_path,
            "enabled": c.id in enabled_ids,
        }
        for c in entries
    ]


def combined_mcp_servers(
    settings, *, catalog: list[Connector] | None = None
) -> list[McpServerSettings]:
    """The full server list for the McpManager: custom `mcp_servers` first (they win any
    id/name collision so a user's own entry is never shadowed), then enabled catalog
    connectors. Used at boot and on every settings apply."""
    custom = list(settings.mcp_servers)
    used = {s.name for s in custom}
    connectors = resolve_connectors(settings.mcp_connectors, catalog=catalog)
    return custom + [c for c in connectors if c.name not in used]
