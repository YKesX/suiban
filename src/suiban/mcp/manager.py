"""MCP server manager: settings.mcp_servers -> running clients -> registry tools.

Lifecycle mirrors the gateways: enabled servers start with the app lifespan
(mcp_servers is requires_restart) and stop with it. Every failure — spawn, handshake,
tools/list, a mid-run crash — becomes an `mcp_server_failed` notice plus tool
removal; the app never crashes over a server (api.md 2026-07-21d).

Tool naming: `mcp_<server>_<tool>` (contract-literal; server names are kebab-case by
settings validation). The server's inputSchema is passed through verbatim to the
model — our subset validator (tools/schema.py) checks only the keywords it knows, so
schema features it does not implement never reject a call the server would accept.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from suiban.config import McpServerSettings
from suiban.mcp.client import DEFAULT_CALL_TIMEOUT_S, McpClient, McpError
from suiban.sched.planner import Notice
from suiban.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class McpTool(Tool):
    """One remote MCP tool, namespaced for the registry. The JSON schema is the
    server's inputSchema verbatim — passed through to the model untouched."""

    timeout_s = DEFAULT_CALL_TIMEOUT_S + 10.0  # registry ceiling; the call is tighter

    def __init__(self, manager: McpManager, server: str, definition: dict) -> None:
        self._manager = manager
        self._server = server
        self.remote_name = str(definition["name"])
        self.name = f"mcp_{server}_{self.remote_name}"
        self.description = (
            str(definition.get("description") or "").strip()
            or f"Tool {self.remote_name!r} provided by MCP server {server!r}."
        )
        schema = definition.get("inputSchema")
        self.parameters = schema if isinstance(schema, dict) and schema else dict(_EMPTY_SCHEMA)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        return await self._manager.call(self._server, self.remote_name, args)


@dataclass
class _ServerState:
    settings: McpServerSettings
    client: McpClient | None = None
    tools: list[McpTool] = field(default_factory=list)
    failed: bool = False


class McpManager:
    """Owns every configured MCP server for the process lifetime."""

    def __init__(
        self, servers: list[McpServerSettings], *, call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    ) -> None:
        self._servers: dict[str, _ServerState] = {s.name: _ServerState(s) for s in servers}
        self._call_timeout = call_timeout_s
        self._notices: list[Notice] = []
        self._resync_tasks: set[asyncio.Task] = set()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Connect every enabled server: spawn, initialize, tools/list. A failed
        server is a notice, never a raise."""
        for state in self._servers.values():
            if state.settings.enabled:
                await self._start_one(state)

    async def _start_one(self, state: _ServerState) -> None:
        """Spawn + handshake + tools/list for one server, recording its tools. Any
        McpError becomes an mcp_server_failed notice and leaves the server toolless —
        never a raise (shared by start() and resync())."""
        client = McpClient(state.settings.name, state.settings.command, state.settings.args)
        try:
            await client.start()
            definitions = await client.list_tools()
        except McpError as exc:
            await client.stop()
            self._record_failure(state, str(exc), client=client)
            return
        state.client = client
        state.failed = False
        state.tools.clear()
        seen: set[str] = set()
        for definition in definitions:
            tool = McpTool(self, state.settings.name, definition)
            if tool.name in seen:  # duplicate names within one server: first wins
                logger.warning(
                    "mcp[%s] duplicate tool %r ignored", state.settings.name, tool.remote_name
                )
                continue
            seen.add(tool.name)
            state.tools.append(tool)
        logger.info(
            "mcp server %r connected: %d tools (%s)",
            state.settings.name,
            len(state.tools),
            ", ".join(t.remote_name for t in state.tools) or "none",
        )

    async def resync(self, servers: list[McpServerSettings]) -> None:
        """Bring the running server set in line with `servers` (custom mcp_servers +
        enabled catalog connectors) WITHOUT a process restart — connectors commit at the
        next idle moment (api.md 2026-07-22c, pending_until_idle). New/changed servers
        start; a server that vanished or was disabled stops; an unchanged, still-alive
        server is left running untouched so custom mcp_servers keep working. Failures
        stay notices, never raises."""
        desired = {s.name: s for s in servers if s.enabled}
        for name in list(self._servers):
            state = self._servers[name]
            want = desired.get(name)
            unchanged = want is not None and (want.command, tuple(want.args)) == (
                state.settings.command,
                tuple(state.settings.args),
            )
            if unchanged and state.client is not None and state.client.alive:
                continue  # already running the desired command — leave it alone
            if state.client is not None:
                await state.client.stop()
                state.client = None
            state.tools.clear()
            if want is None:
                del self._servers[name]
        for name, settings in desired.items():
            existing = self._servers.get(name)
            if existing is not None and existing.client is not None and existing.client.alive:
                continue  # untouched above
            self._servers[name] = _ServerState(settings)
            await self._start_one(self._servers[name])

    def resync_soon(self, servers: list[McpServerSettings]) -> None:
        """Fire-and-forget resync — deferred-apply commits happen inside synchronous idle
        callbacks (app.maybe_commit_staged), so the resync is scheduled, not awaited
        (mirrors ProviderRegistry.refresh_soon)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.resync(servers))
        self._resync_tasks.add(task)
        task.add_done_callback(self._resync_tasks.discard)

    async def shutdown(self) -> None:
        for state in self._servers.values():
            if state.client is not None:
                await state.client.stop()
                state.client = None
            state.tools.clear()

    # -- registry surface --------------------------------------------------
    def tools(self) -> list[Tool]:
        """Live tools from connected servers only — a crashed server's tools are
        removed, so later runs simply do not see them."""
        out: list[Tool] = []
        for state in self._servers.values():
            if state.client is not None and state.client.alive:
                out.extend(state.tools)
        return out

    def notices(self) -> list[Notice]:
        return list(self._notices)

    @property
    def connected(self) -> list[str]:
        return [
            name
            for name, state in self._servers.items()
            if state.client is not None and state.client.alive
        ]

    # -- calls -------------------------------------------------------------
    async def call(self, server: str, remote_name: str, args: dict) -> ToolResult:
        state = self._servers.get(server)
        if state is None or state.client is None or not state.client.alive:
            return ToolResult(
                "error",
                f"MCP server {server!r} is not connected; its tools are unavailable.",
            )
        try:
            text, is_error = await state.client.call_tool(
                remote_name, args, timeout_s=self._call_timeout
            )
        except McpError as exc:
            # Timeout vs crash: a timed-out call on a living server is one failed
            # step; a dead server loses its tools for the rest of the process.
            if not state.client.alive:
                dead_client = state.client
                await dead_client.stop()
                state.client = None
                self._record_failure(state, str(exc), client=dead_client)
            return ToolResult("error", f"mcp call {server}:{remote_name} failed: {exc}")
        status = "error" if is_error else "ok"
        summary = f"mcp {server}:{remote_name} {'failed' if is_error else 'ok'}"
        return ToolResult(status, text or "(empty result)", summary=summary)

    # -- internals ---------------------------------------------------------
    def _record_failure(
        self, state: _ServerState, error: str, *, client: McpClient | None = None
    ) -> None:
        state.failed = True
        state.tools.clear()
        stderr_hint = ""
        if client is not None and client.stderr_tail:
            stderr_hint = f" [stderr: {list(client.stderr_tail)[-1]}]"
        message = (
            f"MCP server {state.settings.name!r} failed: {error}{stderr_hint} "
            "Its tools are unavailable; fix the command/args in settings and restart."
        )
        logger.warning(message)
        self._notices.append(Notice("warn", "mcp_server_failed", message))
