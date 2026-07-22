"""JSON-RPC 2.0 over stdio: the MCP client transport (spec rev 2025-06-18).

One `McpClient` owns one server subprocess. Framing is newline-delimited JSON both
ways (the MCP stdio transport). The lifecycle follows the spec:

    -> initialize {protocolVersion, capabilities, clientInfo}
    <- initialize result {protocolVersion, capabilities, serverInfo}
    -> notifications/initialized
    -> tools/list [paginated via nextCursor]
    -> tools/call {name, arguments}   (per-call timeout, enforced here)

Server-initiated traffic is tolerated, never required: `ping` requests get an empty
result, other requests get -32601, notifications are ignored. A dead/hung server
turns into `McpError` on the pending call and `alive == False` — the manager layer
translates that into an `mcp_server_failed` notice and tool removal; nothing here
ever crashes the app.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from typing import Any

from suiban import __version__

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_REQUEST_TIMEOUT_S = 30.0  # control-plane requests (initialize, tools/list)
DEFAULT_CALL_TIMEOUT_S = 60.0  # tools/call, overridable per call
_STDERR_KEEP_LINES = 50
_MAX_LINE_BYTES = 8 * 1024 * 1024  # tool results can be large; bound the reader


class McpError(Exception):
    """Any failure talking to an MCP server (spawn, protocol, timeout, crash)."""


class McpClient:
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        *,
        request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
    ) -> None:
        self.name = name
        self._command = command
        self._args = list(args or [])
        self._request_timeout = request_timeout_s
        self._proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._dead = False
        self.stderr_tail: deque[str] = deque(maxlen=_STDERR_KEEP_LINES)
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str = ""

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None and not self._dead

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Spawn the subprocess and run the initialize handshake."""
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self._command,
                *self._args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_LINE_BYTES,
            )
        except (OSError, ValueError) as exc:
            raise McpError(f"failed to spawn {self._command!r}: {exc}") from exc
        self._reader_task = asyncio.create_task(self._read_stdout())
        self._stderr_task = asyncio.create_task(self._read_stderr())

        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "suiban", "version": __version__},
            },
        )
        self.protocol_version = str(result.get("protocolVersion", ""))
        info = result.get("serverInfo")
        self.server_info = info if isinstance(info, dict) else {}
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def stop(self) -> None:
        self._dead = True
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._fail_pending("client stopped")

    # -- MCP operations ----------------------------------------------------
    async def list_tools(self) -> list[dict]:
        """All tool definitions ({name, description?, inputSchema?}), paginated."""
        tools: list[dict] = []
        cursor: str | None = None
        for _page in range(16):  # generous pagination bound; no server needs more
            params: dict = {"cursor": cursor} if cursor else {}
            result = await self._request("tools/list", params)
            page = result.get("tools")
            if not isinstance(page, list):
                raise McpError(f"tools/list returned no tools array: {result!r}")
            tools.extend(t for t in page if isinstance(t, dict) and t.get("name"))
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    async def call_tool(
        self, name: str, arguments: dict, *, timeout_s: float = DEFAULT_CALL_TIMEOUT_S
    ) -> tuple[str, bool]:
        """Invoke one tool; returns (text, is_error). Text is the concatenation of
        the result's text-type content items; non-text items are noted, not dropped
        silently."""
        result = await self._request(
            "tools/call", {"name": name, "arguments": arguments}, timeout_s=timeout_s
        )
        parts: list[str] = []
        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"[non-text content: {item.get('type', 'unknown')}]")
        text = "\n".join(p for p in parts if p)
        return text, bool(result.get("isError", False))

    # -- JSON-RPC plumbing -------------------------------------------------
    def _send(self, message: dict) -> None:
        if self._proc is None or self._proc.stdin is None or not self.alive:
            raise McpError(f"mcp server {self.name!r} is not running")
        line = json.dumps(message, ensure_ascii=False) + "\n"
        try:
            self._proc.stdin.write(line.encode("utf-8"))
        except (OSError, RuntimeError) as exc:  # broken pipe: the server died
            self._mark_dead(f"write failed: {exc}")
            raise McpError(f"mcp server {self.name!r} pipe closed: {exc}") from exc

    async def _request(self, method: str, params: dict, *, timeout_s: float | None = None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            timeout = timeout_s if timeout_s is not None else self._request_timeout
            try:
                return await asyncio.wait_for(future, timeout=timeout)
            except TimeoutError:
                raise McpError(
                    f"mcp server {self.name!r}: {method} timed out after {timeout:.0f}s"
                ) from None
        finally:
            # A late response for a timed-out id is dropped by the reader (no
            # pending future) — later calls with fresh ids are unaffected.
            self._pending.pop(request_id, None)

    async def _read_stdout(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stdout = self._proc.stdout
        while True:
            try:
                line = await stdout.readline()
            except (ValueError, OSError) as exc:  # oversized line / closed pipe
                self._mark_dead(f"stdout read failed: {exc}")
                return
            if not line:
                self._mark_dead("server closed stdout (process exited)")
                return
            try:
                message = json.loads(line)
            except ValueError:
                logger.debug("mcp[%s] non-JSON line ignored: %r", self.name, line[:200])
                continue
            if not isinstance(message, dict):
                continue
            self._dispatch(message)

    def _dispatch(self, message: dict) -> None:
        msg_id = message.get("id")
        if msg_id is not None and ("result" in message or "error" in message):
            future = self._pending.pop(msg_id, None) if isinstance(msg_id, int) else None
            if future is None or future.done():
                return  # late/unknown response: dropped (see _request)
            if "error" in message:
                err = message["error"] or {}
                future.set_exception(
                    McpError(
                        f"mcp server {self.name!r} error {err.get('code')}: "
                        f"{err.get('message', 'unknown error')}"
                    )
                )
            else:
                result = message.get("result")
                future.set_result(result if isinstance(result, dict) else {})
            return
        method = message.get("method", "")
        if msg_id is not None:  # server-to-client REQUEST
            if method == "ping":
                reply: dict = {"jsonrpc": "2.0", "id": msg_id, "result": {}}
            else:
                reply = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"method not supported: {method}"},
                }
            with contextlib.suppress(McpError):
                self._send(reply)
            return
        logger.debug("mcp[%s] notification ignored: %s", self.name, method)

    async def _read_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        stderr = self._proc.stderr
        while True:
            try:
                line = await stderr.readline()
            except (ValueError, OSError):
                return
            if not line:
                return
            self.stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())

    def _mark_dead(self, reason: str) -> None:
        if self._dead:
            return
        self._dead = True
        logger.warning("mcp server %r died: %s", self.name, reason)
        self._fail_pending(reason)

    def _fail_pending(self, reason: str) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(McpError(f"mcp server {self.name!r}: {reason}"))
        self._pending.clear()
