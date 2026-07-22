"""Tool interface (MCP-compatible shape): name, description, JSON-schema params,
async run() -> ToolResult.

ToolResult.status mirrors the api.md `tool_result` event statuses: "ok" | "error" |
"denied" ("denied" = a confirm-gated operation that needs a confirmation token; the
result carries the token so clients can re-ask the user).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUMMARY_MAX_CHARS = 200


@dataclass
class ToolContext:
    """Per-run context handed to every tool invocation.

    `role` is the slot role of the model driving the loop (orchestrator | worker |
    utility) — write tools re-check it as defense in depth even though non-orchestrator
    registries never contain them.
    """

    session_id: str
    workdir: Path
    role: str = "orchestrator"
    mode: str = "chat"
    # Issued confirmation tokens: token -> the exact operation it confirms (single use).
    # Covers destructive shell commands AND file mutations (fs_write/fs_undo), api.md
    # 2026-07-22b.
    confirm_tokens: dict[str, str] = field(default_factory=dict)
    # auto_confirm bypass (api.md 2026-07-22b, code/ultra only): when True the confirm
    # gate is skipped for destructive shell commands AND file mutations — they run
    # WITHOUT emitting denied/confirm_token. A dangerous power-user setting; shell.py /
    # fs.py log every auto-confirmed action (logger.warning with command/path).
    auto_confirm: bool = False


@dataclass
class ToolResult:
    status: str  # "ok" | "error" | "denied"
    content: str  # full text handed back to the model
    summary: str = ""  # short line for the tool_result SSE event (truncated)
    # Single-use confirmation token, set ONLY on status "denied" (api.md: the
    # tool_result event carries it so clients can run the confirmation flow).
    confirm_token: str | None = None

    def __post_init__(self) -> None:
        if not self.summary:
            self.summary = self.content
        if len(self.summary) > SUMMARY_MAX_CHARS:
            self.summary = self.summary[: SUMMARY_MAX_CHARS - 1] + "…"


class Tool(abc.ABC):
    """One callable tool. Subclasses define the schema and the behavior."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object
    timeout_s: float = 60.0  # registry-enforced ceiling per invocation

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult: ...

    def openai_schema(self) -> dict:
        """OpenAI `tools` entry — what gets passed to llama-server (--jinja renders it
        into the ChatML template; decoding is grammar-constrained against it)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
