"""Read-only git tool: status / log / diff / show, nothing else.

Mutations never reach git: the subcommand is an enum in the grammar-constrained schema,
and the argument list is scanned for the few read-subcommand flags that can still write
files or execute configured commands (--output, --ext-diff, ...). Rejected calls return
a clear error; the mutation path for models is the confirm-gated shell tool.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from suiban.tools.base import Tool, ToolContext, ToolResult

ALLOWED_SUBCOMMANDS = ("status", "log", "diff", "show")

# Flags of the read-only subcommands that write files or run external commands.
_FORBIDDEN_ARG_PREFIXES = ("--output", "--ext-diff", "--textconv", "--exec")

GIT_TIMEOUT_S = 20.0
OUTPUT_MAX_CHARS = 16_000


class GitRoTool(Tool):
    name = "git_ro"
    description = (
        "Read-only git: status, log, diff, or show in the session workdir. "
        "Anything that mutates the repository is rejected — use the shell tool "
        "(confirm-gated) if the user explicitly wants a mutation."
    )
    parameters = {
        "type": "object",
        "properties": {
            "subcommand": {"type": "string", "enum": list(ALLOWED_SUBCOMMANDS)},
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra arguments, e.g. ['--stat'] or ['HEAD~1..HEAD'].",
            },
        },
        "required": ["subcommand"],
        "additionalProperties": False,
    }
    timeout_s = GIT_TIMEOUT_S + 5.0

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        subcommand: str = args["subcommand"]
        extra: list[str] = args.get("args", [])
        if subcommand not in ALLOWED_SUBCOMMANDS:
            return ToolResult(
                "error",
                f"git_ro only allows {', '.join(ALLOWED_SUBCOMMANDS)}; got {subcommand!r}",
            )
        for arg in extra:
            if arg.startswith(_FORBIDDEN_ARG_PREFIXES):
                return ToolResult("error", f"argument {arg!r} is not allowed (it can write/exec)")

        # Confine repository discovery to the jail (audit 2026-07-22): without a ceiling,
        # `git status` in the workdir walks UP the tree and would read a repository whose
        # .git lives ABOVE the jail root. GIT_CEILING_DIRECTORIES = the workdir's parent
        # stops the upward search at the workdir, so only a repo AT (or below) the jail
        # is ever discovered.
        env = dict(os.environ)
        env["GIT_CEILING_DIRECTORIES"] = str(ctx.workdir.resolve().parent)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "--no-pager",
            subcommand,
            *extra,
            cwd=str(ctx.workdir),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=GIT_TIMEOUT_S)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult("error", f"git {subcommand} timed out after {GIT_TIMEOUT_S:.0f}s")

        body = stdout.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            return ToolResult("error", f"git {subcommand} failed (exit {proc.returncode}): {err}")
        if len(body) > OUTPUT_MAX_CHARS:
            body = body[:OUTPUT_MAX_CHARS] + f"\n… [output truncated at {OUTPUT_MAX_CHARS} chars]"
        return ToolResult("ok", body or "(no output)", summary=f"git {subcommand}")
