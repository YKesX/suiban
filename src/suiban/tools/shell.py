"""Shell tool: subprocess in the session workdir, with a timeout and a confirm gate.

Destructive commands are NOT executed on first request. The tool returns
status "denied" plus a one-shot confirmation token; the caller (via the tool_result
SSE event) asks the human, and the model re-issues the identical command with
`confirm_token`. The token is bound to the exact command string and is single-use —
a confirmed `rm -rf build` cannot be replayed as `rm -rf /`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import secrets
from typing import Any

from suiban.tools.base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
MAX_TIMEOUT_S = 120.0
OUTPUT_MAX_CHARS = 16_000

# Env scrub (audit 2026-07-22): the tool subprocess inherits a COPY of the server env
# with secret-bearing variables removed, so a command like `env` / `printenv` cannot
# read the Telegram/API/HF tokens the server holds. Defense in depth, NOT a boundary —
# a determined command can still read files, and the real sandbox (bwrap/landlock) is a
# KNOWN_ISSUE for v1.1. Match is case-insensitive.
_SECRET_ENV_SUBSTRINGS = ("TOKEN", "KEY", "SECRET", "PASSWORD")
_SECRET_ENV_PREFIXES = ("BONSAI_", "TELEGRAM_", "HF_", "HUGGING", "AWS_")


def scrubbed_env() -> dict[str, str]:
    """os.environ minus any variable whose name looks secret-bearing."""
    out: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(sub in upper for sub in _SECRET_ENV_SUBSTRINGS):
            continue
        if upper.startswith(_SECRET_ENV_PREFIXES):
            continue
        out[name] = value
    return out


# Word-boundary patterns that make a command destructive (confirm-gated). Deliberately
# conservative: false positives cost one confirmation round-trip; false negatives cost
# data. Output redirection counts — it overwrites files.
#
# AUDIT SEAM (security audit, next session): a regex denylist is best-effort by
# construction — quoting/concatenation ('r''m'), $(echo rm), base64 payloads, env-var
# indirection, and shell aliasing all slip past it. The audit should weigh an
# allowlist grammar or an OS-level sandbox (bwrap/landlock) for the shell tool; until
# then these patterns only raise the bar, they are not a boundary.
_DESTRUCTIVE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\brm\b",
        r"\brmdir\b",
        r"\bmv\b",
        r"\bdd\b",
        r"\bmkfs\w*\b",
        r"\bshred\b",
        r"\btruncate\b",
        r"\bchmod\b",
        r"\bchown\b",
        r"\bsudo\b",
        r"\bsu\b",
        r"\bkill\b",
        r"\bpkill\b",
        r"\bkillall\b",
        r"\bsystemctl\b",
        r"\breboot\b",
        r"\bshutdown\b",
        r"\bgit\s+(push|commit|reset|clean|checkout|restore|rebase|merge|tag)\b",
        r"[>|]\s*",  # output redirection / pipes into arbitrary consumers
        r"\bcurl\b.*\|\s*(ba)?sh",
        r"\bwget\b",
        # `find ... -delete` deletes without ever spelling "rm".
        r"\bfind\b.*\s-delete\b",
        # Block-device / filesystem-partition primitives (data-destroying, no "rm").
        r"\b(wipefs|blkdiscard|mkswap|fdisk|sfdisk|sgdisk|parted)\b",
        # Deletion primitives that interpreter one-liners reach for (os.unlink,
        # shutil.rmtree, perl unlink, fs.rmSync ...) — none contain "rm" as a word.
        r"\b(unlink|rmtree|removedirs|rmSync|rmdirSync|unlinkSync)\b",
        r"\bos\s*\.\s*remove\b",
        # Interpreter inline code (-c/-e) that spawns subprocesses: whatever it runs
        # is invisible to the patterns above, so gate the spawn itself.
        r"\b(python[0-9.]*|perl|ruby|node|deno|bun)\b.*\s-[ce]\b"
        r".*\b(system|popen|subprocess|spawn\w*|exec\w*)\b",
    )
)


def is_destructive(command: str) -> bool:
    return any(p.search(command) for p in _DESTRUCTIVE_PATTERNS)


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run a shell command in the session workdir. Destructive operations (rm, mv, "
        "redirection, git mutations, ...) are refused with status 'denied' and a "
        "confirm_token; after the user approves, re-run the identical command with "
        "that confirm_token."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command line to execute."},
            "timeout_s": {
                "type": "number",
                "description": f"Seconds before the command is killed (default "
                f"{DEFAULT_TIMEOUT_S:.0f}, max {MAX_TIMEOUT_S:.0f}).",
                "minimum": 1,
                "maximum": MAX_TIMEOUT_S,
            },
            "confirm_token": {
                "type": "string",
                "description": "Token from a previous 'denied' result, after user approval.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    timeout_s = MAX_TIMEOUT_S + 10.0  # registry ceiling; the subprocess timeout is tighter

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command: str = args["command"]
        timeout = min(float(args.get("timeout_s", DEFAULT_TIMEOUT_S)), MAX_TIMEOUT_S)

        if is_destructive(command):
            if ctx.auto_confirm:
                # auto_confirm bypass (api.md 2026-07-22b): the gate is skipped, but
                # every auto-confirmed destructive action is logged loudly.
                logger.warning(
                    "auto_confirm: destructive shell command run WITHOUT confirmation "
                    "(session %s): %s",
                    ctx.session_id,
                    command,
                )
            else:
                token = args.get("confirm_token")
                expected_command = ctx.confirm_tokens.pop(token, None) if token else None
                if expected_command != command:
                    new_token = secrets.token_urlsafe(12)
                    ctx.confirm_tokens[new_token] = command
                    return ToolResult(
                        "denied",
                        "This command is destructive and needs user confirmation. "
                        f"Ask the user; if they approve, re-run the exact same command with "
                        f'confirm_token "{new_token}". Command: {command}',
                        summary=f"confirmation required: {command}",
                        confirm_token=new_token,
                    )

        ctx.workdir.mkdir(parents=True, exist_ok=True)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(ctx.workdir),
            env=scrubbed_env(),  # secret-bearing vars stripped (defense in depth)
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                "error",
                f"command timed out after {timeout:.0f}s and was killed: {command}",
                summary=f"timeout ({timeout:.0f}s): {command}",
            )

        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        body = out
        if err:
            body += ("\n" if body else "") + "[stderr]\n" + err
        if len(body) > OUTPUT_MAX_CHARS:
            body = body[:OUTPUT_MAX_CHARS] + f"\n… [output truncated at {OUTPUT_MAX_CHARS} chars]"
        body = f"exit code {proc.returncode}\n{body}".rstrip()
        status = "ok" if proc.returncode == 0 else "error"
        return ToolResult(status, body, summary=f"exit {proc.returncode}: {command}")
