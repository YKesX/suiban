"""Mode registry: name -> versioned prompt file, toolset, default effort, endpoint.

Prompt text lives in prompts/*.md with a versioned first-line header ("# code v1");
/v1/modes exposes the version string ("code@1") but never the text (api.md §7).
deep_research is listed as a mode but is NOT a chat mode — its endpoint is /v1/jobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from suiban.config import Effort
from suiban.errors import BonsaiError
from suiban.tools.registry import mode_tool_names

PROMPTS_DIR = Path(__file__).parent / "prompts"

_HEADER_RE = re.compile(r"^#\s*(?P<name>[\w-]+)\s+v(?P<version>\d+)\s*$")


@dataclass(frozen=True)
class Mode:
    name: str
    description: str
    prompt_file: str  # filename under prompts/
    default_effort: Effort
    endpoint: str  # "/v1/chat/completions" | "/v1/jobs"

    @property
    def tools(self) -> list[str]:
        return mode_tool_names(self.name)

    def prompt_path(self) -> Path:
        return PROMPTS_DIR / self.prompt_file

    def prompt_version(self) -> str:
        """ "code@3"-style version parsed from the prompt file's first line."""
        first_line = self.prompt_path().read_text(encoding="utf-8").splitlines()[0]
        match = _HEADER_RE.match(first_line)
        if match is None:  # a broken header is a packaging bug — surface it, honestly
            return f"{self.name}@unversioned"
        return f"{match.group('name')}@{match.group('version')}"

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "system_prompt_version": self.prompt_version(),
            "tools": self.tools,
            "default_effort": self.default_effort,
            "endpoint": self.endpoint,
        }


MODES: dict[str, Mode] = {
    mode.name: mode
    for mode in (
        Mode(
            name="chat",
            description="Direct conversation with the orchestrator; on-demand memory "
            "recall, light browsing.",
            prompt_file="chat.md",
            default_effort="mid",
            endpoint="/v1/chat/completions",
        ),
        Mode(
            name="code",
            description="Agentic coding in the session workdir: plan → act → verify, "
            "confirm-gated shell, read-only git.",
            prompt_file="code.md",
            default_effort="high",
            endpoint="/v1/chat/completions",
        ),
        Mode(
            name="ultra",
            description="Multi-agent mode: the orchestrator decomposes the task and "
            "delegates to contained worker models.",
            prompt_file="ultra.md",
            default_effort="high",
            endpoint="/v1/chat/completions",
        ),
        Mode(
            name="deep_research",
            description="Long-running, multi-source research producing a cited report. "
            "Async job — submit via /v1/jobs, not chat.",
            prompt_file="research.md",
            default_effort="high",
            endpoint="/v1/jobs",
        ),
    )
}

CHAT_MODES: tuple[str, ...] = ("chat", "code", "ultra")


def get_mode(name: str) -> Mode:
    mode = MODES.get(name)
    if mode is None:
        raise BonsaiError(404, f"no such mode: {name!r}", code="mode_not_found")
    return mode


@lru_cache(maxsize=8)
def system_prompt(name: str) -> str:
    """Full prompt text for a mode (internal use only — never exposed over HTTP)."""
    return get_mode(name).prompt_path().read_text(encoding="utf-8")
