"""The plan tool: code mode's plan-before-acting, grammar-constrained.

Calling it does nothing except surface the plan as the api.md `plan` SSE event
({"steps": [...]}) — the agent loop special-cases this tool name for event emission.
Making the plan a tool call (instead of parsing prose) keeps it schema-constrained.
"""

from __future__ import annotations

from typing import Any

from suiban.tools.base import Tool, ToolContext, ToolResult

PLAN_TOOL_NAME = "plan"


class PlanTool(Tool):
    name = PLAN_TOOL_NAME
    description = (
        "Declare your plan before acting: a short ordered list of concrete steps. "
        "Call this once, first, in any non-trivial task; revise by calling it again."
    )
    parameters = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "description": "Ordered, concrete steps (one line each).",
            }
        },
        "required": ["steps"],
        "additionalProperties": False,
    }

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        steps: list[str] = args["steps"]
        if not steps:
            return ToolResult("error", "a plan needs at least one step")
        return ToolResult(
            "ok",
            "plan recorded:\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1)),
            summary=f"plan: {len(steps)} steps",
        )
