"""Grammar-constrained structured completions with retry-with-repair.

One completion whose output must be a JSON instance of a schema. The request carries
OpenAI `response_format: {"type": "json_schema", ...}` — llama-server compiles that to
GBNF, so a real slot cannot emit anything else. Scripted/mock backends (and semantic
violations the grammar cannot see) can still produce bad instances, so the result is
parsed AND validated here, with at most `max_repair_attempts` re-prompts carrying the
error back — after that the caller gets `data=None` and degrades gracefully.

Used by Ultra (plan + worker result envelope) and deep research (query plan).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import httpx

from suiban.tools import schema as schema_mod

CompleteFn = Callable[[dict], Awaitable[dict]]

MAX_REPAIR_ATTEMPTS = 2


@dataclass
class StructuredResult:
    data: dict | None
    error: str | None = None
    usage: dict[str, int] = field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}
    )


def _accumulate(usage: dict[str, int], response: dict) -> None:
    got = response.get("usage") or {}
    usage["prompt_tokens"] += int(got.get("prompt_tokens", 0))
    usage["completion_tokens"] += int(got.get("completion_tokens", 0))
    usage["thinking_tokens"] += int(got.get("thinking_tokens", 0))


def _parse_instance(content: str, schema: dict) -> tuple[dict | None, str | None]:
    try:
        parsed = json.loads(content)
    except ValueError as exc:
        return None, f"output is not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, f"output must be a JSON object, got {type(parsed).__name__}"
    errors = schema_mod.validate(parsed, schema)
    if errors:
        return None, "; ".join(errors)
    return parsed, None


async def request_structured(
    complete: CompleteFn,
    payload: dict,
    schema: dict,
    *,
    schema_name: str,
    max_repair_attempts: int = MAX_REPAIR_ATTEMPTS,
) -> StructuredResult:
    """Run the completion in `payload` (messages/model/sampling already set) with the
    given response schema; validate; repair-retry; never raise on model output —
    backend transport errors surface as `error` too."""
    payload = dict(payload)
    payload["stream"] = False
    payload["response_format"] = {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }
    messages = list(payload.get("messages") or [])
    result = StructuredResult(data=None)

    for attempt in range(1 + max_repair_attempts):
        payload["messages"] = list(messages)
        try:
            response = await complete(payload)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            result.error = f"backend request failed: {exc}"
            return result
        _accumulate(result.usage, response)
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            result.error = "backend returned a malformed completion"
            return result
        data, error = _parse_instance(content, schema)
        if data is not None:
            result.data = data
            result.error = None
            return result
        result.error = error
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your output did not match the required {schema_name} schema: {error}. "
                    f"Repair attempt {attempt + 1}/{max_repair_attempts}: reply with ONLY a "
                    "corrected JSON object."
                ),
            }
        )
    return result
