"""In-process fake llama-server: a deterministic OpenAI-compatible responder.

Enabled by SUIBAN_LLAMA_MOCK=1 (all tests and modelless client dev run against this).
Deterministic on purpose: fixed text, created=0, ids derived from the request — so
recorded fixtures never drift.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from suiban.memory.titling import TITLE_SYSTEM_PROMPT

MOCK_COMPLETION_TEXT = (
    "This is a deterministic canned completion from suiban's mock llama backend. "
    "No model weights were involved."
)

# What the mock "utility model" answers to the session-titling prompt: a plausible
# short title so the auto-titling path runs end to end in tests.
MOCK_SESSION_TITLE = "Mock conversation title"


def _is_title_request(body: dict) -> bool:
    messages = body.get("messages") or []
    return bool(messages) and messages[0].get("content") == TITLE_SYSTEM_PROMPT


def _request_fingerprint(body: dict) -> str:
    canonical = json.dumps(body.get("messages", []), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def instance_for_schema(schema: dict) -> object:
    """Deterministic minimal instance of a JSON schema (the subset our tool/plan
    schemas use). This is what a real llama-server produces grammar-constrained; the
    mock produces the smallest valid instance so structured pipelines (ultra plans,
    research plans, worker result envelopes) run end-to-end modelless."""
    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]
    match schema.get("type"):
        case "object":
            return {
                key: instance_for_schema(sub)
                for key, sub in (schema.get("properties") or {}).items()
            }
        case "array":
            items = schema.get("items")
            return [instance_for_schema(items)] if isinstance(items, dict) else []
        case "string":
            return "mock"
        case "integer" | "number":
            return schema.get("minimum", 0)
        case "boolean":
            return False
        case "null":
            return None
    return "mock"


def _json_schema_of(body: dict) -> dict | None:
    """Schema from an OpenAI response_format: {"type":"json_schema","json_schema":
    {"name":..., "schema": {...}}}. None when the request is unconstrained."""
    response_format = body.get("response_format") or {}
    if response_format.get("type") != "json_schema":
        return None
    schema = (response_format.get("json_schema") or {}).get("schema")
    return schema if isinstance(schema, dict) else None


def _usage(text: str) -> dict:
    # Fake but stable token accounting: 1 token per whitespace-separated word.
    return {
        "prompt_tokens": 7,
        "completion_tokens": len(text.split()),
        "total_tokens": 7 + len(text.split()),
    }


def _mock_tool_call(body: dict) -> dict | None:
    """Deterministic tool call for pass-through testing: only when the request carries
    tools AND demands one (tool_choice \"required\"). Stage 1 fixtures never send
    tools, so their behavior is unchanged."""
    tools = body.get("tools") or []
    if not tools or body.get("tool_choice") != "required":
        return None
    name = (tools[0].get("function") or {}).get("name", "unknown")
    return {
        "id": f"call-mock-{_request_fingerprint(body)}",
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def build_mock_app(slot_model: str = "bonsai-27b") -> FastAPI:
    app = FastAPI(title="suiban-mock-llama", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        model = body.get("model", slot_model)
        completion_id = f"chatcmpl-mock-{_request_fingerprint(body)}"
        schema = _json_schema_of(body)
        if schema is not None:
            text = json.dumps(instance_for_schema(schema))
        elif _is_title_request(body):
            text = MOCK_SESSION_TITLE
        else:
            text = MOCK_COMPLETION_TEXT

        tool_call = _mock_tool_call(body)
        if tool_call is not None:
            if body.get("stream"):

                async def tool_stream() -> AsyncIterator[str]:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [{"index": 0, **tool_call}],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    final = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                        "usage": _usage(""),
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(tool_stream(), media_type="text/event-stream")
            return JSONResponse(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": _usage(""),
                }
            )

        if body.get("stream"):

            async def stream() -> AsyncIterator[str]:
                first = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(first)}\n\n"
                # Structured output streams as one chunk (splitting JSON on spaces
                # would corrupt string values); prose streams word by word.
                pieces = [text] if schema is not None else [w + " " for w in text.split(" ")]
                for piece in pieces:
                    chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                final = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": _usage(text),
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": completion_id,
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": _usage(text),
            }
        )

    return app
