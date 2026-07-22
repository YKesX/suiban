"""Chat-import parsers (api.md 2026-07-22b, POST /v1/memory/sessions/import).

Turn another tool's exported conversation(s) into a normalized shape the session
archive can store. Pure parsing — no I/O, no model, no network: each parser takes the
provider's `data` payload and returns a list of `ParsedSession`. A payload that does not
match the claimed provider shape raises `ImportUnrecognized` → the router answers 400
`import_unrecognized`.

Supported providers:
- `openai`     — the ChatGPT `conversations.json` export (a list of conversations, each
                 with a `mapping` node tree; messages are ordered by create_time).
- `claude`     — the claude.ai data export (conversations with `chat_messages`).
- `claude-code`— a `~/.claude` project transcript: JSONL, one JSON object per line with
                 role/content (passed as the raw JSONL string or a parsed list).
- `generic`    — `{title?, messages:[{role, content}]}` (or a list of those).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

PROVIDERS: tuple[str, ...] = ("openai", "claude", "claude-code", "generic")

# Provider role tokens → our archive roles (system|user|assistant|tool).
_ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
}

_TITLE_MAX_CHARS = 60


class ImportUnrecognized(Exception):
    """The payload does not match the claimed provider's export shape."""


@dataclass(frozen=True)
class ParsedSession:
    title: str | None
    messages: list[dict]  # [{"role": str, "content": str}]


def _norm_role(role: object) -> str:
    return _ROLE_MAP.get(str(role).strip().lower(), "user")


def _text(content: object) -> str:
    """Flatten a message's content to plain text across the shapes providers use:
    a bare string; a list of blocks/strings; or an object with `parts`/`text`."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return "\n".join(str(p) for p in parts if isinstance(p, str) and p)
        if isinstance(content.get("text"), str):
            return content["text"]
    return ""


def transcript_text(messages: list[dict]) -> str:
    """Role-prefixed transcript, used as the compression input for `compress:true`."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def default_title(messages: list[dict]) -> str | None:
    """A title for an untitled import: the first user message's first line, clipped."""
    for message in messages:
        text = message.get("content", "").strip()
        if message.get("role") == "user" and text:
            return text.splitlines()[0][:_TITLE_MAX_CHARS]
    return None


# -- per-provider parsers -----------------------------------------------------
def _parse_generic(data: object) -> list[ParsedSession]:
    items = data if isinstance(data, list) else [data]
    sessions: list[ParsedSession] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("messages"), list):
            raise ImportUnrecognized(
                "generic import expects {title?, messages:[{role, content}]} (or a list of those)"
            )
        messages: list[dict] = []
        for msg in item["messages"]:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                raise ImportUnrecognized(
                    "each generic message must be an object with 'role' and 'content'"
                )
            messages.append({"role": _norm_role(msg["role"]), "content": _text(msg["content"])})
        title = item.get("title") if isinstance(item.get("title"), str) else None
        sessions.append(ParsedSession(title=title, messages=messages))
    if not sessions:
        raise ImportUnrecognized("generic import contained no conversations")
    return sessions


def _conversations(data: object, key: str) -> list:
    """Unwrap a provider export to a list of conversation objects: a bare list, a single
    conversation dict, or a wrapper `{key: [...]}` (e.g. `{"conversations": [...]}`)."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get(key), list):
            return data[key]
        return [data]
    return []


def _parse_openai(data: object) -> list[ParsedSession]:
    sessions: list[ParsedSession] = []
    for conv in _conversations(data, "conversations"):
        if not isinstance(conv, dict) or not isinstance(conv.get("mapping"), dict):
            continue
        nodes: list[tuple[float, str, str]] = []
        for node in conv["mapping"].values():
            if not isinstance(node, dict) or not isinstance(node.get("message"), dict):
                continue
            message = node["message"]
            author = message.get("author") if isinstance(message.get("author"), dict) else {}
            role = author.get("role", "user")
            text = _text(message.get("content"))
            if not text.strip():
                continue
            created = message.get("create_time")
            nodes.append((created if isinstance(created, int | float) else 0.0, role, text))
        nodes.sort(key=lambda item: item[0])
        messages = [{"role": _norm_role(role), "content": text} for _, role, text in nodes]
        if messages:
            title = conv.get("title") if isinstance(conv.get("title"), str) else None
            sessions.append(ParsedSession(title=title, messages=messages))
    if not sessions:
        raise ImportUnrecognized(
            "no OpenAI conversations found (expected conversations.json: a list of "
            "conversations each with a 'mapping' node tree)"
        )
    return sessions


def _parse_claude(data: object) -> list[ParsedSession]:
    sessions: list[ParsedSession] = []
    for conv in _conversations(data, "conversations"):
        if not isinstance(conv, dict) or not isinstance(conv.get("chat_messages"), list):
            continue
        messages: list[dict] = []
        for msg in conv["chat_messages"]:
            if not isinstance(msg, dict):
                continue
            sender = msg.get("sender") or msg.get("role")
            text = msg.get("text") if isinstance(msg.get("text"), str) else ""
            if not text.strip():
                text = _text(msg.get("content"))
            if not text.strip():
                continue
            messages.append({"role": _norm_role(sender), "content": text})
        if messages:
            name = conv.get("name") or conv.get("title")
            title = name if isinstance(name, str) else None
            sessions.append(ParsedSession(title=title, messages=messages))
    if not sessions:
        raise ImportUnrecognized(
            "no Claude conversations found (expected the claude.ai export: objects with "
            "a 'chat_messages' list)"
        )
    return sessions


def _cc_role_content(obj: dict) -> tuple[str | None, object]:
    """Extract (role, content) from one claude-code JSONL record across its variants."""
    if "role" in obj and "content" in obj:
        return obj["role"], obj["content"]
    inner = obj.get("message")
    if isinstance(inner, dict) and "role" in inner:
        return inner.get("role"), inner.get("content")
    kind = obj.get("type")
    if kind in ("user", "assistant") and "content" in obj:
        return kind, obj["content"]
    return None, None


def _parse_claude_code(data: object) -> list[ParsedSession]:
    if isinstance(data, str):
        records: list = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ImportUnrecognized(
                    "claude-code import expects JSONL (one JSON object per line)"
                ) from exc
    elif isinstance(data, list):
        records = data
    else:
        raise ImportUnrecognized(
            "claude-code import expects JSONL text or a list of {role, content} objects"
        )

    messages: list[dict] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        role, content = _cc_role_content(record)
        if role is None:
            continue
        text = _text(content)
        if not text.strip():
            continue
        messages.append({"role": _norm_role(role), "content": text})
    if not messages:
        raise ImportUnrecognized(
            "no messages in the claude-code transcript (expected objects with role/content)"
        )
    return [ParsedSession(title=None, messages=messages)]


_PARSERS = {
    "openai": _parse_openai,
    "claude": _parse_claude,
    "claude-code": _parse_claude_code,
    "generic": _parse_generic,
}


def parse_import(provider: str, data: object) -> list[ParsedSession]:
    """Parse `data` for `provider` into archived-session inputs. `provider` is validated
    by the caller (router) against PROVIDERS before this is reached."""
    parser = _PARSERS[provider]
    return parser(data)
