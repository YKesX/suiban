#!/usr/bin/env python3
"""Tiny MCP stdio server for CI tests — stdlib only, speaks just enough protocol.

Implements the slice of MCP (JSON-RPC 2.0, newline-delimited over stdio) that
suiban's client uses: initialize / notifications/initialized / tools/list /
tools/call, plus deliberate failure modes for the crash/timeout/error tests.

Tools:
  echo  {text}        -> "echo: <text>"
  add   {a, b}        -> str(a + b)
  sleep {seconds}     -> sleeps, then "slept" (for client-timeout tests)
  fail  {}            -> isError:true result
  crash {}            -> exits the process without responding (crash tests)

Run: python mcp_fixture_server.py
"""

import json
import sys
import time

TOOLS = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "sleep",
        "description": "Sleep for the given number of seconds, then return.",
        "inputSchema": {
            "type": "object",
            "properties": {"seconds": {"type": "number"}},
            "required": ["seconds"],
        },
    },
    {
        "name": "fail",
        "description": "Always returns an isError result.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "crash",
        "description": "Exit the server process immediately (never responds).",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def reply(msg_id, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        message["error"] = error
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def text_result(text, is_error=False):
    result = {"content": [{"type": "text", "text": text}]}
    if is_error:
        result["isError"] = True
    return result


def handle_call(msg_id, params):
    name = params.get("name")
    args = params.get("arguments") or {}
    if name == "echo":
        reply(msg_id, text_result("echo: " + str(args.get("text", ""))))
    elif name == "add":
        reply(msg_id, text_result(str(args.get("a", 0) + args.get("b", 0))))
    elif name == "sleep":
        time.sleep(float(args.get("seconds", 0)))
        reply(msg_id, text_result("slept"))
    elif name == "fail":
        reply(msg_id, text_result("deliberate failure", is_error=True))
    elif name == "crash":
        sys.exit(1)
    else:
        reply(msg_id, error={"code": -32602, "message": f"unknown tool: {name!r}"})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method == "initialize":
            reply(
                msg_id,
                {
                    "protocolVersion": (msg.get("params") or {}).get(
                        "protocolVersion", "2025-06-18"
                    ),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "suiban-mcp-fixture", "version": "0.1.0"},
                },
            )
        elif method == "notifications/initialized":
            pass  # notification: no response
        elif method == "tools/list":
            reply(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            handle_call(msg_id, msg.get("params") or {})
        elif msg_id is not None:
            reply(msg_id, error={"code": -32601, "message": f"method not found: {method!r}"})


if __name__ == "__main__":
    main()
