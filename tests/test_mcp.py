"""MCP stdio client + manager (api.md §8, additive 2026-07-21d).

CI runs against the bundled stdlib-only fixture server
(tests/fixtures/mcp_fixture_server.py) — a real subprocess speaking real
newline-delimited JSON-RPC, no network. The same client was verified against the
public @modelcontextprotocol/server-everything and server-filesystem npx servers
(transcripts in the refinement-pass report); those need npx + network, so they are
not CI tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from suiban.config import ConfigManager, McpConnectorSettings, McpServerSettings, Settings
from suiban.mcp.catalog import (
    Connector,
    catalog_view,
    combined_mcp_servers,
    resolve_connectors,
)
from suiban.mcp.client import McpClient, McpError
from suiban.mcp.manager import McpManager
from suiban.memory.service import MemoryService
from suiban.tools.base import ToolContext
from suiban.tools.registry import build_registry

FIXTURE_SERVER = str(Path(__file__).parent / "fixtures" / "mcp_fixture_server.py")
FIXTURE_TOOL_NAMES = {"echo", "add", "sleep", "fail", "crash"}


def fixture_settings(name: str = "fix", *, enabled: bool = True) -> McpServerSettings:
    return McpServerSettings(
        name=name, command=sys.executable, args=[FIXTURE_SERVER], enabled=enabled
    )


def fixture_connector(cid: str = "fix") -> Connector:
    """A catalog entry pointing at the bundled stdio fixture server — a fake connector so
    the catalog enable flow runs with no network/npx in CI."""
    return Connector(
        id=cid,
        name="Fixture",
        description="Bundled test fixture MCP server.",
        command=sys.executable,
        args=[FIXTURE_SERVER],
    )


@pytest.fixture
async def client() -> McpClient:
    client = McpClient("fix", sys.executable, [FIXTURE_SERVER])
    await client.start()
    yield client
    await client.stop()


@pytest.fixture
async def manager() -> McpManager:
    manager = McpManager([fixture_settings()])
    await manager.start()
    yield manager
    await manager.shutdown()


# -- client ------------------------------------------------------------------
async def test_client_initialize_handshake(client: McpClient) -> None:
    assert client.alive
    assert client.protocol_version == "2025-06-18"
    assert client.server_info["name"] == "suiban-mcp-fixture"


async def test_client_lists_tools_with_schemas(client: McpClient) -> None:
    tools = await client.list_tools()
    assert {t["name"] for t in tools} == FIXTURE_TOOL_NAMES
    echo = next(t for t in tools if t["name"] == "echo")
    assert echo["inputSchema"]["required"] == ["text"]  # schema arrives verbatim


async def test_client_calls_a_tool(client: McpClient) -> None:
    text, is_error = await client.call_tool("echo", {"text": "hi"})
    assert (text, is_error) == ("echo: hi", False)
    text, is_error = await client.call_tool("add", {"a": 2, "b": 3})
    assert (text, is_error) == ("5", False)


async def test_client_surfaces_iserror_results(client: McpClient) -> None:
    text, is_error = await client.call_tool("fail", {})
    assert is_error
    assert "deliberate failure" in text


async def test_client_call_timeout_leaves_client_usable(client: McpClient) -> None:
    """A timed-out call raises; the late response for its id is dropped and later
    calls (fresh ids) still work."""
    with pytest.raises(McpError, match="timed out"):
        await client.call_tool("sleep", {"seconds": 1.0}, timeout_s=0.1)
    assert client.alive
    text, is_error = await client.call_tool("echo", {"text": "after"})
    assert (text, is_error) == ("echo: after", False)


async def test_client_detects_server_crash(client: McpClient) -> None:
    with pytest.raises(McpError):
        await client.call_tool("crash", {})
    assert not client.alive
    with pytest.raises(McpError):  # subsequent use fails fast, never hangs
        await client.call_tool("echo", {"text": "x"})


async def test_client_spawn_failure_is_an_mcp_error() -> None:
    client = McpClient("nope", "/nonexistent/definitely-not-a-binary")
    with pytest.raises(McpError, match="spawn"):
        await client.start()


# -- manager -----------------------------------------------------------------
async def test_manager_namespaces_tools_and_passes_schemas_through(
    manager: McpManager,
) -> None:
    names = {t.name for t in manager.tools()}
    assert names == {f"mcp_fix_{n}" for n in FIXTURE_TOOL_NAMES}
    echo = next(t for t in manager.tools() if t.name == "mcp_fix_echo")
    assert echo.parameters["required"] == ["text"]  # inputSchema passthrough
    assert manager.connected == ["fix"]
    assert manager.notices() == []


async def test_manager_tools_run_through_the_registry(manager: McpManager, tmp_path: Path) -> None:
    """The whole path a model call takes: registry -> McpTool -> subprocess."""
    memory = MemoryService(tmp_path / "home")
    memory.startup()
    registry = build_registry("code", "orchestrator", memory=memory, extra_tools=manager.tools())
    assert "mcp_fix_echo" in registry.names
    # Built-ins are intact next to the namespaced MCP tools.
    assert "fs_write" in registry.names

    ctx = ToolContext(session_id="s1", workdir=tmp_path)
    result = await registry.run("mcp_fix_echo", {"text": "over stdio"}, ctx)
    assert result.status == "ok"
    assert result.content == "echo: over stdio"

    # Our subset validator enforces the server's schema before the wire.
    assert registry.validate_args("mcp_fix_echo", {"wrong": 1})

    failed = await registry.run("mcp_fix_fail", {}, ctx)
    assert failed.status == "error"
    assert "deliberate failure" in failed.content
    memory.close()


async def test_manager_server_crash_removes_tools_and_notices(
    manager: McpManager, tmp_path: Path
) -> None:
    ctx = ToolContext(session_id="s1", workdir=tmp_path)
    crash_tool = next(t for t in manager.tools() if t.name == "mcp_fix_crash")
    result = await crash_tool.run({}, ctx)
    assert result.status == "error"  # a tool-shaped failure, never an exception

    assert manager.tools() == []  # the server's tools are gone
    assert manager.connected == []
    codes = [n.code for n in manager.notices()]
    assert codes == ["mcp_server_failed"]

    # Calling into the dead server after the crash stays a graceful error.
    stale = await manager.call("fix", "echo", {"text": "x"})
    assert stale.status == "error"
    assert "not connected" in stale.content


async def test_manager_start_failure_is_a_notice_not_a_crash() -> None:
    manager = McpManager(
        [McpServerSettings(name="broken", command="/nonexistent/mcp-server", enabled=True)]
    )
    await manager.start()  # must not raise
    assert manager.tools() == []
    notices = manager.notices()
    assert len(notices) == 1
    assert notices[0].code == "mcp_server_failed"
    assert "broken" in notices[0].message
    await manager.shutdown()


async def test_manager_skips_disabled_servers() -> None:
    manager = McpManager([fixture_settings(enabled=False)])
    await manager.start()
    assert manager.tools() == []
    assert manager.notices() == []  # disabled is not a failure
    await manager.shutdown()


# -- app lifespan wiring -----------------------------------------------------
def _write_config(bonsai_home: Path, command: str, args: list[str]) -> None:
    bonsai_home.mkdir(parents=True, exist_ok=True)
    args_toml = ", ".join(f'"{a}"' for a in args)
    (bonsai_home / "config.toml").write_text(
        f'[[mcp_servers]]\nname = "fix"\ncommand = "{command}"\n'
        f"args = [{args_toml}]\nenabled = true\n",
        encoding="utf-8",
    )


def test_app_lifespan_starts_mcp_servers(bonsai_home: Path, telemetry_24gb) -> None:
    from fastapi.testclient import TestClient

    from suiban.app import create_app

    _write_config(bonsai_home, sys.executable, [FIXTURE_SERVER])
    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as client:
        current = client.get("/v1/settings").json()["current"]
        assert current["mcp_servers"] == [
            {
                "name": "fix",
                "command": sys.executable,
                "args": [FIXTURE_SERVER],
                "enabled": True,
            }
        ]
        codes = [n["code"] for n in client.get("/v1/system").json()["notices"]]
        assert "mcp_server_failed" not in codes
        state = app.state.bonsai
        assert state.mcp is not None and state.mcp.connected == ["fix"]
        assert {t.name for t in state.mcp.tools()} == {f"mcp_fix_{n}" for n in FIXTURE_TOOL_NAMES}
    # Lifespan exit stopped the subprocess.
    assert state.mcp.tools() == []


def test_app_failed_mcp_server_is_a_notice_never_a_crash(bonsai_home: Path, telemetry_24gb) -> None:
    from fastapi.testclient import TestClient

    from suiban.app import create_app

    _write_config(bonsai_home, "/nonexistent/mcp-server", [])
    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as client:  # boot succeeds
        codes = [n["code"] for n in client.get("/v1/system").json()["notices"]]
        assert "mcp_server_failed" in codes
        # ... and chat serving still works without the server's tools.
        health = client.get("/v1/system/health")
        assert health.status_code == 200


# -- settings ----------------------------------------------------------------
def test_mcp_server_settings_validation() -> None:
    ok = Settings.model_validate(
        {"mcp_servers": [{"name": "my-server", "command": "npx", "args": ["-y", "x"]}]}
    )
    assert ok.mcp_servers[0].enabled is False  # disabled by default

    with pytest.raises(ValueError):  # kebab-case only
        Settings.model_validate({"mcp_servers": [{"name": "My_Server", "command": "x"}]})
    with pytest.raises(ValueError):  # command required non-empty
        Settings.model_validate({"mcp_servers": [{"name": "a", "command": ""}]})
    with pytest.raises(ValueError, match="unique"):
        Settings.model_validate(
            {
                "mcp_servers": [
                    {"name": "dup", "command": "x"},
                    {"name": "dup", "command": "y"},
                ]
            }
        )


def test_mcp_servers_setting_requires_restart(bonsai_home: Path) -> None:
    """api.md 2026-07-21d: mcp_servers is requires_restart — servers start/stop with
    the app lifespan, so /v1/system/apply must report it honestly."""
    manager = ConfigManager(bonsai_home)
    manager.load()
    manager.stage({"mcp_servers": [{"name": "fix", "command": "python", "enabled": False}]})
    requires_restart, pending_idle = manager.pending_effects()
    assert requires_restart == ["mcp_servers"]
    assert pending_idle == []


def test_public_settings_include_mcp_servers(bonsai_home: Path) -> None:
    manager = ConfigManager(bonsai_home)
    settings = manager.load()
    assert settings.public_dict()["mcp_servers"] == []


# -- connector catalog (api.md 2026-07-22c) ----------------------------------
def test_catalog_view_lists_every_connector_with_enabled_flags() -> None:
    view = catalog_view([McpConnectorSettings(id="git", enabled=True)])
    by_id = {c["id"]: c for c in view}
    # The curated well-known servers are all present.
    assert {
        "filesystem",
        "git",
        "fetch",
        "memory",
        "everything",
        "sequential-thinking",
    } <= set(by_id)
    assert by_id["git"]["enabled"] is True  # reflects the selection
    assert by_id["filesystem"]["enabled"] is False
    assert by_id["filesystem"]["requires_path"] is True
    assert by_id["git"]["command"] and isinstance(by_id["git"]["args"], list)


def test_resolve_connectors_enabled_only_and_appends_path(tmp_path: Path) -> None:
    resolved = resolve_connectors(
        [
            McpConnectorSettings(id="filesystem", enabled=True),
            McpConnectorSettings(id="git", enabled=False),
        ],
        default_path=tmp_path,
    )
    assert [s.name for s in resolved] == ["filesystem"]  # disabled git dropped
    assert resolved[0].enabled is True
    assert str(tmp_path) in resolved[0].args  # requires_path connector got the path


def test_resolve_connectors_ignores_unknown_ids() -> None:
    assert resolve_connectors([McpConnectorSettings(id="not-a-real-connector", enabled=True)]) == []


def test_combined_mcp_servers_custom_wins_id_collision() -> None:
    settings = Settings.model_validate(
        {
            "mcp_servers": [{"name": "git", "command": "my-own-git", "enabled": True}],
            "mcp_connectors": [{"id": "git", "enabled": True}],
        }
    )
    combined = combined_mcp_servers(settings)
    gits = [s for s in combined if s.name == "git"]
    assert len(gits) == 1  # the connector does not shadow the user's custom server
    assert gits[0].command == "my-own-git"


async def test_manager_resync_enables_a_connector() -> None:
    """A connector enabled via resync produces a running McpManager server entry —
    without a process restart (api.md 2026-07-22c)."""
    manager = McpManager([])
    await manager.start()
    assert manager.connected == []

    servers = resolve_connectors(
        [McpConnectorSettings(id="fix", enabled=True)], catalog=[fixture_connector()]
    )
    await manager.resync(servers)
    assert manager.connected == ["fix"]
    assert {t.name for t in manager.tools()} == {f"mcp_fix_{n}" for n in FIXTURE_TOOL_NAMES}

    # Disabling it (empty desired set) stops the server and drops its tools.
    await manager.resync([])
    assert manager.connected == []
    assert manager.tools() == []
    await manager.shutdown()


async def test_manager_resync_leaves_unchanged_server_running() -> None:
    """Custom mcp_servers keep working across a resync: an unchanged, still-alive server
    is left running (same client instance), never restarted."""
    manager = McpManager([fixture_settings(name="custom")])
    await manager.start()
    assert manager.connected == ["custom"]
    client_before = manager._servers["custom"].client

    await manager.resync([fixture_settings(name="custom")])
    assert manager.connected == ["custom"]
    assert manager._servers["custom"].client is client_before  # not restarted
    await manager.shutdown()


def test_get_mcp_connectors_endpoint(bonsai_home: Path, telemetry_24gb) -> None:
    # NB: test_mcp.py shadows the conftest `client` fixture with an McpClient fixture, so
    # this HTTP test builds its own app instead of taking `client`.
    from fastapi.testclient import TestClient

    from suiban.app import create_app

    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as http:
        connectors = http.get("/v1/mcp/connectors").json()["connectors"]
    ids = {c["id"] for c in connectors}
    assert {"filesystem", "git", "fetch", "memory", "everything", "sequential-thinking"} <= ids
    assert all(c["enabled"] is False for c in connectors)  # none enabled by default


def test_connector_enable_flow_wires_into_manager(
    bonsai_home: Path, telemetry_24gb, monkeypatch
) -> None:
    """End to end: PATCH mcp_connectors + apply wires the connector into the live
    McpManager (fixture server as a fake catalog entry — no npx/network)."""
    from fastapi.testclient import TestClient

    from suiban.app import create_app

    monkeypatch.setattr("suiban.mcp.catalog.CATALOG", [fixture_connector()])
    app = create_app(
        home=bonsai_home,
        telemetry_provider=telemetry_24gb,
        compute_backend="cuda",
        use_mock=True,
    )
    with TestClient(app) as http:
        before = http.get("/v1/mcp/connectors").json()["connectors"]
        assert [c["id"] for c in before] == ["fix"]
        assert before[0]["enabled"] is False
        assert app.state.bonsai.mcp.connected == []

        http.patch("/v1/settings", json={"mcp_connectors": [{"id": "fix", "enabled": True}]})
        applied = http.post("/v1/system/apply").json()
        assert applied["applied"] is True
        assert "mcp_connectors" in applied["pending_until_idle"]

        after = http.get("/v1/mcp/connectors").json()["connectors"]
        assert after[0]["enabled"] is True
        # The connector is now a running server entry in the live manager.
        assert app.state.bonsai.mcp.connected == ["fix"]
        assert {t.name for t in app.state.bonsai.mcp.tools()} == {
            f"mcp_fix_{n}" for n in FIXTURE_TOOL_NAMES
        }
    # Lifespan exit stopped the connector subprocess.
    assert app.state.bonsai.mcp.tools() == []
