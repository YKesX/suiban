"""Tool layer: fs jail, shell confirm gate, git_ro read-only enforcement, browse_t1
extraction, schema validation, and registry-level capability gating."""

from __future__ import annotations

import os
import socket
from pathlib import Path

import httpx
import pytest

from suiban.memory.service import MemoryService
from suiban.tools import schema as schema_mod
from suiban.tools.base import ToolContext
from suiban.tools.browse import BrowseT1Tool, BrowseT2Tool
from suiban.tools.fs import FsListTool, FsReadTool, FsUndoTool, FsWriteTool
from suiban.tools.git_ro import GitRoTool
from suiban.tools.registry import build_registry
from suiban.tools.shell import ShellTool, is_destructive


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    workdir = tmp_path / "jail"
    workdir.mkdir()
    return ToolContext(session_id="s1", workdir=workdir, role="orchestrator", mode="code")


@pytest.fixture
def actx(tmp_path: Path) -> ToolContext:
    """Like `ctx` but with auto_confirm=True, so the fs/shell confirm gate is bypassed —
    used by the pure edit-mechanics tests (the gate itself has its own tests below)."""
    workdir = tmp_path / "jail"
    workdir.mkdir()
    return ToolContext(
        session_id="s1", workdir=workdir, role="orchestrator", mode="code", auto_confirm=True
    )


# -- fs jail -----------------------------------------------------------------
async def test_fs_roundtrip_inside_jail(actx: ToolContext) -> None:
    write = await FsWriteTool().run({"path": "sub/a.txt", "content": "hello"}, actx)
    assert write.status == "ok"
    read = await FsReadTool().run({"path": "sub/a.txt"}, actx)
    assert read.status == "ok"
    assert read.content == "hello"
    listing = await FsListTool().run({"path": "sub"}, actx)
    assert "a.txt" in listing.content


@pytest.mark.parametrize("escape", ["../outside.txt", "../../etc/passwd", "a/../../b.txt"])
async def test_fs_cannot_escape_jail_with_traversal(ctx: ToolContext, escape: str) -> None:
    result = await FsWriteTool().run({"path": escape, "content": "x"}, ctx)
    assert result.status == "error"
    assert "escapes" in result.content
    assert not (ctx.workdir.parent / "outside.txt").exists()


async def test_fs_rejects_absolute_path_outside_jail(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    result = await FsReadTool().run({"path": str(outside)}, ctx)
    assert result.status == "error"
    # ... but an absolute path INSIDE the jail is fine.
    inside = ctx.workdir / "ok.txt"
    inside.write_text("fine")
    result = await FsReadTool().run({"path": str(inside)}, ctx)
    assert result.status == "ok"


async def test_fs_symlink_escape_is_caught(ctx: ToolContext, tmp_path: Path) -> None:
    outside = tmp_path / "target.txt"
    outside.write_text("outside data")
    os.symlink(outside, ctx.workdir / "sneaky")
    result = await FsReadTool().run({"path": "sneaky"}, ctx)
    assert result.status == "error"


# -- diff-before-apply + undo journal (docs/architecture.md §3.7) -------------
async def test_fs_write_result_carries_unified_diff(actx: ToolContext) -> None:
    write = await FsWriteTool().run({"path": "a.txt", "content": "one\ntwo\n"}, actx)
    assert write.status == "ok"
    # Creation: diff against the empty before-state, marked as created.
    assert "created a.txt" in write.content
    assert "--- a/a.txt" in write.content
    assert "+++ b/a.txt" in write.content
    assert "+one" in write.content and "+two" in write.content
    assert "created a.txt" in write.summary

    rewrite = await FsWriteTool().run({"path": "a.txt", "content": "one\nTWO\n"}, actx)
    assert rewrite.status == "ok"
    assert "rewrote a.txt" in rewrite.content
    assert "-two" in rewrite.content and "+TWO" in rewrite.content
    assert "(+1 -1)" in rewrite.summary
    assert (actx.workdir / "a.txt").read_text() == "one\nTWO\n"  # diff came, then apply


async def test_fs_undo_restores_last_edit_and_by_index(actx: ToolContext) -> None:
    tool = FsWriteTool()
    await tool.run({"path": "f.txt", "content": "v1\n"}, actx)  # edit #1 (created)
    await tool.run({"path": "f.txt", "content": "v2\n"}, actx)  # edit #2
    await tool.run({"path": "g.txt", "content": "gee\n"}, actx)  # edit #3 (created)

    # Undo the last edit: g.txt was created by it, so undo deletes it.
    undo = await FsUndoTool().run({}, actx)
    assert undo.status == "ok"
    assert "deleted g.txt" in undo.content
    assert not (actx.workdir / "g.txt").exists()

    # Undo by index: edit #2 rewrote f.txt v1->v2; undoing restores v1.
    undo2 = await FsUndoTool().run({"index": 2}, actx)
    assert undo2.status == "ok"
    assert "restored f.txt" in undo2.content
    assert (actx.workdir / "f.txt").read_text() == "v1\n"

    # Entries are consumed: undoing #2 again names the remaining indices.
    again = await FsUndoTool().run({"index": 2}, actx)
    assert again.status == "error"
    assert "no journaled edit #2" in again.content


async def test_fs_undo_with_empty_journal_is_an_error(ctx: ToolContext) -> None:
    result = await FsUndoTool().run({}, ctx)
    assert result.status == "error"
    assert "empty" in result.content


async def test_fs_undo_journal_keeps_last_20_entries(actx: ToolContext) -> None:
    from suiban.tools.fs import MAX_UNDO_ENTRIES, load_undo_entries

    tool = FsWriteTool()
    for i in range(25):
        await tool.run({"path": "x.txt", "content": f"rev {i}\n"}, actx)
    entries = load_undo_entries(actx.workdir)
    assert len(entries) == MAX_UNDO_ENTRIES
    seqs = [e["seq"] for _, e in entries]
    assert seqs == list(range(6, 26))  # oldest pruned, newest kept


async def test_undo_dir_is_hidden_from_listings_and_write_protected(
    actx: ToolContext,
) -> None:
    await FsWriteTool().run({"path": "visible.txt", "content": "x"}, actx)
    assert (actx.workdir / ".suiban-undo").is_dir()  # the journal exists...
    listing = await FsListTool().run({}, actx)
    assert ".suiban-undo" not in listing.content  # ...but never in listings
    assert "visible.txt" in listing.content

    forged = await FsWriteTool().run({"path": ".suiban-undo/0001.json", "content": "{}"}, actx)
    assert forged.status == "error"
    assert "edit journal" in forged.content


# -- fs confirm gate (api.md 2026-07-22b) ------------------------------------
async def test_fs_write_denies_without_token_then_confirms(ctx: ToolContext) -> None:
    tool = FsWriteTool()
    args = {"path": "a.txt", "content": "one\ntwo\n"}

    denied = await tool.run(args, ctx)
    assert denied.status == "denied"
    assert denied.confirm_token is not None
    assert denied.confirm_token in ctx.confirm_tokens
    assert not (ctx.workdir / "a.txt").exists()  # nothing written before confirmation
    # The unified diff rides the denial (content for the model, summary for the client).
    assert "+one" in denied.content and "+two" in denied.content
    assert "created a.txt" in denied.summary

    # A token bound to THIS write cannot approve a different write (and is single-use).
    other = await tool.run(
        {"path": "a.txt", "content": "x\n", "confirm_token": denied.confirm_token}, ctx
    )
    assert other.status == "denied"
    assert not (ctx.workdir / "a.txt").exists()

    # A fresh denial for the original, then confirm with its token → applied.
    denied_again = await tool.run(args, ctx)
    approved = await tool.run({**args, "confirm_token": denied_again.confirm_token}, ctx)
    assert approved.status == "ok"
    assert approved.confirm_token is None
    assert (ctx.workdir / "a.txt").read_text() == "one\ntwo\n"

    # Single use: the same token cannot be replayed.
    replay = await tool.run({**args, "confirm_token": denied_again.confirm_token}, ctx)
    assert replay.status == "denied"


async def test_fs_undo_denies_without_token_then_confirms(actx: ToolContext) -> None:
    # Create an edit with the gate bypassed, then turn the gate ON for the undo.
    await FsWriteTool().run({"path": "f.txt", "content": "v1\n"}, actx)
    await FsWriteTool().run({"path": "f.txt", "content": "v2\n"}, actx)
    actx.auto_confirm = False

    tool = FsUndoTool()
    denied = await tool.run({}, actx)
    assert denied.status == "denied"
    assert denied.confirm_token is not None
    assert (actx.workdir / "f.txt").read_text() == "v2\n"  # not reverted yet
    assert "undo edit #2" in denied.summary
    assert "-v2" in denied.content and "+v1" in denied.content

    approved = await tool.run({"confirm_token": denied.confirm_token}, actx)
    assert approved.status == "ok"
    assert (actx.workdir / "f.txt").read_text() == "v1\n"


async def test_fs_undo_confirm_token_is_bound_to_its_edit(actx: ToolContext) -> None:
    """A confirm_token issued for undoing edit #3 cannot approve undoing edit #4: the
    token is bound to the edit's sequence number, so a mismatched target is refused (and
    the token is consumed on the mismatch, exactly like fs_write's per-op binding)."""
    writer = FsWriteTool()
    for rev in ("a1\n", "a2\n", "a3\n", "a4\n"):  # edits #1..#4 on one file
        await writer.run({"path": "a.txt", "content": rev}, actx)
    actx.auto_confirm = False

    undo = FsUndoTool()
    denied3 = await undo.run({"index": 3}, actx)
    assert denied3.status == "denied"
    assert "undo edit #3" in denied3.summary
    token3 = denied3.confirm_token
    assert token3 is not None

    # The #3 token on an undo of #4 is refused, #4 is NOT reverted, and the token is spent.
    cross = await undo.run({"index": 4, "confirm_token": token3}, actx)
    assert cross.status == "denied"
    assert (actx.workdir / "a.txt").read_text() == "a4\n"  # edit #4 untouched
    assert token3 not in actx.confirm_tokens  # single-use: consumed on the mismatch

    # The mismatch handed back a fresh token bound to #4; it approves undo #4.
    ok4 = await undo.run({"index": 4, "confirm_token": cross.confirm_token}, actx)
    assert ok4.status == "ok"
    assert "undid edit #4" in ok4.content


async def test_auto_confirm_bypasses_the_fs_gate(actx: ToolContext) -> None:
    write = await FsWriteTool().run({"path": "a.txt", "content": "hi\n"}, actx)
    assert write.status == "ok"
    assert write.confirm_token is None
    assert actx.confirm_tokens == {}  # no token is even issued when the gate is bypassed


async def test_auto_confirm_bypasses_the_shell_gate(actx: ToolContext) -> None:
    (actx.workdir / "victim.txt").write_text("data")
    result = await ShellTool().run({"command": "rm victim.txt"}, actx)
    assert result.status == "ok"
    assert not (actx.workdir / "victim.txt").exists()
    assert actx.confirm_tokens == {}


# -- shell confirm gate ------------------------------------------------------
def test_destructive_detection() -> None:
    assert is_destructive("rm -rf build")
    assert is_destructive("git push origin main")
    assert is_destructive("echo hi > file.txt")  # redirection overwrites
    assert is_destructive("sudo apt install x")
    assert not is_destructive("ls -la")
    assert not is_destructive("cat file.txt")
    assert not is_destructive("git status")  # handled read-only; not destructive


def test_destructive_detection_broadened_patterns() -> None:
    """Sandbox light pass: deletion routes that never spell `rm` as a word."""
    # find -delete deletes without rm.
    assert is_destructive("find . -name '*.tmp' -delete")
    assert is_destructive("find /tmp -type f -delete -print")
    # git clean wipes untracked files.
    assert is_destructive("git clean -fd")
    # Interpreter one-liners reaching for deletion primitives.
    assert is_destructive("python3 -c \"import shutil; shutil.rmtree('build')\"")
    assert is_destructive("python -c \"import os; os.remove('x.txt')\"")
    assert is_destructive("perl -e 'unlink glob \"*.log\"'")
    assert is_destructive("node -e \"require('fs').rmSync('x', {recursive: true})\"")
    # Interpreter one-liners spawning subprocesses (whatever they run is invisible).
    assert is_destructive("python -c \"import subprocess; subprocess.run(['ls'])\"")
    assert is_destructive("python3 -c \"import os; os.system('true')\"")
    # ... while plain interpreter one-liners and finds stay benign.
    assert not is_destructive('python -c "print(1+1)"')
    assert not is_destructive("find . -name '*.py'")
    assert not is_destructive("python script.py --flag")


async def test_shell_runs_benign_command(ctx: ToolContext) -> None:
    result = await ShellTool().run({"command": "echo hello"}, ctx)
    assert result.status == "ok"
    assert "exit code 0" in result.content
    assert "hello" in result.content


async def test_shell_denies_destructive_without_token_then_allows(ctx: ToolContext) -> None:
    (ctx.workdir / "victim.txt").write_text("data")
    tool = ShellTool()
    command = "rm victim.txt"

    denied = await tool.run({"command": command}, ctx)
    assert denied.status == "denied"
    assert (ctx.workdir / "victim.txt").exists()  # nothing executed
    assert len(ctx.confirm_tokens) == 1
    token = next(iter(ctx.confirm_tokens))
    # The denial result carries the issued token (api.md tool_result.confirm_token)
    # AND keeps the instructing summary/content for the model.
    assert denied.confirm_token == token
    assert token in denied.content
    assert "confirmation required" in denied.summary

    # The token is bound to the exact command: a different command stays denied.
    other = await tool.run({"command": "rm -rf /", "confirm_token": token}, ctx)
    assert other.status == "denied"

    # Fresh token for the original command, then confirmed execution.
    denied_again = await tool.run({"command": command}, ctx)
    token = next(t for t, c in ctx.confirm_tokens.items() if c == command)
    assert denied_again.status == "denied"
    approved = await tool.run({"command": command, "confirm_token": token}, ctx)
    assert approved.status == "ok"
    assert approved.confirm_token is None  # token is a denial-only field
    assert not (ctx.workdir / "victim.txt").exists()

    # Single use: the same token cannot be replayed.
    replay = await tool.run({"command": command, "confirm_token": token}, ctx)
    assert replay.status == "denied"


async def test_shell_timeout_kills_command(ctx: ToolContext) -> None:
    result = await ShellTool().run({"command": "sleep 5", "timeout_s": 1}, ctx)
    assert result.status == "error"
    assert "timed out" in result.content


def test_destructive_detection_block_device_primitives() -> None:
    """Broadened denylist (audit 2026-07-22): partition/block-device wipers gate."""
    assert is_destructive("wipefs -a /dev/sda")
    assert is_destructive("blkdiscard /dev/nvme0n1")
    assert is_destructive("mkswap /dev/sdb1")
    assert is_destructive("fdisk -l /dev/sda")
    assert is_destructive("parted /dev/sda mklabel gpt")
    # Word-boundary discipline: a longer token is not a match.
    assert not is_destructive("echo wipefsX")


async def test_shell_scrubs_secret_env_vars(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env scrub (audit 2026-07-22): secret-bearing vars in the server env are absent
    from the tool subprocess; ordinary vars still pass through."""
    monkeypatch.setenv("MY_API_TOKEN", "supersecret-xyz")
    monkeypatch.setenv("BONSAI_HIDDEN", "bonsai-secret-abc")
    monkeypatch.setenv("HF_TOKEN", "hf-secret-def")
    monkeypatch.setenv("SAFE_VAR", "keepme-123")
    result = await ShellTool().run({"command": "env"}, ctx)
    assert result.status == "ok"
    assert "supersecret-xyz" not in result.content
    assert "bonsai-secret-abc" not in result.content
    assert "hf-secret-def" not in result.content
    assert "keepme-123" in result.content  # non-secret vars remain available


# -- git read-only -----------------------------------------------------------
async def test_git_ro_status_works_in_a_repo(ctx: ToolContext) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=ctx.workdir, check=True)
    result = await GitRoTool().run({"subcommand": "status"}, ctx)
    assert result.status == "ok"


async def test_git_ro_rejects_mutating_subcommands(ctx: ToolContext) -> None:
    for sub in ("push", "commit", "checkout", "reset", "clean", "add"):
        result = await GitRoTool().run({"subcommand": sub}, ctx)
        assert result.status == "error", sub
        assert "only allows" in result.content
    # Schema-level: mutations are not even in the enum the grammar constrains to.
    tool = GitRoTool()
    assert schema_mod.validate({"subcommand": "push"}, tool.parameters)


async def test_git_ro_rejects_write_capable_flags(ctx: ToolContext) -> None:
    result = await GitRoTool().run({"subcommand": "diff", "args": ["--output=/tmp/evil"]}, ctx)
    assert result.status == "error"
    assert "not allowed" in result.content


async def test_git_ro_does_not_discover_repo_above_the_jail(
    ctx: ToolContext, tmp_path: Path
) -> None:
    """GIT_CEILING_DIRECTORIES pin (audit 2026-07-22): a repo whose .git lives ABOVE the
    jail root is not discovered — git_ro sees 'not a git repository', not the parent."""
    import subprocess

    # ctx.workdir is tmp_path/"jail" (fixture); make its PARENT a repo, jail itself not.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    result = await GitRoTool().run({"subcommand": "status"}, ctx)
    assert result.status == "error"


# -- browse ------------------------------------------------------------------
HTML_PAGE = """<!doctype html><html><head><title>Test Article</title></head>
<body><article><h1>Test Article</h1>
<p>This is the main content paragraph with enough words to be extracted by
readability as the dominant text block of the page, repeated for weight.</p>
<p>This is the main content paragraph with enough words to be extracted by
readability as the dominant text block of the page, repeated for weight.</p>
</article><script>ignore();</script></body></html>"""


def _resolver(ip: str = "93.184.216.34"):
    """Fake DNS resolver (getaddrinfo shape) so browse tests never touch the network."""

    async def resolve(host: str) -> list:
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]

    return resolve


async def test_browse_t1_fetches_and_extracts(ctx: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=HTML_PAGE)

    tool = BrowseT1Tool(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=_resolver(),  # example.org -> a public address
    )
    result = await tool.run({"url": "https://example.org/article"}, ctx)
    assert result.status == "ok"
    assert "Test Article" in result.content
    assert "main content paragraph" in result.content
    assert "ignore();" not in result.content  # scripts stripped


@pytest.mark.parametrize(
    "url", ["http://localhost:8686/x", "http://127.0.0.1/x", "ftp://example.org/f"]
)
async def test_browse_t1_blocks_local_and_nonhttp(ctx: ToolContext, url: str) -> None:
    result = await BrowseT1Tool().run({"url": url}, ctx)
    assert result.status == "error"


# -- SSRF hardening (audit 2026-07-22): resolve the host, reject private ranges, and
# re-check every redirect hop. No real network — a fake resolver + MockTransport.
async def test_browse_t1_blocks_hostname_resolving_to_loopback(ctx: ToolContext) -> None:
    tool = BrowseT1Tool(resolver=_resolver("127.0.0.1"))  # a public NAME -> loopback IP
    result = await tool.run({"url": "https://sneaky.example/x"}, ctx)
    assert result.status == "error"
    assert "disallowed address 127.0.0.1" in result.content


async def test_browse_t1_blocks_redirect_to_metadata_ip(ctx: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, text="SHOULD NEVER BE FETCHED")

    tool = BrowseT1Tool(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=_resolver(),  # public.example -> a public address (first hop allowed)
    )
    result = await tool.run({"url": "http://public.example/start"}, ctx)
    assert result.status == "error"
    assert "169.254.169.254" in result.content
    assert "SHOULD NEVER BE FETCHED" not in result.content


async def test_browse_t1_follows_a_safe_redirect(ctx: ToolContext) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://example.org/final"})
        return httpx.Response(200, html=HTML_PAGE)

    tool = BrowseT1Tool(
        client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        resolver=_resolver(),
    )
    result = await tool.run({"url": "https://example.org/start"}, ctx)
    assert result.status == "ok"
    assert "Test Article" in result.content


async def test_browse_t2_reports_honest_unavailability(ctx: ToolContext) -> None:
    result = await BrowseT2Tool(resolver=_resolver()).run({"url": "https://example.org"}, ctx)
    # playwright is not installed in this environment; the tool must say so, not fake.
    assert result.status == "error"
    assert "browse_t1" in result.content


# -- schema validator --------------------------------------------------------
def test_schema_validator_subset() -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "count": {"type": "integer", "minimum": 0, "maximum": 10},
            "kind": {"type": "string", "enum": ["a", "b"]},
            "items": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name"],
        "additionalProperties": False,
    }
    assert schema_mod.validate({"name": "x", "count": 3, "kind": "a", "items": ["y"]}, schema) == []
    assert schema_mod.validate({}, schema)  # missing required
    assert schema_mod.validate({"name": "x", "junk": 1}, schema)  # additionalProperties
    assert schema_mod.validate({"name": "x", "count": 99}, schema)  # maximum
    assert schema_mod.validate({"name": "x", "kind": "z"}, schema)  # enum
    assert schema_mod.validate({"name": "x", "count": True}, schema)  # bool is not int
    assert schema_mod.validate({"name": "x", "items": [1]}, schema)  # item type


# -- registry gating ---------------------------------------------------------
def _memory(tmp_path: Path) -> MemoryService:
    service = MemoryService(tmp_path / "home")
    service.startup()
    return service


def test_write_tools_only_for_orchestrator_in_chat_and_code(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    for mode in ("chat", "code"):
        orch = build_registry(mode, "orchestrator", memory=memory)
        assert {"memory_write", "skill_save", "skill_improve"} <= set(orch.names), mode
        for role in ("worker", "utility"):
            reg = build_registry(mode, role, memory=memory)
            assert not {"memory_write", "skill_save", "skill_improve"} & set(reg.names), (
                mode,
                role,
            )
    # Ultra dispatch and research NEVER get write tools, even as orchestrator.
    for mode in ("ultra", "deep_research"):
        reg = build_registry(mode, "orchestrator", memory=memory)
        assert not {"memory_write", "skill_save", "skill_improve"} & set(reg.names), mode
    memory.close()


def test_recall_tools_registered_in_both_chat_and_code(tmp_path: Path) -> None:
    """Recall in BOTH chat and code (api.md §11 notes, additive 2026-07-21c):
    memory_search AND session_search are present for every role in those modes."""
    memory = _memory(tmp_path)
    for mode in ("chat", "code"):
        for role in ("orchestrator", "worker", "utility"):
            reg = build_registry(mode, role, memory=memory)
            assert {"memory_search", "session_search"} <= set(reg.names), (mode, role)
    memory.close()


async def test_session_search_tool_returns_snippets_and_session_ids(
    tmp_path: Path, ctx: ToolContext
) -> None:
    memory = _memory(tmp_path)
    memory.store.ensure_session("sess-old", "chat")
    memory.store.add_message("sess-old", "user", "we discussed the zeppelin gondola blueprint")
    memory.store.add_message("sess-old", "assistant", "the gondola needs riveting")
    reg = build_registry("code", "orchestrator", memory=memory)

    result = await reg.run("session_search", {"query": "gondola"}, ctx)
    assert result.status == "ok"
    assert "[session sess-old]" in result.content
    assert "gondola" in result.content
    assert "2 session hits" in result.summary

    empty = await reg.run("session_search", {"query": "quixotic"}, ctx)
    assert empty.status == "ok"
    assert "no session-archive hits" in empty.content
    memory.close()


def test_browse_t2_is_capability_gated(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    without = build_registry("chat", "orchestrator", memory=memory, browse_t2_available=False)
    assert "browse_t2" not in without.names
    with_t2 = build_registry("chat", "orchestrator", memory=memory, browse_t2_available=True)
    assert "browse_t2" in with_t2.names
    memory.close()


async def test_registry_survives_crashing_tool(tmp_path: Path) -> None:
    from suiban.tools.base import Tool, ToolResult
    from suiban.tools.registry import ToolRegistry

    class Bomb(Tool):
        name = "bomb"
        description = "explodes"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        async def run(self, args, ctx) -> ToolResult:
            raise RuntimeError("kaboom")

    registry = ToolRegistry([Bomb()])
    result = await registry.run("bomb", {}, ToolContext(session_id="s", workdir=tmp_path))
    assert result.status == "error"
    assert "kaboom" in result.content
