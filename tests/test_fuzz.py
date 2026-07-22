"""Property/fuzz tests (audit 2026-07-22, workstream B).

Every parser that eats external input — config TOML, skill frontmatter, the FTS5
query builder, the ChatRequest validator, the MCP JSON-RPC dispatch, and the
tool-call argument repair — is fuzzed with hypothesis and asserted to NEVER raise an
uncaught exception: the only acceptable failures are a clean BonsaiError (or a
documented ValueError / clean return). These repos run on strangers' machines; a
parser that tracebacks on hostile input is a bug, not an edge case.

Deterministic + fast: derandomized, capped example counts, no network, no GPU, no
subprocess. Fixtures are avoided inside @given bodies (or their health check is
suppressed) so hypothesis controls all inputs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from suiban.agent.loop import AgentLoop
from suiban.config import ConfigManager, Settings
from suiban.errors import BonsaiError
from suiban.mcp.client import McpClient
from suiban.memory.skills import parse_frontmatter, validate_skill_markdown
from suiban.memory.store import FTS_PREFIX, FTS_TOKENIZE, fts_query
from suiban.routers.chat import ChatRequest

FAST = settings(max_examples=200, deadline=None, derandomize=True)

# A JSON-ish value strategy: everything a decoded JSON body can contain, plus the
# adversarial extras (control chars, huge ints, NaN-free floats, deep nesting).
_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**63), max_value=2**63),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)
_json_values = st.recursive(
    _json_scalars,
    lambda children: st.one_of(
        st.lists(children, max_size=6),
        st.dictionaries(st.text(max_size=12), children, max_size=6),
    ),
    max_leaves=25,
)
_json_dicts = st.dictionaries(st.text(max_size=16), _json_values, max_size=8)


# -- 1. config.toml loader ----------------------------------------------------
@given(data=st.binary(max_size=512))
@settings(
    max_examples=200, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_config_load_only_raises_bonsai_error(tmp_path: Path, data: bytes) -> None:
    """Arbitrary bytes as config.toml -> load() returns Settings or raises BonsaiError,
    never a raw TOMLDecodeError/UnicodeDecodeError/ValidationError traceback."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.toml").write_bytes(data)
    try:
        result = ConfigManager(home).load()
    except BonsaiError as exc:
        assert exc.code in {"config_invalid_toml", "config_invalid"}
        return
    assert isinstance(result, Settings)


@given(data=st.binary(max_size=512))
@settings(
    max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
def test_staged_toml_only_raises_bonsai_error(tmp_path: Path, data: bytes) -> None:
    """A corrupt staged.toml is caught the same way as config.toml (both go through
    ConfigManager._read_toml)."""
    home = tmp_path / "home2"
    home.mkdir(exist_ok=True)
    staged = home / "staged.toml"
    # hypothesis reuses tmp_path across examples: clear any staged file a prior example
    # left behind so the setup load reads a clean state.
    staged.unlink(missing_ok=True)
    # A valid config.toml so load() reaches the staged read.
    ConfigManager(home).load()
    staged.write_bytes(data)
    try:
        ConfigManager(home).load()
    except BonsaiError as exc:
        assert exc.code == "config_invalid_toml"


# -- 2. skill frontmatter parser ----------------------------------------------
@given(content=st.text(max_size=400))
@FAST
def test_parse_frontmatter_never_crashes(content: str) -> None:
    result = parse_frontmatter(content)
    assert isinstance(result, dict)
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in result.items())


@given(name=st.text(max_size=40), content=st.text(max_size=400))
@FAST
def test_validate_skill_markdown_returns_clean_error_list(name: str, content: str) -> None:
    """The model-write validator never crashes: it returns a list of human-readable
    rejection strings (empty == valid). A hostile skill body is a rejection, not a
    traceback."""
    errors = validate_skill_markdown(name, content)
    assert isinstance(errors, list)
    assert all(isinstance(e, str) for e in errors)


# -- 3. FTS5 query builder ----------------------------------------------------
def _fts_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        f"CREATE VIRTUAL TABLE t USING fts5(content, "
        f"tokenize='{FTS_TOKENIZE}', prefix='{FTS_PREFIX}')"
    )
    conn.execute("INSERT INTO t(content) VALUES ('hello 盆栽 world deployment café')")
    return conn


@given(
    query=st.text(
        alphabet=st.characters(codec="utf-8"),
        max_size=120,
    )
)
@FAST
def test_fts_query_is_always_a_valid_match_expression(query: str) -> None:
    """The built MATCH string never triggers a sqlite error when run against a real
    FTS5 table — the crux is that operators/quotes/parens/星/盆栽/emoji AND control
    bytes (NUL) are neutralized. Regression: a NUL byte used to raise
    OperationalError('unterminated string')."""
    built = fts_query(query)
    conn = _fts_conn()
    try:
        if built:
            conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (built,)).fetchall()
    finally:
        conn.close()


# Explicit adversarial corpus (belt-and-suspenders next to the random search).
@given(
    query=st.sampled_from(
        [
            "\x00",
            "a\x00b",
            "deploy\x00ment",
            "盆栽",
            "星 AND OR NOT NEAR",
            "NEAR(a b)",
            '"quoted"',
            "(paren) *star* :colon ^caret",
            "😀🔥🌸",
            "café",
            "\x1f\x7f\x08",
            "a" * 100,
        ]
    )
)
@settings(max_examples=12, deadline=None)
def test_fts_query_adversarial_corpus(query: str) -> None:
    built = fts_query(query)
    conn = _fts_conn()
    try:
        if built:
            conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (built,)).fetchall()
    finally:
        conn.close()


# -- 4. ChatRequest validator -------------------------------------------------
@given(body=_json_dicts)
@FAST
def test_chat_request_never_500s(body: dict) -> None:
    """Arbitrary JSON-ish body -> a valid ChatRequest or a clean BonsaiError(400),
    never a 500 or an uncaught exception."""
    try:
        req = ChatRequest(body)
    except BonsaiError as exc:
        assert exc.status == 400
        return
    # If it validated, the honored fields are well-typed.
    assert isinstance(req.model, str) and req.model
    assert isinstance(req.messages, list) and req.messages
    assert req.mode in {"chat", "code", "ultra"}
    assert req.effort in {"low", "mid", "high", "xhigh", "max"}


@given(
    effort_default=st.sampled_from([None, "low", "mid", "high", "xhigh", "max"]), body=_json_dicts
)
@settings(max_examples=100, deadline=None, derandomize=True)
def test_chat_request_with_effort_default_never_500s(effort_default, body: dict) -> None:
    try:
        ChatRequest(body, effort_default=effort_default)
    except BonsaiError as exc:
        assert exc.status == 400


# -- 5a. MCP JSON-RPC dispatch (reachable without a live subprocess) ----------
@given(message=_json_dicts)
@FAST
def test_mcp_dispatch_never_crashes(message: dict) -> None:
    """Feeding arbitrary decoded JSON-RPC messages to the dispatcher never raises:
    server-to-client requests try to reply (McpError suppressed, no subprocess),
    unknown/late responses are dropped, notifications ignored."""
    client = McpClient("fuzz", "does-not-run")
    client._dispatch(message)


@given(line=st.binary(max_size=256))
@FAST
def test_mcp_line_json_parse_is_guarded(line: bytes) -> None:
    """The stdout reader parses each line with json.loads inside a try/except; mimic
    that framing decision here to prove arbitrary bytes are tolerated (non-JSON /
    non-dict lines are skipped, dicts dispatch cleanly)."""
    client = McpClient("fuzz", "does-not-run")
    try:
        message = __import__("json").loads(line)
    except ValueError:
        return  # exactly what _read_stdout does: skip the line
    if isinstance(message, dict):
        client._dispatch(message)


# -- 5b. tool-call argument repair --------------------------------------------
@given(
    raw=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=False),
        st.text(max_size=200),
        st.lists(st.text(max_size=8), max_size=5),
        st.dictionaries(st.text(max_size=8), st.text(max_size=8), max_size=5),
    )
)
@FAST
def test_parse_arguments_never_crashes(raw: object) -> None:
    """The loop's tool-call argument parser turns any malformed input into
    ({}, error_string) — never a crash — so the repair path can re-prompt."""
    args, error = AgentLoop._parse_arguments(raw)
    assert isinstance(args, dict)
    assert error is None or isinstance(error, str)
    # A dict input round-trips unchanged; anything else that failed carries an error.
    if isinstance(raw, dict):
        assert args == raw and error is None
