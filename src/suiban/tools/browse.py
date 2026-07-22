"""Browse tools.

Tier 1 (`browse_t1`): plain httpx fetch + readability-lxml main-content extraction.
Available to every mode that lists it.

Tier 2 (`browse_t2`): Playwright with a sandboxed profile dir under ~/.bonsai/browser/
— capability-gated on a resident 27B (`/v1/system.capabilities.browse_t2`), and never
given user credentials. The interface is defined here; the actual Playwright driving
is wired in the integration pass (TODO below) — until then the tool reports an honest
error instead of pretending.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from suiban import paths
from suiban.tools.base import Tool, ToolContext, ToolResult

FETCH_TIMEOUT_S = 15.0
FETCH_MAX_BYTES = 2 * 1024 * 1024
EXTRACT_MAX_CHARS = 40_000
MAX_REDIRECTS = 5

_BLOCKED_HOSTNAMES = {"localhost"}

# host -> list of getaddrinfo-shaped tuples; injectable so tests never touch DNS.
Resolver = Callable[[str], Awaitable[list]]


def _scheme_error(url: str) -> str | None:
    scheme = urlsplit(url).scheme
    if scheme not in ("http", "https"):
        return f"only http/https URLs are allowed, got scheme {scheme!r}"
    return None


def _ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Every non-public class an SSRF would target."""
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _default_resolver(host: str) -> list:
    return await asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)


async def _host_allowed(host: str, resolver: Resolver) -> str | None:
    """None if `host` is safe to fetch, else a human-readable refusal. A literal IP is
    checked directly; a hostname is RESOLVED and refused if ANY resolved address is
    loopback/private/link-local/reserved/multicast/unspecified — a public name like
    `metadata.example` may resolve to 127.0.0.1 or 169.254.169.254 (SSRF)."""
    if not host:
        return "URL has no host"
    if host.lower() in _BLOCKED_HOSTNAMES:
        return f"refusing to fetch {host}"
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        if _ip_blocked(literal):
            return f"refusing to fetch private/loopback/reserved address {host}"
        return None
    try:
        infos = await resolver(host)
    except OSError as exc:
        return f"could not resolve host {host!r}: {exc}"
    seen = False
    for info in infos:
        ip_str = info[4][0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        seen = True
        if _ip_blocked(addr):
            return f"refusing to fetch {host}: it resolves to disallowed address {ip_str}"
    if not seen:
        return f"could not resolve host {host!r} to any usable address"
    return None


def extract_readable(html: str, url: str) -> tuple[str, str]:
    """(title, main text) via readability-lxml; degrades to a naive tag-strip if the
    optional extractor is unavailable at runtime — never crashes a browse call."""
    try:
        from readability import Document  # readability-lxml

        doc = Document(html, url=url)
        title = (doc.short_title() or "").strip()
        content_html = doc.summary(html_partial=True)
        import lxml.html

        text = lxml.html.fromstring(content_html).text_content()
    except Exception:  # extractor missing or parse blow-up: degrade, don't die
        import re

        title = ""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return title, text


class BrowseT1Tool(Tool):
    name = "browse_t1"
    description = (
        "Fetch a web page (plain HTTP GET) and return its readable main content. "
        "No JavaScript execution — for JS-heavy pages use browse_t2 if available."
    )
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "http(s) URL to fetch."}},
        "required": ["url"],
        "additionalProperties": False,
    }
    timeout_s = FETCH_TIMEOUT_S + 10.0

    def __init__(self, client_factory: Any | None = None, resolver: Resolver | None = None) -> None:
        # Seams for tests: a factory returning an httpx.AsyncClient (follow_redirects
        # MUST stay False — the host is re-checked on every hop) and a DNS resolver.
        self._client_factory = client_factory or (
            lambda: httpx.AsyncClient(follow_redirects=False, timeout=FETCH_TIMEOUT_S)
        )
        self._resolver = resolver or _default_resolver

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        origin: str = args["url"]
        current = origin
        # Follow redirects MANUALLY (client has follow_redirects=False) so the SSRF
        # host check re-runs on every hop — a public URL that 302s to 127.0.0.1 or
        # 169.254.169.254 is blocked AT the hop, never fetched.
        async with self._client_factory() as client:
            for _hop in range(MAX_REDIRECTS + 1):
                scheme_err = _scheme_error(current)
                if scheme_err:
                    return ToolResult("error", scheme_err)
                blocked = await _host_allowed(urlsplit(current).hostname or "", self._resolver)
                if blocked:
                    return ToolResult("error", blocked)
                try:
                    resp = await client.get(current, headers={"User-Agent": "suiban-browse/0.1"})
                except httpx.HTTPError as exc:
                    return ToolResult("error", f"fetch failed: {exc}")
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        return ToolResult(
                            "error", f"HTTP {resp.status_code} redirect with no Location header"
                        )
                    current = str(httpx.URL(current).join(location))
                    continue
                break
            else:
                return ToolResult(
                    "error", f"too many redirects (> {MAX_REDIRECTS}) starting at {origin}"
                )
        if resp.status_code >= 400:
            return ToolResult("error", f"HTTP {resp.status_code} for {current}")
        body = resp.content[:FETCH_MAX_BYTES]
        content_type = resp.headers.get("content-type", "")
        if "html" in content_type or body[:256].lstrip().startswith((b"<!", b"<html", b"<HTML")):
            decoded = body.decode(resp.encoding or "utf-8", "replace")
            title, text = extract_readable(decoded, current)
        else:
            title, text = "", body.decode("utf-8", errors="replace")
        if len(text) > EXTRACT_MAX_CHARS:
            text = text[:EXTRACT_MAX_CHARS] + f"\n… [truncated at {EXTRACT_MAX_CHARS} chars]"
        header = f"# {title}\n\n" if title else ""
        summary = f"fetched {origin} ({title or 'no title'})"
        return ToolResult("ok", f"{header}{text}", summary=summary)


class BrowseT2Tool(Tool):
    """Tier-2 browsing: real browser via Playwright, sandboxed profile, no credentials.

    Capability-gated on a resident 27B — build_registry() only includes this tool when
    /v1/system.capabilities.browse_t2 is true. The profile dir lives under
    ~/.bonsai/browser/profile and is never the user's own browser profile.

    TODO(v1.1): the Playwright driving itself (navigate, wait, extract, screenshots)
    is wired in the integration pass; playwright is intentionally NOT a suiban
    dependency until then. Until wired, calls return an honest error result.
    """

    name = "browse_t2"
    description = (
        "Fetch a JavaScript-heavy page with a real (sandboxed) browser and return its "
        "readable content. Slower than browse_t1 — use only when browse_t1 fails."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "http(s) URL to open."},
            "wait_ms": {
                "type": "integer",
                "description": "Extra settle time after load, in milliseconds.",
                "minimum": 0,
                "maximum": 10_000,
            },
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    timeout_s = 90.0

    def __init__(self, resolver: Resolver | None = None) -> None:
        self._resolver = resolver or _default_resolver

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        scheme_err = _scheme_error(args["url"])
        if scheme_err:
            return ToolResult("error", scheme_err)
        blocked = await _host_allowed(urlsplit(args["url"]).hostname or "", self._resolver)
        if blocked:
            return ToolResult("error", blocked)
        try:
            import playwright  # noqa: F401
        except ImportError:
            return ToolResult(
                "error",
                "browse_t2 is not available in this build: playwright is not installed "
                "(it is wired in the integration pass). Use browse_t1 instead.",
                summary="browse_t2 unavailable (playwright not installed)",
            )
        # Pinned browser dirs, both under ~/.bonsai/browser/: the profile is a
        # throwaway (never the user's own browser profile, so no cookies/passwords
        # leak in), and downloads are quarantined to a dedicated dir instead of
        # ~/Downloads. Created here so the layout is stable before the driving lands.
        profile_dir = paths.browser_profile_dir()
        profile_dir.mkdir(parents=True, exist_ok=True)
        downloads_dir = paths.browser_downloads_dir()
        downloads_dir.mkdir(parents=True, exist_ok=True)
        # TODO(v1.1): launch chromium with user_data_dir=profile_dir,
        # downloads_path=downloads_dir, accept_downloads=False by default, no
        # credentials, extract via extract_readable() — integration pass.
        # AUDIT SEAM (security audit, next session): when the driving lands, verify the
        # pinned dirs are actually enforced by the launch args (not just created) and
        # that redirect chains re-check _host_allowed() per hop, exactly as browse_t1
        # now does (SSRF fix, audit 2026-07-22).
        return ToolResult(
            "error",
            "browse_t2 driving is not wired yet (integration pass). Use browse_t1.",
            summary="browse_t2 not wired yet",
        )
