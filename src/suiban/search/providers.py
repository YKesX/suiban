"""Search providers: one async `search(query, count)` seam, five transports.

Every provider takes an injectable `client_factory` (tests use httpx.MockTransport
clients; no test ever touches the network). Configuration problems (missing
api_key/base_url) and transport failures raise `SearchError` with an honest message —
the callers (the research engine's gather stage, `POST /v1/system/search_test`) catch
and degrade; nothing here can crash a job or the server.

Providers (api.md §11): duckduckgo (keyless default, best-effort HTML scrape) ·
searxng (self-hosted, `base_url` required) · brave · tavily · serper (each with a
write-only `api_key`).
"""

from __future__ import annotations

import html as html_mod
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qs, urlsplit

import httpx

from suiban.config import SearchSettings

logger = logging.getLogger(__name__)

SEARCH_TIMEOUT_S = 10.0


class SearchError(RuntimeError):
    """A search that could not be performed (configuration or transport)."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchProvider(Protocol):
    name: str

    async def search(self, query: str, count: int) -> list[SearchResult]: ...


ClientFactory = Callable[[], httpx.AsyncClient]


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=SEARCH_TIMEOUT_S, follow_redirects=True)


class _HttpSearchProvider:
    name = "unnamed"

    def __init__(
        self, settings: SearchSettings, client_factory: ClientFactory | None = None
    ) -> None:
        self._settings = settings
        self._client_factory = client_factory or _default_client

    def _require_api_key(self) -> str:
        if not self._settings.api_key:
            raise SearchError(
                f"search provider {self.name!r} needs an api_key: PATCH /v1/settings "
                "with search.api_key (write-only) and apply"
            )
        return self._settings.api_key


# -- duckduckgo ---------------------------------------------------------------
_DDG_RESULT_RE = re.compile(
    r'class="result__a"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>', re.DOTALL
)
_DDG_SNIPPET_RE = re.compile(r'class="result__snippet"[^>]*>(?P<snippet>.*?)</a>', re.DOTALL)


def _strip_tags(text: str) -> str:
    return " ".join(html_mod.unescape(re.sub(r"<[^>]+>", " ", text)).split())


def _ddg_href_to_url(href: str) -> str | None:
    """DDG html results link through //duckduckgo.com/l/?uddg=<urlencoded target>;
    unwrap it. Plain http(s) hrefs pass through; anything else (ads, internal links)
    is dropped."""
    if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
        target = parse_qs(urlsplit(href).query).get("uddg", [""])[0]
        return target or None
    if href.startswith("http://") or href.startswith("https://"):
        return href
    return None


def parse_duckduckgo_html(page: str) -> list[SearchResult]:
    """Best-effort regex parse of the html.duckduckgo.com/html markup. Returns [] on
    any parse blow-up — the honest degrade for a scraper with no contract."""
    try:
        anchors = _DDG_RESULT_RE.findall(page)
        snippets = [_strip_tags(s) for s in _DDG_SNIPPET_RE.findall(page)]
        results: list[SearchResult] = []
        for i, (href, raw_title) in enumerate(anchors):
            url = _ddg_href_to_url(html_mod.unescape(href))
            if url is None:
                continue
            results.append(
                SearchResult(
                    title=_strip_tags(raw_title) or url,
                    url=url,
                    snippet=snippets[i] if i < len(snippets) else "",
                )
            )
        return results
    except Exception:  # noqa: BLE001 - a scraper parse failure is empty, never fatal
        logger.warning("duckduckgo html parse failed", exc_info=True)
        return []


class DuckDuckGoSearch(_HttpSearchProvider):
    """Keyless best-effort scrape of the html.duckduckgo.com/html endpoint.

    FRAGILE BY NATURE, and honestly so: this parses DuckDuckGo's non-API HTML with
    regexes. The markup can change — or the endpoint can rate-limit/captcha — at any
    time, and when parsing finds nothing this returns [] rather than guessing; the
    research engine degrades and /v1/system/search_test reports it. The keyed
    providers (brave/tavily/serper) and a self-hosted searxng are the reliable
    paths."""

    name = "duckduckgo"
    ENDPOINT = "https://html.duckduckgo.com/html/"

    async def search(self, query: str, count: int) -> list[SearchResult]:
        try:
            async with self._client_factory() as client:
                response = await client.get(
                    self.ENDPOINT,
                    params={"q": query},
                    headers={"User-Agent": "suiban-search/0.1"},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchError(f"duckduckgo request failed: {exc}") from exc
        return parse_duckduckgo_html(response.text)[:count]


# -- searxng ------------------------------------------------------------------
class SearxngSearch(_HttpSearchProvider):
    """Self-hosted SearXNG instance: `{base_url}/search?q=&format=json`. The
    instance must have its JSON format enabled."""

    name = "searxng"

    async def search(self, query: str, count: int) -> list[SearchResult]:
        base = self._settings.base_url.rstrip("/")
        if not base:
            raise SearchError(
                "search provider 'searxng' needs base_url (your instance's URL): "
                "PATCH /v1/settings with search.base_url and apply"
            )
        try:
            async with self._client_factory() as client:
                response = await client.get(f"{base}/search", params={"q": query, "format": "json"})
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"searxng request failed: {exc}") from exc
        results: list[SearchResult] = []
        for item in data.get("results") or []:
            url = str(item.get("url") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("content") or ""),
                )
            )
        return results[:count]


# -- brave --------------------------------------------------------------------
class BraveSearch(_HttpSearchProvider):
    """Brave Search API: GET api.search.brave.com/res/v1/web/search with the
    X-Subscription-Token header."""

    name = "brave"
    ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

    async def search(self, query: str, count: int) -> list[SearchResult]:
        api_key = self._require_api_key()
        try:
            async with self._client_factory() as client:
                response = await client.get(
                    self.ENDPOINT,
                    params={"q": query, "count": count},
                    headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
                )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"brave request failed: {exc}") from exc
        results: list[SearchResult] = []
        for item in (data.get("web") or {}).get("results") or []:
            url = str(item.get("url") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("description") or ""),
                )
            )
        return results[:count]


# -- tavily -------------------------------------------------------------------
class TavilySearch(_HttpSearchProvider):
    """Tavily: POST api.tavily.com/search with the api_key in the JSON body."""

    name = "tavily"
    ENDPOINT = "https://api.tavily.com/search"

    async def search(self, query: str, count: int) -> list[SearchResult]:
        api_key = self._require_api_key()
        try:
            async with self._client_factory() as client:
                response = await client.post(
                    self.ENDPOINT,
                    json={"api_key": api_key, "query": query, "max_results": count},
                )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"tavily request failed: {exc}") from exc
        results: list[SearchResult] = []
        for item in data.get("results") or []:
            url = str(item.get("url") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("content") or ""),
                )
            )
        return results[:count]


# -- serper -------------------------------------------------------------------
class SerperSearch(_HttpSearchProvider):
    """Serper (Google results): POST google.serper.dev/search with X-API-KEY."""

    name = "serper"
    ENDPOINT = "https://google.serper.dev/search"

    async def search(self, query: str, count: int) -> list[SearchResult]:
        api_key = self._require_api_key()
        try:
            async with self._client_factory() as client:
                response = await client.post(
                    self.ENDPOINT,
                    json={"q": query, "num": count},
                    headers={"X-API-KEY": api_key},
                )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SearchError(f"serper request failed: {exc}") from exc
        results: list[SearchResult] = []
        for item in data.get("organic") or []:
            url = str(item.get("link") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("snippet") or ""),
                )
            )
        return results[:count]


_PROVIDERS: dict[str, type[_HttpSearchProvider]] = {
    "duckduckgo": DuckDuckGoSearch,
    "searxng": SearxngSearch,
    "brave": BraveSearch,
    "tavily": TavilySearch,
    "serper": SerperSearch,
}


def build_search_provider(
    settings: SearchSettings, client_factory: ClientFactory | None = None
) -> SearchProvider:
    """The configured provider instance. Settings validation guarantees the name is
    known; missing keys/base_urls surface at search() time as SearchError."""
    return _PROVIDERS[settings.provider](settings, client_factory)
