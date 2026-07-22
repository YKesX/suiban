"""Web search (api.md §11, additive 2026-07-21c): every provider's request shape and
response parse against injected MockTransports, the duckduckgo graceful
empty-on-parse-failure, missing-config errors, the build factory, and the
POST /v1/system/search_test endpoint (never throws)."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from suiban.config import SearchSettings
from suiban.search import SearchError, SearchResult, build_search_provider
from suiban.search.providers import (
    BraveSearch,
    DuckDuckGoSearch,
    SearxngSearch,
    SerperSearch,
    TavilySearch,
    parse_duckduckgo_html,
)


def _settings(**kwargs) -> SearchSettings:
    return SearchSettings.model_validate(kwargs)


def _factory(handler):
    return lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))


# -- duckduckgo ---------------------------------------------------------------
DDG_PAGE = """
<html><body>
<div class="result">
  <a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fbonsai&amp;rut=abc">
     Bonsai <b>care</b> basics</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=x">Water your tree.</a>
</div>
<div class="result">
  <a rel="nofollow" class="result__a" href="https://plain.example/page">Plain link</a>
  <a class="result__snippet" href="#">Second snippet.</a>
</div>
</body></html>
"""


def test_parse_duckduckgo_html_unwraps_uddg_and_strips_tags() -> None:
    results = parse_duckduckgo_html(DDG_PAGE)
    assert results[0] == SearchResult(
        title="Bonsai care basics",
        url="https://example.com/bonsai",
        snippet="Water your tree.",
    )
    assert results[1].url == "https://plain.example/page"
    assert results[1].snippet == "Second snippet."


def test_parse_duckduckgo_garbage_is_empty_never_raises() -> None:
    assert parse_duckduckgo_html("") == []
    assert parse_duckduckgo_html("<html>captcha wall, markup changed</html>") == []


async def test_duckduckgo_search_requests_the_html_endpoint() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=DDG_PAGE)

    provider = DuckDuckGoSearch(_settings(), _factory(handler))
    results = await provider.search("bonsai care", 1)
    assert seen[0].url.host == "html.duckduckgo.com"
    assert seen[0].url.params["q"] == "bonsai care"
    assert len(results) == 1  # count respected


async def test_duckduckgo_transport_failure_is_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline")

    with pytest.raises(SearchError):
        await DuckDuckGoSearch(_settings(), _factory(handler)).search("q", 3)


# -- searxng ------------------------------------------------------------------
async def test_searxng_uses_base_url_and_json_format() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "T1", "url": "https://a.test/1", "content": "C1"},
                    {"title": "T2", "url": "https://a.test/2", "content": "C2"},
                ]
            },
        )

    provider = SearxngSearch(
        _settings(provider="searxng", base_url="http://sx.test/"), _factory(handler)
    )
    results = await provider.search("q", 1)
    assert str(seen[0].url).startswith("http://sx.test/search?")
    assert seen[0].url.params["format"] == "json"
    assert results == [SearchResult(title="T1", url="https://a.test/1", snippet="C1")]


async def test_searxng_without_base_url_is_search_error() -> None:
    with pytest.raises(SearchError) as err:
        await SearxngSearch(_settings(provider="searxng")).search("q", 3)
    assert "base_url" in str(err.value)


# -- brave --------------------------------------------------------------------
async def test_brave_sends_subscription_token_and_parses_web_results() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {"title": "B", "url": "https://b.test", "description": "D"},
                    ]
                }
            },
        )

    provider = BraveSearch(_settings(provider="brave", api_key="brave-key"), _factory(handler))
    results = await provider.search("q", 3)
    assert seen[0].url.host == "api.search.brave.com"
    assert seen[0].headers["X-Subscription-Token"] == "brave-key"
    assert results == [SearchResult(title="B", url="https://b.test", snippet="D")]


async def test_brave_without_api_key_is_search_error() -> None:
    with pytest.raises(SearchError) as err:
        await BraveSearch(_settings(provider="brave")).search("q", 3)
    assert "api_key" in str(err.value)


# -- tavily -------------------------------------------------------------------
async def test_tavily_posts_api_key_in_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"results": [{"title": "T", "url": "https://t.test", "content": "C"}]},
        )

    provider = TavilySearch(_settings(provider="tavily", api_key="tv-key"), _factory(handler))
    results = await provider.search("why", 2)
    body = json.loads(seen[0].content)
    assert seen[0].url.host == "api.tavily.com"
    assert body == {"api_key": "tv-key", "query": "why", "max_results": 2}
    assert results == [SearchResult(title="T", url="https://t.test", snippet="C")]


# -- serper -------------------------------------------------------------------
async def test_serper_posts_with_x_api_key_and_parses_organic() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={"organic": [{"title": "S", "link": "https://s.test", "snippet": "N"}]},
        )

    provider = SerperSearch(_settings(provider="serper", api_key="sp-key"), _factory(handler))
    results = await provider.search("q", 3)
    assert seen[0].url.host == "google.serper.dev"
    assert seen[0].headers["X-API-KEY"] == "sp-key"
    assert json.loads(seen[0].content) == {"q": "q", "num": 3}
    assert results == [SearchResult(title="S", url="https://s.test", snippet="N")]


async def test_http_error_status_is_search_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    with pytest.raises(SearchError):
        await SerperSearch(_settings(provider="serper", api_key="k"), _factory(handler)).search(
            "q", 3
        )


# -- factory ------------------------------------------------------------------
def test_build_search_provider_maps_every_documented_name() -> None:
    for name, cls in (
        ("duckduckgo", DuckDuckGoSearch),
        ("searxng", SearxngSearch),
        ("brave", BraveSearch),
        ("tavily", TavilySearch),
        ("serper", SerperSearch),
    ):
        provider = build_search_provider(_settings(provider=name))
        assert isinstance(provider, cls)
        assert provider.name == name


# -- POST /v1/system/search_test ---------------------------------------------
def test_search_test_ok_with_default_query(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=DDG_PAGE)

    monkeypatch.setattr("suiban.search.providers._default_client", _factory(handler))
    resp = client.post("/v1/system/search_test", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["provider"] == "duckduckgo"
    assert body["error"] is None
    assert 1 <= len(body["results"]) <= 3
    assert set(body["results"][0]) == {"title", "url"}  # never snippets/internals
    assert seen[0].url.params["q"]  # the default query was used when omitted


def test_search_test_honors_the_given_query_and_caps_at_three(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "custom question"
        many = "".join(f'<a class="result__a" href="https://x.test/{i}">R{i}</a>' for i in range(9))
        return httpx.Response(200, text=f"<html>{many}</html>")

    monkeypatch.setattr("suiban.search.providers._default_client", _factory(handler))
    body = client.post("/v1/system/search_test", json={"query": "custom question"}).json()
    assert body["ok"] is True
    assert len(body["results"]) == 3


def test_search_test_reports_failure_honestly_never_throws(client: TestClient) -> None:
    # The autouse offline transport makes duckduckgo unreachable: ok=false + error.
    body = client.post("/v1/system/search_test", json={"query": "x"}).json()
    assert body == {
        "ok": False,
        "provider": "duckduckgo",
        "results": [],
        "error": body["error"],
    }
    assert "failed" in body["error"]


def test_search_test_reports_missing_key_for_keyed_provider(client: TestClient) -> None:
    client.patch("/v1/settings", json={"search": {"provider": "brave"}})
    assert client.post("/v1/system/apply").json()["applied"] is True
    body = client.post("/v1/system/search_test", json={}).json()
    assert body["ok"] is False
    assert body["provider"] == "brave"
    assert "api_key" in body["error"]


def test_search_test_empty_results_is_not_ok(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>markup that parses to nothing</html>")

    monkeypatch.setattr("suiban.search.providers._default_client", _factory(handler))
    body = client.post("/v1/system/search_test", json={}).json()
    assert body["ok"] is False
    assert "no results" in body["error"]


def test_search_test_tolerates_non_json_bodies(client: TestClient) -> None:
    resp = client.post(
        "/v1/system/search_test",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 200  # never throws (api.md §11)
    assert resp.json()["ok"] is False  # offline transport, honest failure
