"""Pluggable web search for deep research (api.md §11, additive 2026-07-21c)."""

from suiban.search.providers import (
    SearchError,
    SearchProvider,
    SearchResult,
    build_search_provider,
)

__all__ = ["SearchError", "SearchProvider", "SearchResult", "build_search_provider"]
