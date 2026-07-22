"""The deep-research pipeline: plan queries -> search -> gather -> cross-check ->
synthesize.

Runs on the orchestrator slot (deep research is a 27B capability). Stages and their
coarse progress spans (the ONLY thing users ever see):

    planning queries      0 -> 10 %
    collecting sources   10 -> 55 %   (searches, then advances per source fetched)
    cross-checking       55 -> 80 %
    writing report       80 -> 100 %

Gather (additive 2026-07-21c): the plan's sub-questions become web-search queries on
the configured search provider (settings `search`, api.md §11); the top result URLs
are fetched with the tier-1 fetcher (plain HTTP + readability — tier-2 fallback when
available) and become the report's cited sources. When no search seam is wired, or
search fails entirely, the pipeline degrades to the pre-search behavior — the plan's
model-proposed URLs — and (in the failure case) says so in a note at the top of the
report. Queries, URLs, and drafts remain internal: the coarse-progress product rule
is unchanged.

Every stage degrades instead of dying: an unparseable plan falls back to the bare
query, failed searches fall back to planned URLs, failed fetches become "unavailable"
notes, and a run with zero readable sources still produces a report that says exactly
that.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from suiban.agent.structured import request_structured
from suiban.effort import Sampling, thinking_payload_fields
from suiban.search import SearchResult

CompleteFn = Callable[[dict], Awaitable[dict]]
FetchFn = Callable[[str], Awaitable[tuple[bool, str]]]  # (ok, readable text or error)
ProgressFn = Callable[[str, int], Awaitable[None]]
SearchFn = Callable[[str, int], Awaitable[list[SearchResult]]]

STAGE_PLAN = "planning queries"
STAGE_COLLECT = "collecting sources"
STAGE_CROSSCHECK = "cross-checking"
STAGE_WRITE = "writing report"

MAX_SOURCES = 6
MAX_SEARCH_QUERIES = 3  # sub-questions used as search queries
RESULTS_PER_QUERY = 4
NOTE_MAX_CHARS = 8_000  # per-source cap fed to the cross-check/report steps

PLAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "subquestions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"url": {"type": "string", "minLength": 1}},
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subquestions", "sources"],
    "additionalProperties": False,
}


@dataclass
class SourceNote:
    """Internal working state — NEVER serialized into any API response."""

    url: str
    ok: bool
    text: str
    title: str = ""  # from the search result, when the URL came from a search


@dataclass
class ResearchPlan:
    subquestions: list[str]
    source_urls: list[str]


class ResearchEngine:
    """One engine instance per job run, bound to the orchestrator slot's chat seam
    and a fetch seam (both injectable for tests — canned pages, scripted chat)."""

    def __init__(
        self,
        *,
        complete: CompleteFn,
        fetch: FetchFn,
        model: str,
        sampling: Sampling,
        thinking_budget_tokens: int,
        max_sources: int = MAX_SOURCES,
        search: SearchFn | None = None,
        search_provider: str = "",
    ) -> None:
        self._complete = complete
        self._fetch = fetch
        self._model = model
        self._sampling = sampling
        self._thinking = thinking_budget_tokens
        self._max_sources = max_sources
        self._search = search  # None -> pre-search behavior (plan URLs only)
        self._search_provider = search_provider  # name, for the degrade note only

    # -- payload helper ----------------------------------------------------
    def _payload(self, messages: list[dict], *, thinking: int | None = None) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "temperature": self._sampling.temperature,
            "top_p": self._sampling.top_p,
            "top_k": self._sampling.top_k,
            **thinking_payload_fields(self._thinking if thinking is None else thinking),
        }

    async def _text_completion(self, messages: list[dict]) -> str:
        response = await self._complete(self._payload(messages))
        try:
            return response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    # -- stage 1: plan -----------------------------------------------------
    async def _plan(self, query: str, prompt: str) -> ResearchPlan:
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Research question: {query}\n\n"
                    "Stage 1 (Scope) + start of Stage 2: list the sub-questions the "
                    "report must resolve, and the concrete source URLs (primary "
                    "sources preferred) to fetch and read. Only URLs you can defend "
                    f"as likely relevant; at most {self._max_sources}."
                ),
            },
        ]
        result = await request_structured(
            self._complete,
            self._payload(messages, thinking=0),
            PLAN_SCHEMA,
            schema_name="research_plan",
        )
        if result.data is not None:
            return ResearchPlan(
                subquestions=[q for q in result.data["subquestions"] if q.strip()] or [query],
                source_urls=[s["url"] for s in result.data["sources"]][: self._max_sources],
            )
        # Graceful: research the bare query with no pre-planned sources.
        return ResearchPlan(subquestions=[query], source_urls=[])

    # -- stage 2a: search --------------------------------------------------
    async def _select_sources(
        self, query: str, plan: ResearchPlan, progress: ProgressFn
    ) -> tuple[list[tuple[str, str]], str | None]:
        """(url, title) pairs for the gather stage, plus a degrade reason when web
        search could not be used.

        No search seam wired -> the plan's model-proposed URLs, exactly the
        pre-2026-07-21c behavior, no note (that IS the configured behavior). With a
        seam, the sub-questions become search queries; a TOTAL failure (every query
        errored or nothing came back) falls back to the plan URLs and returns the
        honest reason for the report's header note."""
        if self._search is None:
            return [(url, "") for url in plan.source_urls], None
        await progress(STAGE_COLLECT, 10)
        queries = [q for q in plan.subquestions if q.strip()][:MAX_SEARCH_QUERIES] or [query]
        picked: list[tuple[str, str]] = []
        seen: set[str] = set()
        errors: list[str] = []
        for sub in queries:
            try:
                results = await self._search(sub, RESULTS_PER_QUERY)
            except Exception as exc:  # noqa: BLE001 - search failure degrades, never crashes
                errors.append(str(exc))
                continue
            for result in results:
                if result.url and result.url not in seen:
                    seen.add(result.url)
                    picked.append((result.url, result.title))
        if picked:
            return picked[: self._max_sources], None
        reason = "; ".join(dict.fromkeys(errors)) or "search returned no results"
        return [(url, "") for url in plan.source_urls], reason

    def _degrade_note(self, reason: str) -> str:
        provider = f" ({self._search_provider})" if self._search_provider else ""
        return (
            f"> **Note:** web search{provider} was unavailable for this run: {reason}. "
            "Sources fell back to model-proposed URLs, which biases coverage toward "
            "well-known sources.\n\n"
        )

    # -- stage 2b: gather --------------------------------------------------
    async def _gather(
        self, sources: list[tuple[str, str]], progress: ProgressFn
    ) -> list[SourceNote]:
        notes: list[SourceNote] = []
        total = len(sources)
        for i, (url, title) in enumerate(sources):
            await progress(STAGE_COLLECT, 10 + round(45 * i / max(total, 1)))
            try:
                ok, text = await self._fetch(url)
            except Exception as exc:  # noqa: BLE001 - a bad fetch is a note, not a crash
                ok, text = False, f"fetch crashed: {exc!r}"
            notes.append(SourceNote(url=url, ok=ok, text=text[:NOTE_MAX_CHARS], title=title))
        await progress(STAGE_COLLECT, 55)
        return notes

    # -- stage 3: cross-check ----------------------------------------------
    @staticmethod
    def _notes_digest(notes: list[SourceNote]) -> str:
        if not notes:
            return "(no sources were planned)"
        parts = []
        for i, note in enumerate(notes, 1):
            status = "retrieved" if note.ok else "UNAVAILABLE"
            body = note.text if note.ok else f"(not readable: {note.text[:200]})"
            heading = f"--- Source {i} [{status}] {note.url}"
            if note.title:
                heading += f" ({note.title})"
            parts.append(f"{heading}\n{body}")
        return "\n\n".join(parts)

    async def _cross_check(
        self, query: str, plan: ResearchPlan, notes: list[SourceNote], prompt: str
    ) -> str:
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Research question: {query}\n"
                    f"Sub-questions: {'; '.join(plan.subquestions)}\n\n"
                    "Stage 3 (Cross-check). Source material follows. Sort the "
                    "load-bearing claims into corroborated / disputed / single-source "
                    "/ unverifiable, and note the strongest counter-evidence you can "
                    "find in the material. This analysis is internal working state.\n\n"
                    + self._notes_digest(notes)
                ),
            },
        ]
        return await self._text_completion(messages)

    # -- stage 4: synthesize -----------------------------------------------
    async def _write_report(
        self,
        query: str,
        plan: ResearchPlan,
        notes: list[SourceNote],
        cross_check: str,
        prompt: str,
    ) -> str:
        retrieved = sum(1 for n in notes if n.ok)
        coverage = (
            f"{retrieved} of {len(notes)} planned sources were retrieved."
            if notes
            else "No sources could be planned or retrieved for this run."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Research question: {query}\n"
                    f"Sub-questions: {'; '.join(plan.subquestions)}\n"
                    f"Source coverage: {coverage}\n\n"
                    "Stage 4 (Synthesize). Write the final markdown report per your "
                    "instructions: answer first, body by sub-question with inline "
                    "citations to the sources below, then a limitations section. Be "
                    "explicit about anything single-source or unverifiable; if source "
                    "coverage is thin, say so plainly instead of padding.\n\n"
                    "## Source material\n" + self._notes_digest(notes) + "\n\n"
                    "## Cross-check analysis (internal)\n" + (cross_check or "(unavailable)")
                ),
            },
        ]
        report = await self._text_completion(messages)
        if not report.strip():
            # Never persist an empty report for a "completed" job.
            raise RuntimeError("report synthesis produced no text")
        return report

    # -- the pipeline ------------------------------------------------------
    async def run(self, query: str, progress: ProgressFn, *, prompt: str) -> str:
        """Execute the full pipeline; returns the report markdown. `prompt` is the
        deep_research mode system prompt (modes/prompts/research.md)."""
        await progress(STAGE_PLAN, 2)
        plan = await self._plan(query, prompt)
        await progress(STAGE_PLAN, 10)

        sources, degrade_reason = await self._select_sources(query, plan, progress)
        notes = await self._gather(sources, progress)

        await progress(STAGE_CROSSCHECK, 60)
        cross_check = await self._cross_check(query, plan, notes, prompt)
        await progress(STAGE_CROSSCHECK, 80)

        await progress(STAGE_WRITE, 85)
        report = await self._write_report(query, plan, notes, cross_check, prompt)
        await progress(STAGE_WRITE, 100)
        if degrade_reason is not None:
            # Prepended mechanically, not left to the model: the degrade must be
            # visible even if the synthesis ignores it.
            report = self._degrade_note(degrade_reason) + report
        return report
