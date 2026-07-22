"""External provider registry (api.md §11, additive 2026-07-21c).

Polls each ENABLED provider's OpenAI-compatible `{base_url}/v1/models` (Bearer
`api_key` when one is set; short timeout), caches the model list, and marks a failing
provider unreachable — with a `provider_unreachable` notice in /v1/system — instead of
ever crashing or blocking boot. Refreshes happen on boot and after /v1/system/apply
commits.

The honest boundary (docs/architecture.md): suiban never manages a provider's
lifecycle or VRAM. Reachability plus model ids are ALL it knows about them; external
sessions get no VRAM scheduling, no thinking control, and no grammar guarantees.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

from suiban.config import ProviderSettings
from suiban.errors import BonsaiError
from suiban.sched.planner import Notice

logger = logging.getLogger(__name__)

POLL_TIMEOUT_S = 4.0  # reachability probes must never stall boot or apply

# Effort → sampling for external sessions (api.md §1: "effort maps to sampling only").
# External backends get no thinking control and no tool-loop ceiling from us, so the
# effort ladder degrades to a monotone temperature default — used only when the
# request does not set its own temperature; everything else stays on the provider's
# own defaults.
EFFORT_TEMPERATURE: dict[str, float] = {
    "low": 0.2,
    "mid": 0.5,
    "high": 0.7,
    "xhigh": 0.85,
    "max": 1.0,
}


def _default_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=POLL_TIMEOUT_S)


@dataclass
class ProviderState:
    """Cached view of one enabled provider. `api_key` stays internal — write-only in
    settings, used here for Bearer auth only, never serialized anywhere."""

    name: str
    kind: str
    base_url: str
    api_key: str | None
    reachable: bool = False
    models: list[str] = field(default_factory=list)
    error: str | None = None

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}


class ProviderRegistry:
    """Owns the cached provider states. `client_factory` is the injectable transport
    seam (tests hand in httpx.MockTransport clients; nothing here ever needs the
    network in tests) — it is used both for model-list polling and, by the chat
    router, for proxying external sessions."""

    def __init__(self, client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self._client_factory = client_factory
        self._states: dict[str, ProviderState] = {}
        self._tasks: set[asyncio.Task] = set()

    def client(self) -> httpx.AsyncClient:
        factory = self._client_factory or _default_client
        return factory()

    @property
    def states(self) -> list[ProviderState]:
        return list(self._states.values())

    # -- polling -------------------------------------------------------------
    async def refresh(self, providers: list[ProviderSettings]) -> None:
        """Re-poll every ENABLED provider; disabled/removed providers drop out. A
        poll failure keeps the last known model list (entries stay listed with
        `resident: false`) and surfaces a notice — never an exception."""
        new_states: dict[str, ProviderState] = {}
        for settings in providers:
            if not settings.enabled:
                continue
            state = ProviderState(
                name=settings.name,
                kind=settings.kind,
                base_url=settings.base_url.rstrip("/"),
                api_key=settings.api_key,
            )
            previous = self._states.get(settings.name)
            try:
                async with self.client() as client:
                    response = await client.get(
                        f"{state.base_url}/v1/models",
                        headers=state.auth_headers(),
                        timeout=POLL_TIMEOUT_S,
                    )
                response.raise_for_status()
                data = response.json().get("data") or []
                state.models = [
                    item["id"]
                    for item in data
                    if isinstance(item, dict) and isinstance(item.get("id"), str)
                ]
                state.reachable = True
            except Exception as exc:  # noqa: BLE001 - unreachable is a state, not a crash
                logger.warning("external provider %s unreachable: %s", settings.name, exc)
                state.reachable = False
                state.error = str(exc)
                if previous is not None:
                    state.models = list(previous.models)  # last known list survives
            new_states[settings.name] = state
        self._states = new_states

    def refresh_soon(self, providers: list[ProviderSettings]) -> None:
        """Fire-and-forget refresh — deferred-apply commits happen inside synchronous
        idle callbacks, so the poll is scheduled instead of awaited."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self.refresh(providers))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # -- surface -------------------------------------------------------------
    def notices(self) -> list[Notice]:
        return [
            Notice(
                "warn",
                "provider_unreachable",
                f"External provider {state.name!r} is unreachable at {state.base_url} "
                f"({state.error}). Its models stay listed with resident:false; chat "
                "requests to them will fail until it is back.",
            )
            for state in self._states.values()
            if not state.reachable
        ]

    def model_entries(self) -> list[dict]:
        """The GET /v1/models external entries (api.md §2): ids `<provider>/<model>`,
        `bonsai.external` true, role `none`, resident = reachability at last refresh.
        The remaining bonsai keys are kept for shape uniformity but are null — we do
        not know an external model's family/quant/ctx/vision, and pretending would be
        dishonest."""
        entries: list[dict] = []
        for state in self._states.values():
            for model in state.models:
                entries.append(
                    {
                        "id": f"{state.name}/{model}",
                        "object": "model",
                        "owned_by": state.name,
                        "bonsai": {
                            "family": None,
                            "quant": None,
                            "role": "none",
                            "resident": state.reachable,
                            "ctx": None,
                            "vision": None,
                            "downloaded_families": None,
                            "external": True,
                            "provider": state.name,
                        },
                    }
                )
        return entries

    def resolve(self, model_id: str) -> tuple[ProviderState, str]:
        """`<provider>/<model>` → (provider state, bare model id). Unknown provider
        or model → 404 `model_not_found` (api.md §1). A model still in the cache of
        an unreachable provider IS routable — the poll failure may have been
        transient, and the proxy fails honestly if not."""
        name, _, model = model_id.partition("/")
        state = self._states.get(name)
        if state is None:
            raise BonsaiError(
                404,
                f"unknown external model {model_id!r}: no enabled provider {name!r} "
                "(see /v1/settings providers[] and /v1/models)",
                code="model_not_found",
            )
        if model not in state.models:
            detail = (
                "the provider was unreachable at the last refresh"
                if not state.reachable
                else "the provider does not list it"
            )
            raise BonsaiError(
                404,
                f"unknown external model {model_id!r}: {detail} (see /v1/models)",
                code="model_not_found",
            )
        return state, model
