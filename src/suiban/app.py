"""FastAPI application factory. Implements docs/api.md — the frozen v1 contract.

Every response (success or error) carries X-Bonsai-Api-Version: 1; every non-2xx body
is the contract error envelope.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import ClientDisconnect

from suiban import API_VERSION, __version__
from suiban.config import ConfigManager, host_is_loopback
from suiban.errors import STATUS_TO_TYPE, BonsaiError
from suiban.gateways.telegram import (
    TelegramGateway,
    api_send_chat,
    build_gateway,
    research_notification,
)
from suiban.gateways.whatsapp import WhatsAppGateway, build_whatsapp_gateway
from suiban.installer.backend import detect_backend
from suiban.kv import KvState, resolve_kv_state
from suiban.llama import binary as binary_mod
from suiban.llama.backend import mock_enabled
from suiban.llama.load_controller import LoadController
from suiban.llama.manager import LlamaManager
from suiban.mcp.catalog import combined_mcp_servers
from suiban.mcp.manager import McpManager
from suiban.memory import reflection, titling
from suiban.memory.service import MemoryService
from suiban.providers.registry import ProviderRegistry
from suiban.research.jobs import JobManager, JobStore
from suiban.research.wiring import make_run_job
from suiban.sched.budget import BudgetProvider
from suiban.sched.planner import Loadout, Notice, plan_loadout
from suiban.sched.telemetry import TelemetryProvider, TelemetrySnapshot, pick_provider
from suiban.schedules.store import ScheduleStore

if TYPE_CHECKING:
    from suiban.schedules.runner import Scheduler


class ActivityTracker:
    """Counts in-flight chat runs (streams included). /v1/system/apply commits staged
    settings only when this is idle AND no research job is active; a deferred commit
    happens on the transition to idle."""

    def __init__(self) -> None:
        self._count = 0
        self._idle_callbacks: list = []

    @property
    def busy(self) -> bool:
        return self._count > 0

    def add_idle_callback(self, callback) -> None:
        self._idle_callbacks.append(callback)

    def begin(self) -> None:
        self._count += 1

    def end(self) -> None:
        self._count = max(0, self._count - 1)
        if self._count == 0:
            for callback in self._idle_callbacks:
                callback()


@dataclass
class AppState:
    config: ConfigManager
    telemetry_provider: TelemetryProvider
    telemetry: TelemetrySnapshot
    compute_backend: str
    kv: KvState
    loadout: Loadout
    manager: LlamaManager
    budget: BudgetProvider
    memory: MemoryService
    jobs: JobManager
    activity: ActivityTracker
    load: LoadController
    providers: ProviderRegistry
    mcp: McpManager | None = None
    gateway: TelegramGateway | None = None
    whatsapp: WhatsAppGateway | None = None
    scheduler: Scheduler | None = None
    apply_pending: bool = False
    extra_notices: list[Notice] = field(default_factory=list)
    # The host the ASGI server is actually bound to (config host, or a CLI --host
    # override). Drives non-loopback auth (api.md 2026-07-22 security).
    bind_host: str = "127.0.0.1"
    started_at: float = field(default_factory=time.monotonic)

    @property
    def uptime_s(self) -> int:
        return int(time.monotonic() - self.started_at)

    @property
    def auth_required(self) -> bool:
        """Bearer-token auth is required exactly when the bind host is non-loopback."""
        return not host_is_loopback(self.bind_host)

    def notices(self) -> list[Notice]:
        out = list(self.loadout.notices)
        if self.kv.notice is not None:
            out.append(self.kv.notice)
        out.extend(self.manager.notices)
        out.extend(self.providers.notices())
        if self.mcp is not None:
            out.extend(self.mcp.notices())
        out.extend(self.extra_notices)
        return out

    def is_idle(self) -> bool:
        """Idle = no active chat runs/streams and no queued/running jobs. This is
        when staged settings may commit (api.md: 'next idle moment, never mid-run')."""
        return not self.activity.busy and self.jobs.active == 0

    def maybe_commit_staged(self) -> None:
        """Deferred /v1/system/apply: commit once the system reaches idle. Provider
        model lists re-poll after the commit (api.md §11) — scheduled, since idle
        callbacks are synchronous."""
        if self.apply_pending and self.is_idle() and self.config.staged is not None:
            self.config.apply()
            self.apply_pending = False
            self.providers.refresh_soon(self.config.settings.providers)
            # Catalog connectors commit at idle without a restart (api.md 2026-07-22c):
            # re-sync the MCP manager to the new mcp_servers + enabled connectors.
            if self.mcp is not None:
                self.mcp.resync_soon(combined_mcp_servers(self.config.settings))

    def notify_gateways(self, kind: str, title: str, summary: str) -> None:
        """The generalized gateway notification hook: research-job completions and
        scheduled runs both fire it; configured gateways fan the ping out."""
        sent = 0
        if self.gateway is not None:
            self.gateway.notify(kind, title, summary)
            sent += 1
        if self.whatsapp is not None:
            self.whatsapp.notify(kind, title, summary)
            sent += 1
        # Observable even with zero gateways: users debugging a missing ping (and the
        # demo path) can see the hook fired and how many gateways it reached.
        logging.getLogger(__name__).info(
            "gateway notify (%s): %r -> %d gateway(s)%s",
            kind,
            title,
            sent,
            "" if sent else " (none configured)",
        )


def _resolve_backend_supported(compute_backend: str, use_mock: bool) -> bool:
    """TurboQuant kernel availability. Mock backend pretends support (client dev needs
    the happy path); a real binary supports TQ only after `suiban install turboquant`
    wrote the TURBOQUANT marker."""
    if use_mock:
        return True
    return binary_mod.turboquant_installed(compute_backend)


def create_app(
    *,
    home: Path | None = None,
    telemetry_provider: TelemetryProvider | None = None,
    compute_backend: str | None = None,
    use_mock: bool | None = None,
    bind_host: str | None = None,
) -> FastAPI:
    """Build the suiban app. Keyword overrides exist for tests (fake telemetry, forced
    backend, mock llama) — production callers pass nothing. `bind_host` is the address
    the ASGI server will actually bind to (CLI passes `host or settings.server.host`);
    it drives non-loopback auth so a `serve --host 0.0.0.0` override cannot silently
    bypass the token gate."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mock = mock_enabled() if use_mock is None else use_mock
        config = ConfigManager(home)
        settings = config.load()
        # Effective bind host drives non-loopback auth (api.md 2026-07-22 security).
        effective_host = bind_host if bind_host is not None else settings.server.host
        new_token = config.ensure_server_auth_token(effective_host)
        if new_token is not None:
            logging.getLogger(__name__).warning(
                "suiban is bound to non-loopback host %r: API auth is REQUIRED. Send "
                "'Authorization: Bearer %s' on every request except GET "
                "/v1/system/health (token stored in %s).",
                effective_host,
                new_token,
                config.config_file,
            )

        provider = telemetry_provider
        if provider is None:
            forced = os.environ.get("SUIBAN_TELEMETRY")
            if forced:
                from suiban.sched import telemetry as tm

                by_name = {c.name: c for c in tm.DEFAULT_PROVIDERS}
                provider = by_name.get(forced, tm.RamProvider)()
            else:
                provider = pick_provider()
        snapshot = provider.snapshot()

        backend = compute_backend or detect_backend()
        kv = resolve_kv_state(
            settings.kv,
            backend_supported=_resolve_backend_supported(backend, mock),
            # TODO(v1.1): probe the pinned binary for FA support instead of assuming it;
            # every recon'd backend build ships FA, so True is the honest default.
            fa_available=True,
        )
        budget = BudgetProvider()
        loadout = plan_loadout(
            snapshot,
            settings,
            budget,
            backend=backend,
            k_type=kv.k_type,
            v_type=kv.v_type,
        )

        def used_vram_mb(gpu_index: int) -> int | None:
            snap = provider.snapshot()
            if snap.gpus is None:
                return None
            for gpu in snap.gpus:
                if gpu.index == gpu_index:
                    return gpu.vram_used_mb
            return None

        manager = LlamaManager(
            loadout,
            kv,
            compute_backend=backend,
            use_mock=mock,
            budget=budget,
            used_vram_mb=used_vram_mb,
        )

        # -- lazy / keep-alive residency (api.md 2026-07-22c) ------------------
        # NO models are started at boot: serve comes up healthy with zero VRAM used.
        # The LoadController warms the planned loadout on the first inference request
        # and unloads it after runtime.keep_alive idle, but never mid-generation.
        # `is_busy` mirrors AppState.is_idle() (in-flight chats/streams OR a running
        # research job) — activity and jobs are bound below in this same scope, so the
        # closure reads them live at request/reaper time.
        activity = ActivityTracker()

        def _residency_busy() -> bool:
            return activity.busy or jobs.active > 0

        load = LoadController(
            manager,
            is_busy=_residency_busy,
            keep_alive=lambda: config.settings.runtime.keep_alive,
        )

        memory = MemoryService(config.home)
        memory.startup()
        reflection.reset()  # per-session reflection counters are per-process

        # -- deep research jobs (same SQLite db as memory; api.md §3) ----------
        job_store = JobStore(config.home / "memory" / "memory.sqlite")
        jobs = JobManager(
            job_store,
            run_job=make_run_job(
                manager,
                config.home,
                loadout.capabilities(settings),
                search_settings=lambda: config.settings.search,
                # A research job auto-warms the loadout before it needs the orchestrator
                # (lazy residency, api.md 2026-07-22c).
                ensure_loaded=load.ensure_loaded,
            ),
            reports_dir=config.home / "reports",
        )
        jobs.startup()

        extra_notices: list[Notice] = []

        # -- external providers (api.md §11): poll enabled model lists ---------
        providers = ProviderRegistry()
        await providers.refresh(settings.providers)

        # -- MCP stdio servers + connectors (api.md §8 + 2026-07-22c) ----------
        # Custom mcp_servers (requires_restart) start with the lifespan like
        # gateways; enabled catalog connectors (mcp_connectors, pending_until_idle)
        # are resolved into the same manager and re-synced on apply below. A failed
        # server is an mcp_server_failed notice, never a crash — chat serving
        # continues without its tools.
        mcp = McpManager(combined_mcp_servers(settings))
        await mcp.start()

        # -- telegram gateway (long-poll; optional extra; token from config) ---
        gateway = build_gateway(
            settings,
            send_chat=api_send_chat(f"http://127.0.0.1:{settings.server.port}"),
            notices=extra_notices,
            persist_chat_id=config.add_telegram_chat_id,
        )
        if gateway is not None:
            try:
                await gateway.start()
            except Exception as exc:  # noqa: BLE001 - a dead gateway must not kill serve
                extra_notices.append(
                    Notice(
                        "warn",
                        "telegram_start_failed",
                        f"Telegram gateway failed to start: {exc}. Chat serving continues.",
                    )
                )
                gateway = None

        # -- whatsapp gateway (QR device-linking + outbound pings, api.md 2026-07-22b) --
        # The linked-device session lives under ~/.bonsai/whatsapp/, never a repo.
        whatsapp = build_whatsapp_gateway(
            settings,
            notices=extra_notices,
            state_dir=config.home / "whatsapp",
            persist_linked=config.set_whatsapp_linked,
        )
        if whatsapp is not None:
            await whatsapp.start()

        state = AppState(
            config=config,
            telemetry_provider=provider,
            telemetry=snapshot,
            compute_backend=backend,
            kv=kv,
            loadout=loadout,
            manager=manager,
            budget=budget,
            memory=memory,
            jobs=jobs,
            activity=activity,
            load=load,
            providers=providers,
            mcp=mcp,
            gateway=gateway,
            whatsapp=whatsapp,
            extra_notices=extra_notices,
            bind_host=effective_host,
        )

        # Idle-apply + research-complete pings (via the generalized notify hook).
        activity.add_idle_callback(state.maybe_commit_staged)
        # Reset the keep-alive idle timer whenever activity drains, so it measures idle
        # time from when work ENDED, not when it started (api.md 2026-07-22c).
        activity.add_idle_callback(load.touch)
        # Background idle reaper: unloads the loadout after runtime.keep_alive idle.
        load.start()

        def on_job_terminal(job) -> None:
            state.maybe_commit_staged()
            state.notify_gateways("research", *research_notification(job))

        jobs.add_listener(on_job_terminal)

        # -- scheduled runs (same SQLite db as memory; api.md §10) -------------
        # Deferred import: schedules.runner reaches into routers.chat, which
        # imports this module.
        from suiban.schedules.runner import Scheduler, make_run_schedule

        schedule_store = ScheduleStore(config.home / "memory" / "memory.sqlite")
        scheduler = Scheduler(
            schedule_store,
            run_schedule=make_run_schedule(state),
            notify=state.notify_gateways,
        )
        state.scheduler = scheduler
        scheduler.start()

        app.state.bonsai = state
        try:
            yield
        finally:
            await scheduler.shutdown()
            await titling.cancel_pending()
            await reflection.cancel_pending()
            if state.gateway is not None:
                await state.gateway.stop()
            if state.whatsapp is not None:
                await state.whatsapp.stop()
            if state.mcp is not None:
                await state.mcp.shutdown()
            await jobs.shutdown()
            await load.stop()
            await manager.shutdown()
            schedule_store.close()
            job_store.close()
            memory.close()

    app = FastAPI(
        title="suiban",
        version=__version__,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # -- CORS for the first-party clients ------------------------------------
    # dai runs in a browser (vite dev, localhost:5173) or a Tauri webview
    # (tauri://localhost on mac/linux, http://tauri.localhost on windows); both
    # enforce CORS on cross-origin fetches to 127.0.0.1:8686. The server binds
    # loopback-only, so this allows local UIs, not the network.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:1420",  # tauri dev port (dai vite.config)
            "http://127.0.0.1:1420",
            "http://localhost:5173",  # stock vite default, kept for tooling
            "http://127.0.0.1:5173",
            "tauri://localhost",
            "http://tauri.localhost",
        ],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        # X-Bonsai-Client is the client-identity header every first-party UI sends
        # (dai/sentei, additive 2026-07-22b). A browser marks it non-safelisted, so
        # it must be allowed here or every cross-origin request fails its preflight.
        allow_headers=["Content-Type", "Authorization", "X-Bonsai-Client"],
        expose_headers=["X-Bonsai-Api-Version"],
    )

    # -- X-Bonsai-Api-Version on EVERY response ------------------------------
    @app.middleware("http")
    async def api_version_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Bonsai-Api-Version"] = API_VERSION
        return response

    # -- non-loopback auth (api.md 2026-07-22 security) ----------------------
    # Loopback binds stay open (unchanged zero-friction default). When bound to a
    # non-loopback host, every route EXCEPT GET /v1/system/health requires
    # `Authorization: Bearer <token>` (the auto-generated server.auth_token); a
    # mismatch is a 401 `unauthorized` envelope. OPTIONS preflights pass through so
    # CORS still works for first-party clients.
    @app.middleware("http")
    async def require_auth(request: Request, call_next):
        state = getattr(request.app.state, "bonsai", None)
        if (
            state is not None
            and state.auth_required
            and request.method != "OPTIONS"
            and not (request.method == "GET" and request.url.path == "/v1/system/health")
        ):
            token = state.config.settings.server.auth_token
            provided = _bearer_token(request.headers.get("authorization"))
            if not token or provided is None or not secrets.compare_digest(provided, token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "type": "invalid_request_error",
                            "message": "authorization required: this server is bound to a "
                            "non-loopback address; send 'Authorization: Bearer <token>' "
                            "(the token printed to the server console / stored in "
                            "~/.bonsai/config.toml)",
                            "code": "unauthorized",
                        }
                    },
                    headers={"X-Bonsai-Api-Version": API_VERSION},
                )
        return await call_next(request)

    # -- error envelope -------------------------------------------------------
    def _envelope(status: int, error_type: str, message: str, code: str | None) -> JSONResponse:
        # Header set here too: the bare-Exception handler runs outside the header
        # middleware, and the contract promises the header on EVERY response.
        return JSONResponse(
            status_code=status,
            content={"error": {"type": error_type, "message": message, "code": code}},
            headers={"X-Bonsai-Api-Version": API_VERSION},
        )

    @app.exception_handler(BonsaiError)
    async def bonsai_error_handler(request: Request, exc: BonsaiError):
        return _envelope(exc.status, exc.error_type, exc.message, exc.code)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        error_type = STATUS_TO_TYPE.get(exc.status_code, "invalid_request_error")
        code = "method_not_allowed" if exc.status_code == 405 else None
        return _envelope(exc.status_code, error_type, str(exc.detail), code)

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", []))
        message = f"invalid request: {loc}: {first.get('msg', 'validation failed')}"
        return _envelope(400, "invalid_request_error", message, "validation_error")

    @app.exception_handler(ClientDisconnect)
    async def client_disconnect_handler(request: Request, exc: ClientDisconnect):
        # Raised by the chat disconnect watcher after it cancelled the backend run:
        # the client is gone, so nobody reads this. An empty 204 (instead of falling
        # through to the bare-Exception handler) keeps logs free of phantom 500s.
        return Response(status_code=204, headers={"X-Bonsai-Api-Version": API_VERSION})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        return _envelope(500, "server_error", "internal server error", None)

    # -- routers --------------------------------------------------------------
    from suiban.research import router as research_router
    from suiban.routers import (
        chat,
        import_router,
        mcp_router,
        memory_router,
        models,
        modes_router,
        projects_router,
        settings_router,
        skills_router,
        slap_router,
        system,
        whatsapp_router,
    )
    from suiban.schedules import router as schedules_router

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(research_router.router)
    app.include_router(system.router)
    app.include_router(settings_router.router)
    app.include_router(memory_router.router)
    app.include_router(import_router.router)
    app.include_router(skills_router.router)
    app.include_router(mcp_router.router)
    app.include_router(modes_router.router)
    app.include_router(projects_router.router)
    app.include_router(schedules_router.router)
    app.include_router(slap_router.router)
    app.include_router(whatsapp_router.router)
    return app


def _bearer_token(header: str | None) -> str | None:
    """Extract the token from an `Authorization: Bearer <token>` header, or None."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    if scheme.lower() != "bearer" or not value.strip():
        return None
    return value.strip()


def state_of(request: Request) -> AppState:
    return request.app.state.bonsai
