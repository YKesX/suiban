"""~/.bonsai/config.toml loading, validation, and the staged-settings mechanism.

Shape mirrors the `Settings` object of docs/api.md `/v1/settings` exactly. PATCHes are
staged (persisted at ~/.bonsai/staged.toml) and only committed by POST /v1/system/apply.
Secrets (gateways.telegram.token) are write-only: accepted in a patch, stored in the
real config, never echoed — GET only ever shows `token_set`. WhatsApp is QR device-linked
(api.md 2026-07-22b), so it carries no secret at all: only `{enabled, linked, to_number}`.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from suiban import paths
from suiban.errors import BonsaiError

logger = logging.getLogger(__name__)

Effort = Literal["low", "mid", "high", "xhigh", "max"]
QuantFamily = Literal["ternary", "1bit"]
KvPreset = Literal["recommended", "aggressive", "off"]
ProviderKind = Literal["ollama", "openai"]
SearchProviderName = Literal["duckduckgo", "searxng", "brave", "tavily", "serper"]

OLLAMA_DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KvSettings(_Strict):
    turboquant_enabled: bool = True
    preset: KvPreset = "recommended"


class LoadoutSettings(_Strict):
    prefer_workers: int = Field(default=2, ge=0, le=2)
    worker_ctx: int = Field(default=16384, ge=2048)
    orchestrator_ctx: int = Field(default=32768, ge=2048)


class BrowseSettings(_Strict):
    tier2_enabled: bool = True


class ChatSettings(_Strict):
    """Chat-mode behavior toggles (additive 2026-07-22b). `auto_compress` gates the
    in-context rolling compression that fires at ~70% of the slot context
    (memory/compression.py): on by default so long conversations stay in context; turn
    it off to keep the full verbatim history and never fold older turns."""

    auto_compress: bool = True


class RuntimeSettings(_Strict):
    """Lazy/keep-alive model residency (api.md 2026-07-22c, ollama-style). `serve` holds
    no models until the first inference request; the planned loadout warms on demand and
    unloads after `keep_alive` idle.

    keep_alive: "24/7"/"0"/"always" (any case) stays hot forever; a positive minutes
    integer — or a string of minutes — unloads the loadout after that many idle minutes.
    Default "5". Read live by the residency reaper, so a change takes effect at the next
    idle moment after apply (pending_until_idle)."""

    keep_alive: str | int = "5"

    @field_validator("keep_alive")
    @classmethod
    def _keep_alive_not_bool(cls, value: str | int) -> str | int:
        # bool is an int subclass; reject it explicitly (a stray `true` is a config
        # mistake, not "keep alive for 1 minute"). Interpretation of the value lives in
        # llama/load_controller.parse_keep_alive — tolerant on purpose, never crashes.
        if isinstance(value, bool):
            raise ValueError("keep_alive must be a string or an integer, not a boolean")
        return value


class SlapSettings(_Strict):
    """SLAP agent-protocol coordination toggle (api.md 2026-07-22c). On by default: Ultra
    coordinates its sub-agents with SLAP messages and /v1/slap/trace serves the validated
    transcript. Off routes Ultra through the plain structured-dict dispatch path (no SLAP
    messages built or recorded); /v1/slap still serves the schema, but /v1/slap/trace is
    empty. Read per Ultra request, so a change takes effect at the next idle moment after
    apply (pending_until_idle)."""

    enabled: bool = True


class McpConnectorSettings(_Strict):
    """One built-in MCP connector reference (api.md 2026-07-22c). `id` names an entry in
    the built-in connector catalog (distinct from a custom stdio server in `mcp_servers`).
    PATCH replaces the whole list like providers."""

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    enabled: bool = False


class ProviderSettings(_Strict):
    """One external OpenAI-compatible inference provider (api.md §11, additive
    2026-07-21c). `name` prefixes the provider's model ids (`<name>/<model>`), so it
    can never contain `/` or whitespace. `api_key` is a write-only secret exactly
    like the gateway tokens — GET only ever shows `api_key_set`."""

    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    kind: ProviderKind = "openai"
    base_url: str = ""
    enabled: bool = False
    # Write-only secret: accepted via PATCH / config.toml, never serialized publicly.
    api_key: str | None = None

    @model_validator(mode="after")
    def _preset_ollama_base_url(self) -> ProviderSettings:
        """kind "ollama" presets the stock local Ollama endpoint when base_url is
        left empty (api.md §11) — keyless by default, like Ollama itself."""
        if self.kind == "ollama" and not self.base_url:
            self.base_url = OLLAMA_DEFAULT_BASE_URL
        return self


class McpServerSettings(_Strict):
    """One MCP stdio server (api.md §8, additive 2026-07-21d). `name` is kebab-case;
    it namespaces the server's tools as `mcp_<name>_<tool>` for the model. stdio
    transport only in v1: suiban spawns `command args...` as a subprocess and speaks
    JSON-RPC over its stdin/stdout. Changes are requires_restart — servers start and
    stop with the app lifespan, like gateways.

    AUDIT SEAM (security audit, next session): a configured server runs unsandboxed
    with the user's privileges, and its tools execute whatever the server implements
    — config is the trust boundary here. The audit should consider surfacing that
    prominently in clients and/or an allowlist prompt on first tool use.
    """

    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    enabled: bool = False


class SearchSettings(_Strict):
    """Web search for deep research's gather stage (api.md §11, additive
    2026-07-21c). `api_key` is write-only → `api_key_set` in GET."""

    provider: SearchProviderName = "duckduckgo"
    base_url: str = ""  # searxng needs it; the keyed providers have fixed endpoints
    # Write-only secret: accepted via PATCH / config.toml, never serialized publicly.
    api_key: str | None = None


class TelegramSettings(_Strict):
    enabled: bool = False
    # Write-only secret: accepted via PATCH / config.toml, never serialized publicly.
    token: str | None = None
    # Inbound authorization (api.md 2026-07-22 security). Default DENY: only chat ids
    # in `allowed_chat_ids` reach the model. A chat pairs by sending "/pair <code>"
    # with the one-time code printed to the SERVER console at gateway start (never
    # sent over Telegram). require_pairing=false accepts every chat (explicit opt-out,
    # never the default). rate_limit_per_min caps messages per chat per minute
    # (0 disables the limit).
    allowed_chat_ids: list[int] = Field(default_factory=list)
    require_pairing: bool = True
    rate_limit_per_min: int = Field(default=20, ge=0)


class WhatsAppSettings(_Strict):
    """QR device-linked WhatsApp gateway (changed 2026-07-22b): links via the WhatsApp
    Web multi-device protocol (scan a QR), not a Cloud-API token — so there is NO secret
    here. `linked` records whether a device session exists (the session itself lives
    under ~/.bonsai/whatsapp/, never in a repo); `to_number` is the outbound recipient
    for research/scheduled pings. Outbound-only in this version (api.md §8)."""

    enabled: bool = False
    linked: bool = False
    to_number: str = ""


class GatewaysSettings(_Strict):
    telegram: TelegramSettings = TelegramSettings()
    whatsapp: WhatsAppSettings = WhatsAppSettings()


class ServerSettings(_Strict):
    host: str = "127.0.0.1"
    port: int = Field(default=8686, ge=1, le=65535)
    # Reserved (api.md 2026-07-22 security): when true a gateway request MAY use
    # code/ultra. NOT yet honored in v1 — the Telegram relay stays pinned to chat
    # regardless (wiring a remote channel to the shell is the scary path). Default off.
    remote_agentic: bool = False
    # Write-only secret (api.md 2026-07-22 security): auto-generated the first time the
    # server binds to a non-loopback host and applied; then required as an
    # `Authorization: Bearer <token>` header on non-loopback binds. GET /v1/settings
    # only ever shows `server.auth_token_set`. None on loopback binds (no auth —
    # the zero-friction local default is unchanged).
    auth_token: str | None = None


def host_is_loopback(host: str) -> bool:
    """True when binding `host` exposes the server on loopback only (so no HTTP auth is
    required — the unchanged default). `localhost`, 127.0.0.0/8 and ::1 are loopback;
    `0.0.0.0`/`::` (all interfaces), LAN addresses, and names we cannot classify are
    NOT — those require an auth token (api.md 2026-07-22 security)."""
    candidate = host.strip().lower()
    if candidate == "localhost":
        return True
    candidate = candidate.strip("[]")  # tolerate bracketed IPv6 literals
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False  # an unresolvable name is treated as non-loopback (auth on)


class Settings(_Strict):
    quant_family: QuantFamily = "ternary"
    kv: KvSettings = KvSettings()
    dspark_enabled: bool = False
    # Optional override for requests that carry no `effort`: req.effort >
    # effort_default > the mode's default (api.md §1 "default from mode"). None (the
    # default) keeps the per-mode defaults; a set value overrides them ALL — which
    # is why this is nullable rather than presetting "mid" (a preset would silently
    # flatten code/ultra's "high" default for every user).
    effort_default: Effort | None = None
    loadout: LoadoutSettings = LoadoutSettings()
    browse: BrowseSettings = BrowseSettings()
    chat: ChatSettings = ChatSettings()
    # Lazy/keep-alive residency + SLAP toggle (api.md 2026-07-22c).
    runtime: RuntimeSettings = RuntimeSettings()
    slap: SlapSettings = SlapSettings()
    # One disabled local-Ollama entry ships by default (the api.md §8 example shape):
    # the settings UI has something concrete to enable. PATCH replaces the whole list.
    providers: list[ProviderSettings] = Field(
        default_factory=lambda: [ProviderSettings(name="ollama", kind="ollama")]
    )
    search: SearchSettings = SearchSettings()
    # MCP stdio servers (api.md §8, additive 2026-07-21d). Empty by default — users
    # add entries explicitly; PATCH replaces the whole list like providers.
    mcp_servers: list[McpServerSettings] = Field(default_factory=list)
    # Built-in MCP connectors selected from the catalog (api.md 2026-07-22c). Empty by
    # default; PATCH replaces the whole list like providers/mcp_servers.
    mcp_connectors: list[McpConnectorSettings] = Field(default_factory=list)
    gateways: GatewaysSettings = GatewaysSettings()
    server: ServerSettings = ServerSettings()

    @field_validator("providers")
    @classmethod
    def _unique_provider_names(cls, value: list[ProviderSettings]) -> list[ProviderSettings]:
        names = [p.name for p in value]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(f"provider names must be unique; duplicated: {', '.join(duplicated)}")
        return value

    @field_validator("mcp_servers")
    @classmethod
    def _unique_mcp_names(cls, value: list[McpServerSettings]) -> list[McpServerSettings]:
        names = [s.name for s in value]
        duplicated = sorted({n for n in names if names.count(n) > 1})
        if duplicated:
            raise ValueError(
                f"mcp server names must be unique; duplicated: {', '.join(duplicated)}"
            )
        return value

    @field_validator("mcp_connectors")
    @classmethod
    def _unique_connector_ids(cls, value: list[McpConnectorSettings]) -> list[McpConnectorSettings]:
        ids = [c.id for c in value]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise ValueError(
                f"mcp connector ids must be unique; duplicated: {', '.join(duplicated)}"
            )
        return value

    def public_dict(self) -> dict:
        """The /v1/settings `Settings` shape: secrets replaced by `token_set` /
        `api_key_set`. WhatsApp carries no secret (QR-linked, api.md 2026-07-22b) — its
        `{enabled, linked, to_number}` show verbatim."""
        data = self.model_dump()
        tg = data["gateways"]["telegram"]
        token = tg.pop("token", None)
        tg["token_set"] = token is not None
        srv = data["server"]
        auth_token = srv.pop("auth_token", None)
        srv["auth_token_set"] = auth_token is not None
        for provider in data["providers"]:
            api_key = provider.pop("api_key", None)
            provider["api_key_set"] = api_key is not None
        search_key = data["search"].pop("api_key", None)
        data["search"]["api_key_set"] = search_key is not None
        return data


# Which top-level keys need what to take effect (reported by /v1/system/apply).
# gateways is requires_restart: gateways (Telegram, WhatsApp) start/stop with the app
# lifespan only. TODO(v1.1): hot start/stop of gateways at apply time.
# kv + dspark_enabled are requires_restart too (api.md additive 2026-07-21d): they
# map to llama-server LAUNCH flags (--cache-type-k/v, -md) and slots never relaunch
# mid-process — the old "pending_until_idle" label was a lie the report told.
# mcp_servers is requires_restart (api.md 2026-07-21d): server subprocesses start and
# stop with the app lifespan, exactly like gateways.
_REQUIRES_RESTART = {
    "quant_family",
    "kv",
    "dspark_enabled",
    "loadout",
    "server",
    "gateways",
    "mcp_servers",
}
# providers/search commit at idle like the other soft keys: the provider registry
# re-polls model lists right after an apply commits (no restart involved), search is
# read per research job, and effort_default/chat.auto_compress are read per chat request.
# runtime.keep_alive is read live by the residency reaper; slap.enabled is read per Ultra
# request; mcp_connectors is a soft catalog reference — all commit at the next idle moment
# (api.md 2026-07-22c), never a restart.
_PENDING_UNTIL_IDLE = {
    "effort_default",
    "browse",
    "chat",
    "runtime",
    "slap",
    "providers",
    "search",
    "mcp_connectors",
}


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursive dict merge; `patch` wins. Non-dict values replace wholesale."""
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _strip_nones(data: dict) -> dict:
    """Drop None leaves so TOML (which has no null) can serialize the dict. Lists of
    tables (providers) are recursed too."""
    out: dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            out[key] = _strip_nones(value)
        elif isinstance(value, list):
            out[key] = [
                _strip_nones(item) if isinstance(item, dict) else item
                for item in value
                if item is not None
            ]
        elif value is not None:
            out[key] = value
    return out


class ConfigManager:
    """Owns the current Settings + the staged patch. All writes go through here."""

    def __init__(self, home: Path | None = None) -> None:
        self._home = home or paths.bonsai_home()
        self._settings: Settings | None = None
        self._staged: dict | None = None

    # -- paths -------------------------------------------------------------
    @property
    def home(self) -> Path:
        return self._home

    @property
    def config_file(self) -> Path:
        return self._home / "config.toml"

    @property
    def staged_file(self) -> Path:
        return self._home / "staged.toml"

    # -- load / first run --------------------------------------------------
    def _read_toml(self, path: Path) -> dict:
        """Parse a config TOML file, turning a syntax/encoding error into a clean
        BonsaiError instead of a raw TOMLDecodeError/UnicodeDecodeError traceback out
        of `serve`/`doctor` (audit 2026-07-22). A hand-edited config.toml or a stale
        staged.toml can be malformed; name the file and the remedy."""
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise BonsaiError(
                500,
                f"config file {path} is not valid UTF-8 text: {exc}. Re-save it as "
                "UTF-8, or delete it to regenerate defaults.",
                code="config_invalid_toml",
            ) from exc
        try:
            return tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise BonsaiError(
                500,
                f"config file {path} is not valid TOML: {exc}. Fix the TOML syntax, "
                "or delete it to regenerate defaults.",
                code="config_invalid_toml",
            ) from exc

    def load(self) -> Settings:
        """Load config.toml, creating it from packaged defaults on first run."""
        self._home.mkdir(parents=True, exist_ok=True)
        if not self.config_file.exists():
            self._settings = Settings()
            self._write_settings(self._settings)
        else:
            raw = self._read_toml(self.config_file)
            try:
                self._settings = Settings.model_validate(raw)
            except ValidationError as exc:
                raise BonsaiError(
                    500,
                    f"invalid config at {self.config_file}: {exc.errors()[0]['msg']}",
                    code="config_invalid",
                ) from exc
        if self.staged_file.exists():
            self._staged = self._read_toml(self.staged_file)
        else:
            self._staged = None
        return self._settings

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            return self.load()
        return self._settings

    @property
    def staged(self) -> dict | None:
        return deepcopy(self._staged)

    def staged_settings(self) -> Settings | None:
        """Current settings with the staged patch applied (None when nothing staged)."""
        if self._staged is None:
            return None
        merged = deep_merge(self.settings.model_dump(), self._staged)
        return Settings.model_validate(merged)

    # -- staging -----------------------------------------------------------
    def stage(self, patch: dict[str, Any]) -> Settings:
        """Deep-merge a PATCH body into the staged dict; validate; persist. No effect
        on the live settings until apply()."""
        if not isinstance(patch, dict) or not patch:
            raise BonsaiError(400, "PATCH body must be a non-empty JSON object", code="empty_patch")
        staged = deep_merge(self._staged or {}, patch)
        merged = deep_merge(self.settings.model_dump(), staged)
        try:
            preview = Settings.model_validate(merged)
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = ".".join(str(p) for p in err["loc"])
            raise BonsaiError(
                400, f"invalid setting {loc}: {err['msg']}", code="setting_invalid"
            ) from exc
        self._staged = staged
        self.staged_file.parent.mkdir(parents=True, exist_ok=True)
        self.staged_file.write_text(tomli_w.dumps(_strip_nones(staged)), encoding="utf-8")
        return preview

    def pending_effects(self) -> tuple[list[str], list[str]]:
        """(requires_restart, pending_until_idle) for the staged keys — the lists
        /v1/system/apply reports when the commit is deferred to the next idle moment."""
        keys = set(self._staged or {})
        return sorted(keys & _REQUIRES_RESTART), sorted(keys & _PENDING_UNTIL_IDLE)

    def apply(self) -> dict:
        """Commit staged settings to config.toml and clear the stage.

        Returns the /v1/system/apply response body. The caller (HTTP layer) is
        responsible for the "next idle moment" scheduling; in Stage 1 there are no
        active runs, so commit is immediate.
        """
        if self._staged is None:
            return {"applied": False, "requires_restart": [], "pending_until_idle": []}
        changed_keys = set(self._staged.keys())
        new_settings = self.staged_settings()
        assert new_settings is not None
        self._settings = new_settings
        self._write_settings(new_settings)
        self._staged = None
        self.staged_file.unlink(missing_ok=True)
        # First time a non-loopback host is committed, mint + persist the auth token
        # (api.md 2026-07-22 security). server is requires_restart, so auth activates at
        # the next boot; the token is printed here so the user has it before restarting.
        token = self.ensure_server_auth_token()
        if token is not None:
            logger.warning(
                "server.host is now %r (non-loopback): generated an API auth token and "
                "wrote it to %s. Clients must send 'Authorization: Bearer <token>' after "
                "the next restart. Token: %s",
                new_settings.server.host,
                self.config_file,
                token,
            )
        return {
            "applied": True,
            "requires_restart": sorted(changed_keys & _REQUIRES_RESTART),
            "pending_until_idle": sorted(changed_keys & _PENDING_UNTIL_IDLE),
        }

    # -- security helpers --------------------------------------------------
    def ensure_server_auth_token(self, host: str | None = None) -> str | None:
        """Mint + persist an API auth token the first time the server binds to a
        non-loopback host (api.md 2026-07-22 security). Returns the NEW token (so the
        caller can print it to the console once) or None when nothing was generated —
        a loopback bind, or a token already exists. Loopback binds stay tokenless: the
        zero-friction local default is unchanged."""
        settings = self.settings
        effective_host = host if host is not None else settings.server.host
        if host_is_loopback(effective_host) or settings.server.auth_token:
            return None
        token = secrets.token_urlsafe(32)
        settings.server.auth_token = token
        self._write_settings(settings)
        return token

    def add_telegram_chat_id(self, chat_id: int) -> None:
        """Persist a newly-paired Telegram chat id to config.toml immediately. Pairing
        is a runtime gateway action, not a staged setting, so it is written directly;
        the live settings object is updated too, so a restart reads the paired chat."""
        allowed = self.settings.gateways.telegram.allowed_chat_ids
        if chat_id in allowed:
            return
        allowed.append(chat_id)
        self._write_settings(self.settings)

    def set_whatsapp_linked(self, linked: bool) -> None:
        """Persist the WhatsApp linked flag to config.toml immediately. Linking/unlinking
        is a runtime gateway action (scan a QR / POST unlink), not a staged setting, so
        it is written directly — like Telegram pairing — and the live settings object is
        updated so a restart reflects the current link state."""
        if self.settings.gateways.whatsapp.linked == linked:
            return
        self.settings.gateways.whatsapp.linked = linked
        self._write_settings(self.settings)

    # -- internals ---------------------------------------------------------
    def _write_settings(self, settings: Settings) -> None:
        self._home.mkdir(parents=True, exist_ok=True)
        header = (
            "# suiban config — managed by suiban; hand-edits are fine while it is stopped.\n"
            "# Shape documented in config.example.toml (repo) and docs/api.md /v1/settings.\n"
        )
        body = tomli_w.dumps(_strip_nones(settings.model_dump()))
        self.config_file.write_text(header + body, encoding="utf-8")
