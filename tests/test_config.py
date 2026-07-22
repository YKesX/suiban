"""Config load / first-run creation / staging / apply / secrets."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from suiban.config import ConfigManager, Settings, deep_merge
from suiban.errors import BonsaiError


def manager(home: Path) -> ConfigManager:
    return ConfigManager(home)


def test_first_run_creates_config(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    settings = cfg.load()
    assert cfg.config_file.exists()
    assert settings.quant_family == "ternary"
    assert settings.server.port == 8686
    # Round-trips through TOML
    raw = tomllib.loads(cfg.config_file.read_text())
    assert Settings.model_validate(raw) == settings


def test_public_dict_matches_contract_shape(bonsai_home: Path) -> None:
    public = manager(bonsai_home).load().public_dict()
    assert set(public) == {
        "quant_family",
        "kv",
        "dspark_enabled",
        "effort_default",
        "loadout",
        "browse",
        "chat",  # additive 2026-07-22b: chat.auto_compress
        "runtime",  # additive 2026-07-22c: lazy/keep-alive residency
        "slap",  # additive 2026-07-22c: SLAP toggle
        "providers",
        "search",
        "mcp_servers",  # additive 2026-07-21d: MCP stdio servers
        "mcp_connectors",  # additive 2026-07-22c: built-in MCP connector catalog refs
        "gateways",
        "server",
    }
    assert public["chat"] == {"auto_compress": True}
    # Lazy/keep-alive residency defaults (api.md 2026-07-22c).
    assert public["runtime"] == {"keep_alive": "5"}
    assert public["slap"] == {"enabled": True}
    assert public["mcp_connectors"] == []
    assert public["kv"] == {"turboquant_enabled": True, "preset": "recommended"}
    assert public["loadout"] == {
        "prefer_workers": 2,
        "worker_ctx": 16384,
        "orchestrator_ctx": 32768,
    }
    # The default provider list ships one disabled local-Ollama entry (api.md §8
    # example shape) with the ollama base_url preset filled in.
    assert public["providers"] == [
        {
            "name": "ollama",
            "kind": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "enabled": False,
            "api_key_set": False,
        }
    ]
    assert public["search"] == {"provider": "duckduckgo", "base_url": "", "api_key_set": False}
    # Telegram gains the inbound-authorization fields (api.md 2026-07-22 security);
    # allowed_chat_ids/require_pairing/rate_limit_per_min are not secrets — they show.
    assert public["gateways"]["telegram"] == {
        "enabled": False,
        "token_set": False,
        "allowed_chat_ids": [],
        "require_pairing": True,
        "rate_limit_per_min": 20,
    }
    # server exposes auth_token_set (write-only token) + remote_agentic, never the token.
    assert public["server"] == {
        "host": "127.0.0.1",
        "port": 8686,
        "remote_agentic": False,
        "auth_token_set": False,
    }
    # WhatsApp is QR-linked (api.md 2026-07-22b): no secret at all, just link state.
    assert public["gateways"]["whatsapp"] == {
        "enabled": False,
        "linked": False,
        "to_number": "",
    }


def test_stage_is_persisted_and_not_applied(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"kv": {"preset": "aggressive"}})
    # live settings unchanged
    assert cfg.settings.kv.preset == "recommended"
    assert cfg.staged_settings().kv.preset == "aggressive"
    # staged survives a fresh manager (separate persistence)
    cfg2 = manager(bonsai_home)
    cfg2.load()
    assert cfg2.staged == {"kv": {"preset": "aggressive"}}


def test_stage_deep_merges_successive_patches(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"kv": {"preset": "aggressive"}})
    cfg.stage({"kv": {"turboquant_enabled": False}, "effort_default": "high"})
    staged = cfg.staged_settings()
    assert staged.kv.preset == "aggressive"
    assert staged.kv.turboquant_enabled is False
    assert staged.effort_default == "high"


def test_apply_commits_and_clears_stage(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage(
        {"quant_family": "1bit", "kv": {"preset": "aggressive"}, "browse": {"tier2_enabled": True}}
    )
    result = cfg.apply()
    assert result["applied"] is True
    # kv is requires_restart (api.md 2026-07-21d): it maps to llama-server launch
    # flags and slots never relaunch mid-process.
    assert result["requires_restart"] == ["kv", "quant_family"]
    assert result["pending_until_idle"] == ["browse"]
    assert cfg.settings.quant_family == "1bit"
    assert cfg.staged is None
    assert not cfg.staged_file.exists()
    # committed to disk
    assert manager(bonsai_home).load().quant_family == "1bit"


def test_apply_with_nothing_staged(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    assert cfg.apply() == {"applied": False, "requires_restart": [], "pending_until_idle": []}


def test_invalid_patch_rejected(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    with pytest.raises(BonsaiError) as exc_info:
        cfg.stage({"quant_family": "2bit"})
    assert exc_info.value.status == 400
    assert cfg.staged is None  # nothing partially staged


def test_secret_token_write_only(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"gateways": {"telegram": {"enabled": True, "token": "123:abc"}}})
    staged_public = cfg.staged_settings().public_dict()
    assert staged_public["gateways"]["telegram"] == {
        "enabled": True,
        "token_set": True,
        "allowed_chat_ids": [],
        "require_pairing": True,
        "rate_limit_per_min": 20,
    }
    assert "token" not in staged_public["gateways"]["telegram"]
    cfg.apply()
    # secret is stored in the real config (in ~/.bonsai, never a repo file)...
    assert cfg.settings.gateways.telegram.token == "123:abc"
    # ...but never appears in the public shape
    assert cfg.settings.public_dict()["gateways"]["telegram"]["token_set"] is True


def test_whatsapp_qr_settings_round_trip(bonsai_home: Path) -> None:
    """WhatsApp is QR-linked (api.md 2026-07-22b): {enabled, linked, to_number}, no
    secret. The shape round-trips through staging/apply and TOML verbatim."""
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"gateways": {"whatsapp": {"enabled": True, "to_number": "15551234567"}}})
    staged_public = cfg.staged_settings().public_dict()["gateways"]["whatsapp"]
    assert staged_public == {"enabled": True, "linked": False, "to_number": "15551234567"}
    assert "access_token" not in staged_public and "access_token_set" not in staged_public
    cfg.apply()
    assert cfg.settings.gateways.whatsapp.to_number == "15551234567"
    # Committed to disk and reloadable.
    assert manager(bonsai_home).load().gateways.whatsapp.enabled is True


def test_set_whatsapp_linked_persists_directly(bonsai_home: Path) -> None:
    """Linking/unlinking is a runtime action (scan QR / POST unlink), persisted directly
    like Telegram pairing — no staging."""
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.set_whatsapp_linked(True)
    assert cfg.settings.gateways.whatsapp.linked is True
    assert manager(bonsai_home).load().gateways.whatsapp.linked is True  # on disk
    cfg.set_whatsapp_linked(False)
    assert manager(bonsai_home).load().gateways.whatsapp.linked is False


def test_provider_api_key_write_only_and_list_replaces_wholesale(bonsai_home: Path) -> None:
    """providers[] (additive 2026-07-21c): api_key follows the write-only-secret
    pattern; being a list, a PATCH replaces the whole array (deep_merge semantics)."""
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage(
        {
            "providers": [
                {"name": "ollama", "kind": "ollama", "enabled": True},
                {
                    "name": "corp",
                    "kind": "openai",
                    "base_url": "https://llm.example.test",
                    "enabled": True,
                    "api_key": "sk-secret",
                },
            ]
        }
    )
    staged_public = cfg.staged_settings().public_dict()["providers"]
    assert [p["name"] for p in staged_public] == ["ollama", "corp"]
    assert staged_public[0]["base_url"] == "http://127.0.0.1:11434"  # ollama preset
    assert staged_public[1] == {
        "name": "corp",
        "kind": "openai",
        "base_url": "https://llm.example.test",
        "enabled": True,
        "api_key_set": True,
    }
    assert all("api_key" not in p for p in staged_public)
    cfg.apply()
    # secret is stored in the real config (in ~/.bonsai, never a repo file)...
    assert cfg.settings.providers[1].api_key == "sk-secret"
    # ...but never appears in the public shape, and round-trips through TOML.
    assert manager(bonsai_home).load().providers[1].api_key == "sk-secret"


def test_provider_validation(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    with pytest.raises(BonsaiError):  # "/" would collide with the model-id join
        cfg.stage({"providers": [{"name": "bad/name", "kind": "openai"}]})
    with pytest.raises(BonsaiError):  # duplicate names
        cfg.stage({"providers": [{"name": "twin", "kind": "openai"}, {"name": "twin"}]})
    assert cfg.staged is None  # nothing partially staged


def test_search_api_key_write_only(bonsai_home: Path) -> None:
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"search": {"provider": "brave", "api_key": "brave-secret"}})
    staged_public = cfg.staged_settings().public_dict()["search"]
    assert staged_public == {"provider": "brave", "base_url": "", "api_key_set": True}
    assert "api_key" not in staged_public
    cfg.apply()
    assert cfg.settings.search.api_key == "brave-secret"
    assert cfg.settings.public_dict()["search"]["api_key_set"] is True
    with pytest.raises(BonsaiError):
        cfg.stage({"search": {"provider": "bing"}})  # unknown provider name


def test_dspark_and_kv_are_restart_only_effort_default_is_idle(bonsai_home: Path) -> None:
    """Apply-effect classes tell the truth (api.md 2026-07-21d): launch-flag keys
    report requires_restart; effort_default (read per request) applies at idle."""
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.stage({"dspark_enabled": True, "effort_default": "high"})
    result = cfg.apply()
    assert result["requires_restart"] == ["dspark_enabled"]
    assert result["pending_until_idle"] == ["effort_default"]


def test_effort_default_unset_by_default_and_settable(bonsai_home: Path) -> None:
    """effort_default defaults to None (each mode keeps its own default effort);
    setting it round-trips through TOML, clearing it drops the key."""
    cfg = manager(bonsai_home)
    settings = cfg.load()
    assert settings.effort_default is None
    assert cfg.settings.public_dict()["effort_default"] is None
    cfg.stage({"effort_default": "xhigh"})
    cfg.apply()
    assert cfg.settings.effort_default == "xhigh"
    assert manager(bonsai_home).load().effort_default == "xhigh"  # persisted
    with pytest.raises(BonsaiError):
        cfg.stage({"effort_default": "turbo"})


def test_deep_merge_replaces_scalars_merges_dicts() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": [1, 2]}
    patch = {"a": {"b": 9}, "d": [3]}
    assert deep_merge(base, patch) == {"a": {"b": 9, "c": 2}, "d": [3]}


def test_malformed_config_toml_is_clean_error(bonsai_home: Path) -> None:
    """A hand-edited config.toml with a syntax error is a clean BonsaiError naming the
    file + remedy (audit 2026-07-22), not a raw TOMLDecodeError traceback from serve."""
    bonsai_home.mkdir(parents=True, exist_ok=True)
    (bonsai_home / "config.toml").write_text("port = = 8686\n[unclosed\n", encoding="utf-8")
    with pytest.raises(BonsaiError) as exc_info:
        manager(bonsai_home).load()
    assert exc_info.value.code == "config_invalid_toml"
    assert "config.toml" in exc_info.value.message


def test_malformed_staged_toml_is_clean_error(bonsai_home: Path) -> None:
    """A corrupt staged.toml is caught on the same path as config.toml."""
    cfg = manager(bonsai_home)
    cfg.load()
    cfg.staged_file.write_text("this is not toml ][[\n", encoding="utf-8")
    with pytest.raises(BonsaiError) as exc_info:
        manager(bonsai_home).load()
    assert exc_info.value.code == "config_invalid_toml"
    assert "staged.toml" in exc_info.value.message
