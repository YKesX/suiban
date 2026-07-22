"""Messaging gateways (docs/gateways.md).

v1 ships Telegram (long-polling only — outbound connections, no webhooks, no open
ports). Gateways are ordinary API clients: they talk to suiban through the same
frozen HTTP contract as dai and sentei, never through internals.
"""
