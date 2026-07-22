#!/bin/sh
# suiban install — one command from a fresh clone to a ready venv + config.
#
# Does only the light, offline-safe part:
#   1. check for uv
#   2. uv sync --all-extras   (create .venv, install suiban + dev deps)
#   3. create ~/.bonsai/ and seed config.toml from config.example.toml if absent
# then prints the remaining steps: the (large) binary + model downloads and doctor.
#
# The heavyweight downloads (fork prebuilt binaries, GGUF model weights) are done on
# demand by `uv run suiban install ...`, never committed to the repo and never pulled
# behind your back. For one interactive command that also runs those downloads, use
# ./bootstrap.sh --full instead.
set -eu

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required but not found." >&2
    echo "install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "==> uv sync (creating .venv and installing suiban + dev deps)"
uv sync --all-extras

# Seed the real config from the checked-in example when the user has none yet. suiban
# also writes a default config on first run, but seeding the fully-commented example
# puts every option in front of you to edit. Real config lives ONLY under ~/.bonsai.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BONSAI_HOME="${BONSAI_HOME:-$HOME/.bonsai}"
CONFIG="$BONSAI_HOME/config.toml"
if [ -f "$CONFIG" ]; then
    echo "==> $CONFIG already exists — left untouched"
else
    mkdir -p "$BONSAI_HOME"
    cp "$SCRIPT_DIR/config.example.toml" "$CONFIG"
    echo "==> wrote $CONFIG (from config.example.toml)"
fi

echo ""
echo "install done. remaining steps (the large downloads happen here, not before):"
echo "  uv run suiban install binaries    # pinned PrismML llama.cpp fork prebuilts (a few hundred MB)"
echo "  uv run suiban install models      # Bonsai GGUF weights (~11.5 GB ternary; --family 1bit ~6.4 GB)"
echo "  uv run suiban doctor              # the gate: binary / models / config / telemetry"
echo ""
echo "then start the server — the one run command:"
echo "  uv run suiban serve               # http://127.0.0.1:8686"
