#!/bin/sh
# suiban bootstrap — thin by design. Everything heavyweight (fork binaries, model
# weights, TurboQuant build) is downloaded/built by `suiban install ...`, never
# committed to the repo.
#
# Default: only sync the venv and print the next steps.
# --full:  additionally run doctor + install binaries + install models,
#          interactively, asking before every large download.
set -eu

usage() {
    echo "usage: ./bootstrap.sh [--full]"
    echo ""
    echo "  (no flag)  create/sync the uv venv, then print the install steps"
    echo "  --full     also run: doctor, install binaries (a few hundred MB),"
    echo "             install models (~11.5 GB ternary / ~6.4 GB 1-bit),"
    echo "             asking before each download, then a final doctor check"
}

FULL=0
for arg in "$@"; do
    case "$arg" in
        --full) FULL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown option '$arg'" >&2; usage >&2; exit 2 ;;
    esac
done

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv is required but not found." >&2
    echo "install it: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "==> uv sync (creating .venv and installing suiban + dev deps)"
uv sync --all-extras

if [ "$FULL" -eq 0 ]; then
    echo ""
    echo "bootstrap done. next steps (or re-run as: ./bootstrap.sh --full):"
    echo "  uv run suiban install binaries    # pinned PrismML llama.cpp fork prebuilts (a few hundred MB)"
    echo "  uv run suiban install models      # Bonsai GGUF weights (~11.5 GB ternary; --family 1bit ~6.4 GB)"
    echo "  uv run suiban doctor              # the gate: binary / models / config / telemetry"
    echo "  uv run suiban serve               # start the API on http://127.0.0.1:8686"
    exit 0
fi

echo ""
echo "==> suiban doctor (pre-install state; FAILs for binary/models are expected on a fresh checkout)"
uv run suiban doctor || true

echo ""
printf "Download the pinned fork prebuilt binaries into ~/.bonsai/bin/ (a few hundred MB)? [y/N] "
read -r reply
case "$reply" in
    [Yy]*) uv run suiban install binaries ;;
    *) echo "skipped: uv run suiban install binaries" ;;
esac

echo ""
echo "Model weights download from Hugging Face into ~/.bonsai/models/ (one-time):"
echo "  ternary  ~11.5 GB  (default family: 27B + 8B + 4B + 1.7B + vision projector)"
echo "  1bit     ~6.4 GB   (same sizes; what 12 GB / 8 GB cards run)"
echo "  both     ~17.9 GB"
printf "Which family? [ternary/1bit/both/skip] (ternary) "
read -r fam
case "$fam" in
    "" | [Tt]*) uv run suiban install models --family ternary ;;
    [1]*)       uv run suiban install models --family 1bit ;;
    [Bb]*)      uv run suiban install models --family both ;;
    *)          echo "skipped: uv run suiban install models" ;;
esac

echo ""
echo "==> suiban doctor (final check — this is the gate)"
if uv run suiban doctor; then
    echo ""
    echo "bootstrap --full done. start the server with:"
    echo "  uv run suiban serve               # http://127.0.0.1:8686"
    echo "optional (CUDA/CPU): build the TurboQuant-enabled binary:"
    echo "  uv run suiban install turboquant"
else
    echo ""
    echo "doctor reported problems — fix the printed items, then re-check with:"
    echo "  uv run suiban doctor"
    exit 1
fi
