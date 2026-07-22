# Contributing to suiban

Thanks for helping tend the tray. Short version: keep the contract frozen, keep the
docs honest, keep secrets out of the tree.

## Dev setup

```bash
git clone https://github.com/YKesX/suiban && cd suiban
./bootstrap.sh                    # uv venv + deps (add --full for binaries/models)
SUIBAN_LLAMA_MOCK=1 uv run pytest -q    # full suite, no GPU or model files needed
```

`SUIBAN_LLAMA_MOCK=1` swaps the llama-server backend for a mock; the suite (551 tests
at the time of writing) runs on any machine. Real-hardware paths (installers,
`suiban bench kv`, TurboQuant builds) have their own opt-in harnesses documented in
`docs/` and `vendor/README.md`.

## Checks that must pass

```bash
uv run ruff check .           # lint
uv run ruff format --check .  # formatting (line length 100)
SUIBAN_LLAMA_MOCK=1 uv run pytest -q
```

## Style

- Python 3.11+ compatible, `src/` layout, type hints on public functions.
- `ruff` is the formatter and linter; line length 100.
- Prose and docs are honest: no invented benchmark numbers (measured tables cite the
  machine they came from), no fake completeness. Unfinished work is a
  `TODO(v1.1): reason` marker, not silence.

## The one law: the API contract

`docs/api.md` is the **frozen v1 contract** and the only coordination point with the
`dai` (GUI) and `sentei` (CLI) repos. Within v1, changes are **additive only**:

1. Land the change in `docs/api.md` first, with a dated entry in its Changelog.
2. Then implement it in code.
3. Never rename, remove or retype an existing field.

No cross-repo imports, no shared code, no relative paths into sibling repos.

## Secrets and machine-specific paths

- Never commit secrets, tokens or machine-specific paths. Real config lives in
  `~/.bonsai/` (outside the repo); the repo ships `config.example.toml` only.
- Everything heavyweight (venv, binaries, model weights, databases, reports) is
  generated or downloaded into `~/.bonsai/` or gitignored. Keep `.gitignore`
  authoritative.
- API secrets (gateway tokens, provider keys) are write-only over the HTTP surface by
  design; keep them that way.

## Licensing

Apache-2.0 for everything authored here. The TurboQuant TQ3_0 CPU reference is ported
from an MIT-licensed branch: keep the attribution in `vendor/README.md` and
`docs/turboquant.md` intact. The community CUDA gist cited in the TurboQuant docs has
no license: cite its numbers, never copy its code.

## Pull requests

- Small and focused; one concern per PR.
- Lint + format + tests green (commands above).
- A `CHANGELOG.md` entry under `[Unreleased]` for anything user-visible.
- Contract-touching changes include the api.md changelog entry (see above) in the
  same PR, contract first.
- Docs updated in the same PR when behavior changes. A doc that describes the old
  behavior is a bug.
