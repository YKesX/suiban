# suiban 水盤

**Local inference & orchestration core for the bonsai stack.** Runs the
[PrismML Bonsai](https://docs.prismml.com) model family (ternary / 1-bit GGUF) on
consumer GPUs via the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp),
and serves a full agentic experience entirely on your machine: chat, agentic coding,
multi-agent Ultra, deep research, vision, persistent memory and self-improving skills.
No cloud, no telemetry, no vector databases.

Author: Yağızhan Keskin ([github.com/YKesX](https://github.com/YKesX))

> A suiban (水盤) is the shallow tray a bonsai stands in: it holds everything up.
> Clients: [dai](https://github.com/YKesX/dai) (GUI) and
> [sentei](https://github.com/YKesX/sentei) (CLI) both speak plain HTTP to
> `http://127.0.0.1:8686`.

## Features

- **VRAM-aware scheduler.** Measures real model+KV footprints on your hardware, plans a
  loadout (orchestrator + utility + workers) that fits and never swaps models mid-run.
- **Lazy / keep-alive residency (ollama-style).** `serve` comes up healthy with zero VRAM
  in use and warms the planned slots on the first inference request. A background reaper
  then unloads the whole loadout after the idle window (`runtime.keep_alive`, default
  5 minutes; `"24/7"` stays hot). It never unloads mid-generation, and a cold start leads a
  rich stream with a `warming_up` notice.
- **TurboQuant KV cache (default on).** V-cache compressed with 4-bit TurboQuant
  ([Zandieh et al., arXiv:2504.19874](https://arxiv.org/abs/2504.19874)) via our vendored
  fork patchset; K stays q8_0. Roughly 2.4x smaller KV than f16 with near-lossless quality.
  It falls back gracefully (q8_0 then f16) with a visible notice wherever kernels are not
  available. See [docs/turboquant.md](docs/turboquant.md) and the honest state table below.
- **Four modes.** Chat, agentic code (plan then act then verify), Ultra (contained parallel
  sub-agents coordinated over the [SLAP](https://github.com/YKesX/SLAP) protocol) and deep
  research (async 15 to 40 min jobs with pluggable web search, coarse progress only).
- **Effort ladder.** low/mid/high/xhigh/max mapped to Bonsai thinking-token budgets and
  tool-loop limits.
- **Memory & skills, all local.** SQLite FTS5 session archive, bounded state files and
  agentskills.io-compatible skills. Only the 27B orchestrator writes memories and skills,
  in post-task reflection. Chat compression via the resident utility model at ~70% context.
  Import chats from ChatGPT/Claude/Claude-Code/generic exports
  (`POST /v1/memory/sessions/import`) and import skills from openclaw or Hermes
  (`POST /v1/skills/import`).
- **Vision.** Images go to the 27B (mmproj); workers never see images.
- **Confirmation gates.** In code and ultra modes, destructive shell commands and file
  mutations (`fs_write`, `fs_undo`) are refused first with a single-use `confirm_token`
  and a unified diff, so nothing touches disk before you approve. An `auto_confirm` bypass
  exists for power users and logs every auto-confirmed action, never silently.
- **Client identities.** The `X-Bonsai-Client` header (`dai`/`sentei`/`other`) selects an
  editable identity overlay merged into the system prompt on top of the base `identity.md`.
- **MCP servers plus a curated connector catalog.** Attach external MCP servers (stdio
  transport, `settings.mcp_servers[]`), or enable one-click connectors from a built-in
  catalog (filesystem, git, fetch, memory, everything, sequential-thinking, time). Their
  tools join chat/code runs namespaced `mcp_<server>_<tool>`; a crashed server is a notice,
  never a crash.
- **External providers.** Register Ollama or any OpenAI-compatible endpoint
  (`settings.providers[]`); their models appear in `/v1/models` and chat as plain proxies
  (chat mode only, honest limits: no thinking control, no grammar guarantees, no VRAM
  scheduling). The local stack stays primary.
- **Web search for deep research.** Pluggable providers: DuckDuckGo (keyless default),
  SearXNG, Brave, Tavily and Serper; test button via `POST /v1/system/search_test`.
- **Gateways.** Telegram bot (chat relay + pings) and WhatsApp (QR device-linked,
  outbound-only pings; inbound relay is TODO(v1.2)).
- **OpenAI-compatible.** `POST /v1/chat/completions` works with any OpenAI client;
  first-party clients opt into a richer event stream. Contract: [docs/api.md](docs/api.md).

## Hardware (honest numbers, ternary family unless noted)

| VRAM | Loadout | Notes |
|---|---|---|
| 24 GB | 27B + 4B utility + 2×8B workers (~18.5 GiB) | Full experience, parallel Ultra |
| 16 GB | 27B + 4B utility + 1×8B worker | Parallel Ultra with one worker |
| 12 GB | 27B **(1-bit)** + 4B utility + 1×4B worker | Family auto-degrades with notice |
| 8 GB | 27B **(1-bit)** + 1.7B utility | Ultra runs sequentially |
| CPU-only | One model sized to RAM (27B if ≥16 GB) | Slow but functional; no workers |

Vision, tier-2 browsing and skill/memory writing require the 27B, resident at every
GPU tier above (via 1-bit degradation on small cards). KV cost with default
K=q8_0 + V=TQ4: 27B ≈ 26 KiB/token (hybrid attention: only 16 of 64 layers carry KV);
8B/4B ≈ 58.5 KiB/token. Numbers are analytic priors; `suiban bench kv` and the
first-launch measurement replace them with values from *your* machine.

### TurboQuant backend state (v1, honest)

| Backend | TQ4_0 / TQ3_0 kernels | Fallback |
|---|---|---|
| CPU | ✅ built + unit-tested vs the paper's MSE targets | none |
| CUDA | ✅ built + **runtime-validated end-to-end** (27B served with V=tq4_0; PPL delta vs q8_0 within the error bar; needle 10/50/90% pass; see [docs/turboquant.md](docs/turboquant.md)) | q8_0/q8_0 until built |
| Metal | 🚧 TODO(v1.1) | q8_0/q8_0 |
| Vulkan / ROCm | ❌ out of scope for v1 | q8_0/q8_0 |

Prebuilt fork binaries do **not** include TurboQuant kernels; the default install runs
with K/V=q8_0 and says so in `sentei status` / dai's System panel until you run the
source build. Never silent, never a crash.

## Install

Two commands from a fresh clone to a ready server, then one to run it:

```bash
git clone https://github.com/YKesX/suiban && cd suiban
./install.sh                                                   # 1. uv venv + deps, seed ~/.bonsai/config.toml
uv run suiban install binaries && uv run suiban install models # 2. fork prebuilts + GGUF weights (~11.5 GB ternary)
uv run suiban serve                                            # run: http://127.0.0.1:8686
```

`./install.sh` is fast and offline-safe: it checks for `uv`, syncs the venv and seeds
`~/.bonsai/config.toml` from [config.example.toml](config.example.toml). The large
downloads are the explicit second step (`--family 1bit` is ~6.4 GB; `both` ~17.9 GB).

Prefer one interactive command that asks before every large download:

```bash
./bootstrap.sh --full   # venv -> doctor -> fork binaries -> model weights -> doctor
uv run suiban serve     # http://127.0.0.1:8686
```

Optional, CUDA/CPU: build the TurboQuant-enabled fork binary from source:

```bash
uv run suiban install turboquant
```

Model weights download from [Hugging Face `prism-ml`](https://huggingface.co/prism-ml)
(the ternary set includes the 27B vision projector; `--dspark` adds the optional
~1.9 GB speculative drafter). Everything lands under `~/.bonsai/` so the repo stays
clean. `suiban doctor` is the gate: a server without binaries or models still starts
and serves `/v1/system`, but chats fail until both installs have run. Doctor prints
the exact missing piece and its fix command.

**Local or remote.** The clients ([dai](https://github.com/YKesX/dai),
[sentei](https://github.com/YKesX/sentei)) reach suiban over plain HTTP. Same machine is
the zero-config default (loopback `127.0.0.1:8686`, no auth). To serve them from another
machine, set `server.host` to a non-loopback address; suiban then requires a bearer token
on every request (see [Security](#security) below).

## Benchmark it yourself

```bash
uv run suiban bench kv   # perplexity delta + long-context needle: TQ4 vs TQ3 vs q4_0 vs q8_0
```

Backs the settings-page disclaimer with numbers from your own GPU. Reference points:
the paper's 3.5-bit TurboQuant exactly ties the full-precision cache on LongBench
(50.06 avg, Llama-3.1-8B), and our own end-to-end run on an RTX 3070 Ti Laptop 8 GB
(1-bit 27B) measured tq4_0/tq3_0 perplexity inside the q8_0 baseline's error bar with
all needle depths passing. Full measured tables (quality battery + decode-speed
before/after the warp-shfl fast path) live in
[docs/benchmarks.md](docs/benchmarks.md); design detail in
[docs/turboquant.md](docs/turboquant.md).

## Security

suiban runs local models with real tools (shell, filesystem, browsing). The defaults are
built for a single user on their own machine; read this before changing them.

- **Loopback by default.** suiban binds `127.0.0.1:8686` with no authentication, which is
  safe because only your machine can reach it. **If you change `server.host` to a
  non-loopback address (LAN/`0.0.0.0`), suiban requires a bearer token** on every request
  except the health check: a token is generated and printed to the server console on
  first non-loopback start, and stored write-only in `~/.bonsai/config.toml`. Send it as
  `Authorization: Bearer <token>`. CORS is a browser convenience, not an auth boundary.
- **Telegram is default-deny.** The bot answers nobody until they pair: run `/pair <code>`
  with the one-time code printed to the *server console* (never sent over Telegram).
  Unpaired chats get one "not authorized" reply and reach nothing. Paired users are
  confined to chat mode; there is no path from Telegram to the shell or filesystem in v1
  (`server.remote_agentic` is reserved, default off and not honored). Per-chat rate
  limiting is on by default.
- **The shell tool is not a sandbox.** In code mode a model can run shell commands and
  edit files inside the chosen workspace. Destructive commands are confirmation-gated and
  secret-bearing environment variables are stripped from the tool subprocess, but a
  determined prompt-injection can still do what a shell can do. **Run suiban as an
  unprivileged user**; OS-level sandboxing (bwrap/landlock) is planned for v1.1. See
  [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- **Untrusted content is fenced.** Fetched web pages, file contents, skill bodies and MCP
  tool outputs are wrapped in `<<<untrusted …>>>` blocks and the models are instructed to
  treat them as data, never instructions. Browsing blocks private/loopback/link-local
  addresses (including via DNS and per-redirect-hop) to prevent SSRF against cloud
  metadata. MCP servers you configure run unsandboxed with your privileges, so add only
  ones you trust.
- **Downloads are integrity-checked.** Fork binaries are verified against a checked-in
  SHA-256 manifest; model weights are size-tripwired on top of Hugging Face's own hashing.
  All downloads are HTTPS; nothing is piped from a URL into a shell.

Open items and known boundaries: [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

## Built with and credits

suiban stands on other people's work. Weights, the fork and the Python libraries are
downloaded or installed at bootstrap, never redistributed here. Full attribution is in
[NOTICE](NOTICE) and [vendor/README.md](vendor/README.md).

**Models and inference**

- [PrismML Bonsai models](https://huggingface.co/prism-ml) ([docs.prismml.com](https://docs.prismml.com)): the 27B/8B/4B/1.7B family in ternary and 1-bit GGUF.
- [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp): the inference engine (branch `prism`, pinned `prism-b9596-9fcaed7`).
- [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT): the upstream the fork and our patchset build on, plus discussion [#20969](https://github.com/ggml-org/llama.cpp/discussions/20969) for the RHT-substitution consensus.

**KV-cache quantization**

- [TurboQuant](https://arxiv.org/abs/2504.19874) (arXiv:2504.19874): the online vector-quantization algorithm behind our TQ4_0/TQ3_0 types.
- [Aaryan-Kapoor/llama.cpp `turboquant-tq3_0`](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0) (MIT): the CPU reference the TQ3_0 kernels are ported from.

**Skills and memory**

- [agentskills.io](https://agentskills.io): the SKILL.md skill format.
- [Hermes](https://github.com/nousresearch/hermes-agent) (MIT): skill-format interoperability and the layered-memory design inspiration.
- [openclaw](https://github.com/openclaw/openclaw) (MIT): skill-format interoperability.

**Python libraries**

FastAPI, uvicorn, httpx, pydantic, huggingface_hub, python-telegram-bot, qrcode,
nvidia-ml-py, psutil, Typer and readability-lxml, each under its own license.

**Sibling repos**

- [SLAP](https://github.com/YKesX/SLAP): the Structured Lightweight Agent Protocol Ultra uses to coordinate sub-agents.
- [dai](https://github.com/YKesX/dai): desktop GUI client.
- [sentei](https://github.com/YKesX/sentei): CLI client.

## License & credit

Apache-2.0. TurboQuant TQ3_0 CPU reference ported from the MIT-licensed
[Aaryan-Kapoor/llama.cpp `turboquant-tq3_0`](https://github.com/Aaryan-Kapoor/llama.cpp/tree/turboquant-tq3_0)
branch: see [vendor/README.md](vendor/README.md) for full attribution. TurboQuant
algorithm: Zandieh, Daliri, Hadian, Mirrokni (Google Research), arXiv:2504.19874.
