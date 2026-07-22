# suiban FAQ

Real questions this project actually hit, with honest answers.

## The server starts and `/v1/system` answers, but every chat fails. Why?

Because serving and inference are separate pieces, and only the first one ships in the
repo. `./bootstrap.sh` (without `--full`) only creates the venv. It downloads nothing.
A server started after that will happily answer system endpoints while every chat fails,
because there is no llama-server binary and no model weights yet. Run:

```bash
uv run suiban install binaries    # pinned fork prebuilts (a few hundred MB)
uv run suiban install models      # ~11.5 GB ternary / ~6.4 GB 1-bit
uv run suiban doctor              # confirms both, or names what is missing
```

`suiban doctor` is the gate: it prints the exact missing piece and the command that
fixes it. Or run `./bootstrap.sh --full`, which walks all of this interactively.

## Why does Ultra take so long on my 8 GB card?

On the 8 GB tier there is no VRAM for worker slots, so Ultra runs its sub-tasks
**sequentially on the one orchestrator slot**. `/v1/system` reports
`ultra_parallel: false` and a notice says so. Every sub-task is a full agentic run at
local-GPU decode speeds, one after another. suiban bounds this (sub-task effort capped
at `mid` when sequential, at most 3 sub-tasks, per-sub-task timeouts), which took a
measured worst case from "past 10 minutes for a trivial question" to bounded, but
sequential Ultra is still minutes, not seconds. That is the tier, not a bug. Parallel
Ultra starts at 12 GB (one worker) and is the full experience at 24 GB (two).

## Why does K stay q8_0 when V is TurboQuant?

Asymmetric by design. Quantization error in K perturbs attention *logits*, and softmax
turns small logit noise into disproportionate probability shifts, especially for the
few high-attention keys that dominate a head. Error in V only perturbs the
post-softmax weighted average, which is far more forgiving. So suiban quantizes
aggressively where it is safe (V = TQ4_0/TQ3_0) and conservatively where it is not
(K = q8_0). Details and the measured quality battery: `docs/turboquant.md` and
`docs/benchmarks.md`.

## Why don't the prebuilt binaries include TurboQuant?

The TurboQuant kernels live in our vendored patchset, applied on top of the pinned
PrismML fork. The fork's own release assets are built without it. So a default
install runs K/V=q8_0 with a visible notice (never silently), until you build from
source:

```bash
uv run suiban install turboquant   # clone + patch + build + swap (CUDA/CPU only in v1)
```

Metal kernels do not exist yet (TODO v1.1); Vulkan/ROCm are out of scope for v1. On
those backends the command declines honestly and the q8_0 fallback stays.

## I have CUDA 13: why does the installer download a CUDA 12.x build, and does it work?

The fork publishes Linux prebuilts per CUDA toolkit line (12.8 preferred, 12.4 as the
compatibility fallback for older driver branches). What matters at runtime is your
**driver**, not your toolkit: any driver recent enough for CUDA 13 runs the 12.x
binaries fine. Your local toolkit only matters when you build TurboQuant from source:
that build uses *your* nvcc, and with CUDA 13.x plus a very new host compiler (GCC 16)
you may need to point nvcc at an older one:

```bash
uv run suiban install turboquant --cuda-host-compiler /usr/bin/g++-15
```
