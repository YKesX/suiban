# Hardware tiers & VRAM math

What runs on what, with the arithmetic shown. Every number below is an **analytic
prior**, derived from verified model architecture facts and GGUF file sizes, unless
marked measured. After your first launch, suiban measures real footprints on your
machine and the measured values override these priors everywhere (see
[§5](#5-analytic--measured-how-the-numbers-become-real)).

Units: GiB (2³⁰ bytes) and KiB (2¹⁰ bytes) throughout.

## 1. The tiers

| VRAM | Loadout (slots) | Est. total | Notes |
|---|---|---|---|
| 24 GB | ternary 27B + 4B utility + 2×8B workers | ~18.5 GiB | Full experience, parallel Ultra |
| 16 GB | ternary 27B + 4B utility + 1×8B worker | ~14.9 GiB | Parallel Ultra, one worker |
| 12 GB | **1-bit** 27B + 4B utility + 1×4B worker | ~10.8 GiB | Family auto-degrades, with notice |
| 8 GB | **1-bit** 27B + 1.7B utility | ~7.6 GiB | No workers; Ultra runs sequentially |
| CPU-only | one model sized to RAM (27B if ≥16 GB RAM, else 8B) | n/a | Utility = orchestrator; no workers |

Default contexts in the math below: orchestrator 32,768 · workers 16,384 (both
settings-configurable) · utility 8,192 (fixed in v1). KV config is the default
K=`q8_0` + V=`TQ4_0` unless stated.

The 27B orchestrator is present at **every** GPU tier, on 12 GB and 8 GB cards via the
1-bit family. That is deliberate: vision, tier-2 browsing and skill/memory writing all
require the 27B, so degrading the family (with a visible notice, and
`quant_family.configured` vs `.effective` in `/v1/system`) keeps the whole feature set
alive on small cards instead of silently dropping features.

The 8 GB analytic total (~7.6 GiB) looks tight on purpose: the buffer priors are
deliberately conservative, and on a card that also drives a display free VRAM is below
8 GiB anyway. First-launch measurement decides what actually fits on your machine. On
the reference 8 GB machine the planner reduced the orchestrator context to 16K (with a
notice) and the measured buffers came in well under the priors, landing the loadout at
~6.4 GiB with ~1.6 GiB headroom (see [§6](#6-measured-on-real-hardware)). The planner's
safety margin is applied to measured values, not to hope.

## 2. Model weights (GGUF file sizes, verified)

| Model | Ternary `Q2_0` | 1-bit `Q1_0` | Max ctx | Notes |
|---|---|---|---|---|
| bonsai-27b | 6.67 GiB | 3.54 GiB | 262,144 | + mmproj `Q8_0` 629 MB (vision; 27B only) |
| bonsai-8b | 2.03 GiB | 1.08 GiB | 65,536 | Ultra worker |
| bonsai-4b | 1.00 GiB | 0.53 GiB | 32,768 | utility (resident in every GPU loadout) |
| bonsai-1.7b | 0.43 GiB | 0.23 GiB | 32,768 | micro worker / 8 GB-tier utility |

`Q1_0`/`Q2_0` are the PrismML fork's type ids (41/42), not mainline llama.cpp's
`TQ1_0`/`TQ2_0` (TriLM), and fork `Q2_0` uses group size 128, not mainline's 64. Never
mix fork models with mainline builds.

## 3. KV cache: bytes per token, derived

Per-token KV cost for one side (K or V) is:

```
bytes/token/side = kv_layers × kv_heads × head_dim × bytes_per_element(cache type)
```

**Bytes per element**, from the block layouts:

| Cache type | Block layout | Bytes/element | Bits/weight |
|---|---|---|---|
| `f16` | none | 2.0 | 16 |
| `q8_0` | 34 B / 32 elements | 1.0625 | 8.5 |
| `tq4_0` (TurboQuant) | 18 B / 32 (fp16 scale + 16 B codes) | 0.5625 | 4.5 |
| `tq3_0` (TurboQuant) | 14 B / 32 (fp16 scale + 12 B codes) | 0.4375 | 3.5 |

**KV-carrying dimensions per model** (verified): the 27B is a 64-layer hybrid-attention
design where only 16 layers are full attention: per-token KV growth comes from those 16
layers only, which is why its KV is *cheaper* per token than the 8B's. (The remaining
layers' bounded buffers live in the fixed-buffer prior, not here.)

| Model | KV layers × heads × head dim | Elements/token/side |
|---|---|---|
| 27B | 16 × 4 × 256 | 16,384 |
| 8B | 36 × 8 × 128 | 36,864 |
| 4B | 36 × 8 × 128 | 36,864 |
| 1.7B | 28 × 8 × 128 | 28,672 |

**Total KiB per token (K + V)** for every supported combination:

| Model | K=f16 V=f16 | K=q8_0 V=q8_0 | K=q8_0 V=tq4_0 (default) | K=q8_0 V=tq3_0 (aggressive) |
|---|---|---|---|---|
| 27B | 64.0 | 34.0 | **26.0** | 24.0 |
| 8B | 144.0 | 76.5 | **58.5** | 54.0 |
| 4B | 144.0 | 76.5 | **58.5** | 54.0 |
| 1.7B | 112.0 | 59.5 | **45.5** | 42.0 |

Worked check (27B, default config): K = 16,384 × 1.0625 B = 17.0 KiB; V = 16,384 ×
0.5625 B = 9.0 KiB; total 26.0 KiB/token. At the default 32,768 context that is
26.0 KiB × 32,768 = **0.81 GiB** (vs 2.0 GiB at f16, 2.46× smaller). At the 27B's max
context of 262,144 tokens the default-config KV would be 6.5 GiB, essentially a second
copy of the weights, which is why contexts are planned, not maxed.

**KV at the default per-slot contexts:**

| Slot | Ctx | q8_0/tq4_0 | q8_0/q8_0 | f16/f16 |
|---|---|---|---|---|
| 27B orchestrator | 32,768 | 0.81 GiB | 1.06 GiB | 2.00 GiB |
| 8B worker | 16,384 | 0.91 GiB | 1.20 GiB | 2.25 GiB |
| 4B worker | 16,384 | 0.91 GiB | 1.20 GiB | 2.25 GiB |
| 4B utility | 8,192 | 0.46 GiB | 0.60 GiB | 1.13 GiB |
| 1.7B utility | 8,192 | 0.36 GiB | 0.47 GiB | 0.88 GiB |

## 4. Loadout math per tier

Slot cost = weights + mmproj (27B only) + KV + fixed buffer prior. Buffer priors
(compute/scratch/graph allocations) are the `BUFFER_PRIOR_GIB` table in
`sched/budget.py`: **1.2 GiB for the 27B, 0.6 GiB for every other size**, deliberately
conservative round numbers, replaced per machine by first-launch measurement (on the
reference 8 GB machine the measured 27B and 1.7B buffers came in at roughly 80% and 40%
of these priors; see [§6](#6-measured-on-real-hardware)).

**24 GB (ternary):**

| Slot | Weights | mmproj | KV | Buffers | Total |
|---|---|---|---|---|---|
| 27B orchestrator @32K | 6.67 | 0.63 | 0.81 | 1.20 | **9.3** |
| 4B utility @8K | 1.00 | n/a | 0.46 | 0.60 | **2.1** |
| 8B worker @16K | 2.03 | n/a | 0.91 | 0.60 | **3.5** |
| 8B worker @16K | 2.03 | n/a | 0.91 | 0.60 | **3.5** |
| **Loadout** | | | | | **≈18.5 GiB** |

**16 GB (ternary):** 9.3 + 2.1 + 3.5 = **≈14.9 GiB**.

**12 GB (1-bit 27B):** 27B drops to 3.54 GiB weights → slot ≈ 6.2 GiB; + 4B utility
2.1 + 4B worker @16K (1.00 + 0.91 + 0.60) ≈ 2.5 → **≈10.8 GiB**.

**8 GB (1-bit 27B):** 6.2 + 1.7B utility @8K (0.43 + 0.36 + 0.60) ≈ 1.4 → **≈7.6 GiB**
analytic. That exceeds what an 8 GB card that also drives a display can give, which is
exactly what the priors are for: the planner reduces the orchestrator context (16K on
the reference machine, with a notice) and first-launch measurement shrinks the buffer
numbers to reality. No worker slots; Ultra runs sub-tasks sequentially with a notice.

**CPU-only:** no VRAM budget; a single orchestrator is sized against system RAM (27B if
≥16 GB RAM, else 8B), it doubles as the utility model and there are no workers.
`gpus` is `null` and `telemetry_source` is `"ram"` in `/v1/system`.

**DSpark** (speculative drafter for the 27B, CUDA only, opt-in and default off) adds
~1.8 GB on the orchestrator's GPU for an upstream-reported 1.34× decode speedup. The
planner accounts for it only when the toggle is on; the headroom exists comfortably only
at the 24 GB tier.

## 5. Analytic → measured: how the numbers become real

1. **Fresh install:** the planner prices every candidate slot with the analytic prior
   (HF weight bytes + the KiB/token table above + the fixed buffer priors) and plans a
   loadout with a safety margin.
2. **First launch:** real slot launches are bracketed by telemetry snapshots; the
   machine-dependent buffer cost (observed VRAM delta minus the exact weights + KV
   math) is persisted to `~/.bonsai/budget.json`, keyed by **model + family**. Only the
   ctx-independent components (buffers, and weights if they deviate) are stored. KV is
   never "measured" because it scales exactly with context by the table above.
3. **Ever after:** measured buffer/weight values override the priors for that
   model + family at *any* context or KV config; `GET /v1/system/budget` marks each row
   `"source": "analytic"` or `"measured"` so dai/sentei can show which numbers came from
   *your* machine. An unmeasured model/family falls back to the analytic prior until it
   is measured in turn.

Additionally, `suiban bench kv` produces quality numbers (perplexity delta,
long-context needle retrieval and a multi-turn replay for V ∈ {TQ4, TQ3, q4_0, q8_0})
to back the TurboQuant disclaimer with data from your own GPU. Results from the
reference machine are in [benchmarks.md](benchmarks.md).

## 6. Measured on real hardware

Everything above is analytic. This section is **measured, on exactly one machine**:
an RTX 3070 Ti Laptop GPU with 8 GB VRAM (the 8 GB tier), CUDA backend, 1-bit 27B +
1.7B utility loadout. It validates the method, not your hardware; your numbers will
differ, and suiban will measure them.

**Measured buffers vs the priors** (`~/.bonsai/budget.json` on that machine):

| Model (1-bit) | Buffer prior | Measured | Prior / measured |
|---|---|---|---|
| bonsai-27b | 1229 MiB (1.2 GiB) | 970 MiB | 1.27× |
| bonsai-1.7b | 614 MiB (0.6 GiB) | 248 MiB | 2.48× |

(The very first launch on this machine recorded 931 MiB / 269 MiB; a later launch
re-measured 970 MiB / 248 MiB. Run-to-run drift of a few tens of MiB is normal;
budget.json keeps the latest.)

**The loadout that actually booted** (orchestrator context reduced to 16K by the
planner, with a notice; the configured 32K did not fit the 8 GB budget):

| Slot | Weights | mmproj | KV | Buffers (measured) | Predicted | Slot-reported |
|---|---|---|---|---|---|---|
| 27B @16K (1-bit) | 3625 MiB | 645 MiB | 416 MiB | 970 MiB | 5656 MiB | 5671 MiB |
| 1.7B @8K (1-bit) | 236 MiB | n/a | 364 MiB | 248 MiB | 848 MiB | 841 MiB |

Prediction error under 20 MiB per slot once buffers are measured. That is the whole
point of the analytic → measured design. Total ~6.4 GiB with ~1.6 GiB headroom on the
8 GB card.

**Speed on the same machine** (from [benchmarks.md](benchmarks.md), 1-bit 27B,
K=`q8_0` V=`tq4_0`, TurboQuant build): decode ~39.8 tok/s at empty context,
~31.6 tok/s at 4K prefilled depth, ~19.5 tok/s at 16K depth (the q8_0 V-cache control:
46.3 / 42.8 / 35.7). One machine, one model, laptop thermals. Re-measure with the
commands in benchmarks.md.

## 7. Feature availability by tier

Vision, tier-2 browsing and skill/memory writes all require a resident 27B; parallel
Ultra requires at least one worker slot. These are computed, not hardcoded. Clients
read `capabilities` from `GET /v1/system`.

| Capability | 24 GB | 16 GB | 12 GB | 8 GB | CPU-only (27B) | CPU-only (8B) |
|---|---|---|---|---|---|---|
| `vision` | yes | yes | yes | yes | yes (slow) | no |
| `browse_t2` | yes | yes | yes | yes | yes | no |
| `skill_writes` | yes | yes | yes | yes | yes | no |
| `ultra_parallel` | yes (2 workers) | yes (1) | yes (1) | no (sequential) | no (sequential) | no (sequential) |

The CPU-only columns split on the RAM rule from §1: with ≥16 GB RAM the orchestrator is
the 27B and 27B-gated capabilities stay on (at CPU speed); below that the 8B
orchestrates and they are off, reported honestly by the capability flags.

## 8. Multi-GPU placement

With two or more GPUs: orchestrator and utility on GPU 0, workers on GPU 1. Telemetry
and headroom are tracked per GPU, and the worker degrade ladder
(2×8B → 8B+4B → 2×4B → 1×4B → 1×1.7B → none) is applied against the worker GPU's
budget. There is no tensor-parallel splitting of a single model across GPUs in v1
(`TODO(v1.1): evaluate fork support for split loading before promising it`).
