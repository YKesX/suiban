# suiban benchmarks: TurboQuant KV cache, measured

Everything on this page is a **local measurement of one machine**, honestly
labeled: not an official benchmark, not a claim about your hardware. Re-measure
with the commands given (speed) and `suiban bench kv` (quality).

**Machine** (all tables below, unless a row says otherwise):

- RTX 3070 Ti Laptop GPU, 8 GB VRAM (sm_86), the suiban 8 GB hardware tier
- CUDA toolkit 13.2 (nvcc V13.2.86), host compiler g++ 15
- Fork: `PrismML-Eng/llama.cpp` tag `prism-b9596-9fcaed7` + the vendored
  TurboQuant patchset (`suiban/vendor/patches/`)
- Model: Bonsai 27B **1-bit** (`Q1_0`, 3.53 GiB), what this tier actually runs
- K cache `q8_0` throughout (design decision, see
  [`turboquant.md`](turboquant.md))
- Laptop thermals: run-to-run drift of a few percent is normal; deltas well
  beyond that are the signal

## Decode speed: the patch-0007 warp-shfl fast path

Patch 0006 shipped the flash-attention vec kernel with a closed-form parity-sum
V dequant (~32 adds per output element) and an honest TODO: implement a
warp-cooperative fast path if the decode-bound long-context case shows up.
It showed up: at 16K prefilled depth, decode with V=`tq4_0` ran at ~18% of the
q8_0 baseline. Patch 0007 implements the fast path (8 lanes cooperate on a
32-value block: 2 in-register + 3 `__shfl_xor_sync` butterfly stages).

Exact command, run once before (patches 0001–0006) and once after (0001–0007),
same binary directory, `-r 2` repetitions:

```sh
cd suiban/vendor/llama.cpp/build-cuda/bin
LD_LIBRARY_PATH=$PWD ./llama-bench -m <models>/1bit/Bonsai-27B-Q1_0.gguf \
  -fa on -ctk q8_0 -ctv q8_0,tq4_0,tq3_0 -p 512 -n 128 -d 0,4096,16384 -r 2 -o md
```

Generation (`tg128`, tokens/s, the fattn **vec** path 0007 targets):

| V cache | depth 0 before → after | depth 4096 before → after | depth 16384 before → after |
|---|---|---|---|
| `q8_0` (control) | 45.90 → 46.29 | 42.98 → 42.81 | 35.90 → 35.68 |
| `tq4_0` | 39.19 → 39.83 | 17.54 → **31.60** (1.80×) | 6.51 → **19.47** (2.99×) |
| `tq3_0` | 35.46 → 37.44 | 16.06 → **30.11** (1.87×) | 5.99 → **18.62** (3.11×) |

Prompt processing (`pp512`, tokens/s; the prefill path reads V through the
convert kernels, which already had a warp butterfly; 0007 does not touch it):

| V cache | depth 0 before → after | depth 4096 before → after | depth 16384 before → after |
|---|---|---|---|
| `q8_0` (control) | 786.27 → 786.33 | 754.24 → 745.05 | 648.15 → 643.84 |
| `tq4_0` | 748.43 → 707.81 | 674.77 → 652.40 | 567.81 → 566.80 |
| `tq3_0` | 664.91 → 664.92 | 633.35 → 636.14 | 547.49 → 552.37 |

Honest reading:

- The fast path is a **~3× decode speedup at 16K depth** (6.5 → 19.5 t/s for
  the default `tq4_0`), taking long-context decode from unusable to usable on
  the 8 GB tier. At zero depth the dequant cost barely matters, so the gain is
  small there, as expected.
- TurboQuant decode at 16K depth is still ~1.8× slower than `q8_0`
  (19.5 vs 35.7 t/s): decoding a rotation-based format simply costs more than
  per-element `q8_0`. That is the remaining price of the ~25% V-cache VRAM
  saving; `TODO(v1.2): revisit if a tighter warp mapping or shared-memory
  staging closes more of the gap.`
- `pp512` moved within a few percent in both directions (the `tq4_0` depth-0
  drift of −5% reproduces the control's thermal noise band on this laptop, and
  the prefill code path is untouched by 0007); prefill with TurboQuant V sits
  at ~85–95% of the q8_0 baseline from the KV-write quantizer cost.
- Numerics before vs after 0007 are identical within fp32 round-off, verified
  on-device by `vendor/run_kernel_tests.py` stage 7 (fattn NMSE ≤ 3.3e-05 vs
  the CPU reference in both builds; table in
  [`../vendor/README.md`](../vendor/README.md)).

## Quality battery: `suiban bench kv`

Full battery on the same machine and model, TurboQuant binary at
`~/.bonsai/bin/cuda` (patches 0001–0007): real `llama-perplexity` (6 ×
2048-token chunks of a bundled synthetic corpus: absolute PPL on synthetic
text is meaningless, deltas are the signal), needle retrieval at 10/50/90%
depths of 4k/8k/16k-token haystacks and the canned ~30-turn agentic replay
(6 early-fact probes, answers generated live).

```sh
suiban bench kv    # report: ~/.bonsai/reports/bench-kv-<date>.md
```

Result (2026-07-21, slot ctx 21504 booted on the 8 GB card, so the full 16k
needle ladder ran; the corpus is deterministically shuffled real words,
unpredictable by construction, hence the large absolute PPL):

| V cache (K=q8_0) | PPL (llama-perplexity) | delta vs q8_0 | needle 4k | needle 8k | needle 16k | replay |
|---|---|---|---|---|---|---|
| `tq4_0` (default) | 1363.54 ± 67.8 | +0.11% | pass 3/3 | pass 3/3 | pass 3/3 | 6/6 |
| `tq3_0` (aggressive) | 1361.15 ± 67.4 | -0.07% | pass 3/3 | pass 3/3 | pass 3/3 | 6/6 |
| `q4_0` | 1361.04 ± 67.8 | -0.08% | pass 3/3 | pass 3/3 | pass 3/3 | 6/6 |
| `q8_0` (baseline) | 1362.08 ± 67.8 | +0.00% | pass 3/3 | pass 3/3 | pass 3/3 | 6/6 |

Honest reading: every config is statistically indistinguishable on PPL (deltas
two orders of magnitude inside the ±5% error bars), retrieves the planted
needle at every depth of every haystack size up to 16k tokens and answers all
six replay probes. On this machine TurboQuant V-cache is **measured
quality-neutral** while saving ~25% of V-cache VRAM (`tq4_0` vs `q8_0`),
consistent with the integration pass (which measured PPL deltas inside error
bars on a different synthetic corpus), the paper's LongBench tie and community
reports. A synthetic battery cannot prove there is no real-workload
regression; it can only fail to find one.

## Kernel-level numbers

- CPU reference MSE/cosine vs the paper's targets, distribution-robustness
  envelopes (heavy-tailed and channel-outlier inputs, including the honest
  small-channel cost analysis), head-dim row layout checks and the CUDA
  algorithm-equivalence harness: `vendor/run_kernel_tests.py` stages 1–6,
  documented in [`../vendor/README.md`](../vendor/README.md).
- Compiled-CUDA on-device deviations (dequantize rows, SET_ROWS quantize,
  FLASH_ATTN_EXT vec + prefill): stage 7 / `vendor/tools/tq_cuda_numeric.cpp`,
  measured table in [`../vendor/README.md`](../vendor/README.md).

## TTFT hot path + soak (audit 2026-07-22)

This section measures the **CPU-side, pre-first-token orchestration work** the chat
router does in `routers/chat.py::_prepare_loop` before the request is handed to
llama-server: the memory-package injection functions, not the model. It is separate
from the KV-cache numbers above: no GPU, no model, pure Python.

**Measurement rig:** CPython 3.12.13, Linux x86_64, 16-core dev box (same laptop as
above). Harness: [`../tests/perf/bench_ttft.py`](../tests/perf/bench_ttft.py), run via
`python tests/perf/bench_ttft.py`. It reimplements the pre-audit versions verbatim so
before/after run on identical data in one process; each number is the mean over 100
calls via `time.perf_counter`. Laptop thermals give a few-percent run-to-run drift;
the deltas here are 5×–100×, well outside the noise.

### Two O(N) fixes on the per-request path

Both functions ran on **every** chat request (the skill list via
`_inject_skill_context`, the budget guard via the overflow ladder).

| Hot-path function | Before | After | Delta |
|---|---|---|---|
| `SkillStore.list()`, 50 skills | 3241 µs/call (glob + read + parse every `SKILL.md` + `meta.json`) | 666 µs/call (stat-only signature check, cache hit) | **≈4.9× / −79%** |
| `enforce_context_budget`, 400-message conversation | 16 100 µs/call (O(M²): re-`estimate_tokens` over the whole list per popped block and per deleted message) | 166 µs/call (single pass: per-message estimates once + running total) | **≈97× / −99%** |

`cProfile` over 100 hot-path iterations (skills.list → build_skill_context →
enforce_context_budget), cumulative time in the memory package, confirms the shape of
the win: `estimate_tokens` went from **37 900 calls / 8.60 s cumulative** (the O(M²)
budget dominated everything) to falling out of the top of the profile entirely; the
new `enforce_context_budget` is ~0.07 s cumulative and `skills.list()`'s cost is now
the stat-only `_signature` check. (cProfile's per-call instrumentation inflates
absolute times vs the wall-clock table above; read it for the *distribution*, not the
magnitude.)

- **`SkillStore.list()` cache** (`memory/skills.py`): the parsed list is cached and
  invalidated by a cheap stat-only signature: a process-local write generation
  (bumped by `put`/`delete`/`mark_verified`, so in-process writes are instant and
  correct regardless of mtime granularity) mixed with each skill's `SKILL.md`
  mtime+size and `meta.json` mtime. A newly saved skill, a delete, a verification
  flip, a hand-dropped directory and an out-of-band on-disk edit all invalidate on
  the next `list()`; correctness is pinned by
  `tests/test_memory.py::test_skill_list_cache_reflects_writes_and_external_changes`.
- **`enforce_context_budget` single pass** (`memory/injection.py`): per-message token
  estimates are computed once into a parallel list with a running total; each block
  pop or message deletion adjusts the total by one message's delta instead of
  re-summing. The trim ladder and the `context_trimmed` notice are unchanged.
  Byte-for-byte equivalence to the old O(M²) implementation is pinned across five slot
  sizes by
  `tests/test_context_budget.py::test_single_pass_matches_reference_on_400_message_conversation`.

### In-context compression is a deliberate, inherent cost, not a bug

At the 70% trigger (`compression.TRIGGER_FRACTION`) `_prepare_loop` may run a
utility-model summarization round-trip **before the first token**. This is intentional
and is NOT on the optimization list: the summary must exist before the conversation can
proceed within budget, so the round-trip is a correctness cost, not overhead to remove.
It is gated (fires only at ≥70% estimated usage, `should_compress`) and best-effort
wrapped at the call site (`routers/chat.py`: any `httpx`/`ValueError`/`KeyError`
degrades to "no compression" and the request still runs, never fails). The `compression`
SSE event surfaces it. Left as-is by design.

### Bounded in-memory state (leak hunt)

- `reflection._EXCHANGE_COUNTS` (per-session reflection rate-limit key) was an
  unbounded `dict`; a long-lived server seeing many distinct `session_id`s would grow
  it without bound. Now a bounded LRU (`OrderedDict`, cap `MAX_TRACKED_SESSIONS =
  4096`, oldest evicted). Evicting an idle session only costs it one extra reflection
  if it ever returns, the same benign cost as a restart. Pinned by
  `tests/test_reflection.py::test_exchange_counter_is_bounded_lru` and
  `test_lru_touch_protects_recently_used_session`.
- Verified already bounded (no change): the llama backend stderr ring
  (`llama/backend.py`, `deque(maxlen=STDERR_RING_LINES=200)`) and the MCP client
  stderr tail (`mcp/client.py`, `deque(maxlen=50)`).

### Live RSS + VRAM soak (run against the live stack)

The micro-benchmarks above are modelless. The 200-turn RSS growth check and the
repeated-Ultra VRAM/process-leak check need the real server (a loaded model, real
slots) and are run separately against a live `suiban serve`. Driver script and
sampling commands live with the audit; they sample `VmRSS` from
`/proc/<pid>/status` and VRAM from `nvidia-smi` across 200 chat turns on one session
and across repeated Ultra spawns, looking for monotonic growth that does not plateau.

## Other backends

- **CPU**: reference kernels validated by stages 1–6; no speed claims made
  (scalar reference `vec_dot`, `TODO(v1.1)` AVX2 if profiling warrants).
- **Metal**: no Metal TurboQuant kernels exist and this project has no macOS
  hardware. Nothing here is validated for Metal, `TODO(v1.1)`. `suiban
  install turboquant` skips Metal with a notice; Apple Silicon users run the
  q8_0/q8_0 fallback.
- **HIP/ROCm**: out of scope for v1; the 0007 fast path is compiled out on HIP
  by design (closed-form path would be used), never built or run here.
