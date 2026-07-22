#!/usr/bin/env python3
"""Validate the TurboQuant TQ3_0/TQ4_0 CPU kernels in the patched fork build.

Stdlib-only (ctypes + the fork's built binaries). Stages:

1. Runs the fork's `test-quantize-fns` (covers ALL ggml types, so it also proves
   the patchset did not regress existing quants). Extracts the tq3_0/tq4_0 lines.

2. Type registration metadata (names, block sizes, type sizes).

3. Baseline MSE harness: quantizes and dequantizes N_BLOCKS random N(0,1)
   32-value blocks through the real C kernels (via ctypes on libggml-base) and
   checks:
     * relative MSE against the TurboQuant paper targets
       (b=3: <= 0.0345, b=4: <= 0.0096 — arXiv:2504.19874, community-measured
       0.034 / 0.0093),
     * round-trip cosine similarity (theory: ~sqrt(1-MSE), i.e. ~0.9827 for TQ3
       and ~0.9952 for TQ4 on synthetic Gaussian data; the community gist's
       ~0.997 figure was measured on real KV tensors, which quantize better
       than white noise).

4. Distribution robustness (property-style, 10 seeds, worst case reported):
   heavy-tailed (Student-t, df=3) and per-channel-outlier (x100 spikes on 2 of
   32 coordinates) inputs. HONEST expectation: the paper's MSE targets hold for
   the rotated-Gaussian model only. With outliers, the block RMS scale is
   dominated by the spikes and the rotated coordinates are no longer i.i.d.
   N(0,1) — the N(0,1) Lloyd-Max codebook is mismatched and relative MSE
   DEGRADES. We measure that degradation and assert measured envelopes
   (documented at the constants below), we do not pretend white-noise MSE holds.

5. Head-dim rows (64/96/128/256, multi-block rows): quantizes row-shaped
   tensors like real V caches, verifies byte-layout invariance (rows of 32-value
   blocks quantize identically to the same data as one flat row) and Gaussian
   round-trip quality per row length.

6. CUDA-dequant algorithm equivalence, host-side, on EVERY case above: the
   CUDA kernels (patches 0005-0007) do not invert the rotation with the CPU's
   sequential butterfly — fattn's dequantize_V evaluates the closed-form
   parity sum y[j] = d/sqrt(32) * sign[j] * sum_k (-1)^popc(j&k) * C[idx_k]
   (its reference/fallback path; the 0007 warp-shfl fast path computes the
   same butterfly cooperatively), and convert.cu uses a warp shfl-xor
   butterfly (same computation graph). This stage re-implements the
   parity-sum + bit-unpack + sign-mask math in Python, runs it on real
   C-quantized bytes from all distributions, head dims and seeds, and
   compares against the C dequantizer. It validates the algorithm and bit
   layout the CUDA kernels implement, NOT the compiled kernels — stage 7
   does that.

7. Compiled-CUDA numeric run ON A GPU (optional, needs build-cuda + a device):
   compiles tools/tq_cuda_numeric.cpp against the CUDA build and runs the
   dequantize-rows kernels, the SET_ROWS KV-write quantizer and the
   flash-attention V-dequant path on-device against the CPU reference,
   printing max deviations. Skipped with an honest notice when the CUDA build,
   a host compiler or a GPU is unavailable (pass --require-cuda to make that a
   failure instead).

Run `apply_patches.py --clone --cpu-only` first (and `--cuda` for stage 7).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parent
BUILD_BIN = VENDOR_DIR / "llama.cpp" / "build-cpu" / "bin"
CUDA_BUILD_BIN = VENDOR_DIR / "llama.cpp" / "build-cuda" / "bin"
CODEBOOKS_JSON = VENDOR_DIR / "codebooks" / "tq_codebooks.json"
TOOLS_DIR = VENDOR_DIR / "tools"

GGML_TYPE_TQ3_0 = 43
GGML_TYPE_TQ4_0 = 44

N_BLOCKS = 100_000
BLOCK = 32
SEED = 20260721

# type id -> (name, bytes/block, MSE target, cosine-similarity floor)
SPECS = {
    GGML_TYPE_TQ3_0: ("tq3_0", 14, 0.0345, 0.980),
    GGML_TYPE_TQ4_0: ("tq4_0", 18, 0.0096, 0.994),
}

# -- stage 4 constants ---------------------------------------------------------
N_SEEDS = 10
N_BLOCKS_ROBUST = 4000  # per seed per distribution per type

# Per-channel outlier model: 2 of 32 coordinates carry x100 spikes (real KV
# tensors show a few systematically-large channels; x100 is a stress case).
OUTLIER_CHANNELS = (5, 21)
OUTLIER_SCALE = 100.0

# MEASURED envelopes (worst over 10 seeds x 4000 blocks on this harness, with
# ~20-25% headroom), NOT white-noise theory. What we actually measured, and why:
#
# - student_t3 (worst measured: 0.0314 / 0.0086 — statistically the same as the
#   white-noise 0.0325 / 0.0089): a df=3 tail sample inflates the block RMS,
#   and after the orthogonal rotation every rotated coordinate is bounded by
#   ~sqrt(32)/norm — the codebook never sees the raw tail. The RMS scale plus
#   rotation absorb heavy tails; TOTAL relative MSE does not degrade.
#
# - outlier, TOTAL relative MSE (worst measured: 0.0206 / 0.0089 — at or BELOW
#   white noise): the two x100 channels carry ~99.85% of the block energy, so
#   after RMS scaling the rotated vector collapses onto ~two magnitudes
#   +/-|g1+g2|/sqrt(g1^2+g2^2) and +/-|g1-g2|/sqrt(g1^2+g2^2), all bounded by
#   sqrt(2) — inside the fine center of the codebook, with none of the Gaussian
#   tail-clipping loss. Total MSE is dominated by how well the OUTLIER energy
#   quantizes, and that quantizes fine.
#
# - outlier, SMALL-CHANNEL relative MSE (the honest cost, asserted as a BAND):
#   the 30 non-outlier channels hold ~0.15% of the block energy while the
#   inverse rotation hands a large share of the quantization error back to
#   them, so relative to their OWN energy the small channels come back as
#   noise: measured 4.3-5.0x (tq3_0) and 2.9-3.0x (tq4_0) their own energy —
#   reconstruction WORSE than zeroing them. This is the real
#   per-channel-outlier failure mode of a block-RMS format; it simply does not
#   show up in the total because the outlier channels dominate. The band's
#   lower bound (1.0 = worse than zeroing) is asserted too: if a future kernel
#   change made small channels survive, this test should fail so the docs get
#   rewritten.
#
# Deviating from these envelopes on a re-run means kernel behavior changed —
# investigate, do not bump the number.
ROBUST_MSE_ENVELOPES: dict[tuple[str, str], float] = {
    ("student_t3", "tq3_0"): 0.036,
    ("student_t3", "tq4_0"): 0.010,
    ("outlier", "tq3_0"): 0.026,
    ("outlier", "tq4_0"): 0.011,
}
# (low, high) band for the outlier small-channel relative MSE — see above.
OUTLIER_SMALL_CHANNEL_BANDS: dict[str, tuple[float, float]] = {
    "tq3_0": (1.0, 6.5),
    "tq4_0": (1.0, 4.0),
}

HEAD_DIMS = (64, 96, 128, 256)
N_ROWS_HEAD_DIM = 256

# Parity-check subset sizes (pure-Python parity sums are O(32^2) per block)
N_BLOCKS_PARITY_BASE = 2000
N_BLOCKS_PARITY_CASE = 200


def fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    sys.exit(1)


# -- input distributions -------------------------------------------------------
def gen_gauss(rng: random.Random, n: int) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(n)]


def gen_student_t3(rng: random.Random, n: int) -> list[float]:
    """Student-t, df=3: t = g / sqrt(chi2_3 / 3). Heavy tails, infinite kurtosis."""
    out = []
    for _ in range(n):
        chi2 = sum(rng.gauss(0.0, 1.0) ** 2 for _ in range(3))
        # chi2 of 3 gaussians is ~never 0; guard anyway to keep the harness total
        out.append(rng.gauss(0.0, 1.0) / math.sqrt(max(chi2, 1e-12) / 3.0))
    return out


def gen_outlier(rng: random.Random, n: int) -> list[float]:
    """White Gaussian with x100 spikes on 2 fixed per-block channels."""
    return [
        rng.gauss(0.0, 1.0) * (OUTLIER_SCALE if (i % BLOCK) in OUTLIER_CHANNELS else 1.0)
        for i in range(n)
    ]


DISTRIBUTIONS = {
    "gauss": gen_gauss,
    "student_t3": gen_student_t3,
    "outlier": gen_outlier,
}


# -- stages -------------------------------------------------------------------
def stage1_quantize_fns() -> None:
    exe = BUILD_BIN / "test-quantize-fns"
    if not exe.is_file():
        fail(f"{exe} not found — run apply_patches.py --clone --cpu-only first")
    print("== stage 1: fork test-quantize-fns (all types) ==")
    res = subprocess.run([str(exe), "-v"], capture_output=True, text=True)
    tq_lines = [
        ln
        for ln in res.stdout.splitlines()
        if "tq3_0" in ln or "tq4_0" in ln or "tests failed" in ln
    ]
    for ln in tq_lines:
        print(f"  {ln}")
    if res.returncode != 0:
        print(res.stdout[-2000:])
        fail(f"test-quantize-fns exited {res.returncode}")
    for needle in (
        "tq3_0 absolute quantization error:    ok",
        "tq4_0 absolute quantization error:    ok",
        "tq3_0 dot product error:              ok",
        "tq4_0 dot product error:              ok",
    ):
        if not any(needle in ln for ln in tq_lines):
            fail(f"expected line missing from test-quantize-fns output: '{needle}'")
    print("  stage 1 PASS (exit 0, all types)")


def load_ggml() -> ctypes.CDLL:
    lib_path = BUILD_BIN / "libggml-base.so"
    if not lib_path.is_file():
        fail(f"{lib_path} not found — build with BUILD_SHARED_LIBS=ON via apply_patches.py")
    lib = ctypes.CDLL(str(lib_path))
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.ggml_type_name.restype = ctypes.c_char_p
    lib.ggml_type_name.argtypes = [ctypes.c_int]
    lib.ggml_blck_size.restype = ctypes.c_int64
    lib.ggml_blck_size.argtypes = [ctypes.c_int]
    lib.ggml_type_size.restype = ctypes.c_size_t
    lib.ggml_type_size.argtypes = [ctypes.c_int]
    for fn in ("dequantize_row_tq3_0", "dequantize_row_tq4_0"):
        f = getattr(lib, fn)
        f.restype = None
        f.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
    return lib


def stage2_registration(lib: ctypes.CDLL) -> None:
    print("== stage 2: type registration metadata ==")
    for type_id, (name, bytes_per_block, _, _) in SPECS.items():
        got_name = lib.ggml_type_name(type_id).decode()
        got_blck = lib.ggml_blck_size(type_id)
        got_size = lib.ggml_type_size(type_id)
        print(
            f"  type {type_id}: name={got_name} blck_size={got_blck} type_size={got_size}"
            f" ({got_size * 8 / got_blck:.2f} bpw)"
        )
        if got_name != name:
            fail(f"type {type_id} name is '{got_name}', expected '{name}'")
        if got_blck != BLOCK or got_size != bytes_per_block:
            fail(
                f"type {name}: block {got_blck}/{got_size} B, expected {BLOCK}/{bytes_per_block} B"
            )
    print("  stage 2 PASS")


def quantize(
    lib: ctypes.CDLL, type_id: int, src: ctypes.Array, *, nrows: int = 1, n_per_row: int = 0
) -> ctypes.Array:
    """ggml_quantize_chunk through the real C kernel; returns the raw block buffer."""
    n = len(src)
    n_per_row = n_per_row or n
    assert nrows * n_per_row == n
    bytes_per_block = SPECS[type_id][1]
    qbuf = ctypes.create_string_buffer((n // BLOCK) * bytes_per_block)
    written = lib.ggml_quantize_chunk(type_id, src, qbuf, 0, nrows, n_per_row, None)
    if written != len(qbuf):
        fail(f"{SPECS[type_id][0]}: quantize_chunk wrote {written} B, expected {len(qbuf)} B")
    return qbuf


def dequantize(lib: ctypes.CDLL, type_id: int, qbuf: ctypes.Array, n: int) -> ctypes.Array:
    out = (ctypes.c_float * n)()
    getattr(lib, f"dequantize_row_{SPECS[type_id][0]}")(qbuf, out, n)
    return out


def roundtrip_stats(
    lib: ctypes.CDLL, type_id: int, src: ctypes.Array
) -> tuple[float, float, ctypes.Array]:
    """(relative MSE, cosine similarity, quantized buffer) for one round trip."""
    n = len(src)
    qbuf = quantize(lib, type_id, src)
    out = dequantize(lib, type_id, qbuf, n)
    err_sq = x_sq = y_sq = dot = 0.0
    for i in range(n):
        x = src[i]
        y = out[i]
        d = x - y
        err_sq += d * d
        x_sq += x * x
        y_sq += y * y
        dot += x * y
    mse = err_sq / x_sq  # relative MSE (scale-invariant)
    cos = dot / math.sqrt(x_sq * y_sq)
    return mse, cos, qbuf


def stage3_mse(lib: ctypes.CDLL) -> None:
    n = N_BLOCKS * BLOCK
    print(f"== stage 3: MSE harness ({N_BLOCKS} random N(0,1) {BLOCK}-blocks, seed {SEED}) ==")
    rng = random.Random(SEED)
    src = (ctypes.c_float * n)(*gen_gauss(rng, n))

    ok = True
    for type_id, (name, _, mse_target, cos_floor) in SPECS.items():
        mse, cos, _ = roundtrip_stats(lib, type_id, src)
        mse_ok = mse <= mse_target
        cos_ok = cos >= cos_floor
        ok = ok and mse_ok and cos_ok
        print(
            f"  {name}: relative MSE {mse:.6f} (target <= {mse_target}) "
            f"{'ok' if mse_ok else 'FAILED'}"
        )
        print(
            f"  {name}: round-trip cosine similarity {cos:.6f} (floor {cos_floor}) "
            f"{'ok' if cos_ok else 'FAILED'}"
        )
    if not ok:
        fail("stage 3 thresholds not met")
    print("  stage 3 PASS")


def small_channel_mse(src: ctypes.Array, out: ctypes.Array) -> float:
    """Relative MSE restricted to the NON-outlier channels (their own energy)."""
    err_sq = x_sq = 0.0
    for i in range(len(src)):
        if (i % BLOCK) in OUTLIER_CHANNELS:
            continue
        d = src[i] - out[i]
        err_sq += d * d
        x_sq += src[i] * src[i]
    return err_sq / x_sq


def stage4_distribution_robustness(lib: ctypes.CDLL) -> None:
    print(
        f"== stage 4: distribution robustness "
        f"({N_SEEDS} seeds x {N_BLOCKS_ROBUST} blocks per case, worst case asserted) =="
    )
    print(
        "  honest note: all envelopes here are MEASURED on this harness (see the\n"
        "  comments at ROBUST_MSE_ENVELOPES), not white-noise theory. Measured\n"
        "  story: TOTAL relative MSE does NOT degrade under heavy tails or x100\n"
        "  channel outliers (the RMS scale + rotation keep rotated coordinates\n"
        "  bounded), but with channel outliers the NON-outlier channels come back\n"
        "  as noise relative to their own energy — that cost is asserted as a\n"
        "  band below, not hidden inside the healthy-looking total."
    )
    n = N_BLOCKS_ROBUST * BLOCK
    ok = True
    for dist_name in ("student_t3", "outlier"):
        gen = DISTRIBUTIONS[dist_name]
        for type_id, (name, _, gauss_target, _) in SPECS.items():
            worst_mse = 0.0
            worst_seed = None
            cos_min = 1.0
            worst_small = 0.0
            best_small = math.inf
            for seed in range(SEED, SEED + N_SEEDS):
                rng = random.Random(seed)
                src = (ctypes.c_float * n)(*gen(rng, n))
                mse, cos, qbuf = roundtrip_stats(lib, type_id, src)
                cos_min = min(cos_min, cos)
                if mse > worst_mse:
                    worst_mse = mse
                    worst_seed = seed
                if dist_name == "outlier":
                    small = small_channel_mse(src, dequantize(lib, type_id, qbuf, n))
                    worst_small = max(worst_small, small)
                    best_small = min(best_small, small)
            envelope = ROBUST_MSE_ENVELOPES[(dist_name, name)]
            mse_ok = worst_mse <= envelope
            ok = ok and mse_ok
            print(
                f"  {dist_name:>10} {name}: worst relative MSE {worst_mse:.6f} "
                f"(seed {worst_seed}, envelope <= {envelope}, "
                f"{worst_mse / gauss_target:.2f}x the white-noise target) "
                f"min cosine {cos_min:.6f} {'ok' if mse_ok else 'FAILED'}"
            )
            if dist_name == "outlier":
                lo, hi = OUTLIER_SMALL_CHANNEL_BANDS[name]
                small_ok = lo <= best_small and worst_small <= hi
                ok = ok and small_ok
                print(
                    f"  {dist_name:>10} {name}: small-channel relative MSE "
                    f"{best_small:.2f}..{worst_small:.2f} (band {lo}..{hi}) — non-outlier "
                    f"channels reconstruct WORSE than zeroing; documented format cost "
                    f"{'ok' if small_ok else 'FAILED'}"
                )
    if not ok:
        fail("stage 4 measured envelopes exceeded — kernel behavior changed, investigate")
    print("  stage 4 PASS (measured-envelope assertions, not white-noise theory)")


def stage5_head_dim_rows(lib: ctypes.CDLL) -> None:
    print(
        f"== stage 5: head-dim rows ({N_ROWS_HEAD_DIM} rows x D for D in {HEAD_DIMS}, "
        f"seed {SEED}) =="
    )
    ok = True
    for head_dim in HEAD_DIMS:
        n = N_ROWS_HEAD_DIM * head_dim
        rng = random.Random(SEED)
        values = gen_gauss(rng, n)
        src = (ctypes.c_float * n)(*values)
        for type_id, (name, _, mse_target, cos_floor) in SPECS.items():
            # (a) quantize as real row-shaped tensors (nrows x head_dim)
            q_rows = quantize(lib, type_id, src, nrows=N_ROWS_HEAD_DIM, n_per_row=head_dim)
            # (b) layout invariance: identical bytes to the same data as one flat row
            #     (blocks are independent; row shape must not change the encoding)
            q_flat = quantize(lib, type_id, src)
            if q_rows.raw != q_flat.raw:
                fail(f"{name} D={head_dim}: row-shaped and flat quantization bytes differ")
            # (c) round trip quality (Gaussian rows -> white-noise targets apply)
            out = dequantize(lib, type_id, q_rows, n)
            err_sq = x_sq = y_sq = dot = 0.0
            for i in range(n):
                x = src[i]
                y = out[i]
                d = x - y
                err_sq += d * d
                x_sq += x * x
                y_sq += y * y
                dot += x * y
            mse = err_sq / x_sq
            cos = dot / math.sqrt(x_sq * y_sq)
            case_ok = mse <= mse_target and cos >= cos_floor
            ok = ok and case_ok
            blocks_per_row = head_dim // BLOCK
            print(
                f"  {name} D={head_dim} ({blocks_per_row} blocks/row): layout ok, "
                f"relative MSE {mse:.6f} (<= {mse_target}), cosine {cos:.6f} "
                f"(>= {cos_floor}) {'ok' if case_ok else 'FAILED'}"
            )
    if not ok:
        fail("stage 5 head-dim row thresholds not met")
    print("  stage 5 PASS")


# -------- stage 6: CUDA dequant algorithm equivalence (host-side model) --------

# bit j set -> sign[j] = -1; identical to TQ_RHT_SIGN_MASK in the CUDA patch
# (turboquant.cuh) and to the golden-ratio-hash sign table in the C reference.
TQ_RHT_SIGN_MASK = 0x696B4B4A
RSQRT32 = 1.0 / math.sqrt(32.0)
# float64 Python model vs float32 C kernels; inputs are O(1) after the RMS scale
STAGE6_ATOL = 1e-5
# Outlier blocks carry d ~ O(100): absolute round-off scales with d, so the
# tolerance for those cases scales accordingly (same relative accuracy).
STAGE6_ATOL_OUTLIER = 1e-3


def _unpack_codes_tq3_0(qs: bytes) -> list[int]:
    """3-bit code extraction, same bit math as ggml_cuda_tq3_0_get_code."""
    codes = []
    for k in range(32):
        qp_off = 3 * (k >> 3)
        bit = 3 * (k & 7)
        v = qs[qp_off + (bit >> 3)] >> (bit & 7)
        if (bit & 7) > 5:
            v |= qs[qp_off + (bit >> 3) + 1] << (8 - (bit & 7))
        codes.append(v & 7)
    return codes


def _unpack_codes_tq4_0(qs: bytes) -> list[int]:
    """4-bit code extraction, same bit math as ggml_cuda_tq4_0_get_code."""
    return [(qs[k >> 1] >> (4 * (k & 1))) & 0xF for k in range(32)]


def _dequant_parity_sum(d: float, codes: list[int], centroids: list[float]) -> list[float]:
    """The CUDA kernels' closed-form inverse RHT (ggml_cuda_tq_dequant_elem)."""
    c = [centroids[i] for i in codes]
    dn = d * RSQRT32
    out = []
    for j in range(32):
        s = 0.0
        for k in range(32):
            s += -c[k] if (j & k).bit_count() & 1 else c[k]
        sign = -1.0 if (TQ_RHT_SIGN_MASK >> j) & 1 else 1.0
        out.append(s * sign * dn)
    return out


def _parity_check_case(
    lib: ctypes.CDLL,
    type_id: int,
    src: ctypes.Array,
    centroids: list[float],
    atol: float,
) -> float:
    """Max |parity-sum model − C reference| over all blocks of src; fails on excess."""
    name, bytes_per_block, _, _ = SPECS[type_id]
    n = len(src)
    n_blocks = n // BLOCK
    qbuf = quantize(lib, type_id, src)
    ref = dequantize(lib, type_id, qbuf, n)
    unpack = _unpack_codes_tq3_0 if type_id == GGML_TYPE_TQ3_0 else _unpack_codes_tq4_0
    raw = qbuf.raw
    max_diff = 0.0
    for b in range(n_blocks):
        blk = raw[b * bytes_per_block : (b + 1) * bytes_per_block]
        (d,) = struct.unpack("<e", blk[:2])
        model = _dequant_parity_sum(d, unpack(blk[2:]), centroids)
        for j in range(BLOCK):
            max_diff = max(max_diff, abs(model[j] - ref[b * BLOCK + j]))
    if max_diff > atol:
        fail(
            f"{name}: CUDA dequant algorithm diverges from the C reference "
            f"(max diff {max_diff:.3e} > atol {atol})"
        )
    return max_diff


def stage6_cuda_algorithm(lib: ctypes.CDLL) -> None:
    print(
        "== stage 6: CUDA dequant algorithm equivalence (host-side model, "
        "all distributions + head dims) =="
    )
    books = json.loads(CODEBOOKS_JSON.read_text())
    centroids = {
        GGML_TYPE_TQ3_0: [float(v) for v in books["tq3_0"]["emitted_centroids"]],
        GGML_TYPE_TQ4_0: [float(v) for v in books["tq4_0"]["emitted_centroids"]],
    }

    for type_id, (name, _, _, _) in SPECS.items():
        # baseline: white Gaussian, the original stage-4 case
        rng = random.Random(SEED)
        src = (ctypes.c_float * (N_BLOCKS_PARITY_BASE * BLOCK))(
            *gen_gauss(rng, N_BLOCKS_PARITY_BASE * BLOCK)
        )
        max_diff = _parity_check_case(lib, type_id, src, centroids[type_id], STAGE6_ATOL)
        worst = {"gauss": max_diff}

        # every distribution x every seed (reduced block counts: pure-Python
        # parity sums are O(32^2) per block)
        for dist_name in ("student_t3", "outlier"):
            gen = DISTRIBUTIONS[dist_name]
            atol = STAGE6_ATOL_OUTLIER if dist_name == "outlier" else STAGE6_ATOL
            dist_worst = 0.0
            for seed in range(SEED, SEED + N_SEEDS):
                rng = random.Random(seed)
                src = (ctypes.c_float * (N_BLOCKS_PARITY_CASE * BLOCK))(
                    *gen(rng, N_BLOCKS_PARITY_CASE * BLOCK)
                )
                dist_worst = max(
                    dist_worst,
                    _parity_check_case(lib, type_id, src, centroids[type_id], atol),
                )
            worst[dist_name] = dist_worst

        # head-dim rows (row-shaped quantization, multi-block rows)
        dim_worst = 0.0
        for head_dim in HEAD_DIMS:
            n_rows = max(1, (N_BLOCKS_PARITY_CASE * BLOCK) // head_dim)
            n = n_rows * head_dim
            rng = random.Random(SEED)
            src = (ctypes.c_float * n)(*gen_gauss(rng, n))
            dim_worst = max(
                dim_worst, _parity_check_case(lib, type_id, src, centroids[type_id], STAGE6_ATOL)
            )
        worst["head_dims"] = dim_worst

        summary = "  ".join(f"{k}={v:.3e}" for k, v in worst.items())
        print(f"  {name}: max |parity-sum model - C reference| per case: {summary}  ok")
    print(
        "  stage 6 PASS (algorithm + bit layout on all cases; compiled kernels are stage 7)"
    )


# -------- stage 7: compiled CUDA kernels on a GPU (optional) -------------------
def stage7_compiled_cuda(require: bool) -> None:
    print("== stage 7: compiled CUDA kernels on GPU (tools/tq_cuda_numeric) ==")

    def skip(reason: str) -> None:
        if require:
            fail(f"--require-cuda set but stage 7 cannot run: {reason}")
        print(f"  SKIPPED: {reason}")
        print(
            "  (build the CUDA backend with apply_patches.py --cuda on a machine with "
            "an NVIDIA GPU to run this stage; stages 1-6 validate algorithm and layout "
            "but NOT the compiled kernels)"
        )

    src = TOOLS_DIR / "tq_cuda_numeric.cpp"
    if not src.is_file():
        skip(f"{src} not found")
        return
    if not (CUDA_BUILD_BIN / "libggml-cuda.so").is_file():
        skip("no CUDA build (vendor/llama.cpp/build-cuda) — run apply_patches.py --cuda")
        return
    cuda_home = Path(os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda")))
    if not (cuda_home / "include" / "cuda_runtime.h").is_file():
        skip(f"CUDA toolkit headers not found under {cuda_home}")
        return

    exe = TOOLS_DIR / "tq_cuda_numeric"
    include_dirs = [
        VENDOR_DIR / "llama.cpp" / "ggml" / "include",
        cuda_home / "include",
    ]
    compile_cmd = [
        "c++",
        "-O2",
        "-std=c++17",
        str(src),
        "-o",
        str(exe),
        *[f"-I{d}" for d in include_dirs],
        f"-L{CUDA_BUILD_BIN}",
        f"-L{cuda_home}/lib64",
        "-lggml",
        "-lggml-base",
        "-lggml-cpu",
        "-lggml-cuda",
        "-lcudart",
        f"-Wl,-rpath,{CUDA_BUILD_BIN}",
        f"-Wl,-rpath,{cuda_home}/lib64",
    ]
    print(f"  $ {' '.join(compile_cmd)}")
    comp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp.returncode != 0:
        skip(f"host compile failed:\n{comp.stderr[-1500:]}")
        return
    res = subprocess.run([str(exe)], capture_output=True, text=True)
    for ln in res.stdout.splitlines():
        print(f"  {ln}")
    if res.returncode == 3:
        skip("no CUDA device available at runtime")
        return
    if res.returncode != 0:
        print(res.stderr[-1500:])
        fail(f"tq_cuda_numeric exited {res.returncode}")
    print("  stage 7 PASS (compiled kernels, on-device, vs CPU reference)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail (instead of skip) when the compiled-CUDA stage 7 cannot run",
    )
    args = parser.parse_args()

    stage1_quantize_fns()
    lib = load_ggml()
    stage2_registration(lib)
    stage3_mse(lib)
    stage4_distribution_robustness(lib)
    stage5_head_dim_rows(lib)
    stage6_cuda_algorithm(lib)
    stage7_compiled_cuda(args.require_cuda)
    print("ALL KERNEL TESTS PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
