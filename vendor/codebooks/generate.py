#!/usr/bin/env python3
"""Generate the TurboQuant Lloyd-Max codebooks (TQ3_0: 8 levels, TQ4_0: 16 levels).

Stdlib-only, fully deterministic. Runs before `uv sync` (no third-party deps).

Method
------
Optimal (minimum-MSE) scalar quantizer for the standard normal N(0,1) via
Lloyd-Max fixed-point iteration:

  * decision boundaries are midpoints of adjacent centroids:  t_k = (c_k + c_{k+1}) / 2
  * centroids are conditional means over their cell:          c_k = E[X | t_{k-1} < X <= t_k]

Conditional means and per-cell MSE use exact Gaussian integrals
(phi/Phi via math.erf/exp — closed forms, evaluated numerically):

  int_a^b x   phi(x) dx = phi(a) - phi(b)
  int_a^b x^2 phi(x) dx = (Phi(b) - Phi(a)) - (b phi(b) - a phi(a))

The final MSE is cross-checked with composite-Simpson numeric integration.

Cross-checks (hard assertions):
  * 8-level result vs the MIT-licensed Aaryan-Kapoor/llama.cpp `turboquant-tq3_0`
    branch constants: centroids {+-0.2451, +-0.7560, +-1.3439, +-2.1519},
    boundaries {+-0.5005, +-1.0500, +-1.7479} (and 0).
  * 16-level result vs Max (1960, IRE Trans. Inf. Theory) literature values:
    +-0.1284, +-0.3881, +-0.6568, +-0.9424, +-1.2562, +-1.6181, +-2.0690, +-2.7326.

Outputs (both checked in, regenerated only by re-running this script):
  * tq_codebooks.h    — C header consumed by the patchset sources (constants are
                        copied verbatim into patches/0001; keep them in sync).
  * tq_codebooks.json — full-precision values, MSEs, iteration counts, checks.

Emission policy:
  * TQ3_0 emits the MIT branch's exact 4-decimal constants (ported verbatim for
    bit-compatibility with the reference), after verifying our computed optimum
    matches them to <= 2e-4.
  * TQ4_0 emits our computed centroids rounded to 7 decimals; boundaries are the
    exact midpoints of the rounded centroids (encoder/decoder consistency).
  * The RHT sign-flip table (golden-ratio hash 0x9E3779B9) is emitted too, so the
    header is the single source of truth for all TurboQuant constants.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

SQRT_2PI = math.sqrt(2.0 * math.pi)

# Convergence: task requires < 1e-9; we iterate to 1e-12 for margin.
CONVERGENCE_EPS = 1e-12
MAX_ITERS = 200_000

# Literature / reference constants (positive halves).
MIT_TQ3_CENTROIDS = [0.2451, 0.7560, 1.3439, 2.1519]
MIT_TQ3_BOUNDARIES = [0.5005, 1.0500, 1.7479]  # plus the implicit 0.0
MAX1960_TQ4_CENTROIDS = [0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6181, 2.0690, 2.7326]
CROSS_CHECK_TOL = 2e-4  # literature values are 4-decimal roundings

GOLDEN_HASH = 0x9E3779B9


def phi(x: float) -> float:
    """Standard normal pdf."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def Phi(x: float) -> float:
    """Standard normal cdf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def cell_mean(a: float, b: float) -> float:
    """E[X | a < X <= b] for X ~ N(0,1). a may be -inf, b may be +inf."""
    pa = phi(a) if math.isfinite(a) else 0.0
    pb = phi(b) if math.isfinite(b) else 0.0
    mass = Phi(b) - Phi(a)
    if mass <= 0.0:
        raise ArithmeticError(f"empty cell ({a}, {b}]")
    return (pa - pb) / mass


def cell_mse(a: float, b: float, c: float) -> float:
    """int_a^b (x - c)^2 phi(x) dx  (exact closed form)."""
    pa = phi(a) if math.isfinite(a) else 0.0
    pb = phi(b) if math.isfinite(b) else 0.0
    apa = a * pa if math.isfinite(a) else 0.0
    bpb = b * pb if math.isfinite(b) else 0.0
    mass = Phi(b) - Phi(a)
    ex2 = mass - (bpb - apa)  # int x^2 phi
    ex1 = pa - pb  # int x phi
    return ex2 - 2.0 * c * ex1 + c * c * mass


def lloyd_max(levels: int) -> tuple[list[float], list[float], float, int]:
    """Symmetric Lloyd-Max quantizer for N(0,1).

    Returns (centroids ascending, boundaries ascending (levels-1 of them),
    mse, iterations).
    """
    if levels % 2 != 0:
        raise ValueError("only even level counts are used here")
    # Initialize centroids at equally spaced points.
    centroids = [-3.0 + 6.0 * (i + 0.5) / levels for i in range(levels)]
    iters = 0
    while True:
        iters += 1
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(levels - 1)]
        new_centroids = []
        for i in range(levels):
            a = boundaries[i - 1] if i > 0 else -math.inf
            b = boundaries[i] if i < levels - 1 else math.inf
            new_centroids.append(cell_mean(a, b))
        delta = max(abs(n - o) for n, o in zip(new_centroids, centroids, strict=True))
        centroids = new_centroids
        if delta < CONVERGENCE_EPS:
            break
        if iters >= MAX_ITERS:
            raise ArithmeticError(f"Lloyd-Max did not converge in {MAX_ITERS} iterations")
    boundaries = [(centroids[i] + centroids[i + 1]) / 2.0 for i in range(levels - 1)]
    mse = 0.0
    for i in range(levels):
        a = boundaries[i - 1] if i > 0 else -math.inf
        b = boundaries[i] if i < levels - 1 else math.inf
        mse += cell_mse(a, b, centroids[i])
    return centroids, boundaries, mse, iters


def simpson_mse(
    centroids: list[float], boundaries: list[float], span: float = 12.0, n: int = 200_000
) -> float:
    """Cross-check MSE by composite Simpson integration of (x - q(x))^2 phi(x)."""

    def quantize(x: float) -> float:
        idx = 0
        for b in boundaries:
            if x > b:
                idx += 1
        return centroids[idx]

    def f(x: float) -> float:
        e = x - quantize(x)
        return e * e * phi(x)

    a, b = -span, span
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4.0 if i % 2 else 2.0) * f(a + i * h)
    return total * h / 3.0


def rht_signs(n: int = 32) -> list[int]:
    """Deterministic sign-flip pattern: sign[i] = -1 if (i * 0x9E3779B9) >> 31 else +1."""
    return [-1 if ((i * GOLDEN_HASH) & 0xFFFFFFFF) >> 31 else 1 for i in range(n)]


def cross_check(name: str, computed: list[float], reference: list[float], tol: float) -> float:
    worst = max(abs(c - r) for c, r in zip(computed, reference, strict=True))
    if worst > tol:
        raise AssertionError(
            f"{name}: computed values diverge from reference by {worst:.2e} (tol {tol:.0e})\n"
            f"  computed:  {[round(c, 6) for c in computed]}\n"
            f"  reference: {reference}"
        )
    return worst


def fmt_c_floats(values: list[float], per_line: int, indent: str, decimals: int) -> str:
    parts = [f"{v:+.{decimals}f}f" for v in values]
    lines = []
    for i in range(0, len(parts), per_line):
        lines.append(indent + ", ".join(parts[i : i + per_line]) + ",")
    return "\n".join(lines)


def main() -> int:
    out_dir = Path(__file__).resolve().parent

    # --- compute both codebooks -------------------------------------------------
    c8, b8, mse8, it8 = lloyd_max(8)
    c16, b16, mse16, it16 = lloyd_max(16)

    mse8_simpson = simpson_mse(c8, b8)
    mse16_simpson = simpson_mse(c16, b16)
    if abs(mse8 - mse8_simpson) > 1e-7 or abs(mse16 - mse16_simpson) > 1e-7:
        raise AssertionError(
            f"closed-form vs Simpson MSE mismatch: "
            f"8-level {mse8:.9f} vs {mse8_simpson:.9f}, "
            f"16-level {mse16:.9f} vs {mse16_simpson:.9f}"
        )

    # --- cross-check against literature / the MIT branch ------------------------
    pos8 = c8[4:]
    pos16 = c16[8:]
    dev8_c = cross_check("TQ3 centroids vs MIT branch", pos8, MIT_TQ3_CENTROIDS, CROSS_CHECK_TOL)
    dev8_b = cross_check(
        "TQ3 boundaries vs MIT branch", b8[4:], MIT_TQ3_BOUNDARIES, CROSS_CHECK_TOL
    )
    dev16 = cross_check(
        "TQ4 centroids vs Max (1960)", pos16, MAX1960_TQ4_CENTROIDS, CROSS_CHECK_TOL
    )

    # --- emitted (rounded) constants --------------------------------------------
    # TQ3: MIT branch verbatim (negative half mirrored).
    tq3_centroids = [-v for v in reversed(MIT_TQ3_CENTROIDS)] + MIT_TQ3_CENTROIDS
    tq3_boundaries = [-v for v in reversed(MIT_TQ3_BOUNDARIES)] + [0.0] + MIT_TQ3_BOUNDARIES
    # TQ4: our computed values at 7 decimals; boundaries = midpoints of rounded centroids.
    tq4_centroids = [round(v, 7) for v in c16]
    tq4_boundaries = [round((tq4_centroids[i] + tq4_centroids[i + 1]) / 2.0, 7) for i in range(15)]
    signs = rht_signs(32)

    # --- write JSON --------------------------------------------------------------
    payload = {
        "generator": "suiban/vendor/codebooks/generate.py (stdlib-only Lloyd-Max)",
        "distribution": "N(0,1), symmetric minimum-MSE scalar quantizer",
        "convergence_eps": CONVERGENCE_EPS,
        "tq3_0": {
            "levels": 8,
            "iterations": it8,
            "computed_centroids": c8,
            "computed_boundaries": b8,
            "theoretical_mse": mse8,
            "theoretical_mse_simpson_check": mse8_simpson,
            "emitted_centroids": tq3_centroids,
            "emitted_boundaries": tq3_boundaries,
            "emitted_source": "Aaryan-Kapoor/llama.cpp branch turboquant-tq3_0 (MIT), verbatim",
            "max_dev_vs_reference": max(dev8_c, dev8_b),
        },
        "tq4_0": {
            "levels": 16,
            "iterations": it16,
            "computed_centroids": c16,
            "computed_boundaries": b16,
            "theoretical_mse": mse16,
            "theoretical_mse_simpson_check": mse16_simpson,
            "emitted_centroids": tq4_centroids,
            "emitted_boundaries": tq4_boundaries,
            "emitted_source": "computed here; cross-checked vs Max (1960) literature values",
            "max_dev_vs_reference": dev16,
        },
        "rht_signs": {
            "formula": "sign[i] = -1 if ((i * 0x9E3779B9) & 0xFFFFFFFF) >> 31 else +1",
            "values": signs,
        },
    }
    json_path = out_dir / "tq_codebooks.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # --- write C header ----------------------------------------------------------
    header = f"""\
// TurboQuant Lloyd-Max codebooks — GENERATED by suiban/vendor/codebooks/generate.py.
// Do not edit by hand; re-run the generator. These constants are duplicated in
// suiban/vendor/patches/0001-*.patch (ggml-quants.c) and MUST stay in sync.
//
// TQ3_0 (8 levels): ported verbatim from the MIT-licensed
//   Aaryan-Kapoor/llama.cpp branch `turboquant-tq3_0`
//   (commit 1fb1fb3ab6b3e5c3776a362d7ac7f5985328d71b), verified here against an
//   independent Lloyd-Max solve to <= 2e-4.
// TQ4_0 (16 levels): computed here (Lloyd-Max on N(0,1), converged < 1e-12),
//   cross-checked against Max (1960), IRE Trans. Inf. Theory, to <= 2e-4.
// Boundaries are midpoints of adjacent (emitted) centroids.
//
// Theoretical MSE vs N(0,1): 8-level {mse8:.9f}, 16-level {mse16:.9f}.

#ifndef TQ_CODEBOOKS_H
#define TQ_CODEBOOKS_H

// 8-level Lloyd-Max codebook for TQ3_0 (3-bit indices)
static const float TQ3_0_CENTROIDS[8] = {{
{fmt_c_floats(tq3_centroids, 4, "    ", 4)}
}};

static const float TQ3_0_BOUNDARIES[7] = {{
{fmt_c_floats(tq3_boundaries, 4, "    ", 4)}
}};

// 16-level Lloyd-Max codebook for TQ4_0 (4-bit indices)
static const float TQ4_0_CENTROIDS[16] = {{
{fmt_c_floats(tq4_centroids, 4, "    ", 7)}
}};

static const float TQ4_0_BOUNDARIES[15] = {{
{fmt_c_floats(tq4_boundaries, 4, "    ", 7)}
}};

// Deterministic sign-flip pattern for the randomized Hadamard transform:
// sign[i] = ((i * 0x9E3779B9) >> 31) ? -1.0f : +1.0f   (uint32 arithmetic)
static const float TQ_RHT_SIGNS[32] = {{
{fmt_c_floats([float(s) for s in signs], 8, "    ", 1)}
}};

#endif // TQ_CODEBOOKS_H
"""
    header_path = out_dir / "tq_codebooks.h"
    header_path.write_text(header, encoding="utf-8")

    # --- report ------------------------------------------------------------------
    print(
        f"8-level  Lloyd-Max: {it8} iterations, MSE {mse8:.9f} (Simpson check {mse8_simpson:.9f})"
    )
    print(
        f"16-level Lloyd-Max: {it16} iterations, MSE {mse16:.9f} "
        f"(Simpson check {mse16_simpson:.9f})"
    )
    print(f"TQ3 vs MIT branch max deviation:  {max(dev8_c, dev8_b):.2e} (tol {CROSS_CHECK_TOL})")
    print(f"TQ4 vs Max-1960 max deviation:    {dev16:.2e} (tol {CROSS_CHECK_TOL})")
    print(f"wrote {header_path.name}, {json_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
