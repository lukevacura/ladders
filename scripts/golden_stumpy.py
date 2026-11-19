# /// script
# requires-python = ">=3.11"
# dependencies = ["stumpy==1.14.1", "numpy==2.5.4", "numba==0.68.0", "llvmlite==0.50.0"]
# ///
"""Writes STUMPY's batch `stump` output for each golden series to crates/testkit/fixtures/.

Each `<name>.golden` file holds one `key value…` line per field: `m` in decimal, f64 arrays as
16-digit hex bit patterns, index arrays in decimal with STUMPY's -1 written as u64::MAX.
"""

import os

# Thread count decides which of several tied neighbors STUMPY reports.
os.environ["NUMBA_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import stumpy
from stumpy import config, core

FIXTURES = Path(__file__).resolve().parent.parent / "crates" / "testkit" / "fixtures"
NONE = 2**64 - 1
stump_module = sys.modules["stumpy.stump"]


def random_walk(rng, n):
    return np.cumsum(rng.standard_normal(n))


def series_set():
    rng = np.random.default_rng(20261015)
    walk = random_walk(rng, 64)

    flat = random_walk(rng, 64)
    flat[8:24] = flat[8]
    flat[40:52] = 3.0

    gaps = random_walk(rng, 64)
    gaps[10] = np.nan
    gaps[30] = np.inf
    gaps[31] = np.nan
    gaps[50] = -np.inf

    motif = random_walk(rng, 16)
    repeats = np.concatenate([motif, motif, motif, motif])

    return [
        ("random_walk", 8, walk),
        ("random_walk_offset_1e6", 8, walk + 1e6),
        ("random_walk_offset_1e8", 8, walk + 1e8),
        ("flat_segments", 8, flat),
        ("nonfinite_gaps", 8, gaps),
        ("exact_repeats", 8, repeats),
        ("tiny_m", 3, random_walk(rng, 32)),
        ("m_near_n", 24, random_walk(rng, 32)),
    ]


def left_profile(series, m):
    """STUMPY's own left distances, which `stump` computes and then discards."""
    T, μ, σ_inverse, μ_m_1, isfinite, isconstant = core.preprocess_diagonal(series, m)
    exclusion = int(np.ceil(m / config.STUMPY_EXCL_ZONE_DENOM))
    diags = np.arange(exclusion + 1, len(series) - m + 1, dtype=np.int64)
    P, PL, _, I, IL, _ = stump_module._stump(
        T, T, m, μ, μ, σ_inverse, σ_inverse, μ_m_1, μ_m_1,
        isfinite, isfinite, isconstant, isconstant, diags, True, 1,
    )
    return P[:, 0], I[:, 0], PL, IL


def f64_hex(values):
    return " ".join(f"{bits:016x}" for bits in np.asarray(values, np.float64).view(np.uint64))


def indices(values):
    values = np.asarray(values, np.int64)
    assert (values >= -1).all()
    return " ".join(str(NONE if v == -1 else int(v)) for v in values)


def same_bits(a, b):
    return np.array_equal(np.asarray(a, np.float64).view(np.uint64), np.asarray(b, np.float64).view(np.uint64))


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, m, series in series_set():
        out = stumpy.stump(series, m)
        p, i = out[:, 0].astype(np.float64), out[:, 1].astype(np.int64)
        left_i = out[:, 2].astype(np.int64)
        same_p, same_i, left_p, same_left_i = left_profile(series, m)
        assert same_bits(same_p, p) and np.array_equal(same_i, i) and np.array_equal(same_left_i, left_i), name

        lines = [
            f"m {m}",
            f"series {f64_hex(series)}",
            f"left_p {f64_hex(left_p)}",
            f"left_i {indices(left_i)}",
            f"p {f64_hex(p)}",
            f"i {indices(i)}",
        ]
        (FIXTURES / f"{name}.golden").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
