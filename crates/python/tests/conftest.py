"""Deterministic series generators shared by every test module.

Each one is seeded, so a failure is reproducible; none calls any wall clock.

Build the test wheel with `maturin develop --features test-hooks` (from crates/python): the hidden
hooks the GIL test needs exist only in that build, and that test skips without them.
"""

import numpy as np


def random_walk(n: int, seed: int, step: float = 1.0) -> np.ndarray:
    """A plain integer-step walk: steps in {-1, 0, 1}, so the quantized ticks (at `scale=1.0`)
    land exactly on grid points and no window is ever constant by accident of rounding."""
    rng = np.random.default_rng(seed)
    steps = rng.integers(-1, 2, size=n).astype(np.float64)
    return np.cumsum(steps) * step


def market_level_walk(n: int, seed: int, level: float = 1e10) -> np.ndarray:
    """A walk offset by a market-scale level: exercises the narrow/wide anchor decision
    the same way the Rust-side proptests do."""
    return level + random_walk(n, seed)


def walk_with_gaps(n: int, seed: int, gap_every: int = 7, leading_gap: int = 0) -> np.ndarray:
    """A walk with a NaN every `gap_every`-th sample (forward-filled by the arithmetic, never
    by this generator), plus an optional run of leading NaNs before any real sample arrives."""
    series = random_walk(n, seed)
    for k in range(gap_every - 1, n, gap_every):
        series[k] = np.nan
    if leading_gap:
        series[:leading_gap] = np.nan
    return series


def flat_run_series(n: int, seed: int, run_at: int, run_len: int) -> np.ndarray:
    """A walk with one flat (constant) run of `run_len` samples starting at `run_at`: exercises
    the constant-window branch of the kernel, not just the generic valid one."""
    series = random_walk(n, seed)
    run_len = min(run_len, n - run_at)
    series[run_at : run_at + run_len] = series[run_at]
    return series


def assert_mostly_finite(indices, m: int, threshold: float = 0.9) -> None:
    """At least `threshold` of the rows past the exclusion zone (`ceil(m / 4)`, the first rows
    no comparison here ever expects a neighbor for) have a real one (`index != -1`).

    Every bit-for-bit comparison in this suite is only as good as the profiles it compares:
    two empty ones (every row `-1`/`inf`) agree on bits just as trivially as two real ones do.
    This is the guard against that — a scale, or an argument, so wrong every row comes back
    with no neighbor would otherwise pass unnoticed.
    """
    indices = np.asarray(indices)
    exclusion = -(-m // 4)  # ceil(m / 4), the same half-width the kernel itself excludes.
    past_exclusion = indices[exclusion:]
    if past_exclusion.size == 0:
        return
    fraction = np.count_nonzero(past_exclusion != -1) / past_exclusion.size
    assert fraction >= threshold, (
        f"only {fraction:.0%} of rows past the exclusion zone have a neighbor "
        f"(wanted at least {threshold:.0%}); this comparison could otherwise pass "
        "vacuously on two empty profiles"
    )


def reseal(data):
    """`data` with its trailing 8-byte FNV-1a checksum recomputed over all preceding bytes."""
    body = bytes(data[:-8])
    h = 0xCBF29CE484222325
    for b in body:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return body + h.to_bytes(8, "little")


def profile_for(series, m: int) -> np.ndarray:
    """A stand-in for the matrix profile STUMPY's `motifs` takes as its second argument: only its
    length, `len(series) - m + 1`, is ever read."""
    return np.empty(max(len(series) - m + 1, 0))
