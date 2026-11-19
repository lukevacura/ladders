"""`left_profile`: bit-for-bit against `GrowingStream.update_many` on the same series (two
independent engines, the "Python sees Rust's bits" guarantee checked both ways), the `scale=None`
rule, and a few distances pinned by literal bits."""

import math

import numpy as np
import pytest
from ladders import GrowingStream, left_profile

from conftest import (
    assert_mostly_finite,
    flat_run_series,
    market_level_walk,
    random_walk,
    walk_with_gaps,
)

M = 4

# An irrational factor puts a series on no decimal or binary grid, so `scale=None` falls back to
# the fast power-of-two scale.
OFF_GRID = math.pi


def bits(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


SERIES = {
    "random_walk": lambda: random_walk(150, seed=10),
    "market_level": lambda: market_level_walk(150, seed=11),
    "with_gaps": lambda: walk_with_gaps(150, seed=12, gap_every=13, leading_gap=3),
    "flat_run": lambda: flat_run_series(150, seed=13, run_at=30, run_len=15),
}

# `with_gaps` legitimately has fewer matches (a real gap poisons every window it touches, at
# every scale): its own threshold only rules out the vacuous (near-0%) case.
FINITE_THRESHOLD = {"random_walk": 0.9, "market_level": 0.9, "with_gaps": 0.4, "flat_run": 0.9}


@pytest.mark.parametrize("series_name", SERIES)
def test_left_profile_matches_growing_stream_on_bits(series_name):
    series = SERIES[series_name]()
    scale = 1.0

    stream_indices, stream_distances = GrowingStream(M, scale).update_many(series)
    batch_distances, batch_indices, scale_used = left_profile(series, M, scale)

    assert scale_used == scale
    assert np.array_equal(stream_indices, batch_indices)
    assert np.array_equal(
        stream_distances.view(np.uint64), batch_distances.view(np.uint64)
    )
    assert_mostly_finite(batch_indices, M, threshold=FINITE_THRESHOLD[series_name])


def test_left_profile_order_is_pstumpy_p_then_i():
    """`left_profile` returns `(P, I, scale_used)`: distances first, indices second — STUMPY's
    own order, the opposite of `update`'s `(index, distance)`."""
    series = random_walk(30, seed=20)
    p, i, scale_used = left_profile(series, M, 1.0)
    assert p.dtype == np.float64
    assert i.dtype == np.int64
    assert isinstance(scale_used, float)


@pytest.mark.parametrize(
    "series_name,make_series",
    [
        ("ordinary_magnitude", lambda: random_walk(80, seed=30) * OFF_GRID),
        ("tiny_magnitude", lambda: random_walk(80, seed=30) * 1e-6 * OFF_GRID),
    ],
)
def test_scale_none_picks_a_scale_that_is_actually_usable(series_name, make_series):
    """The documented rule is not just "every single tick fits `MAX_TICKS`" — a scale that
    fine can still blow up a window's variance numerator well past `2**63` (the bug this test
    guards against): `scale_used` must be a power of two and must make the profile actually
    usable (`assert_mostly_finite`, not two empty profiles agreeing on bits). The real
    "finest" is pinned exactly on the Rust side (`quantize::tests::
    finest_fitting_scale_is_minimal`)."""
    series = make_series()

    _, indices, scale_used = left_profile(series, M, None)

    log2_scale = math.log2(scale_used)
    assert log2_scale == int(log2_scale), f"{scale_used} is not a power of two"
    assert_mostly_finite(indices, M)

    # A finer scale no longer costs windows their neighbors: a window past the exact domain is
    # ranked by the cold route, so halving the scale leaves every window ranked. Which scale is
    # the finest is pinned on the Rust side, not here.
    _, half_indices, _ = left_profile(series, M, scale_used / 2)
    assert np.count_nonzero(half_indices != -1) == np.count_nonzero(indices != -1)


def test_scale_none_fixes_the_literal_bug_report():
    """The exact shape of series that found this bug: a few thousand samples around a level
    of 100, `m = 50`. Under the old (per-tick-magnitude-only) rule this gave 0 finite entries
    out of 1951; the fix must make most of them finite."""
    series = (100.0 + random_walk(2000, seed=99)) * OFF_GRID
    m = 50

    _, indices, _ = left_profile(series, m, None)

    assert len(indices) == len(series) - m + 1
    assert_mostly_finite(indices, m)


@pytest.mark.parametrize(
    "bad_series",
    [
        np.array([1.0, 2.0]),  # shorter than m
        np.full(60, np.nan),  # no finite sample at all
    ],
    ids=["too_short", "all_nan"],
)
def test_scale_none_raises_value_error_when_no_scale_fits(bad_series):
    with pytest.raises(ValueError):
        left_profile(bad_series, 50, None)


def test_scale_none_matches_using_scale_used_explicitly():
    series = market_level_walk(80, seed=31) * OFF_GRID
    p_auto, i_auto, scale_used = left_profile(series, M, None)
    p_explicit, i_explicit, scale_used_again = left_profile(series, M, scale_used)

    assert scale_used == scale_used_again
    assert np.array_equal(i_auto, i_explicit)
    assert np.array_equal(p_auto.view(np.uint64), p_explicit.view(np.uint64))


def test_scale_finest_is_the_opt_in_finer_rule():
    """`scale="finest"` keeps the finest exact scale: on plain floats at m = 64 it is one power of
    two finer than the default, which keeps every window inside the fast path's bound; on grid
    data both find the grid. Every function inferring a scale takes it, and a stream inferring
    from its history reports it."""
    import ladders as mp

    rng = np.random.default_rng(5)
    noise = rng.uniform(-1.0, 1.0, 4000)
    m = 64
    _, _, fast = left_profile(noise, m, None)
    _, _, finest = left_profile(noise, m, "finest")
    assert fast == 2.0**-25 and finest == 2.0**-26
    p_finest, i_finest, again = left_profile(noise, m, finest)
    assert again == finest
    p_named, i_named, _ = left_profile(noise, m, "finest")
    assert np.array_equal(i_named, i_finest)
    assert np.array_equal(p_named.view(np.uint64), p_finest.view(np.uint64))

    cents = 100.0 + np.round(random_walk(400, seed=3) * 7) / 100
    assert left_profile(cents, M, None)[2] == left_profile(cents, M, "finest")[2] == 0.01

    assert mp.stumpi(noise, m, egress=False, scale="finest").scale_ == finest
    assert mp.stumpi(noise, m, egress=False).scale_ == fast
    aamp_finest = mp.aamp(noise, m, scale="finest")
    aamp_explicit = mp.aamp(noise, m, scale=mp.aampi(noise, m, egress=False, scale="finest").scale_)
    assert np.array_equal(aamp_finest[1], aamp_explicit[1])


@pytest.mark.parametrize("bad", ["fine", "FINEST", ""])
def test_scale_strings_other_than_finest_raise(bad):
    with pytest.raises(ValueError, match="finest"):
        left_profile(random_walk(30, seed=1), M, bad)


def test_distance_bits_are_pinned_on_a_few_generic_entries():
    """A handful of hand-worked entries, pinned by their literal `float64` bit pattern: an
    exact repeat (distance `0.0`), and a window with no eligible neighbor yet (distance
    `inf`) — both via `libm`'s `sqrt`."""
    # Two adjacent, exactly repeated windows of length 4: row 1's nearest neighbor is row 0,
    # at distance exactly 0.0 once both windows are identical.
    series = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    p, i, _ = left_profile(series, M, 1.0)

    last_row = len(p) - 1
    assert i[last_row] == 0
    assert bits(p[last_row]) == bits(0.0)
    assert bits(p[last_row]) == 0  # 0.0's own bit pattern, spelled out literally

    # Row 0 has no eligible neighbor at all (inside the exclusion zone): distance is `inf`,
    # whose bit pattern is the IEEE 754 exponent-all-ones, mantissa-zero pattern.
    assert i[0] == -1
    assert bits(p[0]) == 0x7FF0000000000000


def test_invalid_m_raises_value_error_not_panic():
    series = random_walk(20, seed=50)
    for bad_m in (-1, 0, 1, 2):
        with pytest.raises(ValueError):
            left_profile(series, bad_m, 1.0)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_raises_value_error(bad_scale):
    series = random_walk(20, seed=51)
    with pytest.raises(ValueError):
        left_profile(series, M, bad_scale)


@pytest.mark.parametrize("grid", [0.01, 0.25, 0.05, 1.0, 100.0])
def test_scale_none_infers_the_grid_the_data_lies_on(grid):
    """Data exactly on a grid gets that grid as `scale_used` and the same bits as passing it."""
    series = walk_with_gaps(300, seed=40, gap_every=17, leading_gap=2) * grid
    p_auto, i_auto, scale_used = left_profile(series, 8, None)
    assert scale_used == grid
    p_given, i_given, _ = left_profile(series, 8, grid)
    assert_mostly_finite(i_auto, 8, threshold=0.4)
    assert np.array_equal(i_auto, i_given)
    assert np.array_equal(p_auto.view(np.uint64), p_given.view(np.uint64))


def test_scale_none_on_a_cent_grid_walk_at_a_market_level():
    rng = np.random.default_rng(41)
    cents = 10_000_000 + np.cumsum(rng.integers(-3, 4, size=400))
    series = cents * 0.01
    p, i, scale_used = left_profile(series, 16, None)
    assert scale_used == 0.01
    assert_mostly_finite(i, 16)
    p2, i2, _ = left_profile(series, 16, 0.01)
    assert np.array_equal(p.view(np.uint64), p2.view(np.uint64))
