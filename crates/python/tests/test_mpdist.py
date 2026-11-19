"""`mpdist(t_a, t_b, m, ...)`: STUMPY's matrix profile distance, the order between distances exact,
the thread count never changing a bit, and bad arguments a `ValueError`."""

import numpy as np
import pytest
from ladders import ab_join, mpdist

from conftest import random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

THREADS = [1, 2, 3, 8, None, np.int64(4)]


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def test_the_stumpy_documentation_example():
    a = np.array([-11.1, 23.4, 79.5, 1001.0])
    b = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    want = stumpy.mpdist(a, b, m=3)
    assert abs(want - 0.00019935236191097894) < 1e-9
    assert abs(mpdist(a, b, 3) - want) < 1e-6


@pytest.mark.parametrize("m", [3, 8, 20])
@pytest.mark.parametrize("sizes", [(400, 400), (700, 330), (260, 910)])
def test_equals_stumpy_on_tie_free_series(m, sizes):
    a, b = grid_noise(sizes[0], seed=1), grid_noise(sizes[1], seed=2, level=3.0)
    for threads in [1, 3, None]:
        for percentage in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]:
            got = mpdist(a, b, m, percentage, scale_a=1 / 64, scale_b=1 / 64, threads=threads)
            want = stumpy.mpdist(a, b, m, percentage=percentage)
            assert isinstance(got, float)
            assert abs(got - want) < 1e-6, (percentage, threads)
        for k in [0, 1, 7, 100]:
            got = mpdist(a, b, m, k=k, scale_a=1 / 64, scale_b=1 / 64, threads=threads)
            assert abs(got - stumpy.mpdist(a, b, m, k=k)) < 1e-6, (k, threads)


def test_equals_stumpy_on_series_with_gaps():
    a = walk_with_gaps(500, seed=3, gap_every=37)
    b = walk_with_gaps(420, seed=4, gap_every=29, leading_gap=5)
    for percentage in [0.05, 0.3]:
        got = mpdist(a, b, 12, percentage, scale_a=1.0, scale_b=1.0)
        want = stumpy.mpdist(a, b, 12, percentage=percentage)
        assert abs(got - want) < 1e-6, percentage


def test_k_wins_over_percentage_and_stumpys_clamps_are_errors_naming_the_explicit_value():
    a, b = grid_noise(300, seed=5), grid_noise(250, seed=6)
    scales = dict(scale_a=1 / 64, scale_b=1 / 64)
    windows = (300 - 9) + (250 - 9)
    last = mpdist(a, b, 10, k=windows - 1, **scales)
    assert mpdist(a, b, 10, 0.0, k=windows - 1, **scales) == last
    assert mpdist(a, b, 10, 1.0, **scales) == last
    for k in [windows, windows + 5, 2**62, 2**63 - 1]:
        with pytest.raises(ValueError, match=f"k={k} is past the last position") as error:
            mpdist(a, b, 10, k=k, **scales)
        assert f"pass k={windows - 1}" in str(error.value)
        assert abs(last - stumpy.mpdist(a, b, 10, k=min(k, 2**31))) < 1e-6
    for percentage, clipped in [(2.0, "1.0"), (1e300, "1.0"), (float("inf"), "1.0"),
                                (-1.0, "0.0"), (-1e300, "0.0"), (float("-inf"), "0.0"),
                                (1.0 + 2**-52, "1.0"), (-(2**-1074), "0.0")]:
        with pytest.raises(ValueError, match="is outside \\[0, 1\\]") as error:
            mpdist(a, b, 10, percentage, **scales)
        assert f"pass percentage={clipped}" in str(error.value)
    first = mpdist(a, b, 10, k=0, **scales)
    assert mpdist(a, b, 10, 0.0, **scales) == first
    assert abs(first - stumpy.mpdist(a, b, 10, percentage=-1.0)) < 1e-6
    assert mpdist(a, b, 10, float("nan"), k=3, **scales) == mpdist(a, b, 10, k=3, **scales)
    with pytest.raises(ValueError, match="percentage must not be NaN"):
        mpdist(a, b, 10, float("nan"), **scales)


def test_a_series_is_at_distance_zero_from_itself():
    a = grid_noise(200, seed=7)
    assert mpdist(a, a, 10, scale_a=1 / 64, scale_b=1 / 64) == 0.0


def test_a_side_without_finite_windows_gives_the_largest_finite_distance_or_inf():
    a = random_walk(80, seed=8)
    nothing = np.full(40, np.nan)
    assert mpdist(a, nothing, 10, scale_a=1.0, scale_b=1.0) == float("inf")
    half = np.concatenate([random_walk(40, seed=9), np.full(60, np.nan)])
    got = mpdist(a, half, 10, 1.0, scale_a=1.0, scale_b=1.0)
    assert np.isfinite(got)
    assert abs(got - stumpy.mpdist(a, half, 10, percentage=1.0)) < 1e-6


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    # Long enough for the threaded engine to cut tiles.
    if gaps:
        a = walk_with_gaps(3200, seed=40, gap_every=211)
        b = walk_with_gaps(2700, seed=41, gap_every=301)
    else:
        a, b = random_walk(3200, seed=40), random_walk(2700, seed=41)
    first = mpdist(a, b, 24, scale_a=1.0, scale_b=1.0, threads=1)
    assert np.isfinite(first)
    for threads in THREADS:
        got = mpdist(a, b, 24, scale_a=1.0, scale_b=1.0, threads=threads)
        assert np.float64(got).view(np.uint64) == np.float64(first).view(np.uint64), threads


def test_each_series_infers_its_own_scale():
    a = np.round(1e8 + 0.01 * random_walk(500, seed=5), 2)
    b = random_walk(400, seed=6)
    _, _, scale_a, scale_b = ab_join(a, b, 20)
    assert scale_a != scale_b
    inferred = mpdist(a, b, 20)
    assert inferred == mpdist(a, b, 20, scale_a=scale_a, scale_b=scale_b)


def test_bad_arguments_are_value_errors():
    a, b = random_walk(80, seed=9), random_walk(70, seed=10)
    for bad in (0, -1, 2):
        with pytest.raises(ValueError):
            mpdist(a, b, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            mpdist(a, b, 10, threads=bad)
    with pytest.raises(ValueError):
        mpdist(a, b, 10, float("nan"))
    with pytest.raises(ValueError):
        mpdist(a, b, 10, k=-1)
    with pytest.raises(ValueError):
        mpdist(a, b, 10, scale_a=0.0)
    with pytest.raises(ValueError):
        mpdist(a, b[:9], 10, scale_a=1.0, scale_b=1.0)
    with pytest.raises(ValueError):
        mpdist(a[:9], b, 10, scale_a=1.0, scale_b=1.0)
    with pytest.raises(ValueError):
        mpdist(a, b[:9], 10)
