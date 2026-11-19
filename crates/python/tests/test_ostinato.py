"""`ostinato(ts, m, ...)`: STUMPY's consensus motif `(radius, Ts_idx, subseq_idx)` of a set of
series, the order between distances exact, the thread count never changing a bit, and bad
arguments a `ValueError`."""

import numpy as np
import pytest
from ladders import ostinato

from conftest import random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

THREADS = [1, 2, 3, 8, None, np.int64(4)]


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def planted(count: int, m: int, seed: int, jitter: float = 0.05):
    """`count` noise series of different lengths, each holding a noisy copy of one pattern at a
    different place, at a different level and scale."""
    rng = np.random.default_rng(seed)
    pattern = np.cumsum(rng.normal(size=m))
    series = []
    for k in range(count):
        t = grid_noise(150 + 37 * k, seed=seed * 100 + k, level=5.0 * k)
        at = int(rng.integers(0, len(t) - m))
        t[at : at + m] = 5.0 * k + (1.0 + 0.5 * k) * pattern + jitter * rng.normal(size=m)
        series.append(np.round(t * 64.0) / 64.0)
    return series


@pytest.mark.parametrize("count", [3, 4, 5])
@pytest.mark.parametrize("m", [4, 12])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_equals_stumpy_on_sets_with_a_planted_pattern(count, m, seed):
    ts = planted(count, m, seed)
    want = stumpy.ostinato(ts, m)
    for threads in [1, 3, None]:
        radius, series, window = ostinato(ts, m, scales=[1 / 64] * count, threads=threads)
        assert (series, window) == (want[1], want[2]), threads
        assert abs(radius - want[0]) < 1e-6, threads
        assert isinstance(radius, float) and isinstance(series, int) and isinstance(window, int)


@pytest.mark.parametrize("m", [4, 12])
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_two_series_share_the_radius_of_their_mutual_pair(m, seed):
    # A mutual nearest pair is one distance seen from either series: an exact tie, which
    # STUMPY breaks by rounding noise and this by the first series, so only the radius is compared.
    ts = planted(2, m, seed)
    radius, series, window = ostinato(ts, m, scales=[1 / 64] * 2)
    assert abs(radius - stumpy.ostinato(ts, m)[0]) < 1e-6
    assert series in (0, 1) and 0 <= window < len(ts[series]) - m + 1


@pytest.mark.parametrize("count", [2, 4])
def test_equals_stumpy_on_pure_noise(count):
    ts = [grid_noise(180 + 20 * k, seed=k + 10) for k in range(count)]
    want = stumpy.ostinato(ts, 10)
    radius, series, window = ostinato(ts, 10, scales=[1 / 64] * count)
    assert (series, window) == (want[1], want[2])
    assert abs(radius - want[0]) < 1e-6


def test_equals_stumpy_on_series_with_gaps():
    ts = [
        walk_with_gaps(200, seed=3, gap_every=37),
        walk_with_gaps(170, seed=4, gap_every=29, leading_gap=5),
        random_walk(220, seed=5),
    ]
    want = stumpy.ostinato(ts, 12)
    radius, series, window = ostinato(ts, 12, scales=[1.0, 1.0, 1.0])
    assert (series, window) == (want[1], want[2])
    assert abs(radius - want[0]) < 1e-6


def test_an_identical_pattern_gives_radius_zero():
    pattern = np.array([0.0, 6.0, -3.0, 9.0, 2.0, -7.0, 4.0, 1.0])
    ts = []
    for k, at in enumerate([5, 31, 17]):
        t = grid_noise(80, seed=k + 20)
        t[at : at + 8] = 10.0 * k + 3.0 * pattern
        ts.append(t)
    radius, series, window = ostinato(ts, 8, scales=[1 / 64] * 3)
    assert radius == 0.0
    assert window == [5, 31, 17][series]


def test_a_single_series_is_its_own_consensus():
    t = grid_noise(100, seed=30)
    radius, series, window = ostinato([t], 10, scales=[1 / 64])
    assert (radius, series, window) == (0.0, 0, 0)


def test_a_set_without_a_meeting_window_gives_inf_at_the_origin():
    ts = [random_walk(60, seed=31), np.full(60, np.nan), random_walk(50, seed=32)]
    assert ostinato(ts, 8, scales=[1.0, 1.0, 1.0]) == (float("inf"), 0, 0)


def test_scales_are_inferred_per_series_when_omitted():
    ts = [grid_noise(120, seed=40), grid_noise(90, seed=41, level=1e6)]
    assert ostinato(ts, 10) == ostinato(ts, 10, scales=[None, None])


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    # Long enough for the threaded engine to cut tiles.
    if gaps:
        ts = [walk_with_gaps(1500, seed=40 + k, gap_every=211 + 50 * k) for k in range(3)]
    else:
        ts = [random_walk(1500 - 100 * k, seed=40 + k) for k in range(3)]
    first = ostinato(ts, 24, scales=[1.0] * 3, threads=1)
    assert np.isfinite(first[0])
    for threads in THREADS:
        got = ostinato(ts, 24, scales=[1.0] * 3, threads=threads)
        assert got[1:] == first[1:], threads
        assert np.float64(got[0]).view(np.uint64) == np.float64(first[0]).view(np.uint64), threads


def test_bad_arguments_are_value_errors():
    ts = [random_walk(80, seed=9), random_walk(70, seed=10)]
    for bad in (0, -1, 2):
        with pytest.raises(ValueError):
            ostinato(ts, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            ostinato(ts, 10, threads=bad)
    with pytest.raises(ValueError):
        ostinato([], 10)
    with pytest.raises(ValueError):
        ostinato(ts, 10, scales=[1.0])
    with pytest.raises(ValueError):
        ostinato(ts, 10, scales=[1.0, 0.0])
    with pytest.raises(ValueError):
        ostinato([ts[0], ts[1][:9]], 10, scales=[1.0, 1.0])
    with pytest.raises(ValueError):
        ostinato([ts[0], ts[1][:9]], 10)
