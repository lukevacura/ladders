"""`matrix_profile(t, m, scale, threads)`: STUMPY's `(P, I, left_I, right_I)`. The left side is
`left_profile` on bits, the right side is the left profile of the reversed series, the nearest is
the nearer of the two, and the thread count never changes a bit."""

import numpy as np
import pytest
from ladders import left_profile, matrix_profile

from conftest import flat_run_series, random_walk, walk_with_gaps

M = 20
THREADS = [1, 2, 7, None]


def u64(x):
    return np.asarray(x).view(np.uint64)


def tie_free(n, seed):
    """Continuous values on a fine grid: no two windows are exact repeats, so reversing the
    series cannot change which of two equal candidates wins."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


SERIES = {
    "tie_free": (lambda: tie_free(600, seed=1), 1e-6),
    "random_walk": (lambda: random_walk(600, seed=2), 1.0),
    "flat_run": (lambda: flat_run_series(600, seed=3, run_at=100, run_len=40), 1.0),
    "with_gaps": (lambda: walk_with_gaps(600, seed=4, gap_every=97, leading_gap=3), 1.0),
}


@pytest.mark.parametrize("name", SERIES)
def test_left_side_is_left_profile_and_nearest_is_one_of_the_sides(name):
    make, scale = SERIES[name]
    series = make()
    p, i, left_i, right_i = matrix_profile(series, M, scale, threads=1)
    left_p, left_index, _ = left_profile(series, M, scale, threads=1)

    assert len(p) == len(i) == len(left_i) == len(right_i) == len(series) - M + 1
    assert np.array_equal(left_i, left_index)
    assert np.all((left_i < np.arange(len(i))) | (left_i == -1))
    assert np.all((right_i > np.arange(len(i))) | (right_i == -1))
    assert np.all((i == left_i) | (i == right_i))
    assert np.any(right_i >= 0)

    from_left = i == left_i
    assert np.array_equal(u64(p[from_left]), u64(left_p[from_left]))
    assert np.array_equal(np.isinf(p), i == -1)


def test_right_side_is_the_left_profile_of_the_reversed_series():
    series = tie_free(600, seed=5)
    _, _, _, right_i = matrix_profile(series, M, 1e-6, threads=1)
    _, reversed_i, _ = left_profile(series[::-1].copy(), M, 1e-6, threads=1)
    n_windows = len(right_i)
    mirrored = np.where(reversed_i[::-1] >= 0, n_windows - 1 - reversed_i[::-1], -1)
    assert np.array_equal(right_i, mirrored)


@pytest.mark.parametrize("name", SERIES)
def test_every_thread_count_gives_the_same_bits(name):
    make, scale = SERIES[name]
    series = make()
    expected = matrix_profile(series, M, scale, threads=1)
    for threads in THREADS:
        got = matrix_profile(series, M, scale, threads=threads)
        assert np.array_equal(u64(got[0]), u64(expected[0])), threads
        for a, b in zip(got[1:], expected[1:]):
            assert np.array_equal(a, b), threads


def test_scale_none_infers_the_scale_left_profile_does():
    series = 1e8 + 0.01 * random_walk(600, seed=6)
    p, i, left_i, _ = matrix_profile(series, M, threads=2)
    _, left_index, _ = left_profile(series, M, threads=2)
    assert np.array_equal(left_i, left_index)
    assert np.all(np.isfinite(p[i >= 0]))


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_a_bad_thread_count_is_a_value_error(bad):
    with pytest.raises(ValueError):
        matrix_profile(random_walk(100, seed=7), 10, 1.0, threads=bad)


def test_a_series_shorter_than_the_window_is_a_value_error():
    with pytest.raises(ValueError):
        matrix_profile(np.arange(5.0), 10)
