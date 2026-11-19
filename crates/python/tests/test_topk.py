"""`left_profile(t, m, scale, threads, k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`,
nearest first, padded with `inf` and `-1`. The first column is the `k = 1` result on bits, the
thread count never changes a bit, and the neighbors equal a brute-force z-normalized search."""

import numpy as np
import pytest
from ladders import left_profile

from conftest import flat_run_series, random_walk, walk_with_gaps

M = 20
THREADS = [1, 2, 7, None]
KS = [2, 3, 8]


def u64(x):
    return np.asarray(x).view(np.uint64)


def tie_free(n, seed):
    """Continuous values: no two windows are exact repeats, so no distances tie."""
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


SERIES = {
    "tie_free": (lambda: tie_free(500, seed=1), 1e-6),
    "random_walk": (lambda: random_walk(500, seed=2), 1.0),
    "flat_run": (lambda: flat_run_series(500, seed=3, run_at=100, run_len=40), 1.0),
    "with_gaps": (lambda: walk_with_gaps(500, seed=4, gap_every=97, leading_gap=3), 1.0),
}


def brute_force_left_topk(series, m, k):
    """The `k` nearest earlier z-normalized neighbors per window, outside the exclusion zone,
    ties to the smaller index; `(P, I)` padded with `inf` and `-1`."""
    n = len(series) - m + 1
    exclusion = -(-m // 4)
    windows = np.lib.stride_tricks.sliding_window_view(series, m)
    z = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)
    p = np.full((n, k), np.inf)
    i = np.full((n, k), -1, dtype=np.int64)
    for row in range(exclusion + 1, n):
        d = np.sqrt(np.maximum(2 * m * (1 - z[: row - exclusion] @ z[row] / m), 0.0))
        order = np.argsort(d, kind="stable")[:k]
        p[row, : len(order)] = d[order]
        i[row, : len(order)] = order
    return p, i


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", SERIES)
def test_first_column_is_the_k_one_result_and_rows_are_nearest_first(name, k):
    make, scale = SERIES[name]
    series = make()
    p, i, scale_used = left_profile(series, M, scale, threads=1, k=k)
    p1, i1, scale_one = left_profile(series, M, scale, threads=1)

    n = len(series) - M + 1
    assert p.shape == i.shape == (n, k)
    assert p1.shape == i1.shape == (n,)
    assert scale_used == scale_one
    assert np.array_equal(u64(p[:, 0]), u64(p1))
    assert np.array_equal(i[:, 0], i1)
    # Exact ties may report distances one rounding apart, so order holds up to that.
    assert np.all(p[:, 1:] >= p[:, :-1] * (1 - 1e-14))
    assert np.array_equal(np.isinf(p), i == -1)
    for row in range(n):
        kept = i[row][i[row] >= 0]
        assert len(set(kept)) == len(kept)
        assert np.all(kept < row)


@pytest.mark.parametrize("name", SERIES)
def test_every_thread_count_gives_the_same_bits(name):
    make, scale = SERIES[name]
    series = make()
    expected = left_profile(series, M, scale, threads=1, k=3)
    for threads in THREADS:
        got = left_profile(series, M, scale, threads=threads, k=3)
        assert np.array_equal(u64(got[0]), u64(expected[0])), threads
        assert np.array_equal(got[1], expected[1]), threads


@pytest.mark.parametrize("k", KS)
def test_neighbors_equal_a_brute_force_search(k):
    series = tie_free(300, seed=7)
    p, i, _ = left_profile(series, M, 1e-6, threads=2, k=k)
    expected_p, expected_i = brute_force_left_topk(series, M, k)
    assert np.array_equal(i, expected_i)
    assert np.allclose(p, expected_p, rtol=1e-6, atol=1e-6, equal_nan=False)


def test_a_window_with_fewer_than_k_neighbors_is_padded():
    series = tie_free(60, seed=8)
    p, i, _ = left_profile(series, M, 1e-6, threads=1, k=50)
    n = len(series) - M + 1
    assert p.shape == i.shape == (n, 50)
    assert np.all(i[0] == -1) and np.all(np.isinf(p[0]))
    last = n - 1
    real = (i[last] >= 0).sum()
    assert 0 < real < 50
    assert np.all(i[last, real:] == -1) and np.all(np.isinf(p[last, real:]))


def test_stumpy_agrees_on_windows_whose_k_nearest_are_all_earlier():
    """`stump(T, m, k=k)` ranks over both sides, so it is comparable only on windows whose `k`
    nearest overall are all earlier ones: there its list is the left top-k."""
    stumpy = pytest.importorskip("stumpy")
    series = tie_free(400, seed=9)
    for k in KS:
        p, i, _ = left_profile(series, M, 1e-6, threads=1, k=k)
        reference = stumpy.stump(series, M, k=k)
        ref_p = reference[:, :k].astype(np.float64)
        ref_i = reference[:, k : 2 * k].astype(np.int64)
        rows = np.arange(len(series) - M + 1)[:, None]
        comparable = np.all((ref_i >= 0) & (ref_i < rows), axis=1)
        assert comparable.sum() >= 20, k
        assert np.array_equal(i[comparable], ref_i[comparable]), k
        assert np.allclose(p[comparable], ref_p[comparable], rtol=1e-6, atol=1e-6), k


@pytest.mark.parametrize("bad", [0, -1])
def test_k_below_one_raises_value_error(bad):
    with pytest.raises(ValueError):
        left_profile(random_walk(100, seed=10), M, 1.0, k=bad)
