"""`matrix_profile(t, m, scale, threads, k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`
over both sides, nearest first, padded with `inf` and `-1`, while `left_I` and `right_I` stay each
side's one nearest. The first column is the `k = 1` result on bits, the thread count never changes
a bit, and the neighbors equal a brute-force search and STUMPY's `stump(T, m, k=k)`."""

import numpy as np
import pytest
from ladders import matrix_profile

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


def brute_force_topk(series, m, k):
    """The `k` nearest z-normalized neighbors per window over both sides, outside the exclusion
    zone, ties to the smaller index; `(P, I)` padded with `inf` and `-1`."""
    n = len(series) - m + 1
    exclusion = -(-m // 4)
    windows = np.lib.stride_tricks.sliding_window_view(series, m)
    z = (windows - windows.mean(axis=1, keepdims=True)) / windows.std(axis=1, keepdims=True)
    p = np.full((n, k), np.inf)
    i = np.full((n, k), -1, dtype=np.int64)
    for row in range(n):
        d = np.sqrt(np.maximum(2 * m * (1 - z @ z[row] / m), 0.0))
        d[max(0, row - exclusion) : row + exclusion + 1] = np.inf
        order = np.argsort(d, kind="stable")[:k]
        order = order[np.isfinite(d[order])]
        p[row, : len(order)] = d[order]
        i[row, : len(order)] = order
    return p, i


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", SERIES)
def test_first_column_is_the_k_one_result_and_the_sides_stay_k_one(name, k):
    make, scale = SERIES[name]
    series = make()
    p, i, left_i, right_i = matrix_profile(series, M, scale, threads=1, k=k)
    p1, i1, left_1, right_1 = matrix_profile(series, M, scale, threads=1)

    n = len(series) - M + 1
    assert p.shape == i.shape == (n, k)
    assert p1.shape == i1.shape == left_i.shape == right_i.shape == (n,)
    assert p.dtype == np.float64 and i.dtype == np.int64
    assert np.array_equal(u64(p[:, 0]), u64(p1))
    assert np.array_equal(i[:, 0], i1)
    assert np.array_equal(left_i, left_1) and np.array_equal(right_i, right_1)
    # Exact ties may report distances one rounding apart, so order holds up to that.
    assert np.all(p[:, 1:] >= p[:, :-1] * (1 - 1e-14))
    assert np.array_equal(np.isinf(p), i == -1)
    for row in range(n):
        kept = i[row][i[row] >= 0]
        assert len(set(kept)) == len(kept)


@pytest.mark.parametrize("name", SERIES)
def test_every_thread_count_gives_the_same_bits(name):
    make, scale = SERIES[name]
    series = make()
    expected = matrix_profile(series, M, scale, threads=1, k=3)
    for threads in THREADS:
        got = matrix_profile(series, M, scale, threads=threads, k=3)
        for want, have in zip(expected, got):
            assert np.array_equal(u64(have), u64(want)), threads


@pytest.mark.parametrize("k", KS)
def test_neighbors_equal_a_brute_force_search(k):
    series = tie_free(300, seed=7)
    p, i, _, _ = matrix_profile(series, M, 1e-6, threads=2, k=k)
    expected_p, expected_i = brute_force_topk(series, M, k)
    assert np.array_equal(i, expected_i)
    assert np.allclose(p, expected_p, rtol=1e-6, atol=1e-6)


def test_a_window_with_fewer_than_k_neighbors_is_padded():
    series = tie_free(40, seed=8)
    p, i, _, _ = matrix_profile(series, M, 1e-6, threads=1, k=50)
    n = len(series) - M + 1
    assert p.shape == i.shape == (n, 50)
    real = (i >= 0).sum(axis=1)
    assert 0 < real.max() < 50
    for row in range(n):
        assert np.all(i[row, real[row] :] == -1) and np.all(np.isinf(p[row, real[row] :]))


def test_equals_stumpys_top_k_on_a_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    series = tie_free(400, seed=9)
    for k in KS:
        p, i, left_i, right_i = matrix_profile(series, M, 1e-6, threads=1, k=k)
        reference = stumpy.stump(series, M, k=k)
        assert reference.shape == (len(p), 2 * k + 2)
        assert np.array_equal(i, reference[:, k : 2 * k].astype(np.int64)), k
        assert np.allclose(p, reference[:, :k].astype(np.float64), rtol=1e-6, atol=1e-6), k
        assert np.array_equal(left_i, reference[:, 2 * k].astype(np.int64)), k
        assert np.array_equal(right_i, reference[:, 2 * k + 1].astype(np.int64)), k


@pytest.mark.parametrize("bad", [0, -1])
def test_k_below_one_raises_value_error(bad):
    with pytest.raises(ValueError):
        matrix_profile(random_walk(100, seed=10), M, 1.0, k=bad)


def test_cells_beyond_the_cap_raise_value_error():
    with pytest.raises(ValueError):
        matrix_profile(random_walk(1000, seed=11), M, 1.0, k=2**40)

