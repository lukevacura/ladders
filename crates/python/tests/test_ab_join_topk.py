"""`ab_join(t_a, t_b, m, ..., k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`, each row
the `k` nearest windows of `t_b`, nearest first, padded with `inf` and `-1`. The first column is
the `k = 1` result on bits, the thread count never changes a bit, and the neighbors equal a
brute-force search and STUMPY's `stump(T_A, m, T_B, k=k)`."""

import numpy as np
import pytest
from ladders import ab_join

from conftest import random_walk, walk_with_gaps

M = 20
THREADS = [1, 2, 7, None]
KS = [2, 3, 8]


def u64(x):
    return np.asarray(x).view(np.uint64)


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def brute_force_topk(a, b, m, k):
    """The `k` nearest z-normalized windows of `b` per window of `a`, ties to the smaller index;
    `(P, I)` padded with `inf` and `-1`."""

    def z_windows(series):
        w = np.lib.stride_tricks.sliding_window_view(series, m)
        return (w - w.mean(axis=1, keepdims=True)) / w.std(axis=1, keepdims=True)

    za, zb = z_windows(a), z_windows(b)
    p = np.full((len(za), k), np.inf)
    i = np.full((len(za), k), -1, dtype=np.int64)
    for row in range(len(za)):
        d = np.sqrt(np.maximum(2 * m * (1 - zb @ za[row] / m), 0.0))
        order = np.argsort(d, kind="stable")[:k]
        p[row, : len(order)] = d[order]
        i[row, : len(order)] = order
    return p, i


@pytest.mark.parametrize("k", KS)
def test_first_column_is_the_k_one_result_and_rows_are_nearest_first(k):
    a, b = grid_noise(300, seed=1), grid_noise(250, seed=2, level=3.0)
    p, i, scale_a, scale_b = ab_join(a, b, M, 1 / 64, 1 / 64, threads=1, k=k)
    p1, i1, _, _ = ab_join(a, b, M, 1 / 64, 1 / 64, threads=1)
    assert p.shape == i.shape == (len(a) - M + 1, k)
    assert p.dtype == np.float64 and i.dtype == np.int64
    assert (scale_a, scale_b) == (1 / 64, 1 / 64)
    assert np.array_equal(u64(p[:, 0]), u64(p1)) and np.array_equal(i[:, 0], i1)
    assert np.all(p[:, 1:] >= p[:, :-1] * (1 - 1e-14))
    assert np.array_equal(np.isinf(p), i == -1)


@pytest.mark.parametrize("k", KS)
def test_neighbors_equal_a_brute_force_search(k):
    a, b = grid_noise(200, seed=3), grid_noise(180, seed=4, level=1.0)
    p, i, _, _ = ab_join(a, b, M, 1 / 64, 1 / 64, threads=2, k=k)
    expected_p, expected_i = brute_force_topk(a, b, M, k)
    assert np.array_equal(i, expected_i)
    assert np.allclose(p, expected_p, rtol=1e-6, atol=1e-6)


def test_equals_stumpys_top_k_on_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(400, seed=5), grid_noise(330, seed=6, level=3.0)
    for k in KS:
        p, i, _, _ = ab_join(a, b, M, 1 / 64, 1 / 64, threads=3, k=k)
        reference = stumpy.stump(a, M, b, k=k)
        assert np.array_equal(i, reference[:, k : 2 * k].astype(np.int64)), k
        assert np.allclose(p, reference[:, :k].astype(np.float64), rtol=0, atol=1e-6), k


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    if gaps:
        a = walk_with_gaps(3200, seed=40, gap_every=211)
        b = walk_with_gaps(2700, seed=41, gap_every=301)
    else:
        a, b = random_walk(3200, seed=40), random_walk(2700, seed=41)
    p1, i1, _, _ = ab_join(a, b, 24, 1.0, 1.0, threads=1, k=3)
    assert np.count_nonzero(i1[:, 0] != -1) > len(i1) // 2
    for threads in THREADS:
        p, i, _, _ = ab_join(a, b, 24, 1.0, 1.0, threads=threads, k=3)
        assert np.array_equal(i, i1), threads
        assert np.array_equal(u64(p), u64(p1)), threads


def test_fewer_finite_windows_of_b_than_k_is_padded():
    a, b = grid_noise(60, seed=7), grid_noise(25, seed=8)
    p, i, _, _ = ab_join(a, b, M, 1 / 64, 1 / 64, threads=1, k=50)
    assert p.shape == i.shape == (len(a) - M + 1, 50)
    assert np.all((i >= 0).sum(axis=1) == len(b) - M + 1)
    assert np.all(i[:, len(b) - M + 1 :] == -1) and np.all(np.isinf(p[:, len(b) - M + 1 :]))


@pytest.mark.parametrize("bad", [0, -1])
def test_k_below_one_raises_value_error(bad):
    a = random_walk(100, seed=10)
    with pytest.raises(ValueError):
        ab_join(a, a, M, 1.0, 1.0, k=bad)


def test_cells_beyond_the_cap_raise_value_error():
    a = random_walk(1000, seed=11)
    with pytest.raises(ValueError):
        ab_join(a, a, M, 1.0, 1.0, k=2**40)

