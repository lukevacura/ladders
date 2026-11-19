"""`ab_join(t_a, t_b, m, ...)`: STUMPY's A-side `stump(T_A, m, T_B)` columns, the thread count never
changes a bit, each series infers and reports its own scale, and bad arguments are a
`ValueError`."""

import numpy as np
import pytest
import ladders
from ladders import ab_join, left_profile

from conftest import market_level_walk, random_walk, walk_with_gaps

M = 20
THREADS = [1, 2, 3, 8, None, np.int64(4)]


def u64(x):
    return np.asarray(x).view(np.uint64)


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def test_equals_stumpys_a_side_on_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(700, seed=1), grid_noise(530, seed=2, level=3.0)
    want = stumpy.stump(a, M, b)
    for threads in [1, 3, None]:
        p, i, scale_a, scale_b = ab_join(a, b, M, 1 / 64, 1 / 64, threads=threads)
        assert p.shape == i.shape == (len(a) - M + 1,)
        assert i.dtype == np.int64 and p.dtype == np.float64
        assert np.array_equal(i, want[:, 1].astype(np.int64)), threads
        assert np.allclose(p, want[:, 0].astype(float), rtol=0, atol=1e-6), threads
        assert (scale_a, scale_b) == (1 / 64, 1 / 64)


def test_every_row_meets_a_window_of_b_with_no_exclusion_zone():
    a = grid_noise(300, seed=3)
    p, i, _, _ = ab_join(a, a, M, 1 / 64, 1 / 64, threads=1)
    assert np.array_equal(i, np.arange(len(a) - M + 1))
    assert np.array_equal(u64(p), u64(np.zeros(len(i))))


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    # Long enough for the threaded engine to cut tiles.
    if gaps:
        a = walk_with_gaps(3200, seed=40, gap_every=211)
        b = walk_with_gaps(2700, seed=41, gap_every=301)
    else:
        a, b = random_walk(3200, seed=40), market_level_walk(2700, seed=41, level=1e8)
    scale_a, scale_b = 1.0, 1.0
    p1, i1, _, _ = ab_join(a, b, 24, scale_a, scale_b, threads=1)
    assert np.count_nonzero(i1 != -1) > len(i1) // 2
    for threads in THREADS:
        p, i, _, _ = ab_join(a, b, 24, scale_a, scale_b, threads=threads)
        assert np.array_equal(i, i1), threads
        assert np.array_equal(u64(p), u64(p1)), threads


def test_each_series_infers_and_reports_its_own_scale():
    a = np.round(1e8 + 0.01 * random_walk(500, seed=5), 2)
    b = random_walk(400, seed=6)
    p, i, scale_a, scale_b = ab_join(a, b, M)
    assert scale_a == left_profile(a, M)[2]
    assert scale_b == left_profile(b, M)[2]
    assert scale_a != scale_b
    again = ab_join(a, b, M, scale_a, scale_b)
    assert np.array_equal(i, again[1]) and np.array_equal(u64(p), u64(again[0]))


def test_a_side_without_a_finite_window_of_b_has_no_neighbor():
    a = random_walk(60, seed=7)
    b = np.full(30, np.nan)
    p, i, _, _ = ab_join(a, b, M, 1.0, 1.0)
    assert np.all(i == -1) and np.all(np.isposinf(p))
    short = random_walk(M - 1, seed=8)
    p, i, _, _ = ab_join(a, short, M, 1.0, 1.0)
    assert len(i) == len(a) - M + 1 and np.all(i == -1)
    p, i, _, _ = ab_join(short, a, M, 1.0, 1.0)
    assert len(p) == 0 and len(i) == 0


def test_bad_arguments_are_value_errors():
    a, b = random_walk(80, seed=9), random_walk(70, seed=10)
    for bad in (0, -1):
        with pytest.raises(ValueError):
            ab_join(a, b, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            ab_join(a, b, M, threads=bad)
    with pytest.raises(ValueError):
        ab_join(a, b, M, scale_a=0.0)
    with pytest.raises(ValueError):
        ab_join(a, b[: M - 1], M)
    with pytest.raises(ValueError):
        ab_join(a[: M - 1], b, M)
