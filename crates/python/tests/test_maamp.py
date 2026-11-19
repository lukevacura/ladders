"""`maamp(T, m, include, discords, p, scale, threads)` and `mstump(..., normalize=False)`: STUMPY's
`(P, I)` of the non-normalized multi-dimensional profile on series without ties, an exact integer
brute force on series full of ties, the thread count never changes a bit, and `p` other than 2 is
`NotImplementedError`."""

import math

import numpy as np
import pytest
from ladders import aamp, maamp, mstump

from conftest import random_walk

M = 8
THREADS = [1, 2, 3, 8, None, np.int64(4)]


def bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def grid_noise(d: int, n: int, seed: int) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(size=(d, n)) * 64.0) / 64.0


def brute_force(t, m, scale, include=(), discords=False):
    """Exact integer profile on the quantized values: per pair each dimension's squared distance
    in ticks, dimensions ordered the way the library orders them (included first in the order
    given, the rest nearest first or farthest first, a missing one last, ties to the smaller
    dimension), the mean of the roots summed left to right, ties to the smaller index."""
    d, n = t.shape
    ticks = [[None if not math.isfinite(x) else round(x / scale) for x in row] for row in t]
    windows = n - m + 1
    complete = [[all(v is not None for v in row[i : i + m]) for i in range(windows)] for row in ticks]
    exclusion = -(-m // 4)
    included = list(dict.fromkeys(include))
    rest = [a for a in range(d) if a not in included]
    p = np.full((d, windows), np.inf)
    index = np.full((d, windows), -1)
    for i in range(windows):
        for j in range(windows):
            if abs(i - j) <= exclusion:
                continue
            squared = [
                sum((x - y) ** 2 for x, y in zip(ticks[a][i : i + m], ticks[a][j : j + m]))
                if complete[a][i] and complete[a][j]
                else None
                for a in range(d)
            ]
            present = sorted((a for a in rest if squared[a] is not None), key=lambda a: (squared[a], a))
            if discords:
                present = sorted(present, key=lambda a: (-squared[a], a))
            order = included + present + [a for a in rest if squared[a] is None]
            total = 0.0
            for k, a in enumerate(order):
                if squared[a] is None:
                    break
                total += math.sqrt(float(squared[a]))
                mean = total if k == 0 else total / (k + 1)
                value = mean * scale
                if value < p[k, i]:
                    p[k, i], index[k, i] = value, j
    return p, index


def test_equals_stumpys_maamp_on_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(3, 200, seed=1)
    for include, discords in [(None, False), ([2], False), ([1, 0], False), (None, True), ([2], True)]:
        want_p, want_i = stumpy.maamp(t, M, include=include, discords=discords)
        p, i = maamp(t, M, include=include, discords=discords, scale=1 / 64)
        assert p.shape == i.shape == want_p.shape == (3, 200 - M + 1)
        assert p.dtype == np.float64 and i.dtype == np.int64
        assert np.array_equal(i, want_i.astype(np.int64)), (include, discords)
        assert np.allclose(p, want_p, rtol=1e-12, atol=0), (include, discords)


def test_mstump_without_normalizing_is_maamp():
    t = grid_noise(2, 120, seed=2)
    by_flag = mstump(t, M, include=[1], normalize=False, scales=1 / 64)
    direct = maamp(t, M, include=[1], scale=1 / 64)
    assert np.array_equal(bits(by_flag[0]), bits(direct[0])) and np.array_equal(by_flag[1], direct[1])


def test_one_dimension_is_the_non_normalized_self_join():
    t = grid_noise(1, 150, seed=3)
    p, i = maamp(t, M, scale=1 / 64)
    want = aamp(t[0], M, scale=1 / 64)
    assert np.array_equal(bits(p[0]), bits(want[0])) and np.array_equal(i[0], want[1])


@pytest.mark.parametrize("seed", range(3))
def test_ties_equal_the_integer_brute_force(seed):
    rng = np.random.default_rng(seed)
    t = rng.integers(0, 3, size=(3, 60)).astype(np.float64)
    t[1] = 2.0 * t[0]
    t[2, 10] = np.nan
    for include, discords in [((), False), ((2,), False), ((1, 0), True), ((), True)]:
        p, i = maamp(t, 5, include=list(include), discords=discords, scale=1.0)
        want_p, want_i = brute_force(t, 5, 1.0, include, discords)
        assert np.array_equal(i, want_i), (seed, include, discords)
        assert np.array_equal(bits(p), bits(want_p)), (seed, include, discords)


def test_the_thread_count_never_changes_a_bit():
    rng = np.random.default_rng(4)
    levels = rng.integers(0, 3, size=(3, 160)).astype(np.float64)
    levels[1] = 2.0 * levels[0] + 1.0
    levels[2, 40] = np.nan
    want = maamp(levels, M, include=[2], scale=1.0, threads=1)
    for threads in THREADS:
        p, i = maamp(levels, M, include=[2], scale=1.0, threads=threads)
        assert np.array_equal(bits(p), bits(want[0])) and np.array_equal(i, want[1]), threads


def test_the_scale_is_inferred_from_all_dimensions_together():
    t = np.stack([random_walk(100, seed=5), random_walk(100, seed=6) + 3.0])
    p, i = maamp(t, M)
    q, j = maamp(t, M, scale=1.0)
    assert np.array_equal(bits(p), bits(q)) and np.array_equal(i, j)


def test_a_gap_makes_its_windows_missing_in_that_dimension_only():
    t = grid_noise(2, 100, seed=7)
    t[1, 50] = np.nan
    p, i = maamp(t, M, scale=1 / 64)
    assert np.isfinite(p[0]).all()
    covered = slice(50 - M + 1, 51)
    assert np.isinf(p[1, covered]).all() and (i[1, covered] == -1).all()


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda t: maamp(t, M, p=1.0), NotImplementedError),
        (lambda t: mstump(t, M, normalize=False, p=3.0), NotImplementedError),
        (lambda t: maamp(t, 2), ValueError),
        (lambda t: maamp(t, M, include=[-1]), ValueError),
        (lambda t: maamp(t, M, include=[2]), ValueError),
        (lambda t: maamp(t, M, scale=-1.0), ValueError),
        (lambda t: mstump(t, M, normalize=False, scales=[1.0, 1.0]), ValueError),
        (lambda t: maamp(t, 50), ValueError),
        (lambda t: maamp(t, M, threads=0), ValueError),
        (lambda t: maamp(t[:0], M), ValueError),
    ],
)
def test_bad_arguments(call, error):
    with pytest.raises(error):
        call(grid_noise(2, 30, seed=8))
