"""`aamp(T_A, m, T_B, ignore_trivial, p, k)`: with `k > 1`, `P` and `I` have shape `(windows, k)`,
the `k` nearest neighbors by exact squared distance (over both sides in a self-join, the `k`
nearest windows of `T_B` in an AB-join), nearest first, ties to the smaller index, padded with
`inf` and `-1`, while `left_I` and `right_I` stay each side's one nearest. The first column is the
`k = 1` result on bits, the thread count never changes a bit, and the neighbors equal an exact
integer brute force and STUMPY's `aamp(..., k=k)` on tie-free series."""

import math

import numpy as np
import pytest
from ladders import aamp

from conftest import random_walk, walk_with_gaps

M = 12
THREADS = [1, 2, 7, None]
KS = [2, 3, 8, 500]


def u64(x):
    return np.asarray(x).view(np.uint64)


def grid_noise(n: int, seed: int, level: float = 0.0, grid: int = 1024) -> np.ndarray:
    """Gaussian samples on a 1/1024 grid: no two squared distances tie in practice, and the values
    are exact at `scale=1/1024`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * grid) / grid


def brute_force_topk(a, m, k, scale, b=None):
    """The `k` nearest windows by exact integer squared distance on the quantized values, ties to
    the smaller index; a window with a missing sample has no neighbor and is never one. Returns
    `(P, I)` padded with `inf` and `-1`."""
    ticks = lambda s: [None if not math.isfinite(x) else round(x / scale) for x in s]
    ta, tb = ticks(a), ticks(a if b is None else b)
    whole = lambda t: [all(v is not None for v in t[i : i + m]) for i in range(len(t) - m + 1)]
    wa, wb = whole(ta), whole(tb)
    exclusion = -(-m // 4)
    squared = lambda i, j: sum((x - y) ** 2 for x, y in zip(ta[i : i + m], tb[j : j + m]))
    rows = len(wa)
    p, i_ = np.full((rows, k), np.inf), np.full((rows, k), -1, dtype=np.int64)
    for row in range(rows):
        if not wa[row]:
            continue
        if b is None:
            columns = [j for j in range(len(wb)) if wb[j] and abs(row - j) > exclusion]
        else:
            columns = [j for j in range(len(wb)) if wb[j]]
        found = sorted((squared(row, j), j) for j in columns)[:k]
        for slot, (d, j) in enumerate(found):
            p[row, slot], i_[row, slot] = math.sqrt(float(d)) * scale, j
    return p, i_


def tie_heavy(name):
    n = 90
    if name == "constant":
        return np.full(n, 7.0)
    if name == "period":
        return np.array([0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0])[np.arange(n) % 7]
    if name == "few_levels":
        return np.array([0.0, 7.0, 14.0])[(np.arange(n) * 5 // 2) % 3]
    if name == "runs":
        return np.where((np.arange(n) // 7) % 2 == 0, 0.0, 2.0**33)
    if name == "repeats":
        return (np.arange(n) % 9).astype(float) + np.where(np.arange(n) >= 45, 1e6, 0.0)
    if name == "gaps":
        t = walk_with_gaps(n, seed=6, gap_every=13, leading_gap=2)
        t[-3:] = np.nan
        return t
    if name == "walk":
        return random_walk(n, seed=5)
    raise AssertionError(name)


NAMES = ["constant", "period", "few_levels", "runs", "repeats", "gaps", "walk"]


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", NAMES)
def test_self_join_equals_the_exact_brute_force(name, k):
    t = tie_heavy(name)
    p, i, left_i, right_i = aamp(t, M, scale=1.0, threads=1, k=k)
    want_p, want_i = brute_force_topk(t, M, k, 1.0)
    n = len(t) - M + 1
    assert p.shape == i.shape == (n, k) and left_i.shape == right_i.shape == (n,)
    assert p.dtype == np.float64 and i.dtype == np.int64
    assert np.array_equal(i, want_i)
    assert np.array_equal(u64(p), u64(want_p))
    p1, i1, left_1, right_1 = aamp(t, M, scale=1.0, threads=1)
    assert np.array_equal(u64(p[:, 0]), u64(p1)) and np.array_equal(i[:, 0], i1)
    assert np.array_equal(left_i, left_1) and np.array_equal(right_i, right_1)


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", NAMES)
def test_ab_join_equals_the_exact_brute_force(name, k):
    a, b = tie_heavy(name), tie_heavy("period")[7:]
    p, i, left_i, right_i = aamp(a, M, b, ignore_trivial=False, scale=1.0, threads=1, k=k)
    want_p, want_i = brute_force_topk(a, M, k, 1.0, b)
    assert p.shape == i.shape == (len(a) - M + 1, k)
    assert np.array_equal(i, want_i)
    assert np.array_equal(u64(p), u64(want_p))
    assert np.all(left_i == -1) and np.all(right_i == -1)
    p1, i1, _, _ = aamp(a, M, b, ignore_trivial=False, scale=1.0, threads=1)
    assert np.array_equal(u64(p[:, 0]), u64(p1)) and np.array_equal(i[:, 0], i1)


def test_a_k_beyond_the_windows_is_padding():
    t = tie_heavy("walk")[:40]
    n = len(t) - M + 1
    p, i, _, _ = aamp(t, M, scale=1.0, k=n + 20)
    assert p.shape == (n, n + 20)
    assert np.array_equal(np.isinf(p), i == -1)
    assert np.all(np.isinf(p[:, -1]))


def test_every_thread_count_gives_the_same_bits():
    t = grid_noise(900, seed=21)
    for b, trivial in ((None, True), (grid_noise(700, seed=22, level=1.0), False)):
        want = aamp(t, M, b, ignore_trivial=trivial, scale=1 / 1024, threads=1, k=4)
        for threads in THREADS:
            got = aamp(t, M, b, ignore_trivial=trivial, scale=1 / 1024, threads=threads, k=4)
            for g, w in zip(got, want):
                assert np.array_equal(u64(g) if g.dtype == np.float64 else g, u64(w) if w.dtype == np.float64 else w)


@pytest.mark.parametrize("k", [2, 3, 8])
def test_self_join_equals_stumpys_aamp_on_a_tie_free_series(k):
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(400, seed=9)
    p, i, left_i, right_i = aamp(t, M, scale=1 / 1024, k=k)
    want = stumpy.aamp(t, M, k=k)
    assert want.shape == (len(p), 2 * k + 2)
    assert np.array_equal(i, want[:, k : 2 * k].astype(np.int64))
    assert np.allclose(p, want[:, :k].astype(np.float64), rtol=1e-12, atol=0)
    assert np.array_equal(left_i, want[:, 2 * k].astype(np.int64))
    assert np.array_equal(right_i, want[:, 2 * k + 1].astype(np.int64))


@pytest.mark.parametrize("k", [2, 3, 8])
def test_ab_join_equals_stumpys_aamp_on_tie_free_series(k):
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(300, seed=10), grid_noise(260, seed=11, level=2.0)
    p, i, left_i, right_i = aamp(a, M, b, ignore_trivial=False, scale=1 / 1024, k=k)
    want = stumpy.aamp(a, M, b, ignore_trivial=False, k=k)
    assert np.array_equal(i, want[:, k : 2 * k].astype(np.int64))
    assert np.allclose(p, want[:, :k].astype(np.float64), rtol=1e-12, atol=0)
    assert np.all(left_i == -1) and np.all(right_i == -1)


@pytest.mark.parametrize("make", [
    lambda t: t.astype(np.int64),
    lambda t: t.astype(np.int32),
    lambda t: t.astype(np.float32),
    lambda t: t.tolist(),
])
def test_integer_and_float32_arrays_and_lists_give_the_float64_bits(make):
    t = np.round(random_walk(120, seed=31) * 3.0)
    for b, trivial in ((None, True), (t[10:100] + 2.0, False)):
        want = aamp(t, M, b, ignore_trivial=trivial, scale=1.0, k=3)
        other = None if b is None else make(b)
        got = aamp(make(t), M, other, ignore_trivial=trivial, scale=1.0, k=3)
        for g, w in zip(got, want):
            assert np.array_equal(u64(g) if g.dtype == np.float64 else g, u64(w) if w.dtype == np.float64 else w)


@pytest.mark.parametrize("bad", [0, -1, -2**40])
def test_k_below_one_raises_value_error(bad):
    t = grid_noise(100, seed=12)
    with pytest.raises(ValueError):
        aamp(t, M, k=bad)
    with pytest.raises(ValueError):
        aamp(t, M, t[:60], ignore_trivial=False, k=bad)


def test_cells_beyond_the_cap_raise_value_error():
    t = grid_noise(1000, seed=13)
    with pytest.raises(ValueError):
        aamp(t, M, k=2**40)
    with pytest.raises(ValueError):
        aamp(t, M, t[:500], ignore_trivial=False, k=2**40)
