"""`aamp(T_A, m, T_B, ignore_trivial, p, scale, threads)`: STUMPY's `aamp` columns `(P, I, left_I,
right_I)` on series without ties, an exact integer brute force on series full of ties, the
thread count never changes a bit, and `p` other than 2 is `NotImplementedError`."""

import math

import numpy as np
import pytest
from ladders import aamp

from conftest import random_walk, walk_with_gaps

M = 12
THREADS = [1, 2, 3, 8, None, np.int64(4)]


def u64(x):
    return np.asarray(x).view(np.uint64)


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def brute_force(a, m, scale, b=None):
    """Exact integer profile on the quantized values, ties to the smaller index; a window with a
    missing sample has no neighbor and is never one. Returns `(P, I, left_I, right_I)`."""
    ticks = lambda s: [None if not math.isfinite(x) else round(x / scale) for x in s]
    ta, tb = ticks(a), ticks(a if b is None else b)
    whole = lambda t: [all(v is not None for v in t[i : i + m]) for i in range(len(t) - m + 1)]
    wa, wb = whole(ta), whole(tb)
    exclusion = -(-m // 4)
    squared = lambda i, j: sum((x - y) ** 2 for x, y in zip(ta[i : i + m], tb[j : j + m]))
    rows = len(wa)
    p, i_, left, right = np.full(rows, np.inf), np.full(rows, -1), np.full(rows, -1), np.full(rows, -1)

    def best(row, columns):
        found = None
        for j in columns:
            if wb[j]:
                d = squared(row, j)
                if found is None or d < found[0]:
                    found = (d, j)
        return found

    for row in range(rows):
        if not wa[row]:
            continue
        if b is None:
            l = best(row, range(0, row - exclusion))
            r = best(row, range(row + exclusion + 1, rows))
            left[row] = -1 if l is None else l[1]
            right[row] = -1 if r is None else r[1]
            sides = [s for s in (l, r) if s is not None]
            nearest = min(sides, key=lambda s: (s[0], s[1])) if sides else None
        else:
            nearest = best(row, range(len(wb)))
        if nearest is not None:
            p[row], i_[row] = math.sqrt(float(nearest[0])) * scale, nearest[1]
    return p, i_, left, right


def test_self_join_equals_stumpys_aamp_on_a_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(500, seed=1)
    want = stumpy.aamp(t, M)
    p, i, left_i, right_i = aamp(t, M, scale=1 / 64)
    assert p.shape == i.shape == left_i.shape == right_i.shape == (len(t) - M + 1,)
    assert p.dtype == np.float64 and i.dtype == left_i.dtype == right_i.dtype == np.int64
    assert np.array_equal(i, want[:, 1].astype(np.int64))
    assert np.array_equal(left_i, want[:, 2].astype(np.int64))
    assert np.array_equal(right_i, want[:, 3].astype(np.int64))
    assert np.allclose(p, want[:, 0].astype(float), rtol=1e-12, atol=0)


def test_ab_join_equals_stumpys_aamp_on_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(420, seed=2), grid_noise(300, seed=3, level=2.0)
    want = stumpy.aamp(a, M, b, ignore_trivial=False)
    p, i, left_i, right_i = aamp(a, M, b, ignore_trivial=False, scale=1 / 64)
    assert np.array_equal(i, want[:, 1].astype(np.int64))
    assert np.array_equal(left_i, want[:, 2].astype(np.int64))
    assert np.array_equal(right_i, want[:, 3].astype(np.int64))
    assert np.allclose(p, want[:, 0].astype(float), rtol=1e-12, atol=0)
    assert np.all(left_i == -1) and np.all(right_i == -1)


def test_self_join_is_the_two_sided_profile_of_the_same_values():
    t = grid_noise(300, seed=4)
    p, i, left_i, right_i = aamp(t, M, scale=1 / 64)
    assert np.all((left_i < np.arange(len(i))) | (left_i == -1))
    assert np.all((right_i > np.arange(len(i))) | (right_i == -1))
    assert np.all((i == left_i) | (i == right_i))
    assert np.any(right_i >= 0)
    assert np.array_equal(np.isinf(p), i == -1)


@pytest.mark.parametrize("name", ["walk", "ties", "gaps", "spikes"])
def test_equals_an_integer_brute_force(name):
    if name == "walk":
        t = random_walk(90, seed=5)
    elif name == "ties":
        t = np.tile([0.0, 3.0, 1.0, 4.0, 1.0], 18)
    elif name == "gaps":
        t = walk_with_gaps(90, seed=6, gap_every=13, leading_gap=2)
    else:
        # Spikes of 2**40 ticks put a window's squared distance past 2**64.
        t = random_walk(90, seed=7)
        t[[20, 21, 50, 70]] += 2.0**40
    want = brute_force(t, M, 1.0)
    got = aamp(t, M, scale=1.0, threads=1)
    assert np.array_equal(got[1], want[1])
    assert np.array_equal(got[2], want[2])
    assert np.array_equal(got[3], want[3])
    assert np.allclose(got[0], want[0], rtol=1e-15, atol=0)

    other = random_walk(70, seed=8)
    want = brute_force(t, M, 1.0, b=other)
    got = aamp(t, M, other, ignore_trivial=False, scale=1.0, threads=1)
    assert np.array_equal(got[1], want[1])
    assert np.allclose(got[0], want[0], rtol=1e-15, atol=0)


def test_ignore_trivial_only_matters_when_the_series_are_equal():
    t = grid_noise(200, seed=9)
    selfjoin = aamp(t, M, scale=1 / 64)
    same = aamp(t, M, t.copy(), scale=1 / 64)
    for got, want in zip(same, selfjoin):
        assert np.array_equal(u64(got), u64(want))

    p, i, left_i, right_i = aamp(t, M, t.copy(), ignore_trivial=False, scale=1 / 64)
    assert np.array_equal(i, np.arange(len(t) - M + 1))
    assert np.array_equal(u64(p), u64(np.zeros(len(i))))
    assert np.all(left_i == -1) and np.all(right_i == -1)

    other = grid_noise(150, seed=10)
    default = aamp(t, M, other, scale=1 / 64)
    explicit = aamp(t, M, other, ignore_trivial=False, scale=1 / 64)
    for got, want in zip(default, explicit):
        assert np.array_equal(u64(got), u64(want))


def test_the_inferred_scale_picks_the_same_neighbors_as_any_exact_grid():
    a, b = grid_noise(300, seed=13), grid_noise(260, seed=14, level=1.0)
    inferred = aamp(a, M, b, ignore_trivial=False)
    for scale in (1 / 64, 1 / 128):
        explicit = aamp(a, M, b, ignore_trivial=False, scale=scale)
        assert np.array_equal(inferred[1], explicit[1])
        assert np.allclose(inferred[0], explicit[0], rtol=1e-12, atol=0)
    own = aamp(a, M)
    assert np.array_equal(own[1], aamp(a, M, scale=1 / 64)[1])


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    t = walk_with_gaps(3200, seed=40, gap_every=211) if gaps else random_walk(3200, seed=41)
    other = random_walk(2600, seed=42)
    for b, trivial in ((None, True), (other, False)):
        want = aamp(t, 20, b, ignore_trivial=trivial, scale=1.0, threads=1)
        for threads in THREADS:
            got = aamp(t, 20, b, ignore_trivial=trivial, scale=1.0, threads=threads)
            for g, w in zip(got, want):
                assert np.array_equal(u64(g), u64(w)) if g.dtype == np.float64 else np.array_equal(g, w)


@pytest.mark.parametrize("p", [1.0, 3.0, 0.5, 2.5, float("nan"), float("inf")])
def test_other_values_of_p_are_not_implemented(p):
    t = grid_noise(80, seed=15)
    with pytest.raises(NotImplementedError):
        aamp(t, M, p=p)
    with pytest.raises(NotImplementedError):
        aamp(t, M, t[:60], ignore_trivial=False, p=p)
    p_int = aamp(t, M, p=2, scale=1 / 64)
    assert np.array_equal(p_int[1], aamp(t, M, p=2.0, scale=1 / 64)[1])


def test_a_window_with_a_gap_has_no_neighbor_and_is_never_one():
    t = grid_noise(120, seed=16)
    t[40] = np.nan
    p, i, left_i, right_i = aamp(t, M, scale=1 / 64)
    broken = np.arange(len(i))
    broken = broken[(broken <= 40) & (broken + M > 40)]
    assert np.all(i[broken] == -1) and np.all(np.isinf(p[broken]))
    for column in (i, left_i, right_i):
        assert not np.isin(column, broken).any()


def test_bad_arguments_are_a_value_error():
    t = grid_noise(80, seed=17)
    for bad in (2, 0, -3):
        with pytest.raises(ValueError):
            aamp(t, bad)
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            aamp(t, M, scale=scale)
    for threads in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            aamp(t, M, threads=threads)
    with pytest.raises(ValueError):
        aamp(t[: M - 1], M)
    with pytest.raises(ValueError):
        aamp(np.full(40, np.nan), M)


def test_a_series_shorter_than_the_window_given_a_scale_has_no_rows_or_no_neighbors():
    t = grid_noise(80, seed=18)
    short = t[: M - 1]
    p, i, left_i, right_i = aamp(short, M, scale=1 / 64)
    assert len(p) == len(i) == len(left_i) == len(right_i) == 0
    p, i, _, _ = aamp(t, M, short, ignore_trivial=False, scale=1 / 64)
    assert len(i) == len(t) - M + 1 and np.all(i == -1) and np.all(np.isinf(p))


def test_non_normalized_inference_takes_the_fast_scale_and_finest_is_an_opt_in():
    """Off a grid, the finest exact scale sends the non-normalized pass to its exact routes, so
    `aamp`, `aampi` and `maamp` infer the fast scale the z-normalized functions infer, and
    `scale="finest"` asks for the finer one."""
    import ladders as mp

    noise = np.random.default_rng(8).uniform(-1.0, 1.0, 600)
    m = 64
    fast = mp.left_profile(noise, m)[2]
    assert fast > mp.left_profile(noise, m, "finest")[2]
    assert mp.aampi(noise, m, egress=False).scale_ == fast
    explicit = mp.aamp(noise, m, scale=fast)
    for got, want in zip(mp.aamp(noise, m), explicit):
        assert np.array_equal(u64(got), u64(want))
    dims = np.stack([noise, noise[::-1].copy()])
    for got, want in zip(mp.maamp(dims, m), mp.maamp(dims, m, scale=fast)):
        assert np.array_equal(u64(got), u64(want))
