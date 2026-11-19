"""Unit tests for exact_ref plus a cross-check against the Rust engine's Python bindings
(skipped when `ladders` is not importable). Run: python -m pytest test_exact_ref.py"""
import math
import random
from fractions import Fraction

import pytest

import exact_ref as X

try:
    import numpy as np
    import ladders as mp
except Exception:  # pragma: no cover
    mp = None


# ---- quantize
def test_quantize_half_even_and_missing():
    assert [X.quantize(v, 1.0) for v in (0.5, 1.5, 2.5, -0.5, -1.5)] == [0, 2, 2, 0, -2]
    assert X.quantize(float("nan"), 1.0) is None and X.quantize(float("inf"), 1.0) is None
    assert X.quantize(None, 1.0) is None
    assert X.quantize(0.75, 0.25) == 3
    assert X.quantize(2.0 ** 54, 1.0) is None and X.quantize(2.0 ** 53, 1.0) == 2 ** 53


# ---- window kinds
def test_kinds():
    assert X.window_kind([3, 3, 3, 3]) == X.CONSTANT
    assert X.window_kind([1, 2, 3, 4]) == X.FINITE
    assert X.window_kind([1, None, 3, 4]) == X.NONFINITE
    assert X.window_kind([None] * 4) == X.NONFINITE


def test_variance_limit_is_nonfinite():
    # m=2, window [0, d]: V = 2*d^2 - d^2 = d^2; < 2^63 iff d < 2^31.5
    ok = [0, 3037000499]          # d^2 = 9223372030926249001 < 2^63
    bad = [0, 3037000500]         # d^2 = 9223372037000250000 >= 2^63
    assert X.window_kind(ok) == X.FINITE
    assert X.window_kind(bad) == X.NONFINITE
    # a huge constant window is constant (V = 0), not non-finite
    assert X.window_kind([2 ** 52] * 4) == X.CONSTANT


def test_nonfinite_beats_constant_and_never_chosen():
    assert X.corr([1, 1, 1], [None, 1, 1]) is None
    assert X.dist2([1, 1, 1], [None, 1, 1]) is None


# ---- distances
def test_const_const_zero():
    assert X.dist2([5, 5, 5, 5], [9, 9, 9, 9]) == 0
    assert X.corr([5] * 4, [9] * 4).cmp(X.RHO_ONE) == 0


def test_const_vs_nonconst_is_m():
    for m in (4, 7):
        nc = list(range(m))
        assert X.dist2([2] * m, nc) == m
        assert X.dist2(nc, [2] * m) == m
        assert X.corr([2] * m, nc).cmp(X.RHO_HALF) == 0
        assert X.dist_f64(Fraction(m)) == math.sqrt(m)


def test_exact_values():
    a = [1, 2, 3, 4]
    assert X.dist2(a, [10, 20, 30, 40]) == 0            # scaled copy
    assert X.dist2(a, [4, 3, 2, 1]) == 16               # rho = -1: d^2 = 4m
    assert X.corr(a, [4, 3, 2, 1]).value() == -1
    # rho = 1/2 non-constant pair is exactly representable
    r = X.corr([0, 0, 1, 1, 1, 0][:4], [0, 1, 0, 1])
    assert r.is_rational()


def test_irrational_distance_close_to_float():
    a, b = [0, 1, 5, 2, 8], [3, 1, 4, 1, 5]
    d2 = X.dist2(a, b)
    za = np.array(a, float) if mp else None
    # check against Fraction identity: d2 = 2m(1-rho), rho^2 exact
    r = X.corr(a, b)
    assert abs(d2 - 2 * 5 * (1 - r.value())) == 0
    assert (1 - d2 / 10) ** 2 * 1 - r.r < Fraction(1, 10 ** 80) * 2


def test_dist_f64_correctly_rounded():
    for n in (2, 3, 10, 12345):
        assert X.dist_f64(Fraction(n)) == math.sqrt(n)
    assert X.dist_f64(Fraction(0)) == 0.0
    assert X.dist_f64(Fraction(1, 4)) == 0.5


# ---- ordering
def test_rho_order_across_variances():
    # same shape, different scale: exact tie even though f64 rho could differ
    a = [1, 3, 2, 7]
    r1 = X.corr(a, [2, 6, 4, 14])
    r2 = X.corr(a, [3, 9, 6, 21])
    assert r1.cmp(r2) == 0


def test_ties_go_to_smaller_index():
    # row 0 window [0,1,0,1]; candidates at lag > e identical -> smallest index wins
    t = [0, 1, 0, 1, 5, 9, 0, 1, 0, 1, 0, 1]
    m, e = 4, 1
    L = X.left_profile(t, m)
    # windows 0 and 2 are both exact repeats ([0,1,0,1]) of window 6: smaller index 0 wins
    assert L[6][0] == 0
    R = X.right_profile(t, m)
    assert R[0][0] == 6 and R[0][1].cmp(X.Rho(1, 1, 1)) == 0
    # window 8 = [0,1,0,1] too; row 0's right candidates 6 and 8 tie, 6 wins
    assert R[0][0] == 6


# ---- exclusion zone
def test_exclusion_zone_value_and_edges():
    assert [X.exclusion_zone(m) for m in (4, 5, 8, 9)] == [1, 2, 2, 3]
    t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]   # all windows identical for even starts
    m = 4
    e = X.exclusion_zone(m)
    L = X.left_profile(t, m)
    for i, p in enumerate(L):
        if i - e - 1 < 0:
            assert p is None                  # nothing at distance > e on the left
        else:
            assert p is not None and p[0] <= i - e - 1
    R = X.right_profile(t, m)
    for i, p in enumerate(R):
        if p is not None:
            assert p[0] >= i + e + 1
    # j = i - e exactly is excluded, j = i - e - 1 allowed
    assert L[e] is None and L[e + 1] is not None


def test_two_sided_left_wins_ties():
    A = [0, 2, 1, 5]
    # window 6 = A sits between two exact repeats (windows 0 and 12): tie, left (smaller) wins
    t = A + [9, 3, 8, 7, 4, 6] + A + [9, 3, 8, 7, 4, 6] + A
    nearest, L, R = X.two_sided_profile(t, 4)
    assert L[10][0] == 0 and R[10][0] == 20 and L[10][1].cmp(R[10][1]) == 0
    assert nearest[10] is L[10]
    s = [1, 4, 2, 8, 5, 5, 1, 9, 3, 1, 4, 2, 8]
    nearest, L, R = X.two_sided_profile(s, 4)
    for n, l, r in zip(nearest, L, R):
        assert n is l or n is r or (l is None and r is None)


def test_left_nonfinite_rows_and_candidates():
    t = [0, 3, 1, 5, 2, None, 4, 4, 0, 6, 1, 3, 2, 8]
    m = 4
    W = X.windows(t, m)
    L = X.left_profile(t, m)
    for i, p in enumerate(L):
        if W[i].kind == X.NONFINITE:
            assert p is None
        elif p is not None:
            assert W[p[0]].kind != X.NONFINITE


# ---- AB-join
def test_ab_has_no_exclusion_zone_and_ties_smaller():
    a = [0, 1, 0, 1, 0, 1]
    b = [0, 1, 0, 1, 0, 1]
    m = 4
    j = X.ab_join(a, b, m)
    # self-same series: AB row i finds itself (j = i), a self-join would exclude that
    assert [p[0] for p in j] == [0, 1, 0]     # window 2 == window 0 ties window 2 -> smaller (0)
    assert all(p[1].cmp(X.RHO_ONE) == 0 or p[1].cmp(X.Rho(1, 1, 1)) == 0 for p in j)


def test_ab_different_lengths_nonfinite_b_skipped():
    a = [3, 1, 4, 1, 5, 9]
    b = [None, 3, 1, 4, 1, 5, 9, 2]
    m = 4
    res = X.ab_join(a, b, m)
    assert len(res) == 3
    assert res[0][0] == 1                     # b window 1 == a window 0
    # non-finite A window -> None
    assert X.ab_join([None, 1, 2, 3, 4], a, 4)[0] is None


def test_ab_constants_and_scales():
    a = [2, 2, 2, 2, 9]
    b = [7, 7, 7, 7, 1, 2, 3, 4]
    res = X.ab_join(a, b, 4)
    assert res[0][0] == 0 and res[0][1].cmp(X.RHO_ONE) == 0   # const A matches const B (first)
    assert res[1][0] in range(len(b) - 3)


# ---- top-k
def test_topk_order_and_ties():
    t = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    m = 4
    K = X.topk_left(t, m, 3)
    # window 8 = [0,1,0,1]: left candidates j <= 8-2=6 : even j are exact repeats, odd are rho=-1
    assert [j for j, _ in K[8]] == [0, 2, 4]
    assert [j for j, _ in X.topk_left(t, m, 10)[8]] == [0, 2, 4, 6, 1, 3, 5]
    # nearest first
    for row in K:
        for (_, a), (_, b) in zip(row, row[1:]):
            assert a.cmp(b) >= 0
    assert K[0] == [] and K[1] == []     # j <= i - e - 1 = i - 2
    assert [j for j, _ in K[2]] == [0]


def test_topk_equals_first_of_left():
    rng = random.Random(5)
    t = [rng.randint(-5, 5) for _ in range(40)]
    L, K = X.left_profile(t, 6), X.topk_left(t, 6, 4)
    for l, k in zip(L, K):
        assert (l is None and k == []) or (l[0] == k[0][0])


# ---- discords
def test_discords_greedy_zone_and_noneighbour():
    rng = random.Random(11)
    t = [rng.randint(-9, 9) for _ in range(60)]
    m, e = 6, X.exclusion_zone(6)
    D = X.discords(t, m, 5)
    idx = [i for i, _ in D]
    assert all(abs(a - b) > e for a in idx for b in idx if a != b)
    rs = [r for _, r in D]
    assert all(a.cmp(b) <= 0 for a, b in zip(rs, rs[1:]))      # farthest first
    nearest = X.two_sided_profile(t, m)[0]
    assert idx[0] == min((i for i, p in enumerate(nearest) if p), key=lambda i: (
        __import__("functools").cmp_to_key(lambda x, y: x[1].cmp(y[1]) or x[0] - y[0])((i, nearest[i][1]))))


def test_discords_exclude_windows_without_neighbour():
    t = [1, 4, 2, 8, 3]        # 2 windows of m=4, e=1: |0-1|<=1 excluded => no neighbours
    assert X.discords(t, 4, 3) == []


def test_discords_ties_smaller_index():
    t = [0, 1, 0, 1] * 4
    D = X.discords(t, 4, 3)
    # all windows tie at the same score except odd/even classes; first pick is the smallest index
    nearest = X.two_sided_profile(t, 4)[0]
    top = D[0][0]
    assert all(not (p[1].cmp(nearest[top][1]) < 0) for p in nearest if p)
    assert all(not (p[1].cmp(nearest[top][1]) == 0 and i < top) for i, p in enumerate(nearest) if p)


# ---- compare helpers
def test_compare_helpers():
    t = [0, 3, 1, 5, 2, 8, 1, 4, 4, 0, 6, 1]
    L = X.left_profile(t, 4)
    idx = [X.entry_index(p) for p in L]
    dist = [X.entry_distance_f64(p, 4) for p in L]
    assert X.compare_profile(idx, dist, L, 4) == []
    dist2 = list(dist)
    dist2[-1] += 1e-6
    assert X.compare_profile(idx, dist2, L, 4)
    idx[-1] += 1
    assert X.compare_profile(idx, dist, L, 4)


# ---- cross-check against the Rust engine (Python bindings)
def _gen(rng):
    m = rng.randint(4, 12)
    n = rng.randint(m + 8, 110)
    kind = rng.choice(["walk", "repeat", "scaled", "plateau", "gaps", "small"])
    if kind == "small":
        x = [rng.randint(0, 2) for _ in range(n)]
    elif kind == "plateau":
        x = []
        while len(x) < n:
            x += [rng.randint(-5, 5)] * rng.randint(1, 10)
        x = x[:n]
    else:
        x, v = [], 0
        for _ in range(n):
            v += rng.randint(-4, 4)
            x.append(v)
        if kind in ("repeat", "scaled"):
            p = x[:m]
            for _ in range(3):
                s = rng.randint(0, n - m)
                a = rng.choice([1, 2, 3]) if kind == "scaled" else 1
                x[s:s + m] = [a * q + (3 if kind == "scaled" else 0) for q in p]
    x = [float(v) for v in x]
    if kind == "gaps":
        for _ in range(rng.randint(1, 3)):
            s = rng.randrange(n)
            for q in range(s, min(n, s + rng.randint(1, m))):
                x[q] = float("nan")
    return x, m


@pytest.mark.skipif(mp is None, reason="the ladders package is not installed")
@pytest.mark.parametrize("seed", range(100))
def test_cross_check_engine(seed):
    rng = random.Random(seed)
    x, m = _gen(rng)
    ticks = X.quantize_series(x, 1.0)
    arr = np.array(x)
    nearest, L, R = X.two_sided_profile(ticks, m)
    P, I, LI, RI = mp.matrix_profile(arr, m, 1.0)
    assert X.compare_profile(I, P, nearest, m) == []
    assert [int(v) for v in LI] == [X.entry_index(p) for p in L]
    assert [int(v) for v in RI] == [X.entry_index(p) for p in R]
    p1, i1, _ = mp.left_profile(arr, m, 1.0, threads=1)
    assert X.compare_profile(i1, p1, L, m) == []
    k = rng.choice([2, 3, 5])
    pk, ik, _ = mp.left_profile(arr, m, 1.0, threads=1, k=k)
    K = X.topk_left(ticks, m, k)
    for r, row in enumerate(K):
        for s in range(k):
            want = row[s] if s < len(row) else None
            assert int(ik[r][s]) == X.entry_index(want), (r, s)
            assert X.within_tol(float(pk[r][s]), want, m)
    kd = rng.choice([1, 3, 5])
    d, di = mp.discords(arr, m, kd, 1.0)
    D = X.discords(ticks, m, kd)
    assert [int(v) for v in di] == [i for i, _ in D]
    for got, (i, r) in zip(d, D):
        assert X.within_tol(float(got), (i, r), m)
