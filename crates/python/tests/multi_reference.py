"""Exact references for the multi-dimensional tests, independent of the library and of STUMPY.

Subspace bins: a z-normalized sample's bin is the number of correctly rounded normal quantiles
strictly below its z-value, decided by rational arithmetic on the window's integers. The quantiles
are the edge generator's (mpmath at 200 and 260 bits, its rounding decided), and every bin is
checked again against a second computation that shares no code with the first: the quantiles by
Newton's method on mpmath's normal CDF at 256 bits, each rounded and decided here, and the z-value
by mpmath at 256 bits. Dimensions are ordered by a plain sort.

`mstump` indices: each dimension's distance is decided exactly (its correlation compared by
integer cross-multiplication), and a mean of distances by f64 where the margin is a million times
any rounding, else by mpmath at 320 bits; an exact tie goes to the smaller window start.
"""

import functools
import importlib.util
import math
import statistics
from fractions import Fraction
from pathlib import Path

import mpmath
import numpy as np

# The working precision of the independent checks, in bits.
PRECISION = 256
# A value is decided at that precision when it is farther than this, relative, from what it is
# compared with: far above the computation's error, far below any f64 spacing.
DECIDED = Fraction(1, 2**200)
# STUMPY's edges are scipy's `norm.ppf`, within 1e-12 relative of the correctly rounded quantile on
# every platform (test_subspace checks it). A z-value farther than this from every edge is binned
# alike by both edge sets, so STUMPY can differ from the exact answer only on samples nearer.
NEAR_EDGE = 2.0**-36
# STUMPY's (and the library's) deviation below which a z-normalized window is not scaled.
SMALL_DEVIATION = 1e-7
# The largest variance numerator the library holds; a window at or beyond it has no distance.
VARIANCE_LIMIT = 2**63


@functools.cache
def generator():
    """scripts/gen_subspace_tables.py, which computes the committed tables."""
    script = Path(__file__).resolve().parents[3] / "scripts" / "gen_subspace_tables.py"
    spec = importlib.util.spec_from_file_location("gen_subspace_tables", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def exact_edges(bits):
    """The correctly rounded normal quantiles of `i / 2^bits`, as fractions."""
    return tuple(Fraction(generator().normal_quantile(i, 2**bits)) for i in range(1, 2**bits))


def exact(value):
    """A finite mpmath value as the fraction it is (`man_exp` drops the sign; `_mpf_` keeps it)."""
    sign, mantissa, exponent, _ = mpmath.mpf(value)._mpf_
    return (-1) ** sign * Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def nearest_f64(value):
    """The f64 nearest the mpmath `value`, after checking that `value` is decidedly on one side of
    the midpoints to its neighbours."""
    value = exact(value)
    nearest = float(value)  # Fraction to float is correctly rounded.
    for neighbour in (math.nextafter(nearest, math.inf), math.nextafter(nearest, -math.inf)):
        midpoint = (Fraction(nearest) + Fraction(neighbour)) / 2
        assert abs(value - midpoint) > DECIDED * abs(midpoint), (nearest, "rounding not decided")
    return nearest


@functools.cache
def mpmath_edges(bits):
    """The same quantiles by another route: Newton's method on `ncdf(x) = i / 2^bits` at 256 bits,
    from the standard library's estimate, then rounded to the nearest f64 here."""
    out = []
    with mpmath.workprec(PRECISION):
        for i in range(1, 2**bits):
            p = mpmath.mpf(i) / 2**bits
            x = mpmath.mpf(statistics.NormalDist().inv_cdf(i / 2**bits))
            for _ in range(12):
                x -= (mpmath.ncdf(x) - p) / mpmath.npdf(x)
            assert abs(mpmath.ncdf(x) - p) < mpmath.mpf(2) ** (16 - PRECISION), (bits, i)
            out.append(0.0 if 2 * i == 2**bits else nearest_f64(x))
    return tuple(out)


def integers(values):
    """Exact numbers as integers proportional to them (z-values do not change)."""
    fractions = [Fraction(v) for v in values]
    scale = math.lcm(*(f.denominator for f in fractions))
    return [int(f * scale) for f in fractions]


def above(edge, a, variance):
    """`edge < a / sqrt(variance)`, exactly, for an integer offset `a = m x - S`."""
    if edge == 0:
        return a > 0
    if edge > 0:
        return a > 0 and a * a > edge * edge * variance
    return a >= 0 or a * a < edge * edge * variance


def count_below(edges, below):
    """How many of the increasing `edges` satisfy `below`, which holds for a prefix of them."""
    low, high = 0, len(edges)
    while low < high:
        middle = (low + high) // 2
        if below(edges[middle]):
            low = middle + 1
        else:
            high = middle
    return low


def bins_against(ticks, edges):
    """The bins of a z-normalized integer window against `edges` (fractions or floats), by exact
    rational arithmetic; a constant window is all in the middle bin."""
    m = len(ticks)
    total = sum(ticks)
    variance = m * sum(t * t for t in ticks) - total * total
    if variance == 0:
        return [len(edges) // 2] * m
    edges = [Fraction(e) for e in edges]
    return [count_below(edges, lambda e: above(e, m * t - total, variance)) for t in ticks]


def mpmath_bins(ticks, bits):
    """`bins_against(ticks, exact_edges(bits))` computed independently, with every comparison
    checked to be decided; also how many samples lie within `NEAR_EDGE` of a nonzero edge."""
    m = len(ticks)
    total = sum(ticks)
    variance = m * sum(t * t for t in ticks) - total * total
    edges = mpmath_edges(bits)
    if variance == 0:
        return [len(edges) // 2] * m, 0
    out, near = [], 0
    with mpmath.workprec(PRECISION):
        root = mpmath.sqrt(variance)
        for t in ticks:
            a = m * t - total
            z = mpmath.mpf(a) / root

            def below(e, a=a, z=z):
                if e == 0:
                    return a > 0
                gap = exact(z) - Fraction(e)
                assert abs(gap) > DECIDED * abs(Fraction(e)), (ticks, e, "not decided")
                return gap > 0

            bin_ = count_below(edges, below)
            out.append(bin_)
            # The nearest edges are the two around the bin.
            nearest = edges[max(bin_ - 1, 0) : bin_ + 1]
            near += any(e != 0 and abs(z - e) <= NEAR_EDGE * abs(e) for e in nearest)
    return out, near


class Tally:
    """Counts the binned samples within `NEAR_EDGE` of a nonzero edge: the only samples where
    STUMPY's platform-dependent edges can bin otherwise than the exact ones."""

    def __init__(self):
        self.near = 0
        self.samples = 0


@functools.lru_cache(maxsize=None)
def checked_bins(ticks, bits):
    """`bins_against(ticks, exact_edges(bits))`, after checking it equals `mpmath_bins` (the two
    must agree on every sample, or the reference is wrong); also the near-edge count."""
    out = bins_against(ticks, exact_edges(bits))
    check, near = mpmath_bins(ticks, bits)
    assert out == check, (ticks, bits, out, check)
    return out, near


def exact_bins(ticks, bits, tally=None):
    """The bins of a z-normalized integer window by exact rational arithmetic, checked against
    an independent mpmath computation."""
    out, near = checked_bins(tuple(ticks), bits)
    if tally is not None:
        tally.near += near
        tally.samples += len(ticks)
    return list(out)


def small_bins(ticks, scale, edges, tally=None):
    """The library's (and STUMPY's) rule for a window whose deviation is below `SMALL_DEVIATION`:
    the z-value is the offset from the mean, not divided by the deviation, evaluated in f64 as
    `(m x - S) / m * scale` (every operation correctly rounded, so the same on every platform)."""
    m = len(ticks)
    total = sum(ticks)
    values = [float(m * t - total) / m * scale for t in ticks]
    if tally is not None:
        tally.samples += m
        tally.near += sum(any(e != 0 and abs(z - e) <= NEAR_EDGE * abs(e) for e in edges) for z in values)
    return [count_below(edges, lambda e, z=z: e < z) for z in values]


def window_bins(values, bits, scale=None, flagged=False, tally=None, edges=None):
    """The bins of one z-normalized window of float samples, or None when it has no distance.

    `scale` given: the samples are its ticks, `round(x / scale)` with ties to even (the input rule).
    `scale` None: the exact values themselves, which is what any scale on which the data lie gives.
    `flagged`: the caller marked the window constant (all in the middle bin). `edges` replaces the
    exact edges (STUMPY's, for a comparison); the mpmath check runs only on the exact ones."""
    if not all(math.isfinite(x) for x in values):
        return None
    edge_set = exact_edges(bits) if edges is None else tuple(Fraction(e) for e in edges)
    m = len(values)
    ticks = integers(values) if scale is None else [round(x / scale) for x in values]
    if flagged or len(set(ticks)) == 1:
        return [len(edge_set) // 2] * m
    total = sum(ticks)
    variance = m * sum(t * t for t in ticks) - total * total
    if scale is not None:
        if variance >= VARIANCE_LIMIT:
            return None
        if math.sqrt(float(variance)) / m * scale < SMALL_DEVIATION:
            return small_bins(ticks, scale, [float(e) for e in edge_set], tally)
    else:
        exact_values = [Fraction(x) for x in values]
        mean = sum(exact_values) / m
        deviation_squared = sum((x - mean) ** 2 for x in exact_values) / m
        # Without the scale the library's f64 test of the threshold is unknown: stay far from it.
        assert deviation_squared > 4 * Fraction(SMALL_DEVIATION) ** 2, "pass the scale"
    if edges is None:
        return exact_bins(ticks, bits, tally)
    return bins_against(ticks, edge_set)


def linear_bins(values, low, high, bits):
    """STUMPY's non-normalized bin `round((x - low) / (high - low) * (2^bits - 1)) + 1`, ties to
    even, in f64; a ratio that is not a number (equal extremes, or an infinite span) is zero."""
    if not all(math.isfinite(x) for x in values):
        return None
    steps = float(2**bits - 1)
    span = high - low
    out = []
    for x in values:
        ratio = (x - low) / span if span != 0 else math.nan
        scaled = ratio * steps
        out.append((0 if math.isnan(scaled) else round(scaled)) + 1)
    return out


def pair_bins(t, m, i, j, bits=8, normalize=True, scales=None, flags=None, tally=None, edges=None):
    """Per dimension, the bins of the windows at `i` and `j`, or None where either has no distance.
    `flags[q]` is dimension `q`'s per-window constant flags (or None)."""
    t = np.asarray(t, dtype=np.float64)
    if not normalize:
        finite = t[np.isfinite(t)]
        low, high = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 0.0)
    out = []
    for q, row in enumerate(t):
        sides = []
        for s in (i, j):
            window = [float(x) for x in row[s : s + m]]
            if normalize:
                scale = None if scales is None else scales[q]
                flagged = flags is not None and flags[q] is not None and bool(flags[q][s])
                sides.append(window_bins(window, bits, scale, flagged, tally, edges))
            else:
                sides.append(linear_bins(window, low, high, bits))
        out.append(None if None in sides else tuple(sides))
    return out


def squared_distance(bins):
    a, b = bins
    return sum((x - y) ** 2 for x, y in zip(a, b))


def order(keys, include=(), discords=False):
    """Dimensions by key (None last), ties to the smaller dimension, `include` first."""
    d = len(keys)
    sign = -1 if discords else 1
    by_key = sorted(range(d), key=lambda q: (keys[q] is None, sign * (keys[q] or 0), q))
    return [q for q in by_key if q in include] + [q for q in by_key if q not in include]


def reference(t, m, i, j, k, include=(), discords=False, bits=8, normalize=True, scales=None, flags=None, tally=None, edges=None):
    """The subspace of a pair by exact bins, ordered by a plain sort."""
    pairs = pair_bins(t, m, i, j, bits, normalize, scales, flags, tally, edges)
    keys = [None if p is None else squared_distance(p) for p in pairs]
    return order(keys, include, discords)[: k + 1]


def ticks_reference(t, m, i, j, k, include=(), discords=False, bits=8, scale=1 / 64):
    """`reference` with every dimension quantized at `scale`."""
    return reference(t, m, i, j, k, include, discords, bits, scales=[scale] * len(t))


@functools.cache
def log2(n):
    """`log2(n)` correctly rounded: the generator's mpmath value, its rounding decided."""
    return generator().correctly_rounded_log2(n)


def mdl_reference(t, m, first, second, include=(), discords=False, bits=8, normalize=True, scales=None, flags=None, tally=None, log=log2):
    """STUMPY's description length over the exact subspaces and bins, evaluated in its order in
    f64 with the correctly rounded `log2`: `(whole + (s m) log2(distinct)) + distinct bits`.
    `log` replaces the logarithm (NumPy's, to model STUMPY for a comparison)."""
    d = len(t)
    sizes, subspaces = [], []
    for k in range(d):
        pairs = pair_bins(t, m, first[k], second[k], bits, normalize, scales, flags, tally)
        keys = [None if p is None else squared_distance(p) for p in pairs]
        space = order(keys, include, discords)[: k + 1]
        subspaces.append(space)
        if any(pairs[q] is None for q in space):
            sizes.append(math.inf)
            continue
        distinct = {x - y for q in space for x, y in zip(*pairs[q])}
        whole = bits * (2 * d * m - len(space) * m)
        sizes.append((whole + len(space) * m * float(log(len(distinct)))) + len(distinct) * bits)
    return np.array(sizes), subspaces


# ---------- adversarial windows ----------


def four_squares(n):
    """Non-negative integers whose squares sum to `n` (one exists for every `n`, by Lagrange),
    found by a short search from the largest; None if the search misses."""
    for p in range(math.isqrt(n), max(math.isqrt(n) - 64, -1), -1):
        rest = n - p * p
        for q in range(math.isqrt(rest), max(math.isqrt(rest) - 64, -1), -1):
            last = rest - q * q
            for r in range(math.isqrt(last), -1, -1):
                w2 = last - r * r
                if w2 > r * r:
                    break
                w = math.isqrt(w2)
                if w * w == w2:
                    return p, q, r, w
    return None


def first_z(ticks):
    """A window's first offset `a = m x - S` and variance numerator `V`: its z-value is `a / sqrt(V)`."""
    m = len(ticks)
    total = sum(ticks)
    return m * ticks[0] - total, m * sum(t * t for t in ticks) - total * total


def windows_near(top):
    """Two integer windows of length 11 whose first sample's exact z-value lies within one ulp of
    the positive f64 `top`: one strictly between `top` and the next f64 above it, one strictly
    between the next f64 below it and `top`; every other sample of the two is in the same bin.

    A window `[x, p, -p, q, -q, r, -r, w, -w, s, -s]` has `S = x` and `V = 10 x² + 22 N`, `N` the
    sum of the five squares, so its first z-value is `10 x / sqrt(10 x² + 22 N)`, falling as `N`
    grows. `x` is chosen so that `V` is near 2^61; then the `N` of each one-ulp interval spans some
    twenty integers. The upper window takes an integer `P` in its interval written as four squares
    (`s = 0`), the lower one the same four and the smallest `s` that moves `P + s²` into its own;
    the other samples' z-values then move by less than 2^-25, and their bins are checked equal."""
    x = round(top * 2**30.5 / 10)

    def n_of(z):
        z = Fraction(z)
        return (100 * x * x / (z * z) - 10 * x * x) / 22

    up, down = math.nextafter(top, math.inf), math.nextafter(top, -math.inf)
    for p_value in range(math.floor(n_of(up)) + 1, math.ceil(n_of(top))):
        squares = four_squares(p_value)
        if squares is None:
            continue
        s = next((s for s in range(1, 64) if n_of(top) < p_value + s * s < n_of(down)), None)
        if s is None:
            continue
        pairs = [v for value in squares for v in (value, -value)]
        upper, lower = [x, *pairs, 0, 0], [x, *pairs, s, -s]
        a, high_variance = first_z(upper)
        b, low_variance = first_z(lower)
        assert a == b and max(high_variance, low_variance) < 2**62
        bins_upper = bins_against(upper, exact_edges(8))
        bins_lower = bins_against(lower, exact_edges(8))
        inside = above(Fraction(top), a, high_variance) and not above(Fraction(up), a, high_variance)
        inside &= above(Fraction(down), b, low_variance) and not above(Fraction(top), b, low_variance)
        if inside and bins_upper[1:] == bins_lower[1:]:
            return upper, lower
    raise AssertionError(("no windows found", top))


@functools.cache
def adversarial_windows():
    """`{edge index: (above, below)}` for every nonzero eight-bit edge `e`: two integer windows of
    one length whose first sample's exact z-value lies strictly between `e` and the next f64 above
    it (`above`) and strictly between the next f64 below it and `e` (`below`), every other sample of
    the two in the same bin (`windows_near`; a negative edge negates the windows of `-e`)."""
    out = {}
    for q, e in enumerate(exact_edges(8)):
        e = float(e)
        if e > 0:
            out[q] = windows_near(e)
        elif e < 0:
            upper, lower = windows_near(-e)
            out[q] = ([-v for v in lower], [-v for v in upper])
    return out


# ---------- mstump ----------


def compare_correlations(a, b):
    """The sign of `rho_a - rho_b` for correlations `N / sqrt(P)` given as `(N, P)`, exactly."""
    (na, pa), (nb, pb) = a, b
    if (na >= 0) != (nb >= 0):
        return 1 if na >= 0 else -1
    left, right = na * na * pb, nb * nb * pa
    sign = (left > right) - (left < right)
    return sign if na >= 0 else -sign


def approximate_distance(correlation, m):
    """`sqrt(2 m (1 - N / sqrt(P)))` in f64, without cancellation: within a few ulps."""
    n, p = correlation
    root = math.sqrt(p)
    if n >= 0:
        squared = 2 * m * float(p - n * n) / (root * (root + n))
    else:
        squared = 2 * m * (root - n) / root
    return math.sqrt(squared)


def precise_distance(correlation, m):
    """The same distance at 320 bits."""
    n, p = correlation
    with mpmath.workprec(320):
        root = mpmath.sqrt(p)
        if n >= 0:
            return mpmath.sqrt(2 * m * mpmath.mpf(p - n * n) / (root * (root + n)))
        return mpmath.sqrt(2 * m * (root - n) / root)


# A mean within this of the row's best, relative, is a near tie: STUMPY (fastmath f64 on the
# unquantized samples) may order it otherwise than the exact answer.
NEAR_TIE = 1e-6


class MstumpCells:
    """Every cell's per-dimension correlation `(N, P)` (exact integers, `rho = N / sqrt(P)`) and its
    distance in f64, for samples quantized at `scale`; None where a window holds a non-finite
    sample. Computed once, then ordered for any `include` and `discords`."""

    def __init__(self, t, m, scale):
        t = np.asarray(t, dtype=np.float64)
        self.d, n = t.shape
        self.m = m
        self.windows = n - m + 1
        self.exclusion = -(-m // 4)
        ticks = [[round(float(x) / scale) if math.isfinite(x) else None for x in row] for row in t]
        stats = []
        for row in ticks:
            per = []
            for s in range(self.windows):
                w = row[s : s + m]
                if None in w:
                    per.append(None)
                    continue
                total = sum(w)
                variance = m * sum(x * x for x in w) - total * total
                if variance == 0:
                    raise NotImplementedError("constant windows are not in this reference")
                per.append((total, variance))
            stats.append(per)
        self.cells = {}
        for i in range(self.windows):
            for j in range(self.windows):
                if abs(i - j) <= self.exclusion:
                    continue
                cell = []
                for a in range(self.d):
                    if stats[a][i] is None or stats[a][j] is None:
                        cell.append(None)
                        continue
                    (si, vi), (sj, vj) = stats[a][i], stats[a][j]
                    dot = sum(x * y for x, y in zip(ticks[a][i : i + m], ticks[a][j : j + m]))
                    rho = (m * dot - si * sj, vi * vj)
                    cell.append((rho, approximate_distance(rho, m)))
                self.cells[(i, j)] = cell

    def indices(self, include=(), discords=False, tally=None):
        """`I[k][i]`, as `mstump_reference` defines it."""
        include = list(dict.fromkeys(include))
        sign = 1 if discords else -1

        def order(cell):
            def key_cmp(a, b):
                if (cell[a] is None) != (cell[b] is None):
                    return 1 if cell[a] is None else -1
                if cell[a] is not None:
                    c = compare_correlations(cell[a][0], cell[b][0])
                    if c:
                        return sign * c
                return (a > b) - (a < b)

            rest = sorted((q for q in range(self.d) if q not in include), key=functools.cmp_to_key(key_cmp))
            return [cell[q] for q in include + rest]

        index = np.full((self.d, self.windows), -1, dtype=np.int64)
        for i in range(self.windows):
            columns = [(j, order(self.cells[(i, j)])) for j in range(self.windows) if (i, j) in self.cells]
            for k in range(self.d):
                means = [(j, terms[: k + 1]) for j, terms in columns if None not in terms[: k + 1]]
                if means:
                    index[k, i] = smallest_mean(means, self.m, tally)
        return index


def mstump_reference(t, m, scale, include=(), discords=False, tally=None):
    """`mstump`'s indices `I[k][i]` (-1 for none), exactly, for samples quantized at `scale`.

    A dimension's distance between windows `i` and `j` is `sqrt(2 m (1 - rho))`, `rho` the exact
    correlation of their ticks; it has none when either window holds a non-finite sample. Each
    column orders its dimensions: `include` first as given, then the others nearest first
    (farthest first for `discords`), ties to the smaller dimension, those without a distance last.
    `I[k][i]` is the window `j`, `|i - j| > ceil(m / 4)`, with the smallest mean of the first
    `k + 1` distances; a mean holding a missing distance is missing; a tie goes to the smaller `j`.
    `tally.near` counts the rows and `k` whose runner-up is within `NEAR_TIE` of the best.
    """
    return MstumpCells(t, m, scale).indices(include, discords, tally)


def smallest_mean(means, m, tally=None):
    """The `j` of the smallest sum of distances among `(j, [(correlation, f64 distance)])`, ties to
    the smaller `j`."""
    approximate = [(math.fsum(f for _, f in terms), j, terms) for j, terms in means]
    least = min(value for value, _, _ in approximate)
    if tally is not None:
        tally.samples += 1
        tally.near += sum(value <= least * (1 + NEAR_TIE) for value, _, _ in approximate) > 1
    # Each f64 sum is within 1e-14 relative of its exact value; 1e-8 is decidedly outside that.
    close = [(j, terms) for value, j, terms in approximate if value <= least * (1 + 1e-8)]
    if len(close) == 1:
        return close[0][0]
    with mpmath.workprec(320):
        precise = [(mpmath.fsum(precise_distance(c, m) for c, _ in terms), j, terms) for j, terms in close]
        best_value, best_j, best_terms = min(precise, key=lambda x: (x[0], x[1]))
        for value, j, terms in precise:
            if j == best_j or abs(value - best_value) > mpmath.mpf(2) ** -280 * best_value:
                continue
            same = same_distances([c for c, _ in terms], [c for c, _ in best_terms])
            assert same, ("an undecided near tie", j, best_j)
            best_j = min(best_j, j)
    return best_j


def same_distances(a, b):
    """Whether two lists of correlations hold the same multiset of exact values."""
    rest = list(b)
    for x in a:
        found = next((r for r in rest if compare_correlations(x, r) == 0), None)
        if found is None:
            return False
        rest.remove(found)
    return True
