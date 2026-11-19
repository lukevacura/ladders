"""Exact reference for ladde. Pure Python: ints, Fractions, math.isqrt.

Independent of the Rust oracle: nothing here imports the crate. Inputs are integer tick
sequences (`None` = missing) and the window length `m`. O(n^2 * m); for small n (<= a few hundred).
See README.md for the rules.
"""
from fractions import Fraction
from math import isqrt, floor, isfinite
from typing import List, Optional, Sequence, Tuple

NONFINITE, CONSTANT, FINITE = "nonfinite", "constant", "finite"
V_LIMIT = 1 << 63          # V = m*sum(t^2) - (sum t)^2 must be < 2^63
MAX_TICKS = 1 << 53        # quantize.rs MAX_TICKS


def exclusion_zone(m: int) -> int:
    """e = ceil(m/4); |i-j| <= e is excluded in self-joins."""
    return -(-m // 4)


def quantize(x, scale: float) -> Optional[int]:
    """Ticks of one sample: round-half-even of x/scale in f64; None for NaN/inf or |ticks| > 2^53
    (mirrors crates/core quantize.rs `to_ticks`)."""
    if x is None:
        return None
    x = float(x)
    if not isfinite(x):
        return None
    q = x / float(scale)
    if not isfinite(q):
        return None
    t = round(q)  # Python float round() is round-half-even
    return None if abs(t) > MAX_TICKS else int(t)


def quantize_series(xs, scale: float) -> List[Optional[int]]:
    return [quantize(x, scale) for x in xs]


class Win:
    """A window's exact statistics. kind in {nonfinite, constant, finite} (non-finite first)."""
    __slots__ = ("t", "kind", "S", "V")

    def __init__(self, t: Sequence[Optional[int]], m: int):
        assert len(t) == m
        self.t = t
        if any(x is None for x in t):
            self.kind, self.S, self.V = NONFINITE, 0, 0
            return
        S = sum(t)
        V = m * sum(x * x for x in t) - S * S
        self.S, self.V = S, V
        if V >= V_LIMIT:
            self.kind = NONFINITE
        elif max(t) == min(t):
            self.kind = CONSTANT
        else:
            self.kind = FINITE


def windows(ticks: Sequence[Optional[int]], m: int) -> List[Win]:
    return [Win(ticks[i:i + m], m) for i in range(len(ticks) - m + 1)]


def window_kind(w: Sequence[Optional[int]]) -> str:
    return Win(w, len(w)).kind


class Rho:
    """Exact correlation rho = sign * sqrt(r), r = num/den kept as integers (never a float).
    Total order: larger rho == smaller distance. Comparisons are exact cross-multiplications."""
    __slots__ = ("sign", "num", "den")

    def __init__(self, sign, num, den):
        self.sign, self.num, self.den = sign, num, den

    @property
    def r(self) -> Fraction:
        """rho^2 as a Fraction (exact)."""
        return Fraction(self.num, self.den)

    def cmp(self, o: "Rho") -> int:
        """+1 if self is a larger correlation (nearer) than o, 0 if equal, -1 otherwise."""
        if self.sign != o.sign:
            return 1 if self.sign > o.sign else -1
        if self.sign == 0:
            return 0
        a, b = self.num * o.den, o.num * self.den
        c = (a > b) - (a < b)
        return c if self.sign > 0 else -c

    def is_rational(self) -> bool:
        r = self.r
        return self.sign == 0 or (isqrt(r.numerator) ** 2 == r.numerator
                                  and isqrt(r.denominator) ** 2 == r.denominator)

    def value(self, bits: int = 300) -> Fraction:
        """rho as a Fraction: exact when rational, else within 2^-bits."""
        if self.sign == 0:
            return Fraction(0)
        r = Fraction(self.num, self.den)
        p, q = r.numerator, r.denominator
        if isqrt(p) ** 2 == p and isqrt(q) ** 2 == q:
            return self.sign * Fraction(isqrt(p), isqrt(q))
        s = isqrt((p << (2 * bits)) // q)  # floor(sqrt(r) * 2^bits)
        return self.sign * Fraction(s, 1 << bits)

    def __repr__(self):
        return "Rho(sign=%d, rho^2=%s)" % (self.sign, self.r)


RHO_ONE = Rho(1, 1, 1)     # constant vs constant: d = 0
RHO_HALF = Rho(1, 1, 4)    # constant vs non-constant: rho = 1/2, d^2 = m


def rho(a: Win, b: Win, m: int) -> Optional[Rho]:
    """Exact correlation of two windows, or None if either is non-finite.
    const/const -> 1 (d=0); const/non-const -> 1/2 (d^2 = m); else c / sqrt(Va*Vb),
    c = m*sum(xy) - Sa*Sb."""
    if a.kind == NONFINITE or b.kind == NONFINITE:
        return None
    if a.kind == CONSTANT and b.kind == CONSTANT:
        return RHO_ONE
    if a.kind == CONSTANT or b.kind == CONSTANT:
        return RHO_HALF
    c = m * sum(x * y for x, y in zip(a.t, b.t)) - a.S * b.S
    s = (c > 0) - (c < 0)
    return Rho(s, c * c, a.V * b.V)


def corr(wa, wb) -> Optional[Rho]:
    """Exact correlation between two tick windows (lists of ints/None)."""
    m = len(wa)
    return rho(Win(wa, m), Win(wb, m), m)


def d2_of(r: Rho, m: int, bits: int = 300) -> Fraction:
    """Squared distance 2m(1 - rho), clamped at 0. Exact when rho is rational (always for the
    constant cases and for exact repeats), else within 2m*2^-bits."""
    if r.sign > 0 and r.num == r.den:
        return Fraction(0)
    return max(Fraction(0), 2 * m * (1 - r.value(bits)))


def dist2(wa, wb) -> Optional[Fraction]:
    """Exact (see d2_of) squared z-normalized distance between two tick windows; None if non-finite."""
    r = corr(wa, wb)
    return None if r is None else d2_of(r, len(wa))


def dist_f64(d2: Fraction) -> float:
    """sqrt(d2) correctly rounded to the nearest f64 (sqrt computed to 2^-300 first)."""
    if d2 <= 0:
        return 0.0
    k = 300
    s = isqrt(floor(d2 * (1 << (2 * k))))
    return float(Fraction(s, 1 << k))


def dist_f64_of_rho(r: Rho, m: int) -> float:
    return dist_f64(d2_of(r, m))


# ---------------------------------------------------------------- profiles
# Entry = (index, Rho) or None (no neighbour). Distance of an entry: dist_f64_of_rho(rho, m).

def _better(r: Rho, best: Optional[Rho]) -> bool:
    """Strictly better than best. Scanning candidates in increasing index, strict > keeps the
    smaller index on ties."""
    return best is None or r.cmp(best) > 0


def _search(W: List[Win], m: int, i: int, js) -> Optional[Tuple[int, Rho]]:
    wi = W[i]
    best = None
    for j in js:
        r = rho(wi, W[j], m)
        if r is not None and _better(r, None if best is None else best[1]):
            best = (j, r)
    return best


def left_profile(ticks, m):
    """Per window i: nearest j with j < i - e. None where i is non-finite or no candidate."""
    W, e = windows(ticks, m), exclusion_zone(m)
    return [_search(W, m, i, range(0, i - e)) if W[i].kind != NONFINITE else None
            for i in range(len(W))]


def right_profile(ticks, m):
    """Per window i: nearest j with j > i + e."""
    W, e = windows(ticks, m), exclusion_zone(m)
    return [_search(W, m, i, range(i + e + 1, len(W))) if W[i].kind != NONFINITE else None
            for i in range(len(W))]


def two_sided_profile(ticks, m):
    """Returns (nearest, left, right): nearest = better of left/right, left wins ties (= smaller index)."""
    L, R = left_profile(ticks, m), right_profile(ticks, m)
    nearest = []
    for l, r in zip(L, R):
        if l is None:
            nearest.append(r)
        elif r is None or l[1].cmp(r[1]) >= 0:
            nearest.append(l)
        else:
            nearest.append(r)
    return nearest, L, R


def topk_left(ticks, m, k):
    """Per window: up to k nearest left candidates (j < i - e), nearest first, ties smaller index.
    Trivial matches of each other are NOT removed."""
    import functools
    W, e = windows(ticks, m), exclusion_zone(m)
    out = []
    for i in range(len(W)):
        if W[i].kind == NONFINITE:
            out.append([])
            continue
        c = [(j, rho(W[i], W[j], m)) for j in range(0, i - e)]
        c = [x for x in c if x[1] is not None]
        c.sort(key=functools.cmp_to_key(lambda a, b: -a[1].cmp(b[1]) or a[0] - b[0]))
        out.append(c[:k])
    return out


def ab_join(a_ticks, b_ticks, m):
    """Per A window: nearest over ALL B windows, no exclusion zone, ties smaller B index,
    non-finite B windows never chosen; non-finite A window -> None. A and B are windowed separately."""
    WA, WB = windows(a_ticks, m), windows(b_ticks, m)
    out = []
    for wa in WA:
        out.append(None if wa.kind == NONFINITE else _search_ab(wa, WB, m))
    return out


def _search_ab(wa, WB, m):
    best = None
    for j, wb in enumerate(WB):
        r = rho(wa, wb, m)
        if r is not None and _better(r, None if best is None else best[1]):
            best = (j, r)
    return best


def discords(ticks, m, k, side="two_sided"):
    """Greedy farthest-first: order windows with a neighbour by exact distance, largest
    first (ties smaller index); take a window if it is outside e of every one already taken;
    stop at k. Windows without any neighbour are excluded. `side` picks the profile the score
    comes from: "two_sided" (the batch discord list), "left" or "right". Returns [(index, Rho)]."""
    import functools
    prof = {"two_sided": lambda: two_sided_profile(ticks, m)[0],
            "left": lambda: left_profile(ticks, m),
            "right": lambda: right_profile(ticks, m)}[side]()
    e = exclusion_zone(m)
    cand = [(i, p[1]) for i, p in enumerate(prof) if p is not None]
    cand.sort(key=functools.cmp_to_key(lambda a, b: a[1].cmp(b[1]) or a[0] - b[0]))  # lowest rho first
    out = []
    for i, r in cand:
        if len(out) == k:
            break
        if all(abs(i - q) > e for q, _ in out):
            out.append((i, r))
    return out


# Motifs are NOT covered: their default cutoff/max_distance/min_neighbors rules and f64-threshold
# semantics are not pure functions of exact values; see README.

# ---------------------------------------------------------------- comparison helpers

def entry_index(p) -> int:
    return -1 if p is None else p[0]


def entry_distance_f64(p, m) -> float:
    """Nearest f64 to the exact distance of an entry (inf when there is no neighbour)."""
    return float("inf") if p is None else dist_f64_of_rho(p[1], m)


def within_tol(got: float, p, m: int, rel: float = 1e-14, abs_: float = 1e-12) -> bool:
    """|got - exact| <= max(abs_, rel*exact); inf must match inf."""
    exact = entry_distance_f64(p, m)
    if exact == float("inf") or got == float("inf"):
        return got == exact
    return abs(got - exact) <= max(abs_, rel * exact)


def compare_profile(got_idx, got_dist, ref, m, rel=1e-14, abs_=1e-12):
    """Compare engine (indices, distances) with a reference profile list. Returns a list of
    mismatch strings (empty = pass): exact index equality, distance within tolerance."""
    bad = []
    if len(got_idx) != len(ref):
        return ["length %d != %d" % (len(got_idx), len(ref))]
    for i, p in enumerate(ref):
        if int(got_idx[i]) != entry_index(p):
            bad.append("row %d: index %d != exact %d" % (i, int(got_idx[i]), entry_index(p)))
        elif not within_tol(float(got_dist[i]), p, m, rel, abs_):
            bad.append("row %d: dist %r vs exact %r" % (i, float(got_dist[i]), entry_distance_f64(p, m)))
    return bad
