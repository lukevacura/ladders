"""Exact truth with exact_ref.

'exact'    : the doubles AS PASSED. Every finite double is a dyadic rational, so at scale 2**e
             (e = the largest power of two dividing every sample) each sample is an exact integer
             tick count; ticks are Python ints (no 2^53 cap: that cap is exact_ref.quantize's, not
             part of the maths) and V_LIMIT is lifted, so no window is dropped as 'too large'.
'contract' : ours' contract: quantize (exact_ref.quantize, round-half-even of x/scale) at the scale
             ours reported, exact_ref's own V_LIMIT 2^63 and 2^53 rules.
Rules (exact_ref): exclusion zone e=ceil(m/4), candidate j needs |i-j| > e (two-sided = better of
left j<i-e and right j>i+e, left wins ties); AB-join has no exclusion zone, ties -> smaller B index;
const/const d=0, const/non-const d=sqrt(m); a window with a NaN has no answer and is never a candidate.
"""
import json, math, sys
from fractions import Fraction
from multiprocessing import Pool
import numpy as np
import exact_ref as X
from cells import *

V_LIMIT_CONTRACT = X.V_LIMIT
V_LIMIT_OFF = 1 << 4096


def exact_ticks(arr):
    emin = None
    for v in arr:
        v = float(v)
        if not math.isfinite(v) or v == 0.0:
            continue
        num, den = v.as_integer_ratio()
        e = (num & -num).bit_length() - 1 - (den.bit_length() - 1)
        emin = e if emin is None else min(emin, e)
    emin = 0 if emin is None else emin
    sc = Fraction(2) ** emin
    out = []
    for v in arr:
        v = float(v)
        if not math.isfinite(v):
            out.append(None)
        else:
            q = Fraction(v) / sc
            assert q.denominator == 1
            out.append(int(q))
    return out, emin


def contract_ticks(arr, scale):
    return X.quantize_series([float(v) for v in arr], float(scale))


def f32_round(s: Fraction) -> float:
    """Nearest float32 (ties to even) of a non-negative Fraction, returned as a Python float."""
    if s <= 0:
        return 0.0
    e = s.numerator.bit_length() - s.denominator.bit_length() - 1
    e = max(e, -126) - 23  # ulp exponent (subnormals clamp)
    q = s / Fraction(2) ** e
    n = q.numerator // q.denominator
    r = q - n
    if r > Fraction(1, 2) or (r == Fraction(1, 2) and n & 1):
        n += 1
    return math.ldexp(n, e)


def d_values(r, m):
    """(exact d2 Fraction, f64-rounded distance, f32-rounded distance)."""
    d2 = X.d2_of(r, m)
    k = 300
    sq = Fraction(math.isqrt(math.floor(d2 * (1 << (2 * k)))), 1 << k)
    return d2, X.dist_f64(d2), f32_round(sq)


def pack(entries, m):
    idx, rho, d, d32, d2 = [], [], [], [], []
    for e in entries:
        if e is None:
            idx.append(-1); rho.append(None); d.append(None); d32.append(None); d2.append(None)
        else:
            j, r = e
            q, dd, d3 = d_values(r, m)
            idx.append(j); rho.append([r.sign, str(r.num), str(r.den)])
            d.append(dd); d32.append(d3); d2.append(f"{q.numerator}/{q.denominator}")
    return dict(idx=idx, rho=rho, d=d, d32=d32, d2=d2)


def solve(task):
    wl, mode, m, ta, tb = task  # mode exact | contract
    X.V_LIMIT = V_LIMIT_OFF if mode == "exact" else V_LIMIT_CONTRACT
    if wl == "two_sided":
        ent = X.two_sided_profile(ta, m)[0]
    else:
        ent = X.ab_join(ta, tb, m)
    return pack(ent, m)


def load(kind, n, side):
    return np.fromfile(data_path(kind, n, side), dtype="<f8")


def tasks_for(kind, n, m, wl, odir):
    """[(name, task)] for one cell: exact, plus a contract truth per distinct ours scale."""
    A, B = load(kind, n, "A"), load(kind, n, "B")
    ta, _ = exact_ticks(A)
    tb = exact_ticks(B)[0] if wl == "ab_join" else None
    out = [("exact", (wl, "exact", m, ta, tb))]
    seen = {}
    for var in ("default", "finest"):
        sf = odir / "ours" / var / (key(kind, n, m, wl) + ".scale.npy")
        sc = tuple(float(s) for s in np.load(sf))
        if sc in seen:
            out.append((f"contract_{var}", ("alias", seen[sc])))
            continue
        seen[sc] = f"contract_{var}"
        ca = contract_ticks(A, sc[0])
        cb = contract_ticks(B, sc[1]) if wl == "ab_join" else None
        out.append((f"contract_{var}", (wl, "contract", m, ca, cb)))
    return out


def main():
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    (OUT / "truth").mkdir(parents=True, exist_ok=True)
    jobs, meta = [], []
    for (kind, n, m, wl) in cells():
        for name, t in tasks_for(kind, n, m, wl, OUT):
            if t[0] == "alias":
                meta.append((kind, n, m, wl, name, t[1], None))
            else:
                meta.append((kind, n, m, wl, name, None, len(jobs))); jobs.append(t)
    # heaviest first
    order = sorted(range(len(jobs)), key=lambda i: -len(jobs[i][3]) * len(jobs[i][3]) * jobs[i][2])
    with Pool(procs) as pool:
        res = dict(zip(order, pool.imap(solve, [jobs[i] for i in order], chunksize=1)))
    byname = {}
    for kind, n, m, wl, name, alias, j in meta:
        byname[(kind, n, m, wl, name)] = res[j] if j is not None else byname[(kind, n, m, wl, alias)]
    for (kind, n, m, wl, name), r in byname.items():
        (OUT / "truth" / f"{key(kind, n, m, wl)}.{name}.json").write_text(json.dumps(r))
    print("truth written:", len(byname))


def selftest():
    t = np.array([1.0, 3.0, 2.0, 5.0, 1.0, 3.0, 2.5, 5.0, 4.0, 4.0, 4.0, 4.0])
    m = 4
    tk, e = exact_ticks(t)
    X.V_LIMIT = V_LIMIT_OFF
    ent = X.two_sided_profile(tk, m)[0]
    ex = pack(ent, m)
    # brute force in float
    W = np.array([t[i:i + m] for i in range(len(t) - m + 1)])
    ez = -(-m // 4)
    print(f"scale 2^{e}, n=12 m=4, exclusion: need |i-j| > {ez}")
    print("row | exact_ref (idx, d)            | brute float (idx, d)")
    for i in range(len(W)):
        best = (-1, math.inf)
        for j in range(len(W)):
            if abs(i - j) <= ez:
                continue
            a, b = W[i], W[j]
            if np.ptp(a) == 0 and np.ptp(b) == 0: d = 0.0
            elif np.ptp(a) == 0 or np.ptp(b) == 0: d = math.sqrt(m)
            else:
                za, zb = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
                d = float(np.sqrt(((za - zb) ** 2).sum()))
            if d < best[1] - 1e-12:
                best = (j, d)
        print(f"{i:3d} | {ex['idx'][i]:3d} {ex['d'][i]!s:22.20s} | {best[0]:3d} {best[1]:.17g}")


if __name__ == "__main__":
    selftest() if sys.argv[1:2] == ["--selftest"] else main()
