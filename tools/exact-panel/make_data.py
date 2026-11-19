"""Seeded series (the same formulas as the timed benchmark's synthetic datasets), raw little-endian float64 files
plus manifest.json with sha256, seed, the exact power-of-two scale and the tick magnitude.
Side A is the benchmark series (extra=0); side B is an independent draw (seed extra=1) used as the
second series of the AB-join."""
import hashlib, json, math, sys
import numpy as np
from fractions import Fraction
from cells import *

SEED_BASE = 20261026_000
GRID_TICK, GRID_START_TICKS, MAX_STEP = 0.01, 10_000, 2
BAD_PRINT_FACTORS = (1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DATASETS = ("grid", "float", "ties", "messy_b1", "messy_b10", "messy_b30", "messy_b100")


def seed(name, n, extra=0):
    return SEED_BASE + 1000 * (DATASETS.index(name) + 1) + 10 * int(np.log2(n)) + extra


def _walk(rng, n):
    steps = rng.integers(-MAX_STEP, MAX_STEP + 1, size=n - 1)
    return GRID_START_TICKS + np.concatenate(([0], np.cumsum(steps)))


def _ties(rng, n):
    out = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        kind = rng.integers(0, 3)
        length = int(rng.integers(64, 1024))
        if kind == 0:
            runs = np.repeat(rng.integers(0, 3, size=length), rng.integers(1, 9, size=length))[:length]
            block = runs.astype(np.float64)
        elif kind == 1:
            block = np.resize(rng.integers(0, 3, size=7).astype(np.float64), length)
        else:
            block = np.full(length, float(rng.integers(0, 3)))
        take = min(length, n - i)
        out[i : i + take] = block[:take]
        i += take
    return out


def _messy(rng, n, bad_prints):
    ticks = _walk(rng, n)
    for _ in range(3):
        start = int(rng.integers(0, n - n // 32))
        ticks[start : start + n // 64] = ticks[start]
    series = ticks.astype(np.float64) * GRID_TICK
    positions = rng.choice(n, size=bad_prints, replace=False)
    factors = np.asarray(BAD_PRINT_FACTORS)[rng.integers(0, len(BAD_PRINT_FACTORS), size=bad_prints)]
    series[positions] = np.round(series[positions] * factors / GRID_TICK) * GRID_TICK
    gaps = [(0, 3), (n - 2, 2)]
    gaps += [(int(rng.integers(0, n - 20)), int(rng.integers(1, 21))) for _ in range(4)]
    for start, length in gaps:
        series[start : start + length] = np.nan
    return series


def build(name, n, extra=0):
    rng = np.random.default_rng(seed(name, n, extra))
    if name == "grid":
        return _walk(rng, n).astype(np.float64) * GRID_TICK
    if name == "float":
        return 100.0 + np.cumsum(rng.random(n) - 0.5)
    if name == "ties":
        return _ties(rng, n)
    if name.startswith("messy_b"):
        return _messy(rng, n, int(name[len("messy_b"):]))
    raise ValueError(name)


def exact_scale_exp(x):
    """Largest e with every finite sample an integer multiple of 2**e (0 if all zero), and max |ticks|."""
    emin = None
    for v in x:
        if not math.isfinite(v) or v == 0.0:
            continue
        num, den = float(v).as_integer_ratio()
        e = (num & -num).bit_length() - 1 - (den.bit_length() - 1)
        emin = e if emin is None else min(emin, e)
    emin = 0 if emin is None else emin
    mx = max((abs(Fraction(float(v)) / Fraction(2) ** emin) for v in x if math.isfinite(v)), default=0)
    return emin, int(mx)


def main():
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    man = {}
    for kind in KINDS:
        for n in NS:
            for side, extra in (("A", 0), ("B", 1)):
                x = build(kind, n, extra)
                p = data_path(kind, n, side)
                p.write_bytes(x.astype("<f8").tobytes())
                e, mx = exact_scale_exp(x)
                man[p.name] = dict(kind=kind, n=n, side=side, seed=seed(kind, n, extra),
                                   sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                   nan_count=int(np.isnan(x).sum()), scale_exp2=e,
                                   max_abs_ticks_log2=round(math.log2(mx), 2) if mx else 0,
                                   fits_2_53=bool(mx <= 2**53))
    (OUT / "data" / "manifest.json").write_text(json.dumps(man, indent=1))
    bad = [k for k, v in man.items() if not v["fits_2_53"]]
    print(f"wrote {len(man)} series; not fitting |ticks|<=2^53 at the exact power-of-two scale: {bad}")


if __name__ == "__main__":
    main()
