# /// script
# requires-python = ">=3.11"
# dependencies = ["stumpy==1.14.1", "numpy==2.5.4", "numba==0.68.0", "llvmlite==0.50.0"]
# ///
"""Writes STUMPY's `match` output for a few benchmark series to crates/testkit/fixtures/.

Each `<name>.matches` file holds one `key value...` line per field: `m` in decimal, `scale`,
`series` and `query` as 16-digit hex bit patterns, then per case (`default`, `limited`,
`value`) its threshold (`<case>_max`, `default` or the hex of the value passed), its
`<case>_d` hex distances and `<case>_i` decimal indices. Usage: match_stumpy.py <bench data dir>.
"""

import json
import os
import sys
from pathlib import Path

os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np
import stumpy
from stumpy import core

FIXTURES = Path(__file__).resolve().parent.parent / "crates" / "testkit" / "fixtures"
N, M = 400, 16
# The 1e8-offset slices are left out: STUMPY's own sliding statistics lose them (it reports
# distance 0 for nearly every window), so there is nothing to compare against.
# (fixture name, bench series, slice start, query start, noise on the query)
CASES = [
    ("match_walk_exact", "walk_100_tick0p0001_n10000", 1000, 120, 0.0),
    ("match_walk_noisy", "walk_100_tick0p0001_n10000", 3000, 250, 0.002),
    ("match_gaps", "walk_100_tick0p0001_n10000", 6000, 40, 0.0),
]


def f64_hex(values):
    return " ".join(f"{b:016x}" for b in np.asarray(values, np.float64).view(np.uint64))


def main():
    data = Path(sys.argv[1])
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261017)
    for name, source, start, at, noise in CASES:
        scale = json.loads((data / f"{source}.json").read_text())["scale"]
        series = np.fromfile(data / f"{source}.bin", dtype="<f8")[start : start + N].copy()
        if name == "match_gaps":
            series[[75, 210, 211]] = np.nan
        query = series[at : at + M].copy()
        if not np.isfinite(query).all():
            query = series[at + 40 : at + 40 + M].copy()
        query = query + noise * rng.standard_normal(M)
        # A clean query needs no stray rounding: snap to the series' grid.
        query = np.round(query / scale) * scale if noise == 0.0 else query

        profile = core.mass(query, series.copy())
        finite = profile[np.isfinite(profile)]
        value = float(np.median(finite))
        runs = {
            "default": (None, None),
            "limited": (None, 3),
            "value": (value, None),
        }
        lines = [f"m {M}", f"scale {f64_hex([scale])}", f"series {f64_hex(series)}", f"query {f64_hex(query)}"]
        for case, (max_distance, max_matches) in runs.items():
            out = stumpy.match(query, series.copy(), max_distance=max_distance, max_matches=max_matches)
            lines.append(f"{case}_max {'default' if max_distance is None else f64_hex([max_distance])}")
            lines.append(f"{case}_limit {'all' if max_matches is None else max_matches}")
            lines.append(f"{case}_d {f64_hex([float(r[0]) for r in out])}")
            lines.append(f"{case}_i {' '.join(str(int(r[1])) for r in out)}")
        (FIXTURES / f"{name}.matches").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
