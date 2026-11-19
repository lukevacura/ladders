# /// script
# requires-python = ">=3.11"
# dependencies = ["stumpy==1.14.1", "numpy==2.5.4", "numba==0.68.0", "llvmlite==0.50.0", "mpmath==1.4.1"]
# ///
"""Writes each multi-dimensional golden case to crates/testkit/fixtures/: the exact `mstump`
indices, the expected answer, and STUMPY's `mstump` output beside them, a comparison.

Each `mstump_<name>.mgolden` holds one `key value…` line per field: `d`, `m`, `discords` (0 or 1)
and `include` (decimal), `series<a>` per dimension as 16-digit hex bit patterns, then `exact_i`,
`stumpy_p` and `stumpy_i`, row-major over (dimension count k, window): f64 as hex bits, indices in
decimal with -1 written as u64::MAX. `exact_i` is crates/python/tests/multi_reference.py's
`mstump_reference` at the tests' scale 2^-26: each distance's order decided on integers, a mean's
by mpmath where f64 cannot tell, ties to the smaller window; no library code computes it. Every
series is a random walk, so no two means nearly tie and STUMPY's indices equal the exact ones (this
script checks it). No high-offset case: `mstump`'s FFT first row loses the offset's precision (about
5e-4 at 1e6 here), which is STUMPY's error, not a reference's.

`--exact` rewrites only the `exact_i` lines of the committed fixtures, from their own series,
without running STUMPY.
"""

import os
import struct
import sys

os.environ["NUMBA_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIXTURES = REPO / "crates" / "testkit" / "fixtures"
NONE = 2**64 - 1
SCALE = 2.0**-26
sys.path.insert(0, str(REPO / "crates" / "python" / "tests"))

from multi_reference import Tally, mstump_reference  # noqa: E402


def walks(rng, d, n):
    return np.cumsum(rng.standard_normal((d, n)), axis=1)


def cases():
    rng = np.random.default_rng(20261020)
    plain = walks(rng, 3, 48)
    gaps = walks(rng, 3, 48)
    gaps[1, 20] = np.nan
    gaps[1, 33] = np.inf
    gaps[2, 0] = np.nan
    return [
        ("one_dimension", walks(rng, 1, 40), 5, [], False),
        ("plain", plain, 6, [], False),
        ("include_one", walks(rng, 4, 44), 5, [1], False),
        ("include_two_in_order", walks(rng, 4, 44), 5, [2, 0], False),
        ("discords", walks(rng, 3, 48), 6, [], True),
        ("discords_include", walks(rng, 4, 44), 5, [3], True),
        ("gaps", gaps, 6, [], False),
        ("tiny_m", walks(rng, 3, 30), 3, [], False),
    ]


def f64_hex(values):
    return " ".join(f"{bits:016x}" for bits in np.asarray(values, np.float64).ravel().view(np.uint64))


def indices(values):
    values = np.asarray(values, np.int64).ravel()
    assert (values >= -1).all()
    return " ".join(str(NONE if v == -1 else int(v)) for v in values)


def exact_indices(series, m, include, discords):
    tally = Tally()
    exact = mstump_reference(series, m, SCALE, include, discords, tally)
    assert tally.near == 0, "a near tie: STUMPY may differ, list it in the tests"
    return exact


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    import stumpy

    for name, series, m, include, discords in cases():
        p, i = stumpy.mstump(series, m, include=include or None, discords=discords)
        exact = exact_indices(series, m, include, discords)
        assert np.array_equal(exact, np.asarray(i, np.int64)), name
        lines = [
            f"d {series.shape[0]}",
            f"m {m}",
            f"discords {int(discords)}",
            "include " + " ".join(map(str, include)),
        ]
        lines += [f"series{a} {f64_hex(row)}" for a, row in enumerate(series)]
        lines += [f"exact_i {indices(exact)}", f"stumpy_p {f64_hex(p)}", f"stumpy_i {indices(i)}"]
        (FIXTURES / f"mstump_{name}.mgolden").write_text("\n".join(lines) + "\n")


def rewrite_exact():
    """Recomputes `exact_i` in every committed fixture from its own series."""
    for path in sorted(FIXTURES.glob("mstump_*.mgolden")):
        lines = path.read_text().splitlines()
        fields = dict((line.split(" ", 1) + [""])[:2] for line in lines)
        d, m = int(fields["d"]), int(fields["m"])
        series = np.array(
            [[struct.unpack(">d", bytes.fromhex(h))[0] for h in fields[f"series{a}"].split()] for a in range(d)]
        )
        include = [int(x) for x in fields["include"].split()]
        exact = f"exact_i {indices(exact_indices(series, m, include, fields['discords'] == '1'))}"
        kept = [line for line in lines if not line.startswith("exact_i ")]
        at = next(k for k, line in enumerate(kept) if line.startswith("stumpy_p "))
        path.write_text("\n".join(kept[:at] + [exact] + kept[at:]) + "\n")


if __name__ == "__main__":
    rewrite_exact() if "--exact" in sys.argv else main()
