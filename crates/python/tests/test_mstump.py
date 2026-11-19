"""`mstump(T, m, include, discords, normalize, p, scales, threads)`: STUMPY's `(P, I)` on random
walks, the same bits at any thread count, one dimension equal to the two-sided profile, and every
bad argument refused."""

import math
import struct
from pathlib import Path

import numpy as np
import pytest
from ladders import matrix_profile, mstump

from conftest import random_walk
from multi_reference import MstumpCells, Tally

M = 8
GRID = 1.0 / (1 << 26)
THREADS = [1, 2, 3, 8, None, np.int64(4)]


def walks(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((d, n)), axis=1)


def bits(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def test_indices_equal_the_exact_reference_and_stumpy_beside_it():
    """The indices are exactly calculable: each distance's order is decided on integers and a mean's
    by mpmath where f64 cannot tell (multi_reference.mstump_reference), so they are checked
    exactly. STUMPY is a comparison: its f64 means on the unquantized samples can order two windows
    otherwise only when their means are within `NEAR_TIE` of each other; the reference counts those
    rows, none here, so STUMPY's indices must equal the exact ones. The distances themselves are
    sums of rounded square roots, not exactly calculable, so they are compared within a tolerance."""
    stumpy = pytest.importorskip("stumpy")
    t = walks(3, 200, seed=1)
    cells = MstumpCells(t, M, GRID)
    for include, discords in [(None, False), ([2], False), ([1, 0], False), (None, True), ([2], True)]:
        tally = Tally()
        want = cells.indices(include or (), discords, tally)
        p, i = mstump(t, M, include=include, discords=discords, scales=GRID)
        assert p.shape == i.shape == want.shape == (3, 200 - M + 1)
        assert p.dtype == np.float64 and i.dtype == np.int64
        assert np.array_equal(i, want), (include, discords)
        stumpy_p, stumpy_i = stumpy.mstump(t, M, include=include, discords=discords)
        assert tally.near == 0 and tally.samples == want.size
        assert np.array_equal(stumpy_i.astype(np.int64), want), (include, discords)
        assert np.allclose(p, stumpy_p, rtol=1e-6, atol=1e-6), (include, discords)


FIXTURES = Path(__file__).resolve().parents[2] / "testkit" / "fixtures"


def test_the_golden_fixtures_hold_the_exact_indices_and_stumpys_beside_them():
    """Each `mstump_*.mgolden` fixture's `exact_i`, which the Rust tests take as the expected
    indices, is the exact reference recomputed here from the fixture's own series; STUMPY's
    `stumpy_i` beside it is equal, since no row of these walks has a near tie (counted)."""
    paths = sorted(FIXTURES.glob("mstump_*.mgolden"))
    assert len(paths) >= 8
    for path in paths:
        fields = dict((line.split(" ", 1) + [""])[:2] for line in path.read_text().splitlines())
        d, m = int(fields["d"]), int(fields["m"])
        series = np.array(
            [[struct.unpack(">d", bytes.fromhex(h))[0] for h in fields[f"series{a}"].split()] for a in range(d)]
        )
        include = [int(x) for x in fields["include"].split()]

        def indices(key):
            return np.array([int(x) for x in fields[key].split()], dtype=np.uint64).astype(np.int64).reshape(d, -1)

        tally = Tally()
        want = MstumpCells(series, m, 1.0 / (1 << 26)).indices(include, fields["discords"] == "1", tally)
        assert np.array_equal(indices("exact_i"), want), path.name
        assert tally.near == 0 and np.array_equal(indices("stumpy_i"), want), path.name


def test_default_scales_are_inferred_per_dimension():
    t = walks(2, 150, seed=2)
    p, i = mstump(t, M)
    assert np.isfinite(p).all() and (i >= 0).all()


def test_the_thread_count_never_changes_a_bit():
    rng = np.random.default_rng(3)
    levels = rng.integers(0, 3, size=(3, 160)).astype(np.float64)
    levels[1] = 2.0 * levels[0] + 1.0
    levels[2, 40] = np.nan
    want = mstump(levels, M, include=[2], scales=1.0, threads=1)
    for threads in THREADS:
        p, i = mstump(levels, M, include=[2], scales=1.0, threads=threads)
        assert np.array_equal(bits(p), bits(want[0])) and np.array_equal(i, want[1]), threads


def test_one_dimension_equals_the_two_sided_profile():
    t = random_walk(300, seed=4)
    p, i = mstump(t[None, :], M, scales=1.0)
    want_p, want_i, _, _ = matrix_profile(t, M, scale=1.0)
    assert np.array_equal(bits(p[0]), bits(want_p)) and np.array_equal(i[0], want_i)


def test_tied_scaled_copies_choose_the_smaller_dimension():
    base = random_walk(120, seed=5)
    t = np.stack([base, 3.0 * base + 7.0])
    p, i = mstump(t, M, scales=1.0)
    assert np.array_equal(i[0], i[1])
    assert np.allclose(p[0], p[1], rtol=1e-12)
    single, _ = mstump(base[None, :], M, scales=1.0)
    assert np.array_equal(bits(p[0]), bits(single[0]))


def test_a_gap_makes_its_windows_missing_in_that_dimension_only():
    t = walks(2, 100, seed=6)
    t[1, 50] = np.nan
    p, i = mstump(t, M, scales=GRID)
    assert np.isfinite(p[0]).all()
    covered = slice(50 - M + 1, 51)
    assert np.isinf(p[1, covered]).all() and (i[1, covered] == -1).all()


def test_scales_as_one_number_a_list_or_inferred():
    t = walks(2, 80, seed=7)
    one = mstump(t, M, scales=GRID)
    listed = mstump(t, M, scales=[GRID, GRID])
    assert np.array_equal(bits(one[0]), bits(listed[0]))
    assert np.array_equal(one[1], listed[1])


@pytest.mark.parametrize(
    "call, error",
    [
        (lambda t: mstump(t, M, p=1.0), NotImplementedError),
        (lambda t: mstump(t, 2), ValueError),
        (lambda t: mstump(t, M, include=[-1]), ValueError),
        (lambda t: mstump(t, M, include=[2]), ValueError),
        (lambda t: mstump(t, M, scales=[1.0]), ValueError),
        (lambda t: mstump(t, M, scales=[1.0, -1.0]), ValueError),
        (lambda t: mstump(t, M, scales="fine"), ValueError),
        (lambda t: mstump(t, 50), ValueError),
        (lambda t: mstump(t, M, threads=0), ValueError),
        (lambda t: mstump(t[:0], M), ValueError),
    ],
)
def test_bad_arguments(call, error):
    with pytest.raises(error):
        call(walks(2, 30, seed=8))


def test_duplicate_include_entries_count_once():
    t = walks(3, 90, seed=9)
    once = mstump(t, M, include=[1], scales=GRID)
    twice = mstump(t, M, include=[1, 1], scales=GRID)
    assert np.array_equal(bits(once[0]), bits(twice[0])) and np.array_equal(once[1], twice[1])


def test_exact_float_brute_force_on_a_small_set():
    t = walks(3, 40, seed=10)
    m = 5
    p, _ = mstump(t, m, scales=GRID)
    windows = t.shape[1] - m + 1
    exclusion = -(-m // 4)

    def z(x):
        return (x - x.mean()) / x.std()

    for row in range(windows):
        best = [math.inf] * 3
        for col in range(windows):
            if abs(row - col) <= exclusion:
                continue
            d = sorted(np.linalg.norm(z(t[a, row : row + m]) - z(t[a, col : col + m])) for a in range(3))
            for k in range(3):
                best[k] = min(best[k], sum(d[: k + 1]) / (k + 1))
        assert np.allclose(p[:, row], best, rtol=1e-6)
