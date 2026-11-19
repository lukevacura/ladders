"""`subspace(t, m, subseq_idx, nn_idx, k, ...)`: STUMPY's subspace of a motif pair, the bins decided
exactly, `discords` doing what its documentation says, a gap making a dimension missing (last),
and bad arguments a `ValueError` or `NotImplementedError`.

The basis is the exact reference (multi_reference): every result is checked against it with no
tolerance. STUMPY is a comparison beside it. Its edges, scipy's `norm.ppf`, are within 1e-12
relative of the correctly rounded quantiles on every platform, so it can bin a sample otherwise only
when the sample's z-value is within `NEAR_EDGE` (2^-36) of an edge; the reference counts those
samples, and where there are none STUMPY must agree exactly. The adversarial test places samples
within one ulp of every edge, where the two differ."""

import re
import threading

import ladders
import numpy as np
import pytest
from ladders import subspace

stumpy = pytest.importorskip("stumpy")
from multi_reference import (  # noqa: E402
    Tally,
    adversarial_windows,
    exact_bins,
    exact_edges,
    generator,
    mpmath_bins,
    reference,
)

pytestmark = pytest.mark.filterwarnings("ignore:'where' used without 'out'")

GRID = 1 / 64


def grid_walks(d: int, n: int, seed: int) -> np.ndarray:
    """Gaussian walks on a 1/64 grid: tie-free in practice, exactly the ticks at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.standard_normal((d, n)), axis=1) * 64.0) / 64.0 * 3.0 + 5.0


def pair_of(rng, n: int, m: int):
    i, j = (int(x) for x in rng.integers(0, n - m + 1, size=2))
    return i, j


class Stumpy:
    """STUMPY as a comparison: `check` records whether its answer differs from the exact one, and
    `assert_explained` that no difference is unexplained. With no binned sample near an edge
    (`tally.near == 0`) its edges bin every sample as the exact ones do, so the predicted number of
    differences is zero; the non-normalized bins are the same f64 formula in both, also zero."""

    def __init__(self):
        self.tally = Tally()
        self.cases = 0
        self.differs = 0

    def check(self, theirs, exact):
        self.cases += 1
        self.differs += list(theirs) != list(exact)

    def assert_explained(self):
        assert self.cases > 0
        assert (self.tally.near, self.differs) == (0, 0), (self.tally.near, self.differs, self.cases)


@pytest.mark.parametrize("n_bit", [1, 2, 3, 5, 7, 8])
@pytest.mark.parametrize("normalize", [True, False])
def test_equals_the_exact_reference_for_every_k_and_stumpy_beside_it(n_bit, normalize):
    theirs = Stumpy()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        d, n, m = int(rng.integers(1, 6)), int(rng.integers(30, 70)), int(rng.integers(3, 12))
        t = grid_walks(d, n, seed)
        i, j = pair_of(rng, n, m)
        scales = [GRID] * d if normalize else None
        full = reference(t, m, i, j, d - 1, bits=n_bit, normalize=normalize, scales=scales, tally=theirs.tally)
        for k in range(d):
            got = subspace(t, m, i, j, k, n_bit=n_bit, normalize=normalize, scales=scales)
            assert got.dtype == np.int64
            assert list(got) == full[: k + 1], (seed, k)
            theirs.check(stumpy.subspace(t, m, i, j, k, n_bit=n_bit, normalize=normalize), full[: k + 1])
    theirs.assert_explained()


def test_equals_stumpy_with_inferred_scales_on_gaussian_walks():
    """A STUMPY comparison only, not an exact check: these walks lie on no grid, so the library bins
    the ticks of the scale it infers, and the exact answer for those ticks needs that scale, which
    only the library computes. Every grid-data test above and below checks the exact answer."""
    checked = 0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        d, n, m = int(rng.integers(2, 6)), 60, int(rng.integers(4, 12))
        t = np.cumsum(rng.standard_normal((d, n)), axis=1) * 3.7 + 11.3
        i, j = pair_of(rng, n, m)
        k = int(rng.integers(0, d))
        assert list(subspace(t, m, i, j, k)) == list(stumpy.subspace(t, m, i, j, k))
        checked += 1
    assert checked == 40


@pytest.mark.parametrize("normalize", [True, False])
def test_include_equals_the_exact_reference_and_stumpy(normalize):
    theirs = Stumpy()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        d, n, m = 5, 50, int(rng.integers(3, 10))
        t = grid_walks(d, n, 100 + seed)
        i, j = pair_of(rng, n, m)
        include = [int(x) for x in rng.permutation(d)[: int(rng.integers(1, 4))]]
        scales = [GRID] * d if normalize else None
        full = reference(t, m, i, j, d - 1, include, normalize=normalize, scales=scales, tally=theirs.tally)
        for k in range(d):
            got = subspace(t, m, i, j, k, include=include, normalize=normalize, scales=scales)
            assert list(got) == full[: k + 1], (seed, k, include)
            theirs.check(stumpy.subspace(t, m, i, j, k, include=include, normalize=normalize), full[: k + 1])
    theirs.assert_explained()


def test_included_dimensions_are_ordered_by_distance_not_as_given():
    t = grid_walks(4, 40, 7)
    base = list(subspace(t, 6, 3, 20, 3))
    rev = [d for d in reversed(base)]
    got = list(subspace(t, 6, 3, 20, 3, include=rev))
    assert got == base == reference(t, 6, 3, 20, 3, rev)
    theirs = Stumpy()
    reference(t, 6, 3, 20, 3, tally=theirs.tally)
    theirs.check(stumpy.subspace(t, 6, 3, 20, 3, include=rev), base)
    theirs.assert_explained()


def test_a_repeated_include_entry_counts_once():
    t = grid_walks(4, 40, 8)
    assert list(subspace(t, 6, 3, 20, 3, include=[2, 2, 1])) == list(subspace(t, 6, 3, 20, 3, include=[2, 1]))
    assert list(subspace(t, 6, 3, 20, 3, include=np.array([2, 1]))) == list(subspace(t, 6, 3, 20, 3, include=[2, 1]))


@pytest.mark.parametrize("discords", [False, True])
@pytest.mark.parametrize("bits", [2, 8])
def test_equals_an_exact_rational_reference(discords, bits):
    for seed in range(15):
        rng = np.random.default_rng(seed)
        d, n, m = int(rng.integers(2, 5)), 30, int(rng.integers(3, 8))
        t = grid_walks(d, n, 300 + seed)
        i, j = pair_of(rng, n, m)
        include = [int(rng.integers(0, d))] if seed % 2 else []
        for k in range(d):
            got = subspace(t, m, i, j, k, include=include, discords=discords, n_bit=bits, scales=[GRID] * d)
            assert list(got) == reference(t, m, i, j, k, include, discords, bits, scales=[GRID] * d), (seed, k)


def test_discords_put_the_farthest_dimensions_first_unlike_stumpy():
    # The pair differs most in dimension 3, then 0, least in 1 and 2 (planted shapes).
    rng = np.random.default_rng(5)
    base = np.round(rng.standard_normal((4, 60)) * 8) / 8
    t = base.copy()
    t[:, 40:48] = t[:, 10:18]
    t[3, 40:48] = t[3, 10:18][::-1]
    t[0, 40:48] = t[0, 10:18] * np.array([1, 1, 1, 1, -1, -1, 1, 1])
    near = subspace(t, 8, 10, 40, 3, scales=[0.125] * 4)
    far = subspace(t, 8, 10, 40, 3, discords=True, scales=[0.125] * 4)
    keys = []
    for row in t:
        a, b = (exact_bins([int(round(x * 64)) for x in row[s : s + 8]], 8) for s in (10, 40))
        keys.append(sum((x - y) ** 2 for x, y in zip(a, b)))
    assert list(near) == sorted(range(4), key=lambda q: (keys[q], q))
    assert list(far) == sorted(range(4), key=lambda q: (-keys[q], q))
    assert far[0] == int(np.argmax(keys))
    # STUMPY 1.14.1 sorts the reversed array ascending and reports positions of the reversed
    # array, so its answer is a different set of dimensions.
    differs = 0
    for seed in range(20):
        t = grid_walks(5, 40, 600 + seed)
        k = seed % 4
        ours = subspace(t, 6, 2, 25, k, discords=True)
        assert list(ours) == reference(t, 6, 2, 25, k, discords=True)
        theirs = stumpy.subspace(t, 6, 2, 25, k, discords=True)
        differs += list(ours) != list(theirs)
    assert differs > 0


def test_discords_keep_the_include_dimensions_first():
    t = grid_walks(5, 40, 9)
    far = list(subspace(t, 6, 3, 25, 4, discords=True, include=[2]))
    assert far[0] == 2 and sorted(far) == [0, 1, 2, 3, 4]
    assert far[1:] == [q for q in subspace(t, 6, 3, 25, 4, discords=True) if q != 2]


def test_a_window_with_a_gap_makes_its_dimension_missing_and_last():
    t = grid_walks(4, 40, 11)
    t[1, 5] = np.nan
    t[3, 22] = np.inf
    for discords in (False, True):
        for k in range(4):
            got = list(subspace(t, 6, 3, 20, k, discords=discords, scales=[GRID] * 4))
            assert got == reference(t, 6, 3, 20, k, discords=discords, scales=[GRID] * 4), (discords, k)
        full = list(subspace(t, 6, 3, 20, 3, discords=discords, scales=[GRID] * 4))
        assert full[-2:] == [1, 3]
    # A gap elsewhere in the series changes nothing, as in STUMPY.
    t = grid_walks(3, 50, 12)
    clean = list(subspace(t, 6, 3, 30, 2, scales=[GRID] * 3))
    t[0, 15] = np.nan
    assert list(subspace(t, 6, 3, 30, 2, scales=[GRID] * 3)) == clean == reference(t, 6, 3, 30, 2, scales=[GRID] * 3)
    theirs = Stumpy()
    reference(t, 6, 3, 30, 2, tally=theirs.tally)
    theirs.check(stumpy.subspace(t, 6, 3, 30, 2), clean)
    theirs.assert_explained()


def test_a_gap_elsewhere_changes_nothing_non_normalized_either():
    t = grid_walks(3, 50, 13)
    want = reference(t, 6, 3, 30, 2, normalize=False)
    assert list(subspace(t, 6, 3, 30, 2, normalize=False)) == want
    t[1, 44] = np.nan
    gapped = reference(t, 6, 3, 30, 2, normalize=False)
    assert list(subspace(t, 6, 3, 30, 2, normalize=False)) == gapped
    assert sorted(want) == [0, 1, 2]
    theirs = Stumpy()
    theirs.check(stumpy.subspace(t, 6, 3, 30, 2, normalize=False), gapped)
    theirs.assert_explained()


def test_a_constant_window_is_all_in_the_middle_bin_as_in_stumpy():
    t = grid_walks(3, 50, 14)
    t[1, 10:20] = 4.0
    t[2, 30:40] = 7.5
    theirs = Stumpy()
    for i, j in [(10, 30), (11, 31), (12, 40), (10, 12)]:
        full = reference(t, 6, i, j, 2, tally=theirs.tally)
        for k in range(3):
            assert list(subspace(t, 6, i, j, k)) == full[: k + 1], (i, j, k)
            theirs.check(stumpy.subspace(t, 6, i, j, k), full[: k + 1])
    theirs.assert_explained()


def test_windows_below_the_deviation_threshold_are_not_scaled_as_in_stumpy():
    theirs = Stumpy()
    checked = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        d, n, m = 3, 40, 5
        tiny = 2.0**-30
        t = np.round(np.cumsum(rng.standard_normal((d, n)), axis=1) * 2) * tiny
        t[1] *= 2.0**20
        i, j = pair_of(rng, n, m)
        centered = [
            all(m * row[s + q] != row[s : s + m].sum() for q in range(m))
            for row in t
            for s in (i, j)
        ]
        if not all(centered):
            continue
        scales = [tiny, tiny * 2.0**20, tiny]
        full = reference(t, m, i, j, d - 1, scales=scales, tally=theirs.tally)
        for k in range(d):
            got = subspace(t, m, i, j, k, scales=scales)
            assert list(got) == full[: k + 1], (seed, k)
            theirs.check(stumpy.subspace(t, m, i, j, k), full[: k + 1])
        checked += 1
    assert checked >= 10
    theirs.assert_explained()


def test_sets_of_tied_dimensions_order_by_dimension():
    row = grid_walks(1, 40, 15)[0]
    t = np.stack([row, row, row, row])
    for discords in (False, True):
        assert list(subspace(t, 6, 3, 20, 3, discords=discords)) == [0, 1, 2, 3]
    assert list(subspace(t, 6, 3, 20, 3, include=[3, 1])) == [1, 3, 0, 2]


def test_one_dimension():
    t = grid_walks(1, 40, 16)
    assert list(subspace(t, 6, 3, 20, 0)) == [0]
    assert list(subspace(t, 6, 3, 20, 0, normalize=False)) == [0]


def test_a_window_may_be_paired_with_itself():
    t = grid_walks(3, 40, 17)
    assert list(subspace(t, 6, 4, 4, 2)) == [0, 1, 2]


def test_the_last_window_and_a_whole_series_window_are_valid():
    t = grid_walks(3, 40, 18)
    assert len(subspace(t, 6, 0, 34, 1)) == 2
    assert list(subspace(t, 40, 0, 0, 2)) == [0, 1, 2]


def test_unsupported_arguments_raise_not_implemented():
    t = grid_walks(3, 40, 19)
    with pytest.raises(NotImplementedError, match="discretize_func"):
        subspace(t, 6, 1, 2, 0, discretize_func=lambda a: a)
    with pytest.raises(NotImplementedError, match="p = 2"):
        subspace(t, 6, 1, 2, 0, normalize=False, p=1.0)
    assert list(subspace(t, 6, 1, 20, 1, p=1.0)) == list(subspace(t, 6, 1, 20, 1))
    assert list(subspace(t, 6, 1, 20, 1, discretize_func=None, T_subseq_isconstant=None)) == list(subspace(t, 6, 1, 20, 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 2},
        {"m": 41},
        {"m": -3},
        {"subseq_idx": 35},
        {"nn_idx": 35},
        {"subseq_idx": -1},
        {"nn_idx": -1},
        {"k": 3},
        {"k": -1},
        {"include": [3]},
        {"include": [-1]},
        {"n_bit": 0},
        {"n_bit": 9},
        {"n_bit": -1},
        {"n_bit": 300},
        {"scales": [1.0, 1.0]},
        {"scales": [1.0, 0.0, 1.0]},
        {"scales": [1.0, -1.0, 1.0]},
        {"scales": [1.0, float("nan"), 1.0]},
    ],
)
def test_bad_arguments_raise_value_error(kwargs):
    t = grid_walks(3, 40, 20)
    arguments = {"m": 6, "subseq_idx": 1, "nn_idx": 2, "k": 0}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        subspace(t, **arguments)


def test_scales_with_normalize_false_is_an_error():
    t = grid_walks(3, 40, 21)
    with pytest.raises(ValueError, match="scales"):
        subspace(t, 6, 1, 2, 0, normalize=False, scales=[1.0, 1.0, 1.0])


def test_no_dimensions_and_no_samples_are_errors():
    with pytest.raises(ValueError):
        subspace(np.empty((0, 40)), 6, 1, 2, 0)
    with pytest.raises(ValueError):
        subspace(np.empty((3, 0)), 6, 0, 0, 0)


def test_a_series_with_no_finite_sample_gives_dimension_order_for_non_normalized():
    t = np.full((3, 20), np.nan)
    assert list(subspace(t, 6, 0, 5, 2, normalize=False)) == [0, 1, 2]
    with pytest.raises(ValueError):
        subspace(t, 6, 0, 5, 2)


def test_integer_and_other_numeric_arguments_are_accepted():
    t = grid_walks(3, 40, 22)
    want = list(subspace(t, 6, 1, 20, 1))
    assert list(subspace(t, np.int64(6), np.int64(1), np.int64(20), np.int64(1))) == want


@pytest.mark.skipif(
    not hasattr(ladders, "_hold_arm"),
    reason="needs the test hooks: build the wheel with `maturin develop --features test-hooks`",
)
@pytest.mark.parametrize("normalize", [True, False])
def test_the_gil_is_released_while_a_call_runs(normalize):
    """The call parks inside the engine until this thread tells it to go on, which this thread
    can only do if the call released the GIL; a held GIL makes the call give up after a long
    hang guard and `_hold_release` report it."""
    t = grid_walks(3, 40, 23)
    result = []
    ladders._hold_arm()
    worker = threading.Thread(
        target=lambda: result.append(subspace(t, 6, 1, 20, 1, normalize=normalize)), daemon=True
    )
    worker.start()
    try:
        parked = ladders._hold_wait_parked()
    finally:
        released_in_time = ladders._hold_release()
    # timing-ok: a hang-guard bound only; correct code never waits on the clock
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert parked and released_in_time
    assert len(result) == 1


def committed_edges():
    text = (generator().CORE / "subspace" / "edges.rs").read_text()
    body = text[text.index("[f64; 255] = [") :]
    return np.array([float(x) for x in re.findall(r"^\s+(-?[0-9][0-9.e+-]*),$", body, re.M)])


def test_the_committed_edge_table_is_what_the_generator_computes():
    module = generator()
    assert (module.CORE / "subspace" / "edges.rs").read_text() == module.edges_source()


def test_the_committed_edge_table_is_scipys_norm_ppf_within_a_tolerance():
    """STUMPY's edges are scipy's `norm.ppf(i / 256)`, which varies by platform and is up to 2 ulps
    from the correctly rounded quantile the table holds; 1e-12 relative is far above that."""
    from scipy.stats import norm

    table = committed_edges()
    assert table.shape == (255,)
    want = norm.ppf(np.arange(1, 256) / 256)
    assert table[127] == want[127] == 0.0
    assert np.allclose(table, want, rtol=1e-12, atol=0)


def probe(above, below):
    """Two dimensions, each a pair of windows: the window just above an edge against `[-1, 0, ...]`
    (whose first sample is in bin 0) in dimension 0, the window just below it in dimension 1."""
    m = len(above)
    anchor = [-1.0] + [0.0] * (m - 1)
    return np.array([[*map(float, above), *anchor], [*map(float, below), *anchor]]), m


def moved(q, direction):
    """The exact eight-bit edges with edge `q` moved one ulp in `direction`."""
    edges = [float(e) for e in exact_edges(8)]
    edges[q] = float(np.nextafter(edges[q], direction * np.inf))
    return edges


def test_samples_within_one_ulp_of_every_edge_are_binned_as_the_exact_quantile_says():
    """For each nonzero edge `e`, a window whose first sample's exact z-value is strictly between `e`
    and the next f64 above, and one strictly between the next f64 below and `e`; these are the
    samples where scipy's edges and the correctly rounded ones disagree. mpmath at 256 bits, by a
    route that shares no code with the edge generator, puts the first in bin `q + 1` and the second
    in bin `q`. Paired against a window in bin 0, the first dimension is then exactly farther than
    the second, so the subspace is `[1, 0]`; an edge one ulp off either way puts both samples in one
    bin and the tie goes to `[0, 1]`, which the reference with that edge moved confirms. No STUMPY
    comparison: its z-values carry rounding errors of several ulps, larger than these gaps."""
    windows = adversarial_windows()
    assert len(windows) == 254
    for q, (above, below) in windows.items():
        assert exact_bins(above, 8)[1:] == exact_bins(below, 8)[1:], q
        assert mpmath_bins(above, 8)[0][0] == q + 1, q
        assert mpmath_bins(below, 8)[0][0] == q, q
        t, m = probe(above, below)
        for bits in range(1, 9):
            for k in (0, 1):
                want = reference(t, m, 0, m, k, bits=bits, scales=[1.0, 1.0])
                got = subspace(t, m, 0, m, k, n_bit=bits, scales=[1.0, 1.0])
                assert list(got) == want, (q, bits, k)
        assert reference(t, m, 0, m, 1, scales=[1.0, 1.0]) == [1, 0], q
        for direction in (1, -1):
            assert reference(t, m, 0, m, 1, scales=[1.0, 1.0], edges=moved(q, direction)) == [0, 1], (q, direction)


def test_a_sample_at_the_mean_is_not_above_the_middle_edge_and_one_tick_more_is():
    """The middle edge is zero exactly: a sample at its window's mean (`a = 0`) is not above it,
    and the same sample one tick higher is. Moving the edge one ulp below zero puts the first above
    too, which changes the answer; one ulp above zero no window can resolve, since a nonzero
    z-value of integer ticks here is at least 2^-32."""
    at_mean = [0, 1000, -1000, 1, -1, 2, -2, 3, -3]
    above = [1, 1000, -1000, 1, -1, 2, -2, 3, -3]
    assert mpmath_bins(at_mean, 8)[0][0] == exact_bins(at_mean, 8)[0] == 127
    assert mpmath_bins(above, 8)[0][0] == exact_bins(above, 8)[0] == 128
    assert exact_bins(at_mean, 8)[1:] == exact_bins(above, 8)[1:]
    t, m = probe(above, at_mean)
    for bits in range(1, 9):
        for k in (0, 1):
            want = reference(t, m, 0, m, k, bits=bits, scales=[1.0, 1.0])
            assert list(subspace(t, m, 0, m, k, n_bit=bits, scales=[1.0, 1.0])) == want, (bits, k)
    assert reference(t, m, 0, m, 1, scales=[1.0, 1.0]) == [1, 0]
    assert reference(t, m, 0, m, 1, scales=[1.0, 1.0], edges=moved(127, -1)) == [0, 1]


def test_infinities_outside_the_windows_change_nothing_non_normalized():
    t = grid_walks(3, 50, 30)
    base = list(subspace(t, 6, 3, 30, 2, normalize=False))
    for bad in (np.inf, -np.inf, np.nan):
        u = t.copy()
        u[1, 45] = bad
        u[2, 8] = -bad if bad != np.nan else bad
        want = reference(u, 6, 3, 30, 2, normalize=False)
        assert list(subspace(u, 6, 3, 30, 2, normalize=False)) == want
        theirs = Stumpy()
        theirs.check(stumpy.subspace(u, 6, 3, 30, 2, normalize=False), want)
        theirs.assert_explained()
    u = t.copy()
    u[1, 45] = np.inf
    u[2, 45] = -np.inf
    # The linear range ignores the infinities; the finite samples decide as without them.
    finite_only = np.where(np.isfinite(u), u, t)
    assert list(subspace(u, 6, 3, 30, 2, normalize=False)) == list(subspace(finite_only, 6, 3, 30, 2, normalize=False)) == base


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_a_non_normalized_window_with_an_infinity_is_missing_and_last(bad):
    t = grid_walks(3, 40, 31)
    t[0, 5] = bad
    for discords in (False, True):
        got = list(subspace(t, 6, 3, 20, 2, normalize=False, discords=discords))
        assert got[-1] == 0
        assert list(subspace(t, 6, 3, 20, 1, normalize=False, discords=discords)) == got[:2]
    with pytest.raises(ValueError):
        subspace(t, 6, 3, 20, 2, normalize=False, strict=True)
    from ladders import mdl

    sizes, subspaces = mdl(t, 6, [3, 3, 3], [20, 20, 20], normalize=False)
    assert np.isfinite(sizes[:2]).all() and sizes[2] == np.inf and subspaces[2][-1] == 0


def test_the_deviation_rule_divides_the_offset_by_the_window_length():
    # A spike of a few ticks in a window of 5000 zeros, at a tick of 1e-7 (a deviation below the
    # threshold): the spike is one bin above the rest, not several. Two spikes in dimension 1
    # make its distance larger than dimension 0's single one, in STUMPY and here.
    m = 5000
    tick = 1e-7
    t = np.zeros((2, 2 * m))
    t[0, 100] = 60 * tick
    t[1, 100] = t[1, 200] = 40 * tick
    theirs = Stumpy()
    assert reference(t, m, 0, m, 1, scales=[tick, tick], tally=theirs.tally) == [0, 1]
    assert list(subspace(t, m, 0, m, 1, scales=[tick, tick])) == [0, 1]
    assert list(subspace(t, m, 0, m, 0, scales=[tick, tick])) == [0]
    theirs.check(stumpy.subspace(t, m, 0, m, 1), [0, 1])
    theirs.assert_explained()


def test_stumpys_names_and_keywords():
    from ladders import maamp_subspace

    t = grid_walks(4, 40, 32)
    theirs = Stumpy()
    for k in range(4):
        want = reference(t, 6, 3, 20, k, [2], bits=5, normalize=False)
        got = maamp_subspace(t, 6, 3, 20, k, include=[2], n_bit=5)
        assert list(got) == want == list(subspace(t, 6, 3, 20, k, include=[2], n_bit=5, normalize=False))
        theirs.check(stumpy.maamp_subspace(t, 6, 3, 20, k, include=[2], n_bit=5), want)
    theirs.assert_explained()
    assert list(subspace(T=t, m=6, subseq_idx=3, nn_idx=20, k=2)) == list(subspace(t, 6, 3, 20, 2))
    assert list(maamp_subspace(T=t, m=6, subseq_idx=3, nn_idx=20, k=2, discords=True)) == list(
        subspace(t, 6, 3, 20, 2, normalize=False, discords=True)
    )
    with pytest.raises(NotImplementedError):
        maamp_subspace(t, 6, 3, 20, 2, p=1.0)
    with pytest.raises(NotImplementedError):
        maamp_subspace(t, 6, 3, 20, 2, discretize_func=lambda a: a)
    with pytest.raises(ValueError):
        maamp_subspace(t, 6, 3, 99, 2)


def test_lists_and_other_numeric_arrays_are_converted_to_float64():
    # STUMPY takes lists of floats and float64 arrays only (integer and float32 arrays are a
    # TypeError there); here they are converted, and equal STUMPY on the float64 values.
    ints = np.round(grid_walks(3, 40, 33) * 4).astype(np.int64)
    theirs = Stumpy()
    for normalize in (True, False):
        want = reference(ints, 6, 3, 20, 2, normalize=normalize, tally=theirs.tally)
        theirs.check(stumpy.subspace(ints.astype(np.float64), 6, 3, 20, 2, normalize=normalize), want)
        theirs.check(stumpy.subspace(ints.astype(np.float64).tolist(), 6, 3, 20, 2, normalize=normalize), want)
        for source in (ints, ints.tolist(), ints.astype(np.int32), ints.astype(np.float32), ints.astype(np.float64).tolist()):
            assert list(subspace(source, 6, 3, 20, 2, normalize=normalize)) == want
    theirs.assert_explained()
    with pytest.raises((TypeError, ValueError)):
        subspace(ints[0], 6, 3, 20, 0)


def test_strict_refuses_a_subspace_without_enough_distances():
    t = grid_walks(3, 40, 34)
    t[1, 5] = np.nan
    assert list(subspace(t, 6, 3, 20, 1, strict=True, scales=[GRID] * 3)) == list(subspace(t, 6, 3, 20, 1, scales=[GRID] * 3))
    with pytest.raises(ValueError, match="fewer than k"):
        subspace(t, 6, 3, 20, 2, strict=True, scales=[GRID] * 3)
    assert len(subspace(t, 6, 3, 20, 2, strict=False, scales=[GRID] * 3)) == 3
    # An out-of-range window (variance numerator of 2**63 or more) counts as missing too.
    wild = np.zeros((2, 20))
    wild[0, 3] = 4e9
    wild[1] = np.arange(20) % 3
    with pytest.raises(ValueError, match="fewer than k"):
        subspace(wild, 4, 2, 10, 1, strict=True, scales=[1.0, 1.0])
    assert list(subspace(wild, 4, 2, 10, 1, scales=[1.0, 1.0])) == [1, 0]
