"""`mdl(t, m, subseq_idx, nn_idx, ...)`: STUMPY's description length of a motif pair at every
subspace size, the bins decided exactly, the logarithms correctly rounded, a gap making the size
`inf`, and bad arguments a `ValueError` or `NotImplementedError`.

The basis is the exact reference (multi_reference.mdl_reference): exact bins and subspaces, and the
size formula in STUMPY's order in f64 with the correctly rounded `log2`, so every size is checked
bit for bit. STUMPY is a comparison beside it: its edges can bin otherwise only a sample within
`NEAR_EDGE` of an edge (counted, and none here), and its `log2` is NumPy's, which may differ from
the correctly rounded value on some platform; its sizes must equal the same formula with NumPy's
`log2`, so every difference from the exact size is one of those, counted."""

import re
import struct
import threading
from pathlib import Path

import ladders
import numpy as np
import pytest
from ladders import mdl
from multi_reference import Tally, generator, mdl_reference

stumpy = pytest.importorskip("stumpy")

pytestmark = pytest.mark.filterwarnings("ignore:'where' used without 'out'")

GRID = 1 / 64
REPO = Path(__file__).resolve().parents[3]


def grid_walks(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.standard_normal((d, n)), axis=1) * 64.0) / 64.0 * 3.0 + 5.0


def pairs(rng, d: int, n: int, m: int):
    return (
        rng.integers(0, n - m + 1, size=d),
        rng.integers(0, n - m + 1, size=d),
    )


def same_bits(a, b):
    return np.array_equal(np.asarray(a, dtype=np.float64).view(np.uint64), np.asarray(b, dtype=np.float64).view(np.uint64))


def numpy_log2(n):
    return float(np.log2(np.float64(n)))


class Stumpy:
    """STUMPY's `mdl` as a comparison. `check` asserts it equals the exact computation with
    NumPy's `log2` in place of the correctly rounded one (STUMPY's own formula), and counts the
    sizes where that differs from the exact size (`predicted`) and where STUMPY's differs from the
    exact size (`differs`); `assert_explained` that no sample was near an edge and the counts agree."""

    def __init__(self):
        self.tally = Tally()
        self.predicted = 0
        self.differs = 0
        self.cases = 0

    def check(self, theirs, exact, model):
        self.cases += 1
        assert same_bits(theirs[0], model[0]), (theirs[0], model[0])
        assert [list(x) for x in theirs[1]] == [list(x) for x in model[1]]
        self.predicted += int(np.count_nonzero(np.asarray(model[0]).view(np.uint64) != np.asarray(exact[0]).view(np.uint64)))
        self.differs += int(np.count_nonzero(np.asarray(theirs[0]).view(np.uint64) != np.asarray(exact[0]).view(np.uint64)))

    def assert_explained(self):
        assert self.cases > 0
        assert self.tally.near == 0, self.tally.near
        assert self.differs == self.predicted, (self.differs, self.predicted)


def check_exact(got, t, m, first, second, theirs=None, stumpy_result=None, **options):
    """`got` equals the exact reference bit for bit; STUMPY's result, if given, is compared."""
    exact = mdl_reference(t, m, first, second, tally=None if theirs is None else theirs.tally, **options)
    assert same_bits(got[0], exact[0]), (got[0], exact[0])
    assert [list(s) for s in got[1]] == exact[1]
    if theirs is not None:
        theirs.check(stumpy_result, exact, mdl_reference(t, m, first, second, log=numpy_log2, **options))


DOCTEST = np.array(
    [
        [584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0],
        [1.0, 2.0, 4.0, 8.0, 16.0, 0.0, 32.0],
    ]
)


def test_stumpys_documented_example():
    sizes, subspaces = mdl(DOCTEST, 3, [0, 0], [2, 4])
    assert same_bits(sizes, [80.0, float.fromhex("0x1.be0a02756f9c2p+6")])
    assert [list(s) for s in subspaces] == [[1], [0, 1]]
    sizes, subspaces = mdl(DOCTEST, 3, np.array([0, 0]), np.array([2, 4]), normalize=False)
    assert sizes.view(np.uint64).tolist() == [0x4059305013AB7CE1, 0x405BE0A02756F9C2]
    assert [list(s) for s in subspaces] == [[1], [1, 0]]
    theirs = Stumpy()
    for normalize in (True, False):
        got = mdl(DOCTEST, 3, [0, 0], [2, 4], normalize=normalize)
        check_exact(got, DOCTEST, 3, [0, 0], [2, 4], theirs, stumpy.mdl(DOCTEST, 3, [0, 0], [2, 4], normalize=normalize), normalize=normalize)
    theirs.assert_explained()


def test_result_types():
    sizes, subspaces = mdl(DOCTEST, 3, [0, 0], [2, 4])
    assert isinstance(sizes, np.ndarray) and sizes.dtype == np.float64 and sizes.shape == (2,)
    assert isinstance(subspaces, list) and len(subspaces) == 2
    assert all(isinstance(s, np.ndarray) and s.dtype == np.int64 for s in subspaces)
    assert [len(s) for s in subspaces] == [1, 2]


@pytest.mark.parametrize("n_bit", [1, 2, 4, 6, 8])
@pytest.mark.parametrize("normalize", [True, False])
def test_equals_the_exact_reference_on_arbitrary_pairs_and_stumpy_beside_it(n_bit, normalize):
    theirs = Stumpy()
    for seed in range(10):
        rng = np.random.default_rng(seed)
        d, n, m = int(rng.integers(1, 6)), int(rng.integers(30, 70)), int(rng.integers(3, 12))
        t = grid_walks(d, n, seed)
        first, second = pairs(rng, d, n, m)
        scales = [GRID] * d if normalize else None
        got = mdl(t, m, first, second, n_bit=n_bit, normalize=normalize, scales=scales)
        want = stumpy.mdl(t, m, first, second, n_bit=n_bit, normalize=normalize)
        check_exact(got, t, m, first, second, theirs, want, bits=n_bit, normalize=normalize, scales=scales)
    theirs.assert_explained()


def test_equals_the_exact_reference_on_motif_pairs_of_mstump_and_stumpy_beside_it():
    theirs = Stumpy()
    for seed in range(8):
        t = grid_walks(3, 80, 40 + seed)
        m = 8
        profile, index = stumpy.mstump(t, m)
        motif = np.argsort(profile, axis=1)[:, 0]
        neighbor = index[np.arange(3), motif]
        for normalize in (True, False):
            scales = [GRID] * 3 if normalize else None
            want = stumpy.mdl(t, m, motif, neighbor, normalize=normalize)
            got = mdl(t, m, motif, neighbor, normalize=normalize, scales=scales)
            check_exact(got, t, m, motif, neighbor, theirs, want, normalize=normalize, scales=scales)
    theirs.assert_explained()


def test_equals_stumpy_with_inferred_scales_on_gaussian_walks():
    """A STUMPY comparison only, not an exact check: these walks lie on no grid, so the library bins
    the ticks of the scale it infers, and the exact answer for those ticks needs that scale, which
    only the library computes. The grid-data tests check the exact answer."""
    for seed in range(25):
        rng = np.random.default_rng(2000 + seed)
        d, n, m = int(rng.integers(2, 6)), 60, int(rng.integers(4, 12))
        t = np.cumsum(rng.standard_normal((d, n)), axis=1) * 3.7 + 11.3
        first, second = pairs(rng, d, n, m)
        want = stumpy.mdl(t, m, first, second)
        got = mdl(t, m, first, second)
        assert same_bits(got[0], want[0])
        assert [list(s) for s in got[1]] == [list(s) for s in want[1]]


@pytest.mark.parametrize("normalize", [True, False])
def test_include_equals_the_exact_reference_and_stumpy(normalize):
    theirs = Stumpy()
    for seed in range(12):
        rng = np.random.default_rng(seed)
        d, n, m = 5, 50, int(rng.integers(3, 10))
        t = grid_walks(d, n, 100 + seed)
        first, second = pairs(rng, d, n, m)
        include = [int(x) for x in rng.permutation(d)[: int(rng.integers(1, 4))]]
        scales = [GRID] * d if normalize else None
        want = stumpy.mdl(t, m, first, second, include=include, normalize=normalize)
        got = mdl(t, m, first, second, include=include, normalize=normalize, scales=scales)
        check_exact(got, t, m, first, second, theirs, want, include=include, normalize=normalize, scales=scales)
    theirs.assert_explained()


def reference_mdl(t, m, first, second, include=(), discords=False, bits=8):
    """The exact reference on grid data quantized at `GRID`."""
    return mdl_reference(t, m, first, second, include, discords, bits, scales=[GRID] * len(t))


@pytest.mark.parametrize("discords", [False, True])
@pytest.mark.parametrize("bits", [2, 8])
def test_equals_an_exact_rational_reference(discords, bits):
    for seed in range(12):
        rng = np.random.default_rng(seed)
        d, n, m = int(rng.integers(2, 5)), 30, int(rng.integers(3, 8))
        t = grid_walks(d, n, 300 + seed)
        first, second = pairs(rng, d, n, m)
        include = [int(rng.integers(0, d))] if seed % 2 else []
        sizes, subspaces = mdl(t, m, first, second, include=include, discords=discords, n_bit=bits, scales=[GRID] * d)
        want_sizes, want_subspaces = reference_mdl(t, m, first, second, include, discords, bits)
        assert same_bits(sizes, want_sizes), (seed, sizes, want_sizes)
        assert [list(s) for s in subspaces] == want_subspaces


def test_discords_differ_from_stumpy_in_the_subspaces_only_where_stumpy_is_wrong():
    differs = 0
    for seed in range(15):
        rng = np.random.default_rng(seed)
        t = grid_walks(5, 40, 700 + seed)
        first, second = pairs(rng, 5, 40, 6)
        ours = mdl(t, 6, first, second, discords=True)
        theirs = stumpy.mdl(t, 6, first, second, discords=True)
        differs += [list(s) for s in ours[1]] != [list(s) for s in theirs[1]]
    assert differs > 0


def test_a_window_with_a_gap_gives_an_infinite_size_where_its_dimension_is_chosen():
    t = grid_walks(3, 40, 24)
    t[2, 6] = np.nan
    first, second = np.array([3, 3, 3]), np.array([20, 20, 20])
    sizes, subspaces = mdl(t, 6, first, second, scales=[GRID] * 3)
    assert [list(s)[-1] for s in subspaces][2] == 2
    assert np.isfinite(sizes[:2]).all() and sizes[2] == np.inf
    want_sizes, want_subspaces = reference_mdl(t, 6, first, second)
    assert same_bits(sizes, want_sizes)
    assert [list(s) for s in subspaces] == want_subspaces
    # Another k's pair away from the gap leaves it finite.
    sizes, _ = mdl(t, 6, [3, 3, 20], [20, 20, 30], scales=[GRID] * 3)
    assert np.isfinite(sizes).all()
    assert int(np.argmin(sizes)) in (0, 1, 2)
    # An included gap dimension is chosen first, so every size holds it.
    sizes, subspaces = mdl(t, 6, first, second, include=[2], scales=[GRID] * 3)
    assert (sizes == np.inf).all() and all(s[0] == 2 for s in subspaces)
    # A gap in a dimension outside every subspace does not matter, as in STUMPY.
    t = grid_walks(3, 40, 25)
    clean = mdl(t, 6, first, second, include=[0], scales=[GRID] * 3)
    t[2, 30] = np.nan
    again = mdl(t, 6, first, second, include=[0], scales=[GRID] * 3)
    assert same_bits(clean[0], again[0])


def test_the_same_window_twice_has_one_distinct_difference():
    t = grid_walks(3, 40, 26)
    m, d = 6, 3
    sizes, subspaces = mdl(t, m, [4, 4, 4], [4, 4, 4])
    for k, size in enumerate(sizes):
        s = k + 1
        assert size == 8 * (2 * d * m - s * m) + 0.0 + 1 * 8
    assert [list(s) for s in subspaces] == [[0], [0, 1], [0, 1, 2]]


def test_one_dimension():
    t = grid_walks(1, 40, 27)
    sizes, subspaces = mdl(t, 6, [2], [20])
    assert [list(s) for s in subspaces] == [[0]]
    theirs = Stumpy()
    check_exact((sizes, subspaces), t, 6, [2], [20], theirs, stumpy.mdl(t, 6, [2], [20]))
    theirs.assert_explained()


def test_the_smallest_size_picks_the_planted_motif_pair():
    rng = np.random.default_rng(3)
    t = np.round(rng.standard_normal((4, 100)) * 8) / 8 * 3
    t[:3, 70:78] = t[:3, 10:18]
    first, second = [10] * 4, [70] * 4
    sizes, subspaces = mdl(t, 8, first, second, scales=[0.125] * 4)
    assert sorted(subspaces[2]) == [0, 1, 2]
    assert int(np.argmin(sizes)) == 2


def test_unsupported_arguments_raise_not_implemented():
    with pytest.raises(NotImplementedError, match="discretize_func"):
        mdl(DOCTEST, 3, [0, 0], [2, 4], discretize_func=lambda a: a)
    with pytest.raises(NotImplementedError, match="p = 2"):
        mdl(DOCTEST, 3, [0, 0], [2, 4], normalize=False, p=3.0)
    assert same_bits(mdl(DOCTEST, 3, [0, 0], [2, 4], p=3.0)[0], mdl(DOCTEST, 3, [0, 0], [2, 4])[0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"m": 2},
        {"m": 8},
        {"subseq_idx": [0]},
        {"subseq_idx": [0, 0, 0]},
        {"nn_idx": [2]},
        {"nn_idx": []},
        {"subseq_idx": [0, 5]},
        {"nn_idx": [5, 0]},
        {"nn_idx": [-1, 0]},
        {"subseq_idx": [0, -3]},
        {"include": [2]},
        {"include": [-1]},
        {"n_bit": 0},
        {"n_bit": 9},
        {"scales": [1.0]},
        {"scales": [1.0, 0.0]},
    ],
)
def test_bad_arguments_raise_value_error(kwargs):
    arguments = {"m": 3, "subseq_idx": [0, 0], "nn_idx": [2, 4]}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        mdl(DOCTEST, **arguments)


def test_scales_with_normalize_false_is_an_error():
    with pytest.raises(ValueError, match="scales"):
        mdl(DOCTEST, 3, [0, 0], [2, 4], normalize=False, scales=[1.0, 1.0])


def test_no_dimensions_is_an_error():
    with pytest.raises(ValueError):
        mdl(np.empty((0, 20)), 3, [], [])


@pytest.mark.skipif(
    not hasattr(ladders, "_hold_arm"),
    reason="needs the test hooks: build the wheel with `maturin develop --features test-hooks`",
)
@pytest.mark.parametrize("normalize", [True, False])
def test_the_gil_is_released_while_a_call_runs(normalize):
    """The call parks inside the engine until this thread tells it to go on, which this thread
    can only do if the call released the GIL; a held GIL makes the call give up after a long
    hang guard and `_hold_release` report it."""
    result = []
    ladders._hold_arm()
    worker = threading.Thread(
        target=lambda: result.append(mdl(DOCTEST, 3, [0, 0], [2, 4], normalize=normalize)), daemon=True
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


def committed_log2():
    text = (REPO / "crates" / "core" / "src" / "mdl" / "log2.rs").read_text()
    body = text[text.index("[f64; 512] = [") :]
    return np.array([float(x) for x in re.findall(r"^\s+(-?[0-9][0-9.e+-]*),$", body, re.M)])


def test_the_committed_logarithm_table_is_numpys_within_a_tolerance():
    """NumPy's `log2` comes from the platform's math library, so it is a comparison, not the basis:
    the committed table is correctly rounded (the generator test checks it exactly)."""
    table = committed_log2()
    assert table.shape == (512,)
    want = np.log2(np.arange(1, 513, dtype=np.float64))
    assert np.allclose(table, want, rtol=1e-12, atol=0)


def test_the_committed_tables_are_what_the_generator_computes():
    module = generator()
    assert (module.CORE / "mdl" / "log2.rs").read_text() == module.log2_source()
    assert (module.CORE / "subspace" / "edges.rs").read_text() == module.edges_source()


def test_stumpys_names_keywords_and_integer_input():
    from ladders import maamp_mdl

    t = grid_walks(3, 40, 35)
    first, second = [3, 5, 7], [20, 22, 24]
    theirs = Stumpy()
    got = maamp_mdl(t, 6, first, second, include=[1], n_bit=6)
    want = stumpy.maamp_mdl(t, 6, first, second, include=[1], n_bit=6)
    check_exact(got, t, 6, first, second, theirs, want, include=[1], bits=6, normalize=False)
    assert same_bits(got[0], mdl(t, 6, first, second, include=[1], n_bit=6, normalize=False)[0])
    assert same_bits(
        mdl(T=t, m=6, subseq_idx=first, nn_idx=second, scales=[GRID] * 3)[0],
        mdl(t, 6, first, second, scales=[GRID] * 3)[0],
    )
    assert same_bits(maamp_mdl(T=t, m=6, subseq_idx=first, nn_idx=second)[0], maamp_mdl(t, 6, first, second)[0])
    with pytest.raises(NotImplementedError):
        maamp_mdl(t, 6, first, second, p=3.0)
    ints = np.round(t * 4).astype(np.int64)
    for normalize in (True, False):
        want = stumpy.mdl(ints.astype(np.float64), 6, first, second, normalize=normalize)
        for source in (ints, ints.tolist(), ints.astype(np.int16), ints.astype(np.float64).tolist()):
            got = mdl(source, 6, first, second, normalize=normalize)
            check_exact(got, ints, 6, first, second, theirs, want, normalize=normalize)
    theirs.assert_explained()
