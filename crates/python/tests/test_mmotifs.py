"""`mmotifs(T, P, I, ...)`: STUMPY's multi-dimensional motifs from a multi-dimensional profile, on
tie-free data equal to STUMPY's (indices exactly, distances to rounding), padded as STUMPY pads
them, with real integer and float32 arrays, gaps, the edges of the arguments, and bad arguments a
`ValueError` or `NotImplementedError`."""

import threading
import warnings

import ladders
import numpy as np
import pytest
from ladders import mmotifs, mstump

stumpy = pytest.importorskip("stumpy")

pytestmark = pytest.mark.filterwarnings("ignore:'where' used without 'out'")

GRID = 1 / 64


def grid_walks(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.standard_normal((d, n)), axis=1) * 64.0) / 64.0 * 3.0 + 5.0


def stumpy_inputs(d: int, n: int, m: int, seed: int):
    t = grid_walks(d, n, seed)
    p, i = stumpy.mstump(t, m)
    return t, p, i


def same_bits(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def assert_equals_stumpy(got, want):
    distances, indices, subspaces, sizes = got
    want_distances, want_indices, want_subspaces, want_sizes = want
    assert np.asarray(indices).shape == np.asarray(want_indices).shape
    assert np.array_equal(indices, np.asarray(want_indices, dtype=np.int64))
    assert np.allclose(distances, want_distances, rtol=1e-9, atol=1e-9, equal_nan=True)
    assert [list(s) for s in subspaces] == [list(s) for s in want_subspaces]
    assert len(sizes) == len(want_sizes)
    for ours, theirs in zip(sizes, want_sizes):
        assert np.allclose(ours, theirs, rtol=1e-12, atol=0)


PARAMS = [
    {},
    {"max_motifs": 3},
    {"max_motifs": 6, "max_matches": 4, "min_neighbors": 0},
    {"max_motifs": 3, "k": 1},
    {"max_motifs": 3, "k": 0, "min_neighbors": 2},
    {"max_motifs": 3, "include": [1]},
    {"max_motifs": 4, "max_matches": None, "cutoffs": 1e9, "max_distance": 1e9, "min_neighbors": 3},
    {"max_motifs": 3, "cutoffs": [1e9, 2.5, 1e9, 1e9]},
    {"max_motifs": 3, "atol": 0.5},
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("seed", range(4))
def test_equals_stumpy(seed, params):
    d, n, m = 4, 90 + 10 * seed, 6 + seed
    t, p, i = stumpy_inputs(d, n, m, seed)
    got = mmotifs(t, p, i, scales=GRID, **params)
    want = stumpy.mmotifs(t, p, i, **params)
    assert_equals_stumpy(got, want)


def test_finds_several_motifs_and_stumpys_example():
    t = np.array(
        [[584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0], [1.0, 2.0, 4.0, 8.0, 16.0, 0.0, 32.0]]
    )
    p, i = stumpy.mstump(t, 3)
    got = mmotifs(t, p, i, scales=1.0)
    want = stumpy.mmotifs(t, p, i)
    assert [list(s) for s in got[2]] == [list(s) for s in want[2]] == [[1]]
    assert got[1].tolist() == want[1].tolist() == [[0, 2]]
    assert np.allclose(got[0], want[0], atol=1e-6)
    assert np.allclose(got[3][0], want[3][0])

    t, p, i = stumpy_inputs(3, 160, 8, 11)
    options = {"max_motifs": 100, "min_neighbors": 0, "cutoffs": 1e9}
    many = mmotifs(t, p, i, scales=GRID, **options)
    assert many[1].shape[0] >= 2
    assert_equals_stumpy(many, stumpy.mmotifs(t, p, i, **options))


def test_result_types_and_padding():
    t, p, i = stumpy_inputs(3, 120, 8, 5)
    distances, indices, subspaces, sizes = mmotifs(
        t, p, i, max_motifs=3, max_matches=5, min_neighbors=0, scales=GRID
    )
    assert distances.dtype == np.float64 and indices.dtype == np.int64
    assert distances.shape == indices.shape and distances.shape[0] == len(subspaces) == len(sizes) >= 2
    assert isinstance(subspaces, list) and isinstance(sizes, list)
    assert all(isinstance(s, np.ndarray) and s.dtype == np.int64 for s in subspaces)
    assert all(isinstance(s, np.ndarray) and s.dtype == np.float64 and s.shape == (3,) for s in sizes)
    assert (distances[:, 0] == 0).all()
    assert np.array_equal(np.isnan(distances), indices == -1)
    for row in indices:
        starts = row[row >= 0]
        assert len(set(starts)) == len(starts)


def test_no_motif_is_an_empty_result():
    t, p, i = stumpy_inputs(3, 100, 8, 2)
    distances, indices, subspaces, sizes = mmotifs(t, p, i, cutoffs=-1.0, scales=GRID)
    assert distances.shape == (1, 0) and indices.shape == (1, 0)
    assert subspaces == [] and sizes == []
    got = mmotifs(t, np.full_like(p, np.inf), i, scales=GRID)
    assert got[0].shape == (1, 0) and got[2] == []
    want = stumpy.mmotifs(t, p, i, cutoffs=-1.0)
    assert want[0].shape == (1, 0) and want[2] == []
    assert mmotifs(t, p, i, min_neighbors=1000, scales=GRID)[0].shape == (1, 0)


def test_a_motif_candidate_without_matches_leaves_the_pool():
    t, p, i = stumpy_inputs(3, 100, 8, 3)
    for kwargs in ({"max_matches": 0}, {"max_distance": -1.0}):
        got = mmotifs(t, p, i, max_motifs=5, min_neighbors=0, scales=GRID, **kwargs)
        assert got[0].shape == (1, 0) and got[2] == []


def test_huge_limits_end():
    t, p, i = stumpy_inputs(3, 100, 8, 4)
    got = mmotifs(t, p, i, max_motifs=10**12, min_neighbors=0, max_matches=10**12, scales=GRID)
    assert got[1].shape[0] >= 1
    with pytest.raises(ValueError):
        mmotifs(t, p, i, k=10**12, scales=GRID)


def test_negative_k_counts_from_the_last_dimension():
    t, p, i = stumpy_inputs(3, 100, 8, 6)
    assert_equals_stumpy(
        mmotifs(t, p, i, k=-1, max_motifs=2, scales=GRID), stumpy.mmotifs(t, p, i, k=-1, max_motifs=2)
    )
    with pytest.raises(ValueError):
        mmotifs(t, p, i, k=-4, scales=GRID)


def test_max_motifs_below_one_is_an_error_naming_stumpys_one():
    t, p, i = stumpy_inputs(3, 100, 8, 7)
    with pytest.raises(ValueError, match="max_motifs=0 is below 1") as error:
        mmotifs(t, p, i, max_motifs=0, scales=GRID)
    assert "pass max_motifs=1 to get that" in str(error.value)
    with pytest.warns(UserWarning, match="max_motifs"):
        want = stumpy.mmotifs(t, p, i, max_motifs=0)
    assert_equals_stumpy(mmotifs(t, p, i, max_motifs=1, scales=GRID), want)


@pytest.mark.parametrize("cast", ["int64", "int32", "float32", "list", "int_list"])
def test_real_integer_and_float32_arrays(cast):
    rng = np.random.default_rng(9)
    ints = np.cumsum(rng.integers(-3, 4, size=(3, 100)), axis=1)
    t64 = ints.astype(np.float64)
    p, i = stumpy.mstump(t64, 7)
    want = mmotifs(t64, p, i, max_motifs=3, scales=1.0)
    source = {
        "int64": ints.astype(np.int64),
        "int32": ints.astype(np.int32),
        "float32": ints.astype(np.float32),
        "list": t64.tolist(),
        "int_list": ints.tolist(),
    }[cast]
    for index in (i, i.astype(np.int32), i.tolist()):
        got = mmotifs(source, p, index, max_motifs=3, scales=1.0)
        assert same_bits(got[0], want[0]) and np.array_equal(got[1], want[1])
        assert [list(s) for s in got[2]] == [list(s) for s in want[2]]
    got = mmotifs(source, p.astype(np.float32), i, max_motifs=3, scales=1.0)
    assert got[1].shape[0] >= 1


def test_stumpys_keywords_positions_and_names():
    t, p, i = stumpy_inputs(3, 100, 8, 8)
    positional = mmotifs(t, p, i, 0, None, None, 5, 2, 1e-8, None, None, True, 2.0, None, scales=GRID)
    keyword = mmotifs(
        T=t, P=p, I=i, min_neighbors=0, max_distance=None, cutoffs=None, max_matches=5,
        max_motifs=2, atol=1e-8, k=None, include=None, normalize=True, p=2.0,
        T_subseq_isconstant=None, scales=GRID,
    )
    assert same_bits(positional[0], keyword[0]) and np.array_equal(positional[1], keyword[1])
    # The predicate is STUMPY's: flags that say what the default rule says change nothing.
    flags = np.zeros((3, 93), dtype=bool)
    assert same_bits(
        mmotifs(t, p, i, T_subseq_isconstant=flags, scales=GRID)[0], mmotifs(t, p, i, scales=GRID)[0]
    )
    with pytest.raises(ValueError, match="one flag per window"):
        mmotifs(t, p, i, T_subseq_isconstant=np.zeros((3, 92), dtype=bool), scales=GRID)
    with pytest.raises(NotImplementedError):
        mmotifs(t, p, i, normalize=False, p=3.0)
    # An int `max_distance` is a number here (STUMPY calls it).
    assert same_bits(
        mmotifs(t, p, i, max_distance=4, scales=GRID)[0], mmotifs(t, p, i, max_distance=4.0, scales=GRID)[0]
    )


def test_default_scales_are_inferred_per_dimension():
    t, p, i = stumpy_inputs(3, 100, 8, 12)
    inferred = mmotifs(t, p, i, max_motifs=2)
    given = mmotifs(t, p, i, max_motifs=2, scales=[GRID] * 3)
    assert np.array_equal(inferred[1], given[1]) and np.allclose(inferred[0], given[0], atol=1e-12, equal_nan=True)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, scales=[GRID] * 2)


def test_bad_arguments():
    t, p, i = stumpy_inputs(3, 100, 8, 13)
    with pytest.raises(ValueError):
        mmotifs(t, p[:2], i, scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i[:, :-1], scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t[:, :20], p, i, scales=GRID)
    stray = i.copy()
    stray[0, 5] = p.shape[1]
    with pytest.raises(ValueError):
        mmotifs(t, p, stray, scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, cutoffs=[1.0, 2.0], scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, include=[3], scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, include=[-1], scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, atol=float("nan"), scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, min_neighbors=-1, scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, max_matches=-1, scales=GRID)
    with pytest.raises(ValueError):
        mmotifs(np.empty((0, 100)), p[:0], i[:0], scales=GRID)


def test_minus_one_neighbors_mean_none_and_other_negatives_are_errors():
    t, p, i = stumpy_inputs(3, 100, 8, 14)
    none = i.copy()
    none[:, :] = -1
    got = mmotifs(t, p, none, scales=GRID)
    assert got[0].shape == (1, 0)
    none[1, 5] = -7
    with pytest.raises(ValueError, match="I holds -7") as error:
        mmotifs(t, p, none, scales=GRID)
    assert "pass -1 for no neighbor, or len(T) + (-7)" in str(error.value)


def test_gaps_never_match_and_the_profile_comes_from_the_library():
    t = grid_walks(3, 120, 15)
    t[0, 30] = np.nan
    t[1, :4] = np.nan
    t[2, -1] = np.nan
    m = 8
    p, i = mstump(t, m)
    got = mmotifs(t, p, i, max_motifs=5, min_neighbors=0)
    distances, indices, subspaces, sizes = got
    assert indices.shape[0] >= 1
    for row, subspace in zip(indices, subspaces):
        for start in row[row >= 0]:
            window = t[subspace, start : start + m]
            assert np.isfinite(window).all()
    assert all(np.isfinite(s).any() for s in sizes)


@pytest.mark.skipif(
    not hasattr(ladders, "_hold_arm"),
    reason="needs the test hooks: build the wheel with `maturin develop --features test-hooks`",
)
def test_the_gil_is_released_while_a_call_runs():
    """The call parks inside the engine until this thread tells it to go on, which this thread
    can only do if the call released the GIL; a held GIL makes the call give up after a long
    hang guard and `_hold_release` report it."""
    t, p, i = stumpy_inputs(3, 60, 6, 16)
    result = []
    ladders._hold_arm()
    worker = threading.Thread(
        target=lambda: result.append(mmotifs(t, p, i)), daemon=True
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


def test_no_warning_in_ordinary_use():
    t, p, i = stumpy_inputs(3, 100, 8, 17)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mmotifs(t, p, i, scales=GRID)


def tie_heavy(kind: str, n: int = 80) -> np.ndarray:
    k = np.arange(n)
    rows = {
        "constant": [np.full(n, 5.0)] * 3,
        "period2": [(k + a) % 2 * 1.0 for a in range(3)],
        "period7": [((k * (a + 1)) % 7) * 3.0 for a in range(3)],
        "levels": [(k // (4 + a) % 3) * 2.0 for a in range(3)],
        "repeats": [np.tile(np.arange(10.0) ** (1 + a % 2), n // 10) + 100 * a for a in range(3)],
    }[kind]
    return np.array(rows, dtype=np.float64)


@pytest.mark.parametrize("kind", ["constant", "period2", "period7", "levels", "repeats"])
def test_tie_heavy_series_give_ordered_zone_respecting_motifs(kind):
    m = 6
    t = tie_heavy(kind)
    p, i = mstump(t, m)
    got = mmotifs(t, p, i, max_motifs=5, min_neighbors=0, cutoffs=1e9)
    again = mmotifs(t, p, i, max_motifs=5, min_neighbors=0, cutoffs=1e9)
    assert same_bits(got[0], again[0]) and np.array_equal(got[1], again[1])
    distances, indices, subspaces, sizes = got
    assert indices.shape[0] == len(subspaces) == len(sizes) >= 1
    zone = -(-m // 4)
    for row, distance in zip(indices, distances):
        starts = row[row >= 0]
        found = distance[: len(starts)]
        assert found[0] == 0
        assert np.all(np.diff(found) >= 0)
        for a in range(len(starts)):
            for b in range(a + 1, len(starts)):
                assert abs(starts[a] - starts[b]) > zone


def test_numpy_scalars_and_arrays_are_accepted_as_arguments():
    t, p, i = stumpy_inputs(3, 100, 8, 3)
    options = {"scales": GRID, "max_motifs": 3}
    want = mmotifs(t, p, i, cutoffs=[1e9, 1e9, 1e9], **options)
    assert want[1].shape[0] >= 1
    for cutoffs in (np.array([1e9, 1e9, 1e9]), np.float64(1e9), np.float32(1e9), 10**9):
        got = mmotifs(t, p, i, cutoffs=cutoffs, **options)
        assert same_bits(got[0], want[0]) and np.array_equal(got[1], want[1])
    got = mmotifs(t, p, i, k=np.int64(1), include=np.array([0]), max_matches=np.int64(3), **options)
    assert [list(s) for s in got[2]] == [list(s) for s in mmotifs(t, p, i, k=1, include=[0], max_matches=3, **options)[2]]
    assert same_bits(
        mmotifs(t, p, i, max_distance=np.float32(5), **options)[0],
        mmotifs(t, p, i, max_distance=5.0, **options)[0],
    )


def test_exactly_equal_distances_are_ordered_by_start_not_by_rounding():
    """Windows 3 and 9 are both at exact squared distance 3 from window 11, but their f64 roots
    differ in the last bit: they must come in start order."""
    t = np.array(
        [[2, 0, 4, 0, 3, 3, 2, 2, 4, 2, 1, 2, 2, 3, 4, 0, 1, 2, 1, 1, 2, 2, 3, 1, 4, 2, 0, 0, 4, 1, 0, 0, 4, 4]],
        dtype=np.float64,
    )
    windows = t.shape[1] - 3 + 1
    p = np.full((1, windows), 5.0)
    p[0, 11] = 0.0
    i = np.zeros((1, windows), dtype=np.int64)
    i[0, 11] = 20
    got = mmotifs(t, p, i, scales=1.0, cutoffs=9.0, max_distance=3.0, max_matches=None, min_neighbors=0)
    starts = list(got[1][0][got[1][0] >= 0])
    assert starts.index(3) < starts.index(9), starts
