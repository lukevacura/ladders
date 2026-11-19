"""`aamp_mmotifs(T, P, I, ...)` and `mmotifs(..., normalize=False)`: STUMPY's non-normalized
multi-dimensional motifs, on tie-free data equal to STUMPY's (indices exactly, distances to
rounding), one routing for both names, with real integer and float32 arrays, gaps, and bad
arguments a `ValueError` or `NotImplementedError`."""

import threading

import ladders
import numpy as np
import pytest
from ladders import aamp_mmotifs, maamp, mmotifs

stumpy = pytest.importorskip("stumpy")

pytestmark = pytest.mark.filterwarnings("ignore:'where' used without 'out'")

GRID = 1 / 64


def grid_walks(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.standard_normal((d, n)), axis=1) * 64.0) / 64.0 * 3.0 + 5.0


def stumpy_inputs(d: int, n: int, m: int, seed: int):
    t = grid_walks(d, n, seed)
    p, i = stumpy.maamp(t, m)
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
    {"max_motifs": 3, "cutoffs": [1e9, 8.0, 1e9, 1e9]},
    {"max_motifs": 3, "atol": 0.5},
]


@pytest.mark.parametrize("params", PARAMS)
@pytest.mark.parametrize("seed", range(4))
def test_equals_stumpy(seed, params):
    d, n, m = 4, 90 + 10 * seed, 6 + seed
    t, p, i = stumpy_inputs(d, n, m, seed)
    got = aamp_mmotifs(t, p, i, scale=GRID, **params)
    want = stumpy.aamp_mmotifs(t, p, i, **params)
    assert_equals_stumpy(got, want)


def test_the_two_names_are_one_routing_and_stumpy_reroutes_too():
    t, p, i = stumpy_inputs(3, 120, 8, 21)
    options = {"max_motifs": 4, "min_neighbors": 0, "cutoffs": 1e9}
    direct = aamp_mmotifs(t, p, i, scale=GRID, **options)
    routed = mmotifs(t, p, i, normalize=False, scales=GRID, **options)
    for ours, theirs in zip(direct[:2], routed[:2]):
        assert same_bits(ours, theirs)
    assert [list(s) for s in direct[2]] == [list(s) for s in routed[2]]
    assert all(same_bits(a, b) for a, b in zip(direct[3], routed[3]))
    assert direct[1].shape[0] >= 2
    assert_equals_stumpy(routed, stumpy.mmotifs(t, p, i, normalize=False, **options))
    # The metrics see different distances: the z-normalized motifs of the same series differ.
    zp, zi = stumpy.mstump(t, 8)
    assert not np.array_equal(
        mmotifs(t, zp, zi, scales=GRID, **options)[1], direct[1]
    ) or not same_bits(mmotifs(t, zp, zi, scales=GRID, **options)[0], direct[0])


def test_result_types_and_empty_results():
    t, p, i = stumpy_inputs(3, 120, 8, 5)
    distances, indices, subspaces, sizes = aamp_mmotifs(
        t, p, i, max_motifs=3, max_matches=5, min_neighbors=0, scale=GRID
    )
    assert distances.dtype == np.float64 and indices.dtype == np.int64
    assert distances.shape == indices.shape and distances.shape[0] == len(subspaces) == len(sizes) >= 2
    assert all(isinstance(s, np.ndarray) and s.dtype == np.int64 for s in subspaces)
    assert all(isinstance(s, np.ndarray) and s.dtype == np.float64 and s.shape == (3,) for s in sizes)
    assert (distances[:, 0] == 0).all()
    assert np.array_equal(np.isnan(distances), indices == -1)
    empty = aamp_mmotifs(t, p, i, cutoffs=-1.0, scale=GRID)
    assert empty[0].shape == (1, 0) and empty[1].shape == (1, 0) and empty[2] == [] and empty[3] == []
    assert aamp_mmotifs(t, np.full_like(p, np.inf), i, scale=GRID)[2] == []


def test_distances_are_in_the_series_units():
    t, p, i = stumpy_inputs(3, 120, 8, 6)
    distances = aamp_mmotifs(t, p, i, max_motifs=2, min_neighbors=0, scale=GRID)[0]
    scaled = aamp_mmotifs(t * 4.0, p * 4.0, i, max_motifs=2, min_neighbors=0, scale=GRID * 4)[0]
    assert np.allclose(scaled, distances * 4.0, rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize("cast", ["int64", "int32", "float32", "list", "int_list"])
def test_real_integer_and_float32_arrays(cast):
    rng = np.random.default_rng(9)
    ints = np.cumsum(rng.integers(-3, 4, size=(3, 100)), axis=1)
    t64 = ints.astype(np.float64)
    p, i = stumpy.maamp(t64, 7)
    want = aamp_mmotifs(t64, p, i, max_motifs=3, scale=1.0)
    source = {
        "int64": ints.astype(np.int64),
        "int32": ints.astype(np.int32),
        "float32": ints.astype(np.float32),
        "list": t64.tolist(),
        "int_list": ints.tolist(),
    }[cast]
    for index in (i, i.astype(np.int32), i.tolist()):
        got = aamp_mmotifs(source, p, index, max_motifs=3, scale=1.0)
        assert same_bits(got[0], want[0]) and np.array_equal(got[1], want[1])
        assert [list(s) for s in got[2]] == [list(s) for s in want[2]]
    assert aamp_mmotifs(source, p.astype(np.float32), i, max_motifs=3, scale=1.0)[1].shape[0] >= 1


def test_stumpys_keywords_positions_and_unsupported_arguments():
    t, p, i = stumpy_inputs(3, 100, 8, 8)
    positional = aamp_mmotifs(t, p, i, 0, None, None, 5, 2, 1e-8, None, None, 2.0, GRID)
    keyword = aamp_mmotifs(
        T=t, P=p, I=i, min_neighbors=0, max_distance=None, cutoffs=None, max_matches=5,
        max_motifs=2, atol=1e-8, k=None, include=None, p=2.0, scale=GRID,
    )
    assert same_bits(positional[0], keyword[0]) and np.array_equal(positional[1], keyword[1])
    with pytest.raises(NotImplementedError):
        aamp_mmotifs(t, p, i, p=3.0)
    with pytest.raises(NotImplementedError):
        mmotifs(t, p, i, normalize=False, p=1.0)
    with pytest.raises(ValueError):
        mmotifs(t, p, i, normalize=False, scales=[GRID, GRID, GRID])
    # `p` is ignored when normalized, as in STUMPY.
    zp, zi = stumpy.mstump(t, 8)
    assert same_bits(
        mmotifs(t, zp, zi, p=3.0, scales=GRID)[0], mmotifs(t, zp, zi, scales=GRID)[0]
    )


def test_inferred_scale_and_bad_arguments():
    t, p, i = stumpy_inputs(3, 100, 8, 13)
    inferred = aamp_mmotifs(t, p, i, max_motifs=2)
    given = aamp_mmotifs(t, p, i, max_motifs=2, scale=GRID)
    assert np.array_equal(inferred[1], given[1]) and np.allclose(inferred[0], given[0], equal_nan=True)
    with pytest.raises(ValueError):
        aamp_mmotifs(t, p[:2], i, scale=GRID)
    with pytest.raises(ValueError):
        aamp_mmotifs(t, p, i, k=3, scale=GRID)
    with pytest.raises(ValueError):
        aamp_mmotifs(t, p, i, include=[3], scale=GRID)
    with pytest.raises(ValueError):
        aamp_mmotifs(t, p, i, cutoffs=[1.0], scale=GRID)
    with pytest.raises(ValueError):
        aamp_mmotifs(t, p, i, scale=-1.0)
    with pytest.raises(ValueError, match="pass max_motifs=1 to get that"):
        aamp_mmotifs(t, p, i, max_motifs=0, scale=GRID)
    with pytest.raises(ValueError, match="I holds -2"):
        aamp_mmotifs(t, p, np.full_like(i, -2), scale=GRID)


def test_gaps_never_match_and_the_profile_comes_from_the_library():
    t = grid_walks(3, 120, 15)
    t[0, 30] = np.nan
    t[1, :4] = np.nan
    t[2, -1] = np.nan
    m = 8
    p, i = maamp(t, m)
    distances, indices, subspaces, sizes = aamp_mmotifs(t, p, i, max_motifs=5, min_neighbors=0)
    assert indices.shape[0] >= 1
    for row, subspace in zip(indices, subspaces):
        for start in row[row >= 0]:
            assert np.isfinite(t[subspace, start : start + m]).all()


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
    worker = threading.Thread(target=lambda: result.append(aamp_mmotifs(t, p, i)), daemon=True)
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
    p, i = maamp(t, m)
    got = aamp_mmotifs(t, p, i, max_motifs=5, min_neighbors=0, cutoffs=1e9)
    again = mmotifs(t, p, i, max_motifs=5, min_neighbors=0, cutoffs=1e9, normalize=False)
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
