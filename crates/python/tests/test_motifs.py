"""`motifs(t, profile_for(t, m), ...)`: STUMPY's `(motif_distances, motif_indices)` from the series itself, with
the first column each motif's own window at distance 0, rows padded with `nan` and `-1`, the
thread count never changing a bit, and (when STUMPY is installed) STUMPY's own answer on
tie-free series."""

import warnings

import numpy as np
import pytest
from ladders import motifs

from conftest import profile_for, random_walk

M = 16
SCALE = 2.0**-20


def grid_walk(n, seed):
    """A tie-free walk exactly on the `SCALE` grid, so STUMPY reads the same values."""
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.normal(size=n)) * 2**20) / 2**20


def u64(x):
    return np.asarray(x).view(np.uint64)


def test_the_first_column_is_each_motifs_own_window_at_distance_zero():
    series = grid_walk(800, seed=1)
    distances, indices = motifs(
        series, profile_for(series, M), scale=SCALE, max_motifs=5, max_matches=4, max_distance=np.inf, cutoff=np.inf
    )
    assert distances.shape == indices.shape and distances.shape[0] >= 1
    assert distances.dtype == np.float64 and indices.dtype == np.int64
    assert np.all(distances[:, 0] == 0.0)
    assert np.all(indices[:, 0] >= 0)
    for row in distances:
        found = row[~np.isnan(row)]
        assert np.all(np.diff(found) >= 0)


def test_rows_are_padded_with_nan_and_minus_one_together():
    series = grid_walk(800, seed=2)
    distances, indices = motifs(
        series, profile_for(series, M), scale=SCALE, max_motifs=40, max_matches=10, max_distance=3.0, cutoff=np.inf
    )
    assert distances.shape[0] > 1
    assert np.array_equal(np.isnan(distances), indices == -1)
    assert np.any(indices == -1), "rows of unequal length are padded"


def test_motif_windows_are_outside_each_others_zones():
    series = grid_walk(1200, seed=3)
    _, indices = motifs(series, profile_for(series, M), scale=SCALE, max_motifs=30, max_matches=6, cutoff=np.inf)
    zone = -(-M // 4)
    own = indices[:, 0]
    for k, window in enumerate(own):
        assert np.all(np.abs(own[:k] - window) > zone)
        row = indices[k][indices[k] >= 0]
        assert len(row) == len(set(row))
        assert all(abs(a - b) > zone for i, a in enumerate(row) for b in row[:i])
        for earlier in indices[:k]:
            assert np.all(np.abs(earlier[earlier >= 0] - window) > zone)


def test_no_motif_is_shape_one_by_zero():
    distances, indices = motifs(grid_walk(300, seed=4), profile_for(grid_walk(300, seed=4), M), scale=SCALE, min_neighbors=1000)
    assert distances.shape == indices.shape == (1, 0)


def test_max_matches_none_keeps_every_match():
    series = grid_walk(800, seed=5)
    _, ten = motifs(series, profile_for(series, M), scale=SCALE, max_distance=np.inf, cutoff=np.inf)
    _, every = motifs(series, profile_for(series, M), scale=SCALE, max_distance=np.inf, cutoff=np.inf, max_matches=None)
    assert ten.shape[1] == 10 and every.shape[1] > 10
    assert np.array_equal(every[:, :10], ten)


@pytest.mark.parametrize("seed", [6, 7])
def test_every_thread_count_gives_the_same_bits(seed):
    series = grid_walk(900, seed=seed)
    runs = [
        motifs(series, profile_for(series, M), scale=SCALE, threads=threads, max_motifs=4, cutoff=np.inf)
        for threads in [1, 2, 7, None]
    ]
    for distances, indices in runs[1:]:
        assert np.array_equal(u64(distances), u64(runs[0][0]))
        assert np.array_equal(indices, runs[0][1])


def test_scale_none_infers_the_scale_matrix_profile_does():
    series = random_walk(400, seed=8)
    inferred = motifs(series, profile_for(series, 12), max_motifs=3, cutoff=np.inf)
    explicit = motifs(series, profile_for(series, 12), scale=1.0, max_motifs=3, cutoff=np.inf)
    assert np.array_equal(u64(inferred[0]), u64(explicit[0]))
    assert np.array_equal(inferred[1], explicit[1])


def test_a_max_motifs_below_one_is_an_error_naming_stumpys_one():
    series = grid_walk(500, seed=9)
    for below in [0, -1, -(2**63)]:
        with pytest.raises(ValueError, match=f"max_motifs={below} is below 1") as error:
            motifs(series, profile_for(series, M), scale=SCALE, max_motifs=below)
        assert "pass max_motifs=1 to get that" in str(error.value)
    assert motifs(series, profile_for(series, M), scale=SCALE, max_motifs=1)[1].shape[0] == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"min_neighbors": -1}, {"max_matches": -1}, {"threads": 0}, {"threads": True}],
)
def test_a_bad_argument_is_a_value_error(kwargs):
    with pytest.raises(ValueError):
        motifs(grid_walk(200, seed=10), profile_for(grid_walk(200, seed=10), M), scale=SCALE, **kwargs)


def test_a_series_shorter_than_the_window_is_a_value_error():
    with pytest.raises(ValueError):
        motifs(grid_walk(10, seed=11), profile_for(grid_walk(10, seed=11), M))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("m", [8, 16, 33])
def test_stumpy_motifs_agree_on_tie_free_series(seed, m):
    stumpy = pytest.importorskip("stumpy")
    series = grid_walk(300 + 90 * seed, seed=seed)
    profile = stumpy.stump(series, m)[:, 0].astype(np.float64)
    checked = 0
    for max_motifs in (1, 3, 6):
        for max_matches in (3, 10):
            for kwargs in ({}, {"max_distance": 4.0, "cutoff": 6.0}, {"min_neighbors": 2}):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    theirs = stumpy.motifs(
                        series, profile, max_motifs=max_motifs, max_matches=max_matches, **kwargs
                    )
                ours = motifs(
                    series, profile_for(series, m), scale=SCALE, max_motifs=max_motifs, max_matches=max_matches, **kwargs
                )
                assert ours[1].shape == theirs[1].shape
                assert np.array_equal(ours[1], theirs[1])
                assert np.array_equal(np.isnan(ours[0]), np.isnan(theirs[0]))
                assert np.allclose(
                    np.nan_to_num(ours[0]), np.nan_to_num(theirs[0]), rtol=1e-8, atol=1e-6
                )
                checked += ours[1].size
    assert checked > 0
