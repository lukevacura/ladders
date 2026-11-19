"""`motif_pairs(t, m, *, scale, threads)`: the pair of highest exact correlation, ties to the
smaller (a, b), a < b, equal on bits to the pick over `matrix_profile`'s own P and I at every
thread count, on ties, flat runs and gaps."""

import numpy as np
import pytest
from ladders import infer_scale, matrix_profile, motif_pairs

from conftest import flat_run_series, random_walk, walk_with_gaps

M = 20
THREADS = [1, 2, 7, None]


def tie_free(n, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


def pick_from_profile(p, i):
    """The pair of the windows whose distance is within rounding of the least, as (earlier,
    later), the smallest first; the earlier window's distance: written over the profile in numpy."""
    candidates = np.flatnonzero((i >= 0) & np.isfinite(p))
    if len(candidates) == 0:
        return None
    least = p[candidates].min()
    near = candidates[p[candidates] <= least * (1 + 1e-9) + 1e-12]
    a, b = min((min(int(w), int(i[w])), max(int(w), int(i[w]))) for w in near)
    return a, b, p[a]


def assert_pair(series, m, scale, threads):
    p, i, _, _ = matrix_profile(series, m, scale, threads=1)
    indices, distances, used = motif_pairs(series, m, scale=scale, threads=threads)
    expected = pick_from_profile(p, i)
    assert indices.dtype == np.int64 and distances.dtype == np.float64
    assert used == (scale if scale is not None else infer_scale(series, m))
    if expected is None:
        assert indices.shape == (0, 2) and distances.shape == (0,)
        return
    a, b, distance = expected
    assert indices.shape == (1, 2) and distances.shape == (1,)
    assert indices[0].tolist() == [a, b]
    assert distances.view(np.uint64)[0] == np.float64(distance).view(np.uint64)


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize(
    "series,scale",
    [
        (tie_free(500, 1), 1e-6),
        (random_walk(500, 2), 1.0),
        (random_walk(500, 3), None),
        (walk_with_gaps(500, 4, gap_every=97, leading_gap=3), 1.0),
        (flat_run_series(500, 5, run_at=100, run_len=60), 1.0),
        (np.round(np.random.default_rng(6).integers(0, 3, size=400)).astype(float), 1.0),
    ],
)
def test_the_pair_is_the_pick_over_the_profile(series, scale, threads):
    assert_pair(series, M, scale, threads)


def test_two_flat_runs_are_a_pair_at_distance_zero():
    series = random_walk(500, 7)
    series[60:100] = 5.0
    series[300:340] = 2.0
    indices, distances, _ = motif_pairs(series, M, scale=1.0)
    assert distances.tolist() == [0.0]
    assert_pair(series, M, 1.0, 1)


def test_a_copied_stretch_is_the_pair():
    series = tie_free(600, 8)
    series[400:460] = series[100:160]
    indices, distances, _ = motif_pairs(series, M, scale=1e-6)
    assert indices[0].tolist() == [100, 400]
    assert distances[0] == 0.0


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32, np.float64])
def test_arrays_and_lists_of_every_dtype_are_accepted(dtype):
    series = np.round(random_walk(300, 9) * 4).astype(dtype)
    expected = motif_pairs(series.astype(np.float64), M, scale=1.0)
    for given in (series, series.tolist()):
        got = motif_pairs(given, M, scale=1.0)
        assert got.indices.tolist() == expected.indices.tolist()
        assert got.distances.tolist() == expected.distances.tolist()
        assert got.scale == 1.0


def test_no_pair_gives_empty_arrays():
    indices, distances, used = motif_pairs(random_walk(M + 1, 10), M, scale=1.0)
    assert indices.shape == (0, 2) and indices.dtype == np.int64
    assert distances.shape == (0,) and distances.dtype == np.float64
    assert used == 1.0


def test_a_series_shorter_than_the_window_has_no_pair_and_no_scale_is_refused():
    indices, distances, _ = motif_pairs(random_walk(M - 1, 11), M, scale=1.0)
    assert indices.shape == (0, 2) and distances.shape == (0,)
    with pytest.raises(ValueError):
        motif_pairs(random_walk(M - 1, 11), M)
    with pytest.raises(ValueError):
        motif_pairs(random_walk(100, 12), 0)


def test_a_pair_whose_entries_differ_by_a_rounding_is_reported_earlier_window_first():
    series = np.array([674, 986, 474, 901, 1797, 459, 649, 667, 123, 575, 1431, 1085, 27, 235, 950], dtype=float)
    indices, distances, _ = motif_pairs(series, 8, scale=1.0)
    assert indices.tolist() == [[0, 6]]
    assert distances[0] == 2.345009105857163


def test_the_result_is_a_named_tuple():
    result = motif_pairs(random_walk(200, 13), M, scale=1.0)
    assert result.indices is result[0] and result.distances is result[1] and result.scale == result[2]


def test_options_are_keyword_only():
    with pytest.raises(TypeError):
        motif_pairs(random_walk(200, 14), M, 1.0)
