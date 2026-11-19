"""`discords(t, m, k, scale, threads)`: the top-k windows by exact nearest-neighbor distance,
farthest first, pairwise outside the exclusion zone, the same bits at every thread count."""

import numpy as np
import pytest
from ladders import discords, matrix_profile

from conftest import random_walk, walk_with_gaps

M = 20
ZONE = -(-M // 4)
THREADS = [1, 2, 7, None]


def tie_free(n, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


def greedy_from_profile(p, i, k):
    """The pick done in plain Python from `matrix_profile`'s P: farthest first, ties to the
    smaller index, skipping windows within the zone of a pick."""
    order = sorted((j for j in range(len(p)) if i[j] >= 0), key=lambda j: (-p[j], j))
    picked = []
    for j in order:
        if len(picked) < k and all(abs(j - q) > ZONE for q in picked):
            picked.append(j)
    return picked


@pytest.mark.parametrize("threads", THREADS)
@pytest.mark.parametrize(
    "series,scale",
    [
        (tie_free(500, 1), 1e-6),
        (random_walk(500, 2), 1.0),
        (walk_with_gaps(500, 3, gap_every=97, leading_gap=3), 1.0),
    ],
)
def test_discords_are_the_zone_respecting_greedy_pick_of_the_profile(series, scale, threads):
    p, i, _, _ = matrix_profile(series, M, scale, threads=1)
    d, idx = discords(series, M, 5, scale, threads=threads)
    expected = greedy_from_profile(p, i, 5)
    assert idx.tolist() == expected
    assert d.view(np.uint64).tolist() == p[expected].view(np.uint64).tolist()
    assert all(abs(a - b) > ZONE for n, a in enumerate(idx) for b in idx[:n])
    assert list(d) == sorted(d, reverse=True)


def test_defaults_give_one_discord():
    series = tie_free(300, 4)
    d, idx = discords(series, M)
    p, _, _, _ = matrix_profile(series, M)
    assert len(d) == len(idx) == 1
    assert idx[0] == np.argmax(p)


def test_fewer_than_k_come_back_when_the_zones_leave_no_room():
    d, idx = discords(tie_free(60, 5), M, 50)
    assert 0 < len(idx) < 50
    assert len(d) == len(idx)


def test_no_discords_when_k_is_zero():
    assert len(discords(tie_free(100, 6), M, 0)[0]) == 0


def test_a_series_shorter_than_a_window_is_rejected_as_matrix_profile_does():
    with pytest.raises(ValueError):
        discords(np.arange(5.0), M, 3)


def test_a_negative_k_is_rejected():
    with pytest.raises(ValueError):
        discords(tie_free(100, 7), M, -1)


def test_top_one_equals_stumpys_argmax_on_a_tie_free_series():
    stumpy = pytest.importorskip("stumpy")
    series = tie_free(400, 8)
    _, idx = discords(series, M, 1, scale=1e-6)
    p = stumpy.stump(series, M)[:, 0].astype(float)
    assert idx[0] == int(np.argmax(p))
