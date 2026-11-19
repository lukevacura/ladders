"""`atsc`, `allc` and `chains`: STUMPY's anchored chain and all-chain set over left and right
indices. Fed the same indices they return STUMPY's arrays (`allc`'s set in first-window order,
STUMPY's own order being arbitrary), and `chains` is `allc` over `matrix_profile`'s indices."""

import numpy as np
import pytest
from ladders import allc, atsc, chains, matrix_profile

from conftest import flat_run_series, random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

M = 12


def forward_indices(n, seed):
    """Left neighbors before, right neighbors after, most of them linked back."""
    rng = np.random.default_rng(seed)
    il = np.full(n, -1, dtype=np.int64)
    ir = np.full(n, -1, dtype=np.int64)
    for i in range(n - 1):
        if rng.random() < 0.85:
            ir[i] = rng.integers(i + 1, min(n, i + 4))
    for i in range(1, n):
        if rng.random() < 0.2:
            il[i] = rng.integers(0, i)
    for i in range(n):
        if ir[i] >= 0 and rng.random() < 0.7 and il[ir[i]] == -1:
            il[ir[i]] = i
    return il, ir


def sorted_set(chain_set):
    return sorted((list(c) for c in chain_set), key=lambda c: c[0])


@pytest.mark.parametrize("seed", range(12))
def test_atsc_and_allc_equal_stumpys_on_identical_indices(seed):
    il, ir = forward_indices(60 + 7 * seed, seed)
    expected_set, expected_longest = stumpy.allc(il, ir)
    found_set, found_longest = allc(il, ir)
    assert [list(c) for c in found_set] == sorted_set(expected_set)
    assert np.array_equal(found_longest, expected_longest)
    assert all(c.dtype == np.int64 for c in found_set) and found_longest.dtype == np.int64
    for j in range(len(il)):
        got, want = atsc(il, ir, j), stumpy.atsc(il, ir, j)
        assert np.array_equal(got, want) and got.dtype == want.dtype


def test_the_stumpy_documentation_example():
    t = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    p, i, il, ir = matrix_profile(t, 3, 1e-6)
    assert il.tolist() == [-1, -1, 0, 1, 0] and ir.tolist() == [4, 3, 4, -1, -1]
    assert atsc(il, ir, 1).tolist() == [1, 3]
    found_set, longest = allc(il, ir)
    assert [c.tolist() for c in found_set] == [[0, 4], [1, 3], [2]]
    assert longest.tolist() == [0, 4]


SERIES = {
    "random_walk": lambda: np.cumsum(np.random.default_rng(1).normal(size=700)),
    "sine_noise": lambda: np.sin(np.arange(700) / 5.0)
    + np.random.default_rng(2).normal(size=700) * 0.3,
    "drift": lambda: np.cumsum(np.random.default_rng(3).normal(size=500) + 0.05),
}


@pytest.mark.parametrize("name", SERIES)
def test_chains_of_a_tie_free_series_equal_stumpys(name):
    t = SERIES[name]()
    mp = stumpy.stump(t, M)
    found_set, longest = chains(t, M, 1e-6)
    expected_set, expected_longest = stumpy.allc(mp[:, 2].astype(np.int64), mp[:, 3].astype(np.int64))
    assert [list(c) for c in found_set] == sorted_set(expected_set)
    assert np.array_equal(longest, expected_longest)
    assert len(longest) >= 2


@pytest.mark.parametrize(
    "series",
    [
        random_walk(500, 4),
        flat_run_series(500, 5, run_at=100, run_len=40),
        walk_with_gaps(500, 6, gap_every=53, leading_gap=3),
        np.tile([0.0, 1.0, 3.0], 120),
    ],
)
def test_chains_are_allc_over_matrix_profile_indices_at_every_thread_count(series):
    _, _, il, ir = matrix_profile(series, M, 1.0, threads=1)
    expected_set, expected_longest = allc(il, ir)
    for threads in (1, 2, None):
        found_set, longest = chains(series, M, 1.0, threads=threads)
        assert len(found_set) == len(expected_set)
        assert all(np.array_equal(a, b) for a, b in zip(found_set, expected_set))
        assert np.array_equal(longest, expected_longest)
    assert sum(len(c) for c in expected_set) == len(il)


def test_empty_and_degenerate_inputs():
    empty = np.array([], dtype=np.int64)
    found_set, longest = allc(empty, empty)
    assert found_set == [] and longest.size == 0
    one = np.array([-1], dtype=np.int64)
    assert atsc(one, one, 0).tolist() == [0]


@pytest.mark.parametrize(
    "il, ir, j",
    [
        ([-1, -1], [-1], 0),
        ([-1, -1], [5, -1], 0),
        ([-1, -1], [-2, -1], 0),
        ([-1, 1], [-1, -1], 0),
        ([-1, -1], [0, -1], 0),
        ([-1, -1], [1, -1], 2),
        ([-1, -1], [1, -1], -1),
    ],
)
def test_indices_that_cannot_belong_to_one_series_are_refused(il, ir, j):
    il, ir = np.array(il, dtype=np.int64), np.array(ir, dtype=np.int64)
    with pytest.raises(ValueError):
        atsc(il, ir, j)
    if j == 0:
        with pytest.raises(ValueError):
            allc(il, ir)
