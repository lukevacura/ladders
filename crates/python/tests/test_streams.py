"""`GrowingStream`/`BoundedStream`: the no-neighbor convention, and `update` in a Python loop
equalling `update_many` on bits — the two mutants `py-stream`'s spec calls out by name."""

import numpy as np
import pytest
from ladders import BoundedStream, GrowingStream

from conftest import (
    assert_mostly_finite,
    flat_run_series,
    market_level_walk,
    random_walk,
    walk_with_gaps,
)

M = 4
SCALE = 1.0
W = 9


def bits(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


def loop_updates(stream, series):
    """Every `(index, distance)` `update` returns, in order, skipping the `None`s."""
    out = []
    for x in series:
        result = stream.update(float(x))
        if result is not None:
            out.append(result)
    return out


SERIES = {
    "random_walk": lambda: random_walk(200, seed=1),
    "market_level": lambda: market_level_walk(200, seed=2),
    "with_gaps": lambda: walk_with_gaps(200, seed=3, gap_every=11, leading_gap=5),
    "flat_run": lambda: flat_run_series(200, seed=4, run_at=40, run_len=20),
}

# `with_gaps` legitimately has fewer matches (a real gap poisons every window it touches, at
# every scale): its own threshold only rules out the vacuous (near-0%) case, not the ~90% bar
# the gap-free series clear comfortably.
FINITE_THRESHOLD = {"random_walk": 0.9, "market_level": 0.9, "with_gaps": 0.4, "flat_run": 0.9}


def new_growing():
    return GrowingStream(M, SCALE)


def new_bounded():
    return BoundedStream(M, W, SCALE)


@pytest.mark.parametrize("series_name", SERIES)
@pytest.mark.parametrize("make_stream", [new_growing, new_bounded], ids=["growing", "bounded"])
def test_update_loop_equals_update_many(make_stream, series_name):
    series = SERIES[series_name]()

    sequential = loop_updates(make_stream(), series)

    chunked = make_stream()
    indices, distances = chunked.update_many(series)
    from_many = list(zip(indices.tolist(), distances.tolist()))

    assert len(from_many) == len(sequential)
    for (seq_index, seq_distance), (many_index, many_distance) in zip(sequential, from_many):
        assert seq_index == many_index
        assert bits(seq_distance) == bits(many_distance)
    assert_mostly_finite(indices, M, threshold=FINITE_THRESHOLD[series_name])


@pytest.mark.parametrize("series_name", SERIES)
@pytest.mark.parametrize("make_stream", [new_growing, new_bounded], ids=["growing", "bounded"])
def test_update_many_split_anywhere_equals_one_call(make_stream, series_name):
    """Splitting `update_many` into two calls (mid-window or not) must return exactly what one
    call over the whole series returns, in the same order — the same property the Rust-side
    `update_many_matches_update_for_*` proptests check, from the Python side of the boundary."""
    series = SERIES[series_name]()
    split = len(series) // 3

    whole = make_stream()
    whole_indices, whole_distances = whole.update_many(series)

    split_stream = make_stream()
    first_indices, first_distances = split_stream.update_many(series[:split])
    second_indices, second_distances = split_stream.update_many(series[split:])
    split_indices = np.concatenate([first_indices, second_indices])
    split_distances = np.concatenate([first_distances, second_distances])

    assert np.array_equal(whole_indices, split_indices)
    assert np.array_equal(whole_distances.view(np.uint64), split_distances.view(np.uint64))
    assert_mostly_finite(whole_indices, M, threshold=FINITE_THRESHOLD[series_name])


@pytest.mark.parametrize("make_stream", [new_growing, new_bounded], ids=["growing", "bounded"])
def test_no_neighbor_is_minus_one_and_infinity(make_stream):
    """Every row inside the exclusion zone (here, rows 0 and 1 for `m=4`) has no neighbor:
    index exactly `-1`, distance exactly `inf` (STUMPY's own convention), pinned on bits so a
    mutant that maps "no neighbor" to some other sentinel (e.g. `0`, or the largest `u64` left
    unmapped) is caught rather than coincidentally matching."""
    stream = make_stream()
    results = loop_updates(stream, [1.0, 5.0, 2.0, 9.0, 3.0])

    first_two = results[:2]
    assert [index for index, _ in first_two] == [-1, -1]
    for _, distance in first_two:
        assert bits(distance) == bits(float("inf"))


def test_bounded_matches_growing_before_the_window_fills():
    """Before `w` samples have arrived, `BoundedStream` has nothing to forget yet, so it must
    return exactly what `GrowingStream` does — mirrors the Rust-side
    `bounded_eq_growing_before_full` proptest."""
    series = random_walk(W - 1, seed=5)
    growing = loop_updates(new_growing(), series)
    bounded = loop_updates(BoundedStream(M, W, SCALE), series)

    assert len(growing) == len(bounded)
    for (g_index, g_distance), (b_index, b_distance) in zip(growing, bounded):
        assert g_index == b_index
        assert bits(g_distance) == bits(b_distance)
    # Only `W - 1 = 8` samples here (barely past `m`), so the tail past the exclusion zone is
    # just 4 rows: a lower bar than the longer series above use, still enough to rule out a
    # vacuous (all `-1`) pair of profiles.
    assert_mostly_finite([index for index, _ in growing], M, threshold=0.5)


def test_invalid_m_raises_value_error_not_panic():
    for bad_m in (-1, 0, 1, 2):
        with pytest.raises(ValueError):
            GrowingStream(bad_m, SCALE)
        with pytest.raises(ValueError):
            BoundedStream(bad_m, W, SCALE)


def test_invalid_w_raises_value_error():
    with pytest.raises(ValueError):
        BoundedStream(M, M - 1, SCALE)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_scale_raises_value_error(bad_scale):
    with pytest.raises(ValueError):
        GrowingStream(M, bad_scale)
    with pytest.raises(ValueError):
        BoundedStream(M, W, bad_scale)
