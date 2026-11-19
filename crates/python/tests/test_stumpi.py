"""`stumpi`: `left_P_`/`left_I_` must equal the native class it wraps, bit for bit — the third
mutant `py-stream`'s spec calls out ("the stumpi wrapper reading a different stream than the
native class") — through a warm start plus a run of causal `update` calls."""

import numpy as np
import pytest
from ladders import BoundedStream, GrowingStream, stumpi

from conftest import assert_mostly_finite, market_level_walk, random_walk, walk_with_gaps

M = 4
W = 9
SCALE = 1.0


def bits(x: float) -> int:
    return int(np.float64(x).view(np.uint64))


def native_left(make_native, warm_start, tail):
    stream = make_native()
    stream.update_many(warm_start)
    indices = []
    distances = []
    for t in tail:
        result = stream.update(float(t))
        if result is not None:
            index, distance = result
            indices.append(index)
            distances.append(distance)
    # `update_many`'s own entries for the warm start, then one per completed `update` in the
    # tail: concatenated the same way `stumpi.left_P_`/`left_I_` accumulate across its own
    # warm start and its own `update` calls.
    warm_indices, warm_distances = make_native().update_many(warm_start)
    return (
        np.concatenate([warm_indices, np.array(indices, dtype=np.int64)]),
        np.concatenate([warm_distances, np.array(distances, dtype=np.float64)]),
    )


SERIES = {
    "random_walk": lambda: random_walk(60, seed=40),
    "market_level": lambda: market_level_walk(60, seed=41),
    "with_gaps": lambda: walk_with_gaps(60, seed=42, gap_every=9),
}

# `with_gaps` legitimately has fewer matches (a real gap poisons every window it touches, at
# every scale): its own threshold only rules out the vacuous (near-0%) case.
FINITE_THRESHOLD = {"random_walk": 0.9, "market_level": 0.9, "with_gaps": 0.4}


@pytest.mark.parametrize("series_name", SERIES)
def test_stumpi_egress_false_matches_growing_stream(series_name):
    series = SERIES[series_name]()
    warm_start, tail = series[:30], series[30:]

    s = stumpi(warm_start, M, scale=SCALE, egress=False)
    for t in tail:
        s.update(float(t))

    expected_indices, expected_distances = native_left(
        lambda: GrowingStream(M, SCALE), warm_start, tail
    )

    assert np.array_equal(s.left_I_, expected_indices)
    assert np.array_equal(s.left_P_.view(np.uint64), expected_distances.view(np.uint64))
    assert_mostly_finite(s.left_I_, M, threshold=FINITE_THRESHOLD[series_name])


@pytest.mark.parametrize("series_name", SERIES)
def test_stumpi_egress_true_matches_bounded_stream(series_name):
    series = SERIES[series_name]()
    warm_start, tail = series[:30], series[30:]

    s = stumpi(warm_start, M, scale=SCALE, egress=True, w=W)
    for t in tail:
        s.update(float(t))

    expected_indices, expected_distances = native_left(
        lambda: BoundedStream(M, W, SCALE), warm_start, tail
    )

    assert np.array_equal(s.left_I_, expected_indices)
    assert np.array_equal(s.left_P_.view(np.uint64), expected_distances.view(np.uint64))
    assert_mostly_finite(s.left_I_, M, threshold=FINITE_THRESHOLD[series_name])


def test_stumpi_has_no_p_or_i_attributes():
    """Only the causal `left_P_`/`left_I_` are exposed, never `P_`/`I_` (this library
    never looks ahead)."""
    s = stumpi(random_walk(20, seed=43), M, scale=SCALE, egress=False)
    assert not hasattr(s, "P_")
    assert not hasattr(s, "I_")


def test_stumpi_update_returns_none():
    s = stumpi(random_walk(20, seed=44), M, scale=SCALE, egress=False)
    assert s.update(1.0) is None


def test_egress_true_without_w_keeps_the_length_of_the_history_as_stumpy_does():
    warm = random_walk(20, seed=45)
    default, explicit = (
        stumpi(warm, M, scale=SCALE, egress=True),
        stumpi(warm, M, scale=SCALE, egress=True, w=len(warm)),
    )
    for stream in (default, explicit):
        for x in random_walk(15, seed=46):
            stream.update(x)
    assert np.array_equal(default.left_I_, explicit.left_I_)
    assert np.array_equal(default.left_P_.view(np.uint64), explicit.left_P_.view(np.uint64))
    with pytest.raises(ValueError):
        stumpi(random_walk(M - 1, seed=47), M, scale=SCALE, egress=True)


def test_egress_false_rejects_w():
    with pytest.raises(ValueError):
        stumpi(random_walk(20, seed=46), M, scale=SCALE, egress=False, w=W)


def test_egress_true_rejects_w_below_m():
    with pytest.raises(ValueError):
        stumpi(random_walk(20, seed=47), M, scale=SCALE, egress=True, w=M - 1)


def test_invalid_m_raises_value_error():
    with pytest.raises(ValueError):
        stumpi(random_walk(20, seed=48), 2, scale=SCALE, egress=False)


@pytest.mark.parametrize("egress", [False, True])
def test_a_far_outlier_moves_the_stream_to_the_wide_path(egress):
    series = random_walk(60, seed=41)
    kwargs = {"egress": egress, **({"w": 40} if egress else {})}
    stream = stumpi(series[:30], M, scale=SCALE, **kwargs)
    before = stream.route_counts()
    assert before["narrow_updates"] == 30 and before["wide_updates"] == 0
    assert before["switches_to_wide"] == 0
    # Past twice the narrow anchor's reach, inside the limit past which a tick is withheld.
    stream.update(1.2e9)
    after = stream.route_counts()
    assert after["narrow_updates"] == 30 and after["wide_updates"] == 1
    assert after["switches_to_wide"] == 1
    stream.update(1e12)
    assert stream.route_counts()["wide_updates"] == 2
    assert stream.route_counts()["switches_to_wide"] == 1


@pytest.mark.parametrize("egress", [False, True])
def test_a_print_past_the_exact_domain_is_withheld_and_ranked_exactly(egress):
    series = random_walk(60, seed=43)
    kwargs = {"egress": egress, **({"w": 40} if egress else {})}
    stream = stumpi(series[:30], M, scale=SCALE, **kwargs)
    stream.update(1e12)
    for x in series[30:]:
        stream.update(x)
    counts = stream.route_counts()
    assert counts["withheld_ticks"] == 1 and counts["cold_rows"] > 0
    assert counts["wide_updates"] == 0
    # Every window holding the print has a distance (its variance numerator passes 2^63).
    assert all(np.isfinite(stream.left_P_[30 - M + 1 : 31]))


def test_a_bounded_stream_returns_to_the_narrow_path_once_the_outlier_leaves():
    series = random_walk(400, seed=42)
    stream = stumpi(series[:30], M, scale=SCALE, egress=True, w=40)
    assert stream.route_counts()["switches_to_narrow"] == 0
    stream.update(1.2e9)
    for x in series[30:]:
        stream.update(x)
    counts = stream.route_counts()
    assert counts["switches_to_wide"] == 1
    assert counts["switches_to_narrow"] == 1
    assert counts["narrow_updates"] > 200
