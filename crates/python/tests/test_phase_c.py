"""Discords, patterns and FLOSS: `update` in a loop equals `update_many` on bits, values equal
what the plain streams' own entries give through the same Rust functions, and bad arguments
raise `ValueError`."""

import numpy as np
import pytest
from ladders import (
    BoundedStream,
    DiscordStream,
    FlossStream,
    GrowingStream,
    PatternLibrary,
    PatternStream,
    distance_profile,
)
from ladders import _corrected_arc_curve

from conftest import market_level_walk, random_walk, walk_with_gaps

M = 4
SCALE = 1.0
W = 12

SERIES = {
    "random_walk": lambda: random_walk(150, seed=5),
    "market_level": lambda: market_level_walk(150, seed=6),
    "with_gaps": lambda: walk_with_gaps(150, seed=7, gap_every=13),
}


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64).tolist()


def regime_series():
    rng = np.random.default_rng(11)
    first = np.sin(np.arange(120) * 0.5) * 5 + rng.integers(-1, 2, 120)
    second = np.cumsum(rng.integers(-2, 3, 120)).astype(float)
    return np.concatenate([first, second]).round()


# ---------------------------------------------------------------- discords


def stream_entries(w, series):
    stream = GrowingStream(M, SCALE) if w is None else BoundedStream(M, w, SCALE)
    return stream.update_many(series)


@pytest.mark.parametrize("w", [None, W], ids=["growing", "bounded"])
@pytest.mark.parametrize("name", SERIES)
def test_discord_update_loop_equals_update_many(w, name):
    series = SERIES[name]()
    a = DiscordStream(M, 3, SCALE, w)
    for x in series:
        a.update(float(x))
    b = DiscordStream(M, 3, SCALE, w)
    b.update_many(series[:61])
    b.update_many(series[61:])
    ta, tb = a.top(), b.top()
    assert ta == tb
    assert [bits([d])[0] for _, d in ta] == [bits([d])[0] for _, d in tb]


def test_discord_growing_values_are_the_streams_largest_distances():
    series = random_walk(150, seed=21)
    series[80:84] += 40.0
    indices, distances = stream_entries(None, series)
    d = DiscordStream(M, 1, SCALE)
    d.update_many(series)
    top = d.top()
    assert len(top) == 1
    index, distance = top[0]
    finite = np.where(np.isfinite(distances), distances, -1.0)
    assert bits([distance]) == bits([finite.max()])
    assert distance == finite[index]


def test_discord_lists_at_most_k_nearest_first_and_empty_at_start():
    d = DiscordStream(M, 4, SCALE)
    assert d.top() == []
    d.update_many(random_walk(150, seed=22))
    top = d.top()
    assert 1 <= len(top) <= 4
    distances = [dist for _, dist in top]
    assert distances == sorted(distances, reverse=True)
    assert len({i for i, _ in top}) == len(top)


def test_discord_bounded_forgets_old_windows():
    series = random_walk(200, seed=23)
    series[20:24] += 100.0
    unbounded = DiscordStream(M, 1, SCALE)
    unbounded.update_many(series)
    bounded = DiscordStream(M, 1, SCALE, W)
    bounded.update_many(series)
    assert unbounded.top()[0][0] < 30
    assert bounded.top()[0][0] > 200 - W - M


@pytest.mark.parametrize(
    "args",
    [(2, 1, 1.0), (4, 0, 1.0), (4, -1, 1.0), (4, 1, 0.0), (4, 1, float("nan")), (4, 1, 1.0, 3),
     (4, 2**40, 1.0)],
)
def test_discord_bad_arguments(args):
    with pytest.raises(ValueError):
        DiscordStream(*args)


# ---------------------------------------------------------------- patterns


def test_library_distance_profile_equals_module_function_and_stream_entries():
    series = random_walk(150, seed=31)
    query = series[40 : 40 + M].copy()
    lib = PatternLibrary(M, SCALE)
    qid = lib.register(query)
    assert qid == 0 and len(lib) == 1
    i1, d1 = lib.distance_profile(qid, series)
    i2, d2 = distance_profile(query, series, SCALE)
    assert i1.tolist() == i2.tolist()
    assert bits(d1) == bits(d2)
    assert len(i1) == len(series) - M + 1
    # A window equal to the query is at distance 0.
    assert d1[40] == 0.0 and i1[40] == 40


def test_pattern_stream_equals_distance_profile_at_every_window():
    series = SERIES["random_walk"]()
    lib = PatternLibrary(M, SCALE)
    queries = [series[10 : 10 + M], series[70 : 70 + M] * 2.0, np.array([0.0, 3.0, -2.0, 1.0])]
    for q in queries:
        lib.register(q)
    out = PatternStream(lib).update_many(series)
    assert out.shape == (len(series), 3)
    for k in range(3):
        _, d = lib.distance_profile(k, series)
        assert bits(out[M - 1 :, k]) == bits(d)
    assert np.all(np.isinf(out[: M - 1]))


@pytest.mark.parametrize("name", SERIES)
def test_pattern_stream_update_loop_equals_update_many(name):
    series = SERIES[name]()
    lib = PatternLibrary(M, SCALE)
    lib.register(np.array([0.0, 1.0, 1.0, 3.0]))
    lib.register(np.array([2.0, 0.0, -1.0, 0.0]))
    loop = PatternStream(lib)
    rows = np.array([loop.update(float(x)) for x in series])
    many = PatternStream(lib)
    parts = np.vstack([many.update_many(series[:37]), many.update_many(series[37:])])
    assert rows.shape == parts.shape == (len(series), 2)
    assert bits(rows.ravel()) == bits(parts.ravel())
    assert not np.isnan(rows).any()


def test_pattern_stream_is_independent_of_later_registration():
    lib = PatternLibrary(M, SCALE)
    lib.register(np.array([0.0, 1.0, 2.0, 5.0]))
    stream = PatternStream(lib)
    lib.register(np.array([3.0, 1.0, 2.0, 5.0]))
    assert stream.update(1.0).shape == (1,)
    assert PatternStream(lib).update(1.0).shape == (2,)


def test_pattern_stream_without_queries():
    out = PatternStream(PatternLibrary(M, SCALE)).update_many(random_walk(10, seed=1))
    assert out.shape == (10, 0)


def test_pattern_bad_arguments():
    lib = PatternLibrary(M, SCALE)
    with pytest.raises(ValueError):
        lib.register(np.zeros(M + 1))
    with pytest.raises(ValueError):
        lib.register(np.array([0.0, np.nan, 1.0, 2.0]))
    with pytest.raises(ValueError):
        lib.register(np.array([0.0, np.inf, 1.0, 2.0]))
    with pytest.raises(ValueError):
        lib.distance_profile(0, np.zeros(10))
    lib.register(np.array([0.0, 1.0, 2.0, 4.0]))
    with pytest.raises(ValueError):
        lib.distance_profile(-1, np.zeros(10))
    with pytest.raises(ValueError):
        lib.distance_profile(1, np.zeros(10))
    with pytest.raises(ValueError):
        distance_profile(np.array([]), np.zeros(10), 1.0)
    with pytest.raises(ValueError):
        distance_profile(np.array([1.0, np.nan]), np.zeros(10), 1.0)
    with pytest.raises(ValueError):
        distance_profile(np.array([1.0, 2.0]), np.zeros(10), 0.0)
    with pytest.raises(ValueError):
        PatternLibrary(2, 1.0)
    with pytest.raises(ValueError):
        PatternLibrary(4, -1.0)


# ---------------------------------------------------------------- FLOSS


def left_indices(w, series):
    indices, _ = stream_entries(w, series)
    return indices


@pytest.mark.parametrize("w", [None, 40], ids=["growing", "bounded"])
@pytest.mark.parametrize("series_name", ["regime", "random_walk", "with_gaps"])
def test_floss_curve_equals_recomputed_curve(w, series_name):
    series = regime_series() if series_name == "regime" else SERIES[series_name]()
    width = 100
    f = FlossStream(M, width, SCALE, w, l=6)
    f.update_many(series)
    left = left_indices(w, series)[-width:]
    start = len(series) - M + 1 - len(left)
    expected = _corrected_arc_curve(left, start, M, w, l=6)
    got = f.curve()
    assert len(got) == width
    assert bits(got) == bits(expected)
    assert got.min() < 1.0 or series_name != "regime"


@pytest.mark.parametrize("w", [None, 40], ids=["growing", "bounded"])
def test_floss_update_loop_equals_update_many(w):
    series = regime_series()
    a = FlossStream(M, 80, SCALE, w, l=5)
    for x in series:
        a.update(float(x))
    b = FlossStream(M, 80, SCALE, w, l=5)
    b.update_many(series[:101])
    b.update_many(series[101:])
    assert bits(a.curve()) == bits(b.curve())
    assert a.min_interior() == b.min_interior()


def test_floss_min_interior_matches_curve_and_defaults():
    series = regime_series()
    f = FlossStream(M, 120, SCALE)  # l defaults to m, excl_factor to 5
    assert f.curve().size == 0 and f.min_interior() is None
    f.update_many(series)
    curve = f.curve()
    edge = M * 5
    interior = curve[edge : len(curve) - edge]
    position, score = f.min_interior()
    assert position == edge + int(np.argmin(interior))
    assert bits([score]) == bits([interior.min()])
    explicit = FlossStream(M, 120, SCALE, None, M, 5)
    explicit.update_many(series)
    assert bits(explicit.curve()) == bits(curve)


def test_floss_no_interior_is_none():
    f = FlossStream(M, 10, SCALE, l=5)
    f.update_many(random_walk(60, seed=3))
    assert f.min_interior() is None


@pytest.mark.parametrize(
    "kwargs",
    [dict(m=2, width=10), dict(m=4, width=0), dict(m=4, width=-3), dict(m=4, width=2**40),
     dict(m=4, width=10, scale=0.0), dict(m=4, width=10, w=3), dict(m=4, width=10, l=-1),
     dict(m=4, width=10, excl_factor=-1)],
)
def test_floss_bad_arguments(kwargs):
    kwargs.setdefault("scale", 1.0)
    with pytest.raises(ValueError):
        FlossStream(**kwargs)
