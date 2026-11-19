"""`aampi`: the causal left profile under the Euclidean distance on the raw values. Its
`left_P_`/`left_I_` equal an exact integer brute force on bits, STUMPY's own `aampi` on a series
with no ties, and a checkpoint resumes it bit for bit."""

import math
import pickle

import numpy as np
import pytest
from ladders import aampi

from conftest import random_walk, reseal, walk_with_gaps

M = 4
W = 14
SCALE = 1.0
ENGINES = {"growing": {"egress": False}, "bounded": {"egress": True, "w": W}}


def tie_free(n, seed):
    """Continuous values: no two windows are exact repeats, so no distances tie."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=n) * 5 + np.cumsum(rng.normal(size=n))


def brute_force(series, m, scale, w=None):
    """Exact integer left profile on the quantized values: ties to the smaller index, a window
    with a missing sample neither has a neighbor nor is one. With `w`, only windows inside the
    last `w` samples up to the row's own window are searched."""
    ticks = [None if not math.isfinite(x) else round(x / scale) for x in series]
    rows = len(series) - m + 1
    exclusion = -(-m // 4)
    whole = [all(t is not None for t in ticks[i : i + m]) for i in range(rows)]
    index = np.full(rows, -1, dtype=np.int64)
    distance = np.full(rows, np.inf)
    for i in range(exclusion + 1, rows):
        if not whole[i]:
            continue
        oldest = 0 if w is None else max(0, i + m - w)
        best = None
        for j in range(oldest, i - exclusion):
            if not whole[j]:
                continue
            d = sum((a - b) ** 2 for a, b in zip(ticks[i : i + m], ticks[j : j + m]))
            if best is None or d < best[0]:
                best = (d, j)
        if best is not None:
            index[i] = best[1]
            distance[i] = math.sqrt(float(best[0])) * scale
    return index, distance


def u64(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def run(series, warm, engine, m=M, scale=SCALE):
    stream = aampi(series[:warm], m, scale=scale, **ENGINES[engine])
    for x in series[warm:]:
        stream.update(float(x))
    return stream


SERIES = {
    "random_walk": lambda: random_walk(70, seed=1),
    "with_gaps": lambda: walk_with_gaps(70, seed=2, gap_every=9, leading_gap=3),
    "repeats": lambda: np.tile(np.array([0.0, 1.0, 3.0, 1.0, 0.0, 2.0, 2.0]), 10),
    "flat": lambda: np.full(40, 7.0),
    "levels": lambda: random_walk(70, seed=3) + np.where(np.arange(70) // 20 % 2 == 1, 2.0**40, 0.0),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", SERIES)
def test_matches_the_exact_integer_brute_force_on_bits(name, engine):
    series = SERIES[name]()
    stream = run(series, warm=30, engine=engine)
    index, distance = brute_force(series, M, SCALE, W if engine == "bounded" else None)
    assert np.array_equal(stream.left_I_, index)
    assert np.array_equal(u64(stream.left_P_), u64(distance))
    assert (index != -1).sum() > len(index) // 4


@pytest.mark.parametrize("engine", ENGINES)
def test_the_scale_sets_the_units_of_the_distance(engine):
    series = tie_free(60, seed=4) * 0.01
    stream = run(series, warm=20, engine=engine, scale=0.01)
    index, distance = brute_force(series, M, 0.01, W if engine == "bounded" else None)
    assert np.array_equal(stream.left_I_, index)
    assert np.array_equal(u64(stream.left_P_), u64(distance))


@pytest.mark.parametrize("seed", range(3))
def test_equals_stumpys_aampi_without_egress_on_a_series_without_ties(seed):
    stumpy = pytest.importorskip("stumpy")
    series = tie_free(120, seed)
    theirs = stumpy.aampi(series[:60].copy(), M, egress=False)
    ours = aampi(series[:60], M, scale=1e-9, egress=False)
    for x in series[60:]:
        theirs.update(x)
        ours.update(float(x))
    assert np.array_equal(ours.left_I_, theirs.left_I_)
    assert np.allclose(ours.left_P_, theirs.left_P_, rtol=1e-9, atol=0, equal_nan=True)


@pytest.mark.parametrize("seed", range(3))
def test_equals_stumpys_aampi_with_egress_on_the_newest_window_of_each_update(seed):
    stumpy = pytest.importorskip("stumpy")
    series = tie_free(140, seed)
    w = 50
    theirs = stumpy.aampi(series[:w].copy(), M, egress=True)
    ours = aampi(series[:w], M, scale=1e-9, egress=True, w=w)
    assert np.array_equal(ours.left_I_, theirs.left_I_)
    for x in series[w:]:
        theirs.update(x)
        ours.update(float(x))
        assert ours.left_I_[-1] == theirs.left_I_[-1]
        assert math.isclose(ours.left_P_[-1], theirs.left_P_[-1], rel_tol=1e-9)


def test_has_no_p_or_i_attributes_and_update_returns_none():
    stream = aampi(random_walk(20, seed=5), M, scale=SCALE, egress=False)
    assert not hasattr(stream, "P_")
    assert not hasattr(stream, "I_")
    assert stream.update(1.0) is None


@pytest.mark.parametrize("m", [2, 0, -1])
def test_invalid_m_raises_value_error(m):
    with pytest.raises(ValueError):
        aampi(random_walk(20, seed=6), m, scale=SCALE, egress=False)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scale_raises_value_error(scale):
    with pytest.raises(ValueError):
        aampi(random_walk(20, seed=7), M, scale=scale, egress=False)


def test_egress_true_without_w_keeps_the_length_of_the_history_and_egress_false_rejects_w():
    warm = random_walk(20, seed=8)
    default = aampi(warm, M, scale=SCALE, egress=True)
    explicit = aampi(warm, M, scale=SCALE, egress=True, w=len(warm))
    for x in random_walk(15, seed=9):
        default.update(x)
        explicit.update(x)
    assert np.array_equal(default.left_I_, explicit.left_I_)
    assert np.array_equal(default.left_P_.view(np.uint64), explicit.left_P_.view(np.uint64))
    with pytest.raises(ValueError):
        aampi(warm, M, scale=SCALE, egress=False, w=W)
    with pytest.raises(ValueError):
        aampi(warm, M, scale=SCALE, egress=True, w=M - 1)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_range_past_the_narrow_bound_moves_the_stream_to_the_wide_route(engine):
    stream = aampi(random_walk(30, seed=8), M, scale=SCALE, **ENGINES[engine])
    before = stream.route_counts()
    assert before["narrow_updates"] == 30 and before["wide_updates"] == 0
    stream.update(2.0**40)
    after = stream.route_counts()
    assert after["wide_updates"] == 1 and after["switches_to_wide"] == 1
    stream.update(2.0**40)
    assert stream.route_counts()["wide_updates"] == 2
    assert stream.route_counts()["switches_to_wide"] == 1


def test_a_bounded_stream_returns_to_the_narrow_route_once_the_far_sample_leaves():
    series = random_walk(200, seed=9)
    stream = aampi(series[:30], M, scale=SCALE, egress=True, w=40)
    stream.update(2.0**40)
    for x in series[30:]:
        stream.update(x)
    counts = stream.route_counts()
    assert counts["switches_to_wide"] == 1 and counts["switches_to_narrow"] == 1
    assert counts["narrow_updates"] > 150


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["random_walk", "with_gaps", "levels"])
@pytest.mark.parametrize("cut", [0, 5, 31, 69])
def test_resume_is_bitwise(name, cut, engine):
    series = SERIES[name]()
    whole = run(series, warm=10, engine=engine)
    first = aampi(series[:cut], M, scale=SCALE, **ENGINES[engine])
    restored = aampi.restore(first.checkpoint())
    for x in series[cut:]:
        restored.update(float(x))
    assert np.array_equal(restored.left_I_, whole.left_I_)
    assert np.array_equal(u64(restored.left_P_), u64(whole.left_P_))
    assert restored.checkpoint() == whole.checkpoint()
    assert pickle.loads(pickle.dumps(whole)).checkpoint() == whole.checkpoint()


@pytest.mark.parametrize("engine", ENGINES)
def test_garbled_and_truncated_bytes_raise_value_error(engine):
    stream = run(random_walk(50, seed=9), warm=30, engine=engine)
    data = stream.checkpoint()
    for bad in (b"", b"\x00", b"not a checkpoint at all", b"MPAI", b"MPAI\x01\x00"):
        with pytest.raises(ValueError):
            aampi.restore(bad)
    for cut in range(len(data)):
        with pytest.raises(ValueError):
            aampi.restore(data[:cut])
    with pytest.raises(ValueError):
        aampi.restore(data + b"\x00")
    for pos in range(len(data)):
        for bit in range(8):
            flipped = bytearray(data)
            flipped[pos] ^= 1 << bit
            with pytest.raises(ValueError):
                aampi.restore(bytes(flipped))


def test_old_version_one_bytes_raise_bad_version():
    data = bytearray(run(random_walk(50, seed=9), warm=30, engine="growing").checkpoint())
    data[4:6] = (1).to_bytes(2, "little")
    with pytest.raises(ValueError, match="BadVersion"):
        aampi.restore(reseal(data))
