"""`stumpi(..., look_ahead=True)` and `aampi(..., look_ahead=True)`: the stream also keeps each held
window's nearest later neighbors, so `right_*` and `two_sided_*` read, with no recompute, the
profile the batch two-sided functions give over the samples held (bits, ties and gaps included).
Without it the stream does not look ahead, and a checkpoint carries the lists."""

import pickle

import numpy as np
import pytest
from ladders import aamp, aampi, matrix_profile, stumpi

from conftest import flat_run_series, random_walk, walk_with_gaps

M = 5
KS = [1, 2, 3]

SERIES = {
    "random_walk": lambda: random_walk(90, seed=70),
    "constant": lambda: np.full(60, 4.0),
    "period_4": lambda: np.tile([0.0, 2.0, 1.0, 3.0], 15),
    "few_levels": lambda: np.random.default_rng(71).integers(0, 3, 80).astype(float),
    "flat_run": lambda: flat_run_series(80, seed=72, run_at=30, run_len=25),
    "with_gaps": lambda: walk_with_gaps(80, seed=73, gap_every=11, leading_gap=3),
}


def u64(x):
    return np.asarray(x).view(np.uint64)


def grown(make, series, k, split=30):
    stream = make(series[:split], M, egress=False, k=k, scale=1.0, look_ahead=True)
    for x in series[split:]:
        stream.update(float(x))
    return stream


def distances(batch_p, nonnorm):
    return batch_p


@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("k", KS)
def test_the_growing_stream_equals_the_batch_two_sided_profile(name, k):
    series = SERIES[name]()
    stream = grown(stumpi, series, k)
    P, I, _, right_I = matrix_profile(series, M, 1.0, k=k)
    P, I = P.reshape(len(P), k), I.reshape(len(I), k)
    assert np.array_equal(u64(stream.two_sided_topk_P_), u64(P))
    assert np.array_equal(stream.two_sided_topk_I_, I)
    if k == 1:
        assert np.array_equal(stream.right_I_, right_I)
        assert np.array_equal(u64(stream.two_sided_P_), u64(P[:, 0]))
        assert np.array_equal(stream.two_sided_I_, I[:, 0])
    assert stream.right_topk_P_.shape == (len(P), k)


@pytest.mark.parametrize("name", SERIES)
def test_the_bounded_stream_holds_the_right_side_of_its_last_samples(name):
    series = SERIES[name]()
    w = 40
    stream = stumpi(series[:w], M, egress=True, scale=1.0, look_ahead=True)
    for x in series[w:]:
        stream.update(float(x))
    held = series[-w:]
    _, _, _, right_I = matrix_profile(held, M, 1.0)
    shift = len(series) - w
    expected = np.where(right_I < 0, -1, right_I + shift)
    assert np.array_equal(stream.right_I_, expected)
    assert len(stream.two_sided_I_) == w - M + 1


@pytest.mark.parametrize("name", ["random_walk", "period_4", "with_gaps"])
def test_aampi_equals_the_batch_two_sided_profile(name):
    series = SERIES[name]()
    stream = grown(aampi, series, 2)
    P, I, _, _ = aamp(series, M, k=2, scale=1.0)
    assert np.array_equal(u64(stream.two_sided_topk_P_), u64(P.reshape(len(P), 2)))
    assert np.array_equal(stream.two_sided_topk_I_, I.reshape(len(I), 2))


@pytest.mark.parametrize("make", [stumpi, aampi])
@pytest.mark.parametrize("egress", [False, True])
def test_a_checkpoint_carries_the_right_side(make, egress):
    series = random_walk(90, seed=74)
    stream = make(series[:30], M, egress=egress, k=2, scale=1.0, look_ahead=True)
    for x in series[30:50]:
        stream.update(float(x))
    restored = pickle.loads(pickle.dumps(stream))
    for x in series[50:]:
        stream.update(float(x))
        restored.update(float(x))
    assert restored.checkpoint() == stream.checkpoint()
    assert np.array_equal(u64(restored.two_sided_topk_P_), u64(stream.two_sided_topk_P_))
    assert np.array_equal(restored.right_topk_I_, stream.right_topk_I_)


@pytest.mark.parametrize("make", [stumpi, aampi])
def test_without_look_ahead_the_stream_does_not_look_right(make):
    stream = make(random_walk(60, seed=75), M, scale=1.0)
    with pytest.raises(ValueError, match="look_ahead"):
        stream.right_P_
    with pytest.raises(ValueError, match="look_ahead"):
        stream.two_sided_topk_I_


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.float32])
def test_integer_and_single_precision_samples(dtype):
    series = (random_walk(70, seed=76) * 4).astype(dtype)
    stream = stumpi(series[:30], M, egress=False, scale=1.0, look_ahead=True)
    for x in series[30:]:
        stream.update(float(x))
    _, _, _, right_I = matrix_profile(series.astype(np.float64), M, 1.0)
    assert np.array_equal(stream.right_I_, right_I)
