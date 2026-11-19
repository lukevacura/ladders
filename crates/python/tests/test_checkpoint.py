"""Checkpoint/restore: a restored stream continues bit for bit like one that never stopped,
and bad bytes raise ValueError, never anything else."""

import pickle

import numpy as np
import pytest
from ladders import BoundedStream, GrowingStream, stumpi

from conftest import assert_mostly_finite, reseal, market_level_walk, random_walk, walk_with_gaps

M = 4
W = 12
N = 160


def make_growing():
    return GrowingStream(M, 1.0)


def make_bounded():
    return BoundedStream(M, W, 1.0)


def feed(stream, xs):
    out = []
    for x in xs:
        r = stream.update(float(x))
        if r is not None:
            out.append((r[0], np.float64(r[1]).view(np.uint64).item()))
    return out


SERIES = [
    lambda seed: random_walk(N, seed),
    lambda seed: walk_with_gaps(N, seed, leading_gap=3),
    lambda seed: market_level_walk(N, seed),
]


@pytest.mark.parametrize("make", [make_growing, make_bounded])
@pytest.mark.parametrize("series", SERIES)
@pytest.mark.parametrize("seed", range(4))
def test_resume_is_bitwise(make, series, seed):
    xs = series(seed)
    cut = int(np.random.default_rng(seed + 100).integers(0, N))
    whole = make()
    expected = feed(whole, xs)
    first = make()
    head = feed(first, xs[:cut])
    restored = type(first).restore(first.checkpoint())
    tail = feed(restored, xs[cut:])
    assert head + tail == expected
    assert sum(1 for e in expected if e[0] != -1) > len(expected) // 4
    # The restored stream's own checkpoint equals the uninterrupted one's at the end.
    assert restored.checkpoint() == whole.checkpoint()


@pytest.mark.parametrize("make", [make_growing, make_bounded])
@pytest.mark.parametrize("cut", [30, 45, 100])
def test_resume_after_an_outlier_switched_the_stream_wide(make, cut):
    xs = random_walk(N, 5)
    xs[40:43] += 3e9  # past any one narrow anchor at M = 4; gone from the bounded window by 100.
    whole = make()
    expected = feed(whole, xs)
    first = make()
    head = feed(first, xs[:cut])
    restored = type(first).restore(first.checkpoint())
    assert head + feed(restored, xs[cut:]) == expected
    assert restored.checkpoint() == whole.checkpoint()


@pytest.mark.parametrize("make", [make_growing, make_bounded])
def test_restore_ignores_nothing_it_was_given(make):
    stream = make()
    feed(stream, random_walk(50, 1))
    other = type(stream).restore(stream.checkpoint())
    # Continuing both with different data gives different results: restore carries state.
    assert feed(other, random_walk(30, 2)) != feed(make(), random_walk(30, 2))


def test_wrong_stream_kind_is_rejected():
    g, b = make_growing(), make_bounded()
    feed(g, random_walk(40, 1))
    feed(b, random_walk(40, 1))
    with pytest.raises(ValueError):
        BoundedStream.restore(g.checkpoint())
    with pytest.raises(ValueError):
        GrowingStream.restore(b.checkpoint())


@pytest.mark.parametrize("cls", [GrowingStream, BoundedStream, stumpi])
def test_empty_and_garbled_bytes_raise_value_error(cls):
    for bad in (b"", b"\x00", b"not a checkpoint at all", b"MPCK", b"MPSI\x01\x00"):
        with pytest.raises(ValueError):
            cls.restore(bad)


def make_stumpi(egress):
    warm = random_walk(30, 3)
    if egress:
        return stumpi(warm, M, scale=1.0, egress=True, w=W)
    return stumpi(warm, M, scale=1.0, egress=False)


@pytest.mark.parametrize(
    "make", [make_growing, make_bounded, lambda: make_stumpi(True), lambda: make_stumpi(False)]
)
def test_truncated_and_flipped_bytes_raise_value_error(make):
    stream = make()
    feed_fn = stream.update
    for x in random_walk(40, 5):
        feed_fn(float(x))
    data = stream.checkpoint()
    cls = type(stream)
    for cut in range(len(data)):
        with pytest.raises(ValueError):
            cls.restore(data[:cut])
    with pytest.raises(ValueError):
        cls.restore(data + b"\x00")
    # Every byte position, history and checksum included.
    for pos in range(len(data)):
        for bit in range(8):
            flipped = bytearray(data)
            flipped[pos] ^= 1 << bit
            with pytest.raises(ValueError):
                cls.restore(bytes(flipped))


def test_old_version_one_bytes_raise_bad_version():
    data = bytearray(make_stumpi(True).checkpoint())
    data[4:6] = (1).to_bytes(2, "little")
    with pytest.raises(ValueError, match="BadVersion"):
        stumpi.restore(reseal(data))


@pytest.mark.parametrize("egress", [True, False])
@pytest.mark.parametrize("seed", range(4))
def test_stumpi_resume_restores_history(egress, seed):
    xs = walk_with_gaps(N, seed + 10, leading_gap=2)
    cut = int(np.random.default_rng(seed).integers(1, N))
    warm = random_walk(25, seed)

    def build():
        if egress:
            return stumpi(warm, M, scale=1.0, egress=True, w=W)
        return stumpi(warm, M, scale=1.0, egress=False)

    whole = build()
    for x in xs:
        whole.update(float(x))
    part = build()
    for x in xs[:cut]:
        part.update(float(x))
    restored = stumpi.restore(part.checkpoint())
    # History is back immediately, before any further update.
    assert restored.left_P_.view(np.uint64).tolist() == part.left_P_.view(np.uint64).tolist()
    assert restored.left_I_.tolist() == part.left_I_.tolist()
    assert len(restored.left_P_) > 0
    for x in xs[cut:]:
        restored.update(float(x))
    assert restored.left_P_.view(np.uint64).tolist() == whole.left_P_.view(np.uint64).tolist()
    assert restored.left_I_.tolist() == whole.left_I_.tolist()
    assert_mostly_finite(whole.left_I_, M, 0.4)
    assert restored.checkpoint() == whole.checkpoint()


def test_stumpi_history_changes_checkpoint():
    a = make_stumpi(True)
    b = stumpi(random_walk(30, 3)[:5], M, scale=1.0, egress=True, w=W)
    assert a.checkpoint() != b.checkpoint()


@pytest.mark.parametrize("make", [make_growing, make_bounded, lambda: make_stumpi(True)])
def test_pickle_round_trip(make):
    stream = make()
    for x in random_walk(40, 7):
        stream.update(float(x))
    clone = pickle.loads(pickle.dumps(stream))
    assert type(clone) is type(stream)
    assert clone.checkpoint() == stream.checkpoint()
