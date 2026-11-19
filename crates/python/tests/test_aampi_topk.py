"""`aampi(..., k=k)`: each window keeps its `k` nearest earlier neighbors by exact squared distance,
nearest first, ties to the smaller index, padded with `inf` and `-1` (`left_topk_P_`,
`left_topk_I_`, shape `(windows, k)`), while `left_P_` and `left_I_` stay the nearest one. Both
egress modes equal an exact integer brute force on bits (ties, gaps and a window of only the last
`w` samples included), a checkpoint carries `k`, and on tie-free data the newest row equals
STUMPY's `P_[-1]`/`I_[-1]`."""

import math
import pickle

import numpy as np
import pytest
import stumpy
from ladders import aampi

from conftest import flat_run_series, random_walk, reseal, walk_with_gaps

M = 5
KS = [2, 3, 8, 200]


def u64(x):
    return np.asarray(x).view(np.uint64)


GRID = 2.0**-10


def tie_free(n, seed):
    """A walk on a dyadic grid, so quantizing at `GRID` is lossless; its steps are wide enough
    that no two distances tie."""
    steps = np.random.default_rng(seed).integers(-1000, 1001, size=n)
    return np.cumsum(steps) * GRID


def brute_force(series, m, scale, k, w=None):
    """Each window's `k` nearest earlier windows by exact integer squared distance on the
    quantized values, ties to the smaller index; a window with a missing sample has no neighbor
    and is never one. With `w`, only windows starting inside the last `w` samples up to the row's
    own window are searched. Returns `(P, I)` padded with `inf` and `-1`."""
    ticks = [None if not math.isfinite(x) else round(x / scale) for x in series]
    rows = len(series) - m + 1
    exclusion = -(-m // 4)
    whole = [all(t is not None for t in ticks[i : i + m]) for i in range(rows)]
    p, index = np.full((rows, k), np.inf), np.full((rows, k), -1, dtype=np.int64)
    for i in range(rows):
        if not whole[i]:
            continue
        oldest = 0 if w is None else max(0, i + m - w)
        found = sorted(
            (sum((a - b) ** 2 for a, b in zip(ticks[i : i + m], ticks[j : j + m])), j)
            for j in range(oldest, i - exclusion)
            if whole[j]
        )[:k]
        for slot, (d, j) in enumerate(found):
            p[i, slot], index[i, slot] = math.sqrt(float(d)) * scale, j
    return p, index


SERIES = {
    "random_walk": lambda: random_walk(90, seed=80),
    "constant": lambda: np.full(70, 4.0),
    "period_4": lambda: np.tile([0.0, 2.0, 1.0, 3.0], 20),
    "period_7": lambda: (np.arange(90) % 7).astype(float),
    "few_levels": lambda: np.random.default_rng(81).integers(0, 3, 90).astype(float),
    "flat_run": lambda: flat_run_series(90, seed=82, run_at=30, run_len=25),
    "with_gaps": lambda: walk_with_gaps(90, seed=83, gap_every=11, leading_gap=3),
    "wide": lambda: random_walk(90, seed=84) * 2.0**40,
}


def grown(series, k, split=30, scale=1.0, **kwargs):
    stream = aampi(series[:split], M, egress=False, k=k, scale=scale, **kwargs)
    for x in series[split:]:
        stream.update(float(x))
    return stream


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", SERIES)
def test_the_growing_stream_equals_the_brute_force_on_bits(name, k):
    series = SERIES[name]()
    stream = grown(series, k)
    want_p, want_i = brute_force(series, M, 1.0, k)

    assert stream.left_topk_P_.shape == stream.left_topk_I_.shape == want_p.shape
    assert stream.left_topk_P_.dtype == np.float64 and stream.left_topk_I_.dtype == np.int64
    assert np.array_equal(u64(stream.left_topk_P_), u64(want_p))
    assert np.array_equal(stream.left_topk_I_, want_i)
    assert np.array_equal(stream.left_I_, want_i[:, 0])
    assert np.array_equal(u64(stream.left_P_), u64(want_p[:, 0]))
    # Not vacuous: past the exclusion zone some row has kept every one of its neighbors.
    assert (want_i[:, : min(k, 2)] >= 0).all(axis=1).any()


@pytest.mark.parametrize("k", [2, 3, 8])
@pytest.mark.parametrize("name", SERIES)
@pytest.mark.parametrize("w", [M + 2, 3 * M, 40])
def test_the_bounded_stream_equals_the_brute_force_of_its_last_w_samples(name, k, w):
    series = SERIES[name]()
    stream = aampi(series[:30], M, egress=True, k=k, scale=1.0, w=w)
    for x in series[30:]:
        stream.update(float(x))
    want_p, want_i = brute_force(series, M, 1.0, k, w=w)
    assert stream.left_topk_P_.shape == want_p.shape
    assert np.array_equal(stream.left_topk_I_, want_i)
    assert np.array_equal(u64(stream.left_topk_P_), u64(want_p))


def test_k_of_one_keeps_the_shape_it_had():
    series = random_walk(60, seed=85)
    one = grown(series, 1)
    default = aampi(series[:30], M, egress=False, scale=1.0)
    for x in series[30:]:
        default.update(float(x))
    assert np.array_equal(one.left_I_, default.left_I_)
    assert np.array_equal(u64(one.left_P_), u64(default.left_P_))
    assert one.left_topk_P_.shape == (len(series) - M + 1, 1)
    assert np.array_equal(u64(one.left_topk_P_[:, 0]), u64(default.left_P_))
    assert one.checkpoint() == default.checkpoint()


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [0, -1, -50])
def test_a_k_below_one_is_a_value_error(egress, k):
    with pytest.raises(ValueError):
        aampi(random_walk(20, seed=86), M, egress=egress, k=k, scale=1.0)


def test_a_k_the_arrays_cannot_hold_is_a_value_error_and_a_k_past_the_windows_is_padding():
    series = random_walk(40, seed=87)
    huge = aampi(series, M, egress=False, k=2**40, scale=1.0)
    huge.update(1.0)
    assert len(huge.left_I_) == len(series) - M + 2
    with pytest.raises(ValueError, match="reserved"):
        huge.left_topk_P_
    stream = aampi(series, M, egress=False, k=300, scale=1.0)
    p, i = stream.left_topk_P_, stream.left_topk_I_
    assert p.shape == i.shape == (len(series) - M + 1, 300)
    assert np.array_equal(np.isinf(p), i == -1)
    assert (i[-1] >= 0).sum() == len(series) - M - 2


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [2, 3, 8])
def test_every_input_form_gives_the_float64_bits(egress, k):
    series = np.round(np.cumsum(np.random.default_rng(88).normal(size=60)) * 4)
    kwargs = {"w": 25} if egress else {}
    want = aampi(series[:30], M, egress=egress, k=k, scale=1.0, **kwargs)
    for x in series[30:]:
        want.update(float(x))
    forms = {
        "list": lambda a: a.tolist(),
        "int64": lambda a: a.astype(np.int64),
        "int32": lambda a: a.astype(np.int32),
        "float32": lambda a: a.astype(np.float32),
    }
    for name, form in forms.items():
        got = aampi(form(series[:30]), M, egress=egress, k=k, scale=1.0, **kwargs)
        for x in form(series[30:]):
            got.update(float(x))
        assert np.array_equal(u64(got.left_topk_P_), u64(want.left_topk_P_)), name
        assert np.array_equal(got.left_topk_I_, want.left_topk_I_), name


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [2, 3])
def test_a_checkpoint_and_a_pickle_continue_the_stream_with_its_k(egress, k):
    series = SERIES["few_levels"]()
    kwargs = {"w": 30} if egress else {}
    stream = aampi(series[:35], M, egress=egress, k=k, scale=1.0, **kwargs)
    for x in series[35:60]:
        stream.update(float(x))
    for restored in (aampi.restore(stream.checkpoint()), pickle.loads(pickle.dumps(stream))):
        assert restored.left_topk_P_.shape == stream.left_topk_P_.shape
        for x in series[60:]:
            restored.update(float(x))
        reference = aampi(series[:35], M, egress=egress, k=k, scale=1.0, **kwargs)
        for x in series[35:]:
            reference.update(float(x))
        assert np.array_equal(u64(restored.left_topk_P_), u64(reference.left_topk_P_))
        assert np.array_equal(restored.left_topk_I_, reference.left_topk_I_)
        assert restored.checkpoint() == reference.checkpoint()


def test_a_k_one_checkpoint_is_the_old_format_and_a_damaged_k_checkpoint_is_refused():
    series = random_walk(60, seed=89)
    old = grown(series, 1).checkpoint()
    assert old[4:6] == (2).to_bytes(2, "little")
    assert aampi.restore(old).left_topk_P_.shape[1] == 1
    new = grown(series, 3).checkpoint()
    assert new[4:6] == (3).to_bytes(2, "little")
    assert bytes(new[:4]) == b"MPAI"
    for cut in (len(new) - 9, len(new) // 2, 10):
        with pytest.raises(ValueError):
            aampi.restore(new[:cut])
    flipped = bytearray(new)
    flipped[len(flipped) // 2] ^= 1
    with pytest.raises(ValueError):
        aampi.restore(bytes(flipped))
    # A version-3 header over a one-neighbor engine, with the checksum made right again, is
    # refused as inconsistent, not misread.
    relabeled = bytearray(old)
    relabeled[4:6] = (3).to_bytes(2, "little")
    with pytest.raises(ValueError, match="Inconsistent"):
        aampi.restore(reseal(bytes(relabeled)))


@pytest.mark.parametrize("k", [2, 3, 8])
@pytest.mark.parametrize("seed", [90, 91, 92])
def test_the_newest_row_equals_stumpys_after_every_update(k, seed):
    """STUMPY revises older rows with later windows, so only the row an update completes (which
    has only left neighbors) is comparable; its neighbors match exactly and its distances up to
    STUMPY's own float noise."""
    series = tie_free(120, seed)
    theirs = stumpy.aampi(series[:40].copy(), M, egress=False, k=k)
    ours = aampi(series[:40], M, egress=False, k=k, scale=GRID)
    assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
    for x in series[40:]:
        theirs.update(x)
        ours.update(float(x))
        assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
        assert np.allclose(ours.left_topk_P_[-1], theirs.P_[-1], rtol=1e-12, atol=1e-9)
        assert np.array_equal(np.isinf(ours.left_topk_P_[-1]), np.isinf(theirs.P_[-1]))
        assert ours.left_I_[-1] == theirs.left_I_[-1]


@pytest.mark.parametrize("k", [2, 3])
def test_the_bounded_newest_row_equals_stumpys_egress_stream(k):
    n = 60
    series = tie_free(n + 50, seed=93)
    theirs = stumpy.aampi(series[:n].copy(), M, egress=True, k=k)
    ours = aampi(series[:n], M, egress=True, k=k, scale=GRID)
    for x in series[n:]:
        theirs.update(x)
        ours.update(float(x))
        # STUMPY's egress indices are already positions in the whole stream.
        assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
        assert np.allclose(ours.left_topk_P_[-1], theirs.P_[-1], rtol=1e-12, atol=1e-9)
