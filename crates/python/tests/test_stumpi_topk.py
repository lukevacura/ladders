"""`stumpi(..., k=k)`: each window keeps its `k` nearest earlier neighbors, nearest first, padded with
`inf` and `-1` (`left_topk_P_`, `left_topk_I_`, shape `(windows, k)`), while `left_P_` and `left_I_`
stay the nearest one. The growing stream equals the batch left top-k profile on bits (ties and gaps
included), the bounded stream equals that batch profile run on only its last `w` samples, a
checkpoint carries `k`, and on tie-free data the newest row equals STUMPY's `P_[-1]`/`I_[-1]`."""

import pickle

import numpy as np
import pytest
import stumpy
from ladders import left_profile, stumpi

from conftest import flat_run_series, random_walk, walk_with_gaps

M = 5
KS = [2, 3, 8, 200]


def u64(x):
    return np.asarray(x).view(np.uint64)


GRID = 2.0**-10


def tie_free(n, seed):
    """A walk on a dyadic grid, so quantizing at `GRID` is lossless and only STUMPY's own float
    noise separates the two; its steps are wide enough that no two distances tie."""
    steps = np.random.default_rng(seed).integers(-1000, 1001, size=n)
    return np.cumsum(steps) * GRID


SERIES = {
    "random_walk": lambda: random_walk(90, seed=60),
    "constant": lambda: np.full(70, 4.0),
    "period_4": lambda: np.tile([0.0, 2.0, 1.0, 3.0], 20),
    "period_7": lambda: (np.arange(90) % 7).astype(float),
    "few_levels": lambda: np.random.default_rng(61).integers(0, 3, 90).astype(float),
    "flat_run": lambda: flat_run_series(90, seed=62, run_at=30, run_len=25),
    "with_gaps": lambda: walk_with_gaps(90, seed=63, gap_every=11, leading_gap=3),
}


def grown(series, k, split=30, **kwargs):
    stream = stumpi(series[:split], M, egress=False, k=k, scale=1.0, **kwargs)
    for x in series[split:]:
        stream.update(float(x))
    return stream


@pytest.mark.parametrize("k", KS)
@pytest.mark.parametrize("name", SERIES)
def test_the_growing_stream_equals_the_batch_left_top_k_profile_on_bits(name, k):
    series = SERIES[name]()
    stream = grown(series, k)
    want_p, want_i, _ = left_profile(series, M, scale=1.0, k=k)

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
def test_the_bounded_stream_equals_the_batch_profile_of_its_last_w_samples(name, k, w):
    series = SERIES[name]()
    stream = stumpi(series[:30], M, egress=True, k=k, scale=1.0, w=w)
    for x in series[30:]:
        stream.update(float(x))
    got_p, got_i = stream.left_topk_P_, stream.left_topk_I_
    assert got_p.shape == (len(series) - M + 1, k)

    for row in range(len(series) - M + 1):
        start = max(0, row + M - w)
        want_p, want_i, _ = left_profile(series[start : row + M], M, scale=1.0, k=k)
        want_i = np.where(want_i[-1] >= 0, want_i[-1] + start, -1)
        assert np.array_equal(got_i[row], want_i), (name, row)
        assert np.array_equal(u64(got_p[row]), u64(want_p[-1])), (name, row)


def test_k_of_one_keeps_the_shape_it_had():
    series = random_walk(60, seed=64)
    one = grown(series, 1)
    default = stumpi(series[:30], M, egress=False, scale=1.0)
    for x in series[30:]:
        default.update(float(x))
    assert np.array_equal(one.left_I_, default.left_I_)
    assert np.array_equal(u64(one.left_P_), u64(default.left_P_))
    assert one.left_topk_P_.shape == (len(series) - M + 1, 1)
    assert np.array_equal(u64(one.left_topk_P_[:, 0]), u64(default.left_P_))
    assert one.checkpoint() == default.checkpoint()


@pytest.mark.parametrize("k", [0, -1, -50])
def test_a_k_below_one_is_a_value_error(k):
    with pytest.raises(ValueError):
        stumpi(random_walk(20, seed=65), M, k=k, scale=1.0)


def test_a_k_the_arrays_cannot_hold_is_a_value_error_and_a_k_past_the_windows_is_padding():
    series = random_walk(40, seed=66)
    huge = stumpi(series, M, egress=False, k=2**40, scale=1.0)
    huge.update(1.0)
    assert len(huge.left_I_) == len(series) - M + 2
    with pytest.raises(ValueError, match="reserved"):
        huge.left_topk_P_
    stream = stumpi(series, M, egress=False, k=300, scale=1.0)
    p, i = stream.left_topk_P_, stream.left_topk_I_
    assert p.shape == i.shape == (len(series) - M + 1, 300)
    assert np.array_equal(np.isinf(p), i == -1)
    assert (i[-1] >= 0).sum() == len(series) - M - 2


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [2, 3, 8])
def test_every_input_form_gives_the_float64_bits(egress, k):
    series = np.round(np.cumsum(np.random.default_rng(68).normal(size=60)) * 4)
    kwargs = {"w": 25} if egress else {}
    want = stumpi(series[:30], M, egress=egress, k=k, scale=1.0, **kwargs)
    for x in series[30:]:
        want.update(float(x))
    forms = {
        "list": lambda a: a.tolist(),
        "int64": lambda a: a.astype(np.int64),
        "int32": lambda a: a.astype(np.int32),
        "float32": lambda a: a.astype(np.float32),
    }
    for name, form in forms.items():
        got = stumpi(form(series[:30]), M, egress=egress, k=k, scale=1.0, **kwargs)
        for x in form(series[30:]):
            got.update(float(x))
        assert np.array_equal(u64(got.left_topk_P_), u64(want.left_topk_P_)), name
        assert np.array_equal(got.left_topk_I_, want.left_topk_I_), name


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [2, 3])
def test_a_checkpoint_and_a_pickle_continue_the_stream_with_its_k(egress, k):
    series = SERIES["few_levels"]()
    kwargs = {"w": 30} if egress else {}
    stream = stumpi(series[:35], M, egress=egress, k=k, scale=1.0, **kwargs)
    for x in series[35:60]:
        stream.update(float(x))
    for restored in (stumpi.restore(stream.checkpoint()), pickle.loads(pickle.dumps(stream))):
        assert restored.left_topk_P_.shape == stream.left_topk_P_.shape
        for x in series[60:]:
            restored.update(float(x))
        reference = stumpi(series[:35], M, egress=egress, k=k, scale=1.0, **kwargs)
        for x in series[35:]:
            reference.update(float(x))
        assert np.array_equal(u64(restored.left_topk_P_), u64(reference.left_topk_P_))
        assert np.array_equal(restored.left_topk_I_, reference.left_topk_I_)
        assert restored.checkpoint() == reference.checkpoint()


def test_a_k_one_checkpoint_is_the_old_format_and_a_damaged_k_checkpoint_is_refused():
    series = random_walk(60, seed=69)
    old = grown(series, 1).checkpoint()
    assert old[4:6] == (2).to_bytes(2, "little")
    assert stumpi.restore(old).left_topk_P_.shape[1] == 1
    new = grown(series, 3).checkpoint()
    assert new[4:6] == (4).to_bytes(2, "little")
    assert bytes(new[:4]) == b"MPSI"
    for cut in (len(new) - 9, len(new) // 2, 10):
        with pytest.raises(ValueError):
            stumpi.restore(new[:cut])
    flipped = bytearray(new)
    flipped[len(flipped) // 2] ^= 1
    with pytest.raises(ValueError):
        stumpi.restore(bytes(flipped))


@pytest.mark.parametrize("k", [2, 3, 8])
@pytest.mark.parametrize("seed", [70, 71, 72])
def test_the_newest_row_equals_stumpys_after_every_update(k, seed):
    """STUMPY revises older rows with later windows, so only the row an update completes (which
    has only left neighbors) is comparable; its neighbors match exactly and its distances up to
    STUMPY's own float noise."""
    series = tie_free(120, seed)
    theirs = stumpy.stumpi(series[:40].copy(), M, egress=False, k=k)
    ours = stumpi(series[:40], M, egress=False, k=k, scale=GRID)
    assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
    for x in series[40:]:
        theirs.update(x)
        ours.update(float(x))
        assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
        assert np.allclose(ours.left_topk_P_[-1], theirs.P_[-1], rtol=0, atol=1e-6, equal_nan=False)
        assert np.array_equal(np.isinf(ours.left_topk_P_[-1]), np.isinf(theirs.P_[-1]))
        assert ours.left_I_[-1] == theirs.left_I_[-1]


@pytest.mark.parametrize("k", [2, 3])
def test_the_bounded_newest_row_equals_stumpys_egress_stream(k):
    n = 60
    series = tie_free(n + 50, seed=73)
    theirs = stumpy.stumpi(series[:n].copy(), M, egress=True, k=k)
    ours = stumpi(series[:n], M, egress=True, k=k, scale=GRID)
    for x in series[n:]:
        theirs.update(x)
        ours.update(float(x))
        # STUMPY's egress indices are already positions in the whole stream.
        assert np.array_equal(ours.left_topk_I_[-1], theirs.I_[-1])
        assert np.allclose(ours.left_topk_P_[-1], theirs.P_[-1], rtol=0, atol=1e-6)


def loose_constant(a, w):
    """Flags every window whose range is at most one tick: a rule other than exactly constant."""
    windows = np.lib.stride_tricks.sliding_window_view(np.asarray(a, dtype=np.float64), w)
    return np.ptp(windows, axis=1) <= 1.0


@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [2, 3])
def test_a_checkpoint_with_k_and_a_predicate_round_trips(egress, k):
    series = SERIES["random_walk"]()
    kwargs = {"w": 40} if egress else {}
    flagged = {"T_subseq_isconstant_func": loose_constant}
    stream = stumpi(series[:45], M, egress=egress, k=k, scale=1.0, **flagged, **kwargs)
    for x in series[45:65]:
        stream.update(float(x))
    data = stream.checkpoint()
    assert data[4:6] == (5).to_bytes(2, "little")

    restored = stumpi.restore(data, T_subseq_isconstant_func=loose_constant)
    for x in series[65:]:
        stream.update(float(x))
        restored.update(float(x))
    assert np.array_equal(u64(restored.left_topk_P_), u64(stream.left_topk_P_))
    assert np.array_equal(restored.left_topk_I_, stream.left_topk_I_)
    assert restored.checkpoint() == stream.checkpoint()

    # The rule changes what is kept, and the nearest neighbor is the one-neighbor stream's.
    plain = stumpi(series[:45], M, egress=egress, k=k, scale=1.0, **kwargs)
    one = stumpi(series[:45], M, egress=egress, scale=1.0, **flagged, **kwargs)
    for x in series[45:]:
        plain.update(float(x))
        one.update(float(x))
    assert not np.array_equal(u64(plain.left_topk_P_), u64(stream.left_topk_P_))
    assert np.array_equal(u64(stream.left_P_), u64(one.left_P_))
    assert np.array_equal(stream.left_I_, one.left_I_)

    with pytest.raises(ValueError, match="give it again"):
        stumpi.restore(data)
    with pytest.raises(ValueError, match="without"):
        stumpi.restore(plain.checkpoint(), T_subseq_isconstant_func=loose_constant)
