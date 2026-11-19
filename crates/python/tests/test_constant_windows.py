"""`T_subseq_isconstant`: a caller's predicate for which windows are constant, replacing "all
samples equal", on every function STUMPY gives the keyword. Each is checked against STUMPY on
tie-free data with the predicate given as an array and as a callable, a predicate that marks
non-constant windows constant included; flags that say what the default rule says change no bit;
and wrong lengths, wrong dtypes and a failing callable raise before any computation.

A window the predicate marks non-constant that is in fact constant has no meaningful STUMPY
answer (it divides by a zero standard deviation), so those are checked against what this library
defines instead: such a window is uncorrelated with every other, at distance `sqrt(2m)`.
"""

import pickle

import numpy as np
import pytest
from ladders import (
    BoundedStream,
    FlossStream,
    GrowingStream,
    ab_join,
    discords,
    distance_profile,
    fluss_series,
    left_profile,
    matrix_profile,
    mdl,
    mpdist,
    mstump,
    ostinato,
    snippets,
    stimp,
    stumpi,
    subspace,
)
from ladders import fluss as fluss_from_index
from ladders import match as match_fn
from ladders import mmotifs as mmotifs_fn
from ladders import motifs as motifs_fn

from multi_reference import Tally, mdl_reference, reference

stumpy = pytest.importorskip("stumpy")

GRID = 1e-3
M = 10


def u64(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


def series(n, seed, flats=((50, 70), (120, 135))):
    """Continuous values on a fine grid, with flat runs: no two windows are exact repeats."""
    rng = np.random.default_rng(seed)
    t = np.round(np.cumsum(rng.normal(size=n)) * 1e3) / 1e3
    for start, stop in flats:
        if stop <= n:
            t[start:stop] = t[start]
    return t


def exactly_constant(t, m):
    return np.array([np.ptp(t[k : k + m]) == 0 for k in range(len(t) - m + 1)])


def with_extra(t, m, count, seed):
    """The exactly constant windows of `t` plus `count` other windows marked constant."""
    flags = exactly_constant(t, m)
    flags[np.random.default_rng(seed).choice(len(flags), count, replace=False)] = True
    return flags


def near_constant(a, w):
    """A callable predicate: windows within 0.5 of flat count as constant."""
    return np.array([np.ptp(a[k : k + w]) < 0.5 for k in range(len(a) - w + 1)])


def near_constant_window(a, w):
    """[near_constant] for the one window a streaming predicate is asked about."""
    return near_constant(a, w)


def flagged_rows(flags, indices):
    """Rows neither flagged themselves nor matched to a flagged window: the rows whose nearest
    neighbor is not one of several windows tied at the same distance."""
    safe = ~flags
    valid = indices >= 0
    safe[valid] &= ~flags[indices[valid]]
    return safe & valid


def assert_profile(got_p, got_i, want_p, want_i, flags):
    want_p = np.asarray(want_p, dtype=np.float64)
    want_i = np.asarray(want_i, dtype=np.int64)
    assert np.allclose(got_p, want_p, rtol=1e-6, atol=1e-6, equal_nan=True)
    safe = flagged_rows(flags, np.asarray(want_i))
    assert np.array_equal(np.asarray(got_i)[safe], np.asarray(want_i)[safe])


# ---------- the batch profiles ----------


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_matrix_profile_equals_stumpy(predicate):
    t = series(300, 1)
    flags = with_extra(t, M, 25, 5) if predicate == "array" else near_constant(t, M)
    given = flags if predicate == "array" else near_constant
    want = stumpy.stump(t, M, T_A_subseq_isconstant=given)
    p, i, left_i, right_i = matrix_profile(t, M, GRID, threads=1, T_A_subseq_isconstant=given)
    assert_profile(p, i, want[:, 0], want[:, 1], flags)
    assert np.array_equal(right_i[~flags], want[:, 3][~flags])


def test_matrix_profile_default_flags_change_no_bit():
    t = series(300, 2)
    flags = exactly_constant(t, M)
    for given in (flags, lambda a, w: exactly_constant(a, w), list(flags)):
        got = matrix_profile(t, M, GRID, threads=2, T_A_subseq_isconstant=given)
        want = matrix_profile(t, M, GRID, threads=2)
        for a, b in zip(got, want):
            assert np.array_equal(u64(a), u64(b))


def test_matrix_profile_top_k_reads_the_flags():
    t = series(300, 3)
    flags = with_extra(t, M, 20, 6)
    want = stumpy.stump(t, M, k=3, T_A_subseq_isconstant=flags)
    p, i, _, _ = matrix_profile(t, M, GRID, threads=1, k=3, T_A_subseq_isconstant=flags)
    safe = ~flags
    assert np.allclose(p[safe], want[:, :3][safe].astype(float), rtol=1e-6, atol=1e-6, equal_nan=True)


def test_left_profile_is_the_left_side_of_the_matrix_profile():
    t = series(300, 4)
    flags = with_extra(t, M, 25, 7)
    _, left_index, _ = left_profile(t, M, GRID, T_A_subseq_isconstant=flags)
    _, _, left_i, _ = matrix_profile(t, M, GRID, T_A_subseq_isconstant=flags)
    assert np.array_equal(left_index, left_i)
    p, i, _ = left_profile(t, M, GRID, T_A_subseq_isconstant=near_constant)
    q, j, _ = left_profile(t, M, GRID, T_A_subseq_isconstant=near_constant(t, M))
    assert np.array_equal(u64(p), u64(q)) and np.array_equal(i, j)


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_ab_join_equals_stumpy(predicate):
    a, b = series(250, 8), series(220, 9, flats=((30, 45),))
    if predicate == "array":
        flags_a, flags_b = with_extra(a, M, 15, 1), with_extra(b, M, 12, 2)
        given_a, given_b = flags_a, flags_b
    else:
        flags_a, flags_b = near_constant(a, M), near_constant(b, M)
        given_a = given_b = near_constant
    want = stumpy.stump(a, M, b, ignore_trivial=False, T_A_subseq_isconstant=given_a, T_B_subseq_isconstant=given_b)
    p, i, _, _ = ab_join(a, b, M, GRID, GRID, threads=1, T_A_subseq_isconstant=given_a, T_B_subseq_isconstant=given_b)
    assert np.allclose(p, want[:, 0].astype(float), rtol=1e-6, atol=1e-6)
    # Rows tied among constant windows of `b` have no single answer.
    unambiguous = ~flags_a & ~flags_b[np.asarray(want[:, 1], dtype=np.int64)]
    assert np.array_equal(i[unambiguous], want[:, 1].astype(np.int64)[unambiguous])
    one_side = ab_join(a, b, M, GRID, GRID, threads=1, T_A_subseq_isconstant=given_a)
    both = ab_join(a, b, M, GRID, GRID, threads=3, T_A_subseq_isconstant=given_a, T_B_subseq_isconstant=None)
    assert np.array_equal(u64(one_side[0]), u64(both[0]))


def test_discords_and_fluss_read_the_flags_through_the_profile():
    t = series(300, 10)
    flags = with_extra(t, M, 20, 3)
    _, i, _, _ = matrix_profile(t, M, GRID, T_A_subseq_isconstant=flags)
    cac, locs = fluss_series(t, M, M, 3, scale=GRID, T_A_subseq_isconstant=flags)
    want_cac, want_locs = fluss_from_index(i, M, 3, m=M)
    assert np.array_equal(u64(cac), u64(want_cac)) and np.array_equal(locs, want_locs)
    unflagged_cac, _ = fluss_series(t, M, M, 3, scale=GRID)
    assert not np.array_equal(u64(cac), u64(unflagged_cac))
    far, at = discords(t, M, 3, GRID, T_A_subseq_isconstant=flags)
    p, _, _, _ = matrix_profile(t, M, GRID, T_A_subseq_isconstant=flags)
    assert far[0] == pytest.approx(np.nanmax(p[np.isfinite(p)]), rel=1e-12)
    assert p[at[0]] == far[0]


# ---------- the derived functions ----------


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_mpdist_equals_stumpy(predicate):
    a, b = series(250, 11), series(200, 12, flats=((30, 45),))
    if predicate == "array":
        given_a, given_b = with_extra(a, M, 15, 4), with_extra(b, M, 10, 5)
    else:
        given_a = given_b = near_constant
    for kwargs in ({"percentage": 0.05}, {"k": 40}, {"percentage": 0.5}):
        want = stumpy.mpdist(a, b, M, T_A_subseq_isconstant=given_a, T_B_subseq_isconstant=given_b, **kwargs)
        got = mpdist(a, b, M, scale_a=GRID, scale_b=GRID, T_A_subseq_isconstant=given_a, T_B_subseq_isconstant=given_b, **kwargs)
        assert got == pytest.approx(want, rel=1e-6, abs=1e-6), kwargs


def test_mpdist_default_flags_change_no_bit_and_normalize_false_ignores_them():
    a, b = series(250, 13), series(200, 14)
    flags = exactly_constant(a, M)
    assert mpdist(a, b, M, scale_a=GRID, scale_b=GRID, T_A_subseq_isconstant=flags) == mpdist(a, b, M, scale_a=GRID, scale_b=GRID)
    plain = mpdist(a, b, M, normalize=False, scale=GRID)
    assert mpdist(a, b, M, normalize=False, scale=GRID, T_A_subseq_isconstant=np.ones(len(a) - M + 1, bool)) == plain


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_motifs_equal_stumpy(predicate):
    t = series(300, 15, flats=())
    flags = np.zeros(len(t) - M + 1, bool)
    flags[[40, 170]] = True
    given = flags if predicate == "array" else (lambda a, w: np.isin(np.arange(len(a) - w + 1), [40, 170]))
    mp_ = stumpy.stump(t, M, T_A_subseq_isconstant=flags)
    want_d, want_i = stumpy.motifs(t, mp_[:, 0].astype(float), max_motifs=3, T_subseq_isconstant=flags)
    got_d, got_i = motifs_fn(t, np.zeros(len(t) - M + 1), max_motifs=3, scale=GRID, T_subseq_isconstant=given)
    assert got_i[0, 0] in (40, 170) and got_i[0, 1] in (40, 170)
    assert got_d[0, 1] == 0.0
    rows = min(len(want_d), len(got_d))
    assert np.allclose(got_d[:rows, 0], want_d[:rows, 0], atol=1e-6)
    assert np.array_equal(np.sort(got_i[0, :2]), np.sort(want_i[0, :2]))


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_match_and_distance_profile_equal_stumpy(predicate):
    t = series(300, 16, flats=())
    q = np.round(np.random.default_rng(2).normal(size=M).cumsum() * 1e3) / 1e3
    flags_t = np.zeros(len(t) - M + 1, bool)
    flags_t[[20, 100, 230]] = True
    given_t = flags_t if predicate == "array" else (lambda a, w: np.isin(np.arange(len(a) - w + 1), [20, 100, 230]))
    for q_flag in (False, True):
        given_q = np.array([q_flag]) if predicate == "array" else (lambda a, w, q_flag=q_flag: np.array([q_flag]))
        want = stumpy.mass(q, t, T_subseq_isconstant=flags_t, Q_subseq_isconstant=np.array([q_flag]))
        index, distance = distance_profile(q, t, GRID, T_subseq_isconstant=given_t, Q_subseq_isconstant=given_q)
        assert np.allclose(distance, want, rtol=1e-6, atol=1e-6)
        found = stumpy.match(q, t, max_distance=2.5, T_subseq_isconstant=flags_t, Q_subseq_isconstant=np.array([q_flag]))
        d, i = match_fn(q, t, max_distance=2.5, scale=GRID, T_subseq_isconstant=given_t, Q_subseq_isconstant=given_q)
        assert np.array_equal(i, found[:, 1].astype(np.int64))
        assert np.allclose(d, found[:, 0].astype(float), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_ostinato_equals_stumpy(predicate):
    ts = [series(120, 20), series(100, 21, flats=((30, 50),)), series(110, 22, flats=())]
    if predicate == "array":
        flags = [with_extra(t, M, 6, k) for k, t in enumerate(ts)]
    else:
        flags = [near_constant] * 3
    want = stumpy.ostinato(ts, M, Ts_subseq_isconstant=flags)
    got = ostinato(ts, M, Ts_subseq_isconstant=flags, scales=[GRID] * 3)
    assert got[0] == pytest.approx(want[0], abs=1e-6)
    if want[0] > 1e-9:
        assert (got[1], got[2]) == (want[1], want[2])
    else:
        # Windows flagged constant are at distance 0 from each other: the radius is 0 and
        # which of the tied windows is reported is not decided by the distances.
        assert got[0] == 0.0 and want[0] == pytest.approx(0.0, abs=1e-9)
    assert ostinato(ts, M, Ts_subseq_isconstant=near_constant, scales=[GRID] * 3) == got or predicate == "array"


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_snippets_equal_stumpy(predicate):
    t = series(200, 23)
    s = 8
    flags = with_extra(t, s, 12, 9) if predicate == "array" else near_constant(t, s)
    given = flags if predicate == "array" else near_constant
    want = stumpy.snippets(t, 20, 2, s=s, mpdist_T_subseq_isconstant=given)
    got = snippets(t, 20, 2, s=s, mpdist_T_subseq_isconstant=given, scale=GRID)
    assert np.array_equal(got[1], want[1])
    assert np.allclose(got[2], want[2], rtol=1e-6, atol=1e-6)
    assert np.allclose(got[4], want[4], rtol=1e-6, atol=1e-5)


def test_snippets_default_flags_change_no_bit():
    t = series(200, 24)
    flags = exactly_constant(t, 20)
    got = snippets(t, 20, 2, mpdist_T_subseq_isconstant=flags, scale=GRID)
    want = snippets(t, 20, 2, scale=GRID)
    for a, b in zip(got, want):
        assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_mstump_equals_stumpy(predicate):
    rng = np.random.default_rng(30)
    t = np.round(np.cumsum(rng.normal(size=(3, 150)), axis=1) * 1e3) / 1e3
    t[0, 30:50] = t[0, 30]
    flags = np.array([with_extra(row, M, 8, k) for k, row in enumerate(t)])
    given = flags if predicate == "array" else (lambda a, w: near_constant(a, w))
    if predicate == "callable":
        flags = np.array([near_constant(row, M) for row in t])
    want_p, want_i = stumpy.mstump(t, M, T_subseq_isconstant=given)
    got_p, got_i = mstump(t, M, scales=GRID, threads=2, T_subseq_isconstant=given)
    assert np.allclose(got_p, want_p, rtol=1e-6, atol=1e-6, equal_nan=True)
    listed = mstump(t, M, scales=GRID, T_subseq_isconstant=list(flags))
    assert np.array_equal(u64(listed[0]), u64(got_p)) and np.array_equal(listed[1], got_i)


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_mmotifs_equal_stumpy(predicate):
    rng = np.random.default_rng(32)
    t = np.round(np.cumsum(rng.normal(size=(3, 160)), axis=1) * 1e3) / 1e3
    flags = np.zeros((3, 160 - M + 1), bool)
    flags[:2, [5, 100]] = True
    given = flags
    if predicate == "callable":
        given = [lambda a, w: np.isin(np.arange(len(a) - w + 1), [5, 100])] * 2 + [None]
    p, i = stumpy.mstump(t, M, T_subseq_isconstant=flags)
    # `k` is given: STUMPY's `mmotifs` calls `mdl` without the predicate, so it chooses `k` and
    # the subspaces as if no window were flagged, where this library uses the flags for both.
    want = stumpy.mmotifs(t, p, i, max_motifs=1, k=1, T_subseq_isconstant=flags)
    got = mmotifs_fn(t, p, i, max_motifs=1, k=1, scales=GRID, T_subseq_isconstant=given)
    assert got[0].shape == want[0].shape
    assert np.allclose(got[0], want[0], rtol=1e-6, atol=1e-6, equal_nan=True)
    assert np.array_equal(got[1], want[1]) and got[1][0, 0] == 5
    assert [sorted(s) for s in got[2]] == [sorted(s) for s in want[2]]


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_subspace_and_mdl_equal_the_exact_reference_and_stumpy(predicate):
    """The subspaces and sizes are exactly calculable (multi_reference: a flagged window is all in
    the middle bin, the others binned exactly), so they are checked exactly; STUMPY is compared
    beside them: no binned sample is near an edge (counted), so its subspaces must be equal, and its
    sizes must equal the exact formula with NumPy's `log2` in place of the correctly rounded one."""
    rng = np.random.default_rng(31)
    t = np.round(np.cumsum(rng.normal(size=(4, 120)), axis=1) * 1e3) / 1e3
    t[1, 40:55] = t[1, 40]
    flags = np.array([with_extra(row, 6, 10, k) for k, row in enumerate(t)])
    given = flags if predicate == "array" else (lambda a, w: np.array([np.ptp(a[k : k + w]) < 0.7 for k in range(len(a) - w + 1)]))
    if predicate == "callable":
        flags = np.array([[np.ptp(row[k : k + 6]) < 0.7 for k in range(len(row) - 5)] for row in t])
    scales = [GRID] * 4
    tally = Tally()
    for first, second in ((3, 77), (41, 90), (10, 100)):
        exact = reference(t, 6, first, second, 3, scales=scales, flags=flags, tally=tally)
        for k in range(4):
            got = subspace(t, 6, first, second, k, T_subseq_isconstant=given, scales=scales)
            assert list(got) == exact[: k + 1], (first, second, k)
            want = stumpy.subspace(t, 6, first, second, k, T_subseq_isconstant=given)
            assert list(want) == exact[: k + 1], (first, second, k)
    firsts, seconds = [3, 41, 10, 60], [77, 90, 100, 20]
    got_bits, got_subspaces = mdl(t, 6, firsts, seconds, T_subseq_isconstant=given, scales=scales)
    exact_bits, exact_subspaces = mdl_reference(t, 6, firsts, seconds, scales=scales, flags=flags, tally=tally)
    assert np.array_equal(u64(got_bits), u64(exact_bits))
    assert [list(s) for s in got_subspaces] == exact_subspaces
    want_bits, want_subspaces = stumpy.mdl(t, 6, firsts, seconds, T_subseq_isconstant=given)
    model_bits, _ = mdl_reference(t, 6, firsts, seconds, scales=scales, flags=flags, log=lambda n: np.log2(np.float64(n)))
    assert np.array_equal(u64(want_bits), u64(model_bits))
    assert [list(s) for s in want_subspaces] == exact_subspaces
    assert tally.near == 0 and tally.samples > 0


def drain(pan):
    while len(pan.M_) > sum(np.isfinite(p).all() for p in pan.P_):
        pan.update()
    return pan


@pytest.mark.parametrize("predicate", ["array", "callable"])
def test_stimp_equals_stumpy_when_given_a_function(predicate):
    t = series(120, 40, flats=((20, 40),))
    func = near_constant
    want = stumpy.stimp(t, min_m=4, max_m=10, percentage=1.0, pre_scrump=False, T_subseq_isconstant_func=func)
    got = stimp(t, min_m=4, max_m=10, scale=GRID, T_subseq_isconstant_func=func)
    for _ in range(len(want.M_)):
        want.update()
        got.update()
    assert np.array_equal(want.M_, got.M_)
    for a, b in zip(want.P_, got.P_):
        assert np.allclose(a, b, rtol=1e-6, atol=1e-6)


# ---------- the streams ----------


def test_stumpi_equals_stumpy_with_a_function():
    t = series(150, 50, flats=((20, 40),))
    head, tail = t[:90], t[90:]
    want = stumpy.stumpi(head, M, egress=False, T_subseq_isconstant_func=near_constant)
    got = stumpi(head, M, egress=False, scale=GRID, T_subseq_isconstant_func=near_constant)
    for x in tail:
        want.update(x)
        got.update(x)
    assert np.allclose(got.left_P_, want.left_P_, rtol=1e-6, atol=1e-6, equal_nan=True)
    assert_profile(got.left_P_, got.left_I_, want.left_P_, want.left_I_, near_constant(t, M))


def test_stumpi_default_predicate_changes_no_bit_and_checkpoints_with_it():
    t = series(150, 51, flats=((20, 40),))
    plain = stumpi(t[:60], M, egress=True, scale=GRID, w=50)
    flat = lambda a, w: exactly_constant(a, w)
    flagged = stumpi(t[:60], M, egress=True, scale=GRID, w=50, T_subseq_isconstant_func=flat)
    for x in t[60:100]:
        plain.update(x)
        flagged.update(x)
    assert np.array_equal(u64(plain.left_P_), u64(flagged.left_P_))
    data = flagged.checkpoint()
    restored = stumpi.restore(data, T_subseq_isconstant_func=flat)
    for x in t[100:]:
        plain.update(x)
        flagged.update(x)
        restored.update(x)
    assert np.array_equal(u64(plain.left_P_), u64(flagged.left_P_))
    assert np.array_equal(u64(restored.left_P_), u64(flagged.left_P_))
    assert restored.checkpoint() == flagged.checkpoint()
    with pytest.raises(ValueError, match="give it again"):
        stumpi.restore(data)
    with pytest.raises(ValueError, match="without"):
        stumpi.restore(plain.checkpoint(), T_subseq_isconstant_func=flat)
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(flagged)
    assert pickle.loads(pickle.dumps(plain)).left_I_.tolist() == plain.left_I_.tolist()


def test_a_failing_predicate_leaves_the_stream_unchanged():
    t = series(100, 52)

    class Boom(Exception):
        pass

    def predicate(a, w):
        if a[-1] == t[70]:
            raise Boom("no")
        return exactly_constant(a, w)

    stream = stumpi(t[:60], M, egress=False, scale=GRID, T_subseq_isconstant_func=predicate)
    for x in t[60:70]:
        stream.update(x)
    before = (stream.left_P_.copy(), stream.checkpoint())
    with pytest.raises(Boom):
        stream.update(t[70])
    assert np.array_equal(stream.left_P_, before[0]) and stream.checkpoint() == before[1]
    reference = stumpi(t[:60], M, egress=False, scale=GRID)
    for x in t[60:70]:
        reference.update(x)
    for x in t[71:90]:
        stream.update(x)
        reference.update(x)
    # The sample that failed was left out, so the two differ only by it.
    assert len(stream.left_P_) == len(reference.left_P_)


def test_a_predicate_that_fails_leaves_the_scale_guard_as_it_was():
    """The guard of a stream that inferred its scale sees only the samples the stream took."""
    grid = 2.0**-12
    rng = np.random.default_rng(3)
    t = np.round(np.cumsum(rng.normal(size=60)) / grid) * grid
    big = 0.9e9 * grid
    state = {"fail": True}

    def predicate(a, w):
        if state["fail"] and len(a) == w and a[-1] == big:
            state["fail"] = False
            raise RuntimeError("the predicate fails on this sample")
        return np.zeros(len(a) - w + 1, bool)

    stream = stumpi(t, 7, egress=False, T_subseq_isconstant_func=predicate)
    fresh = stumpi(t, 7, egress=False, T_subseq_isconstant_func=lambda a, w: np.zeros(len(a) - w + 1, bool))
    fresh.update(-big)  # accepted on its own, so it must be after a sample that was never taken
    with pytest.raises(RuntimeError):
        stream.update(big)
    stream.update(-big)
    assert len(stream.left_P_) == len(fresh.left_P_)


def test_floss_stream_with_the_default_predicate_changes_no_bit():
    t = series(200, 53, flats=((60, 90),))
    plain = FlossStream(M, 40, GRID, w=80)
    flagged = FlossStream(M, 40, GRID, w=80, T_subseq_isconstant_func=lambda a, w: exactly_constant(a, w))
    for x in t[:100]:
        plain.update(x)
        flagged.update(x)
    plain.update_many(t[100:])
    flagged.update_many(t[100:])
    assert np.array_equal(u64(plain.curve()), u64(flagged.curve()))
    one_by_one = FlossStream(M, 40, GRID, w=80, T_subseq_isconstant_func=near_constant_window)
    batched = FlossStream(M, 40, GRID, w=80, T_subseq_isconstant_func=near_constant_window)
    for x in t:
        one_by_one.update(x)
    batched.update_many(t)
    assert np.array_equal(u64(one_by_one.curve()), u64(batched.curve()))


# ---------- a window the predicate does not call constant ----------


def test_an_unflagged_constant_window_is_uncorrelated_with_everything():
    t = series(200, 60)
    flags = np.zeros(len(t) - M + 1, bool)
    p, i, _, _ = matrix_profile(t, M, GRID, T_A_subseq_isconstant=flags)
    constant_rows = np.flatnonzero(exactly_constant(t, M))
    assert len(constant_rows) > 0
    for row in constant_rows:
        assert p[row] == pytest.approx(np.sqrt(2 * M), rel=1e-12)
    index, distance = distance_profile(t[50 : 50 + M], t, GRID, Q_subseq_isconstant=[False])
    assert distance[100] == pytest.approx(np.sqrt(2 * M), rel=1e-12) or index[100] == -1


# ---------- bad predicates ----------

SERIES_FUNCTIONS = {
    "matrix_profile": lambda t, **kw: matrix_profile(t, M, GRID, T_A_subseq_isconstant=kw["flags"]),
    "left_profile": lambda t, **kw: left_profile(t, M, GRID, T_A_subseq_isconstant=kw["flags"]),
    "discords": lambda t, **kw: discords(t, M, 2, GRID, T_A_subseq_isconstant=kw["flags"]),
    "fluss": lambda t, **kw: fluss_series(t, M, M, 3, scale=GRID, T_A_subseq_isconstant=kw["flags"]),
    "ab_join": lambda t, **kw: ab_join(t, t, M, GRID, GRID, T_B_subseq_isconstant=kw["flags"]),
    "mpdist": lambda t, **kw: mpdist(t, t, M, scale_a=GRID, scale_b=GRID, T_A_subseq_isconstant=kw["flags"]),
    "motifs": lambda t, **kw: motifs_fn(t, np.zeros(len(t) - M + 1), scale=GRID, T_subseq_isconstant=kw["flags"]),
    "match": lambda t, **kw: match_fn(t[:M], t, scale=GRID, T_subseq_isconstant=kw["flags"]),
    "mass": lambda t, **kw: distance_profile(t[:M], t, GRID, T_subseq_isconstant=kw["flags"]),
    "ostinato": lambda t, **kw: ostinato([t, t], M, Ts_subseq_isconstant=[kw["flags"], None], scales=[GRID, GRID]),
    "snippets": lambda t, **kw: snippets(t, 20, 2, s=M, mpdist_T_subseq_isconstant=kw["flags"], scale=GRID),
    "mmotifs": lambda t, **kw: mmotifs_fn(
        np.array([t, t]), np.zeros((2, len(t) - M + 1)), np.full((2, len(t) - M + 1), -1),
        scales=GRID, T_subseq_isconstant=[kw["flags"], None],
    ),
    "mstump": lambda t, **kw: mstump(np.array([t, t]), M, scales=GRID, T_subseq_isconstant=[kw["flags"], None]),
    "subspace": lambda t, **kw: subspace(np.array([t, t]), M, 1, 2, 0, T_subseq_isconstant=[kw["flags"], None], scales=[GRID] * 2),
    "mdl": lambda t, **kw: mdl(np.array([t, t]), M, [1, 1], [2, 3], T_subseq_isconstant=[kw["flags"], None], scales=[GRID] * 2),
    "stimp": lambda t, **kw: stimp(t, 4, 8, scale=GRID, T_subseq_isconstant_func=kw["flags"]).update(),
    "stumpi": lambda t, **kw: stumpi(t, M, egress=False, scale=GRID, T_subseq_isconstant_func=kw["flags"]),
}

# `ab_join`'s second flags and `ostinato`'s, `mstump`'s and `subspace`'s list entries are the
# arrays under test; the callable-only functions take the callable form.
CALLABLE_ONLY = {"stimp", "stumpi"}
WINDOW_COUNT_OF = {"snippets": M}


def windows(name, t):
    return len(t) - WINDOW_COUNT_OF.get(name, M) + 1


@pytest.mark.parametrize("name", [n for n in SERIES_FUNCTIONS if n not in CALLABLE_ONLY])
def test_wrong_length_arrays_are_value_errors(name):
    t = series(200, 70)
    count = windows(name, t)
    for wrong in (count - 1, count + 1, 0, 1):
        flags = np.zeros(wrong, bool)
        if name == "match" and wrong == 1:
            continue
        with pytest.raises(ValueError, match="one flag per window"):
            SERIES_FUNCTIONS[name](t, flags=flags)


@pytest.mark.parametrize("name", list(SERIES_FUNCTIONS))
def test_a_callable_returning_a_wrong_length_or_dtype_is_a_value_error(name):
    t = series(200, 71)
    count = windows(name, t)
    if name in ("match",):
        count = len(t) - M + 1
    wrong_length = lambda a, w: np.zeros(count - 1, bool)
    with pytest.raises(ValueError):
        SERIES_FUNCTIONS[name](t, flags=wrong_length)
    for result in (np.zeros(count, int), np.zeros(count, float), np.zeros(count, np.int8), ["a"] * count, None, 3):
        with pytest.raises(ValueError, match="bool|dimension|flag"):
            SERIES_FUNCTIONS[name](t, flags=lambda a, w, result=result: result)


@pytest.mark.parametrize("name", [n for n in SERIES_FUNCTIONS if n not in CALLABLE_ONLY])
def test_arrays_must_be_one_dimensional_booleans(name):
    t = series(200, 72)
    count = windows(name, t)
    with pytest.raises(ValueError, match="bool"):
        SERIES_FUNCTIONS[name](t, flags=np.zeros(count, dtype=np.int64))
    with pytest.raises(ValueError, match="bool"):
        SERIES_FUNCTIONS[name](t, flags=np.zeros(count, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension|flag"):
        SERIES_FUNCTIONS[name](t, flags=np.zeros((2, count), dtype=bool))
    with pytest.raises(ValueError):
        SERIES_FUNCTIONS[name](t, flags="constant")


@pytest.mark.parametrize("name", list(SERIES_FUNCTIONS))
def test_an_exception_in_the_callable_propagates(name):
    t = series(200, 73)

    class Boom(Exception):
        pass

    def predicate(a, w):
        raise Boom("the predicate failed")

    with pytest.raises(Boom):
        SERIES_FUNCTIONS[name](t, flags=predicate)


@pytest.mark.parametrize("name", list(SERIES_FUNCTIONS))
def test_a_callable_with_the_wrong_arguments_is_the_callers_type_error(name):
    t = series(200, 74)
    with pytest.raises(TypeError):
        SERIES_FUNCTIONS[name](t, flags=lambda a: np.zeros(1, bool))


@pytest.mark.parametrize("name", ["stimp", "stumpi"])
def test_a_streaming_predicate_must_be_callable(name):
    t = series(200, 75)
    with pytest.raises(ValueError, match="callable"):
        SERIES_FUNCTIONS[name](t, flags=np.zeros(len(t) - M + 1, bool))


def test_the_callable_gets_the_series_with_nan_for_gaps_and_the_window_length():
    t = series(120, 76)
    t[30] = np.inf
    t[60] = np.nan
    seen = []

    def predicate(a, w):
        seen.append((a.copy(), w))
        return exactly_constant(np.nan_to_num(a, nan=0.0), w)

    matrix_profile(t, M, GRID, T_A_subseq_isconstant=predicate)
    (got, w), = seen
    assert w == M and got.dtype == np.float64
    assert np.isnan(got[30]) and np.isnan(got[60])
    assert np.array_equal(got[~np.isnan(got)], t[np.isfinite(t)])


def test_a_flag_on_a_window_with_a_gap_is_ignored():
    t = series(150, 77)
    t[40] = np.nan
    ones = np.ones(len(t) - M + 1, bool)
    p, i, _, _ = matrix_profile(t, M, GRID, T_A_subseq_isconstant=ones)
    assert np.all(np.isinf(p[31:41])) and np.all(i[31:41] == -1)
    assert np.all(p[:31][i[:31] >= 0] == 0.0)


@pytest.mark.parametrize(
    "dtype", [np.int64, np.int32, np.float32, np.float64]
)
def test_series_of_any_numeric_dtype_and_boolean_lists_are_accepted(dtype):
    t = (np.cumsum(np.random.default_rng(5).integers(-2, 3, size=150))).astype(dtype)
    flags = [bool(x) for x in exactly_constant(t.astype(np.float64), M)]
    want = mpdist(t.astype(np.float64), t.astype(np.float64)[::-1].copy(), M, scale_a=1.0, scale_b=1.0)
    got = mpdist(t, t[::-1].copy(), M, scale_a=1.0, scale_b=1.0, T_A_subseq_isconstant=flags, T_B_subseq_isconstant=np.array(flags[::-1]))
    assert got == want
    got = motifs_fn(t, np.zeros(len(t) - M + 1), scale=1.0, T_subseq_isconstant=np.array(flags, dtype=np.bool_))
    assert got[0].shape == motifs_fn(t, np.zeros(len(t) - M + 1), scale=1.0)[0].shape


def test_every_placeholder_is_gone():
    t = series(150, 78)
    for call in (
        lambda: mdl(np.array([t, t]), M, [1, 1], [2, 3], T_subseq_isconstant=near_constant, scales=[GRID] * 2),
        lambda: subspace(np.array([t, t]), M, 1, 2, 0, T_subseq_isconstant=near_constant, scales=[GRID] * 2),
        lambda: snippets(t, 20, 2, s=M, mpdist_T_subseq_isconstant=near_constant, scale=GRID),
        lambda: stimp(t, 4, 6, scale=GRID, T_subseq_isconstant_func=near_constant).update(),
    ):
        call()


def test_the_stimp_object_is_the_default_one_without_a_function():
    t = series(120, 79)
    base, other = stimp(t, 4, 8, scale=GRID), stimp(t, 4, 8, scale=GRID, T_subseq_isconstant_func=lambda a, w: exactly_constant(a, w))
    for _ in range(len(base.M_)):
        base.update()
        other.update()
    for a, b in zip(base.P_, other.P_):
        assert np.array_equal(u64(a), u64(b))
    with pytest.raises(TypeError):
        stimp(t, 4, 8, normalize=False, T_subseq_isconstant_func=near_constant, scale=GRID)
