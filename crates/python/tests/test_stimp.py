"""`stimp(t, min_m, max_m, step, ...)`: STUMPY's pan matrix profile with every row exact. The rows
are `matrix_profile` on bits at any thread count and STUMPY's `stump` on tie-free series, the
visiting order is STUMPY's breadth-first one, and `pan()` is STUMPY's transform of the same rows."""

import numpy as np
import pytest
import math

from ladders import aamp, aamp_stimp, matrix_profile, stimp

from conftest import random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

SCALE = 1 / 64
VARIANTS = [
    {},
    {"normalize": False},
    {"contrast": False},
    {"binary": False},
    {"binary": False, "clip": False},
    {"contrast": False, "binary": False, "clip": False},
    {"threshold": 0.5},
    {"threshold": 0.0, "binary": False},
]


def grid_noise(n, seed):
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(size=n) * 64.0) / 64.0


def finish(pan):
    for _ in range(len(pan.M_)):
        pan.update()
    return pan


def u64(x):
    return np.asarray(x).view(np.uint64)


@pytest.mark.parametrize("n", range(1, 301))
def test_visiting_order_is_stumpys_breadth_first_order(n):
    t = grid_noise(5 * n + 20, seed=n)
    want = stumpy.stimp(t, min_m=3, max_m=3 + n - 1, percentage=1.0, pre_scrump=False)
    got = stimp(t, min_m=3, max_m=3 + n - 1, scale=SCALE)
    assert np.array_equal(got.M_, want.M_)
    assert got.M_.dtype == np.int64


def stumpy_range(n, min_m, max_m):
    """The `(min_m, max_m)` STUMPY runs with: sorted, the smaller raised to 3 and the larger
    lowered to the largest window size, when `max_m` is given."""
    if max_m is None:
        return min_m, None
    low, high = sorted([min_m, max_m])
    return max(3, low), min(n - (n + 3) // 5 - 1, high)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_m": 5, "max_m": 40, "step": 3},
        {"min_m": 40, "max_m": 5, "step": 4},
        {"min_m": 1, "max_m": 12},
        {"min_m": 10, "max_m": 10_000, "step": 60},
        {"min_m": 7, "max_m": 8},
    ],
)
def test_ranges_and_rows_equal_stumpy_on_tie_free_series(kwargs):
    t = grid_noise(160, seed=1)
    want = stumpy.stimp(t, percentage=1.0, pre_scrump=False, **kwargs)
    explicit = dict(kwargs)
    if "max_m" in kwargs:
        explicit["min_m"], explicit["max_m"] = stumpy_range(160, kwargs["min_m"], kwargs["max_m"])
    if explicit != kwargs:
        with pytest.raises(ValueError, match="STUMPY"):
            stimp(t, scale=SCALE, **kwargs)
    got = stimp(t, scale=SCALE, **explicit)
    assert np.array_equal(got.M_, want.M_)
    for _ in range(len(got.M_)):
        got.update()
        want.update()
    assert len(got.P_) == len(want.P_)
    for have, expected in zip(got.P_, want.P_):
        assert have.shape == expected.shape
        assert np.allclose(have, expected, rtol=0, atol=1e-6)


def test_each_row_is_matrix_profile_on_bits_whatever_the_threads():
    t = grid_noise(400, seed=2)
    base = finish(stimp(t, min_m=4, max_m=60, step=5, scale=SCALE, threads=1))
    for threads in [2, 3, 8, None]:
        other = finish(stimp(t, min_m=4, max_m=60, step=5, scale=SCALE, threads=threads))
        assert np.array_equal(u64(base.PAN_), u64(other.PAN_))
        for a, b in zip(base.PI_, other.PI_):
            assert np.array_equal(a, b)
        for a, b in zip(base.P_, other.P_):
            assert np.array_equal(u64(a), u64(b))
    for m, p, i in zip(base.M_, base.P_, base.PI_):
        want_p, want_i, _, _ = matrix_profile(t, int(m), SCALE, threads=1)
        assert np.array_equal(u64(p), u64(want_p))
        assert np.array_equal(i, want_i)


@pytest.mark.parametrize("make", [
    lambda: grid_noise(260, seed=3),
    lambda: random_walk(260, seed=4),
    lambda: walk_with_gaps(260, seed=5, gap_every=41, leading_gap=3),
])
def test_pan_is_stumpys_transform_of_the_same_rows(make):
    t = make()
    ours = stimp(t, min_m=4, max_m=30, scale=1.0)
    theirs = stumpy.stimp(t, min_m=4, max_m=30, percentage=1.0, pre_scrump=False)
    assert np.array_equal(ours.M_, theirs.M_)
    for done in range(len(ours.M_) + 1):
        for slot in range(done):
            row = theirs._bfs_indices[slot]
            theirs._PAN[row, :] = np.inf
            theirs._PAN[row, : len(ours.P_[slot])] = ours.P_[slot]
        theirs._n_processed = done
        for variant in VARIANTS:
            have, want = ours.pan(**variant), theirs.pan(**variant)
            assert have.shape == want.shape
            assert np.allclose(have, want, rtol=0, atol=1e-12, equal_nan=True), (done, variant)
        if done == 0:
            assert np.isnan(ours.PAN_).all()
        else:
            assert np.allclose(ours.PAN_, theirs.PAN_, rtol=0, atol=1e-12)
        if done < len(ours.M_):
            ours.update()


def test_the_image_has_one_row_per_size_and_one_column_per_sample():
    t = grid_noise(120, seed=6)
    pan = stimp(t, min_m=5, max_m=25, step=2, scale=SCALE)
    assert pan.PAN_.shape == (len(pan.M_), 120)
    assert [len(p) for p in pan.P_] == [120 - m + 1 for m in pan.M_]
    assert all((p == np.inf).all() for p in pan.P_)
    assert all((i == -1).all() for i in pan.PI_)
    pan.update()
    assert np.isfinite(pan.P_[0]).any() and not np.isfinite(pan.P_[1]).any()
    finish(pan)
    pan.update()
    assert all(np.isfinite(p).any() for p in pan.P_)


def test_percentage_and_pre_scrump_are_accepted_and_ignored():
    t = grid_noise(150, seed=7)
    base = finish(stimp(t, min_m=4, max_m=20, scale=SCALE))
    for percentage, pre_scrump in [(1.0, False), (0.01, True), (0.5, False), (0.0, True)]:
        other = finish(stimp(t, 4, 20, 1, percentage, pre_scrump, scale=SCALE))
        assert np.array_equal(u64(base.PAN_), u64(other.PAN_))


def test_inferred_scale_gives_the_same_rows_on_a_grid_series():
    t = grid_noise(150, seed=8)
    inferred = finish(stimp(t, min_m=4, max_m=20))
    given = finish(stimp(t, min_m=4, max_m=20, scale=SCALE))
    for a, b in zip(inferred.P_, given.P_):
        assert np.allclose(a, b, rtol=0, atol=1e-9)


def test_bad_arguments_are_value_errors():
    t = grid_noise(100, seed=9)
    for step in [0, -1, -2, -10]:
        with pytest.raises(ValueError):
            stimp(t, step=step)
    with pytest.raises(ValueError):
        stimp(t, min_m=101)
    with pytest.raises(ValueError):
        stimp(t, min_m=2)
    with pytest.raises(ValueError):
        stimp(t, min_m=90, max_m=200)
    with pytest.raises(ValueError):
        stimp(grid_noise(2, seed=1))
    with pytest.raises(ValueError):
        stimp(np.full(50, np.nan))
    for bad in [0, -3, True, 2.5]:
        with pytest.raises(ValueError):
            stimp(t, threads=bad)
    for scale in [0.0, -1.0, np.inf, np.nan]:
        with pytest.raises(ValueError):
            stimp(t, scale=scale)


def test_ranges_stumpy_clamps_are_errors_naming_the_explicit_range():
    t = grid_noise(100, seed=10)
    largest = 100 - 103 // 5 - 1
    cases = [
        ({"min_m": -5, "max_m": 6}, "min_m=-5 is below 3", "pass min_m=3"),
        ({"min_m": 2, "max_m": 6}, "min_m=2 is below 3", "pass min_m=3"),
        ({"min_m": 9, "max_m": 6}, "min_m=9 is above max_m=6", "pass min_m=6, max_m=9"),
        ({"min_m": 3, "max_m": 10**18}, f"max_m={10**18} is above {largest}", f"pass max_m={largest}"),
        ({"min_m": 3, "max_m": largest + 1}, "the largest window size", f"pass max_m={largest}"),
    ]
    for kwargs, named, remedy in cases:
        with pytest.raises(ValueError, match="STUMPY") as error:
            stimp(t, scale=SCALE, **kwargs)
        assert named in str(error.value) and remedy in str(error.value)
        with pytest.raises(ValueError, match=remedy):
            aamp_stimp(t, scale=SCALE, **kwargs)
    low = stimp(t, min_m=3, max_m=6, scale=SCALE)
    assert sorted(low.M_) == [3, 4, 5, 6]
    wide = stimp(t, min_m=3, max_m=largest, step=10**18, scale=SCALE)
    assert list(wide.M_) == [3]
    finish(wide)
    finish(low)
    wide.update()
    assert wide.PAN_.shape == (1, 100)
    edge = stimp(t, min_m=largest, max_m=largest, scale=SCALE)
    assert list(edge.M_) == [largest]


def stumpy_sizes(t, **kwargs):
    """STUMPY's window-size list in visiting order, `None` when it builds none."""
    try:
        return stumpy.stimp(t, percentage=1.0, pre_scrump=False, **kwargs).M_
    except Exception:
        return None


@pytest.mark.parametrize("n", [4, 5, 6, 10, 11, 20, 47, 60, 100])
def test_size_list_equals_stumpys_or_is_an_error_where_stumpy_fails_later(n):
    t = grid_noise(n, seed=n)
    for min_m in [-3, 0, 2, 3, 4, 10, 47, 59, 60, 61, 100]:
        for max_m in [None, -2, 3, 8, 40, 59, 200]:
            for step in [1, 2, 7, 1000]:
                want = stumpy_sizes(t, min_m=min_m, max_m=max_m, step=step)
                invalid = want is None or len(want) == 0 or want.min() < 3 or want.max() > n
                low, high = stumpy_range(n, min_m, max_m)
                if (low, high) != (min_m, max_m):
                    with pytest.raises(ValueError):
                        stimp(t, min_m=min_m, max_m=max_m, step=step, scale=SCALE)
                if invalid:
                    with pytest.raises(ValueError):
                        stimp(t, min_m=low, max_m=high, step=step, scale=SCALE)
                    continue
                got = stimp(t, min_m=low, max_m=high, step=step, scale=SCALE)
                assert np.array_equal(got.M_, want), (n, min_m, max_m, step)
                if len(want) <= 4:
                    finish(got)


def test_the_documented_edge_size_list():
    t = grid_noise(60, seed=1)
    pan = stimp(t, min_m=47, scale=SCALE)
    assert sorted(pan.M_) == [47, 48]
    finish(pan)
    assert np.isfinite(pan.PAN_).all()


def test_unsupported_keywords_say_what_is_missing():
    t = grid_noise(80, seed=2)
    stimp(t, T_subseq_isconstant_func=None, normalize=True, p=3.0, scale=SCALE)
    base = finish(stimp(t, min_m=4, max_m=9, scale=SCALE))
    other = finish(stimp(t, min_m=4, max_m=9, p=1.0, scale=SCALE))
    assert np.array_equal(u64(base.PAN_), u64(other.PAN_))


def test_scales_are_per_size_and_aligned_with_the_sizes():
    t = grid_noise(150, seed=3)
    pan = stimp(t, min_m=4, max_m=12, step=2)
    assert pan.scales_.shape == pan.M_.shape
    assert np.isnan(pan.scales_).all()
    finish(pan)
    for m, used, p in zip(pan.M_, pan.scales_, pan.P_):
        assert used > 0
        for scale in [float(used), None]:
            want, _, _, _ = matrix_profile(t, int(m), scale, threads=1)
            assert np.array_equal(u64(p), u64(want))
    fixed = finish(stimp(t, min_m=4, max_m=12, scale=SCALE))
    assert (fixed.scales_ == SCALE).all()


def test_a_negative_step_is_an_error_where_stumpy_builds_a_descending_range():
    tiny = grid_noise(3, seed=1)
    assert list(stumpy_sizes(tiny, min_m=0, max_m=0, step=-2)) == [3]
    with pytest.raises(ValueError, match="step"):
        stimp(tiny, min_m=0, max_m=0, step=-2)


# --- aamp_stimp: the same procedure on the non-normalized metric --------------------------


def brute_rows(t, m, scale):
    """The nearest non-trivial window of every window on exact integer ticks: ties to the
    smaller start, a window with a gap having no distance, reported as `aamp` reports it."""
    ticks = [None if not math.isfinite(x) else round(x / scale) for x in t]
    windows = [
        None if any(v is None for v in ticks[i : i + m]) else ticks[i : i + m]
        for i in range(len(t) - m + 1)
    ]
    zone = -(-m // 4)
    distances, indices = [], []
    for i, row in enumerate(windows):
        best = None
        for j, other in enumerate(windows):
            if row is None or other is None or abs(i - j) <= zone:
                continue
            squared = sum((a - b) ** 2 for a, b in zip(row, other))
            if best is None or squared < best[0]:
                best = (squared, j)
        distances.append(math.inf if best is None else math.sqrt(float(best[0])) * scale)
        indices.append(-1 if best is None else best[1])
    return np.array(distances), np.array(indices)


def aamp_tie_series():
    return {
        "constant": np.full(60, 5.0),
        "period 2": np.tile([0.0, 3.0], 30),
        "period 7": np.tile([0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 9),
        "few levels": np.random.default_rng(3).integers(0, 3, size=70).astype(np.float64) * 2,
        "gaps": walk_with_gaps(80, seed=11, gap_every=11, leading_gap=2),
        "walk": random_walk(80, seed=12),
    }


@pytest.mark.parametrize("name", list(aamp_tie_series()))
def test_aamp_stimp_rows_are_the_integer_brute_force(name):
    t = aamp_tie_series()[name]
    pan = finish(aamp_stimp(t, min_m=4, max_m=14, step=2, scale=1.0))
    assert pan.M_.tolist() == stimp(t, min_m=4, max_m=14, step=2, scale=1.0).M_.tolist()
    for m, distances, indices in zip(pan.M_, pan.P_, pan.PI_):
        want_d, want_i = brute_rows(t, int(m), 1.0)
        assert np.array_equal(u64(distances), u64(want_d)), (name, m)
        assert np.array_equal(indices, want_i), (name, m)


@pytest.mark.parametrize("seed", range(4))
def test_aamp_stimp_rows_and_sizes_equal_stumpys_on_tie_free_series(seed):
    t = grid_noise(150 + 10 * seed, seed=seed)
    kwargs = {"min_m": 4, "max_m": 40, "step": 3}
    got = aamp_stimp(t, scale=SCALE, **kwargs)
    want = stumpy.aamp_stimp(t, percentage=1.0, pre_scraamp=False, **kwargs)
    assert np.array_equal(got.M_, want.M_)
    for _ in range(len(got.M_)):
        got.update()
        want.update()
    for have, expected in zip(got.P_, want.P_):
        assert np.allclose(have, expected, rtol=1e-12, atol=0)
    scaled = {"contrast": False, "binary": False, "clip": False}
    assert np.allclose(got.pan(**scaled), want.pan(**scaled), rtol=1e-12, atol=0)


@pytest.mark.parametrize("make", [
    lambda: grid_noise(200, seed=3) * 50.0 + 7.0,
    lambda: walk_with_gaps(200, seed=5, gap_every=23, leading_gap=3),
    lambda: np.where(np.arange(120) % 17 == 5, np.inf, random_walk(120, seed=6)),
])
def test_aamp_stimp_normalizes_by_the_range_of_the_finite_samples_as_stumpy_does(make):
    t = make()
    ours = aamp_stimp(t, min_m=4, max_m=24, scale=1.0)
    theirs = stumpy.aamp_stimp(t, min_m=4, max_m=24, percentage=1.0, pre_scraamp=False)
    assert np.array_equal(ours.M_, theirs.M_)
    for done in range(len(ours.M_) + 1):
        for slot in range(done):
            row = theirs._bfs_indices[slot]
            theirs._PAN[row, :] = np.inf
            theirs._PAN[row, : len(ours.P_[slot])] = ours.P_[slot]
        theirs._n_processed = done
        for variant in VARIANTS:
            have, want = ours.pan(**variant), theirs.pan(**variant)
            assert np.allclose(have, want, rtol=0, atol=1e-12, equal_nan=True), (done, variant)
        if done < len(ours.M_):
            ours.update()


def test_aamp_stimp_rows_are_aamp_on_bits_whatever_the_threads():
    t = grid_noise(400, seed=2) * 9.0
    base = finish(aamp_stimp(t, min_m=4, max_m=60, step=5, scale=SCALE, threads=1))
    for threads in [2, 3, 8, None]:
        other = finish(aamp_stimp(t, min_m=4, max_m=60, step=5, scale=SCALE, threads=threads))
        assert np.array_equal(u64(base.PAN_), u64(other.PAN_))
        for a, b in zip(base.P_, other.P_):
            assert np.array_equal(u64(a), u64(b))
    for m, p, i in zip(base.M_, base.P_, base.PI_):
        want = aamp(t, int(m), scale=SCALE, threads=1)
        assert np.array_equal(u64(p), u64(want[0]))
        assert np.array_equal(i, want[1])


def test_normalize_false_routes_stimp_to_aamp_stimp_as_stumpy_does():
    t = grid_noise(120, seed=4)
    via_flag = finish(stimp(t, min_m=4, max_m=20, normalize=False, scale=SCALE))
    direct = finish(aamp_stimp(t, min_m=4, max_m=20, scale=SCALE))
    assert np.array_equal(u64(via_flag.PAN_), u64(direct.PAN_))
    theirs = finish(stumpy.stimp(t, min_m=4, max_m=20, percentage=1.0, pre_scrump=False, normalize=False))
    assert np.array_equal(via_flag.M_, theirs.M_)
    assert np.allclose(via_flag.PAN_, theirs.PAN_, rtol=0, atol=1e-12)
    zn = finish(stimp(t, min_m=4, max_m=20, scale=SCALE))
    assert not np.array_equal(u64(via_flag.P_[0]), u64(zn.P_[0]))


def test_aamp_stimp_ignores_percentage_and_pre_scraamp_and_takes_stumpys_keywords():
    t = grid_noise(150, seed=7)
    base = finish(aamp_stimp(t, min_m=4, max_m=20, scale=SCALE))
    for percentage, pre in [(1.0, False), (0.01, True), (0.5, False), (0.0, True)]:
        other = finish(aamp_stimp(t, 4, 20, 1, percentage, pre, 2.0, scale=SCALE))
        assert np.array_equal(u64(base.PAN_), u64(other.PAN_))
    from_list = finish(aamp_stimp(list(t), min_m=4, max_m=20, scale=SCALE))
    assert np.array_equal(u64(base.PAN_), u64(from_list.PAN_))
    ints = np.arange(40) % 5
    assert finish(aamp_stimp(ints, min_m=4, max_m=8, scale=1.0)).PAN_.shape == (5, 40)
    assert isinstance(base, stimp)


@pytest.mark.parametrize("p", [1.0, 3.0, 0.5, float("inf"), float("nan")])
def test_aamp_stimp_p_other_than_two_is_not_implemented(p):
    t = grid_noise(80, seed=2)
    with pytest.raises(NotImplementedError):
        aamp_stimp(t, p=p)
    with pytest.raises(NotImplementedError):
        stimp(t, normalize=False, p=p)


def test_aamp_stimp_bad_arguments_are_value_errors():
    t = grid_noise(100, seed=9)
    for kwargs in [{"step": 0}, {"min_m": 101}, {"min_m": 2}, {"threads": 0}, {"scale": -1.0}]:
        with pytest.raises(ValueError):
            aamp_stimp(t, **kwargs)
    with pytest.raises(ValueError):
        aamp_stimp(np.full(50, np.nan), scale=1.0)
    with pytest.raises(ValueError):
        aamp_stimp(np.full(50, np.inf))
    with pytest.raises(ValueError, match="pass max_m=79"):
        aamp_stimp(t, min_m=3, max_m=10**18, step=10**18, scale=SCALE)
    huge = finish(aamp_stimp(t, min_m=3, max_m=79, step=10**18, scale=SCALE))
    assert list(huge.M_) == [3]
