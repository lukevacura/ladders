"""`scrump` and `scraamp`: STUMPY's anytime profiles. Under `np.random.seed` the visited diagonals,
the chunks and the profile after every update equal STUMPY's on tie-free series; the profile
after any number of updates is the exact profile over the diagonals visited, on bits, at any
thread count; after the last update it is the full profile; every refused input is an ordinary
error."""

import math

import numpy as np
import pytest
from ladders import aamp, ab_join, matrix_profile, scraamp, scrump

from conftest import assert_mostly_finite, random_walk, walk_with_gaps

M = 12
THREADS = [1, 2, 7, None]
SCALE = 1 / 64


def u64(x):
    return np.asarray(x).view(np.uint64)


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def columns(obj):
    return obj.P_.copy(), obj.I_.copy(), obj.left_I_.copy(), obj.right_I_.copy()


def drain(obj):
    """The columns before the first update and after each one."""
    out = [columns(obj)]
    while obj.n_updates_ < obj.n_chunks_:
        obj.update()
        out.append(columns(obj))
    return out


# ---- STUMPY parity -------------------------------------------------------------------------


@pytest.mark.parametrize("percentage", [0.04, 0.3, 0.9, 1.0])
@pytest.mark.parametrize("k", [1, 3])
def test_scrump_follows_stumpy_update_by_update(percentage, k):
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(420, seed=1)
    np.random.seed(7)
    ours = scrump(t, M, percentage=percentage, k=k, scale=SCALE)
    np.random.seed(7)
    theirs = stumpy.scrump(t, M, percentage=percentage, k=k)
    assert ours.n_chunks_ == theirs._n_chunks
    assert np.isinf(ours.P_).all() and (ours.I_ == -1).all()
    for step in range(ours.n_chunks_):
        ours.update()
        theirs.update()
        stop = theirs._chunk_diags_ranges[step][1]
        assert np.array_equal(ours.visited_, theirs._diags[:stop]), step
        assert ours.n_updates_ == step + 1
        assert ours.P_.shape == theirs.P_.shape == ((420 - M + 1,) if k == 1 else (420 - M + 1, k))
        assert np.array_equal(ours.I_, theirs.I_), step
        assert np.allclose(ours.P_, theirs.P_, rtol=1e-9, atol=1e-6), step
        assert np.array_equal(ours.left_I_, theirs.left_I_), step
        assert np.array_equal(ours.right_I_, theirs.right_I_), step
    assert ours.I_.dtype == ours.left_I_.dtype == ours.right_I_.dtype == np.int64
    assert ours.P_.dtype == np.float64


@pytest.mark.parametrize("percentage", [0.04, 0.3, 1.0])
def test_scraamp_follows_stumpy_update_by_update(percentage):
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(420, seed=2)
    np.random.seed(11)
    ours = scraamp(t, M, percentage=percentage, scale=SCALE)
    np.random.seed(11)
    theirs = stumpy.scraamp(t, M, percentage=percentage)
    assert ours.n_chunks_ == theirs._n_chunks
    for step in range(ours.n_chunks_):
        ours.update()
        theirs.update()
        stop = theirs._chunk_diags_ranges[step][1]
        assert np.array_equal(ours.visited_, theirs._diags[:stop]), step
        assert np.array_equal(ours.I_, theirs.I_), step
        assert np.allclose(ours.P_, theirs.P_, rtol=1e-12, atol=0), step
        assert np.array_equal(ours.left_I_, theirs.left_I_), step
        assert np.array_equal(ours.right_I_, theirs.right_I_), step


def test_normalize_false_is_scraamp_as_in_stumpy():
    t = grid_noise(300, seed=3)
    np.random.seed(5)
    routed = scrump(t, M, percentage=0.2, normalize=False, scale=SCALE)
    np.random.seed(5)
    direct = scraamp(t, M, percentage=0.2, scale=SCALE)
    for a, b in zip(drain(routed), drain(direct)):
        for x, y in zip(a, b):
            assert np.array_equal(u64(x), u64(y))


@pytest.mark.parametrize("cls", [scrump, scraamp])
def test_the_random_state_is_drawn_exactly_as_stumpy_draws_it(cls):
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(200, seed=4)
    np.random.seed(3)
    cls(t, M, percentage=0.2)
    # timing-ok: the global state was seeded on the line above; its next draws are the check
    after_ours = np.random.rand(3)
    np.random.seed(3)
    getattr(stumpy, cls.__name__)(t, M, percentage=0.2)
    # timing-ok: the global state was seeded above, and the same draw is compared
    assert np.array_equal(after_ours, np.random.rand(3))


@pytest.mark.parametrize("n,m", [(60, 5), (97, 8), (150, 20)])
@pytest.mark.parametrize("percentage", [0.01, 0.07, 0.33, 0.5, 0.77])
def test_chunks_and_diagonals_match_stumpy_for_many_shapes(n, m, percentage):
    stumpy = pytest.importorskip("stumpy")
    t = grid_noise(n, seed=n)
    np.random.seed(n + m)
    ours = scrump(t, m, percentage=percentage, scale=SCALE)
    np.random.seed(n + m)
    theirs = stumpy.scrump(t, m, percentage=percentage)
    assert ours.n_chunks_ == theirs._n_chunks
    for step in range(ours.n_chunks_):
        ours.update()
        theirs.update()
        assert np.array_equal(ours.visited_, theirs._diags[: theirs._chunk_diags_ranges[step][1]])


# ---- exactness over the visited diagonals ---------------------------------------------------


def restricted_nonnorm(a, m, scale, diagonals):
    """Exact integer profile over the pairs on `diagonals` alone, ties to the smaller index; a
    window with a missing sample has no neighbor and is never one."""
    ticks = [None if not math.isfinite(x) else round(x / scale) for x in a]
    rows = len(a) - m + 1
    whole = [all(v is not None for v in ticks[i : i + m]) for i in range(rows)]
    squared = lambda i, j: sum((x - y) ** 2 for x, y in zip(ticks[i : i + m], ticks[j : j + m]))
    p, idx, left, right = np.full(rows, np.inf), np.full(rows, -1), np.full(rows, -1), np.full(rows, -1)
    diagonals = sorted({int(d) for d in diagonals})

    def best(row, columns):
        found = None
        for j in columns:
            if 0 <= j < rows and whole[j]:
                d = squared(row, j)
                if found is None or (d, j) < found:
                    found = (d, j)
        return found

    for row in range(rows):
        if not whole[row]:
            continue
        l = best(row, [row - d for d in diagonals])
        r = best(row, [row + d for d in diagonals])
        left[row] = -1 if l is None else l[1]
        right[row] = -1 if r is None else r[1]
        sides = [s for s in (l, r) if s is not None]
        nearest = min(sides) if sides else None
        if nearest is not None:
            p[row], idx[row] = math.sqrt(float(nearest[0])) * scale, nearest[1]
    return p, idx, left, right


@pytest.mark.parametrize("threads", THREADS)
def test_scraamp_after_each_update_is_the_exact_profile_over_the_visited_diagonals(threads):
    t = walk_with_gaps(90, seed=5, gap_every=17)
    np.random.seed(2)
    obj = scraamp(t, 6, percentage=0.15, scale=1.0, threads=threads)
    while obj.n_updates_ < obj.n_chunks_:
        obj.update()
        want = restricted_nonnorm(t, 6, 1.0, obj.visited_)
        got = columns(obj)
        for x, y in zip(got, want):
            assert np.array_equal(u64(x), u64(y)), obj.n_updates_


@pytest.mark.parametrize("threads", THREADS)
def test_scrump_after_the_last_update_is_the_full_profile(threads):
    t = grid_noise(350, seed=6)
    for k in [1, 4]:
        np.random.seed(9)
        obj = scrump(t, M, percentage=0.2, k=k, scale=SCALE, threads=threads)
        states = drain(obj)
        want = matrix_profile(t, M, scale=SCALE, threads=1, k=k)
        for x, y in zip(states[-1], want):
            assert np.array_equal(u64(x), u64(y))
        assert_mostly_finite(want[1] if k == 1 else want[1][:, 0], M)


@pytest.mark.parametrize("threads", THREADS)
def test_scraamp_after_the_last_update_is_the_full_profile(threads):
    t = walk_with_gaps(200, seed=7, gap_every=53)
    np.random.seed(4)
    obj = scraamp(t, M, percentage=0.25, scale=1.0, threads=threads)
    states = drain(obj)
    want = aamp(t, M, scale=1.0, threads=1)
    for x, y in zip(states[-1], want):
        assert np.array_equal(u64(x), u64(y))
    assert_mostly_finite(want[1], M, threshold=0.6)


@pytest.mark.parametrize("threads", THREADS)
def test_windows_past_the_exact_domain_are_ranked_up_front_and_converge_to_the_full_profile(threads):
    # A shock far past the exact domain: the windows that hold it have no distance in the hot
    # passes, so their pairs are ranked when the profile is made, before the first update.
    t = grid_noise(200, seed=31)
    t[90] += 1e9
    shocked = slice(90 - M + 1, 91)
    for k in [1, 3]:
        np.random.seed(5)
        obj = scrump(t, M, percentage=0.25, k=k, scale=SCALE, threads=threads)
        first = obj.P_ if k == 1 else obj.P_[:, 0]
        assert np.isfinite(first[shocked]).all() and obj.n_updates_ == 0
        states = drain(obj)
        want = matrix_profile(t, M, scale=SCALE, threads=1, k=k)
        for x, y in zip(states[-1], want):
            assert np.array_equal(u64(x), u64(y))
    b = grid_noise(150, seed=32, level=1.0)
    np.random.seed(6)
    z = scrump(t, M, b, ignore_trivial=False, percentage=0.25, k=2, scale=SCALE, scale_b=SCALE, threads=threads)
    assert np.isfinite(z.P_[shocked, 0]).all()
    drain(z)
    want = ab_join(t, b, M, scale_a=SCALE, scale_b=SCALE, threads=1, k=2)
    assert np.array_equal(u64(z.P_), u64(want[0])) and np.array_equal(z.I_, want[1])


def test_updates_after_the_last_change_nothing():
    t = grid_noise(150, seed=8)
    np.random.seed(1)
    obj = scrump(t, M, percentage=0.5, scale=SCALE)
    drain(obj)
    before = columns(obj)
    visited = obj.visited_.copy()
    for _ in range(3):
        obj.update()
    assert obj.n_updates_ == obj.n_chunks_
    for x, y in zip(columns(obj), before):
        assert np.array_equal(u64(x), u64(y))
    assert np.array_equal(obj.visited_, visited)
    assert sorted(visited) == list(range(-(-M // 4) + 1, len(t) - M + 1))


def test_nothing_is_visited_before_the_first_update():
    obj = scrump(grid_noise(100, seed=9), M, percentage=0.5, k=2, scale=SCALE)
    assert obj.n_updates_ == 0 and len(obj.visited_) == 0
    assert np.isinf(obj.P_).all() and obj.P_.shape == (100 - M + 1, 2)
    assert (obj.I_ == -1).all() and (obj.left_I_ == -1).all() and (obj.right_I_ == -1).all()


@pytest.mark.parametrize("cls", [scrump, scraamp])
def test_the_same_seed_gives_the_same_profile_at_any_thread_count(cls):
    t = walk_with_gaps(260, seed=10, gap_every=31)
    outputs = []
    for threads in THREADS:
        np.random.seed(21)
        obj = cls(t, M, percentage=0.2, scale=1.0, threads=threads)
        obj.update()
        obj.update()
        outputs.append((obj.visited_.copy(), *columns(obj)))
    for other in outputs[1:]:
        for x, y in zip(outputs[0], other):
            assert np.array_equal(u64(x), u64(y))
    np.random.seed(22)
    different = cls(t, M, percentage=0.2, scale=1.0)
    different.update()
    assert not np.array_equal(different.visited_, outputs[0][0])


@pytest.mark.parametrize("cls", [scrump, scraamp])
@pytest.mark.parametrize(
    "convert",
    [
        lambda a: a.astype(np.int64),
        lambda a: a.astype(np.int32),
        lambda a: a.astype(np.float32),
        lambda a: a.tolist(),
    ],
)
def test_integer_float32_and_list_inputs_equal_float64(cls, convert):
    base = np.round(random_walk(160, seed=11) * 3)
    want_input = base.astype(np.float64)
    np.random.seed(6)
    expected = cls(want_input, 8, percentage=0.4, scale=1.0)
    np.random.seed(6)
    got = cls(convert(base), 8, percentage=0.4, scale=1.0)
    for a, b in zip(drain(expected), drain(got)):
        for x, y in zip(a, b):
            assert np.array_equal(u64(x), u64(y))


# ---- bad input ------------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [scrump, scraamp])
@pytest.mark.parametrize(
    "percentage", [float("nan"), 0.0, -0.0, -0.5, 1.0000001, 2.0, float("inf"), float("-inf")]
)
def test_a_percentage_outside_zero_to_one_is_a_value_error(cls, percentage):
    with pytest.raises(ValueError, match="percentage"):
        cls(grid_noise(100, seed=12), M, percentage=percentage, scale=SCALE)


@pytest.mark.parametrize("cls", [scrump, scraamp])
def test_bad_windows_threads_and_scales_are_value_errors(cls):
    t = grid_noise(100, seed=13)
    for m in [0, 1, 2, -3]:
        with pytest.raises(ValueError, match="m must"):
            cls(t, m, scale=SCALE)
    for m in [100, 101, 5000]:
        with pytest.raises(ValueError, match="window"):
            cls(t, m, scale=SCALE)
    with pytest.raises(ValueError, match="window"):
        cls(np.array([]), M, scale=SCALE)
    for threads in [0, -1, True, 1.5, "2"]:
        with pytest.raises(ValueError, match="threads"):
            cls(t, M, scale=SCALE, threads=threads)
    for scale in [0.0, -1.0, float("nan"), float("inf")]:
        with pytest.raises(ValueError, match="scale"):
            cls(t, M, scale=scale)
    with pytest.raises(ValueError):
        cls(np.full(100, np.nan), M)
    with pytest.raises(ValueError, match="ignore_trivial"):
        cls(t, M, ignore_trivial=False, scale=SCALE)


def test_a_k_out_of_range_is_a_value_error_and_a_large_one_is_padding():
    t = grid_noise(100, seed=14)
    for k in [0, -1]:
        with pytest.raises(ValueError, match="k must"):
            scrump(t, M, k=k, scale=SCALE)
    with pytest.raises(ValueError, match="k"):
        scrump(t, M, k=1 << 40, scale=SCALE)
    np.random.seed(1)
    obj = scrump(t, M, k=100, percentage=1.0, scale=SCALE)
    obj.update()
    assert obj.P_.shape == obj.I_.shape == (100 - M + 1, 100)
    assert np.isinf(obj.P_[:, -1]).all() and (obj.I_[:, -1] == -1).all()


def test_what_is_not_available_yet_is_not_implemented():
    t = grid_noise(100, seed=15)
    with pytest.raises(NotImplementedError, match="pre_scrump"):
        scrump(t, M, pre_scrump=True)
    with pytest.raises(NotImplementedError, match="pre_scraamp"):
        scraamp(t, M, pre_scraamp=True)
    for p in [1.0, 3.0, 2.5]:
        with pytest.raises(NotImplementedError, match="p must be"):
            scraamp(t, M, p=p)
        with pytest.raises(NotImplementedError, match="p must be"):
            scrump(t, M, normalize=False, p=p)
    with pytest.raises(NotImplementedError, match="k"):
        scraamp(t, M, k=2)
    with pytest.raises(NotImplementedError, match="k"):
        scrump(t, M, normalize=False, k=3)
    constant = np.zeros(100 - M + 1, dtype=bool)
    with pytest.raises(NotImplementedError, match="isconstant"):
        scrump(t, M, T_A_subseq_isconstant=constant)
    with pytest.raises(NotImplementedError, match="isconstant"):
        scrump(t, M, T_B_subseq_isconstant=constant)
    with pytest.raises(TypeError, match="isconstant"):
        scrump(t, M, normalize=False, T_A_subseq_isconstant=constant)
    # `s` only matters with pre_scrump, which STUMPY ignores it without.
    np.random.seed(3)
    with_s = scrump(t, M, s=5, scale=SCALE)
    np.random.seed(3)
    without_s = scrump(t, M, scale=SCALE)
    assert np.array_equal(with_s.visited_, without_s.visited_)
    assert with_s.n_chunks_ == without_s.n_chunks_


def test_p_is_ignored_with_normalize_as_in_stumpy():
    t = grid_noise(100, seed=16)
    np.random.seed(2)
    a = scrump(t, M, p=3.0, scale=SCALE, percentage=0.5)
    np.random.seed(2)
    b = scrump(t, M, scale=SCALE, percentage=0.5)
    assert np.array_equal(a.visited_, b.visited_)


def test_an_equal_t_b_with_ignore_trivial_is_the_self_join_as_in_stumpy():
    a = grid_noise(100, seed=17)
    np.random.seed(1)
    same = scrump(a, M, a.copy(), ignore_trivial=True, scale=SCALE)
    np.random.seed(1)
    alone = scrump(a, M, scale=SCALE)
    assert np.array_equal(same.visited_, alone.visited_)


# ---- AB-joins -------------------------------------------------------------------------------


@pytest.mark.parametrize("percentage", [0.05, 0.4, 1.0])
@pytest.mark.parametrize("k", [1, 2])
def test_scrump_ab_join_follows_stumpy_update_by_update(percentage, k):
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(300, seed=21), grid_noise(210, seed=22, level=1.5)
    np.random.seed(5)
    ours = scrump(a, M, b, ignore_trivial=False, percentage=percentage, k=k, scale=SCALE, scale_b=SCALE)
    np.random.seed(5)
    theirs = stumpy.scrump(a, M, b, ignore_trivial=False, percentage=percentage, k=k)
    assert ours.n_chunks_ == theirs._n_chunks
    for step in range(ours.n_chunks_):
        ours.update()
        theirs.update()
        assert np.array_equal(ours.visited_, theirs._diags[: theirs._chunk_diags_ranges[step][1]])
        assert np.array_equal(ours.I_, theirs.I_), step
        assert np.allclose(ours.P_, theirs.P_, rtol=1e-9, atol=1e-6), step
        assert (ours.left_I_ == -1).all() and (ours.right_I_ == -1).all()
        assert ours.left_I_.shape == ours.right_I_.shape == (300 - M + 1,)


@pytest.mark.parametrize("percentage", [0.05, 0.4, 1.0])
def test_scraamp_ab_join_follows_stumpy_update_by_update(percentage):
    stumpy = pytest.importorskip("stumpy")
    a, b = grid_noise(300, seed=23), grid_noise(210, seed=24, level=1.5)
    np.random.seed(6)
    ours = scraamp(a, M, b, ignore_trivial=False, percentage=percentage, scale=SCALE)
    np.random.seed(6)
    theirs = stumpy.scraamp(a, M, b, ignore_trivial=False, percentage=percentage)
    assert ours.n_chunks_ == theirs._n_chunks
    for step in range(ours.n_chunks_):
        ours.update()
        theirs.update()
        assert np.array_equal(ours.visited_, theirs._diags[: theirs._chunk_diags_ranges[step][1]])
        assert np.array_equal(ours.I_, theirs.I_), step
        assert np.allclose(ours.P_, theirs.P_, rtol=1e-12, atol=0), step


@pytest.mark.parametrize("threads", THREADS)
def test_ab_joins_after_the_last_update_are_the_full_joins(threads):
    a, b = grid_noise(260, seed=25), grid_noise(180, seed=26, level=2.0)
    np.random.seed(8)
    z = scrump(a, M, b, ignore_trivial=False, percentage=0.25, k=3, scale=SCALE, scale_b=SCALE, threads=threads)
    drain(z)
    want = ab_join(a, b, M, scale_a=SCALE, scale_b=SCALE, threads=1, k=3)
    assert np.array_equal(u64(z.P_), u64(want[0])) and np.array_equal(z.I_, want[1])
    np.random.seed(8)
    n = scraamp(a, M, b, percentage=0.25, scale=SCALE, threads=threads)
    drain(n)
    full = aamp(a, M, b, ignore_trivial=False, scale=SCALE, threads=1)
    assert np.array_equal(u64(n.P_), u64(full[0])) and np.array_equal(n.I_, full[1])
    assert (n.left_I_ == -1).all() and (n.right_I_ == -1).all()


def test_a_join_visits_every_diagonal_of_both_series_once():
    a, b = grid_noise(70, seed=27), grid_noise(50, seed=28)
    np.random.seed(2)
    obj = scrump(a, M, b, percentage=0.3, scale=SCALE, scale_b=SCALE)
    drain(obj)
    assert sorted(obj.visited_) == list(range(-(70 - M), 50 - M + 1))
    assert (obj.left_I_ == -1).all()


def test_scale_b_is_only_for_a_z_normalized_join_and_scales_must_be_valid():
    a, b = grid_noise(80, seed=29), grid_noise(60, seed=30)
    for kwargs in [dict(), dict(normalize=False)]:
        with pytest.raises(ValueError, match="scale_b"):
            scrump(a, M, scale_b=SCALE, **kwargs)
    with pytest.raises(ValueError, match="scale_b"):
        scrump(a, M, b, normalize=False, scale_b=SCALE)
    with pytest.raises(ValueError, match="scale"):
        scrump(a, M, b, scale_b=-1.0)
    inferred = scrump(a, M, b)
    given = scrump(a, M, b, scale_b=None)
    assert inferred.n_chunks_ >= 1 and given.n_chunks_ >= 1


def test_join_windows_longer_than_either_series_are_value_errors():
    a, b = grid_noise(80, seed=31), grid_noise(20, seed=32)
    for cls in (scrump, scraamp):
        with pytest.raises(ValueError, match="window"):
            cls(a, 25, b)
        with pytest.raises(ValueError, match="window"):
            cls(b, 25, a)
    with pytest.raises(ValueError, match="window"):
        scrump(a, M, np.array([]))
