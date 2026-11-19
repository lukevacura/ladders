"""`aampdist`, `aamp_ostinato`, `mpdist(normalize=False)`, `ostinato(normalize=False)`: STUMPY's
non-normalized MPdist and consensus motif on tie-free series, an exact integer brute force of the
same procedures on series full of ties and gaps, the thread count never changing a bit,
`normalize=False` being the non-normalized function, and `p` other than 2 being
`NotImplementedError`."""

import math

import numpy as np
import pytest
from ladders import aamp_ostinato, aampdist, mpdist, ostinato

from conftest import random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

SCALE = 1 / 64
THREADS = [1, 2, 3, 8, None, np.int64(4)]


def grid_noise(n, seed, level=0.0):
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def planted(count, m, seed, jitter=0.05):
    """`count` noise series of different lengths, each holding a noisy copy of one pattern."""
    rng = np.random.default_rng(seed)
    pattern = 3.0 * np.cumsum(rng.normal(size=m))
    series = []
    for k in range(count):
        t = grid_noise(150 + 37 * k, seed=seed * 100 + k)
        at = int(rng.integers(0, len(t) - m))
        t[at : at + m] = pattern + jitter * rng.normal(size=m)
        series.append(np.round(t * 64.0) / 64.0)
    return series


def bits(x):
    return np.float64(x).view(np.uint64)


# --- an exact brute force of the procedures on integer ticks ------------------------------


def to_ticks(series, scale):
    return [round(x / scale) if math.isfinite(x) else None for x in series]


def windows(ticks, m):
    out = []
    for i in range(len(ticks) - m + 1):
        w = ticks[i : i + m]
        out.append(None if any(v is None for v in w) else w)
    return out


def squared(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def nearest(query, target):
    """`(squared distance, start)` of the exactly nearest window, ties to the smaller start."""
    if query is None:
        return None
    found = [(squared(query, w), j) for j, w in enumerate(target) if w is not None]
    return min(found) if found else None


def brute_aampdist(a, b, m, scale, percentage=0.05, k=None):
    wa, wb = windows(to_ticks(a, scale), m), windows(to_ticks(b, scale), m)
    joined = []
    for side, rows, others in [(0, wa, wb), (1, wb, wa)]:
        for i, w in enumerate(rows):
            hit = nearest(w, others)
            if hit is not None:
                joined.append((hit[0], side, i))
    total = len(wa) + len(wb)
    if k is None:
        k = min(math.ceil(min(max(percentage, 0.0), 1.0) * (len(a) + len(b))), total - 1)
    k = min(k, total - 1)
    joined.sort()
    ordered = [math.sqrt(sq) * scale for sq, _, _ in joined]
    ordered += [math.inf] * (total - len(ordered))
    if math.isfinite(ordered[k]):
        return ordered[k]
    finite_before = sum(math.isfinite(x) for x in ordered[:k])
    return ordered[max(0, finite_before - 1)]


def brute_aamp_ostinato(series, m, scale):
    ws = [windows(to_ticks(s, scale), m) for s in series]
    count = len(ws)
    best = None
    for j in range(count):
        h = (j + 1) % count
        order = sorted(
            (hit[0], q) for q, w in enumerate(ws[j]) if (hit := nearest(w, ws[h])) is not None
        )
        for radius, q in order:
            if best is not None and radius >= best[0]:
                break
            for i in range(count):
                if i in (j, h):
                    continue
                hit = nearest(ws[j][q], ws[i])
                if hit is None:
                    radius = None
                    break
                radius = max(radius, hit[0])
                if best is not None and radius >= best[0]:
                    break
            if radius is not None and (best is None or radius < best[0]):
                best = (radius, j, q)
    if best is None:
        return (math.inf, 0, 0)

    def across(series_at, window_at):
        radii, starts = [], []
        for target in ws:
            hit = nearest(ws[series_at][window_at], target)
            radii.append(math.inf if hit is None else math.sqrt(hit[0]) * scale)
            starts.append(0 if hit is None else hit[1])
        return radii, starts

    radius = math.sqrt(best[0]) * scale
    radii, starts = across(best[1], best[2])
    best_mean = sum(radii) / len(radii)
    found = (radius, best[1], best[2])
    for candidate in range(count):
        if not np.isclose(radii[candidate], radius):
            continue
        cand_radii, _ = across(candidate, starts[candidate])
        if np.isclose(max(cand_radii), radius) and sum(cand_radii) / len(cand_radii) < best_mean:
            found = (radius, candidate, starts[candidate])
            best_mean = sum(cand_radii) / len(cand_radii)
    return found


def few_levels(n, seed, levels=3):
    return np.random.default_rng(seed).integers(0, levels, size=n).astype(np.float64)


def tie_pairs():
    cycle = np.array([3.0, 1.0, 4.0, 1.0])
    return [
        (np.full(25, 3.0), np.full(19, 3.0)),
        (np.full(25, 3.0), np.full(19, 4.0)),
        (np.tile(cycle, 11), np.tile(cycle, 8)[1:]),
        (few_levels(45, 1), few_levels(38, 2)),
        (few_levels(45, 3, 2), few_levels(52, 4, 2)),
        (np.repeat([1.0, 5.0, 1.0], 12), np.repeat([5.0, 1.0], 9)),
    ]


# --- aampdist -------------------------------------------------------------------------------


def test_aampdist_equals_stumpy_on_tie_free_series():
    for m in [3, 8, 20]:
        for sizes in [(400, 400), (700, 330), (260, 910)]:
            a, b = grid_noise(sizes[0], seed=1), grid_noise(sizes[1], seed=2, level=3.0)
            for threads in [1, 3, None]:
                for percentage in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]:
                    got = aampdist(a, b, m, percentage, scale=SCALE, threads=threads)
                    want = stumpy.aampdist(a, b, m, percentage=percentage)
                    assert isinstance(got, float)
                    assert abs(got - want) < 1e-6, (m, sizes, percentage, threads)
                for k in [0, 1, 7, 100]:
                    got = aampdist(a, b, m, k=k, scale=SCALE, threads=threads)
                    assert abs(got - stumpy.aampdist(a, b, m, k=k)) < 1e-6, (m, k, threads)


def test_aampdist_equals_stumpy_on_series_with_gaps():
    a = walk_with_gaps(500, seed=3, gap_every=37)
    b = walk_with_gaps(420, seed=4, gap_every=29, leading_gap=5)
    for percentage in [0.05, 0.3, 1.0]:
        got = aampdist(a, b, 12, percentage, scale=1.0)
        want = stumpy.aampdist(a, b, 12, percentage=percentage)
        assert abs(got - want) < 1e-6, percentage


def test_the_scale_is_inferred_from_both_series_together():
    a, b = grid_noise(300, seed=5), grid_noise(260, seed=6, level=2.0)
    inferred = aampdist(a, b, 10)
    assert abs(inferred - stumpy.aampdist(a, b, 10)) < 1e-6
    assert abs(inferred - aampdist(a, b, 10, scale=SCALE)) < 1e-12


@pytest.mark.parametrize("ties", range(6))
def test_aampdist_equals_the_brute_force_on_series_of_ties(ties):
    a, b = tie_pairs()[ties]
    for m in [3, 4, 5, 8]:
        if len(a) < m or len(b) < m:
            continue
        for kwargs in [{}, {"percentage": 0.5}, {"percentage": 1.0}, {"k": 0}, {"k": 5}]:
            want = brute_aampdist(a, b, m, 1.0, **kwargs)
            for threads in [1, 3]:
                got = aampdist(a, b, m, scale=1.0, threads=threads, **kwargs)
                assert bits(got) == bits(want), (m, kwargs, threads)


def test_aampdist_equals_the_brute_force_with_gaps_at_the_edges():
    base = random_walk(40, seed=11)
    for lead, trail, every in [(0, 0, 9), (4, 0, 0), (0, 5, 0), (6, 6, 7), (30, 0, 0)]:
        a = base.copy()
        if every:
            a[every - 1 :: every] = np.nan
        a[:lead] = np.nan
        a[len(a) - trail :] = np.inf if trail else a[len(a) - trail :]
        b = random_walk(35, seed=12)
        for m in [3, 4, 7]:
            for kwargs in [{}, {"percentage": 1.0}, {"k": 3}]:
                want = brute_aampdist(a, b, m, 1.0, **kwargs)
                assert bits(aampdist(a, b, m, scale=1.0, **kwargs)) == bits(want), (m, kwargs)
                want = brute_aampdist(b, a, m, 1.0, **kwargs)
                assert bits(aampdist(b, a, m, scale=1.0, **kwargs)) == bits(want), (m, kwargs)


def test_a_series_without_finite_windows_gives_inf_and_is_at_zero_from_itself():
    a = random_walk(80, seed=8)
    assert aampdist(a, np.full(40, np.nan), 10, scale=1.0) == float("inf")
    assert aampdist(a, a, 10, scale=1.0) == 0.0


def test_the_distance_is_in_the_callers_units_and_the_scale_applies_once():
    a, b = random_walk(120, seed=1), random_walk(100, seed=2)
    unit = aampdist(a, b, 9, scale=1.0)
    assert unit > 0
    fine = aampdist(a * 0.25, b * 0.25, 9, scale=0.25)
    assert bits(fine) == bits(unit * 0.25)


@pytest.mark.parametrize("gaps", [False, True])
def test_aampdist_every_thread_count_equals_one_thread_on_bits(gaps):
    if gaps:
        a = walk_with_gaps(3200, seed=40, gap_every=211)
        b = walk_with_gaps(2700, seed=41, gap_every=301)
    else:
        a, b = random_walk(3200, seed=40), random_walk(2700, seed=41)
    first = aampdist(a, b, 24, scale=1.0, threads=1)
    assert np.isfinite(first)
    for threads in THREADS:
        assert bits(aampdist(a, b, 24, scale=1.0, threads=threads)) == bits(first), threads


def test_mpdist_with_normalize_false_is_aampdist():
    a, b = grid_noise(300, seed=5), grid_noise(260, seed=6, level=2.0)
    want = stumpy.mpdist(a, b, 10, normalize=False)
    got = mpdist(a, b, 10, normalize=False)
    assert abs(got - want) < 1e-6
    assert got == aampdist(a, b, 10)
    assert mpdist(a, b, 10, 0.3, normalize=False, scale=SCALE) == aampdist(a, b, 10, 0.3, scale=SCALE)
    assert mpdist(a, b, 10, k=4, normalize=False) == aampdist(a, b, 10, k=4)
    # p is ignored when normalizing, as in STUMPY.
    assert mpdist(a, b, 10, p=3.0) == mpdist(a, b, 10)


def test_the_two_metrics_see_different_distances():
    a = np.arange(60.0) % 7
    b = a + 100.0
    assert mpdist(a, b, 6, scale_a=1.0, scale_b=1.0) < 1e-6
    assert aampdist(a, b, 6, scale=1.0) > 200.0


def test_aampdist_takes_any_one_dimensional_array_like():
    a, b = list(range(30)), [3, 1, 4, 1, 5, 9, 2, 6] * 4
    want = aampdist(np.array(a, dtype=float), np.array(b, dtype=float), 5, scale=1.0)
    assert aampdist(a, b, 5, scale=1.0) == want
    assert aampdist(np.array(a), np.array(b, dtype=np.int32), 5, scale=1.0) == want
    assert mpdist(a, b, 5, scale_a=1.0, scale_b=1.0) == mpdist(
        np.array(a, dtype=float), np.array(b, dtype=float), 5, scale_a=1.0, scale_b=1.0
    )


def test_aampdist_stumpys_clamps_are_errors_naming_the_explicit_value():
    a, b = grid_noise(300, seed=5), grid_noise(250, seed=6)
    windows_total = (300 - 9) + (250 - 9)
    last = aampdist(a, b, 10, k=windows_total - 1, scale=SCALE)
    assert aampdist(a, b, 10, 1.0, scale=SCALE) == last
    for k in [windows_total, windows_total + 5, 2**62, 2**63 - 1]:
        with pytest.raises(ValueError, match=f"pass k={windows_total - 1} to get that"):
            aampdist(a, b, 10, k=k, scale=SCALE)
    for percentage, clipped in [(2.0, "1.0"), (float("inf"), "1.0"), (-1.0, "0.0"),
                                (float("-inf"), "0.0")]:
        with pytest.raises(ValueError, match=f"pass percentage={clipped} to get that"):
            aampdist(a, b, 10, percentage, scale=SCALE)
        with pytest.raises(ValueError, match=f"pass percentage={clipped} to get that"):
            mpdist(a, b, 10, percentage, normalize=False, scale=SCALE)
    first = aampdist(a, b, 10, k=0, scale=SCALE)
    assert aampdist(a, b, 10, 0.0, scale=SCALE) == first
    for m in [2**62, 2**63 - 1]:
        with pytest.raises(ValueError):
            aampdist(a, b, m, scale=SCALE)


def test_aampdist_bad_arguments():
    a, b = random_walk(80, seed=9), random_walk(70, seed=10)
    for p in (1.0, 3.0, float("inf"), float("nan")):
        with pytest.raises(NotImplementedError):
            aampdist(a, b, 10, p=p)
        with pytest.raises(NotImplementedError):
            mpdist(a, b, 10, normalize=False, p=p)
    for bad in (0, -1, 2):
        with pytest.raises(ValueError):
            aampdist(a, b, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            aampdist(a, b, 10, threads=bad)
    with pytest.raises(ValueError):
        aampdist(a, b, 10, float("nan"))
    with pytest.raises(ValueError):
        aampdist(a, b, 10, k=-1)
    with pytest.raises(ValueError):
        aampdist(a, b, 10, scale=0.0)
    with pytest.raises(ValueError):
        aampdist(a, b[:9], 10, scale=1.0)
    with pytest.raises(ValueError):
        aampdist(a[:9], b, 10)
    with pytest.raises(ValueError):
        mpdist(a, b, 10, normalize=False, scale_a=1.0)
    with pytest.raises(ValueError):
        mpdist(a, b, 10, scale=1.0)


def seam_series():
    """Two series on a 2**-10 grid whose combined scale inference needs a gap between them: a
    zero in place of the gap would put a window of fake values on a coarser grid."""
    e = 2.0**-10
    v = np.floor(4194304 * 0.7 / e) * e + e
    a = np.full(14, v)
    a[3:5] += e
    b = np.full(14, v)
    b[6] = v + e
    b[7] = v - e
    b[1] = v + 2 * e
    return a, b, e


def test_the_seam_between_series_does_not_change_the_inferred_scale():
    a, b, e = seam_series()
    inferred = aampdist(a, b, 3, percentage=1.0)
    assert inferred == aampdist(a, b, 3, percentage=1.0, scale=e)
    assert abs(inferred - 0.0013810679320049757) < 1e-15
    assert abs(inferred - stumpy.aampdist(a, b, 3, percentage=1.0)) < 1e-15


def test_stumpys_positional_arguments_are_in_stumpys_places():
    a, b = grid_noise(80, seed=1), grid_noise(70, seed=2, level=1.0)
    assert mpdist(a, b, 4, 0.05, None, False) == aampdist(a, b, 4, 0.05)
    assert mpdist(a, b, 4, 0.05, None, False, 2.0) == aampdist(a, b, 4, 0.05)
    assert mpdist(a, b, 4, 0.3, 5, True) == mpdist(a, b, 4, 0.3, 5)
    assert aampdist(a, b, 4, 0.05, None, 2.0) == aampdist(a, b, 4)
    # The eighth and ninth places are STUMPY's `T_A_subseq_isconstant` and `T_B_subseq_isconstant`.
    assert mpdist(a, b, 4, 0.05, None, True, 2.0, None, None) == mpdist(a, b, 4)
    with pytest.raises(TypeError):
        mpdist(a, b, 4, 0.05, None, True, 2.0, None, None, 1.0)
    with pytest.raises(TypeError):
        aampdist(a, b, 4, 0.05, None, 2.0, 1.0)


# --- aamp_ostinato --------------------------------------------------------------------------


@pytest.mark.parametrize("count", [3, 4, 5])
@pytest.mark.parametrize("m", [4, 12])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_aamp_ostinato_equals_stumpy_on_sets_with_a_planted_pattern(count, m, seed):
    ts = planted(count, m, seed)
    want = stumpy.aamp_ostinato(ts, m)
    for threads in [1, 3, None]:
        radius, series, window = aamp_ostinato(ts, m, scale=SCALE, threads=threads)
        assert (series, window) == (want[1], want[2]), threads
        assert abs(radius - want[0]) < 1e-6, threads
        assert isinstance(radius, float) and isinstance(series, int) and isinstance(window, int)


@pytest.mark.parametrize("count", [2, 4])
def test_aamp_ostinato_equals_stumpy_on_pure_noise(count):
    ts = [grid_noise(180 + 20 * k, seed=k + 10, level=0.5 * k) for k in range(count)]
    want = stumpy.aamp_ostinato(ts, 10)
    radius, series, window = aamp_ostinato(ts, 10, scale=SCALE)
    assert abs(radius - want[0]) < 1e-6
    if count > 2:
        # A mutual nearest pair of two series is a tie STUMPY breaks by rounding noise.
        assert (series, window) == (want[1], want[2])


def test_aamp_ostinato_equals_stumpy_on_series_with_gaps():
    ts = [
        walk_with_gaps(200, seed=3, gap_every=37),
        walk_with_gaps(170, seed=4, gap_every=29, leading_gap=5),
        random_walk(220, seed=5),
    ]
    want = stumpy.aamp_ostinato(ts, 12)
    radius, series, window = aamp_ostinato(ts, 12, scale=1.0)
    assert abs(radius - want[0]) < 1e-6
    assert (series, window) == (want[1], want[2])


def test_the_scale_is_inferred_from_all_the_series_together():
    ts = planted(3, 8, 7)
    assert aamp_ostinato(ts, 8) == aamp_ostinato(ts, 8, scale=SCALE)


def test_the_inferred_scale_fits_every_series_not_just_the_first():
    ts = [np.round(random_walk(90, seed=1)), grid_noise(80, seed=2), grid_noise(70, seed=3)]
    assert aamp_ostinato(ts, 8) == aamp_ostinato(ts, 8, scale=SCALE)


@pytest.mark.parametrize("seed", range(8))
def test_aamp_ostinato_equals_the_brute_force_on_series_of_ties(seed):
    rng = np.random.default_rng(seed)
    for levels in (2, 3):
        count = int(rng.integers(2, 6))
        ts = [few_levels(int(rng.integers(20, 40)), seed * 10 + i, levels) for i in range(count)]
        for m in [3, 4, 6]:
            want = brute_aamp_ostinato(ts, m, 1.0)
            for threads in [1, 3]:
                got = aamp_ostinato(ts, m, scale=1.0, threads=threads)
                assert (bits(got[0]), got[1], got[2]) == (bits(want[0]), want[1], want[2]), (
                    levels,
                    m,
                    threads,
                )
    cycle = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    periodic = [np.tile(cycle, 8)[s : s + 30 + s] for s in range(4)]
    for m in [3, 5]:
        want = brute_aamp_ostinato(periodic, m, 1.0)
        got = aamp_ostinato(periodic, m, scale=1.0)
        assert (bits(got[0]), got[1], got[2]) == (bits(want[0]), want[1], want[2])


def test_aamp_ostinato_equals_the_brute_force_with_gaps():
    ts = [walk_with_gaps(40, seed=s, gap_every=7 + s, leading_gap=s) for s in range(4)]
    ts[1][-4:] = np.inf
    for m in [3, 4, 6]:
        want = brute_aamp_ostinato(ts, m, 1.0)
        got = aamp_ostinato(ts, m, scale=1.0)
        assert (bits(got[0]), got[1], got[2]) == (bits(want[0]), want[1], want[2]), m


def test_a_single_series_is_its_own_consensus_and_an_unmeetable_set_is_inf():
    assert aamp_ostinato([grid_noise(100, seed=30)], 10, scale=SCALE) == (0.0, 0, 0)
    ts = [random_walk(60, seed=31), np.full(60, np.nan), random_walk(50, seed=32)]
    assert aamp_ostinato(ts, 8, scale=1.0) == (float("inf"), 0, 0)


@pytest.mark.parametrize("gaps", [False, True])
def test_aamp_ostinato_every_thread_count_equals_one_thread_on_bits(gaps):
    if gaps:
        ts = [walk_with_gaps(1500, seed=40 + k, gap_every=211 + 50 * k) for k in range(3)]
    else:
        ts = [random_walk(1500 - 100 * k, seed=40 + k) for k in range(3)]
    first = aamp_ostinato(ts, 24, scale=1.0, threads=1)
    assert np.isfinite(first[0])
    for threads in THREADS:
        got = aamp_ostinato(ts, 24, scale=1.0, threads=threads)
        assert (bits(got[0]), got[1], got[2]) == (bits(first[0]), first[1], first[2]), threads


def test_ostinato_with_normalize_false_is_aamp_ostinato():
    ts = planted(4, 8, 5)
    want = stumpy.ostinato(ts, 8, normalize=False)
    got = ostinato(ts, 8, normalize=False)
    assert (got[1], got[2]) == (want[1], want[2])
    assert abs(got[0] - want[0]) < 1e-6
    assert got == aamp_ostinato(ts, 8)
    assert ostinato(ts, 8, normalize=False, scale=SCALE) == aamp_ostinato(ts, 8, scale=SCALE)
    # p is ignored when normalizing, as in STUMPY.
    assert ostinato(ts, 8, p=3.0) == ostinato(ts, 8)


def test_the_two_metrics_see_different_radii():
    a = np.arange(60.0) % 7
    ts = [a, a + 100.0]
    assert ostinato(ts, 6, scales=[1.0, 1.0])[0] < 1e-6
    assert aamp_ostinato(ts, 6, scale=1.0)[0] > 200.0


def test_aamp_ostinato_takes_any_one_dimensional_array_likes():
    lists = [list(range(30)), [3, 1, 4, 1, 5, 9, 2, 6] * 4, [2, 7, 1, 8] * 8]
    arrays = [np.array(t, dtype=float) for t in lists]
    assert aamp_ostinato(lists, 5, scale=1.0) == aamp_ostinato(arrays, 5, scale=1.0)
    ints = [np.array(t, dtype=np.int64) for t in lists]
    assert aamp_ostinato(ints, 5, scale=1.0) == aamp_ostinato(arrays, 5, scale=1.0)
    assert ostinato(lists, 5, scales=[1.0] * 3) == ostinato(arrays, 5, scales=[1.0] * 3)


def test_aamp_ostinato_bad_arguments():
    ts = [random_walk(80, seed=9), random_walk(70, seed=10)]
    for p in (1.0, 3.0, float("inf"), float("nan")):
        with pytest.raises(NotImplementedError):
            aamp_ostinato(ts, 10, p=p)
        with pytest.raises(NotImplementedError):
            ostinato(ts, 10, normalize=False, p=p)
    for bad in (0, -1, 2, 2**63 - 1):
        with pytest.raises(ValueError):
            aamp_ostinato(ts, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            aamp_ostinato(ts, 10, threads=bad)
    with pytest.raises(ValueError):
        aamp_ostinato([], 10)
    with pytest.raises(ValueError):
        aamp_ostinato(ts, 10, scale=0.0)
    with pytest.raises(ValueError):
        aamp_ostinato([ts[0], ts[1][:9]], 10, scale=1.0)
    with pytest.raises(ValueError):
        ostinato(ts, 10, normalize=False, scales=[1.0, 1.0])
    with pytest.raises(ValueError):
        ostinato(ts, 10, scale=1.0)


def test_the_seam_between_series_does_not_change_the_inferred_scale_of_a_set():
    a, b, e = seam_series()
    set_ = [a, b, a + e]
    assert aamp_ostinato(set_, 3) == aamp_ostinato(set_, 3, scale=e)
    assert aamp_ostinato(set_, 3) == (0.0009765625, 0, 2)


def test_aamp_ostinato_ties_between_windows_are_decided_by_the_join_order():
    # Series 1 joins against series 2, so the tie between windows 0 and 1 of series 2 goes to 0.
    ts = [np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0, 1.0])]
    assert aamp_ostinato(ts, 3, scale=1.0) == (1.4142135623730951, 2, 0)
    assert stumpy.aamp_ostinato(ts, 3)[1:] == (2, 0)


def test_ostinato_ties_between_windows_are_decided_by_the_join_order():
    ts = [
        np.array([1.0, 2.0, 0.0, 0.0, 2.0]),
        np.array([2.0, 0.0, 1.0, 1.0]),
        np.array([1.0, 2.0, 0.0]),
    ]
    radius, series, window = ostinato(ts, 3, scales=[1.0] * 3)
    assert (series, window) == (2, 0)
    assert abs(radius - 2.449489742783178) < 1e-12
    want = stumpy.ostinato(ts, 3)
    assert (series, window) == (want[1], want[2])


def test_stumpys_positional_arguments_are_in_stumpys_places_for_sets():
    ts = planted(3, 4, 5)
    assert ostinato(ts, 4, False) == aamp_ostinato(ts, 4)
    assert ostinato(ts, 4, False, 2.0) == aamp_ostinato(ts, 4)
    assert ostinato(ts, 4, True) == ostinato(ts, 4)
    assert aamp_ostinato(ts, 4, 2.0) == aamp_ostinato(ts, 4)
    # The fifth place is STUMPY's `Ts_subseq_isconstant`.
    assert ostinato(ts, 4, True, 2.0, [None] * 3) == ostinato(ts, 4)
    with pytest.raises(TypeError):
        ostinato(ts, 4, True, 2.0, [None] * 3, [1.0] * 3)
    with pytest.raises(TypeError):
        aamp_ostinato(ts, 4, 2.0, 1.0)


def test_a_set_of_only_gaps_needs_no_scale_and_has_no_consensus_motif():
    # STUMPY zero-fills the gaps and reports (0.0, 0, 0); a window with a gap has no distance here.
    for ts in ([np.full(30, np.nan)] * 2, [np.full(30, np.nan)], [np.full(20, np.inf), np.full(25, np.nan)]):
        assert aamp_ostinato(ts, 5) == (float("inf"), 0, 0)
        assert ostinato(ts, 5) == (float("inf"), 0, 0)
    mixed = [random_walk(40, seed=3), np.full(30, np.nan)]
    assert ostinato(mixed, 5) == (float("inf"), 0, 0)
    assert aamp_ostinato(mixed, 5) == (float("inf"), 0, 0)
