"""`aamp_motifs`, `aamp_match`, `motifs(normalize=False)`, `match`: STUMPY's non-normalized motifs
and matching on tie-free series, an exact integer brute force of the same procedure on series
full of ties and gaps, the thread count never changing a bit, `normalize=False` being the
non-normalized function, and `p` other than 2 being `NotImplementedError`."""

import math
import warnings

import numpy as np
import pytest
from ladders import aamp_match, aamp_motifs, match, motifs

from conftest import profile_for, random_walk, walk_with_gaps

M = 12
SCALE = 1 / 64
THREADS = [1, 2, 7, None]


def grid_walk(n, seed):
    """A tie-free walk exactly on the `SCALE` grid, so STUMPY reads the same values."""
    rng = np.random.default_rng(seed)
    return np.round(np.cumsum(rng.normal(size=n)) * 64) / 64


def u64(x):
    return np.asarray(x).view(np.uint64)


def columns(found):
    """STUMPY's matches as (distances, indices); it returns a bare empty array for none."""
    if found.ndim == 1:
        return np.empty(0), np.empty(0, dtype=np.int64)
    return found[:, 0].astype(float), found[:, 1].astype(np.int64)


def stumpy_motifs(stumpy, t, m, **kwargs):
    profile = stumpy.aamp(t, m)[:, 0].astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stumpy.aamp_motifs(t, profile, **kwargs)


def assert_motifs_agree(ours, theirs):
    assert ours[1].shape == theirs[1].shape
    assert np.array_equal(ours[1], theirs[1].astype(np.int64))
    assert np.array_equal(np.isnan(ours[0]), np.isnan(theirs[0]))
    assert np.allclose(np.nan_to_num(ours[0]), np.nan_to_num(theirs[0]), rtol=1e-12, atol=1e-12)


# --- an exact brute force of the procedure on integer ticks -------------------------------


def brute_ticks(series, scale):
    return [None if not math.isfinite(x) else round(x / scale) for x in series]


def brute_window(ticks, start, m):
    w = ticks[start : start + m]
    return None if any(v is None for v in w) else w


def brute_profile(query, ticks):
    m = len(query)
    out = []
    for i in range(len(ticks) - m + 1):
        w = brute_window(ticks, i, m)
        out.append(None if w is None else sum((a - b) ** 2 for a, b in zip(query, w)))
    return out


def default_threshold(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    n = len(finite)
    mean = 0.0
    for v in finite:
        mean += v
    mean /= n
    var = 0.0
    for v in finite:
        var += (v - mean) * (v - mean)
    var /= n
    return max(mean - 2.0 * math.sqrt(var), min(finite))


def brute_matches(profile, m, scale, max_distance, limit, anchor=None):
    zone = -(-m // 4)
    d = [math.inf if p is None else math.sqrt(p) * scale for p in profile]
    if anchor is not None:
        d[anchor] = 0.0
    threshold = default_threshold(d) if max_distance is None else max_distance
    if threshold is None:
        return []
    alive = [math.isfinite(x) and x <= threshold for x in d]
    nxt = anchor if anchor is not None and alive[anchor] else None
    out = []
    while limit is None or len(out) < limit:
        if nxt is not None:
            best, nxt = nxt, None
        else:
            live = [i for i in range(len(alive)) if alive[i]]
            if not live:
                break
            best = min(live, key=lambda i: (profile[i], i))
        out.append((best, d[best]))
        for i in range(len(alive)):
            if abs(i - best) <= zone:
                alive[i] = False
    return out


def brute_motifs(series, m, scale, min_neighbors=1, max_distance=None, cutoff=None,
                 max_matches=10, max_motifs=1):
    ticks = brute_ticks(series, scale)
    zone = -(-m // 4)
    count = len(ticks) - m + 1
    near = []
    for i in range(count):
        own = brute_window(ticks, i, m)
        best = None
        if own is not None:
            for j in range(count):
                other = brute_window(ticks, j, m)
                if abs(i - j) > zone and other is not None:
                    c = (sum((a - b) ** 2 for a, b in zip(own, other)), j)
                    best = c if best is None or c < best else best
        near.append(best)
    dist = [math.inf if n is None else math.sqrt(n[0]) * scale for n in near]
    excluded = [False] * count

    def pick():
        live = [i for i in range(count) if not excluded[i] and near[i] is not None]
        return min(live, key=lambda i: (near[i][0], i)) if live else None

    first = pick()
    found = []
    if first is None:
        return found
    if cutoff is None:
        cutoff = max(default_threshold(dist), dist[first])
    while len(found) < max_motifs:
        cand = pick()
        if cand is None or dist[cand] > cutoff or not math.isfinite(dist[cand]):
            break
        if max_distance is not None and dist[cand] > max_distance:
            break
        own = brute_window(ticks, cand, m)
        got = brute_matches(brute_profile(own, ticks), m, scale, max_distance, None, cand)
        for at, _ in got or [(cand, 0)]:
            for i in range(count):
                if abs(i - at) <= zone:
                    excluded[i] = True
        if len(got) > min_neighbors:
            found.append(got if max_matches is None else got[:max_matches])
    return found


def as_rows(distances, indices):
    """Motif arrays as lists of `(index, distance)` per motif, padding dropped."""
    if indices.size == 0:
        return []
    return [
        [(int(i), float(d)) for d, i in zip(drow, irow) if i >= 0]
        for drow, irow in zip(distances, indices)
    ]


def tie_series():
    return {
        "constant": np.full(60, 5.0),
        "period 2": np.tile([0.0, 3.0], 30),
        "period 7": np.tile([0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 9),
        "few levels": np.random.default_rng(3).integers(0, 3, size=70).astype(np.float64) * 2,
        "repeats at offsets": np.concatenate(
            [np.arange(10.0) % 4 + o for o in (0, 6, 0, 6, 0)] + [np.zeros(5)]
        ),
        "gaps": walk_with_gaps(80, seed=11, gap_every=11, leading_gap=2),
        "walk": random_walk(80, seed=12),
    }


# --- parity with STUMPY on tie-free series ------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("m", [8, 16, 33])
def test_aamp_motifs_agree_with_stumpy_on_tie_free_series(seed, m):
    stumpy = pytest.importorskip("stumpy")
    t = grid_walk(300 + 90 * seed, seed=seed)
    checked = 0
    for max_motifs in (1, 3, 6):
        for max_matches in (3, 10):
            for kwargs in (
                {"cutoff": np.inf, "max_distance": np.inf},
                {"max_distance": 4.0, "cutoff": 6.0},
                {"min_neighbors": 2, "cutoff": np.inf},
                {},
            ):
                theirs = stumpy_motifs(
                    stumpy, t, m, max_motifs=max_motifs, max_matches=max_matches, **kwargs
                )
                ours = aamp_motifs(
                    t, profile_for(t, m), scale=SCALE, max_motifs=max_motifs, max_matches=max_matches, **kwargs
                )
                if ours[1].size == 0:
                    assert theirs[1].size == 0
                    continue
                assert_motifs_agree(ours, theirs)
                checked += ours[1].size
    assert checked > 0


@pytest.mark.parametrize("seed", range(3))
def test_normalize_false_routes_motifs_to_aamp_motifs_as_stumpy_does(seed):
    stumpy = pytest.importorskip("stumpy")
    t = grid_walk(400, seed=seed + 20)
    profile = stumpy.aamp(t, M)[:, 0].astype(np.float64)
    kwargs = {"max_motifs": 4, "max_matches": 5, "cutoff": np.inf, "max_distance": np.inf}
    routed = motifs(t, profile_for(t, M), scale=SCALE, normalize=False, **kwargs)
    direct = aamp_motifs(t, profile_for(t, M), scale=SCALE, **kwargs)
    assert np.array_equal(u64(routed[0]), u64(direct[0]))
    assert np.array_equal(routed[1], direct[1])
    assert routed[1].size > 0
    theirs = stumpy.motifs(t, profile, normalize=False, **kwargs)
    assert_motifs_agree(routed, theirs)
    # The normalized default is a different function, with different neighbors.
    assert not np.array_equal(motifs(t, profile_for(t, M), scale=SCALE, **kwargs)[1], direct[1])


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("m", [6, 12, 25])
def test_aamp_match_agrees_with_stumpy_on_tie_free_series(seed, m):
    stumpy = pytest.importorskip("stumpy")
    t = grid_walk(400 + 50 * seed, seed=seed + 40)
    rng = np.random.default_rng(seed)
    queries = [t[s : s + m] for s in rng.integers(0, len(t) - m, size=3)]
    queries.append(np.round(rng.normal(size=m).cumsum() * 64) / 64)
    checked = 0
    for q in queries:
        for max_distance in (None, 1.0, 3.5, np.inf):
            for max_matches in (None, 1, 4):
                theirs = stumpy.aamp_match(q, t, max_distance=max_distance, max_matches=max_matches)
                ours = aamp_match(q, t, scale=SCALE, max_distance=max_distance, max_matches=max_matches)
                want_d, want_i = columns(theirs)
                assert np.array_equal(ours[1], want_i)
                assert np.allclose(ours[0], want_d, rtol=1e-12, atol=1e-12)
                checked += len(ours[1])
    assert checked > 0


@pytest.mark.parametrize("seed", range(3))
def test_normalize_false_routes_match_to_aamp_match_as_stumpy_does(seed):
    stumpy = pytest.importorskip("stumpy")
    t = grid_walk(350, seed=seed + 60)
    q = t[100:100 + M]
    routed = match(q, t, scale=SCALE, max_distance=np.inf, max_matches=6, normalize=False)
    direct = aamp_match(q, t, scale=SCALE, max_distance=np.inf, max_matches=6)
    assert np.array_equal(u64(routed[0]), u64(direct[0]))
    assert np.array_equal(routed[1], direct[1])
    theirs = stumpy.match(q, t, max_distance=np.inf, max_matches=6, normalize=False)
    assert np.array_equal(routed[1], columns(theirs)[1])
    assert routed[1][0] == 100 and routed[0][0] == 0.0


@pytest.mark.parametrize("seed", range(4))
def test_match_agrees_with_stumpy_on_tie_free_series(seed):
    stumpy = pytest.importorskip("stumpy")
    t = grid_walk(400, seed=seed + 80)
    for start in (30, 170):
        q = t[start : start + M]
        for max_distance in (1.5, 4.0, np.inf):
            for max_matches in (None, 3):
                theirs = stumpy.match(q, t, max_distance=max_distance, max_matches=max_matches)
                ours = match(q, t, scale=SCALE, max_distance=max_distance, max_matches=max_matches)
                want_d, want_i = columns(theirs)
                assert np.array_equal(ours[1], want_i)
                assert np.allclose(ours[0], want_d, rtol=1e-6, atol=1e-6)
                assert ours[1][0] == start


# --- an exact integer brute force of the procedure -----------------------------------------


@pytest.mark.parametrize("name", list(tie_series()))
def test_aamp_motifs_equal_the_integer_brute_force(name):
    t = tie_series()[name]
    for kwargs in (
        {},
        {"max_motifs": 4, "max_matches": None, "cutoff": np.inf, "max_distance": np.inf},
        {"max_motifs": 3, "min_neighbors": 0, "max_matches": 3, "cutoff": np.inf},
        {"max_motifs": 2, "max_distance": 2.5, "cutoff": 4.0, "min_neighbors": 2},
    ):
        want = brute_motifs(t, 6, 1.0, **kwargs)
        got = aamp_motifs(t, profile_for(t, 6), scale=1.0, threads=1, **kwargs)
        assert [[i for i, _ in row] for row in as_rows(*got)] == [
            [i for i, _ in row] for row in want
        ], (name, kwargs)
        for ours, theirs in zip(as_rows(*got), want):
            assert np.allclose([d for _, d in ours], [d for _, d in theirs], rtol=1e-15, atol=0)


@pytest.mark.parametrize("name", list(tie_series()))
def test_aamp_match_equals_the_integer_brute_force(name):
    t = tie_series()[name]
    ticks = brute_ticks(t, 1.0)
    for start in (0, 7, 20):
        q = brute_window(ticks, start, 6)
        if q is None:
            continue
        profile = brute_profile(q, ticks)
        for max_distance in (None, 0.0, 3.0, np.inf):
            for limit in (None, 1, 3):
                want = brute_matches(profile, 6, 1.0, max_distance, limit)
                got = aamp_match(np.array(q, dtype=float), t, scale=1.0, max_distance=max_distance,
                                 max_matches=limit)
                assert list(got[1]) == [i for i, _ in want], (name, start, max_distance, limit)
                assert np.allclose(got[0], [d for _, d in want], rtol=1e-15, atol=0)


# --- the rest of the contract --------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2])
def test_every_thread_count_gives_the_same_bits(seed):
    t = grid_walk(800, seed=seed)
    runs = [
        aamp_motifs(t, profile_for(t, M), scale=SCALE, threads=n, max_motifs=4, cutoff=np.inf, max_distance=np.inf)
        for n in THREADS
    ]
    assert runs[0][1].size > 0
    for distances, indices in runs[1:]:
        assert np.array_equal(u64(distances), u64(runs[0][0]))
        assert np.array_equal(indices, runs[0][1])


def test_the_first_column_is_each_motifs_own_window_at_distance_zero():
    t = grid_walk(700, seed=3)
    distances, indices = aamp_motifs(
        t, profile_for(t, M), scale=SCALE, max_motifs=5, max_matches=4, cutoff=np.inf, max_distance=np.inf
    )
    assert distances.dtype == np.float64 and indices.dtype == np.int64
    assert np.all(distances[:, 0] == 0.0) and np.all(indices[:, 0] >= 0)
    assert np.array_equal(np.isnan(distances), indices == -1)
    for row in distances:
        found = row[~np.isnan(row)]
        assert np.all(np.diff(found) >= 0)


def test_no_motif_is_shape_one_by_zero():
    distances, indices = aamp_motifs(grid_walk(300, seed=4), profile_for(grid_walk(300, seed=4), M), scale=SCALE, min_neighbors=1000)
    assert distances.shape == indices.shape == (1, 0)


def test_a_max_motifs_below_one_is_an_error_naming_stumpys_one():
    t = grid_walk(500, seed=5)
    kwargs = {"cutoff": np.inf, "max_distance": np.inf}
    for call in [aamp_motifs, lambda *a, **k: motifs(*a, normalize=False, **k)]:
        with pytest.raises(ValueError, match="max_motifs=0 is below 1") as error:
            call(t, profile_for(t, M), scale=SCALE, max_motifs=0, **kwargs)
        assert "pass max_motifs=1 to get that" in str(error.value)
    assert aamp_motifs(t, profile_for(t, M), scale=SCALE, max_motifs=1, **kwargs)[1].shape[0] == 1


def test_scale_none_infers_the_scale_aamp_does():
    t = random_walk(300, seed=6)
    kwargs = {"max_motifs": 3, "cutoff": np.inf, "max_distance": np.inf}
    inferred = aamp_motifs(t, profile_for(t, 12), **kwargs)
    explicit = aamp_motifs(t, profile_for(t, 12), scale=1.0, **kwargs)
    assert np.array_equal(u64(inferred[0]), u64(explicit[0]))
    assert np.array_equal(inferred[1], explicit[1])
    q = t[50:62]
    a, b = aamp_match(q, t), aamp_match(q, t, scale=1.0)
    assert np.array_equal(u64(a[0]), u64(b[0])) and np.array_equal(a[1], b[1])


def test_distances_are_in_the_callers_units():
    t = grid_walk(300, seed=7)
    distances, indices = aamp_motifs(t, profile_for(t, M), scale=SCALE, max_matches=2, cutoff=np.inf, max_distance=np.inf)
    first, second = indices[0][:2]
    want = math.sqrt(float(np.sum((t[first : first + M] - t[second : second + M]) ** 2)))
    assert distances[0][1] == pytest.approx(want, rel=1e-12)
    bigger = aamp_motifs(t * 8, profile_for(t * 8, M), scale=SCALE * 8, max_matches=2, cutoff=np.inf, max_distance=np.inf)
    assert bigger[1][0][1] == second
    assert bigger[0][0][1] == pytest.approx(8 * distances[0][1], rel=1e-12)


@pytest.mark.parametrize("p", [1.0, 3.0, 0.5, float("inf"), float("nan")])
def test_other_values_of_p_are_not_implemented(p):
    t = grid_walk(120, seed=8)
    q = t[:M]
    with pytest.raises(NotImplementedError):
        aamp_motifs(t, profile_for(t, M), scale=SCALE, p=p)
    with pytest.raises(NotImplementedError):
        aamp_match(q, t, scale=SCALE, p=p)
    with pytest.raises(NotImplementedError):
        motifs(t, profile_for(t, M), scale=SCALE, normalize=False, p=p)
    with pytest.raises(NotImplementedError):
        match(q, t, scale=SCALE, normalize=False, p=p)
    # With the default normalize=True, p is ignored, as in STUMPY.
    assert motifs(t, profile_for(t, M), scale=SCALE, p=p)[1].shape[0] == 1
    assert len(match(q, t, scale=SCALE, p=p)[1]) >= 1


def test_a_gap_window_is_never_a_motif_or_a_match():
    t = grid_walk(300, seed=9)
    t[100:104] = np.nan
    distances, indices = aamp_motifs(t, profile_for(t, M), scale=SCALE, max_motifs=6, cutoff=np.inf, max_distance=np.inf)
    gap = set(range(100 - M + 1, 104))
    assert not gap & set(indices[indices >= 0].ravel().tolist())
    d, i = aamp_match(t[10 : 10 + M], t, scale=SCALE, max_distance=np.inf)
    assert not gap & set(i.tolist())
    assert i[0] == 10


@pytest.mark.parametrize(
    "call",
    [
        lambda t: aamp_motifs(t, profile_for(t, M), scale=SCALE, min_neighbors=-1),
        lambda t: aamp_motifs(t, profile_for(t, M), scale=SCALE, max_matches=-1),
        lambda t: aamp_motifs(t, profile_for(t, M), scale=SCALE, threads=0),
        lambda t: aamp_motifs(t, profile_for(t, 2), scale=SCALE),
        lambda t: aamp_motifs(t[:5], profile_for(t[:5], M)),
        lambda t: aamp_match(t[:M], t, scale=SCALE, max_matches=-1),
        lambda t: aamp_match(t[:2], t, scale=SCALE),
        lambda t: aamp_match(np.full(M, np.nan), t, scale=SCALE),
        lambda t: aamp_match(np.r_[t, t][: len(t) + 1], t, scale=SCALE),
        lambda t: aamp_match(t[:M], t, scale=-1.0),
        lambda t: match(t[:M], t, scale=SCALE, max_matches=-1),
        lambda t: match(np.full(M, np.inf), t, scale=SCALE),
        lambda t: match(np.r_[t, t][: len(t) + 1], t, scale=SCALE),
    ],
)
def test_a_bad_argument_is_a_value_error(call):
    with pytest.raises(ValueError):
        call(grid_walk(200, seed=10))


def test_a_match_shorter_series_or_an_all_gap_series_has_no_matches():
    gap = np.full(30, np.nan)
    q = np.arange(M, dtype=float)
    d, i = aamp_match(q, np.r_[gap, np.arange(5.0)], scale=1.0)
    assert len(d) == len(i) == 0
    assert len(aamp_match(q, gap)[1]) == 0
    assert aamp_motifs(np.full(40, np.nan), profile_for(np.full(40, np.nan), M), scale=1.0)[1].shape == (1, 0)


def test_match_is_the_normalized_function_and_aamp_match_the_raw_one():
    shape = np.array([3.0, -7.0, 5.0, 1.0, -2.0, 8.0, -4.0])
    rng = np.random.default_rng(1)
    t = np.round(rng.normal(size=60) * 20)
    t[10:17] = shape
    t[40:47] = shape + 100.0
    z = match(shape, t, scale=1.0, max_distance=1e-6)
    raw = aamp_match(shape, t, scale=1.0, max_distance=1e-6)
    assert sorted(z[1].tolist()) == [10, 40]
    assert raw[1].tolist() == [10]


def test_stumpys_keyword_names_and_plain_array_likes_are_accepted():
    t = grid_walk(200, seed=12)
    q = t[20 : 20 + M]
    kwargs = {"max_distance": np.inf, "max_matches": 4}
    want = aamp_match(q, t, scale=SCALE, **kwargs)
    for got in (
        aamp_match(Q=q.tolist(), T=t.tolist(), scale=SCALE, **kwargs),
        match(Q=q, T=t, scale=SCALE, normalize=False, **kwargs),
    ):
        assert np.array_equal(u64(got[0]), u64(want[0])) and np.array_equal(got[1], want[1])
    ints = np.arange(40, dtype=np.int64) % 7
    a = aamp_match(Q=ints[:6].tolist(), T=ints.tolist(), scale=1.0)
    b = aamp_match(ints[:6].astype(float), ints.astype(float), scale=1.0)
    assert np.array_equal(a[1], b[1])
    for dtype in (np.int64, np.int32, np.float32):
        q, s = ints[:6].astype(dtype), ints.astype(dtype)
        assert np.array_equal(aamp_match(q, s, scale=1.0)[1], b[1]), dtype
        assert np.array_equal(match(q, s, scale=1.0, normalize=False)[1], b[1]), dtype
        assert match(q, s, scale=1.0)[1].size > 0, dtype
        for call in (aamp_motifs, motifs):
            got = call(s, profile_for(s, 6), scale=1.0, max_motifs=2, cutoff=np.inf)
            assert np.array_equal(got[1], call(ints.astype(float), profile_for(ints, 6), scale=1.0, max_motifs=2, cutoff=np.inf)[1]), dtype
    ours = aamp_motifs(T=t.tolist(), P=profile_for(t, M).tolist(), scale=SCALE, max_motifs=2, cutoff=np.inf)
    assert np.array_equal(ours[1], aamp_motifs(t, profile_for(t, M), scale=SCALE, max_motifs=2, cutoff=np.inf)[1])
    z = motifs(T=t.tolist(), P=profile_for(t, M).tolist(), scale=SCALE, max_motifs=2, cutoff=np.inf)
    assert np.array_equal(z[1], motifs(t, profile_for(t, M), scale=SCALE, max_motifs=2, cutoff=np.inf)[1])
