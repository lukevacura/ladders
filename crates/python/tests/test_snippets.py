"""`snippets(T, m, k, ...)`: STUMPY's snippets, every distance order exact, the thread count never
changing a bit, and bad arguments a `ValueError`."""

import numpy as np
import pytest
import math

from ladders import aampdist_snippets, snippets

from conftest import random_walk, walk_with_gaps

stumpy = pytest.importorskip("stumpy")

THREADS = [1, 2, 3, 8, None, np.int64(4)]
NAMES = ["snippets", "indices", "profiles", "fractions", "areas", "regimes"]


def grid_noise(n: int, seed: int, level: float = 0.0) -> np.ndarray:
    """Gaussian samples on a 1/64 grid: tie-free in practice, exact at `scale=1/64`."""
    rng = np.random.default_rng(seed)
    return level + np.round(rng.normal(size=n) * 64.0) / 64.0


def u64(x):
    return np.asarray(x).view(np.uint64)


def assert_equals_stumpy(got, want, context):
    for name, a, b in zip(NAMES, got, want):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape and a.dtype == b.dtype, (name, context)
        if a.dtype.kind in "iu":
            assert np.array_equal(a, b), (name, context)
        else:
            assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), (name, context)


def stumpy_explicit(m, kwargs):
    """`kwargs` with what STUMPY clamps made explicit: the shares clipped to `[0, 1]`, `s` lowered
    to `m` and `mpdist_k` lowered to the last of the `2(m - s + 1)` joined distances."""
    explicit = dict(kwargs)
    for name in ("percentage", "mpdist_percentage"):
        if name in explicit:
            explicit[name] = min(max(explicit[name], 0.0), 1.0)
    if "s" in explicit:
        explicit["s"] = min(explicit["s"], m)
    s = explicit.get("s", min(math.ceil(explicit.get("percentage", 1.0) * m), m))
    if "mpdist_k" in explicit:
        explicit["mpdist_k"] = min(explicit["mpdist_k"], 2 * (m - s + 1) - 1)
    return explicit


def run_as_stumpy_does(call, t, m, k, kwargs):
    """`call` with `kwargs` made explicit, after checking that `kwargs` as given is refused where
    STUMPY would clamp them."""
    explicit = stumpy_explicit(m, kwargs)
    if explicit != kwargs:
        with pytest.raises(ValueError, match="STUMPY"):
            call(t, m, k, scale=1 / 64, **kwargs)
    return call(t, m, k, scale=1 / 64, **explicit)


@pytest.mark.parametrize(
    "n, m, k, kwargs",
    [
        (300, 20, 3, {}),
        (300, 20, 3, {"percentage": 0.5}),
        (331, 25, 4, {"s": 10, "mpdist_percentage": 0.1}),
        (301, 10, 2, {"mpdist_k": 7}),
        (500, 40, 5, {"percentage": 0.25, "mpdist_percentage": 0.3}),
        (240, 12, 6, {"s": 5}),
        (240, 12, 3, {"s": 3}),
        (240, 12, 3, {"s": 40}),
        (100, 30, 9, {}),
        (96, 3, 4, {}),
    ],
)
def test_equals_stumpy_on_tie_free_series(n, m, k, kwargs):
    t = grid_noise(n, seed=n + m)
    got = run_as_stumpy_does(snippets, t, m, k, kwargs)
    assert_equals_stumpy(got, stumpy.snippets(t, m, k, **kwargs), (n, m, k, kwargs))


def test_equals_stumpy_on_series_with_gaps():
    t = walk_with_gaps(400, seed=3, gap_every=37, leading_gap=5)
    got = snippets(t, 20, 3, scale=1.0)
    assert_equals_stumpy(got, stumpy.snippets(t, 20, 3), "gaps")


def test_the_stumpy_documentation_example():
    t = np.array([584.0, -11.0, 23.0, 79.0, 1001.0, 0.0, -19.0])
    got = snippets(t, 3, 2)
    assert_equals_stumpy(got, stumpy.snippets(t, 3, 2), "doc example")
    assert got[1].tolist() == [0, 3]
    assert got[3].tolist() == [0.6, 0.4]
    assert got[5].tolist() == [[0, 0, 1], [0, 2, 3], [0, 4, 5], [1, 1, 2], [1, 3, 4]]


def test_shapes_and_dtypes():
    t = grid_noise(210, seed=1)
    snips, indices, profiles, fractions, areas, regimes = snippets(t, 30, 4, scale=1 / 64)
    assert snips.shape == (4, 30) and snips.dtype == np.float64
    assert indices.shape == (4,) and indices.dtype == np.int64
    assert profiles.shape == (4, 181) and profiles.dtype == np.float64
    assert fractions.shape == (4,) and areas.shape == (4,)
    assert regimes.ndim == 2 and regimes.shape[1] == 3 and regimes.dtype == np.int64
    for i, start in enumerate(indices):
        assert np.array_equal(snips[i], t[start : start + 30])


def test_k_beyond_the_segment_count_repeats_like_stumpy():
    t = grid_noise(90, seed=2)
    got = snippets(t, 30, 7, scale=1 / 64)
    assert_equals_stumpy(got, stumpy.snippets(t, 30, 7), "k > segments")


@pytest.mark.parametrize("gaps", [False, True])
def test_every_thread_count_equals_one_thread_on_bits(gaps):
    t = walk_with_gaps(1500, seed=40, gap_every=211) if gaps else random_walk(1500, seed=40)
    first = snippets(t, 50, 4, 0.5, scale=1.0, threads=1)
    for threads in THREADS:
        got = snippets(t, 50, 4, 0.5, scale=1.0, threads=threads)
        for name, a, b in zip(NAMES, got, first):
            assert a.tobytes() == b.tobytes(), (name, threads)


def test_the_scale_is_inferred_when_not_given():
    t = grid_noise(200, seed=9)
    assert_equals_stumpy(snippets(t, 20, 3), stumpy.snippets(t, 20, 3), "inferred")


def test_series_without_any_distance_gives_infinite_profiles_without_failing():
    t = np.full(60, np.nan)
    t[:3] = [1.0, 2.0, 4.0]
    got = snippets(t, 10, 2, scale=1.0)
    assert np.isinf(got[2]).all()


def test_bad_arguments_are_value_errors():
    t = random_walk(80, seed=9)
    for bad in (0, -1, 2):
        with pytest.raises(ValueError):
            snippets(t, bad, 2)
    for bad in (41, 1000):
        with pytest.raises(ValueError):
            snippets(t, bad, 2)
    for bad in (0, -3):
        with pytest.raises(ValueError):
            snippets(t, 10, bad)
    for bad in (0, -2, True, 1.5):
        with pytest.raises(ValueError):
            snippets(t, 10, 2, threads=bad)
    for bad in (0, -1, 1, 2):
        with pytest.raises(ValueError):
            snippets(t, 10, 2, s=bad)
    snippets(t, 10, 2, s=3, scale=1.0)
    with pytest.raises(ValueError):
        snippets(t, 10, 2, percentage=0.2)
    snippets(t, 10, 2, percentage=0.3, scale=1.0)
    with pytest.raises(ValueError):
        snippets(t, 10, 2, percentage=0.0)
    with pytest.raises(ValueError):
        snippets(t, 10, 2, percentage=float("nan"))
    with pytest.raises(ValueError):
        snippets(t, 10, 2, mpdist_percentage=float("nan"))
    with pytest.raises(ValueError):
        snippets(t, 10, 2, mpdist_k=-1)
    with pytest.raises(ValueError):
        snippets(t, 10, 2, scale=0.0)
    with pytest.raises(ValueError):
        snippets(t[:19], 10, 2, scale=1.0)
    with pytest.raises(ValueError):
        snippets(np.full(30, np.nan), 10, 2)


def test_stumpys_clamps_are_errors_naming_the_explicit_value():
    t = grid_noise(120, seed=4)
    kw = dict(scale=1 / 64)
    cases = [
        ({"percentage": 7.0}, "percentage=7.0 is outside [0, 1]", "pass percentage=1.0 to get that"),
        ({"percentage": -0.5}, "percentage=-0.5 is outside [0, 1]", "pass percentage=0.0 to get that"),
        ({"mpdist_percentage": 9.0}, "mpdist_percentage=9.0 is outside", "pass mpdist_percentage=1.0"),
        ({"mpdist_k": 10**6}, "mpdist_k=1000000 is past the last position (1)", "pass mpdist_k=1 "),
        ({"s": 10, "mpdist_k": 22}, "mpdist_k=22 is past the last position (21)", "pass mpdist_k=21 "),
        ({"s": 10**9}, "s=1000000000 is above m=20", "pass s=20 to get that"),
        ({"s": 21}, "s=21 is above m=20", "pass s=20 to get that"),
    ]
    for kwargs, named, remedy in cases:
        for call in (snippets, aampdist_snippets):
            with pytest.raises(ValueError) as error:
                call(t, 20, 2, **kwargs, **kw)
            assert named in str(error.value) and remedy in str(error.value), str(error.value)
    assert_equals_stumpy(
        snippets(t, 20, 2, percentage=1.0, mpdist_percentage=1.0, **kw),
        stumpy.snippets(t, 20, 2, percentage=7.0, mpdist_percentage=9.0),
        "clipped shares",
    )
    last = snippets(t, 20, 2, mpdist_k=1, **kw)
    assert_equals_stumpy(last, stumpy.snippets(t, 20, 2, mpdist_k=10**6), "huge mpdist_k")
    assert snippets(t, 20, 2, s=20, **kw)[1].tolist() == snippets(t, 20, 2, **kw)[1].tolist()
    assert snippets(t, 20, 2, s=10, mpdist_k=21, **kw)[1].shape == (2,)


# --- aampdist_snippets: the same procedure on the non-normalized metric -------------------


def brute_profile_value(segment, window, s, position, scale):
    """STUMPY's value at the selected position of the nearest squared tick distances between a
    segment's `s`-windows and a window's, a gap window having none."""
    j = len(segment) - s + 1

    def sub(w, at):
        part = w[at : at + s]
        return None if any(v is None for v in part) else part

    def near(src, dst, a):
        own = sub(src, a)
        if own is None:
            return None
        best = None
        for b in range(j):
            other = sub(dst, b)
            if other is not None:
                d = sum((x - y) ** 2 for x, y in zip(own, other))
                best = d if best is None else min(best, d)
        return best

    values = [near(segment, window, a) for a in range(j)]
    values += [near(window, segment, b) for b in range(j)]
    values = sorted(v for v in values if v is not None)
    if not values:
        return math.inf
    chosen = values[position] if position < len(values) else values[-1]
    return math.sqrt(float(chosen)) * scale


def brute_aamp_profiles(t, m, s, position, scale, drop_last):
    ticks = [None if not math.isfinite(x) else round(x / scale) for x in t]
    segments = len(t) // m - (1 if drop_last and len(t) % m == 0 else 0)
    return np.array(
        [
            [
                brute_profile_value(ticks[i * m : (i + 1) * m], ticks[p : p + m], s, position, scale)
                for p in range(len(t) - m + 1)
            ]
            for i in range(segments)
        ]
    )


def tie_series():
    return {
        "constant": np.full(40, 5.0),
        "period 2": np.tile([0.0, 3.0], 20),
        "period 7": np.tile([0.0, 3.0, 1.0, 4.0, 1.0, 5.0, 9.0], 6),
        "few levels": np.random.default_rng(3).integers(0, 3, size=44).astype(np.float64) * 2,
        "gaps": walk_with_gaps(48, seed=11, gap_every=11, leading_gap=2),
        "infinite": np.where(np.arange(42) % 13 == 4, np.inf, random_walk(42, seed=5)),
    }


@pytest.mark.parametrize("name", list(tie_series()))
@pytest.mark.parametrize("m, s", [(6, 6), (7, 4), (12, 5)])
def test_aampdist_snippets_profiles_are_the_integer_brute_force(name, m, s):
    t = tie_series()[name]
    rank = min(2, 2 * (m - s + 1) - 1)
    got = aampdist_snippets(t, m, 3, s=s, mpdist_k=rank, scale=1.0)
    want = brute_aamp_profiles(t, m, s, rank, 1.0, drop_last=True)
    picked = got[1] // m
    assert np.array_equal(u64(got[2]), u64(want[picked])), (name, m, s)
    assert got[0].shape == (3, m)
    assert np.array_equal(got[0], np.array([t[i : i + m] for i in got[1]]), equal_nan=True)


@pytest.mark.parametrize(
    "n, m, k, kwargs",
    [
        (300, 20, 3, {}),
        (301, 20, 3, {"percentage": 0.5}),
        (331, 25, 4, {"s": 10, "mpdist_percentage": 0.1}),
        (301, 10, 2, {"mpdist_k": 7}),
        (500, 40, 5, {"percentage": 0.25, "mpdist_percentage": 0.3}),
        (240, 12, 6, {"s": 5}),
        (240, 12, 3, {"s": 40}),
        (100, 30, 9, {}),
        (96, 3, 4, {}),
        (60, 10, 5, {}),
    ],
)
def test_aampdist_snippets_equal_stumpy_on_tie_free_series(n, m, k, kwargs):
    t = grid_noise(n, seed=n + m)
    got = run_as_stumpy_does(aampdist_snippets, t, m, k, kwargs)
    want = stumpy.aampdist_snippets(t, m, k, **kwargs)
    assert_equals_stumpy(got, want, (n, m, k, kwargs))


def test_aampdist_snippets_drop_the_last_whole_segment_as_stumpy_does():
    t = grid_noise(60, seed=1)
    got = aampdist_snippets(t, 10, 6, scale=1 / 64)
    assert set(got[1] // 10) <= {0, 1, 2, 3, 4}
    z = snippets(t, 10, 6, scale=1 / 64)
    assert len(set(z[1] // 10)) > 0
    longer = aampdist_snippets(grid_noise(61, seed=1), 10, 6, scale=1 / 64)
    assert set(longer[1] // 10) <= {0, 1, 2, 3, 4, 5}


def test_a_gap_window_has_an_infinite_profile_value_where_stumpy_has_nan():
    """STUMPY's profile is NaN where a window holds a gap, which turns its areas to NaN and makes
    `argmin` pick the first NaN segment; here such a value is infinite and loses the pick."""
    t = grid_noise(200, seed=3)
    t[93] = np.nan
    ours = aampdist_snippets(t, 20, 2, scale=1 / 64)
    theirs = stumpy.aampdist_snippets(t, 20, 2)
    assert np.isnan(theirs[2]).any() and not np.isnan(ours[2]).any()
    for i, profile in zip(ours[1], ours[2]):
        assert np.isinf(profile[74:94]).all()
        assert np.isfinite(profile[:74]).all()
        assert i != 80


def test_normalize_false_routes_snippets_to_aampdist_snippets_as_stumpy_does():
    t = grid_noise(300, seed=2)
    via_flag = snippets(t, 20, 3, normalize=False, scale=1 / 64)
    direct = aampdist_snippets(t, 20, 3, scale=1 / 64)
    for a, b in zip(via_flag, direct):
        assert np.array_equal(u64(a) if a.dtype.kind == "f" else a, u64(b) if b.dtype.kind == "f" else b)
    assert_equals_stumpy(via_flag, stumpy.snippets(t, 20, 3, normalize=False), "routing")
    normalized = snippets(t, 20, 3, scale=1 / 64)
    assert not np.array_equal(u64(normalized[2]), u64(via_flag[2]))


def test_aampdist_snippets_are_in_the_callers_units_and_thread_invariant():
    t = grid_noise(400, seed=6)
    base = aampdist_snippets(t, 25, 4, 0.5, scale=1 / 64, threads=1)
    for threads in THREADS:
        got = aampdist_snippets(t, 25, 4, 0.5, scale=1 / 64, threads=threads)
        for a, b in zip(base, got):
            assert np.array_equal(u64(a) if a.dtype.kind == "f" else a, u64(b) if b.dtype.kind == "f" else b)
    scaled = aampdist_snippets(t * 8.0, 25, 4, 0.5, scale=1 / 8, threads=1)
    assert np.array_equal(base[1], scaled[1])
    assert np.array_equal(u64(base[2] * 8.0), u64(scaled[2]))


@pytest.mark.parametrize("p", [1.0, 3.0, 0.5, float("inf"), float("nan")])
def test_aampdist_snippets_p_other_than_two_is_not_implemented(p):
    t = grid_noise(80, seed=2)
    with pytest.raises(NotImplementedError):
        aampdist_snippets(t, 10, 2, p=p)
    with pytest.raises(NotImplementedError):
        snippets(t, 10, 2, normalize=False, p=p)
    base = snippets(t, 10, 2, scale=1 / 64)
    ignored = snippets(t, 10, 2, p=p, scale=1 / 64)
    assert np.array_equal(u64(base[2]), u64(ignored[2]))


def test_aampdist_snippets_take_stumpys_keywords_and_array_likes():
    t = grid_noise(120, seed=4)
    base = aampdist_snippets(t, 10, 2, scale=1 / 64)
    listed = aampdist_snippets(list(t), 10, 2, scale=1 / 64)
    assert np.array_equal(u64(base[2]), u64(listed[2]))
    named = aampdist_snippets(
        T=t, m=10, k=2, percentage=1.0, s=None, mpdist_percentage=0.05, mpdist_k=None, p=2.0,
        scale=1 / 64,
    )
    assert np.array_equal(u64(base[2]), u64(named[2]))
    assert aampdist_snippets(np.arange(60) % 7, 10, 2, scale=1.0)[0].shape == (2, 10)


def test_aampdist_snippets_bad_arguments_are_value_errors():
    t = random_walk(80, seed=9)
    for kwargs in [{"m": 0}, {"m": 41}, {"k": 0}, {"threads": 0}, {"s": 2}, {"scale": 0.0},
                   {"mpdist_k": -1}, {"mpdist_percentage": float("nan")}]:
        arguments = {"m": 10, "k": 2, "scale": 1.0}
        arguments.update(kwargs)
        with pytest.raises(ValueError):
            aampdist_snippets(t, **arguments)
    got = aampdist_snippets(np.full(60, np.nan), 10, 2, scale=1.0)
    assert np.isinf(got[2]).all()
