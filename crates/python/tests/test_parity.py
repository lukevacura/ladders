"""The Python surface against STUMPY 1.14.1's: `ladders.stumpy` has every name STUMPY
exports but the warm start with STUMPY's exact signature (positional order, keywords and defaults), and every native
function that takes an array takes lists and int64, int32 and float32 arrays and answers the same
on bits as for float64.
"""

import inspect
import types
import warnings

import ladders as mp
import ladders.stumpy as drop_in
import numpy as np
import pytest

from conftest import random_walk

stumpy = pytest.importorskip("stumpy")


def stumpy_function(name):
    found = getattr(stumpy, name, None)
    return None if isinstance(found, types.ModuleType) else found


def stumpy_names():
    return sorted(name for name in dir(stumpy) if not name.startswith("_") and callable(stumpy_function(name)) and getattr(stumpy_function(name), "__module__", "").startswith("stumpy"))


# STUMPY's exact warm start is deliberately absent from the drop-in: scrump is exact once converged.
OMITTED = {"prescraamp", "prescrump"}


def test_the_drop_in_has_every_stumpy_name_but_the_warm_start():
    assert {name for name in stumpy_names() if not hasattr(drop_in, name)} == OMITTED


def test_the_warm_start_is_absent_from_the_drop_in():
    assert not hasattr(drop_in, "prescrump")
    assert not hasattr(drop_in, "prescraamp")


@pytest.mark.parametrize("name", [name for name in stumpy_names() if name not in OMITTED])
def test_the_drop_in_signature_is_stumpys(name):
    assert inspect.signature(getattr(drop_in, name)) == inspect.signature(stumpy_function(name))


# Every array argument as a NumPy array of each dtype and as a list; values are small whole
# numbers, so every conversion is exact and the answers must not differ at all.
FLOAT_FORMS = {
    "int64": lambda a: np.asarray(a).astype(np.int64),
    "int32": lambda a: np.asarray(a).astype(np.int32),
    "float32": lambda a: np.asarray(a).astype(np.float32),
    "list": lambda a: np.asarray(a).tolist(),
}
INDEX_FORMS = {
    "int64": lambda a: np.asarray(a).astype(np.int64),
    "int32": lambda a: np.asarray(a).astype(np.int32),
    "list": lambda a: np.asarray(a).tolist(),
}

M = 8
N = 90


def series(seed, n=N):
    return np.round(random_walk(n, seed) * 3.0)


def rows(seed, d=3, n=N):
    return np.vstack([series(seed + k, n) for k in range(d)])


def flatten(result):
    if isinstance(result, np.ndarray):
        return [result]
    if isinstance(result, (tuple, list)):
        return [part for item in result for part in flatten(item)]
    return [np.asarray(result)]


def same(a, b):
    left, right = flatten(a), flatten(b)
    assert len(left) == len(right)
    for x, y in zip(left, right):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(x, y)


T = series(1)
OTHER = series(2)
Q = T[10 : 10 + M].copy()
ROWS = rows(5)
BASE_P, BASE_I, BASE_LEFT, BASE_RIGHT = mp.matrix_profile(T, M)
PAIR = ([3, 20, 41], [50, 9, 60])
MPROFILE = mp.mstump(ROWS, M)
MAAMP = mp.maamp(ROWS, M)


def stream_result(stream, values):
    out = stream.update_many(values)
    return out


def stimp_result(make):
    pan = make()
    pan.update()
    return [pan.M_, pan.P_, pan.PI_]


def stumpi_result(make, values):
    stream = make()
    for x in values:
        stream.update(x)
    return [stream.left_P_, stream.left_I_]


# Each recipe takes `f` (turns a float array into the form under test) and `g` (the same for
# index arrays) and returns whatever the call gives.
RECIPES = {
    "aamp": lambda f, g: mp.aamp(f(T), M),
    "aamp_with_a_second_series": lambda f, g: mp.aamp(f(T), M, f(OTHER)),
    "aamp_match": lambda f, g: mp.aamp_match(f(Q), f(T)),
    "match": lambda f, g: mp.match(f(Q), f(T)),
    "match_not_normalized": lambda f, g: mp.match(f(Q), f(T), normalize=False),
    "aamp_motifs": lambda f, g: mp.aamp_motifs(f(T), f(np.zeros(N - M + 1)), max_motifs=3, cutoff=np.inf),
    "motifs": lambda f, g: mp.motifs(f(T), f(np.zeros(N - M + 1)), max_motifs=3, cutoff=np.inf),
    "aamp_ostinato": lambda f, g: mp.aamp_ostinato([f(T), f(OTHER), f(series(3))], M),
    "ostinato": lambda f, g: mp.ostinato([f(T), f(OTHER), f(series(3))], M),
    "aampdist": lambda f, g: mp.aampdist(f(T), f(OTHER), M),
    "mpdist": lambda f, g: mp.mpdist(f(T), f(OTHER), M),
    "aampdist_snippets": lambda f, g: mp.aampdist_snippets(f(T), 12, 2),
    "snippets": lambda f, g: mp.snippets(f(T), 12, 2),
    "aamp_stimp": lambda f, g: stimp_result(lambda: mp.aamp_stimp(f(T), 4, 10)),
    "stimp": lambda f, g: stimp_result(lambda: mp.stimp(f(T), 4, 10)),
    "aampi": lambda f, g: stumpi_result(lambda: mp.aampi(f(T[:40]), M), T[40:50]),
    "stumpi": lambda f, g: stumpi_result(lambda: mp.stumpi(f(T[:40]), M), T[40:50]),
    "stumpi_growing": lambda f, g: stumpi_result(
        lambda: mp.stumpi(f(T[:40]), M, egress=False), T[40:50]
    ),
    "allc": lambda f, g: mp.allc(g(BASE_LEFT), g(BASE_RIGHT)),
    "atsc": lambda f, g: mp.atsc(g(BASE_LEFT), g(BASE_RIGHT), 1),
    "fluss": lambda f, g: mp.fluss(g(BASE_I), 12, 3),
    "fluss_series": lambda f, g: mp.fluss_series(f(T), M, 12, 3),
    "mmotifs": lambda f, g: mp.mmotifs(f(ROWS), MPROFILE[0], g(MPROFILE[1]), include=g([0])),
    "aamp_mmotifs": lambda f, g: mp.aamp_mmotifs(f(ROWS), MAAMP[0], g(MAAMP[1]), include=g([0])),
    "maamp": lambda f, g: mp.maamp(f(ROWS), M, g([0])),
    "mstump": lambda f, g: mp.mstump(f(ROWS), M, g([0])),
    "mdl": lambda f, g: mp.mdl(f(ROWS), M, g(PAIR[0]), g(PAIR[1]), g([0])),
    "maamp_mdl": lambda f, g: mp.maamp_mdl(f(ROWS), M, g(PAIR[0]), g(PAIR[1]), g([0])),
    "subspace": lambda f, g: mp.subspace(f(ROWS), M, 3, 50, 1, g([0])),
    "maamp_subspace": lambda f, g: mp.maamp_subspace(f(ROWS), M, 3, 50, 1, g([0])),
    "ab_join": lambda f, g: mp.ab_join(f(T), f(OTHER), M),
    "ab_join_topk": lambda f, g: mp.ab_join(f(T), f(OTHER), M, k=2),
    "chains": lambda f, g: mp.chains(f(T), M),
    "discords": lambda f, g: mp.discords(f(T), M, 2),
    "distance_profile": lambda f, g: mp.distance_profile(f(Q), f(T), 1.0),
    "infer_scale": lambda f, g: mp.infer_scale(f(T), M),
    "infer_scale_with_several": lambda f, g: mp.infer_scale([f(T), f(OTHER)], M),
    "left_profile": lambda f, g: mp.left_profile(f(T), M),
    "left_profile_topk": lambda f, g: mp.left_profile(f(T), M, k=2),
    "left_profile_report": lambda f, g: mp.left_profile_report(f(T), M),
    "matrix_profile": lambda f, g: mp.matrix_profile(f(T), M),
    "matrix_profile_topk": lambda f, g: mp.matrix_profile(f(T), M, k=2),
    "growing_stream": lambda f, g: stream_result(mp.GrowingStream(M, 1.0), f(T)),
    "bounded_stream": lambda f, g: stream_result(mp.BoundedStream(M, 30, 1.0), f(T)),
    "discord_stream": lambda f, g: discord_result(f(T)),
    "floss_stream": lambda f, g: floss_result(f(T)),
    "pattern_library": lambda f, g: pattern_result(f(Q), f(T)),
    "pattern_stream": lambda f, g: pattern_stream_result(f(Q), f(T)),
}  # fmt: skip


def discord_result(values):
    stream = mp.DiscordStream(M, 2, 1.0)
    stream.update_many(values)
    return stream.top()


def floss_result(values):
    stream = mp.FlossStream(M, 40, 1.0, w=40)
    stream.update_many(values)
    return stream.curve()


def pattern_result(query, values):
    library = mp.PatternLibrary(M, 1.0)
    library.register(query)
    return library.distance_profile(0, values)


def pattern_stream_result(query, values):
    library = mp.PatternLibrary(M, 1.0)
    library.register(query)
    return mp.PatternStream(library).update_many(values)


def baseline(recipe):
    return recipe(lambda a: np.asarray(a, dtype=np.float64), lambda a: np.asarray(a, dtype=np.int64))


@pytest.mark.parametrize("form", FLOAT_FORMS)
@pytest.mark.parametrize("name", sorted(RECIPES))
def test_arrays_of_every_dtype_and_lists_answer_as_float64_does(name, form):
    recipe = RECIPES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = baseline(recipe)
        got = recipe(FLOAT_FORMS[form], INDEX_FORMS[form if form in INDEX_FORMS else "int64"])
    same(got, want)


@pytest.mark.parametrize("form", ["int32", "list"])
@pytest.mark.parametrize("name", sorted(RECIPES))
def test_index_arguments_take_int32_arrays_and_lists_too(name, form):
    recipe = RECIPES[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = baseline(recipe)
        got = recipe(lambda a: np.asarray(a, dtype=np.float64), INDEX_FORMS[form])
    same(got, want)


def test_every_function_that_takes_an_array_has_a_recipe():
    functions = {
        name
        for name in dir(mp)
        if not name.startswith("_") and isinstance(getattr(mp, name), types.BuiltinFunctionType)
    }
    taken = {name.split("_with_")[0].split("_topk")[0].split("_not_normalized")[0] for name in RECIPES}
    taken |= {"fluss_series"}
    assert functions - taken == set()


# STUMPY's `match` also takes the window the query came from and a tolerance.
def stumpy_match(q, t, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return stumpy.match(q, t, **kwargs)


def tie_free(n, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=n))


def scale_for(normalize):
    """The z-normalized distance does not depend on the tick (inferred); the raw one needs a fine one."""
    return None if normalize else 1e-9


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("at", [0, 17, 61])
def test_query_idx_takes_the_window_first_as_stumpys_does(normalize, at):
    t = tie_free(120, 7)
    q = t[at : at + 10].copy()
    for max_distance in (None, np.inf):
        want = stumpy_match(
            q, t, query_idx=at, normalize=normalize, max_distance=max_distance, max_matches=6
        )
        got = mp.match(
            q, t, query_idx=at, normalize=normalize, max_distance=max_distance, max_matches=6,
            scale=scale_for(normalize),
        )  # fmt: skip
        assert got[1][0] == at
        assert max_distance is None or len(got[1]) > 1
        assert np.array_equal(got[1], want[:, 1].astype(np.int64))
        assert np.allclose(got[0], want[:, 0].astype(float), atol=1e-6)


def test_query_idx_forces_the_window_first_at_zero_even_when_the_query_differs():
    # STUMPY forces it for the z-normalized distance only (its aamp_match keeps the true one),
    # so the two agree for the raw distance only when the query is the window; see the match test.
    normalize = True
    t = tie_free(120, 12)
    at = 40
    q = t[85:95].copy()  # not the window at `at`: the claim is forced all the same
    for max_distance in (None, np.inf):
        want = stumpy_match(
            q, t, query_idx=at, normalize=normalize, max_distance=max_distance, max_matches=6
        )
        got = mp.match(
            q, t, query_idx=at, normalize=normalize, max_distance=max_distance, max_matches=6,
            scale=scale_for(normalize),
        )  # fmt: skip
        free = mp.match(
            q, t, normalize=normalize, max_distance=max_distance, max_matches=6,
            scale=scale_for(normalize),
        )  # fmt: skip
        assert got[1][0] == at and got[0][0] == 0.0
        assert free[1][0] != at
        assert np.array_equal(got[1], want[:, 1].astype(np.int64))


@pytest.mark.parametrize("normalize", [True, False])
def test_query_idx_changes_nothing_when_the_query_is_the_nearest_anyway(normalize):
    t = tie_free(120, 8)
    q = t[30:40].copy()
    plain = mp.match(q, t, normalize=normalize, scale=scale_for(normalize), max_distance=np.inf)
    anchored = mp.match(
        q, t, normalize=normalize, query_idx=30, scale=scale_for(normalize), max_distance=np.inf
    )
    assert len(plain[1]) > 1
    assert np.array_equal(plain[1], anchored[1])


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("slack", [0.0, 1e-3, 0.5])
@pytest.mark.parametrize("margin", [-1e-3, 1e-3])
def test_atol_widens_the_threshold_as_stumpys_does(normalize, slack, margin):
    t = tie_free(200, 9)
    q = t[20:32].copy()
    scale = scale_for(normalize)
    everything = mp.match(q, t, max_distance=np.inf, normalize=normalize, scale=scale, atol=0.0)
    edge = float(everything[0][3])
    threshold = edge + margin  # a distance is in when it is at most max_distance + atol
    max_distance = threshold - slack
    want = stumpy_match(q, t, max_distance=max_distance, atol=slack, normalize=normalize)
    got = mp.match(q, t, max_distance=max_distance, atol=slack, normalize=normalize, scale=scale)
    assert len(got[1]) == (4 if margin > 0 else 3)
    assert np.array_equal(got[1], want[:, 1].astype(np.int64))


def test_the_default_atol_is_stumpys_and_is_applied():
    t = tie_free(200, 10)
    q = t[20:32].copy()
    everything = mp.match(q, t, max_distance=np.inf, scale=1e-9, normalize=False, atol=0.0)
    edge = float(everything[0][2])
    just_under = edge - 5e-9  # inside the default tolerance, outside none
    kwargs = {"max_distance": just_under, "scale": 1e-9, "normalize": False}
    with_default = mp.match(q, t, **kwargs)
    explicit = mp.match(q, t, atol=1e-8, **kwargs)
    without = mp.match(q, t, atol=0.0, **kwargs)
    assert len(without[1]) == 2 and len(with_default[1]) == 3
    assert np.array_equal(with_default[1], explicit[1])


@pytest.mark.parametrize("name", ["match", "aamp_match"])
def test_a_query_idx_that_starts_no_window_is_refused(name):
    t = tie_free(50, 11)
    q = t[:10].copy()
    function = getattr(mp, name)
    with pytest.raises(ValueError):
        function(q, t, query_idx=len(t) - 9)
    with pytest.raises(ValueError):
        function(q, t, query_idx=-1)


@pytest.mark.parametrize("name", ["match", "aamp_match"])
def test_precomputed_statistics_are_refused_not_ignored(name):
    t = tie_free(50, 12)
    q = t[:10].copy()
    function = getattr(mp, name)
    with pytest.raises(NotImplementedError):
        function(q, t, T_subseq_isfinite=np.ones(41, dtype=bool))
    if name == "match":
        with pytest.raises(NotImplementedError):
            function(q, t, M_T=np.zeros(41))
        with pytest.raises(NotImplementedError):
            function(q, t, Sigma_T=np.ones(41))


@pytest.mark.parametrize("name", ["motifs", "aamp_motifs"])
def test_motifs_infer_the_window_length_from_the_length_of_the_profile(name):
    function = getattr(mp, name)
    scale = {"scale": 1.0}
    real = stumpy.stump(T, M)[:, 0].astype(np.float64)
    from_real = function(T, real, **scale)
    from_zeros = function(T, np.zeros(len(real)), **scale)
    from_list = function(T, [7] * len(real), **scale)
    assert np.array_equal(from_real[1], from_zeros[1]) and np.array_equal(from_real[1], from_list[1])
    other = function(T, np.zeros(len(real) + 2), **scale)  # m = M - 2
    assert not np.array_equal(other[0], from_real[0], equal_nan=True)


@pytest.mark.parametrize("name", ["motifs", "aamp_motifs"])
@pytest.mark.parametrize("length", [0, N - 1, N + 1, N + 5])
def test_a_profile_length_that_implies_a_bad_window_is_a_value_error(name, length):
    # length N - 1 gives m = 2 (below 3); 0 and above N + 1 - 3 give m above len(T) or below 1
    function = getattr(mp, name)
    with pytest.raises(ValueError):
        function(T, np.zeros(length))


@pytest.mark.parametrize("name", ["motifs", "aamp_motifs"])
def test_atol_is_applied_to_each_candidates_matches_as_stumpys_is(name):
    t = tie_free(300, 21)
    m = 10
    normalize = name == "motifs"
    scale = scale_for(normalize)
    profile = stumpy.stump(t, m)[:, 0].astype(np.float64) if normalize else np.zeros(len(t) - m + 1)
    function = getattr(mp, name)
    kwargs = {"scale": scale, "cutoff": np.inf, "max_matches": 10}
    full = function(t, profile, max_distance=np.inf, atol=0.0, **kwargs)
    d = full[0][0]
    assert np.sum(~np.isnan(d)) >= 6
    threshold = (d[3] + d[4]) / 2
    slack = 0.5 * (threshold - d[1])
    narrow = function(t, profile, max_distance=threshold - slack, atol=0.0, **kwargs)
    widened = function(t, profile, max_distance=threshold - slack, atol=slack, **kwargs)
    exact = function(t, profile, max_distance=threshold, atol=0.0, **kwargs)
    assert narrow[1].shape[1] < widened[1].shape[1] == 4
    assert np.array_equal(widened[1], exact[1])
    stumpy_function = stumpy.motifs if normalize else stumpy.aamp_motifs
    stumpy_args = {"normalize": True} if normalize else {}
    want_profile = stumpy.stump(t, m) if normalize else stumpy.aamp(t, m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = stumpy_function(
            t, want_profile[:, 0].astype(np.float64),
            max_distance=threshold - slack, atol=slack, cutoff=np.inf, max_matches=10, **stumpy_args,
        )  # fmt: skip
    assert np.array_equal(widened[1], want[1])


@pytest.mark.parametrize("name", ["stumpi", "aampi"])
def test_a_stream_takes_a_k_above_one_and_a_k_below_one_is_a_value_error(name):
    function = getattr(mp, name)
    assert function(T[:40], M, k=2).left_topk_P_.shape[1] == 2
    with pytest.raises(ValueError):
        function(T[:40], M, k=0)


@pytest.mark.parametrize("name", ["stumpi", "aampi"])
def test_a_precomputed_profile_for_the_stream_is_refused(name):
    with pytest.raises(NotImplementedError):
        getattr(mp, name)(T[:40], M, mp=BASE_P)


def test_stumpi_without_normalization_points_at_aampi():
    with pytest.raises(NotImplementedError):
        mp.stumpi(T[:40], M, normalize=False)


def test_streams_infer_the_scale_when_none_is_given():
    for make in (mp.stumpi, mp.aampi):
        inferred = make(T[:40], M, egress=False)
        given = make(T[:40], M, egress=False, scale=1.0)
        for x in T[40:50]:
            inferred.update(x)
            given.update(x)
        assert np.array_equal(inferred.left_I_, given.left_I_)


def test_fluss_with_an_index_equals_the_one_from_the_series():
    from_index = mp.fluss(BASE_I, 12, 3, m=M)
    from_series = mp.fluss_series(T, M, 12, 3)
    same(from_index, from_series)
    default_m = mp.fluss(BASE_I, M, 3)
    same(default_m, mp.fluss(BASE_I, M, 3, m=M))


@pytest.mark.parametrize("atol", [-1e-9, -1.0, float("nan")])
def test_a_negative_or_nan_atol_is_a_value_error_not_a_clamp(atol):
    q = T[:M].copy()
    for call in (
        lambda: mp.match(q, T, atol=atol),
        lambda: mp.match(q, T, normalize=False, atol=atol),
        lambda: mp.aamp_match(q, T, atol=atol),
        lambda: mp.motifs(T, np.zeros(N - M + 1), atol=atol),
        lambda: mp.aamp_motifs(T, np.zeros(N - M + 1), atol=atol),
    ):
        with pytest.raises(ValueError):
            call()


COMPLEX_CALLS = {
    "left_profile": lambda x: mp.left_profile(x, M),
    "infer_scale": lambda x: mp.infer_scale(x, M),
    "aamp": lambda x: mp.aamp(x, M),
    "match": lambda x: mp.match(x[:M], x),
    "motifs": lambda x: mp.motifs(x, np.zeros(N - M + 1)),
    "mstump": lambda x: mp.mstump(np.vstack([x, x]), M),
    "atsc": lambda x: mp.atsc(x, x, 1),
    "stumpi": lambda x: mp.stumpi(x, M),
    "mpdist": lambda x: mp.mpdist(x, x, M),
}


@pytest.mark.parametrize("form", ["array", "list"])
@pytest.mark.parametrize("name", sorted(COMPLEX_CALLS))
def test_complex_input_is_a_type_error_not_a_cast_that_drops_the_imaginary_part(name, form):
    z = (T + 1j * series(9)).astype(np.complex128)
    given = z if form == "array" else z.tolist()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no ComplexWarning either: the call is refused first
        with pytest.raises(TypeError):
            COMPLEX_CALLS[name](given)


@pytest.mark.parametrize("bad", [-2, len(BASE_I), len(BASE_I) + 7])
def test_fluss_refuses_an_index_entry_that_is_not_minus_one_or_a_window(bad):
    index = BASE_I.copy()
    index[5] = bad
    with pytest.raises(ValueError):
        mp.fluss(index, 12, 3, m=M)
    index[5] = -1
    mp.fluss(index, 12, 3, m=M)


def test_a_stream_with_an_inferred_scale_refuses_a_sample_it_cannot_represent():
    walk = np.cumsum(np.random.default_rng(6).normal(size=150))
    # Later samples move 64 times as far as the history the scale is inferred from.
    walk[60:] = walk[59] + (walk[60:] - walk[59]) * 64
    stream = mp.stumpi(walk[:60], 8, egress=False)
    fed = 0
    with pytest.raises(ValueError, match="scale="):
        for x in walk[60:]:
            stream.update(x)
            fed += 1
    assert fed < 90
    # The refused sample never reached the stream: its profile covers exactly the accepted ones.
    assert len(stream.left_I_) == 60 - 8 + 1 + fed
    # With the tick named, the same stream takes every sample, as STUMPY's takes them.
    named = mp.stumpi(walk[:60], 8, egress=False, scale=2.0**-14)
    for x in walk[60:]:
        named.update(x)
    assert len(named.left_I_) == 150 - 8 + 1
