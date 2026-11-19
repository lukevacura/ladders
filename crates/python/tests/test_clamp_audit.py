"""Every public argument check of the bindings, one test per check: an honest check cannot change a
result (it refuses, pads, or ignores an argument with no effect on an exact result), and every
input STUMPY would silently clamp or round is a `ValueError` whose message names the argument and
its value, says what STUMPY does with it and how to get that explicitly."""

import numpy as np
import pytest

import ladders as mp

from conftest import profile_for, random_walk

M = 8
SCALE = 1.0


def walk(n=120, seed=1):
    return random_walk(n, seed)


def refused(call, *fragments, error=ValueError):
    with pytest.raises(error) as raised:
        call()
    for fragment in fragments:
        assert fragment in str(raised.value), (fragment, str(raised.value))


def u64(x):
    return np.asarray(x, dtype=np.float64).view(np.uint64)


# --- honest checks: a refusal, a padding or an argument that cannot change an exact result ------


def test_window_length_below_three_is_refused_as_stumpy_does():
    refused(lambda: mp.left_profile(walk(), 2), "m must be at least 3")


def test_window_length_beyond_the_platform_is_refused():
    refused(lambda: mp.left_profile(walk(), 2**63 - 1))


def test_stream_bound_below_m_is_refused():
    refused(lambda: mp.BoundedStream(M, M - 1, SCALE), "w must be at least m")


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_a_scale_that_is_no_tick_is_refused(scale):
    refused(lambda: mp.GrowingStream(M, scale), "scale must be positive and finite")


def test_a_count_sizing_an_allocation_is_refused_outside_its_range():
    refused(lambda: mp.DiscordStream(M, 0, SCALE), "k must be at least 1")
    refused(lambda: mp.DiscordStream(M, 2**28 + 1, SCALE), "at most")


def test_a_result_with_too_many_cells_is_refused():
    refused(lambda: mp.left_profile(walk(), M, SCALE, k=2**27), "cells")


def test_a_negative_length_is_refused():
    refused(lambda: mp.fluss_series(walk(), M, 10, 2, excl_factor=-1), "must not be negative")


def test_a_stumpy_input_this_library_computes_itself_is_refused():
    refused(lambda: mp.stumpi(walk(), M, mp=np.zeros((10, 4))), error=NotImplementedError)


@pytest.mark.parametrize("atol", [-1e-9, float("nan")])
def test_a_tolerance_that_would_shrink_or_void_the_threshold_is_refused(atol):
    refused(lambda: mp.match(walk()[:M], walk(), atol=atol), "atol")


def test_a_checkpoint_that_will_not_restore_is_refused():
    refused(lambda: mp.GrowingStream.restore(b"nonsense"), "invalid checkpoint")


def test_a_multi_dimensional_input_core_refuses_is_refused():
    refused(lambda: mp.mstump(np.vstack([walk(), walk(seed=2)]), M, include=[5]), "include")


def test_complex_input_is_refused_rather_than_cast():
    refused(lambda: mp.left_profile(walk() + 1j, M), "complex", error=TypeError)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5])
def test_a_thread_count_that_is_no_count_is_refused(threads):
    refused(lambda: mp.left_profile(walk(), M, SCALE, threads=threads), "threads")


def test_no_scale_to_infer_is_refused():
    refused(lambda: mp.left_profile(np.full(50, np.nan), M), "no scale fits")


def test_k_below_one_is_refused():
    refused(lambda: mp.matrix_profile(walk(), M, SCALE, k=0), "k must be at least 1")


def test_k_beyond_the_windows_is_padded_not_clamped():
    t = walk(20)
    windows = 20 - M + 1
    p, i, _ = mp.left_profile(t, M, SCALE, k=windows + 3)
    assert p.shape == (windows, windows + 3)
    assert (i[:, -3:] == -1).all() and np.isinf(p[:, -3:]).all()


@pytest.mark.parametrize("p", [1.0, 3.0, float("inf")])
def test_a_p_other_than_two_is_refused(p):
    refused(lambda: mp.aamp(walk(), M, p=p), error=NotImplementedError)


def test_ignore_trivial_decides_a_self_join_only_for_an_equal_t_b():
    t = walk()
    self_join = mp.aamp(t, M, scale=SCALE)
    assert np.array_equal(mp.aamp(t, M, t.copy(), scale=SCALE)[1], self_join[1])
    assert (mp.aamp(t, M, t.copy(), ignore_trivial=False, scale=SCALE)[0] == 0).all()


def test_a_step_below_one_is_refused():
    refused(lambda: mp.stimp(walk(), 3, 6, step=0, scale=SCALE), "step must be at least 1")


def test_sizes_without_max_m_outside_the_series_are_refused():
    refused(lambda: mp.stimp(walk(), 2, scale=SCALE), "min_m must be at least 3 when max_m is None")
    refused(lambda: mp.stimp(walk(20), 20, scale=SCALE), "longer than t")


def test_stimp_percentage_and_pre_scrump_cannot_change_an_exact_row():
    t = walk()
    base = mp.stimp(t, 4, 6, scale=SCALE)
    other = mp.stimp(t, 4, 6, percentage=7.0, pre_scrump=False, scale=SCALE)
    for pan in (base, other):
        for _ in range(3):
            pan.update()
    for a, b in zip(base.P_, other.P_):
        assert np.array_equal(u64(a), u64(b))


def test_p_is_ignored_with_normalize_as_in_stumpy():
    t = walk()
    assert mp.mpdist(t, t[::-1].copy(), M, p=3.0) == mp.mpdist(t, t[::-1].copy(), M)


def test_a_constant_predicate_that_cannot_apply_is_refused():
    refused(lambda: mp.stimp(walk(), 4, 6, normalize=False, T_subseq_isconstant_func=lambda a, w: None),
            error=TypeError)
    refused(lambda: mp.stimp(walk(), 4, 6, T_subseq_isconstant_func=3), "callable")


def test_mpdist_nan_percentage_is_refused():
    refused(lambda: mp.mpdist(walk(), walk(seed=2), M, float("nan")), "percentage must not be NaN")


def test_mpdist_negative_k_is_refused():
    refused(lambda: mp.mpdist(walk(), walk(seed=2), M, k=-1), "k must be at least 0")


def test_mpdist_percentage_position_past_the_last_is_stumpys_definition():
    a, b = walk(), walk(seed=2)
    last = (len(a) - M + 1) + (len(b) - M + 1) - 1
    assert mp.mpdist(a, b, M, 1.0) == mp.mpdist(a, b, M, k=last)
    assert mp.mpdist(a, b, M, 0.999) == mp.mpdist(a, b, M, k=last)


def test_mpdist_scale_keywords_belong_to_one_metric():
    refused(lambda: mp.mpdist(walk(), walk(seed=2), M, scale=1.0), "scale is for normalize=False")
    refused(lambda: mp.mpdist(walk(), walk(seed=2), M, normalize=False, scale_a=1.0), "scale_a")


def test_a_profile_whose_length_gives_no_window_is_refused():
    refused(lambda: mp.motifs(walk(), np.empty(200)), "does not fit")


@pytest.mark.parametrize("kwargs", [{"min_neighbors": -1}, {"max_matches": -1}])
def test_a_negative_motif_count_is_refused(kwargs):
    refused(lambda: mp.motifs(walk(), profile_for(walk(), M), scale=SCALE, **kwargs), "at least 0")


def test_motif_counts_beyond_the_windows_return_what_exists():
    t = walk(300)
    windows = 300 - M + 1
    huge = mp.motifs(t, profile_for(t, M), scale=SCALE, max_matches=2**62, max_motifs=2**62)
    exact = mp.motifs(t, profile_for(t, M), scale=SCALE, max_matches=windows, max_motifs=windows)
    assert np.array_equal(huge[1], exact[1])


def test_a_profile_of_the_wrong_shape_is_refused():
    t = np.vstack([walk(), walk(seed=2)])
    p, i = mp.mstump(t, M)
    refused(lambda: mp.mmotifs(t, p[:, :5], i[:, :5]), "P and I")


def test_mmotifs_negative_k_indexes_from_the_last_dimension_as_in_stumpy():
    t = np.vstack([walk(), walk(seed=2), walk(seed=3)])
    p, i = mp.mstump(t, M)
    a, b = mp.mmotifs(t, p, i, k=-1), mp.mmotifs(t, p, i, k=2)
    assert np.array_equal(a[1], b[1])
    refused(lambda: mp.mmotifs(t, p, i, k=-4), "k must be below")


def test_snippet_lengths_without_an_answer_are_refused():
    t = walk(120)
    refused(lambda: mp.snippets(t, 20, 2, s=-1), "s must be non-negative")
    refused(lambda: mp.snippets(t, 20, 2, s=2), "at least 3")
    refused(lambda: mp.snippets(t, 20, 2, mpdist_k=-1), "mpdist_k must be at least 0")
    refused(lambda: mp.snippets(t, 20, 2, mpdist_percentage=float("nan")), "must not be NaN")
    refused(lambda: mp.snippets(t, 61, 2), "too large")


def test_more_snippets_than_segments_pick_segments_again_as_in_stumpy():
    t = walk(60)
    found = mp.snippets(t, 20, 5, scale=SCALE)
    assert len(found[1]) == 5 and len(set(found[1])) <= 3


def test_ostinato_arguments_without_an_answer_are_refused():
    refused(lambda: mp.ostinato([], M), "at least one series")
    refused(lambda: mp.ostinato([walk(), walk(seed=2)], M, scales=[1.0]), "one entry per series")
    refused(lambda: mp.ostinato([walk(), walk(5)], M))


def test_fluss_counts_below_their_floor_are_refused():
    t = walk()
    refused(lambda: mp.fluss_series(t, M, 0, 2), "L must be at least 1")
    refused(lambda: mp.fluss_series(t, M, 10, 0), "n_regimes must be at least 1")


def test_fluss_index_entries_that_are_no_window_are_refused():
    refused(lambda: mp.fluss(np.array([1, 5, 0]), 3, 2, m=3), "window indices")


def test_floss_stream_lengths_are_refused_outside_their_range():
    refused(lambda: mp.FlossStream(M, 0, SCALE), "width must be at least 1")
    refused(lambda: mp.FlossStream(M, 10, SCALE, l=-1), "must not be negative")


def test_a_custom_iac_of_the_wrong_length_is_refused():
    refused(lambda: mp.fluss_series(walk(), M, 10, 2, custom_iac=np.ones(3)), "one value per window")


def test_chain_inputs_that_point_the_wrong_way_are_refused():
    refused(lambda: mp.atsc(np.array([-1, 0]), np.array([1, -1]), 5))
    refused(lambda: mp.allc(np.array([-2, 0]), np.array([1, -1])), "window indices or -1")


def test_discords_negative_k_is_refused_and_a_large_one_lists_what_fits():
    t = walk()
    refused(lambda: mp.discords(t, M, k=-1, scale=SCALE), "non-negative")
    distances, indices = mp.discords(t, M, k=10**6, scale=SCALE)
    assert 0 < len(indices) < 120


def test_a_query_that_cannot_be_registered_is_refused():
    library = mp.PatternLibrary(M, SCALE)
    refused(lambda: library.register(np.ones(M - 1)), "exactly m samples")
    refused(lambda: library.register(np.r_[np.ones(M - 1), np.nan]), "NaN")


def test_an_unknown_query_id_is_refused():
    refused(lambda: mp.PatternLibrary(M, SCALE).distance_profile(0, walk()), "no query with id 0")


def test_match_arguments_without_an_answer_are_refused():
    t = walk()
    refused(lambda: mp.match(t[:M], t, max_matches=-1), "max_matches")
    refused(lambda: mp.match(t[:M], t, query_idx=-1), "query_idx")
    refused(lambda: mp.match(np.r_[t[: M - 1], np.nan], t), "finite samples")


def test_match_max_distance_must_be_a_number():
    refused(lambda: mp.match(walk()[:M], walk(), max_distance=lambda d: 1.0), error=TypeError)


def test_aamp_match_quantizes_the_query_at_the_given_tick():
    t = walk()
    q = t[10 : 10 + M] + 0.2
    distances, indices = mp.aamp_match(q, t, scale=1.0, max_matches=1)
    assert indices[0] == 10 and distances[0] == 0.0


def test_multi_dimensional_options_without_an_answer_are_refused():
    t = np.vstack([walk(), walk(seed=2)])
    refused(lambda: mp.subspace(t, M, 0, 20, 0, n_bit=9), "n_bit")
    refused(lambda: mp.subspace(t, M, 0, 20, 0, discretize_func=np.round), error=NotImplementedError)
    refused(lambda: mp.mstump(t, M, scales=[1.0]), "one entry per dimension")


def test_subspace_strict_turns_missing_dimensions_into_an_error():
    t = np.vstack([walk(), walk(seed=2)])
    t[1, :M] = np.nan
    assert len(mp.subspace(t, M, 0, 40, 1)) == 2
    refused(lambda: mp.subspace(t, M, 0, 40, 1, strict=True), "fewer than k + 1")


def test_scrump_percentage_outside_its_range_is_refused():
    for percentage in [0.0, 1.5, float("nan")]:
        refused(lambda: mp.scrump(walk(), M, percentage=percentage), "percentage")


def test_scrump_options_without_an_exact_version_are_refused():
    refused(lambda: mp.scrump(walk(), M, pre_scrump=True), error=NotImplementedError)
    refused(lambda: mp.scrump(walk(), M, ignore_trivial=False), "ignore_trivial")
    refused(lambda: mp.scrump(walk(), M, scale_b=1.0), "scale_b")


def test_scrump_s_cannot_change_an_exact_profile():
    np.random.seed(3)
    a = mp.scrump(walk(), M, percentage=1.0, s=1, scale=SCALE)
    np.random.seed(3)
    b = mp.scrump(walk(), M, percentage=1.0, s=50, scale=SCALE)
    a.update()
    b.update()
    assert np.array_equal(u64(a.P_), u64(b.P_))


def test_stream_options_without_an_answer_are_refused():
    refused(lambda: mp.stumpi(walk(), M, normalize=False), error=NotImplementedError)
    refused(lambda: mp.stumpi(walk(), M, egress=False, w=50), "w is only used")
    refused(lambda: mp.aampi(walk(), M, k=0), "k must be at least 1")


def test_an_inferred_stream_scale_refuses_a_sample_it_cannot_represent():
    stream = mp.stumpi(walk(), M)
    refused(lambda: stream.update(1e300), "cannot")


def test_a_later_sample_is_quantized_at_the_reported_scale():
    stream = mp.aampi(walk(), M, egress=False)
    stream.update(3.4)
    other = mp.aampi(walk(), M, egress=False)
    other.update(3.0)
    assert np.array_equal(u64(stream.left_P_), u64(other.left_P_))


def test_stumpi_ignores_p_as_stumpy_does():
    a, b = mp.stumpi(walk(), M, p=3.0, scale=SCALE), mp.stumpi(walk(), M, scale=SCALE)
    assert np.array_equal(u64(a.left_P_), u64(b.left_P_))


def test_a_constant_flag_array_of_the_wrong_shape_is_refused():
    refused(lambda: mp.left_profile(walk(), M, SCALE, T_A_subseq_isconstant=np.zeros(3, bool)))


# --- converted: STUMPY clamps or rounds these silently; here each is a ValueError ---------------


def test_an_integer_float64_cannot_hold_is_an_error_naming_stumpys_rounding():
    t = np.arange(100, dtype=np.int64)
    t[7] = 2**53 + 1
    for given in (t, list(map(int, t)), t.astype(np.uint64)):
        refused(lambda: mp.left_profile(given, M), "the integer 9007199254740993",
                "rounds it to 9007199254740992.0", "pass the array as float64")
    t[7] = 2**60
    assert mp.left_profile(t, M)[2] > 0
    assert mp.left_profile(np.arange(100, dtype=np.int32), M)[2] == 1.0
    refused(lambda: mp.left_profile([2**70 + 1] + [1] * 20, M), "rounds it to 1.1805916207174113e21")


def test_an_index_that_is_no_int64_is_an_error_naming_numpys_cast():
    refused(lambda: mp.fluss(np.array([1.5, -1.0, 0.0]), 3, 1, m=3), "holds 1.5", "would make it 1;",
            "pass the array as int64")
    refused(lambda: mp.fluss(np.array([np.nan, 0.0]), 3, 1, m=3), "an arbitrary integer")
    refused(lambda: mp.fluss(np.array([2**64 - 1, 0], dtype=np.uint64), 3, 1, m=3), "wraps it to -1")
    assert len(mp.fluss(np.array([1.0, -1.0, 0.0]), 3, 1, m=3)[0]) == 3


def test_a_sample_beyond_the_ticks_of_a_given_scale_is_an_error_naming_a_scale_that_holds_it():
    t = walk(100) * 0.01
    t[50] = 1e14
    remedy = "pass scale=0.02 to keep it"
    calls = [
        lambda: mp.left_profile(t, M, 0.01),
        lambda: mp.matrix_profile(t, M, 0.01),
        lambda: mp.ab_join(t, walk(50), M, scale_a=0.01),
        lambda: mp.aamp(t, M, scale=0.01),
        lambda: mp.mstump(np.vstack([t, t]), M, scales=0.01),
        lambda: mp.mstump(np.vstack([t, t]), M, scales=[0.01, 1.0]),
        lambda: mp.stimp(t, 4, 6, scale=0.01),
        lambda: mp.stumpi(t, M, scale=0.01),
        lambda: mp.aampi(t, M, scale=0.01),
        lambda: mp.distance_profile(t[:M], t, 0.01),
    ]
    for call in calls:
        refused(call, "scale=0.01 cannot count the sample 100000000000000.0", "STUMPY computes in floats",
                remedy)
    library = mp.PatternLibrary(M, 0.01)
    library.register(walk(M))
    refused(lambda: library.distance_profile(0, t), "build the library with scale=0.02")
    t[50] = 1e13
    assert mp.left_profile(t, M, 0.02)[2] == 0.02


def stream_cases():
    return {
        "growing": lambda: mp.GrowingStream(M, 0.01),
        "bounded": lambda: mp.BoundedStream(M, 40, 0.01),
        "discord": lambda: mp.DiscordStream(M, 2, 0.01),
        "floss": lambda: mp.FlossStream(M, 20, 0.01),
        "pattern": lambda: pattern_stream(),
        "stumpi": lambda: mp.stumpi(walk(30), M, scale=0.01),
        "aampi": lambda: mp.aampi(walk(30), M, scale=0.01),
        "stumpi restored": lambda: restored(mp.stumpi(walk(30), M)),
    }


def pattern_stream():
    library = mp.PatternLibrary(M, 0.01)
    library.register(walk(M))
    return mp.PatternStream(library)


def restored(stream):
    return type(stream).restore(stream.checkpoint())


def state(stream):
    if hasattr(stream, "checkpoint"):
        return bytes(stream.checkpoint())
    if hasattr(stream, "top"):
        return stream.top()
    if hasattr(stream, "curve"):
        return stream.curve().tolist()
    return None


@pytest.mark.parametrize("name", list(stream_cases()))
def test_a_stream_sample_beyond_its_ticks_is_an_error_and_is_not_folded_in(name):
    stream = stream_cases()[name]()
    for x in walk(25, seed=4):
        stream.update(float(x))
    before = state(stream)
    refused(lambda: stream.update(1e300), "cannot count the sample 1e300",
            "start the stream with scale=")
    if hasattr(stream, "update_many"):
        refused(lambda: stream.update_many(np.array([1.0, 2.0, -1e300])), "cannot count the sample -1e300")
    assert state(stream) == before
    stream.update(float("nan"))
    stream.update(3.0)
