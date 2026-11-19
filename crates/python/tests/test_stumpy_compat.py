"""`ladders.stumpy` against STUMPY 1.14.1: every name STUMPY exports but the warm start
(`prescrump`, `prescraamp`) exists with STUMPY's signature, returns STUMPY's types and dtypes, and
answers as STUMPY does on tie-free series; the names this library does not run raise
`NotImplementedError` naming the alternative, and a STUMPY setting it cannot honour is a
`ValueError`."""

import inspect
import types
import warnings

import numpy as np
import pytest

import ladders as mp
import ladders.stumpy as ours

stumpy = pytest.importorskip("stumpy")
import stumpy.mparray  # noqa: E402
from multi_reference import MstumpCells, Tally, reference  # noqa: E402

M = 10
DISTRIBUTED = [
    "aamp_ostinatoed", "aamp_stimped", "aampdisted", "aamped", "maamped", "mpdisted", "mstumped",
    "ostinatoed", "stimped", "stumped",
]  # fmt: skip
GPU = [
    "gpu_aamp", "gpu_aamp_ostinato", "gpu_aamp_stimp", "gpu_aampdist", "gpu_mpdist", "gpu_ostinato",
    "gpu_stimp", "gpu_stump",
]  # fmt: skip


# STUMPY's exact warm start, deliberately absent here: scrump is exact once converged.
OMITTED = {"prescraamp", "prescrump"}


def stumpy_names():
    return sorted(
        name
        for name in dir(stumpy)
        if not name.startswith("_")
        and name not in OMITTED
        and callable(getattr(stumpy, name))
        and not isinstance(getattr(stumpy, name), types.ModuleType)
        and getattr(getattr(stumpy, name), "__module__", "").startswith("stumpy")
    )


def grid_noise(n, seed):
    """Gaussian samples on a 1/64 grid: tie-free in practice."""
    return np.round(np.random.default_rng(seed).normal(size=n) * 64) / 64


def walk(n, seed):
    return np.cumsum(grid_noise(n, seed))


def quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


def close(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-7)


# --- names and signatures ---------------------------------------------------------------------


def test_every_stumpy_name_is_here():
    names = stumpy_names()
    assert len(names) > 40
    missing = [name for name in names if not hasattr(ours, name)]
    assert missing == []
    assert set(names) <= set(ours.__all__)


@pytest.mark.parametrize("name", stumpy_names())
def test_the_signature_is_stumpys(name):
    assert inspect.signature(getattr(ours, name)) == inspect.signature(getattr(stumpy, name)), name


@pytest.mark.parametrize("name", DISTRIBUTED)
def test_a_distributed_name_raises_naming_the_alternative(name):
    with pytest.raises(NotImplementedError, match=f"use {name[:-2] if name.endswith('ed') else name}"):
        function = getattr(ours, name)
        required = [p for p in inspect.signature(function).parameters.values() if p.default is inspect.Parameter.empty]
        function(*[None] * len(required))


@pytest.mark.parametrize("name", GPU)
def test_a_gpu_name_raises_naming_the_alternative(name):
    with pytest.raises(NotImplementedError, match=f"use {name[4:]}"):
        getattr(ours, name)(walk(50, 1), M)


@pytest.mark.parametrize("name", sorted(OMITTED))
def test_the_warm_start_functions_are_absent_not_stubs(name):
    assert callable(getattr(stumpy, name))
    assert not hasattr(ours, name)
    assert name not in ours.__all__


@pytest.mark.parametrize("call, name", [
    (lambda: ours.scrump(walk(80, 1), M, pre_scrump=True), "scrump"),
    (lambda: ours.scraamp(walk(80, 1), M, pre_scraamp=True), "scraamp"),
])  # fmt: skip
def test_the_warm_start_raises_pointing_at_the_call_without_it(call, name):
    with pytest.raises(NotImplementedError, match=f"warm start is not available; call {name} without it"):
        call()


@pytest.mark.parametrize("name, value", [("STUMPY_EXCL_ZONE_DENOM", 2), ("STUMPY_DENOM_THRESHOLD", 1e-6)])
def test_a_stumpy_setting_that_cannot_be_honoured_is_an_error(name, value):
    original = getattr(stumpy.config, name)
    setattr(stumpy.config, name, value)
    try:
        for call in (lambda: ours.stump(walk(80, 1), M), lambda: ours.stumpi(walk(80, 1), M), lambda: ours.mass(walk(80, 1)[:M], walk(80, 1))):
            with pytest.raises(ValueError, match=f"stumpy.config.{name}={value!r} cannot be honoured") as error:
                call()
            assert f"set it back to {original!r}" in str(error.value)
    finally:
        setattr(stumpy.config, name, original)
    ours.stump(walk(80, 1), M)


# --- profiles ----------------------------------------------------------------------------------


def assert_same_mparray(got, want, k):
    assert type(got).__name__ == "mparray" and isinstance(got, np.ndarray)
    assert got.shape == want.shape and got.dtype == want.dtype == object
    assert (got._m, got._k, got._excl_zone_denom) == (want._m, want._k, want._excl_zone_denom)
    for attribute in ("P_", "I_", "left_I_", "right_I_"):
        a, b = getattr(got, attribute), getattr(want, attribute)
        assert a.shape == b.shape and a.dtype == b.dtype, attribute
        assert type(a).__name__ == "mparray", attribute
    close(got.P_, want.P_)
    for attribute in ("I_", "left_I_", "right_I_"):
        assert np.array_equal(getattr(got, attribute), getattr(want, attribute)), attribute
    assert [type(x) for x in got[0]] == [type(x) for x in want[0]]


@pytest.mark.parametrize("k", [1, 3])
def test_stump_is_stumpys_mparray(k):
    t = walk(300, 2)
    assert_same_mparray(ours.stump(t, M, k=k), stumpy.stump(t, M, k=k), k)


@pytest.mark.parametrize("k", [1, 2])
def test_stump_ab_join_is_stumpys(k):
    a, b = walk(200, 3), walk(150, 4)
    assert_same_mparray(ours.stump(a, M, b, ignore_trivial=False, k=k), stumpy.stump(a, M, b, ignore_trivial=False, k=k), k)
    assert_same_mparray(ours.stump(a, M, b, k=k), quiet(stumpy.stump, a, M, b, k=k), k)


def test_stump_without_normalization_is_aamp():
    t = walk(200, 5)
    assert_same_mparray(ours.stump(t, M, normalize=False), stumpy.stump(t, M, normalize=False), 1)


@pytest.mark.parametrize("k", [1, 2])
def test_aamp_is_stumpys_mparray(k):
    t = walk(250, 6)
    assert_same_mparray(ours.aamp(t, M, k=k), stumpy.aamp(t, M, k=k), k)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("query_idx", [None, 40])
def test_mass_is_stumpys(normalize, query_idx):
    t = walk(300, 7)
    q = t[40 : 40 + M] + np.linspace(0.0, 0.5, M)
    got = ours.mass(q, t, normalize=normalize, query_idx=query_idx)
    want = quiet(stumpy.mass, q, t, normalize=normalize, query_idx=query_idx)
    assert (got[40] == 0.0) == (query_idx is not None)
    assert type(got) is np.ndarray and got.dtype == want.dtype
    close(got, want)


def test_mass_refuses_what_it_does_not_take():
    t = walk(100, 7)
    with pytest.raises(NotImplementedError, match="Σ_T"):
        ours.mass(t[:M], t, Σ_T=np.ones(91))
    with pytest.raises(ValueError, match="query_idx must not be negative"):
        ours.mass(t[:M], t, query_idx=-1)


def test_mstump_and_maamp_are_stumpys():
    rows = np.vstack([walk(150, 8), walk(150, 9), walk(150, 10)])
    # The walks lie on the 1/64 grid, which the inferred scale holds exactly: the exact `mstump`
    # indices are those of the ticks at 1/64, and STUMPY's must equal them (no near tie, counted).
    tally = Tally()
    exact = MstumpCells(rows, M, 1 / 64).indices(tally=tally)
    assert tally.near == 0
    for name in ("mstump", "maamp"):
        got, want = getattr(ours, name)(rows, M), getattr(stumpy, name)(rows, M)
        # STUMPY returns its `mparray` named tuple, so `P_` and `I_` read as attributes.
        assert type(got).__name__ == type(want).__name__ == "mparray"
        assert got._fields == want._fields == ("P_", "I_")
        assert [x.dtype for x in got] == [x.dtype for x in want]
        close(got.P_, want.P_)
        assert np.array_equal(got.I_, want.I_)
        if name == "mstump":
            assert np.array_equal(got.I_, exact) and np.array_equal(want.I_, exact)


# --- derived results ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["match", "aamp_match"])
def test_match_is_stumpys_object_array(name):
    t = walk(300, 11)
    q = t[100 : 100 + M].copy()
    for kwargs in ({}, {"max_distance": 3.0}, {"max_matches": 2}, {"max_distance": lambda D: np.median(D)}):
        got, want = getattr(ours, name)(q, t, **kwargs), getattr(stumpy, name)(q, t, **kwargs)
        assert got.dtype == want.dtype == object and got.shape == want.shape, kwargs
        assert [type(x) for x in got[0]] == [type(x) for x in want[0]]
        close(got[:, 0], want[:, 0])
        assert np.array_equal(got[:, 1].astype(np.int64), want[:, 1].astype(np.int64))
    none = getattr(ours, name)(q + np.arange(M) * 50, t, max_distance=0.0)
    assert none.shape == (0,) and none.dtype == object


@pytest.mark.parametrize("name", ["motifs", "aamp_motifs"])
def test_motifs_are_stumpys_including_the_empty_shape(name):
    t = walk(400, 12)
    profile = stumpy.stump(t, M)[:, 0].astype(np.float64)
    for kwargs in ({"max_motifs": 3}, {"max_motifs": 2, "max_distance": 1e-9, "cutoff": 1e-9}):
        got, want = getattr(ours, name)(t, profile, **kwargs), quiet(getattr(stumpy, name), t, profile, **kwargs)
        assert type(got) is tuple and [(x.dtype, x.shape) for x in got] == [(x.dtype, x.shape) for x in want]
        close(got[0], want[0])
        assert np.array_equal(got[1], want[1])
    with pytest.raises(NotImplementedError, match="max_distance as a function"):
        getattr(ours, name)(t, profile, max_distance=lambda D: 1.0)


def test_mmotifs_are_stumpys():
    rows = np.vstack([walk(200, 13), walk(200, 14)])
    P, I = stumpy.mstump(rows, M)
    got, want = ours.mmotifs(rows, P, I, max_motifs=2), stumpy.mmotifs(rows, P, I, max_motifs=2)
    assert type(got) is tuple and [getattr(x, "dtype", type(x)) for x in got] == [getattr(x, "dtype", type(x)) for x in want]
    close(got[0], want[0])
    assert np.array_equal(got[1], want[1])
    P, I = stumpy.maamp(rows, M)
    got, want = ours.aamp_mmotifs(rows, P, I, max_motifs=2), stumpy.aamp_mmotifs(rows, P, I, max_motifs=2)
    assert np.array_equal(got[1], want[1])


@pytest.mark.parametrize("name", ["mpdist", "aampdist"])
def test_mpdist_is_a_numpy_float(name):
    a, b = walk(200, 15), walk(180, 16)
    got, want = getattr(ours, name)(a, b, M), getattr(stumpy, name)(a, b, M)
    assert type(got) is type(want) is np.float64
    close(got, want)


def test_fluss_atsc_and_allc_are_stumpys():
    t = walk(300, 17)
    profile = stumpy.stump(t, M)
    I, left, right = (profile[:, c].astype(np.int64) for c in (1, 2, 3))
    cac = np.linspace(0.5, 1.0, len(I))
    got, want = ours.fluss(I, M, 3, custom_iac=cac), stumpy.fluss(I, M, 3, custom_iac=cac)
    assert type(got) is tuple and [x.dtype for x in got] == [x.dtype for x in want]
    close(got[0], want[0])
    assert np.array_equal(got[1], want[1])
    assert np.array_equal(ours.atsc(left, right, 5), stumpy.atsc(left, right, 5))
    got, want = ours.allc(left, right), stumpy.allc(left, right)
    assert type(got) is tuple and type(got[0]) is list
    assert sorted(map(tuple, got[0])) == sorted(map(tuple, want[0]))
    assert len(got[1]) == len(want[1])


@pytest.mark.parametrize("name", ["ostinato", "aamp_ostinato"])
def test_ostinato_types_are_stumpys(name):
    ts = [walk(120, 18), walk(130, 19), walk(110, 20)]
    got, want = getattr(ours, name)(ts, M), getattr(stumpy, name)(ts, M)
    assert [type(x) for x in got] == [np.float64, int, np.int64]
    assert isinstance(want[0], float) and [type(x) for x in got[1:]] == [type(x) for x in want[1:]]
    close(got[0], want[0])
    assert got[1:] == want[1:]


@pytest.mark.parametrize("name", ["snippets", "aampdist_snippets"])
def test_snippets_are_stumpys(name):
    t = walk(240, 21)
    got, want = getattr(ours, name)(t, 20, 2), getattr(stumpy, name)(t, 20, 2)
    assert type(got) is tuple and [(x.dtype, x.shape) for x in got] == [(x.dtype, x.shape) for x in want]
    assert np.array_equal(got[1], want[1])


def test_mdl_and_subspace_are_stumpys():
    rows = np.vstack([walk(150, 22), walk(150, 23), walk(150, 24)])
    for name in ("mdl", "maamp_mdl"):
        got, want = getattr(ours, name)(rows, M, [3, 3, 3], [80, 80, 80]), getattr(stumpy, name)(rows, M, [3, 3, 3], [80, 80, 80])
        assert type(got) is tuple and got[0].dtype == want[0].dtype and type(got[1]) is list
        close(got[0], want[0])
    tally = Tally()
    for name, normalize in (("subspace", True), ("maamp_subspace", False)):
        got, want = getattr(ours, name)(rows, M, 3, 80, 1), getattr(stumpy, name)(rows, M, 3, 80, 1)
        exact = reference(rows, M, 3, 80, 1, normalize=normalize, tally=tally)
        assert got.dtype == want.dtype and list(got) == list(want) == exact
    # STUMPY may bin otherwise only a sample near an edge; there is none.
    assert tally.near == 0 and tally.samples > 0


# --- anytime and pan profiles ------------------------------------------------------------------


@pytest.mark.parametrize("name", ["scrump", "scraamp"])
def test_scrump_is_stumpys_once_converged(name):
    t = walk(200, 25)
    np.random.seed(1)
    got = getattr(ours, name)(t, M, percentage=1.0)
    np.random.seed(1)
    want = getattr(stumpy, name)(t, M, percentage=1.0)
    got.update()
    want.update()
    for attribute in ("P_", "I_", "left_I_", "right_I_"):
        a, b = getattr(got, attribute), getattr(want, attribute)
        assert a.dtype == b.dtype and a.shape == b.shape
    close(got.P_, want.P_)
    assert np.array_equal(got.I_, want.I_)


@pytest.mark.parametrize("name", ["stimp", "aamp_stimp"])
def test_stimp_is_stumpys_at_every_size(name):
    t = walk(160, 26)
    got = getattr(ours, name)(t, 5, 12)
    want = getattr(stumpy, name)(t, 5, 12, percentage=1.0)
    assert np.array_equal(got.M_, want.M_)
    for _ in range(len(got.M_)):
        got.update()
        want.update()
    for a, b in zip(got.P_, want.P_):
        close(a, b)


# --- streams -----------------------------------------------------------------------------------


def stream_pair(name, egress, k, n=150, more=60, seed=27):
    t = walk(n + more, seed)
    got, want = getattr(ours, name)(t[:n], M, egress=egress, k=k), getattr(stumpy, name)(t[:n], M, egress=egress, k=k)
    for x in t[n:]:
        got.update(x)
        want.update(x)
    return got, want


@pytest.mark.parametrize("name", ["stumpi", "aampi"])
@pytest.mark.parametrize("egress", [False, True])
@pytest.mark.parametrize("k", [1, 2])
def test_a_stream_is_stumpys(name, egress, k):
    got, want = stream_pair(name, egress, k)
    for attribute in ("P_", "I_", "left_P_", "left_I_", "T_"):
        a, b = getattr(got, attribute), getattr(want, attribute)
        assert a.shape == b.shape and a.dtype == b.dtype, attribute
    for attribute in ("P_", "left_P_", "T_"):
        close(getattr(got, attribute), getattr(want, attribute))
    for attribute in ("I_", "left_I_"):
        assert np.array_equal(getattr(got, attribute), getattr(want, attribute)), attribute


def test_stumpi_without_normalization_is_aampi():
    t = walk(120, 28)
    got, want = ours.stumpi(t, M, normalize=False), stumpy.stumpi(t, M, normalize=False)
    got.update(1.0)
    want.update(1.0)
    close(got.P_, want.P_)


def test_floss_right_profile_is_stumpys():
    t = walk(400, 29)
    n = 250
    profile = stumpy.stump(t[:n], M)
    iac = np.linspace(1.0, 50.0, n - M + 1)
    got, want = ours.floss(profile, t[:n], M, M, custom_iac=iac), stumpy.floss(profile, t[:n], M, M, custom_iac=iac)
    assert np.array_equal(got.cac_1d_, want.cac_1d_)
    for x in t[n:]:
        got.update(x)
        want.update(x)
    # STUMPY's update lets a new window be the neighbor of one exactly ceil(m / 4) before it, which
    # its own stump excludes; this library excludes it throughout.
    zone = -(-M // 4)
    local = want.I_ - (len(t) - n)
    agree = (want.I_ < 0) | (local - np.arange(len(local)) > zone)
    assert agree.mean() > 0.9
    close(got.P_[agree], want.P_[agree])
    assert np.array_equal(got.I_[agree], want.I_[agree])
    assert ((got.I_ < 0) | (got.I_ - (len(t) - n) - np.arange(len(local)) > zone)).all()
    close(got.T_, want.T_)
    exact = ours.floss(stumpy.stump(t[-n:], M), t[-n:], M, M, custom_iac=iac)
    exact.update(t[-1])
    assert exact.cac_1d_.shape == want.cac_1d_.shape
    default = ours.floss(profile, t[:n], M, M)
    default.update(t[n])
    curve = default.cac_1d_
    assert curve.shape == want.cac_1d_.shape and ((0 <= curve) & (curve <= 1)).all()
    with pytest.raises(ValueError, match="one row per window"):
        ours.floss(profile[:5], t[:n], M, M)


def test_the_native_results_are_named_tuples():
    t = walk(200, 30)
    result = mp.matrix_profile(t, M)
    P, I, left, right = result
    assert result.P is P and result.I is I and result.left_I is left and result.right_I is right
    assert mp.left_profile(t, M).scale == mp.left_profile(t, M)[2]
    assert mp.ab_join(t, t[::-1].copy(), M).scale_b > 0
    assert mp.discords(t, M, 2).indices.shape == (2,)
    assert inspect.signature(mp.left_profile) == inspect.signature(mp._native.left_profile)
    assert mp.left_profile.__doc__ == mp._native.left_profile.__doc__
