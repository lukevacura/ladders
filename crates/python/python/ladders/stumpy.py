"""STUMPY 1.14.1's API over this library's exact engines: `import ladders.stumpy as stumpy`.

Every name STUMPY exports is here, with STUMPY's positional order, keywords, defaults and return
types: `mparray` where STUMPY returns one (with `P_`, `I_`, `left_I_` and `right_I_`), and the
same tuples, arrays and classes elsewhere. Each scale is inferred from the data as the native API
infers it, and every call uses every core. The native API (`import ladders`) takes a
`scale` and a thread count where this module cannot.

The rule that infers a scale has a number, `ladders.SCALE_RULE`: a result of this module
changes between releases only when that number does. For the same bits under any rule, take the
scale from `ladders.infer_scale` once and give it to the native API from then on.

Differences from STUMPY:
- Every result is exact: neighbors are ranked on exact distances, ties to the smaller index.
- Results are exact for the data quantized to `scale_used`. Data on no decimal or binary grid is
  quantized to the finest power of two at which every window, and every run of m + 128 samples,
  not holding an isolated outlier fits the fast path's bound: about seven significant digits at
  m = 64. Two neighbors whose distances differ by less than that resolution can therefore rank
  in the other order from STUMPY's floats. The native API's `scale="finest"` takes the finest exact scale instead.
- Arguments STUMPY would silently clip or round raise `ValueError`, naming the value that gets
  STUMPY's behavior explicitly.
- `prescrump` and `prescraamp` are absent: the exact warm start is not available. Call `scrump`
  or `scraamp`, which are exact once converged. `scrump(..., pre_scrump=True)` and
  `scraamp(..., pre_scraamp=True)` raise `NotImplementedError` saying so. `stimp` and `aamp_stimp`
  accept and ignore `pre_scrump` and `pre_scraamp`, which cannot change an exact row.
- The Dask/Ray `*ed` names raise `NotImplementedError`: distributed runs are out of scope here.
  Every call already uses every core.
- The `gpu_*` names raise `NotImplementedError` until the exact GPU profile lands. The plain
  name is the exact CPU version.
- A STUMPY setting this library cannot honour raises `ValueError` at the next call: with STUMPY
  imported, a `stumpy.config.STUMPY_EXCL_ZONE_DENOM` other than 4 or a
  `STUMPY_DENOM_THRESHOLD` other than `1e-14`.
"""

import math
import sys
from collections import namedtuple

import numpy as np

from . import _native

# STUMPY's multi-dimensional result (`stumpy.mmparray.mparray`): a named tuple with `P_` and `I_`.
_mmparray = namedtuple("mparray", "P_,I_")

__all__ = [
    "aamp", "aamp_match", "aamp_mmotifs", "aamp_motifs", "aamp_ostinato", "aamp_ostinatoed",
    "aamp_stimp", "aamp_stimped", "aampdist", "aampdist_snippets", "aampdisted", "aamped", "aampi",
    "allc", "atsc", "floss", "fluss", "gpu_aamp", "gpu_aamp_ostinato", "gpu_aamp_stimp",
    "gpu_aampdist", "gpu_mpdist", "gpu_ostinato", "gpu_stimp", "gpu_stump", "maamp", "maamp_mdl",
    "maamp_subspace", "maamped", "mass", "match", "mdl", "mmotifs", "motifs", "mparray", "mpdist",
    "mpdisted", "mstump", "mstumped", "ostinato", "ostinatoed", "scraamp", "scrump", "snippets",
    "stimp", "stimped", "stump", "stumped", "stumpi", "subspace",
]  # fmt: skip

# The STUMPY settings this library's results stand on, and what each must be.
_HONOURED = {
    "STUMPY_EXCL_ZONE_DENOM": (4, "the exclusion zone here is ceil(m / 4), not configurable yet"),
    "STUMPY_DENOM_THRESHOLD": (1e-14, "the arithmetic here is exact and needs no denominator floor"),
}


def _check_config():
    """With STUMPY imported, a setting this library cannot honour is a `ValueError`."""
    config = sys.modules.get("stumpy.config")
    if config is None:
        return
    for name, (value, reason) in _HONOURED.items():
        got = getattr(config, name, value)
        if got != value:
            raise ValueError(
                f"stumpy.config.{name}={got!r} cannot be honoured: {reason}; set it back to "
                f"{value!r} (stumpy.config._reset()) to run"
            )


class mparray(np.ndarray):
    """STUMPY's matrix profile array: one row per window, the `k` distances, the `k` indices, then
    the left and the right index; `P_`, `I_`, `left_I_` and `right_I_` read those columns."""

    def __new__(cls, input_array, m, k, excl_zone_denom):
        array = np.asarray(input_array).view(cls)
        array._m = m
        array._k = k
        array._excl_zone_denom = excl_zone_denom
        return array

    def __array_finalize__(self, array):
        if array is None:
            return
        self._m = getattr(array, "_m", None)
        self._k = getattr(array, "_k", None)
        self._excl_zone_denom = getattr(array, "_excl_zone_denom", None)

    def _columns(self, start, stop, dtype):
        columns = self[:, start:stop].astype(dtype)
        return columns.flatten() if stop - start == 1 or self._k == 1 else columns

    @property
    def P_(self):
        return self._columns(0, self._k, np.float64)

    @property
    def I_(self):
        return self._columns(self._k, 2 * self._k, np.int64)

    @property
    def left_I_(self):
        return self._columns(2 * self._k, 2 * self._k + 1, np.int64)

    @property
    def right_I_(self):
        return self._columns(2 * self._k + 1, 2 * self._k + 2, np.int64)


def _mparray(P, I, left, right, m, k):
    """STUMPY's object array of a profile: `P` and `I` 1-D for `k` of one, `(windows, k)` else."""
    windows = len(left)
    out = np.empty((windows, 2 * k + 2), dtype=object)
    out[:, :k] = np.asarray(P).reshape(windows, k)
    out[:, k : 2 * k] = np.asarray(I).reshape(windows, k)
    out[:, 2 * k] = left
    out[:, 2 * k + 1] = right
    return mparray(out, m, k, _HONOURED["STUMPY_EXCL_ZONE_DENOM"][0])


def _equal(a, b):
    return b is not None and np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _not_here(name, reason, instead):
    def refuse(*args, **kwargs):
        raise NotImplementedError(f"{name} is not available: {reason}; use {instead}")

    refuse.__name__ = refuse.__qualname__ = name
    refuse.__doc__ = f"Not available: {reason}; use {instead}."
    return refuse


# --- profiles ----------------------------------------------------------------------------------


def stump(T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None):
    """STUMPY's `stump`: the exact z-normalized matrix profile as an `mparray`. A `T_B` equal to
    `T_A` with `ignore_trivial=True` is a self-join, any other `T_B` an AB-join whose left and
    right indices are -1. `normalize=False` is `aamp`."""
    _check_config()
    if not normalize:
        return aamp(T_A, m, T_B, ignore_trivial, p, k)
    if T_B is None or (ignore_trivial and _equal(T_A, T_B)):
        P, I, left, right = _native.matrix_profile(T_A, m, k=k, T_A_subseq_isconstant=T_A_subseq_isconstant)
        return _mparray(P, I, left, right, m, k)
    P, I, _, _ = _native.ab_join(
        T_A, T_B, m, k=k, T_A_subseq_isconstant=T_A_subseq_isconstant, T_B_subseq_isconstant=T_B_subseq_isconstant
    )
    none = np.full(len(P), -1, dtype=np.int64)
    return _mparray(P, I, none, none, m, k)


def aamp(T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1):
    """STUMPY's `aamp`: the exact non-normalized matrix profile as an `mparray`."""
    _check_config()
    P, I, left, right = _native.aamp(T_A, m, T_B, ignore_trivial, p, k)
    return _mparray(P, I, left, right, m, k)


def mass(Q, T, M_T=None, Σ_T=None, normalize=True, p=2.0, T_subseq_isfinite=None, T_subseq_isconstant=None, Q_subseq_isconstant=None, query_idx=None):
    """STUMPY's `mass`: the distance from `Q` to every window of `T`, `inf` for a window with a
    gap; with `query_idx`, the window there is at distance 0. `M_T`, `Σ_T` and `T_subseq_isfinite`
    raise `NotImplementedError` (statistics are always computed exactly), and a negative
    `query_idx` is a `ValueError` (STUMPY reads it from the end)."""
    _check_config()
    for name, value in (("M_T", M_T), ("Σ_T", Σ_T), ("T_subseq_isfinite", T_subseq_isfinite)):
        if value is not None:
            raise NotImplementedError(f"{name} is not available: this library computes every statistic exactly itself")
    Q = np.asarray(Q)
    if normalize:
        D = _native.ab_join(T, Q, len(Q), T_A_subseq_isconstant=T_subseq_isconstant, T_B_subseq_isconstant=Q_subseq_isconstant)[0]
    else:
        D = _native.aamp(T, len(Q), Q, False, p)[0]
    if query_idx is not None:
        at = int(query_idx)
        if at < 0:
            raise ValueError(f"query_idx must not be negative, got {at}")
        D[at] = 0.0
    return D


def mstump(T, m, include=None, discords=False, normalize=True, p=2.0, T_subseq_isconstant=None):
    """STUMPY's `mstump`: an `mparray` named tuple `(P_, I_)`, each `(d, windows)`."""
    _check_config()
    return _mmparray(*_native.mstump(T, m, include, discords, normalize, p, T_subseq_isconstant))


def maamp(T, m, include=None, discords=False, p=2.0):
    """STUMPY's `maamp`: an `mparray` named tuple `(P_, I_)`, each `(d, windows)`."""
    _check_config()
    return _mmparray(*_native.maamp(T, m, include, discords, p))


# --- derived results ---------------------------------------------------------------------------


def _matches(distances, indices):
    """STUMPY's match array: rows of `[distance, index]`, shape `(0,)` with none."""
    rows = [[np.float64(d), np.int64(i)] for d, i in zip(distances, indices)]
    return np.array(rows, dtype=object)


def match(Q, T, M_T=None, Σ_T=None, max_distance=None, max_matches=None, atol=1e-08, query_idx=None, normalize=True, p=2.0, T_subseq_isfinite=None, T_subseq_isconstant=None, Q_subseq_isconstant=None):
    """STUMPY's `match`: rows of `[distance, index]`, nearest first. A function `max_distance` is
    given the distance profile, as in STUMPY."""
    _check_config()
    if callable(max_distance):
        D = mass(Q, T, M_T, Σ_T, normalize, p, T_subseq_isfinite, T_subseq_isconstant, Q_subseq_isconstant, query_idx)
        max_distance = float(max_distance(D))
    found = _native.match(
        Q, T, M_T, Σ_T, max_distance, max_matches, atol, query_idx, normalize, p, T_subseq_isfinite,
        T_subseq_isconstant, Q_subseq_isconstant,
    )
    return _matches(*found)


def aamp_match(Q, T, T_subseq_isfinite=None, max_distance=None, max_matches=None, atol=1e-08, query_idx=None, p=2.0):
    """STUMPY's `aamp_match`: rows of `[distance, index]`, nearest first."""
    _check_config()
    if callable(max_distance):
        D = mass(Q, T, None, None, False, p, T_subseq_isfinite, query_idx=query_idx)
        max_distance = float(max_distance(D))
    return _matches(*_native.aamp_match(Q, T, T_subseq_isfinite, max_distance, max_matches, atol, query_idx, p))


def _no_function(name, value):
    if callable(value):
        raise NotImplementedError(f"{name} as a function is not available for motifs yet; pass a number")


def _motif_arrays(found):
    """STUMPY's motif arrays: with no motif, both have shape `(1, 0)` and dtype float64."""
    distances, indices = found[0], found[1]
    if indices.shape[1] == 0:
        indices = indices.astype(np.float64)
    return (distances, indices) + tuple(found[2:])


def motifs(T, P, min_neighbors=1, max_distance=None, cutoff=None, max_matches=10, max_motifs=1, atol=1e-08, normalize=True, p=2.0, T_subseq_isconstant=None):
    """STUMPY's `motifs`: `(motif_distances, motif_indices)`."""
    _check_config()
    _no_function("max_distance", max_distance)
    return _motif_arrays(
        _native.motifs(T, P, min_neighbors, max_distance, cutoff, max_matches, max_motifs, atol, normalize, p, T_subseq_isconstant)
    )


def aamp_motifs(T, P, min_neighbors=1, max_distance=None, cutoff=None, max_matches=10, max_motifs=1, atol=1e-08, p=2.0):
    """STUMPY's `aamp_motifs`: `(motif_distances, motif_indices)`."""
    _check_config()
    _no_function("max_distance", max_distance)
    return _motif_arrays(_native.aamp_motifs(T, P, min_neighbors, max_distance, cutoff, max_matches, max_motifs, atol, p))


def mmotifs(T, P, I, min_neighbors=1, max_distance=None, cutoffs=None, max_matches=10, max_motifs=1, atol=1e-08, k=None, include=None, normalize=True, p=2.0, T_subseq_isconstant=None):
    """STUMPY's `mmotifs`: `(motif_distances, motif_indices, motif_subspaces, motif_mdls)`."""
    _check_config()
    _no_function("max_distance", max_distance)
    return _motif_arrays(
        _native.mmotifs(T, P, I, min_neighbors, max_distance, cutoffs, max_matches, max_motifs, atol, k, include, normalize, p, T_subseq_isconstant)
    )


def aamp_mmotifs(T, P, I, min_neighbors=1, max_distance=None, cutoffs=None, max_matches=10, max_motifs=1, atol=1e-08, k=None, include=None, p=2.0):
    """STUMPY's `aamp_mmotifs`: `(motif_distances, motif_indices, motif_subspaces, motif_mdls)`."""
    _check_config()
    _no_function("max_distance", max_distance)
    return _motif_arrays(
        _native.aamp_mmotifs(T, P, I, min_neighbors, max_distance, cutoffs, max_matches, max_motifs, atol, k, include, p)
    )


def mpdist(T_A, T_B, m, percentage=0.05, k=None, normalize=True, p=2.0, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None):
    """STUMPY's `mpdist`, as a `numpy.float64`."""
    _check_config()
    return np.float64(_native.mpdist(T_A, T_B, m, percentage, k, normalize, p, T_A_subseq_isconstant, T_B_subseq_isconstant))


def aampdist(T_A, T_B, m, percentage=0.05, k=None, p=2.0):
    """STUMPY's `aampdist`, as a `numpy.float64`."""
    _check_config()
    return np.float64(_native.aampdist(T_A, T_B, m, percentage, k, p))


def fluss(I, L, n_regimes, excl_factor=5, custom_iac=None):
    """STUMPY's `fluss`: `(cac, regime_locs)` of a two-sided index `I`, the window length taken
    to be `L`."""
    _check_config()
    return tuple(_native.fluss(I, L, n_regimes, excl_factor, custom_iac))


def atsc(IL, IR, j):
    """STUMPY's `atsc`: the anchored chain from window `j`."""
    _check_config()
    return _native.atsc(IL, IR, j)


def allc(IL, IR):
    """STUMPY's `allc`: `(S, C)`, every maximal chain and the longest one."""
    _check_config()
    return tuple(_native.allc(IL, IR))


def _consensus(found):
    radius, series, window = found
    return np.float64(radius), int(series), np.int64(window)


def ostinato(Ts, m, normalize=True, p=2.0, Ts_subseq_isconstant=None):
    """STUMPY's `ostinato`: `(radius, Ts_idx, subseq_idx)`."""
    _check_config()
    return _consensus(_native.ostinato(Ts, m, normalize, p, Ts_subseq_isconstant))


def aamp_ostinato(Ts, m, p=2.0):
    """STUMPY's `aamp_ostinato`: `(radius, Ts_idx, subseq_idx)`."""
    _check_config()
    return _consensus(_native.aamp_ostinato(Ts, m, p))


def snippets(T, m, k, percentage=1.0, s=None, mpdist_percentage=0.05, mpdist_k=None, normalize=True, p=2.0, mpdist_T_subseq_isconstant=None):
    """STUMPY's `snippets`: `(snippets, indices, profiles, fractions, areas, regimes)`."""
    _check_config()
    return tuple(_native.snippets(T, m, k, percentage, s, mpdist_percentage, mpdist_k, normalize, p, mpdist_T_subseq_isconstant))


def aampdist_snippets(T, m, k, percentage=1.0, s=None, mpdist_percentage=0.05, mpdist_k=None, p=2.0):
    """STUMPY's `aampdist_snippets`: `(snippets, indices, profiles, fractions, areas, regimes)`."""
    _check_config()
    return tuple(_native.aampdist_snippets(T, m, k, percentage, s, mpdist_percentage, mpdist_k, p))


def mdl(T, m, subseq_idx, nn_idx, include=None, discords=False, discretize_func=None, n_bit=8, normalize=True, p=2.0, T_subseq_isconstant=None):
    """STUMPY's `mdl`: `(bit_sizes, S)`."""
    _check_config()
    return tuple(_native.mdl(T, m, subseq_idx, nn_idx, include, discords, discretize_func, n_bit, normalize, p, T_subseq_isconstant))


def maamp_mdl(T, m, subseq_idx, nn_idx, include=None, discords=False, discretize_func=None, n_bit=8, p=2.0):
    """STUMPY's `maamp_mdl`: `(bit_sizes, S)`."""
    _check_config()
    return tuple(_native.maamp_mdl(T, m, subseq_idx, nn_idx, include, discords, discretize_func, n_bit, p))


def subspace(T, m, subseq_idx, nn_idx, k, include=None, discords=False, discretize_func=None, n_bit=8, normalize=True, p=2.0, T_subseq_isconstant=None):
    """STUMPY's `subspace`: the `k + 1` dimensions the pair is closest in."""
    _check_config()
    return _native.subspace(T, m, subseq_idx, nn_idx, k, include, discords, discretize_func, n_bit, normalize, p, T_subseq_isconstant)


def maamp_subspace(T, m, subseq_idx, nn_idx, k, include=None, discords=False, discretize_func=None, n_bit=8, p=2.0):
    """STUMPY's `maamp_subspace`: the `k + 1` dimensions the pair is closest in."""
    _check_config()
    return _native.maamp_subspace(T, m, subseq_idx, nn_idx, k, include, discords, discretize_func, n_bit, p)


# --- anytime and pan profiles ------------------------------------------------------------------

_WARM_START = "the warm start is not available"


def scrump(T_A, m, T_B=None, ignore_trivial=True, percentage=0.01, pre_scrump=False, s=None, normalize=True, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None):
    """STUMPY's `scrump`: the anytime profile, exact over the diagonals visited so far.
    `pre_scrump=True` raises `NotImplementedError`."""
    _check_config()
    if pre_scrump:
        raise NotImplementedError(f"pre_scrump=True: {_WARM_START}; call scrump without it, which is exact once converged")
    return _native.scrump(T_A, m, T_B, ignore_trivial, percentage, False, s, normalize, p, k, T_A_subseq_isconstant, T_B_subseq_isconstant)


def scraamp(T_A, m, T_B=None, ignore_trivial=True, percentage=0.01, pre_scraamp=False, s=None, p=2.0, k=1):
    """STUMPY's `scraamp`: `scrump` on the non-normalized distance. `pre_scraamp=True` raises
    `NotImplementedError`."""
    _check_config()
    if pre_scraamp:
        raise NotImplementedError(f"pre_scraamp=True: {_WARM_START}; call scraamp without it, which is exact once converged")
    return _native.scraamp(T_A, m, T_B, ignore_trivial, percentage, False, s, p, k)


def stimp(T, min_m=3, max_m=None, step=1, percentage=0.01, pre_scrump=True, normalize=True, p=2.0, T_subseq_isconstant_func=None):
    """STUMPY's `stimp`: the pan matrix profile, every row exact (`percentage` and `pre_scrump`
    cannot change an exact row and are ignored)."""
    _check_config()
    return _native.stimp(T, min_m, max_m, step, percentage, pre_scrump, normalize, p, T_subseq_isconstant_func)


def aamp_stimp(T, min_m=3, max_m=None, step=1, percentage=0.01, pre_scraamp=True, p=2.0):
    """STUMPY's `aamp_stimp`: the non-normalized pan matrix profile, every row exact."""
    _check_config()
    return _native.aamp_stimp(T, min_m, max_m, step, percentage, pre_scraamp, p)


# --- streams -----------------------------------------------------------------------------------


class _Stream:
    """STUMPY's `stumpi`/`aampi` over a native stream that looks ahead: `left_P_` and `left_I_` are
    the stream's own left neighbors, and `P_` and `I_` the two-sided profile STUMPY keeps, which
    the stream maintains as samples arrive (a read copies it; it computes nothing).

    With `egress=False` the two-sided profile is that of every sample so far. With `egress=True`
    a window's neighbors are the windows it shared the last `len(T)` samples with: its left
    neighbor as the stream found it (possibly since egressed, as in STUMPY) and its right one
    among the retained samples; on equal distances the smaller index wins."""

    def __init__(self, native, T, m, egress, k):
        self._native = native
        self._m = m
        self._k = k
        self._egress = egress
        self._T = list(np.asarray(T, dtype=np.float64))

    def update(self, t):
        """Folds in `t`; with `egress=True` the oldest sample leaves `T_`."""
        _check_config()
        self._native.update(t)
        self._T.append(float(t))
        if self._egress:
            del self._T[0]

    @property
    def T_(self):
        return np.asarray(self._T, dtype=np.float64)

    def _windows(self):
        return len(self._T) - self._m + 1

    @property
    def left_P_(self):
        return self._view(self._native.left_P_[-self._windows() :])

    @property
    def left_I_(self):
        return self._view(self._native.left_I_[-self._windows() :])

    def _view(self, values):
        return mparray(np.asarray(values), self._m, self._k, _HONOURED["STUMPY_EXCL_ZONE_DENOM"][0])

    @property
    def P_(self):
        return self._view(self._native.two_sided_P_ if self._k == 1 else self._native.two_sided_topk_P_)

    @property
    def I_(self):
        return self._view(self._native.two_sided_I_ if self._k == 1 else self._native.two_sided_topk_I_)


class _ZNormStream(_Stream):
    """`stumpi` over the z-normalized native stream."""


class _NonNormStream(_Stream):
    """`aampi` over the non-normalized native stream."""


def stumpi(T, m, egress=True, normalize=True, p=2.0, k=1, mp=None, T_subseq_isconstant_func=None):
    """STUMPY's `stumpi`: `update(t)`, `P_`, `I_`, `left_P_`, `left_I_` and `T_`, with global
    window indices. `normalize=False` is `aampi`."""
    _check_config()
    if not normalize:
        return aampi(T, m, egress, p, k, mp)
    native = _native.stumpi(T, m, egress, True, p, k, mp, T_subseq_isconstant_func, look_ahead=True)
    return _ZNormStream(native, T, m, egress, k)


def aampi(T, m, egress=True, p=2.0, k=1, mp=None):
    """STUMPY's `aampi`: `stumpi` on the non-normalized distance."""
    _check_config()
    native = _native.aampi(T, m, egress, p, k, mp, look_ahead=True)
    return _NonNormStream(native, T, m, egress, k)


class _Floss:
    """STUMPY's `floss`: the right nearest neighbor of every window of the sliding `T_` and the
    one-directional corrected arc curve over them. `P_` and `I_` (global indices) are exact and
    computed when read; `cac_1d_` is `-1` until the first `update`. The exclusion zone is
    `ceil(m / 4)` throughout, where STUMPY's update lets a new window be the neighbor of the one
    exactly that far before it. With `custom_iac` the curve
    equals STUMPY's for the same indices; without it the idealized arc curve is a closed-form
    expectation, not STUMPY's seeded beta fit. `mp` is checked for its length only: the right
    neighbors are computed exactly from `T`."""

    def __init__(self, mp, T, m, L, excl_factor, custom_iac, normalize, p, predicate):
        T = np.asarray(T, dtype=np.float64)
        if len(np.asarray(mp)) != len(T) - m + 1:
            raise ValueError(f"mp must have one row per window of T ({len(T) - m + 1}), got {len(np.asarray(mp))}")
        if not normalize and p != 2.0:
            raise NotImplementedError(f"p must be 2.0, got {p}: other values of p are not available yet")
        self._T = list(T)
        self._m, self._L, self._excl_factor = m, L, excl_factor
        self._custom_iac, self._normalize, self._predicate = custom_iac, normalize, predicate
        self._appended = 0
        self._cached = None

    def update(self, t):
        """Ingresses `t` and egresses the oldest sample."""
        _check_config()
        self._T.append(float(t))
        del self._T[0]
        self._appended += 1
        self._cached = None

    @property
    def T_(self):
        return np.asarray(self._T, dtype=np.float64)

    def _right(self):
        """Local right indices (-1 for none) and their distances, exact, ties to the smaller."""
        if self._cached is None:
            T, m = self.T_, self._m
            reverse = T[::-1].copy()
            if self._normalize:
                flags = None
                if self._predicate is not None:
                    flags = np.asarray(self._predicate(np.where(np.isfinite(T), T, np.nan), m), dtype=bool)
                right = _native.matrix_profile(T, m, T_A_subseq_isconstant=flags)[3]
                P = _native.left_profile(reverse, m, T_A_subseq_isconstant=None if flags is None else flags[::-1].copy())[0][::-1]
            else:
                right = _native.aamp(T, m)[3]
                P = _native.aampi(reverse, m, egress=False).left_P_[::-1]
            self._cached = (right, P)
        return self._cached

    @property
    def P_(self):
        return np.asarray(self._right()[1], dtype=np.float64)

    @property
    def I_(self):
        right = self._right()[0]
        return np.where(right < 0, -1, right + self._appended)

    @property
    def cac_1d_(self):
        windows = len(self._T) - self._m + 1
        if self._appended == 0:
            return -np.ones(windows, dtype=np.float64)
        right = self._right()[0]
        if self._custom_iac is not None:
            return _native.fluss(right, self._L, 1, self._excl_factor, self._custom_iac, m=self._m)[0]
        mirrored = np.where(right < 0, -1, windows - 1 - right)[::-1].copy()
        curve = _native._corrected_arc_curve(mirrored, 0, self._m, None, self._L, self._excl_factor)
        return np.asarray(curve, dtype=np.float64)[::-1].copy()


def floss(mp, T, m, L, excl_factor=5, n_iter=1000, n_samples=1000, custom_iac=None, normalize=True, p=2.0, T_subseq_isconstant_func=None):
    """STUMPY's `floss`: `update(t)`, `cac_1d_`, `P_`, `I_` and `T_`. `n_iter` and `n_samples`
    only shape STUMPY's beta fit, which this does not run."""
    _check_config()
    return _Floss(mp, T, m, L, excl_factor, custom_iac, normalize, p, T_subseq_isconstant_func)


# --- names this library does not run ------------------------------------------------------------

_DISTRIBUTED = "Dask and Ray runs are out of scope here (every call already uses every core)"
_GPU = "the exact GPU profile is not available yet"


def stumped(client, T_A, m, T_B=None, ignore_trivial=True, normalize=True, p=2.0, k=1, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None):
    raise NotImplementedError(f"stumped is not available: {_DISTRIBUTED}; use stump")


def aamped(client, T_A, m, T_B=None, ignore_trivial=True, p=2.0, k=1):
    raise NotImplementedError(f"aamped is not available: {_DISTRIBUTED}; use aamp")


def mstumped(client, T, m, include=None, discords=False, p=2.0, normalize=True, T_subseq_isconstant=None):
    raise NotImplementedError(f"mstumped is not available: {_DISTRIBUTED}; use mstump")


def maamped(client, T, m, include=None, discords=False, p=2.0):
    raise NotImplementedError(f"maamped is not available: {_DISTRIBUTED}; use maamp")


def mpdisted(client, T_A, T_B, m, percentage=0.05, k=None, normalize=True, p=2.0, T_A_subseq_isconstant=None, T_B_subseq_isconstant=None):
    raise NotImplementedError(f"mpdisted is not available: {_DISTRIBUTED}; use mpdist")


def aampdisted(client, T_A, T_B, m, percentage=0.05, k=None, p=2.0):
    raise NotImplementedError(f"aampdisted is not available: {_DISTRIBUTED}; use aampdist")


def ostinatoed(client, Ts, m, normalize=True, p=2.0, Ts_subseq_isconstant=None):
    raise NotImplementedError(f"ostinatoed is not available: {_DISTRIBUTED}; use ostinato")


def aamp_ostinatoed(client, Ts, m, p=2.0):
    raise NotImplementedError(f"aamp_ostinatoed is not available: {_DISTRIBUTED}; use aamp_ostinato")


def stimped(client, T, min_m=3, max_m=None, step=1, normalize=True, p=2.0, T_subseq_isconstant_func=None):
    raise NotImplementedError(f"stimped is not available: {_DISTRIBUTED}; use stimp")


def aamp_stimped(client, T, min_m=3, max_m=None, step=1, p=2.0):
    raise NotImplementedError(f"aamp_stimped is not available: {_DISTRIBUTED}; use aamp_stimp")


gpu_stump = _not_here("gpu_stump", _GPU, "stump")
gpu_aamp = _not_here("gpu_aamp", _GPU, "aamp")
gpu_mpdist = _not_here("gpu_mpdist", _GPU, "mpdist")
gpu_aampdist = _not_here("gpu_aampdist", _GPU, "aampdist")
gpu_ostinato = _not_here("gpu_ostinato", _GPU, "ostinato")
gpu_aamp_ostinato = _not_here("gpu_aamp_ostinato", _GPU, "aamp_ostinato")
gpu_stimp = _not_here("gpu_stimp", _GPU, "stimp")
gpu_aamp_stimp = _not_here("gpu_aamp_stimp", _GPU, "aamp_stimp")
