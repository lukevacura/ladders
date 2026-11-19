"""Exact matrix profiles for Rust and Python, built for live data streams.

The top level is the native API: this library's names, results as named tuples (`res.P`, `res.I`,
still unpackable as before) and its extras keyword-only. `ladders.stumpy` holds STUMPY
1.14.1's names with STUMPY's signatures and return types, so switching is one line:
`import ladders.stumpy as stumpy`.

Every result is exact for the data quantized to a `scale`. Given a `scale`, a call returns the same
bits in every release. Given none, the scale is inferred by a numbered rule, `SCALE_RULE`, and
the result changes only when that number does; `infer_scale` returns the scale a call would
infer, to record and give back.
"""

import functools
from collections import namedtuple

from . import _native
from ._native import *  # noqa: F401,F403

# The hidden helpers (test hooks among them) stay reachable under their own names.
globals().update(
    {name: value for name, value in vars(_native).items() if name.startswith("_") and not name.startswith("__")}
)

LeftProfile = namedtuple("LeftProfile", ["P", "I", "scale"])
LeftProfile.__doc__ = "`left_profile`'s result: distances, neighbor indices and the scale used."
MatrixProfile = namedtuple("MatrixProfile", ["P", "I", "left_I", "right_I"])
MatrixProfile.__doc__ = "A two-sided profile: distances, indices, and each side's nearest index."
AbJoin = namedtuple("AbJoin", ["P", "I", "scale_a", "scale_b"])
AbJoin.__doc__ = "`ab_join`'s result: distances, indices into `t_b`, and each series' scale."
MultiProfile = namedtuple("MultiProfile", ["P", "I"])
MultiProfile.__doc__ = "A multi-dimensional profile: one row of distances and indices per `k`."
Discords = namedtuple("Discords", ["distances", "indices"])
Discords.__doc__ = "The discords, farthest first."
Fluss = namedtuple("Fluss", ["cac", "regime_locs"])
Fluss.__doc__ = "The corrected arc curve and the regime locations."
Motifs = namedtuple("Motifs", ["motif_distances", "motif_indices"])
Motifs.__doc__ = "One row of match distances and indices per motif."
MotifPairs = namedtuple("MotifPairs", ["indices", "distances", "scale"])
MotifPairs.__doc__ = "The best pair: its two window starts, its distance and the scale used."
MultiMotifs = namedtuple("MultiMotifs", ["motif_distances", "motif_indices", "motif_subspaces", "motif_mdls"])
MultiMotifs.__doc__ = "Multi-dimensional motifs: matches, subspaces and description lengths."
Matches = namedtuple("Matches", ["distances", "indices"])
Matches.__doc__ = "The matches of a query, nearest first."
DistanceProfile = namedtuple("DistanceProfile", ["indices", "distances"])
DistanceProfile.__doc__ = "One index and distance per window of the series."
Snippets = namedtuple("Snippets", ["snippets", "indices", "profiles", "fractions", "areas", "regimes"])
Snippets.__doc__ = "The snippets with their profiles, fractions, areas and regimes."
Consensus = namedtuple("Consensus", ["radius", "Ts_idx", "subseq_idx"])
Consensus.__doc__ = "The consensus motif: its radius, series and window."
AllChains = namedtuple("AllChains", ["S", "C"])
AllChains.__doc__ = "Every maximal chain and the longest one."
Mdl = namedtuple("Mdl", ["bit_sizes", "S"])
Mdl.__doc__ = "The description length and the subspace at every subspace size."


def _named(function, result):
    """`function` returning `result`, a named tuple of what it returned; same name, docstring and
    signature."""

    @functools.wraps(function)
    def call(*args, **kwargs):
        return result(*function(*args, **kwargs))

    return call


_RESULTS = {
    "left_profile": LeftProfile,
    "matrix_profile": MatrixProfile,
    "aamp": MatrixProfile,
    "ab_join": AbJoin,
    "mstump": MultiProfile,
    "maamp": MultiProfile,
    "discords": Discords,
    "fluss": Fluss,
    "fluss_series": Fluss,
    "motifs": Motifs,
    "aamp_motifs": Motifs,
    "motif_pairs": MotifPairs,
    "mmotifs": MultiMotifs,
    "aamp_mmotifs": MultiMotifs,
    "match": Matches,
    "aamp_match": Matches,
    "distance_profile": DistanceProfile,
    "snippets": Snippets,
    "aampdist_snippets": Snippets,
    "ostinato": Consensus,
    "aamp_ostinato": Consensus,
    "allc": AllChains,
    "chains": AllChains,
    "mdl": Mdl,
    "maamp_mdl": Mdl,
}

for _name, _result in _RESULTS.items():
    globals()[_name] = _named(getattr(_native, _name), _result)
del _name, _result
