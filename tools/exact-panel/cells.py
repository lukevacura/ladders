"""Shared cell list and paths for the panel (stdlib + numpy only)."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("PANEL_OUT", HERE / "out"))
KINDS = ("grid", "float", "ties", "messy_b10")
NS = tuple(int(x) for x in os.environ.get("PANEL_NS", "512,1024").split(","))
MS = (16, 64)
WORKLOADS = ("two_sided", "ab_join")


def key(kind, n, m, wl):
    return f"{kind}_n{n}_m{m}_{wl}"


def cells():
    return [(k, n, m, wl) for wl in WORKLOADS for k in KINDS for n in NS for m in MS]


def data_path(kind, n, side):
    return OUT / "data" / f"{kind}_n{n}_{side}.f64"
