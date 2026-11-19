"""Dump ladders (this library's Python package): default scale and scale='finest'. Run with the
interpreter that has it installed.
Saves <OUT>/ours/<default|finest>/<cell>.{P,I,scale}.npy.
The two-sided result carries no scale, so it is read from left_profile (same inference, depends only
on t and m); the profile is then re-run with that scale passed explicitly and required bit-equal."""
import numpy as np
import ladders as mp
from cells import *


def load(kind, n, side):
    return np.fromfile(data_path(kind, n, side), dtype="<f8")


for kind, n, m, wl in cells():
    A = load(kind, n, "A")
    B = load(kind, n, "B")
    for var, sc in (("default", None), ("finest", "finest")):
        d = OUT / "ours" / var
        d.mkdir(parents=True, exist_ok=True)
        k = key(kind, n, m, wl)
        if wl == "two_sided":
            r = mp.matrix_profile(A, m, scale=sc, threads=1)
            s = mp.left_profile(A, m, scale=sc, threads=1).scale
            r2 = mp.matrix_profile(A, m, scale=s, threads=1)
            assert np.array_equal(r.P.view(np.uint64), r2.P.view(np.uint64)) and np.array_equal(r.I, r2.I), k
            P, I, scales = r.P, r.I, [s]
        else:
            r = mp.ab_join(A, B, m, scale_a=sc, scale_b=sc, threads=1)
            P, I, scales = r.P, r.I, [r.scale_a, r.scale_b]
        np.save(d / f"{k}.P.npy", np.asarray(P, dtype=np.float64))
        np.save(d / f"{k}.I.npy", np.asarray(I, dtype=np.int64))
        np.save(d / f"{k}.scale.npy", np.asarray(scales, dtype=np.float64))
print("ours done")
