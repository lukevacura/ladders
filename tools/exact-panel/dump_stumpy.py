"""Dump STUMPY 1.14.1: stump(T, m) / stump(A, m, B, ignore_trivial=False) as the timed benchmark calls it."""
import numpy as np
import stumpy
from cells import *

d = OUT / "stumpy"
d.mkdir(parents=True, exist_ok=True)
for kind, n, m, wl in cells():
    A = np.fromfile(data_path(kind, n, "A"), dtype="<f8")
    B = np.fromfile(data_path(kind, n, "B"), dtype="<f8")
    out = stumpy.stump(A, m) if wl == "two_sided" else stumpy.stump(A, m, B, ignore_trivial=False)
    k = key(kind, n, m, wl)
    np.save(d / f"{k}.P.npy", np.asarray(out[:, 0], dtype=np.float64))
    np.save(d / f"{k}.I.npy", np.asarray(out[:, 1], dtype=np.int64))
print("stumpy done", stumpy.__version__)
