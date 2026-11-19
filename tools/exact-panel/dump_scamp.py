"""Dump pyscamp: selfjoin / abjoin with precision=double, pearson=False (the timed benchmark's settings).
P comes back float32."""
import numpy as np
import pyscamp
from cells import *

KW = {"precision": "double", "pearson": False, "verbose": False}
d = OUT / "scamp"
d.mkdir(parents=True, exist_ok=True)
for kind, n, m, wl in cells():
    A = np.fromfile(data_path(kind, n, "A"), dtype="<f8")
    B = np.fromfile(data_path(kind, n, "B"), dtype="<f8")
    p, i = (pyscamp.selfjoin(A, m, threads=1, **KW) if wl == "two_sided"
            else pyscamp.abjoin(A, B, m, threads=1, **KW))
    k = key(kind, n, m, wl)
    np.save(d / f"{k}.P.npy", np.asarray(p))  # keeps float32
    np.save(d / f"{k}.I.npy", np.asarray(i, dtype=np.int64))
print("scamp done")
