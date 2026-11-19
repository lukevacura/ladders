"""Split every row that disagrees with the exact truth by cause. Reads classify.json.

Arithmetic:
  farther      the chosen neighbor's true distance exceeds the nearest by more than 1e-6
               (relative), and the row and both neighbors are ordinary (non-constant) windows
  near-tie     another neighbor, within 1e-12 (relative) of the nearest
Convention (the truth follows STUMPY's rules, which an engine may define differently):
  zone edge    the chosen neighbor is inside the exclusion zone |i - j| <= ceil(m/4)
  constant     the row or the true neighbor is a constant window (no answer given, or another
               neighbor chosen)
"""
import collections, json
from cells import *

res = json.load(open(OUT / "classify.json"))["exact"]
t = collections.defaultdict(collections.Counter)
for wl, engines in res.items():
    for ek, c in engines.items():
        e = t[ek.split("|")[0]]
        e["rows"] += c["rows"]
        e["near-tie"] += c["f12"]
        e["zone edge"] += c["edge"]
        e["other"] += c["f9"] + c["f6"] + c["bad_neighbor"] + c["na_engine_only"]
        for d in c["sanity_fail"]:
            if d["why"] == "edge":
                e["zone edge, at exactly m/4"] += abs(d["row"] - d["engine_I"]) * 4 == d["m"]
            if d["why"] == "truth_only":
                const = d["row_window"] == "constant"
                e["constant" if const else "no answer, ordinary row"] += 1
        for d in c["far_rows"]:
            if d["bin"] != "fgt6":
                continue
            const = "constant" in (d["row_window"], d["true_nbr_window"], d["engine_nbr_window"])
            e["constant" if const else "farther"] += 1
            if not const:
                e["farther, near a gap"] += bool(d["row_near_nan"])
                e["farther, " + d["kind"]] += 1
cols = ["rows", "farther", "near-tie", "zone edge", "constant", "no answer, ordinary row", "other"]
print("| engine | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 1))
for name, e in t.items():
    print(f"| {name} | " + " | ".join(str(e[c]) for c in cols) + " |")
print()
for name, e in t.items():
    extra = {k: v for k, v in e.items() if k not in cols}
    if extra:
        print(name, extra)
