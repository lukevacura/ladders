"""Classify every row of every engine/cell against exact truth. Numbers only.
Per row (same neighbor): error bins of |P - d_true| / d_true (absolute where d_true = 0):
  e0 (0 ulp) | e4 (1..4 ulp) | e12 (<=1e-12) | e9 (<=1e-9) | e6 (<=1e-6) | gt6.  ulps are counted in P's dtype
  (float32 for SCAMP, against the truth rounded to float32; all others float64 against the correctly rounded truth).
Different neighbor: tie (exact distance equal to the minimum) | farther, binned by relative gap
  (sqrt(d2_chosen/d2_true) - 1): f12 | f9 | f6 | fgt6  | edge (two-sided, inside |i-j| <= ceil(m/4)) | bad_neighbor.
Answers: both_none | na_engine_only (engine answers, truth none) | na_truth_only (truth answers, engine none).
Engine 'no answer' = I < 0 or P not finite.
Sanity (contract group): same answer rows, contract's neighbor, P within 4 ulp of the correctly rounded distance."""
import json, math, sys
from fractions import Fraction
import numpy as np
import exact_ref as X
import truth as T
from cells import *

ENGINES = [("ours_default", "ours/default", "exact"), ("ours_finest", "ours/finest", "exact"),
           ("stumpy", "stumpy", "exact"), ("scamp", "scamp", "exact")]
# An engine whose dump is absent (for example SCAMP, when pyscamp is not installed) is left out.
ENGINES = [e for e in ENGINES if (OUT / e[1]).is_dir()]
CONTRACT = [("ours_default", "ours/default", "contract_default"), ("ours_finest", "ours/finest", "contract_finest")]
EB = ["e0", "e4", "e12", "e9", "e6", "gt6"]
FB = ["f12", "f9", "f6", "fgt6"]
CLASSES = EB + ["tie"] + FB + ["edge", "bad_neighbor", "na_engine_only", "na_truth_only", "both_none"]


def ulpdist(p, t, dt):
    it = np.int32 if dt is np.float32 else np.int64
    a = np.array([p], dtype=dt).view(it)[0]
    b = np.array([t], dtype=dt).view(it)[0]
    return abs(int(a) - int(b))


def ebin(rel):
    return "e12" if rel <= 1e-12 else "e9" if rel <= 1e-9 else "e6" if rel <= 1e-6 else "gt6"


def fbin(g):
    return "f12" if g <= 1e-12 else "f9" if g <= 1e-9 else "f6" if g <= 1e-6 else "fgt6"


def near_nan(arr, i, m):
    lo, hi = max(0, i - m), min(len(arr), i + 2 * m)
    return bool(np.isnan(arr[lo:hi]).any())


def classify_cell(kind, n, m, wl, edir, tname):
    k = key(kind, n, m, wl)
    P = np.load(OUT / edir / f"{k}.P.npy")
    I = np.load(OUT / edir / f"{k}.I.npy")
    dt = P.dtype.type
    tr = json.load(open(OUT / "truth" / f"{k}.{tname}.json"))
    A = T.load(kind, n, "A")
    B = T.load(kind, n, "B") if wl == "ab_join" else None
    if tname == "exact":
        X.V_LIMIT = T.V_LIMIT_OFF
        ta, tb = T.exact_ticks(A)[0], (T.exact_ticks(B)[0] if B is not None else None)
    else:
        sc = np.load(OUT / edir / f"{k}.scale.npy")
        X.V_LIMIT = T.V_LIMIT_CONTRACT
        ta = T.contract_ticks(A, sc[0]); tb = T.contract_ticks(B, sc[1]) if B is not None else None
    WA = X.windows(ta, m)
    WB = X.windows(tb, m) if tb is not None else WA
    BA = B if B is not None else A
    ez = X.exclusion_zone(m)
    c = {x: 0 for x in CLASSES}
    c.update(rows=len(WA), same_nbr=0, max_rel_err_same=0.0, max_ulp_same=0, farther_from_zero=0,
             max_abs_err=0.0, max_abs_err_const=0.0)
    c["worst_farther"] = None
    c["worst_mismatch"] = None
    c["sanity_fail"] = []
    c["far_rows"] = []

    def detail(i, j, why, extra=None):
        tj = tr["idx"][i]
        d = dict(kind=kind, n=n, m=m, wl=wl, row=i, why=why, engine_P=float(P[i]), engine_I=int(I[i]),
                 true_P=tr["d"][i], true_I=tj, row_window=WA[i].kind, row_near_nan=near_nan(A, i, m),
                 engine_nbr_window=(WB[j].kind if 0 <= j < len(WB) else None),
                 engine_nbr_near_nan=(near_nan(BA, j, m) if 0 <= j < len(WB) else None),
                 true_nbr_window=(WB[tj].kind if tj >= 0 else None),
                 true_nbr_near_nan=(near_nan(BA, tj, m) if tj >= 0 else None))
        if extra: d.update(extra)
        return d

    for i in range(len(WA)):
        t_idx = tr["idx"][i]
        e_none = int(I[i]) < 0 or not math.isfinite(float(P[i]))
        t_none = t_idx < 0
        if t_none and e_none:
            c["both_none"] += 1; continue
        if t_none != e_none:
            c["na_truth_only" if e_none else "na_engine_only"] += 1
            w = detail(i, int(I[i]), "truth_only" if e_none else "engine_only")
            sev = tr["d"][i] if e_none else (float(P[i]) if math.isfinite(float(P[i])) else float("inf"))
            if c["worst_mismatch"] is None or sev > c["worst_mismatch"][0]:
                c["worst_mismatch"] = (sev, w)
            c["sanity_fail"].append(w)
            continue
        j = int(I[i])
        dexact = tr["d"][i]
        dtrue = Fraction(tr["d2"][i].split("/")[0]) / Fraction(tr["d2"][i].split("/")[1])
        err = abs(float(Fraction(float(P[i])) - Fraction(dexact)))
        kk = "max_abs_err_const" if WA[i].kind == X.CONSTANT else "max_abs_err"
        c[kk] = max(c[kk], err)
        if j >= len(WB) or WB[j].kind == X.NONFINITE:
            c["bad_neighbor"] += 1; c["sanity_fail"].append(detail(i, j, "bad_neighbor")); continue
        if wl == "two_sided" and abs(i - j) <= ez:
            c["edge"] += 1; c["sanity_fail"].append(detail(i, j, "edge")); continue
        if j == t_idx:
            c["same_nbr"] += 1
            tgt = tr["d32"][i] if dt is np.float32 else dexact
            u = ulpdist(P[i], tgt, dt)
            c["max_ulp_same"] = max(c["max_ulp_same"], u)
            rel = err / dexact if dexact > 0 else err
            c["max_rel_err_same"] = max(c["max_rel_err_same"], rel)
            c["e0" if u == 0 else "e4" if u <= 4 else ebin(rel)] += 1
            if u > 4:
                c["sanity_fail"].append(detail(i, j, "ulp>4", dict(ulps=u, target=tgt)))
            else:
                c.setdefault("sanity_max_ulp", 0); c["sanity_max_ulp"] = max(c["sanity_max_ulp"], ulpdist(P[i], dexact, np.float64) if dt is np.float64 else u)
            continue
        c["sanity_fail"].append(detail(i, j, "other_neighbor"))
        r = X.rho(WA[i], WB[j], m)
        tb_ = tr["rho"][i]
        best = X.Rho(tb_[0], int(tb_[1]), int(tb_[2]))
        if r.cmp(best) == 0:
            c["tie"] += 1
            continue
        d2c = X.d2_of(r, m)
        if dtrue == 0:
            c["farther_from_zero"] += 1; g = float("inf")
        else:
            x = float((d2c - dtrue) / dtrue)
            g = x / (math.sqrt(1 + x) + 1)
        c[fbin(g)] += 1
        c["far_rows"].append(dict(detail(i, j, "farther"), bin=fbin(g)))
        if c["worst_farther"] is None or g > c["worst_farther"][0]:
            c["worst_farther"] = (g, detail(i, j, "farther", dict(rel_gap=g if math.isfinite(g) else "inf",
                                                                  engine_nbr_d=float(X.dist_f64(d2c)))))
    return c


def _job(grp, name, edir, tn, cell):
    return classify_cell(*cell, edir, tn)


def fmt(x):
    return f"{x:.3g}" if isinstance(x, float) else str(x)


def main():
    res = {"exact": {}, "contract": {}}
    jobs = []
    for cell in cells():
        for name, edir, tn in ENGINES:
            jobs.append(("exact", name, edir, tn, cell))
        for name, edir, tn in CONTRACT:
            jobs.append(("contract", name, edir, tn, cell))
    from multiprocessing import Pool
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        outs = pool.starmap(_job, jobs)
    for (grp, name, edir, tn, cell), c in zip(jobs, outs):
        c["cell"] = key(*cell)
        res[grp].setdefault(cell[3], {})[f"{name}|{cell[0]}|n{cell[1]}|m{cell[2]}"] = c
    json.dump(res, open(OUT / "classify.json", "w"), indent=1, default=str)

    md = []
    # sanity
    fails, maxulp, rows = [], 0, 0
    for wl, d in res["contract"].items():
        for ek, c in d.items():
            fails += [dict(cell=ek, wl=wl, **w) for w in c["sanity_fail"]]
            maxulp = max(maxulp, c["max_ulp_same"]); rows += c["rows"]
    if fails:
        sline = f"SANITY HARD FAILURE: {len(fails)} rows; first: {json.dumps(fails[0], default=str)}"
    else:
        sline = f"SANITY PASS: ours default+finest vs contract, {rows} rows, every neighbor and answer equal, largest ulp distance seen {maxulp} (limit 4)"
    md.append(sline)
    # detailed tables
    for wl in WORKLOADS:
        md.append(f"\n### {wl}: vs exact truth of the doubles as passed (SCAMP ulps: float32, truth rounded to float32)\n")
        cols = CLASSES + ["rows", "max_rel_err_same"]
        md.append("| engine | kind | n | m | " + " | ".join(cols) + " |")
        md.append("|" + "---|" * (len(cols) + 4))
        for ek, c in res["exact"][wl].items():
            eng, kind, n, m = ek.split("|")
            md.append(f"| {eng} | {kind} | {n[1:]} | {m[1:]} | " + " | ".join(fmt(c[x]) for x in cols) + " |")
    # summaries
    summ = {}
    for wl in WORKLOADS:
        md.append(f"\n### SUMMARY {wl}: every line against the exact truth of the doubles as passed; cells summed over n, m\n")
        cols = ["engine", "kind", "rows", "answered", "correct", "share", "near_tie_miss", "wrong_nbr", "na_engine_only", "na_truth_only", "max_rel_err_same"]
        md.append("| " + " | ".join(cols) + " |"); md.append("|" + "---|" * len(cols))
        agg = {}
        for ek, c in res["exact"][wl].items():
            eng, kind = ek.split("|")[:2]
            a = agg.setdefault((eng, kind), dict(rows=0, answered=0, correct=0, near=0, wrong=0, ne=0, nt=0, mx=0.0))
            ans = c["rows"] - c["both_none"] - c["na_engine_only"]
            a["rows"] += c["rows"]; a["answered"] += ans
            a["correct"] += c["same_nbr"] + c["tie"]
            a["near"] += c["f12"] + c["f9"]
            a["wrong"] += c["f6"] + c["fgt6"] + c["edge"] + c["bad_neighbor"]
            a["ne"] += c["na_engine_only"]; a["nt"] += c["na_truth_only"]
            a["mx"] = max(a["mx"], c["max_rel_err_same"])
        for (eng, kind), a in agg.items():
            md.append(f"| {eng} | {kind} | {a['rows']} | {a['answered']} | {a['correct']} | {a['correct']/a['answered']:.4f} | {a['near']} | {a['wrong']} | {a['ne']} | {a['nt']} | {a['mx']:.3g} |")
        summ[wl] = {f"{e}|{k}": v for (e, k), v in agg.items()}
    # worst rows per engine
    worst = {}
    for wl in WORKLOADS:
        for ek, c in res["exact"][wl].items():
            eng = ek.split("|")[0]
            w = worst.setdefault(eng, {"farther": None, "mismatch": None})
            if c["worst_farther"] and (w["farther"] is None or float(c["worst_farther"][0]) > float(w["farther"][0])):
                w["farther"] = c["worst_farther"]
            if c["worst_mismatch"] and (w["mismatch"] is None or c["worst_mismatch"][0] > w["mismatch"][0]):
                w["mismatch"] = c["worst_mismatch"]
    md.append("\n### worst rows per engine (farther: largest relative gap; mismatch: largest true distance if truth answers, else largest P)\n")
    for eng, w in worst.items():
        for t_, v in w.items():
            md.append(f"- {eng} {t_}: " + (json.dumps(v[1], default=str) if v else "none"))
    json.dump(dict(summary=summ, worst={e: {k: (v[1] if v else None) for k, v in w.items()} for e, w in worst.items()}),
              open(OUT / "summary.json", "w"), indent=1, default=str)
    (OUT / "classify.md").write_text("\n".join(md) + "\n")
    print(sline)
    if fails:
        for w in fails[:5]: print(json.dumps(w, default=str))
        sys.exit(1)


if __name__ == "__main__":
    main()
