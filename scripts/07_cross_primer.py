"""Cross-primer transfer: train on one amplicon, deploy on another amplicon of held-out countries.

Bacterial genera shared by V1-3 and V4 form a common feature space; CLR is computed on that
shared genus set in each amplicon separately (same reference set -> comparable log-ratios).
Primer-concordance screening: per genus, Lin's concordance correlation between V1-3 and V4 CLR
on TRAINING samples only; genera below a concordance threshold are dropped before fitting.
Scenarios (all leave-one-country-out):
  V13->V13 (matched), V4->V4 (matched), V13->V4 (primer shift), V13+V4 pooled ->V4 (primer augmentation),
  V13->V4 with concordance screening, pooled + screening.
"""
import time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from common import load_meta, load_counts, rel_abund, MIN_DEPTH, RES
from models import metrics, digester_level

warnings.filterwarnings("ignore")


def shared_clr(min_prev=0.1):
    c13, c4 = load_counts("V13", "g"), load_counts("V4", "g")
    c13, c4 = c13[c13.sum(1) >= MIN_DEPTH], c4[c4.sum(1) >= MIN_DEPTH]
    s = c13.index.intersection(c4.index)
    r13, r4 = rel_abund(c13.loc[s]), rel_abund(c4.loc[s])
    common = [g for g in r13.columns.intersection(r4.columns) if not g.startswith("uc_")]
    keep = [g for g in common if (r13[g] > 0).mean() >= min_prev and (r4[g] > 0).mean() >= min_prev]

    def clr(r):
        x = np.log(r[keep] + 1e-5)
        return x.sub(x.mean(1), axis=0)
    return clr(r13), clr(r4), keep


def lin_ccc(a, b):
    ma, mb = a.mean(0), b.mean(0)
    va, vb = a.var(0), b.var(0)
    cov = ((a - ma) * (b - mb)).mean(0)
    return 2 * cov / (va + vb + (ma - mb) ** 2)


def run(scn, A13, A4, y, cty, tr, te, thr=0.5):
    cols = np.arange(A13.shape[1])
    if "screen" in scn:
        ccc = lin_ccc(A13[tr], A4[tr])
        cols = np.where(ccc >= thr)[0]
    src, dst = scn.split("_")[0].split("->")
    Xte = (A4 if dst == "V4" else A13)[te][:, cols]
    if src == "pool":
        Xtr = np.vstack([A13[tr][:, cols], A4[tr][:, cols]])
        ytr = np.concatenate([y[tr], y[tr]])
    else:
        Xtr = (A4 if src == "V4" else A13)[tr][:, cols]
        ytr = y[tr]
    m = ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0)
    m.fit(Xtr, ytr)
    return te, m.predict(Xte), len(cols)


if __name__ == "__main__":
    C13, C4, genera = shared_clr()
    meta = load_meta().loc[C13.index]
    print("shared samples", len(C13), "shared genera", len(genera))
    ccc_all = pd.Series(lin_ccc(C13.values, C4.values), index=genera)
    ccc_all.to_csv(RES / "primer_concordance_V13_V4.csv", header=["CCC"])
    print("CCC quantiles", ccc_all.quantile([.1, .25, .5, .75, .9]).round(2).to_dict())
    T = pd.DataFrame({"Temperature": meta.Temperature, "pH": meta.pH,
                      "log_NH4": np.log10(meta.Ammonium.where(meta.Ammonium > 0))})
    scenarios = ["V13->V13", "V4->V4", "V13->V4", "pool->V4", "V13->V4_screen", "pool->V4_screen"]
    rows = []
    for target in T.columns:
        ok = T[target].notna().values
        A13, A4, y = C13.values[ok], C4.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        splits = list(LeaveOneGroupOut().split(A13, y, cty))
        tasks = [(s, k) for s in scenarios for k in range(len(splits))]
        out = Parallel(n_jobs=30)(delayed(run)(s, A13, A4, y, cty, *splits[k]) for s, k in tasks)
        preds = {s: np.full(len(y), np.nan) for s in scenarios}
        nf = {s: [] for s in scenarios}
        for (s, k), (te, p, n) in zip(tasks, out):
            preds[s][te] = p
            nf[s].append(n)
        for s, p in preds.items():
            yd, pdg, _ = digester_level(p, y, dig)
            rows.append(dict(target=target, scenario=s, n_features=np.median(nf[s]), **metrics(yd, pdg)))
        print(pd.DataFrame([r for r in rows if r["target"] == target]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RES / "cross_primer_LOCO.csv", index=False)
