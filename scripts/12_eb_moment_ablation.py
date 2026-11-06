"""Ablation of the variance-component source in the few-shot EB rule.

tau^2 and sigma^2 estimated from
  blocked : inner leave-one-country-out residuals of the training countries (proposed rule);
  iid     : inner digester-grouped 5-fold residuals (country fingerprint stays in training,
            so the country offset is partly absorbed and tau^2 is expected to be underestimated);
  none    : unshrunk offset (shrinkage = 1);
  raw     : no recalibration, evaluated on the same digesters (reference at each k).
Uses the ExtraTrees predictions cached by 10b_fewshot_conformal.py; no refitting.
"""
import sys
import numpy as np
import pandas as pd
from common import RES
import importlib
fs = importlib.import_module("08_fewshot_recalibration")

KS = [0, 1, 2, 3, 5]
N_DRAWS = 50

if __name__ == "__main__":
    targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Temperature", "pH", "log_NH4", "Digester_age"]
    rows, comp = [], []
    for target in targets:
        d = pd.read_pickle(RES / f"conformal_store_{target}.pkl")
        store, yi, dig, cty = d["store"], d["y"], d["dig"], d["cty"]
        dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
        y_d = fs.dig_avg(yi, dig)
        folds = sorted({k for (kind, k) in store if kind == "outer"})
        prep = []
        for kf in folds:
            po = store[("outer", kf)]
            te = ~np.isnan(po)
            p_d = fs.dig_avg(po[te], dig[te])
            mom = {}
            for kind in ["iid", "blocked"]:
                pp = store[(kind, kf)]
                msk = ~np.isnan(pp)
                r = fs.dig_avg(yi[msk] - pp[msk], dig[msk])
                mom[kind] = fs.moments(r.values, dig_c.loc[r.index].values)
            comp.append(dict(target=target, country=cty[te][0], tau2_blocked=mom["blocked"][0],
                             sig2_blocked=mom["blocked"][1], tau2_iid=mom["iid"][0], sig2_iid=mom["iid"][1]))
            prep.append((y_d.loc[p_d.index].values, p_d.values, mom))
        rng = np.random.default_rng(0)
        for k in KS:
            for draw in range(N_DRAWS):
                acc = {v: ([], []) for v in ["raw", "blocked", "iid", "none"]}
                for yy, pp, mom in prep:
                    if len(yy) < k + 2:
                        continue
                    lab = rng.choice(len(yy), k, replace=False) if k > 0 else np.array([], int)
                    ev = np.setdiff1d(np.arange(len(yy)), lab)
                    rbar = (yy[lab] - pp[lab]).mean() if k > 0 else 0.0
                    for v in acc:
                        sh = {"none": 1.0, "raw": 0.0}.get(v) if v in ("none", "raw") else k * mom[v][0] / (k * mom[v][0] + mom[v][1])
                        acc[v][0].extend(yy[ev])
                        acc[v][1].extend(pp[ev] + (sh * rbar if k > 0 else 0.0))
                for v, (a, b) in acc.items():
                    a, b = np.array(a), np.array(b)
                    rows.append(dict(target=target, k=k, draw=draw, moments=v,
                                     R2=1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)))
        res = pd.DataFrame([r for r in rows if r["target"] == target])
        print(target, "\n", res.groupby(["moments", "k"]).R2.mean().unstack().round(3).to_string(), flush=True)
        c = pd.DataFrame([r for r in comp if r["target"] == target])
        print(c[["tau2_blocked", "sig2_blocked", "tau2_iid", "sig2_iid"]].median().round(4).to_dict(), flush=True)
    pd.DataFrame(rows).to_csv(RES / "eb_moment_ablation.csv", index=False)
    pd.DataFrame(comp).to_csv(RES / "eb_moment_components.csv", index=False)
