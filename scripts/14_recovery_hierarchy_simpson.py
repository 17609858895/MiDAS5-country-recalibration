"""Framework v2 checks.

1. Recovery: share of the GKF5 -> LOCO R2 loss that few-shot EB recovers, evaluated on the same digesters.
2. Continent prior: zero-shot offset from inner-LOCO residuals of training countries in the same continent.
3. Between- vs within-country association of log10 TAN with methanogen outcomes (unadjusted OLS slopes).
"""
import importlib
import numpy as np
import pandas as pd
from common import RES, load_meta

fs = importlib.import_module("08_fewshot_recalibration")
dml = importlib.import_module("05_causal_dml")
TARGETS = ["Temperature", "log_NH4", "Digester_age", "pH"]
KS = [0, 1, 2, 3, 5]
N_DRAWS = 50


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def recovery_and_continent(target):
    d = pd.read_pickle(RES / f"conformal_store_{target}.pkl")
    store, yi, dig, cty = d["store"], d["y"], d["dig"], d["cty"]
    oof = pd.read_pickle(RES / "benchmark20_oof.pkl")[(target, "GKF5", "ExtraTrees")]
    dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
    cont = load_meta().groupby("Digester").Continent.first()
    y_d = fs.dig_avg(yi, dig)
    g_d = fs.dig_avg(oof, dig)
    folds = []
    for kf in sorted({k for (kind, k) in store if kind == "outer"}):
        po = store[("outer", kf)]
        te = ~np.isnan(po)
        p_d = fs.dig_avg(po[te], dig[te])
        pi = store[("blocked", kf)]
        msk = ~np.isnan(pi)
        r_in = fs.dig_avg(yi[msk] - pi[msk], dig[msk])
        tau2, sig2 = fs.moments(r_in.values, dig_c.loc[r_in.index].values)
        c = cty[te][0]
        cn = cont.loc[p_d.index[0]]
        same = [dg for dg in r_in.index if cont.loc[dg] == cn]
        # continent prior: mean of country mean residuals of same-continent training countries, EB-shrunk by J_cont
        if same:
            rc = r_in.loc[same].groupby(dig_c.loc[same].values).mean()
            prior = rc.mean() * (len(rc) * tau2 / (len(rc) * tau2 + sig2 / max(len(same) / len(rc), 1)))
        else:
            prior = 0.0
        folds.append(dict(c=c, cont=cn, idx=p_d.index, y=y_d.loc[p_d.index].values, p=p_d.values,
                          g=g_d.loc[p_d.index].values, tau2=tau2, sig2=sig2, prior=prior, n_same=len(same)))
    rng = np.random.default_rng(0)
    rows = []
    for k in KS:
        for draw in range(N_DRAWS if k else 1):
            Y, P, G, E = [], [], [], []
            for f in folds:
                n = len(f["y"])
                if n < k + 2:
                    continue
                lab = rng.choice(n, k, replace=False) if k else np.array([], int)
                ev = np.setdiff1d(np.arange(n), lab)
                sh = k * f["tau2"] / (k * f["tau2"] + f["sig2"]) if k else 0.0
                b = sh * (f["y"][lab] - f["p"][lab]).mean() if k else 0.0
                Y += list(f["y"][ev]); P += list(f["p"][ev]); G += list(f["g"][ev]); E += list(f["p"][ev] + b)
            Y, P, G, E = map(np.array, (Y, P, G, E))
            rows.append(dict(target=target, k=k, draw=draw, R2_GKF5=r2(Y, G), R2_LOCO=r2(Y, P), R2_EB=r2(Y, E),
                             n_eval=len(Y)))
    rec = pd.DataFrame(rows).groupby(["target", "k"]).mean(numeric_only=True).reset_index()
    rec["recovered_share"] = (rec.R2_EB - rec.R2_LOCO) / (rec.R2_GKF5 - rec.R2_LOCO)
    # zero-shot continent prior
    Y = np.concatenate([f["y"] for f in folds]); P = np.concatenate([f["p"] for f in folds])
    C = np.concatenate([f["p"] + f["prior"] for f in folds])
    cp = dict(target=target, R2_LOCO=r2(Y, P), R2_continent_prior=r2(Y, C),
              countries_improved=sum(np.abs(f["y"] - f["p"] - f["prior"]).mean() < np.abs(f["y"] - f["p"]).mean()
                                     for f in folds), countries=len(folds))
    return rec.drop(columns="draw"), cp


def simpson():
    d = dml.digester_table()
    d = d[d.log_TAN.notna()]
    d = d[d.groupby("Country").Country.transform("size") >= 3]
    outs = ["CLR_Methanoculleus", "log_SAOB", "CLR_Methanothrix", "CLR_Methanobacterium", "CLR_Methanospirillum",
            "LR_hydro_vs_Mthrix"]
    rows = []
    for o in outs:
        x = d[["Country", "log_TAN", o]].dropna()
        cm = x.groupby("Country")[["log_TAN", o]].mean()
        w = x.groupby("Country").size()
        xb, yb = cm.log_TAN, cm[o]
        xm, ym = np.average(xb, weights=w), np.average(yb, weights=w)
        between = np.sum(w * (xb - xm) * (yb - ym)) / np.sum(w * (xb - xm) ** 2)
        xd = x.log_TAN - x.groupby("Country").log_TAN.transform("mean")
        yd = x[o] - x.groupby("Country")[o].transform("mean")
        within = np.sum(xd * yd) / np.sum(xd ** 2)
        xp = x.log_TAN - x.log_TAN.mean()
        pooled = np.sum(xp * (x[o] - x[o].mean())) / np.sum(xp ** 2)
        rows.append(dict(outcome=o, n=len(x), countries=len(cm), slope_between=between, slope_within=within,
                         slope_pooled=pooled))
    return pd.DataFrame(rows), d


if __name__ == "__main__":
    recs, cps = [], []
    for t in TARGETS:
        rec, cp = recovery_and_continent(t)
        recs.append(rec); cps.append(cp)
        print(rec.round(3).to_string(index=False), flush=True)
    rec = pd.concat(recs); rec.to_csv(RES / "eb_recovery_same_digesters.csv", index=False)
    cp = pd.DataFrame(cps); cp.to_csv(RES / "continent_prior_zero_shot.csv", index=False)
    print(cp.round(3).to_string(index=False))
    s, d = simpson()
    s.to_csv(RES / "tan_between_within_slopes.csv", index=False)
    print(s.round(3).to_string(index=False))
