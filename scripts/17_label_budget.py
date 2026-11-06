"""A priori label budget for the few-shot EB rule.

Under y_ic = f(x_ic) + b_c + e_ic with b_c ~ N(0, tau^2) and e_ic ~ N(0, sigma^2), the EB offset
b_hat = w_k * rbar_L with w_k = k tau^2 / (k tau^2 + sigma^2) leaves (1 - w_k) tau^2 of the offset
variance.  The expected reduction of the pooled MSE is therefore w_k tau^2 per evaluated digester,
and the expected R2 gain is sum_c n_c w_k,c tau2_c / SST.  tau2_c and sigma2_c are the inner-LOCO
moments of the training countries of fold c. The unstandardised expected MSE reduction and k_q
are available before local labels; the R2 standardisation uses the evaluation target SST and
is a retrospective diagnostic. It is compared with the observed gain on the same digesters and
draws as in 14_recovery_hierarchy_simpson.py.

The number of labels needed to remove a share q of the offset variance is k_q = q/(1-q) sigma2/tau2.
"""
import importlib
import numpy as np
import pandas as pd
from common import RES

fs = importlib.import_module("08_fewshot_recalibration")
TARGETS = ["Temperature", "log_NH4", "Digester_age", "pH"]
KS = [0, 1, 2, 3, 5]
N_DRAWS = 50


def folds_for(target):
    d = pd.read_pickle(RES / f"conformal_store_{target}.pkl")
    store, yi, dig, cty = d["store"], d["y"], d["dig"], d["cty"]
    oof = pd.read_pickle(RES / "benchmark20_oof.pkl")[(target, "GKF5", "ExtraTrees")]
    dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
    y_d, g_d = fs.dig_avg(yi, dig), fs.dig_avg(oof, dig)
    out = []
    for kf in sorted({k for (kind, k) in store if kind == "outer"}):
        po = store[("outer", kf)]
        te = ~np.isnan(po)
        p_d = fs.dig_avg(po[te], dig[te])
        pi = store[("blocked", kf)]
        msk = ~np.isnan(pi)
        r_in = fs.dig_avg(yi[msk] - pi[msk], dig[msk])
        tau2, sig2 = fs.moments(r_in.values, dig_c.loc[r_in.index].values)
        out.append(dict(c=cty[te][0], y=y_d.loc[p_d.index].values, p=p_d.values,
                        g=g_d.loc[p_d.index].values, tau2=tau2, sig2=sig2))
    return out


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def budget(target):
    folds = folds_for(target)
    rng = np.random.default_rng(0)
    rows = []
    for k in KS:
        for draw in range(N_DRAWS if k else 1):
            Y, P, G, E, gain = [], [], [], [], 0.0
            off_before, off_after, n_tot = 0.0, 0.0, 0
            for f in folds:
                n = len(f["y"])
                if n < k + 2:
                    continue
                lab = rng.choice(n, k, replace=False) if k else np.array([], int)
                ev = np.setdiff1d(np.arange(n), lab)
                w = k * f["tau2"] / (k * f["tau2"] + f["sig2"]) if k else 0.0
                b = w * (f["y"][lab] - f["p"][lab]).mean() if k else 0.0
                r_ev = f["y"][ev] - f["p"][ev]
                Y += list(f["y"][ev]); P += list(f["p"][ev]); G += list(f["g"][ev]); E += list(f["p"][ev] + b)
                gain += len(ev) * w * f["tau2"]
                off_before += len(ev) * r_ev.mean() ** 2
                off_after += len(ev) * (r_ev.mean() - b) ** 2
                n_tot += len(ev)
            Y, P, G, E = map(np.array, (Y, P, G, E))
            sst = np.sum((Y - Y.mean()) ** 2)
            rows.append(dict(target=target, k=k, draw=draw, R2_GKF5=r2(Y, G), R2_LOCO=r2(Y, P), R2_EB=r2(Y, E),
                             dR2_expected=gain / sst, offset_LOCO=off_before / sst, offset_EB=off_after / sst,
                             n_eval=len(Y)))
    df = pd.DataFrame(rows).groupby(["target", "k"]).mean(numeric_only=True).reset_index().drop(columns="draw")
    df["dR2_observed"] = df.R2_EB - df.R2_LOCO
    loss = df.R2_GKF5 - df.R2_LOCO
    df["recovered_observed"] = df.dR2_observed / loss
    df["recovered_expected"] = df.dR2_expected / loss
    t2 = np.median([f["tau2"] for f in folds]); s2 = np.median([f["sig2"] for f in folds])
    comp = dict(target=target, tau2=t2, sigma2=s2, ratio=t2 / s2,
                **{f"k{int(q*100)}": q / (1 - q) * s2 / t2 for q in (0.5, 0.8, 0.9)})
    return df, comp


if __name__ == "__main__":
    res, comps = zip(*(budget(t) for t in TARGETS))
    res = pd.concat(res, ignore_index=True)
    comps = pd.DataFrame(comps)
    res.to_csv(RES / "label_budget_expected_vs_observed.csv", index=False)
    comps.to_csv(RES / "label_budget_k_required.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(comps.round(3).to_string(index=False))
