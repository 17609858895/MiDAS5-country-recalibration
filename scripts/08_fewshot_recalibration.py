"""Few-shot country recalibration (empirical-Bayes country offset) under LOCO.

Observation motivating the rule: in LOCO, a large share of digester-level MSE is a country-wide
offset while within-country ranking is preserved.  Model for a new country c:
    y_ic = f(x_ic) + b_c + e_ic,   b_c ~ N(0, tau^2),   e_ic ~ N(0, sigma^2)
Given k labelled digesters in c, the posterior-mean offset is
    b_hat = k*tau^2 / (k*tau^2 + sigma^2) * mean_{labelled}(y - f(x)).
tau^2 and sigma^2 are estimated from INNER leave-one-country-out residuals of the training
countries only (method-of-moments), so no test-country information enters except the k labels.

Comparators at each k: raw model (k=0), unshrunk offset, country-mean-of-k labels only
(no microbiome), and refitting the model with the k labelled digesters added (fine-tuning).
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from common import RES
from invariance import AnchorLGBM
import importlib
bench = importlib.import_module("03_benchmark20")

warnings.filterwarnings("ignore")
METHODS = ["ET", "Anchor_g0", "Anchor_g0.5", "LGBM"]
KS = [0, 1, 2, 3, 5]
N_DRAWS = 50
N_REFIT_DRAWS = 10


def make(method):
    if method == "ET":
        return ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0)
    gamma = 1.0 if method == "LGBM" else float(method.split("_g")[1])
    return AnchorLGBM(gamma=gamma, n_jobs=1)


def fit_predict(method, X, y, cty, tr, te):
    m = make(method)
    if isinstance(m, AnchorLGBM):
        m.fit(X[tr], y[tr], groups=cty[tr])
    else:
        m.fit(X[tr], y[tr])
    return m.predict(X[te])


def dig_avg(v, dig):
    return pd.Series(v).groupby(dig).mean()


def moments(res, cty_d):
    """Method-of-moments tau^2, sigma^2 from digester-level residuals grouped by country."""
    df = pd.DataFrame({"r": res, "c": cty_d})
    g = df.groupby("c").r
    n = g.size()
    sigma2 = (g.var(ddof=1) * (n - 1)).sum() / max((n - 1).sum(), 1)
    means = g.mean()
    tau2 = max(np.average(means ** 2, weights=n) - np.average(sigma2 / n, weights=n), 1e-6)
    return tau2, sigma2


if __name__ == "__main__":
    targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Temperature", "pH", "log_NH4", "log_SRT", "Digester_age"]
    X, T, meta = bench.build_dataset()
    rows = []
    for target in targets:
        ok = T[target].notna().values
        Xi, yi = X.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        outer = list(LeaveOneGroupOut().split(Xi, yi, cty))
        t0 = time.time()
        # outer predictions + inner LOCO residuals for EB moments
        tasks = []
        for k, (tr, te) in enumerate(outer):
            for mth in METHODS:
                tasks.append(("outer", k, mth, tr, te))
                for itr, ite in LeaveOneGroupOut().split(tr, groups=cty[tr]):
                    tasks.append(("inner", k, mth, tr[itr], tr[ite]))
        out = Parallel(n_jobs=30, batch_size=4)(delayed(fit_predict)(m, Xi, yi, cty, a, b) for (_, _, m, a, b) in tasks)
        pred = {m: np.full(len(yi), np.nan) for m in METHODS}
        inner = {(k, m): np.full(len(yi), np.nan) for k in range(len(outer)) for m in METHODS}
        for (kind, k, m, a, b), p in zip(tasks, out):
            if kind == "outer":
                pred[m][b] = p
            else:
                inner[(k, m)][b] = p
        print(target, f"base fits {time.time()-t0:.0f}s", flush=True)
        dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
        y_d = dig_avg(yi, dig)
        rng = np.random.default_rng(0)
        # store base predictions
        pd.to_pickle(dict(pred=pred, y=yi, dig=dig, cty=cty), RES / f"fewshot_preds_{target}.pkl")
        for k_shot in KS:
            for draw in range(N_DRAWS):
                acc = {key: ([], []) for key in
                       ["CountryMeanOfK"] + [f"{m}|{v}" for m in METHODS for v in ["raw", "offset", "EB"]]}
                for kf, (tr, te) in enumerate(outer):
                    c = cty[te][0]
                    dte = pd.unique(dig[te])
                    if len(dte) < k_shot + 2:
                        continue
                    lab = rng.choice(dte, k_shot, replace=False) if k_shot > 0 else np.array([], dtype=object)
                    ev = np.setdiff1d(dte, lab)
                    ytr_mean = y_d.loc[pd.unique(dig[tr])].mean()
                    base = y_d.loc[lab].mean() if k_shot > 0 else ytr_mean
                    acc["CountryMeanOfK"][0].extend(y_d.loc[ev])
                    acc["CountryMeanOfK"][1].extend([base] * len(ev))
                    for m in METHODS:
                        p_d = dig_avg(pred[m][te], dig[te])
                        ip = inner[(kf, m)]
                        msk = ~np.isnan(ip)
                        r_in = dig_avg(yi[msk] - ip[msk], dig[msk])
                        tau2, sig2 = moments(r_in.values, dig_c.loc[r_in.index].values)
                        rbar = (y_d.loc[lab] - p_d.loc[lab]).mean() if k_shot > 0 else 0.0
                        shrink = k_shot * tau2 / (k_shot * tau2 + sig2)
                        for v, off in [("raw", 0.0), ("offset", rbar), ("EB", shrink * rbar)]:
                            acc[f"{m}|{v}"][0].extend(y_d.loc[ev])
                            acc[f"{m}|{v}"][1].extend(p_d.loc[ev] + off)
                for key, (yy, pp) in acc.items():
                    yy, pp = np.array(yy), np.array(pp)
                    rows.append(dict(target=target, k=k_shot, draw=draw, method=key,
                                     R2=1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2),
                                     MAE=np.mean(np.abs(yy - pp)), n_eval=len(yy)))
        # fine-tuning comparator: refit ET with k labelled test-country digesters added (both replicates)
        for k_shot in [k for k in KS if k > 0]:
            jobs = []
            for draw in range(N_REFIT_DRAWS):
                for kf, (tr, te) in enumerate(outer):
                    dte = pd.unique(dig[te])
                    if len(dte) < k_shot + 2:
                        continue
                    lab = rng.choice(dte, k_shot, replace=False)
                    tr2 = np.concatenate([tr, te[np.isin(dig[te], lab)]])
                    te2 = te[~np.isin(dig[te], lab)]
                    jobs.append((draw, tr2, te2))
            outs = Parallel(n_jobs=30)(delayed(fit_predict)("ET", Xi, yi, cty, a, b) for (_, a, b) in jobs)
            for draw in range(N_REFIT_DRAWS):
                yy, pp = [], []
                for (dr, a, b), p in zip(jobs, outs):
                    if dr != draw:
                        continue
                    p_d = dig_avg(p, dig[b])
                    yy.extend(y_d.loc[p_d.index])
                    pp.extend(p_d.values)
                yy, pp = np.array(yy), np.array(pp)
                rows.append(dict(target=target, k=k_shot, draw=draw, method="ET|refit_with_k",
                                 R2=1 - np.sum((yy - pp) ** 2) / np.sum((yy - yy.mean()) ** 2),
                                 MAE=np.mean(np.abs(yy - pp)), n_eval=len(yy)))
        res = pd.DataFrame([r for r in rows if r["target"] == target])
        print(res.groupby(["method", "k"]).R2.mean().unstack().round(3).to_string(), flush=True)
        print(f"  done {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(RES / f"fewshot_recalibration_{'_'.join(targets)}.csv", index=False)
