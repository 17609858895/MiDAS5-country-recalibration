"""Shift-aware stacking under leave-one-country-out (LOCO) evaluation.

Standard super learner: meta-weights fitted on out-of-fold predictions from random
digester-grouped folds (i.i.d. assumption).  Shift-aware super learner: meta-weights fitted on
inner leave-one-country-out predictions, so that the stacking objective matches the deployment
shift (new country).  Meta-learner = convex (non-negative, sum-to-one) weights.
Fair single-model baseline = base learner selected by inner-CV error (not by outer test error).
"""
import sys, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from common import RES
from models import model_zoo, metrics, digester_level
from invariance import AnchorLGBM
import importlib
bench = importlib.import_module("03_benchmark20")

warnings.filterwarnings("ignore")
BASES = ["Ridge", "KernelRidge", "SVR", "GPR", "RF", "ExtraTrees", "HistGB", "XGBoost", "LightGBM", "CatBoost"]


def make_base(name):
    if name.startswith("Anchor"):
        return AnchorLGBM(gamma=float(name.split("_g")[1]), n_jobs=1)
    return clone(model_zoo(n_jobs=1)[name])


def fit_pred(name, X, y, g, tr, te):
    m = make_base(name)
    if name.startswith("Anchor"):
        m.fit(X[tr], y[tr], groups=g[tr])
    else:
        m.fit(X[tr], y[tr])
    return np.asarray(m.predict(X[te])).ravel()


def convex_weights(P, y):
    w, _ = nnls(P, y)
    if w.sum() == 0:
        w = np.ones(P.shape[1])
    return w / w.sum()


def run_target(X, y, dig, cty, bases, n_jobs=30):
    outer = list(LeaveOneGroupOut().split(X, y, cty))
    tasks = []  # (outer k, scheme, base, inner tr idx, inner te idx) in global index space
    for k, (tr, te) in enumerate(outer):
        for scheme in ["iid", "shift"]:
            inner = (GroupKFold(5).split(tr, groups=dig[tr]) if scheme == "iid"
                     else LeaveOneGroupOut().split(tr, groups=cty[tr]))
            for itr, ite in inner:
                for b in bases:
                    tasks.append((k, scheme, b, tr[itr], tr[ite]))
        for b in bases:
            tasks.append((k, "final", b, tr, te))
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs, batch_size=4)(
        delayed(fit_pred)(b, X, y, cty, a, c) for (_, _, b, a, c) in tasks)
    print(f"  {len(tasks)} fits in {time.time()-t0:.0f}s", flush=True)
    oof = {(k, s): {b: np.full(len(y), np.nan) for b in bases} for k in range(len(outer)) for s in ["iid", "shift"]}
    final = {b: np.full(len(y), np.nan) for b in bases}
    for (k, s, b, a, c), p in zip(tasks, out):
        if s == "final":
            final[b][c] = p
        else:
            oof[(k, s)][b][c] = p
    preds = {b: final[b].copy() for b in bases}
    preds.update({"Average": np.full(len(y), np.nan), "SelectBest_iid": np.full(len(y), np.nan),
                  "SelectBest_shift": np.full(len(y), np.nan), "Stack_iid": np.full(len(y), np.nan),
                  "Stack_shift": np.full(len(y), np.nan)})
    weights = []
    for k, (tr, te) in enumerate(outer):
        F = np.column_stack([final[b][te] for b in bases])
        preds["Average"][te] = F.mean(1)
        for s in ["iid", "shift"]:
            P = np.column_stack([oof[(k, s)][b][tr] for b in bases])
            mse = ((P - y[tr, None]) ** 2).mean(0)
            preds[f"SelectBest_{s}"][te] = F[:, np.argmin(mse)]
            w = convex_weights(P, y[tr])
            preds[f"Stack_{s}"][te] = F @ w
            weights.append(dict(fold=k, country=cty[te][0], scheme=s, **dict(zip(bases, w))))
    return preds, pd.DataFrame(weights)


if __name__ == "__main__":
    targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Temperature", "pH", "log_NH4"]
    extra = sys.argv[2].split(",") if len(sys.argv) > 2 else []
    bases = BASES + extra
    X, T, meta = bench.build_dataset()
    rows, allw = [], []
    for target in targets:
        ok = T[target].notna().values
        Xi, yi = X.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        print(target, flush=True)
        preds, W = run_target(Xi, yi, dig, cty, bases)
        W["target"] = target
        allw.append(W)
        for name, p in preds.items():
            yd, pdg, dd = digester_level(p, yi, dig)
            cd = pd.Series(cty, index=dig).groupby(level=0).first().loc[dd].values
            per = pd.DataFrame({"y": yd, "p": pdg, "c": cd}).groupby("c").apply(lambda g: np.mean(np.abs(g.y - g.p)))
            rows.append(dict(target=target, method=name, **metrics(yd, pdg), country_MAE_median=per.median()))
        res = pd.DataFrame([r for r in rows if r["target"] == target]).sort_values("R2", ascending=False)
        print(res.round(3).to_string(index=False), flush=True)
        pd.to_pickle(dict(preds=preds, y=yi, dig=dig, cty=cty), RES / f"stacking_LOCO_preds_{target}.pkl")
    tag = "_".join(targets) + ("_" + "_".join(extra) if extra else "")
    pd.DataFrame(rows).to_csv(RES / f"stacking_LOCO_{tag}.csv", index=False)
    pd.concat(allw).to_csv(RES / f"stacking_LOCO_weights_{tag}.csv", index=False)
