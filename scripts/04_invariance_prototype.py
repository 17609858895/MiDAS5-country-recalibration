"""Prototype: does country-invariant learning close the LOCO gap?"""
import time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import LeaveOneGroupOut
from common import RES
from models import metrics, digester_level
from invariance import AnchorLGBM, invariant_screen
import importlib
bench = importlib.import_module("03_benchmark20")

warnings.filterwarnings("ignore")


def dig_mean(X, y, dig):
    df = pd.DataFrame(X)
    df["_y"] = y
    df["_d"] = dig
    g = df.groupby("_d").mean()
    return g.drop(columns="_y").values, g._y.values, g.index.values


def run_fold(method, X, y, dig, cty, tr, te):
    Xtr, ytr, ctr = X[tr], y[tr], cty[tr]
    Xte = X[te]
    cols = np.arange(X.shape[1])
    if "screen" in method:
        k = int(method.split("screen")[1].split("_")[0])
        Xd, yd, dd = dig_mean(Xtr, ytr, dig[tr])
        cd = pd.Series(ctr, index=dig[tr]).groupby(level=0).first().loc[dd].values
        cols, _ = invariant_screen(Xd, yd, cd, k=k)
    Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    if "center" in method:  # transductive per-country feature centering (label-free)
        Xtr = Xtr - pd.DataFrame(Xtr).groupby(ctr).transform("mean").values
        Xte = Xte - Xte.mean(0)
    if method.startswith("ET"):
        m = ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0)
        w = None
        if "bal" in method:
            w = 1.0 / pd.Series(ctr).map(pd.Series(ctr).value_counts()).values
        m.fit(Xtr, ytr, sample_weight=w)
    else:
        gamma = float(method.split("_g")[1].split("_")[0]) if method.startswith("Anchor") else 1.0
        m = AnchorLGBM(gamma=gamma, n_jobs=1)
        m.fit(Xtr, ytr, groups=ctr)
    return te, m.predict(Xte)


if __name__ == "__main__":
    X, T, meta = bench.build_dataset()
    methods = ["ET", "ET_bal", "LGBM", "Anchor_g0", "Anchor_g0.5", "Anchor_g3", "Anchor_g10",
               "ET_screen50", "ET_screen100", "ET_screen200", "Anchor_g3_screen100", "ET_center", "Anchor_g0_center"]
    rows = []
    for target in ["Temperature", "pH", "log_NH4", "log_SRT", "Digester_age"]:
        ok = T[target].notna().values
        Xi, yi = X.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        splits = list(LeaveOneGroupOut().split(Xi, yi, cty))
        t0 = time.time()
        tasks = [(mth, k) for mth in methods for k in range(len(splits))]
        out = Parallel(n_jobs=30)(delayed(run_fold)(mth, Xi, yi, dig, cty, *splits[k]) for mth, k in tasks)
        preds = {mth: np.full(len(yi), np.nan) for mth in methods}
        for (mth, k), (te, p) in zip(tasks, out):
            preds[mth][te] = p
        for mth, p in preds.items():
            yd, pdg, dd = digester_level(p, yi, dig)
            cd = pd.Series(cty, index=dig).groupby(level=0).first().loc[dd].values
            per = pd.DataFrame({"y": yd, "p": pdg, "c": cd}).groupby("c").apply(
                lambda g: np.mean(np.abs(g.y - g.p)))
            rows.append(dict(target=target, method=mth, **metrics(yd, pdg), country_MAE_median=per.median()))
        print(target, f"{time.time()-t0:.0f}s")
        print(pd.DataFrame([r for r in rows if r["target"] == target]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RES / "invariance_prototype_LOCO.csv", index=False)
