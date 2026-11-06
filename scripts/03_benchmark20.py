"""20-algorithm benchmark: microbiome (V1-3 bacteria + V3-5 archaea genus CLR) -> process variables.
Two validation schemes: digester-grouped 5-fold (within-distribution) and leave-one-country-out (LOCO).
Training uses both replicate samples; metrics are computed at digester level (mean of A/B predictions)."""
import sys, time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from common import load_meta, amp_features, RES
from models import model_zoo, metrics, digester_level

warnings.filterwarnings("ignore")


def build_dataset():
    meta = load_meta()
    X13, _, _ = amp_features("V13")
    X35, _, _ = amp_features("V35")
    X = X13.join(X35, how="inner")
    meta = meta.loc[X.index]
    T = pd.DataFrame({
        "Temperature": meta.Temperature,
        "pH": meta.pH,
        "log_NH4": np.log10(meta.Ammonium.where(meta.Ammonium > 0)),
        "log_SRT": np.log10(meta.SRT.where(meta.SRT > 0)),
        "Digester_age": meta.Digester_age,
    })
    return X, T, meta


def fit_predict(model, X, y, tr, te):
    m = clone(model)
    m.fit(X[tr], y[tr])
    return te, np.asarray(m.predict(X[te])).ravel()


if __name__ == "__main__":
    X, T, meta = build_dataset()
    zoo = model_zoo(n_jobs=1)
    rows, oof_store = [], {}
    for target in T.columns:
        ok = T[target].notna().values
        Xi, yi = X.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        for scheme in ["GKF5", "LOCO"]:
            splits = list((GroupKFold(5).split(Xi, yi, dig)) if scheme == "GKF5"
                          else LeaveOneGroupOut().split(Xi, yi, cty))
            t0 = time.time()
            tasks = [(name, k) for name in zoo for k in range(len(splits))]
            out = Parallel(n_jobs=30)(delayed(fit_predict)(zoo[name], Xi, yi, *splits[k]) for name, k in tasks)
            preds = {name: np.full(len(yi), np.nan) for name in zoo}
            for (name, k), (te, p) in zip(tasks, out):
                preds[name][te] = p
            for name, p in preds.items():
                yd, pd_, _ = digester_level(p, yi, dig)
                r = dict(target=target, scheme=scheme, model=name, n_samples=len(yi), n_digesters=len(yd),
                         n_countries=len(set(cty)), **metrics(yd, pd_))
                # per-country R2 / MAE at digester level (LOCO only meaningful)
                rows.append(r)
                oof_store[(target, scheme, name)] = p
            print(target, scheme, f"{time.time()-t0:.0f}s", flush=True)
            sub = pd.DataFrame([r for r in rows if r["target"] == target and r["scheme"] == scheme])
            print(sub.sort_values("R2", ascending=False)[["model", "R2", "RMSE"]].head(6).to_string(index=False))
    res = pd.DataFrame(rows)
    res.to_csv(RES / "benchmark20_results.csv", index=False)
    pd.to_pickle(oof_store, RES / "benchmark20_oof.pkl")
    piv = res.pivot_table(index="model", columns=["target", "scheme"], values="R2")
    piv.to_csv(RES / "benchmark20_R2_pivot.csv")
    print(piv.round(3).to_string())
