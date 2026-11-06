"""Cross-country diagnostic classifiers: thermophilic regime, ammonia stress, non-sludge substrate.

ExtraTrees classifier on V1-3 + V3-5 genus CLR; digester-grouped 5-fold vs leave-one-country-out;
probabilities averaged over the two replicates of a digester.  Reports AUC, average precision,
Brier score, and per-country AUC where a held-out country contains both classes.
"""
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from common import RES
import importlib
bench = importlib.import_module("03_benchmark20")

warnings.filterwarnings("ignore")


def fit_proba(X, y, tr, te):
    m = ExtraTreesClassifier(500, max_features="sqrt", class_weight="balanced", n_jobs=1, random_state=0)
    m.fit(X[tr], y[tr])
    return te, m.predict_proba(X[te])[:, 1]


if __name__ == "__main__":
    X, _, meta = bench.build_dataset()
    sub = meta.Primary_substrate.where(meta.Primary_substrate != "Other")
    B = pd.DataFrame({
        "Thermophilic": (meta.Temperature_range == "Thermophilic").astype(float).where(meta.Temperature_range.notna()),
        "TAN_gt2000": (meta.Ammonium > 2000).astype(float).where(meta.Ammonium.notna()),
        "TAN_gt3000": (meta.Ammonium > 3000).astype(float).where(meta.Ammonium.notna()),
        "NonSludge": (sub != "Wastewater sludge").astype(float).where(sub.notna()),
    })
    rows, per_c, store = [], [], {}
    for target in B.columns:
        ok = B[target].notna().values
        Xi, yi = X.values[ok], B[target].values[ok].astype(int)
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        for scheme in ["GKF5", "LOCO"]:
            sp = GroupKFold(5).split(Xi, yi, dig) if scheme == "GKF5" else LeaveOneGroupOut().split(Xi, yi, cty)
            out = Parallel(n_jobs=30)(delayed(fit_proba)(Xi, yi, tr, te) for tr, te in sp)
            p = np.full(len(yi), np.nan)
            for te, pp in out:
                p[te] = pp
            d = pd.DataFrame({"p": p, "y": yi, "dig": dig, "cty": cty}).groupby("dig").agg(
                p=("p", "mean"), y=("y", "max"), cty=("cty", "first"))
            store[(target, scheme)] = d
            rows.append(dict(target=target, scheme=scheme, n_dig=len(d), n_pos=int(d.y.sum()),
                             n_countries_pos=d[d.y == 1].cty.nunique(), AUC=roc_auc_score(d.y, d.p),
                             AP=average_precision_score(d.y, d.p), prevalence=d.y.mean(),
                             Brier=brier_score_loss(d.y, d.p)))
            for c, g in d.groupby("cty"):
                if g.y.nunique() == 2 and len(g) >= 3:
                    per_c.append(dict(target=target, scheme=scheme, country=c, n=len(g), n_pos=int(g.y.sum()),
                                      AUC=roc_auc_score(g.y, g.p)))
        print(pd.DataFrame([r for r in rows if r["target"] == target]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RES / "classification_LOCO.csv", index=False)
    pd.DataFrame(per_c).to_csv(RES / "classification_per_country.csv", index=False)
    pd.to_pickle(store, RES / "classification_oof.pkl")
    print(pd.DataFrame(per_c).groupby(["target", "scheme"]).AUC.describe().round(3).to_string())
