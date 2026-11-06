"""Taxonomic resolution vs transferability: does a coarser (functionally conserved) representation
transfer better across countries, and does a finer one encode more country fingerprint?
Features at phylum/class/order/family/genus/species/ASV for V1-3 bacteria + V3-5 archaea (CLR,
prevalence >=10%), ExtraTrees, GKF5 vs LOCO.  Also the country-identification accuracy of each
representation (how strongly it encodes geography)."""
import time, warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import r2_score, balanced_accuracy_score
from common import load_meta, PROC, MIN_DEPTH, RES
from models import digester_level

warnings.filterwarnings("ignore")
RANKS = ["p", "c", "o", "f", "g", "s", "ASV"]


def rank_counts(amp, rank):
    asv = pd.read_parquet(PROC / f"{amp}_asv_counts.parquet")
    asv = asv[asv.sum(1) >= MIN_DEPTH]
    if rank == "ASV":
        return asv
    tax = pd.read_csv(PROC / f"asv_taxonomy_{amp}.csv", index_col=0).reindex(asv.columns)
    return asv.T.groupby(tax[rank].values).sum().T


def clr_feats(counts, prefix, min_prev=0.1, max_feats=3000):
    rel = counts.div(counts.sum(1), axis=0)
    keep = rel.columns[(rel > 0).mean() >= min_prev]
    keep = rel[keep].mean().sort_values(ascending=False).index[:max_feats]
    comp = rel[keep].copy()
    comp["_o"] = (1 - comp.sum(1)).clip(lower=0)
    x = np.log(comp + 1e-5)
    x = x.sub(x.mean(1), axis=0).drop(columns="_o")
    x.columns = [f"{prefix}|{c}" for c in x.columns]
    return x


def fit_pred(X, y, tr, te, clf=False):
    if clf:
        m = ExtraTreesClassifier(500, max_features="sqrt", n_jobs=1, random_state=0, class_weight="balanced")
    else:
        m = ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0)
    m.fit(X[tr], y[tr])
    return te, m.predict(X[te])


if __name__ == "__main__":
    meta = load_meta()
    rows = []
    for rank in RANKS:
        X = clr_feats(rank_counts("V13", rank), "V13").join(clr_feats(rank_counts("V35", rank), "V35"), how="inner")
        md = meta.loc[X.index]
        T = pd.DataFrame({"Temperature": md.Temperature, "pH": md.pH,
                          "log_NH4": np.log10(md.Ammonium.where(md.Ammonium > 0)),
                          "Digester_age": md.Digester_age})
        t0 = time.time()
        # country identification (geographic fingerprint), digester-grouped folds
        y_c = md.Country.values
        sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y_c, md.Digester.values))
        out = Parallel(n_jobs=5)(delayed(fit_pred)(X.values, y_c, tr, te, True) for tr, te in sp)
        pc = np.empty(len(y_c), dtype=object)
        for te, p in out:
            pc[te] = p
        rows.append(dict(rank=rank, n_features=X.shape[1], target="Country_ID", scheme="GKF5",
                         R2=balanced_accuracy_score(y_c, pc)))
        for target in T.columns:
            ok = T[target].notna().values
            Xi, yi = X.values[ok], T[target].values[ok]
            dig, cty = md.Digester.values[ok], md.Country.values[ok]
            for scheme in ["GKF5", "LOCO"]:
                sp = list(GroupKFold(5).split(Xi, yi, dig) if scheme == "GKF5" else LeaveOneGroupOut().split(Xi, yi, cty))
                out = Parallel(n_jobs=30)(delayed(fit_pred)(Xi, yi, tr, te) for tr, te in sp)
                p = np.full(len(yi), np.nan)
                for te, pp in out:
                    p[te] = pp
                yd, pdg, dd = digester_level(p, yi, dig)
                cd = pd.Series(cty, index=dig).groupby(level=0).first().loc[dd].values
                res = yd - pdg
                bias = pd.Series(res).groupby(cd).transform("mean").values
                yc = yd - pd.Series(yd).groupby(cd).transform("mean").values
                pcn = pdg - pd.Series(pdg).groupby(cd).transform("mean").values
                rows.append(dict(rank=rank, n_features=X.shape[1], target=target, scheme=scheme,
                                 R2=r2_score(yd, pdg), bias_share=np.sum(bias ** 2) / np.sum(res ** 2),
                                 within_corr=np.corrcoef(yc, pcn)[0, 1]))
        print(rank, X.shape, f"{time.time()-t0:.0f}s", flush=True)
        print(pd.DataFrame([r for r in rows if r["rank"] == rank]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RES / "taxonomic_resolution.csv", index=False)
