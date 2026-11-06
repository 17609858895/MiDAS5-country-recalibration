"""Does the model lean on geographic-fingerprint taxa?  SHAP attribution vs country structure.

Per taxon (digester-level CLR, countries with >= 3 digesters):
  eta2_country  = between-country share of variance (how strongly the taxon encodes geography);
  within_rho    = Fisher-pooled within-country Spearman correlation with the target;
  sign_consist  = fraction of countries whose within-country rho has the pooled sign.
Per target: ExtraTrees fitted on all samples, TreeSHAP on one replicate per digester.
Geographic reliance index GRI = sum_j |SHAP|_j * eta2_j / sum_j |SHAP|_j, compared with the
unweighted mean eta2 (attribution-free reference) and with LOCO error decomposition.
SHAP is associational: it describes what the fitted model uses, not causal effects.
"""
import warnings
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from common import RES
from invariance import within_country_rho
import importlib
bench = importlib.import_module("03_benchmark20")

warnings.filterwarnings("ignore")
TARGETS = ["Temperature", "pH", "log_NH4", "log_SRT", "Digester_age"]


def eta2(Xd, c):
    gm = Xd.groupby(c).transform("mean")
    return ((gm - Xd.mean()) ** 2).sum() / ((Xd - Xd.mean()) ** 2).sum()


def shap_chunk(model, Xc):
    return np.abs(shap.TreeExplainer(model).shap_values(Xc, check_additivity=False))


if __name__ == "__main__":
    X, T, meta = bench.build_dataset()
    Xd = X.groupby(meta.Digester.values).mean()
    cd = meta.groupby("Digester").Country.first().loc[Xd.index]
    big = cd.groupby(cd).transform("size") >= 3
    e2 = eta2(Xd[big], cd[big].values)
    rows, per_taxon = [], []
    for target in TARGETS:
        ok = T[target].notna().values
        m = ExtraTreesRegressor(300, max_features=0.3, min_samples_leaf=2, n_jobs=30, random_state=0)
        m.fit(X.values[ok], T[target].values[ok])
        m.set_params(n_jobs=1)
        sub = meta[ok].reset_index().drop_duplicates("Digester").index.values
        Xs = X.values[ok][sub]
        parts = Parallel(n_jobs=30)(delayed(shap_chunk)(m, Xs[i::30]) for i in range(30))
        ms = np.concatenate(parts).mean(0)
        yd = T[target][ok].groupby(meta.Digester.values[ok]).mean()
        keep = yd.index[big.loc[yd.index].values]
        R, w, _ = within_country_rho(Xd.loc[keep].values, yd.loc[keep].values, cd.loc[keep].values)
        z = np.arctanh(np.clip(R, -0.99, 0.99))
        pooled = np.tanh((w[:, None] * z).sum(0) / w.sum())
        sign = (np.sign(R) == np.sign(pooled)).mean(0)
        df = pd.DataFrame(dict(target=target, taxon=X.columns, mean_abs_shap=ms, eta2_country=e2.values,
                               within_rho=pooled, sign_consist=sign))
        df["shap_rank"] = df.mean_abs_shap.rank(ascending=False).astype(int)
        per_taxon.append(df)
        wts = df.mean_abs_shap / df.mean_abs_shap.sum()
        top = df.nsmallest(20, "shap_rank")
        rows.append(dict(target=target, GRI=np.sum(wts * df.eta2_country), mean_eta2_all=df.eta2_country.mean(),
                         top20_eta2=top.eta2_country.mean(), top20_abs_within_rho=top.within_rho.abs().mean(),
                         top20_sign_consist=top.sign_consist.mean(),
                         spearman_shap_eta2=df.mean_abs_shap.corr(df.eta2_country, method="spearman")))
        print(target, pd.Series(rows[-1]).drop("target").round(3).to_dict(), flush=True)
        print(top[["taxon", "mean_abs_shap", "eta2_country", "within_rho", "sign_consist"]].round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).to_csv(RES / "shap_geographic_reliance.csv", index=False)
    pd.concat(per_taxon).to_csv(RES / "shap_per_taxon.csv", index=False)
