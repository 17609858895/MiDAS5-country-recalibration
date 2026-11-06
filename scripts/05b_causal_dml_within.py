"""Within-country (fixed-effects) DML: point estimates for log TAN and log FAN.

Treatment and outcome are identified three ways:
  pooled   : confounder set W only (between- and within-country variation);
  FE       : country indicators added to W (regularised nuisance learners absorb them only partly);
  demeaned : y and t centred within country before cross-fitting on W (exact within transformation).
Countries with <3 digesters are dropped.  Learners: LGBM, RF, Lasso; 5 folds x 10 repetitions.
Refutations (placebos, subsets, leave-one-country-out, random common cause) are in
05c_dml_refutation_parallel.py.
"""
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from common import RES
import importlib
dml = importlib.import_module("05_causal_dml")

warnings.filterwarnings("ignore")
OUTCOMES = ["LR_hydro_vs_Mthrix", "CLR_Methanothrix", "CLR_Methanosarcina", "CLR_Methanoculleus",
            "CLR_Methanobacterium", "CLR_Methanospirillum", "log_SAOB", "log_Cloacimonas"]


def demean(v, g):
    return v - pd.Series(v).groupby(g).transform("mean").values


def job(treat, yname, variant, kind, d):
    base = dml.nuisance

    def single_thread(k, seed):
        m = base(k, seed)
        if k == "LGBM":
            m.set_params(n_jobs=1)
        elif k == "RF":
            m.set_params(randomforestregressor__n_jobs=1)
        return m
    dml.nuisance = single_thread
    sub = d[d[treat].notna() & d[yname].notna()]
    sub = sub[sub.groupby("Country").Country.transform("size") >= 3]
    W = dml.design_W(sub)
    y, t, g = sub[yname].values, sub[treat].values, sub.Country.values
    if variant == "FE":
        W = W.join(pd.get_dummies(sub.Country, prefix="cty", drop_first=True, dtype=float))
    if variant == "demeaned":
        y, t = demean(y, g), demean(t, g)
    th, se = dml.dml_plr(y, t, W, kind)
    return dict(treatment=treat, outcome=yname, variant=variant, learner=kind, n=len(sub),
                n_countries=len(np.unique(g)),
                within_var_share_T=np.var(demean(sub[treat].values, g)) / np.var(sub[treat].values),
                theta=th, se=se, ci_lo=th - 1.96 * se, ci_hi=th + 1.96 * se, z=th / se)


if __name__ == "__main__":
    d = dml.digester_table()
    tasks = [(tr, y, v, k) for tr in ["log_TAN", "log_FAN"] for y in OUTCOMES
             for v in ["pooled", "FE", "demeaned"] for k in ["LGBM", "RF", "Lasso"]]
    rows = Parallel(n_jobs=30)(delayed(job)(*a, d) for a in tasks)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "dml_within_ate.csv", index=False)
    print(res.pivot_table(index=["treatment", "outcome"], columns=["variant", "learner"], values="z").round(2).to_string())
    print(res.pivot_table(index=["treatment", "outcome"], columns=["variant", "learner"], values="theta").round(2).to_string())
