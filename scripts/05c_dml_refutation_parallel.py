"""Parallel refutation of pooled vs within-country (demeaned) DML for log TAN.

For each outcome, with LGBM nuisances (2 cross-fitting repetitions per run):
  real_pooled / real_demeaned : estimates on the observed data;
  placebo_within_pooled       : treatment permuted within country, pooled DML (country means kept,
                                so a non-zero estimate means identification from between-country variation);
  placebo_within_demeaned     : same permutation, demeaned DML (expected ~0 if the within-country
                                estimate is not an artefact);
  placebo_full_demeaned       : treatment fully permuted;
  subset80, leave_country_out, random_common_cause : stability of the demeaned estimate.
Empirical p = (1 + #|placebo| >= |real|) / (1 + n_placebo).
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
N_PLACEBO = 200


def demean(v, g):
    return v - pd.Series(v).groupby(g).transform("mean").values


def job(test, rep, y, t, g, W, sub):
    base = dml.nuisance

    def single_thread(kind, seed):
        return base(kind, seed).set_params(n_jobs=1)
    dml.nuisance = single_thread
    rng = np.random.default_rng(1000 + rep if isinstance(rep, int) else 0)
    yd, td = demean(y, g), demean(t, g)
    perm_in = pd.Series(np.arange(len(t))).groupby(g).transform(lambda s: rng.permutation(s.values)).values
    if test == "real_pooled":
        th, se = dml.dml_plr(y, t, W, "LGBM", n_rep=10)
        return test, rep, th, se
    if test == "real_demeaned":
        th, se = dml.dml_plr(yd, td, W, "LGBM", n_rep=10)
        return test, rep, th, se
    if test == "placebo_within_pooled":
        return test, rep, dml.dml_plr(y, t[perm_in], W, "LGBM", n_rep=2, seed=rep)[0], np.nan
    if test == "placebo_within_demeaned":
        return test, rep, dml.dml_plr(yd, td[perm_in], W, "LGBM", n_rep=2, seed=rep)[0], np.nan
    if test == "placebo_full_demeaned":
        return test, rep, dml.dml_plr(yd, rng.permutation(td), W, "LGBM", n_rep=2, seed=rep)[0], np.nan
    if test == "subset80":
        idx = rng.choice(len(y), int(0.8 * len(y)), replace=False)
        return test, rep, dml.dml_plr(demean(y[idx], g[idx]), demean(t[idx], g[idx]), W.iloc[idx], "LGBM",
                                      n_rep=2, seed=rep)[0], np.nan
    if test == "random_common_cause":
        return test, rep, dml.dml_plr(yd, td, dml.design_W(sub, True, rep), "LGBM", n_rep=2, seed=rep)[0], np.nan
    if test == "leave_country_out":
        keep = g != rep
        return test, rep, dml.dml_plr(demean(y[keep], g[keep]), demean(t[keep], g[keep]), W[keep], "LGBM",
                                      n_rep=2)[0], np.nan


if __name__ == "__main__":
    d = dml.digester_table()
    rows = []
    for yname in OUTCOMES:
        sub = d[d.log_TAN.notna() & d[yname].notna()]
        sub = sub[sub.groupby("Country").Country.transform("size") >= 3]
        W = dml.design_W(sub)
        y, t, g = sub[yname].values, sub.log_TAN.values, sub.Country.values
        tasks = [("real_pooled", 0), ("real_demeaned", 0)]
        tasks += [(k, r) for k in ["placebo_within_pooled", "placebo_within_demeaned", "placebo_full_demeaned"]
                  for r in range(N_PLACEBO)]
        tasks += [("subset80", r) for r in range(30)] + [("random_common_cause", r) for r in range(10)]
        tasks += [("leave_country_out", c) for c in np.unique(g)]
        out = Parallel(n_jobs=30)(delayed(job)(k, r, y, t, g, W, sub) for k, r in tasks)
        df = pd.DataFrame(out, columns=["test", "rep", "theta", "se"]).assign(outcome=yname, n=len(sub),
                                                                             n_countries=len(np.unique(g)))
        rows.append(df)
        real = df.set_index("test").theta
        s = {"real_pooled": real["real_pooled"], "real_demeaned": real["real_demeaned"]}
        for k, ref in [("placebo_within_pooled", "real_pooled"), ("placebo_within_demeaned", "real_demeaned"),
                       ("placebo_full_demeaned", "real_demeaned")]:
            v = df[df.test == k].theta.values
            s[k + "_mean"] = v.mean()
            s[k + "_p"] = (1 + np.sum(np.abs(v) >= abs(real[ref]))) / (1 + len(v))
        for k in ["subset80", "random_common_cause", "leave_country_out"]:
            v = df[df.test == k].theta.values
            s[k + "_range"] = f"[{v.min():.2f}, {v.max():.2f}]"
            s[k + "_same_sign"] = np.mean(np.sign(v) == np.sign(real["real_demeaned"]))
        print(yname, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, flush=True)
    pd.concat(rows).to_csv(RES / "dml_refutation_parallel.csv", index=False)
