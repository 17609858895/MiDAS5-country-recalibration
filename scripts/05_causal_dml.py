"""Double/debiased ML (partially linear model, cross-fitted) for the effect of ammonia on
methanogenic-pathway composition, at digester level (A/B replicates pooled).

Treatment  T: log10 total ammonium nitrogen (TAN); sensitivity: log10 free ammonia nitrogen (FAN).
Outcomes   Y: log-ratio hydrogenotrophic methanogens / Methanothrix (V3-5 archaea), CLR of single
              methanogen genera, log10 SAOB relative abundance (V1-3 bacteria).
Confounders W: temperature, pH, TS, VS, SRT, digester age, volume, substrate, technology, location,
              continent (missing values left as NaN for the tree nuisance models + indicators).
Refutations: placebo (T permuted within country), random common cause, 80% subsets,
             leave-one-country-out stability, alternative nuisance learners.
Heterogeneity: linear CATE in temperature and pH (LinearDML-style final stage).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor
from common import load_meta, load_counts, rel_abund, MIN_DEPTH, RES

warnings.filterwarnings("ignore")
HYDRO = ["Methanoculleus", "Methanospirillum", "Methanolinea", "Methanobacterium", "Methanothermobacter",
         "Methanobrevibacter", "Methanoregula", "uc_Methanoregulaceae", "uc_Methanomicrobiales"]
SAOB_KEYS = ["Syntrophaceticus", "Tepidanaerobacter"]


def digester_table():
    meta = load_meta()
    c35 = load_counts("V35", "g")
    c35 = c35[c35.sum(1) >= MIN_DEPTH]
    c13 = load_counts("V13", "g")
    c13 = c13[c13.sum(1) >= MIN_DEPTH]
    r35 = rel_abund(c35).groupby(meta.loc[c35.index, "Digester"]).mean()
    r13 = rel_abund(c13).groupby(meta.loc[c13.index, "Digester"]).mean()
    num = ["Temperature", "pH", "Ammonium", "TS", "VS", "SRT", "Digester_age", "Volumen", "Latitude"]
    cat = ["Country", "Continent", "Primary_substrate", "Digester_technology", "Digester_location",
           "Temperature_range"]
    d = meta.groupby("Digester")[num].mean().join(meta.groupby("Digester")[cat].first())
    eps = 1e-5
    hyd = r35.reindex(columns=HYDRO, fill_value=0).sum(1)
    out = pd.DataFrame(index=r35.index)
    out["LR_hydro_vs_Mthrix"] = np.log10((hyd + eps) / (r35["Methanothrix"] + eps))
    g35 = np.log(r35 + eps)
    clr35 = g35.sub(g35.mean(1), axis=0)
    for g in ["Methanothrix", "Methanosarcina", "Methanoculleus", "Methanobacterium", "Methanospirillum"]:
        out[f"CLR_{g}"] = clr35[g]
    saob = r13[[c for c in r13.columns if any(k in c for k in SAOB_KEYS)]].sum(1)
    out["log_SAOB"] = np.log10(saob + eps).reindex(out.index)
    out["log_Cloacimonas"] = np.log10(r13["Ca_Cloacimonas"] + eps).reindex(out.index)
    d = d.join(out, how="inner")
    d["log_TAN"] = np.log10(d.Ammonium.where(d.Ammonium > 0))
    T_K = d.Temperature + 273.15
    fan = d.Ammonium * 10 ** d.pH / (np.exp(6344 / T_K) + 10 ** d.pH)
    d["log_FAN"] = np.log10(fan.where(fan > 0))
    return d


def design_W(d, extra_random=False, seed=0):
    num = ["Temperature", "pH", "TS", "VS", "SRT", "Digester_age", "Volumen"]
    W = d[num].copy()
    W["Volumen"] = np.log10(W.Volumen)
    for c in num:
        W[f"miss_{c}"] = W[c].isna().astype(float)
    W = W.loc[:, W.std() > 0]
    dummies = pd.get_dummies(d[["Primary_substrate", "Digester_technology", "Digester_location", "Continent"]]
                             .astype(str), drop_first=True, dtype=float)
    W = W.join(dummies)
    if extra_random:
        W["random_cc"] = np.random.default_rng(seed).normal(size=len(W))
    return W


def nuisance(kind, seed):
    if kind == "LGBM":
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=8,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1,
                             random_state=seed, n_jobs=4)
    if kind == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestRegressor(400, min_samples_leaf=3, max_features=0.5,
                                                   random_state=seed, n_jobs=4))
    if kind == "Lasso":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LassoCV(cv=5, random_state=seed, max_iter=5000))


def dml_plr(y, t, W, kind="LGBM", n_folds=5, n_rep=10, moderators=None, seed=0):
    """Cross-fitted PLR; returns theta, se (median aggregation over repetitions)."""
    y, t = np.asarray(y, float), np.asarray(t, float)
    thetas, ses, cates = [], [], []
    for r in range(n_rep):
        yres, tres = np.zeros_like(y), np.zeros_like(t)
        for tr, te in KFold(n_folds, shuffle=True, random_state=seed + r).split(W):
            my = nuisance(kind, seed + r).fit(W.iloc[tr], y[tr])
            mt = nuisance(kind, seed + r).fit(W.iloc[tr], t[tr])
            yres[te] = y[te] - my.predict(W.iloc[te])
            tres[te] = t[te] - mt.predict(W.iloc[te])
        if moderators is None:
            theta = (tres @ yres) / (tres @ tres)
            psi = (yres - theta * tres) * tres
            se = np.sqrt(np.mean(psi ** 2) / np.mean(tres ** 2) ** 2 / len(y))
            thetas.append(theta)
            ses.append(se)
        else:
            M = np.column_stack([np.ones(len(y)), np.asarray(moderators, float)])
            Z = M * tres[:, None]
            beta, *_ = np.linalg.lstsq(Z, yres, rcond=None)
            res = yres - Z @ beta
            J = Z.T @ Z / len(y)
            S = (Z * res[:, None]).T @ (Z * res[:, None]) / len(y)
            V = np.linalg.inv(J) @ S @ np.linalg.inv(J) / len(y)
            cates.append((beta, np.sqrt(np.diag(V))))
    if moderators is not None:
        b = np.median([c[0] for c in cates], axis=0)
        s = np.median([c[1] for c in cates], axis=0)
        return b, s
    th = np.median(thetas)
    se = np.median(np.sqrt(np.array(ses) ** 2 + (np.array(thetas) - th) ** 2))
    return th, se


if __name__ == "__main__":
    d = digester_table()
    d.to_csv(RES / "digester_level_table.csv")
    outcomes = ["LR_hydro_vs_Mthrix", "CLR_Methanothrix", "CLR_Methanosarcina", "CLR_Methanoculleus",
                "CLR_Methanobacterium", "CLR_Methanospirillum", "log_SAOB", "log_Cloacimonas"]
    rows, refute = [], []
    for treat in ["log_TAN", "log_FAN"]:
        for yname in outcomes:
            sub = d[d[treat].notna() & d[yname].notna()]
            W = design_W(sub)
            y, t = sub[yname].values, sub[treat].values
            naive = np.polyfit(t, y, 1)[0]
            for kind in ["LGBM", "RF", "Lasso"]:
                th, se = dml_plr(y, t, W, kind)
                rows.append(dict(treatment=treat, outcome=yname, learner=kind, n=len(sub),
                                 n_countries=sub.Country.nunique(), naive_slope=naive, theta=th, se=se,
                                 ci_lo=th - 1.96 * se, ci_hi=th + 1.96 * se, z=th / se))
                print(rows[-1], flush=True)
            if treat == "log_TAN":
                rng = np.random.default_rng(1)
                # placebo: permute treatment within country (keeps country composition)
                for rep in range(20):
                    # numpy permutation inside transform: a returned Series would be re-aligned by index (no shuffle)
                    tp = sub.groupby("Country")[treat].transform(lambda s: rng.permutation(s.values)).values
                    th, se = dml_plr(y, tp, W, "LGBM", n_rep=2, seed=rep)
                    refute.append(dict(outcome=yname, test="placebo", rep=rep, theta=th))
                for rep in range(10):
                    th, se = dml_plr(y, t, design_W(sub, True, rep), "LGBM", n_rep=2, seed=rep)
                    refute.append(dict(outcome=yname, test="random_common_cause", rep=rep, theta=th))
                for rep in range(20):
                    idx = rng.choice(len(sub), int(0.8 * len(sub)), replace=False)
                    th, se = dml_plr(y[idx], t[idx], W.iloc[idx], "LGBM", n_rep=2, seed=rep)
                    refute.append(dict(outcome=yname, test="subset80", rep=rep, theta=th))
                for c in sub.Country.unique():
                    keep = (sub.Country != c).values
                    th, se = dml_plr(y[keep], t[keep], W[keep], "LGBM", n_rep=2)
                    refute.append(dict(outcome=yname, test="leave_country_out", rep=c, theta=th))
                # heterogeneity by temperature and pH (centred)
                mods = np.column_stack([sub.Temperature.fillna(sub.Temperature.median()) - 37,
                                        sub.pH.fillna(sub.pH.median()) - 7.4])
                b, s = dml_plr(y, t, W, "LGBM", moderators=mods)
                refute.append(dict(outcome=yname, test="CATE_linear", rep="intercept|Temp|pH",
                                   theta=str(np.round(b, 4)), se=str(np.round(s, 4))))
                print(yname, "CATE", np.round(b, 3), np.round(s, 3), flush=True)
    pd.DataFrame(rows).to_csv(RES / "dml_ate.csv", index=False)
    pd.DataFrame(refute).to_csv(RES / "dml_refutation.csv", index=False)
