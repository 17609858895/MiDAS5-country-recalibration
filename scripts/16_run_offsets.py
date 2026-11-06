"""Sequencing-run component of the country offsets.

1. Nested variance components (grouping / country within grouping / digester) of digester-level LOCO ExtraTrees
   residuals, for the V1-3 run, the V3-5 run, their combination and, for comparison, the continent.
   Permutation test: run labels permuted among countries with group sizes fixed; statistic = between-group F of
   digester-weighted country offsets.
2. Run-mate prior: zero-shot offset from inner-LOCO residuals of training countries sequenced in the same run,
   EB-shrunk as the continent prior of 14_recovery_hierarchy_simpson.py.
3. Run hold-out (argument "holdout"): ExtraTrees with all countries linked by a shared V1-3 or V3-5 run held out
   together, against random country groups of the same sizes.
"""
import importlib
import sys
import warnings
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.base import clone
from common import RES, load_meta
from models import digester_level, model_zoo

bench = importlib.import_module("03_benchmark20")
fs = importlib.import_module("08_fewshot_recalibration")
DATA = RES.parent / "data_extracted"
TARGETS = ["Temperature", "pH", "log_NH4", "log_SRT", "Digester_age"]
GROUPINGS = ["run_V13", "run_V35", "run_pair", "continent"]
N_PERM = 5000


def country_groups():
    meta = load_meta().reset_index()
    g = meta.groupby("Country").Continent.first().rename("continent").to_frame()
    for amp, f in [("V13", "V13_metadata.txt"), ("V35", "V35_metadata.txt")]:
        mp = pd.read_csv(DATA / f, sep="\t", header=None, names=["seq", "SampleID"])
        mp["run"] = mp.seq.str.split("-").str[0]
        mp = mp.merge(meta[["SampleID", "Country"]], on="SampleID")
        runs = mp.groupby("Country").run.agg(lambda s: sorted(set(s)))
        assert runs.map(len).eq(1).all()
        g[f"run_{amp}"] = runs.str[0]
    g["run_pair"] = g.run_V13 + "/" + g.run_V35
    # connected components of countries linked by a shared V1-3 or V3-5 run
    parent = {c: c for c in g.index}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c
    for col in ["run_V13", "run_V35"]:
        for _, cs in g.groupby(col).groups.items():
            cs = list(cs)
            for c in cs[1:]:
                parent[find(c)] = find(cs[0])
    g["run_component"] = [find(c) for c in g.index]
    return g


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def f_stat(o, n, grp):
    """Between-group F of country offsets o weighted by digester counts n."""
    m = np.average(o, weights=n)
    labs = np.unique(grp)
    mr = {g: np.average(o[grp == g], weights=n[grp == g]) for g in labs}
    ssb = sum(n[grp == g].sum() * (mr[g] - m) ** 2 for g in labs)
    ssw = sum(np.sum(n[grp == g] * (o[grp == g] - mr[g]) ** 2) for g in labs)
    dfb, dfw = len(labs) - 1, len(o) - len(labs)
    return (ssb / dfb) / (ssw / dfw), ssb / (ssb + ssw)


def nested_reml(df, grp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = smf.mixedlm("r ~ 1", df, groups=grp, re_formula="1", vc_formula={"country": "0 + C(country)"})
        fit = md.fit(reml=True, method=["lbfgs", "powell"])
    return float(fit.cov_re.iloc[0, 0]), float(fit.vcomp[0]), float(fit.scale)


def variance_shares(cg, rng):
    X, T, meta = bench.build_dataset()
    oof = pd.read_pickle(RES / "benchmark20_oof.pkl")
    rows = []
    for t in TARGETS:
        ok = T[t].notna().values
        y, dig, cty = T[t].values[ok], meta.Digester.values[ok], meta.Country.values[ok]
        dc = pd.Series(cty, index=dig).groupby(level=0).first()
        yd, pdg, d = digester_level(oof[(t, "LOCO", "ExtraTrees")], y, dig)
        df = pd.DataFrame({"r": yd - pdg, "country": dc.loc[d].values})
        df = df.join(cg, on="country")
        co = df.groupby("country").agg(o=("r", "mean"), n=("r", "size")).join(cg)
        for gname in GROUPINGS:
            s_g, s_c, s_e = nested_reml(df, gname)
            grp = co[gname].values
            F, eta2 = f_stat(co.o.values, co.n.values, grp)
            null = np.array([f_stat(co.o.values, co.n.values, rng.permutation(grp)) for _ in range(N_PERM)])
            rows.append(dict(target=t, grouping=gname, countries=len(co), groups=co[gname].nunique(),
                             singleton_groups=int((co[gname].value_counts() == 1).sum()), digesters=len(df),
                             var_group=s_g, var_country=s_c, var_digester=s_e,
                             share_group=s_g / (s_g + s_c) if s_g + s_c > 0 else np.nan,
                             eta2=eta2, eta2_null_mean=null[:, 1].mean(), F=F,
                             p_perm=(1 + np.sum(null[:, 0] >= F)) / (1 + N_PERM)))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def runmate_prior(cg):
    rows = []
    for t in ["Temperature", "pH", "log_NH4", "Digester_age"]:
        d = pd.read_pickle(RES / f"conformal_store_{t}.pkl")
        store, yi, dig, cty = d["store"], d["y"], d["dig"], d["cty"]
        dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
        y_d = fs.dig_avg(yi, dig)
        for gname in GROUPINGS:
            Y, P, C, better, n_prior = [], [], [], 0, 0
            for kf in sorted({k for (kind, k) in store if kind == "outer"}):
                po = store[("outer", kf)]
                te = ~np.isnan(po)
                p_d = fs.dig_avg(po[te], dig[te])
                pi = store[("blocked", kf)]
                msk = ~np.isnan(pi)
                r_in = fs.dig_avg(yi[msk] - pi[msk], dig[msk])
                tau2, sig2 = fs.moments(r_in.values, dig_c.loc[r_in.index].values)
                c = cty[te][0]
                same = [dg for dg in r_in.index if cg.loc[dig_c.loc[dg], gname] == cg.loc[c, gname]]
                if same:
                    rc = r_in.loc[same].groupby(dig_c.loc[same].values).mean()
                    prior = rc.mean() * (len(rc) * tau2 / (len(rc) * tau2 + sig2 / max(len(same) / len(rc), 1)))
                    n_prior += 1
                else:
                    prior = 0.0
                yy, pp = y_d.loc[p_d.index].values, p_d.values
                Y.append(yy); P.append(pp); C.append(pp + prior)
                better += np.abs(yy - pp - prior).mean() < np.abs(yy - pp).mean()
            Y, P, C = map(np.concatenate, (Y, P, C))
            rows.append(dict(target=t, grouping=gname, R2_LOCO=r2(Y, P), R2_prior=r2(Y, C),
                             countries_improved=int(better), countries_with_prior=n_prior,
                             countries=len(set(cty))))
    return pd.DataFrame(rows)


def run_holdout(cg, n_rand=30, seed=0):
    X, T, meta = bench.build_dataset()
    et = model_zoo(n_jobs=32)["ExtraTrees"]
    oof = pd.read_pickle(RES / "benchmark20_oof.pkl")
    rng = np.random.default_rng(seed)
    rows = []
    for t in TARGETS:
        ok = T[t].notna().values
        Xi, y, dig, cty = X.values[ok], T[t].values[ok], meta.Digester.values[ok], meta.Country.values[ok]
        countries = np.unique(cty)
        comp = cg.run_component.loc[countries]
        sizes = comp.value_counts().values

        def score(assign):
            g = np.array([assign[c] for c in cty])
            pred = np.full(len(y), np.nan)
            for k in np.unique(g):
                te = g == k
                pred[te] = clone(et).fit(Xi[~te], y[~te]).predict(Xi[te])
            yd, pdg, _ = digester_level(pred, y, dig)
            return r2(yd, pdg)
        r_run = score(comp.to_dict())
        rand = []
        for _ in range(n_rand):
            perm, assign, i = rng.permutation(countries), {}, 0
            for k, s in enumerate(sizes):
                assign.update({c: k for c in perm[i:i + s]})
                i += s
            rand.append(score(assign))
        rand = np.array(rand)
        yd, pdg, _ = digester_level(oof[(t, "LOCO", "ExtraTrees")], y, dig)
        rows.append(dict(target=t, countries=len(countries), group_sizes="/".join(map(str, sizes)),
                         R2_LOCO=r2(yd, pdg), R2_run_holdout=r_run, R2_random_mean=rand.mean(),
                         R2_random_min=rand.min(), R2_random_max=rand.max(),
                         share_random_below=(rand < r_run).mean(), n_random=n_rand))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cg = country_groups()
    print(cg.to_string())
    if sys.argv[1:] == ["holdout"]:
        ho = run_holdout(cg)
        ho.to_csv(RES / "run_holdout.csv", index=False)
        print(ho.round(3).to_string(index=False))
        sys.exit()
    vs = variance_shares(cg, np.random.default_rng(0))
    vs.to_csv(RES / "run_offset_variance.csv", index=False)
    rp = runmate_prior(cg)
    rp.to_csv(RES / "runmate_prior_zero_shot.csv", index=False)
    pd.set_option("display.width", 250)
    print(vs.round(3).to_string(index=False))
    print(rp.round(3).to_string(index=False))
