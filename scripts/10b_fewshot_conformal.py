"""Few-shot recalibrated conformal prediction for a new country (country-exchangeable calibration).

Deployment model for a new country c:  y_ic = f(x_ic) + b_c + e_ic.
With k labelled digesters L in c, the point prediction is f(x) + b_hat_k, where
    b_hat_k = k*tau^2/(k*tau^2 + sigma^2) * mean_{i in L}(y_i - f(x_i))   (empirical Bayes, as in 08).
Calibration mimics the deployment step inside the training countries: for each inner
leave-one-country-out country j, draw k labelled digesters, apply the same shrunken offset to
the inner-LOCO residuals and take |r_i - b_hat_j| on the remaining digesters of j as scores.
Scores are country-balanced (each calibration country carries equal total weight), so the
exchangeable unit is the country, not the digester.  The interval is f(x) + b_hat_k +/- q_k.

Comparators: standard split conformal with digester-pooled scores from inner GKF5 (iid_split) or
inner LOCO (blocked_split); the same with country-balanced weights (iid, blocked); EB point
recalibration with k = 0 scores (mismatched calibration); and the EB-matched rule.  Rules with
suffix _J apply the finite-sample correction at the country level (J calibration countries).

Rules with suffix _JP use hierarchical jackknife+ weighting (Barber et al. 2021; Lee et al. 2026):
the J leave-one-country-out models each predict the held-out country, country j carries mass
1/(J+1) spread over its scores and +/-inf carries 1/(J+1). The published standard construction
has a 1 - 2*alpha marginal lower bound under hierarchical exchangeability and symmetry.
For EB_matched_JP the k-shot offset of model -j uses a shrinkage weight estimated without
countries j and the held-out country (leave-three-countries-out fits). This exclusion alone
does not establish symmetry of the complete extension: repeated calibration label sets,
small-country eligibility and global label-free filtering also matter. This implementation
reports empirical coverage, without claiming a finite-sample guarantee for the EB extension.
EB_matched_JP_w0 reuses the outer shrinkage weight for every j (sensitivity check).
"""
import sys, time, warnings
from itertools import combinations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from common import RES
import importlib
bench = importlib.import_module("03_benchmark20")
fs = importlib.import_module("08_fewshot_recalibration")

warnings.filterwarnings("ignore")
ALPHAS = [0.1, 0.2]
KS = [0, 1, 2, 3, 5]
N_DRAWS = 50
N_CAL_DRAWS = 20


def fit_pred(X, y, tr, te):
    m = ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=1, random_state=0)
    m.fit(X[tr], y[tr])
    return m.predict(X[te])


def wquantile(s, w, level, n_groups=None):
    """Weighted conformal quantile.  The finite-sample correction uses the effective number of
    scores (1/sum w^2) by default, or the number of calibration countries when n_groups is given;
    with country-balanced weights the latter is the hierarchical conformal threshold (Lee et al.
    2025), capped at the largest score where that threshold is infinite."""
    o = np.argsort(s)
    s, w = s[o], w[o] / w.sum()
    n_eff = n_groups if n_groups else 1.0 / np.sum(w ** 2)
    level = min(level * (1 + 1 / n_eff), 1.0)
    return s[min(np.searchsorted(np.cumsum(w), level), len(s) - 1)]


def calib_scores(r, c, k, shrink, rng):
    """Country-balanced scores after simulated k-shot recalibration inside each calibration country."""
    s, w, J = [], [], 0
    for cc in np.unique(c):
        rc = r[c == cc]
        if len(rc) < k + 1:
            continue
        J += 1
        for _ in range(N_CAL_DRAWS if k > 0 else 1):
            lab = rng.choice(len(rc), k, replace=False) if k > 0 else np.array([], int)
            b = shrink * rc[lab].mean() if k > 0 else 0.0
            rest = np.delete(rc, lab)
            s.append(np.abs(rest - b))
            w.append(np.full(len(rest), 1.0 / len(rest)))
    return np.concatenate(s), np.concatenate(w), J


def pair_moments(triple, yi, dig, cty, t, j):
    """tau^2, sigma^2 from LOCO residuals of the countries other than t and j, each predicted by
    the model fitted without t, j and itself."""
    r, c = [], []
    for m in np.unique(cty):
        if m in (t, j):
            continue
        idx, p = triple[tuple(sorted((t, j, m)))]
        sel = cty[idx] == m
        rr = fs.dig_avg(yi[idx][sel] - p[sel], dig[idx][sel])
        r.append(rr.values)
        c += [m] * len(rr)
    return fs.moments(np.concatenate(r), np.array(c))


def jp_calib(jk, k, shrink_of, rng):
    """Per-country k-shot scores for the hierarchical jackknife+; each country sums to 1/(J+1)."""
    parts = []
    for j, (mu, rj, mom) in jk.items():
        if len(rj) < k + 1:
            continue
        sh = shrink_of(mom) if k > 0 else 0.0
        s, w, _ = calib_scores(rj, np.zeros(len(rj)), k, sh, rng)
        parts.append((mu, s, w / w.sum(), sh))
    J = len(parts)
    return [(mu, s, w / (J + 1), sh) for mu, s, w, sh in parts]


def jp_bounds(parts, y_lab, lab, ev, alphas):
    """Hierarchical jackknife+ bounds for the evaluation digesters ev at each alpha."""
    cent = [mu[ev] + (sh * (y_lab - mu[lab]).mean() if len(lab) else 0.0) for mu, _, _, sh in parts]
    up = np.concatenate([c[:, None] + s[None, :] for c, (_, s, _, _) in zip(cent, parts)], axis=1)
    lo = np.concatenate([-(c[:, None] - s[None, :]) for c, (_, s, _, _) in zip(cent, parts)], axis=1)
    W = np.concatenate([w for _, _, w, _ in parts])
    ou, ol = np.argsort(up, axis=1), np.argsort(lo, axis=1)
    cu, cl = np.cumsum(W[ou], axis=1), np.cumsum(W[ol], axis=1)
    su, sl = np.take_along_axis(up, ou, axis=1), np.take_along_axis(lo, ol, axis=1)
    out = {}
    for a in alphas:
        b = []
        for cw, sv in [(cu, su), (cl, sl)]:
            hit = cw >= 1 - a - 1e-9
            v = sv[np.arange(len(sv)), hit.argmax(axis=1)]
            v[~hit.any(axis=1)] = np.inf
            b.append(v)
        out[a] = (-b[1], b[0])
    return out


if __name__ == "__main__":
    targets = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Temperature", "pH", "log_NH4", "Digester_age"]
    X, T, meta = bench.build_dataset()
    rows = []
    for target in targets:
        ok = T[target].notna().values
        Xi, yi = X.values[ok], T[target].values[ok]
        dig, cty = meta.Digester.values[ok], meta.Country.values[ok]
        outer = list(LeaveOneGroupOut().split(Xi, yi, cty))
        tasks = []
        for k, (tr, te) in enumerate(outer):
            tasks.append(("outer", k, tr, te))
            for itr, ite in GroupKFold(5).split(tr, groups=dig[tr]):
                tasks.append(("iid", k, tr[itr], tr[ite]))
            for itr, ite in LeaveOneGroupOut().split(tr, groups=cty[tr]):
                tasks.append(("blocked", k, tr[itr], tr[ite]))
        t0 = time.time()
        cache = RES / f"conformal_store_{target}.pkl"
        if cache.exists():
            store = pd.read_pickle(cache)["store"]
        else:
            out = Parallel(n_jobs=30, batch_size=2)(delayed(fit_pred)(Xi, yi, a, b) for (_, _, a, b) in tasks)
            store = {}
            for (kind, kf, a, b), p in zip(tasks, out):
                store.setdefault((kind, kf), np.full(len(yi), np.nan))[b] = p
            pd.to_pickle(dict(store=store, y=yi, dig=dig, cty=cty), cache)
        tcache = RES / f"conformal_triple_{target}.pkl"
        if tcache.exists():
            triple = pd.read_pickle(tcache)
        else:
            trip = list(combinations(sorted(np.unique(cty)), 3))
            sel = [np.isin(cty, list(t3)) for t3 in trip]
            out = Parallel(n_jobs=30, batch_size=1)(delayed(fit_pred)(Xi, yi, np.where(~s)[0], np.where(s)[0])
                                                    for s in sel)
            triple = {t3: (np.where(s)[0], p) for t3, s, p in zip(trip, sel, out)}
            pd.to_pickle(triple, tcache)
        print(target, f"{time.time()-t0:.0f}s", flush=True)
        dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
        y_d = fs.dig_avg(yi, dig)
        rng = np.random.default_rng(0)
        rng_jp = np.random.default_rng(1)
        fold_of = {cty[te][0]: kf for kf, (_, te) in enumerate(outer)}
        prep = []
        for kf, (tr, te) in enumerate(outer):
            p_d = fs.dig_avg(store[("outer", kf)][te], dig[te])
            res_te = (y_d.loc[p_d.index] - p_d).values
            cal = {}
            for kind in ["iid", "blocked"]:
                pp = store[(kind, kf)]
                msk = ~np.isnan(pp)
                r = fs.dig_avg(yi[msk] - pp[msk], dig[msk])
                cal[kind] = (r.values, dig_c.loc[r.index].values)
            tau2, sig2 = fs.moments(*cal["blocked"])
            r_b, c_b = cal["blocked"]
            jk = {j: (fs.dig_avg(store[("blocked", fold_of[j])][te], dig[te]).values, r_b[c_b == j],
                      pair_moments(triple, yi, dig, cty, cty[te][0], j)) for j in np.unique(c_b)}
            prep.append((res_te, cal, tau2, sig2, cty[te][0], y_d.loc[p_d.index].values, jk))
        for k_shot in KS:
            for (res_te, cal, tau2, sig2, c, y_te, jk) in prep:
                if len(res_te) < k_shot + 2:
                    continue
                shrink = k_shot * tau2 / (k_shot * tau2 + sig2) if k_shot > 0 else 0.0
                jp = {"blocked_JP": jp_calib(jk, 0, None, rng_jp),
                      "EB_matched_JP": jp_calib(jk, k_shot, lambda m: k_shot * m[0] / (k_shot * m[0] + m[1]), rng_jp),
                      "EB_matched_JP_w0": jp_calib(jk, k_shot, lambda m: shrink, rng_jp)}
                q = {}
                for a in ALPHAS:
                    s, w, _ = calib_scores(*cal["iid"], 0, 0.0, rng)
                    q[("iid_split", a)] = wquantile(s, np.ones_like(s), 1 - a)
                    q[("iid_k0", a)] = wquantile(s, w, 1 - a)
                    s, w, J = calib_scores(*cal["blocked"], 0, 0.0, rng)
                    q[("blocked_split", a)] = wquantile(s, np.ones_like(s), 1 - a)
                    q[("blocked_k0", a)] = wquantile(s, w, 1 - a)
                    q[("blocked_k0_J", a)] = wquantile(s, w, 1 - a, J)
                    s, w, J = calib_scores(*cal["blocked"], k_shot, shrink, rng)
                    q[("blocked_EBk", a)] = wquantile(s, w, 1 - a)
                    q[("blocked_EBk_J", a)] = wquantile(s, w, 1 - a, J)
                for draw in range(N_DRAWS if k_shot > 0 else 1):
                    lab = rng.choice(len(res_te), k_shot, replace=False) if k_shot > 0 else np.array([], int)
                    b = shrink * res_te[lab].mean() if k_shot > 0 else 0.0
                    ev = np.delete(res_te, lab)
                    for a in ALPHAS:
                        for rule, off, qq in [("iid_split", 0.0, q[("iid_split", a)]),
                                              ("blocked_split", 0.0, q[("blocked_split", a)]),
                                              ("iid", 0.0, q[("iid_k0", a)]),
                                              ("blocked", 0.0, q[("blocked_k0", a)]),
                                              ("EB_point+blocked_k0", b, q[("blocked_k0", a)]),
                                              ("EB_point+iid_k0", b, q[("iid_k0", a)]),
                                              ("EB_matched", b, q[("blocked_EBk", a)]),
                                              ("blocked_J", 0.0, q[("blocked_k0_J", a)]),
                                              ("EB_matched_J", b, q[("blocked_EBk_J", a)])]:
                            cov = np.abs(ev - off) <= qq
                            rows.append(dict(target=target, k=k_shot, draw=draw, country=c, rule=rule, alpha=a,
                                             n=len(ev), n_dig=len(res_te), covered=int(cov.sum()), width=2 * qq))
                    keep = np.delete(np.arange(len(res_te)), lab)
                    for rule, parts in jp.items():
                        bounds = jp_bounds(parts, y_te[lab], lab, keep, ALPHAS)
                        for a, (lo, hi) in bounds.items():
                            cov = (y_te[keep] >= lo) & (y_te[keep] <= hi)
                            rows.append(dict(target=target, k=k_shot, draw=draw, country=c, rule=rule, alpha=a,
                                             n=len(keep), n_dig=len(res_te), covered=int(cov.sum()),
                                             width=float(np.mean(hi - lo))))
        res = pd.DataFrame([r for r in rows if r["target"] == target])
        res["cc"] = res.covered / res.n
        summ = res.groupby(["alpha", "rule", "k"]).apply(lambda g: pd.Series(dict(
            coverage=g.covered.sum() / g.n.sum(),
            country_mean_cov=g.groupby("country").cc.mean().mean(),
            worst_country_cov=g.groupby("country").cc.mean().min(),
            width=np.average(g.width, weights=g.n))))
        print(summ.round(3).to_string(), flush=True)
    pd.DataFrame(rows).to_csv(RES / "fewshot_conformal_LOCO.csv", index=False)
