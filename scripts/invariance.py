"""Country-invariance components:
  - AnchorLGBM: gradient boosting with the anchor-regression loss
        L(f) = ||(I - P_A)(y - f)||^2 + gamma * ||P_A (y - f)||^2,
    A = one-hot country anchors, so P_A r is the country-mean residual.
  - invariant_screen: rank taxa by sign-consistent within-country association with the target.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin

LGB_KW = dict(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=8,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, verbose=-1)


def anchor_objective(groups, gamma):
    codes, inv = np.unique(groups, return_inverse=True)
    n_c = np.bincount(inv).astype(float)

    def obj(y_true, y_pred):
        r = y_pred - y_true
        mean_r = np.bincount(inv, weights=r) / n_c
        grad = r + (gamma - 1.0) * mean_r[inv]
        hess = 1.0 + (gamma - 1.0) / n_c[inv]
        return grad, np.maximum(hess, 1e-3)
    return obj


class AnchorLGBM(BaseEstimator, RegressorMixin):
    def __init__(self, gamma=1.0, n_jobs=1, random_state=0, **kw):
        self.gamma = gamma
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kw = kw

    def fit(self, X, y, groups=None):
        y = np.asarray(y, float)
        self.y0_ = y.mean()
        params = {**LGB_KW, **self.kw}
        if groups is None or self.gamma == 1.0:
            self.m_ = LGBMRegressor(**params, n_jobs=self.n_jobs, random_state=self.random_state)
            self.m_.fit(X, y)
            self.custom_ = False
        else:
            self.m_ = LGBMRegressor(**params, objective=anchor_objective(groups, self.gamma),
                                    n_jobs=self.n_jobs, random_state=self.random_state)
            self.m_.fit(X, y - self.y0_)
            self.custom_ = True
        return self

    def predict(self, X):
        p = self.m_.predict(X, raw_score=self.custom_)
        return p + self.y0_ if self.custom_ else p


def within_country_rho(X, y, groups, min_n=4):
    """Spearman rho per (country, taxon) on digester-level data. Returns DataFrame countries x taxa and weights."""
    X = np.asarray(X)
    y = np.asarray(y)
    rows, w, names = [], [], []
    for c in np.unique(groups):
        idx = groups == c
        if idx.sum() < min_n or np.std(y[idx]) == 0:
            continue
        yr = rankdata(y[idx])
        Xr = np.apply_along_axis(rankdata, 0, X[idx])
        Xr = Xr - Xr.mean(0)
        yr = yr - yr.mean()
        den = np.sqrt((Xr ** 2).sum(0) * (yr ** 2).sum())
        rho = np.where(den > 0, (Xr * yr[:, None]).sum(0) / np.where(den > 0, den, 1), 0.0)
        rows.append(rho)
        w.append(idx.sum() - 3)
        names.append(c)
    return np.array(rows), np.array(w, float), names


def invariant_screen(X, y, groups, k=100, min_n=4):
    """Score = |pooled within-country rho| * sign consistency across countries."""
    R, w, _ = within_country_rho(X, y, groups, min_n)
    z = np.arctanh(np.clip(R, -0.99, 0.99))
    pooled = (w[:, None] * z).sum(0) / w.sum()
    consist = (w[:, None] * (np.sign(R) == np.sign(pooled))).sum(0) / w.sum()
    score = np.abs(np.tanh(pooled)) * consist
    order = np.argsort(-score)
    return order[:k], score
