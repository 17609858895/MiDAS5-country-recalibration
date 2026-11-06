"""Model zoo (20 regressors) and evaluation helpers shared by benchmark scripts."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV, BayesianRidge, HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

SEED = 0


class PLSWrap(PLSRegression):
    def predict(self, X):
        return super().predict(X).ravel()


def model_zoo(n_jobs=4):
    sc = StandardScaler
    alphas = np.logspace(-2, 4, 40)
    zoo = {
        "Ridge": make_pipeline(sc(), RidgeCV(alphas=alphas)),
        "Lasso": make_pipeline(sc(), LassoCV(n_alphas=40, cv=5, max_iter=5000, n_jobs=1, random_state=SEED)),
        "ElasticNet": make_pipeline(sc(), ElasticNetCV(l1_ratio=[0.2, 0.5, 0.8], n_alphas=30, cv=5,
                                                       max_iter=5000, n_jobs=1, random_state=SEED)),
        "BayesRidge": make_pipeline(sc(), BayesianRidge()),
        "PLS": make_pipeline(sc(), PLSWrap(n_components=8)),
        "KernelRidge": make_pipeline(sc(), KernelRidge(alpha=1.0, kernel="rbf", gamma=1e-3)),
        "SVR": make_pipeline(sc(), SVR(C=3.0, epsilon=0.05, gamma="scale")),
        "KNN": make_pipeline(sc(), PCA(30, random_state=SEED), KNeighborsRegressor(7, weights="distance")),
        "GPR": make_pipeline(sc(), PCA(40, random_state=SEED), GaussianProcessRegressor(
            ConstantKernel() * RBF(10.0) + WhiteKernel(), normalize_y=True, random_state=SEED)),
        "DecisionTree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=SEED),
        "RF": RandomForestRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=n_jobs, random_state=SEED),
        "ExtraTrees": ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=n_jobs, random_state=SEED),
        "Bagging": BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=3), n_estimators=200, max_features=0.5,
                                    n_jobs=n_jobs, random_state=SEED),
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, learning_rate=0.05,
                                      random_state=SEED),
        "GBRT": GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
                                          max_features=0.3, random_state=SEED),
        "HistGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                                min_samples_leaf=8, l2_regularization=1.0, random_state=SEED),
        "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8,
                                colsample_bytree=0.3, min_child_weight=3, reg_lambda=1.0, n_jobs=n_jobs,
                                random_state=SEED, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15, min_child_samples=8,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0,
                                  n_jobs=n_jobs, random_state=SEED, verbose=-1),
        "CatBoost": CatBoostRegressor(iterations=600, learning_rate=0.05, depth=5, rsm=0.3, l2_leaf_reg=3,
                                      thread_count=n_jobs, random_seed=SEED, verbose=0),
        "MLP": make_pipeline(sc(), MLPRegressor((128, 32), alpha=1.0, learning_rate_init=1e-3, max_iter=800,
                                                early_stopping=True, random_state=SEED)),
    }
    return zoo


def metrics(y, p):
    return dict(R2=r2_score(y, p), RMSE=float(np.sqrt(mean_squared_error(y, p))), MAE=mean_absolute_error(y, p))


def digester_level(pred, y, dig):
    df = pd.DataFrame({"p": pred, "y": y, "d": dig})
    g = df.groupby("d").mean()
    return g.y.values, g.p.values, g.index.values
