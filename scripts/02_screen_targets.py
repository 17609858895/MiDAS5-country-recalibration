"""Quick screen: which metadata variables are predictable from the microbiome
under digester-grouped CV and leave-country-out CV."""
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, balanced_accuracy_score, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from common import load_meta, amp_features, RES

warnings.filterwarnings("ignore")
meta = load_meta()
X13, _, _ = amp_features("V13")
X35, _, _ = amp_features("V35")
X4, _, _ = amp_features("V4")
X = X13.join(X35, how="inner")
print("V13+V35 samples", X.shape)

meta = meta.loc[X.index]
targets_reg = {
    "Temperature": meta.Temperature,
    "log_Ammonium": np.log10(meta.Ammonium.where(meta.Ammonium > 0)),
    "pH": meta.pH,
    "log_SRT": np.log10(meta.SRT.where(meta.SRT > 0)),
    "TS": meta.TS,
    "VS": meta.VS,
    "Digester_age": meta.Digester_age,
    "log_Volume": np.log10(meta.Volumen.where(meta.Volumen > 0)),
    "Latitude": meta.Latitude,
}
targets_clf = {
    "Thermophilic": (meta.Temperature_range == "Thermophilic").astype(float).where(meta.Temperature_range.notna()),
    "Substrate": meta.Primary_substrate.where(meta.Primary_substrate != "Other"),
    "Technology": meta.Digester_technology.where(meta.Digester_technology.isin(["CSTR", "TSAD"])),
    "Location": meta.Digester_location.where(meta.Digester_location.isin(["WWTP", "Agriculture", "Industry"])),
}
rows = []
for scheme in ["digesterGKF", "LOCountry"]:
    for name, y in {**targets_reg, **targets_clf}.items():
        ok = y.notna()
        Xi, yi = X[ok], y[ok]
        groups = meta.loc[ok, "Digester"] if scheme == "digesterGKF" else meta.loc[ok, "Country"]
        cv = GroupKFold(5) if scheme == "digesterGKF" else LeaveOneGroupOut()
        if name in targets_reg:
            et = ExtraTreesRegressor(500, max_features=0.3, min_samples_leaf=2, n_jobs=-1, random_state=0)
            p = cross_val_predict(et, Xi, yi, groups=groups, cv=cv)
            rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 4, 30)))
            p2 = cross_val_predict(rg, Xi, yi, groups=groups, cv=cv)
            rows.append(dict(scheme=scheme, target=name, n=len(yi), n_dig=meta.loc[ok, "Digester"].nunique(),
                             metric="R2", ET=r2_score(yi, p), Ridge=r2_score(yi, p2)))
        else:
            et = ExtraTreesClassifier(500, max_features="sqrt", min_samples_leaf=1, n_jobs=-1,
                                      random_state=0, class_weight="balanced")
            p = cross_val_predict(et, Xi, yi, groups=groups, cv=cv)
            rows.append(dict(scheme=scheme, target=name, n=len(yi), n_dig=meta.loc[ok, "Digester"].nunique(),
                             metric="BalAcc", ET=balanced_accuracy_score(yi, p), Ridge=np.nan,
                             classes=str(yi.value_counts().to_dict())))
        print(rows[-1])
pd.DataFrame(rows).to_csv(RES / "screen_targets.csv", index=False)
