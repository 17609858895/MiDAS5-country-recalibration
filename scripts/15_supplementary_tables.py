"""Supplementary Tables S1-S3.
S1: digesters, labels, DML sample and sequencing run per country and amplicon.
S2: hyperparameters of the 20 regressors, stacking, alarm classifier and DML nuisance learners.
S3: per-country digester-level ExtraTrees residuals under GKF5 and LOCO.
Writes tables/Supplementary_Tables.xlsx and one CSV per table."""
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, Side
from common import load_meta, RES
from models import digester_level
import importlib

bench = importlib.import_module("03_benchmark20")
ROOT = RES.parent
OUT = ROOT / "tables"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data_extracted"
NOTES = {
    "S1": "Note: Counts are digesters. TAN, total ammonia nitrogen; SRT, solids retention time; DML, double "
          "machine learning. Totals in parentheses give countries. Runs are the sequencing-run prefixes of the "
          "data release.",
    "S2": "Note: RBF, radial basis function; DML, double machine learning. Settings not listed were library "
          "defaults.",
    "S3": "Note: Offset, mean digester-level residual (observed − predicted); SD, standard deviation of the target "
          "over labelled digesters; MAE, mean absolute error. Units: °C, pH units, log10 mg L⁻¹ (TAN), log10 d "
          "(SRT), years (digester age).",
    "S4": "Note: Variances (squared target units) from a nested random-effects model of digester-level LOCO "
          "residuals. Run share, run/(run + country) variance; P, 5,000 permutations of run labels among countries. "
          "The run-mate prior was not evaluated for SRT.",
    "S5": "Note: Countries linked by a shared V1-3 or V3-5 run were held out together. Random groups: 30 random "
          "assignments of the same countries to groups of the same sizes.",
}
TITLES = {"S4": "Sequencing-run variance components of ExtraTrees country offsets and run-mate prior",
          "S5": "R² of ExtraTrees with run-linked countries held out together"}
TARGETS = {"Temperature": "Temperature", "pH": "pH", "log_NH4": "TAN", "log_SRT": "SRT",
           "Digester_age": "Digester age"}


def table_s1():
    meta = load_meta().reset_index()
    meta["Digester"] = meta.SampleID.str[:-2]
    X, T, m = bench.build_dataset()
    feat = m.assign(**{c: T[c] for c in T})
    rows = meta.groupby(["Continent", "Country"]).agg(Digesters=("Digester", "nunique"),
                                                       Samples=("SampleID", "size")).reset_index()
    by = lambda ser: rows.Country.map(ser)
    rows["Digesters with community features"] = by(feat.groupby("Country").Digester.nunique())
    for c, lab in TARGETS.items():
        rows[f"Labelled: {lab}"] = by(feat[feat[c].notna()].groupby("Country").Digester.nunique())
    dig = pd.read_csv(RES / "digester_level_table.csv")
    d = dig[dig.log_TAN.notna() & dig.CLR_Methanothrix.notna()].groupby("Country").size()
    rows["DML sample"] = by(d[d >= 3])
    for amp, f in [("V1-3", "V13_metadata.txt"), ("V3-5", "V35_metadata.txt"), ("V4", "V4c_metadata.txt")]:
        mp = pd.read_csv(DATA / f, sep="\t", header=None, names=["seq", "SampleID"])
        mp["run"] = mp.seq.str.split("-").str[0]
        mp = mp.merge(meta[["SampleID", "Country"]], on="SampleID")
        runs = mp.groupby("Country").run.agg(lambda s: "; ".join(sorted(set(s))))
        rows[f"Sequencing run {amp}"] = by(runs)
    cnt = [c for c in rows.columns if c.startswith(("Digesters", "Samples", "Labelled", "DML"))]
    rows[cnt] = rows[cnt].fillna(0).astype(int)
    tot = {"Continent": "Total", "Country": f"{rows.Country.nunique()} countries"}
    tot.update({c: int(rows[c].sum()) for c in cnt})
    for c in cnt[3:]:
        tot[c] = f"{int(rows[c].sum())} ({int((rows[c] > 0).sum())})"
    return pd.concat([rows, pd.DataFrame([tot])], ignore_index=True)


def table_s2():
    r = [
        ("Linear", "Ridge", "Standardised features; regularisation strength from 40 values between 10⁻² and 10⁴ by internal cross-validation"),
        ("Linear", "Lasso", "Standardised features; 40 regularisation strengths, five-fold internal cross-validation"),
        ("Linear", "Elastic net", "Standardised features; l1 ratio 0.2, 0.5 or 0.8 and 30 regularisation strengths, five-fold internal cross-validation"),
        ("Linear", "Bayesian ridge", "Standardised features; default priors"),
        ("Linear", "Partial least squares", "Standardised features; 8 components"),
        ("Kernel", "Kernel ridge", "Standardised features; RBF kernel, γ = 10⁻³, α = 1.0"),
        ("Kernel", "Support-vector regression", "Standardised features; RBF kernel, C = 3.0, ε = 0.05, γ scaled to feature variance"),
        ("Kernel", "Gaussian process", "Standardised features projected on 40 principal components; constant × RBF (length scale 10) + white-noise kernel, normalised target"),
        ("Neighbours", "k-nearest neighbours", "Standardised features projected on 30 principal components; 7 neighbours, distance weighting"),
        ("Tree", "Decision tree", "Maximum depth 6, minimum leaf size 5"),
        ("Tree ensemble", "Random forest", "500 trees, 30% of features per split, minimum leaf size 2"),
        ("Tree ensemble", "ExtraTrees", "500 trees, 30% of features per split, minimum leaf size 2"),
        ("Tree ensemble", "Bagging", "200 trees (minimum leaf size 3), 50% of features per tree"),
        ("Boosting", "AdaBoost", "300 trees of depth 4, learning rate 0.05"),
        ("Boosting", "Gradient boosting", "400 trees of depth 3, learning rate 0.03, row subsample 0.8, 30% of features per split"),
        ("Boosting", "Histogram gradient boosting", "400 iterations, learning rate 0.04, 15 leaves, minimum leaf size 8, L2 penalty 1.0"),
        ("Boosting", "XGBoost", "500 trees of depth 4, learning rate 0.03, row subsample 0.8, column subsample 0.3, minimum child weight 3, L2 penalty 1.0"),
        ("Boosting", "LightGBM", "500 trees, 15 leaves, learning rate 0.03, minimum child samples 8, row subsample 0.8, column subsample 0.3, L2 penalty 1.0"),
        ("Boosting", "CatBoost", "600 iterations of depth 5, learning rate 0.05, feature subsample 0.3, L2 penalty 3"),
        ("Neural network", "Multilayer perceptron", "Standardised features; hidden layers of 128 and 32 units, L2 penalty 1.0, initial learning rate 10⁻³, at most 800 epochs, early stopping"),
        ("Stacking", "Base learners", "Ridge, kernel ridge, support-vector regression, Gaussian process, random forest, ExtraTrees, histogram gradient boosting, XGBoost, LightGBM and CatBoost with the settings above"),
        ("Stacking", "Combiner", "Non-negative least squares without intercept on inner out-of-fold predictions, weights normalised to sum to one"),
        ("Alarm classifier", "ExtraTrees", "500 trees, square root of the number of features per split, balanced class weights"),
        ("DML nuisance", "LightGBM", "300 trees, 7 leaves, learning rate 0.03, minimum child samples 8, row subsample 0.8, column subsample 0.7"),
        ("DML nuisance", "Random forest", "Median imputation; 400 trees, 50% of confounders per split, minimum leaf size 3"),
        ("DML nuisance", "Lasso", "Median imputation and standardisation; regularisation by five-fold internal cross-validation"),
    ]
    return pd.DataFrame(r, columns=["Group", "Model", "Settings"])


def table_s3():
    X, T, meta = bench.build_dataset()
    oof = pd.read_pickle(RES / "benchmark20_oof.pkl")
    rows = []
    for c, lab in TARGETS.items():
        ok = T[c].notna().values
        y, dig, cty = T[c].values[ok], meta.Digester.values[ok], meta.Country.values[ok]
        dc = pd.Series(cty, index=dig).groupby(level=0).first()
        sd = None
        for scheme in ["GKF5", "LOCO"]:
            yd, pdg, d = digester_level(oof[(c, scheme, "ExtraTrees")], y, dig)
            df = pd.DataFrame({"y": yd, "p": pdg, "country": dc.loc[d].values})
            sd = df.y.std(ddof=0) if sd is None else sd
            df["r"] = df.y - df.p
            g = df.groupby("country").agg(n=("y", "size"), obs=("y", "mean"), off=("r", "mean"),
                                          mae=("r", lambda s: s.abs().mean()))
            g["scheme"] = scheme
            rows.append(g.assign(target=lab, sd=sd).reset_index())
    a = pd.concat(rows)
    w = a.pivot_table(index=["target", "country"], columns="scheme", values=["off", "mae"]).reset_index()
    w.columns = ["Target", "Country", "MAE GKF5", "MAE LOCO", "Offset GKF5", "Offset LOCO"]
    base = a[a.scheme == "LOCO"][["target", "country", "n", "obs", "sd"]].rename(
        columns={"target": "Target", "country": "Country", "n": "Digesters", "obs": "Observed mean"})
    t = base.merge(w, on=["Target", "Country"])
    t["Offset LOCO / SD"] = t["Offset LOCO"] / t.pop("sd")
    order = {v: i for i, v in enumerate(TARGETS.values())}
    t = t.sort_values(["Target", "Country"], key=lambda s: s.map(order) if s.name == "Target" else s)
    cols = ["Target", "Country", "Digesters", "Observed mean", "Offset GKF5", "Offset LOCO",
            "Offset LOCO / SD", "MAE GKF5", "MAE LOCO"]
    t = t[cols]
    num = cols[3:]
    t[num] = t[num].round(3)
    return t


GROUPING = {"run_V13": "V1-3 run", "run_V35": "V3-5 run", "run_pair": "V1-3 and V3-5 runs"}


def sig3(v):
    # REML estimates at the zero boundary are reported as 0
    return 0.0 if abs(v) < 1e-6 else float(f"{v:.3g}")


def table_s4():
    vs = pd.read_csv(RES / "run_offset_variance.csv")
    vs = vs[vs.grouping.isin(GROUPING)]
    rp = pd.read_csv(RES / "runmate_prior_zero_shot.csv")
    t = vs.merge(rp[["target", "grouping", "R2_LOCO", "R2_prior"]], on=["target", "grouping"], how="left")
    t = t.assign(Target=t.target.map(TARGETS), Grouping=t.grouping.map(GROUPING))
    t = t.rename(columns={"countries": "Countries", "groups": "Runs", "var_group": "Run variance",
                          "var_country": "Country variance within run", "var_digester": "Digester variance",
                          "share_group": "Run share", "p_perm": "Permutation P", "R2_LOCO": "LOCO R²",
                          "R2_prior": "LOCO R² with run-mate prior"})
    order = {v: i for i, v in enumerate(TARGETS.values())}
    t = t.sort_values(["Target", "grouping"], key=lambda s: s.map(order) if s.name == "Target" else s)
    cols = ["Target", "Grouping", "Countries", "Runs", "Run variance", "Country variance within run",
            "Digester variance", "Run share", "Permutation P", "LOCO R²", "LOCO R² with run-mate prior"]
    t = t[cols].reset_index(drop=True)
    for c in cols[4:7]:
        t[c] = t[c].map(sig3)
    t[cols[7:]] = t[cols[7:]].round(3)
    return t


def table_s5():
    h = pd.read_csv(RES / "run_holdout.csv")
    t = pd.DataFrame({"Target": h.target.map(TARGETS), "Countries": h.countries,
                      "Countries per held-out group": h.group_sizes, "LOCO R²": h.R2_LOCO,
                      "Run hold-out R²": h.R2_run_holdout, "Random groups R², mean": h.R2_random_mean,
                      "Random groups R², minimum": h.R2_random_min, "Random groups R², maximum": h.R2_random_max,
                      "Random groupings with lower R²": h.share_random_below})
    num = t.columns[3:]
    t[num] = t[num].round(3)
    return t


def write(tables):
    wb = Workbook()
    wb.remove(wb.active)
    thin, thick = Side(style="thin"), Side(style="medium")
    for name, (title, df, note) in tables.items():
        ws = wb.create_sheet(name)
        ws.append([f"{name}. {title}"])
        ws["A1"].font = Font(name="Times New Roman", bold=True, size=11)
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and np.isnan(v)) else v for v in row])
        for j, _ in enumerate(df.columns, 1):
            h = ws.cell(2, j)
            h.font = Font(name="Times New Roman", bold=True, size=10)
            h.border = Border(top=thick, bottom=thin)
            h.alignment = Alignment(wrap_text=True, vertical="bottom")
            ws.cell(ws.max_row, j).border = Border(bottom=thick)
            width = max(len(str(df.columns[j - 1])), *(len(str(v)) for v in df.iloc[:, j - 1])) + 2
            ws.column_dimensions[ws.cell(2, j).column_letter].width = min(width, 70)
        for r in ws.iter_rows(min_row=3):
            for c in r:
                c.font = Font(name="Times New Roman", size=10)
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.append([])
        ws.append([note])
        ws.cell(ws.max_row, 1).font = Font(name="Times New Roman", italic=True, size=9)
        ws.freeze_panes = "A3"
    wb.save(OUT / "Supplementary_Tables.xlsx")


if __name__ == "__main__":
    s1, s2, s3, s4, s5 = table_s1(), table_s2(), table_s3(), table_s4(), table_s5()
    for k, v in {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}.items():
        v.to_csv(OUT / f"Table_{k}.csv", index=False, encoding="utf-8-sig")
    write({"Table S1": ("Digesters, process labels and sequencing runs per country", s1, NOTES["S1"]),
           "Table S2": ("Settings of the regression, stacking, classification and nuisance models", s2, NOTES["S2"]),
           "Table S3": ("Country offsets and mean absolute errors of ExtraTrees under GKF5 and LOCO", s3,
                        NOTES["S3"]),
           "Table S4": (TITLES["S4"], s4, NOTES["S4"]),
           "Table S5": (TITLES["S5"], s5, NOTES["S5"])})
    print(s1.to_string())
    print(s3.groupby("Target").size())
    pd.set_option("display.width", 250)
    print(s4.to_string())
    print(s5.to_string())
