"""Fig. 5  Country-aware prediction intervals and cross-country alarm classifiers."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, roc_auc_score
from figstyle import (setup, style_axis, numeric_ticks, facet_tag, outside_legend, between_x, hgrid, Composite,
                      source_data, RES, C, INK, MUTED, PALETTE, validate_series_colors, short)

NAME = "Fig5"
TG = ["Temperature", "log_NH4", "Digester_age", "pH"]
W_UNIT = {"Temperature": "°C", "log_NH4": "log$_{10}$", "Digester_age": "years", "pH": "pH units"}
R_COL = validate_series_colors(PALETTE, {"iid_split": C["blue"], "blocked_split": C["teal"], "blocked_JP": C["lav"],
                                         "EB_matched": C["gold"], "EB_matched_JP": C["mauve"]})
R_LAB = {"iid_split": "i.i.d. calibration", "blocked_split": "Country-blocked", "blocked_JP": "Country jackknife+",
         "EB_matched": "EB k-shot matched", "EB_matched_JP": "EB k-shot jackknife+"}
R_LS = {"iid_split": ":", "blocked_split": "-.", "blocked_JP": "--", "EB_matched": "--", "EB_matched_JP": "-"}
R_MK = {"iid_split": "D", "blocked_split": "^", "blocked_JP": "v", "EB_matched": "s", "EB_matched_JP": "o"}
C_COL = validate_series_colors(PALETTE, {"Thermophilic": C["rose"], "TAN_gt2000": C["gold"],
                                         "TAN_gt3000": C["mauve"], "NonSludge": C["teal"]})
C_LAB = {"Thermophilic": "Thermophilic", "TAN_gt2000": "Ammonium N > 2,000", "TAN_gt3000": "Ammonium N > 3,000",
         "NonSludge": "Non-sludge substrate"}


def classifier_axis_label(target):
    return C_LAB[target].replace("Ammonium N > ", "Ammonium N\n> ").replace("Non-sludge substrate", "Non-sludge")


def pooled(c):
    a = c.assign(wn=c.width * c.n)
    g = a.groupby(["target", "rule", "k", "draw"]).agg(cov=("covered", "sum"), n=("n", "sum"), w=("wn", "sum"))
    g["cov"] /= g.n
    g["w"] /= g.n
    return g.reset_index().groupby(["target", "rule", "k"])[["cov", "w"]].mean().reset_index()


def kw(r):
    return dict(color=R_COL[r], ls=R_LS[r], marker=R_MK[r], mec=INK, mew=0.7, lw=2.4 if "EB" in r else 1.9, ms=6.5)


def draw_a(fig, ss, s):
    g = hgrid(ss, [1, 1, 1, 1], gap=0.62)
    axes = [fig.add_subplot(g[i]) for i in range(4)]
    for ax, t in zip(axes, TG):
        for r in R_COL:
            d = s[(s.target == t) & (s.rule == r)].sort_values("k")
            ax.plot(d.k, d["cov"], **kw(r))
        ax.axhline(0.9, color=MUTED, lw=1.0, zorder=0)
        ax.set_ylim(0.7, 1.01)
        ax.set_xticks([0, 1, 2, 3, 5]); ax.set_xlim(-0.3, 5.3)
        ax.set_yticks([0.7, 0.8, 0.9, 1.0])
        style_axis(ax, xlabel="Labelled digesters, k", ylabel="Coverage (α = 0.1)" if t == TG[0] else None)
        style_axis(ax)
        facet_tag(ax, short(t), x=0.97, y=0.04, ha="right", va="bottom")
    h = [Line2D([], [], label=R_LAB[r], **kw(r)) for r in R_COL]
    outside_legend(axes[1], x=between_x(axes[1], axes[2]), ncol=5, handles=h, handlelength=2.4, columnspacing=1.2,
                   pad_pt=8)
    source_data(NAME, "a_b_coverage_width_alpha0.1.csv", s)


def draw_b(fig, ss, s):
    g = hgrid(ss, [1, 1, 1, 1], gap=0.62)
    axes = [fig.add_subplot(g[i]) for i in range(4)]
    for ax, t in zip(axes, TG):
        st = s[(s.target == t) & s.rule.isin(R_COL)]
        top = 1.12 * st.w[np.isfinite(st.w)].max()
        for r in R_COL:
            d = s[(s.target == t) & (s.rule == r)].sort_values("k")
            fin = np.isfinite(d.w)
            ax.plot(d.k[fin], d.w[fin], **kw(r))
            for kk in d.k[~fin]:
                ax.plot([kk], [top], marker=R_MK[r], ls="", ms=6.5, mfc=R_COL[r], mec=INK, mew=0.7, clip_on=False)
                ax.text(kk, top * 1.015, "∞", ha="center", va="bottom", fontsize=12.5, color=INK)
        ax.set_ylim(0, top)
        ax.set_xticks([0, 1, 2, 3, 5]); ax.set_xlim(-0.3, 5.3)
        style_axis(ax, xlabel="Labelled digesters, k", ylabel="Interval width" if t == TG[0] else None)
        numeric_ticks(ax, x=False, n=4)
        style_axis(ax)
        facet_tag(ax, f"{short(t)} ({W_UNIT[t]})", x=0.97, y=0.04, ha="right", va="bottom")


def draw_c(fig, ss, c):
    ax = fig.add_subplot(ss)
    d = c[(c.alpha == 0.1) & (c.k == 0) & c.rule.isin(["iid_split", "blocked_split", "blocked_JP"])]
    d = d.groupby(["target", "rule", "country"]).agg(covered=("covered", "sum"), n=("n", "sum")).reset_index()
    d["cov"] = d.covered / d.n
    rng = np.random.default_rng(3)
    for i, t in enumerate(TG):
        for j, r in enumerate(["iid_split", "blocked_split", "blocked_JP"]):
            v = d[(d.target == t) & (d.rule == r)]["cov"].values
            x = i + (j - 1) * 0.27
            ax.scatter(x + rng.uniform(-0.06, 0.06, len(v)), v, s=26, color=R_COL[r], edgecolor=INK, lw=0.5,
                       marker=R_MK[r], zorder=3)
            ax.text(x, 1.04, f"{(v < 0.8).sum()}", ha="center", fontsize=10, color=INK)
    ax.axhline(0.9, color=MUTED, lw=1.0, zorder=0)
    ax.axhline(0.8, color=MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_xticks(range(len(TG)))
    ax.set_xticklabels([short(t) for t in TG], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(TG) - 0.4)
    ax.set_ylim(-0.03, 1.1)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    style_axis(ax, ylabel="Country coverage, k = 0")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 1)
    h = [Line2D([], [], marker=R_MK[r], ls="", ms=7, mfc=R_COL[r], mec=INK, label=R_LAB[r])
         for r in ["iid_split", "blocked_split", "blocked_JP"]]
    ax.legend(handles=h, loc="lower right", fontsize=10, handletextpad=0.1, borderaxespad=0.05)
    source_data(NAME, "c_country_coverage_k0.csv", d)


def classifier_data():
    store = pd.read_pickle(RES / "classification_oof.pkl")
    pcs = pd.read_csv(RES / "classification_per_country.csv")
    cl = pd.read_csv(RES / "classification_LOCO.csv").set_index(["target", "scheme"])
    return store, pcs, cl


def draw_d(fig, ss):
    ax = fig.add_subplot(ss)
    store, _, cl = classifier_data()
    rows = []
    for t in C_COL:
        d = store[(t, "LOCO")]
        fpr, tpr, _ = roc_curve(d.y, d.p)
        ax.plot(fpr, tpr, color=C_COL[t], lw=2.4, label=f"{C_LAB[t]} ({roc_auc_score(d.y, d.p):.2f})")
        rows.append(pd.DataFrame({"target": t, "fpr": fpr, "tpr": tpr}))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls="--", zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    style_axis(ax, xlabel="False-positive rate", ylabel="True-positive rate")
    numeric_ticks(ax, n=3)
    style_axis(ax)
    ax.legend(loc="lower right", fontsize=10, title="LOCO AUC", title_fontsize=10, handlelength=1.3,
              borderaxespad=0.05, labelspacing=0.3)
    source_data(NAME, "d_roc_LOCO.csv", pd.concat(rows))
    source_data(NAME, "d_e_f_classification_summary.csv", cl.reset_index())


def draw_e(fig, ss):
    ax = fig.add_subplot(ss)
    _, pcs, cl = classifier_data()
    rng = np.random.default_rng(4)
    p = pcs[pcs.scheme == "LOCO"]
    for i, t in enumerate(C_COL):
        v = p[p.target == t]
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(v)), v.AUC, s=16 + 2.2 * v.n, color=C_COL[t], edgecolor=INK,
                   lw=0.6, alpha=0.95, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [v.AUC.median()] * 2, color=INK, lw=2.2, zorder=4)
        ax.text(i, 1.06, f"{cl.loc[(t, 'LOCO'), 'n_pos']:.0f}/{cl.loc[(t, 'LOCO'), 'n_dig']:.0f}", ha="center",
                fontsize=10, color=INK)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_xticks(range(len(C_COL)))
    ax.set_xticklabels([classifier_axis_label(t) for t in C_COL], rotation=90,
                       ha="center")
    ax.set_xlim(-0.6, len(C_COL) - 0.4)
    ax.set_ylim(-0.04, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax, ylabel="Country AUC (LOCO)")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 1)
    source_data(NAME, "e_country_AUC_LOCO.csv", p)


def draw_f(fig, ss):
    ax = fig.add_subplot(ss)
    store, _, _ = classifier_data()
    rel = []
    for t in C_COL:
        d = store[(t, "LOCO")]
        q = pd.qcut(d.p.rank(method="first"), 5, labels=False)
        b = d.groupby(q).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        ax.plot(b.p, b.y, color=C_COL[t], lw=2.0, marker="o", ms=6.5, mec=INK, mew=0.7, label=C_LAB[t])
        rel.append(b.assign(target=t))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.1, ls="--", zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    style_axis(ax, xlabel="Mean predicted probability", ylabel="Observed frequency")
    numeric_ticks(ax, n=3)
    style_axis(ax)
    source_data(NAME, "f_reliability_quintiles_LOCO.csv", pd.concat(rel).reset_index(drop=True))


if __name__ == "__main__":
    setup()
    c = pd.read_csv(RES / "fewshot_conformal_LOCO.csv")
    s = pooled(c[c.alpha == 0.1])
    s = s[s.rule.isin(R_COL)]
    cf = Composite(NAME, "Fig5_intervals_alarms", height=12.4)
    rows = cf.grid([0.88, 0.8, 1.0], gap=0.5)
    cf.panel("a", rows[0], lambda f, ss: draw_a(f, ss, s))
    cf.panel("b", rows[1], lambda f, ss: draw_b(f, ss, s))
    r3 = hgrid(rows[2], [1.1, 1, 1, 0.9], gap=0.62)
    cf.panel("c", r3[0], lambda f, ss: draw_c(f, ss, c))
    cf.panel("d", r3[1], draw_d)
    cf.panel("e", r3[2], draw_e)
    cf.panel("f", r3[3], draw_f)
    rep = cf.finish()
    print({r["panel"]: round(r["max_residual_pt"], 2) for r in rep})
