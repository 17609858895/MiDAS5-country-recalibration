"""Fig. 6  Within-country ammonia–microbiome associations and pathway context."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.stats import norm
from figstyle import (setup, style_axis, numeric_ticks, outside_legend, between_x, hgrid, Composite, source_data,
                      RES, C, INK, MUTED, RULE, PALETTE, validate_series_colors)

NAME = "Fig6"
OUT = ["CLR_Methanoculleus", "log_SAOB", "LR_hydro_vs_Mthrix", "CLR_Methanobacterium", "CLR_Methanospirillum",
       "CLR_Methanothrix", "CLR_Methanosarcina", "log_Cloacimonas"]
OLAB = {"CLR_Methanoculleus": r"$\mathit{Methanoculleus}$", "log_SAOB": "SAOB (log)",
        "LR_hydro_vs_Mthrix": r"Hydrogenotrophs : $\mathit{Methanothrix}$",
        "CLR_Methanobacterium": r"$\mathit{Methanobacterium}$", "CLR_Methanospirillum": r"$\mathit{Methanospirillum}$",
        "CLR_Methanothrix": r"$\mathit{Methanothrix}$", "CLR_Methanosarcina": r"$\mathit{Methanosarcina}$",
        "log_Cloacimonas": r"$\mathit{Cloacimonas}$ (log)"}
V_COL = validate_series_colors(PALETTE, {"pooled": C["blue"], "FE": C["lav"], "demeaned": C["rose"]})
V_LAB = {"pooled": "Pooled", "FE": "Country dummies in W", "demeaned": "Within-country (demeaned)"}
L_MK = {"LGBM": "o", "RF": "^", "Lasso": "s"}
DIR_COL = validate_series_colors(PALETTE, {"up": C["rose"], "down": C["blue"], "tan": C["gold"],
                                           "aceto": C["lav"], "alt": C["pink"]})
ROB_COL = validate_series_colors(PALETTE, {"placebo": C["lav"], "real": C["rose"], "loco": C["blue"],
                                           "subset": C["gold"], "rcc": C["teal"]})
BW_OUT = ["CLR_Methanoculleus", "log_SAOB", "CLR_Methanobacterium", "CLR_Methanospirillum", "CLR_Methanothrix",
          "LR_hydro_vs_Mthrix"]
BW_COL = validate_series_colors(PALETTE, {"between": C["blue"], "within": C["rose"], "digester": C["lav"],
                                          "country": C["teal"]})


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def load():
    w = pd.read_csv(RES / "dml_within_ate.csv")
    w = w[w.treatment == "log_TAN"].copy()
    w["p"] = 2 * norm.sf(np.abs(w.z))
    w["q"] = w.groupby(["variant", "learner"]).p.transform(bh)
    return w


def box(ax, x0, y0, w, h, fc, text=None, size=11.5):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.25", fc=fc, ec=INK, lw=1.1,
                                zorder=2))
    if text:
        ax.text(x0 + w / 2, y0 + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3,
                zorder=3)


def arrow(ax, a, b, color=INK, lw=1.5, style="-|>", conn="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=15, color=color, lw=lw,
                                 connectionstyle=conn, shrinkA=1, shrinkB=1, zorder=4))


def draw_a(fig, ss, w, r):
    """Pathway schematic with the within-country coefficients (LGBM, demeaned) of a 10-fold ammonium-N increase."""
    ax = fig.add_subplot(ss)
    d = w[(w.variant == "demeaned") & (w.learner == "LGBM")].set_index("outcome")
    real = r[r.test == "real_demeaned"].set_index("outcome").theta
    pl = r[r.test == "placebo_within_demeaned"]
    pval = {o: (1 + np.sum(np.abs(pl[pl.outcome == o].theta.values) >= abs(real[o]))) / 201 for o in OUT}

    def row(x0, x1, y, o, label=None):
        th = d.loc[o, "theta"]
        ns = pval[o] >= 0.05
        ax.text(x0, y, label or OLAB[o], ha="left", va="center", fontsize=11.5, color=INK, zorder=3)
        ax.text(x1, y, f"{th:+.2f}" + (" (n.s.)" if ns else ""), ha="right", va="center", fontsize=11.5,
                color=MUTED if ns else INK, zorder=3)
        ax.scatter([x1 + 0.22], [y], marker="^" if th > 0 else "v", s=46, zorder=3,
                   color="white" if ns else (DIR_COL["up"] if th > 0 else DIR_COL["down"]), edgecolor=INK, lw=0.7)

    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 3.2, 8.85, 3.4, 1.0, DIR_COL["tan"], "Ammonium N")
    box(ax, 0.0, 4.55, 1.55, 1.0, "white", "Acetate")
    box(ax, 8.55, 4.55, 1.4, 1.0, "white", "CH$_4$")
    box(ax, 2.25, 5.85, 5.45, 2.25, DIR_COL["aceto"])
    ax.text(4.97, 7.68, "Acetoclastic methanogenesis", ha="center", va="center", fontsize=11.5, color=INK, zorder=3)
    row(2.45, 7.05, 7.0, "CLR_Methanothrix")
    row(2.45, 7.05, 6.35, "CLR_Methanosarcina")
    box(ax, 0.3, 1.6, 2.95, 2.15, DIR_COL["alt"])
    ax.text(1.77, 3.2, "Syntrophic acetate\noxidation", ha="center", va="center", fontsize=11.5, color=INK, zorder=3)
    row(0.5, 2.6, 2.1, "log_SAOB", label="SAOB")
    box(ax, 4.8, 0.1, 4.4, 3.95, DIR_COL["alt"])
    ax.text(7.0, 3.45, "Hydrogenotrophic\nmethanogenesis", ha="center", va="center", fontsize=11.5, color=INK,
            zorder=3)
    for i, o in enumerate(["CLR_Methanoculleus", "CLR_Methanobacterium", "CLR_Methanospirillum"]):
        row(4.97, 8.55, 2.45 - 0.78 * i, o)
    ax.text(4.02, 2.95, "H$_2$ +\nCO$_2$", ha="center", va="bottom", fontsize=10.5, color=INK, linespacing=1.2)
    arrow(ax, (1.0, 5.55), (2.25, 6.6))
    arrow(ax, (7.7, 6.6), (9.1, 5.55))
    arrow(ax, (0.6, 4.55), (0.9, 3.75))
    arrow(ax, (3.25, 2.75), (4.8, 2.75))
    arrow(ax, (9.15, 2.6), (9.4, 4.55))
    ax.text(4.9, 8.47, "θ per 10-fold Ammonium N", ha="center", va="center", fontsize=11, color=INK)


def panel_b(ax, w, letter=None, xlim=(-4.2, 5.3), treat="Ammonium N", src=(NAME, "b_dml_estimates_logTAN.csv")):
    y = 0
    yt, ylab = [], []
    for o in OUT[::-1]:
        for j, v in enumerate(["pooled", "FE", "demeaned"]):
            yy = y + (j - 1) * 0.27
            for lrn in ["RF", "Lasso"]:
                rr = w[(w.outcome == o) & (w.variant == v) & (w.learner == lrn)].iloc[0]
                ax.scatter(rr.theta, yy, marker=L_MK[lrn], s=28, facecolor="white", edgecolor=INK, lw=0.9, zorder=3)
            rr = w[(w.outcome == o) & (w.variant == v) & (w.learner == "LGBM")].iloc[0]
            ax.errorbar(rr.theta, yy, xerr=[[rr.theta - rr.ci_lo], [rr.ci_hi - rr.theta]], fmt="o", ms=6.5,
                        color=V_COL[v], mec=INK, mew=0.6, elinewidth=2.0, capsize=0, zorder=4)
            if v == "demeaned":
                q = rr.q
                ax.text(xlim[1] + 0.08, yy, f"q = {q:.3f}" if q >= 0.001 else "q < 0.001", va="center", ha="left",
                        fontsize=10.5, color=INK if q < 0.1 else MUTED)
        yt.append(y); ylab.append(OLAB[o])
        if y > 0:
            ax.axhline(y - 0.5, color="#E3E6EA", lw=0.9, zorder=0)
        y += 1
    ax.axvline(0, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab)
    ax.set_ylim(-0.6, y - 0.4)
    ax.set_xlim(*xlim)
    style_axis(ax, xlabel=f"θ per 10-fold {treat} (95% CI)")
    ax.set_xticks([-4, -2, 0, 2, 4])
    style_axis(ax)
    ax.spines["bottom"].set_bounds(*xlim)
    h = [Line2D([], [], marker="o", ls="-", color=V_COL[v], mec=INK, ms=7, lw=2, label=V_LAB[v])
         for v in ["pooled", "FE", "demeaned"]]
    h += [Line2D([], [], marker=L_MK[lr], ls="", mfc="white", mec=INK, ms=6.5, label=f"{lr} nuisance")
          for lr in ["RF", "Lasso"]]
    outside_legend(ax, ncol=3, handles=h, fontsize=10.5, handletextpad=0.3, columnspacing=0.9)
    source_data(*src, w)


def draw_b(fig, ss, w):
    panel_b(fig.add_subplot(ss), w)


def bw_table():
    """Digester table restricted to the DML sample (ammonium-N measured, countries with >= 3 digesters)."""
    import importlib
    import sys
    sys.path.insert(0, str(RES.parent / "scripts"))
    dml = importlib.import_module("05_causal_dml")
    d = dml.digester_table()
    d = d[d.log_TAN.notna()]
    return d[d.groupby("Country").Country.transform("size") >= 3]


def bw_slopes(d):
    import statsmodels.api as sm
    rows = []
    for o in BW_OUT:
        x = d[["Country", "log_TAN", o]].dropna()
        cm = x.groupby("Country")[["log_TAN", o]].mean()
        wt = x.groupby("Country").size()
        b = sm.WLS(cm[o], sm.add_constant(cm.log_TAN), weights=wt).fit()
        xd = x.log_TAN - x.groupby("Country").log_TAN.transform("mean")
        yd = x[o] - x.groupby("Country")[o].transform("mean")
        wi = sm.OLS(yd.values, xd.values).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x.Country)[0]})
        rows.append(dict(outcome=o, n=len(x), countries=len(cm),
                         between=b.params.iloc[1], between_lo=b.conf_int().iloc[1, 0],
                         between_hi=b.conf_int().iloc[1, 1], between_p=b.pvalues.iloc[1],
                         within=wi.params[0], within_lo=wi.conf_int()[0, 0], within_hi=wi.conf_int()[0, 1],
                         within_p=wi.pvalues[0]))
    return pd.DataFrame(rows)


def draw_c(fig, ss, d, s):
    g = hgrid(ss, [1, 1], gap=0.6)
    axes = [fig.add_subplot(g[i]) for i in range(2)]
    for ax, o in zip(axes, ["CLR_Methanothrix", "CLR_Methanoculleus"]):
        x = d[["Country", "log_TAN", o]].dropna()
        ax.scatter(x.log_TAN, x[o], s=16, fc=BW_COL["digester"], ec="none", alpha=0.5, zorder=1)
        row = s.set_index("outcome").loc[o]
        for c, gg in x.groupby("Country"):
            xm, ym = gg.log_TAN.mean(), gg[o].mean()
            xr = np.array([gg.log_TAN.min(), gg.log_TAN.max()])
            ax.plot(xr, ym + row["within"] * (xr - xm), color=BW_COL["within"], lw=2.0, zorder=3)
        cm = x.groupby("Country")[["log_TAN", o]].mean()
        n = x.groupby("Country").size()
        xx = np.linspace(cm.log_TAN.min() - 0.1, cm.log_TAN.max() + 0.1, 20)
        xw = np.average(cm.log_TAN, weights=n); yw = np.average(cm[o], weights=n)
        ax.plot(xx, yw + row["between"] * (xx - xw), color=BW_COL["between"], lw=1.8, ls="--", zorder=4)
        ax.scatter(cm.log_TAN, cm[o], s=24 + 5 * n, fc=BW_COL["country"], ec=INK, lw=0.9, zorder=5)
        ax.text(0.97, 0.04, f"within {row['within']:+.2f}\nbetween {row['between']:+.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=11, color=INK, linespacing=1.35)
        style_axis(ax, xlabel="log$_{10}$ Ammonium N", ylabel="CLR abundance" if o == "CLR_Methanothrix" else None)
        numeric_ticks(ax, n=4)
        style_axis(ax)
        ax.text(0.04, 0.97, OLAB[o], transform=ax.transAxes, ha="left", va="top", fontsize=12, color=INK)
    h = [Line2D([], [], color=BW_COL["within"], lw=2.0, label="Within-country slope"),
         Line2D([], [], color=BW_COL["between"], lw=1.8, ls="--", label="Between-country slope"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=BW_COL["country"], mec=INK, label="Country mean"),
         Line2D([], [], marker="o", ls="", ms=5.5, mfc=BW_COL["digester"], mec="none", label="Digester")]
    outside_legend(axes[0], x=between_x(axes[0], axes[1]), ncol=4, handles=h, fontsize=10.5, handletextpad=0.3,
                   columnspacing=0.9)
    source_data(NAME, "c_digesters_TAN_outcomes.csv", d[["Country", "log_TAN"] + BW_OUT].reset_index())


def draw_d(fig, ss, s):
    ax = fig.add_subplot(ss)
    for i, r in enumerate(s.itertuples()):
        y = len(s) - 1 - i
        for key, dy, mk, col in [("between", 0.17, "s", BW_COL["between"]), ("within", -0.17, "o", BW_COL["within"])]:
            v, lo, hi = getattr(r, key), getattr(r, f"{key}_lo"), getattr(r, f"{key}_hi")
            ax.plot([lo, hi], [y + dy] * 2, color=col, lw=2.2, zorder=2)
            ax.scatter([v], [y + dy], marker=mk, s=56, fc=col, ec=INK, lw=0.8, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.9, ls=":")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([OLAB[o] for o in s.outcome[::-1]])
    ax.set_ylim(-0.6, len(s) - 0.4)
    style_axis(ax, xlabel="Slope per 10-fold Ammonium N (95% CI)")
    ax.set_xticks([-4, -2, 0, 2, 4])
    style_axis(ax)
    h = [Line2D([], [], marker="s", color=BW_COL["between"], lw=2.2, ms=7, mec=INK, label="Between countries"),
         Line2D([], [], marker="o", color=BW_COL["within"], lw=2.2, ms=7, mec=INK, label="Within countries")]
    outside_legend(ax, ncol=2, handles=h, fontsize=10.5, columnspacing=0.9)
    source_data(NAME, "d_between_within_slopes.csv", s)


def draw_e(fig, ss, r):
    ax = fig.add_subplot(ss)
    real = r[r.test == "real_demeaned"].set_index("outcome").theta
    rows = []
    for i, o in enumerate(OUT):
        v = r[(r.test == "placebo_within_demeaned") & (r.outcome == o)].theta.values
        vp = ax.violinplot([v], positions=[i], widths=0.75, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor(ROB_COL["placebo"]); b.set_edgecolor(INK); b.set_alpha(1); b.set_linewidth(0.8)
        p = (1 + np.sum(np.abs(v) >= abs(real[o]))) / (1 + len(v))
        ax.scatter(i, real[o], marker="D", s=70, color=ROB_COL["real"], edgecolor=INK, lw=0.8, zorder=4)
        ax.text(i, 3.45, f"{p:.3f}", ha="center", fontsize=10, color=INK if p < 0.05 else MUTED)
        rows.append(dict(outcome=o, real_demeaned=real[o], placebo_mean=v.mean(), placebo_sd=v.std(), p_empirical=p))
    ax.axhline(0, color=MUTED, lw=0.9, ls="--", zorder=0)
    ax.set_xticks(range(len(OUT)))
    ax.set_xticklabels([OLAB[o] for o in OUT], rotation=35, ha="right")
    ax.text(-0.75, 3.45, "P", ha="right", fontsize=10.5, fontstyle="italic")
    ax.set_ylim(-3.2, 3.9)
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    style_axis(ax, ylabel="θ (within-country)")
    style_axis(ax)
    ax.spines["left"].set_bounds(-3, 3)
    h = [Line2D([], [], marker="s", ls="", ms=10, mfc=ROB_COL["placebo"], mec=INK,
                label="Ammonium N permuted within country (200×)"),
         Line2D([], [], marker="D", ls="", ms=8, mfc=ROB_COL["real"], mec=INK, label="Observed")]
    ax.legend(handles=h, loc="lower left", fontsize=10.5, handletextpad=0.3, borderaxespad=0.05)
    source_data(NAME, "e_within_country_placebo.csv", pd.DataFrame(rows))


def draw_f(fig, ss, r):
    ax = fig.add_subplot(ss)
    real = r[r.test == "real_demeaned"].set_index("outcome").theta
    rng = np.random.default_rng(5)
    for i, o in enumerate(OUT[::-1]):
        s = r[(r.test == "subset80") & (r.outcome == o)].theta
        ax.plot([s.min(), s.max()], [i + 0.22] * 2, color=ROB_COL["subset"], lw=5, solid_capstyle="butt", zorder=2)
        rc = r[(r.test == "random_common_cause") & (r.outcome == o)].theta
        ax.scatter(rc, np.full(len(rc), i + 0.22), s=14, color=ROB_COL["rcc"], edgecolor=INK, lw=0.4, zorder=3)
        lc = r[(r.test == "leave_country_out") & (r.outcome == o)].theta
        ax.scatter(lc, i - 0.18 + rng.uniform(-0.08, 0.08, len(lc)), s=24, color=ROB_COL["loco"], edgecolor=INK,
                   lw=0.5, zorder=3)
        ax.scatter(real[o], i, marker="D", s=66, color=ROB_COL["real"], edgecolor=INK, lw=0.8, zorder=5)
    ax.axvline(0, color=MUTED, lw=1.0, ls="--", zorder=0)
    ax.set_yticks(range(len(OUT)))
    ax.set_yticklabels([OLAB[o] for o in OUT[::-1]])
    ax.set_ylim(-0.6, len(OUT) - 0.4)
    style_axis(ax, xlabel="θ (within-country)")
    numeric_ticks(ax, y=False, n=6)
    style_axis(ax)
    h = [Line2D([], [], color=ROB_COL["subset"], lw=5, label="80% subsets (range, 30×)"),
         Line2D([], [], marker="o", ls="", ms=5.5, mfc=ROB_COL["rcc"], mec=INK, label="Random common cause (10×)"),
         Line2D([], [], marker="o", ls="", ms=6.5, mfc=ROB_COL["loco"], mec=INK, label="Leave one country out"),
         Line2D([], [], marker="D", ls="", ms=7, mfc=ROB_COL["real"], mec=INK, label="Full data")]
    outside_legend(ax, ncol=2, handles=h, fontsize=10.5, handletextpad=0.3, columnspacing=0.9)
    src = r[r.test.isin(["subset80", "random_common_cause", "leave_country_out", "real_demeaned"])]
    source_data(NAME, "f_stability_demeaned.csv", src)


if __name__ == "__main__":
    setup()
    w = load()
    r = pd.read_csv(RES / "dml_refutation_parallel.csv")
    d = bw_table()
    s = bw_slopes(d)
    cf = Composite(NAME, "Fig6_within_country_DML", height=16.2)
    rows = cf.grid([1.2, 0.9, 1.05], gap=0.6)
    r1 = hgrid(rows[0], [1.0, 1.08], gap=0.62)
    cf.panel("a", r1[0], lambda f, ss: draw_a(f, ss, w, r))
    cf.panel("b", r1[1], lambda f, ss: draw_b(f, ss, w))
    r2 = hgrid(rows[1], [1.45, 1.0], gap=0.62)
    cf.panel("c", r2[0], lambda f, ss: draw_c(f, ss, d, s))
    cf.panel("d", r2[1], lambda f, ss: draw_d(f, ss, s))
    r3 = hgrid(rows[2], [1.1, 1.0], gap=0.62)
    cf.panel("e", r3[0], lambda f, ss: draw_e(f, ss, r))
    cf.panel("f", r3[1], lambda f, ss: draw_f(f, ss, r))
    rep = cf.finish()
    print({k["panel"]: round(k["max_residual_pt"], 2) for k in rep})
    print(s.round(3).to_string(index=False))
