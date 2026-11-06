"""Fig. 2  Cross-country ceiling: 20 algorithms, error decomposition, country offsets by run, stacking."""
import importlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.stats import spearmanr
from figstyle import (setup, style_axis, numeric_ticks, facet_tag, outside_legend, between_x, hgrid, vgrid,
                      Composite, source_data, dig_level, offset_decomposition, target_frame, RES, ROOT, C, INK,
                      MUTED, RULE, TARGETS, T_SHORT, T_COL, SCHEME_COL, DECOMP_COL, DIV_CMAP, PALETTE,
                      validate_series_colors, short)

sys.path.insert(0, str(ROOT / "scripts"))
runs = importlib.import_module("16_run_offsets")

NAME = "Fig2"
XMIN = -0.4
STACK_COL = validate_series_colors(PALETTE, {
    "ExtraTrees": C["gold"], "Average": C["green"], "SelectBest_iid": C["blue"], "Stack_iid": C["lav"],
    "SelectBest_shift": C["pink"], "Stack_shift": C["rose"]})
STACK_LAB = {"ExtraTrees": "ExtraTrees", "Average": "Mean of 10", "SelectBest_iid": "Select-best (i.i.d.)",
             "Stack_iid": "Stack (i.i.d.)", "SelectBest_shift": "Select-best (country)",
             "Stack_shift": "Stack (country)"}
SIGN_COL = validate_series_colors(PALETTE, {"under": C["rose"], "over": C["blue"], "digester": C["lav"]})
RUN_COL = [C["blue"], C["gold"], C["teal"], C["pink"], C["green"]]


def decomposition():
    o = pd.read_pickle(RES / "benchmark20_oof.pkl")
    tf = target_frame()
    rows, offs = [], []
    for t in TARGETS:
        for sch in ["GKF5", "LOCO"]:
            y, p, dd = dig_level(o[(t, sch, "ExtraTrees")], tf[t]["y"], tf[t]["dig"])
            c = pd.Series(tf[t]["cty"], index=tf[t]["dig"]).groupby(level=0).first().loc[dd].values
            rows.append(dict(target=t, scheme=sch, **offset_decomposition(y, p, c)))
            r = pd.DataFrame({"r": y - p, "c": c}).groupby("c").r.agg(["mean", "std", "size"])
            r["se"] = r["std"] / np.sqrt(r["size"])
            r["sd_target"] = y.std()
            offs.append(r.assign(target=t, scheme=sch).reset_index())
    return pd.DataFrame(rows), pd.concat(offs)


def draw_a(fig, ss):
    b = pd.read_csv(RES / "benchmark20_results.csv")
    piv = b.pivot_table(index="model", columns=["target", "scheme"], values="R2")
    order = piv.xs("LOCO", axis=1, level=1).clip(lower=XMIN).mean(1).sort_values().index
    g = hgrid(ss, [1] * 5, gap=0.36)
    axes = []
    for j, t in enumerate(TARGETS):
        ax = fig.add_subplot(g[j])
        gv, lv = piv[(t, "GKF5")].loc[order].values, piv[(t, "LOCO")].loc[order].values
        yy = np.arange(len(order))
        for i in yy:
            ax.plot([max(gv[i], XMIN), max(lv[i], XMIN)], [i, i], color=RULE, lw=2.0, zorder=1)
        for vals, sch in [(gv, "GKF5"), (lv, "LOCO")]:
            ok = vals >= XMIN
            ax.scatter(vals[ok], yy[ok], s=46, color=SCHEME_COL[sch], edgecolor=INK, lw=0.7, zorder=3)
            ax.scatter(np.full((~ok).sum(), XMIN), yy[~ok], s=52, marker="<", color=SCHEME_COL[sch],
                       edgecolor=INK, lw=0.7, zorder=3, clip_on=False)
        rho = spearmanr(gv, lv).correlation
        ax.axvline(0, color=MUTED, lw=0.9, ls=":", zorder=0)
        ax.set_xlim(XMIN - 0.04, 0.84)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_yticks(yy)
        ax.set_yticklabels(order if j == 0 else [])
        style_axis(ax, xlabel="R$^2$", tick_size=11.5)
        ax.set_xticks([-0.4, 0, 0.4, 0.8])
        style_axis(ax, tick_size=11.5)
        facet_tag(ax, f"{short(t)} (ρ = {rho:.2f})", x=0.5, y=1.01, ha="center", va="bottom", size=12)
        axes.append(ax)
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=SCHEME_COL["GKF5"], mec=INK,
                label="New digesters, seen countries (GKF5)"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=SCHEME_COL["LOCO"], mec=INK, label="New country (LOCO)")]
    outside_legend(axes[2], ncol=2, handles=h, handletextpad=0.3, columnspacing=1.6, pad_pt=24)
    src = piv.loc[order[::-1]]
    src.columns = [f"{t}|{s}" for t, s in src.columns]
    source_data(NAME, "a_benchmark20_R2.csv", src.reset_index())


def loco_frame(t):
    o = pd.read_pickle(RES / "benchmark20_oof.pkl")
    tf = target_frame()
    y, p, dd = dig_level(o[(t, "LOCO", "ExtraTrees")], tf[t]["y"], tf[t]["dig"])
    c = pd.Series(tf[t]["cty"], index=tf[t]["dig"]).groupby(level=0).first().loc[dd].values
    return pd.DataFrame({"digester": dd, "country": c, "observed": y, "predicted": p})


LABEL_SPOTS = {"Temperature": {"Argentina": (19, 50), "Switzerland": (51, 25), "Czech Republic": (51, 25)},
               "log_NH4": {"Argentina": (1.6, 3.95), "USA": (4.0, 2.2)}}


def draw_b(fig, ss):
    """Observed vs LOCO-predicted values; country means with their vertical distance to the 1:1 line."""
    g = hgrid(ss, [1, 1], gap=0.5)
    axes = [fig.add_subplot(g[i]) for i in range(2)]
    out = []
    for ax, t, lim in zip(axes, ["Temperature", "log_NH4"], [(10, 60), (1.0, 4.6)]):
        d = loco_frame(t)
        ax.scatter(d.observed, d.predicted, s=16, fc=SIGN_COL["digester"], ec="none", alpha=0.45, zorder=2)
        m = d.groupby("country").agg(obs=("observed", "mean"), pred=("predicted", "mean"), n=("observed", "size"))
        m = m[m.n >= 3]
        m["offset"] = m.obs - m.pred
        ax.plot(lim, lim, color=MUTED, lw=1.1, ls="--", zorder=1)
        for c, r in m.iterrows():
            col = SIGN_COL["under"] if r.offset > 0 else SIGN_COL["over"]
            ax.plot([r.obs, r.obs], [r.pred, r.obs], color=col, lw=2.2, zorder=3)
            ax.scatter([r.obs], [r.pred], s=22 + 4.5 * r.n, fc=col, ec=INK, lw=0.9, zorder=4)
        spots = LABEL_SPOTS[t]
        for c, r in m.reindex(m.offset.abs().sort_values(ascending=False).index[:2]).iterrows():
            ax.annotate(c, (r.obs, r.pred), xytext=spots[c], textcoords="data", fontsize=11, color=INK,
                        ha="center", arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=5))
        ax.set_xlim(lim); ax.set_ylim(lim)
        unit = "°C" if t == "Temperature" else "log$_{10}$ mg N L$^{-1}$"
        style_axis(ax, xlabel=f"Observed ({unit})", ylabel="LOCO prediction" if t == "Temperature" else None)
        numeric_ticks(ax, n=4)
        style_axis(ax)
        facet_tag(ax, short(t), x=0.04, y=0.97)
        out.append(m.reset_index().assign(target=t))
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=SIGN_COL["under"], mec=INK, label="Country under-predicted"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=SIGN_COL["over"], mec=INK, label="Country over-predicted"),
         Line2D([], [], marker="o", ls="", ms=6, mfc=SIGN_COL["digester"], mec="none", label="Digester")]
    outside_legend(axes[0], x=between_x(axes[0], axes[1]), ncol=3, handles=h, handletextpad=0.2,
                   columnspacing=0.9, fontsize=11)
    source_data(NAME, "b_country_means_LOCO.csv", pd.concat(out))


def draw_c(fig, ss, dec):
    ax = fig.add_subplot(ss)
    w = 0.36
    x = np.arange(len(TARGETS))
    for k, sch in enumerate(["GKF5", "LOCO"]):
        d = dec[dec.scheme == sch].set_index("target").loc[TARGETS]
        xx = x + (k - 0.5) * (w + 0.04)
        ax.bar(xx, d.within, w, color=DECOMP_COL["within"], edgecolor=INK, lw=0.9, hatch="" if sch == "GKF5" else "//")
        ax.bar(xx, d.offset, w, bottom=d.within, color=DECOMP_COL["offset"], edgecolor=INK, lw=0.9,
               hatch="" if sch == "GKF5" else "//")
    ax.set_xticks(x)
    ax.set_xticklabels([short(t) for t in TARGETS], rotation=30, ha="right")
    ax.set_ylim(0, 1.28)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax, ylabel="Error / total variance")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 1)
    h = [Patch(fc=DECOMP_COL["offset"], ec=INK, label="Country offset"),
         Patch(fc=DECOMP_COL["within"], ec=INK, label="Within country"),
         Patch(fc="white", ec=INK, label="GKF5"), Patch(fc="white", ec=INK, hatch="//", label="LOCO")]
    ax.legend(handles=h, loc="upper left", ncol=2, fontsize=11, handlelength=1.3, columnspacing=0.8,
              borderaxespad=0.05, handletextpad=0.4)
    source_data(NAME, "c_d_error_decomposition_ET.csv", dec)


def draw_d(fig, ss, dec):
    ax = fig.add_subplot(ss)
    for t in TARGETS:
        d = dec[dec.target == t].set_index("scheme").within_corr
        ax.plot([0, 1], [d["GKF5"], d["LOCO"]], color=T_COL[t], lw=2.4, marker="o", ms=8, mec=INK)
        nudge = {"log_NH4": 0.017, "Digester_age": -0.017}.get(t, 0)
        ax.text(1.16, d["LOCO"] + nudge, short(t), va="center", ha="left", fontsize=11.5, color=INK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["GKF5", "LOCO"])
    ax.set_xlim(-0.35, 2.4)
    ax.set_ylim(0.1, 0.8)
    style_axis(ax, ylabel="Within-country $r$")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    ax.spines["bottom"].set_bounds(-0.35, 1.35)


def draw_e(fig, ss, offs):
    """LOCO country offsets in SD units, rows grouped by the V1-3 run and labelled by the V3-5 run."""
    grp = runs.country_groups()
    o = offs[(offs.scheme == "LOCO") & (offs["size"] >= 3)].copy()
    o["z"] = o["mean"] / o["sd_target"]
    Z = o.pivot(index="c", columns="target", values="z").reindex(columns=TARGETS)
    v13 = {r: f"{i + 1}" for i, r in enumerate(sorted(grp.run_V13.unique()))}
    v35 = {r: "ABCDE"[i] for i, r in enumerate(sorted(grp.run_V35.unique()))}
    g = grp.loc[Z.index].assign(r13=lambda d: d.run_V13.map(v13), r35=lambda d: d.run_V35.map(v35))
    g = g.sort_values(["r13", "r35", "continent"])
    Z = Z.loc[g.index]
    gg = hgrid(ss, [0.42, 2.6, 0.12], gap=0.08)
    ax_s, ax, cax = fig.add_subplot(gg[0]), fig.add_subplot(gg[1]), fig.add_subplot(gg[2])
    n = len(Z)
    for i, (c, r) in enumerate(g.iterrows()):
        for j, key in enumerate(["r13", "r35"]):
            fc = RUN_COL[list(v13.values()).index(r[key])] if key == "r13" else "white"
            ax_s.add_patch(Rectangle((j - 0.45, i - 0.45), 0.9, 0.9, fc=fc, ec=INK, lw=0.6))
            ax_s.text(j, i, r[key], ha="center", va="center", fontsize=10.5, color=INK)
    ax_s.set_xlim(-0.55, 1.55); ax_s.set_ylim(n - 0.5, -0.5)
    ax_s.set_xticks([0, 1]); ax_s.set_xticklabels(["V1-3", "V3-5"], rotation=90)
    ax_s.set_yticks(range(n)); ax_s.set_yticklabels(Z.index)
    style_axis(ax_s, tick_size=11)
    ax_s.spines["left"].set_visible(False); ax_s.spines["bottom"].set_visible(False)
    ax_s.tick_params(axis="y", length=0)
    ax_s.tick_params(axis="x", length=0, top=True, bottom=False, labeltop=True, labelbottom=False)
    norm = TwoSlopeNorm(0, -1.2, 1.2)
    im = ax.imshow(Z.values, cmap=DIV_CMAP, norm=norm, aspect="auto")
    for i in range(n):
        for j in range(len(TARGETS)):
            v = Z.values[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec=RULE, lw=0.5, hatch="///"))
            else:
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9.5, color=INK)
    for i in range(1, n):
        if g.r13.iloc[i] != g.r13.iloc[i - 1]:
            for a in (ax, ax_s):
                a.axhline(i - 0.5, color=INK, lw=1.2)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([short(t) for t in TARGETS], rotation=30, ha="right")
    ax.set_yticks([])
    style_axis(ax, tick_size=11)
    ax.tick_params(axis="y", left=False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.9)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("LOCO offset / SD", fontsize=12)
    cb.ax.tick_params(labelsize=10.5, width=1.0, length=3.5)
    cb.set_ticks([-1.2, -0.6, 0, 0.6, 1.2])
    src = Z.copy()
    src.insert(0, "V13_run", g.run_V13); src.insert(1, "V35_run", g.run_V35); src.insert(2, "continent", g.continent)
    source_data(NAME, "e_country_offsets_by_run_LOCO_ET.csv", src.reset_index().rename(columns={"c": "country"}))


def draw_f(fig, ss):
    ax = fig.add_subplot(ss)
    s = pd.read_csv(RES / "stacking_LOCO_Temperature_pH_log_NH4_Digester_age.csv")
    tg = ["Temperature", "pH", "log_NH4", "Digester_age"]
    meth = list(STACK_COL)
    w = 0.13
    for k, m in enumerate(meth):
        v = s[s.method == m].set_index("target").loc[tg].R2.values
        ax.bar(np.arange(len(tg)) + (k - 2.5) * w, v, w, color=STACK_COL[m], edgecolor=INK, lw=0.8,
               label=STACK_LAB[m])
    ax.axhline(0, color=INK, lw=0.9)
    ax.set_xticks(np.arange(len(tg)))
    ax.set_xticklabels([short(t) for t in tg], rotation=30, ha="right")
    ax.set_ylim(-0.02, 0.78)
    style_axis(ax, ylabel="LOCO R$^2$")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    ax.legend(loc="upper right", ncol=2, fontsize=10, handlelength=0.9, columnspacing=0.6, borderaxespad=0.05,
              handletextpad=0.3)
    source_data(NAME, "g_stacking_LOCO_R2.csv", s[s.method.isin(meth)])


HOLD_COL = validate_series_colors(PALETTE, {"random": C["blue"], "run": C["rose"], "loco": C["gold"]})


def draw_g(fig, ss):
    """R2 when run-linked countries are held out together, against random country groups of the same sizes."""
    ax = fig.add_subplot(ss)
    h = pd.read_csv(RES / "run_holdout.csv").set_index("target").loc[TARGETS]
    y = np.arange(len(TARGETS))[::-1]
    for yi, (t, r) in zip(y, h.iterrows()):
        ax.plot([r.R2_random_min, r.R2_random_max], [yi, yi], color=HOLD_COL["random"], lw=6, alpha=0.75,
                solid_capstyle="butt", zorder=1)
        ax.plot([r.R2_random_mean] * 2, [yi - 0.22, yi + 0.22], color=INK, lw=1.8, zorder=2)
        ax.scatter([r.R2_LOCO], [yi], s=46, marker="o", color=HOLD_COL["loco"], edgecolor=INK, lw=0.7, zorder=3)
        ax.scatter([r.R2_run_holdout], [yi], s=62, marker="D", color=HOLD_COL["run"], edgecolor=INK, lw=0.8,
                   zorder=4)
    ax.axvline(0, color=MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([short(t) for t in TARGETS])
    ax.set_ylim(-0.6, len(TARGETS) - 0.4)
    ax.set_xlim(-0.25, 0.58)
    ax.set_xticks([-0.2, 0, 0.2, 0.4])
    style_axis(ax, xlabel="R$^2$")
    style_axis(ax)
    hh = [Line2D([], [], color=HOLD_COL["random"], lw=6, alpha=0.75, label="Random groups (range)"),
          Line2D([], [], color=INK, lw=1.8, label="Random groups (mean)"),
          Line2D([], [], marker="D", ls="", ms=7.5, mfc=HOLD_COL["run"], mec=INK, label="Run hold-out"),
          Line2D([], [], marker="o", ls="", ms=7, mfc=HOLD_COL["loco"], mec=INK, label="LOCO")]
    outside_legend(ax, ncol=2, handles=hh, fontsize=10.5, handlelength=1.4, columnspacing=0.7, handletextpad=0.4)
    source_data(NAME, "f_run_holdout.csv", h.reset_index())


if __name__ == "__main__":
    setup()
    dec, offs = decomposition()
    source_data(NAME, "d_country_offsets_ET.csv", offs.rename(columns={"c": "country"}))
    cf = Composite(NAME, "Fig2_cross_country_ceiling", height=15.6)
    rows = cf.grid([1.5, 0.92, 1.18], gap=0.6)
    cf.panel("a", rows[0], draw_a)
    r2 = hgrid(rows[1], [2.0, 1.2, 0.85], gap=0.62)
    cf.panel("b", r2[0], draw_b)
    cf.panel("c", r2[1], lambda f, s: draw_c(f, s, dec))
    cf.panel("d", r2[2], lambda f, s: draw_d(f, s, dec))
    r3 = hgrid(rows[2], [1.22, 0.9, 0.98], gap=0.62)
    cf.panel("e", r3[0], lambda f, s: draw_e(f, s, offs))
    cf.panel("f", r3[1], draw_g)
    cf.panel("g", r3[2], draw_f)
    rep = cf.finish()
    print({r["panel"]: round(r["max_residual_pt"], 2) for r in rep})
