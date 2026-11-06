"""Fig. 3  Where the cross-country error comes from: resolution, attribution, primer, feature-level remedies."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from figstyle import (setup, style_axis, numeric_ticks, facet_tag, outside_legend, hgrid, Composite, source_data, RES,
                      C, INK, MUTED, RULE, TARGETS, T_COL, DIV_CMAP, PALETTE, validate_series_colors, short)

NAME = "Fig3"
RANKS = ["p", "c", "o", "f", "g", "s", "ASV"]
RANK_LAB = ["Phylum", "Class", "Order", "Family", "Genus", "Species", "ASV"]
PRIMER_COL = validate_series_colors(PALETTE, {"V13->V13": C["lav"], "V4->V4": C["blue"], "V13->V4": C["rose"],
                                              "pool->V4": C["gold"]})
PRIMER_LAB = {"V13->V13": "V1-3 → V1-3", "V4->V4": "V4 → V4", "V13->V4": "V1-3 → V4", "pool->V4": "Pooled → V4"}
CMP_COL = validate_series_colors(PALETTE, {"all": C["blue"], "top20": C["rose"], "resolution": C["lav"]})
BAND = "#EEF0F4"
REMEDY = [("ET", "ET_bal", "ET: country-balanced weights"), ("ET", "ET_screen50", "ET: invariance screen (50)"),
          ("ET", "ET_screen100", "ET: invariance screen (100)"), ("ET", "ET_screen200", "ET: invariance screen (200)"),
          ("ET", "ET_center", "ET: test-country centring"),
          ("LGBM", "Anchor_g0", "LGBM: anchor γ = 0"), ("LGBM", "Anchor_g0.5", "LGBM: anchor γ = 0.5"),
          ("LGBM", "Anchor_g3", "LGBM: anchor γ = 3"), ("LGBM", "Anchor_g10", "LGBM: anchor γ = 10"),
          ("LGBM", "Anchor_g3_screen100", "LGBM: anchor γ = 3 + screen"),
          ("LGBM", "Anchor_g0_center", "LGBM: anchor γ = 0 + centring")]


def draw_a(fig, ss):
    ax = fig.add_subplot(ss)
    tr = pd.read_csv(RES / "taxonomic_resolution.csv")
    x = np.arange(len(RANKS))
    cid = tr[tr.target == "Country_ID"].set_index("rank").loc[RANKS].R2.values
    ax.axvspan(3.6, 4.4, color=BAND, lw=0, zorder=0)
    ax.plot(x, cid, color=CMP_COL["resolution"], lw=2.5, marker="s", ms=8, mec=INK, zorder=3)
    ax.set_ylim(0.65, 0.76)
    ax.set_xticks(x)
    ax.set_xticklabels(RANK_LAB, rotation=40, ha="right")
    style_axis(ax, ylabel="Country identification\n(balanced accuracy)")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    source_data(NAME, "a_b_taxonomic_resolution.csv", tr)


def draw_b(fig, ss):
    ax = fig.add_subplot(ss)
    tr = pd.read_csv(RES / "taxonomic_resolution.csv")
    x = np.arange(len(RANKS))
    ax.axvspan(3.6, 4.4, color=BAND, lw=0, zorder=0)
    lo = tr[(tr.scheme == "LOCO") & (tr.target != "Country_ID")]
    nud = {"Digester_age": 0.012, "log_NH4": -0.012}
    for t in ["Temperature", "pH", "log_NH4", "Digester_age"]:
        v = lo[lo.target == t].set_index("rank").loc[RANKS].R2.values
        ax.plot(x, v, color=T_COL[t], lw=2.4, marker="o", ms=7.5, mec=INK, zorder=3)
        ax.text(6.25, v[-1] + nud.get(t, 0), short(t), color=INK, fontsize=11.5, va="center")
    ax.set_xlim(-0.4, 7.9)
    ax.set_ylim(0, 0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(RANK_LAB, rotation=40, ha="right")
    style_axis(ax, ylabel="LOCO R$^2$")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    ax.spines["bottom"].set_bounds(-0.4, 6.4)


def draw_c(fig, ss):
    ax = fig.add_subplot(ss)
    s = pd.read_csv(RES / "shap_per_taxon.csv")
    g = pd.read_csv(RES / "shap_geographic_reliance.csv").set_index("target")
    eta = s[s.target == "Temperature"].eta2_country.values  # eta2 does not depend on the target
    vp = ax.violinplot([eta], positions=[-0.9], widths=0.9, showextrema=False)
    for b in vp["bodies"]:
        b.set_facecolor(CMP_COL["resolution"]); b.set_edgecolor(INK); b.set_alpha(1); b.set_linewidth(0.9)
    ax.plot([-1.15, -0.65], [eta.mean()] * 2, color=INK, lw=2.2, zorder=3)
    ax.axhline(eta.mean(), color=MUTED, ls="--", lw=1.2, zorder=0)
    for i, t in enumerate(TARGETS):
        ax.scatter(i + 0.4, g.loc[t, "GRI"], s=110, color=T_COL[t], edgecolor=INK, lw=0.9, zorder=4, marker="D")
        ax.scatter(i + 0.4, g.loc[t, "top20_eta2"], s=80, facecolor="white", edgecolor=INK, lw=1.2, zorder=4)
    ax.set_xticks([-0.9] + [i + 0.4 for i in range(len(TARGETS))])
    ax.set_xticklabels(["All taxa"] + [short(t) for t in TARGETS], rotation=40, ha="right")
    ax.set_xlim(-1.6, len(TARGETS) + 0.1)
    ax.set_ylim(0, 0.75)
    style_axis(ax, ylabel="Country η$^2$ of taxon")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    h = [Line2D([], [], marker="D", ls="", ms=8, mfc=C["rose"], mec=INK, label="|SHAP|-weighted (GRI)"),
         Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=INK, mew=1.2, label="Top-20 SHAP taxa"),
         Line2D([], [], ls="--", color=MUTED, lw=1.2, label="Unweighted mean")]
    ax.legend(handles=h, loc="upper right", fontsize=10.5, handletextpad=0.3, borderaxespad=0.05)
    source_data(NAME, "c_geographic_reliance.csv", g.reset_index())


def draw_d(fig, ss):
    g = hgrid(ss, [1, 1], gap=0.55)
    axes = [fig.add_subplot(g[i]) for i in range(2)]
    s = pd.read_csv(RES / "shap_per_taxon.csv")
    out = []
    for ax, t in zip(axes, ["Temperature", "log_NH4"]):
        d = s[s.target == t]
        top = d.shap_rank <= 20
        ax.scatter(d.eta2_country[~top], d.mean_abs_shap[~top], s=13, color=T_COL[t], alpha=0.55, lw=0)
        ax.scatter(d.eta2_country[top], d.mean_abs_shap[top], s=50, color=T_COL[t], edgecolor=INK, lw=0.8)
        ax.set_yscale("log")
        rho = spearmanr(d.eta2_country, d.mean_abs_shap).correlation
        facet_tag(ax, f"{short(t)}\nρ = {rho:.2f}", x=0.97, y=0.03, ha="right", va="bottom", size=11.5)
        ax.set_xlim(0, 0.85)
        style_axis(ax, xlabel="Country η$^2$ of taxon", ylabel="Mean |SHAP|" if t == "Temperature" else None)
        ax.set_xticks([0, 0.4, 0.8])
        style_axis(ax)
        out.append(d)
    source_data(NAME, "d_shap_vs_eta2.csv", pd.concat(out))


def draw_e(fig, ss):
    g = hgrid(ss, [1, 1], gap=0.95)
    axes = [fig.add_subplot(g[i]) for i in range(2)]
    s = pd.read_csv(RES / "shap_per_taxon.csv")
    rows = []
    for t in TARGETS:
        d = s[s.target == t]
        top = d[d.shap_rank <= 20]
        rows.append(dict(target=t, all_abs_rho=d.within_rho.abs().mean(), top20_abs_rho=top.within_rho.abs().mean(),
                         all_sign=d.sign_consist.mean(), top20_sign=top.sign_consist.mean()))
    r = pd.DataFrame(rows)
    x = np.arange(len(TARGETS))
    for ax, (ca, ct, lab, lim) in zip(axes, [("all_abs_rho", "top20_abs_rho", "Within-country |ρ|", (0, 0.6)),
                                             ("all_sign", "top20_sign", "Sign consistency", (0.4, 1.0))]):
        for i in x:
            ax.plot([i, i], [r[ca][i], r[ct][i]], color=RULE, lw=2.0, zorder=1)
        ax.scatter(x, r[ca], s=64, color=CMP_COL["all"], edgecolor=INK, zorder=3, label="All taxa")
        ax.scatter(x, r[ct], s=64, color=CMP_COL["top20"], edgecolor=INK, zorder=3, label="Top-20 SHAP")
        ax.set_xticks(x)
        ax.set_xticklabels([short(t) for t in TARGETS], rotation=40, ha="right")
        ax.set_ylim(*lim)
        ax.set_xlim(-0.5, len(TARGETS) - 0.5)
        style_axis(ax, ylabel=lab)
        numeric_ticks(ax, x=False, n=4)
        style_axis(ax)
    axes[0].legend(loc="upper right", fontsize=10.5, handletextpad=0.2, borderaxespad=0.05)
    source_data(NAME, "e_within_country_consistency.csv", r)


def draw_f(fig, ss):
    ax = fig.add_subplot(ss)
    c = pd.read_csv(RES / "cross_primer_LOCO.csv")
    tg = ["Temperature", "pH", "log_NH4"]
    w = 0.19
    for k, m in enumerate(PRIMER_COL):
        v = c[c.scenario == m].set_index("target").loc[tg].R2.values
        ax.bar(np.arange(len(tg)) + (k - 1.5) * w, v, w, color=PRIMER_COL[m], edgecolor=INK, lw=0.8,
               label=PRIMER_LAB[m])
    ax.set_xticks(np.arange(len(tg)))
    ax.set_xticklabels([short(t) for t in tg])
    ax.set_ylim(0, 0.62)
    style_axis(ax, ylabel="LOCO R$^2$ (V4 test)")
    numeric_ticks(ax, x=False, n=4)
    style_axis(ax)
    ax.legend(loc="upper right", ncol=2, fontsize=10.5, handlelength=1.0, columnspacing=0.7, borderaxespad=0.05,
              handletextpad=0.4)
    source_data(NAME, "f_cross_primer_LOCO.csv", c)


def draw_g(fig, ss):
    g = hgrid(ss, [1, 0.045], gap=0.12)
    ax, cax = fig.add_subplot(g[0]), fig.add_subplot(g[1])
    iv = pd.read_csv(RES / "invariance_prototype_LOCO.csv").pivot_table(index="method", columns="target", values="R2")
    D = np.array([[iv.loc[m, t] - iv.loc[ref, t] for t in TARGETS] for ref, m, _ in REMEDY])
    im = ax.imshow(D, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -0.3, 0.3), aspect="auto")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=10, color=INK)
    ax.set_xticks(range(len(TARGETS)))
    ax.set_xticklabels([short(t) for t in TARGETS], rotation=30, ha="right")
    ax.set_yticks(range(len(REMEDY)))
    ax.set_yticklabels([lab for _, _, lab in REMEDY])
    ax.axhline(4.5, color=INK, lw=1.2)
    style_axis(ax, tick_size=11)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.9)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ΔLOCO R$^2$ vs base model", fontsize=12)
    cb.ax.tick_params(labelsize=10.5, width=1.0, length=3.5)
    src = pd.DataFrame(D, columns=TARGETS)
    src.insert(0, "method", [m for _, m, _ in REMEDY]); src.insert(1, "base", [r for r, _, _ in REMEDY])
    source_data(NAME, "g_feature_level_remedies_dR2.csv", src)


if __name__ == "__main__":
    setup()
    cf = Composite(NAME, "Fig3_error_source", height=12.8)
    rows = cf.grid([1, 1, 1.15], gap=0.55)
    r1 = hgrid(rows[0], [0.8, 1.0, 1.25], gap=0.62)
    cf.panel("a", r1[0], draw_a)
    cf.panel("b", r1[1], draw_b)
    cf.panel("c", r1[2], draw_c)
    r2 = hgrid(rows[1], [1.9, 1.6], gap=0.62)
    cf.panel("d", r2[0], draw_d)
    cf.panel("e", r2[1], draw_e)
    r3 = hgrid(rows[2], [0.85, 1.35], gap=0.62)
    cf.panel("f", r3[0], draw_f)
    cf.panel("g", r3[1], draw_g)
    rep = cf.finish()
    print({r["panel"]: round(r["max_residual_pt"], 2) for r in rep})
