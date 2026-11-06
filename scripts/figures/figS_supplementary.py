"""Supplementary figures S1-S8.  Run all, or pass figure numbers: python figS_supplementary.py 1 3 5"""
import importlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm
from figstyle import (setup, style_axis, numeric_ticks, facet_tag, outside_legend, between_x, hgrid, Composite,
                      source_data, dig_level, target_frame, RES, ROOT, C, INK, MUTED, TARGETS, T_COL, SCHEME_COL,
                      PALETTE, validate_series_colors, short)

sys.path.insert(0, str(ROOT / "scripts"))
TG4 = ["Temperature", "log_NH4", "Digester_age", "pH"]
AMP_COL = validate_series_colors(PALETTE, {"V13": C["rose"], "V35": C["teal"], "V4": C["blue"]})
PAIR_COL = validate_series_colors(PALETTE, {"rep": C["rose"], "within": C["gold"], "between": C["blue"]})
UNIT = {"Temperature": "°C", "pH": "pH", "log_NH4": "log$_{10}$", "log_SRT": "log$_{10}$", "Digester_age": "years"}


# ---------------------------------------------------------------- S1
def fig_s1():
    """Sequencing depth and replicate concordance (QC; digester as the analysis unit)."""
    from common import load_counts, amp_features, MIN_DEPTH
    name = "FigS1"
    dep = {a: load_counts(a, "g").sum(1) for a in ["V13", "V35", "V4"]}
    X, _, _ = amp_features("V13")
    meta = pd.read_csv(ROOT / "processed" / "metadata_clean.csv").set_index("SampleID").loc[X.index]
    Z = X.values
    Z = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    dig, cty = meta.Digester.values, meta.Country.values
    rng = np.random.default_rng(0)
    pairs = {"rep": [], "within": [], "between": []}
    for d in np.unique(dig):
        i = np.where(dig == d)[0]
        if len(i) == 2:
            pairs["rep"].append(np.mean(Z[i[0]] * Z[i[1]]))
    n = len(Z)
    while min(len(pairs["within"]), len(pairs["between"])) < 2000:
        i, j = rng.integers(0, n, 2)
        if dig[i] == dig[j]:
            continue
        k = "within" if cty[i] == cty[j] else "between"
        if len(pairs[k]) < 2000:
            pairs[k].append(np.mean(Z[i] * Z[j]))

    def draw_a(fig, ss):
        ax = fig.add_subplot(ss)
        bins = np.linspace(0.5, 5.5, 45)
        for a, v in dep.items():
            ax.hist(np.log10(v.clip(lower=1)), bins=bins, histtype="stepfilled", alpha=0.55, color=AMP_COL[a],
                    edgecolor=INK, lw=0.8, label={"V13": "V1-3", "V35": "V3-5", "V4": "V4"}[a])
        ax.axvline(np.log10(MIN_DEPTH), color=INK, lw=1.2, ls="--")
        style_axis(ax, xlabel="log$_{10}$ genus-assigned reads", ylabel="Samples")
        numeric_ticks(ax, n=5)
        style_axis(ax)
        ax.legend(loc="upper left", fontsize=11)

    def draw_b(fig, ss):
        ax = fig.add_subplot(ss)
        lab = {"rep": "A/B replicates", "within": "Other digester,\nsame country",
               "between": "Other digester,\nother country"}
        for i, k in enumerate(pairs):
            vp = ax.violinplot([pairs[k]], positions=[i], widths=0.75, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(PAIR_COL[k]); b.set_edgecolor(INK); b.set_alpha(1); b.set_linewidth(0.8)
            ax.plot([i - 0.22, i + 0.22], [np.median(pairs[k])] * 2, color=INK, lw=2.2)
        ax.set_xticks(range(3))
        ax.set_xticklabels([lab[k] for k in pairs])
        ax.set_ylim(-0.35, 1.03)
        style_axis(ax, ylabel="Pearson r (genus CLR, V1-3)", tick_size=11.5)
        ax.set_yticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
        style_axis(ax, tick_size=11.5)

    cf = Composite(name, "FigS1_depth_replicates", height=4.6)
    r = hgrid(cf.grid([1])[0], [1, 1.1], gap=0.7)
    cf.panel("a", r[0], draw_a)
    cf.panel("b", r[1], draw_b)
    cf.finish()
    source_data(name, "a_depth.csv", pd.concat([pd.DataFrame({"amplicon": a, "sample": v.index, "reads": v.values})
                                                for a, v in dep.items()]))
    source_data(name, "b_pair_correlations.csv",
                pd.concat([pd.DataFrame({"pair": k, "r": v}) for k, v in pairs.items()]))
    print({k: (len(v), round(float(np.median(v)), 3)) for k, v in pairs.items()})


# ---------------------------------------------------------------- S2
def fig_s2():
    """Country offsets of the ExtraTrees LOCO predictions for all five targets."""
    name = "FigS2"
    o = pd.read_pickle(RES / "benchmark20_oof.pkl")
    tf = target_frame()
    rows = []
    offs = {}
    for t in TARGETS:
        per = {}
        for sch in ["GKF5", "LOCO"]:
            y, p, dd = dig_level(o[(t, sch, "ExtraTrees")], tf[t]["y"], tf[t]["dig"])
            c = pd.Series(tf[t]["cty"], index=tf[t]["dig"]).groupby(level=0).first().loc[dd].values
            r = pd.DataFrame({"r": y - p, "c": c}).groupby("c").r.agg(["mean", "std", "size"])
            r = r[r["size"] >= 3]
            r["se"] = r["std"] / np.sqrt(r["size"])
            per[sch] = r
        order = per["LOCO"].sort_values("mean").index
        offs[t] = (per, order)
        for s in ["GKF5", "LOCO"]:
            rows.append(per[s].loc[order].assign(target=t, scheme=s).reset_index())

    def draw(t, legend=False):
        def f(fig, ss):
            ax = fig.add_subplot(ss)
            per, order = offs[t]
            for s, d in [("GKF5", 0.18), ("LOCO", -0.18)]:
                rr = per[s].loc[order]
                ax.errorbar(rr["mean"], np.arange(len(order)) + d, xerr=1.96 * rr.se, fmt="o", ms=5.5,
                            color=SCHEME_COL[s], mec=INK, mew=0.6, elinewidth=1.5)
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order)
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.axvline(0, color=MUTED, lw=0.9, ls=":")
            style_axis(ax, xlabel=f"Offset ({UNIT[t]})", tick_size=10.5)
            numeric_ticks(ax, y=False, n=4)
            style_axis(ax, tick_size=10.5)
            facet_tag(ax, short(t), x=0.03, y=0.99, size=12)
            if legend:
                h = [Line2D([], [], marker="o", ls="", ms=7, mfc=SCHEME_COL[s], mec=INK, label=s)
                     for s in ["GKF5", "LOCO"]]
                outside_legend(ax, ncol=2, handles=h)
        return f

    cf = Composite(name, "FigS2_country_offsets", height=9.6)
    gr = cf.grid([1.06, 1], gap=0.5)
    r1, r2 = hgrid(gr[0], [1] * 3, gap=0.5), hgrid(gr[1], [1] * 2, gap=0.5)
    for i, (t, l) in enumerate(zip(TARGETS, "abcde")):
        cf.panel(l, (r1[i] if i < 3 else r2[i - 3]), draw(t, legend=(i == 1)))
    cf.finish()
    source_data(name, "country_offsets_all_targets.csv", pd.concat(rows))


# ---------------------------------------------------------------- S3
def fig_s3():
    """Stacking weights per held-out country (i.i.d. vs country-blocked inner folds)."""
    name = "FigS3"
    w = pd.read_csv(RES / "stacking_LOCO_weights_Temperature_pH_log_NH4_Digester_age.csv")
    bases = ["Ridge", "KernelRidge", "SVR", "GPR", "RF", "ExtraTrees", "HistGB", "XGBoost", "LightGBM", "CatBoost"]
    col = validate_series_colors(PALETTE, {"iid": C["blue"], "shift": C["rose"]})

    def draw(t, legend=False):
        def f(fig, ss):
            ax = fig.add_subplot(ss)
            for j, s in enumerate(["iid", "shift"]):
                d = w[(w.target == t) & (w.scheme == s)][bases]
                pos = np.arange(len(bases)) + (j - 0.5) * 0.36
                bp = ax.boxplot([d[b].values for b in bases], positions=pos, widths=0.3, patch_artist=True,
                                showfliers=False, medianprops=dict(color=INK, lw=1.4))
                for b in bp["boxes"]:
                    b.set_facecolor(col[s]); b.set_edgecolor(INK)
                ax.scatter(np.repeat(pos, len(d)), d.values.T.ravel(), s=8, color=col[s], edgecolor=INK, alpha=0.7, zorder=3, lw=0.25)
            ax.set_xticks(range(len(bases)))
            ax.set_xticklabels(bases, rotation=40, ha="right")
            ax.set_ylim(-0.03, 1.0)
            style_axis(ax, ylabel="Weight", tick_size=11)
            numeric_ticks(ax, x=False, n=5)
            style_axis(ax, tick_size=11)
            facet_tag(ax, short(t), x=0.99, y=0.97, ha="right")
            if legend:
                h = [Patch(fc=col["iid"], ec=INK, label="Weights from inner i.i.d. folds"),
                     Patch(fc=col["shift"], ec=INK, label="Weights from inner leave-one-country-out")]
                outside_legend(ax, ncol=2, handles=h, fontsize=11)
        return f

    cf = Composite(name, "FigS3_stacking_weights", height=8.6)
    rows = cf.grid([1.1, 1], gap=0.5)
    for i, (t, l) in enumerate(zip(["Temperature", "pH", "log_NH4", "Digester_age"], "abcd")):
        r = hgrid(rows[i // 2], [1, 1], gap=0.62)
        cf.panel(l, r[i % 2], draw(t, legend=(i == 0)))
    cf.finish()
    source_data(name, "stacking_weights_per_country.csv", w)


# ---------------------------------------------------------------- S4
def fig_s4():
    """EB ablation over k and EB with other base learners."""
    name = "FigS4"
    a = pd.read_csv(RES / "eb_moment_ablation.csv")
    f = pd.read_csv(RES / "fewshot_recalibration_Temperature_pH_log_NH4_log_SRT_Digester_age.csv")
    mcol = validate_series_colors(PALETTE, {"raw": C["blue"], "none": C["gold"], "iid": C["lav"], "blocked": C["mauve"]})
    mlab = {"raw": "No recalibration", "none": "Unshrunk", "iid": "EB, inner i.i.d. moments",
            "blocked": "EB, inner-LOCO moments"}
    bcol = validate_series_colors(PALETTE, {"ET": C["rose"], "LGBM": C["blue"], "Anchor_g0.5": C["teal"],
                                            "Anchor_g0": C["gold"]})
    blab = {"ET": "ExtraTrees", "LGBM": "LightGBM", "Anchor_g0.5": "Anchor LGBM, γ = 0.5",
            "Anchor_g0": "Anchor LGBM, γ = 0"}
    s = a.groupby(["target", "moments", "k"]).R2.mean().reset_index()
    g = f.groupby(["target", "method", "k"]).R2.mean().reset_index()

    def draw_a(fig, ss):
        gg = hgrid(ss, [1] * 4, gap=0.55)
        axes = [fig.add_subplot(gg[i]) for i in range(4)]
        for ax, t in zip(axes, TG4):
            for m in mcol:
                d = s[(s.target == t) & (s.moments == m)]
                ax.plot(d.k, d.R2, color=mcol[m], lw=2.2, marker="o", ms=6, mec=INK, mew=0.6, label=mlab[m])
            ax.set_xticks([0, 1, 2, 3, 5])
            style_axis(ax, xlabel="Labelled digesters, k", ylabel="LOCO R$^2$" if t == TG4[0] else None)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax)
            facet_tag(ax, short(t), x=0.97, y=0.04, ha="right", va="bottom")
        outside_legend(axes[1], x=between_x(axes[1], axes[2]), ncol=4)

    def draw_b(fig, ss):
        gg = hgrid(ss, [1] * 5, gap=0.5)
        axes = [fig.add_subplot(gg[i]) for i in range(5)]
        for ax, t in zip(axes, ["Temperature", "log_NH4", "Digester_age", "pH", "log_SRT"]):
            for b in bcol:
                for kind, ls in [("EB", "-"), ("raw", ":")]:
                    d = g[(g.target == t) & (g.method == f"{b}|{kind}")]
                    ax.plot(d.k, d.R2, color=bcol[b], ls=ls, lw=2.1 if kind == "EB" else 1.5,
                            marker="o" if kind == "EB" else None, ms=5, mec=INK, mew=0.5)
            ax.set_xticks([0, 1, 2, 3, 5])
            style_axis(ax, xlabel="Labelled digesters, k", ylabel="LOCO R$^2$" if t == "Temperature" else None)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax)
            facet_tag(ax, short(t), x=0.97, y=0.04, ha="right", va="bottom")
        h = [Line2D([], [], color=bcol[b], lw=2.1, label=blab[b]) for b in bcol]
        h += [Line2D([], [], color=INK, lw=2.1, marker="o", ms=5, label="EB"),
              Line2D([], [], color=INK, lw=1.5, ls=":", label="No recalibration")]
        outside_legend(axes[2], ncol=6, handles=h, fontsize=11)

    cf = Composite(name, "FigS4_EB_ablation_learners", height=8.4)
    rows = cf.grid([1, 1], gap=0.6)
    cf.panel("a", rows[0], draw_a)
    cf.panel("b", rows[1], draw_b)
    cf.finish()
    source_data(name, "a_moment_ablation_mean.csv", s)
    source_data(name, "b_base_learners_mean.csv", g)


# ---------------------------------------------------------------- S5
def fig_s5():
    """Interval coverage and width at alpha = 0.2, and country coverage at k = 3."""
    name = "FigS5"
    f5 = importlib.import_module("fig5_intervals_alarms")
    c = pd.read_csv(RES / "fewshot_conformal_LOCO.csv")
    s = f5.pooled(c[c.alpha == 0.2])
    s = s[s.rule.isin(f5.R_COL)]
    d = c[(c.alpha == 0.1) & (c.k == 3) & c.rule.isin(["blocked_JP", "EB_matched_JP"])]
    d = d.groupby(["target", "rule", "country"]).agg(covered=("covered", "sum"), n=("n", "sum")).reset_index()
    d["cov"] = d.covered / d.n

    def draw_a(fig, ss):
        gg = hgrid(ss, [1] * 4, gap=0.62)
        axes = [fig.add_subplot(gg[i]) for i in range(4)]
        for ax, t in zip(axes, TG4):
            for r in f5.R_COL:
                dd = s[(s.target == t) & (s.rule == r)].sort_values("k")
                ax.plot(dd.k, dd["cov"], **f5.kw(r))
            ax.axhline(0.8, color=MUTED, lw=1.0)
            ax.set_ylim(0.55, 1.0)
            ax.set_xticks([0, 1, 2, 3, 5])
            style_axis(ax, xlabel="Labelled digesters, k", ylabel="Coverage (α = 0.2)" if t == TG4[0] else None)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax)
            facet_tag(ax, short(t), x=0.97, y=0.04, ha="right", va="bottom")
        h = [Line2D([], [], label=f5.R_LAB[r], **f5.kw(r)) for r in f5.R_COL]
        outside_legend(axes[1], x=between_x(axes[1], axes[2]), ncol=5, handles=h, handlelength=2.4,
                       columnspacing=1.2)

    def draw_b(fig, ss):
        gg = hgrid(ss, [1] * 4, gap=0.62)
        axes = [fig.add_subplot(gg[i]) for i in range(4)]
        for ax, t in zip(axes, TG4):
            st = s[s.target == t]
            top = 1.1 * st.w[np.isfinite(st.w)].max()
            for r in f5.R_COL:
                dd = s[(s.target == t) & (s.rule == r)].sort_values("k")
                fin = np.isfinite(dd.w)
                ax.plot(dd.k[fin], dd.w[fin], **f5.kw(r))
                for kk in dd.k[~fin]:
                    ax.plot([kk], [top], marker=f5.R_MK[r], ls="", ms=6.5, mfc=f5.R_COL[r], mec=INK, mew=0.6,
                            clip_on=False)
                    ax.text(kk, top * 1.015, "∞", ha="center", va="bottom", fontsize=12, color=INK)
            ax.set_ylim(0, top)
            ax.set_xticks([0, 1, 2, 3, 5])
            style_axis(ax, xlabel="Labelled digesters, k", ylabel="Interval width" if t == TG4[0] else None)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax)
            facet_tag(ax, f"{short(t)} ({f5.W_UNIT[t]})", x=0.97, y=0.04, ha="right", va="bottom")

    def draw_c(fig, ss):
        ax = fig.add_subplot(ss)
        rng = np.random.default_rng(6)
        for i, t in enumerate(TG4):
            for j, r in enumerate(["blocked_JP", "EB_matched_JP"]):
                v = d[(d.target == t) & (d.rule == r)]["cov"].values
                x = i + (j - 0.5) * 0.35
                ax.scatter(x + rng.uniform(-0.08, 0.08, len(v)), v, s=32, color=f5.R_COL[r], marker=f5.R_MK[r],
                           edgecolor=INK, lw=0.6, zorder=3)
                ax.plot([x - 0.13, x + 0.13], [np.median(v)] * 2, color=INK, lw=2.0, zorder=4)
        ax.axhline(0.9, color=MUTED, lw=1.0, zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels([short(t) for t in TG4])
        ax.set_ylim(0.6, 1.03)
        style_axis(ax, ylabel="Country coverage\n(α = 0.1, k = 3)")
        numeric_ticks(ax, x=False, n=5)
        style_axis(ax)
        ax.legend(handles=[Line2D([], [], marker=f5.R_MK[r], ls="", ms=7, mfc=f5.R_COL[r], mec=INK,
                                  label=f5.R_LAB[r]) for r in ["blocked_JP", "EB_matched_JP"]],
                  loc="lower left", fontsize=11)

    cf = Composite(name, "FigS5_intervals_alpha02", height=10.4)
    rows = cf.grid([1.05, 0.95, 0.95], gap=0.55)
    cf.panel("a", rows[0], draw_a)
    cf.panel("b", rows[1], draw_b)
    cf.panel("c", rows[2], draw_c)
    cf.finish()
    source_data(name, "a_b_coverage_width_alpha0.2.csv", s)
    source_data(name, "c_country_coverage_k3_jackknife.csv", d)


# ---------------------------------------------------------------- S6
def fig_s6():
    """DML sensitivity: free ammonia (FAN) as treatment, and agreement with TAN."""
    name = "FigS6"
    f6 = importlib.import_module("fig6_dml")
    w = pd.read_csv(RES / "dml_within_ate.csv")
    fan = w[w.treatment == "log_FAN"].copy()
    fan["p"] = 2 * norm.sf(np.abs(fan.z))
    fan["q"] = fan.groupby(["variant", "learner"]).p.transform(f6.bh)
    zz = w[w.variant == "demeaned"].pivot_table(index=["outcome", "learner"], columns="treatment",
                                                 values="z").reset_index()

    def draw_a(fig, ss):
        f6.panel_b(fig.add_subplot(ss), fan, xlim=(-4.5, 6.2), treat="FAN proxy", src=(name, "a_dml_FAN.csv"))

    def draw_b(fig, ss):
        ax = fig.add_subplot(ss)
        for lrn in ["LGBM", "RF", "Lasso"]:
            x = zz[zz.learner == lrn]
            ax.scatter(x.log_TAN, x.log_FAN, marker=f6.L_MK[lrn], s=60, color=f6.V_COL["demeaned"], edgecolor=INK,
                       lw=0.7, label=lrn)
        lim = [-6, 7.5]
        ax.plot(lim, lim, color=MUTED, lw=1.1, ls="--")
        for v in (-1.96, 1.96):
            ax.axhline(v, color=MUTED, lw=0.8, ls=":"); ax.axvline(v, color=MUTED, lw=0.8, ls=":")
        ax.set_xlim(lim); ax.set_ylim(lim)
        style_axis(ax, xlabel="z, Ammonium N (within-country)", ylabel="z, FAN proxy (within-country)")
        numeric_ticks(ax, n=5)
        style_axis(ax)
        ax.legend(loc="upper left", fontsize=11, title="Nuisance", title_fontsize=11)

    cf = Composite(name, "FigS6_DML_FAN", height=6.6)
    r = hgrid(cf.grid([1])[0], [1.3, 0.85], gap=0.7)
    cf.panel("a", r[0], draw_a)
    cf.panel("b", r[1], draw_b)
    cf.finish()
    source_data(name, "b_TAN_vs_FAN_z.csv", zz)


# ---------------------------------------------------------------- S7
def fig_s7():
    """Cross-primer concordance and screening scenarios."""
    name = "FigS7"
    cc = pd.read_csv(RES / "primer_concordance_V13_V4.csv", index_col=0)
    c = pd.read_csv(RES / "cross_primer_LOCO.csv")
    sc = ["V13->V13", "V4->V4", "V13->V4", "pool->V4", "V13->V4_screen", "pool->V4_screen"]
    col = validate_series_colors(PALETTE, {"V13->V13": C["lav"], "V4->V4": C["blue"], "V13->V4": C["rose"],
                                           "pool->V4": C["gold"], "V13->V4_screen": C["pink"],
                                           "pool->V4_screen": C["green"]})
    lab = {"V13->V13": "V1-3 → V1-3", "V4->V4": "V4 → V4", "V13->V4": "V1-3 → V4", "pool->V4": "Pooled → V4",
           "V13->V4_screen": "V1-3 → V4, CCC screen", "pool->V4_screen": "Pooled → V4, CCC screen"}

    def draw_a(fig, ss):
        ax = fig.add_subplot(ss)
        ax.hist(cc.CCC.dropna(), bins=np.linspace(0, 1, 31), color=C["blue"], edgecolor=INK, lw=0.7)
        ax.axvline(cc.CCC.median(), color=INK, lw=1.6, ls="--")
        ax.text(cc.CCC.median() - 0.02, 0.92, f"median {cc.CCC.median():.2f}", ha="right", fontsize=11, color=INK,
                transform=ax.get_xaxis_transform())
        style_axis(ax, xlabel="Lin's CCC, V1-3 vs V4 genus CLR", ylabel="Shared genera")
        numeric_ticks(ax, n=5)
        style_axis(ax)

    def draw_b(fig, ss):
        ax = fig.add_subplot(ss)
        tg = ["Temperature", "pH", "log_NH4"]
        wd = 0.13
        for k, m in enumerate(sc):
            v = c[c.scenario == m].set_index("target").loc[tg].R2.values
            ax.bar(np.arange(3) + (k - 2.5) * wd, v, wd, color=col[m], edgecolor=INK, lw=0.7, label=lab[m])
        ax.set_xticks(range(3)); ax.set_xticklabels([short(t) for t in tg])
        ax.set_ylim(0, 0.62)
        style_axis(ax, ylabel="LOCO R$^2$ (V4 test)")
        numeric_ticks(ax, x=False, n=4)
        style_axis(ax)
        ax.legend(loc="upper right", ncol=2, fontsize=10.5, handlelength=1.0, columnspacing=0.7)

    cf = Composite(name, "FigS7_cross_primer", height=4.8)
    r = hgrid(cf.grid([1])[0], [0.8, 1.2], gap=0.7)
    cf.panel("a", r[0], draw_a)
    cf.panel("b", r[1], draw_b)
    cf.finish()
    source_data(name, "a_CCC.csv", cc.reset_index())
    source_data(name, "b_cross_primer.csv", c)


# ---------------------------------------------------------------- S8
def fig_s8():
    """Taxonomic resolution (GKF5 vs LOCO) and classifier metrics (GKF5 vs LOCO)."""
    name = "FigS8"
    tr = pd.read_csv(RES / "taxonomic_resolution.csv")
    cl = pd.read_csv(RES / "classification_LOCO.csv")
    ranks = ["p", "c", "o", "f", "g", "s", "ASV"]
    rl = ["Phylum", "Class", "Order", "Family", "Genus", "Species", "ASV"]
    f5 = importlib.import_module("fig5_intervals_alarms")
    ccol, clab = f5.C_COL, {**f5.C_LAB, "NonSludge": "Non-sludge"}
    T4 = ["Temperature", "pH", "log_NH4", "Digester_age"]

    def draw_res(metric, ylab, legend=False):
        def f(fig, ss):
            ax = fig.add_subplot(ss)
            for t in T4:
                for sch, ls in [("GKF5", ":"), ("LOCO", "-")]:
                    v = tr[(tr.target == t) & (tr.scheme == sch)].set_index("rank").loc[ranks][metric].values
                    ax.plot(range(7), v, color=T_COL[t], ls=ls, lw=2.0, marker="o" if sch == "LOCO" else None,
                            ms=5, mec=INK, mew=0.5)
            ax.set_xticks(range(7)); ax.set_xticklabels(rl, rotation=40, ha="right")
            style_axis(ax, ylabel=ylab, tick_size=11.5)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax, tick_size=11.5)
            if legend:
                h = [Line2D([], [], color=T_COL[t], lw=2.1, label=short(t)) for t in T4]
                h += [Line2D([], [], color=INK, lw=1.7, ls=":", label="GKF5"),
                      Line2D([], [], color=INK, lw=1.7, marker="o", ms=5, label="LOCO")]
                outside_legend(ax, ncol=3, handles=h, fontsize=11, columnspacing=0.9)
        return f

    def draw_cls(metric, ylab, legend=False):
        def f(fig, ss):
            ax = fig.add_subplot(ss)
            for j, t in enumerate(ccol):
                d = cl[cl.target == t].set_index("scheme")
                ax.bar([j - 0.18], [d.loc["GKF5", metric]], 0.34, color=ccol[t], edgecolor=INK, lw=0.7)
                ax.bar([j + 0.18], [d.loc["LOCO", metric]], 0.34, color=ccol[t], edgecolor=INK, lw=0.7, hatch="//")
                if metric == "AP":
                    ax.plot([j - 0.38, j + 0.38], [d.loc["LOCO", "prevalence"]] * 2, color=INK, lw=1.4, ls="--")
            ax.set_xticks(range(4)); ax.set_xticklabels([f5.classifier_axis_label(t) for t in ccol], rotation=30, ha="right")
            style_axis(ax, ylabel=ylab, tick_size=11.5)
            numeric_ticks(ax, x=False, n=4)
            style_axis(ax, tick_size=11.5)
            if legend:
                h = [Patch(fc="white", ec=INK, label="GKF5"), Patch(fc="white", ec=INK, hatch="//", label="LOCO"),
                     Line2D([], [], color=INK, ls="--", lw=1.4, label="Prevalence")]
                outside_legend(ax, handles=h, ncol=3, fontsize=10.5, handlelength=1.4, columnspacing=0.8)
        return f

    cf = Composite(name, "FigS8_resolution_classifier_metrics", height=8.8)
    rows = cf.grid([1.1, 1], gap=0.6)
    r1 = hgrid(rows[0], [1, 1, 1], gap=0.62)
    for i, (l, (m, yl)) in enumerate(zip("abc", [("R2", "R$^2$"), ("bias_share", "Country-offset share of MSE"),
                                                 ("within_corr", "Within-country r")])):
        cf.panel(l, r1[i], draw_res(m, yl, legend=(i == 1)))
    r2 = hgrid(rows[1], [1, 1, 1], gap=0.62)
    for i, (l, (m, yl)) in enumerate(zip("def", [("AUC", "AUC"), ("AP", "Average precision"),
                                                 ("Brier", "Brier score")])):
        cf.panel(l, r2[i], draw_cls(m, yl, legend=(i == 2)))
    cf.finish()
    source_data(name, "a_c_taxonomic_resolution.csv", tr)
    source_data(name, "d_f_classification.csv", cl)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Render independent supplementary figures in isolated processes.
        # Passing a number retains the notebook-friendly single-figure entry point.
        import subprocess
        from concurrent.futures import ThreadPoolExecutor
        def render_one(i):
            result = subprocess.run([sys.executable, str(__file__), str(i)], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace")
            print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, end="", flush=True)
            if result.returncode:
                raise RuntimeError(f"Supplementary figure {i} failed")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(render_one, range(1, 9)))
    else:
        setup()
        for i in [int(a) for a in sys.argv[1:]]:
            globals()[f"fig_s{i}"]()
            print(f"S{i} done", flush=True)
