"""Fig. 4  Few-shot empirical-Bayes recalibration of the country offset (point predictions)."""
import importlib
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figstyle import (setup, style_axis, numeric_ticks, facet_tag, outside_legend, between_x, hgrid, Composite, source_data,
                      RES, ROOT, C, INK, MUTED, RULE, T_COL, PALETTE, validate_series_colors, short)

sys.path.insert(0, str(ROOT / "scripts"))
fs = importlib.import_module("08_fewshot_recalibration")

NAME = "Fig4"
TG = ["Temperature", "log_NH4", "Digester_age", "pH"]
M_COL = validate_series_colors(PALETTE, {"ET|EB": C["mauve"], "ET|offset": C["gold"], "ET|refit_with_k": C["teal"],
                                         "ET|raw": C["blue"], "CountryMeanOfK": C["green"], "iid": C["lav"]})
M_LAB = {"ET|EB": "EB shrinkage", "ET|offset": "Unshrunk offset", "ET|refit_with_k": "Refit with k labels",
         "ET|raw": "No recalibration", "CountryMeanOfK": "Mean of k labels only"}
M_LS = {"ET|EB": "-", "ET|offset": "--", "ET|refit_with_k": "-.", "ET|raw": ":", "CountryMeanOfK": "-"}
M_MK = {"ET|EB": "o", "ET|offset": "s", "ET|refit_with_k": "^", "ET|raw": "D", "CountryMeanOfK": "v"}
YLIM = {"Temperature": (-0.35, 0.7), "log_NH4": (-0.45, 0.5), "Digester_age": (-0.3, 0.45), "pH": (-0.55, 0.6)}
K_CTY = 3
N_DRAWS = 50


def per_country(target):
    """Country-level raw vs EB (inner-LOCO moments) at k = K_CTY."""
    d = pd.read_pickle(RES / f"conformal_store_{target}.pkl")
    store, yi, dig, cty = d["store"], d["y"], d["dig"], d["cty"]
    dig_c = pd.Series(cty, index=dig).groupby(level=0).first()
    y_d = fs.dig_avg(yi, dig)
    rng = np.random.default_rng(0)
    rows = []
    for kf in sorted({k for (kind, k) in store if kind == "outer"}):
        po = store[("outer", kf)]
        te = ~np.isnan(po)
        p_d = fs.dig_avg(po[te], dig[te])
        yy, pp = y_d.loc[p_d.index].values, p_d.values
        pp_ = store[("blocked", kf)]
        msk = ~np.isnan(pp_)
        r = fs.dig_avg(yi[msk] - pp_[msk], dig[msk])
        tau2, sig2 = fs.moments(r.values, dig_c.loc[r.index].values)
        if len(yy) < K_CTY + 2:
            continue
        sh = K_CTY * tau2 / (K_CTY * tau2 + sig2)
        acc = []
        for _ in range(N_DRAWS):
            lab = rng.choice(len(yy), K_CTY, replace=False)
            ev = np.setdiff1d(np.arange(len(yy)), lab)
            b = sh * (yy[lab] - pp[lab]).mean()
            acc.append((np.abs(yy[ev] - pp[ev]).mean(), np.abs(yy[ev] - pp[ev] - b).mean(),
                        yy[ev].mean(), pp[ev].mean(), (pp[ev] + b).mean()))
        a = np.array(acc).mean(0)
        rows.append(dict(target=target, country=cty[te][0], n_dig=len(yy), shrink=sh, MAE_raw=a[0], MAE_EB=a[1],
                         obs_mean=a[2], pred_raw_mean=a[3], pred_EB_mean=a[4], offset=(yy - pp).mean(),
                         sd_target=y_d.std()))
    return pd.DataFrame(rows)


def draw_a(fig, ss):
    f = pd.read_csv(RES / "fewshot_recalibration_Temperature_pH_log_NH4_log_SRT_Digester_age.csv")
    f = f[f.method.isin(M_COL)]
    summ = f.groupby(["target", "method", "k"]).R2.agg(["mean", lambda v: v.quantile(0.025),
                                                         lambda v: v.quantile(0.975)]).reset_index()
    summ.columns = ["target", "method", "k", "mean", "lo", "hi"]
    g = hgrid(ss, [1, 1, 1, 1], gap=0.62)
    axes = [fig.add_subplot(g[i]) for i in range(4)]
    for ax, t in zip(axes, TG):
        for m in M_LAB:
            s = summ[(summ.target == t) & (summ.method == m)].sort_values("k")
            lo, hi = YLIM[t]
            ax.plot(s.k, np.clip(s["mean"].values, lo, None), color=M_COL[m], ls=M_LS[m],
                    lw=2.6 if m == "ET|EB" else 2.0, marker=M_MK[m], ms=7.5 if m == "ET|EB" else 6,
                    mec=INK, mew=0.7, zorder=4 if m == "ET|EB" else 3, clip_on=False)
            if m == "ET|EB":
                ax.fill_between(s.k, s.lo, s.hi, color=M_COL[m], alpha=0.25, lw=0, zorder=1)
        ax.axhline(0, color=MUTED, lw=0.9, ls=":")
        ax.set_ylim(*YLIM[t])
        ax.set_xticks([0, 1, 2, 3, 5])
        ax.set_xlim(-0.3, 5.3)
        style_axis(ax, xlabel="Labelled digesters, k", ylabel="LOCO R$^2$" if t == TG[0] else None)
        numeric_ticks(ax, x=False, n=5)
        style_axis(ax)
        facet_tag(ax, short(t), x=0.97, y=0.04, ha="right", va="bottom")
    h = [Line2D([], [], color=M_COL[m], ls=M_LS[m], marker=M_MK[m], mec=INK, lw=2.2, ms=7, label=M_LAB[m])
         for m in M_LAB]
    outside_legend(axes[1], x=between_x(axes[1], axes[2]), ncol=5, handles=h, handlelength=2.4,
                   columnspacing=1.2, pad_pt=8)
    source_data(NAME, "a_R2_vs_k.csv", summ)


def draw_b(fig, ss):
    """Share of the GKF5 -> LOCO R2 loss recovered by EB, evaluated on the same digesters per draw."""
    ax = fig.add_subplot(ss)
    r = pd.read_csv(RES / "eb_recovery_same_digesters.csv")
    for t in ["Temperature", "Digester_age", "log_NH4", "pH"]:
        d = r[r.target == t].sort_values("k")
        ax.plot(d.k, 100 * d.recovered_share, color=T_COL[t], lw=2.4, marker="o", ms=7.5, mec=INK, mew=0.8,
                ls="-" if t != "pH" else ":", zorder=3)
        nudge = {"Digester_age": -1.5, "log_NH4": -2.5}.get(t, 0)
        ax.text(5.3, 100 * d.iloc[-1].recovered_share + nudge, short(t), color=INK, fontsize=11.5, va="center")
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.set_xlim(-0.3, 7.3); ax.set_ylim(-10, 80)
    ax.set_xticks([0, 1, 2, 3, 5])
    style_axis(ax, xlabel="Labelled digesters, k", ylabel="Loss recovered (%)")
    numeric_ticks(ax, x=False, n=5)
    style_axis(ax)
    ax.spines["bottom"].set_bounds(-0.3, 5.3)
    source_data(NAME, "b_recovered_share.csv", r)


def draw_expected(fig, ss):
    """Expected R2 gain from the training-country variance components against the observed gain."""
    ax = fig.add_subplot(ss)
    r = pd.read_csv(RES / "label_budget_expected_vs_observed.csv")
    r = r[r.k > 0]
    ax.plot([0, 0.6], [0, 0.6], color=MUTED, lw=1.1, ls="--", zorder=0)
    for t in TG:
        d = r[r.target == t].sort_values("k")
        ax.plot(d.dR2_expected, d.dR2_observed, color=T_COL[t], lw=1.8, zorder=2)
        ax.scatter(d.dR2_expected, d.dR2_observed, s=18 + 14 * d.k, color=T_COL[t], edgecolor=INK, lw=0.7,
                   zorder=3)
    lab = {"Digester_age": (0.115, 0.19, "center"), "log_NH4": (0.54, 0.19, "center"), "pH": (0.03, -0.035, "left")}
    for t, (x, y, ha) in lab.items():
        ax.text(x, y, short(t), fontsize=11.5, ha=ha, va="center", color=INK)
    t1 = r[(r.target == "Temperature") & (r.k == 1)].iloc[0]
    ax.annotate("Temperature", (t1.dR2_expected, t1.dR2_observed), xytext=(0.115, -0.022), fontsize=11.5,
                ha="left", va="center", color=INK, arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, shrinkB=4))
    for k in [1, 3, 5]:
        ax.scatter([], [], s=18 + 14 * k, color="white", edgecolor=INK, lw=0.7, label=f"k = {k}")
    ax.legend(loc="upper left", handletextpad=0.2, borderaxespad=0.1, labelspacing=0.5, fontsize=11)
    ax.set_xlim(-0.02, 0.6); ax.set_ylim(-0.06, 0.35)
    style_axis(ax, xlabel="Moment-based ΔR$^2$", ylabel="Observed ΔR$^2$")
    numeric_ticks(ax, n=4)
    style_axis(ax)
    source_data(NAME, "f_expected_vs_observed_gain.csv", r)


def draw_weights(fig, ss):
    ax = fig.add_subplot(ss)
    c = pd.read_csv(RES / "eb_moment_components.csv")
    med = c.groupby("target")[["tau2_blocked", "sig2_blocked", "tau2_iid", "sig2_iid"]].median()
    k = np.linspace(0, 10, 200)
    for t in TG:
        tau2, sig2 = med.loc[t, "tau2_blocked"], med.loc[t, "sig2_blocked"]
        k50 = sig2 / tau2
        ax.plot(k, k * tau2 / (k * tau2 + sig2), color=T_COL[t], lw=2.6, label=short(t))
        ti, si = med.loc[t, "tau2_iid"], med.loc[t, "sig2_iid"]
        ax.plot(k, k * ti / (k * ti + si), color=T_COL[t], lw=1.7, ls="--")
        if k50 < 10:
            ax.scatter([k50], [0.5], s=46, color=T_COL[t], edgecolor=INK, lw=0.8, zorder=4)
    ax.axhline(0.5, color=MUTED, lw=0.9, ls=":", zorder=0)
    ax.plot([], [], color=INK, lw=2.4, label="Inner LOCO")
    ax.plot([], [], color=INK, lw=1.7, ls="--", label="Inner i.i.d.")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.45)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax, xlabel="Labelled digesters, k", ylabel="Shrinkage weight $w_k$")
    numeric_ticks(ax, y=False, n=5)
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 1)
    ax.legend(loc="upper left", fontsize=10.5, ncol=2, handlelength=1.4, borderaxespad=0.05, labelspacing=0.3,
              columnspacing=0.8, handletextpad=0.4)
    med["k50"] = med.sig2_blocked / med.tau2_blocked
    source_data(NAME, "d_variance_components_median.csv", med.reset_index())


def draw_ratio(fig, ss):
    ax = fig.add_subplot(ss)
    c = pd.read_csv(RES / "eb_moment_components.csv")
    c = c[c.target != "pH"].copy()  # pH: tau^2 is ~0 under both residual sources, so the ratio is undefined
    c["ratio"] = c.tau2_blocked / c.tau2_iid
    rng = np.random.default_rng(1)
    tg = TG[:3]
    for i, t in enumerate(tg):
        v = c[c.target == t].ratio.values
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=38, color=T_COL[t], edgecolor=INK, lw=0.6)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=2.4)
        ax.text(i, v.max() * 1.35, f"{np.median(v):.1f}×", ha="center", fontsize=11.5)
    ax.axhline(1, color=MUTED, lw=1.1, ls="--")
    ax.set_yscale("log")
    ax.set_ylim(0.8, 60)
    ax.set_xticks(range(len(tg)))
    ax.set_xticklabels([short(t) for t in tg], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(tg) - 0.4)
    style_axis(ax, ylabel="τ$^2$ inner LOCO / inner i.i.d.")
    source_data(NAME, "c_tau2_ratio_per_country.csv", c)


def draw_ablation(fig, ss):
    ax = fig.add_subplot(ss)
    a = pd.read_csv(RES / "eb_moment_ablation.csv")
    a = a[a.k == 3].groupby(["target", "moments"]).R2.agg(["mean", "std"]).reset_index()
    order = [("raw", "ET|raw", "No recalibration"), ("none", "ET|offset", "Unshrunk"),
             ("iid", "iid", "EB, inner i.i.d."), ("blocked", "ET|EB", "EB, inner LOCO")]
    w = 0.2
    for j, (m, ck, lab) in enumerate(order):
        v = a[a.moments == m].set_index("target").loc[TG]
        ax.bar(np.arange(len(TG)) + (j - 1.5) * w, v["mean"], w, yerr=v["std"], color=M_COL[ck], edgecolor=INK,
               lw=0.8, error_kw=dict(ecolor=INK, elinewidth=1.0, capsize=2.2), label=lab)
    ax.set_xticks(range(len(TG)))
    ax.set_xticklabels([short(t) for t in TG], rotation=30, ha="right")
    ax.set_ylim(0, 0.8)
    style_axis(ax, ylabel="LOCO R$^2$ at k = 3")
    numeric_ticks(ax, x=False, n=5)
    style_axis(ax)
    outside_legend(ax, ncol=2, fontsize=10.5, handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
    source_data(NAME, "e_moment_ablation_k3.csv", a)


def draw_country_mae(fig, ss, pc):
    ax = fig.add_subplot(ss)
    for t in TG:
        d = pc[pc.target == t]
        x = np.abs(d.offset) / d.sd_target
        v = 100 * (d.MAE_EB / d.MAE_raw - 1)
        ax.scatter(x, v, s=12 + 2.0 * d.n_dig, color=T_COL[t], edgecolor=INK, lw=0.6, alpha=0.95,
                   label=f"{short(t)} ({(v < 0).sum()}/{len(v)})")
    ax.axhline(0, color=MUTED, lw=1.1, ls="--")
    ax.set_xlim(-0.03, 1.3)
    ax.set_ylim(-65, 35)
    style_axis(ax, xlabel="|Country offset| / SD", ylabel="Country MAE change at k = 3 (%)")
    numeric_ticks(ax, n=4)
    style_axis(ax)
    ax.legend(loc="upper right", fontsize=10.5, handletextpad=0.1, borderaxespad=0.1, labelspacing=0.3,
              markerscale=0.8)


def draw_means(fig, ss, pc):
    g = hgrid(ss, [1, 1], gap=0.55)
    axes = [fig.add_subplot(g[i]) for i in range(2)]
    for ax, t in zip(axes, ["Temperature", "log_NH4"]):
        d = pc[pc.target == t]
        lo = min(d.obs_mean.min(), d.pred_raw_mean.min(), d.pred_EB_mean.min())
        hi = max(d.obs_mean.max(), d.pred_raw_mean.max(), d.pred_EB_mean.max())
        pad = 0.08 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1.1, ls="--", zorder=0)
        for _, r in d.iterrows():
            ax.annotate("", xy=(r.obs_mean, r.pred_EB_mean), xytext=(r.obs_mean, r.pred_raw_mean),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, mutation_scale=10, shrinkA=4, shrinkB=4))
        ax.scatter(d.obs_mean, d.pred_raw_mean, s=52, color=M_COL["ET|raw"], edgecolor=INK, lw=0.8, zorder=3)
        ax.scatter(d.obs_mean, d.pred_EB_mean, s=52, color=M_COL["ET|EB"], edgecolor=INK, lw=0.8, zorder=4)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("auto")
        unit = "°C" if t == "Temperature" else "log$_{10}$"
        style_axis(ax, xlabel=f"Observed mean ({unit})",
                   ylabel="Predicted country mean" if t == "Temperature" else None)
        numeric_ticks(ax, n=4)
        style_axis(ax)
        facet_tag(ax, short(t), x=0.04, y=0.97)
    h = [Line2D([], [], marker="o", ls="", ms=8, mfc=M_COL["ET|raw"], mec=INK, label="k = 0"),
         Line2D([], [], marker="o", ls="", ms=8, mfc=M_COL["ET|EB"], mec=INK, label="EB, k = 3")]
    axes[1].legend(handles=h, loc="lower right", fontsize=11, handletextpad=0.2, borderaxespad=0.1)


if __name__ == "__main__":
    setup()
    pc = pd.concat([per_country(t) for t in TG])
    source_data(NAME, "g_h_per_country_k3.csv", pc)
    cf = Composite(NAME, "Fig4_fewshot_EB", height=12.6)
    rows = cf.grid([0.82, 1.0, 1.0], gap=0.55)
    cf.panel("a", rows[0], draw_a)
    r2 = hgrid(rows[1], [0.95, 0.8, 1.25, 1.1], gap=0.62)
    cf.panel("b", r2[0], draw_b)
    cf.panel("c", r2[1], draw_ratio)
    cf.panel("d", r2[2], draw_weights)
    cf.panel("e", r2[3], draw_ablation)
    r3 = hgrid(rows[2], [1.0, 1.0, 1.8], gap=0.62)
    cf.panel("f", r3[0], draw_expected)
    cf.panel("g", r3[1], lambda f, s: draw_country_mae(f, s, pc))
    cf.panel("h", r3[2], lambda f, s: draw_means(f, s, pc))
    rep = cf.finish()
    print({r["panel"]: round(r["max_residual_pt"], 2) for r in rep})
