"""Fig. 1  Data coverage, country structure of the targets, validation design and the few-shot EB deployment rule."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse, Circle, Patch
from figstyle import (setup, style_axis, numeric_ticks, outside_legend, hgrid, Composite, source_data, RES, ROOT,
                      C, INK, MUTED, RULE, TARGETS, T_COL, SCHEME_COL, PALETTE, validate_series_colors, equal_earth,
                      draw_world, short)

NAME = "Fig1"
T_CMAP = LinearSegmentedColormap.from_list("temp", [C["blue"], C["teal"], C["gold"], C["rose"]])
T_NORM = TwoSlopeNorm(vmin=20, vcenter=42, vmax=56)
EUROPE = (-11, 32, 35, 64)
LANE_COL = validate_series_colors(PALETTE, {"train": C["blue"], "new": C["rose"], "out": C["mauve"]})


def digesters():
    d = pd.read_csv(RES / "digester_level_table.csv")
    d["log_NH4"] = np.log10(d.Ammonium.where(d.Ammonium > 0))
    d["log_SRT"] = np.log10(d.SRT.where(d.SRT > 0))
    return d


def icc(x, g):
    """One-way ANOVA (method-of-moments) intraclass correlation: between-country share of variance."""
    df = pd.DataFrame({"x": x, "g": g}).dropna()
    gr = df.groupby("g").x
    n = gr.size(); N, J = n.sum(), len(n)
    msb = (n * (gr.mean() - df.x.mean()) ** 2).sum() / (J - 1)
    msw = ((df.x - gr.transform("mean")) ** 2).sum() / (N - J)
    n0 = (N - (n ** 2).sum() / N) / (J - 1)
    sb = max((msb - msw) / n0, 0.0)
    return sb / (sb + msw), N, J


def map_frame(d):
    m = pd.read_csv(ROOT / "processed" / "metadata_clean.csv")
    g = m.groupby("Digester").agg(lat=("Latitude", "first"), lon=("Longitude", "first"),
                                  T=("Temperature", "mean"), country=("Country", "first"))
    lab = d.set_index("Digester")[TARGETS].notna().sum(1)
    g["n_labels"] = lab.reindex(g.index).fillna(0).astype(int)
    return g.reset_index()


def size_of(n):
    return 12 + 16 * np.asarray(n)


def scatter_digesters(ax, g, scale=1.0):
    x, y = equal_earth(g.lon.values, g.lat.values)
    has = g["T"].notna().values
    ax.scatter(x[~has], y[~has], s=size_of(g.n_labels.values[~has]) * scale, facecolor="white", edgecolor=INK,
               lw=0.7, zorder=3)
    order = np.argsort(-g.n_labels.values[has])
    ax.scatter(x[has][order], y[has][order], s=size_of(g.n_labels.values[has][order]) * scale,
               c=g["T"].values[has][order], cmap=T_CMAP, norm=T_NORM, edgecolor=INK, lw=0.55, alpha=0.95, zorder=4)


def europe_box(ax):
    lon = np.r_[np.linspace(EUROPE[0], EUROPE[1], 30), np.full(30, EUROPE[1]),
                np.linspace(EUROPE[1], EUROPE[0], 30), np.full(30, EUROPE[0])]
    lat = np.r_[np.full(30, EUROPE[2]), np.linspace(EUROPE[2], EUROPE[3], 30), np.full(30, EUROPE[3]),
                np.linspace(EUROPE[3], EUROPE[2], 30)]
    bx, by = equal_earth(lon, lat)
    ax.plot(bx, by, color=MUTED, lw=1.1, zorder=5)


def draw_a(fig, ss, g):
    gg = hgrid(ss, [2.75, 1.0], gap=0.3)
    ax = fig.add_subplot(gg[0])
    draw_world(ax, extent=(-168, 178, -47, 72))
    ax.set_aspect("equal", adjustable="datalim")
    scatter_digesters(ax, g)
    europe_box(ax)
    ax.text(0.99, 0.03, f"{len(g)} digesters, {g.country.nunique()} countries", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=12.5, color=INK)
    for n, lab in [(0, "No process label"), (1, "1 label"), (3, "3 labels"), (5, "5 labels")]:
        ax.scatter([], [], s=size_of(n), facecolor="white", edgecolor=INK, lw=0.7, label=lab)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.04), fontsize=11.5, labelspacing=0.8, handletextpad=0.4)
    sub = gg[1].subgridspec(2, 1, height_ratios=[1, 0.055], hspace=0.32)
    ins = fig.add_subplot(sub[0])
    draw_world(ins, extent=EUROPE, lw=0.5)
    ins.set_aspect("equal", adjustable="datalim")
    scatter_digesters(ins, g, scale=1.1)
    for s in ins.spines.values():
        s.set_visible(True); s.set_color(MUTED); s.set_linewidth(1.1)
    ins.text(0.03, 0.97, "Europe", transform=ins.transAxes, ha="left", va="top", fontsize=12.5, color=INK)
    cax = fig.add_subplot(sub[1])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=T_NORM, cmap=T_CMAP), cax=cax, orientation="horizontal")
    cb.set_ticks([20, 30, 42, 56])
    cb.set_label("Digester temperature (°C)", fontsize=12)
    cb.ax.tick_params(labelsize=11, width=1.0, length=3.5)
    source_data(NAME, "a_digester_locations.csv", g)


def draw_b(fig, ss, d):
    gg = hgrid(ss, [0.9, 1.0], gap=0.12)
    ax_bar, ax_dot = fig.add_subplot(gg[0]), fig.add_subplot(gg[1])
    cnt = d.groupby("Country").size().sort_values()
    lab = d.groupby("Country")[TARGETS].count().loc[cnt.index]
    y = np.arange(len(cnt))
    ax_bar.barh(y, cnt.values, color=C["blue"], edgecolor=INK, lw=0.7, height=0.72)
    for yi, v in zip(y, cnt.values):
        ax_bar.text(v + 0.6, yi, str(v), va="center", fontsize=9.5, color=INK)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(cnt.index)
    ax_bar.set_xlim(0, 42)
    style_axis(ax_bar, xlabel="Digesters", tick_size=10.5)
    ax_bar.set_xticks([0, 20, 40])
    style_axis(ax_bar, tick_size=10.5)
    for j, t in enumerate(TARGETS):
        v = lab[t].values
        ok = v > 0
        ax_dot.scatter(np.full(ok.sum(), j), y[ok], s=5 * v[ok] + 5, color=T_COL[t], edgecolor=INK, lw=0.5)
        ax_dot.text(j, len(cnt) + 0.1, f"{int(v.sum())}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax_dot.set_xticks(range(len(TARGETS)))
    ax_dot.set_xticklabels([short(t) for t in TARGETS], rotation=40, ha="right")
    ax_dot.set_yticks(y)
    ax_dot.set_yticklabels([])
    ax_dot.set_xlim(-0.6, len(TARGETS) - 0.4)
    style_axis(ax_dot, tick_size=10.5)
    for a in (ax_bar, ax_dot):
        a.set_ylim(-0.7, len(cnt) + 1.0)
        a.spines["left"].set_bounds(-0.7, len(cnt) - 0.3)
    for s in [5, 15, 30]:
        ax_bar.scatter([], [], s=5 * s + 5, color="white", edgecolor=INK, lw=0.7, label=str(s))
    ax_bar.legend(title="Labelled digesters", loc="lower right", bbox_to_anchor=(1.0, 0.0), fontsize=10,
                  title_fontsize=10, ncol=1, labelspacing=0.5, borderaxespad=0.1, handletextpad=0.3)
    src = lab.copy(); src.insert(0, "digesters", cnt)
    source_data(NAME, "b_labels_per_country.csv", src.reset_index())


def draw_c(fig, ss, d):
    ax = fig.add_subplot(ss)
    r = pd.DataFrame([dict(target=t, **dict(zip(["ICC", "n_digesters", "n_countries"], icc(d[t], d.Country))))
                      for t in TARGETS])
    x = np.arange(len(TARGETS))
    ax.bar(x, r.ICC, color=[T_COL[t] for t in TARGETS], edgecolor=INK, lw=0.8, width=0.62)
    ax.bar(x, 1 - r.ICC, bottom=r.ICC, color="white", edgecolor=INK, lw=0.8, width=0.62, hatch="..")
    for xi, v in zip(x, r.ICC):
        ax.text(xi, v / 2, f"{v:.2f}", ha="center", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([short(t) for t in TARGETS], rotation=40, ha="right")
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax, ylabel="Share of target variance")
    style_axis(ax)
    ax.spines["left"].set_bounds(0, 1)
    h = [Patch(fc=C["rose"], ec=INK, label="Between countries"), Patch(fc="white", ec=INK, hatch="..",
                                                                     label="Within country")]
    ax.legend(handles=h, loc="upper left", ncol=1, fontsize=10.5, handlelength=1.3, labelspacing=0.3,
              borderaxespad=0.05)
    source_data(NAME, "c_country_ICC.csv", r)


def draw_d(fig, ss):
    ax = fig.add_subplot(ss)
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 10); ax.axis("off")
    rng = np.random.default_rng(7)
    ncty, nd = 6, 5
    for scheme, x0, title in [("GKF5", 0.25, "GKF5: new digesters,\nseen countries"),
                              ("LOCO", 5.25, "LOCO: new country")]:
        ax.text(x0 + 2.25, 9.75, title, ha="center", va="top", fontsize=11.5, color=INK, linespacing=1.2)
        test = rng.random((ncty, nd)) < 0.2 if scheme == "GKF5" else np.zeros((ncty, nd), bool)
        if scheme == "LOCO":
            test[2, :] = True
        for i in range(ncty):
            yy = 7.0 - i * 1.08
            ax.text(x0 + 0.05, yy + 0.33, f"C{i + 1}", ha="right", va="center", fontsize=10, color=INK)
            for j in range(nd):
                fc = SCHEME_COL[scheme] if test[i, j] else "white"
                ax.add_patch(FancyBboxPatch((x0 + 0.25 + j * 0.84, yy), 0.66, 0.66,
                                            boxstyle="round,pad=0,rounding_size=0.12", fc=fc, ec=INK, lw=0.8))
    for x, fc, lab in [(0.4, SCHEME_COL["GKF5"], "Test (GKF5)"), (3.7, SCHEME_COL["LOCO"], "Test (LOCO)"),
                       (7.1, "white", "Train")]:
        ax.add_patch(FancyBboxPatch((x, 0.05), 0.42, 0.36, boxstyle="round,pad=0,rounding_size=0.08", fc=fc,
                                    ec=INK, lw=0.8))
        ax.text(x + 0.55, 0.23, lab, va="center", fontsize=10.5, color=INK)


# ------------------------------------------------------------------ method schematic (EB deployment rule)
def icon_digester(ax, x, y, w=2.0, h=2.9, fc=None, check=False):
    fc = fc or C["blue"]
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h * 0.8, boxstyle="round,pad=0,rounding_size=0.4",
                                fc=fc, ec=INK, lw=1.0, zorder=3))
    ax.add_patch(Ellipse((x, y - h / 2 + h * 0.8), w, h * 0.4, fc=fc, ec=INK, lw=1.0, zorder=2))
    ax.add_patch(Rectangle((x - 0.17, y + h * 0.42), 0.34, h * 0.16, fc=INK, ec=INK, zorder=4))
    ax.plot([x - w / 2 + 0.3, x + w / 2 - 0.3], [y - h * 0.05] * 2, color=INK, lw=0.8, zorder=4)
    if check:
        ax.add_patch(Circle((x + w / 2 - 0.1, y + h / 2 - 0.1), 0.55, fc=C["gold"], ec=INK, lw=0.8, zorder=5))
        ax.plot([x + w / 2 - 0.38, x + w / 2 - 0.15, x + w / 2 + 0.2], [y + h / 2 - 0.1, y + h / 2 - 0.38,
                y + h / 2 + 0.2], color=INK, lw=1.1, zorder=6)


def icon_bars(ax, x, y):
    cols = [C["teal"], C["gold"], C["lav"], C["rose"]]
    fr = [[0.35, 0.25, 0.25, 0.15], [0.2, 0.4, 0.2, 0.2], [0.45, 0.15, 0.2, 0.2], [0.3, 0.3, 0.3, 0.1],
          [0.25, 0.35, 0.15, 0.25]]
    for i, f in enumerate(fr):
        b = y - 2.0
        for c, v in zip(cols, f):
            ax.add_patch(Rectangle((x - 3.3 + i * 1.35, b), 1.05, 4.0 * v, fc=c, ec=INK, lw=0.6, zorder=3))
            b += 4.0 * v


def sbox(ax, x0, x1, yc, h, text, fc, size=11.5, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=fc, ec=INK, lw=1.1, ls=ls, alpha=1, zorder=2))
    ax.text((x0 + x1) / 2, yc, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.45,
            zorder=3)


def sarrow(ax, a, b, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, color=INK, lw=1.3, shrinkA=1,
                                 shrinkB=1, connectionstyle=f"arc3,rad={rad}", zorder=4))


def tint(c, a=0.42):
    from matplotlib.colors import to_rgb
    r = np.array(to_rgb(c))
    return tuple(1 - a * (1 - r))


def draw_e(fig, ss):
    ax = fig.add_subplot(ss)
    ax.set_xlim(0, 100); ax.set_ylim(0, 25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    yt, yn, h = 17.6, 6.2, 5.9
    for yc, lab, col in [(yt, "Training countries  (j = 1, …, J)", LANE_COL["train"]),
                         (yn, "New country c", LANE_COL["new"])]:
        ax.add_patch(FancyBboxPatch((0.3, yc - 5.0), 99.4, 10.0, boxstyle="round,pad=0,rounding_size=1.0",
                                    fc=tint(col, 0.12), ec="none", zorder=0))
        ax.text(1.2, yc + 4.25, lab, ha="left", va="center", fontsize=11.5, color=INK)
    tr, nw, out = tint(LANE_COL["train"]), tint(LANE_COL["new"]), tint(LANE_COL["out"], 0.5)
    icon_bars(ax, 6.2, yt - 0.6)
    ax.text(6.2, yt - 3.8, "Genus CLR", ha="center", va="center", fontsize=10.5, color=INK)
    sbox(ax, 13.5, 30.5, yt, h, "ExtraTrees $f(x)$\nfitted on J countries", tr)
    sbox(ax, 34.0, 54.0, yt, h, "Inner LOCO residuals\n$r_{ij} = y_{ij} - f_{-j}(x_{ij})$", tr)
    sbox(ax, 57.5, 79.0, yt, h, "Variance components\n$\\hat\\tau^2$ between,  $\\hat\\sigma^2$ within",
         tr)
    sbox(ax, 82.5, 99.0, yt, h, "Label budget\n$k_{50} = \\hat\\sigma^2/\\hat\\tau^2$", tr, ls="--")
    for xa, xb in [(10.0, 13.5), (30.5, 34.0), (54.0, 57.5), (79.0, 82.5)]:
        sarrow(ax, (xa, yt), (xb, yt))
    for i, dx in enumerate([-2.3, 0.0, 2.3]):
        icon_digester(ax, 6.2 + dx, yn - 0.2, w=1.7, h=2.6, fc=C["rose"] if i < 2 else "white", check=i < 2)
    ax.text(6.2, yn - 3.8, "k labelled", ha="center", va="center", fontsize=10.5, color=INK)
    sbox(ax, 13.5, 30.5, yn, h, "Offset of the labels\n$\\bar r_L = \\mathrm{mean}_{i \\in L}\\,(y_i - f(x_i))$", nw,
         size=11)
    sbox(ax, 34.0, 54.0, yn, h, "EB shrinkage  $\\hat b_c = w_k\\,\\bar r_L$\n"
         "$w_k = k\\hat\\tau^2 / (k\\hat\\tau^2 + \\hat\\sigma^2)$", nw)
    sbox(ax, 57.5, 79.0, yn, h, "Prediction, no refitting\n$\\hat y = f(x) + \\hat b_c$", out)
    sbox(ax, 82.5, 99.0, yn, h, "k-shot jackknife+\nprediction intervals", out)
    for xa, xb in [(10.0, 13.5), (30.5, 34.0), (54.0, 57.5), (79.0, 82.5)]:
        sarrow(ax, (xa, yn), (xb, yn))
    sarrow(ax, (22.0, yt - h / 2), (22.0, yn + h / 2))
    sarrow(ax, (68.2, yt - h / 2), (44.0, yn + h / 2))
    sarrow(ax, (72.0, yt - h / 2), (90.7, yn + h / 2))


if __name__ == "__main__":
    setup()
    d = digesters()
    g = map_frame(d)
    cf = Composite(NAME, "Fig1_data_design", height=13.0)
    rows = cf.grid([1.12, 1.18, 0.86], gap=0.55)
    cf.panel("a", rows[0], lambda f, s: draw_a(f, s, g))
    r2 = hgrid(rows[1], [1.45, 0.82, 1.3], gap=0.62)
    cf.panel("b", r2[0], lambda f, s: draw_b(f, s, d))
    cf.panel("c", r2[1], lambda f, s: draw_c(f, s, d))
    cf.panel("d", r2[2], draw_d)
    cf.panel("e", rows[2], draw_e)
    rep = cf.finish()
    print({r["panel"]: round(r["max_residual_pt"], 2) for r in rep})
