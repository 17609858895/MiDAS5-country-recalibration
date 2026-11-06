"""Shared style, colour roles, panel composition and export for the MiDAS 5 manuscript figures.

Composition follows the multi-panel method of the earlier wastewater-microbiome figure set:
each panel is a draw function that fills one grid cell; the outer visual bounds of every panel
(axes, tick and axis labels, legends, colourbars) are fitted to that cell, so panels in a row share
their top and bottom edges and panels in a column share their left and right edges; panel letters
sit in the gutter outside the panel; every panel is also rendered on its own canvas by the same
draw function (subfigures are not crops).  Text is regular-weight Arial.  One palette
(nature_new) is used for every figure; warm hues mark country-aware rules, cool hues the
i.i.d. or pooled baselines.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.text import Text
from matplotlib.transforms import Bbox, ScaledTranslation

HELPERS = Path(__file__).resolve().parents[1] / "figure_helpers"
sys.path.insert(0, str(HELPERS))
from ml_publication_style import configure_publication_style, numeric_ticks  # noqa: E402
from ml_publication_style import style_axis as _style_axis  # noqa: E402
from scientific_palettes import get_palette  # noqa: E402
from audit_palette_usage import validate_series_colors  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
FIG = ROOT / "figures"
PALETTE = "nature_new"
C = dict(zip(["rose", "blue", "green", "teal", "lav", "pink", "gold", "mauve"], get_palette(PALETTE)))
INK = "#30343B"
MUTED = "#6F7782"
RULE = "#C9CDD2"
WEIGHT = "normal"
LETTER_PT = 20
WIDTH = 13.5  # inches; every main figure has the same width

TARGETS = ["Temperature", "pH", "log_NH4", "log_SRT", "Digester_age"]
T_SHORT = {"Temperature": "Temperature", "pH": "pH", "log_NH4": "Ammonium N", "log_SRT": "SRT",
           "Digester_age": "Digester age"}
T_UNIT = {"Temperature": "Temperature (°C)", "pH": "pH", "log_NH4": "log$_{10}$ Ammonium N (mg N L$^{-1}$)",
          "log_SRT": "log$_{10}$ SRT (d)", "Digester_age": "Digester age (years)"}
T_COL = validate_series_colors(PALETTE, {"Temperature": C["rose"], "pH": C["blue"], "log_NH4": C["mauve"],
                                         "log_SRT": C["gold"], "Digester_age": C["teal"]})
SCHEME_COL = validate_series_colors(PALETTE, {"GKF5": C["blue"], "LOCO": C["rose"]})
DECOMP_COL = validate_series_colors(PALETTE, {"offset": C["rose"], "within": C["blue"]})
DIV_CMAP = LinearSegmentedColormap.from_list("pal_div", [C["blue"], "#FFFFFF", C["rose"]])


def seq_cmap(colour, name="pal_seq"):
    return LinearSegmentedColormap.from_list(name, ["#FFFFFF", colour])


def short(t):
    return T_SHORT[t].replace("Digester age", "Age")


def setup():
    configure_publication_style()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.weight": WEIGHT, "axes.labelweight": WEIGHT, "axes.titleweight": WEIGHT,
        "font.size": 12.5, "axes.labelsize": 13, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 11.5, "axes.titlesize": 13, "mathtext.default": "regular",
        "axes.labelpad": 5, "xtick.major.pad": 4, "ytick.major.pad": 4, "legend.frameon": False,
        "axes.linewidth": 1.3, "xtick.major.width": 1.3, "ytick.major.width": 1.3,
        "xtick.major.size": 5.5, "ytick.major.size": 5.5})


def style_axis(ax, xlabel=None, ylabel=None, tick_size=12):
    """Skill axis style (visible ticks, open top/right spines) at this figure set's sizes and weight."""
    _style_axis(ax, xlabel=xlabel, ylabel=ylabel, tick_size=tick_size, label_size=13)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.3)
    ax.tick_params(which="major", length=5.5, width=1.3, pad=4)
    ax.tick_params(which="minor", length=3.2, width=1.0)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 5


def facet_tag(ax, text, x=0.03, y=0.97, ha="left", va="top", size=12.5):
    """Short facet identifier (target or group name) inside the axis corner; not a panel title."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, ha=ha, va=va, color=INK)


def outside_legend(ax, above=True, pad_pt=6, x=0.5, **kw):
    """Legend centred above (or below) the axes at a fixed physical gap; `x` in axes fraction."""
    off = (pad_pt if above else -pad_pt) / 72
    tr = ax.transAxes + ScaledTranslation(0, off, ax.figure.dpi_scale_trans)
    kw.setdefault("borderaxespad", 0)
    return ax.legend(loc="lower center" if above else "upper center", bbox_to_anchor=(x, 1 if above else 0),
                     bbox_transform=tr, **kw)


def between_x(ax_left, ax_right):
    """Axes-fraction x (of ax_left) of the midpoint between two neighbouring axes."""
    a, b = ax_left.get_position(), ax_right.get_position()
    return 1 + (b.x0 - a.x1) / 2 / a.width


# ---------------------------------------------------------------- grid helpers (gaps in inches)
def hgrid(ss, ratios, gap=0.55):
    """Split a cell into columns separated by `gap` inches (room for a panel letter)."""
    fig = ss.get_gridspec().figure
    w = ss.get_position(fig).width * fig.get_size_inches()[0]
    n = len(ratios)
    col = (w - gap * (n - 1)) / n
    return ss.subgridspec(1, n, width_ratios=ratios, wspace=gap / col)


def vgrid(ss, ratios, gap=0.55):
    fig = ss.get_gridspec().figure
    h = ss.get_position(fig).height * fig.get_size_inches()[1]
    n = len(ratios)
    row = (h - gap * (n - 1)) / n
    return ss.subgridspec(n, 1, height_ratios=ratios, hspace=gap / row)


# ---------------------------------------------------------------- composition
class Composite:
    """Register panels drawn into grid cells, fit their outer bounds, letter and export them."""

    def __init__(self, name, stem, height, width=WIDTH, margins=(0.035, 0.995, 0.01, 0.99)):
        self.name, self.stem = name, stem
        self.fig = plt.figure(figsize=(width, height))
        left, right, bottom, top = margins
        self.box = (left, right, bottom, top)
        self.records = []

    def grid(self, ratios, gap=0.7):
        """Rows of the whole figure separated by `gap` inches."""
        left, right, bottom, top = self.box
        h = (top - bottom) * self.fig.get_size_inches()[1]
        n = len(ratios)
        row = (h - gap * (n - 1)) / n
        return self.fig.add_gridspec(n, 1, height_ratios=ratios, hspace=gap / row, left=left, right=right,
                                     bottom=bottom, top=top)

    def panel(self, letter, ss, draw):
        before = set(self.fig.axes)
        draw(self.fig, ss)
        axes = [a for a in self.fig.axes if a not in before]
        self.records.append(dict(key=letter, axes=axes, target=ss.get_position(self.fig).frozen(), draw=draw))

    def finish(self, standalone=True):
        fig = self.fig
        report = refine_layout(fig, self.records)
        letters(fig, report)
        regular_text(fig)
        qa = ROOT / "docs" / "figure_review"
        qa.mkdir(parents=True, exist_ok=True)
        (qa / f"{self.name}_layout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        save_pair(fig, FIG / self.name, self.stem)
        plt.close(fig)
        if standalone:
            sub = FIG / self.name / "subfigures"
            sub.mkdir(parents=True, exist_ok=True)
            W, H = self.fig.get_size_inches()
            for rec in self.records:
                t = rec["target"]
                render_standalone(rec["draw"], (t.width * W, t.height * H), sub, f"{self.stem}_{rec['key']}")
            readme = ["# Subfigures", "",
                      "Each panel is rendered on its own canvas by the same draw function used for the combined "
                      "figure (not cropped from it).", ""]
            readme += [f"- {r['key']}: `{self.stem}_{r['key']}.png`, `{self.stem}_{r['key']}.pdf`"
                       for r in self.records]
            (sub / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
        return report


def _outer(fig, axes, renderer):
    """Union of the axes' tight boxes and their axis labels (long labels can overhang the tight box)."""
    boxes = []
    for a in axes:
        if not a.get_visible():
            continue
        boxes.append(a.get_tightbbox(renderer))
        for lab in (a.xaxis.label, a.yaxis.label):
            if lab.get_visible() and lab.get_text():
                boxes.append(lab.get_window_extent(renderer))
    return Bbox.union([b for b in boxes if b is not None]).transformed(fig.transFigure.inverted())


def refine_layout(fig, records, tol_pt=0.35, iters=24):
    """Fit the outer visual bounds of each panel to its cell (fonts stay in points)."""
    fig.canvas.draw()
    regular_text(fig)
    for rec in records:
        for a in rec["axes"]:
            a.set_axes_locator(None)
    size = np.tile(fig.get_size_inches(), 2) * 72
    report = []
    for rec in records:
        target, axes = rec["target"], rec["axes"]
        for _ in range(iters):
            fig.canvas.draw()
            outer = _outer(fig, axes, fig.canvas.get_renderer())
            err = (np.array(outer.extents) - np.array(target.extents)) * size
            if np.max(np.abs(err)) < tol_pt:
                break
            sx, sy = target.width / outer.width, target.height / outer.height
            for a in axes:
                p = a.get_position(original=True)
                a.set_position([target.x0 + (p.x0 - outer.x0) * sx, target.y0 + (p.y0 - outer.y0) * sy,
                                p.width * sx, p.height * sy], which="both")
        fig.canvas.draw()
        outer = _outer(fig, axes, fig.canvas.get_renderer())
        res = (np.array(outer.extents) - np.array(target.extents)) * size
        report.append(dict(panel=rec["key"], outer=list(outer.extents), max_residual_pt=float(np.abs(res).max())))
    return report


def letters(fig, report):
    W = fig.get_size_inches()[0] * 72
    for p in report:
        x0, y0, x1, y1 = p["outer"]
        fig.text(x0 - 5 / W, y1, p["panel"], ha="right", va="top", fontsize=LETTER_PT, fontweight=WEIGHT,
                 color=INK, zorder=100)


def regular_text(fig):
    for t in fig.findobj(Text):
        if t.get_text():
            t.set_fontweight(WEIGHT)


def save_pair(fig, folder, stem):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{stem}.png", dpi=600, facecolor="white")
    fig.savefig(folder / f"{stem}.pdf", facecolor="white", metadata={"Creator": "", "Producer": "", "Author": ""})


def render_standalone(draw, size, folder, stem):
    fig = plt.figure(figsize=size)
    gs = fig.add_gridspec(1, 1, left=0.01, right=0.99, bottom=0.01, top=0.99)
    before = set(fig.axes)
    draw(fig, gs[0])
    axes = [a for a in fig.axes if a not in before]
    refine_layout(fig, [dict(key="", axes=axes, target=gs[0].get_position(fig).frozen())])
    regular_text(fig)
    save_pair(fig, folder, stem)
    plt.close(fig)


# ---------------------------------------------------------------- data helpers
def dig_level(p, y, dig):
    """Average the A/B replicate predictions of each digester."""
    d = pd.DataFrame({"p": p, "y": y, "dig": dig}).dropna().groupby("dig").mean()
    return d.y.values, d.p.values, d.index.values


def offset_decomposition(y, p, cty):
    """Split digester-level squared error into the country-mean residual (offset) and the remainder."""
    r = y - p
    b = pd.Series(r).groupby(cty).transform("mean").values
    sst = np.sum((y - y.mean()) ** 2)
    yc = y - pd.Series(y).groupby(cty).transform("mean").values
    pc = p - pd.Series(p).groupby(cty).transform("mean").values
    return dict(R2=1 - np.sum(r ** 2) / sst, offset=np.sum(b ** 2) / sst, within=np.sum((r - b) ** 2) / sst,
                bias_share=np.sum(b ** 2) / np.sum(r ** 2), within_corr=np.corrcoef(yc, pc)[0, 1])


def target_frame():
    """y, digester and country arrays (sample level, ordered as in the benchmark) for each target."""
    out = {}
    for t in TARGETS:
        s = pd.read_pickle(RES / f"fewshot_preds_{t}.pkl")
        out[t] = dict(y=np.asarray(s["y"]), dig=np.asarray(s["dig"]), cty=np.asarray(s["cty"]))
    return out


def source_data(name, fname, df):
    d = FIG / name / "source_data"
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / fname, index=False)


# ---------------------------------------------------------------- world maps (Natural Earth, no GIS dependency)
BASEMAP = Path(__file__).resolve().parent / "basemap"
MAP_LAND = "#EEF0F2"
MAP_EDGE = "#C3C7CD"
_EE = (1.340264, -0.081106, 0.000893, 0.003796)


def equal_earth(lon, lat):
    """Equal Earth projection (Šavrič et al., 2019); lon/lat in degrees, returns unit-sphere x, y."""
    lam, phi = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    a1, a2, a3, a4 = _EE
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (9 * a4 * th ** 8 + 7 * a3 * th ** 6 + 3 * a2 * th ** 2 + a1))
    y = a4 * th ** 9 + a3 * th ** 7 + a2 * th ** 3 + a1 * th
    return x, y


def country_polygons():
    """{country name: [ring arrays (lon, lat)]} from Natural Earth 1:110m admin-0."""
    import json
    g = json.load(open(BASEMAP / "ne_110m_admin_0_countries.geojson", encoding="utf-8"))
    out = {}
    for f in g["features"]:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        out[f["properties"]["ADMIN"]] = [np.asarray(p[0]) for p in polys]
    return out


NE_NAME = {"USA": "United States of America", "Republic of Korea": "South Korea", "Czech Republic": "Czechia"}


def draw_world(ax, proj=equal_earth, fill=None, extent=None, lw=0.4):
    """Draw land polygons; `fill` maps MiDAS country names to face colours (others light grey)."""
    from matplotlib.patches import Polygon
    fill = {NE_NAME.get(k, k): v for k, v in (fill or {}).items()}
    for name, rings in country_polygons().items():
        if name == "Antarctica":
            continue
        for r in rings:
            x, y = proj(r[:, 0], r[:, 1])
            ax.add_patch(Polygon(np.c_[x, y], closed=True, fc=fill.get(name, MAP_LAND), ec=MAP_EDGE, lw=lw,
                                 zorder=1))
    if extent is not None:
        latm = float(np.clip(0.0, extent[2], extent[3]))
        (x0, x1), _ = proj(np.array(extent[:2]), np.array([latm, latm]))
        _, (y0, y1) = proj(np.array([0.0, 0.0]), np.array(extent[2:]))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    else:
        ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
