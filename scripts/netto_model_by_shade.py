"""Netto et al. (2005) predicted vs laboratory chlorophyll, with a separate linear fit for each shade block
(B1 outside panel shade, B2 under solar panel). Same layout as netto_model.py; data and models are imported from it.

Writes results/figures/netto_predicted_vs_lab_by_shade.svg (and a PNG to $PREVIEW_DIR if set). Does not write any tables.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from netto_model import FIGS, MODELS, load, metrics, subsets


def figure(df, table):
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
    BC = {1: "#2a78d6", 2: "#eb6834"}; BM = {1: "o", 2: "s"}
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})
    fig = plt.figure(figsize=(16, 11.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.5, wspace=0.36, left=0.08, right=0.925, top=0.86, bottom=0.1)
    fig.text(0.08, 0.972, "Netto et al. (2005) SPAD models vs laboratory chlorophyll, fitted separately by shade block", fontsize=24, fontweight="bold", va="top")
    fig.text(0.08, 0.925, "Predicted = equation applied to each sample's SPAD (n = 270). The equations' units differ from the lab file, so one fitted line per block (n = 135 each) is shown.",
             fontsize=15, color=INK2, va="top")

    def style(ax):
        ax.set_facecolor(SURF); ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
        ax.tick_params(length=0, labelsize=14)

    for j, k in enumerate(("CHL_A", "CHL_B")):
        ax = fig.add_subplot(gs[0, j]); style(ax)
        x, y = df["PRED_" + k].values, df["LAB_" + k[-1]].values
        for b in (1, 2):
            m = (df.Block == b).values
            ax.scatter(x[m], y[m], s=44, marker=BM[b], color=BC[b], alpha=0.75, linewidths=0, zorder=3)
        FL = {1: (0, (5, 3)), 2: "-"}; pts, fits = [], []
        for b, lab in ((1, "B1 · outside shade"), (2, "B2 · under panel")):
            m = (df.Block == b).values
            lr = stats.linregress(x[m], y[m]); xs = np.array([x[m].min(), x[m].max()])
            ax.plot(xs, lr.intercept + lr.slope * xs, color=SURF, lw=5.5, zorder=4)          # halo
            ax.plot(xs, lr.intercept + lr.slope * xs, color=BC[b], lw=2.6, ls=FL[b], zorder=5)
            pts.append(Line2D([], [], color=BC[b], marker=BM[b], ls="", ms=9, label=lab))
            fits.append(Line2D([], [], color=BC[b], lw=2.6, ls=FL[b],
                               label=f"B{b} fit: r = {lr.rvalue:.2f}, R² = {lr.rvalue**2:.2f}\n"
                                     f"y = {lr.slope:.3f}x {'+' if lr.intercept >= 0 else '−'} {abs(lr.intercept):.1f}"))
        ax.set_ylim(0, y.max() * 2.0)
        ax.legend(handles=pts + fits, loc="upper left", frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9,
                  fontsize=12.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.35, handlelength=2.2)
        nm = MODELS[k][0]
        ax.set_title(f"{nm}: predicted vs lab", loc="left", fontsize=18, fontweight="bold", pad=8)
        ax.set_xlabel(f"{nm} predicted from SPAD (equation units)", fontsize=15); ax.set_ylabel(f"{nm} laboratory value", fontsize=15)

    ax = fig.add_subplot(gs[0, 2]); style(ax)
    sp = np.linspace(df.SPAD.min(), df.SPAD.max(), 200)
    for k, col, ls in (("CHL_A", "#2a78d6", "-"), ("CHL_B", "#eb6834", "-"), ("CAROT", "#1baf7a", "-")):
        a, b, c = MODELS[k][1]
        ax.plot(sp, a + b * sp + c * sp ** 2, color=col, lw=2.8, label=MODELS[k][0])
    ax.set_title("The three Netto equations", loc="left", fontsize=18, fontweight="bold", pad=8)
    ax.set_xlabel("SPAD value", fontsize=15); ax.set_ylabel("Predicted pigment (equation units)", fontsize=15)
    ax.legend(loc="upper left", frameon=False, fontsize=13.5)

    ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(SURF)
    subs = [s for s, _ in subsets(df)]
    M = np.array([[table[(table.subset == s) & (table.pigment == p)].r.values[0] for s in subs] for p in ("Chlorophyll a", "Chlorophyll b")])
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88a80", "#f0efec", "#6da7ec", "#184f95"])
    im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=17, fontweight="bold", color="#ffffff" if abs(M[i, j]) > 0.62 else INK)
    ax.set_xticks(range(len(subs))); ax.set_xticklabels([{"All samples": "All\nsamples", "B1 outside shade": "B1\noutside", "B2 under panel": "B2\nunder panel", "T1 · 100%": "T1\n100%", "T2 · 75%": "T2\n75%", "T3 · 50%": "T3\n50%"}.get(s, s) for s in subs], fontsize=14)
    ax.set_yticks(range(2)); ax.set_yticklabels(["Chl a", "Chl b"], fontsize=15)
    ax.set_xticks(np.arange(-.5, len(subs), 1), minor=True); ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
    ax.grid(which="minor", color=SURF, lw=4); ax.tick_params(which="both", length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Correlation (r) between predicted and laboratory value, by subset", loc="left", fontsize=18, fontweight="bold", pad=10)
    ax.set_xlabel("Subset of samples", fontsize=15, labelpad=8); ax.set_ylabel("Pigment", fontsize=15, labelpad=8)
    cb = fig.colorbar(im, cax=ax.inset_axes([1.025, 0.0, 0.015, 1.0])); cb.outline.set_visible(False)
    cb.set_label("Pearson r (legend)", fontsize=14); cb.ax.tick_params(length=0, labelsize=12)
    fig.savefig(FIGS / "netto_predicted_vs_lab_by_shade.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "netto_predicted_vs_lab_by_shade.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)



if __name__ == "__main__":
    df = load()
    rec = []
    for name, d in subsets(df):
        for k in ("CHL_A", "CHL_B"):
            m = metrics(d, k, "LAB_" + k[-1]); m.update(subset=name, pigment=MODELS[k][0]); rec.append(m)
    table = pd.DataFrame(rec)
    figure(df, table)
    print(table[table.subset.str.startswith("B")][["subset", "pigment", "n", "r", "R2", "slope", "intercept"]].round(3).to_string(index=False))
