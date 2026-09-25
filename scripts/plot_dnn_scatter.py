"""Scatter of the neural network (MLP) predictions against the observed values.

dnn_scatter_honest.svg : trained on four harvest cycles, predicting the fifth (leave-one-cycle-out).
dnn_scatter_leaky.svg  : random 5-fold split, where repeats of the same plot sit in both training and test sets.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dnn_feasibility as dnf               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
CYC = {1: ("#2a78d6", "o"), 2: ("#eb6834", "s"), 3: ("#1baf7a", "^"), 4: ("#eda100", "D"), 5: ("#e87ba4", "v")}   # colour + shape
TARGETS = [("SPAD", "SPAD"), ("CHL_A", "Chlorophyll a"), ("CHL_B", "Chlorophyll b")]


def draw(results, scheme, title, subtitle, fname):
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), facecolor=SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.08, hspace=0.42, wspace=0.26)
    fig.text(0.07, 0.972, title, fontsize=25, fontweight="bold", va="top")
    fig.text(0.07, 0.925, subtitle, fontsize=15, color=INK2, va="top")
    for i, fs in enumerate(("Camera colour", "Spectra")):
        for j, (tgt, tl) in enumerate(TARGETS):
            ax = axes[i, j]; ax.set_facecolor(SURF)
            obs, pred, cyc = results[(fs, tgt, scheme)]
            for c, (col, mk) in CYC.items():
                m = cyc == c
                ax.scatter(pred[m], obs[m], s=46, marker=mk, color=col, alpha=0.75, linewidths=0, zorder=3)
            lo, hi = min(obs.min(), pred.min()), max(obs.max(), pred.max()); pad = (hi - lo) * 0.05
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK, lw=1.8, ls=(0, (5, 3)), zorder=4)
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + (hi - lo) * 0.36)
            r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
            ax.text(0.96, 0.05, f"R² = {r2:.2f}\nr = {np.corrcoef(pred, obs)[0, 1]:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=16, fontweight="bold", linespacing=1.25, bbox=dict(boxstyle="round,pad=0.25", fc=SURF, ec="none", alpha=0.9), zorder=5)
            ax.set_title(f"{tl} · {fs} (n = {len(obs)})", loc="left", fontsize=16.5, fontweight="bold", pad=8)
            ax.set_xlabel(f"{tl} predicted by the neural network", fontsize=14.5); ax.set_ylabel(f"{tl} observed", fontsize=15)
            ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
            ax.tick_params(length=0, labelsize=13.5)
            h = [Line2D([], [], color=CYC[c][0], marker=CYC[c][1], ls="", ms=8, label=f"Cycle {c}") for c in CYC] + \
                [Line2D([], [], color=INK, lw=1.8, ls=(0, (5, 3)), label="1:1 line")]
            ax.legend(handles=h, loc="upper left", frameon=False, fontsize=11.5, ncol=3, columnspacing=1.0, handletextpad=0.3)
    fig.savefig(FIGS / f"{fname}.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / f"{fname}.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
    plt.close(fig)


def main():
    df = dnf.build()
    sets = {"Camera colour": (df[[np.isfinite(v).all() for v in df["rgb"]]].reset_index(drop=True), "rgb"),
            "Spectra": (df[df.QC_flag == "ok"].reset_index(drop=True), "spec")}
    results = {}
    for fs, (d, col) in sets.items():
        X = np.vstack(d[col].values); g = d.Harvest.values
        for tgt, _ in TARGETS:
            y = d[tgt].values.astype(float)
            for scheme in ("cycle", "random"):
                results[(fs, tgt, scheme)] = (y, dnf.cv_predict(X, y, g, "reg", "mlp", scheme), g)
                o, p, _ = results[(fs, tgt, scheme)]
                print(fs, tgt, scheme, "R2 = %.2f" % (1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()), flush=True)
    draw(results, "cycle", "Neural network predictions vs observed, tested on an unseen harvest cycle",
         "Trained on four harvest cycles, predicting the fifth. Points on the dashed 1:1 line would be perfect predictions.", "dnn_scatter_honest")
    draw(results, "random", "Neural network predictions vs observed, random split (leaky)",
         "Random 5-fold split: repeats of the same plots appear in both training and test data, so the fit looks better than it really is.", "dnn_scatter_leaky")


if __name__ == "__main__":
    main()
