"""Paired irrigation effect: T2 and T3 compared with the well-watered T1 (same block, replication, repeat, cycle).

Rows = SPAD, chlorophyll a, chlorophyll b; columns = the two blocks. Points are the mean difference in % of the T1 mean with a 95 % confidence
interval; filled markers are significant (paired t-test, p < 0.05, not corrected for multiple tests).
Writes results/figures/paired_irrigation_effect.svg.
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import what_could_work as wcw               # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "results" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
COL = {2: "#eb6834", 3: "#1baf7a"}
MRK = {2: "s", 3: "^"}
LAB = {2: "T2 (75%) − T1", 3: "T3 (50%) − T1"}
TARGETS = [("SPAD", "SPAD"), ("CHL_A", "Chlorophyll a (lab)"), ("CHL_B", "Chlorophyll b (lab)")]
BLOCKS = {1: "B1 · outside panel shade", 2: "B2 · under solar panel"}


def effects(df):
    rec = {}
    for y, _ in TARGETS:
        for (h, b), d in df.groupby(["Harvest", "Block"]):
            p = d.pivot_table(index=["Rep", "X"], columns="T", values=y)
            for t in (2, 3):
                diff = (p[t] - p[1]).dropna()
                m, se = diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff))
                ci = stats.t.ppf(0.975, len(diff) - 1) * se
                base = p[1].mean()
                rec[(y, b, t, h)] = (100 * m / base, 100 * ci / base, stats.ttest_1samp(diff, 0).pvalue)
    return rec


def main():
    rec = effects(wcw.load())
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), facecolor=SURF)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.885, bottom=0.06, hspace=0.5, wspace=0.24)
    fig.text(0.08, 0.972, "Effect of reduced irrigation, paired with the well-watered plot (T1)", fontsize=25, fontweight="bold", va="top")
    fig.text(0.08, 0.93, "Difference in % of the T1 mean, matched by block, replication, repeat and harvest cycle; bars are 95% confidence intervals. "
                         "Below 0 = lower than T1.", fontsize=15, color=INK2, va="top")
    x = np.arange(1, 6)
    for i, (y, ylabel) in enumerate(TARGETS):
        for j, b in enumerate((1, 2)):
            ax = axes[i, j]; ax.set_facecolor(SURF)
            vals = []
            for t, off in ((2, -0.11), (3, 0.11)):
                m = np.array([rec[(y, b, t, h)][0] for h in x]); ci = np.array([rec[(y, b, t, h)][1] for h in x]); p = np.array([rec[(y, b, t, h)][2] for h in x])
                ax.plot(x + off, m, color=COL[t], lw=1.8, alpha=0.6, zorder=2)
                for k in range(5):
                    ax.errorbar(x[k] + off, m[k], yerr=ci[k], color=COL[t], lw=1.6, capsize=4, zorder=3)
                    sig = p[k] < 0.05
                    ax.plot(x[k] + off, m[k], marker=MRK[t], ms=11, mfc=COL[t] if sig else SURF, mec=COL[t], mew=2, zorder=4)
                vals += list(m - ci) + list(m + ci)
            ax.axhline(0, color=INK, lw=1.4, zorder=1)
            lo, hi = min(vals), max(vals); pad = (hi - lo) * 0.08
            ax.set_ylim(lo - pad, hi + (hi - lo) * 0.55)
            ax.set_xticks(x); ax.set_xticklabels([f"C{c}" for c in x]); ax.set_xlim(0.6, 5.4)
            ax.grid(axis="y", color=GRID, lw=1); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
            ax.tick_params(length=0, labelsize=14)
            ax.set_title(f"{ylabel} · {BLOCKS[b]}", loc="left", fontsize=17, fontweight="bold", pad=8)
            ax.set_xlabel("Harvest cycle", fontsize=15); ax.set_ylabel("Difference to T1 (% of T1 mean)", fontsize=15)
            h = [Line2D([], [], color=COL[t], marker=MRK[t], ms=10, mfc=COL[t], lw=1.8, label=LAB[t]) for t in (2, 3)]
            h += [Line2D([], [], color=INK2, marker="o", ms=10, mfc=INK2, ls="", label="filled: p < 0.05"),
                  Line2D([], [], color=INK2, marker="o", ms=10, mfc=SURF, mew=2, ls="", label="open: not significant")]
            ax.legend(handles=h, loc="upper left", frameon=False, fontsize=12.5, ncol=2, columnspacing=1.4, handletextpad=0.5)
    fig.savefig(FIGS / "paired_irrigation_effect.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "paired_irrigation_effect.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
    print({k: tuple(round(float(v), 1) if i < 2 else round(float(v), 3) for i, v in enumerate(rec[k])) for k in [("SPAD", 1, 3, 1), ("SPAD", 2, 3, 4), ("SPAD", 2, 3, 5), ("CHL_A", 1, 3, 1)]})


if __name__ == "__main__":
    main()
