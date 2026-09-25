"""Smoothed mean spectra from Spectrometer_Data.xlsx.

Writes spectrometer_smoothed_by_irrigation.png and spectrometer_smoothed_by_shade.png:
one panel per harvest cycle, one smooth curve per group (mean of its samples, Savitzky-Golay smoothed).
Spectra flagged by spectrometer_analysis.py (cycle 5 T2/T3 copies, outliers) are left out of the means;
cycle 5 T2/T3 are drawn as dotted lines so they stay visible.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.signal import savgol_filter

import spectrometer_analysis as sa

ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / "data"                                        # measurement tables
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
GROUPS = {
    "irrigation": ("T", {1: ("T1 · 100% irrigation", "#2a78d6"), 2: ("T2 · 75%", "#eb6834"), 3: ("T3 · 50%", "#1baf7a")}),
    "shade": ("Block", {1: ("B1 · outside panel shade", "#2a78d6"), 2: ("B2 · under solar panel", "#eb6834")}),
}
CYCLE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def smooth(y, window=41):
    return savgol_filter(y, window, 3)


def main():
    spectra = sa.load_spectra()
    df = sa.quality_flags(sa.add_indices(sa.add_targets(sa.band_table(spectra))))
    ok = {(r.Harvest, r.Sample): r.QC_flag == "ok" for r in df.itertuples()}
    info = {(r.Harvest, r.Sample): (r.T, r.Block) for r in df.itertuples()}
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})

    for key, (col, groups) in GROUPS.items():
        fig, axes = plt.subplots(2, 3, figsize=(17, 11), facecolor=SURF)
        fig.subplots_adjust(left=0.06, right=0.985, top=0.85, bottom=0.085, hspace=0.46, wspace=0.26)
        fig.text(0.06, 0.965, "Spectrometer spectra, smoothed", fontsize=25, fontweight="bold", va="top")
        fig.text(0.06, 0.915, f"Mean reflectance (%) of each group per harvest cycle, 380–950 nm, smoothed with a Savitzky–Golay filter. Grey strips: bands used by the indices.",
                 fontsize=15, color=INK2, va="top")
        ymax = 0
        band_patch = Patch(facecolor="#b9b8b2", alpha=0.6, label="Index bands")
        base = [Line2D([], [], color=colr, lw=3, label=lab) for lab, colr in groups.values()] + [band_patch]
        for ax, c in zip(axes.flat, range(1, 6)):
            wl, cols = spectra[c]; ax.set_facecolor(SURF)
            for lo, hi in sa.BANDS.values(): ax.axvspan(lo, hi, color="#b9b8b2", alpha=0.35, lw=0)
            for g, (lab, colr) in groups.items():
                good = [np.nan_to_num(cols[n]) for n in cols if info[(c, n)][0 if col == "T" else 1] == g and ok[(c, n)]]
                if good:
                    y = smooth(np.mean(good, axis=0)); ax.plot(wl, y, color=colr, lw=2.6); ymax = max(ymax, y.max())
                elif c == 5 and col == "T":                     # excluded copies: keep visible, dotted
                    bad = [np.nan_to_num(cols[n]) for n in cols if info[(c, n)][0] == g and np.nanmax(cols[n]) > 5]
                    y = smooth(np.mean(bad, axis=0)); ax.plot(wl, y, color=colr, lw=2.2, ls=(0, (1.5, 2.5))); ymax = max(ymax, y.max())
            ax.set_title(f"Harvest cycle {c}", loc="left", fontsize=18, fontweight="bold", pad=8)
            ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
            for s in ("top", "right"): ax.spines[s].set_visible(False)
            for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
            ax.tick_params(length=0, labelsize=14)
            ax.set_ylabel("Reflectance (%)", fontsize=16); ax.set_xlabel("Wavelength (nm)", fontsize=16); ax.set_xlim(380, 950)
            hh = base[:-1] + ([Line2D([], [], color="#52514e", lw=2.4, ls=(0, (1.5, 2.5)), label="T2, T3: copies, excluded")] if (c == 5 and key == "irrigation") else []) + [band_patch]
            ax.legend(handles=hh, loc="upper right", frameon=False, fontsize=13, handlelength=1.8)
        ax = axes.flat[5]; ax.set_facecolor(SURF)
        for lo, hi in sa.BANDS.values(): ax.axvspan(lo, hi, color="#b9b8b2", alpha=0.35, lw=0)
        for c in range(1, 6):
            wl, cols = spectra[c]
            good = [np.nan_to_num(cols[n]) for n in cols if ok[(c, n)]]
            ax.plot(wl, smooth(np.mean(good, axis=0)), color=CYCLE_COLORS[c - 1], lw=2.6, label=f"Cycle {c}")
        ax.set_title("All groups, by cycle", loc="left", fontsize=18, fontweight="bold", pad=8)
        ax.legend(handles=ax.get_legend_handles_labels()[0] + [band_patch], loc="upper right", frameon=False, fontsize=13.5, ncol=1, handlelength=1.6)
        ax.grid(color=GRID, lw=1); ax.set_axisbelow(True); ax.set_xlabel("Wavelength (nm)", fontsize=16); ax.set_ylabel("Reflectance (%)", fontsize=16); ax.set_xlim(380, 950)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
        ax.tick_params(length=0, labelsize=14)
        for a in axes.flat: a.set_ylim(0, ymax * 1.5)
        fig.savefig(FIGS / f"spectrometer_smoothed_by_{key}.svg", facecolor=SURF)
        if os.environ.get("PREVIEW_DIR"): fig.savefig(Path(os.environ["PREVIEW_DIR"]) / f"spectrometer_smoothed_by_{key}.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
        plt.close(fig)


if __name__ == "__main__":
    main()
