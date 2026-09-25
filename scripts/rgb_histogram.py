"""Histogram analysis of the leaf RGB values (leaf pixels only, same definition as extract_rgb.py).

Writes Leaf_RGB_Histogram.xlsx (per-image statistics + group summaries), rgb_histogram_by_cycle.png and
rgb_histogram_by_group.png.
"""
import os
import re
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / "data"                                        # measurement tables
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
SRC = ROOT / "data_preprocessed"
CH = ["R", "G", "B"]
CHCOL = {"R": "#d6453d", "G": "#3a9d5d", "B": "#3b6fd4"}     # channel colours (an RGB histogram is read by its own colours)


def leaf_hist(path):
    bgra = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    hsv = cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2HSV)
    leaf = cv2.erode((bgra[..., 3] == 255).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    leaf &= ~(cv2.dilate(cv2.inRange(hsv, (3, 90, 90), (22, 255, 255)), np.ones((3, 3), np.uint8)) > 0)
    out = {}
    for name, k in zip(CH, (2, 1, 0)):                       # OpenCV order is B, G, R
        v = bgra[..., k][leaf]
        out[name] = np.bincount(v, minlength=256).astype(float)
    return out, int(leaf.sum())


def work(args):
    cyc, f = args
    h, n = leaf_hist(f)
    return cyc, f.stem, h, n


def stats_from_hist(h):
    x = np.arange(256); p = h / h.sum()
    mean = (x * p).sum(); sd = np.sqrt(((x - mean) ** 2 * p).sum())
    skew = ((x - mean) ** 3 * p).sum() / sd ** 3
    kurt = ((x - mean) ** 4 * p).sum() / sd ** 4 - 3
    median = x[np.searchsorted(np.cumsum(p), 0.5)]
    mode = int(np.argmax(gaussian_filter1d(h, 2)))
    return mean, median, mode, sd, skew, kurt


def main():
    jobs = [(int(d.name.split("-")[1]), f) for d in sorted(SRC.glob("cycle-*")) for f in sorted(d.glob("*.png"))]
    with Pool(8) as pool:
        res = pool.map(work, jobs)
    rows, H = [], {}
    spad = openpyxl.load_workbook(DATA / "SPAD_Data.xlsx", data_only=True)
    sp = {(c, sw.cell(r, 2).value): sw.cell(r, 3).value for c, sw in enumerate(spad.worksheets, 1) for r in range(3, sw.max_row + 1)}
    lab = {}
    for r in openpyxl.load_workbook(DATA / "Chl_Lab_Data.xlsx", data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
        if r[1] is not None: lab[(int(r[0]), r[1])] = r[2:5]
    for cyc, s, h, n in res:
        t, b = re.match(r"T(\d)B(\d)", s).groups()
        d = {"Harvest": cyc, "Sample": s, "T": int(t), "Block": int(b), "Leaf pixels": n}
        for c in CH:
            m, med, mo, sd, sk, ku = stats_from_hist(h[c])
            d.update({f"{c} mean": m, f"{c} median": med, f"{c} mode": mo, f"{c} std": sd, f"{c} skew": sk, f"{c} kurtosis": ku})
        d["SPAD"] = sp[(cyc, s)]; d["CHL_Total"] = lab[(cyc, s)][2]
        rows.append(d); H[(cyc, s)] = {c: h[c] / h[c].sum() for c in CH}
    df = pd.DataFrame(rows)

    def group_density(mask_fn, ch):
        keys = [k for k in H if mask_fn(k)]
        return np.mean([H[k][ch] for k in keys], axis=0), len(keys)     # every image counts equally

    # ---------------- figures
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})
    plt.rcParams["svg.fonttype"] = "path"
    x = np.arange(256)

    def style(ax, ylabel=False, xlabel=False):
        ax.set_facecolor(SURF); ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#b9b8b2")
        ax.tick_params(length=0, labelsize=13); ax.set_xlim(0, 255)
        if ylabel: ax.set_ylabel("Share of leaf pixels (%)", fontsize=15)
        if xlabel: ax.set_xlabel("Pixel value (0–255)", fontsize=15)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), facecolor=SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.075, hspace=0.42, wspace=0.26)
    fig.text(0.065, 0.965, "Histogram of the leaf RGB values, by harvest cycle", fontsize=25, fontweight="bold", va="top")
    fig.text(0.065, 0.915, "Leaf pixels only (background and rubber bands removed). Each curve is the average of the per-image histograms, so every image counts equally.",
             fontsize=15, color=INK2, va="top")
    panels = [(f"Harvest cycle {c}", (lambda k, c=c: k[0] == c)) for c in range(1, 6)] + [("All cycles", lambda k: True)]
    for ax, (title, fn) in zip(axes.flat, panels):
        for c in CH:
            d, n = group_density(fn, c)
            ax.plot(x, gaussian_filter1d(d, 1.5) * 100, color=CHCOL[c], lw=2.6, label=c)
        ax.set_title(f"{title}  (n = {n})", loc="left", fontsize=17, fontweight="bold", pad=8)
        style(ax, ylabel=True, xlabel=True)
        ax.legend(handles=[plt.Line2D([], [], color=CHCOL[c], lw=3.2, label=f"{c} channel") for c in CH], loc="upper left", frameon=False, fontsize=14)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.42)
    fig.savefig(FIGS / "rgb_histogram_by_cycle.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"): fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "rgb_histogram_by_cycle.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
    plt.close(fig)

    # by shade block and irrigation treatment
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.5), facecolor=SURF)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.85, bottom=0.075, hspace=0.42, wspace=0.26)
    fig.text(0.065, 0.965, "Histogram of the leaf RGB values, by shade and irrigation", fontsize=25, fontweight="bold", va="top")
    fig.text(0.065, 0.915, "Top row: outside vs under the solar panel. Bottom row: irrigation treatments. One panel per colour channel; all cycles pooled.",
             fontsize=15, color=INK2, va="top")
    BL = {1: ("B1 · outside shade", "#2a78d6"), 2: ("B2 · under panel", "#eb6834")}
    TL = {1: ("T1 · 100%", "#2a78d6"), 2: ("T2 · 75%", "#eb6834"), 3: ("T3 · 50%", "#1baf7a")}
    sample_info = {(r.Harvest, r.Sample): (r.T, r.Block) for r in df.itertuples()}
    for j, c in enumerate(CH):
        for i, (groups, idx) in enumerate(((BL, 1), (TL, 0))):
            ax = axes[i, j]
            for g, (lab, col) in groups.items():
                d, n = group_density(lambda k, g=g, idx=idx: sample_info[k][idx] == g, c)
                ax.plot(x, gaussian_filter1d(d, 1.5) * 100, color=col, lw=2.6, label=lab)
            ax.set_title(f"{c} channel · {'by shade' if i == 0 else 'by irrigation'}", loc="left", fontsize=17, fontweight="bold", pad=8)
            style(ax, ylabel=True, xlabel=True)
    for i, groups in enumerate((BL, TL)):
        h = [plt.Line2D([], [], color=col, lw=3.2, label=lab) for lab, col in groups.values()]
        for j in range(3):
            axes[i, j].legend(handles=h, loc="upper left", frameon=False, fontsize=14)
            axes[i, j].set_ylim(0, axes[i, j].get_ylim()[1] * 1.42)
    fig.savefig(FIGS / "rgb_histogram_by_group.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"): fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "rgb_histogram_by_group.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
    plt.close(fig)

    # ---------------- Excel
    wb = openpyxl.Workbook(); wb.remove(wb.active); bold = Font(bold=True)
    ws = wb.create_sheet("Per image"); cols = list(df.columns); ws.append(cols)
    for _, r in df.iterrows(): ws.append([int(r[c]) if c in ("Harvest", "T", "Block", "Leaf pixels") else (r[c] if isinstance(r[c], str) else round(float(r[c]), 3)) for c in cols])
    for c in ws[1]: c.font = bold; c.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.freeze_panes = "C2"
    for i in range(1, len(cols) + 1): ws.column_dimensions[get_column_letter(i)].width = 11
    stat_cols = [f"{c} {s}" for c in CH for s in ("mean", "std", "skew", "mode")]
    for name, key in (("By cycle", "Harvest"), ("By shade block", "Block"), ("By irrigation", "T")):
        g = df.groupby(key)[stat_cols].mean().round(3)
        ws = wb.create_sheet(name); ws.append([key] + stat_cols)
        for c in ws[1]: c.font = bold; c.alignment = Alignment(horizontal="center", wrap_text=True)
        for k, r in g.iterrows(): ws.append([int(k)] + list(r.values))
        for i in range(1, len(stat_cols) + 2): ws.column_dimensions[get_column_letter(i)].width = 11
    # link of the histogram shape to SPAD / chlorophyll (within harvest cycle)
    cen = df.copy()
    num = [c for c in stat_cols + ["R median", "G median", "B median", "R kurtosis", "G kurtosis", "B kurtosis", "SPAD", "CHL_Total"]]
    for c in num: cen[c] = df[c] - df.groupby("Harvest")[c].transform("mean")
    ws = wb.create_sheet("Link to SPAD & Chl"); ws.append(["Pearson r within harvest cycles (each cycle's mean removed), n = %d" % len(df)]); ws["A1"].font = bold
    ws.append(["Statistic", "SPAD", "Total Chl"]); [setattr(c, "font", bold) for c in ws[2]]
    link = {}
    for c in [f"{ch} {s}" for ch in CH for s in ("mean", "std", "skew", "kurtosis", "mode")]:
        link[c] = (stats.pearsonr(cen[c], cen["SPAD"]).statistic, stats.pearsonr(cen[c], cen["CHL_Total"]).statistic)
        ws.append([c, round(link[c][0], 3), round(link[c][1], 3)])
    ws.column_dimensions["A"].width = 18
    wb.save(TABLES / "Leaf_RGB_Histogram.xlsx")
    print(df.groupby("Harvest")[["R mean", "G mean", "B mean", "R std", "G std", "B std", "R skew", "G skew", "B skew"]].mean().round(2).to_string())
    print(df.groupby("Block")[["R mean", "G mean", "B mean", "R std", "G std", "B std"]].mean().round(2).to_string())
    print(df.groupby("T")[["R mean", "G mean", "B mean", "R std", "G std", "B std"]].mean().round(2).to_string())
    print("\nwithin-cycle r (SPAD, Total Chl):"); [print(f"  {k:12s} {v[0]:6.2f} {v[1]:6.2f}") for k, v in sorted(link.items(), key=lambda kv: -abs(kv[1][0]))[:8]]
    print("pixels per image: min %d median %d" % (df["Leaf pixels"].min(), df["Leaf pixels"].median()))


if __name__ == "__main__":
    main()
