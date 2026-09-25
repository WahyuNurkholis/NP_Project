"""Chlorophyll a, chlorophyll b and carotenoids predicted from SPAD with the Netto et al. (2005) models, compared with the lab data.

  Chl a       = 15.5866 + 1.0338*SPAD + 0.0679*SPAD^2
  Chl b       = 30.1471 - 0.4592*SPAD + 0.027*SPAD^2
  Carotenoids = 42.6458 - 0.8595*SPAD + 0.021*SPAD^2

Writes results/tables/Netto_Model_Predictions.xlsx and results/figures/netto_predicted_vs_lab.svg.
The equations' units differ from the lab file (predictions are ~4-9x larger), so besides the raw error the
comparison is also given after a linear rescale of the prediction onto the lab scale.
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"

MODELS = {"CHL_A": ("Chlorophyll a", (15.5866, 1.0338, 0.0679)),
          "CHL_B": ("Chlorophyll b", (30.1471, -0.4592, 0.027)),
          "CAROT": ("Carotenoids", (42.6458, -0.8595, 0.021))}
LAB = {"CHL_A": "Chlorophyll a", "CHL_B": "Chlorophyll b"}      # carotenoids were not measured in the lab file


def load():
    spad = {}
    for c, ws in enumerate(openpyxl.load_workbook(DATA / "SPAD_Data.xlsx", data_only=True).worksheets, 1):
        for r in range(3, ws.max_row + 1):
            spad[(c, ws.cell(r, 2).value)] = ws.cell(r, 3).value
    rows = []
    for r in openpyxl.load_workbook(DATA / "Chl_Lab_Data.xlsx", data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
        if r[1] is None:
            continue
        t, b = re.match(r"T(\d)B(\d)", r[1]).groups()
        rows.append(dict(Harvest=int(r[0]), Sample=r[1], T=int(t), Block=int(b), SPAD=spad[(int(r[0]), r[1])],
                         LAB_A=r[2], LAB_B=r[3], LAB_Total=r[4]))
    df = pd.DataFrame(rows)
    for k, (_, (a, b, c)) in MODELS.items():
        df["PRED_" + k] = a + b * df.SPAD + c * df.SPAD ** 2
    return df


def metrics(d, k, labcol):
    x, y = d["PRED_" + k].values, d[labcol].values
    lr = stats.linregress(x, y)
    return dict(n=len(d), r=lr.rvalue, R2=lr.rvalue ** 2, spearman=stats.spearmanr(x, y).statistic,
                ratio=x.mean() / y.mean(), RMSE_raw=np.sqrt(((x - y) ** 2).mean()),
                slope=lr.slope, intercept=lr.intercept, RMSE_rescaled=np.sqrt(((y - (lr.intercept + lr.slope * x)) ** 2).mean()))


def loco(df, k, labcol):
    """Rescale learnt on 4 harvest cycles, applied to the 5th."""
    pred = np.zeros(len(df))
    for h in sorted(df.Harvest.unique()):
        te = df.Harvest == h
        lr = stats.linregress(df.loc[~te, "PRED_" + k], df.loc[~te, labcol])
        pred[te.values] = lr.intercept + lr.slope * df.loc[te, "PRED_" + k]
    y = df[labcol].values
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt(((y - pred) ** 2).mean())


def subsets(df):
    out = [("All samples", df)]
    out += [(f"B{b} " + ("outside shade" if b == 1 else "under panel"), df[df.Block == b]) for b in (1, 2)]
    out += [(f"T{t} · {['100', '75', '50'][t - 1]}%", df[df["T"] == t]) for t in (1, 2, 3)]
    out += [(f"Cycle {h}", df[df.Harvest == h]) for h in sorted(df.Harvest.unique())]
    return out


def write_excel(df, table):
    wb = openpyxl.Workbook(); wb.remove(wb.active); bold = Font(bold=True)
    ws = wb.create_sheet("Predictions")
    cols = ["Harvest", "Sample", "T", "Block", "SPAD", "PRED_CHL_A", "PRED_CHL_B", "PRED_CAROT", "LAB_A", "LAB_B", "LAB_Total"]
    ws.append(["Harvest", "Sample", "T", "Block", "SPAD", "Chl a (Netto)", "Chl b (Netto)", "Carotenoids (Netto)", "Lab Chl a", "Lab Chl b", "Lab Total Chl"])
    for _, r in df.iterrows():
        ws.append([int(r.Harvest), r.Sample, int(r["T"]), int(r.Block)] + [round(float(r[c]), 4) for c in cols[4:]])
    for c in ws[1]: c.font = bold; c.alignment = Alignment(horizontal="center", wrap_text=True)
    ws.freeze_panes = "C2"
    for i in range(1, 12): ws.column_dimensions[get_column_letter(i)].width = 13
    ws = wb.create_sheet("Comparison")
    ws.append(["Netto et al. (2005) prediction vs laboratory value, by subset of samples"]); ws["A1"].font = bold
    hdr = ["Subset", "Pigment", "n", "Pearson r", "R2", "Spearman rho", "Mean predicted / mean lab", "RMSE raw", "Rescale slope", "Rescale intercept", "RMSE after rescale"]
    ws.append(hdr)
    for c in ws[2]: c.font = bold; c.alignment = Alignment(horizontal="center", wrap_text=True)
    for _, r in table.iterrows():
        ws.append([r.subset, r.pigment, int(r.n)] + [round(float(r[k]), 3) for k in ("r", "R2", "spearman", "ratio", "RMSE_raw", "slope", "intercept", "RMSE_rescaled")])
    for i in range(1, 12): ws.column_dimensions[get_column_letter(i)].width = 15
    ws.column_dimensions["A"].width = 20
    ws.conditional_formatting.add(f"D3:D{ws.max_row}", ColorScaleRule(start_type="num", start_value=-1, start_color="F4A582", mid_type="num", mid_value=0, mid_color="F7F7F7", end_type="num", end_value=1, end_color="4393C3"))
    ws = wb.create_sheet("Cross-cycle check")
    ws.append(["Rescale learnt on 4 harvest cycles, tested on the 5th (leave-one-cycle-out). R2 < 0 = worse than predicting the mean."]); ws["A1"].font = bold
    ws.append(["Pigment", "R2", "RMSE"]); [setattr(c, "font", bold) for c in ws[2]]
    for k, (nm, _) in MODELS.items():
        if k in LAB:
            r2, rm = loco(df, k, "LAB_" + k[-1]); ws.append([nm, round(r2, 3), round(rm, 3)])
    ws = wb.create_sheet("Equations")
    ws.append(["Pigment", "Equation (Netto et al., 2005)"]); [setattr(c, "font", bold) for c in ws[1]]
    for k, (nm, (a, b, c)) in MODELS.items():
        ws.append([nm, f"{a} {'+' if b >= 0 else '-'} {abs(b)} * SPAD {'+' if c >= 0 else '-'} {abs(c)} * SPAD^2"])
    ws.append([]); ws.append(["Notes"]); ws[f"A{ws.max_row}"].font = bold
    for n in ["The equations' pigment units differ from the lab file: predicted values are ~9x (Chl a) and ~4x (Chl b) the lab values on average.",
              "No carotenoid measurements exist in the lab file, so carotenoid predictions cannot be validated.",
              "Chl b is a quadratic with its minimum at SPAD 8.5, so over the measured SPAD range (17-59) all three curves are monotonic in SPAD."]:
        ws.append([n])
    ws.column_dimensions["A"].width = 16; ws.column_dimensions["B"].width = 70
    wb.save(TABLES / "Netto_Model_Predictions.xlsx")


def figure(df, table):
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#3d3c39", "#e2e1dc"
    BC = {1: "#2a78d6", 2: "#eb6834"}; BM = {1: "o", 2: "s"}
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "font.size": 15})
    fig = plt.figure(figsize=(16, 11.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.9], hspace=0.5, wspace=0.36, left=0.08, right=0.925, top=0.86, bottom=0.1)
    fig.text(0.08, 0.972, "Netto et al. (2005) SPAD models vs laboratory chlorophyll", fontsize=25, fontweight="bold", va="top")
    fig.text(0.08, 0.925, "Predicted = equation applied to each sample's SPAD (n = 270). The equations' units differ from the lab file, so the fitted line, not the 1:1 line, is shown.",
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
        lr = stats.linregress(x, y); xs = np.array([x.min(), x.max()])
        ax.plot(xs, lr.intercept + lr.slope * xs, color=INK, lw=2.2, ls=(0, (5, 3)), zorder=4)
        ax.text(0.04, 0.96, f"r = {lr.rvalue:.2f}\nR² = {lr.rvalue**2:.2f}", transform=ax.transAxes, va="top", fontsize=17, fontweight="bold", linespacing=1.25)
        ax.set_ylim(0, y.max() * 1.55)
        ax.legend(handles=[Line2D([], [], color=BC[b], marker=BM[b], ls="", ms=9, label=lab) for b, lab in ((1, "B1 · outside shade"), (2, "B2 · under panel"))] +
                  [Line2D([], [], color=INK, lw=2.2, ls=(0, (5, 3)), label="Linear fit")], loc="upper right", frameon=False, fontsize=13)
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
    fig.savefig(FIGS / "netto_predicted_vs_lab.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "netto_predicted_vs_lab.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)


if __name__ == "__main__":
    df = load()
    rec = []
    for name, d in subsets(df):
        for k in ("CHL_A", "CHL_B"):
            m = metrics(d, k, "LAB_" + k[-1]); m.update(subset=name, pigment=MODELS[k][0]); rec.append(m)
    table = pd.DataFrame(rec)
    write_excel(df, table)
    figure(df, table)
    print(table[table.subset == "All samples"].round(3).to_string())
    print(table.pivot(index="subset", columns="pigment", values="r").round(2).loc[[s for s, _ in subsets(df)]].to_string())
    for k in ("CHL_A", "CHL_B"):
        print(MODELS[k][0], "leave-one-cycle-out after rescale: R2=%.3f RMSE=%.2f" % loco(df, k, "LAB_" + k[-1]))
    print("earlier plain SPAD-vs-lab r:", {k: round(stats.pearsonr(df.SPAD, df["LAB_" + k]).statistic, 3) for k in ("A", "B")})
