"""RGB vegetation indices of the leaf (from Leaf_RGB_Data.xlsx) and their relation to SPAD and lab chlorophyll.

Writes Vegetation_Indices.xlsx (per-sample indices, correlation tables, cross-validation, formulas) and
vegetation_indices_correlation.png.
"""
import colorsys
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
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / "data"                                        # measurement tables
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
TARGETS = ["SPAD", "CHL_A", "CHL_B", "CHL_Total"]
TARGET_LABEL = {"SPAD": "SPAD", "CHL_A": "Chl a", "CHL_B": "Chl b", "CHL_Total": "Total Chl"}

# name -> (source, formula text).  "Reference" = the camera indices table supplied by the user;
# r, g, b are the normalized intensities R/(R+G+B), G/(R+G+B), B/(R+G+B) (Eqs. 4-6 of the reference).
DEFS = {
    "R": ("Channel", "mean red DN (0-255)"),
    "G": ("Channel", "mean green DN (0-255)"),
    "B": ("Channel", "mean blue DN (0-255)"),
    "r": ("Reference", "R / (R+G+B)   (Eq. 4)"),
    "g": ("Reference", "G / (R+G+B)   (Eq. 5)"),
    "b": ("Reference", "B / (R+G+B)   (Eq. 6)"),
    "NDVI rgb": ("Reference", "((g+b) - r) / ((g+b) + r)"),
    "NDVI green": ("Reference", "(g - r) / (g + r)"),
    "SAVI green": ("Reference", "(1+L)(g - r) / ((g + r) + L),  L = 0.5"),
    "NDI": ("Reference", "(g - r) / (g + r + 0.01)"),
    "GMR": ("Reference", "g - r"),
    "SR": ("Reference", "g / r"),
    "Hue (partial)": ("Reference", "120 + 60(B - R) / (max(R,G,B) - min(R,G,B))"),
    "DGCI": ("Reference", "[(Hue-60)/60 + (1 - Saturation/100) + (1 - Brightness/100)] / 3"),
    "VARI": ("Reference", "(g - r) / (g + r - b)"),
    "BRAVI": ("Reference", "N(g - r) / (g + r + N),  N = (r + b)/255"),
    "BRAVI-SR": ("Reference", "N g / (r + N)"),
    "EVI green": ("Reference", "2.5(g - r) / (g + 6r - 7.5b + 1)"),
    "EVI2 green": ("Reference", "2.5(g - r) / (g + 2.4r + 1)"),
    "OSAVI green": ("Reference", "1.5(g - r) / ((g + r) + 0.16)"),
    "SR rgb": ("Reference", "r / (g + b)"),
    "Saturation (%)": ("Additional", "(max - min) / max * 100 of the mean colour"),
    "Brightness (%)": ("Additional", "max(R,G,B) / 255 * 100"),
    "ExG": ("Additional", "2g - r - b   (excess green)"),
    "ExR": ("Additional", "1.4r - g   (excess red)"),
    "ExGR": ("Additional", "ExG - ExR"),
    "GLI": ("Additional", "(2G - R - B) / (2G + R + B)"),
    "MGRVI": ("Additional", "(G^2 - R^2) / (G^2 + R^2)"),
    "RGBVI": ("Additional", "(G^2 - B*R) / (G^2 + B*R)"),
    "CIVE": ("Additional", "0.441r - 0.811g + 0.385b + 18.78745"),
}
REFERENCE = [k for k, (src, _) in DEFS.items() if src == "Reference"]


def load():
    rgb = openpyxl.load_workbook(TABLES / "Leaf_RGB_Data.xlsx", data_only=True)
    spad = openpyxl.load_workbook(DATA / "SPAD_Data.xlsx", data_only=True)
    lab = {}
    for r in openpyxl.load_workbook(DATA / "Chl_Lab_Data.xlsx", data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
        if r[1] is not None:
            lab[(int(r[0]), r[1])] = r[2:5]
    rows = []
    for c, (ws, sw) in enumerate(zip(rgb.worksheets, spad.worksheets), 1):
        for r in range(2, ws.max_row + 1):
            s = ws.cell(r, 2).value
            R, G, B = (ws.cell(r, k).value for k in (3, 4, 5))
            if R is None:
                continue                                    # sample without an image
            assert sw.cell(r + 1, 2).value == s
            t, blk = re.match(r"T(\d)B(\d)", s).groups()
            a, b_, tot = lab[(c, s)]
            rows.append(dict(Harvest=c, Sample=s, T=int(t), Block=int(blk), R=R, G=G, B=B,
                             SPAD=sw.cell(r + 1, 3).value, CHL_A=a, CHL_B=b_, CHL_Total=tot))
    return pd.DataFrame(rows)


def add_indices(df):
    R, G, B = df.R, df.G, df.B
    s = R + G + B
    r, g, b = R / s, G / s, B / s
    mx, mn = np.maximum.reduce([R, G, B]), np.minimum.reduce([R, G, B])
    sat, bri = (mx - mn) / mx * 100, mx / 255 * 100
    hue = 120 + 60 * (B - R) / (mx - mn)
    N, L = (r + b) / 255, 0.5
    vi = {"r": r, "g": g, "b": b,
          "NDVI rgb": ((g + b) - r) / ((g + b) + r), "NDVI green": (g - r) / (g + r),
          "SAVI green": (1 + L) * (g - r) / ((g + r) + L), "NDI": (g - r) / (g + r + 0.01), "GMR": g - r, "SR": g / r,
          "Hue (partial)": hue, "DGCI": ((hue - 60) / 60 + (1 - sat / 100) + (1 - bri / 100)) / 3,
          "VARI": (g - r) / (g + r - b), "BRAVI": N * (g - r) / (g + r + N), "BRAVI-SR": N * g / (r + N),
          "EVI green": 2.5 * (g - r) / (g + 6 * r - 7.5 * b + 1), "EVI2 green": 2.5 * (g - r) / (g + 2.4 * r + 1),
          "OSAVI green": 1.5 * (g - r) / ((g + r) + 0.16), "SR rgb": r / (g + b),
          "Saturation (%)": sat, "Brightness (%)": bri,
          "ExG": 2 * g - r - b, "ExR": 1.4 * r - g, "ExGR": (2 * g - r - b) - (1.4 * r - g),
          "GLI": (2 * G - R - B) / (2 * G + R + B), "MGRVI": (G**2 - R**2) / (G**2 + R**2),
          "RGBVI": (G**2 - B * R) / (G**2 + B * R), "CIVE": 0.441 * r - 0.811 * g + 0.385 * b + 18.78745}
    for k, v in vi.items():
        df[k] = np.asarray(v, float)
    return df


def loco(X, y, groups):
    """Leave-one-harvest-cycle-out linear (ridge) regression: R2 and RMSE of the held-out predictions."""
    pred = np.zeros(len(y))
    for gr in np.unique(groups):
        te, tr = groups == gr, groups != gr
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        beta = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ y[tr])
        pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ beta
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt(((y - pred) ** 2).mean())


def analyse(df):
    names = list(DEFS)
    pooled = pd.DataFrame({t: [stats.pearsonr(df[k], df[t]).statistic for k in names] for t in TARGETS}, index=names)
    spear = pd.DataFrame({t: [stats.spearmanr(df[k], df[t]).statistic for k in names] for t in TARGETS}, index=names)
    cen = df.copy()
    for k in names + TARGETS:
        cen[k] = df[k] - df.groupby("Harvest")[k].transform("mean")          # remove each cycle's own level
    within = pd.DataFrame({t: [stats.pearsonr(cen[k], cen[t]).statistic for k in names] for t in TARGETS}, index=names)
    percycle = {t: pd.DataFrame({f"C{h}": [stats.pearsonr(df[k][df.Harvest == h], df[t][df.Harvest == h]).statistic for k in names]
                                 for h in range(1, 6)}, index=names) for t in ("SPAD", "CHL_Total")}
    cv = {}
    for label, cols in {"DGCI": ["DGCI"], "VARI": ["VARI"], "NDVI green": ["NDVI green"], "NDVI rgb": ["NDVI rgb"],
                        "EVI green": ["EVI green"], "Hue (partial)": ["Hue (partial)"], "r": ["r"], "SR rgb": ["SR rgb"],
                        "R + G + B": ["R", "G", "B"], "DGCI + VARI + Hue + r": ["DGCI", "VARI", "Hue (partial)", "r"]}.items():
        cv[label] = {t: loco(df[cols].values, df[t].values, df.Harvest.values) for t in TARGETS}
    return pooled, spear, within, percycle, cv


def write_excel(df, pooled, spear, within, percycle, cv):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    bold = Font(bold=True)

    def header(ws, row=1):
        for c in ws[row]:
            c.font = bold; c.alignment = Alignment(horizontal="center", wrap_text=True)

    ws = wb.create_sheet("Indices")
    cols = ["Harvest", "Sample", "T", "Block"] + list(DEFS) + TARGETS
    ws.append(cols)
    for _, r in df.iterrows():
        ws.append([int(r.Harvest), r.Sample, int(r["T"]), int(r.Block)] + [round(float(r[k]), 5) for k in list(DEFS) + TARGETS])
    header(ws); ws.freeze_panes = "C2"
    for i in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 12

    def corr_sheet(name, tables, title_note):
        ws = wb.create_sheet(name)
        ws.append([title_note]); ws["A1"].font = bold
        row = 3
        for title, tab in tables:
            ws.cell(row, 1, title).font = bold
            ws.append([]) if False else None
            hdr = ["Index"] + list(tab.columns)
            for j, h in enumerate(hdr, 1):
                c = ws.cell(row + 1, j, TARGET_LABEL.get(h, h)); c.font = bold; c.alignment = Alignment(horizontal="center")
            for i, (k, vals) in enumerate(tab.iterrows(), 1):
                ws.cell(row + 1 + i, 1, k)
                for j, v in enumerate(vals, 2):
                    c = ws.cell(row + 1 + i, j, round(float(v), 3)); c.number_format = "0.00"
            rng = f"B{row + 2}:{get_column_letter(len(hdr))}{row + 1 + len(tab)}"
            ws.conditional_formatting.add(rng, ColorScaleRule(start_type="num", start_value=-1, start_color="F4A582",
                                                              mid_type="num", mid_value=0, mid_color="F7F7F7",
                                                              end_type="num", end_value=1, end_color="4393C3"))
            row += len(tab) + 4
        ws.column_dimensions["A"].width = 16
        for i in range(2, 9): ws.column_dimensions[get_column_letter(i)].width = 11
        return ws

    corr_sheet("Correlation", [("Pearson r, all samples pooled (n = %d)" % len(df), pooled),
                               ("Spearman rho, all samples pooled", spear),
                               ("Pearson r within harvest cycles (each cycle's mean removed first)", within)],
               "Correlation of each RGB index with SPAD and the lab chlorophyll values")
    corr_sheet("Per cycle", [("Pearson r with SPAD, by harvest cycle", percycle["SPAD"]),
                             ("Pearson r with total chlorophyll, by harvest cycle", percycle["CHL_Total"])],
               "Correlation inside each harvest cycle (n = 54 per cycle, 53 in cycle 3)")

    ws = wb.create_sheet("Cross-validation")
    ws.append(["Leave-one-harvest-cycle-out linear regression: train on 4 cycles, predict the 5th. R2 < 0 = worse than predicting the mean."])
    ws["A1"].font = bold
    ws.append(["Predictors"] + [f"{TARGET_LABEL[t]} {m}" for t in TARGETS for m in ("R2", "RMSE")])
    header(ws, 2)
    for k, v in cv.items():
        ws.append([k] + [round(float(x), 3) for t in TARGETS for x in v[t]])
    ws.column_dimensions["A"].width = 24
    for i in range(2, 10): ws.column_dimensions[get_column_letter(i)].width = 13

    ws = wb.create_sheet("Formulas")
    ws.append(["Index", "Source", "Formula"]); header(ws)
    for k, (g, f) in DEFS.items(): ws.append([k, g, f])
    ws.append([]); ws.append(["Reference = camera indices from the user's reference table; r, g, b = normalized intensities; R, G, B = mean leaf colour (0-255) from Leaf_RGB_Data.xlsx."])
    for col, w in zip("ABC", (14, 14, 62)): ws.column_dimensions[col].width = w
    wb.save(TABLES / "Vegetation_Indices.xlsx")


def figure(pooled, within):
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#3d3c39"
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
                         "text.color": INK, "font.size": 15})
    order = pooled.loc[["r", "g", "b"] + REFERENCE[3:], "SPAD"].abs().sort_values(ascending=False).index.tolist()
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88a80", "#f0efec", "#6da7ec", "#184f95"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 11.5), facecolor=SURF)
    fig.subplots_adjust(left=0.115, right=0.93, top=0.83, bottom=0.075, wspace=0.62)
    fig.text(0.04, 0.975, "RGB vegetation indices vs SPAD and laboratory chlorophyll", fontsize=25, fontweight="bold", va="top")
    fig.text(0.04, 0.93, "Pearson r. Left: all samples pooled (n = 269). Right: inside each harvest cycle, removing every cycle's own mean\n"
                         "(so differences in lighting and growth stage between cycles cannot create the correlation).",
             fontsize=15, color=INK2, va="top", linespacing=1.3)
    for ax, tab, ttl in zip(axes, (pooled, within), ("Pooled", "Within harvest cycles")):
        M = tab.loc[order, TARGETS].values
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=13,
                        color="#ffffff" if abs(M[i, j]) > 0.62 else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels([TARGET_LABEL[t] for t in TARGETS], fontsize=15)
        ax.set_xlabel("Measured variable", fontsize=16, labelpad=8); ax.set_ylabel("Vegetation index", fontsize=16, labelpad=8)
        ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=14)
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
        ax.grid(which="minor", color=SURF, lw=2.5); ax.tick_params(which="both", length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(ttl, fontsize=19, fontweight="bold", pad=10, loc="left")
        cb = fig.colorbar(im, cax=ax.inset_axes([1.04, 0.0, 0.035, 1.0])); cb.outline.set_visible(False)
        cb.set_label("Pearson r (legend)", fontsize=15); cb.ax.tick_params(length=0, labelsize=13)
    fig.savefig(FIGS / "vegetation_indices_correlation.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"): fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "vegetation_indices_correlation.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)


if __name__ == "__main__":
    df = add_indices(load())
    pooled, spear, within, percycle, cv = analyse(df)
    write_excel(df, pooled, spear, within, percycle, cv)
    figure(pooled, within)
    print(len(df), "samples")
    print(within.round(2).assign(pooled_SPAD=pooled["SPAD"].round(2), pooled_Total=pooled["CHL_Total"].round(2)).sort_values("SPAD", key=abs, ascending=False).to_string())
