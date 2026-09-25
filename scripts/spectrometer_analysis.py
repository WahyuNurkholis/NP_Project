"""Spectrometer vegetation indices (SRVI, SRRE, NDRE, NDVI, CCCI) and their relation to SPAD and lab chlorophyll.

Bands (mean reflectance over the range, nm): Red 660-675, Red edge 690-740, NIRb 770-785, NIRa 800-810.
Writes Spectrometer_Indices.xlsx and spectrometer_indices_correlation.png.
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
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / "data"                                        # measurement tables
TABLES = ROOT / "results" / "tables"
FIGS = ROOT / "results" / "figures"
BANDS = {"Red": (660, 675), "RedEdge": (690, 740), "NIRb": (770, 785), "NIRa": (800, 810)}
INDICES = ["SRVI", "SRRE", "NDRE", "NDVI", "CCCI"]
TARGETS = ["SPAD", "CHL_A", "CHL_B", "CHL_Total"]
TLAB = {"SPAD": "SPAD", "CHL_A": "Chl a", "CHL_B": "Chl b", "CHL_Total": "Total Chl"}
FORMULAS = {"SRVI": "NIRb / Red", "SRRE": "NIRa / RedEdge", "NDRE": "(NIRa - Red) / (NIRa + Red)   (as printed in the reference table)",
            "NDVI": "(NIRb - Red) / (NIRb + Red)", "CCCI": "NDRE / NDVI"}


def load_spectra():
    wb = openpyxl.load_workbook(DATA / "Spectrometer_Data.xlsx", read_only=True, data_only=True)
    out = {}
    for c, ws in enumerate(wb.worksheets, 1):
        rows = list(ws.iter_rows(min_row=1, max_row=1028, max_col=55, values_only=True))
        wl = np.array([r[0] for r in rows[1:]], float)
        cols = {h: np.array([np.nan if r[j + 1] is None else r[j + 1] for r in rows[1:]], float) for j, h in enumerate(rows[0][1:])}
        out[c] = (wl, cols)
    return out


def band_table(spectra):
    rec = []
    for c, (wl, cols) in spectra.items():
        for name, y in cols.items():
            d = {"Harvest": c, "Sample": name}
            for b, (lo, hi) in BANDS.items():
                d[b] = float(y[(wl >= lo) & (wl <= hi)].mean())
            rec.append(d)
    return pd.DataFrame(rec)


def add_targets(df):
    spad = openpyxl.load_workbook(DATA / "SPAD_Data.xlsx", data_only=True)
    lab = {}
    for r in openpyxl.load_workbook(DATA / "Chl_Lab_Data.xlsx", data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
        if r[1] is not None:
            lab[(int(r[0]), r[1])] = r[2:5]
    sp = {(c, sw.cell(r, 2).value): sw.cell(r, 3).value for c, sw in enumerate(spad.worksheets, 1) for r in range(3, sw.max_row + 1)}
    df["SPAD"] = [sp[(c, s)] for c, s in zip(df.Harvest, df.Sample)]
    for j, t in enumerate(["CHL_A", "CHL_B", "CHL_Total"]):
        df[t] = [lab[(c, s)][j] for c, s in zip(df.Harvest, df.Sample)]
    df["T"] = [int(re.match(r"T(\d)", s).group(1)) for s in df.Sample]
    df["Block"] = [int(re.match(r"T\dB(\d)", s).group(1)) for s in df.Sample]
    return df


def add_indices(df):
    with np.errstate(divide="ignore", invalid="ignore"):
        df["SRVI"] = df.NIRb / df.Red
        df["SRRE"] = df.NIRa / df.RedEdge
        df["NDRE"] = (df.NIRa - df.Red) / (df.NIRa + df.Red)
        df["NDVI"] = (df.NIRb - df.Red) / (df.NIRb + df.Red)
        df["CCCI"] = df.NDRE / df.NDVI
    return df


def quality_flags(df):
    """Flag spectra that are not usable as independent leaf measurements. Reasons are recorded per sample."""
    reason = pd.Series("", index=df.index)
    # cycle 5, T2 and T3: near-identical spectra (red band 54.7 +/- 0.3 across the group), copies/averages of each
    # other in the source workbook, plus one blank spectrum -> not independent measurements
    m5 = (df.Harvest == 5) & (df["T"].isin([2, 3]))
    reason[m5] = "cycle 5 T2/T3: spectra are copies/averages of one another (near-constant), not independent leaf measurements"
    # gross outliers inside the remaining data: robust z-score (median/MAD) of log band means within each cycle
    for c in df.Harvest.unique():
        idx = df.index[(df.Harvest == c) & (reason == "")]
        for b in BANDS:
            v = np.log(df.loc[idx, b].clip(lower=1e-6))
            z = (v - v.median()) / (1.4826 * (v - v.median()).abs().median() + 1e-12)
            bad = idx[np.abs(z) > 4]
            for i in bad:
                if reason[i] == "":
                    reason[i] = f"outlier spectrum ({b} band far from the cycle median)"
    df["QC_flag"] = np.where(reason == "", "ok", "excluded")
    df["QC_reason"] = reason
    return df


def corr_tables(d):
    pooled = pd.DataFrame({t: [stats.pearsonr(d[k], d[t]).statistic for k in INDICES] for t in TARGETS}, index=INDICES)
    spear = pd.DataFrame({t: [stats.spearmanr(d[k], d[t]).statistic for k in INDICES] for t in TARGETS}, index=INDICES)
    cen = d.copy()
    for k in INDICES + TARGETS:
        cen[k] = d[k] - d.groupby("Harvest")[k].transform("mean")
    within = pd.DataFrame({t: [stats.pearsonr(cen[k], cen[t]).statistic for k in INDICES] for t in TARGETS}, index=INDICES)
    per = {}
    for t in ("SPAD", "CHL_Total"):
        per[t] = pd.DataFrame({f"C{h}": [stats.pearsonr(d[k][d.Harvest == h], d[t][d.Harvest == h]).statistic if (d.Harvest == h).sum() > 5 else np.nan
                                         for k in INDICES] for h in sorted(d.Harvest.unique())}, index=INDICES)
    return pooled, spear, within, per


def loco(X, y, groups):
    pred = np.zeros(len(y))
    for gr in np.unique(groups):
        te, tr = groups == gr, groups != gr
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        Z = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd]
        beta = np.linalg.solve(Z.T @ Z + 1e-3 * np.eye(Z.shape[1]), Z.T @ y[tr])
        pred[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sd] @ beta
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum(), np.sqrt(((y - pred) ** 2).mean())


def write_excel(df, ok, pooled, spear, within, per, pooled_all, cv):
    wb = openpyxl.Workbook(); wb.remove(wb.active)
    bold = Font(bold=True)

    ws = wb.create_sheet("Indices")
    cols = ["Harvest", "Sample", "T", "Block"] + list(BANDS) + INDICES + TARGETS + ["QC_flag", "QC_reason"]
    ws.append(cols)
    for _, r in df.iterrows():
        ws.append([int(r.Harvest), r.Sample, int(r["T"]), int(r.Block)] +
                  [None if pd.isna(r[k]) or np.isinf(r[k]) else round(float(r[k]), 5) for k in list(BANDS) + INDICES + TARGETS] + [r.QC_flag, r.QC_reason])
    for c in ws[1]:
        c.font = bold; c.alignment = Alignment(horizontal="center")
    ws.freeze_panes = "C2"
    for i in range(1, len(cols) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 12
    ws.column_dimensions[get_column_letter(len(cols))].width = 70

    def table_sheet(name, note, tables):
        ws = wb.create_sheet(name); ws.append([note]); ws["A1"].font = bold
        row = 3
        for title, tab in tables:
            ws.cell(row, 1, title).font = bold
            hdr = ["Index"] + [TLAB.get(c, c) for c in tab.columns]
            for j, h in enumerate(hdr, 1):
                c = ws.cell(row + 1, j, h); c.font = bold; c.alignment = Alignment(horizontal="center")
            for i, (k, vals) in enumerate(tab.iterrows(), 1):
                ws.cell(row + 1 + i, 1, k)
                for j, v in enumerate(vals, 2):
                    c = ws.cell(row + 1 + i, j, None if pd.isna(v) else round(float(v), 3)); c.number_format = "0.00"
            ws.conditional_formatting.add(f"B{row + 2}:{get_column_letter(len(hdr))}{row + 1 + len(tab)}",
                ColorScaleRule(start_type="num", start_value=-1, start_color="F4A582", mid_type="num", mid_value=0, mid_color="F7F7F7",
                               end_type="num", end_value=1, end_color="4393C3"))
            row += len(tab) + 4
        ws.column_dimensions["A"].width = 16
    n_ok = len(ok)
    table_sheet("Correlation", f"Correlation of each spectrometer index with SPAD and lab chlorophyll (quality-passed spectra only, n = {n_ok})",
                [(f"Pearson r, pooled (n = {n_ok})", pooled), ("Spearman rho, pooled", spear),
                 ("Pearson r within harvest cycles (each cycle's mean removed first)", within),
                 (f"Sensitivity: Pearson r pooled, ALL {len(df)} spectra incl. flagged", pooled_all)])
    table_sheet("Per cycle", "Pearson r inside each harvest cycle (quality-passed spectra)",
                [("with SPAD", per["SPAD"]), ("with total chlorophyll", per["CHL_Total"])])

    ws = wb.create_sheet("Cross-validation")
    ws.append(["Leave-one-harvest-cycle-out linear regression (quality-passed spectra). R2 < 0 = worse than predicting the mean."]); ws["A1"].font = bold
    ws.append(["Predictors"] + [f"{TLAB[t]} {m}" for t in TARGETS for m in ("R2", "RMSE")])
    for c in ws[2]: c.font = bold
    for k, v in cv.items():
        ws.append([k] + [round(float(x), 3) for t in TARGETS for x in v[t]])
    ws.column_dimensions["A"].width = 26

    ws = wb.create_sheet("Formulas & bands")
    ws.append(["Index", "Formula"]); [setattr(c, "font", bold) for c in ws[1]]
    for k, f in FORMULAS.items(): ws.append([k, f])
    ws.append([]); ws.append(["Band", "Wavelength range (nm), mean reflectance"]); [setattr(c, "font", bold) for c in ws[ws.max_row]]
    for b, (lo, hi) in BANDS.items(): ws.append([b, f"{lo}-{hi}"])
    ws.column_dimensions["A"].width = 14; ws.column_dimensions["B"].width = 70

    ws = wb.create_sheet("Data quality notes")
    notes = ["Findings about the source spectra (Spectrometer_Data.xlsx)",
             "1. Cycle 5, T2 and T3 (36 samples): the spectra are essentially one repeated signal (red band 54.7 +/- 0.3 across the group).",
             "   In the original workbook 14 of these columns were exact copies of other columns and about 18 were AVERAGE() formulas;",
             "   T2B1R1(3) is a blank spectrum (~0). All 36 are flagged and excluded from the main analysis.",
             "2. Outlier spectra in other cycles (e.g. cycle 1, group T1B2, red band up to 85 when the cycle median is ~2.5) are flagged by a robust z-score.",
             "3. The near-infrared signal does not show the plateau of healthy leaf reflectance: NIRb and NIRa are similar to (or lower than) the red band,",
             "   so NDVI is close to 0 (healthy leaf normally 0.6-0.9) and falls to ~0 after ~820 nm. Indices from these bands mostly reflect noise or instrument response.",
             "4. The raw spectra are noisy (spiky); the band means smooth this partly. Values are in percent reflectance and exceed 100 near 550 nm."]
    for n in notes: ws.append([n])
    ws["A1"].font = bold; ws.column_dimensions["A"].width = 150
    wb.save(TABLES / "Spectrometer_Indices.xlsx")


def figure(pooled, within, per, n_ok):
    SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#3d3c39"
    plt.rcParams["svg.fonttype"] = "path"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"], "text.color": INK, "font.size": 15})
    cmap = LinearSegmentedColormap.from_list("div", ["#c0392b", "#e88a80", "#f0efec", "#6da7ec", "#184f95"])
    fig = plt.figure(figsize=(16, 9.5), facecolor=SURF)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.62, left=0.075, right=0.97, top=0.86, bottom=0.08)
    fig.text(0.07, 0.965, "Spectrometer vegetation indices vs SPAD and laboratory chlorophyll", fontsize=25, fontweight="bold", va="top")
    fig.text(0.07, 0.925, f"Pearson r. Quality-passed spectra only (n = {n_ok}; cycle 5 T2/T3 and outlier spectra excluded, see notes in the Excel file).",
             fontsize=15, color=INK2, va="top")

    def heat(ax, M, xl, title, im_holder, xlab="Measured variable"):
        im = ax.imshow(M, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, "n/a" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=15, color="#ffffff" if abs(np.nan_to_num(v)) > 0.62 else INK)
        ax.set_xticks(range(len(xl))); ax.set_xticklabels(xl, fontsize=14)
        ax.set_xlabel(xlab, fontsize=15, labelpad=8); ax.set_ylabel("Vegetation index", fontsize=15, labelpad=8)
        ax.set_yticks(range(len(INDICES))); ax.set_yticklabels(INDICES, fontsize=15)
        ax.set_xticks(np.arange(-.5, len(xl), 1), minor=True); ax.set_yticks(np.arange(-.5, len(INDICES), 1), minor=True)
        ax.grid(which="minor", color=SURF, lw=3); ax.tick_params(which="both", length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=17, fontweight="bold", pad=10)
        cb = fig.colorbar(im, cax=ax.inset_axes([1.05, 0.0, 0.045, 1.0])); cb.outline.set_visible(False)
        cb.set_label("Pearson r (legend)", fontsize=14); cb.ax.tick_params(length=0, labelsize=12)
        im_holder.append(im)
    ims = []
    heat(fig.add_subplot(gs[0, 0]), pooled.values, [TLAB[t] for t in TARGETS], "Pooled", ims)
    heat(fig.add_subplot(gs[0, 1]), within.values, [TLAB[t] for t in TARGETS], "Within harvest cycles", ims)
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.text(0, 0.98, "Reading the table", fontsize=16, fontweight="bold", va="top")
    ax.text(0, 0.86, "Pooled: all samples in one group.\nWithin cycles: each cycle's own mean is\nremoved first, so cycle-to-cycle\ndifferences cannot create the link.\n\nValues near 0 mean no relationship.",
            fontsize=14.5, color=INK2, va="top", linespacing=1.35)
    heat(fig.add_subplot(gs[1, 0]), per["SPAD"].values, list(per["SPAD"].columns), "Correlation with SPAD, by cycle", ims, xlab="Harvest cycle")
    heat(fig.add_subplot(gs[1, 1]), per["CHL_Total"].values, list(per["CHL_Total"].columns), "Correlation with total chlorophyll, by cycle", ims, xlab="Harvest cycle")
    fig.savefig(FIGS / "spectrometer_indices_correlation.svg", facecolor=SURF)
    if os.environ.get("PREVIEW_DIR"): fig.savefig(Path(os.environ["PREVIEW_DIR"]) / "spectrometer_indices_correlation.png", dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)


if __name__ == "__main__":
    spectra = load_spectra()
    df = add_targets(band_table(spectra))
    df = quality_flags(add_indices(df))
    df = df.replace([np.inf, -np.inf], np.nan)
    ok = df[(df.QC_flag == "ok")].dropna(subset=INDICES)
    pooled, spear, within, per = corr_tables(ok)
    pooled_all = corr_tables(df.dropna(subset=INDICES))[0]
    cv = {}
    for label, cols in {"SRVI": ["SRVI"], "SRRE": ["SRRE"], "NDRE": ["NDRE"], "NDVI": ["NDVI"], "CCCI": ["CCCI"],
                        "Red + RedEdge + NIRb + NIRa": ["Red", "RedEdge", "NIRb", "NIRa"]}.items():
        cv[label] = {t: loco(ok[cols].values, ok[t].values, ok.Harvest.values) for t in TARGETS}
    write_excel(df, ok, pooled, spear, within, per, pooled_all, cv)
    figure(pooled, within, per, len(ok))
    print("all", len(df), "| passed", len(ok), "| excluded", int((df.QC_flag == "excluded").sum()))
    print(df[df.QC_flag == "excluded"].groupby(["Harvest", "QC_reason"]).size().to_string())
    print("\nindex medians (passed) by cycle:\n", ok.groupby("Harvest")[INDICES].median().round(2).to_string())
    print("\nPooled r:\n", pooled.round(2).to_string(), "\nWithin r:\n", within.round(2).to_string())
    print("\nPer cycle vs SPAD:\n", per["SPAD"].round(2).to_string(), "\nPer cycle vs Total:\n", per["CHL_Total"].round(2).to_string())
    print("\nAll-data pooled:\n", pooled_all.round(2).to_string())
    print("\nCV R2 (SPAD, Total):", {k: (round(v['SPAD'][0], 2), round(v['CHL_Total'][0], 2)) for k, v in cv.items()})
