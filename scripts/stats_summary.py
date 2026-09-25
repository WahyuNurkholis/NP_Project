"""Descriptive statistics, ANOVA and SPAD-vs-lab correlations used in the report.

Writes results/tables/Statistics_summary.xlsx (sheets: Means, ANOVA, SPAD vs lab, Lab Total check).
ANOVA = partial F-tests from ordinary least squares (numpy), each effect tested after the others.
"""
import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA, TABLES = ROOT / "data", ROOT / "results" / "tables"


def load():
    spad = {}
    for c, ws in enumerate(openpyxl.load_workbook(DATA / "SPAD_Data.xlsx", data_only=True).worksheets, 1):
        for r in range(3, ws.max_row + 1):
            spad[(c, ws.cell(r, 2).value)] = ws.cell(r, 3).value
    rows = []
    for r in openpyxl.load_workbook(DATA / "Chl_Lab_Data.xlsx", data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
        if r[1] is None: continue
        t, b, rep = re.match(r"T(\d)B(\d)R(\d)", r[1]).groups()
        rows.append(dict(Harvest=int(r[0]), T=int(t), Block=int(b), Rep=int(rep), SPAD=spad[(int(r[0]), r[1])], CHL_A=r[2], CHL_B=r[3], CHL_Total=r[4]))
    return pd.DataFrame(rows)


def dummies(df, col): return pd.get_dummies(df[col].astype(str), drop_first=True).astype(float).values


def rss(y, X):
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]; return ((y - X @ b) ** 2).sum(), X.shape[1]


def partial_f(y, full, reduced):
    s1, p1 = rss(y, full); s0, p0 = rss(y, reduced); n = len(y)
    F = ((s0 - s1) / (p1 - p0)) / (s1 / (n - p1))
    return F, stats.f.sf(F, p1 - p0, n - p1), (s0 - s1) / ((y - y.mean()) ** 2).sum() * 100, p1 - p0, n - p1


def main():
    df = load()
    H, B, T = dummies(df, "Harvest"), dummies(df, "Block"), dummies(df, "T")
    TH = np.column_stack([T[:, i] * H[:, j] for i in range(T.shape[1]) for j in range(H.shape[1])])
    TB = np.column_stack([T[:, i] * B[:, 0] for i in range(T.shape[1])])
    base = np.c_[H, B]
    anova = []
    for y in ("SPAD", "CHL_A", "CHL_B"):
        v = df[y].values
        for name, full, red in (("Irrigation (T)", np.c_[base, T], base), ("Shade block (B)", np.c_[H, T, B], np.c_[H, T]),
                                ("Harvest cycle (C)", np.c_[B, T, H], np.c_[B, T]), ("Irrigation × block", np.c_[base, T, TB], np.c_[base, T]),
                                ("Irrigation × cycle", np.c_[base, T, TH], np.c_[base, T])):
            F, p, sh, d1, d2 = partial_f(v, full, red)
            anova.append(dict(variable=y, effect=name, df1=d1, df2=d2, F=F, p=p, share_of_variance_pct=sh))
    anova = pd.DataFrame(anova)
    means = df.groupby(["Block", "T"])[["SPAD", "CHL_A", "CHL_B"]].agg(["mean", "std"]).reset_index()
    means.columns = ["Block", "T", "SPAD mean", "SPAD sd", "Chl a mean", "Chl a sd", "Chl b mean", "Chl b sd"]
    cyc = df.groupby("Harvest")[["SPAD", "CHL_A", "CHL_B"]].mean().reset_index()
    rec = []
    subsets = [("All samples", df)] + [(f"B{b}", df[df.Block == b]) for b in (1, 2)] + [(f"T{t}", df[df["T"] == t]) for t in (1, 2, 3)] + [(f"Cycle {h}", df[df.Harvest == h]) for h in range(1, 6)]
    for nm, d in subsets:
        rec.append(dict(subset=nm, n=len(d), r_chl_a=stats.pearsonr(d.SPAD, d.CHL_A).statistic, r_chl_b=stats.pearsonr(d.SPAD, d.CHL_B).statistic,
                        R2_chl_a=stats.pearsonr(d.SPAD, d.CHL_A).statistic ** 2, R2_chl_b=stats.pearsonr(d.SPAD, d.CHL_B).statistic ** 2))
    corr = pd.DataFrame(rec)
    ratio = df.CHL_Total / df.CHL_B
    check = pd.DataFrame([dict(quantity="Total Chl / Chl b (min)", value=ratio.min()), dict(quantity="Total Chl / Chl b (max)", value=ratio.max()),
                          dict(quantity="Pearson r, Total Chl vs Chl b", value=stats.pearsonr(df.CHL_B, df.CHL_Total).statistic),
                          dict(quantity="Pearson r, Total Chl vs Chl a", value=stats.pearsonr(df.CHL_A, df.CHL_Total).statistic),
                          dict(quantity="Chl a values within 0.01 of the maximum", value=int((df.CHL_A > df.CHL_A.max() - 0.01).sum())), dict(quantity="Chl a maximum", value=df.CHL_A.max())])
    wb = openpyxl.Workbook(); wb.remove(wb.active); bold = Font(bold=True)
    for name, t in (("Means", means), ("Means by cycle", cyc), ("ANOVA", anova), ("SPAD vs lab", corr), ("Lab Total check", check)):
        ws = wb.create_sheet(name); ws.append(list(t.columns))
        for c in ws[1]: c.font = bold; c.alignment = Alignment(horizontal="center")
        for _, r in t.iterrows(): ws.append([round(float(v), 4) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else v) for v in r.values])
        for i in range(1, len(t.columns) + 1): ws.column_dimensions[get_column_letter(i)].width = 22
    wb.save(TABLES / "Statistics_summary.xlsx")
    print(means.round(2).to_string(index=False)); print(anova.round(4).to_string(index=False)); print(corr.round(3).to_string(index=False)); print(check.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
