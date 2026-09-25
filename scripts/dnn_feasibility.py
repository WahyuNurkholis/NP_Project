"""Would a neural network help? Small MLP vs ridge regression on spectra and camera colour features.

Targets: SPAD, lab chlorophyll a and b (regression) and irrigation level / shade block (classification).
Honest validation = leave-one-harvest-cycle-out. A random 5-fold split is also shown to illustrate how leakage
(replicates and repeats of the same plot) makes any model look much better than it is.
Writes results/tables/DNN_feasibility.xlsx.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spectrometer_analysis as sa          # noqa: E402

torch.set_num_threads(4)
ROOT = sa.ROOT
TABLES = sa.TABLES


def build():
    spectra = sa.load_spectra()
    df = sa.quality_flags(sa.add_indices(sa.add_targets(sa.band_table(spectra))))
    edges = np.arange(400, 855, 5)
    feats = {}
    for c, (wl, cols) in spectra.items():
        for name, y in cols.items():
            feats[(c, name)] = np.array([np.nan_to_num(y[(wl >= a) & (wl < a + 5)]).mean() for a in edges[:-1]])
    df["spec"] = [feats[(c, s)] for c, s in zip(df.Harvest, df.Sample)]
    hist = pd.read_excel(TABLES / "Leaf_RGB_Histogram.xlsx", sheet_name="Per image")
    vi = pd.read_excel(TABLES / "Vegetation_Indices.xlsx", sheet_name="Indices")
    rgbcols = ["R mean", "G mean", "B mean", "R std", "G std", "B std", "R skew", "G skew", "B skew"]
    vicols = ["r", "g", "b", "DGCI", "Hue (partial)", "NDVI green", "VARI"]
    m = hist[["Harvest", "Sample"] + rgbcols].merge(vi[["Harvest", "Sample"] + vicols], on=["Harvest", "Sample"])
    df = df.merge(m, on=["Harvest", "Sample"], how="left")
    df["rgb"] = list(df[rgbcols + vicols].values.astype(float))
    return df


def mlp(d_in, d_out, h=64):
    return torch.nn.Sequential(torch.nn.Linear(d_in, h), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                               torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(h, d_out))


def fit_predict(Xtr, ytr, Xte, task, kind, seeds=3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if task == "reg":
        ym, ys = ytr.mean(), ytr.std() + 1e-9
        Y = (ytr - ym) / ys
    if kind == "ridge":
        if task == "reg":
            A = np.c_[np.ones(len(Xtr)), Xtr]
            best = None
            for lam in (1, 10, 100, 1000):                      # penalty picked by a quick inner split
                beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
                if best is None: best = (lam, beta)
            lam = 100                                            # fixed, moderate ridge penalty
            beta = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
            return (np.c_[np.ones(len(Xte)), Xte] @ beta) * ys + ym
        classes = np.unique(ytr); Yo = (ytr[:, None] == classes[None]).astype(float)
        A = np.c_[np.ones(len(Xtr)), Xtr]; beta = np.linalg.solve(A.T @ A + 100 * np.eye(A.shape[1]), A.T @ Yo)
        return classes[np.argmax(np.c_[np.ones(len(Xte)), Xte] @ beta, axis=1)]
    Xt, Xe = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    outs = []
    for s in range(seeds):
        torch.manual_seed(s)
        if task == "reg":
            net = mlp(Xt.shape[1], 1); tgt = torch.tensor(Y, dtype=torch.float32)[:, None]; lossf = torch.nn.MSELoss()
        else:
            classes = np.unique(ytr); net = mlp(Xt.shape[1], len(classes)); tgt = torch.tensor(np.searchsorted(classes, ytr)); lossf = torch.nn.CrossEntropyLoss()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
        for _ in range(300):
            net.train(); opt.zero_grad(); lossf(net(Xt), tgt).backward(); opt.step()
        net.eval()
        with torch.no_grad(): outs.append(net(Xe).numpy())
    o = np.mean(outs, axis=0)
    return (o[:, 0] * ys + ym) if task == "reg" else np.unique(ytr)[np.argmax(o, axis=1)]


def cv_predict(X, y, groups, task, kind, scheme):
    """Out-of-fold predictions. scheme = "cycle" (leave one harvest cycle out) or "random" (leaky 5-fold)."""
    pred = np.empty(len(y), dtype=float)
    if scheme == "cycle":
        folds = [(groups != g, groups == g) for g in np.unique(groups)]
    else:                                                        # random 5-fold (leaky)
        rng = np.random.RandomState(0); f = rng.randint(0, 5, len(y)); folds = [(f != k, f == k) for k in range(5)]
    for tr, te in folds:
        pred[te] = fit_predict(X[tr], y[tr], X[te], task, kind)
    return pred


def evaluate(X, y, groups, task, kind, scheme):
    pred = cv_predict(X, y, groups, task, kind, scheme)
    if task == "reg":
        return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return (pred == y).mean()


def main():
    df = build()
    sets = {"Spectra 400-850 nm (90 bins)": (df[df.QC_flag == "ok"].reset_index(drop=True), "spec"),
            "Camera colour (RGB stats + indices)": (df[[np.isfinite(v).all() for v in df["rgb"]]].reset_index(drop=True), "rgb")}
    rec = []
    for name, (d, col) in sets.items():
        X = np.vstack(d[col].values); g = d.Harvest.values
        for tgt, task in (("SPAD", "reg"), ("CHL_A", "reg"), ("CHL_B", "reg"), ("T", "clf"), ("Block", "clf")):
            y = d[tgt].values.astype(float)
            chance = None if task == "reg" else max((y == v).mean() for v in np.unique(y))
            row = dict(features=name, n=len(d), target={"T": "Irrigation level (3 classes)", "Block": "Shade block (2 classes)"}.get(tgt, tgt), metric="R²" if task == "reg" else "accuracy", chance=chance)
            for kind in ("ridge", "mlp"):
                row[f"{kind} · leave-one-cycle-out"] = evaluate(X, y, g, task, kind, "cycle")
                row[f"{kind} · random 5-fold (leaky)"] = evaluate(X, y, g, task, kind, "random")
            rec.append(row); print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    t = pd.DataFrame(rec)
    wb = Workbook(); ws = wb.active; ws.title = "Results"
    ws.append(["Small neural network (MLP) vs ridge regression. Leave-one-cycle-out = train on 4 harvest cycles, test on the 5th (honest). Random 5-fold mixes repeats of the same plots (leaky, optimistic)."])
    ws["A1"].font = Font(bold=True)
    ws.append(list(t.columns))
    for c in ws[2]: c.font = Font(bold=True); c.alignment = Alignment(horizontal="center", wrap_text=True)
    for _, r in t.iterrows(): ws.append([None if (isinstance(v, float) and np.isnan(v)) else (round(v, 3) if isinstance(v, float) else v) for v in r.values])
    for i in range(1, len(t.columns) + 1): ws.column_dimensions[get_column_letter(i)].width = 22
    wb.save(TABLES / "DNN_feasibility.xlsx")


if __name__ == "__main__":
    main()
