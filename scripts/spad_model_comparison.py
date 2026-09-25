"""Compare regression forms for lab chlorophyll vs SPAD: linear, quadratic, cubic, logarithmic, exponential, power.

For each subset (all samples, B1, B2, cycle 5, cycle 5 B1, cycle 5 B2), SPAD source (whole / upper / mid / lower leaf)
and lab measure (Chl a, Chl b, Total Chl) reports R², adjusted R², leave-one-out cross-validated R² (Q²), RMSE and AIC.
All R² values are on the original (untransformed) chlorophyll scale.

Writes results/tables/SPAD_Model_Comparison.xlsx and results/figures/spad_model_comparison.svg (PNG to $PREVIEW_DIR if set).
"""
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from matplotlib.lines import Line2D
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / 'data'
FIGS = ROOT / 'results' / 'figures'
TABLES = ROOT / 'results' / 'tables'

# ---- data (same sources as plot_spad_by_block_regression.py)
SPAD = {'Whole leaf': {}}
for cyc, ws in enumerate(openpyxl.load_workbook(DATA / 'SPAD_Data.xlsx', data_only=True).worksheets, 1):
    for r in range(3, ws.max_row + 1):
        if ws.cell(r, 2).value is not None: SPAD['Whole leaf'][(cyc, ws.cell(r, 2).value)] = ws.cell(r, 3).value
parts = openpyxl.load_workbook(DATA / 'backup_original' / 'SPAD_Data_B3_dropped_per_part.xlsx', data_only=True).worksheets
for part, label in (('up', 'Upper leaf'), ('mid', 'Mid leaf'), ('low', 'Lower leaf')):
    SPAD[label] = {}
    for cyc, ws in enumerate(parts, 1):
        cols = [j for j in range(3, ws.max_column + 1) if ws.cell(1, j).value == part]
        for r in range(3, ws.max_row + 1):
            v = [ws.cell(r, j).value for j in cols]; v = [x for x in v if isinstance(x, (int, float))]
            if ws.cell(r, 2).value is not None and v: SPAD[label][(cyc, ws.cell(r, 2).value)] = sum(v) / len(v)
LAB = [dict(cyc=int(r[0]), s=r[1], blk=int(re.match(r'T\dB(\d)', r[1]).group(1)), a=r[2], b=r[3], t=r[4])
       for r in openpyxl.load_workbook(DATA / 'Chl_Lab_Data.xlsx', data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True)
       if r[1] is not None]
MEAS = (('Chl a', 'a'), ('Chl b', 'b'), ('Total Chl', 't'))
SUBSETS = [('All samples', lambda d: True), ('B1 outside', lambda d: d['blk'] == 1), ('B2 under panel', lambda d: d['blk'] == 2),
           ('Cycle 5', lambda d: d['cyc'] == 5), ('Cycle 5 B1 outside', lambda d: d['cyc'] == 5 and d['blk'] == 1),
           ('Cycle 5 B2 under panel', lambda d: d['cyc'] == 5 and d['blk'] == 2)]


# ---- models: name -> (fit(x, y) -> params, predict(params, x), n params, equation text)
def poly(deg):
    return (lambda x, y: np.polyfit(x, y, deg), lambda p, x: np.polyval(p, x), deg + 1)
def _exp(x, y):
    b, ln_a = np.polyfit(x, np.log(y), 1)                     # start from the log-linear fit, then least squares on y
    return curve_fit(lambda x, a, b: a * np.exp(b * x), x, y, p0=(np.exp(ln_a), b), maxfev=20000)[0]
def _pow(x, y):
    b, ln_a = np.polyfit(np.log(x), np.log(y), 1)
    return curve_fit(lambda x, a, b: a * x ** b, x, y, p0=(np.exp(ln_a), b), maxfev=20000)[0]
MODELS = {'Linear': poly(1), 'Quadratic': poly(2), 'Cubic': poly(3),
          'Logarithmic': (lambda x, y: np.polyfit(np.log(x), y, 1), lambda p, x: p[0] * np.log(x) + p[1], 2),
          'Exponential': (_exp, lambda p, x: p[0] * np.exp(p[1] * x), 2),
          'Power': (_pow, lambda p, x: p[0] * x ** p[1], 2)}


def equation(name, p):
    f = lambda v: f'{v:+.4g}'.replace('+', '+ ').replace('-', '− ')
    if name == 'Linear': return f'y = {p[0]:.4g}x {f(p[1])}'
    if name == 'Quadratic': return f'y = {p[0]:.4g}x² {f(p[1])}x {f(p[2])}'
    if name == 'Cubic': return f'y = {p[0]:.3g}x³ {f(p[1])}x² {f(p[2])}x {f(p[3])}'
    if name == 'Logarithmic': return f'y = {p[0]:.4g} ln(x) {f(p[1])}'
    if name == 'Exponential': return f'y = {p[0]:.4g} e^({p[1]:.4g}x)'
    return f'y = {p[0]:.4g} x^{p[1]:.4g}'


def evaluate(x, y, name):
    fit, pred, k = MODELS[name]
    p = fit(x, y); e = y - pred(p, x); n = len(y)
    ss_res, ss_tot = (e ** 2).sum(), ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    loo = np.array([y[i] - pred(fit(np.delete(x, i), np.delete(y, i)), x[i:i + 1])[0] for i in range(n)])
    return dict(p=p, eq=equation(name, p), k=k, n=n, R2=r2, adjR2=1 - (1 - r2) * (n - 1) / (n - k - 1),
                Q2=1 - (loo ** 2).sum() / ss_tot, RMSE=np.sqrt(ss_res / n), AIC=n * np.log(ss_res / n) + 2 * k)


res = {}
for sub, f in SUBSETS:
    rows = [d for d in LAB if f(d)]
    for src in SPAD:
        x = np.array([SPAD[src][(d['cyc'], d['s'])] for d in rows], float)
        for m, key in MEAS:
            y = np.array([d[key] for d in rows], float)
            for name in MODELS: res[(sub, src, m, name)] = evaluate(x, y, name)

# ---- workbook
bold = Font(bold=True); best = PatternFill('solid', fgColor='CDE2FB'); ctr = Alignment(horizontal='center', vertical='center', wrap_text=True)
wb = openpyxl.Workbook(); ws = wb.active; ws.title = 'Best model'
ws.append(['Best model per subset × SPAD source × lab measure (highest cross-validated R², Q²). Linear R² shown for comparison.'])
ws['A1'].font = bold
hdr = ['Subset', 'SPAD source', 'Lab measure', 'n', 'Linear R²', 'Best model', 'Best R²', 'Best adj. R²', 'Best Q² (LOO-CV)', 'Gain in R² over linear', 'Equation (x = SPAD)']
ws.append(hdr)
for c in ws[2]: c.font = bold; c.alignment = ctr
for sub, _ in SUBSETS:
    for src in SPAD:
        for m, _ in MEAS:
            name = max(MODELS, key=lambda nm: res[(sub, src, m, nm)]['Q2'])
            b, lin = res[(sub, src, m, name)], res[(sub, src, m, 'Linear')]
            ws.append([sub, src, m, b['n'], round(lin['R2'], 3), name, round(b['R2'], 3), round(b['adjR2'], 3), round(b['Q2'], 3),
                       round(b['R2'] - lin['R2'], 3), b['eq']])
for j, w in enumerate((22, 12, 11, 5, 10, 13, 9, 11, 12, 12, 44), 1): ws.column_dimensions[get_column_letter(j)].width = w
ws.freeze_panes = 'A3'; ws.auto_filter.ref = f'A2:K{ws.max_row}'

ws = wb.create_sheet('R2 by model')
ws.append(['R² of each model (rows = subset × SPAD source × lab measure). Highest R² in each row is shaded.']); ws['A1'].font = bold
ws.append(['Subset', 'SPAD source', 'Lab measure', 'n'] + list(MODELS))
for c in ws[2]: c.font = bold; c.alignment = ctr
for sub, _ in SUBSETS:
    for src in SPAD:
        for m, _ in MEAS:
            v = [res[(sub, src, m, nm)]['R2'] for nm in MODELS]
            ws.append([sub, src, m, res[(sub, src, m, 'Linear')]['n']] + [round(x, 3) for x in v])
            ws.cell(ws.max_row, 5 + int(np.argmax(v))).fill = best
for j, w in enumerate((22, 12, 11, 5) + (12,) * len(MODELS), 1): ws.column_dimensions[get_column_letter(j)].width = w
ws.freeze_panes = 'E3'; ws.auto_filter.ref = f'A2:J{ws.max_row}'

ws = wb.create_sheet('All fits')
ws.append(['Subset', 'SPAD source', 'Lab measure', 'Model', 'n', 'Parameters', 'R²', 'Adj. R²', 'Q² (LOO-CV)', 'RMSE', 'AIC', 'Equation (x = SPAD)'])
for c in ws[1]: c.font = bold; c.alignment = ctr
for (sub, src, m, name), r in res.items():
    ws.append([sub, src, m, name, r['n'], r['k'], round(r['R2'], 3), round(r['adjR2'], 3), round(r['Q2'], 3), round(r['RMSE'], 3), round(r['AIC'], 2), r['eq']])
for j, w in enumerate((22, 12, 11, 12, 5, 10, 8, 9, 11, 8, 9, 44), 1): ws.column_dimensions[get_column_letter(j)].width = w
ws.freeze_panes = 'A2'; ws.auto_filter.ref = f'A2:L{ws.max_row}'

ws = wb.create_sheet('Notes')
for t in ['R² = share of variance explained on the original chlorophyll scale (exponential and power fitted by nonlinear least squares, not on log scale).',
          'Adj. R² penalises extra parameters (quadratic 3, cubic 4, others 2).',
          'Q² = leave-one-out cross-validated R²: each sample predicted by a model fitted without it. Q² far below R² = over-fitting; Q² < 0 = worse than the mean.',
          'AIC: lower is better; compare only within the same subset, SPAD source and lab measure.',
          'SPAD sources: whole leaf = SPAD_Data.xlsx; upper / mid / lower = part means from SPAD_Data_B3_dropped_per_part.xlsx (readings present only).']:
    ws.append([t])
ws.column_dimensions['A'].width = 140
wb.save(TABLES / 'SPAD_Model_Comparison.xlsx')

# ---- figure: Total Chl vs mid-leaf SPAD, every model, one panel per subset
FIG_SRC, FIG_M, FIG_KEY = 'Mid leaf', 'Total Chl', 't'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#3d3c39', '#e2e1dc'
BC = {1: '#2a78d6', 2: '#eb6834'}; BM = {1: 'o', 2: 's'}
MC = {'Linear': '#0b0b0b', 'Quadratic': '#1baf7a', 'Cubic': '#8e44ad', 'Logarithmic': '#c0392b', 'Exponential': '#b8860b', 'Power': '#17a2b8'}
ML = {'Linear': '-', 'Quadratic': '-', 'Cubic': '-', 'Logarithmic': (0, (5, 2)), 'Exponential': (0, (2, 1.5)), 'Power': (0, (6, 2, 1.5, 2))}
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'],
                     'mathtext.fontset': 'stix', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 15})
fig, axes = plt.subplots(2, 3, figsize=(22, 15), facecolor=SURF)
fig.subplots_adjust(left=0.06, right=0.99, top=0.87, bottom=0.08, hspace=0.36, wspace=0.26)
fig.text(0.06, 0.975, 'Total chlorophyll vs mid-leaf SPAD: six regression models compared', fontsize=28, fontweight='bold', va='top')
fig.text(0.06, 0.935, 'R² on the original chlorophyll scale; Q² = leave-one-out cross-validated R² (how well the curve predicts a sample it was not fitted on).',
         fontsize=16, color=INK2, va='top')
for ax, (sub, f) in zip(axes.flat, SUBSETS):
    rows = [d for d in LAB if f(d)]
    x = np.array([SPAD[FIG_SRC][(d['cyc'], d['s'])] for d in rows], float); y = np.array([d[FIG_KEY] for d in rows], float)
    bl = np.array([d['blk'] for d in rows])
    ax.set_facecolor(SURF)
    pts = []
    for b in (1, 2):
        m = bl == b
        if m.any():
            ax.scatter(x[m], y[m], s=40, marker=BM[b], color=BC[b], alpha=0.35, linewidths=0, zorder=2)
            pts.append(Line2D([], [], color=BC[b], marker=BM[b], ls='', ms=8, alpha=0.6, label=['B1 outside', 'B2 under panel'][b - 1]))
    xs = np.linspace(x.min(), x.max(), 300); curves = []
    for nm in MODELS:
        r = res[(sub, FIG_SRC, FIG_M, nm)]
        ax.plot(xs, MODELS[nm][1](r['p'], xs), color=MC[nm], lw=2.4, ls=ML[nm], zorder=4)
        curves.append(Line2D([], [], color=MC[nm], lw=2.4, ls=ML[nm], label=f"{nm}: R² = {r['R2']:.2f}, Q² = {r['Q2']:.2f}"))
    lo, hi = y.min(), y.max(); ax.set_ylim(max(0, lo - 0.05 * (hi - lo)), hi + 1.05 * (hi - lo))
    ax.legend(handles=pts + curves, loc='upper left', frameon=True, facecolor=SURF, edgecolor='none', framealpha=0.9,
              fontsize=12, handletextpad=0.5, borderpad=0.3, labelspacing=0.25, handlelength=2.4, ncol=1)
    ax.set_title(f'{sub} (n = {len(rows)})', loc='left', fontsize=18, fontweight='bold', pad=8)
    ax.set_xlabel('Mid-leaf SPAD value', fontsize=15, labelpad=5); ax.set_ylabel('Total chlorophyll (lab value)', fontsize=15, labelpad=5)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
    for s_ in ('left', 'bottom'): ax.spines[s_].set_color('#b9b8b2')
    ax.tick_params(length=0, labelsize=13.5)
fig.text(0.06, 0.025, 'All SPAD sources and lab measures are in results/tables/SPAD_Model_Comparison.xlsx. '
         'Font: Nimbus Roman (Times-compatible).', fontsize=14, color=INK2, style='italic')
fig.savefig(FIGS / 'spad_model_comparison.svg', facecolor=SURF)
if os.environ.get('PREVIEW_DIR'):
    fig.savefig(Path(os.environ['PREVIEW_DIR']) / 'spad_model_comparison.png', dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)

# ---- console summary
for sub, _ in SUBSETS:
    print(f'\n{sub}')
    for m, _ in MEAS:
        line = []
        for src in SPAD:
            name = max(MODELS, key=lambda nm: res[(sub, src, m, nm)]['Q2']); b = res[(sub, src, m, name)]
            line.append(f"{src.split()[0][:5]} lin {res[(sub, src, m, 'Linear')]['R2']:.2f} best {name[:5]} R2 {b['R2']:.2f} Q2 {b['Q2']:.2f}")
        print(f'  {m:9s} | ' + ' | '.join(line))
    print('  max R2 any model:', max((round(res[(sub, s, m, nm)]['R2'], 3), s, m, nm) for s in SPAD for m, _ in MEAS for nm in MODELS))

