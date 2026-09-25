import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / 'data'
FIGS = ROOT / 'results' / 'figures'
import re, openpyxl, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

spad = {}
wb = openpyxl.load_workbook(DATA / 'SPAD_Data.xlsx', data_only=True)
for cyc, ws in enumerate(wb.worksheets, 1):
    for r in range(3, ws.max_row + 1): spad[(cyc, ws.cell(r, 2).value)] = ws.cell(r, 3).value
rows = []
for r in openpyxl.load_workbook(DATA / 'Chl_Lab_Data.xlsx', data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
    if r[1] is None: continue
    rows.append((int(re.match(r'T(\d)', r[1]).group(1)), spad[(int(r[0]), r[1])], r[2], r[3], r[4]))
A = np.array(rows, float); trt, spd = A[:, 0], A[:, 1]
Y = {'Chlorophyll a': A[:, 2], 'Chlorophyll b': A[:, 3], 'Total chlorophyll': A[:, 4]}

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#3d3c39', '#e2e1dc'
COL = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}      # categorical slots 1-3 (validated all-pairs)
MRK = {1: 'o', 2: 's', 3: '^'}                        # shape as secondary encoding (aqua is < 3:1 on white)
LAB = {1: 'T1 · 100%', 2: 'T2 · 75%', 3: 'T3 · 50%'}
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'],
                     'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 15})
fig, axes = plt.subplots(1, 3, figsize=(16, 6.4), facecolor=SURF)
fig.subplots_adjust(left=0.06, right=0.985, top=0.75, bottom=0.135, wspace=0.36)
fig.text(0.06, 0.965, 'SPAD vs laboratory chlorophyll by irrigation treatment', fontsize=24, fontweight='bold', va='top')
fig.text(0.06, 0.895, 'Each point is one sample (n = 90 per treatment: 18 samples × 5 harvest cycles). Lines are linear fits within each treatment.',
         fontsize=15, color=INK2, va='top')
print('n per treatment', {t: int((trt == t).sum()) for t in (1, 2, 3)})
print('%-18s' % '' + ''.join('%14s' % LAB[t] for t in (1, 2, 3)) + '%14s' % 'All')
for ax, (name, y) in zip(axes, Y.items()):
    ax.set_facecolor(SURF)
    hs = []
    line = []
    for t in (1, 2, 3):
        m = trt == t
        ax.scatter(spd[m], y[m], s=46, marker=MRK[t], color=COL[t], alpha=0.72, linewidths=0, zorder=3)
        lr = stats.linregress(spd[m], y[m]); xs = np.array([spd[m].min(), spd[m].max()])
        ax.plot(xs, lr.intercept + lr.slope * xs, color=COL[t], lw=2.6, zorder=4)
        line.append(lr.rvalue)
        hs.append(plt.Line2D([], [], color=COL[t], marker=MRK[t], ms=9, lw=2.6, label=f'{LAB[t]}   r = {lr.rvalue:.2f}'))
    rall = stats.pearsonr(spd, y).statistic
    print('%-18s' % name + ''.join('%14.3f' % r for r in line) + '%14.3f' % rall)
    leg = ax.legend(handles=hs, loc='upper left', frameon=True, framealpha=0.92, edgecolor='none', facecolor=SURF,
                    fontsize=14, handlelength=1.8, borderpad=0.5, labelspacing=0.4)
    for txt in leg.get_texts(): txt.set_color(INK)
    ax.set_title(name, loc='left', fontsize=19, fontweight='bold', pad=10)
    ax.set_xlabel('SPAD value', fontsize=16, labelpad=6)
    ax.set_ylabel('Laboratory value', fontsize=16, labelpad=6)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b2')
    ax.tick_params(length=0, labelsize=14)
    ax.set_ylim(0, y.max() * 1.42)
fig.savefig(FIGS / 'spad_vs_chl_by_irrigation.svg', facecolor=SURF)
if os.environ.get('PREVIEW_DIR'): fig.savefig(os.path.join(os.environ['PREVIEW_DIR'], 'spad_vs_chl_by_irrigation.png'), dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
