import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / 'data'
FIGS = ROOT / 'results' / 'figures'
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats

import re, openpyxl
# SPAD = mean of the 'mid' leaf-part readings only (row 1 header == 'mid'), averaged over the readings present
spad = {}
for cyc, ws in enumerate(openpyxl.load_workbook(DATA / 'backup_original' / 'SPAD_Data_B3_dropped_per_part.xlsx', data_only=True).worksheets, 1):
    cols = [j for j in range(3, ws.max_column + 1) if ws.cell(1, j).value == 'mid']
    for r in range(3, ws.max_row + 1):
        v = [ws.cell(r, j).value for j in cols]; v = [x for x in v if isinstance(x, (int, float))]
        if ws.cell(r, 2).value is not None and v: spad[(cyc, ws.cell(r, 2).value)] = sum(v) / len(v)
_rows = []
for r in openpyxl.load_workbook(DATA / 'Chl_Lab_Data.xlsx', data_only=True).active.iter_rows(min_row=2, max_col=5, values_only=True):
    if r[1] is None: continue
    _rows.append((int(r[0]), spad[(int(r[0]), r[1])], r[2], r[3], r[4], int(re.match(r'T\dB(\d)', r[1]).group(1))))
D = np.array(_rows, float)            # harvest, spad, chl a, chl b, total, block
harv, spd, blk = D[:, 0], D[:, 1], D[:, 5]
Y = {'Chlorophyll a': D[:, 2], 'Chlorophyll b': D[:, 3], 'Total chlorophyll': D[:, 4]}

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#3d3c39', '#e2e1dc'
BC = {1: '#2a78d6', 2: '#eb6834'}; BM = {1: 'o', 2: 's'}
BL = {1: 'B1 – outside panel shade', 2: 'B2 – under solar panel'}
BS = {1: 'B1', 2: 'B2'}; FL = {1: (0, (5, 3)), 2: '-'}
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'],
                     'mathtext.fontset': 'stix', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 15})

fig = plt.figure(figsize=(15, 11.5), facecolor=SURF)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.50, wspace=0.40,
                      left=0.11, right=0.985, top=0.80, bottom=0.19)
fig.text(0.075, 0.965, 'Mid-leaf SPAD vs laboratory chlorophyll, fitted separately by shade block', fontsize=25, fontweight='bold', va='top')
fig.text(0.075, 0.917, 'Each point is one sample (n = 270 = 54 samples × 5 harvest cycles). SPAD is the mean of the mid-leaf readings only. One linear fit per block (n = 135 each).',
         fontsize=15, color=INK2, va='top')

# ---- top row: scatter + separate linear fit for each block (fit stats in the legend)
for k, (name, y) in enumerate(Y.items()):
    ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(SURF)
    pts, fits = [], []
    for b in (1, 2):
        m = blk == b
        ax.scatter(spd[m], y[m], s=48, marker=BM[b], color=BC[b], alpha=0.6, linewidths=0, zorder=3)
        lr = stats.linregress(spd[m], y[m]); xs = np.array([spd[m].min(), spd[m].max()])
        ax.plot(xs, lr.intercept + lr.slope * xs, color=SURF, lw=5.5, zorder=4)          # halo
        ax.plot(xs, lr.intercept + lr.slope * xs, color=BC[b], lw=2.6, ls=FL[b], zorder=5)
        pts.append(Line2D([], [], color=BC[b], marker=BM[b], ls='', ms=10, label=BL[b]))
        fits.append(Line2D([], [], color=BC[b], lw=2.6, ls=FL[b],
                           label=f'{BS[b]} fit: r = {lr.rvalue:.2f}, R² = {lr.rvalue**2:.2f}\n'
                                 f'y = {lr.slope:.2f}x {"+" if lr.intercept >= 0 else "−"} {abs(lr.intercept):.1f}'))
    ax.set_title(name, loc='left', fontsize=19, fontweight='bold', pad=10)
    ax.set_xlabel('Mid-leaf SPAD value', fontsize=16, labelpad=6)
    ax.set_ylabel('Laboratory value', fontsize=16, labelpad=6)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b2')
    ax.tick_params(length=0, labelsize=14)
    ax.set_ylim(0, y.max() * 2.0)
    ax.legend(handles=pts + fits, loc='upper left', frameon=True, facecolor=SURF, edgecolor='none', framealpha=0.9,
              fontsize=12.5, handletextpad=0.5, borderpad=0.3, labelspacing=0.35, handlelength=2.2)

# ---- bottom: Pearson r for every subset (sequential blue, value printed)
subsets = [('All\nsamples', np.ones(len(spd), bool)), ('B1\noutside', blk == 1), ('B2\nunder panel', blk == 2)] + \
          [(f'Cycle {h}', harv == h) for h in range(1, 6)]
R = np.array([[stats.pearsonr(spd[m], y[m]).statistic for y in Y.values()] for _, m in subsets])
N = [int(m.sum()) for _, m in subsets]
cmap = LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b'])
ax = fig.add_subplot(gs[1, :]); ax.set_facecolor(SURF)
im = ax.imshow(R.T, cmap=cmap, vmin=0, vmax=1, aspect='auto')
for i in range(R.shape[0]):
    for j in range(R.shape[1]):
        ax.text(i, j, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=19, fontweight='bold',
                color='#ffffff' if R[i, j] > 0.45 else INK)
ax.set_xticks(range(len(subsets)))
ax.set_xticklabels([f'{s}\n(n = {n})' for (s, _), n in zip(subsets, N)], fontsize=15)
ax.set_yticks(range(3)); ax.set_yticklabels(['Chl a', 'Chl b', 'Total Chl'], fontsize=16)
ax.set_xlabel('Subset of samples', fontsize=16, labelpad=8); ax.set_ylabel('Laboratory measure', fontsize=16, labelpad=8)
ax.set_xticks(np.arange(-.5, len(subsets), 1), minor=True); ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which='minor', color=SURF, lw=4); ax.tick_params(which='both', length=0)
for s in ax.spines.values(): s.set_visible(False)
ax.set_title('Pearson correlation (r) between mid-leaf SPAD and laboratory chlorophyll, by subset', loc='left',
             fontsize=19, fontweight='bold', pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012); cb.outline.set_visible(False)
cb.set_label('Pearson r (legend)', fontsize=15); cb.ax.tick_params(length=0, labelsize=14)

fig.text(0.11, 0.062,
         f'All samples: R² = {R[0, 0]**2:.2f} (Chl a), {R[0, 1]**2:.2f} (Chl b), {R[0, 2]**2:.2f} (Total Chl). '
         f'SPAD source: SPAD_Data_B3_dropped_per_part.xlsx, mid columns (4 readings in cycle 1, 6 in cycles 2–5).',
         fontsize=14, color=INK2, va='top')
fig.text(0.11, 0.028, 'Mean taken over the readings present in each row. Font: Nimbus Roman (Times-compatible).',
         fontsize=13.5, color=INK2, va='top', style='italic')
fig.savefig(FIGS / 'spad_mid_vs_chl_by_shade.svg', facecolor=SURF)
if os.environ.get('PREVIEW_DIR'): fig.savefig(os.path.join(os.environ['PREVIEW_DIR'], 'spad_mid_vs_chl_by_shade.png'), dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)

print('subset'.ljust(12), *[k.ljust(18) for k in Y])
for (nm, _), row in zip(subsets, R): print(nm.replace('\n', ' ').ljust(12), *[f'r={v:.2f} R²={v*v:.2f}'.ljust(18) for v in row])
