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
spad = {}
for cyc, ws in enumerate(openpyxl.load_workbook(DATA / 'SPAD_Data.xlsx', data_only=True).worksheets, 1):
    for r in range(3, ws.max_row + 1): spad[(cyc, ws.cell(r, 2).value)] = ws.cell(r, 3).value
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
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'],
                     'mathtext.fontset': 'stix', 'text.color': INK, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 15})

fig = plt.figure(figsize=(15, 11.5), facecolor=SURF)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.50, wspace=0.40,
                      left=0.11, right=0.985, top=0.80, bottom=0.19)
fig.text(0.075, 0.965, 'SPAD vs laboratory chlorophyll', fontsize=25, fontweight='bold', va='top')
fig.text(0.075, 0.917, 'Each point is one sample (n = 270 = 54 samples × 5 harvest cycles). SPAD is the whole-leaf mean.',
         fontsize=15, color=INK2, va='top')

# ---- top row: scatter + pooled linear fit
leg = [Line2D([], [], color=BC[b], marker=BM[b], ls='', ms=10, label=BL[b]) for b in (1, 2)] + \
      [Line2D([], [], color=INK, lw=2.2, ls=(0, (5, 3)), label='Linear fit, all samples')]
for k, (name, y) in enumerate(Y.items()):
    ax = fig.add_subplot(gs[0, k]); ax.set_facecolor(SURF)
    for b in (1, 2):
        m = blk == b
        ax.scatter(spd[m], y[m], s=48, marker=BM[b], color=BC[b], alpha=0.75, linewidths=0, zorder=3)
    lr = stats.linregress(spd, y); xs = np.array([spd.min(), spd.max()])
    ax.plot(xs, lr.intercept + lr.slope * xs, color=INK, lw=2.2, ls=(0, (5, 3)), zorder=4)
    ax.text(0.04, 0.96, f'r = {lr.rvalue:.2f}\nR² = {lr.rvalue**2:.2f}', transform=ax.transAxes, va='top',
            fontsize=17, fontweight='bold', linespacing=1.25,
            bbox=dict(boxstyle='round,pad=0.25', fc=SURF, ec='none', alpha=0.9), zorder=5)
    ax.set_title(name, loc='left', fontsize=19, fontweight='bold', pad=10)
    ax.set_xlabel('SPAD value', fontsize=16, labelpad=6)
    ax.set_ylabel('Laboratory value', fontsize=16, labelpad=6)
    ax.grid(color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#b9b8b2')
    ax.tick_params(length=0, labelsize=14)
    ax.set_ylim(0, y.max() * 1.55)
    ax.legend(handles=leg, loc='upper right', frameon=False, fontsize=12.5, handletextpad=0.4, borderpad=0.3)

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
ax.set_title('Pearson correlation (r) between SPAD and laboratory chlorophyll, by subset', loc='left',
             fontsize=19, fontweight='bold', pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.012); cb.outline.set_visible(False)
cb.set_label('Pearson r (legend)', fontsize=15); cb.ax.tick_params(length=0, labelsize=14)

fig.text(0.11, 0.062,
         'Overall r ≈ 0.6, but it is carried by cycle 5 and the under-panel block (B2); outside the shade (B1) and in cycles 1–2 r is near zero.',
         fontsize=15, color=INK2, va='top')
fig.text(0.11, 0.028,
         'Chl a is capped near 38 in 11 samples (values 38.083–38.088). Font: Nimbus Roman (Times-compatible).',
         fontsize=13.5, color=INK2, va='top', style='italic')
fig.savefig(FIGS / 'spad_vs_chl_correlation.svg', facecolor=SURF)
if os.environ.get('PREVIEW_DIR'): fig.savefig(os.path.join(os.environ['PREVIEW_DIR'], 'spad_vs_chl_correlation.png'), dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)
