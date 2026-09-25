import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent          # project folder
DATA = ROOT / 'data'
FIGS = ROOT / 'results' / 'figures'
import os, re, openpyxl, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

wb = openpyxl.load_workbook(DATA / 'SPAD_Data.xlsx', data_only=True)
data = {}   # (T,B,cycle) -> list of whole-leaf SPAD values (9 per cell: 3 R x 3 repeats)
for cyc, ws in enumerate(wb.worksheets, 1):
    for r in range(3, ws.max_row + 1):
        m = re.match(r'^T(\d)B(\d)R\d\(\d\)$', ws.cell(r, 2).value)
        t, b = int(m.group(1)), int(m.group(2))
        data.setdefault((t, b, cyc), []).append(ws.cell(r, 3).value)

SURF, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#7a796f', '#e6e5e0'
COL = {1: '#2a78d6', 2: '#eb6834', 3: '#1baf7a'}            # categorical slots 1-3, in order
MRK = {1: 'o', 2: 's', 3: '^'}                              # secondary encoding (aqua contrast < 3:1)
LAB = {1: 'T1 · 100% irrigation', 2: 'T2 · 75%', 3: 'T3 · 50%'}
PANEL = {1: 'B1 · outside panel shade', 2: 'B2 · under solar panel'}

plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman', 'STIXGeneral'], 'font.size': 15, 'text.color': INK, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6), facecolor=SURF)
x = np.arange(1, 6)
for ax, b in zip(axes, (1, 2)):
    ax.set_facecolor(SURF)
    ends = {}
    for t in (1, 2, 3):
        mean = np.array([np.mean(data[(t, b, c)]) for c in x])
        se = np.array([np.std(data[(t, b, c)], ddof=1) / np.sqrt(len(data[(t, b, c)])) for c in x])
        ax.errorbar(x, mean, yerr=se, color=COL[t], lw=2, marker=MRK[t], ms=10, mfc=COL[t], mec=SURF, mew=2,
                    elinewidth=1, capsize=0, ecolor=COL[t], alpha=1, zorder=3, label=LAB[t])
        ends[t] = mean[-1]
    # direct labels at the line ends, nudged apart so they never collide
    order = sorted(ends, key=ends.get); ys = {}
    last = -1e9
    for t in order:
        y = max(ends[t], last + 1.6); ys[t] = y; last = y
    for t in (1, 2, 3):
        ax.annotate(f'T{t}', (5, ends[t]), xytext=(5.13, ys[t]), va='center', ha='left', fontsize=16,
                    color=INK, fontweight='bold', annotation_clip=False)
    ax.set_title(PANEL[b], loc='left', fontsize=19, color=INK, pad=10, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([f'C{c}' for c in x]); ax.set_xlim(0.75, 5.55)
    ax.set_xlabel('Harvest cycle', fontsize=16)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2'); ax.tick_params(length=0)
for a in axes: a.set_ylabel('SPAD (whole leaf)', fontsize=16)
for a in axes: a.tick_params(labelsize=14)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=COL[t], lw=2.4, marker=MRK[t], ms=10, mec=SURF, mew=1.5) for t in (1, 2, 3)]
labels = [LAB[t] for t in (1, 2, 3)]
for a, loc in zip(axes, ('upper right', 'upper left')): a.legend(handles, labels, loc=loc, frameon=False, fontsize=14, labelcolor=INK)
fig.suptitle('SPAD chlorophyll by harvest cycle, irrigation treatment and shade', x=0.06, ha='left', y=0.985,
             fontsize=22, fontweight='bold', color=INK)
fig.text(0.06, 0.915, 'Mean of 9 samples (3 replications × 3 repeats) per point; whiskers show ±1 standard error',
         fontsize=15, color=INK2)
fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.9), w_pad=3)
fig.savefig(FIGS / 'spad_plot.svg', facecolor=SURF)
if os.environ.get('PREVIEW_DIR'): fig.savefig(os.path.join(os.environ['PREVIEW_DIR'], 'spad_plot.png'), dpi=int(os.environ.get('PREVIEW_DPI', 100)), facecolor=SURF)

# also print the table behind the chart (table view)
print('cell counts', sorted({len(v) for v in data.values()}))
print('%-4s %-3s' % ('T', 'B') + ''.join('%8s' % f'C{c}' for c in x))
for b in (1, 2):
    for t in (1, 2, 3):
        print('T%d   B%d ' % (t, b) + ''.join('%8.1f' % np.mean(data[(t, b, c)]) for c in x))
