import os
os.makedirs("figures", exist_ok=True)
# -*- coding: utf-8 -*-
"""Plot the composite ASR of every defense as a horizontal bar chart.

Horizontal bars keep the ten defense names readable, and a logarithmic axis
is used because the values span two orders of magnitude (0.86% to 100%).
"""
import os, json
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": "--", "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
OK = {"green": "#009E73", "red": "#D55E00", "sky": "#56B4E9",
      "orange": "#E69F00", "black": "#000000"}
S5 = ['42', '123', '3407', '2025', '666']
L = lambda p: json.load(open(p))

d = dict(L('outputs/baseline_5seed_results.json')['results'])
d.update(L('outputs/shieldfl_5seed_results.json')['results'])
order = ['FedAvg', 'FedMedian', 'Krum', 'FLAME', 'RLR', 'DeepSight',
         'SHIELD-FL', 'Lockdown_native', 'Lockdown_flowaware', 'TriProbe']
labels = ['FedAvg', 'FedMedian', 'Krum', 'FLAME', 'RLR', 'DeepSight',
          'SHIELD-FL', 'Lockdown', 'Flow-Aware', 'TriProbe']
vals = [mean([d[k][s]['asr'] for s in S5]) * 100 for k in order]
cols = [OK['sky']] * 6 + [OK['orange']] + [OK['red']] * 2 + [OK['green']]

# top-to-bottom reading order: reverse for barh
y = range(len(vals))[::-1]
fig, ax = plt.subplots(figsize=(3.45, 3.5))
ax.barh(list(y), vals, color=cols, edgecolor='black', linewidth=0.5, height=0.72)
ax.set_xscale('log')
ax.set_xlim(0.3, 900)
ax.axvline(5, color=OK['black'], lw=1, ls=':')
ax.text(5.6, -0.75, 'threshold 5%', fontsize=7.5)
for yi, v in zip(y, vals):
    ax.text(v * 1.35, yi, f'{v:.2f}', va='center', fontsize=7.5)
ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=8.5)
ax.set_xlabel('Composite ASR (%, log scale)', fontsize=9)
fig.tight_layout()
for out in ['figures']:
    fig.savefig(f'{out}/fig_baseline_asr.pdf')
plt.close(fig)
print('wrote fig_baseline_asr.pdf')
