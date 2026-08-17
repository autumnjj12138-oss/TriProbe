import os
os.makedirs("figures", exist_ok=True)
"""Plot the main result figures from the stored experiment outputs."""
import os, json
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "grid.linestyle": "--", "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42})
OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "black": "#000000"}
FIG = "figures"
S5 = ['42','123','3407','2025','666']
def L(p): return json.load(open(p))

# ---- Fig A: baseline composite ASR bar ----
d = dict(L('outputs/baseline_5seed_results.json')['results'])
d.update(L('outputs/shieldfl_5seed_results.json')['results'])  # SHIELD-FL (KBS 2025)
order = ['FedAvg','FedMedian','Krum','FLAME','RLR','DeepSight','SHIELD-FL','Lockdown_native','Lockdown_flowaware','TriProbe']
labels = ['FedAvg','FedMedian','Krum','FLAME','RLR','DeepSight','SHIELD-FL','Lockdown','Flow-Aware','TriProbe']
vals = [mean([d[k][s]['asr'] for s in S5])*100 for k in order]
cols = [OK['sky']]*6 + [OK['orange']] + [OK['red']]*2 + [OK['green']]
fig, ax = plt.subplots(figsize=(7, 2.7)) 
b = ax.bar(range(len(vals)), vals, color=cols, edgecolor='black', linewidth=0.6)
# ASR spans two orders of magnitude (0.86% to 100%); a linear axis would render
# the TriProbe bar sub-pixel, so the axis is logarithmic.
ax.set_yscale('log'); ax.set_ylim(0.3, 400)
ax.axhline(5, color=OK['black'], lw=1, ls=':'); ax.text(-0.4, 5.8, 'threshold 5%', fontsize=9)
for i, v in enumerate(vals): ax.text(i, v * 1.18, f'{v:.2f}', ha='center', fontsize=8.5)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_ylabel('Composite ASR (%, log scale)')
fig.savefig(f"{FIG}/fig_baseline_asr.pdf"); plt.close(fig)

# ---- Fig B: trigger types grouped bar (nodef vs TriProbe) ----
ss = L('outputs/single_niid_results.json')['results']['single_subspace']
main = L('outputs/cic_main_f1_results.json')['results']
types = ['Header-only','Temporal-only','Standard','Composite']
nodef = [mean([ss['header_only_FedAvg'][s]['asr'] for s in S5])*100,
         mean([ss['temporal_only_FedAvg'][s]['asr'] for s in S5])*100,
         mean([main[s]['FedAvg_std']['asr'] for s in S5])*100,
         mean([d['FedAvg'][s]['asr'] for s in S5])*100]
tri = [mean([ss['header_only_TriProbe'][s]['asr'] for s in S5])*100,
       mean([ss['temporal_only_TriProbe'][s]['asr'] for s in S5])*100,
       mean([main[s]['TriProbe_std']['asr'] for s in S5])*100,
       mean([main[s]['TriProbe_comp']['asr'] for s in S5])*100]
import numpy as np
x = np.arange(len(types)); w = 0.38
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.bar(x-w/2, nodef, w, label='No defense', color=OK['sky'], edgecolor='black', linewidth=0.6)
ax.bar(x+w/2, tri, w, label='TriProbe', color=OK['green'], edgecolor='black', linewidth=0.6)
# No-defense and TriProbe differ by three orders of magnitude (77.33% vs 0.08%),
# so a logarithmic axis is used to keep every bar visible.
ax.set_yscale('log'); ax.set_ylim(0.03, 400)
for i,v in enumerate(nodef): ax.text(i-w/2, v*1.25, f'{v:.1f}', ha='center', fontsize=7)
for i,v in enumerate(tri): ax.text(i+w/2, v*1.25, f'{v:.2f}', ha='center', fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(types, rotation=18, ha='right', fontsize=9)
ax.set_ylabel('ASR (%, log scale)', fontsize=10)
# Legend placed above the axes so it cannot cover the bars or their value labels.
ax.legend(frameon=False, ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.01),
          fontsize=8.5, columnspacing=1.0, handletextpad=0.4)
fig.savefig(f"{FIG}/fig_trigger_types.pdf"); plt.close(fig)

# ---- Fig C: q sweep line (ASR + benign) ----
q = L('outputs/opus_drop_sensitivity_results.json')
qr = q.get('results', q)
qs = ['0.15','0.20','0.25','0.30','0.35','0.40']
qa = [qr[k]['asr']*100 for k in qs]; qb = [qr[k]['benign']*100 for k in qs]
fig, ax1 = plt.subplots(figsize=(5.2, 3.3))
ax1.plot([float(x) for x in qs], qa, 'o-', color=OK['red'], label='Composite ASR')
ax1.set_xlabel('ASF drop quantile q'); ax1.set_ylabel('ASR (%)', color=OK['red'])
ax1.axhline(5, color='gray', ls=':', lw=1)
ax2 = ax1.twinx(); ax2.plot([float(x) for x in qs], qb, 's--', color=OK['blue'], label='Benign Acc')
ax2.set_ylabel('Benign Acc (%)', color=OK['blue']); ax2.set_ylim(90, 98); ax2.grid(False)
fig.savefig(f"{FIG}/fig_q_sweep.pdf"); plt.close(fig)

# ---- Fig D: probe size line ----
pb = L('outputs/opus_probe_sensitivity_results.json'); pbr = pb.get('results', pb)
Bs = ['32','64','128','256','512']; pa = [pbr[k]['asr']*100 for k in Bs]
fig, ax = plt.subplots(figsize=(5.2, 3.3))
ax.plot(range(len(Bs)), pa, 'o-', color=OK['green'])
ax.set_xticks(range(len(Bs))); ax.set_xticklabels(Bs)
ax.set_xlabel('Probe size B'); ax.set_ylabel('Composite ASR (%)'); ax.set_ylim(0, 3)
ax.axhline(2, color='gray', ls=':', lw=1)
fig.savefig(f"{FIG}/fig_probe_size.pdf"); plt.close(fig)

# ---- Fig E: Non-IID three lines ----
sn = L('outputs/single_niid_results.json')['results']
al = ['0.3','0.5','1.0','5.0']; xa = [float(a) for a in al]
nd = [sn['noniid_nodef'][a]['asr']*100 for a in al]
nb = [sn['noniid_benignprobe'][a]['asr']*100 for a in al]
tp = [sn['noniid'][a]['asr']*100 for a in al]
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(xa, nd, 'o-', color=OK['sky'], label='No defense')
ax.plot(xa, nb, 's--', color=OK['orange'], label='Benign-probe ASF')
ax.plot(xa, tp, '^-', color=OK['green'], label='TriProbe')
ax.set_xscale('log'); ax.set_xticks(xa); ax.set_xticklabels(al)
# TriProbe stays near 1% while the undefended baseline is near 70%; a log axis
# keeps both readable in one panel.
ax.set_yscale('log'); ax.set_ylim(0.2, 200)
ax.axhline(5, color='gray', ls=':', lw=1)
ax.set_xlabel(r'Dirichlet $\alpha$ (smaller = more heterogeneous)')
ax.set_ylabel('Composite ASR (%, log scale)')
# Legend above the axes: the three curves span the full width of the panel.
ax.legend(frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01),
          columnspacing=1.1, handletextpad=0.5, fontsize=9)
fig.savefig(f"{FIG}/fig_noniid.pdf"); plt.close(fig)
print("generated 5 figures OK")
