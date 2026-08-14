import os
os.makedirs("figures", exist_ok=True)
"""Generate mask-density-drift diagnostic figures (Sec 4.1/4.2/4.3) as PDFs.
Okabe-Ito colorblind-safe categorical palette, fixed order, one axis per panel.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    # Elsevier artwork policy allows Type 1 / TrueType only; matplotlib emits
    # Type 3 by default, so force TrueType (Type 42) embedding.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Okabe-Ito (CVD-safe). full=green(good), no-cap=sky, Lockdown=vermillion, flow=orange.
C = {"Lockdown_native": "#D55E00", "FlowAware_Lockdown": "#E69F00",
     "TriProbe_no_cap": "#56B4E9", "TriProbe_full": "#009E73"}
LBL = {"Lockdown_native": "Lockdown (native)", "FlowAware_Lockdown": "Flow-Aware Lockdown",
       "TriProbe_no_cap": "TriProbe w/o cap", "TriProbe_full": "TriProbe (full)"}
ORDER = ["Lockdown_native", "FlowAware_Lockdown", "TriProbe_no_cap", "TriProbe_full"]
RHO_CAP, THETA, M = 0.14, 11, 20
OUT = "figures/"

dd = json.load(open("outputs/density_drift_results.json"))

# ---------- Fig A: density (a) + pruning margin (b), 2 panels ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.45, 5.0))  # one column, stacked
for k in ORDER:
    r = dd[k]["rounds"]
    ax1.plot(r, dd[k]["mask_density_mean"], color=C[k], lw=2, label=LBL[k])
    ax2.plot(r, dd[k]["pruning_margin"], color=C[k], lw=2, label=LBL[k])
ax1.axhline(RHO_CAP, ls="--", lw=1.2, color="#555555")
ax1.text(1, RHO_CAP + 0.012, r"$\rho_{\max}=0.14$", color="#555555", fontsize=9)
ax1.set_xlabel("Communication round"); ax1.set_ylabel("Mean mask density")
ax1.set_title("(a) Effective mask density", fontsize=10); ax1.set_ylim(0, 0.55)
ax2.axhline(0, ls="--", lw=1.0, color="#999999")
ax2.set_xlabel("Communication round"); ax2.set_ylabel(r"Pruning margin ($\theta-\bar v$)")
ax2.set_title("(b) Pruning margin", fontsize=10)
ax1.legend(fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.13),
           ncol=2, frameon=False, columnspacing=1.0, handletextpad=0.4)
fig.tight_layout()
# the legend sits outside the axes, so include it in the saved bounding box
fig.savefig(OUT + "fig_density_margin.pdf", bbox_inches="tight"); plt.close(fig)

# ---------- Fig B: consensus vote distribution (final round) ----------
fig, ax = plt.subplots(figsize=(5.2, 3.6))
x = np.arange(M + 1)
# A vote count is a discrete variable, so the distribution is drawn as side-by-side
# bars rather than a polyline (which would imply interpolation between integers).
KEYS = ["Lockdown_native", "TriProbe_full"]
w = 0.4
for i, k in enumerate(KEYS):
    h = np.array(dd[k]["vote_hist"]["30"], dtype=float); h = h / h.sum()
    ax.bar(x + (i - 0.5) * w, h, w, color=C[k], edgecolor="black",
           linewidth=0.4, label=LBL[k])
# The distribution spans four orders of magnitude: the two modes hold ~38% of the
# mass while vote counts 10-19 hold 1e-4 to 1e-2. On a linear axis the middle of
# the range is sub-pixel, so a logarithmic axis is used.
ax.set_yscale("log")
ax.set_ylim(5e-5, 1.0)
ax.axvline(THETA, ls="--", lw=1.3, color="#555555")
ax.text(THETA + 0.25, 3e-1, r"$\theta$", color="#555555")
ax.set_xlabel("Vote count (clients keeping a position)")
ax.set_ylabel("Fraction of parameter positions")
# Title dropped and legend moved above the axes: with the log axis the bars now
# reach the top of the panel, so an in-panel legend would cover them.
ax.legend(fontsize=9, frameon=False, ncol=2, loc="lower center",
          bbox_to_anchor=(0.5, 1.01))
fig.tight_layout(); fig.savefig(OUT + "fig_votes.pdf"); plt.close(fig)

# ---------- Fig C: rho_max sensitivity ----------
rm = json.load(open("outputs/rhomax_sens_results.json"))
caps = ["0.08", "0.1", "0.12", "0.14", "0.16", "0.18", "0.2", "None"]
xs = list(range(len(caps)))
asr = [rm[c]["asr"] * 100 for c in caps]; ben = [rm[c]["benign"] * 100 for c in caps]
f1 = [rm[c]["macro_f1"] * 100 for c in caps]
fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot(xs, asr, color="#D55E00", lw=2, marker="s", ms=6, label="Composite ASR")
ax.plot(xs, ben, color="#009E73", lw=2, marker="o", ms=6, label="Benign accuracy")
ax.plot(xs, f1, color="#0072B2", lw=2, marker="^", ms=6, label="Macro-F1")
ax.axhline(5, ls=":", lw=1.0, color="#888888"); ax.text(0, 6.5, "5% threshold", fontsize=8, color="#888888")
ax.set_xticks(xs); ax.set_xticklabels(["0.08", "0.10", "0.12", "0.14", "0.16", "0.18", "0.20", "none"])
ax.set_xlabel(r"Hard density cap $\rho_{\max}$"); ax.set_ylabel("Percent (%)")
ax.set_title(r"Sensitivity to $\rho_{\max}$"); ax.legend(fontsize=9, loc="center left")
fig.tight_layout(); fig.savefig(OUT + "fig_rhomax.pdf"); plt.close(fig)

print("WROTE:", OUT + "fig_density_margin.pdf ; fig_votes.pdf ; fig_rhomax.pdf")
