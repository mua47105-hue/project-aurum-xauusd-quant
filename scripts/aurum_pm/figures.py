#!/usr/bin/env python3
"""Figures for the AURUM Phase 0/1 post-mortem report.
Fig 1: full-period walk-forward yearly mean R (exp2b) - learned vs references.
Fig 2: the era-drift vs signal frontier (exp3 series) + neutralization.
Style: IG-1 Ink Gold palette (accent #C9A84C), English labels, constrained_layout."""
import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GOLD, GRAY, LIGHT = "#1A1A1A", "#C9A84C", "#8A8F98", "#F5F2E8"
RED = "#B0533A"; BLUE = "#4A6FA5"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRAY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = "/home/z/my-project/kaggle_work/aurum/local/results"

# ───────────────────────── Figure 1: walk-forward ─────────────────────────
wf = json.load(open(os.path.join(RES, "exp2b_walkforward.json")))
years = sorted(wf["years"].keys())
series = {
    "T5 shallow realized-target (fix)": [wf["years"][y]["T5_shallow"]["meanR"] for y in years],
    "T0 cohort-target (status quo)": [wf["years"][y]["T0_cohort"]["meanR"] for y in years],
    "k-NN strategy (cohort direct)": [wf["years"][y]["cohort_direct"]["meanR"] for y in years],
    "Canonical long cell (beta ref)": [wf["years"][y]["canonical_long"]["meanR"] for y in years],
}
fig, ax = plt.subplots(figsize=(9.2, 3.9), constrained_layout=True)
x = np.arange(len(years)); w = 0.2
styles = [(GOLD, "T5 shallow realized-target (fix)"), (INK, "T0 cohort-target (status quo)"),
          (GRAY, "k-NN strategy (cohort direct)"), (BLUE, "Canonical long cell (beta ref)")]
for i, (color, name) in enumerate(styles):
    vals = np.array(series[name]) * 1000  # milli-R
    ax.bar(x + (i - 1.5) * w, vals, width=w * 0.92, color=color,
           edgecolor=color, label=name, zorder=3)
ax.axhline(0, color=INK, lw=0.8, zorder=2)
ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_ylabel("Mean realized R per trade (mR)", fontsize=11)
ax.set_title("Full-period walk-forward, expanding window, EV-argmax over 8 cells (p>0 filter)",
             fontsize=11.5)
ax.legend(fontsize=9, frameon=False, ncol=2, loc="upper left")
means = {n: np.mean(v) * 1000 for n, v in series.items()}
txt = "yearly means (mR):  " + "   ".join(
    f"{n.split(' (')[0]}: {v:+.1f}" for n, v in means.items())
ax.text(0.0, -0.24, txt, transform=ax.transAxes, fontsize=9, color=INK)
fig.savefig(os.path.join(HERE, "fig1_walkforward.png"), dpi=200)
print("fig1 saved")

# ───────────────────────── Figure 2: drift-signal frontier ─────────────────────────
pts = [
    (0.8048, 0.533, "raw 31 feats (status quo)", INK),
    (0.8688, None, "naive trailing rank", RED),
    (0.7724, None, "quantile-map to train CDF", GRAY),
    (0.6166, 0.482, "qmap + drop worst 8", RED),
    (0.9928, None, "macro innovations (joint)", BLUE),
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
for auc_d, auc_s, name, color in pts:
    ax1.scatter([auc_d], [0.5 if auc_s is None else auc_s], s=70, color=color, zorder=3)
    ax1.annotate(name, (auc_d, 0.5 if auc_s is None else auc_s),
                 textcoords="offset points", xytext=(6, 6), fontsize=8.5, color=color)
ax1.axhline(0.5, color=GRAY, lw=0.8, ls="--")
ax1.axvline(0.60, color=GOLD, lw=1.0, ls="--")
ax1.set_xlabel("Adversarial AUC (2012-22 vs 2023-25)", fontsize=11)
ax1.set_ylabel("Realized-outcome AUC (2023-25)", fontsize=11)
ax1.set_title("A. Drift vs signal: no free lunch", fontsize=11.5)
ax1.set_xlim(0.55, 1.03); ax1.set_ylim(0.44, 0.57)
ax1.annotate("signal destroyed (AUC<0.5)", (0.6166, 0.482), textcoords="offset points",
             xytext=(10, -16), fontsize=8.5, color=RED)
ax1.annotate("gate target 0.60", (0.605, 0.552), fontsize=8.5, color=GOLD)

ax2.bar([0, 1], [0.53167, 0.53337], width=0.5, color=[GRAY, GOLD], edgecolor=INK)
ax2.bar([2, 3], [0.54115, 0.54208], width=0.5, color=[GRAY, GOLD], edgecolor=INK)
for i, v in enumerate([0.53167, 0.53337, 0.54115, 0.54208]):
    ax2.text(i, v + 0.0012, f"{v:.3f}", ha="center", fontsize=9.5, color=INK)
ax2.axhline(0.5, color=GRAY, lw=0.8, ls="--")
ax2.set_xticks(range(4))
ax2.set_xticklabels(["canon-L\nraw", "canon-L\nneutral.", "canon-S\nraw", "canon-S\nneutral."],
                    fontsize=9)
ax2.set_ylabel("Realized-outcome AUC (2023-25)", fontsize=11)
ax2.set_title("B. Neutralization keeps the signal", fontsize=11.5)
ax2.set_ylim(0.48, 0.56)
fig.savefig(os.path.join(HERE, "fig2_frontier.png"), dpi=200)
print("fig2 saved")
