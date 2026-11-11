#!/usr/bin/env python3
"""Figures for the AURUM engineering-verification report.
Fig 1: hindsight-argmax label dispersion vs EV optimum (data from audit_numbers.py).
Fig 2: walk-forward GPU-hour budget vs the 30 h/week Kaggle quota.
Style: IG-1 Ink Gold palette (accent #C9A84C), English labels, constrained_layout."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, GOLD, GRAY, LIGHT = "#1A1A1A", "#C9A84C", "#8A8F98", "#F5F2E8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRAY, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})

# ───────────────────────── Figure 1 ─────────────────────────
ev = np.load("fig1_pnl_mean.npy")            # E[P&L] per stop width (R multiples)
hist = np.load("fig1_hist.npy")              # hindsight-argmax label counts
stops = ["1.0", "1.5", "2.0", "2.5"]
share = hist / hist.sum() * 100
ev_best = int(np.argmax(ev))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.5), constrained_layout=True)
plt.rcParams.update({"font.size": 11})

colors1 = [GOLD if i == ev_best else LIGHT for i in range(4)]
edges1 = [INK if i == ev_best else GRAY for i in range(4)]
ax1.bar(stops, ev, color=colors1, edgecolor=edges1, width=0.62)
for i, v in enumerate(ev):
    ax1.text(i, v + 0.012, f"{v:+.3f}R", ha="center", va="bottom", fontsize=10.5,
             color=INK, fontweight="bold" if i == ev_best else "normal")
ax1.set_xlabel("Stop width (x ATR)", fontsize=12)
ax1.set_ylabel("Expected P&L per trade (R)", fontsize=12)
ax1.set_title("A. Population answer: E[P&L] by stop width", fontsize=11.5)
ax1.set_ylim(0, max(ev) * 1.22)
ax1.annotate("EV-optimal: 2.5x ATR", xy=(ev_best, ev[ev_best]),
             xytext=(ev_best - 1.15, ev[ev_best] * 1.06), fontsize=10.5, color=GOLD,
             fontweight="bold", arrowprops=dict(arrowstyle="->", color=GOLD))

colors2 = [GOLD if i == ev_best else "#C9C4B4" for i in range(4)]
ax2.bar(stops, share, color=colors2, edgecolor=GRAY, width=0.62)
for i, v in enumerate(share):
    ax2.text(i, v + 0.8, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5,
             color=INK, fontweight="bold" if i == ev_best else "normal")
ax2.set_xlabel("Hindsight-argmax label (x ATR)", fontsize=12)
ax2.set_ylabel("Share of paths (%)", fontsize=12)
ax2.set_title("B. Hindsight labels: 20,000 identical-state paths", fontsize=11.5)
ax2.set_ylim(0, 50)
ax2.axhline(25, color=GRAY, lw=0.8, ls=":")
ax2.text(2.98, 26.0, "uniform = pure noise", fontsize=10, color=GRAY, ha="right")
ax1.tick_params(labelsize=11); ax2.tick_params(labelsize=11)
fig.savefig("fig1_hindsight_bias.png", dpi=200)
print("saved fig1_hindsight_bias.png")

# ───────────────────────── Figure 2 ─────────────────────────
acts = [
    ("Stage A-prime CPT (150M tok)", 4.3, 15.4),
    ("Anchor SFT, Stage B (144M tok)", 4.2, 14.8),
    ("3x warm-start refits", 3.0, 11.1),
    ("OOS tables (r = 1, 2, 3)", 0.6, 2.3),
    ("Ablations + controls", 4.0, 8.0),
]
fig2, ax = plt.subplots(figsize=(8.6, 3.2), constrained_layout=True)
plt.rcParams.update({"font.size": 11})
labels = [a[0] for a in acts][::-1]
lo = np.array([a[1] for a in acts][::-1])
hi = np.array([a[2] for a in acts][::-1])
y = np.arange(len(acts))
ax.barh(y, hi - lo, left=lo, height=0.52, color=GOLD, edgecolor=INK, alpha=0.92)
for yi, (l, h) in enumerate(zip(lo, hi)):
    ax.text(h + 0.5, yi, f"{l:.0f}-{h:.0f} h", va="center", fontsize=11, color=INK)
ax.axvline(30, color=INK, lw=1.6, ls="--")
ax.text(30.7, len(acts) - 0.45, "30 GPU-h / week", fontsize=11.5, color=INK, fontweight="bold")
ax.set_yticks(y, labels, fontsize=11.5)
ax.set_xlabel("GPU-hours on 2 x T4", fontsize=12)
ax.set_xlim(0, 40)
ax.set_ylim(-0.6, len(acts) - 0.4)
ax.tick_params(labelsize=11)
fig2.savefig("fig2_walkforward_budget.png", dpi=200)
print("saved fig2_walkforward_budget.png")
