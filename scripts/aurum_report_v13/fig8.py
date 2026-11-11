#!/usr/bin/env python3
"""Figure 8: live-daily v1.3 decision pathway — schematic + real decision card."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R = "/home/z/my-project/kaggle_work/aurum/results"
OUT = "/home/z/my-project/download/AURUM_v13_report/figures"
GOLD = "#C9A84C"; DARK = "#1A1A1A"; GREY = "#9AA0A6"; RED = "#B04A3A"
GREEN = "#7BA68A"; BLUE = "#6B9DAD"
plt.rcParams.update({"font.size": 10, "figure.dpi": 100, "savefig.dpi": 200})

d = json.load(open(f"{R}/live_daily_v13/decision.json"))
dec = d["decision"]; lc = d["lcb_ev"]; gb = d["gbdt_agreement"]
ov = d["overlay_path"]; sz = d["sizing_reference"]; pl = d["pooling"]

fig = plt.figure(figsize=(12.6, 5.6), constrained_layout=True)
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
ax = fig.add_subplot(gs[0, 0])
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=8.6, tc="#1A1A1A", lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                fc=fc, ec=ec, linewidth=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.45)

def arrow(x1, y1, x2, y2, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=1.3))

# Row 1: state + forwards
box(0.2, 8.1, 2.9, 1.5, "STATE (latest bar)\n2025-12-29 00:00 UTC\n338 tokens", "#F5F2E8", GOLD)
box(3.9, 8.1, 3.2, 1.5, "CPU FORWARDS r=1,2,3,4\nh1(48) h2 h3 h4 h5\nall finite", "#EDF3F5", BLUE)
box(7.9, 8.1, 1.9, 1.5, "GBDT fold-2025\ntrain <= 2024\n54 era-safe feats", "#EDF5EF", GREEN)
arrow(3.1, 8.85, 3.9, 8.85); arrow(7.1, 8.85, 7.9, 8.85)

# Row 2: pooling + LCB
box(0.2, 5.5, 4.4, 1.9,
    "MULTI-DEPTH POOL  (F6)\np_bar = .6 p1 + .2 p2 + .2 p3\nsigma_loop = std(p1,p2,p3)\nOOD = mean |p_r - p_3|  (r=1,2,4)",
    "#F5F2E8", GOLD)
box(5.2, 5.5, 4.6, 1.9,
    "LCB-EV per cell (48)\nEV = p_sh*b - (1-p_sh)\n       - 0.5*b*sigma_loop - 0.05/m\nev_min hurdle = 0 (F3)",
    "#F5F2E8", GOLD)
arrow(2.4, 8.1, 2.4, 7.4); arrow(7.5, 8.1, 7.5, 7.4)
arrow(4.6, 6.45, 5.2, 6.45)

# Row 3: gates
box(0.2, 3.0, 9.6, 1.7,
    "GATES (all must clear)   EV_LCB >= 0   |   0.01 < p_bar < 0.99   |   OOD <= 0.10   |   GBDT direction agrees\n"
    "NFP blackout   |   VIX shock   |   overlay levels valid   |   ATR finite   |   quarter-Kelly > 0",
    "#FDF7EF", GOLD)
arrow(2.4, 5.5, 2.4, 4.7); arrow(7.5, 5.5, 7.5, 4.7)

# Row 4: execution
box(0.2, 0.4, 4.4, 2.0,
    "TRADE PLAN (path overlay)\nentry open[i+1]; hard stop 1.0 ATR; TP 4.0 ATR\npartial TP 1.98 ATR (half)  |  trail 0.853 ATR\ntime stop 24 bars - orders placed at entry",
    "#EDF3F5", BLUE, fs=8.2)
box(5.2, 0.4, 4.6, 2.0,
    "RISK SIZING (cap=1)\nf = min(0.25 x Kelly, 1.0%)\nrisk USD 100 on $10,000 ref account\nsize 4.67 oz (0.047 lots)",
    "#EDF5EF", GREEN, fs=8.2)
arrow(2.4, 3.0, 2.4, 2.4); arrow(7.5, 3.0, 7.5, 2.4)
ax.set_title("(a) v1.3 decision pathway - CPU-only, zero lookahead", fontsize=11, loc="left")

# ---------------- panel (b): the actual decision card ----------------
ax2 = fig.add_subplot(gs[0, 1])
ax2.axis("off")
ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
ax2.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 9.4, boxstyle="round,pad=0.15",
                             fc="#1A1A1A", ec=GOLD, linewidth=1.6))
ax2.text(5.0, 9.0, "LIVE DECISION  -  as of 2025-12-29 00:00 UTC", ha="center",
         fontsize=10.5, color="#FFFFFF", fontweight="bold")
ax2.text(5.0, 8.1, "XAUUSD  (Gold / USD)  -  H1 bars", ha="center", fontsize=9, color="#B0B8C0")
rows = [
    ("Action", f"{dec['action']}  ({dec['direction']})", GOLD),
    ("Cell (stop/TP/horizon)", dec["cell"], None),
    ("p_win pooled (p_bar)", f"{dec['p_win_pooled']:.4f}", None),
    ("sigma_loop at cell", f"{pl['sigma_loop_cell']:.4f}", None),
    ("loop-OOD mean", f"{pl['ood_mean']:.4f}  (gate 0.10)", None),
    ("EV_LCB", f"{dec['ev_lcb_R']:+.4f} R", GOLD),
    ("GBDT fold-2025", f"{gb['dir']}  pL={gb['pL']:.3f} / pS={gb['pS']:.3f}  (agrees)", None),
    ("overlay", f"partial TP {ov['partial_tp_atr']} ATR, trail {ov['trail_mae_atr']} ATR, "
                f"stop {ov['hard_stop_atr']} ATR, TP {ov['take_profit_atr']} ATR", None),
    ("risk fraction", f"{dec['risk_fraction_capped_cap1']:.4f} (cap 1.0%; cap3 variant "
                      f"{dec['risk_fraction_capped_cap3']:.4f})", None),
    ("size @ $10k account", f"{sz['position_oz_cap1']:.2f} oz ({sz['position_lots_cap1']:.4f} lots), "
                            f"stop {sz['stop_distance_usd']:.1f} USD", None),
    ("spot close / ATR14", f"{sz['close_usd']:.2f} / {sz['atr14_h1_usd']:.2f} USD", None),
    ("era score (audit)", f"{d['gates']['era_score']:.3f}", None),
]
y = 7.2
for k, v, hl in rows:
    ax2.text(0.7, y, k, fontsize=8.6, color="#90989F", va="center")
    ax2.text(9.3, y, v, fontsize=8.6, ha="right", va="center",
             color=(GOLD if hl else "#E8E8E8"), fontweight=("bold" if hl else "normal"))
    y -= 0.62
ax2.plot([0.7, 9.3], [y + 0.31, y + 0.31], color="#3A3A3A", linewidth=0.8)
ax2.text(0.7, y - 0.05, "G6.0-G6.5 all PASS (6/6)", fontsize=8.4, color=GREEN, va="center")
ax2.text(9.3, y - 0.05, "CPU-only, 0 GPU-s", fontsize=8.4, color="#90989F", ha="right", va="center")
ax2.set_title("(b) decision.json snapshot", fontsize=11, loc="left")

fig.suptitle("Figure 8  Live daily pathway (Step 6a) - the v1.3 policy in production", fontsize=12.5)
fig.savefig(f"{OUT}/fig8_live_pathway.png", dpi=200)
print("saved fig8_live_pathway.png")
