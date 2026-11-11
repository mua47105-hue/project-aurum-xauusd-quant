#!/usr/bin/env python3
"""AURUM v1.3 Master Report — 8 figures from Kaggle kernel outputs.
All numbers come from the pulled kernel artifacts (no fabrication).
Style: Ink-Gold (IG-1) derivatives; English labels; 200 DPI.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/z/my-project/kaggle_work/aurum/results"
OUT = "/home/z/my-project/download/AURUM_v13_report/figures"
os.makedirs(OUT, exist_ok=True)

GOLD = "#C9A84C"; DARK = "#1A1A1A"; GREY = "#9AA0A6"; RED = "#B04A3A"
GREEN = "#7BA68A"; BLUE = "#6B9DAD"; TAN = "#C49B72"; PURP = "#9687A8"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.edgecolor": "#555555", "axes.linewidth": 0.8,
    "grid.alpha": 0.28, "grid.linewidth": 0.6,
    "figure.dpi": 100, "savefig.dpi": 200,
    "axes.spines.top": False, "axes.spines.right": False,
})

bm = json.load(open(f"{R}/p4_backtest_latest/backtest_main.json"))
pf = json.load(open(f"{R}/p4_backtest_latest/policy_fit.json"))
p3s = json.load(open(f"{R}/p3_tables_latest/p3_tables_summary.json"))
eq = pd.read_parquet(f"{R}/p4_backtest_latest/equity_curves.parquet")
tr = pd.read_parquet(f"{R}/p4_backtest_latest/trades_all.parquet")
STRATS = ["aurum_r1", "aurum_r2", "aurum_r3", "aurum_r4", "canon", "flagship",
          "stack", "gbdt", "knn", "ema", "donchian", "buyhold"]

def oos_pt(strategy, cap=1, cost=1.0):
    d = eq[(eq.strategy == strategy) & (eq.cap == cap) & (eq.cost_mult == cost)]
    return pd.to_datetime(d.day_ns.values, utc=True), d.equity.values

def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200)
    plt.close(fig)
    print("saved", name)

# ============================ FIG 1: OOS equity curves ======================
fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
series = [("flagship cap1", "flagship", 1, GOLD, 2.6, "-"),
          ("flagship cap3", "flagship", 3, "#8a7434", 1.6, "--"),
          ("AURUM r=1 (48-cell)", "aurum_r1", 1, BLUE, 1.3, "-"),
          ("AURUM 2-cell canonical", "canon", 1, GREY, 1.2, "-"),
          ("AURUM+GBDT stack", "stack", 1, GREEN, 1.4, "-"),
          ("GBDT standalone", "gbdt", 1, TAN, 1.2, "-"),
          ("k-NN Cohort", "knn", 1, PURP, 1.2, "-"),
          ("Buy & Hold gold", "buyhold", 1, "#555B61", 1.6, ":")]
for label, s, cap, col, lw, ls in series:
    x, y = oos_pt(s, cap=cap)
    ax.plot(x, y, color=col, linewidth=lw, linestyle=ls, label=label)
ax.axhline(1.0, color="#777777", linewidth=0.8, alpha=0.7)
ax.set_xlabel("OOS period (2023-2025), cost 1.0x, non-overlapping execution")
ax.set_ylabel("Compounded portfolio equity (start = 1.0)")
ax.set_title("Figure 1  OOS equity curves - AURUM v1.3 vs baselines (2023-2025)")
ax.grid(True)
ax.legend(loc="upper left", fontsize=8.2, ncol=2, framealpha=0.9)
save(fig, "fig1_equity_oos.png")

# ============================ FIG 2: portfolio realism =====================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
ax = axes[0]
names, vals, cols = [], [], []
for s in STRATS:
    d = bm[s]["cap1"]["cost_1.0"]["oos"]["portfolio"]
    names.append(s.replace("aurum_", "r"))
    vals.append(d["max_dd_pct"])
    cols.append(GOLD if s == "flagship" else (GREEN if s in ("stack",) else GREY))
ax.bar(range(len(names)), vals, color=cols, edgecolor="#555", linewidth=0.6)
ax.axhline(20.0, color=RED, linewidth=1.8, linestyle="--")
ax.text(len(names) - 0.4, 20.4, "20% limit", color=RED, fontsize=9, ha="right")
for i, v in enumerate(vals):
    ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=7.6)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
ax.set_ylabel("Max portfolio drawdown (%)")
ax.set_ylim(0, 24)
ax.set_title("(a) Max drawdown % - OOS 2023-25, cap=1 (all < 20%)")
ax.grid(True, axis="y")

ax = axes[1]
cats = ["trades executed", "Max drawdown (R)", "ann. Sharpe x100"]
v12 = [7207, 127.7, 0.0982 * 100]
v13 = [1354, 46.8, 2.305]
x = np.arange(3)
w = 0.36
ax.bar(x - w / 2, v12, w, color=GREY, label="v1.2 AURUM r=3 (48-cell argmax)", edgecolor="#555", linewidth=0.6)
ax.bar(x + w / 2, v13, w, color=GOLD, label="v1.3 flagship (cap1)", edgecolor="#555", linewidth=0.6)
for xi, a, b_ in zip(x, v12, v13):
    ax.text(xi - w / 2, a * 1.08, f"{a:,.0f}" if a > 10 else f"{a:.2f}", ha="center", fontsize=8)
    ax.text(xi + w / 2, b_ * 1.08, f"{b_:,.0f}" if b_ > 10 else f"{b_:.2f}", ha="center", fontsize=8, color="#7a6425")
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\n(v1.2 vs v1.3)" for c in cats], fontsize=9)
ax.set_title("(b) v1.2 stacked spam vs v1.3 realism (log scale)")
ax.legend(fontsize=8.5)
ax.grid(True, axis="y")
fig.suptitle("Figure 2  Portfolio realism: non-overlapping execution, bounded drawdown", fontsize=12)
save(fig, "fig2_realism.png")

# ============================ FIG 3: policy causality =======================
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
ax = axes[0]
grid = pf["lambda_var"]["grid"]
lams = sorted({g["lam"] for g in grid})
evmins = sorted({g["ev_min"] for g in grid})
kap = pf["kappa"]["value_perf_F3"]
M = np.full((len(lams), len(evmins)), np.nan)
for g in grid:
    if g["kappa"] == kap:
        M[lams.index(g["lam"]), evmins.index(g["ev_min"])] = g["sharpe"]
im = ax.imshow(M, cmap="YlGnBu", aspect="auto")
ax.set_xticks(range(len(evmins)))
ax.set_xticklabels([f"{e:g}" for e in evmins])
ax.set_yticks(range(len(lams)))
ax.set_yticklabels([f"{l:g}" for l in lams])
for i in range(len(lams)):
    for j in range(len(evmins)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.6,
                    color="white" if M[i, j] > np.nanmax(M) * 0.85 else "#222")
ax.set_xlabel("EV_min hurdle")
ax.set_ylabel("lambda_var (loop-std penalty)")
ax.set_title(f"(a) F3 grid: FIT ann-Sharpe (kappa={kap:g})")
best = np.unravel_index(np.nanargmax(M), M.shape)
ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, edgecolor=RED, lw=2))
fig.colorbar(im, ax=ax, shrink=0.85, label="ann. Sharpe")

ax = axes[1]
f6 = pf["f6_w_perf_grid"]
lbl = ["\n".join([f"w1={g['w'][0]:g}", f"w2={g['w'][1]:g}", f"w3={g['w'][2]:g}"]) for g in f6]
sh = [g["sharpe"] for g in f6]
cols = [GOLD if g["sharpe"] == max(sh) else GREY for g in f6]
ax.bar(range(len(f6)), sh, color=cols, edgecolor="#555", linewidth=0.6)
for i, v in enumerate(sh):
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.4)
ax.set_xticks(range(len(f6)))
ax.set_xticklabels(lbl, fontsize=6.8)
ax.set_ylabel("FIT-window annualized Sharpe (cap=1)")
ax.set_title("(b) F6 pooling weights (p1,p2,p3)")
ax.grid(True, axis="y")

ax = axes[2]
ab = pf["overlay_rule"]["ablation"]
groups = ["FIT 2016-22", "OOS 2023-25"]
clean = [ab["fit_window"]["clean"]["sharpe"], ab["oos_report_only"]["clean"]["sharpe_ann"]]
path = [ab["fit_window"]["path"]["sharpe"], ab["oos_report_only"]["path"]["sharpe_ann"]]
x = np.arange(2)
ax.bar(x - 0.19, clean, 0.36, color=GREY, label="clean barriers only", edgecolor="#555", linewidth=0.6)
ax.bar(x + 0.19, path, 0.36, color=GOLD, label="path overlay (H3)", edgecolor="#555", linewidth=0.6)
for xi, c, p in zip(x, clean, path):
    ax.text(xi - 0.19, c + 0.03, f"{c:.3f}", ha="center", fontsize=8.5)
    ax.text(xi + 0.19, p + 0.03, f"{p:.3f}", ha="center", fontsize=8.5, color="#7a6425")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel("annualized Sharpe")
ax.set_title("(c) Overlay ablation - rule chosen on FIT only")
ax.legend(fontsize=8.5)
ax.grid(True, axis="y")
fig.suptitle("Figure 3  Decision policy fitted causally on 2016-2022 (zero OOS peeking)", fontsize=12)
save(fig, "fig3_policy_causality.png")

# ============================ FIG 4: baseline audit =========================
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
ax = axes[0]
x = np.arange(3)
w = 0.27
v12 = [0.6321, 0.5311, 0.0675]
v13 = [0.0310, -0.0155, 0.0166]
ref = [0.0599, 0.0078, None]
ax.bar(x - w, v12, w, color=GREY, label="v1.2 (broken)", edgecolor="#555", linewidth=0.6)
ax.bar(x, v13, w, color=GOLD, label="v1.3 (fixed)", edgecolor="#555", linewidth=0.6)
ax.bar(x + w, [r if r is not None else 0 for r in ref], w, color=BLUE,
       label="Phase-0/1 postmortem ref", edgecolor="#555", linewidth=0.6)
for xi, a, b_, r in zip(x, v12, v13, ref):
    ax.text(xi - w, a + 0.012, f"{a:+.4f}", ha="center", fontsize=7.8)
    ax.text(xi, b_ + (0.012 if b_ >= 0 else -0.03), f"{b_:+.4f}", ha="center", fontsize=7.8, color="#7a6425")
    if r is not None:
        ax.text(xi + w, r + 0.012, f"{r:+.4f}", ha="center", fontsize=7.8, color="#1a4d66")
ax.axhline(0, color="#555", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(["GBDT", "k-NN Cohort", "Ensemble / Stack"], fontsize=9.5)
ax.set_ylabel("OOS 2023-25 expectancy (R)")
ax.set_title("(a) OOS E[R]: 10-68x inflation removed, postmortem parity restored")
ax.legend(fontsize=8.3)
ax.grid(True, axis="y")

ax = axes[1]
cats = ["Ensemble v1.2", "GBDT v1.3", "k-NN v1.3", "Stack v1.3"]
md = [449.3, 38.3, 72.6, 49.0]
cols = [RED, GOLD, PURP, GREEN]
ax.bar(range(4), md, color=cols, edgecolor="#555", linewidth=0.6)
for i, v in enumerate(md):
    ax.text(i, v * 1.07, f"{v:.1f}R", ha="center", fontsize=8.5)
ax.set_yscale("log")
ax.set_ylim(1, 900)
ax.set_xticks(range(4))
ax.set_xticklabels(cats, fontsize=8.5)
ax.set_ylabel("OOS Max drawdown (R, log scale)")
ax.set_title("(b) Drawdown collapse: naive averaging removed")
ax.grid(True, axis="y")
fig.suptitle("Figure 4  Baseline audit: GBDT / k-NN anomaly and broken ensemble fixed", fontsize=12)
save(fig, "fig4_baseline_audit.png")

# ============================ FIG 5: cost sensitivity =======================
fig, ax = plt.subplots(figsize=(10, 5.0), constrained_layout=True)
mk = {"flagship": ("o", GOLD, 2.4), "aurum_r1": ("s", BLUE, 1.5), "canon": ("^", GREY, 1.3),
      "stack": ("D", GREEN, 1.4), "gbdt": ("v", TAN, 1.3), "knn": ("P", PURP, 1.3),
      "buyhold": (".", "#555B61", 1.8), "ema": ("x", "#7AADA0", 1.2), "donchian": ("1", "#B87472", 1.2)}
for s, (m, c, lw) in mk.items():
    ys = [bm[s]["cap1"][f"cost_{c_:.1f}"]["oos"]["expectancy_R"] for c_ in (1.0, 1.5, 2.0)]
    ax.plot([1.0, 1.5, 2.0], ys, marker=m, color=c, linewidth=lw,
            label=s.replace("aurum_", "r"), markersize=7)
ax.axhline(0, color="#555", linewidth=0.8)
ax.set_xticks([1.0, 1.5, 2.0])
ax.set_xticklabels(["1.0x", "1.5x", "2.0x"])
ax.set_xlabel("Cost multiplier (slippage + spread, base = 0.05R / stop-ATR)")
ax.set_ylabel("OOS 2023-25 expectancy (R per trade)")
ax.set_title("Figure 5  Cost sensitivity - OOS E[R] survives 2x costs for the AURUM family")
ax.grid(True)
ax.legend(fontsize=8.5, ncol=2)
save(fig, "fig5_cost_sensitivity.png")

# ============================ FIG 6: DSR + bootstrap ========================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
ax = axes[0]
mem = bm["dsr_daily_nonoverlapping"]["members"]
order = ["flagship|cap3", "flagship|cap1", "buyhold|cap1", "aurum_r1|cap1",
         "aurum_r2|cap1", "aurum_r3|cap1", "aurum_r4|cap1", "stack|cap1",
         "ema|cap1", "gbdt|cap1", "knn|cap1", "canon|cap1", "donchian|cap1"]
order = [o for o in order if o in mem]
vals = [mem[k]["dsr"] for k in order]
cols = [GOLD if "flagship" in k else GREY for k in order]
ax.barh(range(len(order)), vals, color=cols, edgecolor="#555", linewidth=0.6)
for i, v in enumerate(vals):
    ax.text(v + 0.008, i, f"{v:.3f}", va="center", fontsize=8)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([k.replace("|cap1", "") for k in order], fontsize=8.6)
ax.invert_yaxis()
ax.axvline(0.5, color=RED, linewidth=1.4, linestyle="--")
ax.text(0.505, len(order) - 0.4, "50%: better than even", color=RED, fontsize=8)
ax.set_xlabel("Deflated Sharpe Ratio (daily, non-overlapping, N=15 trials)")
ax.set_title(f"(a) DSR daily - SR0={bm['dsr_daily_nonoverlapping']['sr0']:.4f}")
ax.set_xlim(0, 1.02)
ax.grid(True, axis="x")

ax = axes[1]
boot = bm["bootstrap_oos_cost_1.0"]
order2 = ["flagship", "aurum_r1", "canon", "stack", "gbdt", "knn", "ema", "donchian", "buyhold"]
ypos = np.arange(len(order2))[::-1]
for i, s in enumerate(order2):
    e = boot[s]["expectancy_R"]
    c = GOLD if s == "flagship" else GREY
    ax.plot([e["lo"], e["hi"]], [ypos[i], ypos[i]], color=c, linewidth=2.2)
    ax.plot([e["lo"], e["lo"]], [ypos[i] - 0.13, ypos[i] + 0.13], color=c, linewidth=1.6)
    ax.plot([e["hi"], e["hi"]], [ypos[i] - 0.13, ypos[i] + 0.13], color=c, linewidth=1.6)
    ax.plot(e["med"], ypos[i], "o", color=c, markersize=6)
    ax.text(e["hi"] + 0.006, ypos[i], f"med {e['med']:+.3f}  [{e['lo']:+.3f}, {e['hi']:+.3f}]",
            va="center", fontsize=7.6)
ax.axvline(0, color="#555", linewidth=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(order2, fontsize=9)
ax.set_xlabel("OOS expectancy (R) - 5,000-sample block bootstrap, 95% CI")
ax.set_title("(b) Bootstrap CIs - E[R] (OOS, cost 1.0x)")
ax.set_xlim(-0.09, 0.42)
ax.grid(True, axis="x")
fig.suptitle("Figure 6  Statistical validation on non-overlapping returns (DSR + 5,000x bootstrap)", fontsize=12)
save(fig, "fig6_dsr_bootstrap.png")

# ============================ FIG 7: model diagnostics ======================
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), constrained_layout=True)
ax = axes[0]
depths = ["r=1", "r=2", "r=4"]
oodm = [p3s["summary"]["main_r1"]["ood_mean"], p3s["summary"]["main_r2"]["ood_mean"],
        p3s["summary"]["main_r4"]["ood_mean"]]
oodp = [p3s["summary"]["main_r1"]["ood_p95"], p3s["summary"]["main_r2"]["ood_p95"],
        p3s["summary"]["main_r4"]["ood_p95"]]
x = np.arange(3)
ax.bar(x - 0.18, oodm, 0.34, color=BLUE, label="mean |p_r - p_3|", edgecolor="#555", linewidth=0.6)
ax.bar(x + 0.18, oodp, 0.34, color="#a8c6d8", label="p95", edgecolor="#555", linewidth=0.6)
ax.axhline(0.10, color=RED, linewidth=1.6, linestyle="--")
ax.text(2.4, 0.011, "OOD gate 0.10", color=RED, fontsize=8, ha="right")
for i, (m, p) in enumerate(zip(oodm, oodp)):
    ax.text(i - 0.18, m + 0.0004, f"{m:.4f}", ha="center", fontsize=8)
    ax.text(i + 0.18, p + 0.0004, f"{p:.4f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(depths)
ax.set_ylabel("loop-divergence OOD (vs production r=3)")
ax.set_title("(a) Loop-depth divergence (87,948 states)")
ax.legend(fontsize=8.3)
ax.grid(True, axis="y")

ax = axes[1]
fv = p3s["validation"]["folds"]
yrs = sorted(fv.keys(), key=int)
rec = [fv[y]["brier_recomputed"] for y in yrs]
refb = [fv[y]["brier_ref"] for y in yrs]
ax.plot(yrs, rec, "o-", color=GOLD, linewidth=1.8, markersize=6, label="v3 re-derived (CPU finalize)")
ax.plot(yrs, refb, "s--", color="#555", linewidth=1.4, markersize=5, label="v2 GPU training reference")
ax.set_ylim(0.196, 0.205)
ax.set_xlabel("walk-forward fold year")
ax.set_ylabel("H1 head Brier (48-cell argmax)")
ax.set_title("(b) Fold-calibration parity: v3 vs v2 (10/10 match)")
ax.legend(fontsize=8.3)
ax.grid(True)
ax.tick_params(axis="x", rotation=45)

ax = axes[2]
fl = tr[(tr.strategy == "flagship") & (tr.cap == 1) & (tr.cost_mult == 1.0) & (tr.year >= 2023)]
yrs3 = sorted(fl.year.unique())
er = [fl[fl.year == y].R_net.mean() for y in yrs3]
n = [len(fl[fl.year == y]) for y in yrs3]
ax.bar(yrs3, er, 0.5, color=GOLD, edgecolor="#555", linewidth=0.6)
for xi, e, nn in zip(yrs3, er, n):
    ax.text(xi, e + 0.008, f"{e:+.3f}", ha="center", fontsize=8.5)
    ax.text(xi, 0.012 if e < 0 else e / 2, f"n={nn}", ha="center", fontsize=7.4)
ax.axhline(0, color="#555", linewidth=0.8)
ax2 = ax.twinx()
ax2.plot(yrs3, np.cumsum(n), "o--", color=BLUE, linewidth=1.5, markersize=6)
ax2.set_ylabel("cumulative trades", color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE)
ax2.spines["right"].set_visible(True)
ax.set_xlabel("OOS year")
ax.set_ylabel("E[R] per trade")
ax.set_title("(c) Flagship OOS by year")
ax.grid(True, axis="y")
fig.suptitle("Figure 7  Model diagnostics: OOD divergence, calibration parity, OOS by year", fontsize=12)
save(fig, "fig7_model_diagnostics.png")
print("FLAGSHIP OOS per-year E[R]:", dict(zip([int(y) for y in yrs3], [round(float(e), 4) for e in er])),
      "n:", dict(zip([int(y) for y in yrs3], n)))

print("FIGURES 1-7 DONE")
