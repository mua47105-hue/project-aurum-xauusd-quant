#!/usr/bin/env python3
"""Generate all Master Report figures from real pulled result files.
Outputs PNGs to download/report_figs/. Fails loudly if data is missing."""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for f in ("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.environ.get("AURUM_RESULTS", "/home/z/my-project/kaggle_work/aurum")
OUT = "/home/z/my-project/download/report_figs"
os.makedirs(OUT, exist_ok=True)

# palette (IG-1 Ink Gold companion colors for figures)
GOLD = "#C9A84C"; INK = "#1A1A1A"; BLUE = "#1284BA"; GREEN = "#2A7A65"
RED = "#B03A2E"; GREY = "#8A8F98"; TEAL = "#529286"

AURUM = ["aurum_r1", "aurum_r2", "aurum_r3", "aurum_r4"]
BASE = ["gbdt", "cohort", "ema", "donchian", "buyhold"]
ENS = ["aurum_canon", "ens_raw", "ens_neut"]
PRETTY = {"aurum_r1": "AURUM r=1", "aurum_r2": "AURUM r=2", "aurum_r3": "AURUM r=3",
          "aurum_r4": "AURUM r=4", "aurum_canon": "AURUM canon-2cell",
          "ens_raw": "Ensemble (raw)", "ens_neut": "Ensemble (neut)",
          "gbdt": "GBDT", "cohort": "k-NN Cohort", "ema": "EMA 50/200",
          "donchian": "Donchian 20", "buyhold": "Buy & Hold"}
AUC = {  # accent colors per strategy family
    **{a: GOLD for a in AURUM}, "ens_raw": TEAL, "ens_neut": TEAL,
    "aurum_canon": "#D4B96A", "gbdt": BLUE, "cohort": "#6A9FD8",
    "ema": GREY, "donchian": "#A0A6B0", "buyhold": RED}

bt = json.load(open(f"{ROOT}/results/p4_backtest_latest/backtest_main.json"))
eq = pd.read_parquet(f"{ROOT}/results/p4_backtest_latest/equity_curves.parquet")
tr = pd.read_parquet(f"{ROOT}/results/p4_backtest_latest/trades_all.parquet")
OOS = [2023, 2024, 2025]

def S(name, cost="cost_1.0", win="oos"):
    return bt[name][cost][win]

def savefig(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=150, facecolor="white",
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("saved", name)

# ---------- F1: OOS equity curves (cumulative R) ----------
fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
show = AURUM + ["gbdt", "buyhold"]
for s in show:
    d = eq[eq.strategy == s].sort_values("ts_trade")
    ts = pd.to_datetime(d.ts_trade.values)
    d_oos = d[d.ts_trade.map(lambda t: pd.Timestamp(t).year in OOS).values]
    if len(d_oos) == 0:
        continue
    tss = pd.to_datetime(d_oos.ts_trade.values)
    lw = 2.6 if s == "aurum_r3" else 1.5
    ax.plot(tss, d_oos.cum_R.values, label=PRETTY[s], color=AUC[s], lw=lw)
ax.axhline(0, color=INK, lw=0.8, alpha=0.6)
ax.set_title("OOS 2023-2025 cumulative R (cost 1.0x, walk-forward)", fontsize=12)
ax.set_ylabel("Cumulative R")
ax.legend(fontsize=9, ncol=2, frameon=False)
ax.grid(alpha=0.25)
savefig(fig, "f1_equity_oos.png")

# ---------- F2: OOS expectancy with bootstrap 95% CI ----------
boot = bt["bootstrap_oos_cost_1.0"]
names = [s for s in AURUM + ENS + BASE if S(s).get("n_trades", 0) > 0]
names.sort(key=lambda s: S(s).get("expectancy_R", -9))
fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
for i, s in enumerate(names):
    b = boot.get(s, {})
    e = S(s).get("expectancy_R", 0)
    lo = b.get("expectancy_R", {}).get("lo", e)
    hi = b.get("expectancy_R", {}).get("hi", e)
    ax.errorbar([e], [i], xerr=[[max(e - lo, 0)], [max(hi - e, 0)]],
                fmt="o", color=AUC[s], capsize=3, lw=1.6, ms=5)
    n = S(s).get("n_trades", 0)
    ax.annotate(f"n={n}", (hi, i), fontsize=7.5, va="center", color="#666", xytext=(4, 0),
                textcoords="offset points")
ax.set_yticks(range(len(names)))
ax.set_yticklabels([PRETTY[s] for s in names], fontsize=9.5)
ax.axvline(0, color=INK, lw=0.9)
ax.set_xlabel("Expectancy per trade (net R) with 95% block-bootstrap CI")
ax.set_title("OOS 2023-2025 expectancy by strategy (cost 1.0x)", fontsize=12)
ax.grid(axis="x", alpha=0.25)
savefig(fig, "f2_expectancy_ci.png")

# ---------- F3: cost sensitivity ----------
fig, ax = plt.subplots(figsize=(9, 5.0), constrained_layout=True)
xs = [1.0, 1.5, 2.0]
for s in [a for a in AURUM] + ["ens_raw", "gbdt", "buyhold"]:
    ys = [bt[s][f"cost_{m}"]["oos"].get("expectancy_R", np.nan) for m in xs]
    ax.plot(xs, ys, "o-", color=AUC[s], lw=2 if s == "aurum_r3" else 1.4,
            ms=5, label=PRETTY[s])
ax.axhline(0, color=INK, lw=0.8, alpha=0.6)
ax.set_xticks(xs); ax.set_xticklabels(["1.0x", "1.5x", "2.0x"])
ax.set_ylabel("OOS expectancy per trade (R)")
ax.set_title("Cost sensitivity - OOS 2023-2025", fontsize=12)
ax.legend(fontsize=9, frameon=False)
ax.grid(alpha=0.25)
savefig(fig, "f3_cost_sensitivity.png")

# ---------- F4: depth sweep ----------
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
rs = [1, 2, 3, 4]
e_full = [bt[f"aurum_r{r}"]["cost_1.0"]["full"].get("expectancy_R", np.nan) for r in rs]
e_oos = [bt[f"aurum_r{r}"]["cost_1.0"]["oos"].get("expectancy_R", np.nan) for r in rs]
sh_oos = [bt[f"aurum_r{r}"]["cost_1.0"]["oos"].get("sharpe_per_trade", np.nan) for r in rs]
n_oos = [bt[f"aurum_r{r}"]["cost_1.0"]["oos"].get("n_trades", 0) for r in rs]
axes[0].plot(rs, e_full, "o-", color=BLUE, label="2016-2025 full")
axes[0].plot(rs, e_oos, "s-", color=GOLD, label="OOS 2023-25")
axes[0].set_xlabel("loop depth r"); axes[0].set_ylabel("Expectancy (R)")
axes[0].set_title("Expectancy vs loop depth", fontsize=11)
axes[0].legend(fontsize=9, frameon=False); axes[0].grid(alpha=0.25)
ax2 = axes[1]
ax2.plot(rs, sh_oos, "o-", color=GREEN)
ax2.set_xlabel("loop depth r"); ax2.set_ylabel("Per-trade Sharpe")
ax2.set_title("OOS per-trade Sharpe vs depth", fontsize=11)
ax2.grid(alpha=0.25)
for r, n in zip(rs, n_oos):
    ax2.annotate(f"n={n}", (r, sh_oos[rs.index(r)]), fontsize=7.5,
                 xytext=(4, -10), textcoords="offset points", color="#666")
savefig(fig, "f4_depth_sweep.png")

# ---------- F5: reliability diagram (AURUM r=3 OOS) ----------
d = tr[(tr.strategy == "aurum_r3") & (tr.year.isin(OOS))]
p = d.p.values.astype(float); win = (d.R_label.values > 0).astype(float)
m = np.isfinite(p) & np.isfinite(win)
p, win = p[m], win[m]
order = np.argsort(p)
bins = np.array_split(order, 10)
xs, ys, ns = [], [], []
for b in bins:
    if len(b) == 0:
        continue
    xs.append(p[b].mean()); ys.append(win[b].mean()); ns.append(len(b))
fig, ax = plt.subplots(figsize=(5.6, 5.0), constrained_layout=True)
ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1, label="perfect calibration")
ax.plot(xs, ys, "o-", color=GOLD, lw=2, ms=6, label="AURUM r=3 (OOS)")
for x, y, n in zip(xs, ys, ns):
    ax.annotate(f"{n}", (x, y), fontsize=7, color="#777", xytext=(4, 4),
                textcoords="offset points")
brier = float(np.mean((p - win) ** 2))
ax.set_xlabel("Predicted p(win) at traded cell"); ax.set_ylabel("Empirical win rate")
ax.set_title(f"Reliability diagram - OOS trades (Brier {brier:.4f})", fontsize=11)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.grid(alpha=0.25); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
savefig(fig, "f5_reliability.png")

# ---------- F6: OOS drawdown ----------
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
for s in AURUM + ["gbdt", "buyhold"]:
    d = eq[(eq.strategy == s)].sort_values("ts_trade")
    m = d.ts_trade.map(lambda t: pd.Timestamp(t).year in OOS).values
    d = d[m]
    if len(d) == 0:
        continue
    cr = d.cum_R.values
    peak = np.maximum.accumulate(np.concatenate([[0.0], cr]))[1:]
    dd = peak - cr
    ax.plot(pd.to_datetime(d.ts_trade.values), dd, color=AUC[s], lw=2 if s == "aurum_r3" else 1.3,
            label=PRETTY[s])
ax.set_ylabel("Drawdown (R)")
ax.set_title("OOS 2023-2025 drawdown (R, cost 1.0x)", fontsize=12)
ax.legend(fontsize=9, ncol=2, frameon=False)
ax.grid(alpha=0.25)
savefig(fig, "f6_drawdown.png")

# ---------- F7: fold walk-forward validation (training-time) ----------
fr = json.load(open(f"{ROOT}/results/p2_train/fold_results.json"))
years = sorted(int(k) for k in fr if k.isdigit() and fr[k] and not fr[k].get("skipped"))
aucs = [fr[str(y)].get("auc_re") for y in years]
briers = [fr[str(y)].get("brier_sm") for y in years]
fig, ax = plt.subplots(figsize=(9, 4.4), constrained_layout=True)
ax.plot(years, aucs, "o-", color=BLUE, lw=2, label="AUC (realized)")
ax.set_ylim(0.5, 0.65)
ax.set_ylabel("AUC"); ax.set_xlabel("walk-forward fold year")
ax2 = ax.twinx()
ax2.plot(years, briers, "s--", color=GOLD, lw=2, label="Brier (smoothed)")
ax2.set_ylabel("Brier")
ax.set_title("Per-fold walk-forward refits: model trained only on states <= y-1", fontsize=12)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="lower left")
ax.grid(alpha=0.25)
savefig(fig, "f7_folds.png")

# ---------- F8: monthly OOS R heatmap (AURUM r=3) ----------
d = tr[(tr.strategy == "aurum_r3") & (tr.year.isin(OOS))].copy()
d["dt"] = pd.to_datetime(d.ts.values)
d["month"] = d.dt.dt.to_period("M").astype(str)
months = sorted(d.month.unique())
rows = []
for mm in months:
    dd = d[d.month == mm]
    rows.append(dd.R_net_1x.sum())
fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
colors = [GREEN if v >= 0 else RED for v in rows]
ax.bar(range(len(months)), rows, color=colors, alpha=0.85)
ax.set_xticks(range(len(months)))
ax.set_xticklabels([m[2:] for m in months], rotation=60, fontsize=8)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("Net R")
ax.set_title("AURUM r=3 monthly net R - OOS 2023-2025 (cost 1.0x)", fontsize=12)
ax.grid(axis="y", alpha=0.25)
savefig(fig, "f8_monthly.png")

print("\nALL FIGURES DONE ->", OUT)
