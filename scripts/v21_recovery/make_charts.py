#!/usr/bin/env python3
"""Step 7: publication-grade 200-DPI charts from the verified v2.1 artifacts
-> figures/v2_1_figures/. English labels; constrained_layout; no tight_layout."""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for _f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
    if os.path.exists(_f):
        fm.fontManager.addfont(_f)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures", "v2_1_figures")
os.makedirs(OUT, exist_ok=True)
C_MAIN, C_ALT, C_BAD, C_GOOD = "#1a476f", "#2d6a4f", "#9b2226", "#4361a8"

v21 = json.load(open(os.path.join(REPO, "artifacts/v2_1/p4_out/v21_summary.json")))
bm = json.load(open(os.path.join(REPO, "artifacts/v2_1/p4_out/backtest_main.json")))
attr = json.load(open(os.path.join(REPO, "artifacts/v2_1/k6_out/attribution_v21.json")))
u2 = json.load(open(os.path.join(REPO, "artifacts/v2_1/k2_fixed/u2_fix_report.json")))


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- 1. equity
eq = pd.read_parquet(os.path.join(REPO, "artifacts/v2_1/p4_out/equity_curves.parquet"))
fl = eq[(eq["strategy"] == "flagship") & (eq["cap"] == 3) & (eq["cost_mult"] == 1.0)].copy()
fl["date"] = pd.to_datetime(fl["day_ns"])
oos = fl[fl["date"] >= "2023-01-01"].reset_index(drop=True)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                             constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.4, 1]})
a1.plot(oos["date"], oos["equity"], color=C_MAIN, lw=1.4)
a1.set_title("AURUM v1.3 Flagship (cap=3, frozen policy) — OOS 2023-25 equity, "
             "daily MTM, 15% drawdown throttle ON")
a1.set_ylabel("equity (x)")
peak = np.maximum.accumulate(oos["equity"].values)
dd = (peak - oos["equity"].values) / peak
a2.fill_between(oos["date"], -dd * 100, 0, color=C_BAD, alpha=0.55, lw=0)
a2.axhline(-13.06, color="k", ls="--", lw=0.8)
a2.annotate("throttled MaxDD 13.06%", (oos["date"].iloc[40], -12.2), fontsize=8)
a2.axhline(-23.92, color="gray", ls=":", lw=0.9)
a2.annotate("un-throttled 23.92%", (oos["date"].iloc[40], -23.3), fontsize=8)
a2.set_ylabel("drawdown (%)")
a2.set_xlabel("OOS 2023-25 (Sharpe 2.4443, CAGR 54.56%, n=3,341 trades)")
save(fig, "fig1_flagship_oos_equity.png")

# ------------------------------------------------- 2. tier x carry matrix
tm = v21["tier_matrix_flagship_cap3_oos"]
labels = ["throttle15", "unthrottled"]
tiers = ["standard", "institutional_pilot"]
carries = ["no_carry", "carry_base", "carry_punitive"]
x = np.arange(len(tiers) * len(labels))
w = 0.26
fig, ax = plt.subplots(figsize=(9.5, 5), constrained_layout=True)
for i, c in enumerate(carries):
    vals = [tm[f"{t}|{th}|{c}"]["sharpe"] for t in tiers for th in labels]
    dd = [tm[f"{t}|{th}|{c}"]["max_dd_pct"] for t in tiers for th in labels]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=c.replace("_", " "))
    for b, d in zip(bars, dd):
        ax.annotate(f"{d:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7, color="#444")
ax.set_xticks(x)
ax.set_xticklabels(["standard\nthrottle15", "standard\nunthrottled",
                    "pilot 0.5%\nthrottle15", "pilot 0.5%\nunthrottled"],
                   fontsize=9)
ax.axhline(2.4443, color="k", ls="--", lw=0.8)
ax.annotate("shipped reference 2.4443", (0.02, 2.47), fontsize=8)
ax.set_ylabel("OOS Sharpe (cap=3)")
ax.set_title("U5 risk-tier x throttle x overnight-carry matrix — OOS 2023-25\n"
             "(bar labels = MaxDD%; audit carry model: -5.0/+2.5% base, "
             "-5.5/+1.5% punitive, 22:00 UTC, Wed x3)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
save(fig, "fig2_tier_carry_matrix.png")

# ------------------------------------------------- 3. M1 friction matrix
fr = v21["m1_friction_matrix_flagship_cap3_oos"]
names = ["H1 walk\n(shipped)", "M1 pure\npath", "M1 +1min\ndelay",
         "M1 +$0.25/oz", "M1 +$0.35/oz", "M1 2x stop\nslippage"]
keys = ["m1_pure", "m1_delay1", "m1_spread25", "m1_spread35", "m1_slip2x"]
h1_er = bm["flagship"]["cap3"]["cost_1.0"]["oos"]["expectancy_R"]
vals = [h1_er] + [fr[k]["E_R"] for k in keys]
cols = [C_MAIN] + [C_GOOD if v >= 0.10 else C_BAD for v in vals[1:]]
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
bars = ax.bar(names, vals, color=cols, width=0.62)
for b, v in zip(bars, vals):
    ax.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("OOS E[R] per trade (cap=3, net)")
ax.set_title("U6: 1-minute path resolution + friction survival matrix — "
             "flagship OOS 2023-25 (n=3,341)\nentry-minute-inactive, stop-first "
             "minute ties, H1 bar-count window; 5.6M M1 bars")
save(fig, "fig3_m1_friction.png")

# ------------------------------------------------- 4. bridge decomposition
br = attr["bridge_decomposition_E0_E1_E2"]
fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
bs = ax.bar(["E0\nshipped artifact\n(z[i0:i1])",
             "E1\nfaithful\n(entry-bar credit removed)",
             "E2\nfaithful + partial\n(i_part halving)"],
            [br["E0"], br["E1"], br["E2"]], color=[C_MAIN, C_ALT, C_GOOD],
            width=0.55)
for b, v in zip(bs, [br["E0"], br["E1"], br["E2"]]):
    ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom", fontsize=10)
ax.axhline(2.4443, color="k", ls="--", lw=1)
ax.annotate("native trade accounting 2.4443", (0.02, 2.47), fontsize=8)
ax.set_ylabel("continuous-frame OOS Sharpe")
ax.set_ylim(2.2, 2.75)
ax.set_title("U1/D4: faithful-bridge decomposition (E0->E1 = the Flaw #10 "
             "entry-bar credit;\n17.2% 1-bar trades have no hourly "
             "representation — dropped, P&L stays native)")
save(fig, "fig4_bridge_decomposition.png")

# ------------------------------------------------- 5. attribution panel
capm = attr["capm_nw_hac5_vs_buyhold_gold"]
reg = attr["regime_torture_5"]["M0_Champ_v13_E2"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
frames = ["M0_Champ_v13", "M0_Champ_v13_E1", "M0_Champ_v13_E2",
          "M0_Champ_v13_E2_carry"]
fnames = ["E0 artifact", "E1 faithful", "E2 faithful", "E2 + carry"]
oos_a = [capm[f]["oos"]["alpha_pct_yr"] for f in frames]
oos_t = [capm[f]["oos"]["alpha_t"] for f in frames]
bars = a1.bar(fnames, oos_a,
              color=[C_MAIN, C_ALT, C_GOOD, C_BAD], width=0.6)
for b, v, t in zip(bars, oos_a, oos_t):
    a1.annotate(f"{v:+.1f}\n(t={t:.1f})", (b.get_x() + b.get_width() / 2, v),
                ha="center", va="bottom", fontsize=8)
a1.axhline(0, color="k", lw=0.8)
a1.set_ylabel("CAPM alpha %/yr (NW HAC(5), OOS)")
a1.set_title("Alpha vs Buy&Hold gold (daily, Newey-West HAC(5))")
regs = ["2016-19\nrange", "2020\nCOVID", "2021-22\nbear", "2023\ndegraded",
        "2024-25\nparabolic"]
rb = a2.bar(regs, [reg[r] for r in ["2016-19 range", "2020 COVID",
                                    "2021-22 bear", "2023 degraded/chop",
                                    "2024-25 parabolic"]],
            color=C_MAIN, width=0.6)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylabel("annualized Sharpe (daily net)")
a2.set_title("5-regime torture — champion faithful E2 frame (full window)")
save(fig, "fig5_attribution_panel.png")

# ------------------------------------------------- 6. DSR panel
dsr = attr["dsr_panel_project_wide_N"]
Ns = [12, 50, 100, 170, 200, 500]
fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
ax.plot(Ns, [dsr["M0_Champ_v13_E2"][f"N={n}"] for n in Ns], "o-",
        color=C_GOOD, label="champion E2 faithful")
ax.plot(Ns, [dsr["M0_Champ_v13_E2_carry"][f"N={n}"] for n in Ns], "s--",
        color=C_BAD, label="E2 + carry (base)")
ax.axhline(0.95, color="k", ls="--", lw=1)
ax.annotate("pre-registered 0.95 admission bar — MISSED (honest finding)",
            (12.5, 0.955), fontsize=8)
ax.axvline(170, color="gray", ls=":", lw=0.9)
ax.annotate("N=170 (full repo ledger)", (176, 0.62), fontsize=8, rotation=90)
ax.set_xscale("log")
ax.set_xticks(Ns)
ax.set_xticklabels([str(n) for n in Ns])
ax.set_xlabel("N trials (project-wide multiple-testing count)")
ax.set_ylabel("Deflated Sharpe Ratio (Bailey-LdP)")
ax.set_title("Honest project-wide DSR panel — the champion fails its own "
             "0.95 bar at every N >= 12")
ax.legend(loc="upper right", fontsize=9)
save(fig, "fig6_dsr_panel.png")

# ------------------------------------------------- 7. U2 collapse
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
b = a1.bar(["fold-local\n(v2.0.2 artifact)", "causal trailing\n(U2 fixed)"],
           [u2["before"]["oos_sharpe_foldlocal"], u2["after"]["oos_sharpe_fixed"]],
           color=[C_BAD, C_GOOD], width=0.5)
for bb, v in zip(b, [u2["before"]["oos_sharpe_foldlocal"],
                     u2["after"]["oos_sharpe_fixed"]]):
    a1.annotate(f"{v:.3f}", (bb.get_x() + bb.get_width() / 2, v),
                ha="center", va="bottom", fontsize=10)
a1.set_ylabel("M1_GBDT_v2 OOS Sharpe (engine)")
a1.set_title("Flaw #9: the z_gbdt saturation artifact collapses\n"
             "under the program's own Flaw-#8 fix (no retraining)")
yr = u2["after"]["oos_sharpe_fixed_by_year"]
b2 = a2.bar([str(k) for k in yr], list(yr.values()), color=C_MAIN, width=0.55)
for bb, v in zip(b2, yr.values()):
    a2.annotate(f"{v:+.2f}", (bb.get_x() + bb.get_width() / 2, v),
                ha="center", va="bottom", fontsize=9)
a2.axhline(0, color="k", lw=0.8)
a2.set_ylabel("OOS Sharpe by year (fixed transform)")
a2.set_title("No consistent residual alpha — the v2 active stack is "
             "quarantined (U3)")
save(fig, "fig7_u2_collapse.png")

# ------------------------------------------------- 8. strategies
strats = ["flagship", "r1_fast", "r1_clean"]
snames = ["AURUM_v13_Flagship\n(pooled 1:3)", "AURUM_r1_Fast_Flagship\n(r=1 only)",
          "AURUM_r1_Clean_EV\n(raw r=1 EV-argmax)"]
dat = [bm[s]["cap3"]["cost_1.0"]["oos"] for s in strats]
fig, ax = plt.subplots(figsize=(8.6, 4.6), constrained_layout=True)
ers = [d["expectancy_R"] for d in dat]
srs = [d["portfolio"]["sharpe_annualized"] for d in dat]
trs = [d["total_R"] for d in dat]
x = np.arange(3)
b1 = ax.bar(x - 0.18, ers, 0.34, color=C_MAIN, label="E[R] per trade")
axb = ax.twinx()
b2 = axb.bar(x + 0.18, trs, 0.34, color=C_ALT, label="total R")
for bb, v in zip(b1, ers):
    ax.annotate(f"{v:.4f}", (bb.get_x() + bb.get_width() / 2, v),
                ha="center", va="bottom", fontsize=9)
for bb, v in zip(b2, trs):
    axb.annotate(f"{v:.0f}", (bb.get_x() + bb.get_width() / 2, v),
                 ha="center", va="bottom", fontsize=9)
for i, s in enumerate(srs):
    ax.annotate(f"Sharpe {s:.4f}", (x[i], -0.028), ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(snames, fontsize=9)
ax.set_ylabel("OOS E[R] per trade")
axb.set_ylabel("OOS total R")
ax.set_ylim(-0.05, 0.30)
ax.set_title("U4: r=1 strategies vs the pooled flagship — OOS 2023-25, cap=3\n"
             "(the r=1-only machinery matches/beats the pooled flagship with "
             "3x cheaper inference)")
ax.legend(loc="upper left", fontsize=8)
axb.legend(loc="upper right", fontsize=8)
save(fig, "fig8_r1_strategies.png")

print("\nall charts done ->", OUT)
