"""Project AURUM report figures. English labels, Ink Gold palette."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from statistics import NormalDist

GOLD = "#C9A84C"
DARK = "#1A1A1A"
GRAY = "#687078"
LIGHT = "#DDD5C0"
RED = "#A0522D"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

OUT = "/home/z/my-project/scripts/aurum/"

# ------------------------------------------------------------------
# Figure 1: Gold verified price milestones 2001-2026
# ------------------------------------------------------------------
pts = [
    (2001.0, 255), (2011.7, 1912), (2013.9, 1183), (2015.9, 1046),
    (2018.6, 1161), (2020.6, 2089), (2022.1, 2079), (2022.7, 1614),
    (2023.9, 2063), (2024.8, 2802), (2025.8, 4398), (2026.08, 5586),
    (2026.83, 3879), (2026.85, 4053),
]
x = [p[0] for p in pts]
y = [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200, constrained_layout=True)
ax.plot(x, y, "-o", color=GOLD, lw=2.0, ms=4.5, mfc=DARK, mec=GOLD, zorder=3)
ax.axvspan(2026.0, 2026.95, color=RED, alpha=0.10, zorder=1)
ax.annotate("2011 top\n$1,912", xy=(2011.7, 1912), xytext=(2008.6, 2650),
            fontsize=8, color=DARK, ha="center",
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.annotate("2015 low\n$1,046", xy=(2015.9, 1046), xytext=(2015.2, 2100),
            fontsize=8, color=DARK, ha="center",
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.annotate("2020\n$2,089", xy=(2020.6, 2089), xytext=(2019.4, 3350),
            fontsize=8, color=DARK, ha="center",
            arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.annotate("Jan 2026 ATH\n$5,586", xy=(2026.08, 5586), xytext=(2023.9, 5350),
            fontsize=8, color=RED, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
ax.annotate("Nov 6, 2026\n$4,053 (-27%)", xy=(2026.85, 4053), xytext=(2025.35, 1250),
            fontsize=8, color=RED, ha="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=RED, lw=0.8))
ax.set_xlim(2000, 2028)
ax.set_ylim(0, 6200)
ax.set_ylabel("USD / troy ounce", fontsize=9)
ax.set_title("Gold (XAUUSD): verified milestone prices, 2001-2026", fontsize=11,
             color=DARK, fontweight="bold", loc="left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=LIGHT, lw=0.6, alpha=0.6)
ax.tick_params(labelsize=8, colors=DARK)
fig.savefig(OUT + "fig1_gold_milestones.png")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 2: Kaggle free-tier training feasibility
# ------------------------------------------------------------------
scen = [
    ("LoopUS 1.7B, B=20 K=5\n(paper recipe)", 84, 37, 130),
    ("LoopUS 0.6B, B=20 K=5", 287, 75, 500),
    ("LoopUS 0.6B, B=8 K=3\n(scaled recipe)", 318, 160, 475),
    ("Standard QLoRA 1.7B", 400, 300, 500),
    ("Standard QLoRA 0.6B", 900, 600, 1200),
]
labels = [s[0] for s in scen]
mids = np.array([s[1] for s in scen], dtype=float)
los = np.array([s[2] for s in scen], dtype=float)
his = np.array([s[3] for s in scen], dtype=float)

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200, constrained_layout=True)
colors = [GRAY, GRAY, GOLD, GRAY, GRAY]
bars = ax.barh(labels, mids, xerr=[mids - los, his - mids], color=colors,
               edgecolor=DARK, lw=0.6, height=0.62, error_kw=dict(ecolor=DARK, lw=1.0, capsize=3))
ax.axvline(3000, color=RED, lw=1.4, ls="--")
ax.text(3050, 4.45, "LoopUS paper budget\n(3B tokens)", fontsize=8, color=RED, va="center")
ax.axvline(1000, color=DARK, lw=1.2, ls=":")
ax.text(1060, 1.9, "AURUM scaled target\n(~0.5-1B tokens)", fontsize=8, color=DARK, va="center")
ax.set_xlabel("Tokens trainable per GPU-week (2 x T4 DDP, 30 h, estimated)", fontsize=9)
ax.set_title("Free-tier feasibility: what one Kaggle GPU-week buys", fontsize=11,
             color=DARK, fontweight="bold", loc="left")
ax.set_xlim(0, 3600)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color=LIGHT, lw=0.6, alpha=0.6)
ax.tick_params(labelsize=8, colors=DARK)
fig.savefig(OUT + "fig2_kaggle_feasibility.png")
plt.close(fig)

# ------------------------------------------------------------------
# Figure 3: The multiple-testing trap
# ------------------------------------------------------------------
nd = NormalDist()
gamma = 0.5772156649
n_trades = 1500.0     # ~6 years x 250 trades
trades_per_year = 250.0
se_ann = np.sqrt(trades_per_year / n_trades)   # SE of annualized Sharpe

N = np.logspace(0.05, 4, 400)
emax = []
for n in N:
    z1 = nd.inv_cdf(min(1 - 1.0 / n, 1 - 1e-12))
    z2 = nd.inv_cdf(min(1 - 1.0 / (n * np.e), 1 - 1e-12))
    emax.append(((1 - gamma) * z1 + gamma * z2) * se_ann)
emax = np.array(emax)

fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=200, constrained_layout=True)
ax.plot(N, emax, color=GOLD, lw=2.4)
ax.set_xscale("log")
ax.axhline(0.5, color=GRAY, lw=1.0, ls="--")
ax.text(1.35, 0.545, "SR 0.5 - good professional CTA (Carver live: 0.58)", fontsize=8, color=GRAY)
ax.axhline(1.0, color=RED, lw=1.0, ls="--")
ax.text(1.35, 1.045, "SR 1.0 - excellent; rarely verified live", fontsize=8, color=RED)
for n_pos, lab in [(10, "10 configs"), (100, "100 configs"), (1000, "1,000 configs")]:
    idx = int(np.argmin(np.abs(N - n_pos)))
    ax.plot(n_pos, emax[idx], "o", color=DARK, ms=5, zorder=3)
    ax.annotate(f"{lab}\nlucky SR ~ {emax[idx]:.2f}", xy=(n_pos, emax[idx]),
                xytext=(n_pos * 1.55, emax[idx] - 0.16), fontsize=8, color=DARK,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.set_xlabel("Number of strategy configurations tried (log scale)", fontsize=9)
ax.set_ylabel("Expected maximum annualized Sharpe\nfrom pure luck (null hypothesis)", fontsize=9)
ax.set_title("The multiple-testing trap (Bailey & Lopez de Prado)", fontsize=11,
             color=DARK, fontweight="bold", loc="left")
ax.set_ylim(0, 1.75)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(color=LIGHT, lw=0.6, alpha=0.6)
ax.tick_params(labelsize=8, colors=DARK)
fig.savefig(OUT + "fig3_multiple_testing.png")
plt.close(fig)

print("figures done")
