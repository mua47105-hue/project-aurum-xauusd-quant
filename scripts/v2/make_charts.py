#!/usr/bin/env python3
"""AURUM v2.0 -- publication-grade charts for the Master Report (Phase 5).
Consumes the K1/K2/K3/K6 outputs + v1.3 artifacts; renders 10 charts at
200 DPI into download/aurum_v2_report/figs/.

v2.0.2: strategy panel updated to the dual-hybrid tournament (M7_Core_
Satellite + M7_Active_Grand_Hybrid), M2_RFF_Ridge shown de-saturated, fig8
DSR carries the conservative full-panel vintage, and fig11_z_desaturation
added for the Engineering Flaw #8 root-cause narrative.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/z/my-project"
OUT = os.path.join(ROOT, "download", "aurum_v2_report", "figs")
os.makedirs(OUT, exist_ok=True)

C = {"navy": "#1B2A4A", "gold": "#C9A227", "slate": "#5D6D7E",
     "crimson": "#922B21", "teal": "#148F77", "grey": "#BDC3C7",
     "green": "#1E8449", "purple": "#6C3483"}

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8.5,
    "axes.titlesize": 10, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.facecolor": "white",
})

K1 = os.environ.get("AURUM_CH_K1", os.path.join(ROOT, "kaggle_work/v2/k1_out4"))
K2 = os.environ.get("AURUM_CH_K2", os.path.join(ROOT, "kaggle_work/v2/k2_out3"))
K2_PRE = os.environ.get("AURUM_CH_K2_PRE", os.path.join(ROOT, "kaggle_work/v2/k2_out2"))
K3 = os.environ.get("AURUM_CH_K3", os.path.join(ROOT, "kaggle_work/v2/k3_out2"))
K6 = os.environ.get("AURUM_CH_K6", os.path.join(ROOT, "kaggle_work/v2/k6_out1"))


def jload(p, d=None):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return d if d is not None else {}


def fig3_ic(fam_csv, ic_parquet, top_n=22):
    fam = pd.read_csv(fam_csv).sort_values("best_abs_t")
    ic = pd.read_parquet(ic_parquet)
    top = ic.reindex(ic["t_stat"].abs().sort_values(ascending=False).index).head(top_n)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6),
                                 gridspec_kw={"width_ratios": [1, 1.5]})
    a1.barh(fam["family"], fam["best_abs_t"],
            color=[C["gold"] if str(f).startswith(("F2", "F1")) else C["slate"]
                   for f in fam["family"]])
    a1.set_title("Best |t| Rank-IC by feature family\n(2016-22 walk-forward folds)")
    a1.set_xlabel("best |t-stat| in family")
    colors = [C["crimson"] if t < 0 else C["teal"] for t in top["t_stat"]]
    a2.barh(range(len(top)), top["t_stat"], color=colors)
    a2.set_yticks(range(len(top)))
    a2.set_yticklabels(top.index, fontsize=6.2)
    a2.invert_yaxis()
    a2.axvline(0, color="k", lw=0.7)
    a2.set_title(f"Top {top_n} features by |t| (negative = reversal signal)")
    a2.set_xlabel("mean Rank-IC t-stat across folds")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_rank_ic.png"))
    plt.close(fig)


def fig1_equity(net_series, strategies, title_suffix="OOS 2023-25"):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for name, color, ls in strategies:
        col = f"{name}__oos"
        if col not in net_series:
            continue
        s = net_series[col].dropna()
        ax.plot(s.index, s.cumsum(), label=name, color=color, lw=1.2, ls=ls)
    ax.set_title(f"Cumulative net P&L, vol-target units — {title_suffix}")
    ax.set_ylabel("cumulative net return (vol units)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_oos_equity.png"))
    plt.close(fig)


def fig2_year_sharpe(year_df, strategies):
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    years = sorted(year_df["year"].unique())
    w = 0.8 / max(1, len(strategies))
    for i, (name, color) in enumerate(strategies):
        sub = year_df[year_df["strategy"] == name].set_index("year")
        xs = [y + (i - len(strategies) / 2 + 0.5) * w for y in years]
        ax.bar(xs, [sub.loc[y, "sharpe_ann"] if y in sub.index else np.nan
                    for y in years], width=w, label=name, color=color, alpha=0.92)
    ax.axvline(2022.5, color=C["crimson"], lw=1.0, ls="--")
    ax.text(2022.56, ax.get_ylim()[1] * 0.92, "OOS", color=C["crimson"], fontsize=8)
    ax.set_xticks(years)
    ax.set_title("Year-by-year annualized net Sharpe (FIT 2016-22 | OOS 2023-25)")
    ax.set_ylabel("ann. Sharpe")
    ax.legend(frameon=False, ncol=len(strategies), loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_year_sharpe.png"), bbox_inches="tight")
    plt.close(fig)


def fig4_loop_depth(nn_fc, feats_r, feats_sigma):
    """OOS Sharpe of the M5 contractive-loop iterates z_iter1..4 under the
    v2.0.1 CAUSAL accounting: position decided at close of bar t earns bar
    t+1 (net = w[t-1] * r[t])."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    it_cols = [c for c in ["z_iter1", "z_iter2", "z_iter3", "z_iter4"]
               if c in nn_fc.columns]
    shs = []
    for c in it_cols:
        z = nn_fc[c].dropna()
        sel = z.index[(z.index.year >= 2023)]
        if len(sel) < 500:
            sel = z.index[(z.index.year >= 2021)]
        if len(sel) < 500:
            sel = z.index[(z.index.year >= 2019)]
        v = (0.15 / math.sqrt(5786)) / (feats_sigma.reindex(sel) + 1e-12)
        w = z[sel] * v                       # exposure decided at close of t
        w_lag = w.shift(1).fillna(0.0)       # held during bar t
        net = w_lag * feats_r.reindex(sel)
        net = net.dropna()
        shs.append(net.mean() / (net.std() + 1e-12) * math.sqrt(5786)
                   if len(net) > 100 else np.nan)
    r = np.arange(1, len(shs) + 1)
    ax.plot(r, shs, "o-", color=C["navy"], lw=1.6,
            label="M5 contractive loop (v2.0.1 causal accounting)")
    v13 = np.array([0.2145, 0.1566, 0.1644, 0.1572])
    v13n = v13 / v13[0] * (shs[0] if shs and np.isfinite(shs[0]) else 1)
    ax.plot(r, v13n, "s--", color=C["crimson"], lw=1.3,
            label="v1.3 unconstrained loop (E[R]/trade, rescaled)")
    ax.set_xticks(r)
    ax.set_xlabel("loop iterations r")
    ax.set_ylabel("OOS ann. Sharpe (v2) / E[R] rel. (v1.3)")
    ax.set_title("Contractive looping (rho<0.95, KM-damped) vs v1.3 unconstrained")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_loop_depth.png"))
    plt.close(fig)


def fig5_cost_survival(bt):
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    strategies = [("M7_Core_Satellite", C["gold"], "o"),
                  ("M7_Active_Grand_Hybrid", C["crimson"], "P"),
                  ("M0_Champ_v13", C["green"], "X"),
                  ("M6_AURUM_v2_Ensemble", C["navy"], "s"),
                  ("M5_ContinuumNet", C["teal"], "^"),
                  ("M2_RFF_Ridge", C["purple"], "D"),
                  ("M1_GBDT_v2", C["slate"], "v")]
    for name, color, mk in strategies:
        if name not in bt.get("strategies", {}):
            continue
        oos = bt["strategies"][name].get("oos", {})
        xs, ys = [1.0], [oos.get("metrics", {}).get("sharpe_ann", np.nan)]
        for cm in (1.5, 2.0):
            xs.append(cm)
            ys.append(oos.get(f"cost{cm}", {}).get("sharpe_ann", np.nan))
        ax.plot(xs, ys, marker=mk, color=color, lw=1.4, label=name, ms=4.5)
    ax.set_xticks([1.0, 1.5, 2.0])
    ax.set_xlabel("transaction-cost multiplier")
    ax.set_ylabel("OOS ann. net Sharpe")
    ax.set_title("Cost survival: OOS Sharpe at 1.0x / 1.5x / 2.0x costs")
    ax.axhline(0, color="k", lw=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_cost_survival.png"))
    plt.close(fig)


def fig6_buffer(engine_metrics):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.3))
    names, to_full, to_nb, sh_full, sh_nb = [], [], [], [], []
    for key, d in engine_metrics.items():
        if not key.endswith("__fit"):
            continue
        if "full" in d and "no_buffer" in d:
            names.append(key[:-5].replace("_", "\n"))
            to_full.append(d["full"]["turnover_py"])
            to_nb.append(d["no_buffer"]["turnover_py"])
            sh_full.append(d["full"]["metrics"].get("sharpe_ann", np.nan))
            sh_nb.append(d["no_buffer"]["metrics"].get("sharpe_ann", np.nan))
    if not names:
        plt.close(fig)
        return
    x = np.arange(len(names))
    a1.bar(x - 0.18, to_nb, 0.36, color=C["crimson"], label="no buffer")
    a1.bar(x + 0.18, to_full, 0.36, color=C["teal"], label="Carver ±10% buffer")
    a1.set_xticks(x); a1.set_xticklabels(names, fontsize=6.5)
    a1.set_ylabel("round turns / year"); a1.set_title("Turnover (FIT 2016-22)")
    a1.legend(frameon=False)
    a2.bar(x - 0.18, sh_nb, 0.36, color=C["crimson"], label="no buffer")
    a2.bar(x + 0.18, sh_full, 0.36, color=C["teal"], label="buffer")
    a2.set_xticks(x); a2.set_xticklabels(names, fontsize=6.5)
    a2.set_ylabel("FIT ann. Sharpe"); a2.set_title("Net Sharpe (FIT 2016-22)")
    a2.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_buffer.png"))
    plt.close(fig)


def fig7_spectral(rho_log):
    if not rho_log:
        return
    df = pd.DataFrame(rho_log)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for seed, sub in df.groupby("seed"):
        ax.plot(sub["fold"], sub["rho"], "o-", label=f"seed {seed}",
                color=C["navy"] if seed == 1337 else C["gold"], lw=1.3, ms=4)
    ax.axhline(0.95, color=C["crimson"], ls="--", lw=1.2)
    ax.text(df["fold"].min(), 0.955, "acceptance gate rho < 0.95",
            color=C["crimson"], fontsize=8)
    ax.set_xlabel("walk-forward fold (test year)")
    ax.set_ylabel("audited spectral radius rho")
    ax.set_title("M5 LoopBlock spectral-radius governance (JVP power iteration)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_spectral.png"))
    plt.close(fig)


def fig8_dsr(stats):
    rows = [(k, v.get("oos_dsr"), v.get("oos_dsr_conservative"))
            for k, v in stats.items()
            if isinstance(v, dict) and "oos_dsr" in v]
    if not rows:
        return
    rows.sort(key=lambda t: -(t[1] or 0))
    names = [r[0] for r in rows]
    vals = [r[1] or 0 for r in rows]
    cons = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    colors = [C["green"] if v >= 0.95 else (C["gold"] if v >= 0.5 else C["crimson"])
              for v in vals]
    ax.barh(range(len(rows)), vals, color=colors, height=0.62,
            label="primary family (N=12: signals + 1 blend slot)")
    xs = [i for i, c in enumerate(cons) if isinstance(c, (int, float))]
    if xs:
        ax.scatter([cons[i] for i in xs], xs, marker="D", s=16, zorder=3,
                   color=C["navy"], label="conservative full panel (N=14)")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=6.8)
    ax.invert_yaxis()
    ax.axvline(0.95, color=C["green"], ls="--", lw=1.1)
    ax.text(0.955, len(rows) - 0.5, "DSR >= 0.95 gate", color=C["green"], fontsize=7.5)
    meta = stats.get("_meta", {})
    pbo = meta.get("pbo_cscv")
    ax.set_title(f"Deflated Sharpe Ratio, OOS 2023-25 "
                 f"(N={meta.get('N_trials_ledger', '?')} ledger configs"
                 + (f", CSCV-PBO={pbo}" if pbo is not None else "") + ")")
    ax.set_xlabel("DSR probability")
    ax.set_xlim(0, 1.05)
    ax.legend(frameon=False, loc="lower right", fontsize=6.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_dsr.png"))
    plt.close(fig)


def fig9_drawdown(net_series, strategies):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for name, color in strategies:
        col = f"{name}__oos"
        if col not in net_series:
            continue
        s = net_series[col].dropna()
        eq = s.cumsum()
        dd = eq - eq.cummax()
        ax.plot(s.index, dd, label=name, color=color, lw=1.1)
    ax.set_title("OOS drawdown paths (vol-target units)")
    ax.set_ylabel("drawdown (vol units)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_drawdown.png"))
    plt.close(fig)


def fig10_champion(champion_compare, v13_sharpe=2.444):
    rows = [("v1.3 Flagship cap=3\n(trade-based)", v13_sharpe)]
    for k, lbl in (("M0_Champ_v13", "v1.3 champion, continuous\ntranslation"),
                   ("M7_Core_Satellite", "M7 Core-Satellite\n(champion + passive gold)"),
                   ("M7_Active_Grand_Hybrid", "M7 Active Grand Hybrid\n(champion + active v2 stack)"),
                   ("M6_AURUM_v2_Ensemble", "M6 AURUM v2 Ensemble\n(50/50 M1+M2, v2.0.2)"),
                   ("M2_RFF_Ridge", "M2 RFF-Ridge\n(de-saturated, v2.0.2)"),
                   ("M5_ContinuumNet", "M5 ContinuumNet"),
                   ("M1_GBDT_v2", "M1 GBDT v2")):
        v = champion_compare.get(k)
        if isinstance(v, dict) and isinstance(v.get("sharpe_ann"), (int, float)):
            rows.append((lbl, float(v["sharpe_ann"])))
    names = [r[0] for r in rows]
    sh = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = [C["crimson"]] + [C["navy"], C["gold"], C["teal"], C["slate"],
                               C["purple"], C["grey"]][:max(0, len(rows) - 1)]
    ax.barh(range(len(rows)), sh, color=colors[:len(rows)])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    for i, v in enumerate(sh):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=7.5)
    ax.set_xlabel("OOS 2023-25 ann. net Sharpe (continuous engine accounting)")
    ax.set_title("The AURUM lineage: v1.3 champion vs the v2.0 tournament")
    ax.set_xlim(0, max(sh) * 1.18)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig10_lineage.png"))
    plt.close(fig)


def fig11_z_desaturation(fc_pre, fc_new):
    """Engineering Flaw #8 root-cause chart: fold-level pred_rff drift vs
    within-fold dispersion (why fold-local scaling saturated), the v2.0.1
    saturated z_rff (per-fold sign bets), and the v2.0.2 causal trailing
    standardization (genuinely active)."""
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(7.6, 6.4),
                                     sharex=True,
                                     gridspec_kw={"height_ratios": [1, 1, 1.2]})
    pre = fc_pre["pred_rff"].dropna()
    yr = pre.index.year
    stats = pd.DataFrame({"mean": pre.groupby(yr).mean(),
                          "std": pre.groupby(yr).std()})
    a1.bar(stats.index - 0.18, stats["mean"].abs(), 0.36,
           color=C["crimson"], label="|fold mean drift| (mu)")
    a1.bar(stats.index + 0.18, stats["std"], 0.36,
           color=C["slate"], label="within-fold std (sigma)")
    a1.set_yscale("log")
    a1.set_ylabel("pred_rff scale (log)")
    a1.set_title("Raw RFF predictions: fold drift mu >> dispersion sigma "
                 "(up to 100x) -- fold-local scaling collapses")
    a1.legend(frameon=False, fontsize=6.8)
    z_old = fc_pre["z_rff"].dropna()
    oos_old = z_old[z_old.index.year >= 2023]
    a2.plot(z_old.index, z_old.values, lw=0.4, color=C["crimson"])
    a2.set_ylim(-1.15, 1.15)
    a2.set_ylabel("z_rff (v2.0.1)")
    a2.set_title(f"v2.0.1 fold-local z: per-fold sign bets "
                 f"(OOS nunique=1, {oos_old.mode().iloc[0]:+.1f} for "
                 f"{len(oos_old)} bars) -- passive gold in disguise")
    z_new = fc_new["z_rff"].dropna()
    a3.plot(z_new.index, z_new.values, lw=0.35, color=C["teal"])
    a3.set_ylim(-1.15, 1.15)
    a3.set_ylabel("z_rff (v2.0.2)")
    a3.set_title("v2.0.2 causal trailing standardization: genuinely active, "
                 "G4.8-gated (std 0.42, 0% saturation, 558 round-turns/yr)")
    for a in (a1, a2, a3):
        a.axvline(pd.Timestamp("2023-01-01", tz="UTC"), color=C["grey"],
                  ls=":", lw=0.9)
    a3.text(pd.Timestamp("2023-02-15", tz="UTC"), -1.05, "OOS seal",
            color=C["grey"], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig11_z_desaturation.png"))
    plt.close(fig)


def main():
    fam_csv = os.path.join(K1, "ic_family_summary.csv")
    ic_parquet = os.path.join(K1, "ic_audit.parquet")
    if os.path.isfile(fam_csv) and os.path.isfile(ic_parquet):
        fig3_ic(fam_csv, ic_parquet); print("fig3 done")

    em = os.path.join(K2, "engine_metrics.json")
    if os.path.isfile(em):
        fig6_buffer(jload(em)); print("fig6 done")

    nn_path = os.path.join(K3, "v2_neural_forecasts.parquet")
    if os.path.isfile(nn_path):
        nn_fc = pd.read_parquet(nn_path)
        nn_fc.index = pd.DatetimeIndex(nn_fc.index)
        if nn_fc.index.tz is None:
            nn_fc.index = nn_fc.index.tz_localize("UTC")
        # r_xau / sigma travel with the forecasts (K3 parquet carries them)
        feats_r = nn_fc["r_xau"]
        feats_sigma = nn_fc["sigma"]
        fig4_loop_depth(nn_fc, feats_r, feats_sigma); print("fig4 done")
    rho = jload(os.path.join(K3, "rho_log.json"), [])
    if rho:
        fig7_spectral(rho); print("fig7 done")

    bt_path = os.path.join(K6, "backtest_v2.json")
    ns_path = os.path.join(K6, "net_series.parquet")
    yr_path = os.path.join(K6, "year_table.csv")
    st_path = os.path.join(K6, "stats.json")
    cc_path = os.path.join(K6, "champion_compare.json")
    if os.path.isfile(ns_path):
        ns = pd.read_parquet(ns_path)
        fig1_equity(ns, [("M7_Core_Satellite", C["gold"], "-"),
                         ("M7_Active_Grand_Hybrid", C["crimson"], "-"),
                         ("M0_Champ_v13", C["green"], "-"),
                         ("M6_AURUM_v2_Ensemble", C["navy"], "--"),
                         ("M2_RFF_Ridge", C["purple"], "--"),
                         ("M1_GBDT_v2", C["slate"], "--"),
                         ("BuyHold_Gold", C["grey"], ":")])
        fig9_drawdown(ns, [("M7_Core_Satellite", C["gold"]),
                           ("M7_Active_Grand_Hybrid", C["crimson"]),
                           ("M0_Champ_v13", C["green"]),
                           ("M6_AURUM_v2_Ensemble", C["navy"]),
                           ("BuyHold_Gold", C["grey"])])
        print("fig1/fig9 done")
    if os.path.isfile(yr_path):
        ydf = pd.read_csv(yr_path)
        picks = [s for s in
                 [("M7_Core_Satellite", C["gold"]),
                  ("M7_Active_Grand_Hybrid", C["crimson"]),
                  ("M6_AURUM_v2_Ensemble", C["navy"]),
                  ("M2_RFF_Ridge", C["purple"]),
                  ("M1_GBDT_v2", C["slate"]),
                  ("BuyHold_Gold", C["grey"])] if s[0] in set(ydf["strategy"])]
        fig2_year_sharpe(ydf, picks); print("fig2 done")
    if os.path.isfile(bt_path):
        fig5_cost_survival(jload(bt_path)); print("fig5 done")
    if os.path.isfile(st_path):
        fig8_dsr(jload(st_path)); print("fig8 done")
    if os.path.isfile(cc_path):
        fig10_champion(jload(cc_path)); print("fig10 done")
    pre_path = os.path.join(K2_PRE, "v2_cpu_forecasts.parquet")
    new_path = os.path.join(K2, "v2_cpu_forecasts.parquet")
    if os.path.isfile(pre_path) and os.path.isfile(new_path):
        fc_pre = pd.read_parquet(pre_path)
        fc_new = pd.read_parquet(new_path)
        for fc in (fc_pre, fc_new):
            fc.index = pd.DatetimeIndex(fc.index)
            if fc.index.tz is None:
                fc.index = fc.index.tz_localize("UTC")
        fig11_z_desaturation(fc_pre, fc_new); print("fig11 done")
    print("charts written to", OUT)


if __name__ == "__main__":
    main()
