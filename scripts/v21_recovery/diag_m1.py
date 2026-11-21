#!/usr/bin/env python3
"""Diagnose the M1 walker vs H1 walk divergence (m1_pure E[R] 0.1211 vs
audit 0.1712; H1 net 0.1478). Per-trade R comparison by exit reason."""
import os, sys
os.environ.setdefault("AURUM_DS", "data/phase1")
os.environ.setdefault("AURUM_P3", "models")
import numpy as np
import pandas as pd

K = "kaggle_kernels/v1_3/aurum-p4-backtest.py"
sys.path.insert(0, ".")
import importlib.util
spec = importlib.util.spec_from_file_location("p4", K)
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)

# load M1 + H1 price frames directly
m1 = pd.read_parquet("data/phase1/m1_xauusd.parquet")
m1 = m1[~m1.index.duplicated()].sort_index()
px = pd.read_parquet("data/phase1/h1_xauusd.parquet")
px = px[~px.index.duplicated()].sort_index()

# sanity 1: H1 open == first M1 open of the hour?
m1_hour = m1.index.asi8 // 3_600_000_000_000
first_m1 = pd.Series(np.arange(len(m1)), index=m1_hour)
first_m1 = first_m1[~first_m1.index.duplicated()]
h1_hour = px.index.asi8 // 3_600_000_000_000
al = first_m1.reindex(h1_hour).values
ok_al = np.isfinite(al)
h1_open = px["o"].values
m1_first_open = m1["o"].values[al[ok_al].astype(int)]
d = h1_open[ok_al] - m1_first_open
print(f"H1 bars: {len(px)}; hours mapped to M1: {ok_al.sum()}")
print(f"H1 open vs first-M1 open: max|diff|={np.abs(d).max():.6f}  mean={d.mean():.8f}")

# M1 stamp convention: first M1 bar of an hour — minute offset within hour
mins = pd.DatetimeIndex(m1.index).minute.values
first_mask = np.concatenate([[True], m1_hour[1:] != m1_hour[:-1]])
print(f"M1 first-bar minute-of-hour values (sample): {np.unique(mins[first_mask])[:8]}")
print(f"M1 last-bar minute-of-hour (sample): {np.unique(mins[np.concatenate([m1_hour[1:] != m1_hour[:-1], [True]])])[-8:]}")
print(f"M1 bars total: {len(m1)}, span {m1.index[0]} .. {m1.index[-1]}")

# trades: flagship OOS cap3 from the v2.1 p4 run
tr = pd.read_parquet("artifacts/v2_1/p4_out/trades_all.parquet")
sub = tr[(tr["strategy"] == "flagship") & (tr["cap"] == 3) &
         (tr["cost_mult"] == 1.0) & (tr["year"] >= 2023)].copy()
print(f"\nflagship OOS cap3 trades: {len(sub)}")
print("exit reasons via R_gross+overlay: overlay codes:",
      dict(zip(*np.unique(sub["overlay"].values, return_counts=True))))
# R_gross distribution
rg = sub["R_gross"].values
print(f"H1 R_gross: mean={rg.mean():.4f}  |  -1R exact: {(np.abs(rg+1)<1e-9).mean():.3f}"
      f"  |  tight(-q/m): {((rg>-1)&(rg<-0.5)&(sub['overlay']==2)).mean():.3f}"
      f"  |  timeout share approx: {((sub['overlay']==0)&(np.abs(rg - np.round(rg*4)/4)>1e-9)).mean():.3f}")
print(sub[["cell_name", "R_gross", "overlay", "exit_bar", "i_part", "q50", "q25"]]
      .head(8).to_string())
