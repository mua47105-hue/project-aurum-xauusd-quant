#!/usr/bin/env python3
"""Per-trade M1-vs-H1 divergence diagnostic: which trades degrade, why."""
import os, sys, importlib.util
os.environ.setdefault("AURUM_DS", "data/phase1")
os.environ.setdefault("AURUM_P3", "models")
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("p4", "kaggle_kernels/v1_3/aurum-p4-backtest.py")
p4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p4)

m1 = pd.read_parquet("data/phase1/m1_xauusd.parquet")
m1 = m1[~m1.index.duplicated()].sort_index()
m1_ts = m1.index.asi8
hour_ids, first_idx = np.unique(m1_ts // 3_600_000_000_000, return_index=True)

tr = pd.read_parquet("artifacts/v2_1/p4_out/trades_all.parquet")
sub = tr[(tr["strategy"] == "flagship") & (tr["cap"] == 3) &
         (tr["cost_mult"] == 1.0) & (tr["year"] >= 2023)].reset_index(drop=True)

px = pd.read_parquet("data/phase1/h1_xauusd.parquet")
px = px[~px.index.duplicated()].sort_index()
px_ts = px.index.asi8
pos_of_ts = pd.Series(np.arange(len(px)), index=px.index)
px_i = pos_of_ts.reindex(pd.DatetimeIndex(sub["ts"].values, tz="UTC")).values

# q50/q25: rebuild from bundle? Faster: use p4 machinery is heavy; instead
# re-derive from the p4 in-memory path is too heavy — approximate using the
# saved v21 M1 arrays vs H1 R_gross with reason codes from the m1 run saved
# npy + trades overlay/exit_bar.
R_m1 = np.load("artifacts/v2_1/p4_out/v21_m1_R_flagship_cap3_oos.npy")
ip_m1 = np.load("artifacts/v2_1/p4_out/v21_m1_i_part_flagship_cap3_oos.npy")
cost = sub["cost_base"].values
R_h1 = sub["R_gross"].values - cost
d = R_m1 - R_h1
print(f"n={len(d)}  mean diff={d.mean():+.4f}  improved={(d>0.05).sum()}  degraded={(d<-0.05).sum()}  same(|d|<=0.05)={(np.abs(d)<=0.05).sum()}")
big = np.argsort(d)[:12]
print("\nworst 12 degradations (M1 worse than H1):")
print(f"{'idx':>5} {'ts':>20} {'cell':>18} {'ov':>3} {'R_h1':>8} {'R_m1':>8} {'diff':>7} {'exit_bar_h1':>11} {'i_part_h1':>9} {'i_part_m1':>9}")
for i in big:
    print(f"{i:5d} {str(sub['ts'][i])[:16]:>20} {sub['cell_name'][i]:>18} {sub['overlay'][i]:>3} "
          f"{R_h1[i]:8.3f} {R_m1[i]:8.3f} {d[i]:+7.3f} {sub['exit_bar'][i]:>11} {sub['i_part'][i]:>9} {ip_m1[i]:>9}")
# check: are degradations concentrated in overlay==2 (tight-stop) trades?
for ov in (0, 1, 2):
    m = sub["overlay"].values == ov
    if m.any():
        print(f"overlay={ov}: n={m.sum():5d} mean_diff={d[m].mean():+.4f}")
# timeout share in M1 vs H1
print("\nM1 reasons: recompute roughly — R_m1 == -1 exactly (stop):", (np.abs(R_m1 + 1 + cost) < 1e-9).sum(),
      "| tight (R==-q25/m unknown): see overlay=2 rows")
