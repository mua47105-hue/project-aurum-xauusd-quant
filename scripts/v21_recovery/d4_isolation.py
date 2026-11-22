#!/usr/bin/env python3
"""D4 isolation: E0/E1 champion exposure on the SHIPPED v13 trades via
frozen_net. Expect E0=2.6112 (shipped reference) and E1 in 2.36-2.38 (audit).
Whatever we get reveals the bug location."""
import importlib.util
import math
import sys

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("k6", "aurum_v2/kernels/k6_backtest.py")
k6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k6)
sys.path.insert(0, "aurum_v2/model")
from v2_common import frozen_net, BARS_PER_YEAR  # noqa: E402

feats = pd.read_parquet("data/aurum_v2_features.parquet")
feats.index = pd.DatetimeIndex(feats.index)
if feats.index.tz is None:
    feats.index = feats.index.tz_localize("UTC")
# r_xau/sigma are DERIVED (load_v2_data convention), not stored in the parquet
px = pd.read_parquet("data/phase1/h1_xauusd.parquet")
px.index = pd.DatetimeIndex(px.index)
if px.index.tz is None:
    px.index = px.index.tz_localize("UTC")
close = px["c"].reindex(feats.index)
feats["r_xau"] = np.log(close / close.shift(1))
feats["sigma"] = feats["r_xau"].ewm(span=96, min_periods=24).std()
r = feats["r_xau"]; sigma = feats["sigma"]
yr = pd.Series(feats.index.year, index=feats.index)
oos = yr.isin([2023, 2024, 2025])

tr = pd.read_parquet(
    "/home/z/my-project/kaggle_work/v21_restore/aurum-v13-artifacts/trades_all.parquet")

for mode in ("E0", "E1", "E2"):
    z = k6.champion_exposure(tr, feats.index, cap=3, mode=mode)
    zv = z[oos.index[0]:]
    res = frozen_net(z, r, sigma, 1.0)
    net_oos = res["net"][oos]
    sd = net_oos.std(ddof=1)
    sr = float(net_oos.mean() / sd * math.sqrt(BARS_PER_YEAR))
    nz = (z.values != 0).mean()
    print(f"{mode}: OOS SR={sr:.4f}  (E0 ref 2.6112, E1 audit 2.36-2.38)  "
          f"nonzero_bars={nz*100:.1f}%  turnover_py={res['turnover_py']:.1f}")
    # decomposition: gross vs cost
    v = (0.15 / math.sqrt(BARS_PER_YEAR)) / (sigma + 1e-12)
    w = z.values * v.values
    w_lag = np.concatenate([[0.0], w[:-1]])
    gross = (w_lag * r.values)[oos.values]
    dz = np.abs(np.diff(w, prepend=0.0))
    cost = (0.0008 * dz)[oos.values]
    print(f"     OOS gross ann={gross.mean()/gross.std(ddof=1)*math.sqrt(BARS_PER_YEAR):.3f} "
          f"cost_drag={cost.sum()/max(1e-12,-gross.sum())*100:.1f}% of gross")
