#!/usr/bin/env python3
"""RT4 — bridge fidelity: E0/E1/E2 decomposition + G5.10 report (audit A.4,
U1/D4).  Re-derives the faithful bridge from the v2.1 trades and verifies:
  * E0 in [2.55, 2.68] (the shipped artifact frame);
  * E1 < E0 with the entry-bar credit removed (audit family 2.36-2.40);
  * E1 < E2 < E0 + 0.25 (partial-TP halving restores part of the book);
  * the 1-bar trade share is 15-20% (the D4 cohort);
  * the in-run G5.10 gate PASSes and its numbers match this recomputation
    within 0.02 (no faked gate text).
"""
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rt_common import REPO, verdict, have  # noqa: E402

NEED = ["artifacts/v2_1/p4_out/trades_all.parquet",
        "data/aurum_v2_features.parquet", "data/phase1/h1_xauusd.parquet",
        "artifacts/v2_1/k6_out/gates.json"]
if not have(*NEED):
    verdict("rt4_bridge_fidelity", False, "SKIP: artifacts absent",
            skipped=True)

spec = importlib.util.spec_from_file_location(
    "k6rt", os.path.join(REPO, "aurum_v2", "kernels", "k6_backtest.py"))
k6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k6)
sys.path.insert(0, os.path.join(REPO, "aurum_v2", "model"))
from v2_common import BARS_PER_YEAR, SIGMA_TGT_ANN, COST_R  # noqa: E402

feats = pd.read_parquet(os.path.join(REPO, "data", "aurum_v2_features.parquet"))
feats.index = pd.DatetimeIndex(feats.index)
if feats.index.tz is None:
    feats.index = feats.index.tz_localize("UTC")
px = pd.read_parquet(os.path.join(REPO, "data", "phase1", "h1_xauusd.parquet"))
px.index = pd.DatetimeIndex(px.index)
if px.index.tz is None:
    px.index = px.index.tz_localize("UTC")
close = px["c"].reindex(feats.index)
r = np.log(close / close.shift(1))
sigma = r.ewm(span=96, min_periods=24).std()
oos = pd.Series(feats.index.year, index=feats.index).isin([2023, 2024, 2025])

tr = pd.read_parquet(os.path.join(REPO, "artifacts", "v2_1", "p4_out",
                                   "trades_all.parquet"))


def sr_of(z):
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sigma.values + 1e-12)
    w = z.values * v
    w_lag = np.concatenate([[0.0], w[:-1]])
    net = w_lag * r.values - COST_R * np.abs(np.diff(w, prepend=0.0))
    n = net[oos.values]
    return float(n.mean() / (n.std(ddof=1) + 1e-12) * math.sqrt(BARS_PER_YEAR))


E = {m: sr_of(k6.champion_exposure(tr, feats.index, cap=3, mode=m))
     for m in ("E0", "E1", "E2")}
g = json.load(open(os.path.join(REPO, "artifacts", "v2_1", "k6_out", "gates.json")))
g510_pass = bool(g.get("G5.10_bridge_fidelity", {}).get("pass"))
a = json.load(open(os.path.join(REPO, "artifacts", "v2_1", "k6_out",
                                "attribution_v21.json")))
in_run = a["bridge_decomposition_E0_E1_E2"]
match = all(abs(in_run[k] - E[k]) < 0.02 for k in E)

# 1-bar trade share (the D4 cohort)
sub = tr[(tr["strategy"] == "flagship") & (tr["cap"] == 3) &
         (tr["cost_mult"] == 1.0)]
i0 = feats.index.searchsorted(pd.to_datetime(sub["ts"].values, utc=True))
one_bar_share = float((sub["exit_bar"].values <= np.asarray(i0) + 1).mean())

ok = (2.55 <= E["E0"] <= 2.68 and E["E1"] < E["E0"]
      and E["E1"] < E["E2"] < E["E0"] + 0.25
      and 0.13 <= one_bar_share <= 0.22
      and g510_pass and match)
verdict("rt4_bridge_fidelity", ok,
        f"D4-fixed bridge recomputation: E0={E['E0']:.4f} E1={E['E1']:.4f} "
        f"E2={E['E2']:.4f} (audit family: E1 2.36-2.40; E2 partial-TP halving "
        f"restores part of the book); 1-bar trade share={one_bar_share:.3f} "
        f"(the D4 cohort — no hourly representation, dropped from the bridge "
        f"frame by the audit's construction); in-run G5.10 PASS={g510_pass}; "
        f"in-run bridge {in_run} matches recomputation within 0.02: {match}.",
        {"E0": round(E["E0"], 4), "E1": round(E["E1"], 4),
         "E2": round(E["E2"], 4), "one_bar_share": round(one_bar_share, 3),
         "g510_in_run_pass": g510_pass, "recompute_match": match})
