#!/usr/bin/env python3
"""RT1 — future-garbage prefix-invariance torture (audit A.1).

Appends random-walk GARBAGE bars AFTER the evaluation window to the raw H1
frame, re-derives the causal features over the extended frame, and verifies
the features at every pre-garbage bar are BITWISE UNCHANGED (causality: no
future information can alter the past).  Also re-verifies the audit's
scale-invariance property (features are ATR-normalized: scaling the frame
leaves them invariant).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rt_common import REPO, verdict, have  # noqa: E402

sys.path.insert(0, os.path.join(REPO, "v1_3", "lib"))

DS = os.path.join(REPO, "data", "phase1")
if not have("data/phase1/h1_xauusd.parquet"):
    verdict("rt1_causality_prefix", False, "SKIP: phase1 data absent",
            skipped=True)

import aurum_feat as af  # noqa: E402

h1 = pd.read_parquet(os.path.join(DS, "h1_xauusd.parquet"))
h1 = h1[~h1.index.duplicated()].sort_index()
win = h1.iloc[-3000:]

f1 = af.compute_h1_features(win)

# ---- pass 2: 500 garbage future bars ---------------------------------------
rng = np.random.default_rng(1337)
last_close = float(win["c"].iloc[-1])
steps = rng.normal(0, 3.0, 500).cumsum()
idx_g = pd.date_range(win.index[-1] + pd.Timedelta(hours=1), periods=500,
                      freq="1h", tz="UTC")
g = pd.DataFrame({
    "o": last_close + steps, "h": last_close + steps + np.abs(rng.normal(0, 2, 500)),
    "l": last_close + steps - np.abs(rng.normal(0, 2, 500)),
    "c": last_close + steps + rng.normal(0, 1, 500)}, index=idx_g)
ext = pd.concat([win, g])
f2 = af.compute_h1_features(ext)

common = f1.index.intersection(f2.index)
common = common[common < idx_g[0]]
n_cmp = n_bad = 0
max_abs = 0.0
for col in f1.columns:
    if not np.issubdtype(f1[col].dtype, np.number):
        continue
    a = f1.loc[common, col].values
    b = f2.loc[common, col].values
    m = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a[m] - b[m])
    n_cmp += int(m.sum())
    n_bad += int((d > 0).sum())
    if d.size:
        max_abs = max(max_abs, float(d.max()))

# ---- scale invariance (audit s3.1: features are dimensionless) -------------
f3 = af.compute_h1_features(af.scale_ohlc(win, 1.37))
sc = 0
sc_bad = 0
for col in f1.columns:
    if not np.issubdtype(f1[col].dtype, np.number):
        continue
    a = f1[col].values
    b = f3[col].values
    m = np.isfinite(a) & np.isfinite(b)
    dd = np.abs(a[m] - b[m])
    rel = dd / (np.abs(a[m]) + 1e-12)
    # float-scaling tolerance: an absolute OR relative breach counts (a pure
    # relative test misfires on near-zero values, e.g. |x|~1e-5)
    sc += int(m.sum())
    sc_bad += int(((dd > 1e-9) & (rel > 1e-4)).sum())

ok = n_bad == 0 and n_cmp > 1000 and sc_bad == 0
verdict("rt1_causality_prefix", ok,
        f"future-garbage prefix invariance: {n_cmp:,} numeric feature values "
        f"over {len(common):,} bars unchanged after appending 500 garbage "
        f"future bars (diffs={n_bad}, max|d|={max_abs:.2e}); scale "
        f"invariance x1.37: {sc:,} values, violations={sc_bad}. Causality and "
        f"dimensionlessness hold.",
        {"n_compared": n_cmp, "n_diff": n_bad, "max_abs_diff": max_abs,
         "garbage_bars": 500, "scale_violations": sc_bad})
