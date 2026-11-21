#!/usr/bin/env python3
"""Build the v2.1 fixed K2 forecasts parquet (U2, no-retraining path).

The sandbox has 3 GB RAM — the K2 RFF stage is memory-bound here (the audit's
4 GB sandbox hit the same wall), so the producer-side fix (already patched in
aurum_v2/kernels/k2_cpu_models.py, re-run on Kaggle in Step 6) is applied to
the SHIPPED pred_gbdt series — exactly the audit's s5.3 verification method
("applying the program's own Flaw-#8 fix to pred_gbdt, no retraining").

Verifies:
  1. strict G4.8 z-quality on the FIXED z_gbdt (std>0.05, frac|z|>.99<.05,
     frac|z|>.90<.25, |mean|<.35) — the Flaw #9 disease is gone;
  2. the collapse: M1_GBDT_v2 OOS engine Sharpe 1.253 (fold-local artifact)
     -> ~0.16 (honest), via frozen_net on the OOS slice;
  3. the shipped z_gbdt's OOS 2023 pathology reproduced first (mean +0.95,
     ~100% |z|>0.9) so the before/after is on the record.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
K2 = os.path.join(HERE, "..", "..", "aurum_v2", "kernels",
                  "k2_cpu_models.py")
spec = importlib.util.spec_from_file_location("k2mod", K2)
k2 = importlib.util.module_from_spec(spec)
sys.modules["k2mod"] = k2
spec.loader.exec_module(k2)

sys.path.insert(0, os.path.join(HERE, "..", "..", "aurum_v2", "model"))
from v2_common import frozen_net  # noqa: E402

SRC = "/home/z/my-project/kaggle_work/v21_restore/aurum-v2-cpu-preds/v2_cpu_forecasts.parquet"
OUT_DIR = os.path.join(HERE, "..", "..", "artifacts", "v2_1", "k2_fixed")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(SRC)
df.index = pd.DatetimeIndex(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")

r = df["r_xau"]
sigma = df["sigma"]
yr = pd.Series(df.index.year, index=df.index)
oos = yr.isin([2023, 2024, 2025])

out = {"spec": "v2.1.0", "method": "audit s5.3 no-retraining: rff_z_v202(pred_gbdt) on the shipped K2 parquet; producer-side fix patched in k2_cpu_models.py (Kaggle re-run Step 6)", "before": {}, "after": {}}


def strict_audit(z):
    res = {}
    for wname, mask in (("fit", yr.isin(range(2016, 2023))), ("oos", oos)):
        sel = z.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        v = z[sel].values
        res[wname] = {"n": int(len(v)), "std": round(float(np.std(v)), 4),
                      "frac_gt_0p99": round(float((np.abs(v) > 0.99).mean()), 5),
                      "frac_gt_0p90": round(float((np.abs(v) > 0.90).mean()), 5),
                      "abs_mean": round(float(np.abs(np.mean(v))), 4)}
    return res


# ---- before: the shipped fold-local transform (Flaw #9 pathology) ---------
z_old = df["z_gbdt"]
out["before"]["strict_g48"] = strict_audit(z_old)
out["before"]["oos_2023_mean_z"] = round(float(z_old[yr == 2023].mean()), 4)
out["before"]["oos_sharpe_foldlocal"] = round(float(
    frozen_net(z_old[oos], r[oos], sigma[oos], 1.0)["metrics"]["sharpe_ann"]), 4)

# ---- after: the program's own Flaw-#8 fix applied to pred_gbdt ------------
z_new = k2.rff_z_v202(df["pred_gbdt"])
out["after"]["strict_g48"] = strict_audit(z_new)
out["after"]["oos_2023_mean_z"] = round(float(z_new[yr == 2023].mean()), 4)
out["after"]["oos_sharpe_fixed"] = round(float(
    frozen_net(z_new[oos], r[oos], sigma[oos], 1.0)["metrics"]["sharpe_ann"]), 4)
per_year = {}
for y in (2023, 2024, 2025):
    m = yr == y
    per_year[int(y)] = round(float(
        frozen_net(z_new[m], r[m], sigma[m], 1.0)["metrics"]["sharpe_ann"]), 3)
out["after"]["oos_sharpe_fixed_by_year"] = per_year
out["after"]["mean_oos_exposure"] = round(float(z_new[oos].mean()), 4)

# strict G4.8 verdict on the fixed z
ok = True
for w in ("fit", "oos"):
    a = out["after"]["strict_g48"][w]
    ok = ok and a["std"] > 0.05 and a["frac_gt_0p99"] < 0.05 \
        and a["frac_gt_0p90"] < 0.25 and a["abs_mean"] < 0.35
out["after"]["strict_g48_pass"] = bool(ok)

# ---- write the v2.1 forecasts parquet --------------------------------------
df["z_gbdt_v202_foldlocal"] = df["z_gbdt"]     # preserve the artifact for the record
df["z_gbdt"] = z_new                           # fixed transform in the canonical slot
df.to_parquet(os.path.join(OUT_DIR, "v2_cpu_forecasts.parquet"))
with open(os.path.join(OUT_DIR, "u2_fix_report.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
