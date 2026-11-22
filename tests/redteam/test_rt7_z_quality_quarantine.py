#!/usr/bin/env python3
"""RT7 — strict all-producer z-quality + U2/U3 quarantine (audit A.7).

Recomputes the strict G4.8 audit (std > 0.05, frac(|z|>0.99) < 0.05,
frac(|z|>0.90) < 0.25, |mean(z)| < 0.35, FIT + OOS) on ALL continuous z
producers in the v2.1 FIXED forecasts parquet (z_rff, and z_gbdt under the
Flaw-#9 causal trailing standardization), verifies the fold-local artifact
is preserved for the record and FAILS the strict gate (the negative
control), and verifies the U3 quarantine registry in the frozen policy.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _rt_common import REPO, verdict, have, load_json  # noqa: E402

NEED = ["artifacts/v2_1/k2_fixed/v2_cpu_forecasts.parquet",
        "v1_3/config/frozen_policy.json"]
if not have(*NEED):
    verdict("rt7_z_quality_quarantine", False, "SKIP: fixed forecasts absent",
            skipped=True)

df = pd.read_parquet(os.path.join(REPO, "artifacts", "v2_1", "k2_fixed",
                                  "v2_cpu_forecasts.parquet"))
df.index = pd.DatetimeIndex(df.index)
if df.index.tz is None:
    df.index = df.index.tz_localize("UTC")
yr = pd.Series(df.index.year, index=df.index)
fit = yr.isin(range(2016, 2023))
oos = yr.isin([2023, 2024, 2025])


def strict(z):
    out = {}
    for wname, mask in (("fit", fit), ("oos", oos)):
        sel = z.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        v = z[sel].values
        out[wname] = {"n": int(len(v)), "std": float(np.std(v)),
                      "f99": float((np.abs(v) > 0.99).mean()),
                      "f90": float((np.abs(v) > 0.90).mean()),
                      "absmean": float(np.abs(np.mean(v)))}
    return out


def passes(a):
    return all(w["std"] > 0.05 and w["f99"] < 0.05 and w["f90"] < 0.25
               and w["absmean"] < 0.35 for w in a.values())


aud_rff = strict(df["z_rff"])
aud_gbm = strict(df["z_gbdt"])                      # FIXED transform
aud_gbm_old = strict(df["z_gbdt_v202_foldlocal"])   # artifact (negative ctl)

rff_ok = passes(aud_rff)
gbm_ok = passes(aud_gbm)
old_fails = not passes(aud_gbm_old)

fp = load_json("v1_3/config/frozen_policy.json")
q = fp["v21_production_registry"]["quarantined"]
u3 = q["u3_top60_pruning_evidence"]
reg_ok = (len(q["v2_active_stack"]) >= 8
          and q["v2_active_stack"][0] == "M1_GBDT_v2"
          and u3["oos_sharpe_top60_fixed_transform"] == -0.215)

ok = rff_ok and gbm_ok and old_fails and reg_ok
verdict("rt7_z_quality_quarantine", ok,
        f"strict G4.8 on ALL producers: z_rff "
        f"(fit std={aud_rff['fit']['std']:.3f}, oos std={aud_rff['oos']['std']:.3f}) "
        f"PASS={rff_ok}; z_gbdt FIXED (fit std={aud_gbm['fit']['std']:.3f}, "
        f"oos std={aud_gbm['oos']['std']:.3f}, oos |mean|="
        f"{aud_gbm['oos']['absmean']:.3f}) PASS={gbm_ok}; negative control "
        f"fold-local artifact (oos |mean|={aud_gbm_old['oos']['absmean']:.3f}, "
        f"frac|z|>0.99={aud_gbm_old['oos']['f99']:.3f}) FAILS as expected="
        f"{old_fails}; U3 quarantine registry in frozen_policy.json "
        f"(8 quarantined v2 models, top-60 pruning -0.215 OOS recorded) "
        f"OK={reg_ok}.",
        {"rff_pass": rff_ok, "gbm_fixed_pass": gbm_ok,
         "negative_control_fails": old_fails, "quarantine_registry_ok": reg_ok,
         "z_gbdt_oos": aud_gbm["oos"], "z_gbdt_foldlocal_oos": aud_gbm_old["oos"]})
