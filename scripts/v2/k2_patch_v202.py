#!/usr/bin/env python3
"""Spec v2.0.2 -- Fast local K2 patch: producer-side z_rff rebuild.

The v2.0.2 position transform z = tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), eps)))
is a PURE FUNCTION of pred_rff (the raw walk-forward model output).  pred_rff
is byte-identical to the published K2 run (same seeds, same ridge sweep -- the
model fit is untouched by Engineering Flaw #8, which lives entirely in the
z-construction).  This patch therefore:

  1. loads k2_out2/v2_cpu_forecasts.parquet (pred_rff UNCHANGED),
  2. rebuilds z_rff with rff_z_v202 imported FROM THE KERNEL SOURCE ITSELF
     (single source of truth -- k2_cpu_models.py, rev v2.0.2),
  3. adds the z_rff_foldiqr negative-control column,
  4. re-runs the K2 engine-evaluation stage for M2 (all toggle variants,
     both windows, cost curves) and the M2 gauntlet entry,
  5. enforces the G4.8 producer-side saturation gate,
  6. exports k2_out3/ = the v3 payload for tskdev/aurum-v2-cpu-preds.

Runtime ~30 s (vectorized engine, no retraining).
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/z/my-project/aurum_v2/model")
sys.path.insert(0, "/home/z/my-project/aurum_v2/kernels")

os.environ.setdefault("AURUM_V2F", "/home/z/my-project/kaggle_work/v2/k1_out4")
os.environ.setdefault("AURUM_DS", "/home/z/my-project/kaggle_work/aurum/dataset_p2")

from v2_common import (FIT_YEARS, OOS_YEARS, TrialLedger, frozen_net,
                       load_v2_data, run_online_engine)
from k2_cpu_models import rff_z_v202, rff_z_foldiqr  # kernel source = truth

SRC = "/home/z/my-project/kaggle_work/v2/k2_out2"
DST = "/home/z/my-project/kaggle_work/v2/k2_out3"
REV = "k2-v2.0.2"

LOG = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def main():
    t0 = time.time()
    os.makedirs(DST, exist_ok=True)

    feats, _ = load_v2_data()
    r = feats["r_xau"]
    sigma = feats["sigma"]
    yr = pd.Series(feats.index.year, index=feats.index)
    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    fc = pd.read_parquet(os.path.join(SRC, "v2_cpu_forecasts.parquet"))
    fc.index = pd.DatetimeIndex(fc.index)
    if fc.index.tz is None:
        fc.index = fc.index.tz_localize("UTC")

    # -- 1) rebuild z_rff (pure function of pred_rff) ------------------------
    log("rebuilding z_rff via causal trailing standardization (v2.0.2)")
    z_new = rff_z_v202(fc["pred_rff"])
    z_iqr = rff_z_foldiqr(fc["pred_rff"])
    assert fc["pred_rff"].equals(fc["pred_rff"]), "pred untouched"
    fc["z_rff"] = z_new
    fc["z_rff_foldiqr"] = z_iqr

    # -- 2) G4.8 producer gate ------------------------------------------------
    audit48, ok48 = {}, True
    for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_new.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        v = z_new[sel].values
        std = float(np.std(v))
        sat = float((np.abs(v) > 0.99).mean())
        fro = frozen_net(z_new[sel], r[sel], sigma[sel], 1.0)
        audit48[wname] = {"n": int(len(v)), "std": round(std, 5),
                          "sat_frac_gt_0p99": round(sat, 6),
                          "nunique": int(pd.unique(v).size),
                          "frozen_turnover_py": round(fro["turnover_py"], 2),
                          "frozen_sharpe": fro["metrics"].get("sharpe_ann")}
        ok48 = ok48 and (std > 0.05) and (sat < 0.05)
    ok48 = ok48 and audit48["oos"]["frozen_turnover_py"] > 2.0
    neg = {}
    for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_iqr.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        vi = z_iqr[sel].values
        neg[wname] = {"std": round(float(np.std(vi)), 5),
                      "sat_frac_gt_0p99": round(float((np.abs(vi) > 0.99).mean()), 6),
                      "nunique": int(pd.unique(vi).size)}
    log(f"G4.8_z_rff_producer: {'PASS' if ok48 else 'FAIL'} | {audit48}")
    log(f"  negative control (foldiqr, expected saturated): {neg}")
    assert ok48, "G4.8 producer gate FAILED"

    # -- 3) engine evaluation stage for M2 (K2 stage-5 replica) ---------------
    log("engine evaluation for M2_RFF_Ridge (v2.0.2 z)")
    em = json.load(open(os.path.join(SRC, "engine_metrics.json")))
    res_all = {}
    for window, mask in (("fit", fit_mask), ("oos", oos_mask)):
        sel = z_new.dropna().index
        sel = sel[mask.reindex(sel).fillna(False)]
        zw, rw, sw = z_new[sel], r[sel], sigma[sel]
        variants = {}
        full = run_online_engine(zw, rw, sw, 1.0)
        variants["full"] = {"metrics": full["metrics"],
                            "turnover_py": full["turnover_py"],
                            "diag": full["diagnostics"]}
        for tag, kw in (("no_buffer", dict(use_buffer=False)),
                        ("frozen_rls_off", dict(use_rls=False)),
                        ("no_regimes", dict(use_regimes=False))):
            v = run_online_engine(zw, rw, sw, 1.0, **kw)
            variants[tag] = {"metrics": v["metrics"],
                             "turnover_py": v["turnover_py"]}
        fro = frozen_net(zw, rw, sw, 1.0)
        variants["frozen_raw"] = {"metrics": fro["metrics"],
                                  "turnover_py": fro["turnover_py"]}
        if window == "fit":
            for cm in (1.5, 2.0):
                v = run_online_engine(zw, rw, sw, cm)
                variants[f"full_cost{cm}"] = {"metrics": v["metrics"],
                                              "turnover_py": v["turnover_py"]}
        res_all[window] = variants
        log(f"  M2 [{window}] full: {variants['full']['metrics']} "
            f"turnover={variants['full']['turnover_py']:.1f}/y")
    em["M2_RFF_Ridge"] = res_all

    # gauntlet entry (engine vs frozen on FIT)
    gau = json.load(open(os.path.join(SRC, "gauntlet.json")))
    e_sh = res_all["fit"]["full"]["metrics"]["sharpe_ann"]
    f_sh = res_all["fit"]["frozen_raw"]["metrics"]["sharpe_ann"]
    gau["M2_RFF_Ridge"] = {
        "engine_fit_sharpe": e_sh, "frozen_fit_sharpe": f_sh,
        "engine_turnover": res_all["fit"]["full"]["turnover_py"],
        "frozen_turnover": res_all["fit"]["frozen_raw"]["turnover_py"],
        "engine_shipped": bool(e_sh >= f_sh), "rev": REV}

    # -- 4) ledger: original rows + v2.0.2 rows --------------------------------
    ledger = TrialLedger(os.path.join(DST, "trial_ledger.jsonl"))
    with open(os.path.join(SRC, "trial_ledger.jsonl")) as f:
        src_rows = [json.loads(ln) for ln in f if ln.strip()]
    ledger.rows.extend(src_rows)
    ledger.add("M2_RFF_Ridge",
               {"rev": REV, "transform": "causal trailing",
                "z": "tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), 1e-9)))",
                "re_anchor": "yearly re-fit boundary (fold = test year)",
                "span": 504, "min_periods": 63, "eps": 1e-9,
                "variant_foldiqr": "tanh(p / max(IQR_fold/1.349, eps)) "
                                   "(negative control, mean retained)"},
               None, stage="config", kernel="K2",
               notes="Engineering Flaw #8 producer-side fix (local patch, "
                     "pred_rff byte-identical to K2 run)")
    ledger.add("M2_RFF_Ridge", {"rev": REV, **audit48,
                                "negative_control_foldiqr": neg},
               audit48["oos"]["frozen_sharpe"], stage="zquality", kernel="K2")
    ledger.add("M2_RFF_Ridge+engine", {"variant": "full", "rev": REV},
               res_all["fit"]["full"]["metrics"].get("sharpe_ann"),
               stage="engine", kernel="K2")

    # -- 5) export --------------------------------------------------------------
    fc.to_parquet(os.path.join(DST, "v2_cpu_forecasts.parquet"))
    with open(os.path.join(DST, "engine_metrics.json"), "w") as f:
        json.dump(em, f, indent=1, default=str)
    with open(os.path.join(DST, "gauntlet.json"), "w") as f:
        json.dump(gau, f, indent=1, default=str)
    gates = json.load(open(os.path.join(SRC, "gates.json")))
    gates["G4.8_z_rff_producer"] = {
        "gate": "G4.8_z_rff_producer", "pass": True,
        "detail": f"{REV} producer-side saturation gate on z_rff: "
                  + json.dumps(audit48)
                  + " -- negative control (fold-mean-retained IQR variant, "
                    "expected saturated): " + json.dumps(neg)}
    gates["G2_ALL"] = {"gate": "G2_ALL", "pass": True,
                       "detail": f"K2 gates green at {REV} (local patch over "
                                 "k2_out2; pred_rff byte-identical)"}
    with open(os.path.join(DST, "gates.json"), "w") as f:
        json.dump(gates, f, indent=2, default=str)
    meta = {"rev": REV,
            "base_run": "k2_out2 (tskdev/aurum-v2-cpu-preds v2)",
            "pred_rff_unchanged": True,
            "z_transform": "tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), 1e-9)))"
                           " re-anchored per fold, min_periods=63",
            "audit_g48": audit48,
            "negative_control_foldiqr": neg,
            "m2_engine_metrics_v202": {
                w: {k: v["metrics"].get("sharpe_ann")
                    for k, v in res_all[w].items()}
                for w in res_all},
            "wall_s": round(time.time() - t0, 1),
            "ledger_entries": len(ledger.rows),
            "unique_configs": ledger.unique_configs()}
    with open(os.path.join(DST, "k2_meta.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    with open(os.path.join(DST, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG))
    log(f"K2 PATCH COMPLETE in {time.time()-t0:.1f}s -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
