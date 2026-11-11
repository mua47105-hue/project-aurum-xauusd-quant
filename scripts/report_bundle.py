#!/usr/bin/env python3
"""Bundle all result JSONs into report_data.json for the docx generator."""
import os, json, hashlib
import numpy as np
import pandas as pd

ROOT = os.environ.get("AURUM_RESULTS", "/home/z/my-project/kaggle_work/aurum")
OUT = "/home/z/my-project/scripts/report_data.json"

def jload(*p):
    fp = os.path.join(ROOT, *p)
    return json.load(open(fp)) if os.path.isfile(fp) else None

bundle = {}

# ---------------- p2 training ----------------
bundle["stageA"] = jload("results/p2_train/stageA_result.json")
bundle["stageB"] = jload("results/p2_train/stageB_result.json")
bundle["gates_p2"] = jload("results/p2_train/p2_train_gates_verified.json")
fr = jload("results/p2_train/fold_results.json") or {}
bundle["folds"] = {y: fr[y] for y in sorted(fr, key=lambda k: int(k) if k.isdigit() else 0)
                   if y.isdigit()}
bundle["temperatures"] = jload("results/p2_train/temperatures.json")
manifest = jload("results/p2_train/manifest.json") or {}
bundle["manifest"] = manifest

# ---------------- p3 tables ----------------
bundle["gates_p3"] = jload("results/p3_tables_latest/gates.json")
bundle["p3_summary"] = jload("results/p3_tables_latest/p3_tables_summary.json")
bundle["p3_meta"] = jload("results/p3_tables_latest/p3_cell_meta.json")

# ---------------- p4 backtest ----------------
bundle["gates_p4"] = jload("results/p4_backtest_latest/gates.json")
bt = jload("results/p4_backtest_latest/backtest_main.json")
bundle["backtest"] = bt
gbdt_audit = jload("results/p4_backtest_latest/gbdt_audit.json") or []
bundle["gbdt_audit_last"] = gbdt_audit[-2:] if gbdt_audit else []

# ---------------- live ----------------
bundle["live"] = jload("results/live_daily/decision.json")
bundle["live_gates"] = jload("results/live_daily/gates.json")

# ---------------- journal digest ----------------
jp = os.path.join(ROOT, "execution_journal.md")
bundle["journal_chars"] = os.path.getsize(jp) if os.path.isfile(jp) else 0

# ---------------- quick derived stats for the narrative ----------------
if bt:
    r3 = bt.get("aurum_r3", {}).get("cost_1.0", {})
    oos = r3.get("oos", {})
    full = r3.get("full", {})
    bundle["headline"] = {
        "oos_n": oos.get("n_trades"), "oos_E": oos.get("expectancy_R"),
        "oos_sharpe": oos.get("sharpe_per_trade"),
        "oos_pf": oos.get("profit_factor"), "oos_maxdd": oos.get("max_dd_R"),
        "full_n": full.get("n_trades"), "full_E": full.get("expectancy_R"),
        "brier": oos.get("brier"), "ece": oos.get("ece"),
        "ann_sharpe": oos.get("sharpe_annualized"),
    }
    # best/worst baselines on OOS E
    base = [s for s in ("gbdt", "cohort", "ema", "donchian", "buyhold")
            if bt.get(s, {}).get("cost_1.0", {}).get("oos", {}).get("n_trades", 0) > 0]
    es = {s: bt[s]["cost_1.0"]["oos"].get("expectancy_R") for s in base}
    bundle["baseline_rank"] = sorted(es.items(), key=lambda kv: -(kv[1] or -9))
    dsr = bt.get("dsr_oos", {}).get("members", {})
    if "aurum_r3|c1.0" in dsr:
        bundle["dsr_r3"] = dsr["aurum_r3|c1.0"]

json.dump(bundle, open(OUT, "w"), indent=1, default=float)
print("bundle ->", OUT)
print("keys:", sorted(bundle.keys()))
if bt:
    print("headline:", json.dumps(bundle["headline"], indent=1))
    print("dsr_r3:", bundle.get("dsr_r3"))
