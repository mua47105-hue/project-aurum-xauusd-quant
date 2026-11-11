# ============================================================================
# EXP1 - Cohort-Label Memorization Trap: proof + signal ceiling
#
# Q1: Is cohort p_win (K=200 k-NN average over the SAME input features)
#     invertible from features alone by a CHEAP model (XGBoost regressor),
#     i.e. was the 450M transformer's 0.0022 soft-Brier just k-NN replication?
# Q2: What is the achievable signal on REALIZED outcomes from the same
#     features (the actual prediction task)?
# Q3: How does the soft-replication error grow across eras (k-NN estimator
#     drifts out-of-distribution)?
#
# Protocol: train <=2020, val 2021-22, test 2023-25 (matches audit era split).
# All variants use XGBRegressor(squared error) - objective held fixed, only
# the TARGET changes. Brier = MSE for probability targets.
# ============================================================================
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, logloss, auc,
                          REP_IDX, REP_CELLS, logd, save_json)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

def xgb_reg(seed=3, n_jobs=2):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", n_jobs=n_jobs,
                        verbosity=0, random_state=seed)

d = load_all()
feat, R, pw = d["feat"], d["R"], d["pw"]
ts = d["ts"]; yr = year_series(ts).values
mask = d["cohort_valid"]
FEATS = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_dxy","m_y10","m_y2","m_spread","m_vix"]  # 35 (no priors)
X = feat[FEATS].values[mask]
y_year = yr[mask]
tr = y_year <= 2020; te = y_year >= 2023
logd(f"rows={mask.sum()} train={tr.sum()} test={te.sum()}")

res = {"rows": int(mask.sum()), "train": int(tr.sum()), "test": int(te.sum()),
       "cells": {}, "config": "XGBRegressor d5 x300 lr0.1, 35 feats, no priors"}

ERAS = [(2012, 2015), (2016, 2018), (2019, 2020), (2021, 2022), (2023, 2025)]

for ci, cell in zip(REP_IDX, REP_CELLS):
    name = f"{cell[0]}_m{cell[1]}_t{cell[2]}_h{cell[3]}"
    t_soft = pw.values[mask, ci]
    t_real = (R.values[mask, ci] > 0).astype(float)
    row = {}
    for tag, tgt in (("soft", t_soft), ("real", t_real)):
        m = np.isfinite(tgt)
        clf = xgb_reg()
        clf.fit(X[tr & m], tgt[tr & m])
        base = float(np.mean(tgt[tr & m]))
        out = {"base_rate": round(base, 4)}
        for ename, emask in (("val", (y_year >= 2021) & (y_year <= 2022)),
                             ("test", te)):
            mm = emask & m
            p = np.clip(clf.predict(X[mm]), 0, 1)
            y = tgt[mm]
            out[f"{ename}_brier"] = round(brier(p, y), 5)
            out[f"{ename}_logloss"] = round(logloss(p, y), 5)
            out[f"{ename}_auc"] = round(auc(p, y), 5)
            out[f"{ename}_base_brier"] = round(brier(np.full(len(y), base), y), 5)
        era_b = {}
        for a, b_ in ERAS:
            em = (y_year >= a) & (y_year <= b_) & m
            if em.sum() > 100:
                p = np.clip(clf.predict(X[em]), 0, 1)
                era_b[f"{a}-{b_}{'(in-sample)' if a <= 2020 else ''}"] = round(brier(p, tgt[em]), 5)
        out["era_brier"] = era_b
        row[tag] = out
    res["cells"][name] = row
    logd(f"{name}: SOFT test-Brier={row['soft']['test_brier']} (era drift "
         f"{row['soft']['era_brier']}) | REAL test-Brier={row['real']['test_brier']} "
         f"vs base {row['real']['test_base_brier']} auc {row['real']['test_auc']}")

for tag in ("soft", "real"):
    res[f"mean_{tag}_test_brier"] = round(float(np.mean(
        [res["cells"][c][tag]["test_brier"] for c in res["cells"]])), 5)
    res[f"mean_{tag}_test_auc"] = round(float(np.nanmean(
        [res["cells"][c][tag]["test_auc"] for c in res["cells"]])), 5)
res["mean_real_test_base_brier"] = round(float(np.mean(
    [res["cells"][c]["real"]["test_base_brier"] for c in res["cells"]])), 5)

logd("===== EXP1 SUMMARY =====")
logd(f"cohort p_win replication (k-NN inversion), test 2023-25: "
     f"Brier={res['mean_soft_test_brier']}  <- the '0.0022-style' memorization, CHEAPLY replicable")
logd(f"realized outcome prediction, same features:              "
     f"Brier={res['mean_real_test_brier']} vs base-rate {res['mean_real_test_base_brier']}, "
     f"AUC={res['mean_real_test_auc']}")
save_json(res, os.path.join(OUT, "exp1_memorization.json"))
