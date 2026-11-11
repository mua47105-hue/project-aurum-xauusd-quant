# ============================================================================
# EXP2 - Target-Formulation Tournament (the Fix for Failure #1)
#
# Spec v1.1 Sec 4.2 rule: a model trained on soft labels must not lose
# out-of-sample log-loss/Brier against hard realized labels, or soft-label
# machinery is switched off. This experiment runs that rule LOCALLY before
# any GPU is spent.
#
# Variants (same features, same rows, same XGBRegressor objective):
#   T0_cohort    target = cohort p_win (pilot status quo)
#   T1_realized  target = realized win
#   T2_smoothed  target = triangular-smoothed realized win (adjacent cells)
#   T3a_mix30    0.7*realized + 0.3*cohort
#   T3b_mix50    0.5*realized + 0.5*cohort
#   T4_prior_in  target = realized; cohort p_win (all 48) as INPUT features
#   T5_shallow   realized, shallow/strongly-regularized trees (variance cut)
# References: base_rate, cohort-direct (k-NN strategy, no learning),
#             canonical fixed cell.
# Eval on REALIZED outcomes, test era 2023-25 (+ val 2021-22):
#   per-cell Brier/logloss/AUC; 8-cell EV-argmax trade sim (pilot protocol);
#   block-bootstrap CI on canonical cell for key deltas.
# ============================================================================
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, logloss, auc,
                          block_boot_ci, REP_IDX, REP_CELLS, CANON,
                          build_smooth_kernel, logd, save_json)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

def xgb_reg(seed=3, n_jobs=2, depth=5, n_est=300):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", n_jobs=n_jobs, verbosity=0,
                        random_state=seed)

d = load_all()
feat, R, pw, be = d["feat"], d["R"], d["pw"], d["be"]
ts = d["ts"]; yr = year_series(ts).values
mask = d["cohort_valid"]
FEATS = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_dxy","m_y10","m_y2","m_spread","m_vix"]
X0 = feat[FEATS].values[mask]
P48 = pw.values[mask]                      # cohort p_win, all 48 cells
X4 = np.hstack([X0, P48])                  # T4 inputs (83 feats)
B48 = be.values[mask]                      # cohort b_eff (EV sim only)
W48_real = (R.values[mask] > 0).astype(float)   # realized win matrix
R48 = R.values[mask]
y_year = yr[mask]
tr = y_year <= 2020; va = (y_year >= 2021) & (y_year <= 2022); te = y_year >= 2023
week = pd.Series(ts).dt.isocalendar().year.astype(str).str.cat(
       pd.Series(ts).dt.isocalendar().week.astype(str), sep="-").values[mask]

Ksm = build_smooth_kernel()
Wsm = W48_real @ Ksm.T                     # smoothed realized targets (N,48)
assert np.isfinite(Wsm).all()

def make_target(variant, ci):
    t_real = W48_real[:, ci]
    if variant == "T0_cohort":   return P48[:, ci]
    if variant == "T1_realized": return t_real
    if variant == "T2_smoothed": return Wsm[:, ci]
    if variant == "T3a_mix30":   return 0.7 * t_real + 0.3 * P48[:, ci]
    if variant == "T3b_mix50":   return 0.5 * t_real + 0.5 * P48[:, ci]
    if variant in ("T4_prior_in", "T5_shallow"): return t_real
    raise ValueError(variant)

VARIANTS = ["T0_cohort", "T1_realized", "T2_smoothed", "T3a_mix30",
            "T3b_mix50", "T4_prior_in", "T5_shallow"]

res = {"cells": {}, "sim": {}, "protocol": "train<=2020 val21-22 test23-25, "
       "XGBRegressor sq, 35 feats (+48 cohort p_win for T4)"}

# ---- per-cell fits ----
preds = {v: {} for v in VARIANTS}
for ci, cell in zip(REP_IDX, REP_CELLS):
    name = f"{cell[0]}_m{cell[1]}_t{cell[2]}_h{cell[3]}"
    cres = {}
    for v in VARIANTS:
        tgt = make_target(v, ci)
        X_ = X4 if v == "T4_prior_in" else X0
        if v == "T5_shallow":
            clf = xgb_reg(depth=3, n_est=150)
        else:
            clf = xgb_reg()
        clf.fit(X_[tr], tgt[tr])
        p_te = np.clip(clf.predict(X_[te]), 0, 1)
        p_va = np.clip(clf.predict(X_[va]), 0, 1)
        preds[v][ci] = (p_te, p_va)
        y_te = W48_real[te, ci]
        cres[v] = {"test_brier": round(brier(p_te, y_te), 5),
                   "test_logloss": round(logloss(p_te, y_te), 5),
                   "test_auc": round(auc(p_te, y_te), 5),
                   "val_brier": round(brier(p_va, W48_real[va, ci]), 5)}
    # references
    cres["base_rate"] = {"test_brier": round(brier(np.full(te.sum(), W48_real[tr, ci].mean()), W48_real[te, ci]), 5)}
    cres["cohort_direct"] = {"test_brier": round(brier(P48[te, ci], W48_real[te, ci]), 5),
                             "test_auc": round(auc(P48[te, ci], W48_real[te, ci]), 5)}
    res["cells"][name] = cres
    logd(f"{name}: " + " | ".join(
        f"{v}:{res['cells'][name][v]['test_brier']}" for v in VARIANTS) +
        f" | cohort_direct:{cres['cohort_direct']['test_brier']}")

# ---- means ----
for v in VARIANTS + ["cohort_direct"]:
    res[f"mean_{v}_test_brier"] = round(float(np.mean(
        [res["cells"][c][v]["test_brier"] for c in res["cells"]])), 5)
    res[f"mean_{v}_test_auc"] = round(float(np.nanmean(
        [res["cells"][c][v]["test_auc"] for c in res["cells"]])), 5)
res["mean_base_test_brier"] = round(float(np.mean(
    [res["cells"][c]["base_rate"]["test_brier"] for c in res["cells"]])), 5)

# ---- EV-argmax trade sim on the 8 cells (pilot protocol) ----
te_idx = np.flatnonzero(te)
for v in VARIANTS + ["cohort_direct"]:
    P = np.stack([preds[v][ci][0] if v in preds else P48[te, ci] for ci in REP_IDX], 1)
    B = np.stack([B48[te, ci] for ci in REP_IDX], 1)
    B = np.where(np.isfinite(B), B, 1.0)
    evm = P * B - (1 - P)
    picks = np.argmax(evm, 1)
    pr_pick = P[np.arange(len(picks)), picks]
    r_sel = np.stack([R48[te, ci] for ci in REP_IDX], 1)[np.arange(len(picks)), picks]
    ok = np.isfinite(r_sel) & (pr_pick > 0)
    r = r_sel[ok]
    row = {"n": int(ok.sum()), "winrate": round(float((r > 0).mean()), 4),
           "meanR": round(float(r.mean()), 5),
           "medianR": round(float(np.median(r)), 5)}
    # with EV filter
    ev_pick = evm[np.arange(len(picks)), picks]
    ok2 = ok & (ev_pick > 0.05)
    r2 = r_sel[ok2]
    row["ev005"] = {"n": int(ok2.sum()), "winrate": round(float((r2 > 0).mean()), 4),
                    "meanR": round(float(r2.mean()), 5)} if ok2.sum() > 30 else None
    res["sim"][v] = row
# canonical fixed-cell reference
rc = R48[te, [REP_IDX.index(CANON) if CANON in REP_IDX else 0][0]]
res["sim"]["canonical_cell"] = {"n": int(np.isfinite(rc).sum()),
    "winrate": round(float((rc[np.isfinite(rc)] > 0).mean()), 4),
    "meanR": round(float(np.nanmean(rc)), 5)}

# ---- paired block-bootstrap CI on canonical cell: T4/T2/T3b vs T0 ----
ci = CANON if CANON in REP_IDX else REP_IDX[0]
y_te = W48_real[te, ci]; wk = week[te]
res["ci"] = {}
for a in ("T4_prior_in", "T2_smoothed", "T3b_mix50"):
    d0 = (preds[a][ci][0] - y_te) ** 2 - (preds["T0_cohort"][ci][0] - y_te) ** 2
    m, lo, hi, pg = block_boot_ci(d0, wk)
    res["ci"][f"{a}_vs_T0"] = {"delta_brier": round(m, 5), "lo90": round(lo, 5),
                               "hi90": round(hi, 5), "p_better": round(pg, 4)}
    d1 = (preds[a][ci][0] - y_te) ** 2 - (preds["T1_realized"][ci][0] - y_te) ** 2
    m1, lo1, hi1, pg1 = block_boot_ci(d1, wk)
    res["ci"][f"{a}_vs_T1"] = {"delta_brier": round(m1, 5), "lo90": round(lo1, 5),
                               "hi90": round(hi1, 5), "p_better": round(pg1, 4)}

logd("===== EXP2 SUMMARY (test 2023-25, realized outcomes) =====")
for v in VARIANTS + ["cohort_direct"]:
    logd(f"{v:14s} mean test-Brier={res[f'mean_{v}_test_brier']} "
         f"AUC={res[f'mean_{v}_test_auc']} | sim: meanR={res['sim'][v]['meanR']} "
         f"win={res['sim'][v]['winrate']} n={res['sim'][v]['n']}"
         + (f" | EV>0.05: meanR={res['sim'][v]['ev005']['meanR']} n={res['sim'][v]['ev005']['n']}"
            if res['sim'][v].get('ev005') else ""))
logd(f"base_rate       mean test-Brier={res['mean_base_test_brier']}")
logd(f"canonical fixed cell: meanR={res['sim']['canonical_cell']['meanR']}")
logd(f"CI: {json.dumps(res['ci'])}")
save_json(res, os.path.join(OUT, "exp2_targets.json"))
