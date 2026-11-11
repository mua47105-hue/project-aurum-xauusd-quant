# ============================================================================
# EXP2b - Full-period walk-forward: is realized-target EV-argmax real ALPHA
#         or 2023-25 gold bull BETA? (the exp2 caveat)
# Expanding-window yearly refits 2016..2025, T0_cohort vs T5_shallow vs
# references (cohort-direct k-NN strategy, canonical fixed long cell).
# ============================================================================
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, auc,
                          REP_IDX, REP_CELLS, CANON, logd, save_json)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"

def xgb_reg(seed=3, depth=5, n_est=300):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", n_jobs=2, verbosity=0,
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
X = feat[FEATS].values[mask]
P48 = pw.values[mask]; B48 = be.values[mask]
W48 = (R.values[mask] > 0).astype(float); R48 = R.values[mask]
y = yr[mask]
res = {"years": {}, "protocol": "expanding window: train year<=y-1, eval year y; "
       "EV-argmax over 8 rep cells, p>0 filter; T0=cohort target, T5=shallow realized"}

for yv in range(2016, 2026):
    tr = y <= yv - 1; te = y == yv
    if te.sum() < 500:
        continue
    row = {"n_test": int(te.sum())}
    for v in ("T0_cohort", "T5_shallow"):
        if v == "T0_cohort":
            P = np.zeros((te.sum(), len(REP_IDX)))
            for j, ci in enumerate(REP_IDX):
                clf = xgb_reg()
                clf.fit(X[tr], P48[tr, ci])
                P[:, j] = np.clip(clf.predict(X[te]), 0, 1)
        else:
            P = np.zeros((te.sum(), len(REP_IDX)))
            for j, ci in enumerate(REP_IDX):
                clf = xgb_reg(depth=3, n_est=150)
                clf.fit(X[tr], W48[tr, ci])
                P[:, j] = np.clip(clf.predict(X[te]), 0, 1)
        B = np.stack([B48[te, ci] for ci in REP_IDX], 1)
        B = np.where(np.isfinite(B), B, 1.0)
        evm = P * B - (1 - P)
        picks = np.argmax(evm, 1)
        pr = P[np.arange(len(picks)), picks]
        rr = np.stack([R48[te, ci] for ci in REP_IDX], 1)[np.arange(len(picks)), picks]
        ok = np.isfinite(rr) & (pr > 0)
        r = rr[ok]
        row[v] = {"meanR": round(float(r.mean()), 5), "win": round(float((r > 0).mean()), 4),
                  "n": int(ok.sum())}
    # references
    cd = np.stack([P48[te, ci] for ci in REP_IDX], 1)
    B = np.stack([B48[te, ci] for ci in REP_IDX], 1); B = np.where(np.isfinite(B), B, 1.0)
    evc = cd * B - (1 - cd)
    pk = np.argmax(evc, 1); prc = cd[np.arange(len(pk)), pk]
    rrc = np.stack([R48[te, ci] for ci in REP_IDX], 1)[np.arange(len(pk)), pk]
    okc = np.isfinite(rrc) & (prc > 0)
    row["cohort_direct"] = {"meanR": round(float(rrc[okc].mean()), 5),
                            "win": round(float((rrc[okc] > 0).mean()), 4), "n": int(okc.sum())}
    rc = R48[te, REP_IDX.index(CANON)]
    row["canonical_long"] = {"meanR": round(float(np.nanmean(rc)), 5),
                             "win": round(float((rc[np.isfinite(rc)] > 0).mean()), 4),
                             "n": int(np.isfinite(rc).sum())}
    res["years"][str(yv)] = row
    logd(f"{yv}: T0={row['T0_cohort']['meanR']:+.4f} T5={row['T5_shallow']['meanR']:+.4f} "
         f"cohort_direct={row['cohort_direct']['meanR']:+.4f} canon_long={row['canonical_long']['meanR']:+.4f} "
         f"(n={row['n_test']})")

# pooled (equal weight per year)
pooled = {}
for v in ("T0_cohort", "T5_shallow", "cohort_direct", "canonical_long"):
    vals = [res["years"][yy][v]["meanR"] for yy in res["years"]]
    pooled[v] = {"mean_yearly_R": round(float(np.mean(vals)), 5),
                 "pos_years": int(np.sum(np.array(vals) > 0)),
                 "n_years": len(vals)}
res["pooled"] = pooled
logd("POOLED: " + json.dumps(pooled))
save_json(res, os.path.join(OUT, "exp2b_walkforward.json"))
