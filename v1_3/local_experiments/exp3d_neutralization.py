# ============================================================================
# EXP3d - Prediction-level neutralization (Numerai-style): can the realized
# signal survive removing the era-drift direction from predictions?
# f(x) = OOF adversarial score (era fingerprint). Within each YEAR:
#   p_neut = p - cov(p,f)/var(f) * f
# If realized AUC/Brier is retained -> drift exposure is NOT the signal ->
# neutralization is the correct model-level defense (spec amendment).
# ============================================================================
import numpy as np, pandas as pd, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, auc, xgb_clf,
                          CANON, logd, save_json)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"

def xgb_reg(seed=3, depth=3, n_est=150):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", n_jobs=2, verbosity=0,
                        random_state=seed)

d = load_all()
feat, R, pw = d["feat"], d["R"], d["pw"]
ts = d["ts"]; yr = year_series(ts).values
mask = d["cohort_valid"]
FEATS = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_dxy","m_y10","m_y2","m_spread","m_vix"]
X = feat[FEATS].values[mask]
W48 = (R.values[mask] > 0).astype(float)
y = yr[mask]
tr = y <= 2020; te = y >= 2023

# OOF adversarial score on the TRAIN+TEST rows (5-fold over all mask rows)
from sklearn.model_selection import StratifiedKFold
y_adv = (y >= 2023).astype(int)
f = np.zeros(len(y))
skf = StratifiedKFold(5, shuffle=True, random_state=3)
for tr_i, te_i in skf.split(X, y_adv):
    clf = xgb_clf()
    clf.fit(X[tr_i], y_adv[tr_i])
    f[te_i] = clf.predict_proba(X[te_i])[:, 1]

res = {}
for ci, nm in ((CANON, "canon_L"), (CANON + 24, "canon_S")):
    clf = xgb_reg()
    clf.fit(X[tr], W48[tr, ci])
    p = np.clip(clf.predict(X[te]), 0, 1)
    y_te = W48[te, ci]; f_te = f[te]; yr_te = y[te]
    # neutralize within each year (era-local, no cross-era leakage)
    p_neut = p.copy()
    for yv in np.unique(yr_te):
        m = yr_te == yv
        b = np.cov(p[m], f_te[m])[0, 1] / (np.var(f_te[m]) + 1e-12)
        p_neut[m] = p[m] - b * (f_te[m] - f_te[m].mean())
    p_neut = np.clip(p_neut, 0, 1)
    res[nm] = {
        "raw": {"brier": round(brier(p, y_te), 5), "auc": round(auc(p, y_te), 5)},
        "neutralized": {"brier": round(brier(p_neut, y_te), 5),
                        "auc": round(auc(p_neut, y_te), 5)},
        "corr_p_f": round(float(np.corrcoef(p, f_te)[0, 1]), 4)}
    logd(f"{nm}: raw AUC={res[nm]['raw']['auc']} -> neutralized "
         f"AUC={res[nm]['neutralized']['auc']} (corr p,f = {res[nm]['corr_p_f']})")

save_json(res, os.path.join(OUT, "exp3d_neutralization.json"))
logd("exp3d done")
