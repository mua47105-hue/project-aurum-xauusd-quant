# ============================================================================
# EXP3c - Final era-invariant set: macro recovery + signal preservation +
#         importance weighting.
# Final set = M3_qmap_drop8 (round-2 winner) + macro innovations
# (z-diff approximation locally; the kernel fetches raw FRED levels on
# Kaggle and builds exact d20z/rank60 there).
# ============================================================================
import numpy as np, pandas as pd, sys, os, json, time, subprocess, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, auc, xgb_clf,
                          REP_IDX, REP_CELLS, CANON, logd, save_json)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"

RAW31 = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_vix"]
DROP8 = ["atr_ratio","bb_pos","d20","d55","m_vix","ret48_atr","rsi_z","stoch_z"]

d = load_all()
feat = d["feat"]; ts = d["ts"]; yr = year_series(ts).values
R, pw, be = d["R"], d["pw"], d["be"]
label_valid, cohort_valid = d["label_valid"], d["cohort_valid"]
msk = label_valid & np.isfinite(feat[feat.columns].values).all(1)
y_adv = (yr >= 2023).astype(int)
is_train_era = yr <= 2022
yr_m = yr[msk]; y_adv_m = y_adv[msk]

W = 6240
def rank_pct(s): return s.rolling(W, min_periods=W // 4).rank(pct=True)
def qmap_to_train(s):
    r = rank_pct(s).values
    tr_sorted = np.sort(s[is_train_era].dropna().values)
    n = len(tr_sorted)
    out = np.full(len(r), np.nan)
    ok = np.isfinite(r)
    pos = np.clip(r[ok] * (n - 1), 0, n - 1)
    lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, n - 1)
    w = pos - lo
    out[ok] = tr_sorted[lo] * (1 - w) + tr_sorted[hi] * w
    return pd.Series(out, index=s.index)

KEEP = [c for c in RAW31 if c not in DROP8]
FIN = pd.DataFrame(index=feat.index)
for c in KEEP:
    FIN[c] = feat[c] if c in ("m1_shape", "h1_shape") else qmap_to_train(feat[c])

# macro innovations: z-diff approximation of d20z (20 days = 480 H1 bars;
# weekend-aware approx: 20*24=480 calendar-hours ~ 336 trading bars... use
# 480 to stay calendar-aligned like FRED daily shift(20))
mi = {}
for src, name in (("m_y10", "mi_y10_d20z"), ("m_y2", "mi_y2_d20z"),
                  ("m_spread", "mi_spread_d20z"), ("m_dxy", "mi_dxy_d20z")):
    mi[name] = feat[src] - feat[src].shift(480)
mi["mi_vix_d5z"] = feat["m_vix"] - feat["m_vix"].shift(120)
for k, v in mi.items():
    FIN[k] = v
FINAL_COLS = KEEP + list(mi.keys())
logd(f"final set: {len(FINAL_COLS)} cols = qmap({len(KEEP)-2}) + shapes(2) + macro-innov({len(mi)})")

def adv_auc(X, y, seed=3, oof=False):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs, oof_p = [], np.zeros(len(y))
    for tr_i, te_i in skf.split(X, y):
        clf = xgb_clf(seed=seed)
        clf.fit(X[tr_i], y[tr_i])
        p = clf.predict_proba(X[te_i])[:, 1]
        aucs.append(roc_auc_score(y[te_i], p)); oof_p[te_i] = p
    r = {"auc": round(float(np.mean(aucs)), 4), "sd": round(float(np.std(aucs)), 4)}
    if oof: return r, oof_p
    return r

res = {"final_cols": FINAL_COLS}
Xfin = FIN[FINAL_COLS].values[msk]
mm = np.isfinite(Xfin).all(1)
r_fin, _ = adv_auc(Xfin[mm], y_adv_m[mm], oof=True)
r_ref = adv_auc(Xfin[mm & (yr_m <= 2022) & (yr_m >= 2012)],
                (yr_m[mm & (yr_m <= 2022) & (yr_m >= 2012)] >= 2019).astype(int))
res["final_set_audit"] = {**r_fin, "within_era_ref": r_ref["auc"]}
logd(f"FINAL set cross-era AUC={r_fin['auc']} (within-era ref {r_ref['auc']})")

# macro-only
mcols = list(mi.keys())
Xm = FIN[mcols].values[msk]
mmm = np.isfinite(Xm).all(1)
res["macro_innov_only"] = adv_auc(Xm[mmm], y_adv_m[mmm])
from sklearn.metrics import roc_auc_score as ras
ind = []
for c in mcols:
    cm = np.isfinite(FIN[c].values)
    ind.append({"f": c, "auc": round(float(ras(y_adv[msk & cm], FIN[c].values[msk & cm])), 4)})
res["macro_single"] = ind
logd(f"macro innovations only: AUC={res['macro_innov_only']['auc']} singles={ind}")

# ---------------- signal preservation ----------------
def xgb_reg(seed=3, depth=5, n_est=300):
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=depth, n_estimators=n_est, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        objective="reg:squarederror", n_jobs=2, verbosity=0,
                        random_state=seed)

cmask = cohort_valid & mm_full if (mm_full := np.isfinite(FIN[FINAL_COLS].values).all(1)) is not None else None
cmask = cohort_valid & mm_full
tr = yr[cmask] <= 2020; te = yr[cmask] >= 2023
FEATS35 = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
           "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
           "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
           "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
           "x_usdjpy","m_dxy","m_y10","m_y2","m_spread","m_vix"]
sig = {}
for tag, Xfull in (("raw35", feat[FEATS35]),
                   ("final_era_inv", FIN[FINAL_COLS])):
    for ci, cell in [(REP_IDX[i], REP_CELLS[i]) for i in (0, 4, 1)]:  # canon L + canon S + wide L
        t_real = (R.values[:, ci] > 0).astype(float)
        rows = cmask & np.isfinite(t_real)
        X_ = Xfull.values[rows]
        mtr = yr[rows] <= 2020; mte = yr[rows] >= 2023
        clf = xgb_reg(depth=3, n_est=150)
        clf.fit(X_[mtr], t_real[rows][mtr])
        p = np.clip(clf.predict(X_[mte]), 0, 1)
        y_ = t_real[rows][mte]
        sig[f"{tag}:{cell[0]}_m{cell[1]}_t{cell[2]}_h{cell[3]}"] = {
            "test_brier": round(brier(p, y_), 5), "test_auc": round(auc(p, y_), 5)}
res["signal_preservation"] = sig
logd("signal preservation: " + json.dumps(sig))

# ---------------- importance weighting (density-ratio reweighting) ----------------
# OOF adversarial probs on RAW35 -> weights w = n_tr/n_te * p/(1-p), clipped
Xr35 = feat[FEATS35].values[msk]
rw = adv_auc(Xr35, y_adv_m, oof=True)
_, oof_p = rw
w = np.clip(oof_p / np.clip(1 - oof_p, 1e-6, None) * (y_adv_m == 0).sum() / max((y_adv_m == 1).sum(), 1),
            0.25, 4.0)
res["iw_clip_rate"] = float(((w < 0.25) | (w > 4.0)).mean())
ci = CANON
t_real = (R.values[:, ci] > 0).astype(float)
rows = cohort_valid & np.isfinite(t_real)
X_ = feat[FEATS35].values[rows]
w_full = np.ones(len(feat))
w_full[msk] = w
w_ = w_full[rows]
mtr = yr[rows] <= 2020; mte = yr[rows] >= 2023
clf_w = xgb_reg(depth=3, n_est=150)
clf_w.fit(X_[mtr], t_real[rows][mtr], sample_weight=w_[mtr])
p_w = np.clip(clf_w.predict(X_[mte]), 0, 1)
clf_u = xgb_reg(depth=3, n_est=150)
clf_u.fit(X_[mtr], t_real[rows][mtr])
p_u = np.clip(clf_u.predict(X_[mte]), 0, 1)
y_ = t_real[rows][mte]
res["importance_weighting"] = {
    "unweighted": {"brier": round(brier(p_u, y_), 5), "auc": round(auc(p_u, y_), 5)},
    "weighted": {"brier": round(brier(p_w, y_), 5), "auc": round(auc(p_w, y_), 5)},
    "note": "weights from OOF adversarial density ratio on raw35, clip [0.25,4]"}
logd(f"importance weighting: {res['importance_weighting']}")

save_json(res, os.path.join(OUT, "exp3c_final.json"))
logd("exp3c done")
