# ============================================================================
# EXP3 - Era-Invariance: fix G1.4 properly (adversarial AUC 0.8055 -> <=0.60)
#        AND recover the macro panel that was thrown away.
#
# Arms (all evaluated with the EXACT Kaggle audit protocol:
#       XGB d5x300 5-fold StratifiedKFold rs3, y = year>=2023, audit rows):
#   A_raw39        39 raw features (expect AUC ~1.0, replication)
#   A_raw31        the 31 post-drop features (expect ~0.8055, replication)
#   T31_rank       31 price features, era-invariant transforms (rank/scale)
#   T35_macro      T31 + macro innovation features (y10/y2/spread/dxy d20z,
#                  vix rank60/d5z) - the RECOVERED panel
# Then: signal-preservation check (realized-win GBDT on transformed vs raw),
#       importance-weighting arm (density-ratio reweighting of the signal
#       model), per-feature residual drift report.
# ============================================================================
import numpy as np, pandas as pd, sys, os, io, json, time, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, auc, xgb_clf,
                          REP_IDX, REP_CELLS, CANON, logd, save_json)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ART = "/home/z/my-project/kaggle_work/aurum/p1-data/out"

RAW31 = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_vix"]  # from audit.json final_features

d = load_all()
feat = d["feat"]; ts = d["ts"]; yr = year_series(ts).values
label_valid = d["label_valid"]
R = d["R"]

# ---------------- audit row mask (kernel Stage F exact) ----------------
allcols = [c for c in feat.columns]
msk = label_valid & np.isfinite(feat[allcols].values).all(1)
logd(f"audit rows={msk.sum()} (Kaggle gate ran on the same mask)")
y_adv = (yr >= 2023).astype(int)

def adv_auc(X, y, seed=3):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs, imps = [], []
    for tr_i, te_i in skf.split(X, y):
        clf = xgb_clf(seed=seed)
        clf.fit(X[tr_i], y[tr_i])
        aucs.append(roc_auc_score(y[te_i], clf.predict_proba(X[te_i])[:, 1]))
        imps.append(clf.feature_importances_)
    return float(np.mean(aucs)), float(np.std(aucs)), np.mean(imps, 0)

res = {"audit_rows": int(msk.sum())}

# ---- A_raw39 / A_raw31 replication ----
t0 = time.time()
auc39, sd39, imp39 = adv_auc(feat[allcols].values[msk], y_adv[msk])
res["A_raw39"] = round(auc39, 4)
logd(f"A_raw39 AUC={auc39:.4f} ±{sd39:.4f}  ({time.time()-t0:.0f}s)  [Kaggle: 1.0000]")
auc31, sd31, imp31 = adv_auc(feat[RAW31].values[msk], y_adv[msk])
res["A_raw31"] = round(auc31, 4)
logd(f"A_raw31 AUC={auc31:.4f} ±{sd31:.4f}  ({time.time()-t0:.0f}s)  [Kaggle: 0.8055]")

# ---------------- era-invariant transforms ----------------
def trailing_rank(s, W):
    # pandas C-impl rolling rank (pct) - vectorized, unlike .apply
    return s.rolling(W, min_periods=W // 4).rank(pct=True)

def adaptive_scale(s, W):
    q = s.abs().rolling(W, min_periods=W // 4).quantile(0.9)
    return s / (q + 1e-12)

W = 6240  # ~1 trading year of H1 bars
T = pd.DataFrame(index=feat.index)
for c in RAW31:
    s = feat[c]
    if c in ("m1_shape", "h1_shape"):
        T[c] = s
    elif c in ("ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr",
               "macd_atr","body1","body2","body3","uwick1","uwick2","uwick3",
               "lwick1","lwick2","lwick3","x_xagusd","x_eurusd","x_usdjpy"):
        T[c] = adaptive_scale(s, W)
    else:  # bounded/slow: rank
        T[c] = trailing_rank(s, W)
# note: m_vix replaced below by innovation features; drop raw m_vix from T31
T31_cols = [c for c in RAW31 if c != "m_vix"]

# ---------------- macro innovations from raw FRED ----------------
UA = {"User-Agent": "Mozilla/5.0 (research; local verification)"}
def fred(code):
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}",
                     headers=UA, timeout=60)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    df["v"] = pd.to_numeric(df["v"].replace(".", np.nan), errors="coerce")
    return df.set_index("date")["v"].sort_index()

def macro_feats(code):
    lev = fred(code)
    d20 = lev - lev.shift(20)
    z20 = (d20 - d20.rolling(60, min_periods=20).mean()) / (d20.rolling(60, min_periods=20).std() + 1e-9)
    d5 = lev - lev.shift(5)
    z5 = (d5 - d5.rolling(60, min_periods=20).mean()) / (d5.rolling(60, min_periods=20).std() + 1e-9)
    rank60 = lev.rolling(60, min_periods=20).rank(pct=True)
    out = pd.DataFrame({"d20z": z20, "d5z": z5, "rank60": rank60})
    out.index = out.index.tz_localize("UTC")
    return out.shift(1)  # yesterday's value intraday (kernel parity)

MACRO = {"y10": "DGS10", "y2": "DGS2", "spread": "T10Y2Y", "dxy": "DTWEXBGS", "vix": "VIXCLS"}
M = {}
for k, code in MACRO.items():
    try:
        M[k] = macro_feats(code)
        logd(f"FRED {code}: OK {len(M[k])} rows")
    except Exception as e:
        logd(f"FRED {code}: FAIL {e}")
        M[k] = None

hours = pd.Series(ts).dt.normalize()
hour_map = pd.Series(ts.values, index=ts)
Mcols = []
for k in ("y10", "y2", "spread", "dxy"):
    if M[k] is not None:
        for stat in ("d20z", "rank60"):
            s = M[k][stat]
            col = f"mi_{k}_{stat}"
            T[col] = s.reindex(hours.values).ffill(limit=10).values
            Mcols.append(col)
if M["vix"] is not None:
    for stat in ("rank60", "d5z"):
        s = M["vix"][stat]
        col = f"mi_vix_{stat}"
        T[col] = s.reindex(hours.values).ffill(limit=10).values
        Mcols.append(col)
logd(f"macro innovation cols: {Mcols}")

T35_cols = T31_cols + Mcols

# ---- transformed audits ----
Xt31 = T[T31_cols].values[msk]
mt31 = np.isfinite(Xt31).all(1)
aucT, sdT, impT = adv_auc(Xt31[mt31], y_adv[msk][mt31])
res["T31_rank"] = round(aucT, 4)
logd(f"T31_rank AUC={aucT:.4f} ±{sdT:.4f} (rows {mt31.sum()})")

Xt35 = T[T35_cols].values[msk]
mt35 = np.isfinite(Xt35).all(1)
aucM, sdM, impM = adv_auc(Xt35[mt35], y_adv[msk][mt35])
res["T35_macro"] = round(aucM, 4)
logd(f"T35_macro AUC={aucM:.4f} ±{sdM:.4f} (rows {mt35.sum()})  <- gate target <=0.60")

# per-feature residual drift on the recovered panel
indiv = []
for c in T35_cols:
    x = T[c].values[msk & np.isfinite(T[c].values)]
    yy = y_adv[msk & np.isfinite(T[c].values)]
    if len(np.unique(x)) > 10:
        from sklearn.metrics import roc_auc_score as ras
        a = ras(yy, x)
        indiv.append({"feature": c, "auc": round(float(a), 4), "dev": round(abs(a - 0.5), 4)})
indiv.sort(key=lambda z: -z["dev"])
res["residual_single_feature_drift"] = indiv[:10]

save_json(res, os.path.join(OUT, "exp3_era_audit.json"))
logd("exp3 part 1 done (audit arms). Running signal-preservation next.")
