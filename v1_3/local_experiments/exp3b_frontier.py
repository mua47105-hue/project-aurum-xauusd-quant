# ============================================================================
# EXP3b - Era-invariance, round 2: the transform frontier.
#
# Round-1 lesson: naive trailing rank/adaptive-scale made adversarial AUC
# WORSE (0.8048 -> 0.8688): in a trending era, momentum features sit
# persistently high in their local window, so ranks convert trend
# PERSISTENCE into an era level. Stronger menu:
#   M1 qmap      trailing-window quantile mapped through TRAIN-era inverse CDF
#                (marginals era-matched by construction)
#   M2 rankdiff  D24 of the trailing rank (kills persistence signature,
#                keeps local dynamics)
#   M3 qmap+drop qmap plus dropping the top-K joint drifters
#   M4 raw+drop  no transform, drop top-K joint drifters (reference)
# Plus: macro innovations via curl-fallback FRED; within-era reference AUC
# for every arm (is there EXTRA drift beyond natural non-stationarity?);
# signal preservation (T5_shallow realized-win) on the best arm.
# ============================================================================
import numpy as np, pandas as pd, sys, os, io, json, time, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import (load_all, year_series, brier, auc, xgb_clf,
                          REP_IDX, REP_CELLS, CANON, logd, save_json)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
ART = "/home/z/my-project/kaggle_work/aurum/p1-data/out"

RAW31 = ["ret1_atr","ret4_atr","ret12_atr","ret24_atr","ret48_atr","atr_ratio",
         "vol_z","rsi_z","stoch_z","macd_atr","bb_pos","bb_w_z","d20","d55",
         "body1","uwick1","lwick1","body2","uwick2","lwick2","body3","uwick3",
         "lwick3","run","volm_z","m1_shape","h1_shape","x_xagusd","x_eurusd",
         "x_usdjpy","m_vix"]

d = load_all()
feat = d["feat"]; ts = d["ts"]; yr = year_series(ts).values
label_valid = d["label_valid"]
msk = label_valid & np.isfinite(feat[feat.columns].values).all(1)
y_adv = (yr >= 2023).astype(int)
is_train_era = yr <= 2022

def adv_auc(X, y, seed=3, want_imp=False):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    aucs, imps = [], []
    for tr_i, te_i in skf.split(X, y):
        clf = xgb_clf(seed=seed)
        clf.fit(X[tr_i], y[tr_i])
        aucs.append(roc_auc_score(y[te_i], clf.predict_proba(X[te_i])[:, 1]))
        imps.append(clf.feature_importances_)
    r = {"auc": round(float(np.mean(aucs)), 4), "sd": round(float(np.std(aucs)), 4)}
    if want_imp:
        r["imp"] = np.mean(imps, 0)
    return r

res = {"round1_naive_rank": 0.8688, "arms": {}}
yr_m = yr[msk]; y_adv_m = y_adv[msk]

# ---------------- transforms ----------------
W = 6240
def rank_pct(s):
    return s.rolling(W, min_periods=W // 4).rank(pct=True)

TRAIN_END = "2022-12-31"
def qmap_to_train(s):
    """Trailing-window rank mapped through the TRAIN-era empirical inverse CDF:
    marginal distribution matches the train era by construction."""
    r = rank_pct(s)
    tr_vals = s[is_train_era].dropna().values
    tr_sorted = np.sort(tr_vals)
    def inv_cdf(q):
        if not np.isfinite(q):
            return np.nan
        pos = np.clip(q * (len(tr_sorted) - 1), 0, len(tr_sorted) - 1)
        lo = int(np.floor(pos)); hi = min(lo + 1, len(tr_sorted) - 1)
        w = pos - lo
        return tr_sorted[lo] * (1 - w) + tr_sorted[hi] * w
    return r.map(inv_cdf) if hasattr(r, "map") else r.apply(inv_cdf)

t0 = time.time()
RANKS = {c: rank_pct(feat[c]) for c in RAW31 if c not in ("m1_shape", "h1_shape")}
QMAP = {c: qmap_to_train(feat[c]) for c in RANKS}
logd(f"transforms built in {time.time()-t0:.0f}s")

def arm(cols_dict, name, note=""):
    X = pd.DataFrame(cols_dict).values[msk]
    mm = np.isfinite(X).all(1)
    r = adv_auc(X[mm], y_adv_m[mm], want_imp=True)
    # within-era reference: 2012-18 vs 2019-22 on the same columns
    ref_mask = mm & (yr_m <= 2022) & (yr_m >= 2012)
    y_ref = (yr_m[ref_mask] >= 2019).astype(int)
    rref = adv_auc(X[ref_mask], y_ref)
    r["rows"] = int(mm.sum()); r["within_era_ref_auc"] = rref["auc"]; r["note"] = note
    res["arms"][name] = {k: v for k, v in r.items() if k != "imp"}
    res["arms"][name]["top_drifters"] = [
        {"f": c, "imp": round(float(v), 4)}
        for c, v in sorted(zip(list(cols_dict), r["imp"]), key=lambda z: -z[1])[:6]]
    logd(f"{name:22s} cross-era AUC={r['auc']} (within-era ref {rref['auc']}) rows={mm.sum()} {note}")
    return r

# M4 raw+drop reference and M3 need drop ranking - get importances from raw31
Xr = feat[RAW31].values[msk]
r_raw = adv_auc(Xr, y_adv_m, want_imp=True)
order = np.argsort(-r_raw["imp"])
drop_rank = [RAW31[i] for i in order]

# M1: qmap all 29 continuous (keep m1/h1_shape)
M1 = {c: (QMAP[c] if c in QMAP else feat[c]) for c in RAW31}
arm(M1, "M1_qmap")

# M2: rank diff (D24) for the 29 continuous
M2 = {c: (RANKS[c].diff(24) if c in RANKS else feat[c]) for c in RAW31}
arm(M2, "M2_rankdiff24")

# M3: qmap + drop top-K drifters, K in 5, 8
for K in (5, 8):
    drop = set(drop_rank[:K])
    M3 = {c: (QMAP[c] if c in QMAP else feat[c]) for c in RAW31 if c not in drop}
    arm(M3, f"M3_qmap_drop{K}", note=f"dropped {sorted(drop)}")

# M5: rankdiff on the non-momentum subset only (keep momentum as raw qmap)
# momentum family = ret12/24/48, d20, d55, bb_pos, macd
MOMO = {"ret12_atr","ret24_atr","ret48_atr","d20","d55","bb_pos","macd_atr"}
M5 = {}
for c in RAW31:
    if c in ("m1_shape", "h1_shape"): M5[c] = feat[c]
    elif c in MOMO: M5[c] = RANKS[c].diff(24)
    else: M5[c] = QMAP[c]
arm(M5, "M5_momo_rankdiff_rest_qmap")

# ---------------- macro innovations (curl fallback) ----------------
def fred_curl(code):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
    for attempt in range(3):
        p = subprocess.run(["curl", "-sS", "--max-time", "90", "-A",
                            "Mozilla/5.0 (research)", url], capture_output=True,
                           text=True, timeout=120)
        if p.returncode == 0 and "observation_date" in p.stdout[:200]:
            df = pd.read_csv(io.StringIO(p.stdout))
            df.columns = ["date", "v"]
            df["date"] = pd.to_datetime(df["date"])
            df["v"] = pd.to_numeric(df["v"].replace(".", np.nan), errors="coerce")
            return df.set_index("date")["v"].sort_index()
        time.sleep(3)
    return None

MACRO = {"y10": "DGS10", "y2": "DGS2", "spread": "T10Y2Y", "dxy": "DTWEXBGS", "vix": "VIXCLS"}
hours = pd.Series(ts).dt.normalize().values
mi = {}
for k, code in MACRO.items():
    lev = fred_curl(code)
    if lev is None:
        logd(f"FRED {code}: FAILED even via curl"); continue
    d20 = lev - lev.shift(20)
    z20 = (d20 - d20.rolling(60, min_periods=20).mean()) / (d20.rolling(60, min_periods=20).std() + 1e-9)
    d5 = lev - lev.shift(5)
    z5 = (d5 - d5.rolling(60, min_periods=20).mean()) / (d5.rolling(60, min_periods=20).std() + 1e-9)
    rank60 = lev.rolling(60, min_periods=20).rank(pct=True)
    for stat, s in (("d20z", z20), ("d5z", z5), ("rank60", rank60)):
        col = f"mi_{k}_{stat}"
        s = s.copy(); s.index = s.index.tz_localize("UTC")
        mi[col] = s.shift(1).reindex(hours).ffill(limit=10).values
    logd(f"FRED {code}: OK {len(lev)} rows")
res["macro_available"] = sorted(mi.keys())

# macro-only audit (era-invariance of the recovered panel)
if mi:
    Mi = pd.DataFrame(mi, index=feat.index)
    Xmi = Mi.values[msk]; mm = np.isfinite(Xmi).all(1)
    rmi = adv_auc(Xmi[mm], y_adv_m[mm])
    res["macro_only_audit"] = rmi
    logd(f"macro innovations only: cross-era AUC={rmi['auc']} (target <=0.60 for the panel)")

    # single-feature drift of macro innovations
    ind = []
    for c in Mi.columns:
        col_m = np.isfinite(Mi[c].values)
        x = Mi[c].values[msk & col_m]
        from sklearn.metrics import roc_auc_score as ras
        try:
            a = ras(y_adv[msk & col_m], x)
            ind.append({"f": c, "auc": round(float(a), 4)})
        except Exception: pass
    ind.sort(key=lambda z: -abs(z["auc"] - 0.5))
    res["macro_single_feature"] = ind[:8]
    logd(f"worst single macro drift: {ind[:3]}")

    # best price arm + macro: the full recovered panel
    best_arm = min(res["arms"].items(), key=lambda kv: abs(kv[1]["auc"]))
    best_cols = M5 if best_arm[0].startswith("M5") else (M3 if "drop8" in best_arm[0] else M1)
    FM = dict(best_cols); FM.update({c: Mi[c] for c in Mi.columns})
    arm(FM, "BEST_plus_macro", note=f"base={best_arm[0]}")

save_json(res, os.path.join(OUT, "exp3b_frontier.json"))
logd("exp3b done")
