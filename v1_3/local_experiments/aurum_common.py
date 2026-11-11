# ============================================================================
# AURUM local verification commons (post-mortem pass)
# Loads the pulled aurum-p1-data artifacts, reconstructs the exact label_valid
# mask used on Kaggle, provides splits / metrics / smoothing kernels shared by
# exp1..exp4. Pure CPU, <=3GB RAM safe.
# ============================================================================
import os, sys, json, time
import numpy as np
import pandas as pd

ART = "/home/z/my-project/kaggle_work/aurum/p1-data/out"
KERNEL_SRC = "/home/z/my-project/kaggle_work/aurum/p1-data/src/aurum-p1-data.py"

# ---------------- cell geometry (mirror of the kernel) ----------------
BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]
BARRIER_TPS   = [1.5, 2.5, 4.0]
BARRIER_HORS  = [24, 48]
DIRS = ("L", "S")

def cell_axes():
    dirs, ms, ts, hs = [], [], [], []
    for d in DIRS:
        for m in BARRIER_STOPS:
            for t in BARRIER_TPS:
                for h in BARRIER_HORS:
                    dirs.append(d); ms.append(m); ts.append(t); hs.append(h)
    return dirs, ms, ts, hs

def cell_names():
    dirs, ms, ts, hs = cell_axes()
    return [f"{d}_m{m}_t{t}_h{h}" for d, m, t, h in zip(dirs, ms, ts, hs)]

def cell_index(d, m, t, h):
    di = 0 if d == "L" else 1
    mi = BARRIER_STOPS.index(m); ti = BARRIER_TPS.index(t); hi = BARRIER_HORS.index(h)
    return di * 24 + mi * 6 + ti * 2 + hi

CANON = cell_index("L", 1.5, 2.5, 48)     # 9, matches pilot canon

# representative cells for per-cell experiments
REP_CELLS = [
    ("L", 1.5, 2.5, 48), ("L", 1.0, 1.5, 48), ("L", 2.0, 4.0, 48), ("L", 1.5, 2.5, 24),
    ("S", 1.5, 2.5, 48), ("S", 1.0, 1.5, 48), ("S", 2.0, 4.0, 48), ("S", 1.5, 2.5, 24),
]
REP_IDX = [cell_index(*c) for c in REP_CELLS]

COHORT_FEATURES = [
    "ret1_atr", "ret4_atr", "ret12_atr", "ret24_atr", "ret48_atr",
    "atr_ratio", "vol_z", "rsi_z", "stoch_z", "macd_atr", "bb_pos", "bb_w_z",
    "d20", "d55", "body1", "uwick1", "lwick1", "run", "volm_z",
    "m1_shape", "h1_shape",
    "x_xagusd", "x_eurusd", "x_usdjpy", "m_dxy", "m_y10", "m_vix",
]  # 27 cols, mirrors kernel COHORT_FEATURES (order irrelevant)

# ---------------- data loading ----------------
_cache = {}

def load_all():
    if _cache:
        return _cache
    feat = pd.read_parquet(os.path.join(ART, "h1_features.parquet"))
    feat = feat.drop(columns=[c for c in feat.columns if c == "ts"])
    R   = pd.read_parquet(os.path.join(ART, "labels_R_actual.parquet"))
    pw  = pd.read_parquet(os.path.join(ART, "labels_pwin.parquet"))
    be  = pd.read_parquet(os.path.join(ART, "labels_beff.parquet"))
    cls = pd.read_parquet(os.path.join(ART, "labels_cls_actual.parquet"))
    # exact label_valid reconstruction (kernel Stage E):
    valid_R = np.isfinite(R.values).all(1) & (cls.values != 255).all(1)
    Fm = feat[COHORT_FEATURES]
    label_valid = valid_R.copy()
    label_valid &= Fm.notna().all(axis=1).values
    label_valid &= np.isfinite(Fm.values).all(1)
    label_valid[:400] = False
    cohort_valid = np.isfinite(pw.values).all(1) & label_valid
    _cache.update(dict(feat=feat, R=R, pw=pw, be=be, cls=cls, ts=feat.index,
                       valid_R=valid_R, label_valid=label_valid,
                       cohort_valid=cohort_valid))
    return _cache

def year_series(idx):
    return pd.Series(idx.year, index=idx)

# ---------------- XGB config (mirrors the Kaggle audit) ----------------
def xgb_clf(seed=3, n_jobs=2):
    from xgboost import XGBClassifier
    return XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.1,
                         subsample=0.8, colsample_bytree=0.8,
                         eval_metric="logloss", n_jobs=n_jobs, verbosity=0,
                         random_state=seed)

# ---------------- metrics ----------------
def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))

def logloss(p, y, eps=1e-7):
    p = np.clip(np.asarray(p, float), eps, 1 - eps); y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def auc(p, y):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(np.asarray(y), np.asarray(p)))
    except Exception:
        return float("nan")

def block_boot_ci(diff, weeks, n=1000, alpha=0.10, seed=7):
    """diff: per-row paired metric contribution (e.g. (pA-y)^2-(pB-y)^2).
    weeks: per-row week label. Returns (mean, lo, hi, p_gt0)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"d": diff, "w": weeks})
    wk = df.groupby("w")["d"].mean()
    labels = wk.index.values; vals = wk.values
    means = rng.choice(vals, size=(n, len(vals)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(vals.mean()), float(lo), float(hi), float((means > 0).mean())

# ---------------- adjacent-cell triangular smoothing ----------------
def build_smooth_kernel():
    """Triangular kernel over the 48-cell grid: own 1.0, m+-1 0.5, t+-1 0.5,
    (m+-1,t+-1) 0.25, h+-1 (same m,t) 0.5. Rows sum to 1."""
    dirs, ms, ts, hs = cell_axes()
    K = np.zeros((48, 48))
    for i in range(48):
        for j in range(48):
            if dirs[i] != dirs[j]:
                continue
            dm = abs(BARRIER_STOPS.index(ms[i]) - BARRIER_STOPS.index(ms[j]))
            dt = abs(BARRIER_TPS.index(ts[i]) - BARRIER_TPS.index(ts[j]))
            dh = abs(BARRIER_HORS.index(hs[i]) - BARRIER_HORS.index(hs[j]))
            if dh > 1:
                continue
            w = None
            if dm == 0 and dt == 0 and dh == 0: w = 1.0
            elif dm == 1 and dt == 0 and dh == 0: w = 0.5
            elif dm == 0 and dt == 1 and dh == 0: w = 0.5
            elif dm == 1 and dt == 1 and dh == 0: w = 0.25
            elif dm == 0 and dt == 0 and dh == 1: w = 0.5
            if w is not None:
                K[i, j] = w
    K = K / K.sum(axis=1, keepdims=True)
    return K

def logd(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    logd(f"saved {path}")
