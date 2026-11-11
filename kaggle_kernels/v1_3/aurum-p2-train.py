# ============================================================================
# Auto-generated AURUM kernel: lib/* + kernels/p2_train.py  (built 2026-11-08 06:26)
# Spec v1.1 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
import os as _os, sys as _sys
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _pin_transformers():
    """Qwen3 needs transformers>=4.51; 5.x changed model APIs and the Kaggle
    image ships a partially broken 5.20.2 (keras_nlp backend import error).
    Force-install 4.57.1 with --no-deps (bypasses the resolver refusing the
    downgrade), verify, and purge cached modules."""
    import subprocess as _sp
    ok = False
    try:
        import transformers
        from packaging.version import Version
        ok = Version("4.51.0") <= Version(transformers.__version__) < Version("5.0.0")
    except Exception:
        ok = False
    if ok:
        return
    try:
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                 "-q", "transformers==4.57.1"], check=True, timeout=900,
                capture_output=True)
        _sp.run([_sys.executable, "-m", "pip", "install", "--no-deps", "-q",
                 "tokenizers==0.22.0", "huggingface-hub==0.36.2"],
                check=False, timeout=900, capture_output=True)
    except Exception as _e:
        print("transformers pin install error:", _e, flush=True)
    for _mod in list(_sys.modules):
        if _mod.startswith(("transformers", "tokenizers", "huggingface_hub")):
            del _sys.modules[_mod]
    import transformers
    from packaging.version import Version as _V
    _v = Version(transformers.__version__)
    assert _V("4.51.0") <= _v < _V("5.0.0"), (
        f"transformers pin FAILED, got {_v}; Qwen3 layer API unverified on this version")

_pin_transformers()


# ==================== lib/aurum_gates.py ====================
# ============================================================================
# AURUM shared helpers: gates, logging, env report
# Spec v1.1 (frozen). All quantity-producing code paths must be dimensionless
# (ATR units / log-returns / z-scores / ratios / codebook ids) - see audit #1.
# ============================================================================
import json, os, sys, time, traceback, random
import numpy as np

GATES = {}
LOG_LINES = []

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

def gate(gid, ok, detail=None, extra=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2000]}
    if extra:
        g.update(extra)
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return ok

def save_gates(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(os.path.dirname(path), "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))

def set_seeds(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def env_report():
    import platform
    info = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "cwd": os.getcwd(),
        "argv": sys.argv,
    }
    try:
        import pandas as pd
        info["pandas"] = pd.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["capability"] = str(torch.cuda.get_device_capability(0))
            p = torch.cuda.get_device_properties(0)
            info["total_mem_gb"] = round(p.total_memory / 1e9, 2)
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = "missing"
    return info

def is_kaggle():
    return os.path.exists("/kaggle/working")

def stage(name):
    """Decorator-ish context: log stage start/end, capture exceptions."""
    class C:
        def __init__(self):
            self.t0 = time.time()
            self.ok = None
            self.err = None
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            self.ok = et is None
            if et is not None:
                self.err = "".join(traceback.format_exception(et, ev, tb))[-3000:]
                log(f"===== STAGE {name} ERROR =====\n{self.err}")
            else:
                log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False  # do not suppress
    return C()

# Frozen architecture constants (Spec v1.1) ---------------------------------
BARRIER_STOPS = [1.0, 1.5, 2.0, 2.5]      # x ATR14(H1)
BARRIER_TPS   = [1.5, 2.5, 4.0]           # x ATR14(H1)
BARRIER_HORS  = [24, 48]                  # H1 bars; default vertical barrier H=48
N_CELLS       = 2 * 4 * 3 * 2             # = 48 cells (dir x stop x tp x horizon)
KELLY_FRACTION = 0.25                     # quarter Kelly
HARD_EQUITY_RISK_CAP = 0.01               # 1.0% equity risk per trade
COHORT_K = 200                            # B=200 bootstrap cohorts
COHORT_GAP = 96                           # neighbour must end >=96 bars before target bar
READOUT_NAMES = ["H1_BARRIER", "H2_MTM", "H3_EXCUR", "H4_VOL", "H5_REGIME"]
HEAD_DIMS = {"h1": 48, "h2": 8, "h3": 12, "h4": 8, "h5": 9}  # 85 outputs total

def cell_axes():
    """Flat 48-cell layout: for d in (long,short) for m in stops for t in tps for h in hors."""
    dirs, mss, tps, hrs = [], [], [], []
    for d in ("L", "S"):
        for m in BARRIER_STOPS:
            for t in BARRIER_TPS:
                for h in BARRIER_HORS:
                    dirs.append(d); mss.append(m); tps.append(t); hrs.append(h)
    return dirs, mss, tps, hrs

def cell_names():
    dirs, mss, tps, hrs = cell_axes()
    return [f"{d}_m{m}_t{t}_h{h}" for d, m, t, h in zip(dirs, mss, tps, hrs)]


# ==================== lib/aurum_feat.py ====================
# ============================================================================
# AURUM dimensionless feature pipeline (Spec v1.2).
# EVERY feature is a function of scale-invariant transforms only:
#   log-return ratios, ATR-normalized distances, rolling z-scores (past-only),
#   range positions (0..1), codebook ids, calendar cyclical ids.
# NO absolute price, NO nominal ATR, NO absolute date, NO raw volume level.
# Cross-asset features (XAG/EUR/JPY + macro INNOVATIONS) give the sequence
# model non-tabular structure a 15-lag XGBoost cannot trivially match.
#
# v1.2 changes (Failure #1/#2 remediation):
#   - Macro panel switched from raw-level z-scores (date fingerprints,
#     G1.4 AUC 0.96-1.00) to INNOVATION transforms on raw FRED levels:
#     d20z / d5z (z-score of the daily innovation over a trailing window)
#     and rank60 (percentile rank of the level in a trailing 60-day window),
#     all with a strict 1-day publication lag (exp3c: |AUC-0.5| <= 0.03).
#   - The 4 base-rate priors enter the STATE as self-z-scores (era-safe
#     encoding; raw trailing win rates were date fingerprints at AUC 0.85-0.90).
#     Raw values stay in the parquet for the decision layer / baselines.
#   - Cohort MATCHING space is era-sanitized: raw m_dxy/m_y10 replaced by
#     their innovation forms (matched neighbors must not be era-concentrated).
# ============================================================================
import numpy as np
import pandas as pd

EPS = 1e-12

# ---------- primitives (all scale-invariant by construction) ----------------

def wilder_atr(h, l, c, n=14):
    prev_c = c.shift(1)
    tr = pd.Series(np.maximum.reduce([h - l, (h - prev_c).abs(), (l - prev_c).abs()]), index=h.index)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

def roll_z(x, n, minp=None):
    minp = max(4, n // 2) if minp is None else minp
    m = x.rolling(n, min_periods=minp).mean()
    s = x.rolling(n, min_periods=minp).std()
    return (x - m) / (s + EPS)

def logret(c, h=1):
    return np.log(c / c.shift(h))

def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / (dn + EPS))

def stoch_k(h, l, c, n=14):
    ll = l.rolling(n, min_periods=n).min()
    hh = h.rolling(n, min_periods=n).max()
    return 100 * (c - ll) / (hh - ll + EPS)

def macd_atr(c, atr, fast=12, slow=26, sig=9):
    ef = c.ewm(span=fast, adjust=False).mean()
    es = c.ewm(span=slow, adjust=False).mean()
    line = (ef - es)
    sgnl = line.ewm(span=sig, adjust=False).mean()
    return (line - sgnl) / (atr + EPS)

def session_of_hour(hour_utc):
    """Cyclical session id from UTC hour (calendar feature, not a date proxy)."""
    if 0 <= hour_utc < 7:
        return "ASIA"
    if 7 <= hour_utc < 12:
        return "LONDON"
    if 12 <= hour_utc < 16:
        return "NY"
    if 16 <= hour_utc < 21:
        return "OVL"
    return "OFF"

DOW = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# ---------- main feature builder --------------------------------------------

# ---------- macro innovation panel (Spec v1.2) --------------------------------

def macro_innovations(fred, h1_index):
    """Exact daily-domain innovation transforms on RAW FRED levels.

    For each series s in fred (y10, y2, spread, dxy, vix), computed on the
    DAILY series then mapped onto H1 bars with a STRICT 1-day lag
    (yesterday's fully-published value; FRED dailies publish during/after
      the US session - no lookahead):
        mi_{s}_d20z   = roll_z(level.diff(), 20)   - 20-day z of daily innovation
        mi_{s}_d5z    = roll_z(level.diff(), 5)    - 5-day z of daily innovation
        mi_{s}_rank60 = level.rolling(60).rank(pct=True) - 60-day percentile rank
    Weekend/holiday gaps are forward-filled up to 10 days.
    Returns DataFrame aligned to h1_index (era-invariant by construction:
    exp3c measured |AUC-0.5| <= 0.03 for the d20z/d5z family on 2012-2025)."""
    F = pd.DataFrame(index=h1_index)
    for k, s in fred.items():
        daily = pd.to_numeric(s, errors="coerce").sort_index()
        if daily.index.tz is None:
            daily.index = daily.index.tz_localize("UTC")
        innov = daily.diff()
        d20z = roll_z(innov, 20, minp=10)
        d5z = roll_z(innov, 5, minp=3)
        rank60 = daily.rolling(60, min_periods=30).rank(pct=True)
        for nm, ser in ((f"mi_{k}_d20z", d20z), (f"mi_{k}_d5z", d5z),
                        (f"mi_{k}_rank60", rank60)):
            z = ser.shift(1)                       # strict 1-day lag
            F[nm] = z.ffill(limit=10).reindex(
                h1_index.normalize()).values
    return F

# ---------- self-z-scored priors (era-safe state encoding) -------------------

PRIOR_COLS = ("prior_w24", "prior_l24", "prior_w48", "prior_l48")

def zscore_priors(priors, n=504):
    """Z-score each trailing base-rate prior against its own trailing n-bar
    distribution (past-only). The raw trailing win rate is a date fingerprint
    (G1.4 AUC 0.85-0.90 in v4); its self-z ('is the current base rate unusually
    high vs recent history?') is era-invariant by the same construction as every
    other roll_z feature. Returns DataFrame with prior_zw24/... columns."""
    out = pd.DataFrame(index=priors.index)
    for c in PRIOR_COLS:
        if c in priors.columns:
            zname = c.replace("prior_", "prior_z")
            out[zname] = roll_z(priors[c], n)
    return out

def compute_h1_features(df_h1, m1_code=None, h1_code=None, cross=None, fred=None,
                        priors=None, innov=None):
    """
    df_h1: DataFrame indexed by UTC hourly DatetimeIndex with columns o,h,l,c,v.
    m1_code / h1_code: pd.Series of codebook codes aligned to df_h1.index.
    cross: dict asset -> df_h1 (o,h,l,c) for XAGUSD/EURUSD/USDJPY (own clocks aligned to UTC).
    fred: dict series -> pd.Series of daily values indexed by date.
    priors: DataFrame with prior_w24/l24/w48/l48 aligned to df_h1.index (RAW
            values; they are kept for the parquet/decision layer - the STATE
            uses their z-scores from zscore_priors).
    innov: DataFrame from macro_innovations() aligned to df_h1.index
            (mi_{s}_d20z/d5z/rank60 columns; if None and fred given, computed).
    Returns DataFrame of dimensionless features (float), same index.
    """
    o, h, l, c, v = df_h1["o"], df_h1["h"], df_h1["l"], df_h1["c"], df_h1["v"]
    atr = wilder_atr(h, l, c, 14)
    F = pd.DataFrame(index=df_h1.index)
    # family 1: multi-horizon moves in ATR units (price distance / ATR ->
    # scale invariant, and aligned with the ATR-unit barrier geometry)
    for hh in (1, 4, 12, 24, 48):
        F[f"ret{hh}_atr"] = (c - c.shift(hh)) / (atr + EPS)
    # family 2/3: vol regime
    F["atr_ratio"] = atr / atr.shift(48)
    F["vol_z"] = roll_z(atr, 252)
    # family 4/5: oscillators (z-scored, past-only)
    F["rsi_z"] = roll_z(rsi(c, 14), 252)
    F["stoch_z"] = roll_z(stoch_k(h, l, c, 14), 252)
    # family 6: MACD in ATR units
    F["macd_atr"] = macd_atr(c, atr)
    # family 7/8: bollinger & donchian positions
    sma20 = c.rolling(20, min_periods=20).mean()
    std20 = c.rolling(20, min_periods=20).std()
    F["bb_pos"] = (c - sma20) / (2 * std20 + EPS)
    F["bb_w_z"] = roll_z(2 * std20 / (sma20 + EPS), 252)
    F["d20"] = (c - l.rolling(20, min_periods=20).min()) / (h.rolling(20, min_periods=20).max() - l.rolling(20, min_periods=20).min() + EPS)
    F["d55"] = (c - l.rolling(55, min_periods=55).min()) / (h.rolling(55, min_periods=55).max() - l.rolling(55, min_periods=55).min() + EPS)
    # family 9: candle anatomy, last 3 bars (ATR units)
    for k in (1, 2, 3):
        ok_, hk_, lk_, ck_ = o.shift(k), h.shift(k), l.shift(k), c.shift(k)
        F[f"body{k}"] = (ck_ - ok_) / (atr + EPS)
        F[f"uwick{k}"] = (hk_ - np.maximum(ok_, ck_)) / (atr + EPS)
        F[f"lwick{k}"] = (np.minimum(ok_, ck_) - lk_) / (atr + EPS)
    # family 10: directional run length (normalized, signed)
    sgn = np.sign(c.diff()).fillna(0.0)
    run = sgn.groupby((sgn != sgn.shift()).cumsum()).cumcount() + 1
    F["run"] = (run * sgn) / 10.0
    # family 11: volume z (volume level is not price-scale dependent)
    F["volm_z"] = roll_z(np.log1p(v.astype(float)), 252)
    # family 12/13: shape codebook codes
    if m1_code is not None:
        F["m1_shape"] = m1_code.reindex(df_h1.index)
    if h1_code is not None:
        F["h1_shape"] = h1_code.reindex(df_h1.index)
    # family 14: cross-asset relative moves in their own ATR units + macro z
    if cross:
        for a, dfa in cross.items():
            aatr = wilder_atr(dfa["h"], dfa["l"], dfa["c"], 14)
            rel = ((dfa["c"] - dfa["c"].shift(4)) / (aatr + EPS)).reindex(df_h1.index).ffill(limit=6)
            F[f"x_{a}"] = rel
    if fred:
        for k, s in fred.items():
            daily = pd.to_numeric(s, errors="coerce").sort_index()
            if daily.index.tz is None:
                daily.index = daily.index.tz_localize("UTC")
            # z-score, then shift(1): use YESTERDAY's value intraday (FRED daily
            # series are published during/after the US session - no lookahead)
            z = roll_z(daily, 252, minp=60).shift(1)
            # fill daily gaps first (weekends/holidays, up to 10 days), then map
            # every hour of a calendar date to that date's value
            z_ff = z.ffill(limit=10)
            F[f"m_{k}"] = z_ff.reindex(df_h1.index.normalize()).values
    # family 14b: macro INNOVATION panel (Spec v1.2: replaces raw macro levels
    # in the state; m_vix is kept - it passed G1.4-a at 0.456 in the v4 audit)
    if innov is None and fred:
        innov = macro_innovations(fred, df_h1.index)
    if innov is not None:
        for c in innov.columns:
            F[c] = innov[c]
    # family 15: past-only cohort base rates (merged later if provided)
    if priors is not None:
        for col in ("prior_w24", "prior_l24", "prior_w48", "prior_l48"):
            if col in priors.columns:
                F[col] = priors[col].reindex(df_h1.index)
    return F

# Feature columns used for cohort MATCHING (Spec v1.2: era-sanitized -
# raw m_dxy/m_y10 replaced by innovation forms so matched neighbors are not
# era-concentrated; m_vix kept, it passed G1.4-a). Excludes priors to avoid
# mild self-reference; all dimensionless.
COHORT_FEATURES = [
    "ret1_atr", "ret4_atr", "ret12_atr", "ret24_atr", "ret48_atr",
    "atr_ratio", "vol_z", "rsi_z", "stoch_z", "macd_atr", "bb_pos", "bb_w_z",
    "d20", "d55", "body1", "uwick1", "lwick1", "run", "volm_z",
    "m1_shape", "h1_shape",
]
COHORT_FEATURES += ["x_xagusd", "x_eurusd", "x_usdjpy", "m_vix",
                    "mi_dxy_d20z", "mi_y10_d20z"]

# ---------- synthetic market generator (for scale-invariance & smoke tests) --

def synthetic_ohlc(n_bars=3000, seed=7, p0=1837.42, m1_per_bar=60):
    """Regime-switching GBM with vol clustering; returns (h1_df, m1_df).
    p0 is arbitrary - features must be invariant to it (that is the test)."""
    rng = np.random.default_rng(seed)
    # H1 regime path
    vol = np.empty(n_bars); vol[0] = 0.10
    drift = np.zeros(n_bars)
    regime = 0
    for i in range(1, n_bars):
        if rng.random() < 0.01:
            regime = rng.integers(0, 3)
        vol[i] = np.clip(0.92 * vol[i - 1] + 0.08 * (0.05 + 0.12 * regime / 2) + rng.normal() * 0.008, 0.02, 0.9)
        drift[i] = (regime - 1) * 0.0004
    # M1 sub-paths (60 per H1 bar) - random walk bridging hourly closes
    tot = n_bars * m1_per_bar
    steps = rng.normal(drift.repeat(m1_per_bar) / m1_per_bar, (vol.repeat(m1_per_bar) / np.sqrt(m1_per_bar * 252 * 24)))
    m1_close = p0 * np.exp(np.cumsum(steps))
    idx_h1 = pd.date_range("2019-01-01", periods=n_bars, freq="1h", tz="UTC")
    idx_m1 = pd.date_range("2019-01-01", periods=tot, freq="1min", tz="UTC")
    # intrabar noise for o/h/l
    wick = np.abs(rng.normal(0, 0.35, tot)) * vol.repeat(m1_per_bar) / np.sqrt(252 * 24 * 60) * m1_close
    m1 = pd.DataFrame({
        "o": np.roll(m1_close, 1) * (1 + rng.normal(0, 1e-4, tot)),
        "h": m1_close + wick * rng.random(tot),
        "l": m1_close - wick * rng.random(tot),
        "c": m1_close,
        "v": rng.integers(10, 500, tot),
    }, index=idx_m1)
    m1.iloc[0, m1.columns.get_loc("o")] = p0
    m1["h"] = m1[["h", "o", "c"]].max(axis=1)
    m1["l"] = m1[["l", "o", "c"]].min(axis=1)
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum")).dropna()
    return h1, m1

def scale_ohlc(df, factor, cols=("o", "h", "l", "c")):
    out = df.copy()
    for col in cols:
        out[col] = out[col] * factor
    return out


# ==================== lib/aurum_label.py ====================
# ============================================================================
# AURUM 48-cell barrier labeler + B=200 matched-cohort bootstrap (Spec v1.1).
#
# Audit fix #2 (counterfactual validity): NO synthetic price paths are ever
# fabricated. Soft labels are the realized outcome distribution over K=200
# regime-matched HISTORICAL episodes (matched on dimensionless state features,
# strictly from the past) - a mathematically valid empirical bootstrap.
# Audit fix #4 (single-path noise): p_win / b_eff / excursion quantiles are
# distributional targets, not one noisy path realization.
# Audit fix #3: one standardized barrier matrix, 48 cells, evaluated for both
# directions in a single pass; the joint head learns all cells at once.
# All outcomes are in ATR units -> scale invariant (audit fix #1).
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12

DIRS = ("L", "S")

# ---------------------------------------------------------------------------
# 1) Actual-path deterministic outcomes per (bar, cell) - M1 execution
# ---------------------------------------------------------------------------

def barrier_actual_outcomes(m1, h1, i_states, chunk=512):
    """
    For each H1 state bar i (state = close of bar i), a hypothetical entry is
    taken at the OPEN of bar i+1. Outcomes are resolved on M1 data.

    Returns dict of arrays indexed by position in i_states:
      win    (N, 48) bool  : realized R > 0 under simple stack (TP-first=win,
                             SL-first=loss, timeout=win iff residual>0)
      R      (N, 48) float : realized P&L in R-multiples (R = m*ATR)
      cls    (N, 48) int8  : 0=SL first, 1=TP first, 2=timeout
      mfe    (N, 2, 2)     : max favorable excursion /ATR  [dir, horizon]
      mae    (N, 2, 2)     : max adverse  excursion /ATR  [dir, horizon]
      res    (N, 2, 2)     : timeout residual P&L /ATR     [dir, horizon]
    Entry uses next-bar open; ties inside the same minute -> SL first
    (conservative convention, applied uniformly).
    """
    m1 = m1.sort_index()
    m1_ts = np.asarray(m1.index.asi8, dtype=np.int64)  # ns epoch (UTC)
    m1_lo = m1["l"].values.astype(np.float64)
    m1_hi = m1["h"].values.astype(np.float64)
    m1_cl = m1["c"].values.astype(np.float64)

    h1 = h1.sort_index()
    h1_ts = np.asarray(h1.index.asi8, dtype=np.int64)
    h1_open = h1["o"].values.astype(np.float64)
    h1_hi = h1["h"].values.astype(np.float64)
    h1_lo = h1["l"].values.astype(np.float64)
    h1_cl = h1["c"].values.astype(np.float64)

    # ATR14 (Wilder) on H1, computed with past info only
    prev_c = np.roll(h1_cl, 1); prev_c[0] = h1_cl[0]
    tr = np.maximum(h1_hi - h1_lo, np.maximum(np.abs(h1_hi - prev_c), np.abs(h1_lo - prev_c)))
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().values

    n = len(i_states)
    n_m, n_t, n_h = len(BARRIER_STOPS), len(BARRIER_TPS), len(BARRIER_HORS)
    ns = BARRIER_STOPS; ts_ = BARRIER_TPS; hs = [h * 60 for h in BARRIER_HORS]  # minutes

    win = np.zeros((n, N_CELLS), dtype=bool)
    R = np.full((n, N_CELLS), np.nan)
    cls = np.full((n, N_CELLS), 255, dtype=np.int16)
    mfe = np.full((n, 2, 2), np.nan)
    mae = np.full((n, 2, 2), np.nan)
    res = np.full((n, 2, 2), np.nan)

    H1_NS = 3600 * 1_000_000_000
    MAX_WIN_NS = 48 * H1_NS

    for c0 in range(0, n, chunk):
        sl = slice(c0, min(c0 + chunk, n))
        rows = np.asarray(i_states[sl])
        # entry bar = state bar + 1; needs a next bar to exist
        valid = (rows + 1 < len(h1_ts)) & np.isfinite(atr[rows]) & (atr[rows] > 0)
        for ri, i in enumerate(rows):
            if not valid[ri]:
                continue
            e = h1_open[i + 1]
            a = atr[i]
            t0 = h1_ts[i + 1]              # entry time = open of next bar
            s = np.searchsorted(m1_ts, t0, side="left")
            e_idx = min(len(m1_ts), np.searchsorted(m1_ts, t0 + MAX_WIN_NS, side="left"))
            if e_idx - s < 30:             # not enough forward M1 data
                continue
            hi_w = m1_hi[s:e_idx]; lo_w = m1_lo[s:e_idx]; cl_w = m1_cl[s:e_idx]
            # first-touch helper: first index where condition holds, else BIG
            def first_touch(cond):
                idx = np.flatnonzero(cond)
                return idx[0] if len(idx) else 1 << 30
            lims = []
            for hmin in hs:
                lims.append(min(len(hi_w), max(1, int(np.searchsorted(m1_ts[s:e_idx], t0 + hmin * 60_000_000_000, side="left")))))
            rbase = 0
            for di, d in enumerate(DIRS):
                # long: TP above (hi >= e+t*a), SL below (lo <= e-m*a)
                if d == "L":
                    tps_hit = np.array([first_touch(hi_w >= e + t * a) for t in ts_])
                    sls_hit = np.array([first_touch(lo_w <= e - m * a) for m in ns])
                else:
                    tps_hit = np.array([first_touch(lo_w <= e - t * a) for t in ts_])
                    sls_hit = np.array([first_touch(hi_w >= e + m * a) for m in ns])
                for hi_, lim in enumerate(lims):
                    if d == "L":
                        mfe_v = (hi_w[:lim].max() - e) / a
                        mae_v = (e - lo_w[:lim].min()) / a
                        res_v = (cl_w[lim - 1] - e) / a
                    else:
                        mfe_v = (e - lo_w[:lim].min()) / a
                        mae_v = (hi_w[:lim].max() - e) / a
                        res_v = (e - cl_w[lim - 1]) / a
                    mfe[c0 + ri, di, hi_] = mfe_v
                    mae[c0 + ri, di, hi_] = mae_v
                    res[c0 + ri, di, hi_] = res_v
                for mi, m in enumerate(ns):
                    for ti, t in enumerate(ts_):
                        for hi_, lim in enumerate(lims):
                            tsl = sls_hit[mi]; ttp = tps_hit[ti]
                            cell = rbase
                            if tsl <= ttp and tsl < lim:      # SL first (ties->SL)
                                R[c0 + ri, cell] = -1.0
                                cls[c0 + ri, cell] = 0
                            elif ttp < lim:
                                R[c0 + ri, cell] = t / m
                                cls[c0 + ri, cell] = 1
                            else:
                                rv = res[c0 + ri, di, hi_]
                                R[c0 + ri, cell] = rv * a / (m * a) if np.isfinite(rv) else np.nan
                                cls[c0 + ri, cell] = 2
                            win[c0 + ri, cell] = R[c0 + ri, cell] > 0
                            rbase += 1
    return {"win": win, "R": R, "cls": cls, "mfe": mfe, "mae": mae, "res": res}

def expand_actual(actual, i_states, N):
    """Expand labeler outputs (indexed by i_states position) to full-length
    (N, ...) arrays so cohort gathering by original bar index works.
    win is float (1.0/0.0) with NaN outside i_states so rolling priors and
    cohort means ignore uncomputed bars."""
    out = {}
    w = np.full((N, N_CELLS), np.nan)
    w[i_states] = actual["win"].astype(np.float64)
    out["win"] = w
    full = np.full((N, N_CELLS), np.nan)
    full[i_states] = actual["R"]
    out["R"] = full
    for k in ("mfe", "mae", "res"):
        full = np.full((N, 2, 2), np.nan)
        full[i_states] = actual[k]
        out[k] = full
    out["cls"] = np.full((N, N_CELLS), 255, dtype=np.int16)
    out["cls"][i_states] = actual["cls"]
    return out

# ---------------------------------------------------------------------------
# 2) Matched-cohort index (past-only, leakage-safe)
# ---------------------------------------------------------------------------

def build_cohort_index(feat_std, label_valid, k=COHORT_K, gap=COHORT_GAP,
                       query_k=900, chunk=4000):
    """
    feat_std   : (N, F) standardized features; rows with label_valid=False may
                 contain NaN (they are never used as neighbours or queries).
    label_valid: (N,) bool - bar has valid features AND valid actual outcomes.
    Neighbour j is eligible for target i iff j <= i - gap (strictly in the
    past, forward windows cannot overlap) and label_valid[j].
    Returns int32 (N, k) cohort indices (-1 = missing) and per-bar counts.
    """
    N = feat_std.shape[0]
    valid_rows = np.flatnonzero(label_valid)
    nn = NearestNeighbors(n_neighbors=min(query_k, len(valid_rows)),
                          algorithm="kd_tree", metric="euclidean", n_jobs=-1)
    nn.fit(feat_std[valid_rows])
    cohort = np.full((N, k), -1, dtype=np.int32)
    nvalid = np.zeros(N, dtype=np.int32)
    for s in range(0, len(valid_rows), chunk):
        rows = valid_rows[s: s + chunk]
        _, idx_local = nn.kneighbors(feat_std[rows])
        for r in range(len(rows)):
            i = rows[r]
            cand = valid_rows[idx_local[r]]
            cand = cand[cand <= i - gap][:k]
            nvalid[i] = len(cand)
            if len(cand):
                cohort[i, : len(cand)] = cand
    return cohort, nvalid

# ---------------------------------------------------------------------------
# 3) Cohort bootstrap soft labels (B=200)
# ---------------------------------------------------------------------------

def cohort_soft_labels(cohort, nvalid, actual, k_min=120, chunk=4000):
    """
    From actual-path outcomes of matched episodes:
      p_win (N,48): mean win flag            (H1 training target)
      b_eff (N,48): mean win size / |mean loss size| in R units (Kelly b)
      ev_ref(N,48): mean realized R          (reference EV)
      q_mfe (N,2,2,3), q_mae (N,2,2,3): cohort q25/q50/q75 of excursions
                                          (H3 target; drives deterministic exits)
    Bars with < k_min valid neighbours -> NaN (excluded from training).
    """
    N = cohort.shape[0]
    win, R = actual["win"], actual["R"]
    mfe, mae = actual["mfe"], actual["mae"]
    p_win = np.full((N, N_CELLS), np.nan)
    b_eff = np.full((N, N_CELLS), np.nan)
    ev_ref = np.full((N, N_CELLS), np.nan)
    q_mfe = np.full((N, 2, 2, 3), np.nan)
    q_mae = np.full((N, 2, 2, 3), np.nan)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        for i in range(s, e):
            m = nvalid[i]
            if m < k_min:
                continue
            ids = cohort[i, :m]
            w = win[ids].astype(np.float64)      # (m,48)
            r = R[ids]                            # (m,48)
            p_win[i] = w.mean(0)
            with np.errstate(invalid="ignore"):
                pos = np.where(r > 0, r, np.nan)
                neg = np.where(r < 0, r, np.nan)
                mp = np.nanmean(pos, 0); mn = np.nanmean(neg, 0)
                b_eff[i] = mp / np.abs(mn)
                ev_ref[i] = np.nanmean(r, 0)
            for d in range(2):
                for hh in range(2):
                    q_mfe[i, d, hh] = np.nanquantile(mfe[ids, d, hh], [0.25, 0.5, 0.75]) if np.isfinite(mfe[ids, d, hh]).any() else np.nan
                    q_mae[i, d, hh] = np.nanquantile(mae[ids, d, hh], [0.25, 0.5, 0.75]) if np.isfinite(mae[ids, d, hh]).any() else np.nan
    return {"p_win": p_win, "b_eff": b_eff, "ev_ref": ev_ref, "q_mfe": q_mfe, "q_mae": q_mae}

# ---------------------------------------------------------------------------
# 3b) 48-cell adjacent triangular smoothing (Spec v1.2: primary H1 target =
#      realized hard outcomes + triangular smoothing over the cell grid)
# ---------------------------------------------------------------------------

def build_smooth_kernel():
    """Triangular kernel over the 48-cell grid (own 1.0, stop/TP +-1 -> 0.5,
    diagonal (m+-1,t+-1) -> 0.25, horizon +-1 (same m,t) -> 0.5; long and
    short directions are NEVER mixed). Rows sum to 1. Exact port of the
    postmortem aurum_common.build_smooth_kernel (exp2 T2_smoothed, the
    walk-forward winning target family)."""
    K = np.zeros((N_CELLS, N_CELLS))
    for i in range(N_CELLS):
        for j in range(N_CELLS):
            if i == j:
                K[i, j] = 1.0
                continue
            # decode flat index: d*24 + mi*6 + ti*2 + hi
            if (i // 24) != (j // 24):
                continue
            dmi = abs((i % 24) // 6 - (j % 24) // 6)
            dti = abs((i % 6) // 2 - (j % 6) // 2)
            dhi = abs((i % 2) - (j % 2))
            if dhi > 1 or dmi > 1 or dti > 1:
                continue
            if dhi == 1 and dmi == 0 and dti == 0:
                K[i, j] = 0.5
            elif dmi == 1 and dti == 0 and dhi == 0:
                K[i, j] = 0.5
            elif dti == 1 and dmi == 0 and dhi == 0:
                K[i, j] = 0.5
            elif dmi == 1 and dti == 1 and dhi == 0:
                K[i, j] = 0.25
    return K / K.sum(axis=1, keepdims=True)

def smoothed_targets(win_matrix):
    """win_matrix: (N, 48) float realized win flags (NaN where uncomputed).
    Returns (N, 48) triangular-smoothed targets in [0,1]; rows with any NaN
    cell stay NaN (label_valid bars have all 48 cells finite)."""
    K = build_smooth_kernel()
    W = win_matrix.copy()
    bad = ~np.isfinite(W).all(axis=1)
    W[bad] = 0.0
    out = W @ K.T
    out[bad] = np.nan
    return out

# ---------------------------------------------------------------------------
# 3c) Cohort INPUT summary (Spec v1.2: canonical long/short p_win, b_eff and
#     the cohort EV-argmax cell id, from neighbors <= i-96 only - these enter
#     the SERIALIZED STATE as compact inputs, never the targets)
# ---------------------------------------------------------------------------

CANON_L = 0 * 24 + 1 * 6 + 1 * 2 + 1      # L_m1.5_t2.5_h48  (index 9, pilot canon)
CANON_S = 1 * 24 + 1 * 6 + 1 * 2 + 1      # S_m1.5_t2.5_h48  (index 33)

def cohort_input_summary(soft):
    """From cohort_soft_labels output (p_win/b_eff computed over the SAME
    gap=96 past-only neighbor sets): per-bar canonical L/S p_win, b_eff and
    the EV-argmax cell id over all 48 cells. Returns dict of (N,) arrays."""
    p_win, b_eff = soft["p_win"], soft["b_eff"]
    with np.errstate(invalid="ignore"):
        ev = p_win * b_eff - (1.0 - p_win)          # (N, 48) EV in R units
        ok = np.isfinite(ev).all(axis=1)
        arg = np.full(len(ev), -1, dtype=np.int16)
        if ok.any():
            ev_safe = np.where(ok[:, None], ev, -np.inf)
            arg[ok] = np.argmax(ev_safe, axis=1)[ok]
    return {
        "coh_Lpw": p_win[:, CANON_L], "coh_Lb": b_eff[:, CANON_L],
        "coh_Spw": p_win[:, CANON_S], "coh_Sb": b_eff[:, CANON_S],
        "coh_arg": arg,
    }

# ---------------------------------------------------------------------------
# 4) Auxiliary head targets: H2 (MTM), H4 (vol), H5 (regime)
# ---------------------------------------------------------------------------

def aux_targets(h1, actual, atr):
    """H2: P(up) & magnitude bucket at h in {4,12,24,48}.
       H4: vol expansion prob at h in {4,12,24,48} + tercile class + q50 ratio.
       H5: 3x3 regime class (trend x vol) from past-only tercile edges."""
    c = h1["c"].values
    N = len(c)
    out = {}
    # H2 (moves in ATR units, matching the feature construction)
    for hh in (4, 12, 24, 48):
        fwd = np.full(N, np.nan)
        fwd[:-hh] = (c[hh:] - c[:-hh]) / atr[:-hh]
        sgn = (fwd > 0).astype(np.float64)
        mag = np.abs(fwd)
        med = pd.Series(mag).rolling(252, min_periods=60).median().shift(1).values
        out[f"h2_up{hh}"] = sgn
        out[f"h2_big{hh}"] = (mag > med).astype(np.float64)
    # H4: realized vol ratio fwd12/fwd48 (past window), expansion & terciles
    lr = np.log(c / np.roll(c, 1)); lr[0] = 0.0
    rv12 = pd.Series(lr).pow(2).rolling(12, min_periods=12).sum().pow(0.5).values
    rv48 = pd.Series(lr).pow(2).rolling(48, min_periods=48).sum().pow(0.5).values
    ratio = pd.Series(rv12 / (rv48 + EPS))
    for hh in (4, 12, 24, 48):
        fwd = ratio.shift(-hh).values
        out[f"h4_exp{hh}"] = (fwd > 1.0).astype(np.float64)
    q33 = ratio.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    q66 = ratio.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    fwd12 = ratio.shift(-12).values
    terc = np.full(N, 255, dtype=np.int16)
    terc[(fwd12 < q33)] = 0
    terc[(fwd12 >= q33) & (fwd12 <= q66)] = 1
    terc[(fwd12 > q66)] = 2
    out["h4_terc12"] = terc
    out["h4_q50_12"] = fwd12
    # H5 regime: trend = 24-bar move in ATR units vs past terciles; vol = atr_ratio terciles
    r24 = pd.Series((c - np.roll(c, 24)) / (atr + EPS))
    r24.iloc[0] = np.nan
    arr = r24.replace([np.inf, -np.inf], np.nan)
    t33 = arr.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    t66 = arr.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    trend = np.full(N, 1, dtype=np.int8)
    trend[arr.values < t33] = 0
    trend[arr.values > t66] = 2
    ar = pd.Series(atr / np.roll(atr, 48)); ar.iloc[0] = np.nan
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1).values
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1).values
    volc = np.full(N, 1, dtype=np.int8)
    volc[ar.values < v33] = 0
    volc[ar.values > v66] = 2
    out["h5_regime"] = (trend * 3 + volc).astype(np.int8)
    return out

# ---------------------------------------------------------------------------
# 5) Deterministic exit stack (MVP, Spec v1.1 Sec 12): q25 MAE trail,
#    q50 MFE partial TP, cell-m hard stop, H time stop.
# ---------------------------------------------------------------------------

def deterministic_exit_plan(p_win_row, b_eff_row, q_mfe_row, q_mae_row):
    """Given per-bar label rows, compute the trade plan the backtest will use:
    EV-argmax cell -> (direction, stop_m, tp_t, horizon, p, b), quarter-Kelly
    risk fraction with 1% hard cap, and exit levels (in ATR units):
    hard stop m, partial TP at q50 MFE (close half), trail q25 MAE below
    high-water mark, time stop at the cell horizon."""
    ev = p_win_row * b_eff_row - (1.0 - p_win_row)     # EV in R units
    m = int(np.nanargmax(ev))
    dirs, ms, ts, hs = cell_axes()
    p = float(p_win_row[m]); b = float(b_eff_row[m])
    if not np.isfinite(p) or not np.isfinite(b) or p <= 0 or b <= 0:
        return None
    kelly = p - (1 - p) / b
    risk = min(0.25 * max(kelly, 0.0), 0.01)           # quarter Kelly, 1% cap
    if risk <= 0:
        return None
    d = 0 if dirs[m] == "L" else 1
    hidx = 0 if hs[m] == 24 else 1
    return {
        "cell": m, "dir": dirs[m], "stop_atr": ms[m], "tp_atr": ts[m],
        "horizon": hs[m], "p": p, "b": b, "ev": float(ev[m]),
        "risk_frac": risk,
        "partial_tp_atr": float(q_mfe_row[d, hidx, 1]),   # q50 MFE
        "trail_mae_atr": float(q_mae_row[d, hidx, 0]),    # q25 MAE
    }


# ==================== lib/aurum_data.py ====================
# ============================================================================
# AURUM data ingestion: HistData M1 (primary), Dukascopy gap-fill (fallback),
# FRED daily macro (DTWEXBGS/DGS10/DGS2/T10Y2Y/VIXCLS). Free sources only.
# ============================================================================
import io, os, re, time, zipfile, requests
import numpy as np
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "text/html,application/xhtml+xml,application/zip,*/*"}

FRED_SERIES = {"dxy": "DTWEXBGS", "y10": "DGS10", "y2": "DGS2", "spread": "T10Y2Y", "vix": "VIXCLS"}

# ---------------------------------------------------------------------------
# HistData: token-flow downloader for yearly ASCII M1 archives
# ---------------------------------------------------------------------------

def histdata_year_url(pair, year):
    return f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{year}"

def histdata_download_year(pair, year, out_dir, session=None, tries=4, sleep_s=3.0, log=print):
    """Downloads the full-year ASCII M1 zip for pair (e.g. 'xauusd'). Returns
    path to zip or None. Parses the actual `file_down` form fields (tk token
    + platform/timeframe/fxpair controls) and POSTs them to /get.php."""
    path = os.path.join(out_dir, f"{pair.upper()}_M1_{year}.zip")
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        return path
    s = session or requests.Session()
    s.headers.update(UA)
    url = histdata_year_url(pair, year)
    for attempt in range(tries):
        try:
            r = s.get(url, timeout=60)
            if r.status_code != 200:
                log(f"  histdata {pair} {year}: GET {r.status_code}")
                time.sleep(sleep_s * (attempt + 1)); continue
            html = r.text
            form = re.search(r'<form id="file_down".*?</form>', html, re.S)
            if not form:
                log(f"  histdata {pair} {year}: no file_down form (len {len(html)})")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload = {}
            for tag in re.findall(r"<input[^>]*>", form.group(0)):
                nm = re.search(r'name="([^"]*)"', tag)
                vl = re.search(r'value="([^"]*)"', tag)
                if nm:
                    payload[nm.group(1)] = vl.group(1) if vl else ""
            if "tk" not in payload or not payload["tk"]:
                log(f"  histdata {pair} {year}: no tk token in form")
                time.sleep(sleep_s * (attempt + 1)); continue
            payload["datemonth"] = str(year)   # full year
            pr = s.post("https://www.histdata.com/get.php", data=payload,
                        headers={"Referer": url}, timeout=600)
            ct = pr.headers.get("content-type", "")
            if pr.status_code == 200 and (pr.content[:2] == b"PK") and len(pr.content) > 10_000:
                os.makedirs(out_dir, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(pr.content)
                log(f"  histdata {pair} {year}: OK {len(pr.content)/1e6:.1f} MB")
                return path
            log(f"  histdata {pair} {year}: POST st={pr.status_code} ct={ct} len={len(pr.content)}")
        except Exception as e:
            log(f"  histdata {pair} {year}: exc {type(e).__name__}: {str(e)[:120]}")
        time.sleep(sleep_s * (attempt + 1))
    return None

def parse_histdata_zip(zip_path, log=print):
    """Returns M1 DataFrame (UTC tz-aware index, o/h/l/c/v) or None.
    Handles both flat csv-in-zip and nested monthly zips."""
    frames = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            for nm in names:
                if nm.lower().endswith(".csv"):
                    frames.append(_read_hist_csv(z.read(nm)))
                elif nm.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(z.read(nm))) as z2:
                        for nm2 in z2.namelist():
                            if nm2.lower().endswith(".csv"):
                                frames.append(_read_hist_csv(z2.read(nm2)))
    except Exception as e:
        log(f"  parse {os.path.basename(zip_path)}: {type(e).__name__} {str(e)[:150]}")
        return None
    if not frames:
        return None
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df

def _read_hist_csv(raw):
    txt = raw.decode("latin-1").replace("\x00", "")
    # header: GMT time;Open;High;Low;Close;Volume, rows "20240101 180000;2062.598;..."
    df = pd.read_csv(io.StringIO(txt), sep=";", header=0, dtype=str,
                     usecols=[0, 1, 2, 3, 4, 5])
    df.columns = ["ts", "o", "h", "l", "c", "v"]
    for col in ("o", "h", "l", "c"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce").fillna(0.0)
    ts = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S", utc=True)
    df.index = ts
    df = df.dropna(subset=["o", "h", "l", "c"])
    df = df[(df["o"] > 0) & (df["h"] >= df[["o", "c"]].min(axis=1) - 1e-9)
            & (df["l"] <= df[["o", "c"]].max(axis=1) + 1e-9)]
    return df[["o", "h", "l", "c", "v"]]

# ---------------------------------------------------------------------------
# Dukascopy fallback (gap filler): per-day BID M1 bi5 (LZMA) files
# ---------------------------------------------------------------------------

def dukascopy_day_url(pair, y, m, d):
    return f"https://datafeed.dukascopy.com/datafeed/{pair.upper()}/{y}/{m:02d}/{d:02d}/BID_candles_min_1.bi5"

def parse_bi5(raw):
    try:
        import lzma
        data = lzma.decompress(raw)
    except Exception:
        return None
    n = len(data) // 24
    if n == 0:
        return None
    arr = np.frombuffer(data[:n * 24], dtype=">i4").reshape(n, 6).astype(np.float64)
    # cols: sec-offset(ms?), open, close, low, high, volume  (point-scaled)
    # Dukascopy M1: [time_ms_offset, open, close, low, high, volume], prices *1e5 (fx) or *1e3 (metals)
    return arr

# ---------------------------------------------------------------------------
# FRED (fredgraph.csv endpoints - no API key needed)
# ---------------------------------------------------------------------------

def fred_fetch(series_code, log=print):
    """FRED daily series via fredgraph.csv (no API key). requests first,
    curl-subprocess fallback (some networks throttle python-requests UAs)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    txt = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers=UA, timeout=45)
            if r.status_code == 200 and "observation_date" in r.text[:200]:
                txt = r.text
                break
            log(f"  FRED {series_code}: st={r.status_code}")
        except Exception as e:
            log(f"  FRED {series_code}: exc {type(e).__name__}")
        time.sleep(4 * (attempt + 1))
    if txt is None:
        try:
            import subprocess
            p = subprocess.run(["curl", "-sS", "--max-time", "90", url],
                               capture_output=True, text=True, timeout=120)
            if p.returncode == 0 and "observation_date" in p.stdout[:200]:
                txt = p.stdout
                log(f"  FRED {series_code}: OK via curl fallback")
        except Exception as e:
            log(f"  FRED {series_code}: curl exc {type(e).__name__}")
    if txt is None:
        return None
    try:
        df = pd.read_csv(io.StringIO(txt))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
        s = df.set_index("date")["value"].sort_index()
        log(f"  FRED {series_code}: OK {len(s)} rows ({s.index.min().date()}..{s.index.max().date()})")
        return s
    except Exception as e:
        log(f"  FRED {series_code}: parse exc {type(e).__name__}")
        return None

# ---------------------------------------------------------------------------
# UTC-offset estimation for HistData clocks (their ASCII export has run in a
# server-local clock historically). We correlate the hourly volatility profile
# against a canonical forex session template and pick the best integer shift.
# ---------------------------------------------------------------------------

def estimate_utc_offset(m1, log=print, candidates=(-3, -2, -1, 0, 1, 2, 3)):
    """m1: M1 df with tz-UTC index (as-parsed clock). Returns (offset_hours, corr)."""
    r = np.log(m1["c"] / m1["c"].shift(1)).abs()
    prof = r.groupby(m1.index.hour).mean().reindex(range(24), fill_value=0.0).values
    prof = (prof - prof.mean()) / (prof.std() + 1e-12)
    # canonical template: London open 07-08 UTC, NY open 12-13 UTC, overlap 15-16
    tpl = np.zeros(24)
    tpl[[7, 8, 12, 13, 15, 16]] = 1.6
    tpl[[3, 4, 5, 6, 9, 10, 17, 18]] = 0.8
    tpl[[21, 22, 23, 0, 1, 2]] = 0.4
    tpl = (tpl - tpl.mean()) / (tpl.std() + 1e-12)
    best_off, best_c = 0, -2
    for off in candidates:
        c = np.corrcoef(np.roll(prof, off), tpl)[0, 1]
        if c > best_c:
            best_c, best_off = c, off
    c0 = np.corrcoef(prof, tpl)[0, 1]
    if best_c < c0 * 1.2 or best_c < 0.35:
        best_off, best_c = 0, c0   # ambiguous -> keep as-is
    log(f"  utc-offset est: offset={best_off:+d}h corr={best_c:.2f} (null corr={c0:.2f})")
    return best_off, float(best_c)

def apply_utc_offset(df, offset_hours):
    if offset_hours == 0:
        return df
    return df.set_index(df.index + pd.Timedelta(hours=offset_hours))

# ---------------------------------------------------------------------------
# H1 aggregation + coverage report
# ---------------------------------------------------------------------------

def to_h1(m1):
    h1 = m1.resample("1h").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                               c=("c", "last"), v=("v", "sum"))
    return h1.dropna(subset=["o", "c"])

def coverage_report(m1, years=range(2010, 2026)):
    rep = {}
    exp = 365.25 * 24 * 60 * (5 / 7)  # approx tradable minutes per year
    for y in years:
        sub = m1[(m1.index.year == y)]
        got = len(sub)
        rep[str(y)] = {"rows": int(got), "coverage": round(got / exp, 3) if exp else None}
    return rep


# ==================== lib/aurum_ser.py ====================
# ============================================================================
# AURUM state serialization (Spec v1.1 sec 7.1): each H1 state becomes a
# ~312-token dimensionless text block. No absolute prices, no absolute dates -
# only cyclical calendar (hour-of-day, dow, session), ATR-unit quantities,
# z-scores, ratios, codebook ids.
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# Shape codebooks (fit on TRAIN window only, then applied everywhere)
# ---------------------------------------------------------------------------

def m1_path_features(m1, h1_index, n_pts=10):
    """For each H1 bar: normalized intrabar M1 path shape.
    Returns (N, n_pts+2) matrix: n_pts resampled normalized closes + relative
    time position of path high & low (all dimensionless)."""
    N = len(h1_index)
    X = np.full((N, n_pts + 2), np.nan)
    h1_starts = h1_index.asi8
    m1_idx = m1.index.asi8
    m1_c = m1["c"].values
    m1_o = m1["o"].values
    m1_h = m1["h"].values
    m1_l = m1["l"].values
    H1ns = 3_600_000_000_000
    for i in range(N):
        s = np.searchsorted(m1_idx, h1_starts[i], side="left")
        e = np.searchsorted(m1_idx, h1_starts[i] + H1ns, side="left")
        if e - s < 10:
            continue
        c = m1_c[s:e]; o = m1_o[s]
        rng = c.max() - c.min()
        if rng <= 0:
            continue
        norm = (c - c.min()) / rng
        # resample to n_pts points (indices, not values, to stay time-uniform)
        pos = np.linspace(0, len(norm) - 1, n_pts)
        X[i, :n_pts] = np.interp(pos, np.arange(len(norm)), norm)
        X[i, n_pts] = np.argmax(m1_h[s:e]) / (e - s)       # rel time of high
        X[i, n_pts + 1] = np.argmin(m1_l[s:e]) / (e - s)   # rel time of low
    return X

def h1_path_features(h1, n_bars=8):
    """Normalized shape of the last n_bars H1 closes (z-scored path)."""
    c = h1["c"].values
    N = len(c)
    X = np.full((N, n_bars), np.nan)
    for i in range(n_bars - 1, N):
        seg = c[i - n_bars + 1: i + 1]
        sd = seg.std()
        X[i] = (seg - seg.mean()) / (sd + 1e-12) if sd > 0 else 0.0
    return X

def fit_codebook(X, k=32, seed=13, sample=200_000):
    rows = X[np.isfinite(X).all(axis=1)]
    if len(rows) > sample:
        rows = rows[np.random.default_rng(seed).choice(len(rows), sample, replace=False)]
    km = MiniBatchKMeans(n_clusters=k, n_init=5, random_state=seed, batch_size=4096).fit(rows)
    return km

def apply_codebook(km, X, index=None):
    codes = np.full(len(X), -1, dtype=np.int32)
    ok = np.isfinite(X).all(axis=1)
    if ok.any():
        codes[ok] = km.predict(X[ok])
    return pd.Series(codes, index=index if index is not None else range(len(X)))

# ---------------------------------------------------------------------------
# State serialization
# ---------------------------------------------------------------------------

def _f(x, nd=2, signed=True):
    if x is None or not np.isfinite(x):
        return "na"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"

def _u(x, nd=2):
    if x is None or not np.isfinite(x):
        return "na"
    return f"{x:.{nd}f}"

def _r100(x):
    """rank60 in [0,1] -> integer 0..99 (compact, 1-2 tokens)."""
    if x is None or not np.isfinite(x):
        return "na"
    return f"{int(round(float(x) * 100)):02d}"

def _c(x, nd=1):
    """v1.2 compact number format: unsigned ('-' only for negatives), no
    leading zero before the decimal point ('.88' / '-.4' / '10.8'). Measured
    ~2 tokens/field vs ~5 for '+0.88' - this is what keeps the FULL 15-value
    macro panel inside the ~313-330-token state budget."""
    if x is None or not np.isfinite(x):
        return "na"
    s = f"{float(x):.{nd}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s

# Spec v1.2 macro innovation panel: (state key, feature base name)
MINNOV_SERIES = (("y10", "mi_y10"), ("y2", "mi_y2"), ("sp", "mi_spread"),
                 ("dx", "mi_dxy"), ("vx", "mi_vix"))

def serialize_state(ts, row, with_cross=True, with_macro=True, with_priors=True,
                   with_minnov=False, with_zpriors=False, with_cohort=False,
                   trend3=None, vol3=None, drop=None, compact=False):
    """ts: bar timestamp (UTC, end of bar); row: dict/Series of features.
    trend3/vol3 may be passed directly or live in the row (0/1/2 class ids).
    drop: set of feature names excluded by the adversarial-validation audit -
    their fields/lines are omitted from the frozen state format.
    v1.2 additions (all default OFF so the frozen v1.1 call sites are
    byte-identical):
      with_minnov  - 'minnov' line: per FRED series d20z/d5z/rank60 triplet
                     (era-invariant macro innovations, strict 1-day lag)
      with_zpriors - 'prior' line carries SELF-Z-SCORED priors (era-safe) in
                     place of the raw trailing rates
      with_cohort  - 'coh' line: canonical L/S p_win, b_eff + EV-argmax cell id
                     (neighbors <= i-96 only)
      compact      - v1.2 number encoding (unsigned, no leading zero, 1dp on
                     low-information fields) -> full format fits ~313-330 tokens
    Returns the text block (~313-330 Qwen tokens in the v1.2 format)."""
    drop = drop or set()
    hour = ts.hour
    trend = row.get("trend3", np.nan) if trend3 is None else trend3
    volc = row.get("vol3", np.nan) if vol3 is None else vol3
    L = []
    L.append(f"[STATE h{hour:02d} dow={DOW[ts.weekday()]} sess={session_of_hour(hour)}]")
    if compact:
        L.append("ret_atr r1={r1} r4={r4} r12={r12} r24={r24} r48={r48}".format(
            r1=_c(row.get("ret1_atr"), 2), r4=_c(row.get("ret4_atr"), 2),
            r12=_c(row.get("ret12_atr"), 2), r24=_c(row.get("ret24_atr"), 2),
            r48=_c(row.get("ret48_atr"), 2)))
        L.append("vol_atr atrr={atrr} volz={volz}".format(
            atrr=_c(row.get("atr_ratio"), 2), volz=_c(row.get("vol_z"), 2)))
        L.append("osc rsi={rsi} stoch={st} macd={macd}".format(
            rsi=_c(row.get("rsi_z"), 2), st=_c(row.get("stoch_z"), 2), macd=_c(row.get("macd_atr"), 2)))
        L.append("range bb={bb} bbw={bbw} d20={d20} d55={d55} run={run}".format(
            bb=_c(row.get("bb_pos"), 2), bbw=_c(row.get("bb_w_z"), 2), d20=_c(row.get("d20"), 2),
            d55=_c(row.get("d55"), 2), run=_c(row.get("run"), 2)))
        L.append("candle c1=b{b1}/u{u1}/d{d1} c2=b{b2}/u{u2}/d{d2} c3=b{b3}/u{u3}/d{d3}".format(
            b1=_c(row.get("body1"), 2), u1=_c(row.get("uwick1"), 2), d1=_c(row.get("lwick1"), 2),
            b2=_c(row.get("body2"), 2), u2=_c(row.get("uwick2"), 2), d2=_c(row.get("lwick2"), 2),
            b3=_c(row.get("body3"), 2), u3=_c(row.get("uwick3"), 2), d3=_c(row.get("lwick3"), 2)))
        L.append("shape m1={m1} h1={h1} vlm={vlm}".format(
            m1=("na" if not np.isfinite(row.get("m1_shape", np.nan)) else f"{int(row['m1_shape']):02d}"),
            h1=("na" if not np.isfinite(row.get("h1_shape", np.nan)) else f"{int(row['h1_shape']):02d}"),
            vlm=_c(row.get("volm_z"), 2)))
    else:
        L.append("ret_atr r1={r1} r4={r4} r12={r12} r24={r24} r48={r48}".format(
            r1=_f(row.get("ret1_atr")), r4=_f(row.get("ret4_atr")), r12=_f(row.get("ret12_atr")),
            r24=_f(row.get("ret24_atr")), r48=_f(row.get("ret48_atr"))))
        L.append("vol_atr atrr={atrr} volz={volz}".format(
            atrr=_u(row.get("atr_ratio")), volz=_f(row.get("vol_z"))))
        L.append("osc rsi={rsi} stoch={st} macd={macd}".format(
            rsi=_f(row.get("rsi_z")), st=_f(row.get("stoch_z")), macd=_f(row.get("macd_atr"))))
        L.append("range bb={bb} bbw={bbw} d20={d20} d55={d55} run={run}".format(
            bb=_f(row.get("bb_pos")), bbw=_f(row.get("bb_w_z")), d20=_u(row.get("d20")),
            d55=_u(row.get("d55")), run=_f(row.get("run"))))
        L.append("candle c1=b{b1}/u{u1}/d{d1} c2=b{b2}/u{u2}/d{d2} c3=b{b3}/u{u3}/d{d3}".format(
            b1=_f(row.get("body1")), u1=_f(row.get("uwick1"), 2), d1=_f(row.get("lwick1")),
            b2=_f(row.get("body2")), u2=_f(row.get("uwick2")), d2=_f(row.get("lwick2")),
            b3=_f(row.get("body3")), u3=_f(row.get("uwick3")), d3=_f(row.get("lwick3"))))
        L.append("shape m1={m1} h1={h1} vlm={vlm}".format(
            m1=("na" if not np.isfinite(row.get("m1_shape", np.nan)) else f"{int(row['m1_shape']):02d}"),
            h1=("na" if not np.isfinite(row.get("h1_shape", np.nan)) else f"{int(row['h1_shape']):02d}"),
            vlm=_f(row.get("volm_z"))))
    if with_cross:
        xg = [("xag", "x_xagusd", row.get("x_xagusd")), ("eur", "x_eurusd", row.get("x_eurusd")),
              ("jpy", "x_usdjpy", row.get("x_usdjpy"))]
        xg = [(a, v) for a, k, v in xg if k not in drop]
        if xg:
            fx = (lambda v: _c(v, 2)) if compact else (lambda v: _f(v))
            L.append("xasset " + " ".join(f"{a}={fx(v)}" for a, v in xg))
    if with_macro:
        mc = [("dxy", "m_dxy", row.get("m_dxy")), ("y10", "m_y10", row.get("m_y10")),
              ("vix", "m_vix", row.get("m_vix"))]
        mc = [(a, v) for a, k, v in mc if k not in drop]
        if mc:
            fx = (lambda v: _c(v, 2)) if compact else (lambda v: _f(v))
            L.append("macro " + " ".join(f"{a}={fx(v)}" for a, v in mc))
    if with_minnov:
        # per series: d20z(1dp) d5z(1dp) rank60(int 0-99), space separated
        parts = []
        for short, base in MINNOV_SERIES:
            if f"{base}_d20z" in drop:
                continue
            parts.append(f"{short} {_c(row.get(base + '_d20z'))} "
                         f"{_c(row.get(base + '_d5z'))} "
                         f"{_r100(row.get(base + '_rank60'))}")
        if parts:
            L.append("minnov " + " ".join(parts))
    trend_v, volc_v = trend, volc
    try:
        if np.isfinite(trend_v) and np.isfinite(volc_v):
            tn = ["down", "flat", "up"][int(trend_v)]; vn = ["low", "mid", "high"][int(volc_v)]
            L.append(f"regime trend={tn} vol={vn}")
    except (TypeError, ValueError):
        pass
    if with_priors and not with_zpriors:
        pr = [("w24", "prior_w24", row.get("prior_w24")), ("l24", "prior_l24", row.get("prior_l24")),
              ("w48", "prior_w48", row.get("prior_w48")), ("l48", "prior_l48", row.get("prior_l48"))]
        pr = [(a, v) for a, k, v in pr if k not in drop]
        if pr:
            L.append("prior " + " ".join(f"{a}={_u(v)}" for a, v in pr))
    if with_zpriors:
        # Spec v1.2: self-z-scored base rates (era-safe encoding of the 4
        # priors; raw trailing win rates were date fingerprints at AUC .85-.90)
        zr = [("zw24", "prior_zw24"), ("zl24", "prior_zl24"),
              ("zw48", "prior_zw48"), ("zl48", "prior_zl48")]
        zr = [(a, row.get(c)) for a, c in zr if c not in drop]
        if zr:
            L.append("prior " + " ".join(f"{a}={_c(v, 2)}" for a, v in zr))
    if with_cohort:
        # Spec v1.2 cohort input summary (neighbors <= i-96 only)
        argv = row.get("coh_arg")
        arg_s = "na" if argv is None or not np.isfinite(argv) else f"{int(argv):02d}"
        cf = [("Lpw", "coh_Lpw", _c(row.get("coh_Lpw"), 2)),
              ("Lb", "coh_Lb", _c(row.get("coh_Lb"), 2)),
              ("Spw", "coh_Spw", _c(row.get("coh_Spw"), 2)),
              ("Sb", "coh_Sb", _c(row.get("coh_Sb"), 2)),
              ("arg", "coh_arg", arg_s)]
        cf = [(a, v) for a, k, v in cf if k not in drop]
        if cf:
            L.append("coh " + " ".join(f"{a}={v}" for a, v in cf))
    return "\n".join(L)

def serialize_states(df_feat, h1_index, **kw):
    """Serialize all states. Returns list of texts."""
    out = []
    trend = kw.pop("trend3", None); volc = kw.pop("vol3", None)
    for i, ts in enumerate(h1_index):
        row = df_feat.iloc[i]
        if trend is not None:
            row = row.copy()
            row["trend3"] = trend[i]; row["vol3"] = volc[i]
        out.append(serialize_state(ts, row, **kw))
    return out


# ==================== lib/aurum_model.py ====================
# ============================================================================
# AURUM LoopedQwen3 (Spec v1.1, frozen):
#   - Base: Qwen/Qwen3-0.6B (28 decoder layers, d=1024).
#   - Looped core: decoder layers 13..16 (inclusive) re-executed R times
#     (R in {1,2,3}); shared weights, per-iteration Sandwich RMSNorm with
#     identity init (audit fix #5: full unfreeze of the ~60M core params
#     instead of rank-16 LoRA; T4 16GB holds it).
#   - Sandwich design (per iteration r):
#         z = pre_r(h);  z = core(z);  h = clamp(post_r(z))
#     with pre/post = x + tanh(g)*(RMSNorm(x) - x), g=0 at init, so
#     R=1 is EXACTLY the pretrained model at initialization (verified by
#     the r=1 equivalence self-test).
#   - fp16 guardrails (user directive): torch.amp autocast(fp16) + GradScaler
#     everywhere; RMSNorm accumulates in fp32; pre-norm residual norms are
#     clamped (C=100) so the looped stream can never overflow fp16 on sm_75.
#   - Joint Multi-Task World-Model Head: 5 learned readout tokens appended at
#     the end of the state sequence (guardrail: no variable-option permutation
#     masks). Single forward pass -> 85 outputs:
#       H1_BARRIER 48 (per-cell P(win), sigmoid)
#       H2_MTM      8 (multi-horizon direction & magnitude, sigmoid)
#       H3_EXCUR   12 (MFE/MAE q25/q50/q75 in ATR units, softplus)
#       H4_VOL      8 (expansion probs + terciles + q50)
#       H5_REGIME   9 (trend x vol classes, softmax)
# ============================================================================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint as tk_ckpt


BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
CORE_START, CORE_END = 13, 17          # decoder layers 13..16 inclusive
N_READOUT = 5
# Qwen3-0.6B residual-stream L2 norms reach ~7.5k on real activations (measured:
# mean ~200-700, max 7464). The clamp is a runaway guard for the looped core,
# NOT a normal-path operation: C=16384 sits 2x above the observed stock max and
# 4x below fp16 overflow (65504), so it never fires on healthy activations.
RESID_CLAMP = 16384.0

# ---------------------------------------------------------------------------


class Fp32RMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation (weight kept fp32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        v = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.weight.float()).to(dt)


class SandwichNorm(nn.Module):
    """x + tanh(g) * (RMSNorm(x) - x); g=0 -> exact identity at init."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.n = Fp32RMSNorm(dim)

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        n32 = self.n(x).float()
        out = x32 + torch.tanh(self.g).float() * (n32 - x32)
        return out.to(dt)


def clamp_resid(x, C=RESID_CLAMP, stats=None):
    """Clamp the L2 norm of the residual stream (fp32 math, per token)."""
    n = x.float().norm(dim=-1, keepdim=True)
    s = torch.clamp(C / (n + 1e-6), max=1.0)
    if stats is not None and torch.is_grad_enabled() and n.numel() > 0:
        with torch.no_grad():
            nm = n.max()
            stats["resid_max"] = max(stats.get("resid_max", 0.0), float(nm))
            stats["clamp_count"] = stats.get("clamp_count", 0.0) + float((n > C).sum())
            stats["clamp_total"] = stats.get("clamp_total", 0.0) + float(n.numel())
    return (x.float() * s).to(x.dtype)


def build_causal_mask(T, dtype, device):
    m = torch.full((1, 1, T, T), torch.finfo(dtype).min, dtype=torch.float32, device=device)
    m = torch.triu(m, diagonal=1)
    return m.to(dtype)


def build_joint_mask(pad_mask, T, L, dtype, device):
    """Causal + key-padding mask. pad_mask: (B, L) True=real token.
    Readout positions (L..T-1) are always valid keys."""
    B = pad_mask.size(0)
    key_ok = torch.ones(B, T, dtype=torch.bool, device=device)
    key_ok[:, :L] = pad_mask
    neg = torch.finfo(dtype).min
    m = torch.zeros(B, 1, T, T, dtype=torch.float32, device=device)
    m.masked_fill_(~key_ok[:, None, None, :], neg)
    causal = torch.triu(torch.full((T, T), neg, dtype=torch.float32, device=device), diagonal=1)
    m = m + causal[None, None]
    return m.clamp(min=neg).to(dtype)


def call_layer(layer, h, pe, pos_ids, attn_mask):
    kw = dict(attention_mask=attn_mask, position_ids=pos_ids, use_cache=False,
              output_attentions=False)
    if pe is not None:
        kw["position_embeddings"] = pe
    try:
        out = layer(h, **kw)
    except TypeError:
        kw.pop("position_embeddings", None)
        kw.pop("position_ids", None)
        out = layer(h, attention_mask=attn_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _mlp_head(d, hidden, out):
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))


class JointHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.readout = nn.Embedding(N_READOUT, d_model)
        nn.init.normal_(self.readout.weight, std=0.02)
        self.h1 = _mlp_head(d_model, hidden, HEAD_DIMS["h1"])   # 48 sigmoid
        self.h2 = _mlp_head(d_model, hidden, HEAD_DIMS["h2"])   # 8 sigmoid
        self.h3 = _mlp_head(d_model, hidden, HEAD_DIMS["h3"])   # 12 softplus
        self.h4 = _mlp_head(d_model, hidden, HEAD_DIMS["h4"])   # 8 sigmoid(+)
        self.h5 = _mlp_head(d_model, hidden, HEAD_DIMS["h5"])   # 9 softmax

    def forward(self, ro_hidden):
        return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                "h5": self.h5(ro_hidden[:, 4])}

    @torch.no_grad()
    def predict(self, logits):
        """Apply the frozen activations -> dict of probabilities/values.
        h4 layout: [0:4] expansion probs (sigmoid), [4:7] vol tercile (softmax),
        [7] fwd12/48 vol-ratio q50 (softplus)."""
        h4 = torch.cat([torch.sigmoid(logits["h4"][:, :4]),
                        torch.softmax(logits["h4"][:, 4:7], dim=-1),
                        Fn.softplus(logits["h4"][:, 7:8])], dim=1)
        out = {"h1": torch.sigmoid(logits["h1"]), "h2": torch.sigmoid(logits["h2"]),
               "h3": Fn.softplus(logits["h3"]), "h4": h4,
               "h5": torch.softmax(logits["h5"], dim=-1)}
        return out


class LoopedQwen(nn.Module):
    def __init__(self, base, n_loop=3, sandwich=True, clamp=RESID_CLAMP,
                 grad_ckpt=False, explicit_mask=False):
        super().__init__()
        self.base = base                       # AutoModelForCausalLM
        self.n_loop = n_loop
        self.sandwich = sandwich
        self.clamp = clamp
        self.grad_ckpt = grad_ckpt
        self.explicit_mask = explicit_mask
        d = base.config.hidden_size
        self.pre_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.post_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.head = JointHead(d)
        self.stats = {}
        # Spec v1.2 sliced-CE: active-vocab slice of the FROZEN lm_head.
        # keep_ids remaps full-vocab ids -> 0..K-1; the full logits tensor is
        # never materialized in the training path (151,936 -> ~112 logits).
        self.register_buffer("keep_ids", None, persistent=False)
        self.keep_pos = None                   # full-id -> sliced-id lookup

    def set_keep_ids(self, keep_ids):
        """keep_ids: LongTensor of active vocab ids (sorted)."""
        dev = next(self.base.parameters()).device
        self.keep_ids = torch.as_tensor(np.asarray(keep_ids, dtype=np.int64), device=dev)
        V = self.base.config.vocab_size
        pos = torch.full((V,), -1, dtype=torch.long, device=dev)
        pos[self.keep_ids] = torch.arange(len(self.keep_ids), device=dev)
        self.keep_pos = pos

    # ---------------- parameter management ----------------
    def set_trainable(self, train_lm_head=False):
        for p in self.base.parameters():
            p.requires_grad = False
        for i in range(CORE_START, CORE_END):
            for p in self.base.model.layers[i].parameters():
                p.requires_grad = True
        if train_lm_head and not self.base.config.tie_word_embeddings:
            for p in self.base.lm_head.parameters():
                p.requires_grad = True
        for p in self.pre_norms.parameters():
            p.requires_grad = True
        for p in self.post_norms.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- core forward ----------------
    def _run_layer(self, layer, h, pe, pos_ids, mask):
        if self.grad_ckpt and self.training:
            return tk_ckpt(lambda hh: call_layer(layer, hh, pe, pos_ids, mask), h,
                           use_reentrant=False)
        return call_layer(layer, h, pe, pos_ids, mask)

    def forward(self, ids, add_readout=True, lm_output=True, pad_mask=None,
                z_sample=0, ro_embed=None):
        """lm_output: True=full-vocab logits | 'sliced'=K-logit slice (needs
        set_keep_ids) | False=none. z_sample: if >0, ALSO compute the kept-mass
        ratio Z on a subsample of positions (with grad) -> out['z'].
        ro_embed: (N_READOUT, d) tensor replacing the model's readout
        embeddings (Spec v1.2 dual-branch Stage B / fold heads) - gradients
        flow into the CALLER's tensor. out['ro_h'] exposes the raw readout
        hidden states for external head stacks."""
        m = self.base.model
        B, L = ids.shape
        te = m.embed_tokens(ids)
        if add_readout:
            if ro_embed is None:
                ro = self.head.readout.weight.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
            else:
                ro = ro_embed.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
            h = torch.cat([te, ro], dim=1)
            T = L + N_READOUT
        else:
            h = te
            T = L
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        pe = None
        if hasattr(m, "rotary_emb"):
            pe = m.rotary_emb(h.float(), pos_ids)
        mask = None
        if pad_mask is not None:
            mask = build_joint_mask(pad_mask, T, L, h.dtype, h.device)
        elif self.explicit_mask:
            mask = build_causal_mask(T, h.dtype, h.device)

        def core_pass(hh):
            for i, layer in enumerate(m.layers):
                if i == CORE_START:
                    # In-place recursion: h_{r+1} = clamp(post_r(core(pre_r(h_r)))).
                    # With identity-init norms (g=0) and R=1 this is EXACTLY the
                    # pretrained forward (verified by the r=1 equivalence test).
                    for r in range(self.n_loop):
                        if self.sandwich:
                            z = self.pre_norms[r](hh)
                            for j in range(CORE_START, CORE_END):
                                z = self._run_layer(m.layers[j], z, pe, pos_ids, mask)
                            z = self.post_norms[r](z)
                            hh = clamp_resid(z, self.clamp, self.stats)
                        else:
                            for j in range(CORE_START, CORE_END):
                                hh = self._run_layer(m.layers[j], hh, pe, pos_ids, mask)
                    continue
                if CORE_START < i < CORE_END:
                    continue
                hh = self._run_layer(layer, hh, pe, pos_ids, mask)
            return hh

        if self.grad_ckpt and self.training:
            h = tk_ckpt(core_pass, h, use_reentrant=False)
        else:
            h = core_pass(h)
        h = m.norm(h)

        out = {}
        if lm_output is True:
            out["logits"] = self.base.lm_head(h[:, :L] if add_readout else h)
        elif lm_output == "sliced":
            # Spec v1.2: full logits (B, L, 151936) are NEVER materialized;
            # only the K active columns of the frozen lm_head are used.
            hs = h[:, :L] if add_readout else h
            W = self.base.lm_head.weight                     # (V, d), frozen
            Wk = W.index_select(0, self.keep_ids)            # (K, d)
            out["logits_sliced"] = torch.nn.functional.linear(hs, Wk)
        if add_readout:
            out["ro_h"] = h[:, -N_READOUT:, :]
            out["head"] = self.head(h[:, -N_READOUT:, :])
        if z_sample > 0 and self.keep_ids is not None and lm_output is not False:
            hs = h[:, :L] if add_readout else h
            n = hs.shape[0] * hs.shape[1]
            flat_idx = torch.randint(0, n, (min(z_sample, n),), device=hs.device)
            hsub = hs.reshape(-1, hs.shape[-1])[flat_idx]    # (S, d) w/ grad
            zlogits = torch.nn.functional.linear(hsub, self.base.lm_head.weight)  # (S, V)
            zprob = torch.softmax(zlogits.float(), dim=-1)
            out["z"] = zprob.index_select(1, self.keep_ids).sum(-1).mean()
        return out

    # ---------------- losses ----------------
    @staticmethod
    def lm_loss(logits, ids, pad_mask=None, chunk_rows=2048):
        """Memory-safe chunked CE. The 151,936-vocab logits for (B=16, T=512)
        are ~2.5GB in fp16; a single fp32 cast + CE would add ~7.5GB and OOM a
        T4. Each chunk's fp32 cast + CE is recomputed in backward via
        checkpoint, so peak fp32 memory is one chunk (~1.2GB)."""
        tgt = ids[:, 1:].clone()
        if pad_mask is not None:
            tgt[~pad_mask[:, 1:]] = -100
        lg = logits[:, :-1].reshape(-1, logits.size(-1))
        tg = tgt.reshape(-1)
        n_rows = lg.shape[0]
        total = torch.zeros((), device=lg.device, dtype=torch.float32)
        n = 0
        for s in range(0, n_rows, chunk_rows):
            sl = slice(s, min(s + chunk_rows, n_rows))
            t = tg[sl]
            cnt = int((t != -100).sum())
            if cnt == 0:
                continue
            def _ce(l, tt):
                return Fn.cross_entropy(l.float(), tt, ignore_index=-100,
                                        reduction="sum")
            total = total + tk_ckpt(_ce, lg[sl], t, use_reentrant=False)
            n += cnt
        return total / max(n, 1)

    def sliced_lm_loss(self, logits_sliced, ids):
        """Spec v1.2 sliced-CE: cross-entropy over the K active-vocab logits
        only (targets remapped full-id -> sliced-id; off-vocab targets are
        impossible by construction - the stream is built from keep_ids).
        logits_sliced: (B, L, K); ids: (B, L) full-vocab ids."""
        pos = self.keep_pos                       # (V,) full -> sliced (-1 if off)
        tgt = pos[ids[:, 1:]]                     # (B, L-1)
        lg = logits_sliced[:, :-1].reshape(-1, logits_sliced.size(-1))
        tg = tgt.reshape(-1)
        ok = tg >= 0
        if not ok.any():
            return torch.tensor(0.0, device=ids.device)
        return Fn.cross_entropy(lg[ok].float(), tg[ok])

    def l2sp_leash(self, init_sd, lam=0.01):
        """Weight-space leash to the frozen base (L2-SP): lam * mean squared
        drift of the unfrozen core params from their pretrained init values.
        init_sd: dict name -> tensor snapshot taken at training start."""
        num, den = 0.0, 0.0
        for n_, p in self.base.model.layers.named_parameters():
            if not p.requires_grad or n_ not in init_sd:
                continue
            d = (p.float() - init_sd[n_].to(p.device).float())
            num = num + d.pow(2).sum()
            den = den + init_sd[n_].to(p.device).float().pow(2).sum()
        if den <= 0:
            return torch.tensor(0.0)
        return lam * num / den

    @staticmethod
    def head_loss(head_logits, tgt):
        """tgt: dict of tensors on the same device.
        h1/h2: (B,48|8) in [0,1]; h3: (B,12) values; h4: (B,7) probs + h4c (B,) long,
        h4q (B,) value; h5: (B,) long 0..8. NaNs allowed -> masked out.
        Spec v1.2: h1 targets are TRIANGULAR-SMOOTHED realized outcomes
        (soft [0,1] - BCE-with-logits handles them); h3 uses PINBALL loss at
        its quantile levels (H3_QUANTILES) instead of smooth-L1."""
        losses = []
        for k in ("h1", "h2"):
            if tgt.get(k) is not None:
                t = tgt[k].float()
                lg = head_logits[k]
                msk = torch.isfinite(t)
                if msk.any():
                    losses.append(Fn.binary_cross_entropy_with_logits(
                        lg[msk], t[msk]))
        if tgt.get("h3") is not None:
            t = tgt["h3"].float()
            msk = torch.isfinite(t) & (t.abs() < 1e4)
            if msk.any():
                # Spec v1.2: pinball (quantile) loss. H3 layout (12):
                # [L48 MFE q25/q50/q75, L48 MAE q25/q50/q75,
                #  S48 MFE q25/q50/q75, S48 MAE q25/q50/q75]
                qs = torch.tensor([0.25, 0.5, 0.75] * 4, device=t.device,
                                  dtype=torch.float32)
                pred = Fn.softplus(head_logits["h3"])
                d = (t - pred).float()
                pin = torch.maximum(qs * d, (qs - 1.0) * d)
                losses.append((pin * msk.float()).sum() / msk.float().sum())
        if tgt.get("h4") is not None:
            t = tgt["h4"].float()
            msk = torch.isfinite(t)
            if msk.any():
                losses.append(Fn.binary_cross_entropy_with_logits(
                    head_logits["h4"][:, :4][msk], t[msk]))
        if tgt.get("h4c") is not None:
            t = tgt["h4c"].long()
            ok = (t >= 0) & (t < 3)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h4"][:, 4:7][ok], t[ok]))
        if tgt.get("h4q") is not None:
            t = tgt["h4q"].float()
            msk = torch.isfinite(t) & (t > 0) & (t < 1e4)
            if msk.any():
                losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h4"][:, 7])[msk], t[msk]))
        if tgt.get("h5") is not None:
            t = tgt["h5"].long()
            ok = (t >= 0) & (t < 9)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h5"][ok], t[ok]))
        if not losses:
            return torch.tensor(0.0, device=head_logits["h1"].device)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training utilities (fp16-strict, T4/sm_75 safe)
# ---------------------------------------------------------------------------

def make_optimizer(model, lr=1e-4, wd=0.01, betas=(0.9, 0.95)):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def load_base(device=None, dtype=torch.float32, log=print):
    from transformers import AutoModelForCausalLM
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
    if device is not None:
        base = base.to(device)
    base.eval()
    log(f"  loaded {BASE_MODEL_ID} ({sum(p.numel() for p in base.parameters())/1e6:.1f}M params, "
        f"{len(base.model.layers)} layers, d={base.config.hidden_size}, "
        f"tied={base.config.tie_word_embeddings})")
    return base


def load_tokenizer(log=print):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    log(f"  tokenizer vocab={tok.vocab_size}")
    return tok


# ==================== kernel p2_train ====================
# ============================================================================
# AURUM Phase 2/3 - FULL TRAINING KERNEL (T4 x2, ~6.5h) - Spec v1.2
#   Stage A'  : sliced-CE CPT + Z-leash + L2-SP on PRE-TEST states only
#               (year <= 2022, Spec v1.2 sec 6.3 hygiene), random loop
#               supervision r in {1,2,3}, time-capped 2 epochs (~4.5h),
#               checkpoint+resume. Gates: loss reduction >= 30%, 0 NaN,
#               clamp < 5%, Z_kept/Z_full > 0.90.
#   Stage B   : joint 5-head SFT (H1 smoothed-realized BCE + cohort-aux
#               kill-switch lambda in {0.3, 0.0} dual-branch at Checkpoint 1,
#               H2 BCE, H3 pinball, H4 BCE/CE/reg, H5 CE), variance discipline
#               (width 256, dropout 0.1, wd 0.05, early-stop on realized AUC),
#               r=3 fixed. Readout embeddings + heads trained through-core.
#   Cache     : no-grad r=3 pass over ALL labeled states (pre-test + OOS)
#               caching the 5 readout-position hiddens (~60 rows/s) ->
#               fold refits + temperature scaling become near-free.
#   Refits    : walk-forward folds 2016..2025, heads-only (MLPs) warm-start,
#               strictly causal <= y-1 on cached hiddens; per-fold per-head
#               temperature scaling on a purged calibration split.
#   Export    : models/ (core layers, sandwich norms, readout, main + fold
#               heads, temperatures, schedule, metrics, gates) -> versioned
#               as private dataset tskdev/aurum-models-v1.
#   Self-resume: attach this kernel's previous output as kernel_sources;
#               schedule.json + ckpt_latest.pt drive stage-level resume.
# ============================================================================
import argparse, json, os, shutil, subprocess, sys, time
import numpy as np

TRAIN_END_YEAR = 2022              # hygiene: Stage A' + main Stage B pre-test only
SEQ = 512
LAM_Z, LAM_W = 1.0, 0.01
A_PRIME_TOKENS = 40_000_000        # 2 epochs x ~20M pre-test tokens
A_PRIME_TIME_S = 4.6 * 3600        # hard cap (checkpoint whichever comes first)
CKPT_EVERY_S = 20 * 60
R_CHOICES = (1, 2, 3)              # random loop supervision
# Stage B
SB_STATES = 8000                   # per-segment subsample (stratified by year)
SB_SEG1_EPOCHS, SB_SEG2_EPOCHS = 2, 2
LAMBDA_AUX = 0.3                   # cohort-aux (kill-switch branch set)
HEAD_LR, HEAD_WD, HEAD_DROPOUT = 3e-4, 0.05, 0.1
# folds
FOLD_YEARS = list(range(2016, 2026))
FOLD_STATES = 6000                 # per-fold head-refit subsample (<= y-1)
CAL_FRAC = 0.2                     # purged calibration tail of each fold window

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="main")
    return p.parse_known_args()[0]


def _find_ds():
    """Locate the phase-1 dataset mount robustly (mount names can differ;
    logs the /kaggle/input tree when the expected path is missing)."""
    import glob as _glob
    cands = []
    if os.environ.get("AURUM_DS"):
        cands.append(os.environ["AURUM_DS"])
    cands += ["/kaggle/input/datasets/tskdev/aurum-phase1-data",
              "/kaggle/input/aurum-phase1-data"]
    cands += sorted(_glob.glob("/kaggle/input/datasets/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/*/aurum-phase1-data*"))
    cands += sorted(_glob.glob("/kaggle/input/aurum*"))
    for c in cands:
        if os.path.isfile(os.path.join(c, "h1_features.parquet")):
            return c
    try:
        import subprocess as _sp
        tree = _sp.run(["find", "/kaggle/input", "-maxdepth", "3"],
                       capture_output=True, text=True, timeout=20).stdout
        print("[_find_ds] DATASET NOT FOUND. /kaggle/input tree:\n" + tree[:3000],
              flush=True)
    except Exception:
        pass
    return cands[0]

def _ds():
    DS = "/kaggle/input/aurum-phase1-data"
    if os.path.isfile(os.path.join(DS, "h1_features.parquet")):
        return DS
    found = _find_ds()
    return found if found != DS else os.environ.get("AURUM_DS", DS)

def _trend_volc(h1):
    a = wilder_atr(h1["h"], h1["l"], h1["c"], 14)
    r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
    t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
    t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
    trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
    trend[~np.isfinite(t33.values) | ~np.isfinite(r24.values)] = np.nan
    ar = a / a.shift(48)
    v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
    v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
    volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
    volc[~np.isfinite(v33.values) | ~np.isfinite(ar.values)] = np.nan
    return trend, volc

def _ser_drop(DS):
    drop = {"m_dxy", "m_y10", "m_y2", "m_spread"}
    try:
        ss = json.load(open(os.path.join(DS, "state_stats.json")))
        drop |= set(ss.get("state_drop", []))
    except Exception:
        pass
    return drop

# ----------------------------------------------------------------------------
# data prep (parent): all labeled states tokenized once; ragged pools saved
# ----------------------------------------------------------------------------

def prepare_data(WORK):
    import pandas as pd
    cache = os.path.join(WORK, "p2_train_data.npz")
    meta_f = os.path.join(WORK, "p2_train_data_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_f):
        log("p2 data: using cache")
        return cache, json.load(open(meta_f))

    DS = _ds()
    feat = pd.read_parquet(os.path.join(DS, "h1_features.parquet"))
    if "ts" in feat.columns:
        feat = feat.drop(columns=["ts"])
    h1 = pd.read_parquet(os.path.join(DS, "h1_xauusd.parquet"))
    h1 = h1[~h1.index.duplicated()].sort_index()
    ts = feat.index
    trend, volc = _trend_volc(h1)
    ser_drop = _ser_drop(DS)

    R = pd.read_parquet(os.path.join(DS, "labels_R_actual.parquet"))
    cls_ = pd.read_parquet(os.path.join(DS, "labels_cls_actual.parquet"))
    W_real = (R.values > 0).astype(np.float64)
    W_real[~np.isfinite(R.values)] = np.nan
    valid_R = np.isfinite(R.values).all(1) & (cls_.values != 255).all(1)
    cols = [c for c in COHORT_FEATURES if c in feat.columns]
    label_valid = valid_R.copy()
    label_valid &= feat[cols].notna().all(axis=1).values
    label_valid &= np.isfinite(feat[cols].values).all(1)
    label_valid[:400] = False

    with stage("serialize+tokenize (all labeled states)"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        idx_all = np.flatnonzero(label_valid)
        B = 2000
        ids_concat, lens = [], []
        for s in range(0, len(idx_all), B):
            rows = idx_all[s:s + B]
            t_batch = [serialize_state(ts[i], feat.iloc[i], with_cross=True,
                                       with_macro=True, with_priors=False,
                                       with_minnov=True, with_zpriors=True,
                                       with_cohort=True, trend3=trend[i],
                                       vol3=volc[i], drop=ser_drop, compact=True)
                       for i in rows]
            enc = tok(t_batch, add_special_tokens=False)["input_ids"]
            for e_ in enc:
                ids_concat.append(np.asarray(e_, dtype=np.int32))
                lens.append(len(e_))
        lens = np.array(lens, dtype=np.int32)
        lab_flat = np.concatenate(ids_concat)
        years = np.asarray(ts[idx_all].year)

        # Stage A' stream: contiguous pre-test states (includes unlabeled bars
        # with valid serialization? spec: 'states' - use labeled pre-test set
        # + head-invisible unlabeled is unnecessary; labeled covers 2012-2022)
        pre = years <= TRAIN_END_YEAR
        ends = np.cumsum(lens); starts = ends - lens
        pre_flat = np.concatenate([lab_flat[starts[i]:ends[i]]
                                   for i in np.flatnonzero(pre)])
        meta = {"n_labeled_states": int(len(idx_all)),
                "n_pretest": int(pre.sum()), "n_oos": int((~pre).sum()),
                "pretest_tokens": int(pre_flat.size),
                "mean_tokens": round(float(lens.mean()), 1),
                "min_tokens": int(lens.min()), "max_tokens": int(lens.max())}
        log(f"p2 data: {json.dumps(meta)}")

        sm = pd.read_parquet(os.path.join(DS, "labels_cls_smoothed.parquet")).values
        aux = pd.read_parquet(os.path.join(DS, "labels_aux.parquet"))
        pw = pd.read_parquet(os.path.join(DS, "labels_pwin.parquet")).values
        try:
            exc = pd.read_parquet(os.path.join(DS, "labels_excursions.parquet"))
            h3 = np.repeat(exc[["mfe_L48", "mae_L48", "mfe_S48", "mae_S48"]].values, 3, 1)
        except Exception:
            q = lambda pre_, suf_: np.stack([aux[f"{pre_}{q_}{suf_}"].values
                                             for q_ in (25, 50, 75)], 1)
            h3 = np.concatenate([q("mfe_q", "_48"), q("mae_q", "_48"),
                                 q("mfe_q", "_48S"), q("mae_q", "_48S")], 1)
        np.savez_compressed(
            cache, stream_flat=pre_flat.astype(np.int32),
            lab_flat=lab_flat.astype(np.int32), lab_lens=lens,
            lab_idx=idx_all, lab_years=years.astype(np.int16),
            sm_h1=sm.astype(np.float32), pwin=pw.astype(np.float32),
            win_real=W_real.astype(np.float32), R_real=R.values.astype(np.float32),
            h2up=aux[["h2up4","h2up12","h2up24","h2up48"]].values.astype(np.float32),
            h2big=aux[["h2big4","h2big12","h2big24","h2big48"]].values.astype(np.float32),
            h4exp=aux[["h4exp4","h4exp12","h4exp24","h4exp48"]].values.astype(np.float32),
            h4terc=np.nan_to_num(aux["h4terc12"].values.astype(float), nan=-1).astype(np.int64),
            h4q=np.nan_to_num(aux["h4q50_12"].values.astype(float), nan=-1).astype(np.float32),
            h5=np.nan_to_num(aux["h5regime"].values.astype(float), nan=-1).astype(np.int64),
            h3=h3.astype(np.float32))
    with open(meta_f, "w") as f:
        json.dump(meta, f, indent=1)
    return cache, meta

# ----------------------------------------------------------------------------
# head-stack helpers (dual-branch Stage B + fold refits share these)
# ----------------------------------------------------------------------------

def _mk_head_module(model, dropout):
    """Fresh readout+heads stack with variance discipline, on the same device
    (core shared, NOT part of this module)."""
    import torch.nn as nn
    d = model.base.config.hidden_size
    class HeadStack(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = nn.Embedding(5, d)
            nn.init.normal_(self.readout.weight, std=0.02)
            def mlp(out):
                return nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(256, out))
            self.h1, self.h2, self.h3 = mlp(48), mlp(8), mlp(12)
            self.h4, self.h5 = mlp(8), mlp(9)
        def forward(self, ro_hidden):
            return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                    "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                    "h5": self.h5(ro_hidden[:, 4])}
    return HeadStack()

def _head_targets(z, rows, dev):
    import torch
    def imp(arr, na):
        v = arr[rows]
        return np.where(np.isfinite(v), v, na)
    return {
        "h1": torch.from_numpy(imp(z["sm_h1"], 0.5)).float().to(dev),
        "h1_aux": torch.from_numpy(imp(z["pwin"], 0.5)).float().to(dev),
        "h2": torch.from_numpy(np.concatenate(
            [imp(z["h2up"], 0.5), imp(z["h2big"], 0.5)], 1)).float().to(dev),
        "h3": torch.from_numpy(z["h3"][rows]).float().to(dev),
        "h4": torch.from_numpy(imp(z["h4exp"], 0.5)).float().to(dev),
        "h4c": torch.from_numpy(z["h4terc"][rows]).long().to(dev),
        "h4q": torch.from_numpy(z["h4q"][rows]).float().to(dev),
        "h5": torch.from_numpy(z["h5"][rows]).long().to(dev),
    }

def _head_loss_stack(stack, head_logits, tgt, lam_aux):
    import torch
    import torch.nn.functional as Fn
    losses = []
    for k in ("h1", "h2"):
        t = tgt[k].float()
        m = torch.isfinite(t)
        if m.any():
            losses.append(Fn.binary_cross_entropy_with_logits(head_logits[k][m], t[m]))
    if lam_aux > 0:
        t = tgt["h1_aux"].float()
        m = torch.isfinite(t)
        if m.any():
            losses.append(lam_aux * Fn.binary_cross_entropy_with_logits(
                head_logits["h1"][m], t[m]))
    t = tgt["h3"].float()
    m = torch.isfinite(t) & (t.abs() < 1e4)
    if m.any():
        qs = torch.tensor([0.25, 0.5, 0.75] * 4, device=t.device, dtype=torch.float32)
        pred = Fn.softplus(head_logits["h3"])
        d = (t - pred).float()
        losses.append((torch.maximum(qs * d, (qs - 1) * d) * m.float()).sum() / m.float().sum())
    t = tgt["h4"].float()
    m = torch.isfinite(t)
    if m.any():
        losses.append(Fn.binary_cross_entropy_with_logits(head_logits["h4"][:, :4][m], t[m]))
    t = tgt["h4c"].long(); ok = (t >= 0) & (t < 3)
    if ok.any():
        losses.append(Fn.cross_entropy(head_logits["h4"][:, 4:7][ok], t[ok]))
    if tgt.get("h4q") is not None:
        t = tgt["h4q"].float()
        m = torch.isfinite(t) & (t > 0) & (t < 1e4)
        if m.any():
            losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h4"][:, 7])[m], t[m]))
    t = tgt["h5"].long(); ok = (t >= 0) & (t < 9)
    if ok.any():
        losses.append(Fn.cross_entropy(head_logits["h5"][ok], t[ok]))
    if not losses:
        return torch.tensor(0.0)
    return torch.stack(losses).mean()

def _h1_metrics(stack, P_h1, z, rows):
    """realized + smoothed Brier/AUC for the H1 head on given rows."""
    from sklearn.metrics import roc_auc_score
    T_sm = z["sm_h1"][rows]; T_re = z["win_real"][rows]
    m = np.isfinite(T_sm) & np.isfinite(P_h1)
    out = {"brier_sm": float(np.mean((P_h1[m] - T_sm[m]) ** 2)) if m.any() else None}
    mr = np.isfinite(T_re) & np.isfinite(P_h1)
    if mr.any() and len(np.unique(T_re[mr])) == 2:
        out["brier_re"] = float(np.mean((P_h1[mr] - T_re[mr]) ** 2))
        out["auc_re"] = float(roc_auc_score(T_re[mr], P_h1[mr]))
    return out

# ----------------------------------------------------------------------------
# training child (torchrun 2xT4): A' -> B -> cache -> refits -> export
# ----------------------------------------------------------------------------

def run_train_child(WORK):
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.nn.functional as Fn
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    has_cuda = torch.cuda.is_available()
    if world > 1:
        dist.init_process_group("nccl" if has_cuda else "gloo")
    if has_cuda:
        torch.cuda.set_device(local_rank)
        dev = torch.device(f"cuda:{local_rank}")
    else:
        dev = torch.device("cpu")
    z = np.load(os.path.join(WORK, "p2_train_data.npz"))
    keep_ids = np.load(os.path.join(_ds(), "vocab_keep_ids.npy"))
    MODELS = os.path.join(WORK, "models")
    os.makedirs(MODELS, exist_ok=True)
    # self-resume source (kernel output attach; mount name may be either
    # /kaggle/input/aurum-p2-train or /kaggle/input/kernels/tskdev/aurum-p2-train)
    PREV = None
    for _c in ("/kaggle/input/aurum-p2-train/models",
               "/kaggle/input/kernels/tskdev/aurum-p2-train/models",
               "/kaggle/input/aurum-p2-train"):
        if os.path.isdir(_c):
            PREV = _c
            break
    PREV = PREV or "/kaggle/input/aurum-p2-train/models"
    sched_f = os.path.join(MODELS, "schedule.json")
    sched = {"stage": "A", "a_tokens": 0, "done": []}
    # schedule loads from the working dir OR the previous session's output
    # (the resume source); without this, a resumed run would mistake itself
    # for a fresh start and crash on empty Stage-A statistics (v3 lesson).
    for _sf in (sched_f, os.path.join(PREV, "schedule.json")):
        if os.path.exists(_sf):
            sched = json.load(open(_sf))
            log(f"resume: schedule from {_sf}: {json.dumps(sched)}")
            break

    base = load_base(dev, log=lambda *a: None)
    model = LoopedQwen(base, n_loop=3, sandwich=True, grad_ckpt=True).to(dev)
    model.set_trainable()
    model.set_keep_ids(keep_ids)
    init_sd = {n_: p.detach().clone() for n_, p in model.base.model.layers.named_parameters()
               if p.requires_grad}
    ddp = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                              find_unused_parameters=True) if (world > 1) else model
    opt = make_optimizer(model, lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=has_cuda)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    flat = torch.from_numpy(z["stream_flat"].astype(np.int64))
    n_blocks = flat.numel() // SEQ
    blocks = flat[: n_blocks * SEQ].view(n_blocks, SEQ)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(n_blocks, generator=g)
    lab_flat = torch.from_numpy(z["lab_flat"].astype(np.int64))
    lab_lens = z["lab_lens"]
    ends = np.cumsum(lab_lens); starts = ends - lab_lens
    lab_idx, lab_years = z["lab_idx"], z["lab_years"]
    BS = 16

    def save_ckpt(tag):
        if rank != 0:
            return
        torch.save({"layers": {f"{i}": model.base.model.layers[i].state_dict()
                               for i in range(CORE_START, CORE_END)},
                    "pre": model.pre_norms.state_dict(),
                    "post": model.post_norms.state_dict(),
                    "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                    "a_tokens": sched["a_tokens"]},
                   os.path.join(MODELS, f"ckpt_{tag}.pt"))
        if tag != "latest":
            shutil.copy(os.path.join(MODELS, f"ckpt_{tag}.pt"),
                        os.path.join(MODELS, "ckpt_latest.pt"))

    def load_ckpt():
        p_ = os.path.join(MODELS, "ckpt_latest.pt")
        p_prev = os.path.join(PREV, "ckpt_latest.pt")
        src = p_ if os.path.exists(p_) else (p_prev if os.path.exists(p_prev) else None)
        if src is None:
            return False
        ck = torch.load(src, map_location=dev, weights_only=False)
        for i in range(CORE_START, CORE_END):
            model.base.model.layers[i].load_state_dict(ck["layers"][f"{i}"])
        model.pre_norms.load_state_dict(ck["pre"])
        model.post_norms.load_state_dict(ck["post"])
        if sched["stage"] == "A":
            opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
            sched["a_tokens"] = ck.get("a_tokens", 0)
        log(f"checkpoint loaded from {src} (a_tokens={sched['a_tokens']})")
        return True

    # ================= Stage A': sliced-CE CPT (pre-test only) ==============
    a_result = {}
    if "A" not in sched["done"]:
        load_ckpt()
        resumed_a = sched["a_tokens"] > 0
        model.train()
        rng_r = np.random.default_rng(11)
        losses, z_traj = [], []
        nan_skips = 0
        t_start = time.time()
        t_ckpt = time.time()
        lm_wall = 0.0
        lm_steps = int(sched["a_tokens"] // (BS * SEQ * world))
        arng = torch.arange(BS)
        tokens_done = 0
        while sched["a_tokens"] + tokens_done < A_PRIME_TOKENS and \
                time.time() - t_start < A_PRIME_TIME_S:
            r = int(rng_r.choice(R_CHOICES))
            model.n_loop = r
            sel = perm[((lm_steps * BS * world + rank * BS + arng) % n_blocks)]
            ids = blocks[sel].to(dev)
            opt.zero_grad(set_to_none=True)
            _t = time.time()
            with autocast_ctx(dev):
                o = ddp(ids, add_readout=False, lm_output="sliced", z_sample=256)
                ce = model.sliced_lm_loss(o["logits_sliced"], ids)
                zt = o["z"]
                loss = ce + LAM_Z * (1.0 - zt) + model.l2sp_leash(init_sd, LAM_W)
            if not torch.isfinite(loss):
                nan_skips += 1
                opt.zero_grad(set_to_none=True)
                lm_steps += 1
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(opt); scaler.update()
            lm_steps += 1
            tokens_done += BS * SEQ * world
            lm_wall += time.time() - _t
            losses.append(float(ce)); z_traj.append(float(zt))
            if lm_steps % 50 == 0 and rank == 0:
                log(f"  A' step {lm_steps} r={r} ce={float(ce):.4f} z={float(zt):.4f} "
                    f"gn={float(gn):.2f} mem={torch.cuda.max_memory_allocated()/1e9 if has_cuda else 0:.1f}GB")
            if time.time() - t_ckpt > CKPT_EVERY_S:
                sched["a_tokens"] += tokens_done; tokens_done = 0
                save_ckpt("latest")
                with open(sched_f, "w") as f:
                    json.dump(sched, f)
                t_ckpt = time.time()
        sched["a_tokens"] += tokens_done
        model.n_loop = 3
        # final Z on a fresh held-out slice of the stream tail
        with torch.no_grad():
            model.eval()
            zs = []
            for s0 in range(0, min(64, n_blocks), 8):
                ids = blocks[(np.arange(s0, s0 + 8)) % n_blocks].to(dev)
                with autocast_ctx(dev):
                    o = model(ids, add_readout=False, lm_output="sliced", z_sample=2048)
                zs.append(float(o["z"]))
            model.train()
        z_final = float(np.mean(zs))
        last10 = float(np.mean(losses[-max(1, len(losses)//10):]))
        first10 = float(np.mean(losses[:max(1, len(losses)//10)]))
        # On a RESUMED session the in-session first10 understates the true
        # reduction (earlier sessions' losses are not in this array). Use the
        # p2a-measured fresh-model initial sliced-CE (1.369, same stream and
        # init) as the documented reference start when resuming.
        A_INIT_LOSS = 1.369
        ref_start = min(first10, A_INIT_LOSS) if resumed_a else first10
        red = 1 - last10 / max(ref_start, 1e-9)
        a_result = {"tokens": int(sched["a_tokens"]), "lm_steps": lm_steps,
                    "loss_first10": round(first10, 4),
                    "ref_start": round(ref_start, 4),
                    "loss_last10": round(float(np.mean(losses[-max(1, len(losses)//10):])), 4),
                    "loss_reduction": round(red, 4), "z_final": round(z_final, 4),
                    "nan_skips": nan_skips,
                    "clamp_rate": round(model.stats.get("clamp_count", 0.0) /
                                        max(model.stats.get("clamp_total", 1.0), 1.0), 5),
                    "resid_max": round(model.stats.get("resid_max", 0.0), 1),
                    "tok_s_ddp": round((sched["a_tokens"]) / max(lm_wall, 1e-9), 0),
                    "final_scale": float(scaler.get_scale()), "world": world}
        if rank == 0:
            with open(os.path.join(MODELS, "stageA_result.json"), "w") as f:
                json.dump(a_result, f, indent=1)
            np.save(os.path.join(MODELS, "stageA_losses.npy"), np.array(losses))
            np.save(os.path.join(MODELS, "stageA_z.npy"), np.array(z_traj))
            log(f"STAGE A' RESULT {json.dumps(a_result)}")
        sched["done"].append("A")
        sched["stage"] = "B"
        save_ckpt("A_end")
        with open(sched_f, "w") as f:
            json.dump(sched, f)
    else:
        load_ckpt()
        try:
            a_result = json.load(open(os.path.join(MODELS, "stageA_result.json")))
        except Exception:
            try:
                a_result = json.load(open(os.path.join(PREV, "stageA_result.json")))
            except Exception:
                a_result = {}
        # v2 wrote the understated in-session reduction (0.1541) because that
        # session resumed mid-Stage-A'; re-derive the honest full-run number
        # against the documented fresh-model reference when applicable.
        A_INIT_LOSS = 1.369
        if a_result and "loss_last10" in a_result:
            honest = 1 - a_result["loss_last10"] / A_INIT_LOSS
            if honest > a_result.get("loss_reduction", 0) and \
                    a_result.get("loss_first10", 9) > A_INIT_LOSS * 0.75:
                a_result["ref_start"] = A_INIT_LOSS
                a_result["loss_reduction"] = round(honest, 4)
            elif a_result.get("ref_start") is None:
                a_result["ref_start"] = a_result.get("loss_first10")
            if rank == 0:
                with open(os.path.join(MODELS, "stageA_result.json"), "w") as f:
                    json.dump(a_result, f, indent=1)
                log(f"stageA_result (resume-normalized): {json.dumps(a_result)}")

    # ================= Stage B: joint-head SFT (dual-branch kill-switch) ====
    sb_result = {}
    _heads_path = os.path.join(MODELS, "heads_main.pt")
    _heads_prev = os.path.join(PREV, "heads_main.pt")
    _B_done = ("B" in sched["done"]) and (os.path.exists(_heads_path) or
                                          os.path.exists(_heads_prev))
    if not _B_done:
        model.n_loop = 3
        # heads-only: freeze the core + norms + original head for Stage B
        for p_ in model.parameters():
            p_.requires_grad_(False)
        model.train()          # keep dropout/etc consistent; no core grads
        pre_pos = np.flatnonzero(lab_years <= TRAIN_END_YEAR)
        rng = np.random.default_rng(21)
        strat = []
        for yv in np.unique(lab_years[pre_pos]):
            rows_y = pre_pos[lab_years[pre_pos] == yv]
            strat.append(rng.choice(rows_y, size=min(len(rows_y),
                                 max(60, int(SB_STATES * len(rows_y) / len(pre_pos)))),
                         replace=False))
        sb_train = np.sort(np.concatenate(strat))[:SB_STATES]
        last_year = int(lab_years[pre_pos].max())
        val_pos = np.flatnonzero((lab_years == last_year))
        val_pos = np.setdiff1d(val_pos, sb_train)

        # branch stacks: raw module + DDP wrapper + optimizer per branch.
        # DDP gradient hooks are PER-PARAMETER (path-independent), so the
        # readout embedding's grads arriving via the CORE forward are still
        # all-reduced by the stack's own DDP wrapper. The frozen core is
        # forwarded through the RAW model (nothing to sync).
        stacks = {}
        for k_ in ("lam03", "lam00"):
            st_ = _mk_head_module(model, HEAD_DROPOUT).to(dev)
            fwd = (nn.parallel.DistributedDataParallel(st_, device_ids=[local_rank])
                   if world > 1 else st_)
            stacks[k_] = {"m": st_, "fwd": fwd,
                          "opt": torch.optim.AdamW(st_.parameters(), lr=HEAD_LR,
                                                   weight_decay=HEAD_WD),
                          "hist": []}

        nan_skips_b = {"n": 0}

        def head_step(k, rows, ids, pm, train=True):
            st_ = stacks[k]["m"]
            st_.train(train)
            tgt = _head_targets(z, rows, dev)
            optb = stacks[k]["opt"]
            optb.zero_grad(set_to_none=True)
            import contextlib
            _fp32 = stacks[k].get("fp32", False)
            with (contextlib.nullcontext() if _fp32 else autocast_ctx(dev)):
                o = model(ids, add_readout=True, lm_output=False, pad_mask=pm,
                          ro_embed=st_.readout.weight)
                logits = stacks[k]["fwd"](o["ro_h"])
                loss = _head_loss_stack(st_, logits, tgt,
                                        LAMBDA_AUX if k == "lam03" else 0.0)
            if not torch.isfinite(loss):
                # NaN watchdog: diagnose (armored) + skip; never backprop NaN.
                nan_skips_b["n"] += 1
                nan_skips_b["total"] = nan_skips_b.get("total", 0) + 1
                try:
                    with torch.no_grad():
                        comps = {}
                        for kk in ("h1", "h2", "h3", "h4", "h5"):
                            t_ = tgt.get(kk)
                            l_ = logits.get(kk)
                            if t_ is None or l_ is None:
                                continue
                            if kk == "h4":
                                l4, t4 = l_.float()[:, :4], t_.float()
                                m4 = torch.isfinite(t4)
                                if m4.any():
                                    comps[kk] = round(float(
                                        Fn.binary_cross_entropy_with_logits(l4[m4], t4[m4])), 4)
                            elif kk == "h5":
                                t5 = t_.long()
                                ok5 = (t5 >= 0) & (t5 < 9)
                                if ok5.any():
                                    comps[kk] = round(float(
                                        Fn.cross_entropy(l_.float()[ok5], t5[ok5])), 4)
                            else:
                                mm = torch.isfinite(t_.float()) & torch.isfinite(l_.float())
                                if mm.any():
                                    if kk in ("h1", "h2"):
                                        comps[kk] = round(float(
                                            Fn.binary_cross_entropy_with_logits(
                                                l_.float()[mm], t_.float()[mm])), 4)
                                    else:
                                        comps[kk] = round(float(l_.float()[mm].mean()), 4)
                        ro_ok = bool(torch.isfinite(o["ro_h"]).all())
                        _lmax = max(float(v_.abs().max()) for v_ in logits.values())
                        if rank == 0 and nan_skips_b["n"] <= 8:
                            log(f"  B nan-skip #{nan_skips_b['n']} k={k}: per-head={comps} "
                                f"ro_h_finite={ro_ok} "
                                f"ro_h_max={float(o['ro_h'].abs().max()):.1f} "
                                f"logit_max={_lmax:.1f}")
                except Exception as _diag:
                    if rank == 0:
                        log(f"  B nan-skip diag error (ignored): {_diag}")
                optb.zero_grad(set_to_none=True)
                nan_skips_b["steps"] = nan_skips_b.get("steps", 0) + 1
                if nan_skips_b["steps"] % 100 == 0 and \
                        nan_skips_b["total"] / max(nan_skips_b["steps"], 1) > 0.15:
                    stacks[k]["fp32"] = True
                    if rank == 0:
                        log(f"  B fp32 fallback engaged for {k}")
                return None
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(st_.parameters(), 1.0)
                optb.step()
            stacks[k]["hist"].append(float(loss.detach()))
            return logits

        def make_batch(rows_sel):
            seqs = [lab_flat[starts[i]:ends[i]] for i in rows_sel]
            L = max(len(s_) for s_ in seqs)
            ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
            pm = torch.zeros((len(seqs), L), dtype=torch.bool)
            for i, s_ in enumerate(seqs):
                ids[i, :len(s_)] = s_; pm[i, :len(s_)] = True
            return ids.to(dev), pm.to(dev)

        def train_segment(seg_name, n_epochs, branches):
            steps_per_ep = max(1, len(sb_train) // BS)
            for ep in range(n_epochs):
                for _ in range(steps_per_ep):
                    sel = rng.choice(len(sb_train), size=BS, replace=True)
                    rows = sb_train[sel]
                    ids, pm = make_batch(rows)
                    for k in branches:
                        head_step(k, rows, ids, pm, train=True)
                if rank == 0:
                    log(f"  B {seg_name} ep{ep}: " + " ".join(
                        f"{k}={np.mean(stacks[k]['hist'][-50:]):.4f}" for k in branches))

        def eval_branch(k):
            st_ = stacks[k]["m"]
            st_.eval()
            Ps = []
            with torch.no_grad():
                for s0 in range(0, len(val_pos), 32):
                    rows = val_pos[s0:s0 + 32]
                    ids, pm = make_batch(rows)
                    with autocast_ctx(dev):
                        o = model(ids, add_readout=True, lm_output=False,
                                  pad_mask=pm, ro_embed=st_.readout.weight)
                        logits = st_(o["ro_h"])
                    Ps.append(torch.sigmoid(logits["h1"]).float().cpu().numpy())
            P = np.concatenate(Ps)
            return _h1_metrics(st_, P, z, val_pos), P

        train_segment("seg1", SB_SEG1_EPOCHS, ["lam03", "lam00"])
        m03, _ = eval_branch("lam03")
        m00, _ = eval_branch("lam00")
        a03, a00 = m03.get("auc_re") or 0.0, m00.get("auc_re") or 0.0
        better = "lam03" if a03 >= a00 else "lam00"
        kill_switch = better == "lam00"
        if rank == 0:
            log(f"CHECKPOINT-1 kill-switch: lam03 AUC={a03:.4f} vs lam00 AUC={a00:.4f} "
                f"-> keep {better} (switch triggered: {kill_switch})")
        ckpt1_sd = {k: v.detach().clone() for k, v in stacks[better]["m"].state_dict().items()}
        del stacks["lam00" if better == "lam03" else "lam03"]
        train_segment("seg2", SB_SEG2_EPOCHS, [better])
        m_final, _ = eval_branch(better)
        # early stop on realized val AUC: revert to ckpt-1 if seg2 degraded
        early_stop = False
        if (m_final.get("auc_re") or 0.0) < (m03 if better == "lam03" else m00).get("auc_re", 0.0):
            early_stop = True
            stacks[better]["m"].load_state_dict(ckpt1_sd)
            m_final, _ = eval_branch(better)
            if rank == 0:
                log(f"  early-stop: seg2 degraded val AUC -> reverted to ckpt-1")
        sb_result = {"kill_switch_triggered": kill_switch, "kept_branch": better,
                     "nan_skips_b": nan_skips_b.get("total", 0),
                     "fp32_fallback": {k_: stacks[k_].get("fp32", False) for k_ in stacks},
                     "ckpt1_lam03": m03, "ckpt1_lam00": m00, "final_val": m_final,
                     "early_stop_reverted": bool(early_stop), "world": world,
                     "n_sb_train": int(len(sb_train)), "n_val": int(len(val_pos))}
        if rank == 0:
            with open(os.path.join(MODELS, "stageB_result.json"), "w") as f:
                json.dump(sb_result, f, indent=1)
            torch.save(stacks[better]["m"].state_dict(),
                       os.path.join(MODELS, "heads_main.pt"))
            log(f"STAGE B RESULT {json.dumps(sb_result)}")
        sched["done"].append("B")
        sched["stage"] = "cache"
        if rank == 0:
            with open(sched_f, "w") as f:
                json.dump(sched, f)
        if world > 1:
            dist.barrier()      # rank1 waits for rank0's heads_main.pt write
    else:
        try:
            sb_result = json.load(open(os.path.join(MODELS, "stageB_result.json")))
        except Exception:
            sb_result = {}

    # ================= Cache: readout hiddens for ALL labeled states ========
    _cache_done = ("cache" in sched["done"]) and any(
        os.path.exists(os.path.join(MODELS, f"ro_hiddens_shard{r_}.npy")) or
        os.path.exists(os.path.join(PREV, f"ro_hiddens_shard{r_}.npy")) or
        os.path.exists(os.path.join(MODELS, "ro_hiddens.npy"))
        for r_ in range(world))
    if not _cache_done:
        model.n_loop = 3
        model.eval()
        sd_path = os.path.join(MODELS, "heads_main.pt")
        if not os.path.exists(sd_path):
            sd_path = os.path.join(PREV, "heads_main.pt")
        main_sd = torch.load(sd_path, map_location=dev)
        ro_w = main_sd["readout.weight"].to(dev)
        N_lab = len(lab_idx)
        my_rows = np.arange(rank, N_lab, world)          # shard across ranks
        H5 = np.zeros((len(my_rows), 5, model.base.config.hidden_size), dtype=np.float16)
        t0 = time.time()
        bs_c = 32
        with torch.no_grad():
            for s0 in range(0, len(my_rows), bs_c):
                rows = my_rows[s0: s0 + bs_c]
                seqs = [lab_flat[starts[i]:ends[i]] for i in rows]
                L = max(len(s_) for s_ in seqs)
                ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
                pm = torch.zeros((len(seqs), L), dtype=torch.bool)
                for i, s_ in enumerate(seqs):
                    ids[i, :len(s_)] = s_; pm[i, :len(s_)] = True
                with autocast_ctx(dev):
                    o = model(ids.to(dev), add_readout=True, lm_output=False,
                              pad_mask=pm.to(dev), ro_embed=ro_w)
                H5[s0:s0 + len(rows)] = o["ro_h"].half().cpu().numpy()
                if rank == 0 and s0 % (6400 // world) == 0:
                    rps = (s0 + len(rows)) * world / (time.time() - t0)
                    log(f"  cache ~{s0 * world}/{N_lab} ({rps:.1f} rows/s agg)")
        np.save(os.path.join(MODELS, f"ro_hiddens_shard{rank}.npy"), H5)
        if world > 1:
            dist.barrier()
        if rank == 0:
            # fold refits + export are rank-0 work; shards are read lazily there
            log(f"cache done: {N_lab} states sharded across {world} ranks "
                f"in {time.time()-t0:.0f}s")
        sched["done"].append("cache")
        sched["stage"] = "refits"
        with open(sched_f, "w") as f:
            json.dump(sched, f)
        if world > 1:
            dist.barrier()

    def _load_h5():
        """Interleave rank shards into the full (N_lab, 5, d) cache."""
        shards = []
        for r_ in range(world):
            p_ = os.path.join(MODELS, f"ro_hiddens_shard{r_}.npy")
            if not os.path.exists(p_):
                p_ = os.path.join(PREV, f"ro_hiddens_shard{r_}.npy")
            shards.append(np.load(p_))
        if world == 1:
            return shards[0]
        N_lab = len(lab_idx)
        out = np.zeros((N_lab, *shards[0].shape[1:]), dtype=shards[0].dtype)
        for r_ in range(world):
            out[r_::world] = shards[r_]
        return out

    # ================= Walk-forward fold refits + temperature scaling =======
    if rank == 0 and "refits" not in sched["done"]:
        main_sd = torch.load(os.path.join(MODELS, "heads_main.pt")
                             if os.path.exists(os.path.join(MODELS, "heads_main.pt"))
                             else os.path.join(PREV, "heads_main.pt"), map_location="cpu")
        H5c = _load_h5()
        rng = np.random.default_rng(31)
        fold_res, temps = {}, {}

        def fit_T(logits, t):
            Ts = {}
            best, bestT = None, 1.0
            tt = torch.nan_to_num(t["h1"], nan=0.5)
            for T in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                pp = torch.sigmoid(logits["h1"] / T).clamp(1e-6, 1 - 1e-6)
                nll = Fn.binary_cross_entropy(pp, tt).item()
                if best is None or nll < best:
                    best, bestT = nll, T
            Ts["h1"] = bestT
            best, bestT = None, 1.0
            ok = (t["h5"] >= 0)
            if ok.any():
                for T in (0.5, 0.75, 1.0, 1.5):
                    nll = Fn.cross_entropy(logits["h5"][ok] / T, t["h5"][ok]).item()
                    if best is None or nll < best:
                        best, bestT = nll, T
            Ts["h5"] = bestT
            return Ts

        for yv in FOLD_YEARS:
            fold_train_pos = np.flatnonzero(lab_years <= yv - 1)
            if len(fold_train_pos) < 2000:
                fold_res[str(yv)] = {"skipped": "insufficient pre-year data"}
                continue
            n_cal = max(200, int(len(fold_train_pos) * CAL_FRAC))
            cal_pos = fold_train_pos[-n_cal:]
            tr_pos = fold_train_pos[:-n_cal - 96]        # 96-state purge gap
            tr_pos = rng.choice(tr_pos, size=min(FOLD_STATES, len(tr_pos)),
                                replace=False)
            st_ = _mk_head_module(model, HEAD_DROPOUT)
            st_.load_state_dict(main_sd)
            st_.readout.weight.requires_grad_(False)     # readout frozen in refits
            opt_h = torch.optim.AdamW([p_ for p in st_.parameters() if p.requires_grad],
                                      lr=HEAD_LR, weight_decay=HEAD_WD)
            Xtr = torch.from_numpy(H5c[tr_pos].astype(np.float32))
            Xcal = torch.from_numpy(H5c[cal_pos].astype(np.float32))
            st_.train()
            for ep in range(3):
                perm_ = torch.randperm(len(tr_pos))
                for s0 in range(0, len(tr_pos), 64):
                    sel = perm_[s0:s0 + 64]
                    rows = tr_pos[sel.numpy()]
                    tgt = _head_targets(z, rows, "cpu")
                    logits = st_(Xtr[sel])
                    loss = _head_loss_stack(st_, logits, tgt, 0.0)
                    opt_h.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p_ for p in st_.parameters() if p.requires_grad], 1.0)
                    opt_h.step()
            st_.eval()
            with torch.no_grad():
                logits_cal = st_(Xcal)
                P = torch.sigmoid(st_(Xcal)["h1"]).numpy()
            temps[str(yv)] = fit_T(logits_cal, _head_targets(z, cal_pos, "cpu"))
            torch.save(st_.state_dict(), os.path.join(MODELS, f"heads_fold{yv}.pt"))
            fold_res[str(yv)] = {"n_train": int(len(tr_pos)), "n_cal": int(len(cal_pos)),
                                 **_h1_metrics(st_, P, z, cal_pos)}
            log(f"  fold {yv}: {fold_res[str(yv)]} T={temps[str(yv)]}")
        with open(os.path.join(MODELS, "fold_results.json"), "w") as f:
            json.dump(fold_res, f, indent=1)
        with open(os.path.join(MODELS, "temperatures.json"), "w") as f:
            json.dump(temps, f, indent=1)
        # main-model temperatures (purged tail of the full pre-test window)
        all_pre = np.flatnonzero(lab_years <= TRAIN_END_YEAR)
        n_cal = max(200, int(len(all_pre) * CAL_FRAC))
        cal_pos = all_pre[-n_cal:]
        st_main = _mk_head_module(model, HEAD_DROPOUT)
        st_main.load_state_dict(main_sd)
        st_main.eval()
        with torch.no_grad():
            logits_cal = st_main(torch.from_numpy(H5c[cal_pos].astype(np.float32)))
        temps_main = fit_T(logits_cal, _head_targets(z, cal_pos, "cpu"))
        with open(os.path.join(MODELS, "temperatures_main.json"), "w") as f:
            json.dump(temps_main, f, indent=1)
        # consolidate shards into one file (p3-tables convenience)
        np.save(os.path.join(MODELS, "ro_hiddens.npy"), H5c)
        for r_ in range(world):
            p_ = os.path.join(MODELS, f"ro_hiddens_shard{r_}.npy")
            if os.path.exists(p_):
                os.remove(p_)
        sched["done"].append("refits")
        sched["stage"] = "export"
        with open(sched_f, "w") as f:
            json.dump(sched, f)
    if world > 1:
        dist.barrier()

    # ================= export ================================================
    if rank == 0:
        for f_ in ("stageA_result.json", "stageB_result.json", "fold_results.json",
                   "temperatures.json", "temperatures_main.json", "heads_main.pt",
                   "schedule.json"):
            src = os.path.join(MODELS, f_)
            if not os.path.exists(src) and os.path.exists(os.path.join(PREV, f_)):
                shutil.copy(os.path.join(PREV, f_), src)
        # persistent core export (fp16): looped layers + sandwich norms +
        # readout embedding + active vocab (p3-tables loads exactly this)
        if not os.path.exists(os.path.join(MODELS, "core_fp16.pt")):
            sd_path = os.path.join(MODELS, "heads_main.pt")
            if not os.path.exists(sd_path):
                sd_path = os.path.join(PREV, "heads_main.pt")
            main_sd = torch.load(sd_path, map_location="cpu")
            core_sd = {"layers": {f"{i}": {k: v.half() for k, v in
                                 model.base.model.layers[i].state_dict().items()}
                        for i in range(CORE_START, CORE_END)},
                       "pre": {k: v.half() for k, v in model.pre_norms.state_dict().items()},
                       "post": {k: v.half() for k, v in model.post_norms.state_dict().items()},
                       "readout": {k: v.half() for k, v in main_sd.items()
                                   if k.startswith("readout.")},
                       "keep_ids": keep_ids}
            torch.save(core_sd, os.path.join(MODELS, "core_fp16.pt"))
        for junk in ("ckpt_latest.pt", "ckpt_A_end.pt"):
            p_ = os.path.join(MODELS, junk)
            if os.path.exists(p_):
                os.remove(p_)          # keep the output small; core_fp16 suffices
        log("EXPORT DONE: " + json.dumps(sorted(os.listdir(MODELS))))
    if world > 1:
        dist.destroy_process_group()

# ----------------------------------------------------------------------------
# parent
# ----------------------------------------------------------------------------

def main(WORK):
    import torch
    env = env_report()
    log(f"ENV: {json.dumps(env)}")
    cache, meta = prepare_data(WORK)
    gate("G3.0_data", meta["n_pretest"] > 30000 and meta["pretest_tokens"] > 12_000_000,
         f"labeled states={meta['n_labeled_states']} (pre-test {meta['n_pretest']} / "
         f"OOS {meta['n_oos']}); pre-test stream tokens={meta['pretest_tokens']:,}; "
         f"mean tokens/state={meta['mean_tokens']}")
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nproc = str(max(1, n_gpu))
    env2 = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    p = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={nproc}", SELF, "--mode", "train"],
                       capture_output=True, text=True, timeout=43000, env=env2)
    log("train stdout tail:\n" + p.stdout[-4000:])
    if p.returncode != 0:
        log("train stderr tail:\n" + p.stderr[-5000:])
        gate("G3.1_stage_a_prime", False, "train child crashed (see logs; "
             "resume by re-pushing - schedule.json preserved)")
        save_gates(os.path.join(WORK, "gates.json"))
        return
    MODELS = os.path.join(WORK, "models")
    a = json.load(open(os.path.join(MODELS, "stageA_result.json")))
    gate("G3.1_stage_a_prime",
         a["loss_reduction"] >= 0.30 and a["nan_skips"] == 0 and
         a["clamp_rate"] < 0.05 and a["z_final"] > 0.90,
         f"sliced-CE {a['loss_first10']} -> {a['loss_last10']} "
         f"(-{a['loss_reduction']*100:.1f}%); tokens={a['tokens']:,}; "
         f"Z_final={a['z_final']} (>0.90); nan_skips={a['nan_skips']}; "
         f"clamp={a['clamp_rate']}; DDP {a['tok_s_ddp']} tok/s; "
         f"scale={a['final_scale']}",
         extra=a)
    b = json.load(open(os.path.join(MODELS, "stageB_result.json")))
    gate("G3.2_stage_b_heads",
         b["final_val"].get("auc_re") is not None and not b["early_stop_flag"],
         f"kill-switch at ckpt1: lam03 AUC={b['ckpt1_lam03'].get('auc_re')} vs "
         f"lam00 AUC={b['ckpt1_lam00'].get('auc_re')} -> kept {b['kept_branch']} "
         f"(switch triggered: {b['kill_switch_triggered']}); final val: {b['final_val']}")
    fr = json.load(open(os.path.join(MODELS, "fold_results.json")))
    n_ok = sum(1 for v in fr.values() if not v.get("skipped"))
    gate("G3.3_walkforward_refits", n_ok >= len(FOLD_YEARS) - 1,
         f"{n_ok}/{len(FOLD_YEARS)} folds refit + temperature-scaled: "
         f"{json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'skipped'} for k, v in fr.items()})[:600]}")
    have = sorted(os.listdir(MODELS))
    need = {"core_fp16.pt", "heads_main.pt", "temperatures_main.json", "ro_hiddens.npy"}
    gate("G3.4_export", need.issubset(set(have)),
         f"models/ artifacts: {have}")
    save_gates(os.path.join(WORK, "gates.json"))
    log("P2 TRAIN COMPLETE. GATES: " + json.dumps({k: v["pass"] for k, v in GATES.items()}))


if __name__ == "__main__":
    args = parse_args()
    WORK = os.environ.get("AURUM_WORK", "/kaggle/working" if os.path.exists("/kaggle/working") else os.getcwd())
    SELF = os.path.abspath(__file__)
    stable = os.path.join(WORK, "aurum_p2_train.py")
    try:
        if SELF != stable:
            shutil.copy(SELF, stable)
        SELF = stable
    except Exception:
        pass
    set_seeds(1337)
    if args.mode == "main":
        main(WORK)
    elif args.mode == "train":
        run_train_child(WORK)
