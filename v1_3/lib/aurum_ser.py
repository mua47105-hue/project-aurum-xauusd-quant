# ============================================================================
# AURUM state serialization (Spec v1.1 sec 7.1): each H1 state becomes a
# ~312-token dimensionless text block. No absolute prices, no absolute dates -
# only cyclical calendar (hour-of-day, dow, session), ATR-unit quantities,
# z-scores, ratios, codebook ids.
# ============================================================================
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from aurum_feat import session_of_hour, DOW

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

def serialize_state(ts, row, with_cross=True, with_macro=True, with_priors=True,
                   trend3=None, vol3=None, drop=None):
    """ts: bar timestamp (UTC, end of bar); row: dict/Series of features.
    trend3/vol3 may be passed directly or live in the row (0/1/2 class ids).
    drop: set of feature names excluded by the adversarial-validation audit -
    their fields/lines are omitted from the frozen state format.
    Returns the text block (~300-330 Qwen tokens)."""
    drop = drop or set()
    hour = ts.hour
    trend = row.get("trend3", np.nan) if trend3 is None else trend3
    volc = row.get("vol3", np.nan) if vol3 is None else vol3
    L = []
    L.append(f"[STATE h{hour:02d} dow={DOW[ts.weekday()]} sess={session_of_hour(hour)}]")
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
            L.append("xasset " + " ".join(f"{a}={_f(v)}" for a, v in xg))
    if with_macro:
        mc = [("dxy", "m_dxy", row.get("m_dxy")), ("y10", "m_y10", row.get("m_y10")),
              ("vix", "m_vix", row.get("m_vix"))]
        mc = [(a, v) for a, k, v in mc if k not in drop]
        if mc:
            L.append("macro " + " ".join(f"{a}={_f(v)}" for a, v in mc))
    trend_v, volc_v = trend, volc
    try:
        if np.isfinite(trend_v) and np.isfinite(volc_v):
            tn = ["down", "flat", "up"][int(trend_v)]; vn = ["low", "mid", "high"][int(volc_v)]
            L.append(f"regime trend={tn} vol={vn}")
    except (TypeError, ValueError):
        pass
    if with_priors:
        pr = [("w24", "prior_w24", row.get("prior_w24")), ("l24", "prior_l24", row.get("prior_l24")),
              ("w48", "prior_w48", row.get("prior_w48")), ("l48", "prior_l48", row.get("prior_l48"))]
        pr = [(a, v) for a, k, v in pr if k not in drop]
        if pr:
            L.append("prior " + " ".join(f"{a}={_u(v)}" for a, v in pr))
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
