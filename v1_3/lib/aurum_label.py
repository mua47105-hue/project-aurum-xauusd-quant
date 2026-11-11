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
from aurum_gates import BARRIER_STOPS, BARRIER_TPS, BARRIER_HORS, N_CELLS, cell_axes, COHORT_K, COHORT_GAP  # noqa

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
