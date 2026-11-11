"""
AURUM v2.0 -- Feature Module F6: Multi-Horizon Saturating Trend Ladder (UNIFIED)
=================================================================================
Synthesis of BOTH blueprints:
  * Blueprint 2 Section 3.6 -- the Carver EWMAC geometric ladder
    s in {2,4,8,16,32,64,128,256}h with scalar normalization (mean|F|~10,
    cap +-20), CFM saturating response phi, forecast diversification
    multiplier (FDM), and CFM one-sided momentum.
  * Blueprint 1 Section 5.2 -- CFM saturating tanh trend (s*=0.89) at
    24/96/240h, Man AHL damped MACD (Baz et al. 2015) at (8,24)/(16,48)/(32,96)
    with the non-monotonic damped response R(y)=y exp(-y^2/4)/0.89 and the
    ahl_macd_composite, Lim et al. (2019) multi-scale vol-normalized returns,
    and the Wood-Roberts-Zohren (2021) fast causal changepoint features
    (cpd_severity / cpd_location at lookbacks 21 and 126).

  RECONCILIATION NOTE (severity direction): Blueprint 1's literal formula
  nu = 1 - 1/(1+exp(-reduction)) is DECREASING in the changepoint strength
  (reduction >= 0 by construction), which contradicts Wood et al. Eq. 10's
  bounded logistic severity (increasing in the log-likelihood-ratio-like
  statistic).  We implement the paper's direction: nu = 1/(1+exp(-reduction)).
"""

import numpy as np
import pandas as pd

TREND_LADDER_H = (2, 4, 8, 16, 32, 64, 128, 256)
SIGMA_RET_SPAN = 96          # EW std span for vol normalization (bars)
SCALAR_MIN_BARS = 24 * 252 * 2   # >= 2 years expanding before scalar is valid
FORECAST_CAP = 20.0
AVG_ABS_TARGET = 10.0
S_STAR = 0.89
CFM_ONE_SIDED_H = (24, 120, 504)


def _phi_sat(y):
    y = np.asarray(y, dtype="float64")
    return y * np.exp(-y * y / 4.0) / S_STAR


def trend_ladder(h1_close: pd.Series,
                  scalar_min_bars: int = SCALAR_MIN_BARS) -> pd.DataFrame:
    """Trend ladder features on H1 closes (UTC index) -- Blueprint 2 core.
    scalar_min_bars: expanding-scalar warmup (default 2y; the self-test uses
    a smaller value so the mid-trend region is defined on synthetic data)."""
    p = h1_close.astype("float64")
    r = np.log(p / p.shift(1))
    sig_ret = r.ewm(span=SIGMA_RET_SPAN, min_periods=24).std()
    sig_p = p * sig_ret                                     # price-vol units

    out = {}
    sat_cols = []
    for s in TREND_LADDER_H:
        ema_f = p.ewm(span=s, min_periods=s).mean()
        ema_s = p.ewm(span=4 * s, min_periods=4 * s).mean()
        raw = (ema_f - ema_s) / (sig_p + 1e-12)
        out[f"ewmac_raw_{s}h"] = raw
        # expanding scalar, valid only once >= 2y of expanding data (STRICTLY
        # CAUSAL: no backfill -- the blueprint's .bfill() leaked the 2-year
        # scalar into earlier bars, caught by the G1.3 truncation gate;
        # the 2010-2011 warmup rows are NaN and never enter training folds)
        absmean = raw.abs().expanding(min_periods=1).mean()
        n_seen = raw.abs().expanding().count()
        scalar = AVG_ABS_TARGET / (absmean + 1e-12)
        scalar = scalar.where(n_seen >= scalar_min_bars)
        F = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        out[f"ewmac_F_{s}h"] = F
        sat = pd.Series(_phi_sat(F / AVG_ABS_TARGET), index=p.index)
        out[f"ewmac_sat_{s}h"] = sat
        sat_cols.append(sat)

    # CFM one-sided risk-adjusted momentum at 1-day / 5-day / 3-week scales
    for n in CFM_ONE_SIDED_H:
        ema_n = p.ewm(span=n, min_periods=n).mean()
        sig_n = p.diff().abs().ewm(span=n, min_periods=n).mean()
        out[f"cfm_onesided_{n}h"] = (p.shift(1) - ema_n.shift(1)) / (sig_n.shift(1) + 1e-12)

    # blend + FDM (N / sqrt(1' C 1): sqrt(N) when independent, 1 when identical)
    S = pd.concat(sat_cols, axis=1)
    F_blend = S.mean(axis=1)
    N = len(sat_cols)
    W = scalar_min_bars
    means = S.rolling(W, min_periods=W // 2).mean()
    sds = S.rolling(W, min_periods=W // 2).std()
    pair_rows = []
    ii, jj = np.triu_indices(N, k=1)
    for a, b_ in zip(ii, jj):
        exy = (S.iloc[:, a] * S.iloc[:, b_]).rolling(W, min_periods=W // 2).mean()
        cab = (exy - means.iloc[:, a] * means.iloc[:, b_]) / (
            sds.iloc[:, a] * sds.iloc[:, b_] + 1e-12)
        pair_rows.append(cab)
    rho_bar = pd.concat(pair_rows, axis=1).mean(axis=1).reindex(p.index).ffill()
    rho_bar = rho_bar.fillna(1.0).clip(-0.5, 1.0)
    fdm = N / np.sqrt(N * (1 + (N - 1) * rho_bar))
    fdm = fdm.clip(1.0, np.sqrt(N))
    out["F_blend"] = (F_blend * fdm * 10.0).clip(-FORECAST_CAP, FORECAST_CAP)  # back to F units
    out["fdm"] = fdm
    out["sig_ret"] = sig_ret
    return pd.DataFrame(out)


# ============================================================================
# Blueprint 1 Section 5.2 -- CFM tanh trend, AHL damped MACD, norm returns
# ============================================================================
def compute_cfm_and_ahl_trend_features(close: pd.Series) -> pd.DataFrame:
    """Blueprint 1 additions:
    1. CFM (Lemperiere et al. 2014) saturating tanh trend signal (s* = 0.89)
       at n in {24, 96, 240} hours.
    2. Man AHL (Baz et al. 2015) multi-scale volatility-normalized MACD with
       non-monotonic damped response R(y) = y * exp(-y^2/4) / 0.89 at
       (S, L) in {(8,24), (16,48), (32,96)} + the 3-scale composite.
    3. Oxford-Man (Lim et al. 2019) multi-horizon vol-normalized returns
       k in {1, 8, 24, 63, 126, 252} with 252-halflife winsorization.
    """
    df = pd.DataFrame(index=close.index)
    ret = close.pct_change().fillna(0.0)
    # 63-bar and 252-bar EWMA volatility and winsorization (Lim et al. 2019)
    ewm_mean_252 = ret.ewm(halflife=252, min_periods=20).mean()
    ewm_std_252 = ret.ewm(halflife=252, min_periods=20).std().fillna(1e-4)
    ret_win = ret.clip(lower=ewm_mean_252 - 5.0 * ewm_std_252,
                       upper=ewm_mean_252 + 5.0 * ewm_std_252)

    # strictly causal: no .bfill() on the warmup (NaN until min_periods met;
    # the blueprint's bfill leaked future warmup values into early bars)
    sigma_t = ret_win.ewm(span=63, min_periods=20).std() + 1e-8
    df["ex_ante_vol_63"] = sigma_t

    # 1. Multi-scale normalized returns: r_{t-k,t} / (sigma_t * sqrt(k))
    for k in [1, 8, 24, 63, 126, 252]:
        ret_k = close.pct_change(periods=k).fillna(0.0)
        df[f"norm_ret_{k}"] = (ret_k / (sigma_t * np.sqrt(k))).clip(-5.0, 5.0)

    # 2. CFM Saturating Trend Signal: s* tanh(s_n/s*), s* = 0.89
    s_star = 0.89
    diff_abs = close.diff().abs().fillna(0.0)
    for n in [24, 96, 240]:
        p_ema = close.shift(1).ewm(span=n, min_periods=10).mean()
        sig_n = diff_abs.shift(1).ewm(span=n, min_periods=10).mean() * np.sqrt(n) + 1e-8
        s_n = (close.shift(1) - p_ema) / sig_n
        df[f"cfm_tanh_trend_{n}"] = s_star * np.tanh(s_n / s_star)

    # 3. Man AHL (Baz et al. 2015) Damped MACD: (S, L) in {(8,24),(16,48),(32,96)}
    price_std_63 = close.rolling(63, min_periods=20).std() + 1e-8
    macd_damped_list = []
    for s_span, l_span in [(8, 24), (16, 48), (32, 96)]:
        hl_s = np.log(0.5) / np.log(1.0 - 1.0 / s_span)
        hl_l = np.log(0.5) / np.log(1.0 - 1.0 / l_span)
        m_s = close.ewm(halflife=hl_s, min_periods=s_span).mean()
        m_l = close.ewm(halflife=hl_l, min_periods=l_span).mean()
        xi = (m_s - m_l) / price_std_63
        xi_std_252 = xi.rolling(252, min_periods=30).std() + 1e-8
        y_k = (xi / xi_std_252).clip(-4.0, 4.0)
        r_y = (y_k * np.exp(-0.25 * (y_k ** 2))) / 0.89
        df[f"ahl_macd_raw_{s_span}_{l_span}"] = y_k
        df[f"ahl_macd_damped_{s_span}_{l_span}"] = r_y
        macd_damped_list.append(r_y)

    df["ahl_macd_composite"] = sum(macd_damped_list) / 3.0
    return df


# ============================================================================
# Blueprint 1 Section 5.2 -- Online changepoint detection features
# ============================================================================
def fast_causal_changepoint_features(ret: pd.Series, lbw_list=(21, 126)) -> pd.DataFrame:
    """Fast CPU implementation of Oxford-Man (Wood et al. 2021) changepoint
    detection severity score nu_t in (0,1) and normalized location gamma_t in
    (0,1), at lookbacks {21, 126} bars.

    The per-bar inner split search is vectorized with numpy prefix sums
    (engineering adaptation; mathematically identical to the blueprint's
    per-split loop).  Severity uses the Wood et al. Eq. 10 logistic direction
    (see reconciliation note in the module docstring).
    """
    r = np.nan_to_num(ret.fillna(0.0).values, nan=0.0)
    T = len(r)
    out = pd.DataFrame(index=ret.index)

    for l in lbw_list:
        nu = np.full(T, 0.5, dtype=np.float32)
        gamma = np.full(T, 0.5, dtype=np.float32)
        # vectorized per-bar: prefix sums inside the window
        cs_all = np.cumsum(r)
        cq_all = np.cumsum(r * r)
        cs_prev = np.concatenate([[0.0], cs_all[:-1]])
        cq_prev = np.concatenate([[0.0], cq_all[:-1]])
        c_splits = np.arange(2, max(3, l - 1))          # candidate splits
        for t in range(l, T):
            lo = t - l
            win = r[lo: t + 1]
            mu_w = (cs_all[t] - cs_prev[lo]) / (l + 1)
            var_w = (cq_all[t] - cq_prev[lo]) / (l + 1) - mu_w * mu_w
            if var_w < 1e-24:
                continue
            # z-scored window prefix sums (z = (x - mu)/sd)
            sd_w = np.sqrt(var_w)
            cs_win = cs_all[lo: t + 1] - cs_prev[lo]    # cumsum over window, len l+1
            cq_win = cq_all[lo: t + 1] - cq_prev[lo]
            tss = (l + 1)                               # sum of z^2 for standardized win
            # split c: left = x[0:c], right = x[c:l+1]; RSS from raw moments,
            # then rescaled by the window variance (z-scored equivalent)
            c = c_splits
            n1 = c.astype(np.float64)
            n2 = (l + 1) - c.astype(np.float64)
            s1 = cs_win[c - 1]
            q1 = cq_win[c - 1]
            s2 = cs_win[-1] - cs_win[c - 1]
            q2 = cq_win[-1] - cq_win[c - 1]
            rss1 = q1 - s1 * s1 / n1
            rss2 = q2 - s2 * s2 / n2
            reduction = 0.5 * (tss - (rss1 + rss2) / var_w)
            j = int(np.argmax(reduction))
            best_red = reduction[j]
            nu[t] = 1.0 / (1.0 + np.exp(-np.clip(best_red, -30.0, 30.0)))
            gamma[t] = c[j] / float(l)
        out[f"cpd_severity_{l}"] = nu
        out[f"cpd_location_{l}"] = gamma
    return out


def f6_self_test() -> None:
    rng = np.random.default_rng(21)
    T = 30_000
    idx = pd.date_range("2019-01-01", periods=T, freq="h", tz="UTC")
    # persistent trending regime for first 15k bars, chop after
    drift = np.where(np.arange(T) < 15_000, 0.0004, 0.0)
    r = drift + rng.standard_normal(T) * 0.0011
    p = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    f = trend_ladder(p, scalar_min_bars=2000)
    # trend regime: F_blend should be strongly positive mid-trend
    mid = f["F_blend"].iloc[6000:12000].mean()
    assert mid > 2.0, f"F_blend should be positive in trend regime, got {mid:.2f}"
    # chop: F_blend near zero
    chop = f["F_blend"].iloc[20_000:].mean()
    assert abs(chop) < 2.0, f"F_blend should be ~0 in chop, got {chop:.2f}"
    # saturation bounds
    for c in [c for c in f.columns if c.startswith("ewmac_sat_")]:
        assert f[c].abs().max() < 1.6, f"{c} exceeded saturation bound"
    # FDM in [1, sqrt(8)]
    assert f["fdm"].dropna().between(1.0, np.sqrt(8) + 1e-9).all()
    # causality
    cut = 20_000
    fc = trend_ladder(p.iloc[:cut], scalar_min_bars=2000)
    common = idx[3000: cut - 300]
    pd.testing.assert_frame_equal(fc.loc[common], f.loc[common])

    # ---- Blueprint 1 additions ----
    b1 = compute_cfm_and_ahl_trend_features(p)
    assert set(["cfm_tanh_trend_24", "ahl_macd_composite", "norm_ret_63",
                "ex_ante_vol_63"]).issubset(b1.columns)
    assert b1["cfm_tanh_trend_96"].iloc[6000:12000].mean() > 0.3, \
        "CFM tanh trend should be positive in the trend regime"
    assert b1["ahl_macd_composite"].abs().max() <= 1.0 / 0.89 + 1e-9
    # CPD: a variance/drift regime switch at 15000 should elevate severity
    cpd = fast_causal_changepoint_features(pd.Series(r, index=idx))
    assert cpd["cpd_severity_21"].between(0, 1).all()
    # around the switch, severity should exceed the quiet-period baseline
    near_switch = cpd["cpd_severity_21"].iloc[14_900:15_200].max()
    quiet = cpd["cpd_severity_21"].iloc[5_000:10_000].mean()
    assert near_switch > quiet, f"CPD severity failed to fire at regime switch ({near_switch:.3f} vs {quiet:.3f})"
    # CPD causality
    cpdc = fast_causal_changepoint_features(pd.Series(r[:cut], index=idx[:cut]))
    pd.testing.assert_frame_equal(cpdc.loc[idx[3000:cut - 300]],
                                  cpd.loc[idx[3000:cut - 300]])
    print("F6 self-tests passed: trend detected, chop flat, saturation bounded, "
          "FDM, CFM/AHL/CPD additions, causal.")
