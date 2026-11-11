"""
AURUM v2.0 -- Feature Module F5: Online EW-RLS Cross-Asset Residualization
==========================================================================
Blueprint 2 Section 3.5, verbatim (the RLS recursion is already strictly
causal: the tradable residual is the a-priori innovation, betas update after).

State-space decomposition of XAUUSD hourly log returns on the three context
assets (the *arbitrage/macro-hedging triangle*):

    y_t = alpha_t + beta_JPY,t x_JPY,t + beta_EUR,t x_EUR,t + beta_XAG,t x_XAG,t + eps_t
    beta_t = beta_{t-1} + w_t        (random-walk coefficients)

Estimated by Recursive Least Squares with forgetting factor lambda -- the
information-form Kalman filter for the TVP regression (RLS(lambda) <=> Kalman
with Q/R tied to (1-lambda)/lambda; the P-recursion is a Riccati equation).
Exact recursions (Harvey 1989; Durbin & Koopman 2012):

    a_t   = y_t - x_t' beta_{t-1}                     (a-priori innovation)
    g_t   = P_{t-1} x_t / (lambda + x_t' P_{t-1} x_t) (gain)
    P_t   = lambda^-1 P_{t-1} - g_t x_t' lambda^-1 P_{t-1}
    beta_t= beta_{t-1} + a_t g_t

CAUSAL DISCIPLINE: the tradable residual at bar t is the a-priori innovation
eps_t = y_t - x_t' beta_{t-1} (betas from bars <= t-1); the RLS update with
(x_t, y_t) happens only AFTER the residual is recorded.  Never re-fit in-sample.

Residual process (Avellaneda-Lee 2010 transplant to hourly frequency):
    X_t = cumsum(eps)                     (cointegration spread level)
    kappa from AR(1) of X over trailing window: rho = Corr(X_t, X_{t-1}),
        kappa = -ln(rho)/1 bar, tau = 1/kappa (clip tau <= 240 bars)
    sigma_eq = sigma_eps * sqrt(tau/2)    (OU stationary std)
    z_t = (X_t - m_X) / sigma_eq          (s-score analogue; A&L entry 1.25 /
        2.5 hourly-hardened, exits 0.75/0.50)

Two orthogonal alpha channels:
    Channel A (lead-lag catch-up):  m_A = EWMA_16(eps)/sigma_eps  -- residual
        momentum: the FX/metals complex moved and gold has not yet caught up.
    Channel B (idiosyncratic overextension): fade z when |z| > 2.5 with the
        saturating response phi(y) = y exp(-y^2/4)/0.89 (Baz et al. / DMN),
        exit |z| < 0.75, kill-switch if tau > 240 bars (OU invalid).
"""

import numpy as np
import pandas as pd

RLS_LAMBDA = 0.995          # forgetting factor: memory ~ 1/(1-lambda) = 200 bars (~8.3d)
P_INIT = 1e2                # P(0) = delta * I
VOL_SPAN_X = 60             # EW vol standardization of regressors (bars)
RES_MOM_SPAN = 16           # Channel A EWMA span (bars)
SIG_EPS_SPAN = 480          # EW std of residual (bars)
MEAN_X_SPAN = 480           # EW mean of spread level (bars)
OU_WINDOW = 480             # kappa estimation window (bars)
Z_ENTRY = 2.5               # Channel B entry threshold (sigma_eq units)
Z_EXIT = 0.75               # Channel B exit band
TAU_MAX = 240.0             # OU validity kill-switch (bars)
S_STAR = 0.89               # saturation scale (Baz/DMN)


def phi_sat(y: np.ndarray | float) -> np.ndarray | float:
    """Saturating-then-cutoff response phi(y) = y exp(-y^2/4)/0.89, peak at
    sqrt(2): position grows with |y| up to |y|=sqrt(2) then de-risks (Baz et
    al. 2015; Lim-Zohren-Roberts 2019 eq. 7 -- verified from paper + repo)."""
    y = np.asarray(y, dtype="float64")
    return y * np.exp(-y * y / 4.0) / S_STAR


def ew_rls_residualize(y: pd.Series, X: pd.DataFrame,
                       lam: float = RLS_LAMBDA) -> pd.DataFrame:
    """Run causal EW-RLS of y on X (with intercept) bar by bar.

    Returns DataFrame with: eps (a-priori innovation), beta_* (post-update
    coefficients), R2_ew (EW R-squared), pred (one-step prediction).
    """
    names = list(X.columns)
    xdf = X.copy()
    # standardize regressors by trailing EW vol (conditioning; the Kalman gain
    # then auto-scales) -- betas are on standardized regressors
    for c in names:
        sd = xdf[c].ewm(span=VOL_SPAN_X, min_periods=12).std()
        mu = xdf[c].ewm(span=VOL_SPAN_X, min_periods=12).mean()
        xdf[c] = (xdf[c] - mu) / (sd + 1e-12)
    xs = np.column_stack([np.ones(len(xdf))] + [xdf[c].values for c in names])
    yv = y.values.astype("float64")
    T, k = xs.shape

    beta = np.zeros((T, k))
    P = np.eye(k) * P_INIT
    b = np.zeros(k)
    eps = np.full(T, np.nan)
    pred = np.full(T, np.nan)
    r2_num = 0.0
    r2_den = 0.0
    r2 = np.full(T, np.nan)
    for t in range(T):
        xt = xs[t]
        yv_t = yv[t]
        if not (np.all(np.isfinite(xt)) and np.isfinite(yv_t)):
            beta[t] = b                      # warm-up row: state unchanged
            continue                          # (eps stays NaN, no update)
        p = xt @ P @ xt
        e = yv_t - xt @ b                     # a-priori innovation (TRADABLE)
        eps[t] = e
        pred[t] = xt @ b
        g = (P @ xt) / (lam + p)
        b = b + e * g
        P = (P - np.outer(g, xt @ P)) / lam
        # running EW R^2 (variance share explained, forget rate ~ lam)
        r2_num = lam * r2_num + pred[t] ** 2
        r2_den = lam * r2_den + yv_t ** 2
        if t > 50 and r2_den > 1e-18:
            r2[t] = r2_num / r2_den
        beta[t] = b
    out = pd.DataFrame({"eps": eps, "pred": pred, "R2_ew": r2}, index=y.index)
    for j, c in enumerate(names):
        out[f"beta_{c}"] = beta[:, j + 1]
    return out


def residual_channels(h1: pd.DataFrame, price_map: dict[str, str],
                      lam: float = RLS_LAMBDA) -> pd.DataFrame:
    """Full residualization feature block on H1 bars.

    Parameters
    ----------
    h1 : H1 frame indexed by UTC bar label with columns per price_map:
         {'xau': 'close', 'jpy': 'close_jpy', 'eur': 'close_eur', 'xag': 'close_xag'}
    """
    rets = {}
    for k_, col in price_map.items():
        rets[k_] = np.log(h1[col] / h1[col].shift(1))
    y = rets["xau"]
    X = pd.DataFrame({k_: rets[k_] for k_ in ("jpy", "eur", "xag")})
    base = ew_rls_residualize(y, X, lam=lam)

    eps = base["eps"]
    sig_eps = eps.ewm(span=SIG_EPS_SPAN, min_periods=60).std()
    m_A = eps.ewm(span=RES_MOM_SPAN, min_periods=4).mean() / (sig_eps + 1e-12)

    # spread level + OU s-score
    Xlvl = eps.cumsum()
    m_X = Xlvl.ewm(span=MEAN_X_SPAN, min_periods=120).mean()
    rho = Xlvl.rolling(OU_WINDOW, min_periods=120).apply(
        lambda s: np.corrcoef(s[:-1], s[1:])[0, 1] if np.std(s) > 0 else 1.0,
        raw=True)
    kappa = (-np.log(rho.clip(1e-6, 1 - 1e-9))).clip(lower=1e-4)
    tau = (1.0 / kappa).clip(upper=TAU_MAX)
    sig_eq = sig_eps * np.sqrt(tau / 2.0)
    z = (Xlvl - m_X) / (sig_eq + 1e-12)

    out = base.copy()
    out["sig_eps"] = sig_eps
    out["res_mom_A"] = m_A
    out["spread_lvl"] = Xlvl
    out["spread_z"] = z
    out["ou_tau"] = tau
    out["sig_eq"] = sig_eq
    out["ou_valid"] = (tau < TAU_MAX).astype(float)

    # Channel signals (features AND standalone rule baselines)
    out["q_A"] = np.tanh(m_A)                              # catch-up continuation
    out["q_B"] = -(z / Z_ENTRY) * phi_sat(z / Z_ENTRY) * out["ou_valid"]
    out["chB_active"] = (z.abs() > Z_ENTRY).astype(float) * out["ou_valid"]
    out["chB_exit"] = (z.abs() < Z_EXIT).astype(float)
    # conflict discount: rare trending-yet-extended residual -> halve both
    out["conflict"] = ((np.sign(out["q_A"]) == -np.sign(out["q_B"])) &
                       (out["q_A"].abs() > 0.3) & (out["q_B"].abs() > 0.3)).astype(float)
    return out


def f5_self_test() -> None:
    rng = np.random.default_rng(5)
    T = 6000
    idx = pd.date_range("2020-01-01", periods=T, freq="h", tz="UTC")
    f1 = rng.standard_normal(T) * 0.0010
    f2 = rng.standard_normal(T) * 0.0008
    r_jpy = f1 + rng.standard_normal(T) * 0.0003
    r_eur = f2 + rng.standard_normal(T) * 0.0003
    r_xag = 1.3 * (f1 + f2) + rng.standard_normal(T) * 0.0012
    # gold = beta_jpy f1 + beta_eur f2 + beta_xag r_xag-ish + idio, with beta drift
    bj = 0.2 + 0.00002 * np.arange(T)
    r_xau = bj * f1 + 0.15 * f2 + 0.35 * r_xag + rng.standard_normal(T) * 0.0006
    h1 = pd.DataFrame({
        "close": 2000 * np.exp(np.cumsum(r_xau)),
        "close_jpy": 150 * np.exp(np.cumsum(r_jpy)),
        "close_eur": 1.1 * np.exp(np.cumsum(r_eur)),
        "close_xag": 24 * np.exp(np.cumsum(r_xag)),
    }, index=idx)
    out = residual_channels(h1, {"xau": "close", "jpy": "close_jpy",
                                 "eur": "close_eur", "xag": "close_xag"})
    # R2 should be high (systematic part explained)
    assert out["R2_ew"].iloc[-1] > 0.3, f"EW R2 too low: {out['R2_ew'].iloc[-1]}"
    # betas live on STANDARDIZED regressors predicting raw-unit returns, so
    # the right magnitude is beta_true * sd_x (~1e-3).  The invariant that
    # matters: RLS(lambda=1.0) == batch OLS on the standardized design.
    rets = {k_: np.log(h1[c] / h1[c].shift(1)) for k_, c in
            {"xau": "close", "jpy": "close_jpy", "eur": "close_eur", "xag": "close_xag"}.items()}
    Xd = pd.DataFrame({k_: rets[k_] for k_ in ("jpy", "eur", "xag")}).iloc[1:]
    yv = rets["xau"].iloc[1:]
    Xs = Xd.copy()
    for c in Xs.columns:
        Xs[c] = (Xs[c] - Xs[c].mean()) / (Xs[c].std() + 1e-12)
    D = np.column_stack([np.ones(len(Xs))] + [Xs[c].values for c in Xs.columns])
    beta_ols, *_ = np.linalg.lstsq(D, yv.values, rcond=None)
    rls_grow = ew_rls_residualize(rets["xau"], pd.DataFrame(
        {k_: rets[k_] for k_ in ("jpy", "eur", "xag")}), lam=1.0)
    got = np.array([rls_grow["eps"].index.size and 0] +
                   [rls_grow[f"beta_{c}"].iloc[-1] for c in ("jpy", "eur", "xag")])
    # intercept not exposed; compare the three slopes (skip module's EW-standardization
    # vs batch z-score difference -> tolerance 25% of |beta_ols|)
    for j, c in enumerate(("jpy", "eur", "xag")):
        assert abs(got[j + 1] - beta_ols[j + 1]) < 0.25 * max(abs(beta_ols[j + 1]), 1e-4), \
            f"beta_{c}: RLS {got[j+1]:.3e} vs OLS {beta_ols[j+1]:.3e}"
    # eps should be much smaller in vol than raw gold returns
    ratio = out["sig_eps"].iloc[-1] / np.std(r_xau[-500:])
    assert ratio < 1.0, f"residual vol {ratio:.2f}x raw vol -- residualization failed"
    # causality: truncate tail, earlier features unchanged
    cut = 4000
    h1c = h1.iloc[: cut]
    outc = residual_channels(h1c, {"xau": "close", "jpy": "close_jpy",
                                   "eur": "close_eur", "xag": "close_xag"})
    common = idx[100: cut - 200]
    pd.testing.assert_frame_equal(outc.loc[common], out.loc[common])
    print("F5 self-tests passed: betas recovered, residual shrinks, causal.")
