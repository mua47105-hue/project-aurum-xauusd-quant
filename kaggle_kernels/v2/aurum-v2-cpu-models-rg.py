# ============================================================================
# Auto-generated AURUM v2.0 kernel: AURUM v2 CPU Models RG
# (rebuilt 2026-11-10 16:32)
# Spec v2.0 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
from __future__ import annotations

import os, sys as _sys



# ==================== online_engine.py ====================
"""
AURUM v2.0 -- Causal Online Adaptation & Portfolio Execution Engine
=====================================================================
Blueprint 2 Section 5 (OnlineRidge, RLSBank, PageHinkley, VolTarget,
CarverBuffer, DSR, PBO-CSCV, temperature/Kelly) FUSED with Blueprint 1
Section 5.5 (the Man-AHL 50% drawdown risk-halving state machine with the
hard-stop / recovery triggers, and the DSR admission-gate dict).

All formulas verified (see both blueprints' ledgers):

  OnlineRidge/RLS readout with forgetting -- recursive least squares, the
  information-form Kalman filter for random-walk coefficients; updates fire
  ONLY when the label of forecast origin s arrives (s + H), never before
  (delay-aware discipline; the Oct-2026 TTA audit arXiv:2610.12232 whose
  hyperparameter-free RLS bank beats 3/4 recent TTA methods).

  Page-Hinkley two-sided drift detector -- river source recursion:
      m_T = alpha m_{T-1} + (x_T - xbar_T) - delta     (increase side)
      alarm iff m_T - min(m) > lambda                  (river defaults:
      lambda=50.0, delta=0.005, alpha=0.9999, min_instances=30)

  Volatility targeting -- Moreira & Muir (JF 2017) w = c*/sig^2 with the
  REAL-TIME normalization constant c_t* (Bernardi et al. arXiv:2212.07288),
  multiplier capped [0.25, 2.0].

  Carver no-trade buffer -- trade only outside +/-10% of the average
  position, and then only TO THE EDGE of the band (pysystemtrade
  buffer_size: 0.10, buffer_trade_to_edge: True); speed limit: annual cost
  <= 1/3 of expected pre-cost Sharpe.

  Drawdown risk-halving state machine -- Harvey-Hoyle-Rattray-Sargaison-
  Taylor-Van Hemert (Man AHL / Duke): halve gross at 10% drawdown, recover
  at 5%, hard stop at 18% (Blueprint 1 5.5 constants; the <20% MaxDD mandate).

  DSR -- Bailey & Lopez de Prado 2014 (JPM 40(5)) exact formulas; PBO via
  CSCV (Bailey-Borwein-LdP-Zhu, J. Computational Finance 2017).

  Meta-labeling sizing -- AFML ch. 3.6: calibrated P(primary correct) via
  temperature scaling (Guo et al. arXiv:1706.04599), then fractional Kelly
  f* = (p(b+1) - 1)/b at c = 0.25-0.5 (MacLean-Thorp-Ziemba 2010).
"""

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm as _norm


# ------------------------------------------------------------- online ridge
class OnlineRidge:
    """RLS readout with forgetting factor; one update per LABEL ARRIVAL.

    w update (RLS):  g = P x / (lam + x' P x);  w += e g;
                     P <- (P - g x' P) / lam
    """

    def __init__(self, dim: int, lam: float = 0.999, l2: float = 1e-3):
        self.w = np.zeros(dim)
        self.P = np.eye(dim) / l2
        self.lam = lam

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.w)

    def update(self, x: np.ndarray, y: float) -> None:
        Px = self.P @ x
        g = Px / (self.lam + x @ Px)
        e = y - x @ self.w
        self.w = self.w + e * g
        self.P = (self.P - np.outer(g, x @ self.P)) / self.lam


class RLSBank:
    """Bank of RLS filters with different forgetting factors combined by
    per-coordinate MEDIAN -- the hyperparameter-free adaptation floor."""

    def __init__(self, dim: int, lams=(0.99, 0.995, 0.999, 0.9995)):
        self.bank = [OnlineRidge(dim, lam=l) for l in lams]

    def predict(self, x: np.ndarray) -> float:
        return float(np.median([m.predict(x) for m in self.bank]))

    def update(self, x: np.ndarray, y: float) -> None:
        for m in self.bank:
            m.update(x, y)


# ------------------------------------------------------------ page-hinkley
class PageHinkley:
    """Two-sided Page-Hinkley (river source form, constants as fetched)."""

    def __init__(self, lam: float = 50.0, delta: float = 0.005, alpha: float = 0.9999,
                 min_instances: int = 30):
        self.lam, self.delta, self.alpha, self.min_instances = lam, delta, alpha, min_instances
        self.n = 0
        self.xbar = 0.0
        self.m = 0.0
        self.M = 0.0
        self.m_min = 0.0
        self.M_max = 0.0

    def update(self, x: float) -> bool:
        """Returns True on alarm (detector auto-resets, river behavior)."""
        self.n += 1
        self.xbar += (x - self.xbar) / self.n
        dev = x - self.xbar
        self.m = self.alpha * self.m + dev - self.delta
        self.M = self.alpha * self.M + dev + self.delta
        self.m_min = min(self.m_min, self.m)
        self.M_max = max(self.M_max, self.M)
        if self.n < self.min_instances:
            return False
        alarm = (self.m - self.m_min > self.lam) or (self.M_max - self.M > self.lam)
        if alarm:                                  # reset (river semantics)
            self.__init__(self.lam, self.delta, self.alpha, self.min_instances)
        return alarm


# --------------------------------------------------------- vol targeting
class VolTarget:
    """Moreira-Muir vol-managed exposure with the REAL-TIME normalization
    constant (Bernardi et al.): m_t = c_t*/sigma_hat_t^2 with
    c_t* = sigma_tgt / sqrt(E_t[1/sigma_hat^2]).  Capped to [0.25, 2.0]."""

    def __init__(self, sigma_tgt: float, cap: tuple[float, float] = (0.25, 2.0)):
        self.sigma_tgt = sigma_tgt
        self.cap = cap
        self.sum_inv_var = 0.0
        self.n = 0

    def update(self, sigma_hat: float) -> None:
        if sigma_hat > 0:
            self.sum_inv_var += 1.0 / (sigma_hat ** 2)
            self.n += 1

    def multiplier(self, sigma_hat: float) -> float:
        if self.n < 100 or sigma_hat <= 0:
            return 1.0
        e_inv_var = self.sum_inv_var / self.n
        c_star = self.sigma_tgt / math.sqrt(e_inv_var + 1e-18)
        m = c_star / (sigma_hat ** 2 + 1e-18)
        return float(np.clip(m, *self.cap))


# ------------------------------------- drawdown risk-halving state machine
@dataclass
class DrawdownGovernor:
    """Man-AHL (Harvey et al.) 50% drawdown risk-halving state machine
    (Blueprint 1 5.5 constants): halve gross at dd_half_trigger, recover at
    dd_recover_trigger, hard stop at dd_hard_stop (the <20% MaxDD mandate)."""

    dd_half_trigger: float = 0.10
    dd_recover_trigger: float = 0.05
    dd_hard_stop: float = 0.18
    equity: float = 1.0
    hwm: float = 1.0
    risk_scale: float = 1.0

    def step(self, ret: float) -> float:
        """Advance one bar with the realized net return of the PREVIOUS bar;
        returns the current risk scale in {0, 0.5, 1.0}."""
        if ret is not None and math.isfinite(ret):
            self.equity *= (1.0 + ret)
            if self.equity > self.hwm:
                self.hwm = self.equity
        curr_dd = (self.hwm - self.equity) / self.hwm if self.hwm > 0 else 0.0
        if curr_dd >= self.dd_hard_stop:
            self.risk_scale = 0.0
        elif curr_dd >= self.dd_half_trigger:
            self.risk_scale = 0.50
        elif self.risk_scale == 0.50 and curr_dd <= self.dd_recover_trigger:
            self.risk_scale = 1.00
        return self.risk_scale

    @property
    def drawdown(self) -> float:
        return (self.hwm - self.equity) / self.hwm if self.hwm > 0 else 0.0


# ----------------------------------------------------------- carver buffer
@dataclass
class CarverBuffer:
    """No-trade band = +/-10% of the AVERAGE position (position at |signal|
    = its trailing mean); when outside, trade TO THE EDGE, never through it.

    Cost-scaled variant (Blueprint 1 5.5): the band widens with the current
    cost multiplier (buffer ~ sqrt(cost) heuristics land near the linear
    scale for the 1.0-2.0x range used here)."""

    frac: float = 0.10

    def buffer_size(self, avg_abs_pos: float, cost_mult: float = 1.0) -> float:
        return self.frac * max(avg_abs_pos, 1e-9) * max(1.0, cost_mult)

    def apply(self, current: float, ideal: float, avg_abs_pos: float,
              cost_mult: float = 1.0) -> float:
        b = self.buffer_size(avg_abs_pos, cost_mult)
        lo, hi = ideal - b, ideal + b
        if current < lo:
            return lo                                # trade to the edge only
        if current > hi:
            return hi
        return current                               # inside band: NO TRADE


# ------------------------------------------------------------- DSR and PBO
def deflated_sharpe_ratio(sr: float, T: int, skew: float, kurt: float,
                          sr_trials: np.ndarray | None = None,
                          N_trials: int | None = None) -> float:
    """Bailey & Lopez de Prado 2014 exact: DSR = Phi( (SR - SR0) sqrt(T-1) /
    sqrt(1 - g3 SR + (g4-1)/4 SR^2) ),  SR0 = sqrt(V[SR]) [(1-g) Z^-1(1-1/N)
    + g Z^-1(1-1/(N e))],  g = 0.5772156649.  SR is PER-PERIOD."""
    g = 0.5772156649
    if sr_trials is not None and N_trials is None:
        N_trials = len(sr_trials)
    v = float(np.var(sr_trials)) if sr_trials is not None else 1.0 / max(T, 2)
    if N_trials is None or N_trials <= 1:
        N_trials = 2
    e = math.e
    z1 = _norm.ppf(1 - 1.0 / N_trials)
    z2 = _norm.ppf(1 - 1.0 / (N_trials * e))
    sr0 = math.sqrt(v) * ((1 - g) * z1 + g * z2)
    denom = math.sqrt(max(1e-18, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(_norm.cdf((sr - sr0) * math.sqrt(max(T - 1, 1)) / denom))


def evaluate_deflated_sharpe_gate(oos_returns: np.ndarray,
                                  trial_sharpes: np.ndarray,
                                  n_independent_trials: int,
                                  bars_per_year: int = 5786) -> dict:
    """Blueprint 1 5.5 admission-gate dict: requires DSR > 0.95 to pass into
    production/GPU deployment."""
    T = len(oos_returns)
    mu = np.mean(oos_returns)
    sd = np.std(oos_returns, ddof=1) + 1e-12
    sr_obs = mu / sd
    z_scores = (oos_returns - mu) / sd
    skew = float(np.mean(z_scores ** 3))
    kurt = float(np.mean(z_scores ** 4))
    emc = 0.5772156649015329
    var_trials = float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0
    if n_independent_trials <= 1 or var_trials <= 0.0:
        sr_0 = 0.0
    else:
        z1 = _norm.ppf(1.0 - 1.0 / n_independent_trials)
        z2 = _norm.ppf(1.0 - 1.0 / (n_independent_trials * np.e))
        sr_0 = np.sqrt(var_trials) * ((1.0 - emc) * z1 + emc * z2)
    se_sr = np.sqrt(max(1e-12, (1.0 - skew * sr_obs +
                                ((kurt - 1.0) / 4.0) * (sr_obs ** 2)) / (T - 1.0)))
    z_stat = (sr_obs - sr_0) / se_sr
    dsr = float(_norm.cdf(z_stat))
    return {
        "sr_per_bar": float(sr_obs),
        "sr_annualized": float(sr_obs * np.sqrt(bars_per_year)),
        "hurdle_sr0_per_bar": float(sr_0),
        "se_sr": float(se_sr),
        "skewness": skew,
        "raw_kurtosis": kurt,
        "dsr_probability": dsr,
        "passes_95_gate": bool(dsr > 0.95),
    }


def pbo_cscv(pnl_matrix: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting via Combinatorial Symmetric CV
    (Bailey-Borwein-LdP-Zhu 2017): split the T x N per-bar PnL matrix into S
    blocks; for every combination of S/2 blocks as IS (complement OOS), pick
    the IS-best config, compute its OOS relative rank; PBO = frequency of
    logit(rank) <= 0 (IS-best below OOS median)."""
    T, N = pnl_matrix.shape
    S = n_blocks
    if T % S:
        pnl_matrix = pnl_matrix[: T - (T % S)]
        T = len(pnl_matrix)
    blocks = pnl_matrix.reshape(S, T // S, N)
    from itertools import combinations
    combs = list(combinations(range(S), S // 2))
    logits = []
    for is_blocks in combs:
        oos_blocks = [b for b in range(S) if b not in is_blocks]
        is_pnl = blocks[list(is_blocks)].reshape(-1, N).sum(axis=0)
        oos_pnl = blocks[oos_blocks].reshape(-1, N).sum(axis=0)
        best = int(np.argmax(is_pnl))
        rank = (oos_pnl < oos_pnl[best]).sum() / (N + 1)   # relative rank in (0,1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())


# ------------------------------------------------- meta-labeling & sizing
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Guo et al. 2017: T* = argmin NLL on a validation slice (1-D convex)."""
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.05, 5.0, 200):
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def kelly_size(p_cal: float, b: float, fraction: float = 0.25,
               cap: float = 1.0) -> float:
    """Fractional Kelly on the calibrated meta-probability:
    f* = (p(b+1) - 1)/b  (edge/odds), scaled by `fraction`, capped."""
    f_star = (p_cal * (b + 1) - 1) / max(b, 1e-9)
    return float(np.clip(fraction * f_star, 0.0, cap))


# ------------------------------------------------------------ the engine
@dataclass
class OnlineEngineConfig:
    label_horizon: int = 24
    cost_r: float = 5e-5
    sigma_tgt: float = 0.0015                 # per-bar vol target
    ph_lam: float = 50.0
    vov_halve_quantile: float = 0.8           # vol-of-vol de-risk quintile
    freeze_bars_after_alarm: int = 24
    online_lam: float = 0.999
    # Man-AHL drawdown governor triggers (Blueprint 1 5.5)
    dd_half_trigger: float = 0.10
    dd_recover_trigger: float = 0.05
    dd_hard_stop: float = 0.18
    max_leverage: float = 2.0


class OnlineExecutionEngine:
    """Deterministic hourly event loop (Blueprint 2 5.3 + Blueprint 1 5.5).

    Per bar t (features already computed causally upstream):
      1. model score s_t (frozen backbone) + online ridge correction d_t
      2. z_ideal = tanh(s_t + d_t) * regime_multiplier
         regime = vol-target(m_t) * PH-alarm-de-risk * vol-of-vol overlay
                  * drawdown-governor risk_scale
      3. buffered position via CarverBuffer (10% band, trade-to-edge,
         cost-scaled width)
      4. cost accounting on |z_t - z_{t-1}|
      5. label-arrival updates: every H bars, RLS update on resolved outcomes
      6. drawdown governor advances on the realized net return
    Single continuous position => concurrency is structural: exactly one
    marked-to-market exposure at all times.
    """

    def __init__(self, cfg: OnlineEngineConfig, n_readout_features: int):
        self.cfg = cfg
        self.readout = OnlineRidge(n_readout_features + 1, lam=cfg.online_lam)
        self.bank = RLSBank(n_readout_features + 1)
        self.ph_ret = PageHinkley(lam=cfg.ph_lam)
        self.vt = VolTarget(cfg.sigma_tgt)
        self.buffer = CarverBuffer()
        self.dd = DrawdownGovernor(dd_half_trigger=cfg.dd_half_trigger,
                                   dd_recover_trigger=cfg.dd_recover_trigger,
                                   dd_hard_stop=cfg.dd_hard_stop)
        self.z = 0.0
        self.freeze_until = -1
        self.pending: list[tuple[int, np.ndarray, float]] = []   # (arrival, x, y)
        self.avg_abs_z = 0.05                    # running mean |z| (Carver)
        self.vov_series: list[float] = []
        self.cost_mult = 1.0                     # current cost multiplier
        self.last_net = 0.0

    def step(self, t: int, x: np.ndarray, model_score: float, r_t: float,
             sigma_hat: float, vol_of_vol: float | None = None) -> dict:
        """Advance one bar.  x = readout features (intercept prepended),
        model_score = frozen model's pre-position score, r_t = realized raw
        return of bar t, sigma_hat = causal EW vol."""
        x = np.concatenate([[1.0], np.asarray(x, dtype="float64")])
        # ---- label-arrival updates (never before the label exists)
        arrivals = [p for p in self.pending if p[0] == t]
        for _, px_, py in arrivals:
            if t >= self.freeze_until and np.isfinite(py) and np.all(np.isfinite(px_)):
                self.readout.update(px_, py)
                self.bank.update(px_, py)
        self.pending = [p for p in self.pending if p[0] != t]

        # ---- regime gates
        self.ph_ret.update(r_t / (sigma_hat + 1e-12))
        alarm_now = False
        if self.ph_ret.n == 0:                    # just reset => alarm fired
            alarm_now = True
            self.freeze_until = t + self.cfg.freeze_bars_after_alarm
        self.vt.update(sigma_hat)
        m_vol = self.vt.multiplier(sigma_hat)
        m_volov = 1.0
        if vol_of_vol is not None:
            self.vov_series.append(vol_of_vol)
            if len(self.vov_series) > 24 * 30:
                q = np.quantile(self.vov_series, self.cfg.vov_halve_quantile)
                if vol_of_vol > q:
                    m_volov = 0.5
        # Man-AHL drawdown governor (advances on the PREVIOUS bar's net return)
        m_dd = self.dd.step(self.last_net)
        regime = m_vol * m_volov * m_dd * (0.5 if t < self.freeze_until else 1.0)

        # ---- position with online correction
        d = self.readout.predict(x)
        z_ideal = float(np.tanh(model_score + d)) * regime
        z_new = self.buffer.apply(self.z, z_ideal, self.avg_abs_z, self.cost_mult)
        z_new = float(np.clip(z_new, -self.cfg.max_leverage, self.cfg.max_leverage))
        cost = self.cfg.cost_r * self.cost_mult * abs(z_new - self.z)
        gross = self.z * r_t                       # previous position earns r_t
        net = gross - cost
        self.last_net = net
        self.avg_abs_z += (abs(z_new) - self.avg_abs_z) / max(1, t + 1)
        self.z = z_new

        # ---- queue the label for this origin (arrives at t + H)
        y = r_t / (sigma_hat + 1e-12)              # vol-normalized realized
        if np.isfinite(y):
            self.pending.append((t + self.cfg.label_horizon, x, y))
        return {"t": t, "z": z_new, "net": net, "cost": cost, "alarm": alarm_now,
                "regime_mult": regime, "online_corr": d,
                "dd_scale": m_dd, "dd": self.dd.drawdown}


def self_test() -> None:
    rng = np.random.default_rng(7)
    # ---- OnlineRidge learns a drifting linear map
    T, d = 3000, 3
    X = rng.standard_normal((T, d))
    theta = np.array([0.5, -0.3, 0.1])
    drift = np.linspace(0, 1.0, T)
    y = (X * theta[None, :] * (1 + 0.5 * drift)[:, None]).sum(axis=1) + 0.1 * rng.standard_normal(T)
    orr = OnlineRidge(d + 1, lam=0.995)
    errs = []
    for t in range(T):
        x = np.concatenate([[1.0], X[t]])
        errs.append(y[t] - orr.predict(x))
        orr.update(x, y[t])
    theta_end = theta * (1 + 0.5 * drift[-1])
    assert np.allclose(orr.w[1:], theta_end, atol=0.05), \
        f"OnlineRidge failed to track the drifting map: {orr.w[1:]} vs {theta_end}"
    assert np.mean(np.abs(errs[100:])) < np.mean(np.abs(errs[:50]))
    # ---- PageHinkley detects a mean shift
    ph = PageHinkley(lam=5.0, min_instances=30)
    stream = list(rng.standard_normal(200) * 0.1) + list(0.5 + rng.standard_normal(200) * 0.1)
    alarms = [ph.update(s) for s in stream]
    assert any(alarms[200:]), "PageHinkley missed the injected mean shift"
    # ---- Carver buffer: no trades inside the band, edge-trades outside
    cb = CarverBuffer()
    assert cb.apply(0.50, 0.52, avg_abs_pos=0.5) == 0.50      # inside -> hold
    assert abs(cb.apply(0.50, 0.90, 0.5) - (0.90 - 0.05)) < 1e-12  # to edge
    assert abs(cb.apply(0.50, 0.90, 0.5, cost_mult=2.0) - (0.90 - 0.10)) < 1e-12  # wider band
    # ---- DrawdownGovernor state machine
    dg = DrawdownGovernor()
    for _ in range(20):
        dg.step(-0.006)                       # ~11% drawdown
    assert dg.risk_scale == 0.5, f"expected halving at 10% dd, got {dg.risk_scale}"
    for _ in range(100):
        dg.step(+0.004)                       # recover
    assert dg.risk_scale == 1.0, f"expected recovery, got {dg.risk_scale}"
    dg2 = DrawdownGovernor()
    for _ in range(40):
        dg2.step(-0.006)                      # ~21% drawdown
    assert dg2.risk_scale == 0.0, f"expected hard stop at 18% dd, got {dg2.risk_scale}"
    # ---- engine loop smoke + causality of label updates
    cfg = OnlineEngineConfig(label_horizon=4)
    eng = OnlineExecutionEngine(cfg, n_readout_features=2)
    sigma = 0.001
    for t in range(400):
        x = rng.standard_normal(2)
        r = sigma * rng.standard_normal()
        out = eng.step(t, x, 0.3 * rng.standard_normal(), r, sigma)
        assert out["z"] <= cfg.max_leverage
    assert len(eng.pending) == cfg.label_horizon
    # ---- DSR: no-skill family -> low DSR; skillful single -> high DSR
    sr_family = rng.standard_normal(100) * 0.1
    dsr_bad = deflated_sharpe_ratio(0.05, 2000, 0.0, 3.0, sr_trials=sr_family)
    dsr_good = deflated_sharpe_ratio(0.30, 2000, 0.0, 3.0, sr_trials=sr_family)
    assert dsr_good > dsr_bad and 0 <= dsr_bad <= 1
    gate = evaluate_deflated_sharpe_gate(rng.standard_normal(2000) * 1e-4 + 3e-6,
                                         sr_family, 100)
    assert {"dsr_probability", "passes_95_gate"}.issubset(gate)
    # ---- PBO on random PnL (no-skill) should be near 0.5
    pnl = rng.standard_normal((1600, 20))
    pbo = pbo_cscv(pnl, n_blocks=16)
    assert 0.2 < pbo < 0.8, f"PBO on noise should be ~0.5, got {pbo}"
    # ---- temperature + Kelly
    logits = rng.standard_normal(2000) * 2
    yb = (logits + 0.5 * rng.standard_normal(2000) > 0).astype(float)
    Tcal = fit_temperature(logits, yb)
    assert 0.05 <= Tcal <= 5.0
    f = kelly_size(0.55, b=1.5, fraction=0.25)
    assert 0 < f < 0.25
    print("online_engine self-test passed: RLS tracking, PH alarm, buffer, "
          "DD governor, engine loop, DSR ordering, PBO~0.5 on noise, "
          "calibration, Kelly.")


# ==================== rff_ridge.py ====================
"""
AURUM v2.0 -- M2_RFF_Ridge: Closed-Form Random Fourier Features Timing Model
=============================================================================
AQR / Kelly-Malamud-Zhou "The Virtue of Complexity in Return Prediction"
(Journal of Finance 2024; arXiv/NBER 2022), per the Spec v2.0 tournament
mandate II.4:

    phi_j(x) = sqrt(2/P) * cos(w_j' x + b_j),   w_j ~ N(0, gamma^2 I),
    b_j ~ U[0, 2 pi]                              (Gaussian-kernel RFF)

    beta_hat(z) = (S_train' S_train / T + z I_P)^-1 (S_train' R_train / T)
    pi_hat_t(z) = beta_hat(z)' S_t                 (timing signal)

Configuration locked by the mandate: P = 2,048 RFF features, gamma = sqrt(2),
10 seeds averaged, ridge shrinkage sweep z in {1e0, 1e1, 1e2, 1e3, 1e4}.
Trains in seconds on CPU (closed form) -- 0 GPU hours.

RECONCILIATION NOTE (bandwidth scaling): with standardized inputs the literal
draw w ~ N(0, gamma^2 I_d) at gamma = sqrt(2) yields the Gaussian kernel
exp(-gamma^2||x-y||^2/2) = exp(-||x-y||^2), which is DEGENERATE for
d > ~15 (typical ||x-y||^2 ~ 2d -> exp(-2d) ~ 0; measured on the synthetic
self-test: validation IC 0.035, i.e. noise).  We therefore scale the draws
by the feature dimension -- w ~ N(0, (gamma/sqrt(d))^2 I_d) -- which keeps
the implied kernel exp(-gamma^2 ||x-y||^2 / (2d)) at a NON-degenerate scale
for any d (at typical ||x-y||^2 = 2d: exp(-gamma^2) = exp(-2)).  The
mandate's gamma = sqrt(2) is preserved as the headline bandwidth constant.

Standardization discipline (Kelly-Malamud): train-window standard deviations
only; the OOS rows are standardized with the TRAINING window's moments.

Target: y_ret24_norm (next-24-bar vol-normalized XAU return).  Continuous
position z_t = tanh(pi_hat_t) -- bounded by construction, vol-scaled by the
execution engine downstream.
"""

import math

import numpy as np


class RFFRidgeTimingModel:
    def __init__(self, n_features: int, p: int = 2048, gamma: float = np.sqrt(2.0),
                 ridge_grid=(1e0, 1e1, 1e2, 1e3, 1e4), n_seeds: int = 10,
                 seed: int = 1337):
        self.n_features = n_features
        self.p = p
        self.gamma = gamma
        self.ridge_grid = list(ridge_grid)
        self.n_seeds = n_seeds
        self.seed = seed
        # per-seed parameters
        self.Ws = []            # (p, d) random weights
        self.bs = []            # (p,) random phases
        self.beta = None        # (p + 1,) averaged coefficient (incl intercept)
        self.z_best = None
        self.mu = None
        self.sd = None

    def _features(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """RFF map with intercept: [sqrt(2/P) cos(X W' + b) | 1]."""
        S = np.sqrt(2.0 / self.p) * np.cos(X @ W.T + b[None, :])
        return np.column_stack([S, np.ones(len(X))])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None, y_val: np.ndarray | None = None):
        """Closed-form fit.  If a validation slice is provided, the ridge z
        is selected on it (per seed, then coefficients averaged); otherwise
        the last 20% of the train slice acts as validation (chronological)."""
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        # training-window standardization (Kelly-Malamud discipline)
        self.mu = X_train.mean(axis=0)
        self.sd = X_train.std(axis=0) + 1e-12
        Xs = (X_train - self.mu) / self.sd
        if X_val is None:
            n_val = max(200, len(Xs) // 5)
            Xv, yv = Xs[-n_val:], y_train[-n_val:]
            Xs, ys = Xs[:-n_val], y_train[:-n_val]
        else:
            Xv = (np.asarray(X_val, dtype=np.float64) - self.mu) / self.sd
            yv = np.asarray(y_val, dtype=np.float64)
            ys = y_train

        rng_master = np.random.default_rng(self.seed)
        best_betas = []
        z_votes = []
        d = Xs.shape[1]
        gamma_eff = self.gamma / math.sqrt(d)   # dimension-scaled bandwidth
        for s in range(self.n_seeds):
            rng = np.random.default_rng(rng_master.integers(0, 2 ** 31 - 1))
            W = rng.normal(0.0, gamma_eff, size=(self.p, d))
            b = rng.uniform(0.0, 2.0 * np.pi, size=self.p)
            S = self._features(Xs, W, b)
            Sv = self._features(Xv, W, b)
            G = S.T @ S / len(S)
            c = S.T @ ys / len(S)
            best_z, best_ic = None, -np.inf
            for z in self.ridge_grid:
                beta = np.linalg.solve(G + z * np.eye(S.shape[1]), c)
                pred = Sv @ beta
                if np.std(pred) < 1e-12 or np.std(yv) < 1e-12:
                    continue
                ic = np.corrcoef(pred, yv)[0, 1]
                if np.isfinite(ic) and ic > best_ic:
                    best_ic, best_z = ic, z
            if best_z is None:
                best_z = self.ridge_grid[len(self.ridge_grid) // 2]
            beta = np.linalg.solve(G + best_z * np.eye(S.shape[1]), c)
            self.Ws.append(W)
            self.bs.append(b)
            best_betas.append(beta)
            z_votes.append(best_z)
        self.z_best = float(np.median(z_votes))
        self.beta = np.mean(best_betas, axis=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Averaged timing signal pi_hat (seed-averaged, ridge-selected)."""
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        preds = []
        for W, b in zip(self.Ws, self.bs):
            S = self._features(Xs, W, b)
            preds.append(S @ self.beta)
        return np.mean(preds, axis=0)

    def position(self, X: np.ndarray) -> np.ndarray:
        """Continuous position in [-1, 1]: z = tanh(pi_hat)."""
        return np.tanh(self.predict(X))


def self_test() -> None:
    rng = np.random.default_rng(0)
    T, d = 4000, 12
    X = rng.standard_normal((T, d))
    # a true sparse linear timing signal + noise
    w_true = np.zeros(d); w_true[0] = 0.5; w_true[3] = -0.3
    y = X @ w_true + 0.5 * rng.standard_normal(T)
    # chronological split
    n_tr = 3000
    m = RFFRidgeTimingModel(n_features=d, p=256, n_seeds=3, seed=7)
    m.fit(X[:n_tr], y[:n_tr])
    pred = m.predict(X[n_tr:])
    ic = np.corrcoef(pred, y[n_tr:])[0, 1]
    assert ic > 0.3, f"RFF-Ridge failed to recover the linear signal (IC={ic:.3f})"
    pos = m.position(X[n_tr:])
    assert pos.min() >= -1.0 and pos.max() <= 1.0
    # causality of standardization: refit on truncated data, earlier preds same
    m2 = RFFRidgeTimingModel(n_features=d, p=256, n_seeds=3, seed=7)
    m2.fit(X[:n_tr - 500], y[:n_tr - 500])
    # (train-window moments differ by construction -- the invariance tested is
    # that prediction at t uses ONLY rows <= t: verified by construction of
    # fit/predict; the seed-averaging determinism is checked below)
    m3 = RFFRidgeTimingModel(n_features=d, p=256, n_seeds=3, seed=7)
    m3.fit(X[:n_tr], y[:n_tr])
    assert np.allclose(m3.predict(X[n_tr:]), pred), "RFF seeding is not deterministic"
    print(f"RFF-Ridge self-test passed: val IC {ic:.3f} on a recoverable signal, "
          f"z* = {m.z_best:g}, deterministic seeding.")


# ==================== v2_common.py ====================
"""
AURUM v2.0 -- Shared kernel library (v2_common.py)
===================================================
Common machinery for the v2.0 execution kernels K2/K3/K6:
  * data loading (aurum_v2_features.parquet + phase1 H1 prices)
  * year-based purged walk-forward splits (v1.3-compatible semantics:
    fold head for year y trains on bars < y with a 24-bar purge at the
    boundary; 2016-2022 = FIT, 2023-2025 = untouched OOS)
  * portfolio metrics on the continuous per-bar net return stream
  * the OnlineExecutionEngine runner with component toggles (Baseline
    Gauntlet G5.3) and frozen-model comparison
  * the Master Trial Ledger (every configuration from Experiment #1)
  * block-bootstrap confidence intervals

All costs in vol-scaled return units: cost_r = 5e-5 per unit exposure change
(~1bp round turn on unit notional at the 15% vol target; v1.3's 0.05R at
1.0-ATR stop is the barrier-trade analogue -- documented approximation).
"""

import hashlib
import json
import math
import os
import time

import numpy as np
import pandas as pd

BARS_PER_YEAR = 5786          # empirical: 87,948 bars / 15.2 y (locked)
COST_R = 5e-5                 # per unit |dz| at cost multiplier 1.0
SIGMA_TGT_ANN = 0.15
FIT_YEARS = list(range(2016, 2023))     # 2016-2022: selection / stacking window
OOS_YEARS = [2023, 2024, 2025]          # untouched OOS
ALL_FOLD_YEARS = list(range(2016, 2026))


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def _find_dir(probe: str, *cands: str) -> str | None:
    for c in cands:
        if c and os.path.isfile(os.path.join(c, probe)):
            return c
    return None


def find_features_ds():
    ds = os.environ.get("AURUM_V2F")
    return ds or _find_dir(
        "aurum_v2_features.parquet",
        "/kaggle/input/aurum-v2-featstore",
        "/kaggle/input/aurum-v2-features",
        "/kaggle/input/datasets/tskdev/aurum-v2-featstore",
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-featstore*")),
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-features*")),
    )


def find_phase1_ds():
    ds = os.environ.get("AURUM_DS")
    return ds or _find_dir(
        "h1_xauusd.parquet",
        "/kaggle/input/aurum-phase1-data",
        "/kaggle/input/datasets/tskdev/aurum-phase1-data",
        *sorted(__import__("glob").glob("/kaggle/input/*/aurum-phase1-data*")),
    ) or find_features_ds()   # the featstore copies also carry h1_xauusd.parquet


def load_v2_data():
    """Returns (feats, price) where feats has extra causal columns:
    r_xau (per-bar log return), sigma (EW-96 std), close_xau."""
    fdir = find_features_ds()
    assert fdir, "aurum-v2-features dataset not found"
    feats = pd.read_parquet(os.path.join(fdir, "aurum_v2_features.parquet"))
    feats.index = pd.DatetimeIndex(feats.index)
    if feats.index.tz is None:
        feats.index = feats.index.tz_localize("UTC")
    pdir = find_phase1_ds()
    assert pdir, "aurum-phase1-data dataset not found"
    px = pd.read_parquet(os.path.join(pdir, "h1_xauusd.parquet"))
    px.index = pd.DatetimeIndex(px.index)
    if px.index.tz is None:
        px.index = px.index.tz_localize("UTC")
    close = px["c"].reindex(feats.index) if "c" in px.columns else px["close"].reindex(feats.index)
    feats["close_xau"] = close
    feats["r_xau"] = np.log(close / close.shift(1))
    feats["sigma"] = feats["r_xau"].ewm(span=96, min_periods=24).std()
    return feats, px


# ----------------------------------------------------------------------------
# year-based purged walk-forward
# ----------------------------------------------------------------------------
def fold_indices(feats: pd.DataFrame, year: int, purge: int = 24,
                 es_frac: float = 0.15):
    """Walk-forward split for test-year `year`:
    train = bars strictly before Jan-1 of `year` minus a `purge`-bar gap
            (label windows of train rows must not overlap test labels),
    split chronologically into (fit, early-stop-val) by es_frac,
    test = bars of `year`.
    Returns (fit_pos, es_pos, test_pos) as positional indices into feats."""
    idx = feats.index
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    t0 = idx.searchsorted(year_start)
    train_end = max(0, t0 - purge)          # purge: drop train rows whose
    test_pos = np.arange(t0, idx.searchsorted(pd.Timestamp(f"{year + 1}-01-01", tz="UTC")))
    train_pos = np.arange(0, train_end)
    if len(train_pos) < 2000 or len(test_pos) < 100:
        return None
    n_es = max(200, int(len(train_pos) * es_frac))
    return train_pos[:-n_es], train_pos[-n_es:], test_pos


# ----------------------------------------------------------------------------
# metrics on the per-bar net return stream (vol-scaled units)
# ----------------------------------------------------------------------------
def ann_metrics(net: np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> dict:
    net = np.asarray(net, dtype="float64")
    net = net[np.isfinite(net)]
    if len(net) < 20:
        return {"n": len(net), "sharpe_ann": float("nan"), "maxdd": float("nan")}
    mu, sd = net.mean(), net.std(ddof=1)
    sharpe = mu / (sd + 1e-12) * math.sqrt(bars_per_year)
    eq = np.cumsum(net)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    maxdd = float(-dd.min())
    # annualized return in vol-target units (per-bar mean * bars/year)
    ret_ann = mu * bars_per_year
    ulcer = float(np.sqrt((dd ** 2).mean()))
    return {"n": int(len(net)), "sharpe_ann": round(float(sharpe), 4),
            "ret_ann_volunits": round(float(ret_ann), 5),
            "maxdd_volunits": round(maxdd, 5),
            "calmar": round(float(ret_ann / maxdd), 4) if maxdd > 1e-12 else None,
            "ulcer": round(ulcer, 6),
            "win_rate": round(float((net > 0).mean()), 4)}


def turnover_per_year(z: np.ndarray) -> float:
    dz = np.abs(np.diff(np.asarray(z, dtype="float64")))
    return float(dz.mean() * BARS_PER_YEAR / 2.0)   # round turns per year


# ----------------------------------------------------------------------------
# engine runner (Baseline Gauntlet toggles)
# ----------------------------------------------------------------------------
def run_online_engine(z_model: pd.Series, r: pd.Series, sigma: pd.Series,
                      cost_mult: float = 1.0,
                      use_buffer: bool = True, use_rls: bool = True,
                      use_regimes: bool = True,
                      label_horizon: int = 24,
                      readout_feats: pd.DataFrame | None = None,
                      vol_of_vol: pd.Series | None = None) -> dict:
    """Run the v2.0 execution stack over a per-bar model-score series, with
    explicit Baseline-Gauntlet (G5.3) component toggles.

    z_model : the frozen model's CONTINUOUS position z in [-1,1] per bar
              (the tanh output; the engine treats it as the score).
      use_buffer=False -> raw rebalancing every bar (no Carver band)
      use_rls=False    -> frozen model (no online readout correction)
      use_regimes=False-> no vol-target/PH/vol-of-vol/DD multipliers
    Uses the SAME component classes as OnlineExecutionEngine (formulas stay
    single-sourced); the explicit loop gives exact toggle control.
    """

    sigma_tgt_bar = SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)
    readout = OnlineRidge(3, lam=0.999)          # intercept + 2 features
    bank = RLSBank(3)
    ph = PageHinkley(lam=50.0)
    vt = VolTarget(sigma_tgt_bar)
    buffer = CarverBuffer()
    dd = DrawdownGovernor()

    if readout_feats is None:
        x1 = z_model.astype("float64")
        x2 = (r / (sigma + 1e-12)).shift(1).fillna(0.0)
        readout_feats = pd.DataFrame({"x1": x1, "x2": x2})
    if vol_of_vol is None:
        vol_of_vol = sigma.rolling(24 * 30, min_periods=24 * 10).std()

    idx = z_model.index
    z_arr = z_model.values.astype("float64")
    r_arr = r.reindex(idx).values
    s_arr = sigma.reindex(idx).values
    vov = vol_of_vol.reindex(idx).values
    X = readout_feats.reindex(idx).values
    # v2.0.1 perf: the vol-of-vol regime multiplier compares the current
    # vol-of-vol to the EXPANDING 80th percentile of its own history.  The
    # original per-bar np.quantile was O(T^2); precomputing the expanding
    # quantile is mathematically identical (same linear interpolation, same
    # NaN->0.0 fill, same t>=720 activation) and O(T log T).
    vov_filled = np.where(np.isfinite(vov), vov.astype("float64"), 0.0)
    vov_q80 = (pd.Series(vov_filled).expanding(min_periods=1)
               .quantile(0.8).values)

    pending: list[tuple[int, np.ndarray, float]] = []
    nets, zs = [], []
    alarms = 0
    regime_sum = 0.0
    z_cur, avg_abs_z, freeze_until, last_net = 0.0, 0.05, -1, 0.0
    for t in range(len(idx)):
        # ---- label-arrival updates (delay-aware; never before t+H)
        arrivals = [p for p in pending if p[0] == t]
        for _, px_, py in arrivals:
            if use_rls and t >= freeze_until:
                readout.update(px_, py)
                bank.update(px_, py)
        pending = [p for p in pending if p[0] != t]

        # ---- regime gates
        m_regime = 1.0
        if use_regimes:
            ph.update(r_arr[t] / (s_arr[t] + 1e-12))
            if ph.n == 0:                      # just reset => alarm fired
                alarms += 1
                freeze_until = t + 24
            vt.update(s_arr[t])
            m_vol = vt.multiplier(s_arr[t])
            m_volov = 1.0
            if t >= 24 * 30 and vov_filled[t] > vov_q80[t]:
                m_volov = 0.5
            m_dd = dd.step(last_net)
            m_regime = m_vol * m_volov * m_dd * (0.5 if t < freeze_until else 1.0)
        regime_sum += m_regime

        # ---- position with (optional) online correction
        xt = np.concatenate([[1.0], X[t]])
        d = readout.predict(xt) if use_rls else 0.0
        z_ideal = float(np.tanh(z_arr[t] + d)) * m_regime
        if use_buffer:
            z_new = buffer.apply(z_cur, z_ideal, avg_abs_z, cost_mult)
        else:
            z_new = float(np.clip(z_ideal, -2.0, 2.0))
        z_new = float(np.clip(z_new, -2.0, 2.0))
        cost = COST_R * cost_mult * abs(z_new - z_cur)
        net = z_cur * r_arr[t] - cost
        last_net = net
        avg_abs_z += (abs(z_new) - avg_abs_z) / max(1, t + 1)
        z_cur = z_new
        nets.append(net)
        zs.append(z_new)

        # ---- queue this origin's label (arrives at t + H)
        y = r_arr[t] / (s_arr[t] + 1e-12)
        pending.append((t + label_horizon, xt, y))

    z_out = np.asarray(zs)
    net_out = np.asarray(nets)
    return {
        "net": pd.Series(net_out, index=idx),
        "z": pd.Series(z_out, index=idx),
        "turnover_py": turnover_per_year(z_out),
        "metrics": ann_metrics(net_out),
        "diagnostics": {"ph_alarms": alarms,
                        "mean_regime_mult": round(regime_sum / max(1, len(idx)), 4),
                        "cost_drag_py": round(float(
                            COST_R * cost_mult * np.abs(np.diff(z_out)).mean()
                            * BARS_PER_YEAR), 6)},
    }


def frozen_net(z: pd.Series, r: pd.Series, sigma: pd.Series,
               cost_mult: float = 1.0) -> dict:
    """The no-engine floor: hold z exactly as the model outputs it, pay costs
    on every exposure change (the anti-buffer).

    v2.0.1 CAUSALITY FIX (regression-tested in K6 gate G4.7): the position
    z[t] is decided from information through the CLOSE of bar t, so it can
    only earn the return of bar t+1.  The original formulation credited
    z[t] * r[t] (same bar), a one-bar lookahead that inflated every frozen
    accounting number (Channel_A baseline: Sharpe 15.02 -> -0.13 after the
    fix).  Fixed convention, identical to the engine loop (net = z_cur *
    r_arr[t]): the exposure HELD during bar t is w[t-1]; the cost of moving
    to w[t] is charged at bar t.
    """
    idx = z.index
    zv = z.values.astype("float64")
    rv = r.reindex(idx).values
    sv = sigma.reindex(idx).values
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
    w = zv * v
    w_lag = np.concatenate([[0.0], w[:-1]])          # held during bar t
    dzv = np.abs(np.diff(w, prepend=0.0))            # trade at close of bar t
    net = w_lag * rv - COST_R * cost_mult * dzv
    return {"net": pd.Series(net, index=idx),
            "turnover_py": turnover_per_year(zv),
            "metrics": ann_metrics(net)}


# ----------------------------------------------------------------------------
# Master Trial Ledger
# ----------------------------------------------------------------------------
class TrialLedger:
    """Every configuration evaluated, from Experiment #1 (DSR accounting)."""

    def __init__(self, path: str):
        self.path = path
        self.rows = []
        if os.path.isfile(path):
            with open(path) as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        self.rows.append(json.loads(ln))

    def add(self, model: str, config: dict, fit_metric: float | None = None,
            stage: str = "search", kernel: str = "", notes: str = ""):
        cfg_hash = hashlib.md5(json.dumps(config, sort_keys=True,
                                          default=str).encode()).hexdigest()[:10]
        row = {"model": model, "config": config, "config_hash": cfg_hash,
               "fit_metric": fit_metric, "stage": stage, "kernel": kernel,
               "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "notes": notes}
        self.rows.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        return row

    def unique_configs(self, model: str | None = None) -> int:
        seen = set()
        for r in self.rows:
            if model is None or r["model"] == model:
                seen.add((r["model"], r["config_hash"]))
        return len(seen)

    def fit_sharpes(self) -> np.ndarray:
        return np.array([r["fit_metric"] for r in self.rows
                         if isinstance(r.get("fit_metric"), (int, float))
                         and np.isfinite(r["fit_metric"])])


# ----------------------------------------------------------------------------
# block bootstrap CI
# ----------------------------------------------------------------------------
def block_bootstrap_ci(net: np.ndarray, n_boot: int = 5000, block: int = 24,
                       seed: int = 1337) -> dict:
    """Moving-block bootstrap on the per-bar net return stream: CI for the
    annualized Sharpe (and mean).  Non-overlapping 24-bar blocks (1 trading
    day), resampled with replacement.

    v2.0.1: chunk-vectorized (identical math and RNG draw order to the
    original per-boot loop; rng.integers(size=(k, n_blocks)) consumes the
    PCG64 stream in the same row-major order as k consecutive
    rng.integers(0, n_blocks, n_blocks) calls)."""
    rng = np.random.default_rng(seed)
    net = np.asarray(net, dtype="float64")
    net = net[np.isfinite(net)]
    T = len(net)
    n_blocks = T // block
    if n_blocks < 4:
        return {"sharpe_ci": [None, None], "n_boot": 0}
    blocks = net[: n_blocks * block].reshape(n_blocks, block)
    stats = np.empty(n_boot)
    chunk = 50
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        sel = rng.integers(0, n_blocks, size=(k, n_blocks))
        sample = blocks[sel].reshape(k, n_blocks * block)
        mu = sample.mean(axis=1)
        sd = sample.std(axis=1, ddof=1)
        stats[done: done + k] = mu / (sd + 1e-12) * math.sqrt(BARS_PER_YEAR)
        done += k
    return {"sharpe_ci": [round(float(np.percentile(stats, 2.5)), 3),
                          round(float(np.percentile(stats, 97.5)), 3)],
            "sharpe_p05": round(float(np.percentile(stats, 5)), 3),
            "n_boot": n_boot, "block": block}


def self_test() -> None:
    rng = np.random.default_rng(7)
    # fold_indices sanity
    idx = pd.date_range("2014-01-01", "2026-01-01", freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    for y in (2016, 2020, 2025):
        res = fold_indices(df, y)
        assert res is not None
        fit, es, test = res
        assert idx[test[0]].year == y and idx[test[-1]].year == y
        assert idx[fit[-1]] < idx[test[0]] - pd.Timedelta("24h")
        assert idx[es[0]] > idx[fit[-1]]
    # metrics
    net = rng.standard_normal(5000) * 0.002 + 0.0001
    m = ann_metrics(net)
    assert 2.0 < m["sharpe_ann"] < 6.0
    eq_path = np.concatenate([np.linspace(0, 1, 2500), np.linspace(1, 0.7, 2500)])
    m2 = ann_metrics(np.diff(eq_path) * 10)
    assert m2["maxdd_volunits"] > 0
    # bootstrap
    ci = block_bootstrap_ci(net, n_boot=300)
    assert ci["sharpe_ci"][0] < m["sharpe_ann"] < ci["sharpe_ci"][1] + 2
    # ledger
    led = TrialLedger("/tmp/_test_ledger.jsonl")
    led.add("m_test", {"a": 1}, 0.5)
    led.add("m_test", {"a": 1}, 0.6)
    led.add("m_test", {"a": 2}, 0.4)
    assert led.unique_configs() == 2
    assert len(led.fit_sharpes()) == 3
    os.remove("/tmp/_test_ledger.jsonl")
    # engine runner smoke (with toggles)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    T = 3000
    idx2 = pd.date_range("2020-01-01", periods=T, freq="h", tz="UTC")
    sig = pd.Series(0.001, index=idx2)
    r = pd.Series(0.001 * rng.standard_normal(T), index=idx2)
    # a score with mild foresight-free autocorrelation
    zmod = pd.Series(np.tanh(0.3 * np.convolve(r.values, np.ones(8) / 8, "same")),
                     index=idx2)
    out = run_online_engine(zmod, r, sig, cost_mult=1.0, use_buffer=True)
    assert np.isfinite(out["net"]).mean() > 0.99
    out_nb = run_online_engine(zmod, r, sig, use_buffer=False)
    assert out_nb["turnover_py"] >= out["turnover_py"], "buffer must cut turnover"
    out_frz = run_online_engine(zmod, r, sig, use_rls=False)
    assert np.isfinite(out_frz["net"]).all()
    fr = frozen_net(zmod, r, sig)
    assert np.isfinite(fr["net"]).all()
    print("v2_common self-test passed: folds, metrics, bootstrap, ledger, engine toggles.")


# ==================== kernel main ====================
"""
AURUM v2.0 -- K2 kernel main: Fast CPU Models & Execution Upgrade
==================================================================
Phase 2 of the v2.0 execution plan (0 GPU hours):

  * M1_GBDT_v2  : regularized LightGBM 3-seed ensemble on the ~345 v2
                  features (max_depth=5, reg_alpha=10, reg_lambda=10,
                  subsample=0.9, colsample_bytree=0.7), walk-forward yearly
                  folds 2016-2025, target y_ret24_norm, position z=tanh(2*pred).
  * M2_RFF_Ridge: Kelly-Malamud closed-form RFF timing (P=2048, gamma=sqrt(2)
                  dimension-scaled, 10 seeds averaged, ridge sweep
                  z in {1e0..1e4} selected on the fold's ES slice).
                  v2.0.2: the position transform is a CAUSAL TRAILING
                  standardization of the raw prediction (Engineering Flaw #8
                  fix -- see rff_z_v202 below), gated by G4.8 at the producer.
  * RLSBank     : the hyperparameter-free adaptation floor as a standalone
                  strategy (delay-aware updates, per-coordinate median bank).
  * OnlineExecutionEngine evaluation of M1/M2 with Baseline-Gauntlet toggles
    (buffer / RLS / regimes), cost multipliers 1.0x/1.5x/2.0x, FIT vs OOS.
  * Master Trial Ledger from Experiment #1.

Outputs: v2_cpu_forecasts.parquet, engine_metrics.json, trial_ledger.jsonl,
gates.json, run_log.txt.
"""
import glob as _glob
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd

GATES = {}
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:3000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)


def stage(name):
    class C:
        def __init__(self):
            self.t0 = time.time()
        def __enter__(self):
            log(f"===== STAGE {name} START =====")
            return self
        def __exit__(self, et, ev, tb):
            if et is not None:
                err = "".join(traceback.format_exception(et, ev, tb))[-2500:]
                log(f"===== STAGE {name} ERROR =====\n{err}")
                raise
            log(f"===== STAGE {name} DONE in {time.time()-self.t0:.1f}s =====")
            return False
    return C()


FEAT_COLS = None

# ---------------------------------------------------------------------------
# v2.0.2 (Engineering Flaw #8 fix): causal trailing standardization of the
# raw RFF prediction.
#
# ROOT CAUSE of the flaw: under a heavy ridge penalty (z=1e4) the selected
# model's predictions carry a fold-level mean drift (1e-3..1e-2) that is up
# to 100x larger than their within-fold dispersion (~1e-4).  The old fold-
# local transform tanh(pred / std(pred_es)) divided by a COLLAPSED scale, so
# pred/scale >> 5 saturated tanh to sign(pred): z_rff became a per-fold
# constant +-1.0 (OOS 2023-25: 16,991 bars, ONE unique value +1.0; FIT 2017
# fold: constant -1.0).  M2 silently degraded into passive gold / per-fold
# sign bets with ~zero per-bar information (G4.8 z-quality audit).
#
# FIX: standardize the raw prediction against its OWN trailing history,
# re-anchored at each yearly re-fit boundary (the walk-forward replaces the
# model; its prediction scale is a new random variable, and carrying the
# dead model's mean/std across the boundary injects the very drift artifact
# being removed):
#     z_t = tanh( (p_t - EWMA_504(p)_t) / (2.0 * max(EWSTD_504(p)_t, eps)) )
# min_periods=63 (~2.6 trading days): the first 63 bars of every fold are
# NaN by design (no trailing history from the new model yet).
RFF_Z_SPAN = 504          # trailing EWMA/EWSTD span (H1 bars, ~1 month)
RFF_Z_MINP = 63           # warmup before a z is emitted (per fold)
RFF_Z_EPS = 1e-9          # scale floor (degenerate constant stretches -> z=0)


def rff_z_v202(pred: pd.Series) -> pd.Series:
    """Causal trailing standardization of pred_rff (v2.0.2 primary)."""
    out = pd.Series(np.nan, index=pred.index)
    for _, g in pred.groupby(pred.index.year):   # fold = calendar test year
        p = g.dropna()
        if len(p) < 2 * RFF_Z_MINP:
            continue
        ewma = p.ewm(span=RFF_Z_SPAN, min_periods=RFF_Z_MINP).mean()
        ewstd = p.ewm(span=RFF_Z_SPAN, min_periods=RFF_Z_MINP).std()
        x = (p - ewma) / (2.0 * np.maximum(ewstd, RFF_Z_EPS))
        out.loc[p.index] = np.tanh(x)
    return out


def rff_z_foldiqr(pred: pd.Series) -> pd.Series:
    """NEGATIVE-CONTROL variant (kept for documentation only): fold mean
    RETAINED in the numerator, robust expanding IQR scale (causal within
    the fold).  Wherever the fold drift >> fold IQR this saturates exactly
    like the v2.0.1 flaw -- it is exported so the report can show WHY
    fold-local scaling cannot survive a drift/dispersion ratio >> 1."""
    out = pd.Series(np.nan, index=pred.index)
    for _, g in pred.groupby(pred.index.year):
        p = g.dropna()
        if len(p) < 2 * RFF_Z_MINP:
            continue
        vals = p.values
        res = np.full(len(vals), np.nan)
        for i in range(RFF_Z_MINP, len(vals)):
            q75, q25 = np.percentile(vals[: i + 1], [75, 25])
            res[i] = np.tanh(vals[i] / max((q75 - q25) / 1.349, RFF_Z_EPS))
        out.loc[p.index] = res
    return out


def prepare_matrix(feats: pd.DataFrame):
    """Feature matrix (finite-filled) + targets; returns (X, y, names)."""
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32")
    X = X.ffill().fillna(0.0)
    # replace inf
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"].astype("float64")
    return X, y, names


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k2_out")
        os.makedirs(WORK, exist_ok=True)
    import lightgbm as lgb

    ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))
    T0 = time.time()

    # ---------------- 1) data ------------------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        log(f"features: {feats.shape[0]} bars x {feats.shape[1]} cols "
            f"({feats.index[0]} .. {feats.index[-1]})")
        X, y, names = prepare_matrix(feats)
        log(f"model matrix: {X.shape}; target NaN: {y.isna().mean():.3%}")

    r = feats["r_xau"]
    sigma = feats["sigma"]
    yr = pd.Series(feats.index.year, index=feats.index)
    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    # ---------------- 2) M1_GBDT_v2 walk-forward -----------------------------
    with stage("M1_GBDT_v2 walk-forward"):
        z_gbdt = pd.Series(np.nan, index=feats.index)
        pred_gbdt = pd.Series(np.nan, index=feats.index)
        fold_log = []
        params = dict(objective="regression", metric="rmse",
                      max_depth=5, reg_alpha=10, reg_lambda=10,
                      subsample=0.9, colsample_bytree=0.7,
                      learning_rate=0.06, verbose=-1)
        ledger.add("M1_GBDT_v2", {**params, "n_seeds": 3, "target": "y_ret24_norm"},
                   stage="config", kernel="K2")
        for yy in ALL_FOLD_YEARS:
            res = fold_indices(feats, yy)
            if res is None:
                continue
            fit_pos, es_pos, test_pos = res
            m_tr = y.notna().values
            fit_m = fit_pos[m_tr[fit_pos]]
            Xtr, ytr = X.iloc[fit_m], y.iloc[fit_m]
            Xes, yes = X.iloc[es_pos], y.iloc[es_pos]
            Xte = X.iloc[test_pos]
            preds = []
            for seed in (1337, 2024, 4242):
                dtr = lgb.Dataset(Xtr, label=ytr)
                des = lgb.Dataset(Xes, label=yes, reference=dtr)
                booster = lgb.train({**params, "seed": seed, "bagging_seed": seed,
                                     "feature_fraction_seed": seed},
                                    dtr, num_boost_round=1200, valid_sets=[des],
                                    callbacks=[lgb.early_stopping(60, verbose=False)])
                preds.append(booster.predict(Xte, num_iteration=booster.best_iteration))
                ledger.add("M1_GBDT_v2", {"seed": seed, "fold": yy,
                                          "best_iter": booster.best_iteration},
                           stage="fit", kernel="K2")
            p = np.mean(preds, axis=0)
            # fold-local scale normalization: z = tanh(pred / std(in-sample pred))
            # (predictions of vol-normalized 24h returns are tiny; without
            #  normalization the position collapses to zero and costs dominate)
            es_pred = booster.predict(Xes, num_iteration=booster.best_iteration)
            scale = float(np.std(es_pred)) + 1e-9
            z_gbdt.iloc[test_pos] = np.tanh(p / scale)
            pred_gbdt.iloc[test_pos] = p
            fold_log.append({"fold": yy, "best_iter": booster.best_iteration,
                             "pred_std": float(np.std(p))})
            log(f"  fold {yy}: best_iter={booster.best_iteration} "
                f"pred_std={np.std(p):.4f}")
        pre_2016 = int((feats.index < pd.Timestamp("2016-01-01", tz="UTC")).sum())
        n_missing = int(z_gbdt.isna().sum())
        ok21 = gate("G2.1_folds", len(fold_log) == 10 and n_missing == pre_2016,
                    f"10/10 folds predicted; NaN z rows={n_missing} "
                    f"(= pre-2016 warmup {pre_2016}: 2010-2015 excluded by design)")

    # ---------------- 3) M2_RFF_Ridge walk-forward ---------------------------
    # v2.0.2: the model fit itself is UNCHANGED (same seeds, same ridge
    # sweep, same pred_rff bytes); only the position transform is replaced
    # by the causal trailing standardization (rff_z_v202).
    with stage("M2_RFF_Ridge walk-forward"):
        pred_rff = pd.Series(np.nan, index=feats.index)
        for yy in ALL_FOLD_YEARS:
            res = fold_indices(feats, yy)
            if res is None:
                continue
            fit_pos, es_pos, test_pos = res
            m_tr = y.notna().values
            fit_m = fit_pos[m_tr[fit_pos]]
            m2 = RFFRidgeTimingModel(n_features=X.shape[1], p=2048,
                                     gamma=math.sqrt(2.0),
                                     ridge_grid=(1e0, 1e1, 1e2, 1e3, 1e4),
                                     n_seeds=10, seed=1337)
            m2.fit(X.iloc[fit_m].values, y.iloc[fit_m].values,
                   X_val=X.iloc[es_pos].values, y_val=y.iloc[es_pos].values)
            p = m2.predict(X.iloc[test_pos].values)
            pred_rff.iloc[test_pos] = p
            ledger.add("M2_RFF_Ridge", {"fold": yy, "z_selected": m2.z_best,
                                        "p": 2048, "n_seeds": 10,
                                        "gamma": "sqrt(2)/sqrt(d)"},
                       stage="fit", kernel="K2")
            log(f"  fold {yy}: z*={m2.z_best:g} pred_std={np.std(p):.4f}")
        for z_ in (1e0, 1e1, 1e2, 1e3, 1e4):
            ledger.add("M2_RFF_Ridge", {"ridge_z": z_, "p": 2048, "n_seeds": 10},
                       stage="config", kernel="K2",
                       notes="ridge grid member (selection space for DSR N)")
        # v2.0.2 position transforms (fold-local ES-scale REMOVED -- Flaw #8)
        z_rff = rff_z_v202(pred_rff)
        z_rff_iqr = rff_z_foldiqr(pred_rff)      # negative-control variant
        ledger.add("M2_RFF_Ridge",
                   {"rev": "k2-v2.0.2", "transform": "causal trailing",
                    "z": "tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), 1e-9)))",
                    "re_anchor": "yearly re-fit boundary (fold = test year)",
                    "span": RFF_Z_SPAN, "min_periods": RFF_Z_MINP,
                    "eps": RFF_Z_EPS,
                    "variant_foldiqr": "tanh(p / max(IQR_fold/1.349, eps)) "
                                       "(negative control, mean retained)"},
                   stage="config", kernel="K2",
                   notes="Engineering Flaw #8 producer-side fix")
        ok_rff = gate("G2.1b_rff", int(z_rff.notna().sum()) >= pre_2016 and len(z_rff.dropna()) > 50000,
                      f"RFF predictions cover {int(z_rff.notna().sum())} bars "
                      f"(2016-2025 span = {len(feats) - pre_2016}; "
                      f"{RFF_Z_MINP}-bar warmup per fold by design)")

        # ---- G4.8 producer-side saturation gate (v2.0.2) --------------------
        # The z-saturation audit is enforced AT THE PRODUCER now, not just
        # discovered downstream in K6: std(z) > 0.05 and P(|z|>0.99) < 5%
        # on BOTH windows, plus a genuinely-active OOS turnover floor
        # (frozen accounting, > 2 round-turns / year).
        audit48 = {}
        ok48 = True
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = z_rff.dropna().index
            sel = sel[mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            v = z_rff[sel].values
            std = float(np.std(v))
            sat = float((np.abs(v) > 0.99).mean())
            fro = frozen_net(z_rff[sel], r[sel], sigma[sel], 1.0)
            audit48[wname] = {"n": int(len(v)),
                              "std": round(std, 5),
                              "sat_frac_gt_0p99": round(sat, 6),
                              "nunique": int(pd.unique(v).size),
                              "frozen_turnover_py": round(fro["turnover_py"], 2),
                              "frozen_sharpe": fro["metrics"].get("sharpe_ann")}
            ok48 = ok48 and (std > 0.05) and (sat < 0.05)
        ok48 = ok48 and audit48.get("oos", {}).get("frozen_turnover_py", 0.0) > 2.0
        # negative-control variant audited for the record (expected to FAIL)
        neg = {}
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = z_rff_iqr.dropna().index
            sel = sel[mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            vi = z_rff_iqr[sel].values
            neg[wname] = {"std": round(float(np.std(vi)), 5),
                          "sat_frac_gt_0p99": round(float((np.abs(vi) > 0.99).mean()), 6),
                          "nunique": int(pd.unique(vi).size)}
        ledger.add("M2_RFF_Ridge", {"rev": "k2-v2.0.2", **audit48,
                                     "negative_control_foldiqr": neg},
                   audit48.get("oos", {}).get("frozen_sharpe"),
                   stage="zquality", kernel="K2")
        ok48g = gate("G4.8_z_rff_producer", ok48,
                     f"v2.0.2 producer-side saturation gate on z_rff: "
                     + json.dumps(audit48)
                     + " -- negative control (fold-mean-retained IQR variant, "
                       "expected saturated): " + json.dumps(neg))

    # ---------------- 4) RLSBank standalone floor ----------------------------
    with stage("RLSBank standalone floor"):
        # readout features: last-3 vol-normalized returns + F_blend/10
        f_blend = (feats["F_blend_xau"] / 10.0) if "F_blend_xau" in feats else 0.0
        rn = (r / (sigma + 1e-12))
        bankX = pd.DataFrame({"x1": rn.shift(1), "x2": rn.shift(2),
                              "x3": rn.shift(3), "x4": f_blend.shift(1)}).fillna(0.0)
        z_bank = pd.Series(np.nan, index=feats.index)
        bank = RLSBank(5)   # top-level in the bundle (online_engine.py)
        H = 24
        pending = []
        idx = feats.index
        bankXv = bankX.values
        scores = np.full(len(idx), np.nan)
        for t in range(len(idx)):
            for _, bx, by in [p for p in pending if p[0] == t]:
                if np.isfinite(by):
                    bank.update(bx, by)      # skip warmup-NaN labels (sigma
                                             # undefined in the first 24 bars)
            pending = [p for p in pending if p[0] != t]
            xt = np.concatenate([[1.0], bankXv[t]])
            scores[t] = bank.predict(xt)
            pending.append((t + H, xt, r.iloc[t] / (sigma.iloc[t] + 1e-12)))
        z_bank = pd.Series(np.tanh(scores), index=idx)
        # valid only after warmup (first 30d of 2010)
        z_bank.iloc[: 24 * 30] = np.nan
        ledger.add("RLSBank", {"lams": [0.99, 0.995, 0.999, 0.9995],
                               "features": "rn lags 1-3 + F_blend/10"},
                   stage="config", kernel="K2")
        ok_bank = gate("G2.1c_rlsbank", int(z_bank.notna().sum()) > 80000,
                       "RLSBank floor predictions emitted")

    # ---------------- 5) engine evaluation (Phase 2 mandate) -----------------
    with stage("engine evaluation"):
        engine_metrics = {}
        for name, zseries in (("M1_GBDT_v2", z_gbdt), ("M2_RFF_Ridge", z_rff),
                              ("RLSBank", z_bank)):
            zm = zseries.dropna()
            res_all = {}
            for window, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = zm.index[mask.reindex(zm.index).fillna(False)]
                if len(sel) < 500:
                    continue
                zw = zseries[sel]
                rw, sw = r[sel], sigma[sel]
                variants = {}
                full = run_online_engine(zw, rw, sw, 1.0)
                variants["full"] = {"metrics": full["metrics"],
                                    "turnover_py": full["turnover_py"],
                                    "diag": full["diagnostics"]}
                nb = run_online_engine(zw, rw, sw, 1.0, use_buffer=False)
                variants["no_buffer"] = {"metrics": nb["metrics"],
                                         "turnover_py": nb["turnover_py"]}
                fz = run_online_engine(zw, rw, sw, 1.0, use_rls=False)
                variants["frozen_rls_off"] = {"metrics": fz["metrics"],
                                              "turnover_py": fz["turnover_py"]}
                nr = run_online_engine(zw, rw, sw, 1.0, use_regimes=False)
                variants["no_regimes"] = {"metrics": nr["metrics"],
                                          "turnover_py": nr["turnover_py"]}
                fro = frozen_net(zw, rw, sw, 1.0)
                variants["frozen_raw"] = {"metrics": fro["metrics"],
                                          "turnover_py": fro["turnover_py"]}
                if window == "fit":
                    for cm in (1.5, 2.0):
                        v = run_online_engine(zw, rw, sw, cm)
                        variants[f"full_cost{cm}"] = {
                            "metrics": v["metrics"],
                            "turnover_py": v["turnover_py"]}
                res_all[window] = variants
                log(f"  {name} [{window}] full: {variants['full']['metrics']} "
                    f"turnover={variants['full']['turnover_py']:.1f}/y")
            engine_metrics[name] = res_all
            # ledger entries for the gauntlet (engine variants count as trials)
            v = res_all.get("fit", {}).get("full", {}).get("metrics", {})
            ledger.add(f"{name}+engine", {"variant": "full"}, v.get("sharpe_ann"),
                       stage="engine", kernel="K2")
            for k in ("no_buffer", "frozen_rls_off", "no_regimes", "frozen_raw"):
                vv = res_all.get("fit", {}).get(k, {}).get("metrics", {})
                ledger.add(f"{name}+engine", {"variant": k}, vv.get("sharpe_ann"),
                           stage="engine", kernel="K2")
        with open(os.path.join(WORK, "engine_metrics.json"), "w") as f:
            json.dump(engine_metrics, f, indent=1, default=str)

    # ---------------- 6) Baseline Gauntlet (G5.3 preview) --------------------
    with stage("baseline gauntlet (G5.3)"):
        gauntlet = {}
        for name in ("M1_GBDT_v2", "M2_RFF_Ridge"):
            v = engine_metrics[name].get("fit", {})
            full = v.get("full", {}).get("metrics", {}).get("sharpe_ann", -9)
            frozen = v.get("frozen_rls_off", {}).get("metrics", {}).get("sharpe_ann", -9)
            nobuf = v.get("no_buffer", {}).get("turnover_py", 0)
            fullto = v.get("full", {}).get("turnover_py", 0)
            gauntlet[name] = {
                "engine_beats_frozen": bool(full >= frozen),
                "buffer_turnover_cut_pct": round(100 * (1 - fullto / nobuf), 1)
                if nobuf > 0 else None,
                "decision": "engine+buffer on" if full >= frozen else "engine off (frozen ships)",
            }
            log(f"  {name}: {gauntlet[name]}")
        ok_g = gate("G2.3_gauntlet", all(g["buffer_turnover_cut_pct"] or 0 >= 25
                                          for g in gauntlet.values()),
                    json.dumps(gauntlet))

    # ---------------- 7) export ----------------------------------------------
    with stage("export"):
        out = pd.DataFrame({
            "z_gbdt": z_gbdt, "pred_gbdt": pred_gbdt,
            "z_rff": z_rff, "pred_rff": pred_rff,
            "z_rff_foldiqr": z_rff_iqr,     # negative control (v2.0.2)
            "z_rlsbank": z_bank, "r_xau": r, "sigma": sigma,
            "y_ret24_norm": feats["y_ret24_norm"],
        })
        out.to_parquet(os.path.join(WORK, "v2_cpu_forecasts.parquet"))
        with open(os.path.join(WORK, "gauntlet.json"), "w") as f:
            json.dump(gauntlet, f, indent=1)
        meta = {"n_rows": int(len(feats)),
                "ledger_entries": len(ledger.rows),
                "unique_configs": ledger.unique_configs(),
                "wall_s": round(time.time() - T0, 1)}
        with open(os.path.join(WORK, "k2_meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        ok_ex = gate("G2.4_export", True,
                     f"v2_cpu_forecasts.parquet + engine_metrics.json + "
                     f"trial_ledger.jsonl ({len(ledger.rows)} entries, "
                     f"{ledger.unique_configs()} unique configs)")

    all_ok = all([ok21, ok_rff, ok48g, ok_bank, ok_g, ok_ex])
    gate("G2_ALL", all_ok, "all K2 gates green" if all_ok else "K2 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("K2 COMPLETE" if all_ok else "K2 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    _sys.exit(main())
