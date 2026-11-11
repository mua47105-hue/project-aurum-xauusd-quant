# ============================================================================
# Auto-generated AURUM v2.0 kernel: AURUM v2 Live Daily
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
AURUM v2.0 -- Live Daily kernel main (v2.0.2 decision state)
===========================================================
Phase 5: emits the full v2.0 decision object, recomputable from kernel +
datasets alone: continuous target exposure z_t, Carver buffer edge status,
PH / vol-of-vol / drawdown-governor regime multipliers, catastrophic stop
level, cost-budget usage vs the 1/3 speed limit, and the v1.3 barrier
reference.

v2.0.2: the live model remains M1_GBDT_v2 -- the ONLY active CPU model
that passes the Carver speed limit (OOS engine Sharpe 0.938); the
de-saturated M2_RFF_Ridge active content measured as noise (engine FIT
-0.736 / OOS -0.744) and is reference-only.  The K6 v2.0.2 tournament
verdict is embedded as stack_context when the K6 outputs are available.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

GATES = {}
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:2000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(ok)


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_out")
        os.makedirs(WORK, exist_ok=True)
    import lightgbm as lgb

    # ---------------- 1) data ------------------------------------------------
    feats, px = load_v2_data()
    r = feats["r_xau"]
    sigma = feats["sigma"]
    idx = feats.index
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32").ffill().fillna(0.0)
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"]
    asof = idx[-1]
    log(f"as-of bar: {asof} (features frozen at dataset version)")

    # ---------------- 2) M1_GBDT_v2 final refit ------------------------------
    m = y.notna().values
    train_pos = np.arange(len(idx))[m]
    train_pos = train_pos[train_pos < len(idx) - 24]
    params = dict(objective="regression", metric="rmse", max_depth=5,
                  reg_alpha=10, reg_lambda=10, subsample=0.9,
                  colsample_bytree=0.7, learning_rate=0.06, verbose=-1)
    preds = []
    for seed in (1337, 2024, 4242):
        dtr = lgb.Dataset(X.iloc[train_pos], label=y.iloc[train_pos])
        booster = lgb.train({**params, "seed": seed}, dtr, num_boost_round=300)
        preds.append(booster.predict(X.iloc[[-1]])[0])
    pred_now = float(np.mean(preds))
    z_model = float(np.tanh(2.0 * pred_now))

    # ---------------- 3) engine state at the latest bar ----------------------
    k2dir = os.environ.get("AURUM_K2DIR") or _find_dir("v2_cpu_forecasts.parquet",
                      "/kaggle/input/aurum-v2-forecasts-cpu",
                      "/kaggle/input/aurum-v2-cpu-preds",
                      "/kaggle/input/datasets/tskdev/aurum-v2-cpu-preds",
                      *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
                      *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-cpu-preds*")))
    z_hist = None
    if k2dir:
        fc = pd.read_parquet(os.path.join(k2dir, "v2_cpu_forecasts.parquet"))
        fc.index = pd.DatetimeIndex(fc.index)
        if fc.index.tz is None:
            fc.index = fc.index.tz_localize("UTC")
        if "z_gbdt" in fc:
            z_hist = fc["z_gbdt"].reindex(idx)
    if z_hist is None or z_hist.iloc[-48:].notna().sum() < 24:
        z_hist = pd.Series(np.nan, index=idx)
        step = 24 * 30
        for t0 in range(len(idx) - 2 * 365 * 24, len(idx), step):
            tp = np.arange(t0 - 24)[m[:t0 - 24]] if t0 > 24 else np.array([])
            if len(tp) < 5000:
                continue
            ps = []
            dtr = lgb.Dataset(X.iloc[tp], label=y.iloc[tp])
            b = lgb.train({**params, "seed": 1337}, dtr, num_boost_round=200)
            ps.append(b.predict(X.iloc[t0: t0 + step]))
            z_hist.iloc[t0: t0 + step] = np.tanh(2.0 * np.mean(ps, axis=0))
        z_hist = z_hist.ffill()
    z_run = z_hist.fillna(0.0)

    ph = PageHinkley(lam=50.0)
    vt = VolTarget(SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR))
    buf = CarverBuffer()
    dd = DrawdownGovernor()
    z_cur, avg_abs_z, freeze_until, last_net = 0.0, 0.05, -1, 0.0
    vov_series = []
    zs = []
    for t in range(len(idx) - 3 * 365 * 24, len(idx)):
        rt = float(r.iloc[t]); st = float(sigma.iloc[t])
        ph.update(rt / (st + 1e-12))
        if ph.n == 0:
            freeze_until = t + 24
        vt.update(st)
        m_vol = vt.multiplier(st)
        vov = float(sigma.iloc[max(0, t - 24 * 30): t].std())
        vov_series.append(vov)
        m_volov = 1.0
        if len(vov_series) > 24 * 30 and np.isfinite(vov):
            q = np.quantile(vov_series, 0.8)
            if vov > q:
                m_volov = 0.5
        m_dd = dd.step(last_net)
        regime = m_vol * m_volov * m_dd * (0.5 if t < freeze_until else 1.0)
        z_ideal = float(np.tanh(z_run.iloc[t])) * regime
        z_new = buf.apply(z_cur, z_ideal, avg_abs_z, 1.0)
        z_new = float(np.clip(z_new, -2.0, 2.0))
        net = z_cur * rt - COST_R * abs(z_new - z_cur)
        last_net = net
        avg_abs_z += (abs(z_new) - avg_abs_z) / max(1, t + 1)
        z_cur = z_new
        zs.append(z_new)

    regime_now = m_vol * m_volov * m_dd * (0.5 if len(idx) - 1 < freeze_until else 1.0)
    z_ideal_now = float(np.tanh(z_run.iloc[-1])) * regime_now
    band = buf.buffer_size(avg_abs_z, 1.0)
    inside_band = abs(z_cur - z_ideal_now) <= band
    rng14 = (px["h"] - px["l"]) if "h" in px.columns else None
    if rng14 is not None:
        atr14 = float(rng14.rolling(14).mean().reindex(idx).iloc[-1])
    else:
        atr14 = float(sigma.iloc[-1] * math.sqrt(24))
    close_now = float(feats["close_xau"].iloc[-1]) if "close_xau" in feats else None

    v13 = {"source": "tskdev/aurum-v13-artifacts (v1.3 flagship policy)",
           "champion": "flagship cap=3, OOS Sharpe 2.444, MaxDD 13.06%, DSR 0.895"}
    v13dir = _find_dir("trades_all.parquet", "/kaggle/input/aurum-v13-artifacts",
                       *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v13*")))
    if v13dir:
        tr = pd.read_parquet(os.path.join(v13dir, "trades_all.parquet"))
        sub = tr[(tr["strategy"] == "flagship") & (tr["cap"] == 3) & (tr["cost_mult"] == 1.0)]
        last = sub.iloc[-1]
        v13["latest_trade"] = {
            "cell": str(last["cell_name"]),
            "entry_ts": str(pd.Timestamp(last["ts"], unit="ns", tz="UTC")),
            "p": float(last["p"]), "ev": float(last["ev"]),
            "f_eff": float(last["f_eff"]), "exit_bar": int(last["exit_bar"])}
        v13["champion_open_exposure"] = float(
            (1.0 if int(last["exit_bar"]) >= len(idx) - 1 else 0.0) *
            (1 if str(last["cell_name"]).startswith("L") else -1))

    # v2.0.2 tournament verdict (embedded when the K6 outputs are present)
    stack_context = {
        "spec": "v2.0.2",
        "live_model": "M1_GBDT_v2 (only speed-limit-passing active CPU model)",
        "m2_rff_v202": "de-saturated via causal trailing standardization "
                       "(G4.8 producer-gated); active content measured as "
                       "noise -- reference-only",
        "m6_family": "fixed a-priori {M1_GBDT_v2, M2_RFF_Ridge}, 50/50 "
                     "(Carver S24 SR-shrink 0.90 toward equal weights)",
    }
    k6dir = os.environ.get("AURUM_K6OUT")
    if not k6dir:
        k6dir = _find_dir("backtest_v2.json",
                          "/kaggle/input/aurum-v2-backtest",
                          "/kaggle/input/datasets/tskdev/aurum-v2-backtest",
                          *sorted(__import__("glob").glob(
                              "/kaggle/input/*/aurum-v2-backtest*")))
    if k6dir and os.path.isfile(os.path.join(k6dir, "champion_compare.json")):
        try:
            cc = json.load(open(os.path.join(k6dir, "champion_compare.json")))
            for k in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                      "M0_Champ_v13", "M6_AURUM_v2_Ensemble", "M2_RFF_Ridge",
                      "M1_GBDT_v2"):
                if k in cc:
                    stack_context[k] = {
                        "oos_sharpe": cc[k].get("sharpe_ann"),
                        "cost2x_sharpe": cc[k].get("cost2x_sharpe"),
                        "oos_dsr": cc[k].get("oos_dsr")}
        except Exception:
            pass

    decision = {
        "as_of_bar": str(asof),
        "symbol": "XAUUSD",
        "spec": "v2.0.2",
        "v2_state": {
            "model": "M1_GBDT_v2 (3-seed, walk-forward-consistent refit)",
            "pred_y_ret24_norm": round(pred_now, 5),
            "z_model_score": round(z_model, 5),
            "z_target_exposure": round(z_ideal_now, 5),
            "z_executed": round(z_cur, 5),
            "regime": {"vol_target_mult": round(m_vol, 4),
                        "vol_of_vol_mult": m_volov,
                        "dd_governor_scale": m_dd,
                        "ph_frozen": bool(len(idx) - 1 < freeze_until)},
            "carver_buffer": {"band": round(band, 5), "avg_abs_z": round(avg_abs_z, 5),
                               "inside_band": bool(inside_band),
                               "distance_to_edge": round(band - abs(z_cur - z_ideal_now), 5)},
            "catastrophic_stop": {"frame": "2.5 x ATR14", "atr14": round(atr14, 3),
                                   "long_stop_px": round(close_now - 2.5 * atr14, 2)
                                   if close_now else None,
                                   "short_stop_px": round(close_now + 2.5 * atr14, 2)
                                   if close_now else None},
            "cost_budget": {"cost_r": COST_R,
                             "speed_limit": "annual cost <= 1/3 pre-cost SR",
                             "recent_turnover_py": round(float(
                                 np.abs(np.diff(zs[-24 * 90:])).mean()
                                 * BARS_PER_YEAR / 2), 2)},
        },
        "v13_barrier_reference": v13,
        "stack_context_v202": stack_context,
        "provenance": {"features": "tskdev/aurum-v2-featstore (frozen version)",
                        "engine": "v2.0 OnlineExecutionEngine components "
                                  "(river-exact PH, Moreira-Muir vol target, "
                                  "Carver buffer, AHL DD governor)",
                        "recomputable": True},
    }
    with open(os.path.join(WORK, "decision_v2.json"), "w") as f:
        json.dump(decision, f, indent=1)

    ok = gate("G6.4_live_decision", True,
              f"v2.0 decision emitted at {asof}: z_target={z_ideal_now:.4f} "
              f"z_exec={z_cur:.4f} regime={round(regime_now, 3)} "
              f"buffer_inside={inside_band}")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("LIVE COMPLETE" if ok else "LIVE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
