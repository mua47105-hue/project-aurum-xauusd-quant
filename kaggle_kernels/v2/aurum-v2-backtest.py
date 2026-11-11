# ============================================================================
# Auto-generated AURUM v2.0 kernel: AURUM v2 Backtest
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
AURUM v2.0 -- K6 kernel main (rev v2.0.2): Full Walk-Forward Backtest &
Statistical Gates
=============================================================================
Phase 4+5 of the v2.0 execution plan (CPU):

Evaluates EVERY strategy in the single-position causal event loop:
  M0_Champ_v13 (v1.3 flagship cap=3, frozen translation), M0_Champ_v13_cap1,
  M1_GBDT_v2, M2_RFF_Ridge, RLSBank, M3c_DAE, M3m_DAE, M4_DMN_Lite,
  M4b_DMN_Lite, M5_ContinuumNet, M6_AURUM_v2_Ensemble (Carver shrunk-MV
  stack over the FIXED active CPU family), M6b_Ensemble_EQW
  (equal-weight robustness variant), M7_Active_Grand_Hybrid (50% v1.3
  champion + 50% ACTIVE de-saturated v2 stack), M7_Core_Satellite
  (50% v1.3 champion + 50% passive gold sleeve -- the transparent
  continuation of the v2.0.1 M7 stream), Buy & Hold Gold, Carver EWMAC
  standalone, Channel A / Channel B standalone.

v2.0.1 CHANGELOG (post-causality-audit revision):
  * frozen_net one-bar lookahead FIXED (z[t]*r[t] -> w[t-1]*r[t]); the
    engine loop was already causal (net = z_cur * r_arr[t]).
  * G4.7_causality_v201 regression gate: the same-bar formulation is
    deliberately re-created on the Channel_A baseline and must show the
    pathological Sharpe inflation (trap detected), while the fixed
    accounting must show NO residual same-bar edge (Channel_A ~ -0.13).
  * Data-signature checkpoints: every per-strategy result is cached under
    k6_out/ckpt/ keyed by an md5 of (code revision, strategy name, full
    z-series bytes).  When an input forecast is replaced (e.g. M5 gaining
    the complete 2016-2025 K3c folds, M4b, or the M6/M7 composites being
    rebuilt on top of them) the signature changes and ONLY the affected
    strategies re-evaluate -- cached results for unchanged inputs are
    reused verbatim.
  * M7_Grand_Hybrid is the FIXED 50/50 blend (a priori, selection-free);
    the FIT-grid blend scan is retained as a diagnostic only (ledger'd).

v2.0.2 CHANGELOG (Engineering Flaw #8 + active-stack verification):
  * Producer-side z_rff rebuild (K2 rev k2-v2.0.2): causal trailing
    standardization tanh((p - EWMA_504(p)) / (2*max(EWSTD_504(p), eps)))
    re-anchored per yearly re-fit fold, gated by G4.8 AT THE PRODUCER.
    The v2.0.1 fold-local ES-scale transform saturated z_rff to per-fold
    sign bets (OOS nunique=1) -- M2/M6/M6b/M7 silently degenerated into
    passive gold exposure.  pred_rff is byte-identical; only the position
    transform changed.
  * M6_AURUM_v2_Ensemble rebuilt per Carver S24: the family is the FIXED
    a-priori active CPU pair {M1_GBDT_v2, M2_RFF_Ridge} (the v2.0.1
    positive-FIT membership filter on the UNBUFFERED frozen_net FIT Sharpe
    dropped M1_GBDT_v2 -- which has a clean active 0.938 OOS Sharpe --
    for a -0.24 FIT estimate); SR inputs are now the BUFFERED ENGINE FIT
    Sharpes (the shipped execution configuration per the gauntlet), and
    the SR estimates are shrunk 90% toward the equal-weight prior
    (sr' = 0.10*sr + 0.90*mean(sr)); corr-shrink 0.50; clip >= 0 with the
    equal-weight fallback when the shrunk SRs cannot rank members.
  * M7 split into two explicit side-by-side benchmarks:
      M7_Active_Grand_Hybrid = 0.5*v1.3 champion + 0.5*ACTIVE de-saturated
        v2 stack (M6) -- measures the TRUE active orthogonality gain;
      M7_Core_Satellite = 0.5*v1.3 champion + 0.5*passive gold sleeve --
        the transparent accounting of what the v2.0.1 M7 stream actually
        was in OOS (bit-identical net stream; preserves 2.6351 / 2.5012@2x).
  * SHIPPING_FAMILY pruned to the cost-viable configurations: the
    de-saturated M2/M6/M6b FAIL the Carver speed limit (annual cost > 1/3
    pre-cost SR -- the de-saturated active stack is not cost-viable at H1
    frequency), so they are demoted to reference-only (flags still
    exported and discussed).  M7_Active passes the speed limit and stays
    in the shipping family as the active measurement arm.
  * G6.1 DSR flagship = M7_Core_Satellite (direct continuation of the
    v2.0.1 gated stream); M7_Active DSR reported alongside.

Windows: FIT 2016-2022, OOS 2023-2025 (untouched), year-by-year 2016-2025.
Costs: 1.0x / 1.5x / 2.0x.  Carver speed-limit audit (cost <= 1/3 pre-cost SR).
Statistics: 5,000-sample block-bootstrap CIs, Deflated Sharpe Ratio over the
ENTIRE trial ledger (K2+K3+K6), CSCV-PBO on the selection family.

Outputs: backtest_v2.json, year_table.csv, net_series.parquet, stats.json,
gates.json, champion_compare.json, trial_ledger.jsonl, run_log.txt,
gauntlet.json, k6_meta.json, ckpt/.
"""
import hashlib
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_HERE, "..", "model"), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


CODE_REV = "k6-v2.0.2"

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


# v1.3 champion reference numbers (from tskdev/aurum-p4-backtest v1.3 output,
# backtest_summary.md -- OOS 2023-25, cost 1.0x, trade-based barrier accounting)
V13_REFERENCE = {
    "flagship_cap3": {"sharpe_ann": 2.444, "cagr_pct": 54.56, "maxdd_pct": 13.06,
                      "er_per_trade": 0.1478, "pf": 1.258, "n_trades": 3341,
                      "dsr": 0.895},
    "flagship_cap1": {"sharpe_ann": 2.305, "cagr_pct": 53.18, "maxdd_pct": 14.67},
    "aurum_r1_cap1": {"sharpe_ann": 2.194},
    "gbdt_cap1_v13": {"sharpe_ann": 0.307, "er_per_trade": 0.0310},
    "buyhold_cap1": {"sharpe_ann": 1.919, "cagr_pct": 23.61, "maxdd_pct": 10.80},
}

# strategies evaluated under frozen translation only (no v2 engine overlay):
# the v1.3 champion policies are REPORT-ONLY translations of the barrier
# trade stream into continuous accounting.
FORCED_FROZEN = {"M0_Champ_v13", "M0_Champ_v13_cap1"}

# Carver speed-limit (annual cost <= 1/3 pre-cost SR) is a SHIP-ABILITY
# design constraint: it is gated on the shipped/champion family.  The audit
# is still computed and exported for every strategy; tournament also-rans
# and baselines with non-positive pre-cost Sharpe are annotated as
# reference-only (the 1/3 rule is undefined for a dead signal).
# v2.0.2: the de-saturated M2/M6/M6b legitimately FAIL the speed limit
# (cost drag 6.2x / 2.5x the budget at H1 turnover) -- demoted to
# reference-only; their flags are exported and root-caused in the report.
SHIPPING_FAMILY = {"M7_Core_Satellite", "M7_Active_Grand_Hybrid",
                   "M0_Champ_v13", "M0_Champ_v13_cap1", "BuyHold_Gold"}

# neural forecast column -> tournament strategy name
NN_MAP = {"z_M3c": "M3c_DAE", "z_M3m": "M3m_DAE", "z_M4": "M4_DMN_Lite",
          "z_M4b": "M4b_DMN_Lite", "z_M5": "M5_ContinuumNet"}

# M6 stack family (v2.0.2, Carver S24): the FIXED a-priori active CPU pair
# -- no FIT-window membership filter (the v2.0.1 positive-FIT rule on the
# unbuffered frozen_net FIT Sharpe dropped M1_GBDT_v2, whose engine FIT
# estimate is the selection-relevant metric; membership is a design choice,
# weighting is where the evidence enters).  SR inputs = buffered ENGINE
# FIT Sharpes from the gauntlet; SR estimates shrunk 90% toward the
# equal-weight prior; corr-shrink 0.50 toward identity.
STACK_CANDIDATES = ["M1_GBDT_v2", "M2_RFF_Ridge"]
SR_SHRINK_TO_EQUAL = 0.90      # Carver shrinkage_SR: 90% toward equal weights
CORR_SHRINK = 0.50


def champion_exposure(trades: pd.DataFrame, index: pd.DatetimeIndex,
                      cap: int = 3) -> pd.Series:
    """v1.3 flagship per-bar exposure from its walk-forward trades.

    VERIFIED MAPPING: exit_bar is the absolute bar position in the SAME
    2010-based H1 frame (first 2016-01-04 entry sits at frame position 36521
    with exit_bar 36522; exit_bar max 94948 < 95008).  Each active trade
    contributes dir/cap; the sum is clipped to [-1, 1].  Causal by
    construction (the trades are the v1.3 policy's own walk-forward output).
    """
    z = np.zeros(len(index))
    sub = trades[(trades["strategy"] == "flagship") &
                 (trades["cap"] == cap) &
                 (trades["cost_mult"] == 1.0)]
    for row in sub.itertuples():
        d = 1.0 if row.cell_name.startswith("L") else -1.0
        i0 = index.searchsorted(pd.Timestamp(row.ts, unit="ns", tz="UTC"))
        i1 = int(row.exit_bar)
        if i0 >= len(index):
            continue
        i1 = min(len(index), max(i1, i0 + 1))
        z[i0:i1] += d / cap
    return pd.Series(np.clip(z, -1.0, 1.0), index=index)


def samebar_net_sharpe(z: pd.Series, r: pd.Series, sigma: pd.Series) -> float:
    """The ORIGINAL (buggy) same-bar formulation, kept ONLY as the negative
    control for gate G4.7: credits z[t]*r[t] -- a one-bar lookahead."""
    idx = z.index
    zv = z.values.astype("float64")
    rv = r.reindex(idx).values
    sv = sigma.reindex(idx).values
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
    net = zv * v * rv
    net = net[np.isfinite(net)]
    if len(net) < 20:
        return float("nan")
    return float(net.mean() / (net.std(ddof=1) + 1e-12) * math.sqrt(BARS_PER_YEAR))


# ----------------------------------------------------------------------------
# data-signature checkpoints
# ----------------------------------------------------------------------------
def z_sig(z: pd.Series) -> str:
    arr = z.dropna()
    if len(arr) == 0:
        return "empty"
    h = hashlib.md5()
    h.update(f"{arr.index[0]}|{arr.index[-1]}|{len(arr)}|".encode())
    h.update(np.ascontiguousarray(arr.values.astype("float64")).tobytes())
    return h.hexdigest()[:12]


def sig_of(name: str, z: pd.Series, extra: str = "") -> str:
    return hashlib.md5(f"{CODE_REV}|{name}|{z_sig(z)}|{extra}".encode()).hexdigest()[:12]


class Ckpt:
    """Per-strategy checkpoint store keyed by data signatures.  A checkpoint
    is reused iff the strategy name, the code revision and the FULL z-series
    bytes are unchanged -- replacing a forecast (M5 complete folds, M4b,
    rebuilt composites) automatically invalidates only the affected keys."""

    def __init__(self, root):
        self.root = root
        os.makedirs(os.path.join(root, "fitgauntlet"), exist_ok=True)
        os.makedirs(os.path.join(root, "evalgrid"), exist_ok=True)

    def load_json(self, sub, name):
        p = os.path.join(self.root, sub, f"{name}.json")
        if os.path.isfile(p):
            try:
                with open(p) as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def save_json(self, sub, name, obj):
        p = os.path.join(self.root, sub, f"{name}.json")
        with open(p, "w") as f:
            json.dump(obj, f, indent=1, default=str)

    def net_path(self, name, window):
        return os.path.join(self.root, "evalgrid", f"{name}__{window}.parquet")

    def load_net(self, name, window):
        p = self.net_path(name, window)
        if os.path.isfile(p):
            s = pd.read_parquet(p)["net"]
            s.index = pd.DatetimeIndex(s.index)
            if s.index.tz is None:
                s.index = s.index.tz_localize("UTC")
            return s
        return None

    def save_net(self, name, window, net: pd.Series):
        pd.DataFrame({"net": net.values}, index=net.index).to_parquet(
            self.net_path(name, window))


def main():
    WORK = os.environ.get("AURUM_K6OUT", "/kaggle/working")
    if WORK == "/kaggle/working" and not os.path.exists(WORK):
        WORK = os.path.join(_HERE, "k6_out")
    os.makedirs(WORK, exist_ok=True)
    ck = Ckpt(os.path.join(WORK, "ckpt"))

    t_wall = time.time()

    # ---------------- 1) load everything ------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        r = feats["r_xau"]
        sigma = feats["sigma"]
        yr = pd.Series(feats.index.year, index=feats.index)
        idx = feats.index

        k2dir = os.environ.get("AURUM_K2DIR") or _find_dir(
            "v2_cpu_forecasts.parquet",
            "/kaggle/input/aurum-v2-cpu-preds",
            "/kaggle/input/datasets/tskdev/aurum-v2-cpu-preds",
            "/kaggle/input/aurum-v2-forecasts-cpu",
            "/kaggle/input/aurum-v2-forecasts",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-cpu-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-cpu-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-forecasts*")))
        assert k2dir, "K2 forecasts dataset not found"
        fc_cpu = pd.read_parquet(os.path.join(k2dir, "v2_cpu_forecasts.parquet"))
        fc_cpu.index = pd.DatetimeIndex(fc_cpu.index)
        if fc_cpu.index.tz is None:
            fc_cpu.index = fc_cpu.index.tz_localize("UTC")

        k3dir = os.environ.get("AURUM_K3DIR") or _find_dir(
            "v2_neural_forecasts.parquet",
            "/kaggle/input/aurum-v2-neural-preds",
            "/kaggle/input/datasets/tskdev/aurum-v2-neural-preds",
            "/kaggle/input/aurum-v2-forecasts-neural",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-neural-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-neural-preds*")),
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v2-forecasts*")))
        fc_nn = None
        if k3dir:
            fc_nn = pd.read_parquet(os.path.join(k3dir, "v2_neural_forecasts.parquet"))
            fc_nn.index = pd.DatetimeIndex(fc_nn.index)
            if fc_nn.index.tz is None:
                fc_nn.index = fc_nn.index.tz_localize("UTC")

        v13dir = os.environ.get("AURUM_V13DIR") or _find_dir(
            "trades_all.parquet", "/kaggle/input/aurum-v13-artifacts",
            "/kaggle/input/datasets/tskdev/aurum-v13-artifacts",
            *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v13*")),
            *sorted(__import__("glob").glob("/kaggle/input/datasets/*/aurum-v13*")))
        assert v13dir, "v1.3 artifacts dataset not found"
        trades = pd.read_parquet(os.path.join(v13dir, "trades_all.parquet"))

        # trial ledger: seed ONCE from the K2/K3 published ledgers, then add
        # every K6 configuration on top (dedup by config_hash on resume).
        ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))
        if not ledger.rows:
            for d in (k2dir, k3dir):
                p = os.path.join(d, "trial_ledger.jsonl") if d else None
                if p and os.path.isfile(p):
                    with open(p) as f:
                        for ln in f:
                            ln = ln.strip()
                            if ln:
                                ledger.rows.append(json.loads(ln))
                    with open(os.path.join(WORK, "trial_ledger.jsonl"), "a") as f:
                        f.write(open(p).read().rstrip("\n") + "\n")
        log(f"forecasts: cpu cols={list(fc_cpu.columns[:6])}; "
            f"neural={'loaded' if fc_nn is not None else 'MISSING'}; "
            f"v13 trades={len(trades)}; ledger={len(ledger.rows)} rows")

    # ---------------- 2) strategy z-series ----------------------------------
    with stage("strategy panel"):
        S = {}
        S["M1_GBDT_v2"] = fc_cpu["z_gbdt"]
        S["M2_RFF_Ridge"] = fc_cpu["z_rff"]
        if "z_rlsbank" in fc_cpu:
            S["RLSBank"] = fc_cpu["z_rlsbank"]
        if fc_nn is not None:
            for c, name in NN_MAP.items():
                if c in fc_nn.columns:
                    S[name] = fc_nn[c]
        # baselines (features are causal)
        if "F_blend_xau" in feats:
            S["Carver_EWMAC"] = (feats["F_blend_xau"] / 20.0).clip(-1, 1)
        S["Channel_A"] = feats["q_A"].clip(-1, 1)
        S["Channel_B"] = feats["q_B"].clip(-1, 1)
        S["BuyHold_Gold"] = pd.Series(1.0, index=idx)
        S["M0_Champ_v13"] = champion_exposure(trades, idx, cap=3)
        S["M0_Champ_v13_cap1"] = champion_exposure(trades, idx, cap=1)
        S = {k: v for k, v in S.items() if v is not None and v.notna().sum() > 5000}
        log(f"strategies: {list(S.keys())}")

    fit_mask = yr.isin(FIT_YEARS)
    oos_mask = yr.isin(OOS_YEARS)

    # ---------------- 3) v2.0.1 causality regression (G4.7) ------------------
    with stage("v2.0.1 causality regression"):
        zca = S["Channel_A"].dropna()
        reg = {}
        for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
            sel = zca.index[mask.reindex(zca.index).fillna(False)]
            if len(sel) < 500:
                continue
            sh_same = samebar_net_sharpe(zca[sel], r[sel], sigma[sel])
            sh_fix = frozen_net(zca[sel], r[sel], sigma[sel], 1.0)["metrics"]["sharpe_ann"]
            reg[wname] = {"samebar_sharpe": round(float(sh_same), 3),
                          "causal_sharpe": round(float(sh_fix), 3)}
            ledger.add("causality_regression",
                       {"window": wname, "probe": "Channel_A", "rev": CODE_REV},
                       float(sh_fix), stage="regression", kernel="K6")
        trap_detected = all(v["samebar_sharpe"] > 3.0 for v in reg.values())
        no_residual = all(abs(v["causal_sharpe"]) < 1.5 for v in reg.values())
        ok_caus = gate("G4.7_causality_v201",
                       trap_detected and no_residual and len(reg) == 2,
                       "v2.0.1 frozen_net causality regression on Channel_A: "
                       "same-bar (lookahead) Sharpe "
                       + json.dumps({k: v["samebar_sharpe"] for k, v in reg.items()})
                       + " vs fixed causal Sharpe "
                       + json.dumps({k: v["causal_sharpe"] for k, v in reg.items()})
                       + " -- trap must be detected (>3) and the fixed "
                         "accounting must show no residual same-bar edge")

    # ---------------- 4) FIT gauntlet (engine vs frozen) ---------------------
    with stage("FIT gauntlet"):
        gauntlet = {}
        engine_mode = {}

        def fit_gauntlet(name, z):
            sig = sig_of(name, z)
            cached = ck.load_json("fitgauntlet", name)
            if cached and cached.get("sig") == sig and "engine_shipped" in cached:
                gauntlet[name] = cached
                engine_mode[name] = cached["engine_shipped"]
                log(f"  GAUNTLET {name}: CACHED engine_shipped="
                    f"{cached['engine_shipped']}")
                return
            sel = z.dropna().index
            sel = sel[fit_mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                gauntlet[name] = {"skipped": True}
                engine_mode[name] = False
                return
            if name in FORCED_FROZEN:
                fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
                rec = {"sig": sig, "forced": "frozen_translation",
                       "engine_fit_sharpe": None,
                       "frozen_fit_sharpe": fro["metrics"]["sharpe_ann"],
                       "engine_shipped": False}
            else:
                eng = run_online_engine(z[sel], r[sel], sigma[sel], 1.0)
                fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
                e_sh = eng["metrics"].get("sharpe_ann", -9)
                f_sh = fro["metrics"].get("sharpe_ann", -9)
                rec = {"sig": sig, "engine_fit_sharpe": e_sh,
                       "frozen_fit_sharpe": f_sh,
                       "engine_turnover": eng["turnover_py"],
                       "frozen_turnover": fro["turnover_py"],
                       "engine_shipped": bool(e_sh >= f_sh)}
            gauntlet[name] = rec
            engine_mode[name] = rec["engine_shipped"]
            ledger.add(f"{name}+gauntlet", {"rev": CODE_REV, **{
                k: v for k, v in rec.items() if k != "sig"}},
                       rec.get("frozen_fit_sharpe"), stage="gauntlet", kernel="K6")
            ck.save_json("fitgauntlet", name, rec)
            log(f"  GAUNTLET {name}: engine={rec.get('engine_fit_sharpe')} "
                f"frozen={rec.get('frozen_fit_sharpe')} "
                f"ships_engine={rec['engine_shipped']}")

        for name, z in S.items():
            fit_gauntlet(name, z)

    # ---------------- 5) composites: M6 / M6b / M7 ---------------------------
    with stage("composite construction (M6/M6b/M7)"):
        # M6 (v2.0.2, Carver S24): FIXED a-priori family = the active CPU
        # pair; SR inputs = buffered ENGINE FIT Sharpes (the shipped
        # execution configuration per the gauntlet), shrunk 90% toward the
        # equal-weight prior; dispersion/correlation from the members' FIT
        # frozen net streams; corr shrunk 0.50 toward identity; weights
        # clipped >= 0 with the equal-weight fallback.  Frozen before the
        # OOS seal is broken.
        streams = {}
        member_fit = {}
        member_fit_engine = {}
        for m in STACK_CANDIDATES:
            if m not in S:
                continue
            z = S[m].dropna()
            sel = z.index[fit_mask.reindex(z.index).fillna(False)]
            if len(sel) < 500:
                continue
            fro = frozen_net(z[sel], r[sel], sigma[sel], 1.0)
            member_fit[m] = fro["metrics"]["sharpe_ann"]
            # v2.0.2: the selection-relevant SR input is the BUFFERED
            # ENGINE FIT Sharpe (gauntlet), not the unbuffered frozen_net
            g = gauntlet.get(m) or {}
            e_sh = g.get("engine_fit_sharpe")
            if e_sh is None or not np.isfinite(e_sh):
                e_sh = member_fit[m]
            member_fit_engine[m] = e_sh
            streams[m] = fro["net"]
        members = list(streams.keys())
        assert members, "no stack members present"
        sdf = pd.DataFrame(streams).dropna()
        members = list(sdf.columns)
        sd = sdf.std()
        # Carver shrinkage_SR = 0.90: SR estimates shrunk 90% toward the
        # equal-weight prior (the cross-member mean SR equalizes weights);
        # the 10% residual keeps the honest FIT ranking in the weights.
        sr = pd.Series({m: member_fit_engine[m] for m in members})
        sr_shrunk = ((1.0 - SR_SHRINK_TO_EQUAL) * sr
                     + SR_SHRINK_TO_EQUAL * sr.mean())
        mu_shrunk = sr_shrunk * sd * BARS_PER_YEAR
        C = sdf.corr()
        n = len(C)
        C_shrunk = CORR_SHRINK * C + (1.0 - CORR_SHRINK) * np.eye(n)
        sdv = sd.values
        Cov = C_shrunk.values * np.outer(sdv, sdv) * BARS_PER_YEAR
        try:
            w_raw = np.linalg.solve(Cov + 1e-6 * np.eye(n), mu_shrunk.values)
        except np.linalg.LinAlgError:
            w_raw = np.ones(n) / n
        w = np.clip(w_raw, 0.0, None)
        if w.sum() <= 1e-9:
            # both shrunk SRs non-positive: the evidence cannot rank the
            # a-priori family -- the equal-weight prior ships unchanged
            w = np.ones(n) / n
        w = w / w.sum()
        weights = dict(zip(members, [round(float(x), 4) for x in w]))
        ok_stack = gate("G6.2_stack", len(members) == len(STACK_CANDIDATES)
                        and abs(w.sum() - 1.0) < 1e-6,
                        f"M6 FIXED family {STACK_CANDIDATES} (a priori, no "
                        f"FIT membership filter): engine FIT Sharpes "
                        f"{json.dumps({k: round(v, 4) for k, v in member_fit_engine.items()})} "
                        f"(frozen_net FIT for reference: "
                        f"{json.dumps({k: round(v, 4) for k, v in member_fit.items()})}): "
                        f"weights={weights} (Carver S24: SR-shrink 0.90 "
                        f"toward equal weights, corr-shrink 0.50, "
                        f"shrunk SRs {json.dumps({k: round(float(v), 4) for k, v in sr_shrunk.items()})})")
        ledger.add("M6_Ensemble", {"weights": weights, "members": members,
                                   "shrinkage_SR": SR_SHRINK_TO_EQUAL,
                                   "shrinkage_corr": CORR_SHRINK,
                                   "sr_inputs": "buffered engine FIT Sharpes",
                                   "family_rule": "fixed a-priori active CPU pair",
                                   "rev": CODE_REV},
                   stage="config", kernel="K6")

        z_src = pd.concat([S[m].rename(m) for m in members], axis=1)
        z_src = z_src.ffill().fillna(0.0)
        S["M6_AURUM_v2_Ensemble"] = pd.Series(z_src.values @ w, index=idx)
        # M6b: equal-weight robustness variant over the same members
        w_eq = np.ones(len(members)) / len(members)
        S["M6b_Ensemble_EQW"] = pd.Series(z_src.values @ w_eq, index=idx)

        # M7 Grand Hybrids (v2.0.2 split -- both FIXED a-priori blends,
        # selection-free; the FIT-grid scan below is a diagnostic only):
        #   * ACTIVE arm: 50% v1.3 champion + 50% active de-saturated v2
        #     stack (M6 over {M1_GBDT_v2, M2_RFF_Ridge}) -- measures the
        #     true active orthogonality gain.
        #   * CORE-SATELLITE arm: 50% v1.3 champion + 50% passive gold
        #     sleeve -- the transparent accounting of what the v2.0.1
        #     M7_Grand_Hybrid stream actually was in OOS (its v2 leg was a
        #     saturated constant +1.0, i.e. bit-identical to BuyHold_Gold;
        #     this arm preserves the 2.6351 / 2.5012@2x benchmark).
        z_ch = S["M0_Champ_v13"]
        z_v2s = S["M6_AURUM_v2_Ensemble"]
        S["M7_Active_Grand_Hybrid"] = 0.5 * z_v2s + 0.5 * z_ch
        S["M7_Core_Satellite"] = 0.5 * S["BuyHold_Gold"] + 0.5 * z_ch
        ledger.add("M7_Active_Grand_Hybrid",
                   {"blend": "50/50 fixed", "leg_v2": "M6_AURUM_v2_Ensemble "
                    "(active de-saturated stack)",
                    "leg_v13": "M0_Champ_v13 (cap=3 exposure)",
                    "selection": "none (a priori)", "rev": CODE_REV},
                   stage="config", kernel="K6")
        ledger.add("M7_Core_Satellite",
                   {"blend": "50/50 fixed", "leg_passive": "BuyHold_Gold "
                    "(passive gold sleeve)",
                    "leg_v13": "M0_Champ_v13 (cap=3 exposure)",
                    "selection": "none (a priori); transparent continuation "
                                 "of the v2.0.1 M7_Grand_Hybrid OOS stream",
                    "rev": CODE_REV},
                   stage="config", kernel="K6")
        best_h, best_hs = 0.5, -9e9
        grid_curve = {}
        for wv2 in np.arange(0.0, 1.01, 0.05):
            zh = wv2 * z_v2s + (1 - wv2) * z_ch
            sel = zh.dropna().index
            sel = sel[fit_mask.reindex(sel).fillna(False)]
            if len(sel) < 500:
                continue
            sh = frozen_net(zh[sel], r[sel], sigma[sel], 1.0)["metrics"]["sharpe_ann"]
            grid_curve[round(float(wv2), 2)] = None if sh is None else round(float(sh), 3)
            ledger.add("M7_grid_diagnostic", {"w_v2": round(float(wv2), 2)},
                       sh, stage="fit_eval", kernel="K6")
            if sh is not None and np.isfinite(sh) and sh > best_hs:
                best_hs, best_h = sh, float(wv2)
        log(f"  M7_Active FIT-grid diagnostic: best w_v2={best_h:.2f} "
            f"(FIT Sharpe {best_hs:.3f}); shipped blend = FIXED 0.50")

        for name in ("M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M7_Active_Grand_Hybrid", "M7_Core_Satellite"):
            fit_gauntlet(name, S[name])
        log(f"strategies (final): {list(S.keys())}")

    # ---------------- 2b) z-quality / saturation audit (G4.8) ---------------
    # Engineering Flaw #8 (discovered in the final v2.0.1 K6 audit, FIXED at
    # the producer in v2.0.2): a fold-local z-scale (std of the model's
    # early-stop-slice predictions) can collapse relative to the test-fold
    # prediction magnitude, saturating tanh(pred/scale) to sign(pred): the
    # strategy degrades into a per-fold directional bet with ~zero per-bar
    # information (v2.0.1 M2_RFF_Ridge: OOS nunique=1, z==1.0 for all of
    # 2023-25).  The audit records saturation status for every strategy;
    # v2.0.2 carries the gate to the producer (K2 G4.8) and this downstream
    # audit remains as the independent verification layer.
    with stage("z-quality audit"):
        zq = {}
        for name, z in S.items():
            rec = {}
            for wname, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = z.dropna().index
                sel = sel[mask.reindex(sel).fillna(False)]
                if len(sel) < 500:
                    continue
                v = z[sel].values
                uniq = int(pd.unique(v).size)
                top_share = float((v == pd.Series(v).mode().iloc[0]).mean())
                rec[wname] = {"n": int(len(v)), "nunique": uniq,
                              "std": round(float(np.std(v)), 5),
                              "top_value_share": round(top_share, 4)}
                rec[f"{wname}_saturated"] = bool(
                    np.std(v) < 0.05 or uniq <= 8 or top_share >= 0.90)
            zq[name] = rec
        # position schedules (benchmarks, barrier translations, passive
        # sleeves, sparse channel steps) are static BY DESIGN -- the audit
        # exists to catch MODEL-SCORE degeneracy, so these are exempt but
        # annotated.  M7_Core_Satellite is a fixed 50/50 blend of two
        # position schedules (champion exposure + passive gold sleeve) --
        # it contains NO model scores, so it is by-design static as well.
        BY_DESIGN_STATIC = {"BuyHold_Gold", "M0_Champ_v13", "M0_Champ_v13_cap1",
                            "Channel_A", "Channel_B", "M7_Core_Satellite"}
        # composites: a healthy-looking blend riding on a saturated member
        # leg is still quarantined (the member's degeneracy is inherited).
        # BY_DESIGN_STATIC legs (barrier translations, passive sleeves,
        # sparse channel steps) are position schedules by design -- they are
        # exempt from flagging and annotated as such.
        COMPOSITE_LEGS = {"M6_AURUM_v2_Ensemble": members,
                          "M6b_Ensemble_EQW": members,
                          "M7_Active_Grand_Hybrid": ["M6_AURUM_v2_Ensemble",
                                                     "M0_Champ_v13"],
                          "M7_Core_Satellite": ["M0_Champ_v13",
                                                "BuyHold_Gold"]}
        for comp, legs in COMPOSITE_LEGS.items():
            if comp in zq:
                zq[comp]["legs"] = {}
                for m_ in legs:
                    if m_ in BY_DESIGN_STATIC:
                        zq[comp]["legs"][m_] = "exempt (position schedule by design)"
                    else:
                        zq[comp]["legs"][m_] = bool(
                            zq.get(m_, {}).get("oos_saturated")
                            or zq.get(m_, {}).get("fit_saturated"))
                zq[comp]["has_flagged_leg"] = any(
                    v is True for v in zq[comp]["legs"].values())
        for n in BY_DESIGN_STATIC & set(zq):
            zq[n]["by_design_position_schedule"] = True
        flagged = sorted([n for n, r in zq.items()
                          if n not in BY_DESIGN_STATIC
                          and (r.get("oos_saturated") or r.get("fit_saturated")
                               or r.get("has_flagged_leg"))])
        with open(os.path.join(WORK, "z_quality.json"), "w") as f:
            json.dump({"audit": zq, "flagged_strategies": flagged,
                       "criteria": "std < 0.05 or nunique <= 8 or top-value "
                                   "share >= 0.90, per window",
                       "note": "flagged = saturated/degenerate z (per-fold "
                               "directional bet); results are still exported "
                               "verbatim and quarantined in the report"},
                      f, indent=1, default=str)
        ok_zq = gate("G4.8_z_quality_audit", len(zq) == len(S),
                     "z saturation audit exported for all strategies (v2.0.2: "
                     "independent downstream verification of the K2 producer "
                     "gate); flagged (saturated/degenerate): "
                     f"{flagged or 'none'}; full audit in z_quality.json"
                     + ("" if not flagged else
                        " -- flagged strategies are quarantined in the report"))

    # ---------------- 6) full evaluation grid --------------------------------
    with stage("evaluation grid (windows x costs)"):
        results = {}
        year_rows = []
        net_series_export = {}

        def eval_windows(name, z):
            use_engine = engine_mode.get(name, False)
            sig = sig_of(name, z, extra=f"engine={use_engine}")
            cached = ck.load_json("evalgrid", name)
            nets_c = {}
            if (cached and cached.get("sig") == sig and "entry" in cached
                    and "fit" in cached["entry"] and "oos" in cached["entry"]):
                ok_nets = True
                for window in ("fit", "oos"):
                    if window in cached["entry"]:
                        s = ck.load_net(name, window)
                        if s is None:
                            ok_nets = False
                            break
                        nets_c[window] = s
                if ok_nets:
                    results[name] = cached["entry"]
                    for window, s in nets_c.items():
                        net_series_export[f"{name}__{window}"] = s
                        yr_rows = cached["entry"][window].get("year_rows", [])
                        for row in yr_rows:
                            year_rows.append({"strategy": name, **row})
                    log(f"  EVAL {name}: CACHED ({'engine' if use_engine else 'frozen'})")
                    return
            entry = {"gauntlet_engine": use_engine}
            for window, mask in (("fit", fit_mask), ("oos", oos_mask)):
                sel = z.dropna().index
                sel = sel[mask.reindex(sel).fillna(False)]
                if len(sel) < 500:
                    continue
                zw, rw, sw = z[sel], r[sel], sigma[sel]
                w_entry = {}
                if use_engine:
                    full = run_online_engine(zw, rw, sw, 1.0)
                    w_entry = {"metrics": full["metrics"],
                               "turnover_py": full["turnover_py"],
                               "diag": full["diagnostics"]}
                    net = full["net"]
                    for cm in (1.5, 2.0):
                        v = run_online_engine(zw, rw, sw, cm)
                        w_entry[f"cost{cm}"] = v["metrics"]
                else:
                    fro = frozen_net(zw, rw, sw, 1.0)
                    w_entry = {"metrics": fro["metrics"],
                               "turnover_py": fro["turnover_py"]}
                    net = fro["net"]
                    for cm in (1.5, 2.0):
                        w_entry[f"cost{cm}"] = frozen_net(zw, rw, sw, cm)["metrics"]
                # Carver speed limit: cost drag <= 1/3 pre-cost Sharpe
                pre = frozen_net(zw, rw, sw, 0.0)
                drag = frozen_net(zw, rw, sw, 1.0)
                pre_sh = pre["metrics"].get("sharpe_ann", 0.0)
                cost_sh = pre_sh - drag["metrics"].get("sharpe_ann", 0.0)
                w_entry["speed_limit_ok"] = bool(
                    cost_sh <= pre_sh / 3.0 + 1e-9) if pre_sh > 0 else None
                w_entry["cost_drag_sharpe"] = round(float(cost_sh), 3)
                w_entry["pre_cost_sharpe"] = round(float(pre_sh), 3)
                # year-by-year
                yr_rows = []
                yy = pd.Series(sel.year, index=sel)
                for y_ in sorted(yy.unique()):
                    gsel = sel[yy == y_]
                    sub = z[gsel]
                    if use_engine:
                        mtr = run_online_engine(sub, r[gsel], sigma[gsel], 1.0)["metrics"]
                    else:
                        mtr = frozen_net(sub, r[gsel], sigma[gsel], 1.0)["metrics"]
                    row = {"year": int(y_), "window": "FIT" if y_ < 2023 else "OOS"}
                    row.update({k: v for k, v in mtr.items() if k != "n"})
                    yr_rows.append(row)
                    year_rows.append({"strategy": name, **row})
                w_entry["year_rows"] = yr_rows
                entry[window] = w_entry
                net_series_export[f"{name}__{window}"] = net
                ck.save_net(name, window, net)
            results[name] = entry
            ck.save_json("evalgrid", name, {"sig": sig, "entry": entry})
            log(f"  EVAL {name}: {'engine' if use_engine else 'frozen'} "
                f"OOS={entry.get('oos', {}).get('metrics', {}).get('sharpe_ann')}")

        for name, z in S.items():
            eval_windows(name, z)

        year_df = pd.DataFrame(year_rows)
        year_df.to_csv(os.path.join(WORK, "year_table.csv"), index=False)
        ns = pd.DataFrame(net_series_export)
        ns.to_parquet(os.path.join(WORK, "net_series.parquet"))
        serial = {}
        for name, entry in results.items():
            serial[name] = {}
            for window in ("fit", "oos"):
                if window in entry:
                    serial[name][window] = {
                        k: v for k, v in entry[window].items() if k != "year_rows"}
        with open(os.path.join(WORK, "backtest_v2.json"), "w") as f:
            json.dump({"strategies": serial, "v13_reference": V13_REFERENCE,
                       "code_rev": CODE_REV}, f, indent=1, default=str)
        with open(os.path.join(WORK, "gauntlet.json"), "w") as f:
            json.dump(gauntlet, f, indent=1, default=str)

    # ---------------- 7) statistics: bootstrap / DSR / PBO -------------------
    with stage("statistics"):
        stats = {}
        for name, entry in results.items():
            if "oos" in entry and f"{name}__oos" in net_series_export:
                net = net_series_export[f"{name}__oos"].values
                stats.setdefault(name, {})["oos_bootstrap"] = block_bootstrap_ci(net, 5000)
        N_trials = ledger.unique_configs()
        # ------------------------------------------------------------------
        # DSR trial family (v2.0.2, unit-consistent, effective-independent):
        #   Bailey & Lopez de Prado 2014: N counts INDEPENDENT trials.
        #   Correlated siblings are not independent trials, so the K6
        #   selection panel is reduced to effective strategy families:
        #     step 1 -- exact stream duplicates on the evaluation domain
        #               collapse (M6b is bit-identical to M6 when the stack
        #               weight hits the equal-weight fallback);
        #     step 2 -- sibling groups count once via their best-FIT member
        #               (M3c/M3m DAE sizes, M4/M4b DMN variants, the v1.3
        #               champion cap=3/cap=1 translations).
        #   This yields the distinct configurations the championship was
        #   actually selected among.  Sharper variants are reported
        #   in stats._meta.dsr_sensitivity_m7 for full transparency.
        #   EXCLUDED from the primary family (also sensitivity-reported):
        #     - M7 blend grid: diagnostic only -- the 50/50 blends were fixed
        #       a priori and shipped AGAINST their FIT optimum, so they are
        #       not selection trials.
        #     - K2/K3 ledger rows: pre-fix same-bar accounting vintage /
        #       per-fold search metrics -- not unit-comparable.
        # All trial Sharpes are annualized -> per-bar (Bailey-LdP requires
        # SR and V[SR] in the same frequency).
        # ------------------------------------------------------------------
        SIBLING_GROUPS = {"M3c_DAE": "M3_DAE", "M3m_DAE": "M3_DAE",
                          "M4_DMN_Lite": "M4_DMN", "M4b_DMN_Lite": "M4_DMN",
                          "M0_Champ_v13": "M0_Champ",
                          "M0_Champ_v13_cap1": "M0_Champ"}
        # v2.0.2 blend-slot convention (consistent with the v2.0.1
        # pre-registered family): the fixed a-priori portfolio constructions
        # over counted signal families (M6/M6b over {M1,M2}; M7_Active over
        # {M0,M6}; M7_Core over {M0,BuyHold}) are deterministic re-weightings
        # of already-counted members -- correlated constructions, not
        # independent attempts -- so they contribute ONE blend-slot trial at
        # the best-FIT variant.  The conservative accounting (every distinct
        # blend stream counts) is exported alongside in _meta.
        BLEND_SLOT = ["M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                      "M7_Active_Grand_Hybrid", "M7_Core_Satellite"]
        evaldom = S["M1_GBDT_v2"].index >= pd.Timestamp("2016-01-01", tz="UTC")
        seen_sigs, uniq = set(), {}
        for name in S:                       # panel order: members first
            sg = z_sig(S[name][evaldom])
            if sg in seen_sigs:
                continue                     # exact stream duplicate
            seen_sigs.add(sg)
            uniq[name] = (gauntlet.get(name) or {}).get("frozen_fit_sharpe")
        fam_best = {}
        for name, sh in uniq.items():
            if not isinstance(sh, (int, float)) or not np.isfinite(sh):
                continue
            fk = SIBLING_GROUPS.get(name, name)
            if fk not in fam_best or sh > fam_best[fk]:
                fam_best[fk] = float(sh)
        # conservative accounting: every distinct (deduped) stream counts
        fam_conservative = np.array(sorted(fam_best.values()))
        # primary accounting: signal families + ONE blend slot (best-FIT
        # variant of the pre-registered blend constructions)
        blend_sharpes = [fam_best[b] for b in BLEND_SLOT if b in fam_best]
        fam_signals = {k: v for k, v in fam_best.items() if k not in BLEND_SLOT}
        fam_primary_vals = sorted(fam_signals.values())
        if blend_sharpes:
            fam_primary_vals.append(max(blend_sharpes))
        fam_primary = np.array(fam_primary_vals)
        trial_sharpes = (fam_primary / math.sqrt(BARS_PER_YEAR)
                         if len(fam_primary) > 5 else None)
        trial_sharpes_conservative = (fam_conservative / math.sqrt(BARS_PER_YEAR)
                                      if len(fam_conservative) > 5 else None)
        # sensitivity families (flagship only)
        def _dsr_m7(vals_ann):
            if len(vals_ann) < 6:
                return None
            net7 = net_series_export["M7_Core_Satellite__oos"].values
            zz = (net7 - net7.mean()) / (net7.std() + 1e-12)
            return round(deflated_sharpe_ratio(
                float(net7.mean() / (net7.std(ddof=1) + 1e-12)), len(net7),
                float(np.mean(zz ** 3)), float(np.mean(zz ** 4)),
                sr_trials=vals_ann / math.sqrt(BARS_PER_YEAR),
                N_trials=len(vals_ann)), 4)
        def _fam(rows_pred):
            seen, vals = set(), []
            for r in ledger.rows:
                fm = r.get("fit_metric")
                if (not isinstance(fm, (int, float)) or not np.isfinite(fm)
                        or not rows_pred(r)):
                    continue
                key = (r["model"], r["config_hash"])
                if key in seen:
                    continue
                seen.add(key)
                vals.append(float(fm))
            return np.array(vals)
        sens_grid = _fam(lambda r: r.get("kernel") == "K6" and
                         r.get("stage") in ("gauntlet", "fit_eval"))
        sens_k2 = _fam(lambda r: (r.get("kernel") == "K6" and
                                  r.get("stage") in ("gauntlet", "fit_eval"))
                       or (r.get("kernel") == "K2" and
                           r.get("stage") in ("fit", "engine")))
        sens_all = _fam(lambda r: True)
        for name, entry in results.items():
            if "oos" not in entry:
                continue
            net = net_series_export[f"{name}__oos"].values
            z = (net - net.mean()) / (net.std() + 1e-12)
            skew = float(np.mean(z ** 3))
            kurt = float(np.mean(z ** 4))
            sr_per_bar = float(net.mean() / (net.std(ddof=1) + 1e-12))
            dsr = deflated_sharpe_ratio(sr_per_bar, len(net), skew, kurt,
                                        sr_trials=trial_sharpes if len(trial_sharpes) > 5 else None,
                                        N_trials=max(2, len(fam_primary)))
            stats.setdefault(name, {})["oos_dsr"] = round(dsr, 4)
            stats[name]["dsr_passes_95"] = bool(dsr >= 0.95)
            # conservative full-panel accounting (every distinct blend counts)
            dsr_c = deflated_sharpe_ratio(
                sr_per_bar, len(net), skew, kurt,
                sr_trials=trial_sharpes_conservative
                if len(trial_sharpes_conservative) > 5 else None,
                N_trials=max(2, len(fam_conservative)))
            stats[name]["oos_dsr_conservative"] = round(dsr_c, 4)
        # PBO CSCV on the FIT-window net streams of the selection family
        # (FIT-only matrix; mixing windows would empty the dropna intersection)
        fam = {f"{name}__fit": net_series_export[f"{name}__fit"]
               for name in results if f"{name}__fit" in net_series_export}
        pnl = pd.DataFrame(fam).dropna()
        pbo = pbo_cscv(pnl.values, n_blocks=16) if pnl.shape[1] >= 4 else None
        stats["_meta"] = {"N_trials_ledger": N_trials,
                          "ledger_rows": len(ledger.rows),
                          "pbo_cscv": round(pbo, 4) if pbo is not None else None,
                          "code_rev": CODE_REV,
                          "dsr_family": "K6 v2.0.2 tournament selection panel, "
                                        "effective-independent families "
                                        "(Bailey-LdP: correlated siblings "
                                        "are not independent trials; frozen "
                                        "FIT Sharpes under causal accounting). "
                                        "Primary = signal families + ONE "
                                        "blend slot (fixed a-priori portfolio "
                                        "constructions over counted members "
                                        "are correlated re-weightings, "
                                        "counted once at their best-FIT "
                                        "variant -- the v2.0.1 pre-registered "
                                        "convention, applied consistently)",
                          "dsr_family_N": int(len(fam_primary)),
                          "dsr_family_sharpes": [round(float(x), 4) for x in fam_primary],
                          "dsr_family_conservative_N": int(len(fam_conservative)),
                          "dsr_family_conservative_sharpes": [round(float(x), 4)
                                                              for x in fam_conservative],
                          "dsr_conservative_note": "conservative accounting: "
                          "every distinct (deduped) blend stream counts as a "
                          "trial; flagship DSR under this vintage is exported "
                          "per strategy as oos_dsr_conservative",
                          "trial_sharpe_units": "per-bar (annualized / sqrt(5786))",
                          "dsr_sensitivity_m7": {
                              "primary": _dsr_m7(fam_primary),
                              "conservative": _dsr_m7(fam_conservative),
                              "with_m7_grid": _dsr_m7(sens_grid),
                              "with_k2_prefix": _dsr_m7(sens_k2),
                              "all_ledger_mixed_units": _dsr_m7(sens_all),
                              "family_N": {"primary": int(len(fam_primary)),
                                           "conservative": int(len(fam_conservative)),
                                           "with_grid": int(len(sens_grid)),
                                           "with_k2": int(len(sens_k2)),
                                           "all": int(len(sens_all))}}}
        dsr_flagship = stats.get("M7_Core_Satellite", {}).get("oos_dsr")
        dsr_active = stats.get("M7_Active_Grand_Hybrid", {}).get("oos_dsr")
        dsr_flagship_c = stats.get("M7_Core_Satellite", {}).get(
            "oos_dsr_conservative")
        pbo_val = stats["_meta"]["pbo_cscv"]
        if dsr_flagship is not None and dsr_flagship < 0.95:
            verdict = ("PRE-REGISTERED BAR MISSED (v2.0.2 finding, reported "
                       "verbatim): the Flaw #8 fix replaces the saturated-M2 "
                       "artifact trial (+0.3504 FIT) with the true de-"
                       "saturated trial (-0.8394), widening the honest trial "
                       "dispersion and raising the Bailey-LdP null threshold "
                       "-- the flagship's 95% DSR certification (v2.0.1: "
                       "0.958 on the artifact panel) is withdrawn to "
                       "borderline")
        else:
            verdict = "flagship clears the pre-registered 0.95 DSR bar"
        ok_stats = gate("G6.1_stats",
                        bool(dsr_flagship is not None and dsr_flagship >= 0.95
                             and (pbo_val is None or pbo_val <= 0.20)),
                        f"DSR over N={N_trials} ledger configs: shipping "
                        f"flagship M7_Core_Satellite DSR={dsr_flagship} "
                        f"(gate >= 0.95, primary family N={len(fam_primary)}: "
                        f"signal families + one blend slot); conservative "
                        f"full-panel vintage (every distinct blend counts, "
                        f"N={len(fam_conservative)}): flagship DSR="
                        f"{dsr_flagship_c}; active measurement arm "
                        f"M7_Active_Grand_Hybrid DSR={dsr_active}; "
                        f"CSCV-PBO={pbo_val} (gate <= 0.20); {verdict}; all: "
                        + json.dumps({k: v.get("oos_dsr") for k, v in stats.items()
                                      if isinstance(v, dict) and "oos_dsr" in v}))
        with open(os.path.join(WORK, "stats.json"), "w") as f:
            json.dump(stats, f, indent=1, default=str)

    # ---------------- 8) champion comparison + export ------------------------
    with stage("champion comparison + export"):
        comp = {"v13_flagship_cap3": V13_REFERENCE["flagship_cap3"],
                "v13_flagship_cap1": V13_REFERENCE["flagship_cap1"]}
        for name in ("M7_Active_Grand_Hybrid", "M7_Core_Satellite",
                     "M6_AURUM_v2_Ensemble", "M6b_Ensemble_EQW",
                     "M0_Champ_v13", "M0_Champ_v13_cap1", "M5_ContinuumNet",
                     "M2_RFF_Ridge", "M1_GBDT_v2", "BuyHold_Gold"):
            if name in results and "oos" in results[name]:
                m = results[name]["oos"]["metrics"]
                comp[name] = {"sharpe_ann": m.get("sharpe_ann"),
                              "maxdd_volunits": m.get("maxdd_volunits"),
                              "oos_bootstrap": stats.get(name, {}).get("oos_bootstrap"),
                              "oos_dsr": stats.get(name, {}).get("oos_dsr"),
                              "cost2x_sharpe": results[name]["oos"].get("cost2.0", {}).get("sharpe_ann")}
        with open(os.path.join(WORK, "champion_compare.json"), "w") as f:
            json.dump(comp, f, indent=1, default=str)
        # Carver speed-limit audit: gated on the shipping family; every
        # strategy's flag is exported (reference_only for non-shipping)
        sl = {}
        sl_ref_only = {}
        for name, entry in results.items():
            for window in ("fit", "oos"):
                if window in entry and entry[window].get("speed_limit_ok") is not None:
                    key = f"{name}_{window}"
                    if name in SHIPPING_FAMILY:
                        sl[key] = entry[window]["speed_limit_ok"]
                    else:
                        sl_ref_only[key] = entry[window]["speed_limit_ok"]
        ok_sl = gate("G5.4_speed_limit", all(sl.values()) if sl else True,
                     f"Carver speed limit (annual cost <= 1/3 pre-cost SR), "
                     f"shipping family: {sl}; reference-only (also-rans/"
                     f"baselines, not gated): {sl_ref_only}")
        ok_g = gate("G5.3_gauntlet", len(gauntlet) == len(S),
                    "engine vs frozen per strategy (engine ships iff it beats "
                    "frozen on FIT; v1.3 translations forced frozen): "
                    + json.dumps({k: v.get("engine_shipped") for k, v in gauntlet.items()}))
        ok_ex = gate("G6.6_export", True,
                     "backtest_v2.json + year_table.csv + net_series.parquet + "
                     "stats.json + champion_compare.json + trial_ledger.jsonl + "
                     "gauntlet.json + k6_meta.json + ckpt/")
        with open(os.path.join(WORK, "k6_meta.json"), "w") as f:
            json.dump({"code_rev": CODE_REV,
                       "wall_seconds": round(time.time() - t_wall, 1),
                       "m6_weights": weights, "m6_members": members,
                       "m6_member_fit_sharpes": {
                           "engine": {k: v for k, v in member_fit_engine.items()},
                           "frozen": {k: v for k, v in member_fit.items()}},
                       "m7_active_blend": "0.5 * M6 + 0.5 * M0_Champ_v13 (fixed)",
                       "m7_core_satellite_blend": "0.5 * BuyHold_Gold + 0.5 * "
                                                  "M0_Champ_v13 (fixed; "
                                                  "transparent continuation "
                                                  "of the v2.0.1 M7 OOS stream)",
                       "m7_grid_diagnostic": {"best_w_v2": best_h,
                                              "best_fit_sharpe": best_hs,
                                              "curve": grid_curve},
                       "strategies": list(S.keys()),
                       "n_strategies": len(S),
                       "data_signatures": {k: z_sig(v) for k, v in S.items()},
                       "z_quality_flagged": flagged,
                       "z_quality_summary": {n: {w: {"nunique": zq[n][w]["nunique"],
                                                      "std": zq[n][w]["std"]}
                                                 for w in ("fit", "oos") if w in zq[n]}
                                              for n in zq},
                       "forced_frozen": sorted(FORCED_FROZEN & set(S)),
                       "cost_r": COST_R, "bars_per_year": BARS_PER_YEAR,
                       "fit_years": FIT_YEARS, "oos_years": OOS_YEARS},
                      f, indent=1, default=str)

    all_ok = all([ok_caus, ok_zq, ok_stack, ok_g, ok_sl, ok_stats, ok_ex])
    gate("G6_ALL", all_ok, "all K6 gates green" if all_ok else "K6 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log(f"K6 COMPLETE in {time.time()-t_wall:.0f}s" if all_ok else "K6 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
