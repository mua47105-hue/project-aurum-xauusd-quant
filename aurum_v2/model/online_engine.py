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
