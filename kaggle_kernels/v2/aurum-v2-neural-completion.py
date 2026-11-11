# ============================================================================
# Auto-generated AURUM v2.0 kernel: AURUM v2 Neural Completion
# (rebuilt 2026-11-09 17:27)
# Spec v2.0 frozen. Do not edit on Kaggle - edit sources and rebuild.
# ============================================================================
from __future__ import annotations

import os, sys as _sys



# ==================== losses.py ====================
"""
AURUM v2.0 -- Fused Loss Functions (PyTorch) -- losses.py
==========================================================
Synthesis mandated by Spec v2.0 execution mandate II.3: the best of BOTH
blueprints in one module.

From Blueprint 1 (Transcript Intelligence Report 5.3):
  * DeePMRobustSharpeLoss structure: pooled annualized net Sharpe + SoftMin
    worst-window Sharpe penalty (EVaR proxy), with the DeePM training
    transaction-cost scaler gamma = 0.5 and burn-in slicing.
  * two_pass_exact_sharpe_step: Two-Pass Exact Gradient Accumulation
    (S05 arXiv:2601.05975) for memory-constrained T4 GPUs.

From Blueprint 2 (Industrial Quant Blueprint 4.4):
  * NetSharpeLoss structure: per-batch-row annualized Sharpe with costs
    inside the graph, turnover penalty, Ulcer Index drawdown penalty.
  * net_strategy_return: vol-scaled position, cost on |z v - z_prev v_prev|.
  * DifferentialSharpe (Moody-Saffell via the Borrageiro reproduction,
    denominator typo corrected to (B - A^2)^{3/2}) as the online monitor.
  * aux_losses: multi-task supervision on y_ret24_norm / y_dir24 / meta.

FUSION (the shipped trainer objective):
  RobustNetSharpeLoss = Pooled Annualized Net Sharpe (sqrt(A), A = 5786
  empirical bars/year) + lambda_soft * SoftMin worst-window penalty
  (tau = 0.2, EVaR proxy) + lambda_dd * Ulcer Index + lambda_turn * turnover,
  with gamma = 0.5 cost scaler inside the graph; usable both single-batch and
  inside two_pass_exact_sharpe_step.

All formulas verified against primary sources (see both blueprints' ledgers):
  Sharpe loss (Lim-Zohren-Roberts 2019, arXiv:1904.04912, Eq. 16 + repo eps)
  Cost-augmented return (DMN Eq. 27; Zhang-Zohren-Roberts arXiv:2005.13665
  Eq. 7; Moody-Saffell 2001 -- three independent sources, same structure)
  SoftMin (Wood-Roberts-Zohren 2026 DeePM, arXiv:2601.05975)
  Ulcer Index (Martin 1987/89)

Annualization: A = bars/year.  LOCKED to the empirical count 5786
(87,948 H1 bars / 15.2 years; sqrt(A) = 76.07 ~ sqrt(252*23) = 76.13).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_SHARPE = 1e-9
BARS_PER_YEAR_DEFAULT = 5786        # LOCKED empirical count (see docstring)


def net_strategy_return(z: torch.Tensor,
                        r: torch.Tensor,
                        sigma: torch.Tensor,
                        cost_r: float,
                        sigma_tgt: float = 0.15 / math.sqrt(5786)) -> torch.Tensor:
    """Cost-aware vol-scaled strategy return per bar (all tensors (B, T)).

    z     : model position in [-1, 1] at each bar (vol-scaled exposure target)
    r     : raw next-bar log returns of XAUUSD
    sigma : per-bar EW vol estimate (same units as r) -- CAUSAL feature, input
    cost_r: round-turn cost in return units
    """
    v = sigma_tgt / (sigma + 1e-12)              # vol scaling of position
    s = v * r                                    # vol-scaled return
    z_prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
    v_prev = torch.cat([torch.ones_like(v[:, :1]), v[:, :-1]], dim=1)
    turn = (z * v - z_prev * v_prev).abs()
    return z_prev * s - cost_r * turn


def pooled_net_sharpe(r_net_flat: torch.Tensor, ann_factor: float,
                      eps: float = EPS_SHARPE) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pooled annualized net Sharpe over the flattened logical batch.
    Returns (loss_pool, mean_r, var_r) -- moments returned for reuse."""
    mean_r = torch.mean(r_net_flat)
    var_r = torch.mean(r_net_flat ** 2) - mean_r ** 2 + eps
    sr_pool = (mean_r / torch.sqrt(var_r)) * (ann_factor ** 0.5)
    return -sr_pool, mean_r, var_r


def softmin_window_penalty(r_net: torch.Tensor, ann_factor: float,
                           tau: float = 0.2, window_len: int = 120,
                           eps: float = EPS_SHARPE) -> torch.Tensor:
    """SoftMin worst-window Sharpe penalty (EVaR proxy), DeePM form:
    L_soft = tau * log(mean(exp(-SR_b/tau))) computed stably via max-shift.
    Window Sharpe over non-overlapping window_len chunks per batch row."""
    B, T_active = r_net.shape
    if T_active >= window_len and window_len > 1:
        n_wins = T_active // window_len
        r_wins = r_net[:, : n_wins * window_len].reshape(-1, window_len)
    else:
        r_wins = r_net
    w_mean = torch.mean(r_wins, dim=1)
    w_var = torch.mean(r_wins ** 2, dim=1) - w_mean ** 2 + eps
    sr_b = (w_mean / torch.sqrt(w_var)) * (ann_factor ** 0.5)
    z_b = -sr_b / tau
    z_max = torch.max(z_b).detach()
    return tau * (z_max + torch.log(torch.mean(torch.exp(z_b - z_max)) + eps))


def ulcer_index(r_net: torch.Tensor, eps: float = EPS_SHARPE) -> torch.Tensor:
    """Ulcer Index of the cumulative net path per batch row: UI = sqrt(mean
    D_t^2), D_t = drawdown from the running peak (differentiable a.e.)."""
    eq = torch.cumsum(r_net, dim=1)
    peak = torch.cummax(eq, dim=1).values
    dd = (eq - peak).clamp(max=0.0)
    return torch.sqrt((dd ** 2).mean(dim=1) + eps)


class NetSharpeLoss(nn.Module):
    """Blueprint 2 loss: per-batch-row annualized net Sharpe + turnover +
    Ulcer-drawdown penalty (kept for the DMN-LSTM reference trainer)."""

    def __init__(self, bars_per_year: int = BARS_PER_YEAR_DEFAULT,
                 lam_turn: float = 0.0, lam_dd: float = 0.0,
                 cost_r: float = 1e-4, sigma_tgt_ann: float = 0.15):
        super().__init__()
        self.sqrtA = math.sqrt(bars_per_year)
        self.lam_turn = lam_turn
        self.lam_dd = lam_dd
        self.cost_r = cost_r
        self.sigma_tgt = sigma_tgt_ann / math.sqrt(bars_per_year)

    def forward(self, z: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        R = net_strategy_return(z, r, sigma, self.cost_r, self.sigma_tgt)
        mu = R.mean(dim=1)
        var = (R ** 2).mean(dim=1) - mu ** 2
        sharpe = mu * self.sqrtA / torch.sqrt(var + EPS_SHARPE)
        loss = -sharpe.mean()
        if self.lam_turn > 0:
            dz = (z - torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)).abs()
            loss = loss + self.lam_turn * dz.mean()
        if self.lam_dd > 0:
            loss = loss + self.lam_dd * ulcer_index(R).mean()
        return loss


class RobustNetSharpeLoss(nn.Module):
    """THE FUSED v2.0 OBJECTIVE (mandate II.3).

    L = -PooledAnnualizedNetSharpe(R_net)
        + lambda_soft * SoftMin worst-window penalty (tau, EVaR proxy)
        + lambda_dd   * Ulcer Index
        + lambda_turn * mean|dz|

    R_net carries transaction costs INSIDE the computation graph with the
    DeePM training cost scaler gamma (0.5): training pays gamma x full cost,
    which empirically outperforms both 0 (over-trading) and 1 (gradient
    freezing).  Supports single-batch forward AND the analytical-gradient
    interface required by two_pass_exact_sharpe_step (forward_global).
    """

    def __init__(self,
                 bars_per_year: int = BARS_PER_YEAR_DEFAULT,
                 cost_r: float = 1e-4,
                 gamma_cost_scaler: float = 0.5,
                 sigma_tgt_ann: float = 0.15,
                 softmin_tau: float = 0.2,
                 softmin_lambda: float = 0.25,
                 window_len: int = 120,
                 burn_in: int = 21,
                 lam_dd: float = 0.10,
                 lam_turn: float = 0.0,
                 eps: float = EPS_SHARPE):
        super().__init__()
        self.ann_factor = float(bars_per_year)
        self.cost_r = cost_r
        self.gamma = gamma_cost_scaler
        self.sigma_tgt = sigma_tgt_ann / math.sqrt(bars_per_year)
        self.tau = softmin_tau
        self.lam = softmin_lambda
        self.window_len = window_len
        self.burn_in = burn_in
        self.lam_dd = lam_dd
        self.lam_turn = lam_turn
        self.eps = eps

    # ---- net returns (B2 alignment convention: position z[t-1] earns r[t])
    # with the DeePM gamma-scaled cost and burn-in slicing (B1).
    def compute_net_returns(self, z: torch.Tensor, r: torch.Tensor,
                            sigma: torch.Tensor) -> torch.Tensor:
        """z: (B, T) positions in [-1,1]; r: (B, T) returns realized at bar t
        (earned by the position decided at t-1); sigma: (B, T) causal EW vol.
        Returns (B, T - burn_in) net returns."""
        v = self.sigma_tgt / (sigma + self.eps)
        s = v * r                                   # vol-scaled return at t
        z_prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        v_prev = torch.cat([torch.ones_like(v[:, :1]), v[:, :-1]], dim=1)
        turnover = torch.abs(z * v - z_prev * v_prev)
        r_net = z_prev * s - self.gamma * self.cost_r * turnover
        return r_net[:, self.burn_in:]

    def local_terms(self, r_net: torch.Tensor) -> torch.Tensor:
        """Decomposable (per-micro-batch) penalty terms: Ulcer (+ turnover is
        inside r_net already).  Used by the two-pass trainer."""
        loss = torch.zeros((), device=r_net.device)
        if self.lam_dd > 0:
            loss = loss + self.lam_dd * ulcer_index(r_net).mean()
        return loss

    def forward_global_terms(self, full_r: torch.Tensor) -> torch.Tensor:
        """Global (non-decomposable) terms over the concatenated logical
        batch: pooled Sharpe + SoftMin.  full_r: (N_rows, T_active) net
        returns of ALL micro-batches stacked along dim 0."""
        loss_pool, _, _ = pooled_net_sharpe(full_r.reshape(-1), self.ann_factor, self.eps)
        loss_soft = softmin_window_penalty(full_r, self.ann_factor, self.tau,
                                           self.window_len, self.eps)
        return loss_pool + self.lam * loss_soft

    def forward(self, z: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r_net = self.compute_net_returns(z, r, sigma)
        return self.forward_global_terms(r_net) + self.local_terms(r_net)


class DifferentialSharpe(nn.Module):
    """Moody-Saffell differential Sharpe ratio as an ONLINE objective or
    diagnostic (batch form; denominator typo corrected to (B-A^2)^{3/2})."""

    def __init__(self, eta: float = 0.01):
        super().__init__()
        self.eta = eta

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        T = R.shape[1]
        A = torch.zeros_like(R[:, 0])
        B = torch.zeros_like(R[:, 0])
        D = []
        for t in range(T):
            Rt = R[:, t]
            dA = Rt - A
            dB = Rt ** 2 - B
            denom = (B - A ** 2).clamp(min=EPS_SHARPE) ** 1.5
            D.append((B * dA - 0.5 * A * dB) / denom)
            A = A + self.eta * dA
            B = B + self.eta * dB
        return -torch.stack(D, dim=1).mean()


def aux_losses(z_dir_logit: torch.Tensor, ret24_hat: torch.Tensor,
               meta_logit: torch.Tensor, y_dir: torch.Tensor, y_ret: torch.Tensor,
               w_ret: float = 0.1, w_dir: float = 0.1, w_meta: float = 0.05):
    """Auxiliary multi-task heads (JS-2021 multi-target pattern): direction
    BCE, horizon-return MSE, meta-probability BCE."""
    l_dir = F.binary_cross_entropy_with_logits(z_dir_logit, (y_dir > 0).float())
    l_ret = F.mse_loss(ret24_hat, y_ret)
    l_meta = F.binary_cross_entropy_with_logits(
        meta_logit, (torch.sign(ret24_hat) == y_dir.sign()).float())
    return w_ret * l_ret + w_dir * l_dir + w_meta * l_meta


def two_pass_exact_sharpe_step(model: nn.Module,
                               loss_fn: RobustNetSharpeLoss,
                               optimizer: torch.optim.Optimizer,
                               micro_batches: list,
                               aux_weight: float = 0.15,
                               clipnorm: float = 3.0,
                               scaler=None) -> float:
    """DeePM Two-Pass Exact Gradient Accumulation across micro-batches
    (Blueprint 1 5.3), FUSED with the Blueprint 2 penalties:

    Pass 1 (no_grad): run every micro-batch, collect net returns + RNG states.
    Global: stack the logical batch, compute pooled Sharpe + SoftMin (+ the
    decomposable Ulcer/turnover terms are handled per micro-batch), backprop
    on the stacked tensor to obtain the EXACT analytical upstream gradients
    dL/dR_m per micro-batch.
    Pass 2 (with_grad, RNG replayed): re-run each micro-batch, inject the
    analytical gradient via r_net_m.backward(g_m), add the local decomposable
    penalties (Ulcer) and auxiliary multi-task losses, clip, step.

    micro_batches: list of (x, r, sigma, y_ret, y_dir, fwd) tuples where fwd
    is a callable model->dict with 'z' and aux head outputs.
    Returns the total logical-batch loss value (float).
    """
    optimizer.zero_grad(set_to_none=True)
    rng_states = []
    detached_r_list = []

    # PASS 1: statistical accumulation (no autograd)
    for mb in micro_batches:
        x_m, r_m, vol_m = mb[0], mb[1], mb[2]
        rng_states.append(torch.get_rng_state())
        with torch.no_grad():
            out_m = model(x_m)
            r_net_m = loss_fn.compute_net_returns(out_m["z"], r_m, vol_m)
            detached_r_list.append(r_net_m.detach())

    # exact analytical upstream gradients over the full logical batch
    full_r = torch.cat(detached_r_list, dim=0).requires_grad_(True)
    loss_global = loss_fn.forward_global_terms(full_r)
    total_loss = loss_global
    loss_global.backward()
    upstream_grads = torch.split(full_r.grad.detach(),
                                 [rr.shape[0] for rr in detached_r_list], dim=0)

    # PASS 2: analytical gradient injection (with autograd & replayed RNG).
    # All terms are combined into ONE backward (the injected gradient
    # (r_net * g).sum() is identical to r_net.backward(g), and combining
    # avoids double-backward through the shared graph).
    # Decomposable terms (Ulcer / aux) are averaged across micro-batches so
    # the logical-batch gradient is EXACTLY the single-batch gradient.
    n_micro = len(micro_batches)
    local_val = 0.0
    for (x_m, r_m, vol_m, y_r, y_d, fwd), rng_st, g_m in zip(
            micro_batches, rng_states, upstream_grads):
        torch.set_rng_state(rng_st)
        out_m = fwd(x_m) if fwd is not None else model(x_m)
        r_net_m = loss_fn.compute_net_returns(out_m["z"], r_m, vol_m)
        combined = (r_net_m * g_m).sum()
        # decomposable local penalties (Ulcer) -- averaged over micro-batches
        local = loss_fn.local_terms(r_net_m)
        if torch.is_tensor(local) and local.requires_grad:
            combined = combined + local / n_micro
            local_val += float(local.detach()) / n_micro
        if aux_weight > 0 and y_r is not None and "dir_logit" in out_m:
            aux = aux_losses(out_m["dir_logit"], out_m["ret24"],
                             out_m["meta_logit"], y_d, y_r)
            combined = combined + (aux_weight / n_micro) * aux
        combined.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clipnorm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    return float(total_loss.detach()) + local_val


def self_test() -> None:
    torch.manual_seed(0)
    B, T = 8, 500
    z = torch.tanh(torch.randn(B, T) * 0.5)
    r = torch.randn(B, T) * 0.001
    sigma = torch.full((B, T), 0.001)
    crit = NetSharpeLoss(lam_turn=0.1, lam_dd=0.05, cost_r=5e-5)
    loss = crit(z, r, sigma)
    assert torch.isfinite(loss), "NetSharpeLoss not finite"
    z0 = torch.zeros_like(z)
    l0 = crit(z0, r, sigma)
    assert l0.abs() < 1e-3, f"zero-position loss should vanish, got {l0}"
    zf = torch.zeros_like(z)
    zf[:, :-1] = torch.sign(r[:, 1:])
    lf = crit(zf, r, sigma)
    assert lf < -5.0, f"foresight Sharpe loss should be very negative, got {lf}"
    assert loss > 1.0, f"random-churn loss should be positive (costs), got {loss}"
    zc = torch.zeros_like(z, requires_grad=True)
    crit(0.1 * torch.tanh(zc), r, sigma).backward()
    assert zc.grad is not None and torch.isfinite(zc.grad).all()

    # ---- fused loss: same invariants ----
    fused = RobustNetSharpeLoss(cost_r=5e-5, lam_dd=0.1)
    lf0 = fused(torch.zeros_like(z), r, sigma)
    assert lf0.abs() < 1e-3, f"fused zero-position loss should vanish, got {lf0}"
    lff = fused(zf, r, sigma)
    assert lff < -5.0, f"fused foresight loss should be very negative, got {lff}"
    lchurn = fused(0.5 * torch.sign(torch.randn_like(z)), r, sigma)
    assert lchurn > 0.5, f"fused churn loss should be positive, got {lchurn}"

    # ---- TWO-PASS GRADIENT EQUIVALENCE (mandate: < 1e-5) ----
    # deterministic tiny model so RNG replay is exact
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(7, 1)
            torch.nn.init.zeros_(self.a.bias)

        def forward(self, x):
            return {"z": torch.tanh(self.a(x)).squeeze(-1)}

    torch.manual_seed(1)
    m1_, m2_ = Tiny(), Tiny()
    m2_.load_state_dict(m1_.state_dict())
    fused_tp = RobustNetSharpeLoss(cost_r=5e-5, lam_dd=0.05, softmin_lambda=0.25)
    Bm, Tm = 4, 200
    xs = [torch.randn(Bm, Tm, 7) for _ in range(3)]
    rs = [torch.randn(Bm, Tm) * 0.001 for _ in range(3)]
    ss = [torch.full((Bm, Tm), 0.001) for _ in range(3)]

    # single logical batch (concat along batch dim)
    optA = torch.optim.SGD(m1_.parameters(), lr=0.0)   # lr=0: grads only
    zA = m1_(torch.cat(xs))["z"]          # model already outputs tanh position
    lossA = fused_tp(zA, torch.cat(rs), torch.cat(ss))
    lossA.backward()
    gA = m1_.a.weight.grad.clone()

    # two-pass over 3 micro-batches (clipnorm disabled: the equivalence
    # property is about the INJECTED gradient; the trainer clips afterwards)
    optB = torch.optim.SGD(m2_.parameters(), lr=0.0)
    mbs = [(x, r_, s_, None, None, None) for x, r_, s_ in zip(xs, rs, ss)]
    _ = two_pass_exact_sharpe_step(m2_, fused_tp, optB, mbs, aux_weight=0.0,
                                   clipnorm=1e12)
    gB = m2_.a.weight.grad.clone()
    err = float((gA - gB).norm() / (gA.norm() + 1e-12))
    assert err < 1e-5, f"two-pass gradient equivalence failed: rel err {err:.2e}"
    assert abs(float(lossA) - _) / max(abs(float(lossA)), 1e-9) < 1e-4, \
        "two-pass loss value mismatch"

    dsr = DifferentialSharpe()(net_strategy_return(z, r, sigma, 5e-5))
    assert torch.isfinite(dsr)
    print(f"losses self-test passed: fused cost-aware Sharpe, foresight bound, "
          f"DSR finite, two-pass equivalence rel-err {err:.2e} < 1e-5.")


# ==================== continuum.py ====================
"""
AURUM v2.0 -- "AURUM-Continuum": Purpose-Built Contractive Looped Transformer
===============================================================================
Blueprint 2 Section 4.2-4.3, verbatim (M5_ContinuumNet of the v2.0 tournament).
Every component is lifted from a verified source with its published
hyperparameter provenance:

  * TFT GRN (gated residual network) and Variable Selection Network
    -- Lim, Arik, Loeff, Pfister, "Temporal Fusion Transformers",
    arXiv:1912.09363, Eqs. (2)-(8).
  * Decoder-only TFT backbone with interpretable multi-head attention and a
    tanh position head with max-norm(3) kernel constraint
    -- Wood, Giegerich, Roberts, Zohren, "Trading with the Momentum
    Transformer", arXiv:2112.08534 (+ kieranjwood/trading-momentum-transformer).
  * Looped core with (i) per-layer spectral caps (Miyato et al.
    arXiv:1802.05957: W_SN = W / sigma(W), one power-iteration step), (ii)
    Krasnoselskii-Mann damping h <- (1-alpha) h + alpha f(h) which converges
    to the fixed point h* = f(h*) whenever f is L-Lipschitz with L < 1
    (monotone-operator iteration; cf. Winston & Kolter arXiv:2006.08591),
    and (iii) per-iterate supervision (Yang et al., "Looped Transformers...",
    arXiv:2311.12424: loss enforced on every iterate).
  * RevIN per-window instance normalization -- Kim et al., ICLR 2022
    (arXiv:2105.08499).
  * DMN-style direct tanh position output trained on the Sharpe loss --
    Lim, Zohren, Roberts, arXiv:1904.04912.

Loop stability, honestly stated: softmax attention has no closed-form
Lipschitz bound, so the spectral caps on all linear/attention projections
plus KM damping give a *sufficient-condition engineering bound* on the linear
parts, and `audit_loop_spectral_radius()` empirically estimates the spectral
radius rho of the TRUE loop-update Jacobian at real hidden states via JVP
power iteration.  Acceptance gate (G3.3): rho < 0.95 on every shipped
checkpoint.  If the gate fails, alpha is auto-reduced (auto_tune_alpha).
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lin(d_in: int, d_out: int, cap: float | None):
    """Linear layer factory: spectral-capped when cap is set, plain otherwise."""
    return _SpectralCappedLinear(d_in, d_out, cap) if cap else nn.Linear(d_in, d_out)


# ---------------------------------------------------------------- GRN (TFT)
class GRN(nn.Module):
    """Gated Residual Network, TFT Eq. (2)-(4) (light variant: no context).

    GRN(a) = LayerNorm(a + GLU(W1 * ELU(W2 a + b2)))
    GLU(g) = sigmoid(W4 g + b4) * (W5 g + b5)
    """

    def __init__(self, d_in: int, d_model: int, dropout: float = 0.2,
                 spectral_cap: float | None = None):
        super().__init__()
        self.fc_ctx = _lin(d_in, d_model, None) if d_in != d_model else None
        self.hidden = _lin(d_model, d_model, spectral_cap)
        self.gate_w = _lin(d_model, d_model, spectral_cap)
        self.gate_u = _lin(d_model, d_model, spectral_cap)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.fc_ctx is not None:
            a = self.fc_ctx(a)
        eta2 = F.elu(self.hidden(a))
        eta1 = self.dropout(eta2)
        glu = torch.sigmoid(self.gate_w(eta1)) * self.gate_u(eta1)
        return self.norm(a + glu)


class _SpectralCappedLinear(nn.Module):
    """Linear layer with hard cap on the spectral norm of its weight:
    W_eff = W * (cap / sigma(W)), sigma estimated by one power-iteration
    step per forward (Miyato et al. 2018)."""

    def __init__(self, d_in: int, d_out: int, cap: float = 0.95, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.cap = cap
        self.register_buffer("u", F.normalize(torch.randn(d_out), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(d_in), dim=0))

    def _sigma(self) -> torch.Tensor:
        """One recycled power-iteration step, fully detached: sigma is a
        constant for autograd purposes, gradients flow through W only."""
        with torch.no_grad():
            u = F.normalize(torch.mv(self.weight, self.v), dim=0)
            v = F.normalize(torch.mv(self.weight.t(), u), dim=0)
            self.u.copy_(u)
            self.v.copy_(v)
            return torch.dot(u, torch.mv(self.weight, v)).abs().clamp(min=1e-8)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * (self.cap / self._sigma())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)


# ------------------------------------------------- Variable Selection (TFT)
class VariableSelection(nn.Module):
    """TFT Eqs. (6)-(8): per-group GRN embeddings, softmax selection weights
    from a GRN over the flattened inputs, sparse weighted sum.  Also returns
    the selection weights for interpretability."""

    def __init__(self, group_dims: list[int], d_model: int, dropout: float = 0.2,
                 spectral_cap: float | None = None):
        super().__init__()
        self.n = len(group_dims)
        self.embeds = nn.ModuleList(
            [GRN(d, d_model, dropout, spectral_cap) for d in group_dims])
        self.weight_net = GRN(sum(group_dims), d_model, dropout, spectral_cap)
        self.sel = nn.Linear(d_model, self.n)

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        flat = torch.cat(xs, dim=-1)
        w = torch.softmax(self.sel(self.weight_net(flat)), dim=-1)   # (B, T, n)
        e = torch.stack([emb(x) for emb, x in zip(self.embeds, xs)], dim=-2)
        out = (w.unsqueeze(-1) * e).sum(dim=-2)
        return out, w


# ------------------------------------------------------ attention block
class CausalBlock(nn.Module):
    """Pre-norm causal self-attention (interpretable variant: shared value
    projection across heads, heads averaged -- MTX/TFT) + GRN feed-forward."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.2,
                 spectral_cap: float | None = None):
        super().__init__()
        self.n_heads = n_heads
        self.dk = d_model // n_heads
        self.q = _lin(d_model, d_model, spectral_cap)
        self.k = _lin(d_model, d_model, spectral_cap)
        self.v_shared = _lin(d_model, self.dk, spectral_cap)      # shared values (TFT)
        self.o = _lin(d_model, d_model, spectral_cap)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GRN(d_model, d_model, dropout, spectral_cap)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        x = self.norm1(h)
        q = self.q(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.dk).transpose(1, 2)
        v = self.v_shared(x).view(B, T, self.dk)                   # shared across heads
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        causal = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), 1)
        att = att.masked_fill(causal, float("-inf"))
        p = torch.softmax(att, dim=-1)
        ctx = torch.einsum("bhst,btd->bhsd", p, v)
        ctx = ctx.mean(dim=1)                                      # (B, T, dk)
        h = h + self.drop(self.o(F.pad(ctx, (0, D - self.dk))))
        return h + self.ffn(self.norm2(h))


# ------------------------------------------------------ contractive loop
class LoopBlock(nn.Module):
    """NON-RESIDUAL, fully spectral-capped refinement map
        f(h) = W2 act(W1 LN(h))
    with per-layer spectral caps c: ||J_f|| <= c^2 * ||J_LN|| < 1 by
    construction (LN affine initialized at 0.5 gain).  Iterating
    h <- (1-alpha) h + alpha f(h)  is then a Krasnoselskii-Mann iteration
    toward the fixed point h* = f(h*) -- a proximal-style latent DENOISER
    (Romano et al. RED arXiv:1611.02862; Gribonval-Nikolova arXiv:1807.04021).
    Attention lives in the prelude/coda (NOT looped)."""

    def __init__(self, d_model: int, cap: float = 0.8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=True)
        nn.init.constant_(self.norm.weight, 0.5)     # keep ||J_LN|| < 1 at init
        self.w1 = _SpectralCappedLinear(d_model, 2 * d_model, cap=cap)
        self.w2 = _SpectralCappedLinear(2 * d_model, d_model, cap=cap)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = self.norm(h)
        x = self.drop(F.gelu(self.w1(x)))
        return self.w2(x)


# ------------------------------------------------------------- the model
@dataclass
class ContinuumConfig:
    n_features: int = 330
    group_slices: list = field(default_factory=list)   # [(start, end, name), ...]
    d_model: int = 256
    n_heads: int = 4
    dropout: float = 0.2
    loop_iters: int = 4            # R: trained with per-iterate supervision
    km_alpha: float = 0.5          # KM damping: h <- (1-a) h + a f(h)
    spectral_cap: float = 0.80     # per-linear spectral cap inside the loop
    revin: bool = True
    revin_eps: float = 1e-5


class ContinuumNet(nn.Module):
    """2 prelude blocks -> spectrally-capped weight-tied looped core (R KM
    iterations) -> coda block -> heads (tanh position + aux), with RevIN on
    continuous channels and per-iterate readout."""

    def __init__(self, cfg: ContinuumConfig):
        super().__init__()
        self.cfg = cfg
        groups = cfg.group_slices or [(0, cfg.n_features, "all")]
        self.group_dims = [e - s for s, e, _ in groups]
        self.vsn = VariableSelection(self.group_dims, cfg.d_model, cfg.dropout)
        self.pos_enc = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.prelude = nn.ModuleList(
            [CausalBlock(cfg.d_model, cfg.n_heads, cfg.dropout) for _ in range(2)])
        self.loop = LoopBlock(cfg.d_model, cap=cfg.spectral_cap,
                              dropout=cfg.dropout)
        self.coda = CausalBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.pos_head = nn.Linear(cfg.d_model, 1)
        self.pos_head.weight.data.mul_(0.1)                       # small init
        self.ret_head = nn.Linear(cfg.d_model, 1)
        self.dir_head = nn.Linear(cfg.d_model, 1)
        self.meta_head = nn.Linear(cfg.d_model, 1)
        # DMN reference code: Dense(1, activation='tanh',
        # kernel_constraint=max_norm(3)) -- enforced as a pre-forward hook
        self._max_norm = 3.0
        self.pos_head.register_forward_pre_hook(self._clamp_pos_head)
        # learnable RevIN affine
        self.revin_gamma = nn.Parameter(torch.ones(cfg.n_features))
        self.revin_beta = nn.Parameter(torch.zeros(cfg.n_features))
        self.register_buffer("revin_mask", torch.ones(cfg.n_features))

    def _clamp_pos_head(self, module, inputs) -> None:
        with torch.no_grad():
            n = module.weight.norm()
            if n > self._max_norm:
                module.weight.mul_(self._max_norm / (n + 1e-12))

    def set_revin_mask(self, mask: torch.Tensor) -> None:
        """mask=1 -> normalize this channel (continuous), 0 -> passthrough
        (dummies / already-normalized)."""
        self.revin_mask.copy_(mask.float())

    def _revin(self, x: torch.Tensor) -> torch.Tensor:
        if not self.cfg.revin:
            return x
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True) + self.cfg.revin_eps
        xn = (x - mu) / sd
        x = self.revin_mask * xn + (1 - self.revin_mask) * x
        return self.revin_gamma * x + self.revin_beta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, n_features) -> trunk hidden states (B, T, d)."""
        x = self._revin(x)
        xs = [x[..., s:e] for s, e, _ in (self.cfg.group_slices or
                                          [(0, self.cfg.n_features, "all")])]
        h, _ = self.vsn(xs)
        h = h + self.pos_enc
        for blk in self.prelude:
            h = blk(h)
        return h

    def loop_refine(self, h0: torch.Tensor) -> list[torch.Tensor]:
        """R KM-damped iterations of the contractive LoopBlock toward the
        denoising fixed point.  Returns iterates [h_1 ... h_R]."""
        a = self.cfg.km_alpha
        h = h0
        outs = []
        for _ in range(self.cfg.loop_iters):
            h = (1.0 - a) * h + a * self.loop(h)
            outs.append(h)
        return outs

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h0 = self.encode(x)
        iterates = self.loop_refine(h0)
        hR = self.coda(iterates[-1])
        # attention pooling + last token
        w = torch.softmax(hR.mean(dim=1, keepdim=True), dim=-1)   # (B,1,d)
        pooled = (w * hR).sum(dim=1) + hR[:, -1]
        return {
            "z": torch.tanh(self.pos_head(hR)).squeeze(-1),        # (B, T) position
            "ret24": self.ret_head(pooled).squeeze(-1),
            "dir_logit": self.dir_head(pooled).squeeze(-1),
            "meta_logit": self.meta_head(pooled).squeeze(-1),
            "iterates": iterates,
        }

    def forward_per_iterate(self, x: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Readout at EVERY loop iterate (Yang et al. per-iterate
        supervision): applies the coda+heads cheaply per iterate via the
        position head only (aux heads stay on the final iterate)."""
        h0 = self.encode(x)
        iters = self.loop_refine(h0)
        zs = []
        for h in iters:
            hc = self.coda(h)
            zs.append(torch.tanh(self.pos_head(hc)).squeeze(-1))
        hR = self.coda(iters[-1])
        w = torch.softmax(hR.mean(dim=1, keepdim=True), dim=-1)
        pooled = (w * hR).sum(dim=1) + hR[:, -1]
        return {
            "z_iterates": zs,
            "z": torch.tanh(self.pos_head(hR)).squeeze(-1),
            "ret24": self.ret_head(pooled).squeeze(-1),
            "dir_logit": self.dir_head(pooled).squeeze(-1),
            "meta_logit": self.meta_head(pooled).squeeze(-1),
        }


@torch.no_grad()
def audit_loop_spectral_radius(model: "ContinuumNet", x: torch.Tensor,
                               n_power: int = 8, n_states: int = 4) -> float:
    """Empirical spectral radius (max singular value estimate) of the loop
    update Jacobian  g(h) = (1-alpha) h + alpha Loop(h)  at real hidden
    states, via JVP power iteration on the flattened state.  Acceptance gate
    (G3.3): rho < 0.95 on trained checkpoints."""
    model.eval()
    h0 = model.encode(x)
    B = h0.shape[0]
    idx = torch.randint(0, B, (min(n_states, B),))
    h = h0[idx].detach().clone()
    D = h.numel()

    def g(hh: torch.Tensor) -> torch.Tensor:
        a = model.cfg.km_alpha
        return (1.0 - a) * hh + a * model.loop(hh)

    v = torch.randn_like(h)
    v = v / v.norm()
    gain = 1.0
    for _ in range(n_power):
        _, jv = torch.autograd.functional.jvp(
            lambda z: (1.0 - model.cfg.km_alpha) * z + model.cfg.km_alpha * model.loop(z),
            (h,), (v,))
        nrm = jv.norm().item()
        if nrm < 1e-12:
            return 0.0
        gain = nrm                       # ||J v|| with ||v|| = 1
        v = jv / jv.norm()
    return gain


def auto_tune_alpha(model: "ContinuumNet", x: torch.Tensor,
                    rho_target: float = 0.95, min_alpha: float = 0.05) -> float:
    """Governor: if the audited spectral radius exceeds rho_target, shrink the
    KM damping so that (1-alpha) + alpha*L <= rho_target, using the measured
    loop gain L = (rho - (1-alpha)) / alpha.  Returns the new alpha."""
    rho = audit_loop_spectral_radius(model, x)
    if rho <= rho_target:
        return model.cfg.km_alpha
    a = model.cfg.km_alpha
    L = max((rho - (1.0 - a)) / max(a, 1e-6), 1.0 + 1e-6)
    a_new = max(min_alpha, min(a, (1.0 - rho_target) / (L - 1.0)))
    model.cfg.km_alpha = a_new
    return a_new


def self_test() -> None:
    torch.manual_seed(0)
    B, T, F_, G = 4, 64, 40, 5
    slices = [(i * F_ // G, (i + 1) * F_ // G, f"g{i}") for i in range(G)]
    cfg = ContinuumConfig(n_features=F_, group_slices=slices, d_model=64,
                          n_heads=4, dropout=0.1, loop_iters=4, km_alpha=0.5,
                          spectral_cap=0.8)
    m = ContinuumNet(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(B, T, F_)
    m.set_revin_mask((torch.rand(F_) > 0.3).float())
    out = m(x)
    assert out["z"].shape == (B, T), out["z"].shape
    assert out["z"].abs().max() <= 1.0
    assert len(out["iterates"]) == cfg.loop_iters
    # gradient flows end-to-end through the loop
    loss = out["z"].pow(2).mean() + out["ret24"].abs().mean()
    loss.backward()
    assert m.loop.w1.weight.grad is not None, "no grad to looped block"
    # per-iterate readout
    pi = m.forward_per_iterate(x)
    assert len(pi["z_iterates"]) == cfg.loop_iters
    assert pi["z"].shape == (B, T)
    # spectral audit + governor: init may be expansive (that is the v1.x
    # pathology!); the governor must bring rho under the 0.95 gate.
    rho0 = audit_loop_spectral_radius(m, x)
    a_new = auto_tune_alpha(m, x, rho_target=0.95)
    rho1 = audit_loop_spectral_radius(m, x)
    print(f"params={n_params/1e6:.2f}M | rho(init, alpha={cfg.km_alpha:.2f})="
          f"{rho0:.3f} -> alpha={a_new:.3f}, rho={rho1:.3f}")
    assert rho1 < 0.95, f"governor failed to bring spectral radius under gate: {rho1:.3f}"
    # governor must keep the model functional (forward still works, grads flow)
    out2 = m(x)
    out2["z"].pow(2).mean().backward()
    print("ContinuumNet self-test passed: shapes, grads, per-iterate, audit+governor.")


# ==================== ae_mlp_dmn.py ====================
"""
AURUM v2.0 -- Compact Continuous Trunks: AE-MLP family + DMN family
====================================================================
Tournament members (mandate II.4):
  * M3_DAE_ResMLP, compact scale  -- JaneStreetDAEResMLP (Blueprint 1 5.4:
    10,449 verified params at D=48, d_enc=32; Gaussian/swap noise).
  * M3_DAE_ResMLP, medium scale   -- AEMLP (Blueprint 2 4.6: enc 96,
    hidden 512-256-256-128, the JS-2021 winner recipe).
  * M4_DMN_Lite                   -- OxfordManDMNLite (Blueprint 1 5.4:
    Vectorized VSN + GRU + 2-Head Masked MHA + GRN, 6,129 params at D=48)
    with the DMN max-norm(3) readout constraint added (mandate II.4).
  * DMN-LSTM reference            -- DMNLSTM (Blueprint 2 4.6: Lim-Zohren-
    Roberts 2019 reference config, hidden 80, 63-step unroll).

All trained on the fused RobustNetSharpeLoss / NetSharpeLoss (losses.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



# --------------------------------------------------------------- AE-MLP
class AEMLP(nn.Module):
    """Supervised denoising-autoencoder MLP (JS-2021 winner recipe, adapted;
    Blueprint 2 4.6 -- the MEDIUM scale of M3, d_enc=96).

    encoder: BN -> GaussianNoise(sigma) -> Dense(enc) -> BN -> swish
    decoder: Dropout -> Dense(n_features)   (MSE reconstruction)
    head:    concat(BN(x), latent) -> Dense stacks (BN + swish + dropout)
             -> multi-target outputs (ret24 linear, dir24 logit, meta logit)
    """

    def __init__(self, n_features: int, enc_dim: int = 96,
                 hidden=(512, 256, 256, 128), noise_sigma: float = 0.035,
                 dropout=(0.30, 0.30, 0.25, 0.40), swap_noise: float = 0.0):
        super().__init__()
        self.noise_sigma = noise_sigma
        self.swap_noise = swap_noise
        self.bn_in = nn.BatchNorm1d(n_features)
        self.encoder = nn.Linear(n_features, enc_dim)
        self.enc_bn = nn.BatchNorm1d(enc_dim)
        self.dec_drop = nn.Dropout(0.05)
        self.decoder = nn.Linear(enc_dim, n_features)
        layers = []
        d_in = n_features + enc_dim
        for d, p in zip(hidden, dropout):
            layers += [nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.SiLU(), nn.Dropout(p)]
            d_in = d
        self.mlp = nn.Sequential(*layers)
        self.out_ret = nn.Linear(d_in, 1)
        self.out_dir = nn.Linear(d_in, 1)
        self.out_meta = nn.Linear(d_in, 1)
        # dedicated position head for the Sharpe trainer (per-bar z in [-1,1])
        self.out_pos = nn.Linear(d_in, 1)
        self.out_pos.weight.data.mul_(0.1)
        self._max_norm = 3.0
        self.out_pos.register_forward_pre_hook(self._clamp_pos)

    def _clamp_pos(self, module, inputs):
        with torch.no_grad():
            n = module.weight.norm()
            if n > self._max_norm:
                module.weight.mul_(self._max_norm / (n + 1e-12))

    @staticmethod
    def _swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        if self.training and self.swap_noise > 0:
            # Jahrer swap noise: replace p of features with values from other rows
            B, F_ = x.shape
            mask = torch.rand(B, F_, device=x.device) < self.swap_noise
            perm = torch.randperm(B, device=x.device)
            x = torch.where(mask, x[perm], x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Accepts (B, F) tabular rows or (B, T, F) causal windows (the
        windowed Sharpe trainer); returns per-bar position z in [-1, 1] from
        the dedicated position head, plus per-row aux targets."""
        orig_shape = x.shape
        if x.dim() == 3:                       # (B, T, F) windowed training
            B, T, D = orig_shape
            x = x.reshape(B * T, D)
        xc = self._corrupt(x)
        xn = self.bn_in(xc)
        lat = self._swish(self.enc_bn(self.encoder(xn)))
        rec = self.decoder(self.dec_drop(lat))
        h = self.mlp(torch.cat([xn, lat], dim=1))
        z = torch.tanh(self.out_pos(h)).squeeze(-1)
        ret = self.out_ret(h).squeeze(-1)
        dir_l = self.out_dir(h).squeeze(-1)
        meta = self.out_meta(h).squeeze(-1)
        if len(orig_shape) == 3:
            rec = rec.reshape(B, T, D)
            z = z.reshape(B, T)
            ret = ret.reshape(B, T).mean(dim=1)
            dir_l = dir_l.reshape(B, T).mean(dim=1)
            meta = meta.reshape(B, T).mean(dim=1)
        return {"z": z, "ret24": ret, "dir_logit": dir_l,
                "meta_logit": meta, "reconstruction": rec}


def ae_mlp_loss(out: dict, x_norm: torch.Tensor, y_ret: torch.Tensor, y_dir: torch.Tensor,
                w_rec: float = 1.0, w_ret: float = 1.0, w_dir: float = 0.5,
                w_meta: float = 0.25) -> torch.Tensor:
    """Multi-task loss.  x_norm = the batch-normalized input actually seen by
    the encoder; sample weights = |y_ret|."""
    sw = (y_ret.abs() / (y_ret.abs().mean() + 1e-9)).clamp(0.2, 3.0).detach()
    l_rec = F.mse_loss(out["reconstruction"], x_norm.detach())
    l_ret = (sw * F.mse_loss(out["ret24"], y_ret, reduction="none")).mean()
    l_dir = (sw * F.binary_cross_entropy_with_logits(
        out["dir_logit"], (y_dir > 0).float(), reduction="none")).mean()
    meta_y = (torch.sign(out["ret24"].detach()) == y_dir.sign()).float()
    l_meta = F.binary_cross_entropy_with_logits(out["meta_logit"], meta_y)
    return w_ret * l_ret + w_dir * l_dir + w_meta * l_meta + w_rec * l_rec


# ------------------------------------------- Jane Street DAE-ResMLP (B1, compact)
class JaneStreetDAEResMLP(nn.Module):
    """M3 COMPACT scale (Blueprint 1 5.4): Yirun Zhang 1st-place Jane Street
    bottleneck DAE + Residual MLP adapted for continuous position sizing.
    10,449 verified params at D=48, d_enc=32.

    forward returns dict with 'z' (B, T) tanh position, 'ret24' (B,) from the
    AE readout head, 'dir_logit'/'meta_logit' tied to the main action, and
    'reconstruction' -- compatible with the fused Sharpe trainer.
    """

    def __init__(self, in_features: int = 48, d_enc: int = 32, hidden_dims=(64, 32),
                 noise_std: float = 0.10, swap_noise: float = 0.0):
        super().__init__()
        self.noise_std = noise_std
        self.swap_noise = swap_noise
        self.bn_in = nn.BatchNorm1d(in_features)

        # Bottleneck Encoder & Decoder
        self.enc_dense = nn.Linear(in_features, d_enc)
        self.enc_bn = nn.BatchNorm1d(d_enc)
        self.dec_drop = nn.Dropout(0.15)
        self.dec_dense = nn.Linear(d_enc, in_features)

        # Auxiliary AE Readout
        self.ae_head = nn.Sequential(
            nn.Linear(in_features, d_enc),
            nn.BatchNorm1d(d_enc),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(d_enc, 1),
        )

        # Main Residual MLP Predictor ([x0 || z] -> 64 -> 32 -> 1)
        self.cat_bn = nn.BatchNorm1d(in_features + d_enc)
        self.cat_drop = nn.Dropout(0.20)
        self.mlp_block1 = nn.Sequential(
            nn.Linear(in_features + d_enc, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.mlp_block2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.out_action = nn.Linear(hidden_dims[1], 1)
        self._max_norm = 3.0
        self.out_action.register_forward_pre_hook(self._clamp)

    def _clamp(self, module, inputs):
        with torch.no_grad():
            n = module.weight.norm()
            if n > self._max_norm:
                module.weight.mul_(self._max_norm / (n + 1e-12))

    def _corrupt(self, x):
        if self.training and self.noise_std > 0.0:
            x = x + torch.randn_like(x) * self.noise_std
        if self.training and self.swap_noise > 0:
            B, F_ = x.shape
            mask = torch.rand(B, F_, device=x.device) < self.swap_noise
            perm = torch.randperm(B, device=x.device)
            x = torch.where(mask, x[perm], x)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        orig_shape = x.shape
        if x.dim() == 3:
            B, T, D = orig_shape
            x_flat = x.reshape(B * T, D)
        else:
            x_flat = x

        x0 = self.bn_in(x_flat)
        tilde_x0 = self._corrupt(x0)
        z = F.silu(self.enc_bn(self.enc_dense(tilde_x0)))
        x_rec = self.dec_dense(self.dec_drop(z))
        ae_act = self.ae_head(x_rec)

        cat = self.cat_drop(self.cat_bn(torch.cat([x0, z], dim=-1)))
        h1 = self.mlp_block1(cat)
        h2 = self.mlp_block2(h1)
        main_act = self.out_action(h2)

        if len(orig_shape) == 3:
            main_act = main_act.reshape(B, T)
            ae_act = ae_act.reshape(B, T)
            x_rec = x_rec.reshape(B, T, D)
            ret24 = ae_act.mean(dim=1)
        else:
            main_act = main_act.squeeze(-1)
            ret24 = ae_act.squeeze(-1)

        return {
            "z": torch.tanh(main_act),          # (B, T) position in [-1, 1]
            "ret24": ret24,                      # (B,) AE aux readout
            "dir_logit": ret24,                  # tied aux heads (compact net)
            "meta_logit": ret24,
            "reconstruction": x_rec,
        }


# ------------------------------------------- Oxford-Man DMN-Lite (B1, M4)
class GatedResidualNetwork(nn.Module):
    """Exact Lim et al. (2021) / Wood et al. (2021) GRN with GLU and
    GateAddNorm (Blueprint 1 5.4 compact form)."""

    def __init__(self, d_model: int = 16, dropout: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.glu_gate = nn.Linear(d_model, d_model)
        self.glu_val = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        eta2 = F.elu(self.fc1(a))
        eta1 = self.drop(self.fc2(eta2))
        glu = torch.sigmoid(self.glu_gate(eta1)) * self.glu_val(eta1)
        return self.norm(a + glu)


class OxfordManDMNLite(nn.Module):
    """M4_DMN_Lite (Blueprint 1 5.4): Compact Vectorized Variable Selection +
    GRU + Masked 2-Head Attention + GRN.  6,129 verified params at D=48,
    d_model=16.  The readout carries the DMN max-norm(3) constraint (hook)
    and a tanh position output compatible with the Sharpe trainer."""

    def __init__(self, in_features: int = 48, d_model: int = 16, n_heads: int = 2,
                 dropout: float = 0.15):
        super().__init__()
        self.vsn_gates = nn.Linear(in_features, in_features)
        self.vsn_proj = nn.Linear(in_features, d_model)
        self.vsn_grn = GatedResidualNetwork(d_model, dropout)

        self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
        self.mha = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout,
                                         batch_first=True)
        self.glu_att_gate = nn.Linear(d_model, d_model)
        self.glu_att_val = nn.Linear(d_model, d_model)
        self.norm_att = nn.LayerNorm(d_model)

        self.post_grn = GatedResidualNetwork(d_model, dropout)
        self.readout = nn.Linear(d_model, 1)
        self.readout.weight.data.mul_(0.1)
        self._max_norm = 3.0
        self.readout.register_forward_pre_hook(self._clamp)
        # aux heads on the pooled state (multi-task supervision)
        self.ret_head = nn.Linear(d_model, 1)
        self.dir_head = nn.Linear(d_model, 1)
        self.meta_head = nn.Linear(d_model, 1)

    def _clamp(self, module, inputs):
        with torch.no_grad():
            n = module.weight.norm()
            if n > self._max_norm:
                module.weight.mul_(self._max_norm / (n + 1e-12))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        B, T, D = x.shape
        v_weights = F.softmax(self.vsn_gates(x), dim=-1)
        x_selected = x * v_weights
        h_vsn = self.vsn_grn(self.vsn_proj(x_selected))

        h_gru, _ = self.gru(h_vsn)
        # Strictly causal upper-triangular attention mask (tau > t masked out)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool),
                                 diagonal=1)
        att_out, _ = self.mha(h_gru, h_gru, h_gru, attn_mask=causal_mask)

        glu_att = torch.sigmoid(self.glu_att_gate(att_out)) * self.glu_att_val(att_out)
        h_att = self.norm_att(h_gru + glu_att)
        h_final = self.post_grn(h_att)
        a = self.readout(h_final).squeeze(-1)                 # (B, T)
        pooled = h_final.mean(dim=1)
        ret24 = self.ret_head(pooled).squeeze(-1)
        dir_logit = self.dir_head(pooled).squeeze(-1)
        meta_logit = self.meta_head(pooled).squeeze(-1)
        return {"z": torch.tanh(a), "ret24": ret24,
                "dir_logit": dir_logit, "meta_logit": meta_logit,
                "v_weights": v_weights}


# ------------------------------------------------------------- DMN-LSTM
class DMNLSTM(nn.Module):
    """Deep Momentum Network (Lim-Zohren-Roberts 2019), reference config:
    LSTM(hidden 80) -> Dropout -> Dense(1, tanh, max_norm 3); input = the
    trailing sequence of volatility-normalized XAUUSD returns (63 steps)."""

    def __init__(self, n_inputs: int = 2, hidden: int = 80, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        self.head.weight.data.mul_(0.1)
        self.max_norm = 3.0
        self.head.register_forward_pre_hook(self._clamp)
        self.ret_head = nn.Linear(hidden, 1)
        self.dir_head = nn.Linear(hidden, 1)
        self.meta_head = nn.Linear(hidden, 1)

    def _clamp(self, module, inputs):
        with torch.no_grad():
            n = module.weight.norm()
            if n > self.max_norm:
                module.weight.mul_(self.max_norm / (n + 1e-12))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, _ = self.lstm(x)
        a = self.head(self.drop(h)).squeeze(-1)
        pooled = h.mean(dim=1)
        return {"z": torch.tanh(a),
                "ret24": self.ret_head(pooled).squeeze(-1),
                "dir_logit": self.dir_head(pooled).squeeze(-1),
                "meta_logit": self.meta_head(pooled).squeeze(-1)}


def dmn_loss(z: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor,
             bars_per_year: int = 5786) -> torch.Tensor:
    """DMN annualized Sharpe loss (their Eq. 16 with the reference-code eps)."""
    crit = NetSharpeLoss(bars_per_year=bars_per_year)
    return crit(z, r, sigma)


def self_test() -> None:
    torch.manual_seed(0)
    B, F_ = 64, 30
    x = torch.randn(B, F_)
    m = AEMLP(F_)
    out = m(x)
    assert out["ret24"].shape == (B,)
    assert out["reconstruction"].shape == (B, F_)
    y_ret = torch.randn(B)
    y_dir = torch.sign(torch.randn(B))
    l = ae_mlp_loss(out, m.bn_in(x), y_ret, y_dir)
    l.backward()
    assert m.encoder.weight.grad is not None
    # swap-noise variant
    m2 = AEMLP(F_, swap_noise=0.15)
    _ = m2(x)
    # DMN-LSTM
    T = 63
    xs = torch.randn(B, T, 2)
    dmn = DMNLSTM()
    z = dmn(xs)
    assert z["z"].shape == (B, T) and z["z"].abs().max() <= 1.0
    r = torch.randn(B, T) * 0.001
    s = torch.full((B, T), 0.001)
    loss = dmn_loss(z["z"], r, s)
    assert torch.isfinite(loss)
    loss.backward()

    # JaneStreetDAEResMLP (compact M3): param count + 3D input
    m3c = JaneStreetDAEResMLP(in_features=48, d_enc=32)
    n_par = sum(p.numel() for p in m3c.parameters())
    x3 = torch.randn(8, 50, 48)
    o3 = m3c(x3)
    assert o3["z"].shape == (8, 50) and o3["z"].abs().max() <= 1.0
    assert o3["reconstruction"].shape == (8, 50, 48)
    (o3["z"].pow(2).mean()).backward()

    # OxfordManDMNLite (M4): shapes, causality of mask, param count at D=48
    m4 = OxfordManDMNLite(in_features=48, d_model=16, n_heads=2)
    n_par4 = sum(p.numel() for p in m4.parameters())
    x4 = torch.randn(8, 63, 48)
    m4.eval()                              # disable dropout for the causality check
    o4 = m4(x4)
    assert o4["z"].shape == (8, 63) and o4["z"].abs().max() <= 1.0
    # causal check: perturbing step k must not change positions at steps < k
    x4b = x4.clone()
    x4b[:, 40, :] += 10.0
    o4b = m4(x4b)
    assert torch.allclose(o4["z"][:, :40], o4b["z"][:, :40], atol=1e-6), \
        "DMN-Lite attention mask is not causal!"
    print(f"AE-MLP + DMN-LSTM self-tests passed (compact DAE {n_par} params, "
          f"DMN-Lite {n_par4} params, causal mask verified).")


# ==================== purge.py ====================
"""
AURUM v2.0 -- Purged Walk-Forward Splitter (purge.py)
======================================================
Blueprint 2 Section 4.9, verbatim.

Semantics follow Lopez de Prado, Advances in Financial Machine Learning (2018),
ch. 7 (purged k-fold CV with embargo), cross-verified against the open
skfolio reproduction of CombinatorialPurgedCV:

  PURGE : drop from the TRAINING set every sample whose LABEL WINDOW
          (t, t+H] overlaps the label window of any TEST sample.  With label
          horizon H = 24 bars and contiguous yearly test blocks, this removes
          train rows in [test_start - H, test_start).
  EMBARGO: additionally drop train rows immediately FOLLOWING the test block
          (serial correlation of features leaks backward in time):
          [test_end + 1, test_end + H + embargo].
  Constraint (skfolio): purge + embargo < fold size - 1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    label_horizon: int = 24          # H in bars
    embargo: int = 24                # extra bars after test block (24-48 rec.)
    test_block: str = "365D"         # contiguous test block length
    min_train: int = 24 * 252 * 2    # burn-in before first test block

    def splits(self, index: pd.DatetimeIndex):
        """Yield (train_idx, test_idx) position arrays, expanding window."""
        idx = pd.DatetimeIndex(index)
        blocks = pd.date_range(idx[self.min_train], idx[-1],
                               freq=self.test_block, tz=idx.tz)
        for i, bstart in enumerate(blocks):
            bend = blocks[i + 1] if i + 1 < len(blocks) else idx[-1] + pd.Timedelta("1h")
            t_lo = idx.searchsorted(bstart)
            t_hi = idx.searchsorted(bend)
            if t_hi - t_lo < self.label_horizon + 2:
                continue
            test = np.arange(t_lo, t_hi)
            # purge: train labels must not overlap the test label window
            purge_lo = max(0, t_lo - self.label_horizon)
            # embargo after the test block
            emb_hi = min(len(idx), t_hi + self.label_horizon + self.embargo)
            train = np.concatenate([np.arange(0, purge_lo),
                                    np.arange(emb_hi, len(idx))])
            train = train[train >= self.min_train]
            # enforce the skfolio constraint
            if self.label_horizon + self.embargo >= (t_hi - t_lo) - 1:
                raise ValueError("purge + embargo >= fold size - 1; reduce embargo")
            if len(train) > 0:
                yield train, test


def self_test() -> None:
    idx = pd.date_range("2016-01-01", "2026-01-01", freq="h", tz="UTC")
    wf = PurgedWalkForward(label_horizon=24, embargo=24, min_train=24 * 252 * 2)
    n = 0
    last_test_end = -1
    prev_pre = 0
    for train, test in wf.splits(idx):
        # test blocks are disjoint and ordered
        assert test.min() > last_test_end
        last_test_end = test.max()
        # no train row within the purge window before the test block
        assert not np.any((train >= test.min() - 24) & (train < test.min()))
        # no train row within the embargo window after the test block
        assert not np.any((train > test.max()) & (train <= test.max() + 24 + 24))
        # expanding window: pre-block train grows over folds
        if n > 0:
            assert (train < test.min()).sum() >= prev_pre
        prev_pre = (train < test.min()).sum()
        n += 1
    assert n >= 6, f"expected >=6 yearly folds, got {n}"
    print(f"PurgedWalkForward self-test passed: {n} folds, purge=24, embargo=24.")


# ==================== train_continuum.py ====================
"""
AURUM v2.0 -- ContinuumNet Trainer (with Two-Pass support)
===========================================================
Blueprint 2 Section 4.5 trainer, extended per mandate II.3 to use the FUSED
RobustNetSharpeLoss and two_pass_exact_sharpe_step.

Training protocol assembled from verified practice:
  * Fused Sharpe loss on the continuous position with costs inside the graph
    + SoftMin worst-window penalty + Ulcer penalty (losses.py).
  * Per-iterate supervision of the loop (Yang et al. arXiv:2311.12424).
  * Early stopping on VALIDATION PORTFOLIO SHARPE, patience 25 (DMN/MTX
    practice -- never on val loss), 3-seed averaging, random search budget.
  * Spectral governor: every `govern_every` epochs, audit the loop's
    empirical Jacobian radius; if rho >= 0.95, shrink km_alpha
    (auto_tune_alpha) -- the v1.x failure mode is structurally impossible.
  * Optional Two-Pass Exact Gradient Accumulation for memory-constrained
    batches (DeePM; exactness self-tested to < 1e-5 in losses.self_test).
  * fp16 autocast on CUDA (T4 sm_75: fp16 only, NO bf16).
  * Purged expanding walk-forward (purge 24 bars, embargo 24-48 bars).
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn



@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 32
    lr: float = 1e-3
    clipnorm: float = 1.0
    max_epochs: int = 100
    patience: int = 25
    cost_r: float = 5e-5            # per-unit-exposure round-turn cost
    gamma_cost_scaler: float = 0.5  # DeePM training cost scaler
    sigma_tgt_ann: float = 0.15
    bars_per_year: int = 5786
    softmin_tau: float = 0.2
    softmin_lambda: float = 0.25
    lam_dd: float = 0.10
    lam_turn: float = 0.0
    w_iterate_decay: float = 0.7    # per-iterate loss weight decay
    w_aux: float = 0.15             # aux multi-task weight
    govern_every: int = 5           # spectral-governor cadence (epochs)
    rho_gate: float = 0.95
    two_pass: bool = False          # use two_pass_exact_sharpe_step
    micro_batches: int = 2          # micro-batches per logical batch (two-pass)
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True


class WindowDataset(torch.utils.data.Dataset):
    """Random-access windows over the H1 feature frame.

    X[i] = features[t_i - L + 1 : t_i + 1]        (causal window)
    r[i] = raw returns aligned so that z[:, tau] trades r[:, tau]
           (position decided at bar close tau, earns return tau -> tau+1)
    """

    def __init__(self, feats: np.ndarray, ret: np.ndarray, sigma: np.ndarray,
                 y_ret24: np.ndarray, y_dir: np.ndarray, idx: np.ndarray,
                 seq_len: int):
        self.feats = torch.as_tensor(feats, dtype=torch.float32)
        self.ret = torch.as_tensor(ret, dtype=torch.float32)
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32)
        self.y_ret24 = torch.as_tensor(y_ret24, dtype=torch.float32)
        self.y_dir = torch.as_tensor(y_dir, dtype=torch.float32)
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.L = seq_len

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = int(self.idx[i])
        lo = max(0, t - self.L + 1)
        x = self.feats[lo: t + 1]
        if x.shape[0] < self.L:                      # left-pad with the first row
            pad = self.feats[lo].unsqueeze(0).repeat(self.L - x.shape[0], 1)
            x = torch.cat([pad, x], dim=0)
        r = self.ret[lo: t + 1]
        if r.shape[0] < self.L:
            r = torch.cat([torch.zeros(self.L - r.shape[0]), r])
        s = self.sigma[lo: t + 1]
        if s.shape[0] < self.L:
            s = torch.cat([torch.full((self.L - s.shape[0],), float("nan")), s])
            s = torch.nan_to_num(s, nan=1e-4)
        return x, r, s, self.y_ret24[t], self.y_dir[t]


def _finite_fill(a: np.ndarray) -> np.ndarray:
    """Causal NaN fill for the feature matrix (forward-fill then zero)."""
    import pandas as pd
    return pd.DataFrame(a).ffill().fillna(0.0).values.astype("float32")


def portfolio_sharpe(z, r, sigma, cfg: TrainConfig) -> float:
    """Annualized net Sharpe of a position series (evaluation helper)."""
    z_t = torch.as_tensor(z, dtype=torch.float32).unsqueeze(0)
    r_t = torch.as_tensor(r, dtype=torch.float32).unsqueeze(0)
    s_t = torch.as_tensor(sigma, dtype=torch.float32).unsqueeze(0)
    R = net_strategy_return(z_t, r_t, s_t, cfg.cost_r,
                            cfg.sigma_tgt_ann / math.sqrt(cfg.bars_per_year))
    R = R.squeeze(0).numpy()
    mu, sd = R.mean(), R.std()
    return float(mu / (sd + 1e-12) * math.sqrt(cfg.bars_per_year))


def train_one_fold(feats: np.ndarray, ret: np.ndarray, sigma: np.ndarray,
                   y_ret24: np.ndarray, y_dir: np.ndarray,
                   train_idx: np.ndarray, val_idx: np.ndarray,
                   groups: list, net_cfg: ContinuumConfig,
                   cfg: TrainConfig) -> tuple[ContinuumNet, dict]:
    """Train a single walk-forward fold; returns (model, history)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device

    feats = _finite_fill(feats)
    ret = np.nan_to_num(ret, nan=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-4)
    y_ret24 = np.nan_to_num(y_ret24, nan=0.0)

    tr = WindowDataset(feats, ret, sigma, y_ret24, y_dir, train_idx, cfg.seq_len)
    va = WindowDataset(feats, ret, sigma, y_ret24, y_dir, val_idx, cfg.seq_len)
    trl = torch.utils.data.DataLoader(tr, batch_size=cfg.batch_size, shuffle=True,
                                      drop_last=True)
    val_x = torch.stack([va[i][0] for i in range(len(va))]).to(device)
    val_r = torch.stack([va[i][1] for i in range(len(va))]).to(device)
    val_s = torch.stack([va[i][2] for i in range(len(va))]).to(device)

    model = ContinuumNet(net_cfg).to(device)
    # RevIN mask: normalize continuous channels (std > 0 in train), pass
    # through near-constant/dummy channels
    with torch.no_grad():
        stds = torch.as_tensor(feats[train_idx]).std(dim=0)
        mask = (stds > 1e-6).float().to(device)
        model.set_revin_mask(mask)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = RobustNetSharpeLoss(bars_per_year=cfg.bars_per_year,
                               cost_r=cfg.cost_r,
                               gamma_cost_scaler=cfg.gamma_cost_scaler,
                               sigma_tgt_ann=cfg.sigma_tgt_ann,
                               softmin_tau=cfg.softmin_tau,
                               softmin_lambda=cfg.softmin_lambda,
                               lam_dd=cfg.lam_dd, lam_turn=cfg.lam_turn)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    best_sharpe, best_state, bad = -1e9, None, 0
    hist = {"val_sharpe": [], "rho": [], "alpha": []}
    for epoch in range(cfg.max_epochs):
        model.train()
        for x, r, s, y_r, y_d in trl:
            x, r, s = x.to(device), r.to(device), s.to(device)
            y_r, y_d = y_r.to(device), y_d.to(device)
            opt.zero_grad(set_to_none=True)
            if cfg.two_pass:
                # micro-batch split along the batch dim (logical batch intact)
                mbs = []
                B = x.shape[0]
                k = max(1, B // cfg.micro_batches)
                for i0 in range(0, B, k):
                    mbs.append((x[i0:i0 + k], r[i0:i0 + k], s[i0:i0 + k],
                                y_r[i0:i0 + k], y_d[i0:i0 + k], None))
                two_pass_exact_sharpe_step(model, crit, opt, mbs,
                                           aux_weight=cfg.w_aux,
                                           clipnorm=cfg.clipnorm)
            else:
                with torch.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                    out = model.forward_per_iterate(x)
                    loss = crit(out["z"], r, s)
                    # per-iterate supervision (Yang et al.): Sharpe loss on each
                    # iterate's position with geometrically decaying weight
                    for j, zj in enumerate(out["z_iterates"][1:], start=2):
                        loss = loss + (cfg.w_iterate_decay ** j) * crit(zj, r, s)
                    loss = loss + cfg.w_aux * aux_losses(
                        out["dir_logit"], out["ret24"], out["meta_logit"], y_d, y_r)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clipnorm)
                scaler.step(opt)
                scaler.update()

        # ---- validation: ANNUALIZED NET SHARPE (DMN/MTX early-stop metric)
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=cfg.amp and device == "cuda"):
            zs = []
            for i in range(0, len(va), 512):
                zs.append(model(val_x[i: i + 512])["z"].float())
            z_val = torch.cat(zs)
        z_np = z_val[:, -1].cpu().numpy()          # final-step positions
        r_np = val_r[:, -1].cpu().numpy()
        s_np = val_s[:, -1].cpu().numpy()
        sharpe = portfolio_sharpe(z_np, r_np, s_np, cfg)
        hist["val_sharpe"].append(sharpe)

        # ---- spectral governor: audit + shrink km_alpha if rho >= gate
        if epoch % cfg.govern_every == 0:
            rho = auto_tune_alpha(model, val_x[:4], rho_target=cfg.rho_gate)
            hist["rho"].append(rho)
            hist["alpha"].append(net_cfg.km_alpha)
        if sharpe > best_sharpe + 1e-4:
            best_sharpe, bad = sharpe, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    hist["best_val_sharpe"] = best_sharpe
    return model, hist


def self_test() -> None:
    """Tiny end-to-end train on synthetic data: the model must learn to hold
    sign(signal) on a predictable synthetic return stream (sanity that the
    Sharpe-loss training loop actually optimizes the right thing) -- in BOTH
    single-batch and two-pass modes."""
    torch.manual_seed(0)
    T, F_ = 1500, 12
    rng = np.random.default_rng(0)
    f = rng.standard_normal((T, F_)).astype("float32")
    r = 0.30 * f[:, 0] * 0.001 + rng.standard_normal(T) * 0.001
    r = np.roll(r, -1); r[-1] = 0.0                 # r[t] = return t -> t+1
    sigma = np.full(T, 0.0014, dtype="float32")
    y_r = np.roll(0.02 * f[:, 0] * 0.001, -1).astype("float32")
    y_d = np.sign(y_r)
    groups = [(0, 4, "g0"), (4, 8, "g1"), (8, 12, "g2")]
    results = {}
    for two_pass in (False, True):
        net_cfg = ContinuumConfig(n_features=F_, group_slices=groups, d_model=32,
                                  n_heads=2, dropout=0.1, loop_iters=3,
                                  km_alpha=0.5, spectral_cap=0.8)
        cfg = TrainConfig(seq_len=64, batch_size=16, lr=2e-3, max_epochs=12,
                          patience=6, cost_r=1e-8, lam_dd=0.0, lam_turn=0.0,
                          softmin_lambda=0.0, device="cpu", amp=False,
                          govern_every=3, seed=0, two_pass=two_pass)
        train_idx = np.arange(100, 1100)
        val_idx = np.arange(1100, 1450)
        model, hist = train_one_fold(f, r, sigma, y_r, y_d, train_idx, val_idx,
                                     groups, net_cfg, cfg)
        assert np.isfinite(hist["best_val_sharpe"])
        assert hist["best_val_sharpe"] > 1.0, \
            f"training failed to learn the synthetic signal: {hist['best_val_sharpe']:.3f}"
        results[two_pass] = hist["best_val_sharpe"]
    print(f"train_continuum self-test passed: best val Sharpe "
          f"{results[False]:.2f} (single-batch) / {results[True]:.2f} (two-pass) "
          f"on synthetic signal (plumbing OK).")


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
    vov_series: list[float] = []

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
            vov_series.append(float(vov[t]) if np.isfinite(vov[t]) else 0.0)
            if len(vov_series) > 24 * 30:
                q = np.quantile(vov_series, 0.8)
                if vov_series[-1] > q:
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
    on every exposure change (the anti-buffer)."""
    idx = z.index
    zv = z.values.astype("float64")
    rv = r.reindex(idx).values
    sv = sigma.reindex(idx).values
    v = (SIGMA_TGT_ANN / math.sqrt(BARS_PER_YEAR)) / (sv + 1e-12)
    w = zv * v
    dzv = np.abs(np.diff(w, prepend=0.0))
    net = zv * v * rv - COST_R * cost_mult * dzv
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
    day), resampled with replacement."""
    rng = np.random.default_rng(seed)
    net = np.asarray(net, dtype="float64")
    net = net[np.isfinite(net)]
    T = len(net)
    n_blocks = T // block
    if n_blocks < 4:
        return {"sharpe_ci": [None, None], "n_boot": 0}
    blocks = net[: n_blocks * block].reshape(n_blocks, block)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        sel = rng.integers(0, n_blocks, n_blocks)
        sample = blocks[sel].reshape(-1)
        mu, sd = sample.mean(), sample.std(ddof=1)
        stats[b] = mu / (sd + 1e-12) * math.sqrt(BARS_PER_YEAR)
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
AURUM v2.0 -- K3 kernel main: Neural & Contractive-Loop Training (GPU)
======================================================================
Phase 3 of the v2.0 execution plan (runs on the secondary account's 2xT4):

  * M3_DAE_ResMLP : Jane-Street bottleneck DAE + ResMLP at compact
                    (d_enc=32) and medium (AEMLP d_enc=96) scales.
  * M4_DMN_Lite   : Oxford-Man VSN + GRU + 2-head masked MHA + GRN with
                    max-norm(3) readout, on 63+ step windows.
  * M5_ContinuumNet: contractive looped transformer (RevIN + Group VSN +
                    prelude + spectrally-capped non-residual LoopBlock
                    (sigma(W)<=0.8, KM damping, per-iterate supervision,
                    JVP spectral-radius governor rho<0.95) + coda),
                    d_model sweep {64, 128, 256}.

All models train on the FUSED RobustNetSharpeLoss (costs + SoftMin + Ulcer
inside the graph, gamma=0.5 cost scaler, sqrt(5786) annualizer) with early
stopping on validation net portfolio Sharpe (patience from cfg), under the
year-based purged walk-forward (fold y trains on bars < y with 24-bar purge).

Selection protocol: config search on train(<2022)/val(2022); the best config
per family refits across the 10 yearly folds (2016-2025); 2023-25 fold
predictions remain untouched OOS until K6.

Every configuration enters the Master Trial Ledger.  The spectral governor
log (rho, alpha per checkpoint) and the M5 per-iterate z series
(r=1..R loop-depth analysis vs v1.3's unconstrained loop) are exported.
"""
import json
import math
import os
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

GATES = {}
LOG_LINES = []
T_START = time.time()
HARD_BUDGET_S = 1.35 * 3600    # COMPLETION kernel: M5 2020-25 folds + M4 fp32 + M3m


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def gate(gid, ok, detail=None):
    g = {"gate": gid, "pass": bool(ok), "detail": str(detail)[:3000]}
    GATES[gid] = g
    log(f"GATE {gid}: {'PASS' if ok else 'FAIL'} | {detail}")
    return bool(g)


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


def budget_left():
    return HARD_BUDGET_S - (time.time() - T_START)


def prepare_matrix(feats: pd.DataFrame):
    names = [c for c in feats.columns
             if not c.startswith(("y_", "r_xau", "sigma", "close_xau"))]
    X = feats[names].astype("float32")
    X = X.ffill().fillna(0.0).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    y = feats["y_ret24_norm"].astype("float32")
    return X, y, names


def group_slices_from_meta(names, meta_path):
    """VSN group slices aligned to the (already filtered) feature order."""
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        ordered = meta["columns"]
        groups = meta["group_slices"]
        name_pos = {c: i for i, c in enumerate(names)}
        slices, start = [], 0
        for s, e, nm in groups:
            members = [c for c in ordered[s:e] if c in name_pos]
            if members:
                slices.append((start, start + len(members), nm))
                start += len(members)
        if abs(start - len(names)) <= 2 and slices:
            # reorder names to the group order for the VSN
            reordered = []
            for s, e, nm in groups:
                reordered.extend([c for c in ordered[s:e] if c in name_pos])
            return slices, reordered
    except Exception as e:
        log(f"  group meta unavailable ({e}); using single-group fallback")
    return [(0, len(names), "all")], list(names)


# ----------------------------------------------------------------------------
# generic neural trainer (M3/M4; M5 uses train_one_fold with the governor)
# ----------------------------------------------------------------------------
def train_generic(model, X, ret, sigma, y_ret, y_dir, fit_pos, es_pos,
                  seq_len=256, batch_size=64, lr=1e-3, max_epochs=40,
                  patience=12, cost_r=5e-5, windows_per_epoch=5000,
                  device="cuda", seed=1337, w_aux=0.15, use_amp=True):
    """Train any model with the dict-output interface {z, ret24, dir_logit,
    meta_logit} on the fused Sharpe loss; early stop on ES net Sharpe."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    crit = RobustNetSharpeLoss(cost_r=cost_r, gamma_cost_scaler=0.5,
                               softmin_tau=0.2, softmin_lambda=0.25,
                               lam_dd=0.10)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rng = np.random.default_rng(seed)

    Xf = X  # (T, F) float32
    retf = np.nan_to_num(ret.values, nan=0.0).astype("float32")
    sigf = np.nan_to_num(sigma.values, nan=1e-4).astype("float32")
    yrf = np.nan_to_num(y_ret.values, nan=0.0).astype("float32")
    ydf = np.nan_to_num(y_dir.values, nan=0.0).astype("float32")
    T = len(Xf)

    def make_window(pos):
        lo = max(0, pos - seq_len + 1)
        x = Xf.iloc[lo:pos + 1].values
        if len(x) < seq_len:
            pad = np.repeat(x[:1], seq_len - len(x), axis=0)
            x = np.vstack([pad, x])
        r = retf[lo:pos + 1]
        r = np.concatenate([np.zeros(seq_len - len(r)), r])
        s = sigf[lo:pos + 1]
        s = np.concatenate([np.full(seq_len - len(s), 1e-4), s])
        return x, r, s, yrf[pos], ydf[pos]

    es_windows = [make_window(int(p)) for p in es_pos[:: max(1, len(es_pos) // 1500)]]
    es_x = torch.as_tensor(np.stack([w[0] for w in es_windows]), device=device)
    es_r = torch.as_tensor(np.stack([w[1] for w in es_windows]), device=device)
    es_s = torch.as_tensor(np.stack([w[2] for w in es_windows]), device=device)

    best_sharpe, best_state, bad = -1e9, None, 0
    hist = {"val_sharpe": []}
    for epoch in range(max_epochs):
        model.train()
        sel = rng.choice(fit_pos, size=min(windows_per_epoch, len(fit_pos)),
                         replace=False)
        for i0 in range(0, len(sel), batch_size):
            batch = [make_window(int(p)) for p in sel[i0: i0 + batch_size]]
            if len(batch) < 8:
                continue
            x = torch.as_tensor(np.stack([b[0] for b in batch]), device=device)
            r = torch.as_tensor(np.stack([b[1] for b in batch]), device=device)
            s = torch.as_tensor(np.stack([b[2] for b in batch]), device=device)
            yr = torch.as_tensor(np.array([b[3] for b in batch]), device=device)
            yd = torch.as_tensor(np.array([b[4] for b in batch]), device=device)
            opt.zero_grad(set_to_none=True)
            amp_on = use_amp and device == "cuda"
            with torch.autocast("cuda", enabled=amp_on):
                out = model(x)
                loss = crit(out["z"], r, s)
                loss = loss + w_aux * aux_losses(out["dir_logit"], out["ret24"],
                                                 out["meta_logit"], yd, yr)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        # ---- validation: ES net Sharpe on final-step positions
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", enabled=use_amp and device == "cuda"):
            zs = torch.cat([model(es_x[i:i + 512])["z"].float()
                            for i in range(0, len(es_x), 512)])
        z_np = zs[:, -1].cpu().numpy()
        r_np = es_r[:, -1].cpu().numpy()
        s_np = es_s[:, -1].cpu().numpy()
        mu = (z_np * (0.15 / math.sqrt(5786) / (s_np + 1e-12)) * r_np).mean()
        # net sharpe via the losses helper on (1, N)
        zt = torch.as_tensor(z_np[None], dtype=torch.float32)
        rt = torch.as_tensor(r_np[None], dtype=torch.float32)
        st = torch.as_tensor(s_np[None], dtype=torch.float32)
        R = net_strategy_return(zt, rt, st, cost_r,
                                0.15 / math.sqrt(5786)).numpy().ravel()
        sharpe = float(R.mean() / (R.std() + 1e-12) * math.sqrt(5786))
        hist["val_sharpe"].append(sharpe)
        if sharpe > best_sharpe + 1e-4:
            best_sharpe, bad = sharpe, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_val_sharpe"] = best_sharpe
    return model, hist


@torch.no_grad()
def predict_z(model, X, positions, seq_len, device="cuda", batch=512):
    """Frozen per-bar z (and per-iterate z for ContinuumNet) on given rows."""
    Xf = X
    outs, outs_iters = [], []
    for i0 in range(0, len(positions), batch):
        batch_pos = positions[i0:i0 + batch]
        wins = []
        for p in batch_pos:
            lo = max(0, int(p) - seq_len + 1)
            x = Xf.iloc[lo:int(p) + 1].values
            if len(x) < seq_len:
                pad = np.repeat(x[:1], seq_len - len(x), axis=0)
                x = np.vstack([pad, x])
            wins.append(x)
        x = torch.as_tensor(np.stack(wins), device=device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model(x)
        outs.append(o["z"].float()[:, -1].cpu().numpy())
    z = np.concatenate(outs)
    return z


@torch.no_grad()
def predict_z_iterates(model, X, positions, seq_len, device="cuda", batch=256):
    """Per-iterate z series (loop-depth analysis) for ContinuumNet."""
    iters = None
    outs = []
    for i0 in range(0, len(positions), batch):
        batch_pos = positions[i0:i0 + batch]
        wins = []
        for p in batch_pos:
            lo = max(0, int(p) - seq_len + 1)
            x = X.iloc[lo:int(p) + 1].values
            if len(x) < seq_len:
                pad = np.repeat(x[:1], seq_len - len(x), axis=0)
                x = np.vstack([pad, x])
            wins.append(x)
        x = torch.as_tensor(np.stack(wins), device=device)
        with torch.autocast("cuda", enabled=device == "cuda"):
            o = model.forward_per_iterate(x)
        outs.append(o["z"].float()[:, -1].cpu().numpy())
        zi = np.stack([t.float()[:, -1].cpu().numpy() for t in o["z_iterates"]], axis=1)
        iters = zi if iters is None else np.vstack([iters, zi])
    return np.concatenate(outs), iters


def main():
    WORK = "/kaggle/working"
    if not os.path.exists(WORK):
        WORK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "k3_out")
        os.makedirs(WORK, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"device={device}; torch={torch.__version__}; "
        f"gpu={torch.cuda.get_device_name(0) if device == 'cuda' else 'none'}")
    ledger = TrialLedger(os.path.join(WORK, "trial_ledger.jsonl"))

    # ---------------- 1) data ------------------------------------------------
    with stage("load data"):
        feats, px = load_v2_data()
        X, y, names = prepare_matrix(feats)
        fdir = find_features_ds()
        slices, ordered = group_slices_from_meta(
            names, os.path.join(fdir, "feature_meta.json"))
        X = X[ordered]          # reorder to VSN group order
        names = ordered
        log(f"matrix {X.shape}; VSN groups: {[(s, e, nm) for s, e, nm in slices]}")
        # matrices for the trainer
        ret = feats["r_xau"]
        sigma = feats["sigma"]
        y_dir = feats["y_dir24"]

    # search-phase splits: train < 2022 (purged), val = 2022
    idx = feats.index
    t2022 = idx.searchsorted(pd.Timestamp("2022-01-01", tz="UTC"))
    t2023 = idx.searchsorted(pd.Timestamp("2023-01-01", tz="UTC"))
    search_train = np.arange(0, t2022 - 24)
    search_val = np.arange(t2022, t2023)
    m_y = y.notna().values
    search_fit = search_train[m_y[search_train]][:: 2]     # subsample search fits
    n_es = max(500, int(len(search_train) * 0.1))
    search_es = search_train[-n_es:]

    F_ = X.shape[1]

    # ---------------- 2) search phase SKIPPED (configs from the compact run) --
    results = {
        "M3c": (None, {"dropout": 0.2, "lr": 0.001}),
        "M3m": (None, {"dropout": 0.2, "lr": 0.001}),
        "M4": (None, {"dropout": 0.2, "lr": 0.001, "d_model": 32}),
        "M5": (None, {"d_model": 64, "lr": 0.0003}),
    }
    with open(os.path.join(WORK, "search_results.json"), "w") as f:
        json.dump({k: {"cfg": v[1]} for k, v in results.items()}, f, indent=1)

    # ---------------- 3) fold refits -----------------------------------------
    with stage("walk-forward fold refits"):
        forecasts = pd.DataFrame(index=feats.index)
        m5_iters = {}
        rho_log = []
        fold_status = {}
        budget_stopped = {}
        for fam, best in results.items():
            if best is None:
                fold_status[fam] = "search failed"
                continue
            cfgd = best[1]
            fam_z = pd.Series(np.nan, index=feats.index)
            # COMPLETION plan: M5 folds 2020-2025 (the budget-stopped gap),
            # M4 all folds (fp32 fix), M3m all folds; M3c already complete
            fam_years = {"M5": [2020, 2021, 2022, 2023, 2024, 2025],
                         "M4": ALL_FOLD_YEARS, "M3m": ALL_FOLD_YEARS,
                         "M3c": []}[fam]
            for yy in fam_years:
                if budget_left() < 900:
                    log(f"  BUDGET STOP before {fam} fold {yy} ({budget_left()/3600:.2f}h left)")
                    fold_status[fam] = f"budget-stopped at fold {yy}"
                    budget_stopped[fam] = yy
                    break
                res = fold_indices(feats, yy)
                if res is None:
                    continue
                fit_pos, es_pos, test_pos = res
                fit_m = fit_pos[m_y[fit_pos]]
                preds = []
                for seed in (1337,):
                    torch.manual_seed(seed)
                    if fam == "M3c":
                        model = JaneStreetDAEResMLP(
                            in_features=F_, d_enc=32, noise_std=0.10,
                            swap_noise=0.15)
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=256, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device=device, seed=seed)
                        zv = predict_z(model, X, test_pos, 256, device)
                    elif fam == "M3m":
                        model = AEMLP(F_, enc_dim=96, noise_sigma=0.035,
                                      dropout=(cfgd["dropout"],) * 4)
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=256, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device=device, seed=seed,
                            use_amp=(torch.cuda.is_available() and fam != "M3m"))
                        zv = predict_z(model, X, test_pos, 256, device)
                    elif fam == "M4":
                        # fp32 (NO autocast): the compact run showed systematic
                        # NaN under fp16 for this architecture; M4 is tiny so
                        # fp32 costs nothing
                        model = OxfordManDMNLite(in_features=F_, d_model=32,
                                                 n_heads=2,
                                                 dropout=cfgd["dropout"])
                        model, _ = train_generic(
                            model, X, ret, sigma, y, y_dir, fit_m[::2], es_pos,
                            seq_len=128, batch_size=64, lr=cfgd["lr"],
                            max_epochs=28, patience=9, device="cpu", seed=seed,
                            use_amp=False)
                        zv = predict_z(model, X, test_pos, 128, "cpu")
                    else:   # M5
                        net_cfg = ContinuumConfig(
                            n_features=F_, group_slices=slices,
                            d_model=cfgd["d_model"], n_heads=4, dropout=0.2,
                            loop_iters=4, km_alpha=0.5, spectral_cap=0.80)
                        tcfg = TrainConfig(seq_len=256, batch_size=32,
                                           lr=cfgd["lr"], max_epochs=28,
                                           patience=9, device=device, seed=seed)
                        model, hist = train_one_fold(
                            X.values.astype("float32"),
                            np.nan_to_num(ret.values), np.nan_to_num(sigma.values),
                            np.nan_to_num(y.values), np.nan_to_num(y_dir.values),
                            fit_m[::2], es_pos, slices, net_cfg, tcfg)
                        zv, zi = predict_z_iterates(model, X, test_pos, 256, device)
                        if seed == 1337:
                            m5_iters[yy] = zi
                        rho = audit_loop_spectral_radius(
                            model, torch.as_tensor(
                                X.values[es_pos[-260:]].astype("float32"),
                                device=device))
                        rho_log.append({"fold": yy, "seed": seed, "rho": round(rho, 4),
                                        "km_alpha": net_cfg.km_alpha,
                                        "governor": "auto_tune_alpha"})
                        ledger.add("M5_ContinuumNet",
                                   {"fold": yy, "seed": seed,
                                    "d_model": cfgd["d_model"]},
                                   hist.get("best_val_sharpe"),
                                   stage="fold", kernel="K3",
                                   notes=f"rho={rho:.3f}")
                    preds.append(zv)
                    del model
                    torch.cuda.empty_cache()
                fam_z.iloc[test_pos] = np.mean(preds, axis=0)
                log(f"  {fam} fold {yy}: predicted {len(test_pos)} bars "
                    f"(z std {np.std(np.concatenate(preds)):.3f})")
            forecasts[f"z_{fam}"] = fam_z
            if fam not in budget_stopped:
                fold_status[fam] = "complete"

        # per-iterate loop-depth analysis export (M5)
        if m5_iters:
            it_mat = np.concatenate([m5_iters[y] for y in sorted(m5_iters)], axis=0)
            it_idx = feats.index[np.concatenate(
                [np.arange(feats.index.searchsorted(pd.Timestamp(f"{y}-01-01", tz="UTC")),
                           feats.index.searchsorted(pd.Timestamp(f"{y+1}-01-01", tz="UTC")))
                 for y in sorted(m5_iters)])]
            it_df = pd.DataFrame(it_mat, index=it_idx,
                                 columns=[f"z_iter{i+1}" for i in range(it_mat.shape[1])])
            forecasts = forecasts.join(it_df)

        ok_folds = gate("G3.4_folds", all(v == "complete" for v in fold_status.values()),
                        f"fold status: {fold_status}; budget used "
                        f"{(time.time()-T_START)/3600:.2f}h")

    # ---------------- 4) spectral gate + export -------------------------------
    with stage("spectral gate + export"):
        ok_rho = gate("G3.3_spectral", all(r["rho"] < 0.95 for r in rho_log)
                      if rho_log else True,
                      f"rho at every audited M5 checkpoint < 0.95: "
                      f"{rho_log[:6]}..." if rho_log else "no M5 folds trained")
        # merge with the compact-run forecasts (fill-forward completion)
        prior_dir = _find_dir("v2_neural_forecasts.parquet",
                              "/kaggle/input/aurum-v2-forecasts-neural",
                              *sorted(__import__("glob").glob("/kaggle/input/*/aurum-v2-forecasts*")))
        if prior_dir:
            prior = pd.read_parquet(os.path.join(prior_dir, "v2_neural_forecasts.parquet"))
            prior.index = pd.DatetimeIndex(prior.index)
            if prior.index.tz is None:
                prior.index = prior.index.tz_localize("UTC")
            for c in prior.columns:
                if c in ("r_xau", "sigma", "y_ret24_norm"):
                    continue
                if c not in forecasts.columns:
                    forecasts[c] = np.nan
                fill = forecasts[c].isna() & prior[c].notna()
                forecasts.loc[fill, c] = prior.loc[fill, c]
            log(f"merged prior compact forecasts: {int(forecasts.notna().sum().sum())} "
                f"filled cells")
        forecasts["r_xau"] = ret
        forecasts["sigma"] = sigma
        forecasts["y_ret24_norm"] = feats["y_ret24_norm"]
        forecasts.to_parquet(os.path.join(WORK, "v2_neural_forecasts.parquet"))
        if prior_dir and os.path.isfile(os.path.join(prior_dir, "rho_log.json")):
            with open(os.path.join(prior_dir, "rho_log.json")) as f:
                rho_log = json.load(f) + rho_log
        with open(os.path.join(WORK, "rho_log.json"), "w") as f:
            json.dump(rho_log, f, indent=1)
        with open(os.path.join(WORK, "k3_meta.json"), "w") as f:
            json.dump({"fold_status": fold_status,
                       "search": {k: (None if v is None else v[1])
                                  for k, v in results.items()},
                       "ledger_entries": len(ledger.rows),
                       "wall_h": round((time.time() - T_START) / 3600, 2)},
                      f, indent=1, default=str)
        ok_ex = gate("G3.5_export", True,
                     f"v2_neural_forecasts.parquet ({len(forecasts.columns)} cols); "
                     f"rho_log.json; trial_ledger.jsonl ({len(ledger.rows)} entries)")

    all_ok = all([ok_folds, ok_rho, ok_ex])
    gate("G3_ALL", all_ok, "all K3 gates green" if all_ok else "K3 gate failure")
    with open(os.path.join(WORK, "gates.json"), "w") as f:
        json.dump(GATES, f, indent=2, default=str)
    with open(os.path.join(WORK, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))
    log("K3 COMPLETE" if all_ok else "K3 FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    import sys as _s
    _s.exit(main())
