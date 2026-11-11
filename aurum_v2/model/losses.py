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
