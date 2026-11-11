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
from continuum import ContinuumConfig, ContinuumNet, auto_tune_alpha
from losses import (RobustNetSharpeLoss, NetSharpeLoss, aux_losses,
                    net_strategy_return, two_pass_exact_sharpe_step)



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
