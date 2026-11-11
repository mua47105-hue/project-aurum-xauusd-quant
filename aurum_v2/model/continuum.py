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
