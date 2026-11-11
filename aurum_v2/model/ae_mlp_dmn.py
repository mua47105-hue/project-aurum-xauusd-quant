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
from losses import NetSharpeLoss, EPS_SHARPE



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
