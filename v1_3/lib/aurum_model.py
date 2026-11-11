# ============================================================================
# AURUM LoopedQwen3 (Spec v1.1, frozen):
#   - Base: Qwen/Qwen3-0.6B (28 decoder layers, d=1024).
#   - Looped core: decoder layers 13..16 (inclusive) re-executed R times
#     (R in {1,2,3}); shared weights, per-iteration Sandwich RMSNorm with
#     identity init (audit fix #5: full unfreeze of the ~60M core params
#     instead of rank-16 LoRA; T4 16GB holds it).
#   - Sandwich design (per iteration r):
#         z = pre_r(h);  z = core(z);  h = clamp(post_r(z))
#     with pre/post = x + tanh(g)*(RMSNorm(x) - x), g=0 at init, so
#     R=1 is EXACTLY the pretrained model at initialization (verified by
#     the r=1 equivalence self-test).
#   - fp16 guardrails (user directive): torch.amp autocast(fp16) + GradScaler
#     everywhere; RMSNorm accumulates in fp32; pre-norm residual norms are
#     clamped (C=100) so the looped stream can never overflow fp16 on sm_75.
#   - Joint Multi-Task World-Model Head: 5 learned readout tokens appended at
#     the end of the state sequence (guardrail: no variable-option permutation
#     masks). Single forward pass -> 85 outputs:
#       H1_BARRIER 48 (per-cell P(win), sigmoid)
#       H2_MTM      8 (multi-horizon direction & magnitude, sigmoid)
#       H3_EXCUR   12 (MFE/MAE q25/q50/q75 in ATR units, softplus)
#       H4_VOL      8 (expansion probs + terciles + q50)
#       H5_REGIME   9 (trend x vol classes, softmax)
# ============================================================================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.checkpoint import checkpoint as tk_ckpt

from aurum_gates import HEAD_DIMS, READOUT_NAMES  # noqa

BASE_MODEL_ID = "Qwen/Qwen3-0.6B"
CORE_START, CORE_END = 13, 17          # decoder layers 13..16 inclusive
N_READOUT = 5
# Qwen3-0.6B residual-stream L2 norms reach ~7.5k on real activations (measured:
# mean ~200-700, max 7464). The clamp is a runaway guard for the looped core,
# NOT a normal-path operation: C=16384 sits 2x above the observed stock max and
# 4x below fp16 overflow (65504), so it never fires on healthy activations.
RESID_CLAMP = 16384.0

# ---------------------------------------------------------------------------


class Fp32RMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation (weight kept fp32)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        v = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.weight.float()).to(dt)


class SandwichNorm(nn.Module):
    """x + tanh(g) * (RMSNorm(x) - x); g=0 -> exact identity at init."""

    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.zeros(1))
        self.n = Fp32RMSNorm(dim)

    def forward(self, x):
        dt = x.dtype
        x32 = x.float()
        n32 = self.n(x).float()
        out = x32 + torch.tanh(self.g).float() * (n32 - x32)
        return out.to(dt)


def clamp_resid(x, C=RESID_CLAMP, stats=None):
    """Clamp the L2 norm of the residual stream (fp32 math, per token)."""
    n = x.float().norm(dim=-1, keepdim=True)
    s = torch.clamp(C / (n + 1e-6), max=1.0)
    if stats is not None and torch.is_grad_enabled() and n.numel() > 0:
        with torch.no_grad():
            nm = n.max()
            stats["resid_max"] = max(stats.get("resid_max", 0.0), float(nm))
            stats["clamp_count"] = stats.get("clamp_count", 0.0) + float((n > C).sum())
            stats["clamp_total"] = stats.get("clamp_total", 0.0) + float(n.numel())
    return (x.float() * s).to(x.dtype)


def build_causal_mask(T, dtype, device):
    m = torch.full((1, 1, T, T), torch.finfo(dtype).min, dtype=torch.float32, device=device)
    m = torch.triu(m, diagonal=1)
    return m.to(dtype)


def build_joint_mask(pad_mask, T, L, dtype, device):
    """Causal + key-padding mask. pad_mask: (B, L) True=real token.
    Readout positions (L..T-1) are always valid keys."""
    B = pad_mask.size(0)
    key_ok = torch.ones(B, T, dtype=torch.bool, device=device)
    key_ok[:, :L] = pad_mask
    neg = torch.finfo(dtype).min
    m = torch.zeros(B, 1, T, T, dtype=torch.float32, device=device)
    m.masked_fill_(~key_ok[:, None, None, :], neg)
    causal = torch.triu(torch.full((T, T), neg, dtype=torch.float32, device=device), diagonal=1)
    m = m + causal[None, None]
    return m.clamp(min=neg).to(dtype)


def call_layer(layer, h, pe, pos_ids, attn_mask):
    kw = dict(attention_mask=attn_mask, position_ids=pos_ids, use_cache=False,
              output_attentions=False)
    if pe is not None:
        kw["position_embeddings"] = pe
    try:
        out = layer(h, **kw)
    except TypeError:
        kw.pop("position_embeddings", None)
        kw.pop("position_ids", None)
        out = layer(h, attention_mask=attn_mask)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def _mlp_head(d, hidden, out):
    return nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out))


class JointHead(nn.Module):
    def __init__(self, d_model, hidden=256):
        super().__init__()
        self.readout = nn.Embedding(N_READOUT, d_model)
        nn.init.normal_(self.readout.weight, std=0.02)
        self.h1 = _mlp_head(d_model, hidden, HEAD_DIMS["h1"])   # 48 sigmoid
        self.h2 = _mlp_head(d_model, hidden, HEAD_DIMS["h2"])   # 8 sigmoid
        self.h3 = _mlp_head(d_model, hidden, HEAD_DIMS["h3"])   # 12 softplus
        self.h4 = _mlp_head(d_model, hidden, HEAD_DIMS["h4"])   # 8 sigmoid(+)
        self.h5 = _mlp_head(d_model, hidden, HEAD_DIMS["h5"])   # 9 softmax

    def forward(self, ro_hidden):
        return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                "h5": self.h5(ro_hidden[:, 4])}

    @torch.no_grad()
    def predict(self, logits):
        """Apply the frozen activations -> dict of probabilities/values.
        h4 layout: [0:4] expansion probs (sigmoid), [4:7] vol tercile (softmax),
        [7] fwd12/48 vol-ratio q50 (softplus)."""
        h4 = torch.cat([torch.sigmoid(logits["h4"][:, :4]),
                        torch.softmax(logits["h4"][:, 4:7], dim=-1),
                        Fn.softplus(logits["h4"][:, 7:8])], dim=1)
        out = {"h1": torch.sigmoid(logits["h1"]), "h2": torch.sigmoid(logits["h2"]),
               "h3": Fn.softplus(logits["h3"]), "h4": h4,
               "h5": torch.softmax(logits["h5"], dim=-1)}
        return out


class LoopedQwen(nn.Module):
    def __init__(self, base, n_loop=3, sandwich=True, clamp=RESID_CLAMP,
                 grad_ckpt=False, explicit_mask=False):
        super().__init__()
        self.base = base                       # AutoModelForCausalLM
        self.n_loop = n_loop
        self.sandwich = sandwich
        self.clamp = clamp
        self.grad_ckpt = grad_ckpt
        self.explicit_mask = explicit_mask
        d = base.config.hidden_size
        self.pre_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.post_norms = nn.ModuleList([SandwichNorm(d) for _ in range(n_loop)])
        self.head = JointHead(d)
        self.stats = {}

    # ---------------- parameter management ----------------
    def set_trainable(self, train_lm_head=False):
        for p in self.base.parameters():
            p.requires_grad = False
        for i in range(CORE_START, CORE_END):
            for p in self.base.model.layers[i].parameters():
                p.requires_grad = True
        if train_lm_head and not self.base.config.tie_word_embeddings:
            for p in self.base.lm_head.parameters():
                p.requires_grad = True
        for p in self.pre_norms.parameters():
            p.requires_grad = True
        for p in self.post_norms.parameters():
            p.requires_grad = True
        for p in self.head.parameters():
            p.requires_grad = True

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------- core forward ----------------
    def _run_layer(self, layer, h, pe, pos_ids, mask):
        if self.grad_ckpt and self.training:
            return tk_ckpt(lambda hh: call_layer(layer, hh, pe, pos_ids, mask), h,
                           use_reentrant=False)
        return call_layer(layer, h, pe, pos_ids, mask)

    def forward(self, ids, add_readout=True, lm_output=True, pad_mask=None):
        m = self.base.model
        B, L = ids.shape
        te = m.embed_tokens(ids)
        if add_readout:
            ro = self.head.readout.weight.unsqueeze(0).expand(B, -1, -1).to(te.dtype)
            h = torch.cat([te, ro], dim=1)
            T = L + N_READOUT
        else:
            h = te
            T = L
        pos_ids = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        pe = None
        if hasattr(m, "rotary_emb"):
            pe = m.rotary_emb(h.float(), pos_ids)
        mask = None
        if pad_mask is not None:
            mask = build_joint_mask(pad_mask, T, L, h.dtype, h.device)
        elif self.explicit_mask:
            mask = build_causal_mask(T, h.dtype, h.device)

        def core_pass(hh):
            for i, layer in enumerate(m.layers):
                if i == CORE_START:
                    # In-place recursion: h_{r+1} = clamp(post_r(core(pre_r(h_r)))).
                    # With identity-init norms (g=0) and R=1 this is EXACTLY the
                    # pretrained forward (verified by the r=1 equivalence test).
                    for r in range(self.n_loop):
                        if self.sandwich:
                            z = self.pre_norms[r](hh)
                            for j in range(CORE_START, CORE_END):
                                z = self._run_layer(m.layers[j], z, pe, pos_ids, mask)
                            z = self.post_norms[r](z)
                            hh = clamp_resid(z, self.clamp, self.stats)
                        else:
                            for j in range(CORE_START, CORE_END):
                                hh = self._run_layer(m.layers[j], hh, pe, pos_ids, mask)
                    continue
                if CORE_START < i < CORE_END:
                    continue
                hh = self._run_layer(layer, hh, pe, pos_ids, mask)
            return hh

        if self.grad_ckpt and self.training:
            h = tk_ckpt(core_pass, h, use_reentrant=False)
        else:
            h = core_pass(h)
        h = m.norm(h)

        out = {}
        if lm_output:
            out["logits"] = self.base.lm_head(h[:, :L] if add_readout else h)
        if add_readout:
            out["head"] = self.head(h[:, -N_READOUT:, :])
        return out

    # ---------------- losses ----------------
    @staticmethod
    def lm_loss(logits, ids, pad_mask=None, chunk_rows=2048):
        """Memory-safe chunked CE. The 151,936-vocab logits for (B=16, T=512)
        are ~2.5GB in fp16; a single fp32 cast + CE would add ~7.5GB and OOM a
        T4. Each chunk's fp32 cast + CE is recomputed in backward via
        checkpoint, so peak fp32 memory is one chunk (~1.2GB)."""
        tgt = ids[:, 1:].clone()
        if pad_mask is not None:
            tgt[~pad_mask[:, 1:]] = -100
        lg = logits[:, :-1].reshape(-1, logits.size(-1))
        tg = tgt.reshape(-1)
        n_rows = lg.shape[0]
        total = torch.zeros((), device=lg.device, dtype=torch.float32)
        n = 0
        for s in range(0, n_rows, chunk_rows):
            sl = slice(s, min(s + chunk_rows, n_rows))
            t = tg[sl]
            cnt = int((t != -100).sum())
            if cnt == 0:
                continue
            def _ce(l, tt):
                return Fn.cross_entropy(l.float(), tt, ignore_index=-100,
                                        reduction="sum")
            total = total + tk_ckpt(_ce, lg[sl], t, use_reentrant=False)
            n += cnt
        return total / max(n, 1)

    @staticmethod
    def head_loss(head_logits, tgt):
        """tgt: dict of tensors on the same device.
        h1/h2: (B,48|8) in [0,1]; h3: (B,12) values; h4: (B,7) probs + h4c (B,) long,
        h4q (B,) value; h5: (B,) long 0..8. NaNs allowed -> masked out."""
        losses = []
        for k in ("h1", "h2"):
            if tgt.get(k) is not None:
                t = tgt[k].float()
                lg = head_logits[k]
                msk = torch.isfinite(t)
                if msk.any():
                    losses.append(Fn.binary_cross_entropy_with_logits(
                        lg[msk], t[msk]))
        if tgt.get("h3") is not None:
            t = tgt["h3"].float()
            msk = torch.isfinite(t) & (t.abs() < 1e4)
            if msk.any():
                losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h3"])[msk], t[msk]))
        if tgt.get("h4") is not None:
            t = tgt["h4"].float()
            msk = torch.isfinite(t)
            if msk.any():
                losses.append(Fn.binary_cross_entropy_with_logits(
                    head_logits["h4"][:, :4][msk], t[msk]))
        if tgt.get("h4c") is not None:
            t = tgt["h4c"].long()
            ok = (t >= 0) & (t < 3)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h4"][:, 4:7][ok], t[ok]))
        if tgt.get("h4q") is not None:
            t = tgt["h4q"].float()
            msk = torch.isfinite(t) & (t > 0) & (t < 1e4)
            if msk.any():
                losses.append(Fn.smooth_l1_loss(Fn.softplus(head_logits["h4"][:, 7])[msk], t[msk]))
        if tgt.get("h5") is not None:
            t = tgt["h5"].long()
            ok = (t >= 0) & (t < 9)
            if ok.any():
                losses.append(Fn.cross_entropy(head_logits["h5"][ok], t[ok]))
        if not losses:
            return torch.tensor(0.0, device=head_logits["h1"].device)
        return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Training utilities (fp16-strict, T4/sm_75 safe)
# ---------------------------------------------------------------------------

def make_optimizer(model, lr=1e-4, wd=0.01, betas=(0.9, 0.95)):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas)


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def load_base(device=None, dtype=torch.float32, log=print):
    from transformers import AutoModelForCausalLM
    try:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
    if device is not None:
        base = base.to(device)
    base.eval()
    log(f"  loaded {BASE_MODEL_ID} ({sum(p.numel() for p in base.parameters())/1e6:.1f}M params, "
        f"{len(base.model.layers)} layers, d={base.config.hidden_size}, "
        f"tied={base.config.tie_word_embeddings})")
    return base


def load_tokenizer(log=print):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    log(f"  tokenizer vocab={tok.vocab_size}")
    return tok
