#!/usr/bin/env python3
"""AURUM audit report — all quantitative claims computed here.
Qwen3-0.6B architecture math, VRAM budgets, inference FLOPs/time,
walk-forward training budget, barrier-grid sizes, Kelly examples,
and the hindsight-argmax bias simulation (Figure 1 data).
Printed results are transcribed into the report text."""
import numpy as np

print("=" * 72)
print("1. QWEN3-0.6B LAYER ARITHMETIC (hidden=1024, 28 layers, GQA 16Q/8KV, head_dim=128, SwiGLU intermediate=3072)")
q = 1024 * (16 * 128)      # q_proj: 1024 -> 2048
k = 1024 * (8 * 128)       # k_proj: 1024 -> 1024
v = 1024 * (8 * 128)       # v_proj
o = (16 * 128) * 1024      # o_proj: 2048 -> 1024
gate = 1024 * 3072
up = 1024 * 3072
down = 3072 * 1024
per_layer = q + k + v + o + gate + up + down
print(f"   per-layer matmul params: q={q:,} k={k:,} v={v:,} o={o:,} gate={gate:,} up={up:,} down={down:,}")
print(f"   per-layer total = {per_layer:,} (~{per_layer/1e6:.2f}M)")
core4 = 4 * per_layer
trunk20 = 20 * per_layer
embed = 151_936 * 1024
print(f"   core layers 13-16 = {core4:,} (~{core4/1e6:.1f}M)   trunk 20 layers = {trunk20:,} (~{trunk20/1e6:.1f}M)")
print(f"   tied embeddings (vocab 151,936 x 1024) = {embed:,} (~{embed/1e6:.1f}M)")

print("=" * 72)
print("2. NEW TRAINABLE PARAMETER BUDGET (realignment recipe)")
lora_r = 8
projs = {"q": (1024, 2048), "k": (1024, 1024), "v": (1024, 1024), "o": (2048, 1024),
         "gate": (1024, 3072), "up": (1024, 3072), "down": (3072, 1024)}
lora_per_layer_iter = sum(lora_r * (a + b) for a, b in projs.values())
lora_iters = 3  # iterations r=2,3,4 (r=1 uses base weights)
lora_total = lora_per_layer_iter * 4 * lora_iters
sandwich = 2 * 1024 * 4            # pre+post RMSNorm weight per iteration x 4 iterations
ws = 1024 * 1024                   # feedback projection W_s
lam = 3                            # loop gates (scalars)
head = 9_000_000                   # scoring heads (barrier + quantile + aux), ~9M
trainable = core4 + lora_total + sandwich + ws + lam + head
print(f"   unfrozen core (13-16):            {core4/1e6:6.1f}M")
print(f"   per-iteration LoRA r=8 (r=2..4):  {lora_total/1e6:6.2f}M  ({lora_per_layer_iter:,}/layer/iter)")
print(f"   sandwich RMSNorms (4 iters):      {sandwich:,}")
print(f"   feedback W_s + gates:             {(ws+lam)/1e6:6.2f}M")
print(f"   scoring heads:                    {head/1e6:6.1f}M")
print(f"   TOTAL TRAINABLE = {trainable/1e6:.1f}M")

print("=" * 72)
print("3. VRAM BUDGET ON ONE T4 (15.4 GB usable)")
frozen_trunk = (trunk20 - core4) * 2 / 1e9        # fp16 bytes
frozen_embed = embed * 2 / 1e9
train_fp16 = trainable * 2 / 1e9
optim_states = trainable * 12 / 1e9               # fp32 master + AdamW m,v
grads_fp16 = trainable * 2 / 1e9
B, L = 16, 600
acts = B * L * 1024 * 28 * 2 * 3 / 1e9            # rough checkpointed activations (~x3 recompute factor, 28 layer-passes)
print(f"   frozen trunk fp16:        {frozen_trunk:.2f} GB")
print(f"   frozen embeddings fp16:   {frozen_embed:.2f} GB")
print(f"   trainable fp16 copy:      {train_fp16:.2f} GB")
print(f"   fp32 master + AdamW m,v:  {optim_states:.2f} GB")
print(f"   fp16 grads:               {grads_fp16:.2f} GB")
print(f"   activations (B={B}, L={L}, ckpt, ~28 layer-passes): {acts:.2f} GB")
peak = frozen_trunk + frozen_embed + train_fp16 + optim_states + grads_fp16 + acts
print(f"   PEAK ESTIMATE ~ {peak:.1f} GB  vs 15.4 GB T4 -> {'FITS' if peak < 13 else 'TOO BIG'} (both-T4 DDP splits further)")

print("=" * 72)
print("4. INFERENCE COMPUTE (30,000 H1 bars, ~600 tokens/state)")
tokens = 30_000 * 600
print(f"   tokens = {tokens/1e6:.0f}M")
flops = lambda lp: 2 * per_layer * lp * tokens          # FLOPs for a full pass
for r, lp in [(1, 20), (2, 24), (3, 28)]:
    pf = flops(lp) / 1e15                               # PFLOP
    gf = 2 * (per_layer * lp) / 1e9
    t_lo, t_hi = flops(lp) / 23e12 / 60, flops(lp) / 13e12 / 60   # minutes
    print(f"   r={r}: {lp} layer-passes, {gf:.2f} GFLOP/token, total {pf:.1f} PFLOP -> {t_lo:.0f}-{t_hi:.0f} min at 13-23 TFLOPS eff.")
mgmt = 0.30  # in-position bars share
print(f"   + management pass (~{mgmt:.0%} of bars) at r=3: +{flops(28)*mgmt/1e15:.1f} PFLOP")
tot_pf = (flops(20) + flops(24) + flops(28)) * (1 + mgmt) / 1e15
print(f"   ALL-IN (r=1,2,3 x entry+management): ~{tot_pf:.0f} PFLOP")
for tf in (6.5, 13, 23):
    print(f"     at {tf:>4} TFLOPS effective: {tot_pf*1e15/(tf*1e12)/3600:.2f} h")
print(f"   naive CPU alternative: 180,000 passes x 7-40 s = {180000*7/86400:.0f}-{180000*40/86400:.0f} days")

print("=" * 72)
print("5. TRAINING THROUGHPUT AND WALK-FORWARD BUDGET")
tok_s_single = (1500, 4000)
ddp = (1.8, 2.4)
for label, M in [("Stage A' CPT 150M", 150e6), ("Stage A' CPT 300M", 300e6),
                 ("Stage B SFT 144M (2 epochs x 72M)", 144e6),
                 ("warm-start refit 36M (0.5 epoch)", 36e6)]:
    lo = M / (tok_s_single[1] * ddp[1]) / 3600
    hi = M / (tok_s_single[0] * ddp[0]) / 3600
    print(f"   {label}: {lo:.1f}-{hi:.1f} h on 2xT4")
anchor = (144e6 / (tok_s_single[1] * ddp[1]) / 3600, 144e6 / (tok_s_single[0] * ddp[0]) / 3600)
refit = (36e6 / (tok_s_single[1] * ddp[1]) / 3600, 36e6 / (tok_s_single[0] * ddp[0]) / 3600)
inference = (1.0, 3.0)
ablation = (4.0, 8.0)
week = (anchor[0] + 3*refit[0] + inference[0] + ablation[0], anchor[1] + 3*refit[1] + inference[1] + ablation[1])
print(f"   ONE walk-forward campaign (anchor + 3 refits + tables + ablations): {week[0]:.0f}-{week[1]:.0f} GPU-h vs 30 h/week quota")
cpcv = (6*anchor[0] + inference[0], 6*anchor[1] + inference[1])
print(f"   full CPCV K=6 refits: {cpcv[0]:.0f}-{cpcv[1]:.0f} GPU-h -> NOT feasible in one week")

print("=" * 72)
print("6. BARRIER GRID AND HEAD OUTPUT COUNTS")
stops, ratios = [1.0, 1.5, 2.0, 2.5], [1.5, 2.5]
cells = len(stops) * len(ratios)
print(f"   cells = {len(stops)} stops x {len(ratios)} reward ratios = {cells}; outcomes 3 (win/loss/timeout); directions 2")
print(f"   barrier head logits = {cells*3*2}")
mfe = 2 * 3 * 2 * 2   # horizons {24,120} x quantiles {q25,q50,q75} x (MFE,MAE) x 2 directions
print(f"   MFE/MAE quantile outputs = {mfe}; timeout-MTM quantiles = {3*2}; vol = 3; regime aux = 4")
print(f"   total head outputs ~ {cells*3*2 + mfe + 6 + 3 + 4}")

print("=" * 72)
print("7. COSTS IN ATR UNITS AND KELLY EXAMPLES")
for atr in (10.0, 20.0):
    cost = 0.35 / atr
    swap3d = 1.00 / atr
    print(f"   H1 ATR ${atr:.0f}: round-turn $0.35 = {cost:.3f} ATR; 3-day swap ~$1.00 = {swap3d:.3f} ATR")
for p, R in [(0.55, 1.5), (0.50, 2.5), (0.60, 1.5)]:
    f = p - (1 - p) / R
    print(f"   Kelly p={p}, R={R}: f* = {f:.3f}; quarter-Kelly {f/4:.3%} -> capped at 1.0% risk")

print("=" * 72)
print("8. FIGURE 1 DATA — HINDSIGHT-ARGMAX vs EXPECTED-VALUE OPTIMUM (simulation)")
rng = np.random.default_rng(42)
N, H = 20000, 48
sigma_bar, mu_bar = 0.35, 0.012          # per-bar std and drift in ATR units
rets = rng.normal(mu_bar, sigma_bar, size=(N, H))
paths = np.cumsum(rets, axis=1)           # ATR-normalized price excursion paths
stops = np.array([1.0, 1.5, 2.0, 2.5]); R = 1.5
pnl = np.zeros((N, len(stops)))
for j, m in enumerate(stops):
    tp, sl = R * m, -m
    hit_up = paths >= tp
    hit_dn = paths <= sl
    first_up = np.where(hit_up.any(1), hit_up.argmax(1), H + 1)
    first_dn = np.where(hit_dn.any(1), hit_dn.argmax(1), H + 1)
    win = first_up < first_dn
    loss = first_dn < first_up
    timeout = ~(win | loss)
    mtm = np.zeros(N)
    done_idx = np.minimum(first_up, first_dn)
    idx = np.clip(done_idx - 1, 0, H - 1)
    mtm[timeout] = paths[timeout, -1]
    pnl[:, j] = R * m * win - m * loss + mtm * timeout
hindsight = stops[np.argmax(pnl, axis=1)]
ev = pnl.mean(axis=0)
ev_best = int(np.argmax(ev))
counts = {m: int((hindsight == m).sum()) for m in stops}
print(f"   E[P&L] per stop width: " + ", ".join(f"{m:.1f}xATR -> {ev[j]:+.3f}R" for j, m in enumerate(stops)))
print(f"   EV-optimal stop = {stops[ev_best]:.1f}xATR (population answer, stable)")
print(f"   hindsight-argmax label distribution over {N} identical-state paths:")
for m in stops:
    print(f"     {m:.1f}xATR: {counts[m]:6d} ({counts[m]/N:.1%})")
print(f"   paths whose hindsight label equals EV-optimal: {counts[stops[ev_best]]/N:.1%}")
print(f"   -> labels scatter across all 4 widths although every path shares the SAME state")
np.save("/home/z/my-project/scripts/aurum_audit/fig1_pnl_mean.npy", ev)
np.save("/home/z/my-project/scripts/aurum_audit/fig1_hist.npy", np.array([counts[m] for m in stops]))
print("   saved fig1 data")
