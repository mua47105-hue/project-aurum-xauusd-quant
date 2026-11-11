#!/usr/bin/env python3
"""Smoke-test the p3_tables plumbing with a tiny synthetic Qwen3 (no download,
fits in sandbox RAM). Validates: build_core norm expansion (r=4 reuses last
trained norm), forward ro_h shapes, depth sensitivity, head stack + activation
layout, neutralization, EV fields."""
import importlib.util, types, sys
import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    "p3k", "/home/z/my-project/kaggle_work/aurum/dist/aurum-p3-tables.py")
m = importlib.util.module_from_spec(SPEC)
# shrink the loop window for the tiny model BEFORE the module code uses it.
# The constants are read at call time (globals) so patching post-import works.
SPEC.loader.exec_module(m)

# tiny Qwen3 config
from transformers import Qwen3Config, AutoModelForCausalLM
cfg = Qwen3Config(vocab_size=512, hidden_size=64, intermediate_size=128,
                  num_hidden_layers=18, num_attention_heads=4,
                  num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True)
torch.manual_seed(0)
base = AutoModelForCausalLM.from_config(cfg)
base.eval()

# patch loop window to layers 2..4 (3 layers) so the 18-layer tiny model works
m.CORE_START, m.CORE_END = 2, 5

# fake a "trained" core_sd in the export format
model3 = m.LoopedQwen(base, n_loop=3)
core_sd = {"layers": {f"{i}": {k: v.half() for k, v in
                               model3.base.model.layers[i].state_dict().items()}
                      for i in range(m.CORE_START, m.CORE_END)},
           "pre": {k: v.half() for k, v in model3.pre_norms.state_dict().items()},
           "post": {k: v.half() for k, v in model3.post_norms.state_dict().items()},
           "readout": {"readout.weight": torch.randn(5, 64).half()},
           "keep_ids": np.arange(97)}
# perturb the trained norms away from identity so reuse is detectable
with torch.no_grad():
    for r in range(3):
        model3.pre_norms[r].g.fill_(0.7)
        model3.post_norms[r].g.fill_(-0.4)
core_sd["pre"] = {k: v.half() for k, v in model3.pre_norms.state_dict().items()}
core_sd["post"] = {k: v.half() for k, v in model3.post_norms.state_dict().items()}

ro_w = core_sd["readout"]["readout.weight"].float()

def fwd(depth, ids, pm):
    mdl = m.build_core(base, core_sd, depth, log=lambda s: None)
    with torch.no_grad():
        o = mdl(ids, add_readout=True, lm_output=False, pad_mask=pm,
                ro_embed=ro_w)
    return o["ro_h"]

torch.manual_seed(1)
B, L = 6, 40
ids = torch.randint(0, 512, (B, L))
ids[:, 30:] = 0
pm = torch.ones(B, L, dtype=torch.bool); pm[:, 30:] = False

h1, h2, h3, h4 = fwd(1, ids, pm), fwd(2, ids, pm), fwd(3, ids, pm), fwd(4, ids, pm)
assert h1.shape == (B, 5, 64) and h3.shape == (B, 5, 64), (h1.shape, h3.shape)
d12 = (h1 - h2).abs().mean().item(); d23 = (h2 - h3).abs().mean().item()
d34 = (h3 - h4).abs().mean().item()
print(f"depth sensitivity |h_r - h_r'|: 1v2={d12:.4f} 2v3={d23:.4f} 3v4={d34:.4f}")
assert d12 > 1e-4 and d23 > 1e-4 and d34 > 1e-4, "depths must differ"

# norm reuse check: r=4 iteration 4 norm == trained norm 2
mdl4 = m.build_core(base, core_sd, 4, log=lambda s: None)
g3 = mdl4.pre_norms[3].g.item(); g2 = mdl4.pre_norms[2].g.item()
assert abs(g3 - g2) < 1e-6, (g3, g2)
print(f"r=4 extrapolation norm reuse OK (g[3]={g3:.3f} == g[2]={g2:.3f})")

# row-content invariance: a row's output must depend only on its own tokens
# and the batch max length L (readout RoPE positions = L..L+4 by design),
# NOT on the content of other rows in the batch. (Full pad-invariance does
# NOT hold in this architecture: readout positions shift with L. p3 replicates
# the exact cache-stage batching - rank::world shard, bs=32, same order - so
# r=3 recomputation matches the training cache composition.)
idsA = ids.clone(); pmA = pm.clone()                      # batch: [6 rows, L=40]
idsB = ids.clone(); pmB = pm.clone()
idsB[1] = torch.randint(0, 512, (L,)); pmB[1, :] = True   # change row 1 content
h3a, h3b = fwd(3, idsA, pmA), fwd(3, idsB, pmB)
drift_others = (h3a[0] - h3b[0]).abs().max().item()       # row 0 unchanged
drift_self = (h3a[1] - h3b[1]).abs().max().item()
print(f"row0 unaffected by other rows' content: {drift_others:.2e} "
      f"(changed row1 drift {drift_self:.3f})")
assert drift_others < 1e-5, "row outputs must not depend on other rows"
assert drift_self > 1e-3, "changed row content must change its own output"
# same row, same L, padded keys masked:
idsC = idsA.clone(); pmC = pmA.clone()
idsC[0, 30:] = 500                                         # pad token value irrelevant
h3c = fwd(3, idsC, pmC)
drift_pad = (h3a[0] - h3c[0]).abs().max().item()
assert drift_pad < 1e-5, "masked pad values must not affect outputs"
print(f"masked-pad-value invariance: {drift_pad:.2e} OK")

# head stack + activation
st = m._head_stack(64)
x = torch.randn(11, 5, 64)
A = m._head_activate(st(x), 1.0, 1.5)
shapes = {k: tuple(v.shape) for k, v in A.items()}
print("head outputs:", shapes)
assert shapes == {"h1": (11, 48), "h2": (11, 8), "h3": (11, 12),
                  "h4": (11, 8), "h5": (11, 9)}, shapes
assert (A["h1"] >= 0).all() and (A["h1"] <= 1).all()
s5 = A["h5"].sum(1)
assert torch.allclose(s5, torch.ones(11), atol=1e-5), "h5 must sum to 1"
s4 = A["h4"][:, 4:7].sum(1)
assert torch.allclose(s4, torch.ones(11), atol=1e-5), "h4 tercile must sum to 1"

# neutralization + EV fields
rng = np.random.default_rng(0)
P = rng.uniform(0.2, 0.8, size=(500, 48)).astype(np.float32)
f = rng.normal(size=500)
years = np.repeat([2016, 2017, 2018, 2019, 2020], 100)
Pn = m._neutralize_per_year(P, f, years)
assert Pn.shape == P.shape and np.isfinite(Pn).all()
# orthogonalization property: within each year, cov(Pn, f) ~ 0
for yv in np.unique(years):
    msk = years == yv
    c = np.cov(Pn[msk][:, 0], f[msk])[0, 1]
    assert abs(c) < 1e-3, (yv, c)
print("neutralization orthogonalizes per year OK")
b_geo = np.array([t / mm for d, mm, t, h in
                  zip(*[[*ax] for ax in [(["L", "S"] * 24),
                                          [x for _ in range(2) for x in
                                           [1.0, 1.5, 2.0, 2.5] * 3 * 2],
                                          [x for _ in range(2) for _ in range(4)
                                           for x in [1.5, 2.5, 4.0] * 2],
                                          [x for _ in range(2) for _ in range(4)
                                           for _ in range(3) for x in [24, 48]]]])])
cell, p_sel, ev = m._ev_fields(P, b_geo)
assert cell.shape == (500,) and (ev >= -1).all()
improvement = (P * b_geo[None] - (1 - P)).max(1)
assert np.allclose(ev, improvement, atol=1e-5)
print("EV fields OK; argmax EV mean =", round(float(ev.mean()), 4))

# parquet dictionary column trick
import pyarrow as pa
vals = np.array([0, 1, 0, 1], dtype=np.int8)
darr = pa.DictionaryArray.from_arrays(pa.array(vals, type=pa.int8()),
                                      pa.array(["fold", "main"], type=pa.string()))
assert darr.to_pylist() == ["fold", "main", "fold", "main"]
print("parquet dictionary column OK")
print("\nALL SMOKE TESTS PASSED")
