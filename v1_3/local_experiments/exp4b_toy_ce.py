# ============================================================================
# EXP4b - Sliced-CE vs full-CE: divergence and training-dynamics equivalence.
# The sliced head restricts the softmax to the ~97-token state vocabulary.
# This is a HYPOTHESIS-CLASS restriction, not an approximation: with the
# pretrained mass concentrated on state tokens, divergence is small. Measure:
#   (1) loss offset + trunk-gradient cosine vs kept-mass level
#   (2) 300-step training runs, full vs sliced objective, evaluated on the
#       DEPLOYMENT metric (held-out sliced CE) - the pilot's final LM loss
#       0.3267 implies avg target probability ~0.72, i.e. high kept-mass.
# ============================================================================
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn, os, sys, json
torch.manual_seed(0); np.random.seed(0)

OUT = os.path.dirname(os.path.abspath(__file__)) + "/results"
V, K, D, T, B = 1000, 50, 64, 128, 64

E = torch.randn(V, D)                       # frozen tied embedding (stand-in)
trunk = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

# data: tokens only from the keep set (like serialized states)
keep = torch.randperm(V)[:K]
pos_of = {int(v): i for i, v in enumerate(keep)}
x = torch.randn(B * T, D)
tgt = keep[torch.randint(0, K, (B * T,))]

def fwd(h):
    return trunk(h)

res = {}

# ---- (1) divergence vs kept-mass ----
# control kept-mass by scaling the kept-token embedding rows (adapted model
# concentrates mass on state tokens; scale s ~ exp effect on kept logits)
rows = []
keep_mask = torch.zeros(V, dtype=torch.bool); keep_mask[keep] = True
for s in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    Ew = E.clone(); Ew[~keep_mask] /= s   # push mass ONTO the kept set
    with torch.no_grad():
        h0 = fwd(x)
        logits_full = h0 @ Ew.T
        # kept-mass = softmax mass on keep set averaged over positions
        p = Fn.softmax(logits_full, dim=1)
        kept_mass = float(p[:, keep].sum(1).mean())
        tgt_pos = torch.tensor([pos_of[int(t)] for t in tgt])
        l_full = float(Fn.cross_entropy(logits_full, tgt))
        l_slice = float(Fn.cross_entropy(h0 @ Ew[keep].T, tgt_pos))
    h2 = h0.clone().requires_grad_(True)
    lf = Fn.cross_entropy(h2 @ Ew.T, tgt)
    gs = torch.autograd.grad(lf, h2, retain_graph=True)[0]
    h3 = h0.clone().requires_grad_(True)
    ls = Fn.cross_entropy(h3 @ Ew[keep].T, tgt_pos)
    gslice = torch.autograd.grad(ls, h3, retain_graph=True)[0]
    cos = float(Fn.cosine_similarity(gs.flatten(), gslice.flatten(), dim=0))
    rows.append({"scale": s, "kept_mass": round(kept_mass, 4),
                 "full_CE": round(float(l_full), 4), "sliced_CE": round(float(l_slice), 4),
                 "grad_cosine_h": round(cos, 5)})
    print(rows[-1])
res["divergence_vs_kept_mass"] = rows

# ---- (2) training dynamics: 300 steps, same data, two objectives ----
def train(obj):
    torch.manual_seed(1)
    tr = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
    opt = torch.optim.AdamW(tr.parameters(), lr=3e-3)
    for step in range(300):
        h = tr(x)
        if obj == "full":
            loss = Fn.cross_entropy(h @ E.T, tgt)
        else:
            loss = Fn.cross_entropy(h @ E[keep].T, torch.tensor([pos_of[int(t)] for t in tgt]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        h = tr(x)
        tgt_pos = torch.tensor([pos_of[int(t)] for t in tgt])
        return {"held_sliced_CE": round(float(Fn.cross_entropy(h @ E[keep].T, tgt_pos)), 4),
                "held_full_CE": round(float(Fn.cross_entropy(h @ E.T, tgt)), 4)}

res["training_dynamics"] = {"full_obj": train("full"), "sliced_obj": train("sliced")}
print(res["training_dynamics"])

json.dump(res, open(os.path.join(OUT, "exp4b_toy_ce.json"), "w"), indent=1)
print("exp4b done")
