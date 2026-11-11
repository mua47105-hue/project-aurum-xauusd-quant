#!/usr/bin/env python3
"""Local live-pathway smoke test with a TINY mock Qwen3 (same class, random
weights) — validates the full machinery: serialize -> tokens -> build_core ->
forward -> heads -> temps -> decision -> audit. Real values are validated on
Kaggle when the kernel runs."""
import os, sys, json, types
import numpy as np
import torch

REF = "/home/z/my-project/kaggle_work/aurum/ref"
WORK = "/home/z/my-project/kaggle_work/aurum/smoke_live"
os.makedirs(WORK, exist_ok=True)
os.environ["AURUM_DS"] = REF
os.environ["AURUM_MODELS"] = os.path.join(REF, "models")
os.environ["AURUM_WORK"] = WORK

sys.path.insert(0, "/home/z/my-project/kaggle_work/aurum/src/kernels")
import live_daily as L

# ---------------- mocks (only the HF-download parts) ----------------
D, V = 64, 512
L.N_READOUT = 5

class MockTok:
    """Deterministic 330-token mock (valid range for G6.1) with ids < 512."""
    vocab_size = V
    def __call__(self, text, add_special_tokens=False):
        n = 330
        step = max(1, len(text) // n)
        ids = []
        for k in range(n):
            chunk = text[k * step:(k + 1) * step + 1] or "x"
            h = sum(ord(c) * (7 ** (j % 5)) for j, c in enumerate(chunk))
            ids.append(h % V)
        return {"input_ids": ids}

def mock_load_base(device=None, dtype=torch.float32, log=print):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=V, hidden_size=D, intermediate_size=128,
                      num_hidden_layers=20, num_attention_heads=4,
                      num_key_value_heads=2, tie_word_embeddings=True)
    m = Qwen3ForCausalLM(cfg).to(dtype)
    if device is not None:
        m = m.to(device)
    m.eval()
    log(f"  [MOCK] tiny Qwen3 ({sum(p.numel() for p in m.parameters())/1e6:.2f}M params)")
    return m

def mock_core_sd():
    torch.manual_seed(0)
    layers = {}
    for i in range(L.CORE_START, L.CORE_END):
        layers[str(i)] = {k: v.float() for k, v in _tiny_layer_sd(i).items()}
    pre, post = {}, {}
    for r in range(3):
        pre.update({f"{r}.{k}": v for k, v in _tiny_sandwich_sd(f"pre{r}").items()})
        post.update({f"{r}.{k}": v for k, v in _tiny_sandwich_sd(f"post{r}").items()})
    return {"layers": layers, "pre": pre, "post": post,
            "readout": {"readout.weight": torch.randn(5, D)}}

_SD_CACHE = {}
def _tiny_layer_sd(i):
    if i not in _SD_CACHE:
        import transformers
        cfg = transformers.Qwen3Config(vocab_size=V, hidden_size=D,
                                       intermediate_size=128,
                                       num_hidden_layers=20,
                                       num_attention_heads=4,
                                       num_key_value_heads=2)
        from transformers import Qwen3ForCausalLM
        m = Qwen3ForCausalLM(cfg)
        _SD_CACHE[i] = m.model.layers[i].state_dict()
    return _SD_CACHE[i]

def _tiny_sandwich_sd(tag):
    import torch.nn as nn
    sd = nn.ModuleDict({"a": nn.LayerNorm(D), "b": nn.LayerNorm(D)}).state_dict()
    return sd

# patch the HF-dependent pieces
import live_daily
L.load_base = mock_load_base
# transformers.AutoTokenizer patch INSIDE the package so that live_main's
# function-local `from transformers import AutoTokenizer` gets the shim
import transformers
class _TokShim:
    @staticmethod
    def from_pretrained(*a, **k):
        return MockTok()
transformers.AutoTokenizer = _TokShim
live_daily.AutoTokenizer = _TokShim
# keep the real torch.load for heads (shape mismatch vs tiny d=64) -> mock heads
real_heads_sd = torch.load(os.path.join(REF, "models", "heads_main.pt"),
                           map_location="cpu", weights_only=False)
import torch.nn as nn
def mock_head_stack(d):
    class HS(nn.Module):
        def __init__(self):
            super().__init__()
            self.readout = nn.Embedding(5, d)
            def mlp(out):
                return nn.Sequential(nn.Linear(d, 256), nn.GELU(),
                                     nn.Dropout(0.0), nn.Linear(256, out))
            self.h1, self.h2, self.h3 = mlp(48), mlp(8), mlp(12)
            self.h4, self.h5 = mlp(8), mlp(9)
        def forward(self, ro_hidden):
            return {"h1": self.h1(ro_hidden[:, 0]), "h2": self.h2(ro_hidden[:, 1]),
                    "h3": self.h3(ro_hidden[:, 2]), "h4": self.h4(ro_hidden[:, 3]),
                    "h5": self.h5(ro_hidden[:, 4])}
    torch.manual_seed(1)
    return HS()
live_daily._head_stack = mock_head_stack
# torch.load for core_fp16/heads_main -> mocks matching the tiny arch
_live_torch_load = torch.load
def patched_load(path, *a, **k):
    if path.endswith("core_fp16.pt"):
        return mock_core_sd()
    if path.endswith("heads_main.pt"):
        return mock_head_stack(64).state_dict()
    return _live_torch_load(path, *a, **k)
live_daily.torch.load = patched_load

live_daily.live_main(WORK)

print("\n=== GATES ===")
gates = json.load(open(f"{WORK}/gates.json"))
allok = True
for k, v in gates.items():
    print(f"  {k}: {'PASS' if v['pass'] else 'FAIL'} | {str(v['detail'])[:110]}")
    allok &= v["pass"]
dec = json.load(open(f"{WORK}/decision.json"))
print("\n=== DECISION (mock values) ===")
print(json.dumps(dec["decision"], indent=1))
print("sizing:", json.dumps(dec["sizing_reference"], indent=1))
print("gates:", json.dumps(dec["gates"], indent=1)[:400])
print("regime:", dec["heads_detail"]["regime_call"])
txt = open(f"{WORK}/live_state.txt").read()
print(f"\nstate text: {len(txt)} chars; head: {txt[:100]!r}")
assert allok, "GATES FAILED"
print("\nLIVE MACHINERY SMOKE TEST PASSED")
