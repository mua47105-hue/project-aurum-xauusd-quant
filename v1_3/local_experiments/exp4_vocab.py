# ============================================================================
# EXP4 - Vocabulary Slicing (Fix for Failure #3): exact vocab, cost model
#        calibrated against T4 bench.json, gradient-equivalence, projections.
#
# P1: exact active vocabulary of the FULL Phase-2 state format (39 features,
#     re-serialized from the pulled parquets with the EXACT kernel serializer,
#     tokenized with the real Qwen3 tokenizer).
# P2: FLOPs/memory cost model, calibrated against observed bench.json rows.
# P3: sliced-CE vs full-CE divergence on a toy model (loss + trunk-gradient
#     cosine) under realistic kept-mass levels.
# P4: CPU matmul microbenchmark at exact lm_head shapes.
# P5: projected T4 throughput/memory with vocab slicing (+ G0.5 accounting
#     bug finding: DDP bench double-counts tokens).
# ============================================================================
import numpy as np, pandas as pd, sys, os, json, time, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aurum_common import load_all, year_series, logd, save_json

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
ART = "/home/z/my-project/kaggle_work/aurum/p1-data/out"
KSRC = "/home/z/my-project/kaggle_work/aurum/p1-data/src/aurum-p1-data.py"

res = {}

# --------------------------------------------------------------------------
# P1: exact active vocabulary, full 39-feature format
# --------------------------------------------------------------------------
spec = importlib.util.spec_from_file_location("aurum_p1", KSRC)
K = importlib.util.module_from_spec(spec)
sys.modules["aurum_p1"] = K
spec.loader.exec_module(K)          # runs _pin_transformers() (no-op on 4.57.1)

d = load_all()
feat = d["feat"]; ts = d["ts"]
h1 = pd.read_parquet(os.path.join(ART, "h1_xauusd.parquet"))
h1 = h1[~h1.index.duplicated()].sort_index()
assert len(h1) == len(feat)

# trend3/vol3 exactly as kernel Stage G
a = K.wilder_atr(h1["h"], h1["l"], h1["c"], 14)
r24 = (h1["c"] - h1["c"].shift(24)) / (a + 1e-12)
t33 = r24.rolling(252, min_periods=60).quantile(0.33).shift(1)
t66 = r24.rolling(252, min_periods=60).quantile(0.66).shift(1)
trend = np.where(r24 < t33, 0, np.where(r24 > t66, 2, 1)).astype(float)
trend[~np.isfinite(t33.values) | ~np.isfinite(r24.values)] = np.nan
ar = a / a.shift(48)
v33 = ar.rolling(252, min_periods=60).quantile(0.33).shift(1)
v66 = ar.rolling(252, min_periods=60).quantile(0.66).shift(1)
volc = np.where(ar < v33, 0, np.where(ar > v66, 2, 1)).astype(float)
volc[~np.isfinite(v33.values) | ~np.isfinite(ar.values)] = np.nan

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
V_FULL = tok.vocab_size
logd(f"tokenizer loaded, vocab_size={V_FULL}")

t0 = time.time()
texts = []
for i in range(len(feat)):
    texts.append(K.serialize_state(ts[i], feat.iloc[i], with_cross=True,
                                   with_macro=True, with_priors=True,
                                   drop=set(), trend3=trend[i], vol3=volc[i]))
logd(f"serialized {len(texts)} states in {time.time()-t0:.0f}s")

uniq = set()
tok_counts = []
B = 2000
for s in range(0, len(texts), B):
    enc = tok(texts[s:s+B], add_special_tokens=False)["input_ids"]
    for row in enc:
        tok_counts.append(len(row))
        uniq.update(row)
tok_counts = np.array(tok_counts)
vocab_keep = sorted(uniq)
res["p1_exact_vocab"] = {
    "n_states": len(texts),
    "mean_tokens_per_state": round(float(tok_counts.mean()), 1),
    "min": int(tok_counts.min()), "max": int(tok_counts.max()),
    "unique_token_ids_full_format": len(vocab_keep),
    "vocab_size_qwen3": V_FULL,
    "compression_ratio": round(V_FULL / len(vocab_keep), 1),
    "pilot_stream_unique_ids": 89,
}
np.save(os.path.join(OUT, "vocab_keep_ids.npy"), np.array(vocab_keep, dtype=np.int64))
logd(f"P1: full-format active vocab = {len(vocab_keep)} / {V_FULL} "
     f"({V_FULL/len(vocab_keep):.0f}x), tokens/state mean={tok_counts.mean():.1f}")

# pilot stream subset check
z = np.load("/home/z/my-project/kaggle_work/aurum/p0-pilot/out/pilot_data.npz")
pilot_ids = set(np.unique(z["flat"]).tolist())
res["p1_exact_vocab"]["pilot_ids_subset_of_full"] = bool(pilot_ids <= set(vocab_keep))
res["p1_exact_vocab"]["compression_vs_lm_head_151936"] = round(151936 / len(vocab_keep), 1)
res["p1_exact_vocab"]["pilot_ids_not_in_full"] = sorted(pilot_ids - set(vocab_keep))[:20]

# --------------------------------------------------------------------------
# P2: cost model calibrated against bench.json
# --------------------------------------------------------------------------
bench = json.load(open("/home/z/my-project/kaggle_work/aurum/p0-core/out/bench.json"))
D_MODEL, L_TOT, L_CORE, V_QWEN = 1024, 28, 4, 151936
INTER = 3072
P_LAYER = 4 * D_MODEL**2 + 3 * D_MODEL * INTER      # attn qkvo + mlp
P_NONEMB = L_TOT * P_LAYER                          # 381.7M
P_EXTRA_LOOP = 2 * L_CORE * P_LAYER                 # r=3 extra fwd params
FLOP_TOK_CORE = 2 * (P_NONEMB + P_EXTRA_LOOP) / 1e9 # GFLOP/token fwd
FLOP_TOK_HEAD_FULL = 2 * D_MODEL * V_QWEN / 1e9
FLOP_TOK_HEAD_SLICED = 2 * D_MODEL * len(vocab_keep) / 1e9

def step_flops(bs, seq, ckpt, head="full"):
    tok = bs * seq
    f_core = FLOP_TOK_CORE * tok          # GFLOP
    f_head = (FLOP_TOK_HEAD_FULL if head == "full" else FLOP_TOK_HEAD_SLICED) * tok
    mult = 4.0 if ckpt else 3.0                     # fwd+bwd(+recompute)
    return (f_core + f_head) * mult / 1e3           # -> TFLOP/step

calib = []
for row in bench:
    if row.get("OOM") or row.get("r") != 3:
        continue
    tf = step_flops(row["bs"], 512, row["ckpt"])
    eff = tf / (row["step_ms"] / 1000)              # TFLOP/s
    calib.append({"cfg": f"r3 bs{row['bs']} ckpt{int(row['ckpt'])}",
                  "observed_tok_s": row["tok_per_s"], "model_TFLOP": round(tf, 1),
                  "effective_TFLOPs": round(eff, 2),
                  "MFU_vs_65T_peak": round(eff / 65, 3)})
res["p2_calibration"] = calib
eff_tfs = float(np.mean([c["effective_TFLOPs"] for c in calib
                         if "bs16" in c["cfg"]]))
logd(f"P2: calibration vs bench.json -> effective {eff_tfs:.2f} TFLOP/s "
     f"(MFU {eff_tfs/65*100:.1f}% of T4 fp16 peak)")

# memory model (fp16 activations, fp32 master weights)
def mem_gb(bs, seq, ckpt, head="full", n_vocab=None):
    static = 0.537e9 * 4 / 2**30                      # fp32 weights 2.15GB
    train_p = (L_CORE * P_LAYER + 2.5e6)              # looped core + norms + head
    static += (train_p * 4 * 3) / 2**30               # grad + Adam m,v (fp32)
    Vh = V_QWEN if head == "full" else (n_vocab or len(vocab_keep))
    logits = bs * (seq - 1) * Vh * 2 / 2**30          # fp16 logits
    ce_fp32 = logits / 2                               # one chunk transient fp32
    # activations: ~14 saved tensors of B*T*d per layer-execution (36 execs r=3)
    act_per_layer = 14 * bs * seq * D_MODEL * 2 / 2**30
    n_exec = L_TOT - L_CORE + L_CORE * 3
    act = act_per_layer * n_exec * (0.35 if ckpt else 1.0)
    return {"static": round(static, 2), "logits": round(logits, 3),
            "ce_fp32_chunk": round(ce_fp32, 3), "activations": round(act, 2),
            "total_est": round(static + logits + ce_fp32 + act + 0.8, 2)}  # +workspace

mem_rows = []
for row in bench:
    if row.get("OOM") or row.get("r") != 3:
        continue
    m = mem_gb(row["bs"], 512, row["ckpt"])
    mem_rows.append({"cfg": f"r3 bs{row['bs']} ckpt{int(row['ckpt'])}",
                     "observed_gb": row["mem_gb"], "model_est_gb": m["total_est"]})
res["p2_memory_model"] = mem_rows

# projected configs with slicing
projs = []
for bs, ckpt in [(8, True), (16, True), (16, False), (32, False)]:
    tf = step_flops(bs, 512, ckpt, head="sliced")
    t_s = tf / eff_tfs
    toks = bs * 512 / t_s
    m = mem_gb(bs, 512, ckpt, head="sliced")
    projs.append({"cfg": f"sliced r3 bs{bs} ckpt{int(ckpt)}",
                  "model_TFLOP": round(tf, 1), "projected_step_ms": round(t_s * 1000, 0),
                  "projected_tok_s_1gpu": round(toks, 0),
                  "projected_mem_gb": m["total_est"],
                  "fits_15gb": m["total_est"] < 14.5})
res["p2_projections_sliced"] = projs
cur = [r for r in bench if not r.get("OOM") and r["r"] == 3 and r["bs"] == 16 and r["ckpt"]][0]
res["p2_current_best"] = cur
speedup = [p for p in projs if p["cfg"] == "sliced r3 bs16 ckpt0"][0]["projected_tok_s_1gpu"] / cur["tok_per_s"]
res["p2_projected_speedup_1gpu"] = round(speedup, 2)
logd(f"P2: current r3/bs16/ckpt1 = {cur['tok_per_s']} tok/s; projected sliced "
     f"bs16/ckpt0 = {[p for p in projs if 'bs16 ckpt0' in p['cfg']][0]['projected_tok_s_1gpu']} "
     f"tok/s (x{speedup:.2f})")

# G0.5 accounting bug
res["p5_g05_bug"] = {
    "reported_ddp_tok_s": 4334.2,
    "actual_per_gpu_bs": 8, "counted_per_gpu_bs": 16,
    "actual_total_tok_s": round(8 * 512 * 2 / 3.7798, 1),
    "note": "run_ddp_bench uses ids=(8,512) but counts toks=len(times)*16*512*2; "
            "the DDP gate number is 2x inflated; single-GPU bench (1076.9 @ bs8) "
            "is consistent with the corrected DDP figure (~1084/GPU)."}

# --------------------------------------------------------------------------
# P4: CPU matmul microbenchmark at exact shapes
# --------------------------------------------------------------------------
rng = np.random.default_rng(0)
Hm = rng.standard_normal((1024, D_MODEL)).astype(np.float32)   # 1 chunk of bs16x511
for Vh, tag in [(V_QWEN, "full"), (len(vocab_keep), "sliced")]:
    W = rng.standard_normal((D_MODEL, Vh)).astype(np.float32)
    t0 = time.time(); n = 0
    while time.time() - t0 < 2.0:
        Hm @ W; n += 1
    dt = (time.time() - t0) / n
    gf = 2 * Hm.shape[0] * D_MODEL * Vh / 1e9
    res.setdefault("p4_matmul", {})[tag] = {"ms_per_1024rows": round(dt * 1000, 2),
                                            "GFLOP": round(gf, 1),
                                            "per_token_us": round(dt * 1e6 / Hm.shape[0], 3)}
    logd(f"P4 matmul {tag}: {dt*1000:.2f} ms/1024rows ({gf:.0f} GFLOP), "
         f"{dt*1e6/1024:.3f} us/token")
res["p4_matmul"]["ratio"] = round(res["p4_matmul"]["full"]["ms_per_1024rows"] /
                                  res["p4_matmul"]["sliced"]["ms_per_1024rows"], 1)

save_json(res, os.path.join(OUT, "exp4_vocab_cost.json"))
logd("exp4 parts 1/2/4 done; toy-equivalence in exp4b")
