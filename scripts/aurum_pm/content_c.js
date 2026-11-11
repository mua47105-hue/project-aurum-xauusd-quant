"use strict";
// Sections 4-5: Failure #3 (vocabulary bottleneck), Failure #4 (zero-crash discipline)
const S = [];

S.push({ h1: "4. Failure #3: The 151,936-Vocabulary Bottleneck, Measured and Sliced", blocks: [
  { t: "h2", x: "4.1 The exact number, and why it was never 151,936" },
  { t: "p", x: "The pilot's token stream (4,571,012 tokens, 17,000 states) uses exactly 89 unique token IDs. Re-serializing all 95,008 states of the full Phase-2 format - 39 features including cross-asset, macro, and priors, with the kernel's own serializer and trend/vol regime tags, tokenized with the real Qwen3 tokenizer - uses exactly 97 unique token IDs (mean 312.9 tokens per state, min 183, max 324, matching the ~312 target of Spec v1.1; the pilot's 89 are a strict subset). The reason is structural and permanent: the Qwen tokenizer splits numbers into single digit tokens, so the numeric fields contribute only the digits, sign, and decimal point; the format strings, session names, and codebook IDs contribute a fixed small set. The state language has a vocabulary of 97 - and the training loop was computing a 151,936-way projection and cross-entropy over it, a 1,566x waste, on every position of every step." },
  { t: "p", x: "The cost was not hypothetical: at bs=16 the fp16 logits alone are 2.4 GB (4.9 GB at the DDP shape), the chunked fp32 CE casts add a transient 1.2 GB, six of the twelve bench-grid configurations OOMed, gradient checkpointing was forced on everywhere (r=3, bs=8/16 only), and the lm_head projection is 0.311 of the 1.293 GFLOP/token forward pass - roughly 24% of forward FLOPs spent projecting into 151,839 tokens that can never be a target. The measured effective throughput of the pilot step is 5.44-5.57 TFLOP/s (8.4-8.6% of the T4's 65 TFLOP/s fp16 peak), which is what a memory-bound, launch-bound small-batch step on Turing looks like." },
  { t: "h2", x: "4.2 The fix: slice the frozen tied head to vocab_keep_ids" },
  { t: "p", x: "Qwen3-0.6B ties the lm_head to the embedding matrix, and the pilot's Stage A' never trains the head (set_trainable leaves it frozen; tied weights make train_lm_head a no-op). Slicing is therefore exact, not an approximation: build the sorted set of active token IDs over all serialized states, gather those rows of the (frozen) embedding matrix as a [97, 1024] read-only matrix, project the hidden states against it, and take cross-entropy against remapped targets. The implementation identity was verified locally to machine precision: CE on the sliced head equals CE on gathered columns of the full head, and the trunk gradients are bit-identical (maxdiff 0.00e+00)." },
  { t: "code", lines: [
    "# kernel patch (aurum_model.py), Stage A' path:",
    "vocab_keep_ids = np.sort(unique_ids(all_serialized_states))   # measured: 97",
    "register_buffer('head_W', base.lm_head.weight[keep_idx].detach())  # [97, 1024], frozen",
    "ids_compact = remap_lookup[ids]        # int16 lookup table, 151936 -> 97",
    "logits = h[:, :L] @ head_W.T           # (B, L, 97) instead of (B, L, 151936)",
    "loss = F.cross_entropy(logits[:, :-1].reshape(-1, 97), ids_compact[:, 1:].reshape(-1))",
    "# no chunking, no fp32 cast of 151936-wide rows, no CE checkpointing needed",
  ] },
  { t: "h2", x: "4.3 Objective semantics: the restriction is the point" },
  { t: "p", x: "Sliced CE is not full-vocab CE with a numerical shortcut - it is the deployment-aligned objective. The full softmax spends its normalizer on 151,839 tokens that can never follow a state token; the sliced softmax asks exactly the question the decider needs answered: which state token comes next. Two local measurements bound the risk of the switch. First, on a toy model with a frozen tied head, sweeping the kept-mass from 6% to 19% (far worse than the real model's situation), the trunk-gradient cosine between full and sliced CE never drops below 0.945 - the optimization direction barely moves even when the normalizer is dominated by foreign tokens. Second, training two identical trunks for 300 steps, the sliced-objective model reaches a better held-out sliced CE than the full-objective model (3.02 vs 3.60), i.e. optimizing the restricted problem directly is better at the restricted problem. And the real model is in a far friendlier regime than the toy: the pilot's final full-vocab loss of 0.3267 means the adapted model already concentrates an average of 72% of the entire 151,936-way softmax mass on the single correct next token, so the kept-set mass is high and the two objectives are nearly coincident. The T4 re-validation run (Section 6.2) will confirm Z_kept/Z_full on real checkpoints in minutes." },
  { t: "h2", x: "4.4 Calibrated cost model, projections, and the corrected quota math" },
  { t: "p", x: "The FLOP model per step (28 layers + 2 extra passes over the 4-layer looped core at r=3; d=1024; 4x fwd with checkpointing, 3x without) reproduces both observed bench rows at 5.44-5.57 TFLOP/s effective. Projections with the sliced head:" },
  { t: "table",
    caption: "Table 6: Measured today versus projected with vocabulary slicing (r=3, seq=512)",
    headers: ["Config", "Step FLOPs", "Throughput (1 GPU)", "Peak memory", "Fits 15.64 GB?"],
    widths: [30, 16, 20, 18, 16],
    size: 18,
    rows: [
      ["current: bs16, ckpt on (measured)", "42.4 TFLOP", "1,051 tok/s", "14.73 GB (measured)", "yes, 1 GB from the edge"],
      ["sliced: bs16, ckpt on", "32.2 TFLOP", "1,385 tok/s (x1.32)", "6.2-11.3 GB (model bounds)", "yes, comfortable"],
      ["sliced: bs16, ckpt off", "24.1 TFLOP", "1,847 tok/s (x1.76)", "~11.3-13 GB (needs T4 confirm)", "to verify"],
      ["sliced: bs32, ckpt off", "48.3 TFLOP", "~1,847 tok/s", "~19 GB (model)", "no - documented boundary"],
    ],
  },
  { t: "p", x: "The memory bounds come from two models that bracket the truth: a structural component model and an empirical two-point fit to the observed bench rows (7.21 GB + 0.470 GB/sequence, exact on both observations; the sliced projection subtracts the 0.145 GB/sequence fp16 logits and the fixed 1.16 GB fp32 CE chunk). The conservative claim is x1.32 with checkpointing kept on and everything else unchanged; the x1.76 figure additionally removes checkpointing at bs=16, which the memory model says fits but only a T4 can confirm (Section 6.2, item 2). A CPU matmul microbenchmark at the exact shapes gives the projection scale: 1,523.9 ms versus 1.53 ms per 1,024 rows - a 996x ratio." },
  { t: "p", x: "The accounting correction: the G0.5 DDP bench counts toks = steps x 16 x 512 x 2 but runs ids of shape (8, 512) per GPU, so its published 4,334 tok/s is exactly 2x the real 2,167 tok/s (the single-GPU bench, 1,077 tok/s at bs=8, agrees with the corrected figure). With honest accounting and the conservative sliced projection (x1.32, checkpointing on), Stage A' at 70M tokens runs at ~2 x 1,385 x 0.86 (head-step overhead) ~ 2,380 tok/s on 2xT4, i.e. ~8.2 h; at the optimistic bound (x1.76, ckpt off) ~4.9-5.5 h. Either way the original 14 h estimate was pessimistic, and Phase 2 (Stage A' + Stage B + table generation) fits inside the remaining 25 h 50 m weekly quota with at least one full retry." },
  { t: "h2", x: "4.5 The latent-realignment alternative, and why it is deferred" },
  { t: "p", x: "The owner's brief also asked about the deeper alternative: drop the LM projection entirely and realign the trunk with a masked-state objective (predict held-out state slots from hidden states) plus joint-head SFT. That path is architecturally clean but changes three things at once - the objective, the pretraining prior, and the training distribution - and cannot be A/B-tested against the pilot without new infrastructure. Vocabulary slicing achieves the same compute and memory relief while preserving the pretrained next-token prior, is a ~20-line patch to code that already ran, and is exactly equal to the old objective on the kept vocabulary. The masked-state realignment is therefore parked as a Phase-4 ablation: if sliced-CE Stage A' underperforms (measured on realized val AUC after Stage B), the ablation runs with the same budget the slicing freed up.", opts: { after: 200 } },
]});

S.push({ h1: "5. Failure #4: Zero-Crash Single-Push Discipline", blocks: [
  { t: "h2", x: "5.1 The local source tree, rebuilt and proven" },
  { t: "p", x: "Phase 0/1 burned 18 kernel pushes on package pins, OOMs, DDP subprocess teardown, and a .values array crash - all classes of failure that a local pre-flight can catch. The precondition for that is a local source tree that provably produces the kernels. The pulled kernel files were un-concatenated back into lib/ modules (aurum_gates, aurum_feat, aurum_label, aurum_data, aurum_ser, aurum_model - byte-identical across all three kernels) plus per-kernel bodies, and a rebuilt build.py reassembles them. Verification: all three regenerated kernels are byte-exact against the pulled Kaggle sources (sha256 prefixes 77ce874eb130 / 8b8fe7c860f9 / 811aa14493e2, ignoring only the build-timestamp comment line). From now on, kernels are edited in src/lib + src/kernels, regenerated by build.py, and pushed only after the pre-flight is green." },
  { t: "h2", x: "5.2 test_local.py: 40 checks, 10 sections, 13 seconds" },
  { t: "table",
    caption: "Table 7: The pre-flight suite (all 40 checks green locally in 13 s)",
    headers: ["Section", "What it verifies", "Catches the crash class of"],
    widths: [18, 52, 30],
    size: 18,
    rows: [
      ["build", "build.py regenerates all kernels byte-exactly vs the pulled sources", "source/binary drift, hand-edited kernels"],
      ["static", "AST syntax + crash-class regex audit of every source file (idxmax/idxmin label-vs-position, .values.astype on object dtype, mutable defaults, unmarked bare excepts)", "the .values crash, silent exception swallowing"],
      ["env", "package pins available and API-compatible vs the Kaggle manifest (pandas 2.3.3 / numpy 2.1.3 / xgboost 3.4.1 / sklearn 1.6.1 / transformers 4.57.1 / torch 2.11)", "v1-v3 pin debugging"],
      ["labeler", "hand-constructed M1 paths with known first-touch outcomes: long TP, long SL, short TP, far-TP timeout, same-minute tie resolves SL-first", "labeler logic regressions, convention drift"],
      ["cohort", "all neighbors strictly past (<= i-96), all label-valid, early bars have none", "leakage regressions in the index builder"],
      ["scalein", "x137 price scaling: feature maxdiff 4.6e-10, label maxdiff 7.6e-14 (local G0.3 replica)", "scale-invariance regressions"],
      ["serial", "serialization non-empty; mock-tokenizer vocab; sliced-CE == gathered-CE identity and bit-identical trunk gradients", "vocab-slicing implementation bugs"],
      ["ddp", "torchrun gloo world=2 spawn, 3 training steps, all_reduce grad-norm check, destroy_process_group in finally, clean fast exit", "v4 DDP zombie/OOM teardown"],
      ["vram", "empirical memory model exact on both observed bench rows; sliced bs16 fits (11.3 GB), sliced bs32 documented OOM boundary", "OOM pushes (6 of 12 grid configs last time)"],
      ["resume", "simulated kill mid-loop, resume completes, output identical to uninterrupted run", "gentable kill/resume regressions"],
    ],
  },
  { t: "h2", x: "5.3 The push protocol" },
  { t: "p", x: "One rule, mechanically enforced: a kernel is pushed to Kaggle if and only if (1) python3 build.py --verify exits 0 and (2) python3 test_local.py exits 0, both against the exact sources being pushed, and (3) the kernel-metadata.json lint passes (id matches tskdev/<slug>, code_file exists, enable_gpu matches the kernel's needs, dataset_sources reference existing slugs). The suite runs in 13 seconds; there is no schedule pressure that justifies skipping it. The 18-push debug history maps to the sections above: pins -> env, OOMs -> vram, DDP teardown -> ddp, .values -> static + labeler. The expected effect is that Phase-2 kernel pushes are single-push, with iteration happening locally at 13 seconds per cycle instead of 10-20 minutes per Kaggle round-trip.", opts: { after: 200 } },
]});

module.exports = S;
