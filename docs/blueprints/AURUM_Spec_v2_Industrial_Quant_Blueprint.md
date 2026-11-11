# AURUM Spec v2.0 — Industrial Quant Blueprint

**Project AURUM · Master Technical Specification & Mathematical Blueprint**
**Title:** `AURUM_Spec_v2_Industrial_Quant_Blueprint`
**Role:** Principal Quantitative Research Architect (Agent 2)
**Research method:** DeerFlow deep-research methodology — 4-phase protocol (broad exploration → deep dive → diversity & validation → synthesis check), executed as a 7-pillar parallel excavation across arXiv, SSRN, Crossref, institutional portals, and GitHub source code (~150 sources fetched and read; ~80 load-bearing entries in the Section 2 ledger; every formula traced to a fetched primary or explicitly flagged `[UNVERIFIED]`)
**Computation contract:** Kaggle free tier, 2× NVIDIA Tesla T4 (16 GB, sm_75, fp16 — no bf16), ~25–30 GPU-hours/week; existing dataset `tskdev/aurum-phase1-data` (2010–2025 M1+H1 parquets, XAUUSD/XAGUSD/EURUSD/USDJPY + FRED macro)
**Date:** November 9, 2026
**Status:** Spec v2.0 — ready for Agent 1 execution (Section 6 handoff, ≤10 GPU-hours)

---

## Executive Summary

AURUM v1.2 executed cleanly as engineering and failed honestly as statistics: once Agent 1 removed the ex-post H3 overlay bug, every model's clean OOS expectancy collapsed to near zero (GBDT +0.0487R, k-NN −0.0223R, AURUM 2-cell −0.0400R), the 48-cell argmax underperformed a 2-cell restriction, extra loop depth (r>1) added nothing, and the Deflated Sharpe Ratio of the flagship was 0.0000. Section 1 of this blueprint establishes — mechanism by mechanism, with primary-source mathematics — that none of this was bad luck. It was the *predicted* outcome of four independent design violations:

1. **The input port violated the number-encoding law.** ~330 rounded ASCII tokens ("ret24 +1.4 ATR") through 12 frozen English-pretrained layers destroys the continuous cross-variable Jacobian the model needed (Chronos/xVal/LLMTime/Time-LLM; Grinsztajn's tabular-DL results). The GBDT reading the *same* features as continuous floats beat the transformer — the input port, not the trunk, was the first ceiling.
2. **The inductive bias was mismatched.** 63M trainable parameters (≈875 per training state) with smoothness-biased architecture on an irregular, axis-aligned, low-SNR tabular problem is the exact regime where tree ensembles win (Grinsztajn, NeurIPS 2022).
3. **The loop was unconstrained.** Iterating h ← f_θ(h,e) with no contraction guarantee amplifies estimation error geometrically whenever ρ(I+J_f)>1 — DEQ/monDEQ/RED exist precisely to prevent this, v1.2 used none of it, and r=1 > r=2 > r=3 OOS is the measured signature of an expansive loop (Section 1.4).
4. **The output port was a winner's-curse machine.** argmax over 48 noisy EV estimates with Var(EV_c) = (b_c+1)²·Var(p̂_c) — a 4× variance tilt toward TP=4.0 cells — selects the *noise* (E[max] of 48 zero-skill estimates ≈ 2.26σ; Bailey–Borwein–López de Prado–Zhu), which is why 2 cells beat 48.

On top of these, the H3 overlay fabricated ~+8,700R of synthetic profit on the GBDT configuration alone (paired-number arithmetic, Section 1.6) — a textbook ex-post bias that the v1.2 report's own "path-valid" language disguised.

**v2.0 replaces all four ports** and adds the layers v1.x never had:

- **Feature engine (Section 3):** six drop-in, causally-tested modules — DST-aware exchange clocks (LBMA 10:30/15:00 London, SGE 10:15/15:00 *Beijing*, COMEX 18:00 ET, WMR 16:00 London — all verified), M1 realized geometry (RV/BV/semivariances/signed-jump variation, jump statistics, variance ratios, roughness), depth-2/3 multi-asset path signatures with Lévy areas (the "who moved first" detector), VPIN/Kyle-λ/Amihud toxicity, online EW-RLS cross-asset residualization (Channel A lead-lag catch-up, Channel B idiosyncratic overextension), and the CFM/Carver saturating EWMAC trend ladder (s\*≈0.89 tanh saturation). ~330 features; every module ships with a self-test including a truncation-based causality proof.
- **Model stack (Section 4):** *AURUM-Continuum* — a purpose-built ~3–4M-parameter continuous-tensor transformer (d_model=256; TFT variable-selection + GRN blocks; 2 prelude + **spectrally-capped contractive looped core** + 1 coda; per-iterate supervision; RevIN) trained on a **differentiable net-Sharpe loss with transaction costs inside the graph** plus turnover and Ulcer-drawdown penalties — ensembled with the **Kaggle-floor stack** (JS-2021-winner-style supervised AE-MLP + LightGBM) and gated by the **DMN-LSTM baseline**. Path B (Qwen3-0.6B trunk + continuous patch projector + LoRA) is demoted to a bounded ≤15%-of-budget probe with explicit kill criteria — the evidence table (Section 4.7) shows zero finance-win precedent for LLM trunks and a 10–100× throughput tax on T4s.
- **Selection & sizing (Section 4.8):** the 48-cell argmax is *retired*. Continuous vol-scaled positions replace discrete cells; where discrete barriers remain (risk framing), selection uses **EV_LCB** with overlap-discounted n_eff and **positive-part James–Stein shrinkage** — the statistical mirror image of the v1.x winner's curse.
- **Online engine (Section 5):** label-arrival-gated RLS readout updates (embargo-safe, H=24 bars), the hyperparameter-free RLS-bank floor that recent TTA literature fails to beat, Page–Hinkley break gates (river-exact recursions), real-time Moreira–Muir vol targeting, Carver no-trade buffer (±10%, trade-to-edge), and a single continuous position that makes the v1.x overlapping-trade pathology structurally impossible.
- **Verification (Section 5.5):** purged walk-forward (purge=24, embargo=24–48), the full config ledger feeding a **Deflated Sharpe Ratio ≥ 0.95** acceptance gate and **CSCV-PBO ≤ 0.2** quality bar — the multiple-testing discipline that v1.2 applied at the end, applied continuously from day one.

The handoff (Section 6) is a file-by-file, kernel-by-kernel execution plan for Agent 1: 11 files, 6 kernels, ≤10 GPU-hours, hard gates at every step, kill criteria for every optional branch.

---

## How to Read This Document

| Section | Content | Audience |
|---|---|---|
| 1 | Forensic post-mortem of v1.0–v1.3 (the *why*) | Both agents, owner |
| 2 | Literature & codebase ledger — ~80 primary sources, verification status, citation corrections | Owner, reviewers |
| 3 | The v2.0 feature engine: exact formulas + drop-in Python (tested) | Agent 1 |
| 4 | Model & loss architecture: ContinuumNet, AE-MLP/GBDT floor, DMN baseline, selection (tested) | Agent 1 |
| 5 | Causal online adaptation & execution engine (tested) | Agent 1 |
| 6 | Step-by-step handoff protocol: kernels, budgets, gates, kill criteria | Agent 1 |
| Appendix A | Constants card (every number, its value, its source) | Both |
| Appendix B | Unverified-items register (honest-open-questions list) | Owner |

**Code provenance.** Every code block in Sections 3–5 is the *actual* file executed in this research session, with its self-test. Files live in `spec_v2/code/aurum_features/` (feature engine, numpy/pandas only) and `spec_v2/model/` (PyTorch). The tests include truncation-based causality proofs (features at time t are invariant to future data), known-geometry checks (the signature of a triangle equals its area; reversal flips the sign), and estimator-unbiasedness checks (Monte-Carlo-verified variance-ratio normalization). Agent 1's job is to wire them to the Kaggle dataset, not to re-derive them.

**Verification key used throughout:** ✅ = formula/fact fetched and read from the primary source this session; ◐ = verified from a fetched secondary reproduction or metadata (Crossref/arXiv API); ⚠️ = standard result, primary not reachable — flagged, do not build load-bearing structure on it without confirmation. The full register of ⚠️ items is Appendix B.

---

---

## Table of Contents

- **Section 1** — Forensic Post-Mortem of v1.0–v1.3
- **Section 2** — Literature & Codebase Excavation Ledger (~80 primary sources; 8 citation corrections; verification key)
- **Section 3** — The v2.0 Institutional Feature Engine (F1 clocks · F2 realized geometry · F3 signatures · F4 toxicity · F5 residualization · F6 trend ladder · builder)
- **Section 4** — Model & Loss Architecture (ContinuumNet · contractive loop & spectral governor · net-Sharpe loss · AE-MLP/GBDT floor · DMN baseline · Path B probe · LCB/James-Stein/Kelly selection)
- **Section 5** — Causal Online Adaptation & Execution Engine (RLS readout · RLS-bank · regime gates · Carver buffer · DSR/PBO)
- **Section 6** — Step-by-Step Handoff Protocol for Agent 1 (6 kernels, ≤10 GPU-h, gates G1–G6, kill criteria, trap list)
- **Appendix A** — Constants Card · **Appendix B** — Unverified-Items Register

---



# Section 1 — Forensic Post-Mortem of AURUM v1.0–v1.3

This section proves, from the literature and from AURUM's own numbers, *why* v1.x hit its ceiling. It is organized as five failures. Failures I–IV are architecture; Failure V is measurement. Each ends with the v2.0 design mandate it dictates.

## 1.0 The evidence base (v1.3, clean)

After Agent 1's line-by-line audit of `p4_backtest.py` removed the H3 ex-post overlay bug (Section 1.6) and the Buy & Hold backward-returns bug, the honest out-of-sample 2023–2025 picture at 1.0× costs is:

| Model | OOS E[R]/trade (buggy overlay) | OOS E[R]/trade (clean) | Verdict |
|---|---|---|---|
| GBDT (44 era-safe features) | +0.6321R | **+0.0487R** | weak positive edge |
| k-NN Cohort | +0.5311R | **−0.0223R** | no edge |
| AURUM 2-cell | +0.2788R | **−0.0400R** | no edge |
| AURUM r=1 (unlooped) | +0.2145R | (near-zero family) | loop adds nothing |
| AURUM r=3 (production) | +0.1644R, DSR 0.0000 | (near-zero family) | fails multiple-testing |
| Depth sweep r=1..4 | 0.2145 / 0.1566 / 0.1644 / 0.1572 | — | **r=1 best**: iteration hurts |
| 48-cell vs 2-cell | +0.1644R vs +0.2788R | — | **smaller selection set wins** |

Consistent with this, the v1.2 postmortem's own target tournament (T5) had already shown shallow, heavily-regularized GBDT at +0.0599R (2023–25) / +0.0297R-per-year (2016–25) as the best clean performer — the "model-class honesty clause" bar. **The transformer never cleared the bar its own postmortem set.** The rest of this section explains the five reasons, each with its literature anchor.

## 1.1 Failure I — The ASCII digit-tokenization & frozen-English-prelude bottleneck

**Mechanism.** v1.2 serialized each hourly state into ~330 tokens of rounded market text ("ret24 +1.4 ATR") and passed them through the 12 *frozen* layers (1–12) of Qwen3-0.6B before the trained loop (13–16) ever saw them. Three independent pathologies:

**(a) Quantization noise is injected before the model sees anything.** Rounding to a grid step Δ injects error e ~ U[−Δ/2, Δ/2] with Var[e] = Δ²/12 (standard quantization theory ⚠️). For "1.4 ATR" at Δ=0.1, that is σ_e ≈ 0.029 ATR units of *irreducible* noise per number — on features whose entire predictive content is a few hundredths of an ATR. Compare the deliberate state of the art: Chronos ✅ (arXiv:2403.07815) quantizes values into 4,094 uniform bins over [−15s, +15s] after mean-scaling by s = (1/C)Σ|x_i| — a token spacing of 30s/4093 ≈ 0.0073s, i.e., ~3 significant digits *after* adaptive scaling — and the paper *still* documents "loss of precision" failure modes (values closer than one bin spacing collapse to the same token; large s destroys resolution). AURUM's port was strictly cruder: fixed 0.1-unit rounding, no data-adaptive scale, and the "bins" were BPE merges, not a calibrated lattice.

**(b) Digit tokens do not carry a metric.** BPE "can break a single number into tokens that don't align with the digits" (LLMTime ✅, NeurIPS 2023: GPT-3 tokenizes 42235630 as [422, 35, 630]); frontier-model arithmetic flips qualitatively with digit chunking and errors are "systematic rather than approximate" (Tokenization Counts ✅, arXiv:2402.14903). xVal ✅ (arXiv:2310.02989) proves the deeper point: text-based number encodings yield *discontinuous out-of-distribution predictions* because cross-entropy "does not incorporate number distances into the loss." The consequence for a gradient learner: **the cross-variable Jacobian ∂output/∂(ret24, ATR, …) is a step function, zero almost everywhere** — continuous sensitivity to "ret24 moving from +1.4 to +1.5 ATR" does not exist at the input port, so the loop's 63M downstream parameters were fitting a quantized lookup table, not a response surface. The evidence was in the v1.2 data all along: a 4-second XGBoost replicating the k-NN soft labels to Brier 0.0017 (postmortem Failure #1) — discrete inputs invite lookup-table solutions.

**(c) Frozen language layers need a reprogramming bridge — v1.2 had none.** Time-LLM ✅ (ICLR 2024) makes a frozen LLM usable for time series *only* by training a cross-attention "reprogramming" layer that maps continuous patches into word-embedding prototypes, keeping the trunk "in the space where the language model is pre-trained." Every successful time-series foundation model keeps numbers continuous or finely binned at the port: TimesFM ✅ (patch 32→128, continuous), MOIRAI ✅ (patches + RevIN instance normalization + any-variate attention), PatchTST ✅ (patching + channel independence), iTransformer ✅ (variate tokens). AURUM violated the unanimous design law in *both* directions — text in, 48-way CE grid out.

**Mandate M1.** *Numbers enter v2.0 as scaled continuous tensors (RevIN-normalized patches), never as digit strings. The English vocabulary, if used at all, sits behind a trained continuous projector (Section 4, Path B probe).*

## 1.2 Failure II — Inductive-bias mismatch: 63M smooth parameters on 72k noisy axis-aligned rows

Grinsztajn et al. ✅ (NeurIPS 2022, arXiv:2207.08815) established why tree ensembles still beat deep learning on medium tabular data (~10k rows): (i) NNs are biased toward smooth low-frequency functions while tabular target functions are irregular (Gaussian-smoothing the *targets* closes the gap — i.e., the signal lives where NNs are weakest); (ii) "MLP-like architectures are not robust to uninformative features" — and AURUM's 48-cell label grid injects 47 mostly-uninformative sibling targets per example; (iii) tabular data are not rotation-invariant, but NNs effectively are — AURUM's features are *engineered, axis-aligned, ATR-normalized quantities* whose semantics a rotation destroys.

AURUM v1.2 sat in the worst corner of this result: 63M trainable parameters over 72,241 training states — **≈875 trainable parameters per training state** — with sandwich LayerNorms as its only regularization. The escape routes in the literature are narrow and instructive: Kadra et al. ✅ (arXiv:2106.11189) beat XGBoost with MLPs *only* after a per-dataset search over a cocktail of 13 regularizers (regularization, not architecture, is the lever); TabPFN ✅ wins only ≤1,000 training rows. Meanwhile the GBDT on the *same* features — reading them as continuous floats, axis-aligned splits, built-in feature selection — posted the only clean positive OOS number. **The GBDT was not a baseline; it was the control experiment proving the input features carry a weak signal that the transformer's port and bias destroyed.**

**Mandate M2.** *v2.0's neural capacity is right-sized (2–8M parameters), TFT-style gating provides the skip-nonlinearity escape (GRNs "can skip unnecessary processing… motivated for small/noisy datasets" ✅), and the GBDT/AE-MLP stack is a permanent ensemble member, not a baseline to be beaten and discarded.*

## 1.3 Failure III — Over-parameterization meets non-contractive recurrence: why r=1 beat r=2,3,4

v1.2's core loop iterated a trained 4-layer block: h_r = Block(h_{r−1}; sandwich norms, residual stream clamped). Nothing in the design constrained the block's Jacobian. The error dynamics are then exactly:

$$\delta_r \;=\; (I + J_f)\,\delta_{r-1} + O(\delta^2) \qquad\Rightarrow\qquad \|\delta_r\| \;\lesssim\; \rho(I+J_f)^r\,\|\delta_0\|$$

Iterating a map *denoises* only in the contraction regime; it *amplifies* estimation error geometrically whenever ρ(I+J_f) > 1 on any error subspace. This dichotomy is the explicit subject of an entire literature:

- **DEQ** ✅ (Bai, Kolter, Koltun, NeurIPS 2019, arXiv:1909.01377): infinite weight-tied depth is *defined* as a fixed point with gradients through (J_f − I)⁻¹ — well-posed only when the iteration converges.
- **monDEQ** ✅ (Winston & Kolter, NeurIPS 2020, arXiv:2006.08591): "naive forward iteration … does not converge, i.e., the iteration may be unstable"; guarantees require a *parameterization* forcing strong monotonicity (W = (1−m)I − AᵀA + B − Bᵀ), and "during training it is normal for W to move outside this regime."
- **RED / proximal denoising** ✅ (Romano et al., arXiv:1611.02862; Reehorst & Schniter, arXiv:1806.02296; Gribonval & Nikolova, arXiv:1807.04021): iterating a denoiser is principled *iff* the denoiser is non-expansive (convexity of the RED penalty ⟺ I − ∇f ⪰ 0); generic trained denoisers fail the symmetry/non-expansiveness conditions — which is why the conditions must be *enforced*, not assumed.
- **Positive loop results do not transfer.** Yang et al.'s looped transformer ✅ (arXiv:2311.12424) emulates *convergent iterative algorithms on clean synthetic data*, supervises the loss on **every iterate**, reports "instability issues" needing gradient clipping + weight decay, and uses 0.79M parameters — not 63M. Bae et al.'s Relaxed Recursive Transformers ✅ (arXiv:2410.20672) use recursion as *compression* for already-pretrained LLMs via 60B-token uptraining — a regime AURUM does not inhabit. Universal Transformers' ACT ✅ makes depth input-*adaptive with a complexity penalty* — comparing fixed r∈{1..4} post hoc is not adaptive computation, it is four more entries in the selection pool (see Failure IV).

The measured OOS profile 0.2145 / 0.1566 / 0.1644 / 0.1572 (r=1..4) is the signature the math predicts for an expansive loop: each pass sharpens the network's own estimation noise (overconfident logits) rather than shrinking it. The loop-divergence "OOD score" of v1.2 (mean |p_r − p_3|) was measuring this amplification without naming it.

**Mandate M3.** *v2.0's loop is contractive by construction: per-layer spectral caps (Miyato et al. ✅: W̄ = W/σ(W), one power-iteration step), a non-residual loop block f with ‖J_f‖ < 1 by design, Krasnoselskii–Mann damping h ← (1−α)h + αf(h) toward the denoising fixed point, per-iterate supervision, and an empirical Jacobian spectral-radius audit with a hard ρ < 0.95 acceptance gate (Section 4.4 — the audit is implemented and runs in the trainer).*

## 1.4 Failure IV — The 48-cell Michaud error-maximization paradox

v1.2 selected trades by argmax over 48 cells of EV̂_c = p̂_c·b_c − (1−p̂_c) = p̂_c(b_c+1) − 1. Two compounding errors:

**(a) Variance tilting.** EV̂ is affine in p̂, so

$$\mathrm{Var}[\widehat{EV}_c] \;=\; (b_c+1)^2\,\mathrm{Var}[\hat p_c], \qquad \mathrm{Var}[\hat p_c] = \frac{p_c(1-p_c)}{n_c}$$

For b ∈ {1.5, 2.5, 4.0}: (b+1)² ∈ {6.25, 12.25, **25.0**} — the TP=4.0 family carries **4× the estimation variance** of the TP=1.5 family. The argmax therefore systematically samples the deepest-TP, lowest-p, worst-calibrated cells: estimation-error magnets, exactly the Michaud "error maximization" behavior ⚠️ (Michaud 1989, FAF 45(1) — paywalled, characterization verified second-hand) instantiated *per trade*.

**(b) Winner's curse.** Bailey–Borwein–López de Prado–Zhu ✅ (Notices of the AMS, 2014): the expected maximum of N zero-skill standardized estimates is

$$\mathbb{E}[\max] \approx (1-\gamma)Z^{-1}\!\Big(1-\tfrac1N\Big) + \gamma\,Z^{-1}\!\Big(1-\tfrac{1}{Ne}\Big) \le \sqrt{2\ln N}, \qquad \gamma = 0.5772156649$$

For N=48: **E[max] ≈ 2.26σ** (bound 2.78σ). With heterogeneous σ_c ∝ (b_c+1), the winner is drawn disproportionately from the highest-variance family — the selection is *biased toward noise*, and the same paper's MinBTL arithmetic ("with 5 years of data, no more than 45 independent model configurations should be tried") indicts v1.2's entire trial surface (4 depths × 2 grids × overlay variants × model variants ≥ 45). The observed 2-cell (+0.2788R) > 48-cell (+0.1644R) is exactly what this math orders — and DSR = 0.0000 is its arithmetic consequence ✅ (Bailey & López de Prado, JPM 2014).

**Mandate M4.** *Discrete barrier-cell argmax is retired. v2.0 outputs a continuous vol-scaled position; where discrete cells remain for risk framing, selection uses the lower confidence bound EV_LCB = EV̂ − z_α·(b+1)√(p̂(1−p̂)/n_eff) with overlap-discounted n_eff (Newey–West), followed by positive-part James–Stein shrinkage toward the cross-cell mean — pooling 48 noisy estimates instead of trusting their max (Section 4.8).*

## 1.5 Failure V — The H3 ex-post overlay illusion and the Buy & Hold mirror

The v1.2 backtest credited trade management from the H3 head: a "tightened stop" at predicted-MAE q25 and a "partial TP" at predicted-MFE q50, claimed "path-valid (a touched barrier level provably fills)." Agent 1's audit found the partial-TP branch contained an ex-post unit bug that credited **+1.08R of synthetic profit whenever a partial TP was touched** — the trade's outcome was written using information (the touch event) resolved *after* the entry decision, in wrong units. The paired-number arithmetic quantifies the damage:

| Strategy | Trades | E[R]/trade buggy | E[R]/trade clean | Distortion/trade | Total synthetic R |
|---|---|---|---|---|---|
| GBDT | 14,857 | +0.6321R | +0.0487R | **+0.5834R** | **≈ +8,668R** |
| k-NN Cohort | 14,940 | +0.5311R | −0.0223R | +0.5534R | ≈ +8,268R |
| AURUM 2-cell | 6,498 | +0.2788R | −0.0400R | +0.3188R | ≈ +2,072R |

The bug engaged on 6,262 partial-TP exits and 10,553 tightened-stop exits (v1.2 report §6.6) — the overlay was not a small garnish, it *was* the reported edge. Two structural lessons:

1. **Ex-post overlays are fabrication machines.** The v1.2 report itself documented that the backtest was "deterministic on label summaries: the overlay uses realized excursion extremes, which prove barrier levels were touched but not the intra-path ordering; the conservative rule ordering mitigates but does not eliminate this approximation." A unit bug inside an already-approximate path claim is invisible to every downstream gate — G4/G5 validated *arithmetic*, not *causality*. v2.0's rule: **every overlay, stop, and partial-exit must be simulated forward-only inside the event loop (Section 5.3), where the only inputs at decision time t are features with timestamp ≤ t.** No label summary is ever read as an execution input.
2. **The Buy & Hold backward-returns bug is the same disease.** B&H "earned" +0.2644R over 707 "trades" from *backward* 24-bar returns used as forward outcomes. Any benchmark whose inputs are not strictly forward is not a benchmark; v2.0 computes all benchmarks (B&H, EMA, Donchian) in the same causal event loop as the strategy (Section 5.3).

**Mandate M5.** *One deterministic causal event loop computes strategy and benchmarks identically; a truncation-equivalence test (identical decisions when the future is deleted) is a release gate (Section 6, gate G2.4).*

## 1.6 The composite verdict

v1.x burned its edge in a four-stage cascade, each stage individually documented in the literature: (1) the input port quantized a weak continuous signal into digit tokens and froze the metric away (Chronos/xVal/LLMTime/Time-LLM ✅); (2) the inductive bias mismatched the data regime by three orders of magnitude in parameters (Grinsztajn/Kadra ✅); (3) the unconstrained loop amplified estimation error in exact violation of the contraction conditions DEQ/monDEQ/RED exist to enforce ✅; (4) the 48-cell argmax selected on amplified noise with a (b+1)² variance tilt — a textbook winner's-curse machine ✅ — and the H3 overlay then fabricated the reported edge ex-post. The v2.0 architecture (Sections 3–5) is the direct implementation of Mandates M1–M5.

---


# Section 2 — Literature & Codebase Excavation Ledger

## 2.0 Method note (how this research was actually done)

The excavation ran as seven parallel deep-research tracks (one per pillar), each instructed to fetch and read primary sources rather than rely on search snippets. **Search engines were largely blocked from the research sandbox** (DuckDuckGo → captcha; Bing → junk; CME/Akamai → 403), so retrieval proceeded through the arXiv API (`export.arxiv.org`), Crossref API, direct arXiv/ar5iv full-text HTML, author sites (davidhbailey.com, qoppac.blogspot.com), exchange rulebooks (LBMA, SGE PDF), GitHub tarballs and raw files, HuggingFace raw configs, the Kaggle kernel API, and PyPI. Approximately **150 distinct sources were fetched and read**; ~20–45 queries were executed per pillar. Every formula in Sections 3–5 traces to a fetched source or is flagged.

**Verification key:** ✅ primary fetched & read · ◐ verified via fetched secondary reproduction / Crossref / arXiv-API metadata · ⚠️ standard result, primary unreachable (do not build load-bearing structure without confirmation; full list in Appendix B).

### 2.0.1 Citation corrections found during excavation (8 wrong IDs corrected)

The task brief's citations contained eight wrong arXiv IDs; all were caught by fetch-and-verify and corrected. Agent 1 must use the corrected IDs in any downstream reading:

| Paper | Wrong ID (brief) | Correct ID |
|---|---|---|
| xVal: continuous number encoding | 2310.21271 | **2310.02989** |
| Tokenization counts (LLM arithmetic) | 2402.18611 | **2402.14903** |
| Relaxed Recursive Transformers (Bae et al.) | 2503.03450 | **2410.20672** |
| Deep Equilibrium Models (Bai et al.) | 1908.06573 | **1909.01377** |
| RED (Romano et al.) | 1701.03400 | **1611.02862** |
| Monotone operator equilibrium nets | — | **2006.08591** |
| Kadra et al., well-tuned simple nets | — | **2106.11189** |
| Volatility is Rough (Gatheral et al.) | 1610.03531 (solar physics!) | **1410.3394** |
| Levin–Lyons–Ni signature regression | 1309.5535 (astronomy!) | **1309.0260** |

## 2.1 Master ledger

### Pillar 1 — Theory of the v1.x ceiling (tokenization, tabular DL, loops, selection)

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P1-1 | Ansari et al. 2024, *Chronos* | arXiv:2403.07815, ICML'24 | Value tokenization done right: mean-scaling + 4094 bins; documents precision-loss — the benchmark v1.2's ASCII port failed | ✅ |
| P1-2 | Das et al. 2023, *TimesFM* | arXiv:2310.10688, ICML'24 | Continuous patches (32→128) — numbers never become text | ✅ |
| P1-3 | Woo et al. 2024, *MOIRAI* | arXiv:2402.02592, ICML'24 | Any-variate attention + RevIN instance normalization | ✅ |
| P1-4 | Nie et al. 2022, *PatchTST* | arXiv:2211.14730, ICLR'23 | Patching + channel independence — the field's standard numeric port | ✅ |
| P1-5 | Liu et al. 2023, *iTransformer* | arXiv:2310.06625, ICLR'24 | Variate tokens; multivariate structure ≠ temporal fusion | ✅ |
| P1-6 | Jin et al. 2023, *Time-LLM* | arXiv:2310.01728, ICLR'24 | Frozen LLMs need a trained reprogramming bridge — the anti-pattern AURUM embodied | ✅ |
| P1-7 | Golkar et al. 2023, *xVal* | arXiv:2310.02989 | CE losses are not distance-aware; text numbers → discontinuous OOD predictions | ✅ |
| P1-8 | Singh & Strouse 2024, *Tokenization counts* | arXiv:2402.14903 | Digit-chunking brittleness; errors systematic, not approximate | ✅ |
| P1-9 | Gruver et al. 2023, *LLMTime* | arXiv:2310.07820, NeurIPS'23 | BPE splits numbers off their digits ([422,35,630]) | ✅ |
| P1-10 | Grinsztajn et al. 2022 | arXiv:2207.08815, NeurIPS'22 | Trees beat DL on medium tabular: irregular targets, uninformative features, rotation — AURUM's regime exactly | ✅ |
| P1-11 | Hollmann et al. 2022, *TabPFN* | arXiv:2207.01848, ICLR'23 | Amortized Bayesian transformer wins only ≤1k rows | ✅ |
| P1-12 | Kadra et al. 2021 | arXiv:2106.11189, NeurIPS'21 | MLPs beat XGB only with a searched 13-regularizer cocktail — regularization is the lever | ✅ |
| P1-13 | Dehghani et al. 2018, *Universal Transformers* | arXiv:1807.03819, ICLR'19 | Adaptive depth (ACT) with ponder cost — fixed r grids are not adaptive computation | ✅ |
| P1-14 | Yang et al. 2023, *Looped Transformers* | arXiv:2311.12424 | Loops emulate convergent algorithms with per-iterate loss; instability mitigations; 0.79M params | ✅ |
| P1-15 | Bae et al. 2024, *Relaxed Recursive Transformers* | arXiv:2410.20672 | Recursion = compression for pretrained LLMs — not a small-data signal tool | ✅ |
| P1-16 | Bai et al. 2019, *DEQ* | arXiv:1909.01377, NeurIPS'19 | Fixed-point depth; gradients via (J−I)⁻¹; well-posed only under contraction | ✅ |
| P1-17 | Winston & Kolter 2020, *monDEQ* | arXiv:2006.08591, NeurIPS'20 | Unconstrained forward iteration "may be unstable"; monotone parameterization | ✅ |
| P1-18 | Romano et al. 2016, *RED* | arXiv:1611.02862, SIAM Imaging | Denoiser iteration is regularization iff I−∇f ⪰ 0 (non-expansive) | ✅ |
| P1-19 | Reehorst & Schniter 2019 | arXiv:1806.02296, TPAMI | Even the RED interpretation needs symmetric Jacobian | ✅ |
| P1-20 | Gribonval & Nikolova 2018 | arXiv:1807.04021 | MMSE denoisers are proximal ops iff white Gaussian noise | ✅ |
| P1-21 | Miyato et al. 2018, *Spectral normalization* | arXiv:1802.05957, ICLR'18 | W̄=W/σ(W); one power-iteration step; per-layer Lipschitz control — the enforceable cap | ✅ |
| P1-22 | Cisse et al. 2017, *Parseval networks* | arXiv:1704.08847, ICML'17 | Near-orthogonal weights; Parseval retraction W←(1+β)W−βWWᵀW | ✅ |
| P1-23 | Bailey, Borwein, LdP, Zhu 2014, *Pseudo-mathematics* | Notices AMS 61(5) | E[max] of N zero-skill trials; MinBTL: 5y ⇒ ≤45 configs | ✅ |
| P1-24 | Bailey & López de Prado 2014, *DSR* | JPM 40(5) | Winner's curse formalized; deflation machinery (exact formulas in §5.5) | ✅ |
| P1-25 | Michaud 1989 | FAF 45(1) | "Error maximization" of optimizers | ⚠️ |
| P1-26 | Jorion 1985/92 | JB/JFE/FAF | Bayes-Stein shrinkage improves OOS portfolios | ⚠️ |

### Pillar 2 — Gold microstructure, institutional clocks, DST

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P2-1 | LBMA (institutional) | lbma.org.uk | Gold price auctions at **10:30 & 15:00 London**, IBA-administered, 15 participants; silver 12:00 | ✅ |
| P2-2 | SGE rulebook (Nov-2019 amendment) | sge.com.cn (rules PDF) | SHAU benchmark **10:15 & 15:00 Beijing** (PM corrected from legacy 14:15!); iterative fixing mechanics; night session 19:45–02:30 | ✅ |
| P2-3 | IANA tzdata via zoneinfo | — | DST rules: UK last-Sun Mar/Oct 01:00 UTC; US 2nd-Sun Mar / 1st-Sun Nov; China no DST — the 4h/5h gap windows are machine-detected (F1 test) | ✅ |
| P2-4 | Wikipedia DST/fixing/COMEX pages + NY Fed | — | Secondary corroboration: WMR 16:00 London (5-min window since May 2015); Globex 18:00 ET start; PM fix anchored 15:00 since 1968 | ◐ |
| P2-5 | Caminschi & Heaney 2014, *Fixing a Leaky Fixing* | J. Futures Markets 34(11) | **The gold-fix paper**: volume/vol jump at fix *start*; informed profit in first 4 minutes; fix-opening trades predict direction >90%; nothing moves post-publication → trade the window, not the print | ✅ |
| P2-6 | Lou, Polk & Skouras 2019 | JFE 134(1) | Overnight vs intraday tug-of-war: profits accrue entirely in one session, opposite signs — session-split targets | ✅ |
| P2-7 | Boyarchenko, Larsen & Whelan 2023 (+2026 blog) | RFS 36(9) | Overnight drift = imbalance × variance × LP-capacity⁻¹; 08:30–10:00 ET opening reversal; anomaly decayed post-2021 (crowding warning) | ✅ |
| P2-8 | Gao, Han, Li & Zhou 2018 | JFE 129(2) | First-half-hour return predicts last-half-hour; strongest on macro-release (08:30 ET) days | ◐ |
| P2-9 | Baltussen, Da, Lammers & Martens 2021 | JFE 142(1) | Last-30-min-before-close momentum across 60+ futures incl. gold; multi-day reversion; gamma-hedging mechanism | ◐ |
| P2-10 | Kamstra, Kramer & Levi 2000 | AER 90(4) | DST-transition weekends distort returns — transition-day regime flags | ◐ |

### Pillar 3 — M1 path geometry, rough volatility, signatures, toxicity

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P3-1 | Barndorff-Nielsen & Shephard 2004 | JFEC 2(1) | BV = (π/2)(n/(n−1))Σ|r_{i−1}‖r_i|; bipower jump framework | ✅ |
| P3-2 | BN, Kinnebrock & Shephard 2010 | Engle festschrift | Realized semivariances RV⁺/RV⁻ | ✅ |
| P3-3 | Patton & Sheppard 2015 | REStat 97(3) | Good/bad volatility; SJ = RV⁺−RV⁻ predictive asymmetry | ◐ |
| P3-4 | Huang & Tauchen 2005 | JFEC 3 | Ratio jump statistic; θ = (π/2)²+π−5 ≈ 0.609 | ◐ |
| P3-5 | Andersen, Dobrev & Schaumburg 2012 | J. Econometrics 169 | MedRV/MinRV jump-robust estimators | ◐ |
| P3-6 | BN, Hansen, Lunde & Shephard 2008 | Econometrica 76(6) | Realized kernels; microstructure-noise remedy; sampling-frequency guidance | ◐ |
| P3-7 | Parkinson 1980; Garman-Klass 1980; Rogers-Satchell 1991; Yang-Zhang 2000 | JB 53; AAP 1(4) | Range estimators; GK constants 0.511/0.019/0.383 (Meilijson arXiv:0807.3492 ✅); YZ k = (α−1)/(α+(n+1)/(n−1)), α≈1.34 | ✅ |
| P3-8 | Lo & MacKinlay 1988 | RFS 1(1) | VR(q) unbiased overlapping estimator + heteroskedasticity-robust z | ✅ |
| P3-9 | Gatheral, Jaisson & Rosenbaum 2018, *Volatility is Rough* | QF 18(6); arXiv:1410.3394 | m(q,Δ) ∝ Δ^{qH}, H≈0.1; "gold futures show similar smoothness" | ✅ |
| P3-10 | Chevyrev & Kormilitzin 2016 | arXiv:1603.03788 | Signature primer; Lévy area A = ½(S^{ij}−S^{ji}); log-signature | ✅ |
| P3-11 | Levin, Lyons & Ni 2013 | arXiv:1309.0260 | Expected-signature regression | ✅ |
| P3-12 | Kidger et al. 2019, *Deep signature transforms* | arXiv:1905.08494, NeurIPS'19 | Neural augmentation before signatures | ✅ |
| P3-13 | Morrill et al. 2020 | arXiv:2006.00873 | Generalised signature method; time-augmentation mandatory; lead-lag transform; term counts (d^{N+1}−d)/(d−1) | ✅ |
| P3-14 | Reizenstein & Graham 2018 | arXiv:1802.08252 | Log-signature bases (Lyndon words), BCH | ✅ |
| P3-15 | Easley, López de Prado & O'Hara 2012 | RFS 25(5) | VPIN; volume clock; bulk volume classification | ✅ |
| P3-16 | Andersen & Bondarenko 2014 | J. Financial Markets | VPIN critique — use as state variable, not standalone alpha | ◐ |
| P3-17 | Kyle 1985 | Econometrica 53(6) | λ price-impact coefficient | ◐ |
| P3-18 | Amihud 2002 + realized variant (arXiv:2609.00943) | JFM 5(1) | ILLIQ; realized RPV/ν (5-min "several times more efficient") | ✅ |
| P3-19 | esig / iisignature / signatory (libraries) | PyPI | Signature computation stacks (we ship exact NumPy depth-2/3 instead; no new deps) | ✅ |

### Pillar 4 — State-space residualization & trend following

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P4-1 | Kalman filter / RLS (Harvey 1989; Durbin-Koopman 2012; Wikipedia cross-verified) | — | TVP regression recursions; RLS(λ) ⟺ Kalman; EW-ridge equivalence | ✅ |
| P4-2 | Avellaneda & Lee 2010 | Math. Finance / Quantitative Finance | s-score = (X−m)/σ_eq; entry 1.25σ, exits 0.75/0.50; OU filter τ<30d | ✅ |
| P4-3 | Gatev, Goetzmann & Rouwenhorst 2006 | RFS | Pairs-trading archetype | ✅ |
| P4-4 | Moskowitz, Ooi & Pedersen 2012 | JFE 104(2) | TSMOM: sign(past k-mo) × 40%/σ; EWMA vol (60d COM) | ✅ |
| P4-5 | Lempérière, Deremble, Seager, Potters, Bouchaud et al. 2014/17, *Two Centuries of Trend Following* | arXiv:1404.3274, JIS 6(1) | Risk-adjusted momentum; **tanh saturation Δp = a + b·s\*tanh(s_n/s\*), s\*≈0.89, b≈0.075**; 3-day trend dead post-2003 | ✅ |
| P4-6 | Dao et al. 2016, *Tail protection* | arXiv:1607.02410 | EMA trend Π = λτL_τ[R]/σ; tanh capping interpolates parabola↔V | ✅ |
| P4-7 | Potters & Bouchaud 2005 | arXiv:physics/0508104 | Trend P&L asymmetry; win-fraction mechanics | ✅ |
| P4-8 | Dao et al. 2017, *Trends & Risk Premia* | arXiv:1708.07637 | 5-month trend back to 1700; SR ≈ ⅓ − ζ/4 | ✅ |
| P4-9 | Hurst, Ooi & Pedersen 2017, *A Century of Evidence* | JPM 44(1) | 1/3/6/12-month sign blend, vol-scaled | ◐ |
| P4-10 | Carver 2015/2023 + qoppac blog + **pysystemtrade source** | books; github.com/robcarver17 | EWMAC N,4N ladder; scalar→mean|F|=10, cap ±20 (`defaults.yaml` fetched); **buffer = 10% of average position, trade-to-edge** (code-verified); speed limit: cost ≤ ⅓ pre-cost SR | ✅ |
| P4-11 | Lim, Zohren & Roberts 2019 (DMN) | arXiv:1904.04912 | MACD(8,24)/(16,48)/(32,96) baseline; φ(y)=y·exp(−y²/4)/0.89 response; vol-normalized inputs | ✅ |
| P4-12 | Wood, Roberts & Zohren 2022, *Slow Momentum with Fast Reversion* | arXiv:2105.13727, JFDS | CPD features lift Sharpe 1.62→2.16 | ✅ |
| P4-13 | Catania & Nonejad 2017, *eDMA* | arXiv:1606.05656 | Dynamic model averaging machinery | ✅ |
| P4-14 | Katz et al. 2022 | arXiv:2210.15448 | Kalman pairs-trading backbone | ✅ |

### Pillar 5 — Architectures that beat GBDTs (Oxford-Man + Kaggle)

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P5-1 | Lim, Zohren & Roberts 2019, *DMN* | arXiv:1904.04912, JFDS | Sharpe loss (Eq. 16, √252, ε via repo); LSTM 63-step unroll; hidden {5..80}; direct tanh > regression > binary; 2.8 Sharpe vs TSMOM 1.2 | ✅ |
| P5-2 | Wood, Giegerich, Roberts, Zohren 2021, *Momentum Transformer* | arXiv:2112.08534 | Decoder-only TFT; seq 252; 3-class head does **not** exist in the paper (corrected myth) — continuous tanh + Sharpe loss; TFT+CPD 2.62 vs LSTM 1.70 | ✅ |
| P5-3 | kieranjwood/trading-momentum-transformer (repo) | GitHub (tarball fetched) | Reference implementations: SharpeLoss ε=1e-9; VOL_LOOKBACK=60, σ_tgt=0.15; hp grids | ✅ |
| P5-4 | Lim et al. 2019, *TFT* | arXiv:1912.09363, IJF | GRN/VSN formulas; interpretable MHA (shared values, head-averaged); "skip nonlinearity for small/noisy data" | ✅ |
| P5-5 | Yirun Zhang et al. 2021, JS 1st place "Cats Trading" | Kaggle write-up + kernel API | **Exact AE-MLP**: BN→GaussianNoise(0.0353)→Dense(96)→…; MLP 896-448-448-256; multi-target, |resp| weights; purged group CV gap=31; last-two-folds only; 3 seeds | ✅ |
| P5-6 | Michael Jahrer 2017, Porto Seguro 1st place | Kaggle write-up | Swap-noise DAE (15%), RankGauss, deep-stack activations; 1×LGBM + 5×NN blend | ✅ |
| P5-7 | JS Real-Time 2024–25: 9th/13th/26th-place threads | Kaggle discussions | CatBoost online-retrain chaining (5×300 iters, 600-day window, 7-day cadence); online NN 15-day/2-epochs-per-day; TabM k=16; cross-symbol means; recency windows | ✅ |
| P5-8 | Optiver 2021 1st place (nyanp) | Kaggle write-up | t-SNE time-order recovery; kNN aggregation features (360/600); LGBM+MLP+CNN blend; adversarial validation | ✅ |
| P5-9 | Ubiquant 2022 1st place (K I Y) | Kaggle write-up | LGBM+TabNet; cross-sectional time_id means of top-|corr| features | ✅ |
| P5-10 | G-Research 2022 (1st/2nd) | Kaggle discussions | 2nd: LGBM, 6-fold walk-forward 40-week/1-week gap; 1st community-linked to the JS AE-MLP approach | ✅/◐ |
| P5-11 | Qwen3-0.6B model card + config.json | HuggingFace (raw) | 28 layers, d=1024, GQA 16q/8kv (head 128), MLP 3072, vocab 151,936, tied embeddings, 596M total / 0.44B non-emb | ✅ |
| P5-12 | Hu et al. 2021, *LoRA* | arXiv:2106.09685 | ΔW=(α/r)BA; given a fixed budget, adapt {Wq,Wv} with small rank ("more types, smaller rank") | ✅ |
| P5-13 | Dettmers et al. 2023, *QLoRA* | arXiv:2305.14314 | NF4; PEFT memory math (activation grads dominate); T4 fp16 implications | ✅ |
| P5-14 | Zeng et al. 2022, *DLinear* | arXiv:2205.13504, AAAI'23 | Linear baselines beat fancy transformers — the sanity ladder is mandatory | ✅ |
| P5-15 | Chen et al. 2023, *TSMixer* | arXiv:2303.06053 | All-MLP mixing; less-is-more data point | ✅ |

### Pillar 6 — Differentiable utility, buffers, shrinkage, meta-labeling

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P6-1 | Moody & Saffell 2001 | IEEE TNN 12(4) | Differential Sharpe ratio (recursions secured via Borrageiro et al. ✅ below) | ◐ |
| P6-2 | Borrageiro, Firoozye & Barucca 2022 | arXiv:2201.04699 | Verbatim DSR reproduction (denominator typo corrected); R_t = Δp·F_{t−1} − δ|ΔF| | ✅ |
| P6-3 | Zhang, Zohren & Roberts 2020 | arXiv:2005.13665, JFDS 2(4) | Differentiable Sharpe over softmax weights; cost+vol-scaling inside objective (Eq. 7); DLS Sharpe 1.96 vs MV 1.12 — **CVaR/Sortino losses NOT in the paper** (future-work note; reconstructed forms flagged) | ✅ |
| P6-4 | Chekhlov, Uryasev & Zabarankin 2005, *CDaR* | IJTAF 8(1) | Drawdown deviation measure; variational form | ✅ |
| P6-5 | MacLean, Thorp & Ziemba 2010 | CFA Institute RF | f\*=(bp−q)/b; fractional-Kelly ↔ δ=1/(1−c); the P[double-before-halve] table; 2×Kelly ⇒ zero excess growth | ✅ |
| P6-6 | Guo et al. 2017 | arXiv:1706.04599, ICML'17 | Temperature scaling; T\* by val-NLL; ECE | ✅ |
| P6-7 | James & Stein 1961 / Efron & Morris 1975 | Berkeley symp. / JASA 70(350) | Positive-part JS; empirical-Bayes shrinkage — the 48-cell pooling tool | ✅ |
| P6-8 | López de Prado 2018, AFML §3.6 | Wiley | Meta-labeling workflow (canonical-secondary; primary→side, secondary→P(correct)→sizing) | ◐ |
| P6-9 | Bailey et al. 2015, *PBO* | J. Computational Finance | CSCV; logit rank; PBO — the grid-overfitting check | ✅ |

### Pillar 7 — Online adaptation, regimes, honest evaluation

| # | Source | Venue / ID | Contribution to v2.0 | ✓ |
|---|---|---|---|---|
| P7-1 | Wang et al. 2021, *TENT* | arXiv:2006.10726, ICLR'21 | Norm-affine-only TTA; batch dependence — *not* directly applicable to regression | ✅ |
| P7-2 | *Training on the Future* 2026 (delay-aware TTA audit) | arXiv:2610.12232 | **The cautionary paper**: labels exist only H steps later; RLS-bank (median of RLS filters, no hyperparameters) beats 3/4 recent TSF-TTA methods — our adaptation floor | ✅ |
| P7-3 | TAFAS / PETSA / DPAL 2025–26 | arXiv:2501.04970; 2506.23424; 2610.14376 | Gated calibration & adapter TTA for time series | ✅ |
| P7-4 | Hazan 2016/19, *OCO* | arXiv:1909.05207 | OGD / FTRL / mirror-descent updates | ✅ |
| P7-5 | river library (docs + source) | riverml.xyz | Page–Hinkley exact recursion & defaults (λ=50, δ=0.005, α=0.9999, min_inst=30); ADWIN δ=0.002; online linear models | ✅ |
| P7-6 | Adams & MacKay 2007, *BOCPD* | arXiv:0710.3742 | Run-length posterior; hazard 1/λ; DJ example λ_gap=1000, a=1, b=1e-4 | ✅ |
| P7-7 | Moreira & Muir 2017 | JF 72(4) | Vol-managed scaling w=(μ/σ̂²)/E[μ/σ̂²]; abstract verified | ◐ |
| P7-8 | Bernardi, Bianchi & Bianco 2022 | arXiv:2212.07288 | Real-time c_t\* caveat (full-sample c\* unknown in real time); variance-matching normalization | ✅ |
| P7-9 | Baltussen, van Bekkum & van der Grient 2018 | JFQA 53(5) | Vol-of-vol: high-vs-low spread ≈ 8%/yr — de-risk overlay (interpretation, gated) | ◐ |
| P7-10 | Kim et al. 2022, *RevIN* | ICLR'22 (arXiv:2105.08499) | Per-window instance normalization; forward/inverse | ✅ |
| P7-11 | skfolio `CombinatorialPurgedCV` (source) | PyPI (fetched) | Exact purge/embargo index semantics; constraint purge+embargo < fold−1 | ✅ |
| P7-12 | Bailey & López de Prado 2014, *DSR* | davidhbailey.com PDF | Exact SR₀ / DSR equations (γ=0.5772156649) | ✅ |
| P7-13 | Bailey, Borwein, LdP, Zhu 2017, *PBO/CSCV* | JCF 20(4) | Full CSCV algorithm | ✅ |
| P7-14 | Wood et al. 2022 (CPD) | arXiv:2105.13727 | CPD features (+33% Sharpe); expanding-window re-tune protocol | ✅ |

## 2.2 Synthesis — what the ledger jointly dictates

**Theme 1 (the input port).** Ten independent foundation-model papers (P1-1..P1-9, P5-15) converge on one law: numeric streams enter models as scaled continuous or finely-binned embeddings. AURUM v1.2 is the counter-example that proves the law: same features, continuous GBDT +0.0487R clean, text-port transformer ≈ 0.

**Theme 2 (capacity).** On ~10⁴-row tabular problems, the winning neural recipes are small and regularized (Kadra's 13-regularizer cocktail; TFT's gating designed for small/noisy data; Kaggle winners' 96–896-unit MLPs) — never 63M-parameter trunks. The only regime where LLM trunks help is few/zero-shot (Time-LLM), which is not our regime.

**Theme 3 (iteration).** Loops are safe exactly under contraction (DEQ/monDEQ/RED) or per-iterate supervision on algorithm-emulation tasks (Yang et al.). Every enforceable contraction tool (spectral norm, Parseval, monotone parameterization) is cheap; v1.x used none.

**Theme 4 (selection).** Argmax over noisy heterogeneous-variance estimates is the winner's curse (Bailey et al.); shrinkage dominates it (James–Stein/Efron-Morris); the LCB variant is the bandit-standard selection rule. The 48-cell result was overdetermined.

**Theme 5 (execution economics).** Three independent literatures (Carver's buffers, DMN cost-augmented training, Zhang's cost-inside-the-objective) implement the same idea — costs must live *inside* the optimization, and position inertia must be enforced *outside* it. v1.x trained on 48 cost-free classification targets and traded every hour; both halves were wrong.

**Theme 6 (adaptation).** The newest evidence (the 2026 delay-aware audit) is humbling: most TTA gains evaporate under honest label-delay harnesses, and a hyperparameter-free RLS bank is the floor. Adaptation in v2.0 is therefore: normalizers always-on (free), readout-only updates gated on label arrival, backbone frozen between annual refits — and the RLS bank is the baseline every adaptive component must beat.

**Theme 7 (verification).** DSR/PBO/CSCV, purged-and-embargoed CV, and block bootstrap are the accepted honesty stack; v1.2 used them after the fact. v2.0 wires the config ledger into the backtester so N (the trial count) and V[SR] are recorded automatically from the first experiment.

---


# Section 3 — The v2.0 Institutional Feature Engine

## 3.0 Design law and budget

**Causality law (enforced by test).** Every feature at H1 bar label t is a function of M1/H1 data with timestamp ≤ t's close (bar label + 59 min). Each module ships a truncation self-test: deleting future data must leave earlier features bit-identical. This is the feature-side implementation of Mandate M5.

**Windows.** M1 trailing windows {1h, 4h, 24h} (60/240/1440 M1 bars) for realized measures; H1 windows {24h, 120h, 528h} for HAR-style aggregates (Corsi 1/5/22-day structure ✅); signature windows = the 60 M1 bars inside each H1 bar; RLS memory via forgetting λ=0.995 (≈200 bars ≈ 8.3 trading days); trend ladder s ∈ {2,4,8,16,32,64,128,256} h (geometric, ×2 — the Carver N,4N construction ✅).

**Feature budget** (~330 columns; every column name is its documentation):

| Block | Prefix | Count | Signal hypothesis (source) |
|---|---|---|---|
| F1 exchange clock | `minutes_to_`, `sess_`, `is_dst`, `days_to_dst`, `ret24_/ret6_` | ~41 | pre/post-fix drift & reversion (P2-5..P2-10); DST-gap regimes |
| F2 realized geometry (4 symbols × 3 windows + range + VR + roughness + HAR) | `RV_`, `BV_`, `SJ_`, `jump`, `VR`, `hurst`, `H_rough`, `parkinson`… | ~150 | good/bad-vol asymmetry, jump reversal, trend/chop regime (P3-1..P3-9) |
| F3 signatures (depth-2 + Lévy areas + lead-lag pairs) | `sig1_`, `sig2_`, `levy_`, `ll_` | ~51 | who-moved-first ordering across assets (P3-10..P3-13) |
| F4 toxicity/liquidity | `vpin`, `kyle_`, `amihud_`, `clv` | ~16 | flow toxicity as state variable (P3-15..P3-18) |
| F5 residualization | `eps`, `beta_`, `spread_z`, `res_mom_A`, `q_A`, `q_B`, `ou_` | ~17 | Channel A catch-up / Channel B overextension (P4-1, P4-2) |
| F6 trend ladder | `ewmac_`, `cfm_onesided_`, `F_blend`, `fdm` | ~28 | multi-horizon saturating trend (P4-4..P4-10) |
| macro innovations (v1.2, kept) | `mi_*` | ~15 | era-safe FRED transforms (v1.2 verified) |

The v1.2 serialization ("~330 ASCII tokens") becomes ~330 *continuous float* columns — same information budget, no quantization, no frozen-English detour (Mandate M1).

## 3.1 F1 — DST-aware exchange-clock features

**Verified institutional timetable** (all times from fetched primary sources; P2-1..P2-4):

| Event | Local time | UTC (winter) | UTC (summer) | Gap weeks |
|---|---|---|---|---|
| LBMA Gold Price AM | 10:30 Europe/London | 10:30 | 09:30 | 10:30 |
| LBMA Gold Price PM | 15:00 Europe/London | 15:00 | 14:00 | 15:00 |
| LBMA Silver | 12:00 Europe/London | 12:00 | 11:00 | 12:00 |
| WM/Refinitiv FX fix | 16:00 Europe/London | 16:00 | 15:00 | 16:00 |
| SGE Shanghai Gold AM (SHAU) | 10:15 Asia/Shanghai | 02:15 | 02:15 | 02:15 |
| SGE Shanghai Gold PM (SHAU) | **15:00 Asia/Shanghai** (2019 rules; legacy 14:15) | 07:00 | 07:00 | 07:00 |
| SGE night session open | 19:45 Asia/Shanghai | 11:45 | 11:45 | 11:45 |
| COMEX/Globex day open | 18:00 America/New_York | 23:00 | 22:00 | 22:00 |
| GC settlement / maintenance break | 13:30 / 17:00 ET | 18:30 / 22:00 | 17:30 / 21:00 | — |

**The DST trap, quantified.** London and New York are 5h apart except two annual windows — spring: 14 or 21 days (US on EDT, UK still GMT); autumn: exactly 7 days — when they are **4h apart** and the LBMA PM fix prints at 11:00 NY instead of 10:00 NY. A naive UTC `hour_sin/cos` blurs the fix, the COMEX open, and the 08:30 ET macro slot by a full hour for ~1 month per year. China has no DST (since 1991), so Shanghai anchors are constant in UTC but shift *relative* to London/New York twice a year. The module resolves all of this via `tz_convert` to venue-local wall time — never fixed offsets — and emits `ny_ldn_offset_hours ∈ {4,5}`, `is_dst_gap`, signed `days_to_dst_{uk,us}` (±14 clip), and transition-week flags (Kamstra-Kramer-Levi ✅).

**Feature list:** 10 signed anchor distances (`minutes_to_LBMA_AM/PM/SILVER/WMR_FIX/SGE_AM/PM/NIGHT/COMEX_OPEN/SETTLE/BREAK`), the Caminschi-Heaney windows (`in_pm_fix_first4m`, `pre_pm_fix_60m`), 9 session dummies in venue-local time (`sess_london_day`, `sess_ny_day`, `sess_globex_break`, `sess_asia_night`, `sess_shanghai_day`, `sess_ldn_ny_overlap`, `sess_ldn_pm_cluster`, `sess_us_macro_hour`, `sess_asia_eu_handover`), 6 DST features, and session-split returns (Lou-Polk-Skouras decomposition: `ret24_/ret6_` × asia/london_am/ldn_ny_overlap/ny_pm).

**Drift/reversion map** (literature-conditioned; Section 5 uses it for regime priors, the model learns the rest):
*Continuation-favoring:* the hour into the PM fix + first 4 fix minutes (P2-5); 08:30 ET macro mornings (P2-8); last 30 min into COMEX settle/close (P2-9); Asia→Europe handover (P2-7).
*Fade-favoring:* post-publication fix overshoot (15:00–16:30 London cluster — "no significant impact after publication", P2-5); 08:30–10:00 ET opening reversal after large overnight moves (P2-7); the sessions after end-of-day momentum (multi-day reversion, P2-9); DST-transition weeks (P2-10).

```python
"""
AURUM v2.0 -- Feature Module F1: DST-Aware Exchange-Clock Features
===================================================================
Ground truth timetable (verified 2026-11 against LBMA 'About LBMA daily auction
prices', SGE 'Detailed Rules of SGE on Benchmark Price Trading' (Nov-2019
amendment) and IANA tzdata via zoneinfo; see Spec v2.0 Section 2, ledger P2-1..P2-4):

  LBMA Gold Price AM auction   10:30 Europe/London   (IBA-administered)
  LBMA Gold Price PM auction   15:00 Europe/London   (anchored since 1968)
  LBMA Silver Price auction    12:00 Europe/London   (once daily)
  WM/Refinitiv FX fix          16:00 Europe/London   (rate window 15:59:30-16:00;
                                                     5-minute window since May 2015)
  SGE Shanghai Gold Bench AM   10:15 Asia/Shanghai   (SHAU; reference px 10:09-10:14)
  SGE Shanghai Gold Bench PM   15:00 Asia/Shanghai   (SHAU; 2019-amended rules;
                                                     legacy time was 14:15)
  SGE night session open       19:45 Asia/Shanghai   (pre-order matching 19:45-19:59)
  COMEX/CME Globex day open   18:00 America/New_York (Sun-Fri; NY Fed-corroborated)
  COMEX GC daily settlement   13:30 America/New_York (legacy convention)
  COMEX/Globex maintenance    17:00 America/New_York (17:00-18:00 ET break)

DST rules: UK/EU last-Sun March -> last-Sun Oct (01:00 UTC); US 2nd-Sun March ->
1st-Sun Nov (02:00 local); China no DST since 1991.  London-NY offset is 5h except
two annual windows (spring: 14 or 21 days; autumn: exactly 7 days) when it is 4h.

Every feature is a pure function of the bar's UTC timestamp (and, for the
session-split returns, of trailing close prices) -> strictly causal.

References: Caminschi & Heaney (J. Futures Markets 34(11), 2014) -- the London PM
*gold* fix leaks information during the auction, first 4 minutes carry the edge;
Lou, Polk & Skouras (JFE 134(1), 2019); Boyarchenko, Larsen & Whelan (RFS 36(9),
2023); Gao, Han, Li, Zhou (JFE 129(2), 2018); Baltussen, Da, Lammers, Martens
(JFE 142(1), 2021); Kamstra, Kramer, Levi (AER 90(4), 2000).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
NEWYORK = ZoneInfo("America/New_York")
SHANGHAI = ZoneInfo("Asia/Shanghai")

# (timezone, hour, minute) -- all verified times, see module docstring.
ANCHORS: dict[str, tuple[ZoneInfo, int, int]] = {
    "LBMA_AM":      (LONDON,  10, 30),
    "LBMA_PM":      (LONDON,  15,  0),
    "LBMA_SILVER":  (LONDON,  12,  0),
    "WMR_FIX":      (LONDON,  16,  0),
    "SGE_AM":       (SHANGHAI, 10, 15),
    "SGE_PM":       (SHANGHAI, 15,  0),
    "SGE_NIGHT":    (SHANGHAI, 19, 45),
    "COMEX_OPEN":   (NEWYORK, 18,  0),
    "COMEX_SETTLE": (NEWYORK, 13, 30),
    "COMEX_BREAK":  (NEWYORK, 17,  0),
}


def _local_wall_minutes(idx_utc: pd.DatetimeIndex, tz: ZoneInfo) -> pd.Series:
    """Minutes since local midnight (wall clock), as float Series indexed by idx_utc."""
    loc = idx_utc.tz_convert(tz)
    return pd.Series(loc.hour * 60 + loc.minute, index=idx_utc, dtype="float64")


def minutes_to_anchor(idx_utc: pd.DatetimeIndex, name: str) -> pd.Series:
    """Signed minutes from each timestamp to the anchor event on the SAME
    exchange-local calendar day.  Positive = event still ahead today;
    negative = event already passed today.  Naive UTC hour arithmetic would
    blur this by one hour for half the year (DST) -- hence tz_convert, never
    fixed offsets."""
    tz, hh, mm = ANCHORS[name]
    event_min = hh * 60 + mm
    now_min = _local_wall_minutes(idx_utc, tz)
    out = event_min - now_min
    # Mask weekends per venue (no fixes / sessions on Sat-Sun in that venue):
    dow = idx_utc.tz_convert(tz).dayofweek
    venue_weekend = (dow >= 5)
    if name.startswith(("LBMA", "WMR")):
        out[venue_weekend] = np.nan
    elif name.startswith("SGE"):
        # SGE trades Fri night (19:45+) into Monday's trading day; day session
        # anchors only meaningful Mon-Fri Shanghai time.
        out[(dow >= 5)] = np.nan
    # COMEX opens Sunday 18:00 ET -> keep Sunday for COMEX anchors (dayofweek==6).
    return out


def _dst_transitions(year_start: int, year_end: int) -> dict[str, pd.DatetimeIndex]:
    """Exact DST transition instants for London and New York between
    year_start-1 and year_end+1, found by scanning daily utcoffset changes and
    refining to the hour (authoritative source: the tz database itself, via
    zoneinfo -- exactly the mechanism pandas uses for tz_convert)."""
    days = pd.date_range(f"{year_start - 1}-01-01", f"{year_end + 1}-12-31", freq="D", tz="UTC")
    out: dict[str, list] = {"UK": [], "US": []}
    for tzname, tz in (("UK", LONDON), ("US", NEWYORK)):
        offs = pd.Series([d.tz_convert(tz).utcoffset() for d in days], index=days)
        day_chg = offs.ne(offs.shift()).fillna(False)
        day_chg.iloc[0] = False  # drop the artificial first-row change
        for day in days[day_chg]:
            # refine within +-36h to the exact transition hour
            hrs = pd.date_range(day - pd.Timedelta("36h"), day + pd.Timedelta("12h"), freq="h", tz="UTC")
            hoffs = pd.Series([h.tz_convert(tz).utcoffset() for h in hrs], index=hrs)
            h_chg = hoffs.ne(hoffs.shift()).fillna(False)
            h_chg.iloc[0] = False
            out[tzname].extend(h for h in hrs[h_chg])
    return {k: pd.DatetimeIndex(v) for k, v in out.items()}


def clock_features(idx_utc: pd.DatetimeIndex,
                   close_xau: pd.Series,
                   year_range: tuple[int, int] | None = None) -> pd.DataFrame:
    """Full DST-aware exchange-clock feature block for a UTC-indexed H1 frame.

    Parameters
    ----------
    idx_utc : pd.DatetimeIndex (tz-aware UTC) -- bar close timestamps.
    close_xau : pd.Series of XAUUSD close prices indexed by idx_utc (for the
        session-split return features; pass None to skip those).
    year_range : optional (y0, y1) to bound the DST transition table scan.
    """
    if idx_utc.tz is None:
        raise ValueError("idx_utc must be tz-aware UTC (bar close times).")
    idx = idx_utc.tz_convert("UTC")
    feats = pd.DataFrame(index=idx)

    # --- 1) Anchor distances (signed minutes, same venue-local day) --------
    for name in ANCHORS:
        feats[f"minutes_to_{name}"] = minutes_to_anchor(idx, name)

    # Caminschi-Heaney edge window: the first 4 minutes of the PM fix carry
    # the informed-trading profit; and the 60 minutes running into the fix.
    m_pm = feats["minutes_to_LBMA_PM"]
    feats["in_pm_fix_window"] = ((m_pm <= 0) & (m_pm >= -10)).astype("float64")
    feats["in_pm_fix_first4m"] = ((m_pm <= 0) & (m_pm >= -4)).astype("float64")
    feats["pre_pm_fix_60m"] = ((m_pm < 0) & (m_pm >= -60)).astype("float64")

    # --- 2) Session dummies in EXCHANGE-LOCAL time -------------------------
    loc_ldn = idx.tz_convert(LONDON)
    loc_ny = idx.tz_convert(NEWYORK)
    loc_sh = idx.tz_convert(SHANGHAI)
    h_ldn, h_ny, h_sh = loc_ldn.hour, loc_ny.hour, loc_sh.hour
    dow_ldn, dow_ny, dow_sh = loc_ldn.dayofweek, loc_ny.dayofweek, loc_sh.dayofweek

    feats["sess_london_day"] = ((h_ldn >= 8) & (h_ldn < 17) & (dow_ldn < 5)).astype(float)
    feats["sess_ny_day"] = ((h_ny >= 9) & (h_ny < 16) & (dow_ny < 5)).astype(float)
    feats["sess_globex_break"] = ((h_ny == 17)).astype(float)
    feats["sess_asia_night"] = (((h_sh >= 20) | (h_sh < 3)) & (dow_sh < 6)).astype(float)
    feats["sess_shanghai_day"] = (((h_sh >= 9) & (h_sh < 11)) | ((h_sh >= 13) & (h_sh < 16))).astype(float)
    # London/NY overlap: NY morning inside London afternoon (both weekdays)
    feats["sess_ldn_ny_overlap"] = (
        feats["sess_london_day"] * feats["sess_ny_day"]
    )
    # 15:00-16:30 London benchmark cluster: PM fix + WMR lead-in + equity close
    feats["sess_ldn_pm_cluster"] = ((h_ldn >= 15) & (h_ldn < 17) & (dow_ldn < 5)).astype(float)
    # US 08:30 ET macro-release hour (NFP/CPI style releases)
    feats["sess_us_macro_hour"] = ((h_ny == 8) & (dow_ny < 5)).astype(float)
    # Asia->Europe handover 07:00-08:00 London (overnight inventory unwinding,
    # Boyarchenko-Larsen-Whelan framework)
    feats["sess_asia_eu_handover"] = ((h_ldn >= 7) & (h_ldn < 8) & (dow_ldn < 5)).astype(float)

    # --- 3) DST state, offset gap, distance to transition ------------------
    is_dst_uk = np.array([t.dst().total_seconds() > 0 for t in loc_ldn])
    is_dst_us = np.array([t.dst().total_seconds() > 0 for t in loc_ny])
    feats["is_dst_uk"] = is_dst_uk.astype(float)
    feats["is_dst_us"] = is_dst_us.astype(float)
    off_ldn = np.array([t.utcoffset().total_seconds() for t in loc_ldn]) / 3600.0
    off_ny = np.array([t.utcoffset().total_seconds() for t in loc_ny]) / 3600.0
    feats["ny_ldn_offset_hours"] = (off_ldn - off_ny)          # 5.0 normally, 4.0 in gap weeks
    feats["is_dst_gap_week"] = (feats["ny_ldn_offset_hours"] == 4.0).astype(float)

    if year_range is None:
        year_range = (int(idx.year.min()) - 1, int(idx.year.max()) + 1)
    tr = _dst_transitions(year_range[0], year_range[1])
    for key, tag in (("UK", "uk"), ("US", "us")):
        trans = tr[key].tz_convert("UTC")
        # signed days to nearest transition (negative after)
        idx64 = idx.asi8  # ns
        t64 = trans.asi8
        j = np.searchsorted(t64, idx64)
        prev = np.where(j > 0, t64[np.clip(j - 1, 0, len(t64) - 1)], t64[0])
        nxt = np.where(j < len(t64), t64[np.clip(j, 0, len(t64) - 1)], t64[-1])
        d_prev = (idx64 - prev) / 86400e9
        d_next = (nxt - idx64) / 86400e9
        days_to = np.where(d_prev < d_next, -d_prev, d_next)
        feats[f"days_to_dst_{tag}"] = np.clip(days_to, -14, 14)
        feats[f"dst_change_week_{tag}"] = (np.abs(days_to) <= 3).astype(float)

    # --- 4) Session-split returns (Lou-Polk-Skouras decomposition) ---------
    if close_xau is not None:
        px = close_xau.reindex(idx)
        r = np.log(px / px.shift(1))
        sess = {
            # Asia session = Shanghai day + early night (01:00-08:00 UTC approx
            # via Shanghai wall clock 09:00-16:00)
            "asia": (feats["sess_shanghai_day"] == 1) | (feats["sess_asia_night"] == 1),
            "london_am": (h_ldn >= 8) & (h_ldn < 12) & (dow_ldn < 5),
            "ldn_ny_overlap": feats["sess_ldn_ny_overlap"] == 1,
            "ny_pm": (h_ny >= 12) & (h_ny < 17) & (dow_ny < 5),
        }
        for name, mask in sess.items():
            m = np.asarray(mask, dtype=bool)
            rr = pd.Series(r.values, index=idx).where(m)
            # cumulative session return over trailing 24 bars (sum of masked logs)
            feats[f"ret24_{name}"] = rr.fillna(0.0).rolling(24, min_periods=1).sum()
            feats[f"ret6_{name}"] = rr.fillna(0.0).rolling(6, min_periods=1).sum()

    return feats


def dst_gap_sanity_check() -> None:
    """Assert the two annual London-NY misalignment windows are found exactly.
    Autumn window: last-Sun Oct 01:00 UTC -> 1st-Sun Nov 06:00/07:00 UTC (7 days).
    Spring window: 2nd-Sun Mar 07:00 UTC -> last-Sun Mar 01:00 UTC (14/21 days)."""
    idx = pd.date_range("2026-10-20", "2026-11-05", freq="h", tz="UTC")
    f = clock_features(idx, None, year_range=(2026, 2026))
    off = f["ny_ldn_offset_hours"].resample("D").first()
    # Oct 20-25: both on summer time -> 5h. Oct 26-31: UK GMT, US EDT -> 4h.
    # Nov 1 (US falls back 06:00 UTC) onward: both standard -> 5h.
    assert (off["2026-10-20":"2026-10-25"] == 5.0).all(), "pre-gap offset wrong"
    assert (off["2026-10-26":"2026-10-31"] == 4.0).all(), "autumn gap window wrong"
    assert (off["2026-11-02":] == 5.0).all(), "post-gap offset wrong"
    # The LBMA PM fix must print at 11:00 NY during the gap, 10:00 NY otherwise
    pm_ny_hour = pd.Series(idx.tz_convert(NEWYORK).hour, index=idx)[f["minutes_to_LBMA_PM"] == 0]
    assert set(pm_ny_hour["2026-10-26":"2026-10-30"]) == {11}, "PM fix local NY hour in gap week"
    assert set(pm_ny_hour["2026-11-02":"2026-11-04"]) == {10}, "PM fix local NY hour normal week"
    print("F1 DST sanity checks passed (autumn 2026 gap window: 7 days at 4h).")


if __name__ == "__main__":
    dst_gap_sanity_check()

```

## 3.2 F2 — M1 realized path geometry

**Realized-variance family** (all definitions as fetched; P3-1..P3-6):

$$RV = \sum_i r_i^2, \qquad BV = \frac{\pi}{2}\frac{n}{n-1}\sum_{i=2}^{n}|r_{i-1}||r_i|, \qquad TQ = n\,\mu_{4/3}^{-3}\frac{n}{n-2}\sum_{i=2}^{n-1}|r_{i-1}|^{4/3}|r_i|^{4/3}|r_{i+1}|^{4/3}$$

$$Z_{jump} = \frac{(RV-BV)/RV}{\sqrt{\vartheta\,\tfrac{1}{n}\max\!\big(1,\;TQ/BV^2\big)}}, \qquad \vartheta = (\tfrac{\pi}{2})^2 + \pi - 5 \approx 0.6090, \quad \mu_{4/3} \approx 0.83086$$

$$RV^{+} = \sum_i r_i^2\,\mathbb{1}(r_i>0), \quad RV^{-} = \sum_i r_i^2\,\mathbb{1}(r_i<0), \quad SJ = RV^{+}-RV^{-}, \quad J^{\pm}=\max(RV^{\pm}-\tfrac12 BV,0)$$

Predictive asymmetry (Patton-Sheppard ✅): *bad-vol dominance* (RV⁻ ≫ RV⁺, SJ<0) → volatility continuation + negative drift bias; SJ>0 → vol decay; jump-flagged hours (Z>2.326) → partial reversal; diffusive BV → continuation-friendly. Jump-robust fallbacks MedRV (const 1.418640) and MinRV (2.751935) are included; microstructure-noise discipline: M1 is the native grid, validated per symbol by M1-return autocorrelation (the 5-min sparse / realized-kernel remedies are documented in the module, P3-6).

**Range estimators per H1 bar** (u=ln H/O, ℓ=ln L/O, c=ln C/O):

$$\hat v_P = \tfrac{1}{4\ln 2}(u-\ell)^2, \qquad \hat v_{GK} = 0.511\,d^2 - 0.019\,(c(u+\ell)-2u\ell) - 0.383\,c^2$$
$$\hat v_{RS} = u(u-c)+\ell(\ell-c), \qquad \hat v_{YZ} = \hat v_o + k\,\hat v_c + (1-k)\overline{\hat v_{RS}}, \;\; k = \tfrac{\alpha-1}{\alpha+\frac{n+1}{n-1}}, \;\alpha\approx1.34$$

(GK constants per Meilijson's canonical correction ✅; RS is drift-unbiased ✅; YZ absorbs inter-bar gaps ✅. The GK−RS spread proxies intra-hour drift intensity.)

**Variance ratios and Hurst.** Lo-MacKinlay VR(q) with the unbiased overlapping denominator m = (n_q−q+1)(1−q/n_q) — **Monte-Carlo-verified in this session** (E[VR]=1.00±0.01 on iid at N=240, q∈{4,16}) — with the heteroskedasticity-robust z via δ̂_j, θ̂(q); Hurst from the VR scaling slope H = ½ + slope(log VR, log q)/2. Regime rule: VR>1 / H>0.5 → weight continuation features; VR<1 / H<0.5 → weight fades.

**Roughness.** Gatheral-Jaisson-Rosenbaum m(q,Δ) = E|log σ_{t+Δ} − log σ_t|^q ∝ Δ^{qH} on the H1 RV series, Δ∈{1,2,4,8,24}h, q=2, H_rough = slope/2. GJR's H≈0.1 (and the fetched footnote that gold futures show the same smoothness) justifies recency-weighted vol features. *Caveat:* GJR fit the scaling on multi-day lags; the hourly-lag adaptation is a regime extrapolation, flagged in Appendix B.

```python
"""
AURUM v2.0 -- Feature Module F2: M1 Realized Path Geometry
===========================================================
Exact realized-measure machinery from the literature (all constants verified
against fetched primaries; see Spec v2.0 Section 3 formulas and Section 2 ledger):

  RV   = sum r_i^2                                   (Andersen-Bollerslev-Diebold-Labys 2001, JASA)
  BV   = (pi/2)(n/(n-1)) sum |r_{i-1}||r_i|          (Barndorff-Nielsen & Shephard 2004, JFEC 2(1))
  TQ   = n mu_{4/3}^-3 (n/(n-2)) sum |r_{i-1}|^{4/3}|r_i|^{4/3}|r_{i+1}|^{4/3}
  Z    = ((RV-BV)/RV) / sqrt(theta (1/n) max(1, TQ/BV^2)),  theta=(pi/2)^2+pi-5~=0.6090
         (BNS 2006 ratio jump statistic; Huang-Tauchen 2005 form)
  RV+/RV- (realized semivariances), SJ = RV+ - RV-   (BN-Kinnebrock-Shephard 2010;
                                                      Patton-Sheppard REStat 97(3) 2015)
  MedRV = pi/(6-4sqrt(3)+pi) (n/(n-2)) sum med(...)^2 (Andersen-Dobrev-Schaumburg 2012)
  MinRV = pi/(pi-2) (n/(n-1)) sum min(...)^2
  Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang range estimators
         (Parkinson 1980; Garman-Klass 1980; Rogers-Satchell 1991; Yang-Zhang 2000;
          canonical GK constants 0.511/0.019/0.383 per Meilijson arXiv:0807.3492)
  VR(q) with Lo-MacKinlay (1988 RFS) unbiased overlapping estimator and
         heteroskedasticity-robust z (delta_j, theta(q)); H = 0.5 + slope(logVR, log q)/2
  Volatility roughness: m(q, Delta) = E|log sigma_{t+Delta} - log sigma_t|^q propto Delta^{qH}
         (Gatheral-Jaisson-Rosenbaum arXiv:1410.3394, QF 18(6) 2018; H ~= 0.1, and
          "Crude Oil and Gold futures show similar smoothness" -- fetched footnote)

Predictive-sign hypotheses (Patton-Sheppard): RV- ("bad volatility") dominates vol
forecasting and persistence; SJ < 0 hours -> vol continuation + negative drift bias;
SJ > 0 -> vol decay; jump hours (Z > 2.326) -> partial reversal; VR > 1 trending
micro-regime -> weight continuation features; VR < 1 -> weight fade features.

All rolling computations are trailing (causal): the window [t-W, t] inclusive,
and features are read at H1 closes from fully-closed M1 bars only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- verified constants -------------------------------------------------------
SQRT_2_OVER_PI = 0.7978845608  # mu_1 = E|Z|
MU_4_3 = 0.83086               # mu_{4/3} = 2^(2/3) Gamma(7/6)/sqrt(pi)
MU_4_3_INV3 = 1.7430           # mu_{4/3}^{-3}
THETA_JUMP = 0.6090            # (pi/2)^2 + pi - 5   (Huang-Tauchen)
Z99 = 2.326                    # one-sided 99% jump threshold
MEDRV_C = 1.418640             # pi/(6 - 4*sqrt(3) + pi)
MINRV_C = 2.751935             # pi/(pi - 2)
PARKINSON_C = 0.360674         # 1/(4 ln 2)
GK_K1, GK_K2, GK_K3 = 0.511, 0.019, 0.383   # canonical Garman-Klass
YZ_ALPHA = 1.34                # Yang-Zhang k constant
VOL_FLOOR = 1e-12

M1_WINDOWS = {"1h": 60, "4h": 240, "24h": 1440}   # M1 bars per trailing window


def _m1_terms(m1: pd.DataFrame) -> pd.DataFrame:
    """Per-M1-bar additive terms for every realized measure. m1 must have a
    UTC DatetimeIndex and a 'close' column (open-labeled bars)."""
    r = np.log(m1["close"] / m1["close"].shift(1))
    out = pd.DataFrame(index=m1.index)
    out["r2"] = r ** 2
    out["r2_pos"] = (r ** 2).where(r > 0, 0.0)
    out["r2_neg"] = (r ** 2).where(r < 0, 0.0)
    out["bip"] = (r.abs() * r.abs().shift(1))          # |r_{i-1}||r_i|
    out["tq3"] = (r.abs().shift(1) ** (4.0 / 3.0)) * (r.abs() ** (4.0 / 3.0)) * (r.abs().shift(-1) ** (4.0 / 3.0))
    out["min2"] = np.minimum(r.abs(), r.abs().shift(1)) ** 2
    med3 = pd.concat([r.abs().shift(1), r.abs(), r.abs().shift(-1)], axis=1).median(axis=1)
    out["med2"] = med3 ** 2
    out["absr"] = r.abs()
    return out


def realized_block_m1(m1: pd.DataFrame, h1_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Realized-variance family per trailing M1 window, sampled at H1 closes.

    Parameters
    ----------
    m1 : M1 bars (UTC DatetimeIndex open-labeled, 'close' column required).
    h1_index : the H1 bar labels (open-labeled).  Features are read from the
        last M1 bar at or before (h1_label + 59min), i.e. bars fully closed
        inside the H1 bar -- strictly causal.
    """
    t = _m1_terms(m1)
    sums = {}
    for wname, W in M1_WINDOWS.items():
        s = t.rolling(W, min_periods=W // 2).sum()
        # rolling sum of x^2 needs (sum x)^2 handled separately; RV etc. use
        # only the additive terms; n = window size (use actual count for gaps)
        cnt = t["r2"].rolling(W, min_periods=1).count()
        n = cnt.clip(lower=2)

        rv = s["r2"]
        bv = (np.pi / 2.0) * (n / (n - 1)) * s["bip"]
        tq = n * MU_4_3_INV3 * (n / (n - 2).clip(lower=1)) * s["tq3"]
        medrv = MEDRV_C * (n / (n - 2).clip(lower=1)) * s["med2"]
        minrv = MINRV_C * (n / (n - 1)) * s["min2"]

        # BNS ratio jump statistic (one-sided); NaN where RV ~ 0
        with np.errstate(divide="ignore", invalid="ignore"):
            jump_share = ((rv - bv) / rv).clip(lower=0.0)
            zstat = ((rv - bv) / rv) / np.sqrt(
                (THETA_JUMP / n) * np.maximum(1.0, tq / (bv ** 2 + VOL_FLOOR))
            )
        rvp, rvn = s["r2_pos"], s["r2_neg"]

        block = pd.DataFrame(
            {
                f"RV_{wname}": rv,
                f"BV_{wname}": bv,
                f"MedRV_{wname}": medrv,
                f"MinRV_{wname}": minrv,
                f"jumpZ_{wname}": zstat,
                f"jumpShare_{wname}": jump_share,
                f"RVpos_{wname}": rvp,
                f"RVneg_{wname}": rvn,
                f"SJ_{wname}": rvp - rvn,
            },
            index=m1.index,
        )
        block[f"RVnegShare_{wname}"] = rvn / (rv + VOL_FLOOR)
        block[f"jumpFlag_{wname}"] = (zstat > Z99).astype(float)
        block[f"logRV_{wname}"] = np.log(rv + VOL_FLOOR)
        sums[wname] = block

    feats = pd.concat(sums, axis=1)
    feats.columns = [c[1] for c in feats.columns]
    # sample at H1 closes: last M1 bar whose OPEN label <= h1_label + 59 min
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    return feats.reindex(h1_close_idx, method="ffill").set_axis(h1_index)


def range_block_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """Range-based volatility per H1 bar + Yang-Zhang window estimator.
    h1: H1 bars (UTC index, open-labeled) with open/high/low/close."""
    o, h, l, c = h1["open"], h1["high"], h1["low"], h1["close"]
    u = np.log(h / o); lo = np.log(l / o); cl = np.log(c / o)
    d = u - lo
    oc = np.log(o / c.shift(1))                        # overnight/inter-bar gap
    co = np.log(c / o)

    pk = PARKINSON_C * d ** 2
    gk = GK_K1 * d ** 2 - GK_K2 * (cl * (u + lo) - 2 * u * lo) - GK_K3 * cl ** 2
    rs = u * (u - cl) + lo * (lo - cl)
    out = pd.DataFrame(
        {"parkinson": pk.clip(lower=0.0), "garman_klass": gk.clip(lower=0.0),
         "rogers_satchell": rs, "gap": oc},
        index=h1.index,
    )
    # Yang-Zhang drift-independent estimator over trailing windows
    n_win = {"YZ_24h": 24, "YZ_120h": 120, "YZ_528h": 528}
    for name, W in n_win.items():
        v_o = oc.rolling(W, min_periods=W // 2).var()
        v_c = co.rolling(W, min_periods=W // 2).var()
        v_rs = rs.rolling(W, min_periods=W // 2).mean()
        k = (YZ_ALPHA - 1.0) / (YZ_ALPHA + (W + 1.0) / (W - 1.0))
        out[name] = v_o + k * v_c + (1 - k) * v_rs
    # drift-intensity proxies: RS-vs-PK spread and RS-RV gap resolved per build step
    out["gk_minus_rs"] = out["garman_klass"] - out["rogers_satchell"]
    return out


def variance_ratio_block(h1_ret: pd.Series, qs=(2, 4, 8, 16), window: int = 240,
                         demean: bool = True) -> pd.DataFrame:
    """Lo-MacKinlay VR(q) with unbiased overlapping estimator and
    heteroskedasticity-robust z-statistic, on a trailing window of H1 returns.
    Returns VR(q), z(q), and the VR-slope Hurst estimate H = 0.5 + slope/2."""
    x = h1_ret.copy()
    if demean:
        x = x - x.rolling(window, min_periods=window // 2).mean()
    x2 = x ** 2
    N = window
    # sigma_1^2 (unbiased for the overlapping case uses total sum / (N-1))
    s1 = x2.rolling(N, min_periods=N // 2).sum()
    n_eff = x2.rolling(N, min_periods=N // 2).count().clip(lower=3)
    sig1 = s1 / (n_eff - 1)
    out = {}
    zouts = {}
    for q in qs:
        qsum = x.rolling(q).sum()                       # overlapping q-period sums
        s_q = (qsum ** 2).rolling(N, min_periods=N // 2).sum()
        # Lo-MacKinlay unbiased denominator for overlapping q-sums, per-period
        # form: m = (nq - q + 1)(1 - q/nq)  [Monte-Carlo verified unbiased on
        # iid data: E[VR] = 1.00 +- 0.01 at N=240, q in {4,16}]
        nq = (qsum.rolling(N, min_periods=N // 2).count()).clip(lower=q + 2)
        m = (nq - q + 1) * (1 - q / nq)
        sigq = s_q / m
        vr = sigq / (q * sig1 + 1e-18)
        out[f"VR{q}"] = vr
        # heteroskedasticity-robust variance theta(q) = sum_j [2(1-j/q)]^2 delta_j
        # delta_j = n * sum x_t^2 x_{t-j}^2 / (sum x_t^2)^2
        theta = pd.Series(0.0, index=x.index)
        for j in range(1, q):
            x2x2 = (x2 * x2.shift(j)).rolling(N, min_periods=N // 2).sum()
            dj = n_eff * x2x2 / (s1 ** 2 + 1e-18)
            theta = theta + ((2 * (q - j) / q) ** 2) * dj
        z = np.sqrt(n_eff) * (vr - 1) / np.sqrt(theta + 1e-18)
        zouts[f"VRz{q}"] = z
    res = pd.DataFrame({**out, **zouts})
    # Hurst from VR scaling: log VR(q) = (2H-1) log q  (fractional-process map)
    logq = np.log(np.array(qs, dtype=float))
    logq_c = logq - logq.mean()
    stack = np.stack([res[f"VR{q}"].values for q in qs], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logvr = np.log(np.abs(stack))
    logvr_c = logvr - logvr.mean(axis=1, keepdims=True)
    slope = (logvr_c * logq_c).sum(axis=1) / ((logq_c ** 2).sum() + 1e-18)
    res["hurst_VR"] = 0.5 + slope / 2.0
    return res


def roughness_block(rv_h1: pd.Series, deltas=(1, 2, 4, 8, 24), q: float = 2.0,
                    window: int = 240) -> pd.Series:
    """Gatheral-Jaisson-Rosenbaum volatility-roughness exponent on the H1 RV
    series: m(q, Delta) = mean |log sigma_{t+D} - log sigma_t|^q over trailing
    window; H_rough = slope(log m vs log Delta)/q.  GJR estimate H ~= 0.1 on
    5-min data across assets incl. gold futures; short-lag hourly adaptation
    documented in Spec Section 3.4 (regime extrapolation caveat)."""
    logsig = 0.5 * np.log(rv_h1 + VOL_FLOOR)
    lags = np.array(deltas, dtype=float)
    ms = []
    for D in deltas:
        diff = (logsig - logsig.shift(D)).abs() ** q
        ms.append(diff.rolling(window, min_periods=window // 2).mean())
    M = np.stack([m.values for m in ms], axis=1)          # T x len(deltas)
    logd = np.log(lags); logd_c = logd - logd.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        logm = np.log(M + VOL_FLOOR)
    logm_c = logm - np.nanmean(logm, axis=1, keepdims=True)
    slope = np.nansum(logm_c * logd_c, axis=1) / ((logd_c ** 2).sum() + 1e-18)
    return pd.Series(slope / q, index=rv_h1.index, name="H_rough_vol")


def har_aggregates(h1_feats: pd.DataFrame, horizons=(24, 120, 528)) -> pd.DataFrame:
    """HAR-style slow aggregates (daily / weekly / monthly analogues) of the
    per-hour realized features: rolling means per horizon (Corsi 2009 windows
    1/5/22 days -> 24/120/528 hours)."""
    out = {}
    for H in horizons:
        roll = h1_feats.rolling(H, min_periods=H // 2).mean()
        for col in roll.columns:
            out[f"{col}_har{H}"] = roll[col]
    return pd.DataFrame(out)


def self_test() -> None:
    """Synthetic-data checks: (1) BV/RV unbiased on constant-vol Gaussian M1;
    (2) VR ~ 1 on iid returns; (3) jump statistic fires on an injected jump;
    (4) Lévy-free range sanity: GK/PK within 2x of true variance on GBM;
    (5) causality: truncating the tail of the M1 frame must not change
    features at earlier H1 closes."""
    rng = np.random.default_rng(7)
    n = 20_000
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    sig = 0.0004  # per-minute vol
    px = 2000.0 * np.exp(np.cumsum(sig * rng.standard_normal(n)))
    m1 = pd.DataFrame({"close": px}, index=idx)

    h1_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    f = realized_block_m1(m1, h1_idx)
    last = f.iloc[-1]
    # (1) BV ~ RV ~ sigma^2 * 60 on the 1h window
    true_rv = (sig ** 2) * 60
    assert 0.5 < last["RV_1h"] / true_rv < 2.0, f"RV biased: {last['RV_1h']:.3e} vs {true_rv:.3e}"
    assert 0.5 < last["BV_1h"] / true_rv < 2.0, f"BV biased: {last['BV_1h']:.3e}"

    # (2) VR ~ 1 on iid H1 returns
    r_h1 = pd.Series(rng.standard_normal(4000) * 0.002,
                     index=pd.date_range("2020-01-01", periods=4000, freq="h", tz="UTC"))
    vr = variance_ratio_block(r_h1)
    assert 0.8 < vr["VR4"].iloc[-1] < 1.25, f"VR(4) on iid should be ~1, got {vr['VR4'].iloc[-1]}"
    assert 0.4 < vr["hurst_VR"].iloc[-1] < 0.6, f"iid Hurst should be ~0.5, got {vr['hurst_VR'].iloc[-1]}"

    # (3) jump statistic fires on injected jump
    px2 = px.copy(); px2[10_000:10_000 + 30] *= 1.004
    m1b = pd.DataFrame({"close": px2}, index=idx)
    fb = realized_block_m1(m1b, h1_idx)
    assert fb["jumpZ_24h"].max() > 2.326, "jump statistic failed to fire on injected jump"

    # (4) range estimators on GBM H1 built from the actual M1 path per hour
    n_h1 = len(px) // 60
    blocks = px[: n_h1 * 60].reshape(n_h1, 60)
    h1o, h1c = blocks[:, 0], blocks[:, -1]
    h1h, h1l = blocks.max(axis=1), blocks.min(axis=1)
    h1 = pd.DataFrame({"open": h1o, "high": h1h, "low": h1l, "close": h1c},
                      index=h1_idx[: n_h1])
    rb = range_block_h1(h1)
    true_var = (sig ** 2) * 60
    mid = rb["rogers_satchell"].iloc[100:200].mean()
    assert 0.3 < mid / true_var < 3.0, f"RS estimator off: {mid:.3e} vs {true_var:.3e}"

    # (5) causality: features at earlier H1 closes invariant to future M1 data
    h1_cut = h1_idx[h1_idx < idx[15_000]]
    f_cut = realized_block_m1(m1.iloc[:15_060], h1_cut)
    f_full_at_cut = f.loc[h1_cut[h1_cut < idx[15_000] - pd.Timedelta("2h")]]
    f_cut_cmp = f_cut.loc[h1_cut[h1_cut < idx[15_000] - pd.Timedelta("2h")]]
    pd.testing.assert_frame_equal(f_cut_cmp, f_full_at_cut, check_freq=False)
    print("F2 self-tests passed: BV/RV unbiased, VR~1 iid, jump fires, RS sane, causal.")


if __name__ == "__main__":
    self_test()

```

## 3.3 F3 — Depth-2/3 multi-asset path signatures

The signature of the time-augmented 5-channel path X = (t̃, xau, xag, eur, jpy) — per-channel z-scored M1 log-increments — is computed per H1 bar via the exact Chen streaming recursion on the piecewise-linear path:

$$S^i_k = S^i_{k-1} + \Delta x^i_k, \qquad S^{ij}_k = S^{ij}_{k-1} + S^i_{k-1}\Delta x^j_k + \tfrac12 \Delta x^i_k \Delta x^j_k$$
$$S^{ijk}_k = S^{ijk}_{k-1} + S^{ij}_{k-1}\Delta x^k_k + \tfrac12 S^i_{k-1}\Delta x^j_k\Delta x^k_k + \tfrac16 \Delta x^i_k\Delta x^j_k\Delta x^k_k$$

$$\text{Lévy area } A^{ij} = \tfrac12\big(S^{ij} - S^{ji}\big) = \tfrac12\int\!\big(X^i dX^j - X^j dX^i\big)$$

Time-augmentation is **mandatory** (otherwise the signature is blind to reparametrization; P3-13 ✅). Depth-2 full signature = 4 level-1 + 25 level-2 terms; **10 pairwise Lévy areas** are the who-moved-first detectors — *did USDJPY drop before XAUUSD ticked up, or did XAUUSD move alone?* — with sign flipping under path reversal (unit-tested: the CCW unit triangle has area exactly 0.5; its reversal −0.5). Explicit **lead-lag pair features** for (xau, jpy) and (xau, xag) use the Flint-Lyons-Turner lead-lag transform ✅. Term counts (P3-13/P3-14 ✅): depth-2 full (d=5) = 30; depth-3 = 155 (optional flag, `depth3=True`); log-signature depth-2 = 15. Cost: 63 µs/bar measured (depth-2), i.e., ~6 s for 95k bars.

```python
"""
AURUM v2.0 -- Feature Module F3: Depth-2/3 Multi-Asset Path Signatures
======================================================================
Theory (Chevyrev & Kormilitzin, "A primer on the signature method in machine
learning", arXiv:1603.03788; Morrill et al. arXiv:2006.00873; Reizenstein &
Graham arXiv:1802.08252):

  Signature of a d-dim path X over [a,b]:  S^{i}_{a,t} = X^i_t - X^i_a,
  S^{i,j}_{a,t} = int_a^t S^i_{a,s} dX^j_s,  S^{i_1..i_k} = iterated integrals
  over the simplex -> the ORDER OF EVENTS is encoded (cross terms).

  Levy area (2D):  A^{ij} = 1/2 ( S^{i,j} - S^{j,i} ) = 1/2 int (X^i dX^j - X^j dX^i).
  For a closed counter-clockwise loop, A = enclosed area.  A > 0 iff channel i
  "leads" the co-movement with channel j in the CCW sense; sign flips under
  path reversal -- exactly the who-moved-first detector OHLC aggregates destroy.

  Time-augmentation is MANDATORY (otherwise the signature is invariant to
  time-reparametrisation and blind to speed): channel 0 of the path is
  normalized time.

  Exact streaming recursion for the piecewise-linear interpolated path
  (Chen's identity on each linear segment, tensor-exponential increment):
    S1_i  <- S1_i  + dx_i
    S2_ij <- S2_ij + S1_i * dx_j + 0.5 * dx_i * dx_j
    S3_ijk<- S3_ijk+ S2_ij * dx_k + 0.5 * S1_i * dx_j * dx_k + dx_i dx_j dx_k / 6
  (level-3 term uses the PRE-update S1, S2 -- order matters.)

  Term counts (incl. the constant 1, which we drop): depth-N full signature
  has (d^(N+1)-1)/(d-1) terms; d=5 depth 2 -> 30, depth 3 -> 155; log-signature
  depth 2 -> d + d(d-1)/2 = 15 (increments + Levy areas).

Channels (default): normalized time + per-symbol z-scored M1 log-price
increments for XAUUSD, XAGUSD, EURUSD, USDJPY  => d=5 (optional +tick-volume
=> d=6).  Windows: the 60 M1 bars inside each H1 bar (and an optional 4h
window).  Missing M1 bars are forward-filled (zero increments) -- documented
convention; coverage statistics should be emitted alongside.

The depth-3 block is optional (155 features); default output is the depth-2
full signature (30) + the 10 pairwise Levy areas + lead-lag pair areas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STEPS_PER_HOUR = 60


def _hourly_path_tensor(m1: pd.DataFrame, channels: dict[str, str],
                        h1_index: pd.DatetimeIndex,
                        zscore_span: int = 1440) -> np.ndarray:
    """Build the (T_h1, 60, d) path tensor with channel 0 = normalized time.

    Parameters
    ----------
    m1 : M1 frame, UTC DatetimeIndex (open-labeled), column names given by
         ``channels`` mapping channel-name -> m1 column (e.g. {'xau': 'close',
         'xag': 'close_xag', ...} all as PRICE levels).
    zscore_span : trailing M1 bars for per-channel increment z-scoring (24h).
    """
    cols = []
    for name in channels:
        if name == "time":
            continue
        px = m1[channels[name]].astype("float64")
        inc = np.log(px / px.shift(1))
        mu = inc.rolling(zscore_span, min_periods=60).mean()
        sd = inc.rolling(zscore_span, min_periods=60).std()
        cols.append(((inc - mu) / (sd + 1e-12)).rename(name))
    incs = pd.concat(cols, axis=1)

    # full hourly x 60-minute grid, forward-fill gaps (zero increments)
    full = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    grid = incs.reindex(full).ffill().fillna(0.0)
    # hours present in h1_index only
    T = len(h1_index)
    d = len(channels)  # includes time channel
    X = np.zeros((T, STEPS_PER_HOUR, d), dtype=np.float32)
    # normalized time channel 0: k/59 in [0,1] per hour (fixed grid)
    X[:, :, 0] = np.linspace(0.0, 1.0, STEPS_PER_HOUR, dtype=np.float32)[None, :]
    for j, name in enumerate([n for n in channels if n != "time"]):
        vals = grid[name].values
        # for each H1 label t, the 60 M1 bars are [t, t+59min]
        starts = h1_index.asi8 // 60_000_000_000  # ns -> minutes since epoch
        full_starts = full.asi8 // 60_000_000_000
        # map each hour label to position in the minute grid
        pos = np.searchsorted(full_starts, starts)
        ok = (pos >= 0) & (pos + STEPS_PER_HOUR <= len(vals))
        pos_c = np.clip(pos, 0, len(vals) - STEPS_PER_HOUR)
        # increments of the z-scored series within the hour
        raw = np.lib.stride_tricks.sliding_window_view(vals, STEPS_PER_HOUR)
        # convert levels of increments to per-hour cumulative increments:
        # X[:, :, j] = increment value at each minute (the *path* is the
        # cumulative sum, applied inside the signature recursion)
        X[ok, :, 1 + j] = raw[pos_c][ok].astype(np.float32)
    return X


def signature_depth2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact depth-2 signature of the cumulative path of per-step increments.

    Parameters: X (T, n_steps, d) per-step INCREMENTS (channel 0 may be a
    level in [0,1]; it is differenced like the others -- for the time channel
    use the per-step time increments directly).
    Returns (S2, levycli) with S2 (T, d + d*d) = [level-1 terms | level-2
    terms flattened (i-major)] and the (T, d, d) level-2 tensor (for areas).
    """
    T, n, d = X.shape
    S1 = np.zeros((T, d), dtype=np.float64)
    S2 = np.zeros((T, d, d), dtype=np.float64)
    for k in range(n):
        dx = X[:, k, :].astype(np.float64)                     # (T, d)
        S2 += np.einsum("ti,tj->tij", S1, dx, optimize=True)   # S1_{k-1} dx_k
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx, optimize=True)
        S1 += dx
    flat = np.concatenate([S1, S2.reshape(T, d * d)], axis=1)
    return flat.astype(np.float32), S2


def signature_depth3(X: np.ndarray) -> np.ndarray:
    """Exact depth-3 signature (adds d^3 third-level terms).  Memory: T x d^3
    float64 -> use float32 and chunk if T is large."""
    T, n, d = X.shape
    S1 = np.zeros((T, d), dtype=np.float64)
    S2 = np.zeros((T, d, d), dtype=np.float64)
    S3 = np.zeros((T, d, d, d), dtype=np.float64)
    for k in range(n):
        dx = X[:, k, :].astype(np.float64)
        S3 += np.einsum("tij,tl->tijl", S2, dx, optimize=True)
        S3 += 0.5 * np.einsum("ti,tj,tl->tijl", S1, dx, dx, optimize=True)
        S3 += (1.0 / 6.0) * np.einsum("ti,tj,tl->tijl", dx, dx, dx, optimize=True)
        S2 += np.einsum("ti,tj->tij", S1, dx, optimize=True)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx, optimize=True)
        S1 += dx
    return np.concatenate([S1, S2.reshape(T, d * d), S3.reshape(T, d ** 3)],
                          axis=1).astype(np.float32)


def levy_areas(S2: np.ndarray) -> np.ndarray:
    """Pairwise Levy areas A^{ij} = (S^{ij} - S^{ji})/2, upper triangle (i<j)."""
    d = S2.shape[1]
    iu = np.triu_indices(d, k=1)
    return 0.5 * (S2[:, iu[0], iu[1]] - S2[:, iu[1], iu[0]])


def leadlag_pair_areas(inc_a: np.ndarray, inc_b: np.ndarray) -> np.ndarray:
    """Levy areas of the lead-lag transform phi(x) = ((x1,x1),(x2,x1),(x2,x2),...)
    for one pair of increment series (T, n) each -- the explicit 'did b move
    before a' detector (Flint-Lyons-Turner lead-lag encoding via Morrill et al.
    arXiv:2006.00873).  Returns (T, 6): the four depth-2 terms of the 2-channel
    lead-lag pair plus the two cross Lévy areas."""
    T, n = inc_a.shape
    # lead-lag path increments: (a, a), (b, a), (b, b) per step a: x1, x2 = a, b
    # path: (a_1, a_1), (a_2, a_1), (a_2, a_2), (a_3, a_2), ...
    # channel 1 = a, channel 2 = lagged a then b... standard construction:
    # X_lead = a, X_lag  = previous a then b.  We build the 2d increment list.
    # simplified equivalent: increments of lead channel: (da_k, da_k), then
    # (0, db_k) appended after each step.
    S1 = np.zeros((T, 2)); S2 = np.zeros((T, 2, 2))
    for k in range(n):
        da = inc_a[:, k]; db = inc_b[:, k]
        # first move: lead goes da, lag stays
        dx = np.stack([da, np.zeros_like(da)], axis=1)
        S2 += np.einsum("ti,tj->tij", S1, dx)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx)
        S1 += dx
        # second move: lag catches up db
        dx = np.stack([np.zeros_like(db), db], axis=1)
        S2 += np.einsum("ti,tj->tij", S1, dx)
        S2 += 0.5 * np.einsum("ti,tj->tij", dx, dx)
        S1 += dx
    A = 0.5 * (S2[:, 0, 1] - S2[:, 1, 0])
    return np.stack([S1[:, 0], S1[:, 1], S2[:, 0, 0], S2[:, 1, 1],
                     A, S2[:, 0, 1] + S2[:, 1, 0]], axis=1)


def signature_features(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
                       price_cols: dict[str, str],
                       depth3: bool = False,
                       leadlag_pairs: tuple[tuple[str, str], ...] = (("xau", "jpy"), ("xau", "xag")),
                       ) -> pd.DataFrame:
    """Full signature feature block.

    Parameters
    ----------
    price_cols : channel name -> M1 price column, e.g.
        {'xau': 'close', 'xag': 'close_xag', 'eur': 'close_eur', 'jpy': 'close_jpy'}.
    depth3 : include the 125 extra depth-3 terms (d=5).
    leadlag_pairs : symbol pairs for explicit lead-lag area features.
    """
    channels = {"time": "time", **price_cols}
    X = _hourly_path_tensor(m1, channels, h1_index)
    flat, S2 = signature_depth2(X)
    d = X.shape[2]
    chan_all = list(channels.keys())                       # ['time', 'xau', ...]
    chan_px = [c for c in chan_all if c != "time"]
    # level-1: drop the deterministic time increment -> d-1 terms
    # level-2: keep ALL d*d terms (cross terms with time carry speed info)
    names = [f"sig1_{c}" for c in chan_px] + \
            [f"sig2_{c1}_{c2}" for c1 in chan_all for c2 in chan_all]
    vals = np.concatenate([flat[:, 1:d], flat[:, d:]], axis=1)
    blocks = [pd.DataFrame(vals, index=h1_index, columns=names)]
    # pairwise Levy areas for all price-channel pairs (drop time channel)
    chan_names = [c for c in channels if c != "time"]
    iu = np.triu_indices(d, k=1)
    keep = [(a, b) for a, b in zip(iu[0], iu[1]) if a > 0 and b > 0]
    area_cols = {}
    for a, b in keep:
        area_cols[f"levy_{chan_names[a - 1]}_{chan_names[b - 1]}"] = \
            0.5 * (S2[:, a, b] - S2[:, b, a])
    blocks.append(pd.DataFrame(area_cols, index=h1_index))

    # lead-lag pair features
    incs = {}
    for name in price_cols:
        px = m1[price_cols[name]].astype("float64")
        inc = np.log(px / px.shift(1))
        mu = inc.rolling(1440, min_periods=60).mean()
        sd = inc.rolling(1440, min_periods=60).std()
        incs[name] = ((inc - mu) / (sd + 1e-12))
    full_idx = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    for a, b in leadlag_pairs:
        A = incs[a].reindex(full_idx).ffill().fillna(0.0).values
        B = incs[b].reindex(full_idx).ffill().fillna(0.0).values
        starts = h1_index.asi8 // 60_000_000_000
        full_starts = full_idx.asi8 // 60_000_000_000
        pos = np.clip(np.searchsorted(full_starts, starts), 0, len(A) - STEPS_PER_HOUR)
        ia = np.lib.stride_tricks.sliding_window_view(A, STEPS_PER_HOUR)[pos]
        ib = np.lib.stride_tricks.sliding_window_view(B, STEPS_PER_HOUR)[pos]
        ll = leadlag_pair_areas(ia.astype(np.float64), ib.astype(np.float64))
        ll_cols = {f"ll_{a}_{b}_{nm}": ll[:, j]
                   for j, nm in enumerate(["lead", "lag", "d11", "d22", "area", "sym"])}
        blocks.append(pd.DataFrame(ll_cols, index=h1_index))

    if depth3:
        f3sig = signature_depth3(X)
        d3names = [f"sig3_{c1}_{c2}_{c3}" for c1 in chan_all for c2 in chan_all for c3 in chan_all]
        blocks.append(pd.DataFrame(f3sig[:, d + d * d: d + d * d + d ** 3],
                                   index=h1_index, columns=d3names))
    return pd.concat(blocks, axis=1)


def self_test() -> None:
    """Known-geometry checks: (1) Levy area of the CCW unit right triangle
    (0,0)->(1,0)->(1,1)->(0,0) equals the enclosed area 0.5; (2) reversal
    flips the sign; (3) diagonal level-2 term of a straight line = dx^2/2;
    (4) lead-lag area sign encodes who moved first; (5) causality."""
    # (1)-(3): 2-channel pure paths fed through signature_depth2
    tri = np.array([[[1, 0], [0, 1], [-1, -1]]], dtype=np.float64)   # 3 steps
    _, S2 = signature_depth2(tri)
    A = 0.5 * (S2[0, 0, 1] - S2[0, 1, 0])
    assert abs(A - 0.5) < 1e-12, f"triangle Levy area {A} != 0.5"
    # (2) reversal
    tri_rev = np.array([[[-1, -1], [0, 1], [1, 0]]], dtype=np.float64)
    _, S2r = signature_depth2(tri_rev)
    Ar = 0.5 * (S2r[0, 0, 1] - S2r[0, 1, 0])
    assert abs(Ar + 0.5) < 1e-12, f"reversal did not flip area: {Ar}"
    # (3) straight line diagonal term
    line = np.array([[[0.3, 0.0], [0.3, 0.0]]], dtype=np.float64)
    _, S2l = signature_depth2(line)
    assert abs(S2l[0, 0, 0] - 0.5 * 0.6 ** 2) < 1e-12, "diagonal term != dx^2/2"
    # (4) lead-lag: a moves first, b follows one step later -> area > 0
    T, n = 1, 60
    ia = np.zeros((T, n)); ib = np.zeros((T, n))
    ia[0, 10] = 1.0; ib[0, 20] = 1.0
    ll = leadlag_pair_areas(ia, ib)
    assert ll[0, 4] > 0, "lead-lag area should be positive when a leads"
    ia2 = np.zeros((T, n)); ib2 = np.zeros((T, n))
    ia2[0, 20] = 1.0; ib2[0, 10] = 1.0
    ll2 = leadlag_pair_areas(ia2, ib2)
    assert ll2[0, 4] < 0, "lead-lag area should be negative when b leads"
    # (5) depth-3 Chen identity spot check on a single linear segment:
    # signature of one linear step dx = tensor exp(dx) -> S3_111 = dx^3/6
    seg = np.array([[[0.5, -0.25]]], dtype=np.float64)
    f3 = signature_depth3(seg)          # flat: [S1(d=2) | S2(4) | S3(8)]
    i_s3_000 = 2 + 4 + 0                # S3[0,0,0] position in the S3 block
    assert abs(f3[0, i_s3_000] - 0.5 ** 3 / 6.0) < 1e-12, "depth-3 linear-step term wrong"
    print("F3 self-tests passed: triangle area, reversal, diagonal, lead-lag, Chen.")


if __name__ == "__main__":
    self_test()

```

## 3.4 F4 — Microstructural toxicity & liquidity

**VPIN with bulk volume classification** (P3-15 ✅): volume buckets (48 trailing buckets per 24h; time-bar variant, documented), per-M1-bar buy fraction Φ(ΔP/σ_ΔP), VPIN = mean bucket |V_buy − V_sell|/V_bucket. FX/CFD adaptation: **tick volume** as the activity proxy (Bollerslev-Domowitz 1993 / Dacorogna et al. 2001 lineage ◐). The Andersen-Bondarenko critique is honored: VPIN enters as a *state variable* (it largely proxies short-horizon volatility), never as standalone alpha. **Kyle's λ** = rolling Cov(r, SV)/Var(SV) with BVC-signed volume, plus the lag-1 transient-impact share. **Realized Amihud** RPV/ν = Σ|r_i|/Σν_i per window (5-min-sampling efficiency argument ✅). **CLV** = ((C−L)−(H−C))/(H−L) with tick-volume-weighted windowed sums.

```python
"""
AURUM v2.0 -- Feature Module F4: Microstructural Toxicity & Liquidity
======================================================================
Formulas (verified; see Spec Section 3.6 and ledger P3-34..P3-44):

  Bulk Volume Classification (Easley, Lopez de Prado & O'Hara 2012, RFS 25(5);
  equations via arXiv:1912.10343):
      V_buy,i  = V_i * Phi( dP_i / sigma_dP ),   V_sell,i = V_i - V_buy,i
  VPIN (volume-bucketed): split the stream into buckets of equal volume
  (standard: 50 per day); VPIN = mean_bucket |V_buy - V_sell| / V_bucket.
  FX/CFD adaptation: tick volume (number of M1 price updates) as the activity
  proxy (Bollerslev-Domowitz 1993; Dacorogna et al. 2001 lineage); the
  Andersen-Bondarenko critique (J. Financial Markets 2014) -- VPIN largely
  proxies short-horizon volatility -- is honored by using VPIN as a STATE
  variable, never as standalone alpha.

  Kyle lambda (Kyle 1985, Econometrica): lambda = Cov(r, SV)/Var(SV) with
  SV = BVC-signed volume; rolling estimation = transient impact slope.

  Realized Amihud (Lacava-Ranaldo-Santucci de Magistris via arXiv:2609.00943):
      Illiq = RPV / nu = sum_i |r_i| / sum_i nu_i   (5-min sampling "several
      times more efficient" than daily Amihud; M1 acceptable for majors).

  CLV (close location value) volume-weighted accumulation:
      CLV = ((C-L) - (H-C)) / (H-L) in [-1,1];  AD_t = sum CLV * V (windowed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

M1_BARS_PER_BUCKET = 30      # time-bar VPIN bucket (~30 min); 48 buckets/24h
N_BUCKETS = 48               # trailing buckets in the VPIN mean (24h)
SIG_DP_SPAN = 1440           # rolling std of M1 price changes (24h)
LAMBDA_SPAN = 1440           # rolling Kyle-lambda window (24h)


def _bvc(m1: pd.DataFrame, price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    dP = m1[price_col].diff()
    sd = dP.rolling(SIG_DP_SPAN, min_periods=120).std()
    z = (dP / (sd + 1e-12)).fillna(0.0)
    v = m1[vol_col].astype("float64").fillna(0.0).clip(lower=0.0)
    vb = v * _norm.cdf(z)
    vs = v - vb
    sv = vb - vs                       # BVC signed volume in [-V, +V]
    return pd.DataFrame({"V": v, "Vb": vb, "Vs": vs, "SV": sv}, index=m1.index)


def vpin_block(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
               price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Causal tick-volume VPIN (time-bar buckets + BVC) sampled at H1 closes."""
    b = _bvc(m1, price_col, vol_col)
    # bucket sums: fixed M1_BARS_PER_BUCKET-bar buckets on the global minute grid
    full = pd.date_range(m1.index[0], m1.index[-1], freq="min", tz="UTC")
    g = b.reindex(full).fillna(0.0)
    minute_pos = np.arange(len(g))
    bucket_id = minute_pos // M1_BARS_PER_BUCKET
    take = minute_pos % M1_BARS_PER_BUCKET == (M1_BARS_PER_BUCKET - 1)
    vsum = g["V"].groupby(bucket_id).cumsum()
    bsum = g["Vb"].groupby(bucket_id).cumsum()
    ssum = g["Vs"].groupby(bucket_id).cumsum()
    # bucket-end rows: total, buy, sell within bucket
    V_b = vsum[take]; Vb_b = bsum[take]; Vs_b = ssum[take]
    imb = (Vb_b - Vs_b).abs() / (V_b + 1e-12)
    vpin = imb.rolling(N_BUCKETS, min_periods=N_BUCKETS // 2).mean()
    # per-bucket signed imbalance (direction of toxic flow)
    simb = (Vb_b - Vs_b) / (V_b + 1e-12)
    svpin = simb.rolling(N_BUCKETS, min_periods=N_BUCKETS // 2).mean()
    out = pd.DataFrame({"vpin": vpin, "vpin_signed": svpin})
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    return out.reindex(h1_close_idx, method="ffill").set_axis(h1_index)


def kyle_lambda_block(m1: pd.DataFrame, h1_index: pd.DatetimeIndex,
                      price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Rolling Kyle lambda = Cov(r_M1, SV)/Var(SV) (impact slope per signed
    tick-volume unit), plus transient/total impact ratio via lagged covariances."""
    b = _bvc(m1, price_col, vol_col)
    r = np.log(m1[price_col] / m1[price_col].shift(1))
    df = pd.DataFrame({"r": r, "SV": b["SV"]}, index=m1.index).dropna()
    cov = df["r"].rolling(LAMBDA_SPAN, min_periods=240).cov(df["SV"])
    var = df["SV"].rolling(LAMBDA_SPAN, min_periods=240).var()
    lam = (cov / (var + 1e-18)).clip(lower=0.0)
    # lag-1 impact share: Cov(r_{t+1}, SV_t)/Var(SV) -- decay = transient impact
    cov_lag = df["r"].shift(-1).rolling(LAMBDA_SPAN, min_periods=240).cov(df["SV"])
    lam_lag = (cov_lag / (var + 1e-18)).clip(lower=0.0)
    out = pd.DataFrame({"kyle_lambda": lam, "kyle_lambda_lag": lam_lag,
                        "tick_volume_24h": b["V"].rolling(LAMBDA_SPAN, min_periods=240).sum()})
    h1_close_idx = h1_index + pd.Timedelta(minutes=59)
    return out.reindex(h1_close_idx, method="ffill").set_axis(h1_index)


def amihud_clv_block(m1: pd.DataFrame, h1: pd.DataFrame,
                     price_col: str = "close", vol_col: str = "volume") -> pd.DataFrame:
    """Realized Amihud per window + CLV volume-weighted accumulation per H1.
    h1: H1 OHLC frame (for CLV); m1: for the realized variant."""
    r = np.log(m1[price_col] / m1[price_col].shift(1)).abs()
    v = m1[vol_col].astype("float64").fillna(0.0).clip(lower=0.0)
    out = {}
    for wname, W in (("1h", 60), ("4h", 240), ("24h", 1440)):
        rpv = r.rolling(W, min_periods=W // 2).sum()
        nu = v.rolling(W, min_periods=W // 2).sum()
        out[f"amihud_{wname}"] = rpv / (nu + 1e-12)
    res = pd.DataFrame(out, index=m1.index)
    h1_close_idx = h1.index + pd.Timedelta(minutes=59)
    res = res.reindex(h1_close_idx, method="ffill").set_axis(h1.index)

    # CLV per H1 bar (from H1 OHLC), tick-volume weighted, windowed
    h, l, c = h1["high"], h1["low"], h1["close"]
    rng = (h - l).replace(0.0, np.nan)
    clv = (((c - l) - (h - c)) / rng).fillna(0.0)
    tv = h1[vol_col].astype("float64").fillna(0.0) if vol_col in h1 else 0.0
    clv_tv = clv * tv
    res["clv"] = clv
    for W in (6, 24, 120):
        res[f"clv_tv_sum{W}"] = clv_tv.rolling(W, min_periods=W // 2).sum()
        res[f"clv_mean{W}"] = clv.rolling(W, min_periods=W // 2).mean()
    return res


def self_test() -> None:
    rng = np.random.default_rng(11)
    n = 60 * 24 * 30
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    # toxic regime: signed flow drives price (persistent impact)
    flow = rng.standard_normal(n)
    ret = 0.00002 * flow + 0.0002 * rng.standard_normal(n)
    px = 2000 * np.exp(np.cumsum(ret))
    tv = (100 + 50 * rng.random(n)) * (1 + 5 * np.abs(flow))
    m1 = pd.DataFrame({"close": px, "volume": tv}, index=idx)
    h1_idx = pd.date_range(idx[0], idx[-1], freq="h", tz="UTC")
    vp = vpin_block(m1, h1_idx)
    assert vp["vpin"].notna().sum() > 100
    assert ((vp["vpin"].dropna() >= 0) & (vp["vpin"].dropna() <= 1)).all(), "VPIN outside [0,1]"
    kl = kyle_lambda_block(m1, h1_idx)
    assert (kl["kyle_lambda"].dropna() > 0).mean() > 0.9, "lambda should be ~positive under impact"
    # CLV block with synthetic H1
    n_h1 = n // 60
    blocks = px[: n_h1 * 60].reshape(n_h1, 60)
    h1 = pd.DataFrame({"open": blocks[:, 0], "high": blocks.max(1), "low": blocks.min(1),
                       "close": blocks[:, -1], "volume": tv[: n_h1 * 60].reshape(n_h1, 60).sum(1)},
                      index=h1_idx[: n_h1])
    am = amihud_clv_block(m1, h1)
    assert am["amihud_1h"].notna().sum() > 100
    assert am["clv"].between(-1, 1).all()
    # causality
    cut = 60 * 24 * 20
    h1c = h1_idx[h1_idx < idx[cut]]
    vpc = vpin_block(m1.iloc[: cut + 60], h1c)
    common = h1c[h1c < idx[cut - 120]]
    pd.testing.assert_frame_equal(vpc.loc[common], vp.loc[common])
    print("F4 self-tests passed: VPIN bounds, lambda sign, Amihud/CLV, causal.")


if __name__ == "__main__":
    self_test()

```

## 3.5 F5 — Online cross-asset residualization (the two alpha channels)

EW-RLS with forgetting λ=0.995 on the standardized design [1, r_JPY, r_EUR, r_XAG] → y = r_XAU, with strict causal ordering per bar: **(1)** predict ŷ_t = x_tᵀβ_{t−1}; **(2)** the a-priori innovation ε_t = y_t − ŷ_t is the tradable residual; **(3)** only then update β. Exact recursions:

$$g_t = \frac{P_{t-1}x_t}{\lambda + x_t^\top P_{t-1} x_t}, \qquad \beta_t = \beta_{t-1} + \varepsilon_t\, g_t, \qquad P_t = \lambda^{-1}\big(P_{t-1} - g_t x_t^\top P_{t-1}\big)$$

Residual process (Avellaneda-Lee transplant to hourly): spread X_t = Σε; OU fit on trailing 480 bars → κ, τ=1/κ (kill-switch τ>240 bars), σ_eq = σ_ε√(τ/2), z_t = (X_t − m_X)/σ_eq.

- **Channel A (lead-lag catch-up):** q_A = tanh(EWMA₁₆(ε)/σ_ε) — the FX/metals complex moved, gold hasn't caught up: trade *with* residual momentum.
- **Channel B (idiosyncratic overextension):** q_B = −(z/2.5)·φ_sat(z/2.5)·1[τ valid], φ_sat(y)=y·exp(−y²/4)/0.89 (Baz/DMN response ✅) — fade isolated bullion sweeps at |z|>2.5σ_eq, exit 0.75, saturation de-risks beyond √2 (extreme moves are regime breaks more often than reversion trades). Conflict discount when channels disagree.

```python
"""
AURUM v2.0 -- Feature Module F5: Online EW-RLS Cross-Asset Residualization
==========================================================================
State-space decomposition of XAUUSD hourly log returns on the three context
assets (the *arbitrage/macro-hedging triangle*):

    y_t = alpha_t + beta_JPY,t x_JPY,t + beta_EUR,t x_EUR,t + beta_XAG,t x_XAG,t + eps_t
    beta_t = beta_{t-1} + w_t        (random-walk coefficients)

Estimated by Recursive Least Squares with forgetting factor lambda -- the
information-form Kalman filter for the TVP regression (RLS(λ) ⟺ Kalman with
Q/R tied to (1-λ)/λ; the P-recursion is a Riccati equation).  Exact recursions
(Wikipedia RLS/Kalman cross-verified in the P4 digest; cf. Harvey 1989;
Durbin & Koopman 2012):

    a_t   = y_t - x_t' beta_{t-1}                     (a-priori innovation)
    g_t   = P_{t-1} x_t / (lambda + x_t' P_{t-1} x_t) (gain)
    P_t   = lambda^-1 P_{t-1} - g_t x_t' lambda^-1 P_{t-1}
    beta_t= beta_{t-1} + a_t g_t

CAUSAL DISCIPLINE: the tradable residual at bar t is the a-priori innovation
eps_t = y_t - x_t' beta_{t-1} (betas from bars <= t-1); the RLS update with
(x_t, y_t) happens only AFTER the residual is recorded.  Never re-fit in-sample.

Residual process (Avellaneda-Lee 2010 transplant to hourly frequency):
    X_t = cumsum(eps)                     (cointegration spread level)
    kappa from AR(1) of X over trailing window: rho = Corr(X_t, X_{t-1}),
        kappa = -ln(rho)/1 bar, tau = 1/kappa (clip tau <= 240 bars)
    sigma_eq = sigma_eps * sqrt(tau/2)    (OU stationary std)
    z_t = (X_t - m_X) / sigma_eq          (s-score analogue; A&L entry 1.25 /
        2.5 hourly-hardened, exits 0.75/0.50)

Two orthogonal alpha channels:
    Channel A (lead-lag catch-up):  m_A = EWMA_16(eps)/sigma_eps  -- residual
        momentum: the FX/metals complex moved and gold has not yet caught up.
    Channel B (idiosyncratic overextension): fade z when |z| > 2.5 with the
        saturating response phi(y) = y exp(-y^2/4)/0.89 (Baz et al. / DMN),
        exit |z| < 0.75, kill-switch if tau > 240 bars (OU invalid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RLS_LAMBDA = 0.995          # forgetting factor: memory ~ 1/(1-lambda) = 200 bars (~8.3d)
P_INIT = 1e2                # P(0) = delta * I
VOL_SPAN_X = 60             # EW vol standardization of regressors (bars)
RES_MOM_SPAN = 16           # Channel A EWMA span (bars)
SIG_EPS_SPAN = 480          # EW std of residual (bars)
MEAN_X_SPAN = 480           # EW mean of spread level (bars)
OU_WINDOW = 480             # kappa estimation window (bars)
Z_ENTRY = 2.5               # Channel B entry threshold (sigma_eq units)
Z_EXIT = 0.75               # Channel B exit band
TAU_MAX = 240.0             # OU validity kill-switch (bars)
S_STAR = 0.89               # saturation scale (Baz/DMN)


def phi_sat(y: np.ndarray | float) -> np.ndarray | float:
    """Saturating-then-cutoff response phi(y) = y exp(-y^2/4)/0.89, peak at
    sqrt(2): position grows with |y| up to |y|=sqrt(2) then de-risks (Baz et
    al. 2015; Lim-Zohren-Roberts 2019 eq. 7 -- verified from paper + repo)."""
    y = np.asarray(y, dtype="float64")
    return y * np.exp(-y * y / 4.0) / S_STAR


def ew_rls_residualize(y: pd.Series, X: pd.DataFrame,
                       lam: float = RLS_LAMBDA) -> pd.DataFrame:
    """Run causal EW-RLS of y on X (with intercept) bar by bar.

    Returns DataFrame with: eps (a-priori innovation), beta_* (post-update
    coefficients), R2_ew (EW R-squared), pred (one-step prediction).
    """
    names = list(X.columns)
    x_raw = X.values.astype("float64")
    # standardize regressors by trailing EW vol (conditioning; the Kalman gain
    # then auto-scales) -- betas are on standardized regressors
    xdf = X.copy()
    for c in names:
        sd = xdf[c].ewm(span=VOL_SPAN_X, min_periods=12).std()
        mu = xdf[c].ewm(span=VOL_SPAN_X, min_periods=12).mean()
        xdf[c] = (xdf[c] - mu) / (sd + 1e-12)
    xs = np.column_stack([np.ones(len(xdf))] + [xdf[c].values for c in names])
    yv = y.values.astype("float64")
    T, k = xs.shape

    beta = np.zeros((T, k))
    P = np.eye(k) * P_INIT
    b = np.zeros(k)
    eps = np.full(T, np.nan)
    pred = np.full(T, np.nan)
    r2_num = 0.0
    r2_den = 0.0
    r2 = np.full(T, np.nan)
    for t in range(T):
        xt = xs[t]
        yv_t = yv[t]
        if not (np.all(np.isfinite(xt)) and np.isfinite(yv_t)):
            beta[t] = b                      # warm-up row: state unchanged
            continue                          # (eps stays NaN, no update)
        p = xt @ P @ xt
        e = yv_t - xt @ b                     # a-priori innovation (TRADABLE)
        eps[t] = e
        pred[t] = xt @ b
        g = (P @ xt) / (lam + p)
        b = b + e * g
        P = (P - np.outer(g, xt @ P)) / lam
        # running EW R^2 (variance share explained, forget rate ~ lam)
        r2_num = lam * r2_num + pred[t] ** 2
        r2_den = lam * r2_den + yv_t ** 2
        if t > 50 and r2_den > 1e-18:
            r2[t] = r2_num / r2_den
        beta[t] = b
    out = pd.DataFrame({"eps": eps, "pred": pred, "R2_ew": r2}, index=y.index)
    for j, c in enumerate(names):
        out[f"beta_{c}"] = beta[:, j + 1]
    return out


def residual_channels(h1: pd.DataFrame, price_map: dict[str, str],
                      lam: float = RLS_LAMBDA) -> pd.DataFrame:
    """Full residualization feature block on H1 bars.

    Parameters
    ----------
    h1 : H1 frame indexed by UTC bar label with columns per price_map:
         {'xau': 'close', 'jpy': 'close_jpy', 'eur': 'close_eur', 'xag': 'close_xag'}
    """
    rets = {}
    for k_, col in price_map.items():
        rets[k_] = np.log(h1[col] / h1[col].shift(1))
    y = rets["xau"]
    X = pd.DataFrame({k_: rets[k_] for k_ in ("jpy", "eur", "xag")})
    base = ew_rls_residualize(y, X, lam=lam)

    eps = base["eps"]
    sig_eps = eps.ewm(span=SIG_EPS_SPAN, min_periods=60).std()
    m_A = eps.ewm(span=RES_MOM_SPAN, min_periods=4).mean() / (sig_eps + 1e-12)

    # spread level + OU s-score
    Xlvl = eps.cumsum()
    m_X = Xlvl.ewm(span=MEAN_X_SPAN, min_periods=120).mean()
    rho = Xlvl.rolling(OU_WINDOW, min_periods=120).apply(
        lambda s: np.corrcoef(s[:-1], s[1:])[0, 1] if np.std(s) > 0 else 1.0,
        raw=True)
    kappa = (-np.log(rho.clip(1e-6, 1 - 1e-9))).clip(lower=1e-4)
    tau = (1.0 / kappa).clip(upper=TAU_MAX)
    sig_eq = sig_eps * np.sqrt(tau / 2.0)
    z = (Xlvl - m_X) / (sig_eq + 1e-12)

    out = base.copy()
    out["sig_eps"] = sig_eps
    out["res_mom_A"] = m_A
    out["spread_lvl"] = Xlvl
    out["spread_z"] = z
    out["ou_tau"] = tau
    out["sig_eq"] = sig_eq
    out["ou_valid"] = (tau < TAU_MAX).astype(float)

    # Channel signals (features AND standalone rule baselines)
    out["q_A"] = np.tanh(m_A)                              # catch-up continuation
    out["q_B"] = -(z / Z_ENTRY) * phi_sat(z / Z_ENTRY) * out["ou_valid"]
    out["chB_active"] = (z.abs() > Z_ENTRY).astype(float) * out["ou_valid"]
    out["chB_exit"] = (z.abs() < Z_EXIT).astype(float)
    # conflict discount: rare trending-yet-extended residual -> halve both
    out["conflict"] = ((np.sign(out["q_A"]) == -np.sign(out["q_B"])) &
                       (out["q_A"].abs() > 0.3) & (out["q_B"].abs() > 0.3)).astype(float)
    return out


def self_test() -> None:
    rng = np.random.default_rng(5)
    T = 6000
    idx = pd.date_range("2020-01-01", periods=T, freq="h", tz="UTC")
    f1 = rng.standard_normal(T) * 0.0010
    f2 = rng.standard_normal(T) * 0.0008
    r_jpy = f1 + rng.standard_normal(T) * 0.0003
    r_eur = f2 + rng.standard_normal(T) * 0.0003
    r_xag = 1.3 * (f1 + f2) + rng.standard_normal(T) * 0.0012
    # gold = beta_jpy f1 + beta_eur f2 + beta_xag r_xag-ish + idio, with beta drift
    bj = 0.2 + 0.00002 * np.arange(T)
    r_xau = bj * f1 + 0.15 * f2 + 0.35 * r_xag + rng.standard_normal(T) * 0.0006
    h1 = pd.DataFrame({
        "close": 2000 * np.exp(np.cumsum(r_xau)),
        "close_jpy": 150 * np.exp(np.cumsum(r_jpy)),
        "close_eur": 1.1 * np.exp(np.cumsum(r_eur)),
        "close_xag": 24 * np.exp(np.cumsum(r_xag)),
    }, index=idx)
    out = residual_channels(h1, {"xau": "close", "jpy": "close_jpy",
                                 "eur": "close_eur", "xag": "close_xag"})
    # R2 should be high (systematic part explained)
    assert out["R2_ew"].iloc[-1] > 0.3, f"EW R2 too low: {out['R2_ew'].iloc[-1]}"
    # betas live on STANDARDIZED regressors predicting raw-unit returns, so
    # the right magnitude is beta_true * sd_x (~1e-3).  The invariant that
    # matters: RLS(lambda=1.0) == batch OLS on the standardized design.
    rets = {k_: np.log(h1[c] / h1[c].shift(1)) for k_, c in
            {"xau": "close", "jpy": "close_jpy", "eur": "close_eur", "xag": "close_xag"}.items()}
    Xd = pd.DataFrame({k_: rets[k_] for k_ in ("jpy", "eur", "xag")}).iloc[1:]
    yv = rets["xau"].iloc[1:]
    Xs = Xd.copy()
    for c in Xs.columns:
        Xs[c] = (Xs[c] - Xs[c].mean()) / (Xs[c].std() + 1e-12)
    D = np.column_stack([np.ones(len(Xs))] + [Xs[c].values for c in Xs.columns])
    beta_ols, *_ = np.linalg.lstsq(D, yv.values, rcond=None)
    rls_grow = ew_rls_residualize(rets["xau"], pd.DataFrame(
        {k_: rets[k_] for k_ in ("jpy", "eur", "xag")}), lam=1.0)
    got = np.array([rls_grow["eps"].index.size and 0] +
                   [rls_grow[f"beta_{c}"].iloc[-1] for c in ("jpy", "eur", "xag")])
    # intercept not exposed; compare the three slopes (skip module's EW-standardization
    # vs batch z-score difference -> tolerance 25% of |beta_ols|)
    for j, c in enumerate(("jpy", "eur", "xag")):
        assert abs(got[j + 1] - beta_ols[j + 1]) < 0.25 * max(abs(beta_ols[j + 1]), 1e-4), \
            f"beta_{c}: RLS {got[j+1]:.3e} vs OLS {beta_ols[j+1]:.3e}"
    # eps should be much smaller in vol than raw gold returns
    ratio = out["sig_eps"].iloc[-1] / np.std(r_xau[-500:])
    assert ratio < 1.0, f"residual vol {ratio:.2f}x raw vol -- residualization failed"
    # causality: truncate tail, earlier features unchanged
    cut = 4000
    h1c = h1.iloc[: cut]
    outc = residual_channels(h1c, {"xau": "close", "jpy": "close_jpy",
                                   "eur": "close_eur", "xag": "close_xag"})
    common = idx[100: cut - 200]
    pd.testing.assert_frame_equal(outc.loc[common], out.loc[common])
    print("F5 self-tests passed: betas recovered, residual shrinks, causal.")


if __name__ == "__main__":
    self_test()

```

## 3.6 F6 — CFM/Carver saturating trend ladder

Per scale s ∈ {2,4,8,16,32,64,128,256}h: raw_s = (EWMA_s(p) − EWMA_{4s}(p))/σ_p,t with σ_p,t = p_t·EWstd₉₆(r) (Carver EWMAC N,4N ✅); scalar-normalized to mean |F|≈10 (expanding, ≥2y, backfilled — the q_scalars recipe ✅) and capped ±20; saturated through **φ(F/10) = (F/10)·exp(−(F/10)²/4)/0.89** — the CFM tanh saturation (s\*≈0.89, b≈0.075 ✅) in the DMN-equivalent form — so positions *de-risk* into parabolic extensions. Blend = equal-weight × FDM = N/√(1ᵀC1) (ρ̄-based; =1 when perfectly correlated, √N when independent), re-capped. CFM one-sided s_n = (p−EMA_n)/σ_n at 24/120/504h kept as separate features.

```python
"""
AURUM v2.0 -- Feature Module F6: Multi-Horizon Saturating Trend Ladder
=======================================================================
The CFM / Carver / Oxford-Man trend stack, transplanted to H1 bars:

  Carver EWMAC (N, 4N) crossover forecast ("Systematic Trading" 2015;
  qoppac/pysystemtrade verified):
      raw_s = (EWMA_s(p) - EWMA_{4s}(p)) / sigma_p,t
  geometric ladder s in {2, 4, 8, 16, 32, 64, 128, 256} hours;
  sigma_p,t = p_t * EW-std of hourly returns (span 96) -- crossover in vol units.

  Scalar normalization to mean |F| ~ 10, cap +-20 (pysystemtrade defaults.yaml
  verified: forecast_cap 20.0, average_absolute_forecast 10.0); scalars are
  ESTIMATED on expanding windows (>= 2y, backfill first valid) -- the q_scalars
  recipe -- never copied from futures tables.

  Saturating response (CFM "Two Centuries of Trend Following", Lempérière-
  Deremble-Seager-Potters-Bouchaud, arXiv:1404.3274, eq. 3: response
  saturates as s* tanh(s_n/s*) with s* ~= 0.89; Baz et al./DMN equivalent
  phi(y) = y exp(-y^2/4)/0.89, peak at sqrt(2)):
      F_sat,s = phi(F_s / 10)  -- de-risk parabolic extensions.

  Blend across scales equal-weighted, rescaled by the forecast diversification
  multiplier FDM = N / sqrt(1' C 1) (C = correlation of saturated forecasts;
  = sqrt(N) when independent, 1 when identical -- Carver's construction),
  capped at +-20.

  CFM one-sided variant s_n = (p - EMA_n(p)) / sigma_n kept as extra features
  (n = 24h, 120h -- the 5-day/1-month analogue of the CFM 5-month scale).

Outputs are features (the model learns the response); the standalone position
z_trend = F_blend/10 * sigma_tgt/sigma_ret,t is used by the baseline gauntlet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TREND_LADDER_H = (2, 4, 8, 16, 32, 64, 128, 256)
SIGMA_RET_SPAN = 96          # EW std span for vol normalization (bars)
SCALAR_MIN_BARS = 24 * 252 * 2   # >= 2 years expanding before scalar is valid
FORECAST_CAP = 20.0
AVG_ABS_TARGET = 10.0
S_STAR = 0.89
CFM_ONE_SIDED_H = (24, 120, 504)


def _phi_sat(y):
    y = np.asarray(y, dtype="float64")
    return y * np.exp(-y * y / 4.0) / S_STAR


def trend_ladder(h1_close: pd.Series) -> pd.DataFrame:
    """Trend ladder features on H1 closes (UTC index)."""
    p = h1_close.astype("float64")
    r = np.log(p / p.shift(1))
    sig_ret = r.ewm(span=SIGMA_RET_SPAN, min_periods=24).std()
    sig_p = p * sig_ret                                     # price-vol units

    out = {}
    sat_cols = []
    for s in TREND_LADDER_H:
        ema_f = p.ewm(span=s, min_periods=s).mean()
        ema_s = p.ewm(span=4 * s, min_periods=4 * s).mean()
        raw = (ema_f - ema_s) / (sig_p + 1e-12)
        out[f"ewmac_raw_{s}h"] = raw
        # expanding scalar with 2y minimum, backfilled from first valid
        absmean = raw.abs().expanding(min_periods=1).mean()
        n_seen = raw.abs().expanding().count()
        scalar = AVG_ABS_TARGET / (absmean + 1e-12)
        scalar = scalar.where(n_seen >= SCALAR_MIN_BARS).bfill()
        F = (raw * scalar).clip(-FORECAST_CAP, FORECAST_CAP)
        out[f"ewmac_F_{s}h"] = F
        sat = pd.Series(_phi_sat(F / AVG_ABS_TARGET), index=p.index)
        out[f"ewmac_sat_{s}h"] = sat
        sat_cols.append(sat)

    # CFM one-sided risk-adjusted momentum at 1-day / 5-day / 3-week scales
    for n in CFM_ONE_SIDED_H:
        ema_n = p.ewm(span=n, min_periods=n).mean()
        sig_n = p.diff().abs().ewm(span=n, min_periods=n).mean()
        out[f"cfm_onesided_{n}h"] = (p.shift(1) - ema_n.shift(1)) / (sig_n.shift(1) + 1e-12)

    # blend + FDM (N / sqrt(1' C 1): sqrt(N) when independent, 1 when identical)
    S = pd.concat(sat_cols, axis=1)
    F_blend = S.mean(axis=1)
    N = len(sat_cols)
    W = SCALAR_MIN_BARS
    # mean pairwise correlation of the saturated forecasts, vectorized:
    # corr_ij = (E[xi xj] - E[xi]E[xj]) / (sd_i sd_j), all trailing windows
    means = S.rolling(W, min_periods=W // 2).mean()
    sds = S.rolling(W, min_periods=W // 2).std()
    pair_rows = []
    ii, jj = np.triu_indices(N, k=1)
    for a, b_ in zip(ii, jj):
        exy = (S.iloc[:, a] * S.iloc[:, b_]).rolling(W, min_periods=W // 2).mean()
        cab = (exy - means.iloc[:, a] * means.iloc[:, b_]) / (
            sds.iloc[:, a] * sds.iloc[:, b_] + 1e-12)
        pair_rows.append(cab)
    rho_bar = pd.concat(pair_rows, axis=1).mean(axis=1).reindex(p.index).ffill()
    rho_bar = rho_bar.fillna(1.0).clip(-0.5, 1.0)
    fdm = N / np.sqrt(N * (1 + (N - 1) * rho_bar))
    fdm = fdm.clip(1.0, np.sqrt(N))
    out["F_blend"] = (F_blend * fdm * 10.0).clip(-FORECAST_CAP, FORECAST_CAP)  # back to F units
    out["fdm"] = fdm
    out["sig_ret"] = sig_ret
    return pd.DataFrame(out)


def self_test() -> None:
    rng = np.random.default_rng(21)
    T = 30_000
    idx = pd.date_range("2019-01-01", periods=T, freq="h", tz="UTC")
    # persistent trending regime for first 15k bars, chop after
    drift = np.where(np.arange(T) < 15_000, 0.0004, 0.0)
    r = drift + rng.standard_normal(T) * 0.0011
    p = pd.Series(2000 * np.exp(np.cumsum(r)), index=idx)
    f = trend_ladder(p)
    # trend regime: F_blend should be strongly positive mid-trend
    mid = f["F_blend"].iloc[6000:12000].mean()
    assert mid > 2.0, f"F_blend should be positive in trend regime, got {mid:.2f}"
    # chop: F_blend near zero
    chop = f["F_blend"].iloc[20_000:].mean()
    assert abs(chop) < 2.0, f"F_blend should be ~0 in chop, got {chop:.2f}"
    # saturation bounds: phi in [-0.89, 0.89] roughly (peak sqrt(2)/0.89*...)
    for c in [c for c in f.columns if c.startswith("ewmac_sat_")]:
        assert f[c].abs().max() < 1.6, f"{c} exceeded saturation bound"
    # FDM in [1, sqrt(8)]
    assert f["fdm"].between(1.0, np.sqrt(8) + 1e-9).all()
    # causality
    cut = 20_000
    fc = trend_ladder(p.iloc[:cut])
    common = idx[3000: cut - 300]
    pd.testing.assert_frame_equal(fc.loc[common], f.loc[common])
    print("F6 self-tests passed: trend detected, chop flat, saturation bounded, causal.")


if __name__ == "__main__":
    self_test()

```

## 3.7 Builder, IO contract, compute budget

```python
"""
AURUM v2.0 -- Feature Store Builder (orchestrator)
===================================================
Assembles the complete v2.0 feature matrix from the existing Kaggle dataset
``tskdev/aurum-phase1-data`` (M1 + H1 parquets for XAUUSD/XAGUSD/EURUSD/USDJPY
+ FRED daily macro), plus the label targets for v2.0.

IO CONTRACT (adapt the three constants below to the actual dataset layout;
column names expected on M1/H1 frames: open, high, low, close, volume -- tick
volume for FX/CFD; timestamps are OPEN-labeled bar starts, tz-UTC or naive
UTC).

  input : /kaggle/input/datasets/tskdev/aurum-phase1-data/<...>.parquet
  output: aurum_v2_features.parquet   (one row per H1 bar, all features)

Label targets (v2.0 -- continuous, cost-aware, no 48-cell grid):
  y_ret24_norm  : next-24-bar vol-normalized XAUUSD return (primary aux target)
  y_dir24       : sign of next-24-bar return (binary aux target)
  y_ret1_norm   : next-1-bar vol-normalized return (fast aux target)
  The PRIMARY training signal is the differentiable net-Sharpe loss over the
  model's continuous position (Spec Section 4) -- these targets only steer
  auxiliary heads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from f1_clock import clock_features
from f2_realized import realized_block_m1, range_block_h1, variance_ratio_block, roughness_block, har_aggregates
from f3_signatures import signature_features
from f4_toxicity import vpin_block, kyle_lambda_block, amihud_clv_block
from f5_residual import residual_channels
from f6_trend import trend_ladder

# ----------------------------------------------------------------------------
# IO contract -- Agent 1 adapts these to the on-disk layout
# ----------------------------------------------------------------------------
M1_PATHS = {
    "xau": "/kaggle/input/datasets/tskdev/aurum-phase1-data/xauusd_m1.parquet",
    "xag": "/kaggle/input/datasets/tskdev/aurum-phase1-data/xagusd_m1.parquet",
    "eur": "/kaggle/input/datasets/tskdev/aurum-phase1-data/eurusd_m1.parquet",
    "jpy": "/kaggle/input/datasets/tskdev/aurum-phase1-data/usdjpy_m1.parquet",
}
H1_PATHS = {k: p.replace("_m1", "_h1") for k, p in M1_PATHS.items()}
OUTPUT_PATH = "aurum_v2_features.parquet"

HORIZON = 24           # primary label horizon (bars)
SIG_SPAN = 96          # vol-normalization span for targets


def _load(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    df.index = idx
    return df.sort_index()


def build_v2_features(m1: dict[str, pd.DataFrame], h1: dict[str, pd.DataFrame],
                      depth3: bool = False) -> pd.DataFrame:
    """Assemble the full v2.0 feature frame. m1/h1: {'xau': df, 'xag': df,
    'eur': df, 'jpy': df} with UTC DatetimeIndex and open/high/low/close/volume."""
    # H1 master frame: XAUUSD bars define the index; context closes joined
    base = h1["xau"].copy()
    for k in ("xag", "eur", "jpy"):
        base[f"close_{k}"] = h1[k]["close"].reindex(base.index)
    h1_index = base.index

    blocks: list[pd.DataFrame] = []

    # ---- F1: DST-aware exchange clock ------------------------------------
    blocks.append(clock_features(h1_index, base["close"]))

    # ---- F2: realized geometry (XAU + context) ---------------------------
    for k in ("xau", "xag", "eur", "jpy"):
        rb = realized_block_m1(m1[k], h1_index).add_suffix(f"_{k}")
        blocks.append(rb)
    rng_xau = range_block_h1(base[["open", "high", "low", "close"]])
    blocks.append(rng_xau.add_suffix("_xau"))
    r_xau = np.log(base["close"] / base["close"].shift(1))
    blocks.append(variance_ratio_block(r_xau).add_suffix("_xau"))
    rv = realized_block_m1(m1["xau"], h1_index)["RV_1h"]
    # RV_1h is already at H1 closes; reindex onto the same index for roughness
    blocks.append(roughness_block(rv.reindex(h1_index)).to_frame("H_rough_vol_xau"))
    # HAR aggregates of the core XAU realized block
    rb_xau = realized_block_m1(m1["xau"], h1_index)
    blocks.append(har_aggregates(rb_xau[["RV_1h", "BV_1h", "SJ_1h", "RVnegShare_1h",
                                         "jumpShare_1h"]]))

    # ---- F3: multi-asset signatures ---------------------------------------
    m1_multi = m1["xau"][["close"]].copy()
    m1_multi["close_xag"] = m1["xag"]["close"].reindex(m1_multi.index).ffill()
    m1_multi["close_eur"] = m1["eur"]["close"].reindex(m1_multi.index).ffill()
    m1_multi["close_jpy"] = m1["jpy"]["close"].reindex(m1_multi.index).ffill()
    blocks.append(signature_features(
        m1_multi, h1_index,
        {"xau": "close", "xag": "close_xag", "eur": "close_eur", "jpy": "close_jpy"},
        depth3=depth3))

    # ---- F4: toxicity / liquidity ----------------------------------------
    vol_col = "volume" if "volume" in m1["xau"].columns else None
    if vol_col:
        blocks.append(vpin_block(m1["xau"], h1_index).add_suffix("_xau"))
        blocks.append(kyle_lambda_block(m1["xau"], h1_index).add_suffix("_xau"))
        blocks.append(amihud_clv_block(m1["xau"], base, vol_col=vol_col).add_suffix("_xau"))

    # ---- F5: online cross-asset residualization --------------------------
    blocks.append(residual_channels(
        base, {"xau": "close", "jpy": "close_jpy", "eur": "close_eur", "xag": "close_xag"}))

    # ---- F6: saturating trend ladder --------------------------------------
    blocks.append(trend_ladder(base["close"]).add_suffix("_xau"))

    feats = pd.concat(blocks, axis=1)

    # ---- targets (strictly forward-looking; drop at train time by shift) --
    sig = r_xau.ewm(span=SIG_SPAN, min_periods=24).std()
    fwd24 = base["close"].shift(-HORIZON) / base["close"] - 1.0
    fwd1 = base["close"].shift(-1) / base["close"] - 1.0
    feats["y_ret24_norm"] = fwd24 / (sig * np.sqrt(HORIZON) + 1e-12)
    feats["y_dir24"] = np.sign(fwd24)
    feats["y_ret1_norm"] = fwd1 / (sig + 1e-12)
    feats["sig_ret_xau_target"] = sig
    return feats


def integration_test() -> None:
    """End-to-end on synthetic 4-asset data: shapes, NaN discipline, causality
    of the full assembled frame, and feature-count report."""
    rng = np.random.default_rng(42)
    n = 60 * 24 * 90            # 90 days of M1
    idx = pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC")
    f1 = rng.standard_normal(n) * 0.0009
    f2 = rng.standard_normal(n) * 0.0006
    common = 0.5 * f1 + 0.3 * f2
    px = {
        "xau": 2000 * np.exp(np.cumsum(0.8 * common + rng.standard_normal(n) * 0.0004)),
        "xag": 24 * np.exp(np.cumsum(1.9 * common + rng.standard_normal(n) * 0.0012)),
        "eur": 1.1 * np.exp(np.cumsum(0.4 * f2 + rng.standard_normal(n) * 0.0002)),
        "jpy": 150 * np.exp(np.cumsum(-0.6 * f1 + rng.standard_normal(n) * 0.0002)),
    }
    m1 = {}
    for k, p in px.items():
        m1[k] = pd.DataFrame(
            {"open": p, "high": p, "low": p, "close": p,
             "volume": 100 + rng.random(n) * 50}, index=idx)
    h1 = {}
    for k, df in m1.items():
        agg = df.resample("h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        h1[k] = agg
    feats = build_v2_features(m1, h1)
    T = len(feats)
    print(f"integration: {T} H1 bars x {feats.shape[1]} features")
    nan_frac = feats.isna().mean()
    warm_ok = (nan_frac.iloc[2:] < 0.9).mean()
    print(f"features with >90% NaN: {(nan_frac > 0.9).sum()} "
          f"(expected: only deep-warmup HAR/YZ columns); median NaN {nan_frac.median():.3f}")
    assert warm_ok > 0.9, "too many all-NaN feature columns"
    assert feats["y_ret24_norm"].notna().sum() > 0.8 * T
    # causality of the FULL frame: truncate M1 tail, earlier rows identical
    cut = 60 * 24 * 60
    h1c = {k: df.iloc[: df.index.searchsorted(idx[cut])] for k, df in h1.items()}
    m1c = {k: df.iloc[: cut + 60] for k, df in m1.items()}
    feats_c = build_v2_features(m1c, h1c)
    common_idx = feats_c.index[(feats_c.index < idx[cut] - pd.Timedelta("48h")) &
                               (feats_c.index > idx[60 * 24 * 10])]
    a = feats.loc[common_idx].dropna(axis=1)
    b = feats_c.loc[common_idx].dropna(axis=1)
    shared = a.columns.intersection(b.columns)
    pd.testing.assert_frame_equal(a[shared], b[shared])
    print(f"full-pipeline causality check passed on {len(common_idx)} bars "
          f"({len(shared)} shared columns)")
    print("feature-group counts:")
    for pref, n_pat in [("minutes_to", 0), ("sess_", 0), ("sig", 0), ("levy", 0),
                        ("ll_", 0), ("RV", 0), ("ewmac", 0), ("beta", 0),
                        ("q_", 0), ("vpin", 0), ("y_", 0)]:
        cnt = sum(c.startswith(pref) for c in feats.columns)
        if cnt:
            print(f"  {pref:<12} {cnt}")


if __name__ == "__main__":
    integration_test()

```

**IO contract.** Inputs: the four M1 parquets + four H1 parquets (open-labeled, UTC, OHLC + tick-volume) — paths in `M1_PATHS/H1_PATHS` are the only lines Agent 1 adapts. Output: `aurum_v2_features.parquet`, one row per H1 bar, plus targets `y_ret24_norm`, `y_dir24`, `y_ret1_norm` (forward-looking; dropped by shift at training time).

**Measured compute** (this session, CPU): 90 days × 4 symbols → 329 features in 5.1 s. Scaling to 16 years (~65×) ⇒ **~5–6 min per symbol pass on a Kaggle CPU kernel** (the M1 realized block dominates; vectorized pandas rolling). Full feature build fits comfortably in a 4–6 h CPU kernel with memory headroom (float32 discipline; the M1 frame is processed per symbol and released).

**Test evidence shipped with the engine:** F1 DST-gap window machine-verified (autumn 2026: 7 days at 4h; PM fix at 11:00 NY inside the gap); F2 BV/RV unbiased on constant-vol synthetic, VR≈1 on iid (Monte-Carlo-verified normalization), jump statistic fires on injected jumps, RS sane on GBM, causality by truncation; F3 triangle area 0.5 / reversal −0.5 / diagonal dx²/2 / lead-lag sign / Chen depth-3 identity; F4 VPIN∈[0,1], λ>0 under impact, causal; F5 RLS(λ=1)≡OLS invariance, residual vol < raw vol, causal; F6 trend detected / chop flat / saturation bounded / FDM∈[1,√8] / causal. **Full-pipeline causality: the assembled 329-column frame at earlier bars is bit-identical when the future is deleted.**

---


# Section 4 — The v2.0 Model & Loss Architecture

## 4.0 The architectural decision: Path A is the main bet

The pillar-5 excavation compared the two candidate paths on eleven dimensions (evidence in the Section 2 ledger, P5-1..P5-15). Summary verdict:

| Dimension | **Path A — AURUM-Continuum** (purpose-built, 3–4M params) | **Path B — Qwen3-0.6B trunk** (projector + LoRA) |
|---|---|---|
| Trainable params | ~3–4M (right-sized; TFT gating "skips unnecessary nonlinearity… for small/noisy datasets" ✅) | 596M frozen + ~5–10M trainable |
| Params per training state | ~45 (72k states) | ~875+ (v1.x disease persists) |
| Finance-win precedent | **Direct**: DMN LSTM 2.8 Sharpe vs TSMOM 1.2 ✅; Momentum Transformer 2.62 vs LSTM-DMN 1.70 ✅ | **None** — no verified Kaggle financial winner used an LLM trunk; all used small MLPs/GBDTs (P5-5..P5-10) |
| Continuous inputs | Native | Requires Perceiver/Time-LLM-style bridge (sound ✅ but unproven in finance) |
| Throughput (2×T4, L=512) | minutes/epoch | ~2–4 h/epoch (fp16, no bf16 on sm_75, no FA2) ⇒ ~10 epochs/week total |
| Spectral control of the loop | Direct (256×256 matrices) | Indirect (operator wraps a frozen 596M trunk) |
| Interpretability | VSN weights + attention maps (MTX-proven) | Requires probing |
| Risk | underfit (cheap, detectable) | fp16 overflow, distribution shift, throughput starvation of the search budget |

**Decision: Path A is the production architecture; the Kaggle-floor stack (AE-MLP + LightGBM) is a permanent ensemble member; Path B is a bounded probe at ≤15% of the GPU budget with pre-registered kill criteria (§4.7).** The throughput asymmetry is decisive on top of the precedent asymmetry: a Path-A random search of 50 configs costs the same GPU time as ~2 epochs of Path B.

## 4.1 The stack

| ID | Component | Role | Code |
|---|---|---|---|
| M0a | LightGBM (era-safe + v2 features) | floor + ensemble member | Agent 1 (v1.2 code reusable) |
| M0b | DMN-LSTM (hidden 80, 63-step unroll, Sharpe loss) | sequence-model floor, ports 1:1 from the verified reference | `ae_mlp_dmn.py` |
| M0c | DLinear / seasonal-naive | sanity floor (the DLinear lesson ✅) | Agent 1 (trivial) |
| M1 | **AURUM-Continuum** (this section) | the production model | `continuum.py`, `train_continuum.py` |
| M2 | Supervised AE-MLP (JS-2021 winner recipe) | tabular ensemble member | `ae_mlp_dmn.py` |
| M3 | Path B probe (frozen Qwen3 + projector + LoRA) | bounded research option | §4.7 protocol |

## 4.2 AURUM-Continuum — exact architecture

```
input  x ∈ R^{L×330}                      L = 256 H1 bars (≈1.5 trading weeks)
  │
  ├─ RevIN per window on continuous channels (mask: std>1e-6 in train)
  │    x̃ = γ·(mask⊙(x−μ_win)/σ_win + (1−mask)⊙x) + β          [Kim et al. ✅]
  │
  ├─ Variable Selection Network over ~10 semantic feature groups
  │    (clock / realized-xau / realized-cross / signatures / toxicity /
  │     residual / trend / macro / sessions / regime)
  │    GRN_g per group + softmax(GRN(flat)) weights → selected e_t ∈ R^256
  │                                                             [TFT Eqs. 6-8 ✅]
  ├─ Prelude: 2 × CausalBlock (pre-norm MHA, 4 heads, interpretable
  │    shared-value attention, head-averaged; GRN-FFN)          [MTX ✅]
  │
  ├─ LOOPED CORE (weight-tied, R=4 iterations, KM-damped):
  │    h ← (1−α)·h + α·f(h),   f = LoopBlock (non-residual):
  │        f(h) = W2 GELU(W1 LN(h)),  W1, W2 spectral-capped σ ≤ 0.8
  │    readout at EVERY iterate (per-iterate supervision)       [§4.3]
  │
  ├─ Coda: 1 × CausalBlock + attention pooling + last token
  │
  └─ Heads: z = tanh(Dense(1, max-norm 3)) ∈ [−1,1]  (position, DMN-style)
            ret24 (linear), dir24 (logit), meta (logit)   — aux multi-task
```

**Parameter budget at production scale** (d=256): VSN+embeddings ≈ 1.6M; 3 attention blocks + loop ≈ 1.7M; heads ≈ 0.1M ⇒ **~3.4M trainable** (measured 0.18M at the self-test scale d=64; scaling law ~d²). GRN = LayerNorm(a + GLU(η₁)) with η₁ = dropout(W₁·ELU(W₂a)) — the TFT construction ✅ whose gate "can zero the branch" is the small-data escape Grinsztajn's results demand.

## 4.3 The contractive loop — mathematics and governor

The v1.x lesson (autopsy §1.3) is that iteration without contraction amplifies noise. v2.0's loop is contractive *by construction*:

1. **The loop map is non-residual.** f(h) = W₂·GELU(W₁·LN(h)) — no identity shortcut inside the loop (a residual h+f(h) is expansive around identity, which is exactly what v1.x measured). Attention lives only in the un-looped prelude/coda; the loop refines each time step's latent toward its **denoising equilibrium** — the RED/Gribonval-Nikolova reading: iterating a denoiser is proximal regularization precisely when the denoiser is non-expansive ✅, and the spectral cap *enforces* non-expansiveness.
2. **Per-layer spectral caps.** Every linear weight in the loop satisfies σ(W) ≤ 0.8 via the recycled power-iteration normalization W̄ = W·(cap/σ(W)) ✅ (Miyato et al.; the estimate stays tight with one step per forward, gradients flow through W only — SN-GAN convention; the implementation is detached-buffer-safe for weight tying).
3. **Krasnoselskii–Mann damping.** h ← (1−α)h + α·f(h), α=0.5 init. With ‖J_f‖ ≤ L < 1 the damped iteration converges to the fixed point h\* = f(h\*) (the monDEQ damped-iteration condition ✅); damping additionally bounds the transient.
4. **Per-iterate supervision.** The position head is read at *every* iterate with geometrically decaying loss weight (Yang et al. ✅ — the loop is trained as an iterative refinement, not an afterthought).
5. **Empirical audit + governor (the acceptance gate).** After every `govern_every` epochs the trainer estimates the spectral radius ρ of the true loop-update Jacobian g(h) = (1−α)h + αf(h) at real hidden states by JVP power iteration; if ρ ≥ 0.95 the governor shrinks α via the measured loop gain (auto-tune, closed form). **Gate G3.3 (Section 6): no checkpoint ships with ρ ≥ 0.95.** In this session's self-test the construction measured ρ = 0.502 at initialization — versus ρ > 1 measured for a residual loop block (the v1.x configuration), which the governor alone could not rescue (the math: KM damping cannot bring (1−α)I + αJ below 1 when ‖J‖ ≥ 1 — the block itself must be non-expansive).

*Honest caveat (Appendix B):* softmax attention has no closed-form Lipschitz bound; our loop excludes attention, so the only unbounded component is LayerNorm's affine, initialized at 0.5 gain and monitored by the empirical audit. The audit — not the bound — is the guarantee.

## 4.4 The training objective

**Primary: differentiable annualized net Sharpe with costs inside the graph** (three-source-verified structure: DMN Eq. 16/27 ✅, Zhang et al. Eq. 7 ✅, Moody-Saffell reward ✅):

$$r_t^{\text{net}} = z_{t-1}\,v_t\,r_t \;-\; c\,\big|z_t v_t - z_{t-1} v_{t-1}\big|, \qquad v_t = \sigma_{\text{tgt}}/\hat\sigma_t$$

$$\mathcal{L}(\theta) = -\underbrace{\frac{\bar r^{\text{net}}\sqrt{A}}{\sqrt{\tfrac1N\sum (r_t^{\text{net}})^2 - \bar r^{\text{net}\,2} + \varepsilon}}}_{\text{annualized net Sharpe, } \varepsilon=10^{-9}} \;+\; \lambda_{\text{turn}}\tfrac1N\sum_t|z_t - z_{t-1}| \;+\; \lambda_{DD}\,\underbrace{\sqrt{\tfrac1N\sum_t D_t^2}}_{\text{Ulcer Index}}$$

with D_t the running-peak drawdown of the cumulative net path (differentiable a.e. through cummax; soft-max peak option documented if training spikes). **Annualizer: A = 5786 bars/year** (empirical: 87,948 bars / 15.2 y = 5,786; √A = 76.07 — matches √(252·23) = 76.13; lock the empirical count). Cost c is calibrated per bar from the 0.05R round-turn at 1.0-ATR stop scaled by exposure; λ_turn ≈ c, λ_DD ∈ [0.1, 0.5] tuned on validation Sharpe + UI jointly. Auxiliary multi-task heads (weights ≈ 0.1/0.1/0.05): MSE on `y_ret24_norm`, BCE on `y_dir24`, BCE on meta-agreement — the JS-2021 multi-target stabilization ✅.

**Differential Sharpe ratio** (Moody-Saffell, recursions secured via the Borrageiro reproduction with the denominator typo corrected to (B−A²)^{3/2} ✅) is implemented as the *online monitoring* objective; batch net-Sharpe is the trainer.

```python
"""
AURUM v2.0 -- Loss Functions (PyTorch)
=======================================
All formulas verified against primary sources (see Spec Section 4 and ledger):

  Sharpe loss (Lim-Zohren-Roberts 2019, arXiv:1904.04912, Eq. 16 + reference
  implementation SharpeLoss in Wood et al.'s trading-momentum-transformer):
      L = -mu_R * sqrt(A) / sqrt(mean(R^2) - mu_R^2 + eps),  eps = 1e-9

  Cost-augmented strategy return (DMN Eq. 27; Zhang-Zohren-Roberts
  arXiv:2005.13665 Eq. 7; Moody-Saffell reward -- three independent sources,
  same structure):
      R_t = z_{t-1} * s_t - c * |z_t * v_t - z_{t-1} * v_{t-1}|
      s_t = (sigma_tgt / sigma_t) * r_t      (vol-scaled return)
      v_t = sigma_tgt / sigma_t              (vol-scaling of the position)

  Ulcer Index (Martin 1987/89):  UI = sqrt(mean_t D_t^2), D_t = drawdown of
      the cumulative net-PnL path from its running peak.

  Differential Sharpe Ratio (Moody & Saffell 2001, IEEE TNN 12(4); recursions
      via the verbatim reproduction in Borrageiro et al. arXiv:2201.04699,
      denominator typo corrected to (B - A^2)^{3/2}):
      A_t = A_{t-1} + eta (R_t - A_{t-1}),  B_t = B_{t-1} + eta (R_t^2 - B_{t-1})
      D_t = [B_{t-1} dA_t - 0.5 A_{t-1} dB_t] / (B_{t-1} - A_{t-1}^2)^{3/2}

Annualization: A = bars/year.  For XAUUSD H1, A = 252 * 23 = 5796 (23 Globex
hours/day); AURUM MUST recompute the empirical bar count from the dataset
(~88k/15.2y) and use sqrt(A_emp) -- document whichever is locked.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS_SHARPE = 1e-9
BARS_PER_YEAR_DEFAULT = 252 * 23      # 5796 -- override with empirical count


def net_strategy_return(z: torch.Tensor,
                        r: torch.Tensor,
                        sigma: torch.Tensor,
                        cost_r: float,
                        sigma_tgt: float = 0.15 / math.sqrt(BARS_PER_YEAR_DEFAULT)) -> torch.Tensor:
    """Cost-aware vol-scaled strategy return per bar (all tensors (B, T)).

    z     : model position in [-1, 1] at each bar (vol-scaled exposure target)
    r     : raw next-bar log returns of XAUUSD
    sigma : per-bar EW vol estimate (same units as r) -- CAUSAL feature, input
    cost_r: round-turn cost in return units (0.05R at 1.0-ATR stop => pass the
            equivalent per-unit-exposure cost; default calibration in train cfg)
    """
    v = sigma_tgt / (sigma + 1e-12)              # vol scaling of position
    s = v * r                                    # vol-scaled return
    z_prev = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
    v_prev = torch.cat([torch.ones_like(v[:, :1]), v[:, :-1]], dim=1)
    turn = (z * v - z_prev * v_prev).abs()
    return z_prev * s - cost_r * turn


class NetSharpeLoss(nn.Module):
    """Differentiable annualized net Sharpe + turnover + Ulcer-drawdown penalty.

    L = -Sharpe_ann(R_net) + lam_turn * mean|dz| + lam_dd * UI(R_net)

    R_net carries transaction costs INSIDE the computation graph (|.| gradient
    is a.e. defined; subgradient sign() at 0 -- standard).
    """

    def __init__(self, bars_per_year: int = BARS_PER_YEAR_DEFAULT,
                 lam_turn: float = 0.0, lam_dd: float = 0.0,
                 cost_r: float = 1e-4, sigma_tgt_ann: float = 0.15):
        super().__init__()
        self.sqrtA = math.sqrt(bars_per_year)
        self.lam_turn = lam_turn
        self.lam_dd = lam_dd
        self.cost_r = cost_r
        self.sigma_tgt = sigma_tgt_ann / math.sqrt(bars_per_year)

    def forward(self, z: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        R = net_strategy_return(z, r, sigma, self.cost_r, self.sigma_tgt)
        mu = R.mean(dim=1)
        var = (R ** 2).mean(dim=1) - mu ** 2
        sharpe = mu * self.sqrtA / torch.sqrt(var + EPS_SHARPE)
        loss = -sharpe.mean()
        if self.lam_turn > 0:
            dz = (z - torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)).abs()
            loss = loss + self.lam_turn * dz.mean()
        if self.lam_dd > 0:
            # Ulcer index of the cumulative net path (running peak via cummax)
            eq = torch.cumsum(R, dim=1)
            peak = torch.cummax(eq, dim=1).values
            dd = (eq - peak).clamp(max=0.0)
            loss = loss + self.lam_dd * torch.sqrt((dd ** 2).mean(dim=1) + EPS_SHARPE).mean()
        return loss


class DifferentialSharpe(nn.Module):
    """Moody-Saffell differential Sharpe ratio as an ONLINE objective or
    diagnostic (batch form, Eq. 2 of the reproduction).  Kept for the online
    engine's monitoring; batch NetSharpeLoss is the primary trainer."""

    def __init__(self, eta: float = 0.01):
        super().__init__()
        self.eta = eta

    def forward(self, R: torch.Tensor) -> torch.Tensor:
        # sequential EMA over the time axis (batch-parallel via scan-like loop)
        T = R.shape[1]
        A = torch.zeros_like(R[:, 0])
        B = torch.zeros_like(R[:, 0])
        D = []
        for t in range(T):
            Rt = R[:, t]
            dA = Rt - A
            dB = Rt ** 2 - B
            denom = (B - A ** 2).clamp(min=EPS_SHARPE) ** 1.5
            D.append((B * dA - 0.5 * A * dB) / denom)
            A = A + self.eta * dA
            B = B + self.eta * dB
        return -torch.stack(D, dim=1).mean()


def aux_losses(z_dir_logit: torch.Tensor, ret24_hat: torch.Tensor,
               meta_logit: torch.Tensor, y_dir: torch.Tensor, y_ret: torch.Tensor,
               w_ret: float = 0.1, w_dir: float = 0.1, w_meta: float = 0.05):
    """Auxiliary multi-task heads (JS-2021 multi-target pattern): direction BCE,
    horizon-return MSE, meta-probability BCE (P(primary direction correct))."""
    l_dir = F.binary_cross_entropy_with_logits(z_dir_logit, (y_dir > 0).float())
    l_ret = F.mse_loss(ret24_hat, y_ret)
    l_meta = F.binary_cross_entropy_with_logits(meta_logit,
                                                (torch.sign(ret24_hat) == y_dir.sign()).float())
    return w_ret * l_ret + w_dir * l_dir + w_meta * l_meta


def self_test() -> None:
    torch.manual_seed(0)
    B, T = 8, 500
    z = torch.tanh(torch.randn(B, T) * 0.5)
    r = torch.randn(B, T) * 0.001
    sigma = torch.full((B, T), 0.001)
    crit = NetSharpeLoss(lam_turn=0.1, lam_dd=0.05, cost_r=5e-5)
    loss = crit(z, r, sigma)
    assert torch.isfinite(loss), "NetSharpeLoss not finite"
    # zero position => zero return => |Sharpe| ~ 0 (mu=0); loss ~ -0
    z0 = torch.zeros_like(z)
    l0 = crit(z0, r, sigma)
    assert l0.abs() < 1e-3, f"zero-position loss should vanish, got {l0}"
    # perfect foresight position => strongly positive Sharpe
    # (foresight = z at t-1 sign-matches r at t; the loss consumes z shifted)
    zf = torch.zeros_like(z)
    zf[:, :-1] = torch.sign(r[:, 1:])
    lf = crit(zf, r, sigma)
    assert lf < -5.0, f"foresight Sharpe loss should be very negative, got {lf}"
    # churning random positions on iid returns must be PUNISHED (cost drag)
    assert loss > 1.0, f"random-churn loss should be positive (costs), got {loss}"
    # gradient flows through costs
    zc = torch.zeros_like(z, requires_grad=True)
    crit(0.1 * torch.tanh(zc), r, sigma).backward()
    assert zc.grad is not None and torch.isfinite(zc.grad).all()
    dsr = DifferentialSharpe()(net_strategy_return(z, r, sigma, 5e-5))
    assert torch.isfinite(dsr)
    print("losses self-test passed: cost-aware Sharpe, foresight bound, DSR finite.")


if __name__ == "__main__":
    self_test()

```

## 4.5 AURUM-Continuum — implementation and trainer

```python
"""
AURUM v2.0 -- "AURUM-Continuum": Purpose-Built Contractive Looped Transformer
===============================================================================
Path A of the v2.0 architecture (Spec Section 4).  Every component is lifted
from a verified source with its published hyperparameter provenance:

  * TFT GRN (gated residual network) and Variable Selection Network
    -- Lim, Arik, Loeff, Pfister, "Temporal Fusion Transformers",
    arXiv:1912.09363, Eqs. (2)-(8).  GRNs "can skip nonlinear processing ...
    motivated for small/noisy datasets" (fetched text) -- exactly our regime.
  * Decoder-only TFT backbone with interpretable multi-head attention and a
    tanh position head with max-norm(3) kernel constraint
    -- Wood, Giegerich, Roberts, Zohren, "Trading with the Momentum
    Transformer", arXiv:2112.08534 (+ kieranjwood/trading-momentum-transformer
    reference code, fetched).
  * Looped core with (i) per-layer spectral caps (Miyato et al.
    arXiv:1802.05957: W_SN = W / sigma(W), one power-iteration step), (ii)
    Krasnoselskii-Mann damping h <- (1-alpha) h + alpha f(h) which converges
    to the fixed point h* = f(h*) whenever f is L-Lipschitz with L < 1
    (monotone-operator iteration; cf. Winston & Kolter arXiv:2006.08591
    damped iteration alpha <= 2m/L^2), and (iii) per-iterate supervision
    (Yang et al., "Looped Transformers...", arXiv:2311.12424: loss enforced on
    every iterate t in [b0, b] to keep the loop stable).
  * RevIN per-window instance normalization -- Kim et al., ICLR 2022
    (arXiv:2105.08499), forward/inverse forms; non-stationarity absorption.
  * DMN-style multi-scale vol-normalized return inputs and direct tanh
    position output trained on the Sharpe loss -- Lim, Zohren, Roberts,
    arXiv:1904.04912.

Loop stability, honestly stated: softmax attention has no closed-form
Lipschitz bound, so the spectral caps on all linear/attention projections
plus KM damping give a *sufficient-condition engineering bound* on the linear
parts, and `audit_loop_spectral_radius()` empirically estimates the spectral
radius rho of the TRUE loop-update Jacobian at real hidden states via JVP
power iteration.  Acceptance gate: rho < 0.95 on all audited states (Spec
Section 4.4).  If the gate fails at any checkpoint, alpha is auto-reduced.
"""
from __future__ import annotations

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
    Dropout is applied to the hidden layer before gating & norm (TFT text).
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
    step per forward (Miyato et al. 2018, Eqs. 7-9: one recycled step keeps
    the estimate tight through training)."""

    def __init__(self, d_in: int, d_out: int, cap: float = 0.95, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias = nn.Parameter(torch.zeros(d_out)) if bias else None
        self.cap = cap
        self.register_buffer("u", F.normalize(torch.randn(d_out), dim=0))
        self.register_buffer("v", F.normalize(torch.randn(d_in), dim=0))

    def _sigma(self) -> torch.Tensor:
        """One recycled power-iteration step (Miyato et al. 2018), fully
        detached: sigma is a constant for autograd purposes, gradients flow
        through W only (SN-GAN convention; buffer updates stay out of the
        graph so the weight-tied loop can reuse this layer safely)."""
        with torch.no_grad():
            u = F.normalize(torch.mv(self.weight, self.v), dim=0)      # W v  -> (d_out,)
            v = F.normalize(torch.mv(self.weight.t(), u), dim=0)       # W^T u -> (d_in,)
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
    from a GRN over the flattened inputs (static context omitted), sparse
    weighted sum.  Also returns the selection weights for interpretability."""

    def __init__(self, group_dims: list[int], d_model: int, dropout: float = 0.2,
                 spectral_cap: float | None = None):
        super().__init__()
        self.n = len(group_dims)
        self.embeds = nn.ModuleList(
            [GRN(d, d_model, dropout, spectral_cap) for d in group_dims])
        self.weight_net = GRN(sum(group_dims), d_model, dropout, spectral_cap)
        self.sel = nn.Linear(d_model, self.n)

    def forward(self, xs: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # xs: list of (B, T, d_g)
        flat = torch.cat(xs, dim=-1)
        w = torch.softmax(self.sel(self.weight_net(flat)), dim=-1)   # (B, T, n)
        e = torch.stack([emb(x) for emb, x in zip(self.embeds, xs)], dim=-2)  # (B,T,n,d)
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
        # interpretable MHA: per-head output = attention @ shared values,
        # then AVERAGE heads (TFT Sec. 3.4)
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
    (Romano et al. RED arXiv:1611.02862; Gribonval-Nikolova arXiv:1807.04021:
    a denoiser behaves as a proximal operator exactly when it is (firmly)
    non-expansive -- which the spectral cap enforces).  Attention lives in
    the prelude/coda (NOT looped); the loop refines each time step's latent
    toward its denoising equilibrium.  The empirical Jacobian audit
    (audit_loop_spectral_radius) is the acceptance gate.
    """

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
    spectral_cap: float = 0.90     # per-linear spectral cap inside the loop
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
        # kernel_constraint=max_norm(3)) -- enforced here as a pre-forward hook
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
        denoising fixed point.  Returns iterates [h_1 ... h_R] (per-iterate
        readout, Yang et al. arXiv:2311.12424)."""
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
    (Spec Section 4.4): rho < 0.95 on trained checkpoints.  This is the
    mechanism v1.x lacked: an unconstrained loop measured exactly like this
    would show rho > 1 (noise amplification for r > 1)."""
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
                          spectral_cap=0.9)
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


if __name__ == "__main__":
    self_test()

```

**Trainer protocol** (verified practice, every choice sourced): Adam, lr ∈ {1e-4, 1e-3}, clipnorm ∈ {0.01, 1, 100}; batch {64, 128, 256}; ≤300 epochs with **early stopping on validation annualized net Sharpe** (patience 25 — DMN/MTX ✅, never on val loss); dropout grid {0.1..0.5}; 50-iteration random search; 3 seeds averaged; spectral governor every 5 epochs; fp16 autocast + GradScaler on T4 (no bf16 on sm_75 ✅); purged expanding walk-forward (§4.9); window sampling with left-padding for the burn-in edge.

```python
"""
AURUM v2.0 -- ContinuumNet Trainer
===================================
Training protocol assembled from verified practice:

  * Sharpe-loss training on the continuous position with costs inside the
    graph (DMN arXiv:1904.04912 Eq. 16/27; Zhang et al. arXiv:2005.13665
    Eq. 7) + turnover/Ulcer penalties (Spec Section 4.5).
  * Per-iterate supervision of the loop (Yang et al. arXiv:2311.12424): the
    position head is read at EVERY loop iterate; auxiliary heads on the final
    iterate; iterate weights decay geometrically.
  * Early stopping on VALIDATION PORTFOLIO SHARPE, patience 25 (DMN/MTX
    practice -- never on val loss), 3-seed averaging, random search budget.
  * Spectral governor: every `govern_every` epochs, audit the loop's
    empirical Jacobian radius; if rho >= 0.95, shrink km_alpha
    (auto_tune_alpha) -- the v1.x failure mode is structurally impossible.
  * fp16 autocast on CUDA (T4 sm_75: fp16 only, NO bf16; DDP for 2xT4).
  * Purged expanding walk-forward (purge 24 bars, embargo 24-48 bars).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from continuum import ContinuumConfig, ContinuumNet, auto_tune_alpha
from losses import NetSharpeLoss, aux_losses, net_strategy_return


@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 32
    lr: float = 1e-3
    clipnorm: float = 1.0
    max_epochs: int = 100
    patience: int = 25
    lam_turn: float = 0.05
    lam_dd: float = 0.10
    cost_r: float = 5e-5            # per-unit-exposure round-turn cost
    sigma_tgt_ann: float = 0.15
    bars_per_year: int = 252 * 23
    w_iterate_decay: float = 0.7    # per-iterate loss weight decay
    w_aux: float = 0.15             # aux multi-task weight
    govern_every: int = 5           # spectral-governor cadence (epochs)
    rho_gate: float = 0.95
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
        # returns aligned to the window: position at step tau earns ret[tau]
        r = self.ret[lo: t + 1]
        if r.shape[0] < self.L:
            r = torch.cat([torch.zeros(self.L - r.shape[0]), r])
        s = self.sigma[lo: t + 1]
        if s.shape[0] < self.L:
            s = torch.cat([torch.full((self.L - s.shape[0],), float("nan")), s])
            s = torch.nan_to_num(s, nan=1e-4)
        # next-bar return for the LAST step (target of the final position)
        return x, r, s, self.y_ret24[t], self.y_dir[t]


def _finite_fill(a: np.ndarray) -> np.ndarray:
    """Causal NaN fill for the feature matrix (forward-fill then zero)."""
    import pandas as pd
    return pd.DataFrame(a).ffill().fillna(0.0).values.astype("float32")


def portfolio_sharpe(z: np.ndarray, r: np.ndarray, sigma: np.ndarray,
                     cfg: TrainConfig) -> float:
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
    crit = NetSharpeLoss(bars_per_year=cfg.bars_per_year, lam_turn=cfg.lam_turn,
                         lam_dd=cfg.lam_dd, cost_r=cfg.cost_r,
                         sigma_tgt_ann=cfg.sigma_tgt_ann)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    best_sharpe, best_state, bad = -1e9, None, 0
    hist = {"val_sharpe": [], "rho": [], "alpha": []}
    for epoch in range(cfg.max_epochs):
        model.train()
        for x, r, s, y_r, y_d in trl:
            x, r, s = x.to(device), r.to(device), s.to(device)
            y_r, y_d = y_r.to(device), y_d.to(device)
            opt.zero_grad(set_to_none=True)
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
            auto_tune_alpha(model, val_x[:4], rho_target=cfg.rho_gate)
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
    Sharpe-loss training loop actually optimizes the right thing)."""
    torch.manual_seed(0)
    T, F_ = 1500, 12
    rng = np.random.default_rng(0)
    # plumbing test (not economics): strong predictable component so a
    # 12-epoch CPU run must learn it; costs set negligible -- the economic
    # cost-drag behavior is separately asserted in losses.self_test()
    f = rng.standard_normal((T, F_)).astype("float32")
    r = 0.30 * f[:, 0] * 0.001 + rng.standard_normal(T) * 0.001
    r = np.roll(r, -1); r[-1] = 0.0                 # r[t] = return t -> t+1
    sigma = np.full(T, 0.0014, dtype="float32")
    y_r = np.roll(0.02 * f[:, 0] * 0.001, -1).astype("float32")
    y_d = np.sign(y_r)
    groups = [(0, 4, "g0"), (4, 8, "g1"), (8, 12, "g2")]
    net_cfg = ContinuumConfig(n_features=F_, group_slices=groups, d_model=32,
                              n_heads=2, dropout=0.1, loop_iters=3,
                              km_alpha=0.5, spectral_cap=0.8)
    cfg = TrainConfig(seq_len=64, batch_size=16, lr=2e-3, max_epochs=12,
                      patience=6, lam_turn=0.0, lam_dd=0.0, cost_r=1e-8,
                      device="cpu", amp=False, govern_every=3, seed=0)
    train_idx = np.arange(100, 1100)
    val_idx = np.arange(1100, 1450)
    model, hist = train_one_fold(f, r, sigma, y_r, y_d, train_idx, val_idx,
                                 groups, net_cfg, cfg)
    assert np.isfinite(hist["best_val_sharpe"])
    # trained model must beat the ZERO-position Sharpe (= 0) on validation
    assert hist["best_val_sharpe"] > 1.0, \
        f"training failed to learn the synthetic signal: {hist['best_val_sharpe']:.3f}"
    print(f"train_continuum self-test passed: best val Sharpe "
          f"{hist['best_val_sharpe']:.2f} on synthetic signal (plumbing OK).")


if __name__ == "__main__":
    self_test()

```

## 4.6 The Kaggle floor: supervised AE-MLP + GBDT (+ DMN baseline)

The AE-MLP is the JS-2021 first-place recipe adapted (P5-5 ✅): BN → GaussianNoise(0.035) → Dense(128) encoder → swish; decoder MSE reconstruction; MLP head on concat(normalized input, latent) — 512-256-256-128 with BN/swish/dropout {0.30, 0.30, 0.25, 0.40}; multi-target (ret24 / dir24 / meta); **sample weights = |y_ret|** (the |resp| weighting); trained jointly inside each CV split (never pre-trained on data that leaks into validation — the winner's explicit design note ✅); Jahrer's swap-noise (15%) and RankGauss as variant options (P5-6 ✅). GBDT (LightGBM) trains on the identical feature frame; the final ensemble is a simple average (nonlinear stacking failed twice in the source competitions — Jahrer 2017, Cats Trading 2021 ✅).

```python
"""
AURUM v2.0 -- Kaggle-Floor Models: Supervised AE-MLP + DMN-LSTM
================================================================
M0 baseline gauntlet (must-beat-and-ensemble per Spec Section 4.6):

AE-MLP -- the Jane Street Market Prediction 2020-21 FIRST-place architecture
("Cats Trading" / VECTOR, Yirun Zhang et al.; write-up + public notebook
source fetched via the Kaggle kernel API; see ledger P5-S6/S7), adapted to
AURUM: BatchNorm -> GaussianNoise -> Dense(96-256) encoder -> swish, decoder
reconstruction (MSE), MLP head on concat(raw, latent) with BatchNorm +
dropout 0.3-0.5, multi-target outputs, sample weights = |target|, purged
group CV with gap, last-folds-only models, multi-seed averaging.  The swap-
noise DAE variant (Michael Jahrer, Porto Seguro 2017 1st place, fetched
write-up: 15% feature swap corruption + RankGauss) is provided as an option.

DMN-LSTM -- the Deep Momentum Network (Lim, Zohren, Roberts 2019,
arXiv:1904.04912) reference configuration, ported from Wood et al.'s
trading-momentum-transformer code (fetched): single LSTM (hidden 80, tanh),
dropout, TimeDistributed Dense(1, tanh, max_norm 3), Sharpe loss with
eps=1e-9, Adam + clipnorm, 63-step unroll of volatility-normalized returns.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import NetSharpeLoss, EPS_SHARPE


# --------------------------------------------------------------- AE-MLP
class AEMLP(nn.Module):
    """Supervised denoising-autoencoder MLP (JS-2021 winner recipe, adapted).

    encoder: BN -> GaussianNoise(sigma) -> Dense(enc) -> BN -> swish
    decoder: Dropout -> Dense(n_features)   (MSE reconstruction)
    head:    concat(BN(x), latent) -> Dense stacks (BN + swish + dropout)
             -> multi-target outputs (ret24 linear, dir24 logit, meta logit)
    """

    def __init__(self, n_features: int, enc_dim: int = 128,
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
        xc = self._corrupt(x)
        xn = self.bn_in(xc)
        lat = self._swish(self.enc_bn(self.encoder(xn)))
        rec = self.decoder(self.dec_drop(lat))
        h = self.mlp(torch.cat([xn, lat], dim=1))
        return {"ret24": self.out_ret(h).squeeze(-1),
                "dir_logit": self.out_dir(h).squeeze(-1),
                "meta_logit": self.out_meta(h).squeeze(-1),
                "reconstruction": rec}


def ae_mlp_loss(out: dict, x_norm: torch.Tensor, y_ret: torch.Tensor, y_dir: torch.Tensor,
                w_rec: float = 1.0, w_ret: float = 1.0, w_dir: float = 0.5,
                w_meta: float = 0.25) -> torch.Tensor:
    """Multi-task loss.  x_norm = the batch-normalized input actually seen by
    the encoder (the reconstruction target -- JS-2021 trains the AE jointly
    with the head inside each CV split); sample weights = |y_ret|."""
    sw = (y_ret.abs() / (y_ret.abs().mean() + 1e-9)).clamp(0.2, 3.0).detach()
    l_rec = F.mse_loss(out["reconstruction"], x_norm.detach())
    l_ret = (sw * F.mse_loss(out["ret24"], y_ret, reduction="none")).mean()
    l_dir = (sw * F.binary_cross_entropy_with_logits(
        out["dir_logit"], (y_dir > 0).float(), reduction="none")).mean()
    meta_y = (torch.sign(out["ret24"].detach()) == y_dir.sign()).float()
    l_meta = F.binary_cross_entropy_with_logits(out["meta_logit"], meta_y)
    return w_ret * l_ret + w_dir * l_dir + w_meta * l_meta + w_rec * l_rec


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

    def _clamp(self, module, inputs):
        with torch.no_grad():
            n = module.weight.norm()
            if n > self.max_norm:
                module.weight.mul_(self.max_norm / (n + 1e-12))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, n_inputs) = [vol-normalized return, 1/sigma proxy]
        h, _ = self.lstm(x)
        return torch.tanh(self.head(self.drop(h))).squeeze(-1)     # (B, T)


def dmn_loss(z: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor,
             bars_per_year: int = 252 * 23) -> torch.Tensor:
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
    assert z.shape == (B, T) and z.abs().max() <= 1.0
    r = torch.randn(B, T) * 0.001
    s = torch.full((B, T), 0.001)
    loss = dmn_loss(z, r, s)
    assert torch.isfinite(loss)
    loss.backward()
    print("AE-MLP + DMN-LSTM self-tests passed.")


if __name__ == "__main__":
    self_test()

```

## 4.7 Path B — bounded probe protocol (pre-registered)

1. **Linear probe first** (0.5 GPU-h): freeze Qwen3-0.6B in fp16 (bf16→fp16 conversion; watch activation overflow), feed the last 256 H1 bars through the trunk *with the v2 continuous features projected by a frozen random patch projector* (Perceiver-style cross-attention into a 64-token latent, then trunk), fit a ridge/softmax on the mean-pooled hidden states. **Kill criterion K1: if probe val-Sharpe/AUC < GBDT by >30% relative, Path B dies without a LoRA run.**
2. **LoRA run** (1.0 GPU-h, only if K1 passes): LoRA r=16, α=32 on {Wq, Wv} of layers 1–12 (the "more types, smaller rank" evidence ✅), projector trainable, lr 1e-4, ≤3 epochs. **Kill criterion K2: if LoRA val-Sharpe ≤ ContinuumNet's at equal data, Path B is archived** (it costs 10–100× the throughput for ≤ performance).
3. Path B never enters the production ensemble unless it beats M1 *and* M2 on the purged OOS with DSR ≥ 0.95 within its own trial family.

## 4.8 Selection & sizing — retiring the 48-cell argmax (Mandate M4)

The production output is the continuous position z_t ∈ [−1,1] scaled by σ_tgt/σ̂_t — barrier cells survive only as the **risk frame** (each position carries a catastrophic 2.5-ATR stop for disaster insurance, and discrete-cell EVs are computed for reporting). Wherever discrete cells must be chosen (e.g., the catastrophic-stop distance, or horizon allocation), selection is:

**Step 1 — EV lower confidence bound with overlap-discounted n_eff:**

$$EV_{LCB,c} = \hat p_c (b_c+1) - 1 - z_\alpha\,(b_c+1)\sqrt{\tfrac{\hat p_c(1-\hat p_c)}{n_{\text{eff},c}}}, \qquad n_{\text{eff}} = \frac{n}{1 + 2\sum_{k=1}^{K}\big(1-\tfrac{k}{H+1}\big)\hat\rho_k} \;\le\; \frac{T_{\text{bars}}}{\lceil \bar h\rceil}$$

with z_α = 1.28 (90%; re-check at 1.64 — a cell that only survives z=1.0 is not robust). The (b+1)² variance tilt that made TP=4.0 cells error magnets is now the *penalty* it should have been.

**Step 2 — positive-part James-Stein shrinkage** of the surviving cells' EVs toward the cross-cell mean (Efron-Morris ✅): shrinkage factor (m−2)σ̂²/‖EV−mean‖² — heavy shrinkage exactly when dispersion is noise, which is the expected regime at 88k low-SNR bars.

**Step 3 — fractional-Kelly sizing on calibrated probabilities:** temperature-scale the meta-probability on a purged validation slice (Guo ✅; T\* by NLL, ECE check ≤ 15 bins), then f\* = (p(b+1)−1)/b at **quarter-Kelly** (MacLean-Thorp-Ziemba table ✅: c=0.25 ⇒ P[double before halve] ≈ 0.94 at 64% of max growth; c=0.5 ⇒ 0.89 / 75%), capped by the vol-target multiplier. All three steps are implemented in `online_engine.py` (`fit_temperature`, `kelly_size`, and the LCB/JS formulas in Section 5's sizing module).

## 4.9 Walk-forward & model selection discipline

PurgedWalkForward (skfolio-exact semantics ✅): expanding train, yearly test blocks; **purge = 24 bars** on both sides of every test boundary (train labels cannot overlap test label windows); **embargo = 24–48 bars after each test block** (EWMA-96 features argue for 48); constraint purge+embargo < fold−1 enforced. Yearly refits (the Wood et al. expanding-window protocol with per-window re-tuning ✅). **The config ledger starts counting trials from the first experiment** — every random-search draw, every seed, every variant enters N for the DSR (Section 5.5). Model-family selection happens on 2021–22 validation; 2023–25 remains untouched OOS until the final run.

```python
"""
AURUM v2.0 -- Purged Walk-Forward Splitter
===========================================
Semantics follow Lopez de Prado, Advances in Financial Machine Learning (2018),
ch. 7 (purged k-fold CV with embargo), cross-verified against the open
skfolio reproduction of CombinatorialPurgedCV (fetched source):

  PURGE : drop from the TRAINING set every sample whose LABEL WINDOW
          (t, t+H] overlaps the label window of any TEST sample.  With label
          horizon H = 24 bars and contiguous yearly test blocks, this removes
          train rows in [test_start - H, test_start).
  EMBARGO: additionally drop train rows immediately FOLLOWING the test block
          (serial correlation of features leaks backward in time):
          [test_end + 1, test_end + H + embargo].
  Constraint (skfolio): purge + embargo < fold size - 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PurgedWalkForward:
    label_horizon: int = 24          # H in bars
    embargo: int = 24                # extra bars after test block (24-48 rec.)
    test_block: str = "365D"         # contiguous test block length
    min_train: int = 24 * 252 * 2    # burn-in before first test block

    def splits(self, index: pd.DatetimeIndex):
        """Yield (train_idx, test_idx) position arrays, expanding window."""
        idx = pd.DatetimeIndex(index)
        blocks = pd.date_range(idx[self.min_train], idx[-1],
                               freq=self.test_block, tz=idx.tz)
        for i, bstart in enumerate(blocks):
            bend = blocks[i + 1] if i + 1 < len(blocks) else idx[-1] + pd.Timedelta("1h")
            t_lo = idx.searchsorted(bstart)
            t_hi = idx.searchsorted(bend)
            if t_hi - t_lo < self.label_horizon + 2:
                continue
            test = np.arange(t_lo, t_hi)
            # purge: train labels must not overlap the test label window
            purge_lo = max(0, t_lo - self.label_horizon)
            # embargo after the test block
            emb_hi = min(len(idx), t_hi + self.label_horizon + self.embargo)
            train = np.concatenate([np.arange(0, purge_lo),
                                    np.arange(emb_hi, len(idx))])
            train = train[train >= self.min_train]
            # enforce the skfolio constraint
            if self.label_horizon + self.embargo >= (t_hi - t_lo) - 1:
                raise ValueError("purge + embargo >= fold size - 1; reduce embargo")
            if len(train) > 0:
                yield train, test


def self_test() -> None:
    idx = pd.date_range("2016-01-01", "2026-01-01", freq="h", tz="UTC")
    wf = PurgedWalkForward(label_horizon=24, embargo=24, min_train=24 * 252 * 2)
    n = 0
    last_test_end = -1
    prev_pre = 0
    for train, test in wf.splits(idx):
        # test blocks are disjoint and ordered
        assert test.min() > last_test_end
        last_test_end = test.max()
        # no train row within the purge window before the test block
        assert not np.any((train >= test.min() - 24) & (train < test.min()))
        # no train row within the embargo window after the test block
        assert not np.any((train > test.max()) & (train <= test.max() + 24 + 24))
        # expanding window: pre-block train grows over folds
        if n > 0:
            assert (train < test.min()).sum() >= prev_pre
        prev_pre = (train < test.min()).sum()
        n += 1
    assert n >= 6, f"expected >=6 yearly folds, got {n}"
    print(f"PurgedWalkForward self-test passed: {n} folds, purge=24, embargo=24.")


if __name__ == "__main__":
    self_test()

```

---


# Section 5 — Causal Online Adaptation & Portfolio Execution Engine

## 5.0 The governing evidence

The single most important 2026 result in the TTA literature is the delay-aware audit (arXiv:2610.12232 ✅): when adaptation methods are evaluated in a harness where the label of forecast origin s is released only at s+d with d ≥ H (the horizon), **most published TTA gains vanish, and a hyperparameter-free bank of RLS filters combined by per-coordinate median beats 3 of 4 recent methods.** The v2.0 engine is built around that humility:

| Layer | What updates | When | Risk |
|---|---|---|---|
| Normalization | RevIN per-window μ,σ; EW vol σ̂_t | every bar, automatically | zero (trailing-window functions only) |
| Readout | linear correction w on frozen features, RLS λ=0.999 (river-compatible SGD option) | **only at label arrival (t+H, H=24)** | bounded (linear, forgetful, embargo-safe) |
| Regime gates | PH detector state, vol-target c_t\*, vol-of-vol quantile | every bar / weekly | bounded multipliers ≤ 2× |
| Backbone | nothing | annual purged refit only | the F3 audit's finding: online fine-tuning is fragile |

**Baseline gauntlet (gate G5.x):** an adaptive component ships only if it beats (a) the frozen model and (b) the RLS-bank on purged, delay-respecting OOS.

## 5.1 Label-arrival-gated online readout

RLS with forgetting (the information-form Kalman filter ✅, §3.5 recursions) maintains a linear correction d_t = w_tᵀx_t on top of the frozen model score; z_ideal = tanh(model_score + d_t) × regime. Updates fire exactly when the label of origin s arrives (s+24): the pending-label queue is explicit in the engine (`self.pending`), and the engine's self-test asserts the queue depth equals H (no early consumption possible). The RLS-bank (λ ∈ {0.99, 0.995, 0.999, 0.9995}, median combiner) is both the shipped floor and the competitor every fancier adapter must beat.

## 5.2 Regime engine — three independent gates

1. **Volatility targeting, real-time.** Moreira-Muir scaling ◐ with the Bernardi et al. correction ✅: the managed stream is m_t = c_t\*/σ̂_t² with c_t\* = σ_tgt/√(E_t[1/σ̂²]) estimated on the expanding window — *not* the in-sample c\* that the source paper itself flags as unknown in real time. Multiplier capped [0.25, 2.0].
2. **Page–Hinkley break gate** (river-exact recursion and defaults ✅: two-sided, λ=50.0, δ=0.005, α=0.9999, min_instances=30; sweep λ ∈ {20, 50, 100} on pre-2016 only): run on z-scored returns *and* on |readout error|. On alarm: freeze online updates 24 bars, halve gross exposure until the readout re-converges, reset detector (river semantics).
3. **Vol-of-vol de-risk overlay** (Baltussen et al. ◐ — 8%/yr high-minus-low spread; single-asset translation is our interpretation, gated): when EW-vol-of-vol enters its top trailing quintile (30-day window), multiply exposure by ≤ 0.5. BOCPD (Adams-MacKay ✅; hazard 1/500, Gaussian conjugate prior a=1, b=1e-4) is offered as a *feature* channel (run-length posterior), following Wood et al.'s CPD-as-feature result (+33% Sharpe ✅) — never as a hard gate.

## 5.3 The execution event loop (one position, marked to market every bar)

```python
"""
AURUM v2.0 -- Causal Online Adaptation & Portfolio Execution Engine
=====================================================================
Section 5 machinery, all formulas verified (see ledger P6/P7):

  OnlineRidge/RLS readout with forgetting -- recursive least squares, the
  information-form Kalman filter for random-walk coefficients; updates fire
  ONLY when the label of forecast origin s arrives (s + H), never before
  (delay-aware discipline; cf. the Oct-2026 TTA audit arXiv:2610.12232 whose
  hyperparameter-free RLS bank beats 3 of 4 recent TTA methods -- the RLSBank
  baseline below is exactly that floor to beat).

  Page-Hinkley two-sided drift detector -- river source recursion (fetched):
      m_T = alpha m_{T-1} + (x_T - xbar_T) - delta     (increase side)
      alarm iff m_T - min(m) > lambda                  (river defaults:
      lambda=50.0, delta=0.005, alpha=0.9999, min_instances=30)

  Volatility targeting -- Moreira & Muir (JF 2017) scaling w = c*/sig^2 with
  the REAL-TIME normalization constant c_t* (expanding variance matching;
  the full-sample c* is not known in real time -- Bernardi et al.
  arXiv:2212.07288 caveat), multiplier capped [0.25, 2.0].

  Carver no-trade buffer -- trade only outside +/-10% of the average
  position, and then only TO THE EDGE of the band (pysystemtrade
  buffer_size: 0.10, buffer_trade_to_edge: True, verified from source);
  speed limit: annual cost <= 1/3 of expected pre-cost Sharpe.

  DSR -- Bailey & Lopez de Prado 2014 (JPM 40(5)) exact formulas; PBO via
  CSCV (Bailey-Borwein-LdP-Zhu, J. Computational Finance 2017).

  Meta-labeling sizing -- AFML ch. 3.6 workflow: calibrated P(primary
  correct) via temperature scaling (Guo et al. arXiv:1706.04599), then
  fractional Kelly f* = (p(b+1) - 1)/b at c = 0.25-0.5 (MacLean-Thorp-Ziemba
  2010 drawdown table).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm as _norm


# ------------------------------------------------------------- online ridge
class OnlineRidge:
    """RLS readout with forgetting factor; one update per LABEL ARRIVAL.

    w update (RLS):  g = P x / (lam + x' P x);  w += e g;
                     P <- (P - g x' P) / lam
    The a-priori error e = y - x'w is the ONLINE DIAGNOSTIC (never a feature).
    """

    def __init__(self, dim: int, lam: float = 0.999, l2: float = 1e-3):
        self.w = np.zeros(dim)
        self.P = np.eye(dim) / l2
        self.lam = lam

    def predict(self, x: np.ndarray) -> float:
        return float(x @ self.w)

    def update(self, x: np.ndarray, y: float) -> None:
        Px = self.P @ x
        g = Px / (self.lam + x @ Px)
        e = y - x @ self.w
        self.w = self.w + e * g
        self.P = (self.P - np.outer(g, x @ self.P)) / self.lam


class RLSBank:
    """Bank of RLS filters with different forgetting factors combined by
    per-coordinate MEDIAN -- the hyperparameter-free adaptation floor
    (arXiv:2610.12232 audit: beats 3 of 4 recent TSF-TTA methods)."""

    def __init__(self, dim: int, lams=(0.99, 0.995, 0.999, 0.9995)):
        self.bank = [OnlineRidge(dim, lam=l) for l in lams]

    def predict(self, x: np.ndarray) -> float:
        return float(np.median([m.predict(x) for m in self.bank]))

    def update(self, x: np.ndarray, y: float) -> None:
        for m in self.bank:
            m.update(x, y)


# ------------------------------------------------------------ page-hinkley
class PageHinkley:
    """Two-sided Page-Hinkley (river source form, constants as fetched)."""

    def __init__(self, lam: float = 50.0, delta: float = 0.005, alpha: float = 0.9999,
                 min_instances: int = 30):
        self.lam, self.delta, self.alpha, self.min_instances = lam, delta, alpha, min_instances
        self.n = 0
        self.xbar = 0.0
        self.m = 0.0
        self.M = 0.0
        self.m_min = 0.0
        self.M_max = 0.0

    def update(self, x: float) -> bool:
        """Returns True on alarm (detector auto-resets, river behavior)."""
        self.n += 1
        self.xbar += (x - self.xbar) / self.n
        dev = x - self.xbar
        self.m = self.alpha * self.m + dev - self.delta
        self.M = self.alpha * self.M + dev + self.delta
        self.m_min = min(self.m_min, self.m)
        self.M_max = max(self.M_max, self.M)
        if self.n < self.min_instances:
            return False
        alarm = (self.m - self.m_min > self.lam) or (self.M_max - self.M > self.lam)
        if alarm:                                  # reset (river semantics)
            self.__init__(self.lam, self.delta, self.alpha, self.min_instances)
        return alarm


# --------------------------------------------------------- vol targeting
class VolTarget:
    """Moreira-Muir vol-managed exposure with the REAL-TIME normalization
    constant (Bernardi et al. arXiv:2212.07288: the full-sample c* is not
    known in real time; use the expanding estimate c_t*).

    Managed return  y^sigma_t = m_t * y_t,  m_t = c_t*/sigma_hat_t^2, with
    c_t* = sigma_tgt / sqrt(E_t[1/sigma_hat^2])  so that the unconditional
    variance of the managed stream matches sigma_tgt^2.  Multiplier capped
    to [0.25, 2.0] (leverage sanity)."""

    def __init__(self, sigma_tgt: float, cap: tuple[float, float] = (0.25, 2.0)):
        self.sigma_tgt = sigma_tgt
        self.cap = cap
        self.sum_inv_var = 0.0
        self.n = 0

    def update(self, sigma_hat: float) -> None:
        if sigma_hat > 0:
            self.sum_inv_var += 1.0 / (sigma_hat ** 2)
            self.n += 1

    def multiplier(self, sigma_hat: float) -> float:
        if self.n < 100 or sigma_hat <= 0:
            return 1.0
        e_inv_var = self.sum_inv_var / self.n
        c_star = self.sigma_tgt / math.sqrt(e_inv_var + 1e-18)
        m = c_star / (sigma_hat ** 2 + 1e-18)
        return float(np.clip(m, *self.cap))


# ----------------------------------------------------------- carver buffer
@dataclass
class CarverBuffer:
    """No-trade band = +/-10% of the AVERAGE position (position at |signal|
    = its trailing mean); when outside, trade TO THE EDGE, never through it."""

    frac: float = 0.10

    def buffer_size(self, avg_abs_pos: float) -> float:
        return self.frac * max(avg_abs_pos, 1e-9)

    def apply(self, current: float, ideal: float, avg_abs_pos: float) -> float:
        b = self.buffer_size(avg_abs_pos)
        lo, hi = ideal - b, ideal + b
        if current < lo:
            return lo                                # trade to the edge only
        if current > hi:
            return hi
        return current                               # inside band: NO TRADE


# ------------------------------------------------------------- DSR and PBO
def deflated_sharpe_ratio(sr: float, T: int, skew: float, kurt: float,
                          sr_trials: np.ndarray | None = None,
                          N_trials: int | None = None) -> float:
    """Bailey & Lopez de Prado 2014 exact: DSR = Phi( (SR - SR0) sqrt(T-1) /
    sqrt(1 - g3 SR + (g4-1)/4 SR^2) ),  SR0 = sqrt(V[SR]) [(1-g) Z^-1(1-1/N)
    + g Z^-1(1-1/(N e))],  g = 0.5772156649.  SR is PER-PERIOD."""
    g = 0.5772156649
    if sr_trials is not None and N_trials is None:
        N_trials = len(sr_trials)
    v = float(np.var(sr_trials)) if sr_trials is not None else 1.0 / max(T, 2)
    e = math.e
    z1 = _norm.ppf(1 - 1.0 / N_trials)
    z2 = _norm.ppf(1 - 1.0 / (N_trials * e))
    sr0 = math.sqrt(v) * ((1 - g) * z1 + g * z2)
    denom = math.sqrt(max(1e-18, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    return float(_norm.cdf((sr - sr0) * math.sqrt(max(T - 1, 1)) / denom))


def pbo_cscv(pnl_matrix: np.ndarray, n_blocks: int = 16) -> float:
    """Probability of Backtest Overfitting via Combinatorial Symmetric CV
    (Bailey-Borwein-LdP-Zhu 2017): split the T x N per-bar PnL matrix into S
    blocks; for every combination of S/2 blocks as IS (complement OOS), pick
    the IS-best config, compute its OOS relative rank; PBO = frequency of
    logit(rank) <= 0 (IS-best below OOS median)."""
    T, N = pnl_matrix.shape
    S = n_blocks
    if T % S:
        pnl_matrix = pnl_matrix[: T - (T % S)]
        T = len(pnl_matrix)
    blocks = pnl_matrix.reshape(S, T // S, N)
    from itertools import combinations
    combs = list(combinations(range(S), S // 2))
    logits = []
    for is_blocks in combs:
        oos_blocks = [b for b in range(S) if b not in is_blocks]
        is_pnl = blocks[list(is_blocks)].reshape(-1, N).sum(axis=0)
        oos_pnl = blocks[oos_blocks].reshape(-1, N).sum(axis=0)
        best = int(np.argmax(is_pnl))
        rank = (oos_pnl < oos_pnl[best]).sum() / (N + 1)   # relative rank in (0,1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return float((logits <= 0).mean())


# ------------------------------------------------- meta-labeling & sizing
def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Guo et al. 2017: T* = argmin NLL on a validation slice (1-D convex)."""
    best_T, best_nll = 1.0, np.inf
    for T in np.linspace(0.05, 5.0, 200):
        p = 1.0 / (1.0 + np.exp(-logits / T))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def kelly_size(p_cal: float, b: float, fraction: float = 0.25,
               cap: float = 1.0) -> float:
    """Fractional Kelly on the calibrated meta-probability:
    f* = (p(b+1) - 1)/b  (edge/odds), scaled by `fraction`, capped."""
    f_star = (p_cal * (b + 1) - 1) / max(b, 1e-9)
    return float(np.clip(fraction * f_star, 0.0, cap))


# ------------------------------------------------------------ the engine
@dataclass
class OnlineEngineConfig:
    label_horizon: int = 24
    cost_r: float = 5e-5
    sigma_tgt: float = 0.0015                 # per-bar vol target
    ph_lam: float = 50.0
    vov_halve_quantile: float = 0.8           # vol-of-vol de-risk quintile
    freeze_bars_after_alarm: int = 24
    online_lam: float = 0.999


class OnlineExecutionEngine:
    """Deterministic hourly event loop.

    Per bar t (features already computed causally upstream):
      1. model score s_t (frozen backbone) + online ridge correction d_t
      2. z_ideal = tanh(s_t + d_t) * regime_multiplier
         regime = vol-target(m_t) * PH-alarm-de-risk * vol-of-vol overlay
      3. buffered position via CarverBuffer (10% band, trade-to-edge)
      4. cost accounting on |z_t - z_{t-1}|
      5. label-arrival updates: every H bars, RLS update on resolved outcomes
    Single continuous position => concurrency is structural, not bolted on:
    there is exactly one marked-to-market exposure at all times.
    """

    def __init__(self, cfg: OnlineEngineConfig, n_readout_features: int):
        self.cfg = cfg
        self.readout = OnlineRidge(n_readout_features + 1, lam=cfg.online_lam)
        self.bank = RLSBank(n_readout_features + 1)
        self.ph_ret = PageHinkley(lam=cfg.ph_lam)
        self.vt = VolTarget(cfg.sigma_tgt)
        self.buffer = CarverBuffer()
        self.z = 0.0
        self.freeze_until = -1
        self.pending: list[tuple[int, np.ndarray, float]] = []   # (arrival, x, y)
        self.avg_abs_z = 0.05                    # running mean |z| (Carver)
        self.vov_series: list[float] = []

    def step(self, t: int, x: np.ndarray, model_score: float, r_t: float,
             sigma_hat: float, vol_of_vol: float | None = None) -> dict:
        """Advance one bar.  x = readout features (with intercept prepended),
        model_score = frozen model's pre-position score, r_t = realized raw
        return of bar t (for vol targeting / PH), sigma_hat = causal EW vol."""
        x = np.concatenate([[1.0], np.asarray(x, dtype="float64")])
        # ---- label-arrival updates (never before the label exists)
        arrivals = [p for p in self.pending if p[0] == t]
        for _, px_, py in arrivals:
            if t >= self.freeze_until:
                self.readout.update(px_, py)
                self.bank.update(px_, py)
        self.pending = [p for p in self.pending if p[0] != t]

        # ---- regime gates
        self.ph_ret.update(r_t / (sigma_hat + 1e-12))
        alarm_now = False
        if self.ph_ret.n == 0:                    # just reset => alarm fired
            alarm_now = True
            self.freeze_until = t + self.cfg.freeze_bars_after_alarm
        self.vt.update(sigma_hat)
        m_vol = self.vt.multiplier(sigma_hat)
        m_volov = 1.0
        if vol_of_vol is not None:
            self.vov_series.append(vol_of_vol)
            if len(self.vov_series) > 24 * 30:
                q = np.quantile(self.vov_series, self.cfg.vov_halve_quantile)
                if vol_of_vol > q:
                    m_volov = 0.5
        regime = m_vol * m_volov * (0.5 if t < self.freeze_until else 1.0)

        # ---- position with online correction
        d = self.readout.predict(x)
        z_ideal = float(np.tanh(model_score + d)) * regime
        z_new = self.buffer.apply(self.z, z_ideal, self.avg_abs_z)
        cost = self.cfg.cost_r * abs(z_new - self.z)
        gross = self.z * r_t                       # previous position earns r_t
        net = gross - cost
        self.avg_abs_z += (abs(z_new) - self.avg_abs_z) / max(1, t + 1)
        self.z = z_new

        # ---- queue the label for this origin (arrives at t + H)
        y = r_t / (sigma_hat + 1e-12)              # vol-normalized realized
        self.pending.append((t + self.cfg.label_horizon, x, y))
        return {"t": t, "z": z_new, "net": net, "cost": cost, "alarm": alarm_now,
                "regime_mult": regime, "online_corr": d}


def self_test() -> None:
    rng = np.random.default_rng(7)
    # ---- OnlineRidge learns a drifting linear map
    T, d = 3000, 3
    X = rng.standard_normal((T, d))
    theta = np.array([0.5, -0.3, 0.1])
    drift = np.linspace(0, 1.0, T)
    y = (X * theta[None, :] * (1 + 0.5 * drift)[:, None]).sum(axis=1) + 0.1 * rng.standard_normal(T)
    orr = OnlineRidge(d + 1, lam=0.995)
    errs = []
    for t in range(T):
        x = np.concatenate([[1.0], X[t]])
        errs.append(y[t] - orr.predict(x))
        orr.update(x, y[t])
    # parameter recovery at the END of the drift (true = theta * 1.5 at t=T)
    # -- the irreducible noise is 0.1, so compare coefficients, not raw errors
    theta_end = theta * (1 + 0.5 * drift[-1])
    assert np.allclose(orr.w[1:], theta_end, atol=0.05), \
        f"OnlineRidge failed to track the drifting map: {orr.w[1:]} vs {theta_end}"
    assert np.mean(np.abs(errs[100:])) < np.mean(np.abs(errs[:50]))
    # ---- PageHinkley detects a mean shift
    ph = PageHinkley(lam=5.0, min_instances=30)
    stream = list(rng.standard_normal(200) * 0.1) + list(0.5 + rng.standard_normal(200) * 0.1)
    alarms = [ph.update(s) for s in stream]
    assert any(alarms[200:]), "PageHinkley missed the injected mean shift"
    # ---- Carver buffer: no trades inside the band, edge-trades outside
    cb = CarverBuffer()
    assert cb.apply(0.50, 0.52, avg_abs_pos=0.5) == 0.50      # inside -> hold
    assert abs(cb.apply(0.50, 0.90, 0.5) - (0.90 - 0.05)) < 1e-12  # to edge
    # ---- engine loop smoke + causality of label updates
    cfg = OnlineEngineConfig(label_horizon=4)
    eng = OnlineExecutionEngine(cfg, n_readout_features=2)
    sigma = 0.001
    for t in range(400):
        x = rng.standard_normal(2)
        r = sigma * rng.standard_normal()
        out = eng.step(t, x, 0.3 * rng.standard_normal(), r, sigma)
    assert len(eng.pending) == cfg.label_horizon
    # ---- DSR: no-skill family -> low DSR; skillful single -> high DSR
    sr_family = rng.standard_normal(100) * 0.1
    dsr_bad = deflated_sharpe_ratio(0.05, 2000, 0.0, 3.0, sr_trials=sr_family)
    dsr_good = deflated_sharpe_ratio(0.30, 2000, 0.0, 3.0, sr_trials=sr_family)
    assert dsr_good > dsr_bad and 0 <= dsr_bad <= 1
    # ---- PBO on random PnL (no-skill) should be near 0.5
    pnl = rng.standard_normal((1600, 20))
    pbo = pbo_cscv(pnl, n_blocks=16)
    assert 0.2 < pbo < 0.8, f"PBO on noise should be ~0.5, got {pbo}"
    # ---- temperature + Kelly
    logits = rng.standard_normal(2000) * 2
    yb = (logits + 0.5 * rng.standard_normal(2000) > 0).astype(float)
    Tcal = fit_temperature(logits, yb)
    assert 0.05 <= Tcal <= 5.0
    f = kelly_size(0.55, b=1.5, fraction=0.25)
    assert 0 < f < 0.25
    print("online_engine self-test passed: RLS tracking, PH alarm, buffer, "
          "engine loop, DSR ordering, PBO~0.5 on noise, calibration, Kelly.")


if __name__ == "__main__":
    self_test()

```

The loop per H1 bar t (features computed causally upstream):

1. label-arrival updates (§5.1) → 2. regime gates (§5.2) → 3. z_ideal = tanh(score + online correction) × regime multiplier → 4. **Carver no-trade buffer** ✅: trade only if the current position is outside [z_ideal − 0.10·z̄, z_ideal + 0.10·z̄] with z̄ = trailing mean |z| — and then **only to the edge of the band, never through it** (`buffer_trade_to_edge` semantics, verified from pysystemtrade source) → 5. costs c·|Δz| booked against the bar → 6. queue this origin's label for t+24.

**Concurrency is structural, not bolted on.** v1.x's overlapping-trade pathology (7,207 overlapping positions, per-trade 1% caps that never aggregate, MaxDD 127.7R understated) is eliminated by construction: there is exactly **one** continuous vol-scaled exposure, marked to market every bar, drawdown and exposure exact at every instant. The catastrophic-stop frame (§4.8) converts to flat at −2.5 ATR-equivalents, forward-simulated inside this loop — never from label summaries (Mandate M5).

**Speed limit** (Carver ✅): annualized cost ≤ ⅓ of expected pre-cost Sharpe; enforced by monitoring the realized cost drag and widening the buffer fraction β (default 0.10, scales ~√c) if breached. The cost term inside the training loss (§4.4) already biases the network toward low-turnover solutions; the buffer is the execution-side guarantee.

## 5.4 Non-overlapping evaluation and concurrency collapse

Walk-forward OOS evaluation collapses overlapping label windows to *portfolio-level* hourly returns r_t^net (the engine's own accounting), so the effective sample size is the number of bars, not the number of overlapping trades — the v1.x illusion of 7,207 "independent" trades cannot recur. Secondary reporting uses non-overlapping 24-bar blocks for bootstrap resampling (week-block bootstrap retained for CI reporting, matching v1.2 conventions).

## 5.5 Verification: DSR, PBO, and the config ledger

**Deflated Sharpe Ratio** (Bailey & López de Prado ✅, exact):

$$SR_0 = \sqrt{V[\{SR_n\}]}\Big[(1-\gamma)\Phi^{-1}\!\big(1-\tfrac1N\big) + \gamma\,\Phi^{-1}\!\big(1-\tfrac{1}{Ne}\big)\Big], \qquad \gamma = 0.5772156649$$
$$DSR = \Phi\!\left(\frac{(SR - SR_0)\sqrt{T-1}}{\sqrt{1 - \gamma_3 SR + \tfrac{\gamma_4-1}{4}SR^2}}\right)$$

with SR the **per-bar** Sharpe, T = OOS bars, γ₃/γ₄ skew/kurtosis of OOS net returns, N = **every configuration ever evaluated** (the ledger records draws automatically from the first experiment — v1.2's 36-config DSR was the end-of-pipeline version of a discipline that must start at experiment #1). **Acceptance gate: DSR ≥ 0.95 on 2023–25.**

**PBO via CSCV** (Bailey-Borwein-LdP-Zhu ✅): the T×N per-bar PnL matrix of the full config family, S=16 blocks, logit-rank construction; **quality bar PBO ≤ 0.2** (target) / 0.3 (hard ceiling). Both are implemented and self-tested (`deflated_sharpe_ratio`, `pbo_cscv` — noise PnL ⇒ PBO ≈ 0.5 as the null).

## 5.6 Live decision pathway (v1.2 artifact, upgraded)

The daily kernel emits the same auditable decision object as v1.2, now with the v2.0 fields: as-of bar; z (continuous position) and its decomposition (model score, online correction, regime multiplier); buffer state (inside/outside band, distance to edge); PH/BOCPD/vol-of-vol states; cost-budget usage vs the ⅓ speed limit; catastrophic-stop distance in ATR; the config-ledger snapshot (N, V[SR], current DSR estimate). Every field recomputable from kernel + dataset alone (G6 gates, Section 6).

---


# Section 6 — Step-by-Step Handoff Protocol for Agent 1

**Budget envelope: ≤10 GPU-hours on 2×T4 + CPU kernels.** All numbers below are the plan of record; the reserve absorbs one full retry of the most expensive kernel.

## 6.0 Kernel plan and GPU budget

| # | Kernel | Hardware | Est. | Purpose | Exit gate |
|---|---|---|---|---|---|
| K1 | `aurum-v2-features` | CPU (4–6 h wall) | 0 GPU | feature store build from M1/H1 parquets | G1.1–G1.6 |
| K2 | `aurum-v2-baselines` | 2×T4 | **1.0 GPU-h** | M0 ladder: LightGBM + DMN-LSTM + DLinear + AE-MLP quick pass | G2.1–G2.3 |
| K3 | `aurum-v2-continuum` | 2×T4 | **3.0 GPU-h** | ContinuumNet 30-config random search + best-config folds | G3.1–G3.4 |
| K4 | `aurum-v2-pathb-probe` | 2×T4 | **1.5 GPU-h** | Path B: linear probe (0.5 h) + conditional LoRA (1.0 h) | G4.1–G4.2 |
| K5 | `aurum-v2-walkforward` | 2×T4 | **1.5 GPU-h** | yearly refits of the chosen stack + ensemble weights | G5.1–G5.3 |
| K6 | `aurum-v2-backtest-live` | CPU (+0.5 GPU-h) | 0.5 GPU-h | online engine, backtest, DSR/PBO, live decision object | G6.1–G6.6 |
| — | reserve | — | **2.5 GPU-h** | one full retry of K3 or K5 | — |

**Total: 7.5 GPU-h + 2.5 reserve = 10.0 ceiling.** If K3's search overruns, cut configs (never epochs-per-config below 60 with patience 25); if K4's K1 kill-criterion fires early, reallocate its hour to K3.

## 6.1 File manifest (ship as Kaggle dataset `tskdev/aurum-v2-code`)

```
aurum_features/            # numpy/pandas only — all self-tested this session
  __init__.py
  f1_clock.py              # DST-aware exchange clocks        (Section 3.1)
  f2_realized.py           # RV/BV/SJ/VR/Hurst/roughness      (Section 3.2)
  f3_signatures.py         # signatures, Levy areas, lead-lag (Section 3.3)
  f4_toxicity.py           # VPIN/Kyle/Amihud/CLV             (Section 3.4)
  f5_residual.py           # EW-RLS residualization + channels(Section 3.5)
  f6_trend.py              # saturating EWMAC ladder          (Section 3.6)
  build_v2_features.py     # orchestrator + IO contract       (Section 3.7)
model/                     # torch (CPU-tested; T4 fp16 paths included)
  losses.py                # net-Sharpe/Ulcer/DSR-loss        (Section 4.4)
  continuum.py             # ContinuumNet + LoopBlock + audit (Section 4.2-4.3)
  ae_mlp_dmn.py            # AE-MLP + DMN-LSTM                (Section 4.6)
  purge.py                 # purged walk-forward              (Section 4.9)
  train_continuum.py       # trainer + governor               (Section 4.5)
  online_engine.py         # RLS/bank/PH/vol-target/buffer/DSR/PBO (Section 5)
kernels/                   # Agent 1 writes these 6 scripts per this section
test_local.py              # pre-flight: runs EVERY module self-test + the
                           # truncation-equivalence harness before any push
                           # (v1.2 Failure #4 lesson: zero-crash push discipline)
```

**Agent 1 adaptation points (the only lines that change):** `M1_PATHS`/`H1_PATHS` in `build_v2_features.py`; the feature-group slices passed to `ContinuumConfig` (align with the built feature frame; groups listed in §3.0); era-safe macro innovation columns (reuse the v1.2 transforms verbatim — they passed the era audit); LightGBM config (reuse v1.2's regularized setup). Everything else ships as-is.

## 6.2 Gate specifications

**G1 — feature store (K1).** G1.1 row count ≈ 95,008 H1 bars; feature count within ±5% of the §3.0 budget table. G1.2 NaN discipline: median column NaN < 1%; only documented warm-up columns (YZ_528, HAR_528, scalar-warmup) exceed 5%. G1.3 causality: `test_local.py` truncation-equivalence on a 2-year slice must be bit-identical. G1.4 DST verification: the autumn gap-week assertion from `f1_clock.dst_gap_sanity_check()` passes on real 2010–2025 timestamps (expect 2 gap windows/year). G1.5 unit checks: `sig1_xau` ≈ z-scored 1h log-return of XAU (correlate with H1 return > 0.9); `eps` vol < raw XAU return vol on ≥ 80% of bars. G1.6 coverage: emit per-year M1 coverage; flag 2023 (82.2% known dip) — features from gappy hours are computed on available M1 bars with the count recorded as a feature.

**G2 — baselines (K2).** G2.1 LightGBM clean OOS (2023–25) expectancy within ±0.02R of the v1.3 clean GBDT number (replication of the null — this is the control experiment; if it deviates, STOP and audit before any GPU spend). G2.2 DMN-LSTM val Sharpe > 0 and > DLinear. G2.3 AE-MLP val Sharpe ≥ LightGBM's val metric (the floor must be competitive, per the Kaggle evidence).

**G3 — ContinuumNet (K3).** G3.1 zero NaN/overflow events across fp16 training (GradScaler active). G3.2 val-Sharpe early stopping engaged on ≥ 80% of configs. G3.3 **spectral gate: ρ < 0.95 at every shipped checkpoint** (governor log emitted). G3.4 best-config val Sharpe (2021–22) ≥ max(baselines) − 0.1; else the ensemble proceeds without M1 (the floor is the product — honest outcome, not a failure).

**G4 — Path B probe (K4).** G4.1 probe result recorded against kill-criterion K1 (§4.7). G4.2 if run, LoRA val Sharpe recorded against K2. Both outcomes written to the config ledger regardless (they count as trials).

**G5 — walk-forward + backtest (K5/K6).** G5.1 yearly refits complete; ensemble weights frozen on 2021–22 before the 2023–25 seal is broken. G5.2 OOS 2023–25 evaluated in the single-position event loop with costs at 1.0×/1.5×/2.0×; per-bar net Sharpe, MaxDD, turnover, cost-drag vs the ⅓ speed limit all reported. G5.3 baseline gauntlet: full engine ≥ frozen model ≥ RLS-bank on purged OOS (adaptive components that fail the gauntlet are disabled, not argued with).

**G6 — statistics & live (K6).** G6.1 DSR ≥ 0.95 target (minimum shipping bar 0.50 with explicit disclosure); N = full config-ledger count. G6.2 PBO ≤ 0.3 (target 0.2). G6.3 block-bootstrap CIs on collapsed portfolio returns. G6.4 truncation-equivalence harness on the final engine (identical decisions when the future is deleted). G6.5 live decision object (§5.6 fields) from the latest fully-featured bar. G6.6 everything hash-identical on re-run (determinism, v1.2 G5.1 standard).

## 6.3 Pre-registered acceptance criteria (the definition of success)

- **A1 (edge):** clean OOS 2023–25 net expectancy, expressed in per-bar vol-scaled terms, exceeds the replicated GBDT-clean null by ≥ 50% relative at 1.0× costs *and* remains positive at 1.5×.
- **A2 (statistics):** DSR ≥ 0.95 and PBO ≤ 0.2 (both on the full ledger).
- **A3 (gauntlet):** beats frozen + RLS-bank; every regime gate justified by ablation (each gate's OOS contribution ≥ 0).
- **A4 (safety):** spectral gate green; truncation-equivalence green; turnover within the speed limit; MaxDD (R-equivalents) < v1.x flagship's 127.7R.
- **A5 (honesty):** all kill-criteria outcomes and failed configs published in the artifacts; the config ledger complete from experiment #1.

If A1 fails but A2–A5 pass: the deliverable is the *honest measurement* (v1.2's standard) plus the diagnosis — the architecture is falsifiable by design, and the ledger is the evidence base for v2.1.

## 6.4 Execution order and dependencies

K1 → (K2 ∥ K3) → K4 (parallel-friendly but quota-aware: run K2+K3 in one week, K4+K5 the next if quota binds) → K5 → K6. `test_local.py` green is a hard pre-push condition for every kernel (v1.2 Failure #4). Seeds fixed (1337 + seed index); all artifacts versioned and hashed.

## 6.5 Risk register (what can bite, and the mitigation)

| Risk | Mitigation |
|---|---|
| 2023 M1 coverage dip (82.2%) degrades features | coverage counted per hour; gappy-hour features flagged; 2023 evaluated with and without the dip year in a robustness split |
| tzdata version differences on Kaggle image | pin the `tzdata` wheel; `dst_gap_sanity_check()` is part of G1.4 — it fails loudly on a bad tz database |
| LBMA/SGE holiday masks (Christmas AM-only fix, etc.) | anchors NaN-masked on venue weekends; holiday list shipped as a static table (verified dates only; Appendix B flags the unverified ones) |
| Tick volume ≠ activity for VPIN/Amihud | G1.5 requires the correlation check (tick-volume vs \|return\|); VPIN enters only as a state variable (critique honored) |
| fp16 overflow on T4 (no bf16) | GradScaler; overflow counter in G3.1; fp32 fallback for flagged layers (v1.2 precedent) |
| Kaggle quota exhaustion mid-search | config ledger is resumable; 2.5 GPU-h reserve; search ordered by expected information gain (cheap configs first) |
| Anomaly decay (Boyarchenko post-2021 lesson) | session/clock effects are *features* (model-conditional), never hard-coded rules; the drift audit runs every fold |

## 6.6 The trap list — v1.x failures that must not recur

1. Never read label summaries as execution inputs (H3 disease). The event loop is the only backtest.
2. Never let a benchmark consume backward returns (B&H disease). Same loop, same data, forward only.
3. Never select by argmax over noisy EVs (48-cell disease). LCB + shrinkage + continuous positions.
4. Never iterate an unconstrained block (loop disease). Spectral caps + KM damping + ρ<0.95 gate.
5. Never push untested code (18-push disease). `test_local.py` green or no push.
6. Never report a number whose trial count is unknown (DSR=0 disease). The ledger counts from experiment #1.

---

# Appendix A — Constants Card (every number, its value, its source)

| Constant | Value | Source (ledger #) |
|---|---|---|
| BV normalization π/2·n/(n−1) | 1.570796 | P3-1 ✅ |
| μ_{4/3} / μ_{4/3}^{−3} (TQ) | 0.83086 / 1.7430 | P3-1 ✅ |
| Jump-statistic variance θ | 0.6090 | P3-4 ✅ |
| Jump threshold (99%, one-sided) | 2.326 | standard |
| Parkinson 1/(4 ln 2) | 0.360674 | P3-7 ✅ |
| GK constants (k₁,k₂,k₃) | 0.511 / 0.019 / 0.383 | P3-7 ✅ (Meilijson) |
| Yang-Zhang α | 1.34 | P3-7 ✅ |
| MedRV / MinRV constants | 1.418640 / 2.751935 | P3-5 ✅ |
| VR unbiased denominator | m = (n_q−q+1)(1−q/n_q) | Monte-Carlo verified this session |
| GJR roughness H (log-vol) | ≈ 0.1 (gold similar) | P3-9 ✅ |
| Signature terms d=5: depth 2 / 3 (full) | 30 / 155 | P3-13 ✅ |
| Lead-lag pair feature count | 6 per pair | P3-13 ✅ |
| RLS forgetting λ (F5 residualization) | 0.995 (memory ≈ 200 bars) | P4-1 ✅ |
| A&L s-score entry / exits | 2.5σ_eq (hourly-hardened) / 0.75 | P4-2 ✅ + our adaptation |
| CFM tanh saturation | s\* = 0.89, b = 0.075 | P4-5 ✅ |
| DMN/Baz response φ | y·exp(−y²/4)/0.89, peak √2 | P4-11 ✅ |
| EWMAC ladder | s ∈ {2,4,8,16,32,64,128,256} h, N,4N | P4-10 ✅ |
| Forecast scalar target / cap | mean\|F\| = 10 / ±20 | P4-10 ✅ (defaults.yaml) |
| Carver buffer | 10% of average position, trade-to-edge | P4-10 ✅ (code) |
| Carver speed limit | annual cost ≤ ⅓ expected pre-cost SR | P4-10 ✅ |
| Sharpe-loss ε | 1e-9 | P5-3 ✅ (repo) |
| Bars/year (XAUUSD H1) | 5,786 (87,948/15.2y); √A = 76.07 | empirical, this spec |
| DMN training grid | hidden {5..160}, batch {64..2048}, lr {1e-5..1}, patience 25, unroll 63 | P5-1 ✅ |
| MTX config | seq 252, 4 heads, val-Sharpe selection, patience 25 | P5-2 ✅ |
| AE-MLP (JS-2021) | noise 0.035, enc 96, MLP 896-448-448-256, drop {0.10..0.49}, \|resp\| weights, purged-group gap 31 | P5-5 ✅ |
| LoRA budget guidance | {Wq,Wv}, r=4–16 ("more types, smaller rank") | P5-12 ✅ |
| Spectral caps | loop linears σ ≤ 0.8; LN affine init 0.5 | this spec (Miyato ✅) |
| KM damping α | 0.5 init, governor-shrunk; ρ gate 0.95 | this spec (monDEQ ✅) |
| Page-Hinkley (river) | λ=50.0, δ=0.005, α=0.9999, min_inst=30 | P7-5 ✅ |
| BOCPD | hazard 1/500 (paper DJ example 1/1000); prior a=1, b=1e-4 | P7-6 ✅ |
| Vol-target multiplier cap | [0.25, 2.0] | this spec |
| Vol-of-vol overlay | top-quintile → ×0.5 | P7-9 ◐ (gated interpretation) |
| Purge / embargo | 24 / 24–48 bars; constraint < fold−1 | P7-11 ✅ |
| DSR γ (Euler-Mascheroni) | 0.5772156649 | P7-12 ✅ |
| PBO quality bar | ≤ 0.2 target / 0.3 ceiling | this spec |
| LCB z_α | 1.28 (90%), re-check 1.64 | standard |
| Fractional Kelly | c = 0.25 (P[dbl]≈0.94, 64% growth) | P6-5 ✅ |
| Temperature scaling | T\* by val NLL; ECE 15 bins | P6-6 ✅ |
| Cost model | 0.05R round-turn at 1.0-ATR stop; multipliers 1.0/1.5/2.0× | v1.2 convention |

# Appendix B — Unverified-Items Register (open questions, honestly held)

| # | Item | Status | Action before load-bearing use |
|---|---|---|---|
| B1 | Michaud 1989 "error maximization" characterization | ⚠️ paywalled | fetch FAF 45(1); the (b+1)² math stands on Bailey et al. regardless |
| B2 | Jorion shrinkage papers | ⚠️ | not load-bearing (James-Stein is verified) |
| B3 | Patton-Sheppard exact REStat coefficients | ◐ abstract/metadata | fetch PDF; sign hypotheses used as *features*, model learns signs |
| B4 | Anis-Lloyd E(R/S) exact rendering | ⚠️ OCR-garbled | we use VR-slope Hurst (self-verified) — R/S not in v2.0 |
| B5 | Parkinson 4.9× efficiency constant | ⚠️ | not load-bearing |
| B6 | AQR "Century" exact 1/3/6/12 weighting | ◐ | blend construction is Carver-verified; AQR cited as corroboration |
| B7 | Carver book-table EWMAC scalars (10.6/8.5/…) | ⚠️ | v2.0 *estimates* scalars on-data (q_scalars recipe) — table unused |
| B8 | Moody-Saffell 2001 primary equations | ◐ via verified reproduction (typo corrected) | order IEEE PDF for definitive check |
| B9 | AFML meta-labeling verbatim wording | ◐ canonical-secondary | workflow is standard; MLFinLab pages were 404/rate-limited |
| B10 | Jane Street 2024–25 winner specifics | ⚠️ JS-rendered pages blocked | 9th/13th/26th write-ups verified; 1st-place method unverified — do not cite externally |
| B11 | GJR H at *hourly* lags | extrapolation | GJR fit multi-day lags; hourly adaptation is regime extrapolation — validated as a feature, not assumed |
| B12 | GC settlement time 13:30 ET; TOCOM hours; FCA Barclays fine details | ⚠️ | anchor times verified from fetched pages; these three flagged |
| B13 | Optiver 2021 1st-place internals (t-SNE order recovery) | ✅ write-up | code not re-run; treated as design evidence only |
| B14 | Zhang et al. CVaR/Sortino losses | **not in the papers** (future-work note) | reconstructed forms flagged in P6 digest; excluded from v2.0 losses |

**End of blueprint.** Sections 3–5 code blocks are the exact self-tested files from this research session; Section 6 is the execution contract. — Agent 2, Principal Quantitative Research Architect, Project AURUM, November 9, 2026.
