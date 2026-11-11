# AURUM Execution Journal (recreated 2026-11-09 after sandbox reset)

## Session 2026-11-09 (UTC) — Spec v1.3 "Decision-Layer & Realism Upgrade"

**Recovery:** Sandbox reset again (kaggle_work/aurum + token + v1.2 report lost).
Restored Kaggle API token (redacted); pulled all 8 kernel sources + outputs from Kaggle
(they are the source of truth). GPU quota 12,518s/21,600s — untouched (reset 11-14).

**Found mid-flight:** the interrupted v1.3 session had already pushed and run
- aurum-p3-tables v3 (CPU finalize, 89s): G4.0-G4.7 ALL PASS (G4.6 100.00% via
  imputation of 15,684 warmup rows at 0.5)
- aurum-p4-backtest v1.3 (CPU): G5.0-G5.9 ALL PASS (12 gates) — flagship OOS
  E[R]=+0.1468R/1354 trades cap1, Sharpe 2.305, MaxDD 14.67%; cap3 Sharpe 2.444,
  MaxDD 13.06%; GBDT parity +0.0310 vs 0.0599; k-NN -0.0155 vs 0.0078;
  DSR daily 0.844/0.895; policy fit 2016-22 only (G5.7).

**This session completed:**
1. live-daily v1.2 -> v1.3 rewrite (src_push/live/aurum-live-daily.py, 2721 lines):
   4-depth CPU forwards, pooled p̄ (0.6/0.2/0.2), sigma_loop, LCB-EV (κ=0,
   λ=0.5, EV_min=0, cost 0.05/m), GBDT fold agreement gate, path overlay
   (H3 q50/q25; h24 empirical ≤y-1), 1% quarter-Kelly, G6.4b consistency gate.
   - Logic tests T1-T7 PASS on shipped text (2 real bugs found & fixed:
     unmasked base-rate prior; ood_re cell-vs-state mismatch)
   - Pushed v2, ran CPU ~7min: G6.0-G6.5 + G6.4b ALL PASS (7/7);
     d=0.00e+00 on all 4 consistency recomputes; decision: BUY L_m1.0_t4.0_h24
     as of 2025-12-29, EV_LCB +0.4835R, OOD 0.0054, GBDT agrees (0.512/0.386),
     risk 1.0% cap -> $100 / 4.67 oz on $10k.
2. 8 report figures (matplotlib, 200 DPI) from kernel artifacts.
3. Master Report v1.3 compiled (docx skill, R1 cover + IG-1, 3-section Roman/Arabic):
   12 H1 + 24 H2, 8 figures, 11 tables, 5,770 words, 21 PDF pages.
   postcheck 9/9 PASS, 0 errors, 0 warnings. Deliverables:
   download/AURUM_Master_Execution_and_Backtest_Report.{docx,pdf}
   + download/AURUM_v13_report/{figures,artifacts}

**Kernel state:** aurum-p3-tables v3 PASS / aurum-p4-backtest v1.3 PASS /
aurum-live-daily v2 PASS — Steps 4-6 100% green on CPU, 0 GPU-s consumed.
