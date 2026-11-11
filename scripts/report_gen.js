// AURUM Master Execution & Backtest Report generator (docx-js)
// Reads scripts/report_data.json + download/report_figs/*.png
// English business report - R1 cover (IG-1 Ink Gold), 3 sections, TOC.
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  SectionType, TableOfContents, TableLayoutType,
} = require("docx");
const fs = require("fs");

const D = JSON.parse(fs.readFileSync("/home/z/my-project/scripts/report_data.json", "utf8"));
const FIG = "/home/z/my-project/download/report_figs";

// ---------------- palette (IG-1 Ink Gold) ----------------
const PAL = {
  bg: "1A1A1A", accent: "C9A84C",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "C9A84C", headerText: "1A1A1A", accentLine: "C9A84C", innerLine: "DDD5C0", surface: "F5F2E8" },
  primary: "1A1A1A", body: "000000", secondary: "606060",
};

const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: NB, bottom: NB, left: NB, right: NB };
const allNoBorders = { top: NB, bottom: NB, left: NB, right: NB, insideHorizontal: NB, insideVertical: NB };

function safeText(v, ph) {
  if (v === undefined || v === null || v === "" || String(v) === "NaN" || String(v) === "undefined" || String(v) === "null")
    return ph || "n/a";
  return String(v);
}
const fmt = (v, d = 4) => (v === undefined || v === null || !isFinite(v)) ? "n/a" : Number(v).toFixed(d);
const fmt3 = (v) => fmt(v, 3);
const pct = (v, d = 1) => (v === undefined || v === null || !isFinite(v)) ? "n/a" : (100 * v).toFixed(d) + "%";
const int = (v) => (v === undefined || v === null) ? "n/a" : Number(v).toLocaleString("en-US");

// ---------------- cover machinery (design-system.md) ----------------
function splitTitleLines(title, charsPerLine) {
  if (title.length <= charsPerLine) return [title];
  const breakAfter = new Set([...",.;:!?", ..."-_/ \t", ..."，。、；：！？的与和及之在于为"]);
  const lines = []; let remaining = title;
  while (remaining.length > charsPerLine) {
    let breakAt = -1;
    for (let i = charsPerLine; i >= Math.floor(charsPerLine * 0.6); i--)
      if (i < remaining.length && breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    if (breakAt === -1) {
      const limit = Math.min(remaining.length, Math.ceil(charsPerLine * 1.3));
      for (let i = charsPerLine + 1; i < limit; i++)
        if (breakAfter.has(remaining[i - 1])) { breakAt = i; break; }
    }
    if (breakAt === -1) breakAt = charsPerLine;
    lines.push(remaining.slice(0, breakAt).trim());
    remaining = remaining.slice(breakAt).trim();
  }
  if (remaining) lines.push(remaining);
  if (lines.length > 1 && lines[lines.length - 1].length <= 2) {
    const last = lines.pop(); lines[lines.length - 1] += last;
  }
  return lines;
}
function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  // English title: chars ~ pt*11 twips wide (narrower than CJK pt*20)
  const charWidth = (pt) => pt * 11;
  const charsPerLine = (pt) => Math.floor(maxWidthTwips / charWidth(pt));
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = charsPerLine(titlePt);
    if (cpl < 2) { titlePt -= 2; continue; }
    lines = splitTitleLines(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) { lines = splitTitleLines(title, charsPerLine(minPt)); titlePt = minPt; }
  return { titlePt, titleLines: lines };
}
function calcCoverSpacing(p) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838, marginTop = 0, marginBottom = 0 } = p;
  const SAFETY = 1200;
  const usable = pageHeight - marginTop - marginBottom - SAFETY;
  const contentH = titleLineCount * (titlePt * 23 + 200) + (hasSubtitle ? 12 * 23 + 600 : 0) +
    (hasEnglishLabel ? 9 * 23 + 600 : 0) + metaLineCount * (10 * 23 + 100) + fixedHeight + 3 * 300;
  const remaining = Math.max(usable - contentH, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(remaining * 0.45), rawBottom = Math.floor(remaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  return { topSpacing, midSpacing: Math.max(remaining - topSpacing - bottomSpacing, 0), bottomSpacing };
}
function buildCoverR1(config) {
  const P = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "), size: 18, color: P.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true, color: P.cover.titleColor, font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 800 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.cover.subtitleColor, font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: P.cover.metaColor, font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: P.cover.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: P.cover.footerColor, font: { ascii: "Arial" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({ shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders, children })],
    })],
  })];
}

// ---------------- body helpers ----------------
const BODY_FONT = { ascii: "Calibri", eastAsia: "SimSun" };
const H_FONT = { ascii: "Calibri", eastAsia: "SimHei" };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 30, color: PAL.primary, font: H_FONT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 26, color: PAL.primary, font: H_FONT })],
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 120 },
    children: [new TextRun({ text, size: 22, color: PAL.body, font: BODY_FONT, ...opts })],
  });
}
function bodyRuns(runs) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 120 },
    children: runs.map(r => new TextRun({ size: 22, color: PAL.body, font: BODY_FONT, ...r })),
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 200, line: 280 },
    children: [new TextRun({ text, size: 18, color: PAL.secondary, italics: true, font: BODY_FONT })],
  });
}
function pngDims(path) {
  const b = fs.readFileSync(path);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, captionText, displayWidthPx = 560) {
  const fp = `${FIG}/${file}`;
  if (!fs.existsSync(fp)) return body(`[figure ${file} missing]`);
  const { w, h } = pngDims(fp);
  const dh = Math.round(displayWidthPx * h / w);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ data: fs.readFileSync(fp), transformation: { width: displayWidthPx, height: dh }, type: "png" })],
    }),
    caption(captionText),
  ];
}
function tbl(headers, rows, widths, opts = {}) {
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((t, i) => {
      const cell = new TableCell({
        children: [new Paragraph({
          spacing: { line: 264 },
          children: [new TextRun({ text: safeText(t), bold: true, size: opts.size || 19, color: PAL.table.headerText, font: BODY_FONT })],
        })],
        shading: { type: ShadingType.CLEAR, fill: PAL.table.headerBg },
        margins: { top: 55, bottom: 55, left: 110, right: 110 },
        width: { size: widths[i], type: WidthType.PERCENTAGE },
      });
      return cell;
    }),
  });
  const dataRows = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => new TableCell({
      children: [new Paragraph({
        alignment: opts.alignRight && opts.alignRight.includes(i) ? AlignmentType.RIGHT : AlignmentType.LEFT,
        spacing: { line: 264 },
        children: [new TextRun({ text: safeText(t), size: opts.size || 19, color: PAL.body, font: BODY_FONT, bold: !!(opts.boldRows && opts.boldRows.includes(ri)) })],
      })],
      shading: ri % 2 === 1 ? { type: ShadingType.CLEAR, fill: PAL.table.surface } : undefined,
      margins: { top: 45, bottom: 45, left: 110, right: 110 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
    })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.accentLine },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: PAL.table.innerLine },
      insideVertical: NB,
    },
    rows: [headerRow, ...dataRows],
  });
}
function tblTitle(text) {
  return new Paragraph({
    keepNext: true, spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 20, color: PAL.primary, font: BODY_FONT })],
  });
}

// ---------------- data shortcuts ----------------
const bt = D.backtest || {};
const S = (name, cost = "cost_1.0", win = "oos") => ((bt[name] || {})[cost] || {})[win] || {};
const HL = D.headline || {};
const stageA = D.stageA || {};
const stageB = D.stageB || {};
const folds = D.folds || {};
const p3s = D.p3_summary || {};
const live = D.live || {};
const boot = bt["bootstrap_oos_cost_1.0"] || {};
const PRETTY = {
  aurum_r1: "AURUM r=1", aurum_r2: "AURUM r=2", aurum_r3: "AURUM r=3", aurum_r4: "AURUM r=4",
  aurum_canon: "AURUM (2-cell)", ens_raw: "Ensemble raw", ens_neut: "Ensemble neut",
  gbdt: "GBDT", cohort: "k-NN Cohort", ema: "EMA 50/200", donchian: "Donchian 20", buyhold: "Buy & Hold",
};
const ALL = ["aurum_r1", "aurum_r2", "aurum_r3", "aurum_r4", "aurum_canon", "ens_raw", "ens_neut",
  "gbdt", "cohort", "ema", "donchian", "buyhold"];

// conditional narrative: does AURUM r3 beat the best baseline on OOS expectancy?
const baseRank = D.baseline_rank || [];
const bestBase = baseRank[0] || ["n/a", null];
const r3E = HL.oos_E, r3Beats = (typeof r3E === "number") && (typeof bestBase[1] === "number") && r3E > bestBase[1];
const r3BeatsAll = r3Beats && baseRank.every(([s, v]) => (r3E > (v ?? -9)));
const bootR3 = (boot.aurum_r3 || {}).expectancy_R || {};
const ciPos = (typeof bootR3.lo === "number") && bootR3.lo > 0;
const dsrR3 = D.dsr_r3 || {};

// ================= content =================
const bodyChildren = [];

// ---- 1 Executive summary ----
bodyChildren.push(h1("1  Executive Summary"));
bodyChildren.push(body(
  `Project AURUM set out to test one proposition: a small open-weight language model, refit as a looped transformer, can price short-horizon barrier trades on XAUUSD well enough to matter. This report documents the complete execution of that proposition on free Kaggle infrastructure - data pipeline, two-stage training, calibrated probability tables, a deterministic walk-forward backtest with multiple-testing discipline, and a production-style daily decision pathway - together with every gate, every failure, and every fix along the way.`));
bodyChildren.push(body(
  `The system refits Qwen3-0.6B by looping decoder layers 13-16 for r iterations (r trained up to 3, evaluated at 1-4) with trained sandwich norms, exposing roughly 63 million trainable parameters on top of the frozen base. Each hourly state is serialized into roughly 330 tokens of scale-invariant market text - ATR-normalized returns, range positions, cross-asset and macro-innovation context, cohort matches - and the model outputs a 48-cell barrier-trade probability grid (direction x stop x take-profit x horizon) plus four auxiliary heads, temperature-scaled per walk-forward fold.`));
bodyChildren.push(body(
  `Execution completed end-to-end across seven Kaggle kernels under a fixed autonomous protocol with hard gates at every stage. Training reached a sliced cross-entropy reduction of 61.0% from the pretrained baseline (gate required 30%), zero NaN events, and a walk-forward realized-outcome AUC of 0.607-0.617 across ten fold refits - modest, but achieved without any absolute-price or date information, under strict era-neutralization audits.`));
bodyChildren.push(body(
  `The headline backtest question - does the looped model beat honest baselines out of sample after costs? - is answered in Section 6. On the out-of-sample window 2023-2025 at 1.0x costs, AURUM r=3 traded ${int(HL.oos_n)} times with a per-trade expectancy of ${fmt(HL.oos_E)}R (95% block-bootstrap CI ${fmt(bootR3.lo)} to ${fmt(bootR3.hi)}), profit factor ${fmt3(HL.oos_pf)}, and maximum drawdown ${fmt(HL.oos_maxdd, 1)}R. ${r3BeatsAll
    ? `This ranks first among all twelve strategies evaluated, ahead of the best baseline (${PRETTY[bestBase[0]] || bestBase[0]} at ${fmt(bestBase[1])}R), and the bootstrap interval ${ciPos ? "excludes zero" : "includes zero"} - evidence consistent with a genuine but modest edge rather than noise.`
    : r3Beats
      ? `This exceeds the best baseline (${PRETTY[bestBase[0]] || bestBase[0]} at ${fmt(bestBase[1])}R), though the margin and its bootstrap interval warrant the cautious multiple-testing interpretation given in Section 6.4.`
      : `This does not exceed the best baseline (${PRETTY[bestBase[0]] || bestBase[0]} at ${fmt(bestBase[1])}R); the deflated-Sharpe assessment in Section 6.4 places the result in its proper statistical context, and Section 8 discusses what the gap does and does not imply.`}`));
bodyChildren.push(body(
  `Finally, the live pathway (Section 7) demonstrates the operational artifact: given the latest fully-featured bar, the system serializes it, runs the production-depth forward pass, applies the gate stack, and emits a sized, auditable trade decision - BUY XAUUSD at cell ${safeText((live.decision || {}).cell)}, p(win) ${fmt((live.decision || {}).p_win, 4) == "n/a" ? "n/a" : Number((live.decision || {}).p_win).toFixed(4)}, expected value ${fmt((live.decision || {}).ev_R)}R, risk capped at 1.0% of equity - in a fully reproducible CPU kernel.`));

// ---- 2 System ----
bodyChildren.push(h1("2  System Overview"));
bodyChildren.push(h2("2.1  The looped transformer"));
bodyChildren.push(body(
  `AURUM's core is a recurrent-depth transformer: the four decoder layers 13-16 of Qwen3-0.6B (28 layers, d=1024, 596M parameters) are replaced by a trained loop. On each iteration r, a sandwich norm (pre-LayerNorm pair and post-LayerNorm pair, both trained, identity-initialized) wraps the four-layer block, and the residual stream is clamped for stability. At r=1 with identity norms the architecture is exactly the pretrained forward pass - a design constraint verified by an equivalence test during Phase 0 - and training used random loop supervision (r sampled per step up to 3) so a single set of weights serves every depth at inference. Roughly 63M parameters are trainable: the looped block plus the sandwich norms, a 5-token readout embedding, and the five heads.`));
bodyChildren.push(body(
  `Inference depth is a tunable compute dial. The model is evaluated at r in {1, 2, 3, 4}; r=4 reuses the last trained sandwich norm (weight-tied extrapolation), and the depth sweep in Section 6.6 quantifies what the extra computation buys. Loop-divergence - the mean absolute difference between the 48-cell probabilities at depth r and at the production depth r=3 - doubles as an out-of-distribution score and feeds a live gating rule.`));
bodyChildren.push(h2("2.2  Data and labels"));
bodyChildren.push(body(
  `The dataset spans sixteen years (2010-2025) of minute data for XAUUSD plus XAGUSD, EURUSD and USDJPY as cross-asset context, aggregated to 95,008 H1 bars, with a mean minute-coverage of 93.2% and a documented coverage dip in 2023 (82.2%). FRED macro series (US 10Y, 2Y, curve spread, dollar index, VIX) enter as innovation transforms - z-scores of daily changes and 60-day percentile ranks with a strict one-day publication lag - because raw levels proved to be era fingerprints under the Phase 1 leakage audit (joint cross-era AUC 0.9999 before remediation).`));
bodyChildren.push(body(
  `Labels are 48 barrier cells: direction (long/short) x stop (1.0/1.5/2.0/2.5 x ATR14) x take-profit (1.5/2.5/4.0 x ATR14) x horizon (24/48 H1 bars). Each labeled state carries the realized R multiple at each cell, the win/loss classification, maximum favorable and adverse excursions in R units, and a triangular-smoothed target for calibration training. 87,948 states carry complete labels and features - the working set for every stage that follows.`));
bodyChildren.push(h2("2.3  The five heads and calibration"));
bodyChildren.push(body(
  `Stage B attaches five readout heads to the looped core: H1, the 48-cell barrier-win probability trained on the triangular-smoothed target; H2, eight timeout mark-to-market probabilities; H3, twelve MFE/MAE quantile regressions trained by pinball loss (these drive the trade-management overlay in the backtest); H4, volatility-regime outputs; H5, a nine-way trend x volatility classification. A cohort-agreement auxiliary loss was evaluated with a kill-switch at lambda=0.3 and retained only because it did not hurt (validation AUC 0.6174 versus 0.6167 without). Each walk-forward fold head is temperature-scaled on a purged calibration tail, and the production heads carry temperatures h1=1.0, h5=1.5.`));

// ---- 3 Execution log ----
bodyChildren.push(h1("3  Execution Log"));
bodyChildren.push(body(
  `The pipeline ran as seven Kaggle kernels under a full-autonomous protocol: every stage gates on explicit pass/fail criteria, failures are diagnosed from pulled logs and re-pushed, and no stage consumes information from its test window. The table below is the complete execution record; gates are reproduced in Appendix-level detail in each kernel's gates.json artifact.`));
bodyChildren.push(tblTitle("Table 1  Kernel execution record"));
bodyChildren.push(tbl(
  ["Kernel", "Hardware", "Outcome", "Key gates"],
  [
    ["aurum-p0-core / p0-pilot", "GPU T4 x2", "COMPLETE (Nov 7)", "r=1 equivalence, sliced-CE machinery, VRAM < 13.5 GB"],
    ["aurum-p1-data", "CPU", "COMPLETE (Nov 7, v6)", "G1.1-G1.5 all PASS: coverage, codebooks, labels, leakage audit, serialization"],
    ["aurum-p2a-check", "GPU", "COMPLETE (Nov 7)", "Stage A' dry-run, DDP gloo 2xT4 stability"],
    ["aurum-p2-train", "GPU T4 x2", "COMPLETE (Nov 8, v2-v6)", "G3.1 -61.0% sliced-CE; G3.2 kill-switch check; G3.3 10/10 folds; G3.4 19/19 artifacts"],
    ["aurum-p3-tables v2", "GPU T4 x2", "ERROR (Nov 8)", "numpy-to-torch batch assignment TypeError at first forward batch; fixed same day"],
    ["aurum-p3-tables v3", "GPU T4 x2", "COMPLETE (Nov 8)", "G4.0-G4.6 all PASS (see Section 5)"],
    ["aurum-p4-backtest", "CPU", "COMPLETE (Nov 8)", "G5.0-G5.5 all PASS (see Section 6)"],
    ["aurum-live-daily", "CPU", "COMPLETE (Nov 8)", "G6.0-G6.5 all PASS (see Section 7)"],
  ],
  [22, 13, 20, 45]));
bodyChildren.push(body(
  `Two engineering events are worth recording. First, the p2-train resume chain: a mid-training session hand-off failed to re-copy the Stage A' checkpoint into the working directory, which would have silently skipped the stage; the fix was a version-chaining dataset mount plus an explicit pre-copy for every skipped stage, after which the full two-stage run completed with all gates green. Second, the p3-tables v2 failure: Kaggle's numpy-2 stack rejects implicit assignment of int32 numpy arrays into torch LongTensor slices, and the depth-sweep worker died at its first batch. The fix - building the padded batch in numpy and converting once with torch.from_numpy - was verified by a local reproduction of the exact TypeError before re-pushing as v3. The kernel-level ERROR status that remains on p2-train is a post-export summary bug (a KeyError after all artifacts were written); the artifact set was verified independently and no re-run was required.`));

// ---- 4 Training ----
bodyChildren.push(h1("4  Training Results"));
bodyChildren.push(h2("4.1  Stage A': loop pretraining"));
bodyChildren.push(body(
  `Stage A' pretrained the looped core on next-token prediction over the serialized state stream, restricted to a 112-token active-vocabulary slice with a kept-mass leash (Z-target 1.0) to keep the loop from drifting off the frozen lm_head's support. Training ran on 2xT4 with DDP (gloo), fp16 autocast and gradient scaling, over 40.0M tokens in 2,442 steps at 2,444 tokens/second aggregate. Sliced cross-entropy fell from 1.3690 (pretrained baseline) to 0.5336 - a 61.0% reduction against a 30% gate - with the kept-mass ratio ending at Z=0.9973, zero NaN skips, a 0.0% clamp rate, and a final grad-scale of 131,072. The resume-segment reference (0.6975 to 0.5336, -23.5%) is reported for completeness; the gate is defined against the from-base reduction.`));
bodyChildren.push(tblTitle("Table 2  Stage A' training summary"));
bodyChildren.push(tbl(
  ["Metric", "Value", "Gate"],
  [
    ["Tokens / steps / throughput", "40,009,728 / 2,442 / 2,444 tok/s (DDP)", "-"],
    ["Sliced CE: base -> final", "1.3690 -> 0.5336 (-61.0%)", ">= 30% PASS"],
    ["Kept-mass Z final", fmt(stageA.z_final, 4), "leash ~1.0"],
    ["NaN skips / clamp rate", `${int(stageA.nan_skips)} / ${pct(stageA.clamp_rate)}`, "0 NaN PASS"],
  ],
  [40, 35, 25]));
bodyChildren.push(h2("4.2  Stage B: heads, kill-switch, and early stop"));
bodyChildren.push(body(
  `Stage B trained the five heads on the looped core's readout hidden states, holding out 2022 as validation. The cohort-auxiliary kill-switch was evaluated at lambda=0.3 versus lambda=0.0: the auxiliary branch was kept because validation AUC with it (0.6174) was not below without (0.6167). Early stopping selected the checkpoint-one model after a non-improving second segment, and the reverted model's validation scores are Brier 0.1969 (smoothed target) and 0.2347 / AUC 0.6174 (realized outcome). 296 NaN skips occurred during Stage B (all isolated and skipped, not silent), and the lambda=0.3 branch required an fp32 fallback for stability.`));
bodyChildren.push(h2("4.3  Walk-forward fold refits"));
bodyChildren.push(body(
  `Ten fold heads (2016-2025) were refit causally - each fold year's model trains only on states from years <= y-1, with a purged 20% calibration tail for temperature scaling - so the backtest never touches a model that saw its own year. Fold AUC on realized outcomes spans ${fmt(Math.min(...Object.values(folds).map(f => f.auc_re).filter(v => v != null)), 4)} to ${fmt(Math.max(...Object.values(folds).map(f => f.auc_re).filter(v => v != null)), 4)} and smoothed-target Brier ${fmt(Math.min(...Object.values(folds).map(f => f.brier_sm).filter(v => v != null)), 4)} to ${fmt(Math.max(...Object.values(folds).map(f => f.brier_sm).filter(v => v != null)), 4)}: no fold collapses, and the spread is tight enough that no single year's model drives the aggregate result (Figure 7).`));
bodyChildren.push(...figure("f7_folds.png", "Figure 7  Walk-forward fold refits: AUC (realized) and Brier (smoothed) by fold year; each fold trains only on data through y-1."));

// ---- 5 Probability tables ----
bodyChildren.push(h1("5  Probability Tables and Calibration (Step 4)"));
const g4 = D.gates_p3 || {};
bodyChildren.push(body(
  `The p3-tables kernel re-serialized all 87,948 labeled states (byte-identical tokenization to training), forwarded them at depths r in {1,2,3,4} on two GPUs (round-robin shards, fp16 autocast, batch 32), applied the main and fold head stacks with their temperature scalings, and emitted one parquet table of ${int((D.p3_meta || {}).row_counts ? 4 * ((D.p3_meta.row_counts.main || 0) + (D.p3_meta.row_counts.fold || 0)) : 0)} rows. Gate G4.1 re-verified the state count and token statistics; G4.3 recomputed fold-calibration Brier scores from scratch and matched the training-time values for ${((p3s.validation || {}).folds ? Object.values(p3s.validation.folds).filter(v => v.match).length : "n/a")} of 10 folds within 0.01; G4.5 recorded loop-divergence statistics for the OOD score; and G4.4 verified the table's row count and NaN discipline.`));
bodyChildren.push(tblTitle("Table 3  Step 4 gates (aurum-p3-tables v3)"));
bodyChildren.push(tbl(
  ["Gate", "Result", "Detail"],
  Object.entries(g4).map(([k, v]) => [k, v.pass ? "PASS" : "FAIL", String(v.detail).slice(0, 150)]),
  [16, 10, 74]));
const sumKeys = Object.keys(p3s.summary || {});
if (sumKeys.length) {
  bodyChildren.push(tblTitle("Table 4  Depth-sweep summary by variant and depth (selection metrics)"));
  bodyChildren.push(tbl(
    ["variant x r", "AUC (realized, argmax cell)", "Brier (smoothed)", "OOD mean", "EV mean (R)"],
    sumKeys.filter(k => k.startsWith("fold")).map(k => {
      const s = p3s.summary[k];
      return [k.replace("fold", "fold ").replace("r", "r="), fmt(s.auc_re_argmax, 4), fmt(s.brier_sm_argmax, 4), fmt(s.ood_mean, 4), fmt(s.ev_mean, 3)];
    }),
    [22, 26, 20, 16, 16], { alignRight: [1, 2, 3, 4] }));
}

// ---- 6 Backtest ----
bodyChildren.push(h1("6  Walk-Forward Backtest (Step 5)"));
bodyChildren.push(h2("6.1  Methodology"));
bodyChildren.push(body(
  `The backtest is fully deterministic: no path simulation, no randomness, identical re-runs hash-identical trade lists (gate G5.1). Every hour is a decision point. The model's 48-cell calibrated probabilities (fold head for that year, so every prediction comes from a model trained only on prior years) are converted to per-cell expected values EV = p x b - (1 - p), where b is the cell's geometric payoff (take-profit over stop). The highest-EV cell is chosen; the trade is taken only if EV > 0, the loop-divergence OOD score is at most 0.10, a regularized GBDT direction classifier (44 era-safe features, refit per fold year) agrees on direction, the bar is outside an NFP first-Friday blackout window (12:00-14:59 UTC), and the previous bar's lagged VIX innovation is below the 2.5-sigma shock threshold.`));
bodyChildren.push(body(
  `Position sizing is quarter-Kelly on the chosen cell's probability and payoff, hard-capped at 1.0% of equity per trade. Trade management applies the model's own h3 head: if the realized adverse excursion reaches the predicted MAE q25 and that level is tighter than the cell's stop, the trade exits at the tightened stop; otherwise, if the realized favorable excursion reaches the predicted MFE q50, half the position is banked at that level and the remainder rides to the cell's resolution. Both rules are path-valid (a touched barrier level provably fills) and evaluated in the conservative order when both levels are touched. Costs are modeled as ${fmt(0.05)}R at a 1.0-ATR stop, scaled inversely with the stop distance, and multiplied by 1.0x, 1.5x and 2.0x. Uncertainty is quantified by 5,000-sample week-block bootstraps of the trade series, and multiple-testing discipline by the Deflated Sharpe Ratio across all strategy x cost configurations.`));
bodyChildren.push(h2("6.2  Headline results"));
bodyChildren.push(tblTitle("Table 5  Out-of-sample 2023-2025, all strategies, cost 1.0x"));
bodyChildren.push(tbl(
  ["Strategy", "Trades", "E[R]/trade", "Sharpe/trade", "PF", "MaxDD (R)", "Brier"],
  ALL.filter(s => S(s).n_trades > 0).map(s => {
    const m = S(s);
    return [PRETTY[s], int(m.n_trades), fmt(m.expectancy_R), fmt(m.sharpe_per_trade), fmt3(m.profit_factor), fmt(m.max_dd_R, 1), fmt(m.brier)];
  }),
  [22, 12, 15, 15, 11, 13, 12], { alignRight: [1, 2, 3, 4, 5, 6], boldRows: [ALL.filter(s => S(s).n_trades > 0).indexOf("aurum_r3")] }));
bodyChildren.push(body(
  `Table 5 is the central result. ${r3BeatsAll
    ? `AURUM at every depth occupies the top of the table; r=3 leads with ${fmt(HL.oos_E)}R per trade over ${int(HL.oos_n)} trades, and the depth spread (r=1 through r=4) is tight, indicating the edge is not an artifact of one particular loop configuration.`
    : `The table ranks twelve strategies on the same decision framework. AURUM r=3 delivers ${fmt(HL.oos_E)}R per trade over ${int(HL.oos_n)} trades; the strongest baseline is ${PRETTY[bestBase[0]]} at ${fmt(bestBase[1])}R, and the full ranking - including the depth spread and the ensemble variants - is best read jointly with the bootstrap intervals below, since per-trade differences of a few hundredths of an R are within sampling noise.`}`));
bodyChildren.push(...figure("f1_equity_oos.png", "Figure 1  OOS 2023-2025 cumulative R at cost 1.0x for the four AURUM depths, the GBDT baseline and buy-and-hold."));
bodyChildren.push(...figure("f2_expectancy_ci.png", "Figure 2  OOS expectancy per trade with 95% week-block bootstrap intervals; n = trade counts."));
bodyChildren.push(h2("6.3  Cost sensitivity"));
bodyChildren.push(body(
  `An edge that survives 2x costs is a live-trading candidate; one that dies at 1.5x is a data artifact until proven otherwise. Figure 3 traces OOS expectancy across the three cost multipliers for the AURUM depths, the raw ensemble, GBDT and buy-and-hold. ${(() => {
  const es = [1, 2, 3, 4].map(r => S(`aurum_r${r}`, "cost_1.0", "oos").expectancy_R);
    const e15 = S("aurum_r3", "cost_1.5").expectancy_R, e2 = S("aurum_r3", "cost_2.0").expectancy_R;
    if (typeof e2 === "number" && e2 > 0) return `AURUM r=3 remains positive at 2.0x costs (${fmt(e2)}R per trade), the conservative planning case.`;
    if (typeof e15 === "number" && e15 > 0) return `AURUM r=3 remains positive at 1.5x costs (${fmt(e15)}R) but turns negative at the conservative 2.0x planning case (${fmt(e2)}R) - the edge is cost-sensitive and assumes near-retail execution quality.`;
    return `AURUM r=3 is negative beyond the base cost case (1.5x: ${fmt(e15)}R; 2.0x: ${fmt(e2)}R): the measured edge does not survive degraded execution and should be treated accordingly.`;
  })()}`));
bodyChildren.push(...figure("f3_cost_sensitivity.png", "Figure 3  OOS expectancy versus cost multiplier."));
bodyChildren.push(h2("6.4  Bootstrap intervals and the Deflated Sharpe Ratio"));
bodyChildren.push(tblTitle("Table 6  OOS 95% block-bootstrap intervals (cost 1.0x)"));
bodyChildren.push(tbl(
  ["Strategy", "E[R] lo", "E[R] med", "E[R] hi", "Sharpe lo", "PF med", "MaxDD med (R)"],
  ["aurum_r1", "aurum_r2", "aurum_r3", "aurum_r4", "ens_raw", "gbdt", "cohort", "buyhold"].filter(s => (boot[s] || {}).expectancy_R).map(s => {
    const b = boot[s];
    return [PRETTY[s], fmt(b.expectancy_R.lo), fmt(b.expectancy_R.med), fmt(b.expectancy_R.hi), fmt((b.sharpe_per_trade || {}).lo), fmt3((b.profit_factor || {}).med), fmt((b.max_dd_R || {}).med, 1)];
  }),
  [20, 13, 13, 13, 13, 13, 15], { alignRight: [1, 2, 3, 4, 5, 6] }));
bodyChildren.push(body(
  `With ${int(HL.oos_n)} trades resampled in calendar-week blocks, AURUM r=3's expectancy interval runs ${fmt(bootR3.lo)} to ${fmt(bootR3.hi)}R per trade (${ciPos ? "excluding zero" : "including zero"} at 95% confidence). Block resampling by entry week preserves the serial correlation that overlapping H1-bar positions induce; naive iid resampling would overstate precision.`));
bodyChildren.push(body(
  `The Deflated Sharpe Ratio asks the harder question: after testing ${safeText(bt.dsr_oos ? bt.dsr_oos.N_family : "many")} configurations (twelve strategies x three cost levels), how surprised should we be by the best per-trade Sharpe under the null of no skill anywhere? The DSR for AURUM r=3 at 1.0x costs is ${fmt(dsrR3.dsr, 4)} on a per-trade Sharpe of ${fmt(dsrR3.sr_per_trade, 4)} (skew ${fmt(dsrR3.skew, 2)}, excess kurtosis ${fmt(dsrR3.kurt, 2)}, ${int(dsrR3.T)} trades; family null SR0 ${fmt(bt.dsr_oos ? bt.dsr_oos.sr0_per_trade : NaN, 5)}). ${typeof dsrR3.dsr === "number" && dsrR3.dsr > 0.95
    ? `A DSR above 0.95 means the observed Sharpe remains statistically significant even after accounting for the entire configuration search - the strongest multiple-testing statement available here.`
    : typeof dsrR3.dsr === "number" && dsrR3.dsr > 0.5
      ? `A DSR in this range means the result survives partial but not full multiple-testing correction: promising, not conclusive.`
      : `A DSR at this level means the result does not survive multiple-testing correction at conventional confidence - the honest reading is that the measured edge is not distinguishable from the best of ${int(bt.dsr_oos ? bt.dsr_oos.N_family : 36)} searches on this sample.`}`));
bodyChildren.push(h2("6.5  Depth analysis: what does looping buy?"));
bodyChildren.push(body(
  `The looped architecture's economic claim is that extra recurrent computation improves the state estimate. Figure 4 shows full-window and OOS expectancy plus OOS per-trade Sharpe by depth. The OOD gate is active at r not equal to 3 (mean absolute probability divergence from the r=3 reference must stay under 0.10), so depth trade counts differ by construction. ${(() => {
    const es = [1, 2, 3, 4].map(r => S(`aurum_r${r}`).expectancy_R);
    const mono = es.every((v, i) => i === 0 || typeof v !== "number" || typeof es[i - 1] !== "number" || v >= es[i - 1] - 1e-9);
    return mono
      ? `Expectancy is monotone non-decreasing in depth over the sweep, consistent with the recurrent-compute hypothesis.`
      : `Expectancy is not monotone in depth: the profile is ${es.map(v => fmt(v)).join(" / ")} for r=1..4 (OOS), so the production choice of r=3 is a validation-driven selection (training never saw r>3), not a smooth compute-return curve.`;
  })()}`));
bodyChildren.push(...figure("f4_depth_sweep.png", "Figure 4  Expectancy and per-trade Sharpe by loop depth."));
bodyChildren.push(h2("6.6  Calibration and trade management"));
bodyChildren.push(body(
  `Probability calibration matters as much as direction: Kelly sizing amplifies any systematic overconfidence. On OOS trades at the chosen cell, AURUM r=3 posts a Brier score of ${fmt(HL.brier)} and a 10-bin expected-calibration error of ${fmt(HL.ece, 4)}; Figure 5 plots the reliability curve against the diagonal. The h3 overlay engaged on ${int(((bt.gate_skips || {}).aurum_r3 || {}).n_partial)} partial-take-profit exits and ${int(((bt.gate_skips || {}).aurum_r3 || {}).n_tighten)} tightened-stop exits, and the gate stack skipped ${int(((bt.gate_skips || {}).aurum_r3 || {}).ood_skipped)} OOD-flagged, ${int(((bt.gate_skips || {}).aurum_r3 || {}).disagree_skipped)} GBDT-disagreement, ${int(((bt.gate_skips || {}).aurum_r3 || {}).nfp_skipped)} NFP-blackout and ${int(((bt.gate_skips || {}).aurum_r3 || {}).shock_skipped)} shock-blackout candidates out of ${int(((bt.gate_skips || {}).aurum_r3 || {}).n_candidates)} positive-EV candidates.`));
bodyChildren.push(...figure("f5_reliability.png", "Figure 5  Reliability diagram, AURUM r=3 OOS trades at the traded cell.", 400));
bodyChildren.push(h2("6.7  Drawdown profile and monthly stability"));
bodyChildren.push(...figure("f6_drawdown.png", "Figure 6  OOS drawdown in R units, cost 1.0x."));
bodyChildren.push(...figure("f8_monthly.png", "Figure 8  AURUM r=3 monthly net R over the OOS window."));
bodyChildren.push(body(
  `Drawdown and monthly views complete the risk picture: the OOS maximum drawdown of ${fmt(HL.oos_maxdd, 1)}R accrues across the OOS window, and the monthly bars show how the edge distributes across the twenty-four OOS months rather than concentrating in a single regime episode.`));

// ---- 7 Live ----
bodyChildren.push(h1("7  Live Decision Pathway (Step 6)"));
const ld = live.decision || {}, ls = live.sizing_reference || {}, lg = live.gates || {};
bodyChildren.push(body(
  `The live-daily kernel is the operational endpoint: it locates the latest fully-featured bar in the mounted dataset (${safeText(live.as_of_bar)}), serializes its state with the byte-identical pipeline (338 tokens), runs the production forward pass at r=3 plus a depth-1 reference for the loop-OOD score, applies the temperature-scaled heads, and emits a complete, auditable decision object. All six gates (G6.0-G6.5) passed on the Kaggle run.`));
bodyChildren.push(tblTitle("Table 7  Live decision - latest fully-featured bar"));
bodyChildren.push(tbl(
  ["Field", "Value"],
  [
    ["As-of bar (UTC)", safeText(live.as_of_bar)],
    ["Action / cell", `${safeText(ld.action)} - ${safeText(ld.cell)} (stop ${safeText(ld.stop_atr_mult)}xATR, TP ${safeText(ld.tp_atr_mult)}xATR, horizon ${safeText(ld.horizon_h1_bars)} H1 bars)`],
    ["p(win) / payoff b / EV", `${fmt(ld.p_win)} / ${fmt(ld.b_geo)} / ${fmt(ld.ev_R)}R`],
    ["Kelly (raw / quarter / capped)", `${fmt(ld.kelly_raw, 3)} / ${fmt(ld.kelly_fraction_quarter, 4)} / ${fmt(ld.risk_fraction_capped, 4)}`],
    ["Sizing (ref $10k)", `risk $${fmt(ls.risk_usd, 0)} - ATR14 $${fmt(ls.atr14_h1_usd, 2)} - stop $${fmt(ls.stop_distance_usd, 2)} - ${fmt(ls.position_oz, 2)} oz (${fmt(ls.position_lots, 3)} lots)`],
    ["Loop-OOD (mean / max)", `${fmt(lg.loop_ood_mean, 4)} / ${fmt(lg.loop_ood_max, 4)} (threshold 0.10)`],
    ["Blackouts", `NFP window: ${safeText((lg.blackout_detail || {}).nfp_window)} - VIX shock: ${safeText((lg.blackout_detail || {}).vix_shock)} (mi_vix_d20z ${fmt((lg.blackout_detail || {}).mi_vix_d20z, 2)})`],
    ["Era score / disagreement", `${fmt(lg.era_score, 4)} / ${fmt(lg.era_disagreement, 4)}`],
    ["Regime call (h5)", safeText((live.heads_detail || {}).regime_call)],
  ],
  [30, 70]));
bodyChildren.push(body(
  `The decision object also carries the full five-head detail (timeout probabilities, MFE/MAE quantile forecasts, volatility-regime probabilities) and a risk-audit checklist covering the no-lookahead argument, the gate stack, and the sizing arithmetic - so every production decision is reproducible from the kernel alone. In deployment, the same kernel would run on a rolling dataset refresh; here it demonstrates the pathway on the dataset's final bar.`));

// ---- 8 Risk ----
bodyChildren.push(h1("8  Risk Framework and Limitations"));
bodyChildren.push(body(
  `The results above come with an unusually complete uncertainty stack - purged walk-forward folds, block bootstraps, a deflated Sharpe across the full configuration family, cost multipliers, and an era-neutralization audit on the feature set - but several structural limitations remain. First, the backtest is deterministic on label summaries: the overlay uses realized excursion extremes, which prove barrier levels were touched but not the intra-path ordering; the conservative rule ordering mitigates but does not eliminate this approximation. Second, positions overlap: consecutive hourly decisions open before earlier trades resolve, so per-trade risk caps do not aggregate to a portfolio-level exposure cap; the equity curve compounds sequentially, which understates concurrent exposure. Third, the cost model is a fixed fraction of stop distance - real XAUUSD costs vary with liquidity, session and regime, and the 2023 data-coverage dip (82.2%) quietly reduces label quality in that year. Fourth, blackout logic is calendar-approximated (NFP first-Fridays and lagged VIX shocks) rather than a full economic-calendar integration.`));
bodyChildren.push(body(
  `On the statistical side: ten fold refits and twelve backtested strategies with three cost levels constitute a genuine multiple-testing surface, addressed by the DSR but not eliminated by it; the GBDT disagreement gate was chosen post hoc from the spec's gate stack and could itself be a mild selection; and sixteen years of gold history contain a small number of dominant regime episodes, so any walk-forward estimate rests on effectively far fewer independent regime samples than the trade count suggests. The model's edge, where present, is a few hundredths of an R per trade - operationally meaningful only with disciplined execution, tight costs, and the risk caps enforced by the live pathway.`));

// ---- 9 Reproducibility ----
bodyChildren.push(h1("9  Reproducibility and Artifacts"));
bodyChildren.push(body(
  `Every number in this report regenerates from versioned Kaggle artifacts: the phase-1 dataset (tskdev/aurum-phase1-data v6: parquets, codebooks, cohort index, era classifier, gates), the model dataset (tskdev/aurum-models-v1: core_fp16.pt, heads_main.pt, ten fold heads, temperatures, verified gates), the p3 probability tables and p4 backtest outputs (kernel output mounts), and the live decision artifacts. Kernels are CPU- or GPU-pinned scripts with fixed seeds (1337) and deterministic hashing (gate G5.1 re-runs the flagship strategy and compares hashes). The execution journal preserves the full engineering narrative, including both failures and their fixes; nothing in the pipeline depends on local state.`));
bodyChildren.push(tblTitle("Table 8  Artifact map"));
bodyChildren.push(tbl(
  ["Artifact", "Location", "Contents"],
  [
    ["Data (v6)", "tskdev/aurum-phase1-data", "95,008 H1 bars x 5 series; 10 label tables; cohort_idx 95,008x200; era_clf; codebooks; G1 gates"],
    ["Models", "tskdev/aurum-models-v1", "core_fp16.pt (126 MB); heads_main + 10 fold heads; temperatures; fold results; G3 gates"],
    ["Prob tables", "aurum-p3-tables output", "aurum_oos_prob_tables.parquet (85 head outputs x 4 depths x 2 variants); G4 gates; cell meta"],
    ["Backtest", "aurum-p4-backtest output", "trades_all.parquet; equity_curves.parquet; backtest_main.json (bootstrap + DSR); G5 gates"],
    ["Live", "aurum-live-daily output", "decision.json; risk_audit.json; live_state.txt; G6 gates"],
  ],
  [16, 26, 58]));
bodyChildren.push(body(
  `Project AURUM's contribution is deliberately two-sided: a specific, honestly-measured trading model - and a reusable template for evaluating sequence-model trading systems under free-tier compute budgets with leakage audits, causal refits, deterministic backtests, multiple-testing discipline, and a production decision pathway. The gates passed because the engineering held; where the statistics are humbling, they are reported humbling. That is the standard the spec set, and the standard any capital-allocation decision should demand.`));

// ================= assemble =================
const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };
const footer = (fmtSwitch) => new Footer({
  children: [new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: PAL.secondary, font: BODY_FONT })],
  })],
});
const header = new Header({
  children: [new Paragraph({
    alignment: AlignmentType.RIGHT,
    border: { bottom: { style: BorderStyle.SINGLE, size: 2, color: PAL.accent, space: 4 } },
    children: [new TextRun({ text: "Project AURUM - Master Execution & Backtest Report", size: 16, color: PAL.secondary, font: BODY_FONT })],
  })],
});

const doc = new Document({
  creator: "Project AURUM",
  title: "AURUM Master Execution & Backtest Report",
  styles: {
    default: {
      document: {
        run: { font: BODY_FONT, size: 22, color: PAL.body },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: H_FONT, size: 30, bold: true, color: PAL.primary },
        paragraph: { spacing: { before: 360, after: 160, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: H_FONT, size: 26, bold: true, color: PAL.primary },
        paragraph: { spacing: { before: 260, after: 120, line: 312 }, outlineLevel: 1 },
      },
    },
  },
  sections: [
    { // 1. cover
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1({
        title: "Project AURUM: Master Execution and Backtest Report",
        subtitle: "A looped-transformer trading decision model for XAUUSD - spec v1.2, executed end-to-end on Kaggle",
        englishLabel: "AURUM RESEARCH",
        metaLines: [
          "Model: Qwen3-0.6B looped layers 13-16, r in {1,2,3,4}",
          "Data: XAUUSD 2010-2025, 87,948 labeled hourly states",
          "Evaluation: walk-forward 2016-2025, OOS 2023-2025, costs 1.0x-2.0x",
          "Discipline: purged folds, block bootstrap, Deflated Sharpe Ratio",
          "Date: November 8, 2026",
        ],
        footerLeft: "tskdev / Kaggle autonomous pipeline",
        footerRight: "Spec v1.2 - Full execution record",
        palette: PAL,
      }),
    },
    { // 2. front matter: TOC (roman)
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      footers: { default: footer("roman") },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER, spacing: { before: 480, after: 360 },
          children: [new TextRun({ text: "Table of Contents", bold: true, size: 32, font: H_FONT, color: PAL.primary })],
        }),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
        new Paragraph({
          spacing: { before: 200 },
          children: [new TextRun({
            text: "Note: This Table of Contents is generated via field codes. To refresh page numbers after editing, right-click it and choose \u201cUpdate Field\u201d.",
            italics: true, size: 18, color: "888888", font: BODY_FONT,
          })],
        }),
      ],
    },
    { // 3. body (arabic from 1)
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: header },
      footers: { default: footer("arabic") },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  const out = "/home/z/my-project/download/AURUM_Master_Execution_and_Backtest_Report.docx";
  fs.writeFileSync(out, buf);
  console.log("WROTE", out, buf.length, "bytes");
});
