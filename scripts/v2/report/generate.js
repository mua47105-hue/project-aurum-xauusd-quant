"use strict";
// ============================================================================
// AURUM Spec v2.0 Master Execution & Tournament Report — main generator
// Usage: node scripts/v2/report/generate.js
// Output: download/AURUM_Spec_v2_Master_Execution_and_Tournament_Report.docx
// ============================================================================
const H = require("./helpers");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, PageNumber, NumberFormat, AlignmentType, WidthType,
  BorderStyle, ShadingType, TableLayoutType, SectionType, PageBreak,
  TabStopType, LeaderType, fs, path, OUTDIR, P, FONTS, HFONTS, loadData, jload,
} = H;

const c1 = require("./content_1");
const c2 = require("./content_2");
const c3 = require("./content_3");
const c4 = require("./content_4");

const STAMP = new Date().toISOString().slice(0, 10);

// ---------------------------------------------------------------- cover
function cover() {
  const padL = 1200, padR = 800;
  const titleLines = ["AURUM Spec v2.0", "Master Execution and", "Tournament Report"];
  const titlePt = 40;
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: P.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: 2600 } }));
  children.push(new Paragraph({ indent: { left: padL, right: padR }, spacing: { after: 600 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: P.accent, space: 8 } },
    children: [new TextRun({ text: "P R O J E C T   A U R U M", size: 20, color: P.accent,
      font: { ascii: "Arial" }, characterSpacing: 40 })] }));
  titleLines.forEach((ln, i) => children.push(new Paragraph({ indent: { left: padL },
    spacing: { after: i < titleLines.length - 1 ? 120 : 340, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
    children: [new TextRun({ text: ln, size: titlePt * 2, bold: true,
      color: P.cover.titleColor, font: { ascii: "Arial" } })] })));
  children.push(new Paragraph({ indent: { left: padL }, spacing: { after: 700 },
    children: [new TextRun({
      text: "XAU/USD H1 — The Continuous and Contractive Quantitative Engine, K1-K7",
      size: 24, color: P.cover.subtitleColor, font: { ascii: "Arial" } })] }));
  const metaLines = [
    "Code revision v2.0.2 — de-saturated active stack, dual hybrid benchmarks",
    "Champion under test: Spec v1.3 Flagship cap=3 (OOS Sharpe 2.444, DSR 0.895)",
    "Result: M7 Core-Satellite 2.635 OOS / 2.501 at 2.0x costs / DSR 0.941 — pre-registered 0.95 bar honestly missed; M7 Active Grand Hybrid 1.946",
    "18 strategies, 11 of 12 gates green (G6.1 honest miss), CSCV-PBO 0.0049, 5,000-sample block bootstrap",
  ];
  for (const line of metaLines) {
    children.push(new Paragraph({ indent: { left: padL + 200 }, spacing: { after: 90 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 21, color: P.cover.metaColor,
        font: { ascii: "Arial" } })] }));
  }
  children.push(new Paragraph({ spacing: { before: 2400 } }));
  children.push(new Paragraph({ indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: P.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: "AURUM Program — Engineering Series, Volume V", size: 16,
        color: P.cover.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                                  " }),
      new TextRun({ text: STAMP, size: 16, color: P.cover.footerColor, font: { ascii: "Arial" } }),
    ] }));
  const noBorders = { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
    insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE } };
  const cellNoBorders = { top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
    left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE } };
  return [new Table({ width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED, borders: noBorders,
    rows: [new TableRow({ height: { value: 15838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg },
        borders: cellNoBorders, children })] })] })];
}

// ---------------------------------------------------------------- assemble
const D = loadData();

// ---- static two-pass TOC (page numbers filled from page_map.json) --------
const TOC_ENTRIES = [
  ["1. Executive Summary", 0],
  ["2. The Unified v2.0 Codebase", 0],
  ["2.1 The v2.0.1 revision", 1],
  ["2.2 The v2.0.2 revision", 1],
  ["3. Phase 1 — The Unified Feature Store (K1)", 0],
  ["4. Phase 2 — CPU Models and the Execution Engine (K2)", 0],
  ["5. Phase 3 — The Neural and Contractive-Loop Tournament (K3)", 0],
  ["6. Engineering Flaw #8 — The M2 RFF Fold-Scale Saturation", 0],
  ["6.1 Root cause", 1],
  ["6.2 Impact propagation and quarantine", 1],
  ["6.3 Remediation", 1],
  ["7. Phase 4 — The Full Walk-Forward Backtest and Statistical Gates (K6)", 0],
  ["7.1 The v2.0.1 causality regression", 1],
  ["7.2 The tournament", 1],
  ["7.3 The two Grand Hybrids — active vs core-satellite", 1],
  ["7.4 Statistics — bootstrap, DSR, PBO", 1],
  ["8. Phase 5 — The Live v2.0 Decision System (K7)", 0],
  ["9. The Verdict — v1.2 vs v1.3 vs v2.0", 0],
  ["10. Statistical Honesty, Reconciliation and Limitations", 0],
  ["10.1 What the numbers are, and are not", 1],
  ["10.2 Known limitations", 1],
  ["10.3 Reconciliation of quoted vintages", 1],
  ["11. Appendices", 0],
  ["Appendix A — Reproduction map", 1],
  ["Appendix B — Locked constants and gate register", 1],
];
const pageMap = jload(path.join(__dirname, "page_map.json"), {});
function tocLine(text, level, pageNo) {
  const shown = level === 1 ? "        " + text : text;
  return new Paragraph({
    tabStops: [{ type: TabStopType.RIGHT, position: 9210, leader: LeaderType.DOT }],
    spacing: { after: level === 0 ? 60 : 40, line: 280 },
    indent: level === 1 ? { left: 360 } : undefined,
    children: [
      new TextRun({ text: shown, size: level === 0 ? 22 : 20,
        bold: level === 0, color: level === 0 ? P.primary : P.body,
        font: level === 0 ? HFONTS : FONTS }),
      new TextRun({ text: "\t" + (pageNo || ""), size: 20, color: P.body, font: FONTS }),
    ],
  });
}
const tocBlock = [
  new Paragraph({ spacing: { before: 160, after: 160 },
    children: [new TextRun({ text: "Contents", bold: true, size: 30, color: P.primary, font: HFONTS })] }),
  ...TOC_ENTRIES.map(([t, lv]) => tocLine(t, lv, pageMap[t])),
];

const D_UNUSED = null;
const children = [
  ...cover(),
  ...tocBlock,
  new Paragraph({ children: [new PageBreak()] }),
  ...c1.build(D),
  ...c2.build(D),
  ...c3.build(D),
  ...c4.build(D),
];

const pageHeader = new Header({ children: [new Paragraph({
  alignment: AlignmentType.RIGHT, spacing: { after: 60 },
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: P.accent, space: 4 } },
  children: [new TextRun({ text: "AURUM Spec v2.0 — Master Execution and Tournament Report",
    size: 16, color: P.secondary, font: { ascii: "Arial" } })] })] });
const pageFooter = new Footer({ children: [new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "— ", size: 16, color: P.secondary, font: { ascii: "Arial" } }),
    new TextRun({ children: [PageNumber.CURRENT], size: 16, color: P.secondary, font: { ascii: "Arial" } }),
    new TextRun({ text: " —", size: 16, color: P.secondary, font: { ascii: "Arial" } })] })] });

const doc = new Document({
  creator: "AURUM Program",
  title: "AURUM Spec v2.0 Master Execution and Tournament Report",
  description: "K1-K7 execution record and tournament verdict, code rev v2.0.2",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONTS, size: 22, color: P.body } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { bold: true, size: 32, color: P.primary, font: HFONTS },
        paragraph: { spacing: { before: 360, after: 160 } } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { bold: true, size: 26, color: P.primary, font: HFONTS },
        paragraph: { spacing: { before: 260, after: 120 } } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { bold: true, size: 23, color: P.secondary, font: HFONTS },
        paragraph: { spacing: { before: 200, after: 100 } } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 },
        margin: { top: 1247, right: 1100, bottom: 1247, left: 1100 },
        pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      titlePage: true,
    },
    headers: { default: pageHeader },
    footers: { default: pageFooter, first: new Footer({ children: [new Paragraph({})] }) },
    children,
  }],
});

const OUT = path.join(OUTDIR, "AURUM_Spec_v2_Master_Execution_and_Tournament_Report.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("written:", OUT, (buf.length / 1024).toFixed(0) + " KB");
  console.log("sections:", children.length, "children");
}).catch(e => { console.error("PACK ERROR:", e.message); process.exit(1); });
