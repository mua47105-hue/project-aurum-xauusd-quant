// generate.js — assemble the AURUM v1.3 Master Report (.docx)
// 3-section architecture: cover (no page#) / front matter (Roman) / body (Arabic from 1)
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber,
  AlignmentType, HeadingLevel, NumberFormat, SectionType, TableOfContents,
  PageBreak, PageOrientation,
} = require("docx");
const fs = require("fs");
const { P, buildCoverR1 } = require("./helpers");
const c1 = require("./content1");
const c2 = require("./content2");

const OUT = "/home/z/my-project/download/AURUM_v13_report/AURUM_Master_Execution_and_Backtest_Report.docx";

const pgSize = { width: 11906, height: 16838, orientation: PageOrientation.PORTRAIT };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

function runningHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: "single", size: 2, color: "C9A84C", space: 4 } },
      children: [new TextRun({
        text: "AURUM Master Execution & Backtest Report — Spec v1.3",
        size: 16, color: "888888", font: { ascii: "Times New Roman", eastAsia: "SimSun" },
      })],
    })],
  });
}
function pageNumFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "808080" })],
    })],
  });
}

// ---------- cover ----------
const cover = buildCoverR1({
  title: "AURUM Master Execution and Backtest Report",
  subtitle: "Spec v1.3 - The Decision-Layer & Realism Upgrade: recurrent-depth Qwen3-0.6B on XAUUSD (2010-2025), non-overlapping portfolio execution, LCB-EV decision policy, and surviving Deflated Sharpe.",
  englishLabel: "PROJECT AURUM / SPEC V1.3",
  metaLines: [
    "Model:  Qwen3-0.6B re-fitted as recurrent transformer (loop depths r = 1..4)",
    "Market:  XAUUSD H1 2010-2025  |  OOS window: 2023-2025",
    "Policy:  multi-depth pooled posterior + LCB-EV + GBDT agreement gate",
    "Execution:  non-overlapping, concurrency-capped, 1% quarter-Kelly risk",
    "Verification:  Steps 1-6 all gates PASS (54 gates); DSR 0.895 on daily returns",
    "Status:  100% complete  |  0 GPU-seconds consumed by the v1.3 upgrade",
    "Date:  9 November 2026",
  ],
  footerLeft: "Project AURUM - Autonomous Quant Pipeline",
  footerRight: "Kaggle / tskdev",
  palette: { bg: P.bg, accent: P.accent, titleColor: "FFFFFF",
             subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078",
             cover: P.cover },
});

// fix: buildCoverR1 reads palette fields directly — pass IG-1 shape
// (handled inside helpers via PA.cover.* when given; see helpers.js)

// ---------- front matter: TOC ----------
const frontMatter = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 360 },
    children: [new TextRun({ text: "Table of Contents", bold: true, size: 32,
      font: { ascii: "Times New Roman", eastAsia: "SimHei" }, color: "000000" })],
  }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({
    spacing: { before: 200 },
    children: [new TextRun({
      text: "Note: This Table of Contents is generated via field codes. To ensure page number accuracy after editing, please right-click the TOC and select \"Update Field.\"",
      italics: true, size: 18, color: "888888",
      font: { ascii: "Times New Roman", eastAsia: "SimSun" },
    }), new PageBreak()],
  }),
  new Paragraph({
    spacing: { before: 60 },
    children: [new TextRun({
      text: "Revision history:  v1.2 (8 Nov 2026) - first full-pipeline master report.  v1.3 (9 Nov 2026) - decision-layer & realism upgrade; baseline audit, non-overlapping execution, LCB-EV policy, DSR on non-overlapping returns, G4.6/G5.2a repairs.",
      size: 20, color: "555555", italics: true,
      font: { ascii: "Times New Roman", eastAsia: "SimSun" },
    })],
  }),
];

// ---------- body ----------
const bodyChildren = [
  ...c1.sec1(), ...c1.sec2(), ...c1.sec3(), ...c1.sec4(), ...c1.sec5(),
  ...c2.sec6(), ...c2.sec7(), ...c2.sec8(), ...c2.sec9(), ...c2.sec10(), ...c2.secApp(),
];

const doc = new Document({
  creator: "Project AURUM autonomous pipeline",
  title: "AURUM Master Execution and Backtest Report - Spec v1.3",
  description: "Qwen3-0.6B recurrent-depth transformer trading system for XAUUSD: full execution audit and backtest under Spec v1.3.",
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: "000000" },
        paragraph: { spacing: { before: 400, after: 180, line: 312 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 30, bold: true, color: "000000" },
        paragraph: { spacing: { before: 280, after: 130, line: 312 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 26, bold: true, color: "000000" },
        paragraph: { spacing: { before: 220, after: 110, line: 312 }, outlineLevel: 2 },
      },
    },
  },
  sections: [
    { // Section 1: cover — margin 0, no footer, no page numbers
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: cover,
    },
    { // Section 2: front matter — Roman numerals
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin,
                pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: runningHeader() },
      footers: { default: pageNumFooter() },
      children: frontMatter,
    },
    { // Section 3: body — Arabic from 1
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin,
                pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: runningHeader() },
      footers: { default: pageNumFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
});
