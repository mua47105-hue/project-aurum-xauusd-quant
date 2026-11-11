"use strict";
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, LevelFormat, TableOfContents, SectionType,
} = require("docx");
const fs = require("fs");
const H = require("./helpers");
const sectionsA = require("./content_a");
const sectionsB = require("./content_b");
const sectionsC = require("./content_c");
const sectionsD = require("./content_d");

const OUT = "/home/z/my-project/download/AURUM_Phase01_Postmortem_4_Failures_Verified_Fixes.docx";

// ── Cover (Recipe R2, IG-1 Ink Gold — same family as prior AURUM deliverables) ──
const coverChildren = H.buildCoverR2({
  englishLabel: "PHASE 0/1 POST-MORTEM - DEEP RESEARCH PASS",
  title: "AURUM: Four Root-Cause Failures and Their Locally Verified Fixes",
  subtitle: "Cohort-Label Memorization, Era-Invariance and the Macro Panel, the 151,936-Vocabulary Bottleneck, and Zero-Crash Push Discipline - All Verified on Pulled Artifacts, Zero GPU Spent",
  metaLines: [
    "Prepared for: tskdev (Kaggle)",
    "Evidence base: pulled aurum-p1-data / p0-core / p0-pilot artifacts (766 MB), local CPU replication",
    "November 7, 2026",
  ],
  footerRight: "Project AURUM - DeerFlow Deep-Research Methodology - Model Judges, Code Executes",
  palette: H.PAL,
});

function pageFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: H.PAL.gray, font: H.EN_FONT })],
    })],
  });
}
function docHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { bottom: { style: "single", size: 2, color: H.PAL.accent, space: 4 } },
      spacing: { after: 60 },
      children: [new TextRun({ text: "AURUM - Phase 0/1 Post-Mortem: Four Failures, Verified Fixes", size: 18, color: H.PAL.gray, font: H.EN_FONT })],
    })],
  });
}

// ── Front matter: TOC ──
const frontMatter = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 480, after: 200 },
    children: [new TextRun({ text: "Table of Contents", bold: true, size: 32, color: H.PAL.headingColor, font: H.EN_HEAD })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({
      text: "Note: This Table of Contents is generated via field codes. To refresh page numbers after editing, right-click the TOC and select Update Field.",
      italics: true, size: 18, color: "888888", font: H.EN_FONT,
    })],
  }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
];

// ── Body ──
const bodyChildren = [];
for (const sec of [...sectionsA, ...sectionsB, ...sectionsC, ...sectionsD]) {
  bodyChildren.push(H.h1(sec.h1));
  bodyChildren.push(...H.renderBlocks(sec.blocks));
}

// ── Numbering configs (unique reference per list) ──
const mkNum = (ref) => ({
  reference: ref,
  levels: [{
    level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 640, hanging: 360 } } },
  }],
});
const numberingConfig = ["refs"].map(mkNum);

const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  creator: "Project AURUM",
  title: "AURUM - Phase 0/1 Post-Mortem: Four Failures, Verified Fixes",
  styles: {
    default: {
      document: {
        run: { font: H.EN_FONT, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: H.EN_HEAD, size: 32, bold: true, color: H.PAL.headingColor },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: H.EN_HEAD, size: 28, bold: true, color: H.PAL.headingColor },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 },
      },
      heading3: {
        run: { font: H.EN_HEAD, size: 24, bold: true, color: H.PAL.headingColor },
        paragraph: { spacing: { before: 220, after: 120 }, outlineLevel: 2 },
      },
    },
  },
  numbering: { config: numberingConfig },
  sections: [
    { // Section 1: Cover — no footer, no page numbers
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: coverChildren,
    },
    { // Section 2: Front matter — Roman numerals
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: docHeader() },
      footers: { default: pageFooter() },
      children: frontMatter,
    },
    { // Section 3: Body — Arabic from 1
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: docHeader() },
      footers: { default: pageFooter() },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
});
