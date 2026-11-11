"use strict";
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, LevelFormat, TableOfContents, SectionType, PageBreak,
} = require("docx");
const fs = require("fs");
const H = require("./helpers");
const sectionsA = require("./content_a");
const sectionsB = require("./content_b");

const OUT = "/home/z/my-project/download/AURUM_Decider_Research_Report_and_Build_Plan.docx";

// ── Cover (Recipe R2, IG-1 Ink Gold) ──
const coverChildren = H.buildCoverR2({
  englishLabel: "DEEP RESEARCH REPORT AND BUILD PLAN",
  title: "AURUM Decider",
  subtitle: "A Looped-Transformer Decision Language Model for Gold (XAUUSD) Trading — Designed, Trained, and Operated End-to-End on Kaggle",
  metaLines: [
    "Prepared for: tskdev (Kaggle)",
    "Research and Planning Phase — Version 1.0",
    "November 6, 2026",
  ],
  footerRight: "Project AURUM · Research Deliverable · Choice-Order Invariant · Recurrent Depth",
  palette: H.PAL,
});

// ── Footers / headers ──
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
      children: [new TextRun({ text: "AURUM Decider — Deep Research Report and Build Plan", size: 18, color: H.PAL.gray, font: H.EN_FONT })],
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
      text: "Note: This Table of Contents is generated via field codes. To ensure page number accuracy after editing, please right-click the TOC and select \"Update Field.\"",
      italics: true, size: 18, color: "888888", font: H.EN_FONT,
    })],
  }),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
  // NOTE: no trailing PageBreak here — the body section below uses
  // SectionType.NEXT_PAGE, which performs the page break itself.
];

// ── Body ──
const bodyChildren = [];
for (const sec of [...sectionsA, ...sectionsB]) {
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
const numberingConfig = ["exec-recs", "lessons", "loopmech", "refs"].map(mkNum);

const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

const doc = new Document({
  creator: "Project AURUM",
  title: "AURUM Decider — Deep Research Report and Build Plan",
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

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("Written:", OUT, buf.length, "bytes");
});
