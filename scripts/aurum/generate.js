// generate.js - assembles Project AURUM deep research report
const {
  Document, Packer, Paragraph, TextRun, Header, Footer, PageNumber,
  AlignmentType, SectionType, NumberFormat, TableOfContents, PageBreak,
} = require("docx");
const fs = require("fs");
const H = require("./helpers");
const C1 = require("./content1");
const C2 = require("./content2");
const C3 = require("./content3");

const { buildCoverR1, P, FONT } = H;

const pgSize = { width: 11906, height: 16838 };
const pgMargin = { top: 1440, bottom: 1440, left: 1701, right: 1417 };

function makeHeader() {
  return new Header({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ text: "Project AURUM - Deep Research Report", size: 18, color: "808080", font: FONT })],
    })],
  });
}

// Front-matter footer uses marker color 7F7F7F (post-processed to ROMAN); body 808080 (arabic)
function makeFooter(markerColor) {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0, line: 240 },
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: markerColor, font: FONT })],
    })],
  });
}

const coverConfig = {
  title: "Project AURUM",
  subtitle: "A Jev-inspired looped-transformer decision model for gold (XAUUSD): findings, limitations and build plan",
  englishLabel: "DEEP RESEARCH REPORT",
  metaLines: [
    "Prepared for: tskdev (Kaggle account)",
    "Research window: 6 November 2026",
    "Method: DeerFlow deep-research pipeline - 5 parallel threads, 180+ sources",
    "Stack: Kaggle free tier / Hugging Face / Dukascopy / OANDA practice",
  ],
  footerLeft: "Project AURUM",
  footerRight: "November 2026",
  palette: P,
};

// ---------------- front matter ----------------
const frontMatter = [];
frontMatter.push(...C1.execSummary());
// TOC title - MUST NOT use Heading style; pageBreakBefore starts TOC on a fresh page
frontMatter.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  pageBreakBefore: true,
  spacing: { before: 480, after: 360 },
  children: [new TextRun({ text: "Table of Contents", bold: true, size: 32, color: H.HEAD_COLOR, font: H.FONT_HEAD })],
}));
frontMatter.push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }));
frontMatter.push(new Paragraph({
  spacing: { before: 200 },
  children: [new TextRun({
    text: "Note: This Table of Contents is generated via field codes. To ensure page number accuracy after editing, please right-click the TOC and select \u201CUpdate Field.\u201D",
    italics: true, size: 18, color: "888888", font: FONT,
  })],
}));

// ---------------- body ----------------
const bodyChildren = [
  ...C1.ch1(), ...C1.ch2(), ...C1.ch3(), ...C1.ch4(),
  ...C2.ch5(), ...C2.ch6(), ...C2.ch7(), ...C2.ch8(), ...C2.ch9(),
  ...C3.ch10(), ...C3.ch11(), ...C3.ch12(), ...C3.ch13(),
];

const doc = new Document({
  creator: "Project AURUM research pipeline",
  title: "Project AURUM - Deep Research Report",
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 312 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimHei" }, size: 32, bold: true, color: "1A1A1A" },
        paragraph: { spacing: { before: 360, after: 160, line: 340 }, outlineLevel: 0 },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: "1A1A1A" },
        paragraph: { spacing: { before: 260, after: 120, line: 312 }, outlineLevel: 1 },
      },
    },
  },
  sections: [
    { // Section 1: cover - margin 0, no header/footer, no pageNumbers
      properties: { page: { size: pgSize, margin: { top: 0, bottom: 0, left: 0, right: 0 } } },
      children: buildCoverR1(coverConfig),
    },
    { // Section 2: front matter - Roman numerals
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.UPPER_ROMAN } },
      },
      headers: { default: makeHeader() },
      footers: { default: makeFooter("7F7F7F") },
      children: frontMatter,
    },
    { // Section 3: body - Arabic from 1
      properties: {
        type: SectionType.NEXT_PAGE,
        page: { size: pgSize, margin: pgMargin, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } },
      },
      headers: { default: makeHeader() },
      footers: { default: makeFooter("808080") },
      children: bodyChildren,
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("/home/z/my-project/download/Project_AURUM_Deep_Research_Report.docx", buf);
  console.log("DOCX written:", buf.length, "bytes");
});
