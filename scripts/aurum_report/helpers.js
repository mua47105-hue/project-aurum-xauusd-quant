"use strict";
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TabStopType, TabStopPosition, LevelFormat, TableOfContents,
  SectionType, TableLayoutType, ExternalHyperlink,
} = require("docx");

// ─────────────────────────────────────────────────────────────
// Palette: IG-1 Ink Gold (finance / investment) from design-system.md
// ─────────────────────────────────────────────────────────────
const PAL = {
  bg: "1A1A1A", accent: "C9A84C",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "C9A84C", headerText: "1A1A1A", accentLine: "C9A84C", innerLine: "DDD5C0", surface: "F5F2E8" },
  headingColor: "1A1A1A",
  body: "000000",
  gray: "808080",
};

const noBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
};
const allNoBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
  insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
};

const EN_FONT = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "Times New Roman" };
const EN_HEAD = { ascii: "Times New Roman", hAnsi: "Times New Roman", eastAsia: "Times New Roman" };

// ─────────────────────────────────────────────────────────────
// Title layout (English-aware adaptation of calcTitleLayout)
// ─────────────────────────────────────────────────────────────
function splitTitleLinesEN(title, charsPerLine) {
  const words = title.split(/\s+/);
  const lines = [];
  let cur = "";
  for (const w of words) {
    if (cur === "") { cur = w; continue; }
    if ((cur + " " + w).length <= charsPerLine) cur = cur + " " + w;
    else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  // orphan prevention: merge a very short last line into previous
  if (lines.length > 1 && lines[lines.length - 1].length <= 4) {
    const last = lines.pop();
    lines[lines.length - 1] += " " + last;
  }
  return lines;
}

function calcTitleLayoutEN(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  const charWidth = (pt) => pt * 11; // avg Arial char ~0.55em
  let titlePt = preferredPt, lines;
  while (titlePt >= minPt) {
    const cpl = Math.floor(maxWidthTwips / charWidth(titlePt));
    if (cpl < 4) { titlePt -= 2; continue; }
    lines = splitTitleLinesEN(title, cpl);
    if (lines.length <= 3) break;
    titlePt -= 2;
  }
  if (!lines || lines.length > 3) {
    const cpl = Math.floor(maxWidthTwips / charWidth(minPt));
    lines = splitTitleLinesEN(title, cpl);
    titlePt = minPt;
  }
  return { titlePt, titleLines: lines };
}

// ─────────────────────────────────────────────────────────────
// Cover Recipe R2: Double-Rule Frame (Centered) — IG-1 Ink Gold
// Per design-system.md (validated recipe; zero nested tables)
// ─────────────────────────────────────────────────────────────
function buildCoverR2(config) {
  const P = config.palette;
  const padL = 1400, padR = 1400;
  const { titlePt, titleLines } = calcTitleLayoutEN(config.title, 11906 - padL - padR, 40, 24);
  const titleSize = titlePt * 2;
  const thickBorder = { style: BorderStyle.SINGLE, size: 18, color: P.accent, space: 20 };
  const children = [];

  // 1. Top rule
  children.push(new Paragraph({
    indent: { left: padL - 400, right: padR - 400 }, spacing: { before: 1200, after: 200 },
    border: { top: thickBorder }, children: [],
  }));
  // 2. Whitespace
  children.push(new Paragraph({ spacing: { before: 1800 } }));
  // 3. English label
  if (config.englishLabel) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 500 },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: P.accent, font: { ascii: "Calibri" }, characterSpacing: 40 })],
    }));
  }
  // 4. Main title
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: i < titleLines.length - 1 ? 80 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: P.cover.titleColor, font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  // 5. Subtitle
  if (config.subtitle) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 400, line: 340, lineRule: "atLeast" },
      indent: { left: 900, right: 900 },
      children: [new TextRun({ text: config.subtitle, size: 24, color: P.cover.subtitleColor,
        font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  // 6. Whitespace
  children.push(new Paragraph({ spacing: { before: 1200 } }));
  // 7. Meta lines — 18pt
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 100, line: Math.ceil(18 * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: line, size: 36, color: P.cover.metaColor,
        font: { eastAsia: "Arial", ascii: "Arial" } })],
    }));
  }
  // 8. Whitespace
  children.push(new Paragraph({ spacing: { before: 2000 } }));
  // 9. Footer + bottom rule
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    indent: { left: padL - 400, right: padR - 400 }, spacing: { before: 200 },
    border: { bottom: thickBorder },
    children: [new TextRun({ text: config.footerRight || "", size: 18, color: P.cover.footerColor, font: { ascii: "Arial" } })],
  }));

  // Single 16838 wrapper — the ONLY table
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: P.bg }, borders: noBorders,
        children,
      })],
    })],
  })];
}

// ─────────────────────────────────────────────────────────────
// Body builders (English formal report, Profile A, TNR)
// ─────────────────────────────────────────────────────────────
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { before: 400, after: 200, line: 380, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 32, color: PAL.headingColor, font: EN_HEAD })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140, line: 360, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 28, color: PAL.headingColor, font: EN_HEAD })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 120, line: 340, lineRule: "atLeast" },
    children: [new TextRun({ text, bold: true, size: 24, color: PAL.headingColor, font: EN_HEAD })],
  });
}
// body paragraph; segs = string OR array of {text, bold, italics, color, size}
function p(segs, opts = {}) {
  const runs = (typeof segs === "string" ? [{ text: segs }] : segs).map(s =>
    new TextRun({
      text: s.text, bold: !!s.bold, italics: !!s.italics,
      size: s.size || 24, color: s.color || PAL.body, font: EN_FONT,
    }));
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 312, after: opts.after != null ? opts.after : 120 },
    keepNext: !!opts.keepNext,
    children: runs,
  });
}
function note(text) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 312, after: 120 },
    indent: { left: 360, right: 360 },
    border: { left: { style: BorderStyle.SINGLE, size: 8, color: PAL.accent, space: 10 } },
    children: [new TextRun({ text, italics: true, size: 21, color: "505050", font: EN_FONT })],
  });
}
function bullet(text, level = 0) {
  return new Paragraph({
    bullet: { level },
    alignment: AlignmentType.LEFT,
    spacing: { line: 312, after: 60 },
    children: [new TextRun({ text, size: 24, color: PAL.body, font: EN_FONT })],
  });
}
function numbered(text, ref) {
  return new Paragraph({
    numbering: { reference: ref, level: 0 },
    alignment: AlignmentType.LEFT,
    spacing: { line: 312, after: 60 },
    children: [new TextRun({ text, size: 24, color: PAL.body, font: EN_FONT })],
  });
}
function caption(text) {
  return new Paragraph({
    keepNext: true,
    spacing: { before: 160, after: 80, line: 312 },
    children: [new TextRun({ text, bold: true, size: 21, color: "404040", font: EN_FONT })],
  });
}
function cell(text, opts = {}) {
  const runs = (typeof text === "string" ? [{ text }] : text).map(s =>
    new TextRun({ text: s.text, bold: !!s.bold, italics: !!s.italics, size: opts.size || 20,
      color: s.color || (opts.header ? PAL.table.headerText : "222222"), font: EN_FONT }));
  return new TableCell({
    children: [new Paragraph({
      alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
      spacing: { line: 312 },
      children: runs,
    })],
    shading: { type: ShadingType.CLEAR, fill: opts.fill || "FFFFFF" },
    margins: { top: 70, bottom: 70, left: 110, right: 110 },
    width: opts.width ? { size: opts.width, type: WidthType.PERCENTAGE } : undefined,
    verticalAlign: "top",
  });
}
// Standard table: header row gold, zebra rows, open left/right borders
function makeTable(headers, rows, widths) {
  const n = headers.length;
  const w = widths || Array(n).fill(Math.floor(100 / n));
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((t, i) => cell(t, { header: true, fill: PAL.table.headerBg, width: w[i], bold: true })),
  });
  const dataRows = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((t, i) => cell(t, { fill: ri % 2 === 1 ? PAL.table.surface : "FFFFFF", width: w[i] })),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: PAL.table.accentLine },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: PAL.table.innerLine },
      insideVertical: { style: BorderStyle.NONE },
    },
    rows: [headerRow, ...dataRows],
  });
}

// Block renderer: converts declarative blocks to docx elements
function renderBlocks(blocks) {
  const out = [];
  for (const b of blocks) {
    if (b.t === "p") out.push(p(b.x, b.opts || {}));
    else if (b.t === "h2") out.push(h2(b.x));
    else if (b.t === "h3") out.push(h3(b.x));
    else if (b.t === "note") out.push(note(b.x));
    else if (b.t === "bullets") b.items.forEach(it => out.push(bullet(it)));
    else if (b.t === "numbered") b.items.forEach(it => out.push(numbered(it, b.ref)));
    else if (b.t === "table") {
      if (b.caption) out.push(caption(b.caption));
      out.push(makeTable(b.headers, b.rows, b.widths));
      if (b.after) out.push(p(b.after, { after: 160 }));
      else out.push(new Paragraph({ spacing: { after: 80 } }));
    }
  }
  return out;
}

module.exports = {
  PAL, allNoBorders, noBorders, EN_FONT, EN_HEAD,
  buildCoverR2, h1, h2, h3, p, note, bullet, numbered, caption, cell, makeTable, renderBlocks,
};
