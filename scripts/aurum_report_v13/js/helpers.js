// helpers.js — shared builders for the AURUM v1.3 Master Report (docx)
// English formal report: Profile A (TNR body, SimHei headings, pure black),
// IG-1 Ink Gold palette derivatives, R1 cover recipe (from docx skill).
const {
  Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, PageBreak,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableLayoutType, VerticalAlign,
} = require("docx");
const fs = require("fs");
const sizeOf = require("image-size");

// ---------- palette (IG-1 Ink Gold derivatives) ----------
const P = {
  bg: "1A1A1A", accent: "C9A84C",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "C9A84C", headerText: "1A1A1A", accentLine: "C9A84C", innerLine: "DDD5C0", surface: "F5F2E8" },
  body: "000000", primary: "1A1A1A", grey: "888888",
};
const FIGDIR = "/home/z/my-project/download/AURUM_v13_report/figures";

// ---------- text builders ----------
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    alignment: AlignmentType.CENTER,
    spacing: { before: 400, after: 180, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: "000000",
      font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 130, line: 312 },
    children: [new TextRun({ text, bold: true, size: 30, color: "000000",
      font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 220, after: 110, line: 312 },
    children: [new TextRun({ text, bold: true, size: 26, color: "000000",
      font: { ascii: "Times New Roman", eastAsia: "SimHei" } })],
  });
}
// English body: left-aligned, no first-line indent, 1.3x spacing, pure black
function body(text, opts = {}) {
  const runs = [];
  const segs = Array.isArray(text) ? text : [text];
  for (const s of segs) {
    if (typeof s === "string") runs.push(new TextRun({ text: s, size: 24, color: "000000",
      font: { ascii: "Times New Roman", eastAsia: "SimSun" } }));
    else runs.push(new TextRun({ text: s.t, bold: !!s.b, italics: !!s.i, size: 24,
      color: s.c || "000000", font: { ascii: "Times New Roman", eastAsia: "SimSun" } }));
  }
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { after: 120, line: 312 },
    children: runs,
  });
}
function bullet(text, level = 0) {
  const segs = Array.isArray(text) ? text : [text];
  const runs = segs.map(s => typeof s === "string"
    ? new TextRun({ text: s, size: 24, color: "000000", font: { ascii: "Times New Roman", eastAsia: "SimSun" } })
    : new TextRun({ text: s.t, bold: !!s.b, size: 24, color: s.c || "000000", font: { ascii: "Times New Roman", eastAsia: "SimSun" } }));
  return new Paragraph({ bullet: { level }, alignment: AlignmentType.LEFT,
    spacing: { after: 80, line: 312 }, children: runs });
}

// ---------- caption ----------
let figN = 0, tabN = 0;
function figCaption(text) {
  figN += 1;
  return new Paragraph({
    alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240, line: 312 },
    children: [new TextRun({ text: `Figure ${figN}  ${text}`, size: 21, color: "555555",
      font: { ascii: "Times New Roman", eastAsia: "SimSun" } })],
  });
}
function tabCaption(text) {
  tabN += 1;
  return new Paragraph({
    keepNext: true, alignment: AlignmentType.LEFT, spacing: { before: 160, after: 80, line: 312 },
    children: [new TextRun({ text: `Table ${tabN}  ${text}`, bold: true, size: 21, color: "333333",
      font: { ascii: "Times New Roman", eastAsia: "SimSun" } })],
  });
}

// ---------- figure embed (aspect ratio preserved via image-size) ----------
function figure(file, caption, displayWidth = 560) {
  const p = `${FIGDIR}/${file}`;
  const buf = fs.readFileSync(p);
  const dims = sizeOf.imageSize ? sizeOf.imageSize(buf) : sizeOf(buf);
  const h = Math.round(displayWidth * (dims.height / dims.width));
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 },
      children: [new ImageRun({ data: buf, transformation: { width: displayWidth, height: h }, type: "png" })],
    }),
    figCaption(caption),
  ];
}

// ---------- table builder (percentage widths, tableHeader, cantSplit) ----------
function cellPara(text, opts = {}) {
  return new Paragraph({
    alignment: opts.align || AlignmentType.LEFT,
    spacing: { line: 312 },
    children: [new TextRun({ text: String(text), bold: !!opts.bold, size: opts.size || 19,
      color: opts.color || "1A1A1A", font: { ascii: "Times New Roman", eastAsia: "SimSun" } })],
  });
}
function makeTable(headers, rows, widths, opts = {}) {
  const n = headers.length;
  const w = widths || Array(n).fill(Math.floor(100 / n));
  const headerRow = new TableRow({
    tableHeader: true, cantSplit: true,
    children: headers.map((htext, i) => new TableCell({
      children: [cellPara(htext, { bold: true, color: P.table.headerText, align: opts.headAlign || AlignmentType.LEFT, size: opts.hSize || 19 })],
      shading: { type: ShadingType.CLEAR, fill: P.table.headerBg },
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      width: { size: w[i], type: WidthType.PERCENTAGE },
      verticalAlign: VerticalAlign.CENTER,
    })),
  });
  const dataRows = rows.map((r, ri) => new TableRow({
    cantSplit: true,
    children: r.map((cell, i) => {
      const o = typeof cell === "object" && cell !== null ? cell : { t: cell };
      return new TableCell({
        children: [cellPara(o.t, { bold: !!o.b, align: o.align || (opts.aligns ? opts.aligns[i] : AlignmentType.LEFT), size: opts.size || 19, color: o.c })],
        shading: { type: ShadingType.CLEAR, fill: ri % 2 === 1 ? P.table.surface : "FFFFFF" },
        margins: { top: 45, bottom: 45, left: 90, right: 90 },
        width: { size: w[i], type: WidthType.PERCENTAGE },
        verticalAlign: VerticalAlign.CENTER,
      });
    }),
  }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: P.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: P.table.accentLine },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: P.table.innerLine },
      insideVertical: { style: BorderStyle.SINGLE, size: 2, color: "E8E2D4" },
    },
    rows: [headerRow, ...dataRows],
  });
}
const R = AlignmentType.RIGHT, C = AlignmentType.CENTER, L = AlignmentType.LEFT;

// ---------- cover machinery (from design-system.md, ASCII-width aware) ----------
function estWidth(text, pt) {
  let tw = 0;
  for (const ch of text) tw += /[\u4e00-\u9fff\u3000-\u303f]/.test(ch) ? pt * 20 : pt * 20 * 0.56;
  return tw;
}
function splitTitleLines(title, maxWidthTwips, pt) {
  const words = title.split(" ");
  const lines = []; let cur = "";
  for (const w of words) {
    const cand = cur ? cur + " " + w : w;
    if (estWidth(cand, pt) <= maxWidthTwips || !cur) cur = cand;
    else { lines.push(cur); cur = w; }
  }
  if (cur) lines.push(cur);
  if (lines.length > 1 && lines[lines.length - 1].length <= 3) {
    const last = lines.pop(); lines[lines.length - 1] += " " + last;
  }
  return lines;
}
function calcTitleLayout(title, maxWidthTwips, preferredPt = 40, minPt = 24) {
  let pt = preferredPt, lines;
  while (pt >= minPt) {
    lines = splitTitleLines(title, maxWidthTwips, pt);
    if (lines.length <= 3) break;
    pt -= 2;
  }
  if (!lines || lines.length > 3) { lines = splitTitleLines(title, maxWidthTwips, minPt); pt = minPt; }
  return { titlePt: pt, titleLines: lines };
}
function calcCoverSpacing(prm) {
  const { titleLineCount = 1, titlePt = 36, hasSubtitle = false, hasEnglishLabel = false,
    metaLineCount = 0, fixedHeight = 800, pageHeight = 16838 } = prm;
  const SAFETY = 1200;
  const usable = pageHeight - SAFETY;
  const titleH = titleLineCount * (titlePt * 23 + 200);
  const subH = hasSubtitle ? (12 * 23 + 600) + 620 : 0;
  const engH = hasEnglishLabel ? (9 * 23 + 600) : 0;
  const metaH = metaLineCount * (10 * 23 + 100);
  const implicit = 3 * 300;
  const content = titleH + subH + engH + metaH + fixedHeight + implicit;
  const remaining = Math.max(usable - content, 400);
  const FOOTER_MIN = 800;
  const rawTop = Math.floor(remaining * 0.45), rawBottom = Math.floor(remaining * 0.45);
  const bottomSpacing = Math.max(rawBottom, FOOTER_MIN);
  const topSpacing = Math.max(rawTop - Math.max(0, FOOTER_MIN - rawBottom), 400);
  const midSpacing = Math.max(remaining - topSpacing - bottomSpacing, 0);
  return { topSpacing, midSpacing, bottomSpacing };
}
const noBorders = {
  top: { style: BorderStyle.NONE }, bottom: { style: BorderStyle.NONE },
  left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
  insideHorizontal: { style: BorderStyle.NONE }, insideVertical: { style: BorderStyle.NONE },
};
const allNoBorders = noBorders;

function buildCoverR1(config) {
  const PA = config.palette;
  const padL = 1200, padR = 800;
  const availableWidth = 11906 - padL - padR - 300;
  const { titlePt, titleLines } = calcTitleLayout(config.title, availableWidth, 40, 24);
  const titleSize = titlePt * 2;
  const spacing = calcCoverSpacing({
    titleLineCount: titleLines.length, titlePt,
    hasSubtitle: !!config.subtitle, hasEnglishLabel: !!config.englishLabel,
    metaLineCount: (config.metaLines || []).length, fixedHeight: 400,
  });
  const accentLeft = { style: BorderStyle.SINGLE, size: 8, color: PA.accent, space: 12 };
  const children = [];
  children.push(new Paragraph({ spacing: { before: spacing.topSpacing } }));
  if (config.englishLabel) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 500 },
      border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: PA.accent, space: 8 } },
      children: [new TextRun({ text: config.englishLabel.split("").join("  "),
        size: 18, color: PA.accent, font: { ascii: "Calibri", eastAsia: "SimHei" }, characterSpacing: 40 })],
    }));
  }
  for (let i = 0; i < titleLines.length; i++) {
    children.push(new Paragraph({
      indent: { left: padL },
      spacing: { after: i < titleLines.length - 1 ? 100 : 300, line: Math.ceil(titlePt * 23), lineRule: "atLeast" },
      children: [new TextRun({ text: titleLines[i], size: titleSize, bold: true,
        color: PA.cover.titleColor, font: { eastAsia: "SimHei", ascii: "Arial" } })],
    }));
  }
  if (config.subtitle) {
    children.push(new Paragraph({
      indent: { left: padL, right: padR }, spacing: { after: 800, line: 300, lineRule: "atLeast" },
      children: [new TextRun({ text: config.subtitle, size: 24, color: PA.cover.subtitleColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  for (const line of (config.metaLines || [])) {
    children.push(new Paragraph({
      indent: { left: padL + 200 }, spacing: { after: 80 },
      border: { left: accentLeft },
      children: [new TextRun({ text: line, size: 24, color: PA.cover.metaColor,
        font: { eastAsia: "Microsoft YaHei", ascii: "Arial" } })],
    }));
  }
  children.push(new Paragraph({ spacing: { before: spacing.bottomSpacing } }));
  children.push(new Paragraph({
    indent: { left: padL, right: padR },
    border: { top: { style: BorderStyle.SINGLE, size: 2, color: PA.accent, space: 8 } },
    spacing: { before: 200 },
    children: [
      new TextRun({ text: config.footerLeft || "", size: 16, color: PA.cover.footerColor, font: { ascii: "Arial" } }),
      new TextRun({ text: "                                        " }),
      new TextRun({ text: config.footerRight || "", size: 16, color: PA.cover.footerColor, font: { ascii: "Arial" } }),
    ],
  }));
  return [new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    layout: TableLayoutType.FIXED,
    borders: allNoBorders,
    rows: [new TableRow({
      height: { value: 16838, rule: "exact" },
      children: [new TableCell({
        shading: { type: ShadingType.CLEAR, fill: PA.bg }, borders: noBorders,
        children,
      })],
    })],
  })];
}

module.exports = { P, h1, h2, h3, body, bullet, figure, makeTable, tabCaption,
  buildCoverR1, cellPara, R, C, L, figCaption };
