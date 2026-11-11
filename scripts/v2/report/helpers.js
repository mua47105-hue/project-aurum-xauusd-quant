"use strict";
// ============================================================================
// AURUM Spec v2.0 Master Report — shared helpers (docx design system)
// ============================================================================
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, PageBreak, Header, Footer, PageNumber, NumberFormat,
  TabStopType, LeaderType,
  AlignmentType, HeadingLevel, WidthType, BorderStyle, ShadingType,
  TableLayoutType, SectionType, TableOfContents, VerticalAlign,
} = require("docx");
const fs = require("fs");
const path = require("path");
const sizeOf = require("image-size");

const ROOT = "/home/z/my-project";
const K1 = path.join(ROOT, "kaggle_work/v2/k1_out4");
const K2 = path.join(ROOT, "kaggle_work/v2/k2_out3");
const K3 = path.join(ROOT, "kaggle_work/v2/k3_out2");
const K6 = path.join(ROOT, "kaggle_work/v2/k6_out1");
const K7 = path.join(ROOT, "kaggle_work/v2/k7_out1");
const FIGS = path.join(ROOT, "download/aurum_v2_report/figs");
const OUTDIR = path.join(ROOT, "download");
fs.mkdirSync(OUTDIR, { recursive: true });

// ---------------------------------------------------------------- palette
const P = {
  bg: "1A1A1A", accent: "C9A84C",
  cover: { titleColor: "FFFFFF", subtitleColor: "B0B8C0", metaColor: "90989F", footerColor: "687078" },
  table: { headerBg: "C9A84C", headerText: "1A1A1A", accentLine: "C9A84C", innerLine: "DDD5C0", surface: "F5F2E8" },
  primary: "1A1A1A", body: "222222", secondary: "555555",
  crimson: "922B21", green: "1E8449",
};
const FONTS = { ascii: "Times New Roman", eastAsia: "Times New Roman" };
const HFONTS = { ascii: "Arial", eastAsia: "Arial" };

// ---------------------------------------------------------------- utils
const jload = (p, d) => { try { return JSON.parse(fs.readFileSync(p, "utf8")); } catch (e) { return d; } };
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : Number(v).toFixed(d);
const safe = (v, ph) => (v === undefined || v === null || v === "" || String(v) === "NaN") ? (ph || "—") : String(v);

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160, line: 312 },
    children: [new TextRun({ text, bold: true, size: 32, color: P.primary, font: HFONTS })] });
}
function h2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 260, after: 120, line: 312 },
    children: [new TextRun({ text, bold: true, size: 26, color: P.primary, font: HFONTS })] });
}
function h3(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_3, spacing: { before: 200, after: 100, line: 300 },
    children: [new TextRun({ text, bold: true, size: 23, color: P.secondary, font: HFONTS })] });
}
function body(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 100 },
    children: [new TextRun({ text, size: 22, color: P.body, font: FONTS, ...opts })] });
}
function bodyRuns(runs) {
  return new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { line: 312, after: 100 },
    children: runs.map(r => new TextRun({ size: 22, color: P.body, font: FONTS, ...r })) });
}
function bullet(text, opts = {}) {
  return new Paragraph({ alignment: AlignmentType.LEFT, spacing: { line: 300, after: 60 },
    indent: { left: 360, hanging: 220 },
    children: [new TextRun({ text: "•  ", size: 22, color: P.accent, font: FONTS, bold: true }),
      new TextRun({ text, size: 22, color: P.body, font: FONTS, ...opts })] });
}
function caption(text) {
  return new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 220, line: 280 },
    children: [new TextRun({ text, size: 18, italics: true, color: P.secondary, font: FONTS })] });
}
function figure(figFile, capText, widthPt = 470) {
  const fp = path.join(FIGS, figFile);
  if (!fs.existsSync(fp)) return [body(`[figure ${figFile} not found]`)];
  const buf = fs.readFileSync(fp);
  const dim = sizeOf.imageSize ? sizeOf.imageSize(buf) : sizeOf(buf);
  const w = widthPt, h = Math.round(w * dim.height / dim.width);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
      children: [new ImageRun({ data: buf, transformation: { width: w, height: h }, type: "png" })] }),
    caption(capText),
  ];
}
function mkTable(headers, rows, widths, opts = {}) {
  const hSize = opts.hSize || 18, cSize = opts.cSize || 18;
  const headerRow = new TableRow({ tableHeader: true, cantSplit: true,
    children: headers.map((t, i) => new TableCell({
      shading: { type: ShadingType.CLEAR, fill: P.table.headerBg },
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      width: { size: widths[i], type: WidthType.PERCENTAGE },
      verticalAlign: VerticalAlign.CENTER,
      children: [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { line: 240 },
        children: [new TextRun({ text: t, bold: true, size: hSize, color: P.table.headerText, font: HFONTS })] })],
    })) });
  const dataRows = rows.map((r, ri) => new TableRow({ cantSplit: true,
    children: r.map((cell, i) => {
      const isObj = typeof cell === "object" && cell !== null;
      const text = isObj ? cell.t : String(cell);
      return new TableCell({
        shading: (ri % 2 === 1) ? { type: ShadingType.CLEAR, fill: P.table.surface } : undefined,
        margins: { top: 45, bottom: 45, left: 90, right: 90 },
        width: { size: widths[i], type: WidthType.PERCENTAGE },
        verticalAlign: VerticalAlign.CENTER,
        children: [new Paragraph({ alignment: (isObj && cell.left) ? AlignmentType.LEFT : AlignmentType.CENTER,
          spacing: { line: 240 },
          children: [new TextRun({ text: safe(text), size: cSize, bold: isObj && cell.b,
            color: (isObj && cell.c) || P.body, font: FONTS })] })],
      });
    }) }));
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE }, layout: TableLayoutType.FIXED,
    borders: { top: { style: BorderStyle.SINGLE, size: 6, color: P.table.accentLine },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: P.table.accentLine },
      left: { style: BorderStyle.NONE }, right: { style: BorderStyle.NONE },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: P.table.innerLine },
      insideVertical: { style: BorderStyle.NONE } },
    rows: [headerRow, ...dataRows],
  });
}
function tableTitle(text) {
  return new Paragraph({ keepNext: true, spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, bold: true, size: 19, color: P.primary, font: HFONTS })] });
}
function spacer(h = 80) { return new Paragraph({ spacing: { after: h } }); }

// ---------------------------------------------------------------- data
function loadData() {
  const D = {
    k1meta: jload(path.join(K1, "feature_meta.json"), {}),
    k1fam: (function () { try {
      return fs.readFileSync(path.join(K1, "ic_family_summary.csv"), "utf8")
        .trim().split("\n").slice(1).map(l => {
          const p = l.split(",");
          return { family: p[0], n: +p[1], best_abs_t: +p[2] };
        });
    } catch (e) { return []; } })(),
    k2engine: jload(path.join(K2, "engine_metrics.json"), {}),
    k2meta: jload(path.join(K2, "k2_meta.json"), {}),
    k3meta: jload(path.join(K3, "k3_meta.json"), {}),
    k3rho: jload(path.join(K3, "rho_log.json"), []),
    k6bt: jload(path.join(K6, "backtest_v2.json"), { strategies: {} }),
    k6stats: jload(path.join(K6, "stats.json"), {}),
    k6cc: jload(path.join(K6, "champion_compare.json"), {}),
    k6meta: jload(path.join(K6, "k6_meta.json"), {}),
    k6zq: jload(path.join(K6, "z_quality.json"), { audit: {}, flagged_strategies: [] }),
    k6gates: jload(path.join(K6, "gates.json"), {}),
    k7decision: jload(path.join(K7, "decision_v2.json"), {}),
    k7gates: jload(path.join(K7, "gates.json"), {}),
  };
  D.k6year = [];
  try {
    D.k6year = fs.readFileSync(path.join(K6, "year_table.csv"), "utf8")
      .trim().split("\n").slice(1).map(l => {
        const p = l.split(",");
        return { strategy: p[0], year: +p[1], window: p[2],
          sharpe: +p[3], ret: +p[4], maxdd: +p[5] };
      });
  } catch (e) {}
  return D;
}

module.exports = {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  PageBreak, Header, Footer, PageNumber, NumberFormat, AlignmentType,
  TabStopType, LeaderType,
  HeadingLevel, WidthType, BorderStyle, ShadingType, TableLayoutType,
  SectionType, TableOfContents, VerticalAlign,
  fs, path,
  ROOT, K1, K2, K3, K6, K7, FIGS, OUTDIR,
  P, FONTS, HFONTS, jload, fmt, safe,
  h1, h2, h3, body, bodyRuns, bullet, caption, figure, mkTable, tableTitle, spacer,
  loadData,
};
