#!/usr/bin/env python3
"""AURUM Spec v2.0 report postcheck: 0 errors, 0 warnings target.
Checks: docx structure (headings/tables/images), placeholder leftovers,
figure completeness, table integrity, PDF conversion, page count, blank pages.
"""
import json
import re
import subprocess
import sys

from docx import Document

DOCX = "/home/z/my-project/download/AURUM_Spec_v2_Master_Execution_and_Tournament_Report.docx"
PDF = DOCX.replace(".docx", ".pdf")
FIGS_DIR = "/home/z/my-project/download/v2_figures"

errors, warnings = [], []

# ---------------------------------------------------------------- docx checks
doc = Document(DOCX)
paras = doc.paragraphs
tables = doc.tables
texts = [p.text for p in paras]
full_text = "\n".join(texts)
for t in tables:
    for row in t.rows:
        for cell in row.cells:
            full_text += "\n" + cell.text

h1s = [p.text for p in paras if p.style.name == "Heading 1"]
h2s = [p.text for p in paras if p.style.name == "Heading 2"]
imgs = len(doc.inline_shapes)
print(f"structure: {len(paras)} paragraphs, {len(tables)} tables, {imgs} inline images")
print(f"headings: {len(h1s)} H1, {len(h2s)} H2")

# expected sections
expected_h1 = ["1. Executive Summary", "2. The Unified v2.0 Codebase",
               "3. Phase 1", "4. Phase 2", "5. Phase 3",
               "6. Engineering Flaw #8", "7. Phase 4", "8. Phase 5",
               "9. The Verdict", "10. Statistical Honesty", "11. Appendices"]
for e in expected_h1:
    if not any(e in h for h in h1s):
        errors.append(f"missing H1 section: {e}")

# figures: 10 expected
if imgs != 11:
    errors.append(f"expected 11 inline images, found {imgs}")
import os
fig_files = sorted(os.listdir(FIGS_DIR)) if os.path.isdir(FIGS_DIR) else []
if len(fig_files) != 11:
    errors.append(f"expected 10 fig files in v2_figures, found {len(fig_files)}")

# placeholders / leftovers (whole-text patterns; JSON-leak markers are
# additionally checked as EXACT table-cell values below -- prose usage such as
# "null hypothesis" or "blank by design" is legitimate statistics English)
bad_patterns = [
    (r"\bTODO\b", "TODO"), (r"\bXXX\b", "XXX"), (r"待补充", "待补充"),
    (r"【[^】]*】", "【】placeholder"), (r"\bundefined\b", "undefined"),
    (r"\[figure .* not found\]", "missing figure"),
    (r"__", "dunder leftover"),
    (r"\|\s*---", "markdown table"), (r"\*\*", "markdown bold"),
]
for pat, name in bad_patterns:
    hits = re.findall(pat, full_text)
    if hits:
        errors.append(f"placeholder/leftover '{name}': {len(hits)} occurrence(s)")

# JSON-leak markers: exact table-cell values (where generator leaks would land)
for ti, t in enumerate(tables):
    for row in t.rows:
        for cell in row.cells:
            v = cell.text.strip()
            if v in ("null", "NaN", "undefined", "None"):
                errors.append(f"table {ti}: JSON-leak cell value '{v}'")

# empty table cells / headers
for ti, t in enumerate(tables):
    if not t.rows:
        errors.append(f"table {ti}: no rows")
        continue
    hdr = [c.text.strip() for c in t.rows[0].cells]
    if any(not h for h in hdr):
        errors.append(f"table {ti}: empty header cell")
    empty_cells = sum(1 for row in t.rows[1:] for c in row.cells if not c.text.strip())
    if empty_cells:
        warnings.append(f"table {ti} ('{hdr[0][:24]}...'): {empty_cells} empty cell(s)")

# duplicate identical adjacent paragraphs (copy bugs)
dup = 0
for a, b in zip(texts, texts[1:]):
    if a and a == b:
        dup += 1
if dup:
    warnings.append(f"{dup} duplicate adjacent paragraph(s)")

# TOC field present
xml = doc.element.xml
if "TOC" not in xml and "Contents" not in full_text:
    errors.append("no TOC field/heading found")

# ---------------------------------------------------------------- pdf checks
r = subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir",
                    "/home/z/my-project/download", DOCX],
                   capture_output=True, text=True, timeout=300)
if not os.path.exists(PDF):
    errors.append(f"PDF conversion failed: {r.stderr[:200]}")
else:
    try:
        from pypdf import PdfReader
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pypdf"], check=True)
        from pypdf import PdfReader
    reader = PdfReader(PDF)
    npages = len(reader.pages)
    print(f"pdf: {npages} pages, {os.path.getsize(PDF)//1024} KB")
    if npages < 15:
        warnings.append(f"PDF only {npages} pages (expected a fuller report)")
    # blank page detection via text + render
    import glob
    subprocess.run(["rm", "-rf", "/tmp/render"], check=True)
    subprocess.run(["mkdir", "-p", "/tmp/render"], check=True)
    subprocess.run(["pdftoppm", "-r", "50", "-jpeg", "-jpegopt", "quality=50",
                    PDF, "/tmp/render/page"], check=True, timeout=300)
    from PIL import Image
    import numpy as np
    pages = sorted(glob.glob("/tmp/render/page-*.jpg"))
    blank = []
    for pg in pages:
        img = np.asarray(Image.open(pg).convert("L"))
        nonwhite = (img < 245).mean()
        if nonwhite < 0.005:
            blank.append(os.path.basename(pg))
    if blank:
        errors.append(f"blank/near-blank page(s): {blank}")

# ---------------------------------------------------------------- verdict
print()
print(f"CHECKS COMPLETE: {len(errors)} error(s), {len(warnings)} warning(s)")
for e in errors:
    print("  ERROR:", e)
for w in warnings:
    print("  WARN :", w)
sys.exit(1 if (errors or warnings) else 0)
