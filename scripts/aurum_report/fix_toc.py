#!/usr/bin/env python3
"""Surgical TOC fix for add_toc_placeholders.py output:
1) The first TOC1 paragraph carries the TOC field opener (begin/instrText/separate)
   AND a duplicate hyperlink entry (empty PAGEREF). Strip the hyperlink, keep
   the field-control runs so Word can still regenerate the TOC on update.
2) Compact TOC1/2/3 styles (spacing after=0, line=240) so 54 entries fit
   without pushing the section-break paragraph onto an orphan page."""
import re, sys, zipfile, shutil

path = sys.argv[1]
tmp = path + ".tmp"

with zipfile.ZipFile(path, "r") as zin:
    names = zin.namelist()
    data = {n: zin.read(n) for n in names}

doc = data["word/document.xml"].decode("utf-8")

# ── 1. Strip duplicate hyperlink from the field-opener paragraph ──
m = re.search(
    r'(<w:p><w:pPr><w:pStyle w:val="TOC1"/>.*?'
    r'<w:fldChar w:fldCharType="separate"/></w:r>)'
    r'(<w:hyperlink[^>]*>.*?</w:hyperlink>)'
    r'(</w:p>)',
    doc, flags=re.S)
if m and 'TOC \\h' in m.group(0):
    doc = doc[:m.start(2)] + doc[m.end(2):]
    print("Stripped duplicate hyperlink from TOC field-opener paragraph (field controls kept)")
else:
    print("WARNING: field-opener duplicate pattern not found — no change")

data["word/document.xml"] = doc.encode("utf-8")

# ── 2. Compact TOC styles ──
styles = data["word/styles.xml"].decode("utf-8")
changed = 0
for sid in ("TOC1", "TOC2", "TOC3"):
    new, n = re.subn(
        r'(<w:style [^>]*w:styleId="%s"[^>]*>.*?<w:spacing )w:after="100"(/>)' % sid,
        r'\1w:after="0" w:line="240" w:lineRule="auto"\2',
        styles, flags=re.S)
    if n:
        styles = new
        changed += n
data["word/styles.xml"] = styles.encode("utf-8")
print(f"Compacted {changed} TOC styles (after=0, line=240)")

with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])
shutil.move(tmp, path)

# ── Verify field integrity ──
with zipfile.ZipFile(path, "r") as z:
    d = z.read("word/document.xml").decode("utf-8")
seq = [mm.group(1) for mm in re.finditer(r'w:fldCharType="(begin|separate|end)"', d)]
n_toc1 = len(re.findall(r'<w:pStyle w:val="TOC1"', d))
n_toc2 = len(re.findall(r'<w:pStyle w:val="TOC2"', d))
es = len(re.findall(r'Executive Summary', d))
print(f"fldChar sequence: {seq}")
print(f"TOC1={n_toc1} TOC2={n_toc2} 'Executive Summary' x{es}")
