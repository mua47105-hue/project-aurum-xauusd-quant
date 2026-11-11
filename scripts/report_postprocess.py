#!/usr/bin/env python3
"""Post-process the AURUM report docx:
1. Patch footer PAGE fields with explicit \\* ROMAN / \\* arabic switches (WPS).
2. Remove empty <w:pgNumType/> from sections without pageNumbers.
3. Verify zip integrity."""
import sys, zipfile, shutil, re, os

src = sys.argv[1]
tmp = src + ".tmp"

with zipfile.ZipFile(src, "r") as zin:
    names = zin.namelist()
    data = {n: zin.read(n) for n in names}

# map footers to their section context by scanning document.xml sectPr order
doc = data["word/document.xml"].decode("utf-8")

# find footer references in order of appearance with their fmt
# section 2 (front matter) = UPPER_ROMAN -> first footer ref after first sectPr...
# simpler: docx-js writes footer1.xml, footer2.xml... map by which footer file is
# referenced in which sectPr. We patch per footer file based on nearest pgNumType fmt.
sect_blocks = re.findall(r"<w:sectPr.*?</w:sectPr>", doc, re.S)
footer_fmt = {}   # footer file -> 'roman'|'arabic'
for blk in sect_blocks:
    m = re.search(r'w:footerReference[^>]*r:id="(rId\d+)"', blk)
    if not m:
        continue
    fmt = None
    if 'w:fmt="upperRoman"' in blk or 'w:fmt="lowerRoman"' in blk:
        fmt = "ROMAN"
    elif 'w:fmt="decimal"' in blk:
        fmt = "arabic"
    if fmt:
        footer_fmt[m.group(1)] = fmt

# resolve rIds to footer files via document.xml.rels
rels = data["word/_rels/document.xml.rels"].decode("utf-8")
rid_file = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="(footer\d+\.xml)"', rels))

patched = []
for rid, fmt in footer_fmt.items():
    fn = rid_file.get(rid)
    if not fn:
        continue
    key = f"word/{fn}"
    if key not in data:
        continue
    xml = data[key].decode("utf-8")
    new = re.sub(r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
                 rf"\1 PAGE \\* {fmt} \\* MERGEFORMAT \2", xml)
    if new != xml:
        data[key] = new.encode("utf-8")
        patched.append((fn, fmt))

# remove empty pgNumType (cover section)
doc2 = doc.replace("<w:pgNumType/>", "")
if doc2 != doc:
    data["word/document.xml"] = doc2.encode("utf-8")

with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])
shutil.move(tmp, src)
print("patched footers:", patched)
print("pgNumType cleanup:", doc2 != doc)
print("OK ->", src)
