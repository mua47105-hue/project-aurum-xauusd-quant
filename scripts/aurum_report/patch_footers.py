#!/usr/bin/env python3
"""Post-process AURUM docx: patch footer PAGE fields with explicit format
switches (\\* ROMAN / \\* arabic) per section, and strip empty <w:pgNumType/>.
Follows docx skill references/toc.md mandatory post-processing."""
import re, sys, zipfile, shutil, os

path = sys.argv[1]
tmp = path + ".tmp"

with zipfile.ZipFile(path, "r") as zin:
    names = zin.namelist()
    data = {n: zin.read(n) for n in names}

doc = data["word/document.xml"].decode("utf-8")

# 1. Remove empty pgNumType emitted for the cover section
doc = doc.replace("<w:pgNumType/>", "")

# 2. Map sections -> footer rIds by pgNumType format
rels = data["word/_rels/document.xml.rels"].decode("utf-8")
rid_to_target = dict(re.findall(r'Id="(rId\d+)"[^>]*Target="([^"]+)"', rels))

sect_blocks = re.findall(r"<w:sectPr[^>]*>.*?</w:sectPr>", doc, flags=re.S)
roman_footers, arabic_footers = set(), set()
for blk in sect_blocks:
    fmt = None
    m = re.search(r'<w:pgNumType[^>]*w:fmt="([^"]+)"', blk)
    if m:
        fmt = m.group(1)
    for rid in re.findall(r'<w:footerReference[^>]*r:id="(rId\d+)"', blk):
        target = rid_to_target.get(rid, "")
        fname = "word/" + target.lstrip("/") if not target.startswith("word/") else target
        if fmt == "upperRoman":
            roman_footers.add(fname)
        elif fmt in ("decimal", "lowerRoman", "arabic"):
            arabic_footers.add(fname)

def patch_footer(xml, switch):
    # Replace bare PAGE instrText with explicit format switch
    return re.sub(
        r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
        r"\1 PAGE \\* " + switch + r" \\* MERGEFORMAT \2",
        xml,
    )

changed = []
for fname in roman_footers | arabic_footers:
    if fname in data:
        xml = data[fname].decode("utf-8")
        if "PAGE" in xml:
            sw = "ROMAN" if fname in roman_footers else "arabic"
            new = patch_footer(xml, sw)
            if new != xml:
                data[fname] = new.encode("utf-8")
                changed.append((fname, sw))

data["word/document.xml"] = doc.encode("utf-8")

with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])
shutil.move(tmp, path)
print("Patched footers:", changed if changed else "none found")
print("Removed empty pgNumType; sections analyzed:", len(sect_blocks))
