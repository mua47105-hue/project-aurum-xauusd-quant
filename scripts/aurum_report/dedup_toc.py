#!/usr/bin/env python3
"""Remove duplicated first TOC placeholder entry: a TOC-styled paragraph
whose text duplicates the next TOC-styled paragraph but lacks the PAGEREF
field. Deterministic post-fix for add_toc_placeholders.py output."""
import re, sys, zipfile, shutil

path = sys.argv[1]
tmp = path + ".tmp"

with zipfile.ZipFile(path, "r") as zin:
    names = zin.namelist()
    data = {n: zin.read(n) for n in names}

xml = data["word/document.xml"].decode("utf-8")

# All paragraphs with their style
para_re = re.compile(r"<w:p\b[^>]*>.*?</w:p>", flags=re.S)
paras = list(para_re.finditer(xml))

def info(m):
    seg = m.group(0)
    st = re.search(r'<w:pStyle w:val="(TOC\d)"', seg)
    text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", seg))
    return (st.group(1) if st else None, text, "PAGEREF" in seg)

toc = [(m.start(), m.end()) + info(m) for m in paras]
toc = [t for t in toc if t[2]]  # only TOC-styled

to_delete = []
for i in range(len(toc) - 1):
    cur, nxt = toc[i], toc[i + 1]
    # Duplicate pattern (add_toc_placeholders quirk): first entry emitted twice —
    # once as a HYPERLINK+PAGEREF field paragraph (text has NO page number),
    # then again as a plain placeholder WITH the literal page number.
    # Delete the field-paragraph copy; keep the plain one (consistent with the
    # other 53 entries). Word regenerates everything on field update anyway.
    if (cur[2] == nxt[2] and cur[4] and not nxt[4]
            and cur[3].strip() and nxt[3].startswith(cur[3].strip())
            and len(cur[3].strip()) >= 5):
        to_delete.append((cur[0], cur[1]))

if to_delete:
    for start, end in sorted(to_delete, reverse=True):
        xml = xml[:start] + xml[end:]
    print(f"Removed {len(to_delete)} duplicated TOC placeholder paragraph(s)")
else:
    print("No duplicated TOC entries found")

data["word/document.xml"] = xml.encode("utf-8")
with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
    for n in names:
        zout.writestr(n, data[n])
shutil.move(tmp, path)
