"""Post-process the AURUM docx: patch footer PAGE fields (WPS compat), strip empty pgNumType,
and remove docx-js's inline first TOC entry (which duplicates the script-injected placeholder)."""
import re
import shutil
import sys
import zipfile
from io import BytesIO

from lxml import etree

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

def fix_toc_field_paragraph(doc_xml: bytes) -> bytes:
    """docx-js TableOfContents(hyperlink:true) emits its first placeholder entry INSIDE the
    field-start paragraph (after the separate fldChar run). add_toc_placeholders.py cannot
    remove it (it only clears BETWEEN separate-para and end-para), producing a duplicate
    first line. Strip everything after the separate run in that paragraph."""
    root = etree.fromstring(doc_xml)
    for instr in root.iter(f"{W}instrText"):
        if instr.text and "TOC" in instr.text and "PAGEREF" not in instr.text:
            para = instr.getparent().getparent()
            if para.tag != f"{W}p":
                continue
            children = list(para)
            sep_run = None
            for child in children:
                if child.tag == f"{W}r" and any(
                    fc.get(f"{W}fldCharType") == "separate" for fc in child.findall(f"{W}fldChar")
                ):
                    sep_run = child
                    break
            if sep_run is not None:
                idx = children.index(sep_run)
                removed = 0
                for child in children[idx + 1:]:
                    if child.tag == f"{W}hyperlink" or child.tag == f"{W}r":
                        para.remove(child)
                        removed += 1
                if removed:
                    print(f"stripped {removed} inline TOC-entry runs from field-start paragraph")
            break
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)

path = sys.argv[1] if len(sys.argv) > 1 else "/home/z/my-project/download/Project_AURUM_Deep_Research_Report.docx"
tmp = path + ".tmp"

zin = zipfile.ZipFile(path, "r")
zout = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)

for item in zin.infolist():
    data = zin.read(item.filename)
    if item.filename.startswith("word/footer") and item.filename.endswith(".xml"):
        xml = data.decode("utf-8")
        if "7F7F7F" in xml:  # front-matter marker -> Roman
            xml = re.sub(r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
                         r"\1 PAGE \\* ROMAN \\* MERGEFORMAT \2", xml)
            xml = xml.replace("7F7F7F", "808080")
            print(f"patched {item.filename}: ROMAN")
        elif "808080" in xml and "PAGE" in xml:  # body -> arabic
            xml = re.sub(r"(<w:instrText[^>]*>)\s*PAGE\s*(</w:instrText>)",
                         r"\1 PAGE \\* arabic \\* MERGEFORMAT \2", xml)
            print(f"patched {item.filename}: arabic")
        data = xml.encode("utf-8")
    elif item.filename == "word/document.xml":
        xml = data.decode("utf-8")
        n = xml.count("<w:pgNumType/>")
        xml = xml.replace("<w:pgNumType/>", "")
        if n:
            print(f"removed {n} empty pgNumType from document.xml")
        data = fix_toc_field_paragraph(xml.encode("utf-8"))
    zout.writestr(item, data)

zin.close()
zout.close()
shutil.move(tmp, path)
print("post-process complete:", path)
