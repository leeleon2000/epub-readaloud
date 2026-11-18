"""EPUB 2 -> 3 metadata upgrade edge cases."""
from pathlib import Path

from pipeline import epub

OPF = """<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="bookid" version="2.0">
  <metadata>
    <dc:creator xmlns:ns1="http://www.idpf.org/2007/opf" role="aut" ns1:file-as="Example, Jane">Jane Example</dc:creator>
    <dc:identifier id="bookid">urn:uuid:3da4e858</dc:identifier>
    <dc:title>T</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest><item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/></manifest>
  <spine><itemref idref="c1"/></spine>
  <tours>
    </tours>
</package>"""


def test_unprefixed_role_on_creator_is_converted(tmp_path: Path):
    """One commercial EPUB writes role="aut" without opf: (invalid in EPUB 2 and 3); epubcheck flagged it in our output."""
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    (tmp_path / "content.opf").write_text(OPF)
    (tmp_path / "c1.xhtml").write_text('<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                                       '<head><title>t</title></head><body><p>Hi.</p></body></html>')
    pkg = epub.open_package(tmp_path)
    epub.upgrade_to_epub3(pkg)
    opf = (tmp_path / "content.opf").read_text()
    assert ' role="aut"' not in opf
    assert 'property="role" scheme="marc:relators">aut</meta>' in opf
    assert 'property="file-as">Example, Jane</meta>' in opf
    assert "<tours" not in opf                                    # an EPUB 2 only element


def test_zip_skips_appledouble_files(tmp_path: Path):
    """macOS tar adds ._name files when a book dir is copied to Linux; they ended up in three EPUBs (2026-11-16)."""
    import zipfile
    root = tmp_path / "build"
    (root / "OEBPS").mkdir(parents=True)
    (root / "mimetype").write_text("application/epub+zip")
    (root / "OEBPS" / "content.opf").write_text("<package/>")
    (root / "OEBPS" / "._content.opf").write_bytes(b"\x00\x05\x16\x07")
    (root / "._OEBPS").write_bytes(b"\x00")
    (root / ".DS_Store").write_bytes(b"\x00")
    epub.zip_epub(root, tmp_path / "x.epub")
    names = zipfile.ZipFile(tmp_path / "x.epub").namelist()
    assert names == ["mimetype", "OEBPS/content.opf"]


def test_html5_fixups_for_img_sizes_and_content_type_meta():
    """Two commercial EPUBs (2026-11-17): percent image sizes and odd content-type metas."""
    from lxml import etree
    from pipeline.epub import _html5_fixups
    src = ('<html xmlns="http://www.w3.org/1999/xhtml"><head>'
           '<meta http-equiv="Content-Type" content="http://www.w3.org/1999/xhtml; charset=utf-8"/></head><body>'
           '<p><img src="a.jpg" width="20%" alt=""/><img src="b.jpg" width="300px" height="10" alt=""/>'
           '<img src="c.jpg" style="border:0;" height="50%" alt=""/>Text.</p></body></html>')
    tree = etree.ElementTree(etree.fromstring(src))
    _html5_fixups(tree)
    ns = "{http://www.w3.org/1999/xhtml}"
    meta = tree.getroot().find(f"{ns}head/{ns}meta")
    assert meta.get("content") == "text/html; charset=utf-8"
    a, b, c = tree.getroot().iter(f"{ns}img")
    assert a.get("width") is None and a.get("style") == "width:20%"
    assert b.get("width") == "300" and b.get("height") == "10"
    assert c.get("height") is None and c.get("style") == "border:0; height:50%"


def test_empty_dc_elements_are_removed(tmp_path: Path):
    (tmp_path / "META-INF").mkdir()
    (tmp_path / "META-INF" / "container.xml").write_text(
        '<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
        '<rootfiles><rootfile full-path="content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>')
    (tmp_path / "content.opf").write_text(OPF.replace("<dc:title>T</dc:title>", "<dc:title>T</dc:title><dc:subject/><dc:format></dc:format>"))
    (tmp_path / "c1.xhtml").write_text('<?xml version="1.0"?><html xmlns="http://www.w3.org/1999/xhtml">'
                                       '<head><title>t</title></head><body><p>Hi.</p></body></html>')
    pkg = epub.open_package(tmp_path)
    epub.upgrade_to_epub3(pkg)
    opf = (tmp_path / "content.opf").read_text()
    assert "dc:subject" not in opf and "dc:format" not in opf and "<dc:title>T</dc:title>" in opf


def test_display_check_accepts_img_size_moved_to_style():
    """2026-11-17: the img width fixup must not trip the display check, a real change must."""
    import copy
    import pytest
    from lxml import etree
    from pipeline.displaycheck import DisplayMismatch, check
    from pipeline.epub import _html5_fixups
    src = ('<html xmlns="http://www.w3.org/1999/xhtml"><head/><body><p>A '
           '<img src="a.jpg" width="20%" alt=""/><img src="b.jpg" width="300px" alt=""/> B.</p></body></html>')
    tree = etree.ElementTree(etree.fromstring(src))
    body = tree.getroot().find("{http://www.w3.org/1999/xhtml}body")
    orig = copy.deepcopy(body)
    _html5_fixups(tree)
    check(orig, body, set(), "upgrade")                     # same rendering
    body.find(".//{http://www.w3.org/1999/xhtml}img").set("style", "width:50%")
    with pytest.raises(DisplayMismatch):
        check(orig, body, set(), "upgrade")
