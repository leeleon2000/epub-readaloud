import copy

import pytest
from lxml import etree

from pipeline.displaycheck import DisplayMismatch, check
from pipeline.segment_core import segment_document, split_sentences
from pipeline.xhtml import XHTML_NS, body, parse_bytes, serialize

HEAD = ('<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">'
        '<head><title>t</title></head><body>')
TAIL = "</body></html>"


def run(body_xml: str):
    tree = parse_bytes((HEAD + body_xml + TAIL).encode())
    orig = copy.deepcopy(body(tree))
    res = segment_document(tree, "s")
    ids = {s.sid for s in res.sentences}
    check(orig, body(tree), ids, "test")
    return res, tree


def texts(res):
    return [s.text for s in res.sentences]


def inner(tree) -> str:
    s = etree.tostring(body(tree), encoding="unicode")
    return s.split(">", 1)[1].rsplit("<", 1)[0]


def test_simple_paragraph():
    res, tree = run("<p>Hello there. How are you?</p>")
    assert texts(res) == ["Hello there.", "How are you?"]
    assert inner(tree) == '<p><span id="s00001">Hello there.</span> <span id="s00002">How are you?</span></p>'


def test_boundary_inside_em_splits_element():
    res, tree = run('<p>He said <em class="x" id="e1">no. Then</em> he left.</p>')
    assert texts(res) == ["He said no.", "Then he left."]
    out = inner(tree)
    # The space between the sentences was inside <em>, so it stays in an <em> copy.
    assert out == ('<p><span id="s00001">He said <em class="x" id="e1">no.</em></span>'
                   '<em class="x"> </em>'
                   '<span id="s00002"><em class="x">Then</em> he left.</span></p>')


def test_abbreviation_and_ellipsis():
    res, _ = run("<p>Mr. Darcy bowed... She did not. It was 3.5 miles.</p>")
    assert texts(res)[0].startswith("Mr. Darcy")
    assert "It was 3.5 miles." in texts(res)


def test_br_and_noteref_not_spoken():
    res, _ = run('<p>Line one<br/>line two.<a epub:type="noteref" href="#n1">1</a> Next one.</p>'
                 '<aside epub:type="footnote" id="n1"><p>Note text.</p></aside>')
    assert len(res.sentences) == 2
    assert res.sentences[0].tts_text == "Line one line two."
    assert all("Note text" not in s.text for s in res.sentences)


def test_nested_quotes_dialogue():
    res, _ = run("<p>“Is it true?” she asked. “I don’t know,” he said.</p>")
    assert len(res.sentences) >= 2


def test_heading_kind_and_para_index():
    res, _ = run("<h1>Chapter <i>One</i></h1><p>First. Second.</p><p>Third.</p>")
    assert [(s.kind, s.para) for s in res.sentences] == [
        ("heading", 0), ("body", 1), ("body", 1), ("body", 2)]
    assert res.sentences[0].text == "Chapter One"


def test_mixed_block_and_text_in_div():
    res, _ = run("<div>Loose text here. <p>Inside p.</p>Tail text.</div>")
    assert texts(res) == ["Loose text here.", "Inside p.", "Tail text."]


def test_drop_cap_and_link_spanning_sentences():
    res, tree = run('<p><span class="dropcap">I</span>t began. <a href="x.html">See here. And</a> more.</p>')
    assert texts(res) == ["It began.", "See here.", "And more."]


def test_non_speech_paragraph_untouched():
    res, tree = run('<p>* * *</p><p><img src="a.png" alt="x"/></p>')
    assert res.sentences == []
    assert 'span' not in inner(tree)


def test_comment_and_zero_length_anchor_kept():
    res, tree = run('<p><a id="p12"/>One. <!-- c -->Two.<a id="end"/></p>')
    assert texts(res) == ["One.", "Two."]
    out = inner(tree)
    assert '<a id="p12"/>' in out and '<a id="end"/>' in out and "<!-- c -->" in out


def test_named_entities_parse():
    res, _ = run("<p>A&nbsp;B &mdash; C.</p>")
    assert texts(res) == ["A B — C."] or texts(res) == ["A B — C."]


def test_check_detects_change():
    tree = parse_bytes((HEAD + "<p>Hello.</p>" + TAIL).encode())
    orig = copy.deepcopy(body(tree))
    body(tree)[0].text = "Hullo."
    with pytest.raises(DisplayMismatch):
        check(orig, body(tree), set(), "t")


def test_split_sentences_offsets_raw_whitespace():
    raw = "  First   one.\n\n  Second\none.  "
    rs = split_sentences(raw)
    assert [raw[a:b] for a, b in rs] == ["First   one.", "Second\none."]


def test_touching_sentences_inside_link():
    # pySBD splits "Diminish?—Will" with no gap; must not create an empty <a> copy.
    res, tree = run('<p><a href="x"> CHAPTER 105. Does it\n Diminish?—Will He Perish? </a></p>')
    assert texts(res) == ["CHAPTER 105.", "Does it Diminish?", "—Will He Perish?"]
    assert '<a href="x"/>' not in inner(tree)


def test_comment_before_split_text():
    res, _ = run("<p>One.<!-- c --> Two. Three.</p>")
    assert texts(res) == ["One.", "Two.", "Three."]


def test_text_inside_an_illustration_caption_is_marked():
    """A caption is displayed but never read: Project Gutenberg's <div class="caption">, or <figcaption>."""
    res, _ = run('<div class="figcenter"><div class="caption"><p>“A line under the picture”</p>'
                 '<p>[<i>Copyright 1894 by the publisher.</i>]</p></div></div>'
                 '<p>The story goes on. “And so does the talk,” she said.</p>'
                 '<figure><img src="a.jpg" alt=""/><figcaption>The house in winter.</figcaption></figure>'
                 '<p class="captionless">An ordinary paragraph.</p>')
    kinds = {s.text: s.kind for s in res.sentences}
    assert kinds["“A line under the picture”"] == "caption"
    assert kinds["[Copyright 1894 by the publisher.]"] == "caption"
    assert kinds["The house in winter."] == "caption"
    assert kinds["The story goes on."] == "body" and kinds["An ordinary paragraph."] == "body"
