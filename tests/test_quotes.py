from pipeline.quotes import detect_style, split_paragraph


def pieces(texts, style="double"):
    return [[(p.text, p.is_quote, p.quote_id) for p in ps] for ps in split_paragraph(texts, style, "p")]


def test_quote_then_narration():
    assert pieces(["“There you are,” she would tell it, as if she meant it."]) == [[
        ("There you are,", True, "pq1"), ("she would tell it, as if she meant it.", False, None)]]


def test_narration_quote_narration():
    out = pieces(["“Well!” thought Alice to herself, “after such a fall as this!”"])[0]
    assert [(t, q) for t, q, _ in out] == [
        ("Well!", True), ("thought Alice to herself,", False), ("after such a fall as this!", True)]
    assert out[0][2] != out[2][2]


def test_quote_spans_sentences():
    out = pieces(["“I wonder how many miles I’ve fallen?", "Let me see.”", "She stopped."])
    assert out == [[("I wonder how many miles I’ve fallen?", True, "pq1")],
                   [("Let me see.", True, "pq1")],
                   [("She stopped.", False, None)]]


def test_label_in_quotes_and_punctuation_only_pieces_dropped():
    out = pieces(["It was labelled “ORANGE MARMALADE”, but it was empty."])[0]
    assert [(t, q) for t, q, _ in out] == [("It was labelled", False), ("ORANGE MARMALADE", True),
                                           (", but it was empty.", False)]


def test_single_quote_style_ignores_apostrophes():
    texts = ["‘I don’t know,’ said Dinah’s owner. ‘It’s late.’"]
    assert detect_style(texts) == "single"
    out = pieces(texts, "single")[0]
    assert [(t, q) for t, q, _ in out] == [("I don’t know,", True), ("said Dinah’s owner.", False),
                                           ("It’s late.", True)]


def test_straight_double_quotes():
    out = pieces(['"Hi," he said. "Bye."'])[0]
    assert [(t, q) for t, q, _ in out] == [("Hi,", True), ("he said.", False), ("Bye.", True)]
