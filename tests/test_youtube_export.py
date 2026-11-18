"""scripts/youtube_export.py: where a long book is cut into parts, and the timestamp formats."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("youtube_export", Path(__file__).parents[1] / "scripts" / "youtube_export.py")
ye = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ye)
H = 3_600_000


def test_parts_are_cut_at_chapter_headings():
    marks = [(i * H, f"CHAPTER {i + 1}") for i in range(10)]              # ten one-hour chapters
    assert ye.split_points(marks, 10 * H, 12 * H, None) == [0, 10 * H]    # fits in one part
    assert ye.split_points(marks, 10 * H, 4 * H, None) == [0, 3 * H, 7 * H, 10 * H]
    assert ye.split_points(marks, 10 * H, 12 * H, 2) == [0, 5 * H, 10 * H]
    with pytest.raises(SystemExit):                                       # two parts of at most 4 h cannot hold 10 h
        ye.split_points(marks, 10 * H, 4 * H, 2)


def test_parts_can_start_at_named_headings():
    roman = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    marks = [(i * H, f"CHAPTER {r}. In which something happens") for i, r in enumerate(roman)]
    assert ye.cuts_at(marks, 10 * H, 12 * H, ["chapter iv.", "CHAPTER  IX"]) == [0, 3 * H, 8 * H, 10 * H]
    assert ye.cuts_at(marks, 10 * H, 12 * H, ["CHAPTER V"]) == [0, 4 * H, 10 * H]       # not VI, VII or VIII
    for bad in (["CHAPTER XX"],                      # no such heading
                ["CHAPTER"],                         # every heading
                ["CHAPTER IX", "CHAPTER IV"]):       # out of order
        with pytest.raises(SystemExit):
            ye.cuts_at(marks, 10 * H, 12 * H, bad)
    with pytest.raises(SystemExit):                  # the second part would be 7 h
        ye.cuts_at(marks, 10 * H, 6 * H, ["CHAPTER IV"])


def test_the_description_names_the_part():
    d = ye.description("T", "A", 2, 3, [(0, "CHAPTER I"), (H, "CHAPTER II")], H, 2 * H, None, None, False, "kokoro", "Volume")
    assert d.splitlines()[0] == "T, by A. Volume 2 of 3."
    assert "0:00 CHAPTER II" in d and "CHAPTER I\n" not in d


def test_headings_seconds_apart_are_one_chapter():
    marks = [(0, "Oliver Twist"), (3_000, "by Charles Dickens"), (9_000, "CHAPTER I"), (H, "CHAPTER II")]
    lines = ye.description("T", "A", 1, 1, marks, 0, 2 * H, None, None, False, None).split("Chapters\n")[1].splitlines()
    assert lines == ["0:00 CHAPTER I", "1:00:00 CHAPTER II"]
    late = ye.description("T", "A", 1, 1, [(5_000, "CHAPTER I"), (H, "CHAPTER II")], 0, 2 * H, None, None, False, None)
    assert "0:00 CHAPTER I\n" in late                 # a first heading a few seconds in still starts the list at 0:00
    assert ye.description("T", "A", 1, 1, [(60_000, "CHAPTER I")], 0, H, None, None, False, None).endswith("0:00 Start\n1:00 CHAPTER I\n")


def test_a_book_without_headings_cannot_be_cut():
    with pytest.raises(SystemExit):
        ye.split_points([], 30 * H, 12 * H, None)


def test_a_long_heading_is_cut_at_a_word():
    assert ye.short("CHAPTER II.  The second") == "CHAPTER II. The second"
    long = "CHAPTER I. " + "TREATS OF THE PLACE WHERE THE HERO WAS BORN AND OF THE CIRCUMSTANCES ATTENDING HIS BIRTH, AND MORE"
    cut = ye.short(long)
    assert len(cut) <= 97 and cut.endswith("\u2026") and long.startswith(cut[:-1]) and long[len(cut) - 1] == " "
    assert ye.short("x" * 99) == "x" * 99


def test_timestamps():
    assert ye.hms(0) == "0:00" and ye.hms(639_000) == "10:39" and ye.hms(3_723_000) == "1:02:03"
    assert ye.hms(3_723_456, srt=True) == "01:02:03,456"
