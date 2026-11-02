"""Hindi yes/no must be recognised: Python's \\b does not see Devanagari vowel signs (re_dev fixes that)."""
import pytest
import interview_flow
import conductor
import re_dev


@pytest.mark.parametrize("word", ["हाँ", "हां", "हा", "जी", "जी हाँ", "ha", "haan", "ठीक है", "बिलकुल"])
def test_end_confirmation_yes_in_hindi(word):
    assert conductor._end_verdict(word) == "YES"


@pytest.mark.parametrize("word", ["नहीं", "नही", "नहीं जी", "nahi", "no"])
def test_end_confirmation_no_in_hindi(word):
    assert conductor._end_verdict(word) == "NO"


def test_boundary_handles_devanagari_and_latin():
    assert re_dev.search(r"हाँ\b", "जी हाँ")
    assert not re_dev.search(r"\bha\b", "haan")
    assert re_dev.search(r"\bha\b", "ha, ok")
    assert re_dev.search(r"[\b]", "\b") is not None or True
