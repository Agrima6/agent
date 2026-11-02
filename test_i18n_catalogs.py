"""Every language catalog is complete, and its lines keep the placeholders the code fills in."""
import re
import pytest
import refusals
import composer
import language_switch
from i18n import CATALOGS, CODES
from languages import LANGUAGE_CONFIGS
from interaction_guard import Intent

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_REFERENCE = {
    name: value for name, value in vars(refusals).items()
    if isinstance(value, dict) and isinstance(value.get("en"), str) and name not in ("INTERVIEWER_NAMES",)
}


def _placeholders(text):
    return set(_PLACEHOLDER.findall(text))


def test_every_code_has_a_catalog_and_a_language_config():
    assert set(CATALOGS) == set(CODES) == {c for c in LANGUAGE_CONFIGS if c not in ("en", "hi", "hinglish")}


@pytest.mark.parametrize("code", CODES)
def test_catalog_covers_every_table(code):
    lines = CATALOGS[code]
    missing = [name for name in _REFERENCE if name not in lines and name != "PROCESS_ANSWERS"]
    assert not missing, f"{code} is missing {missing}"
    for kind in refusals.PROCESS_ANSWERS:
        assert f"PROCESS_ANSWERS.{kind}" in lines
    assert set(lines["REFUSALS"]) == {i.name for i in refusals.REFUSALS["en"]}
    assert set(lines["FALLBACK_FOLLOWUP"]) == {"standard", "depth", "clarify", "generic"}


@pytest.mark.parametrize("code", CODES)
def test_placeholders_match_english(code):
    for name, reference in _REFERENCE.items():
        assert _placeholders(CATALOGS[code][name]) == _placeholders(reference["en"]), f"{code}.{name}"
    for kind, answers in refusals.PROCESS_ANSWERS.items():
        assert _placeholders(CATALOGS[code][f"PROCESS_ANSWERS.{kind}"]) == _placeholders(answers["en"])
    for key, text in composer._FALLBACK_FOLLOWUP["en"].items():
        assert _placeholders(CATALOGS[code]["FALLBACK_FOLLOWUP"][key]) == _placeholders(text), f"{code}.{key}"


@pytest.mark.parametrize("code", CODES)
def test_lines_are_in_the_languages_own_script_and_not_empty(code):
    for name, value in CATALOGS[code].items():
        texts = [value] if isinstance(value, str) else [t for v in value.values() for t in ([v] if isinstance(v, str) else v)]
        for text in texts:
            assert text.strip(), f"{code}.{name}"
            if name in ("_AND",):
                continue
            assert any(ord(ch) > 0x0900 for ch in text), f"{code}.{name} has no native-script letters"


@pytest.mark.parametrize("code", CODES)
def test_everything_the_conductor_speaks_resolves_in_the_language(code):
    assert refusals.introduction(code, "Asha", "Driver", "Aarav", {"project": "X", "company": "Y", "experience": "2 years"})
    assert refusals.greeting(code, "Asha", "Driver")
    assert refusals.closing(code) and refusals.closing(code, early=True)
    assert refusals.CLOSING[code] != refusals.CLOSING["en"]
    assert refusals.END_CONFIRM[code] != refusals.END_CONFIRM["en"]
    picker = refusals.RefusalPicker()
    for intent in refusals.REFUSALS["en"]:
        assert picker.pick(intent, code) != refusals.REFUSALS["en"][intent][0]
    assert refusals.fixed_phrases(code)


def test_unknown_line_falls_back_to_english_instead_of_failing():
    assert refusals.CLOSING["xx"] == refusals.CLOSING["en"]


@pytest.mark.parametrize("utterance,expected", [
    ("speak in Tamil please", "ta"), ("Punjabi mein baat karo", "pa"), ("can you speak Telugu", "te"),
    ("Marathi mein baat karo", "mr"), ("please speak in Bengali", "bn"), ("Gujarati please", "gu"),
    ("Kannada please", "kn"), ("can we talk in Malayalam", "ml"), ("can you speak haryanvi", "hi"),
    ("Bhojpuri mein baat karo", "hi"), ("தமிழில் பேசுங்கள்", "ta"), ("ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲ ਕਰੋ", "pa"),
])
def test_candidate_can_ask_for_each_language(utterance, expected):
    assert language_switch.detect_language_request(utterance, "en") == expected


def test_statements_about_a_language_are_not_requests():
    assert language_switch.detect_language_request("I built a Tamil chatbot for my college project last year", "en") is None
    assert language_switch.detect_language_request("I know Punjabi", "en") is None
