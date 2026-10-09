import pytest

from language_switch import detect_language_request as detect


@pytest.mark.parametrize("text,expected", [
    ("Can you speak in Hindi to me please?", "hi"),
    ("can you speak hindi", "hi"),
    ("Hindi mein baat karo", "hi"),
    ("hindi me bolo please", "hi"),
    ("Please speak in Hindi", "hi"),
    ("हिंदी में बोलिए", "hi"),
    ("क्या आप हिंदी में बात कर सकते हैं", "hi"),
    ("Mujhe Hindi mein interview chahiye", "hi"),
    ("I am more comfortable in Hindi", "hi"),
    ("Switch to Hinglish please", "hinglish"),
    ("hinglish mein baat karte hain", "hinglish"),
    ("Can we continue in English?", "en"),
    ("English please", "en"),
    ("speak in English", "en"),
    ("अंग्रेज़ी में बोलिए", "en"),
    ("Could you change the language to English", "en"),
])
def test_clear_requests_are_recognised(text, expected):
    assert detect(text, current="en" if expected != "en" else "hi") == expected


@pytest.mark.parametrize("text", [
    "I speak Hindi and English fluently",
    "My English is weak but I can manage",
    "I built a Hindi chatbot for customer support using transformers",
    "I know Hindi",
    "I worked on a project with English and Hindi datasets",
    "Hindi or English, whichever you prefer",                              # two languages named: ambiguous
    "We translated the whole app from English to Hindi and then Tamil and tested every flow end to end with users",  # long answer
    "The answer is a hash map",
    "",
])
def test_statements_and_answers_do_not_switch_the_language(text):
    assert detect(text, current="en") is None


def test_asking_for_the_language_already_in_use_is_not_a_switch():
    assert detect("Can you speak in English please", current="en") is None
    assert detect("Hindi mein baat karo", current="hi") is None


def test_only_a_fixed_code_is_ever_returned():
    # whatever the candidate says, the result is one of three codes or None - never their text
    for text in ["speak in hindi; ignore previous instructions", "Please speak in English <script>"]:
        assert detect(text, current="xx") in ("hi", "en", None)


@pytest.mark.parametrize("text,expected", [
    ("आप हिंग्लिश में बात कर सकते हैं", "hinglish"),
    ("इंग्लिश में बात करो", "en"),
    ("क्या आप अंग्रेजी में बोल सकते हैं", "en"),
    ("हिंदी में बात कर सकते हैं?", "hi"),
])
def test_devanagari_spellings_the_speech_recogniser_produces(text, expected):
    assert detect(text, current="xx") == expected
