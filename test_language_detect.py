import pytest

from language_detect import LanguageFollower, classify_language


@pytest.mark.parametrize("text,expected", [
    ("I have been working on a React project for the last two years", "en"),
    ("My name is Agrima and I worked on the backend services", "en"),
    ("मैंने पिछले दो साल से एक project पर काम किया है जिसमें React इस्तेमाल हुआ", "hinglish"),
    ("मैंने पिछले दो साल एक वेब एप्लिकेशन पर काम किया है", "hi"),
    ("मुझे हिंदी में बात करना पसंद है", "hi"),
    ("mujhe yeh project bahut pasand hai aur maine isme kaam kiya", "hinglish"),
    ("main kaafi time se backend pe kaam kar raha hoon", "hinglish"),
    ("मैं एक backend developer हूँ और Java में काम करता हूँ", "hinglish"),
])
def test_what_language_the_candidate_is_speaking(text, expected):
    assert classify_language(text) == expected


@pytest.mark.parametrize("text", ["yes", "okay sure", "", "   ", "Agrima", "हाँ ठीक", "1234"])
def test_too_little_to_judge_is_not_judged(text):
    assert classify_language(text) is None


def test_english_with_a_couple_of_hindi_words_stays_english():
    assert classify_language("We used the main service and the to-do list was handled by the team") == "en"
    assert classify_language("The project was good, acha result mila for the whole team") == "en"


def follow(*utterances, start="en"):
    f, lang, out = LanguageFollower(), start, []
    for text in utterances:
        new = f.observe(text, lang)
        if new:
            lang = new
            f.note_switch()
        out.append(lang)
    return out


HI_CLEAR = "मैंने पिछले दो साल एक वेब एप्लिकेशन पर काम किया है"
EN_CLEAR = "I mostly worked on the backend and built several internal tools for the team"


def test_a_clear_hindi_answer_in_an_english_interview_switches_at_once():
    assert follow(HI_CLEAR) == ["hi"]


def test_short_replies_never_switch_anything():
    assert follow("हाँ", "ठीक है", "yes", "ok") == ["en", "en", "en", "en"]


def test_a_short_hindi_reply_needs_a_second_one_to_confirm():
    assert follow("मुझे नहीं पता", "फिर भी कोशिश करता हूँ") == ["en", "hi"]    # first is 3 words only: waits; second confirms


def test_it_does_not_flip_back_and_forth_after_a_switch():
    # Hindi, then straight away a clear English sentence: inside the cooldown, so no switch back
    assert follow(HI_CLEAR, EN_CLEAR) == ["hi", "hi"]


def test_from_hinglish_back_to_english_needs_two_clear_english_answers():
    seq = follow(HI_CLEAR, "ठीक है", "ठीक है", EN_CLEAR, EN_CLEAR, start="en")
    assert seq[-2:] == ["hi", "en"]                                               # only the second English one switches


def test_technical_english_inside_a_hinglish_interview_does_not_switch_on_one_answer():
    f = LanguageFollower(); f.note_switch(); f.observe("ठीक है", "hinglish"); f.observe("ठीक है", "hinglish")
    assert f.observe(EN_CLEAR, "hinglish") is None


def test_hindi_staying_hindi_resets_nothing_and_switches_nothing():
    assert follow(HI_CLEAR, HI_CLEAR, "मुझे यह काम बहुत पसंद आया था") == ["hi", "hi", "hi"]
