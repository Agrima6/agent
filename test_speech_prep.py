"""Text preparation before TTS: no markdown / JSON / metadata ever reaches the voice."""
import pytest

from speech_prep import prepare_for_speech, split_sentences


def test_plain_text_passes_through_with_terminal_punctuation():
    assert prepare_for_speech("You mentioned hashing. What happens on a collision?") == \
        "You mentioned hashing. What happens on a collision?"
    assert prepare_for_speech("Tell me more about that") == "Tell me more about that."


def test_markdown_is_removed():
    text = "**Great** point!\n\n- first item\n- second item\n\n## Heading\nUse `HashMap` here."
    out = prepare_for_speech(text)
    for junk in ("**", "##", "`", "- "):
        assert junk not in out
    assert "HashMap" in out and "first item" in out


def test_json_reply_yields_only_the_candidate_facing_text():
    payload = '{"action": "FOLLOW_UP", "topic": "ioc", "candidateFacingText": "Can you explain how inversion of control works in Spring?", "confidence": 0.9}'
    assert prepare_for_speech(payload) == "Can you explain how inversion of control works in Spring?"


def test_json_without_speakable_text_is_dropped():
    assert prepare_for_speech('{"score": 82, "internalReason": "secret"}') == ""


def test_embedded_json_blob_is_stripped():
    out = prepare_for_speech('Sure. {"action": "NEXT_QUESTION", "x": 1} Why is caching useful?')
    assert "action" not in out and "{" not in out
    assert "Why is caching useful?" in out


def test_stage_directions_and_html_are_removed():
    out = prepare_for_speech("[Awaiting your response] <b>Why</b> use an index? (pause)")
    assert "Awaiting" not in out and "<" not in out and "pause" not in out
    assert "Why use an index?" in out.replace("  ", " ")


def test_emoji_links_and_urls_are_removed():
    out = prepare_for_speech("Nice work 🎉 see [the docs](https://example.com/x) or https://example.com/y now.")
    assert "🎉" not in out and "http" not in out and "the docs" in out


def test_typographic_characters_are_normalised():
    # U+2011 (non-breaking hyphen) is emitted by LLMs and mispronounced by TTS engines.
    out = prepare_for_speech("Keep it sub‑second — even under load “quotes”…")
    assert "‑" not in out and "—" not in out and "“" not in out and "…" not in out
    assert "sub-second" in out


@pytest.mark.parametrize("raw,expected_fragment", [
    ("Use a cache, e.g. Redis", "for example"),
    ("SQL vs NoSQL here", "versus"),
    ("This is O(1) lookup", "O of one"),
    ("A scan is O(n) work", "O of n"),
    ("Growth of 30% a year", "30 percent"),
    ("Tom & Jerry", "Tom and Jerry"),
])
def test_symbol_expansion(raw, expected_fragment):
    assert expected_fragment in prepare_for_speech(raw)


def test_technical_terms_are_preserved():
    out = prepare_for_speech("Compare HashMap, C++ templates and snake_case naming.")
    assert "HashMap" in out and "C++" in out and "snake case" in out


def test_hindi_text_is_preserved():
    text = "आपने हैशिंग का ज़िक्र किया। जब दो कीज़ एक ही बकेट में जाती हैं, तब क्या होता है?"
    assert prepare_for_speech(text) == text


def test_empty_and_symbol_only_input():
    for junk in (None, "", "   ", "***", "...", "🎉🎉", "{}", "[ ]"):
        assert prepare_for_speech(junk) == ""


def test_split_sentences_merges_fragments_and_keeps_order():
    text = "Right. You mentioned hashing. What happens when two keys land in the same bucket?"
    parts = split_sentences(text)
    assert " ".join(parts) == text
    assert all(len(p) >= 20 for p in parts[:-1])


def test_split_sentences_does_not_split_on_decimals_or_abbreviations():
    parts = split_sentences("Latency dropped from 2.5 seconds to 0.3 seconds, e.g. after caching. Why?")
    assert " ".join(parts).startswith("Latency dropped from 2.5 seconds")
    assert len(parts) <= 2


def test_split_sentences_handles_devanagari_danda():
    parts = split_sentences("आपने हैशिंग का ज़िक्र किया। जब दो कीज़ एक ही बकेट में जाती हैं, तब क्या होता है?")
    assert len(parts) == 2 or len(parts) == 1
    assert "".join(parts).replace(" ", "") == "आपनेहैशिंगकाज़िक्रकिया।जबदोकीज़एकहीबकेटमेंजातीहैं,तबक्याहोताहै?"
