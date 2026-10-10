"""Language policy, ElevenLabs configuration, turn lifecycle and trace-gap regression tests.

Acoustic accuracy (what Whisper actually writes for real Hindi/Urdu/English audio, how a voice pronounces a term) is NOT
covered here: it needs recordings and live provider credentials. These tests pin the deterministic policy around them.
"""
import pytest

import agent as agent_module
import refusals
import turn_trace
from agent import (ELEVENLABS_ENGLISH_ONLY_MODELS, VoiceProviderError, build_tts, elevenlabs_language,
                   is_candidate_identity, supported_languages_for)
from interaction_guard import Intent, NON_ANSWER_INTENTS
from language_policy import has_unexpected_script, script_profile
from language_switch import detect_unsupported_language_request as unsupported_request
from test_conductor import Harness, Q1, lead_in, strong
from turn_state import TurnPhase, TurnStateMachine
from voice_config import resolve_voice

URDU = "ہمیں ڈیٹا بیس کیشنگ استعمال کرنی چاہیے اور ٹی ٹی ایل رکھنا چاہیے"
URDU_WITH_ENGLISH_TERMS = "یہ cache کو invalidate کرنے کا طریقہ ہے"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def harness_factory():
    made = []

    def make(**kw):
        h = Harness(**kw)
        made.append(h)
        return h

    yield make
    for h in made:
        await h.close()


# ======================================================================= script policy
@pytest.mark.parametrize("text,expected", [
    (URDU, True),
    (URDU_WITH_ENGLISH_TERMS, True),
    ("I would use caching with a short TTL", False),
    ("मैं cache को invalidate करूँगा", False),                     # Devanagari + Latin: Hinglish, supported
    ("मैं डेटाबेस में index लगाऊँगा", False),
    ("", False),
    ("ا", False),                                                   # a single stray character is noise
])
def test_arabic_script_transcripts_are_detected_and_supported_scripts_are_not(text, expected):
    assert has_unexpected_script(text) is expected


def test_script_profile_counts_letters_per_script():
    profile = script_profile("cache को ڈیٹا")
    assert profile["latin"] == 5 and profile["devanagari"] == 2 and profile["arabic"] == 4


@pytest.mark.parametrize("text,label", [
    ("Can I speak in Urdu?", "Urdu"), ("Please speak in Tamil", "Tamil"), ("Could we continue in Punjabi", "Punjabi"),
])
def test_requests_for_unsupported_languages_are_recognised(text, label):
    assert unsupported_request(text) == label


@pytest.mark.parametrize("text", [
    "I know Tamil and built a Tamil chatbot for a bank's customer support team last year",
    "Can I speak in Hindi?", "Hindi or English, whichever you prefer", "The answer is a hash map", "",
])
def test_statements_and_supported_languages_are_not_unsupported_requests(text):
    assert unsupported_request(text) is None


def test_unsupported_language_is_a_non_answer():
    assert Intent.UNSUPPORTED_LANGUAGE in NON_ANSWER_INTENTS


# ======================================================================= conductor behaviour
pytestmark = pytest.mark.anyio


async def test_an_urdu_script_transcript_is_not_evaluated_quoted_or_allowed_to_change_the_language(harness_factory):
    h = harness_factory(language="en", auto_language=True, evaluations=[], composer_responses=[])
    last = await h.say(URDU)
    assert h.conductor.language == "en"                                   # not switched
    assert h.judge_calls == [] and h.posted("coverage-results") == []     # never evaluated as an answer
    assert last.text == refusals.DIDNT_CATCH_LANGUAGE["en"]               # fixed text: nothing of the transcript echoed
    assert "ڈیٹا" not in last.text
    assert h.runner.idx == 0 and h.runner.state.answer_parts == []
    saved = h.posted("turns")[0]
    assert saved["text"] == URDU and saved["intent"] == "unsupported_language"   # source record kept exactly as heard


async def test_a_request_for_an_unsupported_language_stays_in_the_language_and_repeats_the_question(harness_factory):
    h = harness_factory(language="en", evaluations=[], composer_responses=[])
    last = await h.say("Can I speak in Urdu?")
    assert h.conductor.language == "en" and h.runner.idx == 0
    assert refusals.UNSUPPORTED_LANGUAGE["en"] in last.text and Q1["question_text"] in last.text
    assert h.judge_calls == [] and h.posted("language") == []              # nothing saved as a language change
    assert h.posted("turns")[0]["intent"] == "unsupported_language"


async def test_a_voice_that_cannot_speak_hindi_refuses_the_switch_instead_of_attempting_it(harness_factory):
    h = harness_factory(language="en", evaluations=[], composer_responses=[])
    h.conductor.supported_languages = frozenset({"en"})
    last = await h.say("Can you speak in Hindi to me please?")
    assert h.conductor.language == "en" and last.text.startswith(refusals.UNSUPPORTED_LANGUAGE["en"])
    assert h.posted("language") == []


async def test_auto_follow_never_moves_to_a_language_the_voice_cannot_speak(harness_factory):
    h = harness_factory(language="en", auto_language=True, evaluations=[strong()] * 4, composer_responses=[lead_in("")] * 4)
    h.conductor.supported_languages = frozenset({"en"})
    for _ in range(3):
        await h.say("मैं हैश मैप में की को हैश करके bucket चुनता हूँ और collision chain से संभालता हूँ")
    assert h.conductor.language == "en"


async def test_a_single_ambiguous_word_does_not_change_the_language(harness_factory):
    h = harness_factory(language="en", auto_language=True, evaluations=[strong()], composer_responses=[lead_in("")],
                        depth_probe=False)
    await h.say("A HashMap hashes the key, picks a bucket, handles the collision with chaining, theek hai.")
    assert h.conductor.language == "en"


# ======================================================================= ElevenLabs configuration
def _el(monkeypatch, model, language):
    monkeypatch.setattr(agent_module, "ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr(agent_module, "ELEVENLABS_MODEL", model)
    monkeypatch.setattr(agent_module, "ELEVENLABS_VOICE_ID", "voice_1")
    monkeypatch.setattr(agent_module, "ELEVENLABS_VOICE_ID_FEMALE", "")
    monkeypatch.setattr(agent_module, "ELEVENLABS_VOICE_ID_MALE", "")
    return build_tts(resolve_voice(language, provider="elevenlabs"))


@pytest.mark.parametrize("language,code", [("en", "en"), ("hi", "hi"), ("hinglish", "hi")])
def test_elevenlabs_language_is_pinned_so_it_is_not_guessed_per_sentence(monkeypatch, language, code):
    engine, _ = _el(monkeypatch, "eleven_flash_v2_5", language)
    assert elevenlabs_language(language) == code
    assert str(engine._opts.language) == code


def test_a_model_without_language_codes_is_left_unpinned_not_given_an_invalid_option(monkeypatch):
    engine, _ = _el(monkeypatch, "eleven_multilingual_v2", "hi")
    assert not getattr(engine._opts, "language", None)


@pytest.mark.parametrize("model", sorted(ELEVENLABS_ENGLISH_ONLY_MODELS))
def test_an_english_only_model_fails_clearly_for_a_hindi_interview(monkeypatch, model):
    with pytest.raises(VoiceProviderError, match="only speaks English"):
        _el(monkeypatch, model, "hi")
    _el(monkeypatch, model, "en")                                          # ...but is fine for English


def test_supported_languages_follow_the_provider_and_model():
    assert supported_languages_for("elevenlabs", "eleven_flash_v2") == frozenset({"en"})
    assert supported_languages_for("elevenlabs", "eleven_flash_v2_5") == frozenset({"en", "hi", "hinglish"})
    assert supported_languages_for("sarvam", "") == frozenset({"en", "hi", "hinglish"})


def test_only_the_candidate_identity_may_end_a_turn():
    assert is_candidate_identity("candidate-cand_123")
    for other in ("", "agent-AJ_1", "hr-observer", "candidate", None):
        assert not is_candidate_identity(other or "")


# ======================================================================= turn lifecycle
def test_the_turn_machine_accepts_the_normal_sequence_and_blocks_a_second_advance():
    m = TurnStateMachine()
    assert m.go(TurnPhase.QUESTION_DELIVERED) and m.go(TurnPhase.CANDIDATE_SPEAKING) and m.go(TurnPhase.TURN_FINALIZED)
    assert m.go(TurnPhase.ANSWER_SAVED) and m.go(TurnPhase.DECIDING) and m.go(TurnPhase.QUESTION_DELIVERED)
    assert m.turn_number == 1
    assert not m.go(TurnPhase.ANSWER_SAVED)           # cannot save a second answer without a new finalized turn
    assert m.rejected == [("QUESTION_DELIVERED", "ANSWER_SAVED")]


def test_an_answer_cannot_be_saved_before_the_turn_is_finalized_and_completed_is_terminal():
    m = TurnStateMachine()
    m.go(TurnPhase.QUESTION_DELIVERED)
    assert not m.go(TurnPhase.ANSWER_SAVED) and not m.go(TurnPhase.DECIDING)
    assert m.go(TurnPhase.COMPLETED)
    for phase in TurnPhase:
        if phase != TurnPhase.COMPLETED:
            assert not m.go(phase)


def test_re_asking_the_same_question_is_idempotent():
    m = TurnStateMachine()
    m.go(TurnPhase.QUESTION_DELIVERED)
    assert m.go(TurnPhase.QUESTION_DELIVERED) and m.rejected == []


async def test_a_conductor_turn_walks_the_phases_in_order_and_ends_waiting_for_the_next_answer(harness_factory):
    h = harness_factory(evaluations=[strong()], composer_responses=[lead_in("")], depth_probe=False)
    await h.say("A HashMap hashes the key, picks a bucket, handles collisions with chaining and resizes.")
    assert h.conductor.turns.turn_number == 1
    assert h.conductor.turns.phase == TurnPhase.QUESTION_DELIVERED       # the next question has been delivered
    assert h.conductor.turns.rejected == []                              # nothing illegal happened on the way
    assert h.runner.idx == 1


async def test_a_second_finalized_turn_cannot_advance_twice(harness_factory):
    h = harness_factory(evaluations=[strong(), strong()], composer_responses=[lead_in(""), lead_in("")], depth_probe=False)
    await h.say("A HashMap hashes the key, picks a bucket, handles collisions with chaining and resizes.")
    assert h.runner.idx == 1
    h.conductor._advanced_turn = h.conductor.turns.turn_number           # as if this turn had already advanced
    await h.conductor._advance_and_ask(h.runner.generation, [])
    assert h.runner.idx == 1                                             # the duplicate advance was ignored


# ======================================================================= observability gap
def test_elevenlabs_tts_latency_is_attached_to_the_turn_it_belongs_to_exactly_once(caplog):
    import logging
    tracer = turn_trace.TurnTracer("intv_x")
    tracer.begin().mark("user_speech_end")
    tracer.mark("say_called")
    tracer.finish()
    with caplog.at_level(logging.INFO, logger="voice-metrics"):
        first = tracer.note_tts_metrics(0.42, streamed=True, connection_reused=True)
        second = tracer.note_tts_metrics(0.30, streamed=True)              # a later sentence of the same turn
    assert first["tts_ttfa_ms"] == 420 and first["turn_id"] == tracer.last_report["turn_id"]
    assert second is None
    assert sum("turn_trace_tts" in r.getMessage() for r in caplog.records) == 1


def test_a_turn_that_already_has_a_tts_span_is_not_overwritten():
    tracer = turn_trace.TurnTracer("intv_y")
    tracer.begin().mark("user_speech_end")
    tracer.current.mark("tts_request", 10.0)
    tracer.current.mark("tts_first_audio", 10.5)
    tracer.mark("say_called")
    tracer.finish()
    assert tracer.last_report["tts_ttfa_ms"] == 500
    assert tracer.note_tts_metrics(0.9) is None


def test_a_pronunciation_dictionary_is_used_only_when_both_ids_are_configured(monkeypatch):
    monkeypatch.setattr(agent_module, "ELEVENLABS_PRONUNCIATION_DICTIONARY_ID", "dict_1")
    monkeypatch.setattr(agent_module, "ELEVENLABS_PRONUNCIATION_VERSION_ID", "")
    engine, _ = _el(monkeypatch, "eleven_flash_v2_5", "en")
    assert not engine._opts.pronunciation_dictionary_locators
    monkeypatch.setattr(agent_module, "ELEVENLABS_PRONUNCIATION_VERSION_ID", "ver_1")
    engine, _ = _el(monkeypatch, "eleven_flash_v2_5", "en")
    assert [(l.pronunciation_dictionary_id, l.version_id) for l in engine._opts.pronunciation_dictionary_locators] == [("dict_1", "ver_1")]


# ======================================================================= the interviewer's own text
def test_interviewer_text_in_arabic_script_is_rejected_so_the_deterministic_wording_is_used():
    from output_guard import violations
    assert "unsupported_script" in violations("Could you explain ڈیٹا بیس کیشنگ in more detail?")
    for fine in ("Could you explain how caching reduces database load?", "आपने cache कैसे invalidate किया?",
                 "आप database में index कैसे लगाएँगे?"):
        assert "unsupported_script" not in violations(fine)


def test_concepts_in_arabic_script_are_never_quoted_back_to_the_candidate():
    from topic_guard import ground_concepts
    assert ground_concepts(["ڈیٹا بیس", "کیشنگ"], "ہم ڈیٹا بیس میں کیشنگ استعمال کرتے ہیں") == []


# ======================================================================= pause matrix (configuration level, NOT audio)
@pytest.mark.parametrize("pause_s,committed_by_detector", [(1, False), (2, False), (5, False), (10, True)])
def test_thinking_pauses_up_to_five_seconds_are_not_committed_and_ten_seconds_is_by_design(pause_s, committed_by_detector):
    """The detector commits after `min_delay` of silence when the end-of-turn model believes the sentence is complete,
    and after `max_delay` at the latest. A pause shorter than min_delay can never end the turn; one shorter than
    max_delay can end it only if the model calls the sentence finished. This pins the configured numbers, not the
    model's judgement on real speech (needs recordings)."""
    from agent import TURN_HANDLING
    min_delay = TURN_HANDLING["endpointing"]["min_delay"]
    max_delay = TURN_HANDLING["endpointing"]["max_delay"]
    if pause_s < min_delay:
        assert not committed_by_detector
    elif pause_s < max_delay:
        assert not committed_by_detector            # only if the model said "complete" would it end earlier than max
    else:
        assert committed_by_detector                # an unbroken silence longer than max_delay does end the turn


async def test_two_interviews_in_one_process_never_share_turn_state(harness_factory):
    import asyncio
    a = harness_factory(evaluations=[strong()], composer_responses=[lead_in("")], depth_probe=False)
    b = harness_factory(evaluations=[], composer_responses=[])
    await asyncio.gather(
        a.say("A HashMap hashes the key, picks a bucket, handles collisions with chaining and resizes."),
        b.say("Can I speak in Hindi?"))
    assert a.runner.idx == 1 and a.conductor.language == "en"           # advanced normally
    assert b.runner.idx == 0 and b.conductor.language == "hi"           # only language changed
    assert a.conductor.turns is not b.conductor.turns
    assert a.runner.state.answer_parts == [] and b.runner.state.answer_parts == []
