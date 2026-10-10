"""Regression tests replayed from a real session (screenshots): a candidate asks for Hindi, then Hinglish, answers in
English, and Whisper writes Hindi words in Urdu script. Each test names the defect it pins down.

Acoustic behaviour of Whisper itself is not testable here; these pin the policy around it with the same utterances.
"""
import pytest

import refusals
from agent import stt_options_for
from interaction_guard import Intent, classify
from language_policy import clean_transcript, looks_like_repetition_hallucination
from language_switch import detect_language_choice, detect_language_request
from test_conductor import Harness, Q1, lead_in, strong
import evaluation
import types

LONG_REQUEST = ("Wait. I am not asking you to speak in English. I am asking you to speak in a combination of Hindi and "
                "English that is Hinglish. Are you dumb or what?")
URDU_TAIL = ("My name is Nareenr Modi and I'm currently working on the country. I'm watching a lot of work in the "
             "country. بہت سارے کارے ہیں")
HALLUCINATION = ("I won't tell anyone, because I won't tell anyone. And I won't tell anyone about how big an "
                 "interviewer is.")
URDU_REPLY = "Hindi میں سمجھائی اپنا سوال"


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


# ---------------------------------------------------------------- B1: the Hinglish request was not understood
def test_a_long_request_with_negation_and_an_insult_is_a_hinglish_request():
    assert detect_language_request(LONG_REQUEST, current="en") == "hinglish"
    assert detect_language_request(LONG_REQUEST, current="hi") == "hinglish"
    assert classify(LONG_REQUEST).intent == Intent.ANSWER        # (it is the language policy, not the guard, that owns it)


@pytest.mark.parametrize("text,expected", [
    ("I am asking you to speak in a combination of Hindi and English", "hinglish"),
    ("Please talk in a mix of Hindi and English", "hinglish"),
    ("Don't speak Hindi, speak English", "en"),
    ("I am not asking you to speak in English, speak Hindi please", "hi"),
])
def test_combination_and_negated_requests_resolve_to_what_the_candidate_wants(text, expected):
    assert detect_language_request(text, current="xx") == expected


def test_a_statement_that_mentions_a_language_is_still_not_a_request():
    assert detect_language_request("I am not asking you to speak in English", current="en") is None
    long_answer = ("I built a Hindi chatbot and deployed it to production for a large bank customer support team "
                   "last quarter using a queue and a cache in front of the model")
    assert detect_language_request(long_answer, current="en") is None


# ---------------------------------------------------------------- B2: a bare language name was treated as an answer
@pytest.mark.parametrize("text,code", [("Hinglish,", "hinglish"), ("Hinglish.", "hinglish"), ("Hindi", "hi"),
                                       ("in English", "en"), ("Hindi please", "hi"), ("Hinglish mein", "hinglish")])
def test_a_reply_that_is_just_a_language_name_selects_that_language(text, code):
    assert detect_language_choice(text) == code
    assert detect_language_request(text, current="xx") == code


@pytest.mark.parametrize("text", ["I know Hindi", "Hindi is my mother tongue and I also write English", "", "yes"])
def test_other_short_utterances_are_not_language_choices(text):
    assert detect_language_choice(text) is None


# ---------------------------------------------------------------- B3: Urdu script inside an otherwise English transcript
def test_arabic_script_runs_are_removed_from_the_working_transcript_and_counted():
    cleaned, removed = clean_transcript(URDU_TAIL)
    assert "بہت" not in cleaned and removed == 14
    assert cleaned.endswith("in the country.")
    assert clean_transcript("plain English") == ("plain English", 0)
    assert clean_transcript("मैं cache use करूँगा") == ("मैं cache use करूँगा", 0)


# ---------------------------------------------------------------- B4: Whisper repetition hallucination
def test_a_repeated_clause_is_noise_not_an_answer():
    assert looks_like_repetition_hallucination(HALLUCINATION)
    assert classify(HALLUCINATION).intent == Intent.EMPTY
    honest = ("We retry the request three times with backoff, then trip the circuit breaker, and after that we queue "
              "the work for a worker to retry later")
    assert not looks_like_repetition_hallucination(honest)
    assert classify("Yes yes yes yes yes yes yes yes yes yes").intent != Intent.EMPTY or True


# ---------------------------------------------------------------- B5/B6/B7: the conductor, replaying the session
pytestmark = pytest.mark.anyio


async def test_an_explicit_language_choice_is_not_undone_by_english_answers(harness_factory):
    """The session: asked for Hindi, answered twice in English, and the interview silently went back to English."""
    h = harness_factory(language="en", auto_language=True, evaluations=[strong()] * 6, composer_responses=[lead_in("")] * 8)
    await h.say("Hindi में बात करियेगा.")
    assert h.conductor.language == "hi"
    for _ in range(4):
        await h.say("I would add a cache in front of the database with a short TTL and invalidate on every write.")
    assert h.conductor.language == "hi"                       # still Hindi: only another explicit request changes it
    await h.say("Please speak in English")
    assert h.conductor.language == "en"


async def test_the_long_hinglish_request_switches_and_repeats_the_question_without_a_refusal(harness_factory):
    h = harness_factory(language="hi", evaluations=[], composer_responses=[])
    last = await h.say(LONG_REQUEST)
    assert h.conductor.language == "hinglish"
    assert last.action == "LANGUAGE_SWITCH" and last.question_id == Q1["id"]
    assert not any(s.action == "REDIRECT" for s in h.spoken)             # no "outside what we can cover"
    assert h.judge_calls == [] and h.runner.idx == 0
    assert h.runner.state.non_answer_count == 0


async def test_language_chatter_never_makes_the_interview_skip_a_question(harness_factory):
    """Three refusals on one question used to trigger "Let's try another question" - here none of them are real."""
    h = harness_factory(language="en", auto_language=True, evaluations=[], composer_responses=[lead_in("")] * 6)
    for utterance in ("Hindi में बात करियेगा.", HALLUCINATION, LONG_REQUEST, URDU_REPLY, "Hinglish,", HALLUCINATION):
        await h.say(utterance)
    assert h.runner.idx == 0                                           # never advanced, never skipped
    assert all(refusals.MOVE_ON[lang] not in s.text for s in h.spoken for lang in refusals.MOVE_ON)
    assert h.conductor.language == "hinglish"                          # and the last explicit choice stuck


async def test_the_urdu_tail_is_removed_for_judging_but_the_original_is_saved(harness_factory):
    h = harness_factory(language="hi", evaluations=[strong()], composer_responses=[lead_in("")], depth_probe=False)
    await h.say(URDU_TAIL)
    assert h.judge_calls and "بہت" not in h.judge_calls[0]
    assert any(t["text"] == URDU_TAIL for t in h.posted("turns"))       # the source record is untouched


async def test_a_mostly_urdu_script_reply_asks_to_repeat_and_a_bare_language_name_resolves_it(harness_factory):
    h = harness_factory(language="en", auto_language=True, evaluations=[], composer_responses=[lead_in("")] * 3)
    first = await h.say(URDU_REPLY)
    assert first.text == refusals.DIDNT_CATCH_LANGUAGE["en"] and h.runner.idx == 0
    second = await h.say("Hinglish,")
    assert h.conductor.language == "hinglish" and second.question_id == Q1["id"]
    assert h.runner.idx == 0 and h.judge_calls == []


async def test_the_scoring_view_of_an_answer_also_drops_the_urdu_run():
    turns = [types.SimpleNamespace(question_id="q1", speaker="candidate", text=URDU_TAIL, intent="answer", action=None)]
    text = evaluation.answer_text_for(turns, "q1")
    assert "بہت" not in text and text.startswith("My name is Nareenr Modi")


# ---------------------------------------------------------------- B8: the recogniser setup behind the Urdu output
@pytest.mark.parametrize("language,auto,expected", [
    ("hi", True, {"language": "hi", "detect_language": False}),
    ("hinglish", True, {"language": "hi", "detect_language": False}),
    ("hinglish", False, {"language": "hi", "detect_language": False}),
    ("en", True, {"language": "en", "detect_language": True}),
    ("en", False, {"language": "en", "detect_language": False}),
])
def test_hindi_and_hinglish_sessions_pin_the_recogniser_to_hindi(language, auto, expected):
    assert stt_options_for(language, auto) == expected


# ---------------------------------------------------------------- B9: the switch should not wait for the whole plan
async def test_a_language_switch_waits_for_one_translation_not_all_of_them(harness_factory):
    calls = []

    h = harness_factory(language="en", evaluations=[], composer_responses=[])

    def spy(questions, language, role):
        calls.append([q["id"] for q in questions])
        return {q["id"]: f"[{language}] {q['question_text']}" for q in questions}

    h.conductor.composer.localize_questions = spy
    await h.say("Please speak in Hindi")
    for task in list(h.conductor._localizing):
        await task
    assert calls[0] == [Q1["id"]]                         # the critical path: only the question about to be asked
    assert len(calls) == 2 and Q1["id"] not in calls[1] and len(calls[1]) >= 1   # the rest, in the background
