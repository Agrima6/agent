"""Edge cases found by testing a live session: the recogniser echoing its own prompt, repeated words, silence
hallucinations, presence checks, audio complaints, requests for time - and the rule that none of them may be
judged, refused, scored, or counted toward skipping (or ending) the interview.
"""
import types

import pytest

import evaluation
import refusals
from interaction_guard import Intent, classify
from language_policy import (STT_VOCABULARY_HINT, clean_transcript, is_prompt_echo, looks_like_repetition_hallucination,
                             noise_reason, remove_prompt_echo)
from policy import Evaluation
from test_conductor import Harness, Q1, Q2, lead_in, strong

LEGACY_PROMPT_ECHO = "Job interview with Arav from Workmate."          # seen verbatim in a live session
GARBLED_WITH_ECHO = "विक और ईश्यू बता लगए हैं? वीजवीज में दे रहा है ये, आज़, Job interview with Arav from Workmate."


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


# ======================================================================= recogniser noise
def test_the_prompt_is_a_glossary_not_a_sentence():
    assert "." not in STT_VOCABULARY_HINT.rstrip(".") and "Job interview" not in STT_VOCABULARY_HINT


@pytest.mark.parametrize("text", [LEGACY_PROMPT_ECHO, "job interview with arav from workmateiq", "Job interview with Arav",
                                  "Hinglish, Hindi, English, WorkmateIQ, Arav."])
def test_the_recognisers_own_prompt_repeated_back_is_noise(text):
    assert is_prompt_echo(text) and noise_reason(text) and classify(text).intent == Intent.EMPTY


@pytest.mark.parametrize("text", ["Hindi", "Hindi or English", "Hinglish, Hindi, English", "I interviewed with Arav last week",
                                  "I prefer Hindi and English for the interview"])
def test_a_candidate_who_merely_says_a_prompt_word_is_not_an_echo(text):
    assert not is_prompt_echo(text)


def test_an_echo_at_the_end_of_a_real_sentence_is_removed_for_working_use_only():
    cleaned, removed = clean_transcript(GARBLED_WITH_ECHO)
    assert "Arav" not in cleaned and removed > 0
    assert remove_prompt_echo("We would cache reads. Job interview with Arav from Workmate.") == "We would cache reads."
    assert "Arav" in GARBLED_WITH_ECHO                                # the original is untouched


@pytest.mark.parametrize("text", ["चलाज़, चलाज़, चलाज़.", "yes yes yes yes", "okay okay okay okay okay",
                                  "I won't tell anyone, because I won't tell anyone. And I won't tell anyone about it."])
def test_a_repeated_word_or_clause_is_noise(text):
    assert looks_like_repetition_hallucination(text) and classify(text).intent == Intent.EMPTY


@pytest.mark.parametrize("text", ["Thanks for watching!", "Please subscribe to the channel", "Subtitles by the Amara.org community",
                                  "[Music]"])
def test_known_silence_hallucinations_are_noise(text):
    assert noise_reason(text) or classify(text).intent == Intent.EMPTY


@pytest.mark.parametrize("text", ["no, I did not do that", "We use a cache because the cache is fast",
                                  "retry the request, then retry the batch, then give up", "Thank you for the question"])
def test_ordinary_speech_is_not_noise(text):
    assert noise_reason(text) is None


# ======================================================================= the interview survives noise
pytestmark = pytest.mark.anyio


async def test_noise_on_the_last_question_never_ends_the_interview(harness_factory):
    """The live failure: two echoes and a repeated word on the final question triggered 'move on' -> interview over."""
    h = harness_factory(questions=(Q1,), evaluations=[], composer_responses=[])
    for utterance in (LEGACY_PROMPT_ECHO, LEGACY_PROMPT_ECHO, "चलाज़, चलाज़, चलाज़.", "okay okay okay okay"):
        await h.say(utterance)
    assert not h.runner.terminal and h.runner.idx == 0
    assert not any(s.action == "INTERVIEW_COMPLETE" for s in h.spoken)
    assert h.runner.state.non_answer_count == 0 and h.judge_calls == []


async def test_repeated_noise_is_met_with_a_microphone_hint_not_a_skip(harness_factory):
    h = harness_factory(questions=(Q1,), evaluations=[], composer_responses=[])
    spoken = [await h.say(LEGACY_PROMPT_ECHO) for _ in range(3)]
    assert spoken[2].text == refusals.AUDIO_TROUBLE["en"]
    assert spoken[0].text != refusals.AUDIO_TROUBLE["en"]


async def test_a_real_answer_resets_the_noise_streak(harness_factory):
    h = harness_factory(questions=(Q1, Q2), evaluations=[strong()], composer_responses=[lead_in("")], depth_probe=False)
    await h.say(LEGACY_PROMPT_ECHO)
    await h.say("A HashMap hashes the key, picks a bucket, handles collisions with chaining and resizes.")
    assert h.runner.idx == 1                                          # moved on because it was ANSWERED, not skipped
    assert h.runner.state.noise_count == 0


async def test_a_judge_only_off_topic_verdict_is_softened_the_first_time(harness_factory):
    """Garbled transcripts look 'off-topic' to the model. First time per question: 'I didn't catch that', not a refusal."""
    off = Evaluation(judge_intent="off_topic", relevance=0, coverage_score=0.0)
    h = harness_factory(questions=(Q1,), evaluations=[off, off], composer_responses=[])
    first = await h.say("विक और ईश्यू बता लगए हैं वीजवीज में दे रहा है ये आज़ और कुछ भी नहीं")
    assert first.text in refusals.REFUSALS["en"][Intent.EMPTY] or first.text in refusals.REFUSALS["hi"][Intent.EMPTY]
    assert h.runner.state.non_answer_count == 0
    second = await h.say("something unrelated about football and the weather yesterday at the match")
    assert h.runner.state.non_answer_count == 1                       # a repeated verdict is a real redirect


# ======================================================================= presence, audio trouble, time
@pytest.mark.parametrize("utterance,table", [
    ("Are you still there?", "ACK_PRESENCE"), ("Can you hear me?", "ACK_PRESENCE"), ("Hello? Hello?", "ACK_PRESENCE"),
    ("Give me a second", "ACK_NEED_TIME"), ("Let me think", "ACK_NEED_TIME"), ("my internet is slow", "ACK_NEED_TIME"),
])
async def test_presence_checks_and_requests_for_time_get_a_kind_fixed_reply_and_change_nothing(harness_factory, utterance, table):
    h = harness_factory(evaluations=[], composer_responses=[])
    started = h.runner.state.started_at
    last = await h.say(utterance)
    assert last.text == getattr(refusals, table)["en"] and last.action == "ACK"
    assert h.judge_calls == [] and h.posted("coverage-results") == []
    assert (h.runner.idx, h.runner.state.non_answer_count, h.runner.state.started_at) == (0, 0, started)
    assert h.runner.state.answer_parts == []


@pytest.mark.parametrize("utterance", ["I can't hear you", "Please speak slower", "Your voice is breaking", "आवाज़ नहीं आ रही है"])
async def test_audio_complaints_repeat_the_question(harness_factory, utterance):
    h = harness_factory(evaluations=[], composer_responses=[])
    last = await h.say(utterance)
    assert last.action == "REPEAT" and last.question_id == Q1["id"] and h.runner.idx == 0 and h.judge_calls == []


async def test_a_long_answer_that_starts_with_let_me_think_is_still_an_answer(harness_factory):
    h = harness_factory(evaluations=[strong()], composer_responses=[lead_in("")], depth_probe=False)
    await h.say("Let me think about it: I would add a cache in front of the database with a short TTL and invalidate it on "
                "every write so readers never see stale data")
    assert len(h.judge_calls) == 1


async def test_a_question_about_the_interview_itself_is_answered_not_judged(harness_factory):
    h = harness_factory(small_talk_rounds=0, evaluations=[], composer_responses=[])
    last = await h.say("how long will this interview take")
    assert "minutes" in last.text and refusals.PLEASE_CONTINUE["en"] in last.text
    assert h.judge_calls == [] and h.runner.idx == 0


# ======================================================================= idle handling
async def test_idle_nudge_is_gentle_limited_and_never_counted(harness_factory):
    h = harness_factory(evaluations=[], composer_responses=[])
    await h.conductor.idle_nudge()                                      # before any question delivered: not idle yet
    assert h.spoken == []
    h.conductor.turns.phase = h.conductor.turns.phase.QUESTION_DELIVERED
    await h.conductor.idle_nudge()
    await h.conductor.idle_nudge()
    await h.conductor.idle_nudge()                                      # third is ignored
    assert [s.text for s in h.spoken] == [refusals.IDLE_NUDGE["en"], refusals.AUDIO_TROUBLE["en"]]
    assert h.runner.state.non_answer_count == 0 and h.runner.idx == 0


async def test_idle_nudge_never_talks_over_a_turn_in_progress(harness_factory):
    h = harness_factory(evaluations=[], composer_responses=[])
    h.conductor.turns.phase = h.conductor.turns.phase.QUESTION_DELIVERED
    async with h.conductor._lock:
        await h.conductor.idle_nudge()
    assert h.spoken == []


# ======================================================================= scoring never sees noise
def test_noise_turns_are_not_part_of_an_answer_in_the_report():
    turn = lambda text, intent: types.SimpleNamespace(question_id="q1", speaker="candidate", text=text, intent=intent, action=None)
    text = evaluation.answer_text_for(
        [turn(LEGACY_PROMPT_ECHO, "empty"), turn("चलाज़, चलाज़, चलाज़.", "empty"),
         turn("I would add a cache with a short TTL. " + LEGACY_PROMPT_ECHO, "answer")], "q1")
    assert text == "I would add a cache with a short TTL."
