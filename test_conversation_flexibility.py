"""Conversation bugs seen in a real session: invented "Thank you", "repeat that" repeating an apology, "what's the question,
sir?" treated as an answer, a language question answered by re-reading the question, a repeat after a follow-up repeating the
wrong thing, and a question on screen during the opening."""
import pytest

import interview_flow as flow
import refusals
from interaction_guard import Intent, classify
from runner import InterviewRunner
from test_conductor import (CAND_INTRO, INTRO, Q1, Q2, anyio_backend, followup, harness_factory, lead_in, medium, strong)  # noqa: F401
from test_structured_opening import FACTS, RES1, opening_harness, through_readiness

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------- asking for a repeat, however it is phrased
@pytest.mark.parametrize("text", [
    "What's the question, sir?", "what is the question", "Sir, can you repeat?", "can you say that again please",
    "which question?", "what are you asking", "Could you repeat the question please", "pardon", "I didn't get that"])
def test_all_of_these_are_repeat_requests(text):
    assert classify(text).intent == Intent.REPEAT


@pytest.mark.parametrize("text", [
    "We repeat the call until it succeeds", "I would say that again and again until the cache warms up",
    "The question is how the buckets are resized"])
def test_a_real_answer_is_not_mistaken_for_a_repeat_request(text):
    context = ["How would you warm a cache and what eviction policy do you use", "hashmap buckets resize"]
    assert classify(text, context=context).intent != Intent.REPEAT


# ---------------------------------------------------------------------------- recogniser hallucinations
async def test_an_invented_thank_you_is_not_answered_with_an_apology(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    spoken_before = len(h.spoken)
    await h.conductor.handle_candidate_turn("Thank you.")
    assert len(h.spoken) == spoken_before and h.runner.stage == flow.CLARIFICATION     # nothing said back, nothing advanced


async def test_a_second_one_in_a_row_gets_the_pending_question_not_go_ahead_with_your_answer(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    await h.conductor.handle_candidate_turn("Thank you.")
    out = await h.say("Thank you.")
    assert out.text == refusals.CLARIFY_RETRY["en"] and "answer" not in out.text.lower()


async def test_real_speech_in_between_resets_the_streak(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    await h.conductor.handle_candidate_turn("Thank you.")
    await h.say("How long will the interview take?")
    count = len(h.spoken)
    await h.conductor.handle_candidate_turn("Thank you.")
    assert len(h.spoken) == count                                 # the streak started over: still only noted


async def test_hallucinated_words_never_count_as_readiness_or_as_no_questions(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    for _ in range(3):
        await h.conductor.handle_candidate_turn("Thank you.")
    assert h.runner.stage == flow.CLARIFICATION and not h.runner.readiness_confirmed


# ---------------------------------------------------------------------------- repeat in the opening
async def test_repeat_in_the_opening_repeats_what_is_pending_not_the_last_apology(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    await h.conductor.handle_candidate_turn("Thank you.")
    await h.say("Thank you.")                                     # the apology is now the last thing said
    out = await h.say("Can you repeat what you were saying last? Actually, I didn't say thank you.")
    assert out.action == "REPEAT" and "My name is Aarav" in out.text and "clarify" in out.text
    assert out.text != refusals.CLARIFY_RETRY["en"]


async def test_whats_the_question_in_the_opening_repeats_the_pending_prompt(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    await h.say("How long will it take?")                          # a doubt answered: the pending prompt is "any other doubts?"
    out = await h.say("What's the question, sir?")
    assert out.action == "REPEAT" and out.text == refusals.CLARIFY_MORE["en"]


async def test_repeat_at_the_readiness_question_repeats_it(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    await h.say("No questions.")
    out = await h.say("sorry, can you repeat?")
    assert out.text == refusals.READINESS_ASK["en"] and h.runner.stage == flow.READINESS


# ---------------------------------------------------------------------------- repeat in the interview
async def test_whats_the_question_in_the_interview_repeats_it_and_is_never_judged(harness_factory):
    h = harness_factory(questions=(Q1, Q2), small_talk_rounds=0)
    out = await h.say("What's the question, sir?")
    assert out.action == "REPEAT" and h.judge_calls == [] and h.runner.idx == 0


async def test_repeat_after_a_follow_up_repeats_the_follow_up(harness_factory):
    h = harness_factory(questions=(Q1, Q2), small_talk_rounds=0, evaluations=[medium()], max_followups=2,
                        composer_responses=[followup("What trade-off did you weigh when choosing the load factor?")])
    fu = await h.say("A hashmap hashes keys into buckets.")
    assert fu.is_followup
    out = await h.say("Sorry, can you repeat that?")
    assert out.action == "REPEAT" and "load factor" in out.text and Q1["question_text"] not in out.text


# ---------------------------------------------------------------------------- language
async def test_asking_for_the_language_already_in_use_is_answered_not_just_re_asked(harness_factory):
    h = harness_factory(questions=(Q1, Q2), small_talk_rounds=0)
    out = await h.say("Hi. I'm asking you कि क्या आप English में बात कर सकते हो?")
    assert out.action == "REPEAT" and out.text.startswith(refusals.LANGUAGE_ALREADY["en"]) and Q1["question_text"] in out.text
    assert h.conductor.language == "en" and h.judge_calls == []


async def test_asking_for_the_language_in_use_during_the_opening_is_answered_too(harness_factory):
    h = opening_harness(harness_factory)
    await h.conductor.start()
    out = await h.say("Can you speak in English?")
    assert out.text.startswith(refusals.LANGUAGE_ALREADY["en"]) and h.runner.stage == flow.CLARIFICATION


@pytest.mark.parametrize("text", ["Can you speak in Hinglish?", "Hinglish mein baat karo", "kya aap hinglish me baat kar sakte ho"])
async def test_hinglish_can_be_asked_for_in_the_middle_of_the_interview(harness_factory, text):
    h = harness_factory(questions=(Q1, Q2), small_talk_rounds=0, composer_responses=[lead_in("")])
    applied = []
    h.conductor.on_language_change = applied.append
    await h.say(text)
    assert h.conductor.language == "hinglish" and applied == ["hinglish"]
    assert h.spoken[-2].text == refusals.LANGUAGE_SWITCHED["hinglish"] and Q1["question_text"] in h.spoken[-1].text
    assert h.judge_calls == [] and h.runner.idx == 0              # not an answer, and nothing was skipped


async def test_hinglish_can_be_asked_for_after_a_follow_up_and_the_follow_up_is_kept(harness_factory):
    h = harness_factory(questions=(Q1, Q2), small_talk_rounds=0, evaluations=[medium()],
                        composer_responses=[followup("What trade-off did you weigh?"), lead_in("")])
    await h.say("A hashmap hashes keys into buckets.")
    await h.say("Hinglish mein baat karo")
    assert h.conductor.language == "hinglish" and h.runner.state.followup_count == 1


# ---------------------------------------------------------------------------- the screen during the opening
def test_no_question_is_shown_while_the_interviewer_is_still_in_the_opening():
    r = InterviewRunner(None, "i", {"questions": [INTRO, CAND_INTRO, Q1]})
    seen = []
    r.on_progress = seen.append
    r._spawn = lambda coro: coro.close()
    r.persist_progress_nowait("clarification")
    assert seen[-1]["question_id"] is None
    r.set_stage(flow.MAIN_INTERVIEW)
    r.persist_progress_nowait("asking_question")
    assert seen[-1]["question_id"] is not None


async def test_the_opening_never_puts_a_question_on_the_candidates_screen(harness_factory):
    h = opening_harness(harness_factory)
    shown = []
    h.runner.on_progress = shown.append
    await h.conductor.start()
    await h.say("No questions.")
    assert shown and all(p["question_id"] is None for p in shown)
