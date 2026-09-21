"""The interview turn pipeline. Deterministic code owns every decision; LLMs only evaluate and phrase.

    candidate speech
        |
        v
    interaction_guard.classify()          deterministic: what kind of turn is this?
        |
        +-- end / skip / repeat / empty   handled here, no LLM
        +-- restricted (hint, answer, feedback, previous/upcoming, topic change, unrelated,
        |   prompt injection)             fixed pre-approved refusal (refusals.py), no LLM
        +-- answer
              |
              v
          policy.judge_answer()           LLM, structured JSON, validated; internal only
              |
              v
          policy.decide_action()          deterministic: clarify / follow up / go deeper / move on,
              |                           capped by max_followups and time limits
              v
          composer.compose()              LLM, closed context, validated; falls back to fixed text
              |
              v
          speak()                         TTS

The conductor is independent of LiveKit: `speak` is injected, so the whole flow is unit-testable.
State lives on the runner, is changed only here (after a deterministic decision), and is persisted.
"""
import asyncio
import re
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import refusals
from composer import ComposeRequest, Composer
from interaction_guard import Classification, Intent, RESTRICTED_INTENTS, classify
from languages import normalize_language
from policy import Action, Evaluation, decide_action, judge_answer, restricted_intent_from_judge
from runner import InterviewRunner

import turn_trace

logger = logging.getLogger("conductor")

_DURATION_Q = re.compile(r"\b(how long|how much time|duration|how many minutes|kitna time|kitne (?:minute|der)|कितना (?:समय|टाइम)|कितनी देर)\b", re.I)
_FORMAT_Q = re.compile(r"\b(how (?:does|will|is) (?:this|the interview|it) (?:work|go|going)|what (?:is|are) the (?:format|process|rules)|how does the interview|kaise (?:hoga|chalega)|कैसे (?:होगा|चलेगा))\b", re.I)


def _process_question(text: str) -> str | None:
    """Which question ABOUT THE INTERVIEW PROCESS (not its content) the candidate asked, if any."""
    if _DURATION_Q.search(text or ""):
        return "duration"
    if _FORMAT_Q.search(text or ""):
        return "format"
    return None

INTRO_TYPES = ("introduction", "candidate_introduction")
# A candidate is waiting in silence while these run, so each LLM step has a hard deadline. On expiry
# the interview degrades gracefully (no follow-up / plain deterministic wording) instead of stalling.
JUDGE_DEADLINE = 3.5
COMPOSE_DEADLINE = 2.5
MAX_CONSECUTIVE_NON_ANSWERS = 3   # after this many non-answers to one question, move on


@dataclass
class Speech:
    text: str
    action: str                       # GREETING | SMALL_TALK | NEXT_QUESTION | FOLLOW_UP | CLARIFICATION | REDIRECT | REPEAT | INTERVIEW_COMPLETE
    question_id: str = ""
    question_text: str = ""           # canonical text of the question this speech belongs to
    is_followup: bool = False
    interruptible: bool = True        # the closing line is not: it must finish before /complete


class _Stale(Exception):
    """The interview ended while an LLM call was in flight: discard its result."""


class InterviewConductor:
    def __init__(self, runner: InterviewRunner, speak: Callable[[Speech], Awaitable[None]], *,
                 language: str = "en", role: str = "", experience_level: str = "",
                 plain_language: bool = False, composer: Composer | None = None,
                 judge: Callable = judge_answer, interviewer_name: str = "Aarav"):
        self.runner = runner
        self._speak = speak
        self.language = normalize_language(language)
        self.role = role
        self.interviewer_name = interviewer_name
        self.experience_level = experience_level
        self.plain_language = plain_language
        self.composer = composer or Composer()
        self.judge = judge
        self.picker = refusals.RefusalPicker()
        self._lock = asyncio.Lock()      # one candidate turn is processed at a time
        self._last_spoken = ""

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        r = self.runner
        async with self._lock:
            if r.resumed and r.current_question is not None:
                q = r.current_question
                composed = await self._compose(ComposeRequest(
                    action=Action.NEXT_QUESTION, question=q, next_question=q, language=self.language,
                    role=self.role, experience_level=self.experience_level, plain_language=self.plain_language))
                r.last_question_text = composed.question_only or composed.text
                r.asked_texts.append(q["question_text"])
                text = f"{refusals.WELCOME_BACK[self.language]} {composed.text}"
                await self._say(Speech(text, "NEXT_QUESTION", q["id"], q["question_text"]))
                r.persist_progress_nowait("resumed")
                return
            r.persist_progress_nowait("greeting")
            await self._say(Speech(refusals.greeting(self.language, r.candidate_name, self.role, self.interviewer_name), "GREETING",
                                   "p_intro"))

    async def handle_candidate_turn(self, text: str) -> None:
        async with self._lock:
            r = self.runner
            if r.terminal:
                return
            generation = r.generation
            started = time.perf_counter()
            try:
                await self._handle(text or "", generation)
            except _Stale:
                logger.info("discarded stale result interview=%s (interview ended mid-call)", r.interview_id)
            finally:
                logger.info("turn_handled interview=%s question=%s ms=%d", r.interview_id,
                            r.state.question_id if r.state else "-", int((time.perf_counter() - started) * 1000))

    # ---------------------------------------------------------------- helpers
    async def _judge(self, q: dict, text: str, topic: str) -> Evaluation:
        turn_trace.mark("judge_start")
        try:
            return await asyncio.wait_for(asyncio.to_thread(
                self.judge, q["question_text"], list(q.get("expected_topics") or []), text,
                topic=topic, role=self.role), JUDGE_DEADLINE)
        except asyncio.TimeoutError:
            logger.warning("judge missed its %.0fs deadline interview=%s - moving on without a follow-up",
                           JUDGE_DEADLINE, self.runner.interview_id)
            return Evaluation(failed=True, uncertainty=1.0, coverage_score=0.5)
        finally:
            turn_trace.mark("judge_end")

    async def _compose(self, req: ComposeRequest):
        turn_trace.mark("compose_start")
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.composer.compose, req), COMPOSE_DEADLINE)
        except asyncio.TimeoutError:
            logger.warning("composer missed its %.0fs deadline interview=%s - using deterministic wording",
                           COMPOSE_DEADLINE, self.runner.interview_id)
            return self.composer.fallback(req, "deadline_exceeded")
        finally:
            turn_trace.mark("compose_end")

    def _check(self, generation: int) -> None:
        if self.runner.is_stale(generation):
            raise _Stale()

    async def _say(self, speech: Speech) -> None:
        r = self.runner
        if r.terminal and speech.action != "INTERVIEW_COMPLETE":
            return
        self._last_spoken = speech.text
        r.record_turn_nowait(speech.question_id or (r.state.question_id if r.state else ""), "agent", speech.text,
                             is_followup=speech.is_followup, action=speech.action,
                             question_text=speech.question_text)
        try:
            await self._speak(speech)
        except Exception:
            # A TTS failure must never take the interview down: the transcript already has the
            # line, and the candidate can ask for it to be repeated.
            logger.exception("speech failed interview=%s action=%s", r.interview_id, speech.action)

    def _record_candidate(self, text: str, intent: Intent | str, *, followup: bool = False) -> None:
        r = self.runner
        q = r.current_question
        r.record_turn_nowait(q["id"] if q else "", "candidate", text, is_followup=followup,
                             intent=intent.value if isinstance(intent, Intent) else intent,
                             question_text=(q or {}).get("question_text") or "")

    # ---------------------------------------------------------------- turn routing
    async def _handle(self, text: str, generation: int) -> None:
        r = self.runner
        classification = classify(text)
        intent = classification.intent
        q = r.current_question
        if q is None:
            return

        if intent == Intent.END_INTERVIEW:
            return await self._end_early(text)
        if r.small_talk_done < r.small_talk_rounds:
            return await self._small_talk_turn(text, classification, generation)

        if intent == Intent.SKIP:
            self._record_candidate(text, Intent.SKIP)
            return await self._advance_and_ask(generation, [])
        if intent == Intent.DONT_KNOW:
            # A genuine "I don't know": recorded (and scored) as an answer, but never pressed with a
            # follow-up - a real interviewer acknowledges it and moves on.
            self._record_candidate(text, Intent.DONT_KNOW, followup=r.state.followup_count > 0)
            r.state.non_answer_count = 0
            return await self._advance_and_ask(generation, [])
        if intent == Intent.EMPTY:
            return await self._refuse(Intent.EMPTY, text, generation)
        if intent == Intent.REPEAT:
            return await self._repeat(text)
        if intent in RESTRICTED_INTENTS:
            return await self._refuse(intent, text, generation)

        await self._answer_turn(text, classification, generation)

    async def _end_early(self, text: str) -> None:
        r = self.runner
        self._record_candidate(text, Intent.END_INTERVIEW)
        r.end_early()
        await self._say(Speech(refusals.closing(self.language, early=True), "INTERVIEW_COMPLETE",
                               interruptible=False))
        await r.complete()

    async def _repeat(self, text: str) -> None:
        r = self.runner
        q = r.current_question
        self._record_candidate(text, Intent.REPEAT)
        if r.small_talk_done < r.small_talk_rounds:
            spoken = self._last_spoken or refusals.greeting(self.language, r.candidate_name, self.role, self.interviewer_name)
        else:
            spoken = r.last_question_text or (q or {}).get("question_text", "")
        await self._say(Speech(spoken, "REPEAT", (q or {}).get("id", ""), (q or {}).get("question_text", "")))

    async def _refuse(self, intent: Intent, text: str, generation: int) -> None:
        """Fixed, pre-approved response. Never argued with, never generated, never injectable."""
        r = self.runner
        q = r.current_question
        state = r.state
        self._record_candidate(text, intent)
        state.non_answer_count += 1
        if state.non_answer_count >= MAX_CONSECUTIVE_NON_ANSWERS and r.small_talk_done >= r.small_talk_rounds:
            logger.info("moving on after %d non-answers interview=%s question=%s", state.non_answer_count,
                        r.interview_id, state.question_id)
            return await self._advance_and_ask(generation, [], prefix=refusals.MOVE_ON[self.language])
        await self._say(Speech(self.picker.pick(intent, self.language), "REDIRECT", (q or {}).get("id", ""),
                               (q or {}).get("question_text", "")))

    # ---------------------------------------------------------------- small talk
    async def _small_talk_turn(self, text: str, classification: Classification, generation: int) -> None:
        r = self.runner
        intent = classification.intent
        if intent == Intent.REPEAT:
            return await self._repeat(text)
        if intent in RESTRICTED_INTENTS or intent == Intent.EMPTY:
            # Chatting about the weather or asking for a hint before the interview has even begun is
            # harmless, but instructions aimed at the interviewer are not: refuse those only.
            if intent in (Intent.PROMPT_INJECTION, Intent.EMPTY, Intent.REQUEST_ANSWER, Intent.REQUEST_HINT):
                q = r.current_question
                self._record_candidate(text, intent)
                await self._say(Speech(self.picker.pick(intent, self.language), "REDIRECT", (q or {}).get("id", "")))
                return
        # Two-way: the candidate may ask how the interview works before it starts. Those are answered
        # from facts the runner already holds (duration, format) - never anything about the questions.
        topic = _process_question(text)
        if topic and not getattr(self, "_process_answered", False):
            self._process_answered = True
            self._record_candidate(text, "small_talk")
            answer = refusals.PROCESS_ANSWERS[topic][self.language].format(minutes=r.duration_minutes)
            await self._say(Speech(f"{answer} {refusals.READY_TO_BEGIN[self.language]}", "SMALL_TALK", "p_intro"))
            return
        r.small_talk_done += 1
        self._record_candidate(text, "small_talk")
        if r.small_talk_done < r.small_talk_rounds:
            r.persist_progress_nowait("small_talk")
            await self._say(Speech(refusals.SMALL_TALK_FOLLOWUP[self.language], "SMALL_TALK", "p_intro"))
            return
        await self._advance_and_ask(generation, [])

    # ---------------------------------------------------------------- answers
    async def _answer_turn(self, text: str, classification: Classification, generation: int) -> None:
        r = self.runner
        q = r.current_question
        state = r.state

        if q.get("type") in INTRO_TYPES:
            self._record_candidate(text, Intent.ANSWER)
            return await self._advance_and_ask(generation, [])

        evaluation = await self._judge(q, text, state.topic)
        self._check(generation)

        # Second line of defence behind the deterministic guard: phrasings the patterns missed.
        reclassified = restricted_intent_from_judge(evaluation, text)
        if reclassified == Intent.REPEAT:
            return await self._repeat(text)
        if reclassified is not None:
            return await self._refuse(reclassified, text, generation)

        self._record_candidate(text, Intent.ANSWER, followup=state.followup_count > 0)
        state.non_answer_count = 0
        decision = decide_action(state, evaluation, interview_time_exhausted=r.time_exhausted())
        r.record_coverage_nowait(q["id"], evaluation.as_record(), decision.action.value, state.followup_count)
        # Injection-like text inside a long answer: it is still evaluated, but nothing derived from
        # it is allowed to reach the speech composer.
        concepts = [] if classification.suspicious else list(evaluation.mentioned_concepts)
        logger.info("decision interview=%s question=%s action=%s kind=%s reason=%s coverage=%.2f followups=%d/%d",
                    r.interview_id, q["id"], decision.action.value, decision.kind or "-", decision.reason,
                    evaluation.coverage_score, state.followup_count, state.max_followups)

        if decision.action in (Action.FOLLOW_UP, Action.CLARIFICATION):
            state.followup_count += 1
            composed = await self._compose(ComposeRequest(
                action=decision.action, kind=decision.kind, question=q, target=decision.target,
                mentioned_concepts=concepts, followup_number=state.followup_count - 1,
                max_followups=state.max_followups, language=self.language, role=self.role,
                experience_level=self.experience_level, asked_texts=list(r.asked_texts),
                plain_language=self.plain_language))
            self._check(generation)
            state.asked_followups.append(composed.text)
            r.asked_texts.append(composed.text)
            r.persist_progress_nowait("follow_up")
            await self._say(Speech(composed.text, decision.action.value, q["id"], q["question_text"], is_followup=True))
            return

        await self._advance_and_ask(generation, concepts)

    # ---------------------------------------------------------------- moving on
    async def _advance_and_ask(self, generation: int, mentioned: list[str], prefix: str = "") -> None:
        r = self.runner
        previous = r.current_question
        r.advance()
        nq = r.current_question
        if nq is None:
            return await self._finish()
        composed = await self._compose(ComposeRequest(
            action=Action.NEXT_QUESTION, question=previous or nq, next_question=nq, mentioned_concepts=mentioned,
            language=self.language, role=self.role, experience_level=self.experience_level,
            plain_language=self.plain_language))
        self._check(generation)
        r.asked_texts.append(nq["question_text"])
        r.last_question_text = composed.question_only or composed.text
        r.persist_progress_nowait("asking_question")
        await self._say(Speech(f"{prefix} {composed.text}".strip(), "NEXT_QUESTION", nq["id"], nq["question_text"]))

    async def _finish(self) -> None:
        r = self.runner
        r.persist_progress_nowait("completing")
        await self._say(Speech(refusals.closing(self.language), "INTERVIEW_COMPLETE", interruptible=False))
        await r.complete()
