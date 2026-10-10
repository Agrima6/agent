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
from composer import ComposeRequest, Composed, Composer
from language_detect import LanguageFollower
from language_switch import detect_language_request, detect_unsupported_language_request
from language_policy import (MIN_WORDS_AFTER_CLEANING, clean_transcript, has_unexpected_script)
from interaction_guard import Classification, Intent, RESTRICTED_INTENTS, classify
from languages import normalize_language
from policy import Action, Evaluation, decide_action, judge_answer, restricted_intent_from_judge
from runner import InterviewRunner, question_topic

import turn_trace
from turn_state import TurnPhase, TurnStateMachine

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
# Upper bound on the answer text (this question so far) handed to the judge.
MAX_JUDGE_CHARS = 6000
JUDGE_DEADLINE = 3.5
COMPOSE_DEADLINE = 2.5
LOCALIZE_DEADLINE = 8.0        # turning a question into Hindi/Hinglish takes longer than an English lead-in, and falling back to English is worse than waiting
LOCALIZE_BATCH_DEADLINE = 15.0
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
                 judge: Callable = judge_answer, interviewer_name: str = "Aarav", on_language_change=None,
                 auto_language: bool = False, supported_languages=None):
        self.runner = runner
        # Languages the configured voice can actually speak (None = all). A switch - requested or followed - to
        # anything else is refused rather than attempted with a voice that cannot pronounce it.
        self.supported_languages = frozenset(supported_languages) if supported_languages else None
        # Called (sync or async) with the new language code when the candidate asks to switch: the agent job uses it to
        # retune speech recognition and the voice. Failures are logged and never stop the interview.
        self.on_language_change = on_language_change
        # When on, the interview follows the language the candidate is actually speaking (see language_detect.py).
        self._follower = LanguageFollower() if auto_language else None
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
        self._localized: dict[tuple[str, str], str] = {}     # (question id, language) -> the question already rendered in that language
        self._localizing: set[asyncio.Task] = set()
        self._last_spoken = ""
        # Normalised texts of turns queued or being handled right now: a second end-of-turn carrying the same words
        # while the first is still in flight is one answer reported twice (a duplicated end-of-turn event, or a
        # "done speaking" press racing the detector) - it must not be answered twice or advance twice.
        self._inflight_turns: set[str] = set()
        # Once the candidate has explicitly asked for a language, auto-follow is OFF for the rest of the session: a Hinglish
        # speaker answering a technical question in English must not be switched back to English behind their request.
        self._language_locked = False
        self._turn_original: str | None = None    # the transcript exactly as heard, for the turn being handled
        self.turns = TurnStateMachine()          # QUESTION_DELIVERED -> ... -> ANSWER_SAVED -> DECIDING (see turn_state.py)
        self._advanced_turn = 0                  # the finalized-turn number that already advanced the interview

    # ---------------------------------------------------------------- diagnostics
    def _event(self, name: str, **fields) -> None:
        """One structured line per interview event. Carries ids, counts and decisions - never the candidate's words,
        tokens or any other sensitive content."""
        r = self.runner
        base = {"event": name, "interview": r.interview_id, "question": r.state.question_id if r.state else "-",
                "idx": r.idx, "followups": r.state.followup_count if r.state else 0, "phase": self.turns.phase.value,
                "turn": self.turns.turn_number}
        logger.info(" ".join(f"{k}={v}" for k, v in {**base, **fields}.items()))

    @staticmethod
    def _context_for(q: dict | None) -> list[str]:
        """What a genuine answer to this question would talk about (lexical anchors for the guard)."""
        if not q:
            return []
        parts = [q.get("question_text") or "", q.get("topic") or "", q.get("focus") or ""]
        parts += [str(t) for t in (q.get("expected_topics") or [])]
        parts += [str(t) for t in (q.get("followup_topics") or [])]
        parts += [str(c).replace("_", " ") for c in (q.get("competencies") or [])]
        return [p for p in parts if p]

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        r = self.runner
        if self.language != "en":
            await self._localize_upcoming(self.language, wait=False)    # background: questions are ready before they are asked
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
        normalised = " ".join((text or "").lower().split())
        if normalised and normalised in self._inflight_turns:
            self._event("turn_duplicate_ignored", words=len(normalised.split()))
            return
        if normalised:
            self._inflight_turns.add(normalised)
        try:
            async with self._lock:
                r = self.runner
                if r.terminal:
                    return
                generation = r.generation
                started = time.perf_counter()
                if not self.turns.go(TurnPhase.TURN_FINALIZED):
                    # A previous turn ended without the interviewer speaking (e.g. a failed provider call). Resync and
                    # carry on rather than refusing a candidate's answer; the rejected transition is in the log.
                    self._event("turn_phase_rejected", target=TurnPhase.TURN_FINALIZED.value)
                    self.turns.phase = TurnPhase.QUESTION_DELIVERED
                    self.turns.go(TurnPhase.TURN_FINALIZED)
                self._event("turn_received", words=len((text or "").split()), chars=len(text or ""))
                try:
                    await self._handle(text or "", generation)
                except _Stale:
                    logger.info("discarded stale result interview=%s (interview ended mid-call)", r.interview_id)
                finally:
                    logger.info("turn_handled interview=%s question=%s ms=%d", r.interview_id,
                                r.state.question_id if r.state else "-", int((time.perf_counter() - started) * 1000))
        finally:
            self._inflight_turns.discard(normalised)

    # ---------------------------------------------------------------- helpers
    async def _judge(self, q: dict, text: str, topic: str) -> Evaluation:
        turn_trace.mark("judge_start")
        try:
            context = {k: q[k] for k in ("claim", "focus") if q.get(k)}     # only when present: keeps simple judges working
            return await asyncio.wait_for(asyncio.to_thread(
                self.judge, q["question_text"], list(q.get("expected_topics") or []), text,
                topic=topic, role=self.role, **context), JUDGE_DEADLINE)
        except asyncio.TimeoutError:
            logger.warning("judge missed its %.0fs deadline interview=%s - moving on without a follow-up",
                           JUDGE_DEADLINE, self.runner.interview_id)
            return Evaluation(failed=True, uncertainty=1.0, coverage_score=0.5)
        finally:
            turn_trace.mark("judge_end")

    async def _compose(self, req: ComposeRequest):
        turn_trace.mark("compose_start")
        nq = req.next_question or {}
        language = normalize_language(req.language)
        if req.action == Action.NEXT_QUESTION and language != "en":
            cached = self._localized.get((str(nq.get("id")), language))
            if cached:                                   # translated ahead of time: nothing to wait for
                turn_trace.mark("compose_end")
                return Composed(text=cached, action=Action.NEXT_QUESTION, source="cache", question_only=cached,
                                topic=question_topic(nq))
        deadline = LOCALIZE_DEADLINE if (req.action == Action.NEXT_QUESTION and language != "en") else COMPOSE_DEADLINE
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.composer.compose, req), deadline)
        except asyncio.TimeoutError:
            logger.warning("composer missed its %.0fs deadline interview=%s - using deterministic wording",
                           deadline, self.runner.interview_id)
            return self.composer.fallback(req, "deadline_exceeded")
        finally:
            turn_trace.mark("compose_end")

    async def _localize_upcoming(self, language: str, *, wait: bool, limit: int | None = None) -> None:
        """Render the current and remaining questions into `language` in one call and remember them. With wait=False it
        runs in the background (so the interview is never held up by it)."""
        language = normalize_language(language)
        r = self.runner
        todo = [q for q in r.questions[r.idx:] if q.get("question_text") and (str(q.get("id")), language) not in self._localized]
        if limit is not None:
            todo = todo[:limit]
        if language == "en" or not todo:
            return

        async def work() -> None:
            try:
                done = await asyncio.wait_for(
                    asyncio.to_thread(self.composer.localize_questions, todo, language, self.role), LOCALIZE_BATCH_DEADLINE)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001 - best effort; questions fall back to on-demand localisation
                logger.warning("pre-translating questions failed interview=%s language=%s", r.interview_id, language)
                return
            for qid, text in done.items():
                self._localized[(qid, language)] = text
            logger.info("pre-translated %d/%d question(s) into %s interview=%s", len(done), len(todo), language, r.interview_id)

        if wait:
            await work()
        else:
            task = asyncio.get_running_loop().create_task(work())
            self._localizing.add(task)
            task.add_done_callback(self._localizing.discard)

    def _check(self, generation: int) -> None:
        if self.runner.is_stale(generation):
            raise _Stale()

    async def _say(self, speech: Speech) -> None:
        r = self.runner
        if r.terminal and speech.action != "INTERVIEW_COMPLETE":
            return
        self._last_spoken = speech.text
        self.turns.go(TurnPhase.COMPLETED if speech.action == "INTERVIEW_COMPLETE" else TurnPhase.QUESTION_DELIVERED)
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
        r.record_turn_nowait(q["id"] if q else "", "candidate", self._turn_original or text, is_followup=followup,
                             intent=intent.value if isinstance(intent, Intent) else intent,
                             question_text=(q or {}).get("question_text") or "")

    # ---------------------------------------------------------------- turn routing
    async def _handle(self, text: str, generation: int) -> None:
        r = self.runner
        self._turn_original = text
        try:
            await self._handle_turn(text, generation)
        finally:
            self._turn_original = None

    async def _handle_turn(self, text: str, generation: int) -> None:
        r = self.runner
        # "Can you speak in Hindi?" changes the interview language for the rest of the session.
        requested_language = detect_language_request(text, current=self.language)
        if requested_language and self.supported_languages and requested_language not in self.supported_languages:
            return await self._unsupported_language(text, requested=True)
        if requested_language:
            return await self._switch_language(requested_language, text, generation)
        # Asked for the language already in use ("Can I speak in Hindi?" while the interview is in Hindi): there is
        # nothing to switch, and it is not an answer - never score it. Just put the current question back.
        if detect_language_request(text, current=None):
            return await self._repeat(text, Intent.LANGUAGE_REQUEST)
        # A request for a language this interview cannot conduct (Urdu, Tamil...), or a transcript in a script no supported
        # language uses (Whisper labelling Hindustani as Urdu, or genuinely unsupported speech): never an answer, never
        # quoted back, never allowed to move the interview language. The original transcript is still saved as spoken.
        if detect_unsupported_language_request(text):
            return await self._unsupported_language(text, requested=True)
        if has_unexpected_script(text):
            return await self._unsupported_language(text, requested=False)
        # Whisper renders the Hindi words of a Hinglish sentence in Urdu script, even inside an otherwise English
        # transcript. Judge, caption and report work from the transcript WITHOUT those runs; the original stays on record.
        cleaned, removed = clean_transcript(text)
        if removed:
            self._event("unexpected_script_removed", letters=removed)
            if len(cleaned.split()) < MIN_WORDS_AFTER_CLEANING:
                return await self._unsupported_language(text, requested=False)
            text = cleaned
        # The candidate simply started answering in another language: follow them quietly, then handle what they said as
        # a normal answer (no announcement, nothing re-asked) - but never against an explicit request.
        if self._follower is not None and not self._language_locked:
            followed = self._follower.observe(text, self.language)
            if followed and followed != self.language and (
                    not self.supported_languages or followed in self.supported_languages):
                await self._apply_language(followed, generation, automatic=True)
        q = r.current_question
        classification = classify(text, context=self._context_for(q))
        intent = classification.intent
        self._event("turn_classified", intent=intent.value, rule=classification.rule or "-",
                    suspicious=classification.suspicious)
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

    async def _unsupported_language(self, text: str, *, requested: bool) -> None:
        """Fixed reply; the interview language, the question and the question timer are all left exactly as they were."""
        r = self.runner
        q = r.current_question
        self._record_candidate(text, Intent.UNSUPPORTED_LANGUAGE)
        self._event("unsupported_language", requested=requested)
        in_question = q is not None and r.small_talk_done >= r.small_talk_rounds and q.get("question_text")
        if requested:
            spoken = refusals.UNSUPPORTED_LANGUAGE[self.language]
            if in_question:
                spoken = f"{spoken} {r.last_question_text or q['question_text']}"
        else:
            spoken = refusals.DIDNT_CATCH_LANGUAGE[self.language]
        await self._say(Speech(spoken, "REDIRECT", (q or {}).get("id", ""), (q or {}).get("question_text", "")))

    async def _end_early(self, text: str) -> None:
        r = self.runner
        self._record_candidate(text, Intent.END_INTERVIEW)
        r.end_early()
        await self._say(Speech(refusals.closing(self.language, early=True), "INTERVIEW_COMPLETE",
                               interruptible=False))
        await r.complete()

    async def _repeat(self, text: str, intent: Intent = Intent.REPEAT) -> None:
        r = self.runner
        q = r.current_question
        self._record_candidate(text, intent)
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

        # An answer can arrive in several committed turns (a long pause mid-thought). The judge must see everything the
        # candidate has said to THIS question, not just the last fragment, or the early fragment is judged as a thin
        # answer and a follow-up (or a redirect) lands on top of a candidate who is still explaining.
        judged_text = " ".join([*state.answer_parts, text])[-MAX_JUDGE_CHARS:]
        evaluation = await self._judge(q, judged_text, state.topic)
        self._check(generation)

        # Second line of defence behind the deterministic guard: phrasings the patterns missed. A substantive,
        # on-topic answer is never refused on the model's say-so alone.
        reclassified = restricted_intent_from_judge(evaluation, text, self._context_for(q))
        if reclassified == Intent.REPEAT:
            return await self._repeat(text)
        if reclassified is not None:
            return await self._refuse(reclassified, text, generation)

        self._record_candidate(text, Intent.ANSWER, followup=state.followup_count > 0)
        self.turns.go(TurnPhase.ANSWER_SAVED)
        state.answer_parts.append(text)
        state.non_answer_count = 0
        self._event("answer_finalized", parts=len(state.answer_parts), coverage=round(evaluation.coverage_score, 2))
        self.turns.go(TurnPhase.DECIDING)
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

    # ---------------------------------------------------------------- language
    async def _apply_language(self, new_language: str, generation: int, *, automatic: bool = False) -> None:
        """Move the rest of the interview into `new_language`: writing (composer), speaking and listening (via the agent
        callback), and what is saved for a reconnect. The remaining questions are translated ahead of time in the
        background so none of them waits."""
        r = self.runner
        previous = self.language
        self.language = normalize_language(new_language)
        if self._follower is not None:
            self._follower.note_switch()
        logger.info("language %s interview=%s %s -> %s", "followed" if automatic else "switch", r.interview_id, previous, self.language)
        if self.on_language_change is not None:
            try:
                result = self.on_language_change(self.language)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - audio retuning must never end the interview
                logger.exception("applying the new language to speech failed interview=%s", r.interview_id)
        self._check(generation)
        r.record_language_nowait(self.language)
        if automatic:
            await self._localize_upcoming(self.language, wait=False)

    async def _switch_language(self, new_language: str, text: str, generation: int) -> None:
        """The candidate asked to change language: do it, say so, and ask the current question again in the new language.
        Nothing is counted as an answer, a follow-up or a violation."""
        r = self.runner
        q = r.current_question
        self._record_candidate(text, Intent.LANGUAGE_REQUEST)
        self._language_locked = True                 # an explicit choice is sticky: only another request changes it
        await self._apply_language(new_language, generation)
        confirmation = refusals.LANGUAGE_SWITCHED[self.language]
        asking_now = not (q is None or not q.get("question_text") or r.small_talk_done < r.small_talk_rounds)
        # While the confirmation is spoken, translate ONLY the question about to be asked, so the candidate is not left
        # waiting on the whole remaining plan (a batch of every later question could take up to
        # LOCALIZE_BATCH_DEADLINE). The rest is translated in the background, ready before it is needed.
        translating = asyncio.get_running_loop().create_task(self._localize_upcoming(self.language, wait=True, limit=1))
        try:
            await self._say(Speech(confirmation, "LANGUAGE_SWITCH", q["id"] if q else "", ""))
            self._check(generation)
        finally:
            await translating
        await self._localize_upcoming(self.language, wait=False)       # the remaining questions, off the critical path
        if not asking_now:
            return
        self._check(generation)
        composed = await self._compose(ComposeRequest(
            action=Action.NEXT_QUESTION, question=q, next_question=q, language=self.language, role=self.role,
            experience_level=self.experience_level, plain_language=self.plain_language))
        self._check(generation)
        r.last_question_text = composed.question_only or composed.text
        await self._say(Speech(composed.text, "LANGUAGE_SWITCH", q["id"], q["question_text"]))

    # ---------------------------------------------------------------- moving on
    async def _advance_and_ask(self, generation: int, mentioned: list[str], prefix: str = "") -> None:
        r = self.runner
        if self.turns.turn_number and self._advanced_turn == self.turns.turn_number:
            # One finalized turn advances the interview at most once, whatever else is racing.
            self._event("advance_ignored_duplicate")
            return
        self._advanced_turn = self.turns.turn_number
        self.turns.go(TurnPhase.DECIDING)
        previous = r.current_question
        # The answer (and its coverage record) must be saved before the interview moves past its question, so a
        # reconnect or a crash right after the advance cannot lose it. Bounded: a slow API never freezes the interview.
        await r.flush_pending(timeout=2.0)
        self._check(generation)
        from_idx = r.idx
        r.advance()
        nq = r.current_question
        self._event("question_transition", from_idx=from_idx, to_idx=r.idx, final=nq is None)
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
