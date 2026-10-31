"""Server-authoritative interview progression and persistence.

The runner owns the interview's STATE (which question, how many follow-ups, phase, timers, whether
it has ended). No LLM ever writes to it: the conductor calls advance()/follow-up accounting after a
deterministic policy decision, and every change is reported to the API so the server holds the
authoritative copy. A restarted agent worker calls restore() with that copy and resumes instead of
starting the interview over.

Every network call here is fire-and-forget (the live voice loop must never wait on persistence),
except complete(), which first lets in-flight writes land so the final answer is never lost.
"""
import asyncio
import json
import logging
import time

import httpx

from config import AGENT_SERVICE_KEY, API_BASE_URL
import interview_flow as flow
from focus import followup_limit
from policy import QuestionState

logger = logging.getLogger("interview-runner")

DEFAULT_QUESTION_SECONDS = 150
TIME_BUDGET_FRACTION = 0.9      # stop asking follow-ups once 90% of the interview's time is used
_AUTH_HEADERS = {"X-Agent-Key": AGENT_SERVICE_KEY} if AGENT_SERVICE_KEY else {}


def question_topic(question: dict) -> str:
    return (question.get("topic") or (question.get("competencies") or [""])[0] or "").replace("_", " ").strip()


class InterviewRunner:
    def __init__(self, http: httpx.AsyncClient, interview_id: str, plan: dict, *, candidate_name: str = "",
                 coverage_threshold: float = 0.7, max_followups_per_question: int = 2,
                 depth_probe_enabled: bool = True, duration_minutes: int = 30, small_talk_rounds: int = 2,
                 candidate_facts: dict | None = None, clock=time.time):
        self.http = http
        self.interview_id = interview_id
        # Plan order for the structured opening (see interview_flow.py): intro, then 1-2 resume-grounded questions (or the
        # single generic background question), then everything else. Deterministic, so a restarted worker rebuilds the
        # same order from the same stored plan.
        self.questions, self.opening_count = flow.arrange_questions(plan["questions"])
        self.opening_ids = {q["id"] for q in self.questions[1:1 + self.opening_count]} if self.questions and \
            self.questions[0].get("type") == "introduction" else {q["id"] for q in self.questions[:self.opening_count]}
        self.candidate_facts = dict(candidate_facts or {})     # verified against the resume by the API; spoken in the intro
        self.candidate_name = candidate_name
        self.coverage_threshold = coverage_threshold
        self.max_followups_per_question = max_followups_per_question
        self.depth_probe_enabled = depth_probe_enabled
        self.duration_minutes = duration_minutes
        self.small_talk_rounds = small_talk_rounds
        self._clock = clock
        self.started_at = clock()
        self.idx = 0
        self.small_talk_done = 0
        # Where the candidate is in the structured opening (interview_flow.STAGE_ORDER). Changed only by the conductor.
        # small_talk_rounds=0 has always meant "no opening conversation": the interview starts at its first question.
        self.stage = flow.INTRODUCTION if small_talk_rounds > 0 else flow.MAIN_INTERVIEW
        self.readiness_confirmed = False
        self.clarifications = 0           # doubts answered so far
        self.unclear = 0                  # unclear replies in the current stage
        self.env_retries = 0
        self.env_resolved = None          # True / False once the environment check has been decided
        self.preflight_nudged = 0         # silence prompts spoken in the current opening stage (at most one)
        self.resumed = False
        self.phase = "created"
        self.state = self._new_question_state(self.questions[0]) if self.questions else None
        # Main questions and follow-ups already asked (used by code for duplicate checks only; never
        # sent to an LLM) and the wording of the current main question (used by "repeat that").
        self.asked_texts: list[str] = []
        self.last_question_text = ""
        # Set once the interview job may exit: after /complete, or when the session ends early.
        self.done = asyncio.Event()
        # Generation fence: anything that may still be in flight when the interview ends (the
        # judge and composer run in threads) captures `generation` first and must discard its
        # result if `terminal` or `generation` changed by the time it returns.
        self.generation = 0
        self.terminal = False
        self._pending: set[asyncio.Task] = set()
        # Optional hook (set by the agent job): called with a small dict whenever the question/phase changes, so the
        # candidate's screen can show the current question, progress and a per-question timer. Never raises.
        self.on_progress = None

    # ------------------------------------------------------------------ state
    @property
    def in_preflight(self) -> bool:
        """True while the interviewer is still introducing, clarifying, confirming readiness or checking the environment:
        no question is on the table, so nothing the candidate says is an answer."""
        return self.stage in flow.PREFLIGHT_STAGES

    def is_opening_index(self, idx: int) -> bool:
        first = 1 if self.questions and self.questions[0].get("type") == "introduction" else 0
        return first <= idx < first + self.opening_count

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self.unclear = 0
        self.preflight_nudged = 0
        if stage in (flow.RESUME_QUESTIONS, flow.MAIN_INTERVIEW):
            self.small_talk_done = max(self.small_talk_done, self.small_talk_rounds)   # legacy progress readers

    def _new_question_state(self, question: dict) -> QuestionState:
        priority = question.get("focus_priority")
        max_followups = followup_limit(self.max_followups_per_question, priority)
        if question.get("id") in self.opening_ids:
            max_followups = min(max_followups, 1)        # the opening stage stays short: at most one follow-up each
        return QuestionState(
            question_id=question["id"], topic=question_topic(question),
            focus=question.get("focus") or "", focus_priority=priority or "",
            coverage_threshold=self.coverage_threshold,
            max_followups=max_followups,
            max_seconds=int(question.get("time_limit") or DEFAULT_QUESTION_SECONDS),
            depth_probe_enabled=self.depth_probe_enabled,
        )

    @property
    def current_question(self) -> dict | None:
        return self.questions[self.idx] if self.idx < len(self.questions) else None

    def advance(self) -> None:
        self.idx += 1
        question = self.current_question
        if question:
            self.state = self._new_question_state(question)

    def end_early(self) -> None:
        """Jump past all remaining questions so the next step delivers the closing."""
        self.idx = len(self.questions)

    def time_exhausted(self) -> bool:
        return self._clock() - self.started_at >= self.duration_minutes * 60 * TIME_BUDGET_FRACTION

    def is_stale(self, captured_generation: int) -> bool:
        return self.terminal or captured_generation != self.generation

    def progress_snapshot(self) -> dict:
        return {"question_index": self.idx, "followup_count": self.state.followup_count if self.state else 0,
                "small_talk_done": self.small_talk_done, "phase": self.phase, "stage": self.stage,
                "readiness": int(self.readiness_confirmed)}

    def restore(self, progress: dict | None) -> bool:
        """Resume from server-held progress. Returns True if this was a real resume (the candidate had already got past
        the introduction) - a fresh or barely-started interview restarts cleanly.

        A worker that restarts while the candidate is still in the opening (clarifying, confirming readiness, answering
        a resume question, the environment check) continues at that stage; it never introduces itself again."""
        if not progress:
            return False
        stage = progress.get("stage")
        index = int(progress.get("question_index", 0))
        if stage in flow.STAGE_ORDER and stage != flow.INTRODUCTION:
            self.stage = stage
            self.readiness_confirmed = bool(progress.get("readiness")) or flow.stage_rank(stage) > flow.stage_rank(flow.READINESS)
            if index < 1 and stage in flow.PREFLIGHT_STAGES and stage != flow.ENVIRONMENT_CHECK:
                self.idx = 0
                self.state = self._new_question_state(self.questions[0]) if self.questions else None
                self.resumed = True
                self.phase = "resumed"
                return True
        elif index < 1:
            return False
        else:
            # A stored interview from before the structured opening: it was already in its questions.
            self.stage = flow.MAIN_INTERVIEW
        self.idx = min(max(index, 0), len(self.questions))
        self.small_talk_done = max(int(progress.get("small_talk_done", 0)), self.small_talk_rounds) \
            if self.stage in (flow.RESUME_QUESTIONS, flow.MAIN_INTERVIEW) else int(progress.get("small_talk_done", 0))
        question = self.current_question
        if question is None:
            return False
        self.state = self._new_question_state(question)
        self.state.followup_count = min(int(progress.get("followup_count", 0)), self.state.max_followups)
        if self.stage == flow.ENVIRONMENT_CHECK:
            self.env_resolved = None
        self.resumed = True
        self.phase = "resumed"
        return True

    # ------------------------------------------------------------ persistence
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def flush_pending(self, timeout: float = 3.0) -> None:
        if self._pending:
            await asyncio.wait(list(self._pending), timeout=timeout)

    async def _post(self, path: str, data: dict, what: str) -> httpx.Response | None:
        try:
            resp = await self.http.post(f"{API_BASE_URL}/v1/interviews/{self.interview_id}/{path}",
                                        data=data, headers=_AUTH_HEADERS)
        except Exception:
            logger.exception("%s request failed", what)
            return None
        if resp.status_code == 409:
            # The server says the interview already ended (generation fence): stop persisting.
            self.terminal = True
        return resp

    async def start(self) -> None:
        try:
            await self.http.post(f"{API_BASE_URL}/v1/interviews/{self.interview_id}/start", headers=_AUTH_HEADERS)
        except Exception:
            logger.exception("failed to mark interview IN_PROGRESS (non-fatal)")

    def record_turn_nowait(self, question_id: str, speaker: str, text: str, *, is_followup: bool = False,
                           intent: str = "", action: str = "", question_text: str = "") -> None:
        # Checked here, synchronously: a turn recorded just before complete() must still be sent
        # even though the task itself only starts running after `terminal` flips.
        if self.terminal:
            return
        self._spawn(self._post("turns", {
            "question_id": question_id, "question_text": question_text, "speaker": speaker, "text": text,
            "is_followup": is_followup, "intent": intent, "action": action}, "record_turn"))

    def record_coverage_nowait(self, question_id: str, evaluation: dict, action: str, followup_count: int) -> None:
        if self.terminal:
            return
        self._spawn(self._post("coverage-results", {
            "question_id": question_id,
            "coverage_score": evaluation.get("coverage_score", 0.0),
            "covered_topics": json.dumps(evaluation.get("covered_topics", [])),
            "missing_topics": json.dumps(evaluation.get("missing_topics", [])),
            "evaluation": json.dumps(evaluation), "action": action, "followup_count": followup_count,
        }, "record_coverage_result"))

    def record_language_nowait(self, language: str) -> None:
        """Remember the language the interview is now in, so a reconnect or restart resumes in it."""
        if self.terminal:
            return
        self._spawn(self._post("language", {"language": language}, "set_language"))

    def persist_progress_nowait(self, phase: str) -> None:
        if self.terminal:
            return
        self.phase = phase
        snap = self.progress_snapshot()
        if self.on_progress is not None:
            question = self.current_question
            try:
                # During the opening (introduction, doubts, readiness, environment check) no question is on the table, so the
                # candidate's screen must not show one.
                shown = None if self.in_preflight else question
                self.on_progress({"question_id": shown["id"] if shown else None, "question_index": self.idx,
                                  "total": len(self.questions), "followup_count": snap["followup_count"], "phase": phase})
            except Exception:  # noqa: BLE001 - a UI hook must never affect the interview
                pass
        self._spawn(self._post("progress", {k: v for k, v in snap.items()}, "record_progress"))

    def record_tts_metadata_nowait(self, metadata: dict) -> None:
        self._spawn(self._post("tts-metadata", {"metadata": json.dumps(metadata)}, "record_tts_metadata"))

    async def complete(self) -> None:
        """Atomic terminal transition: flip local state BEFORE any network call, so a judge or
        composer still running in a thread sees terminal=True (via is_stale) immediately. In-flight
        writes are then allowed to land before /complete, so the final answer is never lost."""
        if self.terminal:
            return
        self.terminal = True
        self.generation += 1
        try:
            await self.flush_pending()
            await self.http.post(f"{API_BASE_URL}/v1/interviews/{self.interview_id}/complete", headers=_AUTH_HEADERS)
        except Exception:
            logger.exception("complete request failed")
        finally:
            self.done.set()
