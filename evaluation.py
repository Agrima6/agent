"""Turns persisted transcript rows into the per-question results the scorer aggregates.

Pure code over plain objects (anything with question_id / speaker / text / intent / action), so the rules below are
unit-testable without a database or an LLM. The LLM scorers are passed in.

Rules (each one is a verified failure mode - see WORKMATEIQ_AI_AGENT_ROOT_CAUSE_REPORT.md):

  * Only turns that are an attempt to answer are scored (hint/answer requests, repeats, language requests, skips,
    injection attempts and "end the interview" are non-answers - interaction_guard.NON_ANSWER_INTENTS).
  * Every persisted answer is counted exactly once: an identical candidate line repeated for the same question
    (a re-delivered event) is dropped before scoring.
  * A question that was ASKED but not answered counts as 0 for its competencies and is reported as NOT_ANSWERED, so a
    candidate cannot raise their average by skipping the hard questions. A question that was never asked is simply
    not assessed (it neither helps nor hurts, and shows up in the coverage/completeness fields instead).
  * Introductions are context, not assessed answers: they never count toward "answered questions".
  * The communication score of each answered question also feeds the role's "communication" competency, which is
    otherwise never assessed. All scores are 0-100.
"""
from interaction_guard import NON_ANSWER_INTENTS

INTRO_TYPES = ("introduction", "candidate_introduction")
_NON_ANSWER_VALUES = {i.value for i in NON_ANSWER_INTENTS}
# Interviewer actions that put a question to the candidate.
_ASKING_ACTIONS = {"NEXT_QUESTION", "FOLLOW_UP", "CLARIFICATION", "REPEAT", "LANGUAGE_SWITCH"}
UNANSWERED_EXPLANATION = ("This question was asked but the candidate did not give an answer to it "
                          "(skipped, declined, or the interview ended first).")


def _clamp(value, default=0.0) -> float:
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return default


def is_answer_turn(turn) -> bool:
    return turn.speaker == "candidate" and (turn.intent or "answer") not in _NON_ANSWER_VALUES and bool((turn.text or "").strip())


def answer_text_for(turns, question_id: str) -> str:
    kept: list[str] = []
    for turn in turns:
        if turn.question_id != question_id or not is_answer_turn(turn):
            continue
        text = turn.text.strip()
        if kept and kept[-1] == text:        # the same line delivered twice
            continue
        kept.append(text)
    return "\n".join(kept)


def asked_question_ids(turns) -> set[str]:
    asked = set()
    for turn in turns:
        if not turn.question_id:
            continue
        if turn.speaker == "agent" and (turn.action or "") in _ASKING_ACTIONS:
            asked.add(turn.question_id)
        elif turn.speaker == "candidate":
            asked.add(turn.question_id)
    return asked


def ended_early(turns) -> bool:
    return any(t.speaker == "candidate" and (t.intent or "") == "end_interview" for t in turns)


def collect_question_results(plan: dict, turns, *, score_content, score_communication) -> dict:
    """-> {"per_question_scores", "question_reports", "asked_ids", "answered", "asked"}"""
    asked_ids = asked_question_ids(turns)
    per_question_scores: list[dict] = []
    reports: list[dict] = []
    answered = asked = 0

    for q in (plan or {}).get("questions", []):
        if not q.get("question_text"):
            continue
        qid = q["id"]
        is_intro = q.get("type") in INTRO_TYPES
        text = answer_text_for(turns, qid)
        competencies = list(q.get("competencies") or [])

        if text:
            content = score_content(q["question_text"], q.get("expected_topics", []), text)
            comm = score_communication(text)
            content_score = _clamp(content.get("score"))
            reports.append({"question_id": qid, "question": q["question_text"], "candidate_answer": text,
                            "content_score": content, "communication_score": comm, "answered": True,
                            "kind": "introduction" if is_intro else "question"})
            if not is_intro:
                answered += 1
                asked += 1
                per_question_scores.append({"competencies": competencies, "content_score": content_score})
                if isinstance(comm, dict) and comm.get("score") is not None:
                    per_question_scores.append({"competencies": ["communication"], "content_score": _clamp(comm.get("score"))})
        elif qid in asked_ids and not is_intro:
            asked += 1
            reports.append({"question_id": qid, "question": q["question_text"], "candidate_answer": "",
                            "content_score": {"score": 0, "dimensions": {}, "covered_topics": [], "missing_topics": [],
                                              "strengths": [], "improvement_areas": [],
                                              "explanation": UNANSWERED_EXPLANATION},
                            "communication_score": None, "answered": False, "status": "NOT_ANSWERED", "kind": "question"})
            per_question_scores.append({"competencies": competencies, "content_score": 0.0})

    return {"per_question_scores": per_question_scores, "question_reports": reports, "asked_ids": asked_ids,
            "answered": answered, "asked": asked}
