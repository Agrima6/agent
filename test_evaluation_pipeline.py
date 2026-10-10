"""Regression tests for the evaluation pipeline findings (coverage counts, language requests, scales and aggregation,
report states, exactly-once answers, candidate/HR separation). Each test names the defect it pins down."""
import types

import pytest
from fastapi.testclient import TestClient

import api
import evaluation
import refusals
import scoring
import states
import test_end_interview as base
from db import Candidate, Interview, InterviewTurn, Role, transition_interview
from intelligence_report import build_intelligence, serves_focus
from test_conductor import Harness, Q1, Q2, Q3, lead_in, strong

KEY = "evaluation-test-key"
H = {"X-Agent-Key": KEY}


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ----------------------------------------------------------------------------------------------- helpers
def turn(question_id, speaker, text, intent=None, action=None):
    return types.SimpleNamespace(question_id=question_id, speaker=speaker, text=text, intent=intent, action=action)


def fake_content(score):
    return lambda question, expected, text: {"score": score, "dimensions": {}, "explanation": "e"}


def fake_comm(score):
    return lambda text: {"score": score, "dimensions": {}, "explanation": "e"}


ROLE = [{"key": "technical_depth", "weight": 0.30}, {"key": "problem_solving", "weight": 0.25},
        {"key": "practical_application", "weight": 0.20}, {"key": "communication", "weight": 0.15},
        {"key": "ownership", "weight": 0.10}]

# The shape reported: an introduction, two dynamic scenario questions (competencies, but no focus tag), a behavioural
# question that a shared word ("technical") had tagged "Technical Depth", and one HR question.
PLAN = {
    "focus_areas": [
        {"id": "f_1", "name": "Technical Depth", "priority": "HIGH", "sources": ["ROLE"]},
        {"id": "f_2", "name": "Problem Solving", "priority": "HIGH", "sources": ["ROLE"]},
        {"id": "f_3", "name": "Ownership", "priority": "MEDIUM", "sources": ["ROLE"]},
        {"id": "f_4", "name": "Practical Application", "priority": "MEDIUM", "sources": ["ROLE"]},
    ],
    "questions": [
        {"id": "p_intro", "type": "introduction", "question_text": None},
        {"id": "p_candidate_intro", "type": "candidate_introduction", "question_text": "Tell me about yourself"},
        {"id": "q_dynamic_1", "type": "scenario", "question_text": "Design a prototype that summarises tickets",
         "competencies": ["problem_solving", "technical_depth"], "expected_topics": ["data", "demo"]},
        {"id": "q_dynamic_2", "type": "scenario", "question_text": "A demo leaks confidential policy text",
         "competencies": ["problem_solving", "technical_depth"], "expected_topics": ["contain", "prevent"]},
        {"id": "q_general_1", "type": "behavioral", "question_text": "Disagreed with a teammate about a technical decision",
         "competencies": ["ownership", "collaboration"], "focus": "Technical Depth", "expected_topics": ["resolution"]},
        {"id": "hr_1", "type": "hr", "question_text": "Walk me through a production incident",
         "competencies": ["problem_solving"], "expected_topics": ["root cause"]},
    ],
}
ALL_ANSWERED = [
    turn("p_candidate_intro", "candidate", "I work on backend systems", "answer"),
    turn("q_dynamic_1", "agent", "Design a prototype…", action="NEXT_QUESTION"),
    turn("q_dynamic_1", "candidate", "I would collect tickets, build a demo and handle errors", "answer"),
    turn("q_dynamic_2", "agent", "A demo leaks…", action="NEXT_QUESTION"),
    turn("q_dynamic_2", "candidate", "I would contain it, apologise and add guardrails", "answer"),
    turn("q_general_1", "agent", "Disagreed…", action="NEXT_QUESTION"),
    turn("q_general_1", "candidate", "We talked it through and agreed on a design", "answer"),
    turn("hr_1", "agent", "Walk me through…", action="NEXT_QUESTION"),
    turn("hr_1", "candidate", "A queue backed up; I found the root cause in the consumer", "answer"),
]


def intelligence_for(turns, plan=PLAN, content=70):
    results = evaluation.collect_question_results(plan, turns, score_content=fake_content(content), score_communication=fake_comm(60))
    return results, build_intelligence(plan, results["question_reports"], [], asked_ids=results["asked_ids"])


def area(intel, name):
    return next(f for f in intel["hr_focus_coverage"] if f["focus"] == name)


# ============================================================ 1. coverage counts (the reported mismatch)
def test_a_competency_tagged_dynamic_question_counts_toward_its_focus_area():
    """Before: Problem Solving was NOT_ASKED although three answered questions declare problem_solving."""
    _, intel = intelligence_for(ALL_ANSWERED)
    ps = area(intel, "Problem Solving")
    assert ps["status"] != "NOT_ASKED"
    assert (ps["questions_planned"], ps["questions_asked"], ps["questions_answered"]) == (3, 3, 3)
    assert "Problem Solving" not in intel["unasked_high_priority"]


def test_counts_are_planned_asked_answered_and_consistent_with_each_other():
    _, intel = intelligence_for(ALL_ANSWERED)
    td = area(intel, "Technical Depth")
    # two dynamic questions (by competency) + the behavioural one the old tag put there
    assert (td["questions_planned"], td["questions_asked"], td["questions_answered"]) == (3, 3, 3)
    for f in intel["hr_focus_coverage"]:
        assert f["questions_answered"] <= f["questions_asked"] <= f["questions_planned"]


def test_a_planned_question_that_was_never_asked_is_not_counted_as_asked():
    turns = [t for t in ALL_ANSWERED if t.question_id in ("p_candidate_intro", "q_dynamic_1")]
    _, intel = intelligence_for(turns)
    ps = area(intel, "Problem Solving")
    assert (ps["questions_planned"], ps["questions_asked"], ps["questions_answered"]) == (3, 1, 1)
    assert area(intel, "Practical Application")["status"] == "NOT_ASKED"
    assert area(intel, "Ownership")["status"] == "NOT_ASKED"


def test_asked_but_unanswered_is_not_answered_rather_than_not_asked():
    turns = [turn("q_dynamic_1", "agent", "Design…", action="NEXT_QUESTION"),
             turn("q_dynamic_1", "candidate", "Move to the next question", "skip")]
    _, intel = intelligence_for(turns)
    assert area(intel, "Problem Solving")["status"] == "NOT_ANSWERED"
    assert area(intel, "Problem Solving")["questions_asked"] == 1


def test_plans_stored_without_focus_tags_still_map_by_competency():
    plan = {**PLAN, "questions": [{k: v for k, v in q.items() if k != "focus"} for q in PLAN["questions"]]}
    _, intel = intelligence_for(ALL_ANSWERED, plan=plan)
    assert area(intel, "Problem Solving")["questions_answered"] == 3
    assert area(intel, "Technical Depth")["questions_answered"] == 2        # only the two dynamic ones, no stray tag


def test_the_introduction_never_counts_toward_a_focus_area_or_the_answered_total():
    results, intel = intelligence_for(ALL_ANSWERED)
    assert not serves_focus(PLAN["questions"][1], "Technical Depth")
    assert results["answered"] == 4                                          # 2 dynamic + behavioural + HR; not the intro
    assert any(r["question_id"] == "p_candidate_intro" and r["kind"] == "introduction" for r in results["question_reports"])


# ============================================================ 2. language-change requests
pytestmark_async = pytest.mark.anyio


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


@pytest.mark.anyio
@pytest.mark.parametrize("request_text", ["Can I speak in Hindi?", "Can you speak in Hindi please", "Hindi mein baat karo"])
async def test_a_language_request_on_a_technical_question_repeats_that_question_and_does_not_advance(harness_factory, request_text):
    h = harness_factory(composer_responses=[lead_in("")], evaluations=[strong()])
    state_before = (h.runner.idx, h.runner.state.followup_count, h.runner.state.started_at)
    last = await h.say(request_text)
    assert h.conductor.language == "hi"
    assert last.action == "LANGUAGE_SWITCH" and last.question_id == Q1["id"]      # the CURRENT question, asked again
    assert Q1["question_text"] in last.text or last.question_text == Q1["question_text"]
    assert (h.runner.idx, h.runner.state.followup_count, h.runner.state.started_at) == state_before
    assert h.judge_calls == [] and h.posted("coverage-results") == []              # never scored, never finalised as an answer
    assert h.posted("turns")[0]["intent"] == "language_request"
    assert h.runner.state.answer_parts == []                                       # not part of the answer to the question


@pytest.mark.anyio
async def test_a_request_for_the_language_already_in_use_repeats_the_question_too(harness_factory):
    h = harness_factory(language="hi", composer_responses=[], evaluations=[])
    last = await h.say("Can I speak in Hindi?")
    assert last.action == "REPEAT" and last.question_id == Q1["id"] and h.runner.idx == 0
    assert h.judge_calls == [] and h.posted("turns")[0]["intent"] == "language_request"


def test_a_language_request_is_never_an_answer_in_the_evaluation_either():
    """Even if an older agent stored it under the technical question, the scorer must not treat it as an answer."""
    turns = [turn("q_dynamic_1", "agent", "Design…", action="NEXT_QUESTION"),
             turn("q_dynamic_1", "candidate", "Can I speak in Hindi?", "language_request")]
    results = evaluation.collect_question_results(PLAN, turns, score_content=fake_content(99), score_communication=fake_comm(99))
    entry = results["question_reports"][0]
    assert entry["answered"] is False and entry["content_score"]["score"] == 0 and entry["candidate_answer"] == ""
    assert results["answered"] == 0


# ============================================================ 3. scales, weights, aggregation, missing, incomplete
def test_scores_are_clamped_to_0_100_and_communication_feeds_the_communication_competency():
    results = evaluation.collect_question_results(
        {"questions": [PLAN["questions"][2]]}, [turn("q_dynamic_1", "agent", "x", action="NEXT_QUESTION"),
                                                turn("q_dynamic_1", "candidate", "an answer", "answer")],
        score_content=fake_content(250), score_communication=fake_comm(-5))
    scores = results["per_question_scores"]
    assert {"competencies": ["problem_solving", "technical_depth"], "content_score": 100.0} in scores
    assert {"competencies": ["communication"], "content_score": 0.0} in scores
    agg = scoring.aggregate_final_score(scores, ROLE)
    assert set(agg["competency_scores"]) == {"problem_solving", "technical_depth", "communication"}


def test_the_reported_numbers_reproduce_and_the_normalisation_is_documented_by_the_test():
    """Question scores from a stored interview: resume 15, dynamic 0 and 8, behavioural 12 (intro 0, no competencies).
    competency = mean of its questions' scores; final = sum(score * weight) / assessed weight."""
    per_q = [{"competencies": ["technical_depth", "problem_solving"], "content_score": s} for s in (15, 0, 8)]
    per_q.append({"competencies": ["ownership", "collaboration"], "content_score": 12})
    agg = scoring.aggregate_final_score(per_q, ROLE)
    assert round(agg["competency_scores"]["technical_depth"], 2) == 7.67
    assert agg["competency_scores"]["ownership"] == 12
    legacy = 7.6667 * 0.30 + 7.6667 * 0.25 + 12 * 0.10          # old formula: unassessed weight counted as zero
    assert round(legacy, 1) == 5.4
    assert agg["assessed_weight_share"] == 0.65                  # .30 + .25 + .10 of the role's weight
    assert agg["final_score"] == round(legacy / 0.65, 1) == 8.3   # new: normalised over what was actually assessed


def test_unasked_competencies_are_excluded_and_asked_unanswered_ones_count_as_zero():
    turns = [turn("q_dynamic_1", "agent", "x", action="NEXT_QUESTION"),
             turn("q_dynamic_1", "candidate", "Skip this", "skip"),             # asked, not answered -> 0
             turn("q_general_1", "agent", "x", action="NEXT_QUESTION"),
             turn("q_general_1", "candidate", "We agreed on a design together", "answer")]
    results = evaluation.collect_question_results(PLAN, turns, score_content=fake_content(80), score_communication=fake_comm(80))
    agg = scoring.aggregate_final_score(results["per_question_scores"], ROLE)
    assert agg["competency_scores"]["technical_depth"] == 0.0      # asked and skipped
    assert agg["competency_scores"]["ownership"] == 80.0
    assert "practical_application" not in agg["competency_scores"]  # never asked: absent, not zero
    assert results["asked"] == 2 and results["answered"] == 1


def test_skipping_hard_questions_cannot_raise_the_average():
    answered_only = [{"competencies": ["technical_depth"], "content_score": 80}]
    with_skip = answered_only + [{"competencies": ["technical_depth"], "content_score": 0.0}]
    assert scoring.aggregate_final_score(with_skip, ROLE)["final_score"] < scoring.aggregate_final_score(answered_only, ROLE)["final_score"]


def test_an_incomplete_interview_is_flagged_not_presented_as_complete():
    thin = scoring.evaluation_status(answered=1, asked=3, assessed_share=0.3, unasked_high=["Problem Solving"], ended_early=True)
    assert thin["status"] == "INSUFFICIENT_EVIDENCE"
    assert any("ended early" in r for r in thin["reasons"]) and any("asked but not answered" in r for r in thin["reasons"])
    partial = scoring.evaluation_status(answered=4, asked=4, assessed_share=0.9, unasked_high=["Problem Solving"])
    assert partial["status"] == "PARTIAL"
    full = scoring.evaluation_status(answered=5, asked=5, assessed_share=1.0, unasked_high=[])
    assert full["status"] == "COMPLETE" and full["reasons"] == []


def test_the_communication_prompt_states_the_scale():
    assert "0 to 100" in scoring.COMMUNICATION_SCORE_SYSTEM_PROMPT and "NOT 0-10" in scoring.COMMUNICATION_SCORE_SYSTEM_PROMPT


# ============================================================ 6. every persisted answer exactly once
def test_a_redelivered_answer_line_is_scored_once():
    turns = [turn("q1", "candidate", "It hashes the key.", "answer"), turn("q1", "candidate", "It hashes the key.", "answer"),
             turn("q1", "candidate", "Then it picks a bucket.", "answer")]
    assert evaluation.answer_text_for(turns, "q1") == "It hashes the key.\nThen it picks a bucket."


def test_two_distinct_answers_to_the_same_question_are_both_kept_in_order():
    turns = [turn("q1", "candidate", "First part.", "answer"), turn("q1", "agent", "More?", action="FOLLOW_UP"),
             turn("q1", "candidate", "Second part.", "answer")]
    assert evaluation.answer_text_for(turns, "q1") == "First part.\nSecond part."


def test_non_answer_turns_never_reach_the_scorer():
    turns = [turn("q1", "candidate", "What is the answer?", "request_answer"), turn("q1", "candidate", "repeat that", "repeat"),
             turn("q1", "candidate", "Can I speak in Hindi?", "language_request"), turn("q1", "candidate", "skip", "skip"),
             turn("q1", "candidate", "I would use a hash map", "answer")]
    assert evaluation.answer_text_for(turns, "q1") == "I would use a hash map"


# ============================================================ 4/5. report states + candidate vs HR over the real API
def _interview(plan=PLAN):
    db = base.TestSessionLocal()
    role = Role(name="Backend Engineer", competencies=ROLE)
    cand = Candidate(name="Asha", email="asha@example.com")
    db.add_all([role, cand])
    db.commit()
    iv = Interview(candidate_id=cand.id, role_id=role.id, plan=plan, status=states.CREATED)
    db.add(iv)
    db.commit()
    for target in (states.PLANNED, states.READY, states.IN_PROGRESS):
        transition_interview(db, iv, target)
    iid = iv.id
    db.close()
    return iid


def _persist(iid, rows):
    db = base.TestSessionLocal()
    for qid, speaker, text, intent, action in rows:
        db.add(InterviewTurn(interview_id=iid, question_id=qid, speaker=speaker, text=text, intent=intent, action=action))
    db.commit()
    db.close()


@pytest.fixture
def secured(monkeypatch):
    monkeypatch.setattr(api, "AGENT_SERVICE_KEY", KEY)
    monkeypatch.setattr(api, "score_answer_content", lambda q, e, t: {"score": 70, "strengths": [], "explanation": "e"})
    monkeypatch.setattr(api, "score_communication", lambda t: {"score": 60, "dimensions": {}, "explanation": "e"})
    monkeypatch.setattr(api, "generate_overall_summary", lambda *a, **k: {
        "summary": "s", "key_strengths": [], "key_gaps": [], "recommendation": "hire", "recommendation_reason": "r"})
    return TestClient(api.app)


def test_report_status_interview_status_and_evaluation_status_are_distinct_fields(secured):
    iid = _interview()
    _persist(iid, [("q_dynamic_1", "agent", "x", None, "NEXT_QUESTION"),
                   ("q_dynamic_1", "candidate", "I would build a demo and handle errors", "answer", None)])
    secured.post(f"/v1/interviews/{iid}/complete", headers=H)
    body = secured.get(f"/v1/interviews/{iid}/report", headers=H).json()
    assert body["report_status"] == "READY"                       # the report finished generating...
    assert body["interview_status"] == "COMPLETED"                # ...for an interview that has ended...
    assert body["evaluation_status"] == "INSUFFICIENT_EVIDENCE"   # ...that does not support a hiring decision
    assert body["status"] == body["report_status"]                # legacy field keeps its (report generation) meaning
    assert body["content"]["overall"]["recommendation"] == "insufficient_data"
    assert body["content"]["evaluation"]["answered_questions"] == 1


def test_a_candidate_cannot_obtain_any_hr_only_evaluation_data(secured):
    iid = _interview()
    _persist(iid, [("q_dynamic_1", "agent", "x", None, "NEXT_QUESTION"),
                   ("q_dynamic_1", "candidate", "I would build a demo", "answer", None)])
    secured.post(f"/v1/interviews/{iid}/complete", headers=H)
    forbidden = ("final_score", "competency_scores", "recommendation", "evaluation", "evidence", "key_strengths", "key_gaps",
                 "content_score", "hr_focus_coverage", "expected_topics", "report_status", "evaluation_status", "model_recommendation")
    for path in (f"/v1/interviews/{iid}", f"/v1/interviews/{iid}/completion"):
        text = secured.get(path).text
        for word in forbidden:
            assert word not in text, (path, word)
    for path in (f"/v1/interviews/{iid}/report", f"/v1/interviews/{iid}/transcript", f"/v1/interviews/{iid}/agent-context",
                 f"/v1/interviews/{iid}/integrity-events"):
        assert secured.get(path).status_code == 401, path
        assert secured.get(path, headers={"X-Agent-Key": "x"}).status_code == 401, path


def test_repeated_completion_does_not_change_scores_or_create_another_report(secured):
    iid = _interview()
    _persist(iid, [("q_dynamic_1", "agent", "x", None, "NEXT_QUESTION"),
                   ("q_dynamic_1", "candidate", "I would build a demo and handle errors", "answer", None)])
    first = secured.post(f"/v1/interviews/{iid}/complete", headers=H).json()
    report_a = secured.get(f"/v1/interviews/{iid}/report", headers=H).json()
    second = secured.post(f"/v1/interviews/{iid}/complete", headers=H).json()
    third = secured.post(f"/v1/interviews/{iid}/complete", headers=H).json()
    report_b = secured.get(f"/v1/interviews/{iid}/report", headers=H).json()
    assert first["report_id"] == second["report_id"] == third["report_id"]
    assert report_a["final_score"] == report_b["final_score"] and report_a["content"] == report_b["content"]
    assert secured.get(f"/v1/interviews/{iid}/completion").json()["submitted"] is True


def test_the_evaluation_uses_this_interviews_own_turns_plan_and_candidate_only(secured):
    mine, other = _interview(), _interview()
    _persist(mine, [("q_dynamic_1", "agent", "x", None, "NEXT_QUESTION"),
                    ("q_dynamic_1", "candidate", "MY-ANSWER-TEXT build a demo", "answer", None)])
    _persist(other, [("q_dynamic_1", "agent", "x", None, "NEXT_QUESTION"),
                     ("q_dynamic_1", "candidate", "SOMEONE-ELSES-ANSWER", "answer", None)])
    secured.post(f"/v1/interviews/{mine}/complete", headers=H)
    report = secured.get(f"/v1/interviews/{mine}/report", headers=H).json()
    text = str(report["content"])
    assert "MY-ANSWER-TEXT" in text and "SOMEONE-ELSES-ANSWER" not in text
    assert report["content"]["candidate"] == "Asha"
