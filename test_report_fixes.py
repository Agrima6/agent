"""Regression tests for the mistakes found in a real interview report:

  1. "Can I speak in Hindi?" was scored as a (zero) answer to the technical question.
  2. A 15/100 competency was listed as a "high" strength.
  3. A hire/no-hire call was issued from almost no evidence.
  4. A behavioural answer was filed as evidence for "Technical Depth" because of one shared word.
  5. Competencies the interview never reached were scored as 0, deflating the final score.
"""
import pytest

import scoring
from focus import FocusArea, build_focus_areas, focus_for_question
from interaction_guard import Intent, NON_ANSWER_INTENTS
from language_switch import detect_language_request
from test_conductor import Harness, Q1, Q2, Q3, strong

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ----------------------------------------------------------------------------- 1. language requests
def test_a_language_request_is_a_non_answer_intent():
    assert Intent.LANGUAGE_REQUEST.value == "language_request"
    assert Intent.LANGUAGE_REQUEST in NON_ANSWER_INTENTS


@pytest.mark.parametrize("text", ["Can I speak in Hindi?", "Can I answer in Hindi", "May I speak Hindi", "Could I reply in Hindi please"])
def test_first_person_language_requests_are_recognised(text):
    assert detect_language_request(text, current="en") == "hi"


@pytest.mark.parametrize("text", ["I speak Hindi at home", "I built a Hindi chatbot for support", "The answer is a hash map"])
def test_statements_are_still_not_requests(text):
    assert detect_language_request(text, current="en") is None


async def test_a_language_request_switches_language_and_is_never_scored(harness_factory):
    h = harness_factory(evaluations=[], composer_responses=[])
    await h.say("Can I speak in Hindi?")
    assert h.conductor.language == "hi"
    assert h.judge_calls == []                                           # not judged
    assert h.posted("coverage-results") == []                            # not scored
    assert h.posted("turns")[0]["intent"] == "language_request"          # recorded, but as a non-answer


async def test_asking_for_the_language_already_in_use_is_not_an_answer_either(harness_factory):
    h = harness_factory(language="hi", evaluations=[], composer_responses=[])
    await h.say("Can I speak in Hindi?")
    assert h.judge_calls == [] and h.posted("coverage-results") == []
    assert h.posted("turns")[0]["intent"] == "language_request"
    assert h.runner.idx == 0                                             # the question is simply asked again


@pytest.fixture
async def harness_factory():
    made = []

    def make(**kwargs):
        h = Harness(**kwargs)
        made.append(h)
        return h

    yield make
    for h in made:
        await h.close()


# ----------------------------------------------------------------------------- 2. strengths vs scores
def test_a_weak_competency_is_never_kept_as_a_strength():
    strengths = ["High ownership score (15) indicating willingness to take responsibility",
                 "Identified prompt-injection as a root cause",
                 "Clear technical depth on database design"]
    kept = scoring.drop_contradicted_strengths(strengths, {"ownership": 15, "technical_depth": 82})
    assert kept == ["Identified prompt-injection as a root cause", "Clear technical depth on database design"]


def test_competencies_are_labelled_with_the_scale_and_a_band():
    assert scoring.band(15) == "weak" and scoring.band(55) == "adequate" and scoring.band(80) == "strong"


def test_the_summary_prompt_states_the_scale_and_the_model_output_is_filtered(monkeypatch):
    seen = {}

    def fake(system, user, *a, **k):
        seen["user"] = user
        return {"summary": "s", "key_strengths": ["High ownership score (15)", "Solid SQL vs NoSQL reasoning"],
                "key_gaps": [], "recommendation": "no_hire", "recommendation_reason": "r"}

    monkeypatch.setattr(scoring, "structured_json", fake)
    out = scoring.generate_overall_summary("A", "Role", 12.0, {"ownership": 15, "technical_depth": 80}, [])
    assert "15/100 (weak)" in seen["user"] and "80/100 (strong)" in seen["user"]
    assert out["key_strengths"] == ["Solid SQL vs NoSQL reasoning"]


# ----------------------------------------------------------------------------- 3. enough evidence?
def test_too_little_evidence_yields_insufficient_data_not_no_hire():
    overall = {"recommendation": "no_hire", "recommendation_reason": "weak", "summary": "x"}
    out = scoring.guard_recommendation(overall, answered=1, assessed_share=0.3, unasked_high=["Problem Solving"])
    assert out["recommendation"] == "insufficient_data"
    assert out["model_recommendation"] == "no_hire"                      # kept for the reviewer, not hidden
    assert "only 1 answered" in out["recommendation_reason"] and "Problem Solving" in out["recommendation_reason"]


def test_enough_evidence_leaves_the_recommendation_alone():
    overall = {"recommendation": "hire", "recommendation_reason": "good"}
    assert scoring.guard_recommendation(overall, answered=5, assessed_share=0.9, unasked_high=[]) == overall


def test_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("MIN_ANSWERED_FOR_RECOMMENDATION", "1")
    monkeypatch.setenv("MIN_ASSESSED_WEIGHT_SHARE", "0.1")
    overall = {"recommendation": "borderline", "recommendation_reason": "r"}
    assert scoring.guard_recommendation(overall, answered=1, assessed_share=0.2, unasked_high=[]) == overall


# ----------------------------------------------------------------------------- 5. final score
def test_unasked_competencies_do_not_drag_the_final_score_down():
    comps = [{"key": "technical_depth", "weight": 0.5}, {"key": "problem_solving", "weight": 0.3},
             {"key": "communication", "weight": 0.2}]
    out = scoring.aggregate_final_score([{"competencies": ["technical_depth"], "content_score": 80}], comps)
    assert out["final_score"] == 80.0                                    # was 40.0: unasked weight scored as zero
    assert out["assessed_weight_share"] == 0.5                           # ...but how little was assessed is reported


def test_nothing_assessed_is_zero_with_no_division_error():
    out = scoring.aggregate_final_score([], [{"key": "a", "weight": 1}])
    assert out == {"final_score": 0.0, "competency_scores": {}, "assessed_weight_share": 0.0}


# ----------------------------------------------------------------------------- 4. evidence attribution
ROLE_COMPETENCIES = [{"key": "technical_depth", "weight": 0.4}, {"key": "problem_solving", "weight": 0.3},
                     {"key": "ownership", "weight": 0.3}]


def test_a_behavioural_question_is_filed_under_its_own_competency_not_a_shared_word():
    areas = build_focus_areas(None, ROLE_COMPETENCIES)
    q = {"id": "q_general_1", "type": "behavioral", "competencies": ["ownership", "collaboration"],
         "question_text": "Tell me about a time you disagreed with a teammate about a technical decision."}
    area = focus_for_question(q, areas)
    assert area is not None and area.name == "Ownership"                # the old word match chose "Technical Depth"


def test_a_technical_question_still_maps_to_technical_depth():
    areas = build_focus_areas(None, ROLE_COMPETENCIES)
    q = {"id": "q1", "type": "scenario", "competencies": ["technical_depth"], "question_text": "How does a HashMap work?"}
    assert focus_for_question(q, areas).name == "Technical Depth"


def test_a_question_without_competencies_falls_back_to_text_matching():
    areas = [FocusArea(id="f_1", name="Payment Systems", priority="HIGH", sources=["HR"])]
    q = {"id": "h1", "type": "hr", "question_text": "Walk me through a payment systems outage you handled."}
    assert focus_for_question(q, areas).name == "Payment Systems"


def test_a_stray_word_does_not_file_a_competency_tagged_question_under_a_role_area():
    areas = build_focus_areas(None, [{"key": "technical_depth", "weight": 0.5}, {"key": "customer_handling", "weight": 0.5}])
    q = {"id": "q", "type": "behavioral", "competencies": ["customer_handling"],
         "question_text": "Describe a technical decision you explained to a client."}
    assert focus_for_question(q, areas).name == "Customer Handling"


# ----------------------------------------------------------------------------- end to end through the real job
def test_full_report_ignores_the_language_request_and_refuses_to_judge_thin_evidence(monkeypatch):
    """The exact shape of the reported failure: one real answer, one 'Can I speak in Hindi?', most areas unasked."""
    import api
    import states
    import test_end_interview as base          # reuses its throwaway DB + dependency overrides
    from db import Candidate, Interview, InterviewTurn, Role, transition_interview
    from fastapi.testclient import TestClient

    judged = []

    def fake_content(question, expected, text):
        judged.append(text)
        return {"score": 40, "strengths": [], "improvement_areas": [], "explanation": "e"}

    monkeypatch.setattr(api, "score_answer_content", fake_content)
    monkeypatch.setattr(api, "score_communication", lambda text: {"score": 50, "dimensions": {}, "explanation": "e"})
    monkeypatch.setattr(scoring, "structured_json", lambda *a, **k: {
        "summary": "s", "key_strengths": ["High ownership score (15)"], "key_gaps": [],
        "recommendation": "no_hire", "recommendation_reason": "weak"})
    monkeypatch.setattr(api, "generate_overall_summary", scoring.generate_overall_summary)

    db = base.TestSessionLocal()
    role = Role(name="Backend Engineer", competencies=[{"key": "technical_depth", "weight": 0.6},
                                                       {"key": "ownership", "weight": 0.4}])
    cand = Candidate(name="Agrima", email="a@example.com")
    db.add_all([role, cand])
    db.commit()
    plan = {"questions": [
        {"id": "q_tech", "type": "scenario", "competencies": ["technical_depth"], "question_text": "Design a prototype.",
         "expected_topics": ["data", "demo"]},
        {"id": "q_gen", "type": "behavioral", "competencies": ["ownership"], "question_text": "Tell me about a disagreement.",
         "expected_topics": ["resolution"]}]}
    interview = Interview(candidate_id=cand.id, role_id=role.id, plan=plan, status=states.CREATED)
    db.add(interview)
    db.commit()
    for target in (states.PLANNED, states.READY, states.IN_PROGRESS):
        transition_interview(db, interview, target)
    iid = interview.id
    db.add_all([
        InterviewTurn(interview_id=iid, question_id="q_tech", speaker="candidate", text="Can I speak in Hindi?",
                      intent="language_request"),
        InterviewTurn(interview_id=iid, question_id="q_gen", speaker="candidate", text="I resolved it by talking.",
                      intent="answer"),
    ])
    db.commit()
    db.close()

    client = TestClient(api.app)
    client.post(f"/v1/interviews/{iid}/complete")
    body = client.get(f"/v1/interviews/{iid}/report").json()

    assert body["status"] == "READY"
    assert judged == ["I resolved it by talking."]                       # the Hindi request was never scored
    by_id = {q["question_id"]: q for q in body["content"]["questions"]}
    assert set(by_id) == {"q_tech", "q_gen"}
    assert by_id["q_tech"]["answered"] is False and by_id["q_tech"]["status"] == "NOT_ANSWERED"   # asked, not answered
    assert by_id["q_tech"]["content_score"]["score"] == 0 and by_id["q_tech"]["candidate_answer"] == ""
    assert by_id["q_gen"]["answered"] is True
    overall = body["content"]["overall"]
    assert overall["recommendation"] == "insufficient_data"              # one answer is not a hiring decision
    assert overall["model_recommendation"] == "no_hire"
    assert body["content"]["answered_questions"] == 1
    # The asked-but-unanswered technical question is assessed (as 0), so both role competencies are assessed; the thin
    # evidence is reported through the answered count and the recommendation, not by silently dropping the question.
    assert body["content"]["assessed_weight_share"] == 1.0
    assert body["final_score"] == 16.0                                   # 0.6 * 0 (not answered) + 0.4 * 40
    assert body["content"]["evaluation"]["status"] == "INSUFFICIENT_EVIDENCE"
