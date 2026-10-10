from intelligence_report import build_intelligence

PLAN = {"focus_areas": [{"id": "f_1", "name": "Production debugging", "priority": "HIGH", "sources": ["HR"]},
                        {"id": "f_2", "name": "Spring Boot", "priority": "HIGH", "sources": ["HR"]},
                        {"id": "f_3", "name": "Communication", "priority": "LOW", "sources": ["HR"]}],
        "resume_claims": [{"id": "c_1", "claim": "Cut latency 40%"}, {"id": "c_2", "claim": "Led five engineers"},
                          {"id": "c_3", "claim": "Built billing"}],
        "questions": [{"id": "q1", "question_text": "a", "focus": "Production debugging"},
                      {"id": "q2", "question_text": "b", "focus": "Communication"},
                      {"id": "q_validate_c_1", "question_text": "c"}, {"id": "q_validate_c_2", "question_text": "d"},
                      {"id": "q_validate_c_3", "question_text": "e"}]}
REPORTS = [{"question_id": "q1", "content_score": {"score": 80}}, {"question_id": "q2", "content_score": {"score": 40}},
           {"question_id": "q_validate_c_1", "content_score": {"score": 75}},
           {"question_id": "q_validate_c_2", "content_score": {"score": 70}}]
COV = [{"question_id": "q1", "followup_count": 2, "evaluation": {"evidence": ["restarted the pod"], "uncertainty": 0.1}},
       {"question_id": "q_validate_c_1", "followup_count": 1, "evaluation": {"ownership": "own", "evidence": ["I wrote the cache"]}},
       {"question_id": "q_validate_c_2", "followup_count": 0,
        "evaluation": {"ownership": "team", "inconsistency": "Says the team was three people, resume says five."}}]


def test_focus_coverage_and_unasked_high_priority():
    r = build_intelligence(PLAN, REPORTS, COV)
    by = {f["focus"]: f for f in r["hr_focus_coverage"]}
    assert by["Production debugging"]["status"] == "COVERED" and by["Production debugging"]["follow_up_depth"] == 2
    assert by["Production debugging"]["evidence"] == ["restarted the pod"]
    assert by["Communication"]["status"] == "PARTIAL"
    assert by["Spring Boot"]["status"] == "NOT_ASKED"
    assert r["unasked_high_priority"] == ["Spring Boot"]


def test_claims_are_substantiated_flagged_or_unanswered_and_never_auto_rejected():
    r = build_intelligence(PLAN, REPORTS, COV)
    status = {c["claim"]: c["status"] for c in r["resume_validation"]}
    assert status == {"Cut latency 40%": "SUBSTANTIATED", "Led five engineers": "REVIEW_INCONSISTENCY",
                      "Built billing": "NOT_ANSWERED"}
    assert r["contradictions"][0]["needs_human_review"] and r["decision_support_only"]


def test_empty_plan_is_safe():
    r = build_intelligence({}, [], [])
    assert r["hr_focus_coverage"] == [] and r["contradictions"] == []
