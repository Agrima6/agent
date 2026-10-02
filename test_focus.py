"""Focus areas: priorities come from HR weights, HR wins conflicts, depth is configurable (never hard-coded)."""
import pytest

from focus import (
    FocusArea, MAX_FOCUS_AREAS, build_focus_areas, focus_budget, followup_limit, match_focus, priority_from_share,
)

HR = [
    {"name": "Production debugging", "weight": 40},
    {"name": "Spring Boot", "weight": 30},
    {"name": "Communication", "weight": 5},
]


def by_name(areas):
    return {a.name: a for a in areas}


def test_priority_comes_from_each_skills_share_of_the_hr_weights():
    areas = by_name(build_focus_areas(HR))
    assert areas["Production debugging"].priority == "HIGH"      # 40/75 of the weight
    assert areas["Spring Boot"].priority == "HIGH"               # 30/75
    assert areas["Communication"].priority == "LOW"              # 5/75
    assert all(a.sources == ["HR"] for a in areas.values())


def test_an_explicit_priority_always_beats_the_weight():
    areas = by_name(build_focus_areas([{"name": "Ownership", "weight": 1, "priority": "high"}, {"name": "SQL", "weight": 99}]))
    assert areas["Ownership"].priority == "HIGH"


def test_thresholds_are_configurable(monkeypatch):
    assert priority_from_share(0.30) == "HIGH" and priority_from_share(0.15) == "MEDIUM" and priority_from_share(0.05) == "LOW"
    monkeypatch.setenv("FOCUS_HIGH_SHARE", "0.5")
    monkeypatch.setenv("FOCUS_MEDIUM_SHARE", "0.2")
    assert priority_from_share(0.30) == "MEDIUM" and priority_from_share(0.1) == "LOW"


def test_hr_and_role_areas_merge_with_the_higher_priority_and_both_sources():
    role = [{"key": "production_debugging", "weight": 0.1}, {"key": "system_design", "weight": 0.9}]
    areas = by_name(build_focus_areas(HR, role))
    merged = areas["Production debugging"]
    assert merged.sources == ["HR", "ROLE"] and merged.priority == "HIGH"          # HR's HIGH is not diluted by a low role weight
    assert areas["System Design"].sources == ["ROLE"]


def test_areas_are_ordered_by_priority_capped_and_given_stable_ids():
    many = [{"name": f"Skill {i}", "weight": i + 1} for i in range(30)]
    areas = build_focus_areas(many)
    assert len(areas) == MAX_FOCUS_AREAS
    assert [a.id for a in areas][:3] == ["f_1", "f_2", "f_3"]
    ranks = [{"HIGH": 0, "MEDIUM": 1, "LOW": 2}[a.priority] for a in areas]
    assert ranks == sorted(ranks)


def test_junk_input_is_ignored_not_trusted():
    areas = build_focus_areas([{"name": ""}, {"name": "  "}, {"weight": 5}, "not a dict", None, {"name": "<script>SQL</script>", "weight": "x"}])
    assert [a.name for a in areas] == ["SQL"] and areas[0].weight is None      # markup dropped, bad weight ignored
    assert build_focus_areas([{"name": "A", "weight": -5}, {"name": "B", "weight": float("nan")}, {"name": "C", "weight": float("inf")}])[0].weight is None
    assert build_focus_areas(None, None) == [] and build_focus_areas([], []) == []


def test_matching_a_question_to_its_focus_area_requires_real_overlap():
    areas = build_focus_areas(HR)
    assert match_focus("Tell me about a production incident you debugged", areas).name == "Production debugging"
    assert match_focus("How do you structure a Spring Boot service?", areas).name == "Spring Boot"
    assert match_focus("What is your favourite colour?", areas) is None                # no forced, misleading tag
    assert match_focus("", areas) is None


def test_depth_is_configurable_and_bounded(monkeypatch):
    assert followup_limit(2, "HIGH") == 3 and followup_limit(2, "MEDIUM") == 2 and followup_limit(2, "LOW") == 1
    assert followup_limit(0, "LOW") == 0                                                # never negative
    assert followup_limit(5, "HIGH") == 5                                               # never above the hard ceiling
    assert followup_limit(2, None) == 2
    assert (focus_budget("HIGH"), focus_budget("MEDIUM"), focus_budget("LOW")) == (4, 2, 1)
    monkeypatch.setenv("FOCUS_BUDGETS", "HIGH:6,LOW:0")
    monkeypatch.setenv("FOCUS_FOLLOWUP_DELTA", "HIGH:2,bogus:9,MEDIUM:x")
    assert focus_budget("HIGH") == 6 and focus_budget("LOW") == 0 and focus_budget("MEDIUM") == 2   # malformed entries ignored
    assert followup_limit(2, "HIGH") == 4 and followup_limit(2, "MEDIUM") == 2


def test_round_trips_through_json():
    area = build_focus_areas(HR)[0]
    assert FocusArea.from_dict(area.to_dict()) == area
    assert FocusArea.from_dict({"name": "X", "priority": "bogus", "sources": ["nope"]}).priority == "MEDIUM"


def test_judge_evidence_must_come_from_the_candidates_own_words():
    from policy import parse_evaluation
    ev = parse_evaluation({"evidence": ["restarted the payment pod", "invented achievement about kubernetes mastery"],
                           "ownership": "OWN", "inconsistency": "  team size  differs "},
                          "I restarted the payment pod after the alert")
    assert ev.evidence == ["restarted the payment pod"] and ev.ownership == "own" and ev.inconsistency == "team size differs"
    assert parse_evaluation({"ownership": "boss"}, "x").ownership == "unclear"


def test_high_priority_questions_get_extra_followups_low_ones_fewer():
    from runner import InterviewRunner
    plan = {"questions": [{"id": "a", "question_text": "q", "focus_priority": "HIGH"},
                          {"id": "b", "question_text": "q", "focus_priority": "LOW"},
                          {"id": "c", "question_text": "q"}]}
    r = InterviewRunner(None, "i", plan, max_followups_per_question=2)
    assert r.state.max_followups == 3
    r.advance(); assert r.state.max_followups == 1
    r.advance(); assert r.state.max_followups == 2
