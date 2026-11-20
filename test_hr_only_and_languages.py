"""HR can ask the interviewer to use only their own questions, and to limit the languages it speaks."""
import planner
from planner import build_interview_plan

COMPETENCIES = [{"key": "problem_solving", "name": "Problem solving", "weight": 5}, {"key": "practical_application", "name": "Practical", "weight": 4}]
HR = [{"id": "hr_1", "type": "hr", "difficulty": "medium", "competencies": ["problem_solving"], "question_text": "Describe your last project.", "topic": "", "expected_topics": [], "followup_topics": []}]


def _texts(plan):
    return [q["question_text"] for q in plan["questions"] if q.get("question_text")]


def test_only_hr_asks_nothing_but_hr_questions():
    plan = build_interview_plan(COMPETENCIES, "Engineer", {}, 4, 7, cached_dynamic_questions=[], hr_questions=HR, only_hr=True)
    assert [q["id"] for q in plan["questions"]] == ["p_intro", "hr_1"]          # not even a "tell me about yourself" on top
    assert _texts(plan)[-1] == "Describe your last project."


def test_default_may_fill_remaining_slots():
    filler = [{"id": f"d{i}", "type": "dynamic", "difficulty": "medium", "competencies": ["problem_solving"], "question_text": f"Unrelated question number {i} about distinct things {i * 7}", "expected_topics": [], "followup_topics": []} for i in range(3)]
    plan = build_interview_plan(COMPETENCIES, "Engineer", {}, 4, 7, cached_dynamic_questions=filler, hr_questions=HR)
    assert len(plan["questions"]) > 3


def test_recogniser_markers_are_noise_not_speech():
    from language_policy import noise_reason
    for heard in ("<lhil> <lhil>", "[BLANK_AUDIO]", "(noise)", "<|nospeech|>", "  [music] (typing)  "):
        assert noise_reason(heard) == "non-speech-tags", heard
    for spoken in ("I worked on <b>React</b> dashboards", "yes (I can hear you)", "I built a payments service"):
        assert noise_reason(spoken) is None, spoken


def test_each_hr_question_carries_its_own_follow_up_count():
    import json
    from api import _parse_hr_questions
    raw = json.dumps([{"text": "Q one?", "followUps": 2}, {"text": "Q two?", "followUps": 0}, {"text": "Q three?"}, {"text": "Q four?", "followUps": 9}, {"text": "Q five?", "followUps": "x"}])
    got = [q["max_followups"] for q in _parse_hr_questions(raw, COMPETENCIES)]
    assert got == [2, 0, 1, 3, 1]          # absent or unreadable means one; capped at three


def test_the_runner_uses_hr_follow_up_count_not_its_own_defaults():
    from runner import InterviewRunner
    plan = {"questions": [{"id": "hr_1", "type": "hr", "question_text": "q", "max_followups": 2, "focus_priority": "LOW"},
                          {"id": "hr_2", "type": "hr", "question_text": "q", "max_followups": 0},
                          {"id": "g_1", "type": "dynamic", "question_text": "q"}]}
    r = InterviewRunner(None, "i", plan, max_followups_per_question=5)
    seen = [r.state.max_followups]
    r.advance(); seen.append(r.state.max_followups)
    r.advance(); seen.append(r.state.max_followups)
    assert seen == [2, 0, 1]          # HR's 2, HR's 0, and a question the interviewer chose itself is capped at 1
