"""HR reference question: trace one unique question through every layer the agent owns, from the request the platform
sends to the sentence the candidate hears and the report HR reads (docs: HR reference question investigation).

The marker question exists at each checkpoint, or the test names the first one where it does not. The AI provider is a
scripted fake that records exactly what it is sent, so 'reached the provider' is verified on the real prompt, not on an
intermediate log.
"""
import json

import pytest

import conductor as conductor_module
from composer import Composer
from conductor import InterviewConductor
from intelligence_report import build_intelligence
from runner import InterviewRunner
from test_api_interview_config import (  # noqa: F401  (scratch-database fixtures)
    client, create, isolated_db, no_external_calls, role_and_candidate,
)
from testing_fakes import FakeProvider
import httpx

MARK = "HR-REF-TEST-731"
HR_TEXT = f"{MARK}: Explain how you would investigate a production database connection-pool exhaustion incident."
# Exactly the shape client-service sends (a Mongoose document serialised to JSON: camelCase timeLimit, id, topic).
PLATFORM_PAYLOAD = [{"id": "hq1", "text": HR_TEXT, "topic": "Production debugging", "timeLimit": 150},
                    {"id": "hq2", "text": "Tell me about a time you disagreed with a teammate on a design.", "topic": "Collaboration", "timeLimit": 120}]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def plan_for(client, role_and_candidate, **extra):
    role_id, cand_id = role_and_candidate
    resp = create(client, role_id, cand_id, questions_json=json.dumps(PLATFORM_PAYLOAD), **extra)
    assert resp.status_code == 200, resp.text
    return resp.json()["plan"]


def test_checkpoint_api_the_platform_payload_survives_validation_and_planning(client, role_and_candidate):
    plan = plan_for(client, role_and_candidate)
    hr = [q for q in plan["questions"] if q["type"] == "hr"]
    assert [q["question_text"] for q in hr] == [p["text"] for p in PLATFORM_PAYLOAD]      # verbatim, in HR's order
    assert hr[0]["topic"] == "Production debugging" and hr[0]["time_limit"] == 150           # camelCase timeLimit is honoured
    assert [q["id"] for q in plan["questions"][:2]] == ["p_intro", "p_candidate_intro"]       # asked right after the intro pair


def test_checkpoint_api_the_question_is_stored_with_the_interview_and_returned_on_reload(client, role_and_candidate):
    role_id, cand_id = role_and_candidate
    created = create(client, role_id, cand_id, questions_json=json.dumps(PLATFORM_PAYLOAD)).json()
    reloaded = client.get(f"/v1/interviews/{created['id']}").json()                          # what the worker loads
    assert any(MARK in (q.get("question_text") or "") for q in reloaded["plan"]["questions"])


def test_checkpoint_api_hr_questions_are_never_dropped_even_when_the_plan_is_long(client, role_and_candidate):
    many = [{"id": f"h{i}", "text": f"Unique HR question number {i} about topic {i}?", "topic": "t"} for i in range(1, 13)]
    role_id, cand_id = role_and_candidate
    plan = create(client, role_id, cand_id, questions_json=json.dumps(many)).json()["plan"]
    texts = [q["question_text"] for q in plan["questions"] if q["type"] == "hr"]
    assert texts == [m["text"] for m in many]


def test_checkpoint_api_two_similar_hr_questions_are_both_kept(client, role_and_candidate):
    similar = [{"text": "How would you design a rate limiter for an API?", "topic": "a"},
               {"text": "How would you design a rate limiter for a public API?", "topic": "b"}]
    role_id, cand_id = role_and_candidate
    plan = create(client, role_id, cand_id, questions_json=json.dumps(similar)).json()["plan"]
    assert len([q for q in plan["questions"] if q["type"] == "hr"]) == 2                     # HR's wording is never deduped away


def test_an_unusable_hr_question_list_is_refused_loudly_not_ignored(client, role_and_candidate):
    role_id, cand_id = role_and_candidate
    too_many = json.dumps([{"text": f"q {i}", "topic": "t"} for i in range(1, 30)])
    assert create(client, role_id, cand_id, questions_json=too_many).status_code == 422
    assert create(client, role_id, cand_id, questions_json=json.dumps([{"text": "   ", "topic": "t"}])).status_code == 422


class _Env:
    """A conductor running the plan the API produced, with the AI provider replaced by a recorder."""

    def __init__(self, plan, *, language="en", responses=()):
        self.provider = FakeProvider(*responses)
        self.spoken = []

        async def speak(speech):
            self.spoken.append(speech)

        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True}))
        self.http = httpx.AsyncClient(transport=transport)
        self.runner = InterviewRunner(self.http, "intv_t", plan, candidate_name="Asha", small_talk_rounds=0)
        # move past the two intro rows to the first HR question, as the live interview does
        self.runner.idx = next(i for i, q in enumerate(plan["questions"]) if q["type"] == "hr")
        self.runner.state = self.runner._new_question_state(self.runner.current_question)
        self.conductor = InterviewConductor(self.runner, speak, language=language, role="Backend Engineer",
                                            composer=Composer(self.provider))


async def test_checkpoint_candidate_hears_the_hr_question_verbatim(client, role_and_candidate):
    plan = plan_for(client, role_and_candidate)
    env = _Env(plan)
    composed = await env.conductor._compose(conductor_module.ComposeRequest(
        action=conductor_module.Action.NEXT_QUESTION, question=plan["questions"][1], next_question=env.runner.current_question,
        language="en", role="Backend Engineer"))
    assert composed.text == HR_TEXT and composed.question_only == HR_TEXT                     # the exact HR wording
    await env.http.aclose()


async def test_checkpoint_provider_in_hindi_the_provider_request_carries_the_hr_question(client, role_and_candidate):
    plan = plan_for(client, role_and_candidate)
    hindi = f"{MARK}: production database connection-pool exhaustion की जाँच आप कैसे करेंगे?"
    env = _Env(plan, language="hi", responses=[{"action": "NEXT_QUESTION", "candidateFacingText": hindi}])
    composed = await env.conductor._compose(conductor_module.ComposeRequest(
        action=conductor_module.Action.NEXT_QUESTION, question=plan["questions"][1], next_question=env.runner.current_question,
        language="hi", role="Backend Engineer"))
    assert MARK in env.provider.all_prompt_text                                               # what the model was actually sent
    assert composed.text == hindi
    await env.http.aclose()


async def test_checkpoint_provider_follow_up_requests_carry_the_hr_question(client, role_and_candidate):
    plan = plan_for(client, role_and_candidate)
    env = _Env(plan, responses=[{"action": "FOLLOW_UP", "candidateFacingText": "You mentioned pool limits. How did you confirm leaks?",
                                 "topic": "Production debugging", "difficulty": "medium", "confidence": 0.9}])
    from policy import Action
    composed = await env.conductor._compose(conductor_module.ComposeRequest(
        action=Action.FOLLOW_UP, question=env.runner.current_question, kind="standard", target="leak detection",
        mentioned_concepts=["pool limits"], language="en", role="Backend Engineer"))
    assert MARK in env.provider.all_prompt_text and composed.source in ("llm", "fallback")
    await env.http.aclose()


def test_checkpoint_report_shows_which_hr_questions_were_reached_and_which_were_not(client, role_and_candidate):
    plan = plan_for(client, role_and_candidate)
    hr_ids = [q["id"] for q in plan["questions"] if q["type"] == "hr"]
    reports = [{"question_id": hr_ids[0], "content_score": {"score": 82}}]                    # only the first HR question was answered
    result = build_intelligence(plan, reports, [])
    coverage = {c["question_id"]: c for c in result["hr_question_coverage"]}
    assert coverage[hr_ids[0]]["status"] == "ANSWERED" and coverage[hr_ids[0]]["score"] == 82
    assert coverage[hr_ids[1]]["status"] == "NOT_REACHED"                                    # no silent omission
    assert result["hr_questions_not_reached"] == [PLATFORM_PAYLOAD[1]["text"]]
    assert MARK in coverage[hr_ids[0]]["question"]


def test_a_report_for_an_interview_without_hr_questions_is_unchanged():
    result = build_intelligence({"questions": [{"id": "q1", "question_text": "x", "type": "scenario"}]}, [], [])
    assert result["hr_question_coverage"] == [] and result["hr_questions_not_reached"] == []
