"""LLM provider settings and the conductor's live-conversation deadlines.

Regressions found by running the interviewer against the real Groq API:
  * max_tokens=300 was exhausted by hidden reasoning tokens before the JSON finished (400 json_validate_failed),
    so EVERY composer call silently fell back to generic text;
  * SDK retry/backoff on 429s stalled turns for 7-15s while the candidate sat in silence.
"""
import logging
import time
from types import SimpleNamespace

import pytest

import conductor as conductor_module
from composer import ComposeRequest, Composer
from llm_provider import LLMError, OpenAICompatibleProvider, parse_json_object
from policy import Action, Evaluation
from testing_fakes import FakeProvider


class _Completions:
    def __init__(self, content='{"ok": true}', error=None):
        self.calls, self.options, self.content, self.error = [], [], content, error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def make_provider(model="openai/gpt-oss-120b", **kw):
    provider = OpenAICompatibleProvider("groq", "key", "https://example.invalid/v1", model)
    completions = _Completions(**kw)

    class _Client:
        def with_options(self, **options):
            completions.options.append(options)
            return SimpleNamespace(chat=SimpleNamespace(completions=completions))

    provider._client = _Client()
    return provider, completions


def test_reasoning_effort_and_token_budget_are_sent_to_reasoning_models():
    provider, completions = make_provider()
    provider.complete_json("s", "u", max_tokens=1200, reasoning_effort="low", retries=0, timeout=6)
    call = completions.calls[0]
    assert call["reasoning_effort"] == "low" and call["max_tokens"] == 1200
    assert completions.options[0] == {"timeout": 6, "max_retries": 0}


def test_reasoning_effort_is_not_sent_to_models_that_do_not_support_it():
    provider, completions = make_provider(model="gpt-4o-mini")
    provider.complete_json("s", "u", reasoning_effort="low")
    assert "reasoning_effort" not in completions.calls[0]


def test_retries_default_to_one_for_background_calls_and_zero_is_honoured_for_live_calls():
    provider, completions = make_provider()
    provider.complete_json("s", "u")
    provider.complete_json("s", "u", retries=0)
    assert [o["max_retries"] for o in completions.options] == [1, 0]


def test_a_failed_call_raises_a_short_safe_error_and_logs_the_real_cause(caplog):
    provider, _ = make_provider(error=RuntimeError("Error code: 400 - json_validate_failed"))
    with caplog.at_level(logging.WARNING, logger="llm-provider"):
        with pytest.raises(LLMError) as info:
            provider.complete_json("s", "u")
    assert "json_validate_failed" not in str(info.value)               # not leaked to callers
    assert any("json_validate_failed" in r.getMessage() for r in caplog.records)   # but diagnosable in logs


@pytest.mark.parametrize("text,ok", [('{"a": 1}', True), ('```json\n{"a": 1}\n```', True),
                                     ('Sure! {"a": 1} hope that helps', True), ("no json here", False),
                                     ("[1, 2]", False), ('{"a": ', False), ("", False)])
def test_parse_json_object_tolerates_wrapping_but_rejects_non_objects(text, ok):
    if ok:
        assert parse_json_object(text) == {"a": 1}
    else:
        with pytest.raises(LLMError):
            parse_json_object(text)


# ------------------------------------------------------------------ composer request settings
Q = {"id": "q1", "topic": "hashmap internals", "question_text": "How does a HashMap work?",
     "expected_topics": ["hashing", "collision handling"]}


def test_every_composer_llm_call_gets_a_reasoning_safe_budget_low_effort_and_no_retry_sleep():
    fake = FakeProvider({"action": "FOLLOW_UP", "candidateFacingText": "What happens when two keys collide in a bucket?",
                         "topic": "hashmap internals"},
                        {"action": "NEXT_QUESTION", "leadIn": "You mentioned hashing."})
    Composer(fake).compose(ComposeRequest(action=Action.FOLLOW_UP, question=Q, target="collision handling",
                                          mentioned_concepts=["hashing"], asked_texts=[Q["question_text"]]))
    Composer(fake).compose(ComposeRequest(action=Action.NEXT_QUESTION, question=Q, mentioned_concepts=["hashing"],
                                          next_question={"id": "q2", "question_text": "Design a rate limiter."}))
    for call in fake.calls:
        assert call["max_tokens"] >= 600, "budget must leave room for hidden reasoning tokens"
        assert call["reasoning_effort"] == "low" and call["retries"] == 0


def test_depth_fallback_names_the_topic_instead_of_being_generic():
    out = Composer(FakeProvider(LLMError("x"), LLMError("x"))).compose(
        ComposeRequest(action=Action.FOLLOW_UP, kind="depth", question=Q, asked_texts=[]))
    assert "hashmap internals" in out.text


def test_composer_fallback_is_available_without_any_llm():
    composer = Composer(FakeProvider())
    follow = composer.fallback(ComposeRequest(action=Action.FOLLOW_UP, kind="standard", question=Q,
                                              target="collision handling"), "deadline_exceeded")
    assert follow.source == "fallback" and follow.reason == "deadline_exceeded" and "collision handling" in follow.text
    nxt = composer.fallback(ComposeRequest(action=Action.NEXT_QUESTION, question=Q,
                                           next_question={"id": "q2", "question_text": "Design a rate limiter."}), "x")
    assert nxt.text == "Design a rate limiter."


# ------------------------------------------------------------------ conductor deadlines
pytestmark_anyio = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_a_slow_judge_never_stalls_the_candidate(monkeypatch):
    from test_conductor import Harness, Q1, Q2
    monkeypatch.setattr(conductor_module, "JUDGE_DEADLINE", 0.05)
    h = Harness(questions=(Q1, Q2), evaluations=[])

    def slow_judge(*a, **k):
        time.sleep(2.0)
        return Evaluation(coverage_score=0.1, missing_topics=["x"])

    h.conductor.judge = slow_judge
    started = time.perf_counter()
    out = await h.say("It hashes the key.")
    assert time.perf_counter() - started < 1.5                         # answered at the deadline, not after the 2s call
    assert out.action == "NEXT_QUESTION" and Q2["question_text"] in out.text
    await h.close()


@pytest.mark.anyio
async def test_a_slow_composer_falls_back_to_plain_wording_at_the_deadline(monkeypatch):
    from test_conductor import Harness, Q1, medium
    monkeypatch.setattr(conductor_module, "COMPOSE_DEADLINE", 0.05)
    h = Harness(questions=(Q1,), evaluations=[medium()])

    def slow_compose(req):
        time.sleep(2.0)
        raise AssertionError("must not be used")

    h.conductor.composer.compose = slow_compose
    started = time.perf_counter()
    out = await h.say("It hashes the key to find a bucket.")
    assert time.perf_counter() - started < 1.5
    assert out.action == "FOLLOW_UP" and out.text.endswith("?") and "collision handling" in out.text
    await h.close()


# ------------------------------------------------------------------ per-role models
def test_role_model_defaults_split_the_judge_and_the_composer_on_groq(monkeypatch):
    import llm_provider
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("COMPOSER_MODEL", raising=False)
    provider, _ = make_provider()
    llm_provider.set_provider(provider)
    try:
        assert llm_provider.role_model("judge") is None                       # provider default (gpt-oss-120b)
        assert llm_provider.role_model("composer") == "openai/gpt-oss-20b"
        monkeypatch.setenv("COMPOSER_MODEL", "some/other-model")
        assert llm_provider.role_model("composer") == "some/other-model"
        monkeypatch.setenv("JUDGE_MODEL", "judge/model")
        assert llm_provider.role_model("judge") == "judge/model"
    finally:
        llm_provider.set_provider(None)


def test_a_role_model_this_key_cannot_use_falls_back_to_the_default_model(caplog):
    import httpx
    from openai import NotFoundError

    provider, completions = make_provider()
    not_found = NotFoundError("model missing", response=httpx.Response(404, request=httpx.Request("POST", "http://x")), body=None)
    real_create = completions.create
    state = {"n": 0}

    def flaky_create(**kwargs):
        state["n"] += 1
        if kwargs["model"] == "openai/gpt-oss-20b":
            raise not_found
        return real_create(**kwargs)

    completions.create = flaky_create
    with caplog.at_level(logging.WARNING, logger="llm-provider"):
        result = provider.complete_json("s", "u", model="openai/gpt-oss-20b")
    assert result == {"ok": True} and state["n"] == 2
    assert any("falling back" in r.getMessage() for r in caplog.records)


def test_a_missing_default_model_is_a_clean_error_not_an_infinite_loop():
    import httpx
    from openai import NotFoundError

    provider, _ = make_provider(error=NotFoundError(
        "nope", response=httpx.Response(404, request=httpx.Request("POST", "http://x")), body=None))
    with pytest.raises(LLMError):
        provider.complete_json("s", "u")


def test_structured_json_only_lowers_reasoning_effort_when_a_caller_asks_for_it(monkeypatch):
    import llm_client
    import llm_provider
    provider, completions = make_provider()
    llm_provider.set_provider(provider)
    try:
        llm_client.structured_json("s", "u")                                   # scoring / summaries: thorough (default)
        llm_client.structured_json("s", "u", reasoning_effort="low")            # interview creation: fast
    finally:
        llm_provider.set_provider(None)
    assert "reasoning_effort" not in completions.calls[0] and "max_tokens" not in completions.calls[0]
    assert completions.calls[1]["reasoning_effort"] == "low" and completions.calls[1]["max_tokens"] >= 2000
