"""Embedding batching, caching and source consistency.

Regression: interview creation made one sequential OpenAI embedding call per text (6 calls at 2.5-4s =
15-24s), exceeding the web app's 15s request timeout - so candidates could hit "Something went wrong"
before their interview even started.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import embeddings
from retrieval import candidate_text, select_questions, warm_embedding_cache

DIM = 8
BANK = json.loads((Path(__file__).parent / "question_bank.json").read_text())


class FakeOpenAI:
    """Records every request; returns deterministic vectors (optionally out of order)."""
    requests: list[list[str]] = []
    constructed: list[dict] = []
    fail = False
    shuffle = False

    def __init__(self, **kwargs):
        FakeOpenAI.constructed.append(kwargs)
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model, input):
        if FakeOpenAI.fail:
            raise TimeoutError("provider timed out")
        FakeOpenAI.requests.append(list(input))
        data = [SimpleNamespace(index=i, embedding=[float(len(t)), float(i + 1)] + [0.5] * (DIM - 2))
                for i, t in enumerate(input)]
        return SimpleNamespace(data=list(reversed(data)) if FakeOpenAI.shuffle else data)


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    FakeOpenAI.requests, FakeOpenAI.constructed, FakeOpenAI.fail, FakeOpenAI.shuffle = [], [], False, False
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    embeddings.clear_cache()
    yield


def test_many_texts_go_out_in_one_request_and_keep_their_order_even_if_the_api_shuffles():
    FakeOpenAI.shuffle = True
    texts = ["alpha", "bb", "cccc"]
    vectors = embeddings.embed_texts(texts)
    assert len(FakeOpenAI.requests) == 1 and FakeOpenAI.requests[0] == texts
    assert [v[0] for v in vectors] == [5.0, 2.0, 4.0]          # first component encodes len(text): order preserved


def test_results_are_cached_so_repeat_calls_make_no_request():
    embeddings.embed_texts(["alpha", "bb"])
    embeddings.embed_texts(["bb", "alpha"])
    assert len(FakeOpenAI.requests) == 1


def test_only_uncached_texts_are_requested_and_duplicates_are_sent_once():
    embeddings.embed_texts(["alpha"])
    embeddings.embed_texts(["alpha", "new one", "new one"])
    assert FakeOpenAI.requests[1] == ["new one"]


def test_the_client_is_built_with_a_short_timeout_and_no_retry_sleep():
    embeddings.embed_texts(["alpha"])
    assert FakeOpenAI.constructed[0]["timeout"] == embeddings.EMBEDDING_TIMEOUT <= 10
    assert FakeOpenAI.constructed[0]["max_retries"] == 0


def test_a_failing_provider_falls_back_for_the_WHOLE_batch_never_a_mix():
    embeddings.embed_texts(["cached before"])                   # a real vector is now cached
    FakeOpenAI.fail = True
    vectors = embeddings.embed_texts(["cached before", "brand new"])
    assert {len(v) for v in vectors} == {embeddings.FALLBACK_DIM}   # 1 dimension for all, not real + fallback


def test_fallback_vectors_are_never_cached_as_real_ones(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(embeddings, "_now", lambda: now["t"])
    FakeOpenAI.fail = True
    embeddings.embed_texts(["alpha"])
    FakeOpenAI.fail = False
    now["t"] += embeddings.TRANSIENT_COOLDOWN + 1               # the provider recovers and the pause ends
    vector = embeddings.embed_texts(["alpha"])[0]
    assert len(vector) == DIM                                   # a real vector, not the stale fallback


def test_without_an_api_key_everything_is_local_and_deterministic(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "")
    a, b = embeddings.embed_texts(["hello world"]), embeddings.embed_texts(["hello world"])
    assert a == b and len(a[0]) == embeddings.FALLBACK_DIM and FakeOpenAI.requests == []


def test_empty_input_needs_no_request():
    assert embeddings.embed_texts([]) == [] and FakeOpenAI.requests == []


def test_selecting_questions_costs_one_embedding_request_not_one_per_question():
    select_questions(BANK, role_name="Backend Engineer", role_type="backend",
                     competency_keys=["technical_depth", "problem_solving"], count=4)
    assert len(FakeOpenAI.requests) == 1
    assert len(FakeOpenAI.requests[0]) > 3                      # the whole pool + the query in that single call


def test_a_warmed_bank_makes_the_first_selection_nearly_free():
    warm_embedding_cache(BANK)
    warmed_requests = len(FakeOpenAI.requests)
    kwargs = dict(role_name="Backend Engineer", role_type="backend", competency_keys=["technical_depth"], count=3)
    select_questions(BANK, **kwargs)
    assert len(FakeOpenAI.requests) - warmed_requests == 1      # only the role-specific query text was new
    assert FakeOpenAI.requests[-1] == ["Backend Engineer interview question covering: technical_depth"]
    select_questions(BANK, **kwargs)
    assert len(FakeOpenAI.requests) - warmed_requests == 1      # same role again: fully cached


def test_selection_still_works_when_the_provider_is_down():
    FakeOpenAI.fail = True
    picked = select_questions(BANK, role_name="Backend Engineer", role_type="backend",
                              competency_keys=["technical_depth"], count=3)
    assert len(picked) == 3 and len({q["id"] for q in picked}) == 3


def test_warmup_never_raises_even_if_the_provider_is_down():
    FakeOpenAI.fail = True
    warm_embedding_cache(BANK)


def test_candidate_text_is_the_shared_embedding_key():
    q = {"question_text": "How do you scale?", "competencies": ["a", "b"]}
    assert candidate_text(q) == "How do you scale? a b"


# ------------------------------------------------------------------ circuit breaker
class QuotaError(Exception):
    status_code = 429

    def __str__(self):
        return "You have no credits remaining (insufficient_quota)"


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(embeddings, "_now", lambda: now["t"])
    return now


def _break_provider(monkeypatch, error):
    def boom(self, **kwargs):
        raise error

    monkeypatch.setattr(FakeOpenAI, "_create", boom)


def test_an_out_of_credit_provider_is_not_retried_on_every_interview(monkeypatch, clock):
    calls = {"n": 0}

    def quota(self, **kwargs):
        calls["n"] += 1
        raise QuotaError()

    monkeypatch.setattr(FakeOpenAI, "_create", quota)
    for _ in range(5):
        embeddings.embed_texts(["alpha", "bb"])
    assert calls["n"] == 1                                      # tried once, then skipped: no more doomed round trips


def test_the_provider_is_tried_again_after_the_cooldown(monkeypatch, clock):
    _break_provider(monkeypatch, QuotaError())
    embeddings.embed_texts(["alpha"])
    monkeypatch.undo()                                          # simulate: the account was topped up
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    monkeypatch.setattr(embeddings, "_now", lambda: clock["t"])
    assert len(embeddings.embed_texts(["alpha"])[0]) == embeddings.FALLBACK_DIM      # still in cooldown
    clock["t"] += embeddings.QUOTA_COOLDOWN + 1
    assert len(embeddings.embed_texts(["alpha"])[0]) == DIM                            # cooldown over: real again


def test_a_transient_failure_pauses_the_provider_only_briefly(monkeypatch, clock):
    _break_provider(monkeypatch, TimeoutError("slow"))
    embeddings.embed_texts(["alpha"])
    assert embeddings._DISABLED_UNTIL - clock["t"] == embeddings.TRANSIENT_COOLDOWN < embeddings.QUOTA_COOLDOWN


def test_a_quota_failure_is_logged_once_with_a_clear_message(monkeypatch, clock, caplog):
    import logging
    _break_provider(monkeypatch, QuotaError())
    with caplog.at_level(logging.WARNING, logger="embeddings"):
        for _ in range(3):
            embeddings.embed_texts(["alpha"])
    messages = [r.getMessage() for r in caplog.records if "unavailable" in r.getMessage()]
    assert len(messages) == 1 and "QuotaError" in messages[0] and "local fallback" in messages[0]
