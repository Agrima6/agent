"""Test-suite-wide isolation.

The developer's real .env has OPENAI_API_KEY set, so without this every embedding call in the suite
went to the real OpenAI API (slow, costs money, non-deterministic). Tests use the deterministic local
embedding unless they explicitly install a fake provider client.
"""
import pytest

import embeddings


@pytest.fixture(autouse=True)
def hermetic_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_API_KEY", "")
    embeddings.clear_cache()
    yield
    embeddings.clear_cache()


@pytest.fixture(autouse=True)
def hermetic_service_key(monkeypatch):
    """The developer's .env may define AGENT_SERVICE_KEY (shared with the backend). Tests must not depend on it:
    start every test with no key configured; tests that exercise authentication set one themselves."""
    import api
    import runner
    monkeypatch.setattr(api, "AGENT_SERVICE_KEY", "")
    monkeypatch.setattr(runner, "AGENT_SERVICE_KEY", "", raising=False)
    monkeypatch.setattr(runner, "_AUTH_HEADERS", {}, raising=False)


@pytest.fixture(autouse=True)
def hermetic_personalised_questions(monkeypatch):
    """Personalised question generation calls an LLM: tests never reach one. It fails by default (which exercises the
    controlled role-level fallback); tests that want generated questions install their own stub."""
    import planner

    def unavailable(*args, **kwargs):
        raise RuntimeError("no LLM in tests")
    monkeypatch.setattr(planner, "generate_personalized_questions", unavailable)
