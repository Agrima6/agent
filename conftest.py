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
