"""Text embedding provider for semantic question retrieval
(Workmate_production_scalable_fix_plan_v2.md §6, §1 "PostgreSQL + pgvector — vector search only").

Uses OpenAI's embedding API when OPENAI_API_KEY is configured. Falls back to a deterministic
hashed bag-of-words embedding otherwise so semantic retrieval (dedup, MMR diversity selection)
still works end-to-end without a paid key configured — useful for local dev/tests, and matches
this repo's existing pattern of graceful multi-provider fallback (see TTS_PROVIDER in config.py).
The fallback is weaker than a real embedding model but is monotonic in shared-vocabulary overlap,
which is enough to exercise cosine-similarity/MMR logic correctly.
"""
import hashlib
import math
import re

from config import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-small"
FALLBACK_DIM = 256

_word_re = re.compile(r"[a-z0-9]+")


def _fallback_embedding(text: str) -> list[float]:
    vec = [0.0] * FALLBACK_DIM
    words = _word_re.findall((text or "").lower())
    if not words:
        return vec
    for w in words:
        idx = int(hashlib.sha256(w.encode()).hexdigest(), 16) % FALLBACK_DIM
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def embed_text(text: str) -> list[float]:
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text or "")
            return resp.data[0].embedding
        except Exception:
            pass  # network/quota/etc — degrade to the local fallback rather than break retrieval
    return _fallback_embedding(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
