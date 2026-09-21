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
import logging
import math
import re
import time

from config import OPENAI_API_KEY

EMBEDDING_MODEL = "text-embedding-3-small"
FALLBACK_DIM = 256

logger = logging.getLogger("embeddings")

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


# text -> embedding from the REAL model. Bank questions never change, so they are embedded once per
# process. (Only real-model vectors are cached: a fallback vector has a different dimension and must
# never be mixed with them - see embed_texts.)
_CACHE: dict[str, list[float]] = {}
_CACHE_LIMIT = 4000
EMBEDDING_TIMEOUT = 8.0   # seconds: a slow provider must fall back, not stall interview creation


# Circuit breaker. A provider that answers "no credits" / "bad key" will keep answering that, so retrying
# it on every interview only adds latency. After such a failure the provider is skipped for a while;
# a transient failure (timeout, 5xx) pauses it only briefly.
_DISABLED_UNTIL = 0.0
QUOTA_COOLDOWN = 600.0
TRANSIENT_COOLDOWN = 30.0
_now = time.monotonic


def clear_cache() -> None:
    global _DISABLED_UNTIL
    _CACHE.clear()
    _DISABLED_UNTIL = 0.0


def _trip(exc: Exception) -> None:
    global _DISABLED_UNTIL
    status = getattr(exc, "status_code", None)
    lasting = status in (401, 402, 403) or (status == 429 and "quota" in str(exc).lower())
    _DISABLED_UNTIL = _now() + (QUOTA_COOLDOWN if lasting else TRANSIENT_COOLDOWN)
    logger.warning("OpenAI embeddings unavailable (%s%s) - using the local fallback for %ds",
                   type(exc).__name__, f" {status}" if status else "", QUOTA_COOLDOWN if lasting else TRANSIENT_COOLDOWN)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed many texts with ONE provider call, returning vectors in the same order.

    Interview creation used to make one sequential OpenAI call per text (6 calls at 2.5-4s each =
    15-24s), which exceeded the web app's request timeout. Here every uncached text goes out in a
    single batched request, and results are cached.

    All returned vectors always come from the SAME source. Mixing real 1536-dim vectors with the
    256-dim local fallback makes cosine similarity silently return 0, so if the provider can't
    serve the request the WHOLE batch is embedded locally instead.
    """
    if not texts:
        return []
    if OPENAI_API_KEY and _now() >= _DISABLED_UNTIL:
        missing = list(dict.fromkeys(t for t in texts if t not in _CACHE))
        if missing:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_API_KEY, timeout=EMBEDDING_TIMEOUT, max_retries=0)
                resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[t or " " for t in missing])
                if len(_CACHE) + len(missing) > _CACHE_LIMIT:
                    _CACHE.clear()
                for text, item in zip(missing, sorted(resp.data, key=lambda d: d.index)):
                    _CACHE[text] = item.embedding
            except Exception as exc:
                _trip(exc)
                return [_fallback_embedding(t) for t in texts]   # network/quota/timeout: whole batch local
        return [_CACHE[t] for t in texts]
    return [_fallback_embedding(t) for t in texts]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
