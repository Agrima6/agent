"""Semantic question retrieval: metadata filter -> similarity -> dedup -> MMR diversity
selection (Workmate_production_scalable_fix_plan_v2.md §6).

Deliberately implemented in pure Python over an in-memory candidate pool rather than a real
`ORDER BY embedding <=> :query LIMIT n` SQL query — the question bank is currently ~15-30
entries (see planner.py), so a Python scan is fast and, per the plan's own guidance ("do not
introduce a separate vector database until actual retrieval scale justifies it"), building out
a real pgvector-backed SQL path now would be premature. The scoring/dedup/MMR logic here is the
part that matters and is identical either way; swapping the candidate-fetch step for a real
`<=>` similarity query later is a small, isolated change if the bank grows large enough to need it.
"""
from embeddings import cosine_similarity, embed_text, embed_texts

# Two questions whose embeddings are this similar are treated as semantic duplicates and must
# never both appear in the same interview plan (plan §6: "avoid semantically equivalent
# questions", e.g. "how would you scale a cache" vs "how would you handle more cache traffic").
SEMANTIC_DUPLICATE_THRESHOLD = 0.92

# MMR trade-off between relevance to the query and diversity from already-selected questions.
# Lower = more diverse, higher = more relevance-greedy.
MMR_LAMBDA = 0.7


def candidate_text(q: dict) -> str:
    """The text a bank question is embedded as (shared by selection and the startup warm-up so the
    warmed cache entries are exactly the ones selection will look up)."""
    return f"{q['question_text']} {' '.join(q.get('competencies', []))}"


def warm_embedding_cache(candidates: list[dict]) -> None:
    """Embed the whole question bank in one call so the first interview created after startup is as
    fast as every later one. Best-effort: failure just means the first interview pays the cost."""
    try:
        embed_texts([candidate_text(q) for q in candidates])
    except Exception:
        pass


def metadata_filter(questions: list[dict], role_type: str) -> list[dict]:
    """Role/status metadata filter — the first stage before any similarity math (plan §6)."""
    eligible = [q for q in questions
                if role_type in q.get("role_tags", ["general"]) or "general" in q.get("role_tags", [])]
    return eligible if eligible else questions


def semantic_dedup(questions: list[dict], embeddings_by_id: dict[str, list[float]]) -> list[dict]:
    """Drop questions that are near-duplicates of an earlier one in the list (keeps the first
    occurrence). Order-sensitive by design: call this on a relevance-sorted list."""
    kept: list[dict] = []
    for q in questions:
        emb = embeddings_by_id.get(q["id"])
        if emb and any(cosine_similarity(emb, embeddings_by_id.get(k["id"], [])) >= SEMANTIC_DUPLICATE_THRESHOLD
                        for k in kept):
            continue
        kept.append(q)
    return kept


def mmr_select(candidates: list[dict], embeddings_by_id: dict[str, list[float]], query_embedding: list[float],
               k: int, already_selected_ids: set[str] | None = None) -> list[dict]:
    """Maximal-marginal-relevance selection: relevance to the query minus similarity to what's
    already been picked (plan §6/§14 — "relevance + competency coverage - similarity to selected
    questions"). Also honors `already_selected_ids` so a question already used elsewhere in the
    same interview plan (e.g. the resume-derived or dynamic questions) is never picked again
    (plan §6: "never select the same question twice in one interview plan").
    """
    already_selected_ids = already_selected_ids or set()
    pool = [q for q in candidates if q["id"] not in already_selected_ids]
    selected: list[dict] = []
    selected_embeddings: list[list[float]] = []

    while pool and len(selected) < k:
        best_q, best_score = None, float("-inf")
        for q in pool:
            emb = embeddings_by_id.get(q["id"])
            if not emb:
                continue
            relevance = cosine_similarity(emb, query_embedding)
            redundancy = max((cosine_similarity(emb, s) for s in selected_embeddings), default=0.0)
            score = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
            if score > best_score:
                best_q, best_score = q, score
        if best_q is None:
            break
        selected.append(best_q)
        selected_embeddings.append(embeddings_by_id[best_q["id"]])
        pool.remove(best_q)

    return selected


def select_questions(candidates: list[dict], role_name: str, role_type: str, competency_keys: list[str],
                      count: int, already_selected_ids: set[str] | None = None) -> list[dict]:
    """End-to-end pipeline: metadata filter -> embed -> semantic dedup -> MMR diversity select.

    `candidates` is the full eligible pool (e.g. the static question bank); each dict must have
    at least `id` and `question_text`. Embeddings are computed on demand rather than requiring a
    pre-computed store, since the bank is small — see the module docstring.
    """
    if count <= 0 or not candidates:
        return []

    filtered = metadata_filter(candidates, role_type)
    query_text = f"{role_name} interview question covering: {', '.join(competency_keys)}"
    # ONE embedding request for every candidate plus the query (was one sequential call each).
    vectors = embed_texts([candidate_text(q) for q in filtered] + [query_text])
    embeddings_by_id = {q["id"]: vec for q, vec in zip(filtered, vectors)}
    query_embedding = vectors[-1]

    # Rank by relevance first so semantic_dedup keeps the MORE relevant half of any near-duplicate
    # pair, not an arbitrary one.
    ranked = sorted(filtered, key=lambda q: cosine_similarity(embeddings_by_id[q["id"]], query_embedding),
                     reverse=True)
    deduped = semantic_dedup(ranked, embeddings_by_id)

    return mmr_select(deduped, embeddings_by_id, query_embedding, count, already_selected_ids)
