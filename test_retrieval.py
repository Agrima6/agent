"""Unit tests for semantic question retrieval (Workmate_production_scalable_fix_plan_v2.md §6).

Uses embeddings.embed_text's deterministic fallback (no OPENAI_API_KEY in test env), so these
are fast, offline, and exercise the actual cosine-similarity/MMR/dedup logic rather than mocks.
"""
from embeddings import cosine_similarity, embed_text
from retrieval import metadata_filter, mmr_select, select_questions, semantic_dedup


def test_identical_text_has_similarity_near_one():
    a = embed_text("How would you scale a Redis cache under heavy read traffic?")
    b = embed_text("How would you scale a Redis cache under heavy read traffic?")
    assert cosine_similarity(a, b) > 0.999


def test_near_duplicate_questions_score_highly_similar():
    a = embed_text("How would you scale a cache under heavy load")
    b = embed_text("How would you handle more traffic hitting the cache")
    c = embed_text("Describe a time you led a difficult stakeholder negotiation")
    sim_related = cosine_similarity(a, b)
    sim_unrelated = cosine_similarity(a, c)
    assert sim_related > sim_unrelated


def test_metadata_filter_prefers_tagged_role_falls_back_to_all():
    questions = [
        {"id": "q1", "question_text": "x", "role_tags": ["backend"]},
        {"id": "q2", "question_text": "y", "role_tags": ["frontend"]},
        {"id": "q3", "question_text": "z", "role_tags": ["general"]},
    ]
    filtered = metadata_filter(questions, "backend")
    ids = {q["id"] for q in filtered}
    assert "q1" in ids and "q3" in ids and "q2" not in ids

    # A role type with zero matching questions (not even a "general"-tagged one) must fall back
    # to the full pool rather than returning nothing (a role we don't have dedicated content for
    # still gets an interview).
    no_general = [{"id": "q1", "question_text": "x", "role_tags": ["backend"]},
                  {"id": "q2", "question_text": "y", "role_tags": ["frontend"]}]
    filtered_unknown = metadata_filter(no_general, "nonexistent_role_type")
    assert len(filtered_unknown) == 2


def test_semantic_dedup_drops_near_duplicates_keeps_first():
    questions = [
        {"id": "q1", "question_text": "How would you scale a cache under heavy load"},
        {"id": "q2", "question_text": "How would you scale a cache under heavy load"},  # exact dup
        {"id": "q3", "question_text": "Describe leading a difficult stakeholder negotiation"},
    ]
    embeddings_by_id = {q["id"]: embed_text(q["question_text"]) for q in questions}
    deduped = semantic_dedup(questions, embeddings_by_id)
    ids = [q["id"] for q in deduped]
    assert ids == ["q1", "q3"]


def test_mmr_select_never_repeats_already_selected_ids():
    questions = [
        {"id": f"q{i}", "question_text": f"question about topic {i}"} for i in range(5)
    ]
    embeddings_by_id = {q["id"]: embed_text(q["question_text"]) for q in questions}
    query = embed_text("topic")
    selected = mmr_select(questions, embeddings_by_id, query, k=3, already_selected_ids={"q0", "q1"})
    ids = {q["id"] for q in selected}
    assert "q0" not in ids and "q1" not in ids
    assert len(selected) == 3


def test_select_questions_end_to_end_respects_count_and_exclusions():
    bank = [
        {"id": "q1", "question_text": "Scale a Redis cache under load", "competencies": ["scalability"],
         "role_tags": ["backend"]},
        {"id": "q2", "question_text": "Handle a spike in cache traffic", "competencies": ["scalability"],
         "role_tags": ["backend"]},
        {"id": "q3", "question_text": "Resolve a conflict with a difficult teammate",
         "competencies": ["communication"], "role_tags": ["general"]},
        {"id": "q4", "question_text": "Debug a production outage under pressure",
         "competencies": ["problem_solving"], "role_tags": ["backend"]},
    ]
    selected = select_questions(bank, role_name="Backend Engineer", role_type="backend",
                                 competency_keys=["scalability", "problem_solving"], count=2,
                                 already_selected_ids={"q3"})
    ids = {q["id"] for q in selected}
    assert len(selected) == 2
    assert "q3" not in ids


def test_select_questions_zero_count_returns_empty():
    assert select_questions([{"id": "q1", "question_text": "x"}], role_name="x", role_type="general",
                             competency_keys=[], count=0) == []
