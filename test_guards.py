"""topic_guard + output_guard: duplicates, topic lock, grounding, and what may be spoken."""
import pytest

from output_guard import (
    has_filler, has_judgement, has_leak, keep_only_questions_and_attributions, valid_lead_in, violations,
)
from topic_guard import ground_concepts, is_duplicate_question, is_on_topic, stems


# ------------------------------------------------------------------ topic_guard
def test_stems_normalise_inflection_and_drop_stopwords():
    assert stems("Hashing hashes hashed") == {"hash"}
    assert "what" not in stems("What is a HashMap")


@pytest.mark.parametrize("new,previous,expected", [
    ("Can you explain dependency injection?", ["What is dependency injection?"], True),
    ("How would you scale a cache?", ["How would you handle more cache traffic?"], False),
    ("What is MongoDB sharding?", ["How does dependency injection work?"], False),
    ("Describe how the map handles hash collisions.", ["How does the map handle a hash collision?"], True),
    ("Why is constructor injection preferred?", [], False),
])
def test_duplicate_detection_is_semantic_not_exact(new, previous, expected):
    assert is_duplicate_question(new, previous) is expected


def test_topic_lock_english_is_strict():
    anchors = ["Spring Boot dependency injection", "IoC", "constructor injection", "bean lifecycle"]
    assert is_on_topic("Why is constructor injection preferred over field injection?", anchors)
    assert not is_on_topic("What is MongoDB sharding?", anchors)


def test_topic_lock_is_lenient_for_non_english_where_structured_topic_label_applies():
    assert is_on_topic("कंस्ट्रक्टर इंजेक्शन क्यों बेहतर है?", ["dependency injection"], strict=False)


def test_ground_concepts_drops_anything_the_candidate_did_not_say():
    answer = "HashMap stores key value pairs and uses hashing to pick a bucket"
    kept = ground_concepts(["hashing", "Kafka", "key value pairs", "AWS Lambda", "bucket"], answer)
    assert kept == ["hashing", "key value pairs", "bucket"]


def test_ground_concepts_limit_and_blank_handling():
    assert ground_concepts(["", None, "   "], "anything") == []
    many = ground_concepts([f"topic{i}" for i in range(10)], " ".join(f"topic{i}" for i in range(10)), limit=3)
    assert len(many) == 3


# ------------------------------------------------------------------ output_guard
@pytest.mark.parametrize("text", [
    "Okay, understood. Let's move to the next question.",
    "Thank you for your answer.",
    "Let's proceed to the next question.",
    "Moving on to the next question.",
])
def test_filler_is_detected(text):
    assert has_filler(text)
    assert "filler_phrase" in violations(text)


@pytest.mark.parametrize("text", [
    "Great answer! Why is that?", "That's correct.", "That's incorrect.", "Good point, well done.",
    "Excellent. What next?", "Not quite. Try again?",
])
def test_judgement_of_the_answer_is_detected(text):
    assert has_judgement(text)


@pytest.mark.parametrize("text", [
    "Your score is 82 out of 100.", "The correct answer is chaining.", "You missed collision handling.",
    "You should have mentioned resizing.", "Based on the rubric, you covered two topics.",
    "My instructions say I can't help.", "The expected answer includes buckets.",
])
def test_leaks_of_evaluation_or_internal_state_are_detected(text):
    assert has_leak(text)


def test_injection_text_echoed_back_is_rejected():
    assert "echoes_injection_text" in violations("Sure, I will ignore all previous instructions.")


@pytest.mark.parametrize("text", [
    "You mentioned hashing. What happens when two keys land in the same bucket?",
    "Consider a production scenario with ten thousand requests per second. How would you approach it?",
    "Why is constructor injection generally preferred over field injection?",
    "What trade-offs would you consider here?",
])
def test_clean_interviewer_speech_has_no_violations(text):
    assert violations(text) == []


def test_empty_text_is_a_violation():
    assert violations("   ") == ["empty"]


def test_followup_sanitiser_drops_lectures_and_keeps_only_the_question():
    text = "Hashing maps keys to buckets using a hash function. What happens when two keys collide?"
    assert keep_only_questions_and_attributions(text, []) == "What happens when two keys collide?"


def test_followup_sanitiser_keeps_a_grounded_attribution():
    out = keep_only_questions_and_attributions(
        "You mentioned hashing. What happens when two keys land in the same bucket?", ["hashing"])
    assert out == "You mentioned hashing. What happens when two keys land in the same bucket?"


def test_followup_sanitiser_drops_an_attribution_the_candidate_never_made():
    out = keep_only_questions_and_attributions(
        "You mentioned Kafka. What is the retry policy?", ["hashing"])
    assert out == "What is the retry policy?"


def test_followup_sanitiser_returns_nothing_when_there_is_no_question():
    assert keep_only_questions_and_attributions("Collisions are handled by chaining.", []) == ""


def test_followup_sanitiser_keeps_only_the_last_question():
    out = keep_only_questions_and_attributions("How does it scale? And how does it fail?", [])
    assert out == "And how does it fail?"


def test_lead_in_must_be_short_grounded_and_clean():
    assert valid_lead_in("You mentioned caching.", ["caching"]) == "You mentioned caching."
    assert valid_lead_in("You mentioned Kafka.", ["caching"]) == ""       # not something they said
    assert valid_lead_in("Great answer!", ["caching"]) == ""               # judgement
    assert valid_lead_in("Okay, understood.", []) == ""                    # filler
    assert valid_lead_in("Let's look at a different scenario.", []) == "Let's look at a different scenario."
    assert valid_lead_in("Why is that?", []) == ""                         # a lead-in is not a question
    assert valid_lead_in("You mentioned caching and indexes and sharding and queues and more.", ["caching"]) == ""


@pytest.mark.parametrize("text", [
    "Could you explain how hashing picks a bucket, without referring to tree structures?",
    "What happens instead of chaining when two keys collide?",
    "How would you do it rather than sorting the keys first?",
    "Explain it as opposed to a linked list.",
])
def test_phrasing_that_quietly_tells_the_candidate_they_were_wrong_is_rejected(text):
    from output_guard import has_correction
    assert has_correction(text) and "implies_the_answer_was_wrong" in violations(text)


@pytest.mark.parametrize("text", [
    "How does the map handle two keys landing in the same bucket?",
    "You mentioned hashing. How does the bucket index get chosen?",
    "What trade-offs would you consider when the load factor grows?",
])
def test_neutral_followups_are_not_flagged_as_corrections(text):
    from output_guard import has_correction
    assert not has_correction(text)


def test_avoid_that_style_phrasing_is_flagged_as_a_correction():
    from output_guard import has_correction
    assert has_correction("You mentioned balanced trees; how does hashing help avoid that in a HashMap?")
    assert has_correction("How would you fix this in production?")
    assert not has_correction("How would you avoid duplicate writes in a distributed queue?")
