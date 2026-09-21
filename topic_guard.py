"""Deterministic lexical guards used to validate LLM-written interviewer text.

Three questions get answered here, with no network and no LLM:
  * is_duplicate_question - is this (near-)identical to something already asked?
  * is_on_topic           - does a follow-up stay around the current question's topic?
  * ground_concepts       - which "things the candidate said" did the candidate actually say?

The last one is the anti-hallucination check: the interviewer may only attribute a concept to the
candidate ("You mentioned X") if that concept genuinely appears in the candidate's own words.
"""
import re

_TOKEN = re.compile(r"[a-z0-9]+|[ऀ-ॿ]+")

_STOPWORDS = frozenset("""
a an the and or but if then else so of to in on at by for with from into onto over under about as is are was
were be been being am do does did done doing have has had having it its this that these those there here i
you your yours we our us they them their he she his her me my mine what which who whom whose when where why
how can could would should will shall may might must not no yes any some more most much many very just also
too than out up down off again once only own same such both each few other another please tell explain
describe walk through way question answer little bit lot let lets say said mention mentioned talk talked
think thought like get got go going make made give given see seen use used using one two first next then
now still even really actually generally usually typically approach handle handling work works working
""".split())


def _stem(token: str) -> str:
    if len(token) <= 3 or not token.isascii():
        return token
    for suffix in ("ations", "ation", "ings", "ing", "ions", "ion", "ies", "ers", "er", "ed", "es", "ly", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def stems(text: str) -> set[str]:
    """Meaningful, lightly-stemmed tokens of `text` (stopwords removed)."""
    out = set()
    for token in _TOKEN.findall((text or "").lower()):
        if token in _STOPWORDS or (len(token) < 2 and token.isascii()):
            continue
        out.add(_stem(token))
    return out


def similarity(a: str, b: str) -> tuple[float, float]:
    """(jaccard, containment) of the two texts' meaningful stems."""
    sa, sb = stems(a), stems(b)
    if not sa or not sb:
        return 0.0, 0.0
    inter = len(sa & sb)
    return inter / len(sa | sb), inter / min(len(sa), len(sb))


def is_duplicate_question(new_text: str, previous_texts: list[str], threshold: float = 0.6) -> bool:
    """True if `new_text` asks essentially the same thing as any of `previous_texts`.

    'What is dependency injection?' and 'Can you explain dependency injection?' share the same
    meaningful stems and are duplicates, even though the strings differ.
    """
    for prev in previous_texts:
        jaccard, containment = similarity(new_text, prev)
        if jaccard >= threshold:
            return True
        if containment >= 0.85 and min(len(stems(new_text)), len(stems(prev))) >= 3:
            return True
    return False


def is_on_topic(text: str, anchors: list[str], *, strict: bool = True) -> bool:
    """A follow-up must share at least one meaningful stem with the current question's anchors
    (question text, topic, expected/follow-up/missing topics). `strict=False` (used for Hindi and
    Hinglish, where the LLM may legitimately write a technical term in Devanagari) always passes;
    the composer's structured `topic` field still has to be one of the allowed topics there."""
    if not strict:
        return True
    anchor_stems: set[str] = set()
    for anchor in anchors:
        anchor_stems |= stems(anchor)
    if not anchor_stems:
        return True
    return bool(stems(text) & anchor_stems)


def _clean_concept(concept: str) -> str:
    return re.sub(r"\s+", " ", str(concept or "")).strip(" .,;:\"'`")[:60]


def ground_concepts(concepts: list[str], answer_text: str, *, limit: int = 4) -> list[str]:
    """Keep only concepts the candidate actually said: at least 60% of a concept's meaningful
    stems must appear in the candidate's answer. Ungrounded (invented) concepts are dropped."""
    answer_stems = stems(answer_text)
    grounded: list[str] = []
    for raw in concepts or []:
        concept = _clean_concept(raw)
        concept_stems = stems(concept)
        if not concept or not concept_stems:
            continue
        if len(concept_stems & answer_stems) / len(concept_stems) >= 0.6 and concept not in grounded:
            grounded.append(concept)
        if len(grounded) >= limit:
            break
    return grounded
