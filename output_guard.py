"""Validation of candidate-facing interviewer text before it can be spoken.

Whatever an LLM writes is treated as untrusted output. It is only spoken if it passes these checks
(and is otherwise replaced by a deterministic fallback in composer.py):
  * no stock filler ("Okay, understood", "Let's move to the next question", ...)
  * no judgement of the answer ("great", "correct", "wrong", ...)
  * no leaking of evaluation, scores, rubric, expected answers or hidden reasoning
  * no injection-style text echoed back
  * in follow-ups, only QUESTIONS (and short, grounded attributions) - never a lecture

Pure functions, no I/O.
"""
import re

from interaction_guard import contains_injection
from speech_prep import split_sentences
from topic_guard import stems

_FILLER_PHRASES = re.compile(
    r"\b(?:okay,?\s+understood|ok,?\s+understood|thank(?:s| you)\s+for\s+(?:your|the|that|sharing|providing|answering|explaining)"
    r"|thank you\.?$|thanks\.?$|let'?s\s+(?:now\s+)?(?:move|proceed|go)\s+(?:on\s+)?to\s+the\s+next"
    r"|moving\s+on\s+to\s+the\s+next|let'?s\s+move\s+on|on\s+to\s+the\s+next\s+question"
    r"|next\s+question\b|understood\.?\s+let'?s|got\s+it\.?\s+let'?s)",
    re.I,
)
_JUDGEMENT = re.compile(
    r"\b(?:great|good|excellent|perfect|nice|awesome|fantastic|wonderful|impressive|brilliant|correct|incorrect|"
    r"wrong|right|well\s+done|nicely\s+done|spot\s+on|exactly|not\s+quite|that'?s\s+it)\b(?=[\s,.!]|$)",
    re.I,
)
_CORRECTION = re.compile(
    r"\b(?:without\s+(?:referring|using|mentioning|relying|considering|resorting)|instead\s+of|rather\s+than|"
    r"as\s+opposed\s+to|not\s+(?:by|using|through|with)\s+\w+ing|"
    r"(?:avoid|prevent|overcome|fix|solve|replace)\s+(?:that|this|it|those|these|such)\b)", re.I)
_LEAKS = re.compile(
    r"\b(?:score[sd]?|marks?|rating|rated|grade[sd]?|rubric|evaluat\w*|expected\s+(?:topics?|answer)|coverage|"
    r"the\s+(?:correct|right|ideal|expected|model)\s+(?:answer|approach|solution)|the\s+answer\s+(?:is|would\s+be)|"
    r"you\s+should\s+have|you\s+(?:didn'?t|did\s+not|haven'?t|have\s+not|failed\s+to)\s+"
    r"(?:mention|cover|address|explain|include|say)|you\s+(?:missed|forgot|overlooked|skipped|left\s+out)\b|"
    r"missing\s+(?:concepts?|topics?)|internal|hint\s*:|"
    r"the\s+system|my\s+instructions|system\s+prompt)\b",
    re.I,
)
_ATTRIBUTION = re.compile(
    r"^(?:you|your)\b.{0,60}\b(?:mention|said|say|talk|describ|note|spoke|refer|brought\s+up|touch|discuss|outlin|"
    r"explain|use[d]?|built|work)\w*",
    re.I,
)
_CONNECTOR = re.compile(
    r"^(?:let'?s\s+(?:go|look|dig|think|consider|take)|consider|imagine|suppose|picture|building\s+on\s+that|"
    r"going\s+(?:a\s+level\s+)?deeper|thinking\s+about|in\s+a\s+(?:production|real|high[- ]traffic|large|busy)|"
    r"one\s+more\s+angle|say\s+that|for\s+instance|to\s+go\s+deeper)\b",
    re.I,
)


def has_filler(text: str) -> bool:
    return bool(_FILLER_PHRASES.search(text or ""))


def has_judgement(text: str) -> bool:
    return bool(_JUDGEMENT.search(text or ""))


def has_correction(text: str) -> bool:
    """'... without referring to trees' / 'instead of X' quietly tells the candidate they were wrong."""
    return bool(_CORRECTION.search(text or ""))


def has_leak(text: str) -> bool:
    return bool(_LEAKS.search(text or ""))


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def violations(text: str) -> list[str]:
    """Every reason `text` must not be spoken as interviewer speech (empty list = clean)."""
    problems = []
    if not (text or "").strip():
        return ["empty"]
    if has_filler(text):
        problems.append("filler_phrase")
    if has_judgement(text):
        problems.append("judges_the_answer")
    if has_leak(text):
        problems.append("leaks_evaluation_or_internal_state")
    if has_correction(text):
        problems.append("implies_the_answer_was_wrong")
    if contains_injection(text):
        problems.append("echoes_injection_text")
    return problems


def mentions_grounded(sentence: str, grounded_concepts: list[str]) -> bool:
    """True if the sentence refers to at least one concept the candidate genuinely said (>=60% of
    that concept's meaningful stems appear in the sentence)."""
    sentence_stems = stems(sentence)
    for concept in grounded_concepts:
        concept_stems = stems(concept)
        if concept_stems and len(concept_stems & sentence_stems) / len(concept_stems) >= 0.6:
            return True
    return False


def keep_only_questions_and_attributions(text: str, grounded_concepts: list[str], max_sentences: int = 2) -> str:
    """For follow-ups: keep questions, attributions that cite something the candidate really said,
    and short scenario connectors; drop every other statement. Nothing else can be spoken, which is
    what stops an LLM from lecturing or explaining a concept mid-interview."""
    kept: list[str] = []
    for sentence in split_sentences(text, min_chars=1):
        s = sentence.strip()
        if not s:
            continue
        if s.endswith("?"):
            kept.append(s)
        elif _ATTRIBUTION.match(s) and mentions_grounded(s, grounded_concepts):
            kept.append(s)
        elif _CONNECTOR.match(s) and word_count(s) <= 10:
            kept.append(s)
    # A follow-up is only useful if it contains a question; keep the LAST question and at most one
    # lead-in sentence before it.
    if not any(k.endswith("?") for k in kept):
        return ""
    last_q = max(i for i, k in enumerate(kept) if k.endswith("?"))
    lead = kept[last_q - 1] if last_q > 0 and not kept[last_q - 1].endswith("?") else ""
    result = [x for x in (lead, kept[last_q]) if x]
    return " ".join(result[-max_sentences:])


def valid_lead_in(text: str, grounded_concepts: list[str], max_words: int = 10) -> str:
    """A short optional bridge before a fixed next question. '' if it isn't clean and grounded."""
    text = (text or "").strip()
    if not text or word_count(text) > max_words or violations(text) or "?" in text:
        return ""
    if _ATTRIBUTION.match(text):
        return text if mentions_grounded(text, grounded_concepts) else ""
    return text if _CONNECTOR.match(text) else ""
