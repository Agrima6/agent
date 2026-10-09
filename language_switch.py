"""Spot a candidate asking the interviewer to change language mid-interview ("speak in Hindi", "Hindi mein baat
karo", "English please", "हिंदी में बोलिए").

Deliberately conservative. It only returns a language code (en / hi / hinglish) - never any of the candidate's
words - so nothing the candidate says can inject text into later prompts. It must NOT fire on statements about
the candidate ("I speak Hindi", "I built a Hindi chatbot", "my English is weak") or on long answers that merely
mention a language, so it needs a clear request cue and a short utterance.
"""
import re

MAX_WORDS = 14                      # a request is short; a paragraph that mentions Hindi is an answer

_LANGS = {
    "hinglish": r"hinglish|hindlish",
    "hi": r"hindi|hindee|हिंदी|हिन्दी",
    "en": r"english|angrezi|angrejhi|अंग्रेज़ी|अंग्रेजी|इंग्लिश",
}
_ANY_LANG = "|".join(f"(?:{p})" for p in _LANGS.values())

# Phrases that make the language mention a REQUEST. Roman and Devanagari spellings of the common ways to ask.
_REQUEST = re.compile(
    r"(?:"
    r"(?:can|could|would|will|may)\s+(?:you|we|u)\b[^.?!]*?(?:speak|talk|say|ask|reply|respond|continue|switch|change|do|conduct|take|go|use|explain)"
    r"|(?:please|pls|plz|kindly)\b[^.?!]{0,30}?(?:" + _ANY_LANG + r")"
    r"|(?:" + _ANY_LANG + r")\b\s*(?:please|pls|plz|mein|me|men|main|mai|में|मे|bolo|boliye|bol|baat|chalega|chahiye|only|instead)"
    r"|(?:speak|talk|ask|reply|respond|continue|switch|change|convert|interview|conduct|go|use)\b[^.?!]{0,25}?\b(?:in|to|into|with)\b[^.?!]{0,8}?(?:" + _ANY_LANG + r")"
    r"|(?:bolo|boliye|bolna|baat|baatcheet|karo|kariye|kijiye|कीजिए|करो|बोलो|बोलिए|बात)\b[^.?!]{0,25}?(?:" + _ANY_LANG + r")"
    r"|(?:" + _ANY_LANG + r")[^.?!]{0,25}?(?:bolo|boliye|bolna|baat|karo|kariye|kijiye|बोलो|बोलिए|बात|कीजिए|करो|चाहिए|चलेगा)"
    r"|(?:i|we|mujhe|hum|main)\b[^.?!]{0,25}?(?:want|prefer|need|like|chahiye|chahta|chahti|comfortable|better|easier)[^.?!]{0,25}?(?:" + _ANY_LANG + r")"
    r"|(?:switch|change|set)\b[^.?!]{0,20}?language"
    r")",
    re.IGNORECASE,
)

# "I speak Hindi", "my English is weak", "I know Hindi": information about the candidate, not a request.
_STATEMENT = re.compile(
    r"^\W*(?:i|i'm|i am|my|main|mera|meri|mujhe)\b[^.?!]{0,30}?\b(?:speak|know|learnt|learned|studied|fluent|native|born|weak|good at|bad at|"
    r"jaanta|jaanti|aata|aati)\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[\wऀ-ॿ']+")


def detect_language_request(text: str, current: str | None = None) -> str | None:
    """The language code the candidate asked to switch to, or None. A request for the language already in use is
    not a switch (returns None)."""
    if not text:
        return None
    cleaned = " ".join(text.strip().split())
    if len(_WORD.findall(cleaned)) > MAX_WORDS:
        return None
    asked = [code for code, pattern in _LANGS.items() if re.search(pattern, cleaned, re.IGNORECASE)]
    # "hinglish" contains no "hindi", but guard anyway: more than one language named = ambiguous, ask nothing.
    asked = list(dict.fromkeys(asked))
    if len(asked) != 1:
        return None
    if not _REQUEST.search(cleaned):
        return None
    if _STATEMENT.search(cleaned) and not re.search(r"(?:can|could|would|will|may)\s+(?:you|we)|please|pls|plz|chahiye|want|prefer", cleaned, re.IGNORECASE):
        return None
    return None if asked[0] == current else asked[0]
