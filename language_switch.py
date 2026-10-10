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
    "hinglish": r"hinglish|hindlish|हिंग्लिश|हिंगलिश|हिन्ग्लिश|हिंग्लीश|हिंगलीश",
    "hi": r"hindi|hindee|हिंदी|हिन्दी|हिंदि",
    "en": r"english|angrezi|angrejhi|अंग्रेज़ी|अंग्रेजी|अंग्रेज़ी|इंग्लिश|इंगलिश|इंग्लीश|इंगरिश|इंग्रेजी",
}
_ANY_LANG = "|".join(f"(?:{p})" for p in _LANGS.values())

# Phrases that make the language mention a REQUEST. Roman and Devanagari spellings of the common ways to ask.
_REQUEST = re.compile(
    r"(?:"
    r"(?:can|could|would|will|may)\s+(?:you|we|u)\b[^.?!]*?(?:speak|talk|say|ask|reply|respond|continue|switch|change|do|conduct|take|go|use|explain)"
    r"|(?:can|could|may|shall)\s+(?:i|we)\b[^.?!]{0,12}?(?:speak|talk|answer|reply|respond|continue|switch|go)\b"
    r"|(?:please|pls|plz|kindly)\b[^.?!]{0,30}?(?:" + _ANY_LANG + r")"
    r"|(?:" + _ANY_LANG + r")\b\s*(?:please|pls|plz|mein|me|men|main|mai|में|मे|bolo|boliye|bol|baat|chalega|chahiye|only|instead)"
    r"|(?:speak|talk|ask|reply|respond|continue|switch|change|convert|interview|conduct|go|use)\b[^.?!]{0,25}?\b(?:in|to|into|with)\b[^.?!]{0,8}?(?:" + _ANY_LANG + r")"
    r"|(?:bolo|boliye|bolna|baat|baatcheet|karo|kariye|kijiye|कीजिए|करो|बोलो|बोलिए|बात)\b[^.?!]{0,25}?(?:" + _ANY_LANG + r")"
    r"|(?:" + _ANY_LANG + r")[^.?!]{0,25}?(?:bolo|boliye|bolna|baat|karo|kariye|kijiye|बोलो|बोलिए|बोल|बात|कीजिए|करो|चाहिए|चलेगा|सकते|सकता|सकती)"
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


# Languages people may ask for that this interview cannot conduct. Only a fixed label is ever returned.
_UNSUPPORTED = {
    "Urdu": r"urdu|اردو",
    "Punjabi": r"punjabi|panjabi|ਪੰਜਾਬੀ|पंजाबी",
    "Tamil": r"tamil|தமிழ்|तमिल",
    "Telugu": r"telugu|తెలుగు|तेलुगु",
    "Bengali": r"bengali|bangla|বাংলা|बंगाली",
    "Marathi": r"marathi|मराठी",
    "Gujarati": r"gujarati|ગુજરાતી|गुजराती",
    "Kannada": r"kannada|ಕನ್ನಡ|कन्नड़",
    "Malayalam": r"malayalam|മലയാളം|मलयालम",
    "Arabic": r"arabic|العربية",
    "French": r"french|français",
    "Spanish": r"spanish|español",
    "German": r"german|deutsch",
}


def detect_unsupported_language_request(text: str) -> str | None:
    """The (fixed) name of an unsupported language the candidate asked to use, or None.

    Same conservatism as detect_language_request: a short utterance with a clear request cue. "I know Tamil" or a long
    answer that mentions a language is not a request."""
    if not text:
        return None
    cleaned = " ".join(text.strip().split())
    if len(_WORD.findall(cleaned)) > MAX_WORDS:
        return None
    named = [label for label, pattern in _UNSUPPORTED.items() if re.search(pattern, cleaned, re.IGNORECASE)]
    if len(named) != 1:
        return None
    request_cue = re.compile(
        r"(?:can|could|may|shall|would|will)\s+(?:i|we|you|u)\b|\bplease\b|\bpls\b|\bplz\b|\bwant\b|\bprefer\b|\bneed\b|"
        r"\b(?:speak|talk|answer|reply|respond|continue|switch|interview|conduct)\b|\bchahiye\b|\bbol", re.IGNORECASE)
    if not request_cue.search(cleaned) or _STATEMENT.search(cleaned) and not re.search(
            r"(?:can|could|would|will|may)\s+(?:i|you|we)|please|pls|plz|chahiye|want|prefer", cleaned, re.IGNORECASE):
        return None
    return named[0]


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
