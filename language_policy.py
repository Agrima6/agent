"""Which scripts and languages this interview actually supports - and what to do with anything else.

Supported conversation languages are the keys of languages.LANGUAGE_CONFIGS: English, Hindi (Devanagari) and Hinglish
(Devanagari + Latin). None of them is written in Arabic script. So an Arabic-script transcript (Urdu, Arabic, Persian)
is never a correct transcription of a supported session: it is Whisper's well-known habit of labelling Hindustani or
Indian-accented speech as Urdu, or the candidate really speaking an unsupported language. Either way it must not be
evaluated as an answer, quoted back to the candidate, or used to change the interview language.

Pure functions, no I/O. The original transcript is never modified here; callers keep it as the source record.
"""
import re

_ARABIC_SCRIPT = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

MIN_ARABIC_LETTERS = 3          # a stray character is noise, not a language
MIN_ARABIC_SHARE = 0.30         # share of all letters that must be Arabic-script


def script_profile(text: str) -> dict:
    """Letter counts per script, plus the Arabic-script share of all letters."""
    arabic = len(_ARABIC_SCRIPT.findall(text or ""))
    devanagari = len(_DEVANAGARI.findall(text or ""))
    latin = len(_LATIN.findall(text or ""))
    total = arabic + devanagari + latin
    return {"arabic": arabic, "devanagari": devanagari, "latin": latin,
            "arabic_share": (arabic / total) if total else 0.0}


def has_unexpected_script(text: str) -> bool:
    """True when the transcript is substantially in a script no supported interview language uses."""
    profile = script_profile(text)
    return profile["arabic"] >= MIN_ARABIC_LETTERS and profile["arabic_share"] >= MIN_ARABIC_SHARE


_ARABIC_RUN = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿][؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿\s،؛؟۔]*")
MIN_WORDS_AFTER_CLEANING = 3


def clean_transcript(text: str) -> tuple[str, int]:
    """The transcript with any Arabic-script runs removed, and how many letters were removed.

    Whisper often writes the Hindi words of a Hinglish sentence in Urdu script ("... in the country. بہت سارے کارے ہیں").
    Those runs are not something a supported-language candidate said in that script; they are the recogniser's rendering
    of Hindi, and left in they reach the judge, the caption stream and the report. The ORIGINAL text is always kept as
    the source record (interview_turns.text); this view is computed from it on demand, so it is always traceable and is
    never stored in its place."""
    if not text:
        return "", 0
    removed = len(_ARABIC_SCRIPT.findall(text))
    if not removed:
        return text, 0
    cleaned = _ARABIC_RUN.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip(), removed


_WORDS = re.compile(r"[\w']+", re.UNICODE)


def looks_like_repetition_hallucination(text: str) -> bool:
    """Whisper's familiar failure on mumbling or low-level noise: one short clause repeated three or more times
    ("I won't tell anyone, because I won't tell anyone. And I won't tell anyone about ..."). Real answers do not
    say the same 3-6 word phrase three times; this is noise to be re-asked, not an answer to be judged."""
    words = [w.lower() for w in _WORDS.findall(text or "")]
    if len(words) < 9:
        return False
    for n in range(3, 7):
        seen: dict[tuple, int] = {}
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            seen[gram] = seen.get(gram, 0) + 1
            if seen[gram] >= 3:
                return True
    return False


def supported_language_labels() -> list[str]:
    from languages import LANGUAGE_CONFIGS
    return [cfg["label"] for cfg in LANGUAGE_CONFIGS.values()]
