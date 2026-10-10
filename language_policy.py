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


def supported_language_labels() -> list[str]:
    from languages import LANGUAGE_CONFIGS
    return [cfg["label"] for cfg in LANGUAGE_CONFIGS.values()]
