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
    cleaned = _ARABIC_RUN.sub(" ", text) if removed else text
    without_echo = remove_prompt_echo(cleaned)
    removed += max(len(cleaned) - len(without_echo), 0) if without_echo != cleaned else 0
    if not removed:
        return text, 0
    return re.sub(r"\s+", " ", without_echo).strip(), removed


# Devanagari vowel signs and the nukta are combining marks, which \w does not match; without the explicit block
# "चलाज़" falls apart into single letters and no word-level check can see it.
_WORDS = re.compile(r"[\w\u0900-\u097F']+", re.UNICODE)


def looks_like_repetition_hallucination(text: str) -> bool:
    """Whisper's familiar failure on mumbling or low-level noise: one short clause repeated three or more times
    ("I won't tell anyone, because I won't tell anyone. And I won't tell anyone about ..."). Real answers do not
    say the same 3-6 word phrase three times; this is noise to be re-asked, not an answer to be judged."""
    words = [w.lower() for w in _WORDS.findall(text or "")]
    run = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if (a == b and len(a) >= 3) else 1
        if run >= 3:
            return True                      # the same word three times running
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


# The vocabulary hint given to Whisper. Deliberately NOT a sentence: Whisper repeats its prompt back as a transcript when
# the audio is silent or mumbled (a live session showed "Job interview with Arav from Workmate." as the candidate's
# answer, three times, which was judged and refused until the interview ended). A glossary of terms is far less
# likely to be echoed, and anything that is echoed is caught by is_prompt_echo.
STT_VOCABULARY_HINT = "Hinglish, Hindi, English, WorkmateIQ, Arav, Aanya."
_LEGACY_STT_PROMPTS = (
    "Job interview with Arav from WorkmateIQ. Hindi, English, Hinglish, project, React, Python, Java, API, database, "
    "team, manager, experience.",
)
MIN_ECHO_RUN = 4          # a run of this many prompt words, in order, is an echo - not a candidate who says "Hindi"


def _norm_words(text: str) -> list[str]:
    return [w.lower() for w in _WORDS.findall(text or "") if w]


def _same_word(a: str, b: str) -> bool:
    return a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)))


def _echo_runs(words: list[str], prompts: tuple[str, ...]) -> list[tuple[int, int]]:
    """[(start, end)) word-index ranges of the transcript that repeat the prompt, in order, for >= MIN_ECHO_RUN words."""
    runs: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        best = 0
        for prompt in prompts:
            pw = _norm_words(prompt)
            for j in range(len(pw)):
                k = 0
                while i + k < len(words) and j + k < len(pw) and _same_word(words[i + k], pw[j + k]):
                    k += 1
                best = max(best, k)
        if best >= MIN_ECHO_RUN:
            runs.append((i, i + best))
            i += best
        else:
            i += 1
    return runs


def is_prompt_echo(text: str, prompts: tuple[str, ...] | None = None) -> bool:
    """True when the transcript IS the recogniser's own prompt repeated back: a run of at least MIN_ECHO_RUN prompt words
    in order that makes up most (>=60%) of what was "heard". "WorkmateIQ" and "Workmate" count as the same word. A
    candidate who merely says a word that is also in the prompt ("Hindi") is never an echo."""
    words = _norm_words(text)
    if len(words) < MIN_ECHO_RUN:
        return False
    runs = _echo_runs(words, prompts or (STT_VOCABULARY_HINT, *_LEGACY_STT_PROMPTS))
    return bool(runs) and max(e - s for s, e in runs) >= 0.6 * len(words)


def remove_prompt_echo(text: str, prompts: tuple[str, ...] | None = None) -> str:
    """The transcript without any run that repeats the recogniser's prompt (the original is never altered)."""
    spans = [(m.start(), m.end()) for m in _WORDS.finditer(text or "")]
    words = [text[s:e].lower() for s, e in spans]
    runs = _echo_runs(words, prompts or (STT_VOCABULARY_HINT, *_LEGACY_STT_PROMPTS))
    if not runs:
        return text
    out, last = [], 0
    for start, end in runs:
        out.append(text[last:spans[start][0]])
        last = spans[end - 1][1]
    out.append(text[last:])
    joined = re.sub(r"\s+", " ", "".join(out))
    joined = re.sub(r"([.!?।])(?:\s*[.!?।])+", r"\1", joined)        # "reads. ." -> "reads."
    return joined.strip(" ,;:-")


# Text Whisper is known to invent on silence or music.
_SILENCE_PHRASES = ("thanks for watching", "thank you for watching", "please subscribe", "like and subscribe",
                    "subtitles by", "amara.org", "transcribed by", "see you in the next video", "www.", "[music]",
                    "[applause]", "(music)", "♪")


def noise_reason(text: str) -> str | None:
    """Why this transcript is recogniser noise rather than something the candidate said, or None.

    Noise is never judged, never refused, never counted toward skipping a question, and not shown as a caption."""
    stripped = (text or "").strip()
    if not stripped:
        return "empty"
    lowered = stripped.lower()
    if len(_norm_words(stripped)) <= 14 and any(p in lowered for p in _SILENCE_PHRASES):
        return "silence-hallucination"
    if is_prompt_echo(stripped):
        return "prompt-echo"
    if looks_like_repetition_hallucination(stripped):
        return "repetition"
    return None


def supported_language_labels() -> list[str]:
    from languages import LANGUAGE_CONFIGS
    return [cfg["label"] for cfg in LANGUAGE_CONFIGS.values()]
