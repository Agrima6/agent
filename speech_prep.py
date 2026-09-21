"""Text preparation between the interview engine and TTS.

Everything the interviewer says goes through prepare_for_speech() first, so the TTS provider only
ever sees clean spoken-language text: no markdown, no JSON, no stage directions, no emoji, and no
exotic punctuation that engines tend to mispronounce. Pure functions, no I/O.
"""
import json
import re
import unicodedata

_SENTENCE_END = "?!.।"  # ? ! . and the Devanagari danda

_CODE_FENCE = re.compile(r"```.*?```", re.S)
_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_LINK = re.compile(r"\[([^\]]+)\]\((?:https?://|mailto:)[^)]*\)")
_BARE_URL = re.compile(r"https?://\S+|www\.\S+")
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]{0,80}>")
_STAGE_DIRECTION = re.compile(r"\[[^\]\n]{0,80}\]|\((?:pause|laughs?|smiles?|sighs?|thinking|beat)[^)\n]{0,30}\)", re.I)
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*•●▪]|\d{1,2}[.)])\s+", re.M)
_BLOCKQUOTE = re.compile(r"^\s*>+\s?", re.M)
_EMPHASIS = re.compile(r"(\*\*|__|\*|~~)")
_INTRA_WORD_UNDERSCORE = re.compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?।])")
_REPEATED_PUNCT = re.compile(r"([,;:])\1+|([.!?])\2{3,}")
_ZERO_WIDTH = re.compile(r"[​‌‍⁠﻿]")

_JSON_TEXT_KEYS = ("candidateFacingText", "candidate_facing_text", "spoken_text", "text", "question", "message")

# Symbol/abbreviation expansions that TTS engines commonly get wrong. Deliberately small and
# conservative: technical terms (HashMap, O(1), C++) must survive intact.
_ABBREVIATIONS = (
    (re.compile(r"\be\.g\.,?", re.I), "for example,"),
    (re.compile(r"\bi\.e\.,?", re.I), "that is,"),
    (re.compile(r"\betc\.", re.I), "and so on."),
    (re.compile(r"\bvs\.?(?=\s)", re.I), "versus"),
    (re.compile(r"\bw/o\b", re.I), "without"),
    (re.compile(r"\bw/(?=\s)", re.I), "with"),
    (re.compile(r"\s&\s"), " and "),
    (re.compile(r"(?<=\d)\s?%"), " percent"),
    (re.compile(r"\bO\(\s*1\s*\)"), "O of one"),
    (re.compile(r"\bO\(\s*n\s*\^\s*2\s*\)|\bO\(\s*n²\s*\)"), "O of n squared"),
    (re.compile(r"\bO\(\s*log\s*n\s*\)", re.I), "O of log n"),
    (re.compile(r"\bO\(\s*n\s*\)"), "O of n"),
)

_DASHES = {"‐": "-", "‑": "-", "‒": "-", "−": "-"}
_QUOTES = {"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"'}


def _is_emoji_or_symbol(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat in ("So", "Cs", "Co") or 0x1F000 <= ord(ch) <= 0x1FAFF


def _extract_json_text(text: str) -> str | None:
    """If the whole string is a JSON object/array, return the human-facing text inside it (or
    empty string if there is none). None means: not JSON."""
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if isinstance(data, dict):
        for key in _JSON_TEXT_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""


def _drop_json_blobs(text: str) -> str:
    """Remove embedded {...} blobs that look like JSON (contain a quoted key)."""
    return re.sub(r"\{[^{}]*\"[^{}]*\"\s*:[^{}]*\}", " ", text)


def prepare_for_speech(text: str | None) -> str:
    """Return text that is safe and pleasant to hand to a TTS engine ('' if nothing speakable)."""
    if not text:
        return ""

    extracted = _extract_json_text(text)
    if extracted is not None:
        text = extracted
    text = _drop_json_blobs(text)

    text = _CODE_FENCE.sub(" ", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = _STAGE_DIRECTION.sub(" ", text)
    text = _HEADING.sub("", text)
    text = _BLOCKQUOTE.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    text = _INTRA_WORD_UNDERSCORE.sub(" ", text)
    text = _ZERO_WIDTH.sub("", text)

    for src, dst in _DASHES.items():
        text = text.replace(src, dst)
    for src, dst in _QUOTES.items():
        text = text.replace(src, dst)
    text = text.replace("…", "...")
    # En/em dashes are a spoken pause, not a symbol to pronounce.
    text = re.sub(r"\s*[–—]\s*", ", ", text)

    text = "".join(ch for ch in text if not _is_emoji_or_symbol(ch) and (ch in "\n\t" or unicodedata.category(ch)[0] != "C"))

    for pattern, replacement in _ABBREVIATIONS:
        text = pattern.sub(replacement, text)

    text = re.sub(r"\s*\n+\s*", " ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _REPEATED_PUNCT.sub(lambda m: m.group(1) or "...", text)
    text = text.strip(" ,;:-")
    if not text or not any(ch.isalnum() for ch in text):
        return ""
    if text[-1] not in _SENTENCE_END + "\"'":
        text += "."
    return text


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+(?=[\"'(\[]?[A-Z0-9ऀ-ॿ])")


def split_sentences(text: str, min_chars: int = 24) -> list[str]:
    """Split prepared text into speakable sentences. Very short fragments are merged into their
    neighbour so TTS gets natural prosody instead of a burst of one-word requests."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    return merged


_CLAUSE_BREAK = re.compile(r"(?<=[,;:])\s+|\s+(?=(?:and|but|so|because|which|while|then|or)\s)", re.I)


def split_for_tts(text: str, *, min_chars: int = 16, max_chars: int = 110) -> list[str]:
    """Split prepared text into similarly sized speech units (roughly 25-110 characters).

    Sentence boundaries come first; a sentence longer than `max_chars` is broken at clause
    boundaries (commas, 'and', 'but', 'because', ...). Small units keep time-to-first-audio low
    (the first unit is synthesised on its own) and mean every later unit finishes synthesising well
    before the previous one has finished playing. Fragments shorter than `min_chars` are merged
    into a neighbour so TTS never gets a one-word request.
    """
    units: list[str] = []
    for sentence in split_sentences(text, min_chars=min_chars):
        if len(sentence) <= max_chars:
            units.append(sentence)
            continue
        pieces = [p.strip() for p in _CLAUSE_BREAK.split(sentence) if p and p.strip()]
        current = ""
        for piece in pieces:
            if current and len(current) + 1 + len(piece) > max_chars:
                units.append(current)
                current = piece
            else:
                current = f"{current} {piece}".strip()
        if current:
            units.append(current)
    merged: list[str] = []
    for unit in units:
        if merged and len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {unit}"
        else:
            merged.append(unit)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        tail = merged.pop()
        merged[-1] = f"{merged[-1]} {tail}"
    return merged
