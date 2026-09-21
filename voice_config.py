"""Single source of truth for the interviewer's voice.

Nothing else in the agent hard-codes a speaker, pace, language code or sample rate; everything
resolves through resolve_voice(). Resolution priority (highest first):

    1. per-interview values (Interview.voice_gender / voice_speaker / voice_pace via the API)
    2. deployment overrides (TTS_SPEAKER_OVERRIDE / TTS_PACE_OVERRIDE / TTS_GENDER env vars)
    3. the calibrated defaults below

Pace is calibrated PER SPEAKER, not globally. Bulbul v3 speakers have very different base speeds
(at pace 1.0 ratan speaks ~200 wpm while priya speaks ~138 wpm), so one shared pace value cannot
make them all sound calm - and a pace that is right for one is badly wrong for another. The
defaults below were measured against Sarvam's live API on a corpus of realistic interviewer speech
(12 English / 8 Hindi utterances, aggregate words per audio-second, synthesised through the real
plugin path) and target roughly 145-155 words per minute: a calm, clearly intelligible interviewing
rate that is neither rushed nor sluggish. Re-measure and update CALIBRATION if the model changes.

Bulbul v3 does not support pitch or loudness control, so neither is exposed here.
"""
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("voice-config")

SARVAM_MODEL = "bulbul:v3"
# Sarvam rejects anything outside this range with a 400, which would silence the interviewer.
MIN_PACE, MAX_PACE = 0.5, 2.0
ALLOWED_SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100, 48000)
DEFAULT_SAMPLE_RATE = 24000
# Pace for a speaker we have no calibration for. Conservative on purpose: base speeds vary from
# ~150 to ~200 wpm at pace 1.0, so an uncalibrated speaker is more likely too fast than too slow.
UNCALIBRATED_PACE = 0.85

# Speakers accepted by bulbul:v3 (returned by Sarvam's own validation error).
VALID_V3_SPEAKERS = frozenset({
    "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja", "rohan", "simran", "kavya",
    "amit", "dev", "ishita", "shreya", "ratan", "varun", "manan", "sumit", "roopa", "kabir",
    "aayan", "shubh", "advait", "anand", "tanya", "tarun", "sunny", "mani", "gokul", "vijay",
    "shruti", "suhani", "mohit", "kavitha", "rehan", "soham", "rupali",
})

# speaker -> (calibrated pace, measured aggregate words-per-minute at that pace).
CALIBRATION = {
    "ratan": (0.70, 149.5),    # en, male     (0.75 -> 157, 0.80 -> 168, and 0.92 would be ~190: far too fast)
    "ishita": (0.80, 150.0),   # en, female   (0.78 -> 144, 0.83 -> 155)
    "shubh": (0.74, 148.0),    # hi, male     (0.72 -> 143, 0.76 -> 153)
    "priya": (1.06, 147.7),    # hi, female   (base voice is slow: 0.94 -> 138, 1.12 -> 164)
}
TARGET_WPM = (140.0, 160.0)
CALIBRATED_PACE = {speaker: pace for speaker, (pace, _wpm) in CALIBRATION.items()}

# (interview language, gender) -> speaker. Hinglish is spoken with the Hindi voices, matching the
# hi-IN language code the previous implementation already used for it.
DEFAULT_SPEAKERS = {
    ("en", "male"): "ratan",
    ("en", "female"): "ishita",
    ("hi", "male"): "shubh",
    ("hi", "female"): "priya",
}
DEFAULT_GENDER = "female"

SPEAKER_GENDER = {"ratan": "male", "shubh": "male", "aditya": "male", "ishita": "female", "priya": "female"}

LANGUAGE_CODES = {"en": "en-IN", "hi": "hi-IN", "hinglish": "hi-IN"}

# Last-resort voices (Edge TTS) used only if Sarvam is completely unreachable.
EDGE_FALLBACK_VOICES = {
    ("en", "male"): "en-IN-PrabhatNeural",
    ("en", "female"): "en-IN-NeerjaNeural",
    ("hi", "male"): "hi-IN-MadhurNeural",
    ("hi", "female"): "hi-IN-SwaraNeural",
}

_LEGACY_ENV_VARS = ("SARVAM_PACE", "SARVAM_SPEAKER", "SARVAM_LANGUAGE_CODE", "SARVAM_SAMPLE_RATE")
_warned_legacy = False


@dataclass(frozen=True)
class VoiceConfig:
    provider: str
    model: str
    language: str
    language_code: str
    speaker: str
    gender: str
    pace: float
    sample_rate: int

    def as_dict(self) -> dict:
        return {
            "provider": self.provider, "model": self.model, "language": self.language,
            "language_code": self.language_code, "speaker": self.speaker, "gender": self.gender,
            "pace": self.pace, "sample_rate": self.sample_rate,
        }

    @property
    def edge_voice(self) -> str:
        lang = "hi" if self.language in ("hi", "hinglish") else "en"
        return EDGE_FALLBACK_VOICES[(lang, self.gender)]


def _norm_language(language: str | None) -> str:
    language = (language or "en").lower()
    return language if language in LANGUAGE_CODES else "en"


def _norm_gender(gender: str | None) -> str | None:
    if not gender:
        return None
    gender = gender.strip().lower()
    return gender if gender in ("male", "female") else None


def clamp_pace(pace: float) -> float:
    return max(MIN_PACE, min(MAX_PACE, float(pace)))


def _sanitize_sample_rate(rate: int) -> int:
    if rate in ALLOWED_SAMPLE_RATES:
        return rate
    return DEFAULT_SAMPLE_RATE


def _warn_legacy_env_once() -> None:
    global _warned_legacy
    if _warned_legacy:
        return
    _warned_legacy = True
    present = [name for name in _LEGACY_ENV_VARS if os.getenv(name)]
    if present:
        logger.warning(
            "Ignoring legacy voice env vars %s. Voice settings now come from voice_config.py "
            "(calibrated per speaker). Use TTS_SPEAKER_OVERRIDE / TTS_PACE_OVERRIDE / TTS_GENDER / "
            "TTS_SAMPLE_RATE to override, or set a per-interview voice via the API.", present,
        )


def resolve_voice(language: str | None, gender: str | None = None, speaker: str | None = None,
                  pace: float | None = None, provider: str | None = None) -> VoiceConfig:
    """Resolve the effective voice for an interview. Never raises on bad input: an unknown speaker
    or out-of-range pace degrades to a safe calibrated default (and logs) instead of silencing
    the interviewer with an API 400."""
    _warn_legacy_env_once()
    language = _norm_language(language)
    lang_group = "hi" if language in ("hi", "hinglish") else "en"

    speaker = (speaker or os.getenv("TTS_SPEAKER_OVERRIDE") or "").strip().lower() or None
    if speaker and speaker not in VALID_V3_SPEAKERS:
        logger.warning("Unknown bulbul:v3 speaker %r - using the calibrated default instead", speaker)
        speaker = None

    gender = _norm_gender(gender) or _norm_gender(os.getenv("TTS_GENDER")) or SPEAKER_GENDER.get(speaker or "")
    gender = gender or DEFAULT_GENDER
    speaker = speaker or DEFAULT_SPEAKERS[(lang_group, gender)]
    gender = SPEAKER_GENDER.get(speaker, gender)

    pace_override = pace if pace is not None else os.getenv("TTS_PACE_OVERRIDE")
    if pace_override not in (None, ""):
        try:
            resolved_pace = clamp_pace(float(pace_override))
        except (TypeError, ValueError):
            logger.warning("Invalid pace %r - using the calibrated default", pace_override)
            resolved_pace = CALIBRATED_PACE.get(speaker, UNCALIBRATED_PACE)
    else:
        resolved_pace = CALIBRATED_PACE.get(speaker, UNCALIBRATED_PACE)

    try:
        sample_rate = _sanitize_sample_rate(int(os.getenv("TTS_SAMPLE_RATE", DEFAULT_SAMPLE_RATE)))
    except ValueError:
        sample_rate = DEFAULT_SAMPLE_RATE

    return VoiceConfig(
        provider=(provider or os.getenv("TTS_PROVIDER") or "sarvam").lower(),
        model=SARVAM_MODEL,
        language=language,
        language_code=LANGUAGE_CODES[language],
        speaker=speaker,
        gender=gender,
        pace=round(resolved_pace, 3),
        sample_rate=sample_rate,
    )
