"""Interview language configuration shared by the agent and the speech composer."""

LANGUAGE_CONFIGS = {
    "en": {
        "label": "English",
        "stt_language": "en",
        "sarvam_language_code": "en-IN",
        "instruction": "Write in clear, professional English.",
    },
    "hi": {
        "label": "Hindi",
        "stt_language": "hi",
        "sarvam_language_code": "hi-IN",
        "instruction": (
            "Write in Hindi, in Devanagari script (देवनागरी) - never in Roman/Latin letters - because the "
            "text-to-speech engine mispronounces romanized Hindi.\n"
            "Register: plain, everyday spoken Hindi, the ordinary Hindi used in Indian offices and homes today "
            "(नमस्ते, आप, काम, फिर, ठीक है, समझ गया) - NOT heavily Sanskritized Hindi (avoid तत्पश्चात, उपरोक्त, तदनुसार, "
            "अतः) and NOT heavily Urdu-inflected Hindi (avoid गुफ़्तगू, तशरीफ़, मुलाक़ात, इर्शाद).\n"
            "Common English words may stay in Latin script where that is how people actually say them (project, "
            "team, time, manager). Technical terms may stay in Latin script. Avoid gendered first-person verb "
            "forms so the sentence works for either a male or a female voice."
        ),
    },
    "hinglish": {
        "label": "Hinglish",
        "stt_language": "hi",
        "sarvam_language_code": "hi-IN",
        "instruction": (
            "Write in natural Hinglish - the casual Hindi/English mix used in Indian workplaces (Hindi sentence "
            "structure with English technical/professional terms). Write the Hindi portions in Devanagari script "
            "(देवनागरी) and the English words in Latin script inside the same sentence, e.g. \"आपने अपने last project "
            "में कौनसा architecture use किया था?\" - never write the Hindi portion in Roman letters. Keep the Hindi "
            "side plain and everyday, and avoid gendered first-person verb forms."
        ),
    },
}

# Other Indian languages. Each is both LISTENABLE (Whisper has the language) and SPEAKABLE (Sarvam bulbul:v3 has the
# language code); Odia is left out because Whisper cannot transcribe it. Fixed interviewer lines for each live in
# i18n/<code>.py. Speaking uses the same Sarvam voices as Hindi: the voice is multilingual, the language code selects.
_INDIC = {
    # code: (English name, native name, script, writing note)
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "Gurmukhi (ਗੁਰਮੁਖੀ)", "Plain, everyday spoken Punjabi."),
    "ta": ("Tamil", "தமிழ்", "Tamil script (தமிழ்)", "Plain, everyday spoken Tamil, not literary Tamil."),
    "te": ("Telugu", "తెలుగు", "Telugu script (తెలుగు)", "Plain, everyday spoken Telugu."),
    "bn": ("Bengali", "বাংলা", "Bengali script (বাংলা)", "Plain, everyday spoken Bengali (cholito bhasha)."),
    "mr": ("Marathi", "मराठी", "Devanagari (देवनागरी)", "Plain, everyday spoken Marathi."),
    "gu": ("Gujarati", "ગુજરાતી", "Gujarati script (ગુજરાતી)", "Plain, everyday spoken Gujarati."),
    "kn": ("Kannada", "ಕನ್ನಡ", "Kannada script (ಕನ್ನಡ)", "Plain, everyday spoken Kannada."),
    "ml": ("Malayalam", "മലയാളം", "Malayalam script (മലയാളം)", "Plain, everyday spoken Malayalam."),
}
for _code, (_name, _native, _script, _note) in _INDIC.items():
    LANGUAGE_CONFIGS[_code] = {
        "label": _name,
        "native_label": _native,
        "stt_language": _code,
        "sarvam_language_code": f"{_code}-IN",
        "instruction": (
            f"Write in {_name}, in {_script} - never in Roman/Latin letters - because the text-to-speech engine "
            f"mispronounces romanized text. {_note} Common English words may stay in Latin script where that is how "
            "people actually say them (project, team, manager, time). Avoid gendered first-person verb forms where the "
            "language allows, so the sentence works for either a male or a female voice."
        ),
    }
for _code, _cfg in LANGUAGE_CONFIGS.items():
    _cfg.setdefault("native_label", _cfg["label"])

# Regional varieties people ask for by name. None has its own speech engine, so the interview is conducted in the
# standard language they are closest to (understood by the same recogniser, answered in that language).
DIALECT_ALIASES = {
    "haryanvi": "hi", "bhojpuri": "hi", "rajasthani": "hi", "marwari": "hi", "maithili": "hi", "awadhi": "hi",
    "chhattisgarhi": "hi", "magahi": "hi", "bundeli": "hi", "pahari": "hi", "garhwali": "hi", "kumaoni": "hi",
    "braj": "hi", "dogri": "hi", "malvi": "hi", "nimadi": "hi", "tulu": "kn", "konkani": "mr", "kodava": "kn",
    "sylheti": "bn",
}

DEFAULT_LANGUAGE = "en"

PLAIN_LANGUAGE_INSTRUCTIONS = (
    "Language level: this candidate is interviewing for a hands-on, operational role and may not be comfortable "
    "with corporate or technical language. Use short sentences and simple, everyday words. Never use words like "
    "leverage, synergy, stakeholder, optimize, framework, prioritize. Ask about concrete, real situations from "
    "their actual daily work."
)


def normalize_language(language: str | None) -> str:
    return language if language in LANGUAGE_CONFIGS else DEFAULT_LANGUAGE


# Which of our interview languages each voice can actually SPEAK. Kept here (pure data) so the API can publish it and the
# join screen can offer only languages that will work, and the worker can use the same rule.
ELEVENLABS_ENGLISH_ONLY_MODELS = frozenset({"eleven_monolingual_v1", "eleven_turbo_v2", "eleven_flash_v2"})
# The ElevenLabs models that stream cover only Tamil beyond English and Hindi. (Sarvam covers every language.)
ELEVENLABS_V25_LANGUAGES = frozenset({"en", "hi", "hinglish", "ta"})


def languages_for_provider(provider: str, model: str = "") -> frozenset[str]:
    """What ONE provider's voice speaks. ElevenLabs v3 is deliberately not treated as multilingual here: it rejects the
    streaming connection the live interview needs (HTTP 400), so every non-English-only ElevenLabs model gets the v2.5 set."""
    if provider == "elevenlabs":
        return frozenset({"en"}) if (model or "") in ELEVENLABS_ENGLISH_ONLY_MODELS else ELEVENLABS_V25_LANGUAGES
    return frozenset(LANGUAGE_CONFIGS)


def provider_for_language(preferred: str, language: str, model: str = "", sarvam_available: bool = False) -> str | None:
    """The voice provider to use for an interview in `language`: the preferred one when it can speak it, otherwise Sarvam
    (which speaks every language here) when it is configured, otherwise None (the caller falls back to another language)."""
    if language in languages_for_provider(preferred, model):
        return preferred
    return "sarvam" if sarvam_available else None


def offered_languages(preferred: str, model: str = "", sarvam_available: bool = False) -> frozenset[str]:
    """The languages a candidate can be offered: everything when a second voice (Sarvam) can cover what the preferred one cannot."""
    return frozenset(LANGUAGE_CONFIGS) if sarvam_available else languages_for_provider(preferred, model)


def fallback_language(language: str, supported) -> str:
    """The language to use when the one asked for cannot be spoken: the closest one that can."""
    if language in supported:
        return language
    for candidate in ("hinglish", "hi", "en"):          # a candidate who picked an Indian language most likely follows Hindi
        if candidate in supported:
            return candidate
    return DEFAULT_LANGUAGE
