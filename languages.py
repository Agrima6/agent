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
DEFAULT_LANGUAGE = "en"

PLAIN_LANGUAGE_INSTRUCTIONS = (
    "Language level: this candidate is interviewing for a hands-on, operational role and may not be comfortable "
    "with corporate or technical language. Use short sentences and simple, everyday words. Never use words like "
    "leverage, synergy, stakeholder, optimize, framework, prioritize. Ask about concrete, real situations from "
    "their actual daily work."
)


def normalize_language(language: str | None) -> str:
    return language if language in LANGUAGE_CONFIGS else DEFAULT_LANGUAGE
