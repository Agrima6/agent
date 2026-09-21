"""Pre-approved interviewer lines that never involve an LLM.

Refusals, redirects, "didn't catch that", and the closing/greeting fallbacks live here as fixed
text so they cannot be argued with, injected into, or drift in tone. Each intent has several
variants; RefusalPicker rotates through them so the same request twice in a row is not answered
with identical wording. Hindi/Hinglish text is phrased impersonally (no gendered first-person
verbs) so it reads correctly with either a male or a female voice.
"""
from interaction_guard import Intent

_EN = {
    Intent.REQUEST_ANSWER: [
        "I can't provide the answer during the interview. Please answer based on your understanding.",
        "I'm not able to share answers during the interview. Please give it your best attempt.",
        "Answers aren't something I can share here. Please go ahead based on what you know.",
    ],
    Intent.REQUEST_HINT: [
        "I can't provide hints during the interview. Please continue with your answer.",
        "I'm not able to give hints in the interview. Take your time and share your own approach.",
        "Hints aren't something I can offer here. Please continue with your answer as best you can.",
    ],
    Intent.REQUEST_EVALUATION: [
        "I can't discuss the evaluation during the interview. Please continue with your answer.",
        "I'm not able to comment on how the interview is going. Please continue with your answer.",
        "Feedback and scores aren't something I can share during the interview. Please go ahead with your answer.",
    ],
    Intent.REQUEST_PREVIOUS: [
        "I can't discuss previous questions during the interview. Please focus on the current question.",
        "We can't go back to earlier questions in this interview. Please focus on the current question.",
        "Earlier questions aren't something I can revisit. Let's focus on the current question.",
    ],
    Intent.REQUEST_UPCOMING: [
        "I can't reveal upcoming questions. Please focus on the current question.",
        "I'm not able to share what comes next. Please focus on the current question.",
        "Upcoming questions stay confidential. Let's focus on the current one.",
    ],
    Intent.REQUEST_EXPLANATION: [
        "I can't explain concepts during the interview. Please answer based on your understanding.",
        "I'm not able to teach or explain topics here. Share what you know about it in your own words.",
        "Explanations aren't something I can give during the interview. Please go ahead based on your understanding.",
    ],
    Intent.CHANGE_TOPIC: [
        "Let's stay focused on the current interview question.",
        "I'd like to keep us on the current question. Please continue with your answer.",
        "Let's keep to the interview. Please continue with your answer.",
    ],
    Intent.UNRELATED: [
        "Let's stay focused on the current interview question. Please continue with your answer.",
        "That's outside what we can cover in this interview. Please continue with your answer.",
        "I'll keep us on the interview. Please continue with your answer to the current question.",
    ],
    Intent.PROMPT_INJECTION: [
        "Let's stay focused on the interview. Please continue with your answer.",
        "I'll keep to the interview format. Please continue with your answer.",
        "That's not something I can act on. Please continue with your answer.",
    ],
    Intent.EMPTY: [
        "I didn't catch that. Please go ahead with your answer whenever you're ready.",
        "Sorry, I didn't hear you clearly. Could you say that again?",
    ],
}

_HI = {
    Intent.REQUEST_ANSWER: [
        "इंटरव्यू के दौरान उत्तर बताना संभव नहीं है। कृपया अपनी समझ के अनुसार जवाब दीजिए।",
        "इंटरव्यू में सही उत्तर साझा नहीं किया जा सकता। कृपया अपनी पूरी कोशिश से जवाब दीजिए।",
        "यहाँ उत्तर बताना संभव नहीं है। जितना आपको पता है, उसी के आधार पर जवाब दीजिए।",
    ],
    Intent.REQUEST_HINT: [
        "इंटरव्यू के दौरान हिंट देना संभव नहीं है। कृपया अपना जवाब जारी रखिए।",
        "यहाँ हिंट नहीं दिए जा सकते। आराम से अपने तरीके से जवाब दीजिए।",
        "इंटरव्यू में संकेत देना संभव नहीं है। कृपया अपनी समझ के अनुसार जवाब जारी रखिए।",
    ],
    Intent.REQUEST_EVALUATION: [
        "इंटरव्यू के दौरान मूल्यांकन पर चर्चा करना संभव नहीं है। कृपया अपना जवाब जारी रखिए।",
        "स्कोर या फ़ीडबैक इंटरव्यू के दौरान साझा नहीं किया जा सकता। कृपया अपना जवाब जारी रखिए।",
        "इंटरव्यू कैसा चल रहा है, इस पर यहाँ टिप्पणी नहीं की जा सकती। कृपया जवाब जारी रखिए।",
    ],
    Intent.REQUEST_PREVIOUS: [
        "इंटरव्यू के दौरान पिछले सवालों पर चर्चा करना संभव नहीं है। कृपया मौजूदा सवाल पर ध्यान दीजिए।",
        "पिछले सवालों पर वापस जाना संभव नहीं है। कृपया मौजूदा सवाल पर ध्यान दीजिए।",
        "पुराने सवालों पर यहाँ बात नहीं हो सकती। आइए मौजूदा सवाल पर ध्यान दें।",
    ],
    Intent.REQUEST_UPCOMING: [
        "आने वाले सवाल बताना संभव नहीं है। कृपया मौजूदा सवाल पर ध्यान दीजिए।",
        "आगे क्या पूछा जाएगा, यह पहले नहीं बताया जा सकता। कृपया मौजूदा सवाल पर ध्यान दीजिए।",
        "अगले सवाल गोपनीय रहते हैं। आइए मौजूदा सवाल पर ध्यान दें।",
    ],
    Intent.REQUEST_EXPLANATION: [
        "इंटरव्यू के दौरान अवधारणाएँ समझाना संभव नहीं है। कृपया अपनी समझ के अनुसार जवाब दीजिए।",
        "यहाँ किसी विषय को समझाया नहीं जा सकता। आप जो जानते हैं, अपने शब्दों में बताइए।",
        "इंटरव्यू में स्पष्टीकरण देना संभव नहीं है। कृपया अपनी समझ के आधार पर जवाब दीजिए।",
    ],
    Intent.CHANGE_TOPIC: [
        "आइए मौजूदा इंटरव्यू सवाल पर ही ध्यान दें।",
        "कृपया मौजूदा सवाल पर ही बने रहिए और अपना जवाब जारी रखिए।",
        "आइए इंटरव्यू पर ही केंद्रित रहें। कृपया अपना जवाब जारी रखिए।",
    ],
    Intent.UNRELATED: [
        "आइए मौजूदा इंटरव्यू सवाल पर ही ध्यान दें। कृपया अपना जवाब जारी रखिए।",
        "यह इस इंटरव्यू के दायरे से बाहर है। कृपया अपना जवाब जारी रखिए।",
        "आइए इंटरव्यू पर ही बने रहें। कृपया मौजूदा सवाल का जवाब जारी रखिए।",
    ],
    Intent.PROMPT_INJECTION: [
        "आइए इंटरव्यू पर ही ध्यान दें। कृपया अपना जवाब जारी रखिए।",
        "इंटरव्यू के नियमों के अनुसार ही आगे बढ़ा जाएगा। कृपया अपना जवाब जारी रखिए।",
        "यह मेरे लिए संभव नहीं है। कृपया अपना जवाब जारी रखिए।",
    ],
    Intent.EMPTY: [
        "आपकी आवाज़ साफ़ सुनाई नहीं दी। कृपया तैयार होने पर अपना जवाब बताइए।",
        "माफ़ कीजिए, ठीक से सुनाई नहीं दिया। क्या आप दोबारा बता सकते हैं?",
    ],
}

_HINGLISH = {
    Intent.REQUEST_ANSWER: [
        "Interview के दौरान answer बताना possible नहीं है। कृपया अपनी understanding के अनुसार जवाब दीजिए।",
        "Interview में सही answer share नहीं किया जा सकता। कृपया अपनी best कोशिश कीजिए।",
        "यहाँ answer बताना possible नहीं है। जितना आपको पता है, उसी के based पर जवाब दीजिए।",
    ],
    Intent.REQUEST_HINT: [
        "Interview के दौरान hint देना possible नहीं है। कृपया अपना answer जारी रखिए।",
        "यहाँ hints नहीं दिए जा सकते। आराम से अपने तरीके से जवाब दीजिए।",
        "Interview में clue देना possible नहीं है। कृपया अपनी understanding से जवाब जारी रखिए।",
    ],
    Intent.REQUEST_EVALUATION: [
        "Interview के दौरान evaluation पर बात करना possible नहीं है। कृपया अपना answer जारी रखिए।",
        "Score या feedback interview के दौरान share नहीं किया जा सकता। कृपया answer जारी रखिए।",
        "Interview कैसा चल रहा है, इस पर यहाँ comment नहीं किया जा सकता। कृपया जवाब जारी रखिए।",
    ],
    Intent.REQUEST_PREVIOUS: [
        "Interview के दौरान पिछले questions पर बात करना possible नहीं है। कृपया current question पर focus कीजिए।",
        "पिछले questions पर वापस जाना possible नहीं है। कृपया current question पर focus कीजिए।",
        "पुराने questions पर यहाँ बात नहीं हो सकती। आइए current question पर focus करें।",
    ],
    Intent.REQUEST_UPCOMING: [
        "आने वाले questions बताना possible नहीं है। कृपया current question पर focus कीजिए।",
        "आगे क्या पूछा जाएगा, यह पहले नहीं बताया जा सकता। कृपया current question पर focus कीजिए।",
        "अगले questions confidential रहते हैं। आइए current question पर focus करें।",
    ],
    Intent.REQUEST_EXPLANATION: [
        "Interview के दौरान concepts समझाना possible नहीं है। कृपया अपनी understanding के अनुसार जवाब दीजिए।",
        "यहाँ किसी topic को explain नहीं किया जा सकता। आप जो जानते हैं, अपने शब्दों में बताइए।",
        "Interview में explanation देना possible नहीं है। कृपया अपनी understanding के आधार पर जवाब दीजिए।",
    ],
    Intent.CHANGE_TOPIC: [
        "आइए current interview question पर ही focus करें।",
        "कृपया current question पर ही रहिए और अपना answer जारी रखिए।",
        "आइए interview पर ही focused रहें। कृपया अपना answer जारी रखिए।",
    ],
    Intent.UNRELATED: [
        "आइए current interview question पर ही focus करें। कृपया अपना answer जारी रखिए।",
        "यह इस interview के scope से बाहर है। कृपया अपना answer जारी रखिए।",
        "आइए interview पर ही बने रहें। कृपया current question का जवाब जारी रखिए।",
    ],
    Intent.PROMPT_INJECTION: [
        "आइए interview पर ही focus करें। कृपया अपना answer जारी रखिए।",
        "Interview के rules के अनुसार ही आगे बढ़ा जाएगा। कृपया अपना answer जारी रखिए।",
        "यह मेरे लिए possible नहीं है। कृपया अपना answer जारी रखिए।",
    ],
    Intent.EMPTY: [
        "आपकी आवाज़ clear सुनाई नहीं दी। कृपया ready होने पर अपना answer बताइए।",
        "Sorry, ठीक से सुनाई नहीं दिया। क्या आप दोबारा बता सकते हैं?",
    ],
}

REFUSALS = {"en": _EN, "hi": _HI, "hinglish": _HINGLISH}

# Used when the composer LLM is unavailable or its output fails validation. Deliberately plain.
CLOSING = {
    "en": "Thank you for taking the time to complete your interview. The hiring team will review it and notify you by email about the result. You can close this window now.",
    "hi": "इंटरव्यू पूरा करने के लिए आपका धन्यवाद। हायरिंग टीम इसकी समीक्षा करेगी और परिणाम की जानकारी आपको ईमेल से देगी। अब आप यह विंडो बंद कर सकते हैं।",
    "hinglish": "Interview पूरा करने के लिए आपका धन्यवाद। Hiring team इसका review करेगी और result की जानकारी आपको email से देगी। अब आप यह window बंद कर सकते हैं।",
}
EARLY_END = {
    "en": "Of course, we'll end the interview here. Thank you for your time. The hiring team will review what we covered and notify you by email. You can close this window now.",
    "hi": "ठीक है, हम इंटरव्यू यहीं समाप्त करते हैं। आपके समय के लिए धन्यवाद। हायरिंग टीम आपके इंटरव्यू की समीक्षा करेगी और आपको ईमेल से जानकारी देगी। अब आप यह विंडो बंद कर सकते हैं।",
    "hinglish": "ठीक है, हम interview यहीं end करते हैं। आपके समय के लिए धन्यवाद। Hiring team review करेगी और आपको email से बता देगी। अब आप यह window बंद कर सकते हैं।",
}
GREETING = {
    "en": "Hello{name_part}, welcome to your interview. My name is {interviewer}, and I'm your AI interviewer from WorkmateIQ{role_part}. This will be a two-way conversation: I'll ask you questions, listen to your answers, and follow up on what you say. How are you doing today?",
    "hi": "नमस्ते{name_part}, आपके इंटरव्यू में स्वागत है। मेरा नाम {interviewer} है, और मैं WorkmateIQ का AI इंटरव्यूअर हूँ{role_part}। यह दोनों तरफ़ की बातचीत होगी: मैं सवाल पूछूँगा, आपके जवाब सुनूँगा और उन्हीं पर आगे बात करूँगा। आप कैसे हैं?",
    "hinglish": "नमस्ते{name_part}, आपके interview में welcome। मेरा नाम {interviewer} है, और मैं WorkmateIQ का AI interviewer हूँ{role_part}। यह two-way बातचीत होगी: मैं questions पूछूँगा, आपके answers सुनूँगा और उन्हीं पर आगे बात करूँगा। आप कैसे हैं?",
}
INTERVIEWER_NAMES = {"male": "Aarav", "female": "Aanya"}
_ROLE_PART = {"en": " for the {role} position", "hi": " ({role} रोल के लिए)", "hinglish": " ({role} role के लिए)"}
SMALL_TALK_FOLLOWUP = {
    "en": "Good to hear. Are you in a quiet spot and ready to begin? If you have any questions about how this interview works, feel free to ask now.",
    "hi": "अच्छा लगा सुनकर। क्या आप किसी शांत जगह पर हैं और शुरू करने के लिए तैयार हैं? इंटरव्यू कैसे होगा, इस बारे में कोई सवाल हो तो अभी पूछ सकते हैं।",
    "hinglish": "अच्छा लगा सुनकर। क्या आप किसी quiet जगह पर हैं और start करने के लिए ready हैं? Interview कैसे होगा, इस बारे में कोई question हो तो अभी पूछ सकते हैं।",
}
PROCESS_ANSWERS = {
    "duration": {
        "en": "The interview takes about {minutes} minutes.",
        "hi": "इंटरव्यू में लगभग {minutes} मिनट लगेंगे।",
        "hinglish": "Interview में लगभग {minutes} minutes लगेंगे।",
    },
    "format": {
        "en": "It's a spoken conversation. I ask a question, you answer out loud, and I may ask a follow-up based on what you said. You can ask me to repeat any question.",
        "hi": "यह बोलकर होने वाली बातचीत है। मैं सवाल पूछता हूँ, आप बोलकर जवाब देते हैं, और आपके जवाब के आधार पर मैं आगे पूछ सकता हूँ। आप किसी भी सवाल को दोबारा पूछने के लिए कह सकते हैं।",
        "hinglish": "यह spoken conversation है। मैं question पूछता हूँ, आप बोलकर answer देते हैं, और आपके answer के आधार पर मैं follow-up पूछ सकता हूँ। आप कोई भी question repeat करने के लिए कह सकते हैं।",
    },
}
READY_TO_BEGIN = {
    "en": "Shall we begin?",
    "hi": "क्या हम शुरू करें?",
    "hinglish": "क्या हम start करें?",
}
WELCOME_BACK = {
    "en": "Welcome back. Let's pick up where we left off.",
    "hi": "वापस स्वागत है। आइए वहीं से आगे बढ़ते हैं जहाँ हम रुके थे।",
    "hinglish": "Welcome back। आइए वहीं से आगे बढ़ते हैं जहाँ हम रुके थे।",
}
# Spoken when the same question has drawn several non-answers in a row; the engine then moves on.
MOVE_ON = {
    "en": "Let's try another question.",
    "hi": "आइए एक दूसरा सवाल देखते हैं।",
    "hinglish": "आइए एक दूसरा question देखते हैं।",
}


def _lang(language: str | None) -> str:
    return language if language in REFUSALS else "en"


class RefusalPicker:
    """Rotates through an intent's variants so consecutive identical requests get varied wording."""

    def __init__(self) -> None:
        self._counters: dict[Intent, int] = {}

    def pick(self, intent: Intent, language: str | None = "en") -> str:
        variants = REFUSALS[_lang(language)].get(intent) or REFUSALS[_lang(language)][Intent.UNRELATED]
        i = self._counters.get(intent, 0)
        self._counters[intent] = i + 1
        return variants[i % len(variants)]


def _clean(value: str, limit: int) -> str:
    """Names/roles come from user-supplied data and are spoken aloud: drop markup, keep the words."""
    import re
    value = re.sub(r"<[^>]*>", "", value or "")
    return re.sub(r"[<>{}\[\]`*_#]", "", re.sub(r"\s+", " ", value)).strip()[:limit]


def greeting(language: str | None, name: str = "", role: str = "", interviewer: str = "Aarav") -> str:
    lang = _lang(language)
    name = _clean(name, 60)
    role = _clean(role, 60)
    return GREETING[lang].format(
        interviewer=_clean(interviewer, 30) or "Aarav",
        name_part=f" {name}" if name else "",
        role_part=_ROLE_PART[lang].format(role=role) if role else "",
    )


def closing(language: str | None, early: bool = False) -> str:
    return (EARLY_END if early else CLOSING)[_lang(language)]


def fixed_phrases(language: str | None, *, essential_only: bool = False) -> list[str]:
    """Every interviewer line that is fixed text (no name, no LLM): what the TTS layer may cache.

    essential_only=True is the small set worth synthesising ahead of time (the primary wording of each
    refusal plus the common redirects); the other rotation variants are cached the first time they play.
    """
    lang = _lang(language)
    refusals = REFUSALS[lang]
    if essential_only:
        lines = [variants[0] for variants in refusals.values() if variants]
    else:
        lines = [text for variants in refusals.values() for text in variants]
    lines += [MOVE_ON[lang], READY_TO_BEGIN[lang], CLOSING[lang], EARLY_END[lang]]
    if not essential_only:
        lines += [SMALL_TALK_FOLLOWUP[lang], WELCOME_BACK[lang]]
    return list(dict.fromkeys(lines))
