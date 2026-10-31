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
# Spoken when the candidate asks to change language: the interviewer switches, then asks the current question again.
LANGUAGE_SWITCHED = {
    "en": "Sure, let's continue in English.",
    "hi": "ज़रूर, अब हम हिंदी में बात करेंगे।",
    "hinglish": "Sure, अब हम Hinglish में बात करेंगे।",
}
# Ending is not undone, and a speech recogniser mishears: ask once before closing the interview on a spoken request.
END_CONFIRM = {
    "en": "Just to be sure: would you like to end the interview now? Say yes to end it, or no to carry on.",
    "hi": "बस पक्का करने के लिए: क्या आप इंटरव्यू अभी समाप्त करना चाहते हैं? समाप्त करने के लिए हाँ कहिए, या जारी रखने के लिए ना कहिए।",
    "hinglish": "बस sure होने के लिए: क्या आप interview अभी end करना चाहते हैं? End करने के लिए yes कहिए, या जारी रखने के लिए no कहिए।",
}
END_CANCELLED = {
    "en": "No problem, let's carry on.", "hi": "कोई बात नहीं, चलिए जारी रखते हैं।", "hinglish": "कोई बात नहीं, चलिए carry on करते हैं।",
}
# The candidate asked for the language the interview is already in ("can you speak English?"): say so, never just re-read the question.
LANGUAGE_ALREADY = {
    "en": "Yes, we're already speaking in English. If you'd like Hindi or Hinglish instead, just tell me and I'll switch.",
    "hi": "जी, हम अभी हिंदी में ही बात कर रहे हैं। अगर आप अंग्रेज़ी या हिंग्लिश चाहें तो बता दीजिए, मैं बदल दूँगा।",
    "hinglish": "जी, हम अभी Hinglish में ही बात कर रहे हैं। अगर आप English या Hindi चाहें तो बता दीजिए, मैं switch कर दूँगा।",
}
# Spoken when the candidate asks for, or is transcribed in, a language this interview does not support. Fixed text:
# nothing the candidate said is echoed, and the interview stays in the language it was in.
UNSUPPORTED_LANGUAGE = {
    "en": "I can only conduct this interview in English, Hindi or Hinglish. Please continue in one of those, and I'll repeat the question.",
    "hi": "मैं यह इंटरव्यू केवल अंग्रेज़ी, हिंदी या हिंग्लिश में ले सकता हूँ। कृपया इनमें से किसी एक में जवाब दीजिए, मैं सवाल दोबारा दोहराता हूँ।",
    "hinglish": "मैं ये interview सिर्फ़ English, Hindi या Hinglish में ले सकता हूँ। please इनमें से किसी एक में बोलिए, मैं question repeat करता हूँ।",
}
# Spoken when the transcript is in a script no supported language uses (the recogniser mis-heard, or the candidate
# spoke an unsupported language): ask them to say it again rather than evaluating or quoting the garbled text.
DIDNT_CATCH_LANGUAGE = {
    "en": "Sorry, I couldn't make that out clearly. Could you say it again, in English, Hindi or Hinglish?",
    "hi": "माफ़ कीजिए, मैं इसे साफ़ नहीं समझ पाया। क्या आप इसे अंग्रेज़ी, हिंदी या हिंग्लिश में फिर से कह सकते हैं?",
    "hinglish": "Sorry, मैं ये clearly नहीं समझ पाया। क्या आप इसे English, Hindi या Hinglish में फिर से कह सकते हैं?",
}
# Replies to "Are you there?" / "Can you hear me?" and to "give me a moment" / "my internet is slow".
ACK_PRESENCE = {
    "en": "Yes, I can hear you clearly. Please go ahead whenever you're ready.",
    "hi": "जी हाँ, मुझे आपकी आवाज़ साफ़ सुनाई दे रही है। आप जब चाहें शुरू कर सकते हैं।",
    "hinglish": "Yes, मुझे आपकी आवाज़ clearly सुनाई दे रही है। आप जब ready हों, शुरू कीजिए।",
}
ACK_NEED_TIME = {
    "en": "Of course, take your time. I'll be here when you're ready.",
    "hi": "ज़रूर, आराम से सोचिए। जब आप तैयार हों, बता दीजिएगा।",
    "hinglish": "Of course, take your time, आराम से सोचिए। जब आप ready हों, बता दीजिएगा।",
}
PLEASE_CONTINUE = {
    "en": "Please continue with your answer.",
    "hi": "कृपया अपना जवाब जारी रखिए।",
    "hinglish": "Please अपना answer जारी रखिए।",
}
# After a quiet stretch (the candidate has said nothing): a gentle nudge, never a skip or a penalty.
IDLE_NUDGE = {
    "en": "Take your time. I'm here whenever you're ready, and I can repeat the question if that helps.",
    "hi": "आराम से सोचिए। जब आप तैयार हों, मैं यहीं हूँ। चाहें तो मैं सवाल दोबारा दोहरा सकता हूँ।",
    "hinglish": "Take your time, आराम से सोचिए। जब आप ready हों, मैं यहीं हूँ। चाहें तो मैं question repeat कर सकता हूँ।",
}
# After repeated unintelligible audio: it is probably the microphone, not the answer.
AUDIO_TROUBLE = {
    "en": "I'm having trouble hearing you clearly. Could you check your microphone and speak a little closer to it?",
    "hi": "मुझे आपकी आवाज़ साफ़ नहीं आ रही है। क्या आप अपना माइक्रोफ़ोन देख लेंगे और थोड़ा पास आकर बोलेंगे?",
    "hinglish": "मुझे आपकी आवाज़ clearly नहीं आ रही है। क्या आप अपना microphone check करके थोड़ा पास आकर बोलेंगे?",
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


# ------------------------------------------------------------------------------------------------------------
# The structured opening (see interview_flow.py): introduction -> clarification -> readiness -> resume questions ->
# environment check -> main interview. All fixed text: nothing here is generated, so nothing can be invented.
# Hindi/Hinglish keep the impersonal phrasing used above so they read correctly with either voice.
# ------------------------------------------------------------------------------------------------------------
_INTRO_HEAD = {
    "en": "Hi{name_part}, welcome to your interview. My name is {interviewer}, and I'll be conducting your interview{role_part} today.",
    "hi": "नमस्ते{name_part}, आपके इंटरव्यू में स्वागत है। मेरा नाम {interviewer} है, और आज मैं आपका इंटरव्यू{role_part} लूँगा।",
    "hinglish": "Hi{name_part}, आपके interview में welcome। मेरा नाम {interviewer} है, और आज मैं आपका interview{role_part} लूँगा।",
}
_INTRO_ROLE = {"en": " for the {role} position", "hi": " ({role} रोल के लिए)", "hinglish": " ({role} role के लिए)"}
_INTRO_CV_NOTICED = {
    "en": " I've reviewed your CV and was interested to see {noticed}.",
    "hi": " मैंने आपका CV देखा है और उसमें {noticed} देखकर मुझे अच्छा लगा।",
    "hinglish": " मैंने आपका CV देखा है और उसमें {noticed} देखकर अच्छा लगा।",
}
_INTRO_CV_ONLY = {
    "en": " I've reviewed your CV.", "hi": " मैंने आपका CV देखा है।", "hinglish": " मैंने आपका CV देखा है।",
}
_NOTICED_PROJECT = {"en": "your project {value}", "hi": "आपका प्रोजेक्ट {value}", "hinglish": "आपका project {value}"}
_NOTICED_SKILL = {"en": "your experience with {value}", "hi": "{value} में आपका अनुभव", "hinglish": "{value} में आपका experience"}
_NOTICED_COMPANY = {"en": "your work with {value}", "hi": "{value} में आपका काम", "hinglish": "{value} में आपका work"}
_AND = {"en": " and ", "hi": " और ", "hinglish": " और "}
_INTRO_EXPERIENCE = {
    "en": " I understand you have around {value} of experience.",
    "hi": " मैं समझता हूँ कि आपके पास लगभग {value} का अनुभव है।",
    "hinglish": " मैं समझता हूँ कि आपके पास around {value} का experience है।",
}
_INTRO_PLAN_WITH_CV = {
    "en": " I'll ask a couple of questions about your background first, then we'll continue with the interview.",
    "hi": " मैं पहले आपके बैकग्राउंड के बारे में एक-दो सवाल पूछूँगा, फिर हम इंटरव्यू आगे बढ़ाएँगे।",
    "hinglish": " मैं पहले आपके background के बारे में एक-दो questions पूछूँगा, फिर हम interview आगे बढ़ाएँगे।",
}
_INTRO_PLAN_NO_CV = {
    "en": " I'll start with a question about your background, then we'll continue with the interview.",
    "hi": " मैं आपके बैकग्राउंड के बारे में एक सवाल से शुरू करूँगा, फिर हम इंटरव्यू आगे बढ़ाएँगे।",
    "hinglish": " मैं आपके background के बारे में एक question से शुरू करूँगा, फिर हम interview आगे बढ़ाएँगे।",
}
CLARIFY_ASK = {
    "en": " Before we begin, is there anything about the role, the interview format, or the instructions you'd like me to clarify?",
    "hi": " शुरू करने से पहले, क्या रोल, इंटरव्यू के फ़ॉर्मेट या निर्देशों के बारे में कुछ स्पष्ट करना है?",
    "hinglish": " शुरू करने से पहले, क्या role, interview format या instructions के बारे में कुछ clarify करना है?",
}
CLARIFY_MORE = {
    "en": "Does that help? Do you have any other questions or doubts?",
    "hi": "क्या इससे बात साफ़ हुई? क्या कोई और सवाल या संदेह है?",
    "hinglish": "क्या इससे बात clear हुई? क्या कोई और question या doubt है?",
}
CLARIFY_GO_AHEAD = {
    "en": "Of course, please go ahead and ask.", "hi": "ज़रूर, कृपया पूछिए।", "hinglish": "Of course, please पूछिए।",
}
CLARIFY_RETRY = {
    "en": "Sorry, I didn't quite catch that. Do you have any questions about the role or the interview? You can simply say, no questions.",
    "hi": "माफ़ कीजिए, मैं ठीक से समझ नहीं पाया। क्या रोल या इंटरव्यू के बारे में कोई सवाल है? चाहें तो बस कह दीजिए, कोई सवाल नहीं।",
    "hinglish": "Sorry, मैं ठीक से समझ नहीं पाया। क्या role या interview के बारे में कोई question है? चाहें तो बस कह दीजिए, कोई question नहीं।",
}
CLARIFY_UNKNOWN = {
    "en": "I'm not able to confirm that detail from the interview instructions I have, so I don't want to guess. The hiring team can help with it after the interview.",
    "hi": "मेरे पास मौजूद इंटरव्यू निर्देशों से मैं यह जानकारी पक्की नहीं कर सकता, इसलिए अंदाज़ा नहीं लगाना चाहता। इंटरव्यू के बाद हायरिंग टीम इसमें मदद कर सकती है।",
    "hinglish": "मेरे पास जो interview instructions हैं उनसे मैं यह detail confirm नहीं कर सकता, इसलिए guess नहीं करना चाहता। Interview के बाद hiring team इसमें help कर सकती है।",
}
CLARIFY_ROLE = {
    "en": "This interview is for the {role} position. I don't have more detail about the role than that, but the hiring team can share it after the interview.",
    "hi": "यह इंटरव्यू {role} पद के लिए है। रोल के बारे में इससे ज़्यादा जानकारी मेरे पास नहीं है, पर इंटरव्यू के बाद हायरिंग टीम बता सकती है।",
    "hinglish": "यह interview {role} position के लिए है। Role के बारे में इससे ज़्यादा detail मेरे पास नहीं है, पर interview के बाद hiring team बता सकती है।",
}
CLARIFY_RESULT = {
    "en": "The hiring team will review your interview and notify you by email about the result.",
    "hi": "हायरिंग टीम आपके इंटरव्यू की समीक्षा करेगी और परिणाम की जानकारी आपको ईमेल से देगी।",
    "hinglish": "Hiring team आपके interview का review करेगी और result की जानकारी आपको email से देगी।",
}
CLARIFY_LIMIT = {
    "en": "I've noted those questions, and the hiring team can follow up on anything I couldn't cover.",
    "hi": "मैंने आपके सवाल नोट कर लिए हैं, जिन बातों का जवाब मैं नहीं दे सका, उन पर हायरिंग टीम बाद में बात कर सकती है।",
    "hinglish": "मैंने आपके questions note कर लिए हैं, जिन बातों का answer मैं नहीं दे सका, उन पर hiring team बाद में बात कर सकती है।",
}
READINESS_ASK = {
    "en": "Is everything clear, and are you ready to begin the interview?",
    "hi": "क्या सब कुछ स्पष्ट है, और क्या आप इंटरव्यू शुरू करने के लिए तैयार हैं?",
    "hinglish": "क्या सब कुछ clear है, और क्या आप interview शुरू करने के लिए ready हैं?",
}
READINESS_NOT_READY = {
    "en": "No problem. Do you need a moment, or is there something else you'd like to ask before we begin?",
    "hi": "कोई बात नहीं। क्या आपको थोड़ा समय चाहिए, या शुरू करने से पहले कुछ और पूछना है?",
    "hinglish": "कोई बात नहीं। क्या आपको थोड़ा time चाहिए, या शुरू करने से पहले कुछ और पूछना है?",
}
READINESS_RETRY = {
    "en": "Sorry, I want to be sure. Are you ready to begin the interview? A simple yes or no is fine.",
    "hi": "माफ़ कीजिए, मैं पक्का करना चाहता हूँ। क्या आप इंटरव्यू शुरू करने के लिए तैयार हैं? हाँ या ना कह दीजिए।",
    "hinglish": "Sorry, मैं sure होना चाहता हूँ। क्या आप interview शुरू करने के लिए ready हैं? Yes या no कह दीजिए।",
}
READINESS_WAIT = {
    "en": "Take your time. Just let me know when you're ready to begin.",
    "hi": "आराम से तैयार हो जाइए। जब शुरू करना चाहें, बता दीजिएगा।",
    "hinglish": "Take your time। जब शुरू करना चाहें, बता दीजिएगा।",
}
RESUME_LEAD = {
    "en": "Great, let's begin.", "hi": "बढ़िया, चलिए शुरू करते हैं।", "hinglish": "Great, चलिए शुरू करते हैं।",
}
RESUME_NEXT = {
    "en": "Thank you. One more about your background.", "hi": "धन्यवाद। आपके बैकग्राउंड के बारे में एक और सवाल।",
    "hinglish": "Thank you। आपके background के बारे में एक और question।",
}
ENV_ASK = {
    "en": "Thank you, that helps me understand your background. Before we move on, a quick check: are you in a quiet place where you can speak without interruption, and can you hear me clearly?",
    "hi": "धन्यवाद, इससे मुझे आपके बैकग्राउंड को समझने में मदद मिली। आगे बढ़ने से पहले एक छोटी जाँच: क्या आप किसी शांत जगह पर हैं जहाँ बिना रुकावट बोल सकें, और क्या आपको मेरी आवाज़ साफ़ सुनाई दे रही है?",
    "hinglish": "Thank you, इससे मुझे आपका background समझने में help मिली। आगे बढ़ने से पहले एक quick check: क्या आप किसी quiet जगह पर हैं जहाँ बिना interruption बोल सकें, और क्या आपको मेरी आवाज़ clearly सुनाई दे रही है?",
}
ENV_RETRY = {
    "en": "Please move somewhere quieter if you can, and check that your microphone and speakers are working. Tell me when you're ready and I'll continue.",
    "hi": "कृपया हो सके तो किसी शांत जगह पर चले जाइए, और अपना माइक्रोफ़ोन और स्पीकर जाँच लीजिए। तैयार होने पर बता दीजिए, मैं आगे बढ़ूँगा।",
    "hinglish": "Please हो सके तो किसी quiet जगह पर चले जाइए, और अपना microphone और speakers check कर लीजिए। Ready होने पर बता दीजिए, मैं आगे बढ़ूँगा।",
}
ENV_CONTINUE = {
    "en": "Understood. We'll continue now; please do your best to keep background noise down, and tell me if you can't hear me.",
    "hi": "ठीक है। हम अब आगे बढ़ते हैं; कृपया आसपास का शोर कम रखने की कोशिश कीजिए, और अगर मेरी आवाज़ न आए तो बता दीजिए।",
    "hinglish": "Understood। हम अब आगे बढ़ते हैं; please आसपास का noise कम रखने की कोशिश कीजिए, और अगर मेरी आवाज़ न आए तो बता दीजिए।",
}
ENV_RETRY_ASK = {
    "en": "Sorry, I didn't catch that. Are you in a quiet place, and can you hear me clearly?",
    "hi": "माफ़ कीजिए, मैं समझ नहीं पाया। क्या आप शांत जगह पर हैं, और क्या मेरी आवाज़ साफ़ सुनाई दे रही है?",
    "hinglish": "Sorry, मैं समझ नहीं पाया। क्या आप quiet जगह पर हैं, और क्या मेरी आवाज़ clearly सुनाई दे रही है?",
}
MAIN_LEAD = {
    "en": "Thank you. Let's move on to the main part of the interview.",
    "hi": "धन्यवाद। अब हम इंटरव्यू के मुख्य हिस्से पर चलते हैं।",
    "hinglish": "Thank you। अब हम interview के main part पर चलते हैं।",
}


def introduction(language: str | None, name: str = "", role: str = "", interviewer: str = "Aarav",
                 facts: dict | None = None) -> str:
    """The personalised opening. Only facts that were verified against the resume (interview_flow.profile_facts) are
    mentioned; a missing fact is left out, never guessed. It ends with the clarification question."""
    lang = _lang(language)
    facts = facts or {}
    name = _clean(name, 60)
    role = _clean(role, 60)
    text = _INTRO_HEAD[lang].format(
        name_part=f" {name}" if name else "", interviewer=_clean(interviewer, 30) or "Aarav",
        role_part=_INTRO_ROLE[lang].format(role=role) if role else "")
    noticed = []
    if facts.get("project"):
        noticed.append(_NOTICED_PROJECT[lang].format(value=_clean(facts["project"], 60)))
    elif facts.get("skill"):
        noticed.append(_NOTICED_SKILL[lang].format(value=_clean(facts["skill"], 60)))
    if facts.get("company"):
        noticed.append(_NOTICED_COMPANY[lang].format(value=_clean(facts["company"], 60)))
    if noticed:
        text += _INTRO_CV_NOTICED[lang].format(noticed=_AND[lang].join(noticed))
    elif facts:
        text += _INTRO_CV_ONLY[lang]
    if facts.get("experience"):
        text += _INTRO_EXPERIENCE[lang].format(value=_clean(facts["experience"], 30))
    text += (_INTRO_PLAN_WITH_CV if facts else _INTRO_PLAN_NO_CV)[lang]
    return text + CLARIFY_ASK[lang]


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
    lines += [MOVE_ON[lang], READY_TO_BEGIN[lang], CLOSING[lang], EARLY_END[lang],
              UNSUPPORTED_LANGUAGE[lang], DIDNT_CATCH_LANGUAGE[lang], IDLE_NUDGE[lang], AUDIO_TROUBLE[lang],
              ACK_PRESENCE[lang], ACK_NEED_TIME[lang], PLEASE_CONTINUE[lang], LANGUAGE_ALREADY[lang], END_CONFIRM[lang], END_CANCELLED[lang]]
    lines += [READINESS_ASK[lang], CLARIFY_MORE[lang], ENV_ASK[lang]]
    if not essential_only:
        lines += [SMALL_TALK_FOLLOWUP[lang], WELCOME_BACK[lang], CLARIFY_GO_AHEAD[lang], CLARIFY_RETRY[lang],
                  CLARIFY_UNKNOWN[lang], CLARIFY_RESULT[lang], CLARIFY_LIMIT[lang], READINESS_NOT_READY[lang],
                  READINESS_RETRY[lang], READINESS_WAIT[lang], RESUME_LEAD[lang], RESUME_NEXT[lang],
                  ENV_RETRY[lang], ENV_CONTINUE[lang], ENV_RETRY_ASK[lang], MAIN_LEAD[lang]]
    return list(dict.fromkeys(lines))
